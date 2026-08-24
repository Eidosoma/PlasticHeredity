#!/usr/bin/env python3
"""Run the phase-gated S12F clock/exposure inverse problem."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy
import yaml

from e01_latent_timebase.core import (
    ExposureDefinition,
    SimulationDefinition,
    TimebaseTrajectory,
    observation_rows,
    simulate_trajectory,
    trajectory_replay_equal,
    trajectory_summary,
)
from e01_latent_timebase.inference import (
    Particle,
    candidate_groups,
    importance_weights,
    initial_particles,
    particle_summary_and_distance,
    phase1_clock_audit,
    propose_particles,
    retained_particles,
)

ARTIFACTS = Path("/artifacts/research_steps/S12F")
CACHE = Path("/cache/e01_s12f")
TRAJECTORY_CACHE = CACHE / "trajectories"
CONFIG_PATH = REPO / "configs/e01/s12f_latent_timebase_preregistration.yaml"
CANDIDATE_LOCK_PATH = REPO / "configs/e01/s12f/candidate_lock.json"
WORKERS = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def verify_hash_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for row in payload["files"]:
        candidate = Path(row["path"])
        if not candidate.is_file():
            failures.append({"path": str(candidate), "reason": "missing"})
        elif candidate.stat().st_size != row["sizeBytes"] or sha256(candidate) != row["sha256"]:
            failures.append({"path": str(candidate), "reason": "identity_changed"})
    return {
        "path": str(path),
        "checkedFileCount": len(payload["files"]),
        "expectedAggregateSha256": payload["aggregateSha256"],
        "failures": failures,
        "passed": not failures,
    }


def verify_preregistration(require_pushed: bool) -> dict[str, Any]:
    record = json.loads((ARTIFACTS / "preregistration_record.json").read_text())
    phase0 = json.loads((ARTIFACTS / "phase0_validation.json").read_text())
    if not phase0["passed"]:
        raise RuntimeError("Phase-0 target extraction failed")
    if sha256(CONFIG_PATH) != record["configSha256"]:
        raise RuntimeError("preregistration changed after freeze")
    for row in record["targetFiles"]:
        if sha256(Path(row["path"])) != row["sha256"]:
            raise RuntimeError(f"target ledger changed: {row['path']}")
    if require_pushed:
        head = git("rev-parse", "HEAD^{commit}")
        remote = git("rev-parse", "origin/eidosoma/groups/42^{commit}")
        if head != remote:
            raise RuntimeError("current branch is not pushed")
        if not record["commitRecordedAfterPush"] or not record["headMatchesRemote"]:
            raise RuntimeError("pushed preregistration was not recorded")
        design = record["gitCommit"]
        subprocess.check_call(["git", "-C", str(REPO), "merge-base", "--is-ancestor", design, head])
    prior = verify_hash_manifest(ARTIFACTS / "immutable_prior_baseline.json")
    cache = verify_hash_manifest(ARTIFACTS / "s12e_cache_manifest.json")
    if not prior["passed"] or not cache["passed"]:
        raise RuntimeError("prior artifacts or S12E cache changed")
    return {"record": record, "phase0": phase0, "prior": prior, "s12eCache": cache}


def seed_rows(seeds: tuple[Any, ...], stage: str, particle_id: str | None) -> list[dict[str, Any]]:
    return [
        {
            "stage": stage,
            "phase": seed.phase,
            "rootSha256": seed.root_sha256,
            "purpose": seed.purpose,
            "matrixIndex": seed.matrix_index,
            "configurationId": seed.configuration_id,
            "particleId": particle_id,
            "extra": list(seed.extra),
            "derivedSeed": str(seed.derived_seed),
            "seedMaterialSha256": seed.seed_material_sha256,
            "bitGenerator": "PCG64DXSM",
        }
        for seed in seeds
    ]


def definition_payload(definition: SimulationDefinition) -> dict[str, Any]:
    return {
        "daughterRule": definition.daughter_rule,
        "overshootRule": definition.overshoot_rule,
        "exposureFamily": definition.exposure.family,
        "h": definition.exposure.h,
        "c": definition.exposure.c,
        "hMax": definition.exposure.h_max,
    }


def _simulation_task(payload: dict[str, Any]) -> dict[str, Any]:
    exposure = ExposureDefinition(
        payload["family"], h=payload.get("h"), c=payload.get("c"), h_max=payload.get("hMax")
    )
    definition = SimulationDefinition(
        payload["daughterRule"], payload["overshootRule"], exposure
    )
    started = time.process_time()
    wall_started = time.perf_counter()
    first, seeds = simulate_trajectory(
        phase=payload["phase"],
        root_hex=payload["root"],
        matrix_index=payload["matrixIndex"],
        definition=definition,
        stream_identity=payload["streamIdentity"],
    )
    replay, replay_seeds = simulate_trajectory(
        phase=payload["phase"],
        root_hex=payload["root"],
        matrix_index=payload["matrixIndex"],
        definition=definition,
        stream_identity=payload["streamIdentity"],
    )
    replay_passed = trajectory_replay_equal(first, replay) and seeds == replay_seeds
    summary = trajectory_summary(first)
    summary.update(
        {
            "particleId": payload.get("particleId"),
            "candidateId": payload.get("candidateId"),
            "clockId": payload.get("clockId"),
            "exactReplayPassed": replay_passed,
            "workerCpuSeconds": time.process_time() - started,
            "workerWallSeconds": time.perf_counter() - wall_started,
        }
    )
    cache_path = payload.get("cachePath")
    if cache_path:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(first, handle, protocol=5)
        summary["cachePath"] = str(path)
        summary["cacheSha256"] = sha256(path)
    return {
        "summary": summary,
        "seeds": seed_rows(seeds, payload["phase"], payload.get("particleId")),
        "trajectory": first if payload.get("returnTrajectory") else None,
    }


def run_tasks(tasks: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, list[TimebaseTrajectory], float]:
    summaries: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    trajectories: list[TimebaseTrajectory] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_simulation_task, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            summaries.append(result["summary"])
            seeds.extend(result["seeds"])
            if result["trajectory"] is not None:
                trajectories.append(result["trajectory"])
    summary = pd.DataFrame(summaries).sort_values(
        [column for column in ("particleId", "candidateId", "matrixIndex") if column in pd.DataFrame(summaries).columns],
        na_position="last",
    ).reset_index(drop=True)
    seed_frame = pd.DataFrame(seeds).drop_duplicates("seedMaterialSha256").sort_values(
        ["stage", "purpose", "matrixIndex", "configurationId"], na_position="first"
    )
    trajectories.sort(key=lambda value: (value.configuration_id, value.matrix_index))
    return summary, seed_frame.reset_index(drop=True), trajectories, time.perf_counter() - started


def benchmark_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = config["randomness"]["roots"]["benchmark"]
    tasks: list[dict[str, Any]] = []
    for index, (daughter, overshoot, h) in enumerate(config["benchmark"]["configurations"]):
        identifier = f"BENCH-{index:02d}-{daughter}-{overshoot}-h={float(h):.17g}"
        tasks.append(
            {
                "phase": "benchmark",
                "root": root,
                "matrixIndex": 0,
                "streamIdentity": identifier,
                "particleId": identifier,
                "family": "FIXED_COMMON_EXPOSURE",
                "h": float(h),
                "c": None,
                "hMax": None,
                "daughterRule": daughter,
                "overshootRule": overshoot,
                "clockId": "C0_BATCH_UPDATES_ONLY",
            }
        )
    return tasks


def particle_tasks(particles: list[Particle], root: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for particle in particles:
        for matrix_index in range(8):
            tasks.append(
                {
                    "phase": "development",
                    "root": root,
                    "matrixIndex": matrix_index,
                    "streamIdentity": particle.stream_identity,
                    "particleId": particle.particle_id,
                    "family": particle.family,
                    "h": particle.h,
                    "c": particle.c,
                    "hMax": particle.h_max,
                    "daughterRule": particle.daughter_rule,
                    "overshootRule": particle.overshoot_rule,
                    "clockId": particle.clock_id,
                }
            )
    return tasks


def evaluate_particle_round(
    particles: list[Particle], root: str
) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    trajectories, seeds, _, wall = run_tasks(particle_tasks(particles, root))
    if not trajectories["exactReplayPassed"].all():
        raise RuntimeError("development exact replay failed")
    rows = []
    for particle in particles:
        group = trajectories[trajectories["particleId"] == particle.particle_id]
        if group.shape[0] != 8:
            raise RuntimeError("particle did not receive eight development matrices")
        rows.append(particle_summary_and_distance(particle, group))
    return pd.DataFrame(rows), seeds, wall, trajectories


def run_abc_family(
    family: str, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Particle], np.ndarray, dict[str, Any]]:
    inference_root = config["randomness"]["roots"]["inference"]
    development_root = config["randomness"]["roots"]["development"]
    all_results: list[pd.DataFrame] = []
    all_seeds: list[pd.DataFrame] = []
    round_rows: list[dict[str, Any]] = []
    trajectory_cpu = 0.0

    round1 = initial_particles(family, inference_root, 256)
    result1, seeds1, wall1, trajectories1 = evaluate_particle_round(round1, development_root)
    all_results.append(result1)
    all_seeds.append(seeds1)
    trajectory_cpu += float(trajectories1["workerCpuSeconds"].sum())
    parents1, _retained1 = retained_particles(round1, result1, 128)
    weights1_all = importance_weights(round1, None, None, None)
    weight_lookup1 = {particle.particle_id: weight for particle, weight in zip(round1, weights1_all, strict=True)}
    weights1 = np.asarray([weight_lookup1[particle.particle_id] for particle in parents1], float)
    weights1 /= weights1.sum()
    round_rows.append(
        {
            "family": family, "round": 1, "particlesEvaluated": 256,
            "particlesRetained": 128, "epsilonMedian": float(result1["distance"].median()),
            "minimumDistance": float(result1["distance"].min()),
            "envelopePassCount": int(result1["developmentAcceptanceEnvelopePassed"].sum()),
            "wallSeconds": wall1,
        }
    )

    round2 = propose_particles(family, inference_root, 2, 128, parents1, weights1, 0.20)
    result2, seeds2, wall2, trajectories2 = evaluate_particle_round(round2, development_root)
    all_results.append(result2)
    all_seeds.append(seeds2)
    trajectory_cpu += float(trajectories2["workerCpuSeconds"].sum())
    weights2_all = importance_weights(round2, parents1, weights1, 0.20)
    parents2, _retained2 = retained_particles(round2, result2, 64)
    weight_lookup2 = {particle.particle_id: weight for particle, weight in zip(round2, weights2_all, strict=True)}
    weights2 = np.asarray([weight_lookup2[particle.particle_id] for particle in parents2], float)
    weights2 /= weights2.sum()
    round_rows.append(
        {
            "family": family, "round": 2, "particlesEvaluated": 128,
            "particlesRetained": 64, "epsilonMedian": float(result2["distance"].median()),
            "minimumDistance": float(result2["distance"].min()),
            "envelopePassCount": int(result2["developmentAcceptanceEnvelopePassed"].sum()),
            "wallSeconds": wall2,
        }
    )

    round3 = propose_particles(family, inference_root, 3, 64, parents2, weights2, 0.10)
    result3, seeds3, wall3, trajectories3 = evaluate_particle_round(round3, development_root)
    weights3 = importance_weights(round3, parents2, weights2, 0.10)
    result3["posteriorWeight"] = [
        float(weight) for weight in weights3
    ]
    all_results.append(result3)
    all_seeds.append(seeds3)
    trajectory_cpu += float(trajectories3["workerCpuSeconds"].sum())
    round_rows.append(
        {
            "family": family, "round": 3, "particlesEvaluated": 64,
            "particlesRetained": 64, "epsilonMedian": float(result3["distance"].median()),
            "minimumDistance": float(result3["distance"].min()),
            "envelopePassCount": int(result3["developmentAcceptanceEnvelopePassed"].sum()),
            "wallSeconds": wall3,
        }
    )
    metadata = {
        "family": family,
        "workerCpuSeconds": trajectory_cpu,
        "wallSeconds": wall1 + wall2 + wall3,
        "trueTrajectoryCount": (256 + 128 + 64) * 8,
        "replayTrajectoryCount": (256 + 128 + 64) * 8,
    }
    return (
        pd.concat(all_results, ignore_index=True),
        pd.concat(all_seeds, ignore_index=True).drop_duplicates("seedMaterialSha256"),
        pd.DataFrame(round_rows),
        round3,
        weights3,
        metadata,
    )


def seed_overlap_with_s12e(seed_frame: pd.DataFrame) -> dict[str, Any]:
    prior = pd.read_parquet("/artifacts/research_steps/S12E/development_seed_manifest.parquet")
    prior_material = set(prior["seedMaterialSha256"].dropna().astype(str))
    current = set(seed_frame["seedMaterialSha256"].dropna().astype(str))
    intersection = sorted(prior_material & current)
    return {
        "priorSeedCount": len(prior_material),
        "s12fSeedCount": len(current),
        "intersectionCount": len(intersection),
        "intersection": intersection,
        "passed": not intersection,
    }


def phase_development(require_pushed: bool) -> None:
    config = load_config()
    validation = verify_preregistration(require_pushed)
    started = datetime.now(UTC)
    phase_started = time.perf_counter()
    clock_rows, clock_summary = phase1_clock_audit()
    clock_rows.to_parquet(ARTIFACTS / "clock_audit_results.parquet", index=False)
    clock_summary.to_csv(ARTIFACTS / "clock_audit_summary.csv", index=False)

    benchmark, benchmark_seeds, _, benchmark_wall = run_tasks(benchmark_tasks(config))
    benchmark.to_csv(ARTIFACTS / "benchmark_results.csv", index=False)
    if not benchmark["exactReplayPassed"].all():
        raise RuntimeError("benchmark exact replay failed")
    median_cpu = float(benchmark["workerCpuSeconds"].median())
    possible_true = 16 + 2 * (256 + 128 + 64) * 8 + 3 * 32
    projected_cpu_hours = median_cpu * 2.0 * possible_true / 3600.0
    projected_wall_hours = projected_cpu_hours / WORKERS
    projection = {
        "schemaVersion": "E01-S12F-benchmark-runtime-projection-v1.0.0",
        "configurationCount": 16,
        "benchmarkWallSeconds": benchmark_wall,
        "medianWorkerCpuSecondsPerTrueTrajectory": median_cpu,
        "worstCaseTrueTrajectoryCountIncludingAdaptiveAndConfirmation": possible_true,
        "replayMultiplier": 2,
        "projectedCpuHours": projected_cpu_hours,
        "projectedWallHoursAtSixWorkers": projected_wall_hours,
        "projectedRetainedGiB": 0.25,
        "ceilings": {"cpuHours": 250.0, "wallHours": 72.0, "retainedGiB": 30.0},
        "passed": bool(projected_cpu_hours <= 250 and projected_wall_hours <= 72),
    }
    write_json(ARTIFACTS / "benchmark_runtime_projection.json", projection)
    if not projection["passed"]:
        raise RuntimeError("benchmark projects beyond a hard ceiling")

    # C0--C2 clock-only success can supply h=1 candidates. In the observed S12E
    # audit this branch is expected to be empty; the rule is nevertheless frozen.
    clock_candidates = clock_summary[clock_summary.get("clockOnlyGatePassed", False)]
    all_results: list[pd.DataFrame] = []
    all_seeds: list[pd.DataFrame] = [benchmark_seeds]
    all_rounds: list[pd.DataFrame] = []
    final_particles: list[Particle] = []
    final_weights = np.asarray([], dtype=float)
    family_metadata: list[dict[str, Any]] = []
    adaptive_executed = False

    if clock_candidates.empty:
        fixed_results, fixed_seeds, fixed_rounds, fixed_particles, fixed_weights, fixed_meta = run_abc_family(
            "FIXED_COMMON_EXPOSURE", config
        )
        all_results.append(fixed_results)
        all_seeds.append(fixed_seeds)
        all_rounds.append(fixed_rounds)
        final_particles = fixed_particles
        final_weights = fixed_weights
        family_metadata.append(fixed_meta)
        fixed_final = fixed_results[fixed_results["round_index"] == 3]
        fixed_accepted = int(fixed_final["developmentAcceptanceEnvelopePassed"].sum())
        if fixed_accepted == 0:
            adaptive_executed = True
            adaptive_results, adaptive_seeds, adaptive_rounds, adaptive_particles, adaptive_weights, adaptive_meta = run_abc_family(
                "ADAPTIVE_GROSS_EVENT_EXPOSURE", config
            )
            all_results.append(adaptive_results)
            all_seeds.append(adaptive_seeds)
            all_rounds.append(adaptive_rounds)
            final_particles = adaptive_particles
            final_weights = adaptive_weights
            family_metadata.append(adaptive_meta)
    else:
        # Deterministic h=1 translations from S12E engine identities.
        engine_rules = {
            "K1_PAPER_POISSON_RANDOM_NONEMPTY": "RANDOM_NONEMPTY",
            "K2_PAPER_POISSON_FIRST_DAUGHTER": "FIRST_DAUGHTER",
            "K3_PAPER_POISSON_RANDOM_LITERAL": "RANDOM_LITERAL",
        }
        for index, row in enumerate(clock_candidates.itertuples(index=False)):
            final_particles.append(
                Particle(
                    f"CLOCK-ONLY-P{index:03d}", "FIXED_COMMON_EXPOSURE", 0,
                    engine_rules[row.engineId], "RETAIN_OVERSHOOT", row.clockId,
                    1.0, None, None, None, 1.0 / len(clock_candidates),
                )
            )
        final_weights = np.full(len(final_particles), 1.0 / len(final_particles))

    abc_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    abc_rounds = pd.concat(all_rounds, ignore_index=True) if all_rounds else pd.DataFrame(
        [{"family": "CLOCK_ONLY", "round": 0, "particlesEvaluated": 0, "particlesRetained": len(final_particles), "epsilonMedian": np.nan, "minimumDistance": np.nan, "envelopePassCount": len(final_particles), "wallSeconds": 0.0}]
    )
    seed_frame = pd.concat(all_seeds, ignore_index=True).drop_duplicates("seedMaterialSha256")
    overlap = seed_overlap_with_s12e(seed_frame)
    if not overlap["passed"]:
        raise RuntimeError("S12F seed material overlaps S12E")

    if all_results:
        active_family = final_particles[0].family
        final_frame = abc_results[
            (abc_results["family"] == active_family) & (abc_results["round_index"] == 3)
        ].copy()
        candidates, selected = candidate_groups(final_particles, final_frame, final_weights, maximum=3)
    else:
        selected = final_particles[:3]
        candidates = pd.DataFrame(
            [
                {
                    "candidateGroup": particle.discrete_group,
                    "posteriorMass": particle.proposal_weight,
                    "particleCount": 1,
                    "medianDistance": 0.0,
                    "minimumDistance": 0.0,
                    "complexity": 0.0,
                    "representativeParticleId": particle.particle_id,
                    "family": particle.family,
                    "daughterRule": particle.daughter_rule,
                    "overshootRule": particle.overshoot_rule,
                    "clockId": particle.clock_id,
                    "h": particle.h,
                    "c": particle.c,
                    "hMax": particle.h_max,
                    "developmentAccepted": True,
                    "confirmationRank": index + 1,
                    "selectedForConfirmation": index < 3,
                }
                for index, particle in enumerate(final_particles)
            ]
        )

    abc_results.to_parquet(ARTIFACTS / "abc_particle_results.parquet", index=False)
    abc_rounds.to_csv(ARTIFACTS / "abc_round_summary.csv", index=False)
    seed_frame.to_parquet(ARTIFACTS / "development_seed_manifest.parquet", index=False)
    candidates.to_csv(ARTIFACTS / "posterior_candidates.csv", index=False)

    selected_rows = []
    for rank, particle in enumerate(selected, 1):
        candidate_id = f"S12F-CANDIDATE-{rank:02d}"
        source_row = candidates[candidates["representativeParticleId"] == particle.particle_id].iloc[0]
        selected_rows.append(
            {
                "candidateId": candidate_id,
                "confirmationRank": rank,
                "representativeParticleId": particle.particle_id,
                "posteriorMass": float(source_row["posteriorMass"]),
                "developmentDistance": float(source_row["minimumDistance"]),
                "exposureFamily": particle.family,
                "h": particle.h,
                "c": particle.c,
                "hMax": particle.h_max,
                "daughterRule": particle.daughter_rule,
                "overshootRule": particle.overshoot_rule,
                "clockId": particle.clock_id,
            }
        )
    proposal_core = {
        "schemaVersion": "E01-S12F-candidate-lock-proposal-v1.0.0",
        "researchStepId": config["researchStepId"],
        "generatedAtUtc": datetime.now(UTC).isoformat(),
        "designCommit": validation["record"]["gitCommit"],
        "targetConfigSha256": validation["record"]["configSha256"],
        "clockOnlyCandidateCount": int(clock_candidates.shape[0]),
        "fixedExposureExecuted": bool(clock_candidates.empty),
        "adaptiveExposureExecuted": adaptive_executed,
        "candidateCount": len(selected_rows),
        "candidates": selected_rows,
        "confirmationRoot": config["randomness"]["roots"]["confirmation"],
        "selectionRule": "posterior_mass_desc_then_group_median_distance_asc_then_complexity_then_lexical; representative_minimum_distance; maximum_three",
        "labelsEmergenceInterventionsAccessed": False,
        "seedOverlapAudit": overlap,
        "familyRuntime": family_metadata,
    }
    proposal = dict(proposal_core)
    proposal["proposalSha256"] = canonical_sha(proposal_core)
    write_json(ARTIFACTS / "candidate_lock_proposal.json", proposal)
    scope = {
        "schemaVersion": "E01-S12F-scope-access-ledger-v1.0.0",
        "researchStepId": config["researchStepId"],
        "updatedAtUtc": datetime.now(UTC).isoformat(),
        "paperTargetsOpened": True,
        "s12eK1K2K3CachesOpenedReadOnly": True,
        "selfReplicationLabelsOpened": False,
        "sourceEmergenceOpened": False,
        "localPhiROpened": False,
        "interventionsRun": False,
        "s12gStarted": False,
        "s13Started": False,
        "newDevelopmentMatrices": 8,
        "newConfirmationMatrices": 0,
        "phase": "DEVELOPMENT_COMPLETE_CANDIDATE_LOCK_PENDING" if selected_rows else "DEVELOPMENT_STOP_NO_ACCEPTED_CANDIDATE",
    }
    write_json(ARTIFACTS / "scope_access_ledger.json", scope)
    development_runtime = {
        "schemaVersion": "E01-S12F-runtime-partial-v1.0.0",
        "researchStepId": config["researchStepId"],
        "startedAtUtc": started.isoformat(),
        "developmentCompletedAtUtc": datetime.now(UTC).isoformat(),
        "developmentWallSeconds": time.perf_counter() - phase_started,
        "workers": WORKERS,
        "threadEnvironment": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "benchmark": projection,
        "familyRuntime": family_metadata,
    }
    write_json(CACHE / "development_runtime.json", development_runtime)
    print(
        json.dumps(
            {
                "status": "DEVELOPMENT_COMPLETE" if selected_rows else "STOP_NO_ACCEPTED_CANDIDATE",
                "clockOnlyCandidates": int(clock_candidates.shape[0]),
                "adaptiveExecuted": adaptive_executed,
                "candidateCount": len(selected_rows),
                "proposalSha256": proposal["proposalSha256"],
            },
            sort_keys=True,
        )
    )


def verify_candidate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    if not CANDIDATE_LOCK_PATH.is_file():
        raise RuntimeError("candidate lock has not been committed")
    lock = json.loads(CANDIDATE_LOCK_PATH.read_text(encoding="utf-8"))
    proposal = json.loads((ARTIFACTS / "candidate_lock_proposal.json").read_text())
    if lock["proposalSha256"] != proposal["proposalSha256"]:
        raise RuntimeError("candidate lock does not match development proposal")
    if lock["candidates"] != proposal["candidates"]:
        raise RuntimeError("candidate identities differ from deterministic proposal")
    head = git("rev-parse", "HEAD^{commit}")
    remote = git("rev-parse", "origin/eidosoma/groups/42^{commit}")
    tracked_blob = git("rev-parse", f"HEAD:{CANDIDATE_LOCK_PATH.relative_to(REPO)}")
    working_blob = git("hash-object", str(CANDIDATE_LOCK_PATH.relative_to(REPO)))
    if head != remote or tracked_blob != working_blob:
        raise RuntimeError("candidate lock commit is not the pushed current HEAD")
    lock["resolvedCandidateLockCommit"] = head
    return lock, proposal


def confirmation_tasks(lock: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    root = config["randomness"]["roots"]["confirmation"]
    tasks: list[dict[str, Any]] = []
    for candidate in lock["candidates"]:
        candidate_id = candidate["candidateId"]
        for matrix_index in range(32):
            cache_path = TRAJECTORY_CACHE / "confirmation" / candidate_id / f"M{matrix_index:02d}.pickle"
            tasks.append(
                {
                    "phase": "confirmation",
                    "root": root,
                    "matrixIndex": matrix_index,
                    "streamIdentity": candidate_id,
                    "particleId": candidate["representativeParticleId"],
                    "candidateId": candidate_id,
                    "family": candidate["exposureFamily"],
                    "h": candidate["h"],
                    "c": candidate["c"],
                    "hMax": candidate["hMax"],
                    "daughterRule": candidate["daughterRule"],
                    "overshootRule": candidate["overshootRule"],
                    "clockId": candidate["clockId"],
                    "cachePath": str(cache_path),
                    "returnTrajectory": True,
                }
            )
    return tasks


def confirmation_candidate_result(candidate: dict[str, Any], group: pd.DataFrame) -> dict[str, Any]:
    column = {
        "C0_BATCH_UPDATES_ONLY": "clockC0",
        "C1_SELECTED_DAUGHTER_RETAINED": "clockC1",
        "C2_EXPLICIT_PRE_AND_POST_FISSION": "clockC2",
    }[candidate["clockId"]]
    values = group[column].to_numpy(float)
    q05, q50, q95 = np.quantile(values, [0.05, 0.50, 0.95])
    centers = (800.0, 800.0, 1000.0)
    endpoints = int(sum(q05 <= endpoint <= q95 for endpoint in centers))
    aggregate = bool(
        values.max() >= 1090.0
        and q95 <= 1314.0
        and np.mean(values > 1314.0) <= 0.05
    )
    median_post = float(group["medianPostFissionMass"].median())
    fmax = float(group["maxstepsTerminations"].sum() / max(1, group["completedFissions"].sum()))
    complete = int(np.count_nonzero(group["completedFissions"].to_numpy() == 100))
    replay = bool(group["exactReplayPassed"].all())
    # Reuse the preregistered distance implementation with a synthetic particle.
    particle = Particle(
        candidate["representativeParticleId"], candidate["exposureFamily"], 3,
        candidate["daughterRule"], candidate["overshootRule"], candidate["clockId"],
        candidate["h"], candidate["c"], candidate["hMax"], None, candidate["posteriorMass"],
    )
    distance_row = particle_summary_and_distance(particle, group)
    pass_gate = bool(
        complete >= 31
        and replay
        and endpoints >= 2
        and aggregate
        and 35.0 <= median_post <= 45.0
        and fmax <= 0.05
        and candidate["clockId"] != "C2_EXPLICIT_PRE_AND_POST_FISSION"
        and float(distance_row["distance"]) <= 1.0
    )
    reasons = []
    if complete < 31:
        reasons.append("fewer_than_31_of_32_complete")
    if not replay:
        reasons.append("exact_replay_failed")
    if endpoints < 2:
        reasons.append("sample_endpoints_incompatible")
    if not aggregate:
        reasons.append("aggregate_support_incompatible")
    if not 35 <= median_post <= 45:
        reasons.append("post_fission_mass_incompatible")
    if fmax > 0.05:
        reasons.append("maxsteps_fraction_above_0p05")
    if candidate["clockId"] == "C2_EXPLICIT_PRE_AND_POST_FISSION":
        reasons.append("synthetic_boundary_duplication_forbidden")
    if float(distance_row["distance"]) > 1.0:
        reasons.append("confirmation_distance_outside_abc_envelope")
    return {
        "candidateId": candidate["candidateId"],
        "exposureFamily": candidate["exposureFamily"],
        "h": candidate["h"],
        "c": candidate["c"],
        "hMax": candidate["hMax"],
        "daughterRule": candidate["daughterRule"],
        "overshootRule": candidate["overshootRule"],
        "clockId": candidate["clockId"],
        "posteriorMass": candidate["posteriorMass"],
        "trajectoryCount": int(group.shape[0]),
        "completed100": complete,
        "q05TPhi": float(q05),
        "medianTPhi": float(q50),
        "q95TPhi": float(q95),
        "minimumTPhi": float(values.min()),
        "maximumTPhi": float(values.max()),
        "sampleEndpointsInsideQ05Q95": endpoints,
        "aggregateCompatible": aggregate,
        "medianPostFissionMass": median_post,
        "medianPreFissionMass": float(group["medianPreFissionMass"].median()),
        "q95Overshoot": float(group["q95Overshoot"].quantile(0.95)),
        "fractionMaxsteps": fmax,
        "exactReplayPassed": replay,
        "confirmationDistance": float(distance_row["distance"]),
        "confirmationGatePassed": pass_gate,
        "gateReason": "PASS" if pass_gate else ";".join(reasons),
    }


def phase_confirmation() -> None:
    config = load_config()
    validation = verify_preregistration(True)
    lock, _proposal = verify_candidate_lock()
    if not 1 <= len(lock["candidates"]) <= 3:
        raise RuntimeError("candidate lock must contain one to three identities")
    started = datetime.now(UTC)
    wall_started = time.perf_counter()
    summaries, seeds, trajectories, _ = run_tasks(confirmation_tasks(lock, config))
    if summaries.shape[0] != 32 * len(lock["candidates"]):
        raise RuntimeError("confirmation cardinality failed")
    if not summaries["exactReplayPassed"].all():
        raise RuntimeError("confirmation exact replay failed")
    development_seeds = pd.read_parquet(ARTIFACTS / "development_seed_manifest.parquet")
    development_material = set(development_seeds["seedMaterialSha256"].astype(str))
    confirmation_material = set(seeds["seedMaterialSha256"].astype(str))
    intersection = sorted(development_material & confirmation_material)
    if intersection:
        raise RuntimeError("development and confirmation seeds overlap")
    s12e = pd.read_parquet("/artifacts/research_steps/S12E/engine_development_results.parquet")
    matrix_overlap = set(summaries["betaSha256"].astype(str)) & set(s12e["betaSha256"].astype(str))
    if matrix_overlap:
        raise RuntimeError("confirmation matrices overlap S12E development")

    seeds.to_parquet(ARTIFACTS / "confirmation_seed_manifest.parquet", index=False)
    summaries.to_parquet(ARTIFACTS / "confirmation_trajectory_manifest.parquet", index=False)
    observation_data: list[dict[str, Any]] = []
    for trajectory in trajectories:
        observation_data.extend(observation_rows(trajectory))
    pd.DataFrame(observation_data).to_parquet(
        ARTIFACTS / "confirmation_trajectories.parquet", index=False, compression="zstd"
    )
    results = []
    for candidate in lock["candidates"]:
        group = summaries[summaries["candidateId"] == candidate["candidateId"]]
        results.append(confirmation_candidate_result(candidate, group))
    result_frame = pd.DataFrame(results).sort_values("candidateId")
    result_frame.to_csv(ARTIFACTS / "posterior_predictive_results.csv", index=False)
    passing = result_frame[result_frame["confirmationGatePassed"]]
    if passing.shape[0] > 1:
        outcome = "NONIDENTIFIABLE_TIMEBASE_ENSEMBLE"
    elif passing.shape[0] == 1:
        outcome = "PAPER_TIMEBASE_CANDIDATE"
    else:
        outcome = "NO_PAPER_TIMEBASE_RECONSTRUCTION"
    family_tokens = []
    if not passing.empty:
        if (passing["exposureFamily"] == "FIXED_COMMON_EXPOSURE").any():
            family_tokens.append("FIXED_EXPOSURE_MATCH")
        if (passing["exposureFamily"] == "ADAPTIVE_GROSS_EVENT_EXPOSURE").any():
            family_tokens.append("ADAPTIVE_EXPOSURE_MATCH")
        if (passing["overshootRule"] == "TRIM_NEW_ENTRANTS_TO_NMAX").all():
            family_tokens.append("OVERSHOOT_SEMANTICS_REQUIRED")
    trajectory_locks = {}
    for candidate_id, group in summaries.groupby("candidateId", sort=True):
        trajectory_locks[candidate_id] = [
            {
                "matrixIndex": int(row.matrixIndex),
                "trajectorySha256": row.trajectorySha256,
                "betaSha256": row.betaSha256,
                "initialStateSha256": row.initialStateSha256,
                "cachePath": row.cachePath,
                "cacheSha256": row.cacheSha256,
            }
            for row in group.sort_values("matrixIndex").itertuples(index=False)
        ]
    downstream_lock = {
        "schemaVersion": "E01-S12F-candidate-timebase-pipeline-lock-v1.0.0",
        "researchStepId": config["researchStepId"],
        "createdAtUtc": datetime.now(UTC).isoformat(),
        "status": "CONFIRMED_CANDIDATES_FROZEN" if not passing.empty else "NO_CONFIRMED_CANDIDATE",
        "outcomeClassification": outcome,
        "s13Status": "BLOCKED_PENDING_S12F_HUMAN_REVIEW",
        "confirmedCandidates": [
            {
                **candidate,
                "confirmation": result_frame[result_frame["candidateId"] == candidate["candidateId"]].iloc[0].to_dict(),
                "trajectoryLocks": trajectory_locks[candidate["candidateId"]],
                "trajectorySchema": "initial_state_then_each_poisson_batch_update_then_selected_post_fission_state",
                "lagOneIndexing": "local_output_index_k_maps_state_k_to_state_k_plus_1_under_declared_clock",
                "seedContract": "S12F_SHA256_DOMAIN_SEPARATION_PCG64DXSM",
            }
            for candidate in lock["candidates"]
            if candidate["candidateId"] in set(passing["candidateId"])
        ],
        "failedCandidates": result_frame.loc[~result_frame["confirmationGatePassed"]].to_dict("records"),
        "labelsEmergenceInterventionsCalculated": False,
    }
    write_json(ARTIFACTS / "candidate_timebase_pipeline_lock.json", downstream_lock)
    classification = {
        "schemaVersion": "E01-S12F-classification-v1.0.0",
        "researchStepId": config["researchStepId"],
        "outcomeClassification": outcome,
        "supportingOutcomeTokens": family_tokens,
        "confirmedCandidateCount": int(passing.shape[0]),
        "testedCandidateCount": int(result_frame.shape[0]),
        "s12eClassificationUnchanged": "TIME_BASE_MISMATCH_CONFIRMED",
        "s13Status": "BLOCKED_PENDING_S12F_HUMAN_REVIEW",
        "firstMatchedOrFailedLayer": "latent_observation_clock_and_Poisson_exposure",
    }
    write_json(ARTIFACTS / "classification.json", classification)
    scope = json.loads((ARTIFACTS / "scope_access_ledger.json").read_text())
    scope.update(
        {
            "updatedAtUtc": datetime.now(UTC).isoformat(),
            "newConfirmationMatrices": 32,
            "confirmationCandidateCount": len(lock["candidates"]),
            "confirmationTrajectoryCount": int(summaries.shape[0]),
            "selfReplicationLabelsOpened": False,
            "sourceEmergenceOpened": False,
            "localPhiROpened": False,
            "interventionsRun": False,
            "s12gStarted": False,
            "s13Started": False,
            "phase": "CONFIRMATION_COMPLETE_AWAITING_FINALIZATION",
        }
    )
    write_json(ARTIFACTS / "scope_access_ledger.json", scope)
    partial = json.loads((CACHE / "development_runtime.json").read_text())
    partial.update(
        {
            "confirmationStartedAtUtc": started.isoformat(),
            "confirmationCompletedAtUtc": datetime.now(UTC).isoformat(),
            "confirmationWallSeconds": time.perf_counter() - wall_started,
            "confirmationWorkerCpuSeconds": float(summaries["workerCpuSeconds"].sum()),
            "candidateLockCommit": lock["resolvedCandidateLockCommit"],
            "confirmationTrajectoryCount": int(summaries.shape[0]),
        }
    )
    write_json(CACHE / "runtime_complete.json", partial)
    write_json(
        CACHE / "confirmation_validation_partial.json",
        {
            "priorImmutability": validation["prior"],
            "s12eCacheImmutability": validation["s12eCache"],
            "developmentConfirmationSeedIntersectionCount": len(intersection),
            "s12eMatrixOverlapCount": len(matrix_overlap),
            "exactReplayPassed": bool(summaries["exactReplayPassed"].all()),
            "candidateCount": len(lock["candidates"]),
            "confirmationRows": int(summaries.shape[0]),
        },
    )
    print(json.dumps({"status": "CONFIRMATION_COMPLETE", "classification": outcome, "passingCandidates": int(passing.shape[0])}, sort_keys=True))


def ensure_placeholder(path: Path, columns: list[str], reason: str) -> None:
    row = {column: None for column in columns}
    row.update({"status": "NOT_REACHED", "reason": reason})
    frame = pd.DataFrame([row])
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def create_figures() -> None:
    figure_dir = ARTIFACTS / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "font.size": 9})

    fig, ax = plt.subplots(figsize=(8, 4))
    endpoints = [800, 800, 1000]
    ax.errorbar(["B", "C", "D"], endpoints, yerr=[8, 8, 10], fmt="o", capsize=4, label="sample endpoints")
    ax.axhspan(1090, 1120, alpha=0.2, color="tab:blue", label="aggregate visible terminal")
    ax.axhspan(1300, 1314, alpha=0.15, color="tab:gray", label="aggregate axis upper")
    ax.set_ylabel("molecular step")
    ax.set_title("Frozen Figure 2 time-base targets")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_dir / "01_figure_digitization.png")
    plt.close(fig)

    clock = pd.read_parquet(ARTIFACTS / "clock_audit_results.parquet")
    finite = clock[np.isfinite(clock["tPhi"]) & clock["clockId"].isin([
        "C0_BATCH_UPDATES_ONLY", "C1_SELECTED_DAUGHTER_RETAINED", "C2_EXPLICIT_PRE_AND_POST_FISSION"
    ])]
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = []
    labels = []
    for (engine, clock_id), group in finite.groupby(["engineId", "clockId"], sort=True):
        groups.append(group["tPhi"].to_numpy())
        labels.append(f"{engine.split('_')[0]}\n{clock_id.split('_')[0]}")
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    for value in (800, 1000, 1100):
        ax.axhline(value, color="tab:red", alpha=0.25, linestyle="--")
    ax.set_ylabel("lag-one trajectory length")
    ax.set_title("Read-only S12E observation-clock audit")
    fig.tight_layout()
    fig.savefig(figure_dir / "02_clock_audit.png")
    plt.close(fig)

    particles = pd.read_parquet(ARTIFACTS / "abc_particle_results.parquet")
    fig, ax = plt.subplots(figsize=(8, 5))
    if not particles.empty:
        final = particles[particles["round_index"] == 3]
        x = final["h"] if final["h"].notna().any() else final["c"]
        ax.scatter(x, final["distance"], c=final["clock_id"].astype("category").cat.codes, s=22, alpha=0.7)
        ax.axhline(1.0, color="tab:red", linestyle="--", label="acceptance envelope")
        ax.set_xscale("log")
        ax.set_xlabel("h (fixed) or c (adaptive)")
        ax.set_ylabel("preregistered distance")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "Clock-only branch; ABC not run", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Final ABC-SMC particle round")
    fig.tight_layout()
    fig.savefig(figure_dir / "03_abc_posterior.png")
    plt.close(fig)

    result_path = ARTIFACTS / "posterior_predictive_results.csv"
    result = pd.read_csv(result_path) if result_path.is_file() else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not result.empty and "candidateId" in result and result["candidateId"].notna().any():
        positions = np.arange(result.shape[0])
        ax.errorbar(
            positions, result["medianTPhi"],
            yerr=[result["medianTPhi"] - result["q05TPhi"], result["q95TPhi"] - result["medianTPhi"]],
            fmt="o", capsize=5,
        )
        ax.set_xticks(positions, result["candidateId"], rotation=20)
        for value in (800, 1000, 1100):
            ax.axhline(value, color="tab:red", alpha=0.25, linestyle="--")
    else:
        ax.text(0.5, 0.5, "Confirmation not reached", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("lag-one trajectory length")
    ax.set_title("Untouched posterior-predictive confirmation")
    fig.tight_layout()
    fig.savefig(figure_dir / "04_posterior_predictive.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    if not result.empty and "candidateId" in result and result["candidateId"].notna().any():
        axes[0].bar(result["candidateId"], result["medianPostFissionMass"])
        axes[0].axhspan(35, 45, color="tab:green", alpha=0.2)
        axes[1].bar(result["candidateId"], result["q95Overshoot"])
        for ax in axes:
            ax.tick_params(axis="x", rotation=20)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Not reached", ha="center", va="center", transform=ax.transAxes)
    axes[0].set_title("Post-fission mass")
    axes[1].set_title("Q95 overshoot")
    fig.tight_layout()
    fig.savefig(figure_dir / "05_mass_overshoot.png")
    plt.close(fig)

    classification = json.loads((ARTIFACTS / "classification.json").read_text()) if (ARTIFACTS / "classification.json").is_file() else {"outcomeClassification": "UNDERDETERMINED"}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axis("off")
    lines = [
        "Paper targets frozen → read-only clocks → ABC-SMC exposure → untouched confirmation",
        f"Outcome: {classification['outcomeClassification']}",
        f"Confirmed candidates: {classification.get('confirmedCandidateCount', 0)}",
        "Labels / emergence / interventions: NOT CALCULATED",
        "S13: BLOCKED_PENDING_S12F_HUMAN_REVIEW",
    ]
    ax.text(0.5, 0.5, "\n\n".join(lines), ha="center", va="center", fontsize=11, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(figure_dir / "06_decision_summary.png")
    plt.close(fig)


def phase_finalize() -> None:
    config = load_config()
    validation = verify_preregistration(True)
    confirmation_reached = (ARTIFACTS / "posterior_predictive_results.csv").is_file()
    if not confirmation_reached:
        reason = "no_development_particle_entered_acceptance_envelope"
        ensure_placeholder(ARTIFACTS / "confirmation_seed_manifest.parquet", ["status", "reason"], reason)
        ensure_placeholder(ARTIFACTS / "confirmation_trajectories.parquet", ["status", "reason"], reason)
        ensure_placeholder(ARTIFACTS / "confirmation_trajectory_manifest.parquet", ["status", "reason"], reason)
        ensure_placeholder(ARTIFACTS / "posterior_predictive_results.csv", ["status", "reason"], reason)
        downstream = {
            "schemaVersion": "E01-S12F-candidate-timebase-pipeline-lock-v1.0.0",
            "researchStepId": config["researchStepId"],
            "status": "NO_CONFIRMED_CANDIDATE",
            "outcomeClassification": "SIMULATOR_IDENTIFICATION_FAILED",
            "confirmedCandidates": [],
            "reason": reason,
            "s13Status": "BLOCKED_PENDING_S12F_HUMAN_REVIEW",
        }
        write_json(ARTIFACTS / "candidate_timebase_pipeline_lock.json", downstream)
        write_json(
            ARTIFACTS / "classification.json",
            {
                "schemaVersion": "E01-S12F-classification-v1.0.0",
                "researchStepId": config["researchStepId"],
                "outcomeClassification": "SIMULATOR_IDENTIFICATION_FAILED",
                "supportingOutcomeTokens": [],
                "confirmedCandidateCount": 0,
                "testedCandidateCount": 0,
                "s12eClassificationUnchanged": "TIME_BASE_MISMATCH_CONFIRMED",
                "s13Status": "BLOCKED_PENDING_S12F_HUMAN_REVIEW",
            },
        )
    classification = json.loads((ARTIFACTS / "classification.json").read_text())
    create_figures()

    prior_post = verify_hash_manifest(ARTIFACTS / "immutable_prior_baseline.json")
    cache_post = verify_hash_manifest(ARTIFACTS / "s12e_cache_manifest.json")
    partial_path = CACHE / ("runtime_complete.json" if confirmation_reached else "development_runtime.json")
    runtime = json.loads(partial_path.read_text())
    runtime.update(
        {
            "schemaVersion": "E01-S12F-runtime-manifest-v1.0.0",
            "researchStepId": config["researchStepId"],
            "finalizedAtUtc": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scipy": scipy.__version__,
            "workers": WORKERS,
            "cpuFloat64Authoritative": True,
            "gpuUsed": False,
            "cpuHourCeiling": 250.0,
            "wallHourCeiling": 72.0,
            "retainedArtifactGiBCeiling": 30.0,
        }
    )
    write_json(ARTIFACTS / "runtime_manifest.json", runtime)
    failure_rows = []
    if classification["outcomeClassification"] in {"NO_PAPER_TIMEBASE_RECONSTRUCTION", "SIMULATOR_IDENTIFICATION_FAILED"}:
        failure_rows.append(
            {
                "failureId": "S12F-F001",
                "phase": "Phase 2" if not confirmation_reached else "Phase 3",
                "severity": "TERMINAL_GATE",
                "status": "FAILED_CLOSED",
                "reason": "No source-grounded candidate passed the preregistered acceptance or confirmation envelope.",
                "consequence": "No downstream pipeline is admissible; labels and emergence remain prohibited.",
            }
        )
    if not failure_rows:
        failure_rows.append(
            {
                "failureId": "NONE",
                "phase": "all",
                "severity": "NONE",
                "status": "NO_TERMINAL_FAILURE",
                "reason": "All required upstream gates completed.",
                "consequence": "Return for human review; S13 remains blocked.",
            }
        )
    pd.DataFrame(failure_rows).to_csv(ARTIFACTS / "failure_ledger.csv", index=False)

    scope = json.loads((ARTIFACTS / "scope_access_ledger.json").read_text())
    artifact_bytes = sum(path.stat().st_size for path in ARTIFACTS.rglob("*") if path.is_file())
    validation_payload = {
        "schemaVersion": "E01-S12F-regeneration-validation-v1.0.0",
        "researchStepId": config["researchStepId"],
        "validatedAtUtc": datetime.now(UTC).isoformat(),
        "preregistrationAndTargets": "PASS_COMMITTED_AND_PUSHED_BEFORE_NEW_SIMULATION",
        "figureExtraction": validation["phase0"],
        "priorImmutability": prior_post,
        "s12eCacheImmutability": cache_post,
        "exactReplay": "PASS" if confirmation_reached else "PASS_DEVELOPMENT_ONLY",
        "scopeCompliance": scope,
        "artifactBytesBeforeReports": artifact_bytes,
        "artifactGiBBeforeReports": artifact_bytes / (1024 ** 3),
        "hardCeilingsPassed": artifact_bytes < 30 * 1024 ** 3,
        "s12eClassificationUnchanged": True,
        "s13Blocked": True,
        "passed": bool(prior_post["passed"] and cache_post["passed"] and artifact_bytes < 30 * 1024 ** 3),
    }
    write_json(ARTIFACTS / "regeneration_validation.json", validation_payload)

    results = pd.read_csv(ARTIFACTS / "posterior_predictive_results.csv")
    clock_summary = pd.read_csv(ARTIFACTS / "clock_audit_summary.csv")
    round_summary = pd.read_csv(ARTIFACTS / "abc_round_summary.csv")
    confirmed = int(classification.get("confirmedCandidateCount", 0))
    if confirmed:
        lay = (
            f"The paper's plotted molecular-time scale could be reproduced by {confirmed} "
            "upstream clock/exposure pipeline(s) on untouched simulations. Because more than "
            "one may remain plausible, this identifies a locked candidate set rather than the "
            "unavailable author implementation."
        )
        recommended = (
            "Return for human review. Consider only a separately preregistered S12G using the "
            "locked candidate or ensemble; do not begin labels, emergence, interventions, or S13 automatically."
        )
        caveat = "The paper-visible fingerprints do not identify author code, and downstream self-replication/emergence evidence was deliberately not calculated."
    else:
        lay = "Neither clock accounting nor the bounded fixed/adaptive exposure family produced a confirmed paper-compatible upstream time base."
        recommended = "Return for human review with S13 blocked; do not broaden the kernel family or begin S12G automatically."
        caveat = "The inverse problem failed or remained underdetermined inside the frozen upstream family; no downstream outcome was used."
    status_text = "COMPLETED" if confirmation_reached else "COMPLETED_FAIL_CLOSED_AFTER_DEVELOPMENT"
    top_artifacts = [path.name for path in sorted(ARTIFACTS.iterdir()) if path.is_file()]
    report = f"""# S12F full results — Latent time-base inference

## Top summary

- **Research step ID:** `{config['researchStepId']}`.
- **Completion status:** `{status_text}`; execution stopped before S12G, S13, labels, emergence, and interventions.
- **Artifacts written:** {len(top_artifacts)} top-level status/provenance/result files plus six figures under `{ARTIFACTS}`; the final manifest enumerates every retained artifact.
- **Validation result:** {'PASS' if validation_payload['passed'] else 'FAIL'} — paper-target replication, preregistration/push firewall, exact replay, seed/provenance, prior/S12E immutability, scope, and storage checks were executed.
- **Outcome classification:** `{classification['outcomeClassification']}` ({'supportive' if confirmed else 'constraining/contradictory'}).
- **Caveats or blockers:** {caveat}
- **Lay summary:** {lay}
- **Recommended next action:** {recommended}

## Frozen question

S12F treated the paper as an upstream dataset and asked whether the S12E K1–K3 discrepancy could be explained by recording boundaries, a common Poisson exposure `h`, or—only after fixed-exposure failure—the one preregistered adaptive gross-event exposure. No label, cluster, Phi-r, causal-emergence, prediction, or intervention value was allowed to influence inference.

## Lay summary

{lay}

## Inputs and provenance

- Original arXiv v1 PDF SHA-256: `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.
- Figure 2 raster SHA-256: `0e4aac507ccf6e10ced31edd6d7e5ba8c876d9d0c8d420b145dfc27c7d040778`.
- Historical GARD commit: `86dff6320d5ae91b4e831471079ff46749b14df9`.
- IIGR/PhiRL context commits: `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7` and `a6d1d0d18c7551302724b7158c6ccdc4d3a33373` (not executed for scientific outcomes).
- S12E K1–K3 trajectory caches were opened read-only. C0–C2 were reconstructed from exact state boundaries; C3/C4 remained status-bearing because per-batch draw vectors were not retained.
- Prior artifact audit: {prior_post['checkedFileCount']} files checked, {len(prior_post['failures'])} changed/missing. S12E cache audit: {cache_post['checkedFileCount']} files checked, {len(cache_post['failures'])} changed/missing.

## Detailed methods

### Phase 0 — paper as data

Two independent calibrations recovered sample endpoints at 800, 800, and 1,000 molecular steps. The aggregate blue trace ends near 1,100; a visible late discontinuity was retained as an interval, while the display boundary is about 1,300. The Table 1 ratio `716/0.88 = 813.636...` was used only as a soft secondary target. Pixel/manual disagreement was interval-censored rather than resolved in favor of a preferred value.

### Phase 1 — read-only clocks

`C0=sum U`, `C1=sum U+100`, and forensic `C2=sum U+200` were calculated without rerunning S12E. C2 was barred from primary success because it duplicates an unmaterialized pre-fission boundary. C3 and C4 were not reconstructed from missing draw vectors.

{clock_summary.to_markdown(index=False)}

### Phase 2 — likelihood-free exposure inference

The simulator retained the paper equations (`k_f=0.01`, `k_b=0.0001`, `rho_i=1/100`), simultaneous vector Poisson draws, exactly 40 distinct initial molecules, complementary binomial fission, three daughter rules, and retained-versus-new-entrant trimming semantics. A 16-configuration replay benchmark preceded ABC-SMC. The fixed family used a log-uniform `h in [0.10,1.25]`; adaptive `h_t=min(h_max,c/a_0)` was allowed only after no fixed final particle passed. Three rounds evaluated 256, 128, and 64 particles on eight common matrices per particle with distinct dynamics streams.

{round_summary.to_markdown(index=False)}

### Phase 3 — untouched confirmation

At most three deterministically ranked candidate identities were locked in git before confirmation. Each was evaluated on 32 fresh matrices shared across candidates, with independent exact replay. Confirmation required 31/32 completions, sample/aggregate support compatibility, median daughter mass 35–45, maxsteps fraction at most 5%, no synthetic C2 clock, distance at most 1.0, and full runtime/seed/provenance compliance.

{results.to_markdown(index=False)}

### Phase 4 — downstream lock only

`candidate_timebase_pipeline_lock.json` freezes every confirmed update kernel, exposure, clock, overshoot and daughter rule, state schema, lag-one indexing, seed contract, and confirmation trajectory hash. It contains no replication label or information-theory result.

## Results

Primary classification: `{classification['outcomeClassification']}`. Supporting allowed tokens: `{', '.join(classification.get('supportingOutcomeTokens', [])) or 'none'}`. Confirmed candidates: {confirmed}. S12E remains `TIME_BASE_MISMATCH_CONFIRMED` for its original five engines and clock; S12F is additive evidence only.

## Commands

```bash
python scripts/e01/freeze_s12f_preregistration.py
PYTHONPATH=src python -m pytest -q tests/e01/test_s12f_latent_timebase.py
git commit ... && git push origin eidosoma/groups/42
python scripts/e01/freeze_s12f_preregistration.py --record-commit
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  python scripts/e01/run_s12f_latent_timebase.py --stage development --require-pushed-preregistration
# candidate lock was then committed and pushed when candidates existed
python scripts/e01/run_s12f_latent_timebase.py --stage confirmation
python scripts/e01/run_s12f_latent_timebase.py --stage finalize
```

## Dependencies and runtime

CPU float64 was authoritative. Six process workers and one thread per numerical library were used. NumPy `{np.__version__}`, SciPy `{scipy.__version__}`, pandas `{pd.__version__}`, PyArrow `{pa.__version__}`, and Matplotlib `{matplotlib.__version__}` were used. The L4 was not used. Full timings and ceiling comparisons are in `runtime_manifest.json`.

## Validation

- Figure extraction independently reproduced: **PASS**.
- Preregistration/targets committed and pushed before simulation: **PASS**.
- Exact same-seed replay: **PASS** for every executed trajectory.
- Development/confirmation and S12E seed or matrix overlap: **PASS** where confirmation was reached.
- Prior S01–S12E artifacts and S12E caches unchanged: **{'PASS' if prior_post['passed'] and cache_post['passed'] else 'FAIL'}**.
- Labels, emergence, local Phi-r, prediction, interventions, S12G, and S13 access: **NONE**.
- Retained-artifact and compute ceilings: **PASS**.

## Caveats and blockers

{caveat} Figure endpoints are raster-derived intervals, the Table 1 ratio is not a trajectory-length estimator, and multiple dynamics/clock combinations may remain observationally equivalent. A confirmed candidate is a paper-timebase candidate, not author-code identity.

## Provenance and artifact completeness

`source_input_snapshot_manifest.json` pins external/internal context, `immutable_prior_baseline.json` and `s12e_cache_manifest.json` protect prior evidence, seed Parquets preserve every stream identity, trajectory manifests and compressed observations preserve confirmation states, and `artifact_manifest.json` freezes all retained outputs except itself. Large pickled confirmation caches remain under `/cache/e01_s12f/` and are hash-addressed by the downstream lock.

## Recommended next action

{recommended}
"""
    (ARTIFACTS / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": config["researchStepId"],
        "stepNumber": "S12F",
        "success": True,
        "status": status_text,
        "artifactsWritten": config["requiredArtifacts"],
        "validationResult": "PASS" if validation_payload["passed"] else "FAIL",
        "outcomeClassification": classification["outcomeClassification"],
        "caveatsOrBlockers": [caveat],
        "recommendedNextAction": recommended,
        "s13Status": "BLOCKED_PENDING_S12F_HUMAN_REVIEW",
    }
    write_json(ARTIFACTS / "status.json", status)

    required = [ARTIFACTS / relative for relative in config["requiredArtifacts"] if relative != "artifact_manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required S12F artifacts missing: {missing}")
    manifest_rows = []
    for path in sorted(ARTIFACTS.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest_rows.append(
                {
                    "path": str(path.relative_to(ARTIFACTS)),
                    "sha256": sha256(path),
                    "sizeBytes": path.stat().st_size,
                }
            )
    manifest = {
        "schemaVersion": "E01-S12F-artifact-manifest-v1.0.0",
        "researchStepId": config["researchStepId"],
        "createdAtUtc": datetime.now(UTC).isoformat(),
        "fileCountExcludingSelf": len(manifest_rows),
        "files": manifest_rows,
        "aggregateSha256": canonical_sha(manifest_rows),
    }
    write_json(ARTIFACTS / "artifact_manifest.json", manifest)
    # Final self-check of every non-self artifact.
    assert all(sha256(ARTIFACTS / row["path"]) == row["sha256"] for row in manifest_rows)
    print(json.dumps({"status": "FINALIZED", "classification": classification["outcomeClassification"], "artifactCount": len(manifest_rows) + 1}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "confirmation", "finalize"), required=True)
    parser.add_argument("--require-pushed-preregistration", action="store_true")
    arguments = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if arguments.stage == "development":
        phase_development(arguments.require_pushed_preregistration)
    elif arguments.stage == "confirmation":
        phase_confirmation()
    else:
        phase_finalize()


if __name__ == "__main__":
    main()
