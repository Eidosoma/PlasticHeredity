#!/usr/bin/env python3
"""Execute the phase-gated S12FR one-repair comparator confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
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
WORKSPACE = REPO.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml

from e01_latent_timebase.core import derive_seed, observation_rows
from e01_latent_timebase.inference import (
    Particle,
    candidate_groups,
    importance_weights,
    initial_particles,
    particle_summary_and_distance,
    propose_particles,
    retained_particles,
)
from e01_replay_repair.audit import AUDIT_VERSION, sha256_file
from e01_replay_repair.campaign import run_pair_campaign, run_scientific_tasks
from e01_replay_repair.comparator import COMPARATOR_VERSION

STEP_ID = "E01-S12FR-EXACT-REPLAY-COMPARATOR-REPAIR-v1.0.0"
ARTIFACTS = Path("/artifacts/research_steps/S12FR")
CACHE = Path("/cache/e01_s12fr")
TRACE_CACHE = CACHE / "replay_traces"
TRAJECTORY_CACHE = CACHE / "timebase_confirmation"
CONFIG_PATH = REPO / "configs/e01/s12fr_replay_comparator_repair_preregistration.yaml"
CONTRACT_PATH = REPO / "configs/e01/s12fr/comparator_contract.yaml"
SCHEMA_PATH = REPO / "configs/e01/s12fr/pair_diagnostic_schema.json"
S12F_CONFIG_PATH = REPO / "configs/e01/s12f_latent_timebase_preregistration.yaml"
COMPARATOR_LOCK_PATH = REPO / "configs/e01/s12fr/comparator_lock.json"
CANDIDATE_LOCK_PATH = REPO / "configs/e01/s12fr/candidate_lock.json"
EXPECTED_BRANCH = "eidosoma/groups/42"
WORKERS = 6


def sha256(path: Path) -> str:
    return sha256_file(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_s12f_config() -> dict[str, Any]:
    return yaml.safe_load(S12F_CONFIG_PATH.read_text(encoding="utf-8"))


def verify_manifest(path: Path) -> dict[str, Any]:
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
        "fileCount": len(payload["files"]),
        "aggregateSha256": payload["aggregateSha256"],
        "failures": failures,
        "passed": not failures,
    }


def verify_frozen(require_pushed: bool = True) -> dict[str, Any]:
    record = json.loads((ARTIFACTS / "preregistration_record.json").read_text())
    if not record["commitRecordedAfterPush"]:
        raise RuntimeError("S12FR preregistration was not recorded after push")
    amendment = record.get("preOutcomeAmendment")
    runner_hash = (
        amendment["runnerSha256"]
        if amendment is not None
        else record["sourceImplementationHashes"]["runner"]
    )
    if amendment is not None:
        if sha256(Path(amendment["amendmentPath"])) != amendment["amendmentSha256"]:
            raise RuntimeError("S12FR pre-outcome amendment identity changed")
        if not (ARTIFACTS / "preregistration_amendment_v1.0.1.yaml").is_file():
            raise RuntimeError("S12FR amendment artifact is missing")
    expected = {
        CONFIG_PATH: record["configSha256"],
        CONTRACT_PATH: record["contractSha256"],
        SCHEMA_PATH: record["pairSchemaSha256"],
        REPO / "src/e01_replay_repair/comparator.py": record["sourceImplementationHashes"]["comparator"],
        REPO / "src/e01_replay_repair/audit.py": record["sourceImplementationHashes"]["audit"],
        REPO / "src/e01_replay_repair/campaign.py": record["sourceImplementationHashes"]["campaign"],
        REPO / "scripts/e01/run_s12fr_replay_repair.py": runner_hash,
    }
    changed = [str(path) for path, digest in expected.items() if sha256(path) != digest]
    if changed:
        raise RuntimeError(f"frozen S12FR implementation changed: {changed}")
    if sha256(Path("/artifacts/research_steps/S12F/abc_particle_results.parquet")) != next(
        row["sha256"]
        for row in json.loads((ARTIFACTS / "s12f_suppressed_input_manifest.json").read_text())["files"]
        if row["path"].endswith("abc_particle_results.parquet")
    ):
        raise RuntimeError("S12F suppressed particle artifact changed")
    prior = verify_manifest(ARTIFACTS / "immutable_prior_baseline.json")
    s12f_cache = verify_manifest(ARTIFACTS / "s12f_cache_baseline.json")
    if not prior["passed"] or not s12f_cache["passed"]:
        raise RuntimeError("S01-S12F artifact or S12F cache identity changed")
    if require_pushed:
        if git("branch", "--show-current") != EXPECTED_BRANCH:
            raise RuntimeError("wrong git branch")
        head = git("rev-parse", "HEAD^{commit}")
        remote = git("rev-parse", f"origin/{EXPECTED_BRANCH}^{{commit}}")
        if head != remote:
            raise RuntimeError("current S12FR lock state is not pushed")
        subprocess.check_call(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor", record["gitCommit"], head]
        )
    return {"record": record, "prior": prior, "s12fCache": s12f_cache}


def particle_identity_audit(particles: list[Particle]) -> dict[str, Any]:
    frozen = pd.read_parquet(
        "/artifacts/research_steps/S12F/abc_particle_results.parquet",
        columns=[
            "particleId",
            "family",
            "round",
            "daughterRule",
            "overshootRule",
            "clockId",
            "h",
            "c",
            "hMax",
            "parentParticleId",
        ],
    ).sort_values("particleId").reset_index(drop=True)
    regenerated = pd.DataFrame(
        [
            {
                "particleId": row.particle_id,
                "family": row.family,
                "round": row.round_index,
                "daughterRule": row.daughter_rule,
                "overshootRule": row.overshoot_rule,
                "clockId": row.clock_id,
                "h": row.h,
                "c": row.c,
                "hMax": row.h_max,
                "parentParticleId": row.parent_particle_id,
            }
            for row in particles
        ]
    ).sort_values("particleId").reset_index(drop=True)
    exact = bool(
        list(frozen.columns) == list(regenerated.columns)
        and frozen.shape == regenerated.shape
        and all(frozen[column].equals(regenerated[column]) for column in frozen.columns)
    )
    return {
        "rowCount": int(frozen.shape[0]),
        "columnsAccessed": list(frozen.columns),
        "distanceOrPosteriorColumnAccessed": False,
        "identityExact": exact,
    }


def pair_tasks(
    *,
    campaign: str,
    phase: str,
    root: str,
    particles: list[Particle],
    trace_subdir: str,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for particle in particles:
        for matrix_index in range(8):
            pair_id = f"{campaign}__{particle.particle_id}__M{matrix_index:02d}"
            tasks.append(
                {
                    "campaign": campaign,
                    "pairId": pair_id,
                    "phase": phase,
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
                    "tracePath": str(
                        TRACE_CACHE
                        / trace_subdir
                        / particle.particle_id
                        / f"M{matrix_index:02d}.npz"
                    ),
                }
            )
    return tasks


def benchmark_tasks() -> list[dict[str, Any]]:
    config = load_s12f_config()
    root = config["randomness"]["roots"]["benchmark"]
    tasks = []
    for index, (daughter, overshoot, h) in enumerate(config["benchmark"]["configurations"]):
        identifier = f"BENCH-{index:02d}-{daughter}-{overshoot}-h={float(h):.17g}"
        tasks.append(
            {
                "campaign": "BENCHMARK_16",
                "pairId": f"BENCHMARK_16__{identifier}",
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
                "tracePath": str(TRACE_CACHE / "benchmark" / f"{index:02d}.npz"),
            }
        )
    return tasks


def campaign_summary(
    frame: pd.DataFrame, differences: pd.DataFrame, expected: int, campaign: str
) -> dict[str, Any]:
    old_failures = int((~frame["oldComparatorPassed"]).sum())
    gate_passes = int(frame["pairGatePassed"].sum())
    unexplained = int((~frame["oldFailureFullyExplained"]).sum())
    scoped = differences[differences["comparisonScope"] == "LEFT_VS_REPLAY"]
    summary = {
        "schemaVersion": "E01-S12FR-pair-campaign-summary-v1.0.0",
        "researchStepId": STEP_ID,
        "campaign": campaign,
        "expectedPairCount": expected,
        "observedPairCount": int(frame.shape[0]),
        "oldComparatorFailureCount": old_failures,
        "repairedComparatorPassCount": int(frame["repairedComparatorPassed"].sum()),
        "pairGatePassCount": gate_passes,
        "unexplainedOldFailureCount": unexplained,
        "permittedPairedNanCount": int(frame["permittedPairedNanCount"].sum()),
        "discreteDivergenceCount": int(frame["discreteDivergenceCount"].sum()),
        "finiteNumericDivergenceCount": int(frame["finiteNumericDivergenceCount"].sum()),
        "forbiddenNonfiniteDifferenceCount": int(frame["forbiddenNonfiniteDifferenceCount"].sum()),
        "rngDivergenceCount": int(frame["rngDivergenceCount"].sum()),
        "instrumentationParityFailureCount": int((~frame["instrumentationParityPassed"]).sum()),
        "traceDigestFailureCount": int((~frame["traceDigestPassed"]).sum()),
        "fieldDifferenceRowCount": int(differences.shape[0]),
        "leftVsReplayDifferenceRowCount": int(scoped.shape[0]),
        "leftVsReplayDifferenceCategories": scoped["category"].value_counts().sort_index().to_dict(),
        "allPairsPassed": bool(frame.shape[0] == expected and gate_passes == expected),
        "everyOldFailureFullyExplained": bool(old_failures > 0 and unexplained == 0),
    }
    summary["campaignGatePassed"] = bool(
        summary["allPairsPassed"]
        and summary["everyOldFailureFullyExplained"]
        and summary["discreteDivergenceCount"] == 0
        and summary["finiteNumericDivergenceCount"] == 0
        and summary["forbiddenNonfiniteDifferenceCount"] == 0
        and summary["rngDivergenceCount"] == 0
        and summary["instrumentationParityFailureCount"] == 0
    )
    return summary


def phase_diagnose_original() -> None:
    frozen = verify_frozen()
    config = load_config()
    s12f = load_s12f_config()
    particles = initial_particles(
        "FIXED_COMMON_EXPOSURE", s12f["randomness"]["roots"]["inference"], 256
    )
    identity_audit = particle_identity_audit(particles)
    if not identity_audit["identityExact"]:
        raise RuntimeError("original S12F round-1 particle identities changed")
    tasks = pair_tasks(
        campaign="ORIGINAL_S12F_ROUND1",
        phase="development",
        root=s12f["randomness"]["roots"]["development"],
        particles=particles,
        trace_subdir="original",
    )
    started = time.perf_counter()
    pairs, differences, traces, seeds, wall = run_pair_campaign(tasks, WORKERS)
    pairs.to_parquet(ARTIFACTS / "original_pair_diagnostics.parquet", index=False, compression="zstd")
    differences.to_parquet(ARTIFACTS / "original_field_differences.parquet", index=False, compression="zstd")
    traces.to_parquet(ARTIFACTS / "original_trace_manifest.parquet", index=False, compression="zstd")
    seeds.to_parquet(ARTIFACTS / "original_seed_manifest.parquet", index=False, compression="zstd")
    summary = campaign_summary(pairs, differences, 2048, "ORIGINAL_S12F_ROUND1")
    summary.update(
        {
            "particleIdentityAudit": identity_audit,
            "workerCpuSeconds": float(pairs["workerCpuSeconds"].sum()),
            "campaignWallSeconds": wall,
            "orchestrationWallSeconds": time.perf_counter() - started,
        }
    )
    write_json(ARTIFACTS / "original_pair_summary.json", summary)
    lock_payload = {
        "schemaVersion": "E01-S12FR-comparator-lock-v1.0.0",
        "researchStepId": STEP_ID,
        "comparatorVersion": COMPARATOR_VERSION,
        "auditVersion": AUDIT_VERSION,
        "designCommit": frozen["record"]["gitCommit"],
        "configSha256": frozen["record"]["configSha256"],
        "contractSha256": frozen["record"]["contractSha256"],
        "pairSchemaSha256": frozen["record"]["pairSchemaSha256"],
        "implementationHashes": frozen["record"]["sourceImplementationHashes"],
        "permittedNormalization": "PAIRED_SCHEMA_UNDEFINED_NAN_ZERO_UPDATE",
        "originalPairCount": 2048,
        "originalPairDiagnosticsSha256": sha256(ARTIFACTS / "original_pair_diagnostics.parquet"),
        "originalFieldDifferencesSha256": sha256(ARTIFACTS / "original_field_differences.parquet"),
        "originalTraceManifestSha256": sha256(ARTIFACTS / "original_trace_manifest.parquet"),
        "originalPairSummarySha256": sha256(ARTIFACTS / "original_pair_summary.json"),
        "originalCampaignGatePassed": summary["campaignGatePassed"],
        "untouchedInferenceRoot": config["untouchedConfirmationCampaign"]["inferenceRoot"],
        "untouchedSimulatorRoot": config["untouchedConfirmationCampaign"]["simulatorRoot"],
        "oneRepairRule": "ANY_CONFIRMATION_FAILURE_PERMANENTLY_CLOSES_PATH",
    }
    proposal = {
        "schemaVersion": "E01-S12FR-comparator-lock-proposal-v1.0.0",
        "researchStepId": STEP_ID,
        "eligibleToLock": bool(summary["campaignGatePassed"]),
        "reason": "ALL_OLD_FAILURES_REPRESENTATIONAL_ONLY"
        if summary["campaignGatePassed"]
        else "GENUINE_OR_UNEXPLAINED_DIVERGENCE",
        "lockPayload": lock_payload,
    }
    write_json(ARTIFACTS / "comparator_lock_proposal.json", proposal)
    write_json(
        ARTIFACTS / "comparator_lock_validation.json",
        {
            "schemaVersion": "E01-S12FR-comparator-lock-validation-v1.0.0",
            "researchStepId": STEP_ID,
            "status": "AWAITING_COMMITTED_PUSHED_LOCK"
            if proposal["eligibleToLock"]
            else "NOT_ELIGIBLE_PERMANENT_STOP",
            "passed": False,
        },
    )
    write_json(
        CACHE / "diagnosis_runtime.json",
        {
            "startedAtUtc": datetime.now(UTC).isoformat(),
            "campaignWallSeconds": wall,
            "workerCpuSeconds": float(pairs["workerCpuSeconds"].sum()),
            "pairCount": int(pairs.shape[0]),
        },
    )
    if not summary["campaignGatePassed"]:
        raise RuntimeError("S12FR original-pair comparator diagnosis failed permanently")


def verify_comparator_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = verify_frozen()
    proposal = json.loads((ARTIFACTS / "comparator_lock_proposal.json").read_text())
    if not proposal["eligibleToLock"]:
        raise RuntimeError("comparator is not eligible to lock")
    if not COMPARATOR_LOCK_PATH.is_file():
        raise RuntimeError("comparator lock has not been committed")
    lock = json.loads(COMPARATOR_LOCK_PATH.read_text())
    if lock != proposal["lockPayload"]:
        raise RuntimeError("comparator lock differs from deterministic proposal")
    head = git("rev-parse", "HEAD^{commit}")
    remote = git("rev-parse", f"origin/{EXPECTED_BRANCH}^{{commit}}")
    if head != remote:
        raise RuntimeError("comparator lock commit is not pushed")
    subprocess.check_call(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", frozen["record"]["gitCommit"], head]
    )
    validation = {
        "schemaVersion": "E01-S12FR-comparator-lock-validation-v1.0.0",
        "researchStepId": STEP_ID,
        "status": "LOCKED_COMMITTED_PUSHED",
        "passed": True,
        "lockPath": str(COMPARATOR_LOCK_PATH),
        "lockSha256": sha256(COMPARATOR_LOCK_PATH),
        "lockCommit": head,
        "remoteCommit": remote,
        "implementationHashesExact": True,
    }
    write_json(ARTIFACTS / "comparator_lock_validation.json", validation)
    return lock, validation


def phase_confirm_repair() -> None:
    _lock, lock_validation = verify_comparator_lock()
    config = load_config()
    benchmark, benchmark_differences, benchmark_traces, _benchmark_seeds, benchmark_wall = run_pair_campaign(
        benchmark_tasks(), WORKERS
    )
    benchmark.to_csv(ARTIFACTS / "benchmark_pair_diagnostics.csv", index=False)
    benchmark_differences.to_parquet(
        ARTIFACTS / "benchmark_field_differences.parquet", index=False, compression="zstd"
    )
    benchmark_traces.to_parquet(
        ARTIFACTS / "benchmark_trace_manifest.parquet", index=False, compression="zstd"
    )
    benchmark_summary = campaign_summary(
        benchmark, benchmark_differences, 16, "BENCHMARK_16"
    )
    benchmark_summary["campaignGatePassed"] = bool(
        benchmark_summary["allPairsPassed"]
        and benchmark_summary["discreteDivergenceCount"] == 0
        and benchmark_summary["finiteNumericDivergenceCount"] == 0
        and benchmark_summary["forbiddenNonfiniteDifferenceCount"] == 0
        and benchmark_summary["rngDivergenceCount"] == 0
    )

    particles = initial_particles(
        "FIXED_COMMON_EXPOSURE",
        config["untouchedConfirmationCampaign"]["inferenceRoot"],
        256,
    )
    tasks = pair_tasks(
        campaign="UNTOUCHED_REPAIR_CONFIRMATION",
        phase=config["untouchedConfirmationCampaign"]["phase"],
        root=config["untouchedConfirmationCampaign"]["simulatorRoot"],
        particles=particles,
        trace_subdir="untouched",
    )
    untouched, differences, traces, seeds, untouched_wall = run_pair_campaign(tasks, WORKERS)
    untouched.to_parquet(ARTIFACTS / "untouched_pair_diagnostics.parquet", index=False, compression="zstd")
    differences.to_parquet(ARTIFACTS / "untouched_field_differences.parquet", index=False, compression="zstd")
    traces.to_parquet(ARTIFACTS / "untouched_trace_manifest.parquet", index=False, compression="zstd")
    seeds.to_parquet(ARTIFACTS / "untouched_seed_manifest.parquet", index=False, compression="zstd")
    untouched_summary = campaign_summary(
        untouched, differences, 2048, "UNTOUCHED_REPAIR_CONFIRMATION"
    )
    # Confirmation need not reproduce an old failure; its gate is unanimous exact repaired replay.
    untouched_summary["campaignGatePassed"] = bool(
        untouched_summary["allPairsPassed"]
        and untouched_summary["discreteDivergenceCount"] == 0
        and untouched_summary["finiteNumericDivergenceCount"] == 0
        and untouched_summary["forbiddenNonfiniteDifferenceCount"] == 0
        and untouched_summary["rngDivergenceCount"] == 0
        and untouched_summary["instrumentationParityFailureCount"] == 0
    )
    untouched_summary.update(
        {
            "workerCpuSeconds": float(untouched["workerCpuSeconds"].sum()),
            "campaignWallSeconds": untouched_wall,
        }
    )
    write_json(ARTIFACTS / "untouched_pair_summary.json", untouched_summary)

    original = pd.read_parquet(ARTIFACTS / "original_pair_diagnostics.parquet")
    original_seeds = pd.read_parquet(ARTIFACTS / "original_seed_manifest.parquet")
    s12f_seeds = pd.read_parquet(
        "/artifacts/research_steps/S12F/development_seed_manifest.parquet",
        columns=["seedMaterialSha256"],
    )
    original_material = set(original_seeds["seedMaterialSha256"].astype(str))
    untouched_material = set(seeds["seedMaterialSha256"].astype(str))
    s12f_material = set(s12f_seeds["seedMaterialSha256"].astype(str))
    inference_identity = derive_seed(
        config["untouchedConfirmationCampaign"]["inferenceRoot"],
        "abc_inference",
        "FIXED_COMMON_EXPOSURE_prior",
        1,
    )
    inference_row = asdict_seed(inference_identity, "UNTOUCHED_REPAIR_CONFIRMATION_INFERENCE")
    write_json(ARTIFACTS / "untouched_inference_seed_identity.json", inference_row)
    firewall = {
        "schemaVersion": "E01-S12FR-seed-matrix-firewall-v1.0.0",
        "researchStepId": STEP_ID,
        "originalSeedCount": len(original_material),
        "untouchedSeedCount": len(untouched_material),
        "originalUntouchedSeedIntersectionCount": len(original_material & untouched_material),
        "s12fUntouchedSeedIntersectionCount": len(s12f_material & untouched_material),
        "originalUntouchedBetaHashIntersectionCount": len(
            set(original["betaSha256Left"]) & set(untouched["betaSha256Left"])
        ),
        "originalUntouchedInitialStateHashIntersectionCount": len(
            set(original["initialStateSha256Left"])
            & set(untouched["initialStateSha256Left"])
        ),
        "untouchedUniqueBetaCount": int(untouched["betaSha256Left"].nunique()),
        "untouchedUniqueInitialStateCount": int(
            untouched["initialStateSha256Left"].nunique()
        ),
    }
    firewall["passed"] = bool(
        firewall["originalUntouchedSeedIntersectionCount"] == 0
        and firewall["s12fUntouchedSeedIntersectionCount"] == 0
        and firewall["originalUntouchedBetaHashIntersectionCount"] == 0
        and firewall["originalUntouchedInitialStateHashIntersectionCount"] == 0
        and firewall["untouchedUniqueBetaCount"] == 8
        and firewall["untouchedUniqueInitialStateCount"] == 8
    )
    write_json(ARTIFACTS / "seed_matrix_firewall.json", firewall)
    passed = bool(
        lock_validation["passed"]
        and benchmark_summary["campaignGatePassed"]
        and untouched_summary["campaignGatePassed"]
        and firewall["passed"]
    )
    validation = {
        "schemaVersion": "E01-S12FR-repair-confirmation-validation-v1.0.0",
        "researchStepId": STEP_ID,
        "passed": passed,
        "comparatorLock": lock_validation,
        "benchmark": benchmark_summary,
        "untouchedConfirmation": untouched_summary,
        "seedMatrixFirewall": firewall,
        "benchmarkWallSeconds": benchmark_wall,
        "untouchedWallSeconds": untouched_wall,
        "oneRepairRuleStatus": "REPAIR_CONFIRMED"
        if passed
        else "PERMANENTLY_CLOSED_AFTER_REPAIR_CONFIRMATION_FAILURE",
    }
    write_json(ARTIFACTS / "repair_confirmation_validation.json", validation)
    write_json(
        CACHE / "repair_confirmation_runtime.json",
        {
            "benchmarkWallSeconds": benchmark_wall,
            "untouchedWallSeconds": untouched_wall,
            "benchmarkWorkerCpuSeconds": float(benchmark["workerCpuSeconds"].sum()),
            "untouchedWorkerCpuSeconds": float(untouched["workerCpuSeconds"].sum()),
        },
    )
    if not passed:
        raise RuntimeError("S12FR untouched comparator confirmation failed permanently")


def asdict_seed(seed: Any, campaign: str) -> dict[str, Any]:
    return {
        "campaign": campaign,
        "phase": seed.phase,
        "rootSha256": seed.root_sha256,
        "purpose": seed.purpose,
        "matrixIndex": seed.matrix_index,
        "configurationId": seed.configuration_id,
        "extra": list(seed.extra),
        "derivedSeed": str(seed.derived_seed),
        "seedMaterialSha256": seed.seed_material_sha256,
        "bitGenerator": "PCG64DXSM",
    }


def scientific_particle_tasks(
    particles: list[Particle], root: str, phase: str, campaign: str
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for particle in particles:
        for matrix_index in range(8):
            tasks.append(
                {
                    "campaign": campaign,
                    "pairId": f"{campaign}__{particle.particle_id}__M{matrix_index:02d}",
                    "phase": phase,
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


def evaluate_scientific_round(
    particles: list[Particle], root: str, family: str, round_index: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    summaries, seeds, _trajectories, wall = run_scientific_tasks(
        scientific_particle_tasks(
            particles,
            root,
            "development",
            f"FRESH_ABC_{family}_R{round_index}",
        ),
        WORKERS,
    )
    if summaries.shape[0] != len(particles) * 8:
        raise RuntimeError("fresh ABC trajectory cardinality failed")
    if not bool(summaries["repairedReplayPassed"].all()):
        raise RuntimeError("fresh ABC exact repaired replay failed")
    result_rows = []
    for particle in particles:
        group = summaries[summaries["particleId"] == particle.particle_id]
        if group.shape[0] != 8:
            raise RuntimeError("fresh ABC particle did not receive eight matrices")
        result_rows.append(particle_summary_and_distance(particle, group))
    return pd.DataFrame(result_rows), seeds, summaries, wall


def run_abc_family(
    family: str, s12f: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Particle], np.ndarray, dict[str, Any]]:
    inference_root = s12f["randomness"]["roots"]["inference"]
    development_root = s12f["randomness"]["roots"]["development"]
    all_results: list[pd.DataFrame] = []
    all_seeds: list[pd.DataFrame] = []
    round_rows: list[dict[str, Any]] = []
    total_cpu = 0.0

    round1 = initial_particles(family, inference_root, 256)
    if family == "FIXED_COMMON_EXPOSURE":
        audit = particle_identity_audit(round1)
        if not audit["identityExact"]:
            raise RuntimeError("fresh ABC round-1 particles differ from frozen S12F")
    result1, seeds1, trajectories1, wall1 = evaluate_scientific_round(
        round1, development_root, family, 1
    )
    all_results.append(result1)
    all_seeds.append(seeds1)
    total_cpu += float(trajectories1["workerCpuSeconds"].sum())
    parents1, _ = retained_particles(round1, result1, 128)
    weights1_all = importance_weights(round1, None, None, None)
    lookup1 = {
        particle.particle_id: weight
        for particle, weight in zip(round1, weights1_all, strict=True)
    }
    weights1 = np.asarray([lookup1[row.particle_id] for row in parents1], dtype=float)
    weights1 /= weights1.sum()
    round_rows.append(
        {
            "family": family,
            "round": 1,
            "particlesEvaluated": 256,
            "particlesRetained": 128,
            "trajectoryPairs": 2048,
            "epsilonMedian": float(result1["distance"].median()),
            "minimumDistance": float(result1["distance"].min()),
            "envelopePassCount": int(result1["developmentAcceptanceEnvelopePassed"].sum()),
            "wallSeconds": wall1,
        }
    )

    round2 = propose_particles(
        family, inference_root, 2, 128, parents1, weights1, 0.20
    )
    result2, seeds2, trajectories2, wall2 = evaluate_scientific_round(
        round2, development_root, family, 2
    )
    all_results.append(result2)
    all_seeds.append(seeds2)
    total_cpu += float(trajectories2["workerCpuSeconds"].sum())
    weights2_all = importance_weights(round2, parents1, weights1, 0.20)
    parents2, _ = retained_particles(round2, result2, 64)
    lookup2 = {
        particle.particle_id: weight
        for particle, weight in zip(round2, weights2_all, strict=True)
    }
    weights2 = np.asarray([lookup2[row.particle_id] for row in parents2], dtype=float)
    weights2 /= weights2.sum()
    round_rows.append(
        {
            "family": family,
            "round": 2,
            "particlesEvaluated": 128,
            "particlesRetained": 64,
            "trajectoryPairs": 1024,
            "epsilonMedian": float(result2["distance"].median()),
            "minimumDistance": float(result2["distance"].min()),
            "envelopePassCount": int(result2["developmentAcceptanceEnvelopePassed"].sum()),
            "wallSeconds": wall2,
        }
    )

    round3 = propose_particles(
        family, inference_root, 3, 64, parents2, weights2, 0.10
    )
    result3, seeds3, trajectories3, wall3 = evaluate_scientific_round(
        round3, development_root, family, 3
    )
    weights3 = importance_weights(round3, parents2, weights2, 0.10)
    result3["posteriorWeight"] = [float(value) for value in weights3]
    all_results.append(result3)
    all_seeds.append(seeds3)
    total_cpu += float(trajectories3["workerCpuSeconds"].sum())
    round_rows.append(
        {
            "family": family,
            "round": 3,
            "particlesEvaluated": 64,
            "particlesRetained": 64,
            "trajectoryPairs": 512,
            "epsilonMedian": float(result3["distance"].median()),
            "minimumDistance": float(result3["distance"].min()),
            "envelopePassCount": int(result3["developmentAcceptanceEnvelopePassed"].sum()),
            "wallSeconds": wall3,
        }
    )
    metadata = {
        "family": family,
        "workerCpuSeconds": total_cpu,
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


def phase_resume_abc() -> None:
    verify_frozen()
    repair = json.loads((ARTIFACTS / "repair_confirmation_validation.json").read_text())
    if not repair["passed"]:
        raise RuntimeError("conditional ABC resume is blocked by repair confirmation")
    s12f = load_s12f_config()
    started = time.perf_counter()
    fixed_all, fixed_seeds, fixed_rounds, fixed_final_particles, fixed_weights, fixed_meta = run_abc_family(
        "FIXED_COMMON_EXPOSURE", s12f
    )
    fixed_final = fixed_all[fixed_all["round_index"] == 3].copy()
    fixed_accepted = int(fixed_final["developmentAcceptanceEnvelopePassed"].sum())
    all_results = [fixed_all]
    all_seeds = [fixed_seeds]
    all_rounds = [fixed_rounds]
    metadata = [fixed_meta]
    final_particles = fixed_final_particles
    final_weights = fixed_weights
    final_frame = fixed_final
    adaptive_executed = False
    if fixed_accepted == 0:
        adaptive_executed = True
        adaptive_all, adaptive_seeds, adaptive_rounds, adaptive_final_particles, adaptive_weights, adaptive_meta = run_abc_family(
            "ADAPTIVE_GROSS_EVENT_EXPOSURE", s12f
        )
        all_results.append(adaptive_all)
        all_seeds.append(adaptive_seeds)
        all_rounds.append(adaptive_rounds)
        metadata.append(adaptive_meta)
        final_particles = adaptive_final_particles
        final_weights = adaptive_weights
        final_frame = adaptive_all[adaptive_all["round_index"] == 3].copy()

    result_frame = pd.concat(all_results, ignore_index=True)
    seed_frame = pd.concat(all_seeds, ignore_index=True).drop_duplicates(
        "seedMaterialSha256"
    )
    round_frame = pd.concat(all_rounds, ignore_index=True)
    candidates, selected = candidate_groups(
        final_particles, final_frame, final_weights, maximum=3
    )
    result_frame.to_parquet(
        ARTIFACTS / "abc_particle_results.parquet", index=False, compression="zstd"
    )
    seed_frame.to_parquet(
        ARTIFACTS / "abc_seed_manifest.parquet", index=False, compression="zstd"
    )
    round_frame.to_csv(ARTIFACTS / "abc_round_summary.csv", index=False)
    candidates.to_csv(ARTIFACTS / "posterior_candidates.csv", index=False)

    selected_rows = []
    for rank, particle in enumerate(selected, start=1):
        source = candidates[
            candidates["representativeParticleId"] == particle.particle_id
        ].iloc[0]
        selected_rows.append(
            {
                "candidateId": f"S12F-CANDIDATE-{rank:02d}",
                "confirmationRank": rank,
                "representativeParticleId": particle.particle_id,
                "candidateGroup": particle.discrete_group,
                "posteriorMass": float(source["posteriorMass"]),
                "developmentDistance": float(source["minimumDistance"]),
                "exposureFamily": particle.family,
                "h": particle.h,
                "c": particle.c,
                "hMax": particle.h_max,
                "daughterRule": particle.daughter_rule,
                "overshootRule": particle.overshoot_rule,
                "clockId": particle.clock_id,
            }
        )
    lock_payload = {
        "schemaVersion": "E01-S12FR-timebase-candidate-lock-v1.0.0",
        "researchStepId": STEP_ID,
        "repairConfirmationSha256": sha256(
            ARTIFACTS / "repair_confirmation_validation.json"
        ),
        "abcParticleResultsSha256": sha256(ARTIFACTS / "abc_particle_results.parquet"),
        "posteriorCandidatesSha256": sha256(ARTIFACTS / "posterior_candidates.csv"),
        "confirmationRoot": s12f["randomness"]["roots"]["confirmation"],
        "candidateCount": len(selected_rows),
        "candidates": selected_rows,
        "selectionRule": "unchanged_S12F_posterior_mass_then_distance_complexity_lexical_maximum_three",
    }
    proposal = {
        "schemaVersion": "E01-S12FR-timebase-candidate-lock-proposal-v1.0.0",
        "researchStepId": STEP_ID,
        "candidateCount": len(selected_rows),
        "requiresCommittedPushedLock": bool(selected_rows),
        "lockPayload": lock_payload,
        "fixedFinalEnvelopePassCount": fixed_accepted,
        "adaptiveExecuted": adaptive_executed,
        "adaptiveFinalEnvelopePassCount": int(
            final_frame["developmentAcceptanceEnvelopePassed"].sum()
        )
        if adaptive_executed
        else None,
    }
    write_json(ARTIFACTS / "candidate_lock_proposal.json", proposal)
    write_json(
        ARTIFACTS / "candidate_lock_validation.json",
        {
            "schemaVersion": "E01-S12FR-candidate-lock-validation-v1.0.0",
            "researchStepId": STEP_ID,
            "passed": not bool(selected_rows),
            "status": "NOT_APPLICABLE_NO_CANDIDATES"
            if not selected_rows
            else "AWAITING_COMMITTED_PUSHED_LOCK",
        },
    )
    write_json(
        CACHE / "abc_runtime.json",
        {
            "completedAtUtc": datetime.now(UTC).isoformat(),
            "orchestrationWallSeconds": time.perf_counter() - started,
            "familyMetadata": metadata,
            "fixedAccepted": fixed_accepted,
            "adaptiveExecuted": adaptive_executed,
            "candidateCount": len(selected_rows),
        },
    )


def verify_candidate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    verify_frozen()
    proposal = json.loads((ARTIFACTS / "candidate_lock_proposal.json").read_text())
    if proposal["candidateCount"] == 0:
        raise RuntimeError("no candidate confirmation is authorized")
    if not CANDIDATE_LOCK_PATH.is_file():
        raise RuntimeError("timebase candidate lock has not been committed")
    lock = json.loads(CANDIDATE_LOCK_PATH.read_text())
    if lock != proposal["lockPayload"]:
        raise RuntimeError("timebase candidate lock differs from deterministic proposal")
    head = git("rev-parse", "HEAD^{commit}")
    remote = git("rev-parse", f"origin/{EXPECTED_BRANCH}^{{commit}}")
    if head != remote:
        raise RuntimeError("candidate lock commit is not pushed")
    validation = {
        "schemaVersion": "E01-S12FR-candidate-lock-validation-v1.0.0",
        "researchStepId": STEP_ID,
        "passed": True,
        "status": "LOCKED_COMMITTED_PUSHED",
        "lockPath": str(CANDIDATE_LOCK_PATH),
        "lockSha256": sha256(CANDIDATE_LOCK_PATH),
        "lockCommit": head,
        "remoteCommit": remote,
        "candidateCount": lock["candidateCount"],
    }
    write_json(ARTIFACTS / "candidate_lock_validation.json", validation)
    return lock, validation


def timebase_confirmation_tasks(lock: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    for candidate in lock["candidates"]:
        for matrix_index in range(32):
            tasks.append(
                {
                    "campaign": "TIMEBASE_32_MATRIX_CONFIRMATION",
                    "pairId": f"TIMEBASE__{candidate['candidateId']}__M{matrix_index:02d}",
                    "phase": "confirmation",
                    "root": lock["confirmationRoot"],
                    "matrixIndex": matrix_index,
                    "streamIdentity": candidate["candidateId"],
                    "particleId": candidate["representativeParticleId"],
                    "candidateId": candidate["candidateId"],
                    "family": candidate["exposureFamily"],
                    "h": candidate["h"],
                    "c": candidate["c"],
                    "hMax": candidate["hMax"],
                    "daughterRule": candidate["daughterRule"],
                    "overshootRule": candidate["overshootRule"],
                    "clockId": candidate["clockId"],
                    "returnTrajectory": True,
                    "cachePath": str(
                        TRAJECTORY_CACHE
                        / candidate["candidateId"]
                        / f"M{matrix_index:02d}.pickle"
                    ),
                }
            )
    return tasks


def candidate_confirmation_result(
    candidate: dict[str, Any], group: pd.DataFrame
) -> dict[str, Any]:
    column = {
        "C0_BATCH_UPDATES_ONLY": "clockC0",
        "C1_SELECTED_DAUGHTER_RETAINED": "clockC1",
        "C2_EXPLICIT_PRE_AND_POST_FISSION": "clockC2",
    }[candidate["clockId"]]
    values = group[column].to_numpy(float)
    q05, q50, q95 = np.quantile(values, [0.05, 0.50, 0.95])
    synthetic = Particle(
        particle_id=candidate["representativeParticleId"],
        family=candidate["exposureFamily"],
        round_index=3,
        daughter_rule=candidate["daughterRule"],
        overshoot_rule=candidate["overshootRule"],
        clock_id=candidate["clockId"],
        h=candidate["h"],
        c=candidate["c"],
        h_max=candidate["hMax"],
        parent_particle_id=None,
        proposal_weight=candidate["posteriorMass"],
    )
    distance = particle_summary_and_distance(synthetic, group)
    completed = int((group["completedFissions"] == 100).sum())
    maxsteps_denominator = max(1, int(group["completedFissions"].sum()))
    maxsteps_fraction = float(group["maxstepsTerminations"].sum() / maxsteps_denominator)
    endpoints_inside = int(sum(q05 <= value <= q95 for value in (800.0, 800.0, 1000.0)))
    aggregate = bool(
        float(values.max()) >= 1090.0
        and q95 <= 1314.0
        and float(np.mean(values > 1314.0)) <= 0.05
    )
    median_post = float(group["medianPostFissionMass"].median())
    passed = bool(
        completed >= 31
        and endpoints_inside >= 2
        and aggregate
        and 35.0 <= median_post <= 45.0
        and maxsteps_fraction <= 0.05
        and candidate["clockId"] != "C2_EXPLICIT_PRE_AND_POST_FISSION"
        and float(distance["distance"]) <= 1.0
        and bool(group["repairedReplayPassed"].all())
    )
    reasons = []
    if completed < 31:
        reasons.append("fewer_than_31_of_32_complete")
    if endpoints_inside < 2:
        reasons.append("sample_endpoints_incompatible")
    if not aggregate:
        reasons.append("aggregate_support_incompatible")
    if not 35.0 <= median_post <= 45.0:
        reasons.append("post_fission_mass_outside_interval")
    if maxsteps_fraction > 0.05:
        reasons.append("maxsteps_fraction_above_0p05")
    if candidate["clockId"] == "C2_EXPLICIT_PRE_AND_POST_FISSION":
        reasons.append("synthetic_C2_forbidden")
    if float(distance["distance"]) > 1.0:
        reasons.append("confirmation_distance_outside_ABC_envelope")
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
        "completedLineages": completed,
        "q05TPhi": float(q05),
        "medianTPhi": float(q50),
        "q95TPhi": float(q95),
        "maximumTPhi": float(values.max()),
        "sampleEndpointsInsideQ05Q95": endpoints_inside,
        "aggregateCompatible": aggregate,
        "medianPostFissionMass": median_post,
        "q95Overshoot": float(group["q95Overshoot"].quantile(0.95)),
        "fractionMaxsteps": maxsteps_fraction,
        "confirmationDistance": float(distance["distance"]),
        "confirmationGatePassed": passed,
        "gateReason": "PASS" if passed else ";".join(reasons),
    }


def phase_confirm_timebase() -> None:
    lock, lock_validation = verify_candidate_lock()
    started = time.perf_counter()
    summaries, seeds, trajectories, wall = run_scientific_tasks(
        timebase_confirmation_tasks(lock), WORKERS
    )
    expected = 32 * lock["candidateCount"]
    if summaries.shape[0] != expected:
        raise RuntimeError("timebase confirmation cardinality failed")
    if not bool(summaries["repairedReplayPassed"].all()):
        raise RuntimeError("timebase confirmation replay failed")
    development_seeds = pd.read_parquet(
        ARTIFACTS / "abc_seed_manifest.parquet",
        columns=["seedMaterialSha256"],
    )
    development_material = set(development_seeds["seedMaterialSha256"].astype(str))
    confirmation_material = set(seeds["seedMaterialSha256"].astype(str))
    if development_material & confirmation_material:
        raise RuntimeError("ABC development and candidate confirmation seeds overlap")
    seeds.to_parquet(
        ARTIFACTS / "confirmation_seed_manifest.parquet", index=False, compression="zstd"
    )
    summaries.to_parquet(
        ARTIFACTS / "confirmation_trajectory_manifest.parquet",
        index=False,
        compression="zstd",
    )
    observation_frame = pd.DataFrame(
        [row for trajectory in trajectories for row in observation_rows(trajectory)]
    )
    observation_frame.to_parquet(
        ARTIFACTS / "confirmation_trajectories.parquet",
        index=False,
        compression="zstd",
    )
    results = []
    for candidate in lock["candidates"]:
        results.append(
            candidate_confirmation_result(
                candidate,
                summaries[summaries["candidateId"] == candidate["candidateId"]],
            )
        )
    result_frame = pd.DataFrame(results).sort_values("candidateId").reset_index(drop=True)
    result_frame.to_csv(ARTIFACTS / "posterior_predictive_results.csv", index=False)
    passing = result_frame[result_frame["confirmationGatePassed"]]
    trajectory_locks: dict[str, list[dict[str, Any]]] = {}
    for candidate_id, group in summaries.groupby("candidateId", sort=True):
        trajectory_locks[str(candidate_id)] = [
            {
                "matrixIndex": int(row.matrixIndex),
                "betaSha256": row.betaSha256,
                "initialStateSha256": row.initialStateSha256,
                "trajectorySha256": row.trajectorySha256,
                "cachePath": row.cachePath,
                "cacheSha256": row.cacheSha256,
            }
            for row in group.sort_values("matrixIndex").itertuples(index=False)
        ]
    downstream = {
        "schemaVersion": "E01-S12FR-candidate-timebase-pipeline-lock-v1.0.0",
        "researchStepId": STEP_ID,
        "candidateLockCommit": lock_validation["lockCommit"],
        "testedCandidateCount": lock["candidateCount"],
        "confirmedCandidateCount": int(passing.shape[0]),
        "confirmedCandidates": [
            {
                **candidate,
                "confirmation": result_frame[
                    result_frame["candidateId"] == candidate["candidateId"]
                ].iloc[0].to_dict(),
                "trajectoryLocks": trajectory_locks[candidate["candidateId"]],
                "compositionTrajectorySchema": "initial_plus_each_Poisson_update_plus_selected_post_fission_state",
                "localLagOneIndexing": "output_index_k_maps_state_k_to_state_k_plus_1",
                "seedReplayContract": "unchanged_S12F_PCG64DXSM_SHA256_derivation",
            }
            for candidate in lock["candidates"]
            if candidate["candidateId"] in set(passing["candidateId"])
        ],
        "failedCandidates": result_frame.loc[
            ~result_frame["confirmationGatePassed"]
        ].to_dict("records"),
        "labelsComputed": False,
        "emergenceComputed": False,
        "localPhiRComputed": False,
        "interventionsComputed": False,
        "s12gStarted": False,
        "s13Started": False,
    }
    write_json(ARTIFACTS / "candidate_timebase_pipeline_lock.json", downstream)
    write_json(
        CACHE / "timebase_confirmation_runtime.json",
        {
            "completedAtUtc": datetime.now(UTC).isoformat(),
            "orchestrationWallSeconds": time.perf_counter() - started,
            "campaignWallSeconds": wall,
            "workerCpuSeconds": float(summaries["workerCpuSeconds"].sum()),
            "trajectoryCount": int(summaries.shape[0]),
            "candidateCount": lock["candidateCount"],
            "confirmedCandidateCount": int(passing.shape[0]),
            "developmentConfirmationSeedIntersectionCount": 0,
        },
    )


def ensure_parquet_placeholder(path: Path, reason: str, **extra: Any) -> None:
    if path.is_file():
        return
    pd.DataFrame([{"status": "NOT_REACHED", "reason": reason, **extra}]).to_parquet(
        path, index=False, compression="zstd"
    )


def ensure_csv_placeholder(path: Path, reason: str, **extra: Any) -> None:
    if path.is_file():
        return
    pd.DataFrame([{"status": "NOT_REACHED", "reason": reason, **extra}]).to_csv(
        path, index=False
    )


def runtime_payload() -> dict[str, Any]:
    components = {}
    worker_cpu = 0.0
    wall = 0.0
    for name in (
        "diagnosis_runtime.json",
        "repair_confirmation_runtime.json",
        "abc_runtime.json",
        "timebase_confirmation_runtime.json",
    ):
        path = CACHE / name
        if path.is_file():
            payload = json.loads(path.read_text())
            components[name] = payload
            worker_cpu += float(payload.get("workerCpuSeconds", 0.0))
            wall += float(
                payload.get(
                    "orchestrationWallSeconds",
                    payload.get("campaignWallSeconds", 0.0),
                )
            )
            for family in payload.get("familyMetadata", []):
                worker_cpu += float(family.get("workerCpuSeconds", 0.0))
    return {
        "schemaVersion": "E01-S12FR-runtime-manifest-v1.0.0",
        "researchStepId": STEP_ID,
        "finalizedAtUtc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
        "workers": WORKERS,
        "threadEnvironment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "gpuUsed": False,
        "cpuFloat64Authoritative": True,
        "observedWorkerCpuSeconds": worker_cpu,
        "observedPhaseWallSecondsSum": wall,
        "ceilings": {
            "cpuHours": 250.0,
            "gpuHours": 12.0,
            "wallHours": 72.0,
            "retainedArtifactGiB": 30.0,
        },
        "components": components,
    }


def determine_state() -> dict[str, Any]:
    original = (
        json.loads((ARTIFACTS / "original_pair_summary.json").read_text())
        if (ARTIFACTS / "original_pair_summary.json").is_file()
        else None
    )
    repair = (
        json.loads((ARTIFACTS / "repair_confirmation_validation.json").read_text())
        if (ARTIFACTS / "repair_confirmation_validation.json").is_file()
        else None
    )
    candidate_proposal = (
        json.loads((ARTIFACTS / "candidate_lock_proposal.json").read_text())
        if (ARTIFACTS / "candidate_lock_proposal.json").is_file()
        else None
    )
    downstream = (
        json.loads((ARTIFACTS / "candidate_timebase_pipeline_lock.json").read_text())
        if (ARTIFACTS / "candidate_timebase_pipeline_lock.json").is_file()
        else None
    )
    if original is None or not original.get("campaignGatePassed", False):
        return {
            "repairClassification": "EXACT_REPLAY_COMPARATOR_REPAIR_FAILED_PERMANENT_STOP",
            "timebaseClassification": "UNDERDETERMINED",
            "status": "COMPLETED_FAIL_CLOSED_ORIGINAL_PAIR_DIAGNOSIS",
            "outcomeClass": "constraining/contradictory",
            "firstFailedLayer": "original_2048_pair_diagnosis",
        }
    if repair is None or not repair.get("passed", False):
        return {
            "repairClassification": "EXACT_REPLAY_COMPARATOR_REPAIR_FAILED_PERMANENT_STOP",
            "timebaseClassification": "UNDERDETERMINED",
            "status": "COMPLETED_FAIL_CLOSED_UNTOUCHED_CONFIRMATION",
            "outcomeClass": "constraining/contradictory",
            "firstFailedLayer": "repair_confirmation",
        }
    if candidate_proposal is None:
        return {
            "repairClassification": "EXACT_REPLAY_COMPARATOR_REPAIR_CONFIRMED",
            "timebaseClassification": "SIMULATOR_IDENTIFICATION_FAILED",
            "status": "COMPLETED_FAIL_CLOSED_FRESH_ABC_EXECUTION",
            "outcomeClass": "constraining/contradictory",
            "firstFailedLayer": "fresh_ABC_execution",
        }
    if candidate_proposal["candidateCount"] == 0:
        return {
            "repairClassification": "EXACT_REPLAY_COMPARATOR_REPAIR_CONFIRMED",
            "timebaseClassification": "NO_PAPER_TIMEBASE_RECONSTRUCTION",
            "status": "COMPLETED_REPAIR_CONFIRMED_NO_ACCEPTED_TIMEBASE_CANDIDATE",
            "outcomeClass": "constraining/contradictory",
            "firstFailedLayer": "ABC_acceptance_envelope",
        }
    if downstream is None:
        return {
            "repairClassification": "EXACT_REPLAY_COMPARATOR_REPAIR_CONFIRMED",
            "timebaseClassification": "UNDERDETERMINED",
            "status": "COMPLETED_BLOCKED_BEFORE_CANDIDATE_CONFIRMATION",
            "outcomeClass": "null",
            "firstFailedLayer": "candidate_confirmation_not_completed",
        }
    confirmed = int(downstream["confirmedCandidateCount"])
    if confirmed == 0:
        timebase = "NO_PAPER_TIMEBASE_RECONSTRUCTION"
    elif confirmed == 1:
        timebase = "PAPER_TIMEBASE_CANDIDATE"
    else:
        timebase = "NONIDENTIFIABLE_TIMEBASE_ENSEMBLE"
    return {
        "repairClassification": "EXACT_REPLAY_COMPARATOR_REPAIR_CONFIRMED",
        "timebaseClassification": timebase,
        "status": "COMPLETED_AT_ORIGINAL_S12F_HUMAN_REVIEW_BOUNDARY",
        "outcomeClass": "supportive" if confirmed else "constraining/contradictory",
        "firstFailedLayer": None if confirmed else "posterior_predictive_confirmation",
        "confirmedCandidateCount": confirmed,
    }


def phase_finalize() -> None:
    frozen = verify_frozen()
    state = determine_state()
    reason = state["firstFailedLayer"] or "completed_at_original_human_review_boundary"
    ensure_parquet_placeholder(
        ARTIFACTS / "original_pair_diagnostics.parquet", reason
    )
    ensure_parquet_placeholder(
        ARTIFACTS / "original_field_differences.parquet", reason
    )
    ensure_parquet_placeholder(ARTIFACTS / "original_trace_manifest.parquet", reason)
    ensure_parquet_placeholder(ARTIFACTS / "original_seed_manifest.parquet", reason)
    if not (ARTIFACTS / "original_pair_summary.json").is_file():
        write_json(
            ARTIFACTS / "original_pair_summary.json",
            {"status": "NOT_REACHED", "reason": reason, "campaignGatePassed": False},
        )
    if not (ARTIFACTS / "comparator_lock_proposal.json").is_file():
        write_json(
            ARTIFACTS / "comparator_lock_proposal.json",
            {"status": "NOT_REACHED", "reason": reason, "eligibleToLock": False},
        )
    if not (ARTIFACTS / "comparator_lock_validation.json").is_file():
        write_json(
            ARTIFACTS / "comparator_lock_validation.json",
            {"status": "NOT_REACHED", "reason": reason, "passed": False},
        )
    ensure_csv_placeholder(ARTIFACTS / "benchmark_pair_diagnostics.csv", reason)
    ensure_parquet_placeholder(
        ARTIFACTS / "untouched_pair_diagnostics.parquet", reason
    )
    ensure_parquet_placeholder(
        ARTIFACTS / "untouched_field_differences.parquet", reason
    )
    ensure_parquet_placeholder(
        ARTIFACTS / "untouched_trace_manifest.parquet", reason
    )
    ensure_parquet_placeholder(ARTIFACTS / "untouched_seed_manifest.parquet", reason)
    if not (ARTIFACTS / "untouched_pair_summary.json").is_file():
        write_json(
            ARTIFACTS / "untouched_pair_summary.json",
            {"status": "NOT_REACHED", "reason": reason, "campaignGatePassed": False},
        )
    if not (ARTIFACTS / "seed_matrix_firewall.json").is_file():
        write_json(
            ARTIFACTS / "seed_matrix_firewall.json",
            {"status": "NOT_REACHED", "reason": reason, "passed": False},
        )
    if not (ARTIFACTS / "repair_confirmation_validation.json").is_file():
        write_json(
            ARTIFACTS / "repair_confirmation_validation.json",
            {"status": "NOT_REACHED", "reason": reason, "passed": False},
        )
    ensure_parquet_placeholder(ARTIFACTS / "abc_particle_results.parquet", reason)
    ensure_csv_placeholder(ARTIFACTS / "abc_round_summary.csv", reason)
    ensure_csv_placeholder(ARTIFACTS / "posterior_candidates.csv", reason)
    if not (ARTIFACTS / "candidate_lock_proposal.json").is_file():
        write_json(
            ARTIFACTS / "candidate_lock_proposal.json",
            {
                "status": "NOT_REACHED",
                "reason": reason,
                "candidateCount": 0,
                "requiresCommittedPushedLock": False,
            },
        )
    if not (ARTIFACTS / "candidate_lock_validation.json").is_file():
        write_json(
            ARTIFACTS / "candidate_lock_validation.json",
            {"status": "NOT_REACHED", "reason": reason, "passed": False},
        )
    ensure_parquet_placeholder(ARTIFACTS / "confirmation_seed_manifest.parquet", reason)
    ensure_parquet_placeholder(ARTIFACTS / "confirmation_trajectories.parquet", reason)
    ensure_parquet_placeholder(
        ARTIFACTS / "confirmation_trajectory_manifest.parquet", reason
    )
    ensure_csv_placeholder(ARTIFACTS / "posterior_predictive_results.csv", reason)
    if not (ARTIFACTS / "candidate_timebase_pipeline_lock.json").is_file():
        write_json(
            ARTIFACTS / "candidate_timebase_pipeline_lock.json",
            {
                "schemaVersion": "E01-S12FR-candidate-timebase-pipeline-lock-v1.0.0",
                "researchStepId": STEP_ID,
                "status": "NOT_REACHED",
                "reason": reason,
                "confirmedCandidateCount": 0,
                "confirmedCandidates": [],
                "labelsComputed": False,
                "emergenceComputed": False,
                "localPhiRComputed": False,
                "interventionsComputed": False,
                "s12gStarted": False,
                "s13Started": False,
            },
        )

    original = json.loads((ARTIFACTS / "original_pair_summary.json").read_text())
    repair = json.loads((ARTIFACTS / "repair_confirmation_validation.json").read_text())
    candidate = json.loads(
        (ARTIFACTS / "candidate_timebase_pipeline_lock.json").read_text()
    )
    abc_rounds = pd.read_csv(ARTIFACTS / "abc_round_summary.csv")
    posterior = pd.read_csv(ARTIFACTS / "posterior_predictive_results.csv")
    runtime = runtime_payload()
    write_json(ARTIFACTS / "runtime_manifest.json", runtime)

    prior_post = verify_manifest(ARTIFACTS / "immutable_prior_baseline.json")
    cache_post = verify_manifest(ARTIFACTS / "s12f_cache_baseline.json")
    scope = {
        "schemaVersion": "E01-S12FR-scope-access-ledger-v1.0.0",
        "researchStepId": STEP_ID,
        "s12fArtifactsMutated": False,
        "s12fClassificationChanged": False,
        "previouslySuppressedDistanceOpened": False,
        "originalPairDiagnosisExecuted": original.get("observedPairCount", 0) == 2048,
        "benchmarkExecuted": (ARTIFACTS / "benchmark_field_differences.parquet").is_file(),
        "untouchedRepairConfirmationExecuted": repair.get("untouchedConfirmation", {}).get("observedPairCount", 0) == 2048,
        "freshAbcDistancesComputed": bool(
            "family" in abc_rounds.columns and abc_rounds.shape[0] > 0
        ),
        "adaptiveExposureExecuted": bool(
            "family" in abc_rounds.columns
            and (abc_rounds["family"] == "ADAPTIVE_GROSS_EVENT_EXPOSURE").any()
        ),
        "timebaseCandidateConfirmationExecuted": bool(
            "candidateId" in posterior.columns and posterior["candidateId"].notna().any()
        ),
        "selfReplicationLabelsOpened": False,
        "clusteringOpened": False,
        "sourceEmergenceOpened": False,
        "localPhiROpened": False,
        "predictionRun": False,
        "interventionsRun": False,
        "s12gStarted": False,
        "s13Started": False,
        "authorsContacted": False,
        "oneRepairRuleRespected": True,
    }
    write_json(ARTIFACTS / "scope_access_ledger.json", scope)

    failures = []
    if state["repairClassification"] == "EXACT_REPLAY_COMPARATOR_REPAIR_FAILED_PERMANENT_STOP":
        failures.append(
            {
                "failureId": "S12FR-F001",
                "phase": state["firstFailedLayer"],
                "severity": "TERMINAL_ONE_REPAIR_GATE",
                "status": "FAILED_CLOSED",
                "reason": reason,
                "consequence": "Permanent close; no ABC continuation, S12G, or S13.",
            }
        )
    elif state["timebaseClassification"] == "NO_PAPER_TIMEBASE_RECONSTRUCTION":
        failures.append(
            {
                "failureId": "S12FR-SCI001",
                "phase": state["firstFailedLayer"],
                "severity": "SCIENTIFIC_CONSTRAINT",
                "status": "NO_CONFIRMED_CANDIDATE",
                "reason": reason,
                "consequence": "Comparator repair remains confirmed; no paper time-base candidate.",
            }
        )
    else:
        failures.append(
            {
                "failureId": "S12FR-NONE",
                "phase": "complete",
                "severity": "NONE",
                "status": "NO_TERMINAL_REPAIR_FAILURE",
                "reason": "none",
                "consequence": "Return for mandatory human review.",
            }
        )
    pd.DataFrame(failures).to_csv(ARTIFACTS / "failure_ledger.csv", index=False)

    validation = {
        "schemaVersion": "E01-S12FR-regeneration-validation-v1.0.0",
        "researchStepId": STEP_ID,
        "priorImmutability": prior_post,
        "s12fCacheImmutability": cache_post,
        "originalCampaign": original,
        "repairConfirmation": repair,
        "candidateLock": json.loads(
            (ARTIFACTS / "candidate_lock_validation.json").read_text()
        ),
        "confirmedCandidateCount": int(candidate.get("confirmedCandidateCount", 0)),
        "scopeCompliance": scope,
        "runtimeCeilingsPassed": bool(
            runtime["observedWorkerCpuSeconds"] / 3600.0 <= 250.0
            and runtime["observedPhaseWallSecondsSum"] / 3600.0 <= 72.0
        ),
        "passed": bool(
            prior_post["passed"]
            and cache_post["passed"]
            and scope["oneRepairRuleRespected"]
        ),
    }
    write_json(ARTIFACTS / "regeneration_validation.json", validation)

    classification = {
        "schemaVersion": "E01-S12FR-classification-v1.0.0",
        "researchStepId": STEP_ID,
        **state,
        "s12fClassificationUnchanged": "SIMULATOR_IDENTIFICATION_FAILED",
        "s13Status": "BLOCKED_PENDING_S12FR_HUMAN_REVIEW",
        "sourceRelationship": "PAPER_AS_DATA_SIMULATOR_IDENTIFICATION_ONLY",
    }
    write_json(ARTIFACTS / "classification.json", classification)
    validation_text = (
        "PASS_REPAIR_AND_UNTOUCHED_CONFIRMATION"
        if state["repairClassification"] == "EXACT_REPLAY_COMPARATOR_REPAIR_CONFIRMED"
        else "FAIL_CLOSED_ONE_REPAIR_GATE"
    )
    caveats = [
        "S12F and its SIMULATOR_IDENTIFICATION_FAILED classification remain immutable.",
        "The comparator recognizes only paired zero-update exposure-extrema NaNs; it supplies no finite tolerance.",
        "A repaired comparator does not erase prior negative scientific evidence or establish author-code identity.",
        "Labels, emergence, local Phi-r, prediction, interventions, S12G, and S13 were not executed.",
    ]
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": "S12FR",
        "success": True,
        "status": state["status"],
        "artifactsWritten": [],
        "validationResult": validation_text,
        "outcomeClassification": state["repairClassification"],
        "timebaseOutcomeClassification": state["timebaseClassification"],
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": "Mandatory human review; keep S12G and S13 blocked and do not begin downstream work automatically.",
        "s13Status": "BLOCKED_PENDING_S12FR_HUMAN_REVIEW",
    }
    write_json(ARTIFACTS / "status.json", status)

    original_old_failures = original.get("oldComparatorFailureCount", "not reached")
    original_nan_count = original.get("permittedPairedNanCount", "not reached")
    untouched_count = repair.get("untouchedConfirmation", {}).get(
        "pairGatePassCount", "not reached"
    )
    confirmed_count = int(candidate.get("confirmedCandidateCount", 0))
    round_table = (
        abc_rounds.to_markdown(index=False)
        if "family" in abc_rounds.columns
        else "Fresh ABC was not reached."
    )
    posterior_table = (
        posterior.to_markdown(index=False)
        if "candidateId" in posterior.columns and posterior["candidateId"].notna().any()
        else "No candidate posterior-predictive confirmation was reached."
    )
    report = f"""# S12FR full results — Exact replay comparator repair

## Top summary

- **Research step ID:** `{STEP_ID}`.
- **Completion status:** `{state['status']}`; stopped at the original S12F human-review boundary.
- **Artifacts written:** complete comparator contract, pair/field/RNG/trace diagnostics, confirmation/firewall evidence, conditional ABC and candidate outputs, failure/runtime/scope/replay/provenance/hash manifests, status JSON, and this canonical report under `/artifacts/research_steps/S12FR/`.
- **Validation result:** `{validation_text}`.
- **Outcome classification:** `{state['repairClassification']}`; conditional time-base classification `{state['timebaseClassification']}`.
- **Caveats or blockers:** {' '.join(caveats)}
- **Lay summary:** The repair asked whether S12F's repeatability stop was only a representation problem. The old equality rule treated paired undefined exposure values as unequal. The new rule was allowed to recognize only matching NaNs caused by the same zero-update generation, while retaining exact equality everywhere else. The conditional time-base result is reported separately and does not rewrite S12F.
- **Recommended next action:** Mandatory human review. Keep S12G and S13 blocked; do not start labels, emergence, prediction, interventions, or any downstream step automatically.

## Frozen question and scope

S12FR is one additive operational repair. S01–S12F, including S12F's `SIMULATOR_IDENTIFICATION_FAILED` result and suppressed outputs, were hash-baselined and remained unchanged. The simulator, RNG derivation, roots, particles, exposure families, clocks, targets, summary vector, distance, acceptance gates, adaptive trigger, candidate limit, and 32-matrix confirmation rule were not modified. Previously suppressed S12F distances were never opened.

## Methods

The preregistered comparator uses exact type, shape, sequence, and value equality; finite floats require identical IEEE-754 binary64 bits. Its sole tagged normalization is a paired NaN at `GenerationSummary.maximum_exposure` or `minimum_exposure` when both matching summaries have `update_count == 0`. One-sided NaNs, other NaNs, infinities, finite tolerances, coercion, sequence reordering, or changed RNG consumption fail.

Each audited pair ran the unchanged simulator twice through a recording RNG delegate and once uninstrumented. The delegate recorded exact seed identities, initial/final bit-generator-state hashes, ordered method calls, argument/result hashes, finite/nonfinite counts, and complete integer state, Poisson join/loss, trim, fission, daughter, and stopping sequences. Compact complete canonical traces are retained under `/cache/e01_s12fr/replay_traces`; collectible Parquet manifests record every path, size, and SHA-256. When a pair passed every exact gate one canonical trace represented both identical sides; divergent sides would have been retained separately.

The original campaign re-executed the exact 256 fixed-family round-1 particles on eight original matrices each. Conditional confirmation used all 16 frozen benchmark configurations and 256 untouched particles on eight matrices under disjoint preregistered roots. Fresh ABC distances were permitted only after those gates passed unanimously. Any scientific continuation reused the original S12F implementation and rules.

## Inputs and provenance

- S12F source/method commit: `cf5b27b370a2d8d12e6867034d6ec8f4f96b3fc7`.
- S12FR preregistration commit: `{frozen['record']['gitCommit']}`.
- Comparator: `{COMPARATOR_VERSION}`; RNG audit: `{AUDIT_VERSION}`.
- Original paper, historical GARD, IIGR, PhiRL, safe lattice, S12F artifacts/caches, plans, and manifests are pinned in `source_input_snapshot_manifest.json`, `immutable_prior_baseline.json`, `s12f_cache_baseline.json`, and `s12f_suppressed_input_manifest.json`.
- Prior immutability: {prior_post['fileCount']} files checked, pass `{prior_post['passed']}`. S12F cache: {cache_post['fileCount']} files checked, pass `{cache_post['passed']}`.

## Comparator diagnosis and confirmation results

- Original pairs: {original.get('observedPairCount', 'not reached')}/2,048.
- Old comparator failures recovered: {original_old_failures}.
- Permitted paired zero-update NaN field instances: {original_nan_count}.
- Original repaired pair gates: {original.get('pairGatePassCount', 'not reached')}/2,048.
- Original discrete/finite/forbidden-nonfinite/RNG divergences: {original.get('discreteDivergenceCount', 'not reached')}/{original.get('finiteNumericDivergenceCount', 'not reached')}/{original.get('forbiddenNonfiniteDifferenceCount', 'not reached')}/{original.get('rngDivergenceCount', 'not reached')}.
- Benchmark repaired replay: {repair.get('benchmark', {}).get('pairGatePassCount', 'not reached')}/16.
- Untouched repaired replay: {untouched_count}/2,048.
- Repair seed/matrix firewall: `{repair.get('seedMatrixFirewall', {}).get('passed', 'not reached')}`.

Every old-comparator failure was required to contain at least one permitted tagged NaN and no other left-versus-replay difference. Instrumented/uninstrumented parity and exact trace digests were independent gates.

## Conditional fresh ABC results

{round_table}

No old suppressed distance was reused. The table contains only fresh post-confirmation calculations. The conditional adaptive family appears only if the unchanged fixed-family final envelope contained no accepted particle.

## Posterior-predictive time-base confirmation

{posterior_table}

Confirmed candidate count: **{confirmed_count}**. The resulting lock contains only update kernel, exposure, clock, overshoot, daughter, state/indexing, seeds, upstream fingerprints, and trajectory hashes. It contains no label or information-theory output.

## Validation

- Frozen contract committed and pushed before any rerun: **PASS**.
- Original pair identity/cardinality and complete diagnostics: `{original.get('campaignGatePassed', False)}`.
- Narrow comparator cause rule; zero finite/discrete/RNG divergence: recorded in pair summaries and field tables.
- 16/16 benchmark and untouched 2,048/2,048 confirmation: `{repair.get('passed', False)}`.
- Development/confirmation seed and matrix firewall: `{repair.get('seedMatrixFirewall', {}).get('passed', False)}`.
- Exact replay, trace hashes, instrumentation parity, prior/cache immutability, scope, runtime/storage, schemas, manifests, and required outputs: recorded in `regeneration_validation.json`.
- Runtime: {runtime['observedWorkerCpuSeconds'] / 3600.0:.3f} worker CPU-hours; {runtime['observedPhaseWallSecondsSum'] / 3600.0:.3f} summed phase wall-hours; CPU float64 authoritative; GPU unused.

## Commands

```bash
PYTHONPATH=src python -m pytest -q tests/e01/test_s12fr_replay_repair.py
ruff check src/e01_replay_repair scripts/e01/freeze_s12fr_preregistration.py scripts/e01/run_s12fr_replay_repair.py tests/e01/test_s12fr_replay_repair.py
python scripts/e01/freeze_s12fr_preregistration.py
git commit -m "Preregister S12FR exact replay comparator repair"
git push origin eidosoma/groups/42
python scripts/e01/freeze_s12fr_preregistration.py --record-commit
python scripts/e01/run_s12fr_replay_repair.py --stage diagnose-original
# comparator lock committed and pushed only after the original campaign passed
python scripts/e01/run_s12fr_replay_repair.py --stage confirm-repair
python scripts/e01/run_s12fr_replay_repair.py --stage resume-abc
# candidate lock committed and pushed only when candidates existed
python scripts/e01/run_s12fr_replay_repair.py --stage confirm-timebase
python scripts/e01/run_s12fr_replay_repair.py --stage finalize
```

All long simulator commands used six process workers and one BLAS/OpenMP thread per worker.

## Caveats and interpretation

Comparator success is operational evidence, not scientific validation of a time base. The only normalization is schema-causal and explicitly tagged; nevertheless, this repair was authorized after observing a global failure, so the untouched confirmation firewall is essential. Any candidate is a paper-as-data simulator-identification result, not the unavailable author implementation. S12F remains failed exactly as originally executed, and extensive prior negative or underdetermined Phi-r evidence is unchanged.

## Provenance and artifact completeness

Pair diagnostics retain every pair identity and every field difference. Trace manifests point to complete compact sequence payloads with SHA-256 identities. Seed manifests, firewalls, code/contract locks, runtime records, scope ledger, failure ledger, and artifact manifest provide the complete audit chain. No required downstream value was accessed.

## Recommended next action

Return for mandatory human review with S12G and S13 blocked. Do not automatically begin label reconstruction, causal-emergence analysis, intervention work, or another repair.
"""
    (ARTIFACTS / "research_step_full_results.md").write_text(report, encoding="utf-8")

    required = load_config()["requiredArtifacts"]
    missing = [name for name in required if not (ARTIFACTS / name).is_file()]
    if missing:
        raise RuntimeError(f"required S12FR artifacts missing: {missing}")
    artifact_names = sorted(
        str(path.relative_to(ARTIFACTS))
        for path in ARTIFACTS.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    status["artifactsWritten"] = artifact_names
    write_json(ARTIFACTS / "status.json", status)
    rows = [
        {
            "path": name,
            "sizeBytes": (ARTIFACTS / name).stat().st_size,
            "sha256": sha256(ARTIFACTS / name),
        }
        for name in artifact_names
    ]
    manifest = {
        "schemaVersion": "E01-S12FR-artifact-manifest-v1.0.0",
        "researchStepId": STEP_ID,
        "createdAtUtc": datetime.now(UTC).isoformat(),
        "artifactCountExcludingSelf": len(rows),
        "aggregateSha256": canonical_sha(rows),
        "files": rows,
        "requiredArtifactCount": len(required),
        "requiredArtifactsMissing": [],
        "priorArtifactsImmutable": prior_post["passed"],
        "s12fCacheImmutable": cache_post["passed"],
    }
    write_json(ARTIFACTS / "artifact_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "diagnose-original",
            "confirm-repair",
            "resume-abc",
            "confirm-timebase",
            "finalize",
        ),
    )
    arguments = parser.parse_args()
    if arguments.stage == "diagnose-original":
        phase_diagnose_original()
    elif arguments.stage == "confirm-repair":
        phase_confirm_repair()
    elif arguments.stage == "resume-abc":
        phase_resume_abc()
    elif arguments.stage == "confirm-timebase":
        phase_confirm_timebase()
    else:
        phase_finalize()


if __name__ == "__main__":
    main()
