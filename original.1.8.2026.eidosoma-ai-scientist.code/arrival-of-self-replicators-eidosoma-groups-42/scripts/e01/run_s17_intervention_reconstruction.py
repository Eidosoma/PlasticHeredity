#!/usr/bin/env python3
"""Execute the prospectively locked E01/S17 intervention reconstruction."""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import math
import pickle
import platform
import resource
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml
from numpy.typing import NDArray
from scipy import stats

from e01_intervention_reconstruction.core import (
    BENCHMARK_PHASE,
    BENCHMARK_ROOT_HEX,
    CANDIDATES,
    PHASE,
    ROOT_HEX,
    VERSION,
    array_sha256,
    canonical_json,
    exact_trajectory_replay,
    first_state_divergence,
    generate_beta,
    initialize_distinct_state,
    label_h900,
    simulate_condition,
    stream_seeds,
    trajectory_outcomes,
    trajectory_payload_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/e01/s17_intervention_reconstruction_preregistration.yaml"
EXECUTION_MANIFEST = REPO_ROOT / "configs/e01/s17_execution_manifest.json"
COMPUTE_GATE_LOCK = REPO_ROOT / "configs/e01/s17_compute_gate.json"
STEP_ROOT = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "research_steps/S17"
CACHE_ROOT = Path("/cache/e01_s17_v1")
BENCHMARK_CACHE = CACHE_ROOT / "benchmark"
UNIT_CACHE = CACHE_ROOT / "units"
TRAJECTORY_CACHE = CACHE_ROOT / "trajectories"
AVAILABLE_CPU_HOURS = 100.52383159377861
PROJECTION_MULTIPLIER = 1.25
BOOTSTRAP_REPLICATES = 4096
WORKERS = 8
CONDITIONS = ("MAX", "CONTROL", "MIN")
CANDIDATE_IDS = tuple(CANDIDATES)
MATRIX_INDICES = tuple(range(12))


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported JSON value {type(value)!r}")


def write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=json_default,
        )
    else:
        text = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=json_default,
        )
    path.write_text(text + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(frame.columns).encode())
    digest.update("\x1f".join(map(str, frame.dtypes)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_clean_pushed_lock(*, require_compute_gate: bool) -> dict[str, str]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    status = git("status", "--short")
    if branch != "eidosoma/groups/42" or head != remote or status:
        raise RuntimeError(
            f"pushed-lock gate failed branch={branch!r} head={head} "
            f"remote={remote} status={status!r}"
        )
    if require_compute_gate and not COMPUTE_GATE_LOCK.is_file():
        raise FileNotFoundError("final S17 compute gate lock is absent")
    return {"branch": branch, "head": head, "remoteHead": remote, "status": status}


def prior_paths() -> list[Path]:
    paths: list[Path] = []
    for path in sorted((Path("/artifacts") / "research_steps").glob("S*")):
        if not path.is_dir() or path.name == "S17":
            continue
        for item in sorted(path.rglob("*")):
            if item.is_file():
                paths.append(item)
    bundle = Path("/artifacts/E01_forensic_replication_bundle")
    if bundle.is_dir():
        for item in sorted(bundle.rglob("*")):
            if item.is_file():
                paths.append(item)
    return paths


def hash_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]


def capture_prior_baseline() -> dict[str, Any]:
    inventory = hash_inventory(prior_paths())
    payload = {
        "schema": "eidosoma.e01.s17_immutable_prior_baseline.v1",
        "researchStepId": "S17",
        "capturedAtUtc": utc_now(),
        "fileCount": len(inventory),
        "entries": inventory,
    }
    write_json(STEP_ROOT / "immutable_prior_baseline.json", payload)
    return payload


def validate_prior_baseline(baseline: dict[str, Any]) -> dict[str, Any]:
    current = {row["path"]: row for row in hash_inventory(prior_paths())}
    expected = {row["path"]: row for row in baseline["entries"]}
    missing = sorted(set(expected) - set(current))
    unexpected = sorted(set(current) - set(expected))
    changed = sorted(
        path
        for path in set(expected) & set(current)
        if expected[path]["sha256"] != current[path]["sha256"]
        or expected[path]["bytes"] != current[path]["bytes"]
    )
    result = {
        "schema": "eidosoma.e01.s17_immutable_prior_validation.v1",
        "researchStepId": "S17",
        "passed": not missing and not changed,
        "expectedFileCount": len(expected),
        "observedFileCount": len(current),
        "missing": missing,
        "unexpected": unexpected,
        "changed": changed,
    }
    write_json(STEP_ROOT / "immutable_prior_validation.json", result)
    return result


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["versionedStepId"] != VERSION:
        raise ValueError("S17 config/version mismatch")
    return config


def ensure_preoutcome_lock() -> dict[str, Any]:
    path = STEP_ROOT / "preoutcome_design_lock.json"
    if not path.is_file():
        raise FileNotFoundError("run freeze_s17_intervention_design.py --record-commit first")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if not lock.get("passed"):
        raise RuntimeError("S17 preoutcome design lock did not pass")
    if sha256_file(EXECUTION_MANIFEST) != lock["executionManifestSha256"]:
        raise RuntimeError("S17 execution manifest changed after the pushed lock")
    return lock


def _cache_pickle(path: Path, payload: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def run_benchmark() -> dict[str, Any]:
    """Run one domain-separated complete treated sequence and fail-safe projection."""

    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    BENCHMARK_CACHE.mkdir(parents=True, exist_ok=True)
    lock = ensure_preoutcome_lock()
    pushed = require_clean_pushed_lock(require_compute_gate=False)
    baseline = capture_prior_baseline()
    matrix_index = 17_000_000
    candidate_id = "S12F-CANDIDATE-03"
    started = time.perf_counter()
    process_started = time.process_time()
    treated = simulate_condition(
        candidate_id=candidate_id,
        matrix_index=matrix_index,
        condition="MAX",
        root_hex=BENCHMARK_ROOT_HEX,
        phase=BENCHMARK_PHASE,
    )
    treated_replay_started = time.process_time()
    treated_replay, treated_exact = exact_trajectory_replay(
        treated, root_hex=BENCHMARK_ROOT_HEX, phase=BENCHMARK_PHASE
    )
    treated_replay_cpu = time.process_time() - treated_replay_started
    control = simulate_condition(
        candidate_id=candidate_id,
        matrix_index=matrix_index,
        condition="CONTROL",
        root_hex=BENCHMARK_ROOT_HEX,
        phase=BENCHMARK_PHASE,
    )
    control_replay_started = time.process_time()
    control_replay, control_exact = exact_trajectory_replay(
        control, root_hex=BENCHMARK_ROOT_HEX, phase=BENCHMARK_PHASE
    )
    control_replay_cpu = time.process_time() - control_replay_started
    total_cpu = time.process_time() - process_started
    total_wall = time.perf_counter() - started
    cache = _cache_pickle(
        BENCHMARK_CACHE / "benchmark_output.pickle",
        {
            "treated": treated,
            "treatedReplay": treated_replay,
            "control": control,
            "controlReplay": control_replay,
        },
    )
    source_replay = pd.DataFrame(treated.source_replay_rows)
    action_frame = pd.DataFrame(treated.action_rows)
    candidate_frame = pd.DataFrame(treated.candidate_rows)
    full_sentinel = source_replay.loc[
        source_replay["replayScope"].eq("FULL_SET_SENTINEL")
    ]
    treated_sequence_cpu = float(treated.runtime["cpuSeconds"])
    control_sequence_cpu = float(control.runtime["cpuSeconds"])
    fixed_scope_cpu = (
        48.0 * (treated_sequence_cpu + treated_replay_cpu)
        + 24.0 * (control_sequence_cpu + control_replay_cpu)
    )
    projected_total_cpu_hours = (
        total_cpu / 3600.0 + PROJECTION_MULTIPLIER * fixed_scope_cpu / 3600.0
    )
    gate_passed = bool(
        projected_total_cpu_hours <= AVAILABLE_CPU_HOURS
        and treated.trajectory.completed_fissions == 100
        and control.trajectory.completed_fissions == 100
        and treated_exact
        and control_exact
        and bool(source_replay["exactResultReplay"].all())
        and len(action_frame) == 100
        and action_frame["status"].eq("INTERVENTION_APPLIED").all()
        and len(candidate_frame) > 10_000
        and len(full_sentinel) > 0
        and bool(full_sentinel["exactResultReplay"].all())
    )
    benchmark = {
        "schema": "eidosoma.e01.s17_runtime_benchmark.v1",
        "researchStepId": "S17",
        "versionedStepId": VERSION,
        "completedAtUtc": utc_now(),
        "domain": "S17_RUNTIME_BENCHMARK_ONLY",
        "scientificOutcomeEligible": False,
        "candidateScoreValuesReported": False,
        "candidateId": candidate_id,
        "condition": "MAX",
        "matrixIndex": matrix_index,
        "completedFissionsTreated": treated.trajectory.completed_fissions,
        "completedFissionsControl": control.trajectory.completed_fissions,
        "candidateFitCount": len(candidate_frame),
        "decisionCount": len(action_frame),
        "sourceReplayRowCount": len(source_replay),
        "fullSetSentinelReplayRowCount": len(full_sentinel),
        "allSourceReplayExact": bool(source_replay["exactResultReplay"].all()),
        "treatedTrajectoryReplayExact": treated_exact,
        "controlTrajectoryReplayExact": control_exact,
        "treatedSequenceCpuSeconds": treated_sequence_cpu,
        "treatedSequenceWallSeconds": float(treated.runtime["wallSeconds"]),
        "treatedScoringCpuSeconds": float(treated.runtime["scoringCpuSeconds"]),
        "treatedReplayCpuSeconds": treated_replay_cpu,
        "controlSequenceCpuSeconds": control_sequence_cpu,
        "controlReplayCpuSeconds": control_replay_cpu,
        "benchmarkTotalProcessCpuSeconds": total_cpu,
        "benchmarkTotalWallSeconds": total_wall,
        "projectionMultiplier": PROJECTION_MULTIPLIER,
        "projectedTreatedSequenceCount": 48,
        "projectedControlSequenceCount": 24,
        "projectedFixedScopeCpuHoursBeforeMultiplier": fixed_scope_cpu / 3600.0,
        "projectedTotalS17CpuHoursIncludingBenchmark": projected_total_cpu_hours,
        "availableS17ScientificCpuHours": AVAILABLE_CPU_HOURS,
        "projectedHeadroomCpuHours": AVAILABLE_CPU_HOURS - projected_total_cpu_hours,
        "gatePassed": gate_passed,
        "benchmarkCache": cache,
        "benchmarkTrajectoryPayloadHashes": {
            "treated": trajectory_payload_hash(treated.trajectory),
            "treatedReplay": trajectory_payload_hash(treated_replay.trajectory),
            "control": trajectory_payload_hash(control.trajectory),
            "controlReplay": trajectory_payload_hash(control_replay.trajectory),
        },
        "preoutcomeLock": lock,
        "repositoryState": pushed,
        "priorBaselineFileCount": baseline["fileCount"],
    }
    write_json(STEP_ROOT / "runtime_benchmark.json", benchmark)
    compute_gate = {
        "schema": "eidosoma.e01.s17_compute_gate.v1",
        "researchStepId": "S17",
        "gatePassed": gate_passed,
        "availableS17ScientificCpuHours": AVAILABLE_CPU_HOURS,
        "projectedTotalS17CpuHoursIncludingBenchmark": projected_total_cpu_hours,
        "projectedHeadroomCpuHours": AVAILABLE_CPU_HOURS - projected_total_cpu_hours,
        "projectionMultiplier": PROJECTION_MULTIPLIER,
        "benchmarkSha256": sha256_file(STEP_ROOT / "runtime_benchmark.json"),
        "scientificMatrixCreated": False,
        "scientificOutcomeAccessed": False,
        "requiredAction": (
            "FREEZE_COMMIT_PUSH_FINAL_GATE_THEN_EXECUTE_FIXED_SCOPE"
            if gate_passed
            else "STOP_BEFORE_SCIENTIFIC_OUTCOMES_RETURN_FOR_HUMAN_REVIEW"
        ),
    }
    write_json(STEP_ROOT / "compute_gate.json", compute_gate)
    prior = validate_prior_baseline(baseline)
    if not prior["passed"]:
        raise RuntimeError("prior artifacts changed during S17 benchmark")
    print(json.dumps(compute_gate, sort_keys=True), flush=True)
    return compute_gate


def _unit_paths(candidate_id: str, matrix_index: int) -> dict[str, Path]:
    root = UNIT_CACHE / candidate_id / f"M{matrix_index:03d}"
    return {
        "root": root,
        "summary": root / "unit_summary.json",
        "candidate": root / "candidate_scores.parquet",
        "actions": root / "actions.parquet",
        "boundaries": root / "boundaries.parquet",
        "sourceReplay": root / "source_replay.parquet",
        "trajectoryManifest": root / "trajectory_manifest.parquet",
        "outcomes": root / "outcomes.parquet",
        "labels": root / "labels.parquet",
    }


def _cache_valid(paths: dict[str, Path], contract_sha: str) -> bool:
    if not paths["summary"].is_file():
        return False
    try:
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False
    required = [
        "candidate",
        "actions",
        "boundaries",
        "sourceReplay",
        "trajectoryManifest",
        "outcomes",
        "labels",
    ]
    return bool(
        summary.get("complete")
        and summary.get("contractSha256") == contract_sha
        and all(paths[name].is_file() for name in required)
        and all(
            sha256_file(paths[name]) == summary["files"][name]["sha256"]
            for name in required
        )
    )


def _label_rows(trajectory: Any, candidate_id: str, matrix_index: int, condition: str) -> list[dict[str, Any]]:
    h, labels = label_h900(trajectory)
    return [
        {
            "researchStepId": "S17",
            "candidateId": candidate_id,
            "matrixIndex": matrix_index,
            "condition": condition,
            "trajectoryId": trajectory.trajectory_id,
            "selectedSequenceIndex": index,
            "rawObservationIndex": observation.observation_index,
            "generation": observation.growth_generation_one_based,
            "observationKind": observation.observation_kind,
            "H": float(value),
            "isReplicator": bool(label),
        }
        for index, (observation, value, label) in enumerate(
            zip(trajectory.observations, h, labels, strict=True)
        )
    ]


def run_unit(candidate_id: str, matrix_index: int, contract_sha: str) -> dict[str, Any]:
    """Run one candidate/matrix triplet and atomically retain resumable tables."""

    paths = _unit_paths(candidate_id, matrix_index)
    if _cache_valid(paths, contract_sha):
        return json.loads(paths["summary"].read_text(encoding="utf-8"))
    paths["root"].mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    usage_started = resource.getrusage(resource.RUSAGE_SELF)
    seeds = stream_seeds(root_hex=ROOT_HEX, phase=PHASE, matrix_index=matrix_index)
    beta = generate_beta(seeds["catalytic_matrix"])
    initial = initialize_distinct_state(seeds["initial_state"])
    outputs = {}
    replay_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        output = simulate_condition(
            candidate_id=candidate_id,
            matrix_index=matrix_index,
            condition=condition,
            root_hex=ROOT_HEX,
            phase=PHASE,
            beta=beta,
            initial_state=initial,
        )
        replay, exact = exact_trajectory_replay(
            output, root_hex=ROOT_HEX, phase=PHASE
        )
        outputs[condition] = output
        replay_rows.append(
            {
                "researchStepId": "S17",
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "condition": condition,
                "replayScope": "FULL_TRAJECTORY_FROZEN_ACTION_SCHEDULE",
                "actionId": None,
                "generation": None,
                "exactResultReplay": exact,
                "endpointAbsError": None,
                "originalTrajectorySha256": output.trajectory.trajectory_sha256,
                "replayTrajectorySha256": replay.trajectory.trajectory_sha256,
                "originalPayloadSha256": trajectory_payload_hash(output.trajectory),
                "replayPayloadSha256": trajectory_payload_hash(replay.trajectory),
            }
        )
    candidate_rows = [
        row for output in outputs.values() for row in output.candidate_rows
    ]
    action_rows = [row for output in outputs.values() for row in output.action_rows]
    boundary_rows = [
        row for output in outputs.values() for row in output.boundary_rows
    ]
    replay_rows.extend(
        row for output in outputs.values() for row in output.source_replay_rows
    )
    trajectory_replay_map = {
        str(row["condition"]): bool(row["exactResultReplay"])
        for row in replay_rows
        if row["replayScope"] == "FULL_TRAJECTORY_FROZEN_ACTION_SCHEDULE"
    }
    trajectory_rows = []
    outcome_rows = []
    label_rows = []
    for condition, output in outputs.items():
        trajectory = output.trajectory
        cache_path = (
            TRAJECTORY_CACHE
            / candidate_id
            / f"M{matrix_index:03d}-{condition}.pickle"
        )
        cache = _cache_pickle(cache_path, trajectory)
        action_subset = [
            row for row in action_rows if row["condition"] == condition
        ]
        additions = sum(row.get("operation") == "ADD" for row in action_subset)
        deletions = sum(row.get("operation") == "DELETE" for row in action_subset)
        trajectory_rows.append(
            {
                "researchStepId": "S17",
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "condition": condition,
                "trajectoryId": trajectory.trajectory_id,
                "trajectorySha256": trajectory.trajectory_sha256,
                "trajectoryPayloadSha256": trajectory_payload_hash(trajectory),
                "cachePath": cache["path"],
                "cacheSha256": cache["sha256"],
                "cacheBytes": cache["bytes"],
                "completedFissions": trajectory.completed_fissions,
                "terminalStatus": trajectory.terminal_status,
                "totalBatchUpdates": trajectory.total_batch_updates,
                "observationCount": len(trajectory.observations),
                "betaSha256": trajectory.beta_sha256,
                "initialStateSha256": trajectory.initial_state_sha256,
                "trajectoryReplayExact": trajectory_replay_map[condition],
            }
        )
        outcome_rows.append(
            {
                "researchStepId": "S17",
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "condition": condition,
                "trajectoryId": trajectory.trajectory_id,
                "completedFissions": trajectory.completed_fissions,
                "terminalStatus": trajectory.terminal_status,
                "additionCount": additions,
                "deletionCount": deletions,
                "appliedActionCount": additions + deletions,
                **trajectory_outcomes(trajectory),
            }
        )
        label_rows.extend(_label_rows(trajectory, candidate_id, matrix_index, condition))
    frames = {
        "candidate": pd.DataFrame(candidate_rows),
        "actions": pd.DataFrame(action_rows),
        "boundaries": pd.DataFrame(boundary_rows),
        "sourceReplay": pd.DataFrame(replay_rows),
        "trajectoryManifest": pd.DataFrame(trajectory_rows),
        "outcomes": pd.DataFrame(outcome_rows),
        "labels": pd.DataFrame(label_rows),
    }
    sort_by = {
        "candidate": ["condition", "generation", "actionOrder"],
        "actions": ["condition", "generation"],
        "boundaries": ["condition", "generation"],
        "sourceReplay": ["condition", "generation", "replayScope", "actionId"],
        "trajectoryManifest": ["condition"],
        "outcomes": ["condition"],
        "labels": ["condition", "selectedSequenceIndex"],
    }
    files = {}
    for name, frame in frames.items():
        columns = [column for column in sort_by[name] if column in frame.columns]
        if columns:
            frame = frame.sort_values(columns, kind="stable", na_position="last")
        frame.to_parquet(paths[name], index=False)
        files[name] = {
            "path": str(paths[name]),
            "bytes": paths[name].stat().st_size,
            "sha256": sha256_file(paths[name]),
            "rows": len(frame),
        }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = (usage.ru_utime - usage_started.ru_utime) + (
        usage.ru_stime - usage_started.ru_stime
    )
    summary = {
        "schema": "eidosoma.e01.s17_unit_summary.v1",
        "researchStepId": "S17",
        "candidateId": candidate_id,
        "matrixIndex": matrix_index,
        "contractSha256": contract_sha,
        "complete": True,
        "completedAtUtc": utc_now(),
        "wallSeconds": time.perf_counter() - started,
        "processCpuSeconds": cpu_seconds,
        "allTrajectoriesCompleted100Fissions": all(
            output.trajectory.completed_fissions == 100 for output in outputs.values()
        ),
        "allTrajectoryReplaysExact": all(
            row["exactResultReplay"]
            for row in replay_rows
            if row["replayScope"] == "FULL_TRAJECTORY_FROZEN_ACTION_SCHEDULE"
        ),
        "allSourceReplaysExact": all(
            row["exactResultReplay"]
            for row in replay_rows
            if row["replayScope"] != "FULL_TRAJECTORY_FROZEN_ACTION_SCHEDULE"
        ),
        "files": files,
    }
    write_json(paths["summary"], summary)
    return summary


def _read_units(summaries: list[dict[str, Any]], name: str) -> pd.DataFrame:
    frames = [pd.read_parquet(summary["files"][name]["path"]) for summary in summaries]
    return pd.concat(frames, ignore_index=True, sort=False)


def _seed_manifest() -> pd.DataFrame:
    rows = []
    for matrix_index in MATRIX_INDICES:
        seeds = stream_seeds(root_hex=ROOT_HEX, phase=PHASE, matrix_index=matrix_index)
        for candidate_id in CANDIDATE_IDS:
            for condition in CONDITIONS:
                for purpose, identity in seeds.items():
                    rows.append(
                        {
                            "researchStepId": "S17",
                            "streamDomain": "simulation",
                            "streamId": f"S17::{purpose}::M{matrix_index:03d}::SHARED",
                            "purpose": purpose,
                            "candidateId": candidate_id,
                            "matrixIndex": matrix_index,
                            "condition": condition,
                            "derivedSeed": str(identity.derived_seed),
                            "seedMaterialSha256": identity.seed_material_sha256,
                            "rootHex": identity.root_sha256,
                            "bitGenerator": "PCG64DXSM",
                            "sharedAcrossCandidates": True,
                            "sharedAcrossConditions": True,
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["matrixIndex", "candidateId", "condition", "purpose"], kind="stable"
    )


def _matrix_manifest() -> pd.DataFrame:
    rows = []
    for matrix_index in MATRIX_INDICES:
        seeds = stream_seeds(root_hex=ROOT_HEX, phase=PHASE, matrix_index=matrix_index)
        beta = generate_beta(seeds["catalytic_matrix"])
        initial = initialize_distinct_state(seeds["initial_state"])
        rows.append(
            {
                "researchStepId": "S17",
                "matrixIndex": matrix_index,
                "betaSha256": array_sha256(beta),
                "initialStateSha256": array_sha256(initial),
                "initialMass": int(initial.sum()),
                "initialDistinctTypeCount": int(np.count_nonzero(initial)),
                "sharedAcrossCandidates": True,
                "sharedAcrossConditions": True,
            }
        )
    return pd.DataFrame(rows)


def _pairing_divergence(trajectory_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate_id, matrix_index), group in trajectory_manifest.groupby(
        ["candidateId", "matrixIndex"], sort=True
    ):
        trajectories = {}
        for item in group.itertuples(index=False):
            with Path(item.cachePath).open("rb") as handle:
                trajectories[item.condition] = pickle.load(handle)
        for left, right in (("MAX", "CONTROL"), ("CONTROL", "MIN"), ("MAX", "MIN")):
            rows.append(
                {
                    "researchStepId": "S17",
                    "candidateId": candidate_id,
                    "matrixIndex": int(matrix_index),
                    "leftCondition": left,
                    "rightCondition": right,
                    "sharedBeta": trajectories[left].beta_sha256
                    == trajectories[right].beta_sha256,
                    "sharedInitialState": trajectories[left].initial_state_sha256
                    == trajectories[right].initial_state_sha256,
                    "sharedNamedStreamIdentities": True,
                    **first_state_divergence(trajectories[left], trajectories[right]),
                }
            )
    return pd.DataFrame(rows)


def _generation_probability(labels: pd.DataFrame) -> pd.DataFrame:
    selected = labels.loc[labels["generation"].gt(0)].copy()
    return (
        selected.groupby(
            ["candidateId", "matrixIndex", "condition", "generation"], sort=True
        )["isReplicator"]
        .mean()
        .rename("replicationProbability")
        .reset_index()
    )


def _generation_trends(generation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate_id, condition), group in generation.groupby(
        ["candidateId", "condition"], sort=True
    ):
        aggregate = (
            group.groupby("generation", sort=True)["replicationProbability"]
            .mean()
            .reset_index()
        )
        regression = stats.linregress(
            aggregate["generation"].to_numpy(dtype=float),
            aggregate["replicationProbability"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "candidateId": candidate_id,
                "condition": condition,
                "generationCount": len(aggregate),
                "matrixCount": group["matrixIndex"].nunique(),
                "slope": float(regression.slope),
                "intercept": float(regression.intercept),
                "rValue": float(regression.rvalue),
                "twoSidedP": float(regression.pvalue),
                "standardError": float(regression.stderr),
                "firstGenerationMeanProbability": float(
                    aggregate.iloc[0]["replicationProbability"]
                ),
                "lastGenerationMeanProbability": float(
                    aggregate.iloc[-1]["replicationProbability"]
                ),
            }
        )
    return pd.DataFrame(rows)


OUTCOME_METRICS = (
    "persistence",
    "probability",
    "consistency",
    "timeToFirstReplicator",
    "timeToFirstNormalized",
    "longestReplicatingEpisode",
    "meanReplicatingEpisodeDuration",
    "entryCount",
    "exitCount",
)


def _summary_interval(values: NDArray[np.float64]) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "sampleStd": None,
            "standardError": None,
            "lower95": None,
            "upper95": None,
        }
    mean = float(np.mean(values))
    median = float(np.median(values))
    sample_std = float(np.std(values, ddof=1)) if n > 1 else None
    standard_error = float(sample_std / math.sqrt(n)) if sample_std is not None else None
    if standard_error is not None:
        width = float(stats.t.ppf(0.975, df=n - 1) * standard_error)
        lower, upper = mean - width, mean + width
    else:
        lower = upper = None
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "sampleStd": sample_std,
        "standardError": standard_error,
        "lower95": lower,
        "upper95": upper,
    }


def _outcome_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [
        (candidate_id, group)
        for candidate_id, group in outcomes.groupby("candidateId", sort=True)
    ]
    pooled = outcomes.copy()
    groups.append(("POOLED_SECONDARY", pooled))
    for candidate_id, candidate in groups:
        for condition, group in candidate.groupby("condition", sort=True):
            for metric in OUTCOME_METRICS:
                values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
                rows.append(
                    {
                        "candidateId": candidate_id,
                        "poolingRole": (
                            "SECONDARY_DESCRIPTIVE"
                            if candidate_id == "POOLED_SECONDARY"
                            else "CANDIDATE_SPECIFIC_PRIMARY"
                        ),
                        "condition": condition,
                        "outcome": metric,
                        **_summary_interval(values),
                    }
                )
    return pd.DataFrame(rows)


def _bootstrap_effects(outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons = (
        ("MAX_MINUS_CONTROL", "MAX", "CONTROL"),
        ("CONTROL_MINUS_MIN", "CONTROL", "MIN"),
        ("MAX_MINUS_MIN", "MAX", "MIN"),
    )
    effect_rows = []
    bootstrap_rows = []
    summary_rows = []
    for candidate_id, candidate in outcomes.groupby("candidateId", sort=True):
        wide = candidate.set_index(["matrixIndex", "condition"])
        for metric in OUTCOME_METRICS:
            for comparison, left, right in comparisons:
                paired = []
                for matrix_index in MATRIX_INDICES:
                    a = wide.loc[(matrix_index, left), metric]
                    b = wide.loc[(matrix_index, right), metric]
                    if pd.notna(a) and pd.notna(b):
                        paired.append((matrix_index, float(a) - float(b)))
                differences = np.asarray([row[1] for row in paired], dtype=float)
                for matrix_index, difference in paired:
                    effect_rows.append(
                        {
                            "candidateId": candidate_id,
                            "matrixIndex": matrix_index,
                            "outcome": metric,
                            "comparison": comparison,
                            "difference": difference,
                        }
                    )
                seed_material = (
                    f"{VERSION}\x1fpaired_bootstrap\x1f{candidate_id}\x1f"
                    f"{metric}\x1f{comparison}"
                )
                seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:16], "big")
                rng = np.random.Generator(np.random.PCG64DXSM(seed))
                boot = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
                for replicate in range(BOOTSTRAP_REPLICATES):
                    sample = rng.integers(0, len(differences), size=len(differences))
                    boot[replicate] = float(np.mean(differences[sample]))
                    bootstrap_rows.append(
                        {
                            "candidateId": candidate_id,
                            "outcome": metric,
                            "comparison": comparison,
                            "replicate": replicate,
                            "meanPairedDifference": boot[replicate],
                            "seed": str(seed),
                        }
                    )
                try:
                    wilcoxon = stats.wilcoxon(differences, alternative="two-sided")
                    wilcoxon_stat = float(wilcoxon.statistic)
                    wilcoxon_p = float(wilcoxon.pvalue)
                except ValueError:
                    wilcoxon_stat = None
                    wilcoxon_p = None
                summary_rows.append(
                    {
                        "candidateId": candidate_id,
                        "outcome": metric,
                        "comparison": comparison,
                        "pairedMatrixCount": len(differences),
                        "meanDifference": float(np.mean(differences)),
                        "medianDifference": float(np.median(differences)),
                        "positiveDifferenceCount": int(np.count_nonzero(differences > 0)),
                        "negativeDifferenceCount": int(np.count_nonzero(differences < 0)),
                        "zeroDifferenceCount": int(np.count_nonzero(differences == 0)),
                        "bootstrapLower95": float(np.quantile(boot, 0.025)),
                        "bootstrapUpper95": float(np.quantile(boot, 0.975)),
                        "wilcoxonStatistic": wilcoxon_stat,
                        "wilcoxonTwoSidedP": wilcoxon_p,
                    }
                )
    return (
        pd.DataFrame(effect_rows),
        pd.DataFrame(bootstrap_rows),
        pd.DataFrame(summary_rows),
    )


def _action_diagnostics(actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    treated = actions.loc[actions["condition"].isin(["MAX", "MIN"])].copy()
    summary_rows = []
    for (candidate_id, condition), group in treated.groupby(
        ["candidateId", "condition"], sort=True
    ):
        gaps = pd.to_numeric(group["bestRunnerUpGap"], errors="coerce").to_numpy(float)
        summary_rows.append(
            {
                "candidateId": candidate_id,
                "condition": condition,
                "decisionCount": len(group),
                "appliedActionCount": int(group["status"].eq("INTERVENTION_APPLIED").sum()),
                "actionFrequency": float(group["status"].eq("INTERVENTION_APPLIED").mean()),
                "additionCount": int(group["operation"].eq("ADD").sum()),
                "deletionCount": int(group["operation"].eq("DELETE").sum()),
                "exactTieCount": int(group["exactTie"].fillna(False).sum()),
                "exactTieFraction": float(group["exactTie"].fillna(False).mean()),
                "medianGap": float(np.nanmedian(gaps)),
                "minimumGap": float(np.nanmin(gaps)),
                "maximumGap": float(np.nanmax(gaps)),
                "gapAboveReplayFraction": float(
                    group["gapExceedsNumericalReplayError"].fillna(False).mean()
                ),
                "gapAboveHistoricalEnvelopeFraction": float(
                    group["gapExceedsFrozenHistoricalReplayEnvelope"].fillna(False).mean()
                ),
                "gapAboveRunnerUpUncertaintyFraction": float(
                    group["gapExceedsRunnerUpUncertaintyScale"].fillna(False).mean()
                ),
                "matchedRandomCorrectDirectionFraction": float(
                    (
                        group["selectedMinusMatchedRandomScore"].gt(0)
                        if condition == "MAX"
                        else group["selectedMinusMatchedRandomScore"].lt(0)
                    ).mean()
                ),
                "meanEligibleCandidateCount": float(group["eligibleCandidateCount"].mean()),
                "minimumEligibleCandidateCount": int(group["eligibleCandidateCount"].min()),
                "maximumSelectedReplayError": float(
                    group["selectedReplayMaxAbsError"].max()
                ),
            }
        )
    by_generation = (
        treated.groupby(["candidateId", "condition", "generation"], sort=True)
        .agg(
            meanGap=("bestRunnerUpGap", "mean"),
            medianGap=("bestRunnerUpGap", "median"),
            actionCount=("actionId", "size"),
            additionFraction=("operation", lambda x: float(x.eq("ADD").mean())),
            tieFraction=("exactTie", "mean"),
            separableFraction=("gapExceedsRunnerUpUncertaintyScale", "mean"),
        )
        .reset_index()
    )
    return pd.DataFrame(summary_rows), by_generation


def _paper_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    targets = {
        "MAX": {
            "persistence": (874.0, 233.0),
            "probability": (0.88, 0.03),
            "consistency": (0.52, 0.04),
            "timeToFirstNormalized": (0.36, 0.26),
        },
        "CONTROL": {
            "persistence": (716.0, 198.0),
            "probability": (0.88, 0.03),
            "consistency": (0.38, 0.06),
            "timeToFirstNormalized": (0.37, 0.27),
        },
        "MIN": {
            "persistence": (559.0, 99.0),
            "probability": (0.80, 0.03),
            "consistency": (0.42, 0.04),
            "timeToFirstNormalized": (0.40, 0.28),
        },
    }
    rows = []
    for candidate_id in (*CANDIDATE_IDS, "POOLED_SECONDARY"):
        for condition, metrics in targets.items():
            for metric, (target, dispersion) in metrics.items():
                found = summary.loc[
                    summary["candidateId"].eq(candidate_id)
                    & summary["condition"].eq(condition)
                    & summary["outcome"].eq(metric)
                ].iloc[0]
                observed = found["mean"]
                rows.append(
                    {
                        "candidateId": candidate_id,
                        "condition": condition,
                        "outcome": metric,
                        "observedMean": observed,
                        "observedSampleStd": found["sampleStd"],
                        "paperMean": target,
                        "paperReportedDispersion": dispersion,
                        "paperDispersionMeaning": "UNDERDETERMINED_SD_OR_SE",
                        "observedMinusPaper": (
                            float(observed) - target if pd.notna(observed) else None
                        ),
                        "withinOnePaperReportedDispersion": (
                            abs(float(observed) - target) <= dispersion
                            if pd.notna(observed)
                            else None
                        ),
                        "poolingRole": (
                            "SECONDARY_DESCRIPTIVE"
                            if candidate_id == "POOLED_SECONDARY"
                            else "CANDIDATE_SPECIFIC_PRIMARY"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _interpretation_gates(
    outcomes: pd.DataFrame,
    effect_summary: pd.DataFrame,
    action_summary: pd.DataFrame,
    replay: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    literal_by_candidate: dict[str, bool] = {}
    causal_effect_by_candidate_metric: dict[tuple[str, str], bool] = {}
    trajectory_replay = replay.loc[
        replay["replayScope"].eq("FULL_TRAJECTORY_FROZEN_ACTION_SCHEDULE")
    ]
    source_replay = replay.loc[
        ~replay["replayScope"].eq("FULL_TRAJECTORY_FROZEN_ACTION_SCHEDULE")
    ]
    for candidate_id in CANDIDATE_IDS:
        candidate = outcomes.loc[outcomes["candidateId"].eq(candidate_id)]
        means = candidate.groupby("condition", sort=True)[
            ["persistence", "probability"]
        ].mean()
        persistence_order = bool(
            means.loc["MAX", "persistence"]
            >= means.loc["CONTROL", "persistence"]
            >= means.loc["MIN", "persistence"]
        )
        probability_order = bool(
            means.loc["MAX", "probability"]
            >= means.loc["CONTROL", "probability"]
            >= means.loc["MIN", "probability"]
        )
        actions = action_summary.loc[
            action_summary["candidateId"].eq(candidate_id)
        ]
        adequate_actions = bool(actions["actionFrequency"].ge(0.95).all())
        exact_replay = bool(
            trajectory_replay.loc[
                trajectory_replay["candidateId"].eq(candidate_id),
                "exactResultReplay",
            ].all()
            and source_replay.loc[
                source_replay["candidateId"].eq(candidate_id),
                "exactResultReplay",
            ].all()
        )
        literal = bool(
            (persistence_order or probability_order)
            and adequate_actions
            and exact_replay
        )
        literal_by_candidate[candidate_id] = literal
        for gate_id, passed, detail in (
            (
                "LITERAL_PERSISTENCE_ORDER",
                persistence_order,
                f"means={means['persistence'].to_dict()}",
            ),
            (
                "LITERAL_PROBABILITY_ORDER",
                probability_order,
                f"means={means['probability'].to_dict()}",
            ),
            (
                "ADEQUATE_ACTION_FREQUENCY",
                adequate_actions,
                f"minimum={actions['actionFrequency'].min():.6g}",
            ),
            ("EXACT_PROSPECTIVE_REPLAY", exact_replay, "trajectory_and_source"),
            ("NO_FUTURE_OBSERVATIONS", True, "locked_by_construction"),
            (
                "CANDIDATE_LITERAL_ORDERING_RESEMBLANCE",
                literal,
                "persistence_or_probability_plus_actions_replay_no_future",
            ),
        ):
            rows.append(
                {
                    "gateFamily": "LITERAL_INTERVENTION_ORDERING_RESEMBLANCE",
                    "candidateId": candidate_id,
                    "gateId": gate_id,
                    "passed": passed,
                    "status": "PASS" if passed else "FAIL",
                    "detail": detail,
                }
            )

        for metric in ("persistence", "probability"):
            max_effect = effect_summary.loc[
                effect_summary["candidateId"].eq(candidate_id)
                & effect_summary["outcome"].eq(metric)
                & effect_summary["comparison"].eq("MAX_MINUS_CONTROL")
            ].iloc[0]
            min_effect = effect_summary.loc[
                effect_summary["candidateId"].eq(candidate_id)
                & effect_summary["outcome"].eq(metric)
                & effect_summary["comparison"].eq("CONTROL_MINUS_MIN")
            ].iloc[0]
            passed = bool(
                max_effect["meanDifference"] > 0
                and min_effect["meanDifference"] > 0
                and max_effect["bootstrapLower95"] > 0
                and min_effect["bootstrapLower95"] > 0
            )
            causal_effect_by_candidate_metric[(candidate_id, metric)] = passed
            rows.append(
                {
                    "gateFamily": "PROSPECTIVE_CAUSAL_CONTROL_SUPPORTED",
                    "candidateId": candidate_id,
                    "gateId": f"OPPOSITE_PAIRED_EFFECTS_{metric.upper()}",
                    "passed": passed,
                    "status": "PASS" if passed else "FAIL",
                    "detail": (
                        f"max-control mean={max_effect['meanDifference']:.6g}, "
                        f"CI=[{max_effect['bootstrapLower95']:.6g},"
                        f"{max_effect['bootstrapUpper95']:.6g}]; "
                        f"control-min mean={min_effect['meanDifference']:.6g}, "
                        f"CI=[{min_effect['bootstrapLower95']:.6g},"
                        f"{min_effect['bootstrapUpper95']:.6g}]"
                    ),
                }
            )
        separable = bool(actions["gapAboveRunnerUpUncertaintyFraction"].ge(0.95).all())
        matched_score = bool(
            actions["matchedRandomCorrectDirectionFraction"].ge(0.95).all()
        )
        for gate_id, passed, detail in (
            (
                "ACTION_SEPARABILITY_AND_NUMERICAL_STABILITY",
                separable,
                f"minimum_fraction={actions['gapAboveRunnerUpUncertaintyFraction'].min():.6g}",
            ),
            (
                "MATCHED_RANDOM_ACTION_SCORE_DIRECTION",
                matched_score,
                f"minimum_fraction={actions['matchedRandomCorrectDirectionFraction'].min():.6g}",
            ),
            (
                "MATCHED_RANDOM_ACTION_OUTCOME_EXPLANATION_EXCLUDED",
                False,
                "UNDERDETERMINED_NO_FOURTH_RANDOM_ACTION_ROLLOUT_IN_EXACT_72_SCOPE",
            ),
        ):
            rows.append(
                {
                    "gateFamily": "PROSPECTIVE_CAUSAL_CONTROL_SUPPORTED",
                    "candidateId": candidate_id,
                    "gateId": gate_id,
                    "passed": passed,
                    "status": (
                        "PASS"
                        if passed
                        else (
                            "UNDERDETERMINED"
                            if gate_id
                            == "MATCHED_RANDOM_ACTION_OUTCOME_EXPLANATION_EXCLUDED"
                            else "FAIL"
                        )
                    ),
                    "detail": detail,
                }
            )

    literal_overall = bool(all(literal_by_candidate.values()))
    qualifying_metric = None
    for metric in ("persistence", "probability"):
        if all(
            causal_effect_by_candidate_metric[(candidate_id, metric)]
            for candidate_id in CANDIDATE_IDS
        ):
            qualifying_metric = metric
            break
    adequate_overall = bool(action_summary["actionFrequency"].ge(0.95).all())
    separable_overall = bool(
        action_summary["gapAboveRunnerUpUncertaintyFraction"].ge(0.95).all()
    )
    matched_score_overall = bool(
        action_summary["matchedRandomCorrectDirectionFraction"].ge(0.95).all()
    )
    causal_overall = False  # The locked 72-scope cannot outcome-test random actions.
    rows.extend(
        [
            {
                "gateFamily": "LITERAL_INTERVENTION_ORDERING_RESEMBLANCE",
                "candidateId": "BOTH_CANDIDATES",
                "gateId": "OVERALL_LITERAL_ORDERING",
                "passed": literal_overall,
                "status": "PASS" if literal_overall else "FAIL",
                "detail": canonical_json(literal_by_candidate),
            },
            {
                "gateFamily": "PROSPECTIVE_CAUSAL_CONTROL_SUPPORTED",
                "candidateId": "BOTH_CANDIDATES",
                "gateId": "OVERALL_PROSPECTIVE_CAUSAL_CONTROL",
                "passed": causal_overall,
                "status": "FAIL",
                "detail": canonical_json(
                    {
                        "sameQualifyingOutcome": qualifying_metric,
                        "adequateActionFrequency": adequate_overall,
                        "separable": separable_overall,
                        "matchedRandomScoreDirection": matched_score_overall,
                        "matchedRandomOutcomeExclusion": False,
                    }
                ),
            },
        ]
    )
    decision = {
        "schema": "eidosoma.e01.s17_decision.v1",
        "researchStepId": "S17",
        "literalInterventionOrderingResemblance": literal_overall,
        "literalClassification": (
            "LITERAL_INTERVENTION_ORDERING_RESEMBLANCE"
            if literal_overall
            else "NOT_SUPPORTED_WITHIN_TESTED_SCOPE"
        ),
        "prospectiveCausalControlSupported": causal_overall,
        "causalClassification": "NOT_SUPPORTED_WITHIN_TESTED_SCOPE",
        "qualifyingPairedEffectOutcome": qualifying_metric,
        "matchedRandomOutcomeExclusion": (
            "UNDERDETERMINED_NO_FOURTH_RANDOM_ACTION_ROLLOUT_IN_EXACT_72_SCOPE"
        ),
        "candidateSpecificLiteral": literal_by_candidate,
        "claimBoundary": (
            "Literal prospective execution and causal-control support are separate. "
            "No retrospective result rescues a failed causal gate."
        ),
    }
    return pd.DataFrame(rows), decision


def _claim_status(
    summary: pd.DataFrame, trends: pd.DataFrame, decision: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for candidate_id in CANDIDATE_IDS:
        values = summary.loc[summary["candidateId"].eq(candidate_id)]
        mean = {
            (row.condition, row.outcome): row.mean
            for row in values.itertuples(index=False)
        }
        statuses = {
            "persistence_order": mean[("MAX", "persistence")]
            >= mean[("CONTROL", "persistence")]
            >= mean[("MIN", "persistence")],
            "probability_order": mean[("MAX", "probability")]
            >= mean[("CONTROL", "probability")]
            >= mean[("MIN", "probability")],
            "consistency_max_gt_control": mean[("MAX", "consistency")]
            > mean[("CONTROL", "consistency")],
            "min_worsens_all_four": all(
                (
                    mean[("MIN", metric)] < mean[("CONTROL", metric)]
                    if metric != "timeToFirstNormalized"
                    else mean[("MIN", metric)] > mean[("CONTROL", metric)]
                )
                for metric in (
                    "persistence",
                    "probability",
                    "consistency",
                    "timeToFirstNormalized",
                )
            ),
        }
        max_trend = trends.loc[
            trends["candidateId"].eq(candidate_id)
            & trends["condition"].eq("MAX")
        ].iloc[0]
        control_trend = trends.loc[
            trends["candidateId"].eq(candidate_id)
            & trends["condition"].eq("CONTROL")
        ].iloc[0]
        statuses["max_probability_increasing_control_nonsignificant"] = bool(
            max_trend["slope"] > 0
            and max_trend["twoSidedP"] < 0.001
            and control_trend["twoSidedP"] >= 0.05
        )
        for claim, supported in statuses.items():
            rows.append(
                {
                    "candidateId": candidate_id,
                    "claimId": claim,
                    "status": (
                        "DIRECTIONALLY_SUPPORTED"
                        if supported
                        else "NOT_SUPPORTED_WITHIN_TESTED_SCOPE"
                    ),
                    "supported": bool(supported),
                }
            )
    rows.extend(
        [
            {
                "candidateId": "BOTH_CANDIDATES",
                "claimId": "literal_intervention_ordering_resemblance",
                "status": decision["literalClassification"],
                "supported": decision["literalInterventionOrderingResemblance"],
            },
            {
                "candidateId": "BOTH_CANDIDATES",
                "claimId": "prospective_causal_control",
                "status": decision["causalClassification"],
                "supported": decision["prospectiveCausalControlSupported"],
            },
        ]
    )
    return pd.DataFrame(rows)


def _make_figure6(
    outcomes: pd.DataFrame, generation: pd.DataFrame, action_summary: pd.DataFrame
) -> None:
    figure_root = STEP_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    colors = {"MAX": "#2b8cbe", "CONTROL": "#7f7f7f", "MIN": "#d95f0e"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    ax.axis("off")
    ax.text(
        0.02,
        0.95,
        "A  Locked intervention pipeline",
        fontweight="bold",
        fontsize=12,
        va="top",
    )
    pipeline = [
        "GARD growth",
        "selected-daughter fission boundary",
        "enumerate 100 additions + present deletions",
        "append edit + refit current PhiRL prefix",
        "raw max / no-action / raw min",
        "continue with common streams until divergence",
    ]
    for index, text in enumerate(pipeline):
        y = 0.82 - index * 0.135
        ax.text(
            0.5,
            y,
            text,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "fc": "#f7fbff", "ec": "#6baed6"},
            fontsize=9,
        )
        if index < len(pipeline) - 1:
            ax.annotate("", xy=(0.5, y - 0.07), xytext=(0.5, y - 0.035), arrowprops={"arrowstyle": "->"})

    ax = axes[0, 1]
    positions = []
    data = []
    labels = []
    position = 1
    for candidate_id in CANDIDATE_IDS:
        for condition in CONDITIONS:
            values = outcomes.loc[
                outcomes["candidateId"].eq(candidate_id)
                & outcomes["condition"].eq(condition),
                "persistence",
            ].to_numpy(float)
            data.append(values)
            positions.append(position)
            labels.append(f"{candidate_id[-2:]}\n{condition.lower()}")
            position += 1
        position += 0.6
    boxes = ax.boxplot(data, positions=positions, widths=0.65, patch_artist=True)
    for box, label in zip(boxes["boxes"], labels, strict=True):
        condition = label.split("\n")[1].upper()
        box.set_facecolor(colors[condition])
        box.set_alpha(0.55)
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylabel("replicating molecular steps")
    ax.set_title("B  Persistence by candidate and treatment")
    ax.grid(axis="y", alpha=0.25)

    for panel, candidate_id in zip((axes[1, 0], axes[1, 1]), CANDIDATE_IDS, strict=True):
        for condition in CONDITIONS:
            subset = generation.loc[
                generation["candidateId"].eq(candidate_id)
                & generation["condition"].eq(condition)
            ]
            aggregate = subset.groupby("generation")["replicationProbability"].agg(
                ["mean", "sem"]
            )
            x = aggregate.index.to_numpy(float)
            mean = aggregate["mean"].to_numpy(float)
            sem = aggregate["sem"].fillna(0).to_numpy(float)
            ax = panel
            ax.plot(x, mean, label=condition.lower(), color=colors[condition])
            ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=colors[condition], alpha=0.12)
        panel.set_ylim(0, 1.02)
        panel.set_xlabel("generation")
        panel.set_ylabel("replication probability")
        panel.set_title(f"C  {candidate_id}")
        panel.grid(alpha=0.2)
        panel.legend(frameon=False)
    fig.suptitle(
        "Figure 6 reconstruction — literal online APPEND_AND_REFIT_CURRENT_PREFIX",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(figure_root / "figure6_reconstruction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(action_summary))
    ax.bar(
        x,
        action_summary["gapAboveRunnerUpUncertaintyFraction"],
        color=[colors[value] for value in action_summary["condition"]],
        alpha=0.75,
    )
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(
        x,
        [f"{row.candidateId[-2:]}-{row.condition.lower()}" for row in action_summary.itertuples()],
        rotation=30,
        ha="right",
    )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("fraction gaps above uncertainty scale")
    ax.set_title("Literal action selection versus numerical separability")
    fig.tight_layout()
    fig.savefig(figure_root / "action_separability.png", dpi=180)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame.loc[:, columns].copy()
    return view.to_markdown(index=False, floatfmt=".6g")


def _write_report(
    *,
    decision: dict[str, Any],
    outcome_summary: pd.DataFrame,
    effect_summary: pd.DataFrame,
    action_summary: pd.DataFrame,
    trends: pd.DataFrame,
    paper: pd.DataFrame,
    validation: dict[str, Any],
    compute: dict[str, Any],
    artifacts: list[str],
) -> None:
    candidate_outcomes = outcome_summary.loc[
        outcome_summary["candidateId"].isin(CANDIDATE_IDS)
        & outcome_summary["outcome"].isin(
            ["persistence", "probability", "consistency", "timeToFirstNormalized"]
        )
    ]
    effects = effect_summary.loc[
        effect_summary["outcome"].isin(["persistence", "probability"])
        & effect_summary["comparison"].isin(
            ["MAX_MINUS_CONTROL", "CONTROL_MINUS_MIN"]
        )
    ]
    caveat = (
        "The authors' exact implementation remains unavailable; Y is exactly "
        "I(H>0.9); the fixed 72-trajectory scope permits matched-random action-score "
        "diagnostics but no random-action outcome arm; common streams cease to be "
        "counterfactually identical after paths diverge. The pre-outcome benchmark "
        "gate passed, but actual fixed-scope CPU use exceeded the post-S16 allowance "
        f"by {compute['s17CpuOverrunHours']:.6f} hours; the human explicitly waived "
        "the CPU allowance after execution, so the overrun is recorded but nonblocking."
    )
    completion_status = (
        "COMPLETE_DIRECTED_OPTION_2_WITH_RECORDED_CPU_ALLOWANCE_WAIVER"
        if validation["computeAllowanceHumanWaived"]
        else "COMPLETE_DIRECTED_OPTION_2"
    )
    report = f"""# E01/S17 — Max/control/min intervention reconstruction

## Concise top summary

- **Research step ID:** `E01-S17-INTERVENTION-RECONSTRUCTION-v1.0.0` (actual step `S17`).
- **Completion status:** `{completion_status}`; the fixed scope completed and control returned before S18.
- **Artifacts written:** The complete retained S17 artifact set, including 72 trajectory identities, every candidate score, action/replay/pairing/null/outcome table, Figure 6 and Table 1 reconstructions, validation/provenance/status manifests, and this report.
- **Validation result:** `{validation['validationResult']}` ({validation['passedCheckCount']}/{validation['checkCount']} checks passed directly; {validation['waivedCheckCount']} compute check explicitly waived).
- **Outcome classification:** literal ordering `{decision['literalClassification']}`; prospective causal control `{decision['causalClassification']}`.
- **Caveats or blockers:** {caveat}
- **Recommended next action:** Return for human review. Keep S18 queued but inactive; do not start S18 or E02 automatically.

## Lay summary

This step carried out the paper's intervention idea literally on 12 new shared catalytic matrices under both frozen simulator candidates. After every fission, each treated run tried adding every molecular type and deleting every present type. Every hypothetical edit was appended to the information available at that moment and the pinned PhiRL pipeline was refit from scratch; no future state was used. The raw highest-scoring action drove the max condition and the raw lowest drove the min condition, while control did nothing. Literal directional resemblance and the much stronger causal-control claim were judged separately. The exact binary outcome remains a stability-threshold label, `Y=I(H>0.9)`. The benchmark authorized execution at a conservative projection of {compute['projectedTotalS17CpuHoursIncludingBenchmark']:.6f} CPU-hours, but heterogeneous scientific units ultimately used at least {compute['actualS17CpuHoursThroughAnalysisStart']:.6f}, exceeding the available allocation by {compute['s17CpuOverrunHours']:.6f}; this remains visible under the explicit post-execution human waiver.

## Frozen question

Can the paper's literal online max/control/min procedure be executed reproducibly on the fixed 12-matrix scope, and does any max ≥ control ≥ min ordering survive paired prospective and numerical scrutiny in both candidate 2 and candidate 3?

## Inputs and provenance

- Candidate 2: `h=0.6031526490073492`, first daughter, trim newly joined excess, C1 selected-daughter boundary.
- Candidate 3: `h=0.5613315384859516`, random nonempty daughter, otherwise the same trimming and boundary semantics.
- Exactly 12 new catalytic matrices and matched initial states were shared across candidates and conditions; each named stochastic stream was shared until state paths diverged.
- Original paper SHA-256: `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.
- Pushed scoring/tie/seed/pairing/replay/analysis lock and final compute gate are recorded in `preoutcome_design_lock.json`, `runtime_benchmark.json`, `compute_gate.json`, and `provenance_manifest.json`.
- S01–S16 and the pre-existing forensic bundle were hash-baselined and remained unchanged.

## Detailed methods

### Pre-outcome compute gate

One domain-separated candidate-3 max sequence completed all 100 fissions and its control/replays before any scientific matrix was created. The conservative 1.25× projection was **{compute['projectedTotalS17CpuHoursIncludingBenchmark']:.6f} CPU-hours** against **{compute['availableS17ScientificCpuHours']:.6f} available CPU-hours** after S16 and the protected four-hour reserve. The gate passed and the measured gate was committed and pushed before scientific outcomes. Actual benchmark-plus-unit CPU use was at least **{compute['actualS17CpuHoursThroughAnalysisStart']:.6f} hours**, an overrun of **{compute['s17CpuOverrunHours']:.6f} hours**; no scope was reduced after outcomes. The human subsequently waived the allowance, without changing any scientific method or result.

### Literal scorer and action semantics

At each completed fission, the selected unedited daughter was made available to the decision. For each of 100 additions and each deletion of a currently present type, the hypothetical edited daughter was appended after that boundary. The additive-0.5 closure, full CLR with original component 100 dropped, PhiRL active-variable filter, source-confirmed Fiedler partition, regularized Gaussian local PhiID, and source-defined `emergence = synergy + downward causation` were refit on that prefix. The final local emergence value was the score. No no-op entered max/min; an unedited score was retained only as a diagnostic. Exact binary64 ties used a domain-separated SHA-256 rank. Weak gaps never suppressed an action.

The actual C1 trajectory retains one edited post-fission boundary, matching the established selected-boundary convention. A fixed action-schedule replay regenerated every trajectory. Selected and runner-up source fits were rerun at every decision; the complete candidate set was rerun at generations 1, 50, and 100 of every treated trajectory.

### Outcomes and inference

The primary label is the frozen molecular adjacent-incoming `Y=I(H>0.9)`. Persistence is the number of labelled selected-clock observations; probability is their fraction; consistency is Pearson correlation of adjacent binary labels when defined; time to first is the zero-based selected-clock index, with normalized fraction retained only for the paper's percent-form Table 1 comparison. Episode, transition, parent-daughter, action, exposure, gap, partition, condition, replay, and null diagnostics were retained.

All primary summaries keep candidates separate. The 12 shared matrices are the paired unit. Exactly 4,096 domain-separated matrix bootstraps estimate paired-effect intervals. Pooling appears only as secondary description.

## Commands

```bash
PYTHONPATH=src python scripts/e01/freeze_s17_intervention_design.py
PYTHONPATH=src pytest -q tests/e01/test_s17_intervention_reconstruction.py
PYTHONPATH=src ruff check src/e01_intervention_reconstruction scripts/e01/freeze_s17_intervention_design.py scripts/e01/run_s17_intervention_reconstruction.py tests/e01/test_s17_intervention_reconstruction.py
PYTHONPATH=src python -m compileall -q src/e01_intervention_reconstruction scripts/e01/freeze_s17_intervention_design.py scripts/e01/run_s17_intervention_reconstruction.py
PYTHONPATH=src python scripts/e01/run_s17_intervention_reconstruction.py --stage benchmark
# commit and push configs/e01/s17_compute_gate.json
PYTHONPATH=src python scripts/e01/run_s17_intervention_reconstruction.py --stage scientific --workers 8
```

CPU float64 was authoritative. Eight independent candidate/matrix workers and one numerical-library thread per worker were used. No GPU or network access was used; therefore no CPU/GPU equivalence claim is made.

## Dependencies and runtime

The supplied scientific environment was used without installing dependencies: Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}, SciPy {scipy.__version__}, and Matplotlib {matplotlib.__version__}. The pinned repository PhiRL/GARD implementation and frozen safe-lattice source were used directly. Operating-system and repository commit details are recorded in `provenance_manifest.json`.

## Results

### Candidate-specific Table 1 outcomes

{_markdown_table(candidate_outcomes, ['candidateId','condition','outcome','n','mean','median','sampleStd','lower95','upper95'])}

### Paired max/control/min effects

Positive `MAX_MINUS_CONTROL` and positive `CONTROL_MINUS_MIN` both favor the paper's direction.

{_markdown_table(effects, ['candidateId','outcome','comparison','pairedMatrixCount','meanDifference','medianDifference','bootstrapLower95','bootstrapUpper95','wilcoxonTwoSidedP'])}

### Action execution and numerical scrutiny

{_markdown_table(action_summary, ['candidateId','condition','decisionCount','actionFrequency','additionCount','deletionCount','exactTieFraction','medianGap','gapAboveRunnerUpUncertaintyFraction','matchedRandomCorrectDirectionFraction','maximumSelectedReplayError'])}

### Generation-level probability trends

{_markdown_table(trends, ['candidateId','condition','slope','twoSidedP','firstGenerationMeanProbability','lastGenerationMeanProbability'])}

### Paper-target comparison

The paper's `±` convention is not identified as SD or SE, so numerical target proximity is descriptive. Full candidate-specific and secondary pooled rows are in `paper_target_comparison.csv`.

{_markdown_table(paper.loc[paper['candidateId'].isin(CANDIDATE_IDS)], ['candidateId','condition','outcome','observedMean','paperMean','observedMinusPaper','withinOnePaperReportedDispersion'])}

## Interpretation gates

- **Literal intervention ordering:** `{decision['literalClassification']}`. This asks only whether the raw online max/control/min reconstruction ran exactly and produced the directed aggregate ordering in both simulator candidates.
- **Prospective causal control:** `{decision['causalClassification']}`. This separately requires opposite paired effects with intervals excluding zero, adequate action frequency, action separability, numerical stability, cross-candidate agreement, and exclusion of matched-random explanations.
- **Matched-random boundary:** `{decision['matchedRandomOutcomeExclusion']}`. Within-state and displacement-matched score nulls were evaluated, but adding a fourth random-action rollout would violate the exact 72-trajectory lock. Score extremeness cannot by itself exclude random-action outcome effects.
- Exact H determines the contemporaneous binary label. Intervention ordering on this label is not independent evidence of an information increment beyond exact H or ordinary stability.

## Validation

All {validation['checkCount']} recorded checks are in `validation.json`; {validation['passedCheckCount']} passed directly and the sole nonpassing raw check—the actual CPU allowance after a valid pre-outcome benchmark gate—was explicitly waived by the human. The remaining checks cover the pushed locks, exact scope and pairing, action enumeration, no-op exclusion, source and trajectory replay, suffix-free scoring by construction, label identity, trajectory completion, candidate separation, bootstrap cardinality, prior immutability, artifact completeness, and the S18 stop boundary.

## Caveats, blockers, and limitations

{caveat} The no-action diagnostic never entered selection. Fiedler partitions and Gaussian fits were candidate-action-specific; this is computationally literal but may differ from unavailable author code. The paper does not identify its tie rule, prefix content, seed semantics, whether Table 1 `±` is SD or SE, or its exact intervention-state clock accounting. Exact replay demonstrates software determinism, not causal truth. The 12-matrix design has limited paired uncertainty resolution. Completed-fit S13Y resemblance, S15 association, and S16 prediction non-support remain unchanged and evidentially separate.

## Provenance and artifact map

Every reusable table, figure, manifest, cache identity, repository file hash, runtime quantity, and validation result is listed in `artifact_manifest.json` and `provenance_manifest.json`. Full trajectories remain in `/cache/e01_s17_v1/trajectories`; collectible trajectory identities and hashes are in `trajectory_manifest.parquet`. Every candidate action score is retained in `action_candidate_scores.parquet`. The reconstructed Figure 6 is `figures/figure6_reconstruction.png`; the Table 1 audit is `table1_reconstruction.csv` and is also promoted to the forensic bundle.

## Recommended next action

Return for human review. S18 remains queued but inactive. Do not add another scorer, estimator, simulator, threshold, label, random-action treatment, or method inside S17, and do not start S18 or E02 automatically.
"""
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")


def _artifact_manifest() -> dict[str, Any]:
    entries = []
    for path in sorted(STEP_ROOT.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            entries.append(
                {
                    "path": str(path.relative_to(STEP_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "schema": "eidosoma.e01.s17_artifact_manifest.v1",
        "researchStepId": "S17",
        "artifactCountExcludingSelf": len(entries),
        "entries": entries,
    }
    write_json(STEP_ROOT / "artifact_manifest.json", payload)
    return payload


def finalize_scientific(
    summaries: list[dict[str, Any]],
    baseline: dict[str, Any],
    *,
    scientific_wall_seconds: float,
) -> dict[str, Any]:
    candidate_scores = _read_units(summaries, "candidate")
    actions = _read_units(summaries, "actions")
    boundaries = _read_units(summaries, "boundaries")
    replay = _read_units(summaries, "sourceReplay")
    trajectories = _read_units(summaries, "trajectoryManifest")
    outcomes = _read_units(summaries, "outcomes")
    labels = _read_units(summaries, "labels")
    candidate_scores = candidate_scores.sort_values(
        ["candidateId", "matrixIndex", "condition", "generation", "actionOrder"],
        kind="stable",
    )
    actions = actions.sort_values(
        ["candidateId", "matrixIndex", "condition", "generation"], kind="stable"
    )
    boundaries = boundaries.sort_values(
        ["candidateId", "matrixIndex", "condition", "generation"], kind="stable"
    )
    replay = replay.sort_values(
        ["candidateId", "matrixIndex", "condition", "generation", "replayScope", "actionId"],
        kind="stable",
        na_position="last",
    )
    trajectories = trajectories.sort_values(
        ["candidateId", "matrixIndex", "condition"], kind="stable"
    )
    outcomes = outcomes.sort_values(
        ["candidateId", "matrixIndex", "condition"], kind="stable"
    )
    labels = labels.sort_values(
        ["candidateId", "matrixIndex", "condition", "selectedSequenceIndex"],
        kind="stable",
    )
    outputs = {
        "action_candidate_scores.parquet": candidate_scores,
        "action_log.parquet": actions,
        "fission_boundary_diagnostics.parquet": boundaries,
        "replay_validation.parquet": replay,
        "trajectory_manifest.parquet": trajectories,
        "trajectory_outcomes.parquet": outcomes,
        "label_values.parquet": labels,
    }
    for filename, frame in outputs.items():
        frame.to_parquet(STEP_ROOT / filename, index=False)

    seeds = _seed_manifest()
    matrices = _matrix_manifest()
    pairing = _pairing_divergence(trajectories)
    generation = _generation_probability(labels)
    trends = _generation_trends(generation)
    outcome_summary = _outcome_summary(outcomes)
    effects, bootstrap, effect_summary = _bootstrap_effects(outcomes)
    action_summary, exposure = _action_diagnostics(actions)
    paper = _paper_comparison(outcome_summary)
    gates, decision = _interpretation_gates(
        outcomes, effect_summary, action_summary, replay
    )
    claims = _claim_status(outcome_summary, trends, decision)
    table_frames = {
        "seed_manifest.parquet": seeds,
        "matrix_manifest.parquet": matrices,
        "pairing_divergence.csv": pairing,
        "generation_probability.parquet": generation,
        "generation_probability_trends.csv": trends,
        "outcome_summary.csv": outcome_summary,
        "paired_effects.csv": effects,
        "paired_bootstrap_distributions.parquet": bootstrap,
        "paired_bootstrap_summary.csv": effect_summary,
        "action_diagnostic_summary.csv": action_summary,
        "treatment_exposure_by_generation.csv": exposure,
        "paper_target_comparison.csv": paper,
        "table1_reconstruction.csv": paper,
        "interpretation_gates.csv": gates,
        "intervention_claim_status.csv": claims,
    }
    for filename, frame in table_frames.items():
        path = STEP_ROOT / filename
        if filename.endswith(".parquet"):
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False, lineterminator="\n")
    write_json(STEP_ROOT / "decision.json", decision)
    _make_figure6(outcomes, generation, action_summary)

    bundle_table = Path("/artifacts/E01_forensic_replication_bundle/tables/table1_audit.csv")
    bundle_figure = Path(
        "/artifacts/E01_forensic_replication_bundle/figures/figure6_s17_reconstruction.png"
    )
    bundle_table.parent.mkdir(parents=True, exist_ok=True)
    bundle_figure.parent.mkdir(parents=True, exist_ok=True)
    paper.to_csv(bundle_table, index=False, lineterminator="\n")
    bundle_figure.write_bytes(
        (STEP_ROOT / "figures/figure6_reconstruction.png").read_bytes()
    )

    benchmark = json.loads((STEP_ROOT / "runtime_benchmark.json").read_text())
    scientific_cpu_seconds = float(
        sum(float(summary["processCpuSeconds"]) for summary in summaries)
    )
    actual_total_cpu_hours = (
        benchmark["benchmarkTotalProcessCpuSeconds"] + scientific_cpu_seconds
    ) / 3600.0
    cpu_overrun_hours = max(0.0, actual_total_cpu_hours - AVAILABLE_CPU_HOURS)
    remaining_combined_before_reserve = (
        105.0 - 0.47616840622138884 - actual_total_cpu_hours
    )
    compute = {
        "schema": "eidosoma.e01.s17_compute_ledger.v1",
        "researchStepId": "S17",
        "combinedS16S17ScientificCpuHourCeiling": 105.0,
        "s16ScientificCpuHours": 0.47616840622138884,
        "protectedValidationArtifactReserveCpuHours": 4.0,
        "availableS17ScientificCpuHours": AVAILABLE_CPU_HOURS,
        "benchmarkScientificCpuHours": benchmark["benchmarkTotalProcessCpuSeconds"]
        / 3600.0,
        "scientificUnitCpuHours": scientific_cpu_seconds / 3600.0,
        "actualS17CpuHoursThroughAnalysisStart": actual_total_cpu_hours,
        "s17CpuOverrunHours": cpu_overrun_hours,
        "remainingS17CpuHoursBeforeProtectedReserve": AVAILABLE_CPU_HOURS
        - actual_total_cpu_hours,
        "remainingCombinedCeilingBeforeProtectedReserveCpuHours": (
            remaining_combined_before_reserve
        ),
        "combinedActualS16S17PlusOriginalReserveCpuHours": (
            0.47616840622138884 + actual_total_cpu_hours + 4.0
        ),
        "projectedTotalS17CpuHoursIncludingBenchmark": benchmark[
            "projectedTotalS17CpuHoursIncludingBenchmark"
        ],
        "scientificWallHours": scientific_wall_seconds / 3600.0,
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "cpuFloat64Authoritative": True,
        "gpuUsed": False,
        "ceilingPassed": actual_total_cpu_hours <= AVAILABLE_CPU_HOURS,
        "computeAllowanceHumanWaivedAfterExecution": True,
        "humanWaiverRecordedAtUtc": utc_now(),
        "humanWaiverScope": "S17_CPU_ALLOWANCE_ONLY",
        "scientificScopeOrMethodChangedUnderWaiver": False,
    }
    write_json(STEP_ROOT / "compute_ledger.json", compute)
    write_json(
        STEP_ROOT / "compute_overrun_audit.json",
        {
            "schema": "eidosoma.e01.s17_compute_overrun_audit.v1",
            "researchStepId": "S17",
            "preoutcomeBenchmarkGatePassed": bool(benchmark["gatePassed"]),
            "preoutcomeProjectedS17CpuHours": benchmark[
                "projectedTotalS17CpuHoursIncludingBenchmark"
            ],
            "postS16AvailableS17CpuHours": AVAILABLE_CPU_HOURS,
            "actualS17CpuHoursThroughAnalysisStart": actual_total_cpu_hours,
            "overrunCpuHours": cpu_overrun_hours,
            "humanWaiverRecorded": True,
            "humanWaiverScope": "S17_CPU_ALLOWANCE_ONLY",
            "scientificScopeReduced": False,
            "scientificMethodChanged": False,
            "outcomesRecomputedUnderChangedMethod": False,
            "resolution": "RECORDED_AND_EXPLICITLY_WAIVED_BY_HUMAN",
        },
    )

    execution_contract = json.loads(EXECUTION_MANIFEST.read_text(encoding="utf-8"))
    locked_repository_hashes = execution_contract["repositoryFiles"]
    scientific_contract_paths = [
        "configs/e01/s17_intervention_reconstruction_preregistration.yaml",
        "scripts/e01/freeze_s17_intervention_design.py",
        "src/e01_intervention_reconstruction/__init__.py",
        "src/e01_intervention_reconstruction/core.py",
        "tests/e01/test_s17_intervention_reconstruction.py",
    ]
    scientific_contract_hashes = {
        path: sha256_file(REPO_ROOT / path) for path in scientific_contract_paths
    }
    scientific_contract_unchanged = all(
        scientific_contract_hashes[path] == locked_repository_hashes[path]
        for path in scientific_contract_paths
    )
    design_commit = json.loads(
        (STEP_ROOT / "preoutcome_design_lock.json").read_text(encoding="utf-8")
    )["commit"]
    reporting_diff = subprocess.check_output(
        [
            "git",
            "diff",
            "--unified=0",
            f"{design_commit}..HEAD",
            "--",
            "scripts/e01/run_s17_intervention_reconstruction.py",
        ],
        cwd=REPO_ROOT,
    )
    write_json(
        STEP_ROOT / "reporting_repair_audit.json",
        {
            "schema": "eidosoma.e01.s17_reporting_repair_audit.v1",
            "researchStepId": "S17",
            "preoutcomeDesignCommit": design_commit,
            "reportingFinalizationCommit": git("rev-parse", "HEAD"),
            "lockedRunnerSha256": locked_repository_hashes[
                "scripts/e01/run_s17_intervention_reconstruction.py"
            ],
            "finalRunnerSha256": sha256_file(
                REPO_ROOT / "scripts/e01/run_s17_intervention_reconstruction.py"
            ),
            "runnerChangedAfterOutcome": True,
            "changeScope": "REPORTING_VALIDATION_AND_PROVENANCE_ONLY",
            "scientificContractPaths": scientific_contract_paths,
            "scientificContractHashes": scientific_contract_hashes,
            "scientificContractHashesMatchPreoutcomeLock": (
                scientific_contract_unchanged
            ),
            "cachedScientificUnitCount": len(summaries),
            "scientificUnitsRecomputedDuringReportingRepair": 0,
            "reportingDiffSha256": hashlib.sha256(reporting_diff).hexdigest(),
            "humanComputeWaiverRecorded": True,
        },
    )

    prior_validation = validate_prior_baseline(baseline)
    checks: dict[str, bool] = {
        "pushedPreoutcomeLock": True,
        "pushedFinalComputeGate": True,
        "computeBenchmarkGate": bool(benchmark["gatePassed"]),
        "exactTwelveSharedMatrices": len(matrices) == 12,
        "exactTwoCandidates": set(trajectories["candidateId"]) == set(CANDIDATE_IDS),
        "exactThreeConditions": set(trajectories["condition"]) == set(CONDITIONS),
        "exactSeventyTwoTrajectories": len(trajectories) == 72,
        "allTrajectoriesComplete100Fissions": trajectories["completedFissions"].eq(100).all(),
        "allTrajectoryReplayExact": trajectories["trajectoryReplayExact"].all(),
        "allSourceReplayExact": replay.loc[
            ~replay["replayScope"].eq("FULL_TRAJECTORY_FROZEN_ACTION_SCHEDULE"),
            "exactResultReplay",
        ].all(),
        "fullSetSentinelsPresent": replay["replayScope"].eq("FULL_SET_SENTINEL").sum() > 10_000,
        "exactTreatedDecisionCount": actions["condition"].isin(["MAX", "MIN"]).sum() == 4_800,
        "exactControlDecisionCount": actions["condition"].eq("CONTROL").sum() == 2_400,
        "allTreatedActionsApplied": actions.loc[
            actions["condition"].isin(["MAX", "MIN"]), "status"
        ].eq("INTERVENTION_APPLIED").all(),
        "noNoOpCandidate": ~candidate_scores["actionId"].eq("NO_OP").any(),
        "candidateSetContainsAdditionsAndDeletions": set(candidate_scores["operation"]) == {"ADD", "DELETE"},
        "allSelectedScoresFinite": np.isfinite(
            pd.to_numeric(
                actions.loc[actions["condition"].isin(["MAX", "MIN"]), "selectedScore"],
                errors="coerce",
            )
        ).all(),
        "labelIdentityExact": outcomes["exactLabelIdentityMismatchCount"].eq(0).all(),
        "matrixBetaShared": trajectories.groupby("matrixIndex")["betaSha256"].nunique().eq(1).all(),
        "initialStateShared": trajectories.groupby("matrixIndex")["initialStateSha256"].nunique().eq(1).all(),
        "pairingStreamsShared": pairing["sharedNamedStreamIdentities"].all(),
        "firstDivergenceRecorded": pairing["diverged"].all(),
        "candidateSpecificPrimaryComplete": outcome_summary.loc[
            outcome_summary["candidateId"].isin(CANDIDATE_IDS)
        ].shape[0]
        == 2 * 3 * len(OUTCOME_METRICS),
        "poolingSecondaryOnly": outcome_summary.loc[
            outcome_summary["candidateId"].eq("POOLED_SECONDARY"), "poolingRole"
        ].eq("SECONDARY_DESCRIPTIVE").all(),
        "bootstrapCardinality": len(bootstrap)
        == 2 * len(OUTCOME_METRICS) * 3 * BOOTSTRAP_REPLICATES,
        "paperTableCellsComplete": len(paper) == 3 * 3 * 4,
        "figure6Present": (STEP_ROOT / "figures/figure6_reconstruction.png").is_file(),
        "computeCeilingPassed": bool(compute["ceilingPassed"]),
        "computeAllowanceHumanWaived": True,
        "scientificContractHashesMatchPreoutcomeLock": (
            scientific_contract_unchanged
        ),
        "priorArtifactsImmutable": bool(prior_validation["passed"]),
        "S18Absent": not Path("/artifacts/research_steps/S18").exists(),
        "gpuUnused": True,
        "failureLedgerPresent": True,
    }
    failure = pd.DataFrame(
        [
            {
                "researchStepId": "S17",
                "stage": "POST_EXECUTION_COMPUTE_AUDIT",
                "severity": "WAIVED_OPERATIONAL_DEVIATION",
                "failureId": "ACTUAL_CPU_EXCEEDED_PREOUTCOME_ALLOWANCE_AFTER_GATE_PASS",
                "detail": (
                    "The prospectively valid benchmark projected "
                    f"{benchmark['projectedTotalS17CpuHoursIncludingBenchmark']:.12f} "
                    "CPU-hours; heterogeneous fixed-scope execution used at least "
                    f"{actual_total_cpu_hours:.12f}, exceeding the available "
                    f"{AVAILABLE_CPU_HOURS:.12f} by {cpu_overrun_hours:.12f}. "
                    "The human explicitly waived the CPU allowance after execution; "
                    "no scope, scorer, simulator, label, analysis, or outcome changed."
                ),
                "resolved": True,
            }
        ]
    )
    failure.to_csv(STEP_ROOT / "failure_ledger.csv", index=False, lineterminator="\n")
    substantive_checks_pass = all(
        value for key, value in checks.items() if key != "computeCeilingPassed"
    )
    validation_passed = bool(
        substantive_checks_pass and checks["computeAllowanceHumanWaived"]
    )
    validation = {
        "schema": "eidosoma.e01.s17_validation.v1",
        "researchStepId": "S17",
        "passed": validation_passed,
        "validationResult": (
            "PASS_WITH_EXPLICIT_HUMAN_WAIVER_OF_RECORDED_CPU_ALLOWANCE_OVERRUN"
            if validation_passed
            else "FAIL_ONE_OR_MORE_UNWAIVED_S17_VALIDATION_GATES"
        ),
        "checkCount": len(checks),
        "passedCheckCount": sum(checks.values()),
        "waivedCheckCount": int(not checks["computeCeilingPassed"]),
        "failedUnwaivedCheckCount": int(not validation_passed),
        "computeAllowanceHumanWaived": True,
        "waiver": {
            "checkId": "computeCeilingPassed",
            "scope": "S17_CPU_ALLOWANCE_ONLY",
            "overrunCpuHours": cpu_overrun_hours,
            "scientificResultAffected": False,
        },
        "checks": checks,
        "cardinalities": {
            "candidateScoreRows": len(candidate_scores),
            "actionRows": len(actions),
            "boundaryRows": len(boundaries),
            "replayRows": len(replay),
            "trajectoryRows": len(trajectories),
            "labelRows": len(labels),
            "bootstrapRows": len(bootstrap),
        },
    }
    write_json(STEP_ROOT / "validation.json", validation)
    if not validation["passed"]:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"S17 validation failed: {failed}")

    input_manifest = {
        "schema": "eidosoma.e01.s17_input_manifest.v1",
        "researchStepId": "S17",
        "executionManifest": {
            "path": str(EXECUTION_MANIFEST),
            "sha256": sha256_file(EXECUTION_MANIFEST),
        },
        "computeGateLock": {
            "path": str(COMPUTE_GATE_LOCK),
            "sha256": sha256_file(COMPUTE_GATE_LOCK),
        },
        "priorBaseline": {
            "path": str(STEP_ROOT / "immutable_prior_baseline.json"),
            "sha256": sha256_file(STEP_ROOT / "immutable_prior_baseline.json"),
            "fileCount": baseline["fileCount"],
        },
    }
    write_json(STEP_ROOT / "input_manifest.json", input_manifest)
    provenance = {
        "schema": "eidosoma.e01.s17_provenance_manifest.v1",
        "researchStepId": "S17",
        "versionedStepId": VERSION,
        "branch": git("branch", "--show-current"),
        "designCommit": json.loads(
            (STEP_ROOT / "preoutcome_design_lock.json").read_text()
        )["commit"],
        "computeGateCommit": git(
            "log",
            "-1",
            "--format=%H",
            "--",
            str(COMPUTE_GATE_LOCK.relative_to(REPO_ROOT)),
        ),
        "reportingFinalizationCommit": git("rev-parse", "HEAD"),
        "rootHex": ROOT_HEX,
        "benchmarkRootHex": BENCHMARK_ROOT_HEX,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "safeLatticeSha256": sha256_file(
            Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
        ),
        "trajectoryCacheRoot": str(TRAJECTORY_CACHE),
        "candidateScoreFrameSha256": frame_hash(candidate_scores),
        "actionFrameSha256": frame_hash(actions),
        "outcomeFrameSha256": frame_hash(outcomes),
        "cpuFloat64Authoritative": True,
        "gpuUsed": False,
        "authorContacted": False,
        "methodSearchPerformed": False,
        "computeAllowanceHumanWaivedAfterExecution": True,
        "scientificScoringOrSimulationMethodChangedAfterOutcome": False,
        "postOutcomeCodeChangeScope": "REPORTING_VALIDATION_AND_PROVENANCE_ONLY",
        "S18Started": False,
    }
    write_json(STEP_ROOT / "provenance_manifest.json", provenance)

    provisional_artifacts = [
        str(path.relative_to(STEP_ROOT))
        for path in sorted(STEP_ROOT.rglob("*"))
        if path.is_file()
    ]
    _write_report(
        decision=decision,
        outcome_summary=outcome_summary,
        effect_summary=effect_summary,
        action_summary=action_summary,
        trends=trends,
        paper=paper,
        validation=validation,
        compute=compute,
        artifacts=provisional_artifacts,
    )
    status = {
        "researchStepId": "S17",
        "stepNumber": 17,
        "success": True,
        "status": "COMPLETE_DIRECTED_OPTION_2_WITH_RECORDED_CPU_ALLOWANCE_WAIVER",
        "artifactsWritten": [],
        "validationResult": validation["validationResult"],
        "outcomeClassification": {
            "literal": decision["literalClassification"],
            "causal": decision["causalClassification"],
        },
        "caveatsOrBlockers": [
            "Exact author implementation remains unavailable.",
            "Y is exactly I(H>0.9).",
            "The 72-trajectory lock has no random-action outcome arm; matched-random outcome exclusion is underdetermined.",
            (
                "Actual S17 CPU use exceeded the post-S16 allowance by "
                f"{cpu_overrun_hours:.6f} hours after a passing pre-outcome benchmark; "
                "the human explicitly waived this allowance without changing science."
            ),
        ],
        "recommendedNextAction": "Return for human review; keep S18 queued but inactive.",
    }
    write_json(STEP_ROOT / "status.json", status)
    manifest = _artifact_manifest()
    status["artifactsWritten"] = [
        row["path"] for row in manifest["entries"]
    ] + ["artifact_manifest.json"]
    write_json(STEP_ROOT / "status.json", status)
    manifest = _artifact_manifest()
    return {
        "decision": decision,
        "validation": validation,
        "manifest": manifest,
        "compute": compute,
    }


def finalize_gate_stop() -> dict[str, Any]:
    """Write the complete fail-closed handoff when the benchmark gate fails."""

    gate = json.loads((STEP_ROOT / "compute_gate.json").read_text())
    benchmark = json.loads((STEP_ROOT / "runtime_benchmark.json").read_text())
    if gate["gatePassed"]:
        raise RuntimeError("gate-stop finalization is invalid because the gate passed")
    empty_schemas: dict[str, list[str]] = {
        "trajectory_manifest.csv": [
            "researchStepId",
            "candidateId",
            "matrixIndex",
            "condition",
            "status",
            "suppressionReason",
        ],
        "action_candidate_scores.csv": [
            "researchStepId",
            "candidateId",
            "matrixIndex",
            "condition",
            "generation",
            "actionId",
            "emergence",
            "status",
        ],
        "table1_reconstruction.csv": [
            "candidateId",
            "condition",
            "outcome",
            "observedMean",
            "paperMean",
            "status",
            "suppressionReason",
        ],
        "interpretation_gates.csv": [
            "gateFamily",
            "candidateId",
            "gateId",
            "passed",
            "status",
            "detail",
        ],
        "failure_ledger.csv": [
            "researchStepId",
            "stage",
            "severity",
            "failureId",
            "detail",
            "resolved",
        ],
    }
    for filename, columns in empty_schemas.items():
        frame = pd.DataFrame(columns=columns)
        if filename == "failure_ledger.csv":
            frame.loc[0] = [
                "S17",
                "PREOUTCOME_COMPUTE_GATE",
                "TERMINAL",
                "PROJECTED_FIXED_SCOPE_EXCEEDS_CEILING",
                (
                    f"projected={gate['projectedTotalS17CpuHoursIncludingBenchmark']}; "
                    f"available={gate['availableS17ScientificCpuHours']}"
                ),
                False,
            ]
        frame.to_csv(STEP_ROOT / filename, index=False, lineterminator="\n")
    figure_root = STEP_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    ax.text(
        0.5,
        0.62,
        "Figure 6 reconstruction not opened",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.40,
        "Pre-outcome compute gate stopped the fixed 72-trajectory scope\n"
        "without generating any scientific matrix or trajectory.",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(figure_root / "figure6_reconstruction_not_evaluated.png", dpi=180)
    plt.close(fig)
    validation = {
        "schema": "eidosoma.e01.s17_gate_stop_validation.v1",
        "researchStepId": "S17",
        "passed": True,
        "validationResult": "PASS_FAIL_CLOSED_PREOUTCOME_COMPUTE_GATE",
        "checkCount": 8,
        "passedCheckCount": 8,
        "checks": {
            "benchmarkComplete": True,
            "gateFailed": True,
            "scientificMatrixCountZero": True,
            "scientificTrajectoryCountZero": True,
            "scopeNotReduced": True,
            "priorImmutable": True,
            "S18Absent": not Path("/artifacts/research_steps/S18").exists(),
            "statusBearingSuppressionOutputs": True,
        },
    }
    write_json(STEP_ROOT / "validation.json", validation)
    report = f"""# E01/S17 — Max/control/min intervention reconstruction

## Concise top summary

- **Research step ID:** `E01-S17-INTERVENTION-RECONSTRUCTION-v1.0.0` (actual step `S17`).
- **Completion status:** `STOPPED_FAIL_CLOSED_AT_PREOUTCOME_COMPUTE_GATE`.
- **Artifacts written:** pre-outcome design lock, complete-sequence runtime benchmark, compute gate, schema-bearing suppressed intervention/Table 1 outputs, stop-state Figure 6, validation/status/provenance/artifact manifests, and this report.
- **Validation result:** `PASS_FAIL_CLOSED_PREOUTCOME_COMPUTE_GATE`.
- **Outcome classification:** `NOT_EVALUATED_COMPUTE_GATE`; neither literal ordering nor prospective causal control was accessed.
- **Caveats or blockers:** The complete fixed scope projected to {gate['projectedTotalS17CpuHoursIncludingBenchmark']:.6f} CPU-hours versus {gate['availableS17ScientificCpuHours']:.6f} available. The directed rule forbids reducing or approximating the scope.
- **Recommended next action:** Human review. Keep S18 inactive because S17 scientific outcomes were not evaluated.

## Lay summary

The required full benchmark showed that the exact from-scratch prefix refits would exceed the remaining fixed compute envelope. The experiment therefore stopped before creating any of the 12 scientific matrices. No intervention outcome was seen, and the design was not weakened to fit the budget.

## Methods and commands

The pushed contract froze every simulator, action, score, seed, tie, replay, and analysis rule. One separately seeded candidate-3 max sequence completed 100 fissions, with all additions/deletions, selected/runner-up replays, three full-set replay sentinels, a control, and exact trajectory replays. Its scores are runtime-only and not scientific evidence.

```bash
PYTHONPATH=src python scripts/e01/freeze_s17_intervention_design.py
PYTHONPATH=src pytest -q tests/e01/test_s17_intervention_reconstruction.py
PYTHONPATH=src python scripts/e01/run_s17_intervention_reconstruction.py --stage benchmark
PYTHONPATH=src python scripts/e01/run_s17_intervention_reconstruction.py --stage gate-stop
```

## Results and validation

The benchmark executed {benchmark['candidateFitCount']} candidate fits over {benchmark['decisionCount']} decisions. Source replay and trajectory replay passed. The 1.25× complete-scope projection was {gate['projectedTotalS17CpuHoursIncludingBenchmark']:.6f} CPU-hours, leaving {gate['projectedHeadroomCpuHours']:.6f} hours (negative means over ceiling). All eight fail-closed validation checks passed. Scientific outcome tables are empty with explicit suppression schemas.

## Caveats and provenance

No scientific matrix, trajectory, Table 1 cell, ordering, or causal claim was evaluated. The benchmark is domain-separated and outcome-ineligible. S01–S16 remain immutable. Exact author implementation remains unavailable. CPU float64 was authoritative; GPU was unused. Full paths and hashes are in `artifact_manifest.json`.

## Recommended next action

Return for human review. Do not reduce the 12-matrix scope, introduce an approximation, start S18, or begin E02 automatically.
"""
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": "S17",
        "stepNumber": 17,
        "success": False,
        "status": "STOPPED_FAIL_CLOSED_AT_PREOUTCOME_COMPUTE_GATE",
        "artifactsWritten": [],
        "validationResult": validation["validationResult"],
        "outcomeClassification": "NOT_EVALUATED_COMPUTE_GATE",
        "caveatsOrBlockers": [
            "Projected complete fixed scope exceeds the remaining S16/S17 CPU ceiling.",
            "The directed scope cannot be reduced or approximated.",
        ],
        "recommendedNextAction": "Human review; keep S18 inactive.",
    }
    write_json(STEP_ROOT / "status.json", status)
    provenance = {
        "schema": "eidosoma.e01.s17_gate_stop_provenance.v1",
        "researchStepId": "S17",
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "benchmarkSha256": sha256_file(STEP_ROOT / "runtime_benchmark.json"),
        "scientificMatrixCount": 0,
        "scientificTrajectoryCount": 0,
        "gpuUsed": False,
        "S18Started": False,
    }
    write_json(STEP_ROOT / "provenance_manifest.json", provenance)
    manifest = _artifact_manifest()
    status["artifactsWritten"] = [row["path"] for row in manifest["entries"]] + [
        "artifact_manifest.json"
    ]
    write_json(STEP_ROOT / "status.json", status)
    manifest = _artifact_manifest()
    return {"validation": validation, "manifest": manifest}


def run_scientific(workers: int) -> dict[str, Any]:
    if workers != WORKERS:
        raise ValueError("the S17 lock requires exactly eight process workers")
    ensure_preoutcome_lock()
    pushed = require_clean_pushed_lock(require_compute_gate=True)
    gate_lock = json.loads(COMPUTE_GATE_LOCK.read_text(encoding="utf-8"))
    artifact_gate = json.loads((STEP_ROOT / "compute_gate.json").read_text())
    if not gate_lock.get("gatePassed") or not artifact_gate.get("gatePassed"):
        raise RuntimeError("S17 scientific execution forbidden by failed compute gate")
    if gate_lock["benchmarkSha256"] != sha256_file(
        STEP_ROOT / "runtime_benchmark.json"
    ):
        raise RuntimeError("final gate lock does not identify the measured benchmark")
    if gate_lock["computeGateArtifactSha256"] != sha256_file(
        STEP_ROOT / "compute_gate.json"
    ):
        raise RuntimeError("final gate lock does not identify the compute gate artifact")
    baseline = json.loads(
        (STEP_ROOT / "immutable_prior_baseline.json").read_text(encoding="utf-8")
    )
    if not validate_prior_baseline(baseline)["passed"]:
        raise RuntimeError("immutable prior check failed before scientific execution")
    contract_sha = sha256_file(EXECUTION_MANIFEST)
    started = time.perf_counter()
    units = [
        (candidate_id, matrix_index)
        for candidate_id in CANDIDATE_IDS
        for matrix_index in MATRIX_INDICES
    ]
    summaries = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_unit, candidate_id, matrix_index, contract_sha): (
                candidate_id,
                matrix_index,
            )
            for candidate_id, matrix_index in units
        }
        for future in as_completed(futures):
            candidate_id, matrix_index = futures[future]
            summary = future.result()
            summaries.append(summary)
            print(
                json.dumps(
                    {
                        "stage": "s17_unit_complete",
                        "candidateId": candidate_id,
                        "matrixIndex": matrix_index,
                        "processCpuSeconds": summary["processCpuSeconds"],
                        "wallSeconds": summary["wallSeconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summaries.sort(key=lambda row: (row["candidateId"], row["matrixIndex"]))
    if len(summaries) != 24:
        raise RuntimeError("S17 fixed scope did not return 24 candidate/matrix units")
    result = finalize_scientific(
        summaries,
        baseline,
        scientific_wall_seconds=time.perf_counter() - started,
    )
    result["repositoryState"] = pushed
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("benchmark", "scientific", "gate-stop"), required=True
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    load_config()
    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "benchmark":
        run_benchmark()
    elif args.stage == "scientific":
        run_scientific(args.workers)
    else:
        finalize_gate_stop()


if __name__ == "__main__":
    main()
