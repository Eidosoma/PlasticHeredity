#!/usr/bin/env python3
"""Run the frozen two-candidate E01 S13 held-out baseline scale-up."""

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
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy
import yaml
from sklearn.metrics import adjusted_rand_score

import e01_frozen_timebase_ensemble.core as s12g_core
from e01_confirmed_timebase_scaleup.core import (
    ANALYSIS_ROOT_SEED_HEX,
    CANDIDATE_IDS,
    EVIDENCE_CLASS,
    RAW_LJUNG_BOX_AT_LEAST,
    RESEARCH_STEP_ID,
    SPIKED_RUNS_AT_LEAST,
    VERSION,
    analysis_seed_material,
    association_gate,
    candidate_registry,
    combined_classification,
    derive_analysis_seed,
    drift_gate,
    outcome_class,
)
from e01_latent_timebase.core import (
    ExposureDefinition,
    SimulationDefinition,
    array_sha256,
    generate_beta,
    initialize_distinct_state,
    simulate_trajectory,
    trajectory_summary,
)
from e01_latent_timebase.core import (
    derive_seed as derive_simulation_seed,
)
from e01_pigozzi_source_audit.core import SourceImplementation
from e01_replay_repair.comparator import compare_seed_tuples, compare_trajectories
from e01_source_emergence_metric_identity.analysis import significant_opposite
from scripts.e01 import run_s12g_frozen_timebase_ensemble as backend

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S13"
CACHE_ROOT = Path("/cache/e01_s13")
RAW_ROOT = CACHE_ROOT / "raw_trajectories"
RESULT_CACHE = CACHE_ROOT / "source_results"
CONFIG_PATH = REPO / "configs/e01/s13_confirmed_timebase_baseline_scaleup_preregistration.yaml"
S12G_SCHEMA = REPO / "configs/e01/s12g_output_schemas.json"
SAFE_LATTICE = ARTIFACTS / "research_steps/S12B/safe_phi_lattice.json"
FIGURE_ROOT = STEP_ROOT / "figures"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def recursive_file_bytes(path: Path) -> int:
    """Return retained bytes for a file or directory without following links."""

    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def frame_hash(frame: pd.DataFrame) -> str:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, pa.Table.from_pandas(frame, preserve_index=False).schema) as writer:
        writer.write_table(pa.Table.from_pandas(frame, preserve_index=False))
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _candidate_definition(candidate_id: str) -> SimulationDefinition:
    row = {item["candidateId"]: item for item in candidate_registry()}[candidate_id]
    return SimulationDefinition(
        daughter_rule=row["daughterRule"],
        overshoot_rule=row["overshootRule"],
        exposure=ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=float(row["h"])),
    )


def _simulation_seed_row(seed: Any, candidate_id: str | None) -> dict[str, Any]:
    shared = seed.purpose in {"catalytic_matrix", "initial_state"}
    identity_candidate = "SHARED" if shared else str(candidate_id)
    stream_id = (
        f"S13::SIM::{seed.purpose}::M{int(seed.matrix_index):03d}::{identity_candidate}"
    )
    return {
        "researchStepId": RESEARCH_STEP_ID,
        "streamDomain": "simulation",
        "streamId": stream_id,
        "purpose": seed.purpose,
        "candidateId": None if shared else candidate_id,
        "matrixIndex": int(seed.matrix_index),
        "implementationId": None,
        "temporalModeId": None,
        "endpointGeneration": None,
        "derivedSeed": str(seed.derived_seed),
        "seedMaterialSha256": seed.seed_material_sha256,
        "rootHex": seed.root_sha256,
        "bitGenerator": "PCG64DXSM",
        "sharedAcrossCandidates": shared,
    }


def simulate_shared_matrix(matrix_index: int) -> dict[str, Any]:
    """Generate both candidate trajectories for one shared beta/initial unit."""

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    root = ANALYSIS_ROOT_SEED_HEX
    phase = "s13_heldout_scaleup"
    beta_seed = derive_simulation_seed(root, phase, "catalytic_matrix", matrix_index)
    init_seed = derive_simulation_seed(root, phase, "initial_state", matrix_index)
    beta = generate_beta(beta_seed)
    initial = initialize_distinct_state(init_seed)
    beta_hash = array_sha256(beta)
    initial_hash = array_sha256(initial)
    trajectories: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    seed_rows = [_simulation_seed_row(beta_seed, None), _simulation_seed_row(init_seed, None)]
    for candidate_id in CANDIDATE_IDS:
        definition = _candidate_definition(candidate_id)
        common = {
            "phase": phase,
            "root_hex": root,
            "matrix_index": matrix_index,
            "definition": definition,
            "stream_identity": candidate_id,
            "beta": beta,
            "initial_state": initial,
        }
        primary, primary_seeds = simulate_trajectory(**common)
        replay, replay_seeds = simulate_trajectory(**common)
        comparison = compare_trajectories(primary, replay)
        seeds_equal, seed_differences = compare_seed_tuples(primary_seeds, replay_seeds)
        if (
            not comparison.repaired_comparator_passed
            or comparison.discrete_divergence_count
            or comparison.finite_numeric_divergence_count
            or comparison.forbidden_nonfinite_difference_count
            or not seeds_equal
            or seed_differences
        ):
            raise RuntimeError(
                f"trajectory replay divergence {candidate_id}/M{matrix_index:03d}"
            )
        if primary.completed_fissions != 100 or primary.terminal_status != "requested_fissions_completed":
            raise RuntimeError(
                f"incomplete held-out trajectory {candidate_id}/M{matrix_index:03d}: "
                f"{primary.completed_fissions}/{primary.terminal_status}"
            )
        candidate_root = RAW_ROOT / candidate_id
        candidate_root.mkdir(parents=True, exist_ok=True)
        cache_path = candidate_root / f"M{matrix_index:03d}.pickle"
        with cache_path.open("wb") as handle:
            pickle.dump(primary, handle, protocol=5)
        if primary.beta_sha256 != beta_hash or primary.initial_state_sha256 != initial_hash:
            raise RuntimeError("shared beta or initial-state identity changed inside simulator")
        cache_hash = sha256_file(cache_path)
        trajectories.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "trajectoryId": primary.trajectory_id,
                "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
                "cachePath": str(cache_path),
                "cacheSha256": cache_hash,
                "betaSha256": beta_hash,
                "initialStateSha256": initial_hash,
                "trajectorySha256": primary.trajectory_sha256,
                "completedFissions": primary.completed_fissions,
                "terminalStatus": primary.terminal_status,
                "exactReplayPassed": True,
            }
        )
        summary = trajectory_summary(primary)
        summary.update(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": candidate_id,
                "tPhiLockedC1": int(primary.total_batch_updates + primary.completed_fissions),
            }
        )
        summaries.append(summary)
        replay_rows.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "trajectoryId": primary.trajectory_id,
                "oldComparatorPassed": comparison.old_comparator_passed,
                "repairedComparatorPassed": comparison.repaired_comparator_passed,
                "discreteDivergenceCount": comparison.discrete_divergence_count,
                "finiteNumericDivergenceCount": comparison.finite_numeric_divergence_count,
                "permittedPairedNanCount": comparison.permitted_paired_nan_count,
                "forbiddenNonfiniteDifferenceCount": comparison.forbidden_nonfinite_difference_count,
                "seedTupleExact": seeds_equal,
                "trajectorySha256Exact": primary.trajectory_sha256 == replay.trajectory_sha256,
                "passed": True,
            }
        )
        for seed in primary_seeds:
            if seed.purpose not in {"catalytic_matrix", "initial_state"}:
                seed_rows.append(_simulation_seed_row(seed, candidate_id))
    return {
        "matrixIndex": matrix_index,
        "trajectories": trajectories,
        "replay": replay_rows,
        "summaries": summaries,
        "seeds": seed_rows,
        "wallSeconds": time.perf_counter() - started_wall,
        "cpuSeconds": time.process_time() - started_cpu,
    }


def execute_simulation(indices: list[int], workers: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if workers == 1:
        return [simulate_shared_matrix(index) for index in indices]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(simulate_shared_matrix, index): index for index in indices}
        for future in as_completed(futures):
            index = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                raise RuntimeError(f"S13 simulation failed M{index:03d}: {type(exc).__name__}:{exc}") from exc
            rows.append(row)
            print(
                json.dumps(
                    {
                        "stage": "simulation_pair_complete",
                        "matrixIndex": index,
                        "wallSeconds": row["wallSeconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return sorted(rows, key=lambda item: item["matrixIndex"])


def configure_backend() -> None:
    backend.RESEARCH_STEP_ID = RESEARCH_STEP_ID
    backend.VERSION = VERSION
    backend.EVIDENCE_CLASS = EVIDENCE_CLASS
    backend.CANDIDATE_IDS = CANDIDATE_IDS
    backend.STEP_ROOT = STEP_ROOT
    backend.CACHE_ROOT = CACHE_ROOT
    backend.RESULT_CACHE = RESULT_CACHE
    backend.SAFE_LATTICE = SAFE_LATTICE
    backend.CONFIG_PATH = CONFIG_PATH
    backend.SCHEMA_PATH = S12G_SCHEMA
    backend.INPUT_MANIFEST = STEP_ROOT / "trajectory_manifest.parquet"
    backend.FIGURE_ROOT = FIGURE_ROOT
    backend.derive_seed = derive_analysis_seed
    backend._association_gate = association_gate
    backend._drift_gate = drift_gate
    s12g_core.RESEARCH_STEP_ID = RESEARCH_STEP_ID


def process_source_task(task: dict[str, Any]) -> dict[str, Any]:
    configure_backend()
    record = backend.process_trajectory(task)
    seed_path = Path(record["resultRoot"]) / "seeds.parquet"
    seeds = pd.read_parquet(seed_path)
    seeds["streamId"] = seeds["streamId"].astype(str).str.replace(
        r"^S12G::", "S13::SOURCE::", regex=True
    )
    seeds["researchStepId"] = RESEARCH_STEP_ID
    write_parquet(seed_path, seeds)
    return record


def execute_source(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_source_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"S13 source task failed {task['candidateId']}/M{int(task['matrixIndex']):03d}: "
                    f"{type(exc).__name__}:{exc}"
                ) from exc
            records.append(record)
            print(
                json.dumps(
                    {
                        "stage": "source_task_complete",
                        "candidateId": record["candidateId"],
                        "matrixIndex": record["matrixIndex"],
                        "wallSeconds": record["wallSeconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return sorted(records, key=lambda item: (item["candidateId"], item["matrixIndex"]))


def verify_method_lock() -> dict[str, Any]:
    lock = json.loads((STEP_ROOT / "method_lock.json").read_text(encoding="utf-8"))
    if not lock.get("passed"):
        raise RuntimeError("S13 method lock is not passing")
    head, remote = git("rev-parse", "HEAD"), git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote or git("status", "--short"):
        raise RuntimeError("S13 must execute at a clean pushed design commit")
    if head != lock["designCommit"]:
        raise RuntimeError("repository HEAD differs from the frozen S13 design commit")
    for item in lock["files"]:
        path = REPO / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"S13 method-lock file changed: {item['path']}")
    return lock


def validate_prior() -> dict[str, Any]:
    baseline = json.loads((STEP_ROOT / "immutable_prior_baseline.json").read_text())
    changed: list[dict[str, Any]] = []
    for item in baseline["files"]:
        path = Path(item["path"])
        actual = sha256_file(path) if path.is_file() else None
        if actual != item["sha256"]:
            changed.append(
                {
                    "path": str(path),
                    "expectedSha256": item["sha256"],
                    "actualSha256": actual,
                }
            )
    payload = {
        "schema": "eidosoma.e01.s13_immutable_prior_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "fileCount": len(baseline["files"]),
        "changedCount": len(changed),
        "changed": changed,
        "passed": not changed,
    }
    write_json(STEP_ROOT / "immutable_prior_validation.json", payload)
    return payload


def validate_sources() -> dict[str, Any]:
    payload = json.loads((STEP_ROOT / "source_snapshot_manifest.json").read_text())
    for row in payload["checks"]:
        path = Path(row["path"])
        if "expectedCommit" in row:
            actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
            dirty = subprocess.check_output(["git", "status", "--short"], cwd=path, text=True).strip()
            if actual != row["expectedCommit"] or dirty:
                raise RuntimeError(f"pinned source changed: {row['sourceId']}")
        elif sha256_file(path) != row["expectedSha256"]:
            raise RuntimeError(f"safe source artifact changed: {row['sourceId']}")
    if not payload.get("passed"):
        raise RuntimeError("frozen source-equivalence evidence is not passing")
    return payload


def adapter_view(prefix: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    original_columns = list(prefix.columns)
    original_hash = frame_hash(prefix)
    adapted = prefix.copy(deep=True)
    adapted["rawObservationIndex"] = adapted["endpointRawObservationIndex"]
    integer_identity = bool(
        adapted["rawObservationIndex"].notna().all()
        and np.array_equal(
            adapted["rawObservationIndex"].to_numpy(dtype=np.int64),
            adapted["endpointRawObservationIndex"].to_numpy(dtype=np.int64),
        )
    )
    monotone = 0
    groups = 0
    for _, group in adapted.groupby(["candidateId", "trajectoryId", "implementationId"], sort=True):
        groups += 1
        values = group.sort_values("generation")["rawObservationIndex"].to_numpy(dtype=np.int64)
        monotone += int(len(values) == len(np.unique(values)) and np.all(np.diff(values) > 0))
    original_after_hash = frame_hash(adapted[original_columns])
    view = adapted[
        [
            "candidateId",
            "trajectoryId",
            "matrixIndex",
            "implementationId",
            "generation",
            "endpointRawObservationIndex",
            "rawObservationIndex",
        ]
    ].copy()
    write_parquet(STEP_ROOT / "prefix_statistical_view_index.parquet", view)
    payload = {
        "schema": "eidosoma.e01.s13_s12j_adapter_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "formula": "rawObservationIndex := endpointRawObservationIndex",
        "rowCount": len(prefix),
        "rowCountUnchanged": len(prefix) == len(adapted),
        "integerIdentity": integer_identity,
        "groupCount": groups,
        "strictMonotoneGroupCount": monotone,
        "originalFieldHashBefore": original_hash,
        "originalFieldHashAfter": original_after_hash,
        "originalFieldsUnchanged": original_hash == original_after_hash,
        "sourceTableMutated": False,
        "passed": bool(integer_identity and groups == monotone and original_hash == original_after_hash),
    }
    write_json(STEP_ROOT / "adapter_validation.json", payload)
    return adapted, payload


def cross_candidate_results(
    labels: pd.DataFrame,
    association_details: pd.DataFrame,
    drift_details: pd.DataFrame,
    partitions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate_a, candidate_b = CANDIDATE_IDS
    for matrix_index in range(100):
        for label_id in (backend.HISTORICAL_LABEL_ID, backend.ONLINE_LABEL_ID):
            left = labels[(labels["candidateId"] == candidate_a) & (labels["matrixIndex"] == matrix_index) & (labels["labelId"] == label_id)].sort_values("generation")
            right = labels[(labels["candidateId"] == candidate_b) & (labels["matrixIndex"] == matrix_index) & (labels["labelId"] == label_id)].sort_values("generation")
            a, b = left["isReplicator"].astype(bool).to_numpy(), right["isReplicator"].astype(bool).to_numpy()
            for metric, value in (
                (f"{label_id}_binary_agreement", float(np.mean(a == b))),
                (f"{label_id}_adjusted_rand", float(adjusted_rand_score(a, b))),
            ):
                rows.append({"analysisType": "LABEL_COMPARISON", "matrixIndex": matrix_index, "candidateA": candidate_a, "candidateB": candidate_b, "pairingStatus": "PAIRED", "identityMatched": True, "metric": metric, "valueA": value, "valueB": None, "difference": None, "status": "PASS", "reason": None})
        for mode in ("FULL", "PREFIX"):
            detail = association_details[(association_details["matrixIndex"] == matrix_index) & (association_details["implementationId"] == SourceImplementation.IIGR.value) & (association_details["temporalMode"] == mode) & (association_details["estimand"] == "CURRENT_HISTORICAL")]
            values = {}
            for candidate in CANDIDATE_IDS:
                item = detail[detail["candidateId"] == candidate]["correlation"]
                values[candidate] = float(item.iloc[0]) if len(item) and pd.notna(item.iloc[0]) else None
            a, b = values[candidate_a], values[candidate_b]
            rows.append({"analysisType": "PRIMARY_ASSOCIATION", "matrixIndex": matrix_index, "candidateA": candidate_a, "candidateB": candidate_b, "pairingStatus": "PAIRED", "identityMatched": True, "metric": f"IIGR_{mode}_current_historical_spearman", "valueA": a, "valueB": b, "difference": a - b if a is not None and b is not None else None, "status": "PASS", "reason": None})
            detail = drift_details[(drift_details["matrixIndex"] == matrix_index) & (drift_details["implementationId"] == SourceImplementation.IIGR.value) & (drift_details["temporalMode"] == mode)]
            values = {}
            for candidate in CANDIDATE_IDS:
                item = detail[detail["candidateId"] == candidate]["meanDifference"]
                values[candidate] = float(item.iloc[0]) if len(item) and pd.notna(item.iloc[0]) else None
            a, b = values[candidate_a], values[candidate_b]
            rows.append({"analysisType": "REPLICATOR_DRIFT", "matrixIndex": matrix_index, "candidateA": candidate_a, "candidateB": candidate_b, "pairingStatus": "PAIRED", "identityMatched": True, "metric": f"IIGR_{mode}_mean_difference", "valueA": a, "valueB": b, "difference": a - b if a is not None and b is not None else None, "status": "PASS", "reason": None})
        part = partitions[(partitions["matrixIndex"] == matrix_index) & (partitions["implementationId"] == SourceImplementation.IIGR.value) & (partitions["fitKind"] == "completed_trajectory")]
        left, right = part[part["candidateId"] == candidate_a], part[part["candidateId"] == candidate_b]
        ari = backend.partition_ari(left.iloc[0], right.iloc[0]) if len(left) == len(right) == 1 else None
        rows.append({"analysisType": "FULL_PARTITION", "matrixIndex": matrix_index, "candidateA": candidate_a, "candidateB": candidate_b, "pairingStatus": "PAIRED", "identityMatched": True, "metric": "IIGR_full_partition_ARI", "valueA": ari, "valueB": None, "difference": None, "status": "PASS", "reason": None})
    return pd.DataFrame(rows)


def adjudicate(
    associations: pd.DataFrame,
    drift: pd.DataFrame,
    temporal: pd.DataFrame,
    spike: pd.DataFrame,
    summaries: dict[tuple[str, str, str, str], Any],
    full: pd.DataFrame,
    prefix: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        iigr, phirl = SourceImplementation.IIGR.value, SourceImplementation.PHIRL.value
        full_assoc = associations[(associations["candidateId"] == candidate_id) & (associations["implementationId"] == iigr) & (associations["estimand"] == "RETROSPECTIVE_CURRENT_GENERATION")].iloc[0]
        full_drift = drift[(drift["candidateId"] == candidate_id) & (drift["implementationId"] == iigr) & (drift["temporalModeId"].str.endswith("_FULL"))].iloc[0]
        prefix_assoc = associations[(associations["candidateId"] == candidate_id) & (associations["implementationId"] == iigr) & (associations["estimand"] == "CURRENT_HISTORICAL") & (associations["temporalModeId"].str.endswith("_PREFIX_ENDPOINT"))].iloc[0]
        phirl_full_opposite = significant_opposite(summaries[(candidate_id, phirl, "FULL", "CURRENT_HISTORICAL")])
        phirl_prefix_opposite = significant_opposite(summaries[(candidate_id, phirl, "PREFIX", "CURRENT_HISTORICAL")])
        full_coherent = bool(full_assoc["gatePassed"] and full_drift["gatePassed"])
        prefix_gate = bool(prefix_assoc["gatePassed"] and not phirl_prefix_opposite)
        aggregate = temporal[(temporal["candidateId"] == candidate_id) & (temporal["implementationId"] == iigr) & (temporal["rowType"] == "AGGREGATE")].iloc[0]
        candidate_spikes = spike[(spike["candidateId"] == candidate_id) & (spike["implementationId"] == iigr)]
        trajectory_temporal = temporal[(temporal["candidateId"] == candidate_id) & (temporal["implementationId"] == iigr) & (temporal["rowType"] == "TRAJECTORY")]
        runs_spiked = int((candidate_spikes["positive3SigmaCount"] > 0).sum())
        raw_significant = int((trajectory_temporal["ljungBoxPValue"] <= 0.05).sum())
        diff_significant = int((trajectory_temporal["differencedLjungBoxPValue"] <= 0.05).sum())
        punctuated = bool(runs_spiked >= SPIKED_RUNS_AT_LEAST and pd.notna(aggregate["aggregateTrendPValue"]) and float(aggregate["aggregateTrendPValue"]) > 0.05 and raw_significant >= RAW_LJUNG_BOX_AT_LEAST and diff_significant <= 0)
        full_coverage = float(full[full["candidateId"] == candidate_id].groupby("implementationId")["emergence"].apply(lambda values: np.isfinite(pd.to_numeric(values, errors="coerce")).mean()).min())
        eligible = prefix[(prefix["candidateId"] == candidate_id) & (prefix["priorLockedClockTransitions"] >= 256)]
        prefix_coverage = float(eligible.groupby("implementationId")["emergence"].apply(lambda values: np.isfinite(pd.to_numeric(values, errors="coerce")).mean()).min())
        operational = full_coverage >= 0.80 and prefix_coverage >= 0.80
        combined = full_coherent and prefix_gate
        if combined:
            candidate_class = "CANDIDATE_RETROSPECTIVE_AND_PROSPECTIVE_SUPPORT"
        elif full_coherent:
            candidate_class = "CANDIDATE_RETROSPECTIVE_ONLY"
        elif prefix_gate:
            candidate_class = "CANDIDATE_PROSPECTIVE_ONLY"
        else:
            candidate_class = "CANDIDATE_NOT_SUPPORTED"
        rows.append(
            {
                "candidateId": candidate_id,
                "candidateEvidenceStatus": "S12FR_UPSTREAM_CONFIRMED",
                "primaryFullAssociationGate": bool(full_assoc["gatePassed"]),
                "primaryFullDriftGate": bool(full_drift["gatePassed"]),
                "primaryFullCoherent": full_coherent,
                "primaryPrefixGate": prefix_gate,
                "combinedRetrospectiveAndProspectiveGate": combined,
                "punctuatedGate": punctuated,
                "phirlOppositeFull": phirl_full_opposite,
                "phirlOppositePrefix": phirl_prefix_opposite,
                "operationalCoverageGate": operational,
                "candidateClassification": candidate_class,
            }
        )
    frame = pd.DataFrame(rows)
    classification = combined_classification(frame.to_dict("records"))
    payload = {
        "schema": "eidosoma.e01.s13_classification.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "classification": classification,
        "candidateResults": frame.to_dict("records"),
        "positiveRequiresBothConfirmedCandidatesRetrospectiveAndProspective": True,
        "candidate1Excluded": True,
        "candidate1EvidenceStatusRetained": "HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED",
        "candidateWeightsUsed": False,
        "authorIdentityClaimPermitted": False,
        "s14ThroughS18Status": "BLOCKED_PENDING_S13_HUMAN_REVIEW",
    }
    return frame, payload


def compute_statistics(
    full: pd.DataFrame,
    adapted_prefix: pd.DataFrame,
    original_prefix: pd.DataFrame,
    labels: pd.DataFrame,
    partitions: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    configure_backend()
    associations, association_details, drift, drift_details, summaries, _ = backend.run_candidate_statistics(full, adapted_prefix)
    temporal, spike = backend.run_temporal_statistics(full)
    metric_identity = backend.run_metric_identity(full, original_prefix)
    future = backend.run_future_dependence(full, original_prefix, partitions)
    cross = cross_candidate_results(labels, association_details, drift_details, partitions)
    adjudication, classification = adjudicate(associations, drift, temporal, spike, summaries, full, original_prefix)
    return {
        "candidate_associations": associations,
        "candidate_association_details": association_details,
        "replicator_drift_results": drift,
        "replicator_drift_details": drift_details,
        "temporal_dependence_results": temporal,
        "spike_results": spike,
        "metric_identity_results": metric_identity,
        "future_dependence_results": future,
        "cross_candidate_results": cross,
        "ensemble_adjudication": adjudication,
    }, classification


RESULT_FILES = {
    "candidate_associations": "candidate_associations.csv",
    "candidate_association_details": "candidate_association_details.parquet",
    "replicator_drift_results": "replicator_drift_results.csv",
    "replicator_drift_details": "replicator_drift_details.parquet",
    "temporal_dependence_results": "temporal_dependence_results.csv",
    "spike_results": "spike_results.csv",
    "metric_identity_results": "metric_identity_results.csv",
    "future_dependence_results": "future_dependence_results.csv",
    "cross_candidate_results": "cross_candidate_results.csv",
    "ensemble_adjudication": "ensemble_adjudication.csv",
}


def write_results(results: dict[str, pd.DataFrame]) -> None:
    for key, filename in RESULT_FILES.items():
        path = STEP_ROOT / filename
        if path.suffix == ".parquet":
            write_parquet(path, results[key])
        else:
            write_csv(path, results[key])


def validate_statistics_replay(
    first: dict[str, pd.DataFrame],
    second: dict[str, pd.DataFrame],
    first_classification: dict[str, Any],
    second_classification: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key in RESULT_FILES:
        exact = True
        try:
            pd.testing.assert_frame_equal(first[key], second[key], check_exact=True, check_dtype=True)
        except AssertionError:
            exact = False
        rows.append({"resultId": key, "rowCount": len(first[key]), "firstSha256": frame_hash(first[key]), "secondSha256": frame_hash(second[key]), "exact": exact})
    classification_exact = json.dumps(jsonable(first_classification), sort_keys=True) == json.dumps(jsonable(second_classification), sort_keys=True)
    payload = {"schema": "eidosoma.e01.s13_statistics_replay_validation.v1", "researchStepId": RESEARCH_STEP_ID, "results": rows, "classificationExact": classification_exact, "passed": all(row["exact"] for row in rows) and classification_exact}
    write_json(STEP_ROOT / "statistics_replay_validation.json", payload)
    return payload


def _seed_material_row(
    *,
    domain: str,
    stream_id: str,
    purpose: str,
    identity: tuple[Any, ...],
    candidate_id: str | None,
    matrix_index: int | None,
    implementation_id: str | None = None,
    temporal_mode_id: str | None = None,
    endpoint_generation: int | None = None,
) -> dict[str, Any]:
    material = analysis_seed_material(*identity)
    return {
        "researchStepId": RESEARCH_STEP_ID,
        "streamDomain": domain,
        "streamId": stream_id,
        "purpose": purpose,
        "candidateId": candidate_id,
        "matrixIndex": matrix_index,
        "implementationId": implementation_id,
        "temporalModeId": temporal_mode_id,
        "endpointGeneration": endpoint_generation,
        "derivedSeed": str(derive_analysis_seed(*identity)),
        "seedMaterialSha256": hashlib.sha256(material).hexdigest(),
        "rootHex": ANALYSIS_ROOT_SEED_HEX,
        "bitGenerator": "MT19937_via_numpy_RandomState" if domain in {"statistics", "suffix"} else "source_wrapper_seed32",
        "sharedAcrossCandidates": False,
    }


def comprehensive_seed_manifest(
    simulation_rows: list[dict[str, Any]],
    source_seed_rows: pd.DataFrame,
    prefix: pd.DataFrame,
) -> pd.DataFrame:
    rows = list(simulation_rows)
    for item in source_seed_rows.to_dict("records"):
        endpoint = None if pd.isna(item["endpointGeneration"]) else int(item["endpointGeneration"])
        temporal = "FULL" if endpoint is None else "PREFIX_ENDPOINT"
        terminal = "NONE" if endpoint is None else endpoint
        identity = (str(item["purpose"]), str(item["candidateId"]), int(item["matrixIndex"]), str(item["implementationId"]), temporal, terminal)
        rows.append(_seed_material_row(domain="source", stream_id=str(item["streamId"]), purpose=str(item["purpose"]), identity=identity, candidate_id=str(item["candidateId"]), matrix_index=int(item["matrixIndex"]), implementation_id=str(item["implementationId"]), temporal_mode_id=str(item["temporalModeId"]), endpoint_generation=endpoint))
    eligible = prefix[prefix["priorLockedClockTransitions"] >= 256]
    for item in eligible.to_dict("records"):
        for purpose in ("suffix_deterministic_shuffle", "suffix_domain_separated_replacement"):
            identity = (purpose, item["candidateId"], int(item["matrixIndex"]), item["implementationId"], int(item["generation"]))
            stream_id = f"S13::SUFFIX::{purpose}::{item['candidateId']}::M{int(item['matrixIndex']):03d}::{item['implementationId']}::G{int(item['generation']):03d}"
            rows.append(_seed_material_row(domain="suffix", stream_id=stream_id, purpose=purpose, identity=identity, candidate_id=str(item["candidateId"]), matrix_index=int(item["matrixIndex"]), implementation_id=str(item["implementationId"]), temporal_mode_id="PREFIX_ENDPOINT", endpoint_generation=int(item["generation"])))
    for candidate_id in CANDIDATE_IDS:
        for implementation in SourceImplementation:
            implementation_id = implementation.value
            identities = [
                ("FULL_HIST_BOOTSTRAP", ("statistics", candidate_id, implementation_id, "FULL", "HIST", "bootstrap")),
                ("FULL_HIST_CIRCULAR", ("statistics", candidate_id, implementation_id, "FULL", "HIST", "circular")),
                ("FULL_DRIFT_BOOTSTRAP", ("statistics", candidate_id, implementation_id, "FULL", "drift", "bootstrap")),
                ("FULL_DRIFT_PERMUTATION", ("statistics", candidate_id, implementation_id, "FULL", "drift", "permutation")),
                ("PREFIX_DRIFT_BOOTSTRAP", ("statistics", candidate_id, implementation_id, "PREFIX", "drift", "bootstrap")),
                ("PREFIX_DRIFT_PERMUTATION", ("statistics", candidate_id, implementation_id, "PREFIX", "drift", "permutation")),
            ]
            for estimand in ("CURRENT_HISTORICAL", "NEXT_HISTORICAL", "CURRENT_PAST_ONLY_COSINE"):
                identities.extend([
                    (f"PREFIX_{estimand}_BOOTSTRAP", ("statistics", candidate_id, implementation_id, "PREFIX", estimand, "bootstrap")),
                    (f"PREFIX_{estimand}_CIRCULAR", ("statistics", candidate_id, implementation_id, "PREFIX", estimand, "circular")),
                ])
            for purpose, identity in identities:
                rows.append(_seed_material_row(domain="statistics", stream_id=f"S13::STAT::{candidate_id}::{implementation_id}::{purpose}", purpose=purpose, identity=identity, candidate_id=candidate_id, matrix_index=None, implementation_id=implementation_id))
    frame = pd.DataFrame(rows)
    if frame["streamId"].duplicated().any() or frame["seedMaterialSha256"].duplicated().any():
        duplicates = frame[frame["streamId"].duplicated(False) | frame["seedMaterialSha256"].duplicated(False)]
        raise RuntimeError(f"S13 seed identity/material duplicated: {len(duplicates)} rows")
    write_parquet(STEP_ROOT / "seed_manifest.parquet", frame)
    return frame


def prior_seed_sets() -> tuple[set[str], set[str]]:
    streams: set[str] = set()
    materials: set[str] = set()
    for path in sorted((ARTIFACTS / "research_steps").glob("S*/**/*seed*.parquet")):
        if "/S13/" in str(path):
            continue
        frame = pd.read_parquet(path)
        for column in frame.columns:
            name = column.lower()
            values = {str(value) for value in frame[column].dropna().tolist()}
            if name in {"streamid", "stream_id", "identity"}:
                streams.update(values)
            if name in {"seedmaterialsha256", "seed_material_sha256"}:
                materials.update(values)
    return streams, materials


def seed_firewall(frame: pd.DataFrame) -> dict[str, Any]:
    prior_streams, prior_materials = prior_seed_sets()
    stream_overlap = sorted(set(frame["streamId"]) & prior_streams)
    material_overlap = sorted(set(frame["seedMaterialSha256"]) & prior_materials)
    payload = {
        "schema": "eidosoma.e01.s13_seed_firewall.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "s13Rows": len(frame),
        "s13UniqueStreamIds": int(frame["streamId"].nunique()),
        "s13UniqueSeedMaterials": int(frame["seedMaterialSha256"].nunique()),
        "priorStreamIdentityCount": len(prior_streams),
        "priorSeedMaterialCount": len(prior_materials),
        "streamIdentityOverlapCount": len(stream_overlap),
        "seedMaterialOverlapCount": len(material_overlap),
        "streamIdentityOverlap": stream_overlap,
        "seedMaterialOverlap": material_overlap,
        "passed": not stream_overlap and not material_overlap,
    }
    write_json(STEP_ROOT / "seed_firewall.json", payload)
    return payload


def make_figures(
    associations: pd.DataFrame,
    association_details: pd.DataFrame,
    drift_details: pd.DataFrame,
    temporal: pd.DataFrame,
    spike: pd.DataFrame,
    future: pd.DataFrame,
    metric: pd.DataFrame,
    adjudication_frame: pd.DataFrame,
) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    colors = ["#4477AA", "#EE6677"]
    fig, ax = plt.subplots(figsize=(9, 5))
    data, labels = [], []
    for candidate in CANDIDATE_IDS:
        for mode in ("FULL", "PREFIX"):
            data.append(association_details[(association_details["candidateId"] == candidate) & (association_details["implementationId"] == SourceImplementation.IIGR.value) & (association_details["temporalMode"] == mode) & (association_details["estimand"] == "CURRENT_HISTORICAL")]["correlation"].dropna())
            labels.append(f"{candidate[-2:]}\n{mode}")
    ax.boxplot(data, labels=labels)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Within-trajectory Spearman rho")
    ax.set_title("Held-out IIGR association distributions")
    fig.tight_layout(); fig.savefig(FIGURE_ROOT / "association_distributions.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    data, labels = [], []
    for candidate in CANDIDATE_IDS:
        for mode in ("FULL", "PREFIX"):
            data.append(drift_details[(drift_details["candidateId"] == candidate) & (drift_details["implementationId"] == SourceImplementation.IIGR.value) & (drift_details["temporalMode"] == mode)]["meanDifference"].dropna())
            labels.append(f"{candidate[-2:]}\n{mode}")
    ax.boxplot(data, labels=labels); ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Replicator minus drift mean emergence"); ax.set_title("Held-out drift contrasts")
    fig.tight_layout(); fig.savefig(FIGURE_ROOT / "replicator_drift_distributions.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, candidate, color in zip(axes, CANDIDATE_IDS, colors, strict=True):
        subset = spike[(spike["candidateId"] == candidate) & (spike["implementationId"] == SourceImplementation.IIGR.value)]
        axis.hist(subset["positive3SigmaCount"], bins=20, color=color, alpha=0.8)
        aggregate = temporal[(temporal["candidateId"] == candidate) & (temporal["implementationId"] == SourceImplementation.IIGR.value) & (temporal["rowType"] == "AGGREGATE")]
        p = aggregate["aggregateTrendPValue"].iloc[0]
        axis.set_title(f"{candidate}\naggregate trend p={p:.3g}")
        axis.set_xlabel("Positive 3-sigma excursions")
    fig.tight_layout(); fig.savefig(FIGURE_ROOT / "temporal_spike_summary.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, candidate, color in zip(axes, CANDIDATE_IDS, colors, strict=True):
        subset = future[(future["candidateId"] == candidate) & (future["implementationId"] == SourceImplementation.IIGR.value)]
        axis.scatter(subset["fullReplicationAssociation"], subset["prefixReplicationAssociation"], s=10, alpha=0.55, color=color)
        axis.axhline(0, color="black", lw=0.5); axis.axvline(0, color="black", lw=0.5)
        axis.set_title(candidate); axis.set_xlabel("Full-fit association"); axis.set_ylabel("Prefix association")
    fig.tight_layout(); fig.savefig(FIGURE_ROOT / "full_prefix_comparison.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, candidate, color in zip(axes, CANDIDATE_IDS, colors, strict=True):
        subset = metric[(metric["candidateId"] == candidate) & (metric["implementationId"] == SourceImplementation.IIGR.value) & (metric["temporalModeId"] == "FULL")]
        axis.scatter(subset["replicationAssociationLocalPhiR"], subset["replicationAssociationEmergence"], s=10, alpha=0.55, color=color)
        axis.axhline(0, color="black", lw=0.5); axis.axvline(0, color="black", lw=0.5)
        axis.set_title(candidate); axis.set_xlabel("Corrected local Phi-r rho"); axis.set_ylabel("Emergence rho")
    fig.tight_layout(); fig.savefig(FIGURE_ROOT / "metric_identity_comparison.png", dpi=180); plt.close(fig)

    columns = ["primaryFullAssociationGate", "primaryFullDriftGate", "primaryFullCoherent", "primaryPrefixGate", "combinedRetrospectiveAndProspectiveGate", "punctuatedGate", "operationalCoverageGate"]
    matrix = adjudication_frame[columns].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 3.4)); image = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_yticks(range(2), ["Candidate 2", "Candidate 3"]); ax.set_xticks(range(len(columns)), ["Full assoc", "Full drift", "Full coherent", "Prefix", "Combined", "Punctuated", "Operational"], rotation=25, ha="right")
    for row in range(2):
        for column in range(len(columns)):
            ax.text(column, row, "PASS" if matrix[row, column] else "FAIL", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, ticks=[0, 1], shrink=0.7); ax.set_title("Held-out two-candidate decision matrix")
    fig.tight_layout(); fig.savefig(FIGURE_ROOT / "final_decision_matrix.png", dpi=180); plt.close(fig)


def artifact_manifest(config: dict[str, Any]) -> dict[str, Any]:
    required = [*config["artifacts"]["required"], *config["artifacts"]["figures"]]
    entries = []
    missing = []
    for relative in required:
        path = STEP_ROOT / relative
        if not path.is_file():
            missing.append(relative)
        else:
            entries.append({"relativePath": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    total = sum(item["bytes"] for item in entries)
    payload = {"schema": "eidosoma.e01.s13_artifact_manifest.v1", "researchStepId": RESEARCH_STEP_ID, "artifacts": entries, "artifactCountExcludingSelf": len(entries), "totalBytesExcludingSelf": total, "requiredMissing": missing, "under30GiB": total <= 30 * 1024**3, "passed": not missing and total <= 30 * 1024**3}
    write_json(STEP_ROOT / "artifact_manifest.json", payload)
    return payload


def _format(value: Any, digits: int = 5) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "PASS" if bool(value) else "FAIL"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def build_report(
    classification: dict[str, Any],
    results: dict[str, pd.DataFrame],
    runtime: dict[str, Any],
    execution: dict[str, Any],
    failures: pd.DataFrame,
    artifact_count: int,
) -> str:
    adjudication = results["ensemble_adjudication"]
    associations = results["candidate_associations"]
    drift = results["replicator_drift_results"]
    outcome = classification["classification"]
    lines = [
        "# S13 Full Results: Confirmed Time-base Baseline Held-out Scale-up",
        "",
        "## Top summary",
        "",
        f"- **Research step ID:** `{VERSION}` (S13).",
        "- **Completion status:** `COMPLETED_AT_MANDATORY_S13_HUMAN_REVIEW_BOUNDARY`; no later step began.",
        f"- **Artifacts written:** {artifact_count} status-bearing artifacts under `/artifacts/research_steps/S13/`, including 200 trajectory identities, replay/source/suffix/statistics validation, complete scientific tables, figures, manifests, and this report.",
        f"- **Validation result:** `{execution['validationResult']}`.",
        f"- **Outcome classification:** `{outcome}` ({outcome_class(outcome)}).",
        "- **Caveats or blockers:** This source-informed held-out test cannot identify the unavailable author implementation; full fits are retrospective/future-dependent; fixed-window and early-time claims remain unresolved; repeated overrides weaken procedural credibility; candidate 1 remains non-confirmed and excluded.",
        "- **Lay summary:** One hundred new catalytic matrices were simulated under both confirmed time-base candidates. The same frozen label and public-source emergence audit was then applied independently to all 200 trajectories. The result below reports whether both candidates agreed under the strict all-pass rule; it neither replaces prior negative evidence nor authorizes downstream work.",
        "- **Recommended next action:** Mandatory human review. Keep S14-S18, prediction, MLP work, interventions, estimator repair, E02, report-bundle progression, and any further scale-up blocked regardless of outcome.",
        "",
        "## Frozen question and scope",
        "",
        "This separately versioned step tested whether S12J's near-zero/non-support finding persists with greater power on genuinely new matrices. It used only S12FR-confirmed candidates 2 and 3. Candidate 1 was neither run nor reclassified. A positive held-out conclusion required both confirmed candidates to pass both the scaled retrospective-coherence gate and the unchanged prospective-direction/resampling gate.",
        "",
        "## Inputs and provenance",
        "",
        "- Original paper and its reported Figure 2/Table 1 fingerprints were interpretive targets only.",
        "- Candidate 2: `h=0.6031526490073492`, first daughter, trimmed new entrants, C1.",
        "- Candidate 3: `h=0.5613315384859516`, random nonempty daughter, trimmed new entrants, C1.",
        "- IIGR commit `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`; PhiRL commit `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`; safe-lattice SHA-256 `74ecca37f04201088d76a9e8ede7efe04bafebecff85a4882a44f03afbd23aa1`.",
        f"- Pushed design commit: `{runtime['designCommit']}`. New domain root ID: `E01-S13-HELDOUT-ROOT-v1.0.0`.",
        "",
        "## Methods",
        "",
        "Exactly 100 shared catalytic matrices and matched 40-distinct-singleton initial states were generated under a new 256-bit seed root. Each was run for 100 fissions under both candidates, yielding 200 retained primary trajectories plus 200 validation-only replay executions. The simulator, rates, Poisson vector update, new-entrant trimming, fission, daughter rules, and C1 clock were inherited unchanged from the S12FR-confirmed implementation.",
        "",
        "Labels were historical H>0.9 primary and past-only cosine secondary. Counts received additive-0.5 closure, full CLR, and removal of original component 100. IIGR source-defined synergy plus the two downward-causation atoms was primary; PhiRL was robustness; corrected local Phi-r was comparator-only. Full fits are `RETROSPECTIVE_FULL_TRAJECTORY_LOCAL`. Prefixes were independently refit only at post-fission endpoints with at least 256 prior C1 transitions. Every source fit replayed, every eligible prefix had structural suffix deletion/shuffle/replacement checks, and first/middle/last endpoints per trajectory/implementation were executed sentinels.",
        "",
        "The S12J 4,096-replicate trajectory bootstrap, circular-shift, and block-aware procedures were unchanged. Count gates were prospectively scaled to 100 matrices: at least 80 defined, 75 positive associations, 59 positive drift differences, 75 spiked runs, and 88 raw Ljung-Box-significant runs. All continuous, direction, p-value, coverage, replay, and suffix gates were unchanged.",
        "",
        "## Candidate-specific results",
        "",
        "| Candidate | Full median rho | Full association | Full median rep-drift | Drift gate | Full coherent | Prefix median rho | Prefix gate | Combined | Classification |",
        "| --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in adjudication.to_dict("records"):
        candidate = row["candidateId"]
        full_assoc = associations[(associations["candidateId"] == candidate) & (associations["implementationId"] == SourceImplementation.IIGR.value) & (associations["estimand"] == "RETROSPECTIVE_CURRENT_GENERATION")].iloc[0]
        prefix_assoc = associations[(associations["candidateId"] == candidate) & (associations["implementationId"] == SourceImplementation.IIGR.value) & (associations["estimand"] == "CURRENT_HISTORICAL") & (associations["temporalModeId"].str.endswith("_PREFIX_ENDPOINT"))].iloc[0]
        full_drift = drift[(drift["candidateId"] == candidate) & (drift["implementationId"] == SourceImplementation.IIGR.value) & (drift["temporalModeId"].str.endswith("_FULL"))].iloc[0]
        lines.append(f"| {candidate} | {_format(full_assoc['medianCorrelation'])} | {_format(row['primaryFullAssociationGate'])} | {_format(full_drift['medianMeanDifference'])} | {_format(row['primaryFullDriftGate'])} | {_format(row['primaryFullCoherent'])} | {_format(prefix_assoc['medianCorrelation'])} | {_format(row['primaryPrefixGate'])} | {_format(row['combinedRetrospectiveAndProspectiveGate'])} | {row['candidateClassification']} |")
    lines.extend([
        "",
        f"Final two-candidate classification: **`{outcome}`**.",
        "",
        "## Validation",
        "",
        f"- Retained primary trajectories: {execution['retainedTrajectoryCount']}/200; completed 100 fissions: {execution['completeTrajectoryCount']}/200.",
        f"- Exact simulator replay: {execution['trajectoryReplayPassCount']}/200; discrete, finite, and forbidden-nonfinite divergences were zero.",
        f"- Shared matrix/initial pairing: {execution['sharedIdentityCount']}/100.",
        f"- Source tasks: {execution['sourceTaskPassCount']}/200; minimum full coverage {_format(execution['minimumFullCoverage'])}; minimum eligible-prefix coverage {_format(execution['minimumPrefixCoverage'])}.",
        f"- Suffix checks: {execution['structuralSuffixPassCount']}/{execution['structuralSuffixCheckCount']} structural and {execution['executedSuffixPassCount']}/{execution['executedSuffixCount']} executed sentinels.",
        f"- Statistics replay, adapter, seed firewall, source identity, schemas, prior immutability, runtime, storage, and artifact validation: {execution['allValidationGatesPassed']}.",
        f"- Failure-ledger rows: {len(failures)}; expected pre-256 ineligibility is retained and is not a replay failure.",
        "",
        "## Runtime, storage, and compute ledger",
        "",
        f"The pre-simulation cumulative envelope was {runtime['priorCpuEnvelopeHours']:.3f} CPU-hours, with {runtime['preSimulationProjectedCumulativeCpuHours']:.3f}/250 projected after S13. The observed S13 worker CPU was {runtime['observedWorkerCpuHours']:.3f} hours and wall time was {runtime['wallHours']:.3f} hours. GPU use was zero. The first shared matrix under both candidates was the benchmark and remained part of the exactly 200 retained trajectories; no extra scientific trajectory was generated.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m pytest -q tests/e01/test_s13_confirmed_timebase_scaleup.py tests/e01/test_s12g_frozen_timebase_ensemble.py tests/e01/test_s12j_aggregation_interface_repair_confirmation.py",
        "python -m ruff check src/e01_confirmed_timebase_scaleup scripts/e01/freeze_s13_preregistration.py scripts/e01/run_s13_confirmed_timebase_scaleup.py tests/e01/test_s13_confirmed_timebase_scaleup.py",
        "ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s13_preregistration.py --record-commit",
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13_confirmed_timebase_scaleup.py --workers 6",
        "```",
        "",
        "## Caveats and limitations",
        "",
        "- These are public-source reconstructions, not the unpublished author implementation or an exact paper replication.",
        "- The 100 matrices are genuine downstream scientific holdouts, but the candidate family and method were selected through a long, repeatedly overridden E01 chain; that procedural history limits confirmatory credibility.",
        "- Completed-trajectory fits use future observations and support description only. Prefix estimates begin late and cannot recover fixed-window or early-time claims.",
        "- Candidate 1 remains a human-waived non-confirmed sensitivity case from S12J and is absent from this confirmatory gate.",
        "- A positive result would conflict with prior negative evidence and requires human adjudication; a null is scoped to the two confirmed simulator candidates, frozen labels, and source-defined metrics.",
        "",
        "## Provenance and artifact contract",
        "",
        "The preregistration, pushed method lock, cumulative compute ledger, source and safe-lattice identities, 100 shared matrix/initial hashes, 200 raw-cache/trajectory hashes, complete seed identities, exact replay rows, suffix checks, adapter audit, statistics replay, immutable-prior postcheck, runtime/storage records, schemas, scope ledger, and artifact manifest form the audit chain. Large raw trajectory and per-task source caches remain under `/cache/e01_s13/` and are represented by collectible hashes rather than copied into the artifact directory.",
        "",
        "## Recommended next action",
        "",
        "Return for mandatory human review. Do not begin S14-S18, prediction, MLP work, interventions, estimator repair, E02, report-bundle progression, or another scale-up automatically.",
        "",
    ])
    return "\n".join(lines)


def schema_validation(results: dict[str, pd.DataFrame]) -> dict[str, Any]:
    expected = {
        "trajectory_manifest.parquet": 200,
        "simulation_summary.parquet": 200,
        "trajectory_replay_validation.parquet": 200,
        "label_values.parquet": 40000,
        "prefix_endpoint_values.parquet": 40000,
        "candidate_associations.csv": 16,
        "replicator_drift_results.csv": 8,
        "temporal_dependence_results.csv": 404,
        "spike_results.csv": 400,
        "metric_identity_results.csv": 800,
        "future_dependence_results.csv": 400,
        "cross_candidate_results.csv": 900,
        "ensemble_adjudication.csv": 2,
    }
    rows = []
    for filename, count in expected.items():
        path = STEP_ROOT / filename
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        rows.append({"path": filename, "expectedRows": count, "actualRows": len(frame), "passed": len(frame) == count})
    payload = {"schema": "eidosoma.e01.s13_schema_validation.v1", "researchStepId": RESEARCH_STEP_ID, "tables": rows, "passed": all(row["passed"] for row in rows)}
    write_json(STEP_ROOT / "schema_validation.json", payload)
    return payload


def fail_closed(reason: str) -> None:
    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    failure = pd.DataFrame([{"failureId": "S13-TERMINAL", "stage": "execution", "candidateId": None, "trajectoryId": None, "implementationId": None, "temporalModeId": None, "endpointGeneration": None, "severity": "FATAL", "status": "S13_VALIDATION_FAILED_CLOSED", "reason": reason, "gateImpact": "FAIL_CLOSED_NO_REPAIR", "repairAttempted": False}])
    write_csv(STEP_ROOT / "failure_ledger.csv", failure)
    classification = {"schema": "eidosoma.e01.s13_classification.v1", "researchStepId": RESEARCH_STEP_ID, "versionedStepId": VERSION, "classification": "S13_VALIDATION_FAILED_CLOSED", "reason": reason, "candidate1Excluded": True, "s14ThroughS18Status": "BLOCKED_PENDING_S13_HUMAN_REVIEW"}
    write_json(STEP_ROOT / "classification.json", classification)
    status = {"researchStepId": VERSION, "stepNumber": "S13", "success": False, "status": "STOPPED_FAIL_CLOSED", "artifactsWritten": [path.name for path in STEP_ROOT.iterdir() if path.is_file()], "validationResult": f"FAIL_CLOSED:{reason}", "caveatsOrBlockers": [reason, "No repair or scope reduction is authorized.", "S14-S18 and every other downstream activity remain blocked."], "recommendedNextAction": "Mandatory human review; do not continue automatically.", "outcomeClassification": "S13_VALIDATION_FAILED_CLOSED", "s14ThroughS18Status": "BLOCKED_PENDING_S13_HUMAN_REVIEW"}
    write_json(STEP_ROOT / "status.json", status)
    report = "\n".join(["# S13 Full Results: Confirmed Time-base Baseline Held-out Scale-up", "", "## Top summary", "", f"- **Research step ID:** `{VERSION}`.", "- **Completion status:** `STOPPED_FAIL_CLOSED`.", "- **Artifacts written:** Status-bearing failure, classification, provenance available at the stop, and this canonical report.", f"- **Validation result:** `FAIL_CLOSED`: {reason}.", "- **Outcome classification:** `S13_VALIDATION_FAILED_CLOSED` (constraining/contradictory).", f"- **Caveats or blockers:** {reason}; no repair or scope reduction is authorized.", "- **Lay summary:** The held-out scale-up stopped at a frozen validation gate and produced no valid scientific adjudication.", "- **Recommended next action:** Mandatory human review; keep all downstream work blocked.", "", "## Methods, commands, inputs, results, validation, caveats, and provenance", "", "The complete preregistered design, inputs, commands, source identities, compute ledger, and prior-artifact hashes are retained in the S13 directory. The terminal reason above is preserved without a repair. Any partial cache remains unpromoted under `/cache/e01_s13/`. No scientific conclusion may be drawn from partial outputs.", ""])
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers != 6:
        raise RuntimeError("S13 freezes exactly six workers")
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    method_lock = verify_method_lock()
    prior = validate_prior()
    sources = validate_sources()
    compute_gate = json.loads((STEP_ROOT / "compute_gate.json").read_text())
    seed_root = json.loads((STEP_ROOT / "seed_root_lock.json").read_text())
    if not prior["passed"] or not sources["passed"] or not compute_gate["passed"] or not seed_root["passed"]:
        raise RuntimeError("pre-simulation immutability/source/compute/seed-root gate failed")
    if RAW_ROOT.exists() or RESULT_CACHE.exists():
        raise RuntimeError("fresh S13 cache roots already exist")
    RAW_ROOT.mkdir(parents=True)
    RESULT_CACHE.mkdir(parents=True)
    implementation_lock = {"schema": "eidosoma.e01.s13_implementation_lock.v1", "researchStepId": RESEARCH_STEP_ID, "versionedStepId": VERSION, "designCommit": method_lock["designCommit"], "implementationCommit": git("rev-parse", "HEAD"), "remoteCommit": git("rev-parse", "origin/eidosoma/groups/42"), "files": method_lock["files"], "simulationOutcomeOpenedBeforeLock": False, "labelOutcomeOpenedBeforeLock": False, "informationTheoryOutcomeOpenedBeforeLock": False, "passed": True}
    write_json(STEP_ROOT / "implementation_lock.json", implementation_lock)

    configure_backend()
    benchmark_sim = execute_simulation([0], workers=1)
    benchmark_manifest = pd.DataFrame(benchmark_sim[0]["trajectories"])
    benchmark_tasks = benchmark_manifest.to_dict("records")
    benchmark_source = execute_source(benchmark_tasks, workers=2)
    sim_cpu_per_matrix = float(benchmark_sim[0]["cpuSeconds"])
    source_cpu_per_task = float(np.mean([row["cpuSeconds"] for row in benchmark_source]))
    projected_raw_cpu = sim_cpu_per_matrix * 100 / 3600 + source_cpu_per_task * 200 / 3600 + 0.5
    projected_s13_cpu = projected_raw_cpu * 1.25
    prior_cpu = float(json.loads((STEP_ROOT / "compute_ledger.json").read_text())["priorCpuEnvelopeRoundedForDecision"])
    benchmark_retained_bytes = sum(
        recursive_file_bytes(Path(row["cachePath"])) for row in benchmark_tasks
    ) + sum(
        recursive_file_bytes(Path(row["resultRoot"])) for row in benchmark_source
    )
    benchmark = {"schema": "eidosoma.e01.s13_runtime_benchmark.v1", "researchStepId": RESEARCH_STEP_ID, "matrixIndex": 0, "retainedTrajectoryCount": 2, "includedInFinal200": True, "simulationCpuSeconds": sim_cpu_per_matrix, "sourceTaskCpuSeconds": [row["cpuSeconds"] for row in benchmark_source], "projectedS13CpuHoursIncluding25PercentReserve": projected_s13_cpu, "projectedCumulativeCpuHours": prior_cpu + projected_s13_cpu, "cpuCeilingHours": 250.0, "projectedWallHoursAtSixWorkers": projected_raw_cpu / 6 * 1.25, "wallCeilingHours": 72.0, "benchmarkRetainedBytes": benchmark_retained_bytes, "projectedRetainedGiB": benchmark_retained_bytes / 1024**3 * 100, "retainedArtifactGiBCeiling": 30.0}
    benchmark["passed"] = bool(benchmark["projectedCumulativeCpuHours"] <= 250 and benchmark["projectedWallHoursAtSixWorkers"] <= 72 and benchmark["projectedRetainedGiB"] <= 30)
    write_json(STEP_ROOT / "runtime_benchmark.json", benchmark)
    if not benchmark["passed"]:
        raise RuntimeError("S13 benchmark projects beyond a hard ceiling")

    simulation = benchmark_sim + execute_simulation(list(range(1, 100)), workers=args.workers)
    trajectory_rows = [row for unit in simulation for row in unit["trajectories"]]
    replay_rows = [row for unit in simulation for row in unit["replay"]]
    summary_rows = [row for unit in simulation for row in unit["summaries"]]
    simulation_seed_rows = [row for unit in simulation for row in unit["seeds"]]
    trajectory_manifest = pd.DataFrame(trajectory_rows).sort_values(["candidateId", "matrixIndex"]).reset_index(drop=True)
    replay_frame = pd.DataFrame(replay_rows).sort_values(["candidateId", "matrixIndex"]).reset_index(drop=True)
    simulation_summary = pd.DataFrame(summary_rows).sort_values(["candidateId", "matrixIndex"]).reset_index(drop=True)
    if len(trajectory_manifest) != 200 or len(replay_frame) != 200 or len(simulation_summary) != 200:
        raise RuntimeError("S13 did not produce exactly 200 primary trajectory records")
    if not replay_frame["passed"].astype(bool).all():
        raise RuntimeError("S13 trajectory replay gate failed")
    pairing_rows = []
    for matrix_index, group in trajectory_manifest.groupby("matrixIndex", sort=True):
        shared = len(group) == 2 and group["betaSha256"].nunique() == 1 and group["initialStateSha256"].nunique() == 1
        pairing_rows.append({"matrixIndex": int(matrix_index), "candidateCount": len(group), "betaSha256": group["betaSha256"].iloc[0], "initialStateSha256": group["initialStateSha256"].iloc[0], "sharedIdentity": shared})
    pairing = {"schema": "eidosoma.e01.s13_pairing_audit.v1", "researchStepId": RESEARCH_STEP_ID, "rows": pairing_rows, "sharedIdentityCount": sum(row["sharedIdentity"] for row in pairing_rows), "passed": len(pairing_rows) == 100 and all(row["sharedIdentity"] for row in pairing_rows)}
    if not pairing["passed"]:
        raise RuntimeError("S13 shared matrix/initial pairing failed")
    write_parquet(STEP_ROOT / "trajectory_manifest.parquet", trajectory_manifest)
    write_parquet(STEP_ROOT / "trajectory_replay_validation.parquet", replay_frame)
    write_parquet(STEP_ROOT / "simulation_summary.parquet", simulation_summary)
    write_json(STEP_ROOT / "pairing_audit.json", pairing)

    remaining_tasks = trajectory_manifest[trajectory_manifest["matrixIndex"] != 0].to_dict("records")
    remaining_source = execute_source(remaining_tasks, workers=args.workers)
    source_records = sorted(benchmark_source + remaining_source, key=lambda row: (row["candidateId"], row["matrixIndex"]))
    if len(source_records) != 200:
        raise RuntimeError("S13 source execution did not return 200 task records")
    configure_backend()
    frames = backend.collate(source_records)
    labels = frames["label_values.parquet"]
    full = frames["full_source_values.parquet"]
    prefix = frames["prefix_endpoint_values.parquet"]
    partitions = frames["partition_history.parquet"]
    diagnostics = frames["source_diagnostic_outputs.parquet"]
    suffix = frames["replay_suffix_validation.parquet"]
    source_seed_rows = frames["seed_manifest.parquet"]
    worker_failures = frames["worker_failures"]
    adapted_prefix, adapter = adapter_view(prefix)

    eligible_prefix = prefix[prefix["priorLockedClockTransitions"] >= 256]
    full_coverage = full.assign(numeric=np.isfinite(pd.to_numeric(full["emergence"], errors="coerce"))).groupby(["candidateId", "implementationId"])["numeric"].mean()
    prefix_coverage = eligible_prefix.assign(numeric=np.isfinite(pd.to_numeric(eligible_prefix["emergence"], errors="coerce"))).groupby(["candidateId", "implementationId"])["numeric"].mean()
    executed_suffix = suffix[suffix["sentinel"] != "non_sentinel"]
    source_gate = bool(len(worker_failures) == 0 and full["exactReplayPassed"].astype(bool).all() and eligible_prefix["exactReplayPassed"].astype(bool).all() and suffix["structuralExact"].astype(bool).all() and executed_suffix["resultExact"].fillna(False).astype(bool).all() and len(executed_suffix) == 200 * 2 * 3 * 3 and float(full_coverage.min()) >= 0.80 and float(prefix_coverage.min()) >= 0.80 and diagnostics["componentIdentityMaxAbsError"].fillna(0).max() <= 1e-12)
    if not source_gate or not adapter["passed"]:
        raise RuntimeError("S13 source/replay/coverage/suffix/adapter gate failed")

    first_results, first_classification = compute_statistics(full, adapted_prefix, prefix, labels, partitions)
    second_results, second_classification = compute_statistics(full, adapted_prefix, prefix, labels, partitions)
    statistics_replay = validate_statistics_replay(first_results, second_results, first_classification, second_classification)
    if not statistics_replay["passed"]:
        raise RuntimeError("S13 deterministic statistics replay failed")
    write_results(first_results)
    write_json(STEP_ROOT / "classification.json", first_classification)
    failure_rows = backend.failure_rows_from_statuses(full, prefix, worker_failures)
    for row in failure_rows:
        row["failureId"] = str(row["failureId"]).replace("S12G-", "S13-")
    failures = pd.DataFrame(failure_rows, columns=json.loads(S12G_SCHEMA.read_text())["tables"]["failure_ledger.csv"])
    write_csv(STEP_ROOT / "failure_ledger.csv", failures)

    seeds = comprehensive_seed_manifest(simulation_seed_rows, source_seed_rows, prefix)
    firewall = seed_firewall(seeds)
    if not firewall["passed"]:
        raise RuntimeError("S13 seed identity/material firewall failed")
    make_figures(first_results["candidate_associations"], first_results["candidate_association_details"], first_results["replicator_drift_details"], first_results["temporal_dependence_results"], first_results["spike_results"], first_results["future_dependence_results"], first_results["metric_identity_results"], first_results["ensemble_adjudication"])

    prior = validate_prior()
    schemas = schema_validation(first_results)
    if not prior["passed"] or not schemas["passed"]:
        raise RuntimeError("S13 prior immutability or schema validation failed")
    total_cache_bytes = sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
    total_artifact_bytes = sum(path.stat().st_size for path in STEP_ROOT.rglob("*") if path.is_file())
    storage = {"schema": "eidosoma.e01.s13_storage_validation.v1", "researchStepId": RESEARCH_STEP_ID, "cacheBytes": total_cache_bytes, "artifactBytesBeforeFinalReportAndManifest": total_artifact_bytes, "artifactGiBCeiling": 30.0, "passed": total_artifact_bytes <= 30 * 1024**3}
    write_json(STEP_ROOT / "storage_validation.json", storage)
    if not storage["passed"]:
        raise RuntimeError("S13 storage ceiling exceeded")

    simulation_cpu = sum(float(row["cpuSeconds"]) for row in simulation)
    source_cpu = sum(float(row["cpuSeconds"]) for row in source_records)
    runtime = {"schema": "eidosoma.e01.s13_runtime_manifest.v1", "researchStepId": RESEARCH_STEP_ID, "versionedStepId": VERSION, "designCommit": method_lock["designCommit"], "startedAtUtc": datetime.now(timezone.utc).isoformat(), "wallHours": (time.perf_counter() - started_wall) / 3600, "orchestrationProcessCpuHours": (time.process_time() - started_cpu) / 3600, "simulationWorkerCpuHours": simulation_cpu / 3600, "sourceWorkerCpuHours": source_cpu / 3600, "observedWorkerCpuHours": (simulation_cpu + source_cpu) / 3600, "priorCpuEnvelopeHours": prior_cpu, "preSimulationProjectedCumulativeCpuHours": float(compute_gate["projectedCumulativeCpuHours"]), "observedCumulativeCpuEnvelopeHours": prior_cpu + (simulation_cpu + source_cpu) / 3600 + (time.process_time() - started_cpu) / 3600, "cpuCeilingHours": 250.0, "gpuHours": 0.0, "gpuCeilingHours": 80.0, "workers": args.workers, "threadEnvironment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}, "cpuPrecision": "float64_authoritative", "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(), "retainedPrimaryTrajectoryCount": 200, "validationReplayExecutionCount": 200, "sourceTaskCount": 200, "passed": True}
    write_json(STEP_ROOT / "runtime_manifest.json", runtime)
    provenance = {"schema": "eidosoma.e01.s13_provenance_manifest.v1", "researchStepId": RESEARCH_STEP_ID, "designCommit": method_lock["designCommit"], "branch": git("branch", "--show-current"), "sourceCommits": {"IIGR": "7c1c22fe39f539d4a453135476f1f0dd5a6b45f7", "PhiRL": "a6d1d0d18c7551302724b7158c6ccdc4d3a33373", "historicalGARD": "86dff6320d5ae91b4e831471079ff46749b14df9"}, "safeLatticeSha256": sha256_file(SAFE_LATTICE), "candidateIds": list(CANDIDATE_IDS), "candidate1Executed": False, "matrixCount": 100, "trajectoryCount": 200, "rawCacheRoot": str(RAW_ROOT), "sourceCacheRoot": str(RESULT_CACHE), "passed": True}
    write_json(STEP_ROOT / "provenance_manifest.json", provenance)
    access = json.loads((STEP_ROOT / "scope_access_ledger.json").read_text())
    access["events"].append({"stage": "COMPLETE_S13_EXECUTION", "simulationOutcomeOpened": True, "labelOutcomeOpened": True, "informationTheoryOutcomeOpened": True, "candidate1Executed": False, "predictionOrInterventionAccess": False, "s14ThroughS18Access": False, "status": "PASS"})
    access["success"] = True
    write_json(STEP_ROOT / "scope_access_ledger.json", access)
    execution = {"schema": "eidosoma.e01.s13_execution_validation.v1", "researchStepId": RESEARCH_STEP_ID, "retainedTrajectoryCount": len(trajectory_manifest), "completeTrajectoryCount": int((trajectory_manifest["completedFissions"] == 100).sum()), "trajectoryReplayPassCount": int(replay_frame["passed"].sum()), "sharedIdentityCount": pairing["sharedIdentityCount"], "sourceTaskPassCount": sum(bool(row["fullReplayAllPassed"] and row["prefixReplayAllPassed"] and row["futureSuffixAllPassed"] and row["failureRows"] == 0) for row in source_records), "minimumFullCoverage": float(full_coverage.min()), "minimumPrefixCoverage": float(prefix_coverage.min()), "structuralSuffixCheckCount": len(suffix), "structuralSuffixPassCount": int(suffix["structuralExact"].astype(bool).sum()), "executedSuffixCount": len(executed_suffix), "executedSuffixPassCount": int(executed_suffix["resultExact"].fillna(False).astype(bool).sum()), "adapterPassed": adapter["passed"], "statisticsReplayPassed": statistics_replay["passed"], "seedFirewallPassed": firewall["passed"], "priorImmutabilityPassed": prior["passed"], "schemaPassed": schemas["passed"], "runtimePassed": runtime["passed"], "storagePassed": storage["passed"], "allValidationGatesPassed": True, "validationResult": "PASS_ALL_FROZEN_SIMULATION_SOURCE_REPLAY_SUFFIX_STATISTICS_PROVENANCE_AND_ARTIFACT_GATES"}
    write_json(STEP_ROOT / "execution_validation.json", execution)
    report = build_report(first_classification, first_results, runtime, execution, failures, 0)
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {"researchStepId": VERSION, "stepNumber": "S13", "success": True, "status": "COMPLETED_AT_MANDATORY_S13_HUMAN_REVIEW_BOUNDARY", "artifactsWritten": [*config["artifacts"]["required"], *config["artifacts"]["figures"], "artifact_manifest.json"], "validationResult": execution["validationResult"], "caveatsOrBlockers": ["Public-source reconstruction is not author- or paper-primary.", "Full fits are retrospective and future-dependent; fixed-window and early-time claims remain unresolved.", "Repeated E01 overrides weaken procedural credibility.", "Candidate 1 remains HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED and was not run.", "S14-S18 and every other downstream activity remain blocked."], "recommendedNextAction": "Mandatory human review; do not continue automatically.", "outcomeClassification": first_classification["classification"], "outcomeClass": outcome_class(first_classification["classification"]), "s14ThroughS18Status": "BLOCKED_PENDING_S13_HUMAN_REVIEW"}
    write_json(STEP_ROOT / "status.json", status)
    manifest = artifact_manifest(config)
    if not manifest["passed"]:
        raise RuntimeError(f"S13 artifact completeness failed: {manifest['requiredMissing']}")
    report = build_report(first_classification, first_results, runtime, execution, failures, manifest["artifactCountExcludingSelf"] + 1)
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    manifest = artifact_manifest(config)
    if not manifest["passed"]:
        raise RuntimeError("S13 final artifact manifest failed")
    print(json.dumps({"stage": "S13_complete", "classification": first_classification["classification"], "validation": execution["validationResult"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:
        reason = f"{type(error).__name__}:{error}"
        fail_closed(reason)
        print(json.dumps({"stage": "S13_failed_closed", "error": reason}, sort_keys=True), file=sys.stderr, flush=True)
        raise SystemExit(1) from error
