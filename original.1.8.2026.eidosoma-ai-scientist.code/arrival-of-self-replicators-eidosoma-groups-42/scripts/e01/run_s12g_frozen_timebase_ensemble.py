#!/usr/bin/env python3
"""Execute the frozen E01 S12G three-candidate ensemble audit."""

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
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from e01_frozen_timebase_ensemble.core import (
    CANDIDATE_IDS,
    ELIGIBLE_SOURCE_STATUSES,
    EVIDENCE_CLASS,
    HISTORICAL_LABEL_ID,
    ONLINE_LABEL_ID,
    RESEARCH_STEP_ID,
    VERSION,
    derive_seed,
    frozen_clr,
    frozen_generation_labels,
    post_fission_endpoint_records,
    selected_clock_observations,
    sha256_array,
    states_from_observations,
)
from e01_pigozzi_source_audit.core import SourceImplementation
from e01_source_emergence_metric_identity.analysis import (
    excursion_thresholds,
    finite_pearson,
    rank_agreement,
    replicator_drift_summary,
    significant_opposite,
    temporal_structure_rows,
    trajectory_association_summary,
)
from e01_source_emergence_metric_identity.core import (
    result_replay_equal,
    run_emergence_pipeline,
)

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S12G"
CACHE_ROOT = Path("/cache/e01_s12g")
RESULT_CACHE = CACHE_ROOT / "source_results"
SAFE_LATTICE = ARTIFACTS / "research_steps/S12B/safe_phi_lattice.json"
CONFIG_PATH = REPO / "configs/e01/s12g_frozen_timebase_ensemble_preregistration.yaml"
SCHEMA_PATH = REPO / "configs/e01/s12g_output_schemas.json"
INPUT_MANIFEST = STEP_ROOT / "trajectory_input_manifest.parquet"
FIGURE_ROOT = STEP_ROOT / "figures"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], columns: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows), columns=columns).to_csv(
        path, index=False, lineterminator="\n"
    )


def write_parquet(
    path: Path, frame_or_rows: pd.DataFrame | Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = (
        frame_or_rows
        if isinstance(frame_or_rows, pd.DataFrame)
        else pd.DataFrame(list(frame_or_rows))
    )
    frame.to_parquet(path, index=False, compression="zstd")


def concat_parquets(paths: list[Path], output: Path) -> None:
    writer: pq.ParquetWriter | None = None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for path in paths:
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def partition_json(values: tuple[int, ...]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def point_values(result: Any, local_index: int) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for output, attribute in (
        ("synergy", "synergy"),
        ("downwardCausation", "downward_causation"),
        ("emergence", "emergence"),
        ("localPhiR", "local_phi_r"),
    ):
        array = getattr(result, attribute)
        if array is None or local_index < 0 or local_index >= len(array):
            values[output] = None
        else:
            value = float(array[local_index])
            values[output] = value if np.isfinite(value) else None
    return values


def point_status(
    result: Any, replay_passed: bool, values: dict[str, float | None]
) -> tuple[str, str | None]:
    if not replay_passed:
        return "INELIGIBLE_EXACT_REPLAY_FAILED", "source_pipeline_replay_failed"
    if result.status not in ELIGIBLE_SOURCE_STATUSES:
        return str(result.status), result.reason
    if values["emergence"] is None:
        return "INELIGIBLE_NONFINITE_EMERGENCE", "emergence_nonfinite_or_absent"
    return "ELIGIBLE", None


def partition_row(
    result: Any,
    *,
    candidate_id: str,
    trajectory_id: str,
    matrix_index: int,
    clock_id: str,
    implementation_id: str,
    temporal_mode_id: str,
    fit_kind: str,
    endpoint_generation: int | None,
    endpoint_sequence_index: int,
    replay_passed: bool,
) -> dict[str, Any]:
    return {
        "researchStepId": RESEARCH_STEP_ID,
        "candidateId": candidate_id,
        "trajectoryId": trajectory_id,
        "matrixIndex": matrix_index,
        "clockId": clock_id,
        "implementationId": implementation_id,
        "temporalModeId": temporal_mode_id,
        "fitKind": fit_kind,
        "endpointGeneration": endpoint_generation,
        "endpointSelectedSequenceIndex": endpoint_sequence_index,
        "status": result.status,
        "reason": result.reason,
        "retainedVariablesJson": partition_json(result.retained_variables),
        "partition1Json": partition_json(result.partition_1),
        "partition2Json": partition_json(result.partition_2),
        "partitionSize1": len(result.partition_1),
        "partitionSize2": len(result.partition_2),
        "exactReplayPassed": replay_passed,
    }


def diagnostic_row(
    result: Any,
    *,
    candidate_id: str,
    trajectory_id: str,
    matrix_index: int,
    implementation_id: str,
    temporal_mode_id: str,
    fit_kind: str,
    endpoint_generation: int | None,
) -> dict[str, Any]:
    return {
        "researchStepId": RESEARCH_STEP_ID,
        "candidateId": candidate_id,
        "trajectoryId": trajectory_id,
        "matrixIndex": matrix_index,
        "implementationId": implementation_id,
        "temporalModeId": temporal_mode_id,
        "fitKind": fit_kind,
        "endpointGeneration": endpoint_generation,
        "status": result.status,
        "reason": result.reason,
        "componentIdentityMaxAbsError": result.component_identity_max_abs_error,
        "retainedVariableCount": len(result.retained_variables),
        "miFinite": bool(
            result.mi_matrix is not None and np.all(np.isfinite(result.mi_matrix))
        ),
        "partitionAverageFinite": bool(
            result.partition_average is not None
            and np.all(np.isfinite(result.partition_average))
        ),
        "emergenceFiniteCount": int(
            np.sum(np.isfinite(result.emergence))
            if result.emergence is not None
            else 0
        ),
        "localPhiRFiniteCount": int(
            np.sum(np.isfinite(result.local_phi_r))
            if result.local_phi_r is not None
            else 0
        ),
    }


def source_seeds(
    candidate_id: str,
    matrix_index: int,
    implementation_id: str,
    temporal_mode: str,
    endpoint_generation: int | None,
) -> tuple[int, int]:
    identity = (
        candidate_id,
        matrix_index,
        implementation_id,
        temporal_mode,
        "NONE" if endpoint_generation is None else endpoint_generation,
    )
    return (
        derive_seed("source_preprocessing", *identity),
        derive_seed("source_partition", *identity),
    )


def _task_root(candidate_id: str, matrix_index: int) -> Path:
    return RESULT_CACHE / candidate_id / f"M{matrix_index:02d}"


def process_trajectory(task: dict[str, Any]) -> dict[str, Any]:
    """Label, preprocess, and run both source branches for one locked input."""

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    candidate_id = str(task["candidateId"])
    matrix_index = int(task["matrixIndex"])
    trajectory_id = str(task["trajectoryId"])
    clock_id = str(task["clockId"])
    root = _task_root(candidate_id, matrix_index)
    completion = root / "completion.json"
    if completion.is_file():
        cached = json.loads(completion.read_text(encoding="utf-8"))
        if cached.get("inputCacheSha256") == task["cacheSha256"]:
            cached["resumed"] = True
            return cached
        raise RuntimeError(f"stale S12G task cache: {candidate_id}/M{matrix_index:02d}")
    if root.exists():
        raise RuntimeError(f"incomplete S12G task cache exists: {root}")
    root.mkdir(parents=True)
    cache_path = Path(str(task["cachePath"]))
    if sha256_file(cache_path) != task["cacheSha256"]:
        raise RuntimeError(f"input cache hash changed: {cache_path}")
    with cache_path.open("rb") as handle:
        trajectory = pickle.load(handle)
    if (
        str(trajectory.configuration_id) != candidate_id
        or int(trajectory.matrix_index) != matrix_index
        or str(trajectory.trajectory_id) != trajectory_id
        or str(trajectory.trajectory_sha256) != task["trajectorySha256"]
        or str(trajectory.beta_sha256) != task["betaSha256"]
        or str(trajectory.initial_state_sha256) != task["initialStateSha256"]
        or int(trajectory.completed_fissions) != 100
    ):
        raise RuntimeError(f"locked trajectory identity mismatch: {trajectory_id}")

    selected = selected_clock_observations(trajectory, clock_id)
    states = states_from_observations(selected)
    clr, masses, closure_errors = frozen_clr(states)
    labels, historical_map, online_map = frozen_generation_labels(trajectory)
    all_endpoints = post_fission_endpoint_records(
        trajectory, clock_id, minimum_prior_transitions=0
    )
    eligible_endpoints = [
        endpoint
        for endpoint in all_endpoints
        if endpoint.prior_locked_clock_transitions >= 256
    ]
    sentinel_generations = (
        {
            eligible_endpoints[0].generation,
            eligible_endpoints[len(eligible_endpoints) // 2].generation,
            eligible_endpoints[-1].generation,
        }
        if eligible_endpoints
        else set()
    )

    preprocessing = [
        {
            "researchStepId": RESEARCH_STEP_ID,
            "candidateId": candidate_id,
            "trajectoryId": trajectory_id,
            "matrixIndex": matrix_index,
            "clockId": clock_id,
            "observationCount": len(selected),
            "inputDimension": 100,
            "outputDimension": 99,
            "minimumMass": float(np.min(masses)),
            "maximumMass": float(np.max(masses)),
            "maximumClosureError": float(np.max(closure_errors)),
            "finite": bool(np.all(np.isfinite(clr))),
            "clrSha256": sha256_array(clr),
            "status": "ELIGIBLE",
            "reason": None,
        }
    ]
    full_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    suffix_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    evaluations = 0
    full_replay_all = True
    prefix_replay_all = True
    suffix_all = True

    for implementation in SourceImplementation:
        implementation_id = implementation.value
        full_mode = f"{implementation_id}_EMERGENCE_FULL"
        pre_seed, part_seed = source_seeds(
            candidate_id, matrix_index, implementation_id, "FULL", None
        )
        for purpose, seed in (
            ("source_preprocessing", pre_seed),
            ("source_partition", part_seed),
        ):
            seed_rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "implementationId": implementation_id,
                    "temporalModeId": full_mode,
                    "endpointGeneration": None,
                    "purpose": purpose,
                    "streamId": f"S12G::{candidate_id}::M{matrix_index:02d}::{implementation_id}::FULL::{purpose}",
                    "seed": seed,
                }
            )
        result = run_emergence_pipeline(
            clr,
            implementation,
            SAFE_LATTICE,
            preprocessing_seed=pre_seed,
            partition_seed=part_seed,
        )
        replay = run_emergence_pipeline(
            clr,
            implementation,
            SAFE_LATTICE,
            preprocessing_seed=pre_seed,
            partition_seed=part_seed,
        )
        evaluations += 2
        replay_ok = result_replay_equal(result, replay)
        full_replay_all &= replay_ok
        partition_rows.append(
            partition_row(
                result,
                candidate_id=candidate_id,
                trajectory_id=trajectory_id,
                matrix_index=matrix_index,
                clock_id=clock_id,
                implementation_id=implementation_id,
                temporal_mode_id=full_mode,
                fit_kind="completed_trajectory",
                endpoint_generation=100,
                endpoint_sequence_index=len(selected) - 1,
                replay_passed=replay_ok,
            )
        )
        diagnostic_rows.append(
            diagnostic_row(
                result,
                candidate_id=candidate_id,
                trajectory_id=trajectory_id,
                matrix_index=matrix_index,
                implementation_id=implementation_id,
                temporal_mode_id=full_mode,
                fit_kind="completed_trajectory",
                endpoint_generation=100,
            )
        )
        for local_index in range(max(0, len(selected) - result.local_offset)):
            selected_index = local_index + result.local_offset
            observation = selected[selected_index]
            generation = int(observation.growth_generation_one_based)
            values = point_values(result, local_index)
            status, reason = point_status(result, replay_ok, values)
            full_rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "candidateId": candidate_id,
                    "trajectoryId": trajectory_id,
                    "matrixIndex": matrix_index,
                    "clockId": clock_id,
                    "implementationId": implementation_id,
                    "temporalModeId": full_mode,
                    "temporalLabel": "RETROSPECTIVE_FULL_TRAJECTORY_LOCAL",
                    "selectedSequenceIndex": selected_index,
                    "rawObservationIndex": int(observation.observation_index),
                    "observationKind": str(observation.observation_kind),
                    "generation": generation,
                    "molecularStep": int(observation.batch_step),
                    "status": status,
                    "reason": reason,
                    **values,
                    "historicalLabel": historical_map.get(generation),
                    "pastOnlyCosineLabel": (
                        online_map.get(generation)
                        if observation.observation_kind == "post_fission"
                        else None
                    ),
                    "exactReplayPassed": replay_ok,
                }
            )

        prefix_mode = f"{implementation_id}_EMERGENCE_PREFIX_ENDPOINT"
        for endpoint in all_endpoints:
            generation = endpoint.generation
            base = {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": candidate_id,
                "trajectoryId": trajectory_id,
                "matrixIndex": matrix_index,
                "clockId": clock_id,
                "implementationId": implementation_id,
                "temporalModeId": prefix_mode,
                "temporalLabel": "PAST_ONLY_PREFIX_ENDPOINT",
                "generation": generation,
                "endpointSelectedSequenceIndex": endpoint.selected_sequence_index,
                "endpointRawObservationIndex": endpoint.raw_observation_index,
                "endpointObservationKind": endpoint.observation_kind,
                "priorLockedClockTransitions": endpoint.prior_locked_clock_transitions,
                "fitObservationCount": endpoint.selected_sequence_index + 1,
                "historicalLabel": historical_map.get(generation),
                "nextHistoricalLabel": historical_map.get(generation + 1),
                "pastOnlyCosineLabel": online_map.get(generation),
            }
            if endpoint.prior_locked_clock_transitions < 256:
                prefix_rows.append(
                    {
                        **base,
                        "status": "INELIGIBLE_BEFORE_256_TRANSITIONS",
                        "reason": "fewer_than_256_prior_locked_clock_transitions",
                        "synergy": None,
                        "downwardCausation": None,
                        "emergence": None,
                        "localPhiR": None,
                        "exactReplayPassed": None,
                        "futureSuffixStructuralGatePassed": None,
                        "futureSuffixExecutedSentinelPassed": None,
                    }
                )
                continue
            stop = endpoint.selected_sequence_index + 1
            prefix = np.ascontiguousarray(clr[:stop])
            prefix_hash = sha256_array(prefix)
            pre_seed, part_seed = source_seeds(
                candidate_id,
                matrix_index,
                implementation_id,
                "PREFIX_ENDPOINT",
                generation,
            )
            for purpose, seed in (
                ("source_preprocessing", pre_seed),
                ("source_partition", part_seed),
            ):
                seed_rows.append(
                    {
                        "researchStepId": RESEARCH_STEP_ID,
                        "candidateId": candidate_id,
                        "matrixIndex": matrix_index,
                        "implementationId": implementation_id,
                        "temporalModeId": prefix_mode,
                        "endpointGeneration": generation,
                        "purpose": purpose,
                        "streamId": f"S12G::{candidate_id}::M{matrix_index:02d}::{implementation_id}::G{generation:03d}::{purpose}",
                        "seed": seed,
                    }
                )
            result_prefix = run_emergence_pipeline(
                prefix,
                implementation,
                SAFE_LATTICE,
                preprocessing_seed=pre_seed,
                partition_seed=part_seed,
            )
            replay_prefix = run_emergence_pipeline(
                prefix,
                implementation,
                SAFE_LATTICE,
                preprocessing_seed=pre_seed,
                partition_seed=part_seed,
            )
            evaluations += 2
            replay_ok = result_replay_equal(result_prefix, replay_prefix)
            prefix_replay_all &= replay_ok
            values = point_values(result_prefix, len(prefix) - result_prefix.local_offset - 1)
            status, reason = point_status(result_prefix, replay_ok, values)
            sentinel_passed: bool | None = None
            structural_passed = True
            for variant_id in (
                "suffix_deletion",
                "suffix_deterministic_shuffle",
                "suffix_domain_separated_replacement",
            ):
                if variant_id == "suffix_deletion":
                    variant_prefix = np.ascontiguousarray(clr[:stop])
                else:
                    variant_full = clr.copy()
                    suffix_length = len(clr) - stop
                    if suffix_length:
                        rng = np.random.RandomState(
                            derive_seed(
                                variant_id,
                                candidate_id,
                                matrix_index,
                                implementation_id,
                                generation,
                            )
                        )
                        if variant_id == "suffix_deterministic_shuffle":
                            variant_full[stop:] = variant_full[stop:][
                                rng.permutation(suffix_length)
                            ]
                        else:
                            variant_full[stop:] = rng.normal(
                                size=variant_full[stop:].shape
                            )
                    variant_prefix = np.ascontiguousarray(variant_full[:stop])
                variant_hash = sha256_array(variant_prefix)
                structural_exact = variant_hash == prefix_hash and np.array_equal(
                    variant_prefix, prefix, equal_nan=True
                )
                structural_passed &= structural_exact
                result_exact: bool | None = None
                sentinel_name = (
                    "first"
                    if eligible_endpoints and generation == eligible_endpoints[0].generation
                    else (
                        "middle"
                        if eligible_endpoints
                        and generation
                        == eligible_endpoints[len(eligible_endpoints) // 2].generation
                        else (
                            "last"
                            if eligible_endpoints
                            and generation == eligible_endpoints[-1].generation
                            else "non_sentinel"
                        )
                    )
                )
                if generation in sentinel_generations:
                    variant = run_emergence_pipeline(
                        variant_prefix,
                        implementation,
                        SAFE_LATTICE,
                        preprocessing_seed=pre_seed,
                        partition_seed=part_seed,
                    )
                    evaluations += 1
                    result_exact = result_replay_equal(result_prefix, variant)
                    sentinel_passed = bool(
                        (True if sentinel_passed is None else sentinel_passed)
                        and structural_exact
                        and result_exact
                    )
                passed = structural_exact and (result_exact is not False)
                suffix_rows.append(
                    {
                        "researchStepId": RESEARCH_STEP_ID,
                        "candidateId": candidate_id,
                        "trajectoryId": trajectory_id,
                        "matrixIndex": matrix_index,
                        "implementationId": implementation_id,
                        "endpointGeneration": generation,
                        "validationKind": variant_id,
                        "sentinel": sentinel_name,
                        "prefixSha256": prefix_hash,
                        "variantPrefixSha256": variant_hash,
                        "structuralExact": structural_exact,
                        "resultExact": result_exact,
                        "status": "PASS" if passed else "FAIL",
                        "reason": None if passed else "future_suffix_invariance_failed",
                    }
                )
            suffix_all &= structural_passed and sentinel_passed is not False
            if not structural_passed or sentinel_passed is False:
                status = "INELIGIBLE_FUTURE_SUFFIX_INVARIANCE_FAILED"
                reason = "future_suffix_invariance_failed"
                values = {key: None for key in values}
            prefix_rows.append(
                {
                    **base,
                    "status": status,
                    "reason": reason,
                    **values,
                    "exactReplayPassed": replay_ok,
                    "futureSuffixStructuralGatePassed": structural_passed,
                    "futureSuffixExecutedSentinelPassed": sentinel_passed,
                }
            )
            partition_rows.append(
                partition_row(
                    result_prefix,
                    candidate_id=candidate_id,
                    trajectory_id=trajectory_id,
                    matrix_index=matrix_index,
                    clock_id=clock_id,
                    implementation_id=implementation_id,
                    temporal_mode_id=prefix_mode,
                    fit_kind="past_only_prefix_endpoint",
                    endpoint_generation=generation,
                    endpoint_sequence_index=endpoint.selected_sequence_index,
                    replay_passed=replay_ok,
                )
            )
            diagnostic_rows.append(
                diagnostic_row(
                    result_prefix,
                    candidate_id=candidate_id,
                    trajectory_id=trajectory_id,
                    matrix_index=matrix_index,
                    implementation_id=implementation_id,
                    temporal_mode_id=prefix_mode,
                    fit_kind="past_only_prefix_endpoint",
                    endpoint_generation=generation,
                )
            )
            if not replay_ok or not structural_passed or sentinel_passed is False:
                failures.append(
                    {
                        "failureId": f"S12G-{candidate_id}-M{matrix_index:02d}-{implementation_id}-G{generation:03d}",
                        "stage": "source_execution",
                        "candidateId": candidate_id,
                        "trajectoryId": trajectory_id,
                        "implementationId": implementation_id,
                        "temporalModeId": prefix_mode,
                        "endpointGeneration": generation,
                        "severity": "FATAL",
                        "status": status,
                        "reason": reason,
                        "gateImpact": "FAIL_CLOSED",
                        "repairAttempted": False,
                    }
                )

    outputs = {
        "labels.parquet": pd.DataFrame(labels),
        "preprocessing.parquet": pd.DataFrame(preprocessing),
        "full.parquet": pd.DataFrame(full_rows),
        "prefix.parquet": pd.DataFrame(prefix_rows),
        "partition.parquet": pd.DataFrame(partition_rows),
        "diagnostic.parquet": pd.DataFrame(diagnostic_rows),
        "suffix.parquet": pd.DataFrame(suffix_rows),
        "seeds.parquet": pd.DataFrame(seed_rows),
        "failures.parquet": pd.DataFrame(failures),
    }
    for filename, frame in outputs.items():
        frame.to_parquet(root / filename, index=False, compression="zstd")
    record = {
        "candidateId": candidate_id,
        "matrixIndex": matrix_index,
        "trajectoryId": trajectory_id,
        "clockId": clock_id,
        "inputCacheSha256": task["cacheSha256"],
        "resultRoot": str(root),
        "selectedObservationCount": len(selected),
        "eligibleEndpointCount": len(eligible_endpoints),
        "fullRows": len(full_rows),
        "prefixRows": len(prefix_rows),
        "partitionRows": len(partition_rows),
        "suffixRows": len(suffix_rows),
        "failureRows": len(failures),
        "evaluationCount": evaluations,
        "fullReplayAllPassed": full_replay_all,
        "prefixReplayAllPassed": prefix_replay_all,
        "futureSuffixAllPassed": suffix_all,
        "wallSeconds": time.perf_counter() - started_wall,
        "cpuSeconds": time.process_time() - started_cpu,
        "resumed": False,
    }
    write_json(completion, record)
    return record


def execute_tasks(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_trajectory, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"S12G task failed {task['candidateId']}/M{int(task['matrixIndex']):02d}: "
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
                        "failureRows": record["failureRows"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return sorted(records, key=lambda row: (row["candidateId"], row["matrixIndex"]))


def collate(records: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    roots = [Path(record["resultRoot"]) for record in records]
    mapping = {
        "labels.parquet": "label_values.parquet",
        "preprocessing.parquet": "preprocessing_diagnostics.parquet",
        "full.parquet": "full_source_values.parquet",
        "prefix.parquet": "prefix_endpoint_values.parquet",
        "partition.parquet": "partition_history.parquet",
        "diagnostic.parquet": "source_diagnostic_outputs.parquet",
        "suffix.parquet": "replay_suffix_validation.parquet",
        "seeds.parquet": "seed_manifest.parquet",
    }
    result: dict[str, pd.DataFrame] = {}
    for source, target in mapping.items():
        concat_parquets([root / source for root in roots], STEP_ROOT / target)
        result[target] = pd.read_parquet(STEP_ROOT / target)
    failure_frames = [
        frame
        for root in roots
        if len(frame := pd.read_parquet(root / "failures.parquet"))
    ]
    result["worker_failures"] = (
        pd.concat(failure_frames, ignore_index=True)
        if failure_frames
        else pd.DataFrame()
    )
    return result


def _analysis_seed(*identity: object) -> int:
    return derive_seed("statistics", *identity)


def _finite_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def _positive_spike_indices(values: np.ndarray) -> set[int]:
    threshold = excursion_thresholds(values)["positive3Sigma"]
    if not np.isfinite(threshold):
        return set()
    return set(np.flatnonzero(np.isfinite(values) & (values > threshold)).tolist())


def _jaccard(left: set[int], right: set[int]) -> float | None:
    union = left | right
    return float(len(left & right) / len(union)) if union else None


def _association_gate(summary: Any, coverage: float) -> bool:
    return bool(
        coverage >= 0.80
        and summary.defined_count >= 26
        and summary.positive_count >= 24
        and summary.median is not None
        and summary.median > 0
        and summary.bootstrap_lower_95 is not None
        and summary.bootstrap_lower_95 > 0
        and summary.circular_shift_positive_p is not None
        and summary.circular_shift_positive_p <= 0.05
    )


def _drift_gate(summary: Any) -> bool:
    return bool(
        summary.defined_count >= 26
        and summary.positive_count >= 19
        and summary.median_mean_difference is not None
        and summary.median_mean_difference > 0
        and summary.bootstrap_lower_95 is not None
        and summary.bootstrap_lower_95 > 0
        and summary.block_aware_positive_p is not None
        and summary.block_aware_positive_p <= 0.05
    )


def _summary_row(
    summary: Any,
    *,
    candidate_id: str,
    implementation_id: str,
    temporal_mode_id: str,
    label_id: str,
    estimand: str,
    coverage: float,
    gate_passed: bool,
) -> dict[str, Any]:
    return {
        "candidateId": candidate_id,
        "implementationId": implementation_id,
        "temporalModeId": temporal_mode_id,
        "labelId": label_id,
        "estimand": estimand,
        "definedTrajectoryCount": summary.defined_count,
        "positiveTrajectoryCount": summary.positive_count,
        "ordinaryPositivePCount": summary.ordinary_positive_p_lt_0p05_count,
        "meanCorrelation": summary.mean,
        "medianCorrelation": summary.median,
        "bootstrapLower95": summary.bootstrap_lower_95,
        "bootstrapUpper95": summary.bootstrap_upper_95,
        "circularShiftPositiveP": summary.circular_shift_positive_p,
        "circularShiftNegativeP": summary.circular_shift_negative_p,
        "effectiveEpisodeCount": summary.effective_episode_count,
        "medianLagOneAutocorrelation": summary.median_lag_one_autocorrelation,
        "finiteCoverage": coverage,
        "gatePassed": gate_passed,
    }


def _difference_row(
    summary: Any,
    *,
    candidate_id: str,
    implementation_id: str,
    temporal_mode_id: str,
    label_id: str,
    gate_passed: bool,
) -> dict[str, Any]:
    return {
        "candidateId": candidate_id,
        "implementationId": implementation_id,
        "temporalModeId": temporal_mode_id,
        "labelId": label_id,
        "definedTrajectoryCount": summary.defined_count,
        "positiveMeanDifferenceCount": summary.positive_count,
        "medianMeanDifference": summary.median_mean_difference,
        "medianMedianDifference": summary.median_median_difference,
        "bootstrapLower95": summary.bootstrap_lower_95,
        "bootstrapUpper95": summary.bootstrap_upper_95,
        "blockAwarePositiveP": summary.block_aware_positive_p,
        "pooledMannWhitneyU": summary.pooled_mann_whitney_u,
        "pooledMannWhitneyP": summary.pooled_mann_whitney_p,
        "gatePassed": gate_passed,
    }


def run_candidate_statistics(
    full: pd.DataFrame, prefix: pd.DataFrame
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str, str, str], Any],
    dict[tuple[str, str, str], Any],
]:
    association_rows: list[dict[str, Any]] = []
    association_details: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    drift_details: list[dict[str, Any]] = []
    summaries: dict[tuple[str, str, str, str], Any] = {}
    differences: dict[tuple[str, str, str], Any] = {}
    for candidate_id in CANDIDATE_IDS:
        for implementation in SourceImplementation:
            implementation_id = implementation.value
            full_mode = f"{implementation_id}_EMERGENCE_FULL"
            full_frame = full[
                (full["candidateId"] == candidate_id)
                & (full["implementationId"] == implementation_id)
            ].copy()
            full_frame["emergence"] = pd.to_numeric(
                full_frame["emergence"], errors="coerce"
            )
            full_frame["historicalLabel"] = pd.to_numeric(
                full_frame["historicalLabel"], errors="coerce"
            )
            full_coverage = float(np.isfinite(full_frame["emergence"]).mean())
            summary = trajectory_association_summary(
                full_frame,
                value_column="emergence",
                label_column="historicalLabel",
                bootstrap_seed=_analysis_seed(
                    candidate_id, implementation_id, "FULL", "HIST", "bootstrap"
                ),
                circular_seed=_analysis_seed(
                    candidate_id, implementation_id, "FULL", "HIST", "circular"
                ),
                replicates=4096,
            )
            gate = _association_gate(summary, full_coverage)
            summaries[(candidate_id, implementation_id, "FULL", "CURRENT_HISTORICAL")] = summary
            association_rows.append(
                _summary_row(
                    summary,
                    candidate_id=candidate_id,
                    implementation_id=implementation_id,
                    temporal_mode_id=full_mode,
                    label_id=HISTORICAL_LABEL_ID,
                    estimand="RETROSPECTIVE_CURRENT_GENERATION",
                    coverage=full_coverage,
                    gate_passed=gate,
                )
            )
            for trajectory_id, rho in summary.correlations.items():
                group = full_frame[full_frame["trajectoryId"] == trajectory_id]
                association_details.append(
                    {
                        "candidateId": candidate_id,
                        "matrixIndex": int(group["matrixIndex"].iloc[0]),
                        "trajectoryId": trajectory_id,
                        "implementationId": implementation_id,
                        "temporalMode": "FULL",
                        "estimand": "CURRENT_HISTORICAL",
                        "correlation": rho,
                        "ordinaryP": summary.ordinary_p_values[trajectory_id],
                    }
                )
            difference = replicator_drift_summary(
                full_frame,
                value_column="emergence",
                label_column="historicalLabel",
                bootstrap_seed=_analysis_seed(
                    candidate_id, implementation_id, "FULL", "drift", "bootstrap"
                ),
                permutation_seed=_analysis_seed(
                    candidate_id, implementation_id, "FULL", "drift", "permutation"
                ),
                replicates=4096,
            )
            difference_gate = _drift_gate(difference)
            differences[(candidate_id, implementation_id, "FULL")] = difference
            drift_rows.append(
                _difference_row(
                    difference,
                    candidate_id=candidate_id,
                    implementation_id=implementation_id,
                    temporal_mode_id=full_mode,
                    label_id=HISTORICAL_LABEL_ID,
                    gate_passed=difference_gate,
                )
            )
            for trajectory_id, value in difference.mean_differences.items():
                group = full_frame[full_frame["trajectoryId"] == trajectory_id]
                drift_details.append(
                    {
                        "candidateId": candidate_id,
                        "matrixIndex": int(group["matrixIndex"].iloc[0]),
                        "trajectoryId": trajectory_id,
                        "implementationId": implementation_id,
                        "temporalMode": "FULL",
                        "meanDifference": value,
                        "medianDifference": difference.median_differences[trajectory_id],
                    }
                )

            prefix_mode = f"{implementation_id}_EMERGENCE_PREFIX_ENDPOINT"
            prefix_frame = prefix[
                (prefix["candidateId"] == candidate_id)
                & (prefix["implementationId"] == implementation_id)
                & (prefix["priorLockedClockTransitions"] >= 256)
            ].copy()
            prefix_frame["emergence"] = pd.to_numeric(
                prefix_frame["emergence"], errors="coerce"
            )
            prefix_coverage = float(np.isfinite(prefix_frame["emergence"]).mean())
            estimands = (
                ("historicalLabel", HISTORICAL_LABEL_ID, "CURRENT_HISTORICAL"),
                ("nextHistoricalLabel", HISTORICAL_LABEL_ID, "NEXT_HISTORICAL"),
                ("pastOnlyCosineLabel", ONLINE_LABEL_ID, "CURRENT_PAST_ONLY_COSINE"),
            )
            for label_column, label_id, estimand in estimands:
                working = prefix_frame.copy()
                working[label_column] = pd.to_numeric(
                    working[label_column], errors="coerce"
                )
                summary = trajectory_association_summary(
                    working,
                    value_column="emergence",
                    label_column=label_column,
                    bootstrap_seed=_analysis_seed(
                        candidate_id,
                        implementation_id,
                        "PREFIX",
                        estimand,
                        "bootstrap",
                    ),
                    circular_seed=_analysis_seed(
                        candidate_id,
                        implementation_id,
                        "PREFIX",
                        estimand,
                        "circular",
                    ),
                    replicates=4096,
                )
                gate = _association_gate(summary, prefix_coverage)
                summaries[(candidate_id, implementation_id, "PREFIX", estimand)] = summary
                association_rows.append(
                    _summary_row(
                        summary,
                        candidate_id=candidate_id,
                        implementation_id=implementation_id,
                        temporal_mode_id=prefix_mode,
                        label_id=label_id,
                        estimand=estimand,
                        coverage=prefix_coverage,
                        gate_passed=gate,
                    )
                )
                for trajectory_id, rho in summary.correlations.items():
                    group = working[working["trajectoryId"] == trajectory_id]
                    association_details.append(
                        {
                            "candidateId": candidate_id,
                            "matrixIndex": int(group["matrixIndex"].iloc[0]),
                            "trajectoryId": trajectory_id,
                            "implementationId": implementation_id,
                            "temporalMode": "PREFIX",
                            "estimand": estimand,
                            "correlation": rho,
                            "ordinaryP": summary.ordinary_p_values[trajectory_id],
                        }
                    )
            prefix_frame["historicalLabel"] = pd.to_numeric(
                prefix_frame["historicalLabel"], errors="coerce"
            )
            difference = replicator_drift_summary(
                prefix_frame,
                value_column="emergence",
                label_column="historicalLabel",
                bootstrap_seed=_analysis_seed(
                    candidate_id, implementation_id, "PREFIX", "drift", "bootstrap"
                ),
                permutation_seed=_analysis_seed(
                    candidate_id,
                    implementation_id,
                    "PREFIX",
                    "drift",
                    "permutation",
                ),
                replicates=4096,
            )
            differences[(candidate_id, implementation_id, "PREFIX")] = difference
            drift_rows.append(
                _difference_row(
                    difference,
                    candidate_id=candidate_id,
                    implementation_id=implementation_id,
                    temporal_mode_id=prefix_mode,
                    label_id=HISTORICAL_LABEL_ID,
                    gate_passed=_drift_gate(difference),
                )
            )
            for trajectory_id, value in difference.mean_differences.items():
                group = prefix_frame[prefix_frame["trajectoryId"] == trajectory_id]
                drift_details.append(
                    {
                        "candidateId": candidate_id,
                        "matrixIndex": int(group["matrixIndex"].iloc[0]),
                        "trajectoryId": trajectory_id,
                        "implementationId": implementation_id,
                        "temporalMode": "PREFIX",
                        "meanDifference": value,
                        "medianDifference": difference.median_differences[trajectory_id],
                    }
                )
    return (
        pd.DataFrame(association_rows),
        pd.DataFrame(association_details),
        pd.DataFrame(drift_rows),
        pd.DataFrame(drift_details),
        summaries,
        differences,
    )


def run_temporal_statistics(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    temporal_rows: list[dict[str, Any]] = []
    spike_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        for implementation in SourceImplementation:
            implementation_id = implementation.value
            frame = full[
                (full["candidateId"] == candidate_id)
                & (full["implementationId"] == implementation_id)
            ].copy()
            frame["emergence"] = pd.to_numeric(frame["emergence"], errors="coerce")
            rows = temporal_structure_rows(frame, value_column="emergence")
            for row in rows:
                common = {
                    "candidateId": candidate_id,
                    "implementationId": implementation_id,
                    "temporalModeId": f"{implementation_id}_EMERGENCE_FULL",
                }
                temporal_rows.append(
                    {
                        **common,
                        "rowType": row["rowType"],
                        "trajectoryId": row.get("trajectoryId"),
                        "nFinite": row.get("nFinite"),
                        "ljungBoxLag": row.get("ljungBoxLag"),
                        "ljungBoxStatistic": row.get("ljungBoxStatistic"),
                        "ljungBoxPValue": row.get("ljungBoxPValue"),
                        "differencedLjungBoxLag": row.get("differencedLjungBoxLag"),
                        "differencedLjungBoxStatistic": row.get(
                            "differencedLjungBoxStatistic"
                        ),
                        "differencedLjungBoxPValue": row.get(
                            "differencedLjungBoxPValue"
                        ),
                        "aggregateTrendSlope": row.get("aggregateTrendSlope"),
                        "aggregateTrendPValue": row.get("aggregateTrendPValue"),
                    }
                )
                if row["rowType"] == "TRAJECTORY":
                    spike_rows.append(
                        {
                            **common,
                            "trajectoryId": row["trajectoryId"],
                            "nFinite": row["nFinite"],
                            "positive3SigmaCount": row["positive3SigmaCount"],
                            "negative3SigmaCount": row["negative3SigmaCount"],
                            "robustPositiveCount": row["robustPositiveCount"],
                            "robustNegativeCount": row["robustNegativeCount"],
                            "peakCount": row["peakCount"],
                            "medianPeakWidthObservations": row[
                                "medianPeakWidthObservations"
                            ],
                            "medianPeakProminence": row["medianPeakProminence"],
                            "medianPeakSpacingObservations": row[
                                "medianPeakSpacingObservations"
                            ],
                        }
                    )
    return pd.DataFrame(temporal_rows), pd.DataFrame(spike_rows)


def _partition_labels(row: pd.Series, dimensions: set[int]) -> list[int]:
    p1 = set(json.loads(row["partition1Json"]))
    p2 = set(json.loads(row["partition2Json"]))
    return [0 if dimension in p1 else 1 if dimension in p2 else -1 for dimension in sorted(dimensions)]


def partition_ari(left: pd.Series, right: pd.Series) -> float | None:
    left_dims = set(json.loads(left["partition1Json"])) | set(
        json.loads(left["partition2Json"])
    )
    right_dims = set(json.loads(right["partition1Json"])) | set(
        json.loads(right["partition2Json"])
    )
    common = left_dims & right_dims
    if len(common) < 2:
        return None
    return float(
        adjusted_rand_score(
            _partition_labels(left, common), _partition_labels(right, common)
        )
    )


def run_metric_identity(full: pd.DataFrame, prefix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode, frame, index_column in (
        ("FULL", full, "selectedSequenceIndex"),
        ("PREFIX", prefix, "generation"),
    ):
        for keys, group in frame.groupby(
            ["candidateId", "trajectoryId", "matrixIndex", "implementationId"],
            sort=True,
        ):
            candidate_id, trajectory_id, matrix_index, implementation_id = keys
            emergence = pd.to_numeric(group["emergence"], errors="coerce").to_numpy()
            local_phi = pd.to_numeric(group["localPhiR"], errors="coerce").to_numpy()
            labels = pd.to_numeric(group["historicalLabel"], errors="coerce").to_numpy()
            mask = np.isfinite(emergence) & np.isfinite(local_phi)
            x, y = emergence[mask], local_phi[mask]
            if len(x) >= 3:
                spearman = _finite_spearman(x, y)
                pearson = finite_pearson(x, y)
                sign = float(np.mean(np.signbit(x) == np.signbit(y)))
                agreement, changed = rank_agreement(x, y)
                spike = _jaccard(
                    _positive_spike_indices(x), _positive_spike_indices(y)
                )
            else:
                spearman = pearson = sign = agreement = changed = spike = None
            rep_emergence = _finite_spearman(emergence, labels)
            rep_phi = _finite_spearman(local_phi, labels)
            rows.append(
                {
                    "candidateId": candidate_id,
                    "trajectoryId": trajectory_id,
                    "matrixIndex": int(matrix_index),
                    "implementationId": implementation_id,
                    "temporalModeId": mode,
                    "sharedFiniteCount": int(mask.sum()),
                    "spearman": spearman,
                    "pearson": pearson,
                    "signAgreement": sign,
                    "rankAgreement": agreement,
                    "fractionRanksChangedOver10Points": changed,
                    "spikeJaccard": spike,
                    "partitionIdentity": True,
                    "replicationAssociationEmergence": rep_emergence,
                    "replicationAssociationLocalPhiR": rep_phi,
                    "replicationAssociationDifference": (
                        rep_emergence - rep_phi
                        if rep_emergence is not None and rep_phi is not None
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_future_dependence(
    full: pd.DataFrame, prefix: pd.DataFrame, partitions: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible_prefix = prefix[prefix["priorLockedClockTransitions"] >= 256]
    for keys, prefix_group in eligible_prefix.groupby(
        ["candidateId", "trajectoryId", "matrixIndex", "implementationId"], sort=True
    ):
        candidate_id, trajectory_id, matrix_index, implementation_id = keys
        full_group = full[
            (full["candidateId"] == candidate_id)
            & (full["trajectoryId"] == trajectory_id)
            & (full["implementationId"] == implementation_id)
        ][["selectedSequenceIndex", "emergence", "historicalLabel"]].copy()
        joined = prefix_group.merge(
            full_group,
            left_on="endpointSelectedSequenceIndex",
            right_on="selectedSequenceIndex",
            suffixes=("Prefix", "Full"),
            how="left",
            validate="one_to_one",
        ).sort_values("generation")
        prefix_values = pd.to_numeric(joined["emergencePrefix"], errors="coerce").to_numpy()
        full_values = pd.to_numeric(joined["emergenceFull"], errors="coerce").to_numpy()
        labels = pd.to_numeric(joined["historicalLabelPrefix"], errors="coerce").to_numpy()
        mask = np.isfinite(prefix_values) & np.isfinite(full_values)
        x, y = full_values[mask], prefix_values[mask]
        full_iqr = float(np.subtract(*np.quantile(x, [0.75, 0.25]))) if len(x) else None
        median_abs = float(np.median(np.abs(x - y))) if len(x) else None
        _, changed = rank_agreement(x, y) if len(x) >= 2 else (None, None)
        full_rep = _finite_spearman(full_values, labels)
        prefix_rep = _finite_spearman(prefix_values, labels)
        full_partition = partitions[
            (partitions["candidateId"] == candidate_id)
            & (partitions["trajectoryId"] == trajectory_id)
            & (partitions["implementationId"] == implementation_id)
            & (partitions["fitKind"] == "completed_trajectory")
        ]
        prefix_partitions = partitions[
            (partitions["candidateId"] == candidate_id)
            & (partitions["trajectoryId"] == trajectory_id)
            & (partitions["implementationId"] == implementation_id)
            & (partitions["fitKind"] == "past_only_prefix_endpoint")
        ]
        aris: list[float] = []
        if len(full_partition) == 1:
            left = full_partition.iloc[0]
            for _, right in prefix_partitions.iterrows():
                ari = partition_ari(left, right)
                if ari is not None:
                    aris.append(ari)
        rows.append(
            {
                "candidateId": candidate_id,
                "trajectoryId": trajectory_id,
                "matrixIndex": int(matrix_index),
                "implementationId": implementation_id,
                "sharedEndpointCount": int(mask.sum()),
                "medianAbsoluteDifference": median_abs,
                "fullIqr": full_iqr,
                "normalizedMedianAbsoluteDifference": (
                    median_abs / full_iqr
                    if median_abs is not None and full_iqr not in (None, 0.0)
                    else None
                ),
                "spearman": _finite_spearman(x, y),
                "pearson": finite_pearson(x, y) if len(x) >= 3 else None,
                "signAgreement": (
                    float(np.mean(np.signbit(x) == np.signbit(y))) if len(x) else None
                ),
                "fractionRanksChangedOver10Points": changed,
                "spikeJaccard": _jaccard(
                    _positive_spike_indices(x), _positive_spike_indices(y)
                )
                if len(x)
                else None,
                "medianPartitionAdjustedRand": float(np.median(aris)) if aris else None,
                "fullReplicationAssociation": full_rep,
                "prefixReplicationAssociation": prefix_rep,
                "replicationAssociationDifference": (
                    full_rep - prefix_rep
                    if full_rep is not None and prefix_rep is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def run_cross_candidate(
    labels: pd.DataFrame,
    association_details: pd.DataFrame,
    drift_details: pd.DataFrame,
    partitions: pd.DataFrame,
    shared_audit: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairing = {
        int(item["matrixIndex"]): bool(item["sharedIdentity"])
        for item in shared_audit["rows"]
    }
    pairs = (
        (CANDIDATE_IDS[0], CANDIDATE_IDS[1]),
        (CANDIDATE_IDS[0], CANDIDATE_IDS[2]),
        (CANDIDATE_IDS[1], CANDIDATE_IDS[2]),
    )
    for matrix_index in range(32):
        identity_matched = pairing[matrix_index]
        for candidate_a, candidate_b in pairs:
            pairing_status = "PAIRED" if identity_matched else "UNPAIRED"
            for label_id in (HISTORICAL_LABEL_ID, ONLINE_LABEL_ID):
                left = labels[
                    (labels["candidateId"] == candidate_a)
                    & (labels["matrixIndex"] == matrix_index)
                    & (labels["labelId"] == label_id)
                ].sort_values("generation")
                right = labels[
                    (labels["candidateId"] == candidate_b)
                    & (labels["matrixIndex"] == matrix_index)
                    & (labels["labelId"] == label_id)
                ].sort_values("generation")
                if identity_matched and len(left) == len(right) == 100:
                    a = left["isReplicator"].astype(bool).to_numpy()
                    b = right["isReplicator"].astype(bool).to_numpy()
                    agreement = float(np.mean(a == b))
                    ari = float(adjusted_rand_score(a, b))
                    status, reason = "PASS", None
                else:
                    agreement = ari = None
                    status, reason = "UNPAIRED", "matrix_or_initial_identity_not_shared"
                for metric, value in (
                    (f"{label_id}_binary_agreement", agreement),
                    (f"{label_id}_adjusted_rand", ari),
                ):
                    rows.append(
                        {
                            "analysisType": "LABEL_COMPARISON",
                            "matrixIndex": matrix_index,
                            "candidateA": candidate_a,
                            "candidateB": candidate_b,
                            "pairingStatus": pairing_status,
                            "identityMatched": identity_matched,
                            "metric": metric,
                            "valueA": value,
                            "valueB": None,
                            "difference": None,
                            "status": status,
                            "reason": reason,
                        }
                    )
            for temporal_mode in ("FULL", "PREFIX"):
                detail = association_details[
                    (association_details["matrixIndex"] == matrix_index)
                    & (
                        association_details["implementationId"]
                        == SourceImplementation.IIGR.value
                    )
                    & (association_details["temporalMode"] == temporal_mode)
                    & (association_details["estimand"] == "CURRENT_HISTORICAL")
                ]
                a_rows = detail[detail["candidateId"] == candidate_a]
                b_rows = detail[detail["candidateId"] == candidate_b]
                a = (
                    float(a_rows["correlation"].iloc[0])
                    if len(a_rows) and pd.notna(a_rows["correlation"].iloc[0])
                    else None
                )
                b = (
                    float(b_rows["correlation"].iloc[0])
                    if len(b_rows) and pd.notna(b_rows["correlation"].iloc[0])
                    else None
                )
                rows.append(
                    {
                        "analysisType": "PRIMARY_ASSOCIATION",
                        "matrixIndex": matrix_index,
                        "candidateA": candidate_a,
                        "candidateB": candidate_b,
                        "pairingStatus": pairing_status,
                        "identityMatched": identity_matched,
                        "metric": f"IIGR_{temporal_mode}_current_historical_spearman",
                        "valueA": a,
                        "valueB": b,
                        "difference": a - b if a is not None and b is not None else None,
                        "status": "PASS" if identity_matched else "UNPAIRED",
                        "reason": None
                        if identity_matched
                        else "matrix_or_initial_identity_not_shared",
                    }
                )
                difference = drift_details[
                    (drift_details["matrixIndex"] == matrix_index)
                    & (drift_details["implementationId"] == SourceImplementation.IIGR.value)
                    & (drift_details["temporalMode"] == temporal_mode)
                ]
                a_rows = difference[difference["candidateId"] == candidate_a]
                b_rows = difference[difference["candidateId"] == candidate_b]
                a = (
                    float(a_rows["meanDifference"].iloc[0])
                    if len(a_rows) and pd.notna(a_rows["meanDifference"].iloc[0])
                    else None
                )
                b = (
                    float(b_rows["meanDifference"].iloc[0])
                    if len(b_rows) and pd.notna(b_rows["meanDifference"].iloc[0])
                    else None
                )
                rows.append(
                    {
                        "analysisType": "REPLICATOR_DRIFT",
                        "matrixIndex": matrix_index,
                        "candidateA": candidate_a,
                        "candidateB": candidate_b,
                        "pairingStatus": pairing_status,
                        "identityMatched": identity_matched,
                        "metric": f"IIGR_{temporal_mode}_mean_difference",
                        "valueA": a,
                        "valueB": b,
                        "difference": a - b if a is not None and b is not None else None,
                        "status": "PASS" if identity_matched else "UNPAIRED",
                        "reason": None
                        if identity_matched
                        else "matrix_or_initial_identity_not_shared",
                    }
                )
            if identity_matched:
                part = partitions[
                    (partitions["matrixIndex"] == matrix_index)
                    & (partitions["implementationId"] == SourceImplementation.IIGR.value)
                    & (partitions["fitKind"] == "completed_trajectory")
                ]
                left = part[part["candidateId"] == candidate_a]
                right = part[part["candidateId"] == candidate_b]
                ari = (
                    partition_ari(left.iloc[0], right.iloc[0])
                    if len(left) == len(right) == 1
                    else None
                )
            else:
                ari = None
            rows.append(
                {
                    "analysisType": "FULL_PARTITION",
                    "matrixIndex": matrix_index,
                    "candidateA": candidate_a,
                    "candidateB": candidate_b,
                    "pairingStatus": pairing_status,
                    "identityMatched": identity_matched,
                    "metric": "IIGR_full_partition_ARI",
                    "valueA": ari,
                    "valueB": None,
                    "difference": None,
                    "status": "PASS" if identity_matched else "UNPAIRED",
                    "reason": None
                    if identity_matched
                    else "matrix_or_initial_identity_not_shared",
                }
            )
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
        iigr = SourceImplementation.IIGR.value
        phirl = SourceImplementation.PHIRL.value
        full_assoc = associations[
            (associations["candidateId"] == candidate_id)
            & (associations["implementationId"] == iigr)
            & (associations["estimand"] == "RETROSPECTIVE_CURRENT_GENERATION")
        ].iloc[0]
        full_drift = drift[
            (drift["candidateId"] == candidate_id)
            & (drift["implementationId"] == iigr)
            & (drift["temporalModeId"].str.endswith("_FULL"))
        ].iloc[0]
        prefix_assoc = associations[
            (associations["candidateId"] == candidate_id)
            & (associations["implementationId"] == iigr)
            & (associations["estimand"] == "CURRENT_HISTORICAL")
            & (associations["temporalModeId"].str.endswith("_PREFIX_ENDPOINT"))
        ].iloc[0]
        phirl_full_opposite = significant_opposite(
            summaries[(candidate_id, phirl, "FULL", "CURRENT_HISTORICAL")]
        )
        phirl_prefix_opposite = significant_opposite(
            summaries[(candidate_id, phirl, "PREFIX", "CURRENT_HISTORICAL")]
        )
        full_coherent = bool(full_assoc["gatePassed"] and full_drift["gatePassed"])
        prefix_gate = bool(prefix_assoc["gatePassed"] and not phirl_prefix_opposite)
        aggregate = temporal[
            (temporal["candidateId"] == candidate_id)
            & (temporal["implementationId"] == iigr)
            & (temporal["rowType"] == "AGGREGATE")
        ].iloc[0]
        candidate_spikes = spike[
            (spike["candidateId"] == candidate_id)
            & (spike["implementationId"] == iigr)
        ]
        runs_spiked = int((candidate_spikes["positive3SigmaCount"] > 0).sum())
        raw_significant = int(
            (
                temporal[
                    (temporal["candidateId"] == candidate_id)
                    & (temporal["implementationId"] == iigr)
                    & (temporal["rowType"] == "TRAJECTORY")
                ]["ljungBoxPValue"]
                <= 0.05
            ).sum()
        )
        diff_significant = int(
            (
                temporal[
                    (temporal["candidateId"] == candidate_id)
                    & (temporal["implementationId"] == iigr)
                    & (temporal["rowType"] == "TRAJECTORY")
                ]["differencedLjungBoxPValue"]
                <= 0.05
            ).sum()
        )
        punctuated = bool(
            runs_spiked >= 24
            and pd.notna(aggregate["aggregateTrendPValue"])
            and float(aggregate["aggregateTrendPValue"]) > 0.05
            and raw_significant >= 28
            and diff_significant <= 0
        )
        full_coverage = float(
            full[full["candidateId"] == candidate_id]
            .groupby("implementationId")["emergence"]
            .apply(lambda values: np.isfinite(pd.to_numeric(values, errors="coerce")).mean())
            .min()
        )
        eligible_prefix = prefix[
            (prefix["candidateId"] == candidate_id)
            & (prefix["priorLockedClockTransitions"] >= 256)
        ]
        prefix_coverage = float(
            eligible_prefix.groupby("implementationId")["emergence"]
            .apply(lambda values: np.isfinite(pd.to_numeric(values, errors="coerce")).mean())
            .min()
        )
        operational = full_coverage >= 0.80 and prefix_coverage >= 0.80
        if prefix_gate:
            candidate_class = "CANDIDATE_PROSPECTIVE_SUPPORT"
        elif full_coherent:
            candidate_class = "CANDIDATE_RETROSPECTIVE_SUPPORT"
        else:
            candidate_class = "CANDIDATE_NOT_SUPPORTED"
        rows.append(
            {
                "candidateId": candidate_id,
                "primaryFullAssociationGate": bool(full_assoc["gatePassed"]),
                "primaryFullDriftGate": bool(full_drift["gatePassed"]),
                "primaryFullCoherent": full_coherent,
                "primaryPrefixGate": prefix_gate,
                "punctuatedGate": punctuated,
                "phirlOppositeFull": phirl_full_opposite,
                "phirlOppositePrefix": phirl_prefix_opposite,
                "operationalCoverageGate": operational,
                "candidateClassification": candidate_class,
            }
        )
    frame = pd.DataFrame(rows)
    prefix_passes = frame["primaryPrefixGate"].astype(bool).tolist()
    full_passes = frame["primaryFullCoherent"].astype(bool).tolist()
    if all(prefix_passes):
        classification = "ENSEMBLE_PROSPECTIVE_SOURCE_EMERGENCE_SUPPORT"
    elif all(full_passes):
        classification = "ENSEMBLE_RETROSPECTIVE_SOURCE_EMERGENCE_SUPPORT"
    elif any(prefix_passes) or any(full_passes):
        classification = "CANDIDATE_SENSITIVE_UNDERDETERMINED"
    elif not any(prefix_passes) and not any(full_passes):
        classification = "ENSEMBLE_WIDE_NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE"
    else:
        classification = "UNDERDETERMINED"
    payload = {
        "schema": "eidosoma.e01.s12g_classification.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "classification": classification,
        "candidateResults": frame.to_dict("records"),
        "ensemblePositiveRequiresAllThree": True,
        "candidateWeightsUsed": False,
        "s13Status": "BLOCKED_PENDING_S12G_HUMAN_REVIEW",
    }
    return frame, payload


def make_figures(
    labels: pd.DataFrame,
    full: pd.DataFrame,
    prefix: pd.DataFrame,
    association_details: pd.DataFrame,
    metric_identity: pd.DataFrame,
    adjudication: pd.DataFrame,
) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    colors = ["#4477AA", "#EE6677", "#228833"]
    label_summary = (
        labels.assign(isReplicator=labels["isReplicator"].astype(float))
        .groupby(["candidateId", "matrixIndex", "labelId"])["isReplicator"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(3)
    for offset, label_id in zip((-0.18, 0.18), (HISTORICAL_LABEL_ID, ONLINE_LABEL_ID), strict=True):
        values = [
            label_summary[
                (label_summary["candidateId"] == candidate)
                & (label_summary["labelId"] == label_id)
            ]["isReplicator"].to_numpy()
            for candidate in CANDIDATE_IDS
        ]
        ax.boxplot(values, positions=positions + offset, widths=0.28, patch_artist=True)
    ax.set_xticks(positions, ["Candidate 1", "Candidate 2", "Candidate 3"])
    ax.set_ylabel("Replicator-label occupancy")
    ax.set_title("Frozen label fingerprints (historical and past-only cosine)")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "label_fingerprints.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
    for axis, candidate_id, color in zip(axes, CANDIDATE_IDS, colors, strict=True):
        for matrix_index in (0, 1, 2):
            group = full[
                (full["candidateId"] == candidate_id)
                & (full["matrixIndex"] == matrix_index)
                & (full["implementationId"] == SourceImplementation.IIGR.value)
            ].sort_values("selectedSequenceIndex")
            axis.plot(
                group["selectedSequenceIndex"],
                group["emergence"],
                lw=0.7,
                alpha=0.75,
                label=f"M{matrix_index:02d}",
            )
        axis.set_title(candidate_id)
        axis.set_ylabel("Emergence")
        axis.legend(ncol=3, fontsize=7)
    axes[-1].set_xlabel("Locked-clock state index")
    fig.suptitle("IIGR retrospective full-trajectory source emergence")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "full_emergence_trajectories.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    positions, data, labels_out = [], [], []
    position = 0
    for candidate_id in CANDIDATE_IDS:
        for mode in ("FULL", "PREFIX"):
            values = association_details[
                (association_details["candidateId"] == candidate_id)
                & (association_details["implementationId"] == SourceImplementation.IIGR.value)
                & (association_details["temporalMode"] == mode)
                & (association_details["estimand"] == "CURRENT_HISTORICAL")
            ]["correlation"].dropna()
            positions.append(position)
            data.append(values.to_numpy())
            labels_out.append(f"{candidate_id[-2:]}\n{mode}")
            position += 1
        position += 0.4
    ax.boxplot(data, positions=positions, widths=0.65)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(positions, labels_out)
    ax.set_ylabel("Within-trajectory Spearman rho")
    ax.set_title("Primary IIGR current-generation associations")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "association_distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, candidate_id, color in zip(axes, CANDIDATE_IDS, colors, strict=True):
        p = prefix[
            (prefix["candidateId"] == candidate_id)
            & (prefix["implementationId"] == SourceImplementation.IIGR.value)
            & (prefix["priorLockedClockTransitions"] >= 256)
        ]
        f = full[
            (full["candidateId"] == candidate_id)
            & (full["implementationId"] == SourceImplementation.IIGR.value)
        ][["trajectoryId", "selectedSequenceIndex", "emergence"]]
        joined = p.merge(
            f,
            left_on=["trajectoryId", "endpointSelectedSequenceIndex"],
            right_on=["trajectoryId", "selectedSequenceIndex"],
            suffixes=("Prefix", "Full"),
        )
        axis.scatter(joined["emergenceFull"], joined["emergencePrefix"], s=4, alpha=0.25, color=color)
        axis.set_title(candidate_id)
        axis.set_xlabel("Full-fit emergence")
        axis.set_ylabel("Prefix emergence")
    fig.suptitle("Retrospective full fit versus past-only prefix")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "full_prefix_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    subset = metric_identity[
        (metric_identity["implementationId"] == SourceImplementation.IIGR.value)
        & (metric_identity["temporalModeId"] == "FULL")
    ]
    for candidate_id, color in zip(CANDIDATE_IDS, colors, strict=True):
        group = subset[subset["candidateId"] == candidate_id]
        ax.scatter(
            group["replicationAssociationLocalPhiR"],
            group["replicationAssociationEmergence"],
            s=22,
            alpha=0.7,
            color=color,
            label=candidate_id,
        )
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Corrected local Phi-r association")
    ax.set_ylabel("Source-defined emergence association")
    ax.set_title("Frozen metric-identity comparator")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "metric_identity_comparison.png", dpi=180)
    plt.close(fig)

    matrix = adjudication[
        [
            "primaryFullAssociationGate",
            "primaryFullDriftGate",
            "primaryFullCoherent",
            "primaryPrefixGate",
            "punctuatedGate",
            "operationalCoverageGate",
        ]
    ].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_yticks(range(3), ["Candidate 1", "Candidate 2", "Candidate 3"])
    ax.set_xticks(
        range(matrix.shape[1]),
        ["Full assoc", "Full drift", "Full coherent", "Prefix", "Punctuated", "Operational"],
        rotation=25,
        ha="right",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, "PASS" if matrix[row, column] else "FAIL", ha="center", va="center", fontsize=8)
    ax.set_title("Frozen all-candidate decision matrix")
    fig.colorbar(image, ax=ax, ticks=[0, 1], shrink=0.75)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "ensemble_decision_matrix.png", dpi=180)
    plt.close(fig)


def validate_immutable_prior() -> dict[str, Any]:
    baseline = json.loads(
        (STEP_ROOT / "immutable_prior_baseline.json").read_text(encoding="utf-8")
    )
    changed: list[dict[str, Any]] = []
    for item in baseline["researchStepFiles"] + baseline["lockedTrajectoryCaches"]:
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
        "schema": "eidosoma.e01.s12g_immutable_prior_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "researchStepFileCount": len(baseline["researchStepFiles"]),
        "lockedTrajectoryCacheCount": len(baseline["lockedTrajectoryCaches"]),
        "changedCount": len(changed),
        "changed": changed,
        "passed": not changed,
    }
    write_json(STEP_ROOT / "immutable_prior_validation.json", payload)
    return payload


def validate_schemas() -> dict[str, Any]:
    contract = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["tables"]
    rows: list[dict[str, Any]] = []
    for filename, required_columns in contract.items():
        path = STEP_ROOT / filename
        if not path.is_file():
            rows.append(
                {
                    "path": filename,
                    "exists": False,
                    "rowCount": None,
                    "missingColumns": required_columns,
                    "passed": False,
                }
            )
            continue
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        missing = [column for column in required_columns if column not in frame.columns]
        rows.append(
            {
                "path": filename,
                "exists": True,
                "rowCount": len(frame),
                "missingColumns": missing,
                "passed": not missing,
            }
        )
    return {
        "schema": "eidosoma.e01.s12g_schema_validation.v1",
        "tables": rows,
        "passed": all(row["passed"] for row in rows),
    }


def failure_rows_from_statuses(
    full: pd.DataFrame, prefix: pd.DataFrame, worker_failures: pd.DataFrame
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(worker_failures):
        rows.extend(worker_failures.to_dict("records"))
    for mode, frame in (("FULL", full), ("PREFIX", prefix)):
        for keys, group in frame[frame["status"] != "ELIGIBLE"].groupby(
            ["candidateId", "implementationId", "status", "reason"],
            dropna=False,
            sort=True,
        ):
            candidate_id, implementation_id, status, reason = keys
            expected = status == "INELIGIBLE_BEFORE_256_TRANSITIONS"
            rows.append(
                {
                    "failureId": f"S12G-STATUS-{mode}-{candidate_id}-{implementation_id}-{status}",
                    "stage": "status_aggregation",
                    "candidateId": candidate_id,
                    "trajectoryId": None,
                    "implementationId": implementation_id,
                    "temporalModeId": mode,
                    "endpointGeneration": None,
                    "severity": "EXPECTED_INELIGIBILITY" if expected else "NONFATAL",
                    "status": status,
                    "reason": f"{reason};count={len(group)}",
                    "gateImpact": "EXCLUDED_FROM_NUMERIC_ANALYSIS",
                    "repairAttempted": False,
                }
            )
    return rows


def artifact_manifest(required: list[str]) -> dict[str, Any]:
    files = [path for path in sorted(STEP_ROOT.rglob("*")) if path.is_file()]
    entries = [
        {
            "relativePath": str(path.relative_to(STEP_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
        if path.name != "artifact_manifest.json"
    ]
    present = {entry["relativePath"] for entry in entries}
    missing = [item for item in required if item not in present and item != "artifact_manifest.json"]
    payload = {
        "schema": "eidosoma.e01.s12g_artifact_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "artifacts": entries,
        "artifactCountExcludingSelf": len(entries),
        "totalBytesExcludingSelf": sum(item["bytes"] for item in entries),
        "requiredMissing": missing,
        "under30GiB": sum(item["bytes"] for item in entries) <= 30 * 1024**3,
        "passed": not missing
        and sum(item["bytes"] for item in entries) <= 30 * 1024**3,
    }
    write_json(STEP_ROOT / "artifact_manifest.json", payload)
    return payload


def recommendation_for(classification: str) -> str:
    if classification == "ENSEMBLE_PROSPECTIVE_SOURCE_EMERGENCE_SUPPORT":
        return (
            "Return for human review; S13 remains blocked. A separately preregistered "
            "baseline-only S13 could be considered, but S12G authorizes no prediction or intervention."
        )
    if classification == "ENSEMBLE_RETROSPECTIVE_SOURCE_EMERGENCE_SUPPORT":
        return (
            "Return for human review; retain any resemblance as retrospective and potentially "
            "future-dependent, and do not authorize early-warning, intervention, or S13 automatically."
        )
    if classification == "CANDIDATE_SENSITIVE_UNDERDETERMINED":
        return (
            "Do not select a favorable candidate or begin S13; the downstream conclusion depends "
            "on unresolved daughter, overshoot, or clock semantics and requires human review."
        )
    if classification == "ENSEMBLE_WIDE_NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE":
        return (
            "Do not begin S13; close this source-informed E01 emergence branch or consider a "
            "separately authorized methodological experiment only after human review."
        )
    return "Keep S13 blocked and return for mandatory human review without further repair."


def build_report(
    *,
    classification: dict[str, Any],
    associations: pd.DataFrame,
    drift: pd.DataFrame,
    future: pd.DataFrame,
    metric_identity: pd.DataFrame,
    adjudication: pd.DataFrame,
    runtime: dict[str, Any],
    validation: dict[str, Any],
    failures: pd.DataFrame,
    input_manifest: pd.DataFrame,
) -> str:
    outcome = classification["classification"]
    recommendation = recommendation_for(outcome)
    primary = associations[
        associations["implementationId"] == SourceImplementation.IIGR.value
    ]
    full_primary = primary[
        primary["estimand"] == "RETROSPECTIVE_CURRENT_GENERATION"
    ].sort_values("candidateId")
    prefix_primary = primary[
        (primary["estimand"] == "CURRENT_HISTORICAL")
        & (primary["temporalModeId"].str.endswith("_PREFIX_ENDPOINT"))
    ].sort_values("candidateId")
    lines = [
        "# S12G Full Results: Frozen Time-Base Ensemble",
        "",
        "## Top summary",
        "",
        f"- **Research step ID:** `{VERSION}` (S12G)",
        "- **Completion status:** Complete at the mandatory post-S12G human-review boundary; S13 was not begun.",
        f"- **Artifacts written:** {validation['artifactCount']} status-bearing files under `/artifacts/research_steps/S12G/`, including 96-input manifests, labels, full and prefix source outputs, partition and replay evidence, candidate/ensemble analyses, six figures, validation, hashes, and this report.",
        f"- **Validation result:** {validation['summary']}",
        f"- **Outcome classification:** `{outcome}` ({validation['outcomeClass']}).",
        "- **Caveats or blockers:** This is source-informed reconstruction, not author code. The three upstream candidates remain nonidentifiable; completed-fit values are retrospective, and C0 endpoints represent the final pre-fission update because C0 excludes daughter records. Prior negative and failed evidence remains unchanged.",
        f"- **Recommended next action:** {recommendation}",
        "",
        "## Lay summary",
        "",
        (
            "We ran the same replicator-label and public-source causal-emergence analysis on all "
            "three GARD time bases that independently matched the paper's upstream timing evidence. "
            "Each candidate was kept separate and equally carried; a positive ensemble conclusion "
            "was allowed only if all three passed the same predeclared tests. Full-trajectory values "
            "can use future observations and are descriptive, while prefix values were refitted only "
            "from the past and were checked against three future-suffix perturbations."
        ),
        "",
        "## Frozen question and scope",
        "",
        "S12G asks whether the paper-directed emergence/self-replication relationship is robust to all three S12FR-confirmed time-base candidates. It uses exactly 96 existing trajectories (32 per candidate), generates no GARD trajectory, and performs no prediction, MLP, intervention, candidate discovery, estimator repair, or S13 work. S12F remains `SIMULATOR_IDENTIFICATION_FAILED`; S12FR remains `NONIDENTIFIABLE_TIMEBASE_ENSEMBLE`.",
        "",
        "## Inputs and provenance",
        "",
        f"The input lock contains {len(input_manifest)} trajectories and {input_manifest['matrixIndex'].nunique()} shared matrix/initial identities. Every cache hash and all S12FR replay flags passed. Candidate identities were the exact locked h/daughter/overshoot/clock tuples. Pinned source commits were IIGR `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7` and PhiRL `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`; scientific execution used only the audited safe JSON lattice.",
        "",
        "## Methods",
        "",
        "Counts were transformed with additive 0.5 closure, full 100-component CLR, then removal of original component 100. Historical H>0.9 non-drift labels were primary; the frozen past-only cosine threshold graph was secondary. IIGR source-defined emergence (synergy plus the two downward-causation atoms) was primary, PhiRL was regularization robustness, and corrected local Phi-r was comparator-only.",
        "",
        "C0 sequences contained the initial state and Poisson batch-update states only. C1 additionally contained selected post-fission daughters. Prefix fits began only at endpoints with at least 256 prior locked-clock transitions, used the exact prefix through the endpoint, and retained only the final local value. Every fit was replayed. Every prefix had byte-exact structural suffix tests; first/middle/last eligible endpoints per trajectory and implementation were recomputed after suffix deletion, deterministic shuffle, and domain-separated replacement.",
        "",
        "Trajectory-level Spearman correlations, 4,096-resample trajectory bootstraps, 4,096 within-trajectory circular-shift nulls, replicator-minus-drift differences, block-aware shift nulls, temporal dependence, spikes, metric identity, and full-versus-prefix differences followed the preregistration. Counts inherited from the 24-unit S12D gates were scaled by their frozen proportions to 32 units. No S12FR weight entered a scientific result.",
        "",
        "## Primary candidate results",
        "",
        "| Candidate | Full median rho | Full positive/defined | Full 95% bootstrap | Full p_shift+ | Full gate | Prefix median rho | Prefix positive/defined | Prefix 95% bootstrap | Prefix p_shift+ | Prefix gate |",
        "| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for candidate_id in CANDIDATE_IDS:
        full_row = full_primary[full_primary["candidateId"] == candidate_id].iloc[0]
        prefix_row = prefix_primary[prefix_primary["candidateId"] == candidate_id].iloc[0]
        lines.append(
            "| {candidate} | {fm:.6g} | {fp}/{fd} | [{fl:.6g}, {fu:.6g}] | {fnull:.6g} | {fgate} | {pm:.6g} | {pp}/{pd} | [{pl:.6g}, {pu:.6g}] | {pnull:.6g} | {pgate} |".format(
                candidate=candidate_id,
                fm=float(full_row["medianCorrelation"]),
                fp=int(full_row["positiveTrajectoryCount"]),
                fd=int(full_row["definedTrajectoryCount"]),
                fl=float(full_row["bootstrapLower95"]),
                fu=float(full_row["bootstrapUpper95"]),
                fnull=float(full_row["circularShiftPositiveP"]),
                fgate="PASS" if bool(full_row["gatePassed"]) else "FAIL",
                pm=float(prefix_row["medianCorrelation"]),
                pp=int(prefix_row["positiveTrajectoryCount"]),
                pd=int(prefix_row["definedTrajectoryCount"]),
                pl=float(prefix_row["bootstrapLower95"]),
                pu=float(prefix_row["bootstrapUpper95"]),
                pnull=float(prefix_row["circularShiftPositiveP"]),
                pgate="PASS" if bool(prefix_row["gatePassed"]) else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "The complete association table preserves current historical, next historical, and past-only cosine estimands for IIGR and PhiRL. Replicator-minus-drift tests are in `replicator_drift_results.csv`; candidate decisions are in `ensemble_adjudication.csv`.",
            "",
            "## Metric identity and future dependence",
            "",
            f"Across {len(metric_identity)} trajectory/implementation/mode comparisons, source-defined emergence and corrected local Phi-r were retained as distinct scalars. Across {len(future)} full-prefix trajectory/implementation comparisons, all endpoint differences, rank changes, spike overlaps, replication-association changes, and partition ARIs are preserved. Full-fit values remain `RETROSPECTIVE_FULL_TRAJECTORY_LOCAL` and cannot support early warning or control.",
            "",
            "## Ensemble adjudication",
            "",
            adjudication.to_markdown(index=False),
            "",
            f"The frozen all-three result is `{outcome}`. Candidate weights were neither used nor updated.",
            "",
            "## Validation",
            "",
            validation["details"],
            f"The failure ledger contains {len(failures)} aggregated rows; expected pre-256 ineligibility and any nonfinite source values are retained rather than omitted or imputed.",
            "",
            "## Commands and environment",
            "",
            "```bash",
            "PYTHONPATH=src pytest -q tests/e01/test_s12g_frozen_timebase_ensemble.py",
            "ruff check src/e01_frozen_timebase_ensemble scripts/e01/freeze_s12g_preregistration.py scripts/e01/run_s12g_frozen_timebase_ensemble.py tests/e01/test_s12g_frozen_timebase_ensemble.py",
            "ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s12g_preregistration.py --design-commit 0118892d035eef932274b0f44bd1ecc024268fa2",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s12g_frozen_timebase_ensemble.py --workers 6",
            "```",
            "",
            f"CPU float64 was authoritative; GPU use was zero. Runtime was {runtime['wallHours']:.4f} wall-hours and {runtime['workerCpuHours']:.4f} summed worker CPU-hours with six source workers after a three-trajectory benchmark. Python `{runtime['python']}`, NumPy `{runtime['numpy']}`, SciPy `{runtime['scipy']}`, platform `{runtime['platform']}`.",
            "",
            "## Caveats and limitations",
            "",
            "- Public IIGR/PhiRL code is source evidence, not the unavailable author GARD implementation or a paper-primary identity.",
            "- The 96 trajectories were untouched by labels/emergence before S12G, but they had already served upstream time-base confirmation.",
            "- Candidate matrices and initial states are shared, enabling paired diagnostics, but dynamics diverge under different daughter/overshoot/clock semantics.",
            "- Historical labels are retrospective; completed-fit source values use future observations. Prefix results are the only prospective reconstruction here.",
            "- C0 maps a fission decision to its final pre-fission update; it does not contain a daughter-state record.",
            "- Exact replay is bounded to the frozen runtime, source wrappers, CPU float64 policy, and platform identities.",
            "- S12G does not erase S11/S11R failures, S12 negative strict results, S12C non-support, S12D fail-closed evidence, S12E mismatch, S12F failure, or S12FR nonidentifiability.",
            "",
            "## Recommended next action",
            "",
            recommendation,
            "",
            "S13 remains `BLOCKED_PENDING_S12G_HUMAN_REVIEW`. Stop here.",
        ]
    )
    return "\n".join(lines) + "\n"


def placeholder_figure(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def emergency_handoff(exc: BaseException) -> None:
    """Produce a complete fail-closed handoff without repairing the failure."""

    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    contract = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["tables"]
    for filename, columns in contract.items():
        path = STEP_ROOT / filename
        if path.exists():
            continue
        frame = pd.DataFrame(columns=columns)
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False, compression="zstd")
        else:
            frame.to_csv(path, index=False, lineterminator="\n")
    for filename in (
        "label_fingerprints.png",
        "full_emergence_trajectories.png",
        "association_distributions.png",
        "full_prefix_comparison.png",
        "metric_identity_comparison.png",
        "ensemble_decision_matrix.png",
    ):
        path = FIGURE_ROOT / filename
        if not path.exists():
            placeholder_figure(
                path,
                f"S12G stopped fail-closed: {type(exc).__name__}: {exc}",
            )
    failure = {
        "failureId": "S12G-TERMINAL-FAIL-CLOSED",
        "stage": "execution",
        "candidateId": None,
        "trajectoryId": None,
        "implementationId": None,
        "temporalModeId": None,
        "endpointGeneration": None,
        "severity": "FATAL",
        "status": "S12G_VALIDATION_FAILED_CLOSED",
        "reason": f"{type(exc).__name__}:{exc}",
        "gateImpact": "FAIL_CLOSED_NO_REPAIR",
        "repairAttempted": False,
    }
    pd.DataFrame([failure]).to_csv(
        STEP_ROOT / "failure_ledger.csv", index=False, lineterminator="\n"
    )
    classification = {
        "schema": "eidosoma.e01.s12g_classification.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "classification": "S12G_VALIDATION_FAILED_CLOSED",
        "reason": failure["reason"],
        "s13Status": "BLOCKED_PENDING_S12G_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "classification.json", classification)
    for filename, payload in (
        (
            "runtime_benchmark.json",
            {"schema": "eidosoma.e01.s12g_runtime_benchmark.v1", "passed": False, "reason": failure["reason"]},
        ),
        (
            "runtime_manifest.json",
            {"schema": "eidosoma.e01.s12g_runtime_manifest.v1", "passed": False, "reason": failure["reason"]},
        ),
        (
            "regeneration_validation.json",
            {"schema": "eidosoma.e01.s12g_regeneration_validation.v1", "passed": False, "reason": failure["reason"]},
        ),
        (
            "immutable_prior_validation.json",
            {"schema": "eidosoma.e01.s12g_immutable_prior_validation.v1", "passed": False, "reason": "not_reached_or_terminal_failure"},
        ),
    ):
        if not (STEP_ROOT / filename).exists():
            write_json(STEP_ROOT / filename, payload)
    status = {
        "researchStepId": "S12G",
        "stepNumber": "S12G",
        "success": False,
        "status": "FAILED_CLOSED",
        "artifactsWritten": [],
        "validationResult": "FAIL_CLOSED",
        "caveatsOrBlockers": [failure["reason"]],
        "recommendedNextAction": "Keep S13 blocked and return for human review; no S12G repair is authorized.",
        "outcomeClassification": "S12G_VALIDATION_FAILED_CLOSED",
    }
    write_json(STEP_ROOT / "status.json", status)
    report = f"""# S12G Full Results: Frozen Time-Base Ensemble

## Top summary

- **Research step ID:** `{VERSION}` (S12G)
- **Completion status:** Stopped fail-closed before a valid scientific adjudication.
- **Artifacts written:** Status-bearing partial outputs, schemas, failure evidence, validation state, and this report under `/artifacts/research_steps/S12G/`.
- **Validation result:** `FAIL_CLOSED`: `{type(exc).__name__}: {exc}`
- **Outcome classification:** `S12G_VALIDATION_FAILED_CLOSED` (constraining/contradictory operational result).
- **Caveats or blockers:** The frozen stop rule fired; no repair, imputation, candidate selection, gate weakening, S13, prediction, or intervention was attempted.
- **Recommended next action:** Keep S13 blocked and return for human review; no S12G repair is authorized.

## Lay summary

The bounded three-candidate audit could not safely reach a scientific conclusion because a preregistered operational validation failed. The failure is preserved rather than repaired or hidden.

## Methods and inputs

The intended method and all frozen inputs are recorded in `preregistration.yaml`, `method_lock.json`, the registries, and `trajectory_input_manifest.parquet`. Exactly zero new GARD trajectories were authorized.

## Result and validation

Terminal error: `{type(exc).__name__}: {exc}`. Machine-readable failure detail is in `failure_ledger.csv`. Any outputs created before the failure remain status-bearing and are not promoted to scientific evidence.

## Caveats and provenance

S01–S12FR remain immutable. S12F remains `SIMULATOR_IDENTIFICATION_FAILED`; S12FR remains `NONIDENTIFIABLE_TIMEBASE_ENSEMBLE`. S13 remains `BLOCKED_PENDING_S12G_HUMAN_REVIEW`.
"""
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    required = yaml.safe_load((STEP_ROOT / "preregistration.yaml").read_text())["artifacts"]["required"]
    manifest = artifact_manifest(required)
    status["artifactsWritten"] = [
        item["relativePath"] for item in manifest["artifacts"]
    ] + ["artifact_manifest.json"]
    write_json(STEP_ROOT / "status.json", status)
    artifact_manifest(required)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers != 6:
        raise RuntimeError("S12G freezes exactly six source-analysis workers")
    started = time.perf_counter()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    method_lock = json.loads((STEP_ROOT / "method_lock.json").read_text())
    if not method_lock.get("passed"):
        raise RuntimeError("pre-outcome S12G method lock did not pass")
    head = git_value("rev-parse", "HEAD")
    remote = git_value("rev-parse", "origin/eidosoma/groups/42")
    if head != remote or git_value("status", "--short"):
        raise RuntimeError("S12G implementation must be committed, pushed, and clean")
    implementation_lock = {
        "schema": "eidosoma.e01.s12g_implementation_lock.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "designCommit": method_lock["designCommit"],
        "implementationCommit": head,
        "remoteCommit": remote,
        "files": [
            {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256_file(path),
            }
            for path in (
                Path(__file__),
                REPO / "src/e01_frozen_timebase_ensemble/core.py",
                REPO / "src/e01_pigozzi_source_equivalence_confirmation/core.py",
                REPO / "src/e01_source_emergence_metric_identity/core.py",
                CONFIG_PATH,
                SCHEMA_PATH,
            )
        ],
        "labelOutcomeOpenedBeforeLock": False,
        "informationTheoryOutcomeOpenedBeforeLock": False,
        "passed": True,
    }
    write_json(STEP_ROOT / "implementation_lock.json", implementation_lock)

    tasks = pd.read_parquet(INPUT_MANIFEST).sort_values(
        ["candidateId", "matrixIndex"]
    )
    if len(tasks) != 96 or set(tasks["candidateId"]) != set(CANDIDATE_IDS):
        raise RuntimeError("locked S12G task manifest is not exactly 96 rows")
    task_records = tasks.to_dict("records")
    RESULT_CACHE.mkdir(parents=True, exist_ok=True)
    benchmark_tasks = [task for task in task_records if int(task["matrixIndex"]) == 0]
    if len(benchmark_tasks) != 3:
        raise RuntimeError("benchmark triplet is not exactly three matrix-0 inputs")
    benchmark_records = execute_tasks(benchmark_tasks, workers=3)
    mean_cpu = float(np.mean([item["cpuSeconds"] for item in benchmark_records]))
    mean_wall = float(np.mean([item["wallSeconds"] for item in benchmark_records]))
    projected_cpu_hours = mean_cpu * 96 / 3600.0
    projected_wall_hours = mean_wall * 96 / args.workers / 3600.0
    benchmark = {
        "schema": "eidosoma.e01.s12g_runtime_benchmark.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "benchmarkTrajectoryCount": 3,
        "candidateIds": list(CANDIDATE_IDS),
        "matrixIndex": 0,
        "records": benchmark_records,
        "projectedCpuHours": projected_cpu_hours,
        "projectedWallHours": projected_wall_hours,
        "cpuHourCeiling": 250,
        "wallHourCeiling": 72,
        "passed": projected_cpu_hours <= 250 and projected_wall_hours <= 72,
    }
    write_json(STEP_ROOT / "runtime_benchmark.json", benchmark)
    if not benchmark["passed"]:
        raise RuntimeError(
            f"runtime projection exceeded hard ceiling: CPU={projected_cpu_hours}, wall={projected_wall_hours}"
        )
    remaining = [task for task in task_records if int(task["matrixIndex"]) != 0]
    print(json.dumps({"stage": "benchmark_passed", **benchmark}, sort_keys=True), flush=True)
    remaining_records = execute_tasks(remaining, workers=args.workers)
    records = sorted(
        benchmark_records + remaining_records,
        key=lambda row: (row["candidateId"], row["matrixIndex"]),
    )
    if len(records) != 96:
        raise RuntimeError("source execution did not return exactly 96 task records")
    frames = collate(records)
    labels = frames["label_values.parquet"]
    full = frames["full_source_values.parquet"]
    prefix = frames["prefix_endpoint_values.parquet"]
    partitions = frames["partition_history.parquet"]
    diagnostics = frames["source_diagnostic_outputs.parquet"]
    suffix = frames["replay_suffix_validation.parquet"]
    seeds = frames["seed_manifest.parquet"]
    worker_failures = frames["worker_failures"]

    full_replay = bool(full["exactReplayPassed"].astype(bool).all())
    eligible_prefix = prefix[prefix["priorLockedClockTransitions"] >= 256]
    prefix_replay = bool(eligible_prefix["exactReplayPassed"].astype(bool).all())
    suffix_structural = bool(suffix["structuralExact"].astype(bool).all())
    sentinel = suffix[suffix["sentinel"] != "non_sentinel"]
    suffix_executed = bool(
        len(sentinel) == 96 * 2 * 3 * 3
        and sentinel["resultExact"].fillna(False).astype(bool).all()
    )
    full_coverage = (
        full.assign(numeric=np.isfinite(pd.to_numeric(full["emergence"], errors="coerce")))
        .groupby(["candidateId", "implementationId"])["numeric"]
        .mean()
    )
    prefix_coverage = (
        eligible_prefix.assign(
            numeric=np.isfinite(pd.to_numeric(eligible_prefix["emergence"], errors="coerce"))
        )
        .groupby(["candidateId", "implementationId"])["numeric"]
        .mean()
    )
    operational = bool(
        full_replay
        and prefix_replay
        and suffix_structural
        and suffix_executed
        and not len(worker_failures)
        and float(full_coverage.min()) >= 0.80
        and float(prefix_coverage.min()) >= 0.80
        and len(labels) == 96 * 100 * 2
        and len(prefix) == 96 * 100 * 2
        and len(seeds["streamId"].unique()) == len(seeds)
        and diagnostics["componentIdentityMaxAbsError"].fillna(0).max() <= 1e-12
    )
    if not operational:
        raise RuntimeError(
            "source operational gate failed: "
            f"fullReplay={full_replay},prefixReplay={prefix_replay},"
            f"suffixStructural={suffix_structural},suffixExecuted={suffix_executed},"
            f"workerFailures={len(worker_failures)},fullCoverage={float(full_coverage.min())},"
            f"prefixCoverage={float(prefix_coverage.min())}"
        )

    (
        associations,
        association_details,
        drift,
        drift_details,
        summaries,
        _differences,
    ) = run_candidate_statistics(full, prefix)
    temporal, spike = run_temporal_statistics(full)
    metric_identity = run_metric_identity(full, prefix)
    future = run_future_dependence(full, prefix, partitions)
    shared_audit = json.loads(
        (STEP_ROOT / "shared_identity_audit.json").read_text(encoding="utf-8")
    )
    cross_candidate = run_cross_candidate(
        labels, association_details, drift_details, partitions, shared_audit
    )
    adjudication, classification = adjudicate(
        associations, drift, temporal, spike, summaries, full, prefix
    )
    write_csv(STEP_ROOT / "candidate_associations.csv", associations.to_dict("records"))
    write_parquet(STEP_ROOT / "candidate_association_details.parquet", association_details)
    write_csv(STEP_ROOT / "replicator_drift_results.csv", drift.to_dict("records"))
    write_parquet(STEP_ROOT / "replicator_drift_details.parquet", drift_details)
    write_csv(STEP_ROOT / "temporal_dependence_results.csv", temporal.to_dict("records"))
    write_csv(STEP_ROOT / "spike_results.csv", spike.to_dict("records"))
    write_csv(STEP_ROOT / "metric_identity_results.csv", metric_identity.to_dict("records"))
    write_csv(STEP_ROOT / "future_dependence_results.csv", future.to_dict("records"))
    write_csv(STEP_ROOT / "cross_candidate_results.csv", cross_candidate.to_dict("records"))
    write_csv(STEP_ROOT / "ensemble_adjudication.csv", adjudication.to_dict("records"))
    write_json(STEP_ROOT / "classification.json", classification)
    failures = pd.DataFrame(
        failure_rows_from_statuses(full, prefix, worker_failures),
        columns=json.loads(SCHEMA_PATH.read_text())["tables"]["failure_ledger.csv"],
    )
    failures.to_csv(STEP_ROOT / "failure_ledger.csv", index=False, lineterminator="\n")
    make_figures(
        labels,
        full,
        prefix,
        association_details,
        metric_identity,
        adjudication,
    )

    runtime = {
        "schema": "eidosoma.e01.s12g_runtime_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "startedAtUtcApprox": datetime.now(timezone.utc).isoformat(),
        "wallHours": (time.perf_counter() - started) / 3600.0,
        "workerCpuHours": sum(float(item["cpuSeconds"]) for item in records) / 3600.0,
        "workerWallHours": sum(float(item["wallSeconds"]) for item in records) / 3600.0,
        "workers": args.workers,
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
        "cpuPrecision": "float64_authoritative",
        "gpuHours": 0.0,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "taskRecords": records,
        "hardCeilings": config["runtime"]["hardCeilings"],
        "passed": True,
    }
    write_json(STEP_ROOT / "runtime_manifest.json", runtime)
    regeneration = {
        "schema": "eidosoma.e01.s12g_regeneration_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "newGardTrajectoriesGenerated": 0,
        "s12frTrajectoryReplayEvidenceRows": 96,
        "s12frTrajectoryReplayEvidenceAllPassed": bool(
            pd.read_parquet(INPUT_MANIFEST)["cacheHashPassed"].astype(bool).all()
        ),
        "sourceFullReplayAllPassed": full_replay,
        "sourcePrefixReplayAllPassed": prefix_replay,
        "suffixStructuralAllPassed": suffix_structural,
        "suffixExecutedSentinelsAllPassed": suffix_executed,
        "passed": full_replay and prefix_replay and suffix_structural and suffix_executed,
    }
    write_json(STEP_ROOT / "regeneration_validation.json", regeneration)
    immutable = validate_immutable_prior()
    access = json.loads((STEP_ROOT / "scope_access_ledger.json").read_text())
    access["events"].append(
        {
            "stage": "COMPLETE_S12G_EXECUTION",
            "labelOutcomeOpened": True,
            "informationTheoryOutcomeOpened": True,
            "newGardTrajectoryGenerated": False,
            "candidateSelectionOrReweighting": False,
            "predictionOrInterventionAccess": False,
            "s13Access": False,
            "status": "PASS",
        }
    )
    access["success"] = True
    write_json(STEP_ROOT / "scope_access_ledger.json", access)
    schema_validation = validate_schemas()
    if not immutable["passed"] or not schema_validation["passed"]:
        raise RuntimeError(
            f"handoff validation failed: immutable={immutable['passed']},schemas={schema_validation['passed']}"
        )
    outcome = classification["classification"]
    outcome_class = (
        "supportive"
        if outcome
        in {
            "ENSEMBLE_PROSPECTIVE_SOURCE_EMERGENCE_SUPPORT",
            "ENSEMBLE_RETROSPECTIVE_SOURCE_EMERGENCE_SUPPORT",
        }
        else "constraining/contradictory"
    )
    validation = {
        "summary": (
            f"PASS: 96/96 input hashes and S12FR replay evidence, every full and eligible "
            f"prefix source replay, {len(suffix)}/{len(suffix)} structural suffix checks, "
            f"{len(sentinel)}/{len(sentinel)} executed suffix sentinels, source coverage "
            f">=80%, schemas, provenance, scope, runtime, storage, and prior immutability passed."
        ),
        "details": (
            f"Full minimum finite coverage was {float(full_coverage.min()):.6f}; eligible-prefix "
            f"minimum finite coverage was {float(prefix_coverage.min()):.6f}. There were "
            f"{len(full):,} full local rows, {len(prefix):,} prefix status rows, "
            f"{len(partitions):,} partition fits, {len(seeds):,} unique source seeds, and zero "
            "new GARD trajectories. All 32 matrix/initial identities were shared across candidates."
        ),
        "outcomeClass": outcome_class,
        "artifactCount": 0,
    }
    report = build_report(
        classification=classification,
        associations=associations,
        drift=drift,
        future=future,
        metric_identity=metric_identity,
        adjudication=adjudication,
        runtime=runtime,
        validation=validation,
        failures=failures,
        input_manifest=tasks,
    )
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    required = config["artifacts"]["required"]
    status = {
        "researchStepId": "S12G",
        "stepNumber": "S12G",
        "success": True,
        "status": "COMPLETED_AT_MANDATORY_HUMAN_REVIEW_BOUNDARY",
        "artifactsWritten": [],
        "validationResult": validation["summary"],
        "caveatsOrBlockers": [
            "Three time-base candidates remain nonidentifiable and equally carried.",
            "Completed-fit values are retrospective; C0 endpoint values are pre-fission boundary-aligned.",
            "Source-informed reconstruction is not author-code or paper-primary identity.",
            "S13 remains blocked regardless of outcome.",
        ],
        "recommendedNextAction": recommendation_for(outcome),
        "outcomeClassification": outcome,
        "outcomeClass": outcome_class,
        "s13Status": "BLOCKED_PENDING_S12G_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "status.json", status)
    manifest = artifact_manifest(required)
    if not manifest["passed"]:
        raise RuntimeError(f"artifact completeness failed: {manifest['requiredMissing']}")
    validation["artifactCount"] = manifest["artifactCountExcludingSelf"] + 1
    report = build_report(
        classification=classification,
        associations=associations,
        drift=drift,
        future=future,
        metric_identity=metric_identity,
        adjudication=adjudication,
        runtime=runtime,
        validation=validation,
        failures=failures,
        input_manifest=tasks,
    )
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status["artifactsWritten"] = [
        item["relativePath"] for item in manifest["artifacts"]
    ] + ["artifact_manifest.json"]
    write_json(STEP_ROOT / "status.json", status)
    manifest = artifact_manifest(required)
    if not manifest["passed"]:
        raise RuntimeError("final artifact manifest validation failed")
    print(
        json.dumps(
            {
                "stage": "S12G_complete",
                "classification": outcome,
                "validation": validation["summary"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:
        emergency_handoff(error)
        print(
            json.dumps(
                {
                    "stage": "S12G_failed_closed",
                    "error": f"{type(error).__name__}:{error}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from error
