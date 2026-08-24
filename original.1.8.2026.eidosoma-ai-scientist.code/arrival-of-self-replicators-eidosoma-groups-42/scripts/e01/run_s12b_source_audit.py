#!/usr/bin/env python3
"""Execute the frozen E01-S12B Pigozzi source-code audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml
from sklearn.metrics import adjusted_rand_score

from e01_pigozzi_source_audit.analysis import (
    association_summary,
    finite_spearman,
    ljung_box_summary,
    molecular_progress_trend,
    percentile_ranks,
    prospective_candidate,
    retrospective_coherent,
    significant_opposite,
    spike_thresholds,
)
from e01_pigozzi_source_audit.core import (
    AuditResult,
    SourceImplementation,
    derive_seed,
    run_source_pipeline,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/e01/s12b_pigozzi_source_audit_preregistration.yaml"
STEP_ROOT = Path("/artifacts/research_steps/S12B")
S12_ROOT = Path("/artifacts/research_steps/S12")
CACHE_ROOT = Path("/cache/e01_s12b")
INPUT_CACHE = CACHE_ROOT / "trajectory_inputs"
RESULT_CACHE = CACHE_ROOT / "trajectory_results"
SAFE_LATTICE = STEP_ROOT / "safe_phi_lattice.json"
FIGURE_ROOT = STEP_ROOT / "figures"
VERSION = "E01-S12B-PIGOZZI-SOURCE-CODE-AUDIT-v1.0.0"
EVIDENCE_CLASS = "SOURCE_INFORMED_FORENSIC_RECONSTRUCTION"
HISTORICAL_CONFIG = "E01-S08-YH-T1-HGT090-v1.0.0"


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def result_replay_equal(left: AuditResult, right: AuditResult) -> bool:
    if (
        left.implementation != right.implementation
        or left.status != right.status
        or left.reason != right.reason
        or left.retained_variables != right.retained_variables
        or left.partition_1 != right.partition_1
        or left.partition_2 != right.partition_2
        or left.local_offset != right.local_offset
    ):
        return False
    for name in ("mi_matrix", "fiedler_vector", "partition_average", "local_phi_r", "emergence"):
        a, b = getattr(left, name), getattr(right, name)
        if (a is None) != (b is None):
            return False
        if a is not None and not np.array_equal(a, b, equal_nan=True):
            return False
    return True


def result_endpoint(result: AuditResult) -> tuple[str, str | None, float | None, float | None]:
    if result.local_phi_r is None or result.local_phi_r.size == 0:
        return result.status, result.reason, None, None
    phi = float(result.local_phi_r[-1])
    diagnostic = float(result.emergence[-1]) if result.emergence is not None and result.emergence.size else None
    if not np.isfinite(phi) or diagnostic is None or not np.isfinite(diagnostic):
        return "INELIGIBLE_NONFINITE_LOCAL_VALUE", "endpoint_local_phi_r_or_diagnostic_nonfinite", None, None
    if result.status not in {"ELIGIBLE", "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES"}:
        return result.status, result.reason, None, None
    return "ELIGIBLE", None, phi, diagnostic


def partition_json(values: tuple[int, ...]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def partition_hash(values: tuple[int, ...]) -> str:
    return hashlib.sha256(partition_json(values).encode("utf-8")).hexdigest()


def partition_row(
    result: AuditResult,
    *,
    trajectory_id: str,
    matrix_index: int,
    mode_id: str,
    fit_kind: str,
    endpoint_index: int,
    endpoint_generation: int | None,
    fit_count: int,
    input_hash: str,
    preprocessing_seed: int,
    partition_seed: int,
    replay_passed: bool,
) -> dict[str, Any]:
    return {
        "researchStepId": "S12B",
        "preregistrationVersion": VERSION,
        "implementationId": result.implementation,
        "temporalModeId": mode_id,
        "fitKind": fit_kind,
        "trajectoryId": trajectory_id,
        "matrixIndex": matrix_index,
        "endpointObservationIndex": endpoint_index,
        "endpointGeneration": endpoint_generation,
        "fitObservationCount": fit_count,
        "inputSha256": input_hash,
        "status": result.status,
        "reason": result.reason,
        "retainedVariableCount": len(result.retained_variables),
        "retainedVariablesJson": partition_json(result.retained_variables),
        "partition1Count": len(result.partition_1),
        "partition2Count": len(result.partition_2),
        "partition1Json": partition_json(result.partition_1),
        "partition2Json": partition_json(result.partition_2),
        "partition1Sha256": partition_hash(result.partition_1),
        "partition2Sha256": partition_hash(result.partition_2),
        "miMatrixSha256": sha256_array(result.mi_matrix) if result.mi_matrix is not None else None,
        "fiedlerVectorJson": json.dumps(result.fiedler_vector.tolist(), separators=(",", ":")) if result.fiedler_vector is not None else None,
        "fiedlerVectorSha256": sha256_array(result.fiedler_vector) if result.fiedler_vector is not None else None,
        "preprocessingSeed": preprocessing_seed,
        "partitionSeed": partition_seed,
        "exactReplayPassed": replay_passed,
        "sourceRelationship": "SOURCE_INFORMED_RECONSTRUCTION",
    }


def prepare_trajectory_inputs() -> list[dict[str, Any]]:
    INPUT_CACHE.mkdir(parents=True, exist_ok=True)
    observations = pd.read_parquet(S12_ROOT / "baseline_observations.parquet")
    records: list[dict[str, Any]] = []
    for trajectory_id, group in observations.groupby("trajectoryId", sort=True):
        group = group.sort_values("observationIndex").reset_index(drop=True)
        expected = np.arange(len(group), dtype=np.int64)
        if not np.array_equal(group["observationIndex"].to_numpy(dtype=np.int64), expected):
            raise RuntimeError(f"noncontiguous observation indices for {trajectory_id}")
        counts = np.vstack(group["state"].map(np.asarray).to_list()).astype(np.int64)
        if counts.shape[1] != 100 or np.any(counts < 0) or not np.array_equal(counts.sum(axis=1), group["mass"].to_numpy(dtype=np.int64)):
            raise RuntimeError(f"invalid integer states for {trajectory_id}")
        closed = (counts.astype(np.float64) + 0.5) / (counts.sum(axis=1, keepdims=True) + 50.0)
        log_closed = np.log(closed)
        clr = log_closed - log_closed.mean(axis=1, keepdims=True)
        clr = clr[:, :99]
        if clr.shape != (len(group), 99) or not np.all(np.isfinite(clr)):
            raise RuntimeError(f"invalid frozen CLR substrate for {trajectory_id}")
        path = INPUT_CACHE / f"{trajectory_id}.npz"
        np.savez_compressed(
            path,
            clr=clr,
            observation_index=group["observationIndex"].to_numpy(dtype=np.int64),
            observation_kind=group["observationKind"].astype(str).to_numpy(dtype="U32"),
            generation=group["generation"].to_numpy(dtype=np.int64),
            molecular_step=group["molecularStep"].to_numpy(dtype=np.int64),
            matrix_index=np.array(int(group["matrixIndex"].iloc[0]), dtype=np.int64),
            trajectory_id=np.array(str(trajectory_id)),
        )
        records.append({"trajectoryId": str(trajectory_id), "matrixIndex": int(group["matrixIndex"].iloc[0]), "path": str(path), "observationCount": len(group), "molecularEventCount": int((group["observationKind"] == "molecular_event").sum()), "postFissionCount": int((group["observationKind"] == "post_fission").sum()), "clrSha256": sha256_array(clr)})
    if len(records) != 12 or sorted(item["matrixIndex"] for item in records) != list(range(12)):
        raise RuntimeError("S12B requires exactly matrices 0 through 11")
    return sorted(records, key=lambda item: item["matrixIndex"])


def synthetic_fixture(fixture_id: str, root_seed: str) -> np.ndarray:
    rng = np.random.RandomState(derive_seed(root_seed, "synthetic_fixture", fixture_id))
    if fixture_id == "COUPLED_GAUSSIAN_A":
        base = rng.normal(size=(320, 8))
        base[:, 4:] += 0.35 * base[:, :4]
        return base
    if fixture_id == "COUPLED_GAUSSIAN_B":
        innovations = rng.normal(size=(320, 8))
        result = np.zeros_like(innovations)
        for index in range(1, len(result)):
            result[index] = 0.55 * result[index - 1] + innovations[index]
            result[index, 4:] += 0.25 * result[index - 1, :4]
        return result
    if fixture_id == "CONSTANT_INPUT":
        return np.ones((320, 8), dtype=np.float64)
    if fixture_id == "SINGULAR_DUPLICATE_INPUT":
        base = rng.normal(size=(320, 4))
        return np.column_stack([base, base])
    raise ValueError(f"unknown fixture {fixture_id}")


def npz_equal(path_a: Path, path_b: Path) -> bool:
    with np.load(path_a, allow_pickle=False) as a, np.load(path_b, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            return False
        for name in a.files:
            left, right = a[name], b[name]
            if left.dtype.kind in "fc":
                equal = np.array_equal(left, right, equal_nan=True)
            else:
                equal = np.array_equal(left, right)
            if not equal:
                return False
        return True


def max_abs_difference(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    if a.size == 0:
        return 0.0
    with np.errstate(invalid="ignore"):
        difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if np.all(np.isnan(difference)):
        return None
    return float(np.nanmax(difference))


def source_equivalence(config: dict[str, Any]) -> tuple[pd.DataFrame, bool]:
    root_seed = config["randomness"]["rootSeedHex"]
    eq_root = CACHE_ROOT / "source_equivalence"
    eq_root.mkdir(parents=True, exist_ok=True)
    adapter = REPO / "scripts/e01/s12b_original_source_adapter.py"
    source_dirs = {
        SourceImplementation.IIGR: Path(config["sourceSnapshots"][SourceImplementation.IIGR.value]["localCheckout"]),
        SourceImplementation.PHIRL: Path(config["sourceSnapshots"][SourceImplementation.PHIRL.value]["localCheckout"]),
    }
    rows: list[dict[str, Any]] = []
    env = os.environ.copy()
    env.update({name: value for name, value in config["runtimeAndStorage"]["threadEnvironment"].items()})
    env["PYTHONHASHSEED"] = "0"
    for fixture_id in config["sourceEquivalence"]["fixtureIds"]:
        observations = synthetic_fixture(fixture_id, root_seed)
        input_path = eq_root / f"{fixture_id}.npz"
        np.savez_compressed(input_path, observations=observations)
        for implementation in SourceImplementation:
            preprocessing_seed = derive_seed(root_seed, implementation.value, fixture_id, "preprocessing_noise")
            partition_seed = derive_seed(root_seed, implementation.value, fixture_id, "fiedler_initialization")
            original_paths = [eq_root / f"{fixture_id}-{implementation.value}-original-{index}.npz" for index in (1, 2)]
            adapter_name = "IIGR" if implementation is SourceImplementation.IIGR else "PHIRL"
            for output in original_paths:
                subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        str(adapter),
                        "--implementation",
                        adapter_name,
                        "--source-dir",
                        str(source_dirs[implementation]),
                        "--input",
                        str(input_path),
                        "--output",
                        str(output),
                        "--preprocessing-seed",
                        str(preprocessing_seed),
                        "--partition-seed",
                        str(partition_seed),
                    ],
                    cwd=eq_root,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            original_replay = npz_equal(*original_paths)
            with np.load(original_paths[0], allow_pickle=False) as original:
                metadata = json.loads(original["metadata_json"].item())
                original_arrays = {name: original[name].copy() for name in original.files if name != "metadata_json"}
            wrapper = run_source_pipeline(observations, implementation, SAFE_LATTICE, preprocessing_seed=preprocessing_seed, partition_seed=partition_seed)
            wrapper_replay_result = run_source_pipeline(observations, implementation, SAFE_LATTICE, preprocessing_seed=preprocessing_seed, partition_seed=partition_seed)
            wrapper_replay = result_replay_equal(wrapper, wrapper_replay_result)
            retained_equal = np.array_equal(original_arrays.get("retained", np.array([], dtype=int)), np.asarray(wrapper.retained_variables, dtype=int))
            mi_difference = max_abs_difference(original_arrays.get("mi"), wrapper.mi_matrix)
            source_p1 = tuple(map(int, original_arrays.get("partition_1", np.array([], dtype=int))))
            source_p2 = tuple(map(int, original_arrays.get("partition_2", np.array([], dtype=int))))
            direct = source_p1 == wrapper.partition_1 and source_p2 == wrapper.partition_2
            exchanged = source_p1 == wrapper.partition_2 and source_p2 == wrapper.partition_1
            partition_equal = direct or exchanged
            source_average = original_arrays.get("partition_average")
            wrapper_average = wrapper.partition_average
            if exchanged and wrapper_average is not None:
                wrapper_average = wrapper_average[::-1]
            average_difference = max_abs_difference(source_average, wrapper_average)
            phi_difference = max_abs_difference(original_arrays.get("local_phi_r"), wrapper.local_phi_r)
            emergence_difference = max_abs_difference(original_arrays.get("emergence"), wrapper.emergence)
            status_equal = metadata["status"] == wrapper.status
            is_comparison = fixture_id in config["sourceEquivalence"]["comparisonFixtures"]
            numeric_gates = (
                not is_comparison
                or (
                    metadata["status"] in {"ELIGIBLE", "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES"}
                    and wrapper.status in {"ELIGIBLE", "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES"}
                    and retained_equal
                    and mi_difference is not None
                    and mi_difference <= 1e-10
                    and partition_equal
                    and average_difference is not None
                    and average_difference <= 1e-10
                    and phi_difference is not None
                    and phi_difference <= 1e-9
                )
            )
            status_gate = not (fixture_id in config["sourceEquivalence"]["statusFixtures"]) or status_equal
            passed = bool(numeric_gates and status_gate and original_replay and wrapper_replay)
            rows.append(
                {
                    "researchStepId": "S12B",
                    "fixtureId": fixture_id,
                    "implementationId": implementation.value,
                    "sourceStatus": metadata["status"],
                    "wrapperStatus": wrapper.status,
                    "statusIdentical": status_equal,
                    "retainedVariablesIdentical": retained_equal,
                    "miMaxAbsDifference": mi_difference,
                    "miGateAtMost1e10": mi_difference is not None and mi_difference <= 1e-10 if is_comparison else None,
                    "partitionIdenticalUpToSideExchange": partition_equal,
                    "partitionAverageMaxAbsDifference": average_difference,
                    "partitionAverageGateAtMost1e10": average_difference is not None and average_difference <= 1e-10 if is_comparison else None,
                    "localPhiRMaxAbsDifference": phi_difference,
                    "localPhiRGateAtMost1e9": phi_difference is not None and phi_difference <= 1e-9 if is_comparison else None,
                    "emergenceMaxAbsDifference": emergence_difference,
                    "originalExactReplay": original_replay,
                    "wrapperExactReplay": wrapper_replay,
                    "preprocessingSeed": preprocessing_seed,
                    "partitionSeed": partition_seed,
                    "passed": passed,
                }
            )
    frame = pd.DataFrame(rows)
    write_csv(STEP_ROOT / "source_equivalence_results.csv", rows)
    return frame, bool(len(frame) == 8 and frame["passed"].all())


def pipeline_seeds(root_seed: str, implementation: SourceImplementation, trajectory_id: str, mode_id: str, endpoint: int) -> tuple[int, int]:
    return (
        derive_seed(root_seed, implementation.value, trajectory_id, mode_id, endpoint, "preprocessing_noise"),
        derive_seed(root_seed, implementation.value, trajectory_id, mode_id, endpoint, "fiedler_initialization"),
    )


def process_trajectory(input_path: str, config_path: str) -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root_seed = config["randomness"]["rootSeedHex"]
    safe_lattice = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
    with np.load(input_path, allow_pickle=False) as payload:
        clr = payload["clr"].astype(np.float64, copy=False)
        observation_index = payload["observation_index"].astype(np.int64, copy=False)
        observation_kind = payload["observation_kind"].astype(str)
        generation = payload["generation"].astype(np.int64, copy=False)
        molecular_step = payload["molecular_step"].astype(np.int64, copy=False)
        matrix_index = int(payload["matrix_index"])
        trajectory_id = str(payload["trajectory_id"].item())
    trajectory_root = RESULT_CACHE / trajectory_id
    trajectory_root.mkdir(parents=True, exist_ok=True)
    full_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    suffix_all_passed = True
    full_replay_all_passed = True
    prefix_replay_all_passed = True
    timings: list[dict[str, Any]] = []
    post_fission_indices = np.flatnonzero(observation_kind == "post_fission")
    boundary_indices = [int(index) for index in post_fission_indices if molecular_step[index] >= 256]
    sentinel_indices = []
    if boundary_indices:
        sentinel_indices = sorted(
            {
                boundary_indices[0],
                boundary_indices[len(boundary_indices) // 2],
                boundary_indices[-1],
            }
        )

    for implementation in SourceImplementation:
        mode_id = "IIGR_FULL" if implementation is SourceImplementation.IIGR else "PHIRL_FULL"
        preprocessing_seed, partition_seed = pipeline_seeds(root_seed, implementation, trajectory_id, mode_id, int(observation_index[-1]))
        branch_started = time.perf_counter()
        result = run_source_pipeline(clr, implementation, safe_lattice, preprocessing_seed=preprocessing_seed, partition_seed=partition_seed)
        replay = run_source_pipeline(clr, implementation, safe_lattice, preprocessing_seed=preprocessing_seed, partition_seed=partition_seed)
        replay_passed = result_replay_equal(result, replay)
        full_replay_all_passed &= replay_passed
        timings.append({"implementationId": implementation.value, "mode": "full", "evaluations": 2, "wallSeconds": time.perf_counter() - branch_started})
        partition_rows.append(partition_row(result, trajectory_id=trajectory_id, matrix_index=matrix_index, mode_id=mode_id, fit_kind="completed_trajectory", endpoint_index=int(observation_index[-1]), endpoint_generation=int(generation[-1]), fit_count=len(clr), input_hash=sha256_array(clr), preprocessing_seed=preprocessing_seed, partition_seed=partition_seed, replay_passed=replay_passed))
        expected_count = len(clr) - result.local_offset
        for local_index in range(expected_count):
            raw_index = local_index + result.local_offset
            phi = float(result.local_phi_r[local_index]) if result.local_phi_r is not None and local_index < len(result.local_phi_r) else None
            diagnostic = float(result.emergence[local_index]) if result.emergence is not None and local_index < len(result.emergence) else None
            if not replay_passed:
                status, reason = "INELIGIBLE_EXACT_REPLAY_FAILED", "full_pipeline_exact_replay_failed"
                phi_out = None
            elif phi is None or not np.isfinite(phi):
                status, reason, phi_out = "INELIGIBLE_NONFINITE_LOCAL_VALUE", result.reason or "full_local_phi_r_nonfinite_or_absent", None
            elif result.status in {"ELIGIBLE", "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES"}:
                status, reason, phi_out = "ELIGIBLE", None, phi
            else:
                status, reason, phi_out = result.status, result.reason, None
            common = {
                "researchStepId": "S12B",
                "preregistrationVersion": VERSION,
                "implementationId": implementation.value,
                "temporalModeId": mode_id,
                "temporalLabel": "RETROSPECTIVE_FULL_TRAJECTORY_LOCAL",
                "trajectoryId": trajectory_id,
                "matrixIndex": matrix_index,
                "sourceLocalIndex": local_index,
                "rawObservationIndex": raw_index,
                "observationKind": str(observation_kind[raw_index]),
                "generation": int(generation[raw_index]),
                "molecularStep": int(molecular_step[raw_index]),
                "fitObservationCount": len(clr),
                "fitThroughObservationIndex": int(observation_index[-1]),
                "fitUsesFutureRelativeToValue": raw_index < int(observation_index[-1]),
                "localOffset": result.local_offset,
                "preprocessingSeed": preprocessing_seed,
                "partitionSeed": partition_seed,
                "status": status,
                "reason": reason,
                "phiR": phi_out,
                "sourceRelationship": "SOURCE_INFORMED_RECONSTRUCTION",
                "evidenceClass": EVIDENCE_CLASS,
            }
            full_rows.append(common)
            diagnostic_rows.append({**{key: common[key] for key in ("researchStepId", "implementationId", "temporalModeId", "trajectoryId", "matrixIndex", "rawObservationIndex", "generation", "molecularStep", "status", "reason")}, "diagnosticType": "source_named_emergence_synergy_plus_downward_causation", "diagnosticValue": diagnostic if diagnostic is not None and np.isfinite(diagnostic) else None, "comparisonValue": None, "difference": None, "variant": None})
        if not replay_passed:
            failure_rows.append({"failureId": f"{trajectory_id}-{implementation.value}-FULL-REPLAY", "stage": "full", "implementationId": implementation.value, "trajectoryId": trajectory_id, "observationIndex": int(observation_index[-1]), "status": "INELIGIBLE_EXACT_REPLAY_FAILED", "reason": "full_pipeline_exact_replay_failed", "fatal": True})

        cutoff = math.floor(0.25 * int(observation_index[-1]))
        quarter_mode = f"{mode_id}_FIRST_QUARTER_REFIT"
        q_pre_seed, q_part_seed = pipeline_seeds(root_seed, implementation, trajectory_id, quarter_mode, cutoff)
        quarter = run_source_pipeline(clr[: cutoff + 1], implementation, safe_lattice, preprocessing_seed=q_pre_seed, partition_seed=q_part_seed)
        quarter_replay = run_source_pipeline(clr[: cutoff + 1], implementation, safe_lattice, preprocessing_seed=q_pre_seed, partition_seed=q_part_seed)
        quarter_replay_passed = result_replay_equal(quarter, quarter_replay)
        full_replay_all_passed &= quarter_replay_passed
        partition_rows.append(partition_row(quarter, trajectory_id=trajectory_id, matrix_index=matrix_index, mode_id=quarter_mode, fit_kind="first_quarter_refit", endpoint_index=cutoff, endpoint_generation=int(generation[cutoff]), fit_count=cutoff + 1, input_hash=sha256_array(clr[: cutoff + 1]), preprocessing_seed=q_pre_seed, partition_seed=q_part_seed, replay_passed=quarter_replay_passed))
        quarter_expected = cutoff + 1 - quarter.local_offset
        for local_index in range(max(0, quarter_expected)):
            raw_index = local_index + quarter.local_offset
            full_value = float(result.local_phi_r[local_index]) if result.local_phi_r is not None and local_index < len(result.local_phi_r) else None
            quarter_value = float(quarter.local_phi_r[local_index]) if quarter.local_phi_r is not None and local_index < len(quarter.local_phi_r) else None
            finite_pair = full_value is not None and quarter_value is not None and np.isfinite(full_value) and np.isfinite(quarter_value) and replay_passed and quarter_replay_passed
            diagnostic_rows.append({"researchStepId": "S12B", "implementationId": implementation.value, "temporalModeId": quarter_mode, "trajectoryId": trajectory_id, "matrixIndex": matrix_index, "rawObservationIndex": raw_index, "generation": int(generation[raw_index]), "molecularStep": int(molecular_step[raw_index]), "status": "ELIGIBLE" if finite_pair else "INELIGIBLE_FIRST_QUARTER_COMPARISON", "reason": None if finite_pair else "full_or_quarter_value_nonfinite_or_replay_failed", "diagnosticType": "first_quarter_completed_fit_versus_quarter_refit", "diagnosticValue": full_value if finite_pair else None, "comparisonValue": quarter_value if finite_pair else None, "difference": full_value - quarter_value if finite_pair else None, "variant": None})

        prefix_mode = "IIGR_PREFIX_ENDPOINT" if implementation is SourceImplementation.IIGR else "PHIRL_PREFIX_ENDPOINT"
        prefix_started = time.perf_counter()
        evaluated = 0
        for endpoint in map(int, post_fission_indices):
            base = {
                "researchStepId": "S12B",
                "preregistrationVersion": VERSION,
                "implementationId": implementation.value,
                "temporalModeId": prefix_mode,
                "temporalLabel": "PAST_ONLY_PREFIX_ENDPOINT",
                "trajectoryId": trajectory_id,
                "matrixIndex": matrix_index,
                "rawObservationIndex": endpoint,
                "observationKind": "post_fission",
                "generation": int(generation[endpoint]),
                "molecularStep": int(molecular_step[endpoint]),
                "fitObservationCount": endpoint + 1,
                "fitThroughObservationIndex": endpoint,
                "fitUsesFutureRelativeToValue": False,
                "minimumTransitionBoundary": 256,
                "sourceRelationship": "SOURCE_INFORMED_RECONSTRUCTION",
                "evidenceClass": EVIDENCE_CLASS,
            }
            if molecular_step[endpoint] < 256:
                prefix_rows.append({**base, "prefixInputSha256": None, "preprocessingSeed": None, "partitionSeed": None, "status": "INELIGIBLE_BEFORE_256_TRANSITIONS", "reason": "fewer_than_256_preceding_molecular_transitions", "phiR": None, "exactReplayPassed": None, "futureSuffixStructuralGatePassed": True, "futureSuffixExecutedSentinelPassed": None})
                continue
            evaluated += 1
            pre_seed, part_seed = pipeline_seeds(root_seed, implementation, trajectory_id, prefix_mode, endpoint)
            prefix = clr[: endpoint + 1]
            prefix_hash = sha256_array(prefix)
            result_prefix = run_source_pipeline(prefix, implementation, safe_lattice, preprocessing_seed=pre_seed, partition_seed=part_seed)
            replay_prefix = run_source_pipeline(prefix, implementation, safe_lattice, preprocessing_seed=pre_seed, partition_seed=part_seed)
            replay_ok = result_replay_equal(result_prefix, replay_prefix)
            prefix_replay_all_passed &= replay_ok
            status, reason, phi_value, emergence_value = result_endpoint(result_prefix)
            if not replay_ok:
                status, reason, phi_value, emergence_value = "INELIGIBLE_EXACT_REPLAY_FAILED", "prefix_pipeline_exact_replay_failed", None, None
            sentinel_passed: bool | None = None
            if endpoint in sentinel_indices:
                sentinel_passed = True
                variants: list[tuple[str, np.ndarray]] = [("suffix_removed", prefix.copy())]
                suffix_shuffle = clr.copy()
                if endpoint + 1 < len(clr):
                    rng = np.random.RandomState(derive_seed(root_seed, implementation.value, trajectory_id, endpoint, "suffix_shuffle"))
                    order = rng.permutation(len(clr) - endpoint - 1)
                    suffix_shuffle[endpoint + 1 :] = suffix_shuffle[endpoint + 1 :][order]
                variants.append(("suffix_deterministically_shuffled", suffix_shuffle[: endpoint + 1]))
                suffix_replacement = clr.copy()
                if endpoint + 1 < len(clr):
                    rng = np.random.RandomState(derive_seed(root_seed, implementation.value, trajectory_id, endpoint, "suffix_replacement"))
                    suffix_replacement[endpoint + 1 :] = rng.normal(size=suffix_replacement[endpoint + 1 :].shape)
                variants.append(("suffix_replaced_by_domain_separated_gaussian_values", suffix_replacement[: endpoint + 1]))
                for variant_name, variant_prefix in variants:
                    variant = run_source_pipeline(variant_prefix, implementation, safe_lattice, preprocessing_seed=pre_seed, partition_seed=part_seed)
                    variant_ok = sha256_array(variant_prefix) == prefix_hash and result_replay_equal(result_prefix, variant)
                    sentinel_passed &= variant_ok
                    diagnostic_rows.append({"researchStepId": "S12B", "implementationId": implementation.value, "temporalModeId": prefix_mode, "trajectoryId": trajectory_id, "matrixIndex": matrix_index, "rawObservationIndex": endpoint, "generation": int(generation[endpoint]), "molecularStep": int(molecular_step[endpoint]), "status": "PASS" if variant_ok else "FAIL", "reason": None if variant_ok else "future_suffix_variant_changed_prefix_endpoint", "diagnosticType": "future_suffix_invariance", "diagnosticValue": phi_value, "comparisonValue": float(variant.local_phi_r[-1]) if variant.local_phi_r is not None and np.isfinite(variant.local_phi_r[-1]) else None, "difference": 0.0 if variant_ok else None, "variant": variant_name})
                suffix_all_passed &= sentinel_passed
                if not sentinel_passed:
                    status, reason, phi_value = "INELIGIBLE_FUTURE_SUFFIX_INVARIANCE_FAILED", "future_suffix_sentinel_failed", None
                    failure_rows.append({"failureId": f"{trajectory_id}-{implementation.value}-{endpoint}-SUFFIX", "stage": "prefix", "implementationId": implementation.value, "trajectoryId": trajectory_id, "observationIndex": endpoint, "status": status, "reason": reason, "fatal": True})
            prefix_rows.append({**base, "prefixInputSha256": prefix_hash, "preprocessingSeed": pre_seed, "partitionSeed": part_seed, "status": status, "reason": reason, "phiR": phi_value, "exactReplayPassed": replay_ok, "futureSuffixStructuralGatePassed": True, "futureSuffixExecutedSentinelPassed": sentinel_passed})
            diagnostic_rows.append({"researchStepId": "S12B", "implementationId": implementation.value, "temporalModeId": prefix_mode, "trajectoryId": trajectory_id, "matrixIndex": matrix_index, "rawObservationIndex": endpoint, "generation": int(generation[endpoint]), "molecularStep": int(molecular_step[endpoint]), "status": status, "reason": reason, "diagnosticType": "source_named_emergence_synergy_plus_downward_causation", "diagnosticValue": emergence_value, "comparisonValue": None, "difference": None, "variant": None})
            partition_rows.append(partition_row(result_prefix, trajectory_id=trajectory_id, matrix_index=matrix_index, mode_id=prefix_mode, fit_kind="past_only_prefix_endpoint", endpoint_index=endpoint, endpoint_generation=int(generation[endpoint]), fit_count=endpoint + 1, input_hash=prefix_hash, preprocessing_seed=pre_seed, partition_seed=part_seed, replay_passed=replay_ok))
            if not replay_ok:
                failure_rows.append({"failureId": f"{trajectory_id}-{implementation.value}-{endpoint}-REPLAY", "stage": "prefix", "implementationId": implementation.value, "trajectoryId": trajectory_id, "observationIndex": endpoint, "status": "INELIGIBLE_EXACT_REPLAY_FAILED", "reason": "prefix_pipeline_exact_replay_failed", "fatal": True})
        timings.append({"implementationId": implementation.value, "mode": "prefix", "evaluations": evaluated * 2, "wallSeconds": time.perf_counter() - prefix_started})

    write_parquet(trajectory_root / "full.parquet", pd.DataFrame(full_rows))
    write_parquet(trajectory_root / "prefix.parquet", pd.DataFrame(prefix_rows))
    write_parquet(trajectory_root / "partition.parquet", pd.DataFrame(partition_rows))
    write_parquet(trajectory_root / "diagnostic.parquet", pd.DataFrame(diagnostic_rows))
    write_csv(trajectory_root / "failures.csv", failure_rows, columns=["failureId", "stage", "implementationId", "trajectoryId", "observationIndex", "status", "reason", "fatal"])
    return {
        "trajectoryId": trajectory_id,
        "matrixIndex": matrix_index,
        "wallSeconds": time.perf_counter() - started_wall,
        "cpuSeconds": time.process_time() - started_cpu,
        "fullRowCount": len(full_rows),
        "prefixRowCount": len(prefix_rows),
        "partitionRowCount": len(partition_rows),
        "diagnosticRowCount": len(diagnostic_rows),
        "failureCount": len(failure_rows),
        "fullReplayPassed": full_replay_all_passed,
        "prefixReplayPassed": prefix_replay_all_passed,
        "futureSuffixPassed": suffix_all_passed,
        "timings": timings,
        "resultDirectory": str(trajectory_root),
    }


def collate_trajectory_results(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda item: item["matrixIndex"])
    full = pd.concat([pd.read_parquet(Path(item["resultDirectory"]) / "full.parquet") for item in ordered], ignore_index=True)
    prefix = pd.concat([pd.read_parquet(Path(item["resultDirectory"]) / "prefix.parquet") for item in ordered], ignore_index=True)
    partitions = pd.concat([pd.read_parquet(Path(item["resultDirectory"]) / "partition.parquet") for item in ordered], ignore_index=True)
    diagnostics = pd.concat([pd.read_parquet(Path(item["resultDirectory"]) / "diagnostic.parquet") for item in ordered], ignore_index=True)
    failures: list[dict[str, Any]] = []
    for item in ordered:
        failure_path = Path(item["resultDirectory"]) / "failures.csv"
        failure_frame = pd.read_csv(failure_path)
        failures.extend(failure_frame.to_dict("records"))
    return full, prefix, partitions, diagnostics, failures


def attach_historical_labels(full: pd.DataFrame, prefix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_parquet(S12_ROOT / "replicator_labels.parquet")
    labels = labels[labels["configurationId"] == HISTORICAL_CONFIG].copy()
    if len(labels) != 1200 or labels.duplicated(["trajectoryId", "generation"]).any():
        raise RuntimeError("historical S12 label cardinality is not exactly 12x100")
    mapping = {(str(row.trajectoryId), int(row.generation)): bool(row.isReplicator) for row in labels.itertuples()}
    for frame in (full, prefix):
        frame["historicalLabel"] = [mapping.get((str(tid), int(gen))) for tid, gen in zip(frame["trajectoryId"], frame["generation"], strict=True)]
        frame["nextHistoricalLabel"] = [mapping.get((str(tid), int(gen) + 1)) for tid, gen in zip(frame["trajectoryId"], frame["generation"], strict=True)]
    return full, prefix


def summary_fields(summary: Any) -> dict[str, Any]:
    return {
        "definedTrajectoryCorrelations": summary.defined_count,
        "positiveTrajectoryCorrelations": summary.positive_count,
        "medianTrajectoryCorrelation": summary.median,
        "bootstrapLower95": summary.bootstrap_lower,
        "bootstrapUpper95": summary.bootstrap_upper,
        "circularShiftPositiveP": summary.circular_positive_p,
        "circularShiftNegativeP": summary.circular_negative_p,
    }


def analyze_retrospective(full: pd.DataFrame, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root_seed = config["randomness"]["rootSeedHex"]
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for implementation in [item.value for item in SourceImplementation]:
        branch = full[full["implementationId"] == implementation].copy()
        post = branch[branch["observationKind"] == "post_fission"].copy()
        summary = association_summary(post, value_column="phiR", label_column="historicalLabel", bootstrap_seed=derive_seed(root_seed, implementation, "retrospective", "trajectory_bootstrap"), circular_seed=derive_seed(root_seed, implementation, "retrospective", "circular_shift_null"))
        higher_count = 0
        differences: list[float] = []
        median_differences: list[float] = []
        for trajectory_id, group in post.groupby("trajectoryId", sort=True):
            replicator = group.loc[(group["historicalLabel"] == True) & np.isfinite(group["phiR"]), "phiR"].to_numpy(dtype=float)
            drift = group.loc[(group["historicalLabel"] == False) & np.isfinite(group["phiR"]), "phiR"].to_numpy(dtype=float)
            difference = float(np.mean(replicator) - np.mean(drift)) if replicator.size and drift.size else np.nan
            median_difference = float(np.median(replicator) - np.median(drift)) if replicator.size and drift.size else np.nan
            differences.append(difference)
            median_differences.append(median_difference)
            higher_count += int(np.isfinite(difference) and difference > 0.0)
            rows.append({"rowType": "TRAJECTORY_ASSOCIATION", "implementationId": implementation, "trajectoryId": trajectory_id, "estimand": "current_generation", "spearmanRho": summary.trajectory_correlations.get(str(trajectory_id)), "replicatorMeanPhiR": float(np.mean(replicator)) if replicator.size else None, "driftMeanPhiR": float(np.mean(drift)) if drift.size else None, "replicatorMinusDriftMean": difference if np.isfinite(difference) else None, "replicatorMedianPhiR": float(np.median(replicator)) if replicator.size else None, "driftMedianPhiR": float(np.median(drift)) if drift.size else None, "replicatorMinusDriftMedian": median_difference if np.isfinite(median_difference) else None, "finiteCoverage": float(np.isfinite(group["phiR"]).mean()), "n": len(group)})
        finite_coverage = float(np.isfinite(branch["phiR"]).mean())
        coherent, gates = retrospective_coherent(summary, finite_coverage=finite_coverage, runs_higher=higher_count)
        trend = molecular_progress_trend(branch)
        ljung = ljung_box_summary(branch)
        pooled_replicator = post.loc[post["historicalLabel"] == True, "phiR"]
        pooled_drift = post.loc[post["historicalLabel"] == False, "phiR"]
        pooled_rep_mean, pooled_drift_mean = float(pooled_replicator.mean()), float(pooled_drift.mean())
        pooled_rep_median, pooled_drift_median = float(pooled_replicator.median()), float(pooled_drift.median())
        rows.append({"rowType": "IMPLEMENTATION_SUMMARY", "implementationId": implementation, "trajectoryId": None, "estimand": "current_generation", "spearmanRho": None, "replicatorMeanPhiR": pooled_rep_mean, "driftMeanPhiR": pooled_drift_mean, "replicatorMinusDriftMean": pooled_rep_mean - pooled_drift_mean, "replicatorMedianPhiR": pooled_rep_median, "driftMedianPhiR": pooled_drift_median, "replicatorMinusDriftMedian": pooled_rep_median - pooled_drift_median, "medianTrajectoryMeanDifference": float(np.nanmedian(differences)), "medianTrajectoryMedianDifference": float(np.nanmedian(median_differences)), "finiteCoverage": finite_coverage, "n": len(branch), **summary_fields(summary), "runsWithHigherReplicationPhiR": higher_count, "coherentPaperDirectedAssociation": coherent, "coherenceGatesJson": json.dumps(gates, sort_keys=True), "aggregateTrendSlope": trend["slope"], "aggregateTrendPValue": trend["pValue"], "ljungBoxSignificantRuns": sum(bool(item["significantAt0p05"]) for item in ljung if item["significantAt0p05"] is not None)})
        for item in ljung:
            rows.append({"rowType": "LJUNG_BOX", "implementationId": implementation, "trajectoryId": item["trajectoryId"], "estimand": "full_local_series", "spearmanRho": None, "finiteCoverage": None, "n": item["n"], "ljungBoxLag": item["lag"], "ljungBoxStatistic": item["statistic"], "ljungBoxPValue": item["pValue"], "ljungBoxSignificant": item["significantAt0p05"]})
        summaries[implementation] = {"association": summary, "finiteCoverage": finite_coverage, "runsHigher": higher_count, "coherent": coherent, "gates": gates, "trend": trend, "ljungBox": ljung}
    return rows, summaries


def analyze_prospective(prefix: pd.DataFrame, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root_seed = config["randomness"]["rootSeedHex"]
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    boundary = prefix[prefix["molecularStep"] >= 256].copy()
    for implementation in [item.value for item in SourceImplementation]:
        branch = boundary[boundary["implementationId"] == implementation].copy()
        coverage = float((branch["status"] == "ELIGIBLE").mean()) if len(branch) else 0.0
        implementation_summaries: dict[str, Any] = {"coverage": coverage}
        first_eligible: dict[str, int | None] = {}
        for trajectory_id, group in branch.groupby("trajectoryId", sort=True):
            eligible = group[group["status"] == "ELIGIBLE"]
            first_eligible[str(trajectory_id)] = int(eligible["generation"].min()) if len(eligible) else None
        for estimand, label_column in (("current_generation_rho_0", "historicalLabel"), ("next_generation_rho_plus_1", "nextHistoricalLabel")):
            summary = association_summary(branch, value_column="phiR", label_column=label_column, bootstrap_seed=derive_seed(root_seed, implementation, estimand, "trajectory_bootstrap"), circular_seed=derive_seed(root_seed, implementation, estimand, "circular_shift_null"))
            for trajectory_id in sorted(branch["trajectoryId"].unique()):
                group = branch[branch["trajectoryId"] == trajectory_id]
                rows.append({"rowType": "TRAJECTORY_ASSOCIATION", "implementationId": implementation, "trajectoryId": trajectory_id, "estimand": estimand, "spearmanRho": summary.trajectory_correlations.get(str(trajectory_id)), "eligibleCount": int((group["status"] == "ELIGIBLE").sum()), "expectedAfterBoundaryCount": len(group), "eligibleCoverage": float((group["status"] == "ELIGIBLE").mean()), "firstEligibleGeneration": first_eligible[str(trajectory_id)]})
            rows.append({"rowType": "IMPLEMENTATION_SUMMARY", "implementationId": implementation, "trajectoryId": None, "estimand": estimand, "spearmanRho": None, "eligibleCount": int((branch["status"] == "ELIGIBLE").sum()), "expectedAfterBoundaryCount": len(branch), "eligibleCoverage": coverage, "firstEligibleGeneration": float(np.median([value for value in first_eligible.values() if value is not None])) if any(value is not None for value in first_eligible.values()) else None, **summary_fields(summary)})
            implementation_summaries[estimand] = summary
        implementation_summaries["firstEligible"] = first_eligible
        summaries[implementation] = implementation_summaries
    return rows, summaries


def analyze_spikes(full: pd.DataFrame, prefix: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    thresholds: dict[str, dict[str, float]] = {}
    boundary_prefix = prefix[(prefix["molecularStep"] >= 256) & (prefix["status"] == "ELIGIBLE")].copy()
    for implementation in [item.value for item in SourceImplementation]:
        full_branch = full[full["implementationId"] == implementation].copy()
        prefix_branch = boundary_prefix[boundary_prefix["implementationId"] == implementation].copy()
        full_threshold = spike_thresholds(full_branch["phiR"].to_numpy(dtype=float))
        prefix_threshold = spike_thresholds(prefix_branch["phiR"].to_numpy(dtype=float))
        thresholds[implementation] = {**{f"full_{key}": value for key, value in full_threshold.items()}, **{f"prefix_{key}": value for key, value in prefix_threshold.items()}}
        rows.append({"rowType": "IMPLEMENTATION_SUMMARY", "implementationId": implementation, "trajectoryId": None, "finiteValueCount": int(np.isfinite(full_branch["phiR"]).sum()), "positive3SigmaCount": int((full_branch["phiR"] > full_threshold["positive3Sigma"]).sum()), "negative3SigmaCount": int((full_branch["phiR"] < full_threshold["negative3Sigma"]).sum()), "robustMadPositiveCount": int((full_branch["phiR"] > full_threshold["robustPositive"]).sum()), **{f"fullThreshold_{key}": value for key, value in full_threshold.items()}, **{f"prefixThreshold_{key}": value for key, value in prefix_threshold.items()}})
        aggregate_full_set: set[tuple[str, int]] = set()
        aggregate_prefix_set: set[tuple[str, int]] = set()
        for trajectory_id, full_group in full_branch.groupby("trajectoryId", sort=True):
            prefix_group = prefix_branch[prefix_branch["trajectoryId"] == trajectory_id]
            shared = full_group[full_group["rawObservationIndex"].isin(prefix_group["rawObservationIndex"])]
            full_set = {(str(trajectory_id), int(row.rawObservationIndex)) for row in shared.itertuples() if np.isfinite(row.phiR) and row.phiR > full_threshold["positive3Sigma"]}
            prefix_set = {(str(trajectory_id), int(row.rawObservationIndex)) for row in prefix_group.itertuples() if np.isfinite(row.phiR) and row.phiR > prefix_threshold["positive3Sigma"]}
            aggregate_full_set |= full_set
            aggregate_prefix_set |= prefix_set
            union = full_set | prefix_set
            jaccard = len(full_set & prefix_set) / len(union) if union else 1.0
            rows.append({"rowType": "TRAJECTORY", "implementationId": implementation, "trajectoryId": trajectory_id, "finiteValueCount": int(np.isfinite(full_group["phiR"]).sum()), "positive3SigmaCount": int((full_group["phiR"] > full_threshold["positive3Sigma"]).sum()), "negative3SigmaCount": int((full_group["phiR"] < full_threshold["negative3Sigma"]).sum()), "robustMadPositiveCount": int((full_group["phiR"] > full_threshold["robustPositive"]).sum()), "sharedFullSpikeCount": len(full_set), "sharedPrefixSpikeCount": len(prefix_set), "sharedSpikeIntersection": len(full_set & prefix_set), "sharedSpikeUnion": len(union), "spikeJaccard": jaccard})
        aggregate_union = aggregate_full_set | aggregate_prefix_set
        rows.append({"rowType": "SHARED_ENDPOINT_AGGREGATE", "implementationId": implementation, "trajectoryId": None, "sharedFullSpikeCount": len(aggregate_full_set), "sharedPrefixSpikeCount": len(aggregate_prefix_set), "sharedSpikeIntersection": len(aggregate_full_set & aggregate_prefix_set), "sharedSpikeUnion": len(aggregate_union), "spikeJaccard": len(aggregate_full_set & aggregate_prefix_set) / len(aggregate_union) if aggregate_union else 1.0})
    return rows, thresholds


def feature_partition_labels(row: pd.Series) -> np.ndarray:
    labels = np.full(99, -1, dtype=np.int64)
    for value in json.loads(row["partition1Json"]):
        labels[int(value)] = 0
    for value in json.loads(row["partition2Json"]):
        labels[int(value)] = 1
    return labels


def analyze_future_dependence(
    full: pd.DataFrame,
    prefix: pd.DataFrame,
    partitions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    endpoint_ari_rows: list[dict[str, Any]] = []
    for implementation in [item.value for item in SourceImplementation]:
        full_branch = full[(full["implementationId"] == implementation) & (full["observationKind"] == "post_fission")][["trajectoryId", "rawObservationIndex", "generation", "phiR", "historicalLabel"]].rename(columns={"phiR": "fullPhiR"})
        prefix_branch = prefix[(prefix["implementationId"] == implementation) & (prefix["molecularStep"] >= 256) & (prefix["status"] == "ELIGIBLE")][["trajectoryId", "rawObservationIndex", "generation", "phiR", "historicalLabel"]].rename(columns={"phiR": "prefixPhiR", "historicalLabel": "prefixHistoricalLabel"})
        shared = full_branch.merge(prefix_branch, on=["trajectoryId", "rawObservationIndex", "generation"], how="inner")
        shared = shared[np.isfinite(shared["fullPhiR"]) & np.isfinite(shared["prefixPhiR"])].copy()
        full_partitions = partitions[(partitions["implementationId"] == implementation) & (partitions["fitKind"] == "completed_trajectory")].set_index("trajectoryId")
        prefix_partitions = partitions[(partitions["implementationId"] == implementation) & (partitions["fitKind"] == "past_only_prefix_endpoint")].set_index(["trajectoryId", "endpointObservationIndex"])
        for row in shared.itertuples():
            try:
                full_partition = feature_partition_labels(full_partitions.loc[row.trajectoryId])
                prefix_partition = feature_partition_labels(prefix_partitions.loc[(row.trajectoryId, row.rawObservationIndex)])
                ari = float(adjusted_rand_score(full_partition, prefix_partition))
            except (KeyError, TypeError, ValueError):
                ari = np.nan
            endpoint_ari_rows.append({"implementationId": implementation, "trajectoryId": row.trajectoryId, "rawObservationIndex": int(row.rawObservationIndex), "generation": int(row.generation), "partitionAdjustedRandIndex": ari})
        implementation_ari = pd.DataFrame([item for item in endpoint_ari_rows if item["implementationId"] == implementation])
        per_trajectory: list[dict[str, Any]] = []
        for trajectory_id, group in shared.groupby("trajectoryId", sort=True):
            full_values = group["fullPhiR"].to_numpy(dtype=float)
            prefix_values = group["prefixPhiR"].to_numpy(dtype=float)
            delta = full_values - prefix_values
            full_iqr = float(np.quantile(full_values, 0.75) - np.quantile(full_values, 0.25)) if len(full_values) else np.nan
            full_spikes = set(group.loc[group["fullPhiR"] > thresholds[implementation]["full_positive3Sigma"], "rawObservationIndex"].astype(int))
            prefix_spikes = set(group.loc[group["prefixPhiR"] > thresholds[implementation]["prefix_positive3Sigma"], "rawObservationIndex"].astype(int))
            union = full_spikes | prefix_spikes
            current_full = finite_spearman(full_values, group["historicalLabel"].astype(float).to_numpy())
            current_prefix = finite_spearman(prefix_values, group["historicalLabel"].astype(float).to_numpy())
            ari_values = implementation_ari.loc[implementation_ari["trajectoryId"] == trajectory_id, "partitionAdjustedRandIndex"].to_numpy(dtype=float) if len(implementation_ari) else np.array([], dtype=float)
            metrics = {
                "rowType": "SHARED_ENDPOINT_TRAJECTORY",
                "implementationId": implementation,
                "trajectoryId": trajectory_id,
                "sharedPointCount": len(group),
                "medianAbsoluteDifference": float(np.median(np.abs(delta))),
                "fullInterquartileRange": full_iqr,
                "medianAbsoluteDifferenceNormalizedByFullIqr": float(np.median(np.abs(delta)) / full_iqr) if full_iqr > 0 else None,
                "fullPrefixSpearman": finite_spearman(full_values, prefix_values),
                "signAgreement": float(np.mean(np.signbit(full_values) == np.signbit(prefix_values))),
                "spikeJaccard": len(full_spikes & prefix_spikes) / len(union) if union else 1.0,
                "fullReplicationSpearman": current_full,
                "prefixReplicationSpearman": current_prefix,
                "replicationAssociationDifferenceFullMinusPrefix": current_full - current_prefix if current_full is not None and current_prefix is not None else None,
                "medianPartitionAdjustedRandIndex": float(np.nanmedian(ari_values)) if np.isfinite(ari_values).any() else None,
                "rankChangeGreaterThan10PctFraction": float(np.mean(np.abs(percentile_ranks(full_values) - percentile_ranks(prefix_values)) > 0.10)),
            }
            rows.append(metrics)
            per_trajectory.append(metrics)
        if len(shared):
            full_values = shared["fullPhiR"].to_numpy(dtype=float)
            prefix_values = shared["prefixPhiR"].to_numpy(dtype=float)
            delta = full_values - prefix_values
            full_iqr = float(np.quantile(full_values, 0.75) - np.quantile(full_values, 0.25))
            rows.append({"rowType": "SHARED_ENDPOINT_IMPLEMENTATION_SUMMARY", "implementationId": implementation, "trajectoryId": None, "sharedPointCount": len(shared), "medianAbsoluteDifference": float(np.median(np.abs(delta))), "fullInterquartileRange": full_iqr, "medianAbsoluteDifferenceNormalizedByFullIqr": float(np.median(np.abs(delta)) / full_iqr) if full_iqr > 0 else None, "fullPrefixSpearman": finite_spearman(full_values, prefix_values), "signAgreement": float(np.mean(np.signbit(full_values) == np.signbit(prefix_values))), "spikeJaccard": float(np.median([item["spikeJaccard"] for item in per_trajectory])) if per_trajectory else None, "replicationAssociationDifferenceFullMinusPrefix": float(np.nanmedian([item["replicationAssociationDifferenceFullMinusPrefix"] for item in per_trajectory if item["replicationAssociationDifferenceFullMinusPrefix"] is not None])) if any(item["replicationAssociationDifferenceFullMinusPrefix"] is not None for item in per_trajectory) else None, "medianPartitionAdjustedRandIndex": float(np.nanmedian(implementation_ari["partitionAdjustedRandIndex"])) if len(implementation_ari) and np.isfinite(implementation_ari["partitionAdjustedRandIndex"]).any() else None, "rankChangeGreaterThan10PctFraction": float(np.mean(np.abs(percentile_ranks(full_values) - percentile_ranks(prefix_values)) > 0.10))})
        quarter = diagnostics[(diagnostics["implementationId"] == implementation) & (diagnostics["diagnosticType"] == "first_quarter_completed_fit_versus_quarter_refit") & (diagnostics["status"] == "ELIGIBLE")]
        quarter_per: list[dict[str, Any]] = []
        for trajectory_id, group in quarter.groupby("trajectoryId", sort=True):
            full_values = group["diagnosticValue"].to_numpy(dtype=float)
            refit_values = group["comparisonValue"].to_numpy(dtype=float)
            delta = full_values - refit_values
            full_iqr = float(np.quantile(full_values, 0.75) - np.quantile(full_values, 0.25)) if len(full_values) else np.nan
            metric = {"rowType": "FIRST_QUARTER_TRAJECTORY", "implementationId": implementation, "trajectoryId": trajectory_id, "sharedPointCount": len(group), "medianAbsoluteDifference": float(np.median(np.abs(delta))), "fullInterquartileRange": full_iqr, "medianAbsoluteDifferenceNormalizedByFullIqr": float(np.median(np.abs(delta)) / full_iqr) if full_iqr > 0 else None, "fullPrefixSpearman": finite_spearman(full_values, refit_values), "signAgreement": float(np.mean(np.signbit(full_values) == np.signbit(refit_values))), "rankChangeGreaterThan10PctFraction": float(np.mean(np.abs(percentile_ranks(full_values) - percentile_ranks(refit_values)) > 0.10))}
            rows.append(metric)
            quarter_per.append(metric)
        if quarter_per:
            rows.append({"rowType": "FIRST_QUARTER_IMPLEMENTATION_SUMMARY", "implementationId": implementation, "trajectoryId": None, "sharedPointCount": int(sum(item["sharedPointCount"] for item in quarter_per)), "medianAbsoluteDifference": float(np.median([item["medianAbsoluteDifference"] for item in quarter_per])), "medianAbsoluteDifferenceNormalizedByFullIqr": float(np.nanmedian([item["medianAbsoluteDifferenceNormalizedByFullIqr"] for item in quarter_per if item["medianAbsoluteDifferenceNormalizedByFullIqr"] is not None])), "fullPrefixSpearman": float(np.nanmedian([item["fullPrefixSpearman"] for item in quarter_per if item["fullPrefixSpearman"] is not None])), "signAgreement": float(np.median([item["signAgreement"] for item in quarter_per])), "rankChangeGreaterThan10PctFraction": float(np.median([item["rankChangeGreaterThan10PctFraction"] for item in quarter_per]))})
    return rows, pd.DataFrame(endpoint_ari_rows)


def decide_classification(
    retrospective: dict[str, Any],
    prospective: dict[str, Any],
    *,
    replay_passed: bool,
    suffix_passed: bool,
) -> dict[str, Any]:
    implementations = [item.value for item in SourceImplementation]
    opposite = {
        implementation: significant_opposite(prospective[implementation]["current_generation_rho_0"])
        for implementation in implementations
    }
    candidate_flags: dict[str, bool] = {}
    candidate_gates: dict[str, Any] = {}
    for implementation in implementations:
        other = next(item for item in implementations if item != implementation)
        passed, gates = prospective_candidate(
            prospective[implementation]["current_generation_rho_0"],
            coverage=prospective[implementation]["coverage"],
            replay_passed=replay_passed,
            suffix_passed=suffix_passed,
            other_opposite=opposite[other],
        )
        candidate_flags[implementation] = passed
        candidate_gates[implementation] = gates
    full_flags = {implementation: bool(retrospective[implementation]["coherent"]) for implementation in implementations}
    iigr, phirl = SourceImplementation.IIGR.value, SourceImplementation.PHIRL.value
    if full_flags[phirl] and not full_flags[iigr]:
        classification = "REGULARIZATION_DEPENDENT_RESEMBLANCE"
        recommendation = "Do not classify this as GARD-paper replication or begin S13. Return for human review because the resemblance depends on the later regularized public implementation."
    elif any(candidate_flags.values()):
        classification = "SOURCE_FAMILY_PROSPECTIVE_CANDIDATE"
        recommendation = "Return for human review with a proposal limited to baseline-only S13; do not authorize interventions and do not start S13 automatically."
    elif any(full_flags.values()):
        classification = "RETROSPECTIVE_SOURCE_FAMILY_RESEMBLANCE"
        recommendation = "Preserve the retrospective resemblance as a possible explanation of paper-like figures, but do not authorize early-warning/intervention S13; return for human review."
    else:
        classification = "SOURCE_FAMILY_NOT_SUPPORTED"
        recommendation = "Do not authorize S13. Close the E01 Phi-r reconstruction or move replacement-variable work to a separately preregistered E02 after human review."
    return {
        "schema": "eidosoma.e01.s12b_classification.v1",
        "researchStepId": "S12B",
        "preregistrationVersion": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "sourceRelationship": "SOURCE_INFORMED_RECONSTRUCTION",
        "classification": classification,
        "fullRetrospectiveCoherence": full_flags,
        "prospectiveCandidate": candidate_flags,
        "prospectiveCandidateGates": candidate_gates,
        "significantOppositeDirection": opposite,
        "primaryEstimand": "current_generation_rho_0",
        "s12Status": "UNCHANGED_AND_NOT_SUBSTITUTED",
        "s13Status": "BLOCKED_PENDING_S12B_HUMAN_REVIEW",
        "automaticS13Authorized": False,
        "interventionAuthorized": False,
        "recommendedNextAction": recommendation,
    }


def create_figures(
    full: pd.DataFrame,
    prefix: pd.DataFrame,
    retrospective_rows: list[dict[str, Any]],
    prospective_rows: list[dict[str, Any]],
    spike_rows: list[dict[str, Any]],
    endpoint_ari: pd.DataFrame,
    classification: dict[str, Any],
    config: dict[str, Any],
) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    colors = {SourceImplementation.IIGR.value: "#1f77b4", SourceImplementation.PHIRL.value: "#d62728"}
    trajectories = sorted(full["trajectoryId"].unique())
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=False)
    for axis, trajectory_id in zip(axes.ravel(), trajectories, strict=True):
        for implementation, color in colors.items():
            group = full[(full["trajectoryId"] == trajectory_id) & (full["implementationId"] == implementation) & np.isfinite(full["phiR"])].sort_values("molecularStep")
            if len(group) > 2500:
                take = np.linspace(0, len(group) - 1, 2500).astype(int)
                group = group.iloc[take]
            axis.plot(group["molecularStep"], group["phiR"], lw=0.65, alpha=0.8, color=color, label=implementation.replace("_SOURCE", ""))
        axis.set_title(str(trajectory_id), fontsize=9)
        axis.axhline(0.0, color="black", lw=0.4)
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("Molecular step")
    fig.supylabel("Local Phi-r (nats)")
    fig.suptitle("Completed-trajectory local Phi-r: matched public source reconstructions")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "full_trajectory_matched_sources.png", dpi=180)
    plt.close(fig)

    representative = config["analysis"]["representativeFigureMatrixIndices"]
    fig, axes = plt.subplots(2, len(representative), figsize=(15, 7), sharex=False)
    for row_index, implementation in enumerate(colors):
        for column_index, matrix_index in enumerate(representative):
            axis = axes[row_index, column_index]
            trajectory_id = f"E01-S12-B{matrix_index:02d}"
            full_group = full[(full["trajectoryId"] == trajectory_id) & (full["implementationId"] == implementation) & (full["observationKind"] == "post_fission")]
            prefix_group = prefix[(prefix["trajectoryId"] == trajectory_id) & (prefix["implementationId"] == implementation) & (prefix["status"] == "ELIGIBLE")]
            axis.plot(full_group["generation"], full_group["phiR"], color="#4c78a8", lw=1.0, label="full fit")
            axis.plot(prefix_group["generation"], prefix_group["phiR"], color="#f58518", lw=0.8, marker="o", ms=2, label="prefix fit")
            axis.set_title(f"{implementation.split('_')[0]} · B{matrix_index:02d}", fontsize=9)
            axis.axhline(0.0, color="black", lw=0.4)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("Generation (post-fission endpoint)")
    fig.supylabel("Local Phi-r (nats)")
    fig.suptitle("Completed-fit versus past-only prefix endpoint values")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "full_versus_prefix_representative.png", dpi=180)
    plt.close(fig)

    assoc_rows = [row for row in retrospective_rows + prospective_rows if row.get("rowType") == "TRAJECTORY_ASSOCIATION" and row.get("spearmanRho") is not None]
    assoc = pd.DataFrame(assoc_rows)
    fig, axis = plt.subplots(figsize=(11, 6))
    if len(assoc):
        assoc["branch"] = assoc["implementationId"].str.split("_").str[0] + "\n" + assoc["estimand"]
        labels = list(dict.fromkeys(assoc["branch"]))
        rng = np.random.RandomState(12012)
        for index, label in enumerate(labels):
            values = assoc.loc[assoc["branch"] == label, "spearmanRho"].to_numpy(dtype=float)
            axis.boxplot(values, positions=[index], widths=0.55, showfliers=False)
            axis.scatter(index + rng.uniform(-0.08, 0.08, size=len(values)), values, s=18, alpha=0.75)
        axis.set_xticks(range(len(labels)), labels, fontsize=8)
    axis.axhline(0.0, color="black", lw=0.7)
    axis.set_ylabel("Within-trajectory Spearman rho")
    axis.set_title("Retrospective and prefix association distributions")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "association_distributions.png", dpi=180)
    plt.close(fig)

    spike = pd.DataFrame([row for row in spike_rows if row.get("rowType") == "TRAJECTORY"])
    fig, axis = plt.subplots(figsize=(10, 5))
    if len(spike):
        positions = np.arange(len(sorted(spike["trajectoryId"].unique())))
        width = 0.38
        for offset, implementation in zip((-width / 2, width / 2), colors, strict=True):
            values = spike[spike["implementationId"] == implementation].sort_values("trajectoryId")["spikeJaccard"].to_numpy(dtype=float)
            axis.bar(positions + offset, values, width, label=implementation.split("_")[0], color=colors[implementation])
        axis.set_xticks(positions, [item.replace("E01-S12-", "") for item in sorted(spike["trajectoryId"].unique())])
        axis.legend()
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Positive 3-sigma spike Jaccard")
    axis.set_title("Full-fit versus prefix spike overlap")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "spike_overlap.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    if len(endpoint_ari):
        for index, implementation in enumerate(colors):
            values = endpoint_ari.loc[endpoint_ari["implementationId"] == implementation, "partitionAdjustedRandIndex"].dropna().to_numpy(dtype=float)
            axis.violinplot(values, positions=[index], showmedians=True, widths=0.75)
            if len(values):
                take = np.linspace(0, len(values) - 1, min(300, len(values))).astype(int)
                axis.scatter(np.full(len(take), index) + np.linspace(-0.08, 0.08, len(take)), values[take], s=5, alpha=0.25, color=colors[implementation])
        axis.set_xticks(range(len(colors)), [item.split("_")[0] for item in colors])
    axis.set_ylim(-0.05, 1.05)
    axis.set_ylabel("Adjusted Rand index")
    axis.set_title("Completed-fit versus prefix partition stability")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "partition_stability.png", dpi=180)
    plt.close(fig)

    gate_names = ["full coherent", "prefix candidate", "no opposite direction"]
    matrix = np.array([[classification["fullRetrospectiveCoherence"][implementation], classification["prospectiveCandidate"][implementation], not classification["significantOppositeDirection"][implementation]] for implementation in colors], dtype=int).T
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(2), [item.split("_")[0] for item in colors])
    axis.set_yticks(range(3), gate_names)
    for y in range(3):
        for x in range(2):
            axis.text(x, y, "PASS" if matrix[y, x] else "FAIL", ha="center", va="center", fontweight="bold")
    axis.set_title(f"Final decision: {classification['classification']}")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "final_decision_matrix.png", dpi=180)
    plt.close(fig)


def create_placeholder_figures(message: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    names = ["full_trajectory_matched_sources.png", "full_versus_prefix_representative.png", "association_distributions.png", "spike_overlap.png", "partition_stability.png", "final_decision_matrix.png"]
    for name in names:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.axis("off")
        axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=12)
        fig.tight_layout()
        fig.savefig(FIGURE_ROOT / name, dpi=150)
        plt.close(fig)


def ensure_empty_required_tables() -> None:
    parquet_columns = {
        "full_trajectory_local_values.parquet": ["researchStepId", "implementationId", "trajectoryId", "rawObservationIndex", "status", "reason", "phiR"],
        "prefix_endpoint_values.parquet": ["researchStepId", "implementationId", "trajectoryId", "rawObservationIndex", "generation", "molecularStep", "status", "reason", "phiR"],
        "partition_history.parquet": ["researchStepId", "implementationId", "trajectoryId", "fitKind", "status", "reason"],
        "source_diagnostic_outputs.parquet": ["researchStepId", "implementationId", "trajectoryId", "diagnosticType", "status", "reason", "diagnosticValue"],
    }
    for name, columns in parquet_columns.items():
        path = STEP_ROOT / name
        if not path.exists():
            write_parquet(path, pd.DataFrame({column: pd.Series(dtype="object") for column in columns}))
    csv_columns = {
        "retrospective_associations.csv": ["rowType", "implementationId", "trajectoryId", "estimand", "spearmanRho"],
        "prospective_associations.csv": ["rowType", "implementationId", "trajectoryId", "estimand", "spearmanRho"],
        "spike_analysis.csv": ["rowType", "implementationId", "trajectoryId", "positive3SigmaCount", "spikeJaccard"],
        "future_dependence_results.csv": ["rowType", "implementationId", "trajectoryId", "sharedPointCount", "medianAbsoluteDifference"],
    }
    for name, columns in csv_columns.items():
        path = STEP_ROOT / name
        if not path.exists():
            write_csv(path, [], columns=columns)


def write_immutable_post_audit(preflight: dict[str, Any], postflight: dict[str, Any]) -> None:
    write_json(
        STEP_ROOT / "immutable_input_audit.json",
        {
            "schema": "eidosoma.e01.s12b_immutable_input_audit.v1",
            "researchStepId": "S12B",
            "preOutcome": {
                "success": preflight["success"],
                "priorArtifacts": preflight["priorArtifacts"],
                "priorRepository": preflight["priorRepository"],
                "frozenInputs": preflight["frozenInputs"],
            },
            "postOutcome": {
                "success": postflight["success"],
                "priorArtifacts": postflight["priorArtifacts"],
                "priorRepository": postflight["priorRepository"],
                "sources": postflight["sources"],
            },
            "approvedMutableFiles": ["/workspace/RESEARCH_PLAN.md_after_artifact_finalization_only"],
            "success": preflight["success"] and postflight["success"],
        },
    )


def report_markdown(
    *,
    success: bool,
    completion_status: str,
    outcome_classification: str,
    decision: dict[str, Any],
    validation_result: str,
    caveats: list[str],
    recommendation: str,
    runtime: dict[str, Any],
    equivalence: pd.DataFrame,
    retrospective: dict[str, Any] | None,
    prospective: dict[str, Any] | None,
    future_rows: list[dict[str, Any]] | None,
    spike_rows: list[dict[str, Any]] | None,
    artifacts_written: list[str],
) -> str:
    eq_max_mi = float(equivalence["miMaxAbsDifference"].dropna().max()) if len(equivalence) and equivalence["miMaxAbsDifference"].notna().any() else None
    eq_max_phi = float(equivalence["localPhiRMaxAbsDifference"].dropna().max()) if len(equivalence) and equivalence["localPhiRMaxAbsDifference"].notna().any() else None
    lay = (
        "The audit compared local Phi-r calculated after fitting each public source pipeline to an entire completed GARD run with the value obtained when that same pipeline saw only the past available at each fission. "
        + ("The bounded audit completed and its decision was `" + decision["classification"] + "`." if success else "A frozen stop condition fired, so no favorable interpretation or method repair was attempted.")
    )
    summary_lines = []
    if retrospective and prospective:
        for implementation in [item.value for item in SourceImplementation]:
            full_summary = retrospective[implementation]["association"]
            prefix_summary = prospective[implementation]["current_generation_rho_0"]
            summary_lines.append(
                f"| {implementation} | {retrospective[implementation]['finiteCoverage']:.4f} | {full_summary.median if full_summary.median is not None else 'NA'} | {prospective[implementation]['coverage']:.4f} | {prefix_summary.median if prefix_summary.median is not None else 'NA'} | {retrospective[implementation]['coherent']} | {decision['prospectiveCandidate'][implementation]} |"
            )
    results_table = "\n".join(summary_lines) if summary_lines else "| No scientific summary emitted because execution stopped at a mandatory gate. | | | | | | |"
    future_summary = [row for row in (future_rows or []) if row.get("rowType") in {"SHARED_ENDPOINT_IMPLEMENTATION_SUMMARY", "FIRST_QUARTER_IMPLEMENTATION_SUMMARY"}]
    future_text = "\n".join(
        f"- `{row['implementationId']}` / `{row['rowType']}`: n={row.get('sharedPointCount')}, median |difference|={row.get('medianAbsoluteDifference')}, full-vs-refit Spearman={row.get('fullPrefixSpearman')}, rank-change fraction={row.get('rankChangeGreaterThan10PctFraction')}."
        for row in future_summary
    ) or "- Not evaluated because a stop condition fired."
    spike_summary = [row for row in (spike_rows or []) if row.get("rowType") == "IMPLEMENTATION_SUMMARY"]
    spike_text = "\n".join(
        f"- `{row['implementationId']}`: positive 3-sigma={row.get('positive3SigmaCount')}, negative 3-sigma={row.get('negative3SigmaCount')}, robust-MAD positive={row.get('robustMadPositiveCount')}."
        for row in spike_summary
    ) or "- Not evaluated because a stop condition fired."
    artifact_text = ", ".join(f"`{item}`" for item in artifacts_written)
    caveat_text = "\n".join(f"- {item}" for item in caveats)
    return f"""# E01-S12B Source-Code Reconstruction and Future-Dependence Audit of Local Phi-r

## Top summary

- **Research step ID:** S12B (`E01-S12B-PIGOZZI-SOURCE-CODE-AUDIT-v1.0.0`)
- **Completion status:** {completion_status}
- **Artifacts written:** {artifact_text}
- **Validation result:** {validation_result}
- **Outcome classification:** {outcome_classification}; decision `{decision['classification']}`.
- **Caveats or blockers:** {caveats[0] if caveats else 'None beyond the frozen interpretation boundary.'}
- **Lay summary:** {lay}
- **Recommended next action:** {recommendation}

## Frozen question

Do the exact public `IntegratedInformationGeneRegulation` and `PhiRL` source behaviors recover paper-like punctuated local Phi-r and positive replication association on the twelve existing GARD runs, and do those patterns survive when the same implementation is refitted using past-only prefixes?

## Inputs

The audit used only the twelve immutable S12 baseline trajectories (`baseline_observations.parquet` and `baseline_trajectory_events.parquet`), historical S08/S12 replicator labels, matrices/provenance, the additive-0.5 dropped-component 99-dimensional CLR substrate, and the original paper. No new GARD or intervention trajectory was generated. S10, S11, S11R, and S12 directories were verified before and after execution against their frozen aggregate hashes.

Pinned public sources were `pigozzif/IntegratedInformationGeneRegulation@7c1c22fe39f539d4a453135476f1f0dd5a6b45f7` and `pigozzif/PhiRL@a6d1d0d18c7551302724b7158c6ccdc4d3a33373`; PhiRL regularization commit `9030b598f436cd23c39a3c3fc312ff79c79fb2ad` was verified as an ancestor. No license file was detected in either pinned tree. The public code is not the unavailable GARD implementation.

## Detailed methods

Counts were closed with delta 0.5, CLR transformed, and original component 100 removed. All materialized initial, post-molecular-event, and selected post-fission observations remained in order. IIGR used source z-scoring, global-signal regression, lag-one residualization, a second z-score, bidirectional lag-one Gaussian MI, the source unnormalized Fiedler sign split with a `1e-6` graph floor, partition averages, and corrected all-atom `local_phi_r`. PhiRL removed dimensions with standard deviation at or below `1e-8`, z-scored, used its fast MI/Fiedler pipeline, and used trace-scaled covariance regularization with epsilon `1e-6`. CPU float64 was authoritative.

Full mode fitted preprocessing, partition, means, and covariances once to the completed trajectory and labeled every value `RETROSPECTIVE_FULL_TRAJECTORY_LOCAL`. Prefix mode refitted the identical source behavior independently at each post-fission point with at least 256 preceding molecular transitions and retained only the endpoint value. Current-generation Spearman association was the preregistered primary estimand; next-generation association was secondary. Trajectory bootstrap and within-trajectory circular-shift nulls each used 4,096 replicates. Completed-fit and prefix values were also compared at shared fissions and over a direct first-quarter refit.

## Commands

```text
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -I scripts/e01/convert_s12b_phi_lattice.py ...
PYTHONPATH=src pytest -q tests/e01/test_pigozzi_source_audit.py tests/e01/test_s12b_preregistration.py
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src python scripts/e01/freeze_s12b_preregistration.py
git commit ... && git push origin eidosoma/groups/42
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src python scripts/e01/run_s12b_source_audit.py
```

## Dependencies and precision

Python `{runtime.get('pythonVersion')}`, NumPy `{runtime.get('numpyVersion')}`, SciPy `{runtime.get('scipyVersion')}`, NetworkX `{runtime.get('networkxVersion')}`, pandas `{runtime.get('pandasVersion')}`, and PyArrow `{runtime.get('pyarrowVersion')}` were used. BLAS/OpenMP thread counts were one, six source-analysis workers were used after the single-trajectory benchmark, and no GPU computation was used.

## Source equivalence validation

All source-equivalence rows passed: `{bool(len(equivalence) and equivalence['passed'].all())}`. Maximum MI difference was `{eq_max_mi}` (gate `1e-10`) and maximum local Phi-r difference was `{eq_max_phi}` (gate `1e-9`). Retained variables, partitions up to side exchange, partition averages, original/wrapper replay, and singular/constant status gates are recorded row by row in `source_equivalence_results.csv`. Raw pickle loading occurred only in isolated disposable equivalence/conversion processes; the GARD audit consumed `safe_phi_lattice.json`.

## Results

| Implementation | Full finite coverage | Full median rho | Prefix coverage | Prefix median current rho | Full coherence | Prefix candidate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
{results_table}

### Future-dependence audit

{future_text}

### Spike and temporal diagnostics

{spike_text}

The complete positive/negative 3-sigma, robust-MAD, aggregate-trend, Ljung–Box, full-versus-prefix, first-quarter, partition-ARI, sign, rank, and replication-association outputs remain in the machine-readable tables. `synergy + downward causation` is retained only as the source-named `emergence` diagnostic and did not select a classification.

## Decision and interpretation

The frozen decision is **`{decision['classification']}`**. S12 remains unchanged and is not rescued, overturned, or substituted. S13 remains `BLOCKED_PENDING_S12B_HUMAN_REVIEW`; no intervention authorization follows from this audit.

## Validation

{validation_result}. Runtime projection and observed use are in `runtime_manifest.json`; exact seed replay, suffix sentinels, source identities, pre/post immutability, complete status-bearing rows, required files, hashes, and storage are recorded in the corresponding artifacts. The benchmark projection was `{runtime.get('benchmarkProjection')}`.

## Provenance

- Pre-outcome design Git commit: `{runtime.get('designCommit')}`; remote commit: `{runtime.get('remoteDesignCommit')}`.
- Source snapshots, file SHA-256s, Git blobs/trees, regularization ancestry, safe-lattice opcode audit, and license note: `source_snapshot_manifest.json` and `source_audit.md`.
- S12 input hashes and S10–S12 pre/post aggregate identities: `immutable_input_audit.json`.
- Domain-separated preprocessing, Fiedler, bootstrap, shuffle, and suffix-test seeds derive from the frozen S12B root identity; row-level preprocessing and partition seeds are retained in the Parquet outputs.

## Caveats, blockers, failed assumptions, and limitations

{caveat_text}

Full-mode local values use future observations in their fitted preprocessing, Fiedler partition, means, and covariances and are retrospective descriptions only. Prefix values begin only after 256 preceding transitions and are not the paper's fixed window or MLP experiment. The frozen delta, component drop, observation stream, explicit Fiedler RNG, and historical labels are reconstruction choices, not identified unpublished-author defaults. Public source similarity cannot establish causal control, early warning, exact Figures 2–4, or exact GARD implementation identity.

## Recommended next action

{recommendation} Stop here for mandatory human review; do not begin S13, interventions, another repair, new simulations, MLP/RL work, gene-regulatory experiments, or BioModels downloads.
"""


def finalize_artifacts(
    *,
    config: dict[str, Any],
    success: bool,
    completion_status: str,
    outcome_classification: str,
    decision: dict[str, Any],
    validation_result: str,
    caveats: list[str],
    recommendation: str,
    runtime: dict[str, Any],
    equivalence: pd.DataFrame,
    retrospective: dict[str, Any] | None,
    prospective: dict[str, Any] | None,
    future_rows: list[dict[str, Any]] | None,
    spike_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    ensure_empty_required_tables()
    required = config["requiredArtifacts"]["files"] + config["requiredArtifacts"]["figures"]
    artifacts_written = sorted(required)
    report = report_markdown(
        success=success,
        completion_status=completion_status,
        outcome_classification=outcome_classification,
        decision=decision,
        validation_result=validation_result,
        caveats=caveats,
        recommendation=recommendation,
        runtime=runtime,
        equivalence=equivalence,
        retrospective=retrospective,
        prospective=prospective,
        future_rows=future_rows,
        spike_rows=spike_rows,
        artifacts_written=artifacts_written,
    )
    (STEP_ROOT / "S12B_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    shutil.copyfile(STEP_ROOT / "S12B_FULL_RESULTS.md", STEP_ROOT / "research_step_full_results.md")
    status = {
        "researchStepId": "S12B",
        "stepNumber": 12,
        "stepSuffix": "B",
        "success": success,
        "status": completion_status,
        "artifactsWritten": artifacts_written,
        "validationResult": validation_result,
        "outcomeClassification": outcome_classification,
        "decisionClassification": decision["classification"],
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommendation,
        "s13Status": "BLOCKED_PENDING_S12B_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "status.json", status)
    missing = [item for item in required if not (STEP_ROOT / item).is_file() and item != "artifact_manifest.json"]
    if missing:
        raise RuntimeError(f"missing required S12B artifacts before manifest: {missing}")
    paths = sorted(path for path in STEP_ROOT.rglob("*") if path.is_file() and path.name != "artifact_manifest.json")
    records = [
        {
            "path": path.relative_to(STEP_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    body = "".join(f"{item['sha256']}  {item['path']}\n" for item in records)
    manifest = {
        "schema": "eidosoma.e01.s12b_artifact_manifest.v1",
        "researchStepId": "S12B",
        "preregistrationVersion": VERSION,
        "manifestExcludesItself": True,
        "files": records,
        "fileCountExcludingManifest": len(records),
        "byteCountExcludingManifest": sum(item["bytes"] for item in records),
        "aggregateSha256ExcludingManifest": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "requiredArtifactCountIncludingManifest": len(required),
        "requiredArtifactsPresent": not missing and len(records) + 1 == len(required),
        "unexpectedArtifacts": sorted({item["path"] for item in records} - (set(required) - {"artifact_manifest.json"})),
        "reportByteExactCopy": sha256_file(STEP_ROOT / "S12B_FULL_RESULTS.md") == sha256_file(STEP_ROOT / "research_step_full_results.md"),
        "artifactBytesWithin10GiB": sum(item["bytes"] for item in records) <= config["runtimeAndStorage"]["hardNewArtifactBytes"],
    }
    manifest["success"] = manifest["requiredArtifactsPresent"] and not manifest["unexpectedArtifacts"] and manifest["reportByteExactCopy"] and manifest["artifactBytesWithin10GiB"]
    write_json(STEP_ROOT / "artifact_manifest.json", manifest)
    return {"status": status, "manifest": manifest}


def failure_decision(reason: str) -> dict[str, Any]:
    return {
        "schema": "eidosoma.e01.s12b_classification.v1",
        "researchStepId": "S12B",
        "preregistrationVersion": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "sourceRelationship": "SOURCE_INFORMED_RECONSTRUCTION",
        "classification": "SOURCE_RECONSTRUCTION_FAILED",
        "failureReason": reason,
        "fullRetrospectiveCoherence": {item.value: False for item in SourceImplementation},
        "prospectiveCandidate": {item.value: False for item in SourceImplementation},
        "significantOppositeDirection": {item.value: False for item in SourceImplementation},
        "s12Status": "UNCHANGED_AND_NOT_SUBSTITUTED",
        "s13Status": "BLOCKED_PENDING_S12B_HUMAN_REVIEW",
        "automaticS13Authorized": False,
        "interventionAuthorized": False,
        "recommendedNextAction": "Return for human review. Do not repair after outcomes, do not begin S13, and preserve the failed source audit.",
    }


def verify_design_commit() -> dict[str, Any]:
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    status = git_output("status", "--short")
    remote_line = subprocess.run(
        ["git", "ls-remote", "origin", "refs/heads/eidosoma/groups/42"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote = remote_line.split()[0] if remote_line else None
    committed_config = subprocess.run(
        ["git", "show", f"{head}:configs/e01/s12b_pigozzi_source_audit_preregistration.yaml"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    committed_hash = hashlib.sha256(committed_config).hexdigest()
    actual_hash = sha256_file(CONFIG)
    passed = branch == "eidosoma/groups/42" and not status and remote == head and committed_hash == actual_hash
    return {"branch": branch, "head": head, "remote": remote, "workingTreeStatus": status, "committedConfigSha256": committed_hash, "workingConfigSha256": actual_hash, "passed": passed}


def record_design_freeze(design: dict[str, Any]) -> None:
    path = STEP_ROOT / "preregistration_record.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["designFreeze"] = {
        "status": "COMMITTED_AND_PUSHED_BEFORE_S12B_SCIENTIFIC_OUTCOMES",
        "branch": design["branch"],
        "commit": design["head"],
        "remoteCommit": design["remote"],
        "configSha256": design["workingConfigSha256"],
        "workingTreeClean": design["workingTreeStatus"] == "",
    }
    write_json(path, payload)


def runtime_payload(
    *,
    start_utc: str,
    wall_seconds: float,
    design: dict[str, Any],
    benchmark: dict[str, Any] | None,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    command: list[str],
    stop_reason: str | None,
) -> dict[str, Any]:
    worker_cpu = float(sum(item.get("cpuSeconds", 0.0) for item in records))
    worker_wall = float(sum(item.get("wallSeconds", 0.0) for item in records))
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        gpu = []
    payload = {
        "schema": "eidosoma.e01.s12b_runtime_manifest.v1",
        "researchStepId": "S12B",
        "preregistrationVersion": VERSION,
        "startUtc": start_utc,
        "endUtc": pd.Timestamp.now(tz="UTC").isoformat(),
        "wallSeconds": wall_seconds,
        "workerCpuSeconds": worker_cpu,
        "workerCpuHours": worker_cpu / 3600.0,
        "summedWorkerWallSeconds": worker_wall,
        "gpuHours": 0.0,
        "gpuUsed": False,
        "visibleGpus": gpu,
        "designCommit": design.get("head"),
        "remoteDesignCommit": design.get("remote"),
        "branch": design.get("branch"),
        "command": command,
        "pythonVersion": platform.python_version(),
        "pythonFull": sys.version,
        "platform": platform.platform(),
        "cpuCountVisible": os.cpu_count(),
        "numpyVersion": np.__version__,
        "scipyVersion": scipy.__version__,
        "networkxVersion": nx.__version__,
        "pandasVersion": pd.__version__,
        "pyarrowVersion": pyarrow.__version__,
        "precisionPolicy": "CPU_FLOAT64_AUTHORITATIVE",
        "sourceAnalysisWorkers": 6,
        "statisticsWorkers": 1,
        "orchestrationCores": 1,
        "threadEnvironment": {name: os.environ.get(name) for name in config["runtimeAndStorage"]["threadEnvironment"]},
        "benchmark": benchmark,
        "benchmarkProjection": benchmark.get("projection") if benchmark else None,
        "trajectoryRuntimeRecords": records,
        "hardCeilings": {key: config["runtimeAndStorage"][key] for key in ("hardCpuHours", "hardGpuHours", "hardWallHours", "hardNewArtifactBytes")},
        "stopReason": stop_reason,
    }
    return payload


def close_failure(
    *,
    reason: str,
    config: dict[str, Any],
    preflight: dict[str, Any],
    design: dict[str, Any],
    equivalence: pd.DataFrame,
    started_wall: float,
    start_utc: str,
    benchmark: dict[str, Any] | None,
    records: list[dict[str, Any]],
    existing_failures: list[dict[str, Any]] | None = None,
) -> None:
    if records:
        full, prefix, partitions, diagnostics, worker_failures = collate_trajectory_results(records)
        full, prefix = attach_historical_labels(full, prefix)
        write_parquet(STEP_ROOT / "full_trajectory_local_values.parquet", full.sort_values(["implementationId", "matrixIndex", "rawObservationIndex"]))
        write_parquet(STEP_ROOT / "prefix_endpoint_values.parquet", prefix.sort_values(["implementationId", "matrixIndex", "rawObservationIndex"]))
        write_parquet(STEP_ROOT / "partition_history.parquet", partitions.sort_values(["implementationId", "matrixIndex", "fitKind", "endpointObservationIndex"]))
        write_parquet(STEP_ROOT / "source_diagnostic_outputs.parquet", diagnostics.sort_values(["implementationId", "matrixIndex", "diagnosticType", "rawObservationIndex"]))
    else:
        worker_failures = []
    ensure_empty_required_tables()
    failure_rows = list(existing_failures or []) + worker_failures
    failure_rows.append({"failureId": "S12B-FATAL-STOP", "stage": "bounded_audit", "implementationId": None, "trajectoryId": None, "observationIndex": None, "status": "SOURCE_RECONSTRUCTION_FAILED", "reason": reason, "fatal": True})
    write_csv(STEP_ROOT / "failure_ledger.csv", failure_rows, columns=["failureId", "stage", "implementationId", "trajectoryId", "observationIndex", "status", "reason", "fatal"])
    decision = failure_decision(reason)
    write_json(STEP_ROOT / "classification.json", decision)
    create_placeholder_figures(f"S12B stopped at a mandatory preregistered gate:\n{reason}\nNo S13 authorization.")
    sys.path.insert(0, str(REPO / "scripts/e01"))
    import freeze_s12b_preregistration as freezer
    postflight = freezer.validate_preregistration(require_no_outcomes=False)
    write_immutable_post_audit(preflight, postflight)
    runtime = runtime_payload(start_utc=start_utc, wall_seconds=time.perf_counter() - started_wall, design=design, benchmark=benchmark, records=records, config=config, command=sys.argv, stop_reason=reason)
    write_json(STEP_ROOT / "runtime_manifest.json", runtime)
    finalize_artifacts(
        config=config,
        success=False,
        completion_status="STOPPED_AT_PREREGISTERED_GATE",
        outcome_classification="constraining/contradictory",
        decision=decision,
        validation_result=f"FAIL CLOSED — {reason}",
        caveats=[reason, "No post-outcome repair or scope reduction was attempted.", "S10–S12 and the public source snapshots remain immutable."],
        recommendation=decision["recommendedNextAction"],
        runtime=runtime,
        equivalence=equivalence,
        retrospective=None,
        prospective=None,
        future_rows=None,
        spike_rows=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers != 6:
        raise SystemExit("S12B preregistration requires exactly six source-analysis workers")
    started_wall = time.perf_counter()
    start_utc = pd.Timestamp.now(tz="UTC").isoformat()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for name, value in config["runtimeAndStorage"]["threadEnvironment"].items():
        if os.environ.get(name) != value:
            raise SystemExit(f"{name} must be exactly {value} before process start")
    if any((STEP_ROOT / name).exists() for name in ("classification.json", "full_trajectory_local_values.parquet", "prefix_endpoint_values.parquet")):
        raise SystemExit("S12B scientific outputs already exist; immutable completed/failed audit will not be overwritten")
    sys.path.insert(0, str(REPO / "scripts/e01"))
    import freeze_s12b_preregistration as freezer

    equivalence = pd.DataFrame(columns=["fixtureId", "implementationId", "passed", "miMaxAbsDifference", "localPhiRMaxAbsDifference"])
    if not (STEP_ROOT / "source_equivalence_results.csv").exists():
        write_csv(STEP_ROOT / "source_equivalence_results.csv", [], columns=list(equivalence.columns))
    try:
        preflight = freezer.validate_preregistration(require_no_outcomes=False)
    except Exception as exc:  # noqa: BLE001 - every preflight exception must close the audit.
        preflight = {"success": False, "errors": [f"{type(exc).__name__}:{exc}"], "priorArtifacts": {}, "priorRepository": {}, "frozenInputs": [], "sources": {}}
    design: dict[str, Any] = {"branch": None, "head": None, "remote": None, "workingTreeStatus": None, "passed": False}
    if not preflight["success"]:
        reason = "PINNED_SOURCE_OR_IMMUTABLE_INPUT_PREFLIGHT_FAILED:" + ";".join(preflight.get("errors", []))
        close_failure(reason=reason, config=config, preflight=preflight, design=design, equivalence=equivalence, started_wall=started_wall, start_utc=start_utc, benchmark=None, records=[])
        print(json.dumps({"success": False, "status": "STOPPED_AT_PREREGISTERED_GATE", "reason": reason}, sort_keys=True))
        return
    design = verify_design_commit()
    if not design["passed"]:
        reason = "PREOUTCOME_DESIGN_NOT_CLEAN_COMMITTED_AND_PUSHED"
        close_failure(reason=reason, config=config, preflight=preflight, design=design, equivalence=equivalence, started_wall=started_wall, start_utc=start_utc, benchmark=None, records=[])
        print(json.dumps({"success": False, "status": "STOPPED_AT_PREREGISTERED_GATE", "reason": reason}, sort_keys=True))
        return
    record_design_freeze(design)

    equivalence, equivalence_passed = source_equivalence(config)
    if not equivalence_passed:
        reason = "SOURCE_EQUIVALENCE_FAILED"
        close_failure(reason=reason, config=config, preflight=preflight, design=design, equivalence=equivalence, started_wall=started_wall, start_utc=start_utc, benchmark=None, records=[])
        print(json.dumps({"success": False, "status": "STOPPED_AT_PREREGISTERED_GATE", "reason": reason}, sort_keys=True))
        return

    input_records = prepare_trajectory_inputs()
    if RESULT_CACHE.exists():
        shutil.rmtree(RESULT_CACHE)
    RESULT_CACHE.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    benchmark_record = process_trajectory(input_records[0]["path"], str(CONFIG))
    records.append(benchmark_record)
    benchmark_bytes = sum(path.stat().st_size for path in Path(benchmark_record["resultDirectory"]).rglob("*") if path.is_file())
    projection = {
        "cpuHours": benchmark_record["cpuSeconds"] * 12.0 * 1.25 / 3600.0,
        "wallHours": benchmark_record["wallSeconds"] * 12.0 * 1.25 / 3600.0,
        "artifactBytes": math.ceil(benchmark_bytes * 12.0 * 1.25),
        "formula": "observed_complete_matrix0_times_12_plus_25_percent_reserve",
    }
    benchmark = {
        "matrixIndex": 0,
        "trajectoryId": benchmark_record["trajectoryId"],
        "observedWallSeconds": benchmark_record["wallSeconds"],
        "observedCpuSeconds": benchmark_record["cpuSeconds"],
        "observedResultBytes": benchmark_bytes,
        "projection": projection,
        "cpuGatePassed": projection["cpuHours"] <= config["runtimeAndStorage"]["hardCpuHours"],
        "wallGatePassed": projection["wallHours"] <= config["runtimeAndStorage"]["hardWallHours"],
        "storageGatePassed": projection["artifactBytes"] <= config["runtimeAndStorage"]["hardNewArtifactBytes"],
    }
    benchmark["passed"] = benchmark["cpuGatePassed"] and benchmark["wallGatePassed"] and benchmark["storageGatePassed"] and benchmark_record["fullReplayPassed"] and benchmark_record["prefixReplayPassed"] and benchmark_record["futureSuffixPassed"]
    if not benchmark["passed"]:
        reason = "BENCHMARK_PROJECTION_OR_REPLAY_OR_SUFFIX_GATE_FAILED"
        close_failure(reason=reason, config=config, preflight=preflight, design=design, equivalence=equivalence, started_wall=started_wall, start_utc=start_utc, benchmark=benchmark, records=records)
        print(json.dumps({"success": False, "status": "STOPPED_AT_PREREGISTERED_GATE", "reason": reason, "benchmark": benchmark}, sort_keys=True))
        return

    worker_error: str | None = None
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_trajectory, item["path"], str(CONFIG)): item for item in input_records[1:]}
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as exc:  # noqa: BLE001 - worker failures are status-bearing stop conditions.
                item = futures[future]
                worker_error = f"WORKER_FAILED:{item['trajectoryId']}:{type(exc).__name__}:{exc}"
                for pending in futures:
                    pending.cancel()
                break
    if worker_error is not None or len(records) != 12:
        reason = worker_error or f"INCOMPLETE_TRAJECTORY_WORKER_SET:{len(records)}_OF_12"
        close_failure(reason=reason, config=config, preflight=preflight, design=design, equivalence=equivalence, started_wall=started_wall, start_utc=start_utc, benchmark=benchmark, records=records)
        print(json.dumps({"success": False, "status": "STOPPED_AT_PREREGISTERED_GATE", "reason": reason}, sort_keys=True))
        return

    full, prefix, partitions, diagnostics, worker_failures = collate_trajectory_results(records)
    full, prefix = attach_historical_labels(full, prefix)
    full = full.sort_values(["implementationId", "matrixIndex", "rawObservationIndex"]).reset_index(drop=True)
    prefix = prefix.sort_values(["implementationId", "matrixIndex", "rawObservationIndex"]).reset_index(drop=True)
    partitions = partitions.sort_values(["implementationId", "matrixIndex", "fitKind", "endpointObservationIndex"]).reset_index(drop=True)
    diagnostics = diagnostics.sort_values(["implementationId", "matrixIndex", "diagnosticType", "rawObservationIndex", "variant"], na_position="last").reset_index(drop=True)
    write_parquet(STEP_ROOT / "full_trajectory_local_values.parquet", full)
    write_parquet(STEP_ROOT / "prefix_endpoint_values.parquet", prefix)
    write_parquet(STEP_ROOT / "partition_history.parquet", partitions)
    write_parquet(STEP_ROOT / "source_diagnostic_outputs.parquet", diagnostics)

    replay_passed = all(item["fullReplayPassed"] and item["prefixReplayPassed"] for item in records)
    suffix_passed = all(item["futureSuffixPassed"] for item in records)
    stop_reasons: list[str] = []
    coverage_audit: dict[str, Any] = {}
    for implementation in [item.value for item in SourceImplementation]:
        full_branch = full[full["implementationId"] == implementation]
        prefix_branch = prefix[(prefix["implementationId"] == implementation) & (prefix["molecularStep"] >= 256)]
        full_nonfinite = float((full_branch["status"] != "ELIGIBLE").mean())
        prefix_ineligible = float((prefix_branch["status"] != "ELIGIBLE").mean())
        coverage_audit[implementation] = {"expectedFullOutputs": len(full_branch), "fullNonfiniteOrSuppressedFraction": full_nonfinite, "expectedPrefixEndpointsAfterBoundary": len(prefix_branch), "prefixIneligibleFraction": prefix_ineligible}
        if full_nonfinite > 0.20:
            stop_reasons.append(f"FULL_NONFINITE_FRACTION_GREATER_THAN_0.20:{implementation}:{full_nonfinite}")
        if prefix_ineligible > 0.50:
            stop_reasons.append(f"PREFIX_INELIGIBLE_FRACTION_GREATER_THAN_0.50:{implementation}:{prefix_ineligible}")
    if not replay_passed:
        stop_reasons.append("EXACT_SOURCE_WRAPPER_REPLAY_FAILED")
    if not suffix_passed:
        stop_reasons.append("FUTURE_SUFFIX_INVARIANCE_FAILED")
    final_design = verify_design_commit()
    if not final_design["passed"] or final_design["head"] != design["head"]:
        stop_reasons.append("UNDOCUMENTED_POST_OUTCOME_CODE_CHANGE_OR_REMOTE_IDENTITY_CHANGE")
    if stop_reasons:
        reason = ";".join(stop_reasons)
        close_failure(reason=reason, config=config, preflight=preflight, design=design, equivalence=equivalence, started_wall=started_wall, start_utc=start_utc, benchmark=benchmark, records=records, existing_failures=worker_failures)
        print(json.dumps({"success": False, "status": "STOPPED_AT_PREREGISTERED_GATE", "reason": reason, "coverage": coverage_audit}, sort_keys=True))
        return

    retrospective_rows, retrospective = analyze_retrospective(full, config)
    prospective_rows, prospective = analyze_prospective(prefix, config)
    spike_rows, thresholds = analyze_spikes(full, prefix)
    future_rows, endpoint_ari = analyze_future_dependence(full, prefix, partitions, diagnostics, thresholds)
    write_csv(STEP_ROOT / "retrospective_associations.csv", retrospective_rows)
    write_csv(STEP_ROOT / "prospective_associations.csv", prospective_rows)
    write_csv(STEP_ROOT / "spike_analysis.csv", spike_rows)
    write_csv(STEP_ROOT / "future_dependence_results.csv", future_rows)
    decision = decide_classification(retrospective, prospective, replay_passed=replay_passed, suffix_passed=suffix_passed)
    write_json(STEP_ROOT / "classification.json", decision)

    failure_rows = list(worker_failures)
    for table_name, frame in (("full", full), ("prefix", prefix)):
        for (implementation, status), group in frame.groupby(["implementationId", "status"], dropna=False, sort=True):
            if status != "ELIGIBLE":
                failure_rows.append({"failureId": f"STATUS-SUMMARY-{table_name}-{implementation}-{status}", "stage": table_name, "implementationId": implementation, "trajectoryId": None, "observationIndex": None, "status": status, "reason": f"status_bearing_row_count={len(group)}", "fatal": False})
    write_csv(STEP_ROOT / "failure_ledger.csv", failure_rows, columns=["failureId", "stage", "implementationId", "trajectoryId", "observationIndex", "status", "reason", "fatal"])
    create_figures(full, prefix, retrospective_rows, prospective_rows, spike_rows, endpoint_ari, decision, config)

    postflight = freezer.validate_preregistration(require_no_outcomes=False)
    write_immutable_post_audit(preflight, postflight)
    if not postflight["success"]:
        reason = "POST_OUTCOME_IMMUTABILITY_OR_SOURCE_IDENTITY_FAILED:" + ";".join(postflight["errors"])
        close_failure(reason=reason, config=config, preflight=preflight, design=design, equivalence=equivalence, started_wall=started_wall, start_utc=start_utc, benchmark=benchmark, records=records, existing_failures=failure_rows)
        print(json.dumps({"success": False, "status": "STOPPED_AT_PREREGISTERED_GATE", "reason": reason}, sort_keys=True))
        return

    runtime = runtime_payload(start_utc=start_utc, wall_seconds=time.perf_counter() - started_wall, design=design, benchmark=benchmark, records=sorted(records, key=lambda item: item["matrixIndex"]), config=config, command=sys.argv, stop_reason=None)
    runtime["coverageAudit"] = coverage_audit
    runtime["observedTrajectoryCount"] = len(records)
    runtime["observedArtifactBytesBeforeReports"] = sum(path.stat().st_size for path in STEP_ROOT.rglob("*") if path.is_file())
    runtime["runtimeCeilingsPassed"] = runtime["workerCpuHours"] <= config["runtimeAndStorage"]["hardCpuHours"] and runtime["wallSeconds"] / 3600.0 <= config["runtimeAndStorage"]["hardWallHours"]
    write_json(STEP_ROOT / "runtime_manifest.json", runtime)
    if not runtime["runtimeCeilingsPassed"]:
        reason = "OBSERVED_RUNTIME_EXCEEDED_HARD_CEILING"
        close_failure(reason=reason, config=config, preflight=preflight, design=design, equivalence=equivalence, started_wall=started_wall, start_utc=start_utc, benchmark=benchmark, records=records, existing_failures=failure_rows)
        print(json.dumps({"success": False, "status": "STOPPED_AT_PREREGISTERED_GATE", "reason": reason}, sort_keys=True))
        return

    if decision["classification"] == "SOURCE_FAMILY_PROSPECTIVE_CANDIDATE":
        outcome_classification = "supportive"
    elif decision["classification"] == "SOURCE_FAMILY_NOT_SUPPORTED":
        outcome_classification = "constraining/contradictory"
    else:
        outcome_classification = "constraining/contradictory"
    caveats = [
        "The public source family is not the unavailable GARD author implementation and has no author-primary or paper-primary identity.",
        "Completed-trajectory values are retrospective and can use future observations in preprocessing, partitioning, means, and covariances.",
        "The additive-0.5 zero policy, dropped component 100, observation stream, deterministic Fiedler seed, and historical labels are frozen reconstruction choices.",
        "S12B neither rescues nor overturns S12 and authorizes no intervention or automatic S13.",
        "No LICENSE or COPYING file was detected at either pinned public commit; raw source was not redistributed as an artifact.",
    ]
    validation_result = "PASS — source equivalence, exact replay, future-suffix invariance, finite/prefix coverage, runtime/storage, pre/post immutability, status completeness, and required artifact gates passed"
    final = finalize_artifacts(
        config=config,
        success=True,
        completion_status="COMPLETED_STOPPED_FOR_MANDATORY_HUMAN_REVIEW",
        outcome_classification=outcome_classification,
        decision=decision,
        validation_result=validation_result,
        caveats=caveats,
        recommendation=decision["recommendedNextAction"],
        runtime=runtime,
        equivalence=equivalence,
        retrospective=retrospective,
        prospective=prospective,
        future_rows=future_rows,
        spike_rows=spike_rows,
    )
    if not final["manifest"]["success"]:
        raise RuntimeError(f"final artifact manifest failed: {final['manifest']}")
    print(json.dumps({"success": True, "status": "COMPLETED_STOPPED_FOR_MANDATORY_HUMAN_REVIEW", "classification": decision["classification"], "artifactAggregateSha256": final["manifest"]["aggregateSha256ExcludingManifest"], "runtimeSeconds": runtime["wallSeconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
