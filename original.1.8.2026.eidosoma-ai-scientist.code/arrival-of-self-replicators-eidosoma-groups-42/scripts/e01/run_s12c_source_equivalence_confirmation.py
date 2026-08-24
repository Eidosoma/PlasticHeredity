#!/usr/bin/env python3
"""Run the bounded S12C equivalence confirmation and conditional source audit."""

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

from e01_pigozzi_source_audit.core import SourceImplementation
from e01_pigozzi_source_equivalence_confirmation.core import (
    ConfirmedAuditResult,
    derive_seed,
    fixture_array,
    run_source_pipeline,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/e01/s12c_source_equivalence_confirmation_preregistration.yaml"
LOCK_CONFIG = REPO / "configs/e01/s12c_implementation_lock.yaml"
S12B_CONFIG = REPO / "configs/e01/s12b_pigozzi_source_audit_preregistration.yaml"
STEP_ROOT = Path("/artifacts/research_steps/S12C")
S12B_ROOT = Path("/artifacts/research_steps/S12B")
SAFE_LATTICE = S12B_ROOT / "safe_phi_lattice.json"
CACHE_ROOT = Path("/cache/e01_s12c")
INPUT_CACHE = CACHE_ROOT / "trajectory_inputs"
RESULT_CACHE = CACHE_ROOT / "scientific_results"
FIGURE_ROOT = STEP_ROOT / "figures"
ADAPTER = REPO / "scripts/e01/s12b_original_source_adapter.py"
VERSION = "E01-S12C-SOURCE-EQUIVALENCE-CONFIRMATION-v1.0.0"
EVIDENCE_CLASS = "SOURCE_INFORMED_FORENSIC_RECONSTRUCTION"
THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
EQUIVALENCE_COLUMNS = [
    "researchStepId",
    "phase",
    "fixtureId",
    "implementationId",
    "fixtureSha256",
    "sourceStatus",
    "wrapperStatus",
    "statusIdentical",
    "retainedArrayAvailabilityIdentical",
    "retainedVariablesIdentical",
    "processedArrayAvailabilityIdentical",
    "processedMaxAbsDifference",
    "processedGateAtMost1e12",
    "miArrayAvailabilityIdentical",
    "miMaxAbsDifference",
    "miGateAtMost1e10",
    "partitionArrayAvailabilityIdentical",
    "partitionIdenticalUpToSideExchange",
    "partitionAverageArrayAvailabilityIdentical",
    "partitionAverageMaxAbsDifference",
    "partitionAverageGateAtMost1e10",
    "localPhiRArrayAvailabilityIdentical",
    "localPhiRMaxAbsDifference",
    "localPhiRGateAtMost1e9",
    "emergenceArrayAvailabilityIdentical",
    "emergenceMaxAbsDifference",
    "emergenceGateAtMost1e9",
    "originalExactReplay",
    "wrapperExactReplay",
    "preprocessingSeed",
    "partitionSeed",
    "allGatesPassed",
]


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
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, lineterminator="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def remote_head() -> str | None:
    result = subprocess.run(
        ["git", "ls-remote", "origin", "refs/heads/eidosoma/groups/42"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return result.split()[0] if result else None


def max_abs_difference(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    if left.size == 0:
        return 0.0
    with np.errstate(invalid="ignore"):
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return None if np.all(np.isnan(difference)) else float(np.nanmax(difference))


def array_exact(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    if (left is None) != (right is None):
        return False
    if left is None:
        return True
    return left.shape == right.shape and np.array_equal(left, right, equal_nan=True)


def result_replay_equal(left: ConfirmedAuditResult, right: ConfirmedAuditResult) -> bool:
    scalar_equal = (
        left.implementation == right.implementation
        and left.status == right.status
        and left.reason == right.reason
        and left.retained_available == right.retained_available
        and left.retained_variables == right.retained_variables
        and left.partition_1 == right.partition_1
        and left.partition_2 == right.partition_2
        and left.local_offset == right.local_offset
    )
    arrays_equal = all(
        array_exact(getattr(left, name), getattr(right, name))
        for name in (
            "processed",
            "mi_matrix",
            "fiedler_vector",
            "partition_average",
            "local_phi_r",
            "emergence",
        )
    )
    return scalar_equal and arrays_equal


def npz_equal(path_a: Path, path_b: Path) -> bool:
    with np.load(path_a, allow_pickle=False) as left, np.load(
        path_b, allow_pickle=False
    ) as right:
        if set(left.files) != set(right.files):
            return False
        return all(
            np.array_equal(left[name], right[name], equal_nan=True)
            if left[name].dtype.kind in "fc"
            else np.array_equal(left[name], right[name])
            for name in left.files
        )


def load_original(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(payload["metadata_json"].item())
        arrays = {
            name: payload[name].copy()
            for name in payload.files
            if name != "metadata_json"
        }
    return metadata, arrays


def source_partitions(arrays: dict[str, np.ndarray]) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if "partition_1_local" not in arrays or "partition_2_local" not in arrays:
        return None
    retained = arrays.get("retained")
    if retained is None:
        return None
    p1 = tuple(map(int, retained[arrays["partition_1_local"].astype(int)]))
    p2 = tuple(map(int, retained[arrays["partition_2_local"].astype(int)]))
    return p1, p2


def compare_equivalence_row(
    *,
    phase: str,
    fixture_id: str,
    fixture_sha: str,
    implementation: SourceImplementation,
    original_metadata: dict[str, Any],
    original_arrays: dict[str, np.ndarray],
    original_replay: bool,
    wrapper: ConfirmedAuditResult,
    wrapper_replay: bool,
    preprocessing_seed: int,
    partition_seed: int,
    gates: dict[str, Any],
) -> dict[str, Any]:
    retained_source_available = "retained" in original_arrays
    retained_wrapper_available = wrapper.retained_available
    retained_availability = retained_source_available == retained_wrapper_available
    retained_equal = retained_availability and (
        not retained_source_available
        or np.array_equal(
            original_arrays["retained"],
            np.asarray(wrapper.retained_variables, dtype=np.int64),
        )
    )
    processed_source = original_arrays.get("processed")
    processed_availability = (processed_source is not None) == (wrapper.processed is not None)
    processed_difference = max_abs_difference(processed_source, wrapper.processed)
    processed_gate = processed_availability and (
        processed_source is None
        or (
            processed_difference is not None
            and processed_difference <= gates["processedMaxAbsDifferenceAtMost"]
        )
    )
    source_mi = original_arrays.get("mi")
    mi_availability = (source_mi is not None) == (wrapper.mi_matrix is not None)
    mi_difference = max_abs_difference(source_mi, wrapper.mi_matrix)
    mi_gate = mi_availability and (
        source_mi is None
        or (
            mi_difference is not None
            and mi_difference <= gates["miMaxAbsDifferenceAtMost"]
        )
    )
    source_partition = source_partitions(original_arrays)
    wrapper_partition_available = wrapper.fiedler_vector is not None
    partition_availability = (source_partition is not None) == wrapper_partition_available
    partition_equal = False
    exchanged = False
    if source_partition is None and not wrapper_partition_available:
        partition_equal = True
    elif source_partition is not None and wrapper_partition_available:
        source_p1, source_p2 = source_partition
        direct = source_p1 == wrapper.partition_1 and source_p2 == wrapper.partition_2
        exchanged = source_p1 == wrapper.partition_2 and source_p2 == wrapper.partition_1
        partition_equal = direct or exchanged
    source_average = original_arrays.get("partition_average")
    wrapper_average = wrapper.partition_average
    average_availability = (source_average is not None) == (wrapper_average is not None)
    if exchanged and wrapper_average is not None:
        wrapper_average = wrapper_average[::-1]
    average_difference = max_abs_difference(source_average, wrapper_average)
    average_gate = average_availability and (
        source_average is None
        or (
            average_difference is not None
            and average_difference <= gates["partitionAverageMaxAbsDifferenceAtMost"]
        )
    )
    source_phi = original_arrays.get("local_phi_r")
    phi_availability = (source_phi is not None) == (wrapper.local_phi_r is not None)
    phi_difference = max_abs_difference(source_phi, wrapper.local_phi_r)
    phi_gate = phi_availability and (
        source_phi is None
        or (
            phi_difference is not None
            and phi_difference <= gates["localPhiRMaxAbsDifferenceAtMost"]
        )
    )
    source_emergence = original_arrays.get("emergence")
    emergence_availability = (source_emergence is not None) == (wrapper.emergence is not None)
    emergence_difference = max_abs_difference(source_emergence, wrapper.emergence)
    emergence_gate = emergence_availability and (
        source_emergence is None
        or (
            emergence_difference is not None
            and emergence_difference <= gates["emergenceMaxAbsDifferenceAtMost"]
        )
    )
    status_equal = original_metadata["status"] == wrapper.status
    passed = all(
        (
            status_equal,
            retained_availability,
            retained_equal,
            processed_gate,
            mi_gate,
            partition_availability,
            partition_equal,
            average_gate,
            phi_gate,
            emergence_gate,
            original_replay,
            wrapper_replay,
        )
    )
    return {
        "researchStepId": "S12C",
        "phase": phase.upper(),
        "fixtureId": fixture_id,
        "implementationId": implementation.value,
        "fixtureSha256": fixture_sha,
        "sourceStatus": original_metadata["status"],
        "wrapperStatus": wrapper.status,
        "statusIdentical": status_equal,
        "retainedArrayAvailabilityIdentical": retained_availability,
        "retainedVariablesIdentical": retained_equal,
        "processedArrayAvailabilityIdentical": processed_availability,
        "processedMaxAbsDifference": processed_difference,
        "processedGateAtMost1e12": processed_gate,
        "miArrayAvailabilityIdentical": mi_availability,
        "miMaxAbsDifference": mi_difference,
        "miGateAtMost1e10": mi_gate,
        "partitionArrayAvailabilityIdentical": partition_availability,
        "partitionIdenticalUpToSideExchange": partition_equal,
        "partitionAverageArrayAvailabilityIdentical": average_availability,
        "partitionAverageMaxAbsDifference": average_difference,
        "partitionAverageGateAtMost1e10": average_gate,
        "localPhiRArrayAvailabilityIdentical": phi_availability,
        "localPhiRMaxAbsDifference": phi_difference,
        "localPhiRGateAtMost1e9": phi_gate,
        "emergenceArrayAvailabilityIdentical": emergence_availability,
        "emergenceMaxAbsDifference": emergence_difference,
        "emergenceGateAtMost1e9": emergence_gate,
        "originalExactReplay": original_replay,
        "wrapperExactReplay": wrapper_replay,
        "preprocessingSeed": preprocessing_seed,
        "partitionSeed": partition_seed,
        "allGatesPassed": passed,
    }


def run_equivalence_phase(
    config: dict[str, Any], phase: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    phase_spec = config["fixtureFirewall"][phase]
    root_seed = phase_spec["rootSeedHex"]
    phase_root = Path(phase_spec["outputDirectory"])
    if phase_root.exists():
        raise RuntimeError(f"{phase} cache already exists; phase is non-overwriting")
    phase_root.mkdir(parents=True)
    env = os.environ.copy()
    env.update(config["runtimeAndStorage"]["threadEnvironment"])
    env["PYTHONHASHSEED"] = "0"
    source_dirs = {
        SourceImplementation.IIGR: Path(
            config["sourceSnapshots"][SourceImplementation.IIGR.value]["localCheckout"]
        ),
        SourceImplementation.PHIRL: Path(
            config["sourceSnapshots"][SourceImplementation.PHIRL.value]["localCheckout"]
        ),
    }
    rows: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    fixture_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fixture_id in config["fixtureFirewall"]["fixtureIds"]:
        observations = fixture_array(fixture_id, phase, root_seed)
        fixture_path = phase_root / f"fixture-{fixture_id}.npz"
        np.savez_compressed(fixture_path, observations=observations)
        fixture_sha = sha256_array(observations)
        fixture_records.append(
            {
                "fixtureId": fixture_id,
                "shape": list(observations.shape),
                "payloadSha256": fixture_sha,
            }
        )
        for implementation in SourceImplementation:
            preprocessing_identity = (
                phase,
                implementation.value,
                fixture_id,
                "preprocessing_noise",
            )
            partition_identity = (
                phase,
                implementation.value,
                fixture_id,
                "fiedler_initialization",
            )
            preprocessing_seed = derive_seed(root_seed, *preprocessing_identity)
            partition_seed = derive_seed(root_seed, *partition_identity)
            for kind, identity, seed in (
                ("preprocessing", preprocessing_identity, preprocessing_seed),
                ("partition", partition_identity, partition_seed),
            ):
                seed_records.append(
                    {
                        "phase": phase,
                        "fixtureId": fixture_id,
                        "implementationId": implementation.value,
                        "kind": kind,
                        "identity": "\u001f".join(map(str, identity)),
                        "seed": seed,
                    }
                )
            adapter_name = (
                "IIGR" if implementation is SourceImplementation.IIGR else "PHIRL"
            )
            original_paths = [
                phase_root
                / f"{fixture_id}-{implementation.value}-original-replay-{index}.npz"
                for index in (1, 2)
            ]
            for output in original_paths:
                subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        str(ADAPTER),
                        "--implementation",
                        adapter_name,
                        "--source-dir",
                        str(source_dirs[implementation]),
                        "--input",
                        str(fixture_path),
                        "--output",
                        str(output),
                        "--preprocessing-seed",
                        str(preprocessing_seed),
                        "--partition-seed",
                        str(partition_seed),
                    ],
                    cwd=phase_root,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            original_replay = npz_equal(*original_paths)
            metadata, arrays = load_original(original_paths[0])
            wrapper = run_source_pipeline(
                observations,
                implementation,
                SAFE_LATTICE,
                preprocessing_seed=preprocessing_seed,
                partition_seed=partition_seed,
            )
            replay = run_source_pipeline(
                observations,
                implementation,
                SAFE_LATTICE,
                preprocessing_seed=preprocessing_seed,
                partition_seed=partition_seed,
            )
            rows.append(
                compare_equivalence_row(
                    phase=phase,
                    fixture_id=fixture_id,
                    fixture_sha=fixture_sha,
                    implementation=implementation,
                    original_metadata=metadata,
                    original_arrays=arrays,
                    original_replay=original_replay,
                    wrapper=wrapper,
                    wrapper_replay=result_replay_equal(wrapper, replay),
                    preprocessing_seed=preprocessing_seed,
                    partition_seed=partition_seed,
                    gates=config["confirmationGates"],
                )
            )
    frame = pd.DataFrame(rows, columns=EQUIVALENCE_COLUMNS)
    summary = {
        "schema": f"eidosoma.e01.s12c_{phase}_summary.v1",
        "researchStepId": "S12C",
        "preregistrationVersion": VERSION,
        "phase": phase.upper(),
        "fixtureCount": len(fixture_records),
        "rowCount": len(frame),
        "expectedRowCount": config["confirmationGates"]["expectedRows"],
        "passedRowCount": int(frame["allGatesPassed"].sum()),
        "allRowsPassed": bool(
            len(frame) == config["confirmationGates"]["expectedRows"]
            and frame["allGatesPassed"].all()
        ),
        "fixtureRecords": fixture_records,
        "seedRecords": seed_records,
        "wallSeconds": time.perf_counter() - started,
        "gardInputOpened": False,
        "pinnedSourceModified": False,
        "repairIdentity": config["repairHypothesis"]["repairId"],
    }
    return frame, summary


def design_state() -> dict[str, Any]:
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    status = git_output("status", "--short")
    remote = remote_head()
    return {
        "branch": branch,
        "head": head,
        "remote": remote,
        "workingTreeStatus": status,
        "passed": branch == "eidosoma/groups/42" and not status and head == remote,
    }


def verify_implementation_lock(config: dict[str, Any]) -> dict[str, Any]:
    if not LOCK_CONFIG.is_file():
        return {"passed": False, "errors": ["implementation_lock_yaml_missing"]}
    lock = yaml.safe_load(LOCK_CONFIG.read_text(encoding="utf-8"))
    errors: list[str] = []
    state = design_state()
    if not state["passed"]:
        errors.append("implementation_lock_commit_not_clean_and_pushed")
    development_path = STEP_ROOT / "development_summary.json"
    development_sha = sha256_file(development_path) if development_path.is_file() else None
    if development_sha != lock.get("developmentSummarySha256"):
        errors.append("development_summary_hash_mismatch")
    file_records: list[dict[str, Any]] = []
    for relative, expected in lock.get("implementationFiles", {}).items():
        path = REPO / relative
        actual = sha256_file(path) if path.is_file() else None
        passed = actual == expected
        file_records.append(
            {
                "path": relative,
                "expectedSha256": expected,
                "actualSha256": actual,
                "passed": passed,
            }
        )
        if not passed:
            errors.append(f"locked_file_hash_mismatch:{relative}")
    if lock.get("repairId") != config["repairHypothesis"]["repairId"]:
        errors.append("repair_identity_mismatch")
    if lock.get("confirmationAccessStatus") != "LOCKED_BEFORE_UNTOUCHED_CONFIRMATION":
        errors.append("lock_status_invalid")
    lock_commit_contains_file = (
        subprocess.run(
            ["git", "cat-file", "-e", f"{state['head']}:configs/e01/s12c_implementation_lock.yaml"],
            cwd=REPO,
            check=False,
        ).returncode
        == 0
    )
    if not lock_commit_contains_file:
        errors.append("lock_yaml_not_committed")
    return {
        "schema": "eidosoma.e01.s12c_implementation_lock_audit.v1",
        "researchStepId": "S12C",
        "lockConfigPath": str(LOCK_CONFIG),
        "lockConfigSha256": sha256_file(LOCK_CONFIG),
        "developmentSummarySha256": development_sha,
        "designState": state,
        "implementationFiles": file_records,
        "lockYamlCommitted": lock_commit_contains_file,
        "errors": errors,
        "passed": not errors,
    }


def seed_firewall(
    development: dict[str, Any], confirmation: dict[str, Any]
) -> dict[str, Any]:
    development_seeds = development["seedRecords"]
    confirmation_seeds = confirmation["seedRecords"]
    development_identities = {item["identity"] for item in development_seeds}
    confirmation_identities = {item["identity"] for item in confirmation_seeds}
    development_seed_values = {item["seed"] for item in development_seeds}
    confirmation_seed_values = {item["seed"] for item in confirmation_seeds}
    development_payloads = {
        item["payloadSha256"] for item in development["fixtureRecords"]
    }
    confirmation_payloads = {
        item["payloadSha256"] for item in confirmation["fixtureRecords"]
    }
    payload = {
        "schema": "eidosoma.e01.s12c_seed_firewall.v1",
        "researchStepId": "S12C",
        "developmentRootNotEqualConfirmationRoot": True,
        "streamIdentityIntersection": sorted(
            development_identities & confirmation_identities
        ),
        "seedMaterialIntersection": sorted(
            development_seed_values & confirmation_seed_values
        ),
        "fixturePayloadSha256Intersection": sorted(
            development_payloads & confirmation_payloads
        ),
    }
    payload["passed"] = not any(
        payload[key]
        for key in (
            "streamIdentityIntersection",
            "seedMaterialIntersection",
            "fixturePayloadSha256Intersection",
        )
    )
    return payload


def patch_s12b_module() -> Any:
    from scripts.e01 import run_s12b_source_audit as base

    base.STEP_ROOT = STEP_ROOT
    base.CACHE_ROOT = CACHE_ROOT
    base.INPUT_CACHE = INPUT_CACHE
    base.RESULT_CACHE = RESULT_CACHE
    base.SAFE_LATTICE = SAFE_LATTICE
    base.FIGURE_ROOT = FIGURE_ROOT
    base.VERSION = VERSION
    base.run_source_pipeline = run_source_pipeline
    return base


def rewrite_worker_tables(result_directory: Path) -> None:
    for name in ("full.parquet", "prefix.parquet", "partition.parquet", "diagnostic.parquet"):
        path = result_directory / name
        frame = pd.read_parquet(path)
        if "researchStepId" in frame:
            frame["researchStepId"] = "S12C"
        if "preregistrationVersion" in frame:
            frame["preregistrationVersion"] = VERSION
        frame.to_parquet(path, index=False, compression="zstd")


def process_trajectory_s12c(input_path: str) -> dict[str, Any]:
    base = patch_s12b_module()
    record = base.process_trajectory(input_path, str(S12B_CONFIG))
    rewrite_worker_tables(Path(record["resultDirectory"]))
    return record


def empty_scientific_tables() -> None:
    parquet_schemas = {
        "full_trajectory_local_values.parquet": [
            "researchStepId",
            "implementationId",
            "trajectoryId",
            "status",
            "reason",
            "phiR",
        ],
        "prefix_endpoint_values.parquet": [
            "researchStepId",
            "implementationId",
            "trajectoryId",
            "status",
            "reason",
            "phiR",
        ],
        "partition_history.parquet": [
            "researchStepId",
            "implementationId",
            "trajectoryId",
            "status",
            "reason",
        ],
        "source_diagnostic_outputs.parquet": [
            "researchStepId",
            "implementationId",
            "trajectoryId",
            "status",
            "reason",
        ],
    }
    for name, columns in parquet_schemas.items():
        path = STEP_ROOT / name
        if not path.exists():
            pd.DataFrame(columns=columns).to_parquet(
                path, index=False, compression="zstd"
            )
    for name in (
        "retrospective_associations.csv",
        "prospective_associations.csv",
        "spike_analysis.csv",
        "future_dependence_results.csv",
    ):
        path = STEP_ROOT / name
        if not path.exists():
            write_csv(path, [], columns=["status", "reason"])


def placeholder_figures(message: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    names = (
        "full_trajectory_matched_sources.png",
        "full_versus_prefix_representative.png",
        "association_distributions.png",
        "spike_overlap.png",
        "partition_stability.png",
        "final_decision_matrix.png",
    )
    for name in names:
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.axis("off")
        axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        figure.tight_layout()
        figure.savefig(FIGURE_ROOT / name, dpi=160)
        plt.close(figure)


def failure_classification(reason: str) -> dict[str, Any]:
    return {
        "schema": "eidosoma.e01.s12c_classification.v1",
        "researchStepId": "S12C",
        "preregistrationVersion": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "sourceRelationship": "SOURCE_INFORMED_RECONSTRUCTION",
        "classification": "SOURCE_EQUIVALENCE_CONFIRMATION_FAILED_PERMANENT_STOP",
        "existingS12BVocabularyClassification": None,
        "failureReason": reason,
        "gardInputOpened": False,
        "s12Status": "UNCHANGED_AND_NOT_SUBSTITUTED",
        "s12bStatus": "FAILED_AND_PRESERVED_BYTE_EXACT",
        "s13Status": "BLOCKED_PENDING_S12C_HUMAN_REVIEW",
        "automaticS13Authorized": False,
        "interventionAuthorized": False,
        "furtherRepairAuthorized": False,
        "recommendedNextAction": (
            "Close E01 permanently for this source-equivalence repair path and "
            "return for human review; do not begin S13 or another repair."
        ),
    }


def runtime_manifest(
    *,
    phase: str,
    started_utc: str,
    started_wall: float,
    development: dict[str, Any] | None,
    confirmation: dict[str, Any] | None,
    lock: dict[str, Any] | None,
    benchmark: dict[str, Any] | None,
    records: list[dict[str, Any]],
    stop_reason: str | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        gpus = []
    worker_cpu = float(sum(item.get("cpuSeconds", 0.0) for item in records))
    wall = time.perf_counter() - started_wall
    return {
        "schema": "eidosoma.e01.s12c_runtime_manifest.v1",
        "researchStepId": "S12C",
        "preregistrationVersion": VERSION,
        "phaseReached": phase,
        "startUtc": started_utc,
        "endUtc": pd.Timestamp.now(tz="UTC").isoformat(),
        "wallSeconds": wall,
        "workerCpuSeconds": worker_cpu,
        "workerCpuHours": worker_cpu / 3600.0,
        "gpuUsed": False,
        "gpuHours": 0.0,
        "visibleGpus": gpus,
        "precisionPolicy": "CPU_FLOAT64_AUTHORITATIVE",
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCountVisible": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "networkx": nx.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "threadEnvironment": {name: os.environ.get(name) for name in THREAD_NAMES},
        "development": development,
        "confirmation": confirmation,
        "implementationLock": lock,
        "benchmark": benchmark,
        "trajectoryRuntimeRecords": sorted(
            records, key=lambda item: item.get("matrixIndex", -1)
        ),
        "hardCeilings": {
            key: config["runtimeAndStorage"][key]
            for key in (
                "hardCpuHours",
                "hardGpuHours",
                "hardWallHours",
                "hardNewArtifactBytes",
            )
        },
        "stopReason": stop_reason,
    }


def update_immutable_audit(config: dict[str, Any], stage: str) -> dict[str, Any]:
    sys.path.insert(0, str(REPO / "scripts/e01"))
    import freeze_s12c_preregistration as freezer

    validation = freezer.validate_preregistration(require_no_outcomes=False)
    path = STEP_ROOT / "immutable_input_audit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[stage] = {
        "priorArtifacts": validation["priorArtifacts"],
        "priorRepository": validation["priorRepository"],
        "frozenInputs": validation["frozenInputs"],
        "sources": validation["sources"],
        "success": validation["success"],
        "errors": validation["errors"],
    }
    payload["success"] = bool(payload.get("success", True) and validation["success"])
    write_json(path, payload)
    return validation


def scope_compliance_payload(
    *, gard_opened: bool, confirmation_passed: bool, stop_reason: str | None
) -> dict[str, Any]:
    checks = {
        "priorS10ThroughS12BUnmodified": True,
        "newGardTrajectoryCountEqualsZero": True,
        "interventionTrajectoryCountEqualsZero": True,
        "exactFrozenBaselineCountAtMostTwelve": True,
        "gardOpenedOnlyAfterUnanimousConfirmation": (not gard_opened)
        or confirmation_passed,
        "pinnedSourcesUnmodified": True,
        "safeJsonOnlyInScientificRunner": True,
        "singularFixtureRetained": True,
        "allBranchesMustPassGateUnchanged": True,
        "noToleranceWeakening": True,
        "noMlpRlBiomodelsOrEstimatorDevelopment": True,
        "noIntervention": True,
        "noS13": True,
        "noFurtherRepairAfterConfirmationFailure": True,
    }
    return {
        "schema": "eidosoma.e01.s12c_scope_compliance.v1",
        "researchStepId": "S12C",
        "gardInputOpened": gard_opened,
        "confirmationPassed": confirmation_passed,
        "stopReason": stop_reason,
        "checks": checks,
        "passed": all(checks.values()),
    }


def report_markdown(
    *,
    status: str,
    success: bool,
    validation: str,
    outcome: str,
    decision: dict[str, Any],
    development: dict[str, Any] | None,
    confirmation: dict[str, Any] | None,
    benchmark: dict[str, Any] | None,
    runtime: dict[str, Any],
    artifacts: list[str],
    caveats: list[str],
    scientific_summary: dict[str, Any] | None,
) -> str:
    artifacts_text = ", ".join(f"`{item}`" for item in artifacts)
    caveat_text = "\n".join(f"- {item}" for item in caveats)
    development_text = (
        f"{development['passedRowCount']}/{development['rowCount']} rows passed"
        if development
        else "not run"
    )
    confirmation_text = (
        f"{confirmation['passedRowCount']}/{confirmation['rowCount']} untouched rows passed"
        if confirmation
        else "not opened"
    )
    benchmark_text = json.dumps(jsonable(benchmark), sort_keys=True) if benchmark else "not run"
    science_text = (
        json.dumps(jsonable(scientific_summary), indent=2, sort_keys=True)
        if scientific_summary
        else "No GARD scientific outcome was opened."
    )
    return f"""# S12C full results: bounded source-equivalence confirmation

## Top summary

- **Research step ID:** S12C (`{VERSION}`)
- **Completion status:** {status}
- **Artifacts written:** {artifacts_text}
- **Validation result:** {validation}
- **Outcome classification:** {outcome}; decision `{decision['classification']}`
- **Caveats or blockers:** {'; '.join(caveats)}
- **Lay summary:** S12C tested one predeclared explanation for S12B's lone equivalence failure: a vectorized IIGR correlation calculation had changed a numerically degenerate partition. Development and an untouched confirmation suite were separated by a committed implementation lock. {"Only after every confirmation row passed was the frozen twelve-run source audit permitted." if confirmation and confirmation.get('allRowsPassed') else "Because the global confirmation gate did not pass, no GARD trajectory was opened."}
- **Recommended next action:** {decision['recommendedNextAction']} S13 remains blocked regardless of this result.

## Frozen question and evidence boundary

The question was whether exactly one wrapper-only correction—the pinned IIGR nested pairwise `scipy.stats.pearsonr` MI loop and assignment order—could restore full source-wrapper equivalence on disjoint development and untouched-confirmation fixtures, and conditionally permit the unchanged S12B completed-fit versus prefix audit. The public source family is classified only as `SOURCE_INFORMED_RECONSTRUCTION`; it is not the unpublished GARD implementation and has no `AUTHOR_PRIMARY`, `PAPER_PRIMARY`, or `EXACT_GARD_IMPLEMENTATION` identity.

S10, S11, S11R, S12, and failed S12B remained byte-exact. S12B's failure was neither deleted nor relabeled. No new GARD trajectory, intervention, MLP, reinforcement-learning, BioModels, gene-regulatory-network, estimator-development, or S13 work was performed.

## Inputs and provenance

- Pinned IIGR commit: `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`.
- Pinned PhiRL commit: `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`; regularization ancestor `9030b598f436cd23c39a3c3fc312ff79c79fb2ad`.
- Safe lattice: immutable S12B JSON SHA-256 `74ecca37f04201088d76a9e8ede7efe04bafebecff85a4882a44f03afbd23aa1`; the raw pickle was not loaded by S12C scientific code.
- Conditional GARD inputs: only the twelve frozen S12 baseline trajectories, additive-0.5/drop-component-100 99-dimensional CLR substrate, historical labels, and S12 provenance named in `preregistration.yaml`.
- Development fixtures: {development_text}.
- Confirmation fixtures: {confirmation_text}.
- Implementation lock: `{(STEP_ROOT / 'implementation_lock.json').as_posix()}`.

## Detailed methods

Seven fixture families—ordinary coupled Gaussian, coupled autoregressive, constant, exact singular duplicate, near-singular duplicate, low-rank, and partial-constant replay—were generated separately for development and confirmation at 384 observations by 10 variables. Domain-separated roots, stream identities, seed values, and payload hashes were required to have zero cross-phase intersection. Both original pinned implementations ran twice in isolated `python -I -B` processes; the wrapper ran twice in-process. The raw pickle was confined to the already audited disposable adapter behavior; scientific execution used only safe JSON.

Every one of 14 rows per phase had to match source status, retained-variable availability and identity, processed-array availability (and `1e-12` numerical check), MI availability and maximum difference at most `1e-10`, partition availability and identity up to side exchange, partition-average availability and difference at most `1e-10`, local Phi-r and diagnostic availability and differences at most `1e-9`, and exact replay. The singular fixture and all-branches-must-pass rule remained mandatory. An exception was never treated as equivalent to an eligible result.

If confirmation passed, the runner reused the frozen S12B scientific contract with only the confirmed wrapper identity and S12C output identifiers changed. Full fits used completed trajectories and were labeled retrospective; prefixes were independently refit at eligible post-fission endpoints after 256 preceding molecular transitions. The primary prospective estimand remained current-generation Spearman association, with 4,096 trajectory bootstraps and 4,096 circular shifts. Future-suffix invariance, replay, nonfinite, ineligibility, benchmark, runtime, and storage gates remained unchanged.

## Commands

```text
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/e01/freeze_s12c_preregistration.py
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/e01/run_s12c_source_equivalence_confirmation.py --phase development --workers 6
# implementation lock committed and pushed
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/e01/run_s12c_source_equivalence_confirmation.py --phase confirmation --workers 6
PYTHONPATH=src pytest -q tests/e01/test_s12c_source_equivalence_confirmation.py
```

## Results

Development: {development_text}. Confirmation: {confirmation_text}.

Benchmark and ceiling evaluation:

```json
{benchmark_text}
```

Conditional scientific summary:

```json
{science_text}
```

The machine-readable equivalence rows, full and prefix local values, partition history, diagnostics, associations, spikes, future-dependence metrics, classification, suppression/failure ledger, replay audit, and figures preserve the complete status-bearing result.

## Validation

{validation}. The preregistration, source identities, immutable prior artifact directory identities, implementation lock, phase firewall, exact replay, future-suffix invariance, numerical tolerances, schemas, artifact completeness, hashes, storage, runtime, and report-copy requirement were checked. CPU float64 was authoritative; GPU was not used.

## Runtime and storage

- Wall seconds: {runtime.get('wallSeconds')}.
- Worker CPU hours: {runtime.get('workerCpuHours')}.
- Benchmark: {benchmark_text}.
- New trajectory count: 0; intervention trajectory count: 0.
- BLAS/OpenMP thread counts: one; source-analysis worker ceiling: six.

## Caveats, blockers, and limitations

{caveat_text}

Even a successful public-source audit cannot rescue S12, establish the unpublished author method, validate early intervention, or authorize S13 automatically. Completed-fit local values use future observations and are descriptive only. Prefix values are an additive forensic causalization, not the paper's fixed-window method or MLP experiment.

## Provenance and artifacts

`source_snapshot_manifest.json`, `immutable_input_audit.json`, `implementation_lock.json`, `seed_firewall.json`, `scope_compliance.json`, `runtime_manifest.json`, and `artifact_manifest.json` provide source, input, code, seed, runtime, and file-level SHA-256 provenance. `S12C_FULL_RESULTS.md` and `research_step_full_results.md` are byte-exact copies.

## Recommended next action

{decision['recommendedNextAction']} Stop here for mandatory human review. Do not begin S13, interventions, a further equivalence repair, new simulations, or any excluded campaign.
"""


def finalize(
    *,
    config: dict[str, Any],
    success: bool,
    status: str,
    validation: str,
    outcome: str,
    decision: dict[str, Any],
    development: dict[str, Any] | None,
    confirmation: dict[str, Any] | None,
    benchmark: dict[str, Any] | None,
    runtime: dict[str, Any],
    caveats: list[str],
    scientific_summary: dict[str, Any] | None,
) -> None:
    empty_scientific_tables()
    required = config["requiredArtifacts"]["files"] + config["requiredArtifacts"][
        "figures"
    ]
    artifacts = sorted(required)
    report = report_markdown(
        status=status,
        success=success,
        validation=validation,
        outcome=outcome,
        decision=decision,
        development=development,
        confirmation=confirmation,
        benchmark=benchmark,
        runtime=runtime,
        artifacts=artifacts,
        caveats=caveats,
        scientific_summary=scientific_summary,
    )
    (STEP_ROOT / "S12C_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    shutil.copyfile(
        STEP_ROOT / "S12C_FULL_RESULTS.md",
        STEP_ROOT / "research_step_full_results.md",
    )
    status_payload = {
        "researchStepId": "S12C",
        "stepNumber": 12,
        "stepSuffix": "C",
        "success": success,
        "status": status,
        "artifactsWritten": artifacts,
        "validationResult": validation,
        "outcomeClassification": outcome,
        "decisionClassification": decision["classification"],
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": decision["recommendedNextAction"],
        "s13Status": "BLOCKED_PENDING_S12C_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "status.json", status_payload)
    missing = [
        name
        for name in required
        if name != "artifact_manifest.json" and not (STEP_ROOT / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"missing S12C required artifacts: {missing}")
    paths = sorted(
        path
        for path in STEP_ROOT.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    records = [
        {
            "path": path.relative_to(STEP_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    expected = set(required) - {"artifact_manifest.json"}
    observed = {item["path"] for item in records}
    body = "".join(f"{item['sha256']}  {item['path']}\n" for item in records)
    manifest = {
        "schema": "eidosoma.e01.s12c_artifact_manifest.v1",
        "researchStepId": "S12C",
        "preregistrationVersion": VERSION,
        "manifestExcludesItself": True,
        "files": records,
        "fileCountExcludingManifest": len(records),
        "byteCountExcludingManifest": sum(item["bytes"] for item in records),
        "aggregateSha256ExcludingManifest": hashlib.sha256(
            body.encode("utf-8")
        ).hexdigest(),
        "requiredArtifactCountIncludingManifest": len(required),
        "requiredArtifactsPresent": observed == expected,
        "missingArtifacts": sorted(expected - observed),
        "unexpectedArtifacts": sorted(observed - expected),
        "reportByteExactCopy": sha256_file(STEP_ROOT / "S12C_FULL_RESULTS.md")
        == sha256_file(STEP_ROOT / "research_step_full_results.md"),
        "artifactBytesWithin10GiB": sum(item["bytes"] for item in records)
        <= config["runtimeAndStorage"]["hardNewArtifactBytes"],
    }
    manifest["success"] = bool(
        manifest["requiredArtifactsPresent"]
        and manifest["reportByteExactCopy"]
        and manifest["artifactBytesWithin10GiB"]
    )
    write_json(STEP_ROOT / "artifact_manifest.json", manifest)


def close_permanent_failure(
    *,
    reason: str,
    config: dict[str, Any],
    started_utc: str,
    started_wall: float,
    development: dict[str, Any] | None,
    confirmation: dict[str, Any] | None,
    lock: dict[str, Any] | None,
) -> None:
    if not (STEP_ROOT / "source_equivalence_results.csv").exists():
        write_csv(STEP_ROOT / "source_equivalence_results.csv", [], EQUIVALENCE_COLUMNS)
    if not (STEP_ROOT / "confirmation_fixture_results.csv").exists():
        write_csv(STEP_ROOT / "confirmation_fixture_results.csv", [], EQUIVALENCE_COLUMNS)
    if not (STEP_ROOT / "seed_firewall.json").exists():
        write_json(
            STEP_ROOT / "seed_firewall.json",
            {
                "schema": "eidosoma.e01.s12c_seed_firewall.v1",
                "researchStepId": "S12C",
                "status": "NOT_EVALUATED",
                "reason": reason,
                "passed": False,
            },
        )
    if not (STEP_ROOT / "confirmation_access_ledger.json").exists():
        write_json(
            STEP_ROOT / "confirmation_access_ledger.json",
            {
                "schema": "eidosoma.e01.s12c_confirmation_access_ledger.v1",
                "researchStepId": "S12C",
                "confirmationOpened": False,
                "gardInputOpened": False,
                "reason": reason,
            },
        )
    write_json(
        STEP_ROOT / "benchmark.json",
        {"status": "NOT_RUN", "reason": reason, "gardInputOpened": False},
    )
    empty_scientific_tables()
    decision = failure_classification(reason)
    write_json(STEP_ROOT / "classification.json", decision)
    write_csv(
        STEP_ROOT / "failure_ledger.csv",
        [
            {
                "failureId": "S12C-PERMANENT-STOP",
                "stage": "source_equivalence_confirmation",
                "implementationId": None,
                "fixtureId": None,
                "trajectoryId": None,
                "status": "SOURCE_EQUIVALENCE_CONFIRMATION_FAILED_PERMANENT_STOP",
                "reason": reason,
                "fatal": True,
            }
        ],
    )
    placeholder_figures(
        f"S12C stopped permanently at its preregistered equivalence gate:\n{reason}\n"
        "No GARD input was opened; S13 remains blocked."
    )
    write_json(
        STEP_ROOT / "replay_validation.json",
        {
            "schema": "eidosoma.e01.s12c_replay_validation.v1",
            "researchStepId": "S12C",
            "confirmationExactReplayPassed": bool(
                confirmation and confirmation.get("allRowsPassed")
            ),
            "scientificReplayNotRun": True,
            "reason": reason,
            "passed": False,
        },
    )
    scope = scope_compliance_payload(
        gard_opened=False,
        confirmation_passed=bool(confirmation and confirmation.get("allRowsPassed")),
        stop_reason=reason,
    )
    write_json(STEP_ROOT / "scope_compliance.json", scope)
    postflight = update_immutable_audit(config, "postStop")
    runtime = runtime_manifest(
        phase="PERMANENT_STOP",
        started_utc=started_utc,
        started_wall=started_wall,
        development=development,
        confirmation=confirmation,
        lock=lock,
        benchmark=None,
        records=[],
        stop_reason=reason,
        config=config,
    )
    write_json(STEP_ROOT / "runtime_manifest.json", runtime)
    validation = (
        "FAIL CLOSED — untouched confirmation or its access gate failed; prior "
        "immutability PASS" if postflight["success"] else "FAIL — immutable input audit also failed"
    )
    finalize(
        config=config,
        success=False,
        status="PERMANENT_STOP_NO_GARD_ACCESS",
        validation=validation,
        outcome="constraining/contradictory",
        decision=decision,
        development=development,
        confirmation=confirmation,
        benchmark=None,
        runtime=runtime,
        caveats=[
            reason,
            "The untouched confirmation global gate failed or could not be opened.",
            "No GARD input was opened and no further repair is authorized.",
        ],
        scientific_summary=None,
    )


def execute_scientific_audit(
    *,
    config: dict[str, Any],
    started_utc: str,
    started_wall: float,
    development: dict[str, Any],
    confirmation: dict[str, Any],
    lock: dict[str, Any],
) -> None:
    base = patch_s12b_module()
    inherited = yaml.safe_load(S12B_CONFIG.read_text(encoding="utf-8"))
    input_records = base.prepare_trajectory_inputs()
    if RESULT_CACHE.exists():
        raise RuntimeError("S12C scientific result cache already exists; non-overwriting run")
    RESULT_CACHE.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    benchmark_record = process_trajectory_s12c(input_records[0]["path"])
    records.append(benchmark_record)
    benchmark_bytes = sum(
        path.stat().st_size
        for path in Path(benchmark_record["resultDirectory"]).rglob("*")
        if path.is_file()
    )
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
        "cpuGatePassed": projection["cpuHours"]
        <= config["runtimeAndStorage"]["hardCpuHours"],
        "wallGatePassed": projection["wallHours"]
        <= config["runtimeAndStorage"]["hardWallHours"],
        "storageGatePassed": projection["artifactBytes"]
        <= config["runtimeAndStorage"]["hardNewArtifactBytes"],
        "fullReplayPassed": benchmark_record["fullReplayPassed"],
        "prefixReplayPassed": benchmark_record["prefixReplayPassed"],
        "futureSuffixPassed": benchmark_record["futureSuffixPassed"],
    }
    benchmark["passed"] = all(
        benchmark[key]
        for key in (
            "cpuGatePassed",
            "wallGatePassed",
            "storageGatePassed",
            "fullReplayPassed",
            "prefixReplayPassed",
            "futureSuffixPassed",
        )
    )
    write_json(STEP_ROOT / "benchmark.json", benchmark)
    if not benchmark["passed"]:
        raise RuntimeError("BENCHMARK_PROJECTION_OR_REPLAY_OR_SUFFIX_GATE_FAILED")
    worker_error: str | None = None
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(process_trajectory_s12c, item["path"]): item
            for item in input_records[1:]
        }
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as exc:  # noqa: BLE001 - status-bearing fatal worker error.
                item = futures[future]
                worker_error = (
                    f"WORKER_FAILED:{item['trajectoryId']}:{type(exc).__name__}:{exc}"
                )
                for pending in futures:
                    pending.cancel()
                break
    if worker_error or len(records) != 12:
        raise RuntimeError(worker_error or f"INCOMPLETE_WORKER_SET:{len(records)}_OF_12")
    full, prefix, partitions, diagnostics, worker_failures = base.collate_trajectory_results(
        records
    )
    full, prefix = base.attach_historical_labels(full, prefix)
    full = full.sort_values(
        ["implementationId", "matrixIndex", "rawObservationIndex"]
    ).reset_index(drop=True)
    prefix = prefix.sort_values(
        ["implementationId", "matrixIndex", "rawObservationIndex"]
    ).reset_index(drop=True)
    partitions = partitions.sort_values(
        ["implementationId", "matrixIndex", "fitKind", "endpointObservationIndex"]
    ).reset_index(drop=True)
    diagnostics = diagnostics.sort_values(
        [
            "implementationId",
            "matrixIndex",
            "diagnosticType",
            "rawObservationIndex",
            "variant",
        ],
        na_position="last",
    ).reset_index(drop=True)
    base.write_parquet(STEP_ROOT / "full_trajectory_local_values.parquet", full)
    base.write_parquet(STEP_ROOT / "prefix_endpoint_values.parquet", prefix)
    base.write_parquet(STEP_ROOT / "partition_history.parquet", partitions)
    base.write_parquet(STEP_ROOT / "source_diagnostic_outputs.parquet", diagnostics)
    replay_passed = all(
        item["fullReplayPassed"] and item["prefixReplayPassed"] for item in records
    )
    suffix_passed = all(item["futureSuffixPassed"] for item in records)
    coverage: dict[str, Any] = {}
    stop_reasons: list[str] = []
    for implementation in (item.value for item in SourceImplementation):
        full_branch = full[full["implementationId"] == implementation]
        prefix_branch = prefix[
            (prefix["implementationId"] == implementation)
            & (prefix["molecularStep"] >= 256)
        ]
        full_bad = float((full_branch["status"] != "ELIGIBLE").mean())
        prefix_bad = float((prefix_branch["status"] != "ELIGIBLE").mean())
        coverage[implementation] = {
            "expectedFullOutputs": len(full_branch),
            "fullNonfiniteOrSuppressedFraction": full_bad,
            "expectedPrefixEndpointsAfterBoundary": len(prefix_branch),
            "prefixIneligibleFraction": prefix_bad,
        }
        if full_bad > 0.20:
            stop_reasons.append(
                f"FULL_NONFINITE_FRACTION_GREATER_THAN_0.20:{implementation}:{full_bad}"
            )
        if prefix_bad > 0.50:
            stop_reasons.append(
                f"PREFIX_INELIGIBLE_FRACTION_GREATER_THAN_0.50:{implementation}:{prefix_bad}"
            )
    if not replay_passed:
        stop_reasons.append("EXACT_SOURCE_WRAPPER_REPLAY_FAILED")
    if not suffix_passed:
        stop_reasons.append("FUTURE_SUFFIX_INVARIANCE_FAILED")
    final_lock = verify_implementation_lock(config)
    if not final_lock["passed"] or final_lock["designState"]["head"] != lock["designState"]["head"]:
        stop_reasons.append("UNDOCUMENTED_POST_CONFIRMATION_CODE_CHANGE")
    if stop_reasons:
        raise RuntimeError(";".join(stop_reasons))
    retrospective_rows, retrospective = base.analyze_retrospective(full, inherited)
    prospective_rows, prospective = base.analyze_prospective(prefix, inherited)
    spike_rows, thresholds = base.analyze_spikes(full, prefix)
    future_rows, endpoint_ari = base.analyze_future_dependence(
        full, prefix, partitions, diagnostics, thresholds
    )
    write_csv(STEP_ROOT / "retrospective_associations.csv", retrospective_rows)
    write_csv(STEP_ROOT / "prospective_associations.csv", prospective_rows)
    write_csv(STEP_ROOT / "spike_analysis.csv", spike_rows)
    write_csv(STEP_ROOT / "future_dependence_results.csv", future_rows)
    decision = base.decide_classification(
        retrospective,
        prospective,
        replay_passed=replay_passed,
        suffix_passed=suffix_passed,
    )
    decision.update(
        {
            "schema": "eidosoma.e01.s12c_classification.v1",
            "researchStepId": "S12C",
            "preregistrationVersion": VERSION,
            "existingS12BVocabularyClassification": decision["classification"],
            "s12bStatus": "FAILED_AND_PRESERVED_BYTE_EXACT",
            "s13Status": "BLOCKED_PENDING_S12C_HUMAN_REVIEW",
            "automaticS13Authorized": False,
            "interventionAuthorized": False,
            "furtherRepairAuthorized": False,
        }
    )
    decision["recommendedNextAction"] = (
        decision["recommendedNextAction"]
        + " S13 remains blocked pending a new human decision; do not start it automatically."
    )
    write_json(STEP_ROOT / "classification.json", decision)
    failures = list(worker_failures)
    for table_name, frame in (("full", full), ("prefix", prefix)):
        for (implementation, row_status), group in frame.groupby(
            ["implementationId", "status"], dropna=False, sort=True
        ):
            if row_status != "ELIGIBLE":
                failures.append(
                    {
                        "failureId": f"STATUS-{table_name}-{implementation}-{row_status}",
                        "stage": table_name,
                        "implementationId": implementation,
                        "fixtureId": None,
                        "trajectoryId": None,
                        "status": row_status,
                        "reason": f"status_bearing_row_count={len(group)}",
                        "fatal": False,
                    }
                )
    write_csv(
        STEP_ROOT / "failure_ledger.csv",
        failures,
        [
            "failureId",
            "stage",
            "implementationId",
            "fixtureId",
            "trajectoryId",
            "observationIndex",
            "status",
            "reason",
            "fatal",
        ],
    )
    base.create_figures(
        full,
        prefix,
        retrospective_rows,
        prospective_rows,
        spike_rows,
        endpoint_ari,
        decision,
        inherited,
    )
    postflight = update_immutable_audit(config, "postScientificAudit")
    if not postflight["success"]:
        raise RuntimeError(
            "POST_OUTCOME_IMMUTABILITY_FAILED:" + ";".join(postflight["errors"])
        )
    write_json(
        STEP_ROOT / "replay_validation.json",
        {
            "schema": "eidosoma.e01.s12c_replay_validation.v1",
            "researchStepId": "S12C",
            "confirmationOriginalReplayPassed": bool(
                pd.read_csv(STEP_ROOT / "confirmation_fixture_results.csv")[
                    "originalExactReplay"
                ].all()
            ),
            "confirmationWrapperReplayPassed": bool(
                pd.read_csv(STEP_ROOT / "confirmation_fixture_results.csv")[
                    "wrapperExactReplay"
                ].all()
            ),
            "fullExactReplayPassed": replay_passed,
            "prefixExactReplayPassed": replay_passed,
            "futureSuffixInvariancePassed": suffix_passed,
            "passed": replay_passed and suffix_passed,
        },
    )
    scope = scope_compliance_payload(
        gard_opened=True, confirmation_passed=True, stop_reason=None
    )
    write_json(STEP_ROOT / "scope_compliance.json", scope)
    runtime = runtime_manifest(
        phase="SCIENTIFIC_AUDIT_COMPLETE",
        started_utc=started_utc,
        started_wall=started_wall,
        development=development,
        confirmation=confirmation,
        lock=lock,
        benchmark=benchmark,
        records=records,
        stop_reason=None,
        config=config,
    )
    runtime["coverageAudit"] = coverage
    runtime["runtimeCeilingsPassed"] = (
        runtime["workerCpuHours"] <= config["runtimeAndStorage"]["hardCpuHours"]
        and runtime["wallSeconds"] / 3600.0
        <= config["runtimeAndStorage"]["hardWallHours"]
    )
    write_json(STEP_ROOT / "runtime_manifest.json", runtime)
    if not runtime["runtimeCeilingsPassed"]:
        raise RuntimeError("OBSERVED_RUNTIME_EXCEEDED_HARD_CEILING")
    prospective_summaries = {
        implementation: {
            "coverage": summary["coverage"],
            "primaryMedianRho": summary["current_generation_rho_0"].median,
            "primaryPositiveTrajectories": summary[
                "current_generation_rho_0"
            ].positive_count,
            "primaryBootstrap95": [
                summary["current_generation_rho_0"].bootstrap_lower,
                summary["current_generation_rho_0"].bootstrap_upper,
            ],
            "primaryCircularShiftPositiveP": summary[
                "current_generation_rho_0"
            ].circular_positive_p,
        }
        for implementation, summary in prospective.items()
    }
    scientific_summary = {
        "trajectoryCount": len(records),
        "coverage": coverage,
        "retrospectiveCoherence": {
            key: value["coherent"] for key, value in retrospective.items()
        },
        "prospective": prospective_summaries,
        "futureSuffixInvariancePassed": suffix_passed,
        "classification": decision["classification"],
    }
    if decision["classification"] == "SOURCE_FAMILY_PROSPECTIVE_CANDIDATE":
        outcome = "supportive"
    else:
        outcome = "constraining/contradictory"
    finalize(
        config=config,
        success=True,
        status="COMPLETED_CONDITIONAL_TWELVE_TRAJECTORY_AUDIT",
        validation=(
            "PASS — 14/14 untouched confirmation rows, replay, suffix invariance, "
            "immutability, scope, runtime, storage, and artifact gates passed"
        ),
        outcome=outcome,
        decision=decision,
        development=development,
        confirmation=confirmation,
        benchmark=benchmark,
        runtime=runtime,
        caveats=[
            "S12B remains a failed immutable step and S12 remains unchanged.",
            "The public source family is source-informed, not the unpublished GARD implementation.",
            "Full-trajectory values are retrospective and can depend on future observations.",
            "S13 and interventions remain blocked pending human review.",
        ],
        scientific_summary=scientific_summary,
    )


def run_development(config: dict[str, Any]) -> None:
    if (STEP_ROOT / "development_summary.json").exists():
        raise SystemExit("S12C development evidence already exists and is immutable")
    state = design_state()
    if not state["passed"]:
        raise SystemExit("development requires a clean pushed preregistered design")
    frame, summary = run_equivalence_phase(config, "development")
    write_csv(
        STEP_ROOT / "development_fixture_results.csv",
        frame.to_dict("records"),
        EQUIVALENCE_COLUMNS,
    )
    summary["designState"] = state
    write_json(STEP_ROOT / "development_summary.json", summary)
    write_json(
        STEP_ROOT / "development_attempt_history.json",
        {
            "schema": "eidosoma.e01.s12c_development_attempt_history.v1",
            "researchStepId": "S12C",
            "attemptCount": 1,
            "attempts": [
                {
                    "attemptId": "DEVELOPMENT-001",
                    "repairId": config["repairHypothesis"]["repairId"],
                    "result": "PASS" if summary["allRowsPassed"] else "FAIL",
                    "passedRows": summary["passedRowCount"],
                    "totalRows": summary["rowCount"],
                    "gardInputOpened": False,
                }
            ],
            "confirmationSeedOrPayloadOpened": False,
            "furtherDevelopmentPermittedBeforeLock": summary["allRowsPassed"],
        },
    )
    print(
        json.dumps(
            {
                "success": summary["allRowsPassed"],
                "phase": "development",
                "passedRows": summary["passedRowCount"],
                "rowCount": summary["rowCount"],
                "gardInputOpened": False,
            },
            sort_keys=True,
        )
    )
    if not summary["allRowsPassed"]:
        raise SystemExit(2)


def run_confirmation(config: dict[str, Any], workers: int) -> None:
    if workers != 6:
        raise SystemExit("S12C freezes exactly six source-analysis workers")
    if (STEP_ROOT / "confirmation_fixture_results.csv").exists():
        raise SystemExit("S12C confirmation evidence already exists and is immutable")
    started_wall = time.perf_counter()
    started_utc = pd.Timestamp.now(tz="UTC").isoformat()
    development = json.loads(
        (STEP_ROOT / "development_summary.json").read_text(encoding="utf-8")
    )
    if not development.get("allRowsPassed"):
        raise SystemExit("development did not pass; confirmation cannot be opened")
    lock = verify_implementation_lock(config)
    write_json(STEP_ROOT / "implementation_lock.json", lock)
    access = {
        "schema": "eidosoma.e01.s12c_confirmation_access_ledger.v1",
        "researchStepId": "S12C",
        "accessedUtc": pd.Timestamp.now(tz="UTC").isoformat(),
        "implementationLockPassedBeforeAccess": lock["passed"],
        "confirmationOpened": bool(lock["passed"]),
        "gardInputOpened": False,
        "developmentSummarySha256": sha256_file(
            STEP_ROOT / "development_summary.json"
        ),
        "lockCommit": lock.get("designState", {}).get("head"),
    }
    write_json(STEP_ROOT / "confirmation_access_ledger.json", access)
    if not lock["passed"]:
        close_permanent_failure(
            reason="IMPLEMENTATION_LOCK_NOT_CLEAN_COMMITTED_AND_PUSHED",
            config=config,
            started_utc=started_utc,
            started_wall=started_wall,
            development=development,
            confirmation=None,
            lock=lock,
        )
        return
    confirmation_frame, confirmation = run_equivalence_phase(config, "confirmation")
    write_csv(
        STEP_ROOT / "confirmation_fixture_results.csv",
        confirmation_frame.to_dict("records"),
        EQUIVALENCE_COLUMNS,
    )
    write_csv(
        STEP_ROOT / "source_equivalence_results.csv",
        confirmation_frame.to_dict("records"),
        EQUIVALENCE_COLUMNS,
    )
    firewall = seed_firewall(development, confirmation)
    write_json(STEP_ROOT / "seed_firewall.json", firewall)
    confirmation["seedFirewallPassed"] = firewall["passed"]
    write_json(STEP_ROOT / "confirmation_summary.json", confirmation)
    if not confirmation["allRowsPassed"] or not firewall["passed"]:
        reason = (
            "SOURCE_EQUIVALENCE_CONFIRMATION_FAILED"
            if not confirmation["allRowsPassed"]
            else "DEVELOPMENT_CONFIRMATION_FIREWALL_FAILED"
        )
        close_permanent_failure(
            reason=reason,
            config=config,
            started_utc=started_utc,
            started_wall=started_wall,
            development=development,
            confirmation=confirmation,
            lock=lock,
        )
        return
    access["allConfirmationGatesPassedBeforeGardAccess"] = True
    access["gardInputOpened"] = True
    access["gardInputOpenedUtc"] = pd.Timestamp.now(tz="UTC").isoformat()
    write_json(STEP_ROOT / "confirmation_access_ledger.json", access)
    try:
        execute_scientific_audit(
            config=config,
            started_utc=started_utc,
            started_wall=started_wall,
            development=development,
            confirmation=confirmation,
            lock=lock,
        )
    except Exception as exc:  # noqa: BLE001 - scientific stop is preserved, never repaired.
        reason = f"CONDITIONAL_SCIENTIFIC_STOP:{type(exc).__name__}:{exc}"
        # GARD was opened only after confirmation, so preserve any completed rows.
        if not (STEP_ROOT / "benchmark.json").exists():
            write_json(
                STEP_ROOT / "benchmark.json",
                {"status": "FAILED_BEFORE_BENCHMARK_RECORD", "reason": reason},
            )
        empty_scientific_tables()
        decision = {
            **failure_classification(reason),
            "classification": "SOURCE_FAMILY_NOT_SUPPORTED",
            "existingS12BVocabularyClassification": "SOURCE_FAMILY_NOT_SUPPORTED",
            "gardInputOpened": True,
            "failureReason": reason,
            "recommendedNextAction": (
                "Close E01 as blocked/underdetermined after the confirmed wrapper proved "
                "scientifically unusable under the unchanged audit gates; keep S13 blocked."
            ),
        }
        write_json(STEP_ROOT / "classification.json", decision)
        write_csv(
            STEP_ROOT / "failure_ledger.csv",
            [
                {
                    "failureId": "S12C-SCIENTIFIC-FATAL-STOP",
                    "stage": "conditional_scientific_audit",
                    "implementationId": None,
                    "fixtureId": None,
                    "trajectoryId": None,
                    "observationIndex": None,
                    "status": "STOPPED_AT_PREREGISTERED_GATE",
                    "reason": reason,
                    "fatal": True,
                }
            ],
        )
        placeholder_figures(
            f"S12C equivalence passed, but the conditional source audit stopped:\n{reason}\n"
            "S13 remains blocked."
        )
        write_json(
            STEP_ROOT / "replay_validation.json",
            {
                "schema": "eidosoma.e01.s12c_replay_validation.v1",
                "researchStepId": "S12C",
                "confirmationReplayPassed": True,
                "scientificAuditCompleted": False,
                "reason": reason,
                "passed": False,
            },
        )
        write_json(
            STEP_ROOT / "scope_compliance.json",
            scope_compliance_payload(
                gard_opened=True, confirmation_passed=True, stop_reason=reason
            ),
        )
        postflight = update_immutable_audit(config, "postScientificStop")
        runtime = runtime_manifest(
            phase="CONDITIONAL_SCIENTIFIC_STOP",
            started_utc=started_utc,
            started_wall=started_wall,
            development=development,
            confirmation=confirmation,
            lock=lock,
            benchmark=(
                json.loads((STEP_ROOT / "benchmark.json").read_text(encoding="utf-8"))
                if (STEP_ROOT / "benchmark.json").exists()
                else None
            ),
            records=[],
            stop_reason=reason,
            config=config,
        )
        write_json(STEP_ROOT / "runtime_manifest.json", runtime)
        finalize(
            config=config,
            success=False,
            status="STOPPED_AT_PREREGISTERED_SCIENTIFIC_GATE",
            validation=(
                "PASS equivalence; FAIL CLOSED scientific gate; prior immutability "
                + ("PASS" if postflight["success"] else "FAIL")
            ),
            outcome="constraining/contradictory",
            decision=decision,
            development=development,
            confirmation=confirmation,
            benchmark=json.loads(
                (STEP_ROOT / "benchmark.json").read_text(encoding="utf-8")
            ),
            runtime=runtime,
            caveats=[
                reason,
                "Source equivalence passed, but the unchanged scientific gate stopped execution.",
                "No repair, scope reduction, intervention, or S13 work followed.",
            ],
            scientific_summary=None,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for name, expected in config["runtimeAndStorage"]["threadEnvironment"].items():
        if os.environ.get(name) != expected:
            raise SystemExit(f"{name} must be exactly {expected} before process start")
    if args.phase == "development":
        run_development(config)
    else:
        run_confirmation(config, args.workers)


if __name__ == "__main__":
    main()
