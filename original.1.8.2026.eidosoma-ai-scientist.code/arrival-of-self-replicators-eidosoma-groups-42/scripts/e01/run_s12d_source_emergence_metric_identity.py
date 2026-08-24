#!/usr/bin/env python3
"""Execute frozen E01 S12D source-emergence metric-identity confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from matplotlib.colors import ListedColormap
from scipy.stats import combine_pvalues
from sklearn.metrics import adjusted_rand_score

from e01_pigozzi_source_audit.analysis import finite_spearman
from e01_pigozzi_source_audit.core import SourceImplementation
from e01_pigozzi_source_audit.core import derive_seed as s12b_derive_seed
from e01_replicator_labels import (
    ClusterConfiguration,
    cluster_labels,
    historical_technique1_labels,
)
from e01_source_emergence_metric_identity.analysis import (
    association_gate,
    drift_gate,
    excursion_thresholds,
    finite_pearson,
    rank_agreement,
    replicator_drift_summary,
    significant_opposite,
    temporal_structure_rows,
    trajectory_association_summary,
)
from e01_source_emergence_metric_identity.core import (
    ATOM_KEY_STRINGS,
    CONFIRMATION_DATASET_ROLE,
    EVIDENCE_CLASS,
    EXPLORATORY_DATASET_ROLE,
    GARD_SPECIFICATION_ID,
    HISTORICAL_LABEL_ID,
    PAST_ONLY_LABEL_ID,
    RESEARCH_STEP_ID,
    ROOT_SEED_HEX,
    SOURCE_RELATIONSHIP,
    VERSION,
    all_metric_identity_fixtures,
    result_replay_equal,
    run_emergence_pipeline,
    simulate_confirmation_trajectory,
    source_pipeline_seeds,
    statistics_seed,
)
from e01_strict_mrr.core import lineage_event_rows

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S12D"
CACHE_ROOT = Path("/cache/e01_s12d")
CONFIG_PATH = (
    REPO / "configs/e01/s12d_source_emergence_metric_identity_preregistration.yaml"
)
SAFE_LATTICE = ARTIFACTS / "research_steps/S12B/safe_phi_lattice.json"
S12_ROOT = ARTIFACTS / "research_steps/S12"
S12C_ROOT = ARTIFACTS / "research_steps/S12C"
ADAPTER = REPO / "scripts/e01/s12d_original_source_metric_adapter.py"
RESULT_CACHE = CACHE_ROOT / "source_results"
INPUT_CACHE = CACHE_ROOT / "analysis_inputs"
CONFIRMATION_CACHE = CACHE_ROOT / "confirmation"
FIGURE_ROOT = STEP_ROOT / "figures"

HISTORICAL_CONFIG = "E01-S08-YH-T1-HGT090-v1.0.0"
ONLINE_CONFIG = "E01-S08-YC-COS-HGT090-MIN3-ONLINE-v1.0.0"
ELIGIBLE_SOURCE_STATUSES = {"ELIGIBLE", "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES"}
REQUIRED_TABLE_SCHEMAS: dict[str, list[str]] = {
    "full_trajectory_metric_values.parquet": [
        "researchStepId",
        "preregistrationVersion",
        "datasetRole",
        "implementationId",
        "temporalModeId",
        "temporalLabel",
        "trajectoryId",
        "matrixIndex",
        "rawObservationIndex",
        "observationKind",
        "generation",
        "molecularStep",
        "status",
        "reason",
        "synergyStatus",
        "downwardCausationStatus",
        "emergenceStatus",
        "localPhiRStatus",
        "synergy",
        "downwardCausation",
        "emergence",
        "localPhiR",
        "exactReplayPassed",
        "historicalLabel",
        "pastOnlyCosineLabel",
    ],
    "prefix_endpoint_metric_values.parquet": [
        "researchStepId",
        "preregistrationVersion",
        "datasetRole",
        "implementationId",
        "temporalModeId",
        "temporalLabel",
        "trajectoryId",
        "matrixIndex",
        "rawObservationIndex",
        "generation",
        "molecularStep",
        "fitObservationCount",
        "status",
        "reason",
        "synergyStatus",
        "downwardCausationStatus",
        "emergenceStatus",
        "localPhiRStatus",
        "synergy",
        "downwardCausation",
        "emergence",
        "localPhiR",
        "exactReplayPassed",
        "futureSuffixStructuralGatePassed",
        "futureSuffixExecutedSentinelPassed",
        "historicalLabel",
        "nextHistoricalLabel",
        "pastOnlyCosineLabel",
    ],
    "partition_history.parquet": [
        "researchStepId",
        "datasetRole",
        "implementationId",
        "temporalModeId",
        "fitKind",
        "trajectoryId",
        "matrixIndex",
        "endpointObservationIndex",
        "endpointGeneration",
        "status",
        "reason",
        "retainedVariablesJson",
        "partition1Json",
        "partition2Json",
        "exactReplayPassed",
    ],
    "source_diagnostic_outputs.parquet": [
        "researchStepId",
        "datasetRole",
        "implementationId",
        "temporalModeId",
        "trajectoryId",
        "matrixIndex",
        "rawObservationIndex",
        "generation",
        "molecularStep",
        "status",
        "reason",
        "synergy",
        "downwardCausation",
        "emergence",
        "localPhiR",
        "componentIdentityMaxAbsError",
    ],
}
CONFIRMATION_TABLE_SCHEMAS: dict[str, list[str]] = {
    "confirmation_seed_manifest.parquet": [
        "researchStepId",
        "trajectoryId",
        "replicateIndex",
        "purpose",
        "streamId",
        "seedMaterialHex",
        "bitGenerator",
        "initialStateSha256",
    ],
    "confirmation_initial_states.parquet": [
        "researchStepId",
        "datasetRole",
        "trajectoryId",
        "matrixIndex",
        "state",
        "stateSha256",
    ],
    "confirmation_trajectory_events.parquet": [
        "researchStepId",
        "datasetRole",
        "trajectoryId",
        "matrixIndex",
        "recordType",
        "generationIndexOneBased",
        "globalEventIndexOneBased",
        "recordPayloadJson",
    ],
    "confirmation_observations.parquet": [
        "researchStepId",
        "datasetRole",
        "trajectoryId",
        "matrixIndex",
        "observationIndex",
        "observationKind",
        "generation",
        "molecularStep",
        "mass",
        "state",
    ],
    "confirmation_labels.parquet": [
        "researchStepId",
        "datasetRole",
        "trajectoryId",
        "matrixIndex",
        "generation",
        "labelId",
        "labelStatus",
        "isReplicator",
        "ineligibilityReason",
    ],
    "confirmation_preprocessing.parquet": [
        "researchStepId",
        "datasetRole",
        "trajectoryId",
        "matrixIndex",
        "observationIndex",
        "preprocessingId",
        "status",
        "reason",
        "coordinateDimension",
        "finite",
        "closureError",
        "coordinates",
    ],
}


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


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], columns: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows), columns=columns)
    frame.to_csv(path, index=False, lineterminator="\n")


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
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for path in paths:
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError(f"no Parquet inputs supplied for {output}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def write_data_schema() -> None:
    write_json(
        STEP_ROOT / "data_schema.json",
        {
            "schema": "eidosoma.e01.s12d.output_schema.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "preregistrationVersion": VERSION,
            "sourceRelationship": SOURCE_RELATIONSHIP,
            "tables": {
                name: {
                    "requiredColumns": columns,
                    "statusBearing": name
                    in {
                        "full_trajectory_metric_values.parquet",
                        "prefix_endpoint_metric_values.parquet",
                        "partition_history.parquet",
                        "source_diagnostic_outputs.parquet",
                    },
                }
                for name, columns in {
                    **REQUIRED_TABLE_SCHEMAS,
                    **CONFIRMATION_TABLE_SCHEMAS,
                }.items()
            },
            "componentPolicy": {
                "retained": [
                    "synergy",
                    "downwardCausation",
                    "emergence",
                    "localPhiR",
                ],
                "statusColumns": [
                    "synergyStatus",
                    "downwardCausationStatus",
                    "emergenceStatus",
                    "localPhiRStatus",
                ],
                "nonfiniteValues": "null_value_plus_explicit_ineligible_component_status",
                "silentOmissionForbidden": True,
            },
            "primaryKeys": {
                "full_trajectory_metric_values.parquet": [
                    "datasetRole",
                    "implementationId",
                    "trajectoryId",
                    "rawObservationIndex",
                ],
                "prefix_endpoint_metric_values.parquet": [
                    "datasetRole",
                    "implementationId",
                    "trajectoryId",
                    "rawObservationIndex",
                ],
                "partition_history.parquet": [
                    "datasetRole",
                    "implementationId",
                    "trajectoryId",
                    "fitKind",
                    "endpointObservationIndex",
                ],
                "confirmation_observations.parquet": [
                    "trajectoryId",
                    "observationIndex",
                ],
                "confirmation_trajectory_events.parquet": [
                    "trajectoryId",
                    "recordType",
                    "generationIndexOneBased",
                    "globalEventIndexOneBased",
                ],
            },
        },
    )


def verify_lock() -> dict[str, Any]:
    sys.path.insert(0, str(REPO / "scripts/e01"))
    import freeze_s12d_preregistration as freezer

    return freezer.verify_lock()


def max_abs_difference(
    left: np.ndarray | None, right: np.ndarray | None
) -> float | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    if left.size == 0:
        return 0.0
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.any(finite):
        return 0.0 if np.array_equal(np.isfinite(left), np.isfinite(right)) else None
    return float(np.max(np.abs(left[finite] - right[finite])))


def npz_equal(left: Path, right: Path) -> bool:
    with (
        np.load(left, allow_pickle=False) as first,
        np.load(right, allow_pickle=False) as second,
    ):
        if set(first.files) != set(second.files):
            return False
        return all(
            np.array_equal(first[name], second[name], equal_nan=True)
            if first[name].dtype.kind in "fc"
            else np.array_equal(first[name], second[name])
            for name in first.files
        )


def load_original(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        arrays = {
            name: payload[name].copy()
            for name in payload.files
            if name != "metadata_json"
        }
    return metadata, arrays


def run_metric_identity_gate(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare literal source atom arrays with the locked safe-JSON wrapper."""

    root = CACHE_ROOT / "metric_identity"
    if root.exists():
        raise RuntimeError("metric-identity cache exists; S12D is non-overwriting")
    root.mkdir(parents=True)
    source_dirs = {
        SourceImplementation.IIGR: Path(
            config["sourceSnapshots"][SourceImplementation.IIGR.value]["localCheckout"]
        ),
        SourceImplementation.PHIRL: Path(
            config["sourceSnapshots"][SourceImplementation.PHIRL.value]["localCheckout"]
        ),
    }
    env = os.environ.copy()
    env.update(config["runtimeAndStorage"]["threadEnvironment"])
    env["PYTHONHASHSEED"] = "0"
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for suite, fixture_id, observations in all_metric_identity_fixtures():
        fixture_path = root / f"{suite}-{fixture_id}.npz"
        np.savez_compressed(fixture_path, observations=observations)
        fixture_sha = sha256_array(observations)
        for implementation in SourceImplementation:
            preprocessing_seed = statistics_seed(
                "metric_identity",
                suite,
                fixture_id,
                implementation.value,
                "preprocessing",
            )
            partition_seed = statistics_seed(
                "metric_identity", suite, fixture_id, implementation.value, "partition"
            )
            adapter_name = (
                "IIGR" if implementation is SourceImplementation.IIGR else "PHIRL"
            )
            original_paths = [
                root / f"{suite}-{fixture_id}-{implementation.value}-source-{index}.npz"
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
                    cwd=root,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            source_replay = npz_equal(original_paths[0], original_paths[1])
            metadata, arrays = load_original(original_paths[0])
            wrapper = run_emergence_pipeline(
                observations,
                implementation,
                SAFE_LATTICE,
                preprocessing_seed=preprocessing_seed,
                partition_seed=partition_seed,
            )
            replay = run_emergence_pipeline(
                observations,
                implementation,
                SAFE_LATTICE,
                preprocessing_seed=preprocessing_seed,
                partition_seed=partition_seed,
            )
            wrapper_replay = result_replay_equal(wrapper, replay)
            component_rows: dict[str, Any] = {}
            component_gates: list[bool] = []
            for source_name, wrapper_name in (
                ("synergy", "synergy"),
                ("downward_causation", "downward_causation"),
                ("emergence", "emergence"),
            ):
                source = arrays.get(source_name)
                wrapped = getattr(wrapper, wrapper_name)
                availability = (source is None) == (wrapped is None)
                length_identical = availability and (
                    source is None or len(source) == len(wrapped)
                )
                nonfinite_identical = availability and (
                    source is None
                    or np.array_equal(np.isfinite(source), np.isfinite(wrapped))
                )
                difference = max_abs_difference(source, wrapped)
                numeric = source is None or (
                    difference is not None
                    and difference
                    <= config["metricEquivalenceGate"][
                        "maximumAbsoluteDifferenceAtMost"
                    ]
                )
                gate = (
                    availability
                    and length_identical
                    and nonfinite_identical
                    and numeric
                )
                component_gates.append(gate)
                prefix = "".join(
                    part.capitalize() if index else part
                    for index, part in enumerate(source_name.split("_"))
                )
                component_rows.update(
                    {
                        f"{prefix}AvailabilityIdentical": availability,
                        f"{prefix}LengthIdentical": length_identical,
                        f"{prefix}NonfiniteMaskIdentical": nonfinite_identical,
                        f"{prefix}MaxAbsDifference": difference,
                        f"{prefix}GateAtMost1e12": gate,
                    }
                )
            source_identity = arrays.get("emergence") is None or np.array_equal(
                arrays["emergence"],
                arrays["synergy"] + arrays["downward_causation"],
                equal_nan=True,
            )
            wrapper_identity = wrapper.emergence is None or np.array_equal(
                wrapper.emergence,
                wrapper.synergy + wrapper.downward_causation,
                equal_nan=True,
            )
            serialization = metadata.get("atomKeySerialization") == list(
                ATOM_KEY_STRINGS
            )
            status_identical = metadata["status"] == wrapper.status
            passed = bool(
                status_identical
                and all(component_gates)
                and source_identity
                and wrapper_identity
                and serialization
                and source_replay
                and wrapper_replay
            )
            rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "suite": suite,
                    "fixtureId": fixture_id,
                    "fixtureSha256": fixture_sha,
                    "implementationId": implementation.value,
                    "sourceStatus": metadata["status"],
                    "wrapperStatus": wrapper.status,
                    "availabilityStatusIdentical": status_identical,
                    "sourceAtomArrayLength": len(arrays["emergence"])
                    if "emergence" in arrays
                    else 0,
                    "wrapperAtomArrayLength": len(wrapper.emergence)
                    if wrapper.emergence is not None
                    else 0,
                    "sourceFormulaExact": source_identity,
                    "wrapperFormulaExact": wrapper_identity,
                    "canonicalTupleSerializationIdentical": serialization,
                    "sourceExactReplay": source_replay,
                    "wrapperExactReplay": wrapper_replay,
                    "preprocessingSeed": preprocessing_seed,
                    "partitionSeed": partition_seed,
                    **component_rows,
                    "allGatesPassed": passed,
                }
            )
    frame = pd.DataFrame(rows)
    expected = config["metricEquivalenceGate"]["expectedRows"]
    passed = len(frame) == expected and bool(frame["allGatesPassed"].all())
    summary = {
        "schema": "eidosoma.e01.s12d.source_metric_equivalence_summary.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "expectedRows": expected,
        "observedRows": len(frame),
        "passedRows": int(frame["allGatesPassed"].sum()),
        "failedRows": int((~frame["allGatesPassed"]).sum()),
        "maximumObservedComponentDifference": float(
            np.nanmax(
                frame[
                    [
                        "synergyMaxAbsDifference",
                        "downwardCausationMaxAbsDifference",
                        "emergenceMaxAbsDifference",
                    ]
                ].to_numpy(dtype=float)
            )
        ),
        "exactSourceReplayAll": bool(frame["sourceExactReplay"].all()),
        "exactWrapperReplayAll": bool(frame["wrapperExactReplay"].all()),
        "canonicalTupleSerializationAll": bool(
            frame["canonicalTupleSerializationIdentical"].all()
        ),
        "elapsedWallSeconds": time.perf_counter() - started,
        "classificationIfFailed": "SOURCE_EMERGENCE_IDENTITY_RECONSTRUCTION_FAILED",
        "success": passed,
    }
    write_csv(STEP_ROOT / "source_metric_equivalence.csv", rows)
    write_json(STEP_ROOT / "source_metric_equivalence_summary.json", summary)
    return frame, summary


def frozen_clr(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.asarray(states, dtype=np.int64)
    masses = counts.sum(axis=1)
    closed = (counts.astype(np.float64) + 0.5) / (masses[:, None] + 50.0)
    logs = np.log(closed)
    full_clr = logs - logs.mean(axis=1, keepdims=True)
    dropped = full_clr[:, :99]
    if dropped.shape != (len(counts), 99) or not np.all(np.isfinite(dropped)):
        raise RuntimeError("frozen S12D CLR preprocessing failed")
    closure_error = np.abs(closed.sum(axis=1) - 1.0)
    return dropped, masses, closure_error


def label_confirmation(
    trajectory: Any,
) -> tuple[pd.DataFrame, dict[int, bool | None], dict[int, bool | None]]:
    post_indices = np.flatnonzero(
        np.asarray(trajectory.observation_kinds) == "post_fission"
    )
    states = trajectory.states[post_indices]
    ids = tuple(f"generation-{generation:03d}" for generation in range(1, 101))
    historical = historical_technique1_labels(
        states,
        trajectory_id=trajectory.trajectory_id,
        observation_ids=ids,
        configuration_id=HISTORICAL_CONFIG,
        threshold=0.9,
        evidence_class="PINNED_PUBLIC_HISTORICAL_SOURCE_BEHAVIOR",
    )
    online = cluster_labels(
        states,
        trajectory_id=trajectory.trajectory_id,
        observation_ids=ids,
        configuration=ClusterConfiguration(
            configuration_id=ONLINE_CONFIG,
            family_id="Y_C",
            family_name="cosine_threshold_graph",
            evidence_class="VALIDATION_ONLY_RECONSTRUCTION_NOT_AUTHOR_DEFAULT",
            metric="cosine",
            representation="raw_nonnegative_vectors",
            threshold=0.9,
            comparator="strict_greater_than",
            minimum_cluster_size=3,
            temporal_scope="past_only_online",
            zero_policy="zero_sum_observation_is_explicitly_ineligible",
        ),
    )
    rows: list[dict[str, Any]] = []
    historical_map: dict[int, bool | None] = {}
    online_map: dict[int, bool | None] = {}
    for label_id, branch, result, target in (
        (HISTORICAL_LABEL_ID, "historical", historical, historical_map),
        (PAST_ONLY_LABEL_ID, "online_cosine", online, online_map),
    ):
        for generation, record in enumerate(result.rows, start=1):
            row = record.as_dict()
            row.update(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "datasetRole": CONFIRMATION_DATASET_ROLE,
                    "matrixIndex": trajectory.matrix_index,
                    "generation": generation,
                    "labelId": label_id,
                    "labelBranch": branch,
                }
            )
            rows.append(row)
            target[generation] = record.is_replicator
    return pd.DataFrame(rows), historical_map, online_map


def flatten_seed_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "researchStepId": RESEARCH_STEP_ID,
            "seedRole": "confirmation_trajectory",
            "experimentId": payload["experimentId"],
            "specificationId": payload["specificationId"],
            "trajectoryId": payload["trajectoryId"],
            "replicateIndex": payload["replicateIndex"],
            "rootSeedSha256": payload["rootSeedSha256"],
            "couplingPolicy": payload["couplingPolicy"],
            "purpose": purpose,
            **stream,
        }
        for purpose, stream in payload["streams"].items()
    ]


def _generate_confirmation_worker(matrix_index: int) -> dict[str, Any]:
    """Simulate, label, preprocess, and serialize without importing source metrics."""

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    trajectory = simulate_confirmation_trajectory(matrix_index)
    root = CONFIRMATION_CACHE / f"matrix-{matrix_index:02d}"
    root.mkdir(parents=True, exist_ok=False)
    clr, masses, closure_errors = frozen_clr(trajectory.states)
    observations = pd.DataFrame(
        {
            "researchStepId": RESEARCH_STEP_ID,
            "datasetRole": CONFIRMATION_DATASET_ROLE,
            "trajectoryId": trajectory.trajectory_id,
            "matrixIndex": trajectory.matrix_index,
            "observationIndex": np.arange(len(trajectory.states), dtype=np.int64),
            "observationKind": list(trajectory.observation_kinds),
            "generation": trajectory.generations,
            "growthGenerationOneBased": trajectory.growth_generations_one_based,
            "molecularStep": trajectory.molecular_steps,
            "generationLocalStep": trajectory.generation_local_steps,
            "mass": masses,
            "state": [row.tolist() for row in trajectory.states],
        }
    )
    observations.to_parquet(
        root / "observations.parquet", index=False, compression="zstd"
    )
    event_rows: list[dict[str, Any]] = []
    for payload in lineage_event_rows(trajectory):
        event_rows.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "datasetRole": CONFIRMATION_DATASET_ROLE,
                "trajectoryId": payload["trajectoryId"],
                "matrixIndex": payload["matrixIndex"],
                "recordType": payload["recordType"],
                "generationIndexOneBased": payload["generation_index_one_based"],
                "globalEventIndexOneBased": payload["globalEventIndexOneBased"],
                "recordPayloadJson": json.dumps(
                    jsonable(
                        {
                            key: value
                            for key, value in payload.items()
                            if key
                            not in {
                                "trajectoryId",
                                "matrixIndex",
                                "recordType",
                                "globalEventIndexOneBased",
                            }
                        }
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    pd.DataFrame(event_rows).to_parquet(
        root / "events.parquet", index=False, compression="zstd"
    )
    labels, historical_map, online_map = label_confirmation(trajectory)
    labels.to_parquet(root / "labels.parquet", index=False, compression="zstd")
    preprocessing = pd.DataFrame(
        {
            "researchStepId": RESEARCH_STEP_ID,
            "datasetRole": CONFIRMATION_DATASET_ROLE,
            "trajectoryId": trajectory.trajectory_id,
            "matrixIndex": trajectory.matrix_index,
            "observationIndex": np.arange(len(trajectory.states), dtype=np.int64),
            "preprocessingId": "E01-S12D-PREPROC-ADD0p5-DROPCLR-D100-C100-v1.0.0",
            "status": "ELIGIBLE",
            "reason": None,
            "inputMass": masses,
            "zeroCount": np.sum(trajectory.states == 0, axis=1),
            "coordinateDimension": 99,
            "finite": np.all(np.isfinite(clr), axis=1),
            "closureError": closure_errors,
            "coordinates": [row.tolist() for row in clr],
        }
    )
    preprocessing.to_parquet(
        root / "preprocessing.parquet", index=False, compression="zstd"
    )
    np.savez_compressed(
        root / "matrix_and_initial.npz",
        beta=trajectory.beta,
        initial_state=trajectory.states[0],
    )
    np.savez_compressed(
        root / "analysis_input.npz",
        clr=clr,
        observation_index=np.arange(len(clr), dtype=np.int64),
        observation_kind=np.asarray(trajectory.observation_kinds, dtype="U32"),
        generation=trajectory.generations,
        molecular_step=trajectory.molecular_steps,
        matrix_index=np.array(matrix_index, dtype=np.int64),
        trajectory_id=np.array(trajectory.trajectory_id),
        dataset_role=np.array(CONFIRMATION_DATASET_ROLE),
        historical_label=np.asarray(
            [
                -1
                if historical_map.get(int(g)) is None
                else int(bool(historical_map[int(g)]))
                for g in trajectory.generations
            ],
            dtype=np.int8,
        ),
        online_label=np.asarray(
            [
                -1 if online_map.get(int(g)) is None else int(bool(online_map[int(g)]))
                for g in trajectory.generations
            ],
            dtype=np.int8,
        ),
    )
    seed_rows = pd.DataFrame(flatten_seed_payload(trajectory.seed_payload))
    seed_rows.to_parquet(root / "seeds.parquet", index=False, compression="zstd")
    metadata = {
        "trajectoryId": trajectory.trajectory_id,
        "matrixIndex": matrix_index,
        "trajectorySha256": trajectory.trajectory_sha256,
        "betaSha256": sha256_array(trajectory.beta),
        "initialStateSha256": sha256_array(trajectory.states[0]),
        "statesSha256": sha256_array(trajectory.states),
        "clrSha256": sha256_array(clr),
        "observationCount": len(trajectory.states),
        "molecularEventCount": int(
            np.sum(np.asarray(trajectory.observation_kinds) == "molecular_event")
        ),
        "postFissionCount": int(
            np.sum(np.asarray(trajectory.observation_kinds) == "post_fission")
        ),
        "historicalReplicatorCount": int(
            sum(value is True for value in historical_map.values())
        ),
        "pastOnlyCosineReplicatorCount": int(
            sum(value is True for value in online_map.values())
        ),
        "analysisInputSha256": sha256_file(root / "analysis_input.npz"),
        "wallSeconds": time.perf_counter() - started_wall,
        "cpuSeconds": time.process_time() - started_cpu,
        "root": str(root),
    }
    write_json(root / "metadata.json", metadata)
    return metadata


def _seed_overlap_audit(new_seeds: pd.DataFrame) -> dict[str, Any]:
    """Conservatively compare new stream IDs/material against prior seed artifacts."""

    prior_values: set[str] = set()
    prior_text_blobs: list[str] = []
    inspected: list[dict[str, Any]] = []
    for step in (
        "S06",
        "S07",
        "S08",
        "S09",
        "S10",
        "S11",
        "S11R",
        "S12",
        "S12B",
        "S12C",
    ):
        root = ARTIFACTS / f"research_steps/{step}"
        for path in sorted(root.rglob("*seed*")):
            if not path.is_file() or path.suffix.lower() not in {
                ".json",
                ".csv",
                ".parquet",
                ".yaml",
                ".yml",
            }:
                continue
            try:
                if path.suffix == ".parquet":
                    frame = pd.read_parquet(path)
                    for column in frame.columns:
                        if any(
                            token in column.lower()
                            for token in ("seed", "stream", "namespace", "root")
                        ):
                            prior_values.update(
                                str(value) for value in frame[column].dropna()
                            )
                elif path.suffix == ".csv":
                    prior_text_blobs.append(path.read_text(encoding="utf-8"))
                    frame = pd.read_csv(path)
                    for column in frame.columns:
                        if any(
                            token in column.lower()
                            for token in ("seed", "stream", "namespace", "root")
                        ):
                            prior_values.update(
                                str(value) for value in frame[column].dropna()
                            )
                else:
                    text = path.read_text(encoding="utf-8")
                    prior_text_blobs.append(text)
                    prior_values.add(text)
                inspected.append(
                    {"step": step, "path": str(path), "sha256": sha256_file(path)}
                )
            except Exception as exc:  # noqa: BLE001 - retain unreadable evidence.
                inspected.append(
                    {
                        "step": step,
                        "path": str(path),
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
    new_stream_ids = set(new_seeds["streamId"].astype(str))
    new_material = set(new_seeds["seedMaterialHex"].astype(str))
    # Exact structured-value matching catches prior tabular identities.  The
    # root itself is also searched within text-only manifests.
    stream_overlap = sorted(
        (new_stream_ids & prior_values)
        | {
            value
            for value in new_stream_ids
            if any(value in text for text in prior_text_blobs)
        }
    )
    material_overlap = sorted(
        (new_material & prior_values)
        | {
            value
            for value in new_material
            if any(value in text for text in prior_text_blobs)
        }
    )
    root_mentions = [item for item in prior_values if ROOT_SEED_HEX in item]
    return {
        "priorArtifactCountInspected": len(inspected),
        "priorArtifacts": inspected,
        "newStreamIdCount": len(new_stream_ids),
        "newSeedMaterialCount": len(new_material),
        "withinNewStreamIdsUnique": len(new_stream_ids) == len(new_seeds),
        "withinNewSeedMaterialUnique": len(new_material) == len(new_seeds),
        "priorStreamIdentityIntersection": stream_overlap,
        "priorSeedMaterialIntersection": material_overlap,
        "priorRootSeedMentions": len(root_mentions),
        "success": len(new_seeds) == 24 * 9
        and len(new_stream_ids) == len(new_seeds)
        and len(new_material) == len(new_seeds)
        and not stream_overlap
        and not material_overlap
        and not root_mentions,
    }


def generate_confirmation_data(config: dict[str, Any], workers: int) -> dict[str, Any]:
    """Generate and freeze all 24 confirmation trajectories before emergence."""

    if not json.loads(
        (STEP_ROOT / "source_metric_equivalence_summary.json").read_text()
    )["success"]:
        raise RuntimeError("source-metric identity gate did not pass")
    if CONFIRMATION_CACHE.exists():
        raise RuntimeError(
            "confirmation cache already exists; generation is non-overwriting"
        )
    CONFIRMATION_CACHE.mkdir(parents=True)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(workers, 6)) as executor:
        futures = {
            executor.submit(_generate_confirmation_worker, index): index
            for index in range(24)
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: item["matrixIndex"])
    if len(records) != 24 or [item["matrixIndex"] for item in records] != list(
        range(24)
    ):
        raise RuntimeError(
            "confirmation generation did not return exactly matrices 0..23"
        )
    if any(item["postFissionCount"] != 100 for item in records):
        raise RuntimeError(
            "one or more confirmation trajectories lacks exactly 100 fissions"
        )

    roots = [Path(item["root"]) for item in records]
    concat_parquets(
        [root / "observations.parquet" for root in roots],
        STEP_ROOT / "confirmation_observations.parquet",
    )
    concat_parquets(
        [root / "events.parquet" for root in roots],
        STEP_ROOT / "confirmation_trajectory_events.parquet",
    )
    concat_parquets(
        [root / "labels.parquet" for root in roots],
        STEP_ROOT / "confirmation_labels.parquet",
    )
    concat_parquets(
        [root / "preprocessing.parquet" for root in roots],
        STEP_ROOT / "confirmation_preprocessing.parquet",
    )
    concat_parquets(
        [root / "seeds.parquet" for root in roots],
        STEP_ROOT / "confirmation_seed_manifest.parquet",
    )
    matrices: dict[str, np.ndarray] = {}
    initial_rows: list[dict[str, Any]] = []
    for item, root in zip(records, roots, strict=True):
        with np.load(root / "matrix_and_initial.npz", allow_pickle=False) as payload:
            matrices[f"matrix_{item['matrixIndex']:02d}"] = payload["beta"].copy()
            initial = payload["initial_state"].copy()
        initial_rows.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "datasetRole": CONFIRMATION_DATASET_ROLE,
                "trajectoryId": item["trajectoryId"],
                "matrixIndex": item["matrixIndex"],
                "state": initial.tolist(),
                "stateSha256": sha256_array(initial),
            }
        )
    np.savez_compressed(STEP_ROOT / "confirmation_matrices.npz", **matrices)
    write_parquet(STEP_ROOT / "confirmation_initial_states.parquet", initial_rows)
    new_seeds = pd.read_parquet(STEP_ROOT / "confirmation_seed_manifest.parquet")
    seed_audit = _seed_overlap_audit(new_seeds)
    if not seed_audit["success"]:
        raise RuntimeError("S12D seed firewall failed")

    regeneration: list[dict[str, Any]] = []
    for index in (0, 11, 23):
        regenerated = simulate_confirmation_trajectory(index)
        expected = records[index]
        regeneration.append(
            {
                "matrixIndex": index,
                "trajectoryId": regenerated.trajectory_id,
                "expectedTrajectorySha256": expected["trajectorySha256"],
                "regeneratedTrajectorySha256": regenerated.trajectory_sha256,
                "matrixExact": sha256_array(regenerated.beta) == expected["betaSha256"],
                "statesExact": sha256_array(regenerated.states)
                == expected["statesSha256"],
                "success": regenerated.trajectory_sha256 == expected["trajectorySha256"]
                and sha256_array(regenerated.beta) == expected["betaSha256"]
                and sha256_array(regenerated.states) == expected["statesSha256"],
            }
        )
    if not all(item["success"] for item in regeneration):
        raise RuntimeError("confirmation same-engine regeneration failed")

    artifact_paths = {
        name: STEP_ROOT / name
        for name in (
            "confirmation_matrices.npz",
            "confirmation_initial_states.parquet",
            "confirmation_trajectory_events.parquet",
            "confirmation_observations.parquet",
            "confirmation_labels.parquet",
            "confirmation_preprocessing.parquet",
            "confirmation_seed_manifest.parquet",
        )
    }
    trajectory_manifest = {
        "schema": "eidosoma.e01.s12d.confirmation_trajectory_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "preregistrationVersion": VERSION,
        "datasetRole": CONFIRMATION_DATASET_ROLE,
        "gardSpecificationId": GARD_SPECIFICATION_ID,
        "rootSeedSha256": hashlib.sha256(bytes.fromhex(ROOT_SEED_HEX)).hexdigest(),
        "trajectoryCount": 24,
        "fissionsPerTrajectory": 100,
        "emergenceOutcomesComputedBeforeFreeze": False,
        "trajectories": records,
        "regeneration": regeneration,
        "seedFirewall": seed_audit,
        "artifactIdentities": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in artifact_paths.items()
        },
        "generationWallSeconds": time.perf_counter() - started,
        "success": True,
    }
    write_json(STEP_ROOT / "confirmation_trajectory_manifest.json", trajectory_manifest)
    firewall = {
        "schema": "eidosoma.e01.s12d.data_firewall_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "existingRole": EXPLORATORY_DATASET_ROLE,
        "existingCount": 12,
        "confirmationRole": CONFIRMATION_DATASET_ROLE,
        "confirmationCount": 24,
        "allConfirmationGeneratedBeforeEmergence": True,
        "allMatricesFrozen": len(matrices) == 24,
        "allInitialStatesFrozen": len(initial_rows) == 24,
        "allEventsFrozen": True,
        "allFissionsFrozen": all(item["postFissionCount"] == 100 for item in records),
        "allCompleteTrajectoryPayloadsFrozen": True,
        "allLabelsFrozen": True,
        "allPreprocessingFrozen": True,
        "seedFirewallPassed": seed_audit["success"],
        "emergenceOutcomeAccessAuthorizedAfterThisManifest": True,
        "manifestCreatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "success": True,
    }
    write_json(STEP_ROOT / "data_firewall_manifest.json", firewall)
    return trajectory_manifest


def prepare_existing_inputs() -> list[dict[str, Any]]:
    """Materialize the immutable S12 inputs only after the S12D firewall."""

    input_root = INPUT_CACHE / "existing"
    input_root.mkdir(parents=True, exist_ok=False)
    observations = pd.read_parquet(S12_ROOT / "baseline_observations.parquet")
    labels = pd.read_parquet(S12_ROOT / "replicator_labels.parquet")
    historical = labels[labels["configurationId"] == HISTORICAL_CONFIG]
    online = labels[labels["configurationId"] == ONLINE_CONFIG]
    if len(historical) != 1200 or len(online) != 1200:
        raise RuntimeError(
            "S12 existing labels are not exactly 12x100 for both frozen branches"
        )
    historical_map = {
        (str(row.trajectoryId), int(row.generation)): (
            None if pd.isna(row.isReplicator) else bool(row.isReplicator)
        )
        for row in historical.itertuples()
    }
    online_map = {
        (str(row.trajectoryId), int(row.generation)): (
            None if pd.isna(row.isReplicator) else bool(row.isReplicator)
        )
        for row in online.itertuples()
    }
    records: list[dict[str, Any]] = []
    for trajectory_id, group in observations.groupby("trajectoryId", sort=True):
        group = group.sort_values("observationIndex").reset_index(drop=True)
        indices = group["observationIndex"].to_numpy(dtype=np.int64)
        if not np.array_equal(indices, np.arange(len(group), dtype=np.int64)):
            raise RuntimeError(f"noncontiguous S12 input: {trajectory_id}")
        states = np.vstack(group["state"].map(np.asarray)).astype(np.int64)
        clr, masses, _ = frozen_clr(states)
        if not np.array_equal(masses, group["mass"].to_numpy(dtype=np.int64)):
            raise RuntimeError(f"mass mismatch in S12 input: {trajectory_id}")
        generations = group["generation"].to_numpy(dtype=np.int64)
        matrix_index = int(group["matrixIndex"].iloc[0])
        path = input_root / f"{trajectory_id}.npz"
        np.savez_compressed(
            path,
            clr=clr,
            observation_index=indices,
            observation_kind=group["observationKind"].astype(str).to_numpy(dtype="U32"),
            generation=generations,
            molecular_step=group["molecularStep"].to_numpy(dtype=np.int64),
            matrix_index=np.array(matrix_index, dtype=np.int64),
            trajectory_id=np.array(str(trajectory_id)),
            dataset_role=np.array(EXPLORATORY_DATASET_ROLE),
            historical_label=np.asarray(
                [
                    -1
                    if historical_map.get((str(trajectory_id), int(g))) is None
                    else int(bool(historical_map[(str(trajectory_id), int(g))]))
                    for g in generations
                ],
                dtype=np.int8,
            ),
            online_label=np.asarray(
                [
                    -1
                    if online_map.get((str(trajectory_id), int(g))) is None
                    else int(bool(online_map[(str(trajectory_id), int(g))]))
                    for g in generations
                ],
                dtype=np.int8,
            ),
        )
        records.append(
            {
                "trajectoryId": str(trajectory_id),
                "matrixIndex": matrix_index,
                "datasetRole": EXPLORATORY_DATASET_ROLE,
                "path": str(path),
                "observationCount": len(group),
                "clrSha256": sha256_array(clr),
            }
        )
    if len(records) != 12 or sorted(item["matrixIndex"] for item in records) != list(
        range(12)
    ):
        raise RuntimeError("S12D existing input must be exactly S12 matrices 0..11")
    return sorted(records, key=lambda item: item["matrixIndex"])


def prepare_confirmation_inputs() -> list[dict[str, Any]]:
    manifest = json.loads(
        (STEP_ROOT / "confirmation_trajectory_manifest.json").read_text()
    )
    records: list[dict[str, Any]] = []
    for item in manifest["trajectories"]:
        path = Path(item["root"]) / "analysis_input.npz"
        if not path.is_file() or sha256_file(path) != item["analysisInputSha256"]:
            raise RuntimeError(
                f"confirmation analysis input changed: {item['trajectoryId']}"
            )
        records.append(
            {
                "trajectoryId": item["trajectoryId"],
                "matrixIndex": item["matrixIndex"],
                "datasetRole": CONFIRMATION_DATASET_ROLE,
                "path": str(path),
                "observationCount": item["observationCount"],
                "clrSha256": item["clrSha256"],
            }
        )
    if len(records) != 24:
        raise RuntimeError(
            "confirmation input manifest does not contain exactly 24 trajectories"
        )
    return records


def partition_json(values: tuple[int, ...]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def partition_hash(values: tuple[int, ...]) -> str:
    return hashlib.sha256(partition_json(values).encode("utf-8")).hexdigest()


def partition_row(
    result: Any,
    *,
    dataset_role: str,
    trajectory_id: str,
    matrix_index: int,
    mode_id: str,
    fit_kind: str,
    endpoint_index: int,
    endpoint_generation: int,
    fit_count: int,
    input_hash: str,
    preprocessing_seed: int,
    partition_seed: int,
    replay_passed: bool,
) -> dict[str, Any]:
    return {
        "researchStepId": RESEARCH_STEP_ID,
        "preregistrationVersion": VERSION,
        "datasetRole": dataset_role,
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
        "miMatrixSha256": sha256_array(result.mi_matrix)
        if result.mi_matrix is not None
        else None,
        "fiedlerVectorSha256": sha256_array(result.fiedler_vector)
        if result.fiedler_vector is not None
        else None,
        "preprocessingSeed": preprocessing_seed,
        "partitionSeed": partition_seed,
        "exactReplayPassed": replay_passed,
        "sourceRelationship": SOURCE_RELATIONSHIP,
    }


def _nullable_label(value: np.integer | int) -> bool | None:
    return None if int(value) < 0 else bool(value)


def _point_values(result: Any, local_index: int) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for output, attribute in (
        ("synergy", "synergy"),
        ("downwardCausation", "downward_causation"),
        ("emergence", "emergence"),
        ("localPhiR", "local_phi_r"),
    ):
        array = getattr(result, attribute)
        if array is None or local_index >= len(array):
            values[output] = None
        else:
            value = float(array[local_index])
            values[output] = value if np.isfinite(value) else None
    return values


def _point_status(
    result: Any, replay: bool, values: dict[str, float | None]
) -> tuple[str, str | None]:
    if not replay:
        return (
            "INELIGIBLE_EXACT_REPLAY_FAILED",
            "source_metric_pipeline_exact_replay_failed",
        )
    if result.status not in ELIGIBLE_SOURCE_STATUSES:
        return result.status, result.reason
    if values["emergence"] is None:
        return (
            "INELIGIBLE_NONFINITE_EMERGENCE",
            "source_defined_emergence_nonfinite_or_absent",
        )
    return "ELIGIBLE", None


def _component_statuses(
    result: Any, replay: bool, values: dict[str, float | None]
) -> dict[str, str]:
    output: dict[str, str] = {}
    for name in ("synergy", "downwardCausation", "emergence", "localPhiR"):
        if not replay:
            status = "INELIGIBLE_EXACT_REPLAY_FAILED"
        elif result.status not in ELIGIBLE_SOURCE_STATUSES:
            status = result.status
        elif values[name] is None:
            status = f"INELIGIBLE_NONFINITE_OR_ABSENT_{name.upper()}"
        else:
            status = "ELIGIBLE"
        output[f"{name}Status"] = status
    return output


def _prefix_result_equal(left: Any, right: Any) -> bool:
    return result_replay_equal(left, right)


def scientific_source_seeds(
    implementation: SourceImplementation,
    trajectory_id: str,
    mode_id: str,
    endpoint: int,
    dataset_role: str,
) -> tuple[int, int]:
    """Replay S12C seeds for exploratory inputs; use S12D seeds for confirmation."""

    if dataset_role == EXPLORATORY_DATASET_ROLE:
        legacy_mode = (
            (
                "IIGR_FULL"
                if implementation is SourceImplementation.IIGR
                else "PHIRL_FULL"
            )
            if mode_id.endswith("_FULL")
            else (
                "IIGR_PREFIX_ENDPOINT"
                if implementation is SourceImplementation.IIGR
                else "PHIRL_PREFIX_ENDPOINT"
            )
        )
        root = "12b012b012b012b012b012b012b012b012b012b012b012b012b012b012b0"
        return (
            s12b_derive_seed(
                root,
                implementation.value,
                trajectory_id,
                legacy_mode,
                endpoint,
                "preprocessing_noise",
            ),
            s12b_derive_seed(
                root,
                implementation.value,
                trajectory_id,
                legacy_mode,
                endpoint,
                "fiedler_initialization",
            ),
        )
    return source_pipeline_seeds(implementation, trajectory_id, mode_id, endpoint)


def process_source_trajectory(input_path: str) -> dict[str, Any]:
    """Run both locked source wrappers in full and prefix modes for one trace."""

    started_wall, started_cpu = time.perf_counter(), time.process_time()
    with np.load(input_path, allow_pickle=False) as payload:
        clr = payload["clr"].astype(np.float64, copy=False)
        observation_index = payload["observation_index"].astype(np.int64, copy=False)
        observation_kind = payload["observation_kind"].astype(str)
        generation = payload["generation"].astype(np.int64, copy=False)
        molecular_step = payload["molecular_step"].astype(np.int64, copy=False)
        historical_label = payload["historical_label"].astype(np.int8, copy=False)
        online_label = payload["online_label"].astype(np.int8, copy=False)
        matrix_index = int(payload["matrix_index"])
        trajectory_id = str(payload["trajectory_id"].item())
        dataset_role = str(payload["dataset_role"].item())
    result_root = RESULT_CACHE / dataset_role / trajectory_id
    result_root.mkdir(parents=True, exist_ok=False)
    full_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    suffix_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    full_replay_all = prefix_replay_all = suffix_all = True
    post_fission_indices = [
        int(value) for value in np.flatnonzero(observation_kind == "post_fission")
    ]
    boundary_indices = [
        index for index in post_fission_indices if molecular_step[index] >= 256
    ]
    sentinels = (
        sorted(
            {
                boundary_indices[0],
                boundary_indices[len(boundary_indices) // 2],
                boundary_indices[-1],
            }
        )
        if boundary_indices
        else []
    )
    evaluations = 0
    for implementation in SourceImplementation:
        full_mode = (
            "IIGR_EMERGENCE_FULL"
            if implementation is SourceImplementation.IIGR
            else "PHIRL_EMERGENCE_FULL"
        )
        pre_seed, part_seed = scientific_source_seeds(
            implementation,
            trajectory_id,
            full_mode,
            int(observation_index[-1]),
            dataset_role,
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
        replay_ok = result_replay_equal(result, replay)
        full_replay_all &= replay_ok
        evaluations += 2
        partition_rows.append(
            partition_row(
                result,
                dataset_role=dataset_role,
                trajectory_id=trajectory_id,
                matrix_index=matrix_index,
                mode_id=full_mode,
                fit_kind="completed_trajectory",
                endpoint_index=int(observation_index[-1]),
                endpoint_generation=int(generation[-1]),
                fit_count=len(clr),
                input_hash=sha256_array(clr),
                preprocessing_seed=pre_seed,
                partition_seed=part_seed,
                replay_passed=replay_ok,
            )
        )
        expected_count = max(0, len(clr) - result.local_offset)
        for local_index in range(expected_count):
            raw_index = local_index + result.local_offset
            values = _point_values(result, local_index)
            status, reason = _point_status(result, replay_ok, values)
            component_statuses = _component_statuses(result, replay_ok, values)
            base = {
                "researchStepId": RESEARCH_STEP_ID,
                "preregistrationVersion": VERSION,
                "datasetRole": dataset_role,
                "evidenceClass": EVIDENCE_CLASS,
                "sourceRelationship": SOURCE_RELATIONSHIP,
                "implementationId": implementation.value,
                "temporalModeId": full_mode,
                "temporalLabel": "RETROSPECTIVE_FULL_TRAJECTORY_LOCAL",
                "trajectoryId": trajectory_id,
                "matrixIndex": matrix_index,
                "sourceLocalIndex": local_index,
                "rawObservationIndex": raw_index,
                "observationKind": str(observation_kind[raw_index]),
                "generation": int(generation[raw_index]),
                "molecularStep": int(molecular_step[raw_index]),
                "status": status,
                "reason": reason,
                **component_statuses,
                **values,
                "exactReplayPassed": replay_ok,
                "fitUsesFutureRelativeToValue": raw_index < len(clr) - 1,
                "historicalLabel": _nullable_label(historical_label[raw_index]),
                "pastOnlyCosineLabel": _nullable_label(online_label[raw_index]),
            }
            full_rows.append(base)
            diagnostic_rows.append(
                {
                    **{
                        key: base[key]
                        for key in (
                            "researchStepId",
                            "datasetRole",
                            "implementationId",
                            "temporalModeId",
                            "trajectoryId",
                            "matrixIndex",
                            "rawObservationIndex",
                            "generation",
                            "molecularStep",
                            "status",
                            "reason",
                            "synergy",
                            "downwardCausation",
                            "emergence",
                            "localPhiR",
                        )
                    },
                    "componentIdentityMaxAbsError": result.component_identity_max_abs_error,
                }
            )

        prefix_mode = (
            "IIGR_EMERGENCE_PREFIX_ENDPOINT"
            if implementation is SourceImplementation.IIGR
            else "PHIRL_EMERGENCE_PREFIX_ENDPOINT"
        )
        for endpoint in post_fission_indices:
            base = {
                "researchStepId": RESEARCH_STEP_ID,
                "preregistrationVersion": VERSION,
                "datasetRole": dataset_role,
                "evidenceClass": EVIDENCE_CLASS,
                "sourceRelationship": SOURCE_RELATIONSHIP,
                "implementationId": implementation.value,
                "temporalModeId": prefix_mode,
                "temporalLabel": "PAST_ONLY_PREFIX_ENDPOINT",
                "trajectoryId": trajectory_id,
                "matrixIndex": matrix_index,
                "rawObservationIndex": endpoint,
                "generation": int(generation[endpoint]),
                "molecularStep": int(molecular_step[endpoint]),
                "fitObservationCount": endpoint + 1,
                "fitUsesFutureRelativeToValue": False,
                "minimumTransitionBoundary": 256,
                "historicalLabel": _nullable_label(historical_label[endpoint]),
                "nextHistoricalLabel": (
                    _nullable_label(
                        historical_label[
                            post_fission_indices[
                                post_fission_indices.index(endpoint) + 1
                            ]
                        ]
                    )
                    if post_fission_indices.index(endpoint) + 1
                    < len(post_fission_indices)
                    else None
                ),
                "pastOnlyCosineLabel": _nullable_label(online_label[endpoint]),
            }
            if molecular_step[endpoint] < 256:
                prefix_rows.append(
                    {
                        **base,
                        "prefixInputSha256": None,
                        "preprocessingSeed": None,
                        "partitionSeed": None,
                        "status": "INELIGIBLE_BEFORE_256_TRANSITIONS",
                        "reason": "fewer_than_256_preceding_molecular_transitions",
                        "synergyStatus": "INELIGIBLE_BEFORE_256_TRANSITIONS",
                        "downwardCausationStatus": "INELIGIBLE_BEFORE_256_TRANSITIONS",
                        "emergenceStatus": "INELIGIBLE_BEFORE_256_TRANSITIONS",
                        "localPhiRStatus": "INELIGIBLE_BEFORE_256_TRANSITIONS",
                        "synergy": None,
                        "downwardCausation": None,
                        "emergence": None,
                        "localPhiR": None,
                        "exactReplayPassed": None,
                        "futureSuffixStructuralGatePassed": True,
                        "futureSuffixExecutedSentinelPassed": None,
                    }
                )
                continue
            prefix = clr[: endpoint + 1]
            prefix_hash = sha256_array(prefix)
            p_seed, f_seed = scientific_source_seeds(
                implementation, trajectory_id, prefix_mode, endpoint, dataset_role
            )
            result_prefix = run_emergence_pipeline(
                prefix,
                implementation,
                SAFE_LATTICE,
                preprocessing_seed=p_seed,
                partition_seed=f_seed,
            )
            replay_prefix = run_emergence_pipeline(
                prefix,
                implementation,
                SAFE_LATTICE,
                preprocessing_seed=p_seed,
                partition_seed=f_seed,
            )
            replay_ok = result_replay_equal(result_prefix, replay_prefix)
            prefix_replay_all &= replay_ok
            evaluations += 2
            local_index = len(prefix) - result_prefix.local_offset - 1
            values = (
                _point_values(result_prefix, local_index)
                if local_index >= 0
                else {
                    "synergy": None,
                    "downwardCausation": None,
                    "emergence": None,
                    "localPhiR": None,
                }
            )
            status, reason = _point_status(result_prefix, replay_ok, values)
            component_statuses = _component_statuses(result_prefix, replay_ok, values)
            sentinel_passed: bool | None = None
            if endpoint in sentinels:
                sentinel_passed = True
                variants: list[tuple[str, np.ndarray]] = [
                    ("suffix_deletion", prefix.copy())
                ]
                shuffled = clr.copy()
                if endpoint + 1 < len(clr):
                    rng = np.random.RandomState(
                        statistics_seed(
                            "suffix_shuffle",
                            implementation.value,
                            trajectory_id,
                            endpoint,
                        )
                    )
                    shuffled[endpoint + 1 :] = shuffled[endpoint + 1 :][
                        rng.permutation(len(clr) - endpoint - 1)
                    ]
                variants.append(
                    ("suffix_deterministic_shuffle", shuffled[: endpoint + 1])
                )
                replaced = clr.copy()
                if endpoint + 1 < len(clr):
                    rng = np.random.RandomState(
                        statistics_seed(
                            "suffix_replace",
                            implementation.value,
                            trajectory_id,
                            endpoint,
                        )
                    )
                    replaced[endpoint + 1 :] = rng.normal(
                        size=replaced[endpoint + 1 :].shape
                    )
                variants.append(
                    ("suffix_domain_separated_replacement", replaced[: endpoint + 1])
                )
                for variant_id, variant_prefix in variants:
                    variant = run_emergence_pipeline(
                        variant_prefix,
                        implementation,
                        SAFE_LATTICE,
                        preprocessing_seed=p_seed,
                        partition_seed=f_seed,
                    )
                    variant_ok = prefix_hash == sha256_array(
                        variant_prefix
                    ) and _prefix_result_equal(result_prefix, variant)
                    sentinel_passed &= variant_ok
                    evaluations += 1
                    suffix_rows.append(
                        {
                            "researchStepId": RESEARCH_STEP_ID,
                            "datasetRole": dataset_role,
                            "implementationId": implementation.value,
                            "trajectoryId": trajectory_id,
                            "matrixIndex": matrix_index,
                            "rawObservationIndex": endpoint,
                            "generation": int(generation[endpoint]),
                            "variant": variant_id,
                            "prefixInputSha256": prefix_hash,
                            "variantPrefixSha256": sha256_array(variant_prefix),
                            "exactResultIdentical": variant_ok,
                            "status": "PASS" if variant_ok else "FAIL",
                            "reason": None
                            if variant_ok
                            else "future_suffix_variant_changed_prefix_result",
                        }
                    )
                suffix_all &= bool(sentinel_passed)
            if sentinel_passed is False:
                status, reason = (
                    "INELIGIBLE_FUTURE_SUFFIX_INVARIANCE_FAILED",
                    "future_suffix_sentinel_failed",
                )
                values = {key: None for key in values}
                component_statuses = {
                    f"{name}Status": "INELIGIBLE_FUTURE_SUFFIX_INVARIANCE_FAILED"
                    for name in (
                        "synergy",
                        "downwardCausation",
                        "emergence",
                        "localPhiR",
                    )
                }
            prefix_rows.append(
                {
                    **base,
                    "prefixInputSha256": prefix_hash,
                    "preprocessingSeed": p_seed,
                    "partitionSeed": f_seed,
                    "status": status,
                    "reason": reason,
                    **component_statuses,
                    **values,
                    "exactReplayPassed": replay_ok,
                    "futureSuffixStructuralGatePassed": True,
                    "futureSuffixExecutedSentinelPassed": sentinel_passed,
                }
            )
            diagnostic_rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "datasetRole": dataset_role,
                    "implementationId": implementation.value,
                    "temporalModeId": prefix_mode,
                    "trajectoryId": trajectory_id,
                    "matrixIndex": matrix_index,
                    "rawObservationIndex": endpoint,
                    "generation": int(generation[endpoint]),
                    "molecularStep": int(molecular_step[endpoint]),
                    "status": status,
                    "reason": reason,
                    **values,
                    "componentIdentityMaxAbsError": result_prefix.component_identity_max_abs_error,
                }
            )
            partition_rows.append(
                partition_row(
                    result_prefix,
                    dataset_role=dataset_role,
                    trajectory_id=trajectory_id,
                    matrix_index=matrix_index,
                    mode_id=prefix_mode,
                    fit_kind="past_only_prefix_endpoint",
                    endpoint_index=endpoint,
                    endpoint_generation=int(generation[endpoint]),
                    fit_count=endpoint + 1,
                    input_hash=prefix_hash,
                    preprocessing_seed=p_seed,
                    partition_seed=f_seed,
                    replay_passed=replay_ok,
                )
            )
            if not replay_ok or sentinel_passed is False:
                failures.append(
                    {
                        "failureId": f"{trajectory_id}-{implementation.value}-{endpoint}",
                        "stage": "source_analysis",
                        "datasetRole": dataset_role,
                        "implementationId": implementation.value,
                        "trajectoryId": trajectory_id,
                        "observationIndex": endpoint,
                        "status": status,
                        "reason": reason,
                        "fatal": True,
                    }
                )
    write_parquet(result_root / "full.parquet", full_rows)
    write_parquet(result_root / "prefix.parquet", prefix_rows)
    write_parquet(result_root / "partition.parquet", partition_rows)
    write_parquet(result_root / "diagnostic.parquet", diagnostic_rows)
    write_csv(
        result_root / "suffix.csv",
        suffix_rows,
        [
            "researchStepId",
            "datasetRole",
            "implementationId",
            "trajectoryId",
            "matrixIndex",
            "rawObservationIndex",
            "generation",
            "variant",
            "prefixInputSha256",
            "variantPrefixSha256",
            "exactResultIdentical",
            "status",
            "reason",
        ],
    )
    write_csv(
        result_root / "failures.csv",
        failures,
        [
            "failureId",
            "stage",
            "datasetRole",
            "implementationId",
            "trajectoryId",
            "observationIndex",
            "status",
            "reason",
            "fatal",
        ],
    )
    return {
        "trajectoryId": trajectory_id,
        "matrixIndex": matrix_index,
        "datasetRole": dataset_role,
        "resultRoot": str(result_root),
        "wallSeconds": time.perf_counter() - started_wall,
        "cpuSeconds": time.process_time() - started_cpu,
        "evaluationCount": evaluations,
        "fullRows": len(full_rows),
        "prefixRows": len(prefix_rows),
        "partitionRows": len(partition_rows),
        "diagnosticRows": len(diagnostic_rows),
        "suffixRows": len(suffix_rows),
        "failureRows": len(failures),
        "fullReplayAllPassed": full_replay_all,
        "prefixReplayAllPassed": prefix_replay_all,
        "futureSuffixAllPassed": suffix_all,
    }


def collate_source_results(
    records: list[dict[str, Any]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
]:
    ordered = sorted(
        records, key=lambda item: (item["datasetRole"], item["matrixIndex"])
    )
    roots = [Path(item["resultRoot"]) for item in ordered]
    outputs = (
        ("full.parquet", "full_trajectory_metric_values.parquet"),
        ("prefix.parquet", "prefix_endpoint_metric_values.parquet"),
        ("partition.parquet", "partition_history.parquet"),
        ("diagnostic.parquet", "source_diagnostic_outputs.parquet"),
    )
    for source, target in outputs:
        concat_parquets([root / source for root in roots], STEP_ROOT / target)
    suffix_frames = [pd.read_csv(root / "suffix.csv") for root in roots]
    suffix = pd.concat(suffix_frames, ignore_index=True)
    suffix.to_csv(
        STEP_ROOT / "suffix_invariance_results.csv", index=False, lineterminator="\n"
    )
    failures: list[dict[str, Any]] = []
    for root in roots:
        frame = pd.read_csv(root / "failures.csv")
        if len(frame):
            failures.extend(frame.to_dict("records"))
    return (
        pd.read_parquet(STEP_ROOT / "full_trajectory_metric_values.parquet"),
        pd.read_parquet(STEP_ROOT / "prefix_endpoint_metric_values.parquet"),
        pd.read_parquet(STEP_ROOT / "partition_history.parquet"),
        pd.read_parquet(STEP_ROOT / "source_diagnostic_outputs.parquet"),
        suffix,
        failures,
    )


def metric_column(metric_id: str) -> str:
    return "emergence" if metric_id == "SOURCE_DEFINED_EMERGENCE" else "localPhiR"


def metric_finite_frame(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    result = frame.copy()
    result[value_column] = pd.to_numeric(result[value_column], errors="coerce")
    return result[np.isfinite(result[value_column])].copy()


def summary_flat(summary: Any) -> dict[str, Any]:
    return {
        "definedTrajectoryCount": summary.defined_count,
        "positiveTrajectoryCount": summary.positive_count,
        "ordinaryPositivePBelow0p05Count": summary.ordinary_positive_p_lt_0p05_count,
        "meanTrajectoryCorrelation": summary.mean,
        "medianTrajectoryCorrelation": summary.median,
        "trajectoryBootstrapLower95": summary.bootstrap_lower_95,
        "trajectoryBootstrapUpper95": summary.bootstrap_upper_95,
        "circularShiftPositiveP": summary.circular_shift_positive_p,
        "circularShiftNegativeP": summary.circular_shift_negative_p,
        "effectiveEpisodeCount": summary.effective_episode_count,
        "medianLagOneAutocorrelation": summary.median_lag_one_autocorrelation,
    }


def difference_flat(summary: Any) -> dict[str, Any]:
    return {
        "definedTrajectoryCount": summary.defined_count,
        "positiveMeanDifferenceCount": summary.positive_count,
        "medianTrajectoryMeanDifference": summary.median_mean_difference,
        "medianTrajectoryMedianDifference": summary.median_median_difference,
        "trajectoryBootstrapLower95": summary.bootstrap_lower_95,
        "trajectoryBootstrapUpper95": summary.bootstrap_upper_95,
        "blockAwarePositiveP": summary.block_aware_positive_p,
        "pooledMannWhitneyU": summary.pooled_mann_whitney_u,
        "pooledMannWhitneyP": summary.pooled_mann_whitney_p,
    }


def paper_like_fisher(summary: Any) -> tuple[float | None, float | None, int]:
    directional: list[float] = []
    for trajectory_id, rho in summary.correlations.items():
        p_value = summary.ordinary_p_values[trajectory_id]
        if rho is None or p_value is None:
            continue
        directional.append(p_value / 2 if rho > 0 else 1 - p_value / 2)
    if not directional:
        return None, None, 0
    statistic, p_value = combine_pvalues(directional, method="fisher")
    return float(statistic), float(p_value), len(directional)


def analyze_associations(
    full: pd.DataFrame, prefix: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    retrospective_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    prospective_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {"retrospective": {}, "drift": {}, "prospective": {}}
    roles = (EXPLORATORY_DATASET_ROLE, CONFIRMATION_DATASET_ROLE)
    metrics = ("SOURCE_DEFINED_EMERGENCE", "FROZEN_CORRECTED_LOCAL_PHI_R_COMPARATOR")
    label_specs = (
        (HISTORICAL_LABEL_ID, "historicalLabel"),
        (PAST_ONLY_LABEL_ID, "pastOnlyCosineLabel"),
    )
    for role in roles:
        for implementation in SourceImplementation:
            full_branch = full[
                (full["datasetRole"] == role)
                & (full["implementationId"] == implementation.value)
                & (full["observationKind"] == "post_fission")
            ].copy()
            for metric_id in metrics:
                value_column = metric_column(metric_id)
                for label_id, label_column in label_specs:
                    key = f"{role}|{implementation.value}|{metric_id}|{label_id}"
                    summary = trajectory_association_summary(
                        full_branch,
                        value_column=value_column,
                        label_column=label_column,
                        bootstrap_seed=statistics_seed(
                            key, "retrospective", "bootstrap"
                        ),
                        circular_seed=statistics_seed(key, "retrospective", "circular"),
                    )
                    difference = replicator_drift_summary(
                        full_branch,
                        value_column=value_column,
                        label_column=label_column,
                        bootstrap_seed=statistics_seed(key, "drift", "bootstrap"),
                        permutation_seed=statistics_seed(key, "drift", "circular"),
                    )
                    summaries["retrospective"][key] = summary
                    summaries["drift"][key] = difference
                    fisher_stat, fisher_p, fisher_n = paper_like_fisher(summary)
                    for trajectory_id in sorted(summary.correlations):
                        group = full_branch[
                            full_branch["trajectoryId"] == trajectory_id
                        ]
                        retrospective_rows.append(
                            {
                                "rowType": "TRAJECTORY",
                                "datasetRole": role,
                                "implementationId": implementation.value,
                                "metricId": metric_id,
                                "labelId": label_id,
                                "trajectoryId": trajectory_id,
                                "spearmanRho": summary.correlations[trajectory_id],
                                "ordinaryTwoSidedP": summary.ordinary_p_values[
                                    trajectory_id
                                ],
                                "nEligible": int(
                                    np.isfinite(
                                        pd.to_numeric(
                                            group[value_column], errors="coerce"
                                        )
                                    ).sum()
                                ),
                                "finiteCoverage": float(
                                    np.isfinite(
                                        pd.to_numeric(
                                            group[value_column], errors="coerce"
                                        )
                                    ).mean()
                                ),
                            }
                        )
                        drift_rows.append(
                            {
                                "rowType": "TRAJECTORY",
                                "datasetRole": role,
                                "implementationId": implementation.value,
                                "metricId": metric_id,
                                "labelId": label_id,
                                "trajectoryId": trajectory_id,
                                "replicatorMinusDriftMean": difference.mean_differences.get(
                                    trajectory_id
                                ),
                                "replicatorMinusDriftMedian": difference.median_differences.get(
                                    trajectory_id
                                ),
                            }
                        )
                    finite_coverage = (
                        float(
                            np.isfinite(
                                pd.to_numeric(
                                    full_branch[value_column], errors="coerce"
                                )
                            ).mean()
                        )
                        if len(full_branch)
                        else 0.0
                    )
                    retrospective_rows.append(
                        {
                            "rowType": "SUMMARY",
                            "datasetRole": role,
                            "implementationId": implementation.value,
                            "metricId": metric_id,
                            "labelId": label_id,
                            "trajectoryId": None,
                            "finiteCoverage": finite_coverage,
                            **summary_flat(summary),
                            "paperLikeFisherStatistic": fisher_stat,
                            "paperLikeFisherPositiveDirectionP": fisher_p,
                            "paperLikeFisherTrajectoryCount": fisher_n,
                            "paperLikeDiagnosticsControlClassification": False,
                        }
                    )
                    drift_rows.append(
                        {
                            "rowType": "SUMMARY",
                            "datasetRole": role,
                            "implementationId": implementation.value,
                            "metricId": metric_id,
                            "labelId": label_id,
                            "trajectoryId": None,
                            **difference_flat(difference),
                            "pooledDiagnosticsControlClassification": False,
                        }
                    )

            prefix_branch = prefix[
                (prefix["datasetRole"] == role)
                & (prefix["implementationId"] == implementation.value)
                & (prefix["molecularStep"] >= 256)
            ].copy()
            prefix_estimands = (
                ("current_generation_rho_0", HISTORICAL_LABEL_ID, "historicalLabel"),
                (
                    "next_generation_rho_plus_1",
                    HISTORICAL_LABEL_ID,
                    "nextHistoricalLabel",
                ),
                ("current_generation_rho_0", PAST_ONLY_LABEL_ID, "pastOnlyCosineLabel"),
            )
            for metric_id in metrics:
                value_column = metric_column(metric_id)
                for estimand, label_id, label_column in prefix_estimands:
                    key = f"{role}|{implementation.value}|{metric_id}|{label_id}|{estimand}"
                    summary = trajectory_association_summary(
                        prefix_branch,
                        value_column=value_column,
                        label_column=label_column,
                        bootstrap_seed=statistics_seed(key, "prospective", "bootstrap"),
                        circular_seed=statistics_seed(key, "prospective", "circular"),
                    )
                    summaries["prospective"][key] = summary
                    for trajectory_id in sorted(prefix_branch["trajectoryId"].unique()):
                        group = prefix_branch[
                            prefix_branch["trajectoryId"] == trajectory_id
                        ]
                        eligible = group[
                            (group["status"] == "ELIGIBLE")
                            & np.isfinite(
                                pd.to_numeric(group[value_column], errors="coerce")
                            )
                        ]
                        prospective_rows.append(
                            {
                                "rowType": "TRAJECTORY",
                                "datasetRole": role,
                                "implementationId": implementation.value,
                                "metricId": metric_id,
                                "labelId": label_id,
                                "estimand": estimand,
                                "trajectoryId": trajectory_id,
                                "spearmanRho": summary.correlations.get(
                                    str(trajectory_id)
                                ),
                                "ordinaryTwoSidedP": summary.ordinary_p_values.get(
                                    str(trajectory_id)
                                ),
                                "eligibleCount": len(eligible),
                                "expectedAfterBoundaryCount": len(group),
                                "eligibleCoverage": len(eligible) / len(group)
                                if len(group)
                                else 0.0,
                                "firstEligibleGeneration": int(
                                    eligible["generation"].min()
                                )
                                if len(eligible)
                                else None,
                            }
                        )
                    eligible_all = prefix_branch[
                        (prefix_branch["status"] == "ELIGIBLE")
                        & np.isfinite(
                            pd.to_numeric(prefix_branch[value_column], errors="coerce")
                        )
                    ]
                    coverage = (
                        len(eligible_all) / len(prefix_branch)
                        if len(prefix_branch)
                        else 0.0
                    )
                    firsts = [
                        int(group["generation"].min())
                        for _, group in eligible_all.groupby("trajectoryId")
                        if len(group)
                    ]
                    prospective_rows.append(
                        {
                            "rowType": "SUMMARY",
                            "datasetRole": role,
                            "implementationId": implementation.value,
                            "metricId": metric_id,
                            "labelId": label_id,
                            "estimand": estimand,
                            "trajectoryId": None,
                            "eligibleCount": len(eligible_all),
                            "expectedAfterBoundaryCount": len(prefix_branch),
                            "eligibleCoverage": coverage,
                            "firstEligibleGeneration": float(np.median(firsts))
                            if firsts
                            else None,
                            **summary_flat(summary),
                        }
                    )
    return (
        pd.DataFrame(retrospective_rows),
        pd.DataFrame(prospective_rows),
        pd.DataFrame(drift_rows),
        summaries,
    )


def analyze_temporal_and_spikes(
    full: pd.DataFrame, prefix: pd.DataFrame, partitions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    temporal_rows: list[dict[str, Any]] = []
    spike_rows: list[dict[str, Any]] = []
    summary_map: dict[str, Any] = {}
    for role in (EXPLORATORY_DATASET_ROLE, CONFIRMATION_DATASET_ROLE):
        for implementation in SourceImplementation:
            branch = full[
                (full["datasetRole"] == role)
                & (full["implementationId"] == implementation.value)
            ].copy()
            rows = temporal_structure_rows(branch, value_column="emergence")
            for row in rows:
                row.update(
                    {
                        "datasetRole": role,
                        "implementationId": implementation.value,
                        "metricId": "SOURCE_DEFINED_EMERGENCE",
                    }
                )
                temporal_rows.append(row)
            aggregate = next(row for row in rows if row["rowType"] == "AGGREGATE")
            summary_map[f"{role}|{implementation.value}"] = aggregate
            prefix_partitions = partitions[
                (partitions["datasetRole"] == role)
                & (partitions["implementationId"] == implementation.value)
                & (partitions["fitKind"] == "past_only_prefix_endpoint")
            ]
            for trajectory_id, group in branch.groupby("trajectoryId", sort=True):
                group = group.sort_values("rawObservationIndex")
                values = pd.to_numeric(group["emergence"], errors="coerce").to_numpy(
                    dtype=float
                )
                thresholds = excursion_thresholds(values)
                finite = np.isfinite(values)
                spike_mask = finite & (values > thresholds["positive3Sigma"])
                spike_group = group.loc[spike_mask]
                fission_steps = group.loc[
                    group["observationKind"] == "post_fission", "molecularStep"
                ].to_numpy(dtype=float)
                trajectory_prefix_partitions = prefix_partitions[
                    prefix_partitions["trajectoryId"] == trajectory_id
                ].sort_values("endpointGeneration")
                change_generations: list[int] = []
                prior: tuple[str, str] | None = None
                for row in trajectory_prefix_partitions.itertuples():
                    current = tuple(
                        sorted((str(row.partition1Sha256), str(row.partition2Sha256)))
                    )
                    if prior is not None and current != prior:
                        change_generations.append(int(row.endpointGeneration))
                    prior = current
                nonfinite_indices = group.loc[
                    ~finite | (group["status"] != "ELIGIBLE"), "rawObservationIndex"
                ].to_numpy(dtype=float)
                for spike in spike_group.itertuples():
                    spike_rows.append(
                        {
                            "rowType": "SPIKE",
                            "datasetRole": role,
                            "implementationId": implementation.value,
                            "trajectoryId": trajectory_id,
                            "rawObservationIndex": int(spike.rawObservationIndex),
                            "generation": int(spike.generation),
                            "molecularStep": int(spike.molecularStep),
                            "emergence": float(spike.emergence),
                            "threshold": thresholds["positive3Sigma"],
                            "distanceToNearestFissionMolecularSteps": (
                                float(
                                    np.min(np.abs(fission_steps - spike.molecularStep))
                                )
                                if fission_steps.size
                                else None
                            ),
                            "distanceToNearestPrefixPartitionChangeGenerations": (
                                float(
                                    np.min(
                                        np.abs(
                                            np.asarray(change_generations)
                                            - spike.generation
                                        )
                                    )
                                )
                                if change_generations
                                else None
                            ),
                            "distanceToNearestNonfiniteOrIllConditionedObservation": (
                                float(
                                    np.min(
                                        np.abs(
                                            nonfinite_indices
                                            - spike.rawObservationIndex
                                        )
                                    )
                                )
                                if nonfinite_indices.size
                                else None
                            ),
                        }
                    )
                spike_rows.append(
                    {
                        "rowType": "TRAJECTORY_SUMMARY",
                        "datasetRole": role,
                        "implementationId": implementation.value,
                        "trajectoryId": trajectory_id,
                        "positive3SigmaCount": int(spike_mask.sum()),
                        "negative3SigmaCount": int(
                            np.sum(finite & (values < thresholds["negative3Sigma"]))
                        ),
                        "robustPositiveCount": int(
                            np.sum(finite & (values > thresholds["robustPositive"]))
                        ),
                        "robustNegativeCount": int(
                            np.sum(finite & (values < thresholds["robustNegative"]))
                        ),
                        "prefixPartitionChangeCount": len(change_generations),
                        **{
                            f"threshold_{key}": value
                            for key, value in thresholds.items()
                        },
                    }
                )
    return pd.DataFrame(temporal_rows), pd.DataFrame(spike_rows), summary_map


def partition_labels(row: pd.Series, dimension: int = 99) -> np.ndarray:
    labels = np.full(dimension, -1, dtype=np.int16)
    for value in json.loads(row["partition1Json"]):
        labels[int(value)] = 0
    for value in json.loads(row["partition2Json"]):
        labels[int(value)] = 1
    return labels


def _spike_set(group: pd.DataFrame, column: str) -> set[int]:
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    threshold = excursion_thresholds(values)["positive3Sigma"]
    return set(
        group.loc[
            np.isfinite(values) & (values > threshold), "rawObservationIndex"
        ].astype(int)
    )


def analyze_metric_identity(
    full: pd.DataFrame, prefix: pd.DataFrame, association_summaries: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role in (EXPLORATORY_DATASET_ROLE, CONFIRMATION_DATASET_ROLE):
        for implementation in SourceImplementation:
            for mode_type, frame in (("FULL", full), ("PREFIX_ENDPOINT", prefix)):
                branch = frame[
                    (frame["datasetRole"] == role)
                    & (frame["implementationId"] == implementation.value)
                ].copy()
                if mode_type == "FULL":
                    branch = branch[branch["observationKind"] == "post_fission"]
                else:
                    branch = branch[branch["molecularStep"] >= 256]
                trajectory_rows: list[dict[str, Any]] = []
                direction_changes = 0
                direction_total = 0
                for trajectory_id, group in branch.groupby("trajectoryId", sort=True):
                    finite = group[
                        np.isfinite(pd.to_numeric(group["emergence"], errors="coerce"))
                        & np.isfinite(
                            pd.to_numeric(group["localPhiR"], errors="coerce")
                        )
                    ]
                    emergence = finite["emergence"].to_numpy(dtype=float)
                    local_phi = finite["localPhiR"].to_numpy(dtype=float)
                    rank_mean, rank_shift_fraction = rank_agreement(
                        emergence, local_phi
                    )
                    e_spikes, p_spikes = (
                        _spike_set(finite, "emergence"),
                        _spike_set(finite, "localPhiR"),
                    )
                    union = e_spikes | p_spikes
                    labels = finite["historicalLabel"].astype(float).to_numpy()
                    e_rho, p_rho = (
                        finite_spearman(emergence, labels),
                        finite_spearman(local_phi, labels),
                    )
                    e_rep = emergence[labels == 1]
                    e_drift = emergence[labels == 0]
                    p_rep = local_phi[labels == 1]
                    p_drift = local_phi[labels == 0]
                    e_delta = (
                        float(np.mean(e_rep) - np.mean(e_drift))
                        if e_rep.size and e_drift.size
                        else None
                    )
                    p_delta = (
                        float(np.mean(p_rep) - np.mean(p_drift))
                        if p_rep.size and p_drift.size
                        else None
                    )
                    for left, right in ((e_rho, p_rho), (e_delta, p_delta)):
                        if (
                            left is not None
                            and right is not None
                            and left != 0
                            and right != 0
                        ):
                            direction_total += 1
                            direction_changes += int(np.sign(left) != np.sign(right))
                    item = {
                        "rowType": "TRAJECTORY",
                        "datasetRole": role,
                        "implementationId": implementation.value,
                        "modeType": mode_type,
                        "trajectoryId": trajectory_id,
                        "sharedEligibleCount": len(finite),
                        "spearmanEmergenceVsLocalPhiR": finite_spearman(
                            emergence, local_phi
                        ),
                        "pearsonEmergenceVsLocalPhiR": finite_pearson(
                            emergence, local_phi
                        ),
                        "signAgreement": float(
                            np.mean(np.sign(emergence) == np.sign(local_phi))
                        )
                        if len(finite)
                        else None,
                        "meanRankAgreement": rank_mean,
                        "rankShiftOver10PercentilePointsFraction": rank_shift_fraction,
                        "emergenceSpikeCount": len(e_spikes),
                        "localPhiRSpikeCount": len(p_spikes),
                        "spikeJaccard": len(e_spikes & p_spikes) / len(union)
                        if union
                        else 1.0,
                        "emergenceReplicationRho": e_rho,
                        "localPhiRReplicationRho": p_rho,
                        "replicationAssociationDifference": (
                            e_rho - p_rho
                            if e_rho is not None and p_rho is not None
                            else None
                        ),
                        "emergenceReplicatorMinusDriftMean": e_delta,
                        "localPhiRReplicatorMinusDriftMean": p_delta,
                        "driftReplicationDifference": (
                            e_delta - p_delta
                            if e_delta is not None and p_delta is not None
                            else None
                        ),
                        "partitionIdentity": True,
                    }
                    trajectory_rows.append(item)
                    rows.append(item)
                numeric = pd.DataFrame(trajectory_rows)
                rows.append(
                    {
                        "rowType": "SUMMARY",
                        "datasetRole": role,
                        "implementationId": implementation.value,
                        "modeType": mode_type,
                        "trajectoryId": None,
                        "sharedEligibleCount": int(numeric["sharedEligibleCount"].sum())
                        if len(numeric)
                        else 0,
                        "spearmanEmergenceVsLocalPhiR": finite_spearman(
                            pd.to_numeric(branch["emergence"], errors="coerce"),
                            pd.to_numeric(branch["localPhiR"], errors="coerce"),
                        ),
                        "pearsonEmergenceVsLocalPhiR": finite_pearson(
                            pd.to_numeric(
                                branch["emergence"], errors="coerce"
                            ).to_numpy(),
                            pd.to_numeric(
                                branch["localPhiR"], errors="coerce"
                            ).to_numpy(),
                        ),
                        "signAgreement": safe_nanmedian(numeric["signAgreement"])
                        if len(numeric)
                        else None,
                        "meanRankAgreement": safe_nanmedian(
                            numeric["meanRankAgreement"]
                        )
                        if len(numeric)
                        else None,
                        "rankShiftOver10PercentilePointsFraction": safe_nanmedian(
                            numeric["rankShiftOver10PercentilePointsFraction"]
                        )
                        if len(numeric)
                        else None,
                        "spikeJaccard": safe_nanmedian(numeric["spikeJaccard"])
                        if len(numeric)
                        else None,
                        "replicationAssociationDifference": safe_nanmedian(
                            numeric["replicationAssociationDifference"]
                        )
                        if len(numeric)
                        else None,
                        "driftReplicationDifference": safe_nanmedian(
                            numeric["driftReplicationDifference"]
                        )
                        if len(numeric)
                        else None,
                        "partitionIdentity": True,
                        "conclusionDirectionChangeFraction": direction_changes
                        / direction_total
                        if direction_total
                        else None,
                        "conclusionDirectionChangeCount": direction_changes,
                        "conclusionDirectionComparisonCount": direction_total,
                    }
                )
    return pd.DataFrame(rows)


def analyze_future_dependence(
    full: pd.DataFrame, prefix: pd.DataFrame, partitions: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role in (EXPLORATORY_DATASET_ROLE, CONFIRMATION_DATASET_ROLE):
        for implementation in SourceImplementation:
            full_branch = full[
                (full["datasetRole"] == role)
                & (full["implementationId"] == implementation.value)
                & (full["observationKind"] == "post_fission")
            ][
                [
                    "trajectoryId",
                    "rawObservationIndex",
                    "generation",
                    "emergence",
                    "historicalLabel",
                ]
            ].rename(columns={"emergence": "fullEmergence"})
            prefix_branch = prefix[
                (prefix["datasetRole"] == role)
                & (prefix["implementationId"] == implementation.value)
                & (prefix["molecularStep"] >= 256)
                & (prefix["status"] == "ELIGIBLE")
            ][
                ["trajectoryId", "rawObservationIndex", "generation", "emergence"]
            ].rename(columns={"emergence": "prefixEmergence"})
            shared = full_branch.merge(
                prefix_branch,
                on=["trajectoryId", "rawObservationIndex", "generation"],
                how="inner",
            )
            shared = shared[
                np.isfinite(shared["fullEmergence"])
                & np.isfinite(shared["prefixEmergence"])
            ]
            full_parts = partitions[
                (partitions["datasetRole"] == role)
                & (partitions["implementationId"] == implementation.value)
                & (partitions["fitKind"] == "completed_trajectory")
            ].set_index("trajectoryId")
            prefix_parts = partitions[
                (partitions["datasetRole"] == role)
                & (partitions["implementationId"] == implementation.value)
                & (partitions["fitKind"] == "past_only_prefix_endpoint")
            ].set_index(["trajectoryId", "endpointObservationIndex"])
            trajectory_rows: list[dict[str, Any]] = []
            for trajectory_id, group in shared.groupby("trajectoryId", sort=True):
                full_values = group["fullEmergence"].to_numpy(dtype=float)
                prefix_values = group["prefixEmergence"].to_numpy(dtype=float)
                delta = full_values - prefix_values
                iqr = float(
                    np.quantile(full_values, 0.75) - np.quantile(full_values, 0.25)
                )
                rank_mean, rank_shift_fraction = rank_agreement(
                    full_values, prefix_values
                )
                full_spikes, prefix_spikes = (
                    _spike_set(
                        group.rename(columns={"fullEmergence": "value"}), "value"
                    ),
                    _spike_set(
                        group.rename(columns={"prefixEmergence": "value"}), "value"
                    ),
                )
                union = full_spikes | prefix_spikes
                aris: list[float] = []
                for endpoint in group["rawObservationIndex"].astype(int):
                    try:
                        left = full_parts.loc[trajectory_id]
                        right = prefix_parts.loc[(trajectory_id, endpoint)]
                        aris.append(
                            float(
                                adjusted_rand_score(
                                    partition_labels(left), partition_labels(right)
                                )
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
                labels = group["historicalLabel"].astype(float).to_numpy()
                full_rho = finite_spearman(full_values, labels)
                prefix_rho = finite_spearman(prefix_values, labels)
                item = {
                    "rowType": "TRAJECTORY",
                    "datasetRole": role,
                    "implementationId": implementation.value,
                    "trajectoryId": trajectory_id,
                    "sharedEligibleCount": len(group),
                    "medianAbsoluteDifference": float(np.median(np.abs(delta))),
                    "medianAbsoluteDifferenceDividedByFullIqr": float(
                        np.median(np.abs(delta)) / iqr
                    )
                    if iqr > 0
                    else None,
                    "fullPrefixSpearman": finite_spearman(full_values, prefix_values),
                    "fullPrefixPearson": finite_pearson(full_values, prefix_values),
                    "signAgreement": float(
                        np.mean(np.sign(full_values) == np.sign(prefix_values))
                    ),
                    "meanRankAgreement": rank_mean,
                    "rankShiftOver10PercentilePointsFraction": rank_shift_fraction,
                    "fullSpikeCount": len(full_spikes),
                    "prefixSpikeCount": len(prefix_spikes),
                    "spikeJaccard": len(full_spikes & prefix_spikes) / len(union)
                    if union
                    else 1.0,
                    "medianPartitionAdjustedRandIndex": float(np.median(aris))
                    if aris
                    else None,
                    "fullReplicationRho": full_rho,
                    "prefixReplicationRho": prefix_rho,
                    "replicationAssociationDifference": (
                        full_rho - prefix_rho
                        if full_rho is not None and prefix_rho is not None
                        else None
                    ),
                }
                trajectory_rows.append(item)
                rows.append(item)
            frame = pd.DataFrame(trajectory_rows)
            rows.append(
                {
                    "rowType": "SUMMARY",
                    "datasetRole": role,
                    "implementationId": implementation.value,
                    "trajectoryId": None,
                    "sharedEligibleCount": int(frame["sharedEligibleCount"].sum())
                    if len(frame)
                    else 0,
                    "medianAbsoluteDifference": safe_nanmedian(
                        frame["medianAbsoluteDifference"]
                    )
                    if len(frame)
                    else None,
                    "medianAbsoluteDifferenceDividedByFullIqr": safe_nanmedian(
                        frame["medianAbsoluteDifferenceDividedByFullIqr"]
                    )
                    if len(frame)
                    else None,
                    "fullPrefixSpearman": finite_spearman(
                        shared["fullEmergence"], shared["prefixEmergence"]
                    ),
                    "fullPrefixPearson": finite_pearson(
                        shared["fullEmergence"].to_numpy(),
                        shared["prefixEmergence"].to_numpy(),
                    ),
                    "signAgreement": safe_nanmedian(frame["signAgreement"])
                    if len(frame)
                    else None,
                    "meanRankAgreement": safe_nanmedian(frame["meanRankAgreement"])
                    if len(frame)
                    else None,
                    "rankShiftOver10PercentilePointsFraction": safe_nanmedian(
                        frame["rankShiftOver10PercentilePointsFraction"]
                    )
                    if len(frame)
                    else None,
                    "spikeJaccard": safe_nanmedian(frame["spikeJaccard"])
                    if len(frame)
                    else None,
                    "medianPartitionAdjustedRandIndex": safe_nanmedian(
                        frame["medianPartitionAdjustedRandIndex"]
                    )
                    if len(frame)
                    else None,
                    "replicationAssociationDifference": safe_nanmedian(
                        frame["replicationAssociationDifference"]
                    )
                    if len(frame)
                    else None,
                }
            )
    return pd.DataFrame(rows)


def safe_nanmedian(values: Iterable[Any]) -> float | None:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(
        dtype=float
    )
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else None


def verify_s12c_comparator_replay(
    full: pd.DataFrame, prefix: pd.DataFrame
) -> dict[str, Any]:
    """Require exploratory local-Phi/emergence arrays to replay immutable S12C."""

    old_full = pd.read_parquet(S12C_ROOT / "full_trajectory_local_values.parquet")
    old_prefix = pd.read_parquet(S12C_ROOT / "prefix_endpoint_values.parquet")
    old_diagnostics = pd.read_parquet(S12C_ROOT / "source_diagnostic_outputs.parquet")
    new_full = full[full["datasetRole"] == EXPLORATORY_DATASET_ROLE].copy()
    new_prefix = prefix[prefix["datasetRole"] == EXPLORATORY_DATASET_ROLE].copy()
    mappings = {
        "IIGR_CORRECTED_SOURCE": ("IIGR_FULL", "IIGR_PREFIX_ENDPOINT"),
        "PHIRL_REGULARIZED_SOURCE": ("PHIRL_FULL", "PHIRL_PREFIX_ENDPOINT"),
    }
    checks: list[dict[str, Any]] = []
    for implementation, (old_full_mode, old_prefix_mode) in mappings.items():
        for mode_kind, old, new, old_mode in (
            ("full", old_full, new_full, old_full_mode),
            ("prefix", old_prefix, new_prefix, old_prefix_mode),
        ):
            old_branch = old[
                (old["implementationId"] == implementation)
                & (old["temporalModeId"] == old_mode)
            ][["trajectoryId", "rawObservationIndex", "phiR"]].copy()
            new_branch = new[new["implementationId"] == implementation][
                ["trajectoryId", "rawObservationIndex", "localPhiR"]
            ].copy()
            merged = old_branch.merge(
                new_branch,
                on=["trajectoryId", "rawObservationIndex"],
                how="outer",
                indicator=True,
            ).sort_values(["trajectoryId", "rawObservationIndex"])
            local_match = bool(
                (merged["_merge"] == "both").all()
                and np.array_equal(
                    pd.to_numeric(merged["phiR"], errors="coerce").to_numpy(),
                    pd.to_numeric(merged["localPhiR"], errors="coerce").to_numpy(),
                    equal_nan=True,
                )
            )
            diagnostic = old_diagnostics[
                (old_diagnostics["implementationId"] == implementation)
                & (old_diagnostics["temporalModeId"] == old_mode)
                & (
                    old_diagnostics["diagnosticType"]
                    == "source_named_emergence_synergy_plus_downward_causation"
                )
            ][["trajectoryId", "rawObservationIndex", "diagnosticValue"]].copy()
            diagnostic_keys = diagnostic[
                ["trajectoryId", "rawObservationIndex"]
            ].drop_duplicates()
            new_emergence = new[new["implementationId"] == implementation][
                ["trajectoryId", "rawObservationIndex", "emergence"]
            ].merge(
                diagnostic_keys,
                on=["trajectoryId", "rawObservationIndex"],
                how="inner",
            )
            emergence_merged = diagnostic.merge(
                new_emergence,
                on=["trajectoryId", "rawObservationIndex"],
                how="outer",
                indicator=True,
            ).sort_values(["trajectoryId", "rawObservationIndex"])
            emergence_match = bool(
                (emergence_merged["_merge"] == "both").all()
                and np.array_equal(
                    pd.to_numeric(
                        emergence_merged["diagnosticValue"], errors="coerce"
                    ).to_numpy(),
                    pd.to_numeric(
                        emergence_merged["emergence"], errors="coerce"
                    ).to_numpy(),
                    equal_nan=True,
                )
            )
            checks.append(
                {
                    "implementationId": implementation,
                    "modeKind": mode_kind,
                    "s12cRows": len(old_branch),
                    "s12dRows": len(new_branch),
                    "localPhiRByteValueReplay": local_match,
                    "emergenceByteValueReplay": emergence_match,
                    "success": local_match and emergence_match,
                }
            )
    return {
        "schema": "eidosoma.e01.s12d.s12c_comparator_replay.v1",
        "checks": checks,
        "success": all(item["success"] for item in checks),
    }


def decision_classification(
    retrospective: pd.DataFrame,
    prospective: pd.DataFrame,
    drift: pd.DataFrame,
    summaries: dict[str, Any],
    temporal_map: dict[str, Any],
    worker_records: list[dict[str, Any]],
    suffix: pd.DataFrame,
) -> dict[str, Any]:
    role = CONFIRMATION_DATASET_ROLE
    iigr = SourceImplementation.IIGR.value
    phirl = SourceImplementation.PHIRL.value

    def retrospective_key(implementation: str, metric: str) -> str:
        return f"{role}|{implementation}|{metric}|{HISTORICAL_LABEL_ID}"

    def prospective_key(implementation: str, metric: str) -> str:
        return f"{role}|{implementation}|{metric}|{HISTORICAL_LABEL_ID}|current_generation_rho_0"

    gate_payload: dict[str, Any] = {}
    coherent: dict[str, bool] = {}
    for implementation in (iigr, phirl):
        key = retrospective_key(implementation, "SOURCE_DEFINED_EMERGENCE")
        association = summaries["retrospective"][key]
        difference = summaries["drift"][key]
        branch = retrospective[
            (retrospective["rowType"] == "SUMMARY")
            & (retrospective["datasetRole"] == role)
            & (retrospective["implementationId"] == implementation)
            & (retrospective["metricId"] == "SOURCE_DEFINED_EMERGENCE")
            & (retrospective["labelId"] == HISTORICAL_LABEL_ID)
        ].iloc[0]
        association_gates = association_gate(association, confirmation=True)
        association_gates["finiteCoverageAtLeast0p80"] = (
            float(branch["finiteCoverage"]) >= 0.80
        )
        difference_gates = drift_gate(difference, confirmation=True)
        coherent[implementation] = all(association_gates.values()) and all(
            difference_gates.values()
        )
        gate_payload[f"{implementation}_full"] = {
            "association": association_gates,
            "driftDifference": difference_gates,
            "allPassed": coherent[implementation],
        }

    prospective_pass: dict[str, bool] = {}
    for implementation, other in ((iigr, phirl), (phirl, iigr)):
        key = prospective_key(implementation, "SOURCE_DEFINED_EMERGENCE")
        summary = summaries["prospective"][key]
        row = prospective[
            (prospective["rowType"] == "SUMMARY")
            & (prospective["datasetRole"] == role)
            & (prospective["implementationId"] == implementation)
            & (prospective["metricId"] == "SOURCE_DEFINED_EMERGENCE")
            & (prospective["labelId"] == HISTORICAL_LABEL_ID)
            & (prospective["estimand"] == "current_generation_rho_0")
        ].iloc[0]
        other_summary = summaries["prospective"][
            prospective_key(other, "SOURCE_DEFINED_EMERGENCE")
        ]
        replay = all(
            record["fullReplayAllPassed"] and record["prefixReplayAllPassed"]
            for record in worker_records
        )
        suffix_pass = bool(suffix["exactResultIdentical"].astype(bool).all())
        gates = association_gate(summary, confirmation=True)
        gates.update(
            {
                "eligibleCoverageAtLeast0p80": float(row["eligibleCoverage"]) >= 0.80,
                "exactReplay": replay,
                "exactFutureSuffixInvariance": suffix_pass,
                "noSignificantOppositeOtherImplementation": not significant_opposite(
                    other_summary
                ),
            }
        )
        prospective_pass[implementation] = all(gates.values())
        gate_payload[f"{implementation}_prefix"] = {
            "gates": gates,
            "allPassed": prospective_pass[implementation],
        }

    temporal = temporal_map[f"{role}|{iigr}"]
    punctuated_gates = {
        "runsWithPositive3SigmaAtLeast18": temporal["runsWithPositive3Sigma"] >= 18,
        "aggregateTrendPStrictlyGreaterThan0p05": temporal["aggregateTrendPValue"]
        is not None
        and temporal["aggregateTrendPValue"] > 0.05,
        "rawLjungBoxSignificantRunsAtLeast21": temporal["runsRawLjungBoxSignificant"]
        >= 21,
        "differencedLjungBoxSignificantRunsAtMost0": temporal[
            "runsDifferencedLjungBoxSignificant"
        ]
        <= 0,
    }
    gate_payload["punctuatedRetrospectiveResemblance"] = {
        "gates": punctuated_gates,
        "allPassed": all(punctuated_gates.values()),
        "controlsClassification": False,
    }

    if prospective_pass[iigr]:
        classification = "SOURCE_EMERGENCE_PROSPECTIVE_CANDIDATE"
    elif coherent[iigr]:
        classification = "RETROSPECTIVE_SOURCE_EMERGENCE_RESEMBLANCE"
    elif coherent[phirl]:
        classification = "REGULARIZATION_DEPENDENT_SOURCE_EMERGENCE_RESEMBLANCE"
    else:
        classification = "SOURCE_DEFINED_EMERGENCE_NOT_SUPPORTED"

    existing_role = EXPLORATORY_DATASET_ROLE
    existing_key = (
        f"{existing_role}|{iigr}|SOURCE_DEFINED_EMERGENCE|{HISTORICAL_LABEL_ID}"
    )
    existing_assoc = association_gate(
        summaries["retrospective"][existing_key], confirmation=False
    )
    existing_drift = drift_gate(summaries["drift"][existing_key], confirmation=False)
    comparator_key = retrospective_key(iigr, "FROZEN_CORRECTED_LOCAL_PHI_R_COMPARATOR")
    comparator_assoc = association_gate(
        summaries["retrospective"][comparator_key], confirmation=True
    )
    comparator_drift = drift_gate(summaries["drift"][comparator_key], confirmation=True)
    metric_specific = (
        all(existing_assoc.values())
        and all(existing_drift.values())
        and coherent[iigr]
        and not (all(comparator_assoc.values()) and all(comparator_drift.values()))
    )
    return {
        "schema": "eidosoma.e01.s12d.classification.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "preregistrationVersion": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "sourceRelationship": SOURCE_RELATIONSHIP,
        "classification": classification,
        "genericOutcomeClassification": (
            "supportive" if prospective_pass[iigr] else "constraining/contradictory"
        ),
        "primaryImplementation": iigr,
        "regularizationCompanion": phirl,
        "primaryRetrospectiveCoherent": coherent[iigr],
        "primaryProspectivePassed": prospective_pass[iigr],
        "phirlRetrospectiveCoherent": coherent[phirl],
        "phirlProspectivePassed": prospective_pass[phirl],
        "punctuatedRetrospectiveResemblanceGatePassed": all(punctuated_gates.values()),
        "s12cNegativeClassificationSpecificToIntegratedMetricEstablished": metric_specific,
        "metricSpecificityStatus": (
            "ESTABLISHED_BY_FROZEN_EXISTING_AND_UNTOUCHED_CONFIRMATION_GATES"
            if metric_specific
            else "NOT_ESTABLISHED_UNDER_FROZEN_GATES"
        ),
        "s12cClassificationChanged": False,
        "s12cPreservedClassification": "SOURCE_FAMILY_NOT_SUPPORTED",
        "s13Status": "BLOCKED_PENDING_S12D_HUMAN_REVIEW",
        "gateDetails": gate_payload,
        "interpretationBoundary": (
            "Source-informed metric identity only; completed fits are retrospective, "
            "and neither author/paper identity nor exact GARD replication is claimed."
        ),
    }


def create_figures(
    full: pd.DataFrame,
    prefix: pd.DataFrame,
    retrospective: pd.DataFrame,
    prospective: pd.DataFrame,
    spikes: pd.DataFrame,
    future: pd.DataFrame,
    classification: dict[str, Any],
) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    representative = min(
        full.loc[
            full["datasetRole"] == EXPLORATORY_DATASET_ROLE, "trajectoryId"
        ].unique()
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, implementation in zip(axes, SourceImplementation, strict=True):
        group = full[
            (full["trajectoryId"] == representative)
            & (full["implementationId"] == implementation.value)
        ].sort_values("molecularStep")
        axis.plot(
            group["molecularStep"], group["emergence"], lw=0.8, label="source emergence"
        )
        axis.plot(
            group["molecularStep"],
            group["localPhiR"],
            lw=0.7,
            alpha=0.75,
            label="corrected local Phi-r",
        )
        axis.set_title(implementation.value)
        axis.set_ylabel("local value")
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("molecular step")
    fig.suptitle(f"Exploratory metric identity — {representative}")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "exploratory_integrated_vs_emergence.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    for axis, implementation in zip(axes, SourceImplementation, strict=True):
        branch = full[
            (full["datasetRole"] == CONFIRMATION_DATASET_ROLE)
            & (full["implementationId"] == implementation.value)
        ]
        for _, group in branch.groupby("trajectoryId", sort=True):
            axis.plot(group["molecularStep"], group["emergence"], lw=0.45, alpha=0.45)
        axis.set_title(implementation.value)
        axis.set_ylabel("emergence")
    axes[-1].set_xlabel("molecular step")
    fig.suptitle("Untouched confirmation: completed-trajectory local emergence")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "confirmation_full_emergence.png", dpi=170)
    plt.close(fig)

    trajectory_retro = retrospective[
        (retrospective["rowType"] == "TRAJECTORY")
        & (retrospective["datasetRole"] == CONFIRMATION_DATASET_ROLE)
        & (retrospective["metricId"] == "SOURCE_DEFINED_EMERGENCE")
        & (retrospective["labelId"] == HISTORICAL_LABEL_ID)
    ]
    trajectory_prefix = prospective[
        (prospective["rowType"] == "TRAJECTORY")
        & (prospective["datasetRole"] == CONFIRMATION_DATASET_ROLE)
        & (prospective["metricId"] == "SOURCE_DEFINED_EMERGENCE")
        & (prospective["labelId"] == HISTORICAL_LABEL_ID)
        & (prospective["estimand"] == "current_generation_rho_0")
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    positions, values, labels = [], [], []
    for index, (name, frame) in enumerate(
        (
            (
                "IIGR full",
                trajectory_retro[
                    trajectory_retro["implementationId"]
                    == SourceImplementation.IIGR.value
                ],
            ),
            (
                "IIGR prefix",
                trajectory_prefix[
                    trajectory_prefix["implementationId"]
                    == SourceImplementation.IIGR.value
                ],
            ),
            (
                "PhiRL full",
                trajectory_retro[
                    trajectory_retro["implementationId"]
                    == SourceImplementation.PHIRL.value
                ],
            ),
            (
                "PhiRL prefix",
                trajectory_prefix[
                    trajectory_prefix["implementationId"]
                    == SourceImplementation.PHIRL.value
                ],
            ),
        ),
        start=1,
    ):
        finite = (
            pd.to_numeric(frame["spearmanRho"], errors="coerce").dropna().to_numpy()
        )
        positions.append(index)
        values.append(finite)
        labels.append(name)
        ax.scatter(np.full(finite.size, index), finite, s=18, alpha=0.7)
    ax.boxplot(values, positions=positions, widths=0.5, showfliers=False)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(positions, labels, rotation=15)
    ax.set_ylabel("within-trajectory Spearman rho")
    ax.set_title("Retrospective and prefix association distributions")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "confirmation_association_distributions.png", dpi=170)
    plt.close(fig)

    summary_spikes = spikes[
        (spikes["rowType"] == "TRAJECTORY_SUMMARY")
        & (spikes["datasetRole"] == CONFIRMATION_DATASET_ROLE)
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    for implementation, marker in (
        (SourceImplementation.IIGR.value, "o"),
        (SourceImplementation.PHIRL.value, "s"),
    ):
        branch = summary_spikes[summary_spikes["implementationId"] == implementation]
        ax.scatter(
            branch["trajectoryId"],
            branch["positive3SigmaCount"],
            label=implementation,
            marker=marker,
            alpha=0.75,
        )
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("positive 3-sigma excursion count")
    ax.set_title("Untouched confirmation spike structure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "confirmation_spike_structure.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, implementation in zip(axes, SourceImplementation, strict=True):
        full_post = full[
            (full["datasetRole"] == CONFIRMATION_DATASET_ROLE)
            & (full["implementationId"] == implementation.value)
            & (full["observationKind"] == "post_fission")
        ][["trajectoryId", "rawObservationIndex", "emergence"]].rename(
            columns={"emergence": "full"}
        )
        pref = prefix[
            (prefix["datasetRole"] == CONFIRMATION_DATASET_ROLE)
            & (prefix["implementationId"] == implementation.value)
            & (prefix["status"] == "ELIGIBLE")
        ][["trajectoryId", "rawObservationIndex", "emergence"]].rename(
            columns={"emergence": "prefix"}
        )
        shared = full_post.merge(pref, on=["trajectoryId", "rawObservationIndex"])
        axis.scatter(shared["full"], shared["prefix"], s=5, alpha=0.25)
        axis.set_title(implementation.value)
        axis.set_xlabel("full-fit emergence")
        axis.set_ylabel("prefix emergence")
    fig.suptitle("Full versus past-only prefix emergence")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "full_vs_prefix_emergence.png", dpi=170)
    plt.close(fig)

    gate_groups = classification["gateDetails"]
    labels = list(gate_groups)
    values = [int(bool(gate_groups[label]["allPassed"])) for label in labels]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.55 * len(labels))))
    matrix = np.asarray(values, dtype=float)[:, None]
    ax.imshow(
        matrix,
        cmap=ListedColormap(["#b2182b", "#2166ac"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks([0], ["all frozen gates passed"])
    for row, value in enumerate(values):
        ax.text(
            0,
            row,
            "PASS" if value else "FAIL",
            ha="center",
            va="center",
            color="white",
            weight="bold",
        )
    ax.set_title(f"Final decision matrix: {classification['classification']}")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "final_decision_matrix.png", dpi=170)
    plt.close(fig)


def artifact_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": path.relative_to(STEP_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def source_line_evidence() -> list[dict[str, Any]]:
    config = load_config()
    rows: list[dict[str, Any]] = []
    for implementation in SourceImplementation:
        root = Path(config["sourceSnapshots"][implementation.value]["localCheckout"])
        main = (root / "main.py").read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(main, start=1):
            if line.strip().startswith(
                (
                    'info["synergy"] =',
                    'info["causation"] =',
                    'info["integrated"] =',
                    'info["emergence"] =',
                )
            ):
                rows.append(
                    {
                        "implementationId": implementation.value,
                        "file": str(root / "main.py"),
                        "line": line_number,
                        "text": line.strip(),
                        "fileSha256": sha256_file(root / "main.py"),
                    }
                )
    return rows


def validate_outputs(
    *,
    full: pd.DataFrame,
    prefix: pd.DataFrame,
    partitions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    suffix: pd.DataFrame,
    worker_records: list[dict[str, Any]],
    comparator_replay: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, details: Any) -> None:
        checks.append(
            {"checkId": check_id, "passed": bool(passed), "details": jsonable(details)}
        )

    check(
        "source_metric_identity_40_of_40",
        json.loads((STEP_ROOT / "source_metric_equivalence_summary.json").read_text())[
            "success"
        ],
        json.loads((STEP_ROOT / "source_metric_equivalence_summary.json").read_text()),
    )
    confirmation_manifest = json.loads(
        (STEP_ROOT / "confirmation_trajectory_manifest.json").read_text()
    )
    check(
        "confirmation_exactly_24_complete",
        confirmation_manifest["trajectoryCount"] == 24
        and all(
            item["postFissionCount"] == 100
            for item in confirmation_manifest["trajectories"]
        ),
        {"trajectoryCount": confirmation_manifest["trajectoryCount"]},
    )
    check(
        "confirmation_frozen_before_emergence",
        json.loads((STEP_ROOT / "data_firewall_manifest.json").read_text())["success"],
        json.loads((STEP_ROOT / "data_firewall_manifest.json").read_text()),
    )
    check(
        "seed_firewall",
        confirmation_manifest["seedFirewall"]["success"],
        confirmation_manifest["seedFirewall"],
    )
    check(
        "trajectory_regeneration",
        all(item["success"] for item in confirmation_manifest["regeneration"]),
        confirmation_manifest["regeneration"],
    )
    check(
        "source_worker_count",
        len(worker_records) == 36,
        {"observed": len(worker_records), "expected": 36},
    )
    expected_full = 0
    for item in worker_records:
        expected_full += item["fullRows"]
    check(
        "full_row_cardinality",
        len(full) == expected_full,
        {"observed": len(full), "expected": expected_full},
    )
    check(
        "prefix_row_cardinality",
        len(prefix) == 36 * 2 * 100,
        {"observed": len(prefix), "expected": 7200},
    )
    check(
        "all_metric_rows_status_bearing",
        full[
            [
                "status",
                "synergyStatus",
                "downwardCausationStatus",
                "emergenceStatus",
                "localPhiRStatus",
            ]
        ]
        .notna()
        .all()
        .all()
        and prefix[
            [
                "status",
                "synergyStatus",
                "downwardCausationStatus",
                "emergenceStatus",
                "localPhiRStatus",
            ]
        ]
        .notna()
        .all()
        .all(),
        {
            "fullMissing": int(full["status"].isna().sum()),
            "prefixMissing": int(prefix["status"].isna().sum()),
        },
    )
    check(
        "all_full_and_prefix_replay",
        all(
            item["fullReplayAllPassed"] and item["prefixReplayAllPassed"]
            for item in worker_records
        ),
        {
            "failedWorkers": [
                item["trajectoryId"]
                for item in worker_records
                if not item["fullReplayAllPassed"] or not item["prefixReplayAllPassed"]
            ]
        },
    )
    check(
        "future_suffix_invariance",
        len(suffix) == 36 * 2 * 3 * 3
        and bool(suffix["exactResultIdentical"].astype(bool).all()),
        {
            "rows": len(suffix),
            "failed": int((~suffix["exactResultIdentical"].astype(bool)).sum()),
        },
    )
    check(
        "s12c_comparator_exact_replay", comparator_replay["success"], comparator_replay
    )
    check(
        "full_component_identity",
        bool(
            (
                pd.to_numeric(
                    diagnostics["componentIdentityMaxAbsError"], errors="coerce"
                ).fillna(0)
                <= 1e-12
            ).all()
        ),
        {
            "maximum": float(
                pd.to_numeric(
                    diagnostics["componentIdentityMaxAbsError"], errors="coerce"
                )
                .fillna(0)
                .max()
            )
        },
    )
    check(
        "dataset_role_separation",
        set(full["datasetRole"])
        == {EXPLORATORY_DATASET_ROLE, CONFIRMATION_DATASET_ROLE}
        and set(prefix["datasetRole"])
        == {EXPLORATORY_DATASET_ROLE, CONFIRMATION_DATASET_ROLE},
        {
            "fullRoles": sorted(full["datasetRole"].unique()),
            "prefixRoles": sorted(prefix["datasetRole"].unique()),
        },
    )
    check(
        "primary_iigr_hierarchy_preserved",
        classification["primaryImplementation"] == SourceImplementation.IIGR.value,
        classification["primaryImplementation"],
    )
    check(
        "source_relationship_exact",
        set(full["sourceRelationship"]) == {SOURCE_RELATIONSHIP}
        and set(prefix["sourceRelationship"]) == {SOURCE_RELATIONSHIP},
        {"label": SOURCE_RELATIONSHIP},
    )
    check(
        "s13_blocked",
        classification["s13Status"] == "BLOCKED_PENDING_S12D_HUMAN_REVIEW",
        classification["s13Status"],
    )
    for filename, required_columns in REQUIRED_TABLE_SCHEMAS.items():
        frame = pd.read_parquet(STEP_ROOT / filename)
        missing = sorted(set(required_columns) - set(frame.columns))
        check(
            f"schema_{filename}",
            not missing,
            {"missingColumns": missing, "rows": len(frame)},
        )
    for filename, required_columns in CONFIRMATION_TABLE_SCHEMAS.items():
        frame = pd.read_parquet(STEP_ROOT / filename)
        missing = sorted(set(required_columns) - set(frame.columns))
        check(
            f"schema_{filename}",
            not missing,
            {"missingColumns": missing, "rows": len(frame)},
        )
    check(
        "partition_rows_status_bearing",
        partitions["status"].notna().all(),
        {"rows": len(partitions)},
    )
    lock = verify_lock()
    check("implementation_lock_and_prior_immutability", lock["success"], lock)
    success = all(item["passed"] for item in checks)
    return {
        "schema": "eidosoma.e01.s12d.validation_summary.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "preregistrationVersion": VERSION,
        "checks": checks,
        "passedChecks": sum(item["passed"] for item in checks),
        "totalChecks": len(checks),
        "success": success,
    }


def scope_compliance(classification: dict[str, Any]) -> dict[str, Any]:
    forbidden_artifacts = [
        ARTIFACTS / "research_steps/S13",
        STEP_ROOT / "intervention_trajectories.parquet",
        STEP_ROOT / "mlp_results.csv",
        STEP_ROOT / "reinforcement_learning_results.csv",
    ]
    checks = {
        "exactNewGardTrajectoryCount24": json.loads(
            (STEP_ROOT / "confirmation_trajectory_manifest.json").read_text()
        )["trajectoryCount"]
        == 24,
        "interventionTrajectoryCount0": not (
            STEP_ROOT / "intervention_trajectories.parquet"
        ).exists(),
        "noMlp": not (STEP_ROOT / "mlp_results.csv").exists(),
        "noReinforcementLearning": not (
            STEP_ROOT / "reinforcement_learning_results.csv"
        ).exists(),
        "noBioModels": not (STEP_ROOT / "biomodels").exists(),
        "noEstimatorRepair": not (STEP_ROOT / "estimator_repair").exists(),
        "s13NotStartedByS12D": not (ARTIFACTS / "research_steps/S13").exists(),
        "s12cClassificationUnchanged": classification["s12cClassificationChanged"]
        is False,
        "s13Blocked": classification["s13Status"]
        == "BLOCKED_PENDING_S12D_HUMAN_REVIEW",
        "sourceRelationshipExact": classification["sourceRelationship"]
        == SOURCE_RELATIONSHIP,
    }
    return {
        "schema": "eidosoma.e01.s12d.scope_compliance.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "checks": checks,
        "forbiddenArtifactsChecked": [str(path) for path in forbidden_artifacts],
        "success": all(checks.values()),
    }


def report_markdown(
    *,
    classification: dict[str, Any],
    validation: dict[str, Any],
    runtime: dict[str, Any],
    retrospective: pd.DataFrame,
    prospective: pd.DataFrame,
    drift: pd.DataFrame,
    temporal: pd.DataFrame,
    metric_identity: pd.DataFrame,
    future: pd.DataFrame,
    failures: pd.DataFrame,
) -> str:
    role = CONFIRMATION_DATASET_ROLE
    iigr = SourceImplementation.IIGR.value
    phirl = SourceImplementation.PHIRL.value

    def summary_row(frame: pd.DataFrame, **filters: Any) -> pd.Series:
        selected = frame.copy()
        for column, value in filters.items():
            selected = selected[selected[column] == value]
        return selected.iloc[0]

    iigr_full = summary_row(
        retrospective,
        rowType="SUMMARY",
        datasetRole=role,
        implementationId=iigr,
        metricId="SOURCE_DEFINED_EMERGENCE",
        labelId=HISTORICAL_LABEL_ID,
    )
    phirl_full = summary_row(
        retrospective,
        rowType="SUMMARY",
        datasetRole=role,
        implementationId=phirl,
        metricId="SOURCE_DEFINED_EMERGENCE",
        labelId=HISTORICAL_LABEL_ID,
    )
    iigr_drift = summary_row(
        drift,
        rowType="SUMMARY",
        datasetRole=role,
        implementationId=iigr,
        metricId="SOURCE_DEFINED_EMERGENCE",
        labelId=HISTORICAL_LABEL_ID,
    )
    phirl_drift = summary_row(
        drift,
        rowType="SUMMARY",
        datasetRole=role,
        implementationId=phirl,
        metricId="SOURCE_DEFINED_EMERGENCE",
        labelId=HISTORICAL_LABEL_ID,
    )
    iigr_prefix = summary_row(
        prospective,
        rowType="SUMMARY",
        datasetRole=role,
        implementationId=iigr,
        metricId="SOURCE_DEFINED_EMERGENCE",
        labelId=HISTORICAL_LABEL_ID,
        estimand="current_generation_rho_0",
    )
    phirl_prefix = summary_row(
        prospective,
        rowType="SUMMARY",
        datasetRole=role,
        implementationId=phirl,
        metricId="SOURCE_DEFINED_EMERGENCE",
        labelId=HISTORICAL_LABEL_ID,
        estimand="current_generation_rho_0",
    )
    temporal_iigr = summary_row(
        temporal,
        rowType="AGGREGATE",
        datasetRole=role,
        implementationId=iigr,
    )
    identity_iigr = summary_row(
        metric_identity,
        rowType="SUMMARY",
        datasetRole=role,
        implementationId=iigr,
        modeType="FULL",
    )
    future_iigr = summary_row(
        future,
        rowType="SUMMARY",
        datasetRole=role,
        implementationId=iigr,
    )
    eq = json.loads((STEP_ROOT / "source_metric_equivalence_summary.json").read_text())
    manifest = json.loads(
        (STEP_ROOT / "confirmation_trajectory_manifest.json").read_text()
    )
    artifacts = (
        load_config()["requiredArtifacts"]["files"]
        + load_config()["requiredArtifacts"]["figures"]
    )
    outcome = classification["classification"]

    def fmt(value: Any, digits: int = 6) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "NA"
        return f"{numeric:.{digits}g}" if np.isfinite(numeric) else "NA"

    def as_int(value: Any) -> str:
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return "NA"

    recommended = (
        "Keep S13 blocked and return for human review. A separately authorized baseline-only follow-up may be considered only if the human judges the prospective source-emergence evidence sufficient."
        if outcome == "SOURCE_EMERGENCE_PROSPECTIVE_CANDIDATE"
        else "Keep S13 blocked and return for human review; do not repair, intervene, predict, or scale up automatically."
    )
    caveat = "The public-source metric is source-informed only. Completed-trajectory values use future-fitted partitions and Gaussian distributions; S12C remains unchanged; the unchanged historical GARD reconstruction is not the unavailable author implementation."
    return f"""# S12D full results — source-defined causal emergence

## Concise top summary

- **Research step ID:** `{VERSION}` (S12D; step number 12D).
- **Completion status:** `COMPLETED_BOUNDED_METRIC_IDENTITY_AUDIT`; S13 remains `BLOCKED_PENDING_S12D_HUMAN_REVIEW`.
- **Artifacts written:** {len(artifacts)} preregistered report/data/figure paths under `{STEP_ROOT}`; the complete path/hash inventory is in `artifact_manifest.json`.
- **Validation result:** {"PASS" if validation["success"] else "FAIL"} — {validation["passedChecks"]}/{validation["totalChecks"]} final validation checks passed; source-metric identity passed {eq["passedRows"]}/{eq["expectedRows"]}; 24/24 confirmation trajectories had 100 fissions and the three frozen regeneration cases passed.
- **Outcome classification:** `{outcome}`; generic evidence outcome `{classification["genericOutcomeClassification"]}`.
- **Caveats or blockers:** {caveat}
- **Lay summary:** We tested the quantity the public programs literally call “emergence”—synergy plus downward causation—rather than retrospectively replacing S12C's distinct integrated Phi-r result. The decisive evidence comes from 24 fresh simulation runs whose data were generated and sealed before emergence was calculated. The classification above follows the preregistered trajectory-level tests, while full-run curves remain retrospective descriptions.
- **Recommended next action:** {recommended}

## Frozen question and outcome

S12D asked whether changing only the source-selected scalar from corrected `local_phi_r` to exact `synergy + downward_causation` changes the scientific conclusion, whether the resulting series is higher during historical H>0.9 replication, whether it is punctuated, whether it survives past-only prefix fitting, and whether it repeats on new matrices. The frozen decision is **`{outcome}`**. S12C's `SOURCE_FAMILY_NOT_SUPPORTED` result was not edited or reclassified. Metric specificity was **`{classification["metricSpecificityStatus"]}`**.

## Inputs and provenance

- Pinned IIGR commit: `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`.
- Pinned PhiRL commit: `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`; regularization ancestor `9030b598f436cd23c39a3c3fc312ff79c79fb2ad`.
- Audited safe lattice SHA-256: `74ecca37f04201088d76a9e8ede7efe04bafebecff85a4882a44f03afbd23aa1`; no pickle was loaded during GARD analysis.
- Immutable existing development inputs: exactly twelve S12 baselines, labeled `EXPLORATORY_EXISTING_TRAJECTORIES`.
- Untouched confirmation inputs: exactly {manifest["trajectoryCount"]} domain-separated matrices and trajectories, each with {manifest["fissionsPerTrajectory"]} fissions, generated under root SHA-256 `{manifest["rootSeedSha256"]}` and labeled `UNTOUCHED_CONFIRMATION_TRAJECTORIES`.
- Common substrate: additive 0.5 closure, 100-component CLR, then removal of original component 100 to 99 dimensions. Labels were only `HISTORICAL_H090_REPLICATOR` and `PAST_ONLY_COSINE_REPLICATOR`.
- Evidence relationship: `{SOURCE_RELATIONSHIP}`. This is neither author-primary, paper-primary, exact-author implementation, nor exact-GARD replication evidence.

## Methods

### Preregistration and firewall

The complete design, scalar hierarchy, thresholds, 4,096-replicate bootstrap/circular-shift tests, seed root, source fixtures, temporal modes, schemas, runtime ceilings, and classification hierarchy were frozen, validated, committed, and pushed before trajectory-level outcome access. Source identity was then tested on all 14 S12C development fixtures, all 14 S12C confirmation fixtures, and six new fixtures for each of two implementations: 40 rows total. Only after all rows passed were all 24 confirmation trajectories generated, labeled, preprocessed, hashed, and manifested. Emergence was not computed until `data_firewall_manifest.json` existed.

### Exact metric

For every local two-variable PhiID lattice, S12D retained `S = Pi[01->01]`, `D = Pi[01->0] + Pi[01->1]`, and `E = S + D` exactly. It also retained corrected `local_phi_r` as the frozen comparator. Values were neither clipped nor imputed. The wrapper used the S12C-confirmed source call ordering and safe JSON lattice; isolated original-source processes received synthetic fixtures only.

### Temporal modes and statistics

Full mode fit preprocessing, Fiedler partition, means, and covariance to each completed trajectory once and labeled every value `RETROSPECTIVE_FULL_TRAJECTORY_LOCAL`. Prefix mode refit the same implementation from observation zero through each post-fission endpoint after 256 molecular transitions and retained only its last local value. Every prefix was independently replayed; first/middle/last sentinels per trajectory were invariant to suffix deletion, deterministic shuffle, and domain-separated replacement. Primary inference used within-trajectory Spearman correlations followed by a trajectory bootstrap; a within-trajectory circular label shift preserved temporal structure. Replicator-minus-drift means used the same trajectory-level bootstrap and shift null. Pooled Mann–Whitney and Fisher combinations are secondary diagnostics only.

## Commands

```bash
python scripts/e01/freeze_s12d_preregistration.py --action freeze
pytest -q tests/e01/test_s12d_source_emergence_metric_identity.py
ruff check src/e01_source_emergence_metric_identity scripts/e01/freeze_s12d_preregistration.py scripts/e01/run_s12d_source_emergence_metric_identity.py scripts/e01/s12d_original_source_metric_adapter.py tests/e01/test_s12d_source_emergence_metric_identity.py
git commit ... && git push origin eidosoma/groups/42
python scripts/e01/freeze_s12d_preregistration.py --action lock
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/e01/run_s12d_source_emergence_metric_identity.py --workers 6
```

## Source-metric equivalence result

All **{eq["passedRows"]}/{eq["expectedRows"]}** rows passed identical status, availability, array length, nonfinite mask, exact same-seed replay, canonical tuple serialization, and a component tolerance of `1e-12`. The maximum observed component difference was `{fmt(eq["maximumObservedComponentDifference"])}`. The exact formula `emergence == synergy + downward_causation` held on both the pinned source and wrapper sides.

## Untouched retrospective confirmation

| Branch | Finite coverage | Defined / positive trajectories | Median rho | 95% trajectory bootstrap | Circular-shift p+ | Positive mean differences | Median replicator-minus-drift mean | Block-aware p+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IIGR primary | {fmt(iigr_full["finiteCoverage"])} | {as_int(iigr_full["definedTrajectoryCount"])} / {as_int(iigr_full["positiveTrajectoryCount"])} | {fmt(iigr_full["medianTrajectoryCorrelation"])} | [{fmt(iigr_full["trajectoryBootstrapLower95"])}, {fmt(iigr_full["trajectoryBootstrapUpper95"])}] | {fmt(iigr_full["circularShiftPositiveP"])} | {as_int(iigr_drift["positiveMeanDifferenceCount"])} | {fmt(iigr_drift["medianTrajectoryMeanDifference"])} | {fmt(iigr_drift["blockAwarePositiveP"])} |
| PhiRL robustness | {fmt(phirl_full["finiteCoverage"])} | {as_int(phirl_full["definedTrajectoryCount"])} / {as_int(phirl_full["positiveTrajectoryCount"])} | {fmt(phirl_full["medianTrajectoryCorrelation"])} | [{fmt(phirl_full["trajectoryBootstrapLower95"])}, {fmt(phirl_full["trajectoryBootstrapUpper95"])}] | {fmt(phirl_full["circularShiftPositiveP"])} | {as_int(phirl_drift["positiveMeanDifferenceCount"])} | {fmt(phirl_drift["medianTrajectoryMeanDifference"])} | {fmt(phirl_drift["blockAwarePositiveP"])} |

IIGR's frozen full association-plus-drift gate was `{classification["gateDetails"][iigr + "_full"]["allPassed"]}`; PhiRL's was `{classification["gateDetails"][phirl + "_full"]["allPassed"]}`. PhiRL remained a robustness companion and was never promoted over IIGR.

## Punctuated structure

The untouched IIGR series had {int(temporal_iigr["runsWithPositive3Sigma"])}/24 trajectories with at least one positive within-run 3-sigma excursion, {int(temporal_iigr["runsRawLjungBoxSignificant"])}/24 significant raw-series Ljung–Box tests, and {int(temporal_iigr["runsDifferencedLjungBoxSignificant"])}/24 significant differenced-series tests. The aggregate relative-time slope was `{temporal_iigr["aggregateTrendSlope"]}` with p=`{temporal_iigr["aggregateTrendPValue"]}`. The separately frozen punctuated-resemblance gate was `{classification["punctuatedRetrospectiveResemblanceGatePassed"]}` and did not override association inference. Spike widths, prominence, spacing, and proximity to fission, prefix-partition change, and source nonfinite states are preserved in `temporal_structure.csv` and `spike_analysis.csv`.

## Prospective prefix result

| Branch | Eligible coverage | Defined / positive trajectories | Median rho current generation | 95% trajectory bootstrap | Circular-shift p+ | Median first eligible generation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IIGR primary | {fmt(iigr_prefix["eligibleCoverage"])} | {as_int(iigr_prefix["definedTrajectoryCount"])} / {as_int(iigr_prefix["positiveTrajectoryCount"])} | {fmt(iigr_prefix["medianTrajectoryCorrelation"])} | [{fmt(iigr_prefix["trajectoryBootstrapLower95"])}, {fmt(iigr_prefix["trajectoryBootstrapUpper95"])}] | {fmt(iigr_prefix["circularShiftPositiveP"])} | {fmt(iigr_prefix["firstEligibleGeneration"])} |
| PhiRL robustness | {fmt(phirl_prefix["eligibleCoverage"])} | {as_int(phirl_prefix["definedTrajectoryCount"])} / {as_int(phirl_prefix["positiveTrajectoryCount"])} | {fmt(phirl_prefix["medianTrajectoryCorrelation"])} | [{fmt(phirl_prefix["trajectoryBootstrapLower95"])}, {fmt(phirl_prefix["trajectoryBootstrapUpper95"])}] | {fmt(phirl_prefix["circularShiftPositiveP"])} | {fmt(phirl_prefix["firstEligibleGeneration"])} |

The IIGR prospective gate was `{classification["primaryProspectivePassed"]}`. Current-generation historical labels were the primary estimand; next-generation historical and current-generation past-only cosine results remain separate rows in `prospective_associations.csv`. No MLP was trained.

## Metric-identity comparison

On untouched IIGR full post-fission values, emergence versus corrected local Phi-r had Spearman `{identity_iigr["spearmanEmergenceVsLocalPhiR"]}`, Pearson `{identity_iigr["pearsonEmergenceVsLocalPhiR"]}`, sign agreement `{identity_iigr["signAgreement"]}`, median spike Jaccard `{identity_iigr["spikeJaccard"]}`, median replication-association difference `{identity_iigr["replicationAssociationDifference"]}`, and conclusion-direction change fraction `{identity_iigr["conclusionDirectionChangeFraction"]}`. Partitions were identical by construction because only the selected scalar changed. The frozen conclusion that S12C's negative result was specifically caused by selecting `integrated` rather than `emergence` is **{classification["metricSpecificityStatus"]}**. S12C itself remains unchanged.

## Full-versus-prefix future-dependence audit

For untouched IIGR, the trajectory-median absolute full-prefix difference was `{future_iigr["medianAbsoluteDifference"]}`, normalized median difference `{future_iigr["medianAbsoluteDifferenceDividedByFullIqr"]}`, pooled Spearman `{future_iigr["fullPrefixSpearman"]}`, sign agreement `{future_iigr["signAgreement"]}`, median fraction of >10-percentile rank shifts `{future_iigr["rankShiftOver10PercentilePointsFraction"]}`, spike Jaccard `{future_iigr["spikeJaccard"]}`, median partition ARI `{future_iigr["medianPartitionAdjustedRandIndex"]}`, and median change in replication association `{future_iigr["replicationAssociationDifference"]}`. Any completed-fit relationship that lacks a prefix counterpart is retrospective and potentially future-dependent.

## Validation, runtime, and storage

Final validation passed **{validation["passedChecks"]}/{validation["totalChecks"]}** checks. Exact source and wrapper replay, every full/prefix replay, {len(pd.read_csv(STEP_ROOT / "suffix_invariance_results.csv"))} suffix tests, frozen S12C comparator replay, all source identities, prior-artifact immutability, 24-by-100 trajectory completeness, three same-engine regenerations, seed uniqueness/non-overlap, schema/status completeness, scope, runtime, storage, hashes, reports, and figures were audited. There were {len(failures)} retained failure-ledger rows. Runtime was `{fmt(runtime["wallHours"], 4)}` wall-hours and `{fmt(runtime["workerCpuHours"], 4)}` summed worker CPU-hours using six source workers and one-thread BLAS/OpenMP; CPU float64 was authoritative and GPU use was zero.

## Artifacts and schemas

Full local values, prefix endpoints, partitions, and diagnostic components are losslessly stored in Parquet. Confirmation matrices use compressed NPZ without pickle. CSV summaries retain undefined statistics as empty fields and all raw scientific values have an explicit status/reason. `artifact_manifest.json` records SHA-256 and bytes for every collectible artifact except its explicitly declared self-exclusion. The named `S12D_FULL_RESULTS.md` and canonical `research_step_full_results.md` are byte-identical.

## Caveats, blockers, and limitations

- Completed-fit estimates can depend on future observations through preprocessing, partition, and Gaussian parameters; they cannot establish prospective warning or causal control.
- The exact public-source metric identity does not identify the unavailable author GARD code, author randomness, or unpublished preprocessing.
- The frozen confirmation deliberately holds the S12 historical reconstruction fixed; alternate Poisson dynamics, binomial fission, or other GARD semantics were not tested.
- Historical H>0.9 is the primary paper-comparison label but remains a source-traceable retrospective/local rule, not author-code identity. Past-only cosine is secondary.
- PhiRL regularization is a prespecified robustness companion. A favorable PhiRL-only result is regularization-dependent and cannot replace IIGR.
- S12C's failure classification, the S11/S11R failures, S12's 59-claim matrix, and all prior evidence remain byte-exact. S12D does not rescue, overturn, or substitute for them.
- No intervention, prediction, estimator repair, reinforcement learning, BioModels analysis, paper-text dynamics, 100-matrix scale-up, or S13 execution occurred.

## Provenance and source relationship

`source_snapshot_manifest.json`, `implementation_lock.json`, `immutable_prior_audit.json`, `confirmation_seed_manifest.parquet`, `confirmation_trajectory_manifest.json`, `data_firewall_manifest.json`, `runtime_manifest.json`, and `artifact_manifest.json` provide the exact source commits, file hashes, method-lock commit, prior directory identities, nine-stream identities, generation hashes, runtime, and output hashes. The source evidence lines that separately assign `integrated` and `emergence` are recorded in `source_snapshot_manifest.json`. The only permitted relationship label is `{SOURCE_RELATIONSHIP}`.

## Recommended next action

{recommended} Do not begin S13 automatically.
"""


def finalize_success(
    *,
    full: pd.DataFrame,
    prefix: pd.DataFrame,
    partitions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    suffix: pd.DataFrame,
    worker_records: list[dict[str, Any]],
    retrospective: pd.DataFrame,
    prospective: pd.DataFrame,
    drift: pd.DataFrame,
    temporal: pd.DataFrame,
    spikes: pd.DataFrame,
    metric_identity: pd.DataFrame,
    future: pd.DataFrame,
    classification: dict[str, Any],
    failures: list[dict[str, Any]],
    runtime_started: float,
    generation_manifest: dict[str, Any],
    benchmark: dict[str, Any],
) -> None:
    comparator_replay = verify_s12c_comparator_replay(full, prefix)
    write_json(
        STEP_ROOT / "replay_validation.json",
        {
            "schema": "eidosoma.e01.s12d.replay_validation.v1",
            "sourceMetricEquivalence": json.loads(
                (STEP_ROOT / "source_metric_equivalence_summary.json").read_text()
            ),
            "s12cComparatorReplay": comparator_replay,
            "trajectoryRegeneration": generation_manifest["regeneration"],
            "fullReplayAllPassed": all(
                item["fullReplayAllPassed"] for item in worker_records
            ),
            "prefixReplayAllPassed": all(
                item["prefixReplayAllPassed"] for item in worker_records
            ),
            "suffixInvarianceAllPassed": bool(
                suffix["exactResultIdentical"].astype(bool).all()
            ),
            "success": comparator_replay["success"]
            and all(
                item["fullReplayAllPassed"] and item["prefixReplayAllPassed"]
                for item in worker_records
            )
            and bool(suffix["exactResultIdentical"].astype(bool).all()),
        },
    )
    write_json(STEP_ROOT / "classification.json", classification)
    scope = scope_compliance(classification)
    write_json(STEP_ROOT / "scope_compliance.json", scope)
    failure_columns = [
        "failureId",
        "stage",
        "datasetRole",
        "implementationId",
        "trajectoryId",
        "observationIndex",
        "status",
        "reason",
        "fatal",
        "count",
    ]
    status_counts = pd.concat(
        [
            full.assign(table="full")
            .groupby(["datasetRole", "implementationId", "status"], dropna=False)
            .size()
            .reset_index(name="count"),
            prefix.assign(table="prefix")
            .groupby(["datasetRole", "implementationId", "status"], dropna=False)
            .size()
            .reset_index(name="count"),
        ],
        ignore_index=True,
    )
    for index, row in status_counts.iterrows():
        if row["status"] != "ELIGIBLE":
            failures.append(
                {
                    "failureId": f"STATUS-SUMMARY-{index:04d}",
                    "stage": "status_summary",
                    "datasetRole": row["datasetRole"],
                    "implementationId": row["implementationId"],
                    "trajectoryId": None,
                    "observationIndex": None,
                    "status": row["status"],
                    "reason": "status_bearing_ineligible_or_partial_output_retained",
                    "fatal": False,
                    "count": int(row["count"]),
                }
            )
    write_csv(STEP_ROOT / "failure_ledger.csv", failures, failure_columns)
    failure_frame = pd.DataFrame(failures, columns=failure_columns)

    worker_cpu = sum(item["cpuSeconds"] for item in worker_records) + sum(
        item["cpuSeconds"] for item in generation_manifest["trajectories"]
    )
    wall_seconds = time.perf_counter() - runtime_started
    runtime = {
        "schema": "eidosoma.e01.s12d.runtime_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "startedAtUtcApprox": datetime.fromtimestamp(
            time.time() - wall_seconds, tz=timezone.utc
        ).isoformat(),
        "completedAtUtc": datetime.now(timezone.utc).isoformat(),
        "wallSeconds": wall_seconds,
        "wallHours": wall_seconds / 3600,
        "workerCpuSeconds": worker_cpu,
        "workerCpuHours": worker_cpu / 3600,
        "gpuHours": 0.0,
        "gpuUsed": False,
        "cpuFloat64Authoritative": True,
        "sourceWorkers": 6,
        "statisticsWorkers": 1,
        "orchestrationCores": 1,
        "threadEnvironment": load_config()["runtimeAndStorage"]["threadEnvironment"],
        "benchmark": benchmark,
        "workerRecords": worker_records,
        "hardCeilings": {
            "cpuHours": 80.0,
            "gpuHours": 0.0,
            "wallHours": 24.0,
            "newArtifactBytes": 10737418240,
        },
        "withinRuntimeCeilings": worker_cpu / 3600 <= 80 and wall_seconds / 3600 <= 24,
    }
    write_json(STEP_ROOT / "runtime_manifest.json", runtime)
    storage_bytes_before_reports = directory_bytes(STEP_ROOT)
    storage = {
        "schema": "eidosoma.e01.s12d.storage_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "bytesBeforeReportsAndManifest": storage_bytes_before_reports,
        "hardCeilingBytes": 10737418240,
        "withinCeiling": storage_bytes_before_reports <= 10737418240,
        "cacheRoot": str(CACHE_ROOT),
        "cacheExcludedFromCollectibleArtifacts": True,
    }
    write_json(STEP_ROOT / "storage_validation.json", storage)
    validation = validate_outputs(
        full=full,
        prefix=prefix,
        partitions=partitions,
        diagnostics=diagnostics,
        suffix=suffix,
        worker_records=worker_records,
        comparator_replay=comparator_replay,
        classification=classification,
    )
    validation["scopeCompliancePassed"] = scope["success"]
    validation["runtimeCeilingsPassed"] = runtime["withinRuntimeCeilings"]
    validation["storageCeilingPassed"] = storage["withinCeiling"]
    validation["success"] = (
        validation["success"]
        and scope["success"]
        and runtime["withinRuntimeCeilings"]
        and storage["withinCeiling"]
    )
    write_json(STEP_ROOT / "validation_summary.json", validation)
    if not validation["success"]:
        raise RuntimeError(
            "S12D final validation failed; outputs preserved for fail-closed review"
        )

    status = {
        "researchStepId": RESEARCH_STEP_ID,
        "stepNumber": "12D",
        "success": True,
        "status": "COMPLETED_BOUNDED_METRIC_IDENTITY_AUDIT",
        "artifactsWritten": load_config()["requiredArtifacts"]["files"]
        + load_config()["requiredArtifacts"]["figures"],
        "validationResult": f"PASS: {validation['passedChecks']}/{validation['totalChecks']} final checks plus scope/runtime/storage",
        "outcomeClassification": classification["classification"],
        "genericOutcomeClassification": classification["genericOutcomeClassification"],
        "caveatsOrBlockers": [
            "SOURCE_INFORMED_METRIC_IDENTITY only; no author/paper-primary identity",
            "completed-trajectory values are retrospective and may be future-dependent",
            "S12C and all prior evidence remain immutable",
            "S13 remains BLOCKED_PENDING_S12D_HUMAN_REVIEW",
        ],
        "recommendedNextAction": "Return for mandatory human review; do not begin S13 or any further repair, intervention, prediction, or scale-up.",
    }
    write_json(STEP_ROOT / "status.json", status)
    report = report_markdown(
        classification=classification,
        validation=validation,
        runtime=runtime,
        retrospective=retrospective,
        prospective=prospective,
        drift=drift,
        temporal=temporal,
        metric_identity=metric_identity,
        future=future,
        failures=failure_frame,
    )
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    shutil.copyfile(
        STEP_ROOT / "research_step_full_results.md", STEP_ROOT / "S12D_FULL_RESULTS.md"
    )
    if sha256_file(STEP_ROOT / "research_step_full_results.md") != sha256_file(
        STEP_ROOT / "S12D_FULL_RESULTS.md"
    ):
        raise RuntimeError("S12D named and canonical reports differ")

    required = (
        load_config()["requiredArtifacts"]["files"]
        + load_config()["requiredArtifacts"]["figures"]
    )
    missing_before_manifest = [
        name
        for name in required
        if name != "artifact_manifest.json" and not (STEP_ROOT / name).is_file()
    ]
    if missing_before_manifest:
        raise RuntimeError(
            f"missing S12D required artifacts before manifest: {missing_before_manifest}"
        )
    identities = [
        artifact_identity(STEP_ROOT / name)
        for name in required
        if name != "artifact_manifest.json"
    ]
    artifact_manifest = {
        "schema": "eidosoma.e01.s12d.artifact_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "preregistrationVersion": VERSION,
        "implementationCommit": json.loads(
            (STEP_ROOT / "implementation_lock.json").read_text()
        )["headCommit"],
        "sourceRelationship": SOURCE_RELATIONSHIP,
        "requiredArtifactCountIncludingManifest": len(required),
        "recordedArtifactCountExcludingManifestSelf": len(identities),
        "manifestSelfExcludedToAvoidRecursiveHash": True,
        "artifacts": identities,
        "missingRequiredArtifacts": [],
        "allRequiredArtifactsPresent": True,
        "canonicalAndNamedReportsByteIdentical": True,
        "success": True,
    }
    write_json(STEP_ROOT / "artifact_manifest.json", artifact_manifest)
    missing = [name for name in required if not (STEP_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required S12D artifacts after manifest: {missing}")
    if directory_bytes(STEP_ROOT) > 10737418240:
        raise RuntimeError("S12D artifact directory exceeded 10 GiB after reports")


def placeholder_figures(message: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for filename in load_config()["requiredArtifacts"]["figures"]:
        path = STEP_ROOT / filename
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def fail_closed_metric_identity(summary: dict[str, Any], started: float) -> None:
    """Write a complete pre-science stop report without manufacturing results."""

    for filename, columns in REQUIRED_TABLE_SCHEMAS.items():
        write_parquet(STEP_ROOT / filename, pd.DataFrame(columns=columns))
    for filename, columns in {
        "retrospective_associations.csv": ["status", "reason"],
        "prospective_associations.csv": ["status", "reason"],
        "replicator_drift_results.csv": ["status", "reason"],
        "temporal_structure.csv": ["status", "reason"],
        "spike_analysis.csv": ["status", "reason"],
        "metric_identity_results.csv": ["status", "reason"],
        "future_dependence_results.csv": ["status", "reason"],
        "suffix_invariance_results.csv": ["status", "reason"],
        "failure_ledger.csv": ["failureId", "stage", "status", "reason", "fatal"],
    }.items():
        rows = [
            {
                "status": "NOT_EVALUATED",
                "reason": "source_emergence_identity_gate_failed",
            }
        ]
        if filename == "failure_ledger.csv":
            rows = [
                {
                    "failureId": "S12D-SOURCE-METRIC-IDENTITY",
                    "stage": "metric_identity",
                    "status": "SOURCE_EMERGENCE_IDENTITY_RECONSTRUCTION_FAILED",
                    "reason": "one_or_more_source_metric_identity_rows_failed",
                    "fatal": True,
                }
            ]
        write_csv(STEP_ROOT / filename, rows, columns)
    placeholder_figures(
        "S12D stopped before scientific analysis: source-emergence identity gate failed"
    )
    classification = {
        "schema": "eidosoma.e01.s12d.classification.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "classification": "SOURCE_EMERGENCE_IDENTITY_RECONSTRUCTION_FAILED",
        "genericOutcomeClassification": "constraining/contradictory",
        "sourceRelationship": SOURCE_RELATIONSHIP,
        "s12cClassificationChanged": False,
        "s13Status": "BLOCKED_PENDING_S12D_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "classification.json", classification)
    write_json(
        STEP_ROOT / "runtime_manifest.json",
        {
            "researchStepId": RESEARCH_STEP_ID,
            "wallSeconds": time.perf_counter() - started,
            "gpuHours": 0,
            "success": True,
        },
    )
    write_json(
        STEP_ROOT / "replay_validation.json",
        {
            "researchStepId": RESEARCH_STEP_ID,
            "sourceMetricEquivalence": summary,
            "scientificReplayNotEvaluated": True,
            "success": False,
        },
    )
    write_json(
        STEP_ROOT / "scope_compliance.json",
        {
            "researchStepId": RESEARCH_STEP_ID,
            "scientificAnalysisStarted": False,
            "s13Blocked": True,
            "success": True,
        },
    )
    write_json(
        STEP_ROOT / "storage_validation.json",
        {
            "researchStepId": RESEARCH_STEP_ID,
            "bytes": directory_bytes(STEP_ROOT),
            "hardCeilingBytes": 10737418240,
            "withinCeiling": True,
        },
    )
    write_json(
        STEP_ROOT / "validation_summary.json",
        {
            "researchStepId": RESEARCH_STEP_ID,
            "sourceMetricIdentityPassed": False,
            "scientificAnalysisCorrectlySuppressed": True,
            "success": True,
        },
    )
    status = {
        "researchStepId": RESEARCH_STEP_ID,
        "stepNumber": "12D",
        "success": False,
        "status": "STOPPED_BEFORE_SCIENTIFIC_ANALYSIS",
        "artifactsWritten": [path.name for path in STEP_ROOT.iterdir()],
        "validationResult": f"FAIL CLOSED: {summary['passedRows']}/{summary['expectedRows']} source-metric rows passed",
        "outcomeClassification": "SOURCE_EMERGENCE_IDENTITY_RECONSTRUCTION_FAILED",
        "caveatsOrBlockers": [
            "Source-emergence identity reconstruction did not pass every frozen gate"
        ],
        "recommendedNextAction": "Close S12D and return for human review with S13 blocked; no repair is authorized.",
    }
    write_json(STEP_ROOT / "status.json", status)
    report = f"""# S12D full results — source-defined causal emergence

## Concise top summary

- **Research step ID:** `{VERSION}`.
- **Completion status:** `STOPPED_BEFORE_SCIENTIFIC_ANALYSIS`.
- **Artifacts written:** Preregistration, lock, source identities, {summary["observedRows"]}-row metric-equivalence table, status/failure manifests, empty status-bearing scientific schemas, and stop-state figures under `{STEP_ROOT}`.
- **Validation result:** Fail closed: {summary["passedRows"]}/{summary["expectedRows"]} source-metric rows passed.
- **Outcome classification:** `SOURCE_EMERGENCE_IDENTITY_RECONSTRUCTION_FAILED` (constraining/contradictory).
- **Caveats or blockers:** Source-wrapper equality failed before any GARD emergence outcome was opened; no repair is authorized in S12D.
- **Lay summary:** The public-source emergence quantity could not be reproduced within the exact frozen identity gate, so the planned trajectory test was not scientifically admissible and was not run.
- **Recommended next action:** Return for human review with S13 blocked and close this repair path.

## Methods, commands, inputs, results, validation, caveats, and provenance

The design and implementation were frozen, committed, and pushed before the identity check. Synthetic inputs comprised every S12C development and confirmation fixture plus two new ordinary, two singular, and two near-singular fixtures for both pinned implementations. The isolated source adapter used the audited pickle only on synthetic arrays; the wrapper used safe JSON. At least one of status, array length, nonfinite mask, exact replay, tuple serialization, exact formula, or `1e-12` component agreement failed. The complete row evidence is in `source_metric_equivalence.csv`. No S12/S12C trajectory data, new matrix, intervention, MLP, RL, BioModels input, or S13 output was opened or generated. Prior artifacts and pinned sources remained immutable. This remains `{SOURCE_RELATIONSHIP}` only.
"""
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    shutil.copyfile(
        STEP_ROOT / "research_step_full_results.md", STEP_ROOT / "S12D_FULL_RESULTS.md"
    )
    artifacts = [
        artifact_identity(path)
        for path in sorted(STEP_ROOT.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    write_json(
        STEP_ROOT / "artifact_manifest.json",
        {
            "schema": "eidosoma.e01.s12d.artifact_manifest.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "manifestSelfExcludedToAvoidRecursiveHash": True,
            "artifacts": artifacts,
            "success": True,
        },
    )


def execute(workers: int) -> None:
    started = time.perf_counter()
    config = load_config()
    verify_lock()
    write_data_schema()
    _, equivalence = run_metric_identity_gate(config)
    if not equivalence["success"]:
        fail_closed_metric_identity(equivalence, started)
        return
    generation = generate_confirmation_data(config, workers)
    # This manifest transition records the precise authorization point after
    # every confirmation input was frozen and before any source outcome.
    firewall = json.loads((STEP_ROOT / "data_firewall_manifest.json").read_text())
    firewall["emergenceOutcomeAccessStartedAtUtc"] = datetime.now(
        timezone.utc
    ).isoformat()
    firewall["allFrozenArtifactHashesReverifiedBeforeOutcome"] = all(
        sha256_file(Path(item["path"])) == item["sha256"]
        for item in generation["artifactIdentities"].values()
    )
    if not firewall["allFrozenArtifactHashesReverifiedBeforeOutcome"]:
        raise RuntimeError("confirmation input changed before emergence analysis")
    write_json(STEP_ROOT / "data_firewall_manifest.json", firewall)

    existing_inputs = prepare_existing_inputs()
    confirmation_inputs = prepare_confirmation_inputs()
    all_inputs = existing_inputs + confirmation_inputs
    RESULT_CACHE.mkdir(parents=True, exist_ok=True)
    benchmark = process_source_trajectory(existing_inputs[0]["path"])
    projected_worker_cpu_hours = benchmark["cpuSeconds"] * len(all_inputs) / 3600
    projected_analysis_wall_hours = (
        benchmark["wallSeconds"] * len(all_inputs) / min(workers, 6) / 3600
    )
    benchmark.update(
        {
            "projectedWorkerCpuHoursFor36": projected_worker_cpu_hours,
            "projectedParallelWallHoursFor36": projected_analysis_wall_hours,
            "cpuCeilingHours": 80.0,
            "wallCeilingHours": 24.0,
            "projectionPassed": projected_worker_cpu_hours <= 80
            and projected_analysis_wall_hours <= 24,
        }
    )
    if not benchmark["projectionPassed"]:
        raise RuntimeError(
            "benchmark projects beyond frozen runtime ceiling; scope cannot be reduced"
        )
    records = [benchmark]
    remaining = [
        item for item in all_inputs if item["path"] != existing_inputs[0]["path"]
    ]
    with ProcessPoolExecutor(max_workers=min(workers, 6)) as executor:
        futures = {
            executor.submit(process_source_trajectory, item["path"]): item
            for item in remaining
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: (item["datasetRole"], item["matrixIndex"]))
    full, prefix, partitions, diagnostics, suffix, failures = collate_source_results(
        records
    )
    retrospective, prospective, drift, summaries = analyze_associations(full, prefix)
    temporal, spikes, temporal_map = analyze_temporal_and_spikes(
        full, prefix, partitions
    )
    metric_identity = analyze_metric_identity(full, prefix, summaries)
    future = analyze_future_dependence(full, prefix, partitions)
    retrospective.to_csv(
        STEP_ROOT / "retrospective_associations.csv", index=False, lineterminator="\n"
    )
    prospective.to_csv(
        STEP_ROOT / "prospective_associations.csv", index=False, lineterminator="\n"
    )
    drift.to_csv(
        STEP_ROOT / "replicator_drift_results.csv", index=False, lineterminator="\n"
    )
    temporal.to_csv(
        STEP_ROOT / "temporal_structure.csv", index=False, lineterminator="\n"
    )
    spikes.to_csv(STEP_ROOT / "spike_analysis.csv", index=False, lineterminator="\n")
    metric_identity.to_csv(
        STEP_ROOT / "metric_identity_results.csv", index=False, lineterminator="\n"
    )
    future.to_csv(
        STEP_ROOT / "future_dependence_results.csv", index=False, lineterminator="\n"
    )
    classification = decision_classification(
        retrospective, prospective, drift, summaries, temporal_map, records, suffix
    )
    create_figures(
        full, prefix, retrospective, prospective, spikes, future, classification
    )
    finalize_success(
        full=full,
        prefix=prefix,
        partitions=partitions,
        diagnostics=diagnostics,
        suffix=suffix,
        worker_records=records,
        retrospective=retrospective,
        prospective=prospective,
        drift=drift,
        temporal=temporal,
        spikes=spikes,
        metric_identity=metric_identity,
        future=future,
        classification=classification,
        failures=failures,
        runtime_started=started,
        generation_manifest=generation,
        benchmark=benchmark,
    )


def operational_fail_closed(exc: BaseException) -> None:
    """Preserve an unexpected locked-run failure without altering prior evidence."""

    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    failure = {
        "schema": "eidosoma.e01.s12d.execution_failure.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "status": "S12D_OPERATIONAL_VALIDATION_FAILED_CLOSED",
        "exceptionType": type(exc).__name__,
        "exceptionMessage": str(exc),
        "recordedAtUtc": datetime.now(timezone.utc).isoformat(),
        "scientificCodeRepairAuthorized": False,
        "s13Status": "BLOCKED_PENDING_S12D_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "execution_failure.json", failure)
    ledger_path = STEP_ROOT / "failure_ledger.csv"
    columns = [
        "failureId",
        "stage",
        "datasetRole",
        "implementationId",
        "trajectoryId",
        "observationIndex",
        "status",
        "reason",
        "fatal",
        "count",
    ]
    prior = pd.read_csv(ledger_path).to_dict("records") if ledger_path.is_file() else []
    prior.append(
        {
            "failureId": "S12D-LOCKED-EXECUTION-FAILURE",
            "stage": "locked_execution",
            "datasetRole": None,
            "implementationId": None,
            "trajectoryId": None,
            "observationIndex": None,
            "status": failure["status"],
            "reason": f"{type(exc).__name__}:{exc}",
            "fatal": True,
            "count": 1,
        }
    )
    write_csv(ledger_path, prior, columns)
    classification = {
        "schema": "eidosoma.e01.s12d.classification.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "classification": "S12D_OPERATIONAL_VALIDATION_FAILED_CLOSED",
        "genericOutcomeClassification": "constraining/contradictory",
        "scientificOutcomeClassificationAvailable": False,
        "sourceRelationship": SOURCE_RELATIONSHIP,
        "s12cClassificationChanged": False,
        "s13Status": "BLOCKED_PENDING_S12D_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "classification.json", classification)
    write_json(
        STEP_ROOT / "validation_summary.json",
        {
            "researchStepId": RESEARCH_STEP_ID,
            "success": False,
            "status": failure["status"],
            "failure": failure,
        },
    )
    status = {
        "researchStepId": RESEARCH_STEP_ID,
        "stepNumber": "12D",
        "success": False,
        "status": failure["status"],
        "artifactsWritten": sorted(
            path.relative_to(STEP_ROOT).as_posix()
            for path in STEP_ROOT.rglob("*")
            if path.is_file()
        ),
        "validationResult": f"FAIL CLOSED: {type(exc).__name__}: {exc}",
        "outcomeClassification": failure["status"],
        "caveatsOrBlockers": [
            "The locked S12D run encountered an operational or validation failure; no code repair is authorized after outcome access."
        ],
        "recommendedNextAction": "Return for human review with S13 blocked; do not repair or continue automatically.",
    }
    write_json(STEP_ROOT / "status.json", status)
    report = f"""# S12D full results — source-defined causal emergence

## Concise top summary

- **Research step ID:** `{VERSION}`.
- **Completion status:** `S12D_OPERATIONAL_VALIDATION_FAILED_CLOSED`.
- **Artifacts written:** All successfully materialized locked-run artifacts plus `execution_failure.json`, `failure_ledger.csv`, status, validation, classification, and this canonical handoff under `{STEP_ROOT}`.
- **Validation result:** FAIL CLOSED — `{type(exc).__name__}: {exc}`.
- **Outcome classification:** `S12D_OPERATIONAL_VALIDATION_FAILED_CLOSED`; no admissible scientific classification is assigned.
- **Caveats or blockers:** A locked-run operational or validation failure occurred. Existing generated values, if any, remain immutable evidence but do not pass the global S12D handoff gate.
- **Lay summary:** The audit could not complete all of its preregistered checks, so it stopped without treating partial results as scientific confirmation.
- **Recommended next action:** Return for human review with S13 blocked; no automatic repair or continuation.

## Detailed methods, commands, inputs, results, validation, caveats, and provenance

S12D used the frozen preregistration, exact pinned source commits, safe lattice, source-metric identity firewall, unchanged S12 GARD engine, additive-0.5 dropped-component CLR substrate, frozen labels, completed-fit and past-only-prefix modes, 4,096-resample statistics, exact replay, and suffix-invariance policies described in `preregistration.yaml`. The locked execution command was `python scripts/e01/run_s12d_source_emergence_metric_identity.py --workers 6` under one-thread BLAS/OpenMP settings. The precise failure is stored in `execution_failure.json`; all results written before it remain inspectable and no value was silently repaired, omitted, imputed, or promoted. Prior S01–S12C evidence and S12C's `SOURCE_FAMILY_NOT_SUPPORTED` classification remain unchanged. This step remains `{SOURCE_RELATIONSHIP}` only and cannot authorize S13.
"""
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    shutil.copyfile(
        STEP_ROOT / "research_step_full_results.md", STEP_ROOT / "S12D_FULL_RESULTS.md"
    )
    artifacts = [
        artifact_identity(path)
        for path in sorted(STEP_ROOT.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    write_json(
        STEP_ROOT / "artifact_manifest.json",
        {
            "schema": "eidosoma.e01.s12d.artifact_manifest.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "manifestSelfExcludedToAvoidRecursiveHash": True,
            "artifacts": artifacts,
            "success": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 6:
        raise ValueError("S12D allows 1..6 workers and requires six for the frozen run")
    if args.workers != 6:
        raise ValueError(
            "the frozen S12D scientific run requires exactly six source workers"
        )
    try:
        execute(args.workers)
    except BaseException as exc:
        operational_fail_closed(exc)
        raise


if __name__ == "__main__":
    main()
