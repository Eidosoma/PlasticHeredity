#!/usr/bin/env python3
"""Confirm the one permitted S13 schema adapter, then conditionally run statistics.

The program never simulates a trajectory or invokes a source fit.  It reads the
200 hash-frozen S13 task bundles, creates only the three-field typed label views
authorized by the S13R contract, and applies the original strict aggregation
requirements.  Any newly exposed schema or analysis requirement permanently
stops this one-repair path.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
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
os.environ.setdefault("MPLBACKEND", "Agg")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import yaml
from pyarrow import ipc

from e01_pigozzi_source_audit.core import SourceImplementation
from e01_s13r_schema_normalization.core import (
    ADAPTER_FIELD_TYPES,
    ADAPTER_ID,
    EXPECTED_ADAPTED_TASK_IDS,
    RESEARCH_STEP_ID,
    VERSION,
    array_value_sha256,
    normalize_label_table,
    null_mask_sha256,
    schema_sha256,
    table_logical_sha256,
    table_schema_signature,
)
from scripts.e01 import run_s12g_frozen_timebase_ensemble as backend
from scripts.e01 import run_s13_confirmed_timebase_scaleup as s13

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S13R"
S13_ROOT = ARTIFACTS / "research_steps/S13"
SOURCE_CACHE_ROOT = Path("/cache/e01_s13")
TASK_ROOT = SOURCE_CACHE_ROOT / "source_results"
DERIVED_CACHE_ROOT = Path("/cache/e01_s13r")
NORMALIZED_LABEL_ROOT = DERIVED_CACHE_ROOT / "normalized_label_views"
COLLATED_CACHE_ROOT = DERIVED_CACHE_ROOT / "strict_collation"
CONFIG = REPO / "configs/e01/s13r_schema_normalization_confirmation_preregistration.yaml"
S12G_SCHEMAS = REPO / "configs/e01/s12g_output_schemas.json"
CANDIDATE_IDS = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")

SOURCE_TABLE_MAPPING = {
    "labels.parquet": "label_values.parquet",
    "preprocessing.parquet": "preprocessing_diagnostics.parquet",
    "full.parquet": "full_source_values.parquet",
    "prefix.parquet": "prefix_endpoint_values.parquet",
    "partition.parquet": "partition_history.parquet",
    "diagnostic.parquet": "source_diagnostic_outputs.parquet",
    "suffix.parquet": "replay_suffix_validation.parquet",
    "seeds.parquet": "seed_manifest.parquet",
}

RESULT_COLUMNS = {
    "candidate_associations.csv": [
        "candidateId", "implementationId", "temporalModeId", "labelId", "estimand",
        "definedTrajectoryCount", "positiveTrajectoryCount", "ordinaryPositivePCount",
        "meanCorrelation", "medianCorrelation", "bootstrapLower95", "bootstrapUpper95",
        "circularShiftPositiveP", "circularShiftNegativeP", "effectiveEpisodeCount",
        "medianLagOneAutocorrelation", "finiteCoverage", "gatePassed",
    ],
    "candidate_association_details.parquet": [
        "candidateId", "matrixIndex", "trajectoryId", "implementationId",
        "temporalMode", "estimand", "correlation", "ordinaryP",
    ],
    "replicator_drift_results.csv": [
        "candidateId", "implementationId", "temporalModeId", "labelId",
        "definedTrajectoryCount", "positiveMeanDifferenceCount", "medianMeanDifference",
        "medianMedianDifference", "bootstrapLower95", "bootstrapUpper95",
        "blockAwarePositiveP", "pooledMannWhitneyU", "pooledMannWhitneyP", "gatePassed",
    ],
    "replicator_drift_details.parquet": [
        "candidateId", "matrixIndex", "trajectoryId", "implementationId",
        "temporalMode", "meanDifference", "medianDifference",
    ],
    "temporal_dependence_results.csv": [
        "candidateId", "implementationId", "temporalModeId", "rowType", "trajectoryId",
        "nFinite", "ljungBoxLag", "ljungBoxStatistic", "ljungBoxPValue",
        "differencedLjungBoxLag", "differencedLjungBoxStatistic",
        "differencedLjungBoxPValue", "aggregateTrendSlope", "aggregateTrendPValue",
    ],
    "spike_results.csv": [
        "candidateId", "implementationId", "temporalModeId", "trajectoryId", "nFinite",
        "positive3SigmaCount", "negative3SigmaCount", "robustPositiveCount",
        "robustNegativeCount", "peakCount", "medianPeakWidthObservations",
        "medianPeakProminence", "medianPeakSpacingObservations",
    ],
    "metric_identity_results.csv": [
        "candidateId", "trajectoryId", "matrixIndex", "implementationId", "temporalModeId",
        "sharedFiniteCount", "spearman", "pearson", "signAgreement", "rankAgreement",
        "fractionRanksChangedOver10Points", "spikeJaccard", "partitionIdentity",
        "replicationAssociationEmergence", "replicationAssociationLocalPhiR",
        "replicationAssociationDifference",
    ],
    "future_dependence_results.csv": [
        "candidateId", "trajectoryId", "matrixIndex", "implementationId",
        "sharedEndpointCount", "medianAbsoluteDifference", "fullIqr",
        "normalizedMedianAbsoluteDifference", "spearman", "pearson", "signAgreement",
        "fractionRanksChangedOver10Points", "spikeJaccard", "medianPartitionAdjustedRand",
        "fullReplicationAssociation", "prefixReplicationAssociation",
        "replicationAssociationDifference",
    ],
    "cross_candidate_results.csv": [
        "analysisType", "matrixIndex", "candidateA", "candidateB", "pairingStatus",
        "identityMatched", "metric", "valueA", "valueB", "difference", "status", "reason",
    ],
    "ensemble_adjudication.csv": [
        "candidateId", "candidateEvidenceStatus", "primaryFullAssociationGate",
        "primaryFullDriftGate", "primaryFullCoherent", "primaryPrefixGate",
        "combinedRetrospectiveAndProspectiveGate", "punctuatedGate", "phirlOppositeFull",
        "phirlOppositePrefix", "operationalCoverageGate", "candidateClassification",
    ],
}


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
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
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


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def frame_hash(frame: pd.DataFrame) -> str:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def field_physical_sha256(table: pa.Table, name: str) -> str:
    field = table.schema.field(name)
    one = pa.Table.from_arrays([table[name]], schema=pa.schema([field]))
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, one.schema) as writer:
        writer.write_table(one)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def row_identity_sha256(table: pa.Table) -> str:
    keys = [
        "candidateId", "trajectoryId", "matrixIndex", "generation", "labelId"
    ]
    rows = list(zip(*(table[name].to_pylist() for name in keys), strict=True))
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_method_lock() -> dict[str, Any]:
    lock = json.loads((STEP_ROOT / "method_lock.json").read_text(encoding="utf-8"))
    if not lock.get("passed"):
        raise RuntimeError("S13R method lock is not passing")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote or git("status", "--short"):
        raise RuntimeError("S13R must execute at a clean pushed design commit")
    if head != lock["designCommit"]:
        raise RuntimeError("repository HEAD differs from the S13R method lock")
    for row in lock["files"]:
        path = REPO / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"S13R method-lock file changed: {row['path']}")
    return lock


def validate_prior_artifacts() -> dict[str, Any]:
    baseline = json.loads(
        (STEP_ROOT / "immutable_prior_baseline.json").read_text(encoding="utf-8")
    )
    changed = []
    for row in baseline["files"]:
        path = Path(row["path"])
        actual = sha256_file(path) if path.is_file() else None
        if actual != row["sha256"]:
            changed.append(
                {
                    "path": str(path),
                    "expectedSha256": row["sha256"],
                    "actualSha256": actual,
                }
            )
    s13_classification = json.loads(
        (S13_ROOT / "classification.json").read_text(encoding="utf-8")
    )
    passed = bool(
        not changed
        and s13_classification["classification"] == "S13_VALIDATION_FAILED_CLOSED"
        and not s13_classification["candidateSpecificStatisticsComputed"]
        and not s13_classification["twoCandidateAdjudicationPerformed"]
    )
    payload = {
        "schema": "eidosoma.e01.s13r_immutable_prior_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "fileCount": len(baseline["files"]),
        "changedCount": len(changed),
        "changed": changed,
        "s13ClassificationRetained": s13_classification["classification"],
        "s13CandidateStatisticsRetainedFalse": not s13_classification[
            "candidateSpecificStatisticsComputed"
        ],
        "passed": passed,
    }
    write_json(STEP_ROOT / "immutable_prior_validation.json", payload)
    return payload


def validate_source_cache() -> dict[str, Any]:
    manifest = json.loads(
        (STEP_ROOT / "source_cache_input_manifest.json").read_text(encoding="utf-8")
    )
    changed = []
    for row in manifest["files"]:
        path = SOURCE_CACHE_ROOT / row["relativePath"]
        actual = sha256_file(path) if path.is_file() else None
        if actual != row["sha256"] or (
            path.is_file() and path.stat().st_size != row["bytes"]
        ):
            changed.append(
                {
                    "relativePath": row["relativePath"],
                    "expectedSha256": row["sha256"],
                    "actualSha256": actual,
                }
            )
    payload = {
        "schema": "eidosoma.e01.s13r_source_cache_input_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "expectedFileCount": 2000,
        "checkedFileCount": len(manifest["files"]),
        "changedOrMissingCount": len(changed),
        "changedOrMissing": changed,
        "passed": len(manifest["files"]) == 2000 and not changed,
    }
    write_json(STEP_ROOT / "source_cache_input_validation.json", payload)
    return payload


def task_paths(filename: str, *, normalized_labels: bool = False) -> list[Path]:
    rows: list[Path] = []
    for candidate in CANDIDATE_IDS:
        for matrix_index in range(100):
            root = NORMALIZED_LABEL_ROOT if normalized_labels else TASK_ROOT
            if normalized_labels:
                path = root / candidate / f"M{matrix_index:02d}" / "labels.parquet"
            else:
                path = root / candidate / f"M{matrix_index:02d}" / filename
            rows.append(path)
    return rows


def normalize_labels() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if NORMALIZED_LABEL_ROOT.exists():
        raise RuntimeError("S13R derived label-view root already exists; one-repair rerun forbidden")
    canonical_path = TASK_ROOT / CANDIDATE_IDS[0] / "M00/labels.parquet"
    canonical = pq.read_table(canonical_path)
    for name, target in ADAPTER_FIELD_TYPES.items():
        if not canonical.schema.field(name).type.equals(target):
            raise RuntimeError(f"canonical label field {name} is not {target}")
    canonical_schema = canonical.schema
    canonical_signature = table_schema_signature(canonical)
    task_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    adapted_task_ids: set[str] = set()
    all_pass = True
    for candidate in CANDIDATE_IDS:
        for matrix_index in range(100):
            task_id = f"{candidate}/M{matrix_index:02d}"
            source_path = TASK_ROOT / candidate / f"M{matrix_index:02d}/labels.parquet"
            source_hash_before = sha256_file(source_path)
            source = pq.read_table(source_path)
            source_columns = list(source.column_names)
            source_logical_hash = table_logical_sha256(source)
            source_identity_hash = row_identity_sha256(source)
            source_types = {name: str(source.schema.field(name).type) for name in source_columns}
            provisional, adapted_fields = normalize_label_table(source)
            if adapted_fields:
                adapted_task_ids.add(task_id)
            normalized = pa.Table.from_arrays(
                [provisional[name] for name in source_columns], schema=canonical_schema
            )
            view_path = NORMALIZED_LABEL_ROOT / candidate / f"M{matrix_index:02d}/labels.parquet"
            view_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(normalized, view_path, compression="zstd")
            roundtrip = pq.read_table(view_path)
            source_hash_after = sha256_file(source_path)
            candidate_identity = set(roundtrip["candidateId"].to_pylist()) == {candidate}
            matrix_identity = set(roundtrip["matrixIndex"].to_pylist()) == {matrix_index}
            unique_rows = len(
                set(
                    zip(
                        roundtrip["candidateId"].to_pylist(),
                        roundtrip["trajectoryId"].to_pylist(),
                        roundtrip["matrixIndex"].to_pylist(),
                        roundtrip["generation"].to_pylist(),
                        roundtrip["labelId"].to_pylist(),
                        strict=True,
                    )
                )
            ) == roundtrip.num_rows
            canonical_input = not adapted_fields
            task_pass = bool(
                source.num_rows == roundtrip.num_rows
                and source_columns == roundtrip.column_names
                and table_logical_sha256(roundtrip) == source_logical_hash
                and row_identity_sha256(roundtrip) == source_identity_hash
                and table_schema_signature(roundtrip) == canonical_signature
                and roundtrip.schema.equals(canonical_schema, check_metadata=True)
                and source_hash_before == source_hash_after
                and candidate_identity
                and matrix_identity
                and unique_rows
                and ((task_id in EXPECTED_ADAPTED_TASK_IDS) == bool(adapted_fields))
                and set(adapted_fields).issubset(ADAPTER_FIELD_TYPES)
                and (
                    not canonical_input
                    or roundtrip.equals(source, check_metadata=False)
                )
            )
            for name in source_columns:
                source_value_hash = array_value_sha256(source[name])
                view_value_hash = array_value_sha256(roundtrip[name])
                source_null_hash = null_mask_sha256(source[name])
                view_null_hash = null_mask_sha256(roundtrip[name])
                source_physical = field_physical_sha256(source, name)
                view_physical = field_physical_sha256(roundtrip, name)
                allowed_type_change = name in adapted_fields
                field_pass = bool(
                    source_value_hash == view_value_hash
                    and source_null_hash == view_null_hash
                    and (
                        allowed_type_change
                        or (
                            source_types[name] == str(roundtrip.schema.field(name).type)
                            and source_physical == view_physical
                        )
                    )
                )
                field_rows.append(
                    {
                        "taskId": task_id,
                        "candidateId": candidate,
                        "matrixIndex": matrix_index,
                        "fieldName": name,
                        "authorizedAdapterField": name in ADAPTER_FIELD_TYPES,
                        "physicallyAdapted": allowed_type_change,
                        "sourceType": source_types[name],
                        "viewType": str(roundtrip.schema.field(name).type),
                        "sourceValueSha256": source_value_hash,
                        "viewValueSha256": view_value_hash,
                        "valueIdentical": source_value_hash == view_value_hash,
                        "sourceNullMaskSha256": source_null_hash,
                        "viewNullMaskSha256": view_null_hash,
                        "nullMaskIdentical": source_null_hash == view_null_hash,
                        "sourcePhysicalFieldSha256": source_physical,
                        "viewPhysicalFieldSha256": view_physical,
                        "nonAdapterPhysicalIdentity": (
                            None if allowed_type_change else source_physical == view_physical
                        ),
                        "passed": field_pass,
                    }
                )
                task_pass = task_pass and field_pass
            task_rows.append(
                {
                    "taskId": task_id,
                    "candidateId": candidate,
                    "matrixIndex": matrix_index,
                    "rowCount": source.num_rows,
                    "columnCount": source.num_columns,
                    "sourceSchemaSha256": schema_sha256(source),
                    "viewSchemaSha256": schema_sha256(roundtrip),
                    "sourceLogicalSha256": source_logical_hash,
                    "viewLogicalSha256": table_logical_sha256(roundtrip),
                    "sourceRowIdentitySha256": source_identity_hash,
                    "viewRowIdentitySha256": row_identity_sha256(roundtrip),
                    "adaptedFieldsJson": json.dumps(list(adapted_fields)),
                    "canonicalInput": canonical_input,
                    "sourceSha256Before": source_hash_before,
                    "sourceSha256After": source_hash_after,
                    "sourceUnchanged": source_hash_before == source_hash_after,
                    "candidateIdentityPassed": candidate_identity,
                    "matrixIdentityPassed": matrix_identity,
                    "uniqueRowIdentityPassed": unique_rows,
                    "passed": task_pass,
                }
            )
            all_pass = all_pass and task_pass
            manifest_rows.append(
                {
                    "taskId": task_id,
                    "relativePath": str(view_path.relative_to(DERIVED_CACHE_ROOT)),
                    "bytes": view_path.stat().st_size,
                    "sha256": sha256_file(view_path),
                    "schemaSha256": schema_sha256(roundtrip),
                    "rowCount": roundtrip.num_rows,
                }
            )
    task_frame = pd.DataFrame(task_rows)
    field_frame = pd.DataFrame(field_rows)
    write_parquet(STEP_ROOT / "adapter_task_audit.parquet", task_frame)
    write_parquet(STEP_ROOT / "adapter_field_audit.parquet", field_frame)
    schema_payload = {
        "schema": "eidosoma.e01.s13r_normalized_label_schema.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "adapterId": ADAPTER_ID,
        "orderedFields": [
            {"name": field.name, "type": str(field.type)} for field in canonical_schema
        ],
        "orderedSignatureSha256": schema_sha256(canonical),
        "canonicalMetadataSha256": hashlib.sha256(
            json.dumps(
                {
                    str(key): value.decode("utf-8", errors="replace")
                    for key, value in (canonical_schema.metadata or {}).items()
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "passed": True,
    }
    write_json(STEP_ROOT / "normalized_label_schema.json", schema_payload)
    manifest = {
        "schema": "eidosoma.e01.s13r_normalized_view_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "cacheRoot": str(DERIVED_CACHE_ROOT),
        "viewCount": len(manifest_rows),
        "totalBytes": sum(row["bytes"] for row in manifest_rows),
        "files": manifest_rows,
        "sourceFilesMutated": False,
        "passed": len(manifest_rows) == 200 and all_pass,
    }
    write_json(STEP_ROOT / "normalized_view_manifest.json", manifest)
    payload = {
        "schema": "eidosoma.e01.s13r_adapter_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "adapterId": ADAPTER_ID,
        "taskCount": len(task_frame),
        "taskPassCount": int(task_frame["passed"].sum()),
        "fieldAuditCount": len(field_frame),
        "fieldPassCount": int(field_frame["passed"].sum()),
        "canonicalTypedTaskCount": int(task_frame["canonicalInput"].sum()),
        "adaptedTaskCount": int((~task_frame["canonicalInput"]).sum()),
        "adaptedTaskIds": sorted(adapted_task_ids),
        "exactAffectedTaskIdentity": adapted_task_ids == EXPECTED_ADAPTED_TASK_IDS,
        "all200ViewsShareOneDeclaredSchema": task_frame["viewSchemaSha256"].nunique() == 1,
        "allRowsValuesNullMasksAndOrderUnchanged": bool(task_frame["passed"].all()),
        "allNonAdapterFieldsPhysicalIdentity": bool(
            field_frame.loc[~field_frame["physicallyAdapted"], "nonAdapterPhysicalIdentity"]
            .fillna(False)
            .astype(bool)
            .all()
        ),
        "canonical195ValueIdentical": bool(
            task_frame.loc[task_frame["canonicalInput"], "passed"].all()
        ),
        "sourceFilesMutated": False,
        "passed": bool(
            all_pass
            and len(task_frame) == 200
            and len(adapted_task_ids) == 5
            and adapted_task_ids == EXPECTED_ADAPTED_TASK_IDS
            and task_frame["viewSchemaSha256"].nunique() == 1
            and manifest["passed"]
        ),
    }
    write_json(STEP_ROOT / "adapter_validation.json", payload)
    return payload, task_frame, field_frame


def downstream_schema_validation() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_pass = True
    for source_name, target_name in SOURCE_TABLE_MAPPING.items():
        paths = task_paths(
            source_name, normalized_labels=source_name == "labels.parquet"
        )
        signatures: list[str] = []
        schemas: dict[str, list[tuple[str, str]]] = {}
        representatives: dict[str, str] = {}
        for path in paths:
            schema = pq.read_schema(path)
            signature = [(field.name, str(field.type)) for field in schema]
            encoded = json.dumps(signature, separators=(",", ":")).encode()
            digest = hashlib.sha256(encoded).hexdigest()
            signatures.append(digest)
            schemas[digest] = signature
            representatives.setdefault(digest, str(path))
        counts = Counter(signatures)
        passed = len(counts) == 1
        all_pass = all_pass and passed
        rows.append(
            {
                "sourceTable": source_name,
                "targetTable": target_name,
                "taskCount": len(paths),
                "schemaVariantCount": len(counts),
                "variants": [
                    {
                        "schemaSha256": digest,
                        "taskCount": count,
                        "representativePath": representatives[digest],
                        "orderedFields": [
                            {"name": name, "type": field_type}
                            for name, field_type in schemas[digest]
                        ],
                    }
                    for digest, count in sorted(counts.items())
                ],
                "strictFrozenConcatenationCompatible": passed,
            }
        )
    payload = {
        "schema": "eidosoma.e01.s13r_downstream_schema_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "adapterScope": "labels.parquet exact three fields only",
        "tables": rows,
        "incompatibleTables": [
            row["sourceTable"] for row in rows if not row["strictFrozenConcatenationCompatible"]
        ],
        "furtherSchemaNormalizationApplied": False,
        "candidateStatisticOpened": False,
        "passed": all_pass,
    }
    write_json(STEP_ROOT / "downstream_schema_validation.json", payload)
    return payload


def strict_concat(paths: list[Path], output: Path) -> pd.DataFrame:
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
    return pd.read_parquet(output)


def collate_all_sources() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    COLLATED_CACHE_ROOT.mkdir(parents=True, exist_ok=False)
    for source_name, target_name in SOURCE_TABLE_MAPPING.items():
        paths = task_paths(
            source_name, normalized_labels=source_name == "labels.parquet"
        )
        cache_path = COLLATED_CACHE_ROOT / target_name
        frames[target_name] = strict_concat(paths, cache_path)
        artifact_path = STEP_ROOT / target_name
        artifact_path.write_bytes(cache_path.read_bytes())
    failures = []
    for path in task_paths("failures.parquet"):
        table = pq.read_table(path)
        if table.num_rows:
            failures.append(table.to_pandas())
    frames["worker_failures"] = pd.concat(failures, ignore_index=True) if failures else pd.DataFrame()
    return frames


def prefix_statistical_view(prefix: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    groups = 0
    monotone = 0
    for _, group in adapted.groupby(
        ["candidateId", "trajectoryId", "implementationId"], sort=True
    ):
        groups += 1
        values = group.sort_values("generation")["rawObservationIndex"].to_numpy(
            dtype=np.int64
        )
        monotone += int(
            len(values) == len(np.unique(values)) and np.all(np.diff(values) > 0)
        )
    original_after = frame_hash(adapted[original_columns])
    payload = {
        "schema": "eidosoma.e01.s13r_frozen_s13_prefix_interface_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "inheritedAdapter": "rawObservationIndex := endpointRawObservationIndex",
        "newS13RRepair": False,
        "rowCount": len(prefix),
        "integerIdentity": integer_identity,
        "groupCount": groups,
        "strictMonotoneGroupCount": monotone,
        "originalFieldHashBefore": original_hash,
        "originalFieldHashAfter": original_after,
        "passed": bool(integer_identity and groups == monotone and original_hash == original_after),
    }
    write_json(STEP_ROOT / "prefix_interface_validation.json", payload)
    return adapted, payload


def run_frozen_statistics(
    frames: dict[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    prefix = frames["prefix_endpoint_values.parquet"]
    adapted_prefix, prefix_validation = prefix_statistical_view(prefix)
    if not prefix_validation["passed"]:
        raise RuntimeError("frozen S13 prefix statistical interface failed")
    first, first_classification = s13.compute_statistics(
        frames["full_source_values.parquet"],
        adapted_prefix,
        prefix,
        frames["label_values.parquet"],
        frames["partition_history.parquet"],
    )
    second, second_classification = s13.compute_statistics(
        frames["full_source_values.parquet"],
        adapted_prefix,
        prefix,
        frames["label_values.parquet"],
        frames["partition_history.parquet"],
    )
    replay_rows = []
    for key in s13.RESULT_FILES:
        exact = True
        try:
            pd.testing.assert_frame_equal(first[key], second[key], check_exact=True, check_dtype=True)
        except AssertionError:
            exact = False
        replay_rows.append(
            {
                "resultId": key,
                "rowCount": len(first[key]),
                "firstSha256": frame_hash(first[key]),
                "secondSha256": frame_hash(second[key]),
                "exact": exact,
            }
        )
    classification_exact = json.dumps(jsonable(first_classification), sort_keys=True) == json.dumps(
        jsonable(second_classification), sort_keys=True
    )
    replay = {
        "schema": "eidosoma.e01.s13r_statistics_replay_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "statisticsExecutions": 2,
        "results": replay_rows,
        "classificationExact": classification_exact,
        "passed": all(row["exact"] for row in replay_rows) and classification_exact,
    }
    write_json(STEP_ROOT / "statistics_replay_validation.json", replay)
    if not replay["passed"]:
        raise RuntimeError("exact S13 statistics replay failed")
    for key, filename in s13.RESULT_FILES.items():
        path = STEP_ROOT / filename
        if path.suffix == ".parquet":
            write_parquet(path, first[key])
        else:
            write_csv(path, first[key])
    classification = {
        **first_classification,
        "schema": "eidosoma.e01.s13r_classification.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "s13FrozenScientificClassification": first_classification["classification"],
        "evidenceLabel": "POST_FAILURE_SCHEMA_REPAIRED_HELD_OUT_ANALYSIS",
        "s13ClassificationRetained": "S13_VALIDATION_FAILED_CLOSED",
        "s14ThroughS18Status": "BLOCKED_PENDING_S13R_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "classification.json", classification)
    return first, classification, replay


def create_empty_source_and_result_outputs() -> None:
    schemas = json.loads(S12G_SCHEMAS.read_text(encoding="utf-8"))["tables"]
    for target in SOURCE_TABLE_MAPPING.values():
        columns = schemas[target]
        write_parquet(STEP_ROOT / target, pd.DataFrame(columns=columns))
    for filename, columns in RESULT_COLUMNS.items():
        frame = pd.DataFrame(columns=columns)
        if filename.endswith(".parquet"):
            write_parquet(STEP_ROOT / filename, frame)
        else:
            write_csv(STEP_ROOT / filename, frame)


def schema_validation(*, scientific_reached: bool) -> dict[str, Any]:
    rows = []
    for filename, columns in {
        **{target: json.loads(S12G_SCHEMAS.read_text())["tables"][target] for target in SOURCE_TABLE_MAPPING.values()},
        **RESULT_COLUMNS,
    }.items():
        path = STEP_ROOT / filename
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        missing = [column for column in columns if column not in frame.columns]
        rows.append(
            {
                "path": filename,
                "exists": path.is_file(),
                "rowCount": len(frame),
                "missingColumns": missing,
                "scientificStatus": "ELIGIBLE" if scientific_reached else "SUPPRESSED_NOT_REACHED",
                "passed": not missing,
            }
        )
    payload = {
        "schema": "eidosoma.e01.s13r_schema_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "tables": rows,
        "scientificOutputsReached": scientific_reached,
        "passed": all(row["passed"] for row in rows),
    }
    write_json(STEP_ROOT / "schema_validation.json", payload)
    return payload


def artifact_manifest(config: dict[str, Any]) -> dict[str, Any]:
    required = config["artifacts"]["required"]
    entries = [
        {
            "relativePath": str(path.relative_to(STEP_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(STEP_ROOT.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    present = {row["relativePath"] for row in entries}
    missing = [name for name in required if name != "artifact_manifest.json" and name not in present]
    total = sum(row["bytes"] for row in entries)
    payload = {
        "schema": "eidosoma.e01.s13r_artifact_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "artifacts": entries,
        "artifactCountExcludingSelf": len(entries),
        "totalBytesExcludingSelf": total,
        "requiredMissing": missing,
        "under30GiB": total <= 30 * 1024**3,
        "passed": not missing and total <= 30 * 1024**3,
    }
    write_json(STEP_ROOT / "artifact_manifest.json", payload)
    return payload


def stop_report(
    reason: str,
    adapter: dict[str, Any],
    downstream: dict[str, Any],
    runtime: dict[str, Any],
    artifact_count: int,
) -> str:
    incompatible = ", ".join(downstream.get("incompatibleTables", [])) or "none"
    return "\n".join(
        [
            "# S13R Full Results: Schema Normalization Confirmation",
            "",
            "## Top summary",
            "",
            f"- **Research step ID:** `{VERSION}` (S13R).",
            "- **Completion status:** `PERMANENTLY_STOPPED_UNDER_ONE_REPAIR_RULE`; no candidate statistic or two-candidate adjudication was computed.",
            f"- **Artifacts written:** {artifact_count} status-bearing artifacts under `/artifacts/research_steps/S13R/`, including the preregistration/method lock, 200-task and field-level adapter audits, normalized-view hashes, downstream schema diagnostics, suppressed scientific schemas, validation/provenance/failure records, and this report.",
            f"- **Validation result:** The authorized label adapter passed {adapter.get('taskPassCount', 0)}/{adapter.get('taskCount', 0)} task views, but the frozen downstream aggregation contract exposed another incompatible schema and stopped: `{reason}`.",
            "- **Outcome classification:** `S13R_REPAIR_PATH_PERMANENTLY_STOPPED` (constraining/contradictory operational evidence); held-out scientific association remains `NOT_EVALUATED`.",
            f"- **Caveats or blockers:** Incompatible downstream tables: {incompatible}. The authorization permits no second schema adapter, data-loading workaround, subset analysis, source rerun, or method change. S13 remains byte-for-byte unchanged and classified `S13_VALIDATION_FAILED_CLOSED`.",
            "- **Lay summary:** The five label files were safely normalized without changing a row, value, or null. The next untouched table family then revealed a separate all-ineligible-task physical-schema mismatch. Because the human authorized exactly one narrow repair, the analysis stopped before correlations rather than silently adding a second repair.",
            "- **Recommended next action:** Close this repair path and return for mandatory human review. Keep S14–S18, prediction, interventions, E02, report-bundle progression, and further scale-up blocked.",
            "",
            "## Frozen question",
            "",
            "S13R asked whether the five all-null label tables could be represented with the 195-task canonical physical schema and, only after every frozen gate passed, whether the original S13 statistics could adjudicate the held-out two-candidate result. The first question passed; the conditional scientific question was not reached.",
            "",
            "## Inputs",
            "",
            "Only the 200 frozen per-task bundles under `/cache/e01_s13/source_results/` and their S13 manifests were read. The partial S13 concatenation was not read. No trajectory was generated, no source fit was rerun, no candidate was added or removed, and candidate 1 remained excluded.",
            "",
            "## Detailed methods",
            "",
            "Before any candidate statistic, the exact three-field adapter, all validations, the inherited S13 statistics entry points, and the one-repair stop rule were committed and pushed. For each label task, Arrow `null` was changed only to `string` for `clusterId` and `referenceObservationId`, and only to `double` for `metricToReference`. Logical value hashes, null-mask hashes, row/key order, task identity, column order, non-adapter physical-field hashes, and original source-file hashes were checked. The 195 canonical inputs were required to remain value-identical; exactly the five preregistered tasks were allowed a physical type change.",
            "",
            "After the adapter passed, S13R performed a read-only schema compatibility audit for the exact frozen strict concatenation over all 200 tasks. It did not coerce, omit, or normalize another table. Any second schema variant was a terminal condition before scientific access.",
            "",
            "## Commands",
            "",
            "```bash",
            "PYTHONPATH=src python -m pytest -q tests/e01/test_s13r_schema_normalization_confirmation.py tests/e01/test_s13_confirmed_timebase_scaleup.py",
            "python -m ruff check src/e01_s13r_schema_normalization scripts/e01/freeze_s13r_preregistration.py scripts/e01/run_s13r_schema_normalization_confirmation.py tests/e01/test_s13r_schema_normalization_confirmation.py",
            "ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s13r_preregistration.py --record-commit",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13r_schema_normalization_confirmation.py",
            "```",
            "",
            "## Results",
            "",
            f"- Adapter task views: {adapter.get('taskPassCount', 0)}/{adapter.get('taskCount', 0)} passed.",
            f"- Canonical typed tasks: {adapter.get('canonicalTypedTaskCount', 0)}; physically adapted tasks: {adapter.get('adaptedTaskCount', 0)}.",
            f"- Field checks: {adapter.get('fieldPassCount', 0)}/{adapter.get('fieldAuditCount', 0)} passed.",
            f"- All normalized label views share one schema: {adapter.get('all200ViewsShareOneDeclaredSchema')}.",
            f"- Downstream strict schema-compatible tables: {sum(row['strictFrozenConcatenationCompatible'] for row in downstream.get('tables', []))}/{len(downstream.get('tables', []))}.",
            f"- Newly exposed incompatible tables: `{incompatible}`.",
            "- Association, drift, temporal, spike, metric-identity, future-dependence, resampling, paired comparison, statistics replay, and two-candidate adjudication: `NOT_REACHED_ONE_REPAIR_RULE`.",
            "",
            "## Validation",
            "",
            "All prior artifact hashes and all 2,000 S13 cache-file hashes were checked before adaptation. Every original source file remained unchanged. S13's failure classification and lack of scientific adjudication were retained. The adapter and derived-view hashes are complete. Scientific result artifacts are schema-bearing and empty because the downstream gate fired.",
            "",
            "## Runtime and storage",
            "",
            f"S13R used {runtime.get('wallSeconds', 0.0):.3f} wall-seconds and {runtime.get('processCpuSeconds', 0.0):.3f} orchestration CPU-seconds, no simulation/source worker and no GPU time. Retained derived-cache bytes: {runtime.get('derivedCacheBytes', 0)}. Retained artifact bytes before the final manifest: {runtime.get('artifactBytesBeforeManifest', 0)}.",
            "",
            "## Caveats and limitations",
            "",
            "- This is a post-failure human override and further weakens the confirmatory standing of the branch.",
            "- The successful label normalization does not validate any source-emergence association.",
            "- The additional mismatch occurs in task bundles with no eligible prefix endpoint; treating those rows or empty suffix records specially would itself be another schema/data-interface decision and was not authorized.",
            "- Public-source behavior is not the unavailable author implementation; retrospective full fits remain future-dependent, and fixed-window/early-time claims remain unresolved.",
            "",
            "## Provenance",
            "",
            "The pushed method lock, complete S01–S13 artifact baseline, 2,000-file S13 cache manifest, per-field logical/physical hashes, normalized-view manifest, source/postcheck hashes, scope ledger, runtime/storage records, failure ledger, and artifact manifest preserve the entire decision path. Derived views live under `/cache/e01_s13r/`; compact audit evidence is under `/artifacts/research_steps/S13R/`.",
            "",
            "## Recommended next action",
            "",
            "Mandatory human review. Under the one-repair rule, no additional S13/S13R schema adapter or analysis repair is authorized. Do not begin later work automatically.",
            "",
        ]
    )


def complete_report(
    classification: dict[str, Any],
    results: dict[str, pd.DataFrame],
    adapter: dict[str, Any],
    runtime: dict[str, Any],
    artifact_count: int,
) -> str:
    adjudication = results["ensemble_adjudication"]
    associations = results["candidate_associations"]
    drift = results["replicator_drift_results"]
    lines = [
        "# S13R Full Results: Schema Normalization Confirmation",
        "",
        "## Top summary",
        "",
        f"- **Research step ID:** `{VERSION}` (S13R).",
        "- **Completion status:** `COMPLETED_AT_MANDATORY_S13R_HUMAN_REVIEW_BOUNDARY`; no later step began.",
        f"- **Artifacts written:** {artifact_count} complete status-bearing artifacts under `/artifacts/research_steps/S13R/`.",
        "- **Validation result:** `PASS_EXACT_LABEL_SCHEMA_ADAPTER_AND_ALL_FROZEN_S13_STATISTICS_REPLAY_GATES`.",
        f"- **Outcome classification:** `{classification['classification']}` as `POST_FAILURE_SCHEMA_REPAIRED_HELD_OUT_ANALYSIS`.",
        "- **Caveats or blockers:** S13 remains failed and immutable; this is a post-result human override; public-source behavior is not author-primary; completed fits remain retrospective; all later work remains blocked.",
        "- **Lay summary:** The five all-null label files were typed without changing their scientific contents, and the original held-out analysis was then run twice exactly. The two confirmed candidates' result is reported below, but it does not erase S13's failure or authorize continuation.",
        "- **Recommended next action:** Mandatory human review; do not begin S14–S18, prediction, interventions, E02, or another scale-up.",
        "",
        "## Frozen question, inputs, and methods",
        "",
        "Only the 200 S13 per-task outputs were used. Exactly three optional label fields were normalized in derived views; all rows, values, null masks, keys, and other fields were unchanged. The original S13 statistics, 4,096-replicate resampling seeds, gates, candidate identities, and two-candidate adjudication were executed twice with exact replay.",
        "",
        "## Candidate-specific results",
        "",
        "| Candidate | Full median rho | Full association | Full median rep-drift | Drift | Full coherent | Prefix median rho | Prefix | Combined | Classification |",
        "| --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in adjudication.to_dict("records"):
        candidate = row["candidateId"]
        full_assoc = associations[(associations.candidateId == candidate) & (associations.implementationId == SourceImplementation.IIGR.value) & (associations.estimand == "RETROSPECTIVE_CURRENT_GENERATION")].iloc[0]
        prefix_assoc = associations[(associations.candidateId == candidate) & (associations.implementationId == SourceImplementation.IIGR.value) & (associations.estimand == "CURRENT_HISTORICAL") & (associations.temporalModeId.str.endswith("_PREFIX_ENDPOINT"))].iloc[0]
        full_drift = drift[(drift.candidateId == candidate) & (drift.implementationId == SourceImplementation.IIGR.value) & (drift.temporalModeId.str.endswith("_FULL"))].iloc[0]
        lines.append(
            f"| {candidate} | {full_assoc['medianCorrelation']:.5g} | {bool(row['primaryFullAssociationGate'])} | {full_drift['medianMeanDifference']:.5g} | {bool(row['primaryFullDriftGate'])} | {bool(row['primaryFullCoherent'])} | {prefix_assoc['medianCorrelation']:.5g} | {bool(row['primaryPrefixGate'])} | {bool(row['combinedRetrospectiveAndProspectiveGate'])} | {row['candidateClassification']} |"
        )
    lines.extend(
        [
            "",
            f"Final classification: **`{classification['classification']}`**.",
            "",
            "## Validation and provenance",
            "",
            f"The adapter passed {adapter['taskPassCount']}/200 task views and {adapter['fieldPassCount']}/{adapter['fieldAuditCount']} field checks. All prior/cache hashes, source replay and suffix evidence, exact statistics replay, schemas, runtime/storage, and artifact hashes passed. Design and execution provenance is preserved in the method, implementation, cache, adapter, statistics, and artifact manifests.",
            "",
            "## Commands",
            "",
            "The exact commands are recorded in `runtime_manifest.json` and match the preregistration freeze, test/lint, and runner commands in the S13R method lock.",
            "",
            "## Caveats and recommended next action",
            "",
            "This is post-failure schema-repaired evidence. It cannot identify the unavailable author implementation, recover fixed-window or early-time claims, support prediction or intervention, or erase prior negative evidence. Return for mandatory human review with every later step blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def write_common_final_artifacts(
    *,
    config: dict[str, Any],
    method_lock: dict[str, Any],
    adapter: dict[str, Any],
    downstream: dict[str, Any],
    prior: dict[str, Any],
    cache_validation: dict[str, Any],
    started_wall: float,
    started_cpu: float,
    success: bool,
    classification_token: str,
    scientific_status: str,
    reason: str | None,
    results: dict[str, pd.DataFrame] | None,
    classification_payload: dict[str, Any] | None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    source_task = pd.read_parquet(S13_ROOT / "source_task_validation.parquet")
    source_validation = {
        "schema": "eidosoma.e01.s13r_source_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "sourceTaskCount": len(source_task),
        "fullReplayPassCount": int(source_task["fullReplayAllPassed"].astype(bool).sum()),
        "prefixReplayPassCount": int(source_task["prefixReplayAllPassed"].astype(bool).sum()),
        "suffixTaskPassCount": int(source_task["futureSuffixAllPassed"].astype(bool).sum()),
        "taskFailureRowCount": int(source_task["failureRows"].sum()),
        "sourceFitRerun": False,
        "sourceCacheHashValidationPassed": cache_validation["passed"],
        "strictDownstreamSchemaPassed": downstream["passed"],
        "passed": bool(
            len(source_task) == 200
            and source_task["fullReplayAllPassed"].astype(bool).all()
            and source_task["prefixReplayAllPassed"].astype(bool).all()
            and source_task["futureSuffixAllPassed"].astype(bool).all()
            and int(source_task["failureRows"].sum()) == 0
            and cache_validation["passed"]
            and downstream["passed"]
        ),
    }
    write_json(STEP_ROOT / "source_validation.json", source_validation)
    if not (STEP_ROOT / "prefix_interface_validation.json").exists():
        write_json(
            STEP_ROOT / "prefix_interface_validation.json",
            {
                "schema": "eidosoma.e01.s13r_frozen_s13_prefix_interface_validation.v1",
                "researchStepId": RESEARCH_STEP_ID,
                "status": "NOT_REACHED_ONE_REPAIR_RULE",
                "passed": False,
                "reason": reason,
            },
        )
    if not (STEP_ROOT / "statistics_replay_validation.json").exists():
        write_json(
            STEP_ROOT / "statistics_replay_validation.json",
            {
                "schema": "eidosoma.e01.s13r_statistics_replay_validation.v1",
                "researchStepId": RESEARCH_STEP_ID,
                "statisticsExecutions": 0,
                "status": "NOT_REACHED_ONE_REPAIR_RULE",
                "reason": reason,
                "passed": False,
            },
        )
    if results is None:
        create_empty_source_and_result_outputs()
    if classification_payload is None:
        classification_payload = {
            "schema": "eidosoma.e01.s13r_classification.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "classification": classification_token,
            "scientificAssociationClassification": scientific_status,
            "reason": reason,
            "s13ClassificationRetained": "S13_VALIDATION_FAILED_CLOSED",
            "candidate1Excluded": True,
            "candidateSpecificStatisticsComputed": False,
            "twoCandidateAdjudicationPerformed": False,
            "evidenceLabel": "POST_FAILURE_SCHEMA_REPAIRED_HELD_OUT_ANALYSIS_NOT_REACHED",
            "laterWorkStatus": "BLOCKED_PENDING_S13R_HUMAN_REVIEW",
        }
        write_json(STEP_ROOT / "classification.json", classification_payload)
    failure_columns = json.loads(S12G_SCHEMAS.read_text())["tables"]["failure_ledger.csv"]
    if success:
        failures = backend.failure_rows_from_statuses(
            pd.read_parquet(STEP_ROOT / "full_source_values.parquet"),
            pd.read_parquet(STEP_ROOT / "prefix_endpoint_values.parquet"),
            pd.DataFrame(),
        )
        for row in failures:
            row["failureId"] = str(row["failureId"]).replace("S12G-", "S13R-")
        failure_frame = pd.DataFrame(failures, columns=failure_columns)
    else:
        failure_frame = pd.DataFrame(
            [
                {
                    "failureId": "S13R-TERMINAL-ONE-REPAIR-RULE",
                    "stage": "post_adapter_strict_source_aggregation",
                    "candidateId": None,
                    "trajectoryId": None,
                    "implementationId": None,
                    "temporalModeId": None,
                    "endpointGeneration": None,
                    "severity": "FATAL",
                    "status": classification_token,
                    "reason": reason,
                    "gateImpact": "PERMANENT_STOP_NO_SECOND_REPAIR_NO_STATISTICS",
                    "repairAttempted": False,
                }
            ],
            columns=failure_columns,
        )
    write_csv(STEP_ROOT / "failure_ledger.csv", failure_frame)
    schema = schema_validation(scientific_reached=success)
    derived_bytes = sum(
        path.stat().st_size for path in DERIVED_CACHE_ROOT.rglob("*") if path.is_file()
    ) if DERIVED_CACHE_ROOT.exists() else 0
    artifact_bytes = sum(
        path.stat().st_size for path in STEP_ROOT.rglob("*") if path.is_file()
    )
    runtime = {
        "schema": "eidosoma.e01.s13r_runtime_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "startedAtUtc": now,
        "wallSeconds": time.perf_counter() - started_wall,
        "processCpuSeconds": time.process_time() - started_cpu,
        "simulationWorkerCpuSeconds": 0.0,
        "sourceFitWorkerCpuSeconds": 0.0,
        "gpuHours": 0.0,
        "workers": 1,
        "threadEnvironment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            )
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cpuFloat64Authoritative": True,
        "derivedCacheBytes": derived_bytes,
        "artifactBytesBeforeManifest": artifact_bytes,
        "passed": True,
    }
    write_json(STEP_ROOT / "runtime_manifest.json", runtime)
    storage = {
        "schema": "eidosoma.e01.s13r_storage_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "derivedCacheBytes": derived_bytes,
        "artifactBytesBeforeManifest": artifact_bytes,
        "retainedArtifactGiBCeiling": 30.0,
        "passed": artifact_bytes <= 30 * 1024**3,
    }
    write_json(STEP_ROOT / "storage_validation.json", storage)
    execution = {
        "schema": "eidosoma.e01.s13r_execution_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "adapterPassed": adapter["passed"],
        "adapterTaskPassCount": adapter["taskPassCount"],
        "sourceCachePassed": cache_validation["passed"],
        "priorImmutabilityPassed": prior["passed"],
        "downstreamSchemaPassed": downstream["passed"],
        "candidateStatisticsComputed": success,
        "statisticsReplayPassed": success,
        "schemaArtifactsPassed": schema["passed"],
        "validationResult": (
            "PASS_ONE_REPAIR_AND_ALL_FROZEN_S13_GATES"
            if success
            else "FAIL_CLOSED_AFTER_AUTHORIZED_ADAPTER_AT_NEW_UNAUTHORIZED_SCHEMA_REQUIREMENT"
        ),
        "allValidationGatesPassed": success,
    }
    write_json(STEP_ROOT / "execution_validation.json", execution)
    provenance = {
        "schema": "eidosoma.e01.s13r_provenance_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "designCommit": method_lock["designCommit"],
        "branch": git("branch", "--show-current"),
        "s13ClassificationRetained": "S13_VALIDATION_FAILED_CLOSED",
        "s13SourceCacheRoot": str(TASK_ROOT),
        "derivedViewRoot": str(NORMALIZED_LABEL_ROOT),
        "sourceTaskCount": 200,
        "trajectoryGenerated": False,
        "sourceFitRerun": False,
        "partialS13ConcatenationRead": False,
        "subsetAnalysis": False,
        "candidate1Excluded": True,
        "adapterId": ADAPTER_ID,
        "statisticsImplementation": "scripts/e01/run_s13_confirmed_timebase_scaleup.py",
        "statisticsSeedsChanged": False,
        "passed": True,
    }
    write_json(STEP_ROOT / "provenance_manifest.json", provenance)
    scope = json.loads((STEP_ROOT / "scope_access_ledger.json").read_text())
    scope["events"].append(
        {
            "stage": "S13R_TERMINAL" if not success else "S13R_COMPLETE",
            "labelAdapterApplied": True,
            "candidateStatisticOpened": success,
            "sourceFitRerun": False,
            "newTrajectoryGenerated": False,
            "partialS13ConcatenationUsed": False,
            "subsetAnalysis": False,
            "secondRepairApplied": False,
            "laterWorkAccessed": False,
            "status": "PASS" if success else "PERMANENT_STOP",
            "reason": reason,
        }
    )
    scope["success"] = success
    write_json(STEP_ROOT / "scope_access_ledger.json", scope)
    status = {
        "researchStepId": VERSION,
        "stepNumber": "S13R",
        "success": success,
        "status": (
            "COMPLETED_AT_MANDATORY_S13R_HUMAN_REVIEW_BOUNDARY"
            if success
            else "PERMANENTLY_STOPPED_UNDER_ONE_REPAIR_RULE"
        ),
        "artifactsWritten": [
            *config["artifacts"]["required"],
        ],
        "validationResult": execution["validationResult"],
        "caveatsOrBlockers": [
            "S13 remains byte-for-byte immutable and S13_VALIDATION_FAILED_CLOSED.",
            "This is an explicit post-failure human override and weakens procedural credibility.",
            reason or "Public-source reconstruction is not author- or paper-primary.",
            "S14-S18 and every later activity remain blocked.",
        ],
        "recommendedNextAction": (
            "Mandatory human review; close the one-repair path and do not continue automatically."
            if not success
            else "Mandatory human review; do not continue automatically."
        ),
        "outcomeClassification": classification_token,
        "scientificAssociationClassification": scientific_status,
        "laterWorkStatus": "BLOCKED_PENDING_S13R_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "status.json", status)
    if success and results is not None:
        report = complete_report(classification_payload, results, adapter, runtime, 0)
    else:
        report = stop_report(reason or "unknown terminal reason", adapter, downstream, runtime, 0)
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    manifest = artifact_manifest(config)
    if not manifest["passed"]:
        raise RuntimeError(f"S13R artifact manifest incomplete: {manifest['requiredMissing']}")
    if success and results is not None:
        report = complete_report(
            classification_payload, results, adapter, runtime, manifest["artifactCountExcludingSelf"] + 1
        )
    else:
        report = stop_report(
            reason or "unknown terminal reason",
            adapter,
            downstream,
            runtime,
            manifest["artifactCountExcludingSelf"] + 1,
        )
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    final_manifest = artifact_manifest(config)
    if not final_manifest["passed"]:
        raise RuntimeError("S13R final artifact manifest validation failed")


def main() -> int:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    method_lock = verify_method_lock()
    prior = validate_prior_artifacts()
    cache_validation = validate_source_cache()
    if not prior["passed"] or not cache_validation["passed"]:
        raise RuntimeError("S13R prior-artifact or source-cache immutability gate failed")
    write_json(
        STEP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s13r_implementation_lock.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "designCommit": method_lock["designCommit"],
            "implementationCommit": git("rev-parse", "HEAD"),
            "remoteCommit": git("rev-parse", "origin/eidosoma/groups/42"),
            "candidateStatisticOpenedBeforeLock": False,
            "sourceFitRerun": False,
            "newTrajectoryGenerated": False,
            "passed": True,
        },
    )
    adapter, _, _ = normalize_labels()
    if not adapter["passed"]:
        downstream = {
            "schema": "eidosoma.e01.s13r_downstream_schema_validation.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "tables": [],
            "incompatibleTables": [],
            "passed": False,
        }
        write_json(STEP_ROOT / "downstream_schema_validation.json", downstream)
        reason = "AUTHORIZED_LABEL_SCHEMA_ADAPTER_GATE_FAILED"
        write_common_final_artifacts(
            config=config,
            method_lock=method_lock,
            adapter=adapter,
            downstream=downstream,
            prior=prior,
            cache_validation=cache_validation,
            started_wall=started_wall,
            started_cpu=started_cpu,
            success=False,
            classification_token="S13R_REPAIR_PATH_PERMANENTLY_STOPPED",
            scientific_status="NOT_EVALUATED",
            reason=reason,
            results=None,
            classification_payload=None,
        )
        print(json.dumps({"stage": "S13R_permanent_stop", "reason": reason}))
        return 2
    downstream = downstream_schema_validation()
    if not downstream["passed"]:
        reason = "ADDITIONAL_UNAUTHORIZED_SOURCE_TABLE_SCHEMA_NORMALIZATION_REQUIRED:"
        reason += ",".join(downstream["incompatibleTables"])
        write_common_final_artifacts(
            config=config,
            method_lock=method_lock,
            adapter=adapter,
            downstream=downstream,
            prior=prior,
            cache_validation=cache_validation,
            started_wall=started_wall,
            started_cpu=started_cpu,
            success=False,
            classification_token="S13R_REPAIR_PATH_PERMANENTLY_STOPPED",
            scientific_status="NOT_EVALUATED",
            reason=reason,
            results=None,
            classification_payload=None,
        )
        print(json.dumps({"stage": "S13R_permanent_stop", "reason": reason}))
        return 2
    frames = collate_all_sources()
    eligible = frames["prefix_endpoint_values.parquet"][
        frames["prefix_endpoint_values.parquet"]["priorLockedClockTransitions"] >= 256
    ]
    full = frames["full_source_values.parquet"]
    suffix = frames["replay_suffix_validation.parquet"]
    failures = frames["worker_failures"]
    full_coverage = full.assign(
        numeric=np.isfinite(pd.to_numeric(full["emergence"], errors="coerce"))
    ).groupby(["candidateId", "implementationId"])["numeric"].mean()
    prefix_coverage = eligible.assign(
        numeric=np.isfinite(pd.to_numeric(eligible["emergence"], errors="coerce"))
    ).groupby(["candidateId", "implementationId"])["numeric"].mean()
    executed_suffix = suffix[suffix["sentinel"] != "non_sentinel"]
    source_gate = bool(
        len(failures) == 0
        and full["exactReplayPassed"].astype(bool).all()
        and eligible["exactReplayPassed"].astype(bool).all()
        and suffix["structuralExact"].astype(bool).all()
        and executed_suffix["resultExact"].fillna(False).astype(bool).all()
        and len(executed_suffix) == 200 * 2 * 3 * 3
        and float(full_coverage.min()) >= 0.80
        and float(prefix_coverage.min()) >= 0.80
    )
    if not source_gate:
        reason = "FROZEN_S13_SOURCE_REPLAY_COVERAGE_OR_SUFFIX_GATE_FAILED"
        write_common_final_artifacts(
            config=config,
            method_lock=method_lock,
            adapter=adapter,
            downstream=downstream,
            prior=prior,
            cache_validation=cache_validation,
            started_wall=started_wall,
            started_cpu=started_cpu,
            success=False,
            classification_token="S13R_REPAIR_PATH_PERMANENTLY_STOPPED",
            scientific_status="NOT_EVALUATED",
            reason=reason,
            results=None,
            classification_payload=None,
        )
        return 2
    results, classification, _ = run_frozen_statistics(frames)
    s13.FIGURE_ROOT = STEP_ROOT / "figures"
    s13.make_figures(
        results["candidate_associations"],
        results["candidate_association_details"],
        results["replicator_drift_details"],
        results["temporal_dependence_results"],
        results["spike_results"],
        results["future_dependence_results"],
        results["metric_identity_results"],
        results["ensemble_adjudication"],
    )
    write_common_final_artifacts(
        config=config,
        method_lock=method_lock,
        adapter=adapter,
        downstream=downstream,
        prior=prior,
        cache_validation=cache_validation,
        started_wall=started_wall,
        started_cpu=started_cpu,
        success=True,
        classification_token=classification["classification"],
        scientific_status=classification["classification"],
        reason=None,
        results=results,
        classification_payload=classification,
    )
    print(
        json.dumps(
            {
                "stage": "S13R_complete",
                "classification": classification["classification"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
