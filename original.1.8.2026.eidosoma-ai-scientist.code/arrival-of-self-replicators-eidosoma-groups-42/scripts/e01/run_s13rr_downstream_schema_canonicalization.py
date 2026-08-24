#!/usr/bin/env python3
"""Canonicalize exact S13 downstream schemas, then conditionally replay statistics."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
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

from e01_s13r_schema_normalization.core import (
    array_value_sha256,
    null_mask_sha256,
)
from e01_s13rr_downstream_schema_canonicalization.core import (
    ADAPTER_ID,
    DOWNSTREAM_AFFECTED_TASKS,
    LABEL_AFFECTED_TASKS,
    RESEARCH_STEP_ID,
    TABLE_FAMILIES,
    VERSION,
    canonicalize_table,
    physical_field_sha256,
    schema_sha256,
    schema_signature,
)
from scripts.e01 import run_s12g_frozen_timebase_ensemble as backend
from scripts.e01 import run_s13_confirmed_timebase_scaleup as s13
from scripts.e01 import run_s13r_schema_normalization_confirmation as s13r

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S13RR"
S13_ROOT = ARTIFACTS / "research_steps/S13"
S13R_ROOT = ARTIFACTS / "research_steps/S13R"
SOURCE_CACHE_ROOT = Path("/cache/e01_s13")
TASK_ROOT = SOURCE_CACHE_ROOT / "source_results"
DERIVED_ROOT = Path("/cache/e01_s13rr")
VIEW_ROOT = DERIVED_ROOT / "canonical_views"
COLLATED_ROOT = DERIVED_ROOT / "strict_collation"
CONFIG = (
    REPO / "configs/e01/s13rr_downstream_schema_canonicalization_preregistration.yaml"
)
S12G_SCHEMAS = REPO / "configs/e01/s12g_output_schemas.json"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")

SOURCE_TO_TARGET = {
    "labels.parquet": "label_values.parquet",
    "preprocessing.parquet": "preprocessing_diagnostics.parquet",
    "full.parquet": "full_source_values.parquet",
    "prefix.parquet": "prefix_endpoint_values.parquet",
    "partition.parquet": "partition_history.parquet",
    "diagnostic.parquet": "source_diagnostic_outputs.parquet",
    "suffix.parquet": "replay_suffix_validation.parquet",
    "seeds.parquet": "seed_manifest.parquet",
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
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


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


def verify_method_lock() -> dict[str, Any]:
    lock = json.loads((STEP_ROOT / "method_lock.json").read_text())
    if not lock.get("passed"):
        raise RuntimeError("S13RR method lock is not passing")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote or git("status", "--short"):
        raise RuntimeError("S13RR must execute from a clean pushed commit")
    if head != lock["designCommit"]:
        raise RuntimeError("S13RR HEAD differs from the frozen design commit")
    for row in lock["files"]:
        path = REPO / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"S13RR method file changed: {row['path']}")
    return lock


def validate_prior() -> dict[str, Any]:
    baseline = json.loads((STEP_ROOT / "immutable_prior_baseline.json").read_text())
    changed = []
    for row in baseline["files"]:
        path = Path(row["path"])
        actual = sha256_file(path) if path.is_file() else None
        if actual != row["sha256"] or (
            path.is_file() and path.stat().st_size != row["bytes"]
        ):
            changed.append(
                {
                    "path": str(path),
                    "expectedSha256": row["sha256"],
                    "actualSha256": actual,
                }
            )
    s13_class = json.loads((S13_ROOT / "classification.json").read_text())
    s13r_class = json.loads((S13R_ROOT / "classification.json").read_text())
    passed = bool(
        not changed
        and s13_class["classification"] == "S13_VALIDATION_FAILED_CLOSED"
        and s13r_class["classification"] == "S13R_REPAIR_PATH_PERMANENTLY_STOPPED"
        and not s13_class["candidateSpecificStatisticsComputed"]
        and not s13r_class["candidateSpecificStatisticsComputed"]
    )
    payload = {
        "schema": "eidosoma.e01.s13rr_immutable_prior_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "fileCount": len(baseline["files"]),
        "changedCount": len(changed),
        "changed": changed,
        "s13ClassificationRetained": s13_class["classification"],
        "s13rClassificationRetained": s13r_class["classification"],
        "passed": passed,
    }
    write_json(STEP_ROOT / "immutable_prior_validation.json", payload)
    return payload


def validate_cache() -> dict[str, Any]:
    manifest = json.loads((STEP_ROOT / "source_cache_input_manifest.json").read_text())
    changed = []
    for row in manifest["files"]:
        path = SOURCE_CACHE_ROOT / row["relativePath"]
        actual = sha256_file(path) if path.is_file() else None
        if (
            actual != row["sha256"]
            or not path.is_file()
            or path.stat().st_size != row["bytes"]
        ):
            changed.append(
                {"relativePath": row["relativePath"], "actualSha256": actual}
            )
    payload = {
        "schema": "eidosoma.e01.s13rr_source_cache_input_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "expectedFileCount": 2000,
        "checkedFileCount": len(manifest["files"]),
        "changedOrMissingCount": len(changed),
        "changedOrMissing": changed,
        "passed": len(manifest["files"]) == 2000 and not changed,
    }
    write_json(STEP_ROOT / "source_cache_input_validation.json", payload)
    return payload


def source_path(candidate: str, matrix_index: int, family: str) -> Path:
    return TASK_ROOT / candidate / f"M{matrix_index:02d}" / family


def task_identity_ok(table: pa.Table, candidate: str, matrix_index: int) -> bool:
    if table.num_rows == 0:
        return True
    checks = []
    if "candidateId" in table.column_names:
        checks.append(set(table["candidateId"].to_pylist()) == {candidate})
    if "matrixIndex" in table.column_names:
        checks.append(set(table["matrixIndex"].to_pylist()) == {matrix_index})
    return all(checks) if checks else False


def canonicalize_all() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if VIEW_ROOT.exists():
        raise RuntimeError("S13RR derived-view root already exists; rerun forbidden")
    canonical_schemas = {
        family: pq.read_table(source_path(CANDIDATES[0], 0, family)).schema
        for family in TABLE_FAMILIES
    }
    task_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    view_manifest: list[dict[str, Any]] = []
    adaptation_tasks: dict[str, set[str]] = defaultdict(set)
    all_pass = True

    for family in TABLE_FAMILIES:
        canonical_schema = canonical_schemas[family]
        for candidate in CANDIDATES:
            for matrix_index in range(100):
                task_id = f"{candidate}/M{matrix_index:02d}"
                path = source_path(candidate, matrix_index, family)
                source_hash_before = sha256_file(path)
                source = pq.read_table(path)
                view, adapted_fields, operation = canonicalize_table(
                    family=family,
                    task_id=task_id,
                    source=source,
                    canonical_schema=canonical_schema,
                )
                if adapted_fields:
                    adaptation_tasks[family].add(task_id)
                view_path = (
                    VIEW_ROOT
                    / family.removesuffix(".parquet")
                    / candidate
                    / f"M{matrix_index:02d}.parquet"
                )
                view_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(view, view_path, compression="zstd")
                roundtrip = pq.read_table(view_path)
                source_hash_after = sha256_file(path)
                shared_fields = [
                    name
                    for name in source.column_names
                    if name in roundtrip.column_names
                ]
                values_equal = all(
                    array_value_sha256(source[name])
                    == array_value_sha256(roundtrip[name])
                    for name in shared_fields
                )
                masks_equal = all(
                    null_mask_sha256(source[name]) == null_mask_sha256(roundtrip[name])
                    for name in shared_fields
                )
                non_adapter_fields = [
                    name for name in shared_fields if name not in adapted_fields
                ]
                non_adapter_physical_equal = all(
                    physical_field_sha256(source, name)
                    == physical_field_sha256(roundtrip, name)
                    for name in non_adapter_fields
                )
                row_count_unchanged = source.num_rows == roundtrip.num_rows
                zero_row_schema_only = (
                    family == "suffix.parquet"
                    and source.num_rows == 0
                    and source.num_columns == 0
                )
                column_contract = roundtrip.column_names == list(
                    canonical_schema.names
                ) and (
                    source.column_names == roundtrip.column_names
                    or zero_row_schema_only
                )
                task_pass = bool(
                    source_hash_before == source_hash_after
                    and row_count_unchanged
                    and values_equal
                    and masks_equal
                    and non_adapter_physical_equal
                    and column_contract
                    and schema_signature(roundtrip.schema)
                    == schema_signature(canonical_schema)
                    and task_identity_ok(roundtrip, candidate, matrix_index)
                )
                all_pass &= task_pass
                task_rows.append(
                    {
                        "taskId": task_id,
                        "candidateId": candidate,
                        "matrixIndex": matrix_index,
                        "family": family,
                        "operation": operation,
                        "adaptedFieldsJson": json.dumps(
                            list(adapted_fields), separators=(",", ":")
                        ),
                        "sourceRows": source.num_rows,
                        "viewRows": roundtrip.num_rows,
                        "rowsInvented": max(0, roundtrip.num_rows - source.num_rows),
                        "rowsOmitted": max(0, source.num_rows - roundtrip.num_rows),
                        "sourceColumns": source.num_columns,
                        "viewColumns": roundtrip.num_columns,
                        "sourceSha256Before": source_hash_before,
                        "sourceSha256After": source_hash_after,
                        "sourceSchemaSha256": schema_sha256(source.schema),
                        "viewSchemaSha256": schema_sha256(roundtrip.schema),
                        "valuesEqual": values_equal,
                        "nullMasksEqual": masks_equal,
                        "nonAdapterPhysicalEqual": non_adapter_physical_equal,
                        "rowCountAndOrderEqual": row_count_unchanged and values_equal,
                        "taskIdentityEqual": task_identity_ok(
                            roundtrip, candidate, matrix_index
                        ),
                        "passed": task_pass,
                    }
                )
                for name in canonical_schema.names:
                    source_has = name in source.column_names
                    is_adapter = name in adapted_fields
                    field_pass = bool(
                        (not source_has and zero_row_schema_only and is_adapter)
                        or (
                            source_has
                            and array_value_sha256(source[name])
                            == array_value_sha256(roundtrip[name])
                            and null_mask_sha256(source[name])
                            == null_mask_sha256(roundtrip[name])
                            and (
                                is_adapter
                                or physical_field_sha256(source, name)
                                == physical_field_sha256(roundtrip, name)
                            )
                        )
                    )
                    field_rows.append(
                        {
                            "taskId": task_id,
                            "family": family,
                            "field": name,
                            "sourceType": str(source.schema.field(name).type)
                            if source_has
                            else None,
                            "viewType": str(roundtrip.schema.field(name).type),
                            "adapterField": is_adapter,
                            "sourceValueSha256": array_value_sha256(source[name])
                            if source_has
                            else None,
                            "viewValueSha256": array_value_sha256(roundtrip[name]),
                            "sourceNullMaskSha256": null_mask_sha256(source[name])
                            if source_has
                            else None,
                            "viewNullMaskSha256": null_mask_sha256(roundtrip[name]),
                            "physicalIdentityRequired": not is_adapter,
                            "passed": field_pass,
                        }
                    )
                    all_pass &= field_pass
                view_manifest.append(
                    {
                        "taskId": task_id,
                        "family": family,
                        "path": str(view_path),
                        "bytes": view_path.stat().st_size,
                        "sha256": sha256_file(view_path),
                    }
                )

    expected_adaptations = {
        "labels.parquet": set(LABEL_AFFECTED_TASKS),
        "prefix.parquet": set(DOWNSTREAM_AFFECTED_TASKS),
        "suffix.parquet": set(DOWNSTREAM_AFFECTED_TASKS),
        "seeds.parquet": set(DOWNSTREAM_AFFECTED_TASKS),
    }
    exact_sets = all(
        adaptation_tasks.get(family, set()) == expected
        for family, expected in expected_adaptations.items()
    )
    exact_sets &= all(
        not adaptation_tasks.get(family)
        for family in set(TABLE_FAMILIES) - set(expected_adaptations)
    )
    task_frame = pd.DataFrame(task_rows)
    field_frame = pd.DataFrame(field_rows)
    validation = {
        "schema": "eidosoma.e01.s13rr_canonicalization_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "adapterId": ADAPTER_ID,
        "familyCount": len(TABLE_FAMILIES),
        "taskViewCount": len(task_frame),
        "expectedTaskViewCount": 1600,
        "taskPassCount": int(task_frame["passed"].sum()),
        "fieldAuditCount": len(field_frame),
        "fieldPassCount": int(field_frame["passed"].sum()),
        "rowsInvented": int(task_frame["rowsInvented"].sum()),
        "rowsOmitted": int(task_frame["rowsOmitted"].sum()),
        "adaptedTaskSets": {
            key: sorted(value) for key, value in adaptation_tasks.items()
        },
        "exactAffectedTaskSets": exact_sets,
        "sourceFilesMutated": bool(
            (task_frame["sourceSha256Before"] != task_frame["sourceSha256After"]).any()
        ),
        "allRowsValuesNullMasksOrderAndKeysUnchanged": bool(task_frame["passed"].all()),
        "allNonAdapterFieldsPhysicallyUnchanged": bool(
            task_frame["nonAdapterPhysicalEqual"].all()
        ),
        "passed": bool(
            all_pass
            and exact_sets
            and len(task_frame) == 1600
            and not task_frame["rowsInvented"].sum()
            and not task_frame["rowsOmitted"].sum()
        ),
    }
    write_parquet(STEP_ROOT / "canonicalization_task_audit.parquet", task_frame)
    write_parquet(STEP_ROOT / "canonicalization_field_audit.parquet", field_frame)
    write_json(STEP_ROOT / "canonicalization_validation.json", validation)
    write_json(
        STEP_ROOT / "derived_view_manifest.json",
        {
            "schema": "eidosoma.e01.s13rr_derived_view_manifest.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "root": str(VIEW_ROOT),
            "fileCount": len(view_manifest),
            "files": view_manifest,
            "passed": len(view_manifest) == 1600
            and all(
                Path(row["path"]).is_file()
                and sha256_file(Path(row["path"])) == row["sha256"]
                for row in view_manifest
            ),
        },
    )
    return validation, task_frame, field_frame


def strict_collate() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    COLLATED_ROOT.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    family_rows = []
    all_pass = True
    for family in TABLE_FAMILIES:
        paths = [
            VIEW_ROOT
            / family.removesuffix(".parquet")
            / candidate
            / f"M{i:02d}.parquet"
            for candidate in CANDIDATES
            for i in range(100)
        ]
        tables = [pq.read_table(path) for path in paths]
        variants = Counter(schema_sha256(table.schema) for table in tables)
        schema_equal = len(variants) == 1
        try:
            collated = pa.concat_tables(tables)
            concat_passed = True
        except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as exc:
            collated = pa.table({})
            concat_passed = False
            concat_error = f"{type(exc).__name__}:{exc}"
        else:
            concat_error = None
        target = SOURCE_TO_TARGET[family]
        path = COLLATED_ROOT / target
        pq.write_table(collated, path, compression="zstd")
        roundtrip = pq.read_table(path)
        target_path = STEP_ROOT / target
        pq.write_table(roundtrip, target_path, compression="zstd")
        frames[target] = roundtrip.to_pandas()
        row_count_sum = sum(table.num_rows for table in tables)
        passed = bool(
            schema_equal and concat_passed and roundtrip.num_rows == row_count_sum
        )
        all_pass &= passed
        family_rows.append(
            {
                "family": family,
                "target": target,
                "taskCount": 200,
                "schemaVariantCount": len(variants),
                "schemaVariants": dict(variants),
                "strictConcatenationPassed": concat_passed,
                "strictConcatenationError": concat_error,
                "sourceRowTotal": row_count_sum,
                "collatedRowCount": roundtrip.num_rows,
                "collatedSchemaSha256": schema_sha256(roundtrip.schema),
                "passed": passed,
            }
        )
    validation = {
        "schema": "eidosoma.e01.s13rr_strict_collation_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "familyCount": len(family_rows),
        "onePhysicalSchemaFamilyCount": sum(
            row["schemaVariantCount"] == 1 for row in family_rows
        ),
        "strictConcatenationPassCount": sum(row["passed"] for row in family_rows),
        "families": family_rows,
        "permissivePromotionUsed": False,
        "partialOrSubsetAnalysisUsed": False,
        "passed": bool(all_pass and len(family_rows) == 8),
    }
    write_json(
        STEP_ROOT / "family_schema_manifest.json",
        {
            "schema": "eidosoma.e01.s13rr_family_schema_manifest.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "families": family_rows,
            "passed": validation["passed"],
        },
    )
    write_json(STEP_ROOT / "strict_collation_validation.json", validation)
    return frames, validation


def prefix_interface(prefix: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    original_hash = frame_hash(prefix)
    if (
        "rawObservationIndex" in prefix.columns
        or "endpointRawObservationIndex" not in prefix.columns
    ):
        raise RuntimeError("frozen S13 prefix adapter precondition failed")
    adapted = prefix.copy(deep=True)
    adapted["rawObservationIndex"] = adapted["endpointRawObservationIndex"]
    identity = adapted["rawObservationIndex"].equals(
        prefix["endpointRawObservationIndex"]
    )
    original_unchanged = frame_hash(prefix) == original_hash
    validation = {
        "schema": "eidosoma.e01.s13rr_prefix_interface_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "adapter": "FROZEN_S13_rawObservationIndex_EQUALS_endpointRawObservationIndex",
        "rowCount": len(prefix),
        "identityCount": int(
            (
                adapted["rawObservationIndex"] == adapted["endpointRawObservationIndex"]
            ).sum()
        ),
        "originalFrameUnchanged": original_unchanged,
        "passed": bool(identity and original_unchanged and len(prefix) == len(adapted)),
    }
    write_json(STEP_ROOT / "prefix_interface_validation.json", validation)
    return adapted, validation


def source_replay_gate(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    full = frames["full_source_values.parquet"]
    prefix = frames["prefix_endpoint_values.parquet"]
    suffix = frames["replay_suffix_validation.parquet"]
    diagnostics = frames["source_diagnostic_outputs.parquet"]
    task_validation = pd.read_parquet(S13_ROOT / "source_task_validation.parquet")
    eligible = prefix[prefix["priorLockedClockTransitions"] >= 256]
    executed = suffix[suffix["sentinel"] != "non_sentinel"]
    full_coverage = (
        full.assign(
            finite=np.isfinite(pd.to_numeric(full["emergence"], errors="coerce"))
        )
        .groupby(["candidateId", "implementationId"])["finite"]
        .mean()
    )
    prefix_coverage = (
        eligible.assign(
            finite=np.isfinite(pd.to_numeric(eligible["emergence"], errors="coerce"))
        )
        .groupby(["candidateId", "implementationId"])["finite"]
        .mean()
    )
    checks = {
        "workerFailureRowsZero": int(task_validation["failureRows"].sum()) == 0,
        "fullReplayAll": bool(full["exactReplayPassed"].astype(bool).all()),
        "eligiblePrefixReplayAll": bool(
            eligible["exactReplayPassed"].astype(bool).all()
        ),
        "structuralSuffixAll": bool(suffix["structuralExact"].astype(bool).all()),
        "executedSuffixResultAll": bool(
            executed["resultExact"].fillna(False).astype(bool).all()
        ),
        "executedSuffixExactCount": len(executed) == 3600,
        "minimumFullCoverage": float(full_coverage.min()) >= 0.80,
        "minimumPrefixCoverage": float(prefix_coverage.min()) >= 0.80,
        "componentIdentityError": float(
            diagnostics["componentIdentityMaxAbsError"].fillna(0).max()
        )
        <= 1e-12,
    }
    payload = {
        "schema": "eidosoma.e01.s13rr_source_replay_gate_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "frozenExpectedExecutedSuffixCount": 3600,
        "observedExecutedSuffixCount": len(executed),
        "structuralSuffixRowCount": len(suffix),
        "eligiblePrefixRowCount": len(eligible),
        "minimumFullCoverageObserved": float(full_coverage.min()),
        "minimumPrefixCoverageObserved": float(prefix_coverage.min()),
        "maximumComponentIdentityAbsoluteErrorObserved": float(
            diagnostics["componentIdentityMaxAbsError"].fillna(0).max()
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_json(STEP_ROOT / "source_replay_gate_validation.json", payload)
    return payload


def run_statistics(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    prefix = frames["prefix_endpoint_values.parquet"]
    adapted, interface = prefix_interface(prefix)
    if not interface["passed"]:
        raise RuntimeError("frozen S13 prefix interface failed")
    args = (
        frames["full_source_values.parquet"],
        adapted,
        prefix,
        frames["label_values.parquet"],
        frames["partition_history.parquet"],
    )
    first, first_class = s13.compute_statistics(*args)
    second, second_class = s13.compute_statistics(*args)
    rows = []
    for key in s13.RESULT_FILES:
        exact = True
        try:
            pd.testing.assert_frame_equal(
                first[key], second[key], check_exact=True, check_dtype=True
            )
        except AssertionError:
            exact = False
        rows.append(
            {
                "resultId": key,
                "rowCount": len(first[key]),
                "firstSha256": frame_hash(first[key]),
                "secondSha256": frame_hash(second[key]),
                "exact": exact,
            }
        )
    classification_exact = json.dumps(
        jsonable(first_class), sort_keys=True
    ) == json.dumps(jsonable(second_class), sort_keys=True)
    replay = {
        "schema": "eidosoma.e01.s13rr_statistics_replay_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "statisticsExecutions": 2,
        "results": rows,
        "classificationExact": classification_exact,
        "passed": all(row["exact"] for row in rows) and classification_exact,
    }
    write_json(STEP_ROOT / "statistics_replay_validation.json", replay)
    if not replay["passed"]:
        raise RuntimeError("exact twice-run S13 statistics replay failed")
    for key, filename in s13.RESULT_FILES.items():
        path = STEP_ROOT / filename
        (write_parquet if path.suffix == ".parquet" else write_csv)(path, first[key])
    classification = {
        **first_class,
        "schema": "eidosoma.e01.s13rr_classification.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "frozenScientificClassification": first_class["classification"],
        "evidenceLabel": "SECOND_OVERRIDE_POST_FAILURE_SCHEMA_CANONICALIZED_HELD_OUT_ANALYSIS",
        "s13ClassificationRetained": "S13_VALIDATION_FAILED_CLOSED",
        "s13rClassificationRetained": "S13R_REPAIR_PATH_PERMANENTLY_STOPPED",
        "candidate1Excluded": True,
        "laterWorkStatus": "BLOCKED_PENDING_S13RR_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "classification.json", classification)
    return first, classification, replay


def empty_results() -> None:
    for filename, columns in s13r.RESULT_COLUMNS.items():
        frame = pd.DataFrame(columns=columns)
        path = STEP_ROOT / filename
        (write_parquet if path.suffix == ".parquet" else write_csv)(path, frame)


def artifact_manifest(config: dict[str, Any]) -> dict[str, Any]:
    required = config["artifacts"]["required"]
    missing = [
        name
        for name in required
        if name != "artifact_manifest.json" and not (STEP_ROOT / name).is_file()
    ]
    files = [
        path
        for path in sorted(STEP_ROOT.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    rows = [
        {
            "path": str(path.relative_to(STEP_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    payload = {
        "schema": "eidosoma.e01.s13rr_artifact_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "artifactCountExcludingSelf": len(rows),
        "artifacts": rows,
        "requiredMissing": missing,
        "passed": not missing,
    }
    write_json(STEP_ROOT / "artifact_manifest.json", payload)
    return payload


def schema_validation(scientific_reached: bool) -> dict[str, Any]:
    required_columns = json.loads(S12G_SCHEMAS.read_text())["tables"]
    rows = []
    for filename in [*SOURCE_TO_TARGET.values(), *s13r.RESULT_COLUMNS]:
        path = STEP_ROOT / filename
        frame = (
            pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        )
        expected = required_columns.get(filename, s13r.RESULT_COLUMNS.get(filename))
        passed = list(frame.columns) == list(expected)
        rows.append(
            {
                "artifact": filename,
                "rowCount": len(frame),
                "columnsExact": passed,
                "scientificReached": scientific_reached,
            }
        )
    payload = {
        "schema": "eidosoma.e01.s13rr_schema_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "artifacts": rows,
        "passed": all(row["columnsExact"] for row in rows),
    }
    write_json(STEP_ROOT / "schema_validation.json", payload)
    return payload


def build_report(
    *,
    success: bool,
    reason: str | None,
    classification: dict[str, Any],
    canonical: dict[str, Any],
    collation: dict[str, Any],
    source_gate: dict[str, Any],
    runtime: dict[str, Any],
    artifact_count: int,
) -> str:
    scientific = classification.get(
        "frozenScientificClassification",
        classification.get("scientificAssociationClassification", "NOT_EVALUATED"),
    )
    completion = (
        "COMPLETED_AT_MANDATORY_HUMAN_REVIEW_BOUNDARY"
        if success
        else "PERMANENTLY_STOPPED_NO_FURTHER_REPAIR"
    )
    outcome = classification["classification"]
    return "\n".join(
        [
            "# S13RR Full Results: Downstream Schema Canonicalization",
            "",
            "## Top summary",
            "",
            f"- **Research step ID:** `{VERSION}` (S13RR).",
            f"- **Completion status:** `{completion}`.",
            f"- **Artifacts written:** {artifact_count} status-bearing artifacts under `/artifacts/research_steps/S13RR/`, including the frozen contract, 1,600-view task audit, field audit, eight-family schemas, replay gate, complete suppression or scientific tables, provenance, manifests, and this report.",
            f"- **Validation result:** canonical derived views {canonical.get('taskPassCount', 0)}/{canonical.get('taskViewCount', 0)}; strict family schemas {collation.get('onePhysicalSchemaFamilyCount', 0)}/8; frozen source/replay gate `{source_gate.get('passed')}`.",
            f"- **Outcome classification:** `{outcome}`; scientific held-out status `{scientific}`.",
            f"- **Caveats or blockers:** {reason or 'No operational blocker, but this is a second post-failure override and cannot identify the author implementation.'}",
            "- **Lay summary:** The final override can standardize only how known all-null and empty files are represented; it cannot change a simulation or scientific value. The analysis proceeds only if the original replay contract still passes exactly.",
            "- **Recommended next action:** Mandatory human review. Keep S14–S18, prediction, interventions, E02, report-bundle progression, and further scale-up blocked.",
            "",
            "## Frozen question",
            "",
            "Can exact, value-preserving views remove the diagnosed physical-schema variants and allow the unchanged held-out S13 analysis to run twice identically?",
            "",
            "## Inputs",
            "",
            "Only the 200 frozen S13 task bundles (2,000 hash-locked files) and prior S13/S13R evidence were used. No trajectory, source fit, partial concatenation, subset, candidate 1 result, or S12G scientific cache was used.",
            "",
            "## Detailed methods",
            "",
            "The pushed adapter reproduced the exact S13R label typing, typed seven all-null prefix fields and one all-null seed endpoint field in the two matrix-72 tasks, and assigned the canonical schema to their two 0-row/0-column suffix tables. It then required zero invented/omitted rows, unchanged source hashes, logical values, null masks, order, keys, task identities, and physical identity for every non-adapter field. All eight families were strictly concatenated without promotion. The original S13 source/replay gate—including its exact executed-suffix cardinality—was evaluated before any statistic. Conditional statistics were the unmodified twice-run S13 procedures with original 4,096-replicate seeds and gates.",
            "",
            "## Commands",
            "",
            "```bash",
            "PYTHONPATH=src python -m pytest -q tests/e01/test_s13rr_downstream_schema_canonicalization.py tests/e01/test_s13r_schema_normalization_confirmation.py tests/e01/test_s13_confirmed_timebase_scaleup.py",
            "python -m ruff check src/e01_s13rr_downstream_schema_canonicalization scripts/e01/freeze_s13rr_preregistration.py scripts/e01/run_s13rr_downstream_schema_canonicalization.py tests/e01/test_s13rr_downstream_schema_canonicalization.py",
            "ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s13rr_preregistration.py --record-commit",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13rr_downstream_schema_canonicalization.py",
            "```",
            "",
            "## Results",
            "",
            f"- Canonical views: {canonical.get('taskPassCount', 0)}/{canonical.get('taskViewCount', 0)} tasks and {canonical.get('fieldPassCount', 0)}/{canonical.get('fieldAuditCount', 0)} fields passed.",
            f"- Invented rows: {canonical.get('rowsInvented', 0)}; omitted rows: {canonical.get('rowsOmitted', 0)}; source files mutated: {canonical.get('sourceFilesMutated')}.",
            f"- Strict table-family schemas: {collation.get('onePhysicalSchemaFamilyCount', 0)}/8; strict concatenations: {collation.get('strictConcatenationPassCount', 0)}/8.",
            f"- Frozen executed-suffix count: observed {source_gate.get('observedExecutedSuffixCount')} versus required {source_gate.get('frozenExpectedExecutedSuffixCount')}.",
            f"- Candidate statistics and two-candidate adjudication executed: {success}.",
            "",
            "## Validation",
            "",
            "The method was committed and pushed before candidate statistics. All prior artifacts and S13 cache inputs were hash-checked before and after view construction. The original S13 and S13R classifications remain unchanged. The statistical result bundle is either exact across two executions or schema-bearing and explicitly suppressed.",
            "",
            "## Runtime and storage",
            "",
            f"Wall time was {runtime.get('wallSeconds', 0):.3f} seconds and process CPU time {runtime.get('processCpuSeconds', 0):.3f} seconds; no simulator, source worker, or GPU was used. Derived cache bytes: {runtime.get('derivedCacheBytes', 0)}.",
            "",
            "## Caveats and limitations",
            "",
            "- This second override follows a prior promise of permanent termination and substantially weakens confirmatory credibility.",
            "- Typed nulls and schemas do not create scientific observations or establish source validity.",
            "- Full fits remain retrospective and future-dependent; pre-256, fixed-window, prediction, intervention, and author-identity questions remain unresolved.",
            "- No further repair is authorized.",
            "",
            "## Provenance",
            "",
            "The pushed commit, complete prior/cache baselines, task/field audits, source and view hashes, strict collation schemas, replay gate, scope ledger, runtime/storage records, status, and artifact manifest preserve the full chain. Bulky derived views remain under `/cache/e01_s13rr/`.",
            "",
            "## Recommended next action",
            "",
            "Return for mandatory human review and begin no later step automatically.",
            "",
        ]
    )


def finalize(
    *,
    config: dict[str, Any],
    lock: dict[str, Any],
    prior: dict[str, Any],
    cache: dict[str, Any],
    canonical: dict[str, Any],
    collation: dict[str, Any],
    source_gate: dict[str, Any],
    started_wall: float,
    started_cpu: float,
    success: bool,
    reason: str | None,
    results: dict[str, pd.DataFrame] | None,
    classification: dict[str, Any] | None,
) -> None:
    if results is None:
        empty_results()
        write_json(
            STEP_ROOT / "statistics_replay_validation.json",
            {
                "schema": "eidosoma.e01.s13rr_statistics_replay_validation.v1",
                "researchStepId": RESEARCH_STEP_ID,
                "statisticsExecutions": 0,
                "status": "NOT_REACHED_PERMANENT_STOP",
                "reason": reason,
                "passed": False,
            },
        )
        if not (STEP_ROOT / "prefix_interface_validation.json").exists():
            write_json(
                STEP_ROOT / "prefix_interface_validation.json",
                {
                    "schema": "eidosoma.e01.s13rr_prefix_interface_validation.v1",
                    "researchStepId": RESEARCH_STEP_ID,
                    "status": "NOT_REACHED_PERMANENT_STOP",
                    "reason": reason,
                    "passed": False,
                },
            )
    if classification is None:
        classification = {
            "schema": "eidosoma.e01.s13rr_classification.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "classification": "S13RR_REPAIR_PATH_PERMANENTLY_STOPPED",
            "scientificAssociationClassification": "NOT_EVALUATED",
            "reason": reason,
            "evidenceLabel": "SECOND_OVERRIDE_POST_FAILURE_SCHEMA_CANONICALIZED_HELD_OUT_ANALYSIS_NOT_REACHED",
            "s13ClassificationRetained": "S13_VALIDATION_FAILED_CLOSED",
            "s13rClassificationRetained": "S13R_REPAIR_PATH_PERMANENTLY_STOPPED",
            "candidate1Excluded": True,
            "candidateSpecificStatisticsComputed": False,
            "twoCandidateAdjudicationPerformed": False,
            "laterWorkStatus": "BLOCKED_PENDING_S13RR_HUMAN_REVIEW",
        }
        write_json(STEP_ROOT / "classification.json", classification)

    failure_columns = json.loads(S12G_SCHEMAS.read_text())["tables"][
        "failure_ledger.csv"
    ]
    if success:
        full = pd.read_parquet(STEP_ROOT / "full_source_values.parquet")
        prefix = pd.read_parquet(STEP_ROOT / "prefix_endpoint_values.parquet")
        rows = backend.failure_rows_from_statuses(full, prefix, pd.DataFrame())
        for row in rows:
            row["failureId"] = str(row["failureId"]).replace("S12G-", "S13RR-")
        failures = pd.DataFrame(rows, columns=failure_columns)
    else:
        failures = pd.DataFrame(
            [
                {
                    "failureId": "S13RR-TERMINAL-NO-FURTHER-REPAIR",
                    "stage": "post_canonicalization_frozen_gate",
                    "candidateId": None,
                    "trajectoryId": None,
                    "implementationId": None,
                    "temporalModeId": None,
                    "endpointGeneration": None,
                    "severity": "FATAL",
                    "status": "S13RR_REPAIR_PATH_PERMANENTLY_STOPPED",
                    "reason": reason,
                    "gateImpact": "PERMANENT_STOP_NO_STATISTICS_NO_ANOTHER_REPAIR",
                    "repairAttempted": False,
                }
            ],
            columns=failure_columns,
        )
    write_csv(STEP_ROOT / "failure_ledger.csv", failures)

    schema = schema_validation(success)
    prior_post = validate_prior()
    cache_post = validate_cache()
    derived_bytes = sum(
        path.stat().st_size for path in DERIVED_ROOT.rglob("*") if path.is_file()
    )
    artifact_bytes = sum(
        path.stat().st_size for path in STEP_ROOT.rglob("*") if path.is_file()
    )
    runtime = {
        "schema": "eidosoma.e01.s13rr_runtime_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "wallSeconds": time.perf_counter() - started_wall,
        "processCpuSeconds": time.process_time() - started_cpu,
        "simulationWorkerCpuSeconds": 0.0,
        "sourceFitWorkerCpuSeconds": 0.0,
        "gpuHours": 0.0,
        "workers": 1,
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
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "derivedCacheBytes": derived_bytes,
        "artifactBytesBeforeManifest": artifact_bytes,
        "passed": True,
    }
    write_json(STEP_ROOT / "runtime_manifest.json", runtime)
    write_json(
        STEP_ROOT / "storage_validation.json",
        {
            "schema": "eidosoma.e01.s13rr_storage_validation.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "derivedCacheBytes": derived_bytes,
            "artifactBytesBeforeManifest": artifact_bytes,
            "retainedArtifactGiBCeiling": 30.0,
            "passed": artifact_bytes <= 30 * 1024**3,
        },
    )
    write_json(
        STEP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s13rr_implementation_lock.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "designCommit": lock["designCommit"],
            "adapterId": ADAPTER_ID,
            "canonicalizationPassed": canonical["passed"],
            "strictCollationPassed": collation["passed"],
            "sourceReplayGatePassed": source_gate["passed"],
            "additionalModificationRequired": not source_gate["passed"],
            "passed": canonical["passed"] and collation["passed"],
        },
    )
    write_json(
        STEP_ROOT / "provenance_manifest.json",
        {
            "schema": "eidosoma.e01.s13rr_provenance_manifest.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "designCommit": lock["designCommit"],
            "branch": git("branch", "--show-current"),
            "s13ClassificationRetained": "S13_VALIDATION_FAILED_CLOSED",
            "s13rClassificationRetained": "S13R_REPAIR_PATH_PERMANENTLY_STOPPED",
            "sourceCacheRoot": str(TASK_ROOT),
            "derivedViewRoot": str(VIEW_ROOT),
            "sourceTaskCount": 200,
            "derivedViewCount": 1600,
            "trajectoryGenerated": False,
            "sourceFitRerun": False,
            "partialConcatenationRead": False,
            "subsetAnalysis": False,
            "candidate1Excluded": True,
            "statisticsImplementation": "scripts/e01/run_s13_confirmed_timebase_scaleup.py",
            "statisticsSeedsChanged": False,
            "passed": prior_post["passed"] and cache_post["passed"],
        },
    )
    scope = json.loads((STEP_ROOT / "scope_access_ledger.json").read_text())
    scope["events"].append(
        {
            "stage": "S13RR_COMPLETE" if success else "S13RR_PERMANENT_STOP",
            "derivedViewsConstructed": True,
            "candidateStatisticOpened": success,
            "sourceFitRerun": False,
            "newTrajectoryGenerated": False,
            "partialConcatenationUsed": False,
            "subsetAnalysis": False,
            "candidate1Accessed": False,
            "laterWorkAccessed": False,
            "status": "PASS" if success else "PERMANENT_STOP",
            "reason": reason,
        }
    )
    scope["success"] = success
    write_json(STEP_ROOT / "scope_access_ledger.json", scope)
    execution = {
        "schema": "eidosoma.e01.s13rr_execution_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "priorImmutabilityPassed": prior_post["passed"],
        "sourceCachePassed": cache_post["passed"],
        "canonicalizationPassed": canonical["passed"],
        "strictCollationPassed": collation["passed"],
        "sourceReplayGatePassed": source_gate["passed"],
        "candidateStatisticsComputed": success,
        "statisticsReplayPassed": success,
        "schemaPassed": schema["passed"],
        "validationResult": "PASS_ALL_FROZEN_S13RR_AND_S13_GATES"
        if success
        else "FAIL_CLOSED_AT_UNCHANGED_FROZEN_SOURCE_REPLAY_GATE_AFTER_SCHEMA_CANONICALIZATION",
        "allValidationGatesPassed": success,
    }
    write_json(STEP_ROOT / "execution_validation.json", execution)
    status = {
        "researchStepId": VERSION,
        "stepNumber": "S13RR",
        "success": success,
        "status": "COMPLETED_AT_MANDATORY_S13RR_HUMAN_REVIEW_BOUNDARY"
        if success
        else "PERMANENTLY_STOPPED_NO_FURTHER_REPAIR",
        "artifactsWritten": config["artifacts"]["required"],
        "validationResult": execution["validationResult"],
        "caveatsOrBlockers": [
            "S13 remains S13_VALIDATION_FAILED_CLOSED and S13R remains S13R_REPAIR_PATH_PERMANENTLY_STOPPED.",
            "This is a second post-failure override and weakens confirmatory credibility.",
            reason or "Public-source reconstruction is not author- or paper-primary.",
            "S14-S18 and every later activity remain blocked.",
        ],
        "recommendedNextAction": "Mandatory human review; permanently close this path and do not continue automatically.",
        "outcomeClassification": classification["classification"],
        "scientificAssociationClassification": classification.get(
            "frozenScientificClassification",
            classification.get("scientificAssociationClassification"),
        ),
        "laterWorkStatus": "BLOCKED_PENDING_S13RR_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "status.json", status)
    report = build_report(
        success=success,
        reason=reason,
        classification=classification,
        canonical=canonical,
        collation=collation,
        source_gate=source_gate,
        runtime=runtime,
        artifact_count=0,
    )
    (STEP_ROOT / "research_step_full_results.md").write_text(report)
    manifest = artifact_manifest(config)
    if not manifest["passed"]:
        raise RuntimeError(
            f"S13RR artifact completeness failed: {manifest['requiredMissing']}"
        )
    report = build_report(
        success=success,
        reason=reason,
        classification=classification,
        canonical=canonical,
        collation=collation,
        source_gate=source_gate,
        runtime=runtime,
        artifact_count=manifest["artifactCountExcludingSelf"] + 1,
    )
    (STEP_ROOT / "research_step_full_results.md").write_text(report)
    manifest = artifact_manifest(config)
    if not manifest["passed"]:
        raise RuntimeError("S13RR final artifact manifest failed")


def main() -> int:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    config = yaml.safe_load(CONFIG.read_text())
    lock = verify_method_lock()
    prior = validate_prior()
    cache = validate_cache()
    if not prior["passed"] or not cache["passed"]:
        raise RuntimeError("S13RR immutable prior/cache gate failed")
    canonical, _, _ = canonicalize_all()
    if not canonical["passed"]:
        raise RuntimeError("S13RR declared canonicalization gate failed")
    frames, collation = strict_collate()
    if not collation["passed"]:
        source_gate = {
            "passed": False,
            "observedExecutedSuffixCount": None,
            "frozenExpectedExecutedSuffixCount": 3600,
        }
        write_json(
            STEP_ROOT / "source_replay_gate_validation.json",
            {
                "schema": "eidosoma.e01.s13rr_source_replay_gate_validation.v1",
                "researchStepId": RESEARCH_STEP_ID,
                "status": "NOT_REACHED_COLLATION_FAILURE",
                **source_gate,
            },
        )
        finalize(
            config=config,
            lock=lock,
            prior=prior,
            cache=cache,
            canonical=canonical,
            collation=collation,
            source_gate=source_gate,
            started_wall=started_wall,
            started_cpu=started_cpu,
            success=False,
            reason="UNDECLARED_STRICT_COLLATION_FAILURE",
            results=None,
            classification=None,
        )
        return 0
    source_gate = source_replay_gate(frames)
    if not source_gate["passed"]:
        failed = [name for name, passed in source_gate["checks"].items() if not passed]
        reason = "UNCHANGED_FROZEN_SOURCE_REPLAY_GATE_FAILED:" + ",".join(failed)
        finalize(
            config=config,
            lock=lock,
            prior=prior,
            cache=cache,
            canonical=canonical,
            collation=collation,
            source_gate=source_gate,
            started_wall=started_wall,
            started_cpu=started_cpu,
            success=False,
            reason=reason,
            results=None,
            classification=None,
        )
        print(
            json.dumps(
                {"stage": "S13RR_permanent_stop", "reason": reason}, sort_keys=True
            ),
            flush=True,
        )
        return 0
    results, classification, _ = run_statistics(frames)
    finalize(
        config=config,
        lock=lock,
        prior=prior,
        cache=cache,
        canonical=canonical,
        collation=collation,
        source_gate=source_gate,
        started_wall=started_wall,
        started_cpu=started_cpu,
        success=True,
        reason=None,
        results=results,
        classification=classification,
    )
    print(
        json.dumps(
            {
                "stage": "S13RR_complete",
                "classification": classification["classification"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
