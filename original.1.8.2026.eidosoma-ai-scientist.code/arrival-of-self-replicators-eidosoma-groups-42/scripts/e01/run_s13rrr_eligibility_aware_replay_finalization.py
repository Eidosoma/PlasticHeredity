#!/usr/bin/env python3
"""Apply the frozen S13RRR replay rule and replay S13 statistics twice."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
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

from e01_confirmed_timebase_scaleup.core import outcome_class
from e01_s13r_schema_normalization.core import array_value_sha256, null_mask_sha256
from e01_s13rrr_eligibility_aware_replay.core import (
    ENSEMBLE_REPORTING_ORDER,
    ENSEMBLE_SOURCE_ORDER,
    EXACT_UNAVAILABLE_TASKS,
    PREFIX_REPORTING_ORDER,
    RESEARCH_STEP_ID,
    VERSION,
    expected_slots,
    reorder_columns_exact,
)
from scripts.e01 import run_s12g_frozen_timebase_ensemble as backend
from scripts.e01 import run_s13_confirmed_timebase_scaleup as s13
from scripts.e01 import run_s13r_schema_normalization_confirmation as s13r

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S13RRR"
S13_ROOT = ARTIFACTS / "research_steps/S13"
S13R_ROOT = ARTIFACTS / "research_steps/S13R"
S13RR_ROOT = ARTIFACTS / "research_steps/S13RR"
SOURCE_CACHE_ROOT = Path("/cache/e01_s13")
VIEW_ROOT = Path("/cache/e01_s13rr/canonical_views")
DERIVED_ROOT = Path("/cache/e01_s13rrr")
CONFIG = (
    REPO
    / "configs/e01/s13rrr_eligibility_aware_replay_finalization_preregistration.yaml"
)
S12G_SCHEMAS = REPO / "configs/e01/s12g_output_schemas.json"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
FAMILIES = (
    "labels.parquet",
    "preprocessing.parquet",
    "full.parquet",
    "prefix.parquet",
    "partition.parquet",
    "diagnostic.parquet",
    "suffix.parquet",
    "seeds.parquet",
)
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
EVIDENCE_LABEL = "THIRD_OVERRIDE_POST_OUTCOME_ELIGIBILITY_EXCEPTION_ANALYSIS"


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
        raise RuntimeError("S13RRR method lock is not passing")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote or git("status", "--short"):
        raise RuntimeError("S13RRR must execute from a clean pushed commit")
    if head != lock["designCommit"]:
        raise RuntimeError("S13RRR HEAD differs from frozen design commit")
    for row in lock["files"]:
        path = REPO / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"S13RRR method file changed: {row['path']}")
    return lock


def validate_file_manifest(filename: str, *, schema: str) -> dict[str, Any]:
    manifest = json.loads((STEP_ROOT / filename).read_text())
    changed = []
    for row in manifest["files"]:
        path = Path(row["resolvedPath"])
        actual = sha256_file(path) if path.is_file() else None
        if (
            not path.is_file()
            or actual != row["sha256"]
            or path.stat().st_size != row["bytes"]
        ):
            changed.append(
                {
                    "path": str(path),
                    "expectedSha256": row["sha256"],
                    "actualSha256": actual,
                }
            )
    payload = {
        "schema": schema,
        "researchStepId": RESEARCH_STEP_ID,
        "checkedFileCount": len(manifest["files"]),
        "changedOrMissingCount": len(changed),
        "changedOrMissing": changed,
        "passed": not changed and manifest["passed"],
    }
    return payload


def validate_prior() -> dict[str, Any]:
    baseline = json.loads((STEP_ROOT / "immutable_prior_baseline.json").read_text())
    changed = []
    for row in baseline["files"]:
        path = Path(row["path"])
        actual = sha256_file(path) if path.is_file() else None
        if (
            not path.is_file()
            or actual != row["sha256"]
            or path.stat().st_size != row["bytes"]
        ):
            changed.append(
                {
                    "path": str(path),
                    "expectedSha256": row["sha256"],
                    "actualSha256": actual,
                }
            )
    retained = {
        "S13": json.loads((S13_ROOT / "classification.json").read_text())[
            "classification"
        ],
        "S13R": json.loads((S13R_ROOT / "classification.json").read_text())[
            "classification"
        ],
        "S13RR": json.loads((S13RR_ROOT / "classification.json").read_text())[
            "classification"
        ],
    }
    expected = {
        "S13": "S13_VALIDATION_FAILED_CLOSED",
        "S13R": "S13R_REPAIR_PATH_PERMANENTLY_STOPPED",
        "S13RR": "S13RR_REPAIR_PATH_PERMANENTLY_STOPPED",
    }
    payload = {
        "schema": "eidosoma.e01.s13rrr_immutable_prior_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "fileCount": len(baseline["files"]),
        "changedCount": len(changed),
        "changed": changed,
        "historicalClassificationsRetained": retained,
        "passed": not changed and retained == expected,
    }
    write_json(STEP_ROOT / "immutable_prior_validation.json", payload)
    return payload


def strict_collate() -> tuple[
    dict[str, pd.DataFrame], dict[str, Any], list[dict[str, Any]]
]:
    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    reporting_audit: list[dict[str, Any]] = []
    for family in FAMILIES:
        paths = [
            VIEW_ROOT
            / family.removesuffix(".parquet")
            / candidate
            / f"M{i:02d}.parquet"
            for candidate in CANDIDATES
            for i in range(100)
        ]
        tables = [pq.read_table(path) for path in paths]
        schema_variants = Counter(str(table.schema) for table in tables)
        collated = pa.concat_tables(tables)
        source_rows = sum(table.num_rows for table in tables)
        target = SOURCE_TO_TARGET[family]
        before = collated
        if family == "prefix.parquet":
            collated = collated.select(PREFIX_REPORTING_ORDER)
            for name in PREFIX_REPORTING_ORDER:
                reporting_audit.append(
                    {
                        "table": target,
                        "field": name,
                        "beforePosition": before.column_names.index(name),
                        "afterPosition": collated.column_names.index(name),
                        "beforeValueSha256": array_value_sha256(before[name]),
                        "afterValueSha256": array_value_sha256(collated[name]),
                        "beforeNullMaskSha256": null_mask_sha256(before[name]),
                        "afterNullMaskSha256": null_mask_sha256(collated[name]),
                        "valuesAndNullMaskUnchanged": array_value_sha256(before[name])
                        == array_value_sha256(collated[name])
                        and null_mask_sha256(before[name])
                        == null_mask_sha256(collated[name]),
                    }
                )
        target_path = STEP_ROOT / target
        pq.write_table(collated, target_path, compression="zstd")
        roundtrip = pq.read_table(target_path)
        passed = bool(
            len(schema_variants) == 1
            and roundtrip.num_rows == source_rows
            and roundtrip.column_names == collated.column_names
            and all(
                array_value_sha256(roundtrip[name])
                == array_value_sha256(collated[name])
                for name in collated.column_names
            )
        )
        rows.append(
            {
                "family": family,
                "target": target,
                "taskCount": 200,
                "schemaVariantCount": len(schema_variants),
                "sourceRowCount": source_rows,
                "collatedRowCount": roundtrip.num_rows,
                "reportingOrderChanged": family == "prefix.parquet",
                "strictConcatenationPassed": passed,
            }
        )
        frames[target] = roundtrip.to_pandas()
    validation = {
        "schema": "eidosoma.e01.s13rrr_input_collation_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "familyCount": len(rows),
        "onePhysicalSchemaFamilyCount": sum(
            row["schemaVariantCount"] == 1 for row in rows
        ),
        "strictConcatenationPassCount": sum(
            row["strictConcatenationPassed"] for row in rows
        ),
        "partialOrSubsetAnalysisUsed": False,
        "sourceValuesChanged": False,
        "families": rows,
        "passed": len(rows) == 8
        and all(row["strictConcatenationPassed"] for row in rows),
    }
    write_json(STEP_ROOT / "input_collation_validation.json", validation)
    return frames, validation, reporting_audit


def validate_availability(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    prefix = frames["prefix_endpoint_values.parquet"]
    suffix = frames["replay_suffix_validation.parquet"]
    applicable_frames: list[pd.DataFrame] = []
    unavailable_frames: list[pd.DataFrame] = []
    task_rows: list[dict[str, Any]] = []
    actual = suffix[suffix["sentinel"] != "non_sentinel"].copy()
    key_columns = [
        "candidateId",
        "matrixIndex",
        "trajectoryId",
        "implementationId",
        "endpointGeneration",
        "validationKind",
        "sentinel",
    ]
    actual_keys = actual[key_columns].copy()
    duplicate_actual = int(actual_keys.duplicated().sum())
    actual_set = set(map(tuple, actual_keys.itertuples(index=False, name=None)))
    for candidate in CANDIDATES:
        for matrix_index in range(100):
            task = prefix[
                (prefix["candidateId"] == candidate)
                & (prefix["matrixIndex"] == matrix_index)
            ]
            trajectories = task["trajectoryId"].dropna().unique().tolist()
            if len(trajectories) != 1:
                raise RuntimeError(
                    f"task trajectory identity is not unique: {candidate}/M{matrix_index:02d}"
                )
            applicable, unavailable = expected_slots(
                candidate_id=candidate,
                matrix_index=matrix_index,
                trajectory_id=str(trajectories[0]),
                prefix=task,
            )
            if len(applicable):
                applicable_frames.append(applicable)
            if len(unavailable):
                unavailable_frames.append(unavailable)
            task_actual = actual[
                (actual["candidateId"] == candidate)
                & (actual["matrixIndex"] == matrix_index)
            ]
            expected_keys = set()
            for row in applicable.to_dict("records"):
                expected_keys.add(
                    (
                        row["candidateId"],
                        int(row["matrixIndex"]),
                        row["trajectoryId"],
                        row["implementationId"],
                        int(row["endpointGeneration"]),
                        row["validationKind"],
                        row["nominalSentinel"],
                    )
                )
            observed_keys = set(
                map(tuple, task_actual[key_columns].itertuples(index=False, name=None))
            )
            task_rows.append(
                {
                    "taskId": f"{candidate}/M{matrix_index:02d}",
                    "candidateId": candidate,
                    "matrixIndex": matrix_index,
                    "eligibleIigrEndpointCount": int(
                        (
                            (task["implementationId"] == "IIGR_CORRECTED_SOURCE")
                            & (task["priorLockedClockTransitions"] >= 256)
                        ).sum()
                    ),
                    "eligiblePhirlEndpointCount": int(
                        (
                            (task["implementationId"] == "PHIRL_REGULARIZED_SOURCE")
                            & (task["priorLockedClockTransitions"] >= 256)
                        ).sum()
                    ),
                    "nominalSlotCount": 18,
                    "applicableSlotCount": len(applicable),
                    "notApplicableSlotCount": len(unavailable),
                    "executedSlotCount": len(task_actual),
                    "missingApplicableCount": len(expected_keys - observed_keys),
                    "unexpectedExecutedCount": len(observed_keys - expected_keys),
                    "passed": expected_keys == observed_keys
                    and len(task_actual) == len(observed_keys),
                }
            )
    applicable_all = pd.concat(applicable_frames, ignore_index=True)
    unavailable_all = pd.concat(unavailable_frames, ignore_index=True)
    expected_all = applicable_all.rename(columns={"nominalSentinel": "sentinel"})
    expected_set = set(
        map(tuple, expected_all[key_columns].itertuples(index=False, name=None))
    )
    actual["expectedIdentityMatched"] = [
        tuple(row) in expected_set
        for row in actual[key_columns].itertuples(index=False, name=None)
    ]
    task_ledger = pd.DataFrame(task_rows)
    actual_unavailable = {
        row.taskId: int(row.notApplicableSlotCount)
        for row in task_ledger.itertuples()
        if row.notApplicableSlotCount
    }
    all_results_pass = bool(
        actual["structuralExact"].astype(bool).all()
        and actual["resultExact"].fillna(False).astype(bool).all()
        and (actual["status"] == "PASS").all()
    )
    passed = bool(
        len(actual) == 3552
        and len(applicable_all) == 3552
        and len(unavailable_all) == 48
        and len(actual) + len(unavailable_all) == 3600
        and duplicate_actual == 0
        and actual_set == expected_set
        and actual_unavailable == EXACT_UNAVAILABLE_TASKS
        and task_ledger["passed"].all()
        and all_results_pass
    )
    write_csv(STEP_ROOT / "availability_aware_replay_task_ledger.csv", task_ledger)
    write_csv(STEP_ROOT / "unavailable_sentinel_slots.csv", unavailable_all)
    write_parquet(STEP_ROOT / "executed_sentinel_validation.parquet", actual)
    payload = {
        "schema": "eidosoma.e01.s13rrr_availability_rule_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "taskCount": len(task_ledger),
        "nominalSlotCount": 3600,
        "applicableSlotCount": len(applicable_all),
        "notApplicableSlotCount": len(unavailable_all),
        "executedSlotCount": len(actual),
        "duplicateExecutedIdentityCount": duplicate_actual,
        "missingApplicableIdentityCount": len(expected_set - actual_set),
        "unexpectedExecutedIdentityCount": len(actual_set - expected_set),
        "exactUnavailableTasks": actual_unavailable,
        "allExecutedSentinelsPassed": all_results_pass,
        "allTaskLedgersPassed": bool(task_ledger["passed"].all()),
        "passed": passed,
    }
    write_json(STEP_ROOT / "availability_rule_validation.json", payload)
    return payload, task_ledger, unavailable_all


def unchanged_source_gate(
    frames: dict[str, pd.DataFrame], availability: dict[str, Any]
) -> dict[str, Any]:
    full = frames["full_source_values.parquet"]
    prefix = frames["prefix_endpoint_values.parquet"]
    suffix = frames["replay_suffix_validation.parquet"]
    diagnostics = frames["source_diagnostic_outputs.parquet"]
    task_validation = pd.read_parquet(S13_ROOT / "source_task_validation.parquet")
    eligible = prefix[prefix["priorLockedClockTransitions"] >= 256]
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
        "availabilityAwareSuffixGate": availability["passed"],
        "minimumFullCoverage": float(full_coverage.min()) >= 0.80,
        "minimumPrefixCoverage": float(prefix_coverage.min()) >= 0.80,
        "componentIdentityError": float(
            diagnostics["componentIdentityMaxAbsError"].fillna(0).max()
        )
        <= 1e-12,
    }
    payload = {
        "schema": "eidosoma.e01.s13rrr_unchanged_source_gate_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "eligiblePrefixRowCount": len(eligible),
        "structuralSuffixRowCount": len(suffix),
        "executedSuffixRowCount": availability["executedSlotCount"],
        "notApplicableSuffixSlotCount": availability["notApplicableSlotCount"],
        "minimumFullCoverageObserved": float(full_coverage.min()),
        "minimumPrefixCoverageObserved": float(prefix_coverage.min()),
        "maximumComponentIdentityAbsoluteErrorObserved": float(
            diagnostics["componentIdentityMaxAbsError"].fillna(0).max()
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_json(STEP_ROOT / "unchanged_source_gate_validation.json", payload)
    return payload


def prefix_interface(prefix: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if (
        "rawObservationIndex" in prefix.columns
        or "endpointRawObservationIndex" not in prefix.columns
    ):
        raise RuntimeError("frozen prefix-interface precondition failed")
    original_hash = frame_hash(prefix)
    adapted = prefix.copy(deep=True)
    adapted["rawObservationIndex"] = adapted["endpointRawObservationIndex"]
    index = adapted[
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
    groups = 0
    monotone = 0
    for _, group in index.groupby(
        ["candidateId", "trajectoryId", "implementationId"], sort=True
    ):
        groups += 1
        values = group.sort_values("generation")["rawObservationIndex"].to_numpy(
            dtype=np.int64
        )
        monotone += int(
            len(values) == len(np.unique(values)) and np.all(np.diff(values) > 0)
        )
    identity = bool(
        adapted["rawObservationIndex"].notna().all()
        and np.array_equal(
            adapted["rawObservationIndex"].to_numpy(dtype=np.int64),
            adapted["endpointRawObservationIndex"].to_numpy(dtype=np.int64),
        )
    )
    unchanged = frame_hash(adapted[list(prefix.columns)]) == original_hash
    write_parquet(STEP_ROOT / "prefix_statistical_view_index.parquet", index)
    payload = {
        "schema": "eidosoma.e01.s13rrr_prefix_interface_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "formula": "rawObservationIndex := endpointRawObservationIndex",
        "rowCount": len(prefix),
        "integerIdentity": identity,
        "groupCount": groups,
        "strictMonotoneGroupCount": monotone,
        "originalFieldsUnchanged": unchanged,
        "passed": identity and unchanged and groups == monotone,
    }
    write_json(STEP_ROOT / "prefix_interface_validation.json", payload)
    return adapted, payload


def reporting_precheck(audit: list[dict[str, Any]]) -> dict[str, Any]:
    prefix_pass = bool(
        len(audit) == len(PREFIX_REPORTING_ORDER)
        and all(row["valuesAndNullMaskUnchanged"] for row in audit)
        and set(ENSEMBLE_SOURCE_ORDER) == set(ENSEMBLE_REPORTING_ORDER)
        and len(ENSEMBLE_SOURCE_ORDER) == len(ENSEMBLE_REPORTING_ORDER)
    )
    payload = {
        "schema": "eidosoma.e01.s13rrr_reporting_order_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "prefixFieldCount": len(audit),
        "prefixTargetOrderExact": [
            row["field"] for row in sorted(audit, key=lambda row: row["afterPosition"])
        ]
        == list(PREFIX_REPORTING_ORDER),
        "prefixValuesAndNullMasksUnchanged": all(
            row["valuesAndNullMaskUnchanged"] for row in audit
        ),
        "ensembleSourceAndTargetFieldSetsEqual": set(ENSEMBLE_SOURCE_ORDER)
        == set(ENSEMBLE_REPORTING_ORDER),
        "ensemblePostStatisticsAuditStatus": "PENDING",
        "passed": prefix_pass,
    }
    write_parquet(STEP_ROOT / "reporting_order_audit.parquet", pd.DataFrame(audit))
    write_json(STEP_ROOT / "reporting_order_validation.json", payload)
    return payload


def run_statistics(
    frames: dict[str, pd.DataFrame], audit: list[dict[str, Any]]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    prefix = frames["prefix_endpoint_values.parquet"]
    adapted, interface = prefix_interface(prefix)
    if not interface["passed"]:
        raise RuntimeError("S13RRR prefix statistical view failed")
    args = (
        frames["full_source_values.parquet"],
        adapted,
        prefix,
        frames["label_values.parquet"],
        frames["partition_history.parquet"],
    )
    first, first_classification = s13.compute_statistics(*args)
    second, second_classification = s13.compute_statistics(*args)
    first_ensemble_raw = first["ensemble_adjudication"].copy(deep=True)
    second_ensemble_raw = second["ensemble_adjudication"].copy(deep=True)
    first["ensemble_adjudication"] = reorder_columns_exact(
        first_ensemble_raw, ENSEMBLE_REPORTING_ORDER
    )
    second["ensemble_adjudication"] = reorder_columns_exact(
        second_ensemble_raw, ENSEMBLE_REPORTING_ORDER
    )
    for name in ENSEMBLE_REPORTING_ORDER:
        audit.append(
            {
                "table": "ensemble_adjudication.csv",
                "field": name,
                "beforePosition": list(first_ensemble_raw.columns).index(name),
                "afterPosition": list(first["ensemble_adjudication"].columns).index(
                    name
                ),
                "beforeValueSha256": frame_hash(first_ensemble_raw[[name]]),
                "afterValueSha256": frame_hash(first["ensemble_adjudication"][[name]]),
                "beforeNullMaskSha256": hashlib.sha256(
                    first_ensemble_raw[name].isna().to_numpy().tobytes()
                ).hexdigest(),
                "afterNullMaskSha256": hashlib.sha256(
                    first["ensemble_adjudication"][name].isna().to_numpy().tobytes()
                ).hexdigest(),
                "valuesAndNullMaskUnchanged": first_ensemble_raw[name].equals(
                    first["ensemble_adjudication"][name]
                ),
            }
        )
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
        jsonable(first_classification), sort_keys=True
    ) == json.dumps(jsonable(second_classification), sort_keys=True)
    replay = {
        "schema": "eidosoma.e01.s13rrr_statistics_replay_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "statisticsExecutions": 2,
        "results": rows,
        "classificationExact": classification_exact,
        "passed": all(row["exact"] for row in rows) and classification_exact,
    }
    write_json(STEP_ROOT / "statistics_replay_validation.json", replay)
    if not replay["passed"]:
        raise RuntimeError("exact twice-run statistics replay failed")
    for key, filename in s13.RESULT_FILES.items():
        path = STEP_ROOT / filename
        (write_parquet if path.suffix == ".parquet" else write_csv)(path, first[key])
    audit_frame = pd.DataFrame(audit)
    write_parquet(STEP_ROOT / "reporting_order_audit.parquet", audit_frame)
    reporting = json.loads((STEP_ROOT / "reporting_order_validation.json").read_text())
    reporting.update(
        {
            "ensemblePostStatisticsAuditStatus": "PASS",
            "ensembleFieldCount": len(ENSEMBLE_REPORTING_ORDER),
            "ensembleTargetOrderExact": list(first["ensemble_adjudication"].columns)
            == list(ENSEMBLE_REPORTING_ORDER),
            "ensembleValuesAndNullMasksUnchanged": bool(
                audit_frame["valuesAndNullMaskUnchanged"].all()
            ),
            "passed": bool(
                reporting["passed"] and audit_frame["valuesAndNullMaskUnchanged"].all()
            ),
        }
    )
    write_json(STEP_ROOT / "reporting_order_validation.json", reporting)
    classification = {
        **first_classification,
        "schema": "eidosoma.e01.s13rrr_classification.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "frozenScientificClassification": first_classification["classification"],
        "evidenceLabel": EVIDENCE_LABEL,
        "postFailureOverrideOrdinal": 3,
        "historicalClassificationsRetained": {
            "S13": "S13_VALIDATION_FAILED_CLOSED",
            "S13R": "S13R_REPAIR_PATH_PERMANENTLY_STOPPED",
            "S13RR": "S13RR_REPAIR_PATH_PERMANENTLY_STOPPED",
        },
        "candidate1Excluded": True,
        "candidateSpecificStatisticsComputed": True,
        "twoCandidateAdjudicationPerformed": True,
        "laterWorkStatus": "BLOCKED_PENDING_S13RRR_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "classification.json", classification)
    return first, classification, replay


def schema_validation() -> dict[str, Any]:
    source_columns = json.loads(S12G_SCHEMAS.read_text())["tables"]
    expected = {**source_columns, **s13r.RESULT_COLUMNS}
    expected["prefix_endpoint_values.parquet"] = list(PREFIX_REPORTING_ORDER)
    expected["ensemble_adjudication.csv"] = list(ENSEMBLE_REPORTING_ORDER)
    filenames = [
        *SOURCE_TO_TARGET.values(),
        *s13.RESULT_FILES.values(),
        "failure_ledger.csv",
    ]
    rows = []
    for filename in filenames:
        path = STEP_ROOT / filename
        frame = (
            pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        )
        columns_exact = list(frame.columns) == list(expected[filename])
        rows.append(
            {
                "artifact": filename,
                "rowCount": len(frame),
                "columnsExact": columns_exact,
            }
        )
    payload = {
        "schema": "eidosoma.e01.s13rrr_schema_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "artifacts": rows,
        "passed": all(row["columnsExact"] for row in rows),
    }
    write_json(STEP_ROOT / "schema_validation.json", payload)
    return payload


def result_validation(
    results: dict[str, pd.DataFrame], classification: dict[str, Any]
) -> dict[str, Any]:
    candidates = sorted(
        results["ensemble_adjudication"]["candidateId"].unique().tolist()
    )
    checks = {
        "allTenResultFamiliesPresent": set(results) == set(s13.RESULT_FILES),
        "allResultFamiliesNonempty": all(len(frame) > 0 for frame in results.values()),
        "exactCandidateSet": candidates == sorted(CANDIDATES),
        "candidate1Excluded": not any(
            "CANDIDATE-01" in str(value)
            for frame in results.values()
            for value in frame.get("candidateId", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
        ),
        "ensembleHasTwoRows": len(results["ensemble_adjudication"]) == 2,
        "classificationMatchesAdjudication": classification["candidateResults"]
        == results["ensemble_adjudication"]
        .loc[:, ENSEMBLE_SOURCE_ORDER]
        .to_dict("records"),
    }
    payload = {
        "schema": "eidosoma.e01.s13rrr_result_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "resultRows": {key: len(value) for key, value in results.items()},
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_json(STEP_ROOT / "result_validation.json", payload)
    return payload


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
        "schema": "eidosoma.e01.s13rrr_artifact_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "artifactCountExcludingSelf": len(rows),
        "artifacts": rows,
        "requiredMissing": missing,
        "passed": not missing,
    }
    write_json(STEP_ROOT / "artifact_manifest.json", payload)
    return payload


def primary_summary(results: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    associations = results["candidate_associations"]
    drift = results["replicator_drift_results"]
    ensemble = results["ensemble_adjudication"]
    rows = []
    for candidate in CANDIDATES:
        full = associations[
            (associations["candidateId"] == candidate)
            & (associations["implementationId"] == "IIGR_CORRECTED_SOURCE")
            & (associations["estimand"] == "RETROSPECTIVE_CURRENT_GENERATION")
        ].iloc[0]
        prefix = associations[
            (associations["candidateId"] == candidate)
            & (associations["implementationId"] == "IIGR_CORRECTED_SOURCE")
            & (associations["estimand"] == "CURRENT_HISTORICAL")
            & (associations["temporalModeId"].str.endswith("_PREFIX_ENDPOINT"))
        ].iloc[0]
        full_drift = drift[
            (drift["candidateId"] == candidate)
            & (drift["implementationId"] == "IIGR_CORRECTED_SOURCE")
            & (drift["temporalModeId"].str.endswith("_FULL"))
        ].iloc[0]
        adjudication = ensemble[ensemble["candidateId"] == candidate].iloc[0]
        rows.append(
            {
                "candidateId": candidate,
                "retrospectiveDefined": int(full["definedTrajectoryCount"]),
                "retrospectivePositive": int(full["positiveTrajectoryCount"]),
                "retrospectiveMedianRho": float(full["medianCorrelation"]),
                "retrospectiveBootstrap95": [
                    float(full["bootstrapLower95"]),
                    float(full["bootstrapUpper95"]),
                ],
                "retrospectiveAssociationGate": bool(full["gatePassed"]),
                "retrospectiveDriftPositive": int(
                    full_drift["positiveMeanDifferenceCount"]
                ),
                "retrospectiveDriftMedianDifference": float(
                    full_drift["medianMeanDifference"]
                ),
                "retrospectiveDriftGate": bool(full_drift["gatePassed"]),
                "prospectiveDefined": int(prefix["definedTrajectoryCount"]),
                "prospectivePositive": int(prefix["positiveTrajectoryCount"]),
                "prospectiveMedianRho": float(prefix["medianCorrelation"]),
                "prospectiveBootstrap95": [
                    float(prefix["bootstrapLower95"]),
                    float(prefix["bootstrapUpper95"]),
                ],
                "prospectiveCircularShiftP": float(prefix["circularShiftPositiveP"]),
                "prospectiveGate": bool(prefix["gatePassed"]),
                "combinedGate": bool(
                    adjudication["combinedRetrospectiveAndProspectiveGate"]
                ),
                "candidateClassification": adjudication["candidateClassification"],
            }
        )
    return rows


def build_report(
    *,
    classification: dict[str, Any],
    validation: dict[str, Any],
    availability: dict[str, Any],
    source_gate: dict[str, Any],
    results: dict[str, pd.DataFrame],
    runtime: dict[str, Any],
    artifact_count: int,
) -> str:
    token = classification["frozenScientificClassification"]
    summaries = primary_summary(results)
    lines = [
        "# S13RRR Full Results: Eligibility-Aware Replay Finalization",
        "",
        "## Top summary",
        "",
        f"- **Research step ID:** `{VERSION}` (S13RRR).",
        "- **Completion status:** `COMPLETED_AT_MANDATORY_S13RRR_HUMAN_REVIEW_BOUNDARY`.",
        f"- **Artifacts written:** {artifact_count} status-bearing files under `/artifacts/research_steps/S13RRR/`, including the 200-task availability ledger, 48-slot non-applicability ledger, 3,552 executed comparisons, complete twice-run statistics, adjudication, validation, provenance, status, and this report.",
        f"- **Validation result:** `PASS`; {availability['executedSlotCount']}/{availability['applicableSlotCount']} executable suffix sentinels passed, {availability['notApplicableSlotCount']} frozen slots were exactly unavailable, and the unchanged source/replay, reporting, schema, twice-run statistics, provenance, and immutability gates passed.",
        f"- **Outcome classification:** `{token}` ({outcome_class(token)}).",
        "- **Caveats or blockers:** This is an explicitly post-outcome third override of two earlier permanent-stop decisions. It changes the replay cardinality gate after endpoint availability was known, so the result is substantially less confirmatory than a clean preregistered analysis. Full-trajectory fits remain retrospective/future-dependent and this cannot identify the unpublished author implementation.",
        "- **Lay summary:** Every replay comparison that could exist in the frozen data was exact. Running the original held-out analysis twice then gave the same answer both times; the two confirmed simulator candidates did not jointly pass the required retrospective and prospective evidence gates.",
        "- **Recommended next action:** Mandatory human review. Keep all later work blocked; do not authorize another repair or automatic continuation.",
        "",
        "## Frozen question",
        "",
        "Can the exact 48 unavailable suffix slots be treated as not applicable—without changing any other gate—and thereby allow a deterministic answer to the frozen two-candidate held-out S13 question?",
        "",
        "## Inputs and provenance",
        "",
        "Only the 200 frozen S13 task bundles and the 1,600 value-preserving S13RR canonical views were read. Candidate 1 was excluded. No simulation, source fit, task omission, subset, resampling-seed change, candidate change, or downstream step occurred. S13, S13R, and S13RR remain byte-for-byte immutable with their failure classifications unchanged.",
        "",
        "## Detailed methods",
        "",
        "Before statistics, the pushed method reconstructed nominal first/middle/last suffix slots from each implementation's frozen eligible endpoint generations. Duplicate nominal generations were treated according to the already frozen source precedence; zero eligible endpoints generated no executable comparison. The exact ledger required 3,552 executable and 48 not-applicable slots. Candidate 2 matrix 68 contributed six executable comparisons and twelve duplicate nominal slots; candidate 2 matrix 72 and candidate 3 matrix 72 each contributed eighteen unavailable slots. Every other source gate remained unchanged.",
        "",
        "The only table-interface operations reordered existing fields in `prefix_endpoint_values.parquet` and `ensemble_adjudication.csv`. Field sets, rows, values, null masks, and keys were unchanged. The source statistics were then executed twice with the original 4,096-replicate bootstrap, circular-shift, and block-aware seeds and the original two-candidate unanimity rule.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m pytest -q tests/e01/test_s13rrr_eligibility_aware_replay.py tests/e01/test_s13rr_downstream_schema_canonicalization.py tests/e01/test_s13_confirmed_timebase_scaleup.py",
        "python -m ruff check src/e01_s13rrr_eligibility_aware_replay scripts/e01/freeze_s13rrr_preregistration.py scripts/e01/run_s13rrr_eligibility_aware_replay_finalization.py tests/e01/test_s13rrr_eligibility_aware_replay.py",
        "ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s13rrr_preregistration.py --record-commit",
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13rrr_eligibility_aware_replay_finalization.py",
        "```",
        "",
        "## Results",
        "",
        f"The scientific adjudication was `{token}`. Candidate-specific results remained primary:",
        "",
    ]
    for row in summaries:
        lines.append(
            f"- `{row['candidateId']}`: retrospective IIGR median rho {row['retrospectiveMedianRho']:.6g} ({row['retrospectivePositive']}/{row['retrospectiveDefined']} positive; 95% trajectory-bootstrap [{row['retrospectiveBootstrap95'][0]:.6g}, {row['retrospectiveBootstrap95'][1]:.6g}]; association gate `{row['retrospectiveAssociationGate']}`); replicator-minus-drift median mean difference {row['retrospectiveDriftMedianDifference']:.6g} ({row['retrospectiveDriftPositive']} positive; gate `{row['retrospectiveDriftGate']}`); prospective median rho {row['prospectiveMedianRho']:.6g} ({row['prospectivePositive']}/{row['prospectiveDefined']} positive; 95% interval [{row['prospectiveBootstrap95'][0]:.6g}, {row['prospectiveBootstrap95'][1]:.6g}]; circular-shift p={row['prospectiveCircularShiftP']:.6g}; gate `{row['prospectiveGate']}`); combined gate `{row['combinedGate']}`; classification `{row['candidateClassification']}`."
        )
    lines.extend(
        [
            "",
            "This result is labeled exactly as a third-override, post-outcome eligibility-exception analysis. It does not retroactively validate S13, S13R, or S13RR and cannot support fixed-window, early-warning, prediction, intervention, causal-control, or author-identity claims.",
            "",
            "## Validation",
            "",
            "- Availability: 200/200 task ledgers passed; 3,552/3,552 applicable identities existed exactly once and passed structural/result replay; 48/48 unavailable identities were confined to the three declared tasks.",
            f"- Source gates: full and eligible-prefix replay, structural suffix, >=0.80 finite coverage, zero worker failures, and <=1e-12 component identity error all passed (`{source_gate['passed']}`).",
            f"- Complete execution validation: `{validation['allValidationGatesPassed']}`. Both statistics executions were bit-exact at the serialized DataFrame level.",
            "- Source bundles, canonical views, and all earlier artifact files passed pre- and post-run SHA-256 validation.",
            "",
            "## Runtime and storage",
            "",
            f"Wall time was {runtime['wallSeconds']:.3f} seconds and process CPU time was {runtime['processCpuSeconds']:.3f} seconds. No simulation, source-fit worker, or GPU was used. S13RRR retained {runtime['artifactBytesBeforeManifest']} artifact bytes and {runtime['derivedCacheBytes']} disposable derived-cache bytes.",
            "",
            "## Caveats, blockers, and limitations",
            "",
            "- The 3,552 rule was authorized after the three low-availability tasks were known, and two reporting-order corrections were also post-outcome. This is not clean confirmation.",
            "- Repeated waivers and repairs materially reduce procedural credibility even though this operation was value-preserving and exactly replayed.",
            "- The public-source information pipeline remains source-informed only; the paper's unpublished GARD and fixed-window implementation are unavailable.",
            "- Retrospective full-trajectory local values use future observations. Prospective prefix results start only after 256 locked-clock transitions.",
            "- No additional repair or downstream continuation is authorized.",
            "",
            "## Artifact provenance",
            "",
            "The pushed method commit, complete immutable-prior baseline, 2,000 source-cache hashes, 1,600 canonical-view hashes, exact task/slot ledgers, input and reporting audits, twice-run result hashes, schema checks, status, runtime/storage records, and final artifact manifest provide the audit chain. Bulky immutable inputs remain in their existing cache roots.",
            "",
            "## Recommended next action",
            "",
            "Return for mandatory human review and begin no later work automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(
    *,
    config: dict[str, Any],
    lock: dict[str, Any],
    prior: dict[str, Any],
    cache: dict[str, Any],
    views: dict[str, Any],
    collation: dict[str, Any],
    availability: dict[str, Any],
    source_gate: dict[str, Any],
    reporting: dict[str, Any],
    results: dict[str, pd.DataFrame],
    classification: dict[str, Any],
    replay: dict[str, Any],
    started_wall: float,
    started_cpu: float,
) -> None:
    failure_columns = json.loads(S12G_SCHEMAS.read_text())["tables"][
        "failure_ledger.csv"
    ]
    failure_rows = backend.failure_rows_from_statuses(
        pd.read_parquet(STEP_ROOT / "full_source_values.parquet"),
        pd.read_parquet(STEP_ROOT / "prefix_endpoint_values.parquet"),
        pd.DataFrame(),
    )
    for row in failure_rows:
        row["failureId"] = str(row["failureId"]).replace("S12G-", "S13RRR-")
    failures = pd.DataFrame(failure_rows, columns=failure_columns)
    write_csv(STEP_ROOT / "failure_ledger.csv", failures)
    schema = schema_validation()
    result = result_validation(results, classification)
    prior_post = validate_prior()
    cache_post = validate_file_manifest(
        "source_cache_input_manifest.json",
        schema="eidosoma.e01.s13rrr_source_cache_input_validation.v1",
    )
    views_post = validate_file_manifest(
        "canonical_view_input_manifest.json",
        schema="eidosoma.e01.s13rrr_canonical_view_input_validation.v1",
    )
    write_json(STEP_ROOT / "source_cache_input_validation.json", cache_post)
    write_json(STEP_ROOT / "canonical_view_input_validation.json", views_post)
    derived_bytes = (
        sum(path.stat().st_size for path in DERIVED_ROOT.rglob("*") if path.is_file())
        if DERIVED_ROOT.exists()
        else 0
    )
    artifact_bytes = sum(
        path.stat().st_size for path in STEP_ROOT.rglob("*") if path.is_file()
    )
    runtime = {
        "schema": "eidosoma.e01.s13rrr_runtime_manifest.v1",
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
            "schema": "eidosoma.e01.s13rrr_storage_validation.v1",
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
            "schema": "eidosoma.e01.s13rrr_implementation_lock.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "designCommit": lock["designCommit"],
            "replayRuleId": config["replayRuleOverride"]["identifier"],
            "reportingOrderId": config["reportingOrderContract"]["identifier"],
            "applicableSuffixRequirement": 3552,
            "notApplicableSuffixSlots": 48,
            "statisticsImplementation": "scripts/e01/run_s13_confirmed_timebase_scaleup.py",
            "statisticsExecutions": 2,
            "additionalModificationRequired": False,
            "passed": True,
        },
    )
    write_json(
        STEP_ROOT / "provenance_manifest.json",
        {
            "schema": "eidosoma.e01.s13rrr_provenance_manifest.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "designCommit": lock["designCommit"],
            "branch": git("branch", "--show-current"),
            "historicalClassificationsRetained": classification[
                "historicalClassificationsRetained"
            ],
            "sourceCacheRoot": str(SOURCE_CACHE_ROOT),
            "canonicalViewRoot": str(VIEW_ROOT),
            "sourceFileCount": cache["checkedFileCount"],
            "canonicalViewCount": views["checkedFileCount"],
            "trajectoryGenerated": False,
            "sourceFitRerun": False,
            "taskOmitted": False,
            "subsetAnalysis": False,
            "candidate1Excluded": True,
            "statisticsSeedsChanged": False,
            "passed": prior_post["passed"]
            and cache_post["passed"]
            and views_post["passed"],
        },
    )
    scope = json.loads((STEP_ROOT / "scope_access_ledger.json").read_text())
    scope["events"].append(
        {
            "stage": "S13RRR_COMPLETE_AT_HUMAN_REVIEW_BOUNDARY",
            "candidateStatisticOpened": True,
            "sourceFitRerun": False,
            "newTrajectoryGenerated": False,
            "taskOmitted": False,
            "subsetAnalysisUsed": False,
            "candidate1Accessed": False,
            "laterWorkAccessed": False,
            "status": "PASS",
        }
    )
    scope["success"] = True
    write_json(STEP_ROOT / "scope_access_ledger.json", scope)
    all_pass = bool(
        prior_post["passed"]
        and cache_post["passed"]
        and views_post["passed"]
        and collation["passed"]
        and availability["passed"]
        and source_gate["passed"]
        and reporting["passed"]
        and replay["passed"]
        and schema["passed"]
        and result["passed"]
    )
    execution = {
        "schema": "eidosoma.e01.s13rrr_execution_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "priorImmutabilityPassed": prior_post["passed"],
        "sourceCachePassed": cache_post["passed"],
        "canonicalViewsPassed": views_post["passed"],
        "strictCollationPassed": collation["passed"],
        "availabilityAwareReplayPassed": availability["passed"],
        "unchangedSourceGatesPassed": source_gate["passed"],
        "reportingOrderPassed": reporting["passed"],
        "candidateStatisticsComputed": True,
        "statisticsReplayPassed": replay["passed"],
        "schemaPassed": schema["passed"],
        "resultValidationPassed": result["passed"],
        "validationResult": "PASS_ALL_FROZEN_S13RRR_AND_UNCHANGED_S13_GATES"
        if all_pass
        else "FAIL_PERMANENTLY_NO_FURTHER_REPAIR",
        "allValidationGatesPassed": all_pass,
    }
    write_json(STEP_ROOT / "execution_validation.json", execution)
    if not all_pass:
        raise RuntimeError(
            "post-statistics validation failed; S13RRR permanently stopped"
        )
    status = {
        "researchStepId": VERSION,
        "stepNumber": "S13RRR",
        "success": True,
        "status": "COMPLETED_AT_MANDATORY_S13RRR_HUMAN_REVIEW_BOUNDARY",
        "artifactsWritten": config["artifacts"]["required"],
        "validationResult": execution["validationResult"],
        "caveatsOrBlockers": [
            "This is a third post-failure human override and materially weakens confirmatory credibility.",
            "The 3,552 availability gate was authorized after low-eligibility tasks were known.",
            "Retrospective full-trajectory fits remain future-dependent and source-informed only.",
            "All later work remains blocked.",
        ],
        "recommendedNextAction": "Mandatory human review; begin no later step and authorize no further repair automatically.",
        "outcomeClassification": classification["frozenScientificClassification"],
        "outcomeClass": outcome_class(classification["frozenScientificClassification"]),
        "evidenceLabel": EVIDENCE_LABEL,
        "laterWorkStatus": "BLOCKED_PENDING_S13RRR_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "status.json", status)
    report = build_report(
        classification=classification,
        validation=execution,
        availability=availability,
        source_gate=source_gate,
        results=results,
        runtime=runtime,
        artifact_count=0,
    )
    (STEP_ROOT / "research_step_full_results.md").write_text(report)
    manifest = artifact_manifest(config)
    if not manifest["passed"]:
        raise RuntimeError(
            f"S13RRR artifact completeness failed: {manifest['requiredMissing']}"
        )
    report = build_report(
        classification=classification,
        validation=execution,
        availability=availability,
        source_gate=source_gate,
        results=results,
        runtime=runtime,
        artifact_count=manifest["artifactCountExcludingSelf"] + 1,
    )
    (STEP_ROOT / "research_step_full_results.md").write_text(report)
    manifest = artifact_manifest(config)
    if not manifest["passed"]:
        raise RuntimeError("S13RRR final artifact manifest failed")


def main() -> int:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    config = yaml.safe_load(CONFIG.read_text())
    lock = verify_method_lock()
    prior = validate_prior()
    cache = validate_file_manifest(
        "source_cache_input_manifest.json",
        schema="eidosoma.e01.s13rrr_source_cache_input_validation.v1",
    )
    views = validate_file_manifest(
        "canonical_view_input_manifest.json",
        schema="eidosoma.e01.s13rrr_canonical_view_input_validation.v1",
    )
    write_json(STEP_ROOT / "source_cache_input_validation.json", cache)
    write_json(STEP_ROOT / "canonical_view_input_validation.json", views)
    if not prior["passed"] or not cache["passed"] or not views["passed"]:
        raise RuntimeError("S13RRR immutable prior/input gate failed")
    frames, collation, audit = strict_collate()
    if not collation["passed"]:
        raise RuntimeError("S13RRR strict collation failed; no repair permitted")
    availability, _, _ = validate_availability(frames)
    if not availability["passed"]:
        raise RuntimeError(
            "S13RRR availability-aware replay gate failed; no repair permitted"
        )
    source_gate = unchanged_source_gate(frames, availability)
    if not source_gate["passed"]:
        raise RuntimeError("S13RRR unchanged source gate failed; no repair permitted")
    reporting = reporting_precheck(audit)
    if not reporting["passed"]:
        raise RuntimeError(
            "S13RRR reporting-order precheck failed; no repair permitted"
        )
    results, classification, replay = run_statistics(frames, audit)
    reporting = json.loads((STEP_ROOT / "reporting_order_validation.json").read_text())
    if not reporting["passed"]:
        raise RuntimeError(
            "S13RRR reporting-order postcheck failed; no repair permitted"
        )
    finalize(
        config=config,
        lock=lock,
        prior=prior,
        cache=cache,
        views=views,
        collation=collation,
        availability=availability,
        source_gate=source_gate,
        reporting=reporting,
        results=results,
        classification=classification,
        replay=replay,
        started_wall=started_wall,
        started_cpu=started_cpu,
    )
    print(
        json.dumps(
            {
                "stage": "S13RRR_complete",
                "classification": classification["frozenScientificClassification"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
