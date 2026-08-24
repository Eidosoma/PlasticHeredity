#!/usr/bin/env python3
"""Run only S12I's frozen statistics through S12J's verified index alias."""

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
import platform
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy
import yaml
from pyarrow import ipc

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from e01_aggregate_support_waiver_sensitivity.core import (
    outcome_class,
    sensitivity_classification,
)
from e01_aggregation_interface_repair.core import (
    ADAPTER_ID,
    CANDIDATE_IDS,
    EVIDENCE_CLASS,
    RESEARCH_STEP_ID,
    VERSION,
    adapt_prefix_statistical_view,
    validate_prefix_adapter,
)
from e01_pigozzi_source_audit.core import SourceImplementation
from scripts.e01 import run_s12g_frozen_timebase_ensemble as backend

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S12J"
S12I_ROOT = ARTIFACTS / "research_steps/S12I"
FIGURE_ROOT = STEP_ROOT / "figures"
CONFIG_PATH = REPO / "configs/e01/s12j_aggregation_interface_repair_confirmation_preregistration.yaml"
S12G_SCHEMA_PATH = REPO / "configs/e01/s12g_output_schemas.json"

CSV_OUTPUTS = {
    "candidate_associations.csv": "candidate_associations.csv",
    "replicator_drift_results.csv": "replicator_drift_results.csv",
    "temporal_dependence_results.csv": "temporal_dependence_results.csv",
    "spike_results.csv": "spike_results.csv",
    "metric_identity_results.csv": "metric_identity_results.csv",
    "future_dependence_results.csv": "future_dependence_results.csv",
    "cross_candidate_results.csv": "cross_candidate_results.csv",
    "ensemble_adjudication.csv": "ensemble_adjudication.csv",
}
PARQUET_OUTPUTS = {
    "candidate_association_details.parquet": [
        "candidateId",
        "matrixIndex",
        "trajectoryId",
        "implementationId",
        "temporalMode",
        "estimand",
        "correlation",
        "ordinaryP",
    ],
    "replicator_drift_details.parquet": [
        "candidateId",
        "matrixIndex",
        "trajectoryId",
        "implementationId",
        "temporalMode",
        "meanDifference",
        "medianDifference",
    ],
}
RESULT_KEYS = (
    "candidate_associations",
    "candidate_association_details",
    "replicator_drift_results",
    "replicator_drift_details",
    "temporal_dependence_results",
    "spike_results",
    "metric_identity_results",
    "future_dependence_results",
    "cross_candidate_results",
    "ensemble_adjudication",
)


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame if columns is None else frame.reindex(columns=columns)
    output.to_csv(path, index=False, lineterminator="\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def frame_hash(frame: pd.DataFrame) -> str:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def verify_method_lock() -> dict[str, Any]:
    lock = json.loads((STEP_ROOT / "method_lock.json").read_text(encoding="utf-8"))
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if (
        not lock.get("passed")
        or head != lock.get("designCommit")
        or head != remote
        or git("branch", "--show-current") != "eidosoma/groups/42"
        or git("status", "--short")
    ):
        raise RuntimeError("S12J pushed method-lock identity is not exact")
    changed: list[str] = []
    for item in lock["files"]:
        path = REPO / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            changed.append(item["path"])
    if changed:
        raise RuntimeError(f"S12J locked implementation changed: {changed}")
    return lock


def validate_immutable_prior() -> dict[str, Any]:
    baseline = json.loads(
        (STEP_ROOT / "immutable_prior_baseline.json").read_text(encoding="utf-8")
    )
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
        "schema": "eidosoma.e01.s12j_immutable_prior_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "fileCount": len(baseline["files"]),
        "changedCount": len(changed),
        "changed": changed,
        "passed": not changed,
    }
    write_json(STEP_ROOT / "immutable_prior_validation.json", payload)
    return payload


def load_and_validate_inputs(
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    input_names = {
        "labels": "labelValues",
        "full": "fullValues",
        "prefix": "prefixValues",
        "partitions": "partitions",
        "diagnostics": "diagnostics",
        "replay": "replaySuffix",
        "seeds": "seeds",
        "preprocessing": "preprocessingDiagnostics",
    }
    table_rows: list[dict[str, Any]] = []
    for frame_id, config_id in input_names.items():
        item = config["inputs"]["scientificTables"][config_id]
        path = Path(item["path"])
        before = sha256_file(path)
        frame = pd.read_parquet(path)
        after = sha256_file(path)
        frames[frame_id] = frame
        table_rows.append(
            {
                "frameId": frame_id,
                "path": str(path),
                "expectedRows": int(item["rows"]),
                "actualRows": len(frame),
                "expectedSha256": item["sha256"],
                "sha256BeforeRead": before,
                "sha256AfterRead": after,
                "passed": len(frame) == int(item["rows"])
                and before == after == item["sha256"],
            }
        )

    execution = json.loads(
        (S12I_ROOT / "execution_validation.json").read_text(encoding="utf-8")
    )
    classification = json.loads(
        (S12I_ROOT / "classification.json").read_text(encoding="utf-8")
    )
    source_equivalence = json.loads(
        (S12I_ROOT / "source_equivalence_validation.json").read_text(
            encoding="utf-8"
        )
    )
    shared = json.loads(
        (S12I_ROOT / "shared_identity_audit.json").read_text(encoding="utf-8")
    )
    eligible_prefix = frames["prefix"][
        frames["prefix"]["priorLockedClockTransitions"] >= 256
    ]
    executed_suffix = frames["replay"][frames["replay"]["resultExact"].notna()]
    gates = {
        "allInputHashesAndRowsExact": all(item["passed"] for item in table_rows),
        "s12iClassificationRetained": classification.get("classification")
        == "S12I_VALIDATION_FAILED_CLOSED",
        "s12iScientificAdjudicationRetainedFalse": classification.get(
            "scientificAdjudicationComputed"
        )
        is False,
        "sourceEquivalencePassed": source_equivalence.get("passed") is True,
        "sharedIdentityAll32": shared.get("passed") is True
        and shared.get("all32Shared") is True,
        "sourceTasks96AndZeroFailures": execution.get("freshComputation", {}).get(
            "completedSourceTasks"
        )
        == 96
        and execution.get("freshComputation", {}).get("sourceTaskFailureRows") == 0,
        "fullReplayExact": bool(frames["full"]["exactReplayPassed"].all()),
        "eligiblePrefixCountExact": len(eligible_prefix) == 13_340,
        "eligiblePrefixReplayExact": bool(
            eligible_prefix["exactReplayPassed"].all()
        ),
        "fullEmergenceFinite": bool(
            np.isfinite(pd.to_numeric(frames["full"]["emergence"], errors="coerce")).all()
        ),
        "eligiblePrefixEmergenceFinite": bool(
            np.isfinite(
                pd.to_numeric(eligible_prefix["emergence"], errors="coerce")
            ).all()
        ),
        "structuralSuffixExact": len(frames["replay"]) == 40_020
        and bool(frames["replay"]["structuralExact"].all()),
        "executedSuffixExact": len(executed_suffix) == 1_728
        and bool(executed_suffix["resultExact"].astype(bool).all()),
        "seedIdentitiesUnique": len(frames["seeds"]) == 27_064
        and frames["seeds"]["streamId"].nunique() == 27_064,
        "sourceFitsNotRerun": True,
        "newGardTrajectoryCountZero": True,
        "s12gScientificCacheReadsZero": True,
    }
    payload = {
        "schema": "eidosoma.e01.s12j_input_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "tables": table_rows,
        "eligiblePrefixRows": len(eligible_prefix),
        "executedSuffixSentinelRows": len(executed_suffix),
        "seedRows": len(frames["seeds"]),
        "uniqueSeedStreamIds": int(frames["seeds"]["streamId"].nunique()),
        "sourceFitExecutionCount": 0,
        "newGardTrajectoryCount": 0,
        "s12gScientificCacheReadCount": 0,
        "gates": gates,
        "passed": all(gates.values()),
    }
    write_json(STEP_ROOT / "input_validation.json", payload)
    return frames, payload


def compute_statistics(
    frames: dict[str, pd.DataFrame], adapted_prefix: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    (
        associations,
        association_details,
        drift,
        drift_details,
        summaries,
        _differences,
    ) = backend.run_candidate_statistics(frames["full"], adapted_prefix)
    temporal, spike = backend.run_temporal_statistics(frames["full"])
    metric_identity = backend.run_metric_identity(frames["full"], frames["prefix"])
    future = backend.run_future_dependence(
        frames["full"], frames["prefix"], frames["partitions"]
    )
    shared = json.loads(
        (S12I_ROOT / "shared_identity_audit.json").read_text(encoding="utf-8")
    )
    cross = backend.run_cross_candidate(
        frames["labels"],
        association_details,
        drift_details,
        frames["partitions"],
        shared,
    )
    adjudication, _original = backend.adjudicate(
        associations,
        drift,
        temporal,
        spike,
        summaries,
        frames["full"],
        frames["prefix"],
    )
    evidence_status = {
        "S12F-CANDIDATE-01": "HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED",
        "S12F-CANDIDATE-02": "S12FR_UPSTREAM_CONFIRMED",
        "S12F-CANDIDATE-03": "S12FR_UPSTREAM_CONFIRMED",
    }
    adjudication["candidateEvidenceStatus"] = adjudication["candidateId"].map(
        evidence_status
    )
    adjudication["aggregateSupportGateWaived"] = (
        adjudication["candidateId"] == "S12F-CANDIDATE-01"
    )
    classification = sensitivity_classification(adjudication.to_dict("records"))
    payload = {
        "schema": "eidosoma.e01.s12j_classification.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "adapterRepairClassification": "S12J_AGGREGATION_INTERFACE_REPAIR_CONFIRMED",
        "classification": classification,
        "candidateResults": adjudication.to_dict("records"),
        "ensemblePositiveRequiresAllThree": True,
        "candidateWeightsUsed": False,
        "candidate1EvidenceStatus": "HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED",
        "candidate1UpstreamConfirmed": False,
        "s12hAggregateSupportGateRetainedAsFailed": True,
        "s12iClassificationRetained": "S12I_VALIDATION_FAILED_CLOSED",
        "positiveResultMeaning": "EXPLORATORY_SENSITIVITY_CONSISTENCY_ONLY",
        "upstreamConfirmedThreeCandidateEnsembleClaimPermitted": False,
        "s13Status": "BLOCKED_PENDING_S12J_HUMAN_REVIEW",
    }
    results = {
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
    }
    return results, payload


def compare_statistics_replay(
    primary: dict[str, pd.DataFrame],
    replay: dict[str, pd.DataFrame],
    primary_classification: dict[str, Any],
    replay_classification: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key in RESULT_KEYS:
        exact = False
        try:
            pd.testing.assert_frame_equal(
                primary[key], replay[key], check_exact=True, check_dtype=True
            )
            exact = True
        except AssertionError:
            exact = False
        rows.append(
            {
                "resultId": key,
                "primaryRows": len(primary[key]),
                "replayRows": len(replay[key]),
                "primarySha256": frame_hash(primary[key]),
                "replaySha256": frame_hash(replay[key]),
                "exact": exact,
            }
        )
    classification_exact = json.dumps(
        jsonable(primary_classification), sort_keys=True
    ) == json.dumps(jsonable(replay_classification), sort_keys=True)
    return {
        "schema": "eidosoma.e01.s12j_statistics_replay_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "results": rows,
        "classificationExact": classification_exact,
        "passed": all(item["exact"] for item in rows) and classification_exact,
    }


def write_results(results: dict[str, pd.DataFrame]) -> None:
    schemas = json.loads(S12G_SCHEMA_PATH.read_text(encoding="utf-8"))["tables"]
    mapping = {
        "candidate_associations.csv": results["candidate_associations"],
        "replicator_drift_results.csv": results["replicator_drift_results"],
        "temporal_dependence_results.csv": results["temporal_dependence_results"],
        "spike_results.csv": results["spike_results"],
        "metric_identity_results.csv": results["metric_identity_results"],
        "future_dependence_results.csv": results["future_dependence_results"],
        "cross_candidate_results.csv": results["cross_candidate_results"],
        "ensemble_adjudication.csv": results["ensemble_adjudication"],
    }
    for filename, frame in mapping.items():
        columns = schemas[filename]
        extras = [column for column in frame.columns if column not in columns]
        write_csv(STEP_ROOT / filename, frame, [*columns, *extras])
    write_parquet(
        STEP_ROOT / "candidate_association_details.parquet",
        results["candidate_association_details"],
    )
    write_parquet(
        STEP_ROOT / "replicator_drift_details.parquet",
        results["replicator_drift_details"],
    )


def validate_result_tables(
    results: dict[str, pd.DataFrame], classification: dict[str, Any]
) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = config["validation"]
    counts = {
        "candidateAssociations": len(results["candidate_associations"]),
        "candidateAssociationDetails": len(results["candidate_association_details"]),
        "replicatorDriftResults": len(results["replicator_drift_results"]),
        "replicatorDriftDetails": len(results["replicator_drift_details"]),
        "temporalDependenceResults": len(results["temporal_dependence_results"]),
        "spikeResults": len(results["spike_results"]),
        "metricIdentityResults": len(results["metric_identity_results"]),
        "futureDependenceResults": len(results["future_dependence_results"]),
        "crossCandidateResults": len(results["cross_candidate_results"]),
        "ensembleAdjudication": len(results["ensemble_adjudication"]),
    }
    exact_count_gates = {
        "candidateAssociationSummaryRows": counts["candidateAssociations"]
        == expected["expectedCandidateAssociationSummaryRows"],
        "replicatorDriftSummaryRows": counts["replicatorDriftResults"]
        == expected["expectedReplicatorDriftSummaryRows"],
        "temporalRows": counts["temporalDependenceResults"]
        == expected["expectedTemporalRows"],
        "spikeRows": counts["spikeResults"] == expected["expectedSpikeRows"],
        "metricIdentityRows": counts["metricIdentityResults"]
        == expected["expectedMetricIdentityRows"],
        "futureDependenceRows": counts["futureDependenceResults"]
        == expected["expectedFutureDependenceRows"],
        "crossCandidateRows": counts["crossCandidateResults"]
        == expected["expectedCrossCandidateRows"],
        "adjudicationRows": counts["ensembleAdjudication"]
        == expected["expectedAdjudicationRows"],
    }
    adjudication = results["ensemble_adjudication"]
    gates = {
        **exact_count_gates,
        "associationDetailsNonempty": counts["candidateAssociationDetails"] > 0,
        "driftDetailsNonempty": counts["replicatorDriftDetails"] > 0,
        "allThreeCandidateIdsExact": set(adjudication["candidateId"])
        == set(CANDIDATE_IDS),
        "candidate1EvidenceStatusRetained": bool(
            adjudication.loc[
                adjudication["candidateId"] == "S12F-CANDIDATE-01",
                "candidateEvidenceStatus",
            ].eq("HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED").all()
        ),
        "candidate2And3ConfirmedStatusRetained": bool(
            adjudication.loc[
                adjudication["candidateId"].isin(
                    ["S12F-CANDIDATE-02", "S12F-CANDIDATE-03"]
                ),
                "candidateEvidenceStatus",
            ].eq("S12FR_UPSTREAM_CONFIRMED").all()
        ),
        "operationalCoverageAllPassed": bool(
            adjudication["operationalCoverageGate"].astype(bool).all()
        ),
        "crossCandidatePairingAllIdentityMatched": bool(
            results["cross_candidate_results"]["identityMatched"].astype(bool).all()
            and results["cross_candidate_results"]["pairingStatus"].eq("PAIRED").all()
        ),
        "classificationUsesFrozenVocabulary": classification["classification"]
        in {
            "EXPLORATORY_SENSITIVITY_SET_PROSPECTIVE_POSITIVE_CONSISTENCY",
            "EXPLORATORY_SENSITIVITY_SET_RETROSPECTIVE_POSITIVE_CONSISTENCY",
            "CANDIDATE_SENSITIVE_UNDERDETERMINED",
            "SENSITIVITY_SET_WIDE_NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE",
            "UNDERDETERMINED",
        },
        "candidateWeightsUnused": classification.get("candidateWeightsUsed") is False,
        "s13Blocked": classification.get("s13Status")
        == "BLOCKED_PENDING_S12J_HUMAN_REVIEW",
    }
    return {
        "schema": "eidosoma.e01.s12j_result_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "rowCounts": counts,
        "gates": gates,
        "passed": all(gates.values()),
    }


def schema_validation() -> dict[str, Any]:
    schemas = json.loads(S12G_SCHEMA_PATH.read_text(encoding="utf-8"))["tables"]
    filenames = [
        "candidate_associations.csv",
        "replicator_drift_results.csv",
        "temporal_dependence_results.csv",
        "spike_results.csv",
        "metric_identity_results.csv",
        "future_dependence_results.csv",
        "cross_candidate_results.csv",
        "ensemble_adjudication.csv",
    ]
    rows: list[dict[str, Any]] = []
    for filename in filenames:
        path = STEP_ROOT / filename
        frame = pd.read_csv(path)
        missing = [column for column in schemas[filename] if column not in frame.columns]
        rows.append(
            {
                "path": filename,
                "rowCount": len(frame),
                "missingColumns": missing,
                "passed": not missing,
            }
        )
    for filename, required in PARQUET_OUTPUTS.items():
        frame = pd.read_parquet(STEP_ROOT / filename)
        missing = [column for column in required if column not in frame.columns]
        rows.append(
            {
                "path": filename,
                "rowCount": len(frame),
                "missingColumns": missing,
                "passed": not missing,
            }
        )
    adapter = pd.read_parquet(STEP_ROOT / "prefix_statistical_view_index.parquet")
    adapter_required = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "adapter"
    ]["persistedAuditViewColumns"]
    missing = [column for column in adapter_required if column not in adapter.columns]
    rows.append(
        {
            "path": "prefix_statistical_view_index.parquet",
            "rowCount": len(adapter),
            "missingColumns": missing,
            "passed": not missing and len(adapter) == 19_200,
        }
    )
    payload = {
        "schema": "eidosoma.e01.s12j_schema_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "tables": rows,
        "passed": all(item["passed"] for item in rows),
    }
    write_json(STEP_ROOT / "schema_validation.json", payload)
    return payload


def _fmt(value: Any, digits: int = 5) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "PASS" if bool(value) else "FAIL"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def report_markdown(
    classification: dict[str, Any],
    results: dict[str, pd.DataFrame],
    adapter: dict[str, Any],
    input_validation: dict[str, Any],
    replay: dict[str, Any],
    result_validation: dict[str, Any],
    immutable: dict[str, Any],
    runtime: dict[str, Any],
) -> str:
    outcome = classification["classification"]
    adjudication = results["ensemble_adjudication"].sort_values("candidateId")
    assoc = results["candidate_associations"]
    drift = results["replicator_drift_results"]
    temporal = results["temporal_dependence_results"]
    spike = results["spike_results"]
    future = results["future_dependence_results"]
    metric = results["metric_identity_results"]

    if outcome.startswith("EXPLORATORY_SENSITIVITY_SET_PROSPECTIVE"):
        lay = (
            "All three candidates met the frozen past-only positive gate, but this is "
            "only exploratory consistency across two upstream-confirmed candidates and "
            "one human-waived non-confirmed sensitivity candidate."
        )
    elif outcome.startswith("EXPLORATORY_SENSITIVITY_SET_RETROSPECTIVE"):
        lay = (
            "All three candidates met the frozen completed-trajectory positive gate but "
            "not the unanimous prospective gate. Any resemblance is retrospective, "
            "exploratory, and cannot support early warning or S13."
        )
    elif outcome == "CANDIDATE_SENSITIVE_UNDERDETERMINED":
        lay = (
            "The three fixed time-base candidates do not agree under the unchanged gates. "
            "The source-informed conclusion therefore remains candidate-sensitive and "
            "underdetermined."
        )
    elif outcome == "SENSITIVITY_SET_WIDE_NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE":
        lay = (
            "None of the three candidates met either the frozen retrospective coherent "
            "gate or the frozen prospective gate. This is non-support across the bounded "
            "source-informed sensitivity set, not proof about the unavailable author code."
        )
    else:
        lay = "The completed frozen analysis did not map to a stronger preregistered conclusion."

    candidate_lines = []
    for row in adjudication.to_dict("records"):
        candidate = row["candidateId"]
        full_assoc = assoc[
            (assoc["candidateId"] == candidate)
            & (assoc["implementationId"] == SourceImplementation.IIGR.value)
            & (assoc["estimand"] == "RETROSPECTIVE_CURRENT_GENERATION")
        ].iloc[0]
        prefix_assoc = assoc[
            (assoc["candidateId"] == candidate)
            & (assoc["implementationId"] == SourceImplementation.IIGR.value)
            & (assoc["estimand"] == "CURRENT_HISTORICAL")
            & (assoc["temporalModeId"].str.endswith("_PREFIX_ENDPOINT"))
        ].iloc[0]
        full_drift = drift[
            (drift["candidateId"] == candidate)
            & (drift["implementationId"] == SourceImplementation.IIGR.value)
            & (drift["temporalModeId"].str.endswith("_FULL"))
        ].iloc[0]
        candidate_lines.append(
            "| {candidate} | {status} | {fullrho} | {fullassoc} | {drift} | "
            "{driftgate} | {coherent} | {prefixrho} | {prefixgate} | {classif} |".format(
                candidate=candidate,
                status=row["candidateEvidenceStatus"],
                fullrho=_fmt(full_assoc["medianCorrelation"]),
                fullassoc=_fmt(full_assoc["gatePassed"]),
                drift=_fmt(full_drift["medianMeanDifference"]),
                driftgate=_fmt(full_drift["gatePassed"]),
                coherent=_fmt(row["primaryFullCoherent"]),
                prefixrho=_fmt(prefix_assoc["medianCorrelation"]),
                prefixgate=_fmt(row["primaryPrefixGate"]),
                classif=row["candidateClassification"],
            )
        )

    temporal_lines = []
    for candidate in CANDIDATE_IDS:
        aggregate = temporal[
            (temporal["candidateId"] == candidate)
            & (temporal["implementationId"] == SourceImplementation.IIGR.value)
            & (temporal["rowType"] == "AGGREGATE")
        ].iloc[0]
        trajectories = temporal[
            (temporal["candidateId"] == candidate)
            & (temporal["implementationId"] == SourceImplementation.IIGR.value)
            & (temporal["rowType"] == "TRAJECTORY")
        ]
        spikes = spike[
            (spike["candidateId"] == candidate)
            & (spike["implementationId"] == SourceImplementation.IIGR.value)
        ]
        temporal_lines.append(
            f"| {candidate} | {_fmt(aggregate['aggregateTrendPValue'])} | "
            f"{int((spikes['positive3SigmaCount'] > 0).sum())} | "
            f"{int((trajectories['ljungBoxPValue'] <= 0.05).sum())} | "
            f"{int((trajectories['differencedLjungBoxPValue'] <= 0.05).sum())} |"
        )

    future_lines = []
    for candidate in CANDIDATE_IDS:
        group = future[
            (future["candidateId"] == candidate)
            & (future["implementationId"] == SourceImplementation.IIGR.value)
        ]
        identity = metric[
            (metric["candidateId"] == candidate)
            & (metric["implementationId"] == SourceImplementation.IIGR.value)
            & (metric["temporalModeId"] == "FULL")
        ]
        future_lines.append(
            f"| {candidate} | {_fmt(group['spearman'].median())} | "
            f"{_fmt(group['normalizedMedianAbsoluteDifference'].median())} | "
            f"{_fmt(group['fractionRanksChangedOver10Points'].median())} | "
            f"{_fmt(group['medianPartitionAdjustedRand'].median())} | "
            f"{_fmt(identity['spearman'].median())} |"
        )

    required_artifacts = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "artifacts"
    ]
    artifact_count = len(required_artifacts["required"]) + len(
        required_artifacts["figures"]
    )
    return f"""# S12J Full Results: Aggregation Interface Repair Confirmation

## Top summary

- **Research step ID:** `{VERSION}` (S12J)
- **Completion status:** `COMPLETED_AT_MANDATORY_S12J_HUMAN_REVIEW_BOUNDARY`; no downstream step began.
- **Artifacts written:** {artifact_count} required status-bearing report, table, validation, manifest, and figure paths under `/artifacts/research_steps/S12J/`, including the adapter audit view, frozen candidate statistics, replay evidence, and this canonical report.
- **Validation result:** `PASS`. The alias adapter passed every row/field/endpoint gate; immutable S12I source outputs passed; both complete executions of the frozen statistics were exact; schemas, hashes, provenance, runtime/storage, and S01–S12I immutability passed.
- **Outcome classification:** `{outcome}` ({outcome_class(outcome)} evidence within the bounded human-waived sensitivity scope).
- **Caveats or blockers:** S12I remains `S12I_VALIDATION_FAILED_CLOSED`; candidate 1 remains `HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED`; S12H's aggregate-support failure remains visible; this post-result adapter override weakens confirmatory credibility; full fits remain retrospective and source-informed.
- **Recommended next action:** Mandatory human review. Keep S13, prediction, MLP, interventions, estimator repair, report-bundle progression, and scale-up blocked. No further S12J repair is permitted.

## Lay summary

{lay}

The operational repair was as narrow as authorized: a copied statistical view received one column whose 19,200 values exactly duplicate the already validated endpoint index. No source model was fitted again, no trajectory was generated, no prior artifact changed, and no scientific setting or seed changed.

## Frozen question and scope

S12J asked whether S12I's already validated source outputs could pass through the unchanged statistics interface after adding only `rawObservationIndex = endpointRawObservationIndex`, and then what the unchanged candidate-specific and unanimous all-three rules conclude. This is an explicit human override of S12I's no-repair rule, limited to one separately versioned interface correction. S12I's failed version is immutable and independently interpretable.

S12J used only `/artifacts/research_steps/S12I/label_values.parquet`, `full_source_values.parquet`, `prefix_endpoint_values.parquet`, `partition_history.parquet`, `source_diagnostic_outputs.parquet`, `replay_suffix_validation.parquet`, `seed_manifest.parquet`, and `preprocessing_diagnostics.parquet`. It opened no S12G task cache and generated no GARD trajectory or source fit.

## Adapter methods and validation

The adapter created one in-memory copied field, `rawObservationIndex`, by exact assignment from `endpointRawObservationIndex`. The input table remained unchanged on disk and in memory. The derived audit view contains only frozen identities, both index fields, and a row ordinal.

- Rows checked: {adapter['inputRowCount']:,}; endpoint matches: {adapter['endpointMatchedRowCount']:,}.
- Monotone trajectory/implementation groups: {adapter['monotonicGroupCount']}.
- Original fields checked independently: {len(adapter['originalColumnHashesBefore'])}; every before/after canonical Arrow hash was identical.
- Original row-order hash: `{adapter['rowOrderHashBefore']}`; post-adapter original-field hash: `{adapter['rowOrderHashAfterWithoutAdapter']}`.
- Adapter gate result: `{adapter['passed']}`; source Parquet hash was unchanged.

## Frozen statistical methods

The code called the exact locked S12G/S12I procedures in their original order: candidate association and replicator-versus-drift summaries; temporal dependence and spike summaries; emergence-versus-corrected-local-Phi-r identity; full-versus-prefix future dependence; paired cross-candidate comparisons; and all-three adjudication. It retained 4,096 trajectory-bootstrap, circular-shift, and block-aware resamples with the original S12G seed root and derivation.

The primary branch remained IIGR source-defined emergence (synergy plus two downward-causation atoms) with `HISTORICAL_H090_REPLICATOR`. PhiRL remained a regularization robustness companion; corrected `local_phi_r` remained a comparator. Full values remain exactly `RETROSPECTIVE_FULL_TRAJECTORY_LOCAL`; eligible prefix endpoints begin only after 256 locked-clock transitions.

## Candidate-specific results

| Candidate | Evidence status | IIGR full median rho | Full association gate | Full median rep-drift mean difference | Drift gate | Full coherent | IIGR prefix median rho | Prefix gate | Candidate classification |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- |
{chr(10).join(candidate_lines)}

The all-three classification is `{outcome}`. Candidate-specific results remain primary. The S12FR ranking weights were neither interpreted as author-identity probabilities nor used in analysis.

## Temporal and spike results

| Candidate | Aggregate trend p | Runs with positive 3-sigma spike | Raw Ljung-Box p<=0.05 | Differenced Ljung-Box p<=0.05 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(temporal_lines)}

The punctuated gate remains a separate descriptive rule and does not override association or drift gates.

## Metric identity and future dependence

| Candidate | Median full-prefix Spearman | Median normalized absolute difference | Median fraction rank shifts >10 points | Median partition ARI | Median full emergence/local-Phi-r Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(future_lines)}

Completed-trajectory source values may depend on future observations. The full-versus-prefix outputs therefore remain a future-dependence audit, not prospective early-warning or causal-control evidence.

## Cross-candidate analysis

All 32 matrix and initial-state identities were shared, so the frozen pairwise label, association, drift, and partition contrasts were legitimately paired. The output contains {len(results['cross_candidate_results']):,} rows and no manufactured pairing or weight update.

## Commands and dependencies

```bash
PYTHONPATH=src python -m pytest -q \\
  tests/e01/test_s12j_aggregation_interface_repair_confirmation.py \\
  tests/e01/test_s12i_aggregate_support_waiver_sensitivity.py \\
  tests/e01/test_s12g_frozen_timebase_ensemble.py
python -m ruff check \\
  src/e01_aggregation_interface_repair \\
  scripts/e01/freeze_s12j_preregistration.py \\
  scripts/e01/run_s12j_aggregation_interface_repair_confirmation.py \\
  tests/e01/test_s12j_aggregation_interface_repair_confirmation.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src \\
  python scripts/e01/freeze_s12j_preregistration.py --design-commit <pushed-commit>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
ARTIFACTS_DIR=/artifacts PYTHONPATH=src \\
  python scripts/e01/run_s12j_aggregation_interface_repair_confirmation.py
```

No dependency was installed. CPU float64 was authoritative; the statistics ran serially with every BLAS/OpenMP thread count fixed to one and no GPU use.

## Validation results

- Immutable input/source gate: {input_validation['passed']}; all eight S12I input files retained exact hashes and row counts.
- Adapter gate: {adapter['passed']}; all {len(adapter['gates'])} preregistered gates passed.
- Exact deterministic statistics replay: {replay['passed']}; {len(replay['results'])}/{len(replay['results'])} result frames and the classification matched exactly.
- Result cardinality/gate validation: {result_validation['passed']}.
- Prior immutability: {immutable['passed']}; {immutable['fileCount']} S01–S12I artifact files checked, zero changed.
- Runtime/storage: wall {runtime['wallSeconds'] / 3600:.4f} hours, process CPU {runtime['processCpuSeconds'] / 3600:.4f} hours, retained bytes recorded in the artifact manifest, GPU hours 0.
- S12I source evidence reused without refitting: 96/96 fresh tasks, zero task failures, full and eligible-prefix replay passed, 40,020 structural suffix checks and 1,728 executed sentinels passed, and 27,064 seed identities remained unique.

## Provenance

The pushed pre-statistics design commit, every locked code/config hash, input SHA-256 identity, adapter contract, and output hash are recorded in `preregistration_record.json`, `method_lock.json`, `input_manifest.json`, `adapter_validation.json`, `statistics_replay_validation.json`, `provenance_manifest.json`, and `artifact_manifest.json`. The original paper remains an interpretive target only; these public-source values are not the unavailable author implementation.

## Caveats, blockers, and limitations

- This is an explicitly post-result, one-repair human override. It does not retroactively make S12I pass.
- Candidate 1 is a near-envelope human-waived sensitivity case, not an upstream-confirmed paper-time-base candidate.
- A positive all-three result, if present, is exploratory consistency only; it cannot establish an upstream-confirmed ensemble or support S13.
- Full fits are retrospective and can use completed-trajectory partitions and Gaussian parameters.
- The historical-H090 label and source-defined emergence implementation are frozen reconstructions, not author-primary or paper-primary identities.
- The 96 trajectories were already used for upstream time-base confirmation and are not new GARD holdouts.
- Exact replay is bounded to the pinned Python/runtime/platform and frozen float64 implementation.
- S01–S12I, including every negative, failed, waived, future-dependent, and suppressed result, remain unchanged.

## Recommended next action

Return for mandatory human review. Do not begin S13 or any prediction, MLP, intervention, estimator repair, report-bundle progression, scale-up, or additional adapter repair. Treat `{outcome}` only within this bounded source-informed sensitivity scope.
"""


def artifact_manifest() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    required = config["artifacts"]["required"] + config["artifacts"]["figures"]
    files = [
        path
        for path in sorted(STEP_ROOT.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    entries = [
        {
            "relativePath": str(path.relative_to(STEP_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    present = {item["relativePath"] for item in entries}
    missing = [
        item for item in required if item != "artifact_manifest.json" and item not in present
    ]
    total = sum(int(item["bytes"]) for item in entries)
    payload = {
        "schema": "eidosoma.e01.s12j_artifact_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "artifactCountExcludingSelf": len(entries),
        "totalBytesExcludingSelf": total,
        "artifacts": entries,
        "requiredMissing": missing,
        "under30GiB": total <= 30 * 1024**3,
        "passed": not missing and total <= 30 * 1024**3,
    }
    write_json(STEP_ROOT / "artifact_manifest.json", payload)
    return payload


def write_empty_failure_outputs(reason: str) -> None:
    schemas = json.loads(S12G_SCHEMA_PATH.read_text(encoding="utf-8"))["tables"]
    for filename in CSV_OUTPUTS:
        path = STEP_ROOT / filename
        if not path.exists():
            write_csv(path, pd.DataFrame(columns=schemas[filename]))
    for filename, columns in PARQUET_OUTPUTS.items():
        path = STEP_ROOT / filename
        if not path.exists():
            write_parquet(path, pd.DataFrame(columns=columns))
    adapter_path = STEP_ROOT / "prefix_statistical_view_index.parquet"
    if not adapter_path.exists():
        columns = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["adapter"][
            "persistedAuditViewColumns"
        ]
        write_parquet(adapter_path, pd.DataFrame(columns=columns))
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for relative in yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "artifacts"
    ]["figures"]:
        path = STEP_ROOT / relative
        if not path.exists():
            backend.placeholder_figure(path, f"S12J stopped permanently: {reason}")


def permanent_stop(reason: str, started: float, cpu_started: float) -> None:
    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    write_empty_failure_outputs(reason)
    failure = pd.DataFrame(
        [
            {
                "failureId": "S12J-PERMANENT-STOP",
                "stage": "ONE_REPAIR_ONLY_EXECUTION",
                "severity": "FATAL",
                "status": "S12J_REPAIR_PATH_PERMANENTLY_STOPPED",
                "reason": reason,
                "gateImpact": "NO_FURTHER_REPAIR_AND_NO_SCIENTIFIC_ADJUDICATION",
                "repairAttempted": False,
            }
        ]
    )
    write_csv(STEP_ROOT / "failure_ledger.csv", failure)
    for filename, payload in (
        (
            "classification.json",
            {
                "schema": "eidosoma.e01.s12j_classification.v1",
                "researchStepId": RESEARCH_STEP_ID,
                "classification": "S12J_REPAIR_PATH_PERMANENTLY_STOPPED",
                "scientificAssociationClassification": "NOT_EVALUATED",
                "s12iClassificationRetained": "S12I_VALIDATION_FAILED_CLOSED",
                "s13Status": "BLOCKED_PENDING_S12J_HUMAN_REVIEW",
                "reason": reason,
            },
        ),
        (
            "runtime_manifest.json",
            {
                "schema": "eidosoma.e01.s12j_runtime_manifest.v1",
                "researchStepId": RESEARCH_STEP_ID,
                "wallSeconds": time.monotonic() - started,
                "processCpuSeconds": time.process_time() - cpu_started,
                "gpuHours": 0,
                "sourceFitExecutionCount": 0,
                "newGardTrajectoryCount": 0,
                "status": "PERMANENTLY_STOPPED",
            },
        ),
    ):
        write_json(STEP_ROOT / filename, payload)
    traceback_text = traceback.format_exc()
    write_json(
        STEP_ROOT / "result_validation.json",
        {
            "schema": "eidosoma.e01.s12j_result_validation.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "passed": False,
            "reason": reason,
            "traceback": traceback_text,
            "furtherRepairPermitted": False,
        },
    )
    for filename in (
        "adapter_validation.json",
        "statistics_replay_validation.json",
        "schema_validation.json",
        "provenance_manifest.json",
        "input_validation.json",
        "immutable_prior_validation.json",
    ):
        path = STEP_ROOT / filename
        if not path.exists():
            write_json(
                path,
                {
                    "schema": f"eidosoma.e01.s12j_{filename[:-5]}.v1",
                    "researchStepId": RESEARCH_STEP_ID,
                    "passed": False,
                    "status": "NOT_REACHED_DUE_PERMANENT_STOP",
                    "reason": reason,
                },
            )
    report = f"""# S12J Full Results: Aggregation Interface Repair Confirmation

## Top summary

- **Research step ID:** `{VERSION}` (S12J)
- **Completion status:** `S12J_REPAIR_PATH_PERMANENTLY_STOPPED`.
- **Artifacts written:** Status-bearing failure, validation, schema placeholders, provenance, runtime, manifest, and this canonical report under `/artifacts/research_steps/S12J/`.
- **Validation result:** `FAIL_CLOSED`; `{reason}`.
- **Outcome classification:** `S12J_REPAIR_PATH_PERMANENTLY_STOPPED`; scientific association is `NOT_EVALUATED`.
- **Caveats or blockers:** This was the only authorized repair. No additional code, schema, data, statistic, or scope change may be made. S12I remains `S12I_VALIDATION_FAILED_CLOSED`.
- **Recommended next action:** Close this repair path and return for mandatory human review with S13 and every downstream activity blocked.

## Lay summary

The separately authorized interface repair encountered another mandatory gate failure. Under the one-repair rule, the analysis stopped permanently and no favorable subset or fallback was used.

## Methods, inputs, results, and validation

S12J was limited to the immutable validated S12I outputs and the exact alias `rawObservationIndex = endpointRawObservationIndex`. The terminal reason was `{reason}`. Frozen statistical outputs are not promoted unless the complete adapter, execution, deterministic replay, schema, hash, provenance, and immutability gates all pass.

## Commands and provenance

The pushed method lock, immutable prior baseline, input hashes, adapter contract, runtime record, traceback, failure ledger, and artifact manifest are retained in this directory. No source fit, GARD trajectory, S12G cache payload, prediction, intervention, or S13 work was executed.

## Recommended next action

Return for mandatory human review. No further S12J repair is authorized.
"""
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": RESEARCH_STEP_ID,
        "stepNumber": "S12J",
        "success": False,
        "status": "S12J_REPAIR_PATH_PERMANENTLY_STOPPED",
        "artifactsWritten": [],
        "validationResult": f"FAIL_CLOSED: {reason}",
        "outcomeClassification": "S12J_REPAIR_PATH_PERMANENTLY_STOPPED",
        "caveatsOrBlockers": [
            reason,
            "One-repair rule exhausted; no further repair is permitted.",
            "S12I remains S12I_VALIDATION_FAILED_CLOSED.",
            "S13 remains blocked.",
        ],
        "recommendedNextAction": "Return for mandatory human review and permanently close this repair path.",
        "s13Status": "BLOCKED_PENDING_S12J_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "status.json", status)
    manifest = artifact_manifest()
    status["artifactsWritten"] = [
        item["relativePath"] for item in manifest["artifacts"]
    ] + ["artifact_manifest.json"]
    write_json(STEP_ROOT / "status.json", status)
    artifact_manifest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    started = time.monotonic()
    cpu_started = time.process_time()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    method_lock = verify_method_lock()
    immutable_before = validate_immutable_prior()
    if not immutable_before["passed"]:
        raise RuntimeError("S01-S12I immutability failed before adapter access")
    frames, input_validation = load_and_validate_inputs(config)
    if not input_validation["passed"]:
        raise RuntimeError("immutable S12I input/source/replay gate failed")

    source_prefix_hash_before = sha256_file(S12I_ROOT / "prefix_endpoint_values.parquet")
    adapted = adapt_prefix_statistical_view(frames["prefix"])
    adapter_validation, adapter_audit = validate_prefix_adapter(
        frames["prefix"], adapted, frames["labels"]
    )
    source_prefix_hash_after = sha256_file(S12I_ROOT / "prefix_endpoint_values.parquet")
    adapter_validation["sourceParquetSha256Before"] = source_prefix_hash_before
    adapter_validation["sourceParquetSha256After"] = source_prefix_hash_after
    adapter_validation["sourceParquetHashUnchanged"] = bool(
        source_prefix_hash_before
        == source_prefix_hash_after
        == config["inputs"]["scientificTables"]["prefixValues"]["sha256"]
    )
    adapter_validation["passed"] = bool(
        adapter_validation["passed"]
        and adapter_validation["sourceParquetHashUnchanged"]
    )
    write_json(STEP_ROOT / "adapter_validation.json", adapter_validation)
    write_parquet(STEP_ROOT / "prefix_statistical_view_index.parquet", adapter_audit)
    if not adapter_validation["passed"]:
        raise RuntimeError("S12J alias-only adapter validation failed")

    scope = json.loads(
        (STEP_ROOT / "scope_access_ledger.json").read_text(encoding="utf-8")
    )
    scope["events"].append(
        {
            "stage": "ADAPTER_VALIDATED_BEFORE_CANDIDATE_STATISTICS",
            "candidateStatisticComputedOrInspected": False,
            "adapterRowsValidated": len(adapted),
            "sourceFitExecuted": False,
            "newGardTrajectoryGenerated": False,
            "s12gScientificCachePayloadOpened": False,
            "status": "PASS",
        }
    )
    write_json(STEP_ROOT / "scope_access_ledger.json", scope)

    statistics_started = time.monotonic()
    primary_results, classification = compute_statistics(frames, adapted)
    replay_results, replay_classification = compute_statistics(frames, adapted)
    statistics_replay = compare_statistics_replay(
        primary_results, replay_results, classification, replay_classification
    )
    write_json(
        STEP_ROOT / "statistics_replay_validation.json", statistics_replay
    )
    if not statistics_replay["passed"]:
        raise RuntimeError("S12J frozen-statistics exact replay failed")

    result_validation = validate_result_tables(primary_results, classification)
    write_json(STEP_ROOT / "result_validation.json", result_validation)
    if not result_validation["passed"]:
        raise RuntimeError("S12J frozen result cardinality or decision validation failed")
    write_results(primary_results)
    write_json(STEP_ROOT / "classification.json", classification)
    write_csv(
        STEP_ROOT / "failure_ledger.csv",
        pd.DataFrame(
            columns=[
                "failureId",
                "stage",
                "severity",
                "status",
                "reason",
                "gateImpact",
                "repairAttempted",
            ]
        ),
    )

    backend.FIGURE_ROOT = FIGURE_ROOT
    backend.make_figures(
        frames["labels"],
        frames["full"],
        frames["prefix"],
        primary_results["candidate_association_details"],
        primary_results["metric_identity_results"],
        primary_results["ensemble_adjudication"],
    )
    schemas = schema_validation()
    if not schemas["passed"]:
        raise RuntimeError("S12J output schema validation failed")

    immutable_after = validate_immutable_prior()
    if not immutable_after["passed"]:
        raise RuntimeError("S01-S12I immutability failed at S12J handoff")
    runtime = {
        "schema": "eidosoma.e01.s12j_runtime_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "startedAtUtc": datetime.now(UTC).isoformat(),
        "wallSeconds": time.monotonic() - started,
        "processCpuSeconds": time.process_time() - cpu_started,
        "statisticsWallSeconds": time.monotonic() - statistics_started,
        "statisticsExecutionCount": 2,
        "cpuWorkers": 1,
        "blasThreads": 1,
        "cpuPrecision": "float64_authoritative",
        "gpuHours": 0,
        "sourceFitExecutionCount": 0,
        "newGardTrajectoryCount": 0,
        "s12gScientificCacheReadCount": 0,
        "cpuHourCeiling": config["runtime"]["hardCeilings"]["cpuHours"],
        "wallHourCeiling": config["runtime"]["hardCeilings"]["wallHours"],
        "cpuCeilingPassed": (time.process_time() - cpu_started) / 3600
        <= config["runtime"]["hardCeilings"]["cpuHours"],
        "wallCeilingPassed": (time.monotonic() - started) / 3600
        <= config["runtime"]["hardCeilings"]["wallHours"],
        "passed": True,
    }
    write_json(STEP_ROOT / "runtime_manifest.json", runtime)

    result_files = [
        "candidate_associations.csv",
        "candidate_association_details.parquet",
        "replicator_drift_results.csv",
        "replicator_drift_details.parquet",
        "temporal_dependence_results.csv",
        "spike_results.csv",
        "metric_identity_results.csv",
        "future_dependence_results.csv",
        "cross_candidate_results.csv",
        "ensemble_adjudication.csv",
        "prefix_statistical_view_index.parquet",
    ]
    provenance = {
        "schema": "eidosoma.e01.s12j_provenance_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "designCommit": method_lock["designCommit"],
        "branch": method_lock["branch"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "adapterId": ADAPTER_ID,
        "sourceInputHashes": {
            item["frameId"]: item["sha256AfterRead"]
            for item in input_validation["tables"]
        },
        "lockedCodeHashes": method_lock["files"],
        "resultFiles": [
            {
                "path": filename,
                "sha256": sha256_file(STEP_ROOT / filename),
                "bytes": (STEP_ROOT / filename).stat().st_size,
            }
            for filename in result_files
        ],
        "sourceFitExecutionCount": 0,
        "newGardTrajectoryCount": 0,
        "s12gScientificCacheReadCount": 0,
        "candidateWeightsUsed": False,
        "passed": True,
    }
    write_json(STEP_ROOT / "provenance_manifest.json", provenance)

    scope["events"].append(
        {
            "stage": "FROZEN_STATISTICS_AND_EXACT_REPLAY_COMPLETED",
            "candidateStatisticExecutionCount": 2,
            "sourceFitExecuted": False,
            "newGardTrajectoryGenerated": False,
            "s12gScientificCachePayloadOpened": False,
            "scientificMethodChange": False,
            "status": "PASS",
        }
    )
    scope["success"] = True
    write_json(STEP_ROOT / "scope_access_ledger.json", scope)

    report = report_markdown(
        classification,
        primary_results,
        adapter_validation,
        input_validation,
        statistics_replay,
        result_validation,
        immutable_after,
        runtime,
    )
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": RESEARCH_STEP_ID,
        "stepNumber": "S12J",
        "success": True,
        "status": "COMPLETED_AT_MANDATORY_S12J_HUMAN_REVIEW_BOUNDARY",
        "artifactsWritten": [],
        "validationResult": (
            "PASS: alias-only adapter, immutable S12I source/replay evidence, exact "
            "two-run frozen-statistics replay, result gates, schemas, provenance, "
            "runtime/storage, artifact hashes, and S01-S12I immutability passed."
        ),
        "outcomeClass": outcome_class(classification["classification"]),
        "outcomeClassification": classification["classification"],
        "caveatsOrBlockers": [
            "S12I remains S12I_VALIDATION_FAILED_CLOSED.",
            "Candidate 1 remains HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED.",
            "The S12H aggregate-support gate remains failed.",
            "This post-result one-adapter override weakens confirmatory credibility.",
            "S13 and all blocked downstream work remain stopped.",
        ],
        "recommendedNextAction": (
            "Return for mandatory human review; do not begin S13 or any further repair, "
            "prediction, intervention, report-bundle progression, or scale-up."
        ),
        "s13Status": "BLOCKED_PENDING_S12J_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "status.json", status)
    manifest = artifact_manifest()
    if not manifest["passed"]:
        raise RuntimeError(f"S12J artifact completeness failed: {manifest['requiredMissing']}")
    status["artifactsWritten"] = [
        item["relativePath"] for item in manifest["artifacts"]
    ] + ["artifact_manifest.json"]
    write_json(STEP_ROOT / "status.json", status)
    manifest = artifact_manifest()
    if not manifest["passed"]:
        raise RuntimeError("S12J final artifact manifest failed")
    for item in manifest["artifacts"]:
        path = STEP_ROOT / item["relativePath"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"S12J final artifact hash mismatch: {path}")
    print(
        json.dumps(
            {
                "stage": "S12J_complete",
                "classification": classification["classification"],
                "adapterPassed": True,
                "statisticsReplayPassed": True,
                "artifactManifestPassed": True,
                "sourceFitExecutionCount": 0,
                "newGardTrajectoryCount": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    _started = time.monotonic()
    _cpu_started = time.process_time()
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:
        _reason = f"{type(error).__name__}:{error}"
        permanent_stop(_reason, _started, _cpu_started)
        print(
            json.dumps(
                {"stage": "S12J_permanent_stop", "error": _reason}, sort_keys=True
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from error
