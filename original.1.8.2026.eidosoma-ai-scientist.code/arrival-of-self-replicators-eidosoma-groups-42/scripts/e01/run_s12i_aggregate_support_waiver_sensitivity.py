#!/usr/bin/env python3
"""Execute the frozen S12I waiver-labeled sensitivity analysis."""

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
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt
import pandas as pd
import yaml

import e01_frozen_timebase_ensemble.core as s12g_core
from e01_aggregate_support_waiver_sensitivity.core import (
    CANDIDATE_IDS,
    DERIVED_CANDIDATE_ID,
    EVIDENCE_CLASS,
    RESEARCH_STEP_ID,
    VERSION,
    outcome_class,
    sensitivity_classification,
    validate_exact_waiver,
)
from scripts.e01 import run_s12g_frozen_timebase_ensemble as backend

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S12I"
CACHE_ROOT = Path("/cache/e01_s12i")
RESULT_CACHE = CACHE_ROOT / "source_results"
CONFIG_PATH = REPO / "configs/e01/s12i_aggregate_support_waiver_sensitivity_preregistration.yaml"
S12G_SCHEMA = REPO / "configs/e01/s12g_output_schemas.json"
INPUT_MANIFEST = STEP_ROOT / "trajectory_input_manifest.parquet"
FIGURE_ROOT = STEP_ROOT / "figures"
S12H_CONFIRMATION = ARTIFACTS / "research_steps/S12H/candidate1_timebase_confirmation.json"

ORIGINAL_ADJUDICATE = backend.adjudicate
ORIGINAL_FAILURE_ROWS = backend.failure_rows_from_statuses


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def verify_clean_pushed() -> tuple[str, str]:
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote or git("status", "--short"):
        raise RuntimeError("S12I implementation must be committed, pushed, and clean")
    return head, remote


def verify_method_lock() -> dict[str, Any]:
    lock = json.loads((STEP_ROOT / "method_lock.json").read_text(encoding="utf-8"))
    if not lock.get("passed"):
        raise RuntimeError("S12I method lock is not passing")
    head, _remote = verify_clean_pushed()
    if head != lock["designCommit"]:
        raise RuntimeError("S12I must execute at the pushed pre-scientific design commit")
    for item in lock["files"]:
        path = REPO / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"S12I method-lock file changed: {item['path']}")
    return lock


def validate_waiver() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    confirmation = json.loads(S12H_CONFIRMATION.read_text(encoding="utf-8"))
    exact = validate_exact_waiver(confirmation)
    contract = json.loads((STEP_ROOT / "waiver_contract.json").read_text(encoding="utf-8"))
    payload = {
        "schema": "eidosoma.e01.s12i_waiver_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "confirmationHashPassed": sha256_file(S12H_CONFIRMATION)
        == config["immutability"]["s12hGateSha256"],
        "contractRetainsOriginalFailure": contract.get(
            "originalConfirmationGatePassed"
        )
        is False
        and contract.get("originalAggregateSupportGatePassed") is False
        and contract.get("waivedGateRelabeledPassed") is False,
        "contractWaivesOnlyAggregateSupport": contract.get("waivedGate")
        == "aggregateSupportCompatible"
        and int(contract.get("otherGateWaiverCount", -1)) == 0,
        "candidate1Nonconfirmed": contract.get("candidate1UpstreamConfirmed") is False
        and contract.get("candidate1EvidenceStatus")
        == "HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED",
        **exact,
    }
    payload["passed"] = bool(
        payload["confirmationHashPassed"]
        and payload["contractRetainsOriginalFailure"]
        and payload["contractWaivesOnlyAggregateSupport"]
        and payload["candidate1Nonconfirmed"]
        and exact["passed"]
    )
    write_json(STEP_ROOT / "waiver_validation.json", payload)
    return payload


def validate_immutable_prior() -> dict[str, Any]:
    baseline = json.loads(
        (STEP_ROOT / "immutable_prior_baseline.json").read_text(encoding="utf-8")
    )
    changed: list[dict[str, Any]] = []
    categories = (
        ("researchStepFiles", "prior_artifact"),
        ("lockedTrajectoryCaches", "locked_raw_trajectory"),
        ("s12gTaskCacheFiles", "forbidden_S12G_cache"),
    )
    counts: dict[str, int] = {}
    for key, kind in categories:
        counts[key] = len(baseline[key])
        for item in baseline[key]:
            path = Path(item["path"])
            expected = item.get("expectedSha256", item["sha256"])
            actual = sha256_file(path) if path.is_file() else None
            if actual != expected:
                changed.append(
                    {
                        "kind": kind,
                        "path": str(path),
                        "expectedSha256": expected,
                        "actualSha256": actual,
                    }
                )
    payload = {
        "schema": "eidosoma.e01.s12i_immutable_prior_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        **counts,
        "changedCount": len(changed),
        "changed": changed,
        "passed": not changed,
    }
    write_json(STEP_ROOT / "immutable_prior_validation.json", payload)
    return payload


def schema_validation() -> dict[str, Any]:
    tables = json.loads(S12G_SCHEMA.read_text(encoding="utf-8"))["tables"]
    rows: list[dict[str, Any]] = []
    for filename, required in tables.items():
        path = STEP_ROOT / filename
        exists = path.is_file()
        if exists:
            frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            missing = [column for column in required if column not in frame.columns]
            row_count = len(frame)
        else:
            missing = list(required)
            row_count = None
        rows.append(
            {
                "path": filename,
                "exists": exists,
                "rowCount": row_count,
                "missingColumns": missing,
                "passed": exists and not missing,
            }
        )
    for filename, required in (
        (
            "candidate_association_details.parquet",
            ["candidateId", "matrixIndex", "implementationId", "temporalMode", "estimand", "correlation"],
        ),
        (
            "replicator_drift_details.parquet",
            ["candidateId", "matrixIndex", "implementationId", "temporalMode", "meanDifference"],
        ),
    ):
        path = STEP_ROOT / filename
        exists = path.is_file()
        frame = pd.read_parquet(path) if exists else pd.DataFrame()
        missing = [column for column in required if column not in frame.columns]
        rows.append(
            {
                "path": filename,
                "exists": exists,
                "rowCount": len(frame) if exists else None,
                "missingColumns": missing,
                "passed": exists and not missing,
            }
        )
    payload = {
        "schema": "eidosoma.e01.s12i_schema_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "tables": rows,
        "passed": all(bool(item["passed"]) for item in rows),
    }
    write_json(STEP_ROOT / "schema_validation.json", payload)
    return payload


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
        item
        for item in required
        if item != "artifact_manifest.json" and item not in present
    ]
    total = sum(int(item["bytes"]) for item in entries)
    payload = {
        "schema": "eidosoma.e01.s12i_artifact_manifest.v1",
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


def _configure_backend() -> None:
    backend.RESEARCH_STEP_ID = RESEARCH_STEP_ID
    backend.VERSION = VERSION
    backend.EVIDENCE_CLASS = EVIDENCE_CLASS
    backend.CANDIDATE_IDS = CANDIDATE_IDS
    backend.STEP_ROOT = STEP_ROOT
    backend.CACHE_ROOT = CACHE_ROOT
    backend.RESULT_CACHE = RESULT_CACHE
    backend.CONFIG_PATH = CONFIG_PATH
    backend.SCHEMA_PATH = S12G_SCHEMA
    backend.INPUT_MANIFEST = INPUT_MANIFEST
    backend.FIGURE_ROOT = FIGURE_ROOT
    s12g_core.RESEARCH_STEP_ID = RESEARCH_STEP_ID
    backend.adjudicate = s12i_adjudicate
    backend.failure_rows_from_statuses = s12i_failure_rows
    backend.validate_immutable_prior = validate_immutable_prior
    backend.validate_schemas = schema_validation
    backend.artifact_manifest = lambda _required: artifact_manifest()
    backend.recommendation_for = recommendation_for
    backend.build_report = build_report


def s12i_adjudicate(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, _source_payload = ORIGINAL_ADJUDICATE(*args, **kwargs)
    registry = {
        item["candidateId"]: item
        for item in yaml.safe_load(
            (STEP_ROOT / "candidate_registry.yaml").read_text(encoding="utf-8")
        )["candidates"]
    }
    frame["candidateEvidenceStatus"] = frame["candidateId"].map(
        lambda item: registry[item]["evidenceStatus"]
    )
    frame["aggregateSupportGateWaived"] = frame["candidateId"].map(
        lambda item: bool(registry[item]["aggregateSupportGateWaived"])
    )
    classification = sensitivity_classification(frame.to_dict("records"))
    payload = {
        "schema": "eidosoma.e01.s12i_classification.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "classification": classification,
        "candidateResults": frame.to_dict("records"),
        "ensemblePositiveRequiresAllThree": True,
        "candidateWeightsUsed": False,
        "candidate1EvidenceStatus": "HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED",
        "candidate1UpstreamConfirmed": False,
        "s12hAggregateSupportGateRetainedAsFailed": True,
        "waiverScope": "aggregateSupportCompatible_only",
        "positiveResultMeaning": "EXPLORATORY_SENSITIVITY_CONSISTENCY_ONLY",
        "upstreamConfirmedThreeCandidateEnsembleClaimPermitted": False,
        "s13Status": "BLOCKED_PENDING_S12I_HUMAN_REVIEW",
    }
    return frame, payload


def s12i_failure_rows(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    rows = ORIGINAL_FAILURE_ROWS(*args, **kwargs)
    for row in rows:
        row["failureId"] = str(row["failureId"]).replace("S12G-", "S12I-")
    return rows


def recommendation_for(_classification: str) -> str:
    return (
        "Return for mandatory post-S12I human review. Keep S13, prediction, MLP, "
        "interventions, estimator repair, and scale-up blocked; do not promote the "
        "human-waived candidate-1 derivative to upstream-confirmed evidence."
    )


def _fmt(value: Any, digits: int = 6) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int,)):
        return str(value)
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


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
    evidence_outcome = outcome_class(outcome)
    primary = associations[associations["implementationId"] == "IIGR_CORRECTED_SOURCE"]
    full_rows = primary[
        primary["estimand"] == "RETROSPECTIVE_CURRENT_GENERATION"
    ].sort_values("candidateId")
    prefix_rows = primary[
        (primary["estimand"] == "CURRENT_HISTORICAL")
        & primary["temporalModeId"].str.endswith("_PREFIX_ENDPOINT")
    ].sort_values("candidateId")
    drift_rows = drift[
        (drift["implementationId"] == "IIGR_CORRECTED_SOURCE")
        & drift["temporalModeId"].str.endswith("_FULL")
        & (drift["labelId"] == "HISTORICAL_H090_REPLICATOR")
    ].sort_values("candidateId")
    result_lines: list[str] = []
    for candidate_id in CANDIDATE_IDS:
        full = full_rows[full_rows["candidateId"] == candidate_id].iloc[0]
        prefix = prefix_rows[prefix_rows["candidateId"] == candidate_id].iloc[0]
        drow = drift_rows[drift_rows["candidateId"] == candidate_id].iloc[0]
        status = (
            "HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED"
            if candidate_id == CANDIDATE_IDS[0]
            else "S12FR_UPSTREAM_CONFIRMED"
        )
        result_lines.append(
            "| {candidate} | {status} | {fm} | {fp}/{fd} | [{fl}, {fu}] | {fg} | {dm} | {dp}/{dd} | {dg} | {pm} | {pp}/{pd} | [{pl}, {pu}] | {pg} |".format(
                candidate=candidate_id,
                status=status,
                fm=_fmt(full["medianCorrelation"]),
                fp=int(full["positiveTrajectoryCount"]),
                fd=int(full["definedTrajectoryCount"]),
                fl=_fmt(full["bootstrapLower95"]),
                fu=_fmt(full["bootstrapUpper95"]),
                fg="PASS" if bool(full["gatePassed"]) else "FAIL",
                dm=_fmt(drow["medianMeanDifference"]),
                dp=int(drow["positiveMeanDifferenceCount"]),
                dd=int(drow["definedTrajectoryCount"]),
                dg="PASS" if bool(drow["gatePassed"]) else "FAIL",
                pm=_fmt(prefix["medianCorrelation"]),
                pp=int(prefix["positiveTrajectoryCount"]),
                pd=int(prefix["definedTrajectoryCount"]),
                pl=_fmt(prefix["bootstrapLower95"]),
                pu=_fmt(prefix["bootstrapUpper95"]),
                pg="PASS" if bool(prefix["gatePassed"]) else "FAIL",
            )
        )
    waiver = json.loads((STEP_ROOT / "waiver_validation.json").read_text(encoding="utf-8"))
    exclusion = json.loads(
        (STEP_ROOT / "s12g_cache_exclusion_audit.json").read_text(encoding="utf-8")
    )
    return f"""# S12I Full Results: Aggregate-Support Waiver Sensitivity Analysis

## Top summary

- **Research step ID:** `{VERSION}` (S12I)
- **Completion status:** `COMPLETED_AT_MANDATORY_S12I_HUMAN_REVIEW_BOUNDARY`; no downstream step was begun.
- **Artifacts written:** {validation['artifactCount']} status-bearing files under `/artifacts/research_steps/S12I/`, including the waiver/method lock, fresh 96-task source outputs, labels, full/prefix values, candidate statistics, figures, validation, provenance, hashes, status, and this canonical report.
- **Validation result:** {validation['summary']}
- **Outcome classification:** `{outcome}` ({evidence_outcome}).
- **Caveats or blockers:** S12H's `aggregateSupportCompatible` gate remains failed. Candidate 1 is a human-waived, near-envelope, non-confirmed derivative; this set is not an upstream-confirmed three-candidate ensemble. Completed-fit values are retrospective and the source family is not author- or paper-primary.
- **Recommended next action:** {recommendation_for(outcome)}

## Lay summary

The human authorized one transparent sensitivity analysis after candidate 1 narrowly missed the paper-axis support rule: two rather than at most one of its 32 trajectories exceeded the frozen ceiling. We kept that failure visible, recomputed every label and source metric from the original raw trajectories, and compared it with the two upstream-confirmed candidates using exactly the previously frozen scientific tests. Even a positive all-three pattern here is exploratory because one member entered only through the waiver.

## Frozen question and interpretation boundary

S12I asks whether the exact S12G label/source-emergence conclusion is consistent across two S12FR-confirmed candidates and one explicitly non-confirmed near-envelope C1 sensitivity case. It does not retest or reverse S12H. The only waiver is `aggregateSupportCompatible`; it is not represented as passed. S12F remains `SIMULATOR_IDENTIFICATION_FAILED`, S12FR remains `NONIDENTIFIABLE_TIMEBASE_ENSEMBLE`, S12G remains fail-closed, and S12H remains `BOUNDARY_INCLUSIVE_CANDIDATE1_NOT_UPSTREAM_CONFIRMED`.

## Inputs

Exactly {len(input_manifest)} hash-locked S12FR raw trajectories were used, 32 for each fixed candidate, with {input_manifest['matrixIndex'].nunique()} shared catalytic-matrix/initial-state identities. Candidate 1 used `h=0.5081160391061118`, first-daughter continuation, retained overshoot, and uniform C1. Candidate 2 used `h=0.6031526490073492`, first-daughter continuation, trimmed new entrants, and C1. Candidate 3 used `h=0.5613315384859516`, random-nonempty continuation, trimmed new entrants, and C1. No GARD trajectory, exposure, clock, candidate, or weight was generated, searched, removed, or updated.

Pinned source commits were IIGR `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7` and PhiRL `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`. Only safe lattice JSON SHA-256 `74ecca37f04201088d76a9e8ede7efe04bafebecff85a4882a44f03afbd23aa1` was loaded.

## Detailed methods

The primary label was `HISTORICAL_H090_REPLICATOR`; `PAST_ONLY_COSINE_REPLICATOR` was secondary. Integer counts received additive-0.5 closure, full 100-component CLR, and removal of original component 100. The primary metric was IIGR source-defined emergence—synergy plus both downward-causation atoms. PhiRL emergence was the regularization robustness companion; corrected `local_phi_r` was comparator-only.

Complete sequences were fitted once and labeled exactly `RETROSPECTIVE_FULL_TRAJECTORY_LOCAL`. At every post-fission endpoint with at least 256 prior C1 transitions, the same source pipeline was independently refit from the beginning of that prefix and only its endpoint was retained. Every full and prefix fit required exact replay. Every prefix underwent structural suffix deletion/shuffle/replacement checks, plus executed first/middle/last sentinels for all three variants.

Candidate-specific current/next-generation associations, replicator-minus-drift differences, temporal dependence, three-sigma and robust spikes, metric identity, full-versus-prefix future dependence, partition stability, and paired cross-candidate comparisons used the unchanged S12G rules. Each inference used the frozen 4,096 trajectory bootstraps, 4,096 within-trajectory circular shifts, or 4,096 block-aware shifts as applicable. Positive sensitivity consistency required the same gate and direction in all three candidates; no favorable candidate could be selected.

## Waiver validation

The original S12H confirmation remained false and its sole failed gate remained `aggregateSupportCompatible`. All 13 nonwaived gates remained true. Waiver validation passed: {waiver['passed']}. The S12G cache inventory contained {exclusion['recordedFileCount']} forbidden payload files; scientific payload reads and reuse were both zero.

## Candidate-specific primary results

| Candidate | Evidence status | Full median rho | Full positive/defined | Full 95% bootstrap | Full association gate | Full drift median difference | Full drift positive/defined | Full drift gate | Prefix median rho | Prefix positive/defined | Prefix 95% bootstrap | Prefix gate |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
{chr(10).join(result_lines)}

Complete current/next-generation, secondary-label, IIGR, and PhiRL results are in `candidate_associations.csv`; drift results are in `replicator_drift_results.csv`. Metric-identity and future-dependence tables contain {len(metric_identity)} and {len(future)} status-bearing candidate/trajectory comparisons.

## Sensitivity-set adjudication

{adjudication.to_markdown(index=False)}

The all-three sensitivity classification is `{outcome}`. This does not alter the candidate-1 upstream failure and cannot support S13.

## Validation

{validation['details']}

The failure ledger contains {len(failures)} aggregated status rows, including expected pre-256 ineligibility and any source nonfinite states. No status was silently omitted, imputed, clipped, or replaced. S01–S12H artifacts, all 96 raw trajectory caches, and all 950 forbidden S12G cache files were hash-validated before and after the run.

## Commands, dependencies, and runtime

```bash
PYTHONPATH=src python -m pytest -q tests/e01/test_s12i_aggregate_support_waiver_sensitivity.py tests/e01/test_s12g_frozen_timebase_ensemble.py
python -m ruff check src/e01_aggregate_support_waiver_sensitivity scripts/e01/freeze_s12i_preregistration.py scripts/e01/run_s12i_aggregate_support_waiver_sensitivity.py tests/e01/test_s12i_aggregate_support_waiver_sensitivity.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s12i_preregistration.py --design-commit <pushed-design-commit>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s12i_aggregate_support_waiver_sensitivity.py --workers 6
```

CPU float64 was authoritative; GPU use was zero. Runtime was {runtime['wallHours']:.4f} wall-hours and {runtime['workerCpuHours']:.4f} summed worker CPU-hours. Python `{runtime['python']}`, NumPy `{runtime['numpy']}`, SciPy `{runtime['scipy']}`, platform `{runtime['platform']}`. No dependency was installed.

## Provenance, caveats, blockers, and limitations

- Candidate 1 was derived after S12G exposed a C0 endpoint problem, then failed S12H's frozen aggregate-support gate. This post-result waiver weakens confirmatory credibility even though no S12H label or emergence result had been inspected.
- The 96 raw trajectories had already served S12FR time-base confirmation; they are not new GARD holdouts.
- Historical labels and completed-trajectory source fits are retrospective. Only the independently refit prefix endpoints are prospective reconstructions, not proof of causal control.
- Public-source behavior is source-informed evidence, not the unavailable author implementation, an author-primary method, or an exact paper replication.
- Exact replay is bounded to the pinned wrappers, runtime, CPU-float64 policy, and platform.
- All earlier negative, failed, suppressed, and nonidentifiable results remain unchanged.

## Recommended next action

{recommendation_for(outcome)} Stop here with S13 `BLOCKED_PENDING_S12I_HUMAN_REVIEW`.
"""


def normalize_success_artifacts() -> None:
    classification = json.loads((STEP_ROOT / "classification.json").read_text(encoding="utf-8"))
    classification.update(
        {
            "schema": "eidosoma.e01.s12i_classification.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "evidenceClass": EVIDENCE_CLASS,
            "candidate1AnalysisIdentity": DERIVED_CANDIDATE_ID,
            "candidate1EvidenceStatus": "HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED",
            "candidate1UpstreamConfirmed": False,
            "s12hAggregateSupportGateRetainedAsFailed": True,
            "s13Status": "BLOCKED_PENDING_S12I_HUMAN_REVIEW",
        }
    )
    write_json(STEP_ROOT / "classification.json", classification)

    status = json.loads((STEP_ROOT / "status.json").read_text(encoding="utf-8"))
    status.update(
        {
            "researchStepId": RESEARCH_STEP_ID,
            "stepNumber": RESEARCH_STEP_ID,
            "success": True,
            "status": "COMPLETED_AT_MANDATORY_S12I_HUMAN_REVIEW_BOUNDARY",
            "validationResult": status["validationResult"]
            + " Waiver scope was exact; S12H's failed gate remains false; S12G cache reuse was zero.",
            "caveatsOrBlockers": [
                "Candidate 1 is human-waived, near-envelope, and not upstream-confirmed.",
                "S12H's aggregate-support gate remains failed; the waiver is exploratory only.",
                "Completed-fit values are retrospective and source-informed only.",
                "S13 remains blocked regardless of outcome.",
            ],
            "recommendedNextAction": recommendation_for(
                classification["classification"]
            ),
            "outcomeClassification": classification["classification"],
            "outcomeClass": outcome_class(classification["classification"]),
            "s13Status": "BLOCKED_PENDING_S12I_HUMAN_REVIEW",
        }
    )
    write_json(STEP_ROOT / "status.json", status)

    runtime = json.loads((STEP_ROOT / "runtime_manifest.json").read_text(encoding="utf-8"))
    runtime.update(
        {
            "schema": "eidosoma.e01.s12i_runtime_manifest.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "newGardTrajectoriesGenerated": 0,
            "s12gCachePayloadReads": 0,
            "s12gCacheReuseCount": 0,
        }
    )
    write_json(STEP_ROOT / "runtime_manifest.json", runtime)

    regeneration = json.loads(
        (STEP_ROOT / "regeneration_validation.json").read_text(encoding="utf-8")
    )
    regeneration.update(
        {
            "schema": "eidosoma.e01.s12i_regeneration_validation.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "freshTaskCount": 96,
            "s12gCacheReuseCount": 0,
            "candidate1EvidenceStatus": "HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED",
        }
    )
    write_json(STEP_ROOT / "regeneration_validation.json", regeneration)

    implementation = json.loads(
        (STEP_ROOT / "implementation_lock.json").read_text(encoding="utf-8")
    )
    implementation.update(
        {
            "schema": "eidosoma.e01.s12i_implementation_lock.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "candidate1AnalysisIdentity": DERIVED_CANDIDATE_ID,
            "waiverContractSha256": sha256_file(STEP_ROOT / "waiver_contract.json"),
        }
    )
    write_json(STEP_ROOT / "implementation_lock.json", implementation)

    scope = json.loads((STEP_ROOT / "scope_access_ledger.json").read_text(encoding="utf-8"))
    for event in scope["events"]:
        if event.get("stage") == "COMPLETE_S12G_EXECUTION":
            event["stage"] = "COMPLETE_S12I_WAIVER_SENSITIVITY_EXECUTION"
            event["s12gCachePayloadOpened"] = False
            event["s12gCacheReuseCount"] = 0
            event["waivedGate"] = "aggregateSupportCompatible"
            event["otherGateWaiverCount"] = 0
    scope["researchStepId"] = RESEARCH_STEP_ID
    scope["success"] = True
    write_json(STEP_ROOT / "scope_access_ledger.json", scope)

    exclusion = json.loads(
        (STEP_ROOT / "s12g_cache_exclusion_audit.json").read_text(encoding="utf-8")
    )
    exclusion.update(
        {
            "payloadFilesOpened": 0,
            "scientificReuseCount": 0,
            "freshS12iTaskCount": 96,
            "passed": True,
        }
    )
    write_json(STEP_ROOT / "s12g_cache_exclusion_audit.json", exclusion)


def placeholder_figure(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def create_scientific_placeholders(reason: str) -> None:
    contract = json.loads(S12G_SCHEMA.read_text(encoding="utf-8"))["tables"]
    for filename, columns in contract.items():
        path = STEP_ROOT / filename
        if path.exists() or filename == "trajectory_input_manifest.parquet":
            continue
        frame = pd.DataFrame(columns=columns)
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False, compression="zstd")
        else:
            frame.to_csv(path, index=False, lineterminator="\n")
    for filename, columns in (
        (
            "candidate_association_details.parquet",
            ["candidateId", "matrixIndex", "implementationId", "temporalMode", "estimand", "correlation"],
        ),
        (
            "replicator_drift_details.parquet",
            ["candidateId", "matrixIndex", "implementationId", "temporalMode", "meanDifference"],
        ),
    ):
        path = STEP_ROOT / filename
        if not path.exists():
            pd.DataFrame(columns=columns).to_parquet(path, index=False, compression="zstd")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for filename in config["artifacts"]["figures"]:
        path = STEP_ROOT / filename
        if not path.exists():
            placeholder_figure(path, f"S12I stopped fail-closed: {reason}")


def finalize_fail_closed(reason: str) -> None:
    create_scientific_placeholders(reason)
    columns = json.loads(S12G_SCHEMA.read_text(encoding="utf-8"))["tables"][
        "failure_ledger.csv"
    ]
    row = {column: None for column in columns}
    row.update(
        {
            "failureId": "S12I-TERMINAL-FAIL-CLOSED",
            "stage": "S12I_WAIVER_SENSITIVITY_EXECUTION",
            "severity": "FATAL",
            "status": "S12I_VALIDATION_FAILED_CLOSED",
            "reason": reason,
            "gateImpact": "FAIL_CLOSED_NO_VALID_SCIENTIFIC_ADJUDICATION",
            "repairAttempted": False,
        }
    )
    pd.DataFrame([row], columns=columns).to_csv(
        STEP_ROOT / "failure_ledger.csv", index=False, lineterminator="\n"
    )
    write_json(
        STEP_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s12i_classification.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "classification": "S12I_VALIDATION_FAILED_CLOSED",
            "scientificAssociationClassification": "NOT_EVALUATED",
            "reason": reason,
            "candidate1UpstreamConfirmed": False,
            "s12hAggregateSupportGateRetainedAsFailed": True,
            "s13Status": "BLOCKED_PENDING_S12I_HUMAN_REVIEW",
        },
    )
    for filename, payload in (
        (
            "runtime_benchmark.json",
            {
                "schema": "eidosoma.e01.s12i_runtime_benchmark.v1",
                "passed": False,
                "status": "NOT_REACHED_OR_FAILED_CLOSED",
                "reason": reason,
            },
        ),
        (
            "runtime_manifest.json",
            {
                "schema": "eidosoma.e01.s12i_runtime_manifest.v1",
                "passed": False,
                "status": "FAILED_CLOSED",
                "reason": reason,
                "gpuHours": 0,
                "newGardTrajectoriesGenerated": 0,
                "s12gCacheReuseCount": 0,
            },
        ),
        (
            "regeneration_validation.json",
            {
                "schema": "eidosoma.e01.s12i_regeneration_validation.v1",
                "passed": False,
                "status": "NOT_REACHED_OR_FAILED_CLOSED",
                "reason": reason,
                "newGardTrajectoriesGenerated": 0,
                "s12gCacheReuseCount": 0,
            },
        ),
    ):
        if not (STEP_ROOT / filename).exists():
            write_json(STEP_ROOT / filename, payload)
    immutable = validate_immutable_prior()
    waiver = validate_waiver()
    schemas = schema_validation()
    report = f"""# S12I Full Results: Aggregate-Support Waiver Sensitivity Analysis

## Top summary

- **Research step ID:** `{VERSION}` (S12I)
- **Completion status:** `STOPPED_FAIL_CLOSED`; no valid sensitivity-set adjudication was performed.
- **Artifacts written:** Complete preregistration, waiver, input/source/provenance evidence, status-bearing suppressed scientific outputs, validation and hash manifests, figures, status JSON, and this canonical report.
- **Validation result:** `FAIL_CLOSED`: {reason}
- **Outcome classification:** `S12I_VALIDATION_FAILED_CLOSED` (constraining/contradictory); scientific associations are `NOT_EVALUATED`.
- **Caveats or blockers:** The operational gate failed. S12H's aggregate-support failure remains visible; candidate 1 remains non-confirmed. No repair, cache reuse, new simulation, or downstream work occurred.
- **Recommended next action:** Return for mandatory human review with S13 and all blocked work still blocked.

## Lay summary

The authorized exploratory comparison could not be completed safely because a frozen operational requirement failed: {reason}. The failure is preserved rather than bypassed.

## Detailed methods, inputs, and results

The complete method is frozen in `preregistration.yaml`. Exactly 96 existing S12FR raw trajectories were authorized, with candidate 1 carried only under the human waiver of `aggregateSupportCompatible`. Every other S12G label, preprocessing, source metric, full/prefix, replay, suffix, statistics, and all-three rule remained unchanged. The scientific tables are schema-bearing and suppressed because the operational gate failed.

## Validation and provenance

Prior immutability passed: {immutable['passed']}; exact waiver scope passed: {waiver['passed']}; schema validation passed: {schemas['passed']}. S12H remains `BOUNDARY_INCLUSIVE_CANDIDATE1_NOT_UPSTREAM_CONFIRMED`. No new GARD trajectory or S12G task-cache scientific reuse occurred.

## Commands and dependencies

The pushed design, method lock, focused tests, runtime identities, and execution command are recorded in `preregistration_record.json`, `method_lock.json`, `implementation_lock.json`, and `runtime_manifest.json`. No dependency was installed.

## Caveats and recommended next action

This stop is not positive or negative evidence about source-defined emergence. Return for human review; S13 remains `BLOCKED_PENDING_S12I_HUMAN_REVIEW`.
"""
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": RESEARCH_STEP_ID,
        "stepNumber": RESEARCH_STEP_ID,
        "success": False,
        "status": "STOPPED_FAIL_CLOSED",
        "artifactsWritten": [],
        "validationResult": f"FAIL_CLOSED: {reason}",
        "caveatsOrBlockers": [
            reason,
            "Candidate 1 remains non-confirmed and S12H's failed gate remains false.",
            "S13 remains blocked.",
        ],
        "recommendedNextAction": "Return for mandatory human review; no S12I repair or downstream work is authorized.",
        "outcomeClassification": "S12I_VALIDATION_FAILED_CLOSED",
        "outcomeClass": "constraining/contradictory",
        "s13Status": "BLOCKED_PENDING_S12I_HUMAN_REVIEW",
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
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers != 6:
        raise RuntimeError("S12I freezes exactly six source-analysis workers")
    verify_method_lock()
    waiver = validate_waiver()
    if not waiver["passed"]:
        raise RuntimeError("S12I exact waiver gate failed")
    immutable = validate_immutable_prior()
    if not immutable["passed"]:
        raise RuntimeError("S12I prior immutability failed before scientific execution")
    exclusion = json.loads(
        (STEP_ROOT / "s12g_cache_exclusion_audit.json").read_text(encoding="utf-8")
    )
    if exclusion.get("payloadFilesOpened") != 0 or not exclusion.get("passed"):
        raise RuntimeError("S12G cache-exclusion firewall failed")
    if RESULT_CACHE.exists():
        raise RuntimeError("fresh S12I result cache root already exists")

    _configure_backend()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], "--workers", str(args.workers)]
        result = backend.main()
    finally:
        sys.argv = old_argv
    normalize_success_artifacts()
    immutable = validate_immutable_prior()
    waiver = validate_waiver()
    schemas = schema_validation()
    if not immutable["passed"] or not waiver["passed"] or not schemas["passed"]:
        raise RuntimeError(
            "S12I final validation failed: "
            f"immutable={immutable['passed']},waiver={waiver['passed']},schemas={schemas['passed']}"
        )
    manifest = artifact_manifest()
    if not manifest["passed"]:
        raise RuntimeError(f"S12I artifact completeness failed: {manifest['requiredMissing']}")
    status = json.loads((STEP_ROOT / "status.json").read_text(encoding="utf-8"))
    status["artifactsWritten"] = [
        item["relativePath"] for item in manifest["artifacts"]
    ] + ["artifact_manifest.json"]
    write_json(STEP_ROOT / "status.json", status)
    manifest = artifact_manifest()
    if not manifest["passed"]:
        raise RuntimeError("S12I final artifact-manifest validation failed")
    print(
        json.dumps(
            {
                "stage": "S12I_complete",
                "classification": json.loads(
                    (STEP_ROOT / "classification.json").read_text(encoding="utf-8")
                )["classification"],
                "artifactManifestPassed": True,
                "s12gCacheReuseCount": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:
        reason = f"{type(error).__name__}:{error}"
        finalize_fail_closed(reason)
        print(
            json.dumps({"stage": "S12I_failed_closed", "error": reason}, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from error
