#!/usr/bin/env python3
"""Finalize S19-L01 after its mandatory exact-replay stop.

This is a reporting-only finalizer.  It performs no scientific calculation and
does not recover, inspect, or serialize the invalidated in-memory outcomes.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L01"
BASELINE_PATH = ARTIFACT_ROOT / "s18_immutable_baseline.json"
REPORTING_AMENDMENT = REPO_ROOT / "configs/e01/s19_l01_reporting_amendment_002.json"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prior() -> dict[str, Any]:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    mismatches = []
    aggregate = hashlib.sha256()
    for row in baseline["files"]:
        path = Path(row["path"])
        if not path.exists():
            mismatches.append({"path": str(path), "reason": "missing"})
            continue
        current = {
            "path": row["path"],
            "role": row["role"],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        aggregate.update(canonical_json(current).encode())
        aggregate.update(b"\n")
        if current["bytes"] != row["bytes"] or current["sha256"] != row["sha256"]:
            mismatches.append(
                {
                    "path": str(path),
                    "reason": "size_or_hash_changed",
                    "expectedSha256": row["sha256"],
                    "actualSha256": current["sha256"],
                }
            )
    passed = len(mismatches) == 0 and aggregate.hexdigest() == baseline["aggregateSha256"]
    return {
        "fileCount": len(baseline["files"]),
        "expectedAggregateSha256": baseline["aggregateSha256"],
        "actualAggregateSha256": aggregate.hexdigest(),
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "passed": passed,
    }


def empty_frame(columns: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns.items()})


def artifact_manifest(root: Path, required: list[str]) -> dict[str, Any]:
    missing = [name for name in required if not (root / name).is_file()]
    files = []
    own_manifest = root / "artifact_manifest.json"
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != own_manifest):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": "eidosoma.e01.s19_artifact_manifest.v1",
        "root": str(root),
        "status": "LOOP_FAILED_CLOSED",
        "fileCount": len(files),
        "totalBytes": int(sum(row["bytes"] for row in files)),
        "files": files,
        "requiredFiles": required,
        "missing": missing,
        "passed": not missing,
    }


def main() -> None:
    now = datetime.now(timezone.utc)
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    prior = validate_prior()
    if not prior["passed"]:
        raise RuntimeError("immutable S01-S18 baseline changed")
    if not REPORTING_AMENDMENT.is_file():
        raise RuntimeError("reporting-only amendment record is missing")
    shutil.copyfile(REPORTING_AMENDMENT, LOOP_ROOT / "reporting_amendment_002.json")

    execution = pd.DataFrame(
        [
            {
                "attemptId": "S19-L01-ATTEMPT-001",
                "bundleId": "A_METRIC_DISTINCTIVENESS",
                "stage": "BUNDLE_A_THEN_B_INPUT_LOAD",
                "status": "ABORTED_BEFORE_B_OUTCOMES",
                "failureId": "F001",
                "scientificOutcomesSerialized": False,
                "reason": "Frozen-H replay helper used H[0]=1 rather than the locked duplicate-first adjacent convention.",
                "approximateObservedWallSeconds": 6748.0,
                "eligibleForClaimInference": False,
            },
            {
                "attemptId": "S19-L01-ATTEMPT-002",
                "bundleId": "A_METRIC_DISTINCTIVENESS",
                "stage": "RESTARTED_BUNDLE_A",
                "status": "COMPUTED_IN_MEMORY_NOT_SERIALIZED",
                "failureId": "F002",
                "scientificOutcomesSerialized": False,
                "reason": "Whole-loop exact-replay gate later failed in Bundle B; partial results were invalidated prospectively.",
                "approximateObservedWallSeconds": 9185.0,
                "eligibleForClaimInference": False,
            },
            {
                "attemptId": "S19-L01-ATTEMPT-002",
                "bundleId": "B_ALTERNATIVE_PREDICTION_PROPORTIONS",
                "stage": "ALL_PROPORTIONS_THEN_25_75_REPLAY_GATE",
                "status": "EXACT_REPLAY_FAILED_CLOSED",
                "failureId": "F002",
                "scientificOutcomesSerialized": False,
                "reason": "At least one 25/75 split metric did not exactly equal frozen S16; likely locked cutoff-source seed transcription mismatch, not rerun after stop.",
                "approximateObservedWallSeconds": 9185.0,
                "eligibleForClaimInference": False,
            },
            {
                "attemptId": "S19-L01-ATTEMPT-002",
                "bundleId": "C_SPIKE_TIMING_SPACING_HEIGHT",
                "stage": "NOT_STARTED",
                "status": "NOT_EXECUTED_DUE_GLOBAL_STOP",
                "failureId": "F002",
                "scientificOutcomesSerialized": False,
                "reason": "Mandatory exact-replay stop occurred before Bundle C.",
                "approximateObservedWallSeconds": 0.0,
                "eligibleForClaimInference": False,
            },
        ]
    )
    execution.to_parquet(LOOP_ROOT / "execution_status.parquet", index=False)

    specification_rows = []
    prereg = yaml.safe_load((LOOP_ROOT / "preregistration.yaml").read_text(encoding="utf-8"))
    for bundle in prereg["bundles"]:
        for order, specification in enumerate(bundle["specifications"], start=1):
            if bundle["bundleId"] == "A_METRIC_DISTINCTIVENESS":
                status = "ATTEMPTED_IN_MEMORY_INVALIDATED_BY_LOOP_REPLAY_FAILURE"
            elif bundle["bundleId"] == "B_ALTERNATIVE_PREDICTION_PROPORTIONS":
                status = "ATTEMPTED_EXACT_REPLAY_FAILED"
            else:
                status = "NOT_EXECUTED_GLOBAL_STOP"
            specification_rows.append(
                {
                    "bundleId": bundle["bundleId"],
                    "specificationOrder": order,
                    "specificationId": specification,
                    "registeredBeforeOutcome": True,
                    "executionStatus": status,
                    "eligibleForInference": False,
                    "postOutcomeScientificChange": False,
                }
            )
    pd.DataFrame(specification_rows).to_parquet(LOOP_ROOT / "specification_ledger.parquet", index=False)

    result_schema = {
        "bundleId": "string",
        "resultType": "string",
        "claimId": "string",
        "candidateId": "string",
        "specificationId": "string",
        "analysisId": "string",
        "estimate": "float64",
        "pValue": "float64",
        "status": "string",
        "detailsJson": "string",
    }
    empty_frame(result_schema).to_parquet(LOOP_ROOT / "results.parquet", index=False)
    negative = pd.DataFrame(
        [
            {
                "controlId": "FROZEN_H_REPLAY_ATTEMPT_001",
                "attemptId": "S19-L01-ATTEMPT-001",
                "passed": False,
                "status": "VALUE_PRESERVING_AMENDMENT_APPLIED",
                "reason": "Duplicate-first H convention mismatch; amendment S19-L01-VPA-001 restored exact frozen input before restart.",
            },
            {
                "controlId": "S16_25_75_EXACT_REPLAY_ATTEMPT_002",
                "attemptId": "S19-L01-ATTEMPT-002",
                "passed": False,
                "status": "GLOBAL_STOP_TRIGGERED",
                "reason": "One or more reconstructed split metrics differed from frozen S16; no scientific repair or rerun permitted.",
            },
            {
                "controlId": "IMMUTABLE_S01_S18_POSTCHECK",
                "attemptId": "S19-L01-FINALIZATION",
                "passed": True,
                "status": "PASS",
                "reason": None,
            },
        ]
    )
    negative.to_parquet(LOOP_ROOT / "negative_control_results.parquet", index=False)
    empty_frame(
        {
            "bundleId": "string",
            "robustnessFamily": "string",
            "candidateId": "string",
            "specificationId": "string",
            "status": "string",
            "reason": "string",
        }
    ).to_parquet(LOOP_ROOT / "robustness_results.parquet", index=False)

    failures = pd.DataFrame(
        [
            {
                "failureId": "F001",
                "attemptId": "S19-L01-ATTEMPT-001",
                "bundleId": "B_ALTERNATIVE_PREDICTION_PROPORTIONS",
                "stage": "frozen_input_replay",
                "status": "ABORTED",
                "reason": "Exact H helper failed at selected index 0 because it did not duplicate the first adjacent H value.",
                "resolution": "S19-L01-VPA-001 restored the already-locked frozen S13Y/S16 input; all 200 H vectors then replayed exactly before restart.",
                "scientificChange": False,
                "outcomeExclusion": True,
            },
            {
                "failureId": "F002",
                "attemptId": "S19-L01-ATTEMPT-002",
                "bundleId": "B_ALTERNATIVE_PREDICTION_PROPORTIONS",
                "stage": "S16_25_75_exact_replay",
                "status": "GLOBAL_STOP_LOOP_FAILED_CLOSED",
                "reason": "At least one 25/75 split metric was not bit-identical to frozen S16.",
                "resolution": "No repair attempted after the global exact-replay stop. Static audit identifies a likely different cutoff-source seed domain/width, but this remains unverified without prohibited rerun.",
                "scientificChange": False,
                "outcomeExclusion": True,
            },
        ]
    )
    failures.to_csv(LOOP_ROOT / "failure_ledger.csv", index=False)

    # Required bundle-specific artifacts are explicit empty/not-eligible tables,
    # not absent files or fabricated results.
    empty_frame(
        {
            "claimId": "string",
            "candidateId": "string",
            "specificationId": "string",
            "metricId": "string",
            "status": "string",
            "reason": "string",
        }
    ).to_parquet(LOOP_ROOT / "metric_distinctiveness_results.parquet", index=False)
    empty_frame(
        {
            "candidateId": "string",
            "matrixIndex": "int64",
            "specificationId": "string",
            "metricId": "string",
            "value": "float64",
            "status": "string",
        }
    ).to_parquet(LOOP_ROOT / "network_feature_results.parquet", index=False)
    empty_frame(
        {
            "candidateId": "string",
            "matrixIndex": "int64",
            "specificationId": "string",
            "metricId": "string",
            "value": "float64",
            "status": "string",
        }
    ).to_parquet(LOOP_ROOT / "dynamical_feature_results.parquet", index=False)
    empty_frame(
        {
            "candidateId": "string",
            "proportion": "float64",
            "modeId": "string",
            "featureId": "string",
            "repetitionId": "int64",
            "accuracy": "float64",
            "status": "string",
        }
    ).to_parquet(LOOP_ROOT / "prediction_proportion_results.parquet", index=False)
    empty_frame(
        {
            "candidateId": "string",
            "matrixIndex": "int64",
            "temporalSourceMode": "string",
            "specificationId": "string",
            "spikeCount": "int64",
            "status": "string",
        }
    ).to_parquet(LOOP_ROOT / "spike_descriptor_results.parquet", index=False)
    empty_frame(
        {
            "claimId": "string",
            "candidateId": "string",
            "specificationId": "string",
            "statistic": "float64",
            "pValue": "float64",
            "status": "string",
        }
    ).to_csv(LOOP_ROOT / "spike_correlation_results.csv", index=False)

    claim_ids = [*(f"C{value:03d}" for value in range(1, 13)), "C029", "C031", "C032", "C033"]
    overlay = pd.DataFrame(
        [
            {
                "claimId": claim,
                "originalS18Status": "NOT_EVALUATED",
                "s19ExploratoryStatus": "LOOP_FAILED_CLOSED",
                "promotableToS20": False,
                "candidate2Status": "NOT_ADJUDICATED",
                "candidate3Status": "NOT_ADJUDICATED",
                "exactOrDirectional": "NOT_ADJUDICATED",
                "completedFitDependent": None,
                "exactHDependent": None,
                "evidencePath": "loops/L01/failure_ledger.csv",
                "rationale": "The loop-wide exact S16 replay gate failed before an eligible artifact set was frozen; no in-memory result is used.",
            }
            for claim in claim_ids
        ]
    )
    overlay.to_csv(LOOP_ROOT / "claim_status_overlay_C001_C033.csv", index=False)

    runtime = {
        "schema": "eidosoma.e01.s19_l01_runtime_manifest.v1",
        "loopId": "S19-L01",
        "status": "LOOP_FAILED_CLOSED",
        "attempts": [
            {"attemptId": "S19-L01-ATTEMPT-001", "approximateObservedWallSeconds": 6748.0},
            {"attemptId": "S19-L01-ATTEMPT-002", "approximateObservedWallSeconds": 9185.0},
        ],
        "scientificCpuHours": "NOT_RECOVERABLE_AFTER_ABORT; below 100-hour ceiling by process design",
        "workers": 8,
        "threadsPerWorker": 1,
        "cpuFloat64Authoritative": True,
        "gpuHours": 0.0,
        "newGardTrajectories": 0,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "nolds": "0.6.1",
        },
        "initialLockCommit": "3950b84060b4fc45f6108126f67c2973625c78c0",
        "amendedLockCommit": "ebeeb528cb8b3c95804462635c730b305485a10e",
        "reportingCommit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
        ).stdout.strip(),
        "completedUtc": now.isoformat(),
    }
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime)
    storage = {
        "schema": "eidosoma.e01.s19_l01_storage_validation.v1",
        "retainedBytes": sum(path.stat().st_size for path in ARTIFACT_ROOT.rglob("*") if path.is_file()),
        "retainedCeilingBytes": 25 * 1024**3,
        "temporaryCeilingBytes": 75 * 1024**3,
        "passed": True,
    }
    storage["passed"] = storage["retainedBytes"] <= storage["retainedCeilingBytes"]
    write_json(LOOP_ROOT / "storage_validation.json", storage)
    regeneration = {
        "schema": "eidosoma.e01.s19_l01_regeneration_validation.v1",
        "loopId": "S19-L01",
        "validationResult": "FAIL_CLOSED_S16_25_75_EXACT_REPLAY",
        "passed": False,
        "immutablePriorPassed": prior["passed"],
        "frozenHAmendmentReplayPassed": True,
        "s16Exact25ReplayPassed": False,
        "scientificResultsEligible": False,
        "requiredArtifactsFinalizedAsFailureStatuses": True,
        "nextScientificWorkAuthorized": False,
    }
    write_json(LOOP_ROOT / "regeneration_validation.json", regeneration)
    classification = {
        "schema": "eidosoma.e01.s19_l01_classification.v1",
        "loopId": "S19-L01",
        "topLevelClassification": "LOOP_FAILED_CLOSED",
        "confirmatoryVerdictIssued": False,
        "claimCounts": {"LOOP_FAILED_CLOSED": 16},
        "promotableLeadIds": [],
        "historicalS18TotalsUnchanged": {
            "SUPPORTED": 3,
            "DIRECTIONALLY_SUPPORTED": 17,
            "NOT_SUPPORTED_WITHIN_TESTED_SCOPE": 21,
            "UNDERDETERMINED": 2,
            "NOT_EVALUATED": 16
        },
        "humanProposedNextLoopTheme": "REPLICATOR_DEFINITION_AND_TEMPORAL_FINGERPRINT_88_VS_98",
        "requiredHumanReview": True,
        "laterLoopAuthorized": False,
        "s20Active": False,
    }
    write_json(LOOP_ROOT / "classification.json", classification)

    report = f"""# E01/S19 Loop 1 — Unevaluated-claim recovery (failed closed)

## Concise top summary

- **Research step ID:** S19-L01
- **Completion status:** `LOOP_FAILED_CLOSED`; mandatory human-review boundary active
- **Artifacts written:** every required S19 root/L01 status, ledger, empty-not-eligible result table, additive claim overlay, validation, runtime, and artifact manifest; no invalidated scientific value is reported
- **Validation result:** `FAIL_CLOSED_S16_25_75_EXACT_REPLAY`; immutable S01–S18/V1/V2 baseline passed across {prior['fileCount']} files
- **Outcome classification:** `LOOP_FAILED_CLOSED` for all sixteen additive claims; zero S20 promotions
- **Caveats or blockers:** Bundle A and B computations were invalidated before serialization; Bundle C never started; likely cutoff-source seed transcription issue remains unverified because replay failure is a global stop
- **Recommended next action:** human review should consider `CONTINUE_S19` with the narrow replicator-definition/temporal-fingerprint 88%-versus-98% theme; no next loop is active

## Lay summary

Loop 1 did not produce an eligible scientific result. The first attempt stopped because a helper did not reproduce the frozen first H value. A separately recorded amendment restored exact H across all 200 trajectories without changing the declared method. The restarted attempt then failed the decisive check that its 25/75 prediction results exactly reproduce S16. The protocol requires stopping at that point. Rather than expose unvalidated in-memory numbers or patch toward a preferred result, this report preserves the failures and returns control.

The strongest next scientific question is independent of this failure: why the frozen adjacent-similarity label yields about 98% occupancy when the paper reports about 88% and a much later onset. The proposed next loop would reconstruct the replicator state and its full temporal fingerprint, not tune a threshold to match occupancy.

## Frozen question and intended scope

L01 was prospectively locked to evaluate C001–C012, C029, and C031–C033 through three bundles: same-lineage network/dynamical metrics; five prediction proportions under the exact S16 model contract; and two bounded spike definitions. Both simulator candidates were mandatory, pooling secondary, with no new GARD trajectories.

## Inputs

- Frozen S13Y trajectories, completed-fit and past-only PhiRL values, and `Y=I(H>0.9)` labels.
- Frozen S16 matrix splits, seeds, tensor/model rules, and 25/75 results as the replay oracle.
- Original paper, S18 claim matrix, and public same-author source lineages.
- The immutable baseline validated {prior['fileCount']} files ({json.loads(BASELINE_PATH.read_text())['totalBytes']} bytes) with aggregate SHA-256 `{prior['actualAggregateSha256']}`.

## Methods and commands

The preregistration, method lock, candidate ranking, seed manifest, input manifest, and source snapshot were committed and pushed before claim-level outcome access. The locked scientific command was:

```text
PYTHONPATH=src:/cache/e01_s19_l01/python_deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s19_l01.py --workers 8
```

Focused pre-outcome and amendment validation used 13 passing S16/S19 tests. The amendment audit independently replayed all 200 H vectors with maximum absolute error exactly zero and verified `Y=I(H>0.9)` with zero failures.

## Execution chronology

### Attempt 1

Bundle A ran in memory. Bundle B stopped while loading frozen inputs because its independent H helper set the initial value to 1 rather than duplicating the first adjacent similarity. No Bundle B outcome was calculated. Amendment `S19-L01-VPA-001` changed no declared method: it restored the exact S13Y/S16 convention and required use of serialized frozen H.

### Attempt 2

The amended commit was tested, pushed, and matched the clean remote before restart. Bundle A and the five-proportion Bundle B calculation ran in memory. The 25/75 comparison then found at least one metric unequal to frozen S16 and triggered the global exact-replay stop. No result table had been serialized or inspected, and Bundle C had not begun.

A static post-stop audit identifies a likely implementation cause: the new cutoff-source task appears to use a different seed domain token and width than S16's frozen `cutoff_source` 32-bit seed helper. This is a diagnosis, not a verified repair. Testing or repairing it would require another post-failure scientific run and is therefore deferred to human review.

## Results

There are no eligible L01 scientific estimates. Required result tables are present with explicit empty schemas and status ledgers; they are not missing and do not contain fabricated values. The additive overlay retains S18's original `NOT_EVALUATED` field and marks all sixteen S19 entries `LOOP_FAILED_CLOSED`.

| Additive claim family | Claims | L01 status | S20 promotion |
|---|---:|---|---|
| Network/dynamical distinctiveness | C001–C012 | LOOP_FAILED_CLOSED | No |
| Alternative prediction proportions | C029 | LOOP_FAILED_CLOSED | No |
| Spike timing/spacing/height | C031–C033 | LOOP_FAILED_CLOSED | No |

No retrospective, prospective, causal, exact, or directional paper match is inferred from the invalidated computations.

## Validation

- Immutable prior: PASS ({prior['fileCount']} files; zero mismatches).
- Pre-outcome clean pushed lock: PASS at commit `3950b84060b4fc45f6108126f67c2973625c78c0`.
- Value-preserving H amendment: PASS at commit `ebeeb528cb8b3c95804462635c730b305485a10e`.
- Exact 25/75 S16 replay: **FAIL**.
- Loop-level scientific eligibility: **FAIL CLOSED**.
- New GARD trajectories: zero.
- S18 artifacts/status totals: unchanged.

## Self-improvement analysis

- **Belief before:** public lineages could resolve metric and spike ambiguities while frozen S16 could anchor proportion sensitivity.
- **What the loop attempted:** complete sixteen unevaluated claims without method search.
- **What was learned:** the execution harness itself was not yet a trustworthy extension of S16. Exact-H identity was repaired, but prediction replay still failed. No claim result survives that failure.
- **Hypotheses weakened:** confidence that the new L01 prediction extension exactly instantiates S16.
- **What remains plausible:** the paper's replicator state may differ structurally from adjacent `H>0.9`; this is motivated by existing S18 evidence, not by invalid L01 values.
- **Why a next loop could add information:** a label-focused loop attacks one upstream dependency shared by Figures 3–6 and Table 1, using fingerprints independent of emergence. It is not an opportunity to add thresholds until a positive result appears.

## Proposed inactive next-loop theme: replicator definition and temporal fingerprint

The human-proposed next loop should compare a small, source-grounded family:

1. adjacent `H>0.9` as the frozen comparator;
2. dominant recurring-composition/centroid membership;
3. recurring Euclidean composition-cluster membership as described in the paper;
4. historical source-traceable GARD compotype/non-drift machinery.

Labels should be ranked without emergence using the joint fingerprint: occupancy near 88%, persistence, time to first onset, consistency, entry/exit counts, episode duration/structure, actual cluster recurrence, and candidate-2/candidate-3 agreement. The already observed `H>0.97` occupancy resemblance is not an eligible solution because it was outcome-guided and did not reproduce onset or consistency. Any chosen fixed definition would require untouched S20 confirmation. This theme is proposed only; it is not authorized or executed.

## Caveats and provenance

The two attempts consumed several hours but no authoritative child-process CPU total survived abort. Both remained structurally below the 100-CPU-hour ceiling, used eight one-thread CPU workers, no GPU, and no trajectory generation. Failure messages, commits, source identities, hashes, seeds, failure statuses, and the reporting-only finalizer are retained. Public source without a compatible license remains cache-only.

## Mandatory human-review boundary

Choose exactly one: `CONTINUE_S19`, `ACTIVATE_S20_CONFIRMATION`, `ACTIVATE_S20_CLOSEOUT_ONLY`, or `PAUSE_PROGRAM`. Given the zero eligible L01 leads and the human's stated priority, the scientific recommendation is `CONTINUE_S19` only if the reviewer explicitly authorizes the narrow replicator-definition/temporal-fingerprint loop and a fresh compute ceiling. No L02, S20, E02, or report bundle has started.
"""
    decision = f"""# S19-L01 one-page decision summary

## Concise top summary

- **Research step ID:** S19-L01
- **Completion status:** `LOOP_FAILED_CLOSED`; human review required
- **Artifacts written:** complete failure/status package and sixteen-claim additive overlay; no invalid result values retained
- **Validation result:** immutable prior PASS; H amendment replay PASS; exact S16 25/75 replay FAIL
- **Outcome classification:** sixteen × `LOOP_FAILED_CLOSED`; zero promotions
- **Caveats or blockers:** new prediction extension is not yet exact S16; C bundle not run
- **Recommended next action:** consider `CONTINUE_S19` with the narrow 88%-versus-98% replicator-definition fingerprint theme

## Decision

L01 cannot adjudicate C001–C012, C029, or C031–C033. Its second attempt triggered the preregistered exact-replay global stop. S18 remains unchanged.

## Why the proposed pivot is high leverage

The current label is exactly `Y=I(H>0.9)`, yields near-universal occupancy and very early onset, and can encode smooth drift rather than recurrence. The paper describes recurring composition-space clusters. That upstream mismatch affects Figures 3–6, Table 1, and the meaning of “before a replicator.” A next loop should compare a small source-grounded label family on occupancy **and temporal/recurrence fingerprints**, never tune a threshold or use emergence to select the label.

## Required human choice

Select exactly one of `CONTINUE_S19`, `ACTIVATE_S20_CONFIRMATION`, `ACTIVATE_S20_CLOSEOUT_ONLY`, or `PAUSE_PROGRAM`. No option has been activated.
"""
    (LOOP_ROOT / "S19_L01_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "loop_decision_summary.md").write_text(decision, encoding="utf-8")
    (ARTIFACT_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")

    ledger = pd.read_parquet(ARTIFACT_ROOT / "self_improvement_ledger.parquet")
    if not ledger["recordPhase"].eq("POST_LOOP_FAILED_CLOSED").any():
        entry = pd.DataFrame(
            [
                {
                    "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
                    "timestampUtc": now.isoformat(),
                    "loopId": "S19-L01",
                    "recordPhase": "POST_LOOP_FAILED_CLOSED",
                    "beliefBeforeLoop": ledger.iloc[0]["beliefBeforeLoop"],
                    "motivatingEvidence": ledger.iloc[0]["motivatingEvidence"],
                    "failureOrAmbiguityTargeted": ledger.iloc[0]["failureOrAmbiguityTargeted"],
                    "selectedHypotheses": ledger.iloc[0]["selectedHypotheses"],
                    "learned": "The L01 extension failed exact S16 replay and produced no eligible claim result. Exact H duplication was restored, but a likely cutoff-source seed mismatch remains unverified.",
                    "weakenedHypotheses": "The belief that the new multi-proportion runner exactly instantiates frozen S16.",
                    "remainingPlausibleHypotheses": "A structurally different recurring-composition replicator label could explain the 88% versus 98% and temporal-fingerprint discrepancy.",
                    "proposedNextTest": "Human-authorized narrow label-definition and temporal-fingerprint loop; do not use emergence for label selection.",
                    "informationGainRationale": "The label is upstream of Figures 3–6 and Table 1, and joint temporal/recurrence fingerprints distinguish a true attractor definition from occupancy matching.",
                    "appendOnly": True,
                }
            ]
        )
        pd.concat([ledger, entry], ignore_index=True).to_parquet(
            ARTIFACT_ROOT / "self_improvement_ledger.parquet", index=False
        )
        with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open("a", encoding="utf-8") as handle:
            handle.write(
                """

## Entry 002 — S19-L01 failed closed and returned for review

- **What was learned:** L01 produced no eligible result because the exact S16 25/75 replay gate failed. The earlier H[0] issue was restored byte-exactly, but a likely cutoff-source seed transcription mismatch remains unverified.
- **Hypotheses weakened:** the belief that the new proportion runner exactly instantiates S16.
- **Hypotheses remaining plausible:** the frozen adjacent-similarity label may not represent the paper's recurring compositional attractor state.
- **What should be tested next:** only after human authorization, a narrow 88%-versus-98% replicator-definition and temporal-fingerprint loop.
- **Why it adds information:** the label is upstream of Figures 3–6 and Table 1, and the proposed fingerprints distinguish recurrence from smooth drift without using emergence to choose a favorable label.
"""
            )

    history = json.loads((ARTIFACT_ROOT / "human_review_history.json").read_text(encoding="utf-8"))
    history_records = [
            {
                "date": now.date().isoformat(),
                "decision": "S19_L01_VALUE_PRESERVING_AMENDMENT_001",
                "scope": "restore exact frozen H input",
                "source": "execution_failure_and_locked_S16_replay",
            },
            {
                "date": now.date().isoformat(),
                "decision": "S19_L01_LOOP_FAILED_CLOSED",
                "scope": "exact S16 25/75 replay failure",
                "source": "preregistered_global_stop",
            },
            {
                "date": now.date().isoformat(),
                "decision": "HUMAN_PROPOSED_NEXT_THEME",
                "scope": "replicator definition and temporal fingerprint 88 versus 98",
                "source": "explicit_human_direction; inactive until review decision",
            },
        ]
    prior_decisions = {row["decision"] for row in history["history"]}
    history["history"].extend(row for row in history_records if row["decision"] not in prior_decisions)
    history["pendingDecision"] = "HUMAN_REVIEW_REQUIRED"
    write_json(ARTIFACT_ROOT / "human_review_history.json", history)
    loop_registry = yaml.safe_load((ARTIFACT_ROOT / "loop_registry.yaml").read_text(encoding="utf-8"))
    loop_registry["loops"][0].update(
        {
            "status": "LOOP_FAILED_CLOSED_AWAITING_HUMAN_REVIEW",
            "outcomeAccessed": True,
            "completed": True,
            "eligibleScientificResults": False,
            "failureId": "F002",
        }
    )
    loop_registry["proposedNextLoopTheme"] = "REPLICATOR_DEFINITION_AND_TEMPORAL_FINGERPRINT_88_VS_98"
    loop_registry["proposedNextLoopActive"] = False
    (ARTIFACT_ROOT / "loop_registry.yaml").write_text(
        yaml.safe_dump(loop_registry, sort_keys=False), encoding="utf-8"
    )
    status_artifacts = {
        str(path)
        for directory in (ARTIFACT_ROOT, LOOP_ROOT)
        for path in directory.iterdir()
        if path.is_file()
    }
    status_artifacts.update(
        {
            str(ARTIFACT_ROOT / "artifact_manifest.json"),
            str(LOOP_ROOT / "artifact_manifest.json"),
        }
    )
    write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "researchStepId": "S19-L01",
            "stepNumber": 19,
            "success": False,
            "status": "LOOP_FAILED_CLOSED_AWAITING_MANDATORY_HUMAN_REVIEW",
            "artifactsWritten": sorted(status_artifacts),
            "validationResult": "FAIL_CLOSED_S16_25_75_EXACT_REPLAY",
            "caveatsOrBlockers": [
                "no_eligible_L01_scientific_results",
                "S16_25_75_exact_replay_failed",
                "likely_cutoff_source_seed_transcription_unverified",
                "Bundle_C_not_started",
            ],
            "recommendedNextAction": "human_review_consider_CONTINUE_S19_with_inactive_88_vs_98_label_fingerprint_theme",
        },
    )

    loop_required = [
        "preregistration.yaml", "method_lock.json", "candidate_ranking.csv",
        "candidate_bundle_registry.yaml", "seed_manifest.parquet", "input_manifest.json",
        "source_snapshot_manifest.json", "execution_status.parquet", "specification_ledger.parquet",
        "results.parquet", "negative_control_results.parquet", "robustness_results.parquet",
        "failure_ledger.csv", "runtime_manifest.json", "storage_validation.json",
        "regeneration_validation.json", "classification.json", "loop_decision_summary.md",
        "S19_L01_FULL_RESULTS.md", "artifact_manifest.json", "metric_distinctiveness_results.parquet",
        "network_feature_results.parquet", "dynamical_feature_results.parquet",
        "prediction_proportion_results.parquet", "spike_descriptor_results.parquet",
        "spike_correlation_results.csv", "claim_status_overlay_C001_C033.csv",
        "value_preserving_amendment_001.json", "reporting_amendment_002.json"
    ]
    # The manifest cannot require itself until after its first serialization.
    write_json(LOOP_ROOT / "artifact_manifest.json", artifact_manifest(LOOP_ROOT, [x for x in loop_required if x != "artifact_manifest.json"]))
    loop_manifest = artifact_manifest(LOOP_ROOT, loop_required)
    write_json(LOOP_ROOT / "artifact_manifest.json", loop_manifest)
    root_required = [
        "continuation_decision.md", "s18_immutable_baseline.json", "self_improvement_ledger.parquet",
        "SELF_IMPROVEMENT_LEDGER.md", "candidate_registry.parquet", "source_search_ledger.parquet",
        "source_search_report.md", "loop_registry.yaml", "human_review_history.json",
        "s19_status.json", "research_step_full_results.md", "artifact_manifest.json"
    ]
    write_json(
        ARTIFACT_ROOT / "artifact_manifest.json",
        artifact_manifest(ARTIFACT_ROOT, [x for x in root_required if x != "artifact_manifest.json"]),
    )
    write_json(ARTIFACT_ROOT / "artifact_manifest.json", artifact_manifest(ARTIFACT_ROOT, root_required))
    print(
        canonical_json(
            {
                "success": False,
                "status": "LOOP_FAILED_CLOSED_AWAITING_MANDATORY_HUMAN_REVIEW",
                "validationResult": "FAIL_CLOSED_S16_25_75_EXACT_REPLAY",
                "claimCounts": {"LOOP_FAILED_CLOSED": 16},
                "promotableLeadIds": [],
                "immutablePriorPassed": True,
            }
        )
    )


if __name__ == "__main__":
    main()
