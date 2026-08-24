#!/usr/bin/env python3
"""Reporting-only L06 finalizer after the mandatory exact-replay stop.

This script performs no label, fingerprint, bootstrap, permutation, emergence,
prediction, intervention, or trajectory calculation. It does not read or
serialize invalidated worker result caches. It writes the complete failure
handoff and explicit empty/not-eligible scientific tables required by L06.
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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
S19_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = S19_ROOT / "loops/L06"
S13Y_MANIFEST = Path("/artifacts/research_steps/S13Y/trajectory_manifest.parquet")
AMENDMENT = REPO_ROOT / "configs/e01/s19_l06_reporting_amendment_002.json"
LOOP_ID = "S19-L06"
VERSION = "E01-S19-L06-PAST-ONLY-MULTIATTRACTOR-BOUNDARY-RECURRENCE-v1.0.0"
VALIDATION_RESULT = "FAIL_CLOSED_INDEPENDENT_BOUNDARY_SCORE_EXACT_REPLAY"
FAILURE_TRAJECTORY = (
    "E01-S12F-S13Y_CLEAN_DIRECTIONAL_CONFIRMATION-S12F-CANDIDATE-02-M000"
)
SCORE_COUNT = 713
SCORE_MISMATCH_COUNT = 401
MAX_ABS_SCORE_DIFFERENCE = 3.3306690738754696e-16


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


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def empty_frame(columns: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns.items()})


def validate_immutable_prior() -> dict[str, Any]:
    baseline_path = LOOP_ROOT / "immutable_prior_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    aggregate = hashlib.sha256()
    mismatches: list[dict[str, object]] = []
    for expected in baseline["files"]:
        path = Path(expected["path"])
        if not path.is_file():
            mismatches.append({"path": str(path), "reason": "missing"})
            continue
        current = {
            "path": expected["path"],
            "role": expected["role"],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        aggregate.update(canonical_json(current).encode("utf-8"))
        aggregate.update(b"\n")
        if current["bytes"] != expected["bytes"] or current["sha256"] != expected["sha256"]:
            mismatches.append(
                {
                    "path": str(path),
                    "reason": "size_or_hash_changed",
                    "expectedSha256": expected["sha256"],
                    "actualSha256": current["sha256"],
                }
            )
    actual = aggregate.hexdigest()
    passed = not mismatches and actual == baseline["aggregateSha256"]
    return {
        "schema": "eidosoma.e01.s19_l06_immutable_prior_postcheck.v1",
        "fileCount": len(baseline["files"]),
        "expectedAggregateSha256": baseline["aggregateSha256"],
        "actualAggregateSha256": actual,
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "passed": passed,
    }


def manifest(root: Path, required: list[str], schema: str) -> dict[str, Any]:
    own = root / "artifact_manifest.json"
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != own):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    missing = [name for name in required if not (root / name).is_file()]
    return {
        "schema": schema,
        "root": str(root),
        "status": "LOOP_FAILED_CLOSED",
        "fileCount": len(files),
        "totalBytes": int(sum(row["bytes"] for row in files)),
        "files": files,
        "requiredFiles": required,
        "missing": missing,
        "passed": not missing,
    }


def write_empty_scientific_tables() -> list[str]:
    schemas: dict[str, dict[str, str]] = {
        "label_values.parquet": {
            "candidateId": "string", "matrixIndex": "int64", "trajectoryId": "string",
            "selectedClockIndex": "int64", "labelId": "string", "label": "boolean",
            "eligible": "boolean", "status": "string",
        },
        "boundary_projected_label_results.parquet": {
            "candidateId": "string", "matrixIndex": "int64", "trajectoryId": "string",
            "selectedClockIndex": "int64", "generation": "int64", "label": "boolean",
            "status": "string",
        },
        "boundary_activation_results.parquet": {
            "candidateId": "string", "matrixIndex": "int64", "trajectoryId": "string",
            "boundaryGeneration": "int64", "activated": "boolean", "score": "float64",
            "distinctPriorBoundaryCount": "int64", "status": "string",
        },
        "boundary_recurrence_trajectory_diagnostics.parquet": {
            "candidateId": "string", "matrixIndex": "int64", "trajectoryId": "string",
            "activatedBoundaryCount": "int64", "eligibleBoundaryCount": "int64",
            "status": "string",
        },
        "frozen_comparator_replay.parquet": {
            "candidateId": "string", "matrixIndex": "int64", "trajectoryId": "string",
            "passed": "boolean", "status": "string",
        },
        "future_suffix_invariance_results.parquet": {
            "candidateId": "string", "matrixIndex": "int64", "trajectoryId": "string",
            "endpointOrdinal": "int64", "variant": "string", "passed": "boolean",
            "status": "string",
        },
        "fingerprint_results.parquet": {
            "candidateId": "string", "matrixIndex": "int64", "trajectoryId": "string",
            "labelId": "string", "occupancy": "float64", "persistence": "float64",
            "consistency": "float64", "firstOnsetRawIndex0": "float64", "status": "string",
        },
        "results.parquet": {
            "candidateId": "string", "matrixIndex": "int64", "trajectoryId": "string",
            "labelId": "string", "occupancy": "float64", "persistence": "float64",
            "consistency": "float64", "firstOnsetRawIndex0": "float64", "status": "string",
        },
        "fingerprint_summary.parquet": {
            "candidateId": "string", "labelId": "string", "meanOccupancy": "float64",
            "meanPersistence": "float64", "meanConsistency": "float64",
            "meanFirstOnsetRawIndex0": "float64", "jointPaperDistance": "float64",
            "status": "string",
        },
        "episode_results.parquet": {
            "candidateId": "string", "matrixIndex": "int64", "trajectoryId": "string",
            "labelId": "string", "episodeIndex": "int64", "duration": "int64",
            "status": "string",
        },
        "cutoff_results.parquet": {
            "candidateId": "string", "matrixIndex": "int64", "trajectoryId": "string",
            "labelId": "string", "cutoffIndex": "int64", "positiveAtCutoff": "boolean",
            "noOnsetThroughCutoff": "boolean", "status": "string",
        },
        "boundary_recurrence_count_results.parquet": {
            "candidateId": "string", "matrixIndex": "int64", "trajectoryId": "string",
            "boundaryGeneration": "int64", "distinctPriorBoundaryCount": "int64",
            "status": "string",
        },
        "label_overlap_results.parquet": {
            "candidateId": "string", "labelA": "string", "labelB": "string",
            "jaccard": "float64", "adjustedRand": "float64", "status": "string",
        },
        "paper_distance_bootstrap.parquet": {
            "candidateId": "string", "onsetMode": "string", "replicate": "int64",
            "distanceDifference": "float64", "status": "string",
        },
        "bootstrap_metric_differences.parquet": {
            "candidateId": "string", "metric": "string", "replicate": "int64",
            "difference": "float64", "status": "string",
        },
        "leave_one_out_robustness.parquet": {
            "candidateId": "string", "omittedMatrixIndex": "int64", "onsetMode": "string",
            "distanceDifference": "float64", "status": "string",
        },
        "generation_block_permutation_results.parquet": {
            "candidateId": "string", "matrixIndex": "int64", "replicate": "int64",
            "onsetMode": "string", "jointPaperDistance": "float64", "status": "string",
        },
        "negative_control_results.parquet": {
            "controlId": "string", "candidateId": "string", "passed": "boolean",
            "status": "string", "reason": "string",
        },
        "robustness_results.parquet": {
            "candidateId": "string", "analysisId": "string", "passed": "boolean",
            "status": "string", "reason": "string",
        },
    }
    for name, schema in schemas.items():
        empty_frame(schema).to_parquet(LOOP_ROOT / name, index=False)

    csv_schemas: dict[str, dict[str, str]] = {
        "fingerprint_aggregate.csv": {
            "candidateId": "string", "labelId": "string", "metric": "string",
            "estimate": "float64", "status": "string",
        },
        "cross_candidate_agreement.csv": {
            "labelId": "string", "metric": "string", "candidate2": "float64",
            "candidate3": "float64", "difference": "float64", "status": "string",
        },
        "paper_fingerprint_comparison.csv": {
            "candidateId": "string", "labelId": "string", "onsetMode": "string",
            "jointPaperDistance": "float64", "status": "string",
        },
        "fixed_l03_l05_comparison.csv": {
            "candidateId": "string", "comparisonLoop": "string", "metric": "string",
            "fixedValue": "float64", "l06Value": "float64", "status": "string",
        },
    }
    for name, schema in csv_schemas.items():
        empty_frame(schema).to_csv(LOOP_ROOT / name, index=False)
    return [*schemas, *csv_schemas]


def append_ledgers(now: str, classification: dict[str, Any]) -> None:
    ledger_path = S19_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    existing = ledger.loc[
        ledger["loopId"].eq(LOOP_ID) & ledger["recordPhase"].eq("POST_LOOP_FAILED_CLOSED")
    ]
    if existing.empty:
        pre = ledger.loc[
            ledger["loopId"].eq(LOOP_ID) & ledger["recordPhase"].eq("PRE_LOOP_BELIEF_AND_SELECTION")
        ].iloc[-1]
        row = {
            "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
            "timestampUtc": now,
            "loopId": LOOP_ID,
            "recordPhase": "POST_LOOP_FAILED_CLOSED",
            "beliefBeforeLoop": pre["beliefBeforeLoop"],
            "motivatingEvidence": pre["motivatingEvidence"],
            "failureOrAmbiguityTargeted": pre["failureOrAmbiguityTargeted"],
            "selectedHypotheses": pre["selectedHypotheses"],
            "learned": (
                "The L06 boundary-label implementation produced identical boolean labels and "
                "recurrence counts under its independent replay on the first diagnosed trajectory, "
                "but 401 of 713 finite cosine scores differed bitwise by at most "
                "3.3306690738754696e-16. The mandatory exact-score replay gate therefore failed; "
                "all worker outcomes are ineligible and no temporal fingerprint was adjudicated."
            ),
            "weakenedHypotheses": (
                "The belief that the two independently ordered float64 cosine implementations "
                "would reproduce boundary scores bit-for-bit under the locked contract."
            ),
            "remainingPlausibleHypotheses": (
                "Boundary-level multi-attractor recurrence remains scientifically unadjudicated; "
                "the failure neither supports nor refutes its temporal fingerprint."
            ),
            "proposedNextTest": "Mandatory human review; no next scientific loop is active.",
            "informationGainRationale": (
                "A future authorization would need to decide whether to revise the replay contract "
                "prospectively or pursue a different bounded ambiguity; L06 itself cannot be repaired."
            ),
            "appendOnly": True,
        }
        pd.concat([ledger, pd.DataFrame([row])[ledger.columns]], ignore_index=True).to_parquet(
            ledger_path, index=False
        )
        with (S19_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open("a", encoding="utf-8") as handle:
            handle.write(
                "\n\n## S19-L06 post-loop — failed closed at exact score replay\n\n"
                "- **Belief before:** past-only multi-attractor recurrence at selected post-fission boundaries might suppress local drift while preserving recurring attractors.\n"
                "- **What was learned:** on the first diagnosed trajectory, the primary and independent implementations produced identical labels and recurrence counts, but 401/713 finite scores were not bit-identical; maximum absolute difference was `3.3306690738754696e-16`. The locked exact-score gate therefore stopped the loop before any eligible fingerprint.\n"
                "- **Hypothesis weakened:** bit-exact reproducibility of differently ordered float64 cosine calculations under the locked replay contract.\n"
                "- **Hypothesis still plausible:** the scientific boundary-recurrence rule remains unadjudicated; no label result survived validation.\n"
                "- **Next action:** mandatory human review. L06 cannot be repaired or rerun under this authorization, and no L07 or S20 action is active.\n"
                "- **Why another action could add information:** only a new prospectively authorized contract could decide how numerical replay should be defined without retroactively relaxing L06.\n"
            )

    registry_path = S19_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    for row in registry["loops"]:
        if row["loopId"] == LOOP_ID:
            row.update(
                {
                    "status": "LOOP_FAILED_CLOSED_AWAITING_HUMAN_REVIEW",
                    "outcomeAccessed": True,
                    "completed": True,
                    "eligibleScientificResults": False,
                    "failureId": "S19-L06-F002",
                    "promotedLeadCount": 0,
                }
            )
    registry["laterLoopsAuthorized"] = False
    registry["s20Status"] = "DEFINED_INACTIVE"
    registry["proposedNextLoopTheme"] = None
    registry["proposedNextLoopActive"] = False
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    history_path = S19_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    decision = "S19_L06_LOOP_FAILED_CLOSED_EXACT_BOUNDARY_SCORE_REPLAY"
    if decision not in {row["decision"] for row in history["history"]}:
        history["history"].append(
            {
                "date": now[:10],
                "decision": decision,
                "scope": VERSION,
                "source": "preregistered_global_exact_replay_stop",
            }
        )
    history["pendingDecision"] = "POST_S19_L06_MANDATORY_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if not AMENDMENT.is_file():
        raise RuntimeError("reporting-only amendment record missing")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if amendment["scientificMethodChanged"] or amendment["scientificOutcomeRecomputed"]:
        raise RuntimeError("reporting amendment is not value preserving")
    shutil.copyfile(AMENDMENT, LOOP_ROOT / "reporting_amendment_002.json")

    prior = validate_immutable_prior()
    write_json(LOOP_ROOT / "immutable_prior_postcheck.json", prior)
    if not prior["passed"]:
        raise RuntimeError("immutable S01-S18/V1/V2/S19-L01-L05 baseline changed")

    preanalysis = json.loads((LOOP_ROOT / "preanalysis_replay_validation.json").read_text())
    lock = json.loads((LOOP_ROOT / "execution_lock_validation.json").read_text())
    if not preanalysis["passed"] or not lock["passed"]:
        raise RuntimeError("pre-outcome replay or pushed lock no longer passes")

    scientific_files = write_empty_scientific_tables()

    manifest_rows = pd.read_parquet(S13Y_MANIFEST)[
        ["candidateId", "matrixIndex", "trajectoryId"]
    ].copy()
    execution = manifest_rows.assign(
        attemptId="S19-L06-ATTEMPT-002",
        status="INVALIDATED_GLOBAL_EXACT_REPLAY_FAILURE",
        workerCalculationCompleted=True,
        scientificOutcomesSerialized=False,
        eligibleForScientificInference=False,
        failureId="S19-L06-F002",
        reason=(
            "Independent boundary scores were not bit-exact on the first diagnosed trajectory; "
            "all worker outputs are invalidated and excluded."
        ),
    )
    execution.to_parquet(LOOP_ROOT / "execution_status.parquet", index=False)

    replay = pd.DataFrame(
        [
            {
                "candidateId": "S12F-CANDIDATE-02",
                "matrixIndex": 0,
                "trajectoryId": FAILURE_TRAJECTORY,
                "labelId": "PF_PAST_ONLY_MULTIATTRACTOR_BOUNDARY_RECURRENCE_H900",
                "exactTwoPassLabelReplayPassed": True,
                "exactTwoPassScoreReplayPassed": True,
                "exactTwoPassBoundaryEvidenceReplayPassed": True,
                "passed": True,
                "status": "VALIDATION_ONLY_FIRST_DIAGNOSED_TRAJECTORY",
            }
        ]
    )
    replay.to_parquet(LOOP_ROOT / "label_replay_evidence.parquet", index=False)
    independent = pd.DataFrame(
        [
            {
                "candidateId": "S12F-CANDIDATE-02",
                "matrixIndex": 0,
                "trajectoryId": FAILURE_TRAJECTORY,
                "labelId": "PF_PAST_ONLY_MULTIATTRACTOR_BOUNDARY_RECURRENCE_H900",
                "finiteScoreCount": SCORE_COUNT,
                "exactScoreMismatchCount": SCORE_MISMATCH_COUNT,
                "maxAbsScoreDifference": MAX_ABS_SCORE_DIFFERENCE,
                "exactScoresPassed": False,
                "exactLabelsPassed": True,
                "exactRecurrenceCountsPassed": True,
                "passed": False,
                "status": "MANDATORY_GLOBAL_STOP_TRIGGERED",
            }
        ]
    )
    independent.to_parquet(LOOP_ROOT / "independent_label_replay.parquet", index=False)

    write_json(
        LOOP_ROOT / "future_suffix_invariance_summary.json",
        {
            "schema": "eidosoma.e01.s19_l06_future_suffix_invariance_summary.v1",
            "status": "NOT_ELIGIBLE_AFTER_PRIOR_GLOBAL_STOP",
            "sentinelCount": 0,
            "passed": False,
            "reason": "Independent exact boundary-score replay failed before suffix evidence became eligible.",
        },
    )

    failures = pd.DataFrame(
        [
            {
                "failureId": "S19-L06-F001",
                "attemptId": "S19-L06-ATTEMPT-001",
                "phase": "module_import_before_execution_lock_validation",
                "status": "VALUE_PRESERVING_PRE_OUTCOME_AMENDMENT_APPLIED",
                "reason": "ModuleNotFoundError: No module named 'scripts'",
                "resolution": "VPA-001 added repository root to sys.path and preserved the launch failure; no outcome had been accessed.",
                "scientificChange": False,
                "scientificOutcomeEligible": False,
            },
            {
                "failureId": "S19-L06-F002",
                "attemptId": "S19-L06-ATTEMPT-002",
                "phase": "independent_boundary_score_exact_replay",
                "status": "GLOBAL_STOP_LOOP_FAILED_CLOSED",
                "reason": (
                    "Primary matrix-multiplication and independent scalar-dot cosine paths differed "
                    "bitwise for 401/713 finite scores on the first diagnosed trajectory; maximum "
                    "absolute difference 3.3306690738754696e-16. Labels and recurrence counts agreed."
                ),
                "resolution": "No tolerance, repair, scientific change, rerun, aggregation, or result serialization permitted after the locked stop.",
                "scientificChange": False,
                "scientificOutcomeEligible": False,
            },
        ]
    )
    failures.to_csv(LOOP_ROOT / "failure_ledger.csv", index=False)

    negative = pd.DataFrame(
        [
            {
                "controlId": "PREANALYSIS_FROZEN_INPUT_REPLAY",
                "candidateId": "BOTH_SEPARATE",
                "passed": True,
                "status": "PASS",
                "reason": "200 identities/caches/clocks/boundary identities/adjacent-H/frozen labels replayed exactly before outcomes.",
            },
            {
                "controlId": "INDEPENDENT_BOUNDARY_SCORE_EXACT_REPLAY",
                "candidateId": "S12F-CANDIDATE-02",
                "passed": False,
                "status": "GLOBAL_STOP_TRIGGERED",
                "reason": "401/713 finite scores differed bitwise on matrix M000; max absolute difference 3.3306690738754696e-16.",
            },
            {
                "controlId": "IMMUTABLE_PRIOR_POSTCHECK",
                "candidateId": "NOT_APPLICABLE",
                "passed": True,
                "status": "PASS",
                "reason": f"{prior['fileCount']} frozen files passed byte/hash replay.",
            },
        ]
    )
    negative.to_parquet(LOOP_ROOT / "negative_control_results.parquet", index=False)
    pd.DataFrame(
        [
            {
                "candidateId": "S12F-CANDIDATE-02",
                "analysisId": "INDEPENDENT_BOUNDARY_SCORE_REPLAY",
                "passed": False,
                "status": "LOOP_FAILED_CLOSED",
                "reason": "Bit-exact score replay failed before any robustness result became eligible.",
            },
            {
                "candidateId": "S12F-CANDIDATE-03",
                "analysisId": "INDEPENDENT_BOUNDARY_SCORE_REPLAY",
                "passed": False,
                "status": "NOT_ADJUDICATED_GLOBAL_STOP",
                "reason": "Global stop was triggered by the first diagnosed candidate-2 trajectory.",
            },
        ]
    ).to_parquet(LOOP_ROOT / "robustness_results.parquet", index=False)

    classification = {
        "schema": "eidosoma.e01.s19_l06_classification.v1",
        "loopId": LOOP_ID,
        "versionedLoopId": VERSION,
        "topLevelClassification": "LOOP_FAILED_CLOSED",
        "outcomeClass": "CONSTRAINING_OPERATIONAL_VALIDATION_FAILURE",
        "confirmatoryVerdictIssued": False,
        "scientificFingerprintAdjudicated": False,
        "scientificResultsEligible": False,
        "labelClassifications": [
            {
                "labelId": "MOL_ADJACENT_INCOMING_H900",
                "classifications": ["NOT_PROMOTABLE"],
                "reason": "Comparator-only; no new eligible L06 result serialized.",
            },
            {
                "labelId": "PF_PAST_ONLY_MULTIATTRACTOR_BOUNDARY_RECURRENCE_H900",
                "classifications": ["POSSIBLE_PIPELINE_ARTIFACT", "LOOP_FAILED_CLOSED", "NOT_PROMOTABLE"],
                "reason": "Mandatory independent exact-score replay failed before fingerprint adjudication.",
            },
        ],
        "promotedLeadCount": 0,
        "promotedLeadIds": [],
        "requiredHumanReview": True,
        "laterLoopAuthorized": False,
        "s20Active": False,
        "s18VerdictUnchanged": True,
    }
    write_json(LOOP_ROOT / "classification.json", classification)

    runtime = {
        "schema": "eidosoma.e01.s19_l06_runtime_manifest.v1",
        "loopId": LOOP_ID,
        "status": "LOOP_FAILED_CLOSED",
        "workers": 8,
        "threadsPerWorker": 1,
        "cpuFloat64Authoritative": True,
        "scientificCpuHours": None,
        "scientificCpuHoursStatus": "NOT_RETAINED_AFTER_PRE_AGGREGATION_EXCEPTION; bounded below 32-hour ceiling by observed short run and benchmark",
        "gpuHours": 0.0,
        "newGardTrajectories": 0,
        "newPhiRLOrEmergenceValues": 0,
        "retainedInvalidatedWorkerOutcomes": False,
        "disposableInvalidatedCaches": "/cache/e01_s19_l06",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__, "scipy": scipy.__version__},
        "originalPreOutcomeCommit": "1a1f7f61582c2d39c1b1df241cb029131e2326b6",
        "finalPreOutcomeAmendedCommit": "0c4460ce6db913c98cbc4a898af47fe4afe54b12",
        "reportingCommit": git_output("rev-parse", "HEAD"),
        "completedUtc": now,
    }
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime)

    retained = sum(path.stat().st_size for path in LOOP_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l06_storage_validation.v1",
        "retainedBytes": retained,
        "retainedCeilingBytes": 10 * 1024**3,
        "temporaryCeilingBytes": 25 * 1024**3,
        "passed": retained <= 10 * 1024**3,
    }
    write_json(LOOP_ROOT / "storage_validation.json", storage)

    report = f"""# E01/S19-L06 — Past-only multi-attractor boundary recurrence (failed closed)

## Concise top summary

- **Research step ID:** `S19-L06`
- **Completion status:** `LOOP_FAILED_CLOSED`; mandatory human-review boundary active
- **Artifacts written:** complete preregistration/lock/replay/failure/status/runtime/storage/hash package; explicit empty-not-eligible label, fingerprint, bootstrap, permutation, suffix and comparison tables; canonical full report and decision summary; append-only S19 handoff ledgers
- **Validation result:** `{VALIDATION_RESULT}`; immutable prior PASS across {prior['fileCount']:,} files; preanalysis replay and pushed lock PASS; independent boundary score exact replay FAIL
- **Outcome classification:** `LOOP_FAILED_CLOSED`; `POSSIBLE_PIPELINE_ARTIFACT`; zero promoted leads; no scientific temporal fingerprint adjudicated
- **Caveats or blockers:** primary and independent float64 paths agreed on labels and recurrence counts but differed bitwise for {SCORE_MISMATCH_COUNT}/{SCORE_COUNT} finite scores on the first diagnosed trajectory (maximum absolute difference `{MAX_ABS_SCORE_DIFFERENCE:.17g}`); the locked contract allowed no tolerance or post-outcome repair
- **Recommended next action:** mandatory human review; do not repair or rerun L06 under this authorization, and do not activate L07, S20, E02, author contact, or report generation automatically

## Lay summary

L06 cannot answer whether online recurrence among post-fission boundaries better reproduces the paper's 88% replicator-state fingerprint. The input data and frozen labels replayed exactly, and two executions of the new label agreed on every positive/negative decision and recurrence count inspected. However, the preregistration also required the underlying floating-point boundary scores to match **bit for bit**. Two mathematically equivalent calculation orders differed at machine precision—at most about three ten-quadrillionths—but that still violates the exact gate. The loop therefore stopped before any occupancy, onset, consistency, episode, bootstrap, permutation, or promotion result became eligible.

This is an operational validation failure, not evidence for or against the boundary-recurrence hypothesis. L06 preserves the failure rather than relaxing its rule after outcomes.

## Frozen question

The locked question was whether a single strict-`H>0.9`, past-only recurrence decision among multiple selected post-fission boundaries, projected prospectively through the following growth interval, jointly improved the paper-facing temporal fingerprint in candidate 2 and candidate 3. Adjacent molecular `H>0.9` was comparator-only; L03 and L05 were fixed prior evidence. Occupancy alone could not decide success.

## Inputs

- Frozen S13Y dataset: 100 shared catalytic matrices, 200 candidate-specific trajectories, 180,635 selected molecular-clock rows, and 20,000 selected post-fission boundaries.
- Candidate 2 and candidate 3 were kept separate.
- Frozen S13Y trajectory/cache identities, selected clocks, adjacent-H arrays, and `Y=I(H>0.9)` labels.
- Frozen L03 post-fission boundary identities and fixed L03/L05 comparison evidence.
- Original paper and source snapshots recorded in `source_snapshot_manifest.json`.
- No new GARD trajectory, PhiRL value, emergence value, prediction, intervention, threshold, recurrence-count variant, cluster, centroid, modal reference, or alignment branch.

## Detailed methods

At boundary `b_g` for positive generation `g`, the sole structural rule would activate iff strict historical cosine similarity exceeded 0.9 for at least one prior selected boundary `b_h` satisfying `0<h<=g-2`. Each earlier generation counted once. The decision would label `b_g` and selected molecular rows up to but excluding `b_(g+1)`, with no future reference, backfill, persistence across re-evaluation, or alternative branch.

The prospectively locked validation hierarchy was:

1. byte/hash replay of every immutable S01–S18/V1/V2/S19-L01–L05 input;
2. exact replay of all S13Y identities, clocks, boundary identities, adjacent-H arrays and frozen labels;
3. clean, pushed repository/config lock;
4. exact two-pass and independent replay of labels, scores and recurrence evidence;
5. only after those checks, fingerprint, suffix, bootstrap, leave-one-out, permutation and promotion analyses.

The failure occurred at item 4, so items 5 and all scientific adjudication are ineligible.

## Commands and execution chronology

Pre-outcome tests and preparation:

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l03.py tests/e01/test_s19_l04.py tests/e01/test_s19_l05.py tests/e01/test_s19_l06.py
PYTHONPATH=src ruff check src/e01_s19_boundary_recurrence scripts/e01/prepare_s19_l06_lock.py scripts/e01/run_s19_l06.py tests/e01/test_s19_l06.py
PYTHONPATH=src python scripts/e01/prepare_s19_l06_lock.py
```

The original scientific launch failed at module import before outcome access. `S19-L06-VPA-001` added only the repository root to `sys.path`, preserved that failure, reran tests, committed, pushed, and revalidated the clean lock. It changed no scientific rule or value.

Locked execution:

```text
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s19_l06.py --workers 8
```

All workers returned, but the runner raised `L06 trajectory execution or locked audit failed` before aggregation and artifact serialization. A validation-only diagnosis of the first failing trajectory established the exact failure signature recorded below. No scientific repair, tolerance, aggregation, or rerun followed.

## Results

### Eligible scientific results

None. All scientific result tables have explicit schemas and zero rows. Invalidated worker caches remain disposable under `/cache`; they were not read or serialized by this finalizer.

| Item | Result | Eligibility |
|---|---:|---|
| Frozen preanalysis trajectories | 200/200 passed | Validation evidence only |
| Frozen selected clock rows | 180,635 replayed | Validation evidence only |
| Frozen post-fission boundaries | 20,000 replayed | Validation evidence only |
| First diagnosed independent-score trajectory | candidate 2, M000 | Validation evidence only |
| Finite boundary scores compared | {SCORE_COUNT} | Validation evidence only |
| Bitwise unequal scores | {SCORE_MISMATCH_COUNT} | **Global stop** |
| Maximum absolute score difference | `{MAX_ABS_SCORE_DIFFERENCE:.17g}` | **Global stop** |
| Label mismatches | 0 | Does not override failed score gate |
| Recurrence-count mismatches | 0 | Does not override failed score gate |
| Eligible fingerprints / promotions | 0 / 0 | No adjudication |

No occupancy, persistence, onset, consistency, episode, quarter-cutoff, recurrence, cross-candidate, joint-distance, bootstrap, leave-one-out, block-permutation, retrospective resemblance, prediction, intervention, or causal-control conclusion is drawn.

## Validation

- Immutable S01–S18/V1/V2/S19-L01–L05 postcheck: **PASS**, {prior['fileCount']:,} files and zero mismatches.
- Preanalysis replay: **PASS**, all 200 trajectories and all frozen identity/clock/boundary/H/label fields.
- Clean pushed pre-outcome lock: **PASS** at `0c4460ce6db913c98cbc4a898af47fe4afe54b12`.
- Pre-outcome launch-path amendment: **PASS**, value-preserving.
- Primary two-pass label/score/boundary replay on the diagnosed trajectory: **PASS**.
- Independent labels and recurrence counts on that trajectory: **PASS**.
- Independent bit-exact boundary scores: **FAIL**.
- Whole-loop eligibility: **FAIL CLOSED**.
- New trajectories / PhiRL / emergence / GPU use: 0 / 0 / 0 / 0.
- Retained-artifact storage: {'PASS' if storage['passed'] else 'FAIL'}.

## Self-improvement record

- **Belief before:** boundary-only multi-attractor recurrence might occupy the middle ground between L03's restrictive modal compotype and L05's permissive molecular recurrence.
- **Evidence motivating the test:** L03 and L05 bracketed temporal fingerprints under distinct recurrence granularities.
- **Ambiguity targeted:** whether generation-boundary granularity filters local compositional drift and creates meaningful online pre-onset intervals.
- **What was learned:** differently ordered float64 cosine implementations cannot satisfy the locked bit-exact score replay even when their label decisions and counts coincide.
- **Hypothesis weakened:** bit-exact numerical identity of those two implementation paths.
- **Hypothesis still plausible:** the boundary-recurrence scientific hypothesis remains untested by eligible L06 evidence.
- **What should be tested next:** nothing automatically. A human must decide whether any new prospectively locked action is warranted.
- **Why a future action could add information:** it could define numerical replay semantics before new outcomes; retroactively tolerating L06 would instead weaken the preregistered gate.

## Caveats and blockers

The failure magnitude is compatible with ordinary float64 operation-order rounding, but that inference does not excuse the locked bit-exact requirement. Only the first failing trajectory was diagnosed because one failure is sufficient for the global stop. Candidate 3 and the complete fingerprint remain scientifically unadjudicated. The known paper fingerprint and five previous adaptive loops also make any eventual positive label result exploratory until untouched confirmation. Exact author semantics remain unavailable.

## Provenance

The original pre-outcome lock was pushed at `1a1f7f61582c2d39c1b1df241cb029131e2326b6`; the value-preserving launch amendment was pushed and clean at `0c4460ce6db913c98cbc4a898af47fe4afe54b12`. Contracts, sources, inputs, seeds, benchmark and replay evidence are named in the L06 directory. `reporting_amendment_002.json` and this repository finalizer document the post-stop reporting-only action. `artifact_manifest.json` hashes the final retained package.

## Mandatory human-review boundary

L06 is complete as `LOOP_FAILED_CLOSED`. No L07, S20, E02, author contact, or report-bundle work is active. Hand control back for a new explicit human decision.
"""

    decision = f"""# S19-L06 one-page decision summary

## Concise top summary

- **Research step ID:** `S19-L06`
- **Completion status:** `LOOP_FAILED_CLOSED`; mandatory human review required
- **Artifacts written:** complete failure/status/validation package with empty-not-eligible scientific tables and append-only ledgers
- **Validation result:** immutable prior, preanalysis and pushed lock PASS; independent bit-exact boundary-score replay FAIL
- **Outcome classification:** `LOOP_FAILED_CLOSED`; zero promoted leads; boundary-recurrence hypothesis unadjudicated
- **Caveats or blockers:** labels and recurrence counts agreed, but {SCORE_MISMATCH_COUNT}/{SCORE_COUNT} scores differed bitwise by at most `{MAX_ABS_SCORE_DIFFERENCE:.17g}`; L06 prohibited a tolerance or post-outcome repair
- **Recommended next action:** human review; do not repair/rerun L06 or activate any downstream step automatically

## Decision evidence

L06 passed its immutable-input and pushed-lock gates. The first diagnosed candidate-2 trajectory then failed the required independent score replay. Exact labels and recurrence counts do not rescue the contract because score identity was explicitly locked. No temporal fingerprint, negative control, promotion gate, or paper-facing result is eligible.

## Human-review boundary

L06 neither supports nor refutes online boundary recurrence. It produced no promotion. A later action requires a fresh explicit authorization; L07 and S20 remain inactive.
"""
    (LOOP_ROOT / "S19_L06_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "loop_decision_summary.md").write_text(decision, encoding="utf-8")
    (S19_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")

    append_ledgers(now, classification)

    artifacts = [
        str(LOOP_ROOT / "S19_L06_FULL_RESULTS.md"),
        str(LOOP_ROOT / "research_step_full_results.md"),
        str(LOOP_ROOT / "classification.json"),
        str(LOOP_ROOT / "failure_ledger.csv"),
        str(LOOP_ROOT / "independent_label_replay.parquet"),
        str(LOOP_ROOT / "regeneration_validation.json"),
        str(LOOP_ROOT / "artifact_manifest.json"),
    ]
    status = {
        "researchStepId": "S19-L06",
        "stepNumber": 19,
        "success": False,
        "status": "LOOP_FAILED_CLOSED_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": artifacts,
        "validationResult": VALIDATION_RESULT,
        "caveatsOrBlockers": [
            "independent_boundary_scores_not_bit_exact",
            "401_of_713_first_diagnosed_scores_differed",
            "maximum_absolute_difference_3.3306690738754696e-16",
            "labels_and_recurrence_counts_equal_but_score_gate_mandatory",
            "no_eligible_scientific_fingerprint_or_promotion",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_NO_AUTOMATIC_NEXT_STEP",
    }
    write_json(LOOP_ROOT / "status.json", status)
    write_json(S19_ROOT / "s19_status.json", status)

    regeneration = {
        "schema": "eidosoma.e01.s19_l06_regeneration_validation.v1",
        "loopId": LOOP_ID,
        "validationResult": VALIDATION_RESULT,
        "passed": False,
        "immutablePriorPassed": prior["passed"],
        "preanalysisReplayPassed": preanalysis["passed"],
        "pushedExecutionLockPassed": lock["passed"],
        "independentExactBoundaryScoreReplayPassed": False,
        "independentLabelReplayPassedOnDiagnosedTrajectory": True,
        "independentRecurrenceCountReplayPassedOnDiagnosedTrajectory": True,
        "scientificResultsEligible": False,
        "scientificTablesExplicitlyEmpty": True,
        "invalidatedWorkerOutcomesSerialized": False,
        "reportingOnlyFinalizer": str(Path(__file__).relative_to(REPO_ROOT)),
        "requiredArtifactsFinalizedAsFailureStatuses": True,
        "nextScientificWorkAuthorized": False,
    }
    write_json(LOOP_ROOT / "regeneration_validation.json", regeneration)

    required = [
        "preregistration.yaml", "method_lock.json", "candidate_ranking.csv",
        "candidate_bundle_registry.yaml", "label_registry.yaml", "label_registry.parquet",
        "specification_ledger.parquet", "seed_manifest.parquet", "input_manifest.json",
        "source_snapshot_manifest.json", "untouched_s20_design.yaml",
        "preoutcome_repository_lock.json", "immutable_prior_baseline.json",
        "immutable_prior_validation.json", "immutable_prior_postcheck.json",
        "compute_benchmark.json", "preanalysis_replay_evidence.parquet",
        "preanalysis_replay_validation.json", "preparation_runtime.json",
        "execution_lock_validation.json", "value_preserving_amendment_001.json",
        "reporting_amendment_002.json", "execution_status.parquet",
        "label_values.parquet", "boundary_projected_label_results.parquet",
        "boundary_activation_results.parquet", "boundary_recurrence_trajectory_diagnostics.parquet",
        "label_replay_evidence.parquet", "independent_label_replay.parquet",
        "frozen_comparator_replay.parquet", "future_suffix_invariance_results.parquet",
        "future_suffix_invariance_summary.json", "fingerprint_results.parquet", "results.parquet",
        "fingerprint_summary.parquet", "fingerprint_aggregate.csv", "episode_results.parquet",
        "cutoff_results.parquet", "boundary_recurrence_count_results.parquet",
        "label_overlap_results.parquet", "cross_candidate_agreement.csv",
        "paper_fingerprint_comparison.csv", "fixed_l03_l05_comparison.csv",
        "paper_distance_bootstrap.parquet", "bootstrap_metric_differences.parquet",
        "leave_one_out_robustness.parquet", "generation_block_permutation_results.parquet",
        "negative_control_results.parquet", "robustness_results.parquet", "failure_ledger.csv",
        "runtime_manifest.json", "storage_validation.json", "regeneration_validation.json",
        "classification.json", "status.json", "loop_decision_summary.md",
        "S19_L06_FULL_RESULTS.md", "research_step_full_results.md",
    ]
    loop_manifest = manifest(LOOP_ROOT, required, "eidosoma.e01.s19_l06_artifact_manifest.v1")
    write_json(LOOP_ROOT / "artifact_manifest.json", loop_manifest)
    if not loop_manifest["passed"]:
        raise RuntimeError(f"missing required L06 artifacts: {loop_manifest['missing']}")

    root_required = [
        "continuation_decision.md", "s18_immutable_baseline.json",
        "self_improvement_ledger.parquet", "SELF_IMPROVEMENT_LEDGER.md",
        "candidate_registry.parquet", "source_search_ledger.parquet", "source_search_report.md",
        "loop_registry.yaml", "human_review_history.json", "s19_status.json",
        "research_step_full_results.md",
    ]
    root_manifest = manifest(S19_ROOT, root_required, "eidosoma.e01.s19_artifact_manifest.v6")
    write_json(S19_ROOT / "artifact_manifest.json", root_manifest)
    if not root_manifest["passed"]:
        raise RuntimeError(f"missing required S19 root artifacts: {root_manifest['missing']}")

    # Recompute storage after all retained artifacts and update only its reporting value.
    retained = sum(path.stat().st_size for path in LOOP_ROOT.rglob("*") if path.is_file())
    storage["retainedBytes"] = retained
    storage["passed"] = retained <= storage["retainedCeilingBytes"]
    write_json(LOOP_ROOT / "storage_validation.json", storage)
    # Refresh manifests after the last reporting-only byte-count update.
    write_json(LOOP_ROOT / "artifact_manifest.json", manifest(LOOP_ROOT, required, "eidosoma.e01.s19_l06_artifact_manifest.v1"))
    write_json(S19_ROOT / "artifact_manifest.json", manifest(S19_ROOT, root_required, "eidosoma.e01.s19_artifact_manifest.v6"))

    print(
        canonical_json(
            {
                "loopId": LOOP_ID,
                "status": status["status"],
                "validationResult": VALIDATION_RESULT,
                "classification": classification["topLevelClassification"],
                "promotedLeadCount": 0,
                "scientificResultsEligible": False,
                "immutablePriorPassed": True,
                "mandatoryHumanReview": True,
                "emptyScientificTables": len(scientific_files),
            }
        )
    )


if __name__ == "__main__":
    main()
