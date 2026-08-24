#!/usr/bin/env python3
"""Execute additive E01/S19-L06R after the pushed numerical repair lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.e01.run_s19_l06 as base
from e01_frozen_timebase_ensemble.core import selected_clock_observations
from e01_s19_boundary_recurrence.core import (
    boundary_recurrence,
    boundary_recurrence_reference,
)
from e01_s19_boundary_recurrence_repair.core import (
    ABSOLUTE_TOLERANCE,
    LOOP_ID,
    MAXIMUM_ULP_DISTANCE,
    RELATIVE_TOLERANCE,
    VERSION,
    compare_discrete_recurrence,
    compare_float64_scores,
)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L06R"
CACHE_ROOT = Path("/cache/e01_s19_l06r")
LABEL_CACHE = CACHE_ROOT / "labels"
PERMUTATION_CACHE = CACHE_ROOT / "permutation_metrics"
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
PREREG = REPO_ROOT / "configs/e01/s19_l06r_preregistration.yaml"
METHOD_LOCK = REPO_ROOT / "configs/e01/s19_l06r_method_lock.json"
ORIGINAL_WORKER = base.trajectory_worker
ORIGINAL_EXECUTE = base.execute_trajectories
ORIGINAL_CLASSIFY = base.classify
ORIGINAL_REPORT = base.report_text
ORIGINAL_DECISION = base.decision_summary_text
ORIGINAL_MANIFEST = base.artifact_manifest


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


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def configure_base() -> None:
    base.LOOP_ID = LOOP_ID
    base.VERSION = VERSION
    base.LOOP_ROOT = LOOP_ROOT
    base.CACHE_ROOT = CACHE_ROOT
    base.LABEL_CACHE = LABEL_CACHE
    base.PERMUTATION_CACHE = PERMUTATION_CACHE
    base.PREREG = PREREG
    base.METHOD_LOCK = METHOD_LOCK
    base.AMENDMENT = LOOP_ROOT / "repair_decision.md"


def numerical_row(record: dict[str, Any]) -> dict[str, Any]:
    """Recompute only the two locked score paths and exact discrete outputs."""

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    with Path(record["cachePath"]).open("rb") as handle:
        trajectory = pickle.load(handle)
    selected = selected_clock_observations(trajectory, str(record["clockId"]))
    states = np.asarray([item.state for item in selected], dtype=np.int64)
    generations = np.asarray(
        [item.growth_generation_one_based for item in selected], dtype=np.int64
    )
    kinds = np.asarray([item.observation_kind for item in selected], dtype=str)
    indices = np.arange(len(selected), dtype=np.int64)
    canonical = boundary_recurrence(states, generations, kinds, indices)
    independent = boundary_recurrence_reference(states, generations, kinds, indices)
    projected = compare_float64_scores(canonical["scores"], independent["scores"])
    boundary = compare_float64_scores(
        canonical["boundaryScores"], independent["scores"][canonical["boundaryIndices"]]
    )
    discrete = compare_discrete_recurrence(canonical, independent)
    passed = bool(projected["passed"] and boundary["passed"] and all(discrete.values()))
    row: dict[str, Any] = {
        "researchStepId": LOOP_ID,
        "candidateId": str(record["candidateId"]),
        "matrixIndex": int(record["matrixIndex"]),
        "trajectoryId": str(record["trajectoryId"]),
        "selectedClockCount": len(selected),
        "canonicalPath": "boundary_recurrence_CPU_float64",
        "independentPath": "boundary_recurrence_reference_CPU_float64",
        "absoluteTolerance": ABSOLUTE_TOLERANCE,
        "relativeTolerance": RELATIVE_TOLERANCE,
        "maximumAllowedUlpDistance": MAXIMUM_ULP_DISTANCE,
        **{f"projected_{key}": value for key, value in projected.items()},
        **{f"boundary_{key}": value for key, value in boundary.items()},
        **{f"exact_{key}": value for key, value in discrete.items()},
        "exactDiscreteAll": bool(all(discrete.values())),
        "passed": passed,
        "wallSeconds": time.perf_counter() - started_wall,
        "cpuSeconds": time.process_time() - started_cpu,
    }
    return row


def run_numerical_gate(
    manifest: pd.DataFrame, workers: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = manifest.sort_values(
        ["matrixIndex", "candidateId"], kind="stable"
    ).to_dict(orient="records")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(numerical_row, record) for record in records]
        for future in as_completed(futures):
            rows.append(future.result())
    frame = (
        pd.DataFrame(rows)
        .sort_values(["matrixIndex", "candidateId"], kind="stable")
        .reset_index(drop=True)
    )
    frame.to_parquet(LOOP_ROOT / "numerical_equivalence_results.parquet", index=False)
    candidate_summary = []
    for candidate, group in frame.groupby("candidateId", sort=True):
        candidate_summary.append(
            {
                "candidateId": candidate,
                "trajectoryCount": len(group),
                "passedCount": int(group["passed"].sum()),
                "nonBitExactProjectedFinitePairCount": int(
                    group["projected_nonBitExactFinitePairCount"].sum()
                ),
                "finiteProjectedPairCount": int(
                    group["projected_finitePairCount"].sum()
                ),
                "maximumAbsoluteError": float(
                    group["projected_maximumAbsoluteError"].max()
                ),
                "maximumRelativeError": float(
                    group["projected_maximumRelativeError"].max()
                ),
                "maximumUlpDistance": int(group["projected_maximumUlpDistance"].max()),
                "passed": bool(group["passed"].all()),
            }
        )
    summary = {
        "schema": "eidosoma.e01.s19_l06r_numerical_equivalence_summary.v1",
        "loopId": LOOP_ID,
        "trajectoryCount": len(frame),
        "candidateCount": int(frame["candidateId"].nunique()),
        "finiteNonfiniteMasksExactAll": bool(
            frame["projected_finiteMaskExact"].all()
            and frame["boundary_finiteMaskExact"].all()
        ),
        "nonfiniteClassesExactAll": bool(
            frame["projected_nonfiniteClassExact"].all()
            and frame["boundary_nonfiniteClassExact"].all()
        ),
        "exactBooleanLabelsAndRecurrenceOutputsAll": bool(
            frame["exactDiscreteAll"].all()
        ),
        "absoluteTolerancePassedAll": bool(
            frame["projected_absoluteTolerancePassed"].all()
            and frame["boundary_absoluteTolerancePassed"].all()
        ),
        "relativeTolerancePassedAll": bool(
            frame["projected_relativeTolerancePassed"].all()
            and frame["boundary_relativeTolerancePassed"].all()
        ),
        "ulpTolerancePassedAll": bool(
            frame["projected_ulpTolerancePassed"].all()
            and frame["boundary_ulpTolerancePassed"].all()
        ),
        "maximumAbsoluteError": float(
            max(
                frame["projected_maximumAbsoluteError"].max(),
                frame["boundary_maximumAbsoluteError"].max(),
            )
        ),
        "maximumRelativeError": float(
            max(
                frame["projected_maximumRelativeError"].max(),
                frame["boundary_maximumRelativeError"].max(),
            )
        ),
        "maximumUlpDistance": int(
            max(
                frame["projected_maximumUlpDistance"].max(),
                frame["boundary_maximumUlpDistance"].max(),
            )
        ),
        "candidateSummaries": candidate_summary,
        "scientificAnalysisReleased": bool(
            frame["passed"].all()
            and len(frame) == 200
            and frame["candidateId"].nunique() == 2
        ),
        "passed": bool(
            frame["passed"].all()
            and len(frame) == 200
            and frame["candidateId"].nunique() == 2
        ),
    }
    write_json(LOOP_ROOT / "numerical_equivalence_summary.json", summary)
    return frame, summary


def repair_worker(record: dict[str, Any]) -> dict[str, Any]:
    """Run unchanged L06 work after the global numerical release gate."""

    configure_base()
    result = ORIGINAL_WORKER(record)
    gate = numerical_row(record)
    independent = result["independentReplay"]
    discrete_columns = [key for key in gate if key.startswith("exact_")]
    independent.update(
        {
            key: value
            for key, value in gate.items()
            if key.startswith(("projected_", "boundary_", "exact_"))
        }
    )
    independent["numericalFiniteNonfiniteMasksExact"] = bool(
        gate["projected_finiteMaskExact"] and gate["boundary_finiteMaskExact"]
    )
    independent["numericalAllThreeBoundsPassed"] = bool(
        gate["projected_absoluteTolerancePassed"]
        and gate["projected_relativeTolerancePassed"]
        and gate["projected_ulpTolerancePassed"]
        and gate["boundary_absoluteTolerancePassed"]
        and gate["boundary_relativeTolerancePassed"]
        and gate["boundary_ulpTolerancePassed"]
    )
    independent["exactDiscreteRepairFieldsPassed"] = bool(
        all(gate[key] for key in discrete_columns)
    )
    independent["passed"] = bool(
        gate["passed"]
        and independent["materializedLabelsExact"]
        and independent["materializedScoresExact"]
        and independent["materializedDistinctCountsExact"]
        and independent["materializedBoundaryLabelsExact"]
        and independent["materializedBoundaryCountsExact"]
    )
    result["independentReplay"] = independent
    result["success"] = bool(
        all(row["exactTwoPassReplayPassed"] for row in result["replays"])
        and independent["passed"]
        and all(row["passed"] for row in result["suffixAudit"])
    )
    return result


def execute_after_gate(manifest: pd.DataFrame, workers: int):
    outputs = ORIGINAL_EXECUTE(manifest, workers)
    independent = outputs[4]
    independent.to_parquet(
        LOOP_ROOT / "numerical_equivalence_recompute_results.parquet", index=False
    )
    return outputs


def execution_lock_validation() -> dict[str, Any]:
    repository = json.loads(
        (LOOP_ROOT / "preoutcome_repository_lock.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (LOOP_ROOT / "preanalysis_replay_validation.json").read_text(encoding="utf-8")
    )
    immutable = json.loads(
        (LOOP_ROOT / "immutable_prior_validation.json").read_text(encoding="utf-8")
    )
    fixture = json.loads(
        (LOOP_ROOT / "synthetic_fixture_validation.json").read_text(encoding="utf-8")
    )
    benchmark = json.loads(
        (LOOP_ROOT / "compute_benchmark.json").read_text(encoding="utf-8")
    )
    numerical = json.loads(
        (LOOP_ROOT / "numerical_equivalence_summary.json").read_text(encoding="utf-8")
    )
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    clean = not bool(git("status", "--porcelain=v1"))
    hashes = {
        "repositoryPreregistration": sha256_file(PREREG),
        "artifactPreregistration": sha256_file(LOOP_ROOT / "preregistration.yaml"),
        "repositoryMethodLock": sha256_file(METHOD_LOCK),
        "artifactMethodLock": sha256_file(LOOP_ROOT / "method_lock.json"),
    }
    passed = bool(
        repository["passed"]
        and replay["passed"]
        and immutable["passed"]
        and fixture["passed"]
        and benchmark["gatePassed"]
        and numerical["passed"]
        and head == remote == repository["head"]
        and clean
        and hashes["repositoryPreregistration"] == hashes["artifactPreregistration"]
        and hashes["repositoryMethodLock"] == hashes["artifactMethodLock"]
    )
    return {
        "schema": "eidosoma.e01.s19_l06r_execution_lock_validation.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "preparedHead": repository["head"],
        "cleanWorktree": clean,
        "configHashes": hashes,
        "numericalEquivalenceReleasedScientificAnalysis": numerical[
            "scientificAnalysisReleased"
        ],
        "l06WorkerCacheUsed": False,
        "freshCacheRoot": str(CACHE_ROOT),
        "passed": passed,
    }


def classify(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = ORIGINAL_CLASSIFY(*args, **kwargs)
    result["schema"] = "eidosoma.e01.s19_l06r_classification.v1"
    result["researchStepId"] = LOOP_ID
    result["loopId"] = LOOP_ID
    result["versionedLoopId"] = VERSION
    result["additiveRepairOf"] = "S19-L06"
    result["l06RemainsFailedClosed"] = True
    result["numericalRepairPolicy"] = {
        "absoluteTolerance": ABSOLUTE_TOLERANCE,
        "relativeTolerance": RELATIVE_TOLERANCE,
        "maximumUlpDistance": MAXIMUM_ULP_DISTANCE,
        "allThreeRequired": True,
    }
    return result


def _repair_language(text: str) -> str:
    text = text.replace("S19-L06", "S19-L06R")
    text = re.sub(r"\bL06\b", "L06R", text)
    text = text.replace("/cache/e01_s19_l06r", "/cache/e01_s19_l06r")
    text = text.replace("S01-S18/V1/V2/S19-L01-L05", "S01-S18/V1/V2/S19-L01-L06")
    text = text.replace(
        "Exact two-pass, independent boundary/projection, frozen comparator",
        "Exact two-pass, numerically equivalent independent boundary/projection with exact discrete outputs, frozen comparator",
    )
    text = text.replace(
        "- Launch attempt 1 stopped at module import before the execution gate or any trajectory load. Value-preserving amendment `S19-L06R-VPA-001` added only the repository root to the runner import path, preserved the failure, and re-established a clean pushed pre-outcome lock.\n",
        "- The additive L06R contract changed only the independent finite-score comparison gate. Failed L06 remains immutable; no invalidated L06 worker cache entered aggregation.\n",
    )
    text = text.replace("tests/e01/test_s19_l06.py", "tests/e01/test_s19_l06r.py")
    text = text.replace("prepare_s19_l06_lock.py", "prepare_s19_l06r_lock.py")
    text = text.replace("run_s19_l06.py", "run_s19_l06r.py")
    return text


def report_text(*args: Any, **kwargs: Any) -> str:
    text = _repair_language(ORIGINAL_REPORT(*args, **kwargs))
    numerical = json.loads(
        (LOOP_ROOT / "numerical_equivalence_summary.json").read_text(encoding="utf-8")
    )
    section = f"""

## Additive numerical repair result

L06 remains an immutable failed-closed historical record. L06R prospectively replaced only bit-exact independent score equality with the existing S06 policy: identical finite/nonfinite masks and nonfinite classes, exact boolean labels and recurrence outputs, and simultaneous absolute error `<=1e-12`, relative error `<=1e-12`, and ULP distance `<=8` for every finite pair.

- Trajectories checked before scientific release: `{numerical["trajectoryCount"]}/200`.
- Maximum absolute error: `{numerical["maximumAbsoluteError"]:.17g}`.
- Maximum relative error: `{numerical["maximumRelativeError"]:.17g}`.
- Maximum ULP distance: `{numerical["maximumUlpDistance"]}`.
- Exact discrete outputs across all trajectories: `{numerical["exactBooleanLabelsAndRecurrenceOutputsAll"]}`.
- Numerical gate passed: `{numerical["passed"]}`.

Only after that all-trajectory gate passed were the unchanged fingerprint, suffix, bootstrap, leave-one-out, block-permutation, candidate-agreement, quarter-eligibility, and promotion analyses released.
"""
    return text + section


def decision_text(*args: Any, **kwargs: Any) -> str:
    text = _repair_language(ORIGINAL_DECISION(*args, **kwargs))
    numerical = json.loads(
        (LOOP_ROOT / "numerical_equivalence_summary.json").read_text(encoding="utf-8")
    )
    return text + (
        "\n## Repair gate\n\n"
        f"All `{numerical['trajectoryCount']}` trajectories passed exact discrete replay and the all-three finite-score policy; maximum absolute/relative/ULP errors were "
        f"`{numerical['maximumAbsoluteError']:.17g}`, `{numerical['maximumRelativeError']:.17g}`, and `{numerical['maximumUlpDistance']}`. Failed L06 remains unchanged.\n"
    )


def artifact_manifest(root: Path, required: list[str], schema: str) -> dict[str, Any]:
    required = [
        name for name in required if name != "value_preserving_amendment_001.json"
    ]
    return ORIGINAL_MANIFEST(root, required, schema.replace("s19_l06_", "s19_l06r_"))


def append_postloop_ledger(
    aggregate: pd.DataFrame, classification: dict[str, Any], timestamp: str
) -> None:
    path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(path)
    if ledger["loopId"].eq(LOOP_ID).sum() != 1:
        raise RuntimeError("L06R pre-loop ledger cardinality changed")
    structural = aggregate.loc[aggregate["labelId"].eq(base.STRUCTURAL_LABEL_ID)]
    gates = classification["labelClassifications"][1]["promotionGates"]
    failed = [key for key, passed in gates.items() if not passed]
    learned = {
        "candidateFingerprints": structural[
            [
                "candidateId",
                "meanOccupancy",
                "meanPersistence",
                "meanConsistency",
                "meanFirstOnsetRawIndex0",
                "nonreplicatingAtCutoffFraction",
                "noReplicatorThroughCutoffFraction",
            ]
        ].to_dict(orient="records"),
        "failedPromotionGates": failed,
        "promotedLeadIds": classification["promotedLeadIds"],
    }
    row = {
        "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
        "timestampUtc": timestamp,
        "loopId": LOOP_ID,
        "recordPhase": "POST_LOOP_REPAIR_LEARNING_AND_HUMAN_REVIEW_BOUNDARY",
        "beliefBeforeLoop": "L06's bit-exact score gate may have rejected mathematically equivalent CPU-float64 reduction orders despite exact labels and recurrence counts.",
        "motivatingEvidence": "The diagnosed L06 discrepancy was at floating-point scale with exact diagnosed discrete outputs.",
        "failureOrAmbiguityTargeted": "Whether all 200 independent score replays pass the frozen S06 numerical policy and release the unchanged L06 analysis.",
        "selectedHypotheses": "One numerical repair only; all L06 science and gates unchanged.",
        "learned": canonical_json(learned),
        "weakenedHypotheses": "The exact boundary-recurrence lead to the extent it failed any unchanged L06 promotion gate; bit-exact score equality as a necessary same-math replay criterion.",
        "remainingPlausibleHypotheses": "Only a promoted L06R retrospective lead, if any, or separately authorized source-grounded ambiguity; prospective prediction and causal non-support remain unchanged.",
        "proposedNextTest": "Mandatory human review; no automatic L07 or S20.",
        "informationGainRationale": "L06R adjudicated the diagnosed numerical gate once without method branching.",
        "appendOnly": True,
    }
    pd.concat(
        [ledger, pd.DataFrame([row])[ledger.columns]], ignore_index=True
    ).to_parquet(path, index=False)
    with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            "\n## Entry 014 — S19-L06R learning and human-review boundary\n\n"
            "- **What was learned:** The all-trajectory numerical policy and the complete unchanged analysis are recorded in L06R machine evidence.\n"
            f"- **Promoted leads:** `{classification['promotedLeadCount']}`.\n"
            f"- **Failed unchanged promotion gates:** `{', '.join(failed) if failed else 'none'}`.\n"
            "- **Interpretation:** Exploratory only; failed L06 and all prediction/causal conclusions remain unchanged.\n"
            "- **Next action:** Mandatory human review; no automatic downstream work.\n"
        )


def update_root_handoff(
    report: str,
    classification: dict[str, Any],
    validation_result: str,
    artifacts: list[str],
) -> None:
    (ARTIFACT_ROOT / "research_step_full_results.md").write_text(
        report, encoding="utf-8"
    )
    status = {
        "researchStepId": LOOP_ID,
        "stepNumber": 19,
        "success": True,
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": artifacts,
        "validationResult": validation_result,
        "caveatsOrBlockers": [
            "post_failure_repair_is_adaptive",
            "failed_L06_remains_immutable",
            "previously_studied_matrices",
            "exact_author_definition_unavailable",
            "retrospective_paper_facing_scope_only",
            "prediction_and_causal_non_support_unchanged",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_NO_AUTOMATIC_L07_S20_E02_OR_REPORT_GENERATION",
        "promotedLeadCount": classification["promotedLeadCount"],
        "promotedLeadIds": classification["promotedLeadIds"],
    }
    write_json(ARTIFACT_ROOT / "s19_status.json", status)
    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    for loop in registry["loops"]:
        if loop["loopId"] == LOOP_ID:
            loop.update(
                {
                    "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
                    "outcomeAccessed": True,
                    "completed": True,
                    "eligibleScientificResults": True,
                    "promotedLeadCount": classification["promotedLeadCount"],
                }
            )
    registry["laterLoopsAuthorized"] = False
    registry["s20Status"] = "DEFINED_INACTIVE"
    registry["proposedNextLoopTheme"] = None
    registry["proposedNextLoopActive"] = False
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["history"].append(
        {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "decision": "S19_L06R_COMPLETE_MANDATORY_HUMAN_REVIEW",
            "scope": VERSION,
            "source": "locked_execution_result",
        }
    )
    history["pendingDecision"] = "POST_S19_L06R_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)


def configure_monkeypatches() -> None:
    configure_base()
    base.trajectory_worker = repair_worker
    base.execute_trajectories = execute_after_gate
    base.execution_lock_validation = execution_lock_validation
    base.classify = classify
    base.report_text = report_text
    base.decision_summary_text = decision_text
    base.append_postloop_ledger = append_postloop_ledger
    base.update_root_handoff = update_root_handoff
    base.artifact_manifest = artifact_manifest


def rebuild_manifest(root: Path, schema: str) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    files = []
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and item != manifest_path
    ):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": schema,
        "root": str(root),
        "fileCount": len(files),
        "totalBytes": int(sum(row["bytes"] for row in files)),
        "files": files,
        "missing": [],
        "passed": True,
    }


def finalize_success(numerical_cpu_hours: float) -> None:
    report = (LOOP_ROOT / "research_step_full_results.md").read_text(encoding="utf-8")
    (LOOP_ROOT / "S19_L06R_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    runtime_path = LOOP_ROOT / "runtime_manifest.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["schema"] = "eidosoma.e01.s19_l06r_runtime_manifest.v1"
    runtime["researchStepId"] = LOOP_ID
    runtime["numericalGateCpuHours"] = numerical_cpu_hours
    runtime["totalScientificCpuHoursIncludingNumericalGate"] = float(
        runtime["scientificCpuHours"] + numerical_cpu_hours
    )
    runtime["l06WorkerCacheUsed"] = False
    runtime["freshCacheRoot"] = str(CACHE_ROOT)
    if runtime["totalScientificCpuHoursIncludingNumericalGate"] > 32:
        raise RuntimeError("L06R CPU ceiling exceeded")
    write_json(runtime_path, runtime)
    pd.DataFrame(
        [
            {
                "failureId": None,
                "phase": "COMPLETE",
                "status": "NO_FAILURE",
                "reason": None,
                "scientificOutcomesAccessed": True,
                "repairAttempted": True,
                "repairCount": 1,
                "anotherRepairPermitted": False,
            }
        ]
    ).to_csv(LOOP_ROOT / "failure_ledger.csv", index=False)
    status_path = LOOP_ROOT / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["researchStepId"] = LOOP_ID
    status["caveatsOrBlockers"] = [
        "post_failure_repair_is_adaptive",
        "failed_L06_remains_immutable",
        "retrospective_paper_facing_scope_only",
        "exact_author_definition_unavailable",
        "prediction_and_causal_non_support_unchanged",
    ]
    status["recommendedNextAction"] = (
        "MANDATORY_HUMAN_REVIEW_NO_AUTOMATIC_DOWNSTREAM_WORK"
    )
    write_json(status_path, status)
    regeneration_path = LOOP_ROOT / "regeneration_validation.json"
    regeneration = json.loads(regeneration_path.read_text(encoding="utf-8"))
    regeneration["schema"] = "eidosoma.e01.s19_l06r_regeneration_validation.v1"
    regeneration["numericalEquivalenceAll200Passed"] = True
    regeneration["failedL06Unchanged"] = True
    regeneration["l06WorkerCacheUsed"] = False
    write_json(regeneration_path, regeneration)
    write_json(
        LOOP_ROOT / "artifact_manifest.json",
        rebuild_manifest(LOOP_ROOT, "eidosoma.e01.s19_l06r_artifact_manifest.v1"),
    )
    write_json(
        ARTIFACT_ROOT / "artifact_manifest.json",
        rebuild_manifest(ARTIFACT_ROOT, "eidosoma.e01.s19_artifact_manifest.v7"),
    )


def main(workers: int) -> None:
    configure_base()
    if not LOOP_ROOT.is_dir():
        raise RuntimeError("run prepare_s19_l06r_lock.py after the pushed lock")
    if CACHE_ROOT.resolve() == Path("/cache/e01_s19_l06").resolve():
        raise RuntimeError("invalidated L06 cache path selected")
    manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet")
    numerical, summary = run_numerical_gate(manifest, workers)
    numerical_cpu_hours = float(numerical["cpuSeconds"].sum()) / 3600
    if not summary["passed"]:
        raise RuntimeError(
            "L06R numerical equivalence failed permanently; scientific analysis not released"
        )
    configure_monkeypatches()
    base.main(workers)
    finalize_success(numerical_cpu_hours)
    classification = json.loads(
        (LOOP_ROOT / "classification.json").read_text(encoding="utf-8")
    )
    print(
        canonical_json(
            {
                "loopId": LOOP_ID,
                "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
                "numericalEquivalencePassed": True,
                "maximumAbsoluteError": summary["maximumAbsoluteError"],
                "maximumRelativeError": summary["maximumRelativeError"],
                "maximumUlpDistance": summary["maximumUlpDistance"],
                "classification": classification["topLevelClassification"],
                "promotedLeadCount": classification["promotedLeadCount"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 8:
        raise SystemExit("workers must be in [1,8]")
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(variable) not in (None, "1"):
            raise SystemExit(f"{variable} must be unset or 1")
    main(arguments.workers)
