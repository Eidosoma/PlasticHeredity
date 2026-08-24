#!/usr/bin/env python3
"""Freeze the S12H two-stage method without opening scientific outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from e01_boundary_clock_revalidation.core import (
    DERIVED_CANDIDATE_ID,
    RESEARCH_STEP_ID,
    VERSION,
    stage2_candidate_registry,
)

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
ARTIFACTS = Path("/artifacts")
STEP_ROOT = ARTIFACTS / "research_steps/S12H"
CONFIG = REPO / "configs/e01/s12h_candidate1_boundary_clock_revalidation_preregistration.yaml"
S12G_SCHEMA = REPO / "configs/e01/s12g_output_schemas.json"
S12FR_LOCK = ARTIFACTS / "research_steps/S12FR/candidate_timebase_pipeline_lock.json"
S12FR_MANIFEST = ARTIFACTS / "research_steps/S12FR/confirmation_trajectory_manifest.parquet"
S12G_CACHE_MANIFEST = ARTIFACTS / "research_steps/S12G/partial_execution_manifest.json"
S12G_RESULT_CACHE = Path("/cache/e01_s12g/source_results")


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


def audit_source_checkout(path: Path, expected_commit: str) -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=path, text=True).strip()
    return {
        "path": str(path),
        "expectedCommit": expected_commit,
        "actualCommit": head,
        "workingTreeStatus": status,
        "passed": head == expected_commit and not status,
    }


def prior_baseline() -> dict[str, Any]:
    research_files: list[dict[str, Any]] = []
    for root in sorted((ARTIFACTS / "research_steps").glob("S*")):
        if root.name == "S12H" or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                research_files.append(
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                )

    lock = json.loads(S12FR_LOCK.read_text(encoding="utf-8"))
    raw_caches: list[dict[str, Any]] = []
    for candidate in lock["confirmedCandidates"]:
        for item in candidate["trajectoryLocks"]:
            path = Path(item["cachePath"])
            actual = sha256_file(path) if path.is_file() else None
            raw_caches.append(
                {
                    "candidateId": candidate["candidateId"],
                    "matrixIndex": item["matrixIndex"],
                    "path": str(path),
                    "bytes": path.stat().st_size if path.is_file() else None,
                    "sha256": actual,
                    "expectedSha256": item["cacheSha256"],
                    "passed": actual == item["cacheSha256"],
                }
            )

    partial = json.loads(S12G_CACHE_MANIFEST.read_text(encoding="utf-8"))
    task_cache_files: list[dict[str, Any]] = []
    for item in partial["cacheFiles"]:
        path = S12G_RESULT_CACHE / item["relativePath"]
        actual = sha256_file(path) if path.is_file() else None
        task_cache_files.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": actual,
                "expectedSha256": item["sha256"],
                "passed": actual == item["sha256"],
            }
        )
    payload = {
        "schema": "eidosoma.e01.s12h_immutable_prior_baseline.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "createdAtUtc": datetime.now(UTC).isoformat(),
        "researchStepFiles": research_files,
        "lockedTrajectoryCaches": raw_caches,
        "s12gTaskCacheFiles": task_cache_files,
        "researchStepFileCount": len(research_files),
        "lockedTrajectoryCacheCount": len(raw_caches),
        "s12gTaskCacheFileCount": len(task_cache_files),
        "passed": all(item["passed"] for item in raw_caches + task_cache_files),
    }
    return payload


def input_manifest_and_pairing() -> tuple[pd.DataFrame, dict[str, Any]]:
    prior = pd.read_parquet(S12FR_MANIFEST).sort_values(["candidateId", "matrixIndex"])
    if len(prior) != 96 or prior["candidateId"].nunique() != 3:
        raise RuntimeError("S12FR confirmation manifest is not exactly 96 rows")
    registry = {item["candidateId"]: item for item in stage2_candidate_registry()}
    rows: list[dict[str, Any]] = []
    for item in prior.to_dict("records"):
        candidate = registry[str(item["candidateId"])]
        cache = Path(str(item["cachePath"]))
        actual = sha256_file(cache)
        rows.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": item["candidateId"],
                "analysisIdentity": candidate["analysisIdentity"],
                "matrixIndex": int(item["matrixIndex"]),
                "trajectoryId": item["trajectoryId"],
                "originalS12frClockId": item["clockId"],
                "clockId": candidate["clockId"],
                "betaSha256": item["betaSha256"],
                "initialStateSha256": item["initialStateSha256"],
                "trajectorySha256": item["trajectorySha256"],
                "cachePath": item["cachePath"],
                "cacheSha256": item["cacheSha256"],
                "completedFissions": int(item["completedFissions"]),
                "totalBatchUpdates": int(item["totalBatchUpdates"]),
                "selectedObservationCount": int(item["clockC1"]) + 1,
                "selectedTransitionCount": int(item["clockC1"]),
                "cacheHashPassed": actual == item["cacheSha256"],
                "candidateIdentityPassed": bool(
                    float(item["h"]) == float(candidate["h"])
                    and item["daughterRule"] == candidate["daughterRule"]
                    and item["overshootRule"] == candidate["overshootRule"]
                    and int(item["completedFissions"]) == 100
                ),
                "repairedReplayPassed": bool(item["repairedReplayPassed"]),
                "discreteDivergenceCount": int(item["discreteDivergenceCount"]),
                "finiteNumericDivergenceCount": int(item["finiteNumericDivergenceCount"]),
                "forbiddenNonfiniteDifferenceCount": int(item["forbiddenNonfiniteDifferenceCount"]),
                "seedDifferenceCount": int(item["seedDifferenceCount"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["candidateId", "matrixIndex"]).reset_index(drop=True)
    pairing_rows: list[dict[str, Any]] = []
    for matrix_index, group in frame.groupby("matrixIndex", sort=True):
        beta = group["betaSha256"].nunique() == 1
        initial = group["initialStateSha256"].nunique() == 1
        pairing_rows.append(
            {
                "matrixIndex": int(matrix_index),
                "candidateCount": len(group),
                "betaShared": beta,
                "initialStateShared": initial,
                "sharedIdentity": beta and initial and len(group) == 3,
                "betaSha256": group["betaSha256"].iloc[0] if beta else None,
                "initialStateSha256": group["initialStateSha256"].iloc[0] if initial else None,
            }
        )
    pairing = {
        "schema": "eidosoma.e01.s12h_shared_identity_audit.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "rows": pairing_rows,
        "sharedIdentityCount": sum(item["sharedIdentity"] for item in pairing_rows),
        "passed": len(pairing_rows) == 32 and all(item["sharedIdentity"] for item in pairing_rows),
    }
    return frame, pairing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-commit", required=True)
    args = parser.parse_args()
    if STEP_ROOT.exists() and any(STEP_ROOT.iterdir()):
        raise RuntimeError("S12H artifact directory must be empty before preregistration freeze")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != args.design_commit or remote != head or git("status", "--short"):
        raise RuntimeError("S12H design must be committed, pushed, and clean")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["researchStepId"] != VERSION:
        raise RuntimeError("S12H version mismatch")
    STEP_ROOT.mkdir(parents=True)
    shutil.copyfile(CONFIG, STEP_ROOT / "preregistration.yaml")

    baseline = prior_baseline()
    if not baseline["passed"]:
        raise RuntimeError("S12H immutable baseline failed")
    write_json(STEP_ROOT / "immutable_prior_baseline.json", baseline)
    manifest, pairing = input_manifest_and_pairing()
    if not manifest["cacheHashPassed"].all() or not manifest["candidateIdentityPassed"].all():
        raise RuntimeError("S12H locked input identity failed")
    manifest.to_parquet(STEP_ROOT / "trajectory_input_manifest.parquet", index=False, compression="zstd")
    write_json(STEP_ROOT / "shared_identity_audit.json", pairing)

    sources = {
        "historicalGard": audit_source_checkout(Path("/cache/e01_s03/sources/gard-historical"), "86dff6320d5ae91b4e831471079ff46749b14df9"),
        "IIGR_CORRECTED_SOURCE": audit_source_checkout(Path("/cache/e01_s12b/sources/IntegratedInformationGeneRegulation"), "7c1c22fe39f539d4a453135476f1f0dd5a6b45f7"),
        "PHIRL_REGULARIZED_SOURCE": audit_source_checkout(Path("/cache/e01_s12b/sources/PhiRL"), "a6d1d0d18c7551302724b7158c6ccdc4d3a33373"),
    }
    safe = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
    s12c = pd.read_csv("/artifacts/research_steps/S12C/source_equivalence_results.csv")
    s12d = pd.read_csv("/artifacts/research_steps/S12D/source_metric_equivalence.csv")
    source_manifest = {
        "schema": "eidosoma.e01.s12h_source_snapshot_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "sources": sources,
        "safeLattice": {"path": str(safe), "sha256": sha256_file(safe), "passed": sha256_file(safe) == config["sourcePins"]["safeLattice"]["sha256"]},
        "s12cEquivalence": {
            "rows": len(s12c),
            "passingRows": int(s12c["allGatesPassed"].astype(bool).sum()),
            "passed": bool(
                len(s12c) == 14 and s12c["allGatesPassed"].astype(bool).all()
            ),
        },
        "s12dMetricIdentity": {
            "rows": len(s12d),
            "passingRows": int(s12d["allGatesPassed"].astype(bool).sum()),
            "passed": bool(
                len(s12d) == 40 and s12d["allGatesPassed"].astype(bool).all()
            ),
        },
    }
    source_manifest["passed"] = bool(
        all(item["passed"] for item in sources.values())
        and source_manifest["safeLattice"]["passed"]
        and source_manifest["s12cEquivalence"]["passed"]
        and source_manifest["s12dMetricIdentity"]["passed"]
    )
    if not source_manifest["passed"]:
        raise RuntimeError("S12H source/equivalence gate failed")
    write_json(STEP_ROOT / "source_snapshot_manifest.json", source_manifest)

    partial = json.loads(S12G_CACHE_MANIFEST.read_text(encoding="utf-8"))
    exclusion = {
        "schema": "eidosoma.e01.s12h_s12g_cache_exclusion_audit.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "manifestPath": str(S12G_CACHE_MANIFEST),
        "manifestSha256": sha256_file(S12G_CACHE_MANIFEST),
        "recordedFileCount": partial["cacheFileCount"],
        "recordedBytes": partial["cacheBytes"],
        "baselineHashPassCount": sum(item["passed"] for item in baseline["s12gTaskCacheFiles"]),
        "payloadFilesOpened": 0,
        "scientificReusePermitted": False,
        "resultCacheRoot": "/cache/e01_s12h/source_results",
        "passed": len(baseline["s12gTaskCacheFiles"]) == partial["cacheFileCount"] and all(item["passed"] for item in baseline["s12gTaskCacheFiles"]),
    }
    write_json(STEP_ROOT / "s12g_cache_exclusion_audit.json", exclusion)

    registry = {"schema": "eidosoma.e01.s12h_candidate_registry.v1", "candidates": stage2_candidate_registry(), "rankingWeightsUsed": False, "candidateSelectionOrReweightingPermitted": False}
    (STEP_ROOT / "candidate_registry.yaml").write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    for source_name, target_name in (
        ("label_registry.yaml", "label_registry.yaml"),
        ("preprocessing_registry.yaml", "preprocessing_registry.yaml"),
        ("metric_registry.yaml", "metric_registry.yaml"),
        ("analysis_registry.yaml", "analysis_registry.yaml"),
    ):
        shutil.copyfile(ARTIFACTS / "research_steps/S12G" / source_name, STEP_ROOT / target_name)

    code_files = [
        CONFIG,
        S12G_SCHEMA,
        REPO / "src/e01_boundary_clock_revalidation/core.py",
        REPO / "scripts/e01/freeze_s12h_preregistration.py",
        REPO / "scripts/e01/run_s12h_candidate1_boundary_clock_revalidation.py",
        REPO / "tests/e01/test_s12h_candidate1_boundary_clock_revalidation.py",
        REPO / "scripts/e01/run_s12g_frozen_timebase_ensemble.py",
        REPO / "src/e01_frozen_timebase_ensemble/core.py",
    ]
    lock = {
        "schema": "eidosoma.e01.s12h_method_lock.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "designCommit": head,
        "remoteCommit": remote,
        "branch": git("branch", "--show-current"),
        "files": [{"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for path in code_files],
        "priorImmutabilityBaselinePassed": baseline["passed"],
        "sourceSnapshotPassed": source_manifest["passed"],
        "inputCount": len(manifest),
        "sharedIdentityCount": pairing["sharedIdentityCount"],
        "s12gCachePayloadsOpened": 0,
        "labelOutcomesOpened": False,
        "informationTheoryOutcomesOpened": False,
        "passed": True,
    }
    write_json(STEP_ROOT / "method_lock.json", lock)
    write_json(STEP_ROOT / "implementation_lock.json", {**lock, "schema": "eidosoma.e01.s12h_implementation_lock.v1"})
    write_json(
        STEP_ROOT / "preregistration_record.json",
        {
            "schema": "eidosoma.e01.s12h_preregistration_record.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "frozenAtUtc": datetime.now(UTC).isoformat(),
            "designCommit": head,
            "remoteCommit": remote,
            "preregistrationSha256": sha256_file(STEP_ROOT / "preregistration.yaml"),
            "methodLockSha256": sha256_file(STEP_ROOT / "method_lock.json"),
            "passed": True,
        },
    )
    write_json(
        STEP_ROOT / "preoutcome_issue_ledger.json",
        {
            "schema": "eidosoma.e01.s12h_preoutcome_issue_ledger.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "issues": [
                {
                    "issueId": "S12H-PREOUTCOME-001",
                    "failedDesignCommit": "a11ba147032e68414b33510dfc5011c7a69d89de",
                    "stage": "PREREGISTRATION_FREEZE",
                    "error": "TypeError:Object of type bool is not JSON serializable",
                    "cause": "pandas/numpy boolean scalar retained in nested source-equivalence manifest fields",
                    "resolution": "cast only manifest booleans to native Python bool before JSON serialization",
                    "scientificMethodChanged": False,
                    "simulatorChanged": False,
                    "rawTrajectoryOpened": False,
                    "labelOutcomeOpened": False,
                    "informationTheoryOutcomeOpened": False,
                    "s12gCachePayloadOpened": False,
                    "newGardTrajectoryGenerated": False,
                    "failedPartialArtifactsQuarantinedAt": "/cache/e01_s12h/preoutcome_freeze_attempt_a11ba14",
                    "status": "RESOLVED_BEFORE_ANY_OUTCOME_ACCESS",
                },
                {
                    "issueId": "S12H-PREOUTCOME-002",
                    "failedDesignCommit": "415a6e8df674bfa022f271c24c1b927a4bfc563f",
                    "stage": "STAGE1_MODULE_IMPORT",
                    "error": "ModuleNotFoundError:No module named 'scripts'",
                    "cause": "direct script execution placed scripts/e01 rather than the repository root on sys.path",
                    "resolution": "insert the resolved repository root before importing the frozen S12G backend module",
                    "scientificMethodChanged": False,
                    "simulatorChanged": False,
                    "rawTrajectoryOpened": False,
                    "labelOutcomeOpened": False,
                    "informationTheoryOutcomeOpened": False,
                    "s12gCachePayloadOpened": False,
                    "newGardTrajectoryGenerated": False,
                    "failedPartialArtifactsQuarantinedAt": "/cache/e01_s12h/preoutcome_stage1_import_attempt_415a6e8",
                    "status": "RESOLVED_BEFORE_ANY_OUTCOME_ACCESS",
                },
            ],
            "passed": True,
        },
    )
    write_json(
        STEP_ROOT / "scope_access_ledger.json",
        {
            "schema": "eidosoma.e01.s12h_scope_access_ledger.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "events": [{"stage": "PRE_OUTCOME_METHOD_LOCK", "rawTrajectoryOpened": False, "labelOutcomeOpened": False, "informationTheoryOutcomeOpened": False, "s12gCachePayloadOpened": False, "newGardTrajectoryGenerated": False, "status": "PASS"}],
            "success": None,
        },
    )
    print(json.dumps({"stage": "S12H_preregistration_frozen", "designCommit": head, "inputCount": len(manifest), "priorFiles": baseline["researchStepFileCount"], "s12gCacheFiles": baseline["s12gTaskCacheFileCount"], "derivedCandidateId": DERIVED_CANDIDATE_ID}, sort_keys=True))


if __name__ == "__main__":
    main()
