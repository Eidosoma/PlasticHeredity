#!/usr/bin/env python3
"""Freeze S12I's waiver-specific method before scientific access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from e01_aggregate_support_waiver_sensitivity.core import (
    RESEARCH_STEP_ID,
    VERSION,
    sensitivity_candidate_registry,
    validate_exact_waiver,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S12I"
CONFIG = REPO / "configs/e01/s12i_aggregate_support_waiver_sensitivity_preregistration.yaml"
S12G_SCHEMA = REPO / "configs/e01/s12g_output_schemas.json"
S12FR_LOCK = ARTIFACTS / "research_steps/S12FR/candidate_timebase_pipeline_lock.json"
S12FR_MANIFEST = ARTIFACTS / "research_steps/S12FR/confirmation_trajectory_manifest.parquet"
S12G_CACHE_MANIFEST = ARTIFACTS / "research_steps/S12G/partial_execution_manifest.json"
S12G_RESULT_CACHE = Path("/cache/e01_s12g/source_results")
S12H_CLASSIFICATION = ARTIFACTS / "research_steps/S12H/classification.json"
S12H_CONFIRMATION = ARTIFACTS / "research_steps/S12H/candidate1_timebase_confirmation.json"


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
        if root.name == RESEARCH_STEP_ID or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                research_files.append(
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
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
                    "matrixIndex": int(item["matrixIndex"]),
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
    all_cache_rows = raw_caches + task_cache_files
    return {
        "schema": "eidosoma.e01.s12i_immutable_prior_baseline.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "createdAtUtc": datetime.now(UTC).isoformat(),
        "researchStepFiles": research_files,
        "lockedTrajectoryCaches": raw_caches,
        "s12gTaskCacheFiles": task_cache_files,
        "researchStepFileCount": len(research_files),
        "lockedTrajectoryCacheCount": len(raw_caches),
        "s12gTaskCacheFileCount": len(task_cache_files),
        "passed": all(bool(item["passed"]) for item in all_cache_rows),
    }


def input_manifest_and_pairing() -> tuple[pd.DataFrame, dict[str, Any]]:
    prior = pd.read_parquet(S12FR_MANIFEST).sort_values(["candidateId", "matrixIndex"])
    if len(prior) != 96 or prior["candidateId"].nunique() != 3:
        raise RuntimeError("S12FR confirmation manifest is not exactly 96 rows")
    registry = {item["candidateId"]: item for item in sensitivity_candidate_registry()}
    rows: list[dict[str, Any]] = []
    for item in prior.to_dict("records"):
        candidate = registry[str(item["candidateId"])]
        cache = Path(str(item["cachePath"]))
        actual = sha256_file(cache) if cache.is_file() else None
        rows.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": item["candidateId"],
                "analysisIdentity": candidate["analysisIdentity"],
                "candidateEvidenceStatus": candidate["evidenceStatus"],
                "aggregateSupportGateWaived": bool(
                    candidate["aggregateSupportGateWaived"]
                ),
                "upstreamConfirmed": bool(candidate["upstreamConfirmed"]),
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
                    abs(float(item["h"]) - float(candidate["h"])) <= 1e-15
                    and item["daughterRule"] == candidate["daughterRule"]
                    and item["overshootRule"] == candidate["overshootRule"]
                    and int(item["completedFissions"]) == 100
                ),
                "repairedReplayPassed": bool(item["repairedReplayPassed"]),
                "discreteDivergenceCount": int(item["discreteDivergenceCount"]),
                "finiteNumericDivergenceCount": int(item["finiteNumericDivergenceCount"]),
                "forbiddenNonfiniteDifferenceCount": int(
                    item["forbiddenNonfiniteDifferenceCount"]
                ),
                "seedDifferenceCount": int(item["seedDifferenceCount"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["candidateId", "matrixIndex"]).reset_index(drop=True)
    pairing_rows: list[dict[str, Any]] = []
    for matrix_index, group in frame.groupby("matrixIndex", sort=True):
        beta_shared = group["betaSha256"].nunique() == 1
        initial_shared = group["initialStateSha256"].nunique() == 1
        shared = beta_shared and initial_shared and len(group) == 3
        pairing_rows.append(
            {
                "matrixIndex": int(matrix_index),
                "candidateCount": len(group),
                "uniqueBetaHashes": int(group["betaSha256"].nunique()),
                "uniqueInitialStateHashes": int(group["initialStateSha256"].nunique()),
                "sharedIdentity": bool(shared),
                "pairingPolicy": "PAIRED" if shared else "UNPAIRED",
            }
        )
    pairing = {
        "schema": "eidosoma.e01.s12i_shared_identity_audit.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "rows": pairing_rows,
        "sharedIdentityCount": sum(bool(item["sharedIdentity"]) for item in pairing_rows),
        "all32Shared": len(pairing_rows) == 32
        and all(bool(item["sharedIdentity"]) for item in pairing_rows),
        "pairedAnalysisPermitted": True,
        "passed": len(pairing_rows) == 32
        and all(bool(item["sharedIdentity"]) for item in pairing_rows),
    }
    return frame, pairing


def source_evidence(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_checks: list[dict[str, Any]] = []
    for key, checkout in (
        ("primary", Path("/cache/e01_s12b/sources/IntegratedInformationGeneRegulation")),
        ("robustness", Path("/cache/e01_s12b/sources/PhiRL")),
    ):
        source = config["sourceImplementations"][key]
        commit = audit_source_checkout(checkout, source["commit"])
        source_checks.append({"sourceId": source["id"], **commit})
        for filename, hash_key in (
            ("main.py", "mainSha256"),
            ("information.py", "informationSha256"),
        ):
            path = checkout / filename
            actual = sha256_file(path) if path.is_file() else None
            source_checks.append(
                {
                    "sourceId": source["id"],
                    "path": str(path),
                    "expectedSha256": source[hash_key],
                    "actualSha256": actual,
                    "passed": actual == source[hash_key],
                }
            )
    for key in ("confirmedWrapper", "emergenceWrapper"):
        source = config["sourceImplementations"][key]
        path = REPO / source["module"]
        actual = sha256_file(path)
        source_checks.append(
            {
                "sourceId": key,
                "path": str(path),
                "expectedSha256": source["sha256"],
                "actualSha256": actual,
                "passed": actual == source["sha256"],
            }
        )
    safe = config["sourceImplementations"]["safeLattice"]
    safe_path = Path(safe["path"])
    safe_actual = sha256_file(safe_path)
    source_checks.append(
        {
            "sourceId": "SAFE_JSON_LATTICE",
            "path": str(safe_path),
            "expectedSha256": safe["sha256"],
            "actualSha256": safe_actual,
            "passed": safe_actual == safe["sha256"],
        }
    )
    s12c_path = ARTIFACTS / "research_steps/S12C/confirmation_fixture_results.csv"
    s12d_path = ARTIFACTS / "research_steps/S12D/source_metric_equivalence.csv"
    s12c = pd.read_csv(s12c_path)
    s12d = pd.read_csv(s12d_path)
    gates = config["sourceImplementations"]["preOutcomeEvidenceGates"]
    equivalence = {
        "schema": "eidosoma.e01.s12i_source_equivalence_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "s12cRows": len(s12c),
        "s12cArtifactSha256": sha256_file(s12c_path),
        "s12cAllPassed": bool(
            len(s12c) == gates["s12cConfirmationRowsExpected"]
            and s12c["allGatesPassed"].astype(bool).all()
            and sha256_file(s12c_path) == gates["s12cArtifactSha256"]
        ),
        "s12dRows": len(s12d),
        "s12dArtifactSha256": sha256_file(s12d_path),
        "s12dAllPassed": bool(
            len(s12d) == gates["s12dMetricIdentityRowsExpected"]
            and s12d["allGatesPassed"].astype(bool).all()
            and sha256_file(s12d_path) == gates["s12dArtifactSha256"]
        ),
        "sourceChecks": source_checks,
    }
    equivalence["passed"] = bool(
        equivalence["s12cAllPassed"]
        and equivalence["s12dAllPassed"]
        and all(bool(item["passed"]) for item in source_checks)
    )
    manifest = {
        "schema": "eidosoma.e01.s12i_source_snapshot_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "sourceRelationship": config["sourceRelationship"],
        "safeJsonOnly": True,
        "checks": source_checks,
        "passed": all(bool(item["passed"]) for item in source_checks),
    }
    return manifest, equivalence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-commit", required=True)
    args = parser.parse_args()
    if STEP_ROOT.exists() and any(STEP_ROOT.iterdir()):
        raise RuntimeError("S12I artifact directory must be empty before preregistration freeze")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != args.design_commit or remote != head or git("status", "--short"):
        raise RuntimeError("S12I design must be committed, pushed, and clean")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["versionedStepId"] != VERSION:
        raise RuntimeError("S12I version mismatch")
    STEP_ROOT.mkdir(parents=True)
    shutil.copyfile(CONFIG, STEP_ROOT / "preregistration.yaml")

    baseline = prior_baseline()
    if not baseline["passed"]:
        raise RuntimeError("S12I prior or cache baseline failed")
    write_json(STEP_ROOT / "immutable_prior_baseline.json", baseline)

    manifest, pairing = input_manifest_and_pairing()
    if not bool(manifest["cacheHashPassed"].all()) or not bool(
        manifest["candidateIdentityPassed"].all()
    ):
        raise RuntimeError("S12I locked input identity failed")
    if not bool(manifest["repairedReplayPassed"].all()):
        raise RuntimeError("S12FR replay evidence is not unanimous")
    manifest.to_parquet(
        STEP_ROOT / "trajectory_input_manifest.parquet", index=False, compression="zstd"
    )
    write_json(STEP_ROOT / "shared_identity_audit.json", pairing)

    s12h_classification = json.loads(S12H_CLASSIFICATION.read_text(encoding="utf-8"))
    s12h_confirmation = json.loads(S12H_CONFIRMATION.read_text(encoding="utf-8"))
    exact_waiver = validate_exact_waiver(s12h_confirmation)
    waiver_validation = {
        "schema": "eidosoma.e01.s12i_waiver_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "classificationHashPassed": sha256_file(S12H_CLASSIFICATION)
        == config["immutability"]["s12hClassificationSha256"],
        "confirmationHashPassed": sha256_file(S12H_CONFIRMATION)
        == config["immutability"]["s12hGateSha256"],
        "classificationRetained": s12h_classification.get("classification")
        == config["immutability"]["preserveS12HClassification"],
        **exact_waiver,
    }
    waiver_validation["passed"] = bool(
        waiver_validation["classificationHashPassed"]
        and waiver_validation["confirmationHashPassed"]
        and waiver_validation["classificationRetained"]
        and exact_waiver["passed"]
    )
    if not waiver_validation["passed"]:
        raise RuntimeError("S12I waiver scope validation failed")
    write_json(STEP_ROOT / "waiver_validation.json", waiver_validation)
    write_json(
        STEP_ROOT / "waiver_contract.json",
        {
            "schema": "eidosoma.e01.s12i_waiver_contract.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "authorizedOption": "option-2",
            "humanOverride": "aggregateSupportCompatible_only",
            "sourceS12hClassificationPath": str(S12H_CLASSIFICATION),
            "sourceS12hClassificationSha256": sha256_file(S12H_CLASSIFICATION),
            "sourceS12hConfirmationPath": str(S12H_CONFIRMATION),
            "sourceS12hConfirmationSha256": sha256_file(S12H_CONFIRMATION),
            "originalS12hClassification": s12h_classification["classification"],
            "originalConfirmationGatePassed": False,
            "originalAggregateSupportGatePassed": False,
            "waivedGate": "aggregateSupportCompatible",
            "waivedGateRelabeledPassed": False,
            "candidate1EvidenceStatus": "HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED",
            "candidate1UpstreamConfirmed": False,
            "otherGateWaiverCount": 0,
            "scientificEvidenceScope": "EXPLORATORY_SENSITIVITY_ONLY",
            "s13SupportPermitted": False,
            "passed": True,
        },
    )

    source_manifest, equivalence = source_evidence(config)
    if not source_manifest["passed"] or not equivalence["passed"]:
        raise RuntimeError("S12I pinned source/equivalence gate failed")
    write_json(STEP_ROOT / "source_snapshot_manifest.json", source_manifest)
    write_json(STEP_ROOT / "source_equivalence_validation.json", equivalence)

    partial = json.loads(S12G_CACHE_MANIFEST.read_text(encoding="utf-8"))
    exclusion = {
        "schema": "eidosoma.e01.s12i_s12g_cache_exclusion_audit.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "manifestPath": str(S12G_CACHE_MANIFEST),
        "manifestSha256": sha256_file(S12G_CACHE_MANIFEST),
        "recordedFileCount": partial["cacheFileCount"],
        "recordedBytes": partial["cacheBytes"],
        "baselineHashPassCount": sum(
            bool(item["passed"]) for item in baseline["s12gTaskCacheFiles"]
        ),
        "payloadFilesOpened": 0,
        "scientificReusePermitted": False,
        "resultCacheRoot": "/cache/e01_s12i/source_results",
        "passed": bool(
            sha256_file(S12G_CACHE_MANIFEST)
            == config["inputs"]["s12gCacheManifest"]["sha256"]
            and len(baseline["s12gTaskCacheFiles"]) == partial["cacheFileCount"]
            and all(bool(item["passed"]) for item in baseline["s12gTaskCacheFiles"])
        ),
    }
    if not exclusion["passed"]:
        raise RuntimeError("S12G cache exclusion baseline failed")
    write_json(STEP_ROOT / "s12g_cache_exclusion_audit.json", exclusion)

    (STEP_ROOT / "candidate_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "eidosoma.e01.s12i_candidate_registry.v1",
                "candidates": sensitivity_candidate_registry(),
                "rankingWeightsUsed": False,
                "selectionEliminationOrReweightingPermitted": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    for filename, key in (
        ("label_registry.yaml", "labels"),
        ("preprocessing_registry.yaml", "commonPreprocessing"),
        ("metric_registry.yaml", "metrics"),
    ):
        (STEP_ROOT / filename).write_text(
            yaml.safe_dump(
                {
                    "schema": f"eidosoma.e01.s12i_{filename[:-5]}.v1",
                    key: config[key],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    (STEP_ROOT / "analysis_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "eidosoma.e01.s12i_analysis_registry.v1",
                "clockAndIndexing": config["clockAndIndexing"],
                "temporalModes": config["temporalModes"],
                "statistics": config["statistics"],
                "decisionGates": config["decisionGates"],
                "classificationHierarchy": config["classificationHierarchy"],
                "classificationRules": config["classificationRules"],
                "outputSchemas": json.loads(S12G_SCHEMA.read_text(encoding="utf-8")),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    code_files = [
        CONFIG,
        S12G_SCHEMA,
        REPO / "src/e01_aggregate_support_waiver_sensitivity/core.py",
        REPO / "scripts/e01/freeze_s12i_preregistration.py",
        REPO / "scripts/e01/run_s12i_aggregate_support_waiver_sensitivity.py",
        REPO / "tests/e01/test_s12i_aggregate_support_waiver_sensitivity.py",
        REPO / "scripts/e01/run_s12g_frozen_timebase_ensemble.py",
        REPO / "src/e01_frozen_timebase_ensemble/core.py",
    ]
    lock = {
        "schema": "eidosoma.e01.s12i_method_lock.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "designCommit": head,
        "remoteCommit": remote,
        "branch": git("branch", "--show-current"),
        "files": [
            {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
            for path in code_files
        ],
        "priorImmutabilityBaselinePassed": baseline["passed"],
        "sourceEquivalencePassed": equivalence["passed"],
        "waiverScopePassed": waiver_validation["passed"],
        "inputCount": len(manifest),
        "sharedIdentityCount": pairing["sharedIdentityCount"],
        "s12gCachePayloadsOpened": 0,
        "labelOutcomesOpened": False,
        "informationTheoryOutcomesOpened": False,
        "passed": True,
    }
    write_json(STEP_ROOT / "method_lock.json", lock)
    write_json(
        STEP_ROOT / "implementation_lock.json",
        {**lock, "schema": "eidosoma.e01.s12i_implementation_lock.v1"},
    )
    write_json(
        STEP_ROOT / "preregistration_record.json",
        {
            "schema": "eidosoma.e01.s12i_preregistration_record.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "frozenAtUtc": datetime.now(UTC).isoformat(),
            "designCommit": head,
            "remoteCommit": remote,
            "preregistrationSha256": sha256_file(STEP_ROOT / "preregistration.yaml"),
            "methodLockSha256": sha256_file(STEP_ROOT / "method_lock.json"),
            "validatedBeforeScientificOutcomeAccess": True,
            "passed": True,
        },
    )
    write_json(
        STEP_ROOT / "scope_access_ledger.json",
        {
            "schema": "eidosoma.e01.s12i_scope_access_ledger.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "events": [
                {
                    "stage": "PRE_SCIENTIFIC_WAIVER_METHOD_LOCK",
                    "rawTrajectoryPayloadOpened": False,
                    "labelOutcomeOpened": False,
                    "informationTheoryOutcomeOpened": False,
                    "s12gCachePayloadOpened": False,
                    "newGardTrajectoryGenerated": False,
                    "waivedGate": "aggregateSupportCompatible",
                    "otherGateWaiverCount": 0,
                    "status": "PASS",
                }
            ],
            "success": None,
        },
    )
    print(
        json.dumps(
            {
                "stage": "S12I_preregistration_frozen",
                "designCommit": head,
                "inputCount": len(manifest),
                "priorFiles": baseline["researchStepFileCount"],
                "s12gCacheFiles": baseline["s12gTaskCacheFileCount"],
                "waivedGate": "aggregateSupportCompatible",
                "otherGateWaiverCount": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
