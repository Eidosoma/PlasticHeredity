#!/usr/bin/env python3
"""Freeze the one-repair-only S13R design before candidate statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S13R"
S13_ROOT = ARTIFACTS / "research_steps/S13"
SOURCE_CACHE_ROOT = Path("/cache/e01_s13")
CONFIG = REPO / "configs/e01/s13r_schema_normalization_confirmation_preregistration.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def prior_artifact_files() -> list[Path]:
    root = ARTIFACTS / "research_steps"
    paths: list[Path] = []
    for step in sorted(root.iterdir()):
        if step.is_dir() and step.name != "S13R":
            paths.extend(path for path in sorted(step.rglob("*")) if path.is_file())
    return paths


def freeze_prior_artifacts() -> dict[str, Any]:
    rows = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in prior_artifact_files()
    ]
    payload = {
        "schema": "eidosoma.e01.s13r_immutable_prior_baseline.v1",
        "researchStepId": "S13R",
        "scope": "every file under /artifacts/research_steps except S13R",
        "fileCount": len(rows),
        "files": rows,
        "s13Included": any("/research_steps/S13/" in row["path"] for row in rows),
        "passed": bool(rows),
    }
    write_json(STEP_ROOT / "immutable_prior_baseline.json", payload)
    return payload


def freeze_source_cache() -> tuple[dict[str, Any], dict[str, Any]]:
    s13_manifest_path = S13_ROOT / "source_cache_manifest.json"
    s13_manifest = json.loads(s13_manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in s13_manifest["cacheFiles"]:
        path = SOURCE_CACHE_ROOT / row["relativePath"]
        actual = sha256_file(path) if path.is_file() else None
        frozen = {
            "candidateId": row["candidateId"],
            "matrixIndex": int(row["matrixIndex"]),
            "relativePath": row["relativePath"],
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
        }
        rows.append(frozen)
        if actual != row["sha256"] or (path.is_file() and path.stat().st_size != row["bytes"]):
            failures.append(
                {
                    "relativePath": row["relativePath"],
                    "expectedSha256": row["sha256"],
                    "actualSha256": actual,
                }
            )
    manifest = {
        "schema": "eidosoma.e01.s13r_source_cache_input_manifest.v1",
        "researchStepId": "S13R",
        "sourceManifestPath": str(s13_manifest_path),
        "sourceManifestSha256": sha256_file(s13_manifest_path),
        "sourceCacheRoot": str(SOURCE_CACHE_ROOT),
        "taskCount": int(s13_manifest["completeTaskCount"]),
        "fileCount": len(rows),
        "files": rows,
        "partialS13ConcatenationIncluded": False,
        "passed": bool(s13_manifest.get("passed") and len(rows) == 2000),
    }
    validation = {
        "schema": "eidosoma.e01.s13r_source_cache_input_validation.v1",
        "researchStepId": "S13R",
        "expectedFileCount": 2000,
        "actualFileCount": len(rows),
        "changedOrMissingCount": len(failures),
        "changedOrMissing": failures,
        "allS13TaskReplayFlagsRetained": bool(
            s13_manifest["allFullReplayPassed"]
            and s13_manifest["allPrefixReplayPassed"]
            and s13_manifest["allSuffixPassed"]
            and s13_manifest["taskFailureRows"] == 0
        ),
        "candidateStatisticsPreviouslyComputed": bool(
            s13_manifest["candidateStatisticsComputed"]
        ),
        "passed": bool(
            manifest["passed"]
            and not failures
            and not s13_manifest["candidateStatisticsComputed"]
        ),
    }
    write_json(STEP_ROOT / "source_cache_input_manifest.json", manifest)
    write_json(STEP_ROOT / "source_cache_input_validation.json", validation)
    return manifest, validation


def validate_human_override(config: dict[str, Any]) -> dict[str, Any]:
    diagnostic_path = S13_ROOT / "source_schema_failure_diagnostics.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    classification_path = S13_ROOT / "classification.json"
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    variants = {
        (
            row["clusterIdType"],
            row["referenceObservationIdType"],
            row["metricToReferenceType"],
        ): int(row["taskCount"])
        for row in diagnostic["labelSchemaVariants"]
    }
    passed = bool(
        diagnostic["failureToken"] == "SOURCE_LABEL_PARQUET_SCHEMA_MISMATCH"
        and diagnostic["taskCount"] == 200
        and variants.get(("null", "null", "null")) == 5
        and variants.get(("string", "string", "double")) == 195
        and classification["classification"] == "S13_VALIDATION_FAILED_CLOSED"
        and not classification["candidateSpecificStatisticsComputed"]
        and config["adapter"]["expectedAdaptedTaskCount"] == 5
    )
    payload = {
        "schema": "eidosoma.e01.s13r_human_override.v1",
        "researchStepId": "S13R",
        "humanOptionId": config["humanOptionId"],
        "overrideScope": "S13_NO_REPAIR_RULE_FOR_EXACT_THREE_FIELD_LABEL_PHYSICAL_TYPE_ADAPTER_ONLY",
        "s13ClassificationPath": str(classification_path),
        "s13ClassificationSha256": sha256_file(classification_path),
        "s13ClassificationRetained": classification["classification"],
        "s13ScientificAssociationRetained": classification[
            "scientificAssociationClassification"
        ],
        "schemaFailureDiagnosticsPath": str(diagnostic_path),
        "schemaFailureDiagnosticsSha256": sha256_file(diagnostic_path),
        "permittedFields": config["adapter"]["exactPermittedFields"],
        "expectedTypedTaskCount": 195,
        "expectedNullTypedTaskCount": 5,
        "sourceCalculationRerunPermitted": False,
        "anotherRepairPermitted": False,
        "passed": passed,
    }
    write_json(STEP_ROOT / "human_override.json", payload)
    write_json(
        STEP_ROOT / "adapter_contract.json",
        {
            "schema": "eidosoma.e01.s13r_adapter_contract.v1",
            "researchStepId": "S13R",
            "adapterId": config["adapter"]["identifier"],
            "sourceTable": config["adapter"]["sourceTable"],
            "exactPermittedFields": config["adapter"]["exactPermittedFields"],
            "exactAffectedTasks": config["adapter"]["exactAffectedTasks"],
            "gates": config["adapterGates"],
            "sourceMutationPermitted": False,
            "genericSchemaPromotionPermitted": False,
            "furtherSchemaRepairPermitted": False,
            "passed": passed,
        },
    )
    return payload


def method_files() -> list[Path]:
    return [
        CONFIG,
        REPO / "src/e01_s13r_schema_normalization/__init__.py",
        REPO / "src/e01_s13r_schema_normalization/core.py",
        REPO / "scripts/e01/freeze_s13r_preregistration.py",
        REPO / "scripts/e01/run_s13r_schema_normalization_confirmation.py",
        REPO / "tests/e01/test_s13r_schema_normalization_confirmation.py",
        REPO / "src/e01_confirmed_timebase_scaleup/core.py",
        REPO / "scripts/e01/run_s13_confirmed_timebase_scaleup.py",
        REPO / "scripts/e01/run_s12g_frozen_timebase_ensemble.py",
        REPO / "configs/e01/s12g_output_schemas.json",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-commit", action="store_true")
    args = parser.parse_args()
    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    (STEP_ROOT / "preregistration.yaml").write_text(
        CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
    )
    prior = freeze_prior_artifacts()
    cache_manifest, cache_validation = freeze_source_cache()
    override = validate_human_override(config)
    files = method_files()
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"S13R method files missing: {missing}")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    dirty = git("status", "--short")
    if args.record_commit and (head != remote or dirty):
        raise RuntimeError("S13R design must be committed, pushed, and clean")
    lock = {
        "schema": "eidosoma.e01.s13r_method_lock.v1",
        "researchStepId": "S13R",
        "versionedStepId": config["versionedStepId"],
        "designCommit": head if args.record_commit else None,
        "remoteCommit": remote if args.record_commit else None,
        "branch": git("branch", "--show-current"),
        "candidateStatisticOpenedBeforeLock": False,
        "sourceValueInspectedForScientificOutcomeBeforeLock": False,
        "files": [
            {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
            for path in files
        ],
        "priorImmutabilityBaselinePassed": prior["passed"],
        "sourceCacheManifestPassed": cache_manifest["passed"],
        "sourceCacheValidationPassed": cache_validation["passed"],
        "humanOverrideContractPassed": override["passed"],
        "passed": bool(
            args.record_commit
            and prior["passed"]
            and cache_manifest["passed"]
            and cache_validation["passed"]
            and override["passed"]
        ),
    }
    write_json(STEP_ROOT / "method_lock.json", lock)
    record = {
        "schema": "eidosoma.e01.s13r_preregistration_record.v1",
        "researchStepId": "S13R",
        "versionedStepId": config["versionedStepId"],
        "frozenAtUtc": datetime.now(timezone.utc).isoformat(),
        "designCommit": lock["designCommit"],
        "remoteCommit": lock["remoteCommit"],
        "candidateStatisticOpened": False,
        "sourceFitRerun": False,
        "newTrajectoryGenerated": False,
        "partialS13ConcatenationUsed": False,
        "passed": lock["passed"],
    }
    write_json(STEP_ROOT / "preregistration_record.json", record)
    write_json(
        STEP_ROOT / "scope_access_ledger.json",
        {
            "schema": "eidosoma.e01.s13r_scope_access_ledger.v1",
            "researchStepId": "S13R",
            "events": [
                {
                    "stage": "PREREGISTRATION_AND_METHOD_LOCK",
                    "candidateStatisticOpened": False,
                    "sourceFitRerun": False,
                    "newTrajectoryGenerated": False,
                    "partialS13ConcatenationUsed": False,
                    "status": "PASS" if lock["passed"] else "FAIL",
                }
            ],
            "laterWorkStatus": "BLOCKED_PENDING_S13R_HUMAN_REVIEW",
            "success": lock["passed"],
        },
    )
    print(json.dumps({"stage": "S13R_preregistration", "passed": lock["passed"]}))
    return 0 if lock["passed"] or not args.record_commit else 1


if __name__ == "__main__":
    raise SystemExit(main())
