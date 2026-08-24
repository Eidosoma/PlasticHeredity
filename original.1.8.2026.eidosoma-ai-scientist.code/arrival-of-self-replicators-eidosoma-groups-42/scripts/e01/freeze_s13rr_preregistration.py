#!/usr/bin/env python3
"""Freeze and record the S13RR second-override method before statistics."""

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
STEP_ROOT = ARTIFACTS / "research_steps/S13RR"
S13_ROOT = ARTIFACTS / "research_steps/S13"
S13R_ROOT = ARTIFACTS / "research_steps/S13R"
CACHE_ROOT = Path("/cache/e01_s13")
CONFIG = (
    REPO / "configs/e01/s13rr_downstream_schema_canonicalization_preregistration.yaml"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def freeze_prior() -> dict[str, Any]:
    root = ARTIFACTS / "research_steps"
    files = [
        path
        for step in sorted(root.iterdir())
        if step.is_dir() and step.name != "S13RR"
        for path in sorted(step.rglob("*"))
        if path.is_file()
    ]
    rows = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ]
    payload = {
        "schema": "eidosoma.e01.s13rr_immutable_prior_baseline.v1",
        "researchStepId": "S13RR",
        "scope": "every file under /artifacts/research_steps except S13RR",
        "fileCount": len(rows),
        "files": rows,
        "s13Included": any("/research_steps/S13/" in row["path"] for row in rows),
        "s13rIncluded": any("/research_steps/S13R/" in row["path"] for row in rows),
        "passed": bool(rows),
    }
    write_json(STEP_ROOT / "immutable_prior_baseline.json", payload)
    return payload


def freeze_cache() -> tuple[dict[str, Any], dict[str, Any]]:
    source_manifest_path = S13_ROOT / "source_cache_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    rows = []
    failures = []
    for row in source_manifest["cacheFiles"]:
        path = CACHE_ROOT / row["relativePath"]
        actual = sha256_file(path) if path.is_file() else None
        frozen = {
            "candidateId": row["candidateId"],
            "matrixIndex": int(row["matrixIndex"]),
            "relativePath": row["relativePath"],
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
        }
        rows.append(frozen)
        if (
            actual != row["sha256"]
            or not path.is_file()
            or path.stat().st_size != row["bytes"]
        ):
            failures.append(
                {"relativePath": row["relativePath"], "actualSha256": actual}
            )
    manifest = {
        "schema": "eidosoma.e01.s13rr_source_cache_input_manifest.v1",
        "researchStepId": "S13RR",
        "sourceManifestPath": str(source_manifest_path),
        "sourceManifestSha256": sha256_file(source_manifest_path),
        "sourceCacheRoot": str(CACHE_ROOT),
        "taskCount": int(source_manifest["completeTaskCount"]),
        "fileCount": len(rows),
        "files": rows,
        "partialS13ConcatenationIncluded": False,
        "passed": bool(source_manifest.get("passed") and len(rows) == 2000),
    }
    validation = {
        "schema": "eidosoma.e01.s13rr_source_cache_input_validation.v1",
        "researchStepId": "S13RR",
        "expectedFileCount": 2000,
        "actualFileCount": len(rows),
        "changedOrMissingCount": len(failures),
        "changedOrMissing": failures,
        "candidateStatisticsPreviouslyComputedInS13": bool(
            source_manifest["candidateStatisticsComputed"]
        ),
        "passed": bool(
            manifest["passed"]
            and not failures
            and not source_manifest["candidateStatisticsComputed"]
        ),
    }
    write_json(STEP_ROOT / "source_cache_input_manifest.json", manifest)
    write_json(STEP_ROOT / "source_cache_input_validation.json", validation)
    return manifest, validation


def validate_override(config: dict[str, Any]) -> dict[str, Any]:
    s13 = json.loads((S13_ROOT / "classification.json").read_text())
    s13r = json.loads((S13R_ROOT / "classification.json").read_text())
    downstream = json.loads(
        (S13R_ROOT / "downstream_schema_validation.json").read_text()
    )
    passed = bool(
        s13["classification"] == "S13_VALIDATION_FAILED_CLOSED"
        and s13r["classification"] == "S13R_REPAIR_PATH_PERMANENTLY_STOPPED"
        and not s13["candidateSpecificStatisticsComputed"]
        and not s13r["candidateSpecificStatisticsComputed"]
        and downstream["incompatibleTables"]
        == ["prefix.parquet", "suffix.parquet", "seeds.parquet"]
        and set(config["derivedViewContract"]["exactNewAffectedTasks"])
        == {"S12F-CANDIDATE-02/M72", "S12F-CANDIDATE-03/M72"}
    )
    payload = {
        "schema": "eidosoma.e01.s13rr_human_override.v1",
        "researchStepId": "S13RR",
        "humanOptionId": config["humanOptionId"],
        "overrideScope": "SECOND_SCHEMA_OPERATION_FOR_EXACT_MATRIX72_PREFIX_SUFFIX_SEED_VARIANTS_ONLY",
        "s13ClassificationRetained": s13["classification"],
        "s13rClassificationRetained": s13r["classification"],
        "s13CandidateStatisticsRetainedFalse": not s13[
            "candidateSpecificStatisticsComputed"
        ],
        "s13rCandidateStatisticsRetainedFalse": not s13r[
            "candidateSpecificStatisticsComputed"
        ],
        "incompatibleTables": downstream["incompatibleTables"],
        "additionalRepairPermitted": False,
        "passed": passed,
    }
    write_json(STEP_ROOT / "human_override.json", payload)
    contract = {
        "schema": "eidosoma.e01.s13rr_derived_view_contract.v1",
        "researchStepId": "S13RR",
        **config["derivedViewContract"],
        "adapterGates": config["adapterGates"],
        "frozenSourceReplayGate": config["frozenSourceReplayGate"],
        "sourceMutationPermitted": False,
        "additionalRepairPermitted": False,
        "passed": passed,
    }
    write_json(STEP_ROOT / "derived_view_contract.json", contract)
    return payload


def method_files() -> list[Path]:
    return [
        CONFIG,
        REPO / "src/e01_s13rr_downstream_schema_canonicalization/__init__.py",
        REPO / "src/e01_s13rr_downstream_schema_canonicalization/core.py",
        REPO / "scripts/e01/freeze_s13rr_preregistration.py",
        REPO / "scripts/e01/run_s13rr_downstream_schema_canonicalization.py",
        REPO / "tests/e01/test_s13rr_downstream_schema_canonicalization.py",
        REPO / "src/e01_s13r_schema_normalization/core.py",
        REPO / "scripts/e01/run_s13_confirmed_timebase_scaleup.py",
        REPO / "scripts/e01/run_s12g_frozen_timebase_ensemble.py",
        REPO / "configs/e01/s12g_output_schemas.json",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-commit", action="store_true")
    args = parser.parse_args()
    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text())
    (STEP_ROOT / "preregistration.yaml").write_text(CONFIG.read_text())
    prior = freeze_prior()
    cache, cache_validation = freeze_cache()
    override = validate_override(config)
    files = method_files()
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing S13RR method files: {missing}")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    dirty = git("status", "--short")
    if args.record_commit and (head != remote or dirty):
        raise RuntimeError("S13RR design must be committed, pushed, and clean")
    passed = bool(
        args.record_commit
        and prior["passed"]
        and cache["passed"]
        and cache_validation["passed"]
        and override["passed"]
    )
    lock = {
        "schema": "eidosoma.e01.s13rr_method_lock.v1",
        "researchStepId": "S13RR",
        "versionedStepId": config["versionedStepId"],
        "designCommit": head if args.record_commit else None,
        "remoteCommit": remote if args.record_commit else None,
        "branch": git("branch", "--show-current"),
        "candidateStatisticOpenedBeforeLock": False,
        "files": [
            {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
            for path in files
        ],
        "passed": passed,
    }
    write_json(STEP_ROOT / "method_lock.json", lock)
    write_json(
        STEP_ROOT / "preregistration_record.json",
        {
            "schema": "eidosoma.e01.s13rr_preregistration_record.v1",
            "researchStepId": "S13RR",
            "versionedStepId": config["versionedStepId"],
            "frozenAtUtc": datetime.now(timezone.utc).isoformat(),
            "designCommit": lock["designCommit"],
            "remoteCommit": lock["remoteCommit"],
            "candidateStatisticOpened": False,
            "sourceFitRerun": False,
            "newTrajectoryGenerated": False,
            "partialS13ConcatenationUsed": False,
            "passed": passed,
        },
    )
    write_json(
        STEP_ROOT / "scope_access_ledger.json",
        {
            "schema": "eidosoma.e01.s13rr_scope_access_ledger.v1",
            "researchStepId": "S13RR",
            "events": [
                {
                    "stage": "PREREGISTRATION_AND_METHOD_LOCK",
                    "candidateStatisticOpened": False,
                    "sourceFitRerun": False,
                    "newTrajectoryGenerated": False,
                    "partialS13ConcatenationUsed": False,
                    "status": "PASS" if passed else "FAIL",
                }
            ],
            "laterWorkStatus": "BLOCKED_PENDING_S13RR_HUMAN_REVIEW",
            "success": passed,
        },
    )
    print(json.dumps({"stage": "S13RR_preregistration", "passed": passed}))
    return 0 if passed or not args.record_commit else 1


if __name__ == "__main__":
    raise SystemExit(main())
