#!/usr/bin/env python3
"""Freeze the final S13RRR replay exception before candidate statistics."""

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
STEP_ROOT = ARTIFACTS / "research_steps/S13RRR"
S13_ROOT = ARTIFACTS / "research_steps/S13"
S13R_ROOT = ARTIFACTS / "research_steps/S13R"
S13RR_ROOT = ARTIFACTS / "research_steps/S13RR"
SOURCE_CACHE_ROOT = Path("/cache/e01_s13")
VIEW_ROOT = Path("/cache/e01_s13rr/canonical_views")
CONFIG = (
    REPO
    / "configs/e01/s13rrr_eligibility_aware_replay_finalization_preregistration.yaml"
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
        if step.is_dir() and step.name != "S13RRR"
        for path in sorted(step.rglob("*"))
        if path.is_file()
    ]
    rows = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ]
    payload = {
        "schema": "eidosoma.e01.s13rrr_immutable_prior_baseline.v1",
        "researchStepId": "S13RRR",
        "scope": "every file under /artifacts/research_steps except S13RRR",
        "fileCount": len(rows),
        "files": rows,
        "requiredStepsIncluded": {
            step: any(f"/research_steps/{step}/" in row["path"] for row in rows)
            for step in ("S13", "S13R", "S13RR")
        },
        "passed": bool(rows),
    }
    write_json(STEP_ROOT / "immutable_prior_baseline.json", payload)
    return payload


def freeze_manifest(
    *,
    source_manifest_path: Path,
    source_root: Path,
    file_key: str,
    expected_count: int,
    output_stem: str,
    schema_stem: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_manifest = json.loads(source_manifest_path.read_text())
    source_rows = source_manifest[file_key]
    rows: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    for row in source_rows:
        if "relativePath" in row:
            path = source_root / row["relativePath"]
            stored_path = row["relativePath"]
        else:
            path = Path(row["path"])
            stored_path = str(path)
        actual = sha256_file(path) if path.is_file() else None
        frozen = {
            key: row[key]
            for key in (
                "taskId",
                "candidateId",
                "matrixIndex",
                "family",
                "relativePath",
                "path",
            )
            if key in row
        }
        frozen.update(
            {
                "resolvedPath": str(path),
                "bytes": int(row["bytes"]),
                "sha256": row["sha256"],
            }
        )
        rows.append(frozen)
        if (
            not path.is_file()
            or actual != row["sha256"]
            or path.stat().st_size != row["bytes"]
        ):
            changed.append(
                {
                    "path": stored_path,
                    "expectedSha256": row["sha256"],
                    "actualSha256": actual,
                }
            )
    manifest = {
        "schema": f"eidosoma.e01.{schema_stem}_input_manifest.v1",
        "researchStepId": "S13RRR",
        "sourceManifestPath": str(source_manifest_path),
        "sourceManifestSha256": sha256_file(source_manifest_path),
        "sourceRoot": str(source_root),
        "fileCount": len(rows),
        "files": rows,
        "passed": len(rows) == expected_count and not changed,
    }
    validation = {
        "schema": f"eidosoma.e01.{schema_stem}_input_validation.v1",
        "researchStepId": "S13RRR",
        "expectedFileCount": expected_count,
        "checkedFileCount": len(rows),
        "changedOrMissingCount": len(changed),
        "changedOrMissing": changed,
        "passed": manifest["passed"],
    }
    write_json(STEP_ROOT / f"{output_stem}_input_manifest.json", manifest)
    write_json(STEP_ROOT / f"{output_stem}_input_validation.json", validation)
    return manifest, validation


def validate_override(config: dict[str, Any]) -> dict[str, Any]:
    classifications = {
        "S13": json.loads((S13_ROOT / "classification.json").read_text()),
        "S13R": json.loads((S13R_ROOT / "classification.json").read_text()),
        "S13RR": json.loads((S13RR_ROOT / "classification.json").read_text()),
    }
    replay = json.loads((S13RR_ROOT / "source_replay_gate_validation.json").read_text())
    canonical = json.loads(
        (S13RR_ROOT / "canonicalization_validation.json").read_text()
    )
    collation = json.loads(
        (S13RR_ROOT / "strict_collation_validation.json").read_text()
    )
    expected = config["immutability"]["historicalClassifications"]
    passed = bool(
        all(
            classifications[step]["classification"] == token
            for step, token in expected.items()
        )
        and not classifications["S13"]["candidateSpecificStatisticsComputed"]
        and not classifications["S13R"]["candidateSpecificStatisticsComputed"]
        and not classifications["S13RR"]["candidateSpecificStatisticsComputed"]
        and replay["observedExecutedSuffixCount"] == 3552
        and replay["frozenExpectedExecutedSuffixCount"] == 3600
        and not replay["checks"]["executedSuffixExactCount"]
        and all(
            value
            for key, value in replay["checks"].items()
            if key != "executedSuffixExactCount"
        )
        and canonical["passed"]
        and canonical["taskPassCount"] == canonical["taskViewCount"] == 1600
        and collation["passed"]
        and collation["onePhysicalSchemaFamilyCount"] == 8
    )
    payload = {
        "schema": "eidosoma.e01.s13rrr_human_override.v1",
        "researchStepId": "S13RRR",
        "humanOptionId": config["humanOptionId"],
        "overrideOrdinal": 3,
        "overrideScope": "EXACT_3552_AVAILABILITY_GATE_AND_TWO_VALUE_PRESERVING_REPORTING_ORDERS_ONLY",
        "historicalClassificationsRetained": expected,
        "s13rrObservedExecutedSentinels": replay["observedExecutedSuffixCount"],
        "s13rrOnlyFailedReplayCheck": [
            key for key, value in replay["checks"].items() if not value
        ],
        "s13rrCanonicalViewsPassing": canonical["passed"],
        "s13rrStrictCollationPassing": collation["passed"],
        "additionalRepairPermitted": False,
        "passed": passed,
    }
    write_json(STEP_ROOT / "human_override.json", payload)
    write_json(
        STEP_ROOT / "replay_rule_contract.json",
        {
            "schema": "eidosoma.e01.s13rrr_replay_rule_contract.v1",
            "researchStepId": "S13RRR",
            **config["replayRuleOverride"],
            "allOtherSourceGates": config["unchangedSourceGates"],
            "additionalChangePermitted": False,
            "passed": passed,
        },
    )
    write_json(
        STEP_ROOT / "reporting_order_contract.json",
        {
            "schema": "eidosoma.e01.s13rrr_reporting_order_contract.v1",
            "researchStepId": "S13RRR",
            **config["reportingOrderContract"],
            "additionalChangePermitted": False,
            "passed": passed,
        },
    )
    return payload


def method_files() -> list[Path]:
    return [
        CONFIG,
        REPO / "src/e01_s13rrr_eligibility_aware_replay/__init__.py",
        REPO / "src/e01_s13rrr_eligibility_aware_replay/core.py",
        REPO / "scripts/e01/freeze_s13rrr_preregistration.py",
        REPO / "scripts/e01/run_s13rrr_eligibility_aware_replay_finalization.py",
        REPO / "tests/e01/test_s13rrr_eligibility_aware_replay.py",
        REPO / "scripts/e01/run_s13_confirmed_timebase_scaleup.py",
        REPO / "scripts/e01/run_s13rr_downstream_schema_canonicalization.py",
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
    cache, cache_validation = freeze_manifest(
        source_manifest_path=S13_ROOT / "source_cache_manifest.json",
        source_root=SOURCE_CACHE_ROOT,
        file_key="cacheFiles",
        expected_count=2000,
        output_stem="source_cache",
        schema_stem="s13rrr_source_cache",
    )
    views, views_validation = freeze_manifest(
        source_manifest_path=S13RR_ROOT / "derived_view_manifest.json",
        source_root=VIEW_ROOT,
        file_key="files",
        expected_count=1600,
        output_stem="canonical_view",
        schema_stem="s13rrr_canonical_view",
    )
    override = validate_override(config)
    files = method_files()
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing S13RRR method files: {missing}")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    dirty = git("status", "--short")
    if args.record_commit and (head != remote or dirty):
        raise RuntimeError("S13RRR design must be committed, pushed, and clean")
    passed = bool(
        args.record_commit
        and prior["passed"]
        and cache["passed"]
        and cache_validation["passed"]
        and views["passed"]
        and views_validation["passed"]
        and override["passed"]
    )
    lock = {
        "schema": "eidosoma.e01.s13rrr_method_lock.v1",
        "researchStepId": "S13RRR",
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
            "schema": "eidosoma.e01.s13rrr_preregistration_record.v1",
            "researchStepId": "S13RRR",
            "versionedStepId": config["versionedStepId"],
            "frozenAtUtc": datetime.now(timezone.utc).isoformat(),
            "designCommit": lock["designCommit"],
            "remoteCommit": lock["remoteCommit"],
            "candidateStatisticOpened": False,
            "sourceFitRerun": False,
            "newTrajectoryGenerated": False,
            "subsetAnalysisUsed": False,
            "passed": passed,
        },
    )
    write_json(
        STEP_ROOT / "scope_access_ledger.json",
        {
            "schema": "eidosoma.e01.s13rrr_scope_access_ledger.v1",
            "researchStepId": "S13RRR",
            "events": [
                {
                    "stage": "PREREGISTRATION_AND_METHOD_LOCK",
                    "candidateStatisticOpened": False,
                    "sourceFitRerun": False,
                    "newTrajectoryGenerated": False,
                    "subsetAnalysisUsed": False,
                    "status": "PASS" if passed else "FAIL",
                }
            ],
            "laterWorkStatus": "BLOCKED_PENDING_S13RRR_HUMAN_REVIEW",
            "success": passed,
        },
    )
    print(json.dumps({"stage": "S13RRR_preregistration", "passed": passed}))
    return 0 if passed or not args.record_commit else 1


if __name__ == "__main__":
    raise SystemExit(main())
