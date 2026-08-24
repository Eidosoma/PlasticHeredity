#!/usr/bin/env python3
"""Freeze and validate the S12G pre-outcome method and input lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S12G"
CONFIG = REPO / "configs/e01/s12g_frozen_timebase_ensemble_preregistration.yaml"
SCHEMAS = REPO / "configs/e01/s12g_output_schemas.json"
LOCK_PATH = ARTIFACTS / "research_steps/S12FR/candidate_timebase_pipeline_lock.json"
MANIFEST_PATH = ARTIFACTS / "research_steps/S12FR/confirmation_trajectory_manifest.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def prior_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted((ARTIFACTS / "research_steps").glob("S*")):
        if not path.is_dir() or path.name == "S12G":
            continue
        files.extend(item for item in sorted(path.rglob("*")) if item.is_file())
    return files


def candidate_rows(config: dict[str, Any], frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_id = {item["candidateId"]: item for item in config["candidates"]}
    for record in frame.sort_values(["candidateId", "matrixIndex"]).to_dict("records"):
        candidate = by_id[str(record["candidateId"])]
        selected_count = 1 + int(record["totalBatchUpdates"])
        if candidate["clockId"] == "C1_SELECTED_DAUGHTER_RETAINED":
            selected_count += int(record["completedFissions"])
        cache_path = Path(str(record["cachePath"]))
        actual_cache_hash = sha256_file(cache_path) if cache_path.is_file() else None
        identity_passed = bool(
            record["clockId"] == candidate["clockId"]
            and record["daughterRule"] == candidate["daughterRule"]
            and record["overshootRule"] == candidate["overshootRule"]
            and abs(float(record["h"]) - float(candidate["h"])) <= 1e-15
        )
        rows.append(
            {
                "researchStepId": "S12G",
                "candidateId": record["candidateId"],
                "matrixIndex": int(record["matrixIndex"]),
                "trajectoryId": record["trajectoryId"],
                "clockId": record["clockId"],
                "betaSha256": record["betaSha256"],
                "initialStateSha256": record["initialStateSha256"],
                "trajectorySha256": record["trajectorySha256"],
                "cachePath": record["cachePath"],
                "cacheSha256": record["cacheSha256"],
                "completedFissions": int(record["completedFissions"]),
                "totalBatchUpdates": int(record["totalBatchUpdates"]),
                "selectedObservationCount": selected_count,
                "selectedTransitionCount": selected_count - 1,
                "cacheHashPassed": actual_cache_hash == record["cacheSha256"],
                "candidateIdentityPassed": identity_passed,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-commit", required=True)
    args = parser.parse_args()
    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    schemas = json.loads(SCHEMAS.read_text(encoding="utf-8"))
    expected_candidates = {item["candidateId"] for item in config["candidates"]}
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    manifest = pd.read_parquet(MANIFEST_PATH)

    errors: list[str] = []
    if sha256_file(LOCK_PATH) != config["inputs"]["candidateLock"]["sha256"]:
        errors.append("candidate_lock_hash_mismatch")
    if sha256_file(MANIFEST_PATH) != config["inputs"]["trajectoryManifest"]["sha256"]:
        errors.append("trajectory_manifest_hash_mismatch")
    if set(manifest["candidateId"]) != expected_candidates:
        errors.append("candidate_id_set_mismatch")
    locked_candidate_ids = {
        item["candidateId"] for item in lock.get("confirmedCandidates", [])
    }
    if lock.get("confirmedCandidateCount") != 3 or locked_candidate_ids != expected_candidates:
        errors.append("candidate_lock_contents_mismatch")
    if len(manifest) != 96:
        errors.append("trajectory_count_not_96")
    counts = manifest.groupby("candidateId").size().to_dict()
    if any(counts.get(item, 0) != 32 for item in expected_candidates):
        errors.append("candidate_trajectory_count_not_32")
    if not bool((manifest["completedFissions"] == 100).all()):
        errors.append("not_all_inputs_complete_100_fissions")
    if not bool(manifest["repairedReplayPassed"].astype(bool).all()):
        errors.append("s12fr_replay_gate_not_unanimous")

    rows = candidate_rows(config, manifest)
    input_frame = pd.DataFrame(rows)
    if not bool(input_frame["cacheHashPassed"].all()):
        errors.append("one_or_more_cache_hashes_failed")
    if not bool(input_frame["candidateIdentityPassed"].all()):
        errors.append("one_or_more_candidate_identities_failed")

    shared_rows: list[dict[str, Any]] = []
    all_shared = True
    for matrix_index, group in manifest.groupby("matrixIndex", sort=True):
        beta_count = group["betaSha256"].nunique()
        initial_count = group["initialStateSha256"].nunique()
        candidate_count = group["candidateId"].nunique()
        shared = beta_count == 1 and initial_count == 1 and candidate_count == 3
        all_shared &= shared
        shared_rows.append(
            {
                "matrixIndex": int(matrix_index),
                "candidateCount": int(candidate_count),
                "uniqueBetaHashes": int(beta_count),
                "uniqueInitialStateHashes": int(initial_count),
                "sharedIdentity": bool(shared),
                "pairingPolicy": "PAIRED" if shared else "UNPAIRED",
            }
        )
    if len(shared_rows) != 32:
        errors.append("shared_identity_matrix_count_not_32")

    source_checks: list[dict[str, Any]] = []
    for implementation in (
        config["sourceImplementations"]["primary"],
        config["sourceImplementations"]["robustness"],
    ):
        checkout = (
            Path("/cache/e01_s12b/sources/IntegratedInformationGeneRegulation")
            if implementation["id"] == "IIGR_CORRECTED_SOURCE"
            else Path("/cache/e01_s12b/sources/PhiRL")
        )
        for filename, key in (("main.py", "mainSha256"), ("information.py", "informationSha256")):
            path = checkout / filename
            actual = sha256_file(path) if path.is_file() else None
            passed = actual == implementation[key]
            source_checks.append(
                {
                    "sourceId": implementation["id"],
                    "path": str(path),
                    "expectedSha256": implementation[key],
                    "actualSha256": actual,
                    "passed": passed,
                }
            )
            if not passed:
                errors.append(f"source_hash_failed:{implementation['id']}:{filename}")
    for key in ("confirmedWrapper", "emergenceWrapper"):
        item = config["sourceImplementations"][key]
        path = REPO / item["module"]
        actual = sha256_file(path)
        passed = actual == item["sha256"]
        source_checks.append(
            {
                "sourceId": key,
                "path": str(path),
                "expectedSha256": item["sha256"],
                "actualSha256": actual,
                "passed": passed,
            }
        )
        if not passed:
            errors.append(f"wrapper_hash_failed:{key}")
    safe_item = config["sourceImplementations"]["safeLattice"]
    safe_path = Path(safe_item["path"])
    safe_actual = sha256_file(safe_path)
    safe_passed = safe_actual == safe_item["sha256"]
    source_checks.append(
        {
            "sourceId": "SAFE_JSON_LATTICE",
            "path": str(safe_path),
            "expectedSha256": safe_item["sha256"],
            "actualSha256": safe_actual,
            "passed": safe_passed,
        }
    )
    if not safe_passed:
        errors.append("safe_lattice_hash_failed")

    s12c = pd.read_csv(ARTIFACTS / "research_steps/S12C/confirmation_fixture_results.csv")
    s12d_path = ARTIFACTS / "research_steps/S12D/source_metric_equivalence.csv"
    s12d = pd.read_csv(s12d_path)
    source_equivalence = {
        "schema": "eidosoma.e01.s12g_source_equivalence_validation.v1",
        "s12cRows": len(s12c),
        "s12cAllPassed": bool(
            len(s12c) == 14 and s12c["allGatesPassed"].astype(bool).all()
        ),
        "s12dRows": len(s12d),
        "s12dAllPassed": bool(
            len(s12d) == 40 and s12d["allGatesPassed"].astype(bool).all()
        ),
        "s12dArtifactSha256": sha256_file(s12d_path),
        "sourceChecks": source_checks,
    }
    source_equivalence["passed"] = bool(
        source_equivalence["s12cAllPassed"]
        and source_equivalence["s12dAllPassed"]
        and source_equivalence["s12dArtifactSha256"]
        == config["sourceImplementations"]["preOutcomeEvidenceGates"]["s12dMetricIdentityArtifactSha256"]
        and all(item["passed"] for item in source_checks)
    )
    if not source_equivalence["passed"]:
        errors.append("source_equivalence_or_metric_identity_gate_failed")

    current_head = run_git("rev-parse", "HEAD")
    remote_head = run_git("rev-parse", "origin/eidosoma/groups/42")
    if current_head != args.design_commit or remote_head != args.design_commit:
        errors.append("design_commit_not_current_and_pushed")
    if run_git("status", "--short"):
        errors.append("working_tree_not_clean_at_method_lock")

    shutil.copyfile(CONFIG, STEP_ROOT / "preregistration.yaml")
    write_json(
        STEP_ROOT / "preregistration_record.json",
        {
            "schema": "eidosoma.e01.s12g_preregistration_record.v1",
            "researchStepId": "S12G",
            "versionedStepId": config["versionedStepId"],
            "frozenAtUtc": datetime.now(timezone.utc).isoformat(),
            "configPath": str(CONFIG),
            "configSha256": sha256_file(CONFIG),
            "artifactPreregistrationSha256": sha256_file(STEP_ROOT / "preregistration.yaml"),
            "schemaContractPath": str(SCHEMAS),
            "schemaContractSha256": sha256_file(SCHEMAS),
            "designCommit": args.design_commit,
            "branch": run_git("branch", "--show-current"),
            "remoteCommit": remote_head,
            "validatedBeforeScientificOutcomeAccess": not errors,
            "errors": errors,
        },
    )
    input_frame.to_parquet(
        STEP_ROOT / "trajectory_input_manifest.parquet", index=False, compression="zstd"
    )
    write_json(
        STEP_ROOT / "shared_identity_audit.json",
        {
            "schema": "eidosoma.e01.s12g_shared_identity_audit.v1",
            "all32Shared": all_shared,
            "pairedAnalysisPermitted": all_shared,
            "rows": shared_rows,
        },
    )
    write_json(STEP_ROOT / "source_equivalence_validation.json", source_equivalence)
    write_json(
        STEP_ROOT / "source_snapshot_manifest.json",
        {
            "schema": "eidosoma.e01.s12g_source_snapshot_manifest.v1",
            "researchStepId": "S12G",
            "sourceRelationship": config["sourceRelationship"],
            "safeJsonOnly": True,
            "checks": source_checks,
            "passed": all(item["passed"] for item in source_checks),
        },
    )
    (STEP_ROOT / "candidate_registry.yaml").write_text(
        yaml.safe_dump({"schema": "eidosoma.e01.s12g_candidate_registry.v1", "candidates": config["candidates"]}, sort_keys=False),
        encoding="utf-8",
    )
    for filename, key in (
        ("label_registry.yaml", "labels"),
        ("preprocessing_registry.yaml", "commonPreprocessing"),
        ("metric_registry.yaml", "metrics"),
    ):
        (STEP_ROOT / filename).write_text(
            yaml.safe_dump({"schema": f"eidosoma.e01.s12g_{filename[:-5]}.v1", key: config[key]}, sort_keys=False),
            encoding="utf-8",
        )
    (STEP_ROOT / "analysis_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "eidosoma.e01.s12g_analysis_registry.v1",
                "clockAndIndexing": config["clockAndIndexing"],
                "temporalModes": config["temporalModes"],
                "statistics": config["statistics"],
                "decisionGates": config["decisionGates"],
                "classificationHierarchy": config["classificationHierarchy"],
                "classificationRules": config["classificationRules"],
                "outputSchemas": schemas,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    prior_inventory = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in prior_files()
    ]
    cache_inventory = [
        {
            "path": row["cachePath"],
            "bytes": Path(row["cachePath"]).stat().st_size,
            "sha256": row["cacheSha256"],
        }
        for row in rows
    ]
    write_json(
        STEP_ROOT / "immutable_prior_baseline.json",
        {
            "schema": "eidosoma.e01.s12g_immutable_prior_baseline.v1",
            "researchStepFiles": prior_inventory,
            "researchStepFileCount": len(prior_inventory),
            "lockedTrajectoryCaches": cache_inventory,
            "lockedTrajectoryCacheCount": len(cache_inventory),
        },
    )
    method_lock = {
        "schema": "eidosoma.e01.s12g_method_lock.v1",
        "researchStepId": "S12G",
        "versionedStepId": config["versionedStepId"],
        "designCommit": args.design_commit,
        "remoteCommit": remote_head,
        "configSha256": sha256_file(CONFIG),
        "schemaContractSha256": sha256_file(SCHEMAS),
        "candidateLockSha256": sha256_file(LOCK_PATH),
        "trajectoryManifestSha256": sha256_file(MANIFEST_PATH),
        "trajectoryCacheCount": len(rows),
        "sharedMatrixAndInitialIdentityCount": sum(item["sharedIdentity"] for item in shared_rows),
        "sourceEquivalencePassed": source_equivalence["passed"],
        "noLabelOrInformationTheoryOutcomeOpened": True,
        "errors": errors,
        "passed": not errors,
    }
    write_json(STEP_ROOT / "method_lock.json", method_lock)
    write_json(
        STEP_ROOT / "scope_access_ledger.json",
        {
            "schema": "eidosoma.e01.s12g_scope_access_ledger.v1",
            "researchStepId": "S12G",
            "events": [
                {
                    "stage": "PRE_OUTCOME_METHOD_LOCK",
                    "labelOutcomeOpened": False,
                    "informationTheoryOutcomeOpened": False,
                    "newGardTrajectoryGenerated": False,
                    "candidateSelectionOrReweighting": False,
                    "status": "PASS" if not errors else "FAIL",
                }
            ],
        },
    )
    if errors:
        raise SystemExit("pre-outcome method lock failed: " + ";".join(errors))
    print(json.dumps(method_lock, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
