#!/usr/bin/env python3
"""Freeze S12FR comparator contract and immutable inputs before any rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
ARTIFACTS = Path("/artifacts/research_steps/S12FR")
CONFIG = REPO / "configs/e01/s12fr_replay_comparator_repair_preregistration.yaml"
CONTRACT = REPO / "configs/e01/s12fr/comparator_contract.yaml"
PAIR_SCHEMA = REPO / "configs/e01/s12fr/pair_diagnostic_schema.json"
AMENDMENT = REPO / "configs/e01/s12fr/preregistration_amendment_v1.0.1.yaml"
S12F_ARTIFACTS = Path("/artifacts/research_steps/S12F")
S12F_CACHE = Path("/cache/e01_s12f")
S12F_BASELINE = S12F_ARTIFACTS / "immutable_prior_baseline.json"
EXPECTED_BRANCH = "eidosoma/groups/42"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_sha(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["path"]):
        digest.update(str(row["path"]).encode())
        digest.update(b"\0")
        digest.update(str(row["sizeBytes"]).encode())
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def file_row(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def validate_manifest(payload: dict[str, Any]) -> None:
    failures = []
    for row in payload["files"]:
        path = Path(row["path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
        elif path.stat().st_size != row["sizeBytes"] or sha256(path) != row["sha256"]:
            failures.append({"path": str(path), "reason": "identity_changed"})
    if failures:
        raise RuntimeError(f"immutable manifest validation failed: {failures[:3]}")


def prior_baseline() -> dict[str, Any]:
    prior = json.loads(S12F_BASELINE.read_text(encoding="utf-8"))
    research_rows = [
        dict(row)
        for row in prior["files"]
        if not str(row["path"]).startswith("/artifacts/logs/")
    ]
    exclusions = []
    for row in prior["files"]:
        if not str(row["path"]).startswith("/artifacts/logs/"):
            continue
        path = Path(row["path"])
        exclusions.append(
            {
                "path": str(path),
                "reason": "LIVE_PLATFORM_LOG_NOT_RESEARCH_EVIDENCE",
                "s12fRecordedSizeBytes": row["sizeBytes"],
                "s12fRecordedSha256": row["sha256"],
                "currentSizeBytes": path.stat().st_size if path.is_file() else None,
                "currentSha256": sha256(path) if path.is_file() else None,
                "preexistingMismatchDetectedBeforeAnyS12FRSimulatorRerun": True,
            }
        )
    validate_manifest({"files": research_rows})
    rows = research_rows
    known = {row["path"] for row in rows}
    for path in sorted(S12F_ARTIFACTS.rglob("*")):
        if path.is_file() and str(path) not in known:
            rows.append(file_row(path))
    return {
        "schemaVersion": "E01-S12FR-immutable-prior-baseline-v1.0.0",
        "researchStepId": "E01-S12FR-EXACT-REPLAY-COMPARATOR-REPAIR-v1.0.0",
        "capturedAtUtc": datetime.now(UTC).isoformat(),
        "sourceBaseline": str(S12F_BASELINE),
        "baselineExclusions": exclusions,
        "fileCount": len(rows),
        "aggregateSha256": aggregate_sha(rows),
        "files": sorted(rows, key=lambda value: value["path"]),
    }


def directory_manifest(root: Path, schema: str) -> dict[str, Any]:
    rows = [file_row(path) for path in sorted(root.rglob("*")) if path.is_file()]
    return {
        "schemaVersion": schema,
        "researchStepId": "E01-S12FR-EXACT-REPLAY-COMPARATOR-REPAIR-v1.0.0",
        "capturedAtUtc": datetime.now(UTC).isoformat(),
        "root": str(root),
        "fileCount": len(rows),
        "aggregateSha256": aggregate_sha(rows),
        "files": rows,
    }


def source_manifest() -> dict[str, Any]:
    paths = [
        CONFIG,
        CONTRACT,
        PAIR_SCHEMA,
        REPO / "src/e01_replay_repair/__init__.py",
        REPO / "src/e01_replay_repair/comparator.py",
        REPO / "src/e01_replay_repair/audit.py",
        REPO / "src/e01_replay_repair/campaign.py",
        REPO / "scripts/e01/freeze_s12fr_preregistration.py",
        REPO / "scripts/e01/run_s12fr_replay_repair.py",
        REPO / "tests/e01/test_s12fr_replay_repair.py",
        REPO / "configs/e01/s12f_latent_timebase_preregistration.yaml",
        REPO / "src/e01_latent_timebase/core.py",
        REPO / "src/e01_latent_timebase/inference.py",
        REPO / "scripts/e01/run_s12f_latent_timebase.py",
        Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf"),
        Path("/artifacts/research_steps/S12B/safe_phi_lattice.json"),
    ]
    rows = [file_row(path) for path in paths]
    source_repositories = [
        ("historicalGard", Path("/cache/e01_s03/sources/gard-historical"), "86dff6320d5ae91b4e831471079ff46749b14df9"),
        ("iigr", Path("/cache/e01_s12b/sources/IntegratedInformationGeneRegulation"), "7c1c22fe39f539d4a453135476f1f0dd5a6b45f7"),
        ("phirl", Path("/cache/e01_s12b/sources/PhiRL"), "a6d1d0d18c7551302724b7158c6ccdc4d3a33373"),
    ]
    repositories = []
    for identifier, path, expected in source_repositories:
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD^{commit}"], text=True
        ).strip()
        if actual != expected:
            raise RuntimeError(f"{identifier} source identity changed")
        repositories.append({"id": identifier, "path": str(path), "commit": actual})
    return {
        "schemaVersion": "E01-S12FR-source-input-snapshot-manifest-v1.0.0",
        "researchStepId": "E01-S12FR-EXACT-REPLAY-COMPARATOR-REPAIR-v1.0.0",
        "capturedAtUtc": datetime.now(UTC).isoformat(),
        "files": rows,
        "aggregateSha256": aggregate_sha(rows),
        "repositories": repositories,
        "s12fSourceCommit": "cf5b27b370a2d8d12e6867034d6ec8f4f96b3fc7",
    }


def suppressed_manifest() -> dict[str, Any]:
    names = [
        "abc_particle_results.parquet",
        "abc_round_summary.csv",
        "posterior_candidates.csv",
        "posterior_predictive_results.csv",
        "candidate_lock_proposal.json",
        "candidate_timebase_pipeline_lock.json",
        "development_seed_manifest.parquet",
        "failure_ledger.csv",
        "status.json",
        "classification.json",
        "artifact_manifest.json",
        "research_step_full_results.md",
    ]
    rows = [file_row(S12F_ARTIFACTS / name) for name in names]
    particle = pd.read_parquet(S12F_ARTIFACTS / "abc_particle_results.parquet")
    if particle.shape[0] != 256 or bool(particle["scientificOutcomeInspected"].any()):
        raise RuntimeError("S12F suppressed particle identity contract changed")
    if bool(particle["distance"].notna().any()) or bool(particle["posteriorWeight"].notna().any()):
        raise RuntimeError("previously suppressed S12F distances were populated")
    return {
        "schemaVersion": "E01-S12FR-s12f-suppressed-input-manifest-v1.0.0",
        "researchStepId": "E01-S12FR-EXACT-REPLAY-COMPARATOR-REPAIR-v1.0.0",
        "capturedAtUtc": datetime.now(UTC).isoformat(),
        "files": rows,
        "aggregateSha256": aggregate_sha(rows),
        "particleRowCount": int(particle.shape[0]),
        "allDistancesSuppressed": True,
        "allPosteriorWeightsSuppressed": True,
        "scientificOutcomeInspectedCount": 0,
        "accessRule": "identity_schema_and_suppression_status_only; numeric_distance_columns_remain_unopened",
    }


def schema_smoke() -> None:
    schema = json.loads(PAIR_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    digest = "0" * 64
    sample = {
        "campaign": "SCHEMA_ONLY",
        "pairId": "PAIR-0",
        "particleId": "P0",
        "matrixIndex": 0,
        "oldComparatorPassed": False,
        "repairedComparatorPassed": True,
        "seedIdentityPassed": True,
        "rngConsumptionPassed": True,
        "instrumentationParityPassed": True,
        "pairGatePassed": True,
        "betaSha256Left": digest,
        "betaSha256Right": digest,
        "initialStateSha256Left": digest,
        "initialStateSha256Right": digest,
        "trajectorySha256Left": digest,
        "trajectorySha256Right": digest,
        "discreteDivergenceCount": 0,
        "finiteNumericDivergenceCount": 0,
        "permittedPairedNanCount": 2,
        "forbiddenNonfiniteDifferenceCount": 0,
        "rngDivergenceCount": 0,
        "tracePayloadPath": "/cache/schema-only.npz",
        "tracePayloadSha256": digest,
    }
    jsonschema.validate(sample, schema)


def initial_freeze() -> None:
    partial_recovery = False
    partial_copy_hashes: dict[str, str] = {}
    if ARTIFACTS.exists() and any(ARTIFACTS.iterdir()):
        allowed = {
            "preregistration.yaml": CONFIG,
            "comparator_contract.yaml": CONTRACT,
            "pair_diagnostic_schema.json": PAIR_SCHEMA,
        }
        present = {path.name for path in ARTIFACTS.iterdir() if path.is_file()}
        if present != set(allowed):
            raise RuntimeError("unexpected files exist after failed S12FR initial freeze")
        partial_copy_hashes = {
            name: sha256(ARTIFACTS / name) for name in sorted(allowed)
        }
        partial_recovery = True
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if config["researchStepId"] != contract["researchStepId"]:
        raise RuntimeError("S12FR contract/config step mismatch")
    schema_smoke()
    shutil.copy2(CONFIG, ARTIFACTS / "preregistration.yaml")
    shutil.copy2(CONTRACT, ARTIFACTS / "comparator_contract.yaml")
    shutil.copy2(PAIR_SCHEMA, ARTIFACTS / "pair_diagnostic_schema.json")
    prior = prior_baseline()
    cache = directory_manifest(S12F_CACHE, "E01-S12FR-s12f-cache-baseline-v1.0.0")
    source = source_manifest()
    suppressed = suppressed_manifest()
    write_json(ARTIFACTS / "immutable_prior_baseline.json", prior)
    write_json(ARTIFACTS / "s12f_cache_baseline.json", cache)
    write_json(ARTIFACTS / "source_input_snapshot_manifest.json", source)
    write_json(ARTIFACTS / "s12f_suppressed_input_manifest.json", suppressed)
    record = {
        "schemaVersion": "E01-S12FR-preregistration-record-v1.0.0",
        "researchStepId": config["researchStepId"],
        "frozenAtUtc": datetime.now(UTC).isoformat(),
        "branch": git("branch", "--show-current"),
        "gitCommitBeforeDesignCommit": git("rev-parse", "HEAD^{commit}"),
        "gitCommit": None,
        "remoteCommit": None,
        "commitRecordedAfterPush": False,
        "headMatchesRemote": False,
        "configSha256": sha256(CONFIG),
        "contractSha256": sha256(CONTRACT),
        "pairSchemaSha256": sha256(PAIR_SCHEMA),
        "sourceImplementationHashes": {
            "comparator": sha256(REPO / "src/e01_replay_repair/comparator.py"),
            "audit": sha256(REPO / "src/e01_replay_repair/audit.py"),
            "campaign": sha256(REPO / "src/e01_replay_repair/campaign.py"),
            "runner": sha256(REPO / "scripts/e01/run_s12fr_replay_repair.py"),
        },
        "workspacePlanSha256": sha256(WORKSPACE / "RESEARCH_PLAN.md"),
        "priorAggregateSha256": prior["aggregateSha256"],
        "s12fCacheAggregateSha256": cache["aggregateSha256"],
        "suppressedAggregateSha256": suppressed["aggregateSha256"],
        "simulatorRerunOccurred": False,
        "initialFreezeAttemptHistory": [
            {
                "status": "FAILED_BEFORE_BASELINE_WRITE",
                "reason": "S12F_broad_baseline_included_mutating_live_platform_stderr_log",
                "simulatorRerunOccurred": False,
                "recovery": "excluded_only_/artifacts/logs_live_platform_files_from_research_evidence_baseline",
                "partialCopyHashesBeforeFinalFreeze": partial_copy_hashes,
            },
            {
                "status": "FAILED_PARTIAL_RECOVERY_COPY_IDENTITY_CHECK",
                "reason": "preoutcome_config_was_updated_to_document_live_platform_log_policy_after_first_failure",
                "simulatorRerunOccurred": False,
                "recovery": "overwrite_only_three_partial_preregistration_copies_with_final_preoutcome_design",
            }
        ]
        if partial_recovery
        else [],
    }
    write_json(ARTIFACTS / "preregistration_record.json", record)


def record_commit() -> None:
    record_path = ARTIFACTS / "preregistration_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["commitRecordedAfterPush"]:
        raise RuntimeError("S12FR design commit was already recorded")
    if git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("wrong git branch")
    head = git("rev-parse", "HEAD^{commit}")
    remote = git("rev-parse", f"origin/{EXPECTED_BRANCH}^{{commit}}")
    if head != remote:
        raise RuntimeError("S12FR preregistration commit is not pushed")
    for path, expected in (
        (CONFIG, record["configSha256"]),
        (CONTRACT, record["contractSha256"]),
        (PAIR_SCHEMA, record["pairSchemaSha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen design file changed: {path}")
    for name, expected in record["sourceImplementationHashes"].items():
        path = {
            "comparator": REPO / "src/e01_replay_repair/comparator.py",
            "audit": REPO / "src/e01_replay_repair/audit.py",
            "campaign": REPO / "src/e01_replay_repair/campaign.py",
            "runner": REPO / "scripts/e01/run_s12fr_replay_repair.py",
        }[name]
        if sha256(path) != expected:
            raise RuntimeError(f"frozen S12FR implementation changed: {name}")
    record.update(
        {
            "gitCommit": head,
            "remoteCommit": remote,
            "commitRecordedAfterPush": True,
            "headMatchesRemote": True,
            "commitRecordedAtUtc": datetime.now(UTC).isoformat(),
        }
    )
    write_json(record_path, record)


def record_preoutcome_amendment() -> None:
    record_path = ARTIFACTS / "preregistration_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("preOutcomeAmendment") is not None:
        raise RuntimeError("S12FR pre-outcome amendment was already recorded")
    if (ARTIFACTS / "original_pair_diagnostics.parquet").exists():
        raise RuntimeError("pair outcomes already exist; pre-outcome amendment forbidden")
    amendment = yaml.safe_load(AMENDMENT.read_text(encoding="utf-8"))
    if amendment["trigger"]["simulatorRerunOccurred"]:
        raise RuntimeError("amendment is not pre-simulation")
    head = git("rev-parse", "HEAD^{commit}")
    remote = git("rev-parse", f"origin/{EXPECTED_BRANCH}^{{commit}}")
    if head != remote:
        raise RuntimeError("S12FR amendment commit is not pushed")
    shutil.copy2(AMENDMENT, ARTIFACTS / AMENDMENT.name)
    record["preOutcomeAmendment"] = {
        "schemaVersion": amendment["schemaVersion"],
        "recordedAtUtc": datetime.now(UTC).isoformat(),
        "commit": head,
        "remoteCommit": remote,
        "amendmentPath": str(AMENDMENT),
        "amendmentSha256": sha256(AMENDMENT),
        "runnerSha256": sha256(REPO / "scripts/e01/run_s12fr_replay_repair.py"),
        "simulatorRerunOccurredBeforeAmendment": False,
        "pairOutcomeInspectedBeforeAmendment": False,
    }
    record.setdefault("preOutcomeAttemptHistory", []).append(
        {
            "status": "STOPPED_BEFORE_SIMULATOR_TASK_CONSTRUCTION",
            "reason": amendment["trigger"]["failure"],
            "simulatorRerunOccurred": False,
            "outcomeInspected": False,
        }
    )
    write_json(record_path, record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-commit", action="store_true")
    parser.add_argument("--record-preoutcome-amendment", action="store_true")
    arguments = parser.parse_args()
    if arguments.record_preoutcome_amendment:
        record_preoutcome_amendment()
    elif arguments.record_commit:
        record_commit()
    else:
        initial_freeze()


if __name__ == "__main__":
    main()
