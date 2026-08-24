#!/usr/bin/env python3
"""Freeze S12J's one-column adapter and inherited statistics before access."""

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

import pyarrow.parquet as pq
import yaml

from e01_aggregation_interface_repair.core import (
    ADAPTER_ID,
    DERIVED_FIELD,
    ENDPOINT_JOIN_KEYS,
    EVIDENCE_CLASS,
    MONOTONIC_GROUP_KEYS,
    RESEARCH_STEP_ID,
    SOURCE_FIELD,
    UNIQUE_KEYS,
    VERSION,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S12J"
S12I_ROOT = ARTIFACTS / "research_steps/S12I"
CONFIG = REPO / "configs/e01/s12j_aggregation_interface_repair_confirmation_preregistration.yaml"


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


def immutable_prior_baseline() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for root in sorted((ARTIFACTS / "research_steps").glob("S*")):
        if root.name == RESEARCH_STEP_ID or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append(
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return {
        "schema": "eidosoma.e01.s12j_immutable_prior_baseline.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "createdAtUtc": datetime.now(UTC).isoformat(),
        "fileCount": len(rows),
        "files": rows,
        "passed": bool(rows),
    }


def validate_s12i_inputs(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(
        config["inputs"]["validationEvidence"]["s12iArtifactManifest"]["path"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hashes = {
        item["relativePath"]: item["sha256"] for item in manifest["artifacts"]
    }
    scientific_rows: list[dict[str, Any]] = []
    for input_id, item in config["inputs"]["scientificTables"].items():
        path = Path(item["path"])
        actual = sha256_file(path) if path.is_file() else None
        metadata = pq.ParquetFile(path).metadata if path.is_file() else None
        relative = str(path.relative_to(S12I_ROOT)) if path.is_file() else None
        scientific_rows.append(
            {
                "inputId": input_id,
                "path": str(path),
                "expectedSha256": item["sha256"],
                "actualSha256": actual,
                "manifestSha256": manifest_hashes.get(relative),
                "expectedRows": int(item["rows"]),
                "actualRows": metadata.num_rows if metadata is not None else None,
                "passed": bool(
                    path.is_file()
                    and actual == item["sha256"]
                    and manifest_hashes.get(relative) == item["sha256"]
                    and metadata is not None
                    and metadata.num_rows == int(item["rows"])
                ),
            }
        )

    evidence_rows: list[dict[str, Any]] = []
    for evidence_id, item in config["inputs"]["validationEvidence"].items():
        path = Path(item["path"])
        actual = sha256_file(path) if path.is_file() else None
        expected = item.get("sha256")
        passed = bool(path.is_file() and (expected is None or actual == expected))
        evidence_rows.append(
            {
                "evidenceId": evidence_id,
                "path": str(path),
                "expectedSha256": expected,
                "actualSha256": actual,
                "passed": passed,
            }
        )

    prefix_path = Path(config["inputs"]["scientificTables"]["prefixValues"]["path"])
    prefix_schema = pq.read_schema(prefix_path)
    schema_gate = bool(
        SOURCE_FIELD in prefix_schema.names and DERIVED_FIELD not in prefix_schema.names
    )
    classification = json.loads(
        (S12I_ROOT / "classification.json").read_text(encoding="utf-8")
    )
    execution = json.loads(
        (S12I_ROOT / "execution_validation.json").read_text(encoding="utf-8")
    )
    substantive_gate = bool(
        classification.get("classification") == "S12I_VALIDATION_FAILED_CLOSED"
        and classification.get("scientificAdjudicationComputed") is False
        and execution.get("freshComputation", {}).get("completedSourceTasks") == 96
        and execution.get("freshComputation", {}).get("sourceTaskFailureRows") == 0
        and execution.get("operationalSourceValidation", {}).get(
            "fullReplayAllPassed"
        )
        is True
        and execution.get("operationalSourceValidation", {}).get(
            "eligiblePrefixReplayAllPassed"
        )
        is True
        and execution.get("operationalSourceValidation", {}).get(
            "structuralSuffixChecksAllPassed"
        )
        is True
        and execution.get("operationalSourceValidation", {}).get(
            "executedSuffixSentinelsAllPassed"
        )
        is True
    )
    passed = bool(
        manifest.get("passed")
        and all(item["passed"] for item in scientific_rows)
        and all(item["passed"] for item in evidence_rows)
        and schema_gate
        and substantive_gate
    )
    return {
        "schema": "eidosoma.e01.s12j_input_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "s12iArtifactManifestPath": str(manifest_path),
        "s12iArtifactManifestPassed": manifest.get("passed"),
        "scientificInputs": scientific_rows,
        "validationEvidence": evidence_rows,
        "prefixSchemaColumns": prefix_schema.names,
        "sourceFieldPresent": SOURCE_FIELD in prefix_schema.names,
        "derivedFieldAbsent": DERIVED_FIELD not in prefix_schema.names,
        "s12iClassificationRetained": classification.get("classification"),
        "s12iScientificAdjudicationComputed": classification.get(
            "scientificAdjudicationComputed"
        ),
        "s12iOperationalEvidencePassed": substantive_gate,
        "candidateStatisticsInspected": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-commit", required=True)
    args = parser.parse_args()
    if STEP_ROOT.exists() and any(STEP_ROOT.iterdir()):
        raise RuntimeError("S12J artifact directory must be empty before freeze")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if (
        head != args.design_commit
        or remote != head
        or git("branch", "--show-current") != "eidosoma/groups/42"
        or git("status", "--short")
    ):
        raise RuntimeError("S12J design must be committed, pushed, and clean")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["versionedStepId"] != VERSION:
        raise RuntimeError("S12J version mismatch")
    STEP_ROOT.mkdir(parents=True)
    shutil.copyfile(CONFIG, STEP_ROOT / "preregistration.yaml")

    baseline = immutable_prior_baseline()
    if not baseline["passed"]:
        raise RuntimeError("S12J immutable-prior baseline is empty")
    write_json(STEP_ROOT / "immutable_prior_baseline.json", baseline)

    inputs = validate_s12i_inputs(config)
    if not inputs["passed"]:
        raise RuntimeError("S12J immutable S12I input gate failed")
    write_json(STEP_ROOT / "input_manifest.json", inputs)

    adapter_contract = {
        "schema": "eidosoma.e01.s12j_adapter_contract.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "adapterId": ADAPTER_ID,
        "humanOverride": "S12I_no_repair_rule_for_aggregation_interface_only",
        "oneRepairOnly": True,
        "sourceTable": str(
            config["inputs"]["scientificTables"]["prefixValues"]["path"]
        ),
        "sourceTableSha256": config["inputs"]["scientificTables"][
            "prefixValues"
        ]["sha256"],
        "sourceField": SOURCE_FIELD,
        "derivedField": DERIVED_FIELD,
        "formula": f"{DERIVED_FIELD} := {SOURCE_FIELD}",
        "derivedStatisticalViewOnly": True,
        "sourceMutationPermitted": False,
        "otherColumnOrRowChangePermitted": False,
        "uniqueKeys": list(UNIQUE_KEYS),
        "monotonicGroupKeys": list(MONOTONIC_GROUP_KEYS),
        "endpointJoinKeys": list(ENDPOINT_JOIN_KEYS),
        "allPreregisteredAdapterGatesRequired": True,
        "furtherRepairPermitted": False,
        "passed": True,
    }
    write_json(STEP_ROOT / "adapter_contract.json", adapter_contract)
    write_json(
        STEP_ROOT / "analysis_lock.json",
        {
            "schema": "eidosoma.e01.s12j_analysis_lock.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "inheritedContract": config["frozenScientificContract"],
            "executionFirewall": config["executionFirewall"],
            "forbiddenWork": config["forbiddenWork"],
            "candidateStatisticsInspected": False,
            "sourceFitRerunPermitted": False,
            "newGardTrajectoryCount": 0,
            "s12gScientificCacheReadsPermitted": 0,
            "passed": True,
        },
    )

    code_files = [
        CONFIG,
        REPO / "src/e01_aggregation_interface_repair/__init__.py",
        REPO / "src/e01_aggregation_interface_repair/core.py",
        REPO / "scripts/e01/freeze_s12j_preregistration.py",
        REPO / "scripts/e01/run_s12j_aggregation_interface_repair_confirmation.py",
        REPO / "tests/e01/test_s12j_aggregation_interface_repair_confirmation.py",
        REPO / "configs/e01/s12i_aggregate_support_waiver_sensitivity_preregistration.yaml",
        REPO / "configs/e01/s12g_output_schemas.json",
        REPO / "src/e01_aggregate_support_waiver_sensitivity/core.py",
        REPO / "src/e01_frozen_timebase_ensemble/core.py",
        REPO / "src/e01_source_emergence_metric_identity/analysis.py",
        REPO / "scripts/e01/run_s12g_frozen_timebase_ensemble.py",
    ]
    missing_code = [str(path) for path in code_files if not path.is_file()]
    if missing_code:
        raise RuntimeError(f"S12J method-lock files missing: {missing_code}")
    lock = {
        "schema": "eidosoma.e01.s12j_method_lock.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "designCommit": head,
        "remoteCommit": remote,
        "branch": git("branch", "--show-current"),
        "files": [
            {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
            for path in code_files
        ],
        "immutablePriorFileCount": baseline["fileCount"],
        "immutableInputGatePassed": inputs["passed"],
        "adapterUnitTestsRequiredBeforeCommit": True,
        "candidateStatisticsInspected": False,
        "sourceFitRerunPermitted": False,
        "furtherRepairPermitted": False,
        "passed": True,
    }
    write_json(STEP_ROOT / "method_lock.json", lock)
    write_json(
        STEP_ROOT / "implementation_lock.json",
        {**lock, "schema": "eidosoma.e01.s12j_implementation_lock.v1"},
    )
    write_json(
        STEP_ROOT / "preregistration_record.json",
        {
            "schema": "eidosoma.e01.s12j_preregistration_record.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "frozenAtUtc": datetime.now(UTC).isoformat(),
            "designCommit": head,
            "remoteCommit": remote,
            "preregistrationSha256": sha256_file(STEP_ROOT / "preregistration.yaml"),
            "methodLockSha256": sha256_file(STEP_ROOT / "method_lock.json"),
            "adapterContractSha256": sha256_file(STEP_ROOT / "adapter_contract.json"),
            "validatedCommittedAndPushedBeforeCandidateStatistics": True,
            "candidateStatisticsInspected": False,
            "passed": True,
        },
    )
    write_json(
        STEP_ROOT / "scope_access_ledger.json",
        {
            "schema": "eidosoma.e01.s12j_scope_access_ledger.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "events": [
                {
                    "stage": "PRE_STATISTICS_METHOD_AND_INPUT_LOCK",
                    "candidateStatisticComputedOrInspected": False,
                    "scientificSourceTableRowsOpened": False,
                    "sourceFitExecuted": False,
                    "newGardTrajectoryGenerated": False,
                    "s12gScientificCachePayloadOpened": False,
                    "status": "PASS",
                }
            ],
            "success": None,
        },
    )
    print(
        json.dumps(
            {
                "stage": "S12J_preregistration_frozen",
                "designCommit": head,
                "priorArtifactFiles": baseline["fileCount"],
                "adapterId": ADAPTER_ID,
                "candidateStatisticsInspected": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
