#!/usr/bin/env python3
"""Validate and manifest the completed E01 S03 artifact bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_ARTIFACTS = Path("/artifacts")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected object in {path}")
    return data


def run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returnCode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "valid": result.returncode == 0,
    }


def verify_prior_output_manifests(artifacts_root: Path) -> dict[str, Any]:
    checks = []
    for step in ("S01", "S02"):
        manifest_path = artifacts_root / f"research_steps/{step}/artifact_manifest.json"
        manifest = load_json(manifest_path)
        for entry in manifest["entries"]:
            if entry["role"] != f"{step}_output":
                continue
            path = Path(entry["path"])
            actual = sha256_file(path) if path.is_file() else None
            checks.append(
                {
                    "step": step,
                    "path": str(path),
                    "expectedSha256": entry["sha256"],
                    "actualSha256": actual,
                    "valid": actual == entry["sha256"],
                }
            )
    return {"valid": all(item["valid"] for item in checks), "checks": checks}


def expected_output_paths(artifacts_root: Path) -> list[Path]:
    provenance = artifacts_root / "E01_forensic_replication_bundle/provenance"
    specifications = artifacts_root / "E01_forensic_replication_bundle/specifications"
    step = artifacts_root / "research_steps/S03"
    return [
        provenance / "source_manifest.yaml",
        provenance / "source_file_hashes.csv",
        provenance / "environment_report.json",
        provenance / "environment_report.md",
        provenance / "requirements-s03-py313-cu128.lock",
        provenance / "clean_environment_python_freeze.txt",
        provenance / "base_environment_python_freeze.txt",
        provenance / "system_packages.lock",
        provenance / "dependency_artifacts.csv",
        provenance / "dependency_licenses.csv",
        provenance / "precision_policy.yaml",
        provenance / "license_notes.md",
        specifications / "specification_registry_v0.3.0.yaml",
        step / "commit_verification.json",
        step / "author_code_search.json",
        step / "registry_update_audit.csv",
        step / "source_registry_validation.json",
        step / "clean_environment_smoke.json",
        step / "research_step_full_results.md",
    ]


def validate(artifacts_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    expected = expected_output_paths(artifacts_root)
    presence = [
        {
            "path": str(path),
            "exists": path.is_file(),
            "sizeBytes": path.stat().st_size if path.is_file() else None,
            "valid": path.is_file() and path.stat().st_size > 0,
        }
        for path in expected
    ]
    missing = [item["path"] for item in presence if not item["valid"]]
    if missing:
        errors.append(f"Missing or empty expected outputs: {missing}")

    source_validation = load_json(
        artifacts_root / "research_steps/S03/source_registry_validation.json"
    )
    commit_validation = load_json(
        artifacts_root / "research_steps/S03/commit_verification.json"
    )
    environment = load_json(
        artifacts_root
        / "E01_forensic_replication_bundle/provenance/environment_report.json"
    )
    smoke = load_json(
        artifacts_root / "research_steps/S03/clean_environment_smoke.json"
    )
    for label, valid in {
        "source and registry validation": source_validation.get("valid"),
        "immutable commit validation": commit_validation.get("valid"),
        "environment validation": environment.get("validation", {}).get("valid"),
        "clean environment smoke": smoke.get("success"),
    }.items():
        if not valid:
            errors.append(f"{label} failed")

    registry_path = (
        artifacts_root
        / "E01_forensic_replication_bundle/specifications/specification_registry_v0.3.0.yaml"
    )
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    gate = registry["executionGate"]
    registry_checks = {
        "version": registry["registryVersion"],
        "parameterCount": len(registry["parameters"]),
        "unresolvedParameterCount": gate["unresolvedParameterCount"],
        "branchSetCount": gate["unexpandedBranchSetCount"],
        "executionBlockingParameterCount": gate["executionBlockingParameterCount"],
        "executionGateOpen": gate["executable"],
        "noSilentDefaults": gate["noSilentDefaults"],
        "changedOutsideScope": source_validation["registryValidation"][
            "changedOutsideScope"
        ],
        "preservedNonSourceParameterCount": source_validation["registryValidation"][
            "preservedNonSourceParameterCount"
        ],
    }
    expected_registry = {
        "version": "E01-specification-registry-v0.3.0",
        "parameterCount": 120,
        "unresolvedParameterCount": 64,
        "branchSetCount": 21,
        "executionBlockingParameterCount": 85,
        "executionGateOpen": False,
        "noSilentDefaults": True,
        "changedOutsideScope": [],
        "preservedNonSourceParameterCount": 114,
    }
    if registry_checks != expected_registry:
        errors.append(f"Registry checks differ: {registry_checks}")

    dependency_path = (
        artifacts_root
        / "E01_forensic_replication_bundle/provenance/dependency_artifacts.csv"
    )
    with dependency_path.open("r", encoding="utf-8", newline="") as handle:
        dependencies = list(csv.DictReader(handle))
    dependency_checks = {
        "artifactCount": len(dependencies),
        "uniquePackageCount": len({item["package"] for item in dependencies}),
        "validSha256Count": sum(
            len(item["sha256"]) == 64
            and all(character in "0123456789abcdef" for character in item["sha256"])
            for item in dependencies
        ),
        "lockedArtifactCount": sum(
            item["hash_in_compiled_lock"] == "True" for item in dependencies
        ),
        "commitBuiltArtifactCount": sum(
            item["source"] == "pinned git commit build" for item in dependencies
        ),
    }
    if dependency_checks != {
        "artifactCount": 39,
        "uniquePackageCount": 39,
        "validSha256Count": 39,
        "lockedArtifactCount": 38,
        "commitBuiltArtifactCount": 1,
    }:
        errors.append(f"Dependency checks differ: {dependency_checks}")

    prior_outputs = verify_prior_output_manifests(artifacts_root)
    if not prior_outputs["valid"]:
        errors.append("One or more S01/S02 output artifacts changed")

    report_path = artifacts_root / "research_steps/S03/research_step_full_results.md"
    report_text = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    required_report_markers = [
        "## Top summary",
        "S03",
        "Completion status",
        "Artifacts written",
        "Validation result",
        "Outcome classification",
        "Caveats or blockers",
        "Lay summary",
        "Recommended next action",
        "## Frozen question",
        "## Inputs",
        "## Detailed methods",
        "## Commands",
        "## Results",
        "## Validation",
        "## Provenance",
    ]
    missing_markers = [marker for marker in required_report_markers if marker not in report_text]
    if missing_markers:
        errors.append(f"Full-results report markers missing: {missing_markers}")

    s04_path = artifacts_root / "research_steps/S04"
    if s04_path.exists():
        errors.append("S04 artifact directory exists; S03 scope boundary violated")

    commands = {
        "ruff": run(["ruff", "check", "scripts/e01", "tests/e01", "configs/e01"]),
        "pytest": run(["pytest", "-q"]),
        "gitDiffCheck": run(["git", "diff", "--check"]),
    }
    for label, result in commands.items():
        if not result["valid"]:
            errors.append(f"{label} failed: {result['stderr'] or result['stdout']}")

    image_digest = environment["runtimeImage"]["ociImageDigest"]
    if not image_digest.startswith("UNAVAILABLE::"):
        warnings.append("OCI digest unexpectedly became available; report should be reviewed")
    warnings.extend(
        [
            "Parent OCI digest is not exposed; a composite runtime fingerprint is recorded instead.",
            "Author code release, exact Phi^r atom mapping, and redundancy choice remain unresolved sentinels.",
            "Historical GARD has no detected license; modern GARD has no root license.",
            "Two L4 GPUs were visible although the generic plan described one fast GPU; later runs must select by UUID.",
        ]
    )

    return {
        "schema": "eidosoma.e01.s03.validation_summary.v1",
        "researchStepId": "S03",
        "stepNumber": 3,
        "success": not errors,
        "status": "Complete" if not errors else "Failed validation",
        "artifactsWritten": [str(path) for path in expected],
        "validationResult": "PASS" if not errors else "FAIL",
        "caveatsOrBlockers": warnings,
        "recommendedNextAction": (
            "Hand control back; if separately authorized, resolve remaining execution-blocking specifications before or within S04 without starting S04 from this step."
        ),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "presence": presence,
        "sourceValidationValid": source_validation["valid"],
        "commitVerificationValid": commit_validation["valid"],
        "environmentValidationValid": environment["validation"]["valid"],
        "cleanEnvironmentSmokeValid": smoke["success"],
        "registryChecks": registry_checks,
        "dependencyChecks": dependency_checks,
        "priorOutputIntegrity": prior_outputs,
        "reportMarkersMissing": missing_markers,
        "s04NotStarted": not s04_path.exists(),
        "commands": commands,
    }


def manifest_paths(artifacts_root: Path) -> list[tuple[Path, str]]:
    output_paths = expected_output_paths(artifacts_root) + [
        artifacts_root / "research_steps/S03/validation_summary.json"
    ]
    input_paths = [
        WORKSPACE_ROOT / "AGENTS.md",
        WORKSPACE_ROOT / "FULL_PLAN.md",
        WORKSPACE_ROOT / "RESEARCH_PLAN.md",
        WORKSPACE_ROOT / "input-attachments/MANIFEST.json",
        WORKSPACE_ROOT
        / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/_metadata/ATTACHMENT.md",
        WORKSPACE_ROOT
        / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md",
        artifacts_root / "research_steps/S01/research_step_full_results.md",
        artifacts_root / "research_steps/S01/source_reconciliation.csv",
        artifacts_root / "research_steps/S02/research_step_full_results.md",
        artifacts_root / "E01_forensic_replication_bundle/ledgers/claim_ledger.csv",
        artifacts_root / "E01_forensic_replication_bundle/ledgers/ambiguity_ledger.csv",
        artifacts_root / "E01_forensic_replication_bundle/ledgers/discrepancy_taxonomy.csv",
        artifacts_root
        / "E01_forensic_replication_bundle/specifications/specification_registry.yaml",
        *sorted((REPO_ROOT / "configs/e01").glob("s03_*")),
        REPO_ROOT / "scripts/e01/build_source_snapshot.py",
        REPO_ROOT / "scripts/e01/capture_environment_snapshot.py",
        REPO_ROOT / "scripts/e01/s03_clean_smoke.py",
        REPO_ROOT / "scripts/e01/validate_s03_artifacts.py",
        REPO_ROOT / "tests/e01/test_source_snapshot.py",
        REPO_ROOT / "tests/e01/test_environment_snapshot.py",
    ]
    return [(path, "S03_output") for path in output_paths] + [
        (path, "input_or_code") for path in input_paths
    ]


def write_manifest(artifacts_root: Path) -> dict[str, Any]:
    entries = []
    for path, role in sorted(manifest_paths(artifacts_root), key=lambda item: str(item[0])):
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            {
                "path": str(path),
                "role": role,
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    git_commit = run(["git", "rev-parse", "HEAD"])["stdout"]
    manifest = {
        "schema": "eidosoma.e01.s03.artifact_manifest.v1",
        "researchStepId": "S03",
        "experimentId": "E01",
        "generatedOn": "2026-08-01",
        "repository": str(REPO_ROOT),
        "gitCommit": git_commit,
        "entries": entries,
    }
    output = artifacts_root / "research_steps/S03/artifact_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()
    validation = validate(args.artifacts_root)
    output = args.artifacts_root / "research_steps/S03/validation_summary.json"
    output.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not validation["valid"]:
        raise RuntimeError(json.dumps(validation["errors"], indent=2))
    result: dict[str, Any] = {
        "valid": True,
        "validationSummary": str(output),
    }
    if args.manifest:
        manifest = write_manifest(args.artifacts_root)
        result["artifactManifest"] = str(
            args.artifacts_root / "research_steps/S03/artifact_manifest.json"
        )
        result["manifestEntryCount"] = len(manifest["entries"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
