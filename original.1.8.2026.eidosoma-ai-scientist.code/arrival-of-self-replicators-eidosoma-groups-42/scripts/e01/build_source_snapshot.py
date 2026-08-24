#!/usr/bin/env python3
"""Verify E01 S03 source pins and apply source-only registry updates."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PINS = REPO_ROOT / "configs/e01/s03_source_pins.yaml"
DEFAULT_UPDATES = REPO_ROOT / "configs/e01/s03_registry_updates.yaml"
DEFAULT_ARTIFACTS = Path("/artifacts")


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _verify_file(path: Path, expected_sha256: str, expected_size: int | None = None) -> dict[str, Any]:
    exists = path.is_file()
    actual_sha256 = sha256_file(path) if exists else None
    actual_size = path.stat().st_size if exists else None
    valid = exists and actual_sha256 == expected_sha256
    if expected_size is not None:
        valid = valid and actual_size == expected_size
    return {
        "path": str(path),
        "exists": exists,
        "expectedSha256": expected_sha256,
        "actualSha256": actual_sha256,
        "expectedSizeBytes": expected_size,
        "actualSizeBytes": actual_size,
        "valid": valid,
    }


def _remote_ref_commit(remote_url: str, selected_ref: str) -> tuple[str | None, str]:
    result = run(
        [
            "git",
            "ls-remote",
            remote_url,
            selected_ref,
            f"{selected_ref}^{{}}",
        ]
    )
    if result.returncode != 0:
        return None, result.stderr.strip()
    rows = [line.split("\t", maxsplit=1) for line in result.stdout.splitlines()]
    by_ref = {ref: commit for commit, ref in rows if len(commit) == 40}
    return by_ref.get(f"{selected_ref}^{{}}", by_ref.get(selected_ref)), ""


def verify_repository(pin: dict[str, Any], *, verify_remote: bool) -> dict[str, Any]:
    local = Path(pin["localPath"])
    checks: dict[str, Any] = {}
    for name, args in {
        "head": ["git", "rev-parse", "HEAD"],
        "tree": ["git", "rev-parse", "HEAD^{tree}"],
        "objectType": ["git", "cat-file", "-t", pin["commit"]],
        "fsck": ["git", "fsck", "--full", "--strict"],
    }.items():
        result = run(args, cwd=local)
        checks[name] = {
            "returnCode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }

    remote_commit = None
    remote_error = "not requested"
    if verify_remote:
        remote_commit, remote_error = _remote_ref_commit(
            pin["remoteUrl"], pin["selectedRef"]
        )

    archive = _verify_file(
        Path(pin["archivePath"]),
        pin["archiveSha256"],
        int(pin["archiveSizeBytes"]),
    )
    files = [
        _verify_file(local / item["path"], item["sha256"])
        | {"relativePath": item["path"]}
        for item in pin["files"]
    ]
    local_valid = (
        checks["head"]["returnCode"] == 0
        and checks["head"]["stdout"] == pin["commit"]
        and checks["tree"]["returnCode"] == 0
        and checks["tree"]["stdout"] == pin["tree"]
        and checks["objectType"]["returnCode"] == 0
        and checks["objectType"]["stdout"] == "commit"
        and checks["fsck"]["returnCode"] == 0
        and archive["valid"]
        and all(item["valid"] for item in files)
    )
    remote_valid = remote_commit == pin["commit"] if verify_remote else None
    return {
        "sourceId": pin["sourceId"],
        "remoteUrl": pin["remoteUrl"],
        "selectedRef": pin["selectedRef"],
        "expectedCommit": pin["commit"],
        "remoteRefCommit": remote_commit,
        "remoteError": remote_error,
        "expectedTree": pin["tree"],
        "checks": checks,
        "archive": archive,
        "files": files,
        "localValid": local_valid,
        "remoteValid": remote_valid,
        "valid": local_valid and (remote_valid is not False),
    }


def verify_sources(pins: dict[str, Any], *, verify_remote: bool = True) -> dict[str, Any]:
    paper = pins["paper"]
    paper_check = _verify_file(
        Path(paper["localPath"]), paper["sha256"], int(paper["sizeBytes"])
    )
    paper_check["matchesAttachmentOriginalSize"] = (
        paper_check["actualSizeBytes"] == paper["attachmentOriginalSizeBytes"]
    )
    repositories = [
        verify_repository(pin, verify_remote=verify_remote)
        for pin in pins["repositories"]
    ]
    valid = (
        paper_check["valid"]
        and paper_check["matchesAttachmentOriginalSize"]
        and all(item["valid"] for item in repositories)
    )
    return {
        "schema": "eidosoma.e01.s03_commit_verification.v1",
        "researchStepId": "S03",
        "valid": valid,
        "remoteVerificationRequested": verify_remote,
        "paper": paper_check,
        "repositories": repositories,
    }


def apply_registry_updates(
    prior: dict[str, Any], updates: dict[str, Any], prior_sha256: str
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    if prior["registryVersion"] != updates["inputRegistryVersion"]:
        raise ValueError("Input registry version does not match update contract")
    allowed = set(updates["allowedAmbiguityIds"])
    declared = {item["ambiguityId"] for item in updates["updates"]}
    if allowed != declared:
        raise ValueError("Allowed and declared registry update IDs differ")

    registry = copy.deepcopy(prior)
    after_by_id = {
        item["ambiguityId"]: item
        for item in registry["parameters"]
        if item.get("ambiguityId")
    }
    before_by_parameter = {item["parameter"]: item for item in prior["parameters"]}
    after_by_parameter = {item["parameter"]: item for item in registry["parameters"]}
    audit: list[dict[str, str]] = []
    update_fields = [
        "value",
        "resolutionStatus",
        "unresolved",
        "sourceEvidence",
        "resolutionBasis",
        "validationRule",
    ]
    for change in updates["updates"]:
        ambiguity_id = change["ambiguityId"]
        target = after_by_id.get(ambiguity_id)
        if target is None or target["parameter"] != change["parameter"]:
            raise ValueError(f"Registry target mismatch for {ambiguity_id}")
        before = copy.deepcopy(target)
        for field in update_fields:
            target[field] = change[field]
        audit.append(
            {
                "ambiguity_id": ambiguity_id,
                "parameter": target["parameter"],
                "before_status": str(before["resolutionStatus"]),
                "after_status": str(target["resolutionStatus"]),
                "before_unresolved": str(before["unresolved"]).lower(),
                "after_unresolved": str(target["unresolved"]).lower(),
                "before_value": str(before["value"]),
                "after_value": str(target["value"]),
                "source_only_change": "true",
            }
        )

    registry["researchStepId"] = "S03"
    registry["registryVersion"] = updates["outputRegistryVersion"]
    registry["generatedOn"] = "2026-08-01"
    registry["statusDefinitions"]["SOURCE_FIXED"] = (
        "Immutable identity or value recovered from authoritative source evidence."
    )
    unresolved_count = sum(bool(item["unresolved"]) for item in registry["parameters"])
    branch_count = sum(
        item["resolutionStatus"] == "FROZEN_BRANCH_SET"
        for item in registry["parameters"]
    )
    registry["executionGate"].update(
        {
            "executable": unresolved_count == 0 and branch_count == 0,
            "unresolvedParameterCount": unresolved_count,
            "unexpandedBranchSetCount": branch_count,
            "executionBlockingParameterCount": unresolved_count + branch_count,
            "noSilentDefaults": True,
        }
    )
    registry["lineage"] = {
        "priorRegistryVersion": prior["registryVersion"],
        "priorRegistrySha256": prior_sha256,
        "updateContract": "configs/e01/s03_registry_updates.yaml",
        "changeScope": sorted(allowed),
    }

    errors: list[str] = []
    changed_parameters = {change["parameter"] for change in updates["updates"]}
    unchanged_parameters = set(before_by_parameter) - changed_parameters
    changed_outside_scope = sorted(
        parameter
        for parameter in unchanged_parameters
        if before_by_parameter[parameter] != after_by_parameter[parameter]
    )
    if changed_outside_scope:
        errors.append(f"Out-of-scope parameters changed: {changed_outside_scope}")
    if len(registry["parameters"]) != len(prior["parameters"]):
        errors.append("Parameter count changed")
    if len({item["parameter"] for item in registry["parameters"]}) != len(
        registry["parameters"]
    ):
        errors.append("Parameter names are not unique")
    for item in registry["parameters"]:
        value = str(item["value"])
        status = item["resolutionStatus"]
        if item["unresolved"] and status in {
            "UNRESOLVED_REQUIRED",
            "DEFERRED_EVIDENCE",
        }:
            expected = f"UNRESOLVED::{item['ambiguityId']}"
            if value != expected:
                errors.append(f"Invalid unresolved sentinel for {item['ambiguityId']}")
        if status == "CONFLICT_PRESERVED" and not value.startswith("CONFLICT::"):
            errors.append(f"Invalid conflict sentinel for {item['ambiguityId']}")
        if status == "FROZEN_BRANCH_SET" and not value.startswith("BRANCH_SET::"):
            errors.append(f"Invalid branch-set sentinel for {item['ambiguityId']}")
        if status == "SOURCE_FIXED" and (
            item["unresolved"]
            or value.startswith(("UNRESOLVED::", "CONFLICT::", "BRANCH_SET::"))
        ):
            errors.append(f"Invalid source-fixed value for {item['ambiguityId']}")

    expected_resolved = {"E01-A001", "E01-A003", "E01-A004"}
    actual_resolved = {
        item["ambiguityId"]
        for item in registry["parameters"]
        if item["ambiguityId"] in allowed and not item["unresolved"]
    }
    if actual_resolved != expected_resolved:
        errors.append(
            f"Unexpected S03 resolutions: expected {expected_resolved}, got {actual_resolved}"
        )
    validation = {
        "valid": not errors,
        "errors": errors,
        "changedAmbiguityIds": sorted(allowed),
        "changedOutsideScope": changed_outside_scope,
        "parameterCountBefore": len(prior["parameters"]),
        "parameterCountAfter": len(registry["parameters"]),
        "unresolvedCountBefore": prior["executionGate"]["unresolvedParameterCount"],
        "unresolvedCountAfter": unresolved_count,
        "branchSetCountBefore": prior["executionGate"]["unexpandedBranchSetCount"],
        "branchSetCountAfter": branch_count,
        "executionBlockingCountAfter": unresolved_count + branch_count,
        "executionGateOpen": registry["executionGate"]["executable"],
        "noSilentDefaults": registry["executionGate"]["noSilentDefaults"],
        "preservedConflictCount": sum(
            item["resolutionStatus"] == "CONFLICT_PRESERVED"
            for item in registry["parameters"]
        ),
        "preservedNonSourceParameterCount": len(unchanged_parameters),
    }
    return registry, audit, validation


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(
    artifacts_root: Path,
    *,
    pins_path: Path = DEFAULT_PINS,
    updates_path: Path = DEFAULT_UPDATES,
    verify_remote: bool = True,
) -> dict[str, Any]:
    pins = load_yaml(pins_path)
    updates = load_yaml(updates_path)
    source_verification = verify_sources(pins, verify_remote=verify_remote)

    prior_path = Path(pins["priorRegistry"])
    prior_sha256 = sha256_file(prior_path)
    prior = load_yaml(prior_path)
    registry, audit, registry_validation = apply_registry_updates(
        prior, updates, prior_sha256
    )

    provenance_dir = artifacts_root / "E01_forensic_replication_bundle/provenance"
    specification_dir = artifacts_root / "E01_forensic_replication_bundle/specifications"
    step_dir = artifacts_root / "research_steps/S03"
    for directory in (provenance_dir, specification_dir, step_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = copy.deepcopy(pins)
    manifest.pop("cacheRoot", None)
    manifest["schema"] = "eidosoma.e01.s03_source_manifest.v1"
    manifest["manifestVersion"] = "E01-source-manifest-v0.1.0"
    manifest["verification"] = {
        "valid": source_verification["valid"],
        "commitVerificationArtifact": "$ARTIFACTS_DIR/research_steps/S03/commit_verification.json",
        "sourceFileHashArtifact": "$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/source_file_hashes.csv",
    }
    environment_path = provenance_dir / "environment_report.json"
    dependency_path = provenance_dir / "dependency_artifacts.csv"
    if environment_path.is_file() and dependency_path.is_file():
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        with dependency_path.open("r", encoding="utf-8", newline="") as handle:
            dependencies = list(csv.DictReader(handle))
        manifest["environmentSnapshot"] = {
            "environmentReport": "$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/environment_report.json",
            "environmentReportSha256": sha256_file(environment_path),
            "runtimeImage": environment["runtimeImage"],
            "compiledLock": environment["dependencySnapshot"]["compiledLockPath"],
            "compiledLockSha256": environment["dependencySnapshot"][
                "compiledLockSha256"
            ],
            "lockedPackageCount": environment["dependencySnapshot"][
                "lockedPackageCount"
            ],
            "dependencyArtifactCount": len(dependencies),
            "dependencies": [
                {
                    "package": item["package"],
                    "version": item["version"],
                    "filename": item["filename"],
                    "sha256": item["sha256"],
                    "source": item["source"],
                }
                for item in dependencies
            ],
            "cleanEnvironmentSmoke": "$ARTIFACTS_DIR/research_steps/S03/clean_environment_smoke.json",
            "cleanEnvironmentSmokeSha256": sha256_file(
                artifacts_root / "research_steps/S03/clean_environment_smoke.json"
            ),
            "precisionPolicy": "$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/precision_policy.yaml",
        }
    (provenance_dir / "source_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (step_dir / "commit_verification.json").write_text(
        json.dumps(source_verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (step_dir / "author_code_search.json").write_text(
        json.dumps(pins["authorCodeSearch"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (specification_dir / "specification_registry_v0.3.0.yaml").write_text(
        yaml.safe_dump(registry, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _write_csv(
        step_dir / "registry_update_audit.csv",
        audit,
        [
            "ambiguity_id",
            "parameter",
            "before_status",
            "after_status",
            "before_unresolved",
            "after_unresolved",
            "before_value",
            "after_value",
            "source_only_change",
        ],
    )

    file_rows: list[dict[str, Any]] = [
        {
            "source_id": pins["paper"]["sourceId"],
            "kind": "paper_pdf",
            "relative_path": "paper-2607.28250v1.pdf",
            "size_bytes": pins["paper"]["sizeBytes"],
            "sha256": pins["paper"]["sha256"],
            "license": pins["paper"]["license"],
        }
    ]
    for pin in pins["repositories"]:
        file_rows.append(
            {
                "source_id": pin["sourceId"],
                "kind": "git_archive",
                "relative_path": Path(pin["archivePath"]).name,
                "size_bytes": pin["archiveSizeBytes"],
                "sha256": pin["archiveSha256"],
                "license": pin["license"],
            }
        )
        local = Path(pin["localPath"])
        for item in pin["files"]:
            file_rows.append(
                {
                    "source_id": pin["sourceId"],
                    "kind": "source_file",
                    "relative_path": item["path"],
                    "size_bytes": (local / item["path"]).stat().st_size,
                    "sha256": item["sha256"],
                    "license": pin["license"],
                }
            )
    omega = next(
        item for item in pins["repositories"] if item["sourceId"] == "omegaid_optional"
    )
    file_rows.append(
        {
            "source_id": "omegaid_optional",
            "kind": "pypi_sdist",
            "relative_path": "omegaid-0.2.5.tar.gz",
            "size_bytes": omega["pypiSdistSizeBytes"],
            "sha256": omega["pypiSdistSha256"],
            "license": omega["license"],
        }
    )
    _write_csv(
        provenance_dir / "source_file_hashes.csv",
        file_rows,
        ["source_id", "kind", "relative_path", "size_bytes", "sha256", "license"],
    )

    result = {
        "researchStepId": "S03",
        "valid": source_verification["valid"] and registry_validation["valid"],
        "sourceVerification": source_verification,
        "registryValidation": registry_validation,
        "artifacts": [
            str(provenance_dir / "source_manifest.yaml"),
            str(provenance_dir / "source_file_hashes.csv"),
            str(specification_dir / "specification_registry_v0.3.0.yaml"),
            str(step_dir / "commit_verification.json"),
            str(step_dir / "author_code_search.json"),
            str(step_dir / "registry_update_audit.csv"),
        ],
    }
    (step_dir / "source_registry_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not result["valid"]:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    parser.add_argument("--updates", type=Path, default=DEFAULT_UPDATES)
    parser.add_argument("--no-remote", action="store_true")
    args = parser.parse_args()
    result = build(
        args.artifacts_root,
        pins_path=args.pins,
        updates_path=args.updates,
        verify_remote=not args.no_remote,
    )
    print(json.dumps({"valid": result["valid"], "artifacts": result["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
