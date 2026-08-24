#!/usr/bin/env python3
"""Validate and freeze the E01 S12D preregistration and implementation lock."""

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

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
CONFIG = REPO / "configs/e01/s12d_source_emergence_metric_identity_preregistration.yaml"
STEP_ROOT = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "research_steps/S12D"
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
PRIOR_STEPS = tuple(
    [f"S{index:02d}" for index in range(1, 13)] + ["S11R", "S12B", "S12C"]
)
VERSION = "E01-S12D-SOURCE-EMERGENCE-METRIC-IDENTITY-CONFIRMATION-v1.0.0"
ALLOWED_DESIGN_PATHS = {
    "configs/e01/s12d_source_emergence_metric_identity_preregistration.yaml",
    "scripts/e01/freeze_s12d_preregistration.py",
    "scripts/e01/run_s12d_source_emergence_metric_identity.py",
    "scripts/e01/s12d_original_source_metric_adapter.py",
    "src/e01_source_emergence_metric_identity/__init__.py",
    "src/e01_source_emergence_metric_identity/core.py",
    "src/e01_source_emergence_metric_identity/analysis.py",
    "tests/e01/test_s12d_source_emergence_metric_identity.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def git_output(checkout: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=checkout, text=True, capture_output=True, check=False
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def directory_identity(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    records = [
        {
            "path": item.relative_to(path).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in files
    ]
    material = "".join(
        f"{record['sha256']}  ./{record['path']}\n" for record in records
    ).encode("utf-8")
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "fileCount": len(records),
        "totalBytes": sum(record["bytes"] for record in records),
        "aggregateSha256": hashlib.sha256(material).hexdigest(),
        "files": records,
    }


def immutable_snapshot() -> dict[str, Any]:
    roots: dict[str, Any] = {}
    for step in PRIOR_STEPS:
        path = Path("/artifacts/research_steps") / step
        if not path.is_dir():
            raise RuntimeError(f"missing immutable prior artifact directory: {path}")
        roots[step] = directory_identity(path)
    keys = {
        "registry": Path(
            "/artifacts/E01_forensic_replication_bundle/specifications/specification_registry_v0.3.0.yaml"
        ),
        "paperExtraction": Path(
            "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md"
        ),
        "paperPdf": Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf"),
        "inputManifest": Path("/workspace/input-attachments/MANIFEST.json"),
        "attachmentSidecar": Path(
            "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/_metadata/ATTACHMENT.md"
        ),
    }
    key_files = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in keys.items()
    }
    return {
        "schema": "eidosoma.e01.s12d.immutable_prior_audit.v1",
        "researchStepId": "S12D",
        "algorithm": "sha256_of_sorted_sha256_two_spaces_dot_slash_relative_path_newline",
        "priorArtifactDirectories": roots,
        "keyFiles": key_files,
    }


def compare_immutable(reference: dict[str, Any], current: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for step in PRIOR_STEPS:
        left = reference["priorArtifactDirectories"][step]
        right = current["priorArtifactDirectories"][step]
        for key in ("fileCount", "totalBytes", "aggregateSha256"):
            if left[key] != right[key]:
                errors.append(
                    f"immutable {step} {key} changed: {left[key]} != {right[key]}"
                )
    for name, left in reference["keyFiles"].items():
        right = current["keyFiles"][name]
        if left["sha256"] != right["sha256"] or left["bytes"] != right["bytes"]:
            errors.append(f"immutable key file changed: {name}")
    return errors


def verify_source_snapshots(config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metric_lines: list[dict[str, Any]] = []
    for source_id in ("IIGR_CORRECTED_SOURCE", "PHIRL_REGULARIZED_SOURCE"):
        specification = config["sourceSnapshots"][source_id]
        checkout = Path(specification["localCheckout"])
        head = git_output(checkout, "rev-parse", "HEAD")
        tree = git_output(checkout, "rev-parse", "HEAD^{tree}")
        for relative, expected in specification["files"].items():
            path = checkout / relative
            actual_sha = sha256_file(path)
            actual_blob = git_output(checkout, "rev-parse", f"HEAD:{relative}")
            rows.append(
                {
                    "sourceId": source_id,
                    "path": relative,
                    "expectedCommit": specification["commit"],
                    "actualCommit": head,
                    "expectedTree": specification["tree"],
                    "actualTree": tree,
                    "expectedSha256": expected["sha256"],
                    "actualSha256": actual_sha,
                    "expectedGitBlob": expected["gitBlob"],
                    "actualGitBlob": actual_blob,
                    "passed": head == specification["commit"]
                    and tree == specification["tree"]
                    and actual_sha == expected["sha256"]
                    and actual_blob == expected["gitBlob"],
                }
            )
        for line_number, line in enumerate(
            (checkout / "main.py").read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line.strip().startswith(
                (
                    'info["synergy"] =',
                    'info["causation"] =',
                    'info["integrated"] =',
                    'info["emergence"] =',
                )
            ):
                metric_lines.append(
                    {
                        "sourceId": source_id,
                        "path": "main.py",
                        "line": line_number,
                        "text": line.strip(),
                        "mainFileSha256": sha256_file(checkout / "main.py"),
                    }
                )
    safe = config["sourceSnapshots"]["safeLattice"]
    rows.append(
        {
            "sourceId": "SAFE_JSON_LATTICE",
            "path": str(SAFE_LATTICE),
            "expectedSha256": safe["sha256"],
            "actualSha256": sha256_file(SAFE_LATTICE),
            "passed": sha256_file(SAFE_LATTICE) == safe["sha256"],
        }
    )
    if not all(row["passed"] for row in rows):
        raise RuntimeError("one or more pinned source identities changed")
    if len(metric_lines) != 8:
        raise RuntimeError(
            f"expected exactly eight source metric assignment lines, found {len(metric_lines)}"
        )
    return {
        "schema": "eidosoma.e01.s12d.source_snapshot_manifest.v1",
        "sourceRelationship": "SOURCE_INFORMED_METRIC_IDENTITY",
        "sources": rows,
        "metricAssignmentLines": metric_lines,
        "safeJsonOnlyForScientificExecution": True,
        "success": True,
    }


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "researchStepId": "S12D",
        "preregistrationVersion": VERSION,
        "evidenceClass": "SOURCE_INFORMED_METRIC_IDENTITY_CONFIRMATION",
        "sourceRelationship": "SOURCE_INFORMED_METRIC_IDENTITY",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            errors.append(f"{key} must equal {value!r}")
    if config["scopeBoundary"]["exactUntouchedConfirmationMatrixCount"] != 24:
        errors.append("confirmation matrix count must be exactly 24")
    if (
        config["scopeBoundary"]["s13StatusThroughout"]
        != "BLOCKED_PENDING_S12D_HUMAN_REVIEW"
    ):
        errors.append("S13 must remain blocked pending S12D review")
    if config["metricEquivalenceGate"]["expectedRows"] != 40:
        errors.append("source-metric identity gate must contain exactly 40 rows")
    if config["metricEquivalenceGate"]["maximumAbsoluteDifferenceAtMost"] != 1e-12:
        errors.append("source-metric component tolerance must remain 1e-12")
    if (
        config["randomness"]["rootSeedHex"]
        != "14e4e325819ebcda15c9bba605859da22a19a88d283d8c76cc7b859270c8c36f"
    ):
        errors.append("unexpected S12D root seed")
    if len(config["randomness"]["exactMatrixIndices"]) != 24:
        errors.append("exact confirmation matrix index list must contain 24 entries")
    if (
        config["statistics"]["bootstrapReplicates"] != 4096
        or config["statistics"]["circularShiftReplicates"] != 4096
    ):
        errors.append("all inferential resampling counts must remain 4096")
    required = config["requiredArtifacts"]["files"]
    if len(required) != len(set(required)):
        errors.append("required artifact paths must be unique")
    forbidden = set(config["forbiddenIdentities"])
    if forbidden != {
        "AUTHOR_PRIMARY",
        "PAPER_PRIMARY",
        "EXACT_AUTHOR_IMPLEMENTATION",
        "EXACT_GARD_REPLICATION",
    }:
        errors.append("forbidden identity vocabulary changed")
    return errors


def dirty_paths() -> set[str]:
    paths: set[str] = set()
    output = git_output(REPO, "status", "--porcelain=v1", "--untracked-files=all")
    for line in output.splitlines():
        raw = line[3:]
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        paths.add(raw)
    return paths


def freeze() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    errors = validate_config(config)
    unexpected = dirty_paths() - ALLOWED_DESIGN_PATHS
    # RESEARCH_PLAN is outside the repository checkout and intentionally mutable.
    if unexpected:
        errors.append(
            f"unexpected repository changes before design freeze: {sorted(unexpected)}"
        )
    source_manifest = verify_source_snapshots(config)
    snapshot = immutable_snapshot()
    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    reference_path = STEP_ROOT / "immutable_prior_audit.json"
    if reference_path.exists():
        errors.extend(
            compare_immutable(json.loads(reference_path.read_text()), snapshot)
        )
    else:
        write_json(reference_path, snapshot)
    if errors:
        raise RuntimeError(
            "S12D preregistration validation failed: " + "; ".join(errors)
        )
    shutil.copyfile(CONFIG, STEP_ROOT / "preregistration.yaml")
    write_json(STEP_ROOT / "source_snapshot_manifest.json", source_manifest)
    record = {
        "schema": "eidosoma.e01.s12d.preregistration_record.v1",
        "researchStepId": "S12D",
        "preregistrationVersion": VERSION,
        "status": "VALIDATED_AND_FROZEN_BEFORE_OUTCOMES",
        "dateFrozenUtc": datetime.now(timezone.utc).isoformat(),
        "sourceConfigPath": str(CONFIG),
        "sourceConfigSha256": sha256_file(CONFIG),
        "artifactPreregistrationSha256": sha256_file(
            STEP_ROOT / "preregistration.yaml"
        ),
        "priorEvidenceImmutable": True,
        "pinnedSourcesVerified": True,
        "existingS12CDiagnosticOpened": False,
        "gardInputOpened": False,
        "confirmationTrajectoryGenerated": False,
        "s13Status": "BLOCKED_PENDING_S12D_HUMAN_REVIEW",
        "validationErrors": [],
        "success": True,
    }
    write_json(STEP_ROOT / "preregistration_record.json", record)
    return record


def lock() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    errors = validate_config(config)
    if dirty_paths():
        errors.append("implementation lock requires a clean repository worktree")
    head = git_output(REPO, "rev-parse", "HEAD")
    branch = git_output(REPO, "branch", "--show-current")
    git_output(REPO, "fetch", "origin", branch)
    remote = git_output(REPO, "rev-parse", f"origin/{branch}")
    if branch != config["implementationLock"]["branch"]:
        errors.append(f"unexpected branch {branch}")
    if head != remote:
        errors.append(f"local HEAD {head} is not pushed remote HEAD {remote}")
    locked_files: list[dict[str, Any]] = []
    for relative in config["implementationLock"]["lockedFiles"]:
        path = REPO / relative
        if not path.is_file():
            errors.append(f"missing locked file {relative}")
            continue
        locked_files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    reference = json.loads((STEP_ROOT / "immutable_prior_audit.json").read_text())
    errors.extend(compare_immutable(reference, immutable_snapshot()))
    if errors:
        raise RuntimeError("S12D implementation lock failed: " + "; ".join(errors))
    payload = {
        "schema": "eidosoma.e01.s12d.implementation_lock.v1",
        "researchStepId": "S12D",
        "preregistrationVersion": VERSION,
        "status": "LOCKED_CLEAN_AND_PUSHED_BEFORE_SOURCE_METRIC_EQUIVALENCE",
        "lockedAtUtc": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "headCommit": head,
        "remoteHeadCommit": remote,
        "cleanWorktree": True,
        "pushedHead": True,
        "lockedFiles": locked_files,
        "scientificCodeChangesAfterLockForbidden": True,
        "gardInputOpened": False,
        "existingDiagnosticOpened": False,
        "success": True,
    }
    write_json(STEP_ROOT / "implementation_lock.json", payload)
    return payload


def verify_lock() -> dict[str, Any]:
    lock_path = STEP_ROOT / "implementation_lock.json"
    if not lock_path.is_file():
        raise RuntimeError("S12D implementation lock does not exist")
    payload = json.loads(lock_path.read_text())
    errors: list[str] = []
    if dirty_paths():
        errors.append(
            f"repository is dirty after implementation lock: {sorted(dirty_paths())}"
        )
    current_head = git_output(REPO, "rev-parse", "HEAD")
    remote_head = git_output(REPO, "rev-parse", f"origin/{payload['branch']}")
    if (
        current_head != payload["headCommit"]
        or remote_head != payload["remoteHeadCommit"]
    ):
        errors.append("local or remote implementation commit changed after lock")
    for item in payload["lockedFiles"]:
        path = REPO / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            errors.append(f"locked implementation changed: {item['path']}")
    reference = json.loads((STEP_ROOT / "immutable_prior_audit.json").read_text())
    errors.extend(compare_immutable(reference, immutable_snapshot()))
    if errors:
        raise RuntimeError("S12D lock verification failed: " + "; ".join(errors))
    return {
        "success": True,
        "headCommit": payload["headCommit"],
        "remoteHeadCommit": payload["remoteHeadCommit"],
        "cleanWorktree": True,
        "errors": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action", choices=["freeze", "lock", "verify-lock"], required=True
    )
    args = parser.parse_args()
    if args.action == "freeze":
        payload = freeze()
    elif args.action == "lock":
        payload = lock()
    else:
        payload = verify_lock()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
