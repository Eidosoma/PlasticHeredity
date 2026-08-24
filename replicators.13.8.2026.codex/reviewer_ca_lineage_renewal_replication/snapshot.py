"""One-way, data/docs-only adapter for the Stage-3R hypothesis source."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import (
    DEFAULT_ARTIFACTS,
    DEFAULT_LOCAL_BASE,
    SCHEMA_VERSION,
    atomic_write_bytes,
    atomic_write_json,
    load_json,
    sha256_file,
    sha256_json,
)


# Explicitly limited to prose and JSON artifacts.  No globbing is permitted.
NEWIDEAS_ALLOWLIST = {
    "CA_MOTIF_LINEAGE_STAGE3_PROTOCOL.md": "protocols/CA_MOTIF_LINEAGE_STAGE3_PROTOCOL.md",
    "CA_MOTIF_LINEAGE_STAGE3R_PROTOCOL.md": "protocols/CA_MOTIF_LINEAGE_STAGE3R_PROTOCOL.md",
    "results/ca-motif-lineage-stage-3/COHORTS.json": "stage3/COHORTS.json",
    "results/ca-motif-lineage-stage-3/DESIGN.json": "stage3/DESIGN.json",
    "results/ca-motif-lineage-stage-3/RESULTS.json": "stage3/RESULTS.json",
    "results/ca-motif-lineage-stage-3/REPORT.md": "stage3/REPORT.md",
    "results/ca-motif-lineage-stage-3/STAGE_DECISION.json": "stage3/STAGE_DECISION.json",
    "results/ca-motif-lineage-stage-3r/COHORTS.json": "stage3r/COHORTS.json",
    "results/ca-motif-lineage-stage-3r/CONFIRMATION_DESIGN.json": "stage3r/CONFIRMATION_DESIGN.json",
    "results/ca-motif-lineage-stage-3r/DESIGN.json": "stage3r/DESIGN.json",
    "results/ca-motif-lineage-stage-3r/DIAGNOSTIC.json": "stage3r/DIAGNOSTIC.json",
    "results/ca-motif-lineage-stage-3r/FIT_AUDIT.json": "stage3r/FIT_AUDIT.json",
    "results/ca-motif-lineage-stage-3r/QUALIFICATION.json": "stage3r/QUALIFICATION.json",
    "results/ca-motif-lineage-stage-3r/REPAIR_MODELS.json": "stage3r/REPAIR_MODELS.json",
    "results/ca-motif-lineage-stage-3r/RESULTS.json": "stage3r/RESULTS.json",
    "results/ca-motif-lineage-stage-3r/SCREEN.json": "stage3r/SCREEN.json",
    "results/ca-motif-lineage-stage-3r/SELECTION_DECISION.json": "stage3r/SELECTION_DECISION.json",
    "results/ca-motif-lineage-stage-3r/STAGE_DECISION.json": "stage3r/STAGE_DECISION.json",
    "results/ca-motif-lineage-stage-3r/REPORT.md": "stage3r/REPORT.md",
    "results/ca-motif-lineage-stage-3r/LAY_SUMMARY.md": "stage3r/LAY_SUMMARY.md",
}

LOCAL_ALLOWLIST = {
    "input/DONORS.json": "DONORS.json",
    "input/HYPOTHESIS.json": "HYPOTHESIS.json",
    "input/LAUNCH_RESETS.json": "LAUNCH_RESETS.json",
    "input/MANIFEST.json": "BASE_INPUT_MANIFEST.json",
    "cohorts/FRESH_PAIR_POOL.json": "FRESH_PAIR_POOL.json",
    "stage1/CALIBRATION.json": "STAGE1_CALIBRATION.json",
    "stage1/REGISTRATION.json": "STAGE1_REGISTRATION.json",
    "stage1/RESULTS.json": "STAGE1_RESULTS.json",
    "stage1/STAGE_DECISION.json": "STAGE1_DECISION.json",
    "stage1/MANIFEST.json": "STAGE1_MANIFEST.json",
    "stage2/REGISTRATION.json": "STAGE2_REGISTRATION.json",
    "stage2/RESULTS.json": "STAGE2_RESULTS.json",
    "stage2/STAGE_DECISION.json": "STAGE2_DECISION.json",
    "stage2/MANIFEST.json": "STAGE2_MANIFEST.json",
}


def _copy_allowlist(source: Path, destination: Path, mapping: dict[str, str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative, local_name in mapping.items():
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in {".json", ".md"}:
            raise ValueError(f"non-data/document source rejected: {relative}")
        target = destination / local_name
        atomic_write_bytes(target, path.read_bytes())
        hashes[local_name] = sha256_file(target)
    return hashes


def build_snapshot(
    source_root: Path,
    artifacts_root: Path | None = None,
    local_base: Path | None = None,
) -> dict[str, Any]:
    artifacts = (artifacts_root or DEFAULT_ARTIFACTS).resolve()
    if (artifacts / "REGISTRATION.json").exists():
        raise RuntimeError("the data/docs snapshot is frozen by an existing registration")
    input_root = artifacts / "input"
    source_root = source_root.resolve()
    local_base = (local_base or DEFAULT_LOCAL_BASE).resolve()
    newideas_hashes = _copy_allowlist(
        source_root, input_root / "newideas", NEWIDEAS_ALLOWLIST
    )
    local_hashes = _copy_allowlist(local_base, input_root / "local", LOCAL_ALLOWLIST)
    snapshot_digest = sha256_json(
        {"newideas_data_docs": newideas_hashes, "local_frozen_evidence": local_hashes}
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_digest": snapshot_digest,
        "newideas_data_docs": newideas_hashes,
        "local_frozen_evidence": local_hashes,
        "newideas_files_opened": sorted(NEWIDEAS_ALLOWLIST),
        "source_extensions_allowed": [".json", ".md"],
        "source_code_opened": False,
        "source_code_hashed": False,
        "source_code_imported": False,
        "source_code_executed": False,
        "evidential_role": "hypothesis, fixed contract, and donor exclusions only",
    }
    atomic_write_json(input_root / "MANIFEST.json", manifest)
    atomic_write_json(
        input_root / "READ_AUDIT.json",
        {
            "files": sorted(NEWIDEAS_ALLOWLIST),
            "allowed_kinds": ["JSON data", "Markdown prose"],
            "code_access": "none",
        },
    )
    verify_snapshot(input_root)
    return manifest


def verify_snapshot(input_root: Path) -> dict[str, Any]:
    manifest = load_json(input_root / "MANIFEST.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("snapshot schema mismatch")
    actual_newideas = {
        name: sha256_file(input_root / "newideas" / name)
        for name in manifest["newideas_data_docs"]
    }
    actual_local = {
        name: sha256_file(input_root / "local" / name)
        for name in manifest["local_frozen_evidence"]
    }
    if actual_newideas != manifest["newideas_data_docs"]:
        raise ValueError("NewIdeas data/docs snapshot hash mismatch")
    if actual_local != manifest["local_frozen_evidence"]:
        raise ValueError("local evidence snapshot hash mismatch")
    expected = sha256_json(
        {"newideas_data_docs": actual_newideas, "local_frozen_evidence": actual_local}
    )
    if expected != manifest.get("snapshot_digest"):
        raise ValueError("snapshot digest mismatch")
    return manifest
