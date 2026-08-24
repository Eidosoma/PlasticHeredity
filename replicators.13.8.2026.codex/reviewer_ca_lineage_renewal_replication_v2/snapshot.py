"""One-way snapshot from already-frozen local artifacts; never from source code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import (
    DEFAULT_ARTIFACTS,
    DEFAULT_UPSTREAM,
    PACKAGE_ROOT,
    SCHEMA_VERSION,
    atomic_write_bytes,
    atomic_write_json,
    load_json,
    sha256_file,
    sha256_json,
)


UPSTREAM_ALLOWLIST = {
    "input/local/DONORS.json": "local/DONORS.json",
    "input/local/HYPOTHESIS.json": "local/HYPOTHESIS.json",
    "input/local/LAUNCH_RESETS.json": "local/LAUNCH_RESETS.json",
    "input/local/STAGE1_CALIBRATION.json": "local/STAGE1_CALIBRATION.json",
    "input/local/STAGE1_REGISTRATION.json": "local/STAGE1_REGISTRATION.json",
    "input/local/STAGE1_DECISION.json": "local/STAGE1_DECISION.json",
    "input/local/STAGE2_REGISTRATION.json": "local/STAGE2_REGISTRATION.json",
    "input/local/STAGE2_DECISION.json": "local/STAGE2_DECISION.json",
    "input/newideas/protocols/CA_MOTIF_LINEAGE_STAGE3R_PROTOCOL.md": (
        "context/CA_MOTIF_LINEAGE_STAGE3R_PROTOCOL.md"
    ),
    "input/newideas/stage3r/COHORTS.json": "context/STAGE3R_COHORTS.json",
    "input/newideas/stage3r/DESIGN.json": "context/STAGE3R_DESIGN.json",
    "input/newideas/stage3r/RESULTS.json": "context/STAGE3R_RESULTS.json",
    "input/newideas/stage3r/STAGE_DECISION.json": "context/STAGE3R_STAGE_DECISION.json",
    "REGISTRATION.json": "v1/REGISTRATION.json",
    "VALIDATION.json": "v1/VALIDATION.json",
    "confirmation/RESULTS.json": "v1/RESULTS.json",
    "confirmation/REPORTING_AMENDMENT.json": "v1/REPORTING_AMENDMENT.json",
}


FORENSIC_FINDINGS: dict[str, Any] = {
    "date_utc": "2026-08-23",
    "v1_disposition": "NON_COMPARABLE_MODEL_RUN",
    "v1_data_disposition": "preserved; descriptive evidence for its own implementation only",
    "comparability_breakers": [
        {
            "component": "reader_transfer_function",
            "v1": "constant strength for every positive advantage; magnitude ignored",
            "corrected": "strength*tanh(max(advantage,0)/9)",
        },
        {
            "component": "visible_reset",
            "v1": "pair-specific deterministic 50-percent-density board",
            "corrected": "sparse launch-specific donor initial_state_hex",
        },
        {
            "component": "sweep_order",
            "v1": ["CA step", "process noise", "reader"],
            "corrected": ["CA step", "reader", "process noise"],
        },
    ],
    "v1_reader_diagnostic": {
        "gains_with_identical_decisions": [0.5, 1.0, 2.0, 4.0],
        "no_rewrite_crossover_by_generation": {
            "1": 0.697917,
            "2": 0.700684,
            "4": 0.702637,
            "8": 0.699544,
            "16": 0.704427,
        },
    },
    "retained_source_data_audit": {
        "pair_count": 96,
        "unique_pair_ids": True,
        "design_bindings_consistent": True,
        "probability_granularity": "multiples of 1/64",
        "aggregate_recomputation_exact": True,
        "arithmetic_or_integrity_bug_found": False,
    },
    "source_inspection": {
        "authorization": "one-time narrow inspection",
        "role": "recover operational hypothesis specification only",
        "evidential_role": "none",
        "further_source_access": "prohibited for this campaign",
        "frozen_specification": "SOURCE_SPEC.md",
    },
}


def prepare_snapshot(
    upstream_root: Path | None = None, artifacts_root: Path | None = None
) -> dict[str, Any]:
    upstream = (upstream_root or DEFAULT_UPSTREAM).resolve()
    artifacts = (artifacts_root or DEFAULT_ARTIFACTS).resolve()
    if (artifacts / "REGISTRATION.json").exists():
        raise RuntimeError("the v2 input snapshot is frozen by registration")
    input_root = artifacts / "input"
    files: dict[str, str] = {}
    for source_name, destination_name in UPSTREAM_ALLOWLIST.items():
        source = upstream / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() not in {".json", ".md"}:
            raise ValueError(f"non-data/document upstream file rejected: {source_name}")
        destination = input_root / destination_name
        atomic_write_bytes(destination, source.read_bytes())
        files[destination_name] = sha256_file(destination)
    for name in ("SOURCE_SPEC.md", "FORENSIC_AUDIT.md"):
        source = PACKAGE_ROOT / name
        destination = input_root / "forensic" / name
        atomic_write_bytes(destination, source.read_bytes())
        files[f"forensic/{name}"] = sha256_file(destination)
    atomic_write_json(input_root / "forensic/FORENSIC_RECORD.json", FORENSIC_FINDINGS)
    files["forensic/FORENSIC_RECORD.json"] = sha256_file(
        input_root / "forensic/FORENSIC_RECORD.json"
    )
    snapshot_digest = sha256_json(files)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_digest": snapshot_digest,
        "files": files,
        "upstream": "sealed local v1 artifacts only",
        "original_source_tree_accessed": False,
        "source_code_opened_hashed_imported_or_executed": False,
        "source_outcomes_evidential_role": "none; contextual protocol binding only",
        "v1_outcomes_evidential_role": "none for intended-mechanism adjudication",
        "one_time_source_spec": "forensic/SOURCE_SPEC.md",
    }
    atomic_write_json(input_root / "MANIFEST.json", manifest)
    return verify_snapshot(input_root)


def verify_snapshot(input_root: Path) -> dict[str, Any]:
    manifest = load_json(input_root / "MANIFEST.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("snapshot schema mismatch")
    actual = {name: sha256_file(input_root / name) for name in manifest["files"]}
    if actual != manifest["files"]:
        raise ValueError("v2 input snapshot hash mismatch")
    if sha256_json(actual) != manifest.get("snapshot_digest"):
        raise ValueError("v2 snapshot digest mismatch")
    return manifest
