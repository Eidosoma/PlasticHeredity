"""One-way data/document snapshot builder for the clean-room firewall.

Only this module may read the historical research directory.  It accepts an
explicit root, checks an allowlist of data and documentation paths, extracts a
minimal local bundle, and never copies or opens implementation source.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from .contract import (
    SCHEMA_VERSION,
    atomic_write_json,
    load_json,
    sha256_file,
    sha256_json,
)


SOURCE_FILES = (
    "CA_CARRIER_V3_PROTOCOL.md",
    "CA_LINEAGE_FIELD_PROTOCOL.md",
    "CA_MOTIF_LINEAGE_STAGE1_PROTOCOL.md",
    "CA_MOTIF_LINEAGE_STAGE2_PROTOCOL.md",
    "results/ca-carrier-round-3/DESIGN.json",
    "results/ca-carrier-round-3/NARROW_COHORTS.json",
    "results/ca-carrier-round-3/NARROW_HYPOTHESIS.json",
    "results/ca-carrier-round-3/REPORT.md",
    "results/ca-carrier-round-3/narrow_acquire/checkpoints/launch-0.json",
    "results/ca-carrier-round-3/narrow_acquire/checkpoints/launch-1.json",
    "results/ca-carrier-round-3/narrow_acquire/checkpoints/launch-2.json",
    "results/ca-carrier-round-3/narrow_acquire/checkpoints/launch-3.json",
    "results/ca-lineage-field-round-4/CALIBRATION.json",
    "results/ca-lineage-field-round-4/COHORTS.json",
    "results/ca-lineage-field-round-4/REPORT.md",
    "results/ca-motif-lineage-stage-1/CALIBRATION.json",
    "results/ca-motif-lineage-stage-1/COHORTS.json",
    "results/ca-motif-lineage-stage-1/DESIGN.json",
    "results/ca-motif-lineage-stage-1/REPORT.md",
    "results/ca-motif-lineage-stage-1/RESULTS.json",
    "results/ca-motif-lineage-stage-1/SELECTION.json",
    "results/ca-motif-lineage-stage-1/STAGE_DECISION.json",
    "results/ca-motif-lineage-stage-1/screen/checkpoints/discovery-0000.json",
    "results/ca-motif-lineage-stage-2/COHORTS.json",
    "results/ca-motif-lineage-stage-2/DESIGN.json",
    "results/ca-motif-lineage-stage-2/REPORT.md",
    "results/ca-motif-lineage-stage-2/RESULTS.json",
    "results/ca-motif-lineage-stage-2/STAGE_DECISION.json",
    "results/ca-motif-lineage-stage-2/WRITER_AUDIT.json",
)

PAIR_PATTERN = re.compile(
    r"narrow-[0-9]{4}-life-31649-[0-3]-[0-9]+-life-31649-[0-3]-[0-9]+"
)


def _read_source_json(source_root: Path, relative: str) -> Any:
    if relative not in SOURCE_FILES or not relative.endswith(".json"):
        raise PermissionError(f"source path is not allowlisted JSON: {relative}")
    return load_json(source_root / relative)


def _read_source_text(source_root: Path, relative: str) -> str:
    if relative not in SOURCE_FILES or not relative.endswith(".md"):
        raise PermissionError(f"source path is not allowlisted documentation: {relative}")
    return (source_root / relative).read_text(encoding="utf-8")


def _find_pairs(value: Any) -> set[str]:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return set(PAIR_PATTERN.findall(text))


def _minimal_donor(donor: dict[str, Any]) -> dict[str, Any]:
    return {
        "donor_id": donor["donor_id"],
        "prototype_label": donor["prototype_label"],
        "launch_index": int(donor["launch_index"]),
        "density": float(donor["density"]),
        "initial_state_hex": donor["initial_state_hex"],
        "ancestor_state_hex": donor["ancestor_state_hex"],
        "anchor_state_hex": donor["anchor_state_hex"],
        "donor_state_hex": donor["donor_state_hex"],
        "offspring_state_hex": donor["offspring_state_hex"],
        "anchor_terminal2x2": donor["anchor_compositions"]["terminal2x2"],
        "target_primary": donor["target_compositions"]["primary"],
        "target_terminal": donor["target_compositions"]["terminal2x2"],
    }


def _bundle_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    }


def verify_snapshot(snapshot_root: Path) -> dict[str, Any]:
    manifest_path = snapshot_root / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"snapshot manifest missing: {manifest_path}")
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("snapshot schema mismatch")
    actual = _bundle_files(snapshot_root)
    if actual != manifest.get("bundle_files"):
        raise ValueError("local snapshot file hashes do not match MANIFEST.json")
    if sha256_json(actual) != manifest.get("snapshot_digest"):
        raise ValueError("snapshot aggregate digest mismatch")
    return manifest


def build_snapshot(source_root: Path, artifacts_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    target = artifacts_root.resolve() / "input"
    if target.exists():
        return verify_snapshot(target)
    missing = [relative for relative in SOURCE_FILES if not (source_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"allowlisted source inputs missing: {missing}")
    invalid = [relative for relative in SOURCE_FILES if Path(relative).suffix not in {".json", ".md"}]
    if invalid:
        raise PermissionError(f"non-data/document source inputs rejected: {invalid}")

    source_hashes = {
        relative: sha256_file(source_root / relative) for relative in SOURCE_FILES
    }
    acquisitions = [
        _read_source_json(
            source_root,
            f"results/ca-carrier-round-3/narrow_acquire/checkpoints/launch-{launch}.json",
        )
        for launch in range(4)
    ]
    donors = [
        _minimal_donor(donor)
        for acquisition in acquisitions
        for donor in acquisition["result"]["donors"]
    ]
    donor_ids = [donor["donor_id"] for donor in donors]
    if len(donors) != 2048 or len(donor_ids) != len(set(donor_ids)):
        raise ValueError("expected 2,048 unique frozen acquisition donors")
    for launch, acquisition in enumerate(acquisitions):
        source_donors = acquisition["result"]["donors"]
        if any(
            donor["rule"] != 31649
            or donor["notation"] != "B13456/S0578"
            or int(donor["launch_index"]) != launch
            for donor in source_donors
        ):
            raise ValueError(f"unexpected substrate metadata in launch {launch}")
        if len({donor["initial_state_hex"] for donor in source_donors}) != 1:
            raise ValueError(f"launch {launch} does not have one frozen reset")

    narrow_cohorts = _read_source_json(
        source_root, "results/ca-carrier-round-3/NARROW_COHORTS.json"
    )
    narrow_hypothesis = _read_source_json(
        source_root, "results/ca-carrier-round-3/NARROW_HYPOTHESIS.json"
    )
    lineage_field_calibration = _read_source_json(
        source_root, "results/ca-lineage-field-round-4/CALIBRATION.json"
    )
    lineage_field_cohorts = _read_source_json(
        source_root, "results/ca-lineage-field-round-4/COHORTS.json"
    )
    stage1 = {
        name.lower(): _read_source_json(
            source_root, f"results/ca-motif-lineage-stage-1/{name}.json"
        )
        for name in (
            "CALIBRATION",
            "COHORTS",
            "DESIGN",
            "RESULTS",
            "SELECTION",
            "STAGE_DECISION",
        )
    }
    stage2 = {
        name.lower(): _read_source_json(
            source_root, f"results/ca-motif-lineage-stage-2/{name}.json"
        )
        for name in (
            "COHORTS",
            "DESIGN",
            "RESULTS",
            "STAGE_DECISION",
            "WRITER_AUDIT",
        )
    }
    fixture_raw = _read_source_json(
        source_root,
        "results/ca-motif-lineage-stage-1/screen/checkpoints/discovery-0000.json",
    )["result"]
    fixture_ids = (
        "contextual256-w16-s025-d08",
        "contextual256-w32-s025-d08",
        "motif_energy512-w16-s025-d08",
        "motif_energy512-w32-s025-d08",
    )
    parity_fixture = {
        "pair_id": fixture_raw["pair_id"],
        "carrier_mean_abs": {
            configuration_id: fixture_raw["results"][configuration_id]["carrier_mean_abs"]
            for configuration_id in fixture_ids
        },
    }

    pair_ids: set[str] = set()
    for value in (
        narrow_cohorts,
        lineage_field_cohorts,
        stage1,
        stage2,
        parity_fixture,
    ):
        pair_ids.update(_find_pairs(value))
    for relative in (
        "CA_CARRIER_V3_PROTOCOL.md",
        "CA_MOTIF_LINEAGE_STAGE1_PROTOCOL.md",
        "CA_MOTIF_LINEAGE_STAGE2_PROTOCOL.md",
    ):
        pair_ids.update(_find_pairs(_read_source_text(source_root, relative)))

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".input-snapshot-", dir=target.parent))
    try:
        atomic_write_json(temporary / "DONORS.json", {"donors": donors})
        atomic_write_json(
            temporary / "HYPOTHESIS.json",
            {
                "targets": narrow_hypothesis["targets"],
                "selection_basis": narrow_hypothesis["selection_basis"],
                "target_basis": narrow_hypothesis["target_basis"],
                "spatial_latch_benchmark": lineage_field_calibration["mechanisms"]
                ["latch"]["selected"],
            },
        )
        atomic_write_json(
            temporary / "LAUNCH_RESETS.json",
            {
                f"launch{launch}": acquisition["result"]["donors"][0][
                    "initial_state_hex"
                ]
                for launch, acquisition in enumerate(acquisitions)
            },
        )
        atomic_write_json(
            temporary / "LEGACY.json",
            {
                "narrow_cohorts": narrow_cohorts,
                "stage1": stage1,
                "stage2": stage2,
                "parity_fixture": parity_fixture,
            },
        )
        atomic_write_json(
            temporary / "HISTORICAL_PAIR_EXCLUSIONS.json",
            {
                "policy": (
                    "exclude every donor named by any retained round-3, Stage-1, "
                    "Stage-2, development, or parity pair available to the snapshot"
                ),
                "pair_ids": sorted(pair_ids),
            },
        )
        docs = temporary / "protocols"
        docs.mkdir()
        document_names = {
            "CA_CARRIER_V3_PROTOCOL.md": "CA_CARRIER_V3_PROTOCOL.md",
            "CA_LINEAGE_FIELD_PROTOCOL.md": "CA_LINEAGE_FIELD_PROTOCOL.md",
            "CA_MOTIF_LINEAGE_STAGE1_PROTOCOL.md": "CA_MOTIF_LINEAGE_STAGE1_PROTOCOL.md",
            "CA_MOTIF_LINEAGE_STAGE2_PROTOCOL.md": "CA_MOTIF_LINEAGE_STAGE2_PROTOCOL.md",
            "results/ca-carrier-round-3/REPORT.md": "CA_CARRIER_ROUND3_REPORT.md",
            "results/ca-lineage-field-round-4/REPORT.md": "CA_LINEAGE_FIELD_ROUND4_REPORT.md",
            "results/ca-motif-lineage-stage-1/REPORT.md": "CA_MOTIF_STAGE1_REPORT.md",
            "results/ca-motif-lineage-stage-2/REPORT.md": "CA_MOTIF_STAGE2_REPORT.md",
        }
        for relative, local_name in document_names.items():
            shutil.copyfile(source_root / relative, docs / local_name)
        bundle = _bundle_files(temporary)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "evidence_boundary": (
                "historical artifacts are hypothesis/parity inputs only; fresh "
                "daughter trajectories are the replication outcomes"
            ),
            "source_policy": "explicit JSON/Markdown allowlist; no source code opened",
            "source_files": source_hashes,
            "bundle_files": bundle,
            "snapshot_digest": sha256_json(bundle),
            "donor_count": len(donors),
            "historical_pair_exclusion_count": len(pair_ids),
        }
        atomic_write_json(temporary / "MANIFEST.json", manifest)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_snapshot(target)
