"""One-way, allow-listed snapshot of data and documents only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import (
    DEFAULT_ARTIFACTS,
    DEFAULT_LOCAL_INPUT,
    DEFAULT_SOURCE_ROOT,
    SCHEMA_VERSION,
    atomic_write_bytes,
    atomic_write_json,
    load_json,
    sha256_file,
    sha256_json,
)


LOCAL_ALLOWLIST = {
    "input/local/DONORS.json": "input/local/DONORS.json",
    "input/local/HYPOTHESIS.json": "input/local/HYPOTHESIS.json",
    "input/local/LAUNCH_RESETS.json": "input/local/LAUNCH_RESETS.json",
    "REFERENCE.json": "input/local/REFERENCE.json",
}

SOURCE_ALLOWLIST = {
    "CA_MOTIF_LINEAGE_STAGE4_PROTOCOL.md": "input/hypothesis/CA_MOTIF_LINEAGE_STAGE4_PROTOCOL.md",
    "results/ca-motif-lineage-stage-4/DESIGN.json": "input/hypothesis/STAGE4_DESIGN.json",
    "results/ca-motif-lineage-stage-4/CONFIRMATION_DESIGN.json": "input/hypothesis/STAGE4_CONFIRMATION_DESIGN.json",
    "results/ca-motif-lineage-stage-4/CODEC_MODELS.json": "input/hypothesis/CODEC_MODELS.json",
    "results/ca-motif-lineage-stage-4/CODEC_MODELS.npz": "input/hypothesis/CODEC_MODELS.npz",
}


def _copy_allowlist(
    source_root: Path,
    artifacts: Path,
    allowlist: dict[str, str],
    files: dict[str, str],
) -> None:
    for source_name, destination_name in allowlist.items():
        source = source_root / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() not in {".json", ".md", ".npz"}:
            raise ValueError(f"non-data/document input rejected: {source}")
        destination = artifacts / destination_name
        atomic_write_bytes(destination, source.read_bytes())
        files[str(Path(destination_name).relative_to("input"))] = sha256_file(destination)


def prepare_snapshot(
    *,
    artifacts_root: Path | None = None,
    local_root: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    artifacts = (artifacts_root or DEFAULT_ARTIFACTS).resolve()
    if (artifacts / "REGISTRATION.json").exists():
        raise RuntimeError("the input snapshot is frozen by registration")
    local = (local_root or DEFAULT_LOCAL_INPUT).resolve()
    source = (source_root or DEFAULT_SOURCE_ROOT).resolve()
    files: dict[str, str] = {}
    _copy_allowlist(local, artifacts, LOCAL_ALLOWLIST, files)
    _copy_allowlist(source, artifacts, SOURCE_ALLOWLIST, files)

    confirmation = load_json(artifacts / "input/hypothesis/STAGE4_CONFIRMATION_DESIGN.json")
    model_path = artifacts / "input/hypothesis/CODEC_MODELS.npz"
    protocol_path = artifacts / "input/hypothesis/CA_MOTIF_LINEAGE_STAGE4_PROTOCOL.md"
    if sha256_file(model_path) != confirmation.get("model_sha256"):
        raise ValueError("frozen codec archive does not match its documented SHA256")
    if sha256_file(protocol_path) != confirmation.get("protocol_sha256"):
        raise ValueError("frozen protocol does not match its documented SHA256")

    snapshot_digest = sha256_json(files)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_digest": snapshot_digest,
        "files": files,
        "adapters": {
            "local": "previous local clean-room data artifacts",
            "source": "NewIdeas Stage-4 data/documents allowlist",
        },
        "source_code_opened_imported_hashed_or_executed": False,
        "source_results_or_checkpoints_imported": False,
        "source_outcomes_evidential_role": "none",
        "source_role": "outcome-known hypothesis and frozen codec specification only",
        "fresh_confirmation_required": True,
    }
    atomic_write_json(artifacts / "input/MANIFEST.json", manifest)
    return verify_snapshot(artifacts / "input")


def verify_snapshot(input_root: Path) -> dict[str, Any]:
    manifest = load_json(input_root / "MANIFEST.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("snapshot schema mismatch")
    actual = {name: sha256_file(input_root / name) for name in manifest["files"]}
    if actual != manifest["files"]:
        raise ValueError("input snapshot hash mismatch")
    if sha256_json(actual) != manifest.get("snapshot_digest"):
        raise ValueError("input snapshot digest mismatch")
    if manifest.get("source_results_or_checkpoints_imported") is not False:
        raise ValueError("source results cannot enter the replication snapshot")
    return manifest
