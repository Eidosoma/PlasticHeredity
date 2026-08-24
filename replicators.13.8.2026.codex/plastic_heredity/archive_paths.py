"""Relocation-safe access to checksum-sealed repository artifacts.

Historical registrations intentionally retain the absolute paths that were
part of their original canonical digests.  This module maps those paths only
when accessing files and normalizes repository roots only when comparing an
archived protocol with a freshly reconstructed equivalent.  It never rewrites
or re-hashes a sealed artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


CURRENT_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_REPOSITORY_ROOTS = (
    Path("/home/robert/Projects/replications/replicators.13.8.2026.codex"),
)
REPOSITORY_TOKEN = "${CODEX_REPOSITORY_ROOT}"


def _relative_to_known_root(path: Path) -> Path | None:
    for root in (CURRENT_REPOSITORY_ROOT, *LEGACY_REPOSITORY_ROOTS):
        try:
            return path.relative_to(root)
        except ValueError:
            continue
    return None


def relocated_path(value: str | Path, *, require_exists: bool = True) -> Path:
    """Return the current location for an archived repository-local path."""

    archived = Path(value)
    relative = _relative_to_known_root(archived)
    candidate = (
        CURRENT_REPOSITORY_ROOT / relative if relative is not None else archived
    )
    if require_exists and not candidate.exists():
        raise FileNotFoundError(
            f"archived path has no current relocation target: {archived} -> {candidate}"
        )
    return candidate


def normalize_repository_paths(value: Any) -> Any:
    """Replace current or historical repository roots with one stable token."""

    if isinstance(value, dict):
        return {
            key: normalize_repository_paths(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_repository_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_repository_paths(item) for item in value)
    if isinstance(value, str) and value.startswith("/"):
        path = Path(value)
        relative = _relative_to_known_root(path)
        if relative is not None:
            suffix = relative.as_posix()
            return REPOSITORY_TOKEN if suffix == "." else f"{REPOSITORY_TOKEN}/{suffix}"
    return value


def protocols_equal_after_relocation(
    archived: dict[str, Any], reconstructed: dict[str, Any]
) -> bool:
    """Compare protocol substance while ignoring only repository-root movement."""

    left = dict(normalize_repository_paths(archived))
    right = dict(normalize_repository_paths(reconstructed))
    left.pop("protocol_id", None)
    right.pop("protocol_id", None)
    return left == right
