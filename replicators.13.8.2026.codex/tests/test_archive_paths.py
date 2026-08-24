from __future__ import annotations

from pathlib import Path

import pytest

from plastic_heredity.archive_paths import (
    CURRENT_REPOSITORY_ROOT,
    LEGACY_REPOSITORY_ROOTS,
    normalize_repository_paths,
    protocols_equal_after_relocation,
    relocated_path,
)


def test_legacy_repository_path_maps_to_current_file() -> None:
    relative = Path("results_intervention_replication/registration/registration.json")
    legacy = LEGACY_REPOSITORY_ROOTS[0] / relative
    assert relocated_path(legacy) == CURRENT_REPOSITORY_ROOT / relative


def test_current_repository_path_remains_current() -> None:
    current = CURRENT_REPOSITORY_ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    assert relocated_path(current) == current


def test_missing_relocation_target_is_rejected() -> None:
    legacy = LEGACY_REPOSITORY_ROOTS[0] / "does-not-exist" / "missing.json"
    with pytest.raises(FileNotFoundError):
        relocated_path(legacy)


def test_protocol_comparison_ignores_only_repository_root() -> None:
    relative = "results_intervention_replication/.p3_work"
    archived = {
        "protocol_id": "archived-id",
        "work": f"{LEGACY_REPOSITORY_ROOTS[0]}/{relative}",
        "matrices": 40,
    }
    reconstructed = {
        "protocol_id": "new-location-id",
        "work": f"{CURRENT_REPOSITORY_ROOT}/{relative}",
        "matrices": 40,
    }
    assert protocols_equal_after_relocation(archived, reconstructed)
    reconstructed["matrices"] = 41
    assert not protocols_equal_after_relocation(archived, reconstructed)


def test_normalization_does_not_rewrite_unrelated_absolute_path() -> None:
    unrelated = "/tmp/independent-artifact"
    assert normalize_repository_paths(unrelated) == unrelated
