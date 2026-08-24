from __future__ import annotations

from pathlib import Path

import pytest

from reviewer_ca_lineage_renewal_replication_v2.campaign import _context, _cohorts, register
from reviewer_ca_lineage_renewal_replication_v2.contract import (
    CONDITIONS,
    DEFAULT_ARTIFACTS,
    SCHEMA_VERSION,
    atomic_write_json,
    checkpoint_envelope,
    read_checkpoint,
    seal_registration,
    verify_registration,
    write_checkpoint,
)
from reviewer_ca_lineage_renewal_replication_v2.snapshot import UPSTREAM_ALLOWLIST


def test_input_adapter_is_data_and_documents_only() -> None:
    assert UPSTREAM_ALLOWLIST
    assert all(Path(path).suffix in {".json", ".md"} for path in UPSTREAM_ALLOWLIST)


def test_exact_causal_panel_is_frozen() -> None:
    assert CONDITIONS == [
        "intact",
        "zero_every_boundary",
        "shuffle_every_boundary",
        "read_disabled",
        "founder_write_disabled",
        "no_rewrite",
        "ablate_after_g2",
        "rescue_same_enter_g4",
        "rescue_opposite_enter_g4",
        "opposite_founder",
        "carrier_corruption_1",
    ]


def test_registration_checkpoint_and_nonfinite_tamper_detection(tmp_path: Path) -> None:
    registration = seal_registration(
        {"schema_version": SCHEMA_VERSION, "experiment": "test"}
    )
    verify_registration(registration)
    with pytest.raises(ValueError, match="digest"):
        verify_registration({**registration, "experiment": "altered"})
    path = tmp_path / "checkpoint.json"
    write_checkpoint(path, registration["design_digest"], {"value": 3})
    assert read_checkpoint(path, registration["design_digest"])["value"] == 3
    path.write_text(path.read_text().replace('"value": 3', '"value": 4'))
    with pytest.raises(ValueError, match="checksum"):
        read_checkpoint(path, registration["design_digest"])
    assert checkpoint_envelope("x", {"b": 2, "a": 1}) == checkpoint_envelope(
        "x", {"a": 1, "b": 2}
    )
    with pytest.raises(ValueError, match="nonfinite"):
        atomic_write_json(tmp_path / "bad.json", {"value": float("inf")})


def test_registered_untouched_matching_has_all_92_pairs() -> None:
    context = _context(DEFAULT_ARTIFACTS)
    cohorts, audit = _cohorts(context)
    assert len(cohorts["quarantine"]) == 2
    assert len(cohorts["confirmation"]) == 92
    assert audit["untouched_donors"] == 431
    assert audit["confirmation_pairs_by_launch"] == {
        "launch0": 20,
        "launch1": 30,
        "launch2": 21,
        "launch3": 21,
    }
    used = [
        pair[key]
        for pair in cohorts["confirmation"]
        for key in ("a_donor_id", "b_donor_id")
    ]
    assert len(used) == len(set(used)) == 184
    assert max(pair["density_difference"] for pair in cohorts["confirmation"]) <= 0.02


def test_registration_is_blocked_by_a_failed_test_audit(tmp_path: Path) -> None:
    atomic_write_json(tmp_path / "VALIDATION.json", {"valid": True})
    atomic_write_json(tmp_path / "PARITY.json", {"valid": True})
    atomic_write_json(tmp_path / "TEST_AUDIT.json", {"passed": False})
    with pytest.raises(ValueError, match="must pass"):
        register(tmp_path)
