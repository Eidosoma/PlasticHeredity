from __future__ import annotations

from pathlib import Path

import pytest

from reviewer_ca_lineage_renewal_replication.cohorts import (
    allocate,
    eligible_pairs,
    parse_newideas_pair,
)
from reviewer_ca_lineage_renewal_replication.contract import (
    CONDITIONS,
    SCHEMA_VERSION,
    checkpoint_envelope,
    read_checkpoint,
    seal_registration,
    semantic_seed,
    verify_registration,
    write_checkpoint,
)
from reviewer_ca_lineage_renewal_replication.snapshot import NEWIDEAS_ALLOWLIST


def test_source_allowlist_is_data_and_documents_only() -> None:
    assert NEWIDEAS_ALLOWLIST
    assert all(Path(path).suffix in {".json", ".md"} for path in NEWIDEAS_ALLOWLIST)


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


def test_registration_and_checkpoint_tamper_detection(tmp_path: Path) -> None:
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


def test_semantic_seeds_are_stable_and_role_separated() -> None:
    assert semantic_seed("pair", 2, "reader") == semantic_seed("pair", 2, "reader")
    assert semantic_seed("pair", 2, "reader") != semantic_seed("pair", 2, "noise")


def test_combined_local_and_newideas_donor_exclusion() -> None:
    pool = [
        {
            "pair_id": f"fresh-{index}",
            "a_donor_id": f"life-31649-0-{index}",
            "b_donor_id": f"life-31649-0-{100 + index}",
            "density_difference": 0.0,
        }
        for index in range(102)
    ]
    stage1 = {
        "cohorts": {
            "validation": [
                {
                    "pair_id": "local",
                    "a_donor_id": "life-31649-0-0",
                    "b_donor_id": "life-31649-0-100",
                }
            ]
        }
    }
    stage2 = {"cohorts": {}}
    source = {
        "prior_pair_ids_excluded": [
            "narrow-0001-life-31649-0-1-life-31649-0-101"
        ],
        "diagnostic_pair_ids": [],
        "selection_pair_ids": [],
        "confirmation_pair_ids": [],
    }
    eligible, audit = eligible_pairs(pool, stage1, stage2, source)
    # The synthetic identifier ranges overlap: exclusions hit pairs 0, 1, 100,
    # and 101, which also exercises donor-level rather than pair-ID exclusion.
    assert len(eligible) == 98
    assert audit["local_used_donors"] == 2
    assert audit["newideas_exposed_donors"] == 2
    cohorts = allocate(eligible)
    assert len(cohorts["quarantine"]) == 2
    assert len(cohorts["confirmation"]) == 96
    assert parse_newideas_pair(source["prior_pair_ids_excluded"][0]) == (
        "life-31649-0-1",
        "life-31649-0-101",
    )
