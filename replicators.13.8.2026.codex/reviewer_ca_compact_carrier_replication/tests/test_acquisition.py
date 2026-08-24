from __future__ import annotations

import numpy as np

from reviewer_ca_compact_carrier_replication.acquisition import (
    generate_launch_candidates,
    match_donors,
)
from reviewer_ca_compact_carrier_replication.contract import DEFAULT_ARTIFACTS, load_json, sha256_json


def test_fresh_acquisition_is_ready_disjoint_and_historically_clean() -> None:
    acquisition = load_json(DEFAULT_ARTIFACTS / "ACQUISITION.json")
    cohorts = load_json(DEFAULT_ARTIFACTS / "COHORTS.json")
    audit = load_json(DEFAULT_ARTIFACTS / "ACQUISITION_AUDIT.json")
    assert acquisition["state"] == cohorts["state"] == audit["state"] == "READY"
    assert audit["generated_candidates"] == 2048
    assert audit["allocation_counts"] == {
        "engineering": 4,
        "confirmation": 128,
        "audit_reserve": 32,
    }
    assert audit["historical_state_overlap"] == 0
    used = [
        pair[key]
        for cohort in cohorts["cohorts"].values()
        for pair in cohort
        for key in ("a_donor_id", "b_donor_id")
    ]
    assert len(used) == len(set(used)) == 328


def test_small_acquisition_fixture_is_deterministic() -> None:
    hypothesis = load_json(DEFAULT_ARTIFACTS / "input/local/HYPOTHESIS.json")
    resets = load_json(DEFAULT_ARTIFACTS / "input/local/LAUNCH_RESETS.json")
    targets = {
        label: np.asarray(hypothesis["targets"]["primary"][label], dtype=np.float64)
        for label in ("A", "B")
    }
    first, first_audit = generate_launch_candidates(
        launch=0,
        reset_state_hex=resets["launch0"],
        targets=targets,
        historical_states=set(),
        count=12,
    )
    second, second_audit = generate_launch_candidates(
        launch=0,
        reset_state_hex=resets["launch0"],
        targets=targets,
        historical_states=set(),
        count=12,
    )
    assert sha256_json(first) == sha256_json(second)
    assert first_audit == second_audit


def test_matching_is_deterministic_and_donor_disjoint() -> None:
    donors = []
    for launch in range(2):
        for label, offset in (("A", 0.0), ("B", 0.003)):
            for index in range(5):
                donors.append(
                    {
                        "donor_id": f"{launch}-{label}-{index}",
                        "launch_index": launch,
                        "prototype_label": label,
                        "density": 0.4 + 0.01 * index + offset,
                    }
                )
    first = match_donors(donors)
    second = match_donors(reversed(donors))
    assert first == second
    used = [pair[key] for pair in first for key in ("a_donor_id", "b_donor_id")]
    assert len(first) == 10
    assert len(used) == len(set(used))
    assert all(pair["density_difference"] <= 0.02 for pair in first)
