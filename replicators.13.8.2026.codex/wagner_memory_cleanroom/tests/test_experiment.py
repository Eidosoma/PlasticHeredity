from __future__ import annotations

import numpy as np

from wagner_memory_cleanroom.config import load_registration
from wagner_memory_cleanroom.experiment import (
    _advance_carrier,
    _founder_carrier,
    run_carrier_source,
    run_state_source,
)
from wagner_memory_cleanroom.source import generate_rulebook


def test_state_self_and_transplant_are_pathwise_identical():
    registration = load_registration("smoke")
    rulebook = generate_rulebook(0, registration.protocol, "state")
    records = run_state_source(registration, "state", rulebook)
    keys = ("history", "challenge", "age", "half", "n", "correct", "wrong", "both", "unresolved")
    self_rows = sorted(tuple(row[key] for key in keys) for row in records if row["writer"] == "hard-theta-0" and row["arm"] == "self")
    transplant = sorted(tuple(row[key] for key in keys) for row in records if row["writer"] == "hard-theta-0" and row["arm"] == "state_transplant" and row["age"] == 0 and row["challenge"] == "neutral_damage")
    assert self_rows == transplant


def test_no_rewrite_expires_but_rewrite_renews_carrier():
    registration = load_registration("smoke")
    rulebook = generate_rulebook(1, registration.protocol, "carrier")
    full, ttl, _ = _founder_carrier(registration, "carrier", rulebook, "A", "natural_full", 8, 0)
    renewed, renewed_ttl = _advance_carrier(registration, "carrier", rulebook, "A", "natural_full", full.copy(), ttl.copy(), 4, 0)
    expired, expired_ttl = _advance_carrier(registration, "carrier", rulebook, "A", "no_rewrite", full.copy(), ttl.copy(), 4, 0)
    assert np.any(renewed != 0)
    assert np.any(renewed_ttl > 0)
    assert np.all(expired == 0)
    assert np.all(expired_ttl == 0)


def test_ablation_erases_and_rescue_reinstates_a_carrier():
    registration = load_registration("smoke")
    rulebook = generate_rulebook(2, registration.protocol, "carrier")
    founder, ttl, _ = _founder_carrier(registration, "carrier", rulebook, "A", "natural_full", 8, 0)
    ablated, _ = _advance_carrier(registration, "carrier", rulebook, "A", "ablate_generation_2", founder.copy(), ttl.copy(), 2, 0)
    rescued, _ = _advance_carrier(registration, "carrier", rulebook, "A", "ablate_2_rescue_3", founder.copy(), ttl.copy(), 3, 0)
    assert np.all(ablated == 0)
    assert np.any(rescued != 0)


def test_carrier_records_include_every_registered_arm_checkpoint_and_challenge():
    registration = load_registration("smoke")
    rulebook = generate_rulebook(3, registration.protocol, "carrier")
    records = run_carrier_source(registration, "carrier", rulebook)
    assert {row["arm"] for row in records} == set(registration.protocol["carrier"]["arms"])
    assert {row["checkpoint"] for row in records} == set(registration.protocol["carrier"]["checkpoints"])
    assert {row["challenge"] for row in records} == set(registration.protocol["carrier"]["challenges"])
    expected = 2 * 2 * len(registration.protocol["carrier"]["arms"]) * len(registration.protocol["carrier"]["checkpoints"]) * len(registration.protocol["carrier"]["challenges"])
    assert len(records) == expected

