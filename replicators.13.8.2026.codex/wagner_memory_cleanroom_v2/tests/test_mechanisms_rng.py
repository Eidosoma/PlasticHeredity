from __future__ import annotations

import numpy as np

from wagner_memory_cleanroom_v2.config import load_registration
from wagner_memory_cleanroom_v2.experiment import (
    _cell_id,
    _mark_arm,
    _lineage_snapshots,
    _targeted_mask,
    _trained_mark,
    _update_latch,
)
from wagner_memory_cleanroom_v2.rng import jax_key_data, seed64


def _empty_latch() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((1, 3), dtype=np.int8),
        np.zeros((1, 3), dtype=np.int16),
        np.zeros((1, 3), dtype=np.int8),
        np.zeros((1, 3), dtype=np.int16),
    )


def test_latch_is_written_from_trajectory_and_threshold_is_consecutive() -> None:
    carrier, ttl, pending, streak = _empty_latch()
    expression = np.asarray([[[1, -1, 1]]], dtype=np.int8)
    one, ttl_one, _, _ = _update_latch(
        carrier, ttl, pending, streak, expression,
        retention=16, threshold=1, rewrite=True,
    )
    assert np.array_equal(one, expression[-1])
    assert np.all(ttl_one == 16)

    carrier, ttl, pending, streak = _empty_latch()
    first, ttl, pending, streak = _update_latch(
        carrier, ttl, pending, streak, expression,
        retention=16, threshold=2, rewrite=True,
    )
    assert np.all(first == 0)
    second, _, _, _ = _update_latch(
        first, ttl, pending, streak, expression,
        retention=16, threshold=2, rewrite=True,
    )
    assert np.array_equal(second, expression[-1])


def test_no_rewrite_expires_carrier_after_sixteen_adult_observations() -> None:
    carrier = np.asarray([[1, -1]], dtype=np.int8)
    ttl = np.full((1, 2), 16, dtype=np.int16)
    pending = np.zeros_like(carrier)
    streak = np.zeros_like(ttl)
    trajectory = np.repeat(carrier[None, :, :], 16, axis=0)
    result, result_ttl, _, _ = _update_latch(
        carrier, ttl, pending, streak, trajectory,
        retention=16, threshold=1, rewrite=False,
    )
    assert np.all(result == 0)
    assert np.all(result_ttl == 0)


def test_no_rewrite_lineage_ttl_is_16_12_8_then_zero() -> None:
    from wagner_memory_cleanroom_v2.source import generate_rulebook

    registration = load_registration("smoke")
    rulebook = generate_rulebook(0, registration.protocol, "latch-lineage-test")
    snapshots, _ = _lineage_snapshots(
        registration, "carrier", rulebook, "A", 0, "no_rewrite", 0, 1
    )
    for checkpoint, expected in ((0, 16), (1, 12), (2, 8)):
        carrier, ttl = snapshots[checkpoint]
        assert np.all(carrier == rulebook.target_a)
        assert np.all(ttl == expected)
    carrier, ttl = snapshots[4]
    assert np.all(carrier == 0)
    assert np.all(ttl == 0)


def test_mark_is_written_from_adult_with_registered_half_life() -> None:
    adult = np.ones((1, 1, 2), dtype=np.int8)
    mark = _trained_mark(adult, half_life=4)
    assert np.allclose(mark, 1.0 - 2.0 ** (-1.0 / 4.0))


def test_mark_washout_age_advances_the_donor_mark_not_only_expression() -> None:
    from wagner_memory_cleanroom_v2.source import generate_rulebook

    registration = load_registration("smoke")
    rulebook = generate_rulebook(0, registration.protocol, "mark-aging-test")
    _, mark0, _, _, _ = _mark_arm(
        registration, "slow_mark", rulebook, "A", 0, "mark_transplant",
        0, 4, 0.5, 3, 0,
    )
    _, mark8, _, _, _ = _mark_arm(
        registration, "slow_mark", rulebook, "A", 0, "mark_transplant",
        8, 4, 0.5, 3, 0,
    )
    assert not np.array_equal(mark8, mark0)


def test_mark_ablation_removes_top_two_and_random_arm_removes_two() -> None:
    from wagner_memory_cleanroom_v2.source import generate_rulebook

    registration = load_registration("smoke")
    rulebook = generate_rulebook(0, registration.protocol, "mark-ablation-test")
    _, targeted, _, _, _ = _mark_arm(
        registration, "slow_mark", rulebook, "A", 0, "mark_ablation",
        0, 4, 0.5, 3, 0,
    )
    _, random, _, _, _ = _mark_arm(
        registration, "slow_mark", rulebook, "A", 0, "mark_random_ablation",
        0, 4, 0.5, 3, 0,
    )
    targeted_mask = _targeted_mask(rulebook, "A", 2)
    assert np.all(targeted[:, targeted_mask] == 0.0)
    assert np.all(np.count_nonzero(targeted, axis=1) == 8)
    assert np.all(np.count_nonzero(random, axis=1) == 8)


def test_semantic_rng_has_explicit_halves_and_no_arm_coordinate() -> None:
    registration = load_registration("smoke")
    master = registration.protocol["master_seed"]
    left = jax_key_data(master, "carrier", "assay", 0, 0, "A", "forced_break", 0, "latch", 4, 1)
    repeated = jax_key_data(master, "carrier", "assay", 0, 0, "A", "forced_break", 0, "latch", 4, 1)
    right_half = jax_key_data(master, "carrier", "assay", 0, 0, "A", "forced_break", 1, "latch", 4, 1)
    assert np.array_equal(left, repeated)
    assert not np.array_equal(left, right_half)
    assert seed64(master, "x", 0) != seed64(master, "x", 1)


def test_cell_id_includes_schedule_and_every_intervention_dimension() -> None:
    base = {
        "stage": "slow_mark", "source_id": 0, "midpoint": 0, "history": "A",
        "condition": "half-4.mu-0.5", "schedule": "screen", "arm": "reset_both",
        "challenge": "forced_break", "age": 8, "checkpoint": None, "theta": None,
        "half_life": 4, "coupling": .5, "half": 0,
    }
    mechanism = dict(base, schedule="mechanism")
    assert _cell_id(base) != _cell_id(mechanism)
