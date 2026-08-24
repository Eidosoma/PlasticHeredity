from __future__ import annotations

from hashlib import sha256

import numpy as np

from wagner_memory_cleanroom_v2.config import load_registration
from wagner_memory_cleanroom_v2.engine import (
    apply_challenge,
    develop_one_cycle_jax,
    primary_destinations,
    rollout_adult_cycles_jax,
    rollout_latch_cycles_jax,
    rollout_noisy_adult_cycles_jax,
    sequential_sweep_numpy,
    signed_update,
    states_from_int,
    states_to_int,
    strict_destinations,
)
from wagner_memory_cleanroom_v2.rng import jax_key_data
from wagner_memory_cleanroom_v2.source import enumerate_landscape, generate_rulebook


def test_ties_and_sequential_update_contract() -> None:
    assert np.array_equal(
        signed_update(np.zeros(3), np.asarray([1, -1, 1], dtype=np.int8)),
        np.asarray([1, -1, 1], dtype=np.int8),
    )
    weights = np.asarray([[0, 1], [1, 0]], dtype=np.float32)
    result = sequential_sweep_numpy(weights, np.asarray([[1, -1]], dtype=np.int8))
    assert np.array_equal(result, np.asarray([[-1, -1]], dtype=np.int8))


def test_strict_destination_requires_eight_exact_consecutive_adult_cycles() -> None:
    target_a = np.asarray([1, -1], dtype=np.int8)
    target_b = -target_a
    history = np.repeat(target_a[None, None, :], 8, axis=0)
    a, b = strict_destinations(history, target_a, target_b, 8)
    assert a.tolist() == [True]
    assert b.tolist() == [False]
    history[4, 0] = target_b
    a, _ = strict_destinations(history, target_a, target_b, 8)
    assert a.tolist() == [False]


def test_prediction_destination_is_first_exact_point_held_for_three_cycles() -> None:
    target_a = np.asarray([1, 1], dtype=np.int8)
    target_b = -target_a
    other = np.asarray([1, -1], dtype=np.int8)
    other_two = -other
    point_states = np.stack((target_a, target_b, other, other_two))
    # Future 0 reaches an off-target point first. Future 1 alternates two
    # off-target points, which cannot certify, and then reaches A.
    history = np.asarray([
        [other, other],
        [other, other_two],
        [other, other],
        [target_a, target_a],
        [target_a, target_a],
        [target_a, target_a],
    ], dtype=np.int8)
    assert primary_destinations(
        history, target_a, target_b, point_states, stable_run=3
    ).tolist() == [3, 1]


def test_cycle_or_limit_development_cannot_certify_destination_or_strict_hold() -> None:
    target_a = np.asarray([1, 1], dtype=np.int8)
    target_b = -target_a
    history = np.repeat(target_a[None, None, :], 8, axis=0)
    valid = np.ones((8, 1), dtype=bool)
    valid[3, 0] = False
    point_states = np.stack((target_a, target_b))
    destination = primary_destinations(
        history, target_a, target_b, point_states, 8, valid
    )
    hold_a, _ = strict_destinations(history, target_a, target_b, 8, valid)
    assert destination.tolist() == [3]
    assert hold_a.tolist() == [False]


def test_source_landscape_uses_the_stored_float64_matrix() -> None:
    registration = load_registration("smoke")
    rulebook = generate_rulebook(0, registration.protocol, "source-test")
    assert rulebook.weights.dtype == np.float64
    accepted = rulebook.proposal_log[-1]
    assert accepted["weight_sha256"] == sha256(rulebook.weights.tobytes(order="C")).hexdigest()
    landscape = enumerate_landscape(rulebook.weights, 10, 100)
    a, b = (int(value) for value in states_to_int(np.stack((rulebook.target_a, rulebook.target_b))))
    assert a ^ b == 1023
    assert int(landscape.successor[a]) == a
    assert int(landscape.successor[b]) == b
    assert all(np.sum(midpoint != rulebook.target_a) == 5 for midpoint in rulebook.midpoints)
    assert np.array_equal(rulebook.midpoints[0], -rulebook.midpoints[1])
    forced_a, forced_b = (
        int(value) for value in states_to_int(np.stack((rulebook.forced_a, rulebook.forced_b)))
    )
    assert landscape.attractor_index[forced_a] != landscape.attractor_index[a]
    assert landscape.attractor_index[forced_b] != landscape.attractor_index[b]


def test_jax_development_matches_the_enumerated_float64_adult_table() -> None:
    registration = load_registration("smoke")
    rulebook = generate_rulebook(0, registration.protocol, "engine-table-test")
    codes = np.asarray([0, 1, 17, 255, 511, 1023], dtype=np.uint16)
    initial = states_from_int(codes, 10)
    adult, status, steps = develop_one_cycle_jax(
        rulebook.weights,
        initial,
        external_field=None,
        gamma_variance=0.0,
        expression_flip_probability=0.0,
        key_data=jax_key_data("engine-table-test", "development"),
        max_sweeps=100,
    )
    assert np.array_equal(adult, rulebook.adult_table()[codes])
    assert np.array_equal(status, rulebook.landscape.terminal_status[codes])
    assert np.array_equal(steps, rulebook.landscape.terminal_steps[codes])


def test_noisy_cycles_return_a_convergence_status_for_every_adult() -> None:
    registration = load_registration("smoke")
    rulebook = generate_rulebook(0, registration.protocol, "noisy-status-test")
    initial = np.repeat(rulebook.midpoints[0][None, :], 2, axis=0)
    adults, adult, statuses = rollout_noisy_adult_cycles_jax(
        rulebook.weights,
        initial,
        cycles=2,
        max_sweeps=100,
        gamma_variance=0.01,
        expression_flip_probability=0.05,
        key_data=jax_key_data("noisy-status-test", "cycles"),
    )
    assert adults.shape == (2, 2, 10)
    assert adult.shape == (2, 10)
    assert statuses.shape == (2, 2)
    assert set(np.unique(statuses)).issubset({0, 1, 2})


def test_full_hard_clamp_overrides_predevelopment_expression_flips() -> None:
    registration = load_registration("smoke")
    rulebook = generate_rulebook(0, registration.protocol, "hard-clamp-test")
    initial = np.repeat(rulebook.midpoints[0][None, :], 8, axis=0)
    target = np.repeat(rulebook.target_a[None, :], 8, axis=0)
    adult, status, steps = develop_one_cycle_jax(
        rulebook.weights,
        initial,
        external_field=None,
        gamma_variance=0.0,
        expression_flip_probability=1.0,
        key_data=jax_key_data("hard-clamp-test", "development"),
        max_sweeps=100,
        hard_mask=np.ones(10, dtype=bool),
        hard_values=target,
    )
    assert np.array_equal(adult, target)
    assert np.all(status == 1)
    assert np.all(steps == 1)


def test_first_cycle_carrier_read_is_discarded_but_sets_the_adult_trajectory() -> None:
    registration = load_registration("smoke")
    rulebook = generate_rulebook(0, registration.protocol, "carrier-read-test")
    initial = np.repeat(rulebook.midpoints[0][None, :], 4, axis=0)
    mark = np.repeat(rulebook.target_a[None, :], 4, axis=0).astype(np.float64)
    adults, adult, final_mark = rollout_adult_cycles_jax(
        rulebook.adult_table(),
        initial,
        cycles=8,
        expression_flip_probability=0.0,
        key_data=jax_key_data("carrier-read-test", "cycles"),
        read_mode="first",
        mark=mark,
        coupling=1.0,
        write_enabled=False,
    )
    target = np.repeat(rulebook.target_a[None, :], 4, axis=0)
    assert np.array_equal(adults[0], target)
    assert np.array_equal(adult, target)
    assert np.array_equal(final_mark, mark)
    point_states = states_from_int(
        [cycle[0] for cycle in rulebook.landscape.attractors if len(cycle) == 1], 10
    )
    assert np.all(primary_destinations(
        adults, rulebook.target_a, rulebook.target_b, point_states, 3
    ) == 1)
    hold_a, hold_b = strict_destinations(adults, rulebook.target_a, rulebook.target_b, 8)
    assert np.all(hold_a)
    assert not np.any(hold_b)


def test_latch_reads_until_ttl_expires_then_is_discarded() -> None:
    adult_table = states_from_int(np.arange(4, dtype=np.uint16), 2)
    initial = np.asarray([[-1, -1]], dtype=np.int8)
    carrier = np.asarray([[1, 1]], dtype=np.int8)
    ttl = np.asarray([[2, 2]], dtype=np.int16)
    zeros8 = np.zeros_like(carrier, dtype=np.int8)
    zeros16 = np.zeros_like(ttl, dtype=np.int16)
    adults, _, final_carrier, final_ttl, _, _ = rollout_latch_cycles_jax(
        adult_table,
        initial,
        carrier,
        ttl,
        zeros8,
        zeros16,
        cycles=4,
        expression_flip_probability=1.0,
        coupling=1.0,
        retention=2,
        threshold=1,
        read_enabled=True,
        rewrite=False,
        key_data=jax_key_data("latch-ttl-test"),
    )
    assert np.array_equal(adults[:2, 0], np.asarray([[1, 1], [1, 1]], dtype=np.int8))
    assert np.array_equal(adults[2:, 0], np.asarray([[-1, -1], [1, 1]], dtype=np.int8))
    assert np.all(final_carrier == 0)
    assert np.all(final_ttl == 0)


def test_forced_break_is_fixed_not_a_random_flip() -> None:
    states = np.asarray([[1, 1], [-1, -1]], dtype=np.int8)
    forced = np.asarray([1, -1], dtype=np.int8)
    first = apply_challenge(states, "forced_break", forced_state=forced, neutral_damage_fraction=.2, rng=np.random.default_rng(1))
    second = apply_challenge(states, "forced_break", forced_state=forced, neutral_damage_fraction=.2, rng=np.random.default_rng(999))
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.repeat(forced[None, :], 2, axis=0))
