from __future__ import annotations

import numpy as np

from wagner_memory_cleanroom.engine import (
    apply_challenge,
    in_basin,
    longest_true_run,
    rollout_jax,
    sequential_sweep_numpy,
    signed_update,
    strict_destination,
)


def test_ties_retain_the_previous_expression():
    previous = np.asarray([-1, 1, 1, -1], dtype=np.int8)
    assert np.array_equal(signed_update(np.zeros(4), previous), previous)


def test_sequential_order_changes_later_gene_input():
    weights = np.asarray([[0, 1], [1, 0]], dtype=float)
    state = np.asarray([[1, -1]], dtype=np.int8)
    assert np.array_equal(sequential_sweep_numpy(weights, state), [[-1, -1]])


def test_gpu_kernel_contract_replays_on_cpu_backend():
    weights = np.eye(3, dtype=np.float32)
    initial = np.asarray([[1, -1, 1], [-1, 1, -1]], dtype=np.int8)
    field = np.zeros_like(initial, dtype=np.float32)
    key = np.asarray([12, 34], dtype=np.uint32)
    first = rollout_jax(weights, initial, field, sweeps=6, theta=0.0, flip_probability=0.0, key_data=key)
    second = rollout_jax(weights, initial, field, sweeps=6, theta=0.0, flip_probability=0.0, key_data=key)
    assert np.array_equal(first, second)
    assert np.array_equal(first[-1], initial)


def test_strict_destination_uses_consecutive_basin_residence():
    a = np.asarray([1, 1, -1], dtype=np.int8)
    b = -a
    history = np.repeat(a[None, None, :], 8, axis=0)
    correct, wrong = strict_destination(history, a, b, strict_run=8)
    assert correct.tolist() == [True]
    assert wrong.tolist() == [False]
    values = np.asarray([[1], [1], [0], [1], [1], [1]], dtype=bool)
    assert longest_true_run(values).tolist() == [3]


def test_challenges_are_bounded_and_replayable():
    states = np.ones((4, 10), dtype=np.int8)
    target = np.ones(10, dtype=np.int8)
    first = apply_challenge(states, target, "neutral_damage", neutral_damage_fraction=0.2, rng=np.random.default_rng(5))
    second = apply_challenge(states, target, "neutral_damage", neutral_damage_fraction=0.2, rng=np.random.default_rng(5))
    assert np.array_equal(first, second)
    assert np.all(np.sum(first != states, axis=1) == 2)
    forced = apply_challenge(states, target, "forced_break", neutral_damage_fraction=0.2, rng=np.random.default_rng(5))
    assert np.all(np.sum(forced != target, axis=1) >= 3)
    assert not np.any(in_basin(forced, target))

