from __future__ import annotations

import numpy as np

from wagner_memory_cleanroom.config import load_registration
from wagner_memory_cleanroom.rng import generator, jax_key_data, stable_permutation
from wagner_memory_cleanroom.source import generate_rulebook


def test_semantic_domains_are_replayable_and_separate():
    first = generator("seed", "future", 1).integers(0, 2**31, size=16)
    replay = generator("seed", "future", 1).integers(0, 2**31, size=16)
    other = generator("seed", "future", 2).integers(0, 2**31, size=16)
    assert np.array_equal(first, replay)
    assert not np.array_equal(first, other)
    assert np.array_equal(jax_key_data("seed", 1), jax_key_data("seed", 1))


def test_permutation_is_complete_and_replayable():
    first = stable_permutation(10, "seed", "shuffle")
    second = stable_permutation(10, "seed", "shuffle")
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(10))


def test_rulebook_generation_is_exactly_replayable_and_eligible():
    registration = load_registration("smoke")
    first = generate_rulebook(7, registration.protocol, "unit")
    second = generate_rulebook(7, registration.protocol, "unit")
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.target_a, second.target_a)
    assert np.array_equal(first.target_b, second.target_b)
    assert min(first.basin_a, first.basin_b) >= registration.engine["minimum_basin_fraction"]
    assert np.sum(first.target_a != first.target_b) >= registration.engine["minimum_target_hamming"]

