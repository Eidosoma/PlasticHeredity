from __future__ import annotations

import numpy as np

from grn_f12_realistic.network import sample_network
from grn_f12_realistic.rng import generator, stable_seed


def test_semantic_rng_is_repeatable_and_domain_separated():
    first = generator("master", "network", "continuous", 7).normal(size=20)
    second = generator("master", "network", "continuous", 7).normal(size=20)
    other = generator("master", "future", "continuous", 7).normal(size=20)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert stable_seed("master", "a", 1) != stable_seed("master", "a", 2)


def test_network_has_balanced_signs_and_normalized_rows(tiny_protocol):
    network = sample_network(tiny_protocol, "continuous", "development", 3)
    nonzero = network.W[network.W != 0]
    assert abs(np.sum(nonzero > 0) - np.sum(nonzero < 0)) <= 1
    assert np.allclose(np.abs(network.W).sum(axis=1), 1.0)
    assert np.all(np.diag(network.W) == 0)
    assert np.array_equal(network.cue_a, -network.cue_b)
    assert np.count_nonzero(network.cue_a) == 2


def test_network_coordinate_changes_rulebook(tiny_protocol):
    first = sample_network(tiny_protocol, "continuous", "development", 0)
    second = sample_network(tiny_protocol, "continuous", "confirmation", 0)
    assert first.uid != second.uid
    assert not np.array_equal(first.W, second.W)

