from __future__ import annotations

import numpy as np

from grn_f12_realistic.endpoint import calibrated_threshold, classify_f12, phenotype_similarity


def test_similarity_identity_opposition_and_zero_variance_rules():
    state = np.array([[0.1, 0.3, 0.8, 0.9]])
    assert np.allclose(phenotype_similarity(state, state, "continuous"), 1.0)
    assert phenotype_similarity(state, state[:, ::-1], "continuous")[0] < 0.1
    constant = np.ones((1, 4))
    other_constant = np.full((1, 4), 2.0)
    assert phenotype_similarity(constant, constant, "molecular")[0] == 1.0
    assert phenotype_similarity(constant, other_constant, "molecular")[0] == 0.0


def test_f12_requires_break_then_later_three_high_boundaries():
    similarities = np.array([
        [0.4, 0.8, 0.9, 0.95],
        [0.8, 0.9, 0.4, 0.9],
        [0.9, 0.9, 0.9, 0.9],
        [0.4, 0.9, 0.4, 0.9],
    ])
    broken, event, maximum = classify_f12(similarities, threshold=0.5, run=3)
    assert broken.tolist() == [True, True, False, True]
    assert event.tolist() == [True, False, False, False]
    assert maximum.tolist() == [3, 1, 0, 1]


def test_threshold_is_median_of_within_network_quantiles():
    values = [np.arange(100), np.arange(100) + 100, np.arange(100) + 200]
    threshold, percentiles = calibrated_threshold(values, 0.05)
    assert np.isclose(threshold, percentiles[1])
    assert np.all(np.diff(percentiles) > 0)

