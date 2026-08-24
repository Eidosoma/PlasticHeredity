import numpy as np

from plastic_heredity.config import GardConfig
from plastic_heredity.features import (
    HISTORY_FEATURE_NAMES,
    STATE_GRAPH_FEATURE_NAMES,
    history_features,
    state_graph_features,
)
from plastic_heredity.simulator import Snapshot


def test_state_graph_features_are_195_and_relabel_invariant():
    rng = np.random.default_rng(12)
    config = GardConfig()
    beta = np.exp(-4.0 + 4.0 * rng.standard_normal((100, 100)))
    composition = rng.multinomial(40, np.full(100, 0.01)).astype(np.int64)
    permutation = rng.permutation(100)

    original = state_graph_features(composition, beta, config)
    relabeled = state_graph_features(
        composition[permutation], beta[np.ix_(permutation, permutation)], config
    )

    assert len(STATE_GRAPH_FEATURE_NAMES) == 195
    assert original.shape == (195,)
    np.testing.assert_allclose(original, relabeled, rtol=1e-12, atol=1e-12)


def test_history_features_use_prefix_only():
    config = GardConfig()
    snapshot = Snapshot(
        composition=np.r_[np.ones(40, dtype=np.int64), np.zeros(60, dtype=np.int64)],
        generation=5,
        inheritance=(True, True, False, True, True),
        boundary_h=(0.95, 0.96, 0.7, 0.93, 0.94),
    )
    values = history_features(snapshot, config)
    assert values.shape == (len(HISTORY_FEATURE_NAMES),)
    assert values[2] == 0.8
    assert values[3] == 0.8
    assert values[4] == 0.02
    assert values[6] == 0.02
    assert values[7] == 1.0
    assert values[8] == 0.02

