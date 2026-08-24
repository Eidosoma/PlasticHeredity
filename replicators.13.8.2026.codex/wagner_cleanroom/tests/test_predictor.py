from __future__ import annotations

import numpy as np

from wagner_cleanroom.dynamics import sample_rulebook
from wagner_cleanroom.predictor import (
    _history_features,
    _score_future_trajectories,
    _start_states,
    simulate_predictor_rulebook,
)
from wagner_cleanroom.protocol import load_protocol


def fixture_rulebook():
    for proposal in range(100):
        result = sample_rulebook("predictor-test", proposal)
        if result is not None:
            return result
    raise AssertionError("no eligible fixture rulebook")


def test_history_feature_contract() -> None:
    features = _history_features(np.asarray([0, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.uint16), 10)
    assert features.shape == (9,)
    assert np.isfinite(features).all()


def test_start_states_are_deterministic_and_distinct() -> None:
    source = fixture_rulebook()
    first = _start_states(source, 5, 10)
    second = _start_states(source, 5, 10)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 5


def test_predictor_rulebook_shapes() -> None:
    protocol = load_protocol("predictor", "smoke")
    protocol["histories_per_source"] = 3
    protocol["futures_per_state"] = 8
    protocol["horizon"] = 12
    arrays = simulate_predictor_rulebook(fixture_rulebook(), protocol, "development", 0)
    assert arrays["x_history"].shape == (3, 9)
    assert arrays["x_structural"].shape == (3, 6)
    assert arrays["x_full"].shape[0] == 3
    assert np.all(arrays["f12_counts"] <= 8)


def test_similarity_grid_is_scored_independently() -> None:
    trajectory = np.asarray([[3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2]], dtype=np.uint16)
    point = np.ones_like(trajectory, dtype=bool)
    f12, _, sensitivity = _score_future_trajectories(trajectory, point, current=0, genes=10)
    assert int(f12[0]) == 0
    assert int(sensitivity[0, 0]) == 1
    assert int(sensitivity[0, 1]) == 0
