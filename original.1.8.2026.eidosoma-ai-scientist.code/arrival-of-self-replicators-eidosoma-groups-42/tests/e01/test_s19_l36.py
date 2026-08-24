from __future__ import annotations

import numpy as np
import pytest

from e01_onset_discovery.basin_transfer import (
    centroid_similarity,
    close_compositions,
    cosine_scores,
    numerical_equivalence,
    summarize_scores,
)


def test_cosine_multitarget_and_summary() -> None:
    states = np.array([[2, 0, 0], [1, 1, 0], [0, 2, 0]], dtype=np.int64)
    targets = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    scores = cosine_scores(states, targets)
    assert scores.shape == (3, 2)
    assert np.allclose(scores[:, 0], [1, 2**-0.5, 0])
    summary = summarize_scores(scores[:, 0], threshold=0.9)
    assert summary.entered
    assert summary.first_entry_offset_one_based == 1
    assert summary.maximum_score == 1
    assert summary.final_score == 0


def test_closure_scale_and_centroid_invariance() -> None:
    states = np.array([[2, 3], [4, 6]], dtype=np.int64)
    assert np.allclose(close_compositions(states), [[0.4, 0.6], [0.4, 0.6]])
    assert centroid_similarity(np.array([2.0, 3.0]), np.array([4.0, 6.0])) == pytest.approx(1)


def test_fail_closed_inputs_and_empty_summary() -> None:
    with pytest.raises(ValueError):
        close_compositions(np.zeros((1, 3), dtype=np.int64))
    with pytest.raises(ValueError):
        cosine_scores(np.ones((2, 3), dtype=np.int64), np.ones(2))
    summary = summarize_scores(np.array([np.nan, np.nan]))
    assert not summary.entered
    assert summary.first_entry_offset_one_based is None


def test_numerical_equivalence_contract() -> None:
    source = np.array([0.4, 0.8, np.nan], dtype=np.float64)
    replay = source.copy()
    replay[:2] = np.nextafter(replay[:2], np.inf)
    result = numerical_equivalence(
        source,
        replay,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
        maximum_ulp_error=16,
    )
    assert result.passed
    assert result.max_ulp_error == 1
    failed = numerical_equivalence(
        np.array([0.4]),
        np.array([0.4 + 1e-8]),
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
        maximum_ulp_error=16,
    )
    assert not failed.passed


def test_numerical_equivalence_nonfinite_masks_fail_closed() -> None:
    result = numerical_equivalence(
        np.array([np.nan, np.inf]),
        np.array([np.nan, -np.inf]),
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
        maximum_ulp_error=16,
    )
    assert not result.passed
