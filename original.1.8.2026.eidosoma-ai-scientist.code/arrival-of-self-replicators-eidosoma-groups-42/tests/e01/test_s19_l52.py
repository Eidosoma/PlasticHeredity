from __future__ import annotations

import numpy as np
import pytest

from e01_onset_discovery.shooting_regime_compression import (
    fit_shrunk_duration_table,
    hazard_fit_scope,
    transition_scores,
)


def test_matrix_transfer_scope_excludes_target() -> None:
    states = ("a", "b", "c", "d", "e")
    assert hazard_fit_scope("c", states, "MATRIX_OTHER_LANDMARK_SEMIMARKOV") == (
        "a",
        "b",
        "d",
        "e",
    )
    assert hazard_fit_scope("c", states, "STATE_LOCAL_SEMIMARKOV") == ("c",)


def test_invalid_scope_fails() -> None:
    with pytest.raises(ValueError):
        hazard_fit_scope("z", ("a", "b"), "STATE_LOCAL_SEMIMARKOV")
    with pytest.raises(ValueError):
        hazard_fit_scope("a", ("a", "a"), "STATE_LOCAL_SEMIMARKOV")
    with pytest.raises(ValueError):
        hazard_fit_scope("a", ("a",), "MATRIX_OTHER_LANDMARK_SEMIMARKOV")


def test_shrunk_table_uses_cell_anchor_for_empty_cells() -> None:
    anchor = np.asarray([[0.2, 0.4], [0.7, 0.9]], dtype=np.float64)
    table = fit_shrunk_duration_table(
        [False, True, True],
        [1, 1, 4],
        [True, False, True],
        anchor,
        prior_strength=4,
    )
    assert table[0, 0] == pytest.approx((1 + 4 * 0.2) / 5)
    assert table[0, 1] == pytest.approx(0.4)
    assert table[1, 0] == pytest.approx((0 + 4 * 0.7) / 5)
    assert table[1, 1] == pytest.approx((1 + 4 * 0.9) / 5)


def test_transition_scores_match_manual_values() -> None:
    table = np.asarray([[0.25, 0.5], [0.75, 0.9]])
    losses, briers = transition_scores(
        [False, True], [1, 3], [False, True], table
    )
    assert losses.tolist() == pytest.approx([-np.log(0.75), -np.log(0.9)])
    assert briers.tolist() == pytest.approx([0.25**2, 0.1**2])


def test_exact_fit_replay() -> None:
    args = (
        [False, False, True, True],
        [1, 2, 1, 2],
        [False, True, True, True],
        np.asarray([[0.3, 0.4], [0.7, 0.8]]),
    )
    first = fit_shrunk_duration_table(*args, prior_strength=4)
    second = fit_shrunk_duration_table(*args, prior_strength=4)
    assert np.array_equal(first, second)
