from __future__ import annotations

import numpy as np

from e01_onset_discovery.process_shooting_efficiency import (
    bernoulli_scores,
    jeffreys_estimate,
    next_uncertainty_allocation,
)


def test_jeffreys_estimate_boundaries_and_replay() -> None:
    empty = jeffreys_estimate(0, 0)
    assert empty.posterior_mean == 0.5
    assert 0 < empty.lower95 < empty.upper95 < 1
    first = jeffreys_estimate(3, 4)
    second = jeffreys_estimate(3, 4)
    assert first == second
    assert first.posterior_mean == 0.7


def test_bernoulli_score_identity() -> None:
    outcomes = np.asarray([0, 1, 1, 0], dtype=np.int8)
    score = bernoulli_scores(0.5, outcomes)
    assert score["brier"] == 0.25
    assert np.isclose(score["logLoss"], np.log(2.0))


def test_uncertainty_allocation_and_lexical_tie() -> None:
    state_ids = ["b", "a", "c"]
    selected = next_uncertainty_allocation(
        state_ids,
        np.asarray([2, 2, 4]),
        np.asarray([4, 4, 4]),
        np.asarray([4, 4, 4]),
    )
    assert selected == 1
    capped = next_uncertainty_allocation(
        state_ids,
        np.asarray([2, 2, 4]),
        np.asarray([4, 4, 4]),
        np.asarray([64, 4, 64]),
    )
    assert capped == 1


def test_invalid_binomial_counts_fail() -> None:
    for values in ((2, 1), (-1, 2), (0, -1)):
        try:
            jeffreys_estimate(*values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid binomial counts did not fail")
