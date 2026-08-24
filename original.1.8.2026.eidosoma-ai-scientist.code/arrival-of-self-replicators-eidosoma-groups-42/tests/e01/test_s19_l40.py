from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from e01_onset_discovery.recurrence_after_departure import (
    exact_departure_return_order_probability,
    score_recurrence_after_departure,
)

ANCHOR = np.asarray([10, 0], dtype=np.int64)
NEAR = np.asarray([9, 1], dtype=np.int64)
FAR = np.asarray([0, 10], dtype=np.int64)


def score(states):
    count = len(states)
    return score_recurrence_after_departure(
        anchor=ANCHOR,
        future_states=np.asarray(states, dtype=np.int64).reshape((-1, 2)),
        generations=np.arange(3, 3 + count, dtype=np.int64),
        offsets_one_based=np.arange(2, 2 + 2 * count, 2, dtype=np.int64),
        threshold=0.9,
    )


def test_near_far_near_certifies_only_at_return() -> None:
    result = score([NEAR, FAR, NEAR])
    assert result.event
    assert result.departure_boundary_one_based == 2
    assert result.certification_boundary_one_based == 3
    assert result.departure_offset_one_based == 4
    assert result.certification_offset_one_based == 6


def test_near_without_prior_departure_is_not_recurrence() -> None:
    result = score([NEAR, NEAR, NEAR])
    assert not result.event
    assert not result.departure_observed
    assert result.return_progress == 0.0


def test_departure_without_return_is_not_certified() -> None:
    result = score([FAR, FAR, FAR])
    assert not result.event
    assert result.departure_observed
    assert result.maximum_postdeparture_h == pytest.approx(0.0)


def test_far_near_is_minimal_event() -> None:
    result = score([FAR, NEAR])
    assert result.event
    assert result.mixed_membership_opportunity


def test_exact_order_null_matches_exhaustive_enumeration() -> None:
    length = 5
    near_count = 2
    event_count = 0
    total = 0
    for positions in combinations(range(length), near_count):
        near = np.zeros(length, dtype=np.bool_)
        near[list(positions)] = True
        total += 1
        event_count += any((not near[left]) and near[right] for left in range(length) for right in range(left + 1, length))
    assert exact_departure_return_order_probability(length, near_count) == pytest.approx(
        event_count / total
    )


def test_exact_replay() -> None:
    assert score([NEAR, FAR, NEAR]) == score([NEAR, FAR, NEAR])


def test_invalid_clock_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        score_recurrence_after_departure(
            anchor=ANCHOR,
            future_states=np.asarray([FAR, NEAR]),
            generations=np.asarray([2, 2]),
            offsets_one_based=np.asarray([1, 2]),
        )
