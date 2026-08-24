from __future__ import annotations

import numpy as np
import pytest

from e01_onset_discovery.recurrence_inheritance import (
    cosine_h,
    score_recurrence_inheritance,
)


def arrays(states, generations, inherited):
    return (
        np.asarray(states, dtype=np.int64),
        np.asarray(generations, dtype=np.int64),
        np.asarray(inherited, dtype=np.bool_),
    )


def test_cosine_h_identical_and_orthogonal() -> None:
    assert cosine_h(np.array([2, 1]), np.array([4, 2])) == pytest.approx(1.0)
    assert cosine_h(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.0)


def test_recurrence_requires_intervening_generation() -> None:
    prefix = arrays([[10, 1], [1, 10]], [1, 2], [True, True])
    future = arrays([[10, 1]], [3], [True])
    result = score_recurrence_inheritance(
        prefix_states=prefix[0],
        prefix_generations=prefix[1],
        prefix_inherited=prefix[2],
        future_states=future[0],
        future_generations=future[1],
        future_inherited=future[2],
    )
    assert result.event
    assert result.first_event_boundary_one_based == 1
    assert result.matched_reference_generation == 1


def test_adjacent_generation_cannot_establish_recurrence() -> None:
    prefix = arrays([[10, 1]], [2], [True])
    future = arrays([[10, 1]], [3], [True])
    result = score_recurrence_inheritance(
        prefix_states=prefix[0],
        prefix_generations=prefix[1],
        prefix_inherited=prefix[2],
        future_states=future[0],
        future_generations=future[1],
        future_inherited=future[2],
    )
    assert not result.event
    assert result.eligible_comparison_count == 0


@pytest.mark.parametrize(
    ("prefix_inherited", "future_inherited"), [(False, True), (True, False)]
)
def test_both_boundaries_must_be_inherited(
    prefix_inherited: bool, future_inherited: bool
) -> None:
    prefix = arrays([[10, 1]], [1], [prefix_inherited])
    future = arrays([[10, 1]], [3], [future_inherited])
    result = score_recurrence_inheritance(
        prefix_states=prefix[0],
        prefix_generations=prefix[1],
        prefix_inherited=prefix[2],
        future_states=future[0],
        future_generations=future[1],
        future_inherited=future[2],
    )
    assert not result.event


def test_earlier_future_boundary_becomes_past_only_reference() -> None:
    prefix = arrays([[1, 10]], [1], [True])
    future = arrays([[10, 1], [1, 10], [10, 1]], [2, 3, 4], [True, True, True])
    result = score_recurrence_inheritance(
        prefix_states=prefix[0],
        prefix_generations=prefix[1],
        prefix_inherited=prefix[2],
        future_states=future[0],
        future_generations=future[1],
        future_inherited=future[2],
    )
    assert result.event
    assert result.first_event_boundary_one_based == 2
    assert result.first_event_generation == 3


def test_invalid_generation_order_is_rejected() -> None:
    prefix = arrays([[1, 2]], [2], [True])
    future = arrays([[1, 2]], [2], [True])
    with pytest.raises(ValueError, match="follow"):
        score_recurrence_inheritance(
            prefix_states=prefix[0],
            prefix_generations=prefix[1],
            prefix_inherited=prefix[2],
            future_states=future[0],
            future_generations=future[1],
            future_inherited=future[2],
        )
