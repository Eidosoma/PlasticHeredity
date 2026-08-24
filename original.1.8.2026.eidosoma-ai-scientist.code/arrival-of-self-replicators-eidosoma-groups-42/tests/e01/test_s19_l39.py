from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from e01_onset_discovery.sustained_inheritance import (
    exact_order_null_probability,
    maximum_true_run,
    score_sustained_inheritance,
)


def score(values: list[bool]):
    count = len(values)
    return score_sustained_inheritance(
        inherited=np.asarray(values, dtype=np.bool_),
        generations=np.arange(10, 10 + count, dtype=np.int64),
        offsets_one_based=np.arange(2, 2 + 2 * count, 2, dtype=np.int64),
        required_run=3,
    )


def test_online_certification_and_retrospective_onset_are_distinct() -> None:
    result = score([False, True, True, True, False])
    assert result.event
    assert result.certification_boundary_one_based == 4
    assert result.retrospective_onset_boundary_one_based == 2
    assert result.certification_generation == 13
    assert result.retrospective_onset_generation == 11
    assert result.certification_offset_one_based == 8
    assert result.retrospective_onset_offset_one_based == 4


def test_broken_streak_does_not_certify() -> None:
    result = score([True, True, False, True, True])
    assert not result.event
    assert result.maximum_consecutive_inherited == 2
    assert result.fission_opportunity


def test_minimal_three_fission_sequence_certifies() -> None:
    result = score([True, True, True])
    assert result.event
    assert result.certification_boundary_one_based == 3
    assert result.retrospective_onset_boundary_one_based == 1


def test_no_fission_opportunity_is_explicit() -> None:
    result = score([True, True])
    assert not result.event
    assert not result.fission_opportunity
    assert result.exact_order_null_event_probability == 0.0


def test_maximum_true_run() -> None:
    assert maximum_true_run(np.array([True, True, False, True])) == 2
    assert maximum_true_run(np.array([], dtype=np.bool_)) == 0


def test_exact_order_null_matches_exhaustive_enumeration() -> None:
    length = 7
    successes = 4
    total = 0
    events = 0
    for positions in combinations(range(length), successes):
        values = np.zeros(length, dtype=np.bool_)
        values[list(positions)] = True
        total += 1
        events += maximum_true_run(values) >= 3
    assert exact_order_null_probability(length, successes, 3) == pytest.approx(
        events / total
    )


def test_all_successes_have_probability_one() -> None:
    assert exact_order_null_probability(4, 4, 3) == 1.0


def test_exact_replay() -> None:
    left = score([False, True, True, True, False, True])
    right = score([False, True, True, True, False, True])
    assert left == right


def test_invalid_clock_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        score_sustained_inheritance(
            inherited=np.array([True, True, True]),
            generations=np.array([1, 1, 2]),
            offsets_one_based=np.array([1, 2, 3]),
        )
