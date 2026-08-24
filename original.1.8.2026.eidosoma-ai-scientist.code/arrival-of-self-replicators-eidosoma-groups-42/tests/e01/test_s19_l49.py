import numpy as np
import pytest

from e01_onset_discovery.longitudinal_process_risk import (
    jeffreys_mean,
    score_new_hereditary_episode,
    trailing_true_run,
)


def test_break_then_run_three_certifies_online() -> None:
    result = score_new_hereditary_episode([0.95, 0.2, 0.91, 0.92, 0.93, 0.1])
    assert result.break_observed
    assert result.break_boundary_one_based == 2
    assert result.event
    assert result.certification_boundary_one_based == 5
    assert result.maximum_postbreak_run == 3


def test_uninterrupted_inheritance_is_not_a_new_episode() -> None:
    result = score_new_hereditary_episode(np.full(12, 0.95))
    assert not result.break_observed
    assert not result.event
    assert result.postbreak_opportunities == 0


def test_strict_threshold_and_temporal_order() -> None:
    strict = score_new_hereditary_episode([0.9, 0.91, 0.92, 0.93])
    shuffled = score_new_hereditary_episode([0.91, 0.92, 0.93, 0.9])
    assert strict.event
    assert not shuffled.event


def test_trailing_run_and_jeffreys_contract() -> None:
    assert trailing_true_run([False, True, True]) == 2
    assert trailing_true_run([True, False]) == 0
    assert jeffreys_mean(3, 4) == 0.7
    assert jeffreys_mean(0, 0) == 0.5
    with pytest.raises(ValueError):
        jeffreys_mean(2, 1)
