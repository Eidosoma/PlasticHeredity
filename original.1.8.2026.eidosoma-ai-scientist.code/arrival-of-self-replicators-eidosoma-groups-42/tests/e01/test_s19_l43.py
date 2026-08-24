from __future__ import annotations

import numpy as np

from e01_onset_discovery.heredity_recovery_gain import score_heredity_recovery_gain


def score(future: list[list[int]], inheritance: list[float], anchor=None):
    return score_heredity_recovery_gain(
        latest_prefix_daughter=np.asarray([10, 0]),
        future_daughters=np.asarray(future),
        parent_daughter_h=np.asarray(inheritance),
        future_generations=np.arange(1, len(future) + 1),
        future_offsets_one_based=np.arange(1, len(future) + 1),
        recovery_anchor_override=None if anchor is None else np.asarray(anchor),
    )


def test_break_then_continuous_gain_at_resumption() -> None:
    result = score([[0, 10], [5, 5], [8, 2]], [0.4, 0.95, 0.96])
    assert result.break_observed
    assert result.resumption_observed
    assert result.resumption_certification_boundary_one_based == 3
    assert result.recovery_gain is not None and result.recovery_gain > 0.0


def test_uninterrupted_inheritance_is_not_recovery() -> None:
    result = score([[9, 1], [8, 2], [9, 1]], [0.95, 0.95, 0.95])
    assert not result.break_observed
    assert result.recovery_gain is None


def test_resumption_elsewhere_can_have_negative_gain() -> None:
    result = score([[0, 10], [1, 9], [2, 8]], [0.2, 0.95, 0.95])
    assert result.resumption_observed
    assert result.recovery_gain is not None and result.recovery_gain > 0.0
    control = score([[0, 10], [1, 9], [2, 8]], [0.2, 0.95, 0.95], anchor=[0, 10])
    assert control.recovery_gain is not None and control.recovery_gain < 0.0


def test_anchor_control_preserves_break_and_certification() -> None:
    primary = score([[0, 10], [5, 5], [8, 2]], [0.2, 0.95, 0.95])
    control = score([[0, 10], [5, 5], [8, 2]], [0.2, 0.95, 0.95], anchor=[0, 10])
    assert primary.break_boundary_one_based == control.break_boundary_one_based
    assert (
        primary.resumption_certification_boundary_one_based
        == control.resumption_certification_boundary_one_based
    )
    assert primary.recovery_gain != control.recovery_gain


def test_no_resumption_has_no_primary_gain() -> None:
    result = score([[0, 10], [5, 5], [8, 2]], [0.2, 0.95, 0.2])
    assert result.break_observed
    assert not result.resumption_observed
    assert result.recovery_gain is None
    assert result.maximum_recovery_gain is not None


def test_strict_threshold() -> None:
    result = score([[0, 10], [5, 5], [8, 2]], [0.2, 0.9, 0.95])
    assert not result.resumption_observed


def test_exact_replay() -> None:
    kwargs = {
        "future": [[0, 10], [5, 5], [8, 2]],
        "inheritance": [0.2, 0.95, 0.95],
    }
    assert score(**kwargs) == score(**kwargs)
