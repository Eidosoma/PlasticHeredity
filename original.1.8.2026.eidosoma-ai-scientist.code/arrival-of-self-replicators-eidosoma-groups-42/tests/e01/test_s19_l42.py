from __future__ import annotations

import numpy as np

from e01_onset_discovery.heredity_recovery import score_heredity_recovery

A = np.asarray([10, 0], dtype=np.int64)
B = np.asarray([0, 10], dtype=np.int64)
A_NEAR = np.asarray([9, 1], dtype=np.int64)


def score(states, inheritance, override=None):
    return score_heredity_recovery(
        latest_prefix_daughter=A,
        future_daughters=np.asarray(states, dtype=np.int64).reshape((-1, 2)),
        parent_daughter_h=np.asarray(inheritance, dtype=np.float64),
        future_generations=np.arange(2, 2 + len(states)),
        future_offsets_one_based=np.arange(2, 2 + 2 * len(states), 2),
        recovery_anchor_override=override,
    )


def test_break_then_sustained_same_neighbourhood_recovery() -> None:
    result = score([B, A_NEAR, A], [0.8, 0.95, 0.96])
    assert result.break_observed
    assert result.break_boundary_one_based == 1
    assert result.event
    assert result.certification_boundary_one_based == 3
    assert result.inheritance_resumption_event


def test_uninterrupted_inheritance_is_not_recovery() -> None:
    result = score([A_NEAR, A, A_NEAR], [0.95, 0.96, 0.97])
    assert not result.break_observed
    assert not result.event


def test_inheritance_break_without_compositional_departure_is_not_genuine_break() -> None:
    result = score([A_NEAR, A, A_NEAR], [0.8, 0.96, 0.97])
    assert not result.break_observed


def test_resumption_in_different_neighbourhood_is_not_homeostatic_return() -> None:
    result = score([B, B, B], [0.8, 0.96, 0.97])
    assert result.break_observed
    assert result.inheritance_resumption_event
    assert not result.event


def test_anchor_control_changes_only_recovery_membership() -> None:
    primary = score([B, A_NEAR, A], [0.8, 0.95, 0.96])
    control = score([B, A_NEAR, A], [0.8, 0.95, 0.96], override=B)
    assert primary.break_boundary_one_based == control.break_boundary_one_based
    assert primary.inheritance_resumption_event == control.inheritance_resumption_event
    assert primary.event and not control.event


def test_order_null_preserves_qualifying_count() -> None:
    result = score([B, A_NEAR, B, A], [0.8, 0.95, 0.95, 0.96])
    assert result.qualifying_recovery_count == 2
    assert not result.event
    assert result.exact_recovery_order_null_probability > 0


def test_exact_replay() -> None:
    first = score([B, A_NEAR, A], [0.8, 0.95, 0.96])
    second = score([B.copy(), A_NEAR.copy(), A.copy()], [0.8, 0.95, 0.96])
    assert first == second
