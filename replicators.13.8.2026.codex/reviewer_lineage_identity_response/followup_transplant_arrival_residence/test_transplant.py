from __future__ import annotations

import numpy as np

from reviewer_lineage_identity_response.followup_transplant_arrival_residence.transplant_core import (
    bootstrap_mean_ci,
    bray_curtis_similarity,
    choose_rule_permutation,
    cosine,
    first_capture_class,
    first_target_capture,
    first_target_capture_metric,
    generate_mass_preserving_perturbations,
    inverse_permute_state,
    permute_beta,
    permute_state,
    score_future,
    wilson_interval,
)


def test_permutation_and_inverse_preserve_state_and_beta_algebra() -> None:
    state = np.asarray([3, 0, 2, 1], dtype=np.uint8)
    beta = np.arange(16, dtype=float).reshape(4, 4)
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int16)
    moved = permute_state(state, permutation)
    assert np.array_equal(inverse_permute_state(moved, permutation), state)
    assert np.array_equal(permute_beta(beta, permutation), beta[np.ix_(permutation, permutation)])
    assert int(moved.sum()) == int(state.sum())
    assert sorted(moved.tolist()) == sorted(state.tolist())


def test_rule_permutation_selection_is_deterministic_and_displacing() -> None:
    donors = np.zeros((3, 10), dtype=np.uint8)
    donors[0, 0] = 40
    donors[1, 1] = 40
    donors[2, 2] = 40
    left, score_left, values_left = choose_rule_permutation(donors, seed=91, proposals=128)
    right, score_right, values_right = choose_rule_permutation(donors, seed=91, proposals=128)
    assert np.array_equal(left, right)
    assert score_left == score_right == 0.0
    assert np.array_equal(values_left, values_right)


def test_capture_requires_target_and_all_pairwise_strictly_above_threshold() -> None:
    target = np.asarray([10, 0, 0], dtype=np.uint8)
    coherent = np.repeat(target[None, :], 8, axis=0)
    assert first_target_capture(coherent, target, observed=8, horizon=8) == 1
    incoherent = coherent.copy()
    incoherent[-1] = np.asarray([0, 10, 0])
    assert first_target_capture(incoherent, target, observed=8, horizon=8) == -1
    boundary = np.ones(8)
    score = score_future(coherent, boundary, target, observed=8)
    assert score.capture_f16 and score.capture_f32 and score.coherent8
    assert score.first_capture == 1


def test_equality_at_threshold_does_not_arrive_or_capture() -> None:
    target = np.asarray([1.0, 0.0])
    # Construct a vector with cosine exactly 0.9 to the target.
    value = np.asarray([0.9, np.sqrt(1.0 - 0.9**2)])
    daughters = np.repeat(value[None, :], 8, axis=0)
    assert abs(cosine(target, value) - 0.9) < 1e-12
    assert first_target_capture(daughters, target, observed=8, horizon=8) == -1


def test_departure_and_reentry_are_ordered() -> None:
    target = np.asarray([10, 0], dtype=np.uint8)
    daughters = np.zeros((17, 2), dtype=np.uint8)
    daughters[0] = np.asarray([0, 10])
    daughters[1:9] = target
    daughters[9:] = target
    score = score_future(daughters, np.ones(17), target, observed=17)
    assert score.departed
    assert score.reentered
    assert score.first_arrival == 2


def test_incomplete_future_cannot_be_complete_or_capture() -> None:
    target = np.asarray([10, 0], dtype=np.uint8)
    daughters = np.repeat(target[None, :], 32, axis=0)
    score = score_future(daughters, np.ones(32), target, observed=7)
    assert not score.completed
    assert not score.capture_f16
    assert not score.capture_f32


def test_bray_curtis_metric_capture() -> None:
    target = np.asarray([5, 5, 0], dtype=np.uint8)
    daughters = np.repeat(target[None, :], 8, axis=0)
    assert bray_curtis_similarity(target, target) == 1.0
    assert first_target_capture_metric(
        daughters,
        target,
        observed=8,
        horizon=8,
        threshold=0.9,
        metric="bray_curtis",
    ) == 1


def test_first_capture_class_treats_ties_and_no_capture_as_unknown() -> None:
    first = np.asarray([10, 0], dtype=np.uint8)
    second = np.asarray([0, 10], dtype=np.uint8)
    daughters = np.repeat(first[None, :], 8, axis=0)
    assert first_capture_class(daughters, [first, second], observed=8) == 0
    assert first_capture_class(daughters, [first, first], observed=8) == -1
    assert first_capture_class(daughters, [second, second], observed=8) == -1


def test_mass_preserving_perturbations_respect_geometry() -> None:
    form = np.zeros(100, dtype=np.uint8)
    form[:40] = 1
    other = np.zeros(100, dtype=np.uint8)
    other[40:80] = 1
    dose, starts, _ = generate_mass_preserving_perturbations(
        form,
        other,
        seed=17,
        required=3,
        proposals_per_dose=4_096,
        dose_ladder=(4, 8),
    )
    assert dose in {4, 8}
    assert len(starts) == 3
    assert len({item.tobytes() for item in starts}) == 3
    for start in starts:
        assert int(start.sum()) == int(form.sum())
        assert 0.85 <= cosine(form, start) <= 0.95
        assert cosine(other, start) <= 0.85


def test_bootstrap_and_wilson_are_reproducible_and_bounded() -> None:
    values = np.asarray([0.0, 0.5, 1.0])
    first = bootstrap_mean_ci(values, seed=5, repetitions=1_000)
    second = bootstrap_mean_ci(values, seed=5, repetitions=1_000)
    assert first == second
    estimate, lower, upper = wilson_interval(7, 10)
    assert 0.0 <= lower <= estimate <= upper <= 1.0

