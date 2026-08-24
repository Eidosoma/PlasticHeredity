from __future__ import annotations

import numpy as np

from plastic_heredity.intervention_core import MolecularEdit
from plastic_heredity.intervention_cr8_release_challenge import (
    BOOTSTRAP_REPETITIONS,
    CHALLENGE_ARMS,
    CHALLENGE_BRANCHES,
    CHALLENGE_EQUIVALENCE_MARGIN,
    CHALLENGE_HORIZON,
    MATRICES,
    ORIGINS,
    RANDOMIZATION_REPETITIONS,
    RANDOM_DOSES,
    RELEASE_EQUIVALENCE_MARGIN,
    RELEASE_HORIZON,
    REPLICATES,
    SEEDS,
    _challenge_edit_seed,
    _challenge_future_seed,
    _holm_adjust,
    _release_future_seed,
    apply_edits,
    classify_challenge,
    edited_snapshot_many,
    inference_draws,
    protocol,
    random_k_edits,
)
from plastic_heredity.simulator import Snapshot


def test_cr8_design_is_frozen_exactly() -> None:
    frozen = protocol()
    assert MATRICES == 48
    assert REPLICATES == 6
    assert ORIGINS == ("MODEL_DOWN", "RULE_DOWN", "NOOP")
    assert RELEASE_HORIZON == 60
    assert CHALLENGE_BRANCHES == 32
    assert CHALLENGE_HORIZON == 24
    assert RANDOM_DOSES == (0, 2, 4, 8, 16)
    assert CHALLENGE_ARMS == (
        "NONE",
        "RANDOM_K2",
        "RANDOM_K4",
        "RANDOM_K8",
        "RANDOM_K16",
        "ADVERSARIAL",
    )
    assert RELEASE_EQUIVALENCE_MARGIN == 0.03
    assert CHALLENGE_EQUIVALENCE_MARGIN == 0.05
    assert frozen["release"]["interventions_after_release"] == 0
    assert frozen["upstream"]["cr7_active_extension_excluded"] is True


def test_cr8_future_streams_are_paired_and_selection_is_separate() -> None:
    assert len(SEEDS) == len(set(SEEDS.values()))
    assert len({_release_future_seed("02", 4, 3) for _origin in ORIGINS}) == 1
    assert (
        len(
            {
                _challenge_future_seed("03", 7, 2, 11)
                for _origin in ORIGINS
                for _arm in CHALLENGE_ARMS
            }
        )
        == 1
    )
    assert _challenge_edit_seed("03", 7, 2, "NOOP", "RANDOM_K8") != (
        _challenge_future_seed("03", 7, 2, 11)
    )


def test_random_k_has_exact_transport_mass_and_nonnegativity() -> None:
    composition = np.zeros(100, dtype=np.int64)
    composition[:8] = (8, 7, 6, 5, 4, 4, 3, 3)
    for k in RANDOM_DOSES:
        first = random_k_edits(composition, k, np.random.default_rng(100 + k))
        second = random_k_edits(composition, k, np.random.default_rng(100 + k))
        assert first == second
        assert len(first) == k
        edited = apply_edits(composition, first)
        assert int(edited.sum()) == int(composition.sum())
        assert np.all(edited >= 0)
        assert int(np.abs(edited - composition).sum() // 2) == k


def test_multi_edit_keeps_all_history_fields_exact() -> None:
    composition = np.zeros(100, dtype=np.int64)
    composition[:3] = (2, 1, 1)
    snapshot = Snapshot(
        composition=composition,
        generation=120,
        inheritance=(True, False),
        boundary_h=(0.95, 0.8),
        previous_growth_steps=19,
        cumulative_growth_steps=1234,
    )
    edited = edited_snapshot_many(
        snapshot, (MolecularEdit(0, 10), MolecularEdit(1, 11))
    )
    assert edited.generation == snapshot.generation
    assert edited.inheritance == snapshot.inheritance
    assert edited.boundary_h == snapshot.boundary_h
    assert edited.previous_growth_steps == snapshot.previous_growth_steps
    assert edited.cumulative_growth_steps == snapshot.cumulative_growth_steps
    assert int(edited.composition.sum()) == int(snapshot.composition.sum())


def test_classifier_category_precedence_and_strict_thresholds() -> None:
    h = np.asarray([0.95] * CHALLENGE_HORIZON)
    held = classify_challenge(np.asarray([1.0, 0.7, 0.9]), h, 0.5, True)
    assert held["category"] == "held"

    returned = classify_challenge(
        np.asarray([1.0, 0.69, 0.9, 0.91, 0.92, 0.93]), h, 0.5, True
    )
    assert returned["category"] == "returned"
    assert returned["return_certification_time"] == 5

    mode_h = np.asarray([0.8] * 18 + [0.91] * 6)
    mode = classify_challenge(
        np.asarray([1.0, 0.69] + [0.8] * 23), mode_h, 0.45, True
    )
    assert mode["category"] == "mode_recovered"

    lost = classify_challenge(
        np.asarray([1.0, 0.69, 0.8]), h, 0.9, False
    )
    assert lost["category"] == "lost"


def test_return_must_be_strictly_after_departure() -> None:
    h = np.asarray([0.95] * CHALLENGE_HORIZON)
    result = classify_challenge(
        np.asarray([0.69, 0.91, 0.92, 0.93]), h, 0.5, True
    )
    assert result["returned"] is True
    assert result["return_certification_time"] == 3
    broken = classify_challenge(
        np.asarray([0.69, 0.91, 0.9, 0.92, 0.93]), h, 0.5, True
    )
    assert broken["returned"] is False


def test_whole_matrix_draws_are_frozen() -> None:
    draws = inference_draws()
    assert BOOTSTRAP_REPETITIONS == 4096
    assert RANDOMIZATION_REPETITIONS == 4096
    assert draws["bootstrap_indices"].shape == (4096, 48)
    assert draws["randomization_signs"].shape == (4096, 48)
    assert np.all(np.isin(draws["randomization_signs"], (-1.0, 1.0)))


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    raw = [0.04, 0.001, 0.02, 0.5]
    adjusted = _holm_adjust(raw)
    order = np.argsort(raw)
    sorted_adjusted = np.asarray(adjusted)[order]
    assert np.all(np.diff(sorted_adjusted) >= 0)
    assert all(raw[index] <= adjusted[index] <= 1.0 for index in range(len(raw)))
