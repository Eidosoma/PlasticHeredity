from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from reviewer_sequence_history_response.implementation_adapters import source_contract
from reviewer_sequence_history_response.sequence_core import (
    LaggedRidgeModel,
    TransitionModel,
    _binomial_training_rows,
    event_from_flags,
    fit_lagged_ridge,
    fit_transition_model,
    holm_adjust,
    lagged_history_matrix,
    state_branch_log_loss,
    terminal_duration,
    transition_event_probability,
)


def test_strict_event_semantics() -> None:
    assert not event_from_flags([True] * 12)
    assert not event_from_flags([False, True, True, False, True, True])
    assert event_from_flags([True, False, True, True, True])
    assert event_from_flags([False, False, True, True, True])
    assert not event_from_flags([True, True, True, False])


def test_terminal_duration_is_past_only_and_capped() -> None:
    assert terminal_duration([True, False, False]) == 2
    assert terminal_duration([False, True, True, True]) == 3
    assert terminal_duration([True] * 20) == 5


def _exhaustive_probability(
    model: TransitionModel, prefix: list[bool], horizon: int
) -> float:
    total = 0.0

    def recurse(flags: list[bool], remaining: int, probability: float) -> None:
        nonlocal total
        if remaining == 0:
            if event_from_flags(flags[len(prefix) :]):
                total += probability
            return
        row = model.row(flags[-1], terminal_duration(flags))
        for outcome in (0, 1, 2):
            mass = probability * float(row[outcome])
            if outcome == 2:
                if event_from_flags(flags[len(prefix) :]):
                    total += mass
                continue
            recurse(flags + [bool(outcome)], remaining - 1, mass)

    recurse(prefix, horizon, 1.0)
    return total


def test_dynamic_program_matches_exhaustive_enumeration() -> None:
    probability = np.asarray([[0.35, 0.60, 0.05], [0.20, 0.75, 0.05]])
    model = TransitionModel(False, probability, np.ones((2, 3), dtype=np.int64))
    prefix = [True, True]
    expected = _exhaustive_probability(model, prefix, 5)
    observed = transition_event_probability(
        model, np.asarray(prefix, dtype=float), threshold=0.5, horizon=5
    )
    assert abs(expected - observed) < 1e-14


def test_terminal_is_absorbing_failure_before_certification() -> None:
    # Every next outcome terminates; success is impossible.
    probability = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    model = TransitionModel(False, probability, np.zeros((2, 3), dtype=np.int64))
    assert transition_event_probability(model, [0.95], horizon=12) == 0.0


def test_transition_fit_uses_common_outcome_support() -> None:
    histories = np.asarray(
        [[0.95, 0.80, 0.96, 0.97], [0.70, 0.72, 0.95, 0.96]], dtype=float
    )
    lengths = np.asarray([4, 4])
    died = np.asarray([False, True])
    markov = fit_transition_model(histories, lengths, died, duration_aware=False)
    semi = fit_transition_model(histories, lengths, died, duration_aware=True)
    assert int(markov.counts.sum()) == 7  # six transitions plus one terminal
    assert int(semi.counts.sum()) == 7
    assert np.allclose(markov.probabilities.sum(axis=-1), 1.0)
    assert np.allclose(semi.probabilities.sum(axis=-1), 1.0)


def test_lagged_features_are_right_aligned_and_masked() -> None:
    direct = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    histories = np.asarray([[0.8, 0.91, 0.0], [0.7, 0.8, 0.95]])
    lengths = np.asarray([2, 3])
    matrix = lagged_history_matrix(direct, histories, lengths, 3)
    # direct(2), H(3), flags(3), masks(3)
    assert matrix.shape == (2, 11)
    np.testing.assert_allclose(matrix[0, 2:5], [0.0, 0.8, 0.91])
    np.testing.assert_allclose(matrix[0, 5:8], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(matrix[0, 8:11], [0.0, 1.0, 1.0])


def test_weighted_binomial_rows_preserve_branch_log_likelihood() -> None:
    x = np.asarray([[0.0], [1.0]])
    y = np.asarray([[1, 1, 0, 1], [0, 0, 1, 0]], dtype=np.int8)
    rows, labels, weights = _binomial_training_rows(x, y)
    assert weights.sum() == y.size
    p = np.asarray([0.7, 0.2])
    explicit = state_branch_log_loss(y, p)
    compressed = -np.sum(
        weights
        * (
            labels * np.log(np.repeat(p, 2)[weights > 0])
            + (1 - labels) * np.log1p(-np.repeat(p, 2)[weights > 0])
        )
    ) / weights.sum()
    assert abs(explicit - compressed) < 1e-15


def test_grouped_lagged_selection_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(10), 4)
    lengths = np.full(groups.size, 6)
    histories = rng.uniform(0.75, 0.99, size=(groups.size, 6))
    direct = rng.normal(size=(groups.size, 3))
    signal = (histories[:, -2:].mean(axis=1) + 0.15 * direct[:, 0]) > 0.9
    targets = signal.astype(np.int8)[:, None]
    first, audit_first = fit_lagged_ridge(
        direct,
        histories,
        lengths,
        targets,
        groups,
        direct_columns=(0, 1, 2),
        lag_grid=(2, 4),
        c_grid=(0.1, 1.0),
        folds=5,
    )
    second, audit_second = fit_lagged_ridge(
        direct,
        histories,
        lengths,
        targets,
        groups,
        direct_columns=(0, 1, 2),
        lag_grid=(2, 4),
        c_grid=(0.1, 1.0),
        folds=5,
    )
    assert first.lag == second.lag
    assert first.c_value == second.c_value
    np.testing.assert_array_equal(first.coefficient, second.coefficient)
    assert audit_first == audit_second


def test_holm_adjustment_known_example() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_source_contract_is_complete_and_read_only() -> None:
    contract = source_contract()
    assert "manuscript" not in contract
    assert "fable_v2_cohort" in contract
    assert all(len(value["sha256"]) == 64 for value in contract.values())


def test_adapter_contains_no_confirmation_future_simulator_call() -> None:
    source = (
        Path(__file__).with_name("implementation_adapters.py").read_text(encoding="utf-8")
    )
    assert "simulate_future" not in source
    assert "conf_unit(" not in source
