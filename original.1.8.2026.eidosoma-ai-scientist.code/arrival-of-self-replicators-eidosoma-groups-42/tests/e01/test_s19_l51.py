from __future__ import annotations

import itertools

import numpy as np
import pytest

from e01_onset_discovery.regime_hazard import (
    finite_horizon_process_probability,
    fit_iid,
    fit_markov,
    fit_semimarkov,
    posterior_matrix_markov,
    trailing_run_length,
    transition_rows,
    transport_duration_effect,
)


def test_trailing_run_and_transition_rows() -> None:
    assert trailing_run_length([True, False, False]) == 2
    assert transition_rows(True, 2, [True, False, False, True]) == (
        (True, 2, True),
        (True, 3, False),
        (False, 1, False),
        (False, 2, True),
    )


def test_iid_and_markov_fits_are_smoothed() -> None:
    assert fit_iid([True, True, False]) == pytest.approx(0.625)
    fitted = fit_markov([False, False, True, True], [False, True, True, True])
    assert fitted.tolist() == pytest.approx([0.5, 5.0 / 6.0])


def test_semimarkov_uses_duration_and_overflow_bin() -> None:
    current = [True, True, True, True]
    duration = [1, 2, 20, 21]
    following = [False, True, True, True]
    markov = np.asarray([0.5, 0.7])
    fitted = fit_semimarkov(current, duration, following, markov, maximum_duration=3)
    assert fitted.shape == (2, 3)
    assert fitted[1, 0] == pytest.approx(0.35)
    assert fitted[1, 1] == pytest.approx(0.85)
    assert fitted[1, 2] == pytest.approx(2.7 / 3.0)


def test_matrix_posterior_and_duration_transport() -> None:
    pooled = np.asarray([0.4, 0.8])
    posterior = posterior_matrix_markov([False, True, True, False], pooled)
    assert posterior.tolist() == pytest.approx([0.7, 1.8 / 3.0])
    semi = np.asarray([[0.2, 0.6], [0.5, 0.9]])
    transported = transport_duration_effect(posterior, pooled, semi)
    assert transported.shape == semi.shape
    assert np.all((transported > 0) & (transported < 1))


def test_finite_horizon_matches_exhaustive_enumeration() -> None:
    p = 0.72
    result = finite_horizon_process_probability(
        lambda _state, _duration: p,
        initial_state=True,
        initial_duration=4,
        horizon=6,
        required_run=3,
    )
    break_probability = 0.0
    joint_probability = 0.0
    for sequence in itertools.product((False, True), repeat=6):
        mass = np.prod([p if value else 1 - p for value in sequence])
        broken = False
        run = 0
        success = False
        for value in sequence:
            if not value:
                broken = True
                run = 0
            elif broken:
                run += 1
                success = success or run >= 3
        break_probability += mass * broken
        joint_probability += mass * success
    assert result.break_probability == pytest.approx(break_probability)
    assert result.joint_break_run_probability == pytest.approx(joint_probability)
    assert result.run_probability_given_break == pytest.approx(
        joint_probability / break_probability
    )


def test_markov_is_duration_invariant_but_semimarkov_is_not() -> None:
    markov = np.asarray([0.6, 0.85])
    invariant = finite_horizon_process_probability(
        lambda state, _duration: markov[int(state)],
        initial_state=True,
        initial_duration=1,
        horizon=8,
    )
    invariant_long = finite_horizon_process_probability(
        lambda state, _duration: markov[int(state)],
        initial_state=True,
        initial_duration=20,
        horizon=8,
    )
    assert invariant == invariant_long
    semi = np.tile(markov[:, None], (1, 12))
    semi[1] = np.linspace(0.5, 0.98, 12)
    short = finite_horizon_process_probability(
        lambda state, duration: semi[int(state), min(duration, 12) - 1],
        initial_state=True,
        initial_duration=1,
        horizon=8,
    )
    long = finite_horizon_process_probability(
        lambda state, duration: semi[int(state), min(duration, 12) - 1],
        initial_state=True,
        initial_duration=20,
        horizon=8,
    )
    assert short.joint_break_run_probability != pytest.approx(
        long.joint_break_run_probability
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_duration": 0, "horizon": 4},
        {"initial_duration": 1, "horizon": 0},
    ],
)
def test_invalid_process_contract_fails(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        finite_horizon_process_probability(
            lambda _state, _duration: 0.5,
            initial_state=True,
            **kwargs,
        )
