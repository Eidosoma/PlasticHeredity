from __future__ import annotations

import math

import numpy as np

from e01_latent_timebase.core import ExposureDefinition, SimulationDefinition
from e01_onset_discovery.generator_coordinate import (
    analytic_count_moments,
    brownian_hitting_probability,
    composition_linearized_moments,
    cosine_gradient,
    relative_composition,
    sample_complete_kernel,
    truncated_poisson_moments,
)


def definition(candidate: int) -> SimulationDefinition:
    return SimulationDefinition(
        daughter_rule="FIRST_DAUGHTER" if candidate == 2 else "RANDOM_NONEMPTY",
        overshoot_rule="TRIM_NEW_ENTRANTS_TO_NMAX",
        exposure=ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=0.6),
    )


def streams(seed: int) -> tuple[np.random.Generator, ...]:
    return tuple(np.random.Generator(np.random.PCG64DXSM(seed + i)) for i in range(4))


def test_truncated_poisson_matches_direct_sum() -> None:
    rate = 2.3
    cap = 4
    mean, variance = truncated_poisson_moments(rate, cap)
    rng = np.arange(80)
    probabilities = (
        np.exp(-rate)
        * rate**rng
        / np.asarray([math.factorial(int(value)) for value in rng], dtype=object)
    )
    probabilities = probabilities.astype(float)
    clipped = np.minimum(rng, cap)
    direct_mean = np.sum(probabilities * clipped)
    direct_variance = np.sum(probabilities * clipped**2) - direct_mean**2
    assert np.isclose(mean, direct_mean, atol=1e-12)
    assert np.isclose(variance, direct_variance, atol=1e-12)


def test_first_daughter_fission_moments_are_exact() -> None:
    state = np.zeros(100, dtype=np.int64)
    state[:20] = 4
    beta = np.ones((100, 100), dtype=np.float64)
    moments = analytic_count_moments(
        state, beta, definition(2), generation_local_step=12
    )
    assert moments.transition_kind == "FISSION"
    assert np.allclose(moments.mean_delta, -0.5 * state)
    assert np.allclose(np.diag(moments.covariance_delta), 0.25 * state)
    assert np.allclose(
        moments.covariance_delta - np.diag(np.diag(moments.covariance_delta)), 0
    )


def test_random_nonempty_fission_moments_match_exact_enumeration() -> None:
    state = np.zeros(100, dtype=np.int64)
    state[:2] = 1
    beta = np.ones((100, 100), dtype=np.float64)
    moments = analytic_count_moments(
        state, beta, definition(3), generation_local_step=1000
    )
    assert np.allclose(moments.mean_delta[:2], -0.25)
    assert np.allclose(np.diag(moments.covariance_delta)[:2], 0.1875)
    assert np.isclose(moments.covariance_delta[0, 1], -0.0625)


def test_kernel_exact_replay_and_feature_permutation() -> None:
    state = np.zeros(100, dtype=np.int64)
    state[:40] = 1
    beta = np.exp(np.full((100, 100), -4.0, dtype=np.float64))
    target = relative_composition(state)
    first_rng = streams(501)
    second_rng = streams(501)
    first = sample_complete_kernel(
        state,
        beta,
        definition(2),
        target,
        generation_local_step=0,
        event_rng=first_rng[0],
        trim_rng=first_rng[1],
        fission_rng=first_rng[2],
        daughter_rng=first_rng[3],
        samples=128,
    )
    second = sample_complete_kernel(
        state,
        beta,
        definition(2),
        target,
        generation_local_step=0,
        event_rng=second_rng[0],
        trim_rng=second_rng[1],
        fission_rng=second_rng[2],
        daughter_rng=second_rng[3],
        samples=128,
    )
    assert np.array_equal(first.delta_composition, second.delta_composition)
    assert np.array_equal(first.next_scores, second.next_scores)

    permutation = np.arange(100)[::-1]
    moments = analytic_count_moments(
        state, beta, definition(2), generation_local_step=0
    )
    permuted = analytic_count_moments(
        state[permutation],
        beta[np.ix_(permutation, permutation)],
        definition(2),
        generation_local_step=0,
    )
    assert np.allclose(moments.mean_delta[permutation], permuted.mean_delta)
    assert np.allclose(
        moments.covariance_delta[np.ix_(permutation, permutation)],
        permuted.covariance_delta,
    )


def test_composition_moments_gradient_and_hitting_bounds() -> None:
    state = np.zeros(100, dtype=np.int64)
    state[:40] = 1
    beta = np.ones((100, 100), dtype=np.float64)
    target = relative_composition(state)
    moments = analytic_count_moments(
        state, beta, definition(2), generation_local_step=0
    )
    mean, covariance = composition_linearized_moments(state, moments)
    gradient = cosine_gradient(target, target)
    assert mean.shape == (100,)
    assert covariance.shape == (100, 100)
    assert np.allclose(gradient, 0, atol=1e-12)
    assert brownian_hitting_probability(0.1, 0.01, 0.002, 32) > 0
    assert brownian_hitting_probability(0, -1, 0, 32) == 1
