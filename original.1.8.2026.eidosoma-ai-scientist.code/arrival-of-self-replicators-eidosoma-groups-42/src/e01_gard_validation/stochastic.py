"""Calibrated stochastic-law checks used by E01 research step S07.

The analytical targets in this module are independent of both simulator
implementations.  Statistical tests use fixed Monte Carlo algorithms and do
not infer author-code or legacy MATLAB random-number semantics.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from math import comb, isclose
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2, norm, poisson


class StochasticValidationError(ValueError):
    """A validation target, observation, or calibration is malformed."""


@dataclass(frozen=True, slots=True)
class AnalyticalPropensities:
    """Independent calculation of the frozen GARD event law."""

    boost: tuple[float, ...]
    join: tuple[float, ...]
    leave: tuple[float, ...]
    concatenated: tuple[float, ...]
    probabilities: tuple[float, ...]
    total: float


@dataclass(frozen=True, slots=True)
class DistributionTarget:
    """Finite categorical support and exact analytical probabilities."""

    labels: tuple[str, ...]
    outcomes: tuple[Any, ...]
    probabilities: tuple[float, ...]


def _probabilities(values: ArrayLike) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 2:
        raise StochasticValidationError(
            "A probability vector must contain at least two categories."
        )
    if not np.all(np.isfinite(vector)) or np.any(vector < 0):
        raise StochasticValidationError("Probabilities must be finite and nonnegative.")
    total = float(vector.sum())
    if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise StochasticValidationError(
            f"Probabilities must sum to one, not {total!r}."
        )
    if not np.any(vector > 0):
        raise StochasticValidationError("At least one category must be possible.")
    return vector / total


def _counts(values: ArrayLike, *, length: int) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.shape != (length,):
        raise StochasticValidationError(f"Counts must have shape ({length},).")
    numeric = np.asarray(values, dtype=np.float64)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0)
        or not np.all(numeric == np.floor(numeric))
    ):
        raise StochasticValidationError("Counts must be finite nonnegative integers.")
    result = numeric.astype(np.int64)
    if int(result.sum()) <= 0:
        raise StochasticValidationError("At least one observation is required.")
    return result


def analytical_propensities(
    state: ArrayLike,
    *,
    beta: ArrayLike,
    rho: ArrayLike,
    k_f: float,
    k_b: float,
    orientation: str = "historical_orientation_with_diagonal",
) -> AnalyticalPropensities:
    """Calculate the frozen event weights without importing either engine."""

    counts = np.asarray(state, dtype=np.float64)
    if (
        counts.ndim != 1
        or counts.size == 0
        or not np.all(np.isfinite(counts))
        or np.any(counts < 0)
        or not np.all(counts == np.floor(counts))
    ):
        raise StochasticValidationError(
            "state must be a finite, nonnegative integer vector."
        )
    mass = float(counts.sum())
    if mass <= 0:
        raise StochasticValidationError("state must have positive mass.")
    matrix = np.asarray(beta, dtype=np.float64)
    if matrix.shape != (counts.size, counts.size):
        raise StochasticValidationError("beta shape does not match state.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise StochasticValidationError("beta must be finite and nonnegative.")
    reservoir = np.asarray(rho, dtype=np.float64)
    if reservoir.shape != counts.shape:
        raise StochasticValidationError("rho shape does not match state.")
    if not np.all(np.isfinite(reservoir)) or np.any(reservoir < 0):
        raise StochasticValidationError("rho must be finite and nonnegative.")
    if not np.isfinite(k_f) or k_f < 0 or not np.isfinite(k_b) or k_b < 0:
        raise StochasticValidationError("k_f and k_b must be nonnegative.")
    if orientation == "historical_orientation_with_diagonal":
        effective = matrix
    elif orientation == "transposed_with_diagonal":
        effective = matrix.T
    elif orientation == "historical_orientation_zero_diagonal":
        effective = matrix.copy()
        np.fill_diagonal(effective, 0.0)
    else:
        raise StochasticValidationError(f"Unknown beta orientation {orientation!r}.")
    boost = 1.0 + (effective @ counts) / mass
    join = float(k_f) * reservoir * mass * boost
    leave = float(k_b) * counts * boost
    concatenated = np.concatenate((join, leave))
    total = float(concatenated.sum())
    if total <= 0 or not np.all(np.isfinite(concatenated)):
        raise StochasticValidationError("Analytical total propensity must be positive.")
    probabilities = concatenated / total
    return AnalyticalPropensities(
        boost=tuple(float(value) for value in boost),
        join=tuple(float(value) for value in join),
        leave=tuple(float(value) for value in leave),
        concatenated=tuple(float(value) for value in concatenated),
        probabilities=tuple(float(value) for value in probabilities),
        total=total,
    )


def multinomial_deviance(counts: ArrayLike, probabilities: ArrayLike) -> float:
    """Return the log-likelihood deviance G-squared for a fixed target law."""

    target = _probabilities(probabilities)
    observed = _counts(counts, length=target.size)
    structural_violation = (target == 0.0) & (observed > 0)
    if np.any(structural_violation):
        return float("inf")
    positive = target > 0.0
    target = target[positive]
    observed = observed[positive]
    expected = float(observed.sum()) * target
    terms = np.zeros_like(expected)
    nonzero = observed > 0
    terms[nonzero] = observed[nonzero] * np.log(observed[nonzero] / expected[nonzero])
    return float(2.0 * terms.sum())


def exact_multinomial_test(
    counts: ArrayLike,
    probabilities: ArrayLike,
    *,
    generator: np.random.Generator,
    replicates: int,
    batch_size: int,
) -> dict[str, Any]:
    """Parametric-exact multinomial test with a fixed plus-one p-value.

    The null simulator uses the exact multinomial law, so low expected counts do
    not invoke an asymptotic chi-square approximation.  Monte Carlo uncertainty
    is bounded and reported from the preregistered number of replicates.
    """

    if not isinstance(generator, np.random.Generator):
        raise StochasticValidationError("An explicit NumPy Generator is required.")
    if (
        not isinstance(replicates, int)
        or isinstance(replicates, bool)
        or replicates < 999
    ):
        raise StochasticValidationError("replicates must be an integer >= 999.")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise StochasticValidationError("batch_size must be a positive integer.")
    target_all = _probabilities(probabilities)
    observed_all = _counts(counts, length=target_all.size)
    structural = np.flatnonzero((target_all == 0.0) & (observed_all > 0))
    if structural.size:
        return {
            "statistic": float("inf"),
            "pValue": 0.0,
            "exceedances": 0,
            "replicates": replicates,
            "monteCarloStandardError": 0.0,
            "minimumAttainablePValue": 1.0 / (replicates + 1),
            "structuralZeroViolationIndices": structural.tolist(),
        }
    positive = target_all > 0.0
    target = target_all[positive]
    observed = observed_all[positive]
    n = int(observed.sum())
    expected = n * target
    statistic = multinomial_deviance(observed, target)
    exceedances = 0
    completed = 0
    while completed < replicates:
        current = min(batch_size, replicates - completed)
        simulated = generator.multinomial(n, target, size=current)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = simulated / expected
            terms = np.where(simulated > 0, simulated * np.log(ratios), 0.0)
        simulated_statistics = 2.0 * np.sum(terms, axis=1)
        exceedances += int(np.count_nonzero(simulated_statistics >= statistic - 1e-12))
        completed += current
    p_value = (exceedances + 1.0) / (replicates + 1.0)
    standard_error = float(np.sqrt(p_value * (1.0 - p_value) / (replicates + 1)))
    return {
        "statistic": statistic,
        "pValue": p_value,
        "exceedances": exceedances,
        "replicates": replicates,
        "monteCarloStandardError": standard_error,
        "minimumAttainablePValue": 1.0 / (replicates + 1),
        "structuralZeroViolationIndices": [],
    }


def pool_rare_categories(
    counts: ArrayLike,
    probabilities: ArrayLike,
    *,
    labels: Sequence[str],
    minimum_expected: float,
) -> dict[str, Any]:
    """Apply the preregistered diagnostic pooling rule to rare categories."""

    target = _probabilities(probabilities)
    observed = _counts(counts, length=target.size)
    if len(labels) != target.size:
        raise StochasticValidationError("labels length does not match target.")
    if not np.isfinite(minimum_expected) or minimum_expected <= 0:
        raise StochasticValidationError("minimum_expected must be positive.")
    expected = int(observed.sum()) * target
    structural = target == 0.0
    rare = (target > 0.0) & (expected < minimum_expected)
    ordinary = ~(structural | rare)
    pooled_labels = [str(labels[index]) for index in np.flatnonzero(ordinary)]
    pooled_counts = [int(observed[index]) for index in np.flatnonzero(ordinary)]
    pooled_probabilities = [float(target[index]) for index in np.flatnonzero(ordinary)]
    rare_indices = np.flatnonzero(rare)
    if rare_indices.size:
        pooled_labels.append("POOLED_RARE")
        pooled_counts.append(int(observed[rare].sum()))
        pooled_probabilities.append(float(target[rare].sum()))
    pooled_expected = np.asarray(pooled_probabilities) * int(observed.sum())
    eligible = bool(
        pooled_expected.size > 1 and np.all(pooled_expected >= minimum_expected)
    )
    return {
        "minimumExpectedCount": float(minimum_expected),
        "unpooledExpectedCounts": expected.tolist(),
        "rareIndices": rare_indices.tolist(),
        "rareLabels": [str(labels[index]) for index in rare_indices],
        "structuralZeroIndices": np.flatnonzero(structural).tolist(),
        "pooledLabels": pooled_labels,
        "pooledCounts": pooled_counts,
        "pooledProbabilities": pooled_probabilities,
        "pooledExpectedCounts": pooled_expected.tolist(),
        "asymptoticPearsonEligible": eligible,
        "primaryMethod": "EXACT_PARAMETRIC_MONTE_CARLO_UNPOOLED",
    }


def two_sample_target_tv_test(
    left_counts: ArrayLike,
    right_counts: ArrayLike,
    probabilities: ArrayLike,
    *,
    generator: np.random.Generator,
    replicates: int,
    batch_size: int,
) -> dict[str, Any]:
    """Calibrate a two-engine TV distance under the known common target."""

    target = _probabilities(probabilities)
    left = _counts(left_counts, length=target.size)
    right = _counts(right_counts, length=target.size)
    left_n = int(left.sum())
    right_n = int(right.sum())
    observed = float(0.5 * np.abs(left / left_n - right / right_n).sum())
    exceedances = 0
    completed = 0
    while completed < replicates:
        current = min(batch_size, replicates - completed)
        simulated_left = generator.multinomial(left_n, target, size=current)
        simulated_right = generator.multinomial(right_n, target, size=current)
        simulated_tv = 0.5 * np.abs(
            simulated_left / left_n - simulated_right / right_n
        ).sum(axis=1)
        exceedances += int(np.count_nonzero(simulated_tv >= observed - 1e-15))
        completed += current
    p_value = (exceedances + 1.0) / (replicates + 1.0)
    return {
        "statistic": observed,
        "pValue": p_value,
        "exceedances": exceedances,
        "replicates": replicates,
        "monteCarloStandardError": float(
            np.sqrt(p_value * (1.0 - p_value) / (replicates + 1))
        ),
        "minimumAttainablePValue": 1.0 / (replicates + 1),
    }


def _state_outcomes(
    parent: tuple[int, ...], selected_mass: int
) -> Iterable[tuple[int, ...]]:
    for outcome in product(*(range(value + 1) for value in parent)):
        if sum(outcome) == selected_mass:
            yield tuple(int(value) for value in outcome)


def fixed_fission_distribution(parent: Sequence[int]) -> DistributionTarget:
    """Enumerate exact historical fixed-size fission outcomes.

    Even-parent outcomes are child-A states. Odd-parent outcomes are
    ``(child-A, discarded-vector)`` pairs because the historical law discards
    one uniformly selected remaining molecule after forming child A.
    """

    parent_tuple = tuple(int(value) for value in parent)
    if not parent_tuple or any(value < 0 for value in parent_tuple):
        raise StochasticValidationError("parent must be nonempty and nonnegative.")
    mass = sum(parent_tuple)
    if mass <= 0:
        raise StochasticValidationError("parent must have positive mass.")
    selected_mass = mass // 2
    denominator = comb(mass, selected_mass)
    outcomes: list[Any] = []
    probabilities: list[float] = []
    for first in _state_outcomes(parent_tuple, selected_mass):
        combinations = 1
        for available, selected in zip(parent_tuple, first, strict=True):
            combinations *= comb(available, selected)
        first_probability = combinations / denominator
        if mass % 2 == 0:
            outcomes.append(first)
            probabilities.append(first_probability)
            continue
        remainder = tuple(
            available - selected
            for available, selected in zip(parent_tuple, first, strict=True)
        )
        remaining_mass = mass - selected_mass
        for species, available in enumerate(remainder):
            if available == 0:
                continue
            discarded = tuple(
                1 if index == species else 0 for index in range(len(parent_tuple))
            )
            outcomes.append((first, discarded))
            probabilities.append(first_probability * available / remaining_mass)
    order = sorted(range(len(outcomes)), key=lambda index: repr(outcomes[index]))
    sorted_outcomes = tuple(outcomes[index] for index in order)
    sorted_probabilities = tuple(float(probabilities[index]) for index in order)
    if not isclose(sum(sorted_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise StochasticValidationError("Fixed-fission target does not sum to one.")
    return DistributionTarget(
        labels=tuple(repr(outcome) for outcome in sorted_outcomes),
        outcomes=sorted_outcomes,
        probabilities=sorted_probabilities,
    )


def binomial_fission_distribution(
    parent: Sequence[int], *, probability: float
) -> DistributionTarget:
    """Enumerate exact independent per-species binomial child counts."""

    parent_tuple = tuple(int(value) for value in parent)
    if not parent_tuple or any(value < 0 for value in parent_tuple):
        raise StochasticValidationError("parent must be nonempty and nonnegative.")
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise StochasticValidationError("probability must lie in [0, 1].")
    outcomes: list[tuple[int, ...]] = []
    probabilities: list[float] = []
    for outcome in product(*(range(value + 1) for value in parent_tuple)):
        target_probability = 1.0
        for available, selected in zip(parent_tuple, outcome, strict=True):
            target_probability *= (
                comb(available, selected)
                * probability**selected
                * (1.0 - probability) ** (available - selected)
            )
        outcomes.append(tuple(int(value) for value in outcome))
        probabilities.append(float(target_probability))
    order = sorted(range(len(outcomes)), key=lambda index: repr(outcomes[index]))
    sorted_outcomes = tuple(outcomes[index] for index in order)
    sorted_probabilities = tuple(probabilities[index] for index in order)
    if not isclose(sum(sorted_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise StochasticValidationError("Binomial target does not sum to one.")
    return DistributionTarget(
        labels=tuple(repr(outcome) for outcome in sorted_outcomes),
        outcomes=sorted_outcomes,
        probabilities=sorted_probabilities,
    )


def poisson_count_bins(
    rate: float, *, tail_probability_floor: float
) -> DistributionTarget:
    """Create exact Poisson count categories with one final aggregated tail."""

    if not np.isfinite(rate) or rate < 0:
        raise StochasticValidationError("Poisson rate must be finite and nonnegative.")
    if not np.isfinite(tail_probability_floor) or not 0 < tail_probability_floor < 1:
        raise StochasticValidationError(
            "tail_probability_floor must lie strictly between zero and one."
        )
    if rate == 0.0:
        return DistributionTarget(
            labels=("0", ">=1"), outcomes=(0, 1), probabilities=(1.0, 0.0)
        )
    tail_start = 1
    while float(poisson.sf(tail_start - 1, rate)) >= tail_probability_floor:
        tail_start += 1
        if tail_start > 100000:
            raise StochasticValidationError("Poisson tail search did not converge.")
    probabilities = [float(poisson.pmf(value, rate)) for value in range(tail_start)]
    probabilities.append(float(poisson.sf(tail_start - 1, rate)))
    total = sum(probabilities)
    probabilities[-1] += 1.0 - total
    labels = tuple([str(value) for value in range(tail_start)] + [f">={tail_start}"])
    outcomes = tuple(range(tail_start + 1))
    return DistributionTarget(
        labels=labels,
        outcomes=outcomes,
        probabilities=tuple(probabilities),
    )


def bin_observations(values: Sequence[int], target: DistributionTarget) -> list[int]:
    """Count integer observations in exact bins, including the final tail."""

    if not values:
        raise StochasticValidationError("values must not be empty.")
    if not target.labels[-1].startswith(">="):
        raise StochasticValidationError("target does not end with a tail category.")
    tail_start = int(target.labels[-1][2:])
    result = np.zeros(len(target.labels), dtype=np.int64)
    for raw in values:
        value = int(raw)
        if value < 0:
            raise StochasticValidationError("Poisson observations cannot be negative.")
        result[min(value, tail_start)] += 1
    return result.tolist()


def lognormal_log_moment_tests(
    *,
    sample_count: int,
    sample_mean: float,
    sample_variance: float,
    expected_mean: float,
    expected_variance: float,
) -> dict[str, dict[str, float]]:
    """Exact two-sided mean and variance tests for normal log-beta values."""

    if not isinstance(sample_count, int) or sample_count < 3:
        raise StochasticValidationError("sample_count must be an integer >= 3.")
    values = (sample_mean, sample_variance, expected_mean, expected_variance)
    if not all(np.isfinite(value) for value in values):
        raise StochasticValidationError("Moment inputs must be finite.")
    if sample_variance < 0 or expected_variance <= 0:
        raise StochasticValidationError("Variances must be valid and positive.")
    z_statistic = (sample_mean - expected_mean) / np.sqrt(
        expected_variance / sample_count
    )
    mean_p = float(2.0 * norm.sf(abs(z_statistic)))
    degrees = sample_count - 1
    variance_statistic = degrees * sample_variance / expected_variance
    lower = float(chi2.cdf(variance_statistic, degrees))
    upper = float(chi2.sf(variance_statistic, degrees))
    variance_p = min(1.0, 2.0 * min(lower, upper))
    return {
        "mean": {"statistic": float(z_statistic), "pValue": mean_p},
        "variance": {
            "statistic": float(variance_statistic),
            "degreesOfFreedom": float(degrees),
            "pValue": float(variance_p),
        },
    }


def calibrated_moment_intervals(
    *,
    sample_count: int,
    expected_mean: float,
    expected_variance: float,
    alpha: float,
) -> dict[str, list[float]]:
    """Return exact fixed acceptance intervals before sample inspection."""

    if not 0 < alpha < 1:
        raise StochasticValidationError("alpha must lie in (0, 1).")
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    standard_error = np.sqrt(expected_variance / sample_count)
    degrees = sample_count - 1
    variance_lower = expected_variance * float(chi2.ppf(alpha / 2.0, degrees)) / degrees
    variance_upper = (
        expected_variance * float(chi2.ppf(1.0 - alpha / 2.0, degrees)) / degrees
    )
    return {
        "mean": [
            float(expected_mean - critical * standard_error),
            float(expected_mean + critical * standard_error),
        ],
        "variance": [variance_lower, variance_upper],
    }
