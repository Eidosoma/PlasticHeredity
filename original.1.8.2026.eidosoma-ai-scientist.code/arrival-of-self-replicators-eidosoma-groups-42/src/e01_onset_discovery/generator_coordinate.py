"""Exact and simulator-kernel local moments for S19-L29.

The analytical branch is the source-defined birth/death or fission generator.
For growth states it is explicitly the pre-overshoot-trim reaction generator;
the registered simulator-kernel branch estimates the moments of the complete
one-selected-clock-step implementation, including clipping, trimming, fission,
and daughter selection, from independent one-step samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr
from scipy.stats import poisson

from e01_latent_timebase.core import (
    MAX_STEPS,
    N_MAX,
    SimulationDefinition,
    exposure_for_rates,
    rates,
)
from e01_onset_discovery.empirical_committor import cosine_to_reference

VERSION = "E01-S19-L29-EXACT-GARD-GENERATOR-COMMITTOR-COORDINATE-v1.0.0"
TARGET_THRESHOLD = 0.9
KERNEL_SAMPLES = 2048
KERNEL_HALF_SAMPLES = 1024


@dataclass(frozen=True, slots=True)
class CountMoments:
    mean_delta: NDArray[np.float64]
    covariance_delta: NDArray[np.float64]
    transition_kind: str
    semantics: str


@dataclass(frozen=True, slots=True)
class KernelSamples:
    delta_composition: NDArray[np.float64]
    next_scores: NDArray[np.float64]
    empty_next: NDArray[np.bool_]
    transition_kind: str


def relative_composition(state: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
    value = np.asarray(state, dtype=np.float64)
    if value.shape != (100,) or np.any(value < 0) or value.sum() <= 0:
        raise ValueError("state must be a nonempty nonnegative 100-vector")
    return value / value.sum()


def truncated_poisson_moments(rate: float, cap: int) -> tuple[float, float]:
    """Return E[min(Pois(rate), cap)] and its variance."""

    if not np.isfinite(rate) or rate < 0 or cap < 0:
        raise ValueError("invalid truncated-Poisson arguments")
    if cap == 0 or rate == 0:
        return 0.0, 0.0
    values = np.arange(cap, dtype=np.float64)
    probabilities = poisson.pmf(values, rate)
    tail = float(poisson.sf(cap - 1, rate))
    mean = float(np.dot(values, probabilities) + cap * tail)
    second = float(np.dot(values * values, probabilities) + cap * cap * tail)
    return mean, max(0.0, second - mean * mean)


def _random_nonempty_fission_moments(
    state: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exact selected-daughter moments for the frozen random-nonempty rule."""

    counts = np.asarray(state, dtype=np.float64)
    mass = int(counts.sum())
    p_empty = math.ldexp(1.0, -mass)
    mean = (0.5 + p_empty) * counts
    second = 0.25 * np.outer(counts, counts) + p_empty * np.outer(counts, counts)
    diagonal = 0.25 * counts + (0.25 + p_empty) * counts * counts
    np.fill_diagonal(second, diagonal)
    covariance = second - np.outer(mean, mean)
    covariance = (covariance + covariance.T) * 0.5
    return mean, covariance


def analytic_count_moments(
    state: NDArray[np.integer[Any]],
    beta: NDArray[np.floating[Any]],
    definition: SimulationDefinition,
    *,
    generation_local_step: int,
) -> CountMoments:
    """Calculate the frozen source-defined local count drift and covariance."""

    counts = np.asarray(state, dtype=np.int64)
    matrix = np.asarray(beta, dtype=np.float64)
    if counts.shape != (100,) or matrix.shape != (100, 100):
        raise ValueError("invalid state or catalytic matrix shape")
    mass = int(counts.sum())
    if mass <= 0:
        raise ValueError("empty states have no local generator")
    if mass >= N_MAX or generation_local_step >= MAX_STEPS:
        if definition.daughter_rule == "FIRST_DAUGHTER":
            mean_next = 0.5 * counts.astype(np.float64)
            covariance = np.diag(0.25 * counts.astype(np.float64))
            semantics = "EXACT_FIRST_DAUGHTER_BINOMIAL_FISSION"
        elif definition.daughter_rule == "RANDOM_NONEMPTY":
            mean_next, covariance = _random_nonempty_fission_moments(counts)
            semantics = "EXACT_RANDOM_NONEMPTY_DAUGHTER_FISSION"
        else:
            raise ValueError("L29 supports only the frozen candidate daughter rules")
        return CountMoments(
            mean_next - counts,
            covariance,
            "FISSION",
            semantics,
        )

    joins, losses = rates(counts, matrix)
    exposure = exposure_for_rates(definition.exposure, joins, losses)
    join_means = exposure * joins
    loss_means = np.zeros(100, dtype=np.float64)
    loss_variances = np.zeros(100, dtype=np.float64)
    for index, cap in enumerate(counts):
        loss_means[index], loss_variances[index] = truncated_poisson_moments(
            float(exposure * losses[index]), int(cap)
        )
    mean_delta = join_means - loss_means
    covariance = np.diag(join_means + loss_variances)
    return CountMoments(
        mean_delta,
        covariance,
        "MOLECULAR_UPDATE",
        "EXACT_PRETRIM_CLIPPED_POISSON_REACTION_GENERATOR",
    )


def _selected_fission_samples(
    state: NDArray[np.int64],
    definition: SimulationDefinition,
    samples: int,
    fission_rng: np.random.Generator,
    daughter_rng: np.random.Generator,
) -> NDArray[np.int64]:
    child_a = fission_rng.binomial(state, 0.5, size=(samples, 100)).astype(
        np.int64, copy=False
    )
    if definition.daughter_rule == "FIRST_DAUGHTER":
        return child_a
    if definition.daughter_rule != "RANDOM_NONEMPTY":
        raise ValueError("unsupported daughter rule")
    child_b = state[None, :] - child_a
    mass_a = child_a.sum(axis=1)
    mass_b = child_b.sum(axis=1)
    choose_a = daughter_rng.integers(0, 2, size=samples).astype(bool)
    choose_a[mass_a == 0] = False
    choose_a[mass_b == 0] = True
    return np.where(choose_a[:, None], child_a, child_b)


def sample_complete_kernel(
    state: NDArray[np.integer[Any]],
    beta: NDArray[np.floating[Any]],
    definition: SimulationDefinition,
    target_centroid: NDArray[np.floating[Any]],
    *,
    generation_local_step: int,
    event_rng: np.random.Generator,
    trim_rng: np.random.Generator,
    fission_rng: np.random.Generator,
    daughter_rng: np.random.Generator,
    samples: int = KERNEL_SAMPLES,
) -> KernelSamples:
    """Sample the complete frozen one-selected-clock-step transition kernel."""

    counts = np.asarray(state, dtype=np.int64)
    matrix = np.asarray(beta, dtype=np.float64)
    target = np.asarray(target_centroid, dtype=np.float64)
    current = relative_composition(counts)
    mass = int(counts.sum())
    if samples <= 1:
        raise ValueError("at least two kernel samples are required")
    if mass >= N_MAX or generation_local_step >= MAX_STEPS:
        next_counts = _selected_fission_samples(
            counts, definition, samples, fission_rng, daughter_rng
        )
        kind = "FISSION"
    else:
        joins, losses = rates(counts, matrix)
        exposure = exposure_for_rates(definition.exposure, joins, losses)
        join_draws = event_rng.poisson(exposure * joins, size=(samples, 100)).astype(
            np.int64, copy=False
        )
        attempted_losses = event_rng.poisson(
            exposure * losses, size=(samples, 100)
        ).astype(np.int64, copy=False)
        applied_losses = np.minimum(attempted_losses, counts[None, :])
        next_counts = counts[None, :] + join_draws - applied_losses
        if definition.overshoot_rule == "TRIM_NEW_ENTRANTS_TO_NMAX":
            excess = np.maximum(0, next_counts.sum(axis=1) - N_MAX)
            for row_index in np.flatnonzero(excess):
                remove = trim_rng.multivariate_hypergeometric(
                    join_draws[row_index], int(excess[row_index])
                ).astype(np.int64, copy=False)
                next_counts[row_index] -= remove
        if np.any(next_counts < 0):
            raise AssertionError("kernel sample became negative")
        kind = "MOLECULAR_UPDATE"

    masses = next_counts.sum(axis=1)
    empty = masses == 0
    next_compositions = np.zeros((samples, 100), dtype=np.float64)
    next_compositions[~empty] = next_counts[~empty] / masses[~empty, None]
    next_scores = np.zeros(samples, dtype=np.float64)
    if np.any(~empty):
        next_scores[~empty] = cosine_to_reference(next_counts[~empty], target)
    return KernelSamples(
        delta_composition=next_compositions - current[None, :],
        next_scores=next_scores,
        empty_next=empty,
        transition_kind=kind,
    )


def composition_linearized_moments(
    state: NDArray[np.integer[Any]], moments: CountMoments
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    counts = np.asarray(state, dtype=np.float64)
    mass = counts.sum()
    composition = counts / mass
    jacobian = (np.eye(100) - np.outer(composition, np.ones(100))) / mass
    mean = jacobian @ moments.mean_delta
    covariance = jacobian @ moments.covariance_delta @ jacobian.T
    return mean, (covariance + covariance.T) * 0.5


def cosine_gradient(
    composition: NDArray[np.floating[Any]],
    target: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    x = np.asarray(composition, dtype=np.float64)
    c = np.asarray(target, dtype=np.float64)
    norm_x = np.linalg.norm(x)
    norm_c = np.linalg.norm(c)
    score = float(np.dot(x, c) / (norm_x * norm_c))
    return c / (norm_x * norm_c) - score * x / (norm_x * norm_x)


def brownian_hitting_probability(
    distance: float, drift: float, variance: float, horizon: int
) -> float:
    """Finite-time upper-barrier probability under frozen local coefficients."""

    if distance <= 0:
        return 1.0
    if (
        horizon <= 0
        or variance < 0
        or not np.isfinite([distance, drift, variance]).all()
    ):
        raise ValueError("invalid Brownian hitting arguments")
    if variance <= 1e-18:
        return float(drift > 0 and drift * horizon >= distance)
    scale = math.sqrt(variance * horizon)
    first = float(ndtr((drift * horizon - distance) / scale))
    log_factor = 2.0 * drift * distance / variance
    second_cdf = float(ndtr((-drift * horizon - distance) / scale))
    if second_cdf == 0.0:
        second = 0.0
    else:
        second = math.exp(min(700.0, log_factor + math.log(second_cdf)))
    return float(np.clip(first + second, 0.0, 1.0))


def summarize_moments(
    state: NDArray[np.integer[Any]],
    target: NDArray[np.floating[Any]],
    mean_delta_x: NDArray[np.floating[Any]],
    covariance_delta_x: NDArray[np.floating[Any]],
    *,
    score_drift: float | None = None,
    score_variance: float | None = None,
    one_step_hit_probability: float | None = None,
    prefix: str,
) -> dict[str, float]:
    counts = np.asarray(state, dtype=np.float64)
    x = relative_composition(counts.astype(np.int64))
    c = np.asarray(target, dtype=np.float64)
    mu = np.asarray(mean_delta_x, dtype=np.float64)
    covariance = np.asarray(covariance_delta_x, dtype=np.float64)
    score = float(cosine_to_reference(counts.astype(np.int64)[None, :], c)[0])
    gradient = cosine_gradient(x, c)
    radial_drift = (
        float(np.dot(gradient, mu)) if score_drift is None else float(score_drift)
    )
    radial_variance = (
        float(gradient @ covariance @ gradient)
        if score_variance is None
        else float(score_variance)
    )
    radial_variance = max(0.0, radial_variance)
    direction = c - x
    direction_norm = float(np.linalg.norm(direction))
    unit_direction = direction / direction_norm if direction_norm > 0 else direction
    eigmax = float(np.linalg.eigvalsh(covariance)[-1])
    current_norm = float(np.linalg.norm(x))
    mu_norm = float(np.linalg.norm(mu))
    current_alignment = (
        float(np.dot(x, mu) / (current_norm * mu_norm)) if mu_norm > 0 else 0.0
    )
    direction_drift = float(np.dot(unit_direction, mu))
    direction_diffusion = float(unit_direction @ covariance @ unit_direction)
    hit = brownian_hitting_probability(
        max(0.0, TARGET_THRESHOLD - score), radial_drift, radial_variance, 32
    )
    return {
        f"{prefix}MuNorm": mu_norm,
        f"{prefix}DiffusionTrace": float(np.trace(covariance)),
        f"{prefix}DiffusionEigmax": eigmax,
        f"{prefix}CurrentMuAlignment": current_alignment,
        f"{prefix}TargetDirectionDrift": direction_drift,
        f"{prefix}TargetDirectionDiffusion": max(0.0, direction_diffusion),
        f"{prefix}ScoreDrift": radial_drift,
        f"{prefix}ScoreVariance": radial_variance,
        f"{prefix}ScoreSignalNoise": radial_drift
        / math.sqrt(max(radial_variance, 1e-18)),
        f"{prefix}BrownianHit32": hit,
        f"{prefix}OneStepHitProbability": float(one_step_hit_probability)
        if one_step_hit_probability is not None
        else float("nan"),
    }
