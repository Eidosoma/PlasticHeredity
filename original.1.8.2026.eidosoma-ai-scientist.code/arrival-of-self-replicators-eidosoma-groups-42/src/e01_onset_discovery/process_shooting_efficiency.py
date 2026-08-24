"""Registered binomial estimators for the S19-L48 shooting audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True, slots=True)
class JeffreysEstimate:
    successes: int
    trials: int
    posterior_mean: float
    posterior_variance: float
    lower95: float
    upper95: float


def jeffreys_estimate(successes: int, trials: int) -> JeffreysEstimate:
    """Return the frozen Beta(1/2, 1/2) posterior summary."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    alpha = successes + 0.5
    beta_value = trials - successes + 0.5
    total = alpha + beta_value
    return JeffreysEstimate(
        successes=int(successes),
        trials=int(trials),
        posterior_mean=float(alpha / total),
        posterior_variance=float(
            alpha * beta_value / (total**2 * (total + 1.0))
        ),
        lower95=float(beta.ppf(0.025, alpha, beta_value)),
        upper95=float(beta.ppf(0.975, alpha, beta_value)),
    )


def bernoulli_scores(
    probability: float, outcomes: np.ndarray
) -> dict[str, float]:
    """Score one fixed probability against independent Bernoulli outcomes."""
    values = np.asarray(outcomes, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("outcomes must be a nonempty binary vector")
    clipped = float(np.clip(probability, 1e-12, 1 - 1e-12))
    return {
        "brier": float(np.mean((values - clipped) ** 2)),
        "logLoss": float(
            -np.mean(values * np.log(clipped) + (1 - values) * np.log(1 - clipped))
        ),
    }


def next_uncertainty_allocation(
    state_ids: list[str],
    successes: np.ndarray,
    trials: np.ndarray,
    allocated: np.ndarray,
    *,
    cap: int = 64,
) -> int:
    """Choose the eligible state with largest Jeffreys posterior variance.

    Lexicographic state identity is the frozen tie breaker.
    """
    successes_array = np.asarray(successes, dtype=np.int64)
    trials_array = np.asarray(trials, dtype=np.int64)
    allocated_array = np.asarray(allocated, dtype=np.int64)
    if not (
        len(state_ids)
        == len(successes_array)
        == len(trials_array)
        == len(allocated_array)
    ):
        raise ValueError("allocation arrays must align")
    eligible = np.flatnonzero(allocated_array < cap)
    if not len(eligible):
        raise ValueError("no state remains below the allocation cap")
    ranked = sorted(
        (
            -jeffreys_estimate(
                int(successes_array[index]), int(trials_array[index])
            ).posterior_variance,
            state_ids[index],
            int(index),
        )
        for index in eligible
    )
    return ranked[0][2]
