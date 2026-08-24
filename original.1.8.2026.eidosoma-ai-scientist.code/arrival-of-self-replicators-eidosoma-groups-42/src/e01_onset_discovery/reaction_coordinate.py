"""Compact pre-onset window features for the S19-L24 reaction coordinate."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

WINDOW_COUNT = 32
MOLECULE_TYPES = 100
RECURRENCE_H = 0.9


def _slope(values: NDArray[np.floating[Any]]) -> float:
    y = np.asarray(values, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64)
    x -= np.mean(x)
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y - np.mean(y)) / denominator) if denominator else 0.0


def _safe_cosine(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-15:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def extract_window_features(states: NDArray[np.integer[Any]]) -> dict[str, float]:
    counts = np.asarray(states, dtype=np.float64)
    if counts.shape != (WINDOW_COUNT, MOLECULE_TYPES) or np.any(counts < 0.0):
        raise ValueError("reaction-coordinate window must be 32-by-100 nonnegative counts")
    mass = np.sum(counts, axis=1)
    if np.any(mass <= 0.0):
        raise ValueError("reaction-coordinate window contains an empty state")
    composition = counts / mass[:, None]
    positive = composition > 0.0
    entropy = -np.sum(
        np.where(
            positive,
            composition * np.log(np.where(positive, composition, 1.0)),
            0.0,
        ),
        axis=1,
    )
    effective = np.exp(entropy)
    maximum = np.max(composition, axis=1)
    simpson = np.sum(composition * composition, axis=1)
    normalized = composition / np.linalg.norm(composition, axis=1)[:, None]
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    adjacent_h = np.diag(similarity, k=-1)
    chord = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * adjacent_h))
    max_prior = np.full(WINDOW_COUNT, np.nan, dtype=np.float64)
    recurrence_hits = 0
    recurrence_pairs = 0
    for time in range(2, WINDOW_COUNT):
        eligible = similarity[time, : time - 1]
        max_prior[time] = float(np.max(eligible))
        recurrence_hits += int(np.count_nonzero(eligible > RECURRENCE_H))
        recurrence_pairs += len(eligible)
    finite_prior = max_prior[np.isfinite(max_prior)]

    centroid = np.mean(composition, axis=0)
    centered = composition - centroid
    radius = np.linalg.norm(centered, axis=1)
    singular = np.linalg.svd(centered, compute_uv=False)
    eigen = singular * singular
    total_eigen = float(np.sum(eigen))
    participation = (
        float(np.square(total_eigen) / np.sum(eigen * eigen))
        if np.sum(eigen * eigen) > 0.0
        else 0.0
    )
    pc1_fraction = float(eigen[0] / total_eigen) if total_eigen > 0.0 else 0.0
    velocity = np.diff(composition, axis=0)
    acceleration = np.diff(velocity, axis=0)
    directional = [
        _safe_cosine(left, right)
        for left, right in zip(velocity[:-1], velocity[1:], strict=True)
    ]
    result = {
        "log_mass_mean": float(np.mean(np.log(mass))),
        "log_mass_slope": _slope(np.log(mass)),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_std": float(np.std(entropy)),
        "entropy_slope": _slope(entropy),
        "effective_diversity_mean": float(np.mean(effective)),
        "effective_diversity_slope": _slope(effective),
        "maximum_fraction_mean": float(np.mean(maximum)),
        "maximum_fraction_slope": _slope(maximum),
        "simpson_mean": float(np.mean(simpson)),
        "simpson_slope": _slope(simpson),
        "adjacent_h_mean": float(np.mean(adjacent_h)),
        "adjacent_h_std": float(np.std(adjacent_h)),
        "adjacent_h_min": float(np.min(adjacent_h)),
        "adjacent_h_slope": _slope(adjacent_h),
        "chord_speed_mean": float(np.mean(chord)),
        "chord_speed_std": float(np.std(chord)),
        "chord_speed_slope": _slope(chord),
        "max_prior_h_mean": float(np.mean(finite_prior)),
        "max_prior_h_last": float(finite_prior[-1]),
        "max_prior_h_slope": _slope(max_prior[2:]),
        "prior_recurrence_rate_h090": float(recurrence_hits / max(1, recurrence_pairs)),
        "centroid_radius_mean": float(np.mean(radius)),
        "centroid_radius_slope": _slope(radius),
        "pc1_variance_fraction": pc1_fraction,
        "participation_ratio": participation,
        "directional_persistence_mean": float(np.mean(directional)) if directional else 0.0,
        "acceleration_mean": float(np.mean(np.linalg.norm(acceleration, axis=1)))
        if acceleration.size
        else 0.0,
    }
    if not np.isfinite(list(result.values())).all():
        raise RuntimeError("reaction-coordinate feature extraction emitted nonfinite values")
    return result


REACTION_FEATURES = tuple(extract_window_features(np.ones((32, 100), dtype=np.int64)))
EXACT_H_WINDOW_FEATURES = (
    "adjacent_h_mean",
    "adjacent_h_std",
    "adjacent_h_min",
    "adjacent_h_slope",
    "chord_speed_mean",
    "chord_speed_std",
    "chord_speed_slope",
)
ORDINARY_WINDOW_FEATURES = (
    "log_mass_mean",
    "log_mass_slope",
    "entropy_mean",
    "entropy_std",
    "entropy_slope",
    "effective_diversity_mean",
    "effective_diversity_slope",
    "maximum_fraction_mean",
    "maximum_fraction_slope",
    "simpson_mean",
    "simpson_slope",
    *EXACT_H_WINDOW_FEATURES,
)
