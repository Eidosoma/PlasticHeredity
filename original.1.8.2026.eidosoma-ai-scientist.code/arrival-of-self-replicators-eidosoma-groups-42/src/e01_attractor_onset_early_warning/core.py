"""Deterministic core for the S19-L18 landmark-onset experiment.

The outcome is a frozen, completed-run recurring-attractor label.  Every
non-oracle feature in this module is a function only of the first 64 selected
molecular-clock observations.  This separation is deliberate: retrospective
outcome adjudication does not license future-dependent predictors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

VERSION = "E01-S19-L18-RECURRING-ATTRACTOR-ONSET-EARLY-WARNING-v1.0.0"
LANDMARK_COUNT = 64
HORIZON_EXCLUSIVE = 192
RECURRENCE_THRESHOLD = 0.9


def derive_seed(*identity: object) -> int:
    material = "\x1f".join([VERSION, *map(str, identity)])
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def build_landmark_target(labels: Sequence[bool]) -> dict[str, Any]:
    """Construct the fixed 64-to-192 first-onset task."""

    values = np.asarray(labels, dtype=bool)
    if values.ndim != 1 or values.size < HORIZON_EXCLUSIVE:
        raise ValueError("target sequence must contain at least 192 observations")
    positive = np.flatnonzero(values)
    first = int(positive[0]) if positive.size else None
    at_risk = first is None or first >= LANDMARK_COUNT
    event = bool(at_risk and first is not None and first < HORIZON_EXCLUSIVE)
    return {
        "observationCount": int(values.size),
        "wholeTrajectoryOccupancy": float(np.mean(values)),
        "firstOnsetIndex0": first,
        "atRiskAtLandmark": bool(at_risk),
        "eventWithinHorizon": event if at_risk else None,
        "landmarkCount": LANDMARK_COUNT,
        "horizonExclusive": HORIZON_EXCLUSIVE,
    }


def _slope(values: NDArray[np.floating[Any]]) -> float:
    y = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    x = np.arange(y.size, dtype=np.float64)[finite]
    x -= np.mean(x)
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y[finite] - np.mean(y[finite])) / denominator) if denominator else 0.0


def metric_summary(values: NDArray[np.floating[Any]], prefix: str) -> dict[str, float]:
    """Frozen nine-number local-metric summary with explicit finite coverage."""

    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return {f"{prefix}_{name}": float("nan") for name in (
            "mean", "std", "last", "slope", "min", "max", "q90", "positive_fraction"
        )} | {f"{prefix}_finite_fraction": 0.0}
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite, ddof=0)),
        f"{prefix}_last": float(finite[-1]),
        f"{prefix}_slope": _slope(raw),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_q90": float(np.quantile(finite, 0.9)),
        f"{prefix}_positive_fraction": float(np.mean(finite > 0.0)),
        f"{prefix}_finite_fraction": float(finite.size / raw.size),
    }


def _components(adjacency: NDArray[np.bool_]) -> list[tuple[int, ...]]:
    remaining = set(range(adjacency.shape[0]))
    result: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        found: set[int] = set()
        while stack:
            item = stack.pop()
            if item in found:
                continue
            found.add(item)
            stack.extend(int(v) for v in np.flatnonzero(adjacency[item]) if int(v) not in found)
        remaining.difference_update(found)
        result.append(tuple(sorted(found)))
    return result


def _pairwise_cosine(compositions: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(compositions, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("composition has nonpositive cosine norm")
    normalized = compositions / norms[:, None]
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def _recurrence_features(compositions: NDArray[np.float64], generations: NDArray[np.int64]) -> dict[str, float]:
    n = compositions.shape[0]
    similarity = _pairwise_cosine(compositions)
    ii, jj = np.indices((n, n))
    nonadjacent = jj < (ii - 1)
    comparable = int(np.count_nonzero(nonadjacent))
    recurrent = nonadjacent & (similarity >= RECURRENCE_THRESHOLD)
    max_prior = np.full(n, np.nan, dtype=np.float64)
    recurrence_lags: list[int] = []
    for i in range(2, n):
        max_prior[i] = float(np.max(similarity[i, : i - 1]))
        js = np.flatnonzero(recurrent[i, :i])
        recurrence_lags.extend(int(i - j) for j in js)
    graph = recurrent | recurrent.T | np.eye(n, dtype=bool)
    components = _components(graph)

    boundary_indices = np.flatnonzero(np.r_[True, np.diff(generations) > 0])
    if boundary_indices.size >= 3:
        boundary_similarity = similarity[np.ix_(boundary_indices, boundary_indices)]
        bi, bj = np.indices(boundary_similarity.shape)
        boundary_pairs = bj < (bi - 1)
        boundary_hits = boundary_pairs & (boundary_similarity >= RECURRENCE_THRESHOLD)
        boundary_density = float(np.count_nonzero(boundary_hits) / max(1, np.count_nonzero(boundary_pairs)))
        boundary_recurrent = np.zeros(boundary_indices.size, dtype=bool)
        for i in range(2, boundary_indices.size):
            boundary_recurrent[i] = bool(np.any(boundary_hits[i, :i]))
        boundary_fraction = float(np.mean(boundary_recurrent))
    else:
        boundary_density = 0.0
        boundary_fraction = 0.0

    finite_max = max_prior[np.isfinite(max_prior)]
    return {
        "nonadjacent_recurrence_edge_density": float(np.count_nonzero(recurrent) / max(1, comparable)),
        "mean_max_prior_nonadjacent_h": float(np.mean(finite_max)),
        "last_max_prior_nonadjacent_h": float(finite_max[-1]),
        "slope_max_prior_nonadjacent_h": _slope(max_prior),
        "recurrent_state_fraction": float(np.mean(max_prior >= RECURRENCE_THRESHOLD)),
        "largest_recurrence_component_fraction": float(max(map(len, components)) / n),
        "recurrence_component_count": float(len(components)),
        "maximum_recurrence_lag_fraction": float(max(recurrence_lags, default=0) / max(1, n - 1)),
        "post_fission_recurrence_edge_density": boundary_density,
        "post_fission_recurrent_state_fraction": boundary_fraction,
    }


def _organization_features(compositions: NDArray[np.float64]) -> dict[str, float]:
    eps = np.finfo(np.float64).tiny
    entropy = -np.sum(np.where(compositions > 0.0, compositions * np.log(np.maximum(compositions, eps)), 0.0), axis=1)
    effective = np.exp(entropy)
    richness = np.sum(compositions > 0.0, axis=1).astype(np.float64)
    simpson = np.sum(compositions * compositions, axis=1)
    centroid = np.mean(compositions, axis=0)
    radius = np.linalg.norm(compositions - centroid, axis=1)
    centered = compositions - centroid
    singular = np.linalg.svd(centered, compute_uv=False)
    eigen = singular * singular
    participation = float(np.square(np.sum(eigen)) / np.sum(eigen * eigen)) if np.sum(eigen * eigen) > 0 else 0.0
    first_pc = float(eigen[0] / np.sum(eigen)) if np.sum(eigen) > 0 else 0.0
    velocity = np.diff(compositions, axis=0)
    acceleration = np.diff(velocity, axis=0)
    directional: list[float] = []
    for left, right in zip(velocity[:-1], velocity[1:], strict=True):
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator > 0:
            directional.append(float(np.dot(left, right) / denominator))
    return {
        "shannon_mean": float(np.mean(entropy)),
        "shannon_last": float(entropy[-1]),
        "shannon_std": float(np.std(entropy)),
        "shannon_slope": _slope(entropy),
        "effective_species_mean": float(np.mean(effective)),
        "effective_species_last": float(effective[-1]),
        "richness_mean": float(np.mean(richness)),
        "richness_last": float(richness[-1]),
        "simpson_concentration_mean": float(np.mean(simpson)),
        "simpson_concentration_last": float(simpson[-1]),
        "centroid_radius_mean": float(np.mean(radius)),
        "centroid_radius_last": float(radius[-1]),
        "centroid_radius_slope": _slope(radius),
        "participation_ratio": participation,
        "first_pc_variance_fraction": first_pc,
        "acceleration_mean": float(np.mean(np.linalg.norm(acceleration, axis=1))) if acceleration.size else 0.0,
        "directional_persistence_mean": float(np.mean(directional)) if directional else 0.0,
    }


def extract_past_features(states: NDArray[np.integer[Any]], generations: Sequence[int], kinds: Sequence[str]) -> dict[str, float]:
    """Extract all non-Phi features from exactly the first 64 observations."""

    counts = np.asarray(states, dtype=np.float64)
    if counts.shape != (LANDMARK_COUNT, 100) or np.any(counts < 0.0):
        raise ValueError("past feature input must be 64-by-100 nonnegative counts")
    masses = counts.sum(axis=1)
    if np.any(masses <= 0.0):
        raise ValueError("past prefix contains an empty state")
    compositions = counts / masses[:, None]
    generation_values = np.asarray(generations, dtype=np.int64)
    if generation_values.shape != (LANDMARK_COUNT,) or len(kinds) != LANDMARK_COUNT:
        raise ValueError("prefix metadata cardinality mismatch")
    similarity = _pairwise_cosine(compositions)
    adjacent_h = np.diag(similarity, k=-1)
    change = np.linalg.norm(np.diff(compositions, axis=0), axis=1)
    time = {
        "prefix_generation_last": float(generation_values[-1]),
        "prefix_post_fission_count": float(sum(kind == "post_fission" for kind in kinds)),
        "prefix_molecular_update_count": float(sum(kind == "molecular_update" for kind in kinds)),
        "prefix_mass_last": float(masses[-1]),
    }
    stability = {
        "adjacent_h_last": float(adjacent_h[-1]),
        "adjacent_h_mean": float(np.mean(adjacent_h)),
        "adjacent_h_std": float(np.std(adjacent_h)),
        "adjacent_h_slope": _slope(adjacent_h),
        "adjacent_h_min": float(np.min(adjacent_h)),
        "adjacent_h_ge_090_fraction": float(np.mean(adjacent_h >= RECURRENCE_THRESHOLD)),
        "composition_change_last": float(change[-1]),
        "composition_change_mean": float(np.mean(change)),
        "composition_change_std": float(np.std(change)),
        "composition_change_slope": _slope(change),
        "composition_change_max": float(np.max(change)),
        "mass_mean": float(np.mean(masses)),
        "mass_std": float(np.std(masses)),
        "mass_slope": _slope(masses),
    }
    return time | stability | _recurrence_features(compositions, generation_values) | _organization_features(compositions)


FEATURE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "TIME_ONLY": (
        "prefix_generation_last", "prefix_post_fission_count", "prefix_molecular_update_count", "prefix_mass_last",
    ),
    "EXACT_H_STABILITY": (
        "adjacent_h_last", "adjacent_h_mean", "adjacent_h_std", "adjacent_h_slope", "adjacent_h_min",
        "adjacent_h_ge_090_fraction", "composition_change_last", "composition_change_mean", "composition_change_std",
        "composition_change_slope", "composition_change_max", "mass_mean", "mass_std", "mass_slope",
    ),
    "PREFIX_RECURRENCE_GEOMETRY": (
        "nonadjacent_recurrence_edge_density", "mean_max_prior_nonadjacent_h", "last_max_prior_nonadjacent_h",
        "slope_max_prior_nonadjacent_h", "recurrent_state_fraction", "largest_recurrence_component_fraction",
        "recurrence_component_count", "maximum_recurrence_lag_fraction", "post_fission_recurrence_edge_density",
        "post_fission_recurrent_state_fraction",
    ),
    "ORGANIZATION_DYNAMICS": (
        "shannon_mean", "shannon_last", "shannon_std", "shannon_slope", "effective_species_mean",
        "effective_species_last", "richness_mean", "richness_last", "simpson_concentration_mean",
        "simpson_concentration_last", "centroid_radius_mean", "centroid_radius_last", "centroid_radius_slope",
        "participation_ratio", "first_pc_variance_fraction", "acceleration_mean", "directional_persistence_mean",
    ),
}
