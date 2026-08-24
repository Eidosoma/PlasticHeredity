"""Prefix-only multiscale geometry and topology for S19-L20.

The registered input is exactly 64 selected-clock molecular compositions.  H0
features use the float64 minimum-spanning-tree identity, H1 features use a
Vietoris--Rips filtration through GUDHI, intrinsic dimension uses the fixed
Levina--Bickel nearest-neighbour likelihood construction, and path features
describe only the observed prefix order.  No completed-run reference enters.
"""

from __future__ import annotations

from typing import Any

import gudhi
import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

LANDMARK_COUNT = 64
ID_NEIGHBOURS = (5, 10)
DISTANCE_EPSILON = 1e-12
PERSISTENCE_EPSILON = 1e-12


def _entropy(weights: NDArray[np.float64]) -> float:
    values = np.asarray(weights, dtype=np.float64)
    values = values[np.isfinite(values) & (values > PERSISTENCE_EPSILON)]
    if values.size == 0 or float(np.sum(values)) <= 0.0:
        return 0.0
    probability = values / np.sum(values)
    return float(-np.sum(probability * np.log(probability)))


def _closed(states: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
    counts = np.asarray(states, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != 100 or np.any(counts < 0.0):
        raise ValueError("states must be an n-by-100 nonnegative count matrix")
    mass = np.sum(counts, axis=1)
    if np.any(mass <= 0.0):
        raise ValueError("prefix contains an empty molecular state")
    return counts / mass[:, None]


def chord_distance_matrix(compositions: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(compositions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("composition has nonpositive norm")
    unit = values / norms[:, None]
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    distance = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - cosine)))
    np.fill_diagonal(distance, 0.0)
    return distance


def _persistent_segment(compositions: NDArray[np.float64]) -> dict[str, float]:
    distance = chord_distance_matrix(compositions)
    mst = minimum_spanning_tree(distance).toarray()
    h0 = mst[mst > PERSISTENCE_EPSILON].astype(np.float64)
    maximum = float(np.max(distance)) if distance.size else 0.0
    complex_ = gudhi.RipsComplex(
        distance_matrix=distance.tolist(), max_edge_length=maximum
    )
    simplex_tree = complex_.create_simplex_tree(max_dimension=2)
    simplex_tree.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    h1_intervals = simplex_tree.persistence_intervals_in_dimension(1)
    if len(h1_intervals):
        h1 = np.asarray(h1_intervals[:, 1] - h1_intervals[:, 0], dtype=np.float64)
        h1 = h1[np.isfinite(h1) & (h1 > PERSISTENCE_EPSILON)]
    else:
        h1 = np.empty(0, dtype=np.float64)
    return {
        "h0_total_persistence": float(np.sum(h0)),
        "h0_max_persistence": float(np.max(h0)) if h0.size else 0.0,
        "h0_persistence_entropy": _entropy(h0),
        "h1_feature_count": float(h1.size),
        "h1_total_persistence": float(np.sum(h1)),
        "h1_max_persistence": float(np.max(h1)) if h1.size else 0.0,
        "h1_persistence_entropy": _entropy(h1),
    }


def persistent_topology_features(
    compositions: NDArray[np.float64],
) -> dict[str, float]:
    full = _persistent_segment(compositions)
    early = _persistent_segment(compositions[: LANDMARK_COUNT // 2])
    late = _persistent_segment(compositions[LANDMARK_COUNT // 2 :])
    result = {f"topo_{name}_full": value for name, value in full.items()}
    for name in (
        "h0_total_persistence",
        "h0_max_persistence",
        "h1_feature_count",
        "h1_total_persistence",
        "h1_max_persistence",
    ):
        result[f"topo_{name}_late_minus_early"] = late[name] - early[name]
    return result


def _global_intrinsic_dimension(
    distance: NDArray[np.float64], k: int
) -> float:
    with_diagonal_hidden = distance.copy()
    np.fill_diagonal(with_diagonal_hidden, np.inf)
    ordered = np.sort(with_diagonal_hidden, axis=1)
    neighbours = np.maximum(ordered[:, :k], DISTANCE_EPSILON)
    radius = neighbours[:, -1]
    log_ratio = np.log(radius[:, None] / neighbours[:, :-1])
    denominator = float(np.sum(log_ratio))
    numerator = float(len(distance) * (k - 2))
    return numerator / denominator if denominator > DISTANCE_EPSILON else 0.0


def _intrinsic_segment(compositions: NDArray[np.float64]) -> dict[str, float]:
    distance = squareform(pdist(compositions, metric="euclidean"))
    with_diagonal_hidden = distance.copy()
    np.fill_diagonal(with_diagonal_hidden, np.inf)
    ordered = np.sort(with_diagonal_hidden, axis=1)
    nearest = ordered[:, 0]
    k5 = ordered[:, 4]
    k10 = ordered[:, 9]
    return {
        "mle_dimension_k5": _global_intrinsic_dimension(distance, 5),
        "mle_dimension_k10": _global_intrinsic_dimension(distance, 10),
        "nearest_distance_mean": float(np.mean(nearest)),
        "nearest_distance_cv": float(np.std(nearest) / max(np.mean(nearest), DISTANCE_EPSILON)),
        "k5_radius_mean": float(np.mean(k5)),
        "k10_radius_mean": float(np.mean(k10)),
        "pairwise_distance_cv": float(
            np.std(distance[np.triu_indices(len(distance), 1)])
            / max(
                np.mean(distance[np.triu_indices(len(distance), 1)]),
                DISTANCE_EPSILON,
            )
        ),
    }


def intrinsic_geometry_features(
    compositions: NDArray[np.float64],
) -> dict[str, float]:
    full = _intrinsic_segment(compositions)
    early = _intrinsic_segment(compositions[: LANDMARK_COUNT // 2])
    late = _intrinsic_segment(compositions[LANDMARK_COUNT // 2 :])
    result = {f"geom_{name}_full": value for name, value in full.items()}
    for name in (
        "mle_dimension_k5",
        "mle_dimension_k10",
        "nearest_distance_mean",
        "k5_radius_mean",
    ):
        result[f"geom_{name}_late_minus_early"] = late[name] - early[name]
    return result


def _path_segment(compositions: NDArray[np.float64]) -> dict[str, float]:
    differences = np.diff(compositions, axis=0)
    steps = np.linalg.norm(differences, axis=1)
    path_length = float(np.sum(steps))
    displacement = float(np.linalg.norm(compositions[-1] - compositions[0]))
    if len(differences) >= 2:
        left = differences[:-1]
        right = differences[1:]
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        turning = np.divide(
            np.sum(left * right, axis=1),
            denominator,
            out=np.zeros(len(denominator), dtype=np.float64),
            where=denominator > DISTANCE_EPSILON,
        )
    else:
        turning = np.zeros(1, dtype=np.float64)
    time = np.arange(len(steps), dtype=np.float64)
    slope = (
        float(np.polyfit(time, steps, 1)[0])
        if len(steps) >= 2 and np.std(steps) > DISTANCE_EPSILON
        else 0.0
    )
    result = {
        "step_mean": float(np.mean(steps)),
        "step_std": float(np.std(steps)),
        "step_cv": float(np.std(steps) / max(np.mean(steps), DISTANCE_EPSILON)),
        "step_slope": slope,
        "net_displacement": displacement,
        "tortuosity": path_length / max(displacement, DISTANCE_EPSILON),
        "turning_cosine_mean": float(np.mean(turning)),
        "turning_cosine_std": float(np.std(turning)),
    }
    mean_step = max(float(np.mean(steps)), DISTANCE_EPSILON)
    for lag in (2, 4, 8):
        displacement_lag = np.linalg.norm(
            compositions[lag:] - compositions[:-lag], axis=1
        )
        result[f"displacement_ratio_lag{lag}"] = float(
            np.mean(displacement_lag) / (lag * mean_step)
        )
    return result


def path_geometry_features(
    compositions: NDArray[np.float64],
) -> dict[str, float]:
    full = _path_segment(compositions)
    early = _path_segment(compositions[: LANDMARK_COUNT // 2])
    late = _path_segment(compositions[LANDMARK_COUNT // 2 :])
    result = {f"path_{name}_full": value for name, value in full.items()}
    for name in (
        "step_mean",
        "tortuosity",
        "turning_cosine_mean",
        "displacement_ratio_lag4",
    ):
        result[f"path_{name}_late_minus_early"] = late[name] - early[name]
    return result


def extract_multiscale_geometry_features(
    states: NDArray[np.integer[Any]],
) -> dict[str, float]:
    counts = np.asarray(states)
    if counts.shape != (LANDMARK_COUNT, 100):
        raise ValueError("multiscale input must be 64-by-100 counts")
    compositions = _closed(counts)
    result = persistent_topology_features(compositions)
    result.update(intrinsic_geometry_features(compositions))
    result.update(path_geometry_features(compositions))
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("multiscale feature extraction emitted a nonfinite value")
    return result


TOPOLOGY_FEATURES = tuple(
    [
        f"topo_{name}_full"
        for name in (
            "h0_total_persistence",
            "h0_max_persistence",
            "h0_persistence_entropy",
            "h1_feature_count",
            "h1_total_persistence",
            "h1_max_persistence",
            "h1_persistence_entropy",
        )
    ]
    + [
        f"topo_{name}_late_minus_early"
        for name in (
            "h0_total_persistence",
            "h0_max_persistence",
            "h1_feature_count",
            "h1_total_persistence",
            "h1_max_persistence",
        )
    ]
)

INTRINSIC_GEOMETRY_FEATURES = tuple(
    [
        f"geom_{name}_full"
        for name in (
            "mle_dimension_k5",
            "mle_dimension_k10",
            "nearest_distance_mean",
            "nearest_distance_cv",
            "k5_radius_mean",
            "k10_radius_mean",
            "pairwise_distance_cv",
        )
    ]
    + [
        f"geom_{name}_late_minus_early"
        for name in (
            "mle_dimension_k5",
            "mle_dimension_k10",
            "nearest_distance_mean",
            "k5_radius_mean",
        )
    ]
)

PATH_GEOMETRY_FEATURES = tuple(
    [
        f"path_{name}_full"
        for name in (
            "step_mean",
            "step_std",
            "step_cv",
            "step_slope",
            "net_displacement",
            "tortuosity",
            "turning_cosine_mean",
            "turning_cosine_std",
            "displacement_ratio_lag2",
            "displacement_ratio_lag4",
            "displacement_ratio_lag8",
        )
    ]
    + [
        f"path_{name}_late_minus_early"
        for name in (
            "step_mean",
            "tortuosity",
            "turning_cosine_mean",
            "displacement_ratio_lag4",
        )
    ]
)
