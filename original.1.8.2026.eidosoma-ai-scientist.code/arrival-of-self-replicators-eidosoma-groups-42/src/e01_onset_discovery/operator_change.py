"""Past-only operator-change features for the S19-L25 online task."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from e01_onset_discovery.outcome_blind_representation import (
    CHANNEL_NAMES,
    organization_channel_sequence,
)

WINDOW_COUNT = 64
HALF_COUNT = 32
OPERATOR_RIDGE = 1e-3


def _ar1(values: NDArray[np.float64]) -> float:
    left = values[:-1]
    right = values[1:]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _operator(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], float, float, float, float]:
    left = values[:-1]
    right = values[1:]
    gram = left.T @ left
    ridge = OPERATOR_RIDGE * max(float(np.trace(gram)) / len(gram), 1.0)
    operator = np.linalg.solve(gram + ridge * np.eye(len(gram)), left.T @ right)
    prediction = left @ operator
    residual = float(np.mean(np.square(right - prediction)))
    eigen = np.linalg.eigvals(operator)
    radius = float(np.max(np.abs(eigen))) if eigen.size else 0.0
    leading = float(np.linalg.svd(operator, compute_uv=False)[0])
    commutator = operator.T @ operator - operator @ operator.T
    norm = float(np.linalg.norm(operator))
    nonnormality = float(np.linalg.norm(commutator) / max(norm * norm, 1e-15))
    return operator, residual, radius, leading, nonnormality


def _energy_distance(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    cross = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    within_left = np.linalg.norm(left[:, None, :] - left[None, :, :], axis=2)
    within_right = np.linalg.norm(right[:, None, :] - right[None, :, :], axis=2)
    return float(2.0 * np.mean(cross) - np.mean(within_left) - np.mean(within_right))


def extract_operator_change_features(states: NDArray[np.integer[Any]]) -> dict[str, float]:
    counts = np.asarray(states, dtype=np.int64)
    if counts.shape != (WINDOW_COUNT, 100):
        raise ValueError("operator-change window must be 64-by-100")
    channels = organization_channel_sequence(counts)
    reference = channels[:HALF_COUNT]
    recent = channels[HALF_COUNT:]
    reference_mean = np.mean(reference, axis=0)
    reference_scale = np.std(reference, axis=0)
    reference_scale = np.where(reference_scale > 1e-8, reference_scale, 1.0)
    standardized_reference = (reference - reference_mean) / reference_scale
    standardized_recent = (recent - reference_mean) / reference_scale

    result: dict[str, float] = {}
    for index, name in enumerate(CHANNEL_NAMES):
        left = standardized_reference[:, index]
        right = standardized_recent[:, index]
        result[f"mean_shift__{name}"] = float(np.mean(right) - np.mean(left))
        result[f"log_variance_ratio__{name}"] = float(
            np.log((np.var(right) + 1e-8) / (np.var(left) + 1e-8))
        )
        result[f"ar1_shift__{name}"] = _ar1(right) - _ar1(left)

    covariance_reference = np.cov(standardized_reference, rowvar=False, ddof=1)
    covariance_recent = np.cov(standardized_recent, rowvar=False, ddof=1)
    covariance_delta = covariance_recent - covariance_reference
    reference_operator = _operator(standardized_reference)
    recent_operator = _operator(standardized_recent)
    operator_delta = recent_operator[0] - reference_operator[0]
    reference_speed = np.linalg.norm(np.diff(standardized_reference, axis=0), axis=1)
    recent_speed = np.linalg.norm(np.diff(standardized_recent, axis=0), axis=1)
    result.update(
        {
            "energy_distance": _energy_distance(standardized_reference, standardized_recent),
            "mean_shift_l2": float(np.linalg.norm(np.mean(standardized_recent, axis=0))),
            "covariance_delta_frobenius": float(np.linalg.norm(covariance_delta)),
            "covariance_delta_spectral": float(np.linalg.norm(covariance_delta, ord=2)),
            "operator_delta_frobenius": float(np.linalg.norm(operator_delta)),
            "operator_delta_spectral": float(np.linalg.norm(operator_delta, ord=2)),
            "operator_radius_shift": recent_operator[2] - reference_operator[2],
            "operator_leading_singular_shift": recent_operator[3] - reference_operator[3],
            "operator_nonnormality_shift": recent_operator[4] - reference_operator[4],
            "operator_residual_log_ratio": float(
                np.log((recent_operator[1] + 1e-12) / (reference_operator[1] + 1e-12))
            ),
            "reference_speed_mean": float(np.mean(reference_speed)),
            "recent_speed_mean": float(np.mean(recent_speed)),
            "speed_log_ratio": float(
                np.log((np.mean(recent_speed) + 1e-12) / (np.mean(reference_speed) + 1e-12))
            ),
        }
    )
    if not np.isfinite(list(result.values())).all():
        raise RuntimeError("operator-change features are nonfinite")
    return result


OPERATOR_CHANGE_FEATURES = tuple(
    extract_operator_change_features(np.ones((WINDOW_COUNT, 100), dtype=np.int64))
)
CHANNEL_SHIFT_FEATURES = tuple(
    name
    for channel in CHANNEL_NAMES
    for name in (
        f"mean_shift__{channel}",
        f"log_variance_ratio__{channel}",
        f"ar1_shift__{channel}",
    )
)
EXACT_H_CHANGE_FEATURES = tuple(
    name
    for channel in CHANNEL_NAMES[6:]
    for name in (
        f"mean_shift__{channel}",
        f"log_variance_ratio__{channel}",
        f"ar1_shift__{channel}",
    )
)
OPERATOR_ONLY_FEATURES = tuple(
    name for name in OPERATOR_CHANGE_FEATURES if name not in CHANNEL_SHIFT_FEATURES
)
