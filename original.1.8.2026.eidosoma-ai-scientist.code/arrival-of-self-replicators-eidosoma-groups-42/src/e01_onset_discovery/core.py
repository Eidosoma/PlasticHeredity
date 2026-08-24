"""Outcome-blind, prefix-only warning observables for S19-L19.

All values are deterministic functions of exactly 64 selected molecular-clock
compositions.  They implement three predeclared method families: classical
critical-slowing indicators, recurrence-plot line statistics, and local
linear/DMD relaxation diagnostics.  No completed-run target geometry is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

LANDMARK_COUNT = 64
RECURRENCE_H = 0.9
THEILER_WINDOW = 1
MIN_LINE = 2
DMD_MAX_RANK = 8
DMD_RELATIVE_RIDGE = 1e-6


def _safe_corr(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.size < 3 or b.size != a.size or np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _cosine_similarity(values: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("composition has nonpositive norm")
    normalized = values / norms[:, None]
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def _pc1_scores(
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    centered = values - np.mean(values, axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    scores = (
        u[:, 0] * singular[0]
        if singular.size
        else np.zeros(values.shape[0], dtype=np.float64)
    )
    eigen = singular * singular / max(1, values.shape[0] - 1)
    return scores, eigen


def _low_frequency_fraction(values: NDArray[np.float64]) -> float:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    power = np.abs(np.fft.rfft(centered)) ** 2
    if power.size <= 1 or float(np.sum(power[1:])) <= 0.0:
        return 0.0
    nonzero = power[1:]
    low_count = max(1, int(np.ceil(nonzero.size / 4)))
    return float(np.sum(nonzero[:low_count]) / np.sum(nonzero))


def _critical_segment(values: NDArray[np.float64]) -> dict[str, float]:
    scores, eigen = _pc1_scores(values)
    total = float(np.sum(eigen))
    active_ar = []
    for column in values.T:
        if np.std(column) > 1e-15:
            active_ar.append(_safe_corr(column[:-1], column[1:]))
    return {
        "pc1_ar1": _safe_corr(scores[:-1], scores[1:]),
        "mean_species_ar1": float(np.mean(active_ar)) if active_ar else 0.0,
        "cov_lambda1_fraction": float(eigen[0] / total) if total > 0.0 else 0.0,
        "log_total_variance": float(np.log(max(total, np.finfo(np.float64).tiny))),
        "pc1_low_frequency_fraction": _low_frequency_fraction(scores),
    }


def critical_slowing_features(values: NDArray[np.float64]) -> dict[str, float]:
    full = _critical_segment(values)
    early = _critical_segment(values[: LANDMARK_COUNT // 2])
    late = _critical_segment(values[LANDMARK_COUNT // 2 :])
    row = {f"ews_{name}_full": value for name, value in full.items()}
    row.update(
        {f"ews_{name}_late_minus_early": late[name] - early[name] for name in full}
    )
    return row


def _run_lengths(binary: NDArray[np.bool_]) -> list[int]:
    values = np.asarray(binary, dtype=bool)
    if values.size == 0:
        return []
    padded = np.r_[False, values, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        int(stop - start)
        for start, stop in zip(changes[::2], changes[1::2], strict=True)
    ]


def _rqa_segment(values: NDArray[np.float64]) -> dict[str, float]:
    similarity = _cosine_similarity(values)
    n = similarity.shape[0]
    ii, jj = np.indices((n, n))
    eligible = np.abs(ii - jj) > THEILER_WINDOW
    recurrence = (similarity >= RECURRENCE_H) & eligible
    recurrence_points = int(np.count_nonzero(recurrence))
    eligible_points = int(np.count_nonzero(eligible))

    diagonal_lengths: list[int] = []
    for offset in range(-(n - 1), n):
        if abs(offset) <= THEILER_WINDOW:
            continue
        diagonal_lengths.extend(_run_lengths(np.diag(recurrence, k=offset)))
    vertical_lengths: list[int] = []
    for column in range(n):
        vertical_lengths.extend(_run_lengths(recurrence[:, column]))
    diagonal_valid = np.asarray(
        [x for x in diagonal_lengths if x >= MIN_LINE], dtype=np.float64
    )
    vertical_valid = np.asarray(
        [x for x in vertical_lengths if x >= MIN_LINE], dtype=np.float64
    )
    diagonal_points = float(np.sum(diagonal_valid)) if diagonal_valid.size else 0.0
    vertical_points = float(np.sum(vertical_valid)) if vertical_valid.size else 0.0
    if diagonal_valid.size:
        _, counts = np.unique(diagonal_valid, return_counts=True)
        probability = counts / np.sum(counts)
        entropy = float(-np.sum(probability * np.log(probability)))
    else:
        entropy = 0.0
    return {
        "recurrence_rate": float(recurrence_points / max(1, eligible_points)),
        "determinism": float(diagonal_points / max(1, recurrence_points)),
        "mean_diagonal_length": float(np.mean(diagonal_valid))
        if diagonal_valid.size
        else 0.0,
        "max_diagonal_length_fraction": float(np.max(diagonal_valid) / n)
        if diagonal_valid.size
        else 0.0,
        "diagonal_entropy": entropy,
        "laminarity": float(vertical_points / max(1, recurrence_points)),
        "trapping_time": float(np.mean(vertical_valid)) if vertical_valid.size else 0.0,
        "max_vertical_length_fraction": float(np.max(vertical_valid) / n)
        if vertical_valid.size
        else 0.0,
    }


def recurrence_quantification_features(values: NDArray[np.float64]) -> dict[str, float]:
    full = _rqa_segment(values)
    early = _rqa_segment(values[: LANDMARK_COUNT // 2])
    late = _rqa_segment(values[LANDMARK_COUNT // 2 :])
    row = {f"rqa_{name}_full": value for name, value in full.items()}
    for name in ("recurrence_rate", "determinism", "laminarity", "trapping_time"):
        row[f"rqa_{name}_late_minus_early"] = late[name] - early[name]
    return row


def _dmd_segment(values: NDArray[np.float64]) -> dict[str, float]:
    centered = values - np.mean(values, axis=0, keepdims=True)
    x = centered[:-1].T
    y = centered[1:].T
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return {
            "spectral_radius": 0.0,
            "near_unit_fraction": 0.0,
            "relative_reconstruction_error": 0.0,
            "effective_rank": 0.0,
            "nonnormality": 0.0,
        }
    numerical_rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
    rank = max(1, min(DMD_MAX_RANK, numerical_rank))
    ur = u[:, :rank]
    sr = singular[:rank]
    vr = vt[:rank].T
    ridge = DMD_RELATIVE_RIDGE * float(sr[0] ** 2)
    inverse = sr / (sr * sr + ridge)
    reduced = ur.T @ y @ (vr * inverse[None, :])
    eigenvalues = np.linalg.eigvals(reduced)
    magnitudes = np.abs(eigenvalues)
    predicted = ur @ reduced @ (ur.T @ x)
    denominator = float(np.linalg.norm(y))
    reconstruction = (
        float(np.linalg.norm(y - predicted) / denominator) if denominator > 0.0 else 0.0
    )
    commutator = reduced.T @ reduced - reduced @ reduced.T
    norm = float(np.linalg.norm(reduced))
    nonnormality = (
        float(np.linalg.norm(commutator) / (norm * norm)) if norm > 0.0 else 0.0
    )
    weights = sr * sr
    effective_rank = (
        float(np.square(np.sum(weights)) / np.sum(weights * weights))
        if np.sum(weights * weights) > 0.0
        else 0.0
    )
    return {
        "spectral_radius": float(np.max(magnitudes)) if magnitudes.size else 0.0,
        "near_unit_fraction": float(np.mean(magnitudes >= 0.95))
        if magnitudes.size
        else 0.0,
        "relative_reconstruction_error": reconstruction,
        "effective_rank": effective_rank,
        "nonnormality": nonnormality,
    }


def dmd_features(values: NDArray[np.float64]) -> dict[str, float]:
    full = _dmd_segment(values)
    early = _dmd_segment(values[: LANDMARK_COUNT // 2])
    late = _dmd_segment(values[LANDMARK_COUNT // 2 :])
    row = {f"dmd_{name}_full": value for name, value in full.items()}
    for name in ("spectral_radius", "relative_reconstruction_error", "effective_rank"):
        row[f"dmd_{name}_late_minus_early"] = late[name] - early[name]
    return row


def extract_organization_warning_features(
    states: NDArray[np.integer[Any]],
) -> dict[str, float]:
    counts = np.asarray(states, dtype=np.float64)
    if counts.shape != (LANDMARK_COUNT, 100) or np.any(counts < 0.0):
        raise ValueError("warning-feature input must be 64-by-100 nonnegative counts")
    mass = np.sum(counts, axis=1)
    if np.any(mass <= 0.0):
        raise ValueError("warning-feature prefix contains an empty state")
    compositions = counts / mass[:, None]
    result = critical_slowing_features(compositions)
    result.update(recurrence_quantification_features(compositions))
    result.update(dmd_features(compositions))
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("warning-feature extraction emitted a nonfinite value")
    return result


EWS_FEATURES = tuple(
    [
        f"ews_{name}_full"
        for name in (
            "pc1_ar1",
            "mean_species_ar1",
            "cov_lambda1_fraction",
            "log_total_variance",
            "pc1_low_frequency_fraction",
        )
    ]
    + [
        f"ews_{name}_late_minus_early"
        for name in (
            "pc1_ar1",
            "mean_species_ar1",
            "cov_lambda1_fraction",
            "log_total_variance",
            "pc1_low_frequency_fraction",
        )
    ]
)

RQA_FEATURES = tuple(
    [
        f"rqa_{name}_full"
        for name in (
            "recurrence_rate",
            "determinism",
            "mean_diagonal_length",
            "max_diagonal_length_fraction",
            "diagonal_entropy",
            "laminarity",
            "trapping_time",
            "max_vertical_length_fraction",
        )
    ]
    + [
        f"rqa_{name}_late_minus_early"
        for name in (
            "recurrence_rate",
            "determinism",
            "laminarity",
            "trapping_time",
        )
    ]
)

DMD_FEATURES = tuple(
    [
        f"dmd_{name}_full"
        for name in (
            "spectral_radius",
            "near_unit_fraction",
            "relative_reconstruction_error",
            "effective_rank",
            "nonnormality",
        )
    ]
    + [
        f"dmd_{name}_late_minus_early"
        for name in (
            "spectral_radius",
            "relative_reconstruction_error",
            "effective_rank",
        )
    ]
)
