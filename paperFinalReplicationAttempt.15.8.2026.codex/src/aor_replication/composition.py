"""Compositional-data transforms used before information analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def relative_composition(counts: ArrayLike, pseudocount: float = 0.0) -> FloatArray:
    """Convert one or more count vectors to closed compositions.

    A positive pseudocount is required when the result will be log transformed.
    """

    values = np.asarray(counts, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError("counts must be a one- or two-dimensional array")
    if np.any(values < 0):
        raise ValueError("counts cannot be negative")
    adjusted = values + pseudocount
    totals = adjusted.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("each composition must contain at least one molecule")
    return adjusted / totals


def clr_transform(
    counts: ArrayLike,
    *,
    pseudocount: float = 0.5,
    drop_last: bool = True,
) -> FloatArray:
    """Apply the centered log-ratio transform to count trajectories."""

    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive for CLR")
    composition = relative_composition(counts, pseudocount=pseudocount)
    log_values = np.log(composition)
    transformed = log_values - log_values.mean(axis=-1, keepdims=True)
    if drop_last:
        transformed = transformed[..., :-1]
    return np.asarray(transformed, dtype=float)


def cosine_similarity(reference: ArrayLike, values: ArrayLike) -> FloatArray:
    """Cosine/H similarity between a reference composition and rows of values."""

    ref = np.asarray(reference, dtype=float)
    arr = np.asarray(values, dtype=float)
    if ref.ndim != 1:
        raise ValueError("reference must be one-dimensional")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != ref.size:
        raise ValueError("values must have the same component count as reference")
    ref_norm = np.linalg.norm(ref)
    row_norm = np.linalg.norm(arr, axis=1)
    denominator = ref_norm * row_norm
    result = np.zeros(arr.shape[0], dtype=float)
    valid = denominator > 0
    result[valid] = np.einsum("ij,j->i", arr[valid], ref) / denominator[valid]
    return np.clip(result, -1.0, 1.0)
