"""Fixed feature-selection helpers for the S19-L53 regime-capacity audit."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def beta_structure_indices(feature_names: Sequence[str]) -> tuple[int, ...]:
    """Select only graph coordinates determined by beta, not the current state."""

    prefixes = ("beta_log_", "beta_raw_", "beta_singular_")
    selected = tuple(
        index for index, name in enumerate(feature_names) if str(name).startswith(prefixes)
    )
    if len(selected) != 20:
        raise ValueError(f"expected 20 frozen beta-only coordinates, observed {len(selected)}")
    return selected


def center_within_groups(
    values: NDArray[np.floating], groups: Sequence[object]
) -> NDArray[np.float64]:
    """Center one value per state within its catalytic-matrix group."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    labels = np.asarray(tuple(groups), dtype=object).reshape(-1)
    if len(array) != len(labels) or not np.isfinite(array).all():
        raise ValueError("group-centering inputs must align and be finite")
    output = np.empty_like(array)
    for label in np.unique(labels):
        mask = labels == label
        output[mask] = array[mask] - array[mask].mean()
    return output


def binomial_cell_scores(
    probability: NDArray[np.floating],
    successes: NDArray[np.integer],
    trials: NDArray[np.integer],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-state branch log loss and q-scale Brier score."""

    p = np.clip(np.asarray(probability, dtype=np.float64).reshape(-1), 1e-12, 1 - 1e-12)
    k = np.asarray(successes, dtype=np.int64).reshape(-1)
    n = np.asarray(trials, dtype=np.int64).reshape(-1)
    if not (len(p) == len(k) == len(n)) or np.any(n <= 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError("invalid binomial score inputs")
    log_loss = -(k * np.log(p) + (n - k) * np.log1p(-p)) / n
    q = (k + 0.5) / (n + 1.0)
    brier = (q - p) ** 2
    return log_loss, brier
