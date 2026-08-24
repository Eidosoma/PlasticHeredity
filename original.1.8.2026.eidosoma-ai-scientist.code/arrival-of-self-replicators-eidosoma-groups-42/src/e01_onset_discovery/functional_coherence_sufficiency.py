"""Fixed regression utilities for the S19-L47 coherence-sufficiency audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RidgeState:
    medians: NDArray[np.float64]
    means: NDArray[np.float64]
    scales: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    intercept: float
    alpha: float


def fit_ridge(
    features: NDArray[np.floating],
    target: NDArray[np.floating],
    *,
    alpha: float = 1.0,
) -> RidgeState:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 3:
        raise ValueError("ridge inputs must be aligned with at least three rows")
    if alpha <= 0 or not np.isfinite(y).all():
        raise ValueError("alpha and target must be finite")
    medians = np.nanmedian(x, axis=0)
    filled = np.where(np.isfinite(x), x, medians)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0, ddof=0)
    scales = np.where(scales > 0, scales, 1.0)
    z = (filled - means) / scales
    y_mean = float(y.mean())
    coefficients = np.linalg.solve(
        z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y_mean)
    )
    return RidgeState(
        medians=medians,
        means=means,
        scales=scales,
        coefficients=coefficients,
        intercept=y_mean,
        alpha=float(alpha),
    )


def predict_ridge(
    state: RidgeState, features: NDArray[np.floating]
) -> NDArray[np.float64]:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(state.coefficients):
        raise ValueError("prediction features do not match ridge state")
    filled = np.where(np.isfinite(x), x, state.medians)
    return state.intercept + ((filled - state.means) / state.scales) @ state.coefficients


def regression_metrics(
    observed: NDArray[np.floating], predicted: NDArray[np.floating]
) -> dict[str, float]:
    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() < 3:
        return {
            "rmse": float("nan"),
            "rSquared": float("nan"),
            "residualMean": float("nan"),
        }
    y = y[mask]
    p = p[mask]
    residual = y - p
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "rSquared": float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 0
        else float("nan"),
        "residualMean": float(np.mean(residual)),
    }
