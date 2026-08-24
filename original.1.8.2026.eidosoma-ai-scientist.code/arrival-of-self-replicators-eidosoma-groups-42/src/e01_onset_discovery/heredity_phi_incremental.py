"""Fixed PhiID feature summaries and binomial models for S19-L45."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class MetricSummary:
    current: float
    recent_slope: float
    finite_fraction: float
    observations: int


def metric_summary(values: NDArray[np.floating], recent: int = 8) -> MetricSummary:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(array)
    if not len(array):
        return MetricSummary(float("nan"), float("nan"), float("nan"), 0)
    current = float(array[-1]) if finite[-1] else float("nan")
    tail = array[-recent:]
    tail_finite = np.isfinite(tail)
    if tail_finite.sum() >= 3:
        x = np.arange(len(tail), dtype=np.float64)[tail_finite]
        slope = float(np.polyfit(x, tail[tail_finite], 1)[0])
    else:
        slope = float("nan")
    return MetricSummary(
        current=current,
        recent_slope=slope,
        finite_fraction=float(finite.mean()),
        observations=len(array),
    )


def composition_controls(states: NDArray[np.integer]) -> dict[str, float]:
    counts = np.asarray(states, dtype=np.float64)
    if counts.ndim != 2 or len(counts) < 2:
        raise ValueError("composition controls require at least two state rows")
    masses = counts.sum(axis=1)
    if np.any(masses <= 0):
        raise ValueError("composition control states require positive mass")
    composition = counts / masses[:, None]
    left = composition[-2]
    right = composition[-1]
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    adjacent_h = (
        float(np.dot(left, right) / denominator) if denominator else float("nan")
    )
    return {
        "currentAdjacentMolecularH": adjacent_h,
        "currentCompositionChange": float(np.linalg.norm(right - left)),
    }


def fit_binomial_ridge(
    features: NDArray[np.floating],
    successes: NDArray[np.integer],
    trials: NDArray[np.integer],
    *,
    seed: int,
    c: float = 1.0,
) -> Pipeline:
    x = np.asarray(features, dtype=np.float64)
    k = np.asarray(successes, dtype=np.int64)
    n = np.asarray(trials, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(k) or len(k) != len(n):
        raise ValueError("binomial feature and outcome shapes do not align")
    if np.any(k < 0) or np.any(n <= 0) or np.any(k > n):
        raise ValueError("invalid binomial successes/trials")
    expanded_x = np.repeat(x, 2, axis=0)
    expanded_y = np.tile(np.array([1, 0], dtype=np.int64), len(x))
    weights = np.column_stack((k, n - k)).reshape(-1).astype(np.float64)
    positive_weight = weights > 0
    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c,
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight=None,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(
        expanded_x[positive_weight],
        expanded_y[positive_weight],
        model__sample_weight=weights[positive_weight],
    )
    return pipeline


def predict_probability(
    model: Pipeline, features: NDArray[np.floating]
) -> NDArray[np.float64]:
    probability = model.predict_proba(np.asarray(features, dtype=np.float64))[:, 1]
    return np.asarray(probability, dtype=np.float64)


def probability_metrics(
    q_hat: NDArray[np.floating],
    probability: NDArray[np.floating],
    successes: NDArray[np.integer],
    trials: NDArray[np.integer],
) -> dict[str, float]:
    q = np.asarray(q_hat, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    k = np.asarray(successes, dtype=np.float64)
    n = np.asarray(trials, dtype=np.float64)
    if not (len(q) == len(p) == len(k) == len(n)):
        raise ValueError("metric arrays do not align")
    log_loss = -(q * np.log(p) + (1 - q) * np.log(1 - p))
    brier = (q - p) ** 2
    if len(q) >= 3 and len(np.unique(q)) > 1 and len(np.unique(p)) > 1:
        rank = float(spearmanr(q, p).statistic)
        slope, intercept = np.polyfit(p, q, 1)
    else:
        rank = slope = intercept = float("nan")
    saturated = np.clip(q, 1e-12, 1 - 1e-12)
    deviance = 2 * (
        k * np.log(saturated / p) + (n - k) * np.log((1 - saturated) / (1 - p))
    )
    return {
        "qBrier": float(np.mean(brier)),
        "matrixBinomialLogLoss": float(np.mean(log_loss)),
        "meanBinomialDeviance": float(np.mean(deviance)),
        "spearman": rank,
        "calibrationIntercept": float(intercept),
        "calibrationSlope": float(slope),
        "meanPredictedProbability": float(np.mean(p)),
        "meanObservedQ": float(np.mean(q)),
    }
