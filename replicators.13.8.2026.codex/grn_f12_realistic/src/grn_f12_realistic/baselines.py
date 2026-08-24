from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .storage import load_npz, write_npz_atomic


@dataclass
class RidgeHead:
    coefficient: np.ndarray
    intercept: float
    constant: float
    is_constant: bool


@dataclass
class HurdleRidge:
    mean: np.ndarray
    scale: np.ndarray
    break_head: RidgeHead
    recovery_head: RidgeHead

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        transformed = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        break_probability = _predict_head(self.break_head, transformed)
        recovery_probability = _predict_head(self.recovery_head, transformed)
        return break_probability, recovery_probability, break_probability * recovery_probability

    def save(self, path: str | Path) -> None:
        write_npz_atomic(
            path,
            mean=self.mean,
            scale=self.scale,
            break_coefficient=self.break_head.coefficient,
            break_intercept=np.asarray(self.break_head.intercept),
            break_constant=np.asarray(self.break_head.constant),
            break_is_constant=np.asarray(self.break_head.is_constant, dtype=np.uint8),
            recovery_coefficient=self.recovery_head.coefficient,
            recovery_intercept=np.asarray(self.recovery_head.intercept),
            recovery_constant=np.asarray(self.recovery_head.constant),
            recovery_is_constant=np.asarray(self.recovery_head.is_constant, dtype=np.uint8),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HurdleRidge":
        values = load_npz(path)
        return cls(
            mean=values["mean"], scale=values["scale"],
            break_head=RidgeHead(
                values["break_coefficient"], float(values["break_intercept"]),
                float(values["break_constant"]), bool(values["break_is_constant"]),
            ),
            recovery_head=RidgeHead(
                values["recovery_coefficient"], float(values["recovery_intercept"]),
                float(values["recovery_constant"]), bool(values["recovery_is_constant"]),
            ),
        )


def _weighted_scaler(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    total = max(float(weights.sum()), 1.0)
    mean = np.sum(x * weights[:, None], axis=0) / total
    variance = np.sum((x - mean) ** 2 * weights[:, None], axis=0) / total
    scale = np.sqrt(np.maximum(variance, 1e-12))
    return mean, scale


def _fit_head(x: np.ndarray, success: np.ndarray, total: np.ndarray, c_value: float) -> RidgeHead:
    success = np.asarray(success, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)
    failures = total - success
    smoothed = float((success.sum() + 0.5) / (total.sum() + 1.0)) if total.sum() else 0.5
    if success.sum() <= 0 or failures.sum() <= 0:
        return RidgeHead(np.zeros(x.shape[1]), 0.0, smoothed, True)
    expanded_x = np.concatenate((x, x), axis=0)
    labels = np.concatenate((np.ones(len(x)), np.zeros(len(x))))
    weights = np.concatenate((success, failures))
    keep = weights > 0
    model = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=4000, random_state=0)
    model.fit(expanded_x[keep], labels[keep], sample_weight=weights[keep])
    return RidgeHead(model.coef_[0].astype(np.float64), float(model.intercept_[0]), smoothed, False)


def _predict_head(head: RidgeHead, x: np.ndarray) -> np.ndarray:
    if head.is_constant:
        return np.full(len(x), head.constant, dtype=np.float64)
    logits = x @ head.coefficient + head.intercept
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def fit_hurdle_ridge(x: np.ndarray, event_count: np.ndarray, break_count: np.ndarray, total: np.ndarray, c_value: float) -> HurdleRidge:
    values = np.asarray(x, dtype=np.float64)
    mean, scale = _weighted_scaler(values, np.asarray(total))
    transformed = (values - mean) / scale
    return HurdleRidge(
        mean=mean,
        scale=scale,
        break_head=_fit_head(transformed, break_count, total, c_value),
        recovery_head=_fit_head(transformed, event_count, break_count, c_value),
    )

