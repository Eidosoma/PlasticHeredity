from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .rng import generator


def binomial_log_loss(success: np.ndarray, total: np.ndarray | float, probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(success, dtype=np.float64)
    n = np.asarray(total, dtype=np.float64)
    return -(y * np.log(p) + (n - y) * np.log1p(-p)) / np.maximum(n, 1.0)


def binomial_brier(success: np.ndarray, total: np.ndarray | float, probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(success, dtype=np.float64)
    n = np.asarray(total, dtype=np.float64)
    rate = y / np.maximum(n, 1.0)
    return rate * (1.0 - p) ** 2 + (1.0 - rate) * p**2


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else 0.0


def bootstrap_mean(values: np.ndarray, repetitions: int, master: str, domain: str, adjusted_tests: int = 1) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return {"estimate": math.nan, "lower": math.nan, "upper": math.nan, "adjusted_lower": math.nan}
    rng = generator(master, "bootstrap", domain)
    draws = np.empty(repetitions, dtype=np.float64)
    chunk = 256
    for start in range(0, repetitions, chunk):
        count = min(chunk, repetitions - start)
        indices = rng.integers(0, len(data), size=(count, len(data)))
        draws[start : start + count] = data[indices].mean(axis=1)
    alpha = 0.05
    return {
        "estimate": float(data.mean()),
        "lower": float(np.quantile(draws, alpha / 2.0)),
        "upper": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "adjusted_lower": float(np.quantile(draws, alpha / (2.0 * max(adjusted_tests, 1)))),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def split_half_reliability(half0_rate: np.ndarray, half1_rate: np.ndarray) -> dict[str, float]:
    correlation = safe_spearman(half0_rate, half1_rate)
    q = 2.0 * correlation / (1.0 + correlation) if correlation > -1.0 else -1.0
    return {"split_half_spearman": correlation, "spearman_brown_q": float(np.clip(q, -1.0, 1.0))}


def prediction_correlations(prediction: np.ndarray, observed_rate: np.ndarray, cue_index: np.ndarray) -> dict[str, float]:
    within = [safe_spearman(prediction[index], observed_rate[index]) for index in range(prediction.shape[0])]
    return {
        "overall": safe_spearman(prediction, observed_rate),
        "median_within_network": float(np.median(within)),
        "cue_A": safe_spearman(prediction[cue_index == 0], observed_rate[cue_index == 0]),
        "cue_B": safe_spearman(prediction[cue_index == 1], observed_rate[cue_index == 1]),
    }


def permutation_tests(
    event_half0: np.ndarray,
    event_half1: np.ndarray,
    half_total: float,
    full_prediction: np.ndarray,
    history_prediction: np.ndarray,
    repetitions: int,
    master: str,
    tier: str,
) -> dict[str, Any]:
    observed: dict[str, float] = {}
    for half, events in ((0, event_half0), (1, event_half1)):
        observed[f"logloss_half{half}"] = float(np.mean(
            binomial_log_loss(events, half_total, history_prediction)
            - binomial_log_loss(events, half_total, full_prediction)
        ))
        observed[f"brier_half{half}"] = float(np.mean(
            binomial_brier(events, half_total, history_prediction)
            - binomial_brier(events, half_total, full_prediction)
        ))
    exceed = {name: 0 for name in observed}
    rng = generator(master, "whole-network-permutation", tier)
    networks = event_half0.shape[0]
    for _ in range(repetitions):
        order = rng.permutation(networks)
        for half, events in ((0, event_half0[order]), (1, event_half1[order])):
            values = {
                f"logloss_half{half}": float(np.mean(
                    binomial_log_loss(events, half_total, history_prediction)
                    - binomial_log_loss(events, half_total, full_prediction)
                )),
                f"brier_half{half}": float(np.mean(
                    binomial_brier(events, half_total, history_prediction)
                    - binomial_brier(events, half_total, full_prediction)
                )),
            }
            for name, value in values.items():
                exceed[name] += int(value >= observed[name])
    raw = {name: (count + 1.0) / (repetitions + 1.0) for name, count in exceed.items()}
    return {"observed": observed, "raw_p": raw, "holm_p": holm_adjust(raw), "repetitions": repetitions}

