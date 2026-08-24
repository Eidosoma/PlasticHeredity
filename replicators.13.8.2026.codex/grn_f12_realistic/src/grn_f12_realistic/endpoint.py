from __future__ import annotations

import numpy as np


def transform_state(state: np.ndarray, tier: str, volume: float | np.ndarray = 1.0) -> np.ndarray:
    value = np.asarray(state)
    if tier == "continuous":
        clipped = np.clip(value.astype(np.float64), 1e-4, 1.0 - 1e-4)
        transformed = np.log(clipped) - np.log1p(-clipped)
    elif tier == "molecular":
        transformed = np.log1p(value.astype(np.float64) / np.asarray(volume)[..., None])
    else:
        raise ValueError(tier)
    return transformed - transformed.mean(axis=-1, keepdims=True)


def phenotype_similarity(left: np.ndarray, right: np.ndarray, tier: str) -> np.ndarray:
    a = transform_state(left, tier)
    b = transform_state(right, tier)
    numerator = np.sum(a * b, axis=-1)
    norm_a = np.sqrt(np.sum(a * a, axis=-1))
    norm_b = np.sqrt(np.sum(b * b, axis=-1))
    regular = (norm_a > 0) & (norm_b > 0)
    correlation = np.zeros(np.broadcast_shapes(numerator.shape, regular.shape), dtype=np.float64)
    np.divide(numerator, norm_a * norm_b, out=correlation, where=regular)
    similarity = np.clip((1.0 + correlation) / 2.0, 0.0, 1.0)
    degenerate = ~regular
    if np.any(degenerate):
        identical = np.all(np.asarray(left) == np.asarray(right), axis=-1)
        similarity = np.where(degenerate, identical.astype(np.float64), similarity)
    return similarity


def classify_f12(similarities: np.ndarray, threshold: float, run: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(similarities, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("similarities must be futures x generations")
    futures, horizon = values.shape
    broken = np.zeros(futures, dtype=bool)
    event = np.zeros(futures, dtype=bool)
    high_run = np.zeros(futures, dtype=np.int16)
    maximum_run = np.zeros(futures, dtype=np.int16)
    for generation in range(horizon):
        low = values[:, generation] <= threshold
        previously_broken = broken.copy()
        broken |= low
        high = values[:, generation] > threshold
        high_run = np.where(previously_broken & high, high_run + 1, 0)
        maximum_run = np.maximum(maximum_run, high_run)
        event |= high_run >= run
    return broken, event, maximum_run


def event_summary(similarities: np.ndarray, threshold: float) -> dict[str, np.ndarray | int | float]:
    broken, event, maximum_run = classify_f12(similarities, threshold, run=3)
    _, run5, _ = classify_f12(similarities, threshold, run=5)
    half = len(event) // 2
    return {
        "break": broken,
        "event": event,
        "run5": run5,
        "maximum_run": maximum_run,
        "break_count": int(broken.sum()),
        "event_count": int(event.sum()),
        "event_half0": int(event[:half].sum()),
        "event_half1": int(event[half:].sum()),
        "break_half0": int(broken[:half].sum()),
        "break_half1": int(broken[half:].sum()),
        "mean_similarity": float(np.mean(similarities)),
    }


def calibrated_threshold(per_network_similarities: list[np.ndarray], quantile: float) -> tuple[float, np.ndarray]:
    if not per_network_similarities:
        raise ValueError("calibration cohort is empty")
    percentiles = np.asarray([
        np.quantile(np.asarray(values, dtype=np.float64).ravel(), quantile)
        for values in per_network_similarities
    ])
    return float(np.median(percentiles)), percentiles

