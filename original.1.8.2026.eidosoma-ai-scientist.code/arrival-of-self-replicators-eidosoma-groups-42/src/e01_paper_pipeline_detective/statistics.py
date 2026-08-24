"""Predeclared S12E association, spike, and temporal summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.stats import linregress, spearmanr
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass(frozen=True, slots=True)
class CohortInference:
    defined: int
    positive: int
    median: float | None
    mean: float | None
    bootstrap_low: float | None
    bootstrap_high: float | None
    circular_shift_positive_p: float | None


def safe_spearman(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return np.nan, np.nan
    if np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return np.nan, np.nan
    result = spearmanr(x[mask], y[mask])
    return float(result.statistic), float(result.pvalue)


def _ljung_box(values: NDArray[np.float64]) -> tuple[int | None, float | None, bool | None]:
    finite = values[np.isfinite(values)]
    if finite.size < 8:
        return None, None, None
    lag = max(1, min(10, finite.size // 5))
    try:
        result = acorr_ljungbox(finite, lags=[lag], return_df=True)
        p = float(result["lb_pvalue"].iloc[0])
        return lag, p, bool(p < 0.05)
    except Exception:  # noqa: BLE001 - diagnostic remains explicitly undefined.
        return lag, None, None


def trajectory_association(
    values: NDArray[np.float64], labels: NDArray[np.bool_]
) -> dict[str, object]:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    mask = np.isfinite(x)
    rho, p = safe_spearman(x[mask], y[mask].astype(float))
    replicate = x[mask & y]
    drift = x[mask & ~y]
    mean_difference = (
        float(replicate.mean() - drift.mean())
        if replicate.size and drift.size
        else np.nan
    )
    median_difference = (
        float(np.median(replicate) - np.median(drift))
        if replicate.size and drift.size
        else np.nan
    )
    if x.size >= 2:
        dx = np.diff(x)
        dy = y[1:].astype(float)
        difference_rho, difference_p = safe_spearman(dx, dy)
    else:
        difference_rho, difference_p = np.nan, np.nan
    return {
        "finiteCount": int(mask.sum()),
        "totalCount": int(x.size),
        "finiteCoverage": float(mask.mean()) if mask.size else 0.0,
        "rhoText": rho,
        "pTextOrdinary": p,
        "rhoDifferencedCaption": difference_rho,
        "pDifferencedCaption": difference_p,
        "replicatorMean": float(replicate.mean()) if replicate.size else np.nan,
        "driftMean": float(drift.mean()) if drift.size else np.nan,
        "meanDifference": mean_difference,
        "replicatorMedian": float(np.median(replicate)) if replicate.size else np.nan,
        "driftMedian": float(np.median(drift)) if drift.size else np.nan,
        "medianDifference": median_difference,
    }


def spike_summary(
    values: NDArray[np.float64],
    *,
    problematic: NDArray[np.bool_] | None = None,
) -> dict[str, object]:
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return {
            "positive3Sigma": 0,
            "negative3Sigma": 0,
            "robustPositive": 0,
            "robustNegative": 0,
            "punctuated": False,
            "medianPositiveWidth": np.nan,
            "medianPositiveProminence": np.nan,
            "medianPositiveSpacing": np.nan,
            "problematicSpikeFraction": np.nan,
        }
    filled = x.copy()
    median = float(np.nanmedian(filled))
    filled[~finite] = median
    mean = float(np.mean(filled))
    std = float(np.std(filled))
    mad = float(np.median(np.abs(filled - median)))
    robust_scale = 1.4826 * mad
    z = np.zeros_like(filled) if std == 0.0 else (filled - mean) / std
    rz = np.zeros_like(filled) if robust_scale == 0.0 else (filled - median) / robust_scale
    positive = np.flatnonzero(z > 3.0)
    negative = np.flatnonzero(z < -3.0)
    robust_positive = np.flatnonzero(rz > 3.0)
    robust_negative = np.flatnonzero(rz < -3.0)
    peaks, _ = find_peaks(filled, height=mean + 3.0 * std if std > 0 else np.inf)
    widths = peak_widths(filled, peaks)[0] if peaks.size else np.asarray([])
    prominences = peak_prominences(filled, peaks)[0] if peaks.size else np.asarray([])
    spacing = np.diff(peaks) if peaks.size >= 2 else np.asarray([])
    punctuated = bool((positive.size + negative.size) >= 1 and (robust_positive.size + robust_negative.size) >= 1)
    problem_fraction = np.nan
    if problematic is not None and (positive.size + negative.size):
        bad = np.asarray(problematic, dtype=bool)
        expanded = bad.copy()
        expanded[1:] |= bad[:-1]
        expanded[:-1] |= bad[1:]
        excursions = np.unique(np.concatenate((positive, negative)))
        problem_fraction = float(np.mean(expanded[excursions]))
    return {
        "positive3Sigma": int(positive.size),
        "negative3Sigma": int(negative.size),
        "robustPositive": int(robust_positive.size),
        "robustNegative": int(robust_negative.size),
        "punctuated": punctuated,
        "medianPositiveWidth": float(np.median(widths)) if widths.size else np.nan,
        "medianPositiveProminence": float(np.median(prominences)) if prominences.size else np.nan,
        "medianPositiveSpacing": float(np.median(spacing)) if spacing.size else np.nan,
        "problematicSpikeFraction": problem_fraction,
    }


def temporal_summary(values: NDArray[np.float64]) -> dict[str, object]:
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    slope = np.nan
    slope_p = np.nan
    if np.count_nonzero(finite) >= 3:
        regression = linregress(np.flatnonzero(finite), x[finite])
        slope = float(regression.slope)
        slope_p = float(regression.pvalue)
    lag, p, significant = _ljung_box(x)
    dlag, dp, dsignificant = _ljung_box(np.diff(x))
    return {
        "linearSlope": slope,
        "linearSlopeP": slope_p,
        "ljungBoxLag": lag,
        "ljungBoxP": p,
        "ljungBoxSignificant": significant,
        "differencedLjungBoxLag": dlag,
        "differencedLjungBoxP": dp,
        "differencedLjungBoxSignificant": dsignificant,
        "lag1Autocorrelation": (
            float(np.corrcoef(x[:-1], x[1:])[0, 1])
            if x.size >= 3 and np.all(np.isfinite(x)) and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0
            else np.nan
        ),
    }


def cohort_inference(
    trajectory_values: list[NDArray[np.float64]],
    trajectory_labels: list[NDArray[np.bool_]],
    *,
    bootstrap_replicates: int,
    shift_replicates: int,
    bootstrap_rng: np.random.Generator,
    shift_rng: np.random.Generator,
) -> CohortInference:
    correlations = np.asarray(
        [safe_spearman(x, y.astype(float))[0] for x, y in zip(trajectory_values, trajectory_labels, strict=True)],
        dtype=np.float64,
    )
    defined = correlations[np.isfinite(correlations)]
    if defined.size == 0:
        return CohortInference(0, 0, None, None, None, None, None)
    boot = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        boot[index] = float(
            np.median(defined[bootstrap_rng.integers(0, defined.size, size=defined.size)])
        )
    observed = float(np.median(defined))
    null = np.empty(shift_replicates, dtype=np.float64)
    for replicate in range(shift_replicates):
        values: list[float] = []
        for x, y in zip(trajectory_values, trajectory_labels, strict=True):
            if y.size < 2:
                continue
            offset = int(shift_rng.integers(1, y.size))
            rho, _ = safe_spearman(x, np.roll(y, offset).astype(float))
            if np.isfinite(rho):
                values.append(rho)
        null[replicate] = float(np.median(values)) if values else np.nan
    finite_null = null[np.isfinite(null)]
    p = (
        float((1 + np.count_nonzero(finite_null >= observed)) / (1 + finite_null.size))
        if finite_null.size
        else None
    )
    return CohortInference(
        defined=int(defined.size),
        positive=int(np.count_nonzero(defined > 0.0)),
        median=observed,
        mean=float(np.mean(defined)),
        bootstrap_low=float(np.quantile(boot, 0.025)),
        bootstrap_high=float(np.quantile(boot, 0.975)),
        circular_shift_positive_p=p,
    )


def aggregate_molecular_slope(frames: list[pd.DataFrame]) -> tuple[float, float, int]:
    """Fit the paper-like aggregate median series at shared raw molecular steps."""

    if not frames:
        return np.nan, np.nan, 0
    concatenated = pd.concat(frames, ignore_index=True)
    grouped = (
        concatenated.dropna(subset=["value"])
        .groupby("molecularStep", as_index=False)["value"]
        .median()
        .sort_values("molecularStep")
    )
    if grouped.shape[0] < 3:
        return np.nan, np.nan, int(grouped.shape[0])
    result = linregress(grouped["molecularStep"].to_numpy(), grouped["value"].to_numpy())
    return float(result.slope), float(result.pvalue), int(grouped.shape[0])
