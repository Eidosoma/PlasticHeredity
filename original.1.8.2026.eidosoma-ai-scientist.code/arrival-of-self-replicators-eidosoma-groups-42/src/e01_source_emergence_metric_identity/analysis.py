"""Frozen statistical methods for E01 S12D.

All confirmatory decisions are trajectory-level.  Pooled observation tests are
retained only as paper-like diagnostics and never control a classification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.stats import linregress, mannwhitneyu, pearsonr, rankdata, spearmanr
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass(frozen=True)
class TrajectoryAssociationSummary:
    correlations: dict[str, float | None]
    ordinary_p_values: dict[str, float | None]
    defined_count: int
    positive_count: int
    ordinary_positive_p_lt_0p05_count: int
    mean: float | None
    median: float | None
    bootstrap_lower_95: float | None
    bootstrap_upper_95: float | None
    circular_shift_positive_p: float | None
    circular_shift_negative_p: float | None
    effective_episode_count: int
    median_lag_one_autocorrelation: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DifferenceSummary:
    mean_differences: dict[str, float | None]
    median_differences: dict[str, float | None]
    defined_count: int
    positive_count: int
    median_mean_difference: float | None
    median_median_difference: float | None
    bootstrap_lower_95: float | None
    bootstrap_upper_95: float | None
    block_aware_positive_p: float | None
    pooled_mann_whitney_u: float | None
    pooled_mann_whitney_p: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_pair(
    values: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(values) & np.isfinite(labels)
    return values[mask], labels[mask]


def _ordinary_spearman(
    values: np.ndarray, labels: np.ndarray
) -> tuple[float | None, float | None]:
    values, labels = _finite_pair(values, labels)
    if values.size < 3 or np.unique(values).size < 2 or np.unique(labels).size < 2:
        return None, None
    result = spearmanr(values, labels)
    rho, p_value = float(result.statistic), float(result.pvalue)
    if not np.isfinite(rho) or not np.isfinite(p_value):
        return None, None
    return rho, p_value


def _lag_one_autocorrelation(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    if values.size < 4 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return None
    value = float(pearsonr(values[:-1], values[1:]).statistic)
    return value if np.isfinite(value) else None


def _episode_count(labels: np.ndarray) -> int:
    finite = labels[np.isfinite(labels)].astype(bool)
    if finite.size == 0:
        return 0
    return int(finite[0]) + int(np.sum(finite[1:] & ~finite[:-1]))


def trajectory_association_summary(
    frame: pd.DataFrame,
    *,
    value_column: str,
    label_column: str,
    bootstrap_seed: int,
    circular_seed: int,
    replicates: int = 4096,
) -> TrajectoryAssociationSummary:
    """Summarize within-trajectory correlations with a trajectory bootstrap."""

    correlations: dict[str, float | None] = {}
    ordinary_ps: dict[str, float | None] = {}
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    episodes = 0
    autocorrelations: list[float] = []
    for trajectory_id, group in frame.groupby("trajectoryId", sort=True):
        group = group.sort_values(["generation", "rawObservationIndex"])
        values = group[value_column].to_numpy(dtype=np.float64)
        labels = group[label_column].astype(float).to_numpy(dtype=np.float64)
        values, labels = _finite_pair(values, labels)
        rho, p_value = _ordinary_spearman(values, labels)
        correlations[str(trajectory_id)] = rho
        ordinary_ps[str(trajectory_id)] = p_value
        arrays[str(trajectory_id)] = (values, labels)
        episodes += _episode_count(labels)
        autocorrelation = _lag_one_autocorrelation(values)
        if autocorrelation is not None:
            autocorrelations.append(autocorrelation)

    defined = np.asarray(
        [value for value in correlations.values() if value is not None]
    )
    if defined.size == 0:
        return TrajectoryAssociationSummary(
            correlations,
            ordinary_ps,
            0,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            episodes,
            None,
        )
    bootstrap_rng = np.random.RandomState(bootstrap_seed)
    bootstrap = np.asarray(
        [
            np.median(
                defined[bootstrap_rng.randint(0, defined.size, size=defined.size)]
            )
            for _ in range(replicates)
        ],
        dtype=np.float64,
    )
    lower, upper = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    eligible_arrays: list[tuple[np.ndarray, np.ndarray, float]] = []
    for key, value in correlations.items():
        if value is None:
            continue
        values, labels = arrays[key]
        ranked_values = rankdata(values, method="average")
        ranked_labels = rankdata(labels, method="average")
        ranked_values -= ranked_values.mean()
        ranked_labels -= ranked_labels.mean()
        denominator = float(
            np.linalg.norm(ranked_values) * np.linalg.norm(ranked_labels)
        )
        eligible_arrays.append((ranked_values, ranked_labels, denominator))
    circular_rng = np.random.RandomState(circular_seed)
    null = np.full(replicates, np.nan, dtype=np.float64)
    for index in range(replicates):
        shifted: list[float] = []
        for ranked_values, ranked_labels, denominator in eligible_arrays:
            if ranked_labels.size < 2 or denominator == 0:
                continue
            offset = int(circular_rng.randint(1, ranked_labels.size))
            correlation = float(
                np.dot(ranked_values, np.roll(ranked_labels, offset)) / denominator
            )
            if np.isfinite(correlation):
                shifted.append(correlation)
        if shifted:
            null[index] = np.median(shifted)
    finite_null = null[np.isfinite(null)]
    observed = float(np.median(defined))
    denominator = finite_null.size + 1.0
    positive_p = (1.0 + float(np.sum(finite_null >= observed))) / denominator
    negative_p = (1.0 + float(np.sum(finite_null <= observed))) / denominator
    return TrajectoryAssociationSummary(
        correlations=correlations,
        ordinary_p_values=ordinary_ps,
        defined_count=int(defined.size),
        positive_count=int(np.sum(defined > 0)),
        ordinary_positive_p_lt_0p05_count=sum(
            correlation is not None
            and correlation > 0
            and ordinary_ps[key] is not None
            and ordinary_ps[key] < 0.05
            for key, correlation in correlations.items()
        ),
        mean=float(np.mean(defined)),
        median=observed,
        bootstrap_lower_95=float(lower),
        bootstrap_upper_95=float(upper),
        circular_shift_positive_p=float(positive_p),
        circular_shift_negative_p=float(negative_p),
        effective_episode_count=episodes,
        median_lag_one_autocorrelation=(
            float(np.median(autocorrelations)) if autocorrelations else None
        ),
    )


def replicator_drift_summary(
    frame: pd.DataFrame,
    *,
    value_column: str,
    label_column: str,
    bootstrap_seed: int,
    permutation_seed: int,
    replicates: int = 4096,
) -> DifferenceSummary:
    """Trajectory-level replicator-minus-drift differences and circular null."""

    mean_differences: dict[str, float | None] = {}
    median_differences: dict[str, float | None] = {}
    arrays: list[tuple[np.ndarray, np.ndarray]] = []
    pooled_rep: list[float] = []
    pooled_drift: list[float] = []
    for trajectory_id, group in frame.groupby("trajectoryId", sort=True):
        group = group.sort_values(["generation", "rawObservationIndex"])
        values, labels = _finite_pair(
            group[value_column].to_numpy(dtype=np.float64),
            group[label_column].astype(float).to_numpy(dtype=np.float64),
        )
        rep, drift = values[labels == 1], values[labels == 0]
        if rep.size and drift.size:
            mean_differences[str(trajectory_id)] = float(np.mean(rep) - np.mean(drift))
            median_differences[str(trajectory_id)] = float(
                np.median(rep) - np.median(drift)
            )
            arrays.append((values, labels))
            pooled_rep.extend(rep.tolist())
            pooled_drift.extend(drift.tolist())
        else:
            mean_differences[str(trajectory_id)] = None
            median_differences[str(trajectory_id)] = None

    defined = np.asarray([x for x in mean_differences.values() if x is not None])
    defined_median = np.asarray(
        [x for x in median_differences.values() if x is not None]
    )
    if defined.size == 0:
        return DifferenceSummary(
            mean_differences,
            median_differences,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    bootstrap_rng = np.random.RandomState(bootstrap_seed)
    bootstrap = np.asarray(
        [
            np.median(
                defined[bootstrap_rng.randint(0, defined.size, size=defined.size)]
            )
            for _ in range(replicates)
        ]
    )
    lower, upper = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    permutation_rng = np.random.RandomState(permutation_seed)
    null = np.full(replicates, np.nan)
    for index in range(replicates):
        shifted_differences: list[float] = []
        for values, labels in arrays:
            offset = int(permutation_rng.randint(1, labels.size))
            shifted = np.roll(labels, offset)
            rep, drift = values[shifted == 1], values[shifted == 0]
            if rep.size and drift.size:
                shifted_differences.append(float(np.mean(rep) - np.mean(drift)))
        if shifted_differences:
            null[index] = np.median(shifted_differences)
    finite_null = null[np.isfinite(null)]
    observed = float(np.median(defined))
    positive_p = (1.0 + float(np.sum(finite_null >= observed))) / (
        finite_null.size + 1.0
    )
    if pooled_rep and pooled_drift:
        mw = mannwhitneyu(pooled_rep, pooled_drift, alternative="two-sided")
        u_stat, u_p = float(mw.statistic), float(mw.pvalue)
    else:
        u_stat, u_p = None, None
    return DifferenceSummary(
        mean_differences=mean_differences,
        median_differences=median_differences,
        defined_count=int(defined.size),
        positive_count=int(np.sum(defined > 0)),
        median_mean_difference=observed,
        median_median_difference=float(np.median(defined_median)),
        bootstrap_lower_95=float(lower),
        bootstrap_upper_95=float(upper),
        block_aware_positive_p=float(positive_p),
        pooled_mann_whitney_u=u_stat,
        pooled_mann_whitney_p=u_p,
    )


def excursion_thresholds(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return {
            key: np.nan
            for key in (
                "mean",
                "sampleSd",
                "median",
                "mad",
                "positive3Sigma",
                "negative3Sigma",
                "robustPositive",
                "robustNegative",
            )
        }
    mean, sd = float(np.mean(finite)), float(np.std(finite, ddof=1))
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_scale = 1.4826 * mad
    return {
        "mean": mean,
        "sampleSd": sd,
        "median": median,
        "mad": mad,
        "positive3Sigma": mean + 3 * sd,
        "negative3Sigma": mean - 3 * sd,
        "robustPositive": median + 3 * robust_scale,
        "robustNegative": median - 3 * robust_scale,
    }


def temporal_structure_rows(
    frame: pd.DataFrame, *, value_column: str
) -> list[dict[str, Any]]:
    """Per-run spike shape and raw/differenced Ljung-Box diagnostics."""

    rows: list[dict[str, Any]] = []
    aligned: list[np.ndarray] = []
    grid = np.linspace(0.0, 1.0, 1001)
    for trajectory_id, group in frame.groupby("trajectoryId", sort=True):
        group = group.sort_values("rawObservationIndex")
        values = group[value_column].to_numpy(dtype=np.float64)
        steps = group["molecularStep"].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        clean = values[finite]
        clean_steps = steps[finite]
        thresholds = excursion_thresholds(clean)
        if clean.size >= 2 and clean_steps[-1] > clean_steps[0]:
            progress = (clean_steps - clean_steps[0]) / (
                clean_steps[-1] - clean_steps[0]
            )
            aligned.append(np.interp(grid, progress, clean))
        positive_indices = np.flatnonzero(
            finite & (values > thresholds["positive3Sigma"])
        )
        negative_indices = np.flatnonzero(
            finite & (values < thresholds["negative3Sigma"])
        )
        robust_positive = np.flatnonzero(
            finite & (values > thresholds["robustPositive"])
        )
        robust_negative = np.flatnonzero(
            finite & (values < thresholds["robustNegative"])
        )
        if clean.size >= 3:
            peaks, _ = find_peaks(clean)
            prominences = (
                peak_prominences(clean, peaks)[0] if peaks.size else np.asarray([])
            )
            widths = (
                peak_widths(clean, peaks, rel_height=0.5)[0]
                if peaks.size
                else np.asarray([])
            )
            spacing = np.diff(peaks) if peaks.size >= 2 else np.asarray([])
        else:
            peaks = prominences = widths = spacing = np.asarray([])
        lag = min(20, clean.size // 5)
        diff = np.diff(clean)
        diff_lag = min(20, diff.size // 5)
        if lag >= 1:
            lb = acorr_ljungbox(clean, lags=[lag], return_df=True).iloc[0]
            raw_stat, raw_p = float(lb["lb_stat"]), float(lb["lb_pvalue"])
        else:
            raw_stat = raw_p = None
        if diff_lag >= 1:
            dlb = acorr_ljungbox(diff, lags=[diff_lag], return_df=True).iloc[0]
            diff_stat, diff_p = float(dlb["lb_stat"]), float(dlb["lb_pvalue"])
        else:
            diff_stat = diff_p = None
        rows.append(
            {
                "rowType": "TRAJECTORY",
                "trajectoryId": str(trajectory_id),
                "nFinite": int(clean.size),
                "positive3SigmaCount": int(positive_indices.size),
                "negative3SigmaCount": int(negative_indices.size),
                "robustPositiveCount": int(robust_positive.size),
                "robustNegativeCount": int(robust_negative.size),
                "peakCount": int(peaks.size),
                "medianPeakWidthObservations": float(np.median(widths))
                if widths.size
                else None,
                "medianPeakProminence": float(np.median(prominences))
                if prominences.size
                else None,
                "medianPeakSpacingObservations": float(np.median(spacing))
                if spacing.size
                else None,
                "ljungBoxLag": lag,
                "ljungBoxStatistic": raw_stat,
                "ljungBoxPValue": raw_p,
                "differencedLjungBoxLag": diff_lag,
                "differencedLjungBoxStatistic": diff_stat,
                "differencedLjungBoxPValue": diff_p,
                **thresholds,
            }
        )
    if aligned:
        curve = np.mean(np.vstack(aligned), axis=0)
        trend = linregress(grid, curve)
        slope, p_value = float(trend.slope), float(trend.pvalue)
    else:
        slope = p_value = None
    rows.append(
        {
            "rowType": "AGGREGATE",
            "trajectoryId": None,
            "nFinite": int(sum(row["nFinite"] for row in rows)),
            "runsWithPositive3Sigma": int(
                sum(row["positive3SigmaCount"] > 0 for row in rows)
            ),
            "runsRawLjungBoxSignificant": int(
                sum(
                    row["ljungBoxPValue"] is not None and row["ljungBoxPValue"] <= 0.05
                    for row in rows
                )
            ),
            "runsDifferencedLjungBoxSignificant": int(
                sum(
                    row["differencedLjungBoxPValue"] is not None
                    and row["differencedLjungBoxPValue"] <= 0.05
                    for row in rows
                )
            ),
            "aggregateTrendSlope": slope,
            "aggregateTrendPValue": p_value,
        }
    )
    return rows


def finite_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = _finite_pair(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    value = float(pearsonr(x, y).statistic)
    return value if np.isfinite(value) else None


def rank_agreement(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    x, y = _finite_pair(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if x.size < 2:
        return None, None
    rx = (rankdata(x, method="average") - 0.5) / x.size
    ry = (rankdata(y, method="average") - 0.5) / y.size
    return float(np.mean(1.0 - np.abs(rx - ry))), float(np.mean(np.abs(rx - ry) > 0.10))


def association_gate(
    summary: TrajectoryAssociationSummary, *, confirmation: bool
) -> dict[str, bool]:
    return {
        "definedTrajectoryCount": summary.defined_count >= (20 if confirmation else 12),
        "positiveTrajectoryCount": summary.positive_count
        >= (18 if confirmation else 9),
        "medianStrictlyPositive": summary.median is not None and summary.median > 0,
        "bootstrapLowerStrictlyPositive": summary.bootstrap_lower_95 is not None
        and summary.bootstrap_lower_95 > 0,
        "circularShiftPositivePAtMost0p05": summary.circular_shift_positive_p
        is not None
        and summary.circular_shift_positive_p <= 0.05,
    }


def drift_gate(summary: DifferenceSummary, *, confirmation: bool) -> dict[str, bool]:
    return {
        "definedTrajectoryCount": summary.defined_count >= (20 if confirmation else 12),
        "positiveTrajectoryCount": summary.positive_count
        >= (14 if confirmation else 7),
        "medianStrictlyPositive": summary.median_mean_difference is not None
        and summary.median_mean_difference > 0,
        "bootstrapLowerStrictlyPositive": summary.bootstrap_lower_95 is not None
        and summary.bootstrap_lower_95 > 0,
        "blockAwarePositivePAtMost0p05": summary.block_aware_positive_p is not None
        and summary.block_aware_positive_p <= 0.05,
    }


def significant_opposite(summary: TrajectoryAssociationSummary) -> bool:
    return bool(
        summary.median is not None
        and summary.median < 0
        and summary.bootstrap_upper_95 is not None
        and summary.bootstrap_upper_95 < 0
        and summary.circular_shift_negative_p is not None
        and summary.circular_shift_negative_p <= 0.05
    )
