"""Frozen statistical summaries for the E01-S12B source-code audit."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import linregress, rankdata, spearmanr
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass(frozen=True)
class CorrelationSummary:
    trajectory_correlations: dict[str, float | None]
    defined_count: int
    positive_count: int
    median: float | None
    bootstrap_lower: float | None
    bootstrap_upper: float | None
    circular_positive_p: float | None
    circular_negative_p: float | None


def finite_spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    a = np.asarray(list(x), dtype=np.float64)
    b = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.unique(a[mask]).size < 2 or np.unique(b[mask]).size < 2:
        return None
    value = float(spearmanr(a[mask], b[mask]).statistic)
    return value if np.isfinite(value) else None


def association_summary(
    frame: pd.DataFrame,
    *,
    value_column: str,
    label_column: str,
    bootstrap_seed: int,
    circular_seed: int,
    replicates: int = 4096,
) -> CorrelationSummary:
    correlations: dict[str, float | None] = {}
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for trajectory_id, group in frame.groupby("trajectoryId", sort=True):
        group = group.sort_values("generation")
        values = group[value_column].to_numpy(dtype=np.float64)
        labels = group[label_column].astype(float).to_numpy(dtype=np.float64)
        mask = np.isfinite(values) & np.isfinite(labels)
        values, labels = values[mask], labels[mask]
        correlations[str(trajectory_id)] = finite_spearman(values, labels)
        arrays[str(trajectory_id)] = (values, labels)
    defined = np.asarray([value for value in correlations.values() if value is not None], dtype=np.float64)
    if defined.size == 0:
        return CorrelationSummary(correlations, 0, 0, None, None, None, None, None)
    observed = float(np.median(defined))
    bootstrap_rng = np.random.RandomState(bootstrap_seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        bootstrap[index] = np.median(defined[bootstrap_rng.randint(0, defined.size, size=defined.size)])
    lower, upper = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    eligible_arrays = [item for trajectory_id, item in arrays.items() if correlations[trajectory_id] is not None]
    circular_rng = np.random.RandomState(circular_seed)
    null = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        shifted: list[float] = []
        for values, labels in eligible_arrays:
            offset = int(circular_rng.randint(1, labels.size))
            correlation = finite_spearman(values, np.roll(labels, offset))
            if correlation is not None:
                shifted.append(correlation)
        null[index] = np.median(shifted) if shifted else np.nan
    finite_null = null[np.isfinite(null)]
    positive_p = (1.0 + float(np.sum(finite_null >= observed))) / (replicates + 1.0)
    negative_p = (1.0 + float(np.sum(finite_null <= observed))) / (replicates + 1.0)
    return CorrelationSummary(
        correlations,
        int(defined.size),
        int(np.sum(defined > 0.0)),
        observed,
        float(lower),
        float(upper),
        positive_p,
        negative_p,
    )


def retrospective_coherent(summary: CorrelationSummary, *, finite_coverage: float, runs_higher: int) -> tuple[bool, dict[str, bool]]:
    gates = {
        "finiteCoverageAtLeast0p80": finite_coverage >= 0.80,
        "definedTrajectoryCorrelationsExactly12": summary.defined_count == 12,
        "positiveTrajectoryCorrelationsAtLeast9": summary.positive_count >= 9,
        "medianStrictlyPositive": summary.median is not None and summary.median > 0.0,
        "bootstrapLowerStrictlyPositive": summary.bootstrap_lower is not None and summary.bootstrap_lower > 0.0,
        "positiveCircularShiftPAtMost0p05": summary.circular_positive_p is not None and summary.circular_positive_p <= 0.05,
        "runsHigherDuringReplicationAtLeast7": runs_higher >= 7,
    }
    return all(gates.values()), gates


def prospective_candidate(
    summary: CorrelationSummary,
    *,
    coverage: float,
    replay_passed: bool,
    suffix_passed: bool,
    other_opposite: bool,
) -> tuple[bool, dict[str, bool]]:
    gates = {
        "eligibleCoverageAtLeast0p80": coverage >= 0.80,
        "definedTrajectoryCorrelationsExactly12": summary.defined_count == 12,
        "positiveTrajectoryCorrelationsAtLeast9": summary.positive_count >= 9,
        "medianStrictlyPositive": summary.median is not None and summary.median > 0.0,
        "bootstrapLowerStrictlyPositive": summary.bootstrap_lower is not None and summary.bootstrap_lower > 0.0,
        "positiveCircularShiftPAtMost0p05": summary.circular_positive_p is not None and summary.circular_positive_p <= 0.05,
        "exactFutureSuffixInvariance": suffix_passed,
        "exactSourceReplay": replay_passed,
        "noSignificantOppositeOtherImplementation": not other_opposite,
    }
    return all(gates.values()), gates


def significant_opposite(summary: CorrelationSummary) -> bool:
    return bool(
        summary.median is not None
        and summary.median < 0.0
        and summary.bootstrap_upper is not None
        and summary.bootstrap_upper < 0.0
        and summary.circular_negative_p is not None
        and summary.circular_negative_p <= 0.05
    )


def spike_thresholds(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return {"mean": np.nan, "sampleSd": np.nan, "median": np.nan, "mad": np.nan, "positive3Sigma": np.nan, "negative3Sigma": np.nan, "robustPositive": np.nan}
    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1))
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return {
        "mean": mean,
        "sampleSd": sd,
        "median": median,
        "mad": mad,
        "positive3Sigma": mean + 3.0 * sd,
        "negative3Sigma": mean - 3.0 * sd,
        "robustPositive": median + 3.0 * 1.4826 * mad,
    }


def molecular_progress_trend(frame: pd.DataFrame) -> dict[str, float | int | None]:
    grid = np.linspace(0.0, 1.0, 1001)
    aligned: list[np.ndarray] = []
    for _, group in frame.groupby("trajectoryId", sort=True):
        group = group[np.isfinite(group["phiR"])].sort_values("molecularStep")
        if len(group) < 2:
            continue
        steps = group["molecularStep"].to_numpy(dtype=float)
        if steps[-1] <= steps[0]:
            continue
        progress = (steps - steps[0]) / (steps[-1] - steps[0])
        aligned.append(np.interp(grid, progress, group["phiR"].to_numpy(dtype=float)))
    if not aligned:
        return {"trajectoryCount": 0, "slope": None, "intercept": None, "rValue": None, "pValue": None, "standardError": None}
    mean_curve = np.mean(np.vstack(aligned), axis=0)
    fit = linregress(grid, mean_curve)
    return {"trajectoryCount": len(aligned), "slope": float(fit.slope), "intercept": float(fit.intercept), "rValue": float(fit.rvalue), "pValue": float(fit.pvalue), "standardError": float(fit.stderr)}


def ljung_box_summary(frame: pd.DataFrame) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    for trajectory_id, group in frame.groupby("trajectoryId", sort=True):
        values = group.loc[np.isfinite(group["phiR"]), "phiR"].to_numpy(dtype=float)
        lag = min(20, values.size // 5)
        if lag < 1:
            rows.append({"trajectoryId": str(trajectory_id), "n": int(values.size), "lag": lag, "statistic": None, "pValue": None, "significantAt0p05": None})
            continue
        result = acorr_ljungbox(values, lags=[lag], return_df=True)
        rows.append({"trajectoryId": str(trajectory_id), "n": int(values.size), "lag": lag, "statistic": float(result.iloc[0]["lb_stat"]), "pValue": float(result.iloc[0]["lb_pvalue"]), "significantAt0p05": bool(result.iloc[0]["lb_pvalue"] <= 0.05)})
    return rows


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    return (rankdata(values, method="average") - 0.5) / values.size
