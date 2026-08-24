"""Statistical analyses corresponding to the preprint's main results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from .config import CausalConfig, ReplicatorConfig
from .gard import RunTrace
from .information import CausalTrajectory, fit_causal_trajectory
from .replicators import (
    ReplicatorMetrics,
    ReplicatorResult,
    detect_replicators,
    replicator_metrics,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _finite_pair(first: FloatArray, second: FloatArray) -> Tuple[FloatArray, FloatArray]:
    mask = np.isfinite(first) & np.isfinite(second)
    return first[mask], second[mask]


def safe_spearman(first: FloatArray, second: FloatArray) -> Tuple[float, float]:
    x, y = _finite_pair(np.asarray(first, float), np.asarray(second, float))
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _ljung_box_pvalue(values: FloatArray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 8 or np.std(finite) == 0:
        return float("nan")
    lag = max(1, min(10, finite.size // 5))
    result = acorr_ljungbox(finite, lags=[lag], return_df=True)
    return float(result["lb_pvalue"].iloc[-1])


@dataclass(frozen=True)
class AnalyzedRun:
    """Simulation, causal trajectory, labels, and per-run test results."""

    run_index: int
    treatment: str
    trace: RunTrace
    causal: CausalTrajectory
    replicator: ReplicatorResult
    aligned_labels: BoolArray
    metrics: ReplicatorMetrics
    spearman_rho: float
    spearman_p: float
    mann_whitney_p: float
    mean_phi_replicating: float
    mean_phi_drift: float
    ljung_box_p: float
    differenced_ljung_box_p: float
    spike_indices: NDArray[np.int64]

    def record(self) -> Dict[str, Any]:
        return {
            "run_index": self.run_index,
            "seed": self.trace.seed,
            "treatment": self.treatment,
            "molecular_steps": int(self.trace.counts.shape[0]),
            "phi_observations": int(self.causal.values.size),
            "phi_mean": float(np.mean(self.causal.values)),
            "phi_std": float(np.std(self.causal.values, ddof=1)),
            "phi_min": float(np.min(self.causal.values)),
            "phi_max": float(np.max(self.causal.values)),
            "partition_1_size": int((~self.causal.partition).sum()),
            "partition_2_size": int(self.causal.partition.sum()),
            "replicator_support": self.replicator.support,
            "persistence": self.metrics.persistence,
            "probability": self.metrics.probability,
            "consistency": self.metrics.consistency,
            "time_to_first": self.metrics.time_to_first,
            "spearman_rho": self.spearman_rho,
            "spearman_p": self.spearman_p,
            "mann_whitney_p": self.mann_whitney_p,
            "mean_phi_replicating": self.mean_phi_replicating,
            "mean_phi_drift": self.mean_phi_drift,
            "ljung_box_p": self.ljung_box_p,
            "differenced_ljung_box_p": self.differenced_ljung_box_p,
            "spike_count": int(self.spike_indices.size),
        }

    def generation_probabilities(self) -> FloatArray:
        probabilities = np.full(int(self.trace.generations.max()) + 1, np.nan)
        for generation in range(probabilities.size):
            selected = self.trace.generations == generation
            if selected.any():
                probabilities[generation] = self.replicator.labels[selected].mean()
        return probabilities


def analyze_run(
    trace: RunTrace,
    *,
    run_index: int,
    treatment: str,
    causal_config: CausalConfig,
    replicator_config: ReplicatorConfig,
) -> AnalyzedRun:
    """Compute causal-emergence and self-replication statistics for one trace."""

    causal = fit_causal_trajectory(trace.counts, causal_config)
    replicator = detect_replicators(trace, replicator_config)
    aligned_labels = replicator.labels[causal.time_indices]
    phi = causal.values
    rho, rho_p = safe_spearman(phi, aligned_labels.astype(float))

    replicating_phi = phi[aligned_labels]
    drift_phi = phi[~aligned_labels]
    if replicating_phi.size and drift_phi.size:
        mann = stats.mannwhitneyu(
            replicating_phi, drift_phi, alternative="greater", method="auto"
        )
        mann_p = float(mann.pvalue)
        mean_replicating = float(np.mean(replicating_phi))
        mean_drift = float(np.mean(drift_phi))
    else:
        mann_p = float("nan")
        mean_replicating = (
            float(np.mean(replicating_phi)) if replicating_phi.size else float("nan")
        )
        mean_drift = float(np.mean(drift_phi)) if drift_phi.size else float("nan")

    phi_std = float(np.std(phi, ddof=1))
    if phi_std > 0:
        # The text defines spikes as excursions more than three standard
        # deviations *above* the overall mean. Negative excursions remain in
        # the trajectory plots but are not counted as spike events.
        spikes = np.flatnonzero(phi > np.mean(phi) + 3 * phi_std)
    else:
        spikes = np.empty(0, dtype=np.int64)

    return AnalyzedRun(
        run_index=run_index,
        treatment=treatment,
        trace=trace,
        causal=causal,
        replicator=replicator,
        aligned_labels=aligned_labels,
        metrics=replicator_metrics(replicator.labels),
        spearman_rho=rho,
        spearman_p=rho_p,
        mann_whitney_p=mann_p,
        mean_phi_replicating=mean_replicating,
        mean_phi_drift=mean_drift,
        ljung_box_p=_ljung_box_pvalue(phi),
        differenced_ljung_box_p=_ljung_box_pvalue(np.diff(phi)),
        spike_indices=spikes,
    )


def records_frame(runs: Sequence[AnalyzedRun]) -> pd.DataFrame:
    return pd.DataFrame([run.record() for run in runs])


def _fisher_pvalue(pvalues: Iterable[float]) -> float:
    finite = np.asarray(list(pvalues), dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    finite = np.clip(finite, np.finfo(float).tiny, 1.0)
    return float(stats.combine_pvalues(finite, method="fisher").pvalue)


def aggregate_phi(runs: Sequence[AnalyzedRun]) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """Pointwise median/std/count without extrapolating shorter trajectories."""

    if not runs:
        raise ValueError("at least one run is required")
    width = max(run.causal.values.size for run in runs)
    matrix = np.full((len(runs), width), np.nan)
    for row, run in enumerate(runs):
        matrix[row, : run.causal.values.size] = run.causal.values
    count = np.sum(np.isfinite(matrix), axis=0)
    median = np.nanmedian(matrix, axis=0)
    std = np.full(width, np.nan)
    for column in np.flatnonzero(count > 1):
        std[column] = np.nanstd(matrix[:, column], ddof=1)
    return median, std, count


def summarize_control_runs(runs: Sequence[AnalyzedRun]) -> Dict[str, Any]:
    """Aggregate the paper's Figure 2-4 control claims."""

    if not runs:
        raise ValueError("at least one run is required")
    median, _, count = aggregate_phi(runs)
    valid = np.isfinite(median) & (count >= max(3, len(runs) // 10))
    regression = stats.linregress(np.flatnonzero(valid), median[valid])
    rho = np.asarray([run.spearman_rho for run in runs], dtype=float)
    rho_p = np.asarray([run.spearman_p for run in runs], dtype=float)
    mean_test_values = rho[np.isfinite(rho)]
    mean_test = (
        stats.ttest_1samp(mean_test_values, popmean=0.0)
        if mean_test_values.size >= 2
        else None
    )
    higher = np.asarray(
        [run.mean_phi_replicating > run.mean_phi_drift for run in runs], dtype=bool
    )
    positive = rho > 0
    significant = rho_p < 0.05
    evaluable = np.isfinite(rho) & np.isfinite(rho_p)
    return {
        "run_count": len(runs),
        "aggregate_phi_regression": {
            "slope": float(regression.slope),
            "pvalue": float(regression.pvalue),
            "points": int(valid.sum()),
        },
        "spearman": {
            "mean_rho": float(np.nanmean(rho)),
            "evaluable_runs": int(np.sum(evaluable)),
            "not_evaluable_runs": int(np.sum(~evaluable)),
            "positive_runs": int(np.sum(positive)),
            "positive_significant_runs": int(np.sum(positive & significant)),
            "negative_significant_runs": int(np.sum((~positive) & significant)),
            "one_sample_t_pvalue": (
                float(mean_test.pvalue) if mean_test is not None else float("nan")
            ),
        },
        "replicating_phi_higher_runs": int(higher.sum()),
        "mann_whitney_significant_runs": int(
            np.sum(
                [
                    np.isfinite(run.mann_whitney_p) and run.mann_whitney_p < 0.05
                    for run in runs
                ]
            )
        ),
        "fisher_combined_mann_whitney_p": _fisher_pvalue(
            run.mann_whitney_p for run in runs
        ),
        "ljung_box_significant_runs": int(
            np.sum(
                [np.isfinite(run.ljung_box_p) and run.ljung_box_p < 0.05 for run in runs]
            )
        ),
        "differenced_ljung_box_significant_runs": int(
            np.sum(
                [
                    np.isfinite(run.differenced_ljung_box_p)
                    and run.differenced_ljung_box_p < 0.05
                    for run in runs
                ]
            )
        ),
        "median_ljung_box_p": float(
            np.nanmedian([run.ljung_box_p for run in runs])
        ),
        "runs_with_three_sigma_spikes": int(
            np.sum([run.spike_indices.size > 0 for run in runs])
        ),
    }


def spike_correlations(runs: Sequence[AnalyzedRun]) -> Dict[str, Any]:
    """Relate run-level self-replication probability to spike summaries.

    The preprint does not state whether its spike correlations pooled events
    or summarized runs. We use the statistically independent run as the unit:
    mean normalized spike time, mean normalized inter-spike distance, and mean
    spike height are each correlated with that run's replication probability.
    """

    rows: List[Dict[str, float]] = []
    for run in runs:
        indices = run.spike_indices
        length = max(1, run.causal.values.size - 1)
        rows.append(
            {
                "run_index": float(run.run_index),
                "replication_probability": run.metrics.probability,
                "spike_count": float(indices.size),
                "mean_spike_time": (
                    float(np.mean(indices / length)) if indices.size else float("nan")
                ),
                "mean_spike_distance": (
                    float(np.mean(np.diff(indices) / length))
                    if indices.size >= 2
                    else float("nan")
                ),
                "mean_spike_height": (
                    float(np.mean(run.causal.values[indices]))
                    if indices.size
                    else float("nan")
                ),
            }
        )
    frame = pd.DataFrame(rows)
    correlations: Dict[str, Any] = {"unit": "run", "rows": rows}
    for metric in ("mean_spike_time", "mean_spike_distance", "mean_spike_height"):
        rho, pvalue = safe_spearman(
            frame[metric].to_numpy(), frame["replication_probability"].to_numpy()
        )
        correlations[metric] = {
            "rho": rho,
            "pvalue": pvalue,
            "n": int(frame[[metric, "replication_probability"]].dropna().shape[0]),
        }
    return correlations


def intervention_generation_trends(runs: Sequence[AnalyzedRun]) -> Dict[str, Any]:
    """OLS trends in generation-level replication probability by treatment."""

    results: Dict[str, Any] = {}
    for treatment in ("max", "control", "min"):
        selected = [run for run in runs if run.treatment == treatment]
        if not selected:
            continue
        matrix = np.vstack([run.generation_probabilities() for run in selected])
        generation = np.broadcast_to(np.arange(matrix.shape[1]), matrix.shape)
        finite = np.isfinite(matrix)
        regression = stats.linregress(generation[finite], matrix[finite])
        results[treatment] = {
            "slope_probability_per_generation": float(regression.slope),
            "pvalue": float(regression.pvalue),
            "observations": int(finite.sum()),
        }
    return results


def intervention_table(runs: Sequence[AnalyzedRun]) -> pd.DataFrame:
    """Mean and sample standard deviation for Table 1 outcomes."""

    frame = records_frame(runs)
    ordered = [name for name in ("max", "control", "min") if name in set(frame.treatment)]
    rows: List[Mapping[str, Any]] = []
    for treatment in ordered:
        selected = frame[frame.treatment == treatment]
        row: Dict[str, Any] = {"treatment": treatment, "runs": len(selected)}
        for column in ("persistence", "probability", "consistency", "time_to_first"):
            row[f"{column}_mean"] = float(selected[column].mean())
            row[f"{column}_std"] = float(selected[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def intervention_tests(runs: Sequence[AnalyzedRun]) -> Dict[str, Any]:
    """Two-sided Mann-Whitney comparisons for reconstructed Table 1 metrics."""

    frame = records_frame(runs)
    results: Dict[str, Any] = {}
    for metric in ("persistence", "probability", "consistency", "time_to_first"):
        results[metric] = {}
        for first, second in (("max", "control"), ("min", "control"), ("max", "min")):
            x = frame.loc[frame.treatment == first, metric].dropna().to_numpy()
            y = frame.loc[frame.treatment == second, metric].dropna().to_numpy()
            if x.size and y.size:
                test = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
                pvalue = float(test.pvalue)
            else:
                pvalue = float("nan")
            results[metric][f"{first}_vs_{second}"] = pvalue
    return results
