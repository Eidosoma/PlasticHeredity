"""Sensitivity analyses for choices not numerically specified in the preprint."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .analysis import AnalyzedRun, safe_spearman
from .config import CausalConfig, ExperimentConfig, ReplicatorConfig
from .gard import simulate_gard
from .information import fit_causal_trajectory
from .replicators import detect_replicators, replicator_metrics


DEFAULT_THRESHOLDS: Tuple[float, ...] = (
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.98,
)
DEFAULT_TAU_VALUES: Tuple[float, ...] = (0.25, 0.50, 1.00)


def detector_threshold_sensitivity(
    runs: Sequence[AnalyzedRun],
    base: ReplicatorConfig,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    reference_states: Sequence[str] = ("generation_end", "all"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Re-label fixed trajectories across compotype-definition choices."""

    details: List[Dict[str, float | int | str]] = []
    for reference in reference_states:
        for threshold in thresholds:
            config = replace(
                base,
                similarity_threshold=float(threshold),
                reference_states=reference,
            )
            for run in runs:
                result = detect_replicators(run.trace, config)
                metrics = replicator_metrics(result.labels)
                aligned = result.labels[run.causal.time_indices]
                rho, pvalue = safe_spearman(
                    run.causal.values, aligned.astype(float)
                )
                self_phi = run.causal.values[aligned]
                drift_phi = run.causal.values[~aligned]
                details.append(
                    {
                        "reference_states": reference,
                        "similarity_threshold": float(threshold),
                        "run_index": run.run_index,
                        "support": result.support,
                        "persistence": metrics.persistence,
                        "probability": metrics.probability,
                        "spearman_rho": rho,
                        "spearman_p": pvalue,
                        "phi_higher_in_replicator": int(
                            self_phi.size > 0
                            and drift_phi.size > 0
                            and np.mean(self_phi) > np.mean(drift_phi)
                        ),
                    }
                )
    detail = pd.DataFrame(details)
    summaries: List[Dict[str, float | int | str]] = []
    for (reference, threshold), selected in detail.groupby(
        ["reference_states", "similarity_threshold"], sort=False
    ):
        finite = np.isfinite(selected.spearman_rho) & np.isfinite(
            selected.spearman_p
        )
        summaries.append(
            {
                "reference_states": reference,
                "similarity_threshold": threshold,
                "runs": len(selected),
                "evaluable_correlations": int(finite.sum()),
                "positive_correlations": int(
                    np.sum(finite & (selected.spearman_rho > 0))
                ),
                "positive_significant_correlations": int(
                    np.sum(
                        finite
                        & (selected.spearman_rho > 0)
                        & (selected.spearman_p < 0.05)
                    )
                ),
                "phi_higher_runs": int(selected.phi_higher_in_replicator.sum()),
                "probability_mean": float(selected.probability.mean()),
                "probability_std": float(selected.probability.std(ddof=1)),
                "probability_median": float(selected.probability.median()),
                "persistence_mean": float(selected.persistence.mean()),
                "support_mean": float(selected.support.mean()),
            }
        )
    return pd.DataFrame(summaries), detail


def _causal_variants(base: CausalConfig) -> List[Tuple[str, CausalConfig]]:
    """One-factor-at-a-time variants around the registered primary choice."""

    variants = [
        ("primary", base),
        ("pseudocount_0.1", replace(base, pseudocount=0.1)),
        ("pseudocount_1.0", replace(base, pseudocount=1.0)),
        ("median_fiedler_cut", replace(base, partition_cut="median")),
        ("mmi_synergy", replace(base, measure="mmi_synergy")),
    ]
    unique: List[Tuple[str, CausalConfig]] = []
    seen = set()
    for name, config in variants:
        signature = tuple(sorted(asdict(config).items()))
        if signature not in seen:
            unique.append((name, config))
            seen.add(signature)
    return unique


def causal_estimator_sensitivity(
    runs: Sequence[AnalyzedRun], base: CausalConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute Phi-r under one-factor-at-a-time estimator variants."""

    details: List[Dict[str, float | int | str]] = []
    for variant, config in _causal_variants(base):
        for run in runs:
            causal = (
                run.causal
                if config == run.causal.config
                else fit_causal_trajectory(run.trace.counts, config)
            )
            labels = run.replicator.labels[causal.time_indices]
            rho, pvalue = safe_spearman(causal.values, labels.astype(float))
            self_phi = causal.values[labels]
            drift_phi = causal.values[~labels]
            details.append(
                {
                    "variant": variant,
                    "pseudocount": config.pseudocount,
                    "partition_cut": config.partition_cut,
                    "measure": config.measure,
                    "run_index": run.run_index,
                    "phi_mean": float(np.mean(causal.values)),
                    "phi_std": float(np.std(causal.values, ddof=1)),
                    "spearman_rho": rho,
                    "spearman_p": pvalue,
                    "phi_higher_in_replicator": int(
                        self_phi.size > 0
                        and drift_phi.size > 0
                        and np.mean(self_phi) > np.mean(drift_phi)
                    ),
                }
            )
    detail = pd.DataFrame(details)
    summaries: List[Dict[str, float | int | str]] = []
    for variant, selected in detail.groupby("variant", sort=False):
        finite = np.isfinite(selected.spearman_rho) & np.isfinite(
            selected.spearman_p
        )
        summaries.append(
            {
                "variant": variant,
                "pseudocount": float(selected.pseudocount.iloc[0]),
                "partition_cut": str(selected.partition_cut.iloc[0]),
                "measure": str(selected.measure.iloc[0]),
                "runs": len(selected),
                "evaluable_correlations": int(finite.sum()),
                "spearman_mean": float(selected.loc[finite, "spearman_rho"].mean()),
                "positive_correlations": int(
                    np.sum(finite & (selected.spearman_rho > 0))
                ),
                "positive_significant_correlations": int(
                    np.sum(
                        finite
                        & (selected.spearman_rho > 0)
                        & (selected.spearman_p < 0.05)
                    )
                ),
                "phi_higher_runs": int(selected.phi_higher_in_replicator.sum()),
            }
        )
    return pd.DataFrame(summaries), detail


def gard_tau_sensitivity(
    config: ExperimentConfig,
    controls: Sequence[AnalyzedRun],
    *,
    tau_values: Sequence[float] = DEFAULT_TAU_VALUES,
    runs: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify sensitivity to the unreported Poisson leap duration."""

    count = min(runs, config.runs, len(controls))
    details: List[Dict[str, float | int]] = []
    for tau in tau_values:
        gard_config = replace(config.gard, tau=float(tau))
        for run_index in range(count):
            trace = (
                controls[run_index].trace
                if np.isclose(tau, config.gard.tau)
                else simulate_gard(gard_config, config.base_seed + run_index)
            )
            result = detect_replicators(trace, config.replicator)
            metrics = replicator_metrics(result.labels)
            details.append(
                {
                    "tau": float(tau),
                    "run_index": run_index,
                    "molecular_steps": int(trace.counts.shape[0]),
                    "support": result.support,
                    "persistence": metrics.persistence,
                    "probability": metrics.probability,
                }
            )
    detail = pd.DataFrame(details)
    summary = (
        detail.groupby("tau", sort=False)
        .agg(
            runs=("run_index", "size"),
            molecular_steps_mean=("molecular_steps", "mean"),
            molecular_steps_std=("molecular_steps", "std"),
            probability_mean=("probability", "mean"),
            probability_std=("probability", "std"),
            persistence_mean=("persistence", "mean"),
            support_mean=("support", "mean"),
        )
        .reset_index()
    )
    return summary, detail


def intervention_label_sensitivity(
    runs: Sequence[AnalyzedRun],
    base: ReplicatorConfig,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Re-score fixed intervention traces across compotype cutoffs."""

    details: List[Dict[str, float | int | str]] = []
    for threshold in thresholds:
        detector_config = replace(base, similarity_threshold=float(threshold))
        for run in runs:
            result = detect_replicators(run.trace, detector_config)
            metrics = replicator_metrics(result.labels)
            details.append(
                {
                    "similarity_threshold": float(threshold),
                    "treatment": run.treatment,
                    "run_index": run.run_index,
                    "persistence": metrics.persistence,
                    "probability": metrics.probability,
                    "consistency": metrics.consistency,
                    "time_to_first": metrics.time_to_first,
                }
            )
    detail = pd.DataFrame(details)
    summary = (
        detail.groupby(["similarity_threshold", "treatment"], sort=False)
        .agg(
            runs=("run_index", "size"),
            persistence_mean=("persistence", "mean"),
            persistence_std=("persistence", "std"),
            probability_mean=("probability", "mean"),
            probability_std=("probability", "std"),
            consistency_mean=("consistency", "mean"),
            consistency_std=("consistency", "std"),
            time_to_first_mean=("time_to_first", "mean"),
            time_to_first_std=("time_to_first", "std"),
        )
        .reset_index()
    )
    tests: List[Dict[str, float | str]] = []
    for threshold, selected in detail.groupby("similarity_threshold", sort=False):
        for metric in ("persistence", "probability", "consistency", "time_to_first"):
            for first, second in (
                ("max", "control"),
                ("min", "control"),
                ("max", "min"),
            ):
                paired = selected.pivot(
                    index="run_index", columns="treatment", values=metric
                )[[first, second]].dropna()
                if paired.empty:
                    mann_p = float("nan")
                    wilcoxon_p = float("nan")
                else:
                    mann_p = float(
                        stats.mannwhitneyu(
                            paired[first], paired[second], alternative="two-sided"
                        ).pvalue
                    )
                    difference = paired[first] - paired[second]
                    wilcoxon_p = (
                        float(
                            stats.wilcoxon(
                                paired[first], paired[second], alternative="two-sided"
                            ).pvalue
                        )
                        if np.any(difference != 0)
                        else 1.0
                    )
                tests.append(
                    {
                        "similarity_threshold": float(threshold),
                        "metric": metric,
                        "comparison": f"{first}_vs_{second}",
                        "mann_whitney_p": mann_p,
                        "paired_wilcoxon_p": wilcoxon_p,
                    }
                )
    return summary, detail, pd.DataFrame(tests)
