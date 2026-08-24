"""Data figures corresponding to preprint Figures 2-6."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .analysis import AnalyzedRun, aggregate_phi, records_frame


MODEL_LABELS: Mapping[str, str] = {
    "phi": r"$\Phi^r$",
    "composition_change": "change\nin comp.",
    "compositions": "compositions",
    "fluxes": "fluxes",
    "baseline": "baseline",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_figure2(runs: Sequence[AnalyzedRun], path: Path) -> None:
    median, std, count = aggregate_phi(runs)
    x = np.arange(median.size)
    valid = count >= max(3, len(runs) // 10)
    regression = stats.linregress(x[valid], median[valid])
    ranked = sorted(runs, key=lambda run: np.ptp(run.causal.values))
    samples = [ranked[len(ranked) // 4], ranked[len(ranked) // 2], ranked[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x[valid], median[valid], color="#2a7fbb", linewidth=1.2)
    ax.fill_between(x[valid], median[valid] - std[valid], median[valid] + std[valid], color="#2a7fbb", alpha=0.15)
    ax.plot(x[valid], regression.intercept + regression.slope * x[valid], color="red", linewidth=1.2, label=f"p={regression.pvalue:.3g}")
    ax.set_title(r"A) Median $\pm$ std of $\Phi^r$")
    ax.legend(frameon=False)
    for label, sample, axis in zip(("B", "C", "D"), samples, axes.flat[1:]):
        axis.plot(sample.causal.values, color="#2a7fbb", linewidth=1)
        axis.set_title(f"{label}) Sample run {sample.run_index}")
    for axis in axes.flat:
        axis.set_xlabel("molecular step")
        axis.set_ylabel(r"$\Phi^r$ (nats)")
    _save(fig, path)


def plot_figure3(runs: Sequence[AnalyzedRun], path: Path) -> None:
    rho = np.asarray([run.spearman_rho for run in runs])
    pvalue = np.asarray([run.spearman_p for run in runs])
    finite = np.isfinite(rho)
    categories = {
        "Positive &\nsignificant": int(np.sum((rho > 0) & (pvalue < 0.05))),
        "Positive &\nnon-significant": int(np.sum((rho > 0) & (pvalue >= 0.05))),
        "Negative &\nsignificant": int(np.sum((rho < 0) & (pvalue < 0.05))),
        "Negative &\nnon-significant": int(np.sum((rho <= 0) & (pvalue >= 0.05))),
        "Not\nevaluable": int(np.sum(~finite)),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    axes[0].hist(rho[finite], bins=15, color="#2a7fbb")
    axes[0].axvline(0, color="black", linestyle=":", label=r"$\rho=0$")
    axes[0].axvline(np.nanmean(rho), color="red", linestyle="--", label=fr"mean $\rho={np.nanmean(rho):.3f}$")
    axes[0].set(xlabel="Spearman's rho", ylabel="count", title="A) Correlation coefficient")
    axes[0].legend(frameon=False)
    axes[1].bar(
        categories.keys(),
        np.asarray(list(categories.values())) / len(runs),
        color=["#e4c51a", "#3578a8", "#ed7d31", "#cccccc", "#7f7f7f"],
    )
    axes[1].set(ylabel="fraction of runs", title="B) Sign and significance")
    _save(fig, path)


def plot_figure4(runs: Sequence[AnalyzedRun], path: Path) -> None:
    frame = records_frame(runs)
    paired = frame[["mean_phi_drift", "mean_phi_replicating"]].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for row in paired:
        if np.isfinite(row).all():
            axes[0].plot([0, 1], row, alpha=0.55, linewidth=1)
    axes[0].set_xticks([0, 1], ["drift", "self-repl."])
    axes[0].set_title(r"A) Mean $\Phi^r$ per run")
    median = np.nanmedian(paired, axis=0)
    std = np.nanstd(paired, axis=0, ddof=1)
    axes[1].errorbar([0, 1], median, yerr=std, marker="o", color="#2a7fbb", capsize=4)
    axes[1].set_xticks([0, 1], ["drift", "self-repl."])
    axes[1].set_title(r"B) Median $\pm$ std")
    for axis in axes:
        axis.set_ylabel(r"$\Phi^r$ (nats)")
    _save(fig, path)


def plot_figure5(forecast: pd.DataFrame, path: Path) -> None:
    order = ["phi", "composition_change", "compositions", "fluxes", "baseline"]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    sns.boxplot(data=forecast, x="model", y="accuracy", order=order, ax=ax, color="white", width=0.55)
    sns.stripplot(data=forecast, x="model", y="accuracy", order=order, ax=ax, color="#2a7fbb", size=4, jitter=0.12)
    ax.set_xticks(np.arange(len(order)), [MODEL_LABELS[name] for name in order])
    ax.set(xlabel="", ylabel="binary accuracy", title=r"Forecasting future self-replication from the first 25% of each run")
    _save(fig, path)


def plot_figure6(runs: Sequence[AnalyzedRun], path: Path) -> None:
    frame = records_frame(runs)
    order = [name for name in ("max", "control", "min") if name in set(frame.treatment)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    sns.boxplot(data=frame, x="treatment", y="persistence", order=order, ax=axes[0], color="white")
    axes[0].set(xlabel="", ylabel="molecular steps", title="B) Self-replication persistence")
    colors = {"max": "#2a7fbb", "control": "#ff7f0e", "min": "#2ca02c"}
    for treatment in order:
        selected = [run for run in runs if run.treatment == treatment]
        matrix = np.vstack([run.generation_probabilities() for run in selected])
        generation = np.broadcast_to(np.arange(matrix.shape[1]), matrix.shape)
        finite = np.isfinite(matrix)
        x_observed = generation[finite].astype(float)
        y_observed = matrix[finite]
        regression = stats.linregress(x_observed, y_observed)
        x = np.arange(matrix.shape[1], dtype=float)
        fitted = regression.intercept + regression.slope * x
        residual = y_observed - (
            regression.intercept + regression.slope * x_observed
        )
        dof = max(1, y_observed.size - 2)
        residual_scale = np.sqrt(np.sum(residual**2) / dof)
        centered = x_observed - np.mean(x_observed)
        leverage = 1 / y_observed.size + (x - np.mean(x_observed)) ** 2 / np.sum(
            centered**2
        )
        margin = stats.t.ppf(0.975, dof) * residual_scale * np.sqrt(leverage)
        lower = np.clip(fitted - margin, 0.0, 1.0)
        upper = np.clip(fitted + margin, 0.0, 1.0)
        axes[1].plot(
            x,
            100 * fitted,
            label=f"{treatment} (p={regression.pvalue:.2g})",
            color=colors[treatment],
        )
        axes[1].fill_between(
            x, 100 * lower, 100 * upper, color=colors[treatment], alpha=0.15
        )
    axes[1].set(xlabel="GARD generation", ylabel="self-replication probability (%)", title="C) Probability over generations")
    axes[1].legend(frameon=False)
    _save(fig, path)


def plot_sensitivity(
    detector: pd.DataFrame,
    causal: pd.DataFrame,
    tau: pd.DataFrame,
    path: Path,
) -> None:
    """Visual audit of the main choices omitted by the preprint."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for reference, selected in detector.groupby("reference_states", sort=False):
        axes[0].plot(
            selected.similarity_threshold,
            selected.probability_mean,
            marker="o",
            label=reference.replace("_", " "),
        )
        axes[0].fill_between(
            selected.similarity_threshold,
            np.clip(selected.probability_mean - selected.probability_std, 0, 1),
            np.clip(selected.probability_mean + selected.probability_std, 0, 1),
            alpha=0.12,
        )
    axes[0].axhline(0.88, color="black", linestyle=":", label="paper control mean")
    axes[0].set(
        xlabel="composition-similarity cutoff",
        ylabel="self-replication probability",
        title="A) Replicator definition",
        ylim=(0, 1.02),
    )
    axes[0].legend(frameon=False, fontsize=8)

    evaluable = causal.evaluable_correlations.replace(0, np.nan)
    positive_fraction = causal.positive_correlations / evaluable
    variant_labels = [str(value).replace("_", " ") for value in causal.variant]
    axes[1].bar(variant_labels, positive_fraction, color="#2a7fbb")
    axes[1].axhline(0.73, color="black", linestyle=":", label="paper: 73/100")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set(
        xlabel="causal estimator variant",
        ylabel="fraction positive (evaluable runs)",
        title=r"B) $Phi^r$ estimator",
        ylim=(0, 1.02),
    )
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].errorbar(
        tau.tau,
        tau.probability_mean,
        yerr=tau.probability_std,
        marker="o",
        capsize=3,
        color="#2a7fbb",
    )
    axes[2].axhline(0.88, color="black", linestyle=":", label="paper control mean")
    axes[2].set(
        xlabel="Poisson leap duration",
        ylabel="self-replication probability",
        title="C) Unreported stochastic time scale",
        ylim=(0, 1.02),
    )
    axes[2].legend(frameon=False, fontsize=8)
    _save(fig, path)
