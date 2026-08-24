from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_process_prevalence(rows: list[dict[str, Any]], output: Path) -> None:
    table = pd.DataFrame(rows)
    metrics = (
        "break_event",
        "resume_2",
        "episode_3",
        "old_return",
        "persist_5",
        "positive_gain",
        "repeat_return",
    )
    groups = (("CONF", "02"), ("VALI", "02"), ("CONF", "03"), ("VALI", "03"))
    labels: list[str] = []
    estimates: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    colors: list[str] = []
    palette = {"CONF": "#4c78a8", "VALI": "#9ecae9"}
    for cohort, candidate in groups:
        for metric in metrics:
            selected = table[
                (table.cohort == cohort)
                & (table.candidate == candidate)
                & (table.metric == metric)
            ].iloc[0]
            value = float(selected.estimate)
            labels.append(f"{cohort}-{candidate}\n{metric.replace('_', '-')}")
            estimates.append(value)
            lower_errors.append(max(0.0, value - float(selected.lower_95)))
            upper_errors.append(max(0.0, float(selected.upper_95) - value))
            colors.append(palette[cohort])

    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(15, 6.6), constrained_layout=True)
    axis.bar(
        x,
        estimates,
        yerr=np.vstack((lower_errors, upper_errors)),
        color=colors,
        edgecolor="none",
        capsize=2,
        width=0.78,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=62, ha="right", fontsize=8)
    axis.set_ylabel("Branch probability")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Seven distinct plastic-heredity process probabilities")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_rank_transfer(metrics: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    model_names = ("history", "beta", "full")
    labels = ("history", "beta only", "full state")
    positions = np.arange(3)
    for column, candidate in enumerate(("02", "03")):
        candidate_metrics = metrics[candidate]["models"]
        overall = [candidate_metrics[name]["overall_spearman_mean"] for name in model_names]
        centered = [candidate_metrics[name]["centered_spearman_mean"] for name in model_names]
        for row, (values, suffix) in enumerate(
            ((overall, "overall"), (centered, "within matrix"))
        ):
            axis = axes[row, column]
            axis.bar(positions, values, color="#5b9bd5")
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=18)
            axis.set_ylabel("Spearman")
            axis.set_title(f"Candidate {candidate} - {suffix}")
            low = min(-0.1, min(values) - 0.05)
            high = max(0.7 if row else 0.9, max(values) + 0.08)
            axis.set_ylim(low, min(1.0, high))
    fig.suptitle("Frozen development model ranking on untouched matrices", fontsize=16)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_calibration(states: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True, constrained_layout=True)
    for axis, candidate in zip(axes, ("02", "03")):
        selected = states[states.candidate == candidate]
        axis.scatter(
            selected.prediction_full,
            selected.q_half_B,
            s=24,
            alpha=0.68,
            color="#2c7fb8",
            edgecolor="none",
        )
        axis.plot((0, 1), (0, 1), linestyle="--", color="black", linewidth=1.0)
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Frozen past-observable prediction")
        axis.set_title(f"Candidate {candidate}")
    axes[0].set_ylabel("Independent branch-half probability")
    fig.suptitle("Prospective frozen-coordinate calibration view", fontsize=16)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def create_figures(
    metrics: dict[str, Any],
    process_summary: list[dict[str, Any]],
    states: pd.DataFrame,
    output_directory: Path,
) -> None:
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    plot_process_prevalence(
        process_summary, figure_directory / "plastic_heredity_processes.png"
    )
    plot_rank_transfer(metrics, figure_directory / "rank_transfer.png")
    plot_calibration(states, figure_directory / "calibration.png")

