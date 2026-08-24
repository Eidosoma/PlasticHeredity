"""Discovery-only figures for the prospective mechanistic ablation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _primary_gain_figure(metrics: dict[str, Any], output: Path) -> None:
    rows = metrics["primary_tests"]
    labels = [
        f"{row['contrast']}\nc{row['candidate']}-{row['direction']}" for row in rows
    ]
    values = np.asarray([row["log_loss_gain"] for row in rows])
    lower = np.asarray([row["log_loss_gain_ci95"][0] for row in rows])
    upper = np.asarray([row["log_loss_gain_ci95"][1] for row in rows])
    passed = np.asarray([row["passes_gate"] for row in rows])
    colors = np.where(passed, "#2a9d8f", "#9aa0a6")
    x = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(13, 5.8), constrained_layout=True)
    axis.bar(x, values, color=colors, width=0.72)
    axis.errorbar(
        x,
        values,
        yerr=np.vstack((values - lower, upper - values)),
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
    )
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_ylabel("Paired log-loss gain")
    axis.set_title("Prospective nested attribution tests (95% matrix bootstrap CI)")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _rank_figure(metrics: dict[str, Any], output: Path) -> None:
    models = (
        "h10",
        "h10_state",
        "h10_state_beta",
        "h10_state_beta_interaction",
        "legacy_h9",
        "legacy_full",
    )
    labels = ("H10", "H10+S", "H10+S+B", "H10+S+B+I", "legacy H9", "legacy full")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True, constrained_layout=True)
    for axis, candidate in zip(axes, ("02", "03")):
        model_metrics = metrics["candidates"][candidate]["models"]
        values = [model_metrics[name]["centered_spearman_mean"] for name in models]
        axis.bar(np.arange(len(models)), values, color="#4c78a8")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(len(models)))
        axis.set_xticklabels(labels, rotation=32, ha="right")
        axis.set_title(f"Candidate {candidate}")
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Matrix-centered Spearman")
    fig.suptitle("Frozen models on untouched MECHCONF matrices", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _duplicate_figure(metrics: dict[str, Any], output: Path) -> None:
    selected = [
        row
        for row in metrics["descriptive_tests"]
        if row["contrast"] in ("corrected_duplicate", "ridge_duplicate")
    ]
    labels = [
        f"{row['contrast'].replace('_', ' ')}\nc{row['candidate']}-{row['direction']}"
        for row in selected
    ]
    values = np.asarray([row["log_loss_gain"] for row in selected])
    lower = np.asarray([row["log_loss_gain_ci95"][0] for row in selected])
    upper = np.asarray([row["log_loss_gain_ci95"][1] for row in selected])
    x = np.arange(len(selected))
    fig, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    axis.bar(x, values, color="#e9c46a", width=0.72)
    axis.errorbar(
        x,
        values,
        yerr=np.vstack((values - lower, upper - values)),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_ylabel("Paired log-loss gain from duplicated directions")
    axis.set_title("Duplicate-direction negative controls")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _calibration_figure(states: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True, constrained_layout=True)
    for axis, candidate in zip(axes, ("02", "03")):
        selected = states[states["candidate"] == candidate]
        axis.scatter(
            selected["prediction_h10_state_beta_interaction"],
            selected["q_half_B"],
            s=23,
            alpha=0.65,
            color="#2c7fb8",
            edgecolor="none",
        )
        axis.plot((0, 1), (0, 1), "--", color="black", linewidth=1)
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Frozen H10+S+B+I prediction")
        axis.set_title(f"Candidate {candidate}")
    axes[0].set_ylabel("Independent branch-half probability")
    fig.suptitle("Prospective mechanistic-model calibration", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def create_mechanistic_figures(
    metrics: dict[str, Any], states: pd.DataFrame, output_directory: Path
) -> None:
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    _primary_gain_figure(metrics, figure_directory / "primary_log_loss_gains.png")
    _rank_figure(metrics, figure_directory / "model_attribution.png")
    _duplicate_figure(metrics, figure_directory / "duplicate_controls.png")
    _calibration_figure(states, figure_directory / "calibration.png")
