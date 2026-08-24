"""Figures for the registered inheritance-dependence analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _gain_figure(metrics: dict[str, Any], destination: Path) -> None:
    rows = metrics["primary_tests"]
    contrasts = ["markov_vs_iid", "semimarkov_vs_markov"]
    candidates = sorted({row["candidate"] for row in rows})
    labels = ["Markov − IID", "Duration − Markov"]
    colors = {candidate: color for candidate, color in zip(candidates, ("#2166ac", "#b2182b"))}
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    width = 0.24
    for candidate_index, candidate in enumerate(candidates):
        selected = [
            next(
                row
                for row in rows
                if row["candidate"] == candidate and row["contrast"] == contrast
            )
            for contrast in contrasts
        ]
        x = np.arange(len(contrasts)) + (candidate_index - 0.5) * width
        values = np.asarray([row["gain_bits_per_transition"] for row in selected])
        lower = values - np.asarray([row["gain_ci95"][0] for row in selected])
        upper = np.asarray([row["gain_ci95"][1] for row in selected]) - values
        axis.bar(x, values, width=width, color=colors[candidate], alpha=0.85, label=f"Candidate {candidate}")
        axis.errorbar(x, values, yerr=np.vstack((lower, upper)), fmt="none", color="#202020", capsize=4, linewidth=1.2)
        for location, row in zip(x, selected):
            direction_values = [
                item["gain_bits_per_transition"] for item in row["directions"].values()
            ]
            axis.scatter(
                [location - 0.035, location + 0.035],
                direction_values,
                marker="D",
                s=24,
                facecolor="white",
                edgecolor="#202020",
                zorder=4,
            )
    axis.axhline(0.0, color="#303030", linewidth=1.0)
    axis.set_xticks(np.arange(len(contrasts)), labels)
    axis.set_ylabel("Held-out gain (bits per transition)")
    axis.set_title("Support-matched, whole-matrix cross-fitted gains")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _probability_figure(fit_rows: list[dict[str, object]], destination: Path) -> None:
    frame = pd.DataFrame(fit_rows)
    candidates = sorted(frame["candidate"].unique())
    figure, axes = plt.subplots(
        len(candidates), 2, figsize=(10.2, 3.8 * len(candidates)), sharex=True, sharey=True, constrained_layout=True
    )
    if len(candidates) == 1:
        axes = np.asarray([axes])
    direction_colors = {
        direction: color
        for direction, color in zip(
            sorted(frame["direction"].unique()), ("#1b9e77", "#7570b3")
        )
    }
    for row_index, candidate in enumerate(candidates):
        candidate_frame = frame[frame["candidate"] == candidate]
        for previous in (0, 1):
            axis = axes[row_index, previous]
            for direction, color in direction_colors.items():
                selected = candidate_frame[candidate_frame["direction"] == direction]
                iid = selected[selected["model"] == "iid"]
                markov = selected[
                    (selected["model"] == "markov")
                    & (selected["previous"] == previous)
                ]
                semi = selected[
                    (selected["model"] == "semimarkov")
                    & (selected["previous"] == previous)
                ].copy()
                semi["duration_numeric"] = semi["duration_bin"].map(
                    {"1": 1, "2": 2, "3": 3, "4": 4, "5+": 5}
                )
                semi = semi.sort_values("duration_numeric")
                axis.axhline(
                    float(iid.iloc[0]["probability"]),
                    color=color,
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                )
                axis.axhline(
                    float(markov.iloc[0]["probability"]),
                    color=color,
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.9,
                )
                axis.plot(
                    semi["duration_numeric"].to_numpy(dtype=float),
                    semi["probability"].to_numpy(dtype=float),
                    color=color,
                    marker="o",
                    linewidth=1.7,
                    label=direction.replace("_", " "),
                )
            axis.set_title(f"Candidate {candidate}; previous symbol = {previous}")
            axis.set_xticks((1, 2, 3, 4, 5), ("1", "2", "3", "4", "5+"))
            axis.set_xlabel("Past-only run-duration bin")
            axis.set_ylabel("Fitted P(next inherited)")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(color="#e1e1e1", linewidth=0.7)
            if row_index == 0 and previous == 1:
                axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Cross-fitted transition probabilities (solid = duration-aware)")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def create_memory_figures(
    metrics: dict[str, Any], fit_rows: list[dict[str, object]], output: Path
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _gain_figure(metrics, figures / "memory_log_loss_gains.png")
    _probability_figure(fit_rows, figures / "memory_transition_probabilities.png")
