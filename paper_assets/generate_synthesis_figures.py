#!/usr/bin/env python3
"""Generate the paper synthesis figures and V2 companion assets.

The script is deliberately read-only with respect to every scientific result
bundle. It writes only PNG files in the repository-root ``figures`` directory.
Retained local labels 02/03 are never pooled across clean rooms;
figures display the globally unique O-, T1-, and T2-prefixed contract aliases.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

EIDOSOMA_L54 = ROOT / (
    "original.1.8.2026.eidosoma-ai-scientist.stepReports/artifacts/research_steps/"
    "S19/loops/L54/S19_L54_FULL_RESULTS.md"
)
CODEX_F12 = ROOT / "replicators.13.8.2026.codex/results/full/metrics.json"
FABLE_F12 = ROOT / (
    "replicators.13.8.2026.fable/replication/results/confirmation_metrics.json"
)
CODEX_COHERENCE = ROOT / (
    "replicators.13.8.2026.codex/results/episode_coherence_audit/"
    "threshold_sensitivity.csv"
)
FABLE_COHERENCE = ROOT / (
    "replicators.13.8.2026.fable/replication/results_coherence/"
    "coherence_results.json"
)
CODEX_STRICT = ROOT / (
    "replicators.13.8.2026.codex/results/regime_confirmation/"
    "occurrence_metrics.json"
)
CODEX_STRICT_DIAGNOSTIC = ROOT / (
    "replicators.13.8.2026.codex/results/regime_prediction_diagnostic/"
    "metrics.json"
)
FABLE_STRICT = ROOT / (
    "replicators.13.8.2026.fable/replication/results_strict8_occurrence/"
    "strict8_results.json"
)
CODEX_GENERATIVE_NULLS = ROOT / (
    "replicators.13.8.2026.codex/results/generative_null_decomposition/"
    "inference_metrics.json"
)
CODEX_CLOSED_LOOP = ROOT / (
    "replicators.13.8.2026.codex/results_intervention_replication/"
    "cr7_closed_loop_steering/primary_metrics.json"
)
CODEX_CLOSED_LOOP_EXTENSION = ROOT / (
    "replicators.13.8.2026.codex/results_intervention_replication/"
    "cr7_closed_loop_steering/conditional_active_extension/metrics.json"
)
CODEX_STRENGTH = ROOT / (
    "replicators.13.8.2026.codex/results_intervention_replication/"
    "p3c_throughput_confirmation/primary_metrics.json"
)
CODEX_RECOVERY = ROOT / (
    "replicators.13.8.2026.codex/results_intervention_replication/"
    "p4_shared_break_recovery/primary_metrics.json"
)
CODEX_NETWORK_CONFIRMATION = ROOT / (
    "replicators.13.8.2026.codex/results_intervention_replication/"
    "cr4_beta_surgery_confirmation/primary_metrics.json"
)
CODEX_MOLECULAR_RESISTANCE = ROOT / (
    "replicators.13.8.2026.codex/results_intervention_replication/"
    "cr5_resistance_resilience_confirmation/resistance/primary_metrics.json"
)
CODEX_MOLECULAR_RECOVERY = ROOT / (
    "replicators.13.8.2026.codex/results_intervention_replication/"
    "cr5r_shared_break_resilience_confirmation/primary_metrics.json"
)
CALIBRATION_VIEW = ROOT / (
    "replicators.13.8.2026.codex/results/scaled5/figures/calibration.png"
)


NAVY = "#17324D"
BLUE = "#2C78B8"
TEAL = "#1B998B"
GOLD = "#D89B2B"
ORANGE = "#D66A3A"
RED = "#B64040"
INK = "#263238"
MUTED = "#66727A"
LIGHT = "#EFF3F6"
GRID = "#D9E0E5"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def markdown_table(path: Path, heading: str) -> list[dict[str, str]]:
    """Read one Markdown table below an exact level-two heading."""
    lines = path.read_text(encoding="utf-8").splitlines()
    target = f"## {heading}"
    try:
        start = lines.index(target) + 1
    except ValueError as exc:
        raise RuntimeError(f"Missing table heading {target!r} in {path}") from exc

    table_lines: list[str] = []
    in_table = False
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            in_table = True
            table_lines.append(line)
        elif in_table and line.strip():
            break

    if len(table_lines) < 3:
        raise RuntimeError(f"No Markdown table found below {target!r}")

    def cells(line: str) -> list[str]:
        return [part.strip() for part in line.strip().strip("|").split("|")]

    headers = cells(table_lines[0])
    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        values = cells(line)
        if len(values) == len(headers):
            rows.append(dict(zip(headers, values)))
    return rows


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def candidate_short(value: str) -> str:
    return value.rsplit("-", maxsplit=1)[-1]


def matched_f12_summary() -> list[dict[str, Any]]:
    """Return matched 40-matrix evidence from all three implementations."""
    reliability_rows = markdown_table(EIDOSOMA_L54, "Independent branch-half reliability")
    prediction_rows = markdown_table(EIDOSOMA_L54, "F12 joint-event predictive metrics")
    score_rows = markdown_table(EIDOSOMA_L54, "Registered proper-score comparisons")

    output: list[dict[str, Any]] = []
    for candidate in ("02", "03"):
        rel = next(
            row
            for row in reliability_rows
            if candidate_short(row["candidateId"]) == candidate
        )
        pred = [
            row
            for row in prediction_rows
            if candidate_short(row["candidateId"]) == candidate
            and row["targetType"] == "JOINT_BREAK_RUN3"
        ]
        scores = [
            row
            for row in score_rows
            if candidate_short(row["candidateId"]) == candidate
            and row["comparisonId"] == "FULL_VS_DIRECT"
        ]
        full_centered = mean(
            [
                float(row["centeredQSpearman"])
                for row in pred
                if row["modelId"] == "FULL_STATE_GRAPH_HISTORY"
            ]
        )
        direct_centered = mean(
            [
                float(row["centeredQSpearman"])
                for row in pred
                if row["modelId"] == "DIRECT_HISTORY_PHASE"
            ]
        )
        output.append(
            {
                "implementation": "Originating",
                "candidate": candidate,
                "reliability": float(rel["splitHalfSpearman"]),
                "full_centered": full_centered,
                "direct_centered": direct_centered,
                "gain": mean([float(row["logLossImprovement"]) for row in scores]),
                "gain_lower": min(
                    float(row["logLossImprovementLower95"]) for row in scores
                ),
            }
        )

    codex = load_json(CODEX_F12)
    for candidate in ("02", "03"):
        cell = codex[candidate]
        directions = list(cell["directions"].values())
        output.append(
            {
                "implementation": "Clean-room test 1",
                "candidate": candidate,
                "reliability": cell["branch_half_reliability"],
                "full_centered": cell["models"]["full"]["centered_spearman_mean"],
                "direct_centered": cell["models"]["history"]["centered_spearman_mean"],
                "gain": mean([direction["log_loss_gain"] for direction in directions]),
                "gain_lower": min(
                    direction["log_loss_gain_ci95"][0] for direction in directions
                ),
            }
        )

    fable = load_json(FABLE_F12)
    for candidate in ("02", "03"):
        cell = fable[candidate]
        output.append(
            {
                "implementation": "Clean-room test 2",
                "candidate": candidate,
                "reliability": cell["reliability"],
                "full_centered": mean(cell["full_centered"]),
                "direct_centered": mean(cell["direct_centered"]),
                "gain": cell["logloss_gain_full_vs_direct"],
                "gain_lower": cell["logloss_gain_lower95"],
            }
        )
    return output


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_1_flow() -> None:
    fig, ax = plt.subplots(figsize=(20.5, 8.2))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 2.15)
    ax.axis("off")

    columns = [
        (
            (
                "Chapter 1  Phi-r reconstruction",
                "Does Phi-r predict and steer\nfirst-replicator arrival?",
            ),
            (
                "Reconstruction outcome",
                "Retrospective patterns recur;\nprospective ordering not recovered\nin public-source implementations",
            ),
            BLUE,
        ),
        (
            ("Chapter 2  Inspired discovery", "Can a present GARD state encode\nfuture self-maintaining organisation?"),
            (
                "Plastic heredity",
                "Frozen F12 break-and-renewal\ncoordinate confirmed on\nuntouched states",
            ),
            TEAL,
        ),
        (
            ("Chapter 3  Stress tests", "Separate clean rooms reproduce,\nscale and challenge the finding"),
            (
                "Stress testing",
                "Strict-eight recurs in GARD;\nevents precede carrier tests in\ncellular automata and Wagner networks",
            ),
            GOLD,
        ),
        (
            (
                "Chapter 4  Interventions",
                "Molecular edits and\ncatalytic-network perturbations",
            ),
            (
                "Bounded conclusion",
                "Resistance and recovery separate;\nstrength and weight arrangement\nboth alter the process",
            ),
            ORANGE,
        ),
        (
            (
                "Chapter 5  Return to Phi-r",
                "Does causal architecture register\nexperimentally changed heredity?",
            ),
            (
                "Information-dynamic bridge",
                "Phi-r-family gauges respond;\nformula, sign, clock and window\nremain measurement dependent",
            ),
            RED,
        ),
    ]

    for col, (top, bottom, color) in enumerate(columns):
        x = col + 0.08
        for y, content in ((1.24, top), (0.24, bottom)):
            title, body = content
            box = FancyBboxPatch(
                (x, y),
                0.84,
                0.65,
                boxstyle="round,pad=0.025,rounding_size=0.035",
                linewidth=1.8,
                edgecolor=color,
                facecolor="white",
            )
            ax.add_patch(box)
            ax.text(
                x + 0.42,
                y + 0.47,
                title,
                ha="center",
                va="center",
                fontsize=12.3,
                fontweight="bold",
                color=color,
            )
            ax.text(
                x + 0.42,
                y + 0.24,
                body,
                ha="center",
                va="center",
                fontsize=9.6,
                color=INK,
                linespacing=1.35,
            )
        ax.add_patch(
            FancyArrowPatch(
                (x + 0.42, 1.22),
                (x + 0.42, 0.92),
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.5,
                color=color,
            )
        )
        if col < len(columns) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.86, 0.56),
                    (x + 1.05, 1.55),
                    arrowstyle="-|>",
                    mutation_scale=15,
                    connectionstyle="arc3,rad=-0.18",
                    linewidth=1.4,
                    color=MUTED,
                )
            )

    ax.text(
        2.5,
        2.07,
        "Five-chapter evidence path",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        2.5,
        0.05,
        "The project preserves the motivating question, tests a process-level alternative, and returns to causal-architecture measurement.",
        ha="center",
        va="center",
        fontsize=10.8,
        color=MUTED,
    )
    save(fig, "figure1_replication_to_discovery.png")


def figure_1_evidence_architecture_v2() -> None:
    """Generate the V2 evidence ladder and human-directed programme map."""

    fig = plt.figure(figsize=(20.5, 11.8))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.08, 1.0], hspace=0.24)

    ax = fig.add_subplot(grid[0])
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 2.25)
    ax.axis("off")
    ax.text(
        0.02,
        2.15,
        "A  Plastic-heredity evidence ladder",
        fontsize=19,
        fontweight="bold",
        color=NAVY,
        va="top",
    )

    steps = [
        ("Local inheritance", "Parent → daughter\n$H>0.9$", BLUE, "-"),
        ("F12 renewal", "Break, then three\ninherited boundaries\nPredicted and steered", TEAL, "-"),
        ("Coherent episode", "Eight mutually\ncoherent daughters\nOccurrence reproduced", GOLD, "-"),
        ("Active maintenance", "External feedback\nfor 60–120 fissions", ORANGE, "-"),
        ("Transient cue", "CA motif channel\nOne generation", BLUE, "-"),
        ("Renewable carriers", "CA compact payload +\nWagner lineage latch\nThrough generation 16", TEAL, "-"),
        ("Evolvability", "Population selection and\nopen-ended change\nNot tested", MUTED, "--"),
    ]
    y_positions = np.linspace(0.82, 1.28, len(steps))
    for index, ((title, body, colour, linestyle), y) in enumerate(
        zip(steps, y_positions, strict=True)
    ):
        x = index + 0.06
        box = FancyBboxPatch(
            (x, y),
            0.86,
            0.68,
            boxstyle="round,pad=0.025,rounding_size=0.035",
            linewidth=1.8,
            edgecolor=colour,
            facecolor="white",
            linestyle=linestyle,
        )
        ax.add_patch(box)
        ax.text(
            x + 0.43,
            y + 0.49,
            title,
            ha="center",
            va="center",
            fontsize=11.3,
            fontweight="bold",
            color=colour,
        )
        ax.text(
            x + 0.43,
            y + 0.22,
            body,
            ha="center",
            va="center",
            fontsize=9.1,
            color=INK,
            linespacing=1.22,
        )
        if index < len(steps) - 1:
            next_y = y_positions[index + 1]
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.88, y + 0.34),
                    (x + 1.04, next_y + 0.34),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.35,
                    color=MUTED,
                )
            )

    gauge = FancyBboxPatch(
        (1.70, 0.08),
        3.00,
        0.42,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.5,
        edgecolor=RED,
        facecolor="white",
        linestyle="--",
    )
    ax.add_patch(gauge)
    ax.text(
        3.20,
        0.29,
        "Phi-r-family candidate gauges responded to altered trajectories;\nno shared foresight or transferable control gradient",
        ha="center",
        va="center",
        fontsize=10.3,
        color=INK,
    )
    ax.add_patch(
        FancyArrowPatch(
            (3.20, 0.50),
            (3.55, 0.93),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=RED,
            linestyle="--",
        )
    )

    ax = fig.add_subplot(grid[1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.35)
    ax.axis("off")
    ax.text(
        0.02,
        2.25,
        "B  Human-directed AI-scientist research architecture",
        fontsize=19,
        fontweight="bold",
        color=NAVY,
        va="top",
    )

    ax.text(
        5.0,
        2.00,
        "One human-directed programme with separate code paths—not independent laboratories",
        ha="center",
        va="center",
        fontsize=10.4,
        color=MUTED,
    )

    director = FancyBboxPatch(
        (4.0, 1.42),
        2.0,
        0.46,
        boxstyle="round,pad=0.025,rounding_size=0.04",
        linewidth=1.8,
        edgecolor=NAVY,
        facecolor="white",
    )
    ax.add_patch(director)
    ax.text(
        5.0,
        1.65,
        "Human research director",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=NAVY,
    )

    programmes = [
        ("Originating AI scientist", "Reconstruction →\nF12 discovery", BLUE),
        ("Clean-room test 1", "Separate GARD\ncodebase", TEAL),
        ("Clean-room test 2", "Separate GARD\ncodebase", GOLD),
        ("Cellular automata", "Cross-model occurrence,\ncue, and compact carrier", ORANGE),
        ("Wagner networks", "Cross-model occurrence\nand renewable carrier", RED),
    ]
    centres = np.linspace(1.0, 9.0, len(programmes))
    for centre, (title, body, colour) in zip(centres, programmes, strict=True):
        x = centre - 0.82
        box = FancyBboxPatch(
            (x, 0.62),
            1.64,
            0.62,
            boxstyle="round,pad=0.025,rounding_size=0.035",
            linewidth=1.7,
            edgecolor=colour,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(
            centre,
            1.05,
            title,
            ha="center",
            va="center",
            fontsize=11.1,
            fontweight="bold",
            color=colour,
        )
        ax.text(
            centre,
            0.81,
            body,
            ha="center",
            va="center",
            fontsize=9.8,
            color=INK,
            linespacing=1.2,
        )
        ax.add_patch(
            FancyArrowPatch(
                (5.0, 1.42),
                (centre, 1.25),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.15,
                color=MUTED,
                connectionstyle="arc3,rad=0.0",
            )
        )

    safeguards = FancyBboxPatch(
        (0.35, 0.10),
        9.30,
        0.34,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.3,
        edgecolor=NAVY,
        facecolor="#F5F8FB",
    )
    ax.add_patch(safeguards)
    ax.text(
        5.0,
        0.27,
        "Prospective freezing  •  seed firewalls  •  matrix-level inference  •  exact replay  •  retained negative gates  •  public artefacts",
        ha="center",
        va="center",
        fontsize=10.8,
        color=INK,
    )
    fig.suptitle(
        "Plastic heredity: evidence hierarchy and research architecture",
        fontsize=22,
        fontweight="bold",
        color=NAVY,
        y=0.995,
    )
    save(fig, "figure1_v2_evidence_architecture.png")


def figure_2_f12() -> None:
    data = matched_f12_summary()
    assert len(data) == 6
    assert all(row["gain"] > row["gain_lower"] > 0 for row in data)
    display_names = {
        "Originating": "Orig.",
        "Clean-room test 1": "Test 1",
        "Clean-room test 2": "Test 2",
    }
    alias_prefixes = {
        "Originating": "O",
        "Clean-room test 1": "T1",
        "Clean-room test 2": "T2",
    }
    labels = [
        f"{display_names[row['implementation']]}\n{alias_prefixes[row['implementation']]}-{row['candidate']}"
        for row in data
    ]
    colors = [BLUE, BLUE, TEAL, TEAL, GOLD, GOLD]
    markers = ["o" if row["candidate"] == "02" else "s" for row in data]
    x = np.arange(len(data), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True)
    fig.suptitle(
        "Matched 40-matrix confirmation in the originating run and two clean rooms",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
    )

    ax = axes[0]
    for i, row in enumerate(data):
        ax.scatter(
            x[i],
            row["reliability"],
            s=90,
            marker=markers[i],
            color=colors[i],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
    ax.set_ylim(0.82, 0.96)
    ax.set_ylabel("Independent branch-half Spearman ρ")
    ax.set_title("A  State-dependent probability is reliable", color=INK, loc="left")
    ax.set_xticks(x, labels)
    style_axis(ax)

    ax = axes[1]
    width = 0.34
    full = [row["full_centered"] for row in data]
    direct = [row["direct_centered"] for row in data]
    ax.bar(x - width / 2, full, width, label="Frozen composite", color=colors)
    ax.bar(
        x + width / 2,
        direct,
        width,
        label="Direct history",
        color="white",
        edgecolor=colors,
        linewidth=1.6,
    )
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Within-matrix Spearman ρ")
    ax.set_title("B  Composite ranks state-local risk", color=INK, loc="left")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax)

    ax = axes[2]
    for i, row in enumerate(data):
        ax.vlines(
            x[i],
            row["gain_lower"],
            row["gain"],
            color=colors[i],
            linewidth=4,
            alpha=0.8,
        )
        ax.scatter(
            x[i],
            row["gain"],
            s=90,
            marker=markers[i],
            color=colors[i],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_ylim(0, 0.06)
    ax.set_ylabel("Branch log-loss improvement")
    ax.set_title("C  Proper-score gain beyond direct history", color=INK, loc="left")
    ax.set_xticks(x, labels)
    ax.text(
        0.02,
        0.97,
        "Dot: estimate\nLine: reported/minimum 95% lower bound",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=MUTED,
    )
    style_axis(ax)

    fig.text(
        0.5,
        -0.015,
        "O = originating workflow; T1 and T2 are the clean-room tests defined in the text; suffixes preserve retained local labels.",
        ha="center",
        fontsize=10,
        color=MUTED,
    )
    save(fig, "figure3_cross_cleanroom_f12.png")


def figure_2_f12_v2() -> None:
    """Promote calibration while keeping the full figure and caption on one page."""

    if not CALIBRATION_VIEW.is_file():
        raise FileNotFoundError(CALIBRATION_VIEW)

    data = matched_f12_summary()
    assert len(data) == 6
    assert all(row["gain"] > row["gain_lower"] > 0 for row in data)
    display_names = {
        "Originating": "Orig.",
        "Clean-room test 1": "Test 1",
        "Clean-room test 2": "Test 2",
    }
    alias_prefixes = {
        "Originating": "O",
        "Clean-room test 1": "T1",
        "Clean-room test 2": "T2",
    }
    labels = [
        f"{display_names[row['implementation']]}\n{alias_prefixes[row['implementation']]}-{row['candidate']}"
        for row in data
    ]
    colours = [BLUE, BLUE, TEAL, TEAL, GOLD, GOLD]
    markers = ["o" if row["candidate"] == "02" else "s" for row in data]
    x = np.arange(len(data), dtype=float)

    calibration = plt.imread(CALIBRATION_VIEW)
    calibration = calibration[55:, ...]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(17.5, 10.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.90, 1.10]},
    )

    ax = axes[0, 0]
    for i, row in enumerate(data):
        ax.scatter(
            x[i],
            row["reliability"],
            s=90,
            marker=markers[i],
            color=colours[i],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
    ax.set_ylim(0.82, 0.96)
    ax.set_ylabel("Independent branch-half Spearman ρ")
    ax.set_title("A  State-dependent probability is reliable", color=INK, loc="left")
    ax.set_xticks(x, labels)
    style_axis(ax)

    ax = axes[0, 1]
    width = 0.34
    full = [row["full_centered"] for row in data]
    direct = [row["direct_centered"] for row in data]
    ax.bar(x - width / 2, full, width, label="Frozen composite", color=colours)
    ax.bar(
        x + width / 2,
        direct,
        width,
        label="Registered history-only ridge",
        color="white",
        edgecolor=colours,
        linewidth=1.6,
    )
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Within-matrix Spearman ρ")
    ax.set_title("B  Composite ranks state-local risk", color=INK, loc="left")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax)

    ax = axes[1, 0]
    for i, row in enumerate(data):
        ax.vlines(
            x[i],
            row["gain_lower"],
            row["gain"],
            color=colours[i],
            linewidth=4,
            alpha=0.8,
        )
        ax.scatter(
            x[i],
            row["gain"],
            s=90,
            marker=markers[i],
            color=colours[i],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_ylim(0, 0.06)
    ax.set_ylabel("Branch log-loss improvement")
    ax.set_title("C  Proper-score gain beyond the comparator", color=INK, loc="left")
    ax.set_xticks(x, labels)
    ax.text(
        0.02,
        0.97,
        "Dot: estimate\nLine: reported/minimum 95% lower bound",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=MUTED,
    )
    style_axis(ax)

    ax = axes[1, 1]
    ax.axis("off")
    ax.imshow(calibration)
    ax.set_title(
        "D  Frozen prediction versus independent confirmation",
        color=INK,
        loc="left",
        fontsize=13.0,
        fontweight="bold",
    )

    fig.suptitle(
        "Matched-scale F12 prediction and retained confirmation calibration",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
    )
    save(fig, "figure3_v2_cross_cleanroom_f12_calibration.png")


def coherence_summary() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with CODEX_COHERENCE.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["half"] == "combined"
                and math.isclose(float(row["coherence_threshold_strict"]), 0.9)
                and math.isclose(
                    float(row["distinctness_max_anchor_threshold_inclusive"]), 0.9
                )
            ):
                rows.append(
                    {
                        "label": f"Test 1\n{row['source_cohort']} {row['candidate']}",
                        "rate": float(row["coherent_rate"]),
                        "kind": "post-hoc audit",
                    }
                )
    order = {"scaled5": 0, "MECHCONF": 1, "MECHCONF2": 2}
    rows.sort(
        key=lambda row: (
            order[row["label"].splitlines()[1].rsplit(" ", 1)[0]],
            row["label"],
        )
    )
    fable = load_json(FABLE_COHERENCE)
    for candidate in ("02", "03"):
        rows.append(
            {
                "label": f"Test 2\nprospective {candidate}",
                "rate": fable[candidate]["P_coherent_given_joint"],
                "kind": "prospective",
            }
        )
    return rows


def figure_3_coherence() -> None:
    rows = coherence_summary()
    assert len(rows) == 8
    assert all(0 < row["rate"] < 0.1 for row in rows)
    fig = plt.figure(figsize=(17, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.45])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    fig.suptitle(
        "Why a three-fission renewal episode is not a coherent compositional regime",
        fontsize=18,
        fontweight="bold",
        color=NAVY,
    )

    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    points = [(0.12, 0.70), (0.37, 0.76), (0.62, 0.66), (0.87, 0.75)]
    labels = ["post-break\nepisode parent", "daughter 1", "daughter 2", "daughter 3"]
    point_colors = [RED, TEAL, TEAL, TEAL]
    for i, ((px, py), label, color) in enumerate(zip(points, labels, point_colors)):
        ax0.scatter(px, py, s=620, color=color, edgecolor="white", linewidth=2, zorder=3)
        ax0.text(px, py - 0.13, label, ha="center", va="top", fontsize=10.5, color=INK)
        if i > 0:
            x0, y0 = points[i - 1]
            ax0.add_patch(
                FancyArrowPatch(
                    (x0 + 0.04, y0),
                    (px - 0.04, py),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=2,
                    color=TEAL,
                )
            )
            ax0.text(
                (x0 + px) / 2,
                max(y0, py) + 0.065,
                "H > 0.9",
                ha="center",
                fontsize=9.5,
                color=TEAL,
            )
    ax0.add_patch(
        FancyArrowPatch(
            (0.38, 0.54),
            (0.86, 0.54),
            arrowstyle="<->",
            mutation_scale=13,
            linewidth=1.4,
            linestyle="--",
            color=RED,
        )
    )
    ax0.text(
        0.62,
        0.48,
        "non-adjacent similarity\nwas not required",
        ha="center",
        va="top",
        fontsize=10.2,
        color=RED,
    )
    ax0.text(
        0.5,
        0.92,
        "Adjacent inheritance is not transitive",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    callout = (
        "Observed geometry\n\n"
        "Clean-room test 1 (three 200-matrix cohorts):\n"
        "• mean weakest daughter-pair H = 0.681-0.704\n"
        "• 93.2-94.9% stayed outside the old H>0.9 neighborhood\n"
        "• 75.9-78.9% of resolved episodes continued to five\n\n"
        "Clean-room test 2 prospective span medians: 0.720 / 0.728"
    )
    ax0.text(
        0.5,
        0.08,
        callout,
        ha="center",
        va="bottom",
        fontsize=10.2,
        color=INK,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": LIGHT, "edgecolor": GRID},
    )

    x = np.arange(len(rows))
    values = [100 * row["rate"] for row in rows]
    bar_colors = [TEAL if row["kind"] == "post-hoc audit" else GOLD for row in rows]
    bars = ax1.bar(x, values, color=bar_colors, width=0.72)
    ax1.set_ylim(0, 10)
    ax1.set_ylabel("Qualifying F12 episodes meeting criterion (%)")
    ax1.set_title("Registered episode-coherence criterion", loc="left", color=INK)
    ax1.set_xticks(x, [row["label"] for row in rows], rotation=25, ha="right")
    for bar, value in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.22,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=INK,
        )
    ax1.text(
        0.98,
        0.96,
        "The clean-room test 2 language-upgrade gate was ≥80%\n(off scale); it failed in both candidates.",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=10.2,
        color=RED,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF5F5", "edgecolor": "#E7B7B7"},
    )
    ax1.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=TEAL, label="Clean-room test 1: post-hoc all-pairs"),
            plt.Rectangle((0, 0), 1, 1, color=GOLD, label="Clean-room test 2: prospective span"),
        ],
        frameon=False,
        loc="upper left",
    )
    style_axis(ax1)
    save(fig, "figure4_three_fission_coherence.png")


def strict_summary() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    codex = load_json(CODEX_STRICT)
    codex_diag = load_json(CODEX_STRICT_DIAGNOSTIC)
    fable = load_json(FABLE_STRICT)
    rows: list[dict[str, Any]] = []
    for candidate in ("02", "03"):
        for half in ("A", "B"):
            cell = codex["endpoints"]["primary_all8"][candidate]["halves"][half]
            rows.append(
                {
                    "implementation": "Clean-room test 1",
                    "candidate": candidate,
                    "half": half,
                    "rate": cell["rate"],
                    "ci": cell["ci95"],
                    "events": cell["events"],
                    "event_matrices": cell["event_matrices"],
                }
            )
    for candidate in ("02", "03"):
        for half in ("A", "B"):
            cell = fable["cells"][f"{candidate}/{half}"]
            rows.append(
                {
                    "implementation": "Clean-room test 2",
                    "candidate": candidate,
                    "half": half,
                    "rate": cell["rate"],
                    "ci": cell["ci95"],
                    "events": cell["events"],
                    "event_matrices": cell["matrices_with_event"],
                }
            )
    return rows, codex_diag["stage_rates"], fable["components"]


def figure_4_strict8() -> None:
    rows, codex_stages, fable_stages = strict_summary()
    assert len(rows) == 8
    assert all(row["ci"][0] > 0 and row["event_matrices"] > 100 for row in rows)
    labels = [
        f"{'T1' if row['implementation'] == 'Clean-room test 1' else 'T2'}-"
        f"{row['candidate']}/{row['half']}"
        for row in rows
    ]
    y = np.arange(len(rows))
    colors = [
        TEAL if row["implementation"] == "Clean-room test 1" else GOLD
        for row in rows
    ]

    fig = plt.figure(figsize=(17, 8.1), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.18])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(grid[0, 2])
    fig.suptitle(
        "Cross-implementation prospective occurrence of the strict coherent eight-fission episode",
        fontsize=18,
        fontweight="bold",
        color=NAVY,
    )

    for i, row in enumerate(rows):
        rate = 100 * row["rate"]
        low, high = [100 * value for value in row["ci"]]
        ax0.errorbar(
            rate,
            i,
            xerr=[[rate - low], [high - rate]],
            fmt="o" if row["candidate"] == "02" else "s",
            color=colors[i],
            ecolor=colors[i],
            elinewidth=2,
            capsize=4,
            markersize=8,
        )
        ax0.text(high + 0.035, i, f"{rate:.2f}%", va="center", fontsize=9.3, color=INK)
    ax0.set_yticks(y, labels)
    ax0.invert_yaxis()
    ax0.set_xlim(0.9, 3.0)
    ax0.set_xlabel("Occurrence per future (%)")
    ax0.set_title("A  Rate and whole-matrix 95% CI", loc="left", color=INK)
    ax0.grid(axis="x", color=GRID, linewidth=0.8)
    ax0.spines[["top", "right"]].set_visible(False)

    bars = ax1.barh(y, [row["event_matrices"] for row in rows], color=colors, height=0.62)
    for bar, row in zip(bars, rows):
        ax1.text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{row['event_matrices']}/200",
            va="center",
            fontsize=9.3,
            color=INK,
        )
    ax1.set_xlim(0, 165)
    ax1.tick_params(labelleft=False)
    ax1.set_xlabel("Matrices with ≥1 event")
    ax1.set_title("B  Matrix breadth", loc="left", color=INK)
    ax1.grid(axis="x", color=GRID, linewidth=0.8)
    ax1.spines[["top", "right", "left"]].set_visible(False)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("C  Frozen endpoint funnel", loc="left", color=INK)
    stage_ranges = [
        (
            "First break within F32",
            min(
                *(cell["break_rate"] for cell in codex_stages.values()),
                *(cell["frac_break_within"] for cell in fable_stages.values()),
            ),
            max(
                *(cell["break_rate"] for cell in codex_stages.values()),
                *(cell["frac_break_within"] for cell in fable_stages.values()),
            ),
            BLUE,
        ),
        (
            "Post-break run of eight",
            min(
                *(cell["any_run8_rate"] for cell in codex_stages.values()),
                *(cell["frac_run8_after_break"] for cell in fable_stages.values()),
            ),
            max(
                *(cell["any_run8_rate"] for cell in codex_stages.values()),
                *(cell["frac_run8_after_break"] for cell in fable_stages.values()),
            ),
            TEAL,
        ),
        (
            "All 28 daughter pairs H > 0.9\nand every daughter is\nH ≤ 0.85 from old anchor",
            min(row["rate"] for row in rows),
            max(row["rate"] for row in rows),
            ORANGE,
        ),
    ]
    widths = [0.94, 0.76, 0.56]
    ys = [0.72, 0.43, 0.12]
    for i, ((label, low, high, color), width, ypos) in enumerate(
        zip(stage_ranges, widths, ys)
    ):
        x0 = (1 - width) / 2
        box = FancyBboxPatch(
            (x0, ypos),
            width,
            0.19,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            facecolor="white",
            edgecolor=color,
            linewidth=2,
        )
        ax2.add_patch(box)
        ax2.text(
            0.5,
            ypos + 0.125,
            label,
            ha="center",
            va="center",
            fontsize=10.3 if i < 2 else 8.7,
            fontweight="bold",
            color=INK,
        )
        ax2.text(
            0.5,
            ypos + 0.045,
            f"{100 * low:.1f}-{100 * high:.1f}% of futures",
            ha="center",
            va="center",
            fontsize=11,
            color=color,
        )
        if i < 2:
            ax2.add_patch(
                FancyArrowPatch(
                    (0.5, ypos - 0.015),
                    (0.5, ys[i + 1] + 0.215),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=1.4,
                    color=MUTED,
                )
            )
    ax2.text(
        0.5,
        0.965,
        "Successive selected lineage daughters - not siblings",
        ha="center",
        va="top",
        fontsize=9.7,
        color=MUTED,
    )
    save(fig, "figure6_strict8_occurrence.png")


def generative_null_summary() -> list[dict[str, Any]]:
    """Return candidate-separated GN1 rates and transported-rule effects."""
    metrics = load_json(CODEX_GENERATIVE_NULLS)
    summaries = {
        (row["mechanism"], row["candidate"]): row
        for row in metrics["cell_summaries"]
    }
    interventions = {
        (row["mechanism"], row["candidate"]): row
        for row in metrics["intervention"]
    }
    mechanisms = (
        "NATURAL_GARD",
        "HOMOGENEOUS_GENERATIVE",
        "COUPLING_DERANGED",
        "FISSION_ONLY_GENERATIVE",
    )
    rows: list[dict[str, Any]] = []
    for mechanism in mechanisms:
        for candidate in ("02", "03"):
            summary = summaries[(mechanism, candidate)]
            intervention = interventions[(mechanism, candidate)]["up_minus_down"]
            rows.append(
                {
                    "mechanism": mechanism,
                    "candidate": candidate,
                    "f12": summary["f12"]["mean"],
                    "f12_ci": summary["f12"]["ci95"],
                    "strict": summary["strict_all8"]["mean"],
                    "strict_ci": summary["strict_all8"]["ci95"],
                    "rule": intervention["effect"],
                    "rule_ci": intervention["ci95"],
                }
            )
    return rows


def figure_7_generative_nulls() -> None:
    rows = generative_null_summary()
    assert len(rows) == 8
    mechanisms = (
        "NATURAL_GARD",
        "HOMOGENEOUS_GENERATIVE",
        "COUPLING_DERANGED",
        "FISSION_ONLY_GENERATIVE",
    )
    labels = ("Natural\nGARD", "Homogeneous", "Coupling\nderanged", "Fission\nonly")
    positions = np.arange(len(mechanisms), dtype=np.float64)
    offsets = {"02": -0.09, "03": 0.09}
    colors = {"02": BLUE, "03": ORANGE}
    markers = {"02": "o", "03": "s"}

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 6.4), constrained_layout=True)
    fig.suptitle(
        "Generative nulls separate event occurrence from transported control",
        fontsize=19,
        fontweight="bold",
        color=NAVY,
    )

    panels = (
        ("f12", "f12_ci", "F12 break-and-renewal (%)", (0, 61), "A  Local renewal"),
        ("strict", "strict_ci", "Strict coherent-eight (%)", (0, 6.0), "B  Coherent episodes"),
        (
            "rule",
            "rule_ci",
            "F12-risk-raising minus\nrisk-lowering (percentage points)",
            (-3.5, 12.5),
            "C  Transported edit ordering",
        ),
    )
    for panel_index, (value_key, ci_key, ylabel, ylim, title) in enumerate(panels):
        ax = axes[panel_index]
        if value_key == "rule":
            ax.axhspan(-2.5, 2.5, color=LIGHT, zorder=0)
            ax.axhline(0, color=INK, linewidth=1)
        for mechanism_index, mechanism in enumerate(mechanisms):
            for candidate in ("02", "03"):
                row = next(
                    item
                    for item in rows
                    if item["mechanism"] == mechanism
                    and item["candidate"] == candidate
                )
                value = 100.0 * row[value_key]
                low, high = [100.0 * bound for bound in row[ci_key]]
                ax.errorbar(
                    mechanism_index + offsets[candidate],
                    value,
                    yerr=[[value - low], [high - value]],
                    fmt=markers[candidate],
                    color=colors[candidate],
                    ecolor=colors[candidate],
                    elinewidth=1.8,
                    capsize=3.5,
                    markersize=7.5,
                    zorder=3,
                )
        ax.set_xticks(positions, labels)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=12.5)
        ax.set_title(title, color=INK, loc="left", fontsize=13.5)
        style_axis(ax)
        ax.tick_params(axis="both", labelsize=11.5)
    axes[2].text(
        0.98,
        0.04,
        "Shading: preregistered ±2.5 pp\nequivalence region for null rule effects",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color=MUTED,
    )
    axes[0].legend(
        handles=[
            plt.Line2D(
                [0], [0], marker="o", color=BLUE, linestyle="none", label="T1-02"
            ),
            plt.Line2D(
                [0], [0], marker="s", color=ORANGE, linestyle="none", label="T1-03"
            ),
        ],
        frameon=False,
        loc="upper left",
        fontsize=11,
    )
    save(fig, "figure8_generative_nulls.png")


def causal_decomposition_summary() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return sealed composite and stage-specific intervention contrasts."""
    strength = load_json(CODEX_STRENGTH)
    recovery = load_json(CODEX_RECOVERY)
    confirmation = load_json(CODEX_NETWORK_CONFIRMATION)
    molecular_resistance = load_json(CODEX_MOLECULAR_RESISTANCE)
    molecular_recovery = load_json(CODEX_MOLECULAR_RECOVERY)
    output: dict[str, dict[str, list[dict[str, Any]]]] = {
        "strength": {"f12": [], "f6": [], "recovery": []},
        "molecular": {"f6": [], "recovery": []},
        "arrangement": {"f12": [], "f6": [], "recovery": []},
    }

    for cell in confirmation["cells"]:
        target = cell["contrasts"]["up_minus_down"]
        common = {
            "candidate": cell["candidate"],
            "half": cell["branch_half"],
        }
        output["strength"]["f12"].append(
            {
                **common,
                "estimate": target["estimate"],
                "ci": target["bootstrap_ci95"],
            }
        )

    for cell in confirmation["topology"]["cells"]:
        output["arrangement"]["f12"].append(
            {
                "candidate": cell["candidate"],
                "half": cell["branch_half"],
                "estimate": cell["estimate"],
                "ci": cell["bootstrap_ci95"],
            }
        )

    for cell in strength["resistance"]["cells"]:
        target = cell["target_loosen_minus_tighten"]
        arrangement = cell["neutral_minus_noop"]
        common = {
            "candidate": cell["candidate"],
            "half": cell["branch_half"],
        }
        output["strength"]["f6"].append(
            {
                **common,
                "estimate": target["estimate"],
                "ci": target["bootstrap_ci95"],
            }
        )
        output["arrangement"]["f6"].append(
            {
                **common,
                "estimate": arrangement["estimate"],
                "ci": arrangement["bootstrap_ci95"],
            }
        )

    for cell in recovery["cells"]:
        target = cell["strength_tighten_minus_loosen"]
        arrangement = cell["topology_minus_noop"]
        common = {
            "candidate": cell["candidate"],
            "half": cell["branch_half"],
        }
        output["strength"]["recovery"].append(
            {
                **common,
                "estimate": target["estimate"],
                "ci": target["bootstrap_ci95"],
            }
        )
        output["arrangement"]["recovery"].append(
            {
                **common,
                "estimate": arrangement["estimate"],
                "ci": arrangement["bootstrap_ci95"],
            }
        )

    for endpoint, artifact in (
        ("f6", molecular_resistance),
        ("recovery", molecular_recovery),
    ):
        for cell in artifact["cells"]:
            target = cell["contrasts"]["up_minus_down"]
            output["molecular"][endpoint].append(
                {
                    "candidate": cell["candidate"],
                    "half": cell["branch_half"],
                    "estimate": target["estimate"],
                    "ci": target["bootstrap_ci95"],
                }
            )
    return output


def repeated_feedback_summary() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the sealed CR7 60-fission and active-extension results."""
    primary = load_json(CODEX_CLOSED_LOOP)
    extension = load_json(CODEX_CLOSED_LOOP_EXTENSION)
    expected_primary = {
        "MODEL_UP", "RULE_UP", "RANDOM", "NOOP", "RULE_DOWN", "MODEL_DOWN"
    }
    expected_extension = {"NOOP", "RULE_DOWN", "MODEL_DOWN"}
    if not (
        primary["complete_cr7_60_fission_gate"]
        and primary["complete_exact_replay"]
        and primary["noop_callback_plain_bitwise_exact"]
    ):
        raise ValueError("CR7 primary artifact does not retain the passed gate contract")
    primary_candidates = {row["candidate"]: row for row in primary["candidates"]}
    extension_candidates = {row["candidate"]: row for row in extension["candidates"]}
    if set(primary_candidates) != {"02", "03"} or set(extension_candidates) != {"02", "03"}:
        raise ValueError("CR7 artifacts have an unexpected candidate contract")
    if any(set(row["arm_means"]) != expected_primary for row in primary_candidates.values()):
        raise ValueError("CR7 primary artifact has an unexpected arm contract")
    if any(
        set(row["fissions_61_120"]["arm_means"]) != expected_extension
        for row in extension_candidates.values()
    ):
        raise ValueError("CR7 extension artifact has an unexpected arm contract")
    if not (
        extension["launched_because_primary_gate_passed"]
        and extension["active_feedback_not_passive_persistence"]
    ):
        raise ValueError("CR7 extension artifact does not retain its active-control boundary")
    return primary_candidates, extension_candidates


def figure_6_interventions() -> None:
    feedback_primary, feedback_extension = repeated_feedback_summary()
    decomposition = causal_decomposition_summary()
    assert all(
        len(decomposition[family][endpoint]) == 4
        for family in ("strength", "arrangement")
        for endpoint in ("f12", "f6", "recovery")
    )
    assert all(
        len(decomposition["molecular"][endpoint]) == 4
        for endpoint in ("f6", "recovery")
    )
    fig, ax = plt.subplots(figsize=(8.3, 7.2), constrained_layout=True)
    fig.suptitle(
        "Repeated molecular feedback maintained high inherited-boundary frequency",
        fontsize=20,
        fontweight="bold",
        color=NAVY,
    )

    # The sealed CR7 repeated-feedback confirmation. The x order runs
    # from F12-risk-raising to F12-risk-lowering but remains categorical.
    arms = ("MODEL_UP", "RULE_UP", "RANDOM", "NOOP", "RULE_DOWN", "MODEL_DOWN")
    labels = (
        "Predictor\nF12 risk ↑",
        "Rule\nF12 risk ↑",
        "Random",
        "No-op",
        "Rule\nF12 risk ↓",
        "Predictor\nF12 risk ↓",
    )
    positions = np.arange(len(arms), dtype=np.float64)
    candidate_offsets = {"02": -0.11, "03": 0.11}
    for candidate in ("02", "03"):
        color = BLUE if candidate == "02" else ORANGE
        primary_stats = feedback_primary[candidate]["arm_means"]
        y = np.asarray([primary_stats[arm]["inherited_fraction"]["mean"] for arm in arms])
        lows = np.asarray(
            [primary_stats[arm]["inherited_fraction"]["bootstrap_ci95"][0] for arm in arms]
        )
        highs = np.asarray(
            [primary_stats[arm]["inherited_fraction"]["bootstrap_ci95"][1] for arm in arms]
        )
        ax.errorbar(
            positions + candidate_offsets[candidate],
            y,
            yerr=[y - lows, highs - y],
            fmt="o",
            linestyle="none",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            markersize=7,
            zorder=3,
        )

        extension_arms = ("NOOP", "RULE_DOWN", "MODEL_DOWN")
        extension_positions = np.asarray([arms.index(arm) for arm in extension_arms])
        extension_stats = feedback_extension[candidate]["fissions_61_120"]["arm_means"]
        ext_y = np.asarray(
            [extension_stats[arm]["mean_inherited_fraction"] for arm in extension_arms]
        )
        ext_lows = np.asarray(
            [extension_stats[arm]["bootstrap_ci95"][0] for arm in extension_arms]
        )
        ext_highs = np.asarray(
            [extension_stats[arm]["bootstrap_ci95"][1] for arm in extension_arms]
        )
        ax.errorbar(
            extension_positions + candidate_offsets[candidate] + 0.045,
            ext_y,
            yerr=[ext_y - ext_lows, ext_highs - ext_y],
            fmt="s",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.6,
            ecolor=color,
            elinewidth=1.25,
            capsize=2.5,
            markersize=6.5,
            zorder=4,
        )
    ax.set_xlim(-0.5, len(arms) - 0.5)
    ax.set_ylim(0.70, 1.015)
    ax.set_xticks(positions, labels, rotation=34, ha="right")
    ax.set_ylabel("Inherited-boundary frequency", fontsize=13)
    ax.set_xlabel("One external edit after each fission", fontsize=13)
    ax.set_title("Fresh 48-matrix confirmation", loc="left", color=INK, fontsize=14.5)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.03,
        0.97,
        "60-fission gate passed in both candidates\n"
        "Open squares: fissions 61–120, feedback active",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        color=MUTED,
    )
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color=BLUE, linestyle="none", label="T1-02"),
            plt.Line2D([0], [0], marker="o", color=ORANGE, linestyle="none", label="T1-03"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=11,
    )

    fig.text(
        0.5,
        -0.01,
        "Points show whole-matrix estimates with 95% intervals; open squares show fissions 61–120 with feedback still active.",
        ha="center",
        fontsize=12.5,
        color=MUTED,
    )
    save(fig, "figure12_repeated_feedback.png")

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), constrained_layout=True)
    fig.suptitle(
        "One-shot composite and stage controls",
        fontsize=18,
        fontweight="bold",
        color=NAVY,
    )

    endpoint_specs = [
        ("f12", "F12 break + renewal"),
        ("f6", "F6 first break"),
        ("recovery", "F8 post-break recovery"),
    ]
    process_specs = [
        (
            "strength",
            "f12",
            "F12 network scaling\n(weakening − strengthening)",
        ),
        (
            "molecular",
            "f6",
            "F6 molecular edit\n(break-up − break-down)",
        ),
        (
            "strength",
            "f6",
            "F6 network scaling\n(weakening − strengthening)",
        ),
        (
            "molecular",
            "recovery",
            "F8 molecular edit\n(recovery-up − recovery-down)",
        ),
        (
            "strength",
            "recovery",
            "F8 network scaling\n(strengthening − weakening)",
        ),
    ]
    cell_offsets = {("02", "A"): -0.18, ("02", "B"): -0.06,
                    ("03", "A"): 0.06, ("03", "B"): 0.18}

    # Panel A: orient contrasts so positive means more F12/first-break events or
    # more post-break recovery, as stated in each row label.
    ax = axes[0]
    for base, (family, endpoint, _) in enumerate(process_specs):
        for row in decomposition[family][endpoint]:
            estimate = row["estimate"]
            low, high = row["ci"]
            color = BLUE if row["candidate"] == "02" else ORANGE
            marker = "o" if row["half"] == "A" else "s"
            ax.errorbar(
                estimate,
                base + cell_offsets[(row["candidate"], row["half"])],
                xerr=[[estimate - low], [high - estimate]],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1.8,
                capsize=3,
                markersize=6.5,
                zorder=3,
            )
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlim(-0.01, 0.165)
    ax.set_yticks(
        np.arange(len(process_specs)),
        [label for _, _, label in process_specs],
        fontsize=12,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Paired probability effect", fontsize=13)
    ax.set_title(
        "A  Composite and stage controls",
        loc="left",
        color=INK,
        fontsize=14.5,
    )
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel B: the fresh F12 weight-arrangement family plus the earlier registered
    # fixed-starting-throughput resistance and recovery families.
    ax = axes[1]
    for base, (endpoint, _) in enumerate(endpoint_specs):
        for row in decomposition["arrangement"][endpoint]:
            estimate = row["estimate"]
            low, high = row["ci"]
            color = BLUE if row["candidate"] == "02" else ORANGE
            marker = "o" if row["half"] == "A" else "s"
            ax.errorbar(
                estimate,
                base + cell_offsets[(row["candidate"], row["half"])],
                xerr=[[estimate - low], [high - estimate]],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1.8,
                capsize=3,
                markersize=6.5,
                zorder=3,
            )
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlim(-0.065, 0.065)
    ax.set_yticks(
        np.arange(3),
        [
            "F12 fresh arrangement\n(inconclusive)",
            "F6 first break\n(earlier directional)",
            "F8 post-break recovery\n(earlier directional)",
        ],
        fontsize=12,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Fixed-throughput rearrangement − no-op", fontsize=13)
    ax.set_title(
        "B  Network-weight arrangement",
        loc="left",
        color=INK,
        fontsize=14.5,
    )
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color=BLUE, linestyle="none", label="02/A"),
            plt.Line2D([0], [0], marker="s", color=BLUE, linestyle="none", label="02/B"),
            plt.Line2D([0], [0], marker="o", color=ORANGE, linestyle="none", label="03/A"),
            plt.Line2D([0], [0], marker="s", color=ORANGE, linestyle="none", label="03/B"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=11,
        ncol=2,
    )

    save(fig, "figure11_one_shot_controls.png")


def main() -> None:
    required = [
        EIDOSOMA_L54,
        CODEX_F12,
        FABLE_F12,
        CODEX_COHERENCE,
        FABLE_COHERENCE,
        CODEX_STRICT,
        CODEX_STRICT_DIAGNOSTIC,
        FABLE_STRICT,
        CODEX_GENERATIVE_NULLS,
        CODEX_CLOSED_LOOP,
        CODEX_CLOSED_LOOP_EXTENSION,
        CODEX_STRENGTH,
        CODEX_RECOVERY,
        CODEX_NETWORK_CONFIRMATION,
        CODEX_MOLECULAR_RESISTANCE,
        CODEX_MOLECULAR_RECOVERY,
        CALIBRATION_VIEW,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source artifacts:\n" + "\n".join(missing))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "text.color": INK,
        }
    )
    figure_1_flow()
    figure_1_evidence_architecture_v2()
    figure_2_f12()
    figure_2_f12_v2()
    figure_3_coherence()
    figure_4_strict8()
    figure_7_generative_nulls()
    figure_6_interventions()
    v2_aliases = {
        "figure11_one_shot_controls.png": "figure9_v2_one_shot_controls.png",
        "figure12_repeated_feedback.png": "figure10_v2_repeated_feedback.png",
    }
    for source_name, target_name in v2_aliases.items():
        shutil.copyfile(OUT / source_name, OUT / target_name)
    for name in (
        "figure1_replication_to_discovery.png",
        "figure1_v2_evidence_architecture.png",
        "figure3_cross_cleanroom_f12.png",
        "figure3_v2_cross_cleanroom_f12_calibration.png",
        "figure4_three_fission_coherence.png",
        "figure6_strict8_occurrence.png",
        "figure8_generative_nulls.png",
        "figure11_one_shot_controls.png",
        "figure12_repeated_feedback.png",
        "figure9_v2_one_shot_controls.png",
        "figure10_v2_repeated_feedback.png",
    ):
        print(OUT / name)


if __name__ == "__main__":
    main()
