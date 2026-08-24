#!/usr/bin/env python3
"""Generate Figure 13 from retained Chapter 5 result bundles.

The figure places the two clean-room implementations side by side without
pooling candidates, replicates, formulas, or clocks. It writes only derived
figure files and provenance into ``figures/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

FABLE_CONFIRM = ROOT / (
    "replicators.13.8.2026.fable/replication/results_phir_confirm/"
    "phir_confirm_results.json"
)
FABLE_FORESIGHT = ROOT / (
    "replicators.13.8.2026.fable/replication/results_phir_foresight/"
    "foresight_results.json"
)
CODEX_PILOT = ROOT / "replicators.13.8.2026.codex/results/phir_ch5_pilot/primary_metrics.json"
CODEX_BRIDGE = ROOT / (
    "replicators.13.8.2026.codex/results/phir_window_bridge24/"
    "primary_metrics.json"
)
PX3_AUDIT = ROOT / (
    "replicators.13.8.2026.codex/results/phir_extension/"
    "px3_posthoc_support_audit/audit.json"
)

NAVY = "#17324D"
BLUE = "#2C78B8"
ORANGE = "#D66A3A"
INK = "#263238"
MUTED = "#66727A"
GRID = "#D9E0E5"

TEST_STYLE = {
    "Clean-room test 1": {"colour": BLUE, "short": "Test 1"},
    "Clean-room test 2": {"colour": ORANGE, "short": "Test 2"},
}
CANDIDATE_MARKER = {"02": "o", "03": "s"}

plt.rcParams.update(
    {
        "font.size": 11.0,
        "axes.labelsize": 11.0,
        "axes.titlesize": 13.0,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.4,
        "legend.fontsize": 9.8,
    }
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fable_rows(
    result: dict[str, Any],
    test_key: str,
    value_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in ("02", "03"):
        cell = result[candidate]["tests"][test_key]
        rows.append(
            {
                "test": "Clean-room test 2",
                "candidate": candidate,
                "replicate": None,
                "label": f"Test 2 · T2-{candidate}",
                "value": float(cell[value_key]),
                "lower": float(cell["ci"][0]),
                "upper": float(cell["ci"][1]),
            }
        )
    return rows


def fable_generational_typeset_rows(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in ("02", "03"):
        cell = result[candidate]["N3"]["gen_printed"]
        rows.append(
            {
                "test": "Clean-room test 2",
                "candidate": candidate,
                "replicate": None,
                "label": f"Test 2 · T2-{candidate}",
                "value": float(cell["diff"]),
                "lower": float(cell["ci"][0]),
                "upper": float(cell["ci"][1]),
            }
        )
    return rows


def codex_rows(result: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cells = [cell for cell in result["cells"] if cell["metric"] == metric]
    cells.sort(key=lambda cell: (cell["candidate"], int(cell["replicate"])))
    for cell in cells:
        candidate = str(cell["candidate"])
        replicate = int(cell["replicate"]) + 1
        rows.append(
            {
                "test": "Clean-room test 1",
                "candidate": candidate,
                "replicate": replicate,
                "label": f"Test 1 · T1-{candidate} · replicate {replicate}",
                "value": float(cell["effect"]),
                "lower": float(cell["ci95"][0]),
                "upper": float(cell["ci95"][1]),
            }
        )
    if not rows:
        raise RuntimeError(f"No Codex cells found for {metric}")
    return rows


def interval_panel(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    title: str,
    subtitle: str,
    xlabel: str,
    panel_letter: str,
) -> None:
    positions = np.arange(len(rows), dtype=float)
    for position, row in zip(positions, rows, strict=True):
        style = TEST_STYLE[row["test"]]
        value = row["value"]
        axis.errorbar(
            value,
            position,
            xerr=[[value - row["lower"]], [row["upper"] - value]],
            fmt=CANDIDATE_MARKER[row["candidate"]],
            markersize=7.5,
            markerfacecolor=style["colour"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            ecolor=style["colour"],
            elinewidth=2.1,
            capsize=3.2,
            zorder=3,
        )

    axis.set_yticks(positions, [row["label"] for row in rows])
    axis.set_ylim(len(rows) - 0.5, -0.5)
    axis.axvline(0, color=MUTED, linewidth=1.05, linestyle="--", zorder=0)
    axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED, labelcolor=INK)
    axis.set_xlabel(xlabel, color=INK, labelpad=8)
    axis.set_title(title, loc="left", color=NAVY, fontweight="bold", pad=18)
    axis.text(
        0,
        1.01,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=9.6,
    )
    axis.text(
        -0.12,
        1.13,
        panel_letter,
        transform=axis.transAxes,
        color=NAVY,
        fontsize=15,
        fontweight="bold",
        va="top",
    )


def support_panel(axis: plt.Axes, audit: dict[str, Any]) -> None:
    rows = audit["support_remeasurement"]
    positions = np.arange(len(rows), dtype=float)
    pairs = [int(row["pairs_per_block"]) for row in rows]

    for candidate, linestyle in (("02", "-"), ("03", "--")):
        values = [float(row[f"candidate_{candidate}_up_minus_down"]) for row in rows]
        axis.plot(
            positions,
            values,
            color=BLUE,
            linestyle=linestyle,
            linewidth=2.4,
            marker=CANDIDATE_MARKER[candidate],
            markersize=7.5,
            markerfacecolor=BLUE,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=f"Test 1 · T1-{candidate}",
            zorder=3,
        )
        for position, value in zip(positions, values, strict=True):
            axis.annotate(
                f"{value:+.2f}",
                (position, value),
                xytext=(0, 9 if candidate == "02" else -16),
                textcoords="offset points",
                ha="center",
                color=INK,
                fontsize=9.0,
            )

    axis.axhline(0, color=MUTED, linewidth=1.05, linestyle="--", zorder=0)
    axis.set_ylim(-4.2, 14.0)
    axis.set_xticks(positions, [f"{pairs_}\npairs" for pairs_ in pairs])
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED, labelcolor=INK)
    axis.set_ylabel("Full-block PHI_UP − PHI_DOWN", color=INK, labelpad=8)
    axis.set_xlabel("Transition support per score block", color=INK, labelpad=8)
    axis.set_title(
        "Full-block calibration changed with support",
        loc="left",
        color=NAVY,
        fontweight="bold",
        pad=18,
    )
    axis.text(
        0,
        1.01,
        "Registered direct-control audit; same sealed futures",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=9.6,
    )
    axis.text(
        -0.12,
        1.13,
        "D",
        transform=axis.transAxes,
        color=NAVY,
        fontsize=15,
        fontweight="bold",
        va="top",
    )
    axis.legend(frameon=False, loc="upper right", fontsize=9.2)


def main() -> None:
    fable_confirm = load_json(FABLE_CONFIRM)
    fable_foresight = load_json(FABLE_FORESIGHT)
    codex_bridge = load_json(CODEX_BRIDGE)
    codex_pilot = load_json(CODEX_PILOT)
    px3_audit = load_json(PX3_AUDIT)

    heredity_rows = fable_rows(
        fable_confirm,
        "C3_phstab_minus_phdestab_INHERIT",
        "diff",
    ) + codex_rows(codex_bridge, "inherited_31_60")
    revised_rows = fable_rows(
        fable_confirm,
        "C1_phstab_minus_phdestab_PHICODE",
        "diff",
    ) + codex_rows(codex_bridge, "pooled20_clr_revised")
    macro_typeset_rows = fable_generational_typeset_rows(fable_foresight)
    full_typeset_rows = codex_rows(codex_bridge, "pooled20_clr_full_typeset")

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(17.2, 11.6),
        gridspec_kw={"wspace": 0.48, "hspace": 0.58},
    )

    interval_panel(
        axes[0, 0],
        heredity_rows,
        "The heredity dial moved",
        "Shared validity outcome",
        "Raising − lowering controller contrast\ninherited-boundary frequency",
        "A",
    )
    interval_panel(
        axes[0, 1],
        revised_rows,
        "Public PhiRL revised $\\Phi_R$ responded",
        "Same named scalar, opposite directions",
        "Boundary-frequency-raising − lowering\nrevised $\\Phi_R$",
        "B",
    )
    interval_panel(
        axes[1, 0],
        macro_typeset_rows,
        "Generational macro-typeset response",
        "Clean-room test 2",
        "Boundary-frequency-raising − lowering\nmacro-typeset $\\Phi$-r",
        "C",
    )
    support_panel(axes[1, 1], px3_audit)

    figure.suptitle(
        "Several $\\Phi$-r-family readings respond, but no calibration is shared",
        x=0.055,
        y=0.992,
        ha="left",
        color=NAVY,
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.958,
        "Contract- and replicate-specific effects with 95% whole-matrix bootstrap confidence intervals",
        ha="left",
        va="top",
        color=MUTED,
        fontsize=11,
    )

    legend_handles = [
        plt.Line2D([], [], color=BLUE, linewidth=2.5, label="Clean-room test 1"),
        plt.Line2D([], [], color=ORANGE, linewidth=2.5, label="Clean-room test 2"),
        plt.Line2D(
            [],
            [],
            color=INK,
            marker="o",
            linestyle="none",
            markerfacecolor=INK,
            label="T1-02 or T2-02",
        ),
        plt.Line2D(
            [],
            [],
            color=INK,
            marker="s",
            linestyle="none",
            markerfacecolor=INK,
            label="T1-03 or T2-03",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.975, 0.987),
        frameon=False,
        ncol=2,
        handletextpad=0.55,
        columnspacing=1.2,
    )
    figure.text(
        0.055,
        0.025,
        (
            "Zero denotes no arm difference. Formula, clock, window, partition, preprocessing, and transition support are part of the "
            "measurement contract; raw magnitudes are comparable only within a panel. T1/T2 aliases identify each codebase contract."
        ),
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=9.6,
    )
    figure.subplots_adjust(left=0.19, right=0.975, top=0.88, bottom=0.105)

    OUT.mkdir(parents=True, exist_ok=True)
    png_path = OUT / "figure13_phir_gauges.png"
    svg_path = OUT / "figure13_phir_gauges.svg"
    figure.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )

    pilot_typeset = [
        cell
        for cell in codex_pilot["bridge"]["cells"]
        if cell["metric"] == "molecular_typeset" and cell["contrast"] == "model"
    ]
    provenance = {
        "status": "evidence_current_through_2026-08-21",
        "sources": [
            str(FABLE_CONFIRM.relative_to(ROOT)),
            str(FABLE_FORESIGHT.relative_to(ROOT)),
            str(CODEX_PILOT.relative_to(ROOT)),
            str(CODEX_BRIDGE.relative_to(ROOT)),
            str(PX3_AUDIT.relative_to(ROOT)),
        ],
        "panels": {
            "A": "Inherited-boundary-frequency-raising minus -lowering control contrast in inherited-boundary frequency in both tests",
            "B": "Inherited-boundary-frequency-raising minus -lowering control contrast in public PhiRL revised Phi_R in both tests",
            "C": "Clean-room-test-2 generational macro-typeset response",
            "D": "Post-hoc support audit of retained PX3 full-block PHI_UP-minus-PHI_DOWN on the same sealed futures",
        },
        "boundary": (
            "Some readings respond under heredity control, but the instruments are not numerically interchangeable. "
            "PX3's registered direct-control failure remains controlling; its support audit is diagnostic and cannot rescue it."
        ),
        "values": {
            "heredity": heredity_rows,
            "revised_phi_r": revised_rows,
            "macro_typeset": macro_typeset_rows,
            "full_typeset_unplotted": full_typeset_rows,
            "px3_support_audit": px3_audit["support_remeasurement"],
            "codex_pilot_full_typeset_unplotted": pilot_typeset,
        },
    }
    with (OUT / "figure13_phir_gauges_provenance.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
