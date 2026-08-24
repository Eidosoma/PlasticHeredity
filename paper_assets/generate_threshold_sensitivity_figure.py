#!/usr/bin/env python3
"""Generate Figure 7 from the completed endpoint-sensitivity records.

The script reads only verified outputs from the reviewer-prompted metric grid
and strict-event geometry audit. It performs no fitting, recalibration,
endpoint selection, or inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.text import Text


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
EXTENSION = ROOT / (
    "paperFinalReplicationAttempt.15.8.2026.codex/"
    "reviewer_threshold_metric_sensitivity_extension/artifacts/output"
)
GEOMETRY = ROOT / (
    "paperFinalReplicationAttempt.15.8.2026.codex/"
    "reviewer_strict_event_geometry_audit/artifacts/output"
)

F12_CSV = EXTENSION / "f12_sensitivity.csv"
CR1_CSV = EXTENSION / "cr1_cosine_sensitivity.csv"
EXTENSION_VERIFICATION = EXTENSION / "verification.json"
EVENT_POWER_CSV = GEOMETRY / "event_power.csv"
EVENT_OVERLAP_CSV = GEOMETRY / "event_overlap.csv"
PREDICTION_CSV = GEOMETRY / "prediction_comparisons.csv"
GEOMETRY_VERIFICATION = GEOMETRY / "verification_audit.json"

PNG = OUT / "figure7_endpoint_sensitivity.png"
PROVENANCE = OUT / "figure7_endpoint_sensitivity_provenance.json"

THRESHOLDS = [0.85, 0.875, 0.90, 0.925, 0.95]
HORIZONS = [8, 10, 12, 16]
METRICS = ["cosine", "bray_curtis"]

NAVY = "#17324D"
BLUE = "#2C78B8"
TEAL = "#1B998B"
GOLD = "#D89B2B"
ORANGE = "#D66A3A"
INK = "#263238"
MUTED = "#66727A"
GRID = "#D9E0E5"


plt.rcParams.update(
    {
        "font.size": 10.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.8,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_minimum(
    rows: list[dict[str, str]],
    *,
    value_fields: tuple[str, ...],
    metric: str | None = None,
    contrast: str | None = None,
) -> np.ndarray:
    grouped: dict[tuple[float, int], list[float]] = defaultdict(list)
    for row in rows:
        if metric is not None and row.get("metric") != metric:
            continue
        if contrast is not None and row.get("contrast") != contrast:
            continue
        key = (
            float(row["inheritance_threshold_source_cosine"]),
            int(row["horizon_fissions"]),
        )
        grouped[key].extend(float(row[field]) for field in value_fields)

    surface = np.full((len(THRESHOLDS), len(HORIZONS)), np.nan)
    for yi, threshold in enumerate(THRESHOLDS):
        for xi, horizon in enumerate(HORIZONS):
            values = grouped[(threshold, horizon)]
            if not values:
                raise RuntimeError(
                    f"Missing grid cell threshold={threshold}, horizon={horizon}"
                )
            surface[yi, xi] = min(values)
    return surface


def format_threshold(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def configure_heatmap(ax: plt.Axes, surface: np.ndarray, *, digits: int = 3) -> None:
    scale = float(np.nanmax(np.abs(surface))) or 1.0
    for yi in range(surface.shape[0]):
        for xi in range(surface.shape[1]):
            value = float(surface[yi, xi])
            colour = "white" if abs(value) > 0.60 * scale else INK
            ax.text(
                xi,
                yi,
                f"{value:.{digits}f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=colour,
            )
    ax.set_xticks(range(len(HORIZONS)), [f"F{value}" for value in HORIZONS])
    ax.set_yticks(
        range(len(THRESHOLDS)),
        [format_threshold(value) for value in THRESHOLDS],
    )
    ax.add_patch(
        Rectangle(
            (HORIZONS.index(12) - 0.48, THRESHOLDS.index(0.90) - 0.48),
            0.96,
            0.96,
            fill=False,
            edgecolor=INK,
            linewidth=1.8,
        )
    )


def main() -> None:
    extension_check = json.loads(EXTENSION_VERIFICATION.read_text(encoding="utf-8"))
    geometry_check = json.loads(GEOMETRY_VERIFICATION.read_text(encoding="utf-8"))
    if not extension_check.get("all_checks_passed"):
        raise RuntimeError("Metric-sensitivity verification did not pass")
    if not geometry_check.get("all_checks_passed"):
        raise RuntimeError("Strict-event geometry verification did not pass")

    f12_rows = read_csv(F12_CSV)
    cr1_rows = read_csv(CR1_CSV)
    power_rows = read_csv(EVENT_POWER_CSV)
    overlap_rows = read_csv(EVENT_OVERLAP_CSV)
    prediction_rows = read_csv(PREDICTION_CSV)

    f12_surfaces = {
        metric: grid_minimum(
            f12_rows,
            value_fields=("log_loss_gain_A", "log_loss_gain_B"),
            metric=metric,
        )
        for metric in METRICS
    }
    intervention_surface = grid_minimum(
        cr1_rows,
        value_fields=("ci95_lower",),
        contrast="MODEL_UP_minus_MODEL_DOWN",
    )

    fig = plt.figure(figsize=(13.4, 9.3))
    outer = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.97,
        bottom=0.08,
        top=0.84,
        wspace=0.28,
        hspace=0.38,
    )

    panel_a = outer[0, 0].subgridspec(1, 2, wspace=0.18)
    limit = max(float(np.nanmax(np.abs(value))) for value in f12_surfaces.values())
    heatmap_image = None
    for index, metric in enumerate(METRICS):
        ax = fig.add_subplot(panel_a[0, index])
        heatmap_image = ax.imshow(
            f12_surfaces[metric],
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        configure_heatmap(ax, f12_surfaces[metric])
        ax.set_title("Cosine" if metric == "cosine" else "Bray–Curtis", fontweight="bold")
        ax.set_xlabel("Horizon")
        if index == 0:
            ax.set_ylabel("Source cosine cutoff")
        else:
            ax.set_yticklabels([])
    if heatmap_image is None:
        raise RuntimeError("Panel A was not generated")
    colourbar = fig.colorbar(
        heatmap_image,
        ax=[fig.axes[0], fig.axes[1]],
        fraction=0.035,
        pad=0.025,
    )
    colourbar.set_label("Worst composite-over-history gain")
    fig.text(
        0.07,
        0.88,
        "A  F12 frozen-predictor robustness (628/640 gains positive)",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )

    ax = fig.add_subplot(outer[0, 1])
    image_b = ax.imshow(
        intervention_surface,
        origin="lower",
        aspect="auto",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=float(np.nanmax(intervention_surface)),
    )
    configure_heatmap(ax, intervention_surface)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Cosine cutoff")
    ax.set_title(
        "B  Molecular contrast: worst 95% lower bound",
        loc="left",
        fontweight="bold",
    )
    colourbar = fig.colorbar(image_b, ax=ax, fraction=0.046, pad=0.03)
    colourbar.set_label("Higher − lower predicted F12 probability")

    confirmation_power = {
        (row["spec"], row["candidate"]): float(row["prevalence"])
        for row in power_rows
        if row["cohort"] == "confirmation"
    }
    specs = ["cosine_registered", "bray_global", "bray_relation_specific"]
    labels = ["Cosine\nregistered", "Bray\nglobal", "Bray\nrelation-specific"]
    colours = [BLUE, ORANGE, TEAL]
    x = np.arange(len(specs), dtype=float)
    width = 0.34
    ax = fig.add_subplot(outer[1, 0])
    for offset, candidate in ((-width / 2, "02"), (width / 2, "03")):
        values = [100.0 * confirmation_power[(spec, candidate)] for spec in specs]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=f"candidate {candidate}",
            color=colours,
            edgecolor=INK if candidate == "03" else "white",
            linewidth=1.1 if candidate == "03" else 0.6,
            alpha=1.0 if candidate == "02" else 0.72,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Confirmation prevalence")
    ax.set_ylim(0.0, 2.35)
    ax.set_title("C  Strict-event prevalence depends on metric", loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(frameon=False, loc="upper right")

    jaccard = {}
    for row in overlap_rows:
        if not row["jaccard"]:
            continue
        key = (row["candidate"], row["left"], row["right"])
        jaccard[key] = float(row["jaccard"])
    target_matched_relation_passes = sum(
        row["evaluation_target"] == "bray_relation_specific"
        and row["training_target"] == "bray_relation_specific"
        and row["target_matched"] == "True"
        and row["passes_exploratory_gate"] == "True"
        for row in prediction_rows
    )
    ax = fig.add_subplot(outer[1, 1])
    comparison_specs = ["bray_global", "bray_relation_specific"]
    comparison_labels = ["Global mapping", "Relation-specific"]
    x2 = np.arange(len(comparison_specs), dtype=float)
    for offset, candidate, colour in (
        (-width / 2, "02", BLUE),
        (width / 2, "03", ORANGE),
    ):
        values = [
            jaccard[(candidate, "cosine_registered", spec)]
            for spec in comparison_specs
        ]
        bars = ax.bar(x2 + offset, values, width, color=colour, label=f"candidate {candidate}")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.3,
            )
    ax.set_xticks(x2, comparison_labels)
    ax.set_ylabel("Jaccard overlap with cosine labels")
    ax.set_ylim(0.0, 0.23)
    ax.set_title("D  Better mapping does not establish equivalence", loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.96,
        f"Relation-specific target-matched predictor: {target_matched_relation_passes}/4 pass",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.7,
        color=MUTED,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": GRID},
    )

    fig.suptitle(
        "Post-hoc endpoint sensitivity: robust F12, metric-dependent strict eight",
        fontsize=16.0,
        fontweight="bold",
        color=NAVY,
        y=0.975,
    )
    for text_artist in fig.findobj(match=Text):
        x_position, y_position = text_artist.get_position()
        if not (np.isfinite(x_position) and np.isfinite(y_position)):
            raise RuntimeError(
                f"Non-finite text position: {text_artist.get_text()!r} "
                f"at {(x_position, y_position)!r}"
            )

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    sources = (
        F12_CSV,
        CR1_CSV,
        EXTENSION_VERIFICATION,
        EVENT_POWER_CSV,
        EVENT_OVERLAP_CSV,
        PREDICTION_CSV,
        GEOMETRY_VERIFICATION,
    )
    provenance: dict[str, Any] = {
        "figure": PNG.name,
        "analysis_status": "reviewer_prompted_post_hoc",
        "verification_all_checks_passed": True,
        "sources": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sources
        },
        "panel_a": "minimum point gain across candidates, fixed halves, and renewal lengths",
        "panel_b": "minimum CR1 95% lower bound across candidates, fixed halves, and renewal lengths",
        "panel_c": "confirmation prevalence under registered cosine, global-mapped Bray-Curtis, and relation-specific Bray-Curtis",
        "panel_d": "branch-label Jaccard overlap with registered cosine; target-matched relation-specific predictor pass count",
        "registered_outline": "source threshold 0.90 and F12; panel A aggregates renewal lengths 2-5",
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
