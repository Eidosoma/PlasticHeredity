#!/usr/bin/env python3
"""Generate data-backed trajectory figures for the plastic-heredity paper.

The figures use deterministic, auditable exemplar selection rather than manual
choice.  The F12 example is the eligible non-coherent episode closest to the
pooled medians of five geometry/timing coordinates in the retained 145,516-row
episode audit.  The strict-eight example is the retained positive episode
closest to the corresponding archive medians among futures that contain an
earlier inherited eight-run which fails all-pairs coherence.

Scientific source bundles are read only.  Outputs are two PNG figures and a
small JSON provenance record under ``figures/``.
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
CODEX_ROOT = ROOT / "replicators.13.8.2026.codex"

F12_TABLE = CODEX_ROOT / (
    "results/episode_coherence_audit/episode_measurements.csv.gz"
)
F12_SOURCES = {
    "scaled5": ("CONF", CODEX_ROOT / "results/scaled5"),
    "MECHCONF": ("MECHCONF", CODEX_ROOT / "results/mechanistic_confirmation"),
    "MECHCONF2": ("MECHCONF2", CODEX_ROOT / "results/beta_complete_confirmation"),
}
STRICT8_ARCHIVE = ROOT / (
    "replicators.13.8.2026.fable/replication/"
    "results_strict8_occurrence/strict8_units.pkl"
)

if str(CODEX_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEX_ROOT))

from plastic_heredity.config import CANDIDATES  # noqa: E402
from plastic_heredity.episode_coherence import (  # noqa: E402
    _experiment_from_manifest,
    episode_geometry,
)
from plastic_heredity.experiment import StateCase  # noqa: E402
from plastic_heredity.seeds import derive_seed  # noqa: E402
from plastic_heredity.simulator import (  # noqa: E402
    SimulationError,
    cosine_similarity,
    generate_beta,
    generate_initial_composition,
    simulate_future_absorbing,
    simulate_lineage,
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

SIMILARITY_CMAP = LinearSegmentedColormap.from_list(
    "similarity", ["#F7FAFC", "#B7DDE8", "#2C78B8", "#17324D"]
)
COMPOSITION_CMAP = LinearSegmentedColormap.from_list(
    "composition", ["#F7FAFC", "#A7D8D0", "#1B998B", "#0D4D47"]
)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
    }
)


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=MUTED)
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)


def _save(figure: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUT / filename,
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _robust_distance(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, dict[str, Any]]:
    medians = frame[columns].median()
    scales = frame[columns].quantile(0.75) - frame[columns].quantile(0.25)
    scales = scales.mask(scales == 0, 1.0)
    score = (((frame[columns] - medians) / scales) ** 2).sum(axis=1)
    return score, {
        "coordinates": columns,
        "medians": {name: float(medians[name]) for name in columns},
        "interquartile_ranges": {name: float(scales[name]) for name in columns},
    }


def _select_f12_exemplar() -> tuple[pd.Series, dict[str, Any]]:
    table = pd.read_csv(F12_TABLE, dtype={"candidate": str})
    columns = [
        "minimum_pairwise_daughter_similarity",
        "maximum_anchor_similarity",
        "first_break_index",
        "episode_start_index",
        "observed_inherited_run_length",
    ]
    eligible = table.loc[
        (table["completed_horizon_regenerated"] == 1)
        & (table["minimum_pairwise_daughter_similarity"] < 0.9)
        & (table["maximum_anchor_similarity"] <= 0.9)
    ].copy()
    eligible["selection_score"], rule = _robust_distance(eligible, columns)
    eligible = eligible.sort_values(
        [
            "selection_score",
            "source_cohort",
            "candidate",
            "matrix_id",
            "landmark",
            "branch",
        ],
        kind="stable",
    )
    selected = eligible.iloc[0]
    rule.update(
        {
            "population": int(len(table)),
            "eligible_population": int(len(eligible)),
            "eligibility": (
                "completed F12; first qualifying episode has minimum daughter-pair "
                "H < 0.9 and every episode daughter H <= 0.9 to the old anchor"
            ),
            "tie_break": (
                "source cohort, candidate, matrix, landmark, then branch"
            ),
        }
    )
    return selected, rule


def _build_single_case(
    experiment: Any,
    cohort_name: str,
    candidate: str,
    matrix_id: int,
    landmark: int,
) -> StateCase:
    beta_rng = np.random.default_rng(
        derive_seed(experiment.master_seed, f"{cohort_name}.beta", matrix_id)
    )
    initial_rng = np.random.default_rng(
        derive_seed(experiment.master_seed, f"{cohort_name}.initial", matrix_id)
    )
    beta = generate_beta(experiment.gard, beta_rng)
    initial = generate_initial_composition(experiment.gard, initial_rng)
    lineage = None
    for attempt in range(100):
        path_rng = np.random.default_rng(
            derive_seed(
                experiment.master_seed,
                f"{cohort_name}.main_path",
                candidate,
                matrix_id,
                attempt,
            )
        )
        try:
            lineage = simulate_lineage(
                initial,
                beta,
                experiment.gard,
                CANDIDATES[candidate],
                path_rng,
            )
            break
        except SimulationError:
            continue
    if lineage is None:
        raise SimulationError("failed to regenerate selected exemplar lineage")
    snapshots = {snapshot.generation: snapshot for snapshot in lineage}
    return StateCase(
        state_id=f"{cohort_name}-c{candidate}-m{matrix_id:03d}-g{landmark:03d}",
        cohort=cohort_name,
        candidate=candidate,
        matrix_id=matrix_id,
        landmark=landmark,
        beta=beta,
        snapshot=snapshots[landmark],
    )


def _replay_f12(selected: pd.Series) -> tuple[list[Any], dict[str, Any]]:
    source = str(selected["source_cohort"])
    cohort_name, directory = F12_SOURCES[source]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    experiment = _experiment_from_manifest(manifest)
    candidate = str(selected["candidate"]).zfill(2)
    case = _build_single_case(
        experiment,
        cohort_name,
        candidate,
        int(selected["matrix_id"]),
        int(selected["landmark"]),
    )
    if case.state_id != selected["state_id"]:
        raise AssertionError("selected F12 state identifier did not regenerate")
    branch = int(selected["branch"])
    rng = np.random.default_rng(
        derive_seed(
            experiment.master_seed,
            f"{case.cohort}.future",
            case.candidate,
            case.matrix_id,
            case.landmark,
            branch,
        )
    )
    records, completed = simulate_future_absorbing(
        case.snapshot,
        case.beta,
        experiment.gard,
        CANDIDATES[case.candidate],
        experiment.horizon,
        rng,
    )
    geometry = episode_geometry(records, experiment.gard.inheritance_threshold)
    checks = {
        "completed_horizon": bool(completed),
        "first_break_index": geometry.first_break_index == int(selected["first_break_index"]),
        "episode_start_index": geometry.episode_start_index == int(selected["episode_start_index"]),
        "minimum_pairwise_error": abs(
            geometry.minimum_pairwise_daughter_similarity
            - float(selected["minimum_pairwise_daughter_similarity"])
        ),
        "maximum_anchor_error": abs(
            geometry.maximum_anchor_similarity
            - float(selected["maximum_anchor_similarity"])
        ),
    }
    if (
        not checks["completed_horizon"]
        or not checks["first_break_index"]
        or not checks["episode_start_index"]
        or checks["minimum_pairwise_error"] > 1e-14
        or checks["maximum_anchor_error"] > 1e-14
    ):
        raise AssertionError(f"selected F12 exemplar did not replay exactly: {checks}")
    return records, {"geometry": asdict(geometry), "replay_checks": checks}


def _strict_archive_candidates() -> list[dict[str, Any]]:
    with STRICT8_ARCHIVE.open("rb") as handle:
        units = pickle.load(handle)
    candidates: list[dict[str, Any]] = []
    for unit in units:
        futures = {
            (int(item["landmark"]), int(item["branch"])): (index, item)
            for index, item in enumerate(unit["futures"])
        }
        for archive in unit["archives"]:
            if archive.get("kind") != "positive":
                continue
            key = (int(archive["landmark"]), int(archive["branch"]))
            index, future = futures[key]
            if not future["positive"]:
                raise AssertionError("positive strict-eight archive links to a negative future")
            qualifying = next(
                values
                for values in future["eligible"]
                if int(values[0]) == int(future["primary_r"])
            )
            earlier = [
                values
                for values in future["eligible"]
                if int(values[0]) < int(future["primary_r"])
                and float(values[1]) <= 0.9
            ]
            if not earlier:
                continue
            candidates.append(
                {
                    "matrix": int(unit["matrix"]),
                    "candidate": str(unit["candidate"]),
                    "landmark": int(future["landmark"]),
                    "branch": int(future["branch"]),
                    "first_break": int(future["first_break"]),
                    "primary_start": int(future["primary_r"]),
                    "certification": int(future["cert"]),
                    "run_length": int(future["run_len"]),
                    "qualifying_minimum_pairwise": float(qualifying[1]),
                    "qualifying_maximum_anchor": float(qualifying[2]),
                    "first_eligible_start": int(earlier[0][0]),
                    "first_eligible_minimum_pairwise": float(earlier[0][1]),
                    "first_eligible_maximum_anchor": float(earlier[0][2]),
                    "eligible_windows": [
                        [int(values[0]), float(values[1]), float(values[2])]
                        for values in future["eligible"]
                    ],
                    "h": np.asarray(unit["H"][index], dtype=np.float64),
                    "old_anchor": np.asarray(archive["p_old"], dtype=np.int64),
                    "daughters": np.asarray(archive["daughters"], dtype=np.int64),
                }
            )
    return candidates


def _select_strict8_exemplar() -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = _strict_archive_candidates()
    columns = [
        "first_break",
        "primary_start",
        "run_length",
        "qualifying_minimum_pairwise",
        "qualifying_maximum_anchor",
    ]
    frame = pd.DataFrame(
        [{name: item[name] for name in columns} for item in candidates]
    )
    frame["selection_score"], rule = _robust_distance(frame, columns)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            float(frame.loc[index, "selection_score"]),
            candidates[index]["candidate"],
            candidates[index]["matrix"],
            candidates[index]["landmark"],
            candidates[index]["branch"],
        ),
    )
    selected_index = order[0]
    selected = candidates[selected_index]
    selected["selection_score"] = float(frame.loc[selected_index, "selection_score"])
    pairwise = _cosine_matrix(selected["daughters"])
    anchor = np.asarray(
        [
            cosine_similarity(selected["old_anchor"], daughter)
            for daughter in selected["daughters"]
        ],
        dtype=np.float64,
    )
    checks = {
        "minimum_pairwise_error": abs(
            _off_diagonal_minimum(pairwise)
            - selected["qualifying_minimum_pairwise"]
        ),
        "maximum_anchor_error": abs(
            float(anchor.max()) - selected["qualifying_maximum_anchor"]
        ),
        "eight_inherited_boundaries": bool(
            np.all(
                selected["h"][
                    selected["primary_start"] - 1 : selected["certification"]
                ]
                > 0.9
            )
        ),
    }
    if (
        checks["minimum_pairwise_error"] > 1e-14
        or checks["maximum_anchor_error"] > 1e-14
        or not checks["eight_inherited_boundaries"]
    ):
        raise AssertionError(f"strict-eight archive failed verification: {checks}")
    selected["pairwise"] = pairwise
    selected["anchor_similarities"] = anchor
    rule.update(
        {
            "archive_population": len(candidates),
            "eligibility": (
                "retained positive strict-eight episode with an earlier inherited "
                "eight-window whose all-pairs minimum H is <= 0.9"
            ),
            "tie_break": "candidate, matrix, landmark, then branch",
            "archive_checks": checks,
        }
    )
    return selected, rule


def _cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    normalized = values / norms[:, None]
    return np.clip(normalized @ normalized.T, 0.0, 1.0)


def _off_diagonal_minimum(matrix: np.ndarray) -> float:
    indices = np.triu_indices(matrix.shape[0], 1)
    return float(matrix[indices].min())


def _annotated_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    vmin: float,
    annotation_size: float,
) -> Any:
    image = axis.imshow(matrix, vmin=vmin, vmax=1.0, cmap=SIMILARITY_CMAP)
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=annotation_size,
                color="white" if value > max(0.91, vmin + 0.08) else INK,
            )
    return image


def _figure_f12(selected: pd.Series, records: list[Any], replay: dict[str, Any]) -> None:
    geometry = replay["geometry"]
    first_break = int(geometry["first_break_index"])
    episode_start = int(geometry["episode_start_index"])
    episode_end = int(geometry["episode_end_index"])
    fissions = np.arange(1, len(records) + 1)
    boundary_h = np.asarray([record.h for record in records], dtype=np.float64)
    anchor = records[first_break].parent
    anchor_h = np.asarray(
        [cosine_similarity(anchor, record.daughter) for record in records],
        dtype=np.float64,
    )
    daughters = np.vstack(
        [records[index].daughter for index in range(episode_start, episode_end + 1)]
    )
    pairwise = _cosine_matrix(daughters)

    fig = plt.figure(figsize=(17, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.25, 0.75])
    ax_trace = fig.add_subplot(grid[0, :])
    ax_comp = fig.add_subplot(grid[1, 0])
    ax_geom = fig.add_subplot(grid[1, 1])
    fig.suptitle(
        "Plastic heredity in one F12 future: local inheritance renews without one shared composition",
        fontsize=20,
        fontweight="bold",
        color=NAVY,
    )

    ax_trace.plot(
        fissions,
        boundary_h,
        color=NAVY,
        linewidth=2.4,
        marker="o",
        markersize=5.5,
        label=r"parent-to-selected-daughter $H$",
        zorder=3,
    )
    ax_trace.plot(
        fissions,
        anchor_h,
        color=GOLD,
        linewidth=2.0,
        marker="s",
        markersize=4.5,
        label=r"selected daughter $H$ to the old pre-break parent",
        zorder=2,
    )
    ax_trace.axhline(0.9, color=RED, linestyle="--", linewidth=1.5)
    ax_trace.axvspan(
        episode_start + 0.55,
        episode_end + 1.45,
        color=TEAL,
        alpha=0.14,
        label="three inherited fissions that certify renewal",
    )
    ax_trace.scatter(
        [first_break + 1],
        [boundary_h[first_break]],
        s=105,
        color=RED,
        edgecolor="white",
        linewidth=1.3,
        zorder=5,
    )
    ax_trace.annotate(
        "first break",
        xy=(first_break + 1, boundary_h[first_break]),
        xytext=(first_break + 0.45, 0.25),
        arrowprops={"arrowstyle": "->", "color": RED},
        color=RED,
        fontsize=11,
        fontweight="bold",
    )
    ax_trace.annotate(
        "renewal certified\nafter the third success",
        xy=(episode_end + 1, boundary_h[episode_end]),
        xytext=(episode_end + 1.5, 0.54),
        arrowprops={"arrowstyle": "->", "color": TEAL},
        color=TEAL,
        fontsize=11,
        fontweight="bold",
    )
    ax_trace.text(
        0.72,
        0.965,
        r"inheritance: $H>0.9$",
        transform=ax_trace.transAxes,
        color=RED,
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    ax_trace.set_xlim(0.6, len(records) + 0.4)
    ax_trace.set_ylim(0, 1.02)
    ax_trace.set_xticks(fissions)
    ax_trace.set_xlabel("Fission in the twelve-fission future")
    ax_trace.set_ylabel("Cosine similarity, H")
    ax_trace.set_title(
        "A. The event is a temporal relation: break, then three adjacent inheritance successes",
        loc="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    ax_trace.legend(frameon=False, loc="lower right", ncol=2)
    _style_axis(ax_trace)

    states = np.vstack([anchor, daughters])
    abundance = states / states.sum(axis=1, keepdims=True)
    top = np.argsort(abundance.sum(axis=0))[-14:][::-1]
    profile = abundance[:, top]
    image = ax_comp.imshow(profile, aspect="auto", cmap=COMPOSITION_CMAP, vmin=0)
    ax_comp.set_yticks(np.arange(4), ["old parent", "daughter 1", "daughter 2", "daughter 3"])
    ax_comp.set_xticks(
        np.arange(len(top)),
        [f"m{index}" for index in top],
        rotation=45,
        ha="right",
    )
    ax_comp.set_title(
        "B. Molecular composition changes across the locally inherited episode",
        loc="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    ax_comp.set_xlabel("Fourteen most abundant molecule types in these four states")
    colorbar = fig.colorbar(image, ax=ax_comp, fraction=0.026, pad=0.02)
    colorbar.set_label("Molecular fraction")

    heat = _annotated_heatmap(
        ax_geom,
        pairwise,
        ["D1", "D2", "D3"],
        "C. The three daughters are not mutually coherent",
        0.6,
        12,
    )
    colorbar = fig.colorbar(heat, ax=ax_geom, fraction=0.05, pad=0.04)
    colorbar.set_label("Daughter-pair H")
    _save(fig, "figure2_f12_example_trajectory.png")


def _figure_strict8(selected: dict[str, Any]) -> None:
    h = selected["h"]
    fissions = np.arange(1, len(h) + 1)
    first_start = int(selected["first_eligible_start"])
    strict_start = int(selected["primary_start"])
    certification = int(selected["certification"])

    fig = plt.figure(figsize=(17, 10.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 0.95], height_ratios=[0.95, 1.05])
    ax_trace = fig.add_subplot(grid[0, 0])
    ax_windows = fig.add_subplot(grid[0, 1])
    ax_pairwise = fig.add_subplot(grid[1, 0])
    ax_anchor = fig.add_subplot(grid[1, 1])
    fig.suptitle(
        "A stronger event in one F32 future: eight daughters form a coherent new neighbourhood",
        fontsize=20,
        fontweight="bold",
        color=NAVY,
    )

    ax_trace.plot(fissions, h, color=NAVY, marker="o", markersize=4.2, linewidth=2.1)
    ax_trace.axhline(0.9, color=RED, linestyle="--", linewidth=1.4)
    ax_trace.axvspan(
        first_start - 0.45,
        first_start + 7.45,
        color=GOLD,
        alpha=0.16,
        label="earlier inherited eight-window: not coherent",
    )
    ax_trace.axvspan(
        strict_start - 0.45,
        certification + 0.45,
        color=TEAL,
        alpha=0.18,
        label="first strict coherent-eight window",
    )
    ax_trace.scatter(
        selected["first_break"],
        h[selected["first_break"] - 1],
        s=90,
        color=RED,
        edgecolor="white",
        zorder=5,
    )
    ax_trace.annotate(
        "first break",
        xy=(selected["first_break"], h[selected["first_break"] - 1]),
        xytext=(selected["first_break"] + 2, 0.42),
        arrowprops={"arrowstyle": "->", "color": RED},
        color=RED,
        fontsize=10.5,
        fontweight="bold",
    )
    ax_trace.set_xlim(0.5, len(h) + 0.5)
    ax_trace.set_ylim(0, 1.02)
    ax_trace.set_xlabel("Fission in the thirty-two-fission future")
    ax_trace.set_ylabel(r"Parent-to-selected-daughter $H$")
    ax_trace.set_title(
        "A. Adjacent inheritance alone cannot identify episode-wide coherence",
        loc="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    ax_trace.legend(frameon=False, loc="lower right", fontsize=10.2)
    _style_axis(ax_trace)

    windows = np.asarray(selected["eligible_windows"], dtype=np.float64)
    ax_windows.plot(
        windows[:, 0],
        windows[:, 1],
        marker="o",
        linewidth=2.1,
        color=TEAL,
        label="weakest daughter-pair H",
    )
    ax_windows.plot(
        windows[:, 0],
        windows[:, 2],
        marker="s",
        linewidth=2.0,
        color=GOLD,
        label="largest daughter-to-old-parent H",
    )
    ax_windows.axhline(0.9, color=TEAL, linestyle="--", linewidth=1.2)
    ax_windows.axhline(0.85, color=GOLD, linestyle="--", linewidth=1.2)
    ax_windows.scatter(
        [strict_start],
        [selected["qualifying_minimum_pairwise"]],
        s=105,
        color=TEAL,
        edgecolor="white",
        zorder=5,
    )
    ax_windows.annotate(
        "first window satisfying both gates",
        xy=(strict_start, selected["qualifying_minimum_pairwise"]),
        xytext=(strict_start - 4.5, 0.98),
        arrowprops={"arrowstyle": "->", "color": INK},
        color=INK,
        fontsize=10,
    )
    ax_windows.set_ylim(0, 1.02)
    ax_windows.set_xlabel("Start fission of each inherited eight-window")
    ax_windows.set_ylabel("Window geometry")
    ax_windows.set_title(
        "B. A long inherited run becomes strict only when geometry also passes",
        loc="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    ax_windows.legend(frameon=False, loc="lower right", fontsize=10.2)
    _style_axis(ax_windows)

    heat = _annotated_heatmap(
        ax_pairwise,
        selected["pairwise"],
        [f"D{index}" for index in range(1, 9)],
        "C. All 28 daughter pairs pass the coherence gate",
        0.88,
        8.4,
    )
    colorbar = fig.colorbar(heat, ax=ax_pairwise, fraction=0.042, pad=0.03)
    colorbar.set_label("Daughter-pair H")

    x = np.arange(1, 9)
    anchor = selected["anchor_similarities"]
    ax_anchor.bar(x, anchor, color=ORANGE, width=0.68)
    ax_anchor.axhline(0.85, color=RED, linestyle="--", linewidth=1.4)
    ax_anchor.set_ylim(0, 1.0)
    ax_anchor.set_xticks(x, [f"D{index}" for index in x])
    ax_anchor.set_xlabel("Daughter in the qualifying window")
    ax_anchor.set_ylabel(r"$H$ to the old pre-break parent")
    ax_anchor.set_title(
        "D. All eight daughters pass old-anchor distinctness",
        loc="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    ax_anchor.text(
        0.98,
        0.88,
        r"distinctness gate: $H\leq0.85$",
        transform=ax_anchor.transAxes,
        ha="right",
        color=RED,
        fontsize=10.5,
    )
    for index, value in enumerate(anchor, start=1):
        ax_anchor.text(index, value + 0.025, f"{value:.2f}", ha="center", fontsize=8.5)
    _style_axis(ax_anchor)
    _save(fig, "figure5_strict8_example_trajectory.png")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    f12_selected, f12_rule = _select_f12_exemplar()
    f12_records, f12_replay = _replay_f12(f12_selected)
    strict_selected, strict_rule = _select_strict8_exemplar()

    _figure_f12(f12_selected, f12_records, f12_replay)
    _figure_strict8(strict_selected)

    provenance = {
        "format": "plastic-heredity-paper-trajectory-exemplars-v1",
        "purpose": "post-hoc visualization only; no inferential claim is selected by these examples",
        "f12": {
            "source_table": str(F12_TABLE.relative_to(ROOT)),
            "selection_rule": f12_rule,
            "selected": {
                "source_cohort": str(f12_selected["source_cohort"]),
                "state_id": str(f12_selected["state_id"]),
                "candidate": str(f12_selected["candidate"]).zfill(2),
                "matrix_id": int(f12_selected["matrix_id"]),
                "landmark": int(f12_selected["landmark"]),
                "branch": int(f12_selected["branch"]),
                "selection_score": float(f12_selected["selection_score"]),
            },
            "replay": f12_replay,
        },
        "strict8": {
            "source_archive": str(STRICT8_ARCHIVE.relative_to(ROOT)),
            "selection_rule": strict_rule,
            "selected": {
                key: strict_selected[key]
                for key in (
                    "candidate",
                    "matrix",
                    "landmark",
                    "branch",
                    "first_break",
                    "first_eligible_start",
                    "first_eligible_minimum_pairwise",
                    "first_eligible_maximum_anchor",
                    "primary_start",
                    "certification",
                    "run_length",
                    "qualifying_minimum_pairwise",
                    "qualifying_maximum_anchor",
                    "selection_score",
                )
            },
        },
    }
    (OUT / "trajectory_exemplar_provenance.json").write_text(
        json.dumps(_json_ready(provenance), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Generated figure2_f12_example_trajectory.png and "
        "figure5_strict8_example_trajectory.png"
    )


if __name__ == "__main__":
    main()
