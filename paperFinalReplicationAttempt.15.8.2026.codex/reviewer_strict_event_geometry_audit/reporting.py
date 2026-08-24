"""Data-driven figures and reports for the strict-event audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPEC_NAMES = (
    "cosine_registered",
    "bray_global",
    "bray_relation_specific",
)
SPEC_LABELS = {
    "cosine_registered": "Registered cosine",
    "bray_global": "Global-mapped Bray",
    "bray_relation_specific": "Relation-mapped Bray",
}
SPEC_SHORT = {
    "cosine_registered": "Cosine",
    "bray_global": "Bray global",
    "bray_relation_specific": "Bray relation",
}
COLORS = {
    "cosine_registered": "#2f5597",
    "bray_global": "#b24c35",
    "bray_relation_specific": "#3c8d5a",
}


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _pct(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{100.0 * number:.{digits}f}%"


def _save_figure(figure: plt.Figure, directory: Path, stem: str) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for extension in ("png", "pdf"):
        path = directory / f"{stem}.{extension}"
        figure.savefig(path, dpi=220 if extension == "png" else None, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(figure)
    return outputs


def _gate_figure(output_root: Path, figure_root: Path) -> list[str]:
    table = pd.read_csv(output_root / "gate_waterfall.csv")
    table = table.loc[
        (table["cohort"] == "confirmation")
        & ~table["stage"].str.startswith("terminal:")
    ]
    stages = ["observed", "break", "inherited_run8", "coherent_window", "strict_event"]
    labels = ["All", "Break", "Later run-8", "Coherent 8", "Strict event"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for axis, candidate in zip(axes, ("02", "03"), strict=True):
        for spec in SPEC_NAMES:
            selected = table.loc[
                (table["candidate"].astype(str).str.zfill(2) == candidate)
                & (table["spec"] == spec)
            ].set_index("stage")
            values = [float(selected.loc[stage, "fraction"]) for stage in stages]
            axis.plot(
                range(len(stages)),
                values,
                marker="o",
                linewidth=2,
                color=COLORS[spec],
                label=SPEC_SHORT[spec],
            )
        axis.set_title(f"Candidate {candidate}")
        axis.set_xticks(range(len(stages)), labels, rotation=24, ha="right")
        axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Fraction of confirmation branches reaching gate")
    axes[1].legend(frameon=False, fontsize=9)
    figure.suptitle("Where branches are lost along the strict-event definition")
    return _save_figure(figure, figure_root, "gate_waterfall_confirmation")


def _nondegeneracy_figure(output_root: Path, figure_root: Path) -> list[str]:
    effects = pd.read_csv(output_root / "matched_nondegeneracy_effects.csv")
    selected_stats = (
        ("effective_species_mean", "Effective species\n(event − control)"),
        ("occupied_types_mean", "Occupied types\n(event − control)"),
        ("top1_share_mean", "Largest share\n(event − control)"),
        ("adjacent_total_variation_mean", "Turnover (TV)\n(event − control)"),
        ("growth_steps_mean", "Growth updates\n(event − control)"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.1))
    flat_axes = axes.ravel()
    x_positions = np.arange(2, dtype=float)
    offsets = {-1: -0.18, 0: 0.0, 1: 0.18}
    for panel, (statistic, ylabel) in enumerate(selected_stats):
        axis = flat_axes[panel]
        for spec_index, spec in enumerate(SPEC_NAMES):
            points = []
            lower = []
            upper = []
            for candidate in ("02", "03"):
                row = effects.loc[
                    (effects["spec"] == spec)
                    & (effects["candidate"].astype(str).str.zfill(2) == candidate)
                    & (effects["statistic"] == statistic)
                ]
                if row.empty:
                    points.append(np.nan)
                    lower.append(np.nan)
                    upper.append(np.nan)
                else:
                    item = row.iloc[0]
                    points.append(float(item["paired_difference"]))
                    lower.append(float(item["ci95_lower"]))
                    upper.append(float(item["ci95_upper"]))
            points_a = np.asarray(points)
            errors = np.maximum(
                0.0,
                np.vstack(
                    (points_a - np.asarray(lower), np.asarray(upper) - points_a)
                ),
            )
            axis.errorbar(
                x_positions + offsets[spec_index - 1],
                points_a,
                yerr=errors,
                marker="o",
                linestyle="none",
                capsize=3,
                color=COLORS[spec],
                label=SPEC_SHORT[spec],
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axis.set_xticks(x_positions, ["Candidate 02", "Candidate 03"])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
    flat_axes[5].axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    flat_axes[5].legend(handles, labels, loc="center", frameon=False)
    figure.suptitle(
        "Strict-event windows versus same-state precursor controls\n"
        "Matrix-bootstrap 95% intervals"
    )
    figure.tight_layout()
    return _save_figure(figure, figure_root, "matched_nondegeneracy_effects")


def _prediction_figure(output_root: Path, figure_root: Path) -> list[str]:
    table = pd.read_csv(output_root / "prediction_comparisons.csv")
    cells = [("02", "A"), ("02", "B"), ("03", "A"), ("03", "B")]
    cell_labels = ["02-A", "02-B", "03-A", "03-B"]
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), sharey=True)
    for axis, target in zip(axes, SPEC_NAMES, strict=True):
        training_targets = [target]
        if target != "cosine_registered":
            training_targets.append("cosine_registered")
        for series_index, training_target in enumerate(training_targets):
            values = []
            lower = []
            upper = []
            for candidate, half in cells:
                row = table.loc[
                    (table["evaluation_target"] == target)
                    & (table["training_target"] == training_target)
                    & (table["candidate"].astype(str).str.zfill(2) == candidate)
                    & (table["half"] == half)
                ].iloc[0]
                values.append(float(row["log_loss_gain"]))
                lower.append(float(row["log_loss_gain_ci95_lower"]))
                upper.append(float(row["log_loss_gain_ci95_upper"]))
            values_a = np.asarray(values)
            errors = np.maximum(
                0.0,
                np.vstack(
                    (values_a - np.asarray(lower), np.asarray(upper) - values_a)
                ),
            )
            positions = np.arange(4) + (series_index - (len(training_targets) - 1) / 2) * 0.12
            label = (
                "Target-specific fit"
                if training_target == target
                else "Cosine-trained transfer"
            )
            color = COLORS[target] if training_target == target else "#777777"
            axis.errorbar(
                positions,
                values_a,
                yerr=errors,
                marker="o" if training_target == target else "s",
                linestyle="none",
                capsize=3,
                color=color,
                label=label,
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.65)
        axis.set_xticks(range(4), cell_labels, rotation=30)
        axis.set_title(SPEC_LABELS[target])
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Held-out log-loss gain (nats/state)\nh10+state minus h10")
    figure.suptitle("Target-matched prediction and cosine-model transfer")
    return _save_figure(figure, figure_root, "prediction_gains_and_transfer")


def _overlap_figure(output_root: Path, figure_root: Path) -> list[str]:
    table = pd.read_csv(output_root / "event_overlap.csv")
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    heat_image = None
    for axis, candidate in zip(axes[:2], ("02", "03"), strict=True):
        selected = table.loc[
            (table["candidate"].astype(str).str.zfill(2) == candidate)
            & table["left"].isin(SPEC_NAMES)
            & table["right"].isin(SPEC_NAMES)
        ]
        matrix = np.full((3, 3), np.nan)
        for left_index, left in enumerate(SPEC_NAMES):
            for right_index, right in enumerate(SPEC_NAMES):
                matrix[left_index, right_index] = selected.loc[
                    (selected["left"] == left) & (selected["right"] == right),
                    "jaccard",
                ].iloc[0]
        heat_image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    _fmt(matrix[row, column], 2),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] < 0.55 else "black",
                    fontsize=8,
                )
        labels = ["Cos", "Global", "Relation"]
        axis.set_xticks(range(3), labels, rotation=25)
        axis.set_yticks(range(3), labels)
        axis.set_title(f"Candidate {candidate}: Jaccard")
    strata = table.loc[table["left"] == "stratum"].copy()
    bottoms = np.zeros(2)
    candidates = ("02", "03")
    colors = plt.cm.tab20(np.linspace(0.05, 0.9, 8))
    for index, code in enumerate(range(8)):
        name = f"cosine{(code >> 2) & 1}_global{(code >> 1) & 1}_relation{code & 1}"
        values = []
        for candidate in candidates:
            row = strata.loc[
                (strata["candidate"].astype(str).str.zfill(2) == candidate)
                & (strata["right"] == name),
                "raw_agreement",
            ]
            values.append(float(row.iloc[0]))
        axes[2].bar(
            range(2),
            values,
            bottom=bottoms,
            color=colors[index],
            label=f"{(code >> 2) & 1}{(code >> 1) & 1}{code & 1}",
        )
        bottoms += np.asarray(values)
    axes[2].set_xticks(range(2), ["Candidate 02", "Candidate 03"])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Fraction of branches")
    axes[2].set_title("Event-membership strata\n(Cos/Global/Relation)")
    axes[2].legend(ncol=2, fontsize=7, frameon=False, title="Binary stratum")
    if heat_image is not None:
        figure.colorbar(heat_image, ax=axes[:2], fraction=0.025, pad=0.03)
    figure.suptitle("Overlap among strict-event definitions")
    return _save_figure(figure, figure_root, "event_overlap_and_strata")


def _calibration_figure(
    output_root: Path, calibration_root: Path, figure_root: Path
) -> list[str]:
    del output_root
    mapping = json.loads(
        (calibration_root / "relation_specific_calibration.json").read_text()
    )
    with np.load(calibration_root / "comparison_distributions.npz") as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), sharey=True)
    for axis, relation in zip(axes, ("boundary", "coherence", "anchor"), strict=True):
        for metric, color, label in (
            ("cosine", COLORS["cosine_registered"], "Cosine"),
            ("bray_curtis", COLORS["bray_relation_specific"], "Bray–Curtis"),
        ):
            array = np.sort(values[f"{relation}_{metric}"])
            if array.size > 25000:
                indices = np.linspace(0, array.size - 1, 25000).astype(int)
                array = array[indices]
            y = np.linspace(1.0 / len(array), 1.0, len(array))
            axis.plot(array, y, color=color, label=label, linewidth=1.6)
        axis.axvline(
            float(mapping[relation]["source_cutoff"]),
            color=COLORS["cosine_registered"],
            linestyle="--",
            linewidth=1,
        )
        axis.axvline(
            float(mapping[relation]["target_cutoff"]),
            color=COLORS["bray_relation_specific"],
            linestyle="--",
            linewidth=1,
        )
        axis.set_title(relation.title())
        axis.set_xlabel("Similarity")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Development empirical CDF")
    axes[-1].legend(frameon=False)
    figure.suptitle("Relation-specific development-only percentile calibration")
    return _save_figure(figure, figure_root, "relation_specific_calibration")


def make_figures(output_root: Path, calibration_root: Path) -> list[str]:
    figure_root = output_root / "figures"
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    outputs: list[str] = []
    outputs.extend(_gate_figure(output_root, figure_root))
    outputs.extend(_nondegeneracy_figure(output_root, figure_root))
    outputs.extend(_prediction_figure(output_root, figure_root))
    outputs.extend(_overlap_figure(output_root, figure_root))
    outputs.extend(_calibration_figure(output_root, calibration_root, figure_root))
    return outputs


def _power_markdown(power: pd.DataFrame) -> list[str]:
    lines = [
        "| Endpoint | Candidate | Development events | Confirmation events | Confirmation rate | Event matrices | Power rule |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for spec in SPEC_NAMES:
        for candidate in ("02", "03"):
            selected = power.loc[
                (power["spec"] == spec)
                & (power["candidate"].astype(str).str.zfill(2) == candidate)
            ]
            dev = selected.loc[selected["cohort"] == "development"].iloc[0]
            conf = selected.loc[selected["cohort"] == "confirmation"].iloc[0]
            rule = "adequate" if bool(dev["power_adequate"] and conf["power_adequate"]) else "underpowered"
            lines.append(
                f"| {SPEC_LABELS[spec]} | {candidate} | {int(dev['events']):,} | "
                f"{int(conf['events']):,} | {_pct(conf['prevalence'])} | "
                f"{int(conf['event_matrices'])} | {rule} |"
            )
    return lines


def _prediction_markdown(prediction: pd.DataFrame) -> list[str]:
    lines = [
        "| Evaluation target | Fit target | Cell | Log-loss gain | 95% CI | Holm p | Gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    selected = prediction.loc[
        prediction["target_matched"]
        | (prediction["training_target"] == "cosine_registered")
    ].drop_duplicates(
        ["evaluation_target", "training_target", "candidate", "half"]
    )
    for _, row in selected.iterrows():
        holm = _fmt(row["randomization_p_holm"], 4) if bool(row["target_matched"]) else "not tested"
        gate = (
            "pass"
            if bool(row["passes_exploratory_gate"])
            else ("no pass" if bool(row["target_matched"]) else "transfer control")
        )
        lines.append(
            f"| {SPEC_LABELS[row['evaluation_target']]} | "
            f"{SPEC_LABELS[row['training_target']]} | {str(row['candidate']).zfill(2)}-{row['half']} | "
            f"{_fmt(row['log_loss_gain'], 5)} | "
            f"[{_fmt(row['log_loss_gain_ci95_lower'], 5)}, {_fmt(row['log_loss_gain_ci95_upper'], 5)}] | "
            f"{holm} | {gate} |"
        )
    return lines


def _event_summary_value(
    summary: pd.DataFrame, spec: str, candidate: str, statistic: str, field: str = "mean"
) -> float:
    row = summary.loc[
        (summary["spec"] == spec)
        & (summary["candidate"].astype(str).str.zfill(2) == candidate)
        & (summary["statistic"] == statistic)
    ]
    return float(row.iloc[0][field]) if not row.empty else np.nan


def _nondegeneracy_markdown(summary: pd.DataFrame) -> list[str]:
    lines = [
        "| Endpoint | Candidate | Effective species | Occupied types | Largest-species share | Adjacent TV | All 8 top-1 dominated | All 8 top-2 dominated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in SPEC_NAMES:
        for candidate in ("02", "03"):
            lines.append(
                f"| {SPEC_LABELS[spec]} | {candidate} | "
                f"{_fmt(_event_summary_value(summary, spec, candidate, 'effective_species_mean'), 2)} | "
                f"{_fmt(_event_summary_value(summary, spec, candidate, 'occupied_types_mean'), 2)} | "
                f"{_fmt(_event_summary_value(summary, spec, candidate, 'top1_share_mean'), 3)} | "
                f"{_fmt(_event_summary_value(summary, spec, candidate, 'adjacent_total_variation_mean'), 3)} | "
                f"{_pct(_event_summary_value(summary, spec, candidate, 'all_daughters_top1_ge_0_80'))} | "
                f"{_pct(_event_summary_value(summary, spec, candidate, 'all_daughters_top2_ge_0_80'))} |"
            )
    return lines


def _matched_markdown(effects: pd.DataFrame, matching: pd.DataFrame) -> list[str]:
    lines = [
        "| Endpoint | Candidate | Pairs | Statistic | Event − control | 95% matrix-bootstrap CI |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for spec in SPEC_NAMES:
        for candidate in ("02", "03"):
            pair_row = matching.loc[
                (matching["spec"] == spec)
                & (matching["candidate"].astype(str).str.zfill(2) == candidate)
            ].iloc[0]
            for statistic in (
                "effective_species_mean",
                "occupied_types_mean",
                "top1_share_mean",
                "adjacent_total_variation_mean",
                "growth_steps_mean",
            ):
                row = effects.loc[
                    (effects["spec"] == spec)
                    & (effects["candidate"].astype(str).str.zfill(2) == candidate)
                    & (effects["statistic"] == statistic)
                ]
                if row.empty:
                    estimate = lower = upper = np.nan
                else:
                    item = row.iloc[0]
                    estimate = item["paired_difference"]
                    lower = item["ci95_lower"]
                    upper = item["ci95_upper"]
                lines.append(
                    f"| {SPEC_LABELS[spec]} | {candidate} | {int(pair_row['matched_pairs']):,} | "
                    f"{statistic} | {_fmt(estimate, 4)} | [{_fmt(lower, 4)}, {_fmt(upper, 4)}] |"
                )
    return lines


def _overlap_markdown(overlap: pd.DataFrame) -> list[str]:
    lines = [
        "| Candidate | Pair | Intersection | Union | Jaccard |",
        "|---|---|---:|---:|---:|",
    ]
    for candidate in ("02", "03"):
        for right in ("bray_global", "bray_relation_specific"):
            row = overlap.loc[
                (overlap["candidate"].astype(str).str.zfill(2) == candidate)
                & (overlap["left"] == "cosine_registered")
                & (overlap["right"] == right)
            ].iloc[0]
            lines.append(
                f"| {candidate} | Cosine vs {SPEC_SHORT[right]} | "
                f"{int(row['intersection']):,} | {int(row['union']):,} | {_fmt(row['jaccard'], 3)} |"
            )
    return lines


def _classification(
    power: pd.DataFrame,
    prediction: pd.DataFrame,
    overlap: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    diagonal = prediction.loc[prediction["target_matched"]]
    per_target: dict[str, Any] = {}
    for spec in SPEC_NAMES:
        rows = diagonal.loc[diagonal["evaluation_target"] == spec]
        passes = int(rows["passes_exploratory_gate"].sum())
        per_target[spec] = {
            "cells_passing_exploratory_gate": passes,
            "cells_total": len(rows),
            "all_cells_pass": passes == len(rows),
            "positive_gain_cells": int((rows["log_loss_gain"] > 0).sum()),
        }
    overlap_gain: dict[str, float] = {}
    for candidate in ("02", "03"):
        def jaccard(right: str) -> float:
            return float(
                overlap.loc[
                    (overlap["candidate"].astype(str).str.zfill(2) == candidate)
                    & (overlap["left"] == "cosine_registered")
                    & (overlap["right"] == right),
                    "jaccard",
                ].iloc[0]
            )
        overlap_gain[candidate] = jaccard("bray_relation_specific") - jaccard("bray_global")
    top1_all = {
        candidate: _event_summary_value(
            summary,
            "cosine_registered",
            candidate,
            "all_daughters_top1_ge_0_80",
        )
        for candidate in ("02", "03")
    }
    all_power = bool(power["power_adequate"].all())
    return {
        "target_specific_prediction": per_target,
        "relation_mapping_jaccard_improvement_over_global": overlap_gain,
        "registered_event_fraction_all_eight_top1_dominated": top1_all,
        "all_endpoint_candidate_cohorts_meet_power_rule": all_power,
        "interpretive_flags": {
            "relation_mapping_improves_cosine_overlap_in_both_candidates": all(
                value > 0 for value in overlap_gain.values()
            ),
            "registered_events_not_universally_top1_dominated": all(
                np.isfinite(value) and value < 1.0 for value in top1_all.values()
            ),
        },
    }


def build_reports(
    task_root: Path,
    output_root: Path,
    calibration_root: Path,
    model_root: Path,
) -> dict[str, Any]:
    power = pd.read_csv(output_root / "event_power.csv")
    prediction = pd.read_csv(output_root / "prediction_comparisons.csv")
    overlap = pd.read_csv(output_root / "event_overlap.csv")
    cross = pd.read_csv(output_root / "cross_metric_window_evaluation.csv")
    gates = pd.read_csv(output_root / "gate_waterfall.csv")
    summary = pd.read_csv(output_root / "event_nondegeneracy_summary.csv")
    matching = pd.read_csv(output_root / "matching_summary.csv")
    effects = pd.read_csv(output_root / "matched_nondegeneracy_effects.csv")
    reliability = pd.read_csv(output_root / "reliability.csv")
    calibration = json.loads(
        (calibration_root / "relation_specific_calibration.json").read_text()
    )
    seal = json.loads((model_root / "model_seal.json").read_text())
    classification = _classification(power, prediction, overlap, summary)

    relation_passes = classification["target_specific_prediction"][
        "bray_relation_specific"
    ]["cells_passing_exploratory_gate"]
    relation_total = classification["target_specific_prediction"][
        "bray_relation_specific"
    ]["cells_total"]
    overlap_flag = classification["interpretive_flags"][
        "relation_mapping_improves_cosine_overlap_in_both_candidates"
    ]
    dominance_flag = classification["interpretive_flags"][
        "registered_events_not_universally_top1_dominated"
    ]

    cutoff_lines = [
        "| Relation | Cosine cutoff | Matched Bray cutoff | Development comparisons |",
        "|---|---:|---:|---:|",
    ]
    for relation in ("boundary", "coherence", "anchor"):
        item = calibration[relation]
        cutoff_lines.append(
            f"| {relation.title()} | {_fmt(item['source_cutoff'], 5)} | "
            f"{_fmt(item['target_cutoff'], 5)} | {int(item['paired_comparisons']):,} |"
        )

    cross_lines = [
        "| Candidate | Cosine-event window evaluated under | Boundary pass | Pairwise pass | Anchor pass | All conditions |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for candidate in ("02", "03"):
        for target in ("bray_global", "bray_relation_specific"):
            row = cross.loc[
                (cross["candidate"].astype(str).str.zfill(2) == candidate)
                & (cross["source_event"] == "cosine_registered")
                & (cross["target_evaluation"] == target)
            ].iloc[0]
            cross_lines.append(
                f"| {candidate} | {SPEC_LABELS[target]} | "
                f"{_pct(row['fraction_boundary_pass'])} | {_pct(row['fraction_pairwise_pass'])} | "
                f"{_pct(row['fraction_anchor_pass'])} | {_pct(row['fraction_all_target_conditions_pass'])} |"
            )

    gate_lines = [
        "| Endpoint | Candidate | Break | Later run-8 | Coherent window | Strict event |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for spec in SPEC_NAMES:
        for candidate in ("02", "03"):
            selected = gates.loc[
                (gates["cohort"] == "confirmation")
                & (gates["spec"] == spec)
                & (gates["candidate"].astype(str).str.zfill(2) == candidate)
                & ~gates["stage"].str.startswith("terminal:")
            ].set_index("stage")
            gate_lines.append(
                f"| {SPEC_LABELS[spec]} | {candidate} | {_pct(selected.loc['break', 'fraction'])} | "
                f"{_pct(selected.loc['inherited_run8', 'fraction'])} | "
                f"{_pct(selected.loc['coherent_window', 'fraction'])} | "
                f"{_pct(selected.loc['strict_event', 'fraction'])} |"
            )

    reliability_lines = [
        "| Endpoint | Candidate | Split-half Spearman | Matrix-centered Spearman |",
        "|---|---:|---:|---:|",
    ]
    for _, row in reliability.iterrows():
        reliability_lines.append(
            f"| {SPEC_LABELS[row['spec']]} | {str(row['candidate']).zfill(2)} | "
            f"{_fmt(row['ordinary'], 3)} | {_fmt(row['centered'], 3)} |"
        )

    verdict = (
        f"The relation-specific Bray target recovered closer event membership than the prior global mapping in both candidates: {overlap_flag}. "
        f"Its target-specific state-block comparison passed the frozen exploratory gate in {relation_passes}/{relation_total} candidate-by-half cells. "
        f"Registered cosine events were not universally dominated by a single type across all eight daughters: {dominance_flag}."
    )
    result_lines = [
        "# Strict-event geometry, non-degeneracy, and target-specific prediction audit",
        "",
        "## Bottom line",
        "",
        verdict,
        "",
        "This is a locked post-hoc robustness audit. It replays the existing deterministic development and confirmation futures; it does not generate a new prospective confirmation cohort and does not alter the manuscript.",
        "",
        "## What was tested",
        "",
        "Three otherwise identical strict-event definitions were applied to every retained future: the registered cosine definition, the previous globally percentile-mapped Bray–Curtis definition, and a new Bray–Curtis definition whose boundary, within-window coherence, and old-anchor cutoffs were calibrated separately. Relation-specific cutoffs used only fixed development branches and did not match event prevalence.",
        "",
        *cutoff_lines,
        "",
        "## Event counts and power",
        "",
        *_power_markdown(power),
        "",
        "The frozen descriptive power rule requires at least 100 events and 20 event-bearing matrices in both development and confirmation for each endpoint–candidate cell. An underpowered cell remains reported but cannot pass the exploratory prediction gate.",
        "",
        "## Failure-gate localization",
        "",
        *gate_lines,
        "",
        "These are cumulative fractions. They distinguish failure to break, failure to regain eight consecutive inherited selected-lineage fissions, failure of mutual daughter coherence, and failure of old-anchor separation.",
        "",
        "## Event overlap and same-window geometry",
        "",
        *_overlap_markdown(overlap),
        "",
        *cross_lines,
        "",
        "The same-window table asks whether the exact eight-daughter window that qualifies under cosine also satisfies each Bray condition. It therefore localizes geometric disagreement without changing the temporal window.",
        "",
        "## Non-degeneracy of all strict events",
        "",
        *_nondegeneracy_markdown(summary),
        "",
        "Effective species number is exp(Shannon entropy). Occupied-type and composition statistics are evaluated on the eight selected daughters in the earliest qualifying window. `All 8 top-1 dominated` means every daughter assigns at least 80% of its normalized composition to one type; the top-2 column applies the same rule to the two largest types. By construction, all eight event boundaries exceed the endpoint-specific inheritance cutoff; the retained event table also reports the minimum boundary, pairwise-coherence, and anchor-distinctness margins.",
        "",
        "## Matched non-event comparison",
        "",
        *_matched_markdown(effects, matching),
        "",
        "Each event is matched without replacement to a negative branch from the same natural state that nevertheless reached a post-break inherited run of eight. Event windows use the earliest qualifying strict window; controls use the earliest eligible run-8 precursor. Intervals resample catalytic matrices.",
        "",
        "## Target-specific prediction",
        "",
        *_prediction_markdown(prediction),
        "",
        "Positive gain means the original no-PCA `h10 + state` model has lower held-out log loss than `h10` alone. Each target-specific model suite was refit on development labels and sealed before the new relation-specific confirmation labels were scored. Holm adjustment applies only to the four target-matched candidate-by-half cells per endpoint; cosine-trained rows are transfer controls, not additional hypothesis tests.",
        "",
        "## Endpoint reliability",
        "",
        *reliability_lines,
        "",
        "## Interpretation",
        "",
        "- A better relation-specific Bray match would show that much of the previous metric sensitivity came from forcing one global percentile map onto three different geometric relations. It would not make cosine and Bray equivalent.",
        "- A retained target-matched prediction gain would show that present-state information predicts that metric's event beyond the fixed history block. It would not establish a causal mechanism.",
        "- The non-degeneracy results determine whether coherence is typically associated with compositional collapse. Concentration can be a mechanism rather than an artefact, but it changes the biological interpretation.",
        "- Every result concerns parent-to-one-selected-daughter lineage continuity. It does not establish fidelity of both daughters or whole-population reproduction.",
        "",
        "## Reproducibility and claim boundary",
        "",
        f"The model seal is `{seal['seal_id']}`. All replay labels are checked against the archived registered cosine labels and onsets; the globally mapped Bray confirmation labels and onsets are also checked against the prior sensitivity audit. The analysis uses 4,096 matrix bootstraps and 4,096 matrix-block randomizations. This remains a post-hoc diagnostic of simulated selected-lineage geometry and predictability, not an intervention, causal test, or new prospective confirmation.",
        "",
        "## Files",
        "",
        "- `artifacts/output/event_characteristics.csv.gz`: one row per strict event.",
        "- `artifacts/output/event_nondegeneracy_summary.csv`: all-event summaries.",
        "- `artifacts/output/matched_event_control_pairs.csv.gz`: exact matched pairs and differences.",
        "- `artifacts/output/prediction_comparisons.csv`: target-matched and transfer prediction tests.",
        "- `artifacts/output/gate_waterfall.csv`: cumulative and terminal failure gates.",
        "- `artifacts/output/figures/`: calibration, gate, overlap, non-degeneracy, and prediction plots.",
    ]
    (task_root / "RESULTS_REPORT.md").write_text("\n".join(result_lines) + "\n")

    top1_values = classification[
        "registered_event_fraction_all_eight_top1_dominated"
    ]
    lay_lines = [
        "# Lay summary",
        "",
        "We asked why the paper's rare, very strict heredity event looked much less common when composition was compared with Bray–Curtis similarity instead of cosine similarity.",
        "",
        "The key refinement was to stop treating three different requirements as though they had the same geometry. A parent–daughter boundary, similarity among eight daughters, and separation from an old parent are different comparisons. We calibrated a Bray–Curtis threshold for each one using development data only, then applied the frozen rules to confirmation data.",
        "",
        verdict,
        "",
        f"For the original cosine event, the fraction of events in which one molecule type held at least 80% of the composition in all eight daughters was {_pct(top1_values['02'])} for candidate 02 and {_pct(top1_values['03'])} for candidate 03. The full report also compares diversity, largest-species share, turnover, and growth against hard negative trajectories from the same starting state.",
        "",
        "The prediction test was also made fairer to the new target: models were trained on each event definition rather than asking a cosine-trained model to predict a different Bray event. Any gain still means prediction, not proof that the state variables cause the event.",
        "",
        "Finally, 'heredity' here follows one selected daughter after every fission. It is selected-lineage continuity, not evidence that both daughters faithfully reproduce the parent.",
    ]
    (task_root / "LAY_SUMMARY.md").write_text("\n".join(lay_lines) + "\n")

    suggested_lines = [
        "# Suggested manuscript additions (not applied)",
        "",
        "These passages are generated from the completed post-hoc audit and should be checked against the surrounding manuscript before insertion.",
        "",
        "## Results: after the existing threshold/metric sensitivity paragraph",
        "",
        "We further examined why the strict coherent-eight endpoint was sensitive to the compositional similarity metric. Using fixed development branches, we separately percentile-calibrated Bray–Curtis cutoffs for parent-to-selected-daughter inheritance, pairwise daughter coherence, and old-anchor separation, without matching endpoint prevalence. We then replayed the unchanged development and confirmation futures under the registered cosine definition, the prior globally mapped Bray–Curtis definition, and the relation-specific Bray–Curtis definition. [Insert the event-count, overlap, gate-waterfall, and same-window results from `RESULTS_REPORT.md`; retain the wording as a post-hoc robustness analysis.]",
        "",
        "## Results or supplement: non-degeneracy characterization",
        "",
        "For every earliest qualifying coherent-eight window we measured effective species number, occupied-type count, largest- and two-largest-species shares, adjacent compositional turnover, occupied-set turnover, and growth updates. We also matched events within the same natural state to non-event branches that reached a post-break inherited run of eight. [Insert the all-event and matched estimates from `RESULTS_REPORT.md`.] Thus the operational event should be interpreted in light of its observed concentration profile rather than assuming that high cosine coherence alone establishes a compositionally rich copied state.",
        "",
        "## Predictor discussion",
        "",
        "Because changing the event metric also changes the prediction target, we refit the frozen no-PCA history-plus-state model suite separately for each development endpoint and sealed it before confirmation scoring. [Insert target-matched gain estimates and multiplicity-adjusted results.] Predictions from the original cosine-trained model were retained as transfer controls. These post-hoc comparisons test held-out predictive information, not causal attribution.",
        "",
        "## Methods or supplementary methods",
        "",
        "State explicitly: (i) the three endpoint definitions and strict/inclusive inequalities; (ii) the fixed development calibration branches; (iii) the relation-specific empirical-CDF mapping; (iv) the ordered failure gates; (v) the earliest-event and earliest-precursor window rules; (vi) same-state, without-replacement deterministic matching; (vii) matrix-level bootstrap/randomization and Holm families; and (viii) that target-specific fitting used development outcomes only and was sealed before confirmation scoring.",
        "",
        "## Abstract and limitations",
        "",
        "Use 'selected-lineage parent-to-daughter inheritance' at first mention. Add that the endpoint follows one selected daughter per fission and does not measure both-daughter reproductive fidelity. Also state that the relation-specific metric analysis and target-specific refits are post-hoc robustness analyses on replayed deterministic futures, not new prospective confirmation or causal intervention evidence.",
    ]
    (task_root / "SUGGESTED_TEXT.md").write_text("\n".join(suggested_lines) + "\n")

    return classification
