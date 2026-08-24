#!/usr/bin/env python3
"""Generate the cross-model evidence and lineage-carrier figures.

The script reads only retained cellular-automaton and Wagner scientific
artefacts. It preserves the original Figure 9/10 assets and writes updated V2
Figure 11/12 assets plus compact provenance records under ``figures/``. The
visual chronology preserves the experimental ordering: break-and-renewal was
detected before the later carrier interventions.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
CA_ROOT = ROOT / (
    "NewIdeas/preprints/ingressing-minds-v-ruliad-paper-ideas/"
    "codex.reconstructionsAndStressTesting"
)
RETAINED_ATLAS = ROOT / (
    "NewIdeas/preprints/ingressing-minds-v-ruliad-paper-ideas/"
    "eca_atlas/results/full/eca_rules.csv"
)
CLEAN_ATLAS = CA_ROOT / "results/golden-reconciliation/atlas/eca_rules.csv"
PHASE = CA_ROOT / "results/golden-reconciliation/phase/phase.csv"
GOLDEN_RESULTS = CA_ROOT / "results/golden-reconciliation/RESULTS.json"
CAUSAL_RESULTS = CA_ROOT / "results/causal-heredity-round-1/RESULTS.json"
CA_MOTIF_STAGE1_RESULTS = ROOT / (
    "replicators.13.8.2026.codex/reviewer_motif_channel_replication/"
    "artifacts/stage1/RESULTS.json"
)
CA_MOTIF_STAGE2_RESULTS = ROOT / (
    "replicators.13.8.2026.codex/reviewer_motif_channel_replication/"
    "artifacts/stage2/RESULTS.json"
)
CA_COMPACT_RESULTS = ROOT / (
    "replicators.13.8.2026.codex/reviewer_ca_compact_carrier_replication/"
    "artifacts/confirmation/RESULTS.json"
)
CA_COMPACT_VERIFY = ROOT / (
    "replicators.13.8.2026.codex/reviewer_ca_compact_carrier_replication/"
    "artifacts/VERIFY.json"
)
WAGNER_OCCURRENCE = ROOT / (
    "replicators.13.8.2026.codex/wagner_cleanroom/runs/campaign-v2/"
    "predictor/analysis.json"
)
WAGNER_CARRIER = ROOT / (
    "replicators.13.8.2026.codex/wagner_memory_cleanroom_v2/runs/"
    "wagner-memory-v2-full-corrected-20260822/analysis/carrier.json"
)
WAGNER_SUMMARY = ROOT / (
    "replicators.13.8.2026.codex/wagner_memory_cleanroom_v2/runs/"
    "wagner-memory-v2-full-corrected-20260822/analysis/summary.json"
)
OUT = ROOT / "figures"

NAVY = "#17324D"
BLUE = "#2C78B8"
TEAL = "#1B998B"
GOLD = "#D89B2B"
ORANGE = "#D66A3A"
RED = "#B64040"
INK = "#263238"
MUTED = "#66727A"
GRID = "#D9E0E5"
PALE_BLUE = "#E8F1F8"
PALE_GREEN = "#E5F4EF"
PALE_GOLD = "#FBF3E2"
PALE_RED = "#FAEAEA"
WHITE = "#FFFFFF"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_value(value: str) -> bool:
    return value.strip().lower() == "true"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "text.color": INK,
        }
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str,
    fontsize: float = 11.0,
    weight: str = "normal",
    linestyle: str = "-",
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.5,
        linestyle=linestyle,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=INK,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    colour: str = MUTED,
    linestyle: str = "-",
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            linestyle=linestyle,
            color=colour,
            connectionstyle=connectionstyle,
        )
    )


def main() -> None:
    required = [
        RETAINED_ATLAS,
        CLEAN_ATLAS,
        PHASE,
        GOLDEN_RESULTS,
        CAUSAL_RESULTS,
        CA_MOTIF_STAGE1_RESULTS,
        CA_MOTIF_STAGE2_RESULTS,
        CA_COMPACT_RESULTS,
        CA_COMPACT_VERIFY,
        WAGNER_OCCURRENCE,
        WAGNER_CARRIER,
        WAGNER_SUMMARY,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing cross-model source artefacts:\n" + "\n".join(missing))

    retained_rows = read_csv(RETAINED_ATLAS)
    clean_rows = read_csv(CLEAN_ATLAS)
    retained = {int(row["rule"]): row for row in retained_rows}
    clean = {int(row["rule"]): row for row in clean_rows}
    if set(retained) != set(clean) or len(clean) != 88:
        raise RuntimeError("The two ECA atlases do not contain the same 88 rules")

    strict_pairs = [
        (float(retained[rule]["strict"]), float(clean[rule]["strict"]))
        for rule in sorted(clean)
    ]
    if any(left != right for left, right in strict_pairs):
        raise RuntimeError("The clean-room and retained strict atlases are not exact")

    golden = read_json(GOLDEN_RESULTS)
    atlas_comparison = golden["stages"]["atlas_comparison"]
    phase_comparison = golden["stages"]["phase_comparison"]
    particle = golden["stages"]["particle_observer"]["gates"]
    gates = read_json(CAUSAL_RESULTS)["adjudication"]["gates"]
    motif_stage1 = read_json(CA_MOTIF_STAGE1_RESULTS)["adjudication"]
    motif_stage2 = read_json(CA_MOTIF_STAGE2_RESULTS)["adjudication"]
    compact_result = read_json(CA_COMPACT_RESULTS)["adjudication"]
    compact_verify = read_json(CA_COMPACT_VERIFY)
    wagner_occurrence = read_json(WAGNER_OCCURRENCE)
    wagner_carrier = read_json(WAGNER_CARRIER)
    wagner_summary = read_json(WAGNER_SUMMARY)

    if not atlas_comparison["exact_reproduction"]:
        raise RuntimeError("Sealed atlas exact-reproduction gate is false")
    if gates["rule_specificity"] is not True:
        raise RuntimeError("Expected sealed rule-specificity gate to pass")
    if motif_stage1["verdict"] != "ROBUST_LOCAL_MOTIF_CONTROLLABILITY":
        raise RuntimeError("CA motif Stage-1 verdict changed unexpectedly")
    if motif_stage2["verdict"] != "DENSITY_ROBUST_GENERAL_MOTIF_CHANNEL":
        raise RuntimeError("CA motif Stage-2 verdict changed unexpectedly")
    if not motif_stage2["general"] or not motif_stage2["density_robust"]:
        raise RuntimeError("CA motif Stage-2 generalisation gates changed unexpectedly")
    if compact_result["verdict"] != "ROBUST_COMPACT_RENEWED_CA_PLASTIC_HEREDITY":
        raise RuntimeError("CA compact-carrier verdict changed unexpectedly")
    if not compact_result["target_replication_passed"] or not compact_verify["valid"]:
        raise RuntimeError("CA compact-carrier replication or verification no longer passes")
    if wagner_carrier["carrier_verdict"] != "LINEAGE_CARRIER_CONFIRMED":
        raise RuntimeError("Wagner carrier verdict changed unexpectedly")
    if wagner_carrier["causal_verdict"] != "CAUSAL_CARRIER_SUPPORTED":
        raise RuntimeError("Wagner causal verdict changed unexpectedly")
    if wagner_summary["overall_verdict"] != "WAGNER_MEMORY_STACK_CONFIRMED":
        raise RuntimeError("Wagner summary verdict changed unexpectedly")

    style()
    OUT.mkdir(parents=True, exist_ok=True)

    # Figure 9: quantitative occurrence-first evidence across both model families.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16.4, 11.0),
        gridspec_kw={"hspace": 0.42, "wspace": 0.34},
    )

    # A: exact raw-atlas reproduction.
    ax = axes[0, 0]
    class_colours = {1: BLUE, 2: TEAL, 3: ORANGE, 4: RED}
    for wolfram_class in (1, 2, 3, 4):
        points = [
            (left, right)
            for rule, (left, right) in zip(sorted(clean), strict_pairs)
            if int(clean[rule]["wolfram_class"]) == wolfram_class
        ]
        if not points:
            continue
        x, y = np.asarray(points).T
        ax.scatter(
            x,
            y,
            s=40,
            alpha=0.82,
            color=class_colours[wolfram_class],
            edgecolor=WHITE,
            linewidth=0.5,
            label=f"Class {wolfram_class}",
            zorder=3,
        )
    upper = max(max(value for pair in strict_pairs for value in pair), 0.45)
    ax.plot([0, upper], [0, upper], color=INK, linewidth=1.2, linestyle="--")
    ax.set_xlim(-0.012, upper * 1.03)
    ax.set_ylim(-0.012, upper * 1.03)
    ax.set_xlabel("Retained strict-event rate")
    ax.set_ylabel("Clean-room strict-event rate")
    ax.set_title(
        "A  Cellular automata: event detected first",
        loc="left",
        color=NAVY,
        fontsize=14,
    )
    ax.text(
        0.04,
        0.95,
        "88/88 rules exact\nSpearman rho = 1.000\n180,224 new futures",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": WHITE, "edgecolor": GRID},
    )
    ax.legend(frameon=False, loc="lower right", fontsize=9.5, ncol=2)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    stage2_primary = motif_stage2["primary_environments"]
    stage2_density = motif_stage2["stress_environments"]
    if len(stage2_primary) != 8 or len(stage2_density) != 3:
        raise RuntimeError("CA motif Stage-2 environment count changed unexpectedly")
    if not all(value["passed"] for value in [*stage2_primary.values(), *stage2_density.values()]):
        raise RuntimeError("A registered CA motif Stage-2 environment no longer passes")
    stage2_crossovers = [
        value["conditions"]["intact"]["crossover"]
        for value in [*stage2_primary.values(), *stage2_density.values()]
    ]
    stage2_terminal = [value["terminal"]["crossover"] for value in stage2_primary.values()]
    stage2_dose = motif_stage2["dose_response"]
    compact_walsh = compact_result["candidates"]["walsh-r016-q04"]["environments"]
    walsh_ordinary = compact_walsh["ordinary"]["strict"]
    walsh_moderate = compact_walsh["moderate_joint"]["strict"]
    if not walsh_ordinary["stage4_renewed_gate"] or not walsh_moderate["stage4_renewed_gate"]:
        raise RuntimeError("CA Walsh carrier no longer passes both registered environments")

    # B: CA dynamic bounds and later one-generation motif control.
    ax = axes[0, 1]
    phase_rows = read_csv(PHASE)
    etas = sorted({float(row["eta"]) for row in phase_rows})
    counts = {
        wolfram_class: [
            sum(
                1
                for row in phase_rows
                if float(row["eta"]) == eta
                and int(row["wolfram_class"]) == wolfram_class
                and bool_value(row["in_band"])
            )
            for eta in etas
        ]
        for wolfram_class in (1, 2, 3, 4)
    }
    positions = np.arange(len(etas))
    bottom = np.zeros(len(etas))
    for wolfram_class in (1, 2, 3, 4):
        values = np.asarray(counts[wolfram_class])
        ax.bar(
            positions,
            values,
            bottom=bottom,
            color=class_colours[wolfram_class],
            width=0.72,
            label=f"Class {wolfram_class}",
        )
        bottom += values
    ax.set_xticks(positions, [f"{eta:g}" for eta in etas])
    ax.set_xlabel("Per-sweep process-noise probability")
    ax.set_ylabel("Rules in capability band")
    ax.set_title(
        "B  Cellular automata: bounded control",
        loc="left",
        color=NAVY,
        fontsize=14,
    )
    ax.text(
        0.03,
        0.97,
        "Regime- and observer-bounded\n"
        f"five-noise rank rho = {phase_comparison['spearman']['strict']:.3f}\n"
        "Rule 110: 0.00% to 1.81%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": WHITE, "edgecolor": GRID},
    )
    ax.text(
        0.97,
        0.69,
        "Frozen motif reader (Stage 2)\n"
        "11/11 environments passed\n"
        f"crossover {min(stage2_crossovers):.3f}-{max(stage2_crossovers):.3f}; "
        f"dose rho={stage2_dose['spearman']:.3f}\n"
        "one generation; no rewrite test",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.7,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": PALE_GREEN, "edgecolor": TEAL},
    )
    if abs(float(particle["class4_strict"]["110"]) - 0.01806640625) > 1e-15:
        raise RuntimeError("Rule-110 particle result changed unexpectedly")
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9.0, ncol=2)

    # C: Wagner break-and-renewal occurrence before any carrier was added.
    ax = axes[1, 0]
    event_labels = ["F12 break + 3 renewals", "Strict F32 coherent 8"]
    event_values = [
        float(wagner_occurrence["evaluation_prevalence"]),
        float(wagner_occurrence["strict_f32_prevalence"]),
    ]
    bars = ax.barh(
        np.arange(2),
        event_values,
        color=[TEAL, GOLD],
        edgecolor=WHITE,
        height=0.58,
    )
    for bar, value in zip(bars, event_values):
        ax.text(
            value - 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{100 * value:.1f}%",
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=WHITE,
        )
    ax.set_yticks(np.arange(2), event_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.82)
    ax.set_xlabel("Fraction of 81,920 untouched futures")
    ax.set_title(
        "C  Wagner networks: event detected first",
        loc="left",
        color=NAVY,
        fontsize=14,
    )
    ax.text(
        0.98,
        0.08,
        "No carrier present\n"
        f"state-local reliability = {wagner_occurrence['split_half_reliability']:.3f}\n"
        f"history gain = {wagner_occurrence['history_log_loss_gain']['estimate']:.4f} nats\n"
        "below 0.020 advancement gate",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": WHITE, "edgecolor": GRID},
    )
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)

    # D: Wagner carrier persistence and causal controls.
    ax = axes[1, 1]
    metric_names = [
        "generation4_risk_gain",
        "g8.release_only.risk",
        "g8.neutral_damage.risk",
        "g8.forced_break.risk",
        "g16.release_only.risk",
        "g16.neutral_damage.risk",
        "g16.forced_break.risk",
    ]
    metric_labels = [
        "G4",
        "G8 release",
        "G8 damage",
        "G8 break",
        "G16 release",
        "G16 damage",
        "G16 break",
    ]
    estimates = [float(wagner_carrier["metrics"][name]["mean"]) for name in metric_names]
    lowers = [float(wagner_carrier["metrics"][name]["simultaneous_lower"]) for name in metric_names]
    positions = np.arange(len(estimates))
    colours = [BLUE] + [TEAL] * 3 + [ORANGE] * 3
    ax.bar(
        positions,
        estimates,
        color=colours,
        width=0.7,
        edgecolor=WHITE,
        zorder=2,
    )
    ax.errorbar(
        positions,
        estimates,
        yerr=[np.asarray(estimates) - np.asarray(lowers), np.zeros(len(estimates))],
        fmt="none",
        ecolor=INK,
        elinewidth=1.1,
        capsize=3,
        zorder=3,
    )
    ax.set_xticks(positions, metric_labels, rotation=28, ha="right")
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Matching-destination risk gain")
    ax.set_title(
        "D  Added Wagner carrier: multigenerational memory",
        loc="left",
        color=NAVY,
        fontsize=14,
    )
    ax.text(
        0.03,
        0.96,
        "Crossover G4/G8/G16 = 1.000\n"
        f"opposite-history reversal = {wagner_carrier['metrics']['opposite_reversal']['mean']:.3f}\n"
        f"ablation loss = {wagner_carrier['ablation_loss_fraction']:.3f}\n"
        f"rescue = {wagner_carrier['rescue_fraction']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": PALE_GREEN, "edgecolor": TEAL},
    )
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Cross-model evidence: break-and-renewal came before carrier tests",
        fontsize=19,
        fontweight="bold",
        color=NAVY,
        y=0.995,
    )
    figure9_path = OUT / "figure9_cross_model_evidence.png"
    fig.savefig(figure9_path, dpi=220, bbox_inches="tight", facecolor=WHITE)

    # V2 adds the later compact-carrier replication without changing Figure 9.
    v2_ca_ax = axes[0, 1]
    v2_ca_ax.set_title(
        "B  Cellular automata: bounded occurrence and carriers",
        loc="left",
        color=NAVY,
        fontsize=14,
    )
    v2_ca_ax.text(
        0.97,
        0.43,
        "Compact Walsh carrier (direct replication)\n"
        f"G16 ordinary = {walsh_ordinary['intact_generation16']['mean']:.3f}\n"
        f"G16 moderate damage = {walsh_moderate['intact_generation16']['mean']:.3f}\n"
        "complete causal ladder passed in both",
        transform=v2_ca_ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.7,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": PALE_BLUE, "edgecolor": BLUE},
    )
    figure11_v2_path = OUT / "figure11_v2_cross_model_evidence.png"
    fig.savefig(figure11_v2_path, dpi=220, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)

    # Figure 10: explanatory comparison of the two later carrier interventions.
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.2), gridspec_kw={"wspace": 0.12})
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.text(
        0.02,
        0.95,
        "A  Cellular automaton: transient motif cue",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    rounded_box(
        ax,
        (0.08, 0.77),
        0.84,
        0.11,
        "Break-and-renewal detected first\n(no carrier manipulation)",
        face=PALE_BLUE,
        edge=BLUE,
        fontsize=11.5,
        weight="bold",
    )
    rounded_box(ax, (0.05, 0.49), 0.20, 0.13, "A-like or B-like\nlocal motif", face=PALE_GOLD, edge=GOLD, fontsize=11.0, weight="bold")
    rounded_box(ax, (0.39, 0.49), 0.22, 0.13, "Write once into\nidentical daughter", face=PALE_BLUE, edge=BLUE, fontsize=11.0)
    rounded_box(ax, (0.75, 0.49), 0.20, 0.13, "One-generation\nA/B-biased form", face=PALE_GREEN, edge=TEAL, fontsize=11.0, weight="bold")
    arrow(ax, (0.25, 0.555), (0.39, 0.555), colour=GOLD)
    ax.text(0.32, 0.585, "write", ha="center", va="bottom", color=MUTED, fontsize=10)
    arrow(ax, (0.61, 0.555), (0.75, 0.555), colour=BLUE)
    ax.text(0.68, 0.585, "read by CA dynamics", ha="center", va="bottom", color=MUTED, fontsize=10)
    rounded_box(
        ax,
        (0.31, 0.22),
        0.50,
        0.12,
        "Daughter rewrite and\nmultigenerational transmission\nnot tested",
        face=PALE_RED,
        edge=RED,
        fontsize=11.0,
        linestyle="--",
    )
    arrow(ax, (0.85, 0.49), (0.81, 0.34), colour=RED, linestyle="--", connectionstyle="arc3,rad=-0.15")
    ax.text(
        0.50,
        0.10,
        "Frozen reader: 11/11 environments passed\nDose rho=1.000; horizon one generation",
        ha="center",
        va="center",
        fontsize=11.5,
        color=INK,
    )

    ax = axes[1]
    ax.text(
        0.02,
        0.95,
        "B  Wagner network: renewable lineage carrier",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    rounded_box(
        ax,
        (0.08, 0.77),
        0.84,
        0.11,
        "Break-and-renewal detected first\n(unaugmented network)",
        face=PALE_BLUE,
        edge=BLUE,
        fontsize=11.5,
        weight="bold",
    )
    rounded_box(ax, (0.05, 0.50), 0.20, 0.13, "Founder adult\ntrajectory", face=PALE_GOLD, edge=GOLD, fontsize=11.0, weight="bold")
    rounded_box(ax, (0.31, 0.50), 0.20, 0.13, "Write distributed\nhysteretic latch", face=PALE_BLUE, edge=BLUE, fontsize=11.0)
    rounded_box(ax, (0.57, 0.50), 0.16, 0.13, "Bottleneck\ntransmission", face=PALE_BLUE, edge=BLUE, fontsize=11.0)
    rounded_box(ax, (0.79, 0.50), 0.16, 0.13, "Descendant\nreads latch", face=PALE_GREEN, edge=TEAL, fontsize=11.0)
    arrow(ax, (0.25, 0.565), (0.31, 0.565), colour=GOLD)
    arrow(ax, (0.51, 0.565), (0.57, 0.565), colour=BLUE)
    arrow(ax, (0.73, 0.565), (0.79, 0.565), colour=BLUE)
    rounded_box(ax, (0.68, 0.25), 0.24, 0.13, "Adult A/B form\nrewrites latch", face=PALE_GREEN, edge=TEAL, fontsize=11.0, weight="bold")
    arrow(ax, (0.87, 0.50), (0.82, 0.38), colour=TEAL, connectionstyle="arc3,rad=-0.1")
    arrow(ax, (0.68, 0.315), (0.39, 0.50), colour=TEAL, connectionstyle="arc3,rad=-0.28")
    ax.text(
        0.50,
        0.10,
        "Persists through generation 16\nReversal, ablation, and rescue passed",
        ha="center",
        va="center",
        fontsize=11.5,
        color=INK,
    )

    fig.suptitle(
        "From transient cue to renewable lineage carrier",
        fontsize=19,
        fontweight="bold",
        color=NAVY,
        y=0.985,
    )
    footer = fig.text(
        0.5,
        0.012,
        "Break-and-renewal preceded carrier manipulation; the added Wagner carrier demonstrated renewable multigenerational memory.",
        ha="center",
        va="bottom",
        fontsize=12,
        color=MUTED,
    )
    figure10_path = OUT / "figure10_carrier_mechanisms.png"
    fig.savefig(figure10_path, dpi=220, bbox_inches="tight", facecolor=WHITE)

    # V2 replaces the now-obsolete CA boundary with the compact-carrier result.
    ax = axes[0]
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.02,
        0.95,
        "A  Cellular automaton: cue to renewable carrier",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    rounded_box(
        ax,
        (0.08, 0.77),
        0.84,
        0.11,
        "Break-and-renewal detected first\n(no carrier manipulation)",
        face=PALE_BLUE,
        edge=BLUE,
        fontsize=11.5,
        weight="bold",
    )
    rounded_box(
        ax,
        (0.12, 0.57),
        0.76,
        0.11,
        "Frozen local-motif reader: one-generation form channel\n11/11 registered environments; rewriting not tested",
        face=PALE_GOLD,
        edge=GOLD,
        fontsize=10.7,
    )
    arrow(ax, (0.50, 0.77), (0.50, 0.68), colour=GOLD)
    rounded_box(ax, (0.03, 0.31), 0.25, 0.14, "Founder history\nencodes 64-bit\nWalsh payload", face=PALE_GOLD, edge=GOLD, fontsize=10.4, weight="bold")
    rounded_box(ax, (0.37, 0.31), 0.25, 0.14, "Boundary transfer\n+ registered\nlatent damage", face=PALE_BLUE, edge=BLUE, fontsize=10.4)
    rounded_box(ax, (0.71, 0.31), 0.26, 0.14, "Descendant reads\nand rewrites\npayload", face=PALE_GREEN, edge=TEAL, fontsize=10.4, weight="bold")
    arrow(ax, (0.28, 0.38), (0.37, 0.38), colour=GOLD)
    arrow(ax, (0.62, 0.38), (0.71, 0.38), colour=BLUE)
    arrow(ax, (0.82, 0.31), (0.18, 0.31), colour=TEAL, connectionstyle="arc3,rad=-0.32")
    ax.text(
        0.50,
        0.10,
        f"Generation-16 crossover: {walsh_ordinary['intact_generation16']['mean']:.3f} ordinary; "
        f"{walsh_moderate['intact_generation16']['mean']:.3f} moderate damage\n"
        "Rewriting, disabling, reversal, ablation, and rescue gates passed",
        ha="center",
        va="center",
        fontsize=10.8,
        color=INK,
    )
    fig.suptitle(
        "Renewable lineage carriers in two synthetic substrates",
        fontsize=19,
        fontweight="bold",
        color=NAVY,
        y=0.985,
    )
    footer.set_text(
        "Break-and-renewal preceded carrier manipulation; both later carriers were engineered, renewable, and substrate-specific."
    )
    figure12_v2_path = OUT / "figure12_v2_carrier_mechanisms.png"
    fig.savefig(figure12_v2_path, dpi=220, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)

    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in required}
    evidence_provenance = {
        "figure": figure9_path.name,
        "v2_figure": figure11_v2_path.name,
        "sources": source_hashes,
        "ca": {
            "rules": 88,
            "futures": 180224,
            "atlas_exact": True,
            "strict_spearman": 1.0,
            "phase_strict_spearman": phase_comparison["spearman"]["strict"],
            "rule_110_particle_strict": particle["class4_strict"]["110"],
            "motif_stage1_crossover": motif_stage1["conditions"]["intact"]["crossover"],
            "motif_stage1_ci95": motif_stage1["conditions"]["intact"]["ci"],
            "motif_stage2_verdict": motif_stage2["verdict"],
            "motif_stage2_primary_environments_passed": sum(
                value["passed"] for value in stage2_primary.values()
            ),
            "motif_stage2_density_environments_passed": sum(
                value["passed"] for value in stage2_density.values()
            ),
            "motif_stage2_intact_crossover_range": [
                min(stage2_crossovers),
                max(stage2_crossovers),
            ],
            "motif_stage2_terminal_crossover_range": [
                min(stage2_terminal),
                max(stage2_terminal),
            ],
            "motif_stage2_dose_spearman": stage2_dose["spearman"],
            "motif_stage2_dose_crossover": {
                dose: values["crossover"]
                for dose, values in stage2_dose["contrasts"].items()
            },
            "compact_carrier": {
                "verdict": compact_result["verdict"],
                "target_replication_passed": compact_result[
                    "target_replication_passed"
                ],
                "target_replication_contract": compact_result[
                    "target_replication_contract"
                ],
                "walsh_ordinary_generation8": walsh_ordinary[
                    "intact_generation8"
                ],
                "walsh_ordinary_generation16": walsh_ordinary[
                    "intact_generation16"
                ],
                "walsh_moderate_generation8": walsh_moderate[
                    "intact_generation8"
                ],
                "walsh_moderate_generation16": walsh_moderate[
                    "intact_generation16"
                ],
                "checkpoint_count": compact_verify["checkpoint_count"],
                "expected_checkpoint_count": compact_verify[
                    "expected_checkpoint_count"
                ],
                "source_results_or_checkpoints_used": compact_verify[
                    "source_results_or_checkpoints_used"
                ],
                "verification_valid": compact_verify["valid"],
            },
        },
        "wagner_occurrence": wagner_occurrence,
        "wagner_carrier": {
            "futures": wagner_carrier["futures"],
            "carrier_verdict": wagner_carrier["carrier_verdict"],
            "causal_verdict": wagner_carrier["causal_verdict"],
            "distributed_verdict": wagner_carrier["distributed_verdict"],
            "metrics": {name: wagner_carrier["metrics"][name] for name in metric_names},
            "opposite_reversal": wagner_carrier["metrics"]["opposite_reversal"],
            "ablation_loss_fraction": wagner_carrier["ablation_loss_fraction"],
            "rescue_fraction": wagner_carrier["rescue_fraction"],
        },
    }
    with (OUT / "figure9_cross_model_evidence_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(evidence_provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")

    mechanism_provenance = {
        "figure": figure10_path.name,
        "v2_figure": figure12_v2_path.name,
        "sources": source_hashes,
        "ordering": [
            "cellular-automaton break-and-renewal occurrence",
            "cellular-automaton one-generation motif intervention",
            "cellular-automaton frozen-reader generalisation",
            "cellular-automaton compact renewable-carrier direct replication",
            "unaugmented Wagner break-and-renewal occurrence",
            "augmented Wagner renewable-carrier intervention",
        ],
        "claim_boundary": {
            "cellular_automaton": "one-generation motif control followed by an outcome-known, source-isolated direct replication of an engineered compact renewable carrier; synthetic same-substrate sufficiency, not spontaneous origin or universal necessity",
            "wagner": "engineered renewable-carrier sufficiency; not spontaneous acquisition or universal necessity",
        },
    }
    with (OUT / "figure10_carrier_mechanisms_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(mechanism_provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(figure9_path)
    print(figure11_v2_path)
    print(OUT / "figure9_cross_model_evidence_provenance.json")
    print(figure10_path)
    print(figure12_v2_path)
    print(OUT / "figure10_carrier_mechanisms_provenance.json")


if __name__ == "__main__":
    main()
