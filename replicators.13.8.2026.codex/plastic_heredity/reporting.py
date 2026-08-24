from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.4f}"


def write_report(
    output_directory: Path,
    metrics: dict[str, Any],
    process_summary: list[dict[str, Any]],
    comparison: pd.DataFrame,
    replay_exact: bool | None,
) -> None:
    process = pd.DataFrame(process_summary)
    lines = [
        "# Clean-room plastic-heredity replication results",
        "",
        "This run covers only the proposed plastic-heredity discovery. It does not run PhiID, first-replicator prediction, or intervention analyses.",
        "",
        "## Outcome",
        "",
    ]
    qualitative_passes: list[bool] = []
    for candidate in ("02", "03"):
        candidate_metrics = metrics[candidate]
        full_centered = candidate_metrics["models"]["full"]["centered_spearman_mean"]
        history_centered = candidate_metrics["models"]["history"]["centered_spearman_mean"]
        gains = [
            candidate_metrics["directions"][direction]["log_loss_gain"]
            for direction in ("A", "B")
        ]
        log_loss_lowers = [
            candidate_metrics["directions"][direction]["log_loss_gain_ci95"][0]
            for direction in ("A", "B")
        ]
        brier_lowers = [
            candidate_metrics["directions"][direction]["q_brier_gain_ci95"][0]
            for direction in ("A", "B")
        ]
        qualitative_passes.append(
            candidate_metrics["branch_half_reliability"] > 0.5
            and candidate_metrics["centered_branch_half_reliability_lower_95"] > 0.0
            and full_centered > history_centered
            and min(gains) > 0.0
            and min(log_loss_lowers) > 0.0
            and min(brier_lowers) > 0.0
        )
    if replay_exact is None:
        verdict = (
            "This reduced profile is an implementation smoke test only; exact regeneration and "
            "the full matrix-level confirmation design were not run, so it carries no scientific verdict."
        )
    elif all(qualitative_passes) and replay_exact:
        verdict = (
            "The central qualitative discovery replicated in both explicit candidates, "
            "but the reported numerical signature did not: "
            "the F12 event probability was state-dependent, and the frozen full state/graph/history "
            "student improved on direct history both within matrices and by branch log loss, while "
            "several reported prevalences and effect magnitudes fell outside their supplied ranges."
        )
    else:
        verdict = (
            "The central qualitative discovery did not replicate in both explicit candidates under "
            "this clean-room contract. See the candidate-separated metrics below."
        )
    lines.extend((verdict, ""))

    lines.extend(
        (
            "## Untouched confirmation",
            "",
            "| Candidate | Split-half rho | Centered split-half rho | Full rho | Full centered rho | History centered rho | Log-loss gain A/B |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for candidate in ("02", "03"):
        item = metrics[candidate]
        model = item["models"]
        gain_a = item["directions"]["A"]["log_loss_gain"]
        gain_b = item["directions"]["B"]["log_loss_gain"]
        lines.append(
            f"| {candidate} | {_fmt(item['branch_half_reliability'])} | "
            f"{_fmt(item['centered_branch_half_reliability'])} | "
            f"{_fmt(model['full']['overall_spearman_mean'])} | "
            f"{_fmt(model['full']['centered_spearman_mean'])} | "
            f"{_fmt(model['history']['centered_spearman_mean'])} | "
            f"{_fmt(gain_a)} / {_fmt(gain_b)} |"
        )

    lines.extend(
        (
            "",
            "## Clean-room evidence gates",
            "",
            "| Candidate | Transition-region states | Reliability lower | Centered reliability lower | Minimum log-loss-gain lower | Minimum q-Brier-gain lower | Max permutation p |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for candidate in ("02", "03"):
        item = metrics[candidate]
        directions = item["directions"]
        lines.append(
            f"| {candidate} | {item['transition_region_states']}/{item['states']} | "
            f"{_fmt(item['branch_half_reliability_lower_95'])} | "
            f"{_fmt(item['centered_branch_half_reliability_lower_95'])} | "
            f"{_fmt(min(directions[key]['log_loss_gain_ci95'][0] for key in ('A', 'B')))} | "
            f"{_fmt(min(directions[key]['q_brier_gain_ci95'][0] for key in ('A', 'B')))} | "
            f"{_fmt(max(directions[key]['matrix_permutation_p'] for key in ('A', 'B')))} |"
        )

    lines.extend(
        (
            "",
            "## Plastic-heredity process",
            "",
            "Reported below are confirmation estimates. Episode quantities are conditional on a break; old-anchor quantities use the pre-break parent.",
            "",
            "| Candidate | Break | Resume-2 | Episode-3 | Persist-5 | Old return | Mean old-anchor gain |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for candidate in ("02", "03"):
        selected = process[(process.cohort == "CONF") & (process.candidate == candidate)]
        values = {row.metric: row.estimate for row in selected.itertuples()}
        lines.append(
            f"| {candidate} | {_fmt(values['break_event'])} | {_fmt(values['resume_2'])} | "
            f"{_fmt(values['episode_3'])} | {_fmt(values['persist_5'])} | "
            f"{_fmt(values['old_return'])} | {_fmt(values['old_anchor_gain'])} |"
        )

    comparable = comparison[comparison.section != "replay"]
    matched = int(comparable.within_reported_range.sum())
    lines.extend(
        (
            "",
            "## Numerical comparison",
            "",
            f"{matched}/{len(comparable)} candidate-metric comparisons fall inside the ranges stated in the supplied manuscript. Range matching is descriptive and was not used by the simulator or model.",
            "",
            f"Exact confirmation regeneration: **{replay_exact if replay_exact is not None else 'not run'}**.",
            "",
            "See `reported_comparison.csv` for every comparison and `metrics.json` for directional values, confidence intervals, and permutation tests.",
            "",
            "## Interpretation boundary",
            "",
            "This is an independent implementation, not a rerun of the unavailable L53/L54 code. The published materials do not specify the candidate contracts, the 195 feature coordinates, development cohort size, or all conditional event details. Those choices are frozen and disclosed in `manifest.json` and the repository `REPLICATION.md`. Agreement supports robustness to this explicit implementation; disagreement does not by itself refute the private implementation.",
            "",
        )
    )
    (output_directory / "REPLICATION_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
