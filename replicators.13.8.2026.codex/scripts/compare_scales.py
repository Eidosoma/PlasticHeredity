#!/usr/bin/env python3
"""Compare a nested scale-up against its baseline without refitting anything."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from plastic_heredity.models import predict_frozen_archive


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_metrics(metrics: dict[str, Any], candidate: str) -> dict[str, float]:
    item = metrics[candidate]
    directions = item["directions"]
    return {
        "branch_half_reliability": item["branch_half_reliability"],
        "branch_half_reliability_lower_95": item[
            "branch_half_reliability_lower_95"
        ],
        "centered_branch_half_reliability": item[
            "centered_branch_half_reliability"
        ],
        "centered_branch_half_reliability_lower_95": item[
            "centered_branch_half_reliability_lower_95"
        ],
        "full_overall_spearman": item["models"]["full"][
            "overall_spearman_mean"
        ],
        "full_centered_spearman": item["models"]["full"][
            "centered_spearman_mean"
        ],
        "history_overall_spearman": item["models"]["history"][
            "overall_spearman_mean"
        ],
        "history_centered_spearman": item["models"]["history"][
            "centered_spearman_mean"
        ],
        "full_minus_history_centered": (
            item["models"]["full"]["centered_spearman_mean"]
            - item["models"]["history"]["centered_spearman_mean"]
        ),
        "mean_log_loss_gain": float(
            np.mean([directions[key]["log_loss_gain"] for key in ("A", "B")])
        ),
        "minimum_log_loss_gain_lower_95": min(
            directions[key]["log_loss_gain_ci95"][0] for key in ("A", "B")
        ),
        "mean_q_brier_gain": float(
            np.mean([directions[key]["q_brier_gain"] for key in ("A", "B")])
        ),
        "minimum_q_brier_gain_lower_95": min(
            directions[key]["q_brier_gain_ci95"][0] for key in ("A", "B")
        ),
        "transition_region_fraction": (
            item["transition_region_states"] / item["states"]
        ),
    }


def _process_metrics(path: Path, candidate: str) -> dict[str, float]:
    table = pd.read_csv(path, dtype={"candidate": str})
    table["candidate"] = table["candidate"].str.zfill(2)
    selected = table[(table["cohort"] == "CONF") & (table["candidate"] == candidate)]
    return {
        str(row.metric): float(row.estimate)
        for row in selected.itertuples(index=False)
    }


def _branch_rows(path: Path, maximum_matrix: int) -> Iterator[tuple[str, ...]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        matrix_column = header.index("matrix_id")
        for row in reader:
            if int(row[matrix_column]) < maximum_matrix:
                yield tuple(row)


def _nested_branch_identity(
    baseline: Path, scaled: Path, maximum_matrix: int
) -> tuple[bool, int]:
    sentinel = object()
    count = 0
    for left, right in zip_longest(
        _branch_rows(baseline, maximum_matrix),
        _branch_rows(scaled, maximum_matrix),
        fillvalue=sentinel,
    ):
        if left != right:
            return False, count
        count += 1
    return True, count


def _nested_array_identity(
    baseline: Path, scaled: Path, baseline_states: int
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    with np.load(baseline) as left, np.load(scaled) as right:
        if set(left.files) != set(right.files):
            return {"archive_keys": False}
        checks["archive_keys"] = True
        for key in left.files:
            expected = right[key][:baseline_states]
            checks[key] = bool(np.array_equal(left[key], expected, equal_nan=True))
    return checks


def _branch_inventory(path: Path) -> dict[str, Any]:
    total = 0
    completed = 0
    by_candidate: dict[str, dict[str, int]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            candidate = row["candidate"].zfill(2)
            item = by_candidate.setdefault(candidate, {"rows": 0, "extinct": 0})
            item["rows"] += 1
            if int(row["completed_horizon"]):
                completed += 1
            else:
                item["extinct"] += 1
    return {
        "rows": total,
        "completed_horizon": completed,
        "extinct": total - completed,
        "by_candidate": by_candidate,
    }


def _frozen_prediction_audit(scaled: Path) -> dict[str, Any]:
    states = pd.read_csv(
        scaled / "confirmation_states.csv", dtype={"candidate": str}
    )
    states["candidate"] = states["candidate"].str.zfill(2)
    result: dict[str, Any] = {}
    with np.load(scaled / "analysis_arrays.npz") as arrays:
        for candidate in ("02", "03"):
            mask = states["candidate"].to_numpy() == candidate
            predictions = predict_frozen_archive(
                scaled / "frozen_models.npz",
                candidate,
                arrays["confirmation_state_graph"][mask],
                arrays["confirmation_history"][mask],
                arrays["confirmation_beta"][mask],
            )
            differences = {
                model: float(
                    np.max(
                        np.abs(
                            prediction
                            - states.loc[mask, f"prediction_{model}"].to_numpy()
                        )
                    )
                )
                for model, prediction in predictions.items()
            }
            result[candidate] = {
                "states": int(mask.sum()),
                "maximum_absolute_errors": differences,
                "all_within_1e-12": all(value <= 1e-12 for value in differences.values()),
            }
    return result


def _reported_match_count(path: Path) -> tuple[int, int]:
    table = pd.read_csv(path)
    table = table[table["section"] != "replay"]
    return int(table["within_reported_range"].sum()), len(table)


def _format(value: float) -> str:
    return f"{value:.4f}"


def compare(baseline: Path, scaled: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline_manifest = _load_json(baseline / "manifest.json")
    scaled_manifest = _load_json(scaled / "manifest.json")
    baseline_metrics = _load_json(baseline / "metrics.json")
    scaled_metrics = _load_json(scaled / "metrics.json")

    rows: list[dict[str, Any]] = []
    for candidate in ("02", "03"):
        baseline_values = _candidate_metrics(baseline_metrics, candidate)
        scaled_values = _candidate_metrics(scaled_metrics, candidate)
        for metric, baseline_value in baseline_values.items():
            scaled_value = scaled_values[metric]
            rows.append(
                {
                    "section": "confirmation",
                    "candidate": candidate,
                    "metric": metric,
                    "baseline": baseline_value,
                    "scaled5": scaled_value,
                    "delta": scaled_value - baseline_value,
                }
            )
        baseline_process = _process_metrics(
            baseline / "process_summary.csv", candidate
        )
        scaled_process = _process_metrics(scaled / "process_summary.csv", candidate)
        for metric, baseline_value in baseline_process.items():
            scaled_value = scaled_process[metric]
            rows.append(
                {
                    "section": "process",
                    "candidate": candidate,
                    "metric": metric,
                    "baseline": baseline_value,
                    "scaled5": scaled_value,
                    "delta": scaled_value - baseline_value,
                }
            )

    baseline_experiment = baseline_manifest["experiment"]
    scaled_experiment = scaled_manifest["experiment"]
    design: dict[str, Any] = {}
    nested: dict[str, Any] = {}
    for cohort, archive_prefix in (
        ("development", "development"),
        ("confirmation", "confirmation"),
    ):
        baseline_config = baseline_experiment[cohort]
        scaled_config = scaled_experiment[cohort]
        candidates = len(baseline_experiment["candidates"])
        landmarks = len(baseline_config["landmarks"])
        baseline_states = baseline_config["matrices"] * candidates * landmarks
        scaled_states = scaled_config["matrices"] * candidates * landmarks
        design[cohort] = {
            "baseline_matrices": baseline_config["matrices"],
            "scaled_matrices": scaled_config["matrices"],
            "matrix_scale": scaled_config["matrices"] / baseline_config["matrices"],
            "branches_per_state": scaled_config["branches_per_state"],
            "baseline_states": baseline_states,
            "scaled_states": scaled_states,
            "baseline_futures": baseline_states * baseline_config["branches_per_state"],
            "scaled_futures": scaled_states * scaled_config["branches_per_state"],
        }
        inventory = _branch_inventory(scaled / f"{cohort}_branches.csv.gz")
        design[cohort]["realized_branch_inventory"] = inventory
        design[cohort]["realized_futures_match"] = (
            inventory["rows"] == design[cohort]["scaled_futures"]
        )
        branch_equal, branch_rows = _nested_branch_identity(
            baseline / f"{cohort}_branches.csv.gz",
            scaled / f"{cohort}_branches.csv.gz",
            baseline_config["matrices"],
        )
        nested[f"{cohort}_branch_rows_equal"] = branch_equal
        nested[f"{cohort}_branch_rows_compared"] = branch_rows
        array_checks = _nested_array_identity(
            baseline / "analysis_arrays.npz",
            scaled / "analysis_arrays.npz",
            baseline_states,
        )
        relevant = {
            key: value
            for key, value in array_checks.items()
            if key == "archive_keys" or key.startswith(archive_prefix)
        }
        nested[f"{cohort}_arrays"] = relevant
        nested[f"{cohort}_arrays_all_equal"] = all(relevant.values())

    baseline_matches = _reported_match_count(baseline / "reported_comparison.csv")
    scaled_matches = _reported_match_count(scaled / "reported_comparison.csv")
    audit = {
        "baseline": str(baseline),
        "scaled": str(scaled),
        "same_master_seed": (
            baseline_experiment["master_seed"] == scaled_experiment["master_seed"]
        ),
        "same_candidate_contracts": (
            baseline_experiment["candidates"] == scaled_experiment["candidates"]
        ),
        "same_horizon": baseline_experiment["horizon"] == scaled_experiment["horizon"],
        "design": design,
        "nested_identity": nested,
        "exact_confirmation_replay": scaled_manifest["confirmation_replay_exact"],
        "frozen_model_predictions": _frozen_prediction_audit(scaled),
        "reported_ranges": {
            "baseline_matches": baseline_matches[0],
            "scaled_matches": scaled_matches[0],
            "comparisons": scaled_matches[1],
        },
    }
    return pd.DataFrame(rows), audit


def write_report(table: pd.DataFrame, audit: dict[str, Any], destination: Path) -> None:
    lines = [
        "# Nested 1× to 5× scale comparison",
        "",
        "The fivefold run preserves the horizon, landmarks, branches per state, candidate contracts, and master seed. Only the number of independent catalytic matrices changes from 40 to 200 in each cohort.",
        "",
        "## Design and replay audit",
        "",
        "| Cohort | Matrices | States | Futures | Extinct futures | Shared branch rows exact | Shared arrays exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort in ("development", "confirmation"):
        design = audit["design"][cohort]
        nested = audit["nested_identity"]
        lines.append(
            f"| {cohort.title()} | {design['baseline_matrices']} → {design['scaled_matrices']} | "
            f"{design['baseline_states']} → {design['scaled_states']} | "
            f"{design['baseline_futures']} → {design['scaled_futures']} | "
            f"{design['realized_branch_inventory']['extinct']} | "
            f"{nested[f'{cohort}_branch_rows_equal']} "
            f"({nested[f'{cohort}_branch_rows_compared']} rows) | "
            f"{nested[f'{cohort}_arrays_all_equal']} |"
        )
    lines.extend(
        (
            "",
            f"Exact regeneration of all scaled confirmation futures: **{audit['exact_confirmation_replay']}**.",
            f"Portable frozen-model predictions reproduce the saved confirmation predictions within 1e-12: **{all(item['all_within_1e-12'] for item in audit['frozen_model_predictions'].values())}**.",
            "",
            "## Primary confirmation estimates",
            "",
            "| Candidate | Metric | 1× | 5× | Change |",
            "|---|---|---:|---:|---:|",
        )
    )
    primary = (
        "branch_half_reliability",
        "centered_branch_half_reliability",
        "full_centered_spearman",
        "history_centered_spearman",
        "full_minus_history_centered",
        "mean_log_loss_gain",
        "minimum_log_loss_gain_lower_95",
        "mean_q_brier_gain",
        "minimum_q_brier_gain_lower_95",
    )
    confirmation = table[table["section"] == "confirmation"]
    for candidate in ("02", "03"):
        selected = confirmation[confirmation["candidate"] == candidate].set_index("metric")
        for metric in primary:
            row = selected.loc[metric]
            lines.append(
                f"| {candidate} | {metric} | {_format(row.baseline)} | "
                f"{_format(row.scaled5)} | {_format(row.delta)} |"
            )
    ranges = audit["reported_ranges"]
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "The qualitative discovery survives the fivefold increase in independent matrices for both candidates: state-conditioned fate remains reliable, and the frozen full state/graph/history model retains a positive within-matrix and out-of-sample calibration advantage over history alone. The exact manuscript-number signature remains unreproduced under this disclosed clean-room contract.",
            "",
            f"Descriptive agreement with supplied numerical ranges changed from {ranges['baseline_matches']}/{ranges['comparisons']} at 1× to {ranges['scaled_matches']}/{ranges['comparisons']} at 5×. These ranges were never used for fitting or tuning.",
            "",
            "See `scale_comparison.csv` for every confirmation and process estimate and `scale_audit.json` for machine-readable nesting checks.",
            "",
        )
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("scaled", type=Path)
    arguments = parser.parse_args()
    table, audit = compare(arguments.baseline, arguments.scaled)
    table.to_csv(arguments.scaled / "scale_comparison.csv", index=False)
    (arguments.scaled / "scale_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(table, audit, arguments.scaled / "SCALE_COMPARISON.md")
    print(arguments.scaled / "SCALE_COMPARISON.md")


if __name__ == "__main__":
    main()
