#!/usr/bin/env python3
"""Re-derive the F12 descriptive statistics quoted in the manuscript.

This script reads retained outputs from the originating workflow and the two
clean-room implementations.  It does not modify any scientific result
bundle.  The sole output is a deterministic JSON provenance record beside
this script.

The originating L54 report retains Jeffreys-smoothed state probabilities
rather than its branch table.  Its raw event counts are recovered algebraically
from the reported mean, the retained state count and the registered 64-future
budget.  The script verifies that the implied integer reproduces the reported
mean at its retained precision.  The clean-room-test prevalences are counted
directly from their retained branch records.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper_assets/f12_descriptive_provenance.json"

ORIGIN_REPORT = ROOT / (
    "original.1.8.2026.eidosoma-ai-scientist.stepReports/artifacts/"
    "research_steps/S19/loops/L54/S19_L54_FULL_RESULTS.md"
)
CODEX_BRANCHES = ROOT / (
    "replicators.13.8.2026.codex/results/full/confirmation_branches.csv.gz"
)
FABLE_CONFIRMATION = ROOT / (
    "replicators.13.8.2026.fable/replication/results/conf_data.pkl"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def markdown_table(text: str, heading: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    target = f"## {heading}"
    try:
        start = lines.index(target) + 1
    except ValueError as exc:
        raise RuntimeError(f"Missing Markdown table {target!r}") from exc

    table_lines: list[str] = []
    in_table = False
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            table_lines.append(line)
            in_table = True
        elif in_table and line.strip():
            break
    if len(table_lines) < 3:
        raise RuntimeError(f"No rows found below {target!r}")

    def cells(line: str) -> list[str]:
        return [item.strip() for item in line.strip().strip("|").split("|")]

    headers = cells(table_lines[0])
    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        values = cells(line)
        if len(values) == len(headers):
            rows.append(dict(zip(headers, values)))
    return rows


def candidate_short(candidate_id: str) -> str:
    return candidate_id.rsplit("-", maxsplit=1)[-1]


def derive_originating() -> dict[str, Any]:
    text = ORIGIN_REPORT.read_text(encoding="utf-8")
    total_match = re.search(r"([\d,]+) branch futures", text)
    if not total_match:
        raise RuntimeError("Could not recover the L54 total branch count")
    total_futures = int(total_match.group(1).replace(",", ""))

    process_rows = markdown_table(text, "Process probabilities by horizon")
    process_rows = [
        row
        for row in process_rows
        if int(row["horizon"]) == 12
        and row["targetType"] == "JOINT_BREAK_RUN3"
    ]
    if len(process_rows) != 2:
        raise RuntimeError("Expected one L54 F12 process row per candidate")

    total_states = sum(int(row["states"]) for row in process_rows)
    if total_futures % total_states:
        raise RuntimeError("L54 future count is not divisible by its state count")
    futures_per_state = total_futures // total_states
    if futures_per_state != 64:
        raise RuntimeError(f"Unexpected L54 branch budget: {futures_per_state}")

    reliability_rows = markdown_table(text, "Independent branch-half reliability")
    prediction_rows = markdown_table(text, "F12 joint-event predictive metrics")
    comparison_rows = markdown_table(text, "Registered proper-score comparisons")

    output: dict[str, Any] = {}
    relative_reductions: list[float] = []
    history_losses: list[float] = []
    for process in process_rows:
        candidate = candidate_short(process["candidateId"])
        states = int(process["states"])
        mean_q = float(process["meanQ"])
        sd_q = float(process["sdQ"])
        successes_float = mean_q * states * (futures_per_state + 1) - 0.5 * states
        successes = round(successes_float)
        implied_mean_q = (successes + 0.5 * states) / (
            states * (futures_per_state + 1)
        )
        if abs(implied_mean_q - mean_q) > 5e-8:
            raise RuntimeError(
                f"Candidate {candidate}: integer count does not reproduce retained meanQ"
            )

        reliability = next(
            row
            for row in reliability_rows
            if candidate_short(row["candidateId"]) == candidate
        )
        direct_rows = [
            row
            for row in prediction_rows
            if candidate_short(row["candidateId"]) == candidate
            and row["targetType"] == "JOINT_BREAK_RUN3"
            and row["modelId"] == "DIRECT_HISTORY_PHASE"
        ]
        gain_rows = [
            row
            for row in comparison_rows
            if candidate_short(row["candidateId"]) == candidate
            and row["comparisonId"] == "FULL_VS_DIRECT"
        ]
        if len(direct_rows) != 2 or len(gain_rows) != 2:
            raise RuntimeError(f"Candidate {candidate}: incomplete directional score rows")

        direct_by_direction = {
            row["direction"]: float(row["equalMatrixMeanBranchLogLoss"])
            for row in direct_rows
        }
        gain_by_direction = {
            row["direction"]: float(row["logLossImprovement"])
            for row in gain_rows
        }
        reduction_by_direction = {
            direction: 100.0 * gain_by_direction[direction] / loss
            for direction, loss in direct_by_direction.items()
        }
        history_losses.extend(direct_by_direction.values())
        relative_reductions.extend(reduction_by_direction.values())

        raw_futures = states * futures_per_state
        output[candidate] = {
            "states": states,
            "futures_per_state": futures_per_state,
            "event_successes": successes,
            "event_futures": raw_futures,
            "raw_event_prevalence": successes / raw_futures,
            "jeffreys_q_mean_reported": mean_q,
            "jeffreys_q_sd_reported": sd_q,
            "transition_region_states": int(reliability["intermediateProbabilityStates"]),
            "direct_history_log_loss": direct_by_direction,
            "full_vs_direct_log_loss_gain": gain_by_direction,
            "relative_log_loss_reduction_percent": reduction_by_direction,
        }

    return {
        "source": {
            "path": relative(ORIGIN_REPORT),
            "sha256": sha256(ORIGIN_REPORT),
        },
        "candidates": output,
        "history_log_loss_range": [min(history_losses), max(history_losses)],
        "relative_log_loss_reduction_percent_range": [
            min(relative_reductions),
            max(relative_reductions),
        ],
    }


def count_codex_prevalence() -> dict[str, Any]:
    counts = {"02": 0, "03": 0}
    totals = {"02": 0, "03": 0}
    states = {"02": set(), "03": set()}
    with gzip.open(CODEX_BRANCHES, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            candidate = row["candidate"]
            if candidate not in counts:
                raise RuntimeError(f"Unexpected Codex candidate {candidate!r}")
            value = int(row["joint_break_run3"])
            if value not in (0, 1):
                raise RuntimeError("Codex event column is not binary")
            counts[candidate] += value
            totals[candidate] += 1
            states[candidate].add(row["state_id"])

    result: dict[str, Any] = {}
    for candidate in counts:
        if totals[candidate] != 12_800 or len(states[candidate]) != 200:
            raise RuntimeError(f"Unexpected Codex denominator for candidate {candidate}")
        result[candidate] = {
            "states": len(states[candidate]),
            "event_successes": counts[candidate],
            "event_futures": totals[candidate],
            "raw_event_prevalence": counts[candidate] / totals[candidate],
        }
    return {
        "source": {"path": relative(CODEX_BRANCHES), "sha256": sha256(CODEX_BRANCHES)},
        "candidates": result,
    }


def count_fable_prevalence() -> dict[str, Any]:
    # This is a trusted, retained project artefact.  Loading arbitrary pickle
    # files from untrusted sources would be unsafe.
    with FABLE_CONFIRMATION.open("rb") as handle:
        bundle = pickle.load(handle)
    rows = bundle.get("table")
    if not isinstance(rows, list):
        raise RuntimeError("Fable confirmation bundle has no table list")

    result: dict[str, Any] = {}
    for candidate in ("02", "03"):
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        if len(candidate_rows) != 200:
            raise RuntimeError(f"Unexpected Fable state count for candidate {candidate}")
        successes = 0
        futures = 0
        for row in candidate_rows:
            outcomes = [int(value) for value in row["y64"]]
            if len(outcomes) != 64 or any(value not in (0, 1) for value in outcomes):
                raise RuntimeError("Fable y64 branch vector is not 64 binary outcomes")
            successes += sum(outcomes)
            futures += len(outcomes)
        result[candidate] = {
            "states": len(candidate_rows),
            "event_successes": successes,
            "event_futures": futures,
            "raw_event_prevalence": successes / futures,
        }
    return {
        "source": {
            "path": relative(FABLE_CONFIRMATION),
            "sha256": sha256(FABLE_CONFIRMATION),
        },
        "candidates": result,
    }


def build_record() -> dict[str, Any]:
    originating = derive_originating()
    codex = count_codex_prevalence()
    fable = count_fable_prevalence()
    independent_rates = [
        cell["raw_event_prevalence"]
        for implementation in (codex, fable)
        for cell in implementation["candidates"].values()
    ]

    origin_candidates = originating["candidates"]
    summaries = {
        "originating_event_counts": {
            candidate: {
                "successes": cell["event_successes"],
                "futures": cell["event_futures"],
            }
            for candidate, cell in origin_candidates.items()
        },
        "originating_jeffreys_q_mean_sd_rounded_3dp": {
            candidate: {
                "mean": round(cell["jeffreys_q_mean_reported"], 3),
                "sd": round(cell["jeffreys_q_sd_reported"], 3),
            }
            for candidate, cell in origin_candidates.items()
        },
        "originating_direct_history_log_loss_range_rounded_3dp": [
            round(value, 3) for value in originating["history_log_loss_range"]
        ],
        "originating_relative_log_loss_reduction_percent_range_rounded_1dp": [
            round(value, 1)
            for value in originating["relative_log_loss_reduction_percent_range"]
        ],
        "independent_test_raw_prevalence_percent_range_rounded_1dp": [
            round(100.0 * min(independent_rates), 1),
            round(100.0 * max(independent_rates), 1),
        ],
    }

    expected = {
        "originating_event_counts": {
            "02": {"successes": 4535, "futures": 12800},
            "03": {"successes": 4843, "futures": 12800},
        },
        "originating_jeffreys_q_mean_sd_rounded_3dp": {
            "02": {"mean": 0.357, "sd": 0.283},
            "03": {"mean": 0.38, "sd": 0.269},
        },
        "originating_direct_history_log_loss_range_rounded_3dp": [0.51, 0.56],
        "originating_relative_log_loss_reduction_percent_range_rounded_1dp": [
            8.1,
            9.3,
        ],
        "independent_test_raw_prevalence_percent_range_rounded_1dp": [33.0, 38.8],
    }
    if summaries != expected:
        raise RuntimeError(
            "Retained inputs no longer reproduce the manuscript's descriptive claims:\n"
            + json.dumps(summaries, indent=2, sort_keys=True)
        )

    return {
        "schema_version": 1,
        "endpoint": "JOINT_BREAK_RUN3 within F12",
        "derivation_notes": {
            "originating_event_count": (
                "round(mean Jeffreys q * states * (futures_per_state + 1) "
                "- 0.5 * states); the implied mean must reproduce the retained "
                "seven-decimal meanQ"
            ),
            "raw_prevalence": "event_successes / event_futures",
            "relative_log_loss_reduction_percent": (
                "100 * full_vs_direct_log_loss_gain / direct_history_log_loss, "
                "paired within candidate and branch direction"
            ),
        },
        "originating_confirmation": originating,
        "independent_test_1_confirmation": codex,
        "independent_test_2_confirmation": fable,
        "manuscript_summaries": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the retained JSON does not exactly match a fresh derivation",
    )
    args = parser.parse_args()

    record = build_record()
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
            print(
                f"Stale or missing provenance record: {OUTPUT.relative_to(ROOT)}",
                file=sys.stderr,
            )
            return 1
        print(f"Verified {OUTPUT.relative_to(ROOT)}")
        return 0

    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    print(json.dumps(record["manuscript_summaries"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
