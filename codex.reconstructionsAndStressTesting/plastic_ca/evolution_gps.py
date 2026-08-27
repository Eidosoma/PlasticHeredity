"""Fresh-truth navigation on the eight-bit ECA rule hypercube."""

from __future__ import annotations

from collections import Counter, deque
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time
from typing import Any, Callable

from .eca import canonical_rule
from .evolution import EvolutionConfig, _read_atlas, run_evolution
from .rng import stream


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def top_census_forms(atlas_path: Path, count: int = 8) -> tuple[int, ...]:
    """Return forms by frequency, preserving first encounter for ties."""

    census: Counter[int] = Counter()
    with atlas_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for value in row.get("support_masks", "").split("|"):
                if value:
                    census[int(value)] += 1
    return tuple(form for form, _frequency in census.most_common(count))


def _expand_libraries(atlas_path: Path) -> dict[int, frozenset[int]]:
    _fitness, canonical = _read_atlas(atlas_path)
    return {rule: canonical[canonical_rule(rule)] for rule in range(256)}


def _shortest_path(
    start: int,
    holds_target: Callable[[int], bool],
    *,
    max_edits: int = 16,
) -> tuple[int, ...]:
    if holds_target(start):
        return (start,)
    queue: deque[tuple[int, ...]] = deque([(start,)])
    seen = {start}
    while queue:
        path = queue.popleft()
        if len(path) - 1 >= max_edits:
            continue
        current = path[-1]
        # Bit index is the declared deterministic tie break.
        for bit in range(8):
            neighbor = current ^ (1 << bit)
            if neighbor in seen:
                continue
            candidate = path + (neighbor,)
            if holds_target(neighbor):
                return candidate
            seen.add(neighbor)
            queue.append(candidate)
    return (start,)


def _trial_starts(form: int, dev: dict[int, frozenset[int]], count: int = 8) -> tuple[int, ...]:
    eligible = [rule for rule in range(256) if form not in dev[rule]]
    eligible.sort(
        key=lambda rule: hashlib.sha256(f"rulial-evo-gps-start-v1:{form}:{rule}".encode()).digest()
    )
    return tuple(eligible[:count])


def _random_route(
    start: int,
    form: int,
    trial_index: int,
    dev: dict[int, frozenset[int]],
    max_edits: int,
) -> tuple[int, ...]:
    rng = stream("rulial-evo-gps-random-v1", form, trial_index, start)
    path = [start]
    for _ in range(max_edits):
        current = path[-1]
        if form in dev[current]:
            break
        path.append(current ^ (1 << rng.randrange(8)))
    return tuple(path)


def evaluate_gps(
    dev_atlas: Path,
    truth_atlas: Path,
    *,
    max_edits: int = 16,
) -> dict[str, Any]:
    dev = _expand_libraries(dev_atlas)
    truth = _expand_libraries(truth_atlas)
    targets = top_census_forms(dev_atlas, 8)
    records: list[dict[str, Any]] = []
    for form in targets:
        for trial_index, start in enumerate(_trial_starts(form, dev, 8)):
            planned = _shortest_path(start, lambda rule, form=form: form in dev[rule], max_edits=max_edits)
            random_path = _random_route(start, form, trial_index, dev, max_edits)
            oracle = _shortest_path(start, lambda rule, form=form: form in truth[rule], max_edits=max_edits)
            records.append(
                {
                    "target_form": form,
                    "trial_index": trial_index,
                    "start_rule": start,
                    "planned_path": list(planned),
                    "planned_edits": len(planned) - 1,
                    "planned_success": form in truth[planned[-1]],
                    "random_path": list(random_path),
                    "random_edits": len(random_path) - 1,
                    "random_success": form in truth[random_path[-1]],
                    "oracle_path": list(oracle),
                    "oracle_edits": len(oracle) - 1,
                    "oracle_success": form in truth[oracle[-1]],
                }
            )
    n = len(records)
    planned_success = sum(bool(row["planned_success"]) for row in records) / n if n else 0.0
    random_success = sum(bool(row["random_success"]) for row in records) / n if n else 0.0
    oracle_success = sum(bool(row["oracle_success"]) for row in records) / n if n else 0.0

    def successful_median(prefix: str) -> float | None:
        values = [int(row[f"{prefix}_edits"]) for row in records if row[f"{prefix}_success"]]
        return float(median(values)) if values else None

    return {
        "gate_gps": planned_success > random_success
        and oracle_success > 0.0
        and planned_success >= 0.60 * oracle_success,
        "planned_success": planned_success,
        "random_walk_success": random_success,
        "oracle_success": oracle_success,
        "planned_median_edits": successful_median("planned"),
        "random_median_edits": successful_median("random"),
        "oracle_median_edits": successful_median("oracle"),
        "targets": list(targets),
        "n_trials": n,
        "max_edits": max_edits,
        "planning_data": "dev atlas only",
        "scoring_data": "fresh truth atlas only",
        "trials": records,
    }


def run_evolution_gps(
    dev_atlas: Path,
    truth_atlas: Path,
    output: Path,
    *,
    config: EvolutionConfig = EvolutionConfig(),
) -> dict[str, Any]:
    started = time.time()
    search = run_evolution(dev_atlas, output / "search", config)
    gps = evaluate_gps(dev_atlas, truth_atlas)
    summary = {
        "experiment": "rulial_evolution_fresh_truth_gps",
        "elapsed_seconds": time.time() - started,
        "config": search["config"],
        "evolution": search["evolution"],
        "gps": {key: value for key, value in gps.items() if key != "trials"},
        "boundary": "GPS planned on the frozen dev atlas and scored only on an independently simulated truth atlas",
    }
    _atomic_json(output / "gps_trials.json", {"trials": gps["trials"]})
    _atomic_json(output / "evolution_summary.json", summary)
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return summary

