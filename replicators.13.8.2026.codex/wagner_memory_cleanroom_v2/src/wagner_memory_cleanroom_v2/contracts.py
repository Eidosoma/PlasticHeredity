from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable

from .config import Registration, scaled_cell_futures, stage_source_count


BASE_STAGES = ("state", "boundary", "slow_mark", "carrier")
AUDIT_STAGES = ("state_audit", "carrier_audit")
CAMPAIGN_STAGES = BASE_STAGES + AUDIT_STAGES


@dataclass(frozen=True)
class StageExpectation:
    stage: str
    sources: int
    cells_per_source: int
    futures_per_source: int

    @property
    def cells(self) -> int:
        return self.sources * self.cells_per_source

    @property
    def futures(self) -> int:
        return self.sources * self.futures_per_source


def base_stage(stage: str) -> str:
    value = stage.removesuffix("_audit")
    if value not in BASE_STAGES:
        raise ValueError(f"unknown stage: {stage}")
    return value


def _state_per_source(registration: Registration) -> tuple[int, int]:
    protocol = registration.protocol["state"]
    writers = len(protocol["writers"])
    histories = 2
    midpoints = int(registration.engine["midpoint_count"])
    halves = 2
    primary_n = scaled_cell_futures(int(protocol["primary_futures_per_cell"]), registration) // 2
    persistence_n = scaled_cell_futures(int(protocol["persistence_futures_per_cell"]), registration) // 2
    primary_cells = len(protocol["arms"]) + 2
    persistence_cells = 3 * len(protocol["challenges"]) * 4
    cells = writers * histories * midpoints * halves * (primary_cells + persistence_cells)
    futures = writers * histories * midpoints * halves * (
        primary_cells * primary_n + persistence_cells * persistence_n
    )
    return cells, futures


def _boundary_per_source(registration: Registration) -> tuple[int, int]:
    protocol = registration.protocol["boundary"]
    cells = (
        len(protocol["writers"])
        * len(protocol["thetas"])
        * 2
        * int(registration.engine["midpoint_count"])
        * 2
        * 2
    )
    half_n = scaled_cell_futures(int(protocol["futures_per_cell"]), registration) // 2
    return cells, cells * half_n


def _mark_per_source(registration: Registration) -> tuple[int, int]:
    protocol = registration.protocol["slow_mark"]
    outer = (
        len(protocol["half_lives"])
        * len(protocol["couplings"])
        * 2
        * int(registration.engine["midpoint_count"])
        * 2
    )
    screen_cells = (
        len(protocol["screen_arms"])
        * len(protocol["screen_challenges"])
        * len(protocol["ages"])
    )
    mechanism_cells = (
        len(protocol["mechanism_arms"])
        * len(protocol["mechanism_challenges"])
        * len(protocol["ages"])
    )
    screen_n = scaled_cell_futures(int(protocol["screen_futures_per_cell"]), registration) // 2
    mechanism_n = scaled_cell_futures(int(protocol["mechanism_futures_per_cell"]), registration) // 2
    return (
        outer * (screen_cells + mechanism_cells),
        outer * (screen_cells * screen_n + mechanism_cells * mechanism_n),
    )


def _carrier_per_source(registration: Registration) -> tuple[int, int]:
    protocol = registration.protocol["carrier"]
    cells = (
        2
        * int(registration.engine["midpoint_count"])
        * 2
        * len(protocol["arms"])
        * len(protocol["checkpoints"])
        * len(protocol["challenges"])
    )
    half_n = scaled_cell_futures(int(protocol["futures_per_cell"]), registration) // 2
    return cells, cells * half_n


def stage_expectation(stage: str, registration: Registration) -> StageExpectation:
    base = base_stage(stage)
    per_source = {
        "state": _state_per_source,
        "boundary": _boundary_per_source,
        "slow_mark": _mark_per_source,
        "carrier": _carrier_per_source,
    }[base](registration)
    return StageExpectation(
        stage=stage,
        sources=stage_source_count(stage, registration),
        cells_per_source=per_source[0],
        futures_per_source=per_source[1],
    )


def assigned_source_ids(stage: str, registration: Registration, worker: int, workers: int) -> list[int]:
    if workers <= 0 or not 0 <= worker < workers:
        raise ValueError("invalid worker partition")
    return list(range(worker, stage_source_count(stage, registration), workers))


def validate_protocol_counts(registration: Registration) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for stage in BASE_STAGES:
        expected = stage_expectation(stage, registration)
        registered = int(registration.protocol[stage]["expected_futures_full"])
        # Registered totals are deliberately full-profile constants even when a
        # smoke/quick profile is being used.
        full_profile = dict(registration.protocol["profiles"]["full"])
        full_registration = Registration(
            registration.protocol,
            registration.protocol_path,
            registration.protocol_digest,
            "full",
            full_profile,
        )
        computed = stage_expectation(stage, full_registration).futures
        checks[f"{stage}_full_futures"] = registered == computed
        details[stage] = {
            "registered_full_futures": registered,
            "computed_full_futures": computed,
            "profile_sources": expected.sources,
            "profile_cells": expected.cells,
            "profile_futures": expected.futures,
            "cells_per_source": expected.cells_per_source,
            "futures_per_source": expected.futures_per_source,
        }
    checks["full_state_cells"] = stage_expectation("state", full_registration).cells == 168_960
    checks["full_boundary_cells"] = stage_expectation("boundary", full_registration).cells == 18_432
    checks["full_mark_cells"] = stage_expectation("slow_mark", full_registration).cells == 221_184
    checks["full_carrier_cells"] = stage_expectation("carrier", full_registration).cells == 552_960
    return {"checks": checks, "details": details, "valid": all(checks.values())}


def validate_stage_records(
    stage: str,
    registration: Registration,
    records: Iterable[dict[str, Any]],
    source_records: Iterable[dict[str, Any]],
    worker: int,
    workers: int,
) -> dict[str, Any]:
    rows = list(records)
    sources = list(source_records)
    assigned = assigned_source_ids(stage, registration, worker, workers)
    expectation = stage_expectation(stage, registration)
    expected_cells = len(assigned) * expectation.cells_per_source
    expected_futures = len(assigned) * expectation.futures_per_source
    source_ids = [int(row["source_id"]) for row in sources]
    row_sources = Counter(int(row["source_id"]) for row in rows)
    source_futures: dict[int, int] = defaultdict(int)
    outcome_accounting = True
    stage_labels = True
    half_labels = True
    cell_ids: list[str] = []
    for row in rows:
        source_futures[int(row["source_id"])] += int(row["n"])
        outcome_accounting &= sum(
            int(row[name]) for name in ("dest_a", "dest_b", "dest_other")
        ) == int(row["n"])
        outcome_accounting &= all(
            0 <= int(row[name]) <= int(row["n"])
            for name in ("hold_a", "hold_b", "hold_both")
        )
        outcome_accounting &= int(row["hold_both"]) <= min(
            int(row["hold_a"]), int(row["hold_b"])
        )
        stage_labels &= row.get("stage") == base_stage(stage)
        half_labels &= int(row.get("half", -1)) in (0, 1)
        cell_ids.append(str(row["cell_id"]))
    checks = {
        "source_partition": source_ids == assigned,
        "source_records_unique": len(source_ids) == len(set(source_ids)),
        "record_sources_exact": set(row_sources) == set(assigned),
        "cells_exact": len(rows) == expected_cells,
        "cells_per_source": all(row_sources[source] == expectation.cells_per_source for source in assigned),
        "futures_exact": sum(int(row["n"]) for row in rows) == expected_futures,
        "futures_per_source": all(source_futures[source] == expectation.futures_per_source for source in assigned),
        "cell_ids_unique": len(cell_ids) == len(set(cell_ids)),
        "outcomes_partition_n": outcome_accounting,
        "base_stage_labels": stage_labels,
        "future_half_labels": half_labels,
    }
    digest = sha256("\n".join(sorted(cell_ids)).encode()).hexdigest()
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "worker": worker,
        "workers": workers,
        "assigned_sources": assigned,
        "observed_sources": source_ids,
        "expected_cells": expected_cells,
        "observed_cells": len(rows),
        "expected_futures": expected_futures,
        "observed_futures": sum(int(row["n"]) for row in rows),
        "cell_id_set_sha256": digest,
    }
