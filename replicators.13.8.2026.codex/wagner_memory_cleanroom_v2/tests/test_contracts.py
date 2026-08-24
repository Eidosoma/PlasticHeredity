from __future__ import annotations

from dataclasses import replace

from wagner_memory_cleanroom_v2.config import load_registration, scaled_cell_futures
from wagner_memory_cleanroom_v2.contracts import (
    assigned_source_ids,
    stage_expectation,
    validate_protocol_counts,
    validate_stage_records,
)
from wagner_memory_cleanroom_v2.source import generate_rulebook


def test_full_registered_counts_are_exact() -> None:
    registration = load_registration("full")
    expected = {
        "state": (240, 704, 26_624, 6_389_760),
        "boundary": (96, 192, 3_072, 294_912),
        "slow_mark": (96, 2_304, 40_960, 3_932_160),
        "carrier": (240, 2_304, 36_864, 8_847_360),
    }
    for stage, values in expected.items():
        result = stage_expectation(stage, registration)
        assert (result.sources, result.cells_per_source, result.futures_per_source, result.futures) == values
    assert validate_protocol_counts(registration)["valid"]


def test_futures_per_cell_is_total_and_splits_evenly() -> None:
    smoke = load_registration("smoke")
    assert scaled_cell_futures(32, smoke) == 2
    assert scaled_cell_futures(64, smoke) == 4
    assert scaled_cell_futures(128, smoke) == 8


def test_two_worker_partition_is_disjoint_and_complete() -> None:
    registration = load_registration("full")
    left = assigned_source_ids("carrier", registration, 0, 2)
    right = assigned_source_ids("carrier", registration, 1, 2)
    assert not set(left) & set(right)
    assert sorted(left + right) == list(range(240))


def test_diagnostic_source_domains_generate_disjoint_rulebooks() -> None:
    registration = load_registration("full")
    smoke = generate_rulebook(0, registration.protocol, "smoke:state")
    benchmark = generate_rulebook(0, registration.protocol, "benchmark:smoke:state")
    validation = generate_rulebook(0, registration.protocol, "validation:state")
    digests = {
        row.proposal_log[-1]["weight_sha256"] for row in (smoke, benchmark, validation)
    }
    assert len(digests) == 3


def test_stage_contract_rejects_duplicate_cell_ids() -> None:
    registration = load_registration("smoke")
    profile = dict(registration.profile)
    profile["boundary_sources"] = 1
    tiny = replace(registration, profile=profile)
    expectation = stage_expectation("boundary", tiny)
    template = {
        "stage": "boundary",
        "source_id": 0,
        "half": 0,
        "n": 1,
        "dest_a": 0,
        "dest_b": 0,
        "dest_other": 1,
        "hold_a": 0,
        "hold_b": 0,
        "hold_both": 0,
        "cell_id": "duplicate",
    }
    rows = [dict(template) for _ in range(expectation.cells_per_source)]
    result = validate_stage_records("boundary", tiny, rows, [{"source_id": 0}], 0, 1)
    assert not result["valid"]
    assert not result["checks"]["cell_ids_unique"]
