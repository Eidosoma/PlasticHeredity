from __future__ import annotations

from pathlib import Path

from reviewer_motif_channel_replication.contract import FIXED_PRIMARY
from reviewer_motif_channel_replication.inference import (
    AssignmentAccumulator,
    adjudicate_stage1,
    bootstrap_interval,
)


def _metric(crossover: float) -> dict[str, float]:
    if crossover >= 0:
        return {
            "p_a_given_a": crossover,
            "p_a_given_b": 0.0,
            "p_b_given_a": 0.0,
            "p_b_given_b": crossover,
            "direction_a": crossover,
            "direction_b": crossover,
            "crossover": crossover,
            "correct": crossover,
            "resolved": crossover,
        }
    magnitude = -crossover
    return {
        "p_a_given_a": 0.0,
        "p_a_given_b": magnitude,
        "p_b_given_a": magnitude,
        "p_b_given_b": 0.0,
        "direction_a": crossover,
        "direction_b": crossover,
        "crossover": crossover,
        "correct": 0.0,
        "resolved": magnitude,
    }


def _ideal_payload(pair_id: str) -> dict:
    conditions = {}
    effects = {
        "intact": 0.9,
        "zero": 0.0,
        "read_disabled": 0.0,
        "shuffle": 0.0,
        "opposite_history": -0.9,
        "process_noise": 0.8,
        "carrier_sign_corruption": 0.8,
    }
    for condition, effect in effects.items():
        conditions[condition] = {
            "checkpoints": {
                "64": {
                    "primary": _metric(effect),
                    "terminal": _metric(0.8 if condition == "intact" else effect),
                    "survival": 1.0,
                }
            }
        }
    return {
        "pair_id": pair_id,
        "configurations": {
            FIXED_PRIMARY.configuration_id: {"conditions": conditions}
        },
    }


def test_assignment_accumulator_keeps_unresolved_in_denominator() -> None:
    accumulator = AssignmentAccumulator()
    accumulator.add("A", "A")
    accumulator.add("B", None)
    accumulator.add("A", None)
    accumulator.add("B", "B")
    result = accumulator.finish()
    assert result["p_a_given_a"] == 0.5
    assert result["p_b_given_b"] == 0.5
    assert result["crossover"] == 0.5
    assert result["resolved"] == 0.5


def test_incomplete_run_cannot_pass_even_with_ideal_observations() -> None:
    rows = [_ideal_payload(f"pair-{index}") for index in range(6)]
    result = adjudicate_stage1(
        rows,
        FIXED_PRIMARY.configuration_id,
        complete=False,
        namespace="test",
        resamples=100,
    )
    assert result["verdict"] == "INCOMPLETE"
    assert not result["controllable"]
    assert not result["robust"]


def test_registered_gate_passes_constant_ideal_pair_panel() -> None:
    rows = [_ideal_payload(f"pair-{index}") for index in range(6)]
    result = adjudicate_stage1(
        rows,
        FIXED_PRIMARY.configuration_id,
        complete=True,
        namespace="test-pass",
        resamples=100,
    )
    assert result["verdict"] == "ROBUST_LOCAL_MOTIF_CONTROLLABILITY"


def test_bootstrap_is_seed_domain_deterministic() -> None:
    rows = [1.0, 2.0, 3.0, 4.0]
    statistic = lambda sample: sum(sample) / len(sample)
    first = bootstrap_interval(
        rows,
        statistic,
        resamples=200,
        alpha=0.05,
        namespace="bootstrap-test",
        seed_parts=("one",),
    )
    second = bootstrap_interval(
        rows,
        statistic,
        resamples=200,
        alpha=0.05,
        namespace="bootstrap-test",
        seed_parts=("one",),
    )
    assert first == second
