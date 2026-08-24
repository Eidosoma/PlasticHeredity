from __future__ import annotations

from reviewer_ca_compact_carrier_replication.inference import (
    paired_advantage,
    summarize_assignments,
)


def test_pair_cluster_summary_uses_pairs_not_futures() -> None:
    rows = [
        {
            "p_a_given_a": 0.8,
            "p_a_given_b": 0.2,
            "p_b_given_a": 0.2,
            "p_b_given_b": 0.8,
            "direction_a": 0.6,
            "direction_b": 0.6,
            "crossover": 0.6,
            "correct": 0.8,
            "resolved": 1.0,
        },
        {
            "p_a_given_a": 0.6,
            "p_a_given_b": 0.4,
            "p_b_given_a": 0.4,
            "p_b_given_b": 0.6,
            "direction_a": 0.2,
            "direction_b": 0.2,
            "crossover": 0.2,
            "correct": 0.6,
            "resolved": 1.0,
        },
    ]
    summary = summarize_assignments(
        rows, resamples=200, alpha=0.005, seed_parts=("fixture",)
    )
    assert summary["n_pairs"] == 2
    assert summary["mean"] == 0.4
    assert summary["direction_a"] == 0.4


def test_paired_advantage_is_deterministic() -> None:
    first = paired_advantage(
        [0.5, 0.6, 0.7],
        [0.1, 0.2, 0.3],
        resamples=500,
        alpha=0.005,
        seed_parts=("advantage",),
    )
    second = paired_advantage(
        [0.5, 0.6, 0.7],
        [0.1, 0.2, 0.3],
        resamples=500,
        alpha=0.005,
        seed_parts=("advantage",),
    )
    assert first == second
    assert abs(first["mean"] - 0.4) < 1e-15
