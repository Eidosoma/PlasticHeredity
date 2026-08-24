from __future__ import annotations

import numpy as np
import pandas as pd

from plastic_heredity.intervention_cr7_steering import (
    ARMS,
    BOOTSTRAP_REPETITIONS,
    EXTENSION_ARMS,
    EXTENSION_HORIZON,
    HORIZON,
    INHERITANCE_THRESHOLD,
    LANDMARK,
    MATRICES,
    RANDOMIZATION_REPETITIONS,
    RANDOM_EQUIVALENCE_MARGIN,
    REPLICATES,
    SEEDS,
    _action_seed,
    _artificial_case,
    _future_seed,
    compute_inference,
    count_nonoverlapping_episodes,
    inference_draws,
    is_out_of_envelope,
    longest_inherited_run,
    protocol,
)


def test_cr7_design_and_gate_are_frozen_exactly() -> None:
    frozen = protocol()
    assert MATRICES == 48
    assert LANDMARK == 60
    assert REPLICATES == 6
    assert HORIZON == 60
    assert ARMS == (
        "MODEL_UP",
        "MODEL_DOWN",
        "RULE_UP",
        "RULE_DOWN",
        "RANDOM",
        "NOOP",
    )
    assert BOOTSTRAP_REPETITIONS == 4_096
    assert RANDOMIZATION_REPETITIONS == 4_096
    assert RANDOM_EQUIVALENCE_MARGIN == 0.025
    assert frozen["target"]["strict_eight_excluded"] is True
    assert frozen["upstream"]["cr6_complete_gate"] is False
    assert frozen["upstream"]["cr6_not_used_to_tune_or_authorize_cr7"] is True
    assert frozen["cohort"]["complete_replay"] is True


def test_cr7_episode_counter_requires_break_then_three_and_new_break() -> None:
    assert INHERITANCE_THRESHOLD == 0.9
    assert count_nonoverlapping_episodes([0.91, 0.92, 0.93, 0.94]) == 0
    assert count_nonoverlapping_episodes([0.9, 0.91, 0.92]) == 0
    assert count_nonoverlapping_episodes([0.9, 0.91, 0.92, 0.93]) == 1
    assert count_nonoverlapping_episodes([0.8, 0.91, 0.7, 0.92, 0.93, 0.94]) == 1
    assert count_nonoverlapping_episodes(
        [0.8, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]
    ) == 1
    assert count_nonoverlapping_episodes(
        [0.8, 0.91, 0.92, 0.93, 0.7, 0.94, 0.95, 0.96]
    ) == 2


def test_cr7_longest_run_uses_strict_threshold() -> None:
    assert longest_inherited_run([0.91, 0.92, 0.8, 0.93, 0.94, 0.95]) == 3
    assert longest_inherited_run([0.9, 0.9]) == 0


def test_cr7_future_stream_is_arm_free_and_action_stream_is_separate() -> None:
    case, _ = _artificial_case()
    assert len(SEEDS) == len(set(SEEDS.values()))
    assert len({_future_seed(case, 4) for _arm in ARMS}) == 1
    assert _future_seed(case, 4) != _action_seed(case, 4)


def test_cr7_development_envelope_is_inclusive() -> None:
    envelope = {
        "c02__minimum": np.full(21, -1.0),
        "c02__maximum": np.full(21, 1.0),
    }
    assert not is_out_of_envelope(np.zeros(21), "02", envelope)
    assert not is_out_of_envelope(np.ones(21), "02", envelope)
    outside = np.zeros(21)
    outside[7] = 1.0 + 1e-12
    assert is_out_of_envelope(outside, "02", envelope)


def _fixture_matrix_table() -> pd.DataFrame:
    rows = []
    for candidate in ("02", "03"):
        for matrix_id in range(MATRICES):
            jitter = (1 if matrix_id % 2 else -1) * 0.002
            inherited = {
                "MODEL_UP": 0.60 + jitter,
                "MODEL_DOWN": 0.95 + jitter,
                "RULE_UP": 0.66 + jitter,
                "RULE_DOWN": 0.93 + jitter,
                "RANDOM": 0.80 + jitter,
                "NOOP": 0.80 + jitter,
            }
            episodes = {
                "MODEL_UP": 3.0,
                "MODEL_DOWN": 1.0,
                "RULE_UP": 2.5,
                "RULE_DOWN": 1.2,
                "RANDOM": 2.0,
                "NOOP": 2.0,
            }
            for arm in ARMS:
                rows.append(
                    {
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "controller": arm,
                        "inherited_fraction": inherited[arm],
                        "total_breaks": 60 * (1 - inherited[arm]),
                        "episode_count": episodes[arm],
                        "longest_inherited_run": 10.0,
                        "completed_horizon": 1.0,
                        "final_entropy": 2.0,
                        "final_occupied_types": 12.0,
                        "final_top1_share": 0.5,
                        "final_throughput": 1.0,
                        "mean_growth_updates": 100.0,
                        "cross_lineage_final_cosine": 0.9,
                        "distinct_swaps": 40.0 if arm != "NOOP" else 0.0,
                        "repeated_swaps": 20.0 if arm != "NOOP" else 0.0,
                        "immediately_reversing_swaps": 1.0 if arm != "NOOP" else 0.0,
                        "out_of_development_envelope_fraction": 0.1,
                        "final_risk": 0.5,
                        "mean_predicted_action_shift": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_cr7_whole_matrix_fixture_passes_all_primary_gates() -> None:
    metrics, stored = compute_inference(
        _fixture_matrix_table(),
        inference_draws(),
        replay_exact=True,
        noop_plain_exact=True,
    )
    assert metrics["complete_cr7_60_fission_gate"] is True
    assert metrics["conditional_extension_authorized"] is True
    assert all(item["candidate_primary_gate"] for item in metrics["candidates"])
    assert all(
        item["rule_recovery_fraction"]["strong_external_replication"]
        for item in metrics["candidates"]
    )
    assert stored["bootstrap_indices"].shape == (4096, 48)
    assert stored["randomization_signs"].shape == (4096, 48)


def test_cr7_integrity_failure_blocks_extension() -> None:
    metrics, _ = compute_inference(
        _fixture_matrix_table(),
        inference_draws(),
        replay_exact=False,
        noop_plain_exact=True,
    )
    assert metrics["all_candidate_primary_gates"] is True
    assert metrics["complete_cr7_60_fission_gate"] is False
    assert metrics["conditional_extension_authorized"] is False


def test_cr7_extension_is_frozen_as_continued_active_control() -> None:
    frozen = protocol()["conditional_extension"]
    assert EXTENSION_ARMS == ("MODEL_DOWN", "RULE_DOWN", "NOOP")
    assert EXTENSION_HORIZON == 60
    assert frozen["active_feedback_not_passive_persistence"] is True
    assert frozen["launch_only_if_all_primary_and_integrity_gates_pass"] is True

