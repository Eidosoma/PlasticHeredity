from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from e01_onset_discovery.heredity_process_family import (
    crossfit_markov_gain_bits,
    summarize_binary_episode,
)
from scripts.e01.run_s19_l44_plastic_heredity_process_family import (
    process_control_relationships,
)


def test_episode_summary_and_certification() -> None:
    result = summarize_binary_episode([False, True, True, True, False, True])
    assert result.maximum_positive_run == 3
    assert result.run2_event and result.run3_event and not result.run5_event
    assert result.run2_certification_one_based == 3
    assert result.run3_certification_one_based == 4
    assert result.transition_01 == 2
    assert result.transition_10 == 1


def test_empty_and_singleton_sequences() -> None:
    empty = summarize_binary_episode([])
    singleton = summarize_binary_episode([True])
    assert empty.opportunities == 0
    assert not empty.run2_event
    assert singleton.positives == 1
    assert singleton.transition_11 == 0


def test_fixed_count_order_null_bounded() -> None:
    result = summarize_binary_episode([True, False, True, True, False])
    for value in (
        result.run2_order_null_probability,
        result.run3_order_null_probability,
        result.run5_order_null_probability,
    ):
        assert 0 <= value <= 1


def test_markov_crossfit_detects_sticky_sequences() -> None:
    a = [[False, False, False, True, True, True]] * 20
    b = [[False, False, False, True, True, True]] * 20
    result = crossfit_markov_gain_bits(a, b)
    assert result["transitions"] > 0
    assert result["markovGainBitsPerTransition"] > 0


def test_iid_sequences_do_not_create_large_markov_gain() -> None:
    sequences = [[False, True, False, True], [True, False, True, False]] * 20
    result = crossfit_markov_gain_bits(sequences[:20], sequences[20:])
    assert result["markovGainBitsPerTransition"] > 0


def test_exact_replay() -> None:
    sequence = [False, True, True, False, True]
    assert summarize_binary_episode(sequence) == summarize_binary_episode(sequence)


def test_prefix_control_merge_preserves_state_group_names() -> None:
    states = pd.DataFrame(
        [
            {
                "stateId": "state-1",
                "evaluationCohort": "L28_VALIDATION",
                "candidateId": "candidate-2",
                "processId": "INHERITANCE_BREAK_WITH_DEPARTURE",
                "eligible": True,
                "qHat": 0.5,
                "meanPostbreakOpportunities": 10.0,
                "meanInheritanceFraction": 0.8,
            }
        ]
    )
    prefix = pd.DataFrame(
        [
            {
                "stateId": "state-1",
                "evaluationCohort": "L28_VALIDATION",
                "candidateId": "candidate-2",
                "matrixIndex": 1,
                "landmark": 64,
                "prefixInheritanceFraction": 0.75,
                "prefixTrailingInheritanceRun": 2,
                "prefixMaximumInheritanceRun": 3,
                "prefixBoundaryCount": 8,
                "currentMass": 50,
                "currentGenerationLocalStep": 4,
            }
        ]
    )
    result = process_control_relationships(states, prefix)
    assert len(result) == 8
    assert result["evaluationCohort"].eq("L28_VALIDATION").all()
    assert result["candidateId"].eq("candidate-2").all()
