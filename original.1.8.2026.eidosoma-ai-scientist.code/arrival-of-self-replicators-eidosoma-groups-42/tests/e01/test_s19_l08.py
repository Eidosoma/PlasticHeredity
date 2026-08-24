from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from e01_s19_untouched_mechanism.core import (
    BOOTSTRAP_REPLICATES,
    MECHANISM_A,
    MECHANISM_B,
    OBJECT_A_BOUNDARY,
    OBJECT_A_PROJECTED,
    OBJECT_B_MOLECULAR,
    bootstrap_indices,
    label_fingerprint,
    mechanism_setting,
    occupancy_in_band,
    paper_distance,
    terminal_decision,
)
from scripts.e01.run_s19_l08 import simulation_specs


def test_exactly_two_mechanisms_and_four_simulations_are_locked() -> None:
    specs = simulation_specs()
    assert len(specs) == 4
    assert {row["mechanismId"] for row in specs} == {MECHANISM_A, MECHANISM_B}
    assert {(row["mechanismId"], row["candidateId"]) for row in specs} == {
        (MECHANISM_A, "CANDIDATE_2"),
        (MECHANISM_A, "CANDIDATE_3"),
        (MECHANISM_B, "CANDIDATE_2"),
        (MECHANISM_B, "CANDIDATE_3"),
    }
    exposures = {(row["mechanismId"], row["candidateId"]): row["exposure"] for row in specs}
    assert exposures[(MECHANISM_A, "CANDIDATE_2")] == 0.6031526490073492
    assert exposures[(MECHANISM_A, "CANDIDATE_3")] == 0.5613315384859516
    assert exposures[(MECHANISM_B, "CANDIDATE_2")] == 2.875
    assert exposures[(MECHANISM_B, "CANDIDATE_3")] == 2.875


def test_only_three_registered_analysis_objects_exist() -> None:
    boundary = mechanism_setting(MECHANISM_A, OBJECT_A_BOUNDARY)
    projected = mechanism_setting(MECHANISM_A, OBJECT_A_PROJECTED)
    molecular = mechanism_setting(MECHANISM_B, OBJECT_B_MOLECULAR)
    assert boundary["threshold"] == projected["threshold"] == molecular["threshold"] == 0.9
    assert boundary["comparator"] == projected["comparator"] == molecular["comparator"] == "STRICT_GT"
    assert boundary["projection"] == "BOUNDARY_ONLY"
    assert projected["projection"] == "OUTGOING_INTERVAL_PREFIX_INELIGIBLE"
    assert molecular["projection"] == "ALL_OBSERVATIONS"
    with np.testing.assert_raises(ValueError):
        mechanism_setting(MECHANISM_B, OBJECT_A_BOUNDARY)


def test_label_fingerprint_keeps_positive_and_negative_episode_topology() -> None:
    frame = pd.DataFrame(
        {
            "analysisUnitIndex": np.arange(8),
            "rawObservationIndex": np.arange(8),
            "labelStatus": ["INELIGIBLE_LOCKED_PREFIX"] + ["ELIGIBLE"] * 7,
            "isReplicator": [None, False, True, True, False, False, True, False],
        }
    )
    result = label_fingerprint(frame)
    assert result["fingerprintStatus"] == "ELIGIBLE"
    assert result["analysisUnitLength"] == 8
    assert result["eligibleLength"] == 7
    assert result["persistence"] == 3
    assert result["negativePersistence"] == 4
    assert result["occupancy"] == 3 / 7
    assert result["firstOnsetRawIndex0"] == 2
    assert result["firstOnsetRawStep1"] == 3
    assert result["firstOnsetNormalized"] == 2 / 7
    assert result["positiveEpisodeCount"] == 2
    assert result["positiveEpisodeDurations"] == [2, 1]
    assert result["negativeEpisodeCount"] == 3
    assert result["negativeEpisodeDurations"] == [1, 2, 1]
    assert result["positiveEpisodeStartIndices"] == [1, 5]
    assert result["positiveMeanEpisodeSpacing"] == 4.0


def test_all_ineligible_series_is_status_bearing() -> None:
    frame = pd.DataFrame(
        {
            "analysisUnitIndex": [0, 1],
            "rawObservationIndex": [0, 1],
            "labelStatus": ["INELIGIBLE_LOCKED_PREFIX"] * 2,
            "isReplicator": [None, None],
        }
    )
    result = label_fingerprint(frame)
    assert result["fingerprintStatus"] == "INELIGIBLE_NO_LOCKED_ANALYSIS_UNITS"
    assert result["eligibleLength"] == 0
    assert result["labelSha256"] is None


def test_bootstrap_is_exactly_4096_and_domain_separated() -> None:
    root = "39" * 32
    first = bootstrap_indices(root, "A", "C2", "occupancy")
    replay = bootstrap_indices(root, "A", "C2", "occupancy")
    other = bootstrap_indices(root, "A", "C3", "occupancy")
    assert first.shape == (BOOTSTRAP_REPLICATES, 100)
    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, other)
    with np.testing.assert_raises(ValueError):
        bootstrap_indices(root, "bad", replicates=4095)


def test_paper_distance_preserves_onset_ambiguity() -> None:
    exact = {
        "selectedClockLength": 716 / 0.88,
        "persistence": 716.0,
        "occupancy": 0.88,
        "consistency": 0.38,
        "firstOnsetRawStep1": 37.0,
        "firstOnsetNormalized": 0.37,
    }
    assert paper_distance(exact, "RAW_ONSET") == 0.0
    assert paper_distance(exact, "NORMALIZED_ONSET") == 0.0
    shifted = dict(exact, firstOnsetRawStep1=64.0)
    assert paper_distance(shifted, "RAW_ONSET") > 0
    assert paper_distance(shifted, "NORMALIZED_ONSET") == 0.0


def test_occupancy_band_is_inclusive() -> None:
    assert occupancy_in_band(0.85)
    assert occupancy_in_band(0.91)
    assert not occupancy_in_band(np.nextafter(0.85, 0.0))
    assert not occupancy_in_band(np.nextafter(0.91, 1.0))


def test_terminal_resolution_order_is_fail_closed_and_non_favorable() -> None:
    assert terminal_decision(
        operational_integrity_passed=False,
        joint_occupancy_gate_passed=True,
        fission_preference_gates_passed=True,
        exposure_preference_gates_passed=False,
    ) == "LOOP_FAILED_CLOSED"
    assert terminal_decision(
        operational_integrity_passed=True,
        joint_occupancy_gate_passed=False,
        fission_preference_gates_passed=True,
        exposure_preference_gates_passed=False,
    ) == "NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA"
    assert terminal_decision(
        operational_integrity_passed=True,
        joint_occupancy_gate_passed=True,
        fission_preference_gates_passed=True,
        exposure_preference_gates_passed=False,
    ) == "EVIDENCE_FAVORS_FISSION_BOUNDARY_MECHANISM"
    assert terminal_decision(
        operational_integrity_passed=True,
        joint_occupancy_gate_passed=True,
        fission_preference_gates_passed=False,
        exposure_preference_gates_passed=True,
    ) == "EVIDENCE_FAVORS_HIGH_EXPOSURE_MECHANISM"
    assert terminal_decision(
        operational_integrity_passed=True,
        joint_occupancy_gate_passed=True,
        fission_preference_gates_passed=True,
        exposure_preference_gates_passed=True,
    ) == "NONIDENTIFIABLE_BETWEEN_FROZEN_MECHANISMS"
