from __future__ import annotations

import json

import numpy as np
import pandas as pd

from e01_descriptive_causal_emergence.core import (
    COMPLETED_MODE,
    PREFIX_MODE,
    add_excursion_flags,
    aggregate_trajectories,
    compare_completed_prefix,
    excursion_catalog,
    lag_rule,
    partition_change_history,
    prepare_completed,
    prepare_prefix,
    trend_results,
)


def _completed(values_by_run: list[list[float]]) -> pd.DataFrame:
    rows = []
    for matrix_index, values in enumerate(values_by_run):
        for index, value in enumerate(values, start=1):
            rows.append(
                {
                    "candidateId": "S12F-CANDIDATE-02",
                    "trajectoryId": f"T{matrix_index}",
                    "matrixIndex": matrix_index,
                    "temporalLabel": COMPLETED_MODE,
                    "selectedSequenceIndex": index,
                    "rawObservationIndex": index,
                    "observationKind": "post_fission"
                    if index == 3
                    else "molecular_update",
                    "generation": 1,
                    "molecularStep": index,
                    "status": "ELIGIBLE",
                    "emergence": value,
                }
            )
    return prepare_completed(pd.DataFrame(rows))


def _prefix(values_by_run: list[list[float]]) -> pd.DataFrame:
    rows = []
    for matrix_index, values in enumerate(values_by_run):
        for generation, value in enumerate(values, start=1):
            rows.append(
                {
                    "candidateId": "S12F-CANDIDATE-02",
                    "trajectoryId": f"T{matrix_index}",
                    "matrixIndex": matrix_index,
                    "temporalLabel": PREFIX_MODE,
                    "generation": generation,
                    "endpointSelectedSequenceIndex": generation,
                    "endpointRawObservationIndex": generation,
                    "endpointObservationKind": "post_fission",
                    "status": "ELIGIBLE",
                    "emergence": value,
                }
            )
    return prepare_prefix(pd.DataFrame(rows))


def test_aggregate_views_preserve_unequal_length_support() -> None:
    frame = _completed([[1.0, 2.0, 3.0], [2.0, 4.0]])
    aggregate = aggregate_trajectories(frame, normalized_grid_points=5)
    candidate = aggregate[aggregate["candidateScope"].eq("S12F-CANDIDATE-02")]
    available = candidate[candidate["alignmentView"].eq("AVAILABLE_CASE")]
    common = candidate[candidate["alignmentView"].eq("FULL_COHORT_SUPPORT")]
    normalized = candidate[candidate["alignmentView"].eq("NORMALIZED_TIME_101")]
    assert available["contributingTrajectoryCount"].tolist() == [2, 2, 1]
    assert common["timeCoordinate"].tolist() == [1.0, 2.0]
    assert len(normalized) == 5
    assert normalized["contributingTrajectoryCount"].eq(2).all()
    trends = trend_results(aggregate)
    assert set(trends["alignmentView"]) == {
        "AVAILABLE_CASE",
        "FULL_COHORT_SUPPORT",
        "MAJORITY_SUPPORT",
        "NORMALIZED_TIME_101",
    }


def test_three_sigma_and_mad_episodes_are_signed_and_morphological() -> None:
    values = [0.0] * 100 + [20.0, 21.0] + [0.0] * 100 + [-20.0, -21.0]
    frame = _completed([values])
    flagged, thresholds = add_excursion_flags(frame)
    catalog, run_summary, dependency = excursion_catalog(flagged, thresholds)
    three = catalog[catalog["thresholdFamily"].eq("THREE_SIGMA")]
    assert set(three["sign"]) == {"POSITIVE", "NEGATIVE"}
    assert three["episodeWidthObservations"].tolist() == [2, 2]
    assert three["peakProminence"].ge(0).all()
    assert len(run_summary) == 4
    assert len(dependency) == 4


def test_lag_rule_is_exactly_inherited() -> None:
    assert lag_rule(5) == 1
    assert lag_rule(49) == 9
    assert lag_rule(50) == 10
    assert lag_rule(1000) == 10


def test_partition_changes_ignore_side_swaps_but_detect_membership_change() -> None:
    rows = []
    partitions = [([0, 1], [2, 3]), ([2, 3], [0, 1]), ([0, 2], [1, 3])]
    for generation, (left, right) in enumerate(partitions, start=1):
        rows.append(
            {
                "candidateId": "S12F-CANDIDATE-02",
                "trajectoryId": "T0",
                "matrixIndex": 0,
                "fitKind": "past_only_prefix_endpoint",
                "endpointGeneration": generation,
                "endpointSelectedSequenceIndex": generation,
                "status": "ELIGIBLE",
                "partition1Json": json.dumps(left),
                "partition2Json": json.dumps(right),
                "partitionSize1": 2,
                "partitionSize2": 2,
            }
        )
    history = partition_change_history(pd.DataFrame(rows))
    assert pd.isna(history.loc[0, "partitionChangedFromPreviousEligibleFit"])
    assert not bool(history.loc[1, "partitionChangedFromPreviousEligibleFit"])
    assert bool(history.loc[2, "partitionChangedFromPreviousEligibleFit"])
    assert history.loc[1, "partitionARIFromPreviousEligibleFit"] == 1.0


def test_completed_prefix_join_uses_exact_selected_endpoint_identity() -> None:
    completed, _ = add_excursion_flags(_completed([[0.0, 1.0, 2.0]]))
    prefix, _ = add_excursion_flags(_prefix([[0.0, 1.5, 2.5]]))
    partition_rows = []
    for generation in range(1, 4):
        partition_rows.append(
            {
                "candidateId": "S12F-CANDIDATE-02",
                "trajectoryId": "T0",
                "matrixIndex": 0,
                "fitKind": "past_only_prefix_endpoint",
                "endpointGeneration": generation,
                "endpointSelectedSequenceIndex": generation,
                "status": "ELIGIBLE",
                "partition1Json": json.dumps([0, 1]),
                "partition2Json": json.dumps([2, 3]),
                "partitionSize1": 2,
                "partitionSize2": 2,
            }
        )
    history = partition_change_history(pd.DataFrame(partition_rows))
    joined, summary = compare_completed_prefix(completed, prefix, history)
    assert len(joined) == 3
    np.testing.assert_allclose(
        joined["emergenceDifferenceCompletedMinusPastOnly"], [0.0, -0.5, -0.5]
    )
    assert summary.loc[0, "sharedEndpointCount"] == 3
