from __future__ import annotations

import numpy as np

from e01_s19_replicator_definition.core import (
    LABEL_DEFINITIONS,
    closer_dimension_count,
    fingerprint_from_labels,
    paper_fingerprint_distance,
)


def test_registry_has_exactly_four_unique_fixed_families() -> None:
    assert len(LABEL_DEFINITIONS) == 4
    assert [item.ordinal for item in LABEL_DEFINITIONS] == [1, 2, 3, 4]
    assert len({item.label_id for item in LABEL_DEFINITIONS}) == 4
    assert sum(item.comparator_only for item in LABEL_DEFINITIONS) == 1
    assert sum(item.source_grounding_gate for item in LABEL_DEFINITIONS) == 1


def test_temporal_fingerprint_counts_episodes_and_cutoff() -> None:
    labels = [False, False, True, True, False, True, True, True]
    result = fingerprint_from_labels(
        sequence_indices=range(len(labels)),
        labels=labels,
        total_clock_count=len(labels),
        observation_kinds=["initial_selected_state", *(["molecular_update"] * 6), "post_fission"],
        global_reference=True,
    )
    assert result["persistence"] == 5
    assert result["episodeCount"] == 2
    assert result["entryCount"] == 2
    assert result["exitCount"] == 1
    assert result["firstOnsetRawIndex0"] == 2
    assert np.isclose(result["firstOnsetNormalized"], 2 / 7)
    assert result["isNonreplicatingAtCutoff"] is True
    assert result["noReplicatorObservedThroughCutoff"] is True
    assert result["longestEpisode"] == 3


def test_historical_initial_missing_is_retained_not_dropped_from_clock() -> None:
    result = fingerprint_from_labels(
        sequence_indices=range(5),
        labels=[None, False, True, True, False],
        total_clock_count=5,
        observation_kinds=["initial_selected_state", "molecular_update", "molecular_update", "molecular_update", "post_fission"],
        global_reference=False,
    )
    assert result["eligibleCount"] == 4
    assert result["ineligibleCount"] == 1
    assert result["firstOnsetRawIndex0"] == 2
    assert result["persistence"] == 2


def test_paper_distance_requires_consistency_and_keeps_onset_modes_separate() -> None:
    exact = {
        "persistence": 716.0,
        "occupancy": 0.88,
        "consistency": 0.38,
        "firstOnsetRawScore": 37.0,
        "firstOnsetNormalizedScore": 0.37,
    }
    assert paper_fingerprint_distance(exact, onset_mode="RAW") == 0.0
    assert paper_fingerprint_distance(exact, onset_mode="NORMALIZED") == 0.0
    missing = dict(exact, consistency=None)
    assert paper_fingerprint_distance(missing, onset_mode="RAW") is None


def test_occupancy_only_cannot_satisfy_structure_improvement() -> None:
    comparator = {
        "persistence": 800.0,
        "occupancy": 0.98,
        "consistency": 0.1,
        "firstOnsetRawScore": 3.0,
        "firstOnsetNormalizedScore": 0.003,
    }
    occupancy_only = dict(comparator, occupancy=0.88)
    count, structure = closer_dimension_count(occupancy_only, comparator, onset_mode="RAW")
    assert count == 1
    assert structure is False
