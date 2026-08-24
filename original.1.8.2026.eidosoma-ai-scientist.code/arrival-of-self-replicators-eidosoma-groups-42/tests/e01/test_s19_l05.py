import numpy as np

from e01_s19_past_only_recurrence.core import (
    H_THRESHOLD,
    LABEL_DEFINITIONS,
    STRUCTURAL_LABEL_ID,
    past_only_recurrence,
    past_only_recurrence_reference,
    recomputed_generation_block_metrics,
    suffix_endpoint_indices,
)


def state(a: int, b: int, c: int = 0) -> np.ndarray:
    value = np.zeros(100, dtype=np.int64)
    value[:3] = (a, b, c)
    return value


def test_registry_is_singleton_and_threshold_is_fixed() -> None:
    assert H_THRESHOLD == 0.9
    assert len(LABEL_DEFINITIONS) == 2
    assert sum(not item.comparator_only for item in LABEL_DEFINITIONS) == 1
    assert LABEL_DEFINITIONS[1].label_id == STRUCTURAL_LABEL_ID


def test_rule_uses_only_nonadjacent_earlier_generations_without_backfill() -> None:
    states = np.stack(
        [
            state(0, 0, 10),
            state(10, 0),
            state(0, 10),
            state(0, 10),
            state(5, 5),
            state(10, 0),
            state(10, 0),
        ]
    )
    generations = np.asarray([0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    indices = np.arange(len(states), dtype=np.int64)
    result = past_only_recurrence(states, generations, indices)
    assert result["labels"].tolist() == [False, False, False, False, False, True, True]
    assert result["immediateOnlyEvidence"].tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert result["earliestMatchingSequenceIndex"][5] == 1
    assert result["latestMatchingSequenceIndex"][5] == 1
    assert not result["labels"][1]


def test_each_earlier_generation_counts_once_and_reference_replay_is_exact() -> None:
    states = np.stack(
        [
            state(0, 0, 10),
            state(10, 0),
            state(10, 0),
            state(5, 5),
            state(10, 0),
            state(10, 0),
            state(10, 0),
        ]
    )
    generations = np.asarray([0, 1, 1, 2, 2, 2, 3], dtype=np.int64)
    indices = np.arange(len(states), dtype=np.int64)
    result = past_only_recurrence(states, generations, indices)
    reference = past_only_recurrence_reference(states, generations, indices)
    for key in reference:
        if key == "matchingReferenceIndices":
            assert result[key] == reference[key]
        else:
            assert np.array_equal(result[key], reference[key], equal_nan=True)
    assert result["qualifyingPriorStateCount"][6] == 3
    assert result["distinctEarlierGenerationCount"][6] == 2


def test_suffix_endpoints_and_query_stop_ignore_future_replacement() -> None:
    states = np.stack([state(0, 0, 10)] + [state(10, i % 2) for i in range(1, 21)])
    generations = np.asarray(
        [0] + [1 + (i - 1) // 4 for i in range(1, 21)], dtype=np.int64
    )
    indices = np.arange(len(states), dtype=np.int64)
    endpoints = suffix_endpoint_indices(len(states))
    assert len(endpoints) == 5
    endpoint = endpoints[2]
    baseline = past_only_recurrence(states, generations, indices, query_stop=endpoint)
    replacement = states.copy()
    replacement[endpoint + 1 :] = replacement[endpoint + 1 :, ::-1]
    changed = past_only_recurrence(
        replacement, generations, indices, query_stop=endpoint
    )
    for key in (
        "labels",
        "scores",
        "distinctEarlierGenerationCount",
        "qualifyingPriorStateCount",
        "earliestMatchingSequenceIndex",
        "latestMatchingSequenceIndex",
    ):
        assert np.array_equal(baseline[key], changed[key], equal_nan=True)


def test_recomputed_permutation_matches_naive_reordered_trajectories() -> None:
    states = np.stack(
        [
            state(0, 0, 10),
            state(10, 0),
            state(5, 5),
            state(10, 0),
            state(0, 10),
            state(10, 0),
            state(5, 5),
        ]
    )
    generations = np.asarray([0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    orders = np.asarray([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=np.int64)
    fast = recomputed_generation_block_metrics(states, generations, orders)
    for replicate, order in enumerate(orders):
        pieces = [states[:1]]
        new_generations = [np.asarray([0], dtype=np.int64)]
        for new_generation, old_ordinal in enumerate(order, start=1):
            block = states[generations == old_ordinal + 1]
            pieces.append(block)
            new_generations.append(np.full(len(block), new_generation, dtype=np.int64))
        reordered = np.concatenate(pieces)
        reordered_generations = np.concatenate(new_generations)
        result = past_only_recurrence(
            reordered,
            reordered_generations,
            np.arange(len(reordered), dtype=np.int64),
        )
        labels = result["labels"][1:]
        persistence = float(labels.sum())
        occupancy = persistence / len(labels)
        if np.ptp(labels.astype(float)):
            consistency = float(np.corrcoef(labels[:-1], labels[1:])[0, 1])
        else:
            consistency = np.nan
        onset = (
            float(np.flatnonzero(result["labels"])[0])
            if labels.any()
            else float(len(reordered))
        )
        normalized = onset / len(labels) if onset < len(reordered) else 1.0
        assert fast["persistence"][replicate] == persistence
        assert fast["occupancy"][replicate] == occupancy
        assert np.isclose(fast["consistency"][replicate], consistency, equal_nan=True)
        assert fast["firstOnsetRawScore"][replicate] == onset
        assert fast["firstOnsetNormalizedScore"][replicate] == normalized
