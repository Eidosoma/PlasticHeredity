import numpy as np

from e01_s19_boundary_recurrence.core import (
    H_THRESHOLD,
    LABEL_DEFINITIONS,
    STRUCTURAL_LABEL_ID,
    boundary_recurrence,
    boundary_recurrence_reference,
    recomputed_generation_block_metrics,
    suffix_endpoint_indices,
)


def state(a: int, b: int, c: int = 0) -> np.ndarray:
    value = np.zeros(100, dtype=np.int64)
    value[:3] = (a, b, c)
    return value


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = np.stack(
        [
            state(0, 0, 10),
            state(10, 0),
            state(10, 0),
            state(0, 10),
            state(0, 10),
            state(10, 0),
            state(10, 0),
            state(10, 0),
            state(10, 0),
        ]
    )
    generations = np.asarray([0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    kinds = np.asarray(
        [
            "initial_selected_state",
            "molecular_update", "post_fission",
            "molecular_update", "post_fission",
            "molecular_update", "post_fission",
            "molecular_update", "post_fission",
        ]
    )
    return states, generations, kinds, np.arange(len(states), dtype=np.int64)


def test_registry_is_singleton_and_threshold_is_fixed() -> None:
    assert H_THRESHOLD == 0.9
    assert len(LABEL_DEFINITIONS) == 2
    assert sum(not item.comparator_only for item in LABEL_DEFINITIONS) == 1
    assert LABEL_DEFINITIONS[1].label_id == STRUCTURAL_LABEL_ID


def test_boundary_rule_excludes_previous_generation_and_projects_forward() -> None:
    states, generations, kinds, indices = fixture()
    result = boundary_recurrence(states, generations, kinds, indices)
    # b3 may match b1, but b2 cannot use b1 because h<=g-2 is empty for g=2.
    assert result["boundaryLabels"].tolist() == [False, False, True, True]
    # Pre-b1 growth is eligible-negative; b3 and the following generation-4
    # molecular interval are positive until b4 independently reactivates.
    assert result["labels"].tolist() == [False, False, False, False, False, False, True, True, True]
    assert result["sourceBoundaryGeneration"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4]
    assert result["boundaryMatchingGenerations"][2] == (1,)


def test_independent_boundary_and_projection_replay_is_exact() -> None:
    states, generations, kinds, indices = fixture()
    primary = boundary_recurrence(states, generations, kinds, indices)
    replay = boundary_recurrence_reference(states, generations, kinds, indices)
    for key in replay:
        if key == "matchingBoundaryGenerations":
            assert primary[key] == replay[key]
        else:
            assert np.array_equal(primary[key], replay[key], equal_nan=True)


def test_prefix_query_is_future_suffix_invariant() -> None:
    states, generations, kinds, indices = fixture()
    endpoint = 6
    baseline = boundary_recurrence(
        states, generations, kinds, indices, query_stop=endpoint
    )
    changed = states.copy()
    changed[endpoint + 1 :] = changed[endpoint + 1 :, ::-1]
    replay = boundary_recurrence(
        changed, generations, kinds, indices, query_stop=endpoint
    )
    for key in (
        "labels", "scores", "distinctPriorBoundaryCount",
        "qualifyingPriorBoundaryCount", "firstMatchingBoundaryGeneration",
        "lastMatchingBoundaryGeneration", "sourceBoundaryGeneration",
    ):
        assert np.array_equal(baseline[key], replay[key], equal_nan=True)
    assert baseline["matchingBoundaryGenerations"] == replay["matchingBoundaryGenerations"]
    assert len(suffix_endpoint_indices(21)) == 5


def test_recomputed_permutation_matches_naive_reordered_trajectories() -> None:
    states, generations, kinds, _ = fixture()
    orders = np.asarray([[0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0]], dtype=np.int64)
    fast = recomputed_generation_block_metrics(states, generations, kinds, orders)
    for replicate, order in enumerate(orders):
        pieces = [states[:1]]
        generation_pieces = [np.asarray([0], dtype=np.int64)]
        kind_pieces = [kinds[:1]]
        for new_generation, old_ordinal in enumerate(order, start=1):
            mask = generations == old_ordinal + 1
            pieces.append(states[mask])
            generation_pieces.append(np.full(mask.sum(), new_generation, dtype=np.int64))
            kind_pieces.append(kinds[mask])
        reordered = np.concatenate(pieces)
        reordered_generations = np.concatenate(generation_pieces)
        reordered_kinds = np.concatenate(kind_pieces)
        result = boundary_recurrence(
            reordered,
            reordered_generations,
            reordered_kinds,
            np.arange(len(reordered), dtype=np.int64),
        )
        labels = result["labels"][1:]
        persistence = float(labels.sum())
        occupancy = persistence / len(labels)
        left = labels[:-1].astype(float)
        right = labels[1:].astype(float)
        denominator = left.std() * right.std()
        consistency = (
            float(np.mean((left - left.mean()) * (right - right.mean())) / denominator)
            if denominator > 0 else np.nan
        )
        onset = float(np.flatnonzero(result["labels"])[0]) if labels.any() else float(len(reordered))
        normalized = onset / len(labels) if labels.any() else 1.0
        assert fast["persistence"][replicate] == persistence
        assert fast["occupancy"][replicate] == occupancy
        assert np.isclose(fast["consistency"][replicate], consistency, equal_nan=True)
        assert fast["firstOnsetRawScore"][replicate] == onset
        assert fast["firstOnsetNormalizedScore"][replicate] == normalized
