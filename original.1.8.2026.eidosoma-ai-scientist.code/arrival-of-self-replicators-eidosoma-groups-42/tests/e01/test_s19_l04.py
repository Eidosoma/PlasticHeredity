import numpy as np

from e01_s19_cross_generation_recurrence.core import (
    H_THRESHOLD,
    LABEL_DEFINITIONS,
    STRUCTURAL_LABEL_ID,
    binary_consistency_from_counts,
    cross_generation_recurrence,
)


def state(a: int, b: int, c: int = 0) -> np.ndarray:
    value = np.zeros(100, dtype=np.int64)
    value[:3] = (a, b, c)
    return value


def test_registry_has_exactly_one_structural_label_and_one_comparator() -> None:
    assert H_THRESHOLD == 0.9
    assert len(LABEL_DEFINITIONS) == 2
    assert sum(not item.comparator_only for item in LABEL_DEFINITIONS) == 1
    assert LABEL_DEFINITIONS[1].label_id == STRUCTURAL_LABEL_ID


def test_cross_generation_rule_excludes_same_generation_and_immediate_only() -> None:
    states = np.stack(
        [
            state(0, 0, 10),  # initial, ineligible
            state(10, 0),  # generation 1 A; recurs nonadjacently in generation 3
            state(0, 10),  # generation 1 B; only immediate cross-gen recurrence
            state(0, 10),  # generation 2 B
            state(5, 5),  # generation 2 C
            state(10, 0),  # generation 3 A
        ]
    )
    generations = np.asarray([0, 1, 1, 2, 2, 3], dtype=np.int64)
    indices = np.arange(len(states), dtype=np.int64)
    result = cross_generation_recurrence(states, generations, indices)
    assert result["labels"].tolist() == [False, True, False, False, False, True]
    assert result["immediateOnlyEvidence"].tolist() == [False, False, True, True, False, False]
    assert result["distinctOtherGenerationCount"].tolist() == [0, 1, 0, 0, 0, 1]
    assert result["diagnostics"]["exactSymmetryPassed"]


def test_each_reference_generation_counts_once() -> None:
    states = np.stack(
        [
            state(0, 0, 10),
            state(10, 0),
            state(5, 5),
            state(10, 0),
            state(10, 0),
            state(5, 5),
            state(10, 0),
        ]
    )
    generations = np.asarray([0, 1, 1, 2, 2, 2, 3], dtype=np.int64)
    result = cross_generation_recurrence(
        states, generations, np.arange(len(states), dtype=np.int64)
    )
    # The generation-1 A row matches two states in generation 2 and one in 3,
    # but the distinct-visit count is two generations rather than three rows.
    assert result["qualifyingStateCount"][1] == 3
    assert result["distinctOtherGenerationCount"][1] == 2
    assert result["firstMatchingGeneration"][1] == 2
    assert result["lastMatchingGeneration"][1] == 3


def test_strict_similarity_and_generation_zero_are_enforced() -> None:
    states = np.stack(
        [state(10, 0), state(10, 0), state(0, 10), state(10, 0)]
    )
    generations = np.asarray([0, 1, 2, 3], dtype=np.int64)
    result = cross_generation_recurrence(
        states, generations, np.arange(len(states), dtype=np.int64)
    )
    assert not result["labels"][0]
    assert result["labels"][1]
    assert not result["labels"][2]
    assert result["labels"][3]


def test_binary_consistency_from_counts_matches_numpy() -> None:
    labels = np.asarray([False, False, True, True, False, True, True], dtype=bool)
    n00 = int(np.sum(~labels[:-1] & ~labels[1:]))
    n01 = int(np.sum(~labels[:-1] & labels[1:]))
    n10 = int(np.sum(labels[:-1] & ~labels[1:]))
    n11 = int(np.sum(labels[:-1] & labels[1:]))
    observed = binary_consistency_from_counts(n00, n01, n10, n11)
    expected = float(np.corrcoef(labels[:-1].astype(float), labels[1:].astype(float))[0, 1])
    assert observed is not None
    assert np.isclose(observed, expected, atol=1e-15, rtol=0)
