from types import SimpleNamespace

import numpy as np

from e01_s19_boundary_compotype.core import (
    H_THRESHOLD,
    LABEL_DEFINITIONS,
    activate_membership,
    boundary_modal_reference,
    project_boundary_state,
)


def state(a: int, b: int) -> np.ndarray:
    value = np.zeros(100, dtype=np.int64)
    value[0] = a
    value[1] = b
    return value


def observation(index: int, kind: str, generation: int):
    return SimpleNamespace(
        observation_index=index,
        observation_kind=kind,
        growth_generation_one_based=generation,
        state=tuple(state(1, 1)),
    )


def test_registry_is_small_fixed_and_nonfactorial() -> None:
    assert len(LABEL_DEFINITIONS) == 5
    assert sum(not item.comparator_only for item in LABEL_DEFINITIONS) == 4
    assert H_THRESHOLD == 0.9
    assert len({item.label_id for item in LABEL_DEFINITIONS}) == 5


def test_modal_reference_uses_strict_threshold_and_earliest_tie() -> None:
    states = np.stack([state(10, 0), state(9, 1), state(0, 10), state(1, 9)])
    result = boundary_modal_reference(states, [1, 2, 3, 4])
    assert result["referenceIndex"] == 0
    assert result["referenceGeneration"] == 1
    assert result["referenceFrequency"] == 2
    assert result["members"].tolist() == [True, True, False, False]


def test_activation_does_not_backfill_first_recurrence() -> None:
    members = np.asarray([False, True, False, True, True], dtype=bool)
    backfill = activate_membership(members, "BACKFILL_ALL_OCCURRENCES", True)
    activated = activate_membership(members, "SECOND_OCCURRENCE_ONWARD", True)
    assert backfill.tolist() == [False, True, False, True, True]
    assert activated.tolist() == [False, False, False, True, True]
    assert not activate_membership(members, "BACKFILL_ALL_OCCURRENCES", False).any()


def test_incoming_projection_uses_same_generation_boundary() -> None:
    selected = [
        observation(0, "initial_selected_state", 0),
        observation(1, "molecular_update", 1),
        observation(2, "post_fission", 1),
        observation(3, "molecular_update", 2),
        observation(4, "post_fission", 2),
    ]
    labels, _, _, sources = project_boundary_state(
        selected,
        [1, 2],
        np.asarray([False, True]),
        np.asarray([0.91, 0.99]),
        np.asarray([True, True]),
        "INCOMING_BOUNDARY_ENDING",
    )
    assert labels == [None, False, False, True, True]
    assert sources == [None, 1, 1, 2, 2]


def test_outgoing_projection_shifts_growth_but_not_boundary_row() -> None:
    selected = [
        observation(0, "initial_selected_state", 0),
        observation(1, "molecular_update", 1),
        observation(2, "post_fission", 1),
        observation(3, "molecular_update", 2),
        observation(4, "post_fission", 2),
    ]
    labels, _, _, sources = project_boundary_state(
        selected,
        [1, 2],
        np.asarray([False, True]),
        np.asarray([0.91, 0.99]),
        np.asarray([True, True]),
        "OUTGOING_BOUNDARY_STARTING",
    )
    assert labels == [None, None, False, False, True]
    assert sources == [None, None, 1, 1, 2]
