from __future__ import annotations

import numpy as np
import pytest

from core import (
    ARM_NAMES,
    aggregate_transitions,
    apply_intervention,
    bray_pair_decomposition,
    concentration_descriptors,
    transition_masks,
)
from run_analysis import (
    Snapshot,
    _assert_snapshot_preservation,
    _edited_snapshot,
    _intervention_power_row,
    _intervention_q,
)


def test_concentration_descriptors_are_interpretable() -> None:
    values = concentration_descriptors(np.asarray([8, 2, 0, 0]))
    assert values[2] == 2
    assert values[3] == pytest.approx(0.8)
    assert values[4] == pytest.approx(1.0)
    assert values[0] > values[1] > 1.0


def test_transition_masks_and_aggregation_follow_ordered_gates() -> None:
    gates = np.asarray([[0, 1, 2, 3, 4]])
    success, eligible = transition_masks(gates)
    assert success.sum(axis=1).tolist() == [[4, 3, 2, 1]]
    assert eligible.sum(axis=1).tolist() == [[5, 4, 3, 2]]
    successes, trials = aggregate_transitions(gates)
    assert successes.tolist() == [[4.0, 3.0, 2.0, 1.0]]
    assert trials.tolist() == [[5.0, 4.0, 3.0, 2.0]]


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_interventions_preserve_mass_are_deterministic_and_nested(arm: str) -> None:
    composition = np.asarray([12, 8, 5, 4, 3, 2, 1, 1, 1, 1, 0, 0])
    first = apply_intervention(composition, arm, "state", "seed")
    second = apply_intervention(composition, arm, "state", "seed")
    assert np.array_equal(first.composition, second.composition)
    assert first.steps == second.steps
    assert first.mass_before == first.mass_after == int(composition.sum())
    assert np.all(first.composition >= 0)
    if arm.endswith("D4"):
        d1 = apply_intervention(composition, arm[:-1] + "1", "state", "seed")
        assert first.steps[:1] == d1.steps


def test_axis_specific_edit_contracts() -> None:
    composition = np.asarray([10, 6, 4, 2, 1, 1, 0, 0])
    concentrate = apply_intervention(composition, "EVEN_CONCENTRATE_D4", "s", "z")
    flatten = apply_intervention(composition, "EVEN_FLATTEN_D4", "s", "z")
    contract = apply_intervention(composition, "RICH_CONTRACT_D1", "s", "z")
    expand = apply_intervention(composition, "RICH_EXPAND_D1", "s", "z")
    assert concentrate.occupied_after == concentrate.occupied_before
    assert concentrate.simpson_after < concentrate.simpson_before
    assert flatten.occupied_after == flatten.occupied_before
    assert flatten.simpson_after > flatten.simpson_before
    assert contract.occupied_after == contract.occupied_before - 1
    assert expand.occupied_after == expand.occupied_before + 1


def test_bray_decomposition_is_additive() -> None:
    output = bray_pair_decomposition(np.asarray([8, 2, 0, 0, 0, 0]), np.asarray([6, 1, 1, 1, 1, 0]))
    total = output["top1_contribution"] + output["rank2_to5_contribution"] + output["tail6plus_contribution"]
    assert total == pytest.approx(output["bray_distance"])
    assert output["dominant_type_same"] == 1.0


def test_edited_snapshot_preserves_all_noncomposition_fields() -> None:
    original = Snapshot(
        composition=np.asarray([3, 2, 1]),
        generation=20,
        inheritance=(True, False),
        boundary_h=(0.91, 0.82),
        previous_growth_steps=17,
        cumulative_growth_steps=901,
    )
    edited = _edited_snapshot(original, np.asarray([2, 3, 1]))
    _assert_snapshot_preservation(original, edited)
    assert np.array_equal(edited.composition, [2, 3, 1])


def test_intervention_q_uses_event_and_conditional_transition_denominators() -> None:
    gates = np.asarray(
        [
            [[[0], [1], [2], [4]], [[4], [4], [3], [0]]],
            [[[1], [1], [1], [1]], [[0], [0], [0], [0]]],
        ],
        dtype=np.int8,
    )
    event_q, event_n = _intervention_q(gates, 0, 0, slice(0, 4), None)
    assert event_q.tolist() == pytest.approx([0.25, 0.0])
    assert event_n.tolist() == [4, 4]
    run_q, run_n = _intervention_q(gates, 0, 0, slice(0, 4), 1)
    assert run_q.tolist() == pytest.approx([2 / 3, 0.0])
    assert run_n.tolist() == [3.0, 4.0]


def test_intervention_power_counts_paired_branches_and_matrices() -> None:
    labels = np.zeros((4, 2, 3), dtype=np.int8)
    labels[0, 0, :] = 1
    labels[1, 1, :] = 1
    output = _intervention_power_row(labels, np.asarray([0, 0, 1, 1]), 0, 1)
    assert output["left_events"] == 3
    assert output["right_events"] == 3
    assert output["discordant_branch_pairs"] == 6
    assert output["power_adequate"] is False
