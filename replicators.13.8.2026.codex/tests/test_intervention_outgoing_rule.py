from __future__ import annotations

import numpy as np
import pytest

from plastic_heredity import intervention_replication as base
from plastic_heredity.experiment import StateCase
from plastic_heredity.intervention_core import enumerate_legal_edits
from plastic_heredity.intervention_outgoing_rule import (
    LABEL,
    SEED_DOMAINS,
    _future_seed,
    _protocol,
    add_derived_pilot_eligibility,
    outgoing_catalytic_influence,
    phase_spec,
    select_outgoing_rule_edits,
    validation_checks,
)


@pytest.fixture(scope="session")
def validated() -> dict:
    return validation_checks()


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    composition = np.asarray([4, 2, 0, 2], dtype=np.int64)
    beta = np.asarray(
        [
            [1.0, 40.0, 2.0, 3.0],
            [7.0, 1.0, 5.0, 2.0],
            [80.0, 4.0, 1.0, 9.0],
            [2.0, 3.0, 60.0, 1.0],
        ],
        dtype=np.float64,
    )
    return composition, beta


def test_complete_validation_passes_without_scientific_cohort(validated: dict) -> None:
    assert validated["all_checks_passed"] is True
    assert validated["scientific_cohort_generated"] is False
    assert validated["check_count"] == 34


def test_fable_outgoing_formula_is_x_times_beta() -> None:
    composition, beta = _fixture()
    x = composition / composition.sum()
    observed = outgoing_catalytic_influence(composition, beta)
    assert observed == pytest.approx(x @ beta, abs=0.0, rel=0.0)
    assert observed == pytest.approx(beta.T @ x, abs=0.0, rel=0.0)


def test_outgoing_and_incoming_are_distinct_on_asymmetric_beta() -> None:
    composition, beta = _fixture()
    x = composition / composition.sum()
    assert not np.array_equal(x @ beta, beta @ x)


def test_rule_extrema_are_exhaustive_and_have_registered_direction() -> None:
    composition, beta = _fixture()
    influence = outgoing_catalytic_influence(composition, beta)
    legal = enumerate_legal_edits(composition)
    differences = np.asarray(
        [influence[edit.add_type] - influence[edit.remove_type] for edit in legal]
    )
    rules = select_outgoing_rule_edits(composition, beta)
    assert rules["RULE_DOWN"] == legal[int(np.argmax(differences))]
    assert rules["RULE_UP"] == legal[int(np.argmin(differences))]
    assert differences[legal.index(rules["RULE_DOWN"])] > 0.0
    assert differences[legal.index(rules["RULE_UP"])] < 0.0


def test_rule_rejects_empty_composition() -> None:
    with pytest.raises(ValueError):
        outgoing_catalytic_influence(np.zeros(3, dtype=np.int64), np.eye(3))


def test_outgoing_influence_is_permutation_equivariant() -> None:
    composition, beta = _fixture()
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    original = outgoing_catalytic_influence(composition, beta)
    permuted = outgoing_catalytic_influence(
        composition[permutation], beta[np.ix_(permutation, permutation)]
    )
    assert np.array_equal(permuted, original[permutation])


def test_seed_domains_are_fresh_and_future_keys_are_arm_free() -> None:
    composition, beta = _fixture()
    case = StateCase(
        "seed-fixture",
        "FIX",
        "02",
        9,
        20,
        beta,
        base._fixture_snapshot(),
    )
    spec = phase_spec()
    seeds = [_future_seed(spec, case, branch) for branch in range(4)]
    assert len(set(seeds)) == 4
    assert set(SEED_DOMAINS.values()).isdisjoint(base.SEED_DOMAINS.values())
    assert LABEL not in base.PHASE_LABEL.values()


def test_readback_derives_replay_dependent_field() -> None:
    metrics = {"pilot_eligibility_without_replay": True}
    assert add_derived_pilot_eligibility(metrics, True)["pilot_eligibility"] is True
    metrics = {"pilot_eligibility_without_replay": True}
    assert add_derived_pilot_eligibility(metrics, False)["pilot_eligibility"] is False


def test_protocol_freezes_correction_and_budget() -> None:
    protocol = _protocol()
    assert protocol["correction_basis"]["external_frozen_expression"] == "x @ beta"
    assert protocol["correction_basis"]["codex_equivalent"] == "beta.T @ x"
    assert protocol["original_p2"]["scientific_result_unchanged"] is True
    assert protocol["original_p2"]["classification"].startswith(
        "incoming-support negative control"
    )
    assert protocol["design"]["matrices"] == 40
    assert protocol["design"]["branches_per_arm_per_state"] == 32
    assert protocol["lifecycle"][
        "readback_derives_pilot_eligibility_before_comparison"
    ] is True


def test_phase_keeps_registered_arm_and_contrast_order() -> None:
    spec = phase_spec()
    assert spec.arms == ("RULE_UP", "RULE_DOWN", "RANDOM", "NOOP")
    assert spec.contrast == ("RULE_UP", "RULE_DOWN")
