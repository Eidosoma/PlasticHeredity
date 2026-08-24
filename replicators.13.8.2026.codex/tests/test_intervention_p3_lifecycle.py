from __future__ import annotations

import numpy as np

from plastic_heredity import intervention_replication as base
from plastic_heredity.intervention_p3_lifecycle import (
    _protocol,
    add_derived_pilot_eligibility,
    validation_checks,
)


def test_p3_lifecycle_validation_passes_without_scientific_cohort() -> None:
    validation = validation_checks()
    assert validation["all_checks_passed"] is True
    assert validation["scientific_cohort_generated"] is False
    assert validation["check_count"] == 29


def test_readback_field_uses_inference_and_replay() -> None:
    metrics = {"pilot_eligibility_without_replay": True}
    assert add_derived_pilot_eligibility(metrics, True)["pilot_eligibility"] is True
    metrics = {"pilot_eligibility_without_replay": True}
    assert add_derived_pilot_eligibility(metrics, False)["pilot_eligibility"] is False
    metrics = {"pilot_eligibility_without_replay": False}
    assert add_derived_pilot_eligibility(metrics, True)["pilot_eligibility"] is False


def test_protocol_changes_no_scientific_contract() -> None:
    protocol = _protocol()
    frozen = protocol["original_scientific_design"]
    spec = base.pilot_spec("p3")
    assert protocol["scientific_contract_changes"] == []
    assert frozen["arms"] == list(spec.arms)
    assert frozen["contrast"] == list(spec.contrast)
    assert frozen["cohort_seed"] == spec.cohort_seed
    assert frozen["future_seed"] == spec.future_seed
    assert protocol["execution"]["mandatory_stop_after_p3"] is True


def test_present_present_edge_set_is_transpose_invariant() -> None:
    present = np.asarray([1, 3, 8], dtype=np.int64)
    rows, columns = np.meshgrid(present, present, indexing="ij")
    block = set(zip(rows.ravel().tolist(), columns.ravel().tolist(), strict=True))
    assert block == {(column, row) for row, column in block}
