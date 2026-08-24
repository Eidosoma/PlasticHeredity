from __future__ import annotations

import numpy as np

from e01_clean_directional_confirmation.core import (
    CANDIDATE_IDS,
    PRIMARY_LABEL_ID,
    ROOT_SEED_HEX,
    SENSITIVITY_LABEL_ID,
    candidate_registry,
    classify,
    derive_seed,
    exact_label_identity,
    fixed_label_spec,
    prefix_gate,
    primary_association_gate,
    primary_drift_gate,
    seed_material_sha256,
)


def _passing_row() -> dict[str, float | int]:
    return {
        "finiteCoverage": 1.0,
        "definedCorrelationCount": 100,
        "positiveCorrelationFraction": 0.73,
        "medianCorrelation": 0.03,
        "bootstrapLower95": 0.01,
        "circularShiftPositiveP": 1 / 4097,
        "definedDriftCount": 100,
        "higherDuringReplicationFraction": 0.57,
        "medianMeanDifference": 0.1,
        "driftBootstrapLower95": 0.02,
        "driftCircularShiftPositiveP": 1 / 4097,
    }


def test_exact_two_candidate_contract() -> None:
    rows = candidate_registry()
    assert tuple(row["candidateId"] for row in rows) == CANDIDATE_IDS
    assert [row["h"] for row in rows] == [0.6031526490073492, 0.5613315384859516]
    assert rows[0]["daughterRule"] == "FIRST_DAUGHTER"
    assert rows[1]["daughterRule"] == "RANDOM_NONEMPTY"
    assert all(row["clockId"] == "C1_SELECTED_DAUGHTER_RETAINED" for row in rows)


def test_seed_domain_is_replayable_and_material_is_identity_specific() -> None:
    assert len(ROOT_SEED_HEX) == 64
    first = derive_seed("source", "candidate", 0, "full")
    assert first == derive_seed("source", "candidate", 0, "full")
    assert first != derive_seed("source", "candidate", 1, "full")
    assert seed_material_sha256("a") != seed_material_sha256("b")


def test_s13x_label_specs_are_recovered_exactly() -> None:
    primary = fixed_label_spec(PRIMARY_LABEL_ID)
    sensitivity = fixed_label_spec(SENSITIVITY_LABEL_ID)
    assert primary.family == "MOLECULAR_ADJACENT_INCOMING"
    assert primary.threshold == 0.9
    assert sensitivity.threshold == 0.97


def test_label_identity_exposes_structural_circularity() -> None:
    h = np.array([0.8, 0.9, 0.9000001, 0.99])
    labels = h > 0.9
    result = exact_label_identity(h, labels)
    assert result["identityPassed"]
    assert result["mismatchCount"] == 0
    assert result["conditionalEntropyYGivenExactHBits"] == 0.0
    assert result["conditionalInformationEmergenceYGivenExactHBits"] == 0.0


def test_directional_gates_and_all_candidate_classification() -> None:
    row = _passing_row()
    assert primary_association_gate(row)
    assert primary_drift_gate(row)
    assert prefix_gate(row)
    candidates = [
        {"candidateId": candidate, "candidatePrimaryPassed": True}
        for candidate in CANDIDATE_IDS
    ]
    assert (
        classify(candidates, exact_h_identity_passed=True, validation_passed=True)
        == "LABEL_COUPLED_RETROSPECTIVE_RESEMBLANCE"
    )
    candidates[1]["candidatePrimaryPassed"] = False
    assert (
        classify(candidates, exact_h_identity_passed=True, validation_passed=True)
        == "CLEAN_DIRECTIONAL_BRANCH_NOT_CONFIRMED"
    )
    assert (
        classify(candidates, exact_h_identity_passed=True, validation_passed=False)
        == "S13Y_VALIDATION_FAILED_CLOSED"
    )
