from __future__ import annotations

import inspect

import numpy as np

from plastic_heredity.config import GardConfig
from plastic_heredity.intervention_core import FrozenFullPredictor, MolecularEdit, ScoredEdit
from plastic_heredity.phir_extension_px1 import (
    ARMS,
    EXPECTED_MODEL_SHA256,
    FINAL_START,
    HORIZON,
    MODEL_SOURCE,
    PRIMARY_REPRESENTATIONS,
    _extreme_choice,
    _future_seed,
    _random_action_seed,
    phase_protocol,
    scientific_spec,
    smoke_spec,
    validation_checks,
)


def _calibration() -> dict:
    cells = {}
    for candidate in ("02", "03"):
        for replicate in range(2):
            cells[f"{candidate}_r{replicate}"] = {
                representation: {
                    "noop_matrix_sd": 1.0,
                    "equivalence_margin": 0.2,
                }
                for representation in PRIMARY_REPRESENTATIONS
            }
    return {"cells": cells}


def test_px1_scientific_contract_is_fixed() -> None:
    spec = scientific_spec()
    assert spec.matrices == 24
    assert spec.replicates == 2
    assert spec.horizon == 60
    assert spec.final_start == 30
    assert spec.cpu_allocation_seconds == 8 * 3600
    assert ARMS == ("STABILIZE", "DESTABILIZE", "RANDOM", "NOOP")
    assert PRIMARY_REPRESENTATIONS == ("material", "functional_flux")
    assert HORIZON == 60 and FINAL_START == 30


def test_future_stream_has_no_arm_and_random_action_is_separate() -> None:
    assert "arm" not in inspect.signature(_future_seed).parameters
    spec = scientific_spec()
    assert _future_seed(spec, "02", 5, 1) != _random_action_seed(spec, "02", 5, 1)
    assert _future_seed(spec, "02", 5, 1) == _future_seed(spec, "02", 5, 1)


def test_extreme_choice_direction_and_ties_are_deterministic() -> None:
    scores = (
        ScoredEdit(MolecularEdit(1, 3), 0.2, -0.1),
        ScoredEdit(MolecularEdit(0, 2), 0.2, -0.1),
        ScoredEdit(MolecularEdit(1, 0), 0.8, 0.1),
        ScoredEdit(MolecularEdit(0, 1), 0.8, 0.1),
    )
    assert _extreme_choice(scores, True).edit == MolecularEdit(0, 2)
    assert _extreme_choice(scores, False).edit == MolecularEdit(0, 1)


def test_protocol_seals_full_primary_family_and_no_48() -> None:
    protocol = phase_protocol(_calibration())
    assert protocol["primary_family"].startswith("eight")
    assert protocol["no_48_matrix_continuation"]
    assert not protocol["arm_in_future_stream_key"]
    assert protocol["random_action_stream_separate"]
    assert protocol["measurement"].startswith("31 unique")


def test_frozen_model_is_serialization_stable() -> None:
    first = FrozenFullPredictor.load(MODEL_SOURCE)
    second = FrozenFullPredictor.load(MODEL_SOURCE)
    rng = np.random.default_rng(41)
    state = rng.normal(size=(3, 195))
    history = rng.normal(size=(3, 9))
    for candidate in ("02", "03"):
        np.testing.assert_array_equal(
            first.predict_features(candidate, state, history),
            second.predict_features(candidate, state, history),
        )
    assert len(EXPECTED_MODEL_SHA256) == 64


def test_smoke_and_scientific_seed_domains_are_disjoint() -> None:
    assert _future_seed(smoke_spec(), "03", 0, 0) != _future_seed(
        scientific_spec(), "03", 0, 0
    )


def test_complete_px0_validation_checks() -> None:
    checks = validation_checks()
    assert len(checks) >= 20
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]
