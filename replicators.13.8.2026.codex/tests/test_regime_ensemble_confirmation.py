from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import plastic_heredity.regime_ensemble_confirmation as ensemble
from plastic_heredity.regime_ensemble_confirmation import (
    _protocol,
    _seed_domains,
    ensemble_probability,
    prepare_registration,
    score_confirmation,
    verify_failed_pilot_source,
    verify_registration,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "results" / "regime_prediction_registration"
PILOT = ROOT / "results" / "regime_prediction_pilot"


def test_ensemble_is_equal_weight_arithmetic_probability_mean():
    direct = np.asarray((0.1, 0.8, 0.25))
    hurdle = np.asarray((0.3, 0.4, 0.75))
    observed = ensemble_probability(direct, hurdle)
    assert np.array_equal(observed, 0.5 * direct + 0.5 * hurdle)
    assert np.allclose(observed, np.asarray((0.2, 0.6, 0.5)))
    # A logit-scale average would not equal 0.2 for the first pair.
    assert observed[0] != pytest.approx(0.1791287847)
    with pytest.raises(ValueError, match="shapes differ"):
        ensemble_probability(direct, hurdle[:2])
    with pytest.raises(ValueError, match="finite probabilities"):
        ensemble_probability(direct, np.asarray((0.3, 1.1, 0.4)))


def test_protocol_freezes_one_common_fit_free_confirmation():
    protocol = _protocol()
    predictor = protocol["frozen_predictor"]
    cohort = protocol["cohort"]
    assert predictor["direct_weight"] == predictor["hurdle_weight"] == 0.5
    assert predictor["scale"] == "probability"
    assert predictor["same_family_recipe_for_both_candidates"]
    assert cohort["matrices"] == 200
    assert cohort["states"] == 2_000
    assert cohort["primary_futures"] == 256_000
    assert cohort["number_of_confirmation_cohorts"] == 1
    assert cohort["full_exact_replay"]
    assert protocol["primary_inference"]["overall_gate"].startswith("all three")
    prohibited = " ".join(protocol["prohibited_after_registration"])
    assert "refitting" in prohibited
    assert "recalibration" in prohibited
    assert "weight fitting" in prohibited
    assert "family switching" in prohibited
    assert "nonlinear fallback" in prohibited


def test_new_seed_domains_are_unique_from_every_earlier_campaign():
    domains = _seed_domains()
    assert len(domains) == len(set(domains.values()))
    assert domains["ensemble_confirmation"] == ensemble.CONFIRMATION_MASTER_SEED
    assert domains["ensemble_bootstrap"] == ensemble.BOOTSTRAP_MASTER_SEED
    assert domains["ensemble_randomization"] == ensemble.RANDOMIZATION_MASTER_SEED


def test_confirmation_module_has_no_fit_function_call_or_import():
    tree = ast.parse(Path(ensemble.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not any(name.startswith("fit") for name in imported)
    assert not any(name.startswith("fit") for name in called)


def _strong_cases_and_labels():
    cases = [
        SimpleNamespace(candidate=candidate, matrix_id=matrix_id)
        for candidate in ("02", "03")
        for matrix_id in range(20)
        for _ in range(5)
    ]
    labels = np.ones((len(cases), ensemble.BRANCHES), dtype=np.int8)
    predictions = {
        candidate: {
            "h10": np.full(100, 0.5),
            "direct": np.full(100, 0.9),
            "hurdle": np.full(100, 0.9),
            "ensemble": np.full(100, 0.9),
        }
        for candidate in ("02", "03")
    }
    power = {candidate: {"adequate": True} for candidate in ("02", "03")}
    return cases, labels, predictions, power


def test_scoring_applies_all_four_frozen_transfer_gates():
    cases, labels, predictions, power = _strong_cases_and_labels()
    result = score_confirmation(
        cases,
        labels,
        predictions,
        power,
        bootstrap_repetitions=128,
        randomization_repetitions=255,
    )
    assert result["power_adequate"]
    assert result["all_four_transfer_gates_pass"]
    assert result["primary_prediction_supported"]
    assert result["family_size"] == 4
    assert not result["fitting_or_recalibration_performed"]
    assert len(result["primary_tests"]) == 4
    assert all(row["passes_transfer_gate"] for row in result["primary_tests"])
    assert all(row["randomization_p_holm"] < 0.05 for row in result["primary_tests"])


def test_scoring_rejects_any_changed_ensemble_formula():
    cases, labels, predictions, power = _strong_cases_and_labels()
    predictions["02"]["ensemble"] = np.full(100, 0.89)
    with pytest.raises(ValueError, match="ensemble formula mismatch"):
        score_confirmation(
            cases,
            labels,
            predictions,
            power,
            bootstrap_repetitions=8,
            randomization_repetitions=8,
        )


def test_stopped_pilot_is_verified_without_rewriting_its_failure():
    source = verify_failed_pilot_source(DESIGN, PILOT)
    assert source["seal"]["status"] == "stopped_before_confirmation"
    assert source["seal"]["selection"]["passed"] is False
    assert source["manifest"]["confirmation_authorized"] is False
    assert source["seal"]["pilot_seal_id"] == (
        "4db89c5095682c4cf055ed0cb26f9ba80972fd849e8f9a722fe6edf84b3b08a7"
    )


def test_registration_roundtrip_extracts_only_direct_and_hurdle(tmp_path: Path):
    destination = tmp_path / "registration"
    prepare_registration(DESIGN, PILOT, destination)
    payload = verify_registration(destination)
    assert payload["status"] == "sealed_before_confirmation_matrix_generation"
    assert payload["source_pilot"]["pilot_status"] == "stopped_before_confirmation"
    assert payload["source_pilot"]["pilot_selection_passed"] is False
    assert payload["confirmation_count"] == 1
    assert payload["portable_verification"]["states"] == 800
    assert payload["portable_verification"]["all_arrays_exact"]
    audit = __import__("json").loads(
        (destination / "development_audit.json").read_text(encoding="utf-8")
    )
    assert audit["portable_predictions_bit_exact"]
    assert audit["baseline_direct_hurdle_exact"]
    assert set(audit["models"]) == {"02", "03"}
    assert set(audit["models"]["02"]) == {"direct", "hurdle"}
    assert not any(audit["confirmation_operations"].values())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_registration(DESIGN, PILOT, destination)
