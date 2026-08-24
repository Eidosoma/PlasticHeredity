from __future__ import annotations

from math import cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pytest

from plastic_heredity.regime_confirmation import (
    BOOTSTRAP_MASTER_SEED,
    CONFIRMATION_MASTER_SEED,
    DEVELOPMENT_MASTER_SEED,
    ENDPOINTS,
    RANDOMIZATION_MASTER_SEED,
    RegimeBranchBatch,
    _power_counts,
    _protocol,
    _readback_metrics,
    _replay_audit,
    _state_table,
    _write_branch_table,
    _all_labels,
    compute_occurrence_metrics,
    compute_prediction_metrics,
    evaluate_regime,
    register_design,
    run_development,
    verify_design,
)
from plastic_heredity.regime_confirmation_recovery import round_trip_csv_readback
from plastic_heredity.experiment import StateCase
from plastic_heredity.metrics import centered_spearman
from plastic_heredity.simulator import FissionRecord, Snapshot, cosine_similarity


def _vector(angle: float) -> np.ndarray:
    value = radians(angle)
    return np.asarray((0.0, cos(value), sin(value)), dtype=np.float64)


def _record(parent: np.ndarray, daughter: np.ndarray, h: float) -> FissionRecord:
    return FissionRecord(
        parent=np.asarray(parent),
        daughter=np.asarray(daughter),
        h=h,
        growth_steps=1,
    )


def _episode(daughters: list[np.ndarray], inherited_h: float = 0.95):
    anchor = np.asarray((1.0, 0.0, 0.0))
    records = [_record(anchor, daughters[0], 0.50)]
    parent = daughters[0]
    for daughter in daughters:
        records.append(_record(parent, daughter, inherited_h))
        parent = daughter
    return records


def test_identical_distinct_daughters_pass_all_three_frozen_endpoints():
    daughter = _vector(0.0)
    outcome = evaluate_regime(_episode([daughter] * 8))
    assert outcome.primary_all8
    assert outcome.secondary_first5
    assert outcome.secondary_centroid
    assert outcome.primary_all8_onset == 1
    assert outcome.first_break_index == 0
    assert outcome.first_run8_start == 1


def test_first_five_definition_can_pass_when_all_eight_pairwise_fails():
    daughters = [_vector(value) for value in (0, 0, 0, 0, 0, 20, 40, 60)]
    outcome = evaluate_regime(_episode(daughters))
    assert not outcome.primary_all8
    assert outcome.secondary_first5
    assert not outcome.secondary_centroid
    assert outcome.first_run8_minimum_pairwise_all8 < 0.9
    assert outcome.first_run8_minimum_pairwise_first5 == pytest.approx(1.0)


def test_centroid_definition_can_pass_when_pairwise_definitions_fail():
    daughters = [_vector(value) for value in (0, 7, 14, 21, 29, 36, 43, 50)]
    outcome = evaluate_regime(_episode(daughters))
    assert not outcome.primary_all8
    assert not outcome.secondary_first5
    assert outcome.secondary_centroid
    assert outcome.first_run8_minimum_centroid_all8 > 0.9


def test_inheritance_and_coherence_are_strict_at_point_nine():
    daughter = _vector(0.0)
    inheritance_equal = evaluate_regime(_episode([daughter] * 8, inherited_h=0.90))
    assert not any(getattr(inheritance_equal, endpoint) for endpoint in ENDPOINTS)

    boundary = np.asarray((0.0, 0.9, sqrt(1.0 - 0.9**2)))
    assert cosine_similarity(daughter, boundary) == pytest.approx(0.9)
    daughters = [
        daughter,
        boundary,
        daughter,
        daughter,
        daughter,
        daughter,
        daughter,
        daughter,
    ]
    coherence_equal = evaluate_regime(_episode(daughters))
    assert not coherence_equal.primary_all8
    assert not coherence_equal.secondary_first5


def test_distinctness_is_inclusive_at_point_eight_five():
    anchor = np.asarray((1.0, 0.0))
    daughter = np.asarray((0.85, sqrt(1.0 - 0.85**2)))
    assert cosine_similarity(anchor, daughter) == 0.85
    records = [_record(anchor, daughter, 0.5)]
    records.extend(_record(daughter, daughter, 0.95) for _ in range(8))
    outcome = evaluate_regime(records)
    assert outcome.primary_all8
    assert outcome.first_run8_maximum_anchor_all8 == 0.85


def test_any_anchor_similarity_above_distinctness_boundary_fails_primary():
    anchor = np.asarray((1.0, 0.0))
    daughter = np.asarray((0.851, sqrt(1.0 - 0.851**2)))
    records = [_record(anchor, daughter, 0.5)]
    records.extend(_record(daughter, daughter, 0.95) for _ in range(8))
    assert not evaluate_regime(records).primary_all8


def test_no_break_or_incomplete_run_is_negative():
    daughter = _vector(0.0)
    no_break = [_record(daughter, daughter, 0.95) for _ in range(12)]
    assert not evaluate_regime(no_break).primary_all8
    assert not evaluate_regime(_episode([daughter] * 7)).primary_all8


def test_search_selects_earliest_qualifying_window_after_failed_run():
    drift = [_vector(value) for value in (0, 0, 0, 0, 0, 20, 40, 60)]
    records = _episode(drift)
    old = drift[-1]
    new = _vector(0.0)
    records.append(_record(old, new, 0.5))
    records.extend(_record(new, new, 0.95) for _ in range(8))
    outcome = evaluate_regime(records)
    assert outcome.first_run8_start == 1
    assert outcome.primary_all8_onset == 10


def test_endpoint_is_invariant_to_common_molecule_relabelling():
    daughters = [_vector(value) for value in (0, 7, 14, 21, 29, 36, 43, 50)]
    records = _episode(daughters)
    permutation = np.asarray((2, 0, 1))
    relabelled = [
        FissionRecord(
            parent=record.parent[permutation],
            daughter=record.daughter[permutation],
            h=record.h,
            growth_steps=record.growth_steps,
        )
        for record in records
    ]
    left = evaluate_regime(records).to_dict()
    right = evaluate_regime(relabelled).to_dict()
    for key in left:
        if isinstance(left[key], float):
            assert left[key] == pytest.approx(right[key], abs=1e-15)
        else:
            assert left[key] == right[key]


def test_power_gate_requires_both_event_and_matrix_minima():
    candidates = np.asarray(["02"] * 25 + ["03"] * 25)
    matrix_ids = np.concatenate((np.arange(25), np.arange(25)))
    labels = np.zeros((50, 128), dtype=np.int8)
    labels[:20, :5] = 1
    labels[25:45, :5] = 1
    adequate = _power_counts(labels, candidates, matrix_ids)
    assert adequate["02"]["events"] == 100
    assert adequate["02"]["event_matrices"] == 20
    assert adequate["02"]["adequate"]
    labels[44] = 0
    inadequate = _power_counts(labels, candidates, matrix_ids)
    assert not inadequate["03"]["adequate"]


def test_absent_endpoint_reports_underpowered_prediction_without_crashing(monkeypatch):
    monkeypatch.setattr(
        "plastic_heredity.regime_confirmation.BOOTSTRAP_REPETITIONS", 32
    )
    monkeypatch.setattr(
        "plastic_heredity.regime_confirmation.RANDOMIZATION_REPETITIONS", 32
    )
    candidates = np.asarray(["02"] * 10 + ["03"] * 10)
    matrix_ids = np.concatenate((np.arange(10), np.arange(10)))
    labels = {endpoint: np.zeros((20, 8), dtype=np.int8) for endpoint in ENDPOINTS}
    predictions = {
        endpoint: {
            candidate: {
                "h10": np.full(10, 1e-8),
                "h10_state": np.full(10, 1e-8),
            }
            for candidate in ("02", "03")
        }
        for endpoint in ENDPOINTS
    }
    power = {
        candidate: {
            "events": 0,
            "event_matrices": 0,
            "minimum_events": 100,
            "minimum_event_matrices": 20,
            "adequate": False,
        }
        for candidate in ("02", "03")
    }
    metrics = compute_prediction_metrics(
        labels, predictions, candidates, matrix_ids, power, power
    )
    assert not metrics["prediction_power_adequate"]
    assert not metrics["primary_prediction_supported"]
    assert np.isnan(
        metrics["endpoints"][ENDPOINTS[0]]["candidates"]["02"][
            "branch_half_reliability"
        ]
    )


def test_branch_and_state_artifact_readback_recomputes_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr("plastic_heredity.regime_confirmation.BRANCHES", 8)
    monkeypatch.setattr(
        "plastic_heredity.regime_confirmation.BOOTSTRAP_REPETITIONS", 32
    )
    monkeypatch.setattr(
        "plastic_heredity.regime_confirmation.RANDOMIZATION_REPETITIONS", 32
    )
    cases = []
    batches = []
    for candidate in ("02", "03"):
        for matrix_id in range(10):
            snapshot = Snapshot(
                composition=np.asarray((0, 10, 0), dtype=np.int64),
                generation=20,
                inheritance=(),
                boundary_h=(),
            )
            cases.append(
                StateCase(
                    state_id=f"readback-c{candidate}-m{matrix_id:03d}",
                    cohort="REGCONF",
                    candidate=candidate,
                    matrix_id=matrix_id,
                    landmark=20,
                    beta=np.ones((3, 3)),
                    snapshot=snapshot,
                )
            )
            batches.append(
                RegimeBranchBatch(
                    targets=np.zeros((8, 3), dtype=np.int8),
                    onsets=np.full((8, 3), -1, dtype=np.int16),
                    completed_horizon=np.ones(8, dtype=np.int8),
                    observed_fissions=np.full(8, 32, dtype=np.int16),
                    first_break_index=np.zeros(8, dtype=np.int16),
                    first_run8_start=np.full(8, -1, dtype=np.int16),
                    geometry=np.full((8, 5), np.nan),
                )
            )
    labels = _all_labels(batches)
    predictions = {
        endpoint: {
            candidate: {
                "h10": np.full(10, 1e-8),
                "h10_state": np.full(10, 1e-8),
            }
            for candidate in ("02", "03")
        }
        for endpoint in ENDPOINTS
    }
    branch_path = tmp_path / "branches.csv.gz"
    state_path = tmp_path / "states.csv"
    _write_branch_table(branch_path, cases, batches)
    _state_table(cases, labels, predictions).to_csv(state_path, index=False)
    power = {
        candidate: {
            "events": 0,
            "event_matrices": 0,
            "minimum_events": 100,
            "minimum_event_matrices": 20,
            "adequate": False,
        }
        for candidate in ("02", "03")
    }
    read_occurrence, read_prediction, read_power = _readback_metrics(
        branch_path,
        state_path,
        {endpoint: ("h10", "h10_state") for endpoint in ENDPOINTS},
        power,
    )
    candidates = np.asarray([case.candidate for case in cases])
    matrix_ids = np.asarray([case.matrix_id for case in cases])
    expected_occurrence = compute_occurrence_metrics(
        labels, candidates, matrix_ids, "REGCONF"
    )
    assert read_occurrence == expected_occurrence
    assert read_power == power
    assert not read_prediction["primary_prediction_supported"]


def test_round_trip_csv_readback_preserves_centered_ranks(tmp_path):
    """Default parsing can move ULP-separated static-model predictions."""

    matrix_values = np.linspace(0.01, 0.99, 200)
    predictions = np.concatenate(
        [
            [
                value,
                np.nextafter(value, np.inf),
                value,
                np.nextafter(value, -np.inf),
                value,
            ]
            for value in matrix_values
        ]
    )
    outcomes = np.tile(np.asarray((0.0, 0.25, 0.5, 0.75, 1.0)), 200)
    matrix_ids = np.repeat(np.arange(200), 5)
    path = tmp_path / "predictions.csv"
    import pandas as pd

    pd.DataFrame({"prediction": predictions}).to_csv(path, index=False)
    default = pd.read_csv(path)["prediction"].to_numpy()
    assert not np.array_equal(default, predictions)

    with round_trip_csv_readback():
        restored = pd.read_csv(path)["prediction"].to_numpy()

    assert np.array_equal(restored, predictions)
    assert centered_spearman(restored, outcomes, matrix_ids) == centered_spearman(
        predictions, outcomes, matrix_ids
    )


def _batch(geometry_value: float = 0.75) -> RegimeBranchBatch:
    return RegimeBranchBatch(
        targets=np.zeros((2, 3), dtype=np.int8),
        onsets=np.full((2, 3), -1, dtype=np.int16),
        completed_horizon=np.ones(2, dtype=np.int8),
        observed_fissions=np.full(2, 32, dtype=np.int16),
        first_break_index=np.zeros(2, dtype=np.int16),
        first_run8_start=np.ones(2, dtype=np.int16),
        geometry=np.full((2, 5), geometry_value, dtype=np.float64),
    )


def test_replay_requires_exact_discrete_and_tolerant_continuous_values():
    audit = _replay_audit([_batch()], [_batch(0.75 + 5e-15)])
    assert audit["discrete_exact"]
    assert audit["continuous_within_1e-14"]
    changed = _batch()
    changed.targets[0, 0] = 1
    with pytest.raises(ValueError, match="discrete endpoint replay mismatch"):
        _replay_audit([_batch()], [changed])
    with pytest.raises(ValueError, match="continuous replay mismatch"):
        _replay_audit([_batch()], [_batch(0.75 + 2e-14)])


def test_design_registration_roundtrip_and_seed_domains(tmp_path: Path):
    registration = tmp_path / "design"
    register_design(registration)
    payload = verify_design(registration)
    assert payload["status"] == "sealed_before_development_matrix_generation"
    seeds = {
        DEVELOPMENT_MASTER_SEED,
        CONFIRMATION_MASTER_SEED,
        BOOTSTRAP_MASTER_SEED,
        RANDOMIZATION_MASTER_SEED,
    }
    assert len(seeds) == 4
    protocol = _protocol()
    assert protocol["inference"]["secondary_endpoints_may_rescue_primary"] is False
    assert protocol["feature_model"]["uses_pca"] is False


def test_expensive_stages_refuse_existing_output_before_validation(tmp_path: Path):
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        register_design(destination)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_development(tmp_path / "missing", destination, workers=1)
