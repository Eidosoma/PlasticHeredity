from __future__ import annotations

from math import cos, radians, sin, sqrt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from plastic_heredity.config import ExperimentConfig, GardConfig
from plastic_heredity.regime_prediction import (
    PredictionBranchBatch,
    _protocol,
    _strict_labels_from_branch_table,
    read_checkpoint_status,
    register_design,
    replay_audit,
    run_prediction_branches,
    select_model_family,
    verify_design,
)
from plastic_heredity.regime_prediction_endpoints import (
    WINDOW_METRIC_NAMES,
    evaluate_rich_regime,
)
from plastic_heredity.regime_prediction_features import (
    DYNAMIC_FEATURE_NAMES,
    POST_BREAK_FEATURE_NAMES,
    local_dynamics_features,
    prediction_provenance_contract,
)
from plastic_heredity.regime_prediction_models import (
    HurdleModel,
    MODEL_FAMILIES,
    _crossfit_sequential_predictions,
    fit_sequential_ridge,
    matrix_folds,
)
from plastic_heredity.regime_confirmation import evaluate_regime
from plastic_heredity.simulator import FissionRecord, cosine_similarity


def _vector(angle: float) -> np.ndarray:
    theta = radians(angle)
    return np.asarray((0.0, cos(theta), sin(theta)))


def _record(parent: np.ndarray, daughter: np.ndarray, h: float) -> FissionRecord:
    return FissionRecord(parent=parent, daughter=daughter, h=h, growth_steps=1)


def _episode(daughters: list[np.ndarray], inherited_h: float = 0.95):
    anchor = np.asarray((1.0, 0.0, 0.0))
    records = [_record(anchor, daughters[0], 0.5)]
    parent = daughters[0]
    for daughter in daughters:
        records.append(_record(parent, daughter, inherited_h))
        parent = daughter
    return records


def test_rich_endpoint_binary_is_exactly_best_margin_positive():
    daughter = _vector(0.0)
    outcome = evaluate_rich_regime(_episode([daughter] * 8))
    assert outcome.primary_all8 == (outcome.best_strict_margin > 0.0)
    assert outcome.secondary_first5 == (outcome.best_first5_margin > 0.0)
    assert outcome.secondary_centroid == (outcome.best_centroid_margin > 0.0)
    assert outcome.break_event
    assert outcome.any_run8_after_break
    assert outcome.run8_window_count == 1
    assert outcome.longest_post_break_inheritance_run == 8
    assert outcome.windows[0].to_row().shape == (len(WINDOW_METRIC_NAMES),)


def test_rich_endpoint_preserves_inclusive_distinctness_boundary():
    anchor = np.asarray((1.0, 0.0))
    daughter = np.asarray((0.85, sqrt(1.0 - 0.85**2)))
    assert cosine_similarity(anchor, daughter) == 0.85
    records = [_record(anchor, daughter, 0.5)]
    records.extend(_record(daughter, daughter, 0.95) for _ in range(8))
    outcome = evaluate_rich_regime(records)
    assert outcome.primary_all8
    assert outcome.best_strict_margin > 0.0


def test_rich_endpoint_retains_failed_and_later_qualifying_windows():
    drifting = [_vector(value) for value in (0, 0, 0, 0, 0, 20, 40, 60)]
    records = _episode(drifting)
    new = _vector(0.0)
    records.append(_record(drifting[-1], new, 0.5))
    records.extend(_record(new, new, 0.95) for _ in range(8))
    outcome = evaluate_rich_regime(records)
    assert outcome.first_run8_start == 1
    assert outcome.primary_all8_onset == 10
    assert outcome.run8_window_count == 2
    assert outcome.windows[0].strict_margin <= 0.0
    assert outcome.windows[1].strict_margin > 0.0


def test_rich_endpoint_matches_sealed_binary_and_onset_contract_randomized():
    rng = np.random.default_rng(44)
    for _ in range(40):
        compositions = rng.integers(0, 6, size=(33, 10))
        compositions[compositions.sum(axis=1) == 0, 0] = 1
        records = [
            FissionRecord(
                parent=compositions[index],
                daughter=compositions[index + 1],
                h=float(rng.uniform()),
                growth_steps=1,
            )
            for index in range(32)
        ]
        sealed = evaluate_regime(records)
        rich = evaluate_rich_regime(records)
        assert rich.targets == (
            sealed.primary_all8,
            sealed.secondary_first5,
            sealed.secondary_centroid,
        )
        assert rich.onsets == (
            sealed.primary_all8_onset,
            sealed.secondary_first5_onset,
            sealed.secondary_centroid_onset,
        )
        assert rich.first_break_index == sealed.first_break_index
        assert rich.first_run8_start == sealed.first_run8_start


def test_dynamic_features_are_invariant_to_simultaneous_relabelling():
    rng = np.random.default_rng(3)
    config = GardConfig(n_types=8, n_min=4, n_max=8)
    experiment = ExperimentConfig(gard=config)
    composition = np.asarray((2, 0, 1, 3, 0, 2, 1, 0))
    previous = np.asarray((1, 1, 1, 2, 0, 2, 1, 1))
    beta = np.exp(rng.normal(-4.0, 1.0, size=(8, 8)))
    permutation = rng.permutation(8)
    original = local_dynamics_features(composition, beta, experiment, "02", previous)
    relabelled = local_dynamics_features(
        composition[permutation],
        beta[np.ix_(permutation, permutation)],
        experiment,
        "02",
        previous[permutation],
    )
    assert original.shape == (len(DYNAMIC_FEATURE_NAMES),)
    assert np.allclose(original, relabelled, rtol=1e-11, atol=1e-12)


def test_prediction_provenance_marks_velocity_as_history_dependent():
    contract = prediction_provenance_contract()
    assert set(contract) == {
        "h10",
        "state",
        "beta",
        "interaction",
        "dynamics",
        "post_break",
    }
    velocity = next(
        item
        for item in contract["dynamics"]
        if item["name"].endswith("velocity_drift_cosine")
    )
    assert velocity["provenance"]["depends_on_state"]
    assert velocity["provenance"]["depends_on_beta"]
    assert velocity["provenance"]["depends_on_history"]
    assert len(POST_BREAK_FEATURE_NAMES) == 44
    assert len(contract["post_break"]) == len(POST_BREAK_FEATURE_NAMES)


def test_matrix_folds_never_split_a_matrix():
    matrix_ids = np.repeat(np.arange(12), 5)
    folds = matrix_folds(matrix_ids)
    for matrix_id in np.unique(matrix_ids):
        assert np.unique(folds[matrix_ids == matrix_id]).size == 1
    assert set(folds) == set(range(5))


def test_sequential_ridge_handles_duplicate_and_constant_coordinates():
    rng = np.random.default_rng(4)
    matrix_ids = np.repeat(np.arange(10), 3)
    signal = rng.normal(size=matrix_ids.size)
    raw = {
        "h10": np.column_stack((signal, signal, np.ones(signal.size))),
        "added": np.column_stack((signal**2, np.zeros(signal.size))),
    }
    probability = 1.0 / (1.0 + np.exp(-(signal + 0.25 * signal**2)))
    trials = np.full(signal.size, 16.0)
    successes = rng.binomial(trials.astype(int), probability).astype(float)
    model = fit_sequential_ridge(raw, successes, trials, matrix_ids, ("h10", "added"))
    predictions = model.predict(raw)
    assert np.isfinite(predictions).all()
    assert np.all((predictions > 0.0) & (predictions < 1.0))
    assert model.transforms["h10"].kept_indices.size == 1
    assert model.transforms["added"].kept_indices.size == 1


def test_crossfit_feature_prediction_does_not_read_heldout_matrix_labels():
    rng = np.random.default_rng(8)
    matrix_ids = np.repeat(np.arange(10), 2)
    raw = {"h10": rng.normal(size=(20, 2))}
    labels = rng.binomial(1, 0.4, size=(20, 4)).astype(np.int8)
    folds = matrix_folds(matrix_ids)
    heldout = folds == 0
    first = _crossfit_sequential_predictions(
        raw, labels, matrix_ids, ("h10",), fixed_lambdas={"h10": 0.0}
    )
    changed = labels.copy()
    changed[heldout] = 1 - changed[heldout]
    second = _crossfit_sequential_predictions(
        raw, changed, matrix_ids, ("h10",), fixed_lambdas={"h10": 0.0}
    )
    assert np.array_equal(first[heldout], second[heldout])


def test_hurdle_prediction_is_product_of_registered_components():
    class Fixed:
        def __init__(self, value):
            self.value = value

        def predict(self, raw):
            return np.full(3, self.value)

    model = HurdleModel(Fixed(0.8), Fixed(0.5), Fixed(0.25))
    assert np.allclose(model.predict({}), 0.1)


def _selection_cases():
    return [
        SimpleNamespace(candidate=candidate, matrix_id=matrix_id)
        for candidate in ("02", "03")
        for matrix_id in range(10)
        for _ in range(2)
    ]


def test_selection_stops_when_any_candidate_half_has_no_gain(monkeypatch):
    monkeypatch.setattr(
        "plastic_heredity.regime_prediction.BOOTSTRAP_SELECTION_REPETITIONS", 32
    )
    cases = _selection_cases()
    rng = np.random.default_rng(9)
    strict = rng.binomial(1, 0.2, size=(len(cases), 8)).astype(np.int8)
    labels = {"strict": strict}
    oof = {family: {} for family in MODEL_FAMILIES}
    offset = 0
    for candidate in ("02", "03"):
        count = 20
        base = np.full(count, 0.2)
        for family in MODEL_FAMILIES:
            oof[family][candidate] = {"h10": base, "enhanced": base.copy()}
        offset += count
    result = select_model_family(cases, labels, oof)
    assert not result["passed"]
    assert result["selected_family"] is None


def test_selection_accepts_a_uniquely_stable_predictive_family(monkeypatch):
    monkeypatch.setattr(
        "plastic_heredity.regime_prediction.BOOTSTRAP_SELECTION_REPETITIONS", 32
    )
    cases = _selection_cases()
    rng = np.random.default_rng(11)
    strict = rng.binomial(1, 0.25, size=(len(cases), 8)).astype(np.int8)
    labels = {"strict": strict}
    oof = {family: {} for family in MODEL_FAMILIES}
    offset = 0
    for candidate in ("02", "03"):
        values = strict[offset : offset + 20]
        q = values.mean(axis=1)
        for family in MODEL_FAMILIES:
            oof[family][candidate] = {
                "h10": np.full(20, 0.5),
                "enhanced": (
                    np.clip(q, 0.05, 0.95)
                    if family == "direct_ridge"
                    else np.full(20, 0.5)
                ),
            }
        offset += 20
    result = select_model_family(cases, labels, oof)
    assert result["passed"]
    assert result["selected_family"] == "direct_ridge"
    assert result["bootstrap_unit"].startswith("paired catalytic matrix")


def test_selection_standard_error_pairs_candidates_by_catalytic_matrix(monkeypatch):
    monkeypatch.setattr(
        "plastic_heredity.regime_prediction.BOOTSTRAP_SELECTION_REPETITIONS", 32
    )
    cases = _selection_cases()
    strict = np.ones((len(cases), 8), dtype=np.int8)
    labels = {"strict": strict}
    oof = {family: {} for family in MODEL_FAMILIES}
    first = np.linspace(0.4, 0.8, 10)
    # For y=1, -log(p02) + -log(p03) is constant when p02*p03 is
    # constant. The two candidate losses vary, but their paired matrix mean
    # therefore has zero sampling variation.
    probabilities = {
        "02": np.repeat(first, 2),
        "03": np.repeat(0.32 / first, 2),
    }
    for candidate in ("02", "03"):
        for family in MODEL_FAMILIES:
            oof[family][candidate] = {
                "h10": np.full(20, 0.1),
                "enhanced": probabilities[candidate],
            }
    result = select_model_family(cases, labels, oof)
    row = next(item for item in result["families"] if item["family"] == "direct_ridge")
    assert row["matrix_loss_standard_error"] < 1e-14


def _empty_batch(value: float = 0.1) -> PredictionBranchBatch:
    windows = (
        np.full((1, len(WINDOW_METRIC_NAMES)), value),
        np.empty((0, len(WINDOW_METRIC_NAMES))),
    )
    return PredictionBranchBatch(
        targets=np.zeros((2, 3), dtype=np.int8),
        stages=np.zeros((2, 2), dtype=np.int8),
        onsets=np.full((2, 3), -1, dtype=np.int16),
        completed_horizon=np.ones(2, dtype=np.int8),
        observed_fissions=np.full(2, 12, dtype=np.int16),
        first_break_index=np.full(2, -1, dtype=np.int16),
        first_run8_start=np.full(2, -1, dtype=np.int16),
        longest_run=np.zeros(2, dtype=np.int16),
        window_count=np.asarray((1, 0), dtype=np.int16),
        best_margins=np.full((2, 3), value),
        post_break_features=np.full((2, len(POST_BREAK_FEATURE_NAMES)), np.nan),
        windows=windows,
    )


def test_rich_replay_checks_all_discrete_and_continuous_outputs():
    audit = replay_audit([_empty_batch()], [_empty_batch(0.1 + 5e-15)])
    assert audit["discrete_exact"]
    assert audit["continuous_within_1e-14"]
    changed = _empty_batch()
    changed.window_count[0] = 2
    with pytest.raises(ValueError, match="discrete replay mismatch"):
        replay_audit([_empty_batch()], [changed])
    with pytest.raises(ValueError, match="continuous replay mismatch"):
        replay_audit([_empty_batch()], [_empty_batch(0.1 + 2e-14)])


def test_per_state_checkpoint_resumes_without_recomputing(tmp_path: Path, monkeypatch):
    cases = [SimpleNamespace(state_id="s0"), SimpleNamespace(state_id="s1")]
    checkpoint = tmp_path / "work" / "generate"
    calls = []

    def worker(arguments):
        calls.append(arguments[0].state_id)
        return _empty_batch()

    monkeypatch.setattr("plastic_heredity.regime_prediction._branch_worker", worker)
    experiment = ExperimentConfig.quick()
    first = run_prediction_branches(
        cases,
        experiment,
        branches=2,
        workers=1,
        label="checkpoint-test",
        checkpoint_directory=checkpoint,
    )
    assert len(first) == 2
    assert calls == ["s0", "s1"]

    def must_not_run(arguments):  # pragma: no cover - assertion path
        raise AssertionError("checkpointed state was recomputed")

    monkeypatch.setattr(
        "plastic_heredity.regime_prediction._branch_worker", must_not_run
    )
    second = run_prediction_branches(
        cases,
        experiment,
        branches=2,
        workers=1,
        label="checkpoint-test",
        checkpoint_directory=checkpoint,
    )
    assert len(second) == 2
    status = read_checkpoint_status(tmp_path / "work")
    assert status["stages"]["generate"]["completed_states"] == 2
    assert status["stages"]["generate"]["fraction_complete"] == 1.0


def test_branch_artifact_readback_requires_exact_state_and_branch_order(tmp_path: Path):
    import pandas as pd

    cases = [SimpleNamespace(state_id="s0"), SimpleNamespace(state_id="s1")]
    table = pd.DataFrame(
        {
            "state_id": np.repeat(("s0", "s1"), 2),
            "branch": np.tile(np.arange(2), 2),
            "primary_all8": (0, 1, 1, 0),
        }
    )
    path = tmp_path / "branches.csv.gz"
    table.to_csv(path, index=False)
    restored = _strict_labels_from_branch_table(path, cases, branches=2)
    assert np.array_equal(restored, np.asarray(((0, 1), (1, 0)), dtype=np.int8))
    table.loc[0, "state_id"] = "wrong"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="state order mismatch"):
        _strict_labels_from_branch_table(path, cases, branches=2)


def test_prediction_registration_roundtrip_and_stop_contract(tmp_path: Path):
    destination = tmp_path / "registration"
    register_design(destination)
    payload = verify_design(destination)
    assert payload["status"] == "sealed_before_pilot_matrix_generation"
    assert "plastic_heredity/mechanistic_features.py" in payload["source_hashes"]
    assert "plastic_heredity/metrics.py" in payload["source_hashes"]
    protocol = _protocol()
    endpoint = protocol["endpoint_contract"]
    assert endpoint["required_consecutive_inherited_fissions"] == 8
    assert endpoint["coherence"]["threshold"] == 0.9
    assert endpoint["old_anchor_distinctness"]["threshold"] == 0.85
    assert endpoint["search_every_eligible_window"]
    assert protocol["selection"]["stop_if_no_model_passes"]
    assert protocol["confirmation"]["secondary_can_rescue"] is False
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        register_design(destination)
