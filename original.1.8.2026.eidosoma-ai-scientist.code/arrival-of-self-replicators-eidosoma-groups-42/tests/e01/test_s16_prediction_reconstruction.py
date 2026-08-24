from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from e01_prediction_reconstruction.core import (
    CUTOFF_MODE,
    EXPECTED_PARAMETER_COUNT,
    FEATURE_IDS,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    MaskedSequenceMLP,
    apply_channel_scaler,
    binary_metrics,
    build_split_manifest,
    fit_channel_scaler,
    parameter_count,
    predict_probabilities,
    preonset_masks,
    train_masked_mlp,
)

ROOT = Path(__file__).resolve().parents[2]


def test_preregistration_locks_one_layout_two_modes_and_exact_feature_set() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/e01/s16_first_quarter_prediction_preregistration.yaml").read_text()
    )
    assert config["scope"]["onlyActiveStep"] == "S16"
    assert config["scope"]["architectureTournamentPermitted"] is False
    assert [item["modeId"] for item in config["temporalModes"]] == [
        "RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE",
        CUTOFF_MODE,
    ]
    assert tuple(item["featureId"] for item in config["featureFamilies"]) == FEATURE_IDS
    assert config["tensorLayout"]["maximumInputLength"] == MAX_INPUT_LENGTH
    assert config["tensorLayout"]["maximumTargetLength"] == MAX_TARGET_LENGTH
    assert config["tensorLayout"]["resamplingOrInterpolation"] == "none"
    assert config["tensorLayout"]["targetMaskIsNeverModelInput"] is True


def test_generated_split_manifest_is_exact_paired_and_outcome_blind() -> None:
    expected = build_split_manifest()
    observed = pd.read_csv(ROOT / "configs/e01/s16_split_manifest.csv")
    pd.testing.assert_frame_equal(observed, expected, check_dtype=False)
    assert len(observed) == 1_000
    counts = observed.groupby(["repetitionId", "splitRole"]).size().unstack()
    assert counts["FIT"].eq(64).all()
    assert counts["VALIDATION"].eq(16).all()
    assert counts["TEST"].eq(20).all()
    assert not observed["outcomeStratified"].astype(bool).any()


def test_tensor_manifest_records_preoutcome_status_and_parameter_identity() -> None:
    payload = json.loads(
        (ROOT / "configs/e01/s16_tensor_model_manifest.json").read_text()
    )
    assert payload["predictionOutcomeAccessed"] is False
    assert payload["observedSchemaOnly"]["primaryCandidateMatrixUnits"] == 200
    assert payload["observedSchemaOnly"]["maximumT"] == 1468
    assert payload["observedSchemaOnly"]["trainableParameterCount"] == EXPECTED_PARAMETER_COUNT
    model = MaskedSequenceMLP().to(dtype=torch.float64)
    assert parameter_count(model) == EXPECTED_PARAMETER_COUNT
    values = torch.zeros((2, MAX_INPUT_LENGTH, 100), dtype=torch.float64)
    channel_mask = torch.zeros_like(values)
    time_mask = torch.zeros((2, MAX_INPUT_LENGTH), dtype=torch.float64)
    assert model(values, channel_mask, time_mask).shape == (2, MAX_TARGET_LENGTH)


def test_scaler_uses_only_true_cells_and_restores_masked_zero() -> None:
    values = np.zeros((2, 3, 100), dtype=np.float64)
    mask = np.zeros_like(values, dtype=bool)
    values[:, :, 0] = [[1.0, 2.0, 999.0], [3.0, 4.0, -999.0]]
    mask[:, :2, 0] = True
    scaler = fit_channel_scaler(values, mask)
    np.testing.assert_allclose(scaler.mean[0], 2.5)
    np.testing.assert_allclose(scaler.scale[0], np.sqrt(1.25))
    transformed = apply_channel_scaler(values, mask, scaler)
    assert np.all(transformed[~mask] == 0.0)
    assert scaler.valid_count[0] == 4
    assert np.all(scaler.valid_count[1:] == 0)


def test_metric_and_preonset_contracts() -> None:
    target = np.array([False, True, True, False])
    probability = np.array([0.1, 0.9, 0.8, 0.2])
    metrics = binary_metrics(target, probability)
    assert metrics["accuracy"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["balancedAccuracy"] == 1.0
    input_labels = np.array([[False, False], [False, True]])
    target_labels = np.array([[False, False, True, True], [False, True, True, True]])
    target_mask = np.ones_like(target_labels, dtype=bool)
    eligible, risk = preonset_masks(input_labels, target_labels, target_mask)
    assert eligible.tolist() == [True, False]
    assert risk[0].tolist() == [True, True, True, False]
    assert not risk[1].any()


def test_tiny_training_replay_is_exact() -> None:
    rng = np.random.default_rng(10)
    n_fit, n_validation = 4, 2
    fit_values = np.zeros((n_fit, MAX_INPUT_LENGTH, 100), dtype=np.float64)
    validation_values = np.zeros((n_validation, MAX_INPUT_LENGTH, 100), dtype=np.float64)
    fit_values[:, :3, 0] = rng.normal(size=(n_fit, 3))
    validation_values[:, :3, 0] = rng.normal(size=(n_validation, 3))
    fit_channel = np.zeros_like(fit_values, dtype=bool)
    validation_channel = np.zeros_like(validation_values, dtype=bool)
    fit_channel[:, :3, 0] = True
    validation_channel[:, :3, 0] = True
    fit_time = fit_channel.any(axis=2)
    validation_time = validation_channel.any(axis=2)
    fit_target = np.zeros((n_fit, MAX_TARGET_LENGTH), dtype=np.float64)
    validation_target = np.zeros((n_validation, MAX_TARGET_LENGTH), dtype=np.float64)
    fit_target[:, :4] = rng.integers(0, 2, size=(n_fit, 4))
    validation_target[:, :4] = rng.integers(0, 2, size=(n_validation, 4))
    fit_target_mask = np.zeros_like(fit_target, dtype=bool)
    validation_target_mask = np.zeros_like(validation_target, dtype=bool)
    fit_target_mask[:, :4] = True
    validation_target_mask[:, :4] = True
    kwargs = {
        "fit_values": fit_values,
        "fit_channel_mask": fit_channel,
        "fit_time_mask": fit_time,
        "fit_targets": fit_target,
        "fit_target_mask": fit_target_mask,
        "validation_values": validation_values,
        "validation_channel_mask": validation_channel,
        "validation_time_mask": validation_time,
        "validation_targets": validation_target,
        "validation_target_mask": validation_target_mask,
        "model_seed": 12345,
        "maximum_epochs": 3,
        "patience": 2,
    }
    first = train_masked_mlp(**kwargs)
    second = train_masked_mlp(**kwargs)
    pd.testing.assert_frame_equal(first.history, second.history)
    first_probability = predict_probabilities(
        first.model, validation_values, validation_channel, validation_time
    )
    second_probability = predict_probabilities(
        second.model, validation_values, validation_channel, validation_time
    )
    np.testing.assert_array_equal(first_probability, second_probability)
