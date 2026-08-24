from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from e01_s19_padding_leakage.core import (
    MASK_CONDITIONS,
    S00,
    S01,
    S10,
    S11,
    accuracy_decomposition,
    boundary_predictions,
    included_training_prevalence,
    loss_mask,
    obfuscate_padded_input_values,
    padded_target,
    padding_arithmetic,
    paper_interval,
    permute_valid_labels_preserving_padding,
    permute_valid_time,
    pixel_to_accuracy,
    score_mask,
)

ROOT = Path(__file__).resolve().parents[2]


def test_lock_has_exact_scope_and_four_conditions() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/e01/s19_l14_padding_leakage.yaml").read_text()
    )
    assert config["versionedStepId"].startswith("E01-S19-L14-")
    assert config["primaryTarget"]["targetId"] == "S16_ADJACENT_INCOMING_H090"
    assert (
        tuple(item["conditionId"] for item in config["maskConditions"])
        == MASK_CONDITIONS
    )
    assert len(config["features"]["learned"]) == 6
    assert config["resources"]["gpuHours"] == 0


def test_variable_length_padding_arithmetic_and_no_padding_identity() -> None:
    target, mask = padded_target(
        [
            np.array([1, 1], dtype=bool),
            np.array([1, 0, 1], dtype=bool),
            np.array([0, 1, 1, 1], dtype=bool),
        ],
        width=4,
    )
    result = padding_arithmetic(target, mask)
    assert result["validCellCount"] == 9
    assert result["allCellCount"] == 12
    assert result["identityAbsoluteError"] <= 1e-15
    target_equal, mask_equal = padded_target(
        [np.array([1, 0]), np.array([0, 1])], width=2
    )
    equal = padding_arithmetic(target_equal, mask_equal)
    assert equal["validPrevalence"] == equal["paddedPrevalence"]


def test_four_mask_semantics_and_training_prevalence() -> None:
    valid = np.array([[True, True, False], [True, False, False]])
    target = np.array([[1, 1, 0], [1, 0, 0]], dtype=float)
    assert np.array_equal(loss_mask(valid, S00), valid)
    assert np.array_equal(loss_mask(valid, S01), valid)
    assert loss_mask(valid, S10).all()
    assert loss_mask(valid, S11).all()
    assert np.array_equal(score_mask(valid, S00), valid)
    assert score_mask(valid, S01).all()
    assert np.array_equal(score_mask(valid, S10), valid)
    assert score_mask(valid, S11).all()
    assert included_training_prevalence(target, valid, S00) == 1.0
    assert included_training_prevalence(target, valid, S11) == 0.5


def test_accuracy_decomposition_is_exact() -> None:
    target = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=bool)
    valid = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)
    probability = np.array([[0.9, 0.8, 0.7, 0.1], [0.8, 0.1, 0.2, 0.2]])
    result = accuracy_decomposition(target, probability, valid)
    assert result["absoluteError"] <= 1e-12
    assert result["reconstructedAllCellAccuracy"] == result["allCellAccuracy"]


def test_pixel_calibration_and_intervals() -> None:
    rows = np.array([64, 129, 194, 259, 324])
    values = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    assert abs(pixel_to_accuracy(194, rows, values) - 0.8) < 1e-12
    lower, upper = paper_interval(318, 2, rows, values)
    assert lower < pixel_to_accuracy(318, rows, values) < upper


def test_padding_boundary_rule_uses_cutoff_only() -> None:
    cutoff = np.array([2, 3])
    prediction = boundary_predictions(cutoff, 12, True)
    assert prediction[0, :8].all() and not prediction[0, 8:].any()
    assert prediction[1, :11].all() and not prediction[1, 11:].any()


def test_registered_negative_control_transformations() -> None:
    target = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    valid = np.array([[1, 1, 0], [1, 1, 0]], dtype=bool)
    shuffled = permute_valid_labels_preserving_padding(
        target, valid, seed_identity=("fixture",)
    )
    assert sorted(shuffled[valid].tolist()) == sorted(target[valid].tolist())
    assert np.all(shuffled[~valid] == 0)
    values = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    channel = np.ones_like(values, dtype=bool)
    time = np.array([[1, 1, 0], [1, 1, 1]], dtype=bool)
    permuted, permuted_channel = permute_valid_time(
        values, channel, time, seed_identity=("fixture",)
    )
    for index in range(2):
        positions = np.flatnonzero(time[index])
        assert sorted(permuted[index, positions, 0]) == sorted(
            values[index, positions, 0]
        )
    assert np.array_equal(permuted_channel, channel)
    obfuscated = obfuscate_padded_input_values(values, time, seed_identity=("fixture",))
    pad = np.broadcast_to(~time[:, :, None], values.shape)
    assert np.array_equal(obfuscated[~pad], values[~pad])
    assert np.any(obfuscated[pad] != values[pad])
