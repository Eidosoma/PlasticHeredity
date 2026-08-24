from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from e01_s19_padding_length_discrimination.core import (
    B1,
    FIT_COUNT,
    MATRIX_COUNT,
    S00,
    S01,
    S10,
    S11,
    TEST_COUNT,
    VALIDATION_COUNT,
    accuracy_decomposition,
    build_split_manifest,
    incoming_h,
    infer_output_length,
    mask_pair,
    normalized_compositions,
    padding_identity,
    seed128,
)

ROOT = "acfa3303704ae10b337193136b44a2986a02ab148c3fb56541490292faf521c2"


def test_preregistration_yaml_is_parseable_and_complete() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/e01/s19_l15_untouched_padding_panel.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert len(config["diagnostics"]["models"]) == 4
    assert len(config["negativeControls"]["controls"]) == 6
    assert config["preoutcomeTechnicalAmendments"][0]["outcomeAccessed"] is False


def test_split_manifest_is_paired_matrix_level_and_unique() -> None:
    frame = build_split_manifest(ROOT)
    assert len(frame) == 10 * MATRIX_COUNT
    for _, group in frame.groupby("repetitionId"):
        assert group["splitRole"].value_counts().to_dict() == {
            "FIT": FIT_COUNT,
            "TEST": TEST_COUNT,
            "VALIDATION": VALIDATION_COUNT,
        }
    assert (
        len(
            {
                tuple(group.loc[group["splitRole"].eq("TEST"), "matrixIndex"])
                for _, group in frame.groupby("repetitionId")
            }
        )
        == 10
    )


def test_seed_is_domain_separated_and_stable() -> None:
    assert seed128(ROOT, "a", 1) == seed128(ROOT, "a", 1)
    assert seed128(ROOT, "a", 1) != seed128(ROOT, "a", 2)
    assert seed128(ROOT, "a", 1) != seed128(ROOT, "b", 1)


def test_adjacent_h_contract() -> None:
    states = np.array(
        [[1, 0, 1] + [0] * 97, [2, 0, 2] + [0] * 97, [0, 1, 1] + [0] * 97]
    )
    compositions = normalized_compositions(states)
    values = incoming_h(compositions)
    assert values.shape == (3,)
    assert np.allclose(values[:2], 1.0, rtol=0.0, atol=2e-15)
    assert np.isclose(values[2], 0.5, rtol=0.0, atol=2e-15)


def test_four_mask_semantics() -> None:
    valid = np.array([[True, True, False]])
    assert np.array_equal(mask_pair(valid, S00)[0], valid)
    assert np.array_equal(mask_pair(valid, S00)[1], valid)
    assert np.array_equal(mask_pair(valid, S01)[0], valid)
    assert mask_pair(valid, S01)[1].all()
    assert mask_pair(valid, S10)[0].all()
    assert np.array_equal(mask_pair(valid, S10)[1], valid)
    assert mask_pair(valid, S11)[0].all() and mask_pair(valid, S11)[1].all()


def test_padding_identity_and_accuracy_decomposition() -> None:
    target = np.array([[True, True, False, False], [True, False, False, False]])
    valid = np.array([[True, True, True, False], [True, True, False, False]])
    probability = np.array([[0.9, 0.9, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1]])
    p_valid = float(target[valid].mean())
    q = float(valid.mean())
    assert padding_identity(p_valid, q) == float(target.mean())
    result = accuracy_decomposition(target, probability, valid)
    assert result["allCellAccuracy"] == 1.0
    assert result["absoluteError"] <= 1e-15


def test_boundary_midpoint_rule() -> None:
    cutoff = np.array([0, 1, 10, 367])
    assert np.array_equal(infer_output_length(cutoff), np.array([2, 5, 32, 1103]))


def test_registered_feature_constant_imports() -> None:
    assert B1 == "B1_COMPOSITION_CHANGE"
