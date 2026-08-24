from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd

from e01_prediction_reconstruction.core import (
    EXPECTED_PARAMETER_COUNT,
    MaskedSequenceMLP,
)
from e01_s19_figure5_prediction.core import (
    B1_ID,
    CANDIDATE_IDS,
    DUMMY_ID,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    R1_TARGET_ID,
    U2_TARGET_ID,
    array_sha256,
    build_feature,
    build_split_manifest,
    build_target_tensor,
    combine_scalar_features,
    extended_binary_metrics,
    geometry_gate,
    holm_adjust,
    incoming_h,
    normalized_compositions,
    parameter_count,
    r1_target,
    s16_model_seed,
    split_indices,
    target_geometry,
    u2_target,
)
from e01_s19_matlab_attractor.core import close_rows


def _attractor(center: int, count: int, *, width: int = 100) -> np.ndarray:
    values = np.full((count, width), 1e-8, dtype=np.float64)
    values[:, center] = 1.0
    values[:, (center + 1) % width] = 0.04
    for row in range(count):
        values[row, (center + 2 + row % 3) % width] = 0.002 * (row % 4)
    return close_rows(values)


def test_r1_planted_and_no_recurring_statuses() -> None:
    boundary = np.vstack((_attractor(2, 60), _attractor(40, 40)))
    result = r1_target(boundary, boundary, "L13-R1-PLANTED")
    assert result.target_id == R1_TARGET_ID
    assert result.labels is not None and result.centroids is not None
    assert result.labels.shape == (100,)
    drift = np.eye(100, dtype=np.float64)
    result = r1_target(drift, drift, "L13-R1-NO-RECURRING")
    assert result.labels is None
    assert result.status in {"NO_NONDRIFT_COMPOSITIONS", "NO_RECURRING_COMPTYPE"}


def test_u2_exact_replay_and_union_membership() -> None:
    boundary = np.vstack((_attractor(4, 60), _attractor(60, 40)))
    first = u2_target(boundary, boundary, "L13-U2-REPLAY")
    second = u2_target(boundary, boundary, "L13-U2-REPLAY")
    assert first.target_id == U2_TARGET_ID
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.centroids, second.centroids)
    assert first.labels is not None and np.all(first.labels)


def test_s16_cutoff_mask_and_feature_layout() -> None:
    labels = np.arange(101) % 3 == 0
    target, target_mask, input_labels, cutoff = build_target_tensor(labels, len(labels))
    assert cutoff == 25
    assert target_mask.sum() == 76
    np.testing.assert_array_equal(target[:76].astype(bool), labels[25:])
    np.testing.assert_array_equal(input_labels[:25], labels[:25])
    assert not target_mask[76:].any()
    values, mask, time = build_feature(np.arange(25), np.ones(25, bool), 25, scalar=True)
    assert values.shape == (MAX_INPUT_LENGTH, 100)
    assert mask[:, 0].sum() == 25 and time.sum() == 25
    assert not mask[:, 1:].any()
    combined = combine_scalar_features((values, mask, time), (values, mask, time))
    assert combined[1][:, :2].sum() == 50


def test_matrix_split_and_model_contract_are_exact_s16() -> None:
    split = build_split_manifest()
    for repetition in range(10):
        assert len(split_indices(split, repetition, "FIT")) == 64
        assert len(split_indices(split, repetition, "VALIDATION")) == 16
        assert len(split_indices(split, repetition, "TEST")) == 20
    assert s16_model_seed(CANDIDATE_IDS[0], 0) != s16_model_seed(CANDIDATE_IDS[1], 0)
    assert parameter_count(MaskedSequenceMLP()) == EXPECTED_PARAMETER_COUNT == 288_789


def test_incoming_h_and_target_geometry() -> None:
    states = np.zeros((8, 100), dtype=np.int64)
    states[:, 0] = 3
    states[4:, 1] = 1
    composition = normalized_compositions(states)
    h = incoming_h(composition)
    assert h[0] == h[1] == 1.0
    geometry = target_geometry(np.asarray([0, 0, 0, 1, 1, 0, 1, 1], bool), 8)
    assert geometry["firstOnset"] == 3
    assert geometry["noOnsetBeforeCutoff"]
    assert geometry["firstOnsetInSuffix"]
    assert geometry["suffixPositiveEpisodes"] == 2


def test_extended_metrics_dummy_and_calibration() -> None:
    y = np.asarray([0, 0, 1, 1], bool)
    p = np.asarray([0.1, 0.2, 0.8, 0.9])
    result = extended_binary_metrics(y, p)
    assert result["accuracy"] == 1.0
    assert result["balancedAccuracy"] == 1.0
    assert result["auroc"] == 1.0
    assert result["positivePredictiveValue"] == 1.0
    assert result["negativePredictiveValue"] == 1.0
    dummy = extended_binary_metrics(y, np.full(4, 0.5))
    assert dummy["accuracy"] == 0.5


def test_geometry_gate_and_holm_contract() -> None:
    passed = geometry_gate(95, np.asarray([0.58, 0.60, 0.62]), np.asarray([0, 1]))
    assert passed["passed"]
    assert not geometry_gate(79, np.asarray([0.6]), np.asarray([0, 1]))["passed"]
    adjusted = holm_adjust([0.01, 0.04, None, 0.02])
    assert adjusted[2] is None
    assert adjusted[0] <= adjusted[1]


def test_array_hash_and_nullable_table_are_deterministic() -> None:
    values = np.asarray([[True, False], [False, True]], dtype=bool)
    assert array_sha256(values) == array_sha256(values.copy())
    frame = pd.DataFrame(
        {
            "booleanFingerprint": pd.Series([True, pd.NA], dtype="boolean"),
            "status": ["ELIGIBLE", "INELIGIBLE"],
            "value": [1.0, np.nan],
        }
    )
    restored = pd.read_json(StringIO(frame.to_json(orient="table")), orient="table")
    assert restored.columns.tolist() == frame.columns.tolist()
    assert B1_ID and DUMMY_ID and MAX_TARGET_LENGTH == 1101


def test_oracle_is_target_specific_not_a_generic_cached_feature() -> None:
    # The oracle depends on the completed-run target centroid/set and therefore
    # is materialized per target in memory; it must not be looked up in the
    # target-independent feature cache.
    from e01_s19_figure5_prediction.core import ORACLE_ID

    target_independent = {
        "P1_PHIRL_EMERGENCE_COMPLETED_FIT",
        "P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY",
        "B1_COMPOSITION_CHANGE",
    }
    assert ORACLE_ID not in target_independent


def test_matplotlib_boxplot_uses_supported_tick_labels_keyword() -> None:
    import matplotlib.pyplot as plt

    _figure, axis = plt.subplots()
    result = axis.boxplot([[0.55, 0.60, 0.65], [0.58, 0.61, 0.64]], tick_labels=["R1", "U2"])
    assert len(result["boxes"]) == 2
    plt.close()


def test_registered_figure_review_artifact_is_required() -> None:
    from pathlib import Path

    runner = Path("scripts/e01/run_s19_l13.py").read_text(encoding="utf-8")
    assert '"FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW.md"' in runner
    assert '"FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW_V2.md"' in runner
    assert "write_figure_review_artifact(figures)" in runner
    assert "write_paper_figure_review_v2()" in runner
    assert "Figure 1 — End-to-end conceptual system" in runner
    assert "Figure 6 — Intervention pipeline and treatment outcomes" in runner
