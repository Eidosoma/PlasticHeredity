from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

from e01_association_replicator_state.core import (
    CHANGE_ANALYSIS,
    HISTORICAL_BRANCH,
    LEVEL_ANALYSIS,
    PREFIX_BRANCH,
    PRIMARY_BRANCH,
    all_cyclic_shift_metrics,
    circular_shift_control,
    prepare_analysis_rows,
    runwise_statistics,
    summarize_correlations,
    trajectory_bootstrap,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    full_rows = []
    prefix_rows = []
    for candidate in ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03"):
        for matrix_index in range(3):
            trajectory_id = f"{candidate}-M{matrix_index:03d}"
            values = np.array([0.0, 1.0, 3.0, 6.0, 10.0]) + matrix_index
            labels = np.array([False, False, True, True, True])
            for index, (value, label) in enumerate(zip(values, labels), start=1):
                h = 0.95 if label else 0.8
                full_rows.append(
                    {
                        "candidateId": candidate,
                        "trajectoryId": trajectory_id,
                        "matrixIndex": matrix_index,
                        "selectedSequenceIndex": index,
                        "rawObservationIndex": index,
                        "status": "ELIGIBLE",
                        "emergence": value,
                        "incomingCosineH": h,
                        "euclideanL2ClosedCompositionChange": 1.0 - h,
                        "molecularH090Label": label,
                        "historicalH090Label": not label,
                    }
                )
                prefix_rows.append(
                    {
                        "candidateId": candidate,
                        "trajectoryId": trajectory_id,
                        "matrixIndex": matrix_index,
                        "endpointSelectedSequenceIndex": index,
                        "endpointRawObservationIndex": index,
                        "status": "ELIGIBLE",
                        "emergence": -value,
                        "currentIncomingCosineH": h,
                        "currentMolecularH090Label": label,
                    }
                )
    return pd.DataFrame(full_rows), pd.DataFrame(prefix_rows)


def test_preregistration_is_parseable_and_locks_both_analyses() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs/e01/s15_association_replicator_state_preregistration.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    assert config["noOutcomeBasedReplacement"] is True
    assert [item["analysisId"] for item in config["analyses"]] == [
        LEVEL_ANALYSIS,
        CHANGE_ANALYSIS,
    ]


def test_level_and_change_are_both_materialized_with_current_label_alignment() -> None:
    full, prefix = _inputs()
    rows = prepare_analysis_rows(full, prefix)
    assert set(rows["branchId"]) == {
        PRIMARY_BRANCH,
        HISTORICAL_BRANCH,
        PREFIX_BRANCH,
    }
    assert set(rows["analysisId"]) == {LEVEL_ANALYSIS, CHANGE_ANALYSIS}
    primary = rows.loc[
        rows["branchId"].eq(PRIMARY_BRANCH)
        & rows["candidateId"].eq("S12F-CANDIDATE-02")
        & rows["matrixIndex"].eq(0)
    ]
    level = primary.loc[primary["analysisId"].eq(LEVEL_ANALYSIS)]
    change = primary.loc[primary["analysisId"].eq(CHANGE_ANALYSIS)]
    assert level["analysisValue"].tolist() == [0.0, 1.0, 3.0, 6.0, 10.0]
    assert change["analysisValue"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert change["label"].tolist() == [False, True, True, True]
    assert change["previousObservationOrder"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_runwise_statistics_keep_candidate_and_analysis_separate() -> None:
    full, prefix = _inputs()
    rows = prepare_analysis_rows(full, prefix)
    correlations, states = runwise_statistics(rows)
    assert len(correlations) == 3 * 2 * 3 * 2
    assert len(states) == len(correlations)
    primary_level = correlations.loc[
        correlations["branchId"].eq(PRIMARY_BRANCH)
        & correlations["analysisId"].eq(LEVEL_ANALYSIS)
    ]
    assert len(primary_level) == 6
    assert primary_level["spearmanRho"].gt(0).all()
    summary = summarize_correlations(correlations)
    candidate = summary.loc[
        summary["branchId"].eq(PRIMARY_BRANCH)
        & summary["analysisId"].eq(LEVEL_ANALYSIS)
        & summary["candidateScope"].eq("S12F-CANDIDATE-02")
    ].iloc[0]
    pooled = summary.loc[
        summary["branchId"].eq(PRIMARY_BRANCH)
        & summary["analysisId"].eq(LEVEL_ANALYSIS)
        & summary["candidateScope"].eq("POOLED_SECONDARY")
    ].iloc[0]
    assert candidate["evidenceRole"] == "CANDIDATE_SPECIFIC_PRIMARY"
    assert candidate["trajectoryCount"] == 3
    assert pooled["evidenceRole"] == "POOLED_SECONDARY_ONLY"
    assert pooled["trajectoryCount"] == 6


def test_fft_cyclic_metrics_equal_direct_rotation() -> None:
    values = np.array([0.5, -1.0, 3.0, 2.0, 8.0, -4.0, 0.25])
    labels = np.array([False, True, True, False, False, True, False])
    actual = all_cyclic_shift_metrics(values, labels)
    for shift in range(len(values)):
        rotated = np.roll(labels, shift)
        expected_spearman = spearmanr(values, rotated).statistic
        expected_pearson = pearsonr(values, rotated).statistic
        expected_difference = np.mean(values[rotated]) - np.mean(values[~rotated])
        np.testing.assert_allclose(actual["spearman"][shift], expected_spearman)
        np.testing.assert_allclose(actual["pearson"][shift], expected_pearson)
        np.testing.assert_allclose(
            actual["meanDifference"][shift], expected_difference, atol=1e-12
        )


def test_resampling_and_shift_replay_are_exact() -> None:
    full, prefix = _inputs()
    rows = prepare_analysis_rows(full, prefix)
    correlations, states = runwise_statistics(rows)
    first_distribution, first_summary = trajectory_bootstrap(
        correlations,
        states,
        replicates=32,
        seed_root_hex="8a8e32f1c848880425008c43a8f6bd1198758f36a241c7d27fc1ddcb599896b7",
    )
    second_distribution, second_summary = trajectory_bootstrap(
        correlations,
        states,
        replicates=32,
        seed_root_hex="8a8e32f1c848880425008c43a8f6bd1198758f36a241c7d27fc1ddcb599896b7",
    )
    pd.testing.assert_frame_equal(first_distribution, second_distribution)
    pd.testing.assert_frame_equal(first_summary, second_summary)
    shift_one, shift_summary_one = circular_shift_control(
        rows,
        replicates=32,
        seed_root_hex="8a8e32f1c848880425008c43a8f6bd1198758f36a241c7d27fc1ddcb599896b7",
    )
    shift_two, shift_summary_two = circular_shift_control(
        rows,
        replicates=32,
        seed_root_hex="8a8e32f1c848880425008c43a8f6bd1198758f36a241c7d27fc1ddcb599896b7",
    )
    pd.testing.assert_frame_equal(shift_one, shift_two)
    pd.testing.assert_frame_equal(shift_summary_one, shift_summary_two)
