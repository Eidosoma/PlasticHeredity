"""Outcome-blind unit tests for the E01/S19-L01 lock."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from e01_s19_iterative_replication.bundle_a import network_metrics, source_preprocess
from e01_s19_iterative_replication.core import (
    all_pair_mean_distance,
    correlation_inference,
    excursion_episodes,
    expected_parameter_count,
    holm_adjust,
    parameter_count,
    predict_locked_mlp,
    rank_candidate,
    train_locked_mlp,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_rank_formula_and_holm_are_frozen() -> None:
    score = rank_candidate(
        {
            "sourceGrounding": 5,
            "paperFingerprintSpecificity": 5,
            "explanatoryLeverage": 5,
            "testability": 5,
            "crossCandidateDiscriminability": 5,
            "computeEfficiency": 4,
            "independenceFromPriorOutcomeSelection": 5,
        },
        {
            "outcomeGuidedThresholdSelection": 0,
            "deterministicHReuse": 0,
            "completedFitLeakage": 0,
            "candidateSpecificSuccess": 0,
            "undefinedAuthorSemantics": 2,
            "branchCount": 4,
        },
    )
    assert score == 42.0
    assert holm_adjust([0.01, 0.04, None, 0.03]) == [0.03, 0.06, None, 0.06]


def test_correlation_handles_constant_and_replays() -> None:
    constant = correlation_inference(
        np.ones(10),
        np.arange(10, dtype=float),
        method="spearman",
        seed_identity=("test", "constant"),
        bootstrap_replicates=32,
    )
    assert constant["status"] == "UNDEFINED"
    x = np.arange(12, dtype=float)
    y = x**2
    first = correlation_inference(
        x,
        y,
        method="spearman",
        seed_identity=("test", "defined"),
        bootstrap_replicates=64,
        permutation_replicates=64,
    )
    second = correlation_inference(
        x,
        y,
        method="spearman",
        seed_identity=("test", "defined"),
        bootstrap_replicates=64,
        permutation_replicates=64,
    )
    assert first == second
    assert first["statistic"] == 1.0


def test_spike_episode_and_all_pair_spacing_contract() -> None:
    values = np.array([0.0, 4.0, 5.0, 0.0, 6.0, 6.0, 0.0])
    episodes = excursion_episodes(values, 3.0)
    assert [(item.start, item.end, item.peak_position) for item in episodes] == [
        (1, 2, 2),
        (4, 5, 4),
    ]
    assert all_pair_mean_distance(np.array([1.0, 4.0, 10.0])) == 6.0
    assert all_pair_mean_distance(np.array([1.0])) is None


def test_complete_positive_graph_exposes_constant_primary_means() -> None:
    beta = np.exp(np.arange(10_000, dtype=float).reshape(100, 100) / 100_000.0)
    rows = pd.DataFrame(network_metrics(beta, "A_GRAPH_UNWEIGHTED_POSITIVE_SUPPORT"))
    values = dict(zip(rows["metricId"] + ":" + rows["summaryId"], rows["value"]))
    assert values["C001_NUMBER_OF_NODES:PRIMARY"] == 100.0
    assert values["C002_NUMBER_OF_EDGES:PRIMARY"] == 10_000.0
    assert values["C003_IN_DEGREE:PRIMARY_MEAN"] == 100.0
    assert values["C004_OUT_DEGREE:PRIMARY_MEAN"] == 100.0
    assert values["C005_BETWEENNESS:PRIMARY_MEAN"] == 0.0
    assert np.isclose(values["C006_PAGERANK:PRIMARY_MEAN"], 0.01)


def test_source_preprocessing_shapes_are_locked() -> None:
    rng = np.random.default_rng(4)
    values = rng.random((251, 100))
    direct = source_preprocess(values, "A_DYNAMICS_DIRECT_SELECTED_CLOCK")
    windowed = source_preprocess(values, "A_DYNAMICS_SOURCE_WINDOW100")
    assert direct.shape == (251, 100)
    assert windowed.shape == (3, 100)
    assert np.all(np.isfinite(direct)) and np.all(np.isfinite(windowed))


def test_variable_shape_model_replays_and_25_percent_count_is_exact() -> None:
    assert expected_parameter_count(367, 1101) == 288_789
    rng = np.random.default_rng(17)
    fit_values = rng.normal(size=(4, 3, 100))
    fit_channel = np.ones_like(fit_values, dtype=bool)
    fit_time = np.ones((4, 3), dtype=bool)
    fit_target = rng.integers(0, 2, size=(4, 4)).astype(float)
    fit_target_mask = np.ones_like(fit_target, dtype=bool)
    validation_values = rng.normal(size=(2, 3, 100))
    validation_channel = np.ones_like(validation_values, dtype=bool)
    validation_time = np.ones((2, 3), dtype=bool)
    validation_target = rng.integers(0, 2, size=(2, 4)).astype(float)
    validation_target_mask = np.ones_like(validation_target, dtype=bool)
    kwargs = dict(
        fit_values=fit_values,
        fit_channel_mask=fit_channel,
        fit_time_mask=fit_time,
        fit_targets=fit_target,
        fit_target_mask=fit_target_mask,
        validation_values=validation_values,
        validation_channel_mask=validation_channel,
        validation_time_mask=validation_time,
        validation_targets=validation_target,
        validation_target_mask=validation_target_mask,
        model_seed=123,
        maximum_epochs=3,
        patience=2,
    )
    first = train_locked_mlp(**kwargs)
    second = train_locked_mlp(**kwargs)
    assert parameter_count(first.model) == expected_parameter_count(3, 4)
    assert first.history.equals(second.history)
    p1 = predict_locked_mlp(first.model, validation_values, validation_channel, validation_time)
    p2 = predict_locked_mlp(second.model, validation_values, validation_channel, validation_time)
    assert np.array_equal(p1, p2)


def test_preregistration_has_exact_bundles_and_specification_ceiling() -> None:
    prereg = yaml.safe_load(
        (REPO_ROOT / "configs/e01/s19_l01_preregistration.yaml").read_text(encoding="utf-8")
    )
    assert prereg["outcomeAccessedAtLock"] is False
    assert len(prereg["bundles"]) == 3
    assert all(len(bundle["specifications"]) <= 8 for bundle in prereg["bundles"])
    assert prereg["bundles"][1]["proportions"] == [0.10, 0.20, 0.25, 0.33, 0.50]
