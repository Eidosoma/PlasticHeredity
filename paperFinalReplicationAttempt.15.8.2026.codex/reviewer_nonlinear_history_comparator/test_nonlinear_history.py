from __future__ import annotations

import numpy as np

from nonlinear_core import (
    PCA_COMPONENTS,
    SplineInteractionTransformer,
    bootstrap_interval,
    fit_boosted_history,
    fit_capacity_matched_ridge,
    grouped_development_cv,
    holm_adjust,
    predict_boosted_history,
    predict_capacity_matched_ridge,
    sign_randomization_p,
    state_log_loss,
)


def synthetic(seed: int = 7, matrices: int = 20, rows_per_matrix: int = 5):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(matrices), rows_per_matrix)
    x = rng.normal(size=(matrices * rows_per_matrix, 9))
    logits = 0.7 * x[:, 0] ** 2 - 0.5 * x[:, 1] * x[:, 2] + 0.2 * x[:, 3]
    probability = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probability[:, None], size=(len(x), 4)).astype(np.int8)
    return x, y, groups


def test_spline_transform_is_deterministic_and_twelve_dimensional():
    x, _, _ = synthetic()
    first = SplineInteractionTransformer().fit(x)
    second = SplineInteractionTransformer().fit(x)
    a = first.transform(x)
    b = second.transform(x)
    assert a.shape == (len(x), PCA_COMPONENTS)
    assert np.allclose(a, b, atol=1e-12, rtol=0.0)
    assert first.audit()["library_dimension_after_filter"] >= PCA_COMPONENTS


def test_capacity_matched_ridge_handles_branch_targets():
    x, y, _ = synthetic()
    bundle = fit_capacity_matched_ridge(x, y, "codex")
    probability = predict_capacity_matched_ridge(bundle, x)
    assert probability.shape == (len(x),)
    assert np.isfinite(probability).all()
    assert (probability > 0).all() and (probability < 1).all()


def test_capacity_matched_ridge_handles_single_targets():
    x, y, _ = synthetic()
    bundle = fit_capacity_matched_ridge(x, y[:, 0], "fable")
    probability = predict_capacity_matched_ridge(bundle, x)
    assert probability.shape == (len(x),)


def test_boosted_model_and_state_loss():
    x, y, _ = synthetic()
    model = fit_boosted_history(x, y, max_leaf_nodes=3, seed=11)
    probability = predict_boosted_history(model, x)
    loss = state_log_loss(probability, y)
    assert loss.shape == (len(x),)
    assert np.isfinite(loss).all()


def test_grouped_development_selection_returns_one_family():
    x, y, groups = synthetic(matrices=10)
    records, summary = grouped_development_cv(x, y, groups, "codex", "test", "02")
    assert len(records) == 5 * 4
    assert summary["selected_family"] in {
        "spline_interaction_pca12_ridge",
        "gradient_boosted_history",
    }
    assert summary["selected_tree_leaves"] in {3, 7, 15}


def test_cluster_inference_and_holm_are_reproducible():
    values = np.linspace(-0.2, 0.4, 20)
    groups = np.repeat(np.arange(10), 2)
    assert bootstrap_interval(values, groups, 3) == bootstrap_interval(values, groups, 3)
    assert sign_randomization_p(values, groups, 4) == sign_randomization_p(values, groups, 4)
    adjusted = holm_adjust([0.01, 0.03, 0.02])
    assert all(0 <= value <= 1 for value in adjusted)

