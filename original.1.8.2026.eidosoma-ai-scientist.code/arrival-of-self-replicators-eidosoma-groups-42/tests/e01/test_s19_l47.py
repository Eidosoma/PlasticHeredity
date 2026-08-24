from __future__ import annotations

import numpy as np

from e01_onset_discovery.functional_coherence_sufficiency import (
    fit_ridge,
    predict_ridge,
    regression_metrics,
)


def test_ridge_exact_replay_and_train_only_imputation() -> None:
    x = np.asarray([[0.0, 1.0], [1.0, np.nan], [2.0, 3.0], [3.0, 4.0]])
    y = np.asarray([1.0, 2.0, 3.0, 4.0])
    first = fit_ridge(x, y, alpha=1.0)
    second = fit_ridge(x, y, alpha=1.0)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(predict_ridge(first, x), predict_ridge(second, x))
    assert first.medians[1] == 3.0


def test_ridge_feature_permutation_equivalence() -> None:
    rng = np.random.default_rng(47)
    x = rng.normal(size=(30, 4))
    y = x @ np.asarray([0.2, -0.4, 0.1, 0.5])
    order = np.asarray([2, 0, 3, 1])
    original = fit_ridge(x, y)
    permuted = fit_ridge(x[:, order], y)
    np.testing.assert_allclose(
        predict_ridge(original, x), predict_ridge(permuted, x[:, order]), atol=1e-12
    )


def test_regression_metrics() -> None:
    result = regression_metrics(np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0]))
    assert result["rmse"] == 0.0
    assert result["rSquared"] == 1.0
    assert result["residualMean"] == 0.0
