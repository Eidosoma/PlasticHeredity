import numpy as np

from plastic_heredity.mechanistic_models import (
    _binomial_objective_and_gradient,
    fit_block_transform,
    fit_registered_model,
)


def test_block_transform_removes_constant_and_affine_duplicate_columns():
    x = np.linspace(-2.0, 2.0, 41)
    values = np.column_stack((x, 2.0 * x + 9.0, np.ones_like(x), x**2))
    transform = fit_block_transform(
        "example", values, ("x", "affine_x", "constant", "quadratic"), None
    )
    assert transform.kept_names == ("x", "quadratic")
    assert transform.dropped["constant"] == "zero_variance"
    assert transform.dropped["affine_x"].startswith("affine_duplicate_of")
    assert transform.transform(values).shape == (41, 2)


def test_binomial_objective_analytic_gradient_matches_finite_difference():
    rng = np.random.default_rng(301)
    design = rng.normal(size=(17, 4))
    trials = np.full(17, 8.0)
    successes = rng.binomial(8, 0.45, size=17).astype(float)
    parameters = rng.normal(scale=0.2, size=5)
    mask = np.asarray((False, True, False, True))
    _, analytic = _binomial_objective_and_gradient(
        parameters, design, successes, trials, mask, 0.1
    )
    numerical = np.empty_like(parameters)
    epsilon = 1e-6
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_value = _binomial_objective_and_gradient(
            plus, design, successes, trials, mask, 0.1
        )[0]
        minus_value = _binomial_objective_and_gradient(
            minus, design, successes, trials, mask, 0.1
        )[0]
        numerical[index] = (plus_value - minus_value) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-6, atol=2e-6)


def test_unpenalized_common_block_neutralizes_duplicate_penalty_artifact():
    rng = np.random.default_rng(302)
    common = rng.normal(size=(240, 2))
    probability = 1.0 / (1.0 + np.exp(-(0.3 + 1.4 * common[:, 0] - common[:, 1])))
    trials = np.full(240, 6.0)
    successes = rng.binomial(6, probability).astype(float)
    blocks = {"h": common, "d": common.copy()}
    baseline = fit_registered_model(
        "baseline", ("h",), frozenset(), blocks, successes, trials, 0.1
    )
    corrected = fit_registered_model(
        "corrected", ("h", "d"), frozenset(("d",)), blocks, successes, trials, 0.1
    )
    ridge = fit_registered_model(
        "ridge", ("h",), frozenset(("h",)), blocks, successes, trials, 0.1
    )
    ridge_duplicate = fit_registered_model(
        "ridge_duplicate",
        ("h", "d"),
        frozenset(("h", "d")),
        blocks,
        successes,
        trials,
        0.1,
    )
    np.testing.assert_allclose(
        baseline.predict(blocks), corrected.predict(blocks), atol=2e-7, rtol=0
    )
    assert np.max(np.abs(ridge.predict(blocks) - ridge_duplicate.predict(blocks))) > 1e-5
