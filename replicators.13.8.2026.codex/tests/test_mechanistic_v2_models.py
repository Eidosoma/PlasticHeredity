import numpy as np
from scipy.special import expit

from plastic_heredity.mechanistic_v2_models import (
    LinearFit,
    _objective_and_gradient,
    fit_block_transform,
    matrix_cv_fold,
)


def test_v2_offset_objective_gradient_matches_finite_difference():
    rng = np.random.default_rng(711)
    design = rng.normal(size=(23, 6))
    trials = np.full(23, 7.0)
    successes = rng.binomial(7, 0.4, size=23).astype(float)
    offset = rng.normal(scale=0.2, size=23)
    parameters = rng.normal(scale=0.1, size=7)
    _, analytic = _objective_and_gradient(
        parameters, design, successes, trials, offset, 10.0
    )
    numerical = np.empty_like(parameters)
    epsilon = 1e-6
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = (
            _objective_and_gradient(plus, design, successes, trials, offset, 10.0)[0]
            - _objective_and_gradient(minus, design, successes, trials, offset, 10.0)[0]
        ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=3e-6, atol=3e-6)


def test_zero_offset_stage_reproduces_baseline_exactly():
    rng = np.random.default_rng(712)
    baseline = rng.normal(size=31)
    block = rng.normal(size=(31, 9))
    zero = LinearFit(
        name="zero",
        block="added",
        coefficient=np.zeros(9),
        intercept=0.0,
        ridge_lambda=1.0,
        objective=0.0,
        gradient_max_abs=0.0,
        iterations=0,
    )
    np.testing.assert_array_equal(expit(baseline), expit(baseline + zero.correction(block)))


def test_no_pca_transform_retains_every_unique_nonconstant_direction():
    x = np.linspace(-2.0, 2.0, 51)
    values = np.column_stack((x, 3 * x + 8, np.ones_like(x), x**2, x**3))
    transform = fit_block_transform(
        "example", values, ("x", "copy", "constant", "square", "cube")
    )
    assert transform.output_names == ("x", "square", "cube")
    assert transform.output_features == 3
    assert transform.residual_coefficient is None


def test_cv_assignment_is_deterministic_and_never_splits_a_matrix():
    matrix_ids = np.repeat(np.arange(20), 5)
    first = matrix_cv_fold(matrix_ids)
    second = matrix_cv_fold(matrix_ids)
    np.testing.assert_array_equal(first, second)
    for matrix_id in np.unique(matrix_ids):
        assert np.unique(first[matrix_ids == matrix_id]).size == 1
    assert set(first) == set(range(5))

