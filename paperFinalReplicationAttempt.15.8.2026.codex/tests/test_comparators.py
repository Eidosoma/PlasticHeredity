import numpy as np
import pandas as pd

from aor_replication.comparators import (
    correlation_dimension,
    detrended_fluctuation,
    established_metric_correlations,
    generalized_hurst,
    largest_lyapunov,
    sample_entropy,
)


def test_nonlinear_summaries_are_finite_for_stochastic_series() -> None:
    series = np.random.default_rng(41).normal(size=300)
    estimates = (
        sample_entropy(series),
        correlation_dimension(series),
        largest_lyapunov(series),
        detrended_fluctuation(series),
        generalized_hurst(series),
    )
    assert np.isfinite(estimates).all()


def test_constant_series_is_not_misrepresented_as_complex() -> None:
    series = np.ones(100)
    assert np.isnan(sample_entropy(series))
    assert np.isnan(detrended_fluctuation(series))
    assert np.isnan(generalized_hurst(series))


def test_comparator_correlations_handle_constant_and_adjust_pvalues() -> None:
    values = pd.DataFrame(
        {
            "run_index": np.arange(8),
            "phi_mean": np.arange(8, dtype=float),
            "phi_std": np.ones(8),
            "network_varying": np.arange(8, dtype=float),
            "network_constant": np.ones(8),
            "dynamic_reverse": np.arange(7, -1, -1, dtype=float),
        }
    )
    result = established_metric_correlations(values).set_index("metric")
    assert result.loc["network_varying", "pearson_r"] > 0.99
    assert result.loc["dynamic_reverse", "spearman_rho"] < -0.99
    assert np.isnan(result.loc["network_constant", "pearson_p"])
    assert np.isfinite(result.loc["network_varying", "pearson_q_bh"])
