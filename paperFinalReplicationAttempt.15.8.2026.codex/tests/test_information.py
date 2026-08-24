import numpy as np

from aor_replication.config import CausalConfig
from aor_replication.information import (
    LocalGaussianModel,
    fiedler_bipartition,
    fit_causal_trajectory,
    lagged_gaussian_mi_matrix,
)


def _independent_ar(seed: int = 8, samples: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = np.zeros((samples, 2))
    noise = rng.normal(size=(samples, 2))
    for index in range(1, samples):
        data[index] = np.array([0.7, 0.5]) * data[index - 1] + noise[index]
    return data


def test_gaussian_wms_is_near_zero_for_independent_ar_parts() -> None:
    grouped = _independent_ar()
    model = LocalGaussianModel.fit(grouped)
    values = model.score_transitions(grouped[:-1], grouped[1:], measure="wms")
    assert abs(float(values.mean())) < 0.03


def test_mmi_synergy_mean_is_not_below_wms_mean() -> None:
    grouped = _independent_ar(samples=1000)
    model = LocalGaussianModel.fit(grouped)
    wms = model.score_transitions(grouped[:-1], grouped[1:], measure="wms")
    synergy = model.score_transitions(grouped[:-1], grouped[1:], measure="mmi_synergy")
    assert float(synergy.mean()) >= float(wms.mean()) - 1e-12


def test_fiedler_partition_is_nonempty_and_deterministic() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(size=(300, 2))
    data = np.column_stack(
        (base[:, 0], base[:, 0] + 0.01 * rng.normal(size=300), base[:, 1], base[:, 1])
    )
    first, affinity, _ = fiedler_bipartition(data)
    second, _, _ = fiedler_bipartition(data)
    np.testing.assert_array_equal(first, second)
    assert first.any() and (~first).any()
    np.testing.assert_allclose(affinity, affinity.T)


def test_lagged_mi_detects_predictive_pair() -> None:
    rng = np.random.default_rng(11)
    data = rng.normal(size=(1000, 3))
    data[1:, 1] = data[:-1, 0] + 0.05 * rng.normal(size=999)
    matrix = lagged_gaussian_mi_matrix(data)
    assert matrix[0, 1] > matrix[0, 2]


def test_full_causal_pipeline_shapes() -> None:
    rng = np.random.default_rng(5)
    counts = rng.poisson(3, size=(300, 8))
    counts[:, 0] += 1
    result = fit_causal_trajectory(counts, CausalConfig())
    assert result.values.shape == (299,)
    assert result.partition.shape == (7,)
    assert result.grouped.shape == (300, 2)
    assert np.isfinite(result.values).all()

