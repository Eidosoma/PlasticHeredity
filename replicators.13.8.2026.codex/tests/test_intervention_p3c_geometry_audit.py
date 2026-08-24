import numpy as np

from plastic_heredity.intervention_p3c_geometry_audit import (
    block_geometry,
    catalytic_throughput,
    geometry_shift,
    radial_projection,
)


def test_geometry_uses_codex_target_by_catalyst_orientation() -> None:
    composition = np.asarray([3, 1], dtype=np.int64)
    beta = np.asarray([[2.0, 5.0], [7.0, 11.0]], dtype=np.float64)
    x = composition / composition.sum()
    assert catalytic_throughput(composition, beta) == float(x @ beta @ x)


def test_uniform_tightening_has_exact_geometry_ratios() -> None:
    composition = np.asarray([2, 1, 0], dtype=np.int64)
    beta = np.asarray(
        [[2.0, 3.0, 13.0], [5.0, 7.0, 17.0], [19.0, 23.0, 29.0]],
        dtype=np.float64,
    )
    tightened = beta.copy()
    tightened[:2, :2] *= 1.5
    before = block_geometry(composition, beta)
    after = block_geometry(composition, tightened)
    shift = geometry_shift(after, before)
    assert np.isclose(shift["log_throughput_ratio"], np.log(1.5))
    assert np.isclose(shift["relative_block_sum"], 0.5)
    assert np.isclose(shift["log_spectral_radius_ratio"], np.log(1.5))
    assert np.isclose(shift["relative_block_frobenius"], 0.5)


def test_radial_projection_separates_radial_and_orthogonal_changes() -> None:
    before = np.asarray([3.0, 4.0])
    assert np.isclose(radial_projection(before, 1.5 * before), 0.5)
    orthogonal = np.asarray([-4.0, 3.0])
    assert np.isclose(radial_projection(before, before + orthogonal), 0.0)
