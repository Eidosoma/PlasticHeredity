import numpy as np

from plastic_heredity.intervention_p3c_interpretation import (
    _gini,
    extended_geometry,
    geometry_shift,
)


def test_extended_geometry_respects_uniform_scaling() -> None:
    composition = np.asarray([5, 3, 0], dtype=np.int64)
    beta = np.asarray([[2.0, 1.0, 0.5], [0.7, 3.0, 0.4], [1.2, 0.9, 2.5]])
    before = extended_geometry(composition, beta)
    present = np.flatnonzero(composition > 0)
    changed = beta.copy()
    changed[np.ix_(present, present)] *= 1.5
    after = extended_geometry(composition, changed)
    shift = geometry_shift(after, before)
    assert np.isclose(shift["log_throughput_ratio"], np.log(1.5))
    assert np.isclose(shift["relative_block_sum"], 0.5)
    assert np.isclose(shift["log_perron_ratio"], np.log(1.5))
    assert np.isclose(shift["log_singular_ratio"], np.log(1.5))


def test_gini_is_zero_for_uniform_and_positive_for_concentrated() -> None:
    assert np.isclose(_gini(np.ones(4)), 0.0)
    assert _gini(np.asarray([1.0, 1.0, 1.0, 10.0])) > 0.0

