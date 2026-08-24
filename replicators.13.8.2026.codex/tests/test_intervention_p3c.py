import numpy as np
import pytest

from plastic_heredity.intervention_p3c import (
    ARMS,
    SURGERY_NORM_FRACTION,
    _has_run,
    _slope_and_rank_statistics,
    acquire_natural_broken_state,
    catalytic_throughput,
    select_surgeries,
    throughput_neutral_pp_surgery,
)
from plastic_heredity.config import ExperimentConfig
from plastic_heredity.experiment import StateCase
from plastic_heredity.simulator import FissionRecord, Snapshot


@pytest.fixture
def state() -> tuple[np.ndarray, np.ndarray]:
    composition = np.asarray([5, 3, 2, 0], dtype=np.int64)
    beta = np.asarray(
        [
            [2.0, 0.7, 1.3, 0.4],
            [1.1, 3.0, 0.8, 0.9],
            [0.6, 1.4, 2.5, 1.2],
            [0.5, 0.3, 0.4, 1.8],
        ],
        dtype=np.float64,
    )
    return composition, beta


def test_throughput_neutral_surgery_preserves_throughput_and_norm(state) -> None:
    composition, beta = state
    surgery = throughput_neutral_pp_surgery(
        composition, beta, np.random.default_rng(914)
    )
    present = np.flatnonzero(composition > 0)
    block = beta[np.ix_(present, present)]
    assert np.isclose(
        surgery.observed_norm,
        SURGERY_NORM_FRACTION * np.linalg.norm(block),
        atol=1e-11,
        rtol=0.0,
    )
    assert np.isclose(
        catalytic_throughput(composition, surgery.beta),
        catalytic_throughput(composition, beta),
        atol=1e-10,
        rtol=1e-12,
    )
    assert np.all(surgery.beta > 0.0)


def test_neutral_selection_is_deterministic(state) -> None:
    composition, beta = state
    left = throughput_neutral_pp_surgery(
        composition, beta, np.random.default_rng(99)
    )
    right = throughput_neutral_pp_surgery(
        composition, beta, np.random.default_rng(99)
    )
    assert np.array_equal(left.beta, right.beta)


def test_all_arms_have_frozen_order_and_singleton_is_structural_noop(state) -> None:
    _composition, beta = state
    singleton = np.asarray([0, 8, 0, 0], dtype=np.int64)
    surgeries = select_surgeries(
        singleton,
        beta,
        np.random.default_rng(1),
        np.random.default_rng(2),
    )
    assert len(surgeries) == len(ARMS)
    assert all(item is None for item in surgeries)


def test_neutral_surgery_rejects_singleton(state) -> None:
    _composition, beta = state
    with pytest.raises(ValueError, match="two present"):
        throughput_neutral_pp_surgery(
            np.asarray([0, 8, 0, 0], dtype=np.int64),
            beta,
            np.random.default_rng(3),
        )


def test_run_fixture_covers_horizon_edges() -> None:
    assert _has_run(np.asarray([True, True, True]), 3) == (True, 3)
    assert _has_run(np.asarray([False, True, True, True]), 3) == (True, 4)
    assert _has_run(np.asarray([True, True]), 3) == (False, -1)


def test_state_centered_slope_and_rank_have_registered_sign() -> None:
    x = np.asarray([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])
    y = -0.5 * x
    ids = np.asarray([0, 1], dtype=np.int64)
    draws = np.asarray([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=np.int64)
    summary, stored = _slope_and_rank_statistics(x, y, ids, draws)
    assert np.isclose(summary["state_centered_slope"], -0.5)
    assert np.isclose(summary["mean_within_state_spearman"], -1.0)
    assert np.allclose(stored["slope"], -0.5)


def test_natural_break_acquisition_saves_exact_selected_daughter(monkeypatch) -> None:
    composition = np.asarray([2, 1], dtype=np.int64)
    beta = np.ones((2, 2), dtype=np.float64)
    snapshot = Snapshot(
        composition=composition,
        generation=7,
        inheritance=(True,),
        boundary_h=(0.95,),
        previous_growth_steps=4,
        cumulative_growth_steps=9,
    )
    case = StateCase("fixture", "FIX", "02", 0, 20, beta, snapshot)
    daughter = np.asarray([1, 1], dtype=np.int64)

    def fixed_advance(*_args, **_kwargs):
        return FissionRecord(
            parent=np.asarray([3, 1], dtype=np.int64),
            daughter=daughter,
            h=0.9,
            growth_steps=6,
        )

    monkeypatch.setattr(
        "plastic_heredity.intervention_p3c.advance_fission", fixed_advance
    )
    broken, anchor, audit = acquire_natural_broken_state(
        case, ExperimentConfig.quick()
    )
    assert audit["eligible"] is True
    assert np.array_equal(broken.snapshot.composition, daughter)
    assert np.array_equal(anchor, np.asarray([3, 1]))
    assert broken.snapshot.inheritance == (True, False)
    assert broken.snapshot.cumulative_growth_steps == 15
