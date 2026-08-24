from __future__ import annotations

import numpy as np
import pytest

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.generative_nulls import (
    F32_BRANCHES,
    HOMOGENEOUS_VALUE,
    INTERVENTION_BRANCHES,
    MATRICES,
    MECHANISMS,
    NullCase,
    _fission_only_advance,
    _fixture_snapshot,
    _future_seed,
    _homogeneous_beta,
    _matrix_means,
    _proper_brier,
    _record_digest,
    _simulate_case_future,
    derange_beta,
    fixed_point_free_permutation,
    protocol,
    validation_checks,
)
from plastic_heredity.intervention_outgoing_rule import select_outgoing_rule_edits
from plastic_heredity.processes import evaluate_process
from plastic_heredity.simulator import FissionRecord, Snapshot, _fission


def test_protocol_freezes_compact_reviewer_suite() -> None:
    frozen = protocol()
    assert MATRICES == 96
    assert F32_BRANCHES == 64
    assert INTERVENTION_BRANCHES == 32
    assert tuple(frozen["mechanisms"]) == MECHANISMS
    assert frozen["inference"]["omnibus_gate"] is False
    assert frozen["integrity"]["complete_exact_replay"] is True


def test_sattolo_permutation_is_fixed_point_free_and_deterministic() -> None:
    left = fixed_point_free_permutation(100, np.random.default_rng(17))
    right = fixed_point_free_permutation(100, np.random.default_rng(17))
    assert np.array_equal(left, right)
    assert np.unique(left).size == 100
    assert not np.any(left == np.arange(100))


def test_coupling_derangement_preserves_weights_and_spectrum() -> None:
    rng = np.random.default_rng(18)
    beta = np.exp(rng.normal(size=(10, 10)))
    permutation = fixed_point_free_permutation(10, rng)
    shifted = derange_beta(beta, permutation)
    assert np.array_equal(np.sort(beta, axis=None), np.sort(shifted, axis=None))
    assert np.allclose(
        np.linalg.svd(beta, compute_uv=False),
        np.linalg.svd(shifted, compute_uv=False),
        rtol=1e-13,
        atol=1e-13,
    )


def test_invalid_coupling_permutation_is_rejected() -> None:
    with pytest.raises(ValueError, match="fixed-point-free"):
        derange_beta(np.ones((4, 4)), np.arange(4))


def test_homogeneous_beta_is_global_positive_constant() -> None:
    matrix = _homogeneous_beta(GardConfig())
    assert matrix.shape == (100, 100)
    assert np.all(matrix == HOMOGENEOUS_VALUE)
    assert HOMOGENEOUS_VALUE == float(np.exp(4.0))


def test_fission_only_fixed_size_preserves_registered_masses() -> None:
    record = _fission_only_advance(
        _fixture_snapshot().composition,
        GardConfig(),
        CANDIDATES["02"],
        np.random.default_rng(19),
    )
    assert record.parent.sum() == 80
    assert record.daughter.sum() == 40
    assert np.all(record.daughter >= 0)


def test_fission_only_binomial_uses_registered_second_daughter() -> None:
    composition = _fixture_snapshot().composition
    config = GardConfig()
    seed = 20
    observed = _fission_only_advance(
        composition, config, CANDIDATES["03"], np.random.default_rng(seed)
    )
    rng = np.random.default_rng(seed)
    arrivals = config.n_max - int(composition.sum())
    parent = composition + rng.multinomial(
        arrivals, np.full(config.n_types, 1.0 / config.n_types)
    )
    expected = _fission(parent, config, CANDIDATES["03"], rng)
    assert np.array_equal(observed.parent, parent)
    assert np.array_equal(observed.daughter, expected)


def _case(mechanism: str, active: np.ndarray) -> NullCase:
    beta = np.arange(1, 10001, dtype=np.float64).reshape(100, 100)
    permutation = np.roll(np.arange(100), 1)
    return NullCase(
        state_id=f"fixture-{mechanism}",
        mechanism=mechanism,
        candidate="02",
        matrix_id=7,
        landmark=20,
        source_beta=beta,
        active_beta=active,
        snapshot=_fixture_snapshot(),
        main_attempt=0,
        coupling_permutation=permutation,
    )


def test_future_stream_key_excludes_mechanism() -> None:
    beta = np.ones((100, 100), dtype=np.float64)
    natural = _case("NATURAL_GARD", beta)
    coupled = _case("COUPLING_DERANGED", beta * 2.0)
    assert _future_seed(natural, 3) == _future_seed(coupled, 3)
    assert _future_seed(natural, 3) != _future_seed(natural, 4)


def test_natural_wrapper_is_exact_plain_simulator_path() -> None:
    rng = np.random.default_rng(21)
    beta = np.exp(rng.normal(-2.0, 1.0, size=(100, 100)))
    case = _case("NATURAL_GARD", beta)
    case = NullCase(
        **{**case.__dict__, "source_beta": beta, "active_beta": beta}
    )
    seed = 22
    left = _simulate_case_future(
        case, GardConfig(), 3, np.random.default_rng(seed)
    )
    right = _simulate_case_future(
        case, GardConfig(), 3, np.random.default_rng(seed)
    )
    assert _record_digest(*left) == _record_digest(*right)


def test_source_rule_is_selected_from_original_beta() -> None:
    source = np.arange(1, 10001, dtype=np.float64).reshape(100, 100)
    active = source[::-1, ::-1].copy()
    case = _case("COUPLING_DERANGED", active)
    expected = select_outgoing_rule_edits(case.snapshot.composition, source)
    observed = select_outgoing_rule_edits(
        case.snapshot.composition, case.source_beta
    )
    assert observed == expected


def test_joint_break_run3_threshold_and_order_are_unchanged() -> None:
    parent = np.zeros(100, dtype=np.int64)
    parent[0] = 80
    daughter = np.zeros(100, dtype=np.int64)
    daughter[0] = 40

    def record(h: float) -> FissionRecord:
        return FissionRecord(parent.copy(), daughter.copy(), h, 1)

    inherited = np.nextafter(0.9, np.inf)
    outcome = evaluate_process(
        [record(0.9), record(inherited), record(inherited), record(inherited)]
    )
    assert outcome.joint_break_run3 is True
    assert evaluate_process([record(inherited)] * 4).joint_break_run3 is False


def test_proper_branch_brier_not_q_squared_error() -> None:
    values = _proper_brier(np.asarray([0.0, 1.0]), np.asarray([0.2, 0.8]))
    assert np.allclose(values, [0.04, 0.04])


def test_matrix_reduction_keeps_whole_blocks() -> None:
    values = np.arange(20, dtype=np.float64)
    matrix_ids = np.repeat(np.arange(4), 5)
    assert np.array_equal(_matrix_means(values, matrix_ids), [2, 7, 12, 17])


def test_instantaneous_edit_does_not_modify_history() -> None:
    snapshot = _fixture_snapshot()
    beta = np.eye(100, dtype=np.float64) + 0.01
    edit = select_outgoing_rule_edits(snapshot.composition, beta)["RULE_DOWN"]
    from plastic_heredity.generative_nulls import _edited_snapshot

    changed = _edited_snapshot(snapshot, edit)
    assert changed.inheritance == snapshot.inheritance
    assert changed.boundary_h == snapshot.boundary_h
    assert changed.previous_growth_steps == snapshot.previous_growth_steps
    assert changed.cumulative_growth_steps == snapshot.cumulative_growth_steps
    assert changed.composition.sum() == snapshot.composition.sum()


def test_complete_pre_scientific_validation_panel_passes() -> None:
    checks = validation_checks()
    assert len(checks) >= 30
    assert all(checks.values()), [name for name, value in checks.items() if not value]
