from __future__ import annotations

import inspect

import numpy as np

from plastic_heredity.phir_extension_px2 import (
    ACQUISITION_LIMIT,
    ACQUISITION_MATRICES,
    ACQUISITION_START,
    ARMS,
    BRANCHES,
    HALVES,
    HORIZON,
    REPRESENTATIONS,
    TARGET_MATRICES,
    AcquiredState,
    AcquisitionBatch,
    _batch_digest,
    _eligible,
    _future_seed,
    _selection_seed,
    protocol,
    scientific_spec,
    validation_checks,
)


def _batch(matrix_id: int, candidates: tuple[str, ...]) -> AcquisitionBatch:
    states = tuple(
        AcquiredState(
            candidate,
            12,
            np.asarray([2, 1, 1, 0], dtype=np.int16),
            12,
            (True, False),
            (0.95, 0.8),
            4,
            31,
            np.asarray([3, 2, 2, 1], dtype=np.int16),
            f"path-{candidate}",
        )
        for candidate in candidates
    )
    provisional = AcquisitionBatch(
        matrix_id,
        np.eye(4),
        np.asarray([1, 1, 1, 1], dtype=np.int16),
        states,
        (),
        1.5,
        "",
    )
    return AcquisitionBatch(
        provisional.matrix_id,
        provisional.beta,
        provisional.initial_composition,
        provisional.states,
        provisional.acquisition_rows,
        provisional.cpu_seconds,
        _batch_digest(provisional),
    )


def test_px2_design_is_fixed_and_bounded() -> None:
    spec = scientific_spec()
    assert ACQUISITION_MATRICES == 32
    assert TARGET_MATRICES == 24
    assert ACQUISITION_START == 10
    assert ACQUISITION_LIMIT == 60
    assert BRANCHES == 64 and HORIZON == 8
    assert HALVES == {"A": (0, 32), "B": (32, 64)}
    assert ARMS == ("RENEWAL_UP", "RENEWAL_DOWN", "RANDOM", "NOOP")
    assert REPRESENTATIONS == ("material", "functional_flux")
    assert spec.cpu_allocation_seconds == 10 * 3600


def test_paired_eligibility_is_seed_ordered_and_never_replaced() -> None:
    batches = [
        _batch(4, ("02", "03")),
        _batch(1, ("02",)),
        _batch(3, ("02", "03")),
        _batch(2, ("02", "03")),
    ]
    selected = _eligible(batches, 2)
    assert [batch.matrix_id for batch in selected] == [2, 3]


def test_batch_digest_excludes_cpu_timing_but_not_science() -> None:
    first = _batch(0, ("02", "03"))
    changed_time = AcquisitionBatch(
        first.matrix_id,
        first.beta,
        first.initial_composition,
        first.states,
        first.acquisition_rows,
        999.0,
        first.scientific_digest,
    )
    assert _batch_digest(first) == _batch_digest(changed_time)
    changed_state = _batch(1, ("02", "03"))
    assert _batch_digest(first) != _batch_digest(changed_state)


def test_future_stream_is_arm_free_and_selection_separate() -> None:
    assert "arm" not in inspect.signature(_future_seed).parameters
    spec = scientific_spec()
    assert _future_seed(spec, "02", 7, 9) != _selection_seed(spec, "02", 7)
    assert len({_future_seed(spec, "02", 7, 9) for _arm in ARMS}) == 1


def test_protocol_excludes_branch_stitching_and_strict_eight() -> None:
    frozen = protocol()
    assert frozen["explicit_pairs"]
    assert frozen["cross_branch_transitions"] is False
    assert frozen["no_48_matrix_continuation"]
    assert "strict-eight is excluded" in frozen["claim_boundary"]


def test_complete_px2_validation_suite() -> None:
    checks = validation_checks()
    assert len(checks) >= 14
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]
