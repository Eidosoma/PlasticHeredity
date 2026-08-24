from __future__ import annotations

import pickle

import numpy as np

from plastic_heredity.phir_ch5 import (
    AUTHORIZATION as CH5_CONFIRMATION_AUTHORIZATION,
    DEFAULT_CONFIRMATION as CH5_CONFIRMATION,
    DEFAULT_CONFIRMATION_WORK as CH5_CONFIRMATION_WORK,
)
from plastic_heredity.phir_instruments import score_phi_window
from plastic_heredity.phir_window_bridge import (
    ARMS,
    MATRICES,
    WindowBridgeBatch,
    _batch_digest,
    _future_seed,
    partition_disagreement,
    score_counts,
    scientific_spec,
    validation_checks,
)


def fixture_counts() -> np.ndarray:
    return np.asarray(
        [
            [2 + ((time + 3 * molecule) % 7) for molecule in range(100)]
            for time in range(48)
        ],
        dtype=np.int64,
    )


def test_scientific_cohort_and_windows_are_fixed() -> None:
    spec = scientific_spec()
    assert spec.matrices == MATRICES == 24
    assert spec.replicates == 2
    assert spec.natural_generations == 60
    assert spec.bridge_horizon == 60
    assert spec.pooled20_start == 41
    assert spec.pooled30_start == 31
    assert spec.rolling_window == 512
    assert ARMS == ("MODEL_STABILIZE", "MODEL_DESTABILIZE")


def test_clr_revised_and_full_typeset_match_sealed_instrument() -> None:
    counts = fixture_counts()
    bridge = score_counts(counts, "clr", include_full_typeset=True)
    sealed = score_phi_window(counts, include_typeset=True)
    assert abs(bridge.revised - sealed.revised_phi_r) < 1e-12
    assert abs(bridge.full_typeset - sealed.typeset_phi_r) < 1e-12
    assert np.allclose(bridge.atoms, sealed.atoms, atol=1e-12, rtol=0)


def test_macro_and_full_typeset_are_not_conflated() -> None:
    score = score_counts(fixture_counts(), "clr", include_full_typeset=True)
    assert np.isfinite(score.macro_typeset)
    assert np.isfinite(score.full_typeset)
    assert not np.isclose(score.macro_typeset, score.full_typeset)


def test_raw_count_is_an_explicit_finite_sensitivity() -> None:
    score = score_counts(fixture_counts(), "raw_count", include_full_typeset=False)
    assert np.isfinite(score.revised)
    assert np.isfinite(score.macro_typeset)
    assert np.isnan(score.full_typeset)


def test_partition_disagreement_is_label_invariant() -> None:
    assert partition_disagreement((0, 1), (2, 3), (2, 3), (0, 1)) == 0.0
    assert partition_disagreement((0, 1), (2, 3), (0, 2), (1, 3)) == 0.5


def test_batch_digest_is_pickle_boundary_stable_and_content_sensitive() -> None:
    batch = WindowBridgeBatch(
        matrix_id=0,
        beta=np.eye(2),
        initial_composition=np.asarray([1, 0], dtype=np.int16),
        lineage_rows=({"reading": float("nan")},),
        window_rows=({"reading": 1.0},),
        selected_edit_rows=(),
        scientific_digest="",
    )
    transported = pickle.loads(pickle.dumps(batch, protocol=5))
    assert _batch_digest(batch) == _batch_digest(transported)
    changed = WindowBridgeBatch(**{**batch.__dict__, "matrix_id": 1})
    assert _batch_digest(batch) != _batch_digest(changed)


def test_future_stream_key_cannot_depend_on_arm() -> None:
    spec = scientific_spec()
    first = _future_seed(spec, "02", 7, 1)
    second = _future_seed(spec, "02", 7, 1)
    assert first == second


def test_original_confirmation_remains_locked() -> None:
    assert not CH5_CONFIRMATION.exists()
    assert not CH5_CONFIRMATION_WORK.exists()
    assert not CH5_CONFIRMATION_AUTHORIZATION.exists()


def test_validation_suite_has_29_passing_checks() -> None:
    checks = validation_checks()
    assert len(checks) == 29
    assert all(checks.values())
