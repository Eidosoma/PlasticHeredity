from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from plastic_heredity.intervention_core import (
    MolecularEdit,
    ScoredEdit,
    apply_molecular_edit,
    enumerate_legal_edits,
)
from plastic_heredity.phir_ch5 import (
    AUTHORIZATION as CH5_CONFIRMATION_AUTHORIZATION,
    DEFAULT_CONFIRMATION as CH5_CONFIRMATION,
    DEFAULT_CONFIRMATION_WORK as CH5_CONFIRMATION_WORK,
)
from plastic_heredity.phir_feedback_dose import (
    ARMS,
    DOSES,
    DoseBatch,
    _action_seed,
    _batch_digest,
    _directed_arm,
    _dose_series,
    _future_seed,
    _summary,
    scientific_spec,
    select_dose_choices,
    validation_checks,
)


def fixture_scores() -> tuple[ScoredEdit, ...]:
    return tuple(
        ScoredEdit(
            MolecularEdit(index, index + 1),
            probability,
            probability - 0.52,
        )
        for index, probability in enumerate((0.1, 0.3, 0.5, 0.7, 0.9))
    )


def test_scientific_cohort_and_dose_ladder_are_fixed() -> None:
    spec = scientific_spec()
    assert spec.matrices == 24
    assert spec.replicates == 2
    assert spec.natural_generations == 60
    assert spec.control_horizon == 60
    assert spec.pooled30_start == 31
    assert spec.rolling_window == 512
    assert DOSES == (0.25, 0.5, 0.75, 1.0)
    assert len(ARMS) == 11


def test_neutral_is_closest_to_noop() -> None:
    choices = select_dose_choices(0.52, fixture_scores())
    assert choices["NEUTRAL"].selected_probability == 0.5


def test_half_strength_hits_fixture_targets() -> None:
    choices = select_dose_choices(0.52, fixture_scores())
    assert choices["STABILIZE_50"].target_probability == 0.3
    assert choices["STABILIZE_50"].selected_probability == 0.3
    assert choices["DESTABILIZE_50"].target_probability == 0.7
    assert choices["DESTABILIZE_50"].selected_probability == 0.7


def test_full_strength_is_exact_extreme() -> None:
    choices = select_dose_choices(0.52, fixture_scores())
    assert choices["STABILIZE_100"].selected_probability == 0.1
    assert choices["DESTABILIZE_100"].selected_probability == 0.9


def test_selected_probabilities_are_monotone() -> None:
    choices = select_dose_choices(0.52, fixture_scores())
    down = [choices[_directed_arm("STABILIZE", dose)].selected_probability for dose in DOSES]
    up = [choices[_directed_arm("DESTABILIZE", dose)].selected_probability for dose in DOSES]
    assert all(left >= right for left, right in zip(down, down[1:]))
    assert all(left <= right for left, right in zip(up, up[1:]))


def test_ties_are_resolved_by_molecular_indices() -> None:
    scores = (
        ScoredEdit(MolecularEdit(4, 5), 0.4, -0.1),
        ScoredEdit(MolecularEdit(1, 2), 0.4, -0.1),
        ScoredEdit(MolecularEdit(3, 4), 0.6, 0.1),
        ScoredEdit(MolecularEdit(0, 1), 0.6, 0.1),
    )
    choices = select_dose_choices(0.5, scores)
    assert choices["STABILIZE_100"].edit == MolecularEdit(1, 2)
    assert choices["DESTABILIZE_100"].edit == MolecularEdit(0, 1)


def test_legal_substitutions_preserve_mass() -> None:
    composition = np.asarray([2, 0, 1, 0], dtype=np.int64)
    legal = enumerate_legal_edits(composition)
    assert len(legal) == 6
    for edit in legal:
        changed = apply_molecular_edit(composition, edit)
        assert changed.sum() == composition.sum()
        assert np.all(changed >= 0)
        assert np.issubdtype(changed.dtype, np.integer)


def test_future_seed_is_arm_free_and_action_stream_is_separate() -> None:
    spec = scientific_spec()
    shared = _future_seed(spec, "02", 4, 1)
    assert shared == _future_seed(spec, "02", 4, 1)
    assert shared != _action_seed(spec, "02", 4, 1)


def test_batch_digest_is_pickle_stable_and_content_sensitive() -> None:
    batch = DoseBatch(
        matrix_id=0,
        beta=np.eye(2),
        initial_composition=np.asarray([1, 0], dtype=np.int16),
        lineage_rows=({"value": float("nan")},),
        rolling_rows=({"value": 1.0},),
        selected_edit_rows=(),
        scientific_digest="",
    )
    transported = pickle.loads(pickle.dumps(batch, protocol=5))
    assert _batch_digest(batch) == _batch_digest(transported)
    changed = DoseBatch(**{**batch.__dict__, "matrix_id": 1})
    assert _batch_digest(batch) != _batch_digest(changed)


def test_dose_series_is_paired_within_matrix() -> None:
    frame = pd.DataFrame(
        [
            {"matrix_id": matrix, "candidate": "02", "replicate": 0, "arm": arm, "metric": value}
            for matrix, values in ((0, (0.8, 0.2)), (1, (0.6, 0.3)))
            for arm, value in zip(("STABILIZE_50", "DESTABILIZE_50"), values, strict=True)
        ]
    )
    observed = _dose_series(frame, "metric", 0.5, "02", 0)
    assert list(observed.index) == [0, 1]
    assert np.allclose(observed.to_numpy(), (0.6, 0.3), atol=1e-15, rtol=0)


def test_matrix_summary_is_deterministic() -> None:
    first_arrays: dict[str, np.ndarray] = {}
    second_arrays: dict[str, np.ndarray] = {}
    values = np.asarray([0.1, 0.2, 0.3], dtype=float)
    first = _summary(values, 64, "fixture", first_arrays)
    second = _summary(values, 64, "fixture", second_arrays)
    assert first == second
    assert first_arrays.keys() == second_arrays.keys()
    assert all(np.array_equal(first_arrays[key], second_arrays[key]) for key in first_arrays)


def test_original_confirmation_remains_locked() -> None:
    assert not CH5_CONFIRMATION.exists()
    assert not CH5_CONFIRMATION_WORK.exists()
    assert not CH5_CONFIRMATION_AUTHORIZATION.exists()


def test_complete_validation_suite_passes() -> None:
    checks = validation_checks()
    assert len(checks) == 38
    assert all(checks.values())
