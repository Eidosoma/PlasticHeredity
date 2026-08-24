from __future__ import annotations

import inspect

import numpy as np

from plastic_heredity.intervention_core import MolecularEdit, ScoredEdit
from plastic_heredity.phir_protocol_adjudication import (
    DIRECTIONS,
    LAUNCHES,
    PANEL_SIZE,
    REPRESENTATIONS,
    SELECTORS,
    Segment,
    _extreme,
    _future_seed,
    _panel_seed,
    resample_phase,
    sample_panel_edits,
    scientific_spec,
    trace_representations,
    validation_checks,
)
from plastic_heredity.simulator import FissionRecord


def _segments() -> list[Segment]:
    segments: list[Segment] = []
    current = np.zeros(100, dtype=np.int64)
    current[:2] = (20, 20)
    for step in range(1, 4):
        parent = current.copy()
        parent[0] += 1
        daughter = current.copy()
        post = daughter.copy()
        post[0] -= 1
        post[2] += 1
        segments.append(
            Segment(
                step=step,
                pre_growth=current.copy(),
                growth_observations=(parent.copy(),),
                record=FissionRecord(parent, daughter, 0.95, 1),
                post_control=post,
                edit=MolecularEdit(0, 2),
            )
        )
        current = post
    return segments


def test_scientific_factorial_is_frozen() -> None:
    spec = scientific_spec()
    assert spec.matrices == 24
    assert spec.replicates == 2
    assert spec.control_horizon == 60
    assert spec.final_start == 31
    assert spec.phase_points == 16
    assert spec.panel_size == 12
    assert LAUNCHES == ("FRESH", "MATURE")
    assert SELECTORS == ("PANEL12", "EXHAUSTIVE")
    assert DIRECTIONS == ("STABILIZE", "DESTABILIZE")
    assert REPRESENTATIONS == (
        "endpoint_explicit",
        "fable_style",
        "phase_normalized",
        "generational",
    )


def test_panel_contains_only_legal_mass_preserving_edits() -> None:
    composition = np.asarray([2, 1, 0, 0], dtype=np.int64)
    first = sample_panel_edits(composition, np.random.default_rng(77), PANEL_SIZE)
    second = sample_panel_edits(composition, np.random.default_rng(77), PANEL_SIZE)
    assert first == second
    assert len(first) == PANEL_SIZE
    for edit in first:
        assert composition[edit.remove_type] > 0
        assert edit.remove_type != edit.add_type


def test_extreme_selection_is_directional_and_tie_deterministic() -> None:
    scores = (
        ScoredEdit(MolecularEdit(1, 3), 0.2, 0.0),
        ScoredEdit(MolecularEdit(0, 2), 0.2, 0.0),
        ScoredEdit(MolecularEdit(1, 0), 0.8, 0.0),
        ScoredEdit(MolecularEdit(0, 1), 0.8, 0.0),
    )
    assert _extreme(scores, "STABILIZE").edit == MolecularEdit(0, 2)
    assert _extreme(scores, "DESTABILIZE").edit == MolecularEdit(0, 1)


def test_future_seed_has_no_arm_parameter_and_action_stream_is_separate() -> None:
    spec = scientific_spec()
    assert "arm" not in inspect.signature(_future_seed).parameters
    assert "selector" not in inspect.signature(_future_seed).parameters
    assert "launch" not in inspect.signature(_future_seed).parameters
    assert _future_seed(spec, "02", 3, 1) != _panel_seed(
        spec, "02", 3, 1, "FRESH", 7
    )


def test_phase_resampling_preserves_endpoints() -> None:
    path = np.asarray([[1.0, 3.0], [2.0, 5.0], [5.0, 9.0]])
    result = resample_phase(path, 9)
    assert result.shape == (9, 2)
    np.testing.assert_array_equal(result[0], path[0])
    np.testing.assert_array_equal(result[-1], path[-1])


def test_trace_encodings_have_registered_endpoint_contracts() -> None:
    segments = _segments()
    result = trace_representations(segments, final_start=2, phase_points=4, include_registered=True)
    assert result["fable_style"].shape[0] == 2
    assert result["phase_normalized"].shape[0] == 8
    assert result["generational"].shape[0] == 3
    assert result["endpoint_explicit"].shape[0] == 6
    assert result["registered_explicit"].shape[0] == 7
    np.testing.assert_array_equal(
        result["endpoint_explicit"][-1], segments[-1].record.daughter
    )
    np.testing.assert_array_equal(
        result["registered_explicit"][-1], segments[-1].post_control
    )


def test_complete_pre_scientific_validation_suite() -> None:
    checks = validation_checks()
    assert len(checks) >= 32
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]
