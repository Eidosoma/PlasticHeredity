from __future__ import annotations

import numpy as np

from e01_onset_discovery.untouched_regime_confirmation import (
    confirmation_gate,
    confirmation_state_id,
    exact_probability_replay,
    scientific_manifest_equal,
)


def test_confirmation_state_id_is_deterministic_and_coordinate_specific() -> None:
    first = confirmation_state_id("v", "c2", 3, 20)
    assert first == confirmation_state_id("v", "c2", 3, 20)
    assert first != confirmation_state_id("v", "c2", 3, 35)
    assert first != confirmation_state_id("v", "c3", 3, 20)


def test_scientific_manifest_ignores_unregistered_fields() -> None:
    left = [{"id": 1, "hash": "a", "path": "first", "time": 1.0}]
    right = [{"id": 1, "hash": "a", "path": "second", "time": 9.0}]
    assert scientific_manifest_equal(left, right, ("id", "hash"))
    assert not scientific_manifest_equal(left, right, ("id", "path"))


def test_exact_probability_replay_is_bit_exact() -> None:
    values = np.asarray([0.1, 0.2], dtype=np.float64)
    assert exact_probability_replay(values, values.copy())
    changed = values.copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    assert not exact_probability_replay(values, changed)


def test_confirmation_gate_is_conjunctive() -> None:
    arguments = {
        "availability": True,
        "reliability": True,
        "proper_score": True,
        "overall_rank": True,
        "within_matrix_rank": True,
        "permutation": True,
        "replay": True,
    }
    assert confirmation_gate(**arguments)
    for key in arguments:
        modified = arguments | {key: False}
        assert not confirmation_gate(**modified)
