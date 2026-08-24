from __future__ import annotations

from copy import deepcopy

import numpy as np

from wagner_cleanroom.dynamics import sample_rulebook
from wagner_cleanroom.experiment import (
    ARM_CODE,
    _events,
    _simulate_cell,
    simulate_primary_rulebook,
)
from wagner_cleanroom.protocol import load_protocol


def fixture_rulebook():
    for proposal in range(100):
        result = sample_rulebook("experiment-test", proposal)
        if result is not None:
            return result
    raise AssertionError("no eligible fixture rulebook")


def tiny_protocol():
    protocol = deepcopy(load_protocol("primary", "smoke"))
    for condition in protocol["conditions"]:
        condition["futures"] = 4
    protocol["horizon"] = 12
    return protocol


def test_self_and_transplant_are_pathwise_identical() -> None:
    source = fixture_rulebook()
    protocol = tiny_protocol()
    left = _simulate_cell(source, protocol, "primary", "self_continuation", 0, 0, "neutral_damage", 0, 4)
    right = _simulate_cell(source, protocol, "primary", "state_transplant", 0, 0, "neutral_damage", 0, 4)
    assert np.array_equal(left["trajectory_digest"], right["trajectory_digest"])
    assert np.array_equal(left["destination"], right["destination"])


def test_primary_rulebook_has_unique_semantic_coordinates() -> None:
    rows = simulate_primary_rulebook(fixture_rulebook(), tiny_protocol())
    fields = ["condition", "arm", "history", "midpoint", "challenge", "age", "future"]
    assert len(np.unique(rows[fields])) == len(rows)
    assert set(np.unique(rows["arm"]).tolist()) == set(ARM_CODE.values())


def test_f12_event_fixture() -> None:
    old = np.asarray([0], dtype=np.uint16)
    trajectory = np.asarray([[1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]], dtype=np.uint16)
    f12, strict = _events(trajectory, old, 10)
    assert int(f12[0]) == 1
    assert int(strict[0]) == 1

