from __future__ import annotations

import json

import pytest

from grn_f12_realistic.config import load_protocol, protocol_for_profile, validate_protocol
from grn_f12_realistic.taskqueue import prepare_queue, queue_status, task_specifications


def test_full_protocol_registered_sizes_and_limits():
    protocol = protocol_for_profile(load_protocol(), "full")
    assert protocol["tiers"]["continuous"]["confirmation_networks"] == 320
    assert protocol["tiers"]["molecular"]["confirmation_networks"] == 160
    assert protocol["operations"]["hard_limit_hours"] == 12.0
    assert protocol["operations"]["required_gpus"] == 2


def test_protocol_rejects_odd_future_halves():
    protocol = protocol_for_profile(load_protocol(), "smoke")
    protocol["tiers"]["continuous"]["futures"] = 15
    with pytest.raises(ValueError, match="equal frozen halves"):
        validate_protocol(protocol)


def test_queue_is_idempotent_and_has_both_tiers(tmp_path):
    protocol = protocol_for_profile(load_protocol(), "smoke")
    first = prepare_queue(tmp_path, "calibration", protocol)
    second = prepare_queue(tmp_path, "calibration", protocol)
    assert first == second
    status = queue_status(tmp_path, "calibration")
    expected = len(task_specifications("calibration", protocol))
    assert status == {"tasks": expected, "done": 0, "failed": 0, "locked": 0}
    manifest = json.loads((first / "queue.json").read_text())
    assert {task["tier"] for task in manifest["tasks"]} == {"continuous", "molecular"}

