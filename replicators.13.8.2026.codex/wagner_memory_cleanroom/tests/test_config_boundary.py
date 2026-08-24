from __future__ import annotations

import json

from wagner_memory_cleanroom.config import DEFAULT_PROTOCOL, load_registration, scaled_futures
from wagner_memory_cleanroom.validation import validate


def test_protocol_loads_and_profiles_are_non_substitutable():
    smoke = load_registration("smoke")
    full = load_registration("full")
    assert smoke.protocol_digest == full.protocol_digest
    assert smoke.scientific is False
    assert full.scientific is True
    assert full.profile["state_sources"] == 240
    assert full.profile["carrier_sources"] == 240
    assert scaled_futures(128, smoke) == 4
    assert scaled_futures(128, full) == 128


def test_protocol_has_no_invalid_full_width_comparator():
    protocol = json.loads(DEFAULT_PROTOCOL.read_text())
    assert "targeted_k10" not in protocol["carrier"]["arms"]
    assert "random_k10" not in protocol["carrier"]["arms"]


def test_clean_room_validation_passes():
    result = validate(load_registration("smoke"))
    assert result["valid"]
    assert result["forbidden_hits"] == []

