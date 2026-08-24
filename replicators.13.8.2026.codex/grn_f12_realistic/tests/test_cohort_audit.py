from __future__ import annotations

import numpy as np

from grn_f12_realistic.cohort import audit_one, simulate_one


def test_observational_shard_contains_registered_secondary_endpoints_and_replays(tiny_protocol):
    thresholds = {"q025": 0.85, "q10": 0.95}
    stored = simulate_one(
        tiny_protocol, "continuous", "confirmation", 0, 0.9,
        sensitivity_thresholds=thresholds,
    )
    assert stored["event_count"].shape == (4,)
    assert stored["event_count_q025"].shape == (4,)
    assert stored["event_count_q10"].shape == (4,)
    assert np.all(stored["run5_count"] <= stored["f24_count"])
    audit = audit_one(tiny_protocol, "continuous", 0, 0.9, stored, thresholds)
    assert audit["pass"]
    assert audit["secondary_equal"]

