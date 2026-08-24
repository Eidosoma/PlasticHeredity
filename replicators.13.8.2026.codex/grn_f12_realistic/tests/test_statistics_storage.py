from __future__ import annotations

import numpy as np

from grn_f12_realistic.statistics import holm_adjust, permutation_tests, split_half_reliability
from grn_f12_realistic.storage import seal_run, verify_run, write_json_atomic


def test_holm_adjustment_is_monotone_and_conservative():
    raw = {"a": 0.001, "b": 0.02, "c": 0.2}
    adjusted = holm_adjust(raw)
    assert adjusted["a"] >= raw["a"]
    assert adjusted["b"] >= adjusted["a"]
    assert adjusted["c"] >= adjusted["b"]


def test_split_half_reliability_reaches_one_for_identical_panels():
    values = np.linspace(0, 1, 30).reshape(3, 10)
    result = split_half_reliability(values, values)
    assert np.isclose(result["spearman_brown_q"], 1.0)


def test_whole_network_permutation_null_is_reproducible():
    rng = np.random.default_rng(4)
    events0 = rng.binomial(20, 0.4, size=(8, 10))
    events1 = rng.binomial(20, 0.4, size=(8, 10))
    full = np.full((8, 10), 0.4)
    history = np.full((8, 10), 0.42)
    first = permutation_tests(events0, events1, 20, full, history, 32, "master", "continuous")
    second = permutation_tests(events0, events1, 20, full, history, 32, "master", "continuous")
    assert first == second


def test_checksum_corruption_is_detected(tmp_path):
    write_json_atomic(tmp_path / "result.json", {"value": 3})
    write_json_atomic(tmp_path / "STATUS.json", {"phase": "sealing"})
    seal_run(tmp_path)
    assert verify_run(tmp_path)["verified"]
    write_json_atomic(tmp_path / "STATUS.json", {"phase": "complete"})
    assert verify_run(tmp_path)["verified"]
    write_json_atomic(tmp_path / "result.json", {"value": 4})
    result = verify_run(tmp_path)
    assert not result["verified"]
    assert "result.json" in result["mismatches"]
