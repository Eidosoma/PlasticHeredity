from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _runner() -> object:
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/e01/run_s19_l32_committor_ordered_transition_tube.py"
    )
    spec = importlib.util.spec_from_file_location("test_s19_l32_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_seed_material_matches_the_actual_rng_key() -> None:
    runner = _runner()
    parts = ("bootstrap", "L31_CONFIRMATION", "S12F-CANDIDATE-02")
    assert runner.derived_seed(*parts) == runner.derived_seed(*parts)
    assert runner.seed_material_sha256(*parts) != runner.seed_material_sha256(
        "bootstrap", "L28_VALIDATION", "S12F-CANDIDATE-02"
    )


def test_response_permutation_is_landmark_stratified_and_paired() -> None:
    runner = _runner()
    frame = pd.DataFrame(
        {
            "landmark": [32, 32, 64, 64],
            "successes": [1, 2, 3, 4],
            "qHat": np.asarray([1, 2, 3, 4], dtype=np.float64) / 128,
        }
    )
    result = runner.permute_within_landmark(frame, np.random.default_rng(3201))
    for landmark in (32, 64):
        source = frame[frame["landmark"].eq(landmark)]
        observed = result[result["landmark"].eq(landmark)]
        assert sorted(observed["successes"]) == sorted(source["successes"])
        assert all(
            np.isclose(row.qHat, row.successes / 128)
            for row in observed.itertuples(index=False)
        )


def test_frozen_l32_view_contract_retains_temporal_direction() -> None:
    runner = _runner()
    rng = np.random.default_rng(3202)
    states = rng.poisson(2.0, size=(32, 100)).astype(np.int64)
    states[:, 0] += 1
    original = runner.L27.transition_tube_views(states)
    reversed_views = runner.L27.transition_tube_views(states[::-1])
    assert tuple(original) == runner.VIEWS
    assert [original[key].shape for key in runner.VIEWS] == [(693,), (315,), (378,)]
    assert all(
        not np.array_equal(original[key], reversed_views[key])
        for key in runner.VIEWS
    )
