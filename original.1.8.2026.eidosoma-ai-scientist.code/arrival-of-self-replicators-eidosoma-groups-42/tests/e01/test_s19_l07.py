from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from e01_s19_occupancy_search.core import (
    ExploratoryExposureDefinition,
    aggregate_occupancy,
    adjacent_scores,
    boundary_scores,
    fingerprint,
)
from scripts.e01.run_s19_l07 import (
    _positive_float64_ulp_distance,
    validate_fresh_seed_firewall,
)


def _obs(index: int, kind: str, generation: int, state: tuple[int, ...]):
    return SimpleNamespace(
        observation_index=index,
        observation_kind=kind,
        growth_generation_one_based=generation,
        state=state,
    )


def _state(a: int, b: int) -> tuple[int, ...]:
    values = [0] * 100
    values[0] = a
    values[1] = b
    return tuple(values)


def test_adjacent_incoming_and_average_are_literal() -> None:
    observations = (
        _obs(0, "initial_selected_state", 0, _state(1, 0)),
        _obs(1, "molecular_update", 1, _state(1, 1)),
        _obs(2, "molecular_update", 1, _state(0, 1)),
    )
    incoming = adjacent_scores(observations, alignment="INCOMING_DUPLICATE_FIRST")
    average = adjacent_scores(observations, alignment="TWO_NEIGHBOR_AVERAGE")
    expected = 1 / np.sqrt(2)
    assert np.allclose(incoming, [expected, expected, expected], atol=0, rtol=1e-15)
    assert np.allclose(average, incoming, atol=0, rtol=1e-15)


def test_parent_daughter_boundary_scores_use_exact_preceding_parent() -> None:
    trajectory = SimpleNamespace(
        completed_fissions=2,
        observations=(
            _obs(0, "initial_selected_state", 0, _state(1, 0)),
            _obs(1, "molecular_update", 1, _state(2, 0)),
            _obs(2, "post_fission", 1, _state(1, 0)),
            _obs(3, "molecular_update", 2, _state(1, 1)),
            _obs(4, "post_fission", 2, _state(0, 1)),
        ),
    )
    scores = boundary_scores(
        trajectory,
        boundary_object="PARENT_TO_SELECTED_DAUGHTER",
        alignment="INCOMING_DUPLICATE_FIRST",
    )
    assert np.allclose(scores, [1.0, 1 / np.sqrt(2)], atol=0, rtol=1e-15)


def test_fingerprint_and_bootstrap_target_are_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "analysisUnitIndex": np.arange(5),
            "isReplicator": [False, True, True, False, True],
        }
    )
    result = fingerprint(frame)
    assert result["occupancy"] == 0.6
    assert result["persistence"] == 3
    assert result["episodeCount"] == 2
    rows = pd.DataFrame(
        {
            "roundId": ["R"] * 3,
            "settingId": ["S"] * 3,
            "settingPairId": ["P"] * 3,
            "candidateId": ["C"] * 3,
            "matrixIndex": [0, 1, 2],
            "occupancy": [0.87, 0.88, 0.89],
        }
    )
    first = aggregate_occupancy(rows, bootstrap_replicates=128)
    second = aggregate_occupancy(rows, bootstrap_replicates=128)
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "withinPaperApproximateBand"]


def test_positive_float64_ulp_distance_obeys_exact_units() -> None:
    value = np.asarray([0.9], dtype=np.float64)
    next_value = np.nextafter(value, np.asarray([1.0], dtype=np.float64))
    assert _positive_float64_ulp_distance(value, value).tolist() == [0]
    assert _positive_float64_ulp_distance(value, next_value).tolist() == [1]


def test_l07_extended_exposure_is_bounded_and_identity_compatible() -> None:
    exposure = ExploratoryExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=3.0)
    exposure.validate()
    assert exposure.identity == "FIXED-h=3"
    with np.testing.assert_raises(ValueError):
        ExploratoryExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=1.25).validate()


def test_fresh_seed_firewall_rejects_frozen_hash_overlap(monkeypatch) -> None:
    frozen = pd.DataFrame(
        {
            "betaSha256": ["old-beta"],
            "initialStateSha256": ["old-initial"],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: frozen)
    rows = []
    for simulation in ("A", "B"):
        for matrix in range(100):
            rows.append(
                {
                    "simulationId": simulation,
                    "matrixIndex": matrix,
                    "betaSha256": "old-beta" if matrix == 0 else f"new-beta-{matrix}",
                    "initialStateSha256": f"new-initial-{matrix}",
                }
            )
    result = validate_fresh_seed_firewall(pd.DataFrame(rows))
    assert not result["passed"]
    assert result["betaOverlapCount"] == 1
    assert result["initialStateOverlapCount"] == 0
