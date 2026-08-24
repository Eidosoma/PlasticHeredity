from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reviewer_cross_substrate_response.analysis import (
    pilot_eligibility,
    sensitivity_summaries,
)
from reviewer_cross_substrate_response.campaign import (
    assign_matched_strangers,
    mechanics_jobs,
    mechanics_trial,
)
from reviewer_cross_substrate_response.core import (
    calibrated_threshold,
    block_bootstrap_interval,
    canonical_similarity,
    connected_components_torus,
    crop_component,
    derive_seed,
    exact_order_null_probability,
    score_binary_break_renewal,
    score_break_renewal,
)
from reviewer_cross_substrate_response.models import (
    EVOLOOP_SEED,
    SMOKE_PROFILE,
    EvoloopParameters,
    EvoloopRule,
    ProtocellParameters,
    advance_evoloop_boundary,
    advance_protocell_boundary,
    evoloop_initial,
    protocell_initial,
    protocell_sweep,
)
from reviewer_cross_substrate_response import run_experiment as runner


def test_endpoint_is_strict_and_ordered() -> None:
    assert score_break_renewal([0.5, 0.91, 0.92, 0.93], 0.9).event
    assert not score_break_renewal([0.91, 0.92, 0.93, 0.5], 0.9).event
    assert not score_break_renewal([0.5, 0.9, 0.91, 0.92], 0.9).event
    result = score_break_renewal([0.5, 0.91, 0.92, 0.93, 0.1], 0.9)
    assert result.break_index == 0
    assert result.renewal_start == 1


def test_exact_order_null_matches_direct_enumeration() -> None:
    length = 8
    inherited_count = 5
    outcomes = []
    for positions in combinations(range(length), inherited_count):
        sequence = [index in positions for index in range(length)]
        outcomes.append(score_binary_break_renewal(sequence))
    assert exact_order_null_probability(length, inherited_count) == np.mean(outcomes)


def test_similarity_is_centroid_c4_and_translation_invariant() -> None:
    raster = np.zeros((7, 9), dtype=np.uint8)
    raster[0, 1] = 1
    raster[1:5, 3] = 2
    raster[6, 8] = 1
    padded = np.pad(raster, ((3, 1), (2, 4)))
    assert canonical_similarity(raster, raster) == 1.0
    assert canonical_similarity(raster, np.rot90(raster)) == 1.0
    assert canonical_similarity(raster, padded) == 1.0
    changed = raster.copy()
    changed[1, 3] = 1
    assert 0.0 <= canonical_similarity(raster, changed) < 1.0
    assert canonical_similarity(raster, np.zeros_like(raster)) == 0.0


def test_periodic_components_and_crop_join_wrapped_cells() -> None:
    grid = np.zeros((8, 8), dtype=np.uint8)
    grid[0, 0] = 1
    grid[0, 7] = 2
    grid[7, 0] = 1
    components = connected_components_torus(grid > 0)
    assert len(components) == 1
    crop = crop_component(grid, components[0])
    assert crop.shape == (2, 2)
    assert np.count_nonzero(crop) == 3


def test_threshold_uses_registered_higher_quantile() -> None:
    values = np.arange(20, dtype=float) / 20.0
    assert calibrated_threshold(values) == np.quantile(values, 0.95, method="higher")


def test_bootstrap_weights_world_blocks_equally() -> None:
    compact = block_bootstrap_interval(
        [0.0, 10.0], [0, 1], repetitions=256, seed=10
    )
    duplicated = block_bootstrap_interval(
        [0.0] * 100 + [10.0], [0] * 100 + [1], repetitions=256, seed=10
    )
    assert compact == duplicated


def test_evoloop_public_fixture() -> None:
    rule = EvoloopRule()
    assert rule.covered_neighborhoods == 55_139
    grid = np.zeros((96, 96), dtype=np.uint8)
    grid[40:49, 40:49] = EVOLOOP_SEED
    counts = {0: int(np.count_nonzero(grid))}
    for tick in range(1, 151):
        grid = rule.step(grid)
        if tick in (1, 10, 50, 100, 150):
            counts[tick] = int(np.count_nonzero(grid))
    assert counts == {0: 60, 1: 60, 10: 65, 50: 86, 100: 114, 150: 137}
    assert len(connected_components_torus(grid > 0, min_size=20)) == 2


def test_protocell_seeded_sweep_fixture() -> None:
    grid = protocell_initial(64).grid
    protocell_sweep(
        grid,
        ProtocellParameters.from_pair(0.1, 1e-4),
        np.random.default_rng(123),
    )
    assert np.isin(grid, [0, 1, 2]).all()
    assert (np.count_nonzero(grid == 1), np.count_nonzero(grid == 2)) == (99, 1)


def test_boundary_caps_include_persistence_lookahead() -> None:
    parameters = ProtocellParameters.from_pair(1e-2, 1e-4)
    _, transition = advance_protocell_boundary(
        protocell_initial(64), parameters, np.random.default_rng(1), cap=3, persistence=8
    )
    assert transition.elapsed_updates <= 3

    rng = np.random.default_rng(1)
    world = evoloop_initial(64, 1, rng)
    _, transition = advance_evoloop_boundary(
        world,
        EvoloopParameters(1, 0.0),
        EvoloopRule(),
        rng,
        cap=3,
        persistence=8,
        arm_window=32,
    )
    assert transition.elapsed_updates <= 3


def test_smoke_mechanics_reaches_one_boundary_in_each_substrate() -> None:
    for model in ("protocell", "evoloop"):
        result = mechanics_trial(*mechanics_jobs(model, SMOKE_PROFILE)[0])
        assert result.passed
        assert result.completed_boundaries == 1
        assert result.total_updates <= SMOKE_PROFILE.mechanics_cap


def test_strangers_must_come_from_another_block() -> None:
    observations = []
    for block_id, state in ((0, 1), (0, 1), (1, 2), (1, 2)):
        raster = np.asarray([[state]], dtype=np.uint8)
        observations.append(
            {
                "block_id": block_id,
                "parent_size": 1,
                "child_size": 1,
                "parent_crop": raster,
                "child_crop": raster,
            }
        )
    assign_matched_strangers(
        observations,
        seed_parts=("test", "cross-block"),
        different_key="block_id",
    )
    assert [item["stranger_similarity"] for item in observations] == [0.0] * 4


def test_stranger_matching_uses_offspring_not_parent_size() -> None:
    raster = np.asarray([[1]], dtype=np.uint8)
    observations = [
        {
            "block_id": 0,
            "parent_size": 200,
            "child_size": 100,
            "parent_crop": raster,
            "child_crop": raster,
        },
        {
            "block_id": 1,
            "parent_size": 300,
            "child_size": 100,
            "parent_crop": raster,
            "child_crop": raster,
        },
    ]
    assign_matched_strangers(
        observations,
        seed_parts=("test", "offspring-size"),
        different_key="block_id",
    )
    assert [item["stranger_similarity"] for item in observations] == [1.0, 1.0]


def test_compressed_sidecars_drive_cross_block_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "TASK_ROOT", tmp_path)
    paths = {"work": tmp_path / "work"}
    model_root = paths["work"] / "pilot" / "protocell"
    for block_id, state in ((0, 1), (1, 2)):
        raster = np.asarray([[state]], dtype=np.uint8)
        row = {
            "model": "protocell",
            "stage": "pilot",
            "block_id": block_id,
            "parameter_key": "fixture",
            "landmark": 1,
            "branch": 0,
            "half": "A",
            "future_id": f"fixture:{block_id}",
            "boundary": 0,
            "similarity": 1.0,
            "stranger_similarity": np.nan,
            "inherited": 1,
            "parent_size": 1,
            "child_size": 1,
            "elapsed_updates": 1,
            "observation_index": 0,
        }
        stem = f"block_{block_id:04d}"
        runner._write_csv(
            model_root / f"{stem}_boundaries.csv",
            pd.DataFrame([row], columns=runner.BOUNDARY_COLUMNS),
        )
        runner._write_crop_archive(
            model_root / f"{stem}_crops.npz",
            [
                {
                    "observation_index": 0,
                    "landmark": 1,
                    "boundary": 0,
                    "branch": 0,
                    "parent_crop": raster,
                    "child_crop": raster,
                }
            ],
        )
        runner._write_json(model_root / f"{stem}.json", {})

    result = runner._match_stage_strangers(paths, "pilot", "smoke", "protocell")
    assert result == {"matched_boundaries": 2, "unmatched_boundaries": 0}
    for block_id in (0, 1):
        frame = pd.read_csv(
            model_root / f"block_{block_id:04d}_boundaries.csv"
        )
        assert frame.loc[0, "stranger_similarity"] == 0.0


def test_calibration_checkpoint_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "TASK_ROOT", tmp_path)
    paths = {"work": tmp_path / "work"}
    parent = np.asarray([[1, 0], [2, 1]], dtype=np.uint8)
    child = np.asarray([[1, 2]], dtype=np.uint8)
    observations = [
        {
            "model": "protocell",
            "block_id": 3,
            "parameter_key": "fixture",
            "boundary_index": 0,
            "lineage_attempt": 1,
            "parent_size": 3,
            "child_size": 2,
            "actual_similarity": 0.25,
            "parent_crop": parent,
            "child_crop": child,
        }
    ]
    runner._write_calibration_checkpoint(paths, "protocell", 3, observations)
    restored = runner._read_calibration_checkpoint(paths, "protocell", 3)
    assert len(restored) == 1
    assert restored[0]["parameter_key"] == "fixture"
    assert restored[0]["actual_similarity"] == observations[0]["actual_similarity"]
    assert np.array_equal(restored[0]["parent_crop"], parent)
    assert np.array_equal(restored[0]["child_crop"], child)


def test_exhausted_main_block_remains_in_future_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner, "generate_landmarks", lambda *args, **kwargs: ([], [], None)
    )
    parameters = ProtocellParameters.from_pair(1e-2, 1e-4)
    result = runner._simulate_block_job(
        (
            "pilot",
            "protocell",
            7,
            {
                "kind": "protocell",
                "parameter_key": parameters.key,
                "p_y": parameters.p_y,
                "a_y": parameters.a_y,
                "p_x": parameters.p_x,
                "a_x": parameters.a_x,
            },
            SMOKE_PROFILE,
            0.9,
            SMOKE_PROFILE.pilot_branches,
        )
    )
    assert not result["main_complete"]
    assert len(result["future_rows"]) == (
        len(SMOKE_PROFILE.landmarks) * SMOKE_PROFILE.pilot_branches
    )
    assert all(row["event"] == 0 for row in result["future_rows"])
    assert all(row["failure"].startswith("main_unavailable") for row in result["future_rows"])


def _future_frame(blocks: range, branches: int) -> pd.DataFrame:
    rows = []
    for block_id in blocks:
        for landmark in (20, 35, 50, 65, 80):
            for branch in range(branches):
                rows.append(
                    {
                        "block_id": block_id,
                        "landmark": landmark,
                        "branch": branch,
                        "main_complete": 1,
                        "complete_horizon": 1,
                        "break_index": 0,
                        "event": 1,
                    }
                )
    return pd.DataFrame(rows)


def test_pilot_gate_counts_complete_world_blocks() -> None:
    full = _future_frame(range(24), 32)
    result = pilot_eligibility(full, 32, type(SMOKE_PROFILE)(
        **{**SMOKE_PROFILE.__dict__, "name": "full", "landmarks": (20, 35, 50, 65, 80), "pilot_branches": 32}
    ))
    assert result["complete_blocks"] == 24
    assert result["eligible"]

    incomplete = full.drop(full[(full["block_id"] == 0) & (full["landmark"] == 20)].index[:1])
    result = pilot_eligibility(incomplete, 32, type(SMOKE_PROFILE)(
        **{**SMOKE_PROFILE.__dict__, "name": "full", "landmarks": (20, 35, 50, 65, 80), "pilot_branches": 32}
    ))
    assert result["complete_blocks"] == 23
    assert not result["eligible"]


def test_preregistered_sensitivities_are_descriptive() -> None:
    futures = pd.DataFrame(
        [
            {"future_id": "a", "half": "A"},
            {"future_id": "b", "half": "B"},
        ]
    )
    sequence = [0.2, 0.95, 0.96, 0.97] + [0.98] * 12
    boundaries = pd.DataFrame(
        [
            {"future_id": future_id, "boundary": index, "similarity": similarity}
            for future_id in ("a", "b")
            for index, similarity in enumerate(sequence)
        ]
    )
    result = sensitivity_summaries(futures, boundaries, 0.9)
    assert result["non_rescuing"]
    assert set(result["variants"]) == {
        "raw_S_gt_0.9_F12_R3",
        "calibrated_F8_R3",
        "calibrated_F16_R3",
        "calibrated_F12_R2",
        "calibrated_F12_R4",
    }
    assert all(item["events"] == 2 for item in result["variants"].values())


def test_seed_domains_are_deterministic_and_separate() -> None:
    assert derive_seed("pilot", 1) == derive_seed("pilot", 1)
    assert len({derive_seed(domain, 1) for domain in ("mechanics", "calibration", "pilot", "confirmation")}) == 4
