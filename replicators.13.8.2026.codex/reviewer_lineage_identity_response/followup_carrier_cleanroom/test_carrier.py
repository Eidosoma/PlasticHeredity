from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import advance_fission, generate_beta, generate_initial_composition
from reviewer_lineage_identity_response.followup_carrier_cleanroom.carrier_core import (
    ArmPolicy,
    CarrierSetting,
    advance_carrier_fission,
    bootstrap_mean_ci,
    influence_mask,
    paired_bootstrap_ci,
    permutation_equivariance,
    random_mask,
    reservoir_field,
    score_trajectory,
    simulate_carrier_future,
    update_carrier,
    writer_signal,
)
from reviewer_lineage_identity_response.followup_carrier_cleanroom import run_campaign


def _fixture() -> tuple[GardConfig, np.ndarray, np.ndarray]:
    config = GardConfig(max_growth_steps=1_000)
    rng = np.random.default_rng(1317)
    return config, generate_beta(config, rng), generate_initial_composition(config, rng)


def test_writer_is_centered_bounded_and_masked() -> None:
    state = np.arange(1, 101, dtype=np.int64)
    mask = np.zeros(100, dtype=bool)
    mask[[1, 7, 23, 88]] = True
    signal = writer_signal(state, mask)
    assert signal.shape == (100,)
    assert np.max(np.abs(signal)) <= 1.0
    assert np.all(signal[~mask] == 0.0)
    assert np.any(signal[mask] != 0.0)


def test_reservoir_field_is_uniform_at_zero_and_normalized() -> None:
    zero = reservoir_field(np.zeros(100), 2.0)
    assert np.array_equal(zero, np.full(100, 0.01))
    field = reservoir_field(np.linspace(-1.0, 1.0, 100), 2.0)
    assert np.isclose(field.sum(), 1.0)
    assert np.all(field > 0.0)
    assert field[-1] > field[0]


@pytest.mark.parametrize("k", [8, 16, 32, 100])
def test_masks_have_exact_registered_size(k: int) -> None:
    _, beta, _ = _fixture()
    assert int(influence_mask(beta, k).sum()) == k
    assert int(random_mask(100, k, 812 + k).sum()) == k
    assert np.array_equal(random_mask(100, k, 812 + k), random_mask(100, k, 812 + k))


def test_ideal_carrier_update_is_bounded_and_deterministic() -> None:
    config, beta, state = _fixture()
    mask = influence_mask(beta, 32)
    setting = CarrierSetting(k=32, half_life=4, coupling=1.0, copy_mode="ideal")
    initial = writer_signal(state, mask)
    left = update_carrier(initial, state, setting, mask, np.random.default_rng(1), renewal=True)
    right = update_carrier(initial, state, setting, mask, np.random.default_rng(999), renewal=True)
    assert np.array_equal(left, right)
    assert np.max(np.abs(left)) <= 1.0
    assert np.all(left[~mask] == 0.0)


def test_nominal_carrier_update_is_seed_reproducible() -> None:
    _, beta, state = _fixture()
    mask = influence_mask(beta, 32)
    setting = CarrierSetting(k=32, half_life=8, coupling=2.0, copy_mode="nominal")
    initial = writer_signal(state, mask)
    left = update_carrier(initial, state, setting, mask, np.random.default_rng(41), renewal=True)
    right = update_carrier(initial, state, setting, mask, np.random.default_rng(41), renewal=True)
    assert np.array_equal(left, right)
    assert np.all(left[~mask] == 0.0)


@pytest.mark.parametrize("candidate", ["02", "03"])
def test_no_carrier_future_is_bitwise_base_simulator(candidate: str) -> None:
    config, beta, initial = _fixture()
    setting = CarrierSetting(k=100, half_life=4, coupling=2.0, copy_mode="nominal")
    mask = np.ones(100, dtype=bool)
    dynamics_seed = 2200 + int(candidate)
    readout, boundary_h, _, _ = simulate_carrier_future(
        initial,
        initial,
        None,
        beta,
        config,
        CANDIDATES[candidate],
        setting,
        mask,
        ArmPolicy("no_carrier", initial="zero", renewal=False, no_carrier=True),
        dynamics_seed=dynamics_seed,
        carrier_seed=991,
        horizon=5,
    )
    rng = np.random.default_rng(dynamics_seed)
    current = initial.copy()
    states = []
    expected_h = []
    for _ in range(5):
        record = advance_fission(current, beta, config, CANDIDATES[candidate], rng)
        states.append(record.daughter.copy())
        expected_h.append(record.h)
        current = record.daughter
    digest = hashlib.sha256()
    digest.update(np.asarray(states, dtype="<i8").tobytes(order="C"))
    digest.update((5).to_bytes(4, "little", signed=False))
    assert readout.state_digest == digest.hexdigest()
    assert np.array_equal(boundary_h, np.asarray(expected_h), equal_nan=True)


@pytest.mark.parametrize("candidate", ["02", "03"])
def test_zero_coupling_uses_exact_base_path(candidate: str) -> None:
    config, beta, initial = _fixture()
    rng_left = np.random.default_rng(8831)
    rng_right = np.random.default_rng(8831)
    base = advance_fission(initial, beta, config, CANDIDATES[candidate], rng_left)
    carried = advance_carrier_fission(
        initial,
        beta,
        config,
        CANDIDATES[candidate],
        rng_right,
        np.linspace(-1.0, 1.0, 100),
        0.0,
        reader=True,
    )
    assert np.array_equal(base.parent, carried.parent)
    assert np.array_equal(base.daughter, carried.daughter)
    assert base.h == carried.h
    assert base.growth_steps == carried.growth_steps


def test_joint_relabeling_preserves_writer_reader_and_mask_geometry() -> None:
    _, beta, state = _fixture()
    order = np.random.default_rng(77).permutation(100)
    mask = influence_mask(beta, 16)
    carrier = writer_signal(state, mask)
    beta_p, state_p, carrier_p, mask_p = permutation_equivariance(beta, state, carrier, mask, order)
    assert np.array_equal(writer_signal(state_p, mask_p), carrier[order])
    assert np.allclose(reservoir_field(carrier_p, 1.7), reservoir_field(carrier, 1.7)[order])
    assert np.array_equal(influence_mask(beta_p, 16), mask[order])


def test_strict_capture_and_extinction_readouts() -> None:
    target = np.zeros(100, dtype=np.int64)
    target[3] = 40
    daughters = np.repeat(target[None, :], 16, axis=0)
    carrier = writer_signal(target, np.ones(100, dtype=bool))
    score = score_trajectory(
        daughters,
        np.full(16, 0.95),
        carrier,
        target,
        None,
        np.ones(100, dtype=bool),
        observed=16,
    )
    assert score.capture_any_f16
    assert score.terminal8_f16 == 1
    assert score.terminal8_f32 == -1
    assert not score.departed


def test_multiform_pair_similarity_does_not_overflow_uint8() -> None:
    left = np.zeros(100, dtype=np.uint8)
    right = np.zeros(100, dtype=np.uint8)
    left[3] = 40
    right[3] = 40
    table = pd.DataFrame(
        [
            {"bank_index": 0, "lineage": 1, "final_B": json.dumps(left.tolist())},
            {"bank_index": 1, "lineage": 2, "final_B": json.dumps(right.tolist())},
        ]
    )
    assert run_campaign._pair_for_cell(table) is None


def test_bootstraps_are_seed_deterministic_and_paired() -> None:
    left = np.asarray([0.7, 0.8, 0.9])
    right = np.asarray([0.1, 0.2, 0.3])
    assert bootstrap_mean_ci(left, seed=9, repetitions=128) == bootstrap_mean_ci(left, seed=9, repetitions=128)
    point, lower, upper = paired_bootstrap_ci(left, right, seed=10, repetitions=128)
    assert point == pytest.approx(0.6)
    assert lower <= point <= upper


def test_semantic_seed_domains_do_not_collide() -> None:
    values = {
        derive_seed(run_campaign.MASTER_SEED, domain, "02", 11, 0)
        for domain in (
            "calibration.dynamics",
            "calibration.carrier",
            "confirmation.dynamics",
            "confirmation.carrier",
            "benchmark",
            "smoke",
        )
    }
    assert len(values) == 6


def test_cleanroom_firewall_rejects_code_paths_and_has_no_imports() -> None:
    audit = run_campaign._firewall_audit()
    assert audit["passed"]
    source = Path(run_campaign.__file__).read_text(encoding="utf-8")
    assert "from NewIdeas" not in source
    assert "import NewIdeas" not in source


def test_atomic_checkpoint_round_trip() -> None:
    scratch = run_campaign.TASK_ROOT / "artifacts" / f"unit_test_scratch_{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=False)
    path = scratch / "checkpoint.npz"
    try:
        run_campaign._atomic_npz(path, values=np.arange(7, dtype=np.int16))
        loaded = run_campaign._load_npz(path)
        assert np.array_equal(loaded["values"], np.arange(7, dtype=np.int16))
        assert not list(scratch.glob("*.tmp.*"))
    finally:
        if path.is_file():
            path.unlink()
        scratch.rmdir()


def test_walltime_constants_are_nested_and_restart_guarded() -> None:
    assert 0 < run_campaign.TIER_BUDGET_SECONDS < run_campaign.SOFT_LIMIT_SECONDS < run_campaign.HARD_LIMIT_SECONDS
    source = Path(run_campaign.__file__).read_text(encoding="utf-8")
    assert "active_cumulative_at_start" in source
    assert "recovered_interrupted_run_at_epoch" in source
