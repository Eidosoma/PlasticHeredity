from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import FissionRecord, advance_fission, generate_beta, generate_initial_composition
from reviewer_lineage_identity_response.followup_dynamic_regime_cleanroom.regime_core import scaled_config
from reviewer_lineage_identity_response.followup_strict8_switch_lock_cleanroom import run_campaign
from reviewer_lineage_identity_response.followup_strict8_switch_lock_cleanroom.switch_lock_core import (
    ARMS,
    LockArm,
    Trajectory,
    phase_aligned_similarity,
    score_prospective_event,
    shuffled_writer_signal,
    simulate_lock_future,
    wave_amplitude,
)


def _record(parent: np.ndarray, daughter: np.ndarray, h: float) -> FissionRecord:
    return FissionRecord(parent=parent.copy(), daughter=daughter.copy(), h=h, growth_steps=1)


def test_strict_extension_uses_third_daughter_as_b() -> None:
    anchor = np.asarray([40, 0, 0, 0], dtype=np.int64)
    form = np.asarray([0, 40, 0, 0], dtype=np.int64)
    records = [_record(anchor, form, 0.4)]
    daughters = []
    for index in range(8):
        daughter = form.copy()
        daughter[1] -= index % 2
        daughter[2] += index % 2
        daughters.append(daughter)
        records.append(_record(form, daughter, 0.95))
    outcome = score_prospective_event(records, 4)
    assert outcome.f12
    assert outcome.strict_extension
    assert outcome.any_strict8
    assert not outcome.f12_only
    assert outcome.run3_start == 1
    assert outcome.run3_end == 3
    assert np.array_equal(outcome.b_state, daughters[2])
    assert not np.array_equal(outcome.b_state, daughters[-1])


def test_f12_only_excludes_any_later_strict_window() -> None:
    anchor = np.asarray([40, 0, 0, 0], dtype=np.int64)
    form = np.asarray([0, 40, 0, 0], dtype=np.int64)
    records = [_record(anchor, form, 0.4)]
    records.extend(_record(form, form, 0.95) for _ in range(3))
    records.extend(_record(form, form, 0.4) for _ in range(8))
    outcome = score_prospective_event(records, 4)
    assert outcome.f12
    assert outcome.f12_only
    assert not outcome.strict_extension
    assert not outcome.any_strict8


def test_no_break_is_not_a_donor() -> None:
    state = np.asarray([20, 20, 0, 0], dtype=np.int64)
    outcome = score_prospective_event([_record(state, state, 0.95) for _ in range(12)], 4)
    assert not outcome.f12
    assert not outcome.strict_extension
    assert not outcome.f12_only


def test_wave_has_registered_period_and_phase() -> None:
    assert [wave_amplitude(index) for index in range(1, 5)] == pytest.approx([1.0, 0.5, 0.0, 0.5])
    assert wave_amplitude(1, np.pi) == pytest.approx(0.0)
    assert wave_amplitude(5) == pytest.approx(1.0)


def test_shuffled_writer_preserves_mask_and_spectrum() -> None:
    signal = np.linspace(-1.0, 1.0, 8)
    mask = np.asarray([1, 1, 0, 1, 0, 1, 0, 0], dtype=bool)
    signal[~mask] = 0.0
    shuffled = shuffled_writer_signal(signal, mask, 93)
    assert np.all(shuffled[~mask] == 0.0)
    assert np.array_equal(np.sort(shuffled[mask]), np.sort(signal[mask]))
    assert np.array_equal(shuffled, shuffled_writer_signal(signal, mask, 93))


def test_phase_alignment_recovers_a_registered_cycle_shift() -> None:
    basis = np.eye(4, dtype=np.int64) * 40
    left_states = np.vstack([basis[index % 4] for index in range(8)])
    right_states = np.roll(left_states, 1, axis=0)
    left = Trajectory(left_states, np.ones(8), 8, "left")
    right = Trajectory(right_states, np.ones(8), 8, "right")
    assert phase_aligned_similarity(left, right, 8) == pytest.approx(1.0)


@pytest.mark.parametrize("candidate", ["02", "03"])
def test_control_arm_is_bitwise_frozen_simulator(candidate: str) -> None:
    config = GardConfig(max_growth_steps=1_000)
    beta = generate_beta(config, np.random.default_rng(771))
    initial = generate_initial_composition(config, np.random.default_rng(772))
    seed = 9020 + int(candidate)
    observed = simulate_lock_future(
        initial,
        initial,
        beta,
        config,
        scaled_config(config, 0.5),
        CANDIDATES[candidate],
        LockArm("control"),
        dynamics_seed=seed,
        shuffle_seed=5,
        horizon=5,
    )
    rng = np.random.default_rng(seed)
    current = initial.copy()
    expected = []
    expected_h = []
    for _ in range(5):
        record = advance_fission(current, beta, config, CANDIDATES[candidate], rng)
        expected.append(record.daughter.copy())
        expected_h.append(record.h)
        current = record.daughter
    assert np.array_equal(observed.states, np.asarray(expected))
    assert np.array_equal(observed.boundary_h, np.asarray(expected_h))


def test_matching_is_unique_and_uses_frozen_feature_distance() -> None:
    features = np.asarray(
        [
            [1, 3, 0.95, 0.80, 40],
            [5, 7, 0.91, 0.70, 41],
            [1, 3, 0.949, 0.801, 40],
            [5, 7, 0.909, 0.699, 41],
            [9, 9, 0.99, 0.84, 60],
        ],
        dtype=float,
    )
    selected = run_campaign._match_controls(
        [0, 1], [2, 3, 4], features, candidate="02", matrix_id=0, anchor_id="b1p0_l1p0"
    )
    assert selected == [2, 3]
    assert len(set(selected)) == 2


def test_arm_set_contains_release_and_specificity_controls() -> None:
    names = [arm.name for arm in ARMS]
    assert names == [
        "control",
        "quench",
        "wave",
        "quench_wave",
        "quench_static",
        "quench_shuffled",
        "quench_phase_pi",
        "pulse_release",
    ]
    assert ARMS[-1].release_after == 8


def test_tiers_are_nested_and_limits_are_ordered() -> None:
    assert run_campaign.TIERS["A"]["matrices"] > run_campaign.TIERS["B"]["matrices"] > run_campaign.TIERS["C"]["matrices"]
    assert run_campaign.TIERS["A"]["donor_lineages"] > run_campaign.TIERS["B"]["donor_lineages"] > run_campaign.TIERS["C"]["donor_lineages"]
    assert 0 < run_campaign.PROJECTION_BUDGET_SECONDS < run_campaign.SOFT_LIMIT_SECONDS < run_campaign.HARD_LIMIT_SECONDS


def test_seed_domains_are_disjoint() -> None:
    domains = ("donor.lineage", "future.dynamics", "future.shuffle", "benchmark", "smoke")
    values = {derive_seed(run_campaign.MASTER_SEED, domain, "02", 0, 0) for domain in domains}
    assert len(values) == len(domains)


def test_cleanroom_firewall_and_reporting_boundary() -> None:
    assert run_campaign._firewall_audit()["passed"]
    source = Path(run_campaign.__file__).read_text(encoding="utf-8")
    assert "from NewIdeas" not in source
    assert "import NewIdeas" not in source
    boundary = (run_campaign.TASK_ROOT / "REPORTING_BOUNDARY.md").read_text(encoding="utf-8")
    assert "not evidence for the current" in boundary
    assert "cannot rescue" in boundary


def test_atomic_checkpoint_round_trip() -> None:
    scratch = run_campaign.TASK_ROOT / "artifacts" / f"unit_test_scratch_{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=False)
    path = scratch / "checkpoint.npz"
    try:
        run_campaign._atomic_npz(path, values=np.arange(8, dtype=np.int16))
        assert np.array_equal(run_campaign._load_npz(path)["values"], np.arange(8, dtype=np.int16))
        assert not list(scratch.glob("*.tmp.*"))
    finally:
        if path.is_file():
            path.unlink()
        scratch.rmdir()


def test_array_replay_comparison_treats_nans_as_exact() -> None:
    left = np.asarray([1.0, np.nan, 3.0])
    assert run_campaign._arrays_equal(left, left.copy())
    right = left.copy()
    right[2] = 4.0
    assert not run_campaign._arrays_equal(left, right)

