from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.simulator import FissionRecord, generate_beta, generate_initial_composition
from reviewer_lineage_identity_response.followup_plasticity_persistence_cleanroom import run_campaign
from reviewer_lineage_identity_response.followup_plasticity_persistence_cleanroom.campaign_core import (
    factorial_shapley,
    f12_decomposition,
    last8_coherence,
    reproducible_multiform,
    score_detailed_records,
    simulate_detailed_lineage,
    stable_components,
)


def _state(index: int, mass: int = 40) -> np.ndarray:
    values = np.zeros(100, dtype=np.int64)
    values[index] = mass
    return values


def _records(h_values: list[float], daughter: np.ndarray, anchor: np.ndarray) -> list[FissionRecord]:
    return [
        FissionRecord(parent=anchor.copy() if index == 0 else daughter.copy(), daughter=daughter.copy(), h=h, growth_steps=1)
        for index, h in enumerate(h_values)
    ]


def test_f12_and_strict8_components_are_separate_and_nested() -> None:
    h = [0.80] + [0.95] * 31
    outcome = score_detailed_records(_records(h, _state(2), _state(1, 80)), 100)
    assert outcome.break12
    assert outcome.recovery3_given_break
    assert outcome.f12
    assert outcome.first_run3_end == 3
    assert outcome.run8 and outcome.coherent8 and outcome.distinct8 and outcome.strict8
    assert np.array_equal(outcome.b_state, _state(2))
    assert np.array_equal(outcome.strict_b_state, _state(2))


def test_no_break_has_no_conditional_recovery_or_b_state() -> None:
    outcome = score_detailed_records(_records([0.95] * 32, _state(2), _state(2, 80)), 100)
    assert not outcome.break12
    assert not outcome.recovery3_given_break
    assert not outcome.f12
    assert not np.any(outcome.b_state)


def test_f12_decomposition_is_exact() -> None:
    total, supply, recovery = f12_decomposition(0.9, 0.8, 0.7, 0.6)
    assert total == pytest.approx(0.9 * 0.8 - 0.7 * 0.6)
    assert supply + recovery == pytest.approx(total)


def test_factorial_shapley_sums_to_corner_difference() -> None:
    weights = np.asarray([1.0, -2.0, 0.5, 4.0])
    values = {tuple(bits): float(np.dot(bits, weights)) for bits in np.ndindex(2, 2, 2, 2)}
    result = factorial_shapley(values)
    assert np.allclose(result, weights)
    assert result.sum() == pytest.approx(values[(1, 1, 1, 1)] - values[(0, 0, 0, 0)])


def test_factorial_shapley_rejects_missing_corner() -> None:
    with pytest.raises(ValueError):
        factorial_shapley({(0, 0): 0.0, (1, 1): 1.0})


def test_stable_multiform_requires_two_reproduced_separated_forms() -> None:
    blocks = np.zeros((40, 8, 100), dtype=np.uint8)
    for lineage in range(40):
        form = _state(3) if lineage % 4 in (0, 1) else _state(77)
        blocks[lineage] = np.repeat(form[None, :], 8, axis=0)
    passed, first, second = reproducible_multiform(blocks)
    assert passed
    assert first >= 2 and second >= 2
    assert len(stable_components(blocks)) == 2
    assert last8_coherence(blocks[0]) == pytest.approx(1.0)


def test_unstable_blocks_do_not_form_attractors() -> None:
    blocks = np.zeros((8, 8, 100), dtype=np.uint8)
    for lineage in range(8):
        for generation in range(8):
            blocks[lineage, generation] = _state(generation)
    assert stable_components(blocks) == []


def test_factorial_corners_match_candidate_contracts_and_paths() -> None:
    assert run_campaign._contract_corner_parity()
    config = GardConfig()
    rng = np.random.default_rng(55)
    beta = generate_beta(config, rng)
    for candidate, bits in (("02", (0, 0, 0, 0)), ("03", (1, 1, 1, 1))):
        contract = next(item for _, item_bits, item in run_campaign.CONTRACTS if item_bits == bits)
        left = simulate_detailed_lineage(beta, config, CANDIDATES[candidate], seed=77)
        right = simulate_detailed_lineage(beta, config, contract, seed=77)
        assert left.scalars() == right.scalars()
        assert np.array_equal(left.boundary_h, right.boundary_h)


def test_permutation_stranger_preserves_spectrum_and_mass() -> None:
    state = np.arange(100, dtype=np.int64) % 5
    permutation, _, _ = run_campaign._proposal_permutation(state, seed=91)
    stranger = state[permutation]
    assert stranger.sum() == state.sum()
    assert np.array_equal(np.sort(stranger), np.sort(state))


def test_grid_and_anchors_are_frozen_and_complete() -> None:
    assert len(run_campaign.GRID) == 21
    assert len(run_campaign.CONTRACTS) == 16
    assert set(run_campaign.ANCHORS) <= set(run_campaign.GRID_LOOKUP)


def test_fresh_matrix_domains_are_distinct() -> None:
    assert not np.array_equal(run_campaign._matrix_beta("surface", 0), run_campaign._matrix_beta("factorial", 0))


def test_cleanroom_firewall_and_claim_boundary() -> None:
    assert run_campaign._firewall_audit()["passed"]
    protocol = run_campaign._protocol_payload()
    assert protocol["reporting_boundary"]["not_current_preprint_evidence"]
    assert protocol["reporting_boundary"]["edge_of_chaos_not_a_success_criterion"]


def test_atomic_checkpoint_roundtrip() -> None:
    scratch = run_campaign.ARTIFACT_ROOT / f"unit_test_scratch_{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=False)
    path = scratch / "values.npz"
    try:
        run_campaign._atomic_npz(path, values=np.arange(11))
        assert np.array_equal(run_campaign._load_npz(path)["values"], np.arange(11))
    finally:
        if path.is_file():
            path.unlink()
        scratch.rmdir()


def test_elapsed_status_reader_does_not_mutate_active_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / "ledger.json"
    monkeypatch.setattr(run_campaign, "LEDGER_PATH", path)
    ledger = run_campaign._initial_ledger()
    ledger["active_started_epoch"] = time.time() - 2
    ledger["active_cumulative_at_start"] = 5.0
    path.write_text(json.dumps(ledger), encoding="utf-8")
    before = path.read_bytes()
    assert run_campaign._ledger_elapsed() >= 7.0
    assert path.read_bytes() == before


def test_walltime_tiers_are_nested() -> None:
    assert run_campaign.PROJECTION_BUDGET_SECONDS < run_campaign.SOFT_LIMIT_SECONDS < run_campaign.HARD_LIMIT_SECONDS
    assert run_campaign.TIERS["A"]["surface_matrices"] > run_campaign.TIERS["B"]["surface_matrices"] > run_campaign.TIERS["C"]["surface_matrices"]

