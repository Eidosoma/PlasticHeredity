from __future__ import annotations

import numpy as np
import pytest

from plastic_heredity import intervention_replication as base
from plastic_heredity.experiment import StateCase
from plastic_heredity.intervention_metrics import generate_inference_draws
from plastic_heredity.intervention_p3b_dose_bridge import (
    ARMS,
    BOOTSTRAP_REPETITIONS,
    BRANCHES,
    FABLE_LOOSEN_FACTOR,
    FABLE_TIGHTEN_FACTOR,
    GENERALIZATION_LANDMARKS,
    LABEL,
    LANDMARKS,
    MATRICES,
    RANDOMIZATION_REPETITIONS,
    SEED_DOMAINS,
    SMALL_LOOSEN_FACTOR,
    SMALL_TIGHTEN_FACTOR,
    _future_seed,
    _present_block,
    _protocol,
    add_replay_gates,
    compute_bridge_inference,
    phase_spec,
    select_surgeries,
    validation_checks,
)


@pytest.fixture(scope="session")
def validated() -> dict:
    return validation_checks()


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    composition = np.asarray([4, 0, 2, 3], dtype=np.int64)
    beta = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float64,
    )
    return composition, beta


def test_complete_validation_passes_without_scientific_outcomes(validated: dict) -> None:
    assert validated["all_checks_passed"] is True
    assert validated["scientific_cohort_generated"] is False
    assert validated["scientific_effect_sizes_computed"] is False
    assert validated["check_count"] >= 37


def test_registered_targeted_contract_is_exact_and_log_symmetric() -> None:
    composition, beta = _fixture()
    surgeries = dict(
        zip(
            ARMS,
            select_surgeries(composition, beta, np.random.default_rng(11)),
            strict=True,
        )
    )
    _present, flat, before = _present_block(composition, beta)
    expected = {
        "SMALL_LOOSEN": SMALL_LOOSEN_FACTOR,
        "SMALL_TIGHTEN": SMALL_TIGHTEN_FACTOR,
        "FABLE_LOOSEN": FABLE_LOOSEN_FACTOR,
        "FABLE_TIGHTEN": FABLE_TIGHTEN_FACTOR,
    }
    for arm, factor in expected.items():
        surgery = surgeries[arm]
        assert surgery is not None
        assert np.array_equal(surgery.flat_indices, flat)
        assert np.array_equal(surgery.before, before)
        assert np.array_equal(surgery.after, before * factor)
    assert np.log(FABLE_TIGHTEN_FACTOR) == pytest.approx(
        -np.log(FABLE_LOOSEN_FACTOR), abs=1e-15
    )


def test_fable_target_pair_has_registered_frobenius_asymmetry() -> None:
    composition, beta = _fixture()
    surgeries = dict(
        zip(
            ARMS,
            select_surgeries(composition, beta, np.random.default_rng(12)),
            strict=True,
        )
    )
    _present, _flat, before = _present_block(composition, beta)
    norm = np.linalg.norm(before)
    tighten = surgeries["FABLE_TIGHTEN"]
    loosen = surgeries["FABLE_LOOSEN"]
    assert tighten is not None and loosen is not None
    assert tighten.observed_norm == pytest.approx(0.5 * norm, rel=1e-15)
    assert loosen.observed_norm == pytest.approx(norm / 3.0, rel=1e-15)
    assert loosen.observed_norm / tighten.observed_norm == pytest.approx(2.0 / 3.0)


def test_random_controls_are_exact_positive_pp_and_share_direction() -> None:
    composition, beta = _fixture()
    surgeries = dict(
        zip(
            ARMS,
            select_surgeries(composition, beta, np.random.default_rng(13)),
            strict=True,
        )
    )
    _present, flat, before = _present_block(composition, beta)
    block_norm = np.linalg.norm(before)
    small = surgeries["SMALL_RANDOM_PP"]
    fable = surgeries["FABLE_RANDOM_PP"]
    assert small is not None and fable is not None
    for surgery, target in ((small, 0.05 * block_norm), (fable, 0.5 * block_norm)):
        assert np.array_equal(surgery.flat_indices, flat)
        assert np.count_nonzero(surgery.after != surgery.before) == flat.size
        assert np.all(surgery.beta > 0.0)
        assert surgery.observed_norm == pytest.approx(target, abs=1e-12)
        assert np.log(surgery.after / surgery.before).sum() == pytest.approx(
            0.0, abs=1e-12
        )
    small_log = np.log(small.after / small.before)
    fable_log = np.log(fable.after / fable.before)
    assert small_log / np.linalg.norm(small_log) == pytest.approx(
        fable_log / np.linalg.norm(fable_log), abs=1e-12
    )


def test_targeted_surgery_is_permutation_equivariant() -> None:
    composition, beta = _fixture()
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    original = dict(
        zip(
            ARMS,
            select_surgeries(composition, beta, np.random.default_rng(14)),
            strict=True,
        )
    )
    permuted = dict(
        zip(
            ARMS,
            select_surgeries(
                composition[permutation],
                beta[np.ix_(permutation, permutation)],
                np.random.default_rng(15),
            ),
            strict=True,
        )
    )
    for arm in (
        "SMALL_LOOSEN",
        "SMALL_TIGHTEN",
        "FABLE_LOOSEN",
        "FABLE_TIGHTEN",
    ):
        assert original[arm] is not None and permuted[arm] is not None
        assert np.array_equal(
            permuted[arm].beta,
            original[arm].beta[np.ix_(permutation, permutation)],
        )


def test_protocol_freezes_fresh_budget_contract_and_phase_specific_gate() -> None:
    protocol = _protocol()
    assert protocol["design"]["matrices"] == 80
    assert protocol["design"]["states"] == 960
    assert protocol["design"]["primary_futures"] == 215_040
    assert protocol["design"]["replay_futures"] == 215_040
    assert protocol["design"]["primary_landmark"] == 60
    assert protocol["design"]["generalization_landmarks"] == list(
        GENERALIZATION_LANDMARKS
    )
    assert protocol["reason"]["external_fable_contract"]["tighten"] == (
        "beta[P,P] *= 1.5"
    )
    assert protocol["primary_inference"]["cr1_only_gates_excluded"] == [
        "each target arm separately differs from NOOP",
        "random effect no greater than 25% of target contrast",
    ]


def test_seed_domains_are_fresh_and_future_keys_are_arm_free() -> None:
    _composition, beta = _fixture()
    case = StateCase(
        "p3b-seed-fixture",
        "FIX",
        "02",
        9,
        60,
        beta,
        base._fixture_snapshot(),
    )
    spec = phase_spec()
    seeds = [_future_seed(spec, case, branch) for branch in range(4)]
    assert len(set(seeds)) == 4
    assert len(set(SEED_DOMAINS.values())) == len(SEED_DOMAINS)
    assert set(SEED_DOMAINS.values()).isdisjoint(base.SEED_DOMAINS.values())
    assert LABEL not in base.PHASE_LABEL.values()


def test_synthetic_registered_effect_passes_phase_specific_gates() -> None:
    composition, beta = _fixture()
    snapshot = base._fixture_snapshot()
    cases: list[StateCase] = []
    for matrix_id in range(MATRICES):
        for candidate in ("02", "03"):
            for landmark in LANDMARKS:
                cases.append(
                    StateCase(
                        f"synthetic-c{candidate}-m{matrix_id:03d}-g{landmark:03d}",
                        "SYNTHETIC",
                        candidate,
                        matrix_id,
                        landmark,
                        beta,
                        snapshot,
                    )
                )
    targets = np.zeros((len(cases), len(ARMS), BRANCHES), dtype=np.int8)
    index = {arm: position for position, arm in enumerate(ARMS)}
    targets[:, index["FABLE_LOOSEN"], :] = 1
    targets[:, index["SMALL_LOOSEN"], 0:4] = 1
    targets[:, index["SMALL_LOOSEN"], BRANCHES // 2 : BRANCHES // 2 + 4] = 1
    predictions = np.full((len(cases), len(ARMS)), 0.5, dtype=np.float64)
    draws = generate_inference_draws(
        MATRICES,
        BOOTSTRAP_REPETITIONS,
        RANDOMIZATION_REPETITIONS,
        np.random.default_rng(1001),
        np.random.default_rng(1002),
    )
    metrics, matrix_rows = compute_bridge_inference(
        cases, targets, predictions, draws
    )
    add_replay_gates(metrics, True)
    assert len(matrix_rows) == 2 * 4 * MATRICES
    assert metrics["primary_replication_gate_pass"] is True
    assert metrics["five_landmark_generalization_gate_pass"] is True
    assert metrics["landmark60_two_dose_gate_pass"] is True
    assert metrics["five_landmark_two_dose_gate_pass"] is True
    for scope in (metrics["primary"], metrics["generalization"]):
        for cell in scope["cells"]:
            assert cell["contrasts"]["fable_effect"]["estimate"] == 1.0
            assert cell["contrasts"]["small_effect"]["estimate"] == 0.25
            assert cell["fable_random_noop_equivalence"]["tost_equivalent"] is True

