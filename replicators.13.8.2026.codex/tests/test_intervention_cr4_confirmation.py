from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from plastic_heredity import intervention_replication as base
from plastic_heredity.config import CohortConfig, ExperimentConfig
from plastic_heredity.experiment import StateCase, build_cohort
from plastic_heredity.intervention_cr4_confirmation import (
    ARMS,
    BRANCHES,
    EQUIVALENCE_MARGIN,
    LANDMARKS,
    LOOSEN_FACTOR,
    MATRICES,
    MINIMUM_CPU_BUDGET_HOURS,
    MINIMUM_FREE_DISK_BYTES,
    ORIGINAL_REGISTRATION,
    SEEDS,
    SURGERY_NORM_FRACTION,
    TIGHTEN_FACTOR,
    _future_seed,
    _geometry_audit,
    _global_selection_seed,
    _inference,
    _present_block,
    _readback,
    _topology_selection_seed,
    _write_inference_arrays,
    add_cr4_gate_fields,
    phase_spec,
    protocol,
    run_phase_batches,
    select_surgeries,
    validation_checks,
)
from plastic_heredity.intervention_p3c import catalytic_throughput
from plastic_heredity.intervention_metrics import generate_inference_draws


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    composition = np.asarray([4, 2, 0, 2], dtype=np.int64)
    beta = np.asarray(
        [
            [1.0, 4.0, 2.0, 3.0],
            [7.0, 1.0, 5.0, 2.0],
            [8.0, 4.0, 1.0, 9.0],
            [2.0, 3.0, 6.0, 1.0],
        ],
        dtype=np.float64,
    )
    return composition, beta


def _cell(*, equivalent: bool = True) -> dict:
    return {
        "contrasts": {
            "up_minus_down": {
                "estimate": 0.1,
                "bootstrap_ci95": (0.05, 0.15),
            }
        },
        "up_down_randomization_p_holm": 0.01,
        "random_noop_equivalence": {"tost_equivalent": equivalent},
    }


def test_full_cr4_design_and_future_count_are_frozen() -> None:
    frozen = protocol()
    assert MATRICES == 200
    assert BRANCHES == 64
    assert LANDMARKS == (20, 35, 50, 65, 80)
    assert ARMS == (
        "LOOSEN",
        "TIGHTEN",
        "GLOBAL_RANDOM_SURGERY",
        "THROUGHPUT_NEUTRAL_TOPOLOGY",
        "NOOP",
    )
    assert frozen["cohort"]["states"] == 2_000
    assert frozen["futures"]["primary_futures"] == 640_000
    assert frozen["futures"]["replay_futures"] == 640_000
    assert frozen["futures"]["halves"] == {"A": [0, 31], "B": [32, 63]}
    assert frozen["inference"]["global_random_noop_tost_margin"] == [
        -EQUIVALENCE_MARGIN,
        EQUIVALENCE_MARGIN,
    ]


def test_corrected_fable_pair_is_log_symmetric_but_frobenius_asymmetric() -> None:
    assert TIGHTEN_FACTOR == 1.5
    assert LOOSEN_FACTOR == 1.0 / 1.5
    assert np.isclose(
        np.log(TIGHTEN_FACTOR), -np.log(LOOSEN_FACTOR), rtol=0.0, atol=1e-15
    )
    assert abs(TIGHTEN_FACTOR - 1.0) == 0.5
    assert np.isclose(
        abs(LOOSEN_FACTOR - 1.0), 1.0 / 3.0, rtol=0.0, atol=1e-15
    )


def test_cr4_surgeries_are_exact_positive_and_deterministic() -> None:
    composition, beta = _fixture()
    left = select_surgeries(
        composition,
        beta,
        np.random.default_rng(11),
        np.random.default_rng(13),
    )
    right = select_surgeries(
        composition,
        beta,
        np.random.default_rng(11),
        np.random.default_rng(13),
    )
    by_name = dict(zip(ARMS, left, strict=True))
    _, pp_flat, pp_before = _present_block(composition, beta)
    target_norm = SURGERY_NORM_FRACTION * np.linalg.norm(pp_before)
    assert by_name["NOOP"] is None
    for first, second in zip(left[:-1], right[:-1], strict=True):
        assert first is not None and second is not None
        assert np.array_equal(first.beta, second.beta)
        assert np.all(first.beta > 0.0)
        assert first.observed_norm == second.observed_norm
        assert np.isclose(first.observed_norm, first.requested_norm, rtol=1e-12)
    loosen = by_name["LOOSEN"]
    tighten = by_name["TIGHTEN"]
    global_random = by_name["GLOBAL_RANDOM_SURGERY"]
    topology = by_name["THROUGHPUT_NEUTRAL_TOPOLOGY"]
    assert loosen is not None and tighten is not None
    assert global_random is not None and topology is not None
    assert set(loosen.flat_indices.tolist()) == set(pp_flat.tolist())
    assert set(tighten.flat_indices.tolist()) == set(pp_flat.tolist())
    assert np.array_equal(loosen.after, pp_before * LOOSEN_FACTOR)
    assert np.array_equal(tighten.after, pp_before * 1.5)
    assert len(set(global_random.flat_indices.tolist())) == pp_before.size
    assert np.isclose(global_random.observed_norm, target_norm, rtol=1e-12)
    assert set(topology.flat_indices.tolist()) == set(pp_flat.tolist())
    assert np.isclose(topology.observed_norm, target_norm, rtol=1e-12)
    assert np.isclose(
        catalytic_throughput(composition, topology.beta),
        catalytic_throughput(composition, beta),
        rtol=1e-11,
        atol=1e-12,
    )


def test_global_location_draw_does_not_use_present_type_identities() -> None:
    composition, beta = _fixture()
    permuted_composition = composition[[2, 1, 0, 3]]
    # Both states have three present types. With an identical RNG, the whole-
    # matrix location draw must therefore be identical even though P differs.
    left = select_surgeries(
        composition,
        beta,
        np.random.default_rng(19),
        np.random.default_rng(23),
    )[ARMS.index("GLOBAL_RANDOM_SURGERY")]
    right = select_surgeries(
        permuted_composition,
        beta,
        np.random.default_rng(19),
        np.random.default_rng(29),
    )[ARMS.index("GLOBAL_RANDOM_SURGERY")]
    assert left is not None and right is not None
    assert np.array_equal(left.flat_indices, right.flat_indices)


def test_singleton_contract_retains_state_as_all_arm_noop() -> None:
    _, beta = _fixture()
    surgeries = select_surgeries(
        np.asarray([8, 0, 0, 0], dtype=np.int64),
        beta,
        np.random.default_rng(31),
        np.random.default_rng(37),
    )
    assert surgeries == tuple(None for _ in ARMS)


def test_cr4_seeds_are_new_arm_free_and_selection_separated() -> None:
    assert len(SEEDS) == len(set(SEEDS.values()))
    assert set(SEEDS.values()).isdisjoint(base.SEED_DOMAINS.values())
    case = StateCase(
        "cr4-seed-fixture",
        "FIX",
        "02",
        9,
        20,
        np.eye(4),
        base._fixture_snapshot(),
    )
    spec = phase_spec()
    futures = [_future_seed(spec, case, branch) for branch in range(4)]
    assert len(set(futures)) == 4
    assert _global_selection_seed(spec, case) not in futures
    assert _topology_selection_seed(spec, case) not in futures
    assert _global_selection_seed(spec, case) != _topology_selection_seed(spec, case)


def test_cr4_gate_uses_strength_and_global_location_control_only() -> None:
    metrics = {"cells": [_cell() for _ in range(4)]}
    add_cr4_gate_fields(metrics)
    assert metrics["cr4_all_four_cells_scientific_pass"] is True
    for cell in metrics["cells"]:
        assert set(cell["cr4_registered_gates"]) == {
            "loosen_minus_tighten_positive",
            "loosen_minus_tighten_bootstrap_lower_positive",
            "holm_randomization_below_0_05",
            "global_random_tost_equivalent_to_noop",
        }


def test_cr4_gate_requires_global_random_equivalence_in_every_cell() -> None:
    cells = [_cell() for _ in range(4)]
    cells[1] = _cell(equivalent=False)
    metrics = {"cells": cells}
    add_cr4_gate_fields(metrics)
    assert metrics["cr4_all_four_cells_scientific_pass"] is False


def test_cr4_operational_boundary_is_frozen() -> None:
    frozen = protocol()["operational"]
    assert MINIMUM_CPU_BUDGET_HOURS == 20.0
    assert MINIMUM_FREE_DISK_BYTES == 3_000_000_000
    assert frozen["no_mid_phase_kill"] is True
    assert frozen["mandatory_review_stop_after_seal"] is True


def test_cr4_validation_generates_no_scientific_cohort() -> None:
    validated = validation_checks()
    assert validated["all_checks_passed"] is True
    assert validated["scientific_matrices_generated"] == 0
    assert validated["scientific_futures_generated"] == 0


def test_cr4_non_scientific_worker_checkpoint_replay_and_audit(
    tmp_path: Path,
) -> None:
    spec = replace(
        phase_spec(),
        role="non-scientific test fixture",
        matrices=1,
        branches=2,
        landmarks=(5,),
        cohort_seed=SEEDS["smoke_cohort"],
        global_selection_seed=SEEDS["smoke_global_selection"],
        topology_selection_seed=SEEDS["smoke_topology_selection"],
        future_seed=SEEDS["smoke_future"],
    )
    cohort = CohortConfig(1, 2, (5,))
    current_experiment = ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=12,
        master_seed=spec.cohort_seed,
        bootstrap_repetitions=8,
        permutation_repetitions=8,
    )
    cases = build_cohort(current_experiment, "INTCR4_TEST_SMOKE", cohort)
    model = ORIGINAL_REGISTRATION / "frozen_full_predictor.npz"
    generated = run_phase_batches(
        cases,
        current_experiment,
        spec,
        model,
        "non-scientific-test-registration",
        tmp_path / "generate",
        1,
        "generate",
    )
    replayed = run_phase_batches(
        cases,
        current_experiment,
        spec,
        model,
        "non-scientific-test-registration",
        tmp_path / "replay",
        1,
        "replay",
    )
    assert base.replay_audit(generated, replayed)[
        "state_edit_endpoint_and_process_digests_exact"
    ]
    _, _, audit = _geometry_audit(cases, generated, spec)
    assert audit["all_audits_pass"] is True


def test_cr4_inference_arrays_round_trip_with_separate_topology_family(
    tmp_path: Path,
) -> None:
    spec = replace(phase_spec(), matrices=4, branches=4, landmarks=(20,))
    cases = [
        StateCase(
            f"cr4-inference-c{candidate}-m{matrix_id}",
            "FIX",
            candidate,
            matrix_id,
            20,
            np.eye(4),
            base._fixture_snapshot(),
        )
        for candidate in ("02", "03")
        for matrix_id in range(4)
    ]
    targets = np.zeros((len(cases), len(ARMS), spec.branches), dtype=np.int8)
    targets[:, ARMS.index("LOOSEN"), :] = 1
    targets[::2, ARMS.index("THROUGHPUT_NEUTRAL_TOPOLOGY"), :] = 1
    arrays = {
        "targets": targets,
        "predictions": np.full((len(cases), len(ARMS)), 0.5, dtype=np.float64),
    }
    draws = generate_inference_draws(
        4,
        32,
        32,
        np.random.default_rng(41),
        np.random.default_rng(43),
    )
    metrics, rows, topology_rows, topology_arrays = _inference(
        cases, arrays, draws, spec
    )
    np.savez_compressed(tmp_path / "branch_arrays.npz", **arrays)
    _write_inference_arrays(
        tmp_path / "inference_arrays.npz", draws, metrics, topology_arrays
    )
    audit = _readback(
        tmp_path, cases, spec, metrics, rows, topology_rows
    )
    assert audit["primary_metrics_exact"] is True
    assert audit["matrix_effects_exact"] is True
    assert audit["topology_matrix_effects_exact"] is True
    assert metrics["topology"][
        "cannot_rescue_or_invalidate_primary_strength_gate"
    ] is True
