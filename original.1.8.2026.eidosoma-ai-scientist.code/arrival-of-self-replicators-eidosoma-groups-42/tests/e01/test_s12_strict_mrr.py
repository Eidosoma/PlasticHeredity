from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from e01_information_dynamics.backends import InformationBackendError, run_phyid
from e01_strict_mrr import analysis as strict_analysis
from e01_strict_mrr.core import (
    MINIMUM_EFFECTIVE_SAMPLES,
    PREPROCESSING_IDS,
    PartitionLock,
    RunningStrictEstimator,
    action_null_envelope,
    build_baseline_specification,
    find_past_only_partition_lock,
    mapped_part_series,
    preprocess_states,
    score_action_candidates,
)
from e01_strict_mrr.intervention import _choose_action, intervention_seed_bundle


def test_full_scale_specification_is_explicit_historical_reconstruction() -> None:
    specification = build_baseline_specification()
    assert specification.n_species == 100
    assert specification.n_min == 40
    assert specification.n_max == 80
    assert specification.n_generations == 100
    assert specification.max_steps is None
    assert specification.update_kernel.value == "categorical_single_event"
    assert specification.fission_semantics.value == (
        "fixed_size_without_replacement_odd_discard"
    )
    assert specification.initial_state_semantics.value == "with_replacement_counts"
    assert sum(specification.rho) == 1.0


def test_preprocessing_retains_every_integer_state_and_inverts() -> None:
    rng = np.random.Generator(np.random.PCG64DXSM(120901))
    states = rng.multinomial(40, np.full(100, 0.01), size=64).astype(np.int64)
    result = preprocess_states(states)
    assert set(result.coordinates) == set(PREPROCESSING_IDS)
    assert all(value.shape == (64, 99) for value in result.coordinates.values())
    assert all(np.all(np.isfinite(value)) for value in result.coordinates.values())
    assert all(
        np.max(value) <= 1.0e-10 for value in result.maximum_inverse_errors.values()
    )
    assert np.array_equal(result.masses, np.full(64, 40))


def test_running_strict_scalar_matches_pinned_phyid_mmi_and_ccs() -> None:
    rng = np.random.Generator(np.random.PCG64DXSM(120902))
    count = 640
    source = np.empty(count, dtype=np.float64)
    target = np.empty(count, dtype=np.float64)
    source[0] = 0.0
    target[0] = 0.0
    for index in range(1, count):
        source[index] = 0.72 * source[index - 1] + rng.normal()
        target[index] = (
            0.31 * source[index - 1] + 0.55 * target[index - 1] + rng.normal()
        )
    strict = RunningStrictEstimator.from_series(source, target).estimate()
    assert strict.status == "ELIGIBLE_NUMERIC_STRICT_EXPANDING"
    assert strict.n_eff == count - 1
    for redundancy in ("MMI", "CCS"):
        reference = run_phyid(
            source, target, tau=1, kind="gaussian", redundancy=redundancy
        )
        assert reference.status == "ELIGIBLE"
        means = reference.means()
        assert means is not None
        assert abs(strict.value - means["paperEquationAggregateDirect"]) <= 1.0e-10
        assert (
            abs(
                means["paperEquationAggregateFromAtoms"]
                - means["paperEquationAggregateDirect"]
            )
            <= 1.0e-10
        )


def test_partition_lock_is_past_only_and_feature_relabel_equivariant() -> None:
    rng = np.random.Generator(np.random.PCG64DXSM(120903))
    rows = 620
    factor_a = rng.normal(size=(rows, 1))
    factor_b = rng.normal(size=(rows, 1))
    coordinates = np.concatenate(
        [
            factor_a + 0.1 * rng.normal(size=(rows, 49)),
            factor_b + 0.1 * rng.normal(size=(rows, 50)),
        ],
        axis=1,
    )
    kinds = tuple(
        "post_fission" if index in (512, 600) else "molecular_event"
        for index in range(rows)
    )
    generations = np.arange(rows, dtype=np.int64) // 40
    molecular = np.arange(rows, dtype=np.int64)
    lock = find_past_only_partition_lock(
        coordinates,
        preprocessing_id=PREPROCESSING_IDS[0],
        observation_kinds=kinds,
        generations=generations,
        molecular_steps=molecular,
        estimator_rng=np.random.Generator(np.random.PCG64DXSM(120904)),
    )
    assert lock.status == "ELIGIBLE_LOCKED"
    assert lock.observation_index == MINIMUM_EFFECTIVE_SAMPLES
    assert lock.replay_minimum_ari == 1.0
    assert lock.replay_maximum_objective_error <= 1.0e-10
    assert lock.minimum_side_fraction >= 0.1


def _fixture_lock(preprocessing_id: str) -> PartitionLock:
    part_a = tuple(range(49))
    return PartitionLock(
        status="ELIGIBLE_LOCKED",
        reason=None,
        preprocessing_id=preprocessing_id,
        observation_index=512,
        generation=13,
        molecular_step=500,
        part_a=part_a,
        part_b=tuple(range(49, 99)),
        partition_id=f"fixture::{preprocessing_id}",
        objective=0.0,
        relative_eigengap=0.1,
        minimum_side_fraction=49 / 99,
        replay_maximum_objective_error=0.0,
        replay_minimum_ari=1.0,
        history=(),
    )


def test_complete_candidate_scoring_and_null_are_reproducible() -> None:
    rng = np.random.Generator(np.random.PCG64DXSM(120905))
    states = rng.multinomial(40, np.full(100, 0.01), size=620).astype(np.int64)
    preprocessing = preprocess_states(states)
    locks = {key: _fixture_lock(key) for key in PREPROCESSING_IDS}
    rows_first = score_action_candidates(
        states[-1],
        preprocessing_coordinates=preprocessing.coordinates,
        locks=locks,
    )
    rows_second = score_action_candidates(
        states[-1],
        preprocessing_coordinates=preprocessing.coordinates,
        locks=locks,
    )
    assert rows_first == rows_second
    expected_candidates = 1 + 100 + int(np.count_nonzero(states[-1]))
    assert len(rows_first) == expected_candidates * 2
    assert all(
        row["status"] == "ELIGIBLE_NUMERIC_STRICT_EXPANDING" for row in rows_first
    )
    subset = [
        row for row in rows_first if row["preprocessingId"] == PREPROCESSING_IDS[0]
    ]
    first = action_null_envelope(
        subset,
        direction="max",
        rng=np.random.Generator(np.random.PCG64DXSM(120906)),
    )
    second = action_null_envelope(
        subset,
        direction="max",
        rng=np.random.Generator(np.random.PCG64DXSM(120906)),
    )
    assert first == second
    assert first["status"] == "ELIGIBLE"
    assert first["families"] == 4096


def test_s12_core_does_not_import_failed_fixed_window_packages() -> None:
    assert "e01_time_localized_phir" not in sys.modules
    assert "e01_time_localized_phir_repair" not in sys.modules


def test_group_mean_mapping_produces_two_scalar_series() -> None:
    coordinates = np.arange(20 * 99, dtype=np.float64).reshape(20, 99)
    source, target = mapped_part_series(coordinates, tuple(range(49)))
    assert source.shape == target.shape == (20,)
    assert np.array_equal(source, np.mean(coordinates[:, :49], axis=1))
    assert np.array_equal(target, np.mean(coordinates[:, 49:], axis=1))


def test_intervention_seed_pairing_shares_only_gard_streams() -> None:
    max_bundle = intervention_seed_bundle(3, "max").to_payload()
    control_bundle = intervention_seed_bundle(3, "control").to_payload()
    common = {
        "catalytic_matrix",
        "initial_state",
        "event",
        "waiting_time",
        "fission",
        "daughter_selection",
    }
    for purpose in common:
        assert (
            max_bundle["streams"][purpose]["seedMaterialHex"]
            == control_bundle["streams"][purpose]["seedMaterialHex"]
        )
    for purpose in ("intervention", "estimator", "machine_learning"):
        assert (
            max_bundle["streams"][purpose]["seedMaterialHex"]
            != control_bundle["streams"][purpose]["seedMaterialHex"]
        )


def test_candidate_tie_is_suppressed_without_index_tiebreak() -> None:
    rows = []
    for preprocessing_id in PREPROCESSING_IDS:
        for candidate_id, score in (("noop", 1.0), ("add:0", 1.0), ("add:1", 0.0)):
            rows.append(
                {
                    "candidateId": candidate_id,
                    "actionClass": "noop" if candidate_id == "noop" else "addition",
                    "preprocessingId": preprocessing_id,
                    "status": "ELIGIBLE_NUMERIC_STRICT_EXPANDING",
                    "score": score,
                    "candidateState": [1] * 100,
                }
            )
    decision, diagnostics = _choose_action(
        rows,
        [dict(row) for row in rows],
        direction="max",
        intervention_rng=np.random.Generator(np.random.PCG64DXSM(120907)),
    )
    assert decision["status"] == "INELIGIBLE_ACTION_NOT_SEPARABLE"
    assert decision["reason"] == "MULTIPLE_CANDIDATES_WITHIN_NUMERICAL_TIE_TOLERANCE"
    assert diagnostics == []


def test_nonfinite_pinned_source_failure_is_retained_not_imputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_source(*args: object, **kwargs: object) -> None:
        raise InformationBackendError("Backend returned a nonfinite decomposition.")

    monkeypatch.setattr(strict_analysis, "run_phyid", fail_source)
    source = np.linspace(-1.0, 1.0, 640)
    result, summary, local = strict_analysis._source_values(
        source, source[::-1], redundancy="E01-S10-REDUNDANCY-MMI-v1.0.0"
    )
    assert result is None
    assert summary["status"] == "INELIGIBLE"
    assert summary["reason"].startswith("PINNED_PHYID_SOURCE_FAILURE::")
    assert local.size == 0


def test_sparse_interventions_remain_underdetermined() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/e01/run_s12_strict_mrr.py"
    spec = importlib.util.spec_from_file_location("run_s12_strict_mrr_test", script)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    association = {}
    for preprocessing_id in PREPROCESSING_IDS:
        row = {
            "preprocessingId": preprocessing_id,
            "definedTrajectoryCount": 11,
            "positiveTrajectoryCount": 1,
            "status": "ELIGIBLE",
            "spearmanRho": -0.1,
            "bootstrapLower95": -0.2,
            "bootstrapUpper95": -0.05,
            "circularShiftPermutationPPositive": 0.9,
        }
        association[f"continuing_replication::{preprocessing_id}::MMI"] = row
    whole = [
        {
            "analysisType": "whole_aggregate_linear_trend",
            "preprocessingId": preprocessing_id,
            "redundancyId": redundancy_id,
            "status": "INELIGIBLE",
            "pValue": None,
        }
        for preprocessing_id in PREPROCESSING_IDS
        for redundancy_id in ("MMI", "CCS")
    ]
    gate = {"success": True, "selectedMatrixIndices": list(range(6))}
    metrics = [
        {
            "matrixIndex": matrix_index,
            "condition": condition,
            "separableActionsApplied": 0,
            "status": "ELIGIBLE",
            "contrast": None,
        }
        for matrix_index in range(6)
        for condition in ("max", "control", "min")
    ]
    metrics.extend(
        {
            "matrixIndex": None,
            "condition": None,
            "separableActionsApplied": None,
            "status": None,
            "contrast": contrast,
            "positivePairCount": 0,
            "bootstrapLower95": 0.0,
            "bootstrapUpper95": 0.0,
        }
        for contrast in ("max_minus_control", "control_minus_min")
    )
    claims = runner.classify_claims(association, whole, gate, metrics)
    by_id = {row["claimId"]: row for row in claims}
    for claim_id in ("E01-C046", "E01-C054", "E01-C058", "E01-C059"):
        assert by_id[claim_id]["status"] == "UNDERDETERMINED"
