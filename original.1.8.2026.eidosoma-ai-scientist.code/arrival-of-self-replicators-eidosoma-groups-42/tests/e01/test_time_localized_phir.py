from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e01_information_dynamics import ATOM_IDS, strict_sample_gate
from e01_time_localized_phir import (
    calibrated_means,
    fixed_window_index,
    map_partition,
    partition_ari,
    run_small_window_phiid,
    sliding_endpoints,
    stable_partition_candidates,
    whole_trajectory_index,
)
from e01_time_localized_phir.estimator import oas_covariance
from e01_time_localized_phir.partition import select_partition_candidate
from e01_time_localized_phir.synthetic import (
    ccs_population_oracle,
    highdim_independent_null,
    independent_white,
    planted_two_block_ar,
    redundant_covariance,
)

CONFIG_PATH = REPOSITORY_ROOT / "configs/e01/s11_time_localized_phir_preregistration.yaml"
EXPECTED_CONFIG_SHA256 = "1c21ad91927929626edb6b2e14dfa745674decbaa89c3ac57f2cfdc678458f40"


def _means(data: np.ndarray, tau: int = 1, redundancy: str = "MMI") -> dict:
    result = run_small_window_phiid(
        data[:, 0], data[:, 1], tau=tau, redundancy=redundancy  # type: ignore[arg-type]
    )
    assert result.status == "ELIGIBLE", result.reason
    means = result.means()
    assert means is not None
    return means


def test_preregistration_is_frozen_complete_and_does_not_relax_s10() -> None:
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == EXPECTED_CONFIG_SHA256
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert config["researchStepId"] == "S11"
    assert config["status"] == "FROZEN_BEFORE_ANY_CANONICAL_S11_BENCHMARK_OUTCOME"
    assert config["scopeBoundary"]["nextStepForbidden"] == "S12"
    pairs = config["fixedWindowGrid"]["pairs"]
    assert len(pairs) == 16
    assert len({(item["windowLength"], item["lag"]) for item in pairs}) == 16
    assert all(item["effectiveSampleCount"] == item["windowLength"] - item["lag"] for item in pairs)
    assert max(item["effectiveSampleCount"] for item in pairs) == 255
    assert config["immutableS10Boundary"]["minimumEffectiveSamples"] == 512
    assert config["immutableS10Boundary"]["action"] == "DO_NOT_RELAX_RENAME_OR_BACKPORT"
    assert len(config["failureInjections"]) == 12
    for item in config["frozenInputs"]:
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]


@pytest.mark.parametrize("window,tau", [(32, 8), (64, 4), (128, 2), (256, 1)])
@pytest.mark.parametrize("redundancy", ["MMI", "CCS"])
def test_small_window_branch_is_finite_and_closes_lattice(
    window: int, tau: int, redundancy: str
) -> None:
    fixture = independent_white(
        pair_id=f"test-{window}-{tau}", replicate_index=0, length=window, domain="unit"
    )
    result = run_small_window_phiid(
        fixture.data[:, 0], fixture.data[:, 1], tau=tau, redundancy=redundancy  # type: ignore[arg-type]
    )
    assert result.status == "ELIGIBLE", result.reason
    assert result.effective_sample_count == window - tau
    means = result.means()
    assert means is not None
    assert abs(means["latticeClosureError"]) <= 5e-10
    assert abs(means["paperEquationClosureError"]) <= 5e-10
    assert all(np.isfinite(value).all() for value in result.atoms.values())  # type: ignore[union-attr]
    assert min(fold["evaluationRows"] for fold in result.diagnostics["folds"]) >= 6


def test_affine_and_source_target_relabel_invariance() -> None:
    fixture = independent_white(
        pair_id="affine", replicate_index=2, length=64, domain="unit"
    ).data
    transformed = fixture.copy()
    transformed[:, 0] = 17.0 * transformed[:, 0] + 23.0
    transformed[:, 1] = -0.125 * transformed[:, 1] + 5.0
    original = _means(fixture, tau=4)
    affine = _means(transformed, tau=4)
    np.testing.assert_allclose(
        [original["atomMeans"][atom] for atom in ATOM_IDS],
        [affine["atomMeans"][atom] for atom in ATOM_IDS],
        atol=1e-9,
        rtol=1e-9,
    )
    swapped = _means(fixture[:, ::-1], tau=4)
    swap = {"r": "r", "x": "y", "y": "x", "s": "s", "t": "t"}
    for atom in ATOM_IDS:
        mapped = "".join(swap[value] for value in atom)
        assert original["atomMeans"][atom] == pytest.approx(
            swapped["atomMeans"][mapped], abs=1e-9, rel=1e-9
        )


def test_null_centering_preserves_both_linear_closures() -> None:
    first = independent_white(pair_id="cal", replicate_index=0, length=64, domain="unit").data
    second = independent_white(pair_id="cal", replicate_index=1, length=64, domain="unit").data
    calibrated = calibrated_means(_means(first), _means(second))
    assert abs(calibrated["latticeClosureError"]) <= 5e-10
    assert abs(calibrated["paperEquationClosureError"]) <= 5e-10


def test_oas_regularization_repairs_a_singular_covariance_without_fallback() -> None:
    values = np.column_stack([np.arange(18.0)] * 4)
    standardized = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)
    unregularized, raw = oas_covariance(standardized, shrinkage_multiplier=0.0)
    regularized, oas = oas_covariance(standardized, shrinkage_multiplier=1.0)
    assert raw["minimumEigenvalue"] <= 1e-10
    assert oas["minimumEigenvalue"] > 1e-10
    assert np.linalg.matrix_rank(unregularized) == 1
    assert np.linalg.matrix_rank(regularized) == 4


def test_planted_partition_is_recovered_and_null_fails_closed() -> None:
    signal = planted_two_block_ar(
        pair_id="partition", replicate_index=0, length=32, dimension=99
    )
    rng = np.random.Generator(np.random.PCG64DXSM(9911))
    result = stable_partition_candidates(signal.data, tau=8, rng=rng)
    assert result.status == "ELIGIBLE", result.reason
    assert result.selected_part_a is not None and signal.planted_part_a is not None
    assert partition_ari(result.selected_part_a, signal.planted_part_a, 99) >= 0.95
    for mapping in ("zscore_group_mean", "zscore_pc1"):
        first, second, diagnostics = map_partition(
            signal.data, result.selected_part_a, mapping=mapping  # type: ignore[arg-type]
        )
        assert first.shape == second.shape == (32,)
        assert diagnostics["mappingId"].startswith("E01-S11-PARTMAP")
        winner, scores = select_partition_candidate(
            signal.data,
            result,
            tau=8,
            mapping=mapping,  # type: ignore[arg-type]
            objective="synchronous_mi",
            normalization="none",
        )
        assert winner["status"] == "ELIGIBLE"
        assert scores and all(item["status"] == "ELIGIBLE" for item in scores)

    null = highdim_independent_null(
        pair_id="partition-null", replicate_index=0, length=32, dimension=99
    )
    rejected = stable_partition_candidates(
        null.data, tau=8, rng=np.random.Generator(np.random.PCG64DXSM(9922))
    )
    assert rejected.status == "INELIGIBLE"
    winner, scores = select_partition_candidate(
        null.data,
        rejected,
        tau=8,
        mapping="zscore_group_mean",
        objective="synchronous_mi",
        normalization="none",
    )
    assert winner["normalizedObjective"] is None
    assert scores == []


def test_feature_permutation_maps_partition_back_exactly() -> None:
    signal = planted_two_block_ar(
        pair_id="permutation", replicate_index=0, length=64, dimension=100
    )
    original = stable_partition_candidates(
        signal.data, tau=4, rng=np.random.Generator(np.random.PCG64DXSM(11))
    )
    permutation = np.random.Generator(np.random.PCG64DXSM(12)).permutation(100)
    changed = stable_partition_candidates(
        signal.data[:, permutation],
        tau=4,
        rng=np.random.Generator(np.random.PCG64DXSM(11)),
    )
    assert original.status == changed.status == "ELIGIBLE"
    mapped_back = tuple(sorted(int(permutation[index]) for index in changed.selected_part_a))  # type: ignore[union-attr]
    assert partition_ari(original.selected_part_a, mapped_back, 100) == 1.0  # type: ignore[arg-type]


def test_temporal_indices_never_access_future_rows() -> None:
    index = fixed_window_index(window_end=255, window_length=32, lag=8)
    assert index.effective_sample_count == 24
    assert index.future_index_max == index.window_end
    assert index.to_payload()["usesFutureBeyondWindowEnd"] is False
    endpoints = sliding_endpoints(2048, 256)
    assert endpoints[0] == 255 and endpoints[-1] == 2047
    whole = whole_trajectory_index(total_length=2048, lag=8)
    assert whole["prospective"] is False
    assert whole["scopeLabel"] == "NON_PROSPECTIVE_WHOLE_TRAJECTORY_DESCRIPTION"


def test_s10_strict_gate_remains_unchanged_and_rejects_fixed_windows() -> None:
    fixture = independent_white(pair_id="strict", replicate_index=0, length=256, domain="unit").data
    gate = strict_sample_gate(fixture[:, 0], fixture[:, 1], tau=1, kind="gaussian")
    assert gate["specificationId"] == "E01-S10-SAMPLE-GATE-STRICT-v1.0.0"
    assert gate["status"] == "INELIGIBLE"
    assert gate["reason"] == "INSUFFICIENT_EFFECTIVE_SAMPLES"


def test_ccs_population_oracle_is_deterministic() -> None:
    covariance = redundant_covariance(2)
    first = ccs_population_oracle(covariance, scramble_seed=1102, power=10)
    second = ccs_population_oracle(covariance, scramble_seed=1102, power=10)
    assert first == second


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy not available in this interpreter")
def test_cpu_gpu_small_window_agreement() -> None:
    fixture = independent_white(pair_id="gpu", replicate_index=0, length=64, domain="unit").data
    cpu = run_small_window_phiid(fixture[:, 0], fixture[:, 1], tau=4, redundancy="MMI")
    gpu = run_small_window_phiid(
        fixture[:, 0], fixture[:, 1], tau=4, redundancy="MMI", backend="cupy"
    )
    assert cpu.status == gpu.status == "ELIGIBLE"
    np.testing.assert_allclose(
        [cpu.means()["atomMeans"][atom] for atom in ATOM_IDS],  # type: ignore[index]
        [gpu.means()["atomMeans"][atom] for atom in ATOM_IDS],  # type: ignore[index]
        atol=1e-9,
        rtol=1e-8,
    )
