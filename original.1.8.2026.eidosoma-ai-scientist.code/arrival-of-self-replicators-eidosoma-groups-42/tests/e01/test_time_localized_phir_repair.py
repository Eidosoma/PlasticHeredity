"""Focused tests for the separately versioned S11R validation branch."""

from __future__ import annotations

import numpy as np
import pytest

from e01_information_dynamics.validation import ATOM_IDS
from e01_time_localized_phir_repair import (
    highdim_independent_null,
    noisy_redundant_ar,
    partition_ari,
    planted_two_block_ar,
    repair_rng,
    run_wishart_local_phiid,
    threshold_component_partition,
)
from e01_time_localized_phir_repair.estimator import wishart_mean_correction
from e01_time_localized_phir_repair.partition import RepairPartitionError, _components


def _result(data: np.ndarray, *, tau: int = 4, redundancy: str = "MMI"):
    result = run_wishart_local_phiid(
        data[:, 0], data[:, 1], tau=tau, redundancy=redundancy
    )
    assert result.status == "ELIGIBLE", result.reason
    assert result.means() is not None
    return result


def test_wishart_correction_and_closure() -> None:
    fixture = noisy_redundant_ar(
        phase="development",
        pair_id="unit-estimator",
        replicate_index=0,
        length=64,
        domain="unit-estimator-source",
    )
    for redundancy in ("MMI", "CCS"):
        result = _result(fixture.data, redundancy=redundancy)
        means = result.means()
        assert means is not None
        assert abs(means["latticeClosureError"]) <= 5.0e-10
        assert abs(means["paperEquationClosureError"]) <= 5.0e-10
        assert all(np.all(np.isfinite(value)) for value in result.atoms.values())
    assert wishart_mean_correction(4, 24) > 0


def test_estimator_affine_and_source_relabel_invariance() -> None:
    fixture = noisy_redundant_ar(
        phase="development",
        pair_id="unit-invariance",
        replicate_index=0,
        length=128,
        domain="unit-invariance-source",
    ).data
    base = _result(fixture).means()
    affine_data = fixture * np.asarray([2.7, 0.4]) + np.asarray([11.0, -3.0])
    affine = _result(affine_data).means()
    assert base is not None and affine is not None
    assert (
        max(
            abs(base["atomMeans"][atom] - affine["atomMeans"][atom])
            for atom in ATOM_IDS
        )
        < 1.0e-9
    )
    swapped = _result(fixture[:, ::-1]).means()
    assert swapped is not None
    swap = {"r": "r", "x": "y", "y": "x", "s": "s", "t": "t"}
    for atom in ATOM_IDS:
        mapped = "".join(swap[value] for value in atom)
        assert base["atomMeans"][atom] == pytest.approx(
            swapped["atomMeans"][mapped], abs=1.0e-9, rel=1.0e-9
        )


def test_estimator_fails_closed_without_deletion_or_regularization() -> None:
    finite = np.arange(32, dtype=np.float64)
    bad = finite.copy()
    bad[4] = np.nan
    assert (
        run_wishart_local_phiid(bad, finite, tau=1, redundancy="MMI").reason
        == "NONFINITE_INPUT_NO_ROW_DELETION"
    )
    assert (
        run_wishart_local_phiid(
            finite[:24], finite[:24], tau=1, redundancy="MMI"
        ).reason
        == "EFFECTIVE_SAMPLE_COUNT_BELOW_24"
    )
    singular = run_wishart_local_phiid(finite, finite, tau=1, redundancy="MMI")
    assert singular.status == "INELIGIBLE"
    assert singular.reason == "SINGULAR_SAMPLE_COVARIANCE_NO_REGULARIZATION_FALLBACK"


@pytest.mark.parametrize("dimension", [8, 99, 100])
def test_threshold_partition_recovers_planted_components(dimension: int) -> None:
    fixture = planted_two_block_ar(
        phase="development",
        pair_id=f"unit-partition-{dimension}",
        replicate_index=0,
        length=64,
        dimension=dimension,
        domain="unit-partition-signal",
    )
    rng, _ = repair_rng(
        phase="development",
        domain="unit-partition-bootstrap",
        pair_id=f"unit-partition-{dimension}",
        replicate_index=0,
        dimension=dimension,
    )
    result = threshold_component_partition(fixture.data, tau=4, rng=rng)
    assert result.status == "ELIGIBLE", result.reason
    assert (
        partition_ari(result.selected_part_a, fixture.planted_part_a, dimension) == 1.0
    )


def test_threshold_partition_rejects_null_and_threshold_tie() -> None:
    fixture = highdim_independent_null(
        phase="development",
        pair_id="unit-partition-null",
        replicate_index=0,
        length=64,
        dimension=100,
        domain="unit-partition-null-source",
    )
    rng, _ = repair_rng(
        phase="development",
        domain="unit-partition-null-bootstrap",
        pair_id="unit-partition-null",
        replicate_index=0,
        dimension=100,
    )
    result = threshold_component_partition(fixture.data, tau=4, rng=rng)
    assert result.status == "INELIGIBLE"
    affinity = np.zeros((4, 4), dtype=np.float64)
    affinity[0, 1] = affinity[1, 0] = 0.90
    with pytest.raises(RepairPartitionError, match="THRESHOLD_EDGE_TIE"):
        _components(affinity)


def test_partition_feature_permutation_and_positive_affine_equivariance() -> None:
    dimension = 100
    pair = "unit-partition-equivariance"
    fixture = planted_two_block_ar(
        phase="development",
        pair_id=pair,
        replicate_index=0,
        length=64,
        dimension=dimension,
        domain="unit-equivariance-source",
    )

    def calculate(data: np.ndarray):
        rng, _ = repair_rng(
            phase="development",
            domain="unit-equivariance-bootstrap",
            pair_id=pair,
            replicate_index=0,
            dimension=dimension,
        )
        return threshold_component_partition(data, tau=4, rng=rng)

    base = calculate(fixture.data)
    transform_rng, _ = repair_rng(
        phase="development",
        domain="unit-equivariance-transform",
        pair_id=pair,
        replicate_index=0,
        dimension=dimension,
    )
    permutation = transform_rng.permutation(dimension)
    permuted = calculate(fixture.data[:, permutation])
    mapped = tuple(int(permutation[index]) for index in permuted.selected_part_a)
    assert partition_ari(base.selected_part_a, mapped, dimension) == 1.0
    scales = np.exp(transform_rng.uniform(-2.0, 2.0, dimension))
    shifts = transform_rng.normal(size=dimension)
    affine = calculate(fixture.data * scales + shifts)
    assert partition_ari(base.selected_part_a, affine.selected_part_a, dimension) == 1.0


def test_development_and_confirmation_seed_firewall() -> None:
    _, development = repair_rng(
        phase="development", domain="unit-seed", pair_id="unit", replicate_index=0
    )
    _, confirmation = repair_rng(
        phase="confirmation", domain="unit-seed", pair_id="unit", replicate_index=0
    )
    assert development["streamId"] != confirmation["streamId"]
    assert development["seedMaterialHex"] != confirmation["seedMaterialHex"]
    assert development["bitGenerator"] == confirmation["bitGenerator"] == "PCG64DXSM"
