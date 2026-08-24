from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import loadmat

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e01_information_dynamics import (
    ATOM_IDS,
    aggregate_means,
    all_bipartitions,
    backend_identity,
    coupled_ar_covariance,
    discrete_exact_oracle,
    exact_redundant_pmf,
    exact_xor_pmf,
    gaussian_mmi_oracle,
    noisy_redundant_covariance,
    run_omegaid,
    run_phyid,
    strict_sample_gate,
)
from e01_information_dynamics.synthetic import (
    affine_transform,
    block_ar4,
    coupled_ar,
    discrete_relabel,
    independent_gaussian,
    redundant_discrete,
)
from e01_information_dynamics.validation import (
    InformationValidationError,
    exhaustive_partition_search,
    gaussian_partition_objective,
    spectral_partition,
)

CONFIG_PATH = (
    REPOSITORY_ROOT / "configs/e01/s10_information_dynamics_preregistration.yaml"
)
EXPECTED_CONFIG_SHA256 = (
    "5c54b8f88e8e8634a4b7f39783e3359084e25ce44cbed2291a141da85e19f3dd"
)


def test_preregistration_is_frozen_and_complete() -> None:
    assert (
        hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == EXPECTED_CONFIG_SHA256
    )
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert config["researchStepId"] == "S10"
    assert config["status"] == "FROZEN_BEFORE_ANY_CANONICAL_S10_BENCHMARK_OUTCOME"
    assert config["scopeBoundary"]["nextStepForbidden"] == "S11"
    assert len(config["atomCatalog"]["nativeAtoms"]) == 16
    assert {item["atomId"] for item in config["atomCatalog"]["nativeAtoms"]} == set(
        ATOM_IDS
    )
    assert len(config["syntheticDesign"]["systems"]) == 6
    assert len(config["failureInjections"]) == 10
    for item in config["frozenInputs"]:
        assert (
            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )


@pytest.mark.parametrize("redundancy", ["MMI", "CCS"])
def test_exact_discrete_oracles_recover_redundancy_and_xor_synergy(
    redundancy: str,
) -> None:
    states, probabilities = exact_redundant_pmf(0.1)
    redundant = discrete_exact_oracle(states, probabilities, redundancy=redundancy)  # type: ignore[arg-type]
    theory = np.log(2.0) + 0.9 * np.log(0.9) + 0.1 * np.log(0.1)
    assert redundant["totalMi"] == pytest.approx(theory, abs=1e-14)
    assert redundant["atomMeans"]["rtr"] == pytest.approx(theory, abs=1e-14)
    assert redundant["paperEquationAggregate"] == pytest.approx(-theory, abs=1e-14)
    assert (
        max(abs(value) for key, value in redundant["atomMeans"].items() if key != "rtr")
        < 1e-14
    )

    states, probabilities = exact_xor_pmf()
    xor = discrete_exact_oracle(states, probabilities, redundancy=redundancy)  # type: ignore[arg-type]
    assert xor["totalMi"] == pytest.approx(np.log(2.0), abs=1e-14)
    assert xor["atomMeans"]["stx"] == pytest.approx(np.log(2.0), abs=1e-14)
    assert xor["paperEquationAggregate"] == pytest.approx(np.log(2.0), abs=1e-14)
    assert (
        max(abs(value) for key, value in xor["atomMeans"].items() if key != "stx")
        < 1e-14
    )


def test_gaussian_population_oracles_have_preregistered_directions() -> None:
    independent = gaussian_mmi_oracle(np.eye(4))
    assert independent["totalMi"] == pytest.approx(0.0, abs=1e-14)
    assert all(
        value == pytest.approx(0.0, abs=1e-14)
        for value in independent["atomMeans"].values()
    )

    redundant = gaussian_mmi_oracle(noisy_redundant_covariance())
    assert redundant["atomMeans"]["rtr"] > 0
    assert redundant["paperEquationAggregate"] < 0

    directional = gaussian_mmi_oracle(coupled_ar_covariance())
    assert directional["miMeans"]["I_xtb"] > 0
    assert directional["miMeans"]["I_yta"] == pytest.approx(0.0, abs=1e-14)


def test_sample_gate_fails_closed_without_row_deletion_or_regularization() -> None:
    independent = independent_gaussian(0, n=1025).data
    eligible = strict_sample_gate(
        independent[:, 0], independent[:, 1], tau=1, kind="gaussian"
    )
    assert eligible["status"] == "ELIGIBLE"
    assert eligible["rowsDeleted"] == 0

    exact_copy = np.tile([[0.0, 0.0], [1.0, 1.0]], (300, 1))
    singular = strict_sample_gate(
        exact_copy[:, 0], exact_copy[:, 1], tau=1, kind="gaussian"
    )
    assert singular["status"] == "INELIGIBLE"
    assert singular["reason"] == "JOINT_COVARIANCE_RANK_DEFICIENT"

    nonfinite = independent.copy()
    nonfinite[7, 0] = np.nan
    rejected = strict_sample_gate(
        nonfinite[:, 0], nonfinite[:, 1], tau=1, kind="gaussian"
    )
    assert rejected["reason"] == "NONFINITE_INPUT_NO_ROW_DELETION"


def test_pinned_phyid_wrapper_closes_lattice_and_is_affine_invariant() -> None:
    series = coupled_ar(2, n=2049)
    transformed = affine_transform(series)
    for redundancy in ("MMI", "CCS"):
        original = run_phyid(
            series.data[:, 0],
            series.data[:, 1],
            tau=1,
            kind="gaussian",
            redundancy=redundancy,
        )  # type: ignore[arg-type]
        changed = run_phyid(
            transformed.data[:, 0],
            transformed.data[:, 1],
            tau=1,
            kind="gaussian",
            redundancy=redundancy,
        )  # type: ignore[arg-type]
        assert original.status == changed.status == "ELIGIBLE"
        original_means = original.means()
        changed_means = changed.means()
        assert original_means is not None and changed_means is not None
        assert abs(original_means["latticeClosureError"]) <= 1e-12
        assert abs(original_means["paperEquationClosureError"]) <= 1e-12
        np.testing.assert_allclose(
            [original_means["atomMeans"][atom] for atom in ATOM_IDS],
            [changed_means["atomMeans"][atom] for atom in ATOM_IDS],
            atol=1e-12,
            rtol=1e-12,
        )


def test_pinned_source_identity_and_matlab_fixture_parity() -> None:
    identity = backend_identity()
    assert identity["phyidCommit"] == "6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44"
    assert identity["omegaidCommit"] == "7fcf1fa8e288e0634f81423283d2b349ed88440e"
    assert identity["phyidCalculate"].startswith("/cache/e01_s03/sources/phyid/")

    fixture = loadmat(
        "/cache/e01_s03/sources/omegaid/temp_data/PhiID-test-simple-1.mat"
    )
    # The source regression fixture has only six effective samples and is therefore
    # deliberately not passed through the S10 science-sample gate here.
    from phyid.calculate import calc_PhiID

    atoms, _ = calc_PhiID(
        fixture["src"].squeeze(),
        fixture["trg"].squeeze(),
        int(fixture["tau"].squeeze()),
        kind="gaussian",
        redundancy="MMI",
    )
    observed = np.asarray([atoms[atom] for atom in ATOM_IDS])
    np.testing.assert_allclose(
        observed, fixture["PhiIDFull_MMI_L"], atol=1e-8, rtol=1e-5
    )


def test_discrete_relabel_constraint_is_preserved_for_omegaid_cpu() -> None:
    series = redundant_discrete(0, n=1025)
    relabeled = discrete_relabel(series)
    assert (
        series.seed_payload["streams"]["estimator"]["streamId"]
        != relabeled.seed_payload["streams"]["estimator"]["streamId"]
    )
    reference = run_phyid(
        series.data[:, 0],
        series.data[:, 1],
        tau=1,
        kind="discrete",
        redundancy="MMI",
    )
    reference_relabeled = run_phyid(
        relabeled.data[:, 0],
        relabeled.data[:, 1],
        tau=1,
        kind="discrete",
        redundancy="MMI",
    )
    omega = run_omegaid(
        series.data[:, 0],
        series.data[:, 1],
        tau=1,
        kind="discrete",
        redundancy="MMI",
        backend_name="numpy",
    )
    omega_relabeled = run_omegaid(
        relabeled.data[:, 0],
        relabeled.data[:, 1],
        tau=1,
        kind="discrete",
        redundancy="MMI",
        backend_name="numpy",
    )
    assert reference.means() is not None and reference_relabeled.means() is not None
    assert omega.means() is not None and omega_relabeled.means() is not None
    assert reference.means()["totalMi"] == pytest.approx(
        reference_relabeled.means()["totalMi"], abs=1e-12
    )
    assert abs(omega.means()["totalMi"] - omega_relabeled.means()["totalMi"]) > 0.1


def test_partition_enumeration_and_planted_controls_are_explicit() -> None:
    assert all_bipartitions(4) == [
        (0,),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
    ]
    assert all_bipartitions(4, balanced_only=True) == [(0, 1), (0, 2), (0, 3)]
    data = block_ar4(0, n=5000).data
    spectral = spectral_partition(data)
    assert spectral["status"] == "ELIGIBLE"
    assert tuple(spectral["partA"]) == (0, 1)
    for objective in ("synchronous_mi", "bidirectional_lagged_mi"):
        winner, candidates = exhaustive_partition_search(
            data,
            mapping="group_mean",
            objective=objective,
            normalization="none",
            balanced_only=False,
        )
        assert len(candidates) == 7
        assert tuple(winner["partA"]) == (0, 1)

    rejected = gaussian_partition_objective(
        data,
        (1, 2),
        mapping="group_mean",
        objective="synchronous_mi",
        normalization="none",
    )
    assert rejected["status"] == "INELIGIBLE"
    assert "canonical" in rejected["reason"]


def test_aggregate_rejects_missing_or_unknown_atoms() -> None:
    with pytest.raises(InformationValidationError, match="16 catalogued atoms"):
        aggregate_means(
            {"not_an_atom": np.zeros(3)},
            {
                key: np.zeros(3)
                for key in (
                    "I_xytab",
                    "I_xta",
                    "I_xtb",
                    "I_yta",
                    "I_ytb",
                    "I_xyta",
                    "I_xytb",
                    "I_xtab",
                    "I_ytab",
                )
            },
        )
