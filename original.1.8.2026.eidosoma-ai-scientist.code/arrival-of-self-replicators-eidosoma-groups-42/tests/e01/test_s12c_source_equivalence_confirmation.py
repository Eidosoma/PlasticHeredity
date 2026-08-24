from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr

from e01_pigozzi_source_audit.core import SourceImplementation
from e01_pigozzi_source_equivalence_confirmation.core import (
    derive_seed,
    fixture_array,
    iigr_pairwise_source_mi,
    run_source_pipeline,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/e01/s12c_source_equivalence_confirmation_preregistration.yaml"
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")


def literal_source_mi(data: np.ndarray) -> np.ndarray:
    result = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(i):
            r1, p1 = pearsonr(data[i, :-1], data[j, 1:])
            r2, p2 = pearsonr(data[i, 1:], data[j, :-1])
            mi1 = -0.5 * np.log(1.0 - r1**2.0) if p1 < 1 else 0
            mi2 = -0.5 * np.log(1.0 - r2**2.0) if p2 < 1 else 0
            result[i, j] = mi1 + mi2
            result[j, i] = mi1 + mi2
    return np.array(result)


def test_pairwise_iigr_mi_is_operation_exact() -> None:
    rng = np.random.RandomState(7721)
    data = rng.normal(size=(10, 384))
    observed = iigr_pairwise_source_mi(data)
    expected = literal_source_mi(data)
    assert np.array_equal(observed, expected)


def test_fixture_firewall_payloads_and_seeds_are_disjoint() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    development_root = config["fixtureFirewall"]["development"]["rootSeedHex"]
    confirmation_root = config["fixtureFirewall"]["confirmation"]["rootSeedHex"]
    development_payloads = {
        fixture_array(fixture, "development", development_root).tobytes()
        for fixture in config["fixtureFirewall"]["fixtureIds"]
    }
    confirmation_payloads = {
        fixture_array(fixture, "confirmation", confirmation_root).tobytes()
        for fixture in config["fixtureFirewall"]["fixtureIds"]
    }
    assert development_payloads.isdisjoint(confirmation_payloads)
    development_seeds = {
        derive_seed(development_root, "development", implementation.value, fixture, kind)
        for implementation in SourceImplementation
        for fixture in config["fixtureFirewall"]["fixtureIds"]
        for kind in ("preprocessing_noise", "fiedler_initialization")
    }
    confirmation_seeds = {
        derive_seed(confirmation_root, "confirmation", implementation.value, fixture, kind)
        for implementation in SourceImplementation
        for fixture in config["fixtureFirewall"]["fixtureIds"]
        for kind in ("preprocessing_noise", "fiedler_initialization")
    }
    assert development_seeds.isdisjoint(confirmation_seeds)


def test_phirl_constant_status_matches_frozen_source_policy() -> None:
    observations = np.ones((384, 10), dtype=np.float64)
    result = run_source_pipeline(
        observations,
        SourceImplementation.PHIRL,
        SAFE_LATTICE,
        preprocessing_seed=1,
        partition_seed=2,
    )
    assert result.status == "INELIGIBLE_TOO_FEW_ACTIVE_DIMENSIONS"
    assert result.retained_available
    assert result.retained_variables == ()
    assert result.processed is None
    assert result.local_phi_r is None


def test_preregistered_global_gate_is_not_weakened() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    gates = config["confirmationGates"]
    assert gates["allRowsMustPass"]
    assert gates["expectedRows"] == 14
    assert gates["statusIdentical"]
    assert gates["miMaxAbsDifferenceAtMost"] == 1e-10
    assert gates["partitionAverageMaxAbsDifferenceAtMost"] == 1e-10
    assert gates["localPhiRMaxAbsDifferenceAtMost"] == 1e-9
    assert gates["eligibleSourceCannotMatchWrapperException"]
    assert "SINGULAR_DUPLICATE_INPUT" in config["fixtureFirewall"]["fixtureIds"]
    assert config["scopeBoundary"]["additionalRepairAfterConfirmationFailureForbidden"]
