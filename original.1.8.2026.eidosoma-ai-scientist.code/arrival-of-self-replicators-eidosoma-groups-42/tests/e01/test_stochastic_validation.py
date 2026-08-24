from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e01_gard_historical import compute_propensities as historical_propensities
from e01_gard_independent import calculate_propensities, specification_from_mapping
from e01_gard_reproducibility import (
    CANONICAL_STREAM_PURPOSES,
    CouplingPolicy,
    SeedRequest,
    StreamPurpose,
    derive_seed_bundle,
    isolated_stream_namespace,
)
from e01_gard_validation.stochastic import (
    analytical_propensities,
    binomial_fission_distribution,
    exact_multinomial_test,
    fixed_fission_distribution,
    lognormal_log_moment_tests,
    pool_rare_categories,
)

CONFIG_PATH = (
    REPOSITORY_ROOT / "configs/e01/s07_stochastic_validation_preregistration.yaml"
)


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, REPOSITORY_ROOT / "scripts/e01" / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preregistered_familywise_gate_and_test_registry() -> None:
    config = _config()
    design = config["statisticalDesign"]
    tests = config["primaryTests"]
    ids = [record["testId"] for record in tests]
    assert ids == [f"S07-T{index:02d}" for index in range(1, 27)]
    assert len(ids) == len(set(ids)) == design["primaryTestCount"] == 26
    assert design["multiplicityCorrection"] == "BONFERRONI_SINGLE_GLOBAL_FAMILY"
    assert np.isclose(
        design["perTestAlpha"],
        design["globalFamilywiseAlpha"] / design["primaryTestCount"],
        rtol=0.0,
        atol=1e-18,
    )
    assert 1.0 / (design["monteCarloReplicates"] + 1) <= design["perTestAlpha"] / 20


def test_all_profiles_are_complete_explicit_branch_instances() -> None:
    config = _config()
    for profile_id, payload in config["profiles"].items():
        profile = specification_from_mapping(payload)
        assert profile.specification_id == profile_id
    assert config["scopeBoundary"]["authorCodeIdentity"].startswith("UNAVAILABLE::")
    assert config["scopeBoundary"]["legacyMatlabRngIdentity"].startswith("UNRESOLVED::")
    assert (
        config["scopeBoundary"]["historicalHarnessIdentity"]
        == "NUMPY_GENERATOR_EXPLICIT_NOT_MATLAB_LEGACY"
    )


def test_frozen_evidence_hashes_match() -> None:
    config = _config()
    for record in config["frozenEvidence"].values():
        path = Path(record["path"])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_independent_analytical_event_oracle_matches_both_engines() -> None:
    config = _config()
    fixture = config["fixtures"]["eventSelection"][0]
    profile = specification_from_mapping(config["profiles"][fixture["profileId"]])
    target = analytical_propensities(
        fixture["state"],
        beta=fixture["beta"],
        rho=profile.rho,
        k_f=profile.k_f,
        k_b=profile.k_b,
    )
    historical = historical_propensities(
        fixture["state"],
        beta=fixture["beta"],
        rho=profile.rho,
        k_f=profile.k_f,
        k_b=profile.k_b,
    )
    independent = calculate_propensities(
        fixture["state"], beta=fixture["beta"], specification=profile
    )
    np.testing.assert_array_equal(target.concatenated, historical.concatenated)
    np.testing.assert_array_equal(target.concatenated, independent.concatenated)
    assert np.isclose(sum(target.probabilities), 1.0, rtol=0.0, atol=1e-15)


def test_rare_category_rule_is_triggered_before_sampling() -> None:
    config = _config()
    fixture = config["fixtures"]["eventSelection"][1]
    profile = specification_from_mapping(config["profiles"][fixture["profileId"]])
    target = analytical_propensities(
        fixture["state"],
        beta=fixture["beta"],
        rho=profile.rho,
        k_f=profile.k_f,
        k_b=profile.k_b,
    )
    expected = np.asarray(target.probabilities) * fixture["drawsPerEngine"]
    assert np.count_nonzero(expected < 25) == 3
    deterministic_counts = np.rint(expected).astype(int)
    deterministic_counts[0] += fixture["drawsPerEngine"] - deterministic_counts.sum()
    pooling = pool_rare_categories(
        deterministic_counts,
        target.probabilities,
        labels=["j1", "j2", "j3", "l1", "l2", "l3"],
        minimum_expected=25,
    )
    assert pooling["rareIndices"] == [3, 4, 5]
    assert pooling["primaryMethod"] == "EXACT_PARAMETRIC_MONTE_CARLO_UNPOOLED"
    assert not pooling["asymptoticPearsonEligible"]


def test_exact_fission_targets_are_normalized_and_conservative() -> None:
    even = fixed_fission_distribution((2, 2, 2))
    odd = fixed_fission_distribution((2, 2, 1))
    binomial = binomial_fission_distribution((2, 3, 1), probability=0.5)
    assert np.isclose(sum(even.probabilities), 1.0, rtol=0.0, atol=1e-12)
    assert np.isclose(sum(odd.probabilities), 1.0, rtol=0.0, atol=1e-12)
    assert np.isclose(sum(binomial.probabilities), 1.0, rtol=0.0, atol=1e-12)
    assert all(sum(outcome) == 3 for outcome in even.outcomes)
    assert all(
        sum(outcome[0]) == 2 and sum(outcome[1]) == 1 for outcome in odd.outcomes
    )
    assert len(binomial.outcomes) == 24


def test_exact_multinomial_detector_accepts_center_and_rejects_fault() -> None:
    probabilities = np.asarray([0.5, 0.3, 0.15, 0.05])
    centered = np.asarray([5000, 3000, 1500, 500])
    centered_result = exact_multinomial_test(
        centered,
        probabilities,
        generator=np.random.default_rng(7001),
        replicates=9999,
        batch_size=1000,
    )
    fault = np.asarray([4500, 3500, 1500, 500])
    fault_result = exact_multinomial_test(
        fault,
        probabilities,
        generator=np.random.default_rng(7002),
        replicates=9999,
        batch_size=1000,
    )
    assert centered_result["pValue"] > 0.1
    assert fault_result["pValue"] == 1.0 / 10000


def test_exact_log_beta_moment_detectors() -> None:
    centered = lognormal_log_moment_tests(
        sample_count=10000,
        sample_mean=-4.0,
        sample_variance=16.0,
        expected_mean=-4.0,
        expected_variance=16.0,
    )
    shifted = lognormal_log_moment_tests(
        sample_count=10000,
        sample_mean=-3.8,
        sample_variance=16.0,
        expected_mean=-4.0,
        expected_variance=16.0,
    )
    assert centered["mean"]["pValue"] == 1.0
    assert centered["variance"]["pValue"] > 0.99
    assert shifted["mean"]["pValue"] < 1e-5


def test_s06_seed_contract_gives_nine_unique_validation_streams() -> None:
    config = _config()
    specification_id = "E01-S07-TEST-SEED"
    trajectory_id = "E01-S07-TEST-SEED-R0"
    namespace = isolated_stream_namespace(
        experiment_id="E01",
        specification_id=specification_id,
        trajectory_id=trajectory_id,
        replicate_index=0,
    )
    request = SeedRequest(
        experiment_id="E01",
        specification_id=specification_id,
        trajectory_id=trajectory_id,
        replicate_index=0,
        engine_id="e01_s07_test@1.0.0",
        root_seed_hex=config["randomness"]["rootSeedHex"],
        coupling_policy=CouplingPolicy.TRAJECTORY_ISOLATED,
        coupling_reason=None,
        stream_namespaces={purpose: namespace for purpose in CANONICAL_STREAM_PURPOSES},
    )
    bundle = derive_seed_bundle(request)
    assert len(bundle.streams) == 9
    assert len({stream.stream_id for stream in bundle.streams.values()}) == 9
    assert bundle.to_payload()["uncertaintyBoundary"][
        "legacyMatlabRngIdentity"
    ].startswith("UNRESOLVED::")
    assert StreamPurpose.WAITING_TIME in bundle.streams


def test_calibration_builder_creates_all_analytical_targets_without_outcomes() -> None:
    freeze_module = _script_module(
        "freeze_s07_preregistration_test", "freeze_s07_preregistration.py"
    )
    runner_module = _script_module(
        "run_s07_stochastic_validation_test", "run_s07_stochastic_validation.py"
    )
    config = _config()
    verification = freeze_module.verify_contract(config)
    assert verification["valid"], verification["errors"]
    fixtures = freeze_module.analytical_fixtures(config)
    assert set(fixtures["eventSelection"]) == {
        "S07-EVENT-COMMON",
        "S07-EVENT-RARE",
        "S07-EVENT-MODERN",
    }
    assert set(fixtures["fission"]) == {
        "S07-FISSION-FIXED-EVEN",
        "S07-FISSION-FIXED-ODD",
        "S07-FISSION-BINOMIAL",
    }
    assert len(fixtures["paperPoisson"]["channels"]) == 6
    assert len(runner_module.raw_tasks(config)) == 13
    assert len(config["failureInjection"]["cases"]) == 7


def test_historical_raw_event_task_uses_real_engine_record(monkeypatch) -> None:
    runner_module = _script_module(
        "run_s07_stochastic_validation_historical_task_test",
        "run_s07_stochastic_validation.py",
    )
    config = deepcopy(_config())
    fixture = config["fixtures"]["eventSelection"][0]
    fixture["drawsPerEngine"] = 32
    monkeypatch.setattr(runner_module, "load_config", lambda: config)

    result = runner_module.run_event_task(fixture["fixtureId"], "historical_reference")

    assert sum(result["counts"]) == 32
    assert result["invariants"] == {
        "massFailures": 0,
        "stateReconstructionFailures": 0,
        "nonnegativeFailures": 0,
        "branchIdentityFailures": 0,
    }


def test_completed_artifacts_are_consistent_when_present() -> None:
    step_dir = Path("/artifacts/research_steps/S07")
    summary_path = step_dir / "validation_summary.json"
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["researchStepId"] == "S07"
    assert summary["checks"]["primaryTestCount"] == 26
    assert summary["checks"]["registryPreserved"]
    assert summary["checks"]["s08ArtifactsAbsent"]
    assert all(
        Path(path).is_file()
        for path in summary["artifactsWritten"]
        if not path.endswith("research_step_full_results.md")
    )
