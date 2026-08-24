#!/usr/bin/env python3
"""Freeze S07 statistical tolerances and analytical fixtures before outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from e01_gard_independent import specification_from_mapping
from e01_gard_validation.stochastic import (
    analytical_propensities,
    binomial_fission_distribution,
    calibrated_moment_intervals,
    fixed_fission_distribution,
    poisson_count_bins,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPOSITORY_ROOT / "configs/e01/s07_stochastic_validation_preregistration.yaml"
)
RESULT_ONLY_NAMES = (
    "seed_manifest.json",
    "goodness_of_fit_summary.csv",
    "goodness_of_fit_details.json",
    "moment_tests.csv",
    "invariant_checks.csv",
    "failure_injection.json",
    "diagnostic_event_probabilities.png",
    "diagnostic_beta_moments.png",
    "diagnostic_fission_probabilities.png",
    "diagnostic_independent_only_branches.png",
    "registry_preservation.json",
    "validation_summary.json",
    "artifact_manifest.json",
    "research_step_full_results.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_config() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("S07 preregistration must be a YAML object.")
    return payload


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_contract(config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("researchStepId") != "S07":
        errors.append("researchStepId must be S07")
    statistical = config["statisticalDesign"]
    tests = config["primaryTests"]
    test_ids = [record["testId"] for record in tests]
    if len(test_ids) != len(set(test_ids)):
        errors.append("primary test IDs are not unique")
    if len(tests) != int(statistical["primaryTestCount"]):
        errors.append("primaryTestCount does not equal primaryTests length")
    calculated_alpha = float(statistical["globalFamilywiseAlpha"]) / len(tests)
    if not np.isclose(
        calculated_alpha,
        float(statistical["perTestAlpha"]),
        rtol=0.0,
        atol=1e-18,
    ):
        errors.append("perTestAlpha is not the preregistered Bonferroni value")
    minimum_p = 1.0 / (int(statistical["monteCarloReplicates"]) + 1)
    if minimum_p > calculated_alpha / 20.0:
        errors.append("Monte Carlo p-value resolution is too coarse")
    profiles = config["profiles"]
    for profile_id, profile in profiles.items():
        specification = specification_from_mapping(profile)
        if specification.specification_id != profile_id:
            errors.append(f"profile ID mismatch: {profile_id}")
    evidence_results = []
    for evidence_id, record in config["frozenEvidence"].items():
        path = Path(record["path"])
        actual = sha256(path) if path.is_file() else None
        valid = actual == record["sha256"]
        evidence_results.append(
            {
                "evidenceId": evidence_id,
                "path": str(path),
                "expectedSha256": record["sha256"],
                "actualSha256": actual,
                "valid": valid,
            }
        )
        if not valid:
            errors.append(f"frozen evidence mismatch: {evidence_id}")
    expected_artifacts = set(config["expectedArtifacts"])
    if not set(RESULT_ONLY_NAMES).issubset(expected_artifacts):
        errors.append("expectedArtifacts omits one or more canonical result files")
    return {
        "valid": not errors,
        "errors": errors,
        "primaryTestCount": len(tests),
        "primaryTestIds": test_ids,
        "perTestAlpha": calculated_alpha,
        "minimumMonteCarloPValue": minimum_p,
        "frozenEvidence": evidence_results,
    }


def _jsonable_outcome(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable_outcome(item) for item in value]
    return value


def _distribution_payload(
    target: Any, *, draws: int, minimum_expected: float
) -> dict[str, Any]:
    probabilities = np.asarray(target.probabilities, dtype=np.float64)
    expected = probabilities * draws
    rare = np.flatnonzero((probabilities > 0) & (expected < minimum_expected))
    structural = np.flatnonzero(probabilities == 0)
    return {
        "labels": list(target.labels),
        "outcomes": [_jsonable_outcome(outcome) for outcome in target.outcomes],
        "probabilities": probabilities.tolist(),
        "expectedCounts": expected.tolist(),
        "rareIndices": rare.tolist(),
        "rareLabels": [target.labels[index] for index in rare],
        "structuralZeroIndices": structural.tolist(),
        "asymptoticDiagnosticEligibleWithoutPooling": bool(
            np.all(expected[probabilities > 0] >= minimum_expected)
        ),
        "primaryMethod": "EXACT_PARAMETRIC_MONTE_CARLO_UNPOOLED",
    }


def analytical_fixtures(config: dict[str, Any]) -> dict[str, Any]:
    minimum_expected = float(
        config["statisticalDesign"]["minimumExpectedCountForAsymptoticDiagnostic"]
    )
    profiles = config["profiles"]
    events: dict[str, Any] = {}
    for fixture in config["fixtures"]["eventSelection"]:
        profile = profiles[fixture["profileId"]]
        target = analytical_propensities(
            fixture["state"],
            beta=fixture["beta"],
            rho=profile["rho"],
            k_f=profile["k_f"],
            k_b=profile["k_b"],
            orientation=profile["catalytic_matrix_branch"],
        )
        probabilities = np.asarray(target.probabilities, dtype=np.float64)
        expected = int(fixture["drawsPerEngine"]) * probabilities
        labels = [
            *[f"join_{index + 1}" for index in range(profile["n_species"])],
            *[f"leave_{index + 1}" for index in range(profile["n_species"])],
        ]
        rare = np.flatnonzero((probabilities > 0) & (expected < minimum_expected))
        events[fixture["fixtureId"]] = {
            "profileId": fixture["profileId"],
            "engines": fixture["engines"],
            "state": fixture["state"],
            "beta": fixture["beta"],
            "drawsPerEngine": fixture["drawsPerEngine"],
            "labels": labels,
            "boost": list(target.boost),
            "join": list(target.join),
            "leave": list(target.leave),
            "totalPropensity": target.total,
            "probabilities": probabilities.tolist(),
            "expectedCounts": expected.tolist(),
            "rareIndices": rare.tolist(),
            "rareLabels": [labels[index] for index in rare],
            "exactTestRequired": bool(rare.size),
        }
        if "waitingTimeUniformBins" in fixture:
            bins = int(fixture["waitingTimeUniformBins"])
            events[fixture["fixtureId"]]["waitingTimeTarget"] = {
                "cdfTransform": "u=1-exp(-a0*delta_t)",
                "binCount": bins,
                "probabilities": [1.0 / bins] * bins,
                "expectedCounts": [fixture["drawsPerEngine"] / bins] * bins,
            }

    fissions: dict[str, Any] = {}
    for fixture in config["fixtures"]["fission"]:
        if fixture["targetLaw"].startswith("fixed_size"):
            target = fixed_fission_distribution(fixture["parent"])
        else:
            profile = profiles[fixture["profileId"]]
            target = binomial_fission_distribution(
                fixture["parent"], probability=float(profile["fission_probability"])
            )
        fissions[fixture["fixtureId"]] = {
            "profileId": fixture["profileId"],
            "engines": fixture["engines"],
            "parent": fixture["parent"],
            "targetLaw": fixture["targetLaw"],
            "drawsPerEngine": fixture["drawsPerEngine"],
            **_distribution_payload(
                target,
                draws=int(fixture["drawsPerEngine"]),
                minimum_expected=minimum_expected,
            ),
        }
        if "daughterSelectionTarget" in fixture:
            fissions[fixture["fixtureId"]]["daughterSelectionTarget"] = fixture[
                "daughterSelectionTarget"
            ]

    poisson_fixture = config["fixtures"]["paperPoisson"]
    poisson_profile = profiles[poisson_fixture["profileId"]]
    props = analytical_propensities(
        poisson_fixture["state"],
        beta=poisson_fixture["beta"],
        rho=poisson_profile["rho"],
        k_f=poisson_profile["k_f"],
        k_b=poisson_profile["k_b"],
        orientation=poisson_profile["catalytic_matrix_branch"],
    )
    exposure = float(poisson_profile["poisson_exposure"])
    rates = np.asarray((*props.join, *props.leave), dtype=np.float64) * exposure
    poisson_channels: dict[str, Any] = {}
    for label, rate in zip(poisson_fixture["targetChannels"], rates, strict=True):
        target = poisson_count_bins(
            float(rate),
            tail_probability_floor=float(poisson_fixture["tailProbabilityFloor"]),
        )
        poisson_channels[label] = {
            "rate": float(rate),
            **_distribution_payload(
                target,
                draws=int(poisson_fixture["draws"]),
                minimum_expected=minimum_expected,
            ),
        }
    matrix = config["fixtures"]["catalyticMatrixMoments"]
    total_entries = int(matrix["totalEntriesPerEngine"])
    alpha = float(config["statisticalDesign"]["perTestAlpha"])
    expected_variance = float(matrix["expectedLogVariance"])
    moment_intervals = calibrated_moment_intervals(
        sample_count=total_entries,
        expected_mean=float(matrix["expectedLogMean"]),
        expected_variance=expected_variance,
        alpha=alpha,
    )
    return {
        "schema": "eidosoma.e01.s07_validation_fixtures.v1",
        "researchStepId": "S07",
        "scopeBoundary": config["scopeBoundary"],
        "profiles": profiles,
        "eventSelection": events,
        "catalyticMatrixMoments": {
            **matrix,
            "expectedLogStandardDeviation": float(np.sqrt(expected_variance)),
            "bonferroniAcceptanceIntervals": moment_intervals,
        },
        "fission": fissions,
        "paperPoisson": {
            "fixtureId": poisson_fixture["fixtureId"],
            "profileId": poisson_fixture["profileId"],
            "state": poisson_fixture["state"],
            "beta": poisson_fixture["beta"],
            "draws": poisson_fixture["draws"],
            "exposure": exposure,
            "appliedLossRule": poisson_fixture["appliedLossRule"],
            "channels": poisson_channels,
        },
    }


def freeze(artifact_root: Path) -> dict[str, Any]:
    config = load_config()
    step_dir = artifact_root / "research_steps/S07"
    step_dir.mkdir(parents=True, exist_ok=True)
    present_results = [name for name in RESULT_ONLY_NAMES if (step_dir / name).exists()]
    if present_results:
        raise RuntimeError(
            "S07 outcome artifacts already exist; preregistration freeze refused: "
            + ", ".join(present_results)
        )
    verification = verify_contract(config)
    if not verification["valid"]:
        raise RuntimeError(
            "Invalid S07 preregistration: " + "; ".join(verification["errors"])
        )
    status = git_output("status", "--porcelain")
    if status:
        raise RuntimeError(
            "Preregistration must be frozen from a clean committed worktree."
        )
    commit = git_output("rev-parse", "HEAD")
    fixtures = analytical_fixtures(config)
    preregistration_copy = step_dir / "preregistration.yaml"
    shutil.copyfile(CONFIG_PATH, preregistration_copy)
    fixture_path = step_dir / "validation_fixtures.json"
    write_json(fixture_path, fixtures)
    statistical = config["statisticalDesign"]
    tolerances = {
        "schema": "eidosoma.e01.s07_calibrated_tolerances.v1",
        "researchStepId": "S07",
        "preregistrationVersion": config["preregistrationVersion"],
        "calibrationBasis": (
            "Exact target laws and a global Bonferroni familywise bound frozen "
            "before canonical engine sampling."
        ),
        "globalFamilywiseAlpha": statistical["globalFamilywiseAlpha"],
        "primaryTestCount": statistical["primaryTestCount"],
        "perTestAlpha": statistical["perTestAlpha"],
        "monteCarloReplicates": statistical["monteCarloReplicates"],
        "minimumAttainableMonteCarloPValue": verification["minimumMonteCarloPValue"],
        "monteCarloBatchSize": statistical["monteCarloBatchSize"],
        "minimumExpectedCountForAsymptoticDiagnostic": statistical[
            "minimumExpectedCountForAsymptoticDiagnostic"
        ],
        "rareCategoryRule": statistical["rareCategoryRule"],
        "structuralInvariantAllowedFailures": config["deterministicInvariants"][
            "allowedFailures"
        ],
        "failureInjectionRequiredDetectedCount": config["failureInjection"][
            "requiredDetectedCount"
        ],
        "momentAcceptanceIntervals": fixtures["catalyticMatrixMoments"][
            "bonferroniAcceptanceIntervals"
        ],
        "primaryTests": config["primaryTests"],
        "passRule": statistical["passRule"],
    }
    tolerance_path = step_dir / "calibrated_tolerances.json"
    write_json(tolerance_path, tolerances)
    record = {
        "schema": "eidosoma.e01.s07_preregistration_record.v1",
        "researchStepId": "S07",
        "stepNumber": 7,
        "status": "frozen_before_canonical_outcomes",
        "frozenAtUtc": datetime.now(UTC).isoformat(),
        "repository": str(REPOSITORY_ROOT),
        "repositoryBranch": git_output("branch", "--show-current"),
        "preregistrationCommit": commit,
        "gitWorktreeCleanAtFreeze": True,
        "canonicalOutcomeArtifactsAbsentAtFreeze": True,
        "absentOutcomeArtifactNames": list(RESULT_ONLY_NAMES),
        "preregistrationSourcePath": str(CONFIG_PATH),
        "preregistrationSourceSha256": sha256(CONFIG_PATH),
        "preregistrationArtifactSha256": sha256(preregistration_copy),
        "calibratedTolerancesSha256": sha256(tolerance_path),
        "validationFixturesSha256": sha256(fixture_path),
        "contractVerification": verification,
        "boundaryStatement": config["scopeBoundary"],
        "recommendedNextAction": (
            "Run the canonical S07 validation exactly once from this frozen record; "
            "do not alter tolerances after inspecting engine outcomes."
        ),
    }
    record_path = step_dir / "preregistration_record.json"
    write_json(record_path, record)
    return {
        "success": True,
        "stepDirectory": str(step_dir),
        "preregistrationCommit": commit,
        "preregistrationSha256": sha256(preregistration_copy),
        "calibratedTolerancesSha256": sha256(tolerance_path),
        "validationFixturesSha256": sha256(fixture_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(freeze(args.artifacts_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
