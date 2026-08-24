#!/usr/bin/env python3
"""Validate and freeze the S12 strict-MRR preregistration before outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/e01/s12_strict_mrr_preregistration.yaml"
AMENDMENT = REPO / "configs/e01/s12_strict_mrr_preregistration_amendment_1.yaml"
AMENDMENT_2 = REPO / "configs/e01/s12_strict_mrr_preregistration_amendment_2.yaml"
STEP_ROOT = Path("/artifacts/research_steps/S12")
ALLOWED_PREOUTCOME_FILES = {
    "preregistration.yaml",
    "preregistration_record.json",
    "preregistration_amendment_1.yaml",
    "preregistration_amendment_record.json",
    "preregistration_amendment_2.yaml",
    "preregistration_amendment_2_record.json",
}
CLAIM_VOCABULARY = [
    "SUPPORTED",
    "DIRECTIONALLY_SUPPORTED",
    "NOT_SUPPORTED_WITHIN_STRICT_SCOPE",
    "UNDERDETERMINED",
    "NOT_EVALUATED",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_files(root: Path, paths: list[Path]) -> str:
    records = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in sorted(paths)
    ]
    return hashlib.sha256("".join(records).encode("utf-8")).hexdigest()


def tracked_prior_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO, check=True, capture_output=True, text=True
    )
    tracked = result.stdout.splitlines()
    exact = {
        "configs/e01/s10_information_dynamics_preregistration.yaml",
        "configs/e01/s11_time_localized_phir_preregistration.yaml",
        "configs/e01/s11r_confirmation_method_lock.yaml",
        "configs/e01/s11r_time_localized_phir_repair_preregistration.yaml",
        "scripts/e01/run_s10_information_dynamics_validation.py",
        "scripts/e01/run_s11_time_localized_phir.py",
        "scripts/e01/run_s11r_time_localized_phir_repair.py",
        "tests/e01/test_information_dynamics.py",
        "tests/e01/test_time_localized_phir.py",
        "tests/e01/test_time_localized_phir_repair.py",
    }
    prefixes = (
        "src/e01_information_dynamics/",
        "src/e01_time_localized_phir/",
        "src/e01_time_localized_phir_repair/",
    )
    selected = [name for name in tracked if name in exact or name.startswith(prefixes)]
    return [REPO / name for name in sorted(selected)]


def validate_preregistration(*, require_no_outcomes: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    checks: list[dict[str, Any]] = []
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def check(check_id: str, condition: bool, detail: Any) -> None:
        checks.append(
            {"checkId": check_id, "passed": bool(condition), "detail": detail}
        )
        if not condition:
            errors.append(f"{check_id}: {detail}")

    check(
        "schema",
        data.get("schema") == "eidosoma.e01.s12_strict_mrr_preregistration.v1",
        data.get("schema"),
    )
    check(
        "step_identity",
        data.get("researchStepId") == "S12" and data.get("stepNumber") == 12,
        [data.get("researchStepId"), data.get("stepNumber")],
    )
    check(
        "version",
        data.get("preregistrationVersion") == "E01-S12-STRICT-MRR-v1.0.0",
        data.get("preregistrationVersion"),
    )
    check(
        "preoutcome_status",
        data.get("status") == "FROZEN_BEFORE_ANY_GARD_SCIENTIFIC_OUTCOME",
        data.get("status"),
    )

    frozen_input_results: list[dict[str, Any]] = []
    for item in data.get("frozenInputs", []):
        path = Path(item["path"])
        actual = sha256_file(path) if path.is_file() else None
        passed = actual == item["sha256"]
        frozen_input_results.append(
            {
                "inputId": item["inputId"],
                "path": str(path),
                "expectedSha256": item["sha256"],
                "actualSha256": actual,
                "passed": passed,
            }
        )
    check(
        "frozen_inputs",
        len(frozen_input_results) == 28
        and all(x["passed"] for x in frozen_input_results),
        {
            "count": len(frozen_input_results),
            "failures": [x for x in frozen_input_results if not x["passed"]],
        },
    )

    immutable = data["immutablePriorEvidence"]
    artifact_results: dict[str, Any] = {}
    for step_id in ("S10", "S11", "S11R"):
        root = Path("/artifacts/research_steps") / step_id
        paths = [path for path in root.rglob("*") if path.is_file()]
        actual = aggregate_files(root, paths)
        expected = immutable["artifactDirectories"][step_id]
        passed = (
            len(paths) == expected["fileCount"]
            and actual == expected["aggregateSha256"]
        )
        artifact_results[step_id] = {
            "fileCount": len(paths),
            "expectedFileCount": expected["fileCount"],
            "aggregateSha256": actual,
            "expectedAggregateSha256": expected["aggregateSha256"],
            "passed": passed,
        }
    check(
        "prior_artifact_immutability",
        all(v["passed"] for v in artifact_results.values()),
        artifact_results,
    )

    prior_paths = tracked_prior_paths()
    repo_actual = aggregate_files(REPO, prior_paths)
    repo_expected = immutable["repositoryFileSet"]
    check(
        "prior_repository_immutability",
        len(prior_paths) == repo_expected["fileCount"]
        and repo_actual == repo_expected["aggregateSha256"],
        {
            "fileCount": len(prior_paths),
            "expectedFileCount": repo_expected["fileCount"],
            "aggregateSha256": repo_actual,
            "expectedAggregateSha256": repo_expected["aggregateSha256"],
        },
    )

    scope = data["scopeBoundary"]
    check(
        "scope",
        scope["exactBaselineMatrixCount"] == 12
        and scope["maximumInterventionTriplets"] == 6
        and scope["interventionTripletCountRule"] == "exactly_zero_or_exactly_six"
        and scope["nextStepForbidden"] == "S13"
        and scope["fixedWindowRestorationForbidden"]
        and scope["s11FixedEstimatesForbidden"]
        and scope["s11rFixedEstimatesForbidden"]
        and scope["paperPrimarySelectionForbidden"]
        and scope["authorPrimarySelectionForbidden"],
        scope,
    )
    check(
        "strict_boundary",
        data["lagAndProspectiveIndexing"]["minimumEffectiveSamples"] == 512
        and data["strictEstimateGates"]["gateId"] == "E01-S10-SAMPLE-GATE-STRICT-v1.0.0"
        and data["strictEstimateGates"]["noFallbackRegularization"],
        data["lagAndProspectiveIndexing"],
    )
    check(
        "branches_explicit",
        len(data["preprocessingBranches"]["branches"]) == 2
        and len(data["estimatorBranches"]["branches"]) == 2
        and data["estimatorBranches"]["omegaPolicy"]["discrete"] == "EXCLUDED"
        and data["estimatorBranches"]["omegaPolicy"]["moreThan2x2Doublet"]
        == "EXCLUDED",
        {
            "preprocessing": [
                x["preprocessingId"] for x in data["preprocessingBranches"]["branches"]
            ],
            "redundancy": [
                x["redundancyId"] for x in data["estimatorBranches"]["branches"]
            ],
            "omegaPolicy": data["estimatorBranches"]["omegaPolicy"],
        },
    )
    check(
        "partition_lock",
        data["partitionBranch"]["searchId"]
        == "E01-S10-MIB-SEARCH-SPECTRAL-FIXED-CANDIDATE-v1.0.0"
        and data["partitionBranch"]["gates"]["featureRelabelReplay"]["permutationCount"]
        == 3
        and data["partitionBranch"]["gates"]["featureRelabelReplay"]["ariMustEqual"]
        == 1.0,
        data["partitionBranch"],
    )
    feasibility = data["baselineFeasibilityGate"]
    check(
        "intervention_gate",
        feasibility["frozenBeforeBaselineOutcomes"]
        and feasibility["allMustPass"]["minimumQualifyingTrajectories"] == 6
        and feasibility["outcome"]["pass"] == "run_exactly_six_triplets"
        and feasibility["outcome"]["fail"]
        == "run_zero_triplets_and_retain_every_gate_reason",
        feasibility,
    )
    separation = data["interventionDesign"]["separation"]
    check(
        "candidate_discriminability",
        data["interventionDesign"]["candidates"]["fullSetRequired"]
        and data["interventionDesign"]["scoring"][
            "everyCandidateMustPassBothPreprocessingStrictGates"
        ]
        and separation["nullEnvelope"]["families"] == 4096
        and separation["nullEnvelope"]["threshold"]
        == "empirical_0.99_quantile_method_higher"
        and data["interventionDesign"]["suppression"]["exactStatus"]
        == "INELIGIBLE_ACTION_NOT_SEPARABLE",
        {
            "separation": separation,
            "suppression": data["interventionDesign"]["suppression"],
        },
    )
    root_seed = data["randomness"]["rootSeedHex"]
    check(
        "seeds",
        len(root_seed) == 64
        and all(ch in "0123456789abcdef" for ch in root_seed)
        and data["randomness"]["baselineMatrixIndices"] == list(range(12))
        and len(data["randomness"]["streamPurposes"]) == 9,
        data["randomness"],
    )
    check(
        "claim_vocabulary",
        data["claimClassification"]["vocabulary"] == CLAIM_VOCABULARY,
        data["claimClassification"]["vocabulary"],
    )
    check(
        "whole_label",
        data["wholeTrajectoryBranch"]["labelExact"] == "DESCRIPTIVE_NONPROSPECTIVE",
        data["wholeTrajectoryBranch"],
    )
    check(
        "status_schema",
        data["statusJsonRequiredFields"]
        == [
            "researchStepId",
            "stepNumber",
            "success",
            "status",
            "artifactsWritten",
            "validationResult",
            "caveatsOrBlockers",
            "recommendedNextAction",
        ],
        data["statusJsonRequiredFields"],
    )
    check(
        "required_outputs",
        len(data["requiredOutputs"]) == 38
        and "research_step_full_results.md" in data["requiredOutputs"],
        data["requiredOutputs"],
    )

    existing = []
    if STEP_ROOT.exists():
        existing = [
            path.relative_to(STEP_ROOT).as_posix()
            for path in STEP_ROOT.rglob("*")
            if path.is_file()
        ]
    check(
        "no_scientific_outcomes_before_freeze",
        (not require_no_outcomes) or set(existing).issubset(ALLOWED_PREOUTCOME_FILES),
        existing,
    )

    return {
        "schema": "eidosoma.e01.s12_preregistration_validation.v1",
        "researchStepId": "S12",
        "preregistrationVersion": data["preregistrationVersion"],
        "success": not errors,
        "configPath": str(CONFIG),
        "configSha256": sha256_file(CONFIG),
        "checks": checks,
        "errors": errors,
        "frozenInputs": frozen_input_results,
    }


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_amendment() -> dict[str, Any]:
    """Validate the pre-outcome operational clarification against its parent."""

    data = yaml.safe_load(AMENDMENT.read_text(encoding="utf-8"))
    checks = {
        "schema": data.get("schema")
        == "eidosoma.e01.s12_strict_mrr_preregistration_amendment.v1",
        "identity": data.get("amendmentId") == "E01-S12-STRICT-MRR-v1.0.0-AMENDMENT-01",
        "status": data.get("status")
        == "FROZEN_CLARIFICATION_BEFORE_ANY_GARD_SCIENTIFIC_OUTCOME",
        "parentConfig": data.get("parentPreregistrationSha256") == sha256_file(CONFIG),
        "parentRecord": data.get("parentFreezeRecordSha256")
        == sha256_file(STEP_ROOT / "preregistration_record.json"),
        "strictBoundary": data["scopePreservation"]["strictMinimumEffectiveSamples"]
        == 512,
        "scope": data["scopePreservation"]["baselineMatrices"] == 12
        and data["scopePreservation"]["interventionTriplets"]
        == "exactly_zero_or_exactly_six"
        and data["scopePreservation"]["nextStepForbidden"] == "S13",
        "actionObservation": not data["actualActionObservationClarification"][
            "appliedActionState"
        ]["inserted_as_additional_estimator_observation"],
        "tie": data["actionTieClarification"]["numericalTieTolerance"] == 1.0e-12,
        "endpoint": data["restrictedInterventionEndpoint"]["direction"]
        == "max_greater_than_control_greater_than_min",
        "noFixedWindowRepair": data["scopePreservation"][
            "fixedWindowBranchesRemainIneligible"
        ],
    }
    return {
        "success": all(checks.values()),
        "amendmentPath": str(AMENDMENT),
        "amendmentSha256": sha256_file(AMENDMENT),
        "checks": checks,
        "errors": [key for key, passed in checks.items() if not passed],
    }


def validate_amendment_2() -> dict[str, Any]:
    """Validate the final pre-outcome descriptive/claim clarification."""

    data = yaml.safe_load(AMENDMENT_2.read_text(encoding="utf-8"))
    whole = data["wholeTrajectoryDirectionalComparison"]
    claims = data["claimClassificationClarification"]
    checks = {
        "schema": data.get("schema")
        == "eidosoma.e01.s12_strict_mrr_preregistration_amendment.v1",
        "identity": data.get("amendmentId") == "E01-S12-STRICT-MRR-v1.0.0-AMENDMENT-02",
        "status": data.get("status")
        == "FROZEN_CLARIFICATION_BEFORE_ANY_GARD_SCIENTIFIC_OUTCOME",
        "parentConfig": data.get("parentPreregistrationSha256") == sha256_file(CONFIG),
        "parentAmendment": data.get("parentAmendment01Sha256")
        == sha256_file(AMENDMENT),
        "parentAmendmentRecord": data.get("parentAmendment01RecordSha256")
        == sha256_file(STEP_ROOT / "preregistration_amendment_record.json"),
        "wholeScope": whole["scopeLabelExact"] == "DESCRIPTIVE_NONPROSPECTIVE"
        and whole["predictionUseForbidden"]
        and whole["actionSelectionUseForbidden"],
        "trend": whole["aggregateTrend"]["normalizedProgressGridPoints"] == 1001
        and whole["aggregateTrend"]["directionalSupportRule"]
        == "all_four_branch_p_values_greater_than_0.05",
        "temporal": whole["temporalDependence"]["alpha"] == 0.05
        and whole["temporalDependence"]["lagRule"]
        == "min_20_or_floor_series_length_divided_by_5",
        "claimBoundary": claims["localSpikePaperClaimStatus"] == "UNDERDETERMINED"
        and claims["earlyWarningPaperClaimStatus"] == "UNDERDETERMINED",
        "actionDensity": claims["interventionDirectionalSupportActionDensity"]
        == "at_least_4_of_6_triplets_each_have_at_least_3_applied_actions_in_both_noncontrol_conditions",
        "scope": data["scopePreservation"]["baselineMatrices"] == 12
        and data["scopePreservation"]["interventionTriplets"]
        == "exactly_zero_or_exactly_six"
        and data["scopePreservation"]["strictMinimumEffectiveSamples"] == 512
        and data["scopePreservation"]["nextStepForbidden"] == "S13",
    }
    return {
        "success": all(checks.values()),
        "amendmentPath": str(AMENDMENT_2),
        "amendmentSha256": sha256_file(AMENDMENT_2),
        "checks": checks,
        "errors": [key for key, passed in checks.items() if not passed],
    }


def write_record(result: dict[str, Any], frozen_commit: str) -> dict[str, Any]:
    if not result["success"]:
        raise RuntimeError("cannot freeze invalid preregistration")
    commit = git_output("rev-parse", f"{frozen_commit}^{{commit}}")
    committed = subprocess.run(
        ["git", "show", f"{commit}:configs/e01/s12_strict_mrr_preregistration.yaml"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != result["configSha256"]:
        raise RuntimeError(
            "frozen commit does not contain the validated preregistration bytes"
        )
    remote = subprocess.run(
        ["git", "ls-remote", "origin", "refs/heads/eidosoma/groups/42"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote_commit = remote.split()[0] if remote else None
    if remote_commit != commit:
        raise RuntimeError(f"remote branch is {remote_commit}, expected {commit}")

    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG, STEP_ROOT / "preregistration.yaml")
    record = {
        "schema": "eidosoma.e01.s12_preregistration_record.v1",
        "researchStepId": "S12",
        "stepNumber": 12,
        "preregistrationVersion": "E01-S12-STRICT-MRR-v1.0.0",
        "status": "FROZEN_VALIDATED_COMMITTED_AND_PUSHED_BEFORE_GARD_OUTCOMES",
        "frozenAtUtc": "2026-08-02T00:00:00Z",
        "repository": "Eidosoma/arrival-of-self-replicators",
        "branch": "eidosoma/groups/42",
        "commit": commit,
        "remoteCommit": remote_commit,
        "configPath": str(CONFIG),
        "configSha256": result["configSha256"],
        "artifactCopyPath": str(STEP_ROOT / "preregistration.yaml"),
        "artifactCopySha256": sha256_file(STEP_ROOT / "preregistration.yaml"),
        "researchPlanSha256": sha256_file(Path("/workspace/RESEARCH_PLAN.md")),
        "validationSuccess": True,
        "validationCheckCount": len(result["checks"]),
        "validationChecksPassed": sum(check["passed"] for check in result["checks"]),
        "scientificOutcomeFilesPresentAtFreeze": [],
    }
    (STEP_ROOT / "preregistration_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def write_amendment_record(
    result: dict[str, Any], frozen_commit: str
) -> dict[str, Any]:
    """Record the pushed amendment while the outcome directory is still clean."""

    if not result["success"]:
        raise RuntimeError("cannot freeze invalid amendment")
    commit = git_output("rev-parse", f"{frozen_commit}^{{commit}}")
    relative = "configs/e01/s12_strict_mrr_preregistration_amendment_1.yaml"
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != result["amendmentSha256"]:
        raise RuntimeError(
            "frozen commit does not contain the validated amendment bytes"
        )
    remote = subprocess.run(
        ["git", "ls-remote", "origin", "refs/heads/eidosoma/groups/42"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote_commit = remote.split()[0] if remote else None
    if remote_commit != commit:
        raise RuntimeError(f"remote branch is {remote_commit}, expected {commit}")
    existing = [
        path.relative_to(STEP_ROOT).as_posix()
        for path in STEP_ROOT.rglob("*")
        if path.is_file()
    ]
    if not set(existing).issubset(ALLOWED_PREOUTCOME_FILES):
        raise RuntimeError(
            f"scientific outcome existed before amendment freeze: {existing}"
        )
    shutil.copyfile(AMENDMENT, STEP_ROOT / "preregistration_amendment_1.yaml")
    record = {
        "schema": "eidosoma.e01.s12_preregistration_amendment_record.v1",
        "researchStepId": "S12",
        "stepNumber": 12,
        "amendmentId": "E01-S12-STRICT-MRR-v1.0.0-AMENDMENT-01",
        "status": "FROZEN_VALIDATED_COMMITTED_AND_PUSHED_BEFORE_GARD_OUTCOMES",
        "frozenAtUtc": "2026-08-02T00:00:00Z",
        "branch": "eidosoma/groups/42",
        "commit": commit,
        "remoteCommit": remote_commit,
        "amendmentSha256": result["amendmentSha256"],
        "artifactCopySha256": sha256_file(
            STEP_ROOT / "preregistration_amendment_1.yaml"
        ),
        "parentPreregistrationSha256": sha256_file(CONFIG),
        "parentFreezeRecordSha256": sha256_file(
            STEP_ROOT / "preregistration_record.json"
        ),
        "validationSuccess": True,
        "validationChecks": result["checks"],
        "scientificOutcomeFilesPresentAtFreeze": [],
    }
    (STEP_ROOT / "preregistration_amendment_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def write_amendment_2_record(
    result: dict[str, Any], frozen_commit: str
) -> dict[str, Any]:
    """Record the pushed final clarification before any GARD outcome."""

    if not result["success"]:
        raise RuntimeError("cannot freeze invalid amendment 2")
    commit = git_output("rev-parse", f"{frozen_commit}^{{commit}}")
    relative = "configs/e01/s12_strict_mrr_preregistration_amendment_2.yaml"
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != result["amendmentSha256"]:
        raise RuntimeError(
            "frozen commit does not contain the validated amendment 2 bytes"
        )
    remote = subprocess.run(
        ["git", "ls-remote", "origin", "refs/heads/eidosoma/groups/42"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote_commit = remote.split()[0] if remote else None
    if remote_commit != commit:
        raise RuntimeError(f"remote branch is {remote_commit}, expected {commit}")
    existing = [
        path.relative_to(STEP_ROOT).as_posix()
        for path in STEP_ROOT.rglob("*")
        if path.is_file()
    ]
    if not set(existing).issubset(ALLOWED_PREOUTCOME_FILES):
        raise RuntimeError(
            f"scientific outcome existed before amendment 2 freeze: {existing}"
        )
    shutil.copyfile(AMENDMENT_2, STEP_ROOT / "preregistration_amendment_2.yaml")
    record = {
        "schema": "eidosoma.e01.s12_preregistration_amendment_record.v1",
        "researchStepId": "S12",
        "stepNumber": 12,
        "amendmentId": "E01-S12-STRICT-MRR-v1.0.0-AMENDMENT-02",
        "status": "FROZEN_VALIDATED_COMMITTED_AND_PUSHED_BEFORE_GARD_OUTCOMES",
        "frozenAtUtc": "2026-08-02T00:00:00Z",
        "branch": "eidosoma/groups/42",
        "commit": commit,
        "remoteCommit": remote_commit,
        "amendmentSha256": result["amendmentSha256"],
        "artifactCopySha256": sha256_file(
            STEP_ROOT / "preregistration_amendment_2.yaml"
        ),
        "parentPreregistrationSha256": sha256_file(CONFIG),
        "parentAmendment01Sha256": sha256_file(AMENDMENT),
        "parentAmendment01RecordSha256": sha256_file(
            STEP_ROOT / "preregistration_amendment_record.json"
        ),
        "validationSuccess": True,
        "validationChecks": result["checks"],
        "scientificOutcomeFilesPresentAtFreeze": [],
    }
    (STEP_ROOT / "preregistration_amendment_2_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-record", action="store_true")
    parser.add_argument("--validate-amendment", action="store_true")
    parser.add_argument("--write-amendment-record", action="store_true")
    parser.add_argument("--validate-amendment-2", action="store_true")
    parser.add_argument("--write-amendment-2-record", action="store_true")
    parser.add_argument("--frozen-commit")
    args = parser.parse_args()
    result = validate_preregistration(require_no_outcomes=True)
    if args.write_record:
        if not args.frozen_commit:
            parser.error("--write-record requires --frozen-commit")
        result["record"] = write_record(result, args.frozen_commit)
    if args.validate_amendment or args.write_amendment_record:
        amendment = validate_amendment()
        result["amendment"] = amendment
        result["success"] = result["success"] and amendment["success"]
    if args.write_amendment_record:
        if not args.frozen_commit:
            parser.error("--write-amendment-record requires --frozen-commit")
        result["amendmentRecord"] = write_amendment_record(
            result["amendment"], args.frozen_commit
        )
    if args.validate_amendment_2 or args.write_amendment_2_record:
        amendment_2 = validate_amendment_2()
        result["amendment2"] = amendment_2
        result["success"] = result["success"] and amendment_2["success"]
    if args.write_amendment_2_record:
        if not args.frozen_commit:
            parser.error("--write-amendment-2-record requires --frozen-commit")
        result["amendment2Record"] = write_amendment_2_record(
            result["amendment2"], args.frozen_commit
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
