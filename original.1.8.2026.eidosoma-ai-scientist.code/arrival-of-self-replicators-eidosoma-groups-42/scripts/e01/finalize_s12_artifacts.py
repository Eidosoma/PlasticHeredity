#!/usr/bin/env python3
"""Finalize and validate compact artifacts for E01-S12-STRICT-MRR-v1.0.0."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
STEP_ROOT = Path("/artifacts/research_steps/S12")
PREREG = REPO / "configs/e01/s12_strict_mrr_preregistration.yaml"
RUNNER = REPO / "scripts/e01/run_s12_strict_mrr.py"
ALLOWED_CLAIM_STATUSES = {
    "SUPPORTED",
    "DIRECTIONALLY_SUPPORTED",
    "NOT_SUPPORTED_WITHIN_STRICT_SCOPE",
    "UNDERDETERMINED",
    "NOT_EVALUATED",
}


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("e01_s12_finalizer_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen S12 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def correct_derived_metadata() -> dict[str, Any]:
    """Correct two outcome-independent derived metadata bugs without recomputation."""

    numerical_path = STEP_ROOT / "numerical_validation.json"
    numerical = json_read(numerical_path)
    numerical["expectedCheckpointCount"] = 36
    numerical["allPassed"] = len(numerical["rows"]) == 36 and all(
        row["status"] == "PASS" for row in numerical["rows"]
    )
    numerical["metadataCorrection"] = {
        "status": "CORRECTED_WITHOUT_SCIENTIFIC_RECOMPUTATION",
        "field": "expectedCheckpointCount",
        "oldValue": 72,
        "newValue": 36,
        "derivation": "3_matrices_x_2_preprocessing_x_3_checkpoints_x_2_redundancies",
        "scientificValuesChanged": False,
    }
    json_write(numerical_path, numerical)

    association_frame = pd.read_csv(STEP_ROOT / "association_results.csv")
    association_summary: dict[str, dict[str, Any]] = {}
    for row in association_frame[association_frame["rowType"] == "summary"].to_dict(
        "records"
    ):
        association_summary[
            f"{row['estimandId']}::{row['preprocessingId']}::{row['redundancyId']}"
        ] = row
    whole = pd.read_csv(STEP_ROOT / "whole_descriptive_analysis.csv").replace(
        {np.nan: None}
    )
    gate = json_read(STEP_ROOT / "intervention_feasibility_gate.json")
    intervention = pd.read_csv(STEP_ROOT / "intervention_results.csv").replace(
        {np.nan: None}
    )
    runner = load_runner()
    claims = runner.classify_claims(
        association_summary,
        whole.to_dict("records"),
        gate,
        intervention.to_dict("records"),
    )
    pd.DataFrame(claims).to_csv(
        STEP_ROOT / "claim_status_matrix.csv", index=False, lineterminator="\n"
    )

    validation = json_read(STEP_ROOT / "validation_summary.json")
    validation["checks"]["checkpointCardinalityMetadataCorrected"] = True
    validation["checks"]["sparseInterventionClaimBoundaryCorrected"] = True
    validation["success"] = all(validation["checks"].values())
    validation["status"] = (
        "COMPUTATION_VALIDATION_PASS"
        if validation["success"]
        else "COMPUTATION_VALIDATION_FAIL_WITH_PRESERVED_SOURCE_LIMITS"
    )
    json_write(STEP_ROOT / "validation_summary.json", validation)

    summary = json_read(STEP_ROOT / "run_summary.json")
    summary["validationSuccess"] = validation["success"]
    summary["validationChecks"] = validation["checks"]
    summary["claimStatusCounts"] = {
        str(key): int(value)
        for key, value in pd.Series(
            [row["status"] for row in claims]
        ).value_counts().items()
    }
    summary["derivedMetadataCorrections"] = [
        "expected_checkpoint_count_72_to_36_arithmetic_correction",
        "sparse_intervention_claims_restored_to_frozen_UNDERDETERMINED_boundary",
    ]
    json_write(STEP_ROOT / "run_summary.json", summary)
    return {
        "numerical": numerical,
        "claims": claims,
        "validation": validation,
        "runSummary": summary,
    }


def artifact_checks() -> dict[str, Any]:
    config = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    extras = {
        "candidate_pilot_results.json",
        "execution_attempt_001_failure.json",
        "intervention_branch_status.json",
        "intervention_event_logs.parquet",
        "intervention_expanding_estimates.parquet",
        "intervention_labels.parquet",
        "intervention_partition_history.parquet",
        "intervention_phi_summary.csv",
        "preregistration_amendment_1.yaml",
        "preregistration_amendment_2.yaml",
        "preregistration_amendment_2_record.json",
        "preregistration_amendment_record.json",
        "run_summary.json",
        "whole_descriptive_analysis.csv",
        "whole_trajectory_local_values.parquet",
    }
    expected = set(config["requiredOutputs"]) | extras
    files = {
        path.relative_to(STEP_ROOT).as_posix(): path
        for path in STEP_ROOT.rglob("*")
        if path.is_file()
    }
    generated_after_this_check = {"artifact_completeness.json"}
    missing = sorted(
        name
        for name in expected
        if name not in files and name not in generated_after_this_check
    )
    empty = sorted(name for name, path in files.items() if path.stat().st_size == 0)

    matrices = np.load(STEP_ROOT / "baseline_matrices.npz")
    matrix_check = matrices.files == [f"matrix_{index:02d}" for index in range(12)]
    matrix_check &= all(
        matrices[name].shape == (100, 100)
        and matrices[name].dtype == np.float64
        and np.all(np.isfinite(matrices[name]))
        for name in matrices.files
    )
    observations = pd.read_parquet(STEP_ROOT / "baseline_observations.parquet")
    expanding = pd.read_parquet(STEP_ROOT / "expanding_estimates.parquet")
    post_fission = pd.read_parquet(STEP_ROOT / "post_fission_estimates.parquet")
    intervention_observations = pd.read_parquet(
        STEP_ROOT / "intervention_trajectories.parquet"
    )
    intervention_expanding = pd.read_parquet(
        STEP_ROOT / "intervention_expanding_estimates.parquet"
    )
    candidates = pd.read_parquet(STEP_ROOT / "intervention_candidate_scores.parquet")
    actions = pd.read_parquet(STEP_ROOT / "intervention_action_log.parquet")
    whole = pd.read_parquet(STEP_ROOT / "whole_trajectory_estimates.parquet")
    claims = pd.read_csv(STEP_ROOT / "claim_status_matrix.csv")
    seeds = pd.read_parquet(STEP_ROOT / "seed_manifest.parquet")
    trajectory_manifest = json_read(STEP_ROOT / "trajectory_manifest.json")
    gate = json_read(STEP_ROOT / "intervention_feasibility_gate.json")
    pairing = json_read(STEP_ROOT / "intervention_pairing_audit.json")

    estimate_values_valid = bool(
        expanding.loc[
            expanding["status"] == "ELIGIBLE_NUMERIC_STRICT_EXPANDING", "value"
        ]
        .notna()
        .all()
        and expanding.loc[
            expanding["status"] != "ELIGIBLE_NUMERIC_STRICT_EXPANDING", "value"
        ]
        .isna()
        .all()
        and (
            expanding.loc[
                expanding["status"] == "ELIGIBLE_NUMERIC_STRICT_EXPANDING", "nEff"
            ]
            >= 512
        ).all()
    )
    candidate_replay_exact = bool(
        (
            (candidates["score"] == candidates["replayScore"])
            | (candidates["score"].isna() & candidates["replayScore"].isna())
        ).all()
    )
    checks = {
        "allExpectedFilesPresent": not missing,
        "allFilesNonempty": not empty,
        "exactTwelveFiniteMatrices": bool(matrix_check),
        "trajectoryManifestExactTwelve": trajectory_manifest["baselineCount"] == 12
        and len(trajectory_manifest["trajectories"]) == 12,
        "baselineEstimateCardinality": len(expanding) == len(observations) * 4,
        "postFissionEstimateCardinality": len(post_fission) == 12 * 100 * 4,
        "baselineStatusValueContract": estimate_values_valid,
        "interventionCountRule": gate["success"]
        and intervention_observations["trajectoryId"].nunique() == 18,
        "interventionEstimateCardinality": len(intervention_expanding)
        == len(intervention_observations) * 4,
        "candidateReplayExact": candidate_replay_exact,
        "allCandidateScoresStrictEligible": bool(
            (candidates["status"] == "ELIGIBLE_NUMERIC_STRICT_EXPANDING").all()
        ),
        "actionLogComplete": len(actions) == 18 * 100,
        "pairingAudit": pairing["success"],
        "wholeScopeLabelExact": bool(
            (whole["scopeLabel"] == "DESCRIPTIVE_NONPROSPECTIVE").all()
        ),
        "claimMatrixComplete": len(claims) == 59
        and set(claims["status"]).issubset(ALLOWED_CLAIM_STATUSES),
        "sparseInterventionClaimsUnderdetermined": bool(
            (
                claims.set_index("claimId").loc[
                    ["E01-C046", "E01-C054", "E01-C058", "E01-C059"],
                    "status",
                ]
                == "UNDERDETERMINED"
            ).all()
        ),
        "nineSeedPurposesPresent": seeds["purpose"].nunique() == 9,
    }
    return {
        "expected": sorted(expected),
        "missing": missing,
        "empty": empty,
        "checks": checks,
        "success": all(checks.values()),
        "counts": {
            "files": len(files),
            "baselineObservations": len(observations),
            "baselineExpandingRows": len(expanding),
            "interventionObservations": len(intervention_observations),
            "interventionExpandingRows": len(intervention_expanding),
            "candidateRows": len(candidates),
            "actionRows": len(actions),
            "claimRows": len(claims),
        },
    }


def write_manifest(implementation_commit: str) -> None:
    records = []
    for path in sorted(STEP_ROOT.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(STEP_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    aggregate_payload = "".join(
        f"{row['sha256']}  {row['path']}\n" for row in records
    ).encode("utf-8")
    manifest = {
        "schema": "eidosoma.e01.s12_artifact_manifest.v1",
        "researchStepId": "S12",
        "stepNumber": 12,
        "preregistrationVersion": "E01-S12-STRICT-MRR-v1.0.0",
        "implementationCommit": implementation_commit,
        "manifestSelfExcluded": True,
        "fileCountExcludingManifest": len(records),
        "aggregateSha256": hashlib.sha256(aggregate_payload).hexdigest(),
        "files": records,
        "bundlePointer": "/artifacts/E01_forensic_replication_bundle/data/s12_strict_mrr/bundle_pointer.json",
    }
    json_write(STEP_ROOT / "artifact_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-report", action="store_true")
    args = parser.parse_args()
    if (
        args.require_report
        and not (STEP_ROOT / "research_step_full_results.md").is_file()
    ):
        raise RuntimeError(
            "research_step_full_results.md must exist before finalization"
        )
    if git_output("status", "--short"):
        raise RuntimeError("finalization requires a clean committed worktree")
    implementation_commit = git_output("rev-parse", "HEAD")
    remote_commit = git_output(
        "ls-remote", "origin", "refs/heads/eidosoma/groups/42"
    ).split()[0]
    if implementation_commit != remote_commit:
        raise RuntimeError("finalization commit is not pushed")

    corrected = correct_derived_metadata()
    validation = corrected["validation"]
    true_checks = sum(bool(value) for value in validation["checks"].values())
    total_checks = len(validation["checks"])
    caveats = [
        "Pinned phyid local Gaussian density underflow made 10 of 48 whole-trajectory source-atom branches ineligible.",
        "Only 22 of 36 frozen source/CPU/GPU checkpoints passed the joint 1e-10 policy; strict direct expanding values were retained, not replaced.",
        "Only one of 1,090 post-G0 treated action opportunities was separable; intervention claims are underdetermined.",
        "Historical H>0.9 labels are source-traceable retrospective labels, not author-code identity.",
        "Fixed-window, pre-eligibility, early-warning, every-fission, Figure 6, and Table 1 claims remain unavailable.",
    ]
    predicted_files = sorted(
        {
            path.relative_to(STEP_ROOT).as_posix()
            for path in STEP_ROOT.rglob("*")
            if path.is_file()
        }
        | {"artifact_completeness.json", "artifact_manifest.json", "status.json"}
    )
    status = {
        "researchStepId": "S12",
        "stepNumber": 12,
        "success": False,
        "status": "COMPLETED_CONSTRAINING_WITH_PRESERVED_VALIDATION_FAILURES",
        "artifactsWritten": predicted_files,
        "validationResult": {
            "status": validation["status"],
            "checksPassed": true_checks,
            "checksTotal": total_checks,
            "failedChecks": [
                key for key, passed in validation["checks"].items() if not passed
            ],
            "artifactCompleteness": "PENDING_FINALIZER_CHECK",
        },
        "outcomeClassification": "CONSTRAINING_CONTRADICTORY",
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": "Do not run S13. Close E01 Phi-r reconstruction as restricted and underdetermined outside the valid post-eligibility estimand; consider alternative causal-architecture work only in a separately preregistered E02 after human review.",
    }
    json_write(STEP_ROOT / "status.json", status)

    excluded_final_metadata = {
        "artifact_manifest.json",
        "artifact_completeness.json",
        "status.json",
    }
    measured = sum(
        path.stat().st_size
        for path in STEP_ROOT.rglob("*")
        if path.is_file()
        and path.relative_to(STEP_ROOT).as_posix() not in excluded_final_metadata
    )
    cache_bytes = sum(
        path.stat().st_size
        for path in Path("/cache/e01_s12").rglob("*")
        if path.is_file()
    )
    storage = {
        "schema": "eidosoma.e01.s12_storage_validation.v1",
        "researchStepId": "S12",
        "measurementBoundary": "payload_files_excluding_three_self_referential_final_metadata_files",
        "payloadBytes": measured,
        "cacheBytes": cache_bytes,
        "combinedPayloadAndCacheBytes": measured + cache_bytes,
        "byteCeiling": 20 * 1024**3,
        "freeBytes": shutil.disk_usage(STEP_ROOT).free,
        "forbiddenArtifactCacheEntries": [
            path.relative_to(STEP_ROOT).as_posix()
            for path in STEP_ROOT.rglob("*")
            if path.is_file()
            and (
                "__pycache__" in path.parts
                or path.suffix in {".pyc", ".pyo", ".so", ".o"}
            )
        ],
    }
    storage["success"] = (
        storage["combinedPayloadAndCacheBytes"] <= storage["byteCeiling"]
        and not storage["forbiddenArtifactCacheEntries"]
    )
    json_write(STEP_ROOT / "storage_validation.json", storage)

    write_manifest(implementation_commit)
    completeness = artifact_checks()
    completeness.update(
        {
            "schema": "eidosoma.e01.s12_artifact_completeness.v1",
            "researchStepId": "S12",
            "stepNumber": 12,
            "implementationCommit": implementation_commit,
        }
    )
    json_write(STEP_ROOT / "artifact_completeness.json", completeness)
    if not completeness["success"]:
        raise RuntimeError(f"artifact completeness failed: {completeness}")

    status["validationResult"]["artifactCompleteness"] = "PASS"
    json_write(STEP_ROOT / "status.json", status)
    write_manifest(implementation_commit)
    print(
        json.dumps(
            {
                "success": True,
                "implementationCommit": implementation_commit,
                "claimStatusCounts": corrected["runSummary"]["claimStatusCounts"],
                "validationStatus": validation["status"],
                "artifactCompleteness": completeness["success"],
                "artifactFileCount": completeness["counts"]["files"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
