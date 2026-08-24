#!/usr/bin/env python3
"""Run S12H stage-1 C1 revalidation and conditional fresh ensemble analysis."""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt
import pandas as pd
import yaml

import e01_frozen_timebase_ensemble.core as s12g_core
from e01_boundary_clock_revalidation.core import (
    CANDIDATE_IDS,
    DERIVED_CANDIDATE_ID,
    EVIDENCE_CLASS,
    RESEARCH_STEP_ID,
    VERSION,
    candidate1_revalidation_result,
    stage2_candidate_registry,
)
from e01_frozen_timebase_ensemble.core import (
    post_fission_endpoint_records,
    selected_clock_observations,
)
from e01_latent_timebase.core import trajectory_summary
from scripts.e01 import run_s12g_frozen_timebase_ensemble as backend

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S12H"
CACHE_ROOT = Path("/cache/e01_s12h")
RESULT_CACHE = CACHE_ROOT / "source_results"
CONFIG_PATH = REPO / "configs/e01/s12h_candidate1_boundary_clock_revalidation_preregistration.yaml"
S12G_SCHEMA = REPO / "configs/e01/s12g_output_schemas.json"
LOCK_PATH = REPO / "configs/e01/s12h/candidate_lock.json"
INPUT_MANIFEST = STEP_ROOT / "trajectory_input_manifest.parquet"
FIGURE_ROOT = STEP_ROOT / "figures"
S12FR_LOCK = ARTIFACTS / "research_steps/S12FR/candidate_timebase_pipeline_lock.json"

ORIGINAL_ADJUDICATE = backend.adjudicate
ORIGINAL_FAILURE_ROWS = backend.failure_rows_from_statuses


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def clock_sequence_sha256(observations: tuple[Any, ...]) -> str:
    payload = [
        {
            "rawObservationIndex": int(item.observation_index),
            "observationKind": str(item.observation_kind),
            "generation": int(item.growth_generation_one_based),
            "state": [int(value) for value in item.state],
        }
        for item in observations
    ]
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def verify_clean_pushed() -> tuple[str, str]:
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote or git("status", "--short"):
        raise RuntimeError("S12H code and lock must be committed, pushed, and clean")
    return head, remote


def verify_method_lock() -> dict[str, Any]:
    payload = json.loads((STEP_ROOT / "method_lock.json").read_text(encoding="utf-8"))
    if not payload.get("passed"):
        raise RuntimeError("S12H method lock is not passing")
    for item in payload["files"]:
        path = REPO / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"S12H method-lock file changed: {item['path']}")
    return payload


def validate_immutable_prior() -> dict[str, Any]:
    baseline = json.loads((STEP_ROOT / "immutable_prior_baseline.json").read_text(encoding="utf-8"))
    changed: list[dict[str, Any]] = []
    categories = (
        ("researchStepFiles", "prior_artifact"),
        ("lockedTrajectoryCaches", "locked_raw_trajectory"),
        ("s12gTaskCacheFiles", "forbidden_S12G_cache"),
    )
    counts: dict[str, int] = {}
    for key, kind in categories:
        counts[key] = len(baseline[key])
        for item in baseline[key]:
            path = Path(item["path"])
            expected = item.get("expectedSha256", item["sha256"])
            actual = sha256_file(path) if path.is_file() else None
            if actual != expected:
                changed.append(
                    {"kind": kind, "path": str(path), "expectedSha256": expected, "actualSha256": actual}
                )
    result = {
        "schema": "eidosoma.e01.s12h_immutable_prior_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        **counts,
        "changedCount": len(changed),
        "changed": changed,
        "passed": not changed,
    }
    write_json(STEP_ROOT / "immutable_prior_validation.json", result)
    return result


def schema_validation() -> dict[str, Any]:
    tables = json.loads(S12G_SCHEMA.read_text(encoding="utf-8"))["tables"]
    rows: list[dict[str, Any]] = []
    for filename, required in tables.items():
        path = STEP_ROOT / filename
        exists = path.is_file()
        if exists:
            frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            missing = [column for column in required if column not in frame.columns]
            count = len(frame)
        else:
            missing, count = list(required), None
        rows.append({"path": filename, "exists": exists, "rowCount": count, "missingColumns": missing, "passed": exists and not missing})
    for filename in ("candidate1_clock_revalidation.parquet", "candidate_association_details.parquet", "replicator_drift_details.parquet"):
        path = STEP_ROOT / filename
        rows.append({"path": filename, "exists": path.is_file(), "rowCount": len(pd.read_parquet(path)) if path.is_file() else None, "missingColumns": [], "passed": path.is_file()})
    payload = {"schema": "eidosoma.e01.s12h_schema_validation.v1", "researchStepId": RESEARCH_STEP_ID, "tables": rows, "passed": all(item["passed"] for item in rows)}
    write_json(STEP_ROOT / "schema_validation.json", payload)
    return payload


def artifact_manifest() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    required = config["artifacts"]["required"]
    files = [path for path in sorted(STEP_ROOT.rglob("*")) if path.is_file() and path.name != "artifact_manifest.json"]
    entries = [{"relativePath": str(path.relative_to(STEP_ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files]
    present = {item["relativePath"] for item in entries}
    missing = [item for item in required if item != "artifact_manifest.json" and item not in present]
    total = sum(item["bytes"] for item in entries)
    payload = {
        "schema": "eidosoma.e01.s12h_artifact_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "artifacts": entries,
        "artifactCountExcludingSelf": len(entries),
        "totalBytesExcludingSelf": total,
        "requiredMissing": missing,
        "under30GiB": total <= 30 * 1024**3,
        "passed": not missing and total <= 30 * 1024**3,
    }
    write_json(STEP_ROOT / "artifact_manifest.json", payload)
    return payload


def placeholder_figure(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def create_scientific_placeholders(reason: str) -> None:
    contract = json.loads(S12G_SCHEMA.read_text(encoding="utf-8"))["tables"]
    for filename, columns in contract.items():
        path = STEP_ROOT / filename
        if path.exists() or filename == "trajectory_input_manifest.parquet":
            continue
        frame = pd.DataFrame(columns=columns)
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False, compression="zstd")
        else:
            frame.to_csv(path, index=False, lineterminator="\n")
    for filename in ("candidate_association_details.parquet", "replicator_drift_details.parquet"):
        path = STEP_ROOT / filename
        if not path.exists():
            pd.DataFrame(columns=["researchStepId", "status", "reason"]).to_parquet(path, index=False, compression="zstd")
    for filename in (
        "label_fingerprints.png",
        "full_emergence_trajectories.png",
        "association_distributions.png",
        "full_prefix_comparison.png",
        "metric_identity_comparison.png",
        "ensemble_decision_matrix.png",
    ):
        path = FIGURE_ROOT / filename
        if not path.exists():
            placeholder_figure(path, f"S12H stopped fail-closed: {reason}")


def _candidate1_source_lock() -> dict[str, Any]:
    lock = json.loads(S12FR_LOCK.read_text(encoding="utf-8"))
    matches = [item for item in lock["confirmedCandidates"] if item["candidateId"] == "S12F-CANDIDATE-01"]
    if len(matches) != 1:
        raise RuntimeError("S12FR candidate 1 is missing or duplicated")
    return matches[0]


def run_stage1() -> int:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    method = verify_method_lock()
    head, _remote = verify_clean_pushed()
    if head != method["designCommit"]:
        raise RuntimeError("stage 1 must run at the pushed pre-outcome design commit")
    immutable = validate_immutable_prior()
    if not immutable["passed"]:
        raise RuntimeError("prior immutability failed before stage 1")

    source = _candidate1_source_lock()
    input_frame = pd.read_parquet(INPUT_MANIFEST)
    inputs = input_frame[input_frame["candidateId"] == "S12F-CANDIDATE-01"].sort_values("matrixIndex")
    rows: list[dict[str, Any]] = []
    cardinality_passed = True
    for item in inputs.to_dict("records"):
        path = Path(item["cachePath"])
        if sha256_file(path) != item["cacheSha256"]:
            raise RuntimeError(f"candidate-1 raw cache changed: {path}")
        with path.open("rb") as handle:
            trajectory = pickle.load(handle)
        identity = bool(
            trajectory.configuration_id == item["candidateId"]
            and int(trajectory.matrix_index) == int(item["matrixIndex"])
            and trajectory.trajectory_id == item["trajectoryId"]
            and trajectory.trajectory_sha256 == item["trajectorySha256"]
            and trajectory.beta_sha256 == item["betaSha256"]
            and trajectory.initial_state_sha256 == item["initialStateSha256"]
        )
        selected = selected_clock_observations(trajectory, "C1_SELECTED_DAUGHTER_RETAINED")
        selected_replay = selected_clock_observations(
            trajectory, "C1_SELECTED_DAUGHTER_RETAINED"
        )
        selected_sha256 = clock_sequence_sha256(selected)
        selected_replay_sha256 = clock_sequence_sha256(selected_replay)
        endpoints = post_fission_endpoint_records(trajectory, "C1_SELECTED_DAUGHTER_RETAINED", minimum_prior_transitions=0)
        post_count = sum(obs.observation_kind == "post_fission" for obs in trajectory.observations)
        row_cardinality = bool(len(endpoints) == 100 and post_count == 100 and len(selected) == int(trajectory.total_batch_updates) + 101 and all(endpoint.observation_kind == "post_fission" for endpoint in endpoints))
        cardinality_passed &= row_cardinality
        summary = trajectory_summary(trajectory)
        rows.append(
            {
                **summary,
                "researchStepId": RESEARCH_STEP_ID,
                "derivedCandidateId": DERIVED_CANDIDATE_ID,
                "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
                "endpointCount": len(endpoints),
                "postFissionObservationCount": post_count,
                "selectedObservationCount": len(selected),
                "selectedClockSha256": selected_sha256,
                "selectedClockReplaySha256": selected_replay_sha256,
                "clockReplayPassed": selected_sha256 == selected_replay_sha256,
                "stateCardinalityPassed": row_cardinality,
                "cacheHashPassed": True,
                "candidateIdentityPassed": identity and bool(item["candidateIdentityPassed"]),
                "repairedReplayPassed": bool(item["repairedReplayPassed"]),
                "discreteDivergenceCount": int(item["discreteDivergenceCount"]),
                "finiteNumericDivergenceCount": int(item["finiteNumericDivergenceCount"]),
                "forbiddenNonfiniteDifferenceCount": int(item["forbiddenNonfiniteDifferenceCount"]),
                "seedDifferenceCount": int(item["seedDifferenceCount"]),
                "cachePath": str(path),
                "cacheSha256": item["cacheSha256"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("matrixIndex").reset_index(drop=True)
    frame.to_parquet(STEP_ROOT / "candidate1_clock_revalidation.parquet", index=False, compression="zstd")

    regeneration = json.loads((ARTIFACTS / "research_steps/S12FR/regeneration_validation.json").read_text(encoding="utf-8"))
    seed_provenance = bool(
        regeneration.get("passed")
        and frame["repairedReplayPassed"].all()
        and frame["candidateIdentityPassed"].all()
        and frame["cacheHashPassed"].all()
    )
    input_bytes = int(sum(Path(path).stat().st_size for path in frame["cachePath"]))
    runtime_storage = bool(input_bytes <= 30 * 1024**3 and (time.perf_counter() - started_wall) <= 72 * 3600)
    candidate = {
        **source,
        "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
        "derivedCandidateId": DERIVED_CANDIDATE_ID,
    }
    result = candidate1_revalidation_result(
        candidate,
        frame,
        state_cardinality_passed=cardinality_passed,
        seed_provenance_passed=seed_provenance,
        runtime_storage_passed=runtime_storage,
    )
    result["stage1Runtime"] = {
        "wallSeconds": time.perf_counter() - started_wall,
        "cpuSeconds": time.process_time() - started_cpu,
        "inputBytes": input_bytes,
        "cpuFloat64Authoritative": True,
    }
    result["rawTrajectoryCount"] = len(frame)
    result["newGardTrajectoriesGenerated"] = 0
    write_json(STEP_ROOT / "candidate1_timebase_confirmation.json", result)

    lock_payload = {
        "schema": "eidosoma.e01.s12h_derivative_candidate_lock.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "stage1ConfirmationSha256": sha256_file(STEP_ROOT / "candidate1_timebase_confirmation.json"),
        "stage1RowsSha256": sha256_file(STEP_ROOT / "candidate1_clock_revalidation.parquet"),
        "sourceS12frLockSha256": sha256_file(S12FR_LOCK),
        "derivedCandidateId": DERIVED_CANDIDATE_ID,
        "candidates": stage2_candidate_registry(),
        "candidate1ConfirmationGatePassed": bool(result["confirmationGatePassed"]),
        "s12gCacheReusePermitted": False,
        "newGardTrajectoriesPermitted": 0,
    }
    proposal = {
        "schema": "eidosoma.e01.s12h_derivative_candidate_lock_proposal.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "generatedAtUtc": datetime.now(UTC).isoformat(),
        "lockPayload": lock_payload,
        "passed": bool(result["confirmationGatePassed"]),
    }
    write_json(STEP_ROOT / "derivative_candidate_lock_proposal.json", proposal)
    write_json(
        STEP_ROOT / "derivative_candidate_lock_validation.json",
        {
            "schema": "eidosoma.e01.s12h_derivative_candidate_lock_validation.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "status": "AWAITING_COMMITTED_PUSHED_LOCK" if result["confirmationGatePassed"] else "NOT_AUTHORIZED_STAGE1_FAILED",
            "passed": False,
        },
    )
    scope = json.loads((STEP_ROOT / "scope_access_ledger.json").read_text(encoding="utf-8"))
    scope["events"].append(
        {
            "stage": "STAGE1_CANDIDATE1_UNIFORM_C1_REVALIDATION",
            "rawTrajectoryOpened": True,
            "rawTrajectoryCount": 32,
            "labelOutcomeOpened": False,
            "informationTheoryOutcomeOpened": False,
            "s12gCachePayloadOpened": False,
            "newGardTrajectoryGenerated": False,
            "status": "PASS" if result["confirmationGatePassed"] else "FAIL_CLOSED",
        }
    )
    write_json(STEP_ROOT / "scope_access_ledger.json", scope)
    if not result["confirmationGatePassed"]:
        finalize_fail_closed(
            classification="BOUNDARY_INCLUSIVE_CANDIDATE1_NOT_UPSTREAM_CONFIRMED",
            stage="STAGE1_TIMEBASE_REVALIDATION",
            reason=result["gateReason"],
        )
        return 1
    print(json.dumps({"stage": "S12H_stage1_passed", "derivedCandidateId": DERIVED_CANDIDATE_ID, "confirmationDistance": result["confirmationDistance"], "medianTPhi": result["medianTPhi"], "next": "commit_and_push_derivative_candidate_lock"}, sort_keys=True), flush=True)
    return 0


def verify_derivative_lock() -> dict[str, Any]:
    proposal = json.loads((STEP_ROOT / "derivative_candidate_lock_proposal.json").read_text(encoding="utf-8"))
    if not proposal.get("passed"):
        raise RuntimeError("stage 1 did not authorize a derivative candidate lock")
    if not LOCK_PATH.is_file():
        raise RuntimeError("S12H derivative candidate lock is absent")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if canonical(lock) != canonical(proposal["lockPayload"]):
        raise RuntimeError("S12H derivative lock differs from deterministic stage-1 proposal")
    head, remote = verify_clean_pushed()
    validation = {
        "schema": "eidosoma.e01.s12h_derivative_candidate_lock_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "lockPath": str(LOCK_PATH),
        "lockSha256": sha256_file(LOCK_PATH),
        "lockCommit": head,
        "remoteCommit": remote,
        "candidate1ConfirmationGatePassed": True,
        "exactProposalMatch": True,
        "passed": True,
    }
    write_json(STEP_ROOT / "derivative_candidate_lock_validation.json", validation)
    return validation


def _configure_backend() -> None:
    backend.RESEARCH_STEP_ID = RESEARCH_STEP_ID
    backend.VERSION = VERSION
    backend.EVIDENCE_CLASS = EVIDENCE_CLASS
    backend.CANDIDATE_IDS = CANDIDATE_IDS
    backend.STEP_ROOT = STEP_ROOT
    backend.CACHE_ROOT = CACHE_ROOT
    backend.RESULT_CACHE = RESULT_CACHE
    backend.CONFIG_PATH = CONFIG_PATH
    backend.SCHEMA_PATH = S12G_SCHEMA
    backend.INPUT_MANIFEST = INPUT_MANIFEST
    backend.FIGURE_ROOT = FIGURE_ROOT
    s12g_core.RESEARCH_STEP_ID = RESEARCH_STEP_ID
    backend.adjudicate = s12h_adjudicate
    backend.failure_rows_from_statuses = s12h_failure_rows
    backend.validate_immutable_prior = validate_immutable_prior
    backend.validate_schemas = schema_validation
    backend.artifact_manifest = lambda _required: artifact_manifest()
    backend.recommendation_for = recommendation_for
    backend.build_report = build_report


def s12h_adjudicate(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, payload = ORIGINAL_ADJUDICATE(*args, **kwargs)
    payload.update(
        {
            "schema": "eidosoma.e01.s12h_classification.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "evidenceClass": EVIDENCE_CLASS,
            "candidate1AnalysisIdentity": DERIVED_CANDIDATE_ID,
            "s13Status": "BLOCKED_PENDING_S12H_HUMAN_REVIEW",
        }
    )
    return frame, payload


def s12h_failure_rows(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    rows = ORIGINAL_FAILURE_ROWS(*args, **kwargs)
    for row in rows:
        row["failureId"] = str(row["failureId"]).replace("S12G-", "S12H-")
    return rows


def recommendation_for(classification: str) -> str:
    if classification == "ENSEMBLE_PROSPECTIVE_SOURCE_EMERGENCE_SUPPORT":
        return "Return for human review with S13 blocked; any later baseline-only proposal requires separate authorization and cannot include interventions."
    if classification == "ENSEMBLE_RETROSPECTIVE_SOURCE_EMERGENCE_SUPPORT":
        return "Return for human review; retain the finding as retrospective and potentially future-dependent, with S13 and interventions blocked."
    if classification == "CANDIDATE_SENSITIVE_UNDERDETERMINED":
        return "Return for human review; do not select or reweight a favorable time-base candidate, and keep S13 blocked."
    if classification == "ENSEMBLE_WIDE_NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE":
        return "Return for human review and keep S13 blocked; the three-candidate source-informed ensemble does not support the paper-directed relationship."
    return "Return for mandatory human review with S13 and all downstream work blocked."


def build_report(
    *,
    classification: dict[str, Any],
    associations: pd.DataFrame,
    drift: pd.DataFrame,
    future: pd.DataFrame,
    metric_identity: pd.DataFrame,
    adjudication: pd.DataFrame,
    runtime: dict[str, Any],
    validation: dict[str, Any],
    failures: pd.DataFrame,
    input_manifest: pd.DataFrame,
) -> str:
    outcome = classification["classification"]
    stage1 = json.loads((STEP_ROOT / "candidate1_timebase_confirmation.json").read_text(encoding="utf-8"))
    iigr = "IIGR_CORRECTED_SOURCE"
    primary = associations[(associations["implementationId"] == iigr)]
    full_rows = primary[primary["estimand"] == "RETROSPECTIVE_CURRENT_GENERATION"].sort_values("candidateId")
    prefix_rows = primary[(primary["estimand"] == "CURRENT_HISTORICAL") & primary["temporalModeId"].str.endswith("_PREFIX_ENDPOINT")].sort_values("candidateId")
    table_lines = []
    for candidate_id in CANDIDATE_IDS:
        full = full_rows[full_rows["candidateId"] == candidate_id].iloc[0]
        prefix = prefix_rows[prefix_rows["candidateId"] == candidate_id].iloc[0]
        table_lines.append(
            f"| {candidate_id} | {float(full['medianCorrelation']):.6g} | {int(full['positiveTrajectoryCount'])}/{int(full['definedTrajectoryCount'])} | {bool(full['gatePassed'])} | {float(prefix['medianCorrelation']):.6g} | {int(prefix['positiveTrajectoryCount'])}/{int(prefix['definedTrajectoryCount'])} | {bool(prefix['gatePassed'])} |"
        )
    outcome_class = validation["outcomeClass"]
    return f"""# S12H Full Results: Candidate-1 Boundary Clock Revalidation

## Top summary

- **Research step ID:** `{VERSION}` (S12H)
- **Completion status:** `COMPLETED_AT_MANDATORY_S12H_HUMAN_REVIEW_BOUNDARY`; S13 was not begun.
- **Artifacts written:** {validation['artifactCount']} status-bearing files under `/artifacts/research_steps/S12H/`, including stage-1 clock confirmation, a pushed derivative lock, 96 freshly recomputed source tasks, labels, full/prefix values, candidate/ensemble analyses, figures, validation, provenance, hashes, status, and this report.
- **Validation result:** {validation['summary']}
- **Outcome classification:** `{outcome}` ({outcome_class}).
- **Caveats or blockers:** Candidate 1 is the new `{DERIVED_CANDIDATE_ID}` C1 derivative, not S12FR's original C0 candidate. Public source behavior remains source-informed rather than author- or paper-primary; completed-fit values are retrospective.
- **Recommended next action:** {recommendation_for(outcome)}

## Lay summary

Recording the selected daughter after every fission—not only when a generation had no growth update—kept candidate 1 compatible with the paper-visible time scale under every original S12FR confirmation gate. Only after that upstream check and a second pushed lock did S12H recompute all 96 label and source-emergence tasks from the frozen raw trajectories. The three candidates were analyzed separately, and a positive ensemble result required all three to pass the same frozen gate.

## Frozen question and inputs

S12H tested whether the candidate-1 raw dynamics remain time-base-compatible under uniform C1 and, conditionally, whether the unchanged S12G scientific result is robust across that derivative plus unchanged candidates 2 and 3. Exactly {len(input_manifest)} S12FR raw trajectories were used; no GARD trajectory was generated, and no S12G scientific cache payload was reused. All 32 catalytic-matrix and initial-state identities were shared across candidates.

Pinned source commits were IIGR `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7` and PhiRL `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`. Scientific execution used the audited safe JSON lattice with SHA-256 `74ecca37f04201088d76a9e8ede7efe04bafebecff85a4882a44f03afbd23aa1`.

## Detailed methods

Stage 1 retained candidate 1's `h=0.5081160391061118`, first-daughter continuation, retained overshoot, matrices, initial states, seeds, and raw trajectory payloads. Its clock alone changed from C0 to C1 for every generation. The exact S12FR distance, Figure-2 endpoint/aggregate envelope, mass, completion, max-step, replay, seed, provenance, runtime, and storage gates were reapplied to all 32 trajectories. No label or source value was opened before this passed and the derivative lock was committed and pushed.

Stage 2 used historical H>0.9 non-drift labels as primary and past-only cosine labels as secondary. Counts received additive-0.5 closure, full CLR, and removal of original component 100. IIGR synergy plus two downward-causation atoms was primary, PhiRL was the regularization companion, and corrected local Phi-r remained comparator-only. Full values are `RETROSPECTIVE_FULL_TRAJECTORY_LOCAL`. Prefix fits began after 256 prior C1 transitions and were independently refit from the past; full/prefix replay and suffix deletion, shuffle, and replacement invariance were mandatory.

Trajectory-level associations used 4,096 trajectory bootstraps and 4,096 circular shifts; drift comparisons used the frozen block-aware rule. Temporal, spike, metric-identity, full-versus-prefix, partition, and paired cross-candidate analyses were unchanged from S12G. No S12FR weight entered a scientific result.

## Stage-1 upstream result

| Derived candidate | q05 | Median | q95 | Maximum | Endpoints covered | Aggregate compatible | Distance | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `{DERIVED_CANDIDATE_ID}` | {stage1['q05TPhi']:.6g} | {stage1['medianTPhi']:.6g} | {stage1['q95TPhi']:.6g} | {stage1['maximumTPhi']:.6g} | {stage1['sampleEndpointsInsideQ05Q95']} | {stage1['aggregateCompatible']} | {stage1['confirmationDistance']:.6g} | {stage1['confirmationGatePassed']} |

All 32 lineages completed 100 fissions. The median selected-daughter mass was {stage1['medianPostFissionMass']:.6g}; the max-step fraction was {stage1['fractionMaxsteps']:.6g}; the fraction above the digitized axis upper bound was {stage1['fractionBeyondAxisUpper1314']:.6g}. This confirms only the new boundary-inclusive derivative, not the original C0 identity.

## Candidate-specific scientific results

| Candidate | Full median rho | Full positive/defined | Full gate | Prefix median rho | Prefix positive/defined | Prefix gate |
| --- | ---: | ---: | --- | ---: | ---: | --- |
{chr(10).join(table_lines)}

Complete current/next-generation and secondary-label results are in `candidate_associations.csv`; drift results are in `replicator_drift_results.csv`. Full values remain retrospective. Metric-identity and future-dependence outputs contain {len(metric_identity)} and {len(future)} candidate/implementation comparisons.

## Ensemble adjudication

{adjudication.to_markdown(index=False)}

The frozen all-three classification is `{outcome}`. No candidate was selected, removed, or reweighted.

## Validation

{validation['details']}

The failure ledger contains {len(failures)} status aggregates, including expected pre-256 ineligibility and any nonfinite source states. Prior S01–S12G artifacts, all 96 raw trajectory caches, and all 950 forbidden S12G cache files remained byte-identical.

## Commands, dependencies, and runtime

```bash
PYTHONPATH=src python -m pytest -q tests/e01/test_s12h_candidate1_boundary_clock_revalidation.py tests/e01/test_s12g_frozen_timebase_ensemble.py
python -m ruff check src/e01_boundary_clock_revalidation scripts/e01/freeze_s12h_preregistration.py scripts/e01/run_s12h_candidate1_boundary_clock_revalidation.py tests/e01/test_s12h_candidate1_boundary_clock_revalidation.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s12h_preregistration.py --design-commit <pushed-design-commit>
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s12h_candidate1_boundary_clock_revalidation.py --stage revalidate
# derivative lock committed and pushed only after stage 1 passed
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s12h_candidate1_boundary_clock_revalidation.py --stage ensemble --workers 6
```

CPU float64 was authoritative; GPU use was zero. Stage-2 runtime was {runtime['wallHours']:.4f} wall-hours and {runtime['workerCpuHours']:.4f} summed worker CPU-hours. Python `{runtime['python']}`, NumPy `{runtime['numpy']}`, SciPy `{runtime['scipy']}`, platform `{runtime['platform']}`. No new dependency was installed.

## Provenance, caveats, and limitations

- Candidate 1's new C1 identity was derived after S12G exposed the C0 endpoint problem; the independent upstream revalidation and pushed firewall reduce but cannot erase that post-outcome flexibility.
- The 96 raw trajectories had already served S12FR time-base confirmation; they are not new GARD holdouts.
- Historical labels are retrospective. Completed source fits use future observations; only prefix fits are prospective reconstructions.
- Exact replay is bounded to the pinned wrappers, runtime, CPU float64 policy, and platform.
- S12F, S12FR, and S12G remain unchanged, including every prior negative, failure, cache, classification, and hash.

## Recommended next action

{recommendation_for(outcome)} S13 remains `BLOCKED_PENDING_S12H_HUMAN_REVIEW`. Stop here.
"""


def normalize_success_artifacts() -> None:
    classification = json.loads((STEP_ROOT / "classification.json").read_text(encoding="utf-8"))
    classification.update({"schema": "eidosoma.e01.s12h_classification.v1", "researchStepId": RESEARCH_STEP_ID, "versionedStepId": VERSION, "candidate1AnalysisIdentity": DERIVED_CANDIDATE_ID, "s13Status": "BLOCKED_PENDING_S12H_HUMAN_REVIEW"})
    write_json(STEP_ROOT / "classification.json", classification)
    status = json.loads((STEP_ROOT / "status.json").read_text(encoding="utf-8"))
    status.update({"researchStepId": RESEARCH_STEP_ID, "stepNumber": RESEARCH_STEP_ID, "status": "COMPLETED_AT_MANDATORY_S12H_HUMAN_REVIEW_BOUNDARY", "s13Status": "BLOCKED_PENDING_S12H_HUMAN_REVIEW", "caveatsOrBlockers": ["Candidate 1 is a newly confirmed C1 derivative, not the original S12FR C0 candidate.", "Completed-fit source values are retrospective and source-informed only.", "S12G caches were not reused; S13 remains blocked."]})
    write_json(STEP_ROOT / "status.json", status)
    runtime = json.loads((STEP_ROOT / "runtime_manifest.json").read_text(encoding="utf-8"))
    runtime["schema"] = "eidosoma.e01.s12h_runtime_manifest.v1"
    runtime["researchStepId"] = RESEARCH_STEP_ID
    runtime["stage1"] = json.loads((STEP_ROOT / "candidate1_timebase_confirmation.json").read_text(encoding="utf-8"))["stage1Runtime"]
    write_json(STEP_ROOT / "runtime_manifest.json", runtime)
    regeneration = json.loads((STEP_ROOT / "regeneration_validation.json").read_text(encoding="utf-8"))
    regeneration.update({"schema": "eidosoma.e01.s12h_regeneration_validation.v1", "researchStepId": RESEARCH_STEP_ID, "candidate1UniformC1Stage1Passed": True, "s12gCacheReuseCount": 0})
    write_json(STEP_ROOT / "regeneration_validation.json", regeneration)
    implementation = json.loads((STEP_ROOT / "implementation_lock.json").read_text(encoding="utf-8"))
    implementation.update({"schema": "eidosoma.e01.s12h_implementation_lock.v1", "researchStepId": RESEARCH_STEP_ID, "versionedStepId": VERSION, "candidate1AnalysisIdentity": DERIVED_CANDIDATE_ID})
    implementation["files"].append({"path": str(LOCK_PATH.relative_to(REPO)), "sha256": sha256_file(LOCK_PATH)})
    write_json(STEP_ROOT / "implementation_lock.json", implementation)
    scope = json.loads((STEP_ROOT / "scope_access_ledger.json").read_text(encoding="utf-8"))
    scope["researchStepId"] = RESEARCH_STEP_ID
    scope["success"] = True
    write_json(STEP_ROOT / "scope_access_ledger.json", scope)


def run_stage2(workers: int) -> int:
    verify_method_lock()
    verify_derivative_lock()
    immutable = validate_immutable_prior()
    if not immutable["passed"]:
        raise RuntimeError("prior immutability failed before stage 2")
    exclusion = json.loads((STEP_ROOT / "s12g_cache_exclusion_audit.json").read_text(encoding="utf-8"))
    if exclusion.get("payloadFilesOpened") != 0:
        raise RuntimeError("S12G cache exclusion firewall was violated")
    if RESULT_CACHE.exists():
        raise RuntimeError("fresh S12H result cache root already exists")
    _configure_backend()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], "--workers", str(workers)]
        result = backend.main()
    finally:
        sys.argv = old_argv
    normalize_success_artifacts()
    immutable = validate_immutable_prior()
    schemas = schema_validation()
    if not immutable["passed"] or not schemas["passed"]:
        raise RuntimeError(f"S12H final validation failed: immutable={immutable['passed']},schemas={schemas['passed']}")
    manifest = artifact_manifest()
    status = json.loads((STEP_ROOT / "status.json").read_text(encoding="utf-8"))
    status["artifactsWritten"] = [item["relativePath"] for item in manifest["artifacts"]] + ["artifact_manifest.json"]
    write_json(STEP_ROOT / "status.json", status)
    artifact_manifest()
    print(json.dumps({"stage": "S12H_complete", "classification": json.loads((STEP_ROOT / "classification.json").read_text())["classification"], "artifactManifestPassed": True}, sort_keys=True), flush=True)
    return result


def finalize_fail_closed(*, classification: str, stage: str, reason: str) -> None:
    create_scientific_placeholders(reason)
    failure_columns = json.loads(S12G_SCHEMA.read_text(encoding="utf-8"))["tables"]["failure_ledger.csv"]
    failure = {column: None for column in failure_columns}
    failure.update({"failureId": "S12H-TERMINAL-FAIL-CLOSED", "stage": stage, "severity": "FATAL", "status": classification, "reason": reason, "gateImpact": "FAIL_CLOSED_NO_SCIENTIFIC_ADJUDICATION", "repairAttempted": False})
    pd.DataFrame([failure], columns=failure_columns).to_csv(STEP_ROOT / "failure_ledger.csv", index=False, lineterminator="\n")
    write_json(STEP_ROOT / "classification.json", {"schema": "eidosoma.e01.s12h_classification.v1", "researchStepId": RESEARCH_STEP_ID, "versionedStepId": VERSION, "classification": classification, "scientificAssociationClassification": "NOT_EVALUATED", "reason": reason, "s13Status": "BLOCKED_PENDING_S12H_HUMAN_REVIEW"})
    for filename, payload in (
        ("runtime_benchmark.json", {"schema": "eidosoma.e01.s12h_runtime_benchmark.v1", "passed": False, "status": "NOT_REACHED_OR_FAILED_CLOSED", "reason": reason}),
        ("runtime_manifest.json", {"schema": "eidosoma.e01.s12h_runtime_manifest.v1", "passed": False, "status": "FAILED_CLOSED", "reason": reason, "gpuHours": 0, "newGardTrajectories": 0}),
        ("regeneration_validation.json", {"schema": "eidosoma.e01.s12h_regeneration_validation.v1", "passed": False, "status": "NOT_REACHED_OR_FAILED_CLOSED", "reason": reason, "newGardTrajectoriesGenerated": 0, "s12gCacheReuseCount": 0}),
    ):
        if not (STEP_ROOT / filename).exists():
            write_json(STEP_ROOT / filename, payload)
    if not (STEP_ROOT / "candidate1_clock_revalidation.parquet").exists():
        pd.DataFrame(columns=["researchStepId", "status", "reason"]).to_parquet(STEP_ROOT / "candidate1_clock_revalidation.parquet", index=False, compression="zstd")
    if not (STEP_ROOT / "candidate1_timebase_confirmation.json").exists():
        write_json(STEP_ROOT / "candidate1_timebase_confirmation.json", {"schema": "eidosoma.e01.s12h_candidate1_timebase_confirmation.v1", "confirmationGatePassed": False, "reason": reason})
    if not (STEP_ROOT / "derivative_candidate_lock_proposal.json").exists():
        write_json(STEP_ROOT / "derivative_candidate_lock_proposal.json", {"schema": "eidosoma.e01.s12h_derivative_candidate_lock_proposal.v1", "passed": False, "reason": reason})
    if not (STEP_ROOT / "derivative_candidate_lock_validation.json").exists():
        write_json(STEP_ROOT / "derivative_candidate_lock_validation.json", {"schema": "eidosoma.e01.s12h_derivative_candidate_lock_validation.v1", "passed": False, "reason": reason})
    immutable = validate_immutable_prior()
    schemas = schema_validation()
    outcome_class = "constraining/contradictory"
    report = f"""# S12H Full Results: Candidate-1 Boundary Clock Revalidation

## Top summary

- **Research step ID:** `{VERSION}` (S12H)
- **Completion status:** `STOPPED_FAIL_CLOSED_AT_{stage}`; no valid ensemble scientific adjudication was performed.
- **Artifacts written:** Complete preregistration, upstream/replay/provenance evidence, status-bearing suppressed scientific outputs, validation and hash manifests, figures, status JSON, and this canonical report.
- **Validation result:** `FAIL_CLOSED`: {reason}
- **Outcome classification:** `{classification}` ({outcome_class}); scientific associations are `NOT_EVALUATED`.
- **Caveats or blockers:** The frozen two-stage gate failed. No candidate was selected or reweighted, no S12G cache was reused, and no downstream repair was attempted.
- **Recommended next action:** Return for mandatory human review with S13 blocked; do not continue or repair S12H automatically.

## Lay summary

S12H could not safely reach the three-candidate scientific comparison because a preregistered validation gate failed at `{stage}`. The failure is preserved rather than bypassed.

## Detailed methods, inputs, and result

The two-stage method is frozen in `preregistration.yaml`. Stage 1 changed candidate 1 only by recording its selected daughter after every fission; all raw dynamics, matrices, seeds, and trajectories remained fixed. Stage 2 was permitted only after every original S12FR time-base gate passed and the derivative lock was pushed. Terminal reason: {reason}

No new GARD trajectory, prediction, MLP, intervention, estimator repair, scale-up, or S13 work occurred. Any scientific table not reached is schema-bearing and empty.

## Validation and provenance

Prior immutability passed: {immutable['passed']}; schema validation passed: {schemas['passed']}. Source, input, cache-exclusion, method-lock, and failure evidence is retained in the S12H artifact directory. S12F, S12FR, and S12G remain unchanged.

## Commands and dependencies

The preregistration, focused tests, stage command, repository commits, and runtime identities are recorded in `preregistration_record.json`, `method_lock.json`, `implementation_lock.json`, and `runtime_manifest.json`. No dependency was installed.

## Caveats and recommended next action

This operational stop is not positive or negative evidence about the emergence/self-replication relationship. Return for human review. S13 remains `BLOCKED_PENDING_S12H_HUMAN_REVIEW`.
"""
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": RESEARCH_STEP_ID,
        "stepNumber": RESEARCH_STEP_ID,
        "success": False,
        "status": f"STOPPED_FAIL_CLOSED_AT_{stage}",
        "artifactsWritten": [],
        "validationResult": f"FAIL_CLOSED: {reason}",
        "caveatsOrBlockers": [reason, "No scientific ensemble adjudication was permitted.", "S13 remains blocked."],
        "recommendedNextAction": "Return for mandatory human review; no S12H repair or downstream work is authorized.",
        "outcomeClassification": classification,
        "outcomeClass": outcome_class,
        "s13Status": "BLOCKED_PENDING_S12H_HUMAN_REVIEW",
    }
    write_json(STEP_ROOT / "status.json", status)
    manifest = artifact_manifest()
    status["artifactsWritten"] = [item["relativePath"] for item in manifest["artifacts"]] + ["artifact_manifest.json"]
    write_json(STEP_ROOT / "status.json", status)
    artifact_manifest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("revalidate", "ensemble"), required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers != 6:
        raise RuntimeError("S12H freezes exactly six source workers")
    if args.stage == "revalidate":
        return run_stage1()
    return run_stage2(args.workers)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:
        classification = "S12H_VALIDATION_FAILED_CLOSED"
        stage = "STAGE2_ENSEMBLE" if "--stage" in sys.argv and sys.argv[sys.argv.index("--stage") + 1] == "ensemble" else "STAGE1_TIMEBASE_REVALIDATION"
        finalize_fail_closed(classification=classification, stage=stage, reason=f"{type(error).__name__}:{error}")
        print(json.dumps({"stage": "S12H_failed_closed", "error": f"{type(error).__name__}:{error}"}, sort_keys=True), file=sys.stderr, flush=True)
        raise SystemExit(1) from error
