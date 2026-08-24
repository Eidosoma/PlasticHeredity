#!/usr/bin/env python3
"""Freeze and validate the outcome-blind E01 S13Y design."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e01_clean_directional_confirmation.core import (
    CANDIDATE_IDS,
    FULL_MODE_ID,
    IMPLEMENTATION_ID,
    PREFIX_MODE_ID,
    RESEARCH_STEP_ID,
    ROOT_SEED_HEX,
    SIMULATION_PHASE,
    VERSION,
    seed_material_sha256,
)
from e01_latent_timebase.core import derive_seed as derive_simulation_seed

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S13Y"
CONFIG = REPO / "configs/e01/s13y_clean_directional_confirmation_preregistration.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def design_state() -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    status = git("status", "--short")
    passed = branch == "eidosoma/groups/42" and head == remote and status == ""
    return {
        "branch": branch,
        "head": head,
        "remoteHead": remote,
        "workingTreeStatus": status,
        "passed": passed,
    }


def prior_artifact_baseline() -> dict[str, Any]:
    root = ARTIFACTS / "research_steps"
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or STEP_ROOT in path.parents:
            continue
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise RuntimeError("no prior research-step artifacts were found")
    return {
        "schema": "eidosoma.e01.s13y_immutable_prior_baseline.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "scope": "all files under /artifacts/research_steps excluding S13Y",
        "fileCount": len(rows),
        "totalBytes": sum(row["bytes"] for row in rows),
        "files": rows,
        "passed": True,
    }


def compute_ledger() -> dict[str, Any]:
    s13 = json.loads(
        (ARTIFACTS / "research_steps/S13/runtime_manifest.json").read_text()
    )
    r = json.loads(
        (ARTIFACTS / "research_steps/S13R/runtime_manifest.json").read_text()
    )
    rr = json.loads(
        (ARTIFACTS / "research_steps/S13RR/runtime_manifest.json").read_text()
    )
    rrr = json.loads(
        (ARTIFACTS / "research_steps/S13RRR/runtime_manifest.json").read_text()
    )
    x = json.loads(
        (ARTIFACTS / "research_steps/S13X/runtime_manifest.json").read_text()
    )
    post_s13 = {
        "S13R": float(r["processCpuSeconds"]) / 3600.0,
        "S13RR": float(rr["processCpuSeconds"]) / 3600.0,
        "S13RRR": float(rrr["processCpuSeconds"]) / 3600.0,
        "S13X": 8.0
        * sum(
            float(x[field])
            for field in (
                "stage1WallSeconds",
                "focusedNeighborhoodObservedWallSecondsApproximate",
                "paperDirectedCheckObservedWallSecondsApproximate",
                "interventionPilotWallSeconds",
                "stage1LabelSeconds",
            )
        )
        / 3600.0,
    }
    prior = float(s13["observedCumulativeE01CpuEnvelopeHours"]) + sum(post_s13.values())
    proposed = 20.0
    projected = prior + proposed
    cpu_ceiling = 250.0
    gpu_ceiling = 80.0
    projected_gpu = 2.0
    passed = projected <= cpu_ceiling and projected_gpu <= gpu_ceiling
    return {
        "schema": "eidosoma.e01.s13y_cumulative_compute_ledger.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "accountingPolicy": {
            "throughS13": "S13 observed cumulative E01 CPU envelope",
            "S13RThroughS13RRR": "recorded process CPU seconds",
            "S13X": "sum of reported stage wall fields times eight; intentionally conservative and partly overlapping",
            "S13Y": "20 CPU-hour prospective envelope including simulation, PhiRL-only full/prefix source work, statistics, and reserve",
        },
        "throughS13CpuEnvelopeHours": float(
            s13["observedCumulativeE01CpuEnvelopeHours"]
        ),
        "postS13EntriesCpuHours": post_s13,
        "priorCpuEnvelopeHours": prior,
        "proposedS13YCpuEnvelopeHoursIncludingReserve": proposed,
        "projectedCumulativeCpuHours": projected,
        "cpuCeilingHours": cpu_ceiling,
        "priorGpuEnvelopeHours": 2.0,
        "proposedS13YGpuHours": 0.0,
        "projectedCumulativeGpuHours": projected_gpu,
        "gpuCeilingHours": gpu_ceiling,
        "passed": passed,
    }


def _prior_seed_values() -> tuple[set[str], set[str], set[str]]:
    identities: set[str] = set()
    materials: set[str] = set()
    roots: set[str] = set()
    root = ARTIFACTS / "research_steps"
    hex64 = re.compile(r"^[0-9a-f]{64}$")
    for path in sorted(root.rglob("*seed*")):
        if not path.is_file() or STEP_ROOT in path.parents:
            continue
        try:
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
            elif path.suffix == ".csv":
                frame = pd.read_csv(path)
            else:
                payload = path.read_text(encoding="utf-8", errors="ignore")
                for token in re.findall(r"[0-9a-f]{64}", payload):
                    roots.add(token)
                continue
        except Exception:  # noqa: BLE001, S112 - heterogeneous historical manifests.
            continue
        for field in ("streamId", "streamIdentity", "identity"):
            if field in frame:
                identities.update(frame[field].dropna().astype(str))
        for field in ("seedMaterialSha256", "seed_material_sha256"):
            if field in frame:
                materials.update(frame[field].dropna().astype(str))
        for field in ("rootHex", "root_sha256", "rootSeedHex"):
            if field in frame:
                roots.update(
                    value
                    for value in frame[field].dropna().astype(str)
                    if hex64.match(value)
                )
    return identities, materials, roots


def _anticipated_seed_values() -> tuple[set[str], set[str]]:
    identities: set[str] = set()
    materials: set[str] = set()
    for matrix_index in range(100):
        for purpose in ("catalytic_matrix", "initial_state"):
            seed = derive_simulation_seed(
                ROOT_SEED_HEX, SIMULATION_PHASE, purpose, matrix_index
            )
            identity = f"S13Y::SIM::{purpose}::M{matrix_index:03d}::SHARED"
            identities.add(identity)
            materials.add(seed.seed_material_sha256)
        for candidate_id in CANDIDATE_IDS:
            for purpose in (
                "poisson_update",
                "overshoot_trim",
                "fission",
                "daughter_selection",
            ):
                seed = derive_simulation_seed(
                    ROOT_SEED_HEX,
                    SIMULATION_PHASE,
                    purpose,
                    matrix_index,
                    candidate_id,
                )
                identity = f"S13Y::SIM::{purpose}::M{matrix_index:03d}::{candidate_id}"
                identities.add(identity)
                materials.add(seed.seed_material_sha256)
            for mode, endpoint in [
                (FULL_MODE_ID, "FULL"),
                *[(PREFIX_MODE_ID, g) for g in range(1, 101)],
            ]:
                for purpose in ("source_preprocessing", "source_partition"):
                    identity = (
                        f"S13Y::SOURCE::{candidate_id}::M{matrix_index:03d}::"
                        f"{IMPLEMENTATION_ID}::{endpoint}::{purpose}"
                    )
                    identities.add(identity)
                    materials.add(
                        seed_material_sha256(
                            "source",
                            candidate_id,
                            matrix_index,
                            IMPLEMENTATION_ID,
                            mode,
                            endpoint,
                            purpose,
                        )
                    )
            for generation in range(1, 101):
                for purpose in (
                    "suffix_deterministic_shuffle",
                    "suffix_domain_separated_replacement",
                ):
                    identity = (
                        f"S13Y::SUFFIX::{candidate_id}::M{matrix_index:03d}::"
                        f"G{generation:03d}::{purpose}"
                    )
                    identities.add(identity)
                    materials.add(
                        seed_material_sha256(
                            "suffix", candidate_id, matrix_index, generation, purpose
                        )
                    )
    for candidate_id in CANDIDATE_IDS:
        for analysis in (
            "retrospective_H900",
            "retrospective_H970",
            "historical_comparator",
            "prefix_H900_current",
            "prefix_H900_next",
            "prefix_H970_current",
            "prefix_H970_next",
            "circularity_smooth_diagnostic",
            "paired_candidate",
        ):
            for purpose in ("bootstrap", "circular_shift"):
                identity = f"S13Y::STAT::{candidate_id}::{analysis}::{purpose}"
                identities.add(identity)
                materials.add(
                    seed_material_sha256("statistics", candidate_id, analysis, purpose)
                )
    return identities, materials


def seed_firewall() -> dict[str, Any]:
    prior_identities, prior_materials, prior_roots = _prior_seed_values()
    anticipated_identities, anticipated_materials = _anticipated_seed_values()
    identity_overlap = sorted(prior_identities & anticipated_identities)
    material_overlap = sorted(prior_materials & anticipated_materials)
    root_seen = ROOT_SEED_HEX in prior_roots
    return {
        "schema": "eidosoma.e01.s13y_seed_firewall.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "rootHex": ROOT_SEED_HEX,
        "priorIdentityCount": len(prior_identities),
        "priorMaterialCount": len(prior_materials),
        "priorCandidateRootTokenCount": len(prior_roots),
        "anticipatedIdentityCount": len(anticipated_identities),
        "anticipatedMaterialCount": len(anticipated_materials),
        "anticipatedIdentityUnique": len(anticipated_identities),
        "anticipatedMaterialUnique": len(anticipated_materials),
        "identityOverlapCount": len(identity_overlap),
        "materialOverlapCount": len(material_overlap),
        "rootPreviouslySeen": root_seen,
        "identityOverlap": identity_overlap[:20],
        "materialOverlap": material_overlap[:20],
        "passed": not identity_overlap and not material_overlap and not root_seen,
    }


def source_snapshot() -> dict[str, Any]:
    paths = [
        REPO / "src/e01_latent_timebase/core.py",
        REPO / "src/e01_source_emergence_metric_identity/core.py",
        REPO / "src/e01_pigozzi_source_equivalence_confirmation/core.py",
        REPO / "src/e01_frozen_timebase_ensemble/core.py",
        REPO / "src/e01_creative_directional_search/core.py",
        ARTIFACTS / "research_steps/S12B/safe_phi_lattice.json",
        ARTIFACTS / "research_steps/S12C/source_equivalence_results.csv",
        ARTIFACTS / "research_steps/S12D/source_metric_equivalence.csv",
        ARTIFACTS / "research_steps/S12FR/candidate_timebase_pipeline_lock.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    return {
        "schema": "eidosoma.e01.s13y_source_snapshot_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "sourceCommits": {
            "historicalGARD": "86dff6320d5ae91b4e831471079ff46749b14df9",
            "IIGR_contextOnly": "7c1c22fe39f539d4a453135476f1f0dd5a6b45f7",
            "PhiRL": "a6d1d0d18c7551302724b7158c6ccdc4d3a33373",
        },
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
            if path.is_file()
        ],
        "missing": missing,
        "passed": not missing,
    }


def fixed_branch_lock() -> dict[str, Any]:
    registry_path = ARTIFACTS / "research_steps/S13X/candidate_registry.csv"
    registry = pd.read_csv(registry_path)
    row = registry[registry["pipelineId"] == "S13X-P-684e66c4cffe914c"]
    expected = {
        "implementationId": "PHIRL_REGULARIZED_SOURCE",
        "metric": "emergence",
        "transform": "LEVEL",
        "labelId": "MOL_ADJACENT_INCOMING_H900",
        "alignment": "SAME_STATE",
    }
    matches = len(row) == 1 and all(
        str(row.iloc[0][key]) == value for key, value in expected.items()
    )
    return {
        "schema": "eidosoma.e01.s13y_fixed_branch_lock.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "s13xPipelineId": "S13X-P-684e66c4cffe914c",
        "registryPath": str(registry_path),
        "registrySha256": sha256_file(registry_path),
        "expected": expected,
        "observed": row.iloc[0].to_dict() if len(row) == 1 else None,
        "s13xCorePath": str(REPO / "src/e01_creative_directional_search/core.py"),
        "s13xCoreSha256": sha256_file(
            REPO / "src/e01_creative_directional_search/core.py"
        ),
        "passed": bool(matches),
    }


def main() -> int:
    if STEP_ROOT.exists() and any(STEP_ROOT.iterdir()):
        raise RuntimeError(f"S13Y artifact directory is not empty: {STEP_ROOT}")
    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if (
        config["versionedStepId"] != VERSION
        or config["randomness"]["rootHex"] != ROOT_SEED_HEX
    ):
        raise RuntimeError("S13Y configuration and implementation identities disagree")
    state = design_state()
    if not state["passed"]:
        raise RuntimeError(f"design commit is not pushed and clean: {state}")
    shutil.copyfile(CONFIG, STEP_ROOT / "preregistration.yaml")
    baseline = prior_artifact_baseline()
    ledger = compute_ledger()
    firewall = seed_firewall()
    source = source_snapshot()
    branch = fixed_branch_lock()
    if not all(
        (ledger["passed"], firewall["passed"], source["passed"], branch["passed"])
    ):
        raise RuntimeError("one or more S13Y pre-outcome gates failed")
    method_files = [
        CONFIG,
        REPO / "src/e01_clean_directional_confirmation/core.py",
        REPO / "scripts/e01/freeze_s13y_preregistration.py",
        REPO / "scripts/e01/run_s13y_clean_directional_confirmation.py",
        REPO / "tests/e01/test_s13y_clean_directional_confirmation.py",
    ]
    lock = {
        "schema": "eidosoma.e01.s13y_method_lock.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "lockedAtUtc": datetime.now(timezone.utc).isoformat(),
        "designState": state,
        "files": [
            {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
            for path in method_files
        ],
        "passed": True,
    }
    record = {
        "schema": "eidosoma.e01.s13y_preregistration_record.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "frozenAtUtc": datetime.now(timezone.utc).isoformat(),
        "designCommit": state["head"],
        "branch": state["branch"],
        "preregistrationSha256": sha256_file(STEP_ROOT / "preregistration.yaml"),
        "outcomeAccessed": False,
        "matrixGenerated": False,
        "passed": True,
    }
    write_json(STEP_ROOT / "immutable_prior_baseline.json", baseline)
    write_json(STEP_ROOT / "compute_ledger.json", ledger)
    write_json(
        STEP_ROOT / "compute_gate.json",
        {
            "schema": "eidosoma.e01.s13y_compute_gate.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "matrixGenerationPermitted": ledger["passed"],
            "projectedCumulativeCpuHours": ledger["projectedCumulativeCpuHours"],
            "cpuCeilingHours": ledger["cpuCeilingHours"],
            "passed": ledger["passed"],
        },
    )
    write_json(
        STEP_ROOT / "seed_root_lock.json",
        {
            "schema": "eidosoma.e01.s13y_seed_root_lock.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "rootId": "E01-S13Y-CLEAN-CONFIRMATION-ROOT-v1.0.0",
            "rootHex": ROOT_SEED_HEX,
            "phaseId": SIMULATION_PHASE,
            "passed": True,
        },
    )
    write_json(STEP_ROOT / "seed_firewall.json", firewall)
    write_json(STEP_ROOT / "source_snapshot_manifest.json", source)
    write_json(STEP_ROOT / "fixed_branch_lock.json", branch)
    write_json(STEP_ROOT / "method_lock.json", lock)
    write_json(STEP_ROOT / "preregistration_record.json", record)
    write_json(
        STEP_ROOT / "status.json",
        {
            "researchStepId": RESEARCH_STEP_ID,
            "stepNumber": "S13Y",
            "success": False,
            "status": "PREREGISTERED_BEFORE_MATRIX_GENERATION",
            "artifactsWritten": sorted(path.name for path in STEP_ROOT.iterdir()),
            "validationResult": "PASS_PRE_OUTCOME_DESIGN_COMPUTE_SOURCE_BRANCH_SEED_AND_IMMUTABILITY_BASELINE_GATES",
            "outcomeClassification": "NOT_YET_EVALUATED",
            "caveatsOrBlockers": [
                "S13X selected the fixed branch adaptively.",
                "Completed-fit values are retrospective and future-fitted.",
                "The primary binary label is deterministically defined by incoming H.",
            ],
            "recommendedNextAction": "Run only the frozen S13Y campaign, then return for mandatory human review.",
        },
    )
    print(
        json.dumps(
            {
                "stage": "s13y_preregistration_frozen",
                "designCommit": state["head"],
                "computePassed": ledger["passed"],
                "seedFirewallPassed": firewall["passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
