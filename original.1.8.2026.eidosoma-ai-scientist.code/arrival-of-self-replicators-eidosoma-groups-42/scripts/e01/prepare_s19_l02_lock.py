#!/usr/bin/env python3
"""Validate and materialize the pushed pre-outcome lock for E01/S19-L02."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_clean_directional_confirmation.core import fixed_label_spec
from e01_creative_directional_search.core import derive_seed as s13x_seed
from e01_creative_directional_search.core import label_trajectory
from e01_frozen_timebase_ensemble.core import selected_clock_observations
from e01_gard_historical import historical_nondrift_technique1
from e01_s19_iterative_replication.core import rank_candidate
from e01_s19_replicator_definition.core import (
    BOOTSTRAP_REPLICATES,
    CANDIDATE_IDS,
    LABEL_DEFINITIONS,
    LOOP_ID,
    ROOT_SEED_HEX,
    VERSION,
    derive_seed128,
)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L02"
CACHE_ROOT = Path("/cache/e01_s19_l02")
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
PREREG = REPO_ROOT / "configs/e01/s19_l02_preregistration.yaml"
METHOD_LOCK = REPO_ROOT / "configs/e01/s19_l02_method_lock.json"
S18_BASELINE = ARTIFACT_ROOT / "s18_immutable_baseline.json"
L01_ROOT = ARTIFACT_ROOT / "loops/L01"
PAPER = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(canonical_json(list(array.shape)).encode())
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def validate_clean_pushed_lock() -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    status = git("status", "--porcelain=v1")
    passed = branch == "eidosoma/groups/42" and head == remote and not status
    result = {
        "schema": "eidosoma.e01.s19_l02_preoutcome_repository_lock.v1",
        "loopId": LOOP_ID,
        "branch": branch,
        "head": head,
        "remoteHead": remote,
        "cleanWorktree": not bool(status),
        "headEqualsRemote": head == remote,
        "passed": passed,
        "outcomeAccessed": False,
    }
    if not passed:
        raise RuntimeError(f"clean pushed lock gate failed: {result}")
    return result


def validate_s18_baseline() -> dict[str, Any]:
    baseline = json.loads(S18_BASELINE.read_text(encoding="utf-8"))
    missing: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for row in baseline["files"]:
        path = Path(row["path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        actual = sha256_file(path)
        size = path.stat().st_size
        if actual != row["sha256"] or size != row["bytes"]:
            mismatches.append(
                {
                    "path": str(path),
                    "expectedSha256": row["sha256"],
                    "actualSha256": actual,
                    "expectedBytes": row["bytes"],
                    "actualBytes": size,
                }
            )
    passed = not missing and not mismatches
    return {
        "schema": "eidosoma.e01.s19_l02_s18_immutable_validation.v1",
        "storedAggregateSha256": baseline["aggregateSha256"],
        "fileCount": len(baseline["files"]),
        "missing": missing,
        "mismatches": mismatches,
        "passed": passed,
    }


def l01_baseline_and_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = L01_ROOT / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in L01_ROOT.rglob("*") if item.is_file()):
        row = {
            "path": str(path),
            "relativePath": str(path.relative_to(L01_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        files.append(row)
        aggregate.update(canonical_json(row).encode())
        aggregate.update(b"\n")
    expected = {row["path"]: row for row in manifest["files"]}
    missing = []
    mismatches = []
    for relative, row in expected.items():
        path = L01_ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            mismatches.append(relative)
    baseline = {
        "schema": "eidosoma.e01.s19_l02_immutable_prior_baseline.v1",
        "loopId": LOOP_ID,
        "historicalBoundary": "S01-S18_V1_V2_S19-L01_AND_S17_WAIVER",
        "s18BaselineAggregateSha256": json.loads(S18_BASELINE.read_text())["aggregateSha256"],
        "l01FileCount": len(files),
        "l01AggregateSha256": aggregate.hexdigest(),
        "l01Files": files,
        "l01ScientificValuesReadOrReinterpreted": False,
    }
    validation = {
        "schema": "eidosoma.e01.s19_l02_immutable_prior_validation.v1",
        "l01ManifestSha256": sha256_file(manifest_path),
        "l01ManifestDeclaredPassed": bool(manifest.get("passed")),
        "l01ExpectedFileCount": len(expected),
        "l01ObservedFileCount": len(files),
        "missing": missing,
        "mismatches": mismatches,
        "passed": bool(manifest.get("passed")) and not missing and not mismatches,
    }
    return baseline, validation


def synthetic_benchmark() -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64DXSM(derive_seed128("synthetic-benchmark")))
    observations = []
    state = rng.integers(0, 3, size=100, dtype=np.int64)
    if not state.any():
        state[0] = 1
    index = 0
    observations.append(
        SimpleNamespace(
            observation_index=index,
            observation_kind="initial_selected_state",
            growth_generation_one_based=0,
            state=tuple(int(x) for x in state),
        )
    )
    updates_per_generation = 8
    for generation in range(1, 101):
        for _ in range(updates_per_generation):
            index += 1
            add = int(rng.integers(0, 100))
            state = state.copy()
            state[add] += 1
            observations.append(
                SimpleNamespace(
                    observation_index=index,
                    observation_kind="molecular_update",
                    growth_generation_one_based=generation,
                    state=tuple(int(x) for x in state),
                )
            )
        index += 1
        keep = rng.binomial(state, 0.5).astype(np.int64)
        if not keep.any():
            keep[int(np.argmax(state))] = 1
        state = keep
        observations.append(
            SimpleNamespace(
                observation_index=index,
                observation_kind="post_fission",
                growth_generation_one_based=generation,
                state=tuple(int(x) for x in state),
            )
        )
    trajectory = SimpleNamespace(
        observations=tuple(observations),
        total_batch_updates=800,
        completed_fissions=100,
        configuration_id="SYNTHETIC-BENCHMARK",
        trajectory_id="SYNTHETIC-BENCHMARK",
        matrix_index=-1,
    )
    timings: dict[str, float] = {}
    for label_id in (
        "MOL_ADJACENT_INCOMING_H900",
        "PF_DOMINANT_COMPONENT_CENTROID_H900",
        "PF_EUCLIDEAN_KMEANS_DOMINANT",
    ):
        started = time.process_time()
        label_trajectory(trajectory, fixed_label_spec(label_id))
        timings[label_id] = time.process_time() - started
    post = np.asarray(
        [item.state for item in observations if item.observation_kind == "post_fission"],
        dtype=np.float64,
    )
    started = time.process_time()
    historical_nondrift_technique1(post.T, threshold=0.9)
    timings["PF_HISTORICAL_ADJACENT_AVERAGE_H090"] = time.process_time() - started
    seconds_per_trajectory_pass = float(sum(timings.values()))
    projected_scientific_cpu_hours = seconds_per_trajectory_pass * 200 * 2 / 3600 + 1.0
    projected_wall_hours = projected_scientific_cpu_hours / 8 + 0.5
    passed = projected_scientific_cpu_hours <= 43.2 and projected_wall_hours <= 7.2
    return {
        "schema": "eidosoma.e01.s19_l02_compute_benchmark.v1",
        "input": "deterministic_non_scientific_synthetic_901_row_100_fission_trajectory",
        "scientificOutcomeAccessed": False,
        "cpuSecondsByFamily": timings,
        "cpuSecondsPerTrajectoryPass": seconds_per_trajectory_pass,
        "projectedPasses": 2,
        "projectedTrajectoryCount": 200,
        "projectedScientificCpuHoursIncludingOneHourAnalysisAllowance": projected_scientific_cpu_hours,
        "projectedWallHoursEightWorkersIncludingHalfHourFinalization": projected_wall_hours,
        "cpuCeilingHours": 48.0,
        "wallCeilingHours": 8.0,
        "validationReserveFraction": 0.10,
        "gatePassed": passed,
    }


def exact_preanalysis_replay() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet").sort_values(
        ["matrixIndex", "candidateId"], kind="stable"
    )
    labels = pd.read_parquet(S13Y_ROOT / "label_values.parquet")
    expected = labels.loc[labels["labelId"].eq("MOL_ADJACENT_INCOMING_H900")].copy()
    expected = expected.sort_values(
        ["candidateId", "matrixIndex", "selectedSequenceIndex"], kind="stable"
    )
    rows: list[dict[str, Any]] = []
    spec = fixed_label_spec("MOL_ADJACENT_INCOMING_H900")
    for item in manifest.itertuples(index=False):
        path = Path(item.cachePath)
        cache_sha = sha256_file(path)
        with path.open("rb") as handle:
            trajectory = pickle.load(handle)
        selected = selected_clock_observations(trajectory, str(item.clockId))
        subset = expected.loc[
            expected["candidateId"].eq(item.candidateId)
            & expected["matrixIndex"].eq(int(item.matrixIndex))
        ].sort_values("selectedSequenceIndex", kind="stable")
        fresh, _ = label_trajectory(trajectory, spec, clock_id=str(item.clockId))
        fresh = fresh.sort_values("selectedSequenceIndex", kind="stable")
        fresh_h = fresh["labelScore"].to_numpy(dtype=np.float64)
        frozen_h = subset["labelScore"].to_numpy(dtype=np.float64)
        fresh_y = fresh["isReplicator"].to_numpy(dtype=bool)
        frozen_y = subset["isReplicator"].to_numpy(dtype=bool)
        fresh_raw = fresh["rawObservationIndex"].to_numpy(dtype=np.int64)
        frozen_raw = subset["rawObservationIndex"].to_numpy(dtype=np.int64)
        fresh_generation = fresh["generation"].to_numpy(dtype=np.int64)
        frozen_generation = subset["generation"].to_numpy(dtype=np.int64)
        fresh_kind = fresh["observationKind"].astype(str).tolist()
        frozen_kind = subset["observationKind"].astype(str).tolist()
        candidate_pass = str(trajectory.configuration_id) == str(item.candidateId)
        identity_pass = (
            str(trajectory.trajectory_id) == str(item.trajectoryId)
            and int(trajectory.matrix_index) == int(item.matrixIndex)
            and str(trajectory.trajectory_sha256) == str(item.trajectorySha256)
        )
        cache_pass = cache_sha == str(item.cacheSha256)
        clock_pass = (
            len(selected) == len(subset)
            and np.array_equal(fresh_raw, frozen_raw)
            and np.array_equal(fresh_generation, frozen_generation)
            and fresh_kind == frozen_kind
        )
        h_pass = np.array_equal(fresh_h, frozen_h, equal_nan=True)
        label_pass = np.array_equal(fresh_y, frozen_y)
        rows.append(
            {
                "candidateId": item.candidateId,
                "matrixIndex": int(item.matrixIndex),
                "trajectoryId": item.trajectoryId,
                "rowCount": len(fresh),
                "candidateIdentityPassed": candidate_pass,
                "trajectoryIdentityPassed": identity_pass,
                "cacheSha256Passed": cache_pass,
                "molecularClockPassed": clock_pass,
                "adjacentHBitwisePassed": h_pass,
                "frozenLabelPassed": label_pass,
                "freshAdjacentHSha256": sha256_array(fresh_h),
                "frozenAdjacentHSha256": sha256_array(frozen_h),
                "freshLabelSha256": sha256_array(fresh_y),
                "frozenLabelSha256": sha256_array(frozen_y),
                "passed": bool(candidate_pass and identity_pass and cache_pass and clock_pass and h_pass and label_pass),
            }
        )
    frame = pd.DataFrame(rows)
    cardinality = (
        len(frame) == 200
        and set(frame["candidateId"]) == set(CANDIDATE_IDS)
        and frame.groupby("candidateId")["matrixIndex"].nunique().eq(100).all()
    )
    summary = {
        "schema": "eidosoma.e01.s19_l02_preanalysis_replay_validation.v1",
        "trajectoryCount": len(frame),
        "selectedClockRowCount": int(frame["rowCount"].sum()),
        "candidateIdentityFailures": int((~frame["candidateIdentityPassed"]).sum()),
        "trajectoryIdentityFailures": int((~frame["trajectoryIdentityPassed"]).sum()),
        "cacheHashFailures": int((~frame["cacheSha256Passed"]).sum()),
        "molecularClockFailures": int((~frame["molecularClockPassed"]).sum()),
        "adjacentHFailures": int((~frame["adjacentHBitwisePassed"]).sum()),
        "frozenLabelFailures": int((~frame["frozenLabelPassed"]).sum()),
        "cardinalityPassed": bool(cardinality),
        "passed": bool(cardinality and frame["passed"].all()),
        "failureAction": "LOOP_FAILED_CLOSED_BEFORE_NEW_LABEL_ANALYSIS",
    }
    return frame, summary


def append_candidate_registry() -> pd.DataFrame:
    path = ARTIFACT_ROOT / "candidate_registry.parquet"
    existing = pd.read_parquet(path)
    ids = [f"S19-L02-LABEL-{item.ordinal:02d}" for item in LABEL_DEFINITIONS]
    if existing["candidateId"].isin(ids).any():
        raise RuntimeError("L02 candidate registry rows already exist; append-only replay refused")
    scores = {
        1: (5, 5, 5, 5, 5, 5, 5, 0, 5, 0, 0, 0, 1),
        2: (2, 4, 5, 5, 5, 5, 3, 0, 0, 0, 0, 4, 1),
        3: (2, 5, 5, 5, 5, 4, 3, 0, 0, 0, 0, 4, 1),
        4: (5, 3, 5, 5, 5, 5, 4, 0, 0, 0, 0, 2, 1),
    }
    rows = []
    computed = []
    for item in LABEL_DEFINITIONS:
        values = scores[item.ordinal]
        positive_names = [
            "sourceGrounding",
            "paperFingerprintSpecificity",
            "explanatoryLeverage",
            "testability",
            "crossCandidateDiscriminability",
            "computeEfficiency",
            "independenceFromPriorOutcomeSelection",
        ]
        penalty_names = [
            "outcomeGuidedThresholdSelection",
            "deterministicHReuse",
            "completedFitLeakage",
            "candidateSpecificSuccess",
            "undefinedAuthorSemantics",
            "branchCount",
        ]
        positive = dict(zip(positive_names, values[:7], strict=True))
        penalties = dict(zip(penalty_names, values[7:], strict=True))
        score = rank_candidate(positive, penalties)
        computed.append((item.ordinal, score))
        row = {
            "candidateId": f"S19-L02-LABEL-{item.ordinal:02d}",
            "bundleId": "L02_REPLICATOR_DEFINITION_TEMPORAL_FINGERPRINT",
            "selected": True,
            **positive,
            **penalties,
            "proposedSpecification": item.label_id,
            "selectionReason": "exact human-directed family; prior score is descriptive and did not select among the four",
            "rankingScore": score,
            "frozenRank": 0,
            "registryOrder": len(existing) + item.ordinal,
        }
        rows.append(row)
    rank = {ordinal: index + 1 for index, (ordinal, _) in enumerate(sorted(computed, key=lambda value: (-value[1], value[0])))}
    for row, item in zip(rows, LABEL_DEFINITIONS, strict=True):
        row["frozenRank"] = rank[item.ordinal]
    appended = pd.concat([existing, pd.DataFrame(rows)[existing.columns]], ignore_index=True)
    appended.to_parquet(path, index=False)
    return pd.DataFrame(rows)[existing.columns]


def append_source_ledger(retrieved: str) -> pd.DataFrame:
    path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    existing = pd.read_parquet(path)
    rows = [
        {
            "sourceId": "L02_PAPER_REPLICATOR_DEFINITION",
            "sourceType": "PRIMARY_PAPER",
            "url": "https://arxiv.org/abs/2607.28250v1",
            "repositoryIdentity": None,
            "commitOrVersion": "arXiv:2607.28250v1",
            "treeIdentity": None,
            "retrievalDate": retrieved,
            "retainedPath": str(PAPER),
            "sha256": sha256_file(PAPER),
            "licenseStatus": "CC-BY-4.0",
            "evidenceClass": "DIRECT_PAPER_EVIDENCE",
            "finding": "Self-replicators are recurring Euclidean composition clusters/attractors relative to the most recurring composition; exact clustering and threshold mechanics are omitted.",
            "redistributionStatus": "CITABLE_INPUT",
        },
        {
            "sourceId": "L02_HISTORICAL_GARD_TGS_NONDRIFT",
            "sourceType": "PINNED_HISTORICAL_PUBLIC_SOURCE",
            "url": "https://github.com/ModelingOriginsofLife/GARD",
            "repositoryIdentity": "ModelingOriginsofLife/GARD",
            "commitOrVersion": "86dff6320d5ae91b4e831471079ff46749b14df9",
            "treeIdentity": "a602fc99b494982c04c60405bc6422af9db5a77a",
            "retrievalDate": retrieved,
            "retainedPath": "/cache/e01_s03/sources/gard-historical/tgs_nondrift.m",
            "sha256": "0800359cdbef869fb545e80ab353d5aad15be2af845c90134e020f44ac860663",
            "licenseStatus": "NO_LICENSE_FILE_DETECTED",
            "evidenceClass": "DIRECT_PUBLIC_HISTORICAL_SOURCE_NOT_TARGET_PAPER_CODE",
            "finding": "Technique 1 uses the strict H>0.9 local average of adjacent post-fission similarities.",
            "redistributionStatus": "HASH_AND_IDENTITY_ONLY",
        },
        {
            "sourceId": "L02_S08_LABEL_CONTRACT",
            "sourceType": "FROZEN_INTERNAL_METHOD_ARTIFACT",
            "url": None,
            "repositoryIdentity": "Eidosoma/arrival-of-self-replicators",
            "commitOrVersion": "S08-frozen",
            "treeIdentity": None,
            "retrievalDate": retrieved,
            "retainedPath": "/artifacts/research_steps/S08/preregistration.yaml",
            "sha256": sha256_file(Path("/artifacts/research_steps/S08/preregistration.yaml")),
            "licenseStatus": "INTERNAL_RESEARCH_ARTIFACT",
            "evidenceClass": "PRIOR_FROZEN_RECONSTRUCTION_CONTRACT",
            "finding": "Fixes historical technique-1 and makes explicit that cluster thresholds/linkage are validation reconstructions, not recovered author defaults.",
            "redistributionStatus": "INTERNAL_ARTIFACT",
        },
        {
            "sourceId": "L02_S13X_LABEL_IMPLEMENTATION",
            "sourceType": "FROZEN_INTERNAL_SOURCE_IMPLEMENTATION",
            "url": "https://github.com/Eidosoma/arrival-of-self-replicators",
            "repositoryIdentity": "Eidosoma/arrival-of-self-replicators",
            "commitOrVersion": git("rev-parse", "HEAD"),
            "treeIdentity": git("rev-parse", "HEAD^{tree}"),
            "retrievalDate": retrieved,
            "retainedPath": str(REPO_ROOT / "src/e01_creative_directional_search/core.py"),
            "sha256": sha256_file(REPO_ROOT / "src/e01_creative_directional_search/core.py"),
            "licenseStatus": "PROJECT_REPOSITORY",
            "evidenceClass": "PRIOR_FROZEN_EXPLORATORY_IMPLEMENTATION",
            "finding": "Provides the exact dominant-component centroid and Euclidean silhouette/K-means implementations reused without branches.",
            "redistributionStatus": "REPOSITORY_REFERENCE",
        },
        {
            "sourceId": "L02_S13Y_FROZEN_INPUTS",
            "sourceType": "FROZEN_INTERNAL_DATASET",
            "url": None,
            "repositoryIdentity": None,
            "commitOrVersion": "E01-S13Y-CLEAN-DIRECTIONAL-CONFIRMATION-v1.0.0",
            "treeIdentity": None,
            "retrievalDate": retrieved,
            "retainedPath": str(S13Y_ROOT / "trajectory_manifest.parquet"),
            "sha256": sha256_file(S13Y_ROOT / "trajectory_manifest.parquet"),
            "licenseStatus": "INTERNAL_GENERATED_EVIDENCE",
            "evidenceClass": "FROZEN_TRAJECTORY_AND_LABEL_INPUT",
            "finding": "Exactly 100 shared matrix identities and 200 candidate-specific trajectories; no new simulation is permitted.",
            "redistributionStatus": "INTERNAL_ARTIFACT",
        },
    ]
    additions = pd.DataFrame(rows)[existing.columns]
    if existing["sourceId"].isin(additions["sourceId"]).any():
        raise RuntimeError("L02 source rows already exist; append-only replay refused")
    pd.concat([existing, additions], ignore_index=True).to_parquet(path, index=False)
    return additions


def append_self_improvement_preloop(timestamp: str) -> None:
    path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    existing = pd.read_parquet(path)
    if existing["loopId"].eq(LOOP_ID).any():
        raise RuntimeError("L02 self-improvement row already exists; append-only replay refused")
    row = {
        "ledgerSequence": int(existing["ledgerSequence"].max()) + 1,
        "timestampUtc": timestamp,
        "loopId": LOOP_ID,
        "recordPhase": "PRE_LOOP_BELIEF_AND_SELECTION",
        "beliefBeforeLoop": "The frozen adjacent H>0.9 label calls about 98% of molecular states replicating, begins almost immediately, and is exactly determined by ordinary adjacent similarity.",
        "motivatingEvidence": "The paper instead describes recurring composition clusters/attractors relative to the most recurring composition; H>0.97 moved occupancy toward 88% but did not recover onset or consistency.",
        "failureOrAmbiguityTargeted": "Determine whether a structural replicator-definition mismatch jointly explains occupancy, persistence, consistency, onset, episodes, and cutoff eligibility.",
        "selectedHypotheses": "Exactly four fixed families: adjacent-H comparator, dominant centroid, Euclidean cluster membership, and historical technique-1 non-drift.",
        "learned": None,
        "weakenedHypotheses": None,
        "remainingPlausibleHypotheses": None,
        "proposedNextTest": "Pending bounded L02 execution; stop for human review regardless of result.",
        "informationGainRationale": "A joint temporal fingerprint tests a high-leverage upstream ambiguity without emergence, new simulation, threshold tuning, or dependence on L01's failed prediction harness.",
        "appendOnly": True,
    }
    pd.concat([existing, pd.DataFrame([row])[existing.columns]], ignore_index=True).to_parquet(path, index=False)
    md = ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    with md.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Entry 003 — S19-L02 pre-loop belief and selection\n\n"
            "- **Belief before the loop:** Adjacent `H>0.9` appears to measure short-term smoothness rather than the paper's recurring attractor state.\n"
            "- **Motivating evidence:** Tightening adjacent H moved occupancy toward 88% but left onset and consistency far from the paper, so threshold adjustment is not the authorized hypothesis.\n"
            "- **Ambiguity targeted:** The replicator label is upstream of Figures 3–6 and Table 1.\n"
            "- **Selected hypotheses:** Exactly the four frozen label families in the L02 method lock; no additional branch.\n"
            "- **Expected information gain:** Joint raw/normalized onset, consistency, episodes, recurrence, and cutoff state can distinguish attractor structure from occupancy matching.\n"
            "- **What was learned / weakened / remains plausible:** Pending L02 execution.\n"
            "- **Next test:** Pending; mandatory human review follows L02.\n"
        )


def update_root_registries(timestamp: str) -> None:
    loop_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(loop_path.read_text(encoding="utf-8"))
    if any(item["loopId"] == LOOP_ID for item in registry["loops"]):
        raise RuntimeError("L02 loop registry row already exists")
    registry["loops"].append(
        {
            "loopId": LOOP_ID,
            "versionedLoopId": VERSION,
            "status": "PREANALYSIS_REPLAY_PASSED_READY_FOR_LOCKED_EXECUTION",
            "authorized": True,
            "outcomeAccessed": False,
            "humanReviewRequiredAfter": True,
            "completed": False,
            "eligibleScientificResults": False,
        }
    )
    registry["laterLoopsAuthorized"] = False
    registry["s20Status"] = "DEFINED_INACTIVE"
    registry["proposedNextLoopTheme"] = None
    registry["proposedNextLoopActive"] = False
    loop_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["history"].append(
        {
            "date": timestamp[:10],
            "decision": "CONTINUE_S19_OPTION_1_RUN_ONE_LABEL_DEFINITION_LOOP",
            "scope": VERSION,
            "source": "explicit_human_direction",
        }
    )
    history["pendingDecision"] = "POST_S19_L02_HUMAN_REVIEW"
    write_json(history_path, history)
    source_report = ARTIFACT_ROOT / "source_search_report.md"
    with source_report.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## S19-L02 additive source refresh — replicator-state definition\n\n"
            "The paper directly describes recurring steady compositions, Euclidean composition-space clusters, attractor-like homeostatic growth, entry/exit across a similarity threshold, and reference to the most recurring composition. It does not specify the clustering algorithm, recurrence/persistence minimum, threshold value, centroid/medoid rule, molecular-to-generation propagation, or ties. The pinned historical GARD v10 source directly specifies technique-1 local adjacent-H averaging at strict `H>0.9`, but it is not target-paper code. L02 therefore reuses exactly two already-frozen S13X paper-inferred cluster implementations and the frozen historical implementation; unresolved choices are not expanded into branches and fail the source-grounding promotion gate. No author was contacted and no raw unlicensed source was copied into artifacts.\n"
        )


def label_registry_payload() -> dict[str, Any]:
    return {
        "schema": "eidosoma.e01.s19_l02_label_registry.v1",
        "loopId": LOOP_ID,
        "outcomeAccessedAtLock": False,
        "labelCount": len(LABEL_DEFINITIONS),
        "labels": [
            {
                "ordinal": item.ordinal,
                "labelId": item.label_id,
                "familyName": item.family_name,
                "implementationId": item.implementation_id,
                "evidenceClass": item.evidence_class,
                "temporalScope": item.temporal_scope,
                "compositionalCoordinates": item.compositional_coordinates,
                "distanceOrSimilarity": item.distance_or_similarity,
                "recurrenceRule": item.recurrence_rule,
                "referenceRule": item.reference_rule,
                "missingDataRule": item.missing_data_rule,
                "globalReference": item.global_reference,
                "comparatorOnly": item.comparator_only,
                "sourceGroundingGate": item.source_grounding_gate,
                "unresolvedMaterialChoice": item.unresolved_material_choice,
            }
            for item in LABEL_DEFINITIONS
        ],
        "H097CandidatePresent": False,
        "thresholdGridPresent": False,
        "emergenceSelectionPresent": False,
    }


def seed_manifest() -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATE_IDS:
        for matrix in range(100):
            for k in range(2, 11):
                rows.append(
                    {
                        "loopId": LOOP_ID,
                        "streamDomain": "INHERITED_S13X_KMEANS",
                        "streamId": f"S13X::{candidate}::M{matrix:03d}::PF_EUCLIDEAN_KMEANS_DOMINANT::k{k}",
                        "candidateId": candidate,
                        "matrixIndex": matrix,
                        "labelId": "PF_EUCLIDEAN_KMEANS_DOMINANT",
                        "purpose": f"KMeans_k{k}",
                        "derivedSeed": str(s13x_seed("label", candidate, matrix, "PF_EUCLIDEAN_KMEANS_DOMINANT", k)),
                        "rootHex": "ed70c5404d73015a6742dc4a37ca15f4388dc571fce71f9129088ae23bc27c23",
                        "generator": "sklearn_legacy_RandomState",
                        "inheritedFrozenStream": True,
                    }
                )
        for label in LABEL_DEFINITIONS:
            for purpose in ("matrix_bootstrap", "cross_candidate_bootstrap"):
                rows.append(
                    {
                        "loopId": LOOP_ID,
                        "streamDomain": "S19_L02_ANALYSIS",
                        "streamId": f"{LOOP_ID}::{candidate}::{label.label_id}::{purpose}",
                        "candidateId": candidate,
                        "matrixIndex": None,
                        "labelId": label.label_id,
                        "purpose": purpose,
                        "derivedSeed": str(derive_seed128(candidate, label.label_id, purpose)),
                        "rootHex": ROOT_SEED_HEX,
                        "generator": "PCG64DXSM",
                        "inheritedFrozenStream": False,
                    }
                )
    return pd.DataFrame(rows)


def build_input_manifest() -> dict[str, Any]:
    inputs = [
        S13Y_ROOT / "trajectory_manifest.parquet",
        S13Y_ROOT / "label_values.parquet",
        S13Y_ROOT / "simulation_summary.parquet",
        Path("/artifacts/research_steps/S08/preregistration.yaml"),
        Path("/artifacts/research_steps/S13X/label_registry.json"),
        Path("/artifacts/research_steps/S13X/label_replay_validation.json"),
        Path("/artifacts/research_steps/S18/matrix_a_59_claims.csv"),
        PAPER,
    ]
    return {
        "schema": "eidosoma.e01.s19_l02_input_manifest.v1",
        "loopId": LOOP_ID,
        "newGardTrajectories": 0,
        "newPhiRLOrEmergenceValues": 0,
        "trajectoryCacheRoot": "/cache/e01_s13y_v1/raw_trajectories",
        "trajectoryCount": 200,
        "sharedMatrixCount": 100,
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in inputs
        ],
    }


def main() -> None:
    started = datetime.now(timezone.utc)
    if LOOP_ROOT.exists():
        raise RuntimeError("L02 artifact directory already exists; overwrite refused")
    LOOP_ROOT.mkdir(parents=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    repository_lock = validate_clean_pushed_lock()
    s18_validation = validate_s18_baseline()
    prior_baseline, l01_validation = l01_baseline_and_validation()
    benchmark = synthetic_benchmark()
    write_json(LOOP_ROOT / "preoutcome_repository_lock.json", repository_lock)
    write_json(LOOP_ROOT / "s18_immutable_validation.json", s18_validation)
    write_json(LOOP_ROOT / "immutable_prior_baseline.json", prior_baseline)
    write_json(LOOP_ROOT / "immutable_prior_validation.json", l01_validation)
    write_json(LOOP_ROOT / "compute_benchmark.json", benchmark)
    shutil.copy2(PREREG, LOOP_ROOT / "preregistration.yaml")
    shutil.copy2(METHOD_LOCK, LOOP_ROOT / "method_lock.json")
    (LOOP_ROOT / "label_registry.yaml").write_text(
        yaml.safe_dump(label_registry_payload(), sort_keys=False), encoding="utf-8"
    )
    label_rows = pd.DataFrame(label_registry_payload()["labels"])
    label_rows.to_parquet(LOOP_ROOT / "label_registry.parquet", index=False)
    seeds = seed_manifest()
    seeds.to_parquet(LOOP_ROOT / "seed_manifest.parquet", index=False)
    write_json(LOOP_ROOT / "input_manifest.json", build_input_manifest())
    replay_rows, replay_summary = exact_preanalysis_replay()
    replay_rows.to_parquet(LOOP_ROOT / "preanalysis_replay_evidence.parquet", index=False)
    write_json(LOOP_ROOT / "preanalysis_replay_validation.json", replay_summary)
    all_gates = bool(
        repository_lock["passed"]
        and s18_validation["passed"]
        and l01_validation["passed"]
        and benchmark["gatePassed"]
        and replay_summary["passed"]
    )
    if not all_gates:
        failure = pd.DataFrame(
            [
                {
                    "failureId": "L02-PREFLIGHT-001",
                    "phase": "PREANALYSIS_REPLAY_GATE",
                    "status": "LOOP_FAILED_CLOSED",
                    "reason": canonical_json(
                        {
                            "repository": repository_lock["passed"],
                            "s18": s18_validation["passed"],
                            "l01": l01_validation["passed"],
                            "benchmark": benchmark["gatePassed"],
                            "replay": replay_summary["passed"],
                        }
                    ),
                    "scientificOutcomesAccessed": False,
                    "repairAttempted": False,
                }
            ]
        )
        failure.to_csv(LOOP_ROOT / "failure_ledger.csv", index=False)
        raise RuntimeError("L02 failed a mandatory preanalysis gate")
    candidate_additions = append_candidate_registry()
    candidate_additions.to_csv(LOOP_ROOT / "label_candidate_prior_ranking.csv", index=False)
    source_additions = append_source_ledger(started.date().isoformat())
    write_json(
        LOOP_ROOT / "source_snapshot_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l02_source_snapshot_manifest.v1",
            "loopId": LOOP_ID,
            "sourceCount": len(source_additions),
            "sources": source_additions.to_dict(orient="records"),
            "unlicensedSourceCopiedToArtifacts": False,
        },
    )
    append_self_improvement_preloop(started.isoformat())
    update_root_registries(started.isoformat())
    pd.DataFrame(
        columns=[
            "failureId",
            "phase",
            "status",
            "candidateId",
            "matrixIndex",
            "labelId",
            "reason",
            "excludedFromPrimary",
            "scientificRepairApplied",
        ]
    ).to_csv(LOOP_ROOT / "failure_ledger.csv", index=False)
    runtime = {
        "schema": "eidosoma.e01.s19_l02_preparation_runtime.v1",
        "loopId": LOOP_ID,
        "stage": "PREOUTCOME_LOCK_AND_EXACT_FROZEN_COMPARATOR_REPLAY",
        "startedUtc": started.isoformat(),
        "completedUtc": datetime.now(timezone.utc).isoformat(),
        "repositoryCommit": repository_lock["head"],
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "workers": 8,
        "threadEnvironment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "gpuUsed": False,
        "newTrajectoryCount": 0,
        "newPhiRLOrEmergenceCount": 0,
        "preanalysisReplayPassed": True,
    }
    write_json(LOOP_ROOT / "preparation_runtime.json", runtime)
    write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "researchStepId": LOOP_ID,
            "stepNumber": 19,
            "success": False,
            "status": "PREANALYSIS_REPLAY_PASSED_READY_FOR_LOCKED_EXECUTION",
            "artifactsWritten": [str(path) for path in sorted(LOOP_ROOT.iterdir()) if path.is_file()],
            "validationResult": "PASS_IMMUTABLE_BASELINE_CLEAN_PUSHED_LOCK_BENCHMARK_AND_EXACT_ADJACENT_H_REPLAY",
            "caveatsOrBlockers": [
                "cluster_and_centroid_mechanics_not_uniquely_resolved_by_paper_or_public_source",
                "all_full_run_cluster_labels_are_retrospective",
            ],
            "recommendedNextAction": "execute_only_locked_S19_L02_then_stop_for_human_review",
        },
    )
    print(
        canonical_json(
            {
                "success": True,
                "repositoryCommit": repository_lock["head"],
                "replayedTrajectories": replay_summary["trajectoryCount"],
                "replayedRows": replay_summary["selectedClockRowCount"],
                "projectedCpuHours": benchmark["projectedScientificCpuHoursIncludingOneHourAnalysisAllowance"],
            }
        )
    )


if __name__ == "__main__":
    main()
