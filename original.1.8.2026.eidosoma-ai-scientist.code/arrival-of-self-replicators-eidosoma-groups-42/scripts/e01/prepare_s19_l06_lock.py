#!/usr/bin/env python3
"""Materialize and validate the pushed pre-outcome lock for E01/S19-L06."""

from __future__ import annotations

import hashlib
import json
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
from e01_creative_directional_search.core import (
    label_trajectory as frozen_label_trajectory,
)
from e01_frozen_timebase_ensemble.core import selected_clock_observations
from e01_s19_boundary_recurrence.core import (
    BOOTSTRAP_REPLICATES,
    CANDIDATE_IDS,
    COMPARATOR_LABEL_ID,
    LABEL_DEFINITIONS,
    LOOP_ID,
    PERMUTATION_REPLICATES,
    ROOT_SEED_HEX,
    STRUCTURAL_LABEL_ID,
    SUFFIX_ENDPOINT_QUANTILES,
    SUFFIX_VARIANTS,
    VERSION,
    derive_seed128,
    label_trajectory,
    recomputed_generation_block_metrics,
)
from e01_s19_iterative_replication.core import rank_candidate

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L06"
CACHE_ROOT = Path("/cache/e01_s19_l06")
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
L03_ROOT = ARTIFACT_ROOT / "loops/L03"
L05_ROOT = ARTIFACT_ROOT / "loops/L05"
PREREG = REPO_ROOT / "configs/e01/s19_l06_preregistration.yaml"
METHOD_LOCK = REPO_ROOT / "configs/e01/s19_l06_method_lock.json"
S18_BASELINE = ARTIFACT_ROOT / "s18_immutable_baseline.json"
PAPER = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")
PAPER_MARKDOWN = Path(
    "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md"
)
HISTORICAL_ROOT = Path("/cache/e01_s03/sources/gard-historical")


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


def clean_pushed_lock() -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    status = git("status", "--porcelain=v1")
    passed = branch == "eidosoma/groups/42" and head == remote and not status
    result = {
        "schema": "eidosoma.e01.s19_l06_preoutcome_repository_lock.v1",
        "loopId": LOOP_ID,
        "branch": branch,
        "head": head,
        "remoteHead": remote,
        "cleanWorktree": not bool(status),
        "headEqualsRemote": head == remote,
        "outcomeAccessed": False,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"clean pushed lock gate failed: {result}")
    return result


def manifest_rows(root: Path, manifest_path: Path, role: str) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("passed"):
        raise RuntimeError(f"prior manifest is not passing: {manifest_path}")
    rows = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            raise RuntimeError(f"missing prior artifact: {path}")
        actual = sha256_file(path)
        size = path.stat().st_size
        if actual != entry["sha256"] or size != entry["bytes"]:
            raise RuntimeError(f"changed prior artifact: {path}")
        rows.append({"path": str(path), "role": role, "bytes": size, "sha256": actual})
    rows.append(
        {
            "path": str(manifest_path),
            "role": role,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        }
    )
    return rows


def immutable_baseline_and_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    s18 = json.loads(S18_BASELINE.read_text(encoding="utf-8"))
    rows = []
    for entry in s18["files"]:
        path = Path(entry["path"])
        if not path.is_file():
            raise RuntimeError(f"missing S18 historical file: {path}")
        actual = sha256_file(path)
        size = path.stat().st_size
        if actual != entry["sha256"] or size != entry["bytes"]:
            raise RuntimeError(f"changed S18 historical file: {path}")
        rows.append(
            {"path": str(path), "role": entry["role"], "bytes": size, "sha256": actual}
        )
    for loop in ("L01", "L02", "L03", "L04", "L05"):
        root = ARTIFACT_ROOT / f"loops/{loop}"
        rows.extend(
            manifest_rows(root, root / "artifact_manifest.json", f"IMMUTABLE_S19_{loop}")
        )
    by_path = {row["path"]: row for row in rows}
    ordered = [by_path[key] for key in sorted(by_path)]
    digest = hashlib.sha256()
    for row in ordered:
        digest.update(canonical_json(row).encode())
        digest.update(b"\n")
    baseline = {
        "schema": "eidosoma.e01.s19_l06_immutable_prior_baseline.v1",
        "loopId": LOOP_ID,
        "historicalBoundary": "S01-S18_V1_V2_S19-L01_THROUGH_L05_AND_S17_WAIVER",
        "fileCount": len(ordered),
        "totalBytes": int(sum(row["bytes"] for row in ordered)),
        "aggregateSha256": digest.hexdigest(),
        "files": ordered,
    }
    validation = {
        "schema": "eidosoma.e01.s19_l06_immutable_prior_validation.v1",
        "fileCount": len(ordered),
        "aggregateSha256": digest.hexdigest(),
        "missing": [],
        "mismatches": [],
        "passed": True,
    }
    return baseline, validation


def synthetic_trajectory() -> Any:
    rng = np.random.Generator(np.random.PCG64DXSM(derive_seed128("synthetic-benchmark")))
    state = rng.integers(0, 3, size=100, dtype=np.int64)
    state[0] += int(not state.any())
    observations = [
        SimpleNamespace(
            observation_index=0,
            observation_kind="initial_selected_state",
            growth_generation_one_based=0,
            state=tuple(int(value) for value in state),
        )
    ]
    index = 0
    updates = 0
    for generation in range(1, 101):
        for _ in range(8):
            index += 1
            updates += 1
            state = state.copy()
            state[int(rng.integers(0, 100))] += 1
            observations.append(
                SimpleNamespace(
                    observation_index=index,
                    observation_kind="molecular_update",
                    growth_generation_one_based=generation,
                    state=tuple(int(value) for value in state),
                )
            )
        index += 1
        state = rng.binomial(state, 0.5).astype(np.int64)
        state[int(np.argmax(state))] += int(not state.any())
        observations.append(
            SimpleNamespace(
                observation_index=index,
                observation_kind="post_fission",
                growth_generation_one_based=generation,
                state=tuple(int(value) for value in state),
            )
        )
    return SimpleNamespace(
        observations=tuple(observations),
        total_batch_updates=updates,
        completed_fissions=100,
        configuration_id="SYNTHETIC-BENCHMARK",
        trajectory_id="SYNTHETIC-BENCHMARK",
        matrix_index=-1,
    )


def synthetic_benchmark() -> dict[str, Any]:
    trajectory = synthetic_trajectory()
    label_seconds = 0.0
    for definition in LABEL_DEFINITIONS:
        started = time.process_time()
        label_trajectory(trajectory, definition)
        label_seconds += time.process_time() - started
    selected = selected_clock_observations(trajectory, "C1_SELECTED_DAUGHTER_RETAINED")
    states = np.asarray([item.state for item in selected], dtype=np.int64)
    generations = np.asarray(
        [item.growth_generation_one_based for item in selected], dtype=np.int64
    )
    kinds = np.asarray([item.observation_kind for item in selected], dtype=str)
    benchmark_replicates = 64
    rng = np.random.Generator(np.random.PCG64DXSM(derive_seed128("permutation-benchmark")))
    orders = np.vstack([rng.permutation(100) for _ in range(benchmark_replicates)])
    started = time.process_time()
    recomputed_generation_block_metrics(states, generations, kinds, orders)
    permutation_seconds = time.process_time() - started
    projected_label_cpu = label_seconds * 200 * 2 / 3600
    projected_permutation_cpu = (
        permutation_seconds * 200 * PERMUTATION_REPLICATES / benchmark_replicates / 3600
    )
    projected_suffix_cpu = label_seconds * 200 * 5 * 4 / 3600
    projected_scientific = 3.0 * (
        projected_label_cpu + projected_permutation_cpu + projected_suffix_cpu
    )
    reserved = 3.2
    total = projected_scientific + reserved
    return {
        "schema": "eidosoma.e01.s19_l06_compute_benchmark.v1",
        "syntheticClockRows": len(selected),
        "labelCpuSeconds": label_seconds,
        "permutationBenchmarkReplicates": benchmark_replicates,
        "permutationCpuSeconds": permutation_seconds,
        "safetyMultiplier": 3.0,
        "projectedScientificCpuHours": projected_scientific,
        "reservedValidationFinalizationCpuHours": reserved,
        "projectedTotalCpuHours": total,
        "cpuCeilingHours": 32.0,
        "wallCeilingHours": 8.0,
        "gpuCeilingHours": 0.0,
        "gatePassed": total <= 32.0,
    }


def exact_preanalysis_replay() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet").sort_values(
        ["matrixIndex", "candidateId"], kind="stable"
    )
    frozen_labels = pd.read_parquet(S13Y_ROOT / "label_values.parquet")
    expected_labels = frozen_labels.loc[
        frozen_labels["labelId"].eq(COMPARATOR_LABEL_ID)
    ].sort_values(["candidateId", "matrixIndex", "selectedSequenceIndex"], kind="stable")
    l03_boundaries = pd.read_parquet(L03_ROOT / "boundary_membership_results.parquet")
    expected_boundaries = l03_boundaries.loc[
        l03_boundaries["labelId"].eq("PF_MODAL_MEDOID_BACKFILL_INCOMING_H900")
    ].sort_values(["candidateId", "matrixIndex", "boundaryIndex"], kind="stable")
    spec = fixed_label_spec(COMPARATOR_LABEL_ID)
    rows = []
    for item in manifest.itertuples(index=False):
        path = Path(item.cachePath)
        with path.open("rb") as handle:
            trajectory = pickle.load(handle)
        selected = selected_clock_observations(trajectory, str(item.clockId))
        subset = expected_labels.loc[
            expected_labels["candidateId"].eq(item.candidateId)
            & expected_labels["matrixIndex"].eq(int(item.matrixIndex))
        ].sort_values("selectedSequenceIndex", kind="stable")
        fresh, _ = frozen_label_trajectory(trajectory, spec, clock_id=str(item.clockId))
        fresh = fresh.sort_values("selectedSequenceIndex", kind="stable")
        fresh_h = fresh["labelScore"].to_numpy(dtype=np.float64)
        frozen_h = subset["labelScore"].to_numpy(dtype=np.float64)
        fresh_y = fresh["isReplicator"].to_numpy(dtype=bool)
        frozen_y = subset["isReplicator"].to_numpy(dtype=bool)
        boundary_selected = [
            (index, obs) for index, obs in enumerate(selected)
            if str(obs.observation_kind) == "post_fission"
        ]
        frozen_boundary = expected_boundaries.loc[
            expected_boundaries["candidateId"].eq(item.candidateId)
            & expected_boundaries["matrixIndex"].eq(int(item.matrixIndex))
        ].sort_values("boundaryIndex", kind="stable")
        boundary_pass = (
            len(boundary_selected) == len(frozen_boundary) == 100
            and [index for index, _ in boundary_selected]
            == [
                int(np.flatnonzero(
                    fresh["rawObservationIndex"].to_numpy(dtype=np.int64)
                    == int(raw_index)
                )[0])
                for raw_index in frozen_boundary["rawObservationIndex"]
            ]
            and [int(obs.growth_generation_one_based) for _, obs in boundary_selected]
            == frozen_boundary["boundaryGeneration"].astype(int).tolist()
            and [int(obs.observation_index) for _, obs in boundary_selected]
            == frozen_boundary["rawObservationIndex"].astype(int).tolist()
            and frozen_boundary["observationKind"].astype(str).eq("post_fission").all()
        )
        boundary_states = np.asarray([obs.state for _, obs in boundary_selected], dtype=np.int64)
        clock_pass = (
            len(selected) == len(subset)
            and np.array_equal(
                fresh["rawObservationIndex"].to_numpy(dtype=np.int64),
                subset["rawObservationIndex"].to_numpy(dtype=np.int64),
            )
            and np.array_equal(
                fresh["generation"].to_numpy(dtype=np.int64),
                subset["generation"].to_numpy(dtype=np.int64),
            )
            and fresh["observationKind"].astype(str).tolist()
            == subset["observationKind"].astype(str).tolist()
        )
        identity_pass = (
            str(trajectory.configuration_id) == str(item.candidateId)
            and str(trajectory.trajectory_id) == str(item.trajectoryId)
            and int(trajectory.matrix_index) == int(item.matrixIndex)
            and str(trajectory.trajectory_sha256) == str(item.trajectorySha256)
        )
        cache_pass = sha256_file(path) == str(item.cacheSha256)
        h_pass = np.array_equal(fresh_h, frozen_h, equal_nan=True)
        label_pass = np.array_equal(fresh_y, frozen_y)
        rows.append(
            {
                "candidateId": item.candidateId,
                "matrixIndex": int(item.matrixIndex),
                "trajectoryId": item.trajectoryId,
                "rowCount": len(fresh),
                "boundaryCount": len(boundary_selected),
                "candidateAndTrajectoryIdentityPassed": identity_pass,
                "cacheSha256Passed": cache_pass,
                "molecularClockPassed": clock_pass,
                "postFissionBoundaryIdentityPassed": boundary_pass,
                "adjacentHBitwisePassed": h_pass,
                "frozenH900LabelPassed": label_pass,
                "freshAdjacentHSha256": sha256_array(fresh_h),
                "frozenAdjacentHSha256": sha256_array(frozen_h),
                "freshLabelSha256": sha256_array(fresh_y),
                "frozenLabelSha256": sha256_array(frozen_y),
                "postFissionBoundaryStateSha256": sha256_array(boundary_states),
                "passed": bool(
                    identity_pass and cache_pass and clock_pass and boundary_pass
                    and h_pass and label_pass
                ),
            }
        )
    frame = pd.DataFrame(rows)
    cardinality = (
        len(frame) == 200
        and set(frame["candidateId"]) == set(CANDIDATE_IDS)
        and frame.groupby("candidateId")["matrixIndex"].nunique().eq(100).all()
        and frame["boundaryCount"].eq(100).all()
    )
    summary = {
        "schema": "eidosoma.e01.s19_l06_preanalysis_replay_validation.v1",
        "trajectoryCount": len(frame),
        "selectedClockRowCount": int(frame["rowCount"].sum()),
        "postFissionBoundaryCount": int(frame["boundaryCount"].sum()),
        "identityFailures": int((~frame["candidateAndTrajectoryIdentityPassed"]).sum()),
        "cacheHashFailures": int((~frame["cacheSha256Passed"]).sum()),
        "molecularClockFailures": int((~frame["molecularClockPassed"]).sum()),
        "postFissionBoundaryFailures": int((~frame["postFissionBoundaryIdentityPassed"]).sum()),
        "adjacentHFailures": int((~frame["adjacentHBitwisePassed"]).sum()),
        "frozenLabelFailures": int((~frame["frozenH900LabelPassed"]).sum()),
        "cardinalityPassed": bool(cardinality),
        "passed": bool(cardinality and frame["passed"].all()),
        "failureAction": "LOOP_FAILED_CLOSED_BEFORE_NEW_BOUNDARY_LABEL_ANALYSIS",
    }
    return frame, summary


def label_registry_payload() -> dict[str, Any]:
    rows = []
    for item in LABEL_DEFINITIONS:
        rows.append(
            {
                "ordinal": item.ordinal,
                "labelId": item.label_id,
                "role": item.role,
                "evidenceClass": item.evidence_class,
                "temporalScope": item.temporal_scope,
                "comparatorOnly": item.comparator_only,
                "promotableScope": item.promotable_scope,
                "coordinates": "L1_CLOSED_100_COMPONENT_COMPOSITION",
                "similarity": "HISTORICAL_COSINE_H_STRICTLY_GREATER_THAN_0.9",
                "boundarySubstrate": None if item.comparator_only else "SELECTED_POST_FISSION",
                "referenceRule": "ADJACENT_INCOMING" if item.comparator_only else "ZERO_LT_H_LE_G_MINUS_2",
                "projectionRule": None if item.comparator_only else "B_G_INCLUSIVE_TO_B_G_PLUS_1_EXCLUSIVE",
                "backfill": False,
                "carryAcrossNextBoundary": False,
            }
        )
    return {
        "schema": "eidosoma.e01.s19_l06_label_registry.v1",
        "loopId": LOOP_ID,
        "outcomeAccessedAtLock": False,
        "labelCount": 2,
        "structuralCandidateCount": 1,
        "labels": rows,
        "fixedComparisonLoops": ["S19-L03", "S19-L05"],
        "fixedComparisonsRecomputed": False,
        "thresholdGridPresent": False,
        "recurrenceCountSearchPresent": False,
        "variantCount": 1,
    }


def seed_manifest() -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATE_IDS:
        rows.append(
            {
                "loopId": LOOP_ID,
                "streamDomain": "S19_L06_ANALYSIS",
                "streamId": f"{LOOP_ID}::{candidate}::{STRUCTURAL_LABEL_ID}::paired_matrix_bootstrap",
                "candidateId": candidate,
                "matrixIndex": None,
                "endpointOrdinal": None,
                "purpose": "paired_matrix_bootstrap",
                "derivedSeed": str(derive_seed128(candidate, STRUCTURAL_LABEL_ID, "paired_matrix_bootstrap")),
                "rootHex": ROOT_SEED_HEX,
                "generator": "PCG64DXSM",
            }
        )
        for matrix in range(100):
            rows.append(
                {
                    "loopId": LOOP_ID,
                    "streamDomain": "S19_L06_ANALYSIS",
                    "streamId": f"{LOOP_ID}::{candidate}::M{matrix:03d}::generation_block_permutation",
                    "candidateId": candidate,
                    "matrixIndex": matrix,
                    "endpointOrdinal": None,
                    "purpose": "generation_block_permutation",
                    "derivedSeed": str(derive_seed128(candidate, matrix, "generation_block_permutation")),
                    "rootHex": ROOT_SEED_HEX,
                    "generator": "PCG64DXSM",
                }
            )
            for endpoint_ordinal in range(len(SUFFIX_ENDPOINT_QUANTILES)):
                for variant in ("SHUFFLE", "REPLACE"):
                    rows.append(
                        {
                            "loopId": LOOP_ID,
                            "streamDomain": "S19_L06_SUFFIX_AUDIT",
                            "streamId": f"{LOOP_ID}::{candidate}::M{matrix:03d}::E{endpoint_ordinal}::{variant}",
                            "candidateId": candidate,
                            "matrixIndex": matrix,
                            "endpointOrdinal": endpoint_ordinal,
                            "purpose": f"suffix_{variant.lower()}",
                            "derivedSeed": str(derive_seed128(candidate, matrix, endpoint_ordinal, "suffix", variant)),
                            "rootHex": ROOT_SEED_HEX,
                            "generator": "PCG64DXSM",
                        }
                    )
    return pd.DataFrame(rows)


def append_candidate_registry() -> pd.DataFrame:
    path = ARTIFACT_ROOT / "candidate_registry.parquet"
    existing = pd.read_parquet(path)
    ids = ["S19-L06-LABEL-01", "S19-L06-LABEL-02"]
    if existing["candidateId"].isin(ids).any():
        raise RuntimeError("L06 candidate rows already exist; append-only replay refused")
    positive_names = [
        "sourceGrounding", "paperFingerprintSpecificity", "explanatoryLeverage",
        "testability", "crossCandidateDiscriminability", "computeEfficiency",
        "independenceFromPriorOutcomeSelection",
    ]
    penalty_names = [
        "outcomeGuidedThresholdSelection", "deterministicHReuse",
        "completedFitLeakage", "candidateSpecificSuccess",
        "undefinedAuthorSemantics", "branchCount",
    ]
    values_by_ordinal = {
        1: (5, 4, 5, 5, 5, 5, 5, 0, 5, 0, 0, 0, 1),
        2: (4, 5, 5, 5, 5, 5, 4, 0, 1, 0, 0, 2, 1),
    }
    rows = []
    scored = []
    for item in LABEL_DEFINITIONS:
        values = values_by_ordinal[item.ordinal]
        positives = dict(zip(positive_names, values[:7], strict=True))
        penalties = dict(zip(penalty_names, values[7:], strict=True))
        score = rank_candidate(positives, penalties)
        scored.append((item.ordinal, score))
        rows.append(
            {
                "candidateId": ids[item.ordinal - 1],
                "bundleId": "L06_PAST_ONLY_MULTIATTRACTOR_BOUNDARY_RECURRENCE",
                "selected": True,
                **positives,
                **penalties,
                "proposedSpecification": item.label_id,
                "selectionReason": "human-directed singleton online boundary-recurrence rule plus frozen adjacent comparator",
                "rankingScore": score,
                "frozenRank": 0,
                "registryOrder": len(existing) + item.ordinal,
            }
        )
    ranks = {
        ordinal: rank + 1
        for rank, (ordinal, _) in enumerate(sorted(scored, key=lambda value: (-value[1], value[0])))
    }
    for row, item in zip(rows, LABEL_DEFINITIONS, strict=True):
        row["frozenRank"] = ranks[item.ordinal]
    additions = pd.DataFrame(rows)[existing.columns]
    pd.concat([existing, additions], ignore_index=True).to_parquet(path, index=False)
    return additions


def source_rows(retrieved: str) -> pd.DataFrame:
    sources = [
        (
            "L06_PAPER_BOUNDARY_RECURRENCE_CONTEXT", "PRIMARY_PAPER", PAPER,
            "arXiv:2607.28250v1", "CC-BY-4.0",
            "Paper describes inherited recurrence across generations and post-fission interventions, but does not identify the exact online boundary-label algorithm.",
        ),
        (
            "L06_PAPER_MARKDOWN_CONTEXT", "PRIMARY_PAPER_DERIVED_TEXT", PAPER_MARKDOWN,
            "docling-derived-from-arXiv:2607.28250v1", "DERIVED_FROM_CC-BY-4.0_INPUT",
            "Local paper text preserves Figure 1C recurrence and Table 1 temporal fingerprints.",
        ),
        (
            "L06_HISTORICAL_GARD_BOUNDARY_TRACE", "PINNED_HISTORICAL_PUBLIC_SOURCE",
            HISTORICAL_ROOT / "getcomposometime_v10.m",
            "86dff6320d5ae91b4e831471079ff46749b14df9", "NO_LICENSE_FILE_DETECTED",
            "Historical source grounds generation-boundary composition traces but not the exact L06 recurrence activation and projection.",
        ),
        (
            "L06_S13Y_FROZEN_INPUTS", "FROZEN_INTERNAL_DATASET",
            S13Y_ROOT / "trajectory_manifest.parquet",
            "E01-S13Y-CLEAN-DIRECTIONAL-CONFIRMATION-v1.0.0", "INTERNAL_GENERATED_EVIDENCE",
            "Exactly 100 shared matrix identities and 200 candidate trajectories; no new simulation is permitted.",
        ),
        (
            "L06_L03_FIXED_BOUNDARY_EVIDENCE", "FROZEN_INTERNAL_EVIDENCE",
            L03_ROOT / "artifact_manifest.json",
            "E01-S19-L03-BOUNDARY-COMPOTYPE-MOLECULAR-PROJECTION-v1.0.0", "INTERNAL_GENERATED_EVIDENCE",
            "L03 supplies frozen post-fission boundary identities and comparison evidence only.",
        ),
        (
            "L06_L05_FIXED_MOLECULAR_RECURRENCE_EVIDENCE", "FROZEN_INTERNAL_EVIDENCE",
            L05_ROOT / "artifact_manifest.json",
            "E01-S19-L05-PAST-ONLY-CROSS-GENERATION-RECURRENCE-ACTIVATION-v1.0.0", "INTERNAL_GENERATED_EVIDENCE",
            "L05 supplies fixed molecular-level past-only comparison evidence only.",
        ),
    ]
    rows = []
    for source_id, source_type, path, version, license_status, finding in sources:
        public = source_type == "PINNED_HISTORICAL_PUBLIC_SOURCE"
        rows.append(
            {
                "sourceId": source_id,
                "sourceType": source_type,
                "url": "https://github.com/ModelingOriginsofLife/GARD" if public else (
                    "https://arxiv.org/abs/2607.28250v1" if "PRIMARY_PAPER" in source_type else None
                ),
                "repositoryIdentity": "ModelingOriginsofLife/GARD" if public else None,
                "commitOrVersion": version,
                "treeIdentity": "a602fc99b494982c04c60405bc6422af9db5a77a" if public else None,
                "retrievalDate": retrieved,
                "retainedPath": str(path),
                "sha256": sha256_file(path),
                "licenseStatus": license_status,
                "evidenceClass": "DIRECT_PUBLIC_HISTORICAL_SOURCE_NOT_TARGET_PAPER_CODE" if public else (
                    "DIRECT_PAPER_EVIDENCE" if "PRIMARY_PAPER" in source_type else "FROZEN_INTERNAL_EVIDENCE"
                ),
                "finding": finding,
                "redistributionStatus": "HASH_AND_IDENTITY_ONLY" if public else "INTERNAL_OR_CITABLE_REFERENCE",
            }
        )
    return pd.DataFrame(rows)


def append_root_ledgers(timestamp: str) -> pd.DataFrame:
    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    existing_sources = pd.read_parquet(source_path)
    additions = source_rows(timestamp[:10])[existing_sources.columns]
    if existing_sources["sourceId"].isin(additions["sourceId"]).any():
        raise RuntimeError("L06 source rows already exist")
    pd.concat([existing_sources, additions], ignore_index=True).to_parquet(source_path, index=False)

    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    if ledger["loopId"].eq(LOOP_ID).any():
        raise RuntimeError("L06 self-improvement rows already exist")
    row = {
        "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
        "timestampUtc": timestamp,
        "loopId": LOOP_ID,
        "recordPhase": "PRE_LOOP_BELIEF_AND_SELECTION",
        "beliefBeforeLoop": "L03's single modal boundary compotype was too restrictive, while L05's molecular recurrence was too permissive and activated too early.",
        "motivatingEvidence": "The untested middle ground is past-only recurrence among multiple selected post-fission boundary states projected through following growth intervals.",
        "failureOrAmbiguityTargeted": "Whether generation-boundary granularity suppresses local molecular drift while retaining multiple recurring attractors and meaningful pre-onset intervals.",
        "selectedHypotheses": "Exactly one strict-H>0.9, h<=g-2, past-only post-fission-boundary recurrence rule with prospective outgoing-interval projection.",
        "learned": None,
        "weakenedHypotheses": None,
        "remainingPlausibleHypotheses": None,
        "proposedNextTest": "Pending bounded L06 execution and mandatory human review.",
        "informationGainRationale": "This changes only recurrence granularity and fixed projection; it does not retune threshold, search counts, or access emergence or downstream outcomes.",
        "appendOnly": True,
    }
    pd.concat([ledger, pd.DataFrame([row])[ledger.columns]], ignore_index=True).to_parquet(ledger_path, index=False)
    with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Entry 011 — S19-L06 pre-loop belief and selection\n\n"
            "- **Belief before the loop:** L03's single modal boundary compotype was too restrictive, while L05's all-molecular-state recurrence was too permissive and activated too early.\n"
            "- **Motivating evidence:** Multiple past boundary attractors projected through growth intervals are the untested structural middle ground.\n"
            "- **Failure or ambiguity targeted:** Whether post-fission granularity suppresses local drift and creates meaningful pre-onset intervals.\n"
            "- **Selected hypothesis:** One strict-`H>0.9`, `0<h<=g-2` post-fission-boundary recurrence rule, projected from `b_g` through the next interval.\n"
            "- **Expected information gain:** Isolates recurrence granularity without a threshold, count, alignment, emergence, or downstream search.\n"
            "- **What was learned / weakened / remains plausible:** Pending locked execution.\n"
            "- **Next test:** Pending; mandatory human review follows L06.\n"
        )
    with (ARTIFACT_ROOT / "source_search_report.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## S19-L06 additive source refresh — online boundary recurrence\n\n"
            "The paper and historical GARD lineage ground recurring compositions across generations and generation-boundary composition traces, but they do not uniquely identify L06's exact one-sided boundary activation and projection. L06 therefore executes only the human-locked singleton rule. No author was contacted, no new web search was required, and no unlicensed source was copied into artifacts.\n"
        )
    with (ARTIFACT_ROOT / "continuation_decision.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Additive human decision — authorize S19-L06 only\n\n"
            "The human selected Option 2 and authorized only `E01-S19-L06-PAST-ONLY-MULTIATTRACTOR-BOUNDARY-RECURRENCE-v1.0.0`. L06 tests one past-only post-fission-boundary recurrence label on frozen S13Y trajectories and stops for mandatory human review. L07, S20, E02, author contact, and report generation remain inactive.\n"
        )

    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if any(item["loopId"] == LOOP_ID for item in registry["loops"]):
        raise RuntimeError("L06 loop registry row already exists")
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
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["history"].append(
        {
            "date": timestamp[:10],
            "decision": "CONTINUE_S19_OPTION_2_AUTHORIZE_L06_ONLY",
            "scope": VERSION,
            "source": "explicit_human_direction",
        }
    )
    history["pendingDecision"] = "POST_S19_L06_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)
    return additions


def input_manifest() -> dict[str, Any]:
    paths = [
        S13Y_ROOT / "trajectory_manifest.parquet",
        S13Y_ROOT / "label_values.parquet",
        S13Y_ROOT / "simulation_summary.parquet",
        L03_ROOT / "artifact_manifest.json",
        L03_ROOT / "boundary_membership_results.parquet",
        L03_ROOT / "fingerprint_summary.parquet",
        L05_ROOT / "artifact_manifest.json",
        L05_ROOT / "fingerprint_summary.parquet",
        L05_ROOT / "paper_fingerprint_comparison.csv",
        Path("/artifacts/research_steps/S08/preregistration.yaml"),
        PAPER,
        PAPER_MARKDOWN,
    ]
    return {
        "schema": "eidosoma.e01.s19_l06_input_manifest.v1",
        "loopId": LOOP_ID,
        "newGardTrajectories": 0,
        "newPhiRLOrEmergenceValues": 0,
        "trajectoryCacheRoot": "/cache/e01_s13y_v1/raw_trajectories",
        "trajectoryCount": 200,
        "sharedMatrixCount": 100,
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths
        ],
    }


def specification_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "loopId": LOOP_ID,
                "specificationId": "L06-SPEC-001-COMPARATOR",
                "labelId": COMPARATOR_LABEL_ID,
                "role": "FROZEN_PRIMARY_COMPARATOR_ONLY",
                "threshold": 0.9,
                "boundarySubstrate": None,
                "referenceRule": "ADJACENT_INCOMING",
                "projectionRule": None,
                "futureReferences": False,
                "backfill": False,
                "carryAcrossNextBoundary": False,
                "selected": True,
                "variantOrdinal": 1,
            },
            {
                "loopId": LOOP_ID,
                "specificationId": "L06-SPEC-002-PRIMARY",
                "labelId": STRUCTURAL_LABEL_ID,
                "role": "PRIMARY_STRUCTURAL_EXPLORATORY",
                "threshold": 0.9,
                "boundarySubstrate": "SELECTED_POST_FISSION",
                "referenceRule": "ZERO_LT_H_LE_G_MINUS_2_UNIQUE_BOUNDARIES",
                "projectionRule": "B_G_INCLUSIVE_TO_B_G_PLUS_1_EXCLUSIVE",
                "futureReferences": False,
                "backfill": False,
                "carryAcrossNextBoundary": False,
                "selected": True,
                "variantOrdinal": 1,
            },
        ]
    )


def main() -> None:
    started = datetime.now(timezone.utc)
    if LOOP_ROOT.exists():
        raise RuntimeError("L06 artifact directory already exists; overwrite refused")
    LOOP_ROOT.mkdir(parents=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    repository = clean_pushed_lock()
    baseline, immutable = immutable_baseline_and_validation()
    benchmark = synthetic_benchmark()
    replay_rows, replay = exact_preanalysis_replay()
    gates = repository["passed"] and immutable["passed"] and benchmark["gatePassed"] and replay["passed"]

    write_json(LOOP_ROOT / "preoutcome_repository_lock.json", repository)
    write_json(LOOP_ROOT / "immutable_prior_baseline.json", baseline)
    write_json(LOOP_ROOT / "immutable_prior_validation.json", immutable)
    write_json(LOOP_ROOT / "compute_benchmark.json", benchmark)
    replay_rows.to_parquet(LOOP_ROOT / "preanalysis_replay_evidence.parquet", index=False)
    write_json(LOOP_ROOT / "preanalysis_replay_validation.json", replay)
    shutil.copy2(PREREG, LOOP_ROOT / "preregistration.yaml")
    shutil.copy2(METHOD_LOCK, LOOP_ROOT / "method_lock.json")
    registry = label_registry_payload()
    (LOOP_ROOT / "label_registry.yaml").write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    pd.DataFrame(registry["labels"]).to_parquet(LOOP_ROOT / "label_registry.parquet", index=False)
    seed_manifest().to_parquet(LOOP_ROOT / "seed_manifest.parquet", index=False)
    specification_ledger().to_parquet(LOOP_ROOT / "specification_ledger.parquet", index=False)
    write_json(LOOP_ROOT / "input_manifest.json", input_manifest())
    (LOOP_ROOT / "untouched_s20_design.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "eidosoma.e01.s19_l06_untouched_s20_design.v1",
                "loopId": LOOP_ID,
                "status": "INACTIVE_CONDITIONAL_ON_PROMOTION_AND_HUMAN_ACTIVATION",
                "scope": "RETROSPECTIVE_PAPER_FACING_BOUNDARY_LABEL_CONFIRMATION_ONLY",
                "sharedMatrixCount": 100,
                "candidateIds": list(CANDIDATE_IDS),
                "fissionsPerEligibleTrajectory": 100,
                "labelId": STRUCTURAL_LABEL_ID,
                "formula": "AT_B_G_STRICT_H_GT_0.9_TO_ANY_B_H_WITH_ZERO_LT_H_LE_G_MINUS_2",
                "projection": "B_G_INCLUSIVE_TO_B_G_PLUS_1_EXCLUSIVE",
                "futureDependence": "NONE_PAST_ONLY_NO_BACKFILL",
                "suffixAudit": list(SUFFIX_VARIANTS),
                "seedFirewall": "NEW_DOMAIN_SEPARATED_256_BIT_ROOT_ZERO_PRIOR_OVERLAP",
                "selectionInputsExcluded": ["emergence", "association", "prediction", "intervention"],
                "confirmationGate": "EXACT_REPLAY_SUFFIX_PERMUTATION_AND_SAME_LOCKED_JOINT_FINGERPRINT_GATES",
                "outcomeAccessed": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if not gates:
        pd.DataFrame(
            [
                {
                    "failureId": "L06-PREFLIGHT-001",
                    "phase": "PREANALYSIS_GATE",
                    "status": "LOOP_FAILED_CLOSED",
                    "reason": canonical_json(
                        {
                            "repository": repository["passed"],
                            "immutable": immutable["passed"],
                            "benchmark": benchmark["gatePassed"],
                            "replay": replay["passed"],
                        }
                    ),
                    "scientificOutcomesAccessed": False,
                    "repairAttempted": False,
                }
            ]
        ).to_csv(LOOP_ROOT / "failure_ledger.csv", index=False)
        raise RuntimeError("L06 failed a mandatory preanalysis gate")

    candidates = append_candidate_registry()
    candidates.to_csv(LOOP_ROOT / "candidate_ranking.csv", index=False)
    bundle = {
        "schema": "eidosoma.e01.s19_l06_candidate_bundle_registry.v1",
        "loopId": LOOP_ID,
        "bundleCount": 1,
        "bundles": [
            {
                "bundleId": "L06_PAST_ONLY_MULTIATTRACTOR_BOUNDARY_RECURRENCE",
                "selected": True,
                "structuralSpecificationCount": 1,
                "comparatorCount": 1,
                "fixedL03EvidenceCount": 1,
                "fixedL05EvidenceCount": 1,
                "labelIds": [COMPARATOR_LABEL_ID, STRUCTURAL_LABEL_ID],
                "outcomeAccessedAtSelection": False,
            }
        ],
    }
    (LOOP_ROOT / "candidate_bundle_registry.yaml").write_text(
        yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8"
    )
    sources = append_root_ledgers(started.isoformat())
    write_json(
        LOOP_ROOT / "source_snapshot_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l06_source_snapshot_manifest.v1",
            "loopId": LOOP_ID,
            "sourceCount": len(sources),
            "sources": sources.to_dict(orient="records"),
            "unlicensedSourceCopiedToArtifacts": False,
        },
    )
    write_json(
        LOOP_ROOT / "preparation_runtime.json",
        {
            "schema": "eidosoma.e01.s19_l06_preparation_runtime.v1",
            "startedUtc": started.isoformat(),
            "completedUtc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "pyarrow": pyarrow.__version__,
            "repositoryCommit": repository["head"],
            "workersPlanned": 8,
            "numericalThreadsPerWorker": 1,
            "gpuUsed": False,
            "bootstrapReplicates": BOOTSTRAP_REPLICATES,
            "permutationReplicates": PERMUTATION_REPLICATES,
            "suffixEndpointCountPerTrajectory": len(SUFFIX_ENDPOINT_QUANTILES),
            "suffixVariantCount": len(SUFFIX_VARIANTS),
        },
    )
    print(
        canonical_json(
            {
                "loopId": LOOP_ID,
                "preanalysisReplay": replay["passed"],
                "postFissionBoundaryReplay": replay["postFissionBoundaryFailures"] == 0,
                "immutablePrior": immutable["passed"],
                "immutablePriorFileCount": immutable["fileCount"],
                "benchmark": benchmark["gatePassed"],
                "projectedTotalCpuHours": benchmark["projectedTotalCpuHours"],
                "repositoryCommit": repository["head"],
                "readyForLockedExecution": True,
            }
        )
    )


if __name__ == "__main__":
    main()
