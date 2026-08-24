#!/usr/bin/env python3
"""Materialize and validate the pushed pre-outcome lock for E01/S19-L04."""

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
from e01_s19_cross_generation_recurrence.core import (
    BOOTSTRAP_REPLICATES,
    CANDIDATE_IDS,
    COMPARATOR_LABEL_ID,
    LABEL_DEFINITIONS,
    LOOP_ID,
    PERMUTATION_REPLICATES,
    ROOT_SEED_HEX,
    STRUCTURAL_LABEL_ID,
    VERSION,
    binary_consistency_from_counts,
    derive_seed128,
    label_trajectory,
)
from e01_s19_iterative_replication.core import rank_candidate

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L04"
CACHE_ROOT = Path("/cache/e01_s19_l04")
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
PREREG = REPO_ROOT / "configs/e01/s19_l04_preregistration.yaml"
METHOD_LOCK = REPO_ROOT / "configs/e01/s19_l04_method_lock.json"
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
        "schema": "eidosoma.e01.s19_l04_preoutcome_repository_lock.v1",
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
    for loop in ("L01", "L02", "L03"):
        root = ARTIFACT_ROOT / f"loops/{loop}"
        rows.extend(manifest_rows(root, root / "artifact_manifest.json", f"IMMUTABLE_S19_{loop}"))
    by_path = {row["path"]: row for row in rows}
    ordered = [by_path[key] for key in sorted(by_path)]
    digest = hashlib.sha256()
    for row in ordered:
        digest.update(canonical_json(row).encode())
        digest.update(b"\n")
    baseline = {
        "schema": "eidosoma.e01.s19_l04_immutable_prior_baseline.v1",
        "loopId": LOOP_ID,
        "historicalBoundary": "S01-S18_V1_V2_S19-L01_S19-L02_S19-L03_AND_S17_WAIVER",
        "fileCount": len(ordered),
        "totalBytes": int(sum(row["bytes"] for row in ordered)),
        "aggregateSha256": digest.hexdigest(),
        "files": ordered,
    }
    validation = {
        "schema": "eidosoma.e01.s19_l04_immutable_prior_validation.v1",
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


def permutation_microbenchmark() -> float:
    rng = np.random.Generator(np.random.PCG64DXSM(derive_seed128("perm-benchmark")))
    blocks = [rng.integers(0, 2, size=8, dtype=np.int8) for _ in range(100)]
    started = time.process_time()
    for _ in range(256):
        order = rng.permutation(100)
        first = np.asarray([blocks[index][0] for index in order], dtype=np.int8)
        last = np.asarray([blocks[index][-1] for index in order], dtype=np.int8)
        n01 = int(np.sum((last[:-1] == 0) & (first[1:] == 1)))
        n10 = int(np.sum((last[:-1] == 1) & (first[1:] == 0)))
        binary_consistency_from_counts(100, n01, n10, 500)
    return (time.process_time() - started) / 256


def synthetic_benchmark() -> dict[str, Any]:
    trajectory = synthetic_trajectory()
    timings = {}
    for definition in LABEL_DEFINITIONS:
        started = time.process_time()
        label_trajectory(trajectory, definition)
        timings[definition.label_id] = time.process_time() - started
    seconds_per_permutation_trajectory = permutation_microbenchmark()
    label_cpu = float(sum(timings.values())) * 200 * 2 / 3600
    permutation_cpu = (
        seconds_per_permutation_trajectory * PERMUTATION_REPLICATES * 200 / 3600
    )
    projected_cpu = label_cpu + permutation_cpu + 2.0
    projected_wall = projected_cpu / 8 + 0.5
    passed = projected_cpu <= 28.8 and projected_wall <= 7.2
    return {
        "schema": "eidosoma.e01.s19_l04_compute_benchmark.v1",
        "input": "deterministic_non_scientific_synthetic_901_row_100_generation_trajectory",
        "scientificOutcomeAccessed": False,
        "cpuSecondsByLabel": timings,
        "cpuSecondsPerGenerationBlockPermutationTrajectory": seconds_per_permutation_trajectory,
        "projectedTrajectoryCount": 200,
        "projectedLabelPasses": 2,
        "permutationReplicates": PERMUTATION_REPLICATES,
        "projectedScientificCpuHoursIncludingTwoHourValidationAllowance": projected_cpu,
        "projectedWallHoursIncludingFinalization": projected_wall,
        "cpuCeilingHours": 32.0,
        "scientificComputeMaximumAfterTenPercentReserveHours": 28.8,
        "gpuCeilingHours": 0.0,
        "wallCeilingHours": 8.0,
        "wallMaximumAfterTenPercentReserveHours": 7.2,
        "validationReserveFraction": 0.10,
        "gatePassed": passed,
    }


def exact_preanalysis_replay() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet").sort_values(
        ["matrixIndex", "candidateId"], kind="stable"
    )
    labels = pd.read_parquet(S13Y_ROOT / "label_values.parquet")
    expected = labels.loc[labels["labelId"].eq(COMPARATOR_LABEL_ID)].sort_values(
        ["candidateId", "matrixIndex", "selectedSequenceIndex"], kind="stable"
    )
    spec = fixed_label_spec(COMPARATOR_LABEL_ID)
    rows = []
    for item in manifest.itertuples(index=False):
        path = Path(item.cachePath)
        with path.open("rb") as handle:
            trajectory = pickle.load(handle)
        selected = selected_clock_observations(trajectory, str(item.clockId))
        subset = expected.loc[
            expected["candidateId"].eq(item.candidateId)
            & expected["matrixIndex"].eq(int(item.matrixIndex))
        ].sort_values("selectedSequenceIndex", kind="stable")
        fresh, _ = frozen_label_trajectory(trajectory, spec, clock_id=str(item.clockId))
        fresh = fresh.sort_values("selectedSequenceIndex", kind="stable")
        fresh_h = fresh["labelScore"].to_numpy(dtype=np.float64)
        frozen_h = subset["labelScore"].to_numpy(dtype=np.float64)
        fresh_y = fresh["isReplicator"].to_numpy(dtype=bool)
        frozen_y = subset["isReplicator"].to_numpy(dtype=bool)
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
                "candidateAndTrajectoryIdentityPassed": identity_pass,
                "cacheSha256Passed": cache_pass,
                "molecularClockPassed": clock_pass,
                "adjacentHBitwisePassed": h_pass,
                "frozenH900LabelPassed": label_pass,
                "freshAdjacentHSha256": sha256_array(fresh_h),
                "frozenAdjacentHSha256": sha256_array(frozen_h),
                "freshLabelSha256": sha256_array(fresh_y),
                "frozenLabelSha256": sha256_array(frozen_y),
                "passed": bool(identity_pass and cache_pass and clock_pass and h_pass and label_pass),
            }
        )
    frame = pd.DataFrame(rows)
    cardinality = (
        len(frame) == 200
        and set(frame["candidateId"]) == set(CANDIDATE_IDS)
        and frame.groupby("candidateId")["matrixIndex"].nunique().eq(100).all()
    )
    summary = {
        "schema": "eidosoma.e01.s19_l04_preanalysis_replay_validation.v1",
        "trajectoryCount": len(frame),
        "selectedClockRowCount": int(frame["rowCount"].sum()),
        "identityFailures": int((~frame["candidateAndTrajectoryIdentityPassed"]).sum()),
        "cacheHashFailures": int((~frame["cacheSha256Passed"]).sum()),
        "molecularClockFailures": int((~frame["molecularClockPassed"]).sum()),
        "adjacentHFailures": int((~frame["adjacentHBitwisePassed"]).sum()),
        "frozenLabelFailures": int((~frame["frozenH900LabelPassed"]).sum()),
        "cardinalityPassed": bool(cardinality),
        "passed": bool(cardinality and frame["passed"].all()),
        "failureAction": "LOOP_FAILED_CLOSED_BEFORE_NEW_CROSS_GENERATION_LABEL_ANALYSIS",
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
                "generationRule": (
                    "NOT_APPLICABLE"
                    if item.comparator_only
                    else "POSITIVE_DIFFERENT_GENERATION_AND_NONADJACENT_SELECTED_ROW"
                ),
                "visitCountRule": (
                    "NOT_APPLICABLE" if item.comparator_only else "UNIQUE_REFERENCE_GENERATIONS"
                ),
            }
        )
    return {
        "schema": "eidosoma.e01.s19_l04_label_registry.v1",
        "loopId": LOOP_ID,
        "outcomeAccessedAtLock": False,
        "labelCount": 2,
        "structuralCandidateCount": 1,
        "labels": rows,
        "thresholdGridPresent": False,
        "H097CandidatePresent": False,
        "variantCount": 1,
    }


def seed_manifest() -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATE_IDS:
        for purpose in ("paired_matrix_bootstrap", "generation_block_permutation"):
            rows.append(
                {
                    "loopId": LOOP_ID,
                    "streamDomain": "S19_L04_ANALYSIS",
                    "streamId": f"{LOOP_ID}::{candidate}::{STRUCTURAL_LABEL_ID}::{purpose}",
                    "candidateId": candidate,
                    "labelId": STRUCTURAL_LABEL_ID,
                    "purpose": purpose,
                    "derivedSeed": str(derive_seed128(candidate, STRUCTURAL_LABEL_ID, purpose)),
                    "rootHex": ROOT_SEED_HEX,
                    "generator": "PCG64DXSM",
                }
            )
    return pd.DataFrame(rows)


def append_candidate_registry() -> pd.DataFrame:
    path = ARTIFACT_ROOT / "candidate_registry.parquet"
    existing = pd.read_parquet(path)
    ids = ["S19-L04-LABEL-01", "S19-L04-LABEL-02"]
    if existing["candidateId"].isin(ids).any():
        raise RuntimeError("L04 candidate rows already exist; append-only replay refused")
    positives = [
        "sourceGrounding",
        "paperFingerprintSpecificity",
        "explanatoryLeverage",
        "testability",
        "crossCandidateDiscriminability",
        "computeEfficiency",
        "independenceFromPriorOutcomeSelection",
    ]
    penalties = [
        "outcomeGuidedThresholdSelection",
        "deterministicHReuse",
        "completedFitLeakage",
        "candidateSpecificSuccess",
        "undefinedAuthorSemantics",
        "branchCount",
    ]
    score_values = {
        1: (5, 4, 5, 5, 5, 5, 5, 0, 5, 0, 0, 0, 1),
        2: (4, 5, 5, 5, 5, 5, 3, 0, 1, 4, 0, 2, 1),
    }
    rows = []
    ranked = []
    for item in LABEL_DEFINITIONS:
        values = score_values[item.ordinal]
        positive = dict(zip(positives, values[:7], strict=True))
        penalty = dict(zip(penalties, values[7:], strict=True))
        score = rank_candidate(positive, penalty)
        ranked.append((item.ordinal, score))
        rows.append(
            {
                "candidateId": ids[item.ordinal - 1],
                "bundleId": "L04_CROSS_GENERATION_RECURRENCE_MEMBERSHIP",
                "selected": True,
                **positive,
                **penalty,
                "proposedSpecification": item.label_id,
                "selectionReason": "human-directed exact singleton structural label plus frozen comparator",
                "rankingScore": score,
                "frozenRank": 0,
                "registryOrder": len(existing) + item.ordinal,
            }
        )
    ranks = {
        ordinal: rank + 1
        for rank, (ordinal, _) in enumerate(sorted(ranked, key=lambda value: (-value[1], value[0])))
    }
    for row, item in zip(rows, LABEL_DEFINITIONS, strict=True):
        row["frozenRank"] = ranks[item.ordinal]
    additions = pd.DataFrame(rows)[existing.columns]
    pd.concat([existing, additions], ignore_index=True).to_parquet(path, index=False)
    return additions


def source_rows(retrieved: str) -> pd.DataFrame:
    sources = [
        (
            "L04_PAPER_ACROSS_GENERATIONS",
            "PRIMARY_PAPER",
            PAPER,
            "arXiv:2607.28250v1",
            "CC-BY-4.0",
            "Paper directly calls self-replicators recurring compositions inherited across generations and steady compositions similar from one generation to the next.",
        ),
        (
            "L04_PAPER_MARKDOWN_CONTEXT",
            "PRIMARY_PAPER_DERIVED_TEXT",
            PAPER_MARKDOWN,
            "docling-derived-from-arXiv:2607.28250v1",
            "DERIVED_FROM_CC-BY-4.0_INPUT",
            "Local equation-bearing text preserves the across-generation wording and Figure 1C cluster description.",
        ),
        (
            "L04_HISTORICAL_GARD_GENERATION_TRACE",
            "PINNED_HISTORICAL_PUBLIC_SOURCE",
            HISTORICAL_ROOT / "tgs_acluster.m",
            "86dff6320d5ae91b4e831471079ff46749b14df9",
            "NO_LICENSE_FILE_DETECTED",
            "Historical clustering consumes an NG-by-generations trace and assigns every generation a compotype tag; it supports generation-level recurrence context but not the exact L04 molecular rule.",
        ),
        (
            "L04_S13Y_FROZEN_INPUTS",
            "FROZEN_INTERNAL_DATASET",
            S13Y_ROOT / "trajectory_manifest.parquet",
            "E01-S13Y-CLEAN-DIRECTIONAL-CONFIRMATION-v1.0.0",
            "INTERNAL_GENERATED_EVIDENCE",
            "Exactly 100 shared matrix identities and 200 candidate-specific trajectories; no new simulation is permitted.",
        ),
        (
            "L04_PRIOR_FIXED_LABEL_CONTEXT",
            "FROZEN_INTERNAL_EVIDENCE",
            ARTIFACT_ROOT / "loops/L03/artifact_manifest.json",
            "E01-S19-L03-BOUNDARY-COMPOTYPE-MOLECULAR-PROJECTION-v1.0.0",
            "INTERNAL_GENERATED_EVIDENCE",
            "L02/L03 are fixed comparison evidence only; L04 does not extract a new variant from their outcomes.",
        ),
    ]
    rows = []
    for source_id, source_type, path, version, license_status, finding in sources:
        public = source_type == "PINNED_HISTORICAL_PUBLIC_SOURCE"
        rows.append(
            {
                "sourceId": source_id,
                "sourceType": source_type,
                "url": (
                    "https://github.com/ModelingOriginsofLife/GARD"
                    if public
                    else ("https://arxiv.org/abs/2607.28250v1" if "PRIMARY_PAPER" in source_type else None)
                ),
                "repositoryIdentity": "ModelingOriginsofLife/GARD" if public else None,
                "commitOrVersion": version,
                "treeIdentity": "a602fc99b494982c04c60405bc6422af9db5a77a" if public else None,
                "retrievalDate": retrieved,
                "retainedPath": str(path),
                "sha256": sha256_file(path),
                "licenseStatus": license_status,
                "evidenceClass": (
                    "DIRECT_PUBLIC_HISTORICAL_SOURCE_NOT_TARGET_PAPER_CODE"
                    if public
                    else ("DIRECT_PAPER_EVIDENCE" if "PRIMARY_PAPER" in source_type else "FROZEN_INTERNAL_EVIDENCE")
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
        raise RuntimeError("L04 source rows already exist")
    pd.concat([existing_sources, additions], ignore_index=True).to_parquet(source_path, index=False)

    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    if ledger["loopId"].eq(LOOP_ID).any():
        raise RuntimeError("L04 self-improvement rows already exist")
    row = {
        "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
        "timestampUtc": timestamp,
        "loopId": LOOP_ID,
        "recordPhase": "PRE_LOOP_BELIEF_AND_SELECTION",
        "beliefBeforeLoop": "Adjacent molecular smoothness saturates near 98 percent, while single-compotype labels are too sparse and temporally overcompact.",
        "motivatingEvidence": "The paper directly says recurring compositions are inherited across generations; permitting multiple recurring regions is a nonduplicative middle ground.",
        "failureOrAmbiguityTargeted": "Whether recurrence in any distinct growth-fission generation, rather than one dominant compotype, defines the replicator state.",
        "selectedHypotheses": "Exactly one completed-run strict-H>0.9 cross-generation nonadjacent recurrence-membership label plus adjacent-H comparator.",
        "learned": None,
        "weakenedHypotheses": None,
        "remainingPlausibleHypotheses": None,
        "proposedNextTest": "Pending bounded L04 execution and mandatory human review.",
        "informationGainRationale": "The singleton rule tests a structurally distinct across-generation hypothesis without threshold, cluster, alignment, emergence, prediction, or intervention search.",
        "appendOnly": True,
    }
    pd.concat([ledger, pd.DataFrame([row])[ledger.columns]], ignore_index=True).to_parquet(
        ledger_path, index=False
    )
    with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Entry 007 — S19-L04 pre-loop belief and selection\n\n"
            "- **Belief before the loop:** The unresolved middle ground may be recurrence in any distinct generation, not adjacent smoothness or one dominant compotype.\n"
            "- **Motivating evidence:** The paper directly describes recurring compositions inherited across generations.\n"
            "- **Failure or ambiguity targeted:** Multiple cross-generation recurring states versus a single modal attractor.\n"
            "- **Selected hypothesis:** One completed-run strict-`H>0.9` nonadjacent cross-generation membership rule; adjacent H is comparator-only.\n"
            "- **Expected information gain:** It tests one nonduplicative structural claim without adding thresholds or downstream outcome selection.\n"
            "- **What was learned / weakened / remains plausible:** Pending locked execution.\n"
            "- **Next test:** Pending; mandatory human review follows L04.\n"
        )
    with (ARTIFACT_ROOT / "source_search_report.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## S19-L04 additive source refresh — cross-generation recurrence\n\n"
            "The paper directly describes self-replicators as recurring compositions inherited across generations and as steady compositions similar from one generation to the next. Historical GARD clustering consumes one trace state per generation and assigns generation-level compotype tags. Neither source fixes the exact molecular-row rule used by the target authors. L04 therefore follows the human-directed singleton reconstruction: strict historical cosine `H>0.9`, another positive-numbered generation, nonadjacent selected-clock evidence, and distinct-generation visit counts. This is source- and paper-informed exploratory reconstruction, not author-code identity. No author was contacted, no new web search was required, and no unlicensed source was copied into artifacts.\n"
        )
    with (ARTIFACT_ROOT / "continuation_decision.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Additive human decision — authorize S19-L04 only\n\n"
            "The human selected Option 1 and authorized only `E01-S19-L04-CROSS-GENERATION-RECURRENCE-MEMBERSHIP-v1.0.0`. L04 tests one strict cross-generation recurrence label on frozen S13Y trajectories and stops for mandatory human review. L05, S20, E02, author contact, and report-bundle generation remain inactive.\n"
        )

    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if any(item["loopId"] == LOOP_ID for item in registry["loops"]):
        raise RuntimeError("L04 loop registry row already exists")
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
            "decision": "CONTINUE_S19_OPTION_1_AUTHORIZE_L04_ONLY",
            "scope": VERSION,
            "source": "explicit_human_direction",
        }
    )
    history["pendingDecision"] = "POST_S19_L04_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)
    return additions


def input_manifest() -> dict[str, Any]:
    paths = [
        S13Y_ROOT / "trajectory_manifest.parquet",
        S13Y_ROOT / "label_values.parquet",
        Path("/artifacts/research_steps/S08/preregistration.yaml"),
        Path("/artifacts/research_steps/S13X/label_registry.json"),
        ARTIFACT_ROOT / "loops/L01/artifact_manifest.json",
        ARTIFACT_ROOT / "loops/L02/artifact_manifest.json",
        ARTIFACT_ROOT / "loops/L03/artifact_manifest.json",
        PAPER,
        PAPER_MARKDOWN,
    ]
    return {
        "schema": "eidosoma.e01.s19_l04_input_manifest.v1",
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
                "specificationId": "L04-SPEC-001-COMPARATOR",
                "labelId": COMPARATOR_LABEL_ID,
                "role": "FROZEN_COMPARATOR_ONLY",
                "threshold": 0.9,
                "generationRule": "ADJACENT_INCOMING_NOT_CROSS_GENERATION",
                "immediateNeighborExcluded": False,
                "completedRunFutureDependent": False,
                "selected": True,
                "variantOrdinal": 1,
            },
            {
                "loopId": LOOP_ID,
                "specificationId": "L04-SPEC-002-PRIMARY",
                "labelId": STRUCTURAL_LABEL_ID,
                "role": "PRIMARY_STRUCTURAL_EXPLORATORY",
                "threshold": 0.9,
                "generationRule": "POSITIVE_DIFFERENT_GENERATION_UNIQUE_VISITS",
                "immediateNeighborExcluded": True,
                "completedRunFutureDependent": True,
                "selected": True,
                "variantOrdinal": 1,
            },
        ]
    )


def main() -> None:
    started = datetime.now(timezone.utc)
    if LOOP_ROOT.exists():
        raise RuntimeError("L04 artifact directory already exists; overwrite refused")
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
                "schema": "eidosoma.e01.s19_l04_untouched_s20_design.v1",
                "loopId": LOOP_ID,
                "status": "INACTIVE_CONDITIONAL_ON_PROMOTION_AND_HUMAN_ACTIVATION",
                "scope": "RETROSPECTIVE_PAPER_FACING_CONFIRMATION_ONLY",
                "sharedMatrixCount": 100,
                "candidateIds": list(CANDIDATE_IDS),
                "fissionsPerEligibleTrajectory": 100,
                "labelId": STRUCTURAL_LABEL_ID,
                "formula": "STRICT_H_GT_0.9_TO_NONADJACENT_STATE_IN_DIFFERENT_POSITIVE_GENERATION",
                "visitCount": "UNIQUE_REFERENCE_GENERATIONS",
                "generationZero": "RETAIN_INELIGIBLE",
                "futureDependence": "COMPLETED_RUN_RETROSPECTIVE",
                "seedFirewall": "NEW_DOMAIN_SEPARATED_256_BIT_ROOT_ZERO_PRIOR_OVERLAP",
                "selectionInputsExcluded": [
                    "emergence",
                    "association",
                    "prediction",
                    "intervention",
                ],
                "confirmationGate": "EXACT_REPLAY_AND_SAME_LOCKED_JOINT_FINGERPRINT_GATES",
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
                    "failureId": "L04-PREFLIGHT-001",
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
        raise RuntimeError("L04 failed a mandatory preanalysis gate")

    candidates = append_candidate_registry()
    candidates.to_csv(LOOP_ROOT / "candidate_ranking.csv", index=False)
    bundle = {
        "schema": "eidosoma.e01.s19_l04_candidate_bundle_registry.v1",
        "loopId": LOOP_ID,
        "bundleCount": 1,
        "bundles": [
            {
                "bundleId": "L04_CROSS_GENERATION_RECURRENCE_MEMBERSHIP",
                "selected": True,
                "structuralSpecificationCount": 1,
                "comparatorCount": 1,
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
            "schema": "eidosoma.e01.s19_l04_source_snapshot_manifest.v1",
            "loopId": LOOP_ID,
            "sourceCount": len(sources),
            "sources": sources.to_dict(orient="records"),
            "unlicensedSourceCopiedToArtifacts": False,
        },
    )
    write_json(
        LOOP_ROOT / "preparation_runtime.json",
        {
            "schema": "eidosoma.e01.s19_l04_preparation_runtime.v1",
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
        },
    )
    print(
        canonical_json(
            {
                "loopId": LOOP_ID,
                "preanalysisReplay": replay["passed"],
                "immutablePrior": immutable["passed"],
                "benchmark": benchmark["gatePassed"],
                "repositoryCommit": repository["head"],
                "readyForLockedExecution": True,
            }
        )
    )


if __name__ == "__main__":
    main()
