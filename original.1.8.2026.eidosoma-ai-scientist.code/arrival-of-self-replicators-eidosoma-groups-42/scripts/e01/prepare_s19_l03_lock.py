#!/usr/bin/env python3
"""Materialize and validate the pushed pre-outcome lock for E01/S19-L03."""

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
from e01_s19_boundary_compotype.core import (
    CANDIDATE_IDS,
    LABEL_DEFINITIONS,
    LOOP_ID,
    ROOT_SEED_HEX,
    VERSION,
    derive_seed128,
    label_trajectory,
)
from e01_s19_iterative_replication.core import rank_candidate

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L03"
CACHE_ROOT = Path("/cache/e01_s19_l03")
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
PREREG = REPO_ROOT / "configs/e01/s19_l03_preregistration.yaml"
METHOD_LOCK = REPO_ROOT / "configs/e01/s19_l03_method_lock.json"
S18_BASELINE = ARTIFACT_ROOT / "s18_immutable_baseline.json"
PAPER = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")
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
        "schema": "eidosoma.e01.s19_l03_preoutcome_repository_lock.v1",
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
        raise RuntimeError(f"prior manifest is not declared passing: {manifest_path}")
    rows = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            raise RuntimeError(f"missing prior artifact: {path}")
        actual = sha256_file(path)
        size = path.stat().st_size
        if actual != entry["sha256"] or size != entry["bytes"]:
            raise RuntimeError(f"changed prior artifact: {path}")
        rows.append(
            {
                "path": str(path),
                "role": role,
                "bytes": size,
                "sha256": actual,
            }
        )
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
            {
                "path": str(path),
                "role": entry["role"],
                "bytes": size,
                "sha256": actual,
            }
        )
    for loop in ("L01", "L02"):
        root = ARTIFACT_ROOT / f"loops/{loop}"
        rows.extend(manifest_rows(root, root / "artifact_manifest.json", f"IMMUTABLE_S19_{loop}"))
    by_path = {row["path"]: row for row in rows}
    ordered = [by_path[key] for key in sorted(by_path)]
    digest = hashlib.sha256()
    for row in ordered:
        digest.update(canonical_json(row).encode())
        digest.update(b"\n")
    baseline = {
        "schema": "eidosoma.e01.s19_l03_immutable_prior_baseline.v1",
        "loopId": LOOP_ID,
        "historicalBoundary": "S01-S18_V1_V2_S19-L01_S19-L02_AND_S17_WAIVER",
        "fileCount": len(ordered),
        "totalBytes": int(sum(row["bytes"] for row in ordered)),
        "aggregateSha256": digest.hexdigest(),
        "files": ordered,
    }
    validation = {
        "schema": "eidosoma.e01.s19_l03_immutable_prior_validation.v1",
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
    timings = {}
    for definition in LABEL_DEFINITIONS:
        started = time.process_time()
        label_trajectory(trajectory, definition)
        timings[definition.label_id] = time.process_time() - started
    seconds = float(sum(timings.values()))
    projected_cpu = seconds * 200 * 2 / 3600 + 2.0
    projected_wall = projected_cpu / 8 + 1.0
    passed = projected_cpu <= 90 and projected_wall <= 64.8
    return {
        "schema": "eidosoma.e01.s19_l03_compute_benchmark.v1",
        "input": "deterministic_non_scientific_synthetic_901_row_100_fission_trajectory",
        "scientificOutcomeAccessed": False,
        "cpuSecondsByLabel": timings,
        "projectedTrajectoryCount": 200,
        "projectedPasses": 2,
        "projectedScientificCpuHoursIncludingAllowance": projected_cpu,
        "projectedWallHoursIncludingFinalization": projected_wall,
        "cpuCeilingHours": 100.0,
        "gpuCeilingHours": 0.0,
        "wallCeilingHours": 72.0,
        "validationReserveFraction": 0.10,
        "gatePassed": passed,
    }


def exact_preanalysis_replay() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet").sort_values(
        ["matrixIndex", "candidateId"], kind="stable"
    )
    labels = pd.read_parquet(S13Y_ROOT / "label_values.parquet")
    expected = labels.loc[labels["labelId"].eq("MOL_ADJACENT_INCOMING_H900")].sort_values(
        ["candidateId", "matrixIndex", "selectedSequenceIndex"], kind="stable"
    )
    spec = fixed_label_spec("MOL_ADJACENT_INCOMING_H900")
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
        "schema": "eidosoma.e01.s19_l03_preanalysis_replay_validation.v1",
        "trajectoryCount": len(frame),
        "selectedClockRowCount": int(frame["rowCount"].sum()),
        "identityFailures": int((~frame["candidateAndTrajectoryIdentityPassed"]).sum()),
        "cacheHashFailures": int((~frame["cacheSha256Passed"]).sum()),
        "molecularClockFailures": int((~frame["molecularClockPassed"]).sum()),
        "adjacentHFailures": int((~frame["adjacentHBitwisePassed"]).sum()),
        "frozenLabelFailures": int((~frame["frozenH900LabelPassed"]).sum()),
        "cardinalityPassed": bool(cardinality),
        "passed": bool(cardinality and frame["passed"].all()),
        "failureAction": "LOOP_FAILED_CLOSED_BEFORE_NEW_BOUNDARY_LABEL_ANALYSIS",
    }
    return frame, summary


def label_registry_payload() -> dict[str, Any]:
    labels = []
    for item in LABEL_DEFINITIONS:
        labels.append(
            {
                "ordinal": item.ordinal,
                "labelId": item.label_id,
                "role": item.role,
                "boundarySubstrate": item.boundary_substrate,
                "activationRule": item.activation_rule,
                "projectionRule": item.projection_rule,
                "evidenceClass": item.evidence_class,
                "temporalScope": item.temporal_scope,
                "comparatorOnly": item.comparator_only,
                "promotableScope": item.promotable_scope,
                "coordinates": "L1_CLOSED_100_COMPONENT_COMPOSITION",
                "similarity": "HISTORICAL_COSINE_H_STRICTLY_GREATER_THAN_0.9",
                "referenceRule": "MAXIMUM_STRICT_H_NEIGHBOR_BOUNDARY_MEDOID",
                "tieRule": "EARLIEST_GENERATION_THEN_EARLIEST_INDEX",
                "minimumRecurrence": 2,
            }
        )
    return {
        "schema": "eidosoma.e01.s19_l03_label_registry.v1",
        "loopId": LOOP_ID,
        "outcomeAccessedAtLock": False,
        "labelCount": len(labels),
        "structuralCandidateCount": sum(not item.comparator_only for item in LABEL_DEFINITIONS),
        "labels": labels,
        "thresholdGridPresent": False,
        "H097CandidatePresent": False,
        "emergencePredictionInterventionSelectionPresent": False,
    }


def seed_manifest() -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATE_IDS:
        for definition in LABEL_DEFINITIONS:
            for purpose in ("paired_matrix_bootstrap", "projection_contrast_bootstrap"):
                rows.append(
                    {
                        "loopId": LOOP_ID,
                        "streamDomain": "S19_L03_ANALYSIS",
                        "streamId": f"{LOOP_ID}::{candidate}::{definition.label_id}::{purpose}",
                        "candidateId": candidate,
                        "labelId": definition.label_id,
                        "purpose": purpose,
                        "derivedSeed": str(derive_seed128(candidate, definition.label_id, purpose)),
                        "rootHex": ROOT_SEED_HEX,
                        "generator": "PCG64DXSM",
                    }
                )
    return pd.DataFrame(rows)


def append_candidate_registry() -> pd.DataFrame:
    path = ARTIFACT_ROOT / "candidate_registry.parquet"
    existing = pd.read_parquet(path)
    ids = [f"S19-L03-LABEL-{item.ordinal:02d}" for item in LABEL_DEFINITIONS]
    if existing["candidateId"].isin(ids).any():
        raise RuntimeError("L03 candidate rows already exist; append-only replay refused")
    rows = []
    scores = {
        1: (5, 4, 5, 5, 5, 5, 5, 0, 5, 0, 0, 0, 1),
        2: (4, 5, 5, 5, 5, 5, 4, 0, 0, 4, 0, 2, 1),
        3: (4, 5, 5, 5, 5, 5, 4, 0, 0, 4, 0, 2, 1),
        4: (4, 4, 5, 5, 5, 5, 4, 0, 0, 4, 0, 3, 1),
        5: (5, 5, 5, 5, 5, 5, 4, 0, 0, 4, 0, 2, 1),
    }
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
    ranked = []
    for item in LABEL_DEFINITIONS:
        values = scores[item.ordinal]
        positive = dict(zip(positive_names, values[:7], strict=True))
        penalties = dict(zip(penalty_names, values[7:], strict=True))
        score = rank_candidate(positive, penalties)
        ranked.append((item.ordinal, score))
        rows.append(
            {
                "candidateId": f"S19-L03-LABEL-{item.ordinal:02d}",
                "bundleId": "L03_BOUNDARY_COMPOTYPE_MOLECULAR_PROJECTION",
                "selected": True,
                **positive,
                **penalties,
                "proposedSpecification": item.label_id,
                "selectionReason": "human-directed nonfactorial structural isolation; ranking frozen before outcomes",
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
            "L03_PAPER_BOUNDARY_COMPOTYPE",
            "PRIMARY_PAPER",
            PAPER,
            "arXiv:2607.28250v1",
            "CC-BY-4.0",
            "Paper depicts recurring composition clusters and says state crosses a threshold relative to the most recurring composition, but omits boundary and projection semantics.",
        ),
        (
            "L03_HISTORICAL_TGS_NONDRIFT",
            "PINNED_HISTORICAL_PUBLIC_SOURCE",
            HISTORICAL_ROOT / "tgs_nondrift.m",
            "86dff6320d5ae91b4e831471079ff46749b14df9",
            "NO_LICENSE_FILE_DETECTED",
            "Historical technique 1 defines non-drift at generation trace states with strict H>0.9 local adjacent averaging.",
        ),
        (
            "L03_HISTORICAL_TGS_AGARD",
            "PINNED_HISTORICAL_PUBLIC_SOURCE",
            HISTORICAL_ROOT / "tgs_agard_v10.m",
            "86dff6320d5ae91b4e831471079ff46749b14df9",
            "NO_LICENSE_FILE_DETECTED",
            "Historical trace stores the critical-size generation-end composition before fission.",
        ),
        (
            "L03_HISTORICAL_GETCOMPOTIME",
            "PINNED_HISTORICAL_PUBLIC_SOURCE",
            HISTORICAL_ROOT / "getcomposometime_v10.m",
            "86dff6320d5ae91b4e831471079ff46749b14df9",
            "NO_LICENSE_FILE_DETECTED",
            "Historical analysis selects the most frequent nonzero compotype and backfills all of its tagged intervals.",
        ),
        (
            "L03_S13X_MEDOID_IMPLEMENTATION",
            "FROZEN_INTERNAL_SOURCE_IMPLEMENTATION",
            REPO_ROOT / "src/e01_creative_directional_search/core.py",
            git("rev-parse", "HEAD"),
            "PROJECT_REPOSITORY",
            "Frozen S13X supplies the already-registered maximum-neighbor medoid idea; L03 changes only boundary projection under its own lock.",
        ),
        (
            "L03_S13Y_FROZEN_INPUTS",
            "FROZEN_INTERNAL_DATASET",
            S13Y_ROOT / "trajectory_manifest.parquet",
            "E01-S13Y-CLEAN-DIRECTIONAL-CONFIRMATION-v1.0.0",
            "INTERNAL_GENERATED_EVIDENCE",
            "Exactly 100 shared matrix identities and 200 candidate-specific trajectories; no new simulation is permitted.",
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
                    else ("https://arxiv.org/abs/2607.28250v1" if source_type == "PRIMARY_PAPER" else None)
                ),
                "repositoryIdentity": (
                    "ModelingOriginsofLife/GARD"
                    if public
                    else ("Eidosoma/arrival-of-self-replicators" if "SOURCE_IMPLEMENTATION" in source_type else None)
                ),
                "commitOrVersion": version,
                "treeIdentity": "a602fc99b494982c04c60405bc6422af9db5a77a" if public else None,
                "retrievalDate": retrieved,
                "retainedPath": str(path),
                "sha256": sha256_file(path),
                "licenseStatus": license_status,
                "evidenceClass": (
                    "DIRECT_PUBLIC_HISTORICAL_SOURCE_NOT_TARGET_PAPER_CODE"
                    if public
                    else ("DIRECT_PAPER_EVIDENCE" if source_type == "PRIMARY_PAPER" else "FROZEN_INTERNAL_EVIDENCE")
                ),
                "finding": finding,
                "redistributionStatus": "HASH_AND_IDENTITY_ONLY" if public else "INTERNAL_OR_CITABLE_REFERENCE",
            }
        )
    return pd.DataFrame(rows)


def append_root_ledgers(timestamp: str, candidate_additions: pd.DataFrame) -> pd.DataFrame:
    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    source_existing = pd.read_parquet(source_path)
    additions = source_rows(timestamp[:10])[source_existing.columns]
    if source_existing["sourceId"].isin(additions["sourceId"]).any():
        raise RuntimeError("L03 source rows already exist")
    pd.concat([source_existing, additions], ignore_index=True).to_parquet(source_path, index=False)

    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    if ledger["loopId"].eq(LOOP_ID).any():
        raise RuntimeError("L03 self-improvement rows already exist")
    row = {
        "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
        "timestampUtc": timestamp,
        "loopId": LOOP_ID,
        "recordPhase": "PRE_LOOP_BELIEF_AND_SELECTION",
        "beliefBeforeLoop": "Adjacent molecular H>0.9 is saturated; L02 cluster membership and local non-drift did not jointly recover the paper fingerprint.",
        "motivatingEvidence": "The paper describes the most recurring composition, while historical GARD records and clusters generation-boundary states before projecting compotype intervals.",
        "failureOrAmbiguityTargeted": "Whether compotype identity is assigned only at fission/generation boundaries, activated on recurrence, and projected onto incoming or outgoing molecular intervals.",
        "selectedHypotheses": "One strict-H>0.9 modal boundary medoid; four nonfactorial structural candidates plus the frozen adjacent comparator.",
        "learned": None,
        "weakenedHypotheses": None,
        "remainingPlausibleHypotheses": None,
        "proposedNextTest": "Pending bounded L03 execution and mandatory human review.",
        "informationGainRationale": "The contrasts isolate backfill, alignment, and boundary substrate without threshold search or downstream outcome selection.",
        "appendOnly": True,
    }
    pd.concat([ledger, pd.DataFrame([row])[ledger.columns]], ignore_index=True).to_parquet(
        ledger_path, index=False
    )
    with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Entry 005 — S19-L03 pre-loop belief and selection\n\n"
            "- **Belief before the loop:** The unresolved label may be a boundary-defined compotype projected onto molecular time, not molecular adjacent smoothness.\n"
            "- **Motivating evidence:** The paper names the most recurring composition; historical GARD stores generation-boundary traces and selects the most frequent compotype.\n"
            "- **Failure or ambiguity targeted:** Boundary substrate, recurrence activation, and incoming/outgoing interval alignment.\n"
            "- **Selected hypotheses:** One strict-`H>0.9` modal-medoid construction and four nonfactorial structural candidates; adjacent H is comparator-only.\n"
            "- **Expected information gain:** Pairwise isolations distinguish a genuine boundary semantic from another opportunity to tune prevalence.\n"
            "- **What was learned / weakened / remains plausible:** Pending locked execution.\n"
            "- **Next test:** Pending; mandatory human review follows L03.\n"
        )
    with (ARTIFACT_ROOT / "source_search_report.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## S19-L03 additive source refresh — boundary compotype projection\n\n"
            "L02 did not fully test the activated boundary-projection ambiguity. The paper says a run enters or exits replication relative to its most recurring composition and Figure 1C depicts a recurring cluster, but gives no boundary-to-molecular projection rule. Pinned historical GARD v10 stores one critical-size pre-fission composition per generation, identifies non-drift before compotype clustering, and `getcomposometime_v10.m` selects the most frequent compotype with full-run backfill. L03 therefore freezes one maximum-neighbor boundary medoid at strict `H>0.9` and isolates backfill, second-occurrence activation, incoming/outgoing projection, and post-/pre-fission substrate without a grid. This is source-informed reconstruction, not author-code identity. No author was contacted and no unlicensed source was copied into artifacts.\n"
        )
    with (ARTIFACT_ROOT / "continuation_decision.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Additive human override — authorize S19-L03\n\n"
            "The human explicitly superseded the two-loop closeout default, imposed no fixed two- or three-loop cap, and authorized only `E01-S19-L03-BOUNDARY-COMPOTYPE-MOLECULAR-PROJECTION-v1.0.0`. Each bounded loop still stops for validation and human review. S20, E02, author contact, and report-bundle generation remain inactive.\n"
        )

    loop_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(loop_path.read_text(encoding="utf-8"))
    if any(item["loopId"] == LOOP_ID for item in registry["loops"]):
        raise RuntimeError("L03 loop registry row already exists")
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
    registry["fixedLoopCap"] = None
    registry["humanOverrideNoFixedLoopCap"] = True
    loop_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["history"].append(
        {
            "date": timestamp[:10],
            "decision": "CONTINUE_S19_NO_FIXED_LOOP_CAP_AUTHORIZE_L03_ONLY",
            "scope": VERSION,
            "source": "explicit_human_direction",
        }
    )
    history["pendingDecision"] = "POST_S19_L03_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)
    return additions


def input_manifest() -> dict[str, Any]:
    paths = [
        S13Y_ROOT / "trajectory_manifest.parquet",
        S13Y_ROOT / "label_values.parquet",
        Path("/artifacts/research_steps/S08/preregistration.yaml"),
        Path("/artifacts/research_steps/S13X/label_registry.json"),
        Path("/artifacts/research_steps/S18/matrix_a_59_claims.csv"),
        ARTIFACT_ROOT / "loops/L01/artifact_manifest.json",
        ARTIFACT_ROOT / "loops/L02/artifact_manifest.json",
        PAPER,
    ]
    return {
        "schema": "eidosoma.e01.s19_l03_input_manifest.v1",
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


def main() -> None:
    started = datetime.now(timezone.utc)
    if LOOP_ROOT.exists():
        raise RuntimeError("L03 artifact directory already exists; overwrite refused")
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
    write_json(LOOP_ROOT / "input_manifest.json", input_manifest())
    if not gates:
        pd.DataFrame(
            [
                {
                    "failureId": "L03-PREFLIGHT-001",
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
        raise RuntimeError("L03 failed a mandatory preanalysis gate")
    candidates = append_candidate_registry()
    candidates.to_csv(LOOP_ROOT / "candidate_ranking.csv", index=False)
    sources = append_root_ledgers(started.isoformat(), candidates)
    write_json(
        LOOP_ROOT / "source_snapshot_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l03_source_snapshot_manifest.v1",
            "loopId": LOOP_ID,
            "sourceCount": len(sources),
            "sources": sources.to_dict(orient="records"),
            "unlicensedSourceCopiedToArtifacts": False,
        },
    )
    write_json(
        LOOP_ROOT / "preparation_runtime.json",
        {
            "schema": "eidosoma.e01.s19_l03_preparation_runtime.v1",
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
