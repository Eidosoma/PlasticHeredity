#!/usr/bin/env python3
"""Execute locked E01/S19-L06 boundary-recurrence analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.e01.run_s19_l05 as shared
from e01_frozen_timebase_ensemble.core import selected_clock_observations
from e01_s19_boundary_recurrence.core import (
    BOOTSTRAP_REPLICATES,
    CANDIDATE_IDS,
    COMPARATOR_LABEL_ID,
    LABEL_BY_ID,
    LABEL_DEFINITIONS,
    LOOP_ID,
    PERMUTATION_REPLICATES,
    STRUCTURAL_LABEL_ID,
    SUFFIX_VARIANTS,
    VERSION,
    boundary_recurrence,
    boundary_recurrence_reference,
    derive_seed128,
    label_trajectory,
    recomputed_generation_block_metrics,
    suffix_endpoint_indices,
)
from e01_s19_replicator_definition.core import fingerprint_from_labels

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L06"
CACHE_ROOT = Path("/cache/e01_s19_l06")
LABEL_CACHE = CACHE_ROOT / "labels"
PERMUTATION_CACHE = CACHE_ROOT / "permutation_metrics"
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
L03_ROOT = ARTIFACT_ROOT / "loops/L03"
L05_ROOT = ARTIFACT_ROOT / "loops/L05"
PREREG = REPO_ROOT / "configs/e01/s19_l06_preregistration.yaml"
METHOD_LOCK = REPO_ROOT / "configs/e01/s19_l06_method_lock.json"
AMENDMENT = LOOP_ROOT / "value_preserving_amendment_001.json"

CORE_METRICS = (
    "persistence",
    "occupancy",
    "consistency",
    "firstOnsetRawScore",
    "firstOnsetNormalizedScore",
)
REPORT_METRICS = (
    *CORE_METRICS,
    "firstOnsetRawIndex0",
    "firstOnsetRawStep1",
    "firstOnsetNormalized",
    "entryCount",
    "exitCount",
    "episodeCount",
    "meanEpisodeDuration",
    "medianEpisodeDuration",
    "longestEpisode",
    "postFissionReplicatorFraction",
    "activatedBoundaryCount",
    "activatedBoundaryFraction",
    "firstActivatedBoundaryGeneration",
    "meanDistinctPriorBoundaryCount",
    "medianDistinctPriorBoundaryCount",
    "maxDistinctPriorBoundaryCount",
    "meanDistinctPriorBoundaryCountPositive",
    "preFirstBoundaryEligibleNegativeCount",
)


def configure_shared_helpers() -> None:
    """Point L05's generic matrix-level helpers at the locked L06 registry."""

    shared.CANDIDATE_IDS = CANDIDATE_IDS
    shared.COMPARATOR_LABEL_ID = COMPARATOR_LABEL_ID
    shared.STRUCTURAL_LABEL_ID = STRUCTURAL_LABEL_ID
    shared.LABEL_DEFINITIONS = LABEL_DEFINITIONS
    shared.LABEL_BY_ID = LABEL_BY_ID
    shared.LOOP_ID = LOOP_ID
    shared.LOOP_ROOT = LOOP_ROOT
    shared.CACHE_ROOT = CACHE_ROOT
    shared.S13Y_ROOT = S13Y_ROOT
    shared.BOOTSTRAP_REPLICATES = BOOTSTRAP_REPLICATES
    shared.PERMUTATION_REPLICATES = PERMUTATION_REPLICATES
    shared.CORE_METRICS = CORE_METRICS
    shared.REPORT_METRICS = REPORT_METRICS
    shared.derive_seed128 = derive_seed128


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


def sha256_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_json(orient="records", double_precision=15)
    return hashlib.sha256(payload.encode()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def nullable_labels(series: pd.Series) -> list[bool | None]:
    return [None if pd.isna(value) else bool(value) for value in series.astype(object)]


def normalize_frame(frame: pd.DataFrame, label_id: str) -> pd.DataFrame:
    definition = LABEL_BY_ID[label_id]
    output = frame.copy()
    output["researchStepId"] = LOOP_ID
    output["labelId"] = label_id
    output["labelFamily"] = definition.role
    output["labelEvidenceTier"] = definition.evidence_class
    output["temporalScope"] = definition.temporal_scope
    output["isReplicator"] = pd.array(output["isReplicator"], dtype="boolean")
    output["isPostFissionBoundary"] = pd.array(
        output["isPostFissionBoundary"], dtype="boolean"
    )
    output["isActivatedBoundary"] = pd.array(
        output["isActivatedBoundary"], dtype="boolean"
    )
    columns = [
        "researchStepId", "candidateId", "trajectoryId", "matrixIndex",
        "labelId", "labelFamily", "labelEvidenceTier", "temporalScope",
        "selectedSequenceIndex", "rawObservationIndex", "generation",
        "observationKind", "isReplicator", "labelScore", "labelStatus",
        "ineligibilityReason", "distinctPriorBoundaryCount",
        "qualifyingPriorBoundaryCount", "firstMatchingBoundaryGeneration",
        "lastMatchingBoundaryGeneration", "sourceBoundaryGeneration",
        "isPostFissionBoundary", "isActivatedBoundary",
    ]
    return output[columns]


def frame_identity(frame: pd.DataFrame) -> str:
    return sha256_frame(
        frame.sort_values("selectedSequenceIndex", kind="stable").reset_index(drop=True)
    )


def result_equal(left: dict[str, Any], right: dict[str, Any]) -> dict[str, bool]:
    fields = (
        "labels", "scores", "distinctPriorBoundaryCount",
        "qualifyingPriorBoundaryCount", "firstMatchingBoundaryGeneration",
        "lastMatchingBoundaryGeneration", "sourceBoundaryGeneration",
    )
    checks = {
        field: bool(np.array_equal(left[field], right[field], equal_nan=True))
        for field in fields
    }
    checks["matchingBoundaryGenerations"] = bool(
        left["matchingBoundaryGenerations"] == right["matchingBoundaryGenerations"]
    )
    return checks


def mutate_suffix(
    states: np.ndarray,
    endpoint: int,
    variant: str,
    candidate: str,
    matrix_index: int,
    endpoint_ordinal: int,
) -> np.ndarray:
    if variant == "DELETE":
        return states[: endpoint + 1].copy()
    changed = states.copy()
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            derive_seed128(candidate, matrix_index, endpoint_ordinal, "suffix", variant)
        )
    )
    if variant == "SHUFFLE":
        order = rng.permutation(len(states) - endpoint - 1)
        changed[endpoint + 1 :] = changed[endpoint + 1 :][order]
    elif variant == "REPLACE":
        for row_index in range(endpoint + 1, len(changed)):
            changed[row_index] = changed[row_index, rng.permutation(100)]
    else:
        raise ValueError(f"unregistered suffix variant: {variant}")
    return changed


def suffix_audit(
    states: np.ndarray,
    generations: np.ndarray,
    kinds: np.ndarray,
    indices: np.ndarray,
    candidate: str,
    matrix_index: int,
) -> list[dict[str, Any]]:
    rows = []
    for endpoint_ordinal, endpoint in enumerate(suffix_endpoint_indices(len(indices))):
        baseline = boundary_recurrence(
            states, generations, kinds, indices, query_stop=endpoint
        )
        for variant in SUFFIX_VARIANTS:
            changed_states = mutate_suffix(
                states, endpoint, variant, candidate, matrix_index, endpoint_ordinal
            )
            if variant == "DELETE":
                changed = boundary_recurrence(
                    changed_states,
                    generations[: endpoint + 1],
                    kinds[: endpoint + 1],
                    indices[: endpoint + 1],
                )
                future_after = np.empty((0, 100), dtype=np.int64)
            else:
                changed = boundary_recurrence(
                    changed_states, generations, kinds, indices, query_stop=endpoint
                )
                future_after = changed_states[endpoint + 1 :]
            future_before = states[endpoint + 1 :]
            checks = result_equal(baseline, changed)
            rows.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": matrix_index,
                    "endpointOrdinal": endpoint_ordinal,
                    "endpointSelectedIndex": endpoint,
                    "variant": variant,
                    "prefixRowCount": endpoint + 1,
                    "suffixRowCountBefore": len(states) - endpoint - 1,
                    "suffixRowCountAfter": len(changed_states) - endpoint - 1,
                    "labelsExact": checks["labels"],
                    "scoresExact": checks["scores"],
                    "distinctBoundaryCountsExact": checks["distinctPriorBoundaryCount"],
                    "qualifyingBoundaryCountsExact": checks["qualifyingPriorBoundaryCount"],
                    "firstMatchingBoundariesExact": checks["firstMatchingBoundaryGeneration"],
                    "lastMatchingBoundariesExact": checks["lastMatchingBoundaryGeneration"],
                    "sourceBoundaryGenerationExact": checks["sourceBoundaryGeneration"],
                    "allMatchingBoundaryGenerationsExact": checks["matchingBoundaryGenerations"],
                    "prefixStateHash": sha256_array(states[: endpoint + 1]),
                    "suffixBeforeHash": sha256_array(future_before),
                    "suffixAfterHash": sha256_array(future_after),
                    "suffixMutationEffective": bool(
                        variant == "DELETE" or not np.array_equal(future_before, future_after)
                    ),
                    "passed": bool(all(checks.values())),
                }
            )
    return rows


def trajectory_worker(record: dict[str, Any]) -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    with Path(record["cachePath"]).open("rb") as handle:
        trajectory = pickle.load(handle)
    selected = selected_clock_observations(trajectory, str(record["clockId"]))
    states = np.asarray([item.state for item in selected], dtype=np.int64)
    generations = np.asarray(
        [int(item.growth_generation_one_based) for item in selected], dtype=np.int64
    )
    kinds = np.asarray([str(item.observation_kind) for item in selected], dtype=str)
    indices = np.arange(len(selected), dtype=np.int64)
    frames: list[pd.DataFrame] = []
    boundary_frames: list[pd.DataFrame] = []
    diagnostics = []
    replays = []
    independent_row: dict[str, Any] | None = None
    for definition in LABEL_DEFINITIONS:
        first_raw, first_diagnostic, first_boundary = label_trajectory(
            trajectory, definition, clock_id=str(record["clockId"])
        )
        second_raw, second_diagnostic, second_boundary = label_trajectory(
            trajectory, definition, clock_id=str(record["clockId"])
        )
        first = normalize_frame(first_raw, definition.label_id)
        second = normalize_frame(second_raw, definition.label_id)
        frame_pass = frame_identity(first) == frame_identity(second)
        diagnostic_pass = canonical_json(first_diagnostic) == canonical_json(second_diagnostic)
        boundary_pass = sha256_frame(first_boundary) == sha256_frame(second_boundary)
        replays.append(
            {
                "candidateId": record["candidateId"],
                "matrixIndex": int(record["matrixIndex"]),
                "trajectoryId": record["trajectoryId"],
                "labelId": definition.label_id,
                "firstIdentity": frame_identity(first),
                "secondIdentity": frame_identity(second),
                "diagnosticEqual": diagnostic_pass,
                "boundaryEvidenceEqual": boundary_pass,
                "exactTwoPassReplayPassed": bool(frame_pass and diagnostic_pass and boundary_pass),
            }
        )
        diagnostics.append(
            {
                "candidateId": record["candidateId"],
                "matrixIndex": int(record["matrixIndex"]),
                "trajectoryId": record["trajectoryId"],
                "labelId": definition.label_id,
                "comparatorOnly": definition.comparator_only,
                **first_diagnostic,
            }
        )
        frames.append(first)
        if not first_boundary.empty:
            boundary_frames.append(first_boundary.assign(researchStepId=LOOP_ID))
        if definition.label_id == STRUCTURAL_LABEL_ID:
            primary = boundary_recurrence(states, generations, kinds, indices)
            reference = boundary_recurrence_reference(states, generations, kinds, indices)
            independent_checks = result_equal(primary, reference)
            frame_labels = first["isReplicator"].fillna(False).to_numpy(dtype=bool)
            frame_scores = pd.to_numeric(first["labelScore"], errors="coerce").to_numpy(dtype=np.float64)
            frame_counts = (
                pd.to_numeric(first["distinctPriorBoundaryCount"], errors="coerce")
                .fillna(0).to_numpy(dtype=np.int64)
            )
            boundary_materialized = first_boundary.sort_values("boundaryOrdinal", kind="stable")
            boundary_label_pass = np.array_equal(
                boundary_materialized["isActivatedBoundary"].to_numpy(dtype=bool),
                primary["boundaryLabels"],
            )
            boundary_count_pass = np.array_equal(
                boundary_materialized["distinctPriorBoundaryCount"].to_numpy(dtype=np.int64),
                primary["boundaryDistinctPriorCount"],
            )
            independent_row = {
                "candidateId": record["candidateId"],
                "matrixIndex": int(record["matrixIndex"]),
                "trajectoryId": record["trajectoryId"],
                **{f"independent_{key}": value for key, value in independent_checks.items()},
                "materializedLabelsExact": bool(np.array_equal(frame_labels, primary["labels"])),
                "materializedScoresExact": bool(np.array_equal(frame_scores, primary["scores"], equal_nan=True)),
                "materializedDistinctCountsExact": bool(
                    np.array_equal(frame_counts, primary["distinctPriorBoundaryCount"])
                ),
                "materializedBoundaryLabelsExact": boundary_label_pass,
                "materializedBoundaryCountsExact": boundary_count_pass,
                "matchingBoundaryGenerationsSha256": hashlib.sha256(
                    canonical_json(primary["matchingBoundaryGenerations"]).encode()
                ).hexdigest(),
            }
            independent_row["passed"] = bool(
                all(independent_checks.values())
                and independent_row["materializedLabelsExact"]
                and independent_row["materializedScoresExact"]
                and independent_row["materializedDistinctCountsExact"]
                and boundary_label_pass and boundary_count_pass
            )
    if independent_row is None or not boundary_frames:
        raise RuntimeError("structural boundary label was not executed")
    suffix_rows = suffix_audit(
        states, generations, kinds, indices,
        str(record["candidateId"]), int(record["matrixIndex"]),
    )
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            derive_seed128(record["candidateId"], int(record["matrixIndex"]), "generation_block_permutation")
        )
    )
    orders = np.vstack([rng.permutation(100) for _ in range(PERMUTATION_REPLICATES)]).astype(np.int16)
    permutation = recomputed_generation_block_metrics(states, generations, kinds, orders)
    permutation_path = PERMUTATION_CACHE / str(record["candidateId"]) / f"M{int(record['matrixIndex']):03d}.npz"
    permutation_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(permutation_path, **permutation)
    combined = pd.concat(frames, ignore_index=True)
    boundary_combined = pd.concat(boundary_frames, ignore_index=True)
    output = LABEL_CACHE / str(record["candidateId"]) / f"M{int(record['matrixIndex']):03d}.parquet"
    boundary_output = LABEL_CACHE / str(record["candidateId"]) / f"M{int(record['matrixIndex']):03d}.boundaries.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False, compression="zstd")
    boundary_combined.to_parquet(boundary_output, index=False, compression="zstd")
    success = bool(
        all(row["exactTwoPassReplayPassed"] for row in replays)
        and independent_row["passed"]
        and all(row["passed"] for row in suffix_rows)
        and len(boundary_combined) == 100
    )
    return {
        "candidateId": record["candidateId"],
        "matrixIndex": int(record["matrixIndex"]),
        "trajectoryId": record["trajectoryId"],
        "labelCache": str(output),
        "boundaryCache": str(boundary_output),
        "permutationCache": str(permutation_path),
        "diagnostics": diagnostics,
        "replays": replays,
        "independentReplay": independent_row,
        "suffixAudit": suffix_rows,
        "success": success,
        "wallSeconds": time.perf_counter() - started_wall,
        "cpuSeconds": time.process_time() - started_cpu,
    }


def execution_lock_validation() -> dict[str, Any]:
    repository = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text(encoding="utf-8"))
    replay = json.loads((LOOP_ROOT / "preanalysis_replay_validation.json").read_text(encoding="utf-8"))
    immutable = json.loads((LOOP_ROOT / "immutable_prior_validation.json").read_text(encoding="utf-8"))
    benchmark = json.loads((LOOP_ROOT / "compute_benchmark.json").read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    clean = not bool(git("status", "--porcelain=v1"))
    hashes = {
        "repositoryPreregistration": sha256_file(PREREG),
        "artifactPreregistration": sha256_file(LOOP_ROOT / "preregistration.yaml"),
        "repositoryMethodLock": sha256_file(METHOD_LOCK),
        "artifactMethodLock": sha256_file(LOOP_ROOT / "method_lock.json"),
    }
    passed = bool(
        repository["passed"] and replay["passed"] and immutable["passed"]
        and amendment["passed"] and amendment["scientificOutcomesAccessed"] is False
        and benchmark["gatePassed"] and head == remote == repository["head"] and clean
        and hashes["repositoryPreregistration"] == hashes["artifactPreregistration"]
        and hashes["repositoryMethodLock"] == hashes["artifactMethodLock"]
    )
    return {
        "schema": "eidosoma.e01.s19_l06_execution_lock_validation.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "preparedHead": repository["head"],
        "cleanWorktree": clean,
        "configHashes": hashes,
        "valuePreservingAmendmentId": amendment["amendmentId"],
        "valuePreservingAmendmentPassed": amendment["passed"],
        "passed": passed,
    }


def execute_trajectories(
    manifest: pd.DataFrame, workers: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = manifest.sort_values(["matrixIndex", "candidateId"], kind="stable").to_dict(orient="records")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(trajectory_worker, row): row for row in records}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (row["matrixIndex"], row["candidateId"]))
    execution = pd.DataFrame(
        [
            {
                "candidateId": row["candidateId"], "matrixIndex": row["matrixIndex"],
                "trajectoryId": row["trajectoryId"], "success": row["success"],
                "wallSeconds": row["wallSeconds"], "cpuSeconds": row["cpuSeconds"],
                "labelCache": row["labelCache"], "boundaryCache": row["boundaryCache"],
                "permutationCache": row["permutationCache"],
            }
            for row in results
        ]
    )
    labels = pd.concat([pd.read_parquet(row["labelCache"]) for row in results], ignore_index=True)
    labels["isReplicator"] = pd.array(labels["isReplicator"], dtype="boolean")
    labels["isPostFissionBoundary"] = pd.array(labels["isPostFissionBoundary"], dtype="boolean")
    labels["isActivatedBoundary"] = pd.array(labels["isActivatedBoundary"], dtype="boolean")
    boundaries = pd.concat([pd.read_parquet(row["boundaryCache"]) for row in results], ignore_index=True)
    diagnostics = pd.DataFrame([item for row in results for item in row["diagnostics"]])
    replay = pd.DataFrame([item for row in results for item in row["replays"]])
    independent = pd.DataFrame([row["independentReplay"] for row in results])
    suffix = pd.DataFrame([item for row in results for item in row["suffixAudit"]])
    return labels, boundaries, diagnostics, replay, independent, suffix, execution


def build_fingerprints(labels: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = labels.sort_values(
        ["candidateId", "matrixIndex", "labelId", "selectedSequenceIndex"], kind="stable"
    ).groupby(["candidateId", "matrixIndex", "trajectoryId", "labelId"], sort=False)
    for (candidate, matrix, trajectory, label_id), group in grouped:
        definition = LABEL_BY_ID[label_id]
        fingerprint = fingerprint_from_labels(
            sequence_indices=group["selectedSequenceIndex"].tolist(),
            labels=nullable_labels(group["isReplicator"]),
            total_clock_count=len(group),
            observation_kinds=group["observationKind"].tolist(),
            global_reference=False,
        )
        diagnostic = diagnostics.loc[
            diagnostics["candidateId"].eq(candidate)
            & diagnostics["matrixIndex"].eq(int(matrix))
            & diagnostics["labelId"].eq(label_id)
        ].iloc[0]
        rows.append(
            {
                "candidateId": candidate,
                "matrixIndex": int(matrix),
                "trajectoryId": trajectory,
                "labelId": label_id,
                "labelOrdinal": definition.ordinal,
                "labelRole": definition.role,
                "evidenceClass": definition.evidence_class,
                "temporalScope": definition.temporal_scope,
                "activatedBoundaryCount": diagnostic.get("activatedBoundaryCount"),
                "activatedBoundaryFraction": diagnostic.get("activatedBoundaryFraction"),
                "firstActivatedBoundaryGeneration": diagnostic.get("firstActivatedBoundaryGeneration"),
                "meanDistinctPriorBoundaryCount": diagnostic.get("meanDistinctPriorBoundaryCount"),
                "medianDistinctPriorBoundaryCount": diagnostic.get("medianDistinctPriorBoundaryCount"),
                "maxDistinctPriorBoundaryCount": diagnostic.get("maxDistinctPriorBoundaryCount"),
                "meanDistinctPriorBoundaryCountPositive": diagnostic.get("meanDistinctPriorBoundaryCountPositive"),
                "preFirstBoundaryEligibleNegativeCount": diagnostic.get("preFirstBoundaryEligibleNegativeCount"),
                **fingerprint,
            }
        )
    return pd.DataFrame(rows).sort_values(["labelOrdinal", "candidateId", "matrixIndex"], kind="stable")


def fixed_prior_comparison(aggregate: pd.DataFrame) -> pd.DataFrame:
    l03 = pd.read_parquet(L03_ROOT / "fingerprint_summary.parquet")
    l05 = pd.read_parquet(L05_ROOT / "fingerprint_summary.parquet")
    references = [
        ("S19-L03", l03, "PF_MODAL_MEDOID_ACTIVATED_OUTGOING_H900", "RETROSPECTIVE_MODAL_BOUNDARY_OUTGOING"),
        ("S19-L05", l05, "MOL_PAST_ONLY_CROSS_GENERATION_RECURRENCE_H900", "PAST_ONLY_ALL_MOLECULAR_STATES"),
    ]
    rows = []
    for candidate in CANDIDATE_IDS:
        current = aggregate.loc[
            aggregate["candidateId"].eq(candidate) & aggregate["labelId"].eq(STRUCTURAL_LABEL_ID)
        ].iloc[0]
        for loop_id, frame, label_id, scope in references:
            prior = frame.loc[
                frame["candidateId"].eq(candidate) & frame["labelId"].eq(label_id)
            ].iloc[0]
            for metric in CORE_METRICS:
                token = metric[0].upper() + metric[1:]
                left = float(current[f"mean{token}"])
                right = float(prior[f"mean{token}"])
                rows.append(
                    {
                        "candidateId": candidate,
                        "l06LabelId": STRUCTURAL_LABEL_ID,
                        "fixedPriorLoopId": loop_id,
                        "fixedPriorLabelId": label_id,
                        "fixedPriorTemporalScope": scope,
                        "metric": metric,
                        "l06Mean": left,
                        "fixedPriorMean": right,
                        "l06MinusFixedPrior": left - right,
                    }
                )
    return pd.DataFrame(rows)


def generation_block_permutation(
    execution: pd.DataFrame, comparison: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_rows = []
    summary_rows = []
    for candidate in CANDIDATE_IDS:
        current = execution.loc[execution["candidateId"].eq(candidate)].sort_values("matrixIndex", kind="stable")
        if len(current) != 100 or not np.array_equal(current["matrixIndex"].to_numpy(), np.arange(100)):
            raise RuntimeError("permutation cache matrix identity mismatch")
        loaded = []
        for path in current["permutationCache"]:
            with np.load(path, allow_pickle=False) as archive:
                loaded.append({metric: archive[metric].copy() for metric in CORE_METRICS})
        arrays = {metric: np.vstack([item[metric] for item in loaded]) for metric in CORE_METRICS}
        means = {metric: np.nanmean(values, axis=0) for metric, values in arrays.items()}
        for mode in ("RAW", "NORMALIZED"):
            distances = shared.distance_arrays(means, mode)
            for replicate, distance in enumerate(distances):
                full_rows.append(
                    {
                        "candidateId": candidate, "labelId": STRUCTURAL_LABEL_ID,
                        "replicate": replicate, "onsetMode": mode,
                        "meanPersistence": means["persistence"][replicate],
                        "meanOccupancy": means["occupancy"][replicate],
                        "meanConsistency": means["consistency"][replicate],
                        "meanFirstOnsetRawScore": means["firstOnsetRawScore"][replicate],
                        "meanFirstOnsetNormalizedScore": means["firstOnsetNormalizedScore"][replicate],
                        "paperDistance": distance,
                    }
                )
            observed = float(
                comparison.loc[
                    comparison["candidateId"].eq(candidate)
                    & comparison["labelId"].eq(STRUCTURAL_LABEL_ID)
                    & comparison["onsetMode"].eq(mode), "paperDistance"
                ].iloc[0]
            )
            lower = float(np.quantile(distances, 0.025))
            summary_rows.append(
                {
                    "candidateId": candidate, "labelId": STRUCTURAL_LABEL_ID,
                    "onsetMode": mode,
                    "controlId": "RECOMPUTED_COMPLETE_GROWTH_FISSION_BLOCK_ORDER_PERMUTATION",
                    "replicates": PERMUTATION_REPLICATES,
                    "observedPaperDistance": observed,
                    "nullLower2_5": lower,
                    "nullMedian": float(np.median(distances)),
                    "nullUpper97_5": float(np.quantile(distances, 0.975)),
                    "lowerTailP": float((1 + np.count_nonzero(distances <= observed)) / (len(distances) + 1)),
                    "negativeControlPassed": bool(observed < lower),
                    "labelsRecomputedAfterBlockPermutation": True,
                    "blockInternalOrderPreserved": True,
                    "generationNumbersSequentiallyReassigned": True,
                    "boundaryRecurrenceAndProjectionRecomputed": True,
                }
            )
    return pd.DataFrame(full_rows), pd.DataFrame(summary_rows)


def classify(
    aggregate: pd.DataFrame,
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    loo: pd.DataFrame,
    negative: pd.DataFrame,
    cross: pd.DataFrame,
    replay: pd.DataFrame,
    independent: pd.DataFrame,
    comparator_replay: pd.DataFrame,
    suffix: pd.DataFrame,
) -> dict[str, Any]:
    gates: dict[str, bool] = {
        "structuralNotComparator": True,
        "exactTwoPassReplayAll400LabelTrajectories": bool(replay["exactTwoPassReplayPassed"].all()),
        "independentStructuralReplayAll200Trajectories": bool(independent["passed"].all()),
        "exactFrozenAdjacentComparatorReplay": bool(comparator_replay["passed"].all()),
        "exactFutureSuffixInvarianceAll3000Sentinels": bool(suffix["passed"].all()),
        "allSuffixMutationsEffective": bool(suffix["suffixMutationEffective"].all()),
        "preciseHumanLockedPaperRelationship": True,
        "noOutcomeTunedChoice": True,
        "untouchedS20DesignComplete": (LOOP_ROOT / "untouched_s20_design.yaml").is_file(),
    }
    for candidate in CANDIDATE_IDS:
        agg = aggregate.loc[
            aggregate["candidateId"].eq(candidate) & aggregate["labelId"].eq(STRUCTURAL_LABEL_ID)
        ].iloc[0]
        comp = comparison.loc[
            comparison["candidateId"].eq(candidate) & comparison["labelId"].eq(STRUCTURAL_LABEL_ID)
        ]
        boot = bootstrap.loc[
            bootstrap["candidateId"].eq(candidate) & bootstrap["labelId"].eq(STRUCTURAL_LABEL_ID)
        ]
        influence = loo.loc[
            loo["candidateId"].eq(candidate) & loo["labelId"].eq(STRUCTURAL_LABEL_ID)
        ]
        control = negative.loc[
            negative["candidateId"].eq(candidate) & negative["labelId"].eq(STRUCTURAL_LABEL_ID)
        ]
        gates[f"occupancyCloser_{candidate}"] = bool(comp["occupancyCloser"].all())
        gates[f"jointDistanceBetterBothModes_{candidate}"] = bool(
            (comp["distanceDifferenceCandidateMinusComparator"] < 0).all()
        )
        gates[f"threeDimensionsIncludingOnsetOrConsistency_{candidate}"] = bool(
            (comp["closerDimensionCount"] >= 3).all() and comp["structureDimensionImproved"].all()
        )
        gates[f"bootstrapUpperBelowZeroBothModes_{candidate}"] = bool((boot["upper95"] < 0).all())
        gates[f"allLeaveOneOutImproved_{candidate}"] = bool((influence["distanceDifference"] < 0).all())
        gates[f"generationBlockPermutationControlBothModes_{candidate}"] = bool(control["negativeControlPassed"].all())
        gates[f"coverage_{candidate}"] = bool(
            int(agg["definedConsistencyCount"]) >= 95 and int(agg["observedOnsetCount"]) >= 95
        )
        gates[f"nonreplicatingAtQuarterPositive_{candidate}"] = bool(
            float(agg["nonreplicatingAtCutoffFraction"]) > 0
        )
        gates[f"noOnsetThroughQuarterPositive_{candidate}"] = bool(
            float(agg["noReplicatorThroughCutoffFraction"]) > 0
        )
    structural_cross = cross.loc[cross["labelId"].eq(STRUCTURAL_LABEL_ID)]
    differences = dict(zip(structural_cross["metric"], structural_cross["absoluteMeanDifference"], strict=True))
    gates["crossCandidateAgreement"] = bool(
        pd.notna(differences.get("occupancy"))
        and pd.notna(differences.get("consistency"))
        and pd.notna(differences.get("firstOnsetNormalizedScore"))
        and float(differences["occupancy"]) <= 0.05
        and float(differences["consistency"]) <= 0.10
        and float(differences["firstOnsetNormalizedScore"]) <= 0.10
    )
    promoted = bool(all(gates.values()))
    structural_comparison = comparison.loc[comparison["labelId"].eq(STRUCTURAL_LABEL_ID)]
    paper_match = bool((structural_comparison["paperDistance"] <= 1).all())
    directional = bool(
        structural_comparison["occupancyCloser"].all()
        and (structural_comparison["closerDimensionCount"] >= 3).all()
        and structural_comparison["structureDimensionImproved"].all()
    )
    primary_class = "EXPLORATORY_PAPER_MATCH" if paper_match else (
        "EXPLORATORY_DIRECTIONAL_MATCH" if directional else "EXPLORATORY_NON_SUPPORT"
    )
    structural_classes = [
        primary_class, "METHOD_DEPENDENT_LEAD", "AUTHOR_AMBIGUITY_UNRESOLVED",
        "PROMOTABLE_TO_S20" if promoted else "NOT_PROMOTABLE",
    ]
    return {
        "schema": "eidosoma.e01.s19_l06_classification.v1",
        "researchStepId": LOOP_ID,
        "confirmatoryVerdictIssued": False,
        "topLevelClassification": "PROMOTABLE_TO_S20" if promoted else primary_class,
        "outcomeClass": "SUPPORTIVE_EXPLORATORY" if promoted else "CONSTRAINING_OR_NULL_EXPLORATORY",
        "promotedLeadCount": int(promoted),
        "promotedLeadIds": [STRUCTURAL_LABEL_ID] if promoted else [],
        "labelClassifications": [
            {
                "labelId": COMPARATOR_LABEL_ID,
                "classifications": ["POSSIBLE_STABILITY_PROXY", "NOT_PROMOTABLE"],
                "promoted": False,
            },
            {
                "labelId": STRUCTURAL_LABEL_ID,
                "classifications": structural_classes,
                "promotionGates": gates,
                "promoted": promoted,
                "pastOnlyConstruction": True,
                "boundaryGranularity": True,
                "futureSuffixInvariant": bool(suffix["passed"].all()),
                "promotableScope": "RETROSPECTIVE_PAPER_FACING_ONLY",
                "prospectivePredictionClaim": False,
                "causalControlClaim": False,
            },
        ],
        "laterLoopActivated": False,
        "s20Activated": False,
        "mandatoryHumanReview": True,
    }


def verify_immutable_prior() -> dict[str, Any]:
    baseline = json.loads((LOOP_ROOT / "immutable_prior_baseline.json").read_text(encoding="utf-8"))
    mismatches = []
    for row in baseline["files"]:
        path = Path(row["path"])
        if not path.is_file():
            mismatches.append({"path": str(path), "reason": "missing"})
        elif sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            mismatches.append({"path": str(path), "reason": "size_or_hash_changed"})
    return {
        "schema": "eidosoma.e01.s19_l06_immutable_prior_postcheck.v1",
        "fileCount": len(baseline["files"]),
        "expectedAggregateSha256": baseline["aggregateSha256"],
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame if columns is None else frame[columns]
    return selected.to_markdown(index=False, floatfmt=".4f")


def report_text(
    aggregate: pd.DataFrame,
    comparison: pd.DataFrame,
    fixed_prior: pd.DataFrame,
    bootstrap: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    negative: pd.DataFrame,
    suffix: pd.DataFrame,
    overlap: pd.DataFrame,
    cross: pd.DataFrame,
    classification: dict[str, Any],
    validation_result: str,
    runtime: dict[str, Any],
    immutable_count: int,
) -> str:
    aggregate_view = aggregate[
        [
            "candidateId", "labelId", "meanOccupancy", "meanPersistence",
            "meanConsistency", "meanFirstOnsetRawIndex0", "meanFirstOnsetNormalized",
            "meanEntryCount", "meanExitCount", "meanEpisodeCount",
            "meanMeanEpisodeDuration", "meanLongestEpisode",
            "nonreplicatingAtCutoffFraction", "noReplicatorThroughCutoffFraction",
            "meanActivatedBoundaryCount", "meanActivatedBoundaryFraction",
            "meanFirstActivatedBoundaryGeneration",
        ]
    ].copy()
    comparison_view = comparison.loc[comparison["labelId"].eq(STRUCTURAL_LABEL_ID)][
        [
            "candidateId", "onsetMode", "paperDistance", "comparatorDistance",
            "distanceDifferenceCandidateMinusComparator", "distanceImprovementFraction",
            "closerDimensionCount", "structureDimensionImproved", "occupancyCloser",
        ]
    ]
    suffix_summary = suffix.groupby(["candidateId", "variant"], as_index=False).agg(
        sentinels=("passed", "size"), passed=("passed", "sum"),
        mutationsEffective=("suffixMutationEffective", "sum"),
    )
    fixed_view = fixed_prior.loc[
        fixed_prior["metric"].isin(
            ["occupancy", "persistence", "consistency", "firstOnsetRawScore", "firstOnsetNormalizedScore"]
        )
    ][
        ["candidateId", "fixedPriorLoopId", "metric", "l06Mean", "fixedPriorMean", "l06MinusFixedPrior"]
    ]
    gates = classification["labelClassifications"][1]["promotionGates"]
    gate_table = pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()])
    promoted = classification["promotedLeadCount"]
    conclusion = (
        "The singleton boundary label passed every promotion gate and is an exploratory retrospective paper-facing lead for untouched S20 confirmation."
        if promoted else
        "The singleton boundary label failed at least one locked promotion gate and was not promoted."
    )
    return f"""# E01/S19-L06 Full Results — Past-Only Multi-Attractor Boundary Recurrence

## Concise handoff summary

- **Research step ID:** `S19-L06`
- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Artifacts written:** complete L06 preregistration/method/label/bundle/seed locks; exact trajectory, molecular-clock, post-fission-boundary, adjacent-H and frozen-label replay; structural/comparator label rows; boundary activation evidence; complete temporal fingerprints; 3,000 suffix sentinels; 4,096-replicate paired matrix bootstrap and recomputed generation-block controls; leave-one-out, fixed-L03/L05 comparisons, validation, runtime/storage/status/ledger/hash artifacts; canonical full report and one-page decision summary
- **Validation result:** `{validation_result}`
- **Outcome classification:** `{classification['topLevelClassification']}`; `{classification['outcomeClass']}`; `{promoted}` lead(s) promoted
- **Caveats or blockers:** adaptive exploration on previously studied matrices; unavailable exact author label semantics; projected intervals may be artificially persistent; known paper fingerprints informed gates; no emergence, prediction, intervention, or causal-control evidence
- **Recommended next action:** mandatory human review. L07, S20, E02, author contact, and report generation remain inactive.

## Lay summary

L03 selected one dominant post-fission composition and was much too sparse. L05 allowed any earlier molecular state and activated too early. L06 tested the locked middle ground: only post-fission boundary compositions can establish recurrence, multiple recurring boundary states are allowed, and a decision is carried only through the immediately following growth interval before being recomputed. {conclusion}

The requested 88% occupancy remained one anchor, never the sole criterion. Persistence, onset, consistency, episodes, quarter-cutoff eligibility, recurrence counts, candidate agreement, bootstrap uncertainty, influence, exact future-suffix invariance, and a block-order negative control were evaluated jointly.

## Frozen question

Does strict past-only recurrence among selected post-fission boundaries, projected prospectively through the following growth interval, improve the locked four-dimensional paper fingerprint over adjacent molecular `H>0.9` in both candidates?

This is exploratory label reconstruction only. Even a promoted lead is restricted to retrospective paper-facing untouched confirmation and supplies no early-warning, prediction, intervention, or causal-control evidence.

## Inputs

- Frozen S13Y `trajectory_manifest.parquet`, `label_values.parquet`, and 200 trajectory caches: 100 shared matrices under each of candidate 2 and candidate 3.
- Frozen L03 post-fission boundary identities and modal-boundary comparison evidence; frozen L05 molecular past-only comparison evidence. Neither prior result was recomputed or changed.
- Original arXiv v1 paper and pinned historical GARD generation-trace source context.
- No new GARD trajectory, PhiRL/emergence calculation, prediction model, intervention, GPU computation, web outcome search, or author contact.

## Detailed methods

### Label contract

At selected post-fission boundary `b_g`, L06 activates the boundary iff strict historical cosine `H>0.9` links it to at least one already observed boundary `b_h` with `0<h<=g-2`. Each earlier generation counts once. The decision labels `b_g` and subsequent selected observations through, but excluding, `b_(g+1)`. Molecular rows before `b_1` are eligible negative; the single generation-zero initial state is ineligible. There is no future boundary, backfill, carry across a new boundary without reactivation, dominant compotype, clustering, centroid, threshold/count search, or alignment variant.

### Temporal fingerprint and paper distance

The catalytic matrix trajectory is the independent unit. The analysis reports occupancy, persistence, zero- and one-based raw onset, normalized onset, consecutive-label Pearson consistency, entries/exits, episode count and duration, longest episode, quarter-cutoff state and no-onset status, activated-boundary and recurrence counts, and cross-candidate agreement.

The locked joint distance is the RMS of four standardized deviations from paper control targets: persistence `716 +/- 198`, occupancy `0.88 +/- 0.03`, consistency `0.38 +/- 0.06`, and either raw onset `37 +/- 27` or normalized onset `0.37 +/- 0.27`. Raw and normalized onset remain separate because the paper's units are ambiguous.

### Robustness and falsification

- Both labels were materialized twice on all 200 trajectories.
- A separate scalar-loop implementation replayed every boundary decision, score, matching generation, projected row, and source-boundary identity.
- Deletion, shuffle, and component-replacement of the future suffix were tested at five locked endpoints per trajectory: 3,000 sentinels.
- Paired uncertainty used 4,096 matrix bootstraps per candidate; every leave-one-matrix-out omission was evaluated.
- The negative control permuted all 100 complete growth-fission blocks, preserved internal order, renumbered generations, and recomputed boundary recurrence plus projection for 4,096 replicates per matrix. Passing required observed raw and normalized distance below the null 2.5th percentile in each candidate.

## Results

### Candidate-specific temporal fingerprints

{markdown_table(aggregate_view)}

Full matrix distributions, medians, standard deviations, episodes, cutoff values, boundary activation rows, and recurrence diagnostics are machine-readable.

### Joint comparison with adjacent `H>0.9`

{markdown_table(comparison_view)}

Negative distance differences favor L06. Directional improvement is distinct from exact numerical fit and cannot bypass any promotion gate.

### Fixed L03 and L05 comparisons

{markdown_table(fixed_view)}

These are read-only descriptive contrasts with the nearest locked boundary and molecular past-only predecessors; they are not branch selection.

### Matrix bootstrap

{markdown_table(bootstrap, ['candidateId', 'onsetMode', 'meanDistanceDifference', 'lower95', 'upper95', 'probabilityDistanceImprovement'])}

{markdown_table(bootstrap_metrics, ['candidateId', 'metric', 'meanDifference', 'lower95', 'upper95'])}

### Future-suffix invariance

{markdown_table(suffix_summary)}

Every sentinel compares the complete prefix label, boundary score, recurrence counts, matching-boundary identities, and projected source-boundary generation.

### Recomputed generation-block negative control

{markdown_table(negative, ['candidateId', 'onsetMode', 'observedPaperDistance', 'nullLower2_5', 'nullMedian', 'nullUpper97_5', 'lowerTailP', 'negativeControlPassed'])}

### Comparator overlap and cross-candidate agreement

{markdown_table(overlap)}

{markdown_table(cross)}

### Promotion gates

{markdown_table(gate_table)}

At most one retrospective paper-facing lead could be promoted. Both quarter gates—nonreplicating at cutoff and no onset through cutoff—were separately mandatory in both candidates.

## Validation

`{validation_result}`

- `{immutable_count}` immutable S01-S18/V1/V2/S19-L01-L05 files, including the S17 waiver, matched their frozen sizes and SHA-256 values before and after execution.
- All 200 trajectory/cache identities, 180,635 selected-clock rows, 20,000 post-fission boundary identities, adjacent-H arrays, and frozen `H>0.9` labels replayed before outcomes.
- Exact two-pass, independent boundary/projection, frozen comparator, 3,000 suffix, 4,096 paired bootstrap, every leave-one-out, recomputed block-order, scope, storage, regeneration, and artifact checks passed.
- Launch attempt 1 stopped at module import before the execution gate or any trajectory load. Value-preserving amendment `S19-L06-VPA-001` added only the repository root to the runner import path, preserved the failure, and re-established a clean pushed pre-outcome lock.

## Commands, runtime, and dependencies

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l06.py
ruff check src/e01_s19_boundary_recurrence scripts/e01/prepare_s19_l06_lock.py scripts/e01/run_s19_l06.py tests/e01/test_s19_l06.py
python scripts/e01/prepare_s19_l06_lock.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s19_l06.py --workers 8
```

- Repository lock commit: `{runtime['repositoryCommit']}` on `eidosoma/groups/42`, pushed before outcome access.
- CPU float64 was authoritative; 8 one-thread workers; GPU unused.
- Scientific CPU hours: `{runtime['scientificCpuHours']:.6f}`; wall hours: `{runtime['wallHours']:.6f}`.
- Python `{runtime['python']}`, NumPy `{runtime['numpy']}`, pandas `{runtime['pandas']}`, SciPy `{runtime['scipy']}`, scikit-learn `{runtime['sklearn']}`, PyArrow `{runtime['pyarrow']}`.

## Provenance and regeneration

The exact contract is in `preregistration.yaml` and `method_lock.json`; inputs and source identities are hashed in `input_manifest.json` and `source_snapshot_manifest.json`; RNG identities are in `seed_manifest.parquet`; matrix results are in `results.parquet`; all boundary decisions are in `boundary_activation_results.parquet`; robustness evidence is in its named Parquet/CSV artifacts; and `artifact_manifest.json` records every retained file. Disposable label and permutation caches remain under `/cache/e01_s19_l06`.

## Caveats, blockers, and interpretation boundary

1. These 100 matrices and paper fingerprints were already studied, so L06 is adaptive and exploratory despite its pre-outcome lock.
2. The paper does not uniquely specify this past-only boundary activation/projection rule; source grounding does not establish author-code identity.
3. Projecting a boundary decision across a growth interval may produce artificial persistence.
4. Matching occupancy cannot rescue wrong onset, consistency, episodes, cutoff eligibility, uncertainty, negative controls, or candidate disagreement.
5. A past-only label construction is not itself an early-warning predictor. No emergence, prediction, intervention, or causal result changes.

## Outcome and recommended next action

**Classification:** `{classification['topLevelClassification']}` with `{promoted}` promoted lead(s). Stop at the mandatory human-review boundary. No L07, S20, E02, author contact, or report generation is active automatically.
"""


def decision_summary_text(
    aggregate: pd.DataFrame,
    negative: pd.DataFrame,
    suffix: pd.DataFrame,
    classification: dict[str, Any],
    validation_result: str,
) -> str:
    structural = aggregate.loc[aggregate["labelId"].eq(STRUCTURAL_LABEL_ID)][
        [
            "candidateId", "meanOccupancy", "meanPersistence", "meanConsistency",
            "meanFirstOnsetRawIndex0", "nonreplicatingAtCutoffFraction",
            "noReplicatorThroughCutoffFraction", "meanActivatedBoundaryCount",
        ]
    ]
    gates = classification["labelClassifications"][1]["promotionGates"]
    failed = [key for key, passed in gates.items() if not passed]
    return f"""# S19-L06 Decision Summary

## Concise handoff summary

- **Research step ID:** `S19-L06`
- **Completion status:** complete; mandatory human-review boundary active
- **Artifacts written:** full L06 lock, label/boundary/fingerprint/control evidence, validation and hash manifests, canonical full report, and append-only root ledgers
- **Validation result:** `{validation_result}`
- **Outcome classification:** `{classification['topLevelClassification']}`; promoted leads: `{classification['promotedLeadCount']}`
- **Caveats or blockers:** adaptive studied-matrix evidence; exact author semantics unavailable; no downstream emergence, prediction, intervention, or causal inference
- **Recommended next action:** human review only; no downstream action is active

## One-page decision evidence

{markdown_table(structural)}

- Future-suffix sentinels passed: `{int(suffix['passed'].sum())}/{len(suffix)}`.
- Generation-block controls passed: `{int(negative['negativeControlPassed'].sum())}/{len(negative)}`.
- Failed promotion gates: `{', '.join(failed) if failed else 'none'}`.
- Promoted lead IDs: `{', '.join(classification['promotedLeadIds']) if classification['promotedLeadIds'] else 'none'}`.

L06 is exploratory. A directional paper-facing resemblance cannot be called exact replication, early warning, prediction, intervention benefit, or causal control. Return control for mandatory human review.
"""


def artifact_manifest(root: Path, required: list[str], schema: str) -> dict[str, Any]:
    missing = [name for name in required if not (root / name).is_file()]
    manifest_path = root / "artifact_manifest.json"
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != manifest_path):
        files.append(
            {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return {
        "schema": schema, "root": str(root), "requiredFiles": required,
        "missing": missing, "fileCount": len(files),
        "totalBytes": int(sum(row["bytes"] for row in files)),
        "files": files, "passed": not missing,
    }


def append_postloop_ledger(
    aggregate: pd.DataFrame, classification: dict[str, Any], timestamp: str
) -> None:
    path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(path)
    if ledger["loopId"].eq(LOOP_ID).sum() != 1:
        raise RuntimeError("L06 pre-loop self-improvement row cardinality changed")
    compact = aggregate[
        [
            "candidateId", "labelId", "meanOccupancy", "meanPersistence",
            "meanConsistency", "meanFirstOnsetRawIndex0", "meanEpisodeCount",
            "nonreplicatingAtCutoffFraction", "noReplicatorThroughCutoffFraction",
            "meanActivatedBoundaryCount",
        ]
    ].to_dict(orient="records")
    gates = classification["labelClassifications"][1]["promotionGates"]
    failed = [key for key, passed in gates.items() if not passed]
    row = {
        "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
        "timestampUtc": timestamp,
        "loopId": LOOP_ID,
        "recordPhase": "POST_LOOP_LEARNING_AND_HUMAN_REVIEW_BOUNDARY",
        "beliefBeforeLoop": "Multiple past post-fission boundary attractors might bridge L03's sparse modal state and L05's permissive molecular recurrence.",
        "motivatingEvidence": "L03 and L05 bracketed the paper fingerprint under two different granularity extremes.",
        "failureOrAmbiguityTargeted": "Whether boundary-only recurrence with prospective interval projection creates a closer joint temporal fingerprint and quarter pre-onset eligibility.",
        "selectedHypotheses": "Exactly one strict-H>0.9, h<=g-2 post-fission-boundary recurrence and following-interval projection rule.",
        "learned": canonical_json(
            {"aggregateFingerprint": compact, "failedPromotionGates": failed, "promotedLeadIds": classification["promotedLeadIds"]}
        ),
        "weakenedHypotheses": "The exact singleton boundary rule to the extent it failed joint, suffix, replay, influence, permutation, coverage, quarter, or cross-candidate gates.",
        "remainingPlausibleHypotheses": "Only any promoted exact L06 lead or a separately human-authorized nonduplicative ambiguity; exact author semantics remain unavailable.",
        "proposedNextTest": "Mandatory human review; no automatic L07 or S20.",
        "informationGainRationale": "L06 isolated recurrence granularity and projection without changing threshold or accessing downstream outcomes.",
        "appendOnly": True,
    }
    pd.concat([ledger, pd.DataFrame([row])[ledger.columns]], ignore_index=True).to_parquet(path, index=False)
    with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Entry 012 — S19-L06 learning and human-review boundary\n\n"
            "- **Belief before the loop:** Multiple prior post-fission attractors might bridge L03's sparse modal label and L05's permissive molecular label.\n"
            "- **What was tested:** Exactly one strict-`H>0.9`, `0<h<=g-2` boundary recurrence rule projected through the following interval.\n"
            f"- **What was learned:** {classification['promotedLeadCount']} lead(s) passed every locked gate; exact results and failed gates are preserved in the L06 evidence.\n"
            "- **Hypotheses weakened:** This exact singleton rule to the extent it failed any joint, replay, suffix, influence, permutation, coverage, cutoff, or agreement gate.\n"
            "- **What remains plausible:** Only promoted leads, if any, or a separately authorized nonduplicative author ambiguity.\n"
            "- **Next action:** Mandatory human review; no automatic L07 or S20.\n"
            "- **Why another loop could add information:** It must isolate a different unresolved dependency and cannot retune L06.\n"
        )


def update_root_handoff(
    report: str,
    classification: dict[str, Any],
    validation_result: str,
    artifacts: list[str],
) -> None:
    (ARTIFACT_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": LOOP_ID,
        "stepNumber": 19,
        "success": True,
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": artifacts,
        "validationResult": validation_result,
        "caveatsOrBlockers": [
            "exploratory_previously_studied_matrices",
            "exact_author_replicator_definition_unavailable",
            "boundary_projection_may_inflate_persistence",
            "known_paper_fingerprint_informed_exploratory_gate",
            "no_emergence_prediction_intervention_or_causal_inference",
            "L07_S20_E02_inactive",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_SELECT_NEXT_BOUNDED_ACTION",
        "promotedLeadCount": classification["promotedLeadCount"],
        "promotedLeadIds": classification["promotedLeadIds"],
        "fixedLoopCap": None,
    }
    write_json(ARTIFACT_ROOT / "s19_status.json", status)
    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    for loop in registry["loops"]:
        if loop["loopId"] == LOOP_ID:
            loop.update(
                {
                    "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
                    "outcomeAccessed": True,
                    "completed": True,
                    "eligibleScientificResults": True,
                    "promotedLeadCount": classification["promotedLeadCount"],
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
            "date": datetime.now(timezone.utc).date().isoformat(),
            "decision": "S19_L06_COMPLETE_MANDATORY_HUMAN_REVIEW",
            "scope": VERSION,
            "source": "locked_execution_result",
        }
    )
    history["pendingDecision"] = "POST_S19_L06_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)


def main(workers: int) -> None:
    configure_shared_helpers()
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    started_utc = datetime.now(timezone.utc)
    lock = execution_lock_validation()
    write_json(LOOP_ROOT / "execution_lock_validation.json", lock)
    if not lock["passed"]:
        raise RuntimeError("L06 execution lock validation failed")

    manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet")
    labels, boundaries, diagnostics, replay, independent, suffix, execution = execute_trajectories(manifest, workers)
    if not execution["success"].all():
        raise RuntimeError("L06 trajectory execution or locked audit failed")
    if not replay["exactTwoPassReplayPassed"].all():
        raise RuntimeError("L06 exact two-pass replay failed")
    if not independent["passed"].all():
        raise RuntimeError("L06 independent structural replay failed")
    if not suffix["passed"].all() or not suffix["suffixMutationEffective"].all():
        raise RuntimeError("L06 future-suffix invariance failed")
    comparator_replay = shared.frozen_comparator_replay(labels)
    if not comparator_replay["passed"].all():
        raise RuntimeError("L06 frozen adjacent-H comparator replay failed")

    fingerprints = build_fingerprints(labels, diagnostics)
    episodes = shared.episode_table(labels)
    aggregate = shared.aggregate_fingerprints(fingerprints)
    comparison = shared.paper_comparison(aggregate)
    fixed_prior = fixed_prior_comparison(aggregate)
    bootstrap, bootstrap_metrics = shared.bootstrap_analysis(fingerprints)
    loo = shared.leave_one_out(fingerprints)
    overlaps = shared.overlap_results(labels)
    cross = shared.cross_candidate_agreement(fingerprints, aggregate)
    permutation, negative = generation_block_permutation(execution, comparison)
    robustness = shared.robustness_table(bootstrap, loo, negative, suffix)
    classification = classify(
        aggregate, comparison, bootstrap, loo, negative, cross,
        replay, independent, comparator_replay, suffix,
    )

    labels.to_parquet(LOOP_ROOT / "label_values.parquet", index=False, compression="zstd")
    labels.loc[labels["labelId"].eq(STRUCTURAL_LABEL_ID)].to_parquet(
        LOOP_ROOT / "boundary_projected_label_results.parquet", index=False, compression="zstd"
    )
    boundaries.to_parquet(LOOP_ROOT / "boundary_activation_results.parquet", index=False, compression="zstd")
    diagnostics.to_parquet(LOOP_ROOT / "boundary_recurrence_trajectory_diagnostics.parquet", index=False)
    replay.to_parquet(LOOP_ROOT / "label_replay_evidence.parquet", index=False)
    independent.to_parquet(LOOP_ROOT / "independent_label_replay.parquet", index=False)
    comparator_replay.to_parquet(LOOP_ROOT / "frozen_comparator_replay.parquet", index=False)
    suffix.to_parquet(LOOP_ROOT / "future_suffix_invariance_results.parquet", index=False)
    write_json(
        LOOP_ROOT / "future_suffix_invariance_summary.json",
        {
            "schema": "eidosoma.e01.s19_l06_future_suffix_invariance_summary.v1",
            "sentinelCount": len(suffix), "endpointCountPerTrajectory": 5,
            "variantCount": len(SUFFIX_VARIANTS), "passedCount": int(suffix["passed"].sum()),
            "mutationEffectiveCount": int(suffix["suffixMutationEffective"].sum()),
            "passed": bool(suffix["passed"].all() and suffix["suffixMutationEffective"].all()),
        },
    )
    execution.to_parquet(LOOP_ROOT / "execution_status.parquet", index=False)
    fingerprints.to_parquet(LOOP_ROOT / "fingerprint_results.parquet", index=False)
    fingerprints.to_parquet(LOOP_ROOT / "results.parquet", index=False)
    aggregate.to_parquet(LOOP_ROOT / "fingerprint_summary.parquet", index=False)
    aggregate.to_csv(LOOP_ROOT / "fingerprint_aggregate.csv", index=False)
    episodes.to_parquet(LOOP_ROOT / "episode_results.parquet", index=False)
    fingerprints[
        [
            "candidateId", "matrixIndex", "labelId", "cutoffCount", "cutoffIndex0",
            "isNonreplicatingAtCutoff", "noReplicatorObservedThroughCutoff",
        ]
    ].to_parquet(LOOP_ROOT / "cutoff_results.parquet", index=False)
    diagnostics.loc[diagnostics["labelId"].eq(STRUCTURAL_LABEL_ID)].to_parquet(
        LOOP_ROOT / "boundary_recurrence_count_results.parquet", index=False
    )
    overlaps.to_parquet(LOOP_ROOT / "label_overlap_results.parquet", index=False)
    cross.to_csv(LOOP_ROOT / "cross_candidate_agreement.csv", index=False)
    comparison.to_csv(LOOP_ROOT / "paper_fingerprint_comparison.csv", index=False)
    fixed_prior.to_csv(LOOP_ROOT / "fixed_l03_l05_comparison.csv", index=False)
    bootstrap.to_parquet(LOOP_ROOT / "paper_distance_bootstrap.parquet", index=False)
    bootstrap_metrics.to_parquet(LOOP_ROOT / "bootstrap_metric_differences.parquet", index=False)
    loo.to_parquet(LOOP_ROOT / "leave_one_out_robustness.parquet", index=False)
    permutation.to_parquet(LOOP_ROOT / "generation_block_permutation_results.parquet", index=False)
    negative.to_parquet(LOOP_ROOT / "negative_control_results.parquet", index=False)
    robustness.to_parquet(LOOP_ROOT / "robustness_results.parquet", index=False)
    pd.DataFrame(
        [
            {
                "failureId": "L06-LAUNCH-001",
                "phase": "PRE_EXECUTION_MODULE_IMPORT",
                "status": "VALUE_PRESERVING_AMENDMENT_APPLIED",
                "reason": "MODULE_NOT_FOUND_SCRIPTS_REPOSITORY_ROOT_ABSENT_FROM_SYS_PATH",
                "scientificOutcomesAccessed": False,
                "repairAttempted": True,
            }
        ]
    ).to_csv(LOOP_ROOT / "failure_ledger.csv", index=False)
    write_json(LOOP_ROOT / "classification.json", classification)

    immutable = verify_immutable_prior()
    write_json(LOOP_ROOT / "immutable_prior_postcheck.json", immutable)
    if not immutable["passed"]:
        raise RuntimeError("L06 immutable prior changed")
    completed_utc = datetime.now(timezone.utc)
    scientific_cpu_seconds = float(execution["cpuSeconds"].sum()) + (time.process_time() - started_cpu)
    runtime = {
        "schema": "eidosoma.e01.s19_l06_runtime_manifest.v1",
        "startedUtc": started_utc.isoformat(), "completedUtc": completed_utc.isoformat(),
        "wallHours": (time.perf_counter() - started_wall) / 3600,
        "scientificCpuHours": scientific_cpu_seconds / 3600,
        "cpuCeilingHours": 32.0, "gpuHours": 0.0, "gpuCeilingHours": 0.0,
        "workerCount": workers, "numericalLibraryThreadsPerWorker": 1,
        "cpuFloat64Authoritative": True, "gpuUsed": False,
        "newGardTrajectories": 0, "newPhiRLOrEmergenceValues": 0,
        "repositoryCommit": lock["repositoryHead"], "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "pyarrow": pyarrow.__version__,
    }
    if runtime["scientificCpuHours"] > 32 or runtime["wallHours"] > 8:
        raise RuntimeError("L06 compute ceiling exceeded")
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime)

    retained_bytes = sum(path.stat().st_size for path in LOOP_ROOT.rglob("*") if path.is_file())
    temporary_bytes = sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l06_storage_validation.v1",
        "retainedBytesBeforeManifest": retained_bytes,
        "retainedGiB": retained_bytes / (1024**3), "retainedCeilingGiB": 10.0,
        "temporaryBytes": temporary_bytes, "temporaryGiB": temporary_bytes / (1024**3),
        "temporaryCeilingGiB": 25.0,
        "passed": bool(retained_bytes <= 10 * 1024**3 and temporary_bytes <= 25 * 1024**3),
    }
    write_json(LOOP_ROOT / "storage_validation.json", storage)
    if not storage["passed"]:
        raise RuntimeError("L06 storage ceiling exceeded")

    validation_result = "PASS_ALL_LOCK_PREANALYSIS_BOUNDARY_PRIMARY_INDEPENDENT_COMPARATOR_SUFFIX_IMMUTABILITY_BOOTSTRAP_LOO_RECOMPUTED_GENERATION_BLOCK_STORAGE_REGENERATION_AND_HASH_CHECKS"
    report = report_text(
        aggregate, comparison, fixed_prior, bootstrap, bootstrap_metrics,
        negative, suffix, overlaps, cross, classification, validation_result,
        runtime, immutable["fileCount"],
    )
    decision = decision_summary_text(aggregate, negative, suffix, classification, validation_result)
    regenerated = report_text(
        aggregate, comparison, fixed_prior, bootstrap, bootstrap_metrics,
        negative, suffix, overlaps, cross, classification, validation_result,
        runtime, immutable["fileCount"],
    )
    if report != regenerated:
        raise RuntimeError("L06 report regeneration mismatch")
    (LOOP_ROOT / "S19_L06_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "loop_decision_summary.md").write_text(decision, encoding="utf-8")

    regeneration = {
        "schema": "eidosoma.e01.s19_l06_regeneration_validation.v1",
        "labelCount": int(labels["labelId"].nunique()), "structuralCandidateCount": 1,
        "candidateCount": int(labels["candidateId"].nunique()),
        "matrixCountPerCandidate": labels.groupby("candidateId")["matrixIndex"].nunique().to_dict(),
        "trajectoryCount": int(execution.shape[0]), "labelRowCount": len(labels),
        "structuralLabelRowCount": int(labels["labelId"].eq(STRUCTURAL_LABEL_ID).sum()),
        "boundaryActivationRowCount": len(boundaries), "fingerprintRowCount": len(fingerprints),
        "suffixSentinelCount": len(suffix), "permutationResultRowCount": len(permutation),
        "exactTwoPassReplayPassed": bool(replay["exactTwoPassReplayPassed"].all()),
        "independentPrimaryReplayPassed": bool(independent["passed"].all()),
        "frozenComparatorReplayPassed": bool(comparator_replay["passed"].all()),
        "futureSuffixInvariancePassed": bool(suffix["passed"].all()),
        "allSuffixMutationsEffective": bool(suffix["suffixMutationEffective"].all()),
        "immutablePriorPassed": immutable["passed"], "reportDeterministic": True,
        "thresholdGridAbsent": True, "recurrenceCountSearchAbsent": True,
        "variantCount": 1, "newGardTrajectories": 0, "newPhiRLOrEmergenceValues": 0,
        "passed": bool(
            labels["labelId"].nunique() == 2 and labels["candidateId"].nunique() == 2
            and execution.shape[0] == 200 and fingerprints.shape[0] == 400
            and len(boundaries) == 20000 and len(suffix) == 200 * 5 * len(SUFFIX_VARIANTS)
            and len(permutation) == 2 * 2 * PERMUTATION_REPLICATES
            and replay["exactTwoPassReplayPassed"].all() and independent["passed"].all()
            and comparator_replay["passed"].all() and suffix["passed"].all()
            and suffix["suffixMutationEffective"].all() and immutable["passed"] and storage["passed"]
        ),
    }
    write_json(LOOP_ROOT / "regeneration_validation.json", regeneration)
    if not regeneration["passed"]:
        raise RuntimeError("L06 final regeneration validation failed")

    loop_artifacts = [
        str(LOOP_ROOT / "S19_L06_FULL_RESULTS.md"), str(LOOP_ROOT / "classification.json"),
        str(LOOP_ROOT / "fingerprint_results.parquet"),
        str(LOOP_ROOT / "boundary_activation_results.parquet"),
        str(LOOP_ROOT / "paper_fingerprint_comparison.csv"),
        str(LOOP_ROOT / "future_suffix_invariance_results.parquet"),
        str(LOOP_ROOT / "generation_block_permutation_results.parquet"),
        str(LOOP_ROOT / "negative_control_results.parquet"),
    ]
    loop_status = {
        "researchStepId": LOOP_ID, "stepNumber": 19, "success": True,
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": loop_artifacts, "validationResult": validation_result,
        "caveatsOrBlockers": [
            "exploratory_previously_studied_matrices",
            "exact_author_replicator_definition_unavailable",
            "boundary_projection_may_inflate_persistence",
            "no_emergence_prediction_intervention_or_causal_inference",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_SELECT_NEXT_BOUNDED_ACTION",
    }
    write_json(LOOP_ROOT / "status.json", loop_status)
    append_postloop_ledger(aggregate, classification, completed_utc.isoformat())
    update_root_handoff(report, classification, validation_result, loop_artifacts)

    required = [
        "preregistration.yaml", "method_lock.json", "candidate_ranking.csv",
        "candidate_bundle_registry.yaml", "label_registry.yaml", "label_registry.parquet",
        "specification_ledger.parquet", "seed_manifest.parquet", "input_manifest.json",
        "source_snapshot_manifest.json", "untouched_s20_design.yaml",
        "preoutcome_repository_lock.json", "immutable_prior_baseline.json",
        "value_preserving_amendment_001.json",
        "immutable_prior_validation.json", "compute_benchmark.json",
        "preanalysis_replay_evidence.parquet", "preanalysis_replay_validation.json",
        "execution_lock_validation.json", "execution_status.parquet", "label_values.parquet",
        "boundary_projected_label_results.parquet", "boundary_activation_results.parquet",
        "boundary_recurrence_trajectory_diagnostics.parquet", "label_replay_evidence.parquet",
        "independent_label_replay.parquet", "frozen_comparator_replay.parquet",
        "future_suffix_invariance_results.parquet", "future_suffix_invariance_summary.json",
        "fingerprint_results.parquet", "results.parquet", "fingerprint_summary.parquet",
        "fingerprint_aggregate.csv", "episode_results.parquet", "cutoff_results.parquet",
        "boundary_recurrence_count_results.parquet", "label_overlap_results.parquet",
        "cross_candidate_agreement.csv", "paper_fingerprint_comparison.csv",
        "fixed_l03_l05_comparison.csv", "paper_distance_bootstrap.parquet",
        "bootstrap_metric_differences.parquet", "leave_one_out_robustness.parquet",
        "generation_block_permutation_results.parquet", "negative_control_results.parquet",
        "robustness_results.parquet", "failure_ledger.csv", "runtime_manifest.json",
        "storage_validation.json", "regeneration_validation.json", "classification.json",
        "status.json", "loop_decision_summary.md", "S19_L06_FULL_RESULTS.md",
        "research_step_full_results.md", "preparation_runtime.json",
    ]
    loop_manifest = artifact_manifest(LOOP_ROOT, required, "eidosoma.e01.s19_l06_artifact_manifest.v1")
    if not loop_manifest["passed"]:
        raise RuntimeError(f"missing L06 artifacts: {loop_manifest['missing']}")
    write_json(LOOP_ROOT / "artifact_manifest.json", loop_manifest)
    root_required = [
        "continuation_decision.md", "s18_immutable_baseline.json",
        "self_improvement_ledger.parquet", "SELF_IMPROVEMENT_LEDGER.md",
        "candidate_registry.parquet", "source_search_ledger.parquet",
        "source_search_report.md", "loop_registry.yaml", "human_review_history.json",
        "s19_status.json", "research_step_full_results.md",
    ]
    root_manifest = artifact_manifest(ARTIFACT_ROOT, root_required, "eidosoma.e01.s19_artifact_manifest.v6")
    if not root_manifest["passed"]:
        raise RuntimeError(f"missing S19 root artifacts: {root_manifest['missing']}")
    write_json(ARTIFACT_ROOT / "artifact_manifest.json", root_manifest)
    print(
        canonical_json(
            {
                "loopId": LOOP_ID, "status": loop_status["status"],
                "validationResult": validation_result,
                "classification": classification["topLevelClassification"],
                "promotedLeadCount": classification["promotedLeadCount"],
                "promotedLeadIds": classification["promotedLeadIds"],
                "labelRows": len(labels), "boundaryRows": len(boundaries),
                "fingerprintRows": len(fingerprints), "suffixSentinels": len(suffix),
                "permutationRows": len(permutation),
                "scientificCpuHours": runtime["scientificCpuHours"],
                "wallHours": runtime["wallHours"], "mandatoryHumanReview": True,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 8:
        raise SystemExit("workers must be in [1,8]")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(variable) not in (None, "1"):
            raise SystemExit(f"{variable} must be unset or 1")
    main(arguments.workers)
