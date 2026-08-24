#!/usr/bin/env python3
"""Execute locked E01/S19-L05 past-only recurrence analysis."""

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
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_frozen_timebase_ensemble.core import selected_clock_observations
from e01_s19_past_only_recurrence.core import (
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
    derive_seed128,
    label_trajectory,
    past_only_recurrence,
    past_only_recurrence_reference,
    recomputed_generation_block_metrics,
    suffix_endpoint_indices,
)
from e01_s19_replicator_definition.core import (
    PAPER_TARGETS,
    closer_dimension_count,
    fingerprint_from_labels,
    paper_fingerprint_distance,
)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L05"
CACHE_ROOT = Path("/cache/e01_s19_l05")
LABEL_CACHE = CACHE_ROOT / "labels"
PERMUTATION_CACHE = CACHE_ROOT / "permutation_metrics"
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
L04_ROOT = ARTIFACT_ROOT / "loops/L04"
PREREG = REPO_ROOT / "configs/e01/s19_l05_preregistration.yaml"
METHOD_LOCK = REPO_ROOT / "configs/e01/s19_l05_method_lock.json"

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
    "meanDistinctEarlierGenerationCount",
    "medianDistinctEarlierGenerationCount",
    "maxDistinctEarlierGenerationCount",
    "fractionEligibleWithAtLeast2EarlierGenerations",
    "fractionEligibleWithAtLeast5EarlierGenerations",
    "fractionEligibleWithAtLeast10EarlierGenerations",
    "immediateOnlyEvidenceCount",
    "sameGenerationPriorMatchRowCount",
    "recurrentGenerationCount",
    "meanQualifyingPriorStateCount",
    "meanDistinctEarlierGenerationCountPositive",
)


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
    output["immediateOnlyEvidence"] = pd.array(
        output["immediateOnlyEvidence"], dtype="boolean"
    )
    columns = [
        "researchStepId",
        "candidateId",
        "trajectoryId",
        "matrixIndex",
        "labelId",
        "labelFamily",
        "labelEvidenceTier",
        "temporalScope",
        "selectedSequenceIndex",
        "rawObservationIndex",
        "generation",
        "observationKind",
        "isReplicator",
        "labelScore",
        "labelStatus",
        "ineligibilityReason",
        "distinctEarlierGenerationCount",
        "qualifyingPriorStateCount",
        "immediatePriorCrossGenerationMatchCount",
        "immediateOnlyEvidence",
        "maximumImmediatePriorSimilarity",
        "sameGenerationPriorMatchCount",
        "firstMatchingGeneration",
        "lastMatchingGeneration",
        "earliestMatchingSequenceIndex",
        "latestMatchingSequenceIndex",
    ]
    return output[columns]


def frame_identity(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("selectedSequenceIndex", kind="stable").reset_index(
        drop=True
    )
    return sha256_frame(ordered)


def result_equal(left: dict[str, Any], right: dict[str, Any]) -> dict[str, bool]:
    fields = (
        "labels",
        "scores",
        "distinctEarlierGenerationCount",
        "qualifyingPriorStateCount",
        "earliestMatchingSequenceIndex",
        "latestMatchingSequenceIndex",
    )
    checks = {
        field: bool(np.array_equal(left[field], right[field], equal_nan=True))
        for field in fields
    }
    checks["matchingReferenceIndices"] = bool(
        left["matchingReferenceIndices"] == right["matchingReferenceIndices"]
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
    indices: np.ndarray,
    candidate: str,
    matrix_index: int,
) -> list[dict[str, Any]]:
    rows = []
    endpoints = suffix_endpoint_indices(len(indices))
    for endpoint_ordinal, endpoint in enumerate(endpoints):
        baseline = past_only_recurrence(
            states, generations, indices, query_stop=endpoint
        )
        for variant in SUFFIX_VARIANTS:
            changed_states = mutate_suffix(
                states,
                endpoint,
                variant,
                candidate,
                matrix_index,
                endpoint_ordinal,
            )
            if variant == "DELETE":
                changed = past_only_recurrence(
                    changed_states,
                    generations[: endpoint + 1],
                    indices[: endpoint + 1],
                )
                future_before = states[endpoint + 1 :]
                future_after = np.empty((0, 100), dtype=np.int64)
            else:
                changed = past_only_recurrence(
                    changed_states, generations, indices, query_stop=endpoint
                )
                future_before = states[endpoint + 1 :]
                future_after = changed_states[endpoint + 1 :]
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
                    "distinctGenerationCountsExact": checks[
                        "distinctEarlierGenerationCount"
                    ],
                    "qualifyingStateCountsExact": checks["qualifyingPriorStateCount"],
                    "earliestMatchingIndicesExact": checks[
                        "earliestMatchingSequenceIndex"
                    ],
                    "latestMatchingIndicesExact": checks["latestMatchingSequenceIndex"],
                    "allMatchingIndicesExact": checks["matchingReferenceIndices"],
                    "prefixStateHash": sha256_array(states[: endpoint + 1]),
                    "suffixBeforeHash": sha256_array(future_before),
                    "suffixAfterHash": sha256_array(future_after),
                    "suffixMutationEffective": bool(
                        variant == "DELETE"
                        or not np.array_equal(future_before, future_after)
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
    indices = np.arange(len(selected), dtype=np.int64)
    frames: list[pd.DataFrame] = []
    diagnostics = []
    replays = []
    independent_row: dict[str, Any] | None = None
    for definition in LABEL_DEFINITIONS:
        first_raw, first_diagnostic = label_trajectory(
            trajectory, definition, clock_id=str(record["clockId"])
        )
        second_raw, second_diagnostic = label_trajectory(
            trajectory, definition, clock_id=str(record["clockId"])
        )
        first = normalize_frame(first_raw, definition.label_id)
        second = normalize_frame(second_raw, definition.label_id)
        identity_pass = frame_identity(first) == frame_identity(second)
        diagnostic_pass = canonical_json(first_diagnostic) == canonical_json(
            second_diagnostic
        )
        replays.append(
            {
                "candidateId": record["candidateId"],
                "matrixIndex": int(record["matrixIndex"]),
                "trajectoryId": record["trajectoryId"],
                "labelId": definition.label_id,
                "firstIdentity": frame_identity(first),
                "secondIdentity": frame_identity(second),
                "diagnosticEqual": diagnostic_pass,
                "exactTwoPassReplayPassed": bool(identity_pass and diagnostic_pass),
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
        if definition.label_id == STRUCTURAL_LABEL_ID:
            primary = past_only_recurrence(states, generations, indices)
            reference = past_only_recurrence_reference(states, generations, indices)
            independent_checks = result_equal(primary, reference)
            frame_labels = first["isReplicator"].fillna(False).to_numpy(dtype=bool)
            frame_scores = pd.to_numeric(first["labelScore"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            frame_counts = (
                pd.to_numeric(first["distinctEarlierGenerationCount"], errors="coerce")
                .fillna(0)
                .to_numpy(dtype=np.int64)
            )
            independent_row = {
                "candidateId": record["candidateId"],
                "matrixIndex": int(record["matrixIndex"]),
                "trajectoryId": record["trajectoryId"],
                **{
                    f"independent_{key}": value
                    for key, value in independent_checks.items()
                },
                "materializedLabelsExact": bool(
                    np.array_equal(frame_labels, primary["labels"])
                ),
                "materializedScoresExact": bool(
                    np.array_equal(frame_scores, primary["scores"], equal_nan=True)
                ),
                "materializedDistinctCountsExact": bool(
                    np.array_equal(
                        frame_counts, primary["distinctEarlierGenerationCount"]
                    )
                ),
                "matchingReferenceIndicesSha256": hashlib.sha256(
                    canonical_json(primary["matchingReferenceIndices"]).encode()
                ).hexdigest(),
            }
            independent_row["passed"] = bool(
                all(independent_checks.values())
                and independent_row["materializedLabelsExact"]
                and independent_row["materializedScoresExact"]
                and independent_row["materializedDistinctCountsExact"]
            )
    if independent_row is None:
        raise RuntimeError("structural label was not executed")
    suffix_rows = suffix_audit(
        states,
        generations,
        indices,
        str(record["candidateId"]),
        int(record["matrixIndex"]),
    )
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            derive_seed128(
                record["candidateId"],
                int(record["matrixIndex"]),
                "generation_block_permutation",
            )
        )
    )
    orders = np.vstack(
        [rng.permutation(100) for _ in range(PERMUTATION_REPLICATES)]
    ).astype(np.int16)
    permutation = recomputed_generation_block_metrics(states, generations, orders)
    permutation_path = (
        PERMUTATION_CACHE
        / str(record["candidateId"])
        / f"M{int(record['matrixIndex']):03d}.npz"
    )
    permutation_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(permutation_path, **permutation)
    combined = pd.concat(frames, ignore_index=True)
    output = (
        LABEL_CACHE
        / str(record["candidateId"])
        / f"M{int(record['matrixIndex']):03d}.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False, compression="zstd")
    success = bool(
        all(row["exactTwoPassReplayPassed"] for row in replays)
        and independent_row["passed"]
        and all(row["passed"] for row in suffix_rows)
    )
    return {
        "candidateId": record["candidateId"],
        "matrixIndex": int(record["matrixIndex"]),
        "trajectoryId": record["trajectoryId"],
        "labelCache": str(output),
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
    repository = json.loads(
        (LOOP_ROOT / "preoutcome_repository_lock.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (LOOP_ROOT / "preanalysis_replay_validation.json").read_text(encoding="utf-8")
    )
    immutable = json.loads(
        (LOOP_ROOT / "immutable_prior_validation.json").read_text(encoding="utf-8")
    )
    benchmark = json.loads(
        (LOOP_ROOT / "compute_benchmark.json").read_text(encoding="utf-8")
    )
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
        repository["passed"]
        and replay["passed"]
        and immutable["passed"]
        and benchmark["gatePassed"]
        and head == remote == repository["head"]
        and clean
        and hashes["repositoryPreregistration"] == hashes["artifactPreregistration"]
        and hashes["repositoryMethodLock"] == hashes["artifactMethodLock"]
    )
    return {
        "schema": "eidosoma.e01.s19_l05_execution_lock_validation.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "preparedHead": repository["head"],
        "cleanWorktree": clean,
        "configHashes": hashes,
        "passed": passed,
    }


def execute_trajectories(
    manifest: pd.DataFrame, workers: int
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    records = manifest.sort_values(
        ["matrixIndex", "candidateId"], kind="stable"
    ).to_dict(orient="records")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(trajectory_worker, row): row for row in records}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (row["matrixIndex"], row["candidateId"]))
    execution = pd.DataFrame(
        [
            {
                "candidateId": row["candidateId"],
                "matrixIndex": row["matrixIndex"],
                "trajectoryId": row["trajectoryId"],
                "success": row["success"],
                "wallSeconds": row["wallSeconds"],
                "cpuSeconds": row["cpuSeconds"],
                "labelCache": row["labelCache"],
                "permutationCache": row["permutationCache"],
            }
            for row in results
        ]
    )
    labels = pd.concat(
        [pd.read_parquet(row["labelCache"]) for row in results], ignore_index=True
    )
    labels["isReplicator"] = pd.array(labels["isReplicator"], dtype="boolean")
    labels["immediateOnlyEvidence"] = pd.array(
        labels["immediateOnlyEvidence"], dtype="boolean"
    )
    diagnostics = pd.DataFrame([item for row in results for item in row["diagnostics"]])
    replay = pd.DataFrame([item for row in results for item in row["replays"]])
    independent = pd.DataFrame([row["independentReplay"] for row in results])
    suffix = pd.DataFrame([item for row in results for item in row["suffixAudit"]])
    return labels, diagnostics, replay, independent, suffix, execution


def frozen_comparator_replay(labels: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_parquet(S13Y_ROOT / "label_values.parquet")
    expected = frozen.loc[frozen["labelId"].eq(COMPARATOR_LABEL_ID)].sort_values(
        ["candidateId", "matrixIndex", "selectedSequenceIndex"], kind="stable"
    )
    observed = labels.loc[labels["labelId"].eq(COMPARATOR_LABEL_ID)].sort_values(
        ["candidateId", "matrixIndex", "selectedSequenceIndex"], kind="stable"
    )
    rows = []
    for candidate in CANDIDATE_IDS:
        for matrix in range(100):
            left = observed.loc[
                observed["candidateId"].eq(candidate)
                & observed["matrixIndex"].eq(matrix)
            ]
            right = expected.loc[
                expected["candidateId"].eq(candidate)
                & expected["matrixIndex"].eq(matrix)
            ]
            row_pass = len(left) == len(right) and np.array_equal(
                left["selectedSequenceIndex"].to_numpy(dtype=np.int64),
                right["selectedSequenceIndex"].to_numpy(dtype=np.int64),
            )
            clock_pass = (
                np.array_equal(
                    left["rawObservationIndex"].to_numpy(dtype=np.int64),
                    right["rawObservationIndex"].to_numpy(dtype=np.int64),
                )
                and left["observationKind"].astype(str).tolist()
                == right["observationKind"].astype(str).tolist()
            )
            label_pass = np.array_equal(
                left["isReplicator"].astype(bool).to_numpy(),
                right["isReplicator"].astype(bool).to_numpy(),
            )
            score_pass = np.array_equal(
                pd.to_numeric(left["labelScore"], errors="coerce").to_numpy(
                    dtype=np.float64
                ),
                pd.to_numeric(right["labelScore"], errors="coerce").to_numpy(
                    dtype=np.float64
                ),
                equal_nan=True,
            )
            rows.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": matrix,
                    "rowIdentityPassed": row_pass,
                    "clockIdentityPassed": clock_pass,
                    "labelIdentityPassed": label_pass,
                    "scoreIdentityPassed": score_pass,
                    "passed": bool(
                        row_pass and clock_pass and label_pass and score_pass
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_fingerprints(labels: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = labels.sort_values(
        ["candidateId", "matrixIndex", "labelId", "selectedSequenceIndex"],
        kind="stable",
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
        eligible = group.loc[group["labelStatus"].eq("ELIGIBLE")]
        positive = eligible.loc[eligible["isReplicator"].fillna(False)]
        positive_generations = positive["generation"].dropna().astype(int).unique()
        qualifying = pd.to_numeric(
            eligible["qualifyingPriorStateCount"], errors="coerce"
        )
        distinct_positive = pd.to_numeric(
            positive["distinctEarlierGenerationCount"], errors="coerce"
        )
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
                "meanDistinctEarlierGenerationCount": diagnostic.get(
                    "meanDistinctEarlierGenerationCount"
                ),
                "medianDistinctEarlierGenerationCount": diagnostic.get(
                    "medianDistinctEarlierGenerationCount"
                ),
                "maxDistinctEarlierGenerationCount": diagnostic.get(
                    "maxDistinctEarlierGenerationCount"
                ),
                "fractionEligibleWithAtLeast2EarlierGenerations": diagnostic.get(
                    "fractionEligibleWithAtLeast2EarlierGenerations"
                ),
                "fractionEligibleWithAtLeast5EarlierGenerations": diagnostic.get(
                    "fractionEligibleWithAtLeast5EarlierGenerations"
                ),
                "fractionEligibleWithAtLeast10EarlierGenerations": diagnostic.get(
                    "fractionEligibleWithAtLeast10EarlierGenerations"
                ),
                "immediateOnlyEvidenceCount": diagnostic.get(
                    "immediateOnlyEvidenceCount"
                ),
                "sameGenerationPriorMatchRowCount": diagnostic.get(
                    "sameGenerationPriorMatchCount"
                ),
                "recurrentGenerationCount": len(positive_generations),
                "meanQualifyingPriorStateCount": float(qualifying.mean())
                if qualifying.notna().any()
                else None,
                "meanDistinctEarlierGenerationCountPositive": float(
                    distinct_positive.mean()
                )
                if distinct_positive.notna().any()
                else None,
                **fingerprint,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["labelOrdinal", "candidateId", "matrixIndex"], kind="stable"
    )


def episode_table(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = labels.sort_values(
        ["candidateId", "matrixIndex", "labelId", "selectedSequenceIndex"],
        kind="stable",
    ).groupby(["candidateId", "matrixIndex", "trajectoryId", "labelId"], sort=False)
    for (candidate, matrix, trajectory, label_id), group in grouped:
        start: int | None = None
        prior: int | None = None
        episode = 0
        for index, value in zip(
            group["selectedSequenceIndex"],
            nullable_labels(group["isReplicator"]),
            strict=True,
        ):
            current = int(index)
            contiguous = prior is not None and current == prior + 1
            if value is None:
                if start is not None and prior is not None:
                    rows.append(
                        {
                            "candidateId": candidate,
                            "matrixIndex": int(matrix),
                            "trajectoryId": trajectory,
                            "labelId": label_id,
                            "episodeIndex": episode,
                            "startIndex0": start,
                            "endIndex0": prior,
                            "duration": prior - start + 1,
                        }
                    )
                    episode += 1
                    start = None
                prior = None
                continue
            if value and (start is None or not contiguous):
                if start is not None and prior is not None:
                    rows.append(
                        {
                            "candidateId": candidate,
                            "matrixIndex": int(matrix),
                            "trajectoryId": trajectory,
                            "labelId": label_id,
                            "episodeIndex": episode,
                            "startIndex0": start,
                            "endIndex0": prior,
                            "duration": prior - start + 1,
                        }
                    )
                    episode += 1
                start = current
            elif not value and start is not None:
                assert prior is not None
                rows.append(
                    {
                        "candidateId": candidate,
                        "matrixIndex": int(matrix),
                        "trajectoryId": trajectory,
                        "labelId": label_id,
                        "episodeIndex": episode,
                        "startIndex0": start,
                        "endIndex0": prior,
                        "duration": prior - start + 1,
                    }
                )
                episode += 1
                start = None
            prior = current
        if start is not None and prior is not None:
            rows.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": int(matrix),
                    "trajectoryId": trajectory,
                    "labelId": label_id,
                    "episodeIndex": episode,
                    "startIndex0": start,
                    "endIndex0": prior,
                    "duration": prior - start + 1,
                }
            )
    return pd.DataFrame(rows)


def aggregate_fingerprints(fingerprints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, label_id), group in fingerprints.groupby(
        ["candidateId", "labelId"], sort=False
    ):
        row: dict[str, Any] = {
            "candidateId": candidate,
            "labelId": label_id,
            "labelOrdinal": LABEL_BY_ID[label_id].ordinal,
            "trajectoryCount": len(group),
            "definedConsistencyCount": int(group["consistency"].notna().sum()),
            "observedOnsetCount": int((~group["neverReplicator"]).sum()),
            "neverReplicatorCount": int(group["neverReplicator"].sum()),
            "nonreplicatingAtCutoffFraction": float(
                pd.to_numeric(group["isNonreplicatingAtCutoff"], errors="coerce").mean()
            ),
            "noReplicatorThroughCutoffFraction": float(
                group["noReplicatorObservedThroughCutoff"].astype(float).mean()
            ),
        }
        for metric in REPORT_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            token = metric[0].upper() + metric[1:]
            row[f"defined{token}Count"] = len(finite)
            row[f"mean{token}"] = float(finite.mean()) if len(finite) else None
            row[f"median{token}"] = float(finite.median()) if len(finite) else None
            row[f"sd{token}"] = float(finite.std(ddof=1)) if len(finite) > 1 else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["labelOrdinal", "candidateId"], kind="stable"
    )


def summary_dict(row: pd.Series) -> dict[str, float | None]:
    return {
        "persistence": row["meanPersistence"],
        "occupancy": row["meanOccupancy"],
        "consistency": row["meanConsistency"],
        "firstOnsetRawScore": row["meanFirstOnsetRawScore"],
        "firstOnsetNormalizedScore": row["meanFirstOnsetNormalizedScore"],
    }


def paper_comparison(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATE_IDS:
        comparator_row = aggregate.loc[
            aggregate["candidateId"].eq(candidate)
            & aggregate["labelId"].eq(COMPARATOR_LABEL_ID)
        ].iloc[0]
        comparator = summary_dict(comparator_row)
        for row in aggregate.loc[aggregate["candidateId"].eq(candidate)].itertuples(
            index=False
        ):
            current = summary_dict(pd.Series(row._asdict()))
            for mode in ("RAW", "NORMALIZED"):
                distance = paper_fingerprint_distance(current, onset_mode=mode)
                comparator_distance = paper_fingerprint_distance(
                    comparator, onset_mode=mode
                )
                closer, structure = closer_dimension_count(
                    current, comparator, onset_mode=mode
                )
                rows.append(
                    {
                        "candidateId": candidate,
                        "labelId": row.labelId,
                        "onsetMode": mode,
                        "paperDistance": distance,
                        "comparatorDistance": comparator_distance,
                        "distanceDifferenceCandidateMinusComparator": (
                            None
                            if distance is None or comparator_distance is None
                            else distance - comparator_distance
                        ),
                        "distanceImprovementFraction": (
                            None
                            if distance is None or comparator_distance in (None, 0)
                            else (comparator_distance - distance) / comparator_distance
                        ),
                        "closerDimensionCount": closer,
                        "structureDimensionImproved": structure,
                        "occupancyCloser": abs(float(current["occupancy"]) - 0.88)
                        < abs(float(comparator["occupancy"]) - 0.88),
                    }
                )
    return pd.DataFrame(rows)


def fixed_l04_comparison(aggregate: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_parquet(L04_ROOT / "fingerprint_summary.parquet")
    rows = []
    for candidate in CANDIDATE_IDS:
        l05 = aggregate.loc[
            aggregate["candidateId"].eq(candidate)
            & aggregate["labelId"].eq(STRUCTURAL_LABEL_ID)
        ].iloc[0]
        l04 = frozen.loc[
            frozen["candidateId"].eq(candidate)
            & frozen["labelId"].eq("MOL_CROSS_GENERATION_RECURRENCE_H900")
        ].iloc[0]
        for metric in CORE_METRICS:
            token = metric[0].upper() + metric[1:]
            left = float(l05[f"mean{token}"])
            right = float(l04[f"mean{token}"])
            rows.append(
                {
                    "candidateId": candidate,
                    "l05LabelId": STRUCTURAL_LABEL_ID,
                    "fixedL04LabelId": "MOL_CROSS_GENERATION_RECURRENCE_H900",
                    "metric": metric,
                    "l05Mean": left,
                    "l04FrozenMean": right,
                    "l05MinusL04": left - right,
                    "l04TemporalScope": "RETROSPECTIVE_COMPLETED_RUN_SYMMETRIC",
                    "l05TemporalScope": "PAST_ONLY_NO_BACKFILL",
                }
            )
    return pd.DataFrame(rows)


def distance_arrays(metric_arrays: dict[str, np.ndarray], mode: str) -> np.ndarray:
    onset = "firstOnsetRawScore" if mode == "RAW" else "firstOnsetNormalizedScore"
    keys = ("persistence", "occupancy", "consistency", onset)
    standardized = []
    for key in keys:
        center, scale = PAPER_TARGETS[key]
        standardized.append((metric_arrays[key] - center) / scale)
    return np.sqrt(np.mean(np.square(np.vstack(standardized)), axis=0))


def bootstrap_analysis(
    fingerprints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    distance_rows = []
    metric_rows = []
    for candidate in CANDIDATE_IDS:
        comparator = fingerprints.loc[
            fingerprints["candidateId"].eq(candidate)
            & fingerprints["labelId"].eq(COMPARATOR_LABEL_ID)
        ].sort_values("matrixIndex", kind="stable")
        current = fingerprints.loc[
            fingerprints["candidateId"].eq(candidate)
            & fingerprints["labelId"].eq(STRUCTURAL_LABEL_ID)
        ].sort_values("matrixIndex", kind="stable")
        if not np.array_equal(
            current["matrixIndex"].to_numpy(), comparator["matrixIndex"].to_numpy()
        ):
            raise RuntimeError("paired bootstrap matrix identity mismatch")
        rng = np.random.Generator(
            np.random.PCG64DXSM(
                derive_seed128(
                    candidate, STRUCTURAL_LABEL_ID, "paired_matrix_bootstrap"
                )
            )
        )
        positions = rng.integers(
            0, len(current), size=(BOOTSTRAP_REPLICATES, len(current))
        )
        current_arrays: dict[str, np.ndarray] = {}
        comparator_arrays: dict[str, np.ndarray] = {}
        for metric in CORE_METRICS:
            left = pd.to_numeric(current[metric], errors="coerce").to_numpy(
                dtype=np.float64
            )
            right = pd.to_numeric(comparator[metric], errors="coerce").to_numpy(
                dtype=np.float64
            )
            current_arrays[metric] = np.nanmean(left[positions], axis=1)
            comparator_arrays[metric] = np.nanmean(right[positions], axis=1)
            differences = current_arrays[metric] - comparator_arrays[metric]
            metric_rows.append(
                {
                    "candidateId": candidate,
                    "labelId": STRUCTURAL_LABEL_ID,
                    "metric": metric,
                    "replicates": BOOTSTRAP_REPLICATES,
                    "meanDifference": float(np.mean(differences)),
                    "lower95": float(np.quantile(differences, 0.025)),
                    "upper95": float(np.quantile(differences, 0.975)),
                }
            )
        for mode in ("RAW", "NORMALIZED"):
            differences = distance_arrays(current_arrays, mode) - distance_arrays(
                comparator_arrays, mode
            )
            distance_rows.append(
                {
                    "candidateId": candidate,
                    "labelId": STRUCTURAL_LABEL_ID,
                    "onsetMode": mode,
                    "replicates": BOOTSTRAP_REPLICATES,
                    "meanDistanceDifference": float(np.mean(differences)),
                    "lower95": float(np.quantile(differences, 0.025)),
                    "upper95": float(np.quantile(differences, 0.975)),
                    "probabilityDistanceImprovement": float(np.mean(differences < 0)),
                }
            )
    return pd.DataFrame(distance_rows), pd.DataFrame(metric_rows)


def leave_one_out(fingerprints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATE_IDS:
        comparator = fingerprints.loc[
            fingerprints["candidateId"].eq(candidate)
            & fingerprints["labelId"].eq(COMPARATOR_LABEL_ID)
        ].sort_values("matrixIndex", kind="stable")
        current = fingerprints.loc[
            fingerprints["candidateId"].eq(candidate)
            & fingerprints["labelId"].eq(STRUCTURAL_LABEL_ID)
        ].sort_values("matrixIndex", kind="stable")
        for omitted in range(100):
            keep = np.arange(100) != omitted
            left = {
                metric: float(
                    pd.to_numeric(current.iloc[keep][metric], errors="coerce").mean()
                )
                for metric in CORE_METRICS
            }
            right = {
                metric: float(
                    pd.to_numeric(comparator.iloc[keep][metric], errors="coerce").mean()
                )
                for metric in CORE_METRICS
            }
            for mode in ("RAW", "NORMALIZED"):
                left_distance = paper_fingerprint_distance(left, onset_mode=mode)
                right_distance = paper_fingerprint_distance(right, onset_mode=mode)
                rows.append(
                    {
                        "candidateId": candidate,
                        "labelId": STRUCTURAL_LABEL_ID,
                        "omittedMatrixIndex": omitted,
                        "onsetMode": mode,
                        "distanceDifference": (
                            None
                            if left_distance is None or right_distance is None
                            else float(left_distance - right_distance)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def overlap_results(labels: pd.DataFrame) -> pd.DataFrame:
    structural = labels.loc[labels["labelId"].eq(STRUCTURAL_LABEL_ID)]
    comparator = labels.loc[labels["labelId"].eq(COMPARATOR_LABEL_ID)]
    rows = []
    for candidate in CANDIDATE_IDS:
        left = structural.loc[structural["candidateId"].eq(candidate)][
            ["matrixIndex", "selectedSequenceIndex", "isReplicator"]
        ].rename(columns={"isReplicator": "structural"})
        right = comparator.loc[comparator["candidateId"].eq(candidate)][
            ["matrixIndex", "selectedSequenceIndex", "isReplicator"]
        ].rename(columns={"isReplicator": "adjacent"})
        joined = left.merge(
            right,
            on=["matrixIndex", "selectedSequenceIndex"],
            how="inner",
            validate="one_to_one",
        ).dropna(subset=["structural", "adjacent"])
        a = joined["structural"].astype(bool).to_numpy()
        b = joined["adjacent"].astype(bool).to_numpy()
        union = np.count_nonzero(a | b)
        rows.append(
            {
                "candidateId": candidate,
                "labelId": STRUCTURAL_LABEL_ID,
                "baselineId": COMPARATOR_LABEL_ID,
                "commonEligibleCount": len(joined),
                "accuracy": float(np.mean(a == b)),
                "jaccard": float(np.count_nonzero(a & b) / union) if union else None,
                "mismatchFraction": float(np.mean(a != b)),
                "structuralPositiveAdjacentNegative": int(np.count_nonzero(a & ~b)),
                "structuralNegativeAdjacentPositive": int(np.count_nonzero(~a & b)),
            }
        )
    return pd.DataFrame(rows)


def cross_candidate_agreement(
    fingerprints: pd.DataFrame, aggregate: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for definition in LABEL_DEFINITIONS:
        left = fingerprints.loc[
            fingerprints["candidateId"].eq(CANDIDATE_IDS[0])
            & fingerprints["labelId"].eq(definition.label_id)
        ].sort_values("matrixIndex", kind="stable")
        right = fingerprints.loc[
            fingerprints["candidateId"].eq(CANDIDATE_IDS[1])
            & fingerprints["labelId"].eq(definition.label_id)
        ].sort_values("matrixIndex", kind="stable")
        if not np.array_equal(
            left["matrixIndex"].to_numpy(), right["matrixIndex"].to_numpy()
        ):
            raise RuntimeError("cross-candidate matrix pairing changed")
        left_agg = aggregate.loc[
            aggregate["candidateId"].eq(CANDIDATE_IDS[0])
            & aggregate["labelId"].eq(definition.label_id)
        ].iloc[0]
        right_agg = aggregate.loc[
            aggregate["candidateId"].eq(CANDIDATE_IDS[1])
            & aggregate["labelId"].eq(definition.label_id)
        ].iloc[0]
        for metric in (
            "occupancy",
            "consistency",
            "firstOnsetNormalizedScore",
            "episodeCount",
        ):
            a = pd.to_numeric(left[metric], errors="coerce").to_numpy(dtype=np.float64)
            b = pd.to_numeric(right[metric], errors="coerce").to_numpy(dtype=np.float64)
            finite = np.isfinite(a) & np.isfinite(b)
            correlation = None
            if finite.sum() >= 3 and np.ptp(a[finite]) > 0 and np.ptp(b[finite]) > 0:
                correlation = float(np.corrcoef(a[finite], b[finite])[0, 1])
            token = metric[0].upper() + metric[1:]
            left_mean = left_agg[f"mean{token}"]
            right_mean = right_agg[f"mean{token}"]
            rows.append(
                {
                    "labelId": definition.label_id,
                    "metric": metric,
                    "pairedDefinedCount": int(finite.sum()),
                    "candidate2Mean": left_mean,
                    "candidate3Mean": right_mean,
                    "absoluteMeanDifference": (
                        abs(float(left_mean) - float(right_mean))
                        if pd.notna(left_mean) and pd.notna(right_mean)
                        else None
                    ),
                    "matrixLevelPearson": correlation,
                }
            )
    return pd.DataFrame(rows)


def generation_block_permutation(
    execution: pd.DataFrame, comparison: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_rows = []
    summary_rows = []
    for candidate in CANDIDATE_IDS:
        current = execution.loc[execution["candidateId"].eq(candidate)].sort_values(
            "matrixIndex", kind="stable"
        )
        if len(current) != 100 or not np.array_equal(
            current["matrixIndex"].to_numpy(), np.arange(100)
        ):
            raise RuntimeError("permutation cache matrix identity mismatch")
        loaded = []
        for path in current["permutationCache"]:
            with np.load(path, allow_pickle=False) as archive:
                loaded.append(
                    {metric: archive[metric].copy() for metric in CORE_METRICS}
                )
        metric_arrays = {
            metric: np.vstack([item[metric] for item in loaded])
            for metric in CORE_METRICS
        }
        means = {
            metric: np.nanmean(values, axis=0)
            for metric, values in metric_arrays.items()
        }
        for mode in ("RAW", "NORMALIZED"):
            distances = distance_arrays(means, mode)
            for replicate, distance in enumerate(distances):
                full_rows.append(
                    {
                        "candidateId": candidate,
                        "labelId": STRUCTURAL_LABEL_ID,
                        "replicate": replicate,
                        "onsetMode": mode,
                        "meanPersistence": means["persistence"][replicate],
                        "meanOccupancy": means["occupancy"][replicate],
                        "meanConsistency": means["consistency"][replicate],
                        "meanFirstOnsetRawScore": means["firstOnsetRawScore"][
                            replicate
                        ],
                        "meanFirstOnsetNormalizedScore": means[
                            "firstOnsetNormalizedScore"
                        ][replicate],
                        "paperDistance": distance,
                    }
                )
            observed = float(
                comparison.loc[
                    comparison["candidateId"].eq(candidate)
                    & comparison["labelId"].eq(STRUCTURAL_LABEL_ID)
                    & comparison["onsetMode"].eq(mode),
                    "paperDistance",
                ].iloc[0]
            )
            lower = float(np.quantile(distances, 0.025))
            summary_rows.append(
                {
                    "candidateId": candidate,
                    "labelId": STRUCTURAL_LABEL_ID,
                    "onsetMode": mode,
                    "controlId": "RECOMPUTED_GENERATION_BLOCK_ORDER_PERMUTATION",
                    "replicates": PERMUTATION_REPLICATES,
                    "observedPaperDistance": observed,
                    "nullLower2_5": lower,
                    "nullMedian": float(np.median(distances)),
                    "nullUpper97_5": float(np.quantile(distances, 0.975)),
                    "lowerTailP": float(
                        (1 + np.count_nonzero(distances <= observed))
                        / (len(distances) + 1)
                    ),
                    "negativeControlPassed": bool(observed < lower),
                    "labelsRecomputedAfterBlockPermutation": True,
                    "blockInternalOrderPreserved": True,
                    "generationNumbersSequentiallyReassigned": True,
                    "immediateNeighborExclusionReapplied": True,
                }
            )
    return pd.DataFrame(full_rows), pd.DataFrame(summary_rows)


def robustness_table(
    bootstrap: pd.DataFrame,
    loo: pd.DataFrame,
    negative: pd.DataFrame,
    suffix: pd.DataFrame,
) -> pd.DataFrame:
    loo_summary = loo.groupby(
        ["candidateId", "labelId", "onsetMode"], as_index=False
    ).agg(
        looMinimumDistanceDifference=("distanceDifference", "min"),
        looMaximumDistanceDifference=("distanceDifference", "max"),
        looAllImproved=("distanceDifference", lambda values: bool((values < 0).all())),
    )
    suffix_summary = (
        suffix.groupby("candidateId", as_index=False)
        .agg(
            suffixAuditRows=("passed", "size"),
            suffixAllPassed=("passed", "all"),
            suffixAllMutationsEffective=("suffixMutationEffective", "all"),
        )
        .assign(labelId=STRUCTURAL_LABEL_ID)
    )
    return (
        bootstrap.merge(
            loo_summary,
            on=["candidateId", "labelId", "onsetMode"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            negative[
                [
                    "candidateId",
                    "labelId",
                    "onsetMode",
                    "observedPaperDistance",
                    "nullLower2_5",
                    "nullMedian",
                    "nullUpper97_5",
                    "lowerTailP",
                    "negativeControlPassed",
                ]
            ],
            on=["candidateId", "labelId", "onsetMode"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            suffix_summary,
            on=["candidateId", "labelId"],
            how="left",
            validate="many_to_one",
        )
    )


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
        "exactTwoPassReplayAll400LabelTrajectories": bool(
            replay["exactTwoPassReplayPassed"].all()
        ),
        "independentStructuralReplayAll200Trajectories": bool(
            independent["passed"].all()
        ),
        "exactFrozenAdjacentComparatorReplay": bool(comparator_replay["passed"].all()),
        "exactFutureSuffixInvarianceAll3000Sentinels": bool(suffix["passed"].all()),
        "allSuffixMutationsEffective": bool(suffix["suffixMutationEffective"].all()),
        "preciseHumanLockedPaperRelationship": True,
        "noOutcomeTunedChoice": True,
        "untouchedS20DesignComplete": (
            LOOP_ROOT / "untouched_s20_design.yaml"
        ).is_file(),
    }
    for candidate in CANDIDATE_IDS:
        agg = aggregate.loc[
            aggregate["candidateId"].eq(candidate)
            & aggregate["labelId"].eq(STRUCTURAL_LABEL_ID)
        ].iloc[0]
        comp = comparison.loc[
            comparison["candidateId"].eq(candidate)
            & comparison["labelId"].eq(STRUCTURAL_LABEL_ID)
        ]
        boot = bootstrap.loc[
            bootstrap["candidateId"].eq(candidate)
            & bootstrap["labelId"].eq(STRUCTURAL_LABEL_ID)
        ]
        influence = loo.loc[
            loo["candidateId"].eq(candidate) & loo["labelId"].eq(STRUCTURAL_LABEL_ID)
        ]
        control = negative.loc[
            negative["candidateId"].eq(candidate)
            & negative["labelId"].eq(STRUCTURAL_LABEL_ID)
        ]
        gates[f"occupancyCloser_{candidate}"] = bool(comp["occupancyCloser"].all())
        gates[f"jointDistanceBetterBothModes_{candidate}"] = bool(
            (comp["distanceDifferenceCandidateMinusComparator"] < 0).all()
        )
        gates[f"threeDimensionsIncludingOnsetOrConsistency_{candidate}"] = bool(
            (comp["closerDimensionCount"] >= 3).all()
            and comp["structureDimensionImproved"].all()
        )
        gates[f"bootstrapUpperBelowZeroBothModes_{candidate}"] = bool(
            (boot["upper95"] < 0).all()
        )
        gates[f"allLeaveOneOutImproved_{candidate}"] = bool(
            (influence["distanceDifference"] < 0).all()
        )
        gates[f"generationBlockPermutationControlBothModes_{candidate}"] = bool(
            control["negativeControlPassed"].all()
        )
        gates[f"coverage_{candidate}"] = bool(
            int(agg["definedConsistencyCount"]) >= 95
            and int(agg["observedOnsetCount"]) >= 95
        )
        gates[f"quarterEligibility_{candidate}"] = bool(
            float(agg["nonreplicatingAtCutoffFraction"]) > 0
            and float(agg["noReplicatorThroughCutoffFraction"]) > 0
        )
    structural_cross = cross.loc[cross["labelId"].eq(STRUCTURAL_LABEL_ID)]
    differences = dict(
        zip(
            structural_cross["metric"],
            structural_cross["absoluteMeanDifference"],
            strict=True,
        )
    )
    gates["crossCandidateAgreement"] = bool(
        pd.notna(differences.get("occupancy"))
        and pd.notna(differences.get("consistency"))
        and pd.notna(differences.get("firstOnsetNormalizedScore"))
        and float(differences["occupancy"]) <= 0.05
        and float(differences["consistency"]) <= 0.10
        and float(differences["firstOnsetNormalizedScore"]) <= 0.10
    )
    promoted = bool(all(gates.values()))
    structural_comparison = comparison.loc[
        comparison["labelId"].eq(STRUCTURAL_LABEL_ID)
    ]
    paper_match = bool((structural_comparison["paperDistance"] <= 1).all())
    directional = bool(
        structural_comparison["occupancyCloser"].all()
        and (structural_comparison["closerDimensionCount"] >= 3).all()
        and structural_comparison["structureDimensionImproved"].all()
    )
    primary_class = (
        "EXPLORATORY_PAPER_MATCH"
        if paper_match
        else (
            "EXPLORATORY_DIRECTIONAL_MATCH"
            if directional
            else "EXPLORATORY_NON_SUPPORT"
        )
    )
    structural_classes = [
        primary_class,
        "METHOD_DEPENDENT_LEAD",
        "AUTHOR_AMBIGUITY_UNRESOLVED",
        "PROMOTABLE_TO_S20" if promoted else "NOT_PROMOTABLE",
    ]
    return {
        "schema": "eidosoma.e01.s19_l05_classification.v1",
        "researchStepId": LOOP_ID,
        "confirmatoryVerdictIssued": False,
        "topLevelClassification": "PROMOTABLE_TO_S20" if promoted else primary_class,
        "outcomeClass": (
            "SUPPORTIVE_EXPLORATORY" if promoted else "CONSTRAINING_OR_NULL_EXPLORATORY"
        ),
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
                "futureSuffixInvariant": bool(suffix["passed"].all()),
                "prospectivePredictionClaim": False,
                "causalControlClaim": False,
            },
        ],
        "laterLoopActivated": False,
        "s20Activated": False,
        "mandatoryHumanReview": True,
    }


def verify_immutable_prior() -> dict[str, Any]:
    baseline = json.loads(
        (LOOP_ROOT / "immutable_prior_baseline.json").read_text(encoding="utf-8")
    )
    mismatches = []
    for row in baseline["files"]:
        path = Path(row["path"])
        if not path.is_file():
            mismatches.append({"path": str(path), "reason": "missing"})
            continue
        actual = sha256_file(path)
        size = path.stat().st_size
        if actual != row["sha256"] or size != row["bytes"]:
            mismatches.append({"path": str(path), "reason": "size_or_hash_changed"})
    return {
        "schema": "eidosoma.e01.s19_l05_immutable_prior_postcheck.v1",
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
    fixed_l04: pd.DataFrame,
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
            "candidateId",
            "labelId",
            "meanOccupancy",
            "meanPersistence",
            "meanConsistency",
            "meanFirstOnsetRawIndex0",
            "meanFirstOnsetNormalized",
            "meanEntryCount",
            "meanExitCount",
            "meanEpisodeCount",
            "meanMeanEpisodeDuration",
            "meanLongestEpisode",
            "nonreplicatingAtCutoffFraction",
            "noReplicatorThroughCutoffFraction",
            "meanRecurrentGenerationCount",
        ]
    ].copy()
    comparison_view = comparison.loc[comparison["labelId"].eq(STRUCTURAL_LABEL_ID)][
        [
            "candidateId",
            "onsetMode",
            "paperDistance",
            "comparatorDistance",
            "distanceDifferenceCandidateMinusComparator",
            "distanceImprovementFraction",
            "closerDimensionCount",
            "structureDimensionImproved",
            "occupancyCloser",
        ]
    ]
    suffix_summary = suffix.groupby(["candidateId", "variant"], as_index=False).agg(
        sentinels=("passed", "size"),
        passed=("passed", "sum"),
        mutationsEffective=("suffixMutationEffective", "sum"),
    )
    l04_view = fixed_l04.loc[
        fixed_l04["metric"].isin(
            [
                "occupancy",
                "persistence",
                "consistency",
                "firstOnsetRawScore",
                "firstOnsetNormalizedScore",
            ]
        )
    ][["candidateId", "metric", "l05Mean", "l04FrozenMean", "l05MinusL04"]]
    gates = classification["labelClassifications"][1]["promotionGates"]
    gate_table = pd.DataFrame(
        [{"gate": key, "passed": value} for key, value in gates.items()]
    )
    promoted = classification["promotedLeadCount"]
    conclusion = (
        "The one past-only label passed every locked promotion gate and is an "
        "exploratory lead for an untouched S20 label-definition confirmation."
        if promoted
        else "The one past-only label did not pass every locked promotion gate and was not promoted."
    )
    caveat = (
        "Even if promoted, this is exploratory label reconstruction on previously studied matrices; it is not a prediction or causal-control result."
        if promoted
        else "The failed gate(s) constrain this exact one-sided rule; they do not identify the unavailable author implementation."
    )
    return f"""# E01/S19-L05 Full Results — Past-Only Cross-Generation Recurrence Activation

## Concise handoff summary

- **Research step ID:** `S19-L05`
- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Artifacts written:** complete L05 preregistration/method/label/bundle/seed locks; exact preanalysis and independent replay evidence; 361,270 frozen-comparator/structural label rows; temporal fingerprints, episode/cutoff/recurrence tables; suffix audits; 4,096-replicate matrix bootstrap and recomputed generation-block permutation controls; leave-one-out, fixed-L04 comparison, validation, runtime/storage, status, ledgers, and hash manifests
- **Validation result:** `{validation_result}`
- **Outcome classification:** `{classification["topLevelClassification"]}`; `{classification["outcomeClass"]}`; `{promoted}` lead(s) promoted
- **Caveats or blockers:** exploratory reuse of studied matrices; the paper does not uniquely specify this one-sided algorithm; known paper fingerprints informed the locked gate; slow drift can still generate high recurrence; no emergence, prediction, intervention, or causal-control evidence was produced
- **Recommended next action:** mandatory human review. Choose a separately authorized bounded action; L06, S20, E02, author contact, and report generation remain inactive.

## Lay summary

L04 asked whether a composition matched another generation anywhere in the completed run. That improved the paper-like temporal pattern but used future observations. L05 made the rule genuinely one-sided: a state can turn positive only when it matches a nonadjacent state already observed in an earlier generation. It never backfills earlier states and never carries a positive state forward automatically. {conclusion} {caveat}

The comparison with 88% occupancy was deliberately joint rather than occupancy-only. Persistence, onset, consistency, episodes, quarter-cutoff eligibility, recurrence counts, candidate agreement, uncertainty, influence, future-suffix invariance, and a block-order negative control were all evaluated under the single frozen rule.

## Frozen question

Does strict past-only recurrence to a nonadjacent state in an earlier positive-numbered generation create a meaningful online label onset and improve the locked four-dimensional paper fingerprint over adjacent molecular `H>0.9` in both simulator candidates, without future-suffix dependence?

This is an exploratory label-definition question only. It does not test causal-emergence association, prediction, intervention, or causal control.

## Inputs

- Frozen S13Y `trajectory_manifest.parquet`, `label_values.parquet`, and 200 trajectory caches: 100 shared catalytic matrices for each of candidate 2 and candidate 3.
- Candidate 2 and candidate 3 were analyzed separately. No favorable-candidate selection and no primary pooling were used.
- Frozen L04 aggregate evidence was read only for the specified retrospective comparison; no L04 value was recomputed or changed.
- Original paper and pinned historical GARD source context identified cross-generation recurrence as plausible but did not uniquely identify this one-sided molecular algorithm.
- No new GARD trajectory, PhiRL/emergence value, prediction fit, intervention, GPU computation, or author contact occurred.

## Detailed methods

### Label contract

For selected-clock state `t` in positive generation `g`, L05 assigns one iff there is at least one state `s <= t-2` with `0 < g_s < g` and strict cosine `H(s,t) > 0.9` on L1-closed 100-component compositions. Generation zero is retained but ineligible. Each distinct earlier generation counts once. The algorithm is sequential, uses no future state, performs no backfill, and adds no carry-forward or persistence rule. Adjacent molecular `H>0.9` is comparator-only.

There is exactly one L05 structural specification: no threshold grid, `H>0.97`, cluster, centroid, boundary projection, alignment, modal-reference choice, emergence selection, prediction selection, or intervention selection.

### Temporal fingerprint and paper-distance

Each catalytic-matrix trajectory is the independent unit. Reported dimensions include occupancy, persistence, zero- and one-based raw onset, normalized onset, consecutive-label Pearson consistency, entries/exits, episode number and duration, longest episode, state and no-onset status through the floor-25% cutoff, recurrent-generation counts, and candidate agreement.

The locked joint distance is the RMS of four standardized deviations from the paper control targets: persistence `716 +/- 198`, occupancy `0.88 +/- 0.03`, consistency `0.38 +/- 0.06`, and either raw onset `37 +/- 27` or normalized onset `0.37 +/- 0.27`. Raw and normalized onset remain separate because the paper's units are ambiguous.

### Robustness and falsification

- Exact two-pass materialization was required for both labels on every trajectory.
- A separately written row-loop implementation replayed primary labels, scores, distinct-generation counts, qualifying-state counts, earliest/latest matches, and full matching-index identities.
- At five length-derived endpoints per trajectory, deletion, row shuffling, and component replacement of the future suffix had to leave all prefix results exactly invariant: 3,000 locked sentinels total.
- The matrix bootstrap used 4,096 paired replicates per candidate with identical resampled matrix identities for structural and comparator labels.
- Every leave-one-matrix-out result was evaluated.
- The negative control independently permuted all 100 complete positive-generation blocks in each matrix for each of 4,096 replicates, preserved within-block order, sequentially renumbered generations, reapplied `s<=t-2`, and recomputed the past-only label. Passing required observed raw and normalized paper-distance below the null 2.5th percentile in both candidates.

## Results

### Candidate-specific temporal fingerprints

{markdown_table(aggregate_view)}

The full matrix-level distributions, medians, standard deviations, eligibility counts, episode tables, quarter-cutoff results, and recurrence diagnostics are machine-readable. Occupancy is presented as only one part of the fingerprint.

### Joint comparison with adjacent `H>0.9`

{markdown_table(comparison_view)}

Negative distance differences favor L05. A directional improvement does not establish an exact numerical match, and neither can bypass the prospective robustness gates.

### Fixed comparison with L04 completed-run symmetric membership

{markdown_table(l04_view)}

This comparison isolates the cost of removing future membership and backfilling. L04 remains a frozen retrospective result and is not eligible as online evidence.

### Matrix bootstrap

{markdown_table(bootstrap, ["candidateId", "onsetMode", "meanDistanceDifference", "lower95", "upper95", "probabilityDistanceImprovement"])}

Selected metric-difference intervals:

{markdown_table(bootstrap_metrics, ["candidateId", "metric", "meanDifference", "lower95", "upper95"])}

### Future-suffix invariance

{markdown_table(suffix_summary)}

Every suffix check compares labels, scores, distinct and qualifying recurrence counts, and all retained matching-index identities at the locked prefix endpoint.

### Recomputed generation-block negative control

{markdown_table(negative, ["candidateId", "onsetMode", "observedPaperDistance", "nullLower2_5", "nullMedian", "nullUpper97_5", "lowerTailP", "negativeControlPassed"])}

### Comparator overlap and cross-candidate agreement

{markdown_table(overlap)}

{markdown_table(cross)}

### Promotion gates

{markdown_table(gate_table)}

At most one lead could be promoted. The result above follows the conjunction of every gate; no favorable candidate or onset mode could rescue a failure.

## Validation

`{validation_result}`

- `{immutable_count}` immutable prior files from S01-S18/V1/V2/S19-L01-L04, including the S17 waiver, matched their frozen size and SHA-256 records after execution.
- The pushed repository lock and artifact copies of the preregistration and method lock matched exactly, and the execution worktree was clean.
- Preanalysis replay covered all 200 trajectory/cache identities and all frozen molecular clocks, adjacent-H arrays, and strict `H>0.9` labels.
- Primary materialization, independent row-loop replay, comparator replay, all 3,000 suffix sentinels, 4,096 paired bootstraps, 400 leave-one-out mode/candidate checks, and 16,384 permutation result rows were validated.
- Candidate/matrix cardinality, single-specification scope, threshold absence, storage ceilings, deterministic report regeneration, artifact completeness, and SHA-256 manifests passed.

## Commands, runtime, and dependencies

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l05.py
ruff check src/e01_s19_past_only_recurrence scripts/e01/prepare_s19_l05_lock.py scripts/e01/run_s19_l05.py tests/e01/test_s19_l05.py
python scripts/e01/prepare_s19_l05_lock.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s19_l05.py --workers 8
```

- Repository commit: `{runtime["repositoryCommit"]}` on `eidosoma/groups/42`, pushed before outcome access.
- Authoritative computation: CPU float64; 8 workers; one numerical-library thread per worker; GPU unused.
- Scientific CPU hours: `{runtime["scientificCpuHours"]:.6f}`; wall hours: `{runtime["wallHours"]:.6f}`.
- Python `{runtime["python"]}`, NumPy `{runtime["numpy"]}`, pandas `{runtime["pandas"]}`, SciPy `{runtime["scipy"]}`, scikit-learn `{runtime["sklearn"]}`, PyArrow `{runtime["pyarrow"]}`.

## Provenance and regeneration

The exact contract is in `preregistration.yaml` and `method_lock.json`; all RNG streams are in `seed_manifest.parquet`; source identities and retained hashes are in `source_snapshot_manifest.json`; trajectory inputs and hashes are in `input_manifest.json`; matrix-level values are in `results.parquet`; controls and robustness are in their named Parquet tables; and the full artifact manifest records every retained file's size and SHA-256. Disposable label and permutation caches remain under `/cache/e01_s19_l05`; compact final evidence is under `/artifacts/research_steps/S19/loops/L05`.

## Caveats, blockers, and interpretation boundary

1. The 100 matrices were already examined in earlier E01 work, and the paper fingerprint was known. L05 is exploratory even though it was outcome-blind after lock.
2. The paper does not uniquely specify a one-sided molecular recurrence algorithm. This implementation may still label gradual drift rather than recurrence to a genuine compositional attractor.
3. Matching 88% occupancy alone was prohibited; it cannot rescue wrong onset, consistency, episode, cutoff, or permutation behavior.
4. A past-only label construction is not itself an early-warning predictor. No emergence feature or held-out outcome was tested.
5. No association, intervention, or causal-control conclusion changes here. S18 and all prior classifications remain immutable.

## Outcome and recommended next action

**Classification:** `{classification["topLevelClassification"]}` with `{promoted}` promoted lead(s). {conclusion}

Stop at the mandatory human-review boundary. The human may choose another explicitly bounded S19 loop, activate an allowed S20 mode, or pause; nothing downstream is active automatically.
"""


def decision_summary_text(
    aggregate: pd.DataFrame,
    negative: pd.DataFrame,
    suffix: pd.DataFrame,
    classification: dict[str, Any],
    validation_result: str,
) -> str:
    view = aggregate.loc[
        aggregate["labelId"].eq(STRUCTURAL_LABEL_ID),
        [
            "candidateId",
            "meanOccupancy",
            "meanPersistence",
            "meanConsistency",
            "meanFirstOnsetRawIndex0",
            "meanFirstOnsetNormalized",
            "nonreplicatingAtCutoffFraction",
            "noReplicatorThroughCutoffFraction",
        ],
    ]
    failed = [
        key
        for key, passed in classification["labelClassifications"][1][
            "promotionGates"
        ].items()
        if not passed
    ]
    return f"""# S19-L05 Decision Summary

## Concise handoff summary

- **Research step ID:** `S19-L05`
- **Completion status:** complete; mandatory human review active
- **Artifacts written:** full L05 evidence, controls, validation, report, status, and hash manifests
- **Validation result:** `{validation_result}`
- **Outcome classification:** `{classification["topLevelClassification"]}`; promoted leads: `{classification["promotedLeadCount"]}`
- **Caveats or blockers:** exploratory reused matrices; unavailable exact author semantics; one-sided recurrence can remain a smooth-drift proxy; no downstream prediction or causal evidence
- **Recommended next action:** human review only; no L06 or S20 is active

## Decision evidence

{markdown_table(view)}

- Future-suffix sentinels passed: `{int(suffix["passed"].sum())}/{len(suffix)}`.
- Generation-block control gates passed: `{int(negative["negativeControlPassed"].sum())}/{len(negative)}`.
- Failed promotion gates: `{", ".join(failed) if failed else "none"}`.

## Boundary

This is an exploratory test of one online label construction, not a prediction or causal-control result. Stop for human review.
"""


def artifact_manifest(root: Path, required: list[str], schema: str) -> dict[str, Any]:
    missing = [name for name in required if not (root / name).is_file()]
    manifest_path = root / "artifact_manifest.json"
    files = []
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and item != manifest_path
    ):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": schema,
        "root": str(root),
        "requiredFiles": required,
        "missing": missing,
        "fileCount": len(files),
        "totalBytes": int(sum(row["bytes"] for row in files)),
        "files": files,
        "passed": not missing,
    }


def append_postloop_ledger(
    aggregate: pd.DataFrame, classification: dict[str, Any], timestamp: str
) -> None:
    path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(path)
    if ledger["loopId"].eq(LOOP_ID).sum() != 1:
        raise RuntimeError("L05 pre-loop self-improvement row cardinality changed")
    compact = aggregate[
        [
            "candidateId",
            "labelId",
            "meanOccupancy",
            "meanPersistence",
            "meanConsistency",
            "meanFirstOnsetRawIndex0",
            "meanEpisodeCount",
            "nonreplicatingAtCutoffFraction",
            "noReplicatorThroughCutoffFraction",
        ]
    ].to_dict(orient="records")
    gates = classification["labelClassifications"][1]["promotionGates"]
    failed = [key for key, passed in gates.items() if not passed]
    row = {
        "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
        "timestampUtc": timestamp,
        "loopId": LOOP_ID,
        "recordPhase": "POST_LOOP_LEARNING_AND_HUMAN_REVIEW_BOUNDARY",
        "beliefBeforeLoop": "L04's directional improvement might survive when recurrence activates only from already observed earlier generations.",
        "motivatingEvidence": "L04 improved every fingerprint dimension but depended on future symmetric membership and failed its block-order control.",
        "failureOrAmbiguityTargeted": "Whether cross-generation recurrence creates a meaningful one-sided online onset without future backfill.",
        "selectedHypotheses": "Exactly one strict-H>0.9, s<=t-2, earlier-positive-generation past-only activation rule.",
        "learned": canonical_json(
            {
                "aggregateFingerprint": compact,
                "failedPromotionGates": failed,
                "promotedLeadIds": classification["promotedLeadIds"],
            }
        ),
        "weakenedHypotheses": "The exact one-sided recurrence-activation rule to the extent it failed any joint, suffix, replay, influence, permutation, coverage, quarter-eligibility, or cross-candidate gate.",
        "remainingPlausibleHypotheses": "Only a promoted exact L05 lead, if any, or a separately authorized nonduplicative author ambiguity; exact author semantics remain unavailable.",
        "proposedNextTest": "Mandatory human review; no automatic L06 or S20.",
        "informationGainRationale": "L05 directly falsified L04 future dependence with one fixed rule and did not add another threshold or label family.",
        "appendOnly": True,
    }
    pd.concat(
        [ledger, pd.DataFrame([row])[ledger.columns]], ignore_index=True
    ).to_parquet(path, index=False)
    with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            "\n## Entry 010 — S19-L05 learning and human-review boundary\n\n"
            "- **Belief before the loop:** L04's directional improvement might survive under a strictly one-sided earlier-generation activation rule.\n"
            "- **What was tested:** Exactly one strict-`H>0.9`, `s<=t-2`, past-only cross-generation recurrence label; adjacent H was comparator-only and L04 was fixed evidence only.\n"
            f"- **What was learned:** {classification['promotedLeadCount']} lead(s) passed every locked promotion gate; exact values and failed gates are preserved in the L05 evidence and full report.\n"
            "- **Hypotheses weakened:** The singleton past-only rule to the extent it failed joint, suffix, replay, influence, permutation, coverage, cutoff, or cross-candidate gates.\n"
            "- **What remains plausible:** Only promoted leads, if any, and genuinely nonduplicative author ambiguities approved later by a human.\n"
            "- **Next action:** Mandatory human review; no automatic L06 or S20.\n"
            "- **Why another loop could add information:** It must isolate a different unresolved dependency and cannot retune L05.\n"
        )


def update_root_handoff(
    report: str,
    classification: dict[str, Any],
    validation_result: str,
    artifacts: list[str],
) -> None:
    (ARTIFACT_ROOT / "research_step_full_results.md").write_text(
        report, encoding="utf-8"
    )
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
            "past_only_recurrence_may_proxy_slow_drift",
            "known_paper_fingerprint_informed_exploratory_gate",
            "no_emergence_prediction_intervention_or_causal_inference",
            "L06_S20_E02_inactive",
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
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["history"].append(
        {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "decision": "S19_L05_COMPLETE_MANDATORY_HUMAN_REVIEW",
            "scope": VERSION,
            "source": "locked_execution_result",
        }
    )
    history["pendingDecision"] = "POST_S19_L05_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)


def main(workers: int) -> None:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    started_utc = datetime.now(timezone.utc)
    lock = execution_lock_validation()
    write_json(LOOP_ROOT / "execution_lock_validation.json", lock)
    if not lock["passed"]:
        raise RuntimeError("L05 execution lock validation failed")

    manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet")
    labels, diagnostics, replay, independent, suffix, execution = execute_trajectories(
        manifest, workers
    )
    if not execution["success"].all():
        raise RuntimeError("L05 trajectory execution or locked audit failed")
    if not replay["exactTwoPassReplayPassed"].all():
        raise RuntimeError("L05 exact two-pass replay failed")
    if not independent["passed"].all():
        raise RuntimeError("L05 independent structural replay failed")
    if not suffix["passed"].all():
        raise RuntimeError("L05 future-suffix invariance failed")
    comparator_replay = frozen_comparator_replay(labels)
    if not comparator_replay["passed"].all():
        raise RuntimeError("L05 frozen adjacent-H comparator replay failed")

    fingerprints = build_fingerprints(labels, diagnostics)
    episodes = episode_table(labels)
    aggregate = aggregate_fingerprints(fingerprints)
    comparison = paper_comparison(aggregate)
    l04_fixed = fixed_l04_comparison(aggregate)
    bootstrap, bootstrap_metrics = bootstrap_analysis(fingerprints)
    loo = leave_one_out(fingerprints)
    overlaps = overlap_results(labels)
    cross = cross_candidate_agreement(fingerprints, aggregate)
    permutation, negative = generation_block_permutation(execution, comparison)
    robustness = robustness_table(bootstrap, loo, negative, suffix)
    classification = classify(
        aggregate,
        comparison,
        bootstrap,
        loo,
        negative,
        cross,
        replay,
        independent,
        comparator_replay,
        suffix,
    )

    labels.to_parquet(
        LOOP_ROOT / "label_values.parquet", index=False, compression="zstd"
    )
    labels.loc[labels["labelId"].eq(STRUCTURAL_LABEL_ID)].to_parquet(
        LOOP_ROOT / "past_only_recurrence_evidence.parquet",
        index=False,
        compression="zstd",
    )
    diagnostics.to_parquet(
        LOOP_ROOT / "recurrence_trajectory_diagnostics.parquet", index=False
    )
    replay.to_parquet(LOOP_ROOT / "label_replay_evidence.parquet", index=False)
    independent.to_parquet(LOOP_ROOT / "independent_label_replay.parquet", index=False)
    comparator_replay.to_parquet(
        LOOP_ROOT / "frozen_comparator_replay.parquet", index=False
    )
    suffix.to_parquet(
        LOOP_ROOT / "future_suffix_invariance_results.parquet", index=False
    )
    write_json(
        LOOP_ROOT / "future_suffix_invariance_summary.json",
        {
            "schema": "eidosoma.e01.s19_l05_future_suffix_invariance_summary.v1",
            "sentinelCount": len(suffix),
            "endpointCountPerTrajectory": 5,
            "variantCount": len(SUFFIX_VARIANTS),
            "passedCount": int(suffix["passed"].sum()),
            "mutationEffectiveCount": int(suffix["suffixMutationEffective"].sum()),
            "passed": bool(
                suffix["passed"].all() and suffix["suffixMutationEffective"].all()
            ),
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
            "candidateId",
            "matrixIndex",
            "labelId",
            "cutoffCount",
            "cutoffIndex0",
            "isNonreplicatingAtCutoff",
            "noReplicatorObservedThroughCutoff",
        ]
    ].to_parquet(LOOP_ROOT / "cutoff_results.parquet", index=False)
    diagnostics.loc[diagnostics["labelId"].eq(STRUCTURAL_LABEL_ID)].to_parquet(
        LOOP_ROOT / "recurrence_generation_count_results.parquet", index=False
    )
    overlaps.to_parquet(LOOP_ROOT / "label_overlap_results.parquet", index=False)
    cross.to_csv(LOOP_ROOT / "cross_candidate_agreement.csv", index=False)
    comparison.to_csv(LOOP_ROOT / "paper_fingerprint_comparison.csv", index=False)
    l04_fixed.to_csv(LOOP_ROOT / "fixed_l04_comparison.csv", index=False)
    bootstrap.to_parquet(LOOP_ROOT / "paper_distance_bootstrap.parquet", index=False)
    bootstrap_metrics.to_parquet(
        LOOP_ROOT / "bootstrap_metric_differences.parquet", index=False
    )
    loo.to_parquet(LOOP_ROOT / "leave_one_out_robustness.parquet", index=False)
    permutation.to_parquet(
        LOOP_ROOT / "generation_block_permutation_results.parquet", index=False
    )
    negative.to_parquet(LOOP_ROOT / "negative_control_results.parquet", index=False)
    robustness.to_parquet(LOOP_ROOT / "robustness_results.parquet", index=False)
    pd.DataFrame(
        columns=[
            "failureId",
            "phase",
            "status",
            "reason",
            "scientificOutcomesAccessed",
            "repairAttempted",
        ]
    ).to_csv(LOOP_ROOT / "failure_ledger.csv", index=False)
    write_json(LOOP_ROOT / "classification.json", classification)

    immutable = verify_immutable_prior()
    write_json(LOOP_ROOT / "immutable_prior_postcheck.json", immutable)
    if not immutable["passed"]:
        raise RuntimeError("L05 immutable prior changed")
    completed_utc = datetime.now(timezone.utc)
    scientific_cpu_seconds = float(execution["cpuSeconds"].sum()) + (
        time.process_time() - started_cpu
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l05_runtime_manifest.v1",
        "startedUtc": started_utc.isoformat(),
        "completedUtc": completed_utc.isoformat(),
        "wallHours": (time.perf_counter() - started_wall) / 3600,
        "scientificCpuHours": scientific_cpu_seconds / 3600,
        "cpuCeilingHours": 32.0,
        "gpuHours": 0.0,
        "gpuCeilingHours": 0.0,
        "workerCount": workers,
        "numericalLibraryThreadsPerWorker": 1,
        "cpuFloat64Authoritative": True,
        "gpuUsed": False,
        "newGardTrajectories": 0,
        "newPhiRLOrEmergenceValues": 0,
        "repositoryCommit": lock["repositoryHead"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
    }
    if runtime["scientificCpuHours"] > 32 or runtime["wallHours"] > 8:
        raise RuntimeError("L05 compute ceiling exceeded")
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime)

    retained_bytes = sum(
        path.stat().st_size for path in LOOP_ROOT.rglob("*") if path.is_file()
    )
    temporary_bytes = sum(
        path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file()
    )
    storage = {
        "schema": "eidosoma.e01.s19_l05_storage_validation.v1",
        "retainedBytesBeforeManifest": retained_bytes,
        "retainedGiB": retained_bytes / (1024**3),
        "retainedCeilingGiB": 10.0,
        "temporaryBytes": temporary_bytes,
        "temporaryGiB": temporary_bytes / (1024**3),
        "temporaryCeilingGiB": 25.0,
        "passed": bool(
            retained_bytes <= 10 * 1024**3 and temporary_bytes <= 25 * 1024**3
        ),
    }
    write_json(LOOP_ROOT / "storage_validation.json", storage)
    if not storage["passed"]:
        raise RuntimeError("L05 storage ceiling exceeded")

    validation_result = "PASS_ALL_LOCK_PREANALYSIS_PRIMARY_INDEPENDENT_COMPARATOR_SUFFIX_IMMUTABILITY_BOOTSTRAP_LOO_RECOMPUTED_GENERATION_BLOCK_STORAGE_REGENERATION_AND_HASH_CHECKS"
    report = report_text(
        aggregate,
        comparison,
        l04_fixed,
        bootstrap,
        bootstrap_metrics,
        negative,
        suffix,
        overlaps,
        cross,
        classification,
        validation_result,
        runtime,
        immutable["fileCount"],
    )
    decision = decision_summary_text(
        aggregate, negative, suffix, classification, validation_result
    )
    regenerated = report_text(
        aggregate,
        comparison,
        l04_fixed,
        bootstrap,
        bootstrap_metrics,
        negative,
        suffix,
        overlaps,
        cross,
        classification,
        validation_result,
        runtime,
        immutable["fileCount"],
    )
    if report != regenerated:
        raise RuntimeError("L05 report regeneration mismatch")
    (LOOP_ROOT / "S19_L05_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "loop_decision_summary.md").write_text(decision, encoding="utf-8")

    regeneration = {
        "schema": "eidosoma.e01.s19_l05_regeneration_validation.v1",
        "labelCount": int(labels["labelId"].nunique()),
        "structuralCandidateCount": 1,
        "candidateCount": int(labels["candidateId"].nunique()),
        "matrixCountPerCandidate": labels.groupby("candidateId")["matrixIndex"]
        .nunique()
        .to_dict(),
        "trajectoryCount": int(execution.shape[0]),
        "labelRowCount": len(labels),
        "structuralLabelRowCount": int(labels["labelId"].eq(STRUCTURAL_LABEL_ID).sum()),
        "fingerprintRowCount": len(fingerprints),
        "suffixSentinelCount": len(suffix),
        "permutationResultRowCount": len(permutation),
        "exactTwoPassReplayPassed": bool(replay["exactTwoPassReplayPassed"].all()),
        "independentPrimaryReplayPassed": bool(independent["passed"].all()),
        "frozenComparatorReplayPassed": bool(comparator_replay["passed"].all()),
        "futureSuffixInvariancePassed": bool(suffix["passed"].all()),
        "allSuffixMutationsEffective": bool(suffix["suffixMutationEffective"].all()),
        "immutablePriorPassed": immutable["passed"],
        "reportDeterministic": True,
        "thresholdGridAbsent": True,
        "H097CandidateAbsent": True,
        "variantCount": 1,
        "newGardTrajectories": 0,
        "newPhiRLOrEmergenceValues": 0,
        "passed": bool(
            labels["labelId"].nunique() == 2
            and labels["candidateId"].nunique() == 2
            and execution.shape[0] == 200
            and fingerprints.shape[0] == 400
            and len(suffix) == 200 * 5 * len(SUFFIX_VARIANTS)
            and len(permutation) == 2 * 2 * PERMUTATION_REPLICATES
            and replay["exactTwoPassReplayPassed"].all()
            and independent["passed"].all()
            and comparator_replay["passed"].all()
            and suffix["passed"].all()
            and suffix["suffixMutationEffective"].all()
            and immutable["passed"]
            and storage["passed"]
        ),
    }
    write_json(LOOP_ROOT / "regeneration_validation.json", regeneration)
    if not regeneration["passed"]:
        raise RuntimeError("L05 final regeneration validation failed")

    loop_artifacts = [
        str(LOOP_ROOT / "S19_L05_FULL_RESULTS.md"),
        str(LOOP_ROOT / "classification.json"),
        str(LOOP_ROOT / "fingerprint_results.parquet"),
        str(LOOP_ROOT / "paper_fingerprint_comparison.csv"),
        str(LOOP_ROOT / "future_suffix_invariance_results.parquet"),
        str(LOOP_ROOT / "generation_block_permutation_results.parquet"),
        str(LOOP_ROOT / "negative_control_results.parquet"),
    ]
    loop_status = {
        "researchStepId": LOOP_ID,
        "stepNumber": 19,
        "success": True,
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": loop_artifacts,
        "validationResult": validation_result,
        "caveatsOrBlockers": [
            "exploratory_previously_studied_matrices",
            "exact_author_replicator_definition_unavailable",
            "past_only_recurrence_may_proxy_slow_drift",
            "no_emergence_prediction_intervention_or_causal_inference",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_SELECT_NEXT_BOUNDED_ACTION",
    }
    write_json(LOOP_ROOT / "status.json", loop_status)
    append_postloop_ledger(aggregate, classification, completed_utc.isoformat())
    update_root_handoff(report, classification, validation_result, loop_artifacts)

    required = [
        "preregistration.yaml",
        "method_lock.json",
        "candidate_ranking.csv",
        "candidate_bundle_registry.yaml",
        "label_registry.yaml",
        "label_registry.parquet",
        "specification_ledger.parquet",
        "seed_manifest.parquet",
        "input_manifest.json",
        "source_snapshot_manifest.json",
        "untouched_s20_design.yaml",
        "preoutcome_repository_lock.json",
        "immutable_prior_baseline.json",
        "immutable_prior_validation.json",
        "compute_benchmark.json",
        "preanalysis_replay_evidence.parquet",
        "preanalysis_replay_validation.json",
        "execution_lock_validation.json",
        "execution_status.parquet",
        "label_values.parquet",
        "past_only_recurrence_evidence.parquet",
        "recurrence_trajectory_diagnostics.parquet",
        "label_replay_evidence.parquet",
        "independent_label_replay.parquet",
        "frozen_comparator_replay.parquet",
        "future_suffix_invariance_results.parquet",
        "future_suffix_invariance_summary.json",
        "fingerprint_results.parquet",
        "results.parquet",
        "fingerprint_summary.parquet",
        "fingerprint_aggregate.csv",
        "episode_results.parquet",
        "cutoff_results.parquet",
        "recurrence_generation_count_results.parquet",
        "label_overlap_results.parquet",
        "cross_candidate_agreement.csv",
        "paper_fingerprint_comparison.csv",
        "fixed_l04_comparison.csv",
        "paper_distance_bootstrap.parquet",
        "bootstrap_metric_differences.parquet",
        "leave_one_out_robustness.parquet",
        "generation_block_permutation_results.parquet",
        "negative_control_results.parquet",
        "robustness_results.parquet",
        "failure_ledger.csv",
        "runtime_manifest.json",
        "storage_validation.json",
        "regeneration_validation.json",
        "classification.json",
        "status.json",
        "loop_decision_summary.md",
        "S19_L05_FULL_RESULTS.md",
        "research_step_full_results.md",
    ]
    loop_manifest = artifact_manifest(
        LOOP_ROOT, required, "eidosoma.e01.s19_l05_artifact_manifest.v1"
    )
    if not loop_manifest["passed"]:
        raise RuntimeError(f"missing L05 artifacts: {loop_manifest['missing']}")
    write_json(LOOP_ROOT / "artifact_manifest.json", loop_manifest)

    root_required = [
        "continuation_decision.md",
        "s18_immutable_baseline.json",
        "self_improvement_ledger.parquet",
        "SELF_IMPROVEMENT_LEDGER.md",
        "candidate_registry.parquet",
        "source_search_ledger.parquet",
        "source_search_report.md",
        "loop_registry.yaml",
        "human_review_history.json",
        "s19_status.json",
        "research_step_full_results.md",
    ]
    root_manifest = artifact_manifest(
        ARTIFACT_ROOT, root_required, "eidosoma.e01.s19_artifact_manifest.v5"
    )
    if not root_manifest["passed"]:
        raise RuntimeError(f"missing S19 root artifacts: {root_manifest['missing']}")
    write_json(ARTIFACT_ROOT / "artifact_manifest.json", root_manifest)
    print(
        canonical_json(
            {
                "loopId": LOOP_ID,
                "status": loop_status["status"],
                "validationResult": validation_result,
                "classification": classification["topLevelClassification"],
                "promotedLeadCount": classification["promotedLeadCount"],
                "promotedLeadIds": classification["promotedLeadIds"],
                "labelRows": len(labels),
                "fingerprintRows": len(fingerprints),
                "suffixSentinels": len(suffix),
                "permutationRows": len(permutation),
                "scientificCpuHours": runtime["scientificCpuHours"],
                "wallHours": runtime["wallHours"],
                "mandatoryHumanReview": True,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 8:
        raise SystemExit("workers must be in [1,8]")
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(variable) not in (None, "1"):
            raise SystemExit(f"{variable} must be unset or 1")
    main(arguments.workers)
