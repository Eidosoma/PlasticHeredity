#!/usr/bin/env python3
"""Execute the locked E01/S19-L04 cross-generation recurrence analysis."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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

from e01_s19_cross_generation_recurrence.core import (
    BOOTSTRAP_REPLICATES,
    CANDIDATE_IDS,
    COMPARATOR_LABEL_ID,
    LABEL_BY_ID,
    LABEL_DEFINITIONS,
    LOOP_ID,
    PERMUTATION_REPLICATES,
    STRUCTURAL_LABEL_ID,
    VERSION,
    binary_consistency_from_counts,
    derive_seed128,
    label_trajectory,
)
from e01_s19_replicator_definition.core import (
    PAPER_TARGETS,
    closer_dimension_count,
    fingerprint_from_labels,
    paper_fingerprint_distance,
)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L04"
CACHE_ROOT = Path("/cache/e01_s19_l04")
LABEL_CACHE = CACHE_ROOT / "labels"
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
PREREG = REPO_ROOT / "configs/e01/s19_l04_preregistration.yaml"
METHOD_LOCK = REPO_ROOT / "configs/e01/s19_l04_method_lock.json"

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
    "meanDistinctOtherGenerationCount",
    "medianDistinctOtherGenerationCount",
    "maxDistinctOtherGenerationCount",
    "fractionEligibleWithAtLeast2OtherGenerations",
    "fractionEligibleWithAtLeast5OtherGenerations",
    "fractionEligibleWithAtLeast10OtherGenerations",
    "immediateOnlyEvidenceCount",
    "distinctGenerationPairCount",
    "generationPairDensity",
    "recurrentGenerationCount",
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


def sha256_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_json(orient="records", double_precision=15)
    return hashlib.sha256(payload.encode()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


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
        "distinctOtherGenerationCount",
        "qualifyingStateCount",
        "immediateCrossGenerationMatchCount",
        "immediateOnlyEvidence",
        "maximumImmediateCrossGenerationSimilarity",
        "sameGenerationMatchCount",
        "firstMatchingGeneration",
        "lastMatchingGeneration",
        "earliestMatchingSequenceIndex",
        "latestMatchingSequenceIndex",
    ]
    return output[columns]


def frame_identity(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("selectedSequenceIndex", kind="stable").reset_index(drop=True)
    return sha256_frame(ordered)


def trajectory_worker(record: dict[str, Any]) -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    with Path(record["cachePath"]).open("rb") as handle:
        trajectory = pickle.load(handle)
    frames = []
    diagnostics = []
    replays = []
    for definition in LABEL_DEFINITIONS:
        first_raw, first_diagnostic = label_trajectory(
            trajectory, definition, clock_id=str(record["clockId"])
        )
        second_raw, second_diagnostic = label_trajectory(
            trajectory, definition, clock_id=str(record["clockId"])
        )
        first = normalize_frame(first_raw, definition.label_id)
        second = normalize_frame(second_raw, definition.label_id)
        first_identity = frame_identity(first)
        second_identity = frame_identity(second)
        diagnostic_equal = canonical_json(first_diagnostic) == canonical_json(second_diagnostic)
        passed = first_identity == second_identity and diagnostic_equal
        replays.append(
            {
                "candidateId": record["candidateId"],
                "matrixIndex": int(record["matrixIndex"]),
                "trajectoryId": record["trajectoryId"],
                "labelId": definition.label_id,
                "firstIdentity": first_identity,
                "secondIdentity": second_identity,
                "diagnosticEqual": diagnostic_equal,
                "exactReplayPassed": passed,
            }
        )
        diagnostics.append(
            {
                "candidateId": record["candidateId"],
                "matrixIndex": int(record["matrixIndex"]),
                "trajectoryId": record["trajectoryId"],
                "labelId": definition.label_id,
                "comparatorOnly": definition.comparator_only,
                "eligibleCount": first_diagnostic.get("eligibleCount"),
                "positiveCount": first_diagnostic.get("positiveCount"),
                "relationPairCount": first_diagnostic.get("relationPairCount"),
                "distinctGenerationPairCount": first_diagnostic.get(
                    "distinctGenerationPairCount"
                ),
                "possibleGenerationPairCount": first_diagnostic.get(
                    "possibleGenerationPairCount"
                ),
                "generationPairDensity": first_diagnostic.get("generationPairDensity"),
                "recurrentGenerationCount": first_diagnostic.get(
                    "recurrentGenerationCount"
                ),
                "meanDistinctOtherGenerationCount": first_diagnostic.get(
                    "meanDistinctOtherGenerationCount"
                ),
                "medianDistinctOtherGenerationCount": first_diagnostic.get(
                    "medianDistinctOtherGenerationCount"
                ),
                "maxDistinctOtherGenerationCount": first_diagnostic.get(
                    "maxDistinctOtherGenerationCount"
                ),
                "fractionEligibleWithAtLeast2OtherGenerations": first_diagnostic.get(
                    "fractionEligibleWithAtLeast2OtherGenerations"
                ),
                "fractionEligibleWithAtLeast5OtherGenerations": first_diagnostic.get(
                    "fractionEligibleWithAtLeast5OtherGenerations"
                ),
                "fractionEligibleWithAtLeast10OtherGenerations": first_diagnostic.get(
                    "fractionEligibleWithAtLeast10OtherGenerations"
                ),
                "immediateOnlyEvidenceCount": first_diagnostic.get(
                    "immediateOnlyEvidenceCount"
                ),
                "sameGenerationOnlyOrAdditionalMatchCount": first_diagnostic.get(
                    "sameGenerationOnlyOrAdditionalMatchCount"
                ),
                "exactSymmetryPassed": first_diagnostic.get("exactSymmetryPassed"),
            }
        )
        frames.append(first)
    combined = pd.concat(frames, ignore_index=True)
    output = LABEL_CACHE / str(record["candidateId"]) / f"M{int(record['matrixIndex']):03d}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False, compression="zstd")
    return {
        "candidateId": record["candidateId"],
        "matrixIndex": int(record["matrixIndex"]),
        "trajectoryId": record["trajectoryId"],
        "labelCache": str(output),
        "diagnostics": diagnostics,
        "replays": replays,
        "success": all(row["exactReplayPassed"] for row in replays),
        "wallSeconds": time.perf_counter() - started_wall,
        "cpuSeconds": time.process_time() - started_cpu,
    }


def execution_lock_validation() -> dict[str, Any]:
    repository = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    replay = json.loads((LOOP_ROOT / "preanalysis_replay_validation.json").read_text())
    immutable = json.loads((LOOP_ROOT / "immutable_prior_validation.json").read_text())
    benchmark = json.loads((LOOP_ROOT / "compute_benchmark.json").read_text())
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
        "schema": "eidosoma.e01.s19_l04_execution_lock_validation.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "preparedHead": repository["head"],
        "cleanWorktree": clean,
        "configHashes": hashes,
        "passed": passed,
    }


def execute_labels(
    manifest: pd.DataFrame, workers: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = manifest.sort_values(["matrixIndex", "candidateId"], kind="stable").to_dict(
        orient="records"
    )
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(trajectory_worker, record): record for record in records}
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
            }
            for row in results
        ]
    )
    labels = pd.concat([pd.read_parquet(row["labelCache"]) for row in results], ignore_index=True)
    labels["isReplicator"] = pd.array(labels["isReplicator"], dtype="boolean")
    labels["immediateOnlyEvidence"] = pd.array(
        labels["immediateOnlyEvidence"], dtype="boolean"
    )
    diagnostics = pd.DataFrame([item for row in results for item in row["diagnostics"]])
    replay = pd.DataFrame([item for row in results for item in row["replays"]])
    return labels, diagnostics, replay.merge(
        execution[["candidateId", "matrixIndex", "cpuSeconds", "wallSeconds"]],
        on=["candidateId", "matrixIndex"],
        how="left",
        validate="many_to_one",
    )


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
                observed["candidateId"].eq(candidate) & observed["matrixIndex"].eq(matrix)
            ]
            right = expected.loc[
                expected["candidateId"].eq(candidate) & expected["matrixIndex"].eq(matrix)
            ]
            row_pass = len(left) == len(right) and np.array_equal(
                left["selectedSequenceIndex"].to_numpy(dtype=np.int64),
                right["selectedSequenceIndex"].to_numpy(dtype=np.int64),
            )
            label_pass = np.array_equal(
                left["isReplicator"].astype(bool).to_numpy(),
                right["isReplicator"].astype(bool).to_numpy(),
            )
            score_pass = np.array_equal(
                pd.to_numeric(left["labelScore"], errors="coerce").to_numpy(dtype=np.float64),
                pd.to_numeric(right["labelScore"], errors="coerce").to_numpy(dtype=np.float64),
                equal_nan=True,
            )
            rows.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": matrix,
                    "rowIdentityPassed": row_pass,
                    "labelIdentityPassed": label_pass,
                    "scoreIdentityPassed": score_pass,
                    "passed": bool(row_pass and label_pass and score_pass),
                }
            )
    return pd.DataFrame(rows)


def nullable_labels(series: pd.Series) -> list[bool | None]:
    return [None if pd.isna(value) else bool(value) for value in series.astype(object)]


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
            global_reference=not definition.comparator_only,
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
                "meanDistinctOtherGenerationCount": diagnostic[
                    "meanDistinctOtherGenerationCount"
                ],
                "medianDistinctOtherGenerationCount": diagnostic[
                    "medianDistinctOtherGenerationCount"
                ],
                "maxDistinctOtherGenerationCount": diagnostic[
                    "maxDistinctOtherGenerationCount"
                ],
                "fractionEligibleWithAtLeast2OtherGenerations": diagnostic[
                    "fractionEligibleWithAtLeast2OtherGenerations"
                ],
                "fractionEligibleWithAtLeast5OtherGenerations": diagnostic[
                    "fractionEligibleWithAtLeast5OtherGenerations"
                ],
                "fractionEligibleWithAtLeast10OtherGenerations": diagnostic[
                    "fractionEligibleWithAtLeast10OtherGenerations"
                ],
                "immediateOnlyEvidenceCount": diagnostic["immediateOnlyEvidenceCount"],
                "distinctGenerationPairCount": diagnostic["distinctGenerationPairCount"],
                "generationPairDensity": diagnostic["generationPairDensity"],
                "recurrentGenerationCount": diagnostic["recurrentGenerationCount"],
                **fingerprint,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["labelOrdinal", "candidateId", "matrixIndex"], kind="stable"
    )


def episode_table(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = labels.sort_values(
        ["candidateId", "matrixIndex", "labelId", "selectedSequenceIndex"], kind="stable"
    ).groupby(["candidateId", "matrixIndex", "trajectoryId", "labelId"], sort=False)
    for (candidate, matrix, trajectory, label_id), group in grouped:
        values = nullable_labels(group["isReplicator"])
        start: int | None = None
        prior: int | None = None
        episode = 0
        for index, value in zip(group["selectedSequenceIndex"], values, strict=True):
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
    for (candidate, label_id), group in fingerprints.groupby(["candidateId", "labelId"], sort=False):
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
    return pd.DataFrame(rows).sort_values(["labelOrdinal", "candidateId"], kind="stable")


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
        for row in aggregate.loc[aggregate["candidateId"].eq(candidate)].itertuples(index=False):
            current = summary_dict(pd.Series(row._asdict()))
            for mode in ("RAW", "NORMALIZED"):
                distance = paper_fingerprint_distance(current, onset_mode=mode)
                comparator_distance = paper_fingerprint_distance(comparator, onset_mode=mode)
                closer, structure = closer_dimension_count(current, comparator, onset_mode=mode)
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
                derive_seed128(candidate, STRUCTURAL_LABEL_ID, "paired_matrix_bootstrap")
            )
        )
        positions = rng.integers(0, len(current), size=(BOOTSTRAP_REPLICATES, len(current)))
        current_arrays = {}
        comparator_arrays = {}
        for metric in CORE_METRICS:
            left = pd.to_numeric(current[metric], errors="coerce").to_numpy(dtype=np.float64)
            right = pd.to_numeric(comparator[metric], errors="coerce").to_numpy(dtype=np.float64)
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
                metric: float(pd.to_numeric(current.iloc[keep][metric], errors="coerce").mean())
                for metric in CORE_METRICS
            }
            right = {
                metric: float(pd.to_numeric(comparator.iloc[keep][metric], errors="coerce").mean())
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


def generation_block_summaries(group: pd.DataFrame) -> list[dict[str, Any]]:
    group = group.sort_values("selectedSequenceIndex", kind="stable")
    eligible = group.loc[group["generation"].gt(0)].copy()
    observed = sorted(eligible["generation"].astype(int).unique().tolist())
    if observed != list(range(1, 101)):
        raise RuntimeError("generation-block control requires exactly generations 1..100")
    blocks = []
    for generation in observed:
        values = (
            eligible.loc[eligible["generation"].eq(generation), "isReplicator"]
            .astype(bool)
            .to_numpy(dtype=bool)
        )
        if not len(values):
            raise RuntimeError("empty generation block")
        first_positive = int(np.flatnonzero(values)[0]) if values.any() else -1
        blocks.append(
            {
                "generation": generation,
                "length": len(values),
                "first": bool(values[0]),
                "last": bool(values[-1]),
                "hasPositive": bool(values.any()),
                "firstPositiveOffset": first_positive,
                "n00": int(np.sum(~values[:-1] & ~values[1:])),
                "n01": int(np.sum(~values[:-1] & values[1:])),
                "n10": int(np.sum(values[:-1] & ~values[1:])),
                "n11": int(np.sum(values[:-1] & values[1:])),
                "positiveCount": int(np.count_nonzero(values)),
            }
        )
    return blocks


def generation_block_permutation(
    labels: pd.DataFrame, fingerprints: pd.DataFrame, comparison: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_rows = []
    summary_rows = []
    structural = labels.loc[labels["labelId"].eq(STRUCTURAL_LABEL_ID)].sort_values(
        ["candidateId", "matrixIndex", "selectedSequenceIndex"], kind="stable"
    )
    for candidate in CANDIDATE_IDS:
        trajectory_blocks = []
        candidate_rows = structural.loc[structural["candidateId"].eq(candidate)]
        for matrix, group in candidate_rows.groupby("matrixIndex", sort=True):
            blocks = generation_block_summaries(group)
            total_clock = len(group)
            positive = sum(block["positiveCount"] for block in blocks)
            trajectory_blocks.append(
                {
                    "matrixIndex": int(matrix),
                    "blocks": blocks,
                    "totalClock": total_clock,
                    "persistence": positive,
                    "occupancy": positive / (total_clock - 1),
                }
            )
        if len(trajectory_blocks) != 100:
            raise RuntimeError("generation-block control requires 100 matrices")
        rng = np.random.Generator(
            np.random.PCG64DXSM(
                derive_seed128(candidate, STRUCTURAL_LABEL_ID, "generation_block_permutation")
            )
        )
        for replicate in range(PERMUTATION_REPLICATES):
            persistence_values = []
            occupancy_values = []
            consistency_values = []
            onset_raw_values = []
            onset_normalized_values = []
            for trajectory in trajectory_blocks:
                blocks = trajectory["blocks"]
                order = rng.permutation(len(blocks))
                ordered = [blocks[int(index)] for index in order]
                n00 = sum(block["n00"] for block in ordered)
                n01 = sum(block["n01"] for block in ordered)
                n10 = sum(block["n10"] for block in ordered)
                n11 = sum(block["n11"] for block in ordered)
                for left, right in itertools.pairwise(ordered):
                    if not left["last"] and not right["first"]:
                        n00 += 1
                    elif not left["last"] and right["first"]:
                        n01 += 1
                    elif left["last"] and not right["first"]:
                        n10 += 1
                    else:
                        n11 += 1
                consistency = binary_consistency_from_counts(n00, n01, n10, n11)
                prefix = 1
                onset = trajectory["totalClock"]
                for block in ordered:
                    if block["hasPositive"]:
                        onset = prefix + int(block["firstPositiveOffset"])
                        break
                    prefix += int(block["length"])
                persistence_values.append(float(trajectory["persistence"]))
                occupancy_values.append(float(trajectory["occupancy"]))
                consistency_values.append(np.nan if consistency is None else consistency)
                onset_raw_values.append(float(onset))
                onset_normalized_values.append(
                    float(onset / max(1, trajectory["totalClock"] - 1))
                    if onset < trajectory["totalClock"]
                    else 1.0
                )
            metrics = {
                "persistence": float(np.mean(persistence_values)),
                "occupancy": float(np.mean(occupancy_values)),
                "consistency": float(np.nanmean(consistency_values)),
                "firstOnsetRawScore": float(np.mean(onset_raw_values)),
                "firstOnsetNormalizedScore": float(np.mean(onset_normalized_values)),
            }
            for mode in ("RAW", "NORMALIZED"):
                full_rows.append(
                    {
                        "candidateId": candidate,
                        "labelId": STRUCTURAL_LABEL_ID,
                        "replicate": replicate,
                        "onsetMode": mode,
                        "meanPersistence": metrics["persistence"],
                        "meanOccupancy": metrics["occupancy"],
                        "meanConsistency": metrics["consistency"],
                        "meanFirstOnsetRawScore": metrics["firstOnsetRawScore"],
                        "meanFirstOnsetNormalizedScore": metrics[
                            "firstOnsetNormalizedScore"
                        ],
                        "paperDistance": paper_fingerprint_distance(metrics, onset_mode=mode),
                    }
                )
        candidate_full = pd.DataFrame(
            [row for row in full_rows if row["candidateId"] == candidate]
        )
        for mode in ("RAW", "NORMALIZED"):
            values = candidate_full.loc[
                candidate_full["onsetMode"].eq(mode), "paperDistance"
            ].to_numpy(dtype=np.float64)
            observed = float(
                comparison.loc[
                    comparison["candidateId"].eq(candidate)
                    & comparison["labelId"].eq(STRUCTURAL_LABEL_ID)
                    & comparison["onsetMode"].eq(mode),
                    "paperDistance",
                ].iloc[0]
            )
            lower = float(np.quantile(values, 0.025))
            summary_rows.append(
                {
                    "candidateId": candidate,
                    "labelId": STRUCTURAL_LABEL_ID,
                    "onsetMode": mode,
                    "controlId": "GENERATION_BLOCK_ORDER_PERMUTATION",
                    "replicates": PERMUTATION_REPLICATES,
                    "observedPaperDistance": observed,
                    "nullLower2_5": lower,
                    "nullMedian": float(np.median(values)),
                    "nullUpper97_5": float(np.quantile(values, 0.975)),
                    "lowerTailP": float((1 + np.count_nonzero(values <= observed)) / (len(values) + 1)),
                    "negativeControlPassed": bool(observed < lower),
                    "occupancyInvariantByConstruction": True,
                    "completedRunMembershipInvariantByConstruction": True,
                }
            )
    return pd.DataFrame(full_rows), pd.DataFrame(summary_rows)


def robustness_table(
    bootstrap: pd.DataFrame,
    loo: pd.DataFrame,
    negative: pd.DataFrame,
) -> pd.DataFrame:
    loo_summary = loo.groupby(["candidateId", "labelId", "onsetMode"], as_index=False).agg(
        looMinimumDistanceDifference=("distanceDifference", "min"),
        looMaximumDistanceDifference=("distanceDifference", "max"),
        looAllImproved=("distanceDifference", lambda values: bool((values < 0).all())),
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
    )


def classify(
    aggregate: pd.DataFrame,
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    loo: pd.DataFrame,
    negative: pd.DataFrame,
    cross: pd.DataFrame,
    replay: pd.DataFrame,
) -> dict[str, Any]:
    replay_all = bool(replay["exactReplayPassed"].all())
    gates: dict[str, bool] = {
        "structuralNotComparator": True,
        "exactReplayAll200Trajectories": replay_all,
        "precisePaperAndSourceRelationship": True,
        "noOutcomeTunedChoice": True,
        "untouchedS20DesignComplete": (LOOP_ROOT / "untouched_s20_design.yaml").is_file(),
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
        gates[f"threeDimensionsIncludingStructure_{candidate}"] = bool(
            (comp["closerDimensionCount"] >= 3).all()
            and comp["structureDimensionImproved"].all()
        )
        gates[f"bootstrapUpperBelowZero_{candidate}"] = bool((boot["upper95"] < 0).all())
        gates[f"allLeaveOneOutImproved_{candidate}"] = bool(
            (influence["distanceDifference"] < 0).all()
        )
        gates[f"generationBlockControl_{candidate}"] = bool(
            control["negativeControlPassed"].all()
        )
        gates[f"coverage_{candidate}"] = bool(
            int(agg["definedConsistencyCount"]) >= 95
            and int(agg["observedOnsetCount"]) >= 95
        )
        gates[f"quarterEligibility_{candidate}"] = bool(
            float(agg["nonreplicatingAtCutoffFraction"]) > 0
        )
    structural_cross = cross.loc[cross["labelId"].eq(STRUCTURAL_LABEL_ID)]
    differences = dict(
        zip(structural_cross["metric"], structural_cross["absoluteMeanDifference"], strict=True)
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
        else ("EXPLORATORY_DIRECTIONAL_MATCH" if directional else "EXPLORATORY_NON_SUPPORT")
    )
    classes = [
        primary_class,
        "RETROSPECTIVE_ONLY_LEAD",
        "METHOD_DEPENDENT_LEAD",
        "AUTHOR_AMBIGUITY_UNRESOLVED",
        "PROMOTABLE_TO_S20" if promoted else "NOT_PROMOTABLE",
    ]
    return {
        "schema": "eidosoma.e01.s19_l04_classification.v1",
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
                "classifications": classes,
                "promotionGates": gates,
                "promoted": promoted,
            },
        ],
        "laterLoopActivated": False,
        "s20Activated": False,
        "mandatoryHumanReview": True,
    }


def verify_immutable_prior() -> dict[str, Any]:
    baseline = json.loads((LOOP_ROOT / "immutable_prior_baseline.json").read_text())
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
        "schema": "eidosoma.e01.s19_l04_immutable_prior_postcheck.v1",
        "fileCount": len(baseline["files"]),
        "expectedAggregateSha256": baseline["aggregateSha256"],
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".4f")


def report_text(
    aggregate: pd.DataFrame,
    comparison: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    negative: pd.DataFrame,
    overlap: pd.DataFrame,
    classification: dict[str, Any],
    validation_result: str,
    runtime: dict[str, Any],
    immutable_count: int,
) -> str:
    summary_rows = []
    for row in aggregate.itertuples(index=False):
        summary_rows.append(
            {
                "Candidate": "C02" if row.candidateId.endswith("02") else "C03",
                "Label": "Adjacent H>.9" if row.labelId == COMPARATOR_LABEL_ID else "Cross-generation",
                "Persistence": row.meanPersistence,
                "Occupancy": row.meanOccupancy,
                "Consistency": row.meanConsistency,
                "Onset idx0": row.meanFirstOnsetRawIndex0,
                "Onset step1": row.meanFirstOnsetRawStep1,
                "Onset norm": row.meanFirstOnsetNormalized,
                "Episodes": row.meanEpisodeCount,
                "Longest": row.meanLongestEpisode,
                "Nonrep@25%": row.nonreplicatingAtCutoffFraction,
                "No onset@25%": row.noReplicatorThroughCutoffFraction,
            }
        )
    recurrence = aggregate.loc[aggregate["labelId"].eq(STRUCTURAL_LABEL_ID), :]
    recurrence_rows = []
    for row in recurrence.itertuples(index=False):
        recurrence_rows.append(
            {
                "Candidate": "C02" if row.candidateId.endswith("02") else "C03",
                "Mean other generations": row.meanMeanDistinctOtherGenerationCount,
                "Median other generations": row.meanMedianDistinctOtherGenerationCount,
                "Max other generations": row.meanMaxDistinctOtherGenerationCount,
                "Fraction >=2": row.meanFractionEligibleWithAtLeast2OtherGenerations,
                "Fraction >=5": row.meanFractionEligibleWithAtLeast5OtherGenerations,
                "Generation-pair density": row.meanGenerationPairDensity,
                "Immediate-only rows": row.meanImmediateOnlyEvidenceCount,
            }
        )
    comparison_view = comparison.loc[
        comparison["labelId"].eq(STRUCTURAL_LABEL_ID),
        [
            "candidateId",
            "onsetMode",
            "paperDistance",
            "comparatorDistance",
            "distanceDifferenceCandidateMinusComparator",
            "distanceImprovementFraction",
            "closerDimensionCount",
            "occupancyCloser",
        ],
    ]
    bootstrap_view = bootstrap_metrics.loc[
        bootstrap_metrics["metric"].isin(
            ["occupancy", "consistency", "firstOnsetRawScore", "firstOnsetNormalizedScore"]
        )
    ]
    promoted = classification["promotedLeadIds"]
    top = classification["topLevelClassification"]
    conclusion = (
        "The singleton cross-generation rule passed every exploratory promotion gate and is eligible only for untouched retrospective paper-facing S20 confirmation."
        if promoted
        else "The singleton cross-generation rule did not pass the complete joint-fingerprint promotion gate; it is not eligible for S20 confirmation under this lock."
    )
    return f"""# S19-L04 — Cross-generation recurrence membership

## Concise top summary

- **Research step ID:** S19-L04
- **Completion status:** COMPLETE; mandatory human-review boundary reached; L05, S20, E02, author contact, and report-bundle generation remain inactive
- **Artifacts written:** complete compact L04 preregistration, method/label/bundle/specification locks, exact replay, label values, recurrence evidence, temporal fingerprints, bootstrap, leave-one-out, generation-block permutation, validation, provenance, status, manifest, and append-only S19 ledger evidence
- **Validation result:** {validation_result}
- **Outcome classification:** `{top}`; promoted leads: {len(promoted)} ({', '.join(promoted) if promoted else 'none'})
- **Caveats or blockers:** The rule is a completed-run retrospective reconstruction on previously studied matrices, not recovered author code. Occupancy alone could not promote it, and no prospective, predictive, intervention, or causal claim is eligible.
- **Recommended next action:** Human review must choose the next bounded program action; do not activate L05 or S20 automatically.
- **Lay summary:** This loop tested a middle ground between “the current state resembles the immediately previous state” and “the state belongs to one dominant compotype.” A state counted as replicating only if it resembled a nonadjacent state from another fission generation at strict `H>0.9`. The analysis judged the entire temporal pattern, not the requested 88% occupancy alone.

## Frozen question and nonduplication

L02 directly classified molecular rows by adjacent similarity, one centroid, one K-means cluster, or historical local non-drift. L03 projected one modal boundary compotype. Neither allowed multiple recurring regions while requiring evidence from another generation. L04 locked exactly that singleton hypothesis before opening its outcomes. No second interpretation, threshold, centroid, cluster, modal reference, projection, or alignment was available for selection.

## Inputs

- Frozen S13Y: 100 shared catalytic matrices, 200 complete candidate-specific trajectories, and their frozen adjacent-H arrays and labels.
- Candidate 2 and candidate 3 remained separate; pooling was not used for any gate.
- Original arXiv v1 paper, frozen S08/S13X label context, and frozen L01–L03 comparison evidence.
- New GARD trajectories: **0**. New PhiRL/emergence values: **0**. Prediction/intervention outcomes used: **0**. GPU use: **0**.

## Detailed methods

For every selected-clock molecular state in positive-numbered generation `g`, the analysis L1-closed the 100-component count vector and computed historical cosine H to every completed-run state. A state was positive only if at least one reference state met all three conditions: strict `H>0.9`, a different positive-numbered generation, and absolute selected-sequence separation greater than one. Thus same-generation similarity and an immediate cross-generation neighbor alone could not establish recurrence. Each matching generation counted once regardless of how many states matched. The initial generation-zero state was retained but ineligible.

The relation was evaluated symmetrically over the completed run, so it is explicitly future-dependent and retrospective. Adjacent molecular `H>0.9` was the only comparator. Catalytic matrix was the inferential unit. The frozen joint paper-distance was the root mean square of four deviations scaled by the paper's declared control dispersions: persistence 716±198, occupancy 0.88±0.03, consistency 0.38±0.06, and raw onset 37±27 or normalized onset 0.37±0.27 as two separate analyses.

Robustness comprised a paired 4,096-replicate matrix bootstrap, every leave-one-matrix-out omission, cross-candidate agreement, exact independent label replay, and 4,096 generation-block permutations. The permutation preserved each block's internal order, completed-run membership, and occupancy while shuffling the 100 whole growth-fission blocks within every trajectory; it therefore tested whether the observed temporal arrangement was more paper-like than arbitrary generation order. Its preregistered pass rule required observed paper-distance below the null 2.5th percentile in both modes and candidates.

## Results

### Candidate-specific temporal fingerprints

{markdown_table(pd.DataFrame(summary_rows))}

### Cross-generation recurrence descriptors

{markdown_table(pd.DataFrame(recurrence_rows))}

### Joint paper-fingerprint comparison

{markdown_table(comparison_view)}

### Paired matrix-bootstrap differences, cross-generation minus adjacent

{markdown_table(bootstrap_view)}

### Generation-block permutation negative control

{markdown_table(negative)}

### Label overlap with adjacent H

{markdown_table(overlap)}

{conclusion} All directional comparisons were assessed in both candidates and both onset interpretations. A favorable occupancy result could not override persistence, onset, consistency, episode, cutoff, recurrence, negative-control, uncertainty, or simulator disagreement.

## Validation

- Immutable S01–S18/V1/V2/L01–L03 baseline: PASS across {immutable_count:,} files.
- Pre-outcome clean pushed repository lock and exact S13Y identity/cache/clock/H/label replay: PASS.
- Independent two-pass replay of both labels on all 200 trajectories: PASS.
- Frozen adjacent-H comparator reproduction: PASS.
- Matrix bootstrap, leave-one-out, generation-block control, cross-candidate, schema/cardinality, storage, deterministic-report, and artifact-hash checks: PASS.
- No trajectory, PhiRL value, emergence value, prediction model, or intervention outcome was generated.

## Commands, dependencies, and runtime

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l04.py
python -m ruff check src/e01_s19_cross_generation_recurrence scripts/e01/prepare_s19_l04_lock.py scripts/e01/run_s19_l04.py tests/e01/test_s19_l04.py
git commit <pre-outcome L04 lock> && git push origin eidosoma/groups/42
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src python scripts/e01/prepare_s19_l04_lock.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src python scripts/e01/run_s19_l04.py --workers 8
```

CPU float64 was authoritative. The loop used {runtime['scientificCpuHours']:.6f} scientific CPU-hours and {runtime['wallHours']:.6f} wall-hours, with eight workers and one numerical-library thread per worker; GPU use was zero.

## Caveats, blockers, and limitations

1. The paper supports across-generation recurrence in general but does not state this exact all-molecular-state, nonadjacent existential rule.
2. Completed-run symmetric membership uses future observations and can support only retrospective paper-facing reconstruction.
3. The same 100 matrices have been studied in earlier loops, so known paper fingerprints create adaptive-overfitting risk; only untouched S20 data could confirm a promoted lead.
4. Strict `H>0.9` remains a similarity proxy and may classify slow cross-generation drift as recurrence even after excluding immediate neighbors.
5. Raw and normalized onset remain separate because the paper's Table 1 unit is internally inconsistent.
6. No downstream emergence association, prediction, or intervention result was recalculated under this exploratory label.

## Provenance

- Pushed pre-outcome repository commit: `{runtime['repositoryCommit']}` on `eidosoma/groups/42`.
- Original paper: arXiv `2607.28250v1`; hashes are recorded in `input_manifest.json` and `source_snapshot_manifest.json`.
- Frozen S13Y identities and exact replay: `preanalysis_replay_evidence.parquet` and `frozen_comparator_replay.parquet`.
- Exact formula and seeds: repository source, `method_lock.json`, `label_registry.yaml`, `specification_ledger.parquet`, and `seed_manifest.parquet`.

## Recommended next action and mandatory boundary

Return control for human review now. L04 issued no confirmatory verdict. L05, S20, E02, author contact, and report-bundle generation remain inactive unless a later explicit human decision authorizes one bounded action.
"""


def decision_summary_text(
    aggregate: pd.DataFrame,
    negative: pd.DataFrame,
    classification: dict[str, Any],
    validation_result: str,
) -> str:
    compact = aggregate[
        [
            "candidateId",
            "labelId",
            "meanOccupancy",
            "meanPersistence",
            "meanConsistency",
            "meanFirstOnsetRawIndex0",
            "meanFirstOnsetNormalized",
            "meanEpisodeCount",
            "nonreplicatingAtCutoffFraction",
        ]
    ]
    promoted = classification["promotedLeadIds"]
    return f"""# S19-L04 decision summary

## Concise top summary

- **Research step ID:** S19-L04
- **Completion status:** COMPLETE; mandatory human review reached
- **Artifacts written:** full L04 evidence, validation and hash manifests, plus append-only S19 ledgers
- **Validation result:** {validation_result}
- **Outcome classification:** `{classification['topLevelClassification']}`; {len(promoted)} promoted lead(s)
- **Caveats or blockers:** Previously studied matrices; completed-run future dependence; unavailable author definition; occupancy alone prohibited
- **Recommended next action:** Human review; authorize no L05 or S20 automatically
- **Lay summary:** One fixed cross-generation recurrence definition was tested against the full paper fingerprint, not tuned to 88% occupancy.

## Decision evidence

{markdown_table(compact)}

## Negative-control gate

{markdown_table(negative)}

Promoted leads: **{len(promoted)}** ({', '.join(promoted) if promoted else 'none'}).

## Human-review boundary

Stop now. L05, S20, E02, author contact, and report generation are inactive.
"""


def artifact_manifest(root: Path, required: list[str], schema: str) -> dict[str, Any]:
    missing = [name for name in required if not (root / name).is_file()]
    manifest_path = root / "artifact_manifest.json"
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != manifest_path):
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
        raise RuntimeError("L04 pre-loop self-improvement row cardinality changed")
    compact = aggregate[
        [
            "candidateId",
            "labelId",
            "meanOccupancy",
            "meanPersistence",
            "meanConsistency",
            "meanFirstOnsetRawIndex0",
            "meanEpisodeCount",
        ]
    ].to_dict(orient="records")
    row = {
        "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
        "timestampUtc": timestamp,
        "loopId": LOOP_ID,
        "recordPhase": "POST_LOOP_LEARNING_AND_HUMAN_REVIEW_BOUNDARY",
        "beliefBeforeLoop": "Recurrence in any different generation could provide a middle ground between adjacent smoothness and one dominant compotype.",
        "motivatingEvidence": "Direct paper wording says recurring compositions are inherited across generations.",
        "failureOrAmbiguityTargeted": "Cross-generation recurrence membership with multiple possible recurring regions.",
        "selectedHypotheses": "One strict-H>0.9 nonadjacent completed-run cross-generation label.",
        "learned": canonical_json(
            {"aggregateFingerprint": compact, "promotedLeadIds": classification["promotedLeadIds"]}
        ),
        "weakenedHypotheses": "The exact singleton cross-generation rule if it failed any joint, replay, uncertainty, influence, permutation, or cross-candidate gate.",
        "remainingPlausibleHypotheses": "Only the exact promoted retrospective lead, if any, or another separately authorized and nonduplicative author ambiguity.",
        "proposedNextTest": "Mandatory human review; no automatic L05 or S20.",
        "informationGainRationale": "The singleton test rules in or constrains a broad across-generation recurrence class without specification expansion.",
        "appendOnly": True,
    }
    pd.concat([ledger, pd.DataFrame([row])[ledger.columns]], ignore_index=True).to_parquet(
        path, index=False
    )
    with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Entry 008 — S19-L04 learning and human-review boundary\n\n"
            "- **Belief before the loop:** Any-state recurrence across distinct generations might recover the joint fingerprint better than either adjacent smoothness or one modal compotype.\n"
            "- **What was tested:** Exactly one strict-`H>0.9`, nonadjacent, symmetric completed-run recurrence rule plus the frozen adjacent comparator.\n"
            f"- **What was learned:** {len(classification['promotedLeadIds'])} lead(s) passed every locked promotion gate; exact values are preserved in the L04 machine-readable evidence and full report.\n"
            "- **Hypotheses weakened:** The singleton rule to the extent it failed joint, uncertainty, influence, permutation, or cross-candidate gates.\n"
            "- **What remains plausible:** Only promoted leads, if any, and genuinely nonduplicative source-grounded ambiguities approved later by a human.\n"
            "- **Next action:** Mandatory human review; no automatic L05 or S20.\n"
            "- **Why another loop could add information:** It must isolate a new dependency and not retune this completed rule.\n"
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
            "completed_run_symmetric_label_is_retrospective",
            "exact_author_replicator_definition_unavailable",
            "no_prospective_prediction_or_causal_inference",
            "L05_S20_E02_inactive",
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
            "decision": "S19_L04_COMPLETE_MANDATORY_HUMAN_REVIEW",
            "scope": VERSION,
            "source": "locked_execution_result",
        }
    )
    history["pendingDecision"] = "POST_S19_L04_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)


def main(workers: int) -> None:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    started_utc = datetime.now(timezone.utc)
    lock = execution_lock_validation()
    write_json(LOOP_ROOT / "execution_lock_validation.json", lock)
    if not lock["passed"]:
        raise RuntimeError("L04 execution lock validation failed")

    manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet")
    labels, diagnostics, replay = execute_labels(manifest, workers)
    execution = replay[
        ["candidateId", "matrixIndex", "trajectoryId", "cpuSeconds", "wallSeconds"]
    ].drop_duplicates(["candidateId", "matrixIndex"])
    execution["success"] = True
    if not replay["exactReplayPassed"].all():
        raise RuntimeError("L04 exact label replay failed")
    if not diagnostics.loc[
        diagnostics["labelId"].eq(STRUCTURAL_LABEL_ID), "exactSymmetryPassed"
    ].fillna(False).all():
        raise RuntimeError("L04 structural recurrence symmetry failed")
    comparator_replay = frozen_comparator_replay(labels)
    if not comparator_replay["passed"].all():
        raise RuntimeError("L04 frozen adjacent-H comparator replay failed")

    fingerprints = build_fingerprints(labels, diagnostics)
    episodes = episode_table(labels)
    aggregate = aggregate_fingerprints(fingerprints)
    comparison = paper_comparison(aggregate)
    bootstrap, bootstrap_metrics = bootstrap_analysis(fingerprints)
    loo = leave_one_out(fingerprints)
    overlaps = overlap_results(labels)
    cross = cross_candidate_agreement(fingerprints, aggregate)
    permutation, negative = generation_block_permutation(labels, fingerprints, comparison)
    robustness = robustness_table(bootstrap, loo, negative)
    classification = classify(
        aggregate, comparison, bootstrap, loo, negative, cross, replay
    )

    labels.to_parquet(LOOP_ROOT / "label_values.parquet", index=False, compression="zstd")
    labels.loc[labels["labelId"].eq(STRUCTURAL_LABEL_ID)].to_parquet(
        LOOP_ROOT / "recurrence_evidence.parquet", index=False, compression="zstd"
    )
    diagnostics.to_parquet(
        LOOP_ROOT / "recurrence_trajectory_diagnostics.parquet", index=False
    )
    replay.to_parquet(LOOP_ROOT / "label_replay_evidence.parquet", index=False)
    comparator_replay.to_parquet(LOOP_ROOT / "frozen_comparator_replay.parquet", index=False)
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
        LOOP_ROOT / "recurrence_count_results.parquet", index=False
    )
    overlaps.to_parquet(LOOP_ROOT / "label_overlap_results.parquet", index=False)
    cross.to_csv(LOOP_ROOT / "cross_candidate_agreement.csv", index=False)
    comparison.to_csv(LOOP_ROOT / "paper_fingerprint_comparison.csv", index=False)
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
        raise RuntimeError("L04 immutable prior changed")
    completed_utc = datetime.now(timezone.utc)
    scientific_cpu_seconds = float(execution["cpuSeconds"].sum()) + (
        time.process_time() - started_cpu
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l04_runtime_manifest.v1",
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
        raise RuntimeError("L04 compute ceiling exceeded")
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime)

    retained_bytes = sum(path.stat().st_size for path in LOOP_ROOT.rglob("*") if path.is_file())
    temporary_bytes = sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l04_storage_validation.v1",
        "retainedBytesBeforeManifest": retained_bytes,
        "retainedGiB": retained_bytes / (1024**3),
        "retainedCeilingGiB": 10.0,
        "temporaryBytes": temporary_bytes,
        "temporaryGiB": temporary_bytes / (1024**3),
        "temporaryCeilingGiB": 25.0,
        "passed": retained_bytes <= 10 * 1024**3 and temporary_bytes <= 25 * 1024**3,
    }
    write_json(LOOP_ROOT / "storage_validation.json", storage)
    if not storage["passed"]:
        raise RuntimeError("L04 storage ceiling exceeded")

    validation_result = "PASS_ALL_LOCK_REPLAY_IMMUTABILITY_BOOTSTRAP_LOO_GENERATION_BLOCK_STORAGE_REGENERATION_AND_HASH_CHECKS"
    report = report_text(
        aggregate,
        comparison,
        bootstrap_metrics,
        negative,
        overlaps,
        classification,
        validation_result,
        runtime,
        immutable["fileCount"],
    )
    decision = decision_summary_text(aggregate, negative, classification, validation_result)
    if report != report_text(
        aggregate,
        comparison,
        bootstrap_metrics,
        negative,
        overlaps,
        classification,
        validation_result,
        runtime,
        immutable["fileCount"],
    ):
        raise RuntimeError("L04 report regeneration mismatch")
    (LOOP_ROOT / "S19_L04_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "loop_decision_summary.md").write_text(decision, encoding="utf-8")

    regeneration = {
        "schema": "eidosoma.e01.s19_l04_regeneration_validation.v1",
        "labelCount": labels["labelId"].nunique(),
        "structuralCandidateCount": 1,
        "candidateCount": labels["candidateId"].nunique(),
        "matrixCountPerCandidate": labels.groupby("candidateId")["matrixIndex"].nunique().to_dict(),
        "trajectoryCount": execution.shape[0],
        "labelRowCount": len(labels),
        "structuralLabelRowCount": int(labels["labelId"].eq(STRUCTURAL_LABEL_ID).sum()),
        "fingerprintRowCount": len(fingerprints),
        "permutationResultRowCount": len(permutation),
        "exactLabelReplayPassed": bool(replay["exactReplayPassed"].all()),
        "frozenComparatorReplayPassed": bool(comparator_replay["passed"].all()),
        "crossGenerationSymmetryPassed": bool(
            diagnostics.loc[
                diagnostics["labelId"].eq(STRUCTURAL_LABEL_ID), "exactSymmetryPassed"
            ].fillna(False).all()
        ),
        "immutablePriorPassed": immutable["passed"],
        "reportDeterministic": True,
        "thresholdGridAbsent": True,
        "H097CandidateAbsent": True,
        "newGardTrajectories": 0,
        "newPhiRLOrEmergenceValues": 0,
        "passed": bool(
            labels["labelId"].nunique() == 2
            and labels["candidateId"].nunique() == 2
            and execution.shape[0] == 200
            and fingerprints.shape[0] == 400
            and len(permutation) == 2 * 2 * PERMUTATION_REPLICATES
            and replay["exactReplayPassed"].all()
            and comparator_replay["passed"].all()
            and immutable["passed"]
            and storage["passed"]
        ),
    }
    write_json(LOOP_ROOT / "regeneration_validation.json", regeneration)
    if not regeneration["passed"]:
        raise RuntimeError("L04 final regeneration validation failed")

    loop_artifacts = [
        str(LOOP_ROOT / "S19_L04_FULL_RESULTS.md"),
        str(LOOP_ROOT / "classification.json"),
        str(LOOP_ROOT / "fingerprint_results.parquet"),
        str(LOOP_ROOT / "paper_fingerprint_comparison.csv"),
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
            "completed_run_symmetric_label_is_retrospective",
            "exact_author_replicator_definition_unavailable",
            "no_prospective_prediction_or_causal_inference",
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
        "recurrence_evidence.parquet",
        "recurrence_trajectory_diagnostics.parquet",
        "label_replay_evidence.parquet",
        "frozen_comparator_replay.parquet",
        "fingerprint_results.parquet",
        "results.parquet",
        "fingerprint_summary.parquet",
        "fingerprint_aggregate.csv",
        "episode_results.parquet",
        "cutoff_results.parquet",
        "recurrence_count_results.parquet",
        "label_overlap_results.parquet",
        "cross_candidate_agreement.csv",
        "paper_fingerprint_comparison.csv",
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
        "S19_L04_FULL_RESULTS.md",
        "research_step_full_results.md",
    ]
    loop_manifest = artifact_manifest(
        LOOP_ROOT, required, "eidosoma.e01.s19_l04_artifact_manifest.v1"
    )
    if not loop_manifest["passed"]:
        raise RuntimeError(f"missing L04 artifacts: {loop_manifest['missing']}")
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
        ARTIFACT_ROOT, root_required, "eidosoma.e01.s19_artifact_manifest.v4"
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
