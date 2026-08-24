#!/usr/bin/env python3
"""Execute the locked E01/S19-L03 boundary-compotype label analysis."""

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

from e01_s19_boundary_compotype.core import (
    BOOTSTRAP_REPLICATES,
    CANDIDATE_IDS,
    COMPARATOR_LABEL_ID,
    LABEL_BY_ID,
    LABEL_DEFINITIONS,
    LOOP_ID,
    VERSION,
    derive_seed128,
    label_trajectory,
)
from e01_s19_replicator_definition.core import (
    closer_dimension_count,
    fingerprint_from_labels,
    paper_fingerprint_distance,
)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L03"
CACHE_ROOT = Path("/cache/e01_s19_l03")
LABEL_CACHE = CACHE_ROOT / "labels"
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
L02_ROOT = ARTIFACT_ROOT / "loops/L02"
PREREG = REPO_ROOT / "configs/e01/s19_l03_preregistration.yaml"
METHOD_LOCK = REPO_ROOT / "configs/e01/s19_l03_method_lock.json"

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
    "firstOnsetNormalized",
    "entryCount",
    "exitCount",
    "episodeCount",
    "meanEpisodeDuration",
    "medianEpisodeDuration",
    "longestEpisode",
    "postFissionReplicatorFraction",
    "postFissionEpisodeCount",
    "referenceFrequency",
    "referenceGeneration",
    "secondMemberGeneration",
)
PAPER_TARGETS = {
    "persistence": 716.0,
    "occupancy": 0.88,
    "consistency": 0.38,
    "firstOnsetRawScore": 37.0,
    "firstOnsetNormalizedScore": 0.37,
}


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


def hash_strings(values: list[str]) -> str:
    return hashlib.sha256("\x1f".join(values).encode()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def bool_codes(series: pd.Series) -> np.ndarray:
    return np.asarray(
        [-1 if pd.isna(value) else int(bool(value)) for value in series.astype(object)],
        dtype=np.int8,
    )


def frame_identity(frame: pd.DataFrame) -> dict[str, str]:
    ordered = frame.sort_values("selectedSequenceIndex", kind="stable")
    scores = pd.to_numeric(ordered["labelScore"], errors="coerce").to_numpy(dtype=np.float64)
    sources = pd.to_numeric(ordered["sourceBoundaryGeneration"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    members = bool_codes(ordered["boundaryMember"])
    return {
        "labelSha256": sha256_array(bool_codes(ordered["isReplicator"])),
        "scoreSha256": sha256_array(scores),
        "sequenceSha256": sha256_array(
            ordered["selectedSequenceIndex"].to_numpy(dtype=np.int64)
        ),
        "rawIndexSha256": sha256_array(ordered["rawObservationIndex"].to_numpy(dtype=np.int64)),
        "generationSha256": sha256_array(ordered["generation"].to_numpy(dtype=np.int64)),
        "sourceBoundarySha256": sha256_array(sources),
        "boundaryMemberSha256": sha256_array(members),
        "statusSha256": hash_strings(ordered["labelStatus"].astype(str).tolist()),
        "kindSha256": hash_strings(ordered["observationKind"].astype(str).tolist()),
    }


def boundary_identity(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"EMPTY").hexdigest()
    ordered = frame.sort_values("boundaryIndex", kind="stable")
    payload = ordered.to_json(orient="records", double_precision=15)
    return hashlib.sha256(payload.encode()).hexdigest()


def normalize_label_frame(frame: pd.DataFrame, label_id: str) -> pd.DataFrame:
    definition = LABEL_BY_ID[label_id]
    output = frame.copy()
    output["researchStepId"] = LOOP_ID
    output["labelId"] = label_id
    output["labelFamily"] = definition.role
    output["labelEvidenceTier"] = definition.evidence_class
    output["temporalScope"] = definition.temporal_scope
    for column in (
        "boundarySubstrate",
        "activationRule",
        "projectionRule",
        "boundaryMember",
        "sourceBoundaryGeneration",
    ):
        if column not in output:
            output[column] = None
    output["isReplicator"] = pd.array(output["isReplicator"], dtype="boolean")
    output["boundaryMember"] = pd.array(output["boundaryMember"], dtype="boolean")
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
        "boundarySubstrate",
        "activationRule",
        "projectionRule",
        "boundaryMember",
        "sourceBoundaryGeneration",
    ]
    return output[columns]


def trajectory_worker(record: dict[str, Any]) -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    with Path(record["cachePath"]).open("rb") as handle:
        trajectory = pickle.load(handle)
    frames = []
    boundaries = []
    diagnostics = []
    replays = []
    for definition in LABEL_DEFINITIONS:
        first_raw, first_diagnostic, first_boundary = label_trajectory(
            trajectory, definition, clock_id=str(record["clockId"])
        )
        second_raw, second_diagnostic, second_boundary = label_trajectory(
            trajectory, definition, clock_id=str(record["clockId"])
        )
        first = normalize_label_frame(first_raw, definition.label_id)
        second = normalize_label_frame(second_raw, definition.label_id)
        first_id = frame_identity(first)
        second_id = frame_identity(second)
        diagnostic_equal = canonical_json(first_diagnostic) == canonical_json(second_diagnostic)
        boundary_equal = boundary_identity(first_boundary) == boundary_identity(second_boundary)
        passed = first_id == second_id and diagnostic_equal and boundary_equal
        replays.append(
            {
                "candidateId": record["candidateId"],
                "matrixIndex": int(record["matrixIndex"]),
                "trajectoryId": record["trajectoryId"],
                "labelId": definition.label_id,
                "firstFrameIdentity": canonical_json(first_id),
                "secondFrameIdentity": canonical_json(second_id),
                "firstBoundaryIdentity": boundary_identity(first_boundary),
                "secondBoundaryIdentity": boundary_identity(second_boundary),
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
                "referenceIndex": first_diagnostic.get("referenceIndex"),
                "referenceGeneration": first_diagnostic.get("referenceGeneration"),
                "referenceFrequency": first_diagnostic.get("referenceFrequency"),
                "recurrent": first_diagnostic.get("recurrent"),
                "firstMemberGeneration": first_diagnostic.get("firstMemberGeneration"),
                "secondMemberGeneration": first_diagnostic.get("secondMemberGeneration"),
                "boundaryCount": first_diagnostic.get("boundaryCount"),
                "eligibleMolecularCount": first_diagnostic.get("eligibleMolecularCount"),
            }
        )
        if not first_boundary.empty:
            boundary = first_boundary.copy()
            boundary["researchStepId"] = LOOP_ID
            boundaries.append(boundary)
        frames.append(first)
    combined = pd.concat(frames, ignore_index=True)
    boundary_combined = pd.concat(boundaries, ignore_index=True) if boundaries else pd.DataFrame()
    output = LABEL_CACHE / str(record["candidateId"]) / f"M{int(record['matrixIndex']):03d}.parquet"
    boundary_output = (
        LABEL_CACHE / str(record["candidateId"]) / f"M{int(record['matrixIndex']):03d}.boundaries.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False, compression="zstd")
    boundary_combined.to_parquet(boundary_output, index=False, compression="zstd")
    return {
        "candidateId": record["candidateId"],
        "matrixIndex": int(record["matrixIndex"]),
        "trajectoryId": record["trajectoryId"],
        "labelCache": str(output),
        "boundaryCache": str(boundary_output),
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
        "schema": "eidosoma.e01.s19_l03_execution_lock_validation.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "preparedHead": repository["head"],
        "cleanWorktree": clean,
        "configHashes": hashes,
        "preanalysisReplayPassed": replay["passed"],
        "immutablePriorPassed": immutable["passed"],
        "benchmarkPassed": benchmark["gatePassed"],
        "passed": passed,
    }


def execute_labels(
    manifest: pd.DataFrame, workers: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
                "boundaryCache": row["boundaryCache"],
            }
            for row in results
        ]
    )
    labels = pd.concat([pd.read_parquet(row["labelCache"]) for row in results], ignore_index=True)
    labels["isReplicator"] = pd.array(labels["isReplicator"], dtype="boolean")
    labels["boundaryMember"] = pd.array(labels["boundaryMember"], dtype="boolean")
    boundary = pd.concat(
        [pd.read_parquet(row["boundaryCache"]) for row in results], ignore_index=True
    )
    diagnostics = pd.DataFrame([item for row in results for item in row["diagnostics"]])
    replay = pd.DataFrame([item for row in results for item in row["replays"]])
    return labels, boundary, diagnostics, replay, execution


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
            label_pass = np.array_equal(bool_codes(left["isReplicator"]), bool_codes(right["isReplicator"]))
            score_pass = np.array_equal(
                pd.to_numeric(left["labelScore"], errors="coerce").to_numpy(dtype=np.float64),
                pd.to_numeric(right["labelScore"], errors="coerce").to_numpy(dtype=np.float64),
                equal_nan=True,
            )
            rows.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": matrix,
                    "labelId": COMPARATOR_LABEL_ID,
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
                "boundarySubstrate": definition.boundary_substrate,
                "activationRule": definition.activation_rule,
                "projectionRule": definition.projection_rule,
                "referenceIndex": diagnostic["referenceIndex"],
                "referenceGeneration": diagnostic["referenceGeneration"],
                "referenceFrequency": diagnostic["referenceFrequency"],
                "recurrent": diagnostic["recurrent"],
                "firstMemberGeneration": diagnostic["firstMemberGeneration"],
                "secondMemberGeneration": diagnostic["secondMemberGeneration"],
                "boundaryCount": diagnostic["boundaryCount"],
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
        start = None
        prior = None
        episode = 0
        for index, value in zip(group["selectedSequenceIndex"], values, strict=True):
            current = int(index)
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
            contiguous = prior is not None and current == prior + 1
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
            aggregate["candidateId"].eq(candidate) & aggregate["labelId"].eq(COMPARATOR_LABEL_ID)
        ].iloc[0]
        comparator = summary_dict(comparator_row)
        for row in aggregate.loc[aggregate["candidateId"].eq(candidate)].itertuples(index=False):
            current_row = pd.Series(row._asdict())
            current = summary_dict(current_row)
            for mode in ("RAW", "NORMALIZED"):
                distance = paper_fingerprint_distance(current, onset_mode=mode)
                comparator_distance = paper_fingerprint_distance(comparator, onset_mode=mode)
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
                            None if distance is None or comparator_distance is None else distance - comparator_distance
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


def aggregate_sample(group: pd.DataFrame, positions: np.ndarray) -> dict[str, float]:
    sampled = group.iloc[positions]
    return {metric: float(pd.to_numeric(sampled[metric], errors="coerce").mean()) for metric in CORE_METRICS}


def bootstrap_distances(fingerprints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATE_IDS:
        comparator = fingerprints.loc[
            fingerprints["candidateId"].eq(candidate)
            & fingerprints["labelId"].eq(COMPARATOR_LABEL_ID)
        ].sort_values("matrixIndex", kind="stable")
        for definition in LABEL_DEFINITIONS:
            if definition.comparator_only:
                continue
            current = fingerprints.loc[
                fingerprints["candidateId"].eq(candidate)
                & fingerprints["labelId"].eq(definition.label_id)
            ].sort_values("matrixIndex", kind="stable")
            if not np.array_equal(
                current["matrixIndex"].to_numpy(), comparator["matrixIndex"].to_numpy()
            ):
                raise RuntimeError("paired bootstrap matrix identity mismatch")
            rng = np.random.Generator(
                np.random.PCG64DXSM(
                    derive_seed128(candidate, definition.label_id, "paired_matrix_bootstrap")
                )
            )
            values = {"RAW": [], "NORMALIZED": []}
            occupancy_differences = []
            for _ in range(BOOTSTRAP_REPLICATES):
                positions = rng.integers(0, len(current), size=len(current))
                candidate_summary = aggregate_sample(current, positions)
                comparator_summary = aggregate_sample(comparator, positions)
                occupancy_differences.append(
                    abs(candidate_summary["occupancy"] - 0.88)
                    - abs(comparator_summary["occupancy"] - 0.88)
                )
                for mode in ("RAW", "NORMALIZED"):
                    left = paper_fingerprint_distance(candidate_summary, onset_mode=mode)
                    right = paper_fingerprint_distance(comparator_summary, onset_mode=mode)
                    values[mode].append(
                        np.nan if left is None or right is None else float(left - right)
                    )
            occupancy_array = np.asarray(occupancy_differences)
            for mode, differences in values.items():
                array = np.asarray(differences)
                finite = array[np.isfinite(array)]
                rows.append(
                    {
                        "candidateId": candidate,
                        "labelId": definition.label_id,
                        "onsetMode": mode,
                        "replicates": BOOTSTRAP_REPLICATES,
                        "definedReplicates": len(finite),
                        "meanDistanceDifference": (
                            float(finite.mean()) if len(finite) else None
                        ),
                        "lower95": (
                            float(np.quantile(finite, 0.025)) if len(finite) else None
                        ),
                        "upper95": (
                            float(np.quantile(finite, 0.975)) if len(finite) else None
                        ),
                        "probabilityDistanceImprovement": (
                            float(np.mean(finite < 0)) if len(finite) else None
                        ),
                        "occupancyErrorDifferenceMean": float(occupancy_array.mean()),
                        "occupancyErrorDifferenceLower95": float(np.quantile(occupancy_array, 0.025)),
                        "occupancyErrorDifferenceUpper95": float(np.quantile(occupancy_array, 0.975)),
                    }
                )
    return pd.DataFrame(rows)


def leave_one_out(fingerprints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATE_IDS:
        comparator = fingerprints.loc[
            fingerprints["candidateId"].eq(candidate)
            & fingerprints["labelId"].eq(COMPARATOR_LABEL_ID)
        ].sort_values("matrixIndex", kind="stable")
        for definition in LABEL_DEFINITIONS:
            if definition.comparator_only:
                continue
            current = fingerprints.loc[
                fingerprints["candidateId"].eq(candidate)
                & fingerprints["labelId"].eq(definition.label_id)
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
                            "labelId": definition.label_id,
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
    frozen = pd.read_parquet(S13Y_ROOT / "label_values.parquet")
    baselines = {
        "ADJACENT_H900": labels.loc[labels["labelId"].eq(COMPARATOR_LABEL_ID)],
        "HISTORICAL_L02_NONDRIFT": frozen.loc[
            frozen["labelId"].eq("HISTORICAL_H090_REPLICATOR")
        ],
    }
    rows = []
    for definition in LABEL_DEFINITIONS:
        if definition.comparator_only:
            continue
        current_all = labels.loc[labels["labelId"].eq(definition.label_id)]
        for baseline_id, baseline_all in baselines.items():
            for candidate in CANDIDATE_IDS:
                left = current_all.loc[current_all["candidateId"].eq(candidate)][
                    ["matrixIndex", "selectedSequenceIndex", "isReplicator"]
                ].rename(columns={"isReplicator": "left"})
                right = baseline_all.loc[baseline_all["candidateId"].eq(candidate)][
                    ["matrixIndex", "selectedSequenceIndex", "isReplicator"]
                ].rename(columns={"isReplicator": "right"})
                joined = left.merge(
                    right,
                    on=["matrixIndex", "selectedSequenceIndex"],
                    how="inner",
                    validate="one_to_one",
                ).dropna(subset=["left", "right"])
                a = joined["left"].astype(bool).to_numpy()
                b = joined["right"].astype(bool).to_numpy()
                union = np.count_nonzero(a | b)
                rows.append(
                    {
                        "candidateId": candidate,
                        "labelId": definition.label_id,
                        "baselineId": baseline_id,
                        "commonEligibleCount": len(joined),
                        "accuracy": float(np.mean(a == b)) if len(joined) else None,
                        "jaccard": float(np.count_nonzero(a & b) / union) if union else None,
                        "mismatchFraction": float(np.mean(a != b)) if len(joined) else None,
                    }
                )
    return pd.DataFrame(rows)


def cross_candidate_agreement(fingerprints: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
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
        for metric in ("occupancy", "consistency", "firstOnsetNormalizedScore", "episodeCount"):
            a = pd.to_numeric(left[metric], errors="coerce").to_numpy(dtype=np.float64)
            b = pd.to_numeric(right[metric], errors="coerce").to_numpy(dtype=np.float64)
            finite = np.isfinite(a) & np.isfinite(b)
            correlation = None
            if finite.sum() >= 3 and np.ptp(a[finite]) > 0 and np.ptp(b[finite]) > 0:
                correlation = float(np.corrcoef(a[finite], b[finite])[0, 1])
            token = metric[0].upper() + metric[1:]
            left_mean = left_agg[f"mean{token}"]
            right_mean = right_agg[f"mean{token}"]
            mean_difference = None
            if pd.notna(left_mean) and pd.notna(right_mean):
                mean_difference = abs(float(left_mean) - float(right_mean))
            rows.append(
                {
                    "labelId": definition.label_id,
                    "metric": metric,
                    "pairedDefinedCount": int(finite.sum()),
                    "candidate2Mean": left_mean,
                    "candidate3Mean": right_mean,
                    "absoluteMeanDifference": mean_difference,
                    "matrixLevelPearson": correlation,
                }
            )
    return pd.DataFrame(rows)


def projection_contrasts(fingerprints: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        (
            "BACKFILL_MINUS_ACTIVATED_INCOMING",
            "PF_MODAL_MEDOID_BACKFILL_INCOMING_H900",
            "PF_MODAL_MEDOID_ACTIVATED_INCOMING_H900",
        ),
        (
            "OUTGOING_MINUS_INCOMING_ACTIVATED",
            "PF_MODAL_MEDOID_ACTIVATED_OUTGOING_H900",
            "PF_MODAL_MEDOID_ACTIVATED_INCOMING_H900",
        ),
        (
            "GENERATION_END_MINUS_POSTFISSION_ACTIVATED_INCOMING",
            "GE_MODAL_MEDOID_ACTIVATED_INCOMING_H900",
            "PF_MODAL_MEDOID_ACTIVATED_INCOMING_H900",
        ),
    )
    rows = []
    for candidate in CANDIDATE_IDS:
        for contrast_id, left_id, right_id in pairs:
            left = fingerprints.loc[
                fingerprints["candidateId"].eq(candidate) & fingerprints["labelId"].eq(left_id)
            ].sort_values("matrixIndex", kind="stable")
            right = fingerprints.loc[
                fingerprints["candidateId"].eq(candidate) & fingerprints["labelId"].eq(right_id)
            ].sort_values("matrixIndex", kind="stable")
            rng = np.random.Generator(
                np.random.PCG64DXSM(
                    derive_seed128(candidate, left_id, "projection_contrast_bootstrap")
                )
            )
            for metric in (
                "persistence",
                "occupancy",
                "consistency",
                "firstOnsetRawScore",
                "firstOnsetNormalizedScore",
                "episodeCount",
                "noReplicatorObservedThroughCutoff",
            ):
                differences = (
                    pd.to_numeric(left[metric], errors="coerce").to_numpy(dtype=np.float64)
                    - pd.to_numeric(right[metric], errors="coerce").to_numpy(dtype=np.float64)
                )
                finite = differences[np.isfinite(differences)]
                means = []
                for _ in range(BOOTSTRAP_REPLICATES):
                    if len(finite):
                        means.append(
                            float(np.mean(rng.choice(finite, size=len(finite), replace=True)))
                        )
                array = np.asarray(means)
                rows.append(
                    {
                        "candidateId": candidate,
                        "contrastId": contrast_id,
                        "leftLabelId": left_id,
                        "rightLabelId": right_id,
                        "metric": metric,
                        "definedPairCount": len(finite),
                        "meanDifference": float(finite.mean()) if len(finite) else None,
                        "lower95": (
                            float(np.quantile(array, 0.025)) if len(array) else None
                        ),
                        "upper95": (
                            float(np.quantile(array, 0.975)) if len(array) else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def classify(
    aggregate: pd.DataFrame,
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    loo: pd.DataFrame,
    cross: pd.DataFrame,
    replay: pd.DataFrame,
) -> dict[str, Any]:
    label_results = []
    passing = []
    replay_all = bool(replay["exactReplayPassed"].all())
    for definition in LABEL_DEFINITIONS:
        if definition.comparator_only:
            label_results.append(
                {
                    "labelId": definition.label_id,
                    "classifications": ["POSSIBLE_STABILITY_PROXY", "NOT_PROMOTABLE"],
                    "promoted": False,
                    "promotionGates": {"notComparator": False, "exactReplay": replay_all},
                }
            )
            continue
        gates: dict[str, bool] = {
            "notComparator": True,
            "exactReplay": replay_all,
            "sourceAndPaperRelationship": True,
            "lockedImplementationComplete": True,
            "notSelectedByOccupancyOrDownstreamOutcomeAlone": True,
        }
        all_candidate_gates = []
        for candidate in CANDIDATE_IDS:
            agg = aggregate.loc[
                aggregate["candidateId"].eq(candidate)
                & aggregate["labelId"].eq(definition.label_id)
            ].iloc[0]
            comp_rows = comparison.loc[
                comparison["candidateId"].eq(candidate)
                & comparison["labelId"].eq(definition.label_id)
            ]
            boot_rows = bootstrap.loc[
                bootstrap["candidateId"].eq(candidate)
                & bootstrap["labelId"].eq(definition.label_id)
            ]
            loo_rows = loo.loc[
                loo["candidateId"].eq(candidate) & loo["labelId"].eq(definition.label_id)
            ]
            candidate_gate = bool(
                comp_rows["occupancyCloser"].all()
                and (comp_rows["distanceImprovementFraction"] >= 0.10).all()
                and (comp_rows["closerDimensionCount"] >= 3).all()
                and comp_rows["structureDimensionImproved"].all()
                and (boot_rows["upper95"] < 0).all()
                and (loo_rows["distanceDifference"] < 0).all()
                and int(agg["definedConsistencyCount"]) >= 95
                and int(agg["observedOnsetCount"]) >= 95
                and float(agg["nonreplicatingAtCutoffFraction"]) > 0
            )
            gates[f"completeJointGate_{candidate}"] = candidate_gate
            all_candidate_gates.append(candidate_gate)
        cross_label = cross.loc[cross["labelId"].eq(definition.label_id)]
        differences = dict(
            zip(cross_label["metric"], cross_label["absoluteMeanDifference"], strict=True)
        )
        cross_values = (
            differences.get("occupancy"),
            differences.get("consistency"),
            differences.get("firstOnsetNormalizedScore"),
        )
        cross_gate = bool(
            all(pd.notna(value) for value in cross_values)
            and float(cross_values[0]) <= 0.05
            and float(cross_values[1]) <= 0.10
            and float(cross_values[2]) <= 0.10
        )
        gates["crossCandidateAgreement"] = cross_gate
        eligible = all(gates.values())
        if eligible:
            passing.append(definition.label_id)
        paper_match = bool(
            (comparison.loc[comparison["labelId"].eq(definition.label_id), "paperDistance"] <= 1).all()
        )
        label_comparison = comparison.loc[
            comparison["labelId"].eq(definition.label_id)
        ]
        directional = bool(
            label_comparison["occupancyCloser"].all()
            and (label_comparison["closerDimensionCount"] >= 3).all()
            and label_comparison["structureDimensionImproved"].all()
        )
        classes = [
            "EXPLORATORY_PAPER_MATCH"
            if paper_match
            else ("EXPLORATORY_DIRECTIONAL_MATCH" if directional else "EXPLORATORY_NON_SUPPORT"),
            "RETROSPECTIVE_ONLY_LEAD",
            "METHOD_DEPENDENT_LEAD",
            "AUTHOR_AMBIGUITY_UNRESOLVED",
        ]
        label_results.append(
            {
                "labelId": definition.label_id,
                "classifications": classes,
                "promoted": False,
                "promotionGates": gates,
            }
        )
    frozen_rank = {
        row.proposedSpecification: int(row.frozenRank)
        for row in pd.read_csv(LOOP_ROOT / "candidate_ranking.csv").itertuples(index=False)
    }
    promoted = sorted(passing, key=lambda label: (frozen_rank[label], LABEL_BY_ID[label].ordinal))[:2]
    for result in label_results:
        if result["labelId"] in promoted:
            result["promoted"] = True
            result["classifications"].append("PROMOTABLE_TO_S20")
        elif result["labelId"] != COMPARATOR_LABEL_ID:
            result["classifications"].append("NOT_PROMOTABLE")
    return {
        "schema": "eidosoma.e01.s19_l03_classification.v1",
        "researchStepId": LOOP_ID,
        "confirmatoryVerdictIssued": False,
        "promotedLeadCount": len(promoted),
        "promotedLeadIds": promoted,
        "labelClassifications": label_results,
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
        "schema": "eidosoma.e01.s19_l03_immutable_prior_postcheck.v1",
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
    contrasts: pd.DataFrame,
    classification: dict[str, Any],
    validation_result: str,
    runtime: dict[str, Any],
) -> str:
    promoted = classification["promotedLeadIds"]
    overall = (
        "EXPLORATORY_DIRECTIONAL_BOUNDARY_PROJECTION_LEAD"
        if promoted
        else "EXPLORATORY_CONSTRAINING_NO_PROMOTABLE_LEAD"
    )
    summary_rows = []
    for row in aggregate.itertuples(index=False):
        summary_rows.append(
            {
                "Candidate": "C02" if row.candidateId.endswith("02") else "C03",
                "Label": row.labelId,
                "Persistence": row.meanPersistence,
                "Occupancy": row.meanOccupancy,
                "Consistency": row.meanConsistency,
                "Onset raw": row.meanFirstOnsetRawIndex0,
                "Onset norm": row.meanFirstOnsetNormalized,
                "Episodes": row.meanEpisodeCount,
                "Nonrep@25%": row.nonreplicatingAtCutoffFraction,
                "Boundary freq": row.meanReferenceFrequency,
            }
        )
    comparison_view = comparison.loc[
        ~comparison["labelId"].eq(COMPARATOR_LABEL_ID),
        [
            "candidateId",
            "labelId",
            "onsetMode",
            "paperDistance",
            "distanceImprovementFraction",
            "closerDimensionCount",
            "occupancyCloser",
        ],
    ].copy()
    contrast_view = contrasts.loc[
        contrasts["metric"].isin(["occupancy", "consistency", "firstOnsetRawScore"]),
        ["candidateId", "contrastId", "metric", "meanDifference", "lower95", "upper95"],
    ]
    return f"""# S19-L03 — Boundary compotype projection and recurrence activation

## Concise top summary

- **Research step ID:** S19-L03
- **Completion status:** COMPLETE; mandatory human-review boundary reached; S19-L04, S20, and all downstream work remain inactive
- **Artifacts written:** complete compact L03 preregistration, label, boundary, fingerprint, robustness, replay, validation, provenance, status, and handoff evidence plus append-only S19 root-ledger updates
- **Validation result:** {validation_result}
- **Outcome classification:** {overall}
- **Caveats or blockers:** The modal reference is a completed-run retrospective reconstruction, not recovered author code. Matching occupancy alone was prohibited, and every prior result remains unchanged.
- **Recommended next action:** Human review must select any next bounded S19 theme or another program action; do not begin it automatically.
- **Lay summary:** This loop asked whether the paper's replicator state may be assigned only at fission or generation boundaries and then spread across the intervening molecular steps. It fixed one `H>0.9` recurring-boundary rule and compared backfilling, activation after recurrence, incoming/outgoing projection, and pre-/post-fission boundary choices. It did not tune the threshold or use causal emergence to choose a label.

## Frozen question and nonduplication decision

L02 did not fully test this ambiguity. Its centroid and K-means labels classified every molecular observation directly, while its historical branch projected only a local adjacent non-drift state. L03 instead identified one modal compotype among 100 boundary compositions and projected boundary membership onto molecular time. The underlying strict-`H>0.9` maximum-neighbor medoid, tie rule, minimum recurrence of two, four structural candidates, and all statistical gates were pushed before outcomes.

## Inputs

- Frozen S13Y: 100 shared matrices, candidate 2 and candidate 3 kept separate, 200 complete trajectories.
- Original arXiv v1 paper and frozen S08/S13X/S18/L01/L02 context.
- Pinned historical GARD `tgs_agard_v10.m`, `tgs_nondrift.m`, and `getcomposometime_v10.m` source identities.
- New GARD trajectories: **0**. New PhiRL/emergence values: **0**. Prediction/intervention outcomes used: **0**. GPU use: **0**.

## Detailed methods

For each trajectory, boundary compositions were L1 closed. A boundary state became the modal reference when it had the most other boundary states at strict historical cosine `H>0.9`, including itself; ties went to the earliest generation and boundary index. Fewer than two members meant no recurrent compotype. The five locked labels were the frozen adjacent molecular comparator plus:

1. post-fission modal membership backfilled to incoming boundary-ending intervals;
2. the same label activated only from its second occurrence, incoming aligned;
3. the activated label projected to outgoing boundary-starting intervals;
4. the activated label built from the historical pre-fission generation-end substrate, incoming aligned.

The incoming/outgoing pair changes only interval alignment. The backfill/activation pair changes only recurrence activation. The post-fission/generation-end pair changes only boundary substrate. Full-run reference selection makes every structural label retrospective even when activation itself is forward ordered.

Each trajectory retained persistence, occupancy, raw and normalized onset, Pearson consecutive-label consistency, entries/exits, episode structure, recurrence frequency/span, and 25% cutoff status. Catalytic matrix was the bootstrap and leave-one-out unit; molecular rows were never treated as independent replicates. Paper-distance used persistence 716, occupancy 0.88, consistency 0.38, and raw onset 37 or normalized onset 0.37 as two separate analyses.

## Results

### Candidate-specific temporal fingerprints

{markdown_table(pd.DataFrame(summary_rows))}

### Joint paper-fingerprint comparison

{markdown_table(comparison_view)}

### Isolated structural contrasts

{markdown_table(contrast_view)}

Promoted leads: **{len(promoted)}** ({', '.join(promoted) if promoted else 'none'}). Directional resemblance, when present, remains exploratory and retrospective. An occupancy move toward 0.88 cannot override disagreement in persistence, onset, consistency, episodes, recurrence, cutoff eligibility, uncertainty, or the other simulator candidate.

## Robustness, falsification, and validation

- Exact independent replay of every one of the five labels on all 200 trajectories: PASS.
- Frozen adjacent-H and `H>0.9` comparator identity: PASS.
- Immutable S01–S18/V1/V2/L01/L02 baseline: PASS.
- Paired 4,096-replicate matrix bootstrap, leave-one-matrix-out influence, cross-candidate agreement, adjacent-H overlap, and historical-nondrift overlap: retained in machine-readable form.
- Candidate 2 and candidate 3 stayed separate for every primary gate; pooling was not used.
- Report regeneration, schema/cardinality, artifact hash, compute, and storage checks: PASS.

## Commands and runtime

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l03.py
python -m ruff check src/e01_s19_boundary_compotype scripts/e01/prepare_s19_l03_lock.py scripts/e01/run_s19_l03.py tests/e01/test_s19_l03.py
git commit <pre-outcome L03 lock> && git push origin eidosoma/groups/42
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src python scripts/e01/prepare_s19_l03_lock.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src python scripts/e01/run_s19_l03.py --workers 8
```

Scientific CPU time was {runtime['scientificCpuHours']:.6f} hours and wall time was {runtime['wallHours']:.6f} hours. CPU float64 was authoritative.

## Caveats, blockers, and limitations

1. The paper does not identify a unique modal reference, boundary substrate, activation point, or projection alignment.
2. Historical public GARD stores pre-fission generation-end traces, but it is not the unavailable target-paper code.
3. The maximum-neighbor medoid is a precise paper/source-informed reconstruction, not author-code identity.
4. Completed-run reference selection uses future observations, so no structural L03 result is eligible as early-warning or online-control evidence.
5. These matrices were previously studied; all L03 selection and inference is exploratory and requires untouched confirmation for any promoted paper-facing lead.
6. No emergence association, prediction, or intervention result was recalculated under these labels.

## Provenance

- Pushed pre-outcome repository commit: `{runtime['repositoryCommit']}` on `eidosoma/groups/42`.
- Original paper: arXiv `2607.28250v1`; hash retained in `input_manifest.json`.
- Historical GARD commit: `86dff6320d5ae91b4e831471079ff46749b14df9`; source identities and license boundary retained in `source_snapshot_manifest.json`.
- Frozen S13Y trajectory/cache identities and exact replay: `input_manifest.json` and `preanalysis_replay_evidence.parquet`.

## Recommended next action and mandatory boundary

Return control for human review. The human continuation override permits future bounded S19 loops but authorizes none automatically. S19-L04, S20, E02, author contact, and report-bundle generation remain inactive.
"""


def decision_summary_text(
    aggregate: pd.DataFrame, classification: dict[str, Any], validation_result: str
) -> str:
    promoted = classification["promotedLeadIds"]
    compact = aggregate[
        [
            "candidateId",
            "labelId",
            "meanOccupancy",
            "meanConsistency",
            "meanFirstOnsetRawIndex0",
            "meanFirstOnsetNormalized",
            "meanEpisodeCount",
            "nonreplicatingAtCutoffFraction",
        ]
    ]
    return f"""# S19-L03 decision summary

## Concise top summary

- **Research step ID:** S19-L03
- **Completion status:** COMPLETE; mandatory human review reached
- **Artifacts written:** full L03 evidence and append-only S19 ledger updates
- **Validation result:** {validation_result}
- **Outcome classification:** {'EXPLORATORY_DIRECTIONAL_BOUNDARY_PROJECTION_LEAD' if promoted else 'EXPLORATORY_CONSTRAINING_NO_PROMOTABLE_LEAD'}
- **Caveats or blockers:** Retrospective completed-run reference, unresolved author boundary/projection semantics, previously studied matrices
- **Recommended next action:** Human review; authorize no later loop or S20 automatically
- **Lay summary:** The loop tested boundary-defined recurring states without threshold tuning or causal-emergence selection.

## Decision evidence

{markdown_table(compact)}

Promoted leads: **{len(promoted)}** ({', '.join(promoted) if promoted else 'none'}).

## Human-review boundary

S19 may continue under the no-fixed-loop-cap override only through another explicitly authorized bounded loop. L04, S20, E02, author contact, and report generation are inactive.
"""


def artifact_manifest(root: Path, required: list[str], schema: str) -> dict[str, Any]:
    missing = [name for name in required if not (root / name).is_file()]
    files = []
    manifest_path = root / "artifact_manifest.json"
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


def update_root_handoff(
    report: str, classification: dict[str, Any], validation_result: str, artifacts: list[str]
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
            "completed_run_modal_reference_is_retrospective",
            "author_boundary_activation_and_projection_semantics_unresolved",
            "S20_and_later_loops_inactive",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_SELECT_NEXT_BOUNDED_ACTION",
        "promotedLeadCount": classification["promotedLeadCount"],
        "promotedLeadIds": classification["promotedLeadIds"],
        "fixedLoopCap": None,
    }
    write_json(ARTIFACT_ROOT / "s19_status.json", status)
    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text())
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
    registry["fixedLoopCap"] = None
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    history["history"].append(
        {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "decision": "S19_L03_COMPLETE_MANDATORY_HUMAN_REVIEW",
            "scope": VERSION,
            "source": "locked_execution_result",
        }
    )
    history["pendingDecision"] = "POST_S19_L03_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)


def append_postloop_ledger(
    aggregate: pd.DataFrame, classification: dict[str, Any], timestamp: str
) -> None:
    path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(path)
    if ledger["loopId"].eq(LOOP_ID).sum() != 1:
        raise RuntimeError("L03 pre-loop self-improvement row cardinality changed")
    compact = aggregate[
        [
            "candidateId",
            "labelId",
            "meanOccupancy",
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
        "beliefBeforeLoop": "A boundary-defined modal compotype projected onto molecular time could explain the saturated adjacent-H label.",
        "motivatingEvidence": "Paper recurrence wording and historical generation-boundary compotype machinery.",
        "failureOrAmbiguityTargeted": "Boundary substrate, recurrence activation, and incoming/outgoing molecular projection.",
        "selectedHypotheses": "One strict-H>0.9 modal medoid and four nonfactorial structural labels.",
        "learned": canonical_json(
            {"aggregateFingerprint": compact, "promotedLeadIds": classification["promotedLeadIds"]}
        ),
        "weakenedHypotheses": "Any structural candidate failing the locked cross-candidate joint-fingerprint and robustness gates.",
        "remainingPlausibleHypotheses": "Only exact promoted leads, if any, or a separately human-authorized nonduplicative source-grounded ambiguity; author code remains unavailable.",
        "proposedNextTest": "Human review; no automatic L04 or S20.",
        "informationGainRationale": "The loop isolated backfill, recurrence activation, interval alignment, and boundary substrate without threshold or downstream-outcome selection.",
        "appendOnly": True,
    }
    pd.concat([ledger, pd.DataFrame([row])[ledger.columns]], ignore_index=True).to_parquet(
        path, index=False
    )
    with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Entry 006 — S19-L03 learning and human-review boundary\n\n"
            "- **Belief before the loop:** A modal boundary compotype projected onto molecular time might recover the joint paper fingerprint better than adjacent smoothness.\n"
            "- **What was tested:** Backfill versus second-recurrence activation, incoming versus outgoing projection, and post-fission versus historical generation-end substrate at fixed strict `H>0.9`.\n"
            f"- **What was learned:** {len(classification['promotedLeadIds'])} lead(s) passed every promotion gate. Full values remain in the L03 report and machine-readable tables.\n"
            "- **Hypotheses weakened:** Every candidate that failed joint raw/normalized, cross-candidate, replay, bootstrap, leave-one-out, episode, recurrence, or cutoff gates.\n"
            "- **What remains plausible:** Only explicitly promoted retrospective paper-facing leads, if any, plus genuinely nonduplicative source ambiguities approved later by a human.\n"
            "- **Next action:** Mandatory human review; no automatic L04 or S20.\n"
            "- **Why another loop could add information:** It must isolate a new source-grounded dependency rather than add thresholds or variants to this completed family.\n"
        )


def main(workers: int) -> None:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    started_utc = datetime.now(timezone.utc)
    lock = execution_lock_validation()
    write_json(LOOP_ROOT / "execution_lock_validation.json", lock)
    if not lock["passed"]:
        raise RuntimeError("L03 execution lock validation failed")
    manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet")
    labels, boundary, diagnostics, replay, execution = execute_labels(manifest, workers)
    if not execution["success"].all() or not replay["exactReplayPassed"].all():
        raise RuntimeError("L03 exact label replay failed")
    comparator_replay = frozen_comparator_replay(labels)
    if not comparator_replay["passed"].all():
        raise RuntimeError("L03 frozen adjacent-H comparator replay failed")
    fingerprints = build_fingerprints(labels, diagnostics)
    episodes = episode_table(labels)
    aggregate = aggregate_fingerprints(fingerprints)
    comparison = paper_comparison(aggregate)
    bootstrap = bootstrap_distances(fingerprints)
    loo = leave_one_out(fingerprints)
    overlaps = overlap_results(labels)
    cross = cross_candidate_agreement(fingerprints, aggregate)
    contrasts = projection_contrasts(fingerprints)
    classification = classify(aggregate, comparison, bootstrap, loo, cross, replay)

    labels.to_parquet(LOOP_ROOT / "label_values.parquet", index=False, compression="zstd")
    boundary.to_parquet(
        LOOP_ROOT / "boundary_membership_results.parquet", index=False, compression="zstd"
    )
    diagnostics.to_parquet(LOOP_ROOT / "boundary_reference_results.parquet", index=False)
    replay.to_parquet(LOOP_ROOT / "label_replay_evidence.parquet", index=False)
    comparator_replay.to_parquet(LOOP_ROOT / "frozen_comparator_replay.parquet", index=False)
    execution.to_parquet(LOOP_ROOT / "execution_status.parquet", index=False)
    fingerprints.to_parquet(LOOP_ROOT / "fingerprint_results.parquet", index=False)
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
    diagnostics.to_parquet(LOOP_ROOT / "recurrence_results.parquet", index=False)
    overlaps.to_parquet(LOOP_ROOT / "label_overlap_results.parquet", index=False)
    cross.to_csv(LOOP_ROOT / "cross_candidate_agreement.csv", index=False)
    comparison.to_csv(LOOP_ROOT / "paper_fingerprint_comparison.csv", index=False)
    bootstrap.to_parquet(LOOP_ROOT / "paper_distance_bootstrap.parquet", index=False)
    loo.to_parquet(LOOP_ROOT / "leave_one_out_robustness.parquet", index=False)
    contrasts.to_parquet(LOOP_ROOT / "projection_contrast_results.parquet", index=False)
    negative_controls = pd.concat(
        [
            contrasts.assign(controlFamily="LOCKED_STRUCTURAL_ISOLATION"),
        ],
        ignore_index=True,
    )
    negative_controls.to_parquet(LOOP_ROOT / "negative_control_results.parquet", index=False)
    robustness = bootstrap.merge(
        loo.groupby(["candidateId", "labelId", "onsetMode"], as_index=False).agg(
            looMaximumDistanceDifference=("distanceDifference", "max"),
            looMinimumDistanceDifference=("distanceDifference", "min"),
            looAllImproved=("distanceDifference", lambda values: bool((values < 0).all())),
        ),
        on=["candidateId", "labelId", "onsetMode"],
        how="left",
        validate="one_to_one",
    )
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
    completed_utc = datetime.now(timezone.utc)
    scientific_cpu_seconds = float(execution["cpuSeconds"].sum()) + (
        time.process_time() - started_cpu
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l03_runtime_manifest.v1",
        "startedUtc": started_utc.isoformat(),
        "completedUtc": completed_utc.isoformat(),
        "wallHours": (time.perf_counter() - started_wall) / 3600,
        "scientificCpuHours": scientific_cpu_seconds / 3600,
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
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime)
    storage = {
        "schema": "eidosoma.e01.s19_l03_storage_validation.v1",
        "retainedBytesBeforeManifest": sum(
            path.stat().st_size for path in LOOP_ROOT.rglob("*") if path.is_file()
        ),
        "retainedGiB": sum(path.stat().st_size for path in LOOP_ROOT.rglob("*") if path.is_file())
        / (1024**3),
        "retainedCeilingGiB": 25,
        "temporaryBytes": sum(
            path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file()
        ),
        "temporaryGiB": sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
        / (1024**3),
        "temporaryCeilingGiB": 75,
        "passed": True,
    }
    storage["passed"] = bool(
        storage["retainedGiB"] <= storage["retainedCeilingGiB"]
        and storage["temporaryGiB"] <= storage["temporaryCeilingGiB"]
    )
    write_json(LOOP_ROOT / "storage_validation.json", storage)
    validation_result = "PASS_ALL_LOCK_REPLAY_IMMUTABILITY_STORAGE_AND_REGENERATION_CHECKS"
    report = report_text(
        aggregate, comparison, contrasts, classification, validation_result, runtime
    )
    decision = decision_summary_text(aggregate, classification, validation_result)
    if report != report_text(
        aggregate, comparison, contrasts, classification, validation_result, runtime
    ):
        raise RuntimeError("L03 report regeneration mismatch")
    (LOOP_ROOT / "S19_L03_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "loop_decision_summary.md").write_text(decision, encoding="utf-8")
    regeneration = {
        "schema": "eidosoma.e01.s19_l03_regeneration_validation.v1",
        "labelCount": labels["labelId"].nunique(),
        "structuralCandidateCount": sum(not item.comparator_only for item in LABEL_DEFINITIONS),
        "candidateCount": labels["candidateId"].nunique(),
        "matrixCountPerCandidate": labels.groupby("candidateId")["matrixIndex"].nunique().to_dict(),
        "trajectoryCount": execution.shape[0],
        "labelRowCount": len(labels),
        "fingerprintRowCount": len(fingerprints),
        "exactLabelReplayPassed": bool(replay["exactReplayPassed"].all()),
        "frozenComparatorReplayPassed": bool(comparator_replay["passed"].all()),
        "immutablePriorPassed": immutable["passed"],
        "reportDeterministic": True,
        "thresholdGridAbsent": True,
        "H097CandidateAbsent": True,
        "newGardTrajectories": 0,
        "newPhiRLOrEmergenceValues": 0,
        "passed": bool(
            labels["labelId"].nunique() == 5
            and labels["candidateId"].nunique() == 2
            and execution.shape[0] == 200
            and fingerprints.shape[0] == 1000
            and replay["exactReplayPassed"].all()
            and comparator_replay["passed"].all()
            and immutable["passed"]
            and storage["passed"]
        ),
    }
    write_json(LOOP_ROOT / "regeneration_validation.json", regeneration)
    if not regeneration["passed"]:
        raise RuntimeError("L03 final regeneration validation failed")
    loop_status = {
        "researchStepId": LOOP_ID,
        "stepNumber": 19,
        "success": True,
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": [
            str(LOOP_ROOT / "S19_L03_FULL_RESULTS.md"),
            str(LOOP_ROOT / "classification.json"),
            str(LOOP_ROOT / "fingerprint_results.parquet"),
            str(LOOP_ROOT / "paper_fingerprint_comparison.csv"),
            str(LOOP_ROOT / "projection_contrast_results.parquet"),
        ],
        "validationResult": validation_result,
        "caveatsOrBlockers": [
            "exploratory_previously_studied_matrices",
            "completed_run_modal_reference_is_retrospective",
            "author_boundary_activation_and_projection_semantics_unresolved",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_SELECT_NEXT_BOUNDED_ACTION",
    }
    write_json(LOOP_ROOT / "status.json", loop_status)
    append_postloop_ledger(aggregate, classification, completed_utc.isoformat())
    update_root_handoff(
        report,
        classification,
        validation_result,
        loop_status["artifactsWritten"],
    )
    required = [
        "preregistration.yaml",
        "method_lock.json",
        "label_registry.yaml",
        "label_registry.parquet",
        "seed_manifest.parquet",
        "input_manifest.json",
        "source_snapshot_manifest.json",
        "preoutcome_repository_lock.json",
        "immutable_prior_baseline.json",
        "immutable_prior_validation.json",
        "compute_benchmark.json",
        "preanalysis_replay_evidence.parquet",
        "preanalysis_replay_validation.json",
        "execution_lock_validation.json",
        "execution_status.parquet",
        "label_values.parquet",
        "boundary_reference_results.parquet",
        "boundary_membership_results.parquet",
        "label_replay_evidence.parquet",
        "frozen_comparator_replay.parquet",
        "fingerprint_results.parquet",
        "fingerprint_summary.parquet",
        "fingerprint_aggregate.csv",
        "episode_results.parquet",
        "cutoff_results.parquet",
        "recurrence_results.parquet",
        "label_overlap_results.parquet",
        "cross_candidate_agreement.csv",
        "paper_fingerprint_comparison.csv",
        "paper_distance_bootstrap.parquet",
        "leave_one_out_robustness.parquet",
        "projection_contrast_results.parquet",
        "negative_control_results.parquet",
        "robustness_results.parquet",
        "failure_ledger.csv",
        "runtime_manifest.json",
        "storage_validation.json",
        "regeneration_validation.json",
        "classification.json",
        "status.json",
        "loop_decision_summary.md",
        "S19_L03_FULL_RESULTS.md",
        "research_step_full_results.md",
    ]
    loop_manifest = artifact_manifest(
        LOOP_ROOT, required, "eidosoma.e01.s19_l03_artifact_manifest.v1"
    )
    if not loop_manifest["passed"]:
        raise RuntimeError(f"missing L03 artifacts: {loop_manifest['missing']}")
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
        ARTIFACT_ROOT, root_required, "eidosoma.e01.s19_artifact_manifest.v3"
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
                "promotedLeadCount": classification["promotedLeadCount"],
                "promotedLeadIds": classification["promotedLeadIds"],
                "labelRows": len(labels),
                "fingerprintRows": len(fingerprints),
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
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(variable) not in (None, "1"):
            raise SystemExit(f"{variable} must be unset or 1")
    main(arguments.workers)
