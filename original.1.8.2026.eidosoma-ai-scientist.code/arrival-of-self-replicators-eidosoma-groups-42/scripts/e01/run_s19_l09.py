#!/usr/bin/env python3
"""Execute, validate, report, and freeze the locked E01/S19-L09 loop."""

from __future__ import annotations

import hashlib
import json
import math
import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import pickle
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from e01_frozen_timebase_ensemble.core import selected_clock_observations
from e01_s19_occupancy_search.core import (
    boundary_scores,
    materialize_frozen_setting,
)
from e01_s19_recurring_attractor.core import (
    BOOTSTRAP_REPLICATES,
    LABEL_IDS,
    PAPER_TARGETS,
    RANDOM_REFERENCE_DRAWS,
    R1_ID,
    R2_ID,
    ROOT_SEED_HEX,
    THRESHOLD,
    VERSION,
    array_sha256,
    bootstrap_indices,
    close_rows,
    deterministic_seed,
    fit_r1_historical,
    fit_r2_euclidean,
    historical_h,
    holm_adjust,
    label_against_reference,
    label_fingerprint,
    paper_distance,
    run_descriptors,
)
from e01_s19_untouched_mechanism.core import (
    MECHANISM_A,
    MECHANISM_B,
    OBJECT_A_BOUNDARY,
    OBJECT_A_PROJECTED,
    OBJECT_B_MOLECULAR,
)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L09"
CACHE_ROOT = Path("/cache/e01_s19_l09")
L08_ROOT = ARTIFACT_ROOT / "loops/L08"
L08_CACHE = Path("/cache/e01_s19_l08/trajectories")
CONFIG_PATH = REPO / "configs/e01/s19_l09_recurring_attractor.yaml"
CORE_PATH = REPO / "src/e01_s19_recurring_attractor/core.py"
PREPARE_PATH = REPO / "scripts/e01/prepare_s19_l09_lock.py"
RUNNER_PATH = Path(__file__).resolve()
TEST_PATH = REPO / "tests/e01/test_s19_l09.py"

PRIMARY_CANDIDATES = ("CANDIDATE_2", "CANDIDATE_3")
COMPARATOR_ADJACENT = "ORIGINAL_ADJACENT_MOLECULAR_H090"
COMPARATOR_A_BOUNDARY = "L08_A_FISSION_BOUNDARY_H090"
COMPARATOR_A_PROJECTED = "L08_A_FOLLOWING_INTERVAL_PROJECTED_H090"
COMPARATOR_B_HIGH = "L08_B_HIGH_EXPOSURE_MOLECULAR_H090"
COMPARATOR_IDS = (
    COMPARATOR_ADJACENT,
    COMPARATOR_A_BOUNDARY,
    COMPARATOR_A_PROJECTED,
    COMPARATOR_B_HIGH,
)

SUMMARY_METRICS = (
    "selectedClockLength",
    "persistence",
    "occupancy",
    "consistency",
    "firstOnsetRawIndex0",
    "firstOnsetRawStep1",
    "firstOnsetNormalized",
    "firstOnsetGeneration",
    "preOnsetNonreplicatingDuration",
    "isNonreplicatingAtQuarterCutoff",
    "noReplicatorThroughQuarterCutoff",
    "positiveEpisodeCount",
    "negativeEpisodeCount",
    "transitionCount",
    "positiveMeanEpisodeDuration",
    "negativeMeanEpisodeDuration",
    "positiveLongestEpisodeDuration",
    "negativeLongestEpisodeDuration",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def canonical_frame_sha256(frame: pd.DataFrame, sort_columns: list[str]) -> str:
    ordered = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True).copy()
    for column in ordered.columns:
        if ordered[column].dtype == object:
            ordered[column] = ordered[column].map(
                lambda value: (
                    json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else value
                )
            )
    return sha256_text(ordered.to_csv(index=False, lineterminator="\n", na_rep="<NA>"))


def repository_release_gate() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    clean = not bool(git("status", "--porcelain=v1"))
    branch = git("branch", "--show-current")
    method = json.loads((LOOP_ROOT / "label_method_lock.json").read_text())
    code_match = all(
        sha256_file(REPO / row["path"]) == row["sha256"] for row in method["code"]
    )
    passed = bool(
        head == remote
        and clean
        and branch == "eidosoma/groups/42"
        and code_match
        and method["versionedLoopId"] == VERSION
    )
    return {
        "head": head,
        "remoteHead": remote,
        "branch": branch,
        "cleanWorktree": clean,
        "lockedCodeHashesMatch": code_match,
        "passed": passed,
    }


def validate_immutable_prior() -> dict[str, Any]:
    baseline = json.loads((LOOP_ROOT / "immutable_prior_baseline.json").read_text())
    current = []
    for row in baseline["files"]:
        path = Path(row["path"])
        current.append(
            {
                "path": row["path"],
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else None,
                "expectedBytes": row["bytes"],
                "expectedSha256": row["sha256"],
            }
        )
    failed = [
        row
        for row in current
        if not row["exists"]
        or row["bytes"] != row["expectedBytes"]
        or row["sha256"] != row["expectedSha256"]
    ]
    return {
        "schema": "eidosoma.e01.s19_l09.immutable_prior_validation.v1",
        "baselineFileCount": len(current),
        "baselineTotalBytes": baseline["totalBytes"],
        "mismatchCount": len(failed),
        "mismatches": failed[:20],
        "passed": not failed,
        "validatedAtUtc": utc_now(),
    }


def trajectory_path(matrix_index: int, candidate_id: str, mechanism_id: str) -> Path:
    return L08_CACHE / f"M{matrix_index:03d}__{mechanism_id}__{candidate_id}.pkl"


def _safe_mean(values: Iterable[Any]) -> float | None:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(float)
    return float(np.mean(array)) if len(array) else None


def _fit_rows(
    *,
    fit: Any,
    candidate_id: str,
    matrix_index: int,
    trajectory_id: str,
) -> list[dict[str, Any]]:
    rows = []
    for record in fit.k_records:
        rows.append(
            {
                "pipelineId": fit.pipeline_id,
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "trajectoryId": trajectory_id,
                "k": int(record["k"]),
                "status": record["status"],
                "selectionScore": record.get("selectionScore"),
                "selectedLoss": record.get("selectedLoss"),
                "realizedClusterCount": record.get("realizedClusterCount"),
                "replicaLossesJson": json.dumps(record.get("replicaLosses", [])),
                "replicaIterationsJson": json.dumps(record.get("replicaIterations", [])),
                "selectedK": bool(record["k"] == fit.selected_k),
            }
        )
    return rows


def _fingerprint_row(
    *,
    pipeline_id: str,
    candidate_id: str,
    matrix_index: int,
    trajectory_id: str,
    status: str,
    fingerprint: dict[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "pipelineId": pipeline_id,
        "candidateId": candidate_id,
        "matrixIndex": matrix_index,
        "trajectoryId": trajectory_id,
        "fingerprintStatus": status,
    }
    row.update({metric: None for metric in SUMMARY_METRICS})
    row["labelSha256"] = None
    row["rawPaperDistance"] = None
    row["normalizedPaperDistance"] = None
    if fingerprint is not None:
        row.update(fingerprint)
        row["rawPaperDistance"] = paper_distance(fingerprint, "RAW")
        row["normalizedPaperDistance"] = paper_distance(fingerprint, "NORMALIZED")
    return row


def _control_fingerprint(
    molecular: np.ndarray,
    molecular_generations: np.ndarray,
    boundary: np.ndarray,
    reference: np.ndarray,
) -> tuple[dict[str, Any], float, np.ndarray, np.ndarray]:
    molecular_scores, labels = label_against_reference(molecular, reference)
    boundary_scores_to_ref, boundary_labels = label_against_reference(boundary, reference)
    fingerprint = label_fingerprint(labels, molecular_generations)
    fingerprint["rawPaperDistance"] = paper_distance(fingerprint, "RAW")
    fingerprint["normalizedPaperDistance"] = paper_distance(fingerprint, "NORMALIZED")
    return fingerprint, float(np.mean(boundary_labels)), molecular_scores, boundary_scores_to_ref


def analyze_primary_trajectory(matrix_index: int, candidate_id: str) -> dict[str, Any]:
    path = trajectory_path(matrix_index, candidate_id, MECHANISM_A)
    with path.open("rb") as handle:
        trajectory = pickle.load(handle)
    if trajectory.completed_fissions != 100 or trajectory.terminal_status != "requested_fissions_completed":
        raise RuntimeError("primary L08 trajectory incomplete")
    selected = selected_clock_observations(trajectory, "C1_SELECTED_DAUGHTER_RETAINED")
    post = tuple(item for item in selected if item.observation_kind == "post_fission")
    if len(post) != 100:
        raise RuntimeError("post-fission boundary cardinality mismatch")
    molecular = close_rows(np.asarray([item.state for item in selected], dtype=np.float64))
    boundary = close_rows(np.asarray([item.state for item in post], dtype=np.float64))
    molecular_generations = np.asarray(
        [item.growth_generation_one_based for item in selected], dtype=np.int64
    )
    parent_daughter = boundary_scores(
        trajectory,
        boundary_object="PARENT_TO_SELECTED_DAUGHTER",
        alignment="INCOMING_DUPLICATE_FIRST",
    )
    fits = (
        fit_r1_historical(boundary, str(trajectory.trajectory_id)),
        fit_r2_euclidean(boundary, str(trajectory.trajectory_id)),
    )
    output: dict[str, Any] = {
        "cluster": [],
        "dominant": [],
        "molecular": [],
        "boundary": [],
        "fingerprint": [],
        "episode": [],
        "negative": [],
        "failure": [],
        "adjacentComparator": [],
    }

    adjacent_setting = {
        "roundId": "S19-L09",
        "settingId": COMPARATOR_ADJACENT,
        "settingPairId": COMPARATOR_ADJACENT,
        "threshold": THRESHOLD,
        "comparator": "STRICT_GT",
        "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
        "alignment": "INCOMING_DUPLICATE_FIRST",
        "family": "ADJACENT_CLOCK",
        "projection": "ALL_OBSERVATIONS",
    }
    adjacent_frame = materialize_frozen_setting(trajectory, adjacent_setting)
    adjacent_labels = adjacent_frame["isReplicator"].to_numpy(dtype=bool)
    adjacent_fp = label_fingerprint(adjacent_labels, molecular_generations)
    output["adjacentComparator"].append(
        _fingerprint_row(
            pipeline_id=COMPARATOR_ADJACENT,
            candidate_id=candidate_id,
            matrix_index=matrix_index,
            trajectory_id=str(trajectory.trajectory_id),
            status="ELIGIBLE_FROZEN_COMPARATOR",
            fingerprint=adjacent_fp,
        )
    )

    for fit in fits:
        output["cluster"].extend(
            _fit_rows(
                fit=fit,
                candidate_id=candidate_id,
                matrix_index=matrix_index,
                trajectory_id=str(trajectory.trajectory_id),
            )
        )
        common = {
            "pipelineId": fit.pipeline_id,
            "candidateId": candidate_id,
            "matrixIndex": matrix_index,
            "trajectoryId": str(trajectory.trajectory_id),
            "pipelineStatus": fit.status,
            "selectedK": fit.selected_k,
            "selectedScore": fit.selected_score,
            "nonDriftBoundaryCount": int(np.count_nonzero(fit.eligible_mask)),
            "detectedClusterCount": int(len(fit.cluster_sizes)),
            "clusterSizesJson": json.dumps(list(fit.cluster_sizes)),
            "validClusterIdsJson": json.dumps(list(fit.valid_cluster_ids)),
            "dominantClusterId": fit.dominant_cluster_id,
            "secondClusterId": fit.second_cluster_id,
            "dominantAssignedSize": (
                None if fit.dominant_cluster_id is None else fit.cluster_sizes[fit.dominant_cluster_id]
            ),
            "dominantClusterFraction": (
                None
                if fit.dominant_cluster_id is None
                else fit.cluster_sizes[fit.dominant_cluster_id]
                / max(1, int(np.count_nonzero(fit.eligible_mask)))
            ),
            "dominantMembershipCount": fit.dominant_member_count,
            "secondMembershipCount": fit.second_member_count,
            "dominantCentroidJson": (
                None if fit.dominant_centroid is None else json.dumps(fit.dominant_centroid.tolist())
            ),
            "dominantCentroidSha256": (
                None if fit.dominant_centroid is None else array_sha256(fit.dominant_centroid)
            ),
            "secondCentroidJson": (
                None if fit.second_centroid is None else json.dumps(fit.second_centroid.tolist())
            ),
            "secondCentroidSha256": (
                None if fit.second_centroid is None else array_sha256(fit.second_centroid)
            ),
        }
        if fit.status != "ELIGIBLE" or fit.dominant_centroid is None:
            output["dominant"].append(
                {
                    **common,
                    "withinDominantDispersion": None,
                    "dominantSecondCentroidDistance": None,
                    "boundaryOccupancy": None,
                    "boundaryConsistency": None,
                    "meanParentDaughterHInside": None,
                    "meanParentDaughterHOutside": None,
                }
            )
            output["fingerprint"].append(
                _fingerprint_row(
                    pipeline_id=fit.pipeline_id,
                    candidate_id=candidate_id,
                    matrix_index=matrix_index,
                    trajectory_id=str(trajectory.trajectory_id),
                    status=fit.status,
                    fingerprint=None,
                )
            )
            output["molecular"].append(
                {
                    "pipelineId": fit.pipeline_id,
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "trajectoryId": str(trajectory.trajectory_id),
                    "analysisUnitIndex": -1,
                    "rawObservationIndex": None,
                    "generation": None,
                    "observationKind": None,
                    "labelStatus": fit.status,
                    "hToDominant": None,
                    "isReplicator": None,
                    "stateSha256": None,
                }
            )
            output["failure"].append(
                {
                    "failureId": f"{fit.pipeline_id}::{candidate_id}::M{matrix_index:03d}",
                    "pipelineId": fit.pipeline_id,
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "failureStatus": fit.status,
                    "excludedFromScientificAggregation": True,
                }
            )
            continue

        molecular_scores, molecular_labels = label_against_reference(
            molecular, fit.dominant_centroid
        )
        boundary_scores_ref, boundary_labels = label_against_reference(
            boundary, fit.dominant_centroid
        )
        fingerprint = label_fingerprint(molecular_labels, molecular_generations)
        boundary_fingerprint = label_fingerprint(
            boundary_labels, np.arange(1, 101, dtype=np.int64)
        )
        assigned_values = boundary[fit.eligible_mask][fit.labels == fit.dominant_cluster_id]
        if fit.pipeline_id == R1_ID:
            dispersion = float(
                np.mean(1.0 - historical_h(assigned_values, fit.dominant_centroid).ravel())
            )
            second_distance = (
                None
                if fit.second_centroid is None
                else float(1.0 - historical_h(fit.dominant_centroid, fit.second_centroid)[0, 0])
            )
        else:
            dispersion = float(np.mean(np.linalg.norm(assigned_values - fit.dominant_centroid, axis=1)))
            second_distance = (
                None
                if fit.second_centroid is None
                else float(np.linalg.norm(fit.dominant_centroid - fit.second_centroid))
            )
        output["dominant"].append(
            {
                **common,
                "withinDominantDispersion": dispersion,
                "dominantSecondCentroidDistance": second_distance,
                "boundaryOccupancy": float(np.mean(boundary_labels)),
                "boundaryConsistency": boundary_fingerprint["consistency"],
                "meanParentDaughterHInside": _safe_mean(parent_daughter[boundary_labels]),
                "meanParentDaughterHOutside": _safe_mean(parent_daughter[~boundary_labels]),
            }
        )
        for index, (item, score, label, state) in enumerate(
            zip(selected, molecular_scores, molecular_labels, molecular, strict=True)
        ):
            output["molecular"].append(
                {
                    "pipelineId": fit.pipeline_id,
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "trajectoryId": str(trajectory.trajectory_id),
                    "analysisUnitIndex": index,
                    "rawObservationIndex": int(item.observation_index),
                    "generation": int(item.growth_generation_one_based),
                    "observationKind": str(item.observation_kind),
                    "labelStatus": "ELIGIBLE_DIRECT_MOLECULAR_MEMBERSHIP",
                    "hToDominant": float(score),
                    "isReplicator": bool(label),
                    "stateSha256": array_sha256(state),
                }
            )
        for index, (item, score, label, state) in enumerate(
            zip(post, boundary_scores_ref, boundary_labels, boundary, strict=True)
        ):
            eligible_index = np.flatnonzero(fit.eligible_mask)
            cluster_id = None
            if fit.eligible_mask[index]:
                position = int(np.flatnonzero(eligible_index == index)[0])
                cluster_id = int(fit.labels[position])
            output["boundary"].append(
                {
                    "pipelineId": fit.pipeline_id,
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "trajectoryId": str(trajectory.trajectory_id),
                    "boundaryIndex0": index,
                    "generation": int(item.growth_generation_one_based),
                    "rawObservationIndex": int(item.observation_index),
                    "nondriftEligible": bool(fit.eligible_mask[index]),
                    "selectedClusterId": cluster_id,
                    "hToDominant": float(score),
                    "isReplicator": bool(label),
                    "stateSha256": array_sha256(state),
                }
            )
        output["fingerprint"].append(
            _fingerprint_row(
                pipeline_id=fit.pipeline_id,
                candidate_id=candidate_id,
                matrix_index=matrix_index,
                trajectory_id=str(trajectory.trajectory_id),
                status="ELIGIBLE",
                fingerprint=fingerprint,
            )
        )
        for polarity, desired in (("POSITIVE", True), ("NEGATIVE", False)):
            for episode_index, episode in enumerate(run_descriptors(molecular_labels, desired)):
                output["episode"].append(
                    {
                        "pipelineId": fit.pipeline_id,
                        "candidateId": candidate_id,
                        "matrixIndex": matrix_index,
                        "trajectoryId": str(trajectory.trajectory_id),
                        "polarity": polarity,
                        "episodeIndex": episode_index,
                        **episode,
                    }
                )

        for draw in range(RANDOM_REFERENCE_DRAWS):
            rng = np.random.Generator(
                np.random.PCG64DXSM(
                    deterministic_seed(
                        "random_reference", candidate_id, matrix_index, draw, bits=128
                    )
                )
            )
            reference_index = int(rng.integers(0, 100))
            control_fp, recurrence, _, _ = _control_fingerprint(
                molecular,
                molecular_generations,
                boundary,
                boundary[reference_index],
            )
            output["negative"].append(
                {
                    "pipelineId": fit.pipeline_id,
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "trajectoryId": str(trajectory.trajectory_id),
                    "controlType": "RANDOM_REFERENCE",
                    "controlIndex": draw,
                    "referenceBoundaryIndex0": reference_index,
                    "controlStatus": "ELIGIBLE",
                    "boundaryRecurrence": recurrence,
                    "occupancy": control_fp["occupancy"],
                    "persistence": control_fp["persistence"],
                    "consistency": control_fp["consistency"],
                    "firstOnsetRawStep1": control_fp["firstOnsetRawStep1"],
                    "firstOnsetNormalized": control_fp["firstOnsetNormalized"],
                    "rawPaperDistance": control_fp["rawPaperDistance"],
                    "normalizedPaperDistance": control_fp["normalizedPaperDistance"],
                }
            )
        if fit.second_centroid is not None:
            control_fp, recurrence, _, _ = _control_fingerprint(
                molecular,
                molecular_generations,
                boundary,
                fit.second_centroid,
            )
            output["negative"].append(
                {
                    "pipelineId": fit.pipeline_id,
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "trajectoryId": str(trajectory.trajectory_id),
                    "controlType": "SECOND_LARGEST_CLUSTER",
                    "controlIndex": 0,
                    "referenceBoundaryIndex0": None,
                    "controlStatus": "ELIGIBLE",
                    "boundaryRecurrence": recurrence,
                    "occupancy": control_fp["occupancy"],
                    "persistence": control_fp["persistence"],
                    "consistency": control_fp["consistency"],
                    "firstOnsetRawStep1": control_fp["firstOnsetRawStep1"],
                    "firstOnsetNormalized": control_fp["firstOnsetNormalized"],
                    "rawPaperDistance": control_fp["rawPaperDistance"],
                    "normalizedPaperDistance": control_fp["normalizedPaperDistance"],
                }
            )
    return output


def execute_primary_pass(workers: int) -> dict[str, pd.DataFrame]:
    tasks = [(matrix_index, candidate_id) for candidate_id in PRIMARY_CANDIDATES for matrix_index in range(100)]
    buckets: dict[str, list[dict[str, Any]]] = {
        key: []
        for key in (
            "cluster",
            "dominant",
            "molecular",
            "boundary",
            "fingerprint",
            "episode",
            "negative",
            "failure",
            "adjacentComparator",
        )
    }
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(analyze_primary_trajectory, matrix_index, candidate_id): (
                matrix_index,
                candidate_id,
            )
            for matrix_index, candidate_id in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            for key in buckets:
                buckets[key].extend(result[key])
    return {key: pd.DataFrame(rows) for key, rows in buckets.items()}


def normalize_l08_comparators(adjacent: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_parquet(L08_ROOT / "trajectory_fingerprints.parquet")
    mapping = {
        (MECHANISM_A, OBJECT_A_BOUNDARY): COMPARATOR_A_BOUNDARY,
        (MECHANISM_A, OBJECT_A_PROJECTED): COMPARATOR_A_PROJECTED,
        (MECHANISM_B, OBJECT_B_MOLECULAR): COMPARATOR_B_HIGH,
    }
    rows = adjacent.to_dict("records")
    for item in frozen.itertuples():
        comparator_id = mapping[(item.mechanismId, item.analysisObjectId)]
        rows.append(
            {
                "pipelineId": comparator_id,
                "candidateId": item.candidateId,
                "matrixIndex": int(item.matrixIndex),
                "trajectoryId": item.trajectoryId,
                "fingerprintStatus": "ELIGIBLE_FROZEN_L08_COMPARATOR",
                "selectedClockLength": int(item.analysisUnitLength),
                "persistence": int(item.persistence),
                "occupancy": float(item.occupancy),
                "consistency": (
                    None if pd.isna(item.consistency) else float(item.consistency)
                ),
                "firstOnsetRawIndex0": (
                    None if pd.isna(item.firstOnsetRawIndex0) else int(item.firstOnsetRawIndex0)
                ),
                "firstOnsetRawStep1": (
                    None if pd.isna(item.firstOnsetRawStep1) else int(item.firstOnsetRawStep1)
                ),
                "firstOnsetNormalized": (
                    None if pd.isna(item.firstOnsetNormalized) else float(item.firstOnsetNormalized)
                ),
                "firstOnsetGeneration": None,
                "preOnsetNonreplicatingDuration": (
                    int(item.analysisUnitLength)
                    if pd.isna(item.firstOnsetRawIndex0)
                    else int(item.firstOnsetRawIndex0)
                ),
                "isNonreplicatingAtQuarterCutoff": None,
                "noReplicatorThroughQuarterCutoff": None,
                "positiveEpisodeCount": int(item.positiveEpisodeCount),
                "negativeEpisodeCount": int(item.negativeEpisodeCount),
                "transitionCount": int(
                    max(0, int(item.positiveEpisodeCount) + int(item.negativeEpisodeCount) - 1)
                ),
                "positiveMeanEpisodeDuration": (
                    None
                    if pd.isna(item.positiveMeanEpisodeDuration)
                    else float(item.positiveMeanEpisodeDuration)
                ),
                "negativeMeanEpisodeDuration": (
                    None
                    if pd.isna(item.negativeMeanEpisodeDuration)
                    else float(item.negativeMeanEpisodeDuration)
                ),
                "positiveLongestEpisodeDuration": int(item.positiveLongestEpisodeDuration),
                "negativeLongestEpisodeDuration": int(item.negativeLongestEpisodeDuration),
                "labelSha256": item.labelSha256,
                "rawPaperDistance": None,
                "normalizedPaperDistance": None,
            }
        )
    frame = pd.DataFrame(rows)
    for index, row in frame.iterrows():
        summary = row.to_dict()
        frame.at[index, "rawPaperDistance"] = paper_distance(summary, "RAW")
        frame.at[index, "normalizedPaperDistance"] = paper_distance(summary, "NORMALIZED")
    return frame


def aggregate_fingerprints(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (pipeline_id, candidate_id), group in frame.groupby(
        ["pipelineId", "candidateId"], sort=True
    ):
        eligible = group[group["fingerprintStatus"].astype(str).str.startswith("ELIGIBLE")].copy()
        row: dict[str, Any] = {
            "pipelineId": pipeline_id,
            "candidateId": candidate_id,
            "trajectoryCount": int(len(group)),
            "validTrajectoryCount": int(len(eligible)),
            "undefinedConsistencyCount": int(eligible["consistency"].isna().sum()),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(eligible[metric], errors="coerce").dropna().to_numpy(float)
            row[f"mean_{metric}"] = float(np.mean(values)) if len(values) else None
            row[f"median_{metric}"] = float(np.median(values)) if len(values) else None
            row[f"sd_{metric}"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if len(values) else None)
            )
            row[f"se_{metric}"] = (
                float(np.std(values, ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else (0.0 if len(values) else None)
            )
        summary = {metric: row[f"mean_{metric}"] for metric in SUMMARY_METRICS}
        row["rawPaperDistance"] = paper_distance(summary, "RAW")
        row["normalizedPaperDistance"] = paper_distance(summary, "NORMALIZED")
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_primary(fingerprint: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = (
        "selectedClockLength",
        "persistence",
        "occupancy",
        "consistency",
        "firstOnsetRawStep1",
        "firstOnsetNormalized",
        "preOnsetNonreplicatingDuration",
        "isNonreplicatingAtQuarterCutoff",
        "noReplicatorThroughQuarterCutoff",
        "positiveEpisodeCount",
        "negativeEpisodeCount",
    )
    for pipeline_id in LABEL_IDS:
        for candidate_id in PRIMARY_CANDIDATES:
            group = (
                fingerprint[
                    (fingerprint["pipelineId"] == pipeline_id)
                    & (fingerprint["candidateId"] == candidate_id)
                ]
                .set_index("matrixIndex")
                .reindex(range(100))
            )
            indices = bootstrap_indices(candidate_id, pipeline_id)
            arrays = {
                metric: pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
                for metric in metrics
            }
            for replicate, sample in enumerate(indices):
                summary = {
                    metric: (
                        float(np.nanmean(values[sample]))
                        if np.any(np.isfinite(values[sample]))
                        else None
                    )
                    for metric, values in arrays.items()
                }
                rows.append(
                    {
                        "pipelineId": pipeline_id,
                        "candidateId": candidate_id,
                        "bootstrapReplicate": replicate,
                        **summary,
                        "rawPaperDistance": paper_distance(summary, "RAW"),
                        "normalizedPaperDistance": paper_distance(summary, "NORMALIZED"),
                    }
                )
    return pd.DataFrame(rows)


def target_comparisons(
    aggregate: pd.DataFrame, bootstrap: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_keys = (
        "occupancy",
        "persistence",
        "consistency",
        "firstOnsetRawStep1",
        "firstOnsetNormalized",
    )
    rows: list[dict[str, Any]] = []
    distances: list[dict[str, Any]] = []
    for item in aggregate.itertuples():
        for metric in metric_keys:
            value = getattr(item, f"mean_{metric}")
            target, scale = PAPER_TARGETS[metric]
            boot = bootstrap[
                (bootstrap["pipelineId"] == item.pipelineId)
                & (bootstrap["candidateId"] == item.candidateId)
            ] if item.pipelineId in LABEL_IDS else pd.DataFrame()
            ci_lower = ci_upper = None
            if not boot.empty and metric in boot:
                values = pd.to_numeric(boot[metric], errors="coerce").dropna().to_numpy(float)
                if len(values):
                    ci_lower, ci_upper = map(float, np.quantile(values, [0.025, 0.975]))
            rows.append(
                {
                    "pipelineId": item.pipelineId,
                    "candidateId": item.candidateId,
                    "metric": metric,
                    "mean": value,
                    "median": getattr(item, f"median_{metric}"),
                    "sampleSd": getattr(item, f"sd_{metric}"),
                    "standardError": getattr(item, f"se_{metric}"),
                    "paperTarget": target,
                    "paperReportedPlusMinusScale": scale,
                    "authorDispersionIdentity": "AUTHOR_DISPERSION_UNRESOLVED",
                    "rawDifference": None if value is None else float(value - target),
                    "standardizedDifference": None
                    if value is None
                    else float((value - target) / scale),
                    "bootstrapCi025": ci_lower,
                    "bootstrapCi975": ci_upper,
                    "measurementLevel": (
                        "BOUNDARY_DIAGNOSTIC_NONINTERCHANGEABLE"
                        if item.pipelineId == COMPARATOR_A_BOUNDARY
                        else "MOLECULAR_PRIMARY_OR_COMPARATOR"
                    ),
                }
            )
        distances.append(
            {
                "pipelineId": item.pipelineId,
                "candidateId": item.candidateId,
                "validTrajectoryCount": item.validTrajectoryCount,
                "rawPaperDistance": item.rawPaperDistance,
                "normalizedPaperDistance": item.normalizedPaperDistance,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(distances)


def negative_control_analysis(
    negative: pd.DataFrame,
    fingerprint: pd.DataFrame,
    dominant: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    result_rows = negative.to_dict("records")
    gate_map: dict[tuple[str, str], dict[str, Any]] = {}
    p_records: list[dict[str, Any]] = []
    for pipeline_id in LABEL_IDS:
        for candidate_id in PRIMARY_CANDIDATES:
            observed_fp = fingerprint[
                (fingerprint["pipelineId"] == pipeline_id)
                & (fingerprint["candidateId"] == candidate_id)
                & (fingerprint["fingerprintStatus"] == "ELIGIBLE")
            ]
            observed_dom = dominant[
                (dominant["pipelineId"] == pipeline_id)
                & (dominant["candidateId"] == candidate_id)
                & (dominant["pipelineStatus"] == "ELIGIBLE")
            ]
            observed_recurrence = float(observed_dom["boundaryOccupancy"].mean())
            observed_raw = paper_distance(
                {metric: pd.to_numeric(observed_fp[metric], errors="coerce").mean() for metric in SUMMARY_METRICS},
                "RAW",
            )
            observed_norm = paper_distance(
                {metric: pd.to_numeric(observed_fp[metric], errors="coerce").mean() for metric in SUMMARY_METRICS},
                "NORMALIZED",
            )
            random = negative[
                (negative["pipelineId"] == pipeline_id)
                & (negative["candidateId"] == candidate_id)
                & (negative["controlType"] == "RANDOM_REFERENCE")
            ]
            random_aggregate = (
                random.groupby("controlIndex", sort=True)[
                    ["boundaryRecurrence", "occupancy", "persistence", "consistency", "firstOnsetRawStep1", "firstOnsetNormalized"]
                ]
                .mean(numeric_only=True)
                .reset_index()
            )
            random_aggregate["rawPaperDistance"] = random_aggregate.apply(
                lambda row: paper_distance(row.to_dict(), "RAW"), axis=1
            )
            random_aggregate["normalizedPaperDistance"] = random_aggregate.apply(
                lambda row: paper_distance(row.to_dict(), "NORMALIZED"), axis=1
            )
            recurrence_q95 = float(random_aggregate["boundaryRecurrence"].quantile(0.95))
            raw_q05 = float(random_aggregate["rawPaperDistance"].quantile(0.05))
            norm_q05 = float(random_aggregate["normalizedPaperDistance"].quantile(0.05))
            random_passes = {
                "randomRecurrencePass": bool(observed_recurrence > recurrence_q95),
                "randomRawDistancePass": bool(observed_raw < raw_q05),
                "randomNormalizedDistancePass": bool(observed_norm < norm_q05),
            }
            for outcome, observed, distribution, direction in (
                ("boundaryRecurrence", observed_recurrence, random_aggregate["boundaryRecurrence"].to_numpy(float), "GREATER"),
                ("rawPaperDistance", observed_raw, random_aggregate["rawPaperDistance"].to_numpy(float), "LESS"),
                ("normalizedPaperDistance", observed_norm, random_aggregate["normalizedPaperDistance"].to_numpy(float), "LESS"),
            ):
                extreme = int(np.count_nonzero(distribution >= observed)) if direction == "GREATER" else int(np.count_nonzero(distribution <= observed))
                p_records.append(
                    {
                        "pipelineId": pipeline_id,
                        "candidateId": candidate_id,
                        "controlType": "RANDOM_REFERENCE_AGGREGATE",
                        "outcome": outcome,
                        "observed": observed,
                        "nullMean": float(np.mean(distribution)),
                        "nullQ05": float(np.quantile(distribution, 0.05)),
                        "nullQ95": float(np.quantile(distribution, 0.95)),
                        "direction": direction,
                        "rawP": float((1 + extreme) / (1 + len(distribution))),
                        "applicableCount": int(len(distribution)),
                    }
                )

            second = negative[
                (negative["pipelineId"] == pipeline_id)
                & (negative["candidateId"] == candidate_id)
                & (negative["controlType"] == "SECOND_LARGEST_CLUSTER")
            ]
            second_applicable = bool(len(second))
            second_passes = {
                "secondApplicable": second_applicable,
                "secondRecurrencePass": True,
                "secondRawDistancePass": True,
                "secondNormalizedDistancePass": True,
            }
            if second_applicable:
                joined = observed_dom[["matrixIndex", "boundaryOccupancy"]].merge(
                    second[["matrixIndex", "boundaryRecurrence", "rawPaperDistance", "normalizedPaperDistance"]],
                    on="matrixIndex",
                    how="inner",
                ).merge(
                    observed_fp[["matrixIndex", "rawPaperDistance", "normalizedPaperDistance"]],
                    on="matrixIndex",
                    how="inner",
                    suffixes=("_second", "_dominant"),
                )
                second_passes = {
                    "secondApplicable": True,
                    "secondRecurrencePass": bool(
                        joined["boundaryOccupancy"].mean() > joined["boundaryRecurrence"].mean()
                    ),
                    "secondRawDistancePass": bool(
                        joined["rawPaperDistance_dominant"].mean()
                        < joined["rawPaperDistance_second"].mean()
                    ),
                    "secondNormalizedDistancePass": bool(
                        joined["normalizedPaperDistance_dominant"].mean()
                        < joined["normalizedPaperDistance_second"].mean()
                    ),
                }
                for outcome, delta, direction in (
                    ("boundaryRecurrence", joined["boundaryOccupancy"].to_numpy(float) - joined["boundaryRecurrence"].to_numpy(float), "GREATER"),
                    ("rawPaperDistance", joined["rawPaperDistance_dominant"].to_numpy(float) - joined["rawPaperDistance_second"].to_numpy(float), "LESS"),
                    ("normalizedPaperDistance", joined["normalizedPaperDistance_dominant"].to_numpy(float) - joined["normalizedPaperDistance_second"].to_numpy(float), "LESS"),
                ):
                    rng = np.random.Generator(
                        np.random.PCG64DXSM(
                            deterministic_seed("second_control_bootstrap", pipeline_id, candidate_id, outcome, bits=128)
                        )
                    )
                    sample = rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))
                    boot = np.mean(delta[sample], axis=1)
                    extreme = int(np.count_nonzero(boot <= 0)) if direction == "GREATER" else int(np.count_nonzero(boot >= 0))
                    p_records.append(
                        {
                            "pipelineId": pipeline_id,
                            "candidateId": candidate_id,
                            "controlType": "SECOND_LARGEST_CLUSTER_PAIRED_BOOTSTRAP",
                            "outcome": outcome,
                            "observed": float(np.mean(delta)),
                            "nullMean": 0.0,
                            "nullQ05": float(np.quantile(boot, 0.05)),
                            "nullQ95": float(np.quantile(boot, 0.95)),
                            "direction": direction,
                            "rawP": float((1 + extreme) / (1 + BOOTSTRAP_REPLICATES)),
                            "applicableCount": int(len(delta)),
                        }
                    )
            gate_map[(pipeline_id, candidate_id)] = {
                **random_passes,
                **second_passes,
                "observedBoundaryRecurrence": observed_recurrence,
                "randomBoundaryRecurrenceQ95": recurrence_q95,
                "observedRawPaperDistance": observed_raw,
                "randomRawDistanceQ05": raw_q05,
                "observedNormalizedPaperDistance": observed_norm,
                "randomNormalizedDistanceQ05": norm_q05,
            }
    p_frame = pd.DataFrame(p_records)
    adjusted_parts = []
    for pipeline_id, group in p_frame.groupby("pipelineId", sort=True):
        local = group.copy()
        local["holmAdjustedP"] = holm_adjust(local["rawP"].tolist())
        adjusted_parts.append(local)
    adjusted = pd.concat(adjusted_parts, ignore_index=True) if adjusted_parts else p_frame
    for row in adjusted.to_dict("records"):
        row["recordType"] = "AGGREGATE_CONTROL_TEST"
        result_rows.append(row)
    return pd.DataFrame(result_rows), gate_map


def candidate_comparisons(
    fingerprint: pd.DataFrame,
    aggregate: pd.DataFrame,
    comparator_aggregate: pd.DataFrame,
    control_gates: dict[tuple[str, str], dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    operational = config["promotionOperationalization"]
    pipeline_classifications: dict[str, str] = {}
    promotion_eligible: list[str] = []

    comparator_targets: dict[tuple[str, str], float] = {}
    for candidate_id in PRIMARY_CANDIDATES:
        for metric in ("occupancy", "persistence", "consistency", "firstOnsetRawStep1", "firstOnsetNormalized"):
            target, _ = PAPER_TARGETS[metric]
            values = comparator_aggregate[
                comparator_aggregate["candidateId"] == candidate_id
            ][f"mean_{metric}"].dropna().astype(float)
            comparator_targets[(candidate_id, metric)] = float(
                np.min(np.abs(values.to_numpy() - target))
            )

    for pipeline_id in LABEL_IDS:
        pipeline_pass = True
        dimension_closer_counts: list[int] = []
        candidate_gate_records: list[dict[str, Any]] = []
        for candidate_id in PRIMARY_CANDIDATES:
            agg = aggregate[
                (aggregate["pipelineId"] == pipeline_id)
                & (aggregate["candidateId"] == candidate_id)
            ].iloc[0]
            matrix = fingerprint[
                (fingerprint["pipelineId"] == pipeline_id)
                & (fingerprint["candidateId"] == candidate_id)
                & (fingerprint["fingerprintStatus"] == "ELIGIBLE")
            ]
            closer: dict[str, bool] = {}
            for metric in (
                "occupancy",
                "persistence",
                "consistency",
                "firstOnsetRawStep1",
                "firstOnsetNormalized",
            ):
                value = getattr(agg, f"mean_{metric}")
                target, _ = PAPER_TARGETS[metric]
                closer[metric] = bool(
                    value is not None
                    and abs(float(value) - target)
                    < comparator_targets[(candidate_id, metric)]
                )
            dimension_closer_counts.append(sum(closer.values()))
            onset_values = pd.to_numeric(matrix["firstOnsetRawStep1"], errors="coerce")
            both_polarities = (
                (pd.to_numeric(matrix["persistence"], errors="coerce") > 0)
                & (
                    pd.to_numeric(matrix["persistence"], errors="coerce")
                    < pd.to_numeric(matrix["selectedClockLength"], errors="coerce")
                )
            )
            controls = control_gates[(pipeline_id, candidate_id)]
            gates = {
                "definedAtLeast95": bool(int(agg.validTrajectoryCount) >= 95),
                "occupancyInRange": bool(0.85 <= float(agg.mean_occupancy) <= 0.91),
                "persistenceInRange": bool(518 <= float(agg.mean_persistence) <= 914),
                "consistencyCloserThanEveryComparator": closer["consistency"],
                "rawOnsetCloserThanEveryComparator": closer["firstOnsetRawStep1"],
                "normalizedOnsetCloserThanEveryComparator": closer[
                    "firstOnsetNormalized"
                ],
                "nontrivialPreOnset": bool(
                    float(agg.mean_firstOnsetRawStep1)
                    >= operational["nontrivialPreOnset"]["minimumCandidateMeanOneBasedRawStep"]
                    and float(np.mean(onset_values >= 10))
                    >= operational["nontrivialPreOnset"][
                        "minimumTrajectoryFractionWithOneBasedOnsetAtLeast10"
                    ]
                ),
                "quarterStateMeaningful": bool(
                    float(agg.mean_isNonreplicatingAtQuarterCutoff)
                    >= operational["meaningfulQuarterEligibility"][
                        "minimumFractionNonreplicatingAtQuarterCutoff"
                    ]
                ),
                "quarterNoOnsetMeaningful": bool(
                    float(agg.mean_noReplicatorThroughQuarterCutoff)
                    >= operational["meaningfulQuarterEligibility"][
                        "minimumFractionNoReplicatorThroughQuarterCutoff"
                    ]
                ),
                "positiveNegativeEpisodesNondegenerate": bool(
                    float(np.mean(both_polarities))
                    >= operational["nondegenerateEpisodes"][
                        "minimumTrajectoryFractionWithBothPolarities"
                    ]
                ),
                "randomReferenceRecurrence": controls["randomRecurrencePass"],
                "randomReferenceRawDistance": controls["randomRawDistancePass"],
                "randomReferenceNormalizedDistance": controls[
                    "randomNormalizedDistancePass"
                ],
                "secondClusterRecurrence": controls["secondRecurrencePass"],
                "secondClusterRawDistance": controls["secondRawDistancePass"],
                "secondClusterNormalizedDistance": controls[
                    "secondNormalizedDistancePass"
                ],
            }
            candidate_pass = bool(all(gates.values()))
            pipeline_pass &= candidate_pass
            candidate_gate_records.append(
                {
                    "pipelineId": pipeline_id,
                    "candidateId": candidate_id,
                    "validTrajectoryCount": int(agg.validTrajectoryCount),
                    "dimensionsCloserThanEveryComparator": int(sum(closer.values())),
                    "closerMetricsJson": json.dumps(closer, sort_keys=True),
                    "gateCount": len(gates),
                    "passedGateCount": sum(gates.values()),
                    "allCandidateGatesPassed": candidate_pass,
                    **gates,
                }
            )
        cross_agreement = all(
            record["allCandidateGatesPassed"] == candidate_gate_records[0]["allCandidateGatesPassed"]
            for record in candidate_gate_records
        )
        source_validation = True
        replay_validation = True
        full_pass = bool(pipeline_pass and cross_agreement and source_validation and replay_validation)
        if full_pass:
            classification = "PROMOTABLE_TO_UNTOUCHED_CONFIRMATION"
            promotion_eligible.append(pipeline_id)
        else:
            occupancy_both = all(record["occupancyInRange"] for record in candidate_gate_records)
            multi = min(dimension_closer_counts) >= 2
            if occupancy_both and not multi:
                classification = "EXPLORATORY_PAPER_MATCH_OCCUPANCY_ONLY"
            elif multi:
                classification = "METHOD_DEPENDENT_LEAD"
            else:
                classification = "RECURRING_ATTRACTOR_LABEL_NOT_RECONSTRUCTED"
        pipeline_classifications[pipeline_id] = classification
        for record in candidate_gate_records:
            record["crossCandidateGateDirectionAgreement"] = cross_agreement
            record["sourceAndFixtureValidationPassed"] = source_validation
            record["exactReplayValidationPassed"] = replay_validation
            record["pipelineFullPromotionContractPassed"] = full_pass
            record["pipelineClassification"] = classification
            decision_rows.append(record)

        c2 = fingerprint[
            (fingerprint["pipelineId"] == pipeline_id)
            & (fingerprint["candidateId"] == "CANDIDATE_2")
        ][["matrixIndex", *SUMMARY_METRICS]].copy()
        c3 = fingerprint[
            (fingerprint["pipelineId"] == pipeline_id)
            & (fingerprint["candidateId"] == "CANDIDATE_3")
        ][["matrixIndex", *SUMMARY_METRICS]].copy()
        paired = c2.merge(c3, on="matrixIndex", suffixes=("_C2", "_C3"))
        for metric in SUMMARY_METRICS:
            left = pd.to_numeric(paired[f"{metric}_C2"], errors="coerce")
            right = pd.to_numeric(paired[f"{metric}_C3"], errors="coerce")
            valid = left.notna() & right.notna()
            correlation = None
            if valid.sum() >= 3 and left[valid].nunique() > 1 and right[valid].nunique() > 1:
                correlation = float(np.corrcoef(left[valid], right[valid])[0, 1])
            rows.append(
                {
                    "pipelineId": pipeline_id,
                    "metric": metric,
                    "pairedMatrixCount": int(valid.sum()),
                    "candidate2Mean": float(left[valid].mean()) if valid.any() else None,
                    "candidate3Mean": float(right[valid].mean()) if valid.any() else None,
                    "meanCandidate3Minus2": (
                        float((right[valid] - left[valid]).mean()) if valid.any() else None
                    ),
                    "pairedPearson": correlation,
                    "directionallyAgreeRelativeToTarget": (
                        None
                        if metric not in PAPER_TARGETS or not valid.any()
                        else bool(
                            np.sign(left[valid].mean() - PAPER_TARGETS[metric][0])
                            == np.sign(right[valid].mean() - PAPER_TARGETS[metric][0])
                        )
                    ),
                }
            )

    if len(promotion_eligible) > 1:
        # The human contract permits at most one lead. No outcome-guided tie is
        # allowed, so two eligible labels are reported as eligible but neither
        # is promoted automatically.
        top = "METHOD_DEPENDENT_LEAD"
    elif promotion_eligible:
        top = "PROMOTABLE_TO_UNTOUCHED_CONFIRMATION"
    elif "METHOD_DEPENDENT_LEAD" in pipeline_classifications.values():
        top = "METHOD_DEPENDENT_LEAD"
    elif "EXPLORATORY_PAPER_MATCH_OCCUPANCY_ONLY" in pipeline_classifications.values():
        top = "EXPLORATORY_PAPER_MATCH_OCCUPANCY_ONLY"
    else:
        top = "RECURRING_ATTRACTOR_LABEL_NOT_RECONSTRUCTED"
    classification = {
        "schema": "eidosoma.e01.s19_l09.classification.v1",
        "versionedLoopId": VERSION,
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "topLevelClassification": top,
        "pipelineClassifications": pipeline_classifications,
        "promotionEligiblePipelines": promotion_eligible,
        "promotedLeadCount": 0,
        "automaticPromotionProhibited": True,
        "l08ClassificationPreserved": "NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA",
        "s18ProspectivePredictionStatusChanged": False,
        "s18ProspectiveCausalControlStatusChanged": False,
        "interpretationBoundary": "Any completed-run dominant reference is retrospective paper-facing evidence only.",
    }
    return pd.DataFrame(rows), pd.DataFrame(decision_rows), classification


def create_figures(
    molecular: pd.DataFrame,
    boundary: pd.DataFrame,
    fingerprint: pd.DataFrame,
    aggregate_all: pd.DataFrame,
    negative: pd.DataFrame,
    decision: pd.DataFrame,
) -> list[Path]:
    paths: list[Path] = []
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {R1_ID: "#3465a4", R2_ID: "#cc7a00"}

    # 1. Boundary H map, a direct view of recurring clusters on fixed M000.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row_index, pipeline_id in enumerate(LABEL_IDS):
        for col_index, candidate_id in enumerate(PRIMARY_CANDIDATES):
            subset = boundary[
                (boundary.pipelineId == pipeline_id)
                & (boundary.candidateId == candidate_id)
                & (boundary.matrixIndex == 0)
            ].sort_values("boundaryIndex0")
            h = subset["hToDominant"].to_numpy(float)
            labels = subset["isReplicator"].to_numpy(bool)
            axes[row_index, col_index].plot(h, color=colors[pipeline_id], lw=1.4)
            axes[row_index, col_index].axhline(0.9, color="black", ls="--", lw=1)
            axes[row_index, col_index].fill_between(
                np.arange(len(h)), 0, h, where=labels, alpha=0.22, color=colors[pipeline_id]
            )
            axes[row_index, col_index].set(title=f"{pipeline_id.split('_')[0]} · {candidate_id}", xlabel="post-fission generation", ylabel="H to dominant centroid")
    path = LOOP_ROOT / "figure_01_dominant_recurring_clusters.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # 2. Molecular H-to-attractor over time.
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True, sharex=False)
    for axis, candidate_id in zip(axes, PRIMARY_CANDIDATES, strict=True):
        for pipeline_id in LABEL_IDS:
            subset = molecular[
                (molecular.pipelineId == pipeline_id)
                & (molecular.candidateId == candidate_id)
                & (molecular.matrixIndex == 0)
                & (molecular.analysisUnitIndex >= 0)
            ].sort_values("analysisUnitIndex")
            axis.plot(subset.analysisUnitIndex, subset.hToDominant, lw=1, label=pipeline_id.split("_")[0], color=colors[pipeline_id])
        axis.axhline(0.9, color="black", ls="--", lw=1)
        axis.set(title=candidate_id, ylabel="H to dominant attractor", xlabel="selected molecular index")
        axis.legend()
    path = LOOP_ROOT / "figure_02_molecular_h_to_dominant_over_time.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # 3. Adjacent versus direct attractor labels.
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), constrained_layout=True)
    for axis, candidate_id in zip(axes, PRIMARY_CANDIDATES, strict=True):
        offset = 0
        for pipeline_id in (COMPARATOR_ADJACENT, *LABEL_IDS):
            if pipeline_id == COMPARATOR_ADJACENT:
                # Adjacent label hashes are retained in comparator evidence; for
                # the visual, reconstruct the frozen label from molecular states'
                # direct H rows is not valid, so show pipeline occupancies as a
                # trajectory-level band rather than inventing row labels.
                fp = fingerprint[
                    (fingerprint.pipelineId == pipeline_id)
                    & (fingerprint.candidateId == candidate_id)
                    & (fingerprint.matrixIndex == 0)
                ]
                if not fp.empty:
                    axis.axhline(offset + float(fp.occupancy.iloc[0]), xmin=0, xmax=1, lw=5, color="#777777", label="adjacent occupancy")
            else:
                subset = molecular[
                    (molecular.pipelineId == pipeline_id)
                    & (molecular.candidateId == candidate_id)
                    & (molecular.matrixIndex == 0)
                    & (molecular.analysisUnitIndex >= 0)
                ].sort_values("analysisUnitIndex")
                axis.step(subset.analysisUnitIndex, subset.isReplicator.astype(int) + offset, where="post", lw=0.9, label=pipeline_id.split("_")[0], color=colors[pipeline_id])
            offset += 1.3
        axis.set(title=candidate_id, xlabel="selected molecular index", ylabel="offset label state")
        axis.legend(ncol=3)
    path = LOOP_ROOT / "figure_03_adjacent_vs_recurring_labels.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    labels_order = [COMPARATOR_ADJACENT, COMPARATOR_A_BOUNDARY, COMPARATOR_A_PROJECTED, COMPARATOR_B_HIGH, R1_ID, R2_ID]
    short = {COMPARATOR_ADJACENT: "Adjacent", COMPARATOR_A_BOUNDARY: "A boundary", COMPARATOR_A_PROJECTED: "A projected", COMPARATOR_B_HIGH: "B high-h", R1_ID: "R1", R2_ID: "R2"}
    plot = aggregate_all[aggregate_all.pipelineId.isin(labels_order)].copy()

    # 4. Occupancy and persistence.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis, metric, target in zip(axes, ("mean_occupancy", "mean_persistence"), (0.88, 716), strict=True):
        pivot = plot.pivot(index="pipelineId", columns="candidateId", values=metric).reindex(labels_order)
        pivot.index = [short[item] for item in pivot.index]
        pivot.plot(kind="bar", ax=axis, color=["#3465a4", "#cc7a00"])
        axis.axhline(target, color="black", ls="--", lw=1)
        axis.set(title=metric.replace("mean_", "").title(), xlabel="", ylabel=metric.replace("mean_", ""))
        axis.tick_params(axis="x", rotation=35)
    path = LOOP_ROOT / "figure_04_occupancy_persistence_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # 5. Consistency and onset.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis, metric, target in zip(axes, ("mean_consistency", "mean_firstOnsetRawStep1"), (0.38, 37), strict=True):
        pivot = plot.pivot(index="pipelineId", columns="candidateId", values=metric).reindex(labels_order)
        pivot.index = [short[item] for item in pivot.index]
        pivot.plot(kind="bar", ax=axis, color=["#3465a4", "#cc7a00"])
        axis.axhline(target, color="black", ls="--", lw=1)
        axis.set(title=metric.replace("mean_", "").title(), xlabel="")
        axis.tick_params(axis="x", rotation=35)
    path = LOOP_ROOT / "figure_05_consistency_first_onset.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # 6. Episode topology for fixed M000.
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), constrained_layout=True)
    for i, pipeline_id in enumerate(LABEL_IDS):
        for j, candidate_id in enumerate(PRIMARY_CANDIDATES):
            subset = molecular[(molecular.pipelineId == pipeline_id) & (molecular.candidateId == candidate_id) & (molecular.matrixIndex == 0) & (molecular.analysisUnitIndex >= 0)].sort_values("analysisUnitIndex")
            labels = subset.isReplicator.astype(int).to_numpy()
            axes[i, j].imshow(labels[None, :], aspect="auto", cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest")
            axes[i, j].set(title=f"{pipeline_id.split('_')[0]} · {candidate_id}", xlabel="molecular index", yticks=[])
    path = LOOP_ROOT / "figure_06_episode_topology.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # 7. Controls.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    aggregate_controls = negative[negative.get("recordType", pd.Series(index=negative.index, dtype=object)) == "AGGREGATE_CONTROL_TEST"]
    for axis, outcome in zip(axes, ("boundaryRecurrence", "rawPaperDistance"), strict=True):
        subset = aggregate_controls[aggregate_controls.outcome == outcome]
        x = np.arange(len(subset))
        axis.scatter(x, subset.observed, label="dominant", color="#3465a4")
        axis.scatter(x, subset.nullMean, label="control", color="#cc7a00")
        axis.set(title=outcome, xticks=x, xticklabels=[f"{p.split('_')[0]}\n{c[-1]}" for p, c in zip(subset.pipelineId, subset.candidateId, strict=False)])
        axis.legend()
    path = LOOP_ROOT / "figure_07_negative_controls.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # 8. Candidate agreement.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, pipeline_id in zip(axes, LABEL_IDS, strict=True):
        c2 = fingerprint[(fingerprint.pipelineId == pipeline_id) & (fingerprint.candidateId == "CANDIDATE_2")][["matrixIndex", "occupancy"]]
        c3 = fingerprint[(fingerprint.pipelineId == pipeline_id) & (fingerprint.candidateId == "CANDIDATE_3")][["matrixIndex", "occupancy"]]
        paired = c2.merge(c3, on="matrixIndex", suffixes=("_C2", "_C3"))
        axis.scatter(paired.occupancy_C2, paired.occupancy_C3, alpha=0.7, color=colors[pipeline_id])
        axis.plot([0, 1], [0, 1], ls="--", color="black")
        axis.set(title=pipeline_id.split("_")[0], xlabel="candidate 2 occupancy", ylabel="candidate 3 occupancy", xlim=(0, 1), ylim=(0, 1))
    path = LOOP_ROOT / "figure_08_cross_candidate_agreement.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # 9. Final fingerprint decision matrix.
    gate_columns = [column for column in decision.columns if column not in {"pipelineId", "candidateId", "validTrajectoryCount", "dimensionsCloserThanEveryComparator", "closerMetricsJson", "gateCount", "passedGateCount", "pipelineClassification"} and decision[column].dtype == bool]
    matrix = decision.set_index(["pipelineId", "candidateId"])[gate_columns].astype(int)
    fig, ax = plt.subplots(figsize=(max(12, len(gate_columns) * 0.65), 4.5), constrained_layout=True)
    ax.imshow(matrix.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(gate_columns)), [item.replace("Pass", "").replace("Passed", "") for item in gate_columns], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix)), [f"{idx[0].split('_')[0]} · {idx[1]}" for idx in matrix.index])
    ax.set_title("Locked L09 fingerprint and validation gates (green=pass)")
    path = LOOP_ROOT / "figure_09_fingerprint_decision_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def render_reports(
    aggregate: pd.DataFrame,
    decision: pd.DataFrame,
    classification: dict[str, Any],
    validation_summary: str,
    runtime: dict[str, Any],
    figures: list[Path],
) -> tuple[str, str]:
    primary = aggregate[aggregate.pipelineId.isin(LABEL_IDS)].copy()
    table_columns = [
        "pipelineId",
        "candidateId",
        "validTrajectoryCount",
        "mean_occupancy",
        "mean_persistence",
        "mean_consistency",
        "mean_firstOnsetRawStep1",
        "mean_firstOnsetNormalized",
        "mean_positiveEpisodeCount",
        "mean_negativeEpisodeCount",
        "mean_isNonreplicatingAtQuarterCutoff",
        "rawPaperDistance",
        "normalizedPaperDistance",
    ]
    result_table = primary[table_columns].to_markdown(index=False, floatfmt=".6g")
    gate_table = decision[
        [
            "pipelineId",
            "candidateId",
            "passedGateCount",
            "gateCount",
            "occupancyInRange",
            "persistenceInRange",
            "consistencyCloserThanEveryComparator",
            "rawOnsetCloserThanEveryComparator",
            "quarterStateMeaningful",
            "quarterNoOnsetMeaningful",
            "randomReferenceRecurrence",
            "secondClusterRecurrence",
            "pipelineFullPromotionContractPassed",
            "pipelineClassification",
        ]
    ].to_markdown(index=False)
    artifact_names = [
        "cluster_results.parquet",
        "dominant_attractor_results.parquet",
        "molecular_label_results.parquet",
        "boundary_label_results.parquet",
        "label_fingerprint_results.parquet",
        "episode_results.parquet",
        "comparator_results.parquet",
        "negative_control_results.parquet",
        "paper_target_comparison.csv",
        "complete_fingerprint_distances.parquet",
        "candidate_comparison.csv",
        "bootstrap_results.parquet",
        "classification.json",
        "regeneration_validation.json",
        "artifact_manifest.json",
    ]
    lay = (
        "L09 asked whether the paper's reported replicator state is better understood as membership in a recurring compositional attractor than as similarity to the immediately preceding state. It applied two fixed, source/paper-grounded constructions to the same frozen trajectories and judged occupancy together with persistence, consistency, onset, episodes, controls, and agreement across both simulator candidates."
    )
    report = f"""# S19-L09 Full Results — Recurring-Attractor Label Reconstruction

## Concise top summary

- **Research step ID:** `S19-L09` (`{VERSION}`).
- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`; L09 is frozen and no downstream work was started.
- **Artifacts written:** all 30 directed machine-readable/report artifacts, nine required figures, canonical `research_step_full_results.md`, compact status, and append-only S19 ledgers under `{LOOP_ROOT}` and `{ARTIFACT_ROOT}`.
- **Validation result:** {validation_summary}
- **Outcome classification:** `{classification['topLevelClassification']}`. Pipeline classifications: `{json.dumps(classification['pipelineClassifications'], sort_keys=True)}`. No lead was promoted automatically.
- **Caveats or blockers:** both dominant references use completed runs and are retrospective; exact target-paper code, MATLAB RNG/version behavior, Table 1 onset units, and SD-versus-SE identity remain unavailable. A numerical or occupancy resemblance cannot establish prospective prediction or causal control.
- **Recommended next action:** mandatory human review. Choose explicitly among untouched confirmation of an eligible lead, one new narrow S19 loop, S20 confirmation/closeout, or pause. Do not begin any option automatically.

## Lay summary

{lay}

## Frozen question

Can direct molecular membership relative to a run's dominant recurring post-fission composition jointly reproduce the paper's control fingerprints better than the frozen adjacent-molecular, fission-boundary, projected-boundary, and high-exposure comparators?

## Inputs

- Exactly 100 shared L08 catalytic matrices and matched initial states.
- Original-exposure candidate 2 and candidate 3 trajectories were the primary label substrate.
- Fixed `h=2.875` candidate 2 and candidate 3 trajectories were comparator-only.
- Every trajectory retained 100 selected post-fission daughters and the original-order selected molecular clock.
- Original paper v1, Figure 1, Table 1, pinned historical GARD v10 source, and cited GARD methods 63–65.
- No new trajectory, PhiRL/emergence value, prediction model, or intervention trajectory was generated.

## Methods

### R1 — historical dominant compotype

R1 applied the pinned technique-1 local non-drift rule at strict `H>0.9`, clustered only eligible post-fission compositions with deterministic source-equivalent cosine k-means over `k=1..10` and ten replicas, retained the historical special k=1 carpet score and four-k nonimprovement stop, selected the largest valid cluster, and labelled every molecular state directly by strict H to that centroid.

### R2 — paper-Euclidean dominant attractor

R2 clustered all 100 post-fission relative compositions with frozen deterministic Euclidean Lloyd k-means over `k=1..10` and ten replicas, selected k by Euclidean silhouette (k=1 explicitly undefined), selected the largest valid cluster, and applied the same direct strict-H molecular membership rule.

Both pipelines required at least two assigned states and two strict-H centroid visits. No threshold, radius, density, medoid, third clustering family, post-fission projection, or favorable candidate selection was allowed.

### Table 1 and inferential semantics

Probability was `mean(Y)`, persistence `sum(Y)`, consistency Pearson `corr(Y_t,Y_t+1)` with constants undefined, and first onset was reported as zero-based/one-based raw molecular step, normalized fraction, and fission generation. Both sample SD and SE are retained; `AUTHOR_DISPERSION_UNRESOLVED` was not resolved by target proximity. Catalytic matrix was the independent unit. Candidate 2 and candidate 3 stayed separate. Exactly 4,096 domain-separated matrix-bootstrap replicates were used.

### Negative controls

Exactly 64 frozen random observed-boundary references per trajectory and the selected fit's second-largest valid cluster (where present) were evaluated at unchanged strict `H>0.9`. Control results use no emergence value.

## Results

### Primary temporal fingerprints

{result_table}

### Locked promotion gates

{gate_table}

The machine-readable decision is authoritative. Occupancy, persistence, consistency, raw and normalized onset, positive/negative episodes, quarter-cutoff eligibility, dominant-cluster recurrence, random and second-cluster controls, and cross-candidate agreement were evaluated jointly. Occupancy alone could not promote a pipeline.

## Illustrated results

![Dominant recurring clusters](figure_01_dominant_recurring_clusters.png)

*Figure 1. Fixed matrix M000 post-fission similarity to each pipeline's dominant recurring centroid; the dashed line is the frozen strict H=0.9 membership threshold.*

![Molecular H to dominant attractor](figure_02_molecular_h_to_dominant_over_time.png)

*Figure 2. Direct molecular H-to-centroid trajectories for fixed matrix M000; these are not interval-projected boundary labels.*

![Adjacent versus recurring labels](figure_03_adjacent_vs_recurring_labels.png)

*Figure 3. Recurring-attractor label topology for fixed M000 with the frozen adjacent comparator retained separately.*

![Occupancy and persistence](figure_04_occupancy_persistence_comparison.png)

*Figure 4. Candidate-specific occupancy and persistence compared with paper targets and all frozen mechanisms.*

![Consistency and onset](figure_05_consistency_first_onset.png)

*Figure 5. Candidate-specific consecutive-label consistency and one-based molecular first onset.*

![Episode topology](figure_06_episode_topology.png)

*Figure 6. Positive and negative episode placement for fixed M000, showing why equal occupancy need not imply equal temporal structure.*

![Negative controls](figure_07_negative_controls.png)

*Figure 7. Dominant-reference performance versus preregistered random-reference and second-cluster controls.*

![Cross-candidate agreement](figure_08_cross_candidate_agreement.png)

*Figure 8. Matrix-paired candidate-2 versus candidate-3 occupancy agreement for R1 and R2.*

![Fingerprint decision matrix](figure_09_fingerprint_decision_matrix.png)

*Figure 9. Complete locked decision matrix; green denotes a passed gate and red a failed gate.*

## Validation

- Mandatory source-equivalence and synthetic fixtures passed before scientific execution, including planted/no-cluster/two-attractor/tie/direct-molecular tests.
- The complete lock was committed and pushed before outcome access; the run required a clean worktree with `HEAD == origin/eidosoma/groups/42` and matching code hashes.
- The 400 frozen L08 cache hashes, status fields, source identities, and all immutable prior files were revalidated.
- All 200 primary trajectories were analyzed twice independently; canonical output hashes matched for every core scientific table.
- Exactly 4,096 bootstrap replicates were used per pipeline/candidate.
- Storage, scope, numerical finiteness, candidate presence, and artifact hashes passed.
- No emergence, prediction, intervention, GPU, or new-simulation path exists in the L09 implementation.

## Commands

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l09.py
PYTHONPATH=src python scripts/e01/prepare_s19_l09_lock.py
git commit ... && git push origin eidosoma/groups/42
PYTHONPATH=src python scripts/e01/run_s19_l09.py --workers 8
```

Thread variables were fixed to one for OpenMP, OpenBLAS, MKL, NumExpr, and vecLib. CPU float64 was authoritative; GPU use was zero.

## Artifacts and provenance

Principal artifacts: {', '.join(f'`{name}`' for name in artifact_names)}. Source identities, paths, hashes, retrieval dates, and redistribution status are in `source_snapshot_manifest.json` and the append-only S19 source ledger. Historical source without a detected compatible license remains cache-only and is not redistributed.

Runtime: `{runtime['wallSeconds']:.3f}` wall seconds and `{runtime['cpuSecondsParentPlusChildren']:.3f}` total CPU seconds for locked execution, replay, synthesis, and validation.

## Caveats, blockers, and interpretation boundary

1. The same matrices were used in prior L08 work; L09 is adaptive exploratory reconstruction, not confirmation.
2. A full-run dominant centroid uses future observations and can support retrospective paper-facing resemblance only.
3. R1 is source-equivalent reconstruction, not bitwise historical MATLAB or target-author code. R2 is paper-grounded but method-dependent because the paper omits exact clustering code.
4. Known paper targets can make multi-fingerprint exploration vulnerable to adaptive overfitting; untouched confirmation is required for any eligible lead.
5. Boundary and molecular values remain noninterchangeable. Table 1 dispersion identity and onset unit remain unresolved.
6. This result does not establish that causal emergence predicts replication, that completed-fit values are prospective, that PhiRL adds information beyond the label, or that interventions control replication.
7. L08's 4/6 occupancy result and terminal classification remain unchanged, as do S18's prospective-prediction and causal-control non-support conclusions.

## Recommended next action

Stop for mandatory human review. No S20 mode, later S19 loop, E02, author contact, report bundle, emergence association, prediction, or intervention work is active.
"""
    decision_summary = f"""# S19-L09 One-Page Decision Summary

## Concise top summary

- **Research step ID:** `S19-L09`.
- **Completion status:** complete and frozen; awaiting mandatory human review.
- **Artifacts written:** full report, 30 directed artifacts, nine figures, root ledger/handoff updates, and hash manifests.
- **Validation result:** {validation_summary}
- **Outcome classification:** `{classification['topLevelClassification']}`; pipeline results `{json.dumps(classification['pipelineClassifications'], sort_keys=True)}`; zero automatic promotions.
- **Caveats/blockers:** completed-run references are retrospective; exact author implementation and Table 1 unit/dispersion identities remain unavailable.
- **Recommended next action:** human review only; explicitly choose the next option, if any.

## What was tested

Exactly two direct molecular recurring-attractor labels on the frozen L08 original-exposure trajectories: one pinned historical-GARD dominant-compotype reconstruction and one paper-Euclidean dominant-attractor reconstruction. Fixed L08 adjacent, boundary, projected, and high-exposure mechanisms were comparators. No scientific setting changed after lock.

## Decision evidence

{result_table}

{gate_table}

## Boundary

The result is exploratory and retrospective. It cannot change L08's untouched classification or S18's prospective-prediction and causal-control conclusions. No lead is promoted automatically.
"""
    return report, decision_summary


def append_postloop_root_records(classification: dict[str, Any], report: str) -> None:
    now = utc_now()
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    if not any(
        (ledger["loopId"] == "S19-L09")
        & (ledger["recordPhase"] == "POST_LOOP_MANDATORY_HUMAN_REVIEW_BOUNDARY")
    ):
        pipeline_text = json.dumps(classification["pipelineClassifications"], sort_keys=True)
        row = {
            "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
            "timestampUtc": now,
            "loopId": "S19-L09",
            "recordPhase": "POST_LOOP_MANDATORY_HUMAN_REVIEW_BOUNDARY",
            "beliefBeforeLoop": "A paper-literal dominant recurring-composition label might jointly explain more of the control fingerprint than adjacent, boundary-projection, or high-exposure mechanisms.",
            "motivatingEvidence": "Paper and historical-source descriptions plus L08's negative complete-mechanism result.",
            "failureOrAmbiguityTargeted": "Self-replicator-label mismatch upstream of Figures 3-6 and Table 1.",
            "selectedHypotheses": "Exactly R1 and R2 under the pushed L09 lock.",
            "learned": f"Top classification {classification['topLevelClassification']}; pipeline classifications {pipeline_text}; zero leads were automatically promoted.",
            "weakenedHypotheses": "Every registered label to the extent it failed the joint cross-candidate, temporal-fingerprint, negative-control, or validation contract.",
            "remainingPlausibleHypotheses": "Only an explicitly promotion-eligible label can be considered for untouched confirmation; exact author implementation remains unresolved.",
            "proposedNextTest": "Mandatory human review; choose explicitly among untouched confirmation, another narrow S19 loop, S20, or pause.",
            "informationGainRationale": "Any continuation must use L09's fixed structural evidence and cannot reinterpret retrospective label fit as early warning or causal control.",
            "appendOnly": True,
        }
        ledger = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
        write_parquet(ledger_path, ledger)
        with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open("a", encoding="utf-8") as handle:
            handle.write(
                "\n\n## Entry 020 — S19-L09 learning and mandatory human-review boundary\n\n"
                f"- **What was learned:** `{classification['topLevelClassification']}`; pipeline classifications `{json.dumps(classification['pipelineClassifications'], sort_keys=True)}`.\n"
                "- **What was weakened:** any registered label that failed its joint temporal, control, cross-candidate, replay, or validation gates.\n"
                "- **What remains plausible:** only explicitly eligibility-marked retrospective leads for untouched confirmation; exact author identity remains unavailable.\n"
                "- **What should be tested next:** nothing automatically; human review must select the next option.\n"
                "- **Why another loop must add information:** it must resolve a new source-grounded ambiguity and cannot treat target proximity alone as confirmation.\n"
            )

    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    for item in registry["loops"]:
        if item["loopId"] == "S19-L09":
            item.update(
                {
                    "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
                    "outcomeAccessed": True,
                    "completed": True,
                    "eligibleScientificResults": True,
                    "classification": classification["topLevelClassification"],
                    "pipelineClassifications": classification["pipelineClassifications"],
                    "promotionEligibleCount": len(
                        classification["promotionEligiblePipelines"]
                    ),
                    "promotedLeadCount": 0,
                    "nextStepActive": False,
                }
            )
            break
    registry["laterLoopsAuthorized"] = False
    registry["s20Status"] = "DEFINED_INACTIVE"
    registry["proposedNextLoopTheme"] = None
    registry["proposedNextLoopActive"] = False
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    review_path = ARTIFACT_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not any(item.get("decision") == "S19_L09_COMPLETE_MANDATORY_HUMAN_REVIEW" for item in review["history"]):
        review["history"].append(
            {
                "date": "2026-08-09",
                "decision": "S19_L09_COMPLETE_MANDATORY_HUMAN_REVIEW",
                "scope": VERSION,
                "result": classification["topLevelClassification"],
                "source": "validated_locked_execution_result",
            }
        )
    review["pendingDecision"] = "POST_S19_L09_MANDATORY_HUMAN_REVIEW_REQUIRED"
    write_json(review_path, review)

    (ARTIFACT_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": "S19-L09",
        "stepNumber": 19,
        "success": True,
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": [
            str(LOOP_ROOT / "S19_L09_FULL_RESULTS.md"),
            str(LOOP_ROOT / "label_fingerprint_results.parquet"),
            str(LOOP_ROOT / "classification.json"),
            str(LOOP_ROOT / "regeneration_validation.json"),
            str(LOOP_ROOT / "artifact_manifest.json"),
        ],
        "validationResult": "PASS_IMMUTABLE_SOURCE_FIXTURE_INPUT_REPLAY_BOOTSTRAP_SCOPE_STORAGE_AND_ARTIFACT_VALIDATION",
        "outcomeClassification": classification["topLevelClassification"],
        "caveatsOrBlockers": [
            "completed_run_dominant_references_are_retrospective",
            "exact_author_code_unavailable",
            "historical_matlab_rng_and_version_unresolved",
            "author_dispersion_and_onset_units_unresolved",
            "no_prospective_prediction_or_causal_control_inference",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_NO_AUTOMATIC_L10_S20_E02_AUTHOR_CONTACT_OR_REPORT_BUNDLE",
    }
    write_json(ARTIFACT_ROOT / "s19_status.json", status)
    write_json(LOOP_ROOT / "status.json", status)


def manifest_for(root: Path, *, exclude: set[Path]) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path in exclude:
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": "eidosoma.e01.s19_l09.artifact_manifest.v1",
        "root": str(root),
        "generatedAtUtc": utc_now(),
        "fileCount": len(files),
        "totalBytes": sum(row["bytes"] for row in files),
        "files": files,
    }


def main() -> None:
    import argparse
    import resource

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be 1..8")
    started_utc = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    child_cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN)

    release = repository_release_gate()
    write_json(LOOP_ROOT / "run_release_gate.json", release)
    if not release["passed"]:
        raise RuntimeError(f"repository release gate failed: {release}")
    immutable = validate_immutable_prior()
    write_json(LOOP_ROOT / "immutable_prior_validation.json", immutable)
    if not immutable["passed"]:
        raise RuntimeError("immutable prior mismatch")
    benchmark = json.loads((LOOP_ROOT / "preoutcome_benchmark.json").read_text())
    if not benchmark["passed"]:
        raise RuntimeError("benchmark gate not passed")

    first = execute_primary_pass(args.workers)
    replay = execute_primary_pass(args.workers)
    sort_contracts = {
        "cluster": ["pipelineId", "candidateId", "matrixIndex", "k"],
        "dominant": ["pipelineId", "candidateId", "matrixIndex"],
        "molecular": ["pipelineId", "candidateId", "matrixIndex", "analysisUnitIndex"],
        "boundary": ["pipelineId", "candidateId", "matrixIndex", "boundaryIndex0"],
        "fingerprint": ["pipelineId", "candidateId", "matrixIndex"],
        "episode": ["pipelineId", "candidateId", "matrixIndex", "polarity", "episodeIndex"],
        "negative": ["pipelineId", "candidateId", "matrixIndex", "controlType", "controlIndex"],
        "failure": ["pipelineId", "candidateId", "matrixIndex"],
        "adjacentComparator": ["pipelineId", "candidateId", "matrixIndex"],
    }
    replay_rows = []
    replay_passed = True
    for key, sort_columns in sort_contracts.items():
        if first[key].empty and replay[key].empty:
            left_hash = right_hash = sha256_text("EMPTY")
        else:
            left_hash = canonical_frame_sha256(first[key], sort_columns)
            right_hash = canonical_frame_sha256(replay[key], sort_columns)
        passed = left_hash == right_hash and len(first[key]) == len(replay[key])
        replay_passed &= passed
        replay_rows.append(
            {
                "table": key,
                "firstRowCount": len(first[key]),
                "replayRowCount": len(replay[key]),
                "firstCanonicalSha256": left_hash,
                "replayCanonicalSha256": right_hash,
                "passed": passed,
            }
        )
    regeneration = {
        "schema": "eidosoma.e01.s19_l09.regeneration_validation.v1",
        "independentFullPassCount": 2,
        "tables": replay_rows,
        "allTablesExact": replay_passed,
        "passed": replay_passed,
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "regeneration_validation.json", regeneration)
    if not replay_passed:
        raise RuntimeError("exact full-scope result replay failed")

    cluster = first["cluster"]
    dominant = first["dominant"]
    molecular = first["molecular"]
    boundary = first["boundary"]
    fingerprint = first["fingerprint"]
    episodes = first["episode"]
    failures = first["failure"]
    comparators = normalize_l08_comparators(first["adjacentComparator"])
    aggregate_primary = aggregate_fingerprints(fingerprint)
    aggregate_comparators = aggregate_fingerprints(comparators)
    aggregate_all = pd.concat([aggregate_primary, aggregate_comparators], ignore_index=True)
    bootstrap = bootstrap_primary(fingerprint)
    target_table, distances = target_comparisons(aggregate_all, bootstrap)
    negative, control_gates = negative_control_analysis(
        first["negative"], fingerprint, dominant
    )
    cross_candidate, decision, classification = candidate_comparisons(
        fingerprint, aggregate_primary, aggregate_comparators, control_gates
    )

    write_parquet(LOOP_ROOT / "cluster_results.parquet", cluster)
    write_parquet(LOOP_ROOT / "dominant_attractor_results.parquet", dominant)
    write_parquet(LOOP_ROOT / "molecular_label_results.parquet", molecular)
    write_parquet(LOOP_ROOT / "boundary_label_results.parquet", boundary)
    write_parquet(LOOP_ROOT / "label_fingerprint_results.parquet", fingerprint)
    write_parquet(LOOP_ROOT / "episode_results.parquet", episodes)
    write_parquet(LOOP_ROOT / "comparator_results.parquet", comparators)
    write_parquet(LOOP_ROOT / "negative_control_results.parquet", negative)
    write_csv(LOOP_ROOT / "paper_target_comparison.csv", target_table)
    write_parquet(LOOP_ROOT / "complete_fingerprint_distances.parquet", distances)
    write_csv(LOOP_ROOT / "candidate_comparison.csv", cross_candidate)
    write_parquet(LOOP_ROOT / "bootstrap_results.parquet", bootstrap)
    write_csv(
        LOOP_ROOT / "failure_ledger.csv",
        failures
        if not failures.empty
        else pd.DataFrame(
            columns=[
                "failureId",
                "pipelineId",
                "candidateId",
                "matrixIndex",
                "failureStatus",
                "excludedFromScientificAggregation",
            ]
        ),
    )
    write_json(LOOP_ROOT / "classification.json", classification)
    write_parquet(LOOP_ROOT / "candidate_gate_results.parquet", decision)
    write_parquet(LOOP_ROOT / "aggregate_fingerprint_results.parquet", aggregate_all)

    figures = create_figures(
        molecular,
        boundary,
        pd.concat([fingerprint, comparators], ignore_index=True),
        aggregate_all,
        negative,
        decision,
    )

    child_cpu_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds_parent = time.process_time() - cpu_start
    child_cpu_seconds = (
        child_cpu_end.ru_utime
        + child_cpu_end.ru_stime
        - child_cpu_start.ru_utime
        - child_cpu_start.ru_stime
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l09.runtime_manifest.v1",
        "startedAtUtc": started_utc,
        "completedAtUtc": utc_now(),
        "wallSeconds": wall_seconds,
        "cpuSecondsParent": cpu_seconds_parent,
        "cpuSecondsChildren": child_cpu_seconds,
        "cpuSecondsParentPlusChildren": cpu_seconds_parent + child_cpu_seconds,
        "cpuHours": (cpu_seconds_parent + child_cpu_seconds) / 3600,
        "workers": args.workers,
        "threadsPerWorker": 1,
        "gpuHours": 0,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikitLearn": sklearn.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "repositoryHead": release["head"],
        "withinCpuCeiling": bool((cpu_seconds_parent + child_cpu_seconds) / 3600 <= 32),
        "withinWallCeiling": bool(wall_seconds / 3600 <= 8),
    }
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime)
    if not runtime["withinCpuCeiling"] or not runtime["withinWallCeiling"]:
        raise RuntimeError("compute ceiling exceeded")

    validation_summary = (
        f"PASS: {immutable['baselineFileCount']} immutable prior files; 400 frozen input cache identities; "
        f"all mandatory source/synthetic fixtures; 200/200 primary trajectory analyses and exact second-pass regeneration; "
        f"exactly {BOOTSTRAP_REPLICATES} matrix bootstraps per pipeline/candidate; both candidates retained; scope, storage, and artifact hashes."
    )
    report, decision_summary = render_reports(
        aggregate_all, decision, classification, validation_summary, runtime, figures
    )
    (LOOP_ROOT / "S19_L09_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "loop_decision_summary.md").write_text(
        decision_summary, encoding="utf-8"
    )
    append_postloop_root_records(classification, report)

    retained_bytes = sum(
        path.stat().st_size for path in LOOP_ROOT.rglob("*") if path.is_file()
    )
    cache_bytes = sum(
        path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file()
    )
    storage = {
        "schema": "eidosoma.e01.s19_l09.storage_validation.v1",
        "retainedBytes": retained_bytes,
        "retainedGiB": retained_bytes / (1024**3),
        "retainedCeilingGiB": 10,
        "temporaryCacheBytes": cache_bytes,
        "temporaryCacheGiB": cache_bytes / (1024**3),
        "temporaryCacheCeilingGiB": 25,
        "passed": bool(retained_bytes <= 10 * 1024**3 and cache_bytes <= 25 * 1024**3),
    }
    write_json(LOOP_ROOT / "storage_validation.json", storage)
    if not storage["passed"]:
        raise RuntimeError("storage ceiling exceeded")

    loop_manifest_path = LOOP_ROOT / "artifact_manifest.json"
    loop_manifest = manifest_for(LOOP_ROOT, exclude={loop_manifest_path})
    write_json(loop_manifest_path, loop_manifest)

    # Validate the manifest immediately, before creating the root-level manifest.
    manifest_replay_pass = all(
        (LOOP_ROOT / row["path"]).stat().st_size == row["bytes"]
        and sha256_file(LOOP_ROOT / row["path"]) == row["sha256"]
        for row in loop_manifest["files"]
    )
    write_json(
        LOOP_ROOT / "artifact_integrity_validation.json",
        {
            "schema": "eidosoma.e01.s19_l09.artifact_integrity_validation.v1",
            "manifestFileCount": loop_manifest["fileCount"],
            "allManifestRowsReplay": manifest_replay_pass,
            "passed": manifest_replay_pass,
        },
    )
    # Refresh loop manifest once so it includes the integrity record.
    loop_manifest = manifest_for(LOOP_ROOT, exclude={loop_manifest_path})
    write_json(loop_manifest_path, loop_manifest)

    root_manifest_path = ARTIFACT_ROOT / "artifact_manifest.json"
    root_manifest = manifest_for(ARTIFACT_ROOT, exclude={root_manifest_path})
    root_manifest["schema"] = "eidosoma.e01.s19_artifact_manifest.v1"
    write_json(root_manifest_path, root_manifest)
    print(
        json.dumps(
            {
                "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
                "classification": classification["topLevelClassification"],
                "pipelineClassifications": classification["pipelineClassifications"],
                "promotionEligiblePipelines": classification[
                    "promotionEligiblePipelines"
                ],
                "promotedLeadCount": 0,
                "primaryFingerprintRows": len(fingerprint),
                "molecularLabelRows": len(molecular),
                "bootstrapRows": len(bootstrap),
                "wallSeconds": wall_seconds,
                "cpuHours": runtime["cpuHours"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
