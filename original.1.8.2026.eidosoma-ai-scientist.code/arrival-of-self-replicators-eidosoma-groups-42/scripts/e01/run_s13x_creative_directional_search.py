#!/usr/bin/env python3
"""Run S13X stage-1 label/metric/transform/alignment search and diagnostic split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.stats.diagnostic import acorr_ljungbox

from e01_creative_directional_search.core import (
    ALIGNMENTS,
    CANDIDATE_IDS,
    DEVELOPMENT_INDICES,
    DIAGNOSTIC_INDICES,
    EVIDENCE_CLASS,
    RESEARCH_STEP_ID,
    TRANSFORMS,
    VERSION,
    association_summary,
    derive_seed,
    label_specs,
    label_trajectory,
    resemblance_score,
    transform_values,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps" / "S13X"
CACHE_ROOT = Path("/cache/e01_s13x_v1")
LABEL_CACHE = CACHE_ROOT / "labels"
SOURCE_VALUES = ARTIFACTS / "research_steps" / "S13RRR" / "full_source_values.parquet"

IMPLEMENTATIONS = ("IIGR_CORRECTED_SOURCE", "PHIRL_REGULARIZED_SOURCE")
METRICS = ("synergy", "downwardCausation", "emergence", "localPhiR")
BOOTSTRAP_REPLICATES = 4096
SHIFT_REPLICATES = 4096


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(frame.columns).encode("utf-8"))
    digest.update("\x1f".join(map(str, frame.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    def json_default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"unsupported JSON value: {type(value)!r}")

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False, compression="zstd")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def validate_frozen_inputs() -> dict[str, Any]:
    manifest = json.loads(
        (STEP_ROOT / "input_manifest.json").read_text(encoding="utf-8")
    )
    mismatches = []
    for item in manifest["inputs"]:
        path = Path(item["path"])
        actual = sha256_file(path) if path.is_file() else None
        if actual != item["sha256"]:
            mismatches.append(
                {
                    "path": str(path),
                    "expectedSha256": item["sha256"],
                    "actualSha256": actual,
                }
            )
    payload = {
        "schema": "eidosoma.e01.s13x_input_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "inputCount": len(manifest["inputs"]),
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    write_json(STEP_ROOT / "input_validation.json", payload)
    if mismatches:
        raise RuntimeError("one or more frozen S13X inputs changed")
    return payload


def _label_cache_path(candidate_id: str, matrix_index: int) -> Path:
    return LABEL_CACHE / candidate_id / f"M{matrix_index:03d}.parquet"


def _raw_path(candidate_id: str, matrix_index: int) -> Path:
    return Path(
        f"/cache/e01_s13/raw_trajectories/{candidate_id}/M{matrix_index:03d}.pickle"
    )


def _label_task(
    candidate_id: str, matrix_index: int, overwrite: bool = False
) -> dict[str, Any]:
    cache_path = _label_cache_path(candidate_id, matrix_index)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = _raw_path(candidate_id, matrix_index)
    if cache_path.is_file() and not overwrite:
        frame = pd.read_parquet(cache_path)
        fingerprint_path = cache_path.with_suffix(".fingerprints.json")
        fingerprints = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        return {
            "candidateId": candidate_id,
            "matrixIndex": matrix_index,
            "cachePath": str(cache_path),
            "cacheSha256": sha256_file(cache_path),
            "cacheFrameHash": frame_hash(frame),
            "rowCount": len(frame),
            "fingerprints": fingerprints,
            "reused": True,
        }
    with raw_path.open("rb") as handle:
        # Trusted internal S13 object whose identity was hash-frozen before S13X.
        trajectory = pickle.load(handle)
    if (
        trajectory.configuration_id != candidate_id
        or trajectory.matrix_index != matrix_index
    ):
        raise RuntimeError("raw trajectory task identity mismatch")
    frames: list[pd.DataFrame] = []
    fingerprints: list[dict[str, Any]] = []
    for spec in label_specs():
        values, fingerprint = label_trajectory(trajectory, spec)
        frames.append(values)
        fingerprints.append(fingerprint)
    frame = pd.concat(frames, ignore_index=True)
    frame.sort_values(
        ["labelId", "selectedSequenceIndex"],
        kind="stable",
        inplace=True,
        ignore_index=True,
    )
    frame.to_parquet(cache_path, index=False, compression="zstd")
    cache_path.with_suffix(".fingerprints.json").write_text(
        json.dumps(fingerprints, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "candidateId": candidate_id,
        "matrixIndex": matrix_index,
        "cachePath": str(cache_path),
        "cacheSha256": sha256_file(cache_path),
        "cacheFrameHash": frame_hash(frame),
        "rowCount": len(frame),
        "fingerprints": fingerprints,
        "reused": False,
    }


def build_label_caches(workers: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    start = time.perf_counter()
    tasks = [
        (candidate, matrix) for candidate in CANDIDATE_IDS for matrix in range(100)
    ]
    records: list[dict[str, Any]] = []
    fingerprints: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_label_task, candidate, matrix): (candidate, matrix)
            for candidate, matrix in tasks
        }
        for future in as_completed(futures):
            candidate, matrix = futures[future]
            result = future.result()
            fingerprints.extend(result.pop("fingerprints"))
            records.append(result)
            print(
                json.dumps(
                    {
                        "stage": "label_task_complete",
                        "candidateId": candidate,
                        "matrixIndex": matrix,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    manifest = pd.DataFrame(records).sort_values(["candidateId", "matrixIndex"])
    fingerprint_frame = pd.DataFrame(fingerprints).sort_values(
        ["candidateId", "matrixIndex", "labelId"]
    )
    return manifest, fingerprint_frame, time.perf_counter() - start


def validate_label_replay(manifest: pd.DataFrame) -> dict[str, Any]:
    sentinel_tasks = [
        (candidate, matrix) for candidate in CANDIDATE_IDS for matrix in (0, 59, 60, 99)
    ]
    rows = []
    for candidate, matrix in sentinel_tasks:
        original = manifest[
            (manifest["candidateId"] == candidate) & (manifest["matrixIndex"] == matrix)
        ].iloc[0]
        replay = _label_task(candidate, matrix, overwrite=True)
        rows.append(
            {
                "candidateId": candidate,
                "matrixIndex": matrix,
                "expectedFrameHash": original["cacheFrameHash"],
                "replayFrameHash": replay["cacheFrameHash"],
                "expectedRowCount": int(original["rowCount"]),
                "replayRowCount": int(replay["rowCount"]),
                "exact": bool(
                    original["cacheFrameHash"] == replay["cacheFrameHash"]
                    and int(original["rowCount"]) == int(replay["rowCount"])
                ),
            }
        )
    payload = {
        "schema": "eidosoma.e01.s13x_label_replay_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "sentinelTaskCount": len(rows),
        "rows": rows,
        "passed": all(item["exact"] for item in rows),
    }
    write_json(STEP_ROOT / "label_replay_validation.json", payload)
    if not payload["passed"]:
        raise RuntimeError("S13X label replay failed")
    return payload


def _phase(matrix_index: int) -> str:
    return "DEVELOPMENT" if matrix_index in DEVELOPMENT_INDICES else "DIAGNOSTIC"


def label_fingerprint_summary(fingerprints: pd.DataFrame) -> pd.DataFrame:
    work = fingerprints.copy()
    work["analysisPhase"] = work["matrixIndex"].map(_phase)
    rows = []
    for (phase, candidate, label_id), group in work.groupby(
        ["analysisPhase", "candidateId", "labelId"], sort=True
    ):
        rows.append(
            {
                "analysisPhase": phase,
                "candidateId": candidate,
                "labelId": label_id,
                "trajectoryCount": len(group),
                "probabilityMean": float(group["probability"].mean()),
                "probabilityMedian": float(group["probability"].median()),
                "persistenceMean": float(group["persistence"].mean()),
                "persistenceSd": float(group["persistence"].std(ddof=1)),
                "consistencyMean": float(group["consistency"].mean()),
                "timeToFirstMean": float(group["timeToFirst"].mean()),
                "episodeCountMean": float(group["episodeCount"].mean()),
            }
        )
    return pd.DataFrame(rows)


def pipeline_id(
    implementation_id: str,
    metric: str,
    transform: str,
    label_id: str,
    alignment: str,
) -> str:
    identity = f"{implementation_id}|{metric}|{transform}|{label_id}|{alignment}"
    token = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"S13X-P-{token}"


def pipeline_registry() -> pd.DataFrame:
    specs = {item.label_id: item for item in label_specs()}
    rows = []
    sequence = 0
    for implementation_id in IMPLEMENTATIONS:
        for metric in METRICS:
            for transform in TRANSFORMS:
                allowed = (
                    ("SAME_STATE", "NEXT_GENERATION", "PREVIOUS_GENERATION")
                    if transform.startswith("GENERATION_")
                    else ALIGNMENTS
                )
                for label_id, spec in specs.items():
                    for alignment in allowed:
                        rows.append(
                            {
                                "searchSequence": sequence,
                                "pipelineId": pipeline_id(
                                    implementation_id,
                                    metric,
                                    transform,
                                    label_id,
                                    alignment,
                                ),
                                "implementationId": implementation_id,
                                "metric": metric,
                                "transform": transform,
                                "labelId": label_id,
                                "labelFamily": spec.family,
                                "labelEvidenceTier": spec.evidence_tier,
                                "alignment": alignment,
                                "metricEvidenceTier": (
                                    "SOURCE_DEFINED_PRIMARY_OR_COMPARATOR"
                                    if metric in {"emergence", "localPhiR"}
                                    else "SOURCE_ATOM_IDENTITY_EXPLORATION"
                                ),
                                "transformEvidenceTier": (
                                    "DIRECT_PAPER_TEXT"
                                    if transform == "LEVEL"
                                    else (
                                        "DIRECT_FIGURE3_CAPTION"
                                        if transform
                                        in {"BACKWARD_DIFFERENCE", "FORWARD_DIFFERENCE"}
                                        else "SPECULATIVE_TEMPORAL_SUMMARY"
                                    )
                                ),
                            }
                        )
                        sequence += 1
    frame = pd.DataFrame(rows)
    if frame["pipelineId"].duplicated().any():
        raise RuntimeError("pipeline identity collision")
    return frame


def _label_arrays(
    label_frame: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    label_ids = [item.label_id for item in label_specs()]
    max_index = int(label_frame["selectedSequenceIndex"].max())
    matrix = np.full((max_index + 1, len(label_ids)), np.nan, dtype=np.float64)
    for column, label_id in enumerate(label_ids):
        subset = label_frame[label_frame["labelId"] == label_id].sort_values(
            "selectedSequenceIndex", kind="stable"
        )
        indices = subset["selectedSequenceIndex"].to_numpy(dtype=np.int64)
        matrix[indices, column] = subset["isReplicator"].astype(float).to_numpy()
    generations = label_frame["generation"].dropna().astype(int)
    max_generation = max(100, int(generations.max()))
    majority = np.full((max_generation + 2, len(label_ids)), np.nan, dtype=np.float64)
    endpoint = np.full_like(majority, np.nan)
    for column, label_id in enumerate(label_ids):
        subset = label_frame[
            (label_frame["labelId"] == label_id) & (label_frame["generation"] > 0)
        ]
        for generation, group in subset.groupby("generation", sort=True):
            generation = int(generation)
            majority[generation, column] = float(
                group["isReplicator"].astype(bool).mean() >= 0.5
            )
            endpoint[generation, column] = float(
                group.sort_values("selectedSequenceIndex", kind="stable").iloc[-1][
                    "isReplicator"
                ]
            )
    return label_ids, matrix, majority, endpoint


def _aligned_label_matrix(
    transformed: pd.DataFrame,
    label_matrix: np.ndarray,
    generation_majority: np.ndarray,
    generation_endpoint: np.ndarray,
    *,
    transform: str,
    alignment: str,
) -> np.ndarray:
    indices = transformed["selectedSequenceIndex"].to_numpy(dtype=np.int64)
    generations = transformed["generation"].to_numpy(dtype=np.int64)
    generation_level = transform.startswith("GENERATION_")
    if alignment == "SAME_STATE":
        if generation_level:
            source = (
                generation_endpoint
                if transform == "GENERATION_ENDPOINT"
                else generation_majority
            )
            return source[generations]
        return label_matrix[indices]
    if alignment == "NEXT_STATE":
        output = np.full((len(indices), label_matrix.shape[1]), np.nan)
        valid = indices + 1 < label_matrix.shape[0]
        output[valid] = label_matrix[indices[valid] + 1]
        return output
    if alignment == "PREVIOUS_STATE":
        output = np.full((len(indices), label_matrix.shape[1]), np.nan)
        valid = indices > 0
        output[valid] = label_matrix[indices[valid] - 1]
        return output
    if alignment == "NEXT_GENERATION":
        valid = generations + 1 < generation_majority.shape[0]
        output = np.full((len(indices), label_matrix.shape[1]), np.nan)
        output[valid] = generation_majority[generations[valid] + 1]
        return output
    output = np.full((len(indices), label_matrix.shape[1]), np.nan)
    valid = generations > 1
    output[valid] = generation_majority[generations[valid] - 1]
    return output


def _spike_temporal(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return {
            "positiveThreeSigma": False,
            "robustSpike": False,
            "rawLjungBoxSignificant": False,
            "differencedLjungBoxSignificant": False,
        }
    mean, std = float(np.mean(x)), float(np.std(x))
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    robust_scale = 1.4826 * mad
    positive = bool(std > 0 and np.any(x > mean + 3.0 * std))
    robust = bool(robust_scale > 0 and np.any(np.abs(x - median) > 3.0 * robust_scale))
    lag = max(1, min(10, x.size // 5))
    raw_p = float(acorr_ljungbox(x, lags=[lag], return_df=True)["lb_pvalue"].iloc[0])
    dx = np.diff(x)
    diff_lag = max(1, min(10, dx.size // 5))
    diff_p = float(
        acorr_ljungbox(dx, lags=[diff_lag], return_df=True)["lb_pvalue"].iloc[0]
    )
    return {
        "positiveThreeSigma": positive,
        "robustSpike": robust,
        "rawLjungBoxSignificant": bool(raw_p < 0.05),
        "differencedLjungBoxSignificant": bool(diff_p < 0.05),
    }


def evaluate(
    source: pd.DataFrame,
    registry: pd.DataFrame,
    fingerprint_summary: pd.DataFrame,
    *,
    phase: str,
    pipeline_ids: set[str] | None = None,
    include_details: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]],
]:
    indices = DEVELOPMENT_INDICES if phase == "DEVELOPMENT" else DIAGNOSTIC_INDICES
    active_registry = (
        registry
        if pipeline_ids is None
        else registry[registry["pipelineId"].isin(pipeline_ids)]
    )
    active_by_tuple = {
        (
            row.implementationId,
            row.metric,
            row.transform,
            row.labelId,
            row.alignment,
        ): row.pipelineId
        for row in active_registry.itertuples(index=False)
    }
    accumulator: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )
    temporal: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    aggregate: dict[tuple[str, str, str, str], list[pd.DataFrame]] = defaultdict(list)
    details: list[dict[str, Any]] = []
    inference_payloads: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}

    source_groups = {
        (candidate, int(matrix), implementation): group.sort_values(
            "selectedSequenceIndex", kind="stable"
        )
        for (candidate, matrix, implementation), group in source[
            source["matrixIndex"].isin(indices)
        ].groupby(["candidateId", "matrixIndex", "implementationId"], sort=True)
    }
    for candidate in CANDIDATE_IDS:
        for matrix_index in indices:
            label_frame = pd.read_parquet(_label_cache_path(candidate, matrix_index))
            label_ids, label_matrix, generation_majority, generation_endpoint = (
                _label_arrays(label_frame)
            )
            label_column = {value: index for index, value in enumerate(label_ids)}
            for implementation_id in IMPLEMENTATIONS:
                frame = source_groups[(candidate, matrix_index, implementation_id)]
                for metric in METRICS:
                    for transform in TRANSFORMS:
                        relevant = active_registry[
                            (active_registry["implementationId"] == implementation_id)
                            & (active_registry["metric"] == metric)
                            & (active_registry["transform"] == transform)
                        ]
                        if relevant.empty:
                            continue
                        transformed = transform_values(frame, metric, transform)
                        key = (candidate, implementation_id, metric, transform)
                        feature = _spike_temporal(
                            transformed["value"].to_numpy(dtype=float)
                        )
                        temporal[key].append(feature)
                        aggregate[key].append(
                            transformed[["timeIndex", "value"]].copy()
                        )
                        for alignment in relevant["alignment"].unique():
                            aligned = _aligned_label_matrix(
                                transformed,
                                label_matrix,
                                generation_majority,
                                generation_endpoint,
                                transform=transform,
                                alignment=str(alignment),
                            )
                            for label_id in relevant[
                                relevant["alignment"] == alignment
                            ]["labelId"]:
                                pid = active_by_tuple[
                                    (
                                        implementation_id,
                                        metric,
                                        transform,
                                        label_id,
                                        str(alignment),
                                    )
                                ]
                                values = transformed["value"].to_numpy(dtype=float)
                                labels = aligned[:, label_column[label_id]]
                                result = association_summary(values, labels)
                                item = accumulator[(candidate, pid)]
                                for name, value in result.items():
                                    item[name].append(value)
                                if include_details:
                                    details.append(
                                        {
                                            "analysisPhase": phase,
                                            "candidateId": candidate,
                                            "matrixIndex": matrix_index,
                                            "trajectoryId": frame["trajectoryId"].iloc[
                                                0
                                            ],
                                            "pipelineId": pid,
                                            **result,
                                        }
                                    )
                                    mask = np.isfinite(values) & np.isfinite(labels)
                                    inference_payloads[
                                        (pid, candidate, matrix_index)
                                    ] = (
                                        values[mask].astype(float),
                                        labels[mask].astype(float),
                                    )
            print(
                json.dumps(
                    {
                        "stage": f"{phase.lower()}_analysis_task_complete",
                        "candidateId": candidate,
                        "matrixIndex": matrix_index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    temporal_summary: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for key, features in temporal.items():
        frames = pd.concat(aggregate[key], ignore_index=True)
        grouped = (
            frames.dropna(subset=["value"])
            .groupby("timeIndex", as_index=False)["value"]
            .median()
            .sort_values("timeIndex")
        )
        trend_p = None
        trend_slope = None
        if len(grouped) >= 3:
            fit = linregress(grouped["timeIndex"], grouped["value"])
            trend_p = float(fit.pvalue)
            trend_slope = float(fit.slope)
        temporal_summary[key] = {
            "positiveThreeSigmaRunFraction": float(
                np.mean([item["positiveThreeSigma"] for item in features])
            ),
            "robustSpikeRunFraction": float(
                np.mean([item["robustSpike"] for item in features])
            ),
            "rawLjungBoxFraction": float(
                np.mean([item["rawLjungBoxSignificant"] for item in features])
            ),
            "differencedLjungBoxFraction": float(
                np.mean([item["differencedLjungBoxSignificant"] for item in features])
            ),
            "aggregateTrendSlope": trend_slope,
            "aggregateTrendP": trend_p,
            "trajectoryCount": len(features),
        }

    registry_lookup = registry.set_index("pipelineId")
    rows = []
    for (candidate, pid), values in sorted(accumulator.items()):
        meta = registry_lookup.loc[pid]
        rhos = np.asarray(
            [value for value in values["rho"] if value is not None], dtype=float
        )
        pvalues = np.asarray(
            [
                p
                for rho, p in zip(
                    values["rho"], values["ordinaryTwoSidedP"], strict=True
                )
                if rho is not None and p is not None
            ],
            dtype=float,
        )
        rho_for_p = np.asarray(
            [
                rho
                for rho, p in zip(
                    values["rho"], values["ordinaryTwoSidedP"], strict=True
                )
                if rho is not None and p is not None
            ],
            dtype=float,
        )
        differences = np.asarray(
            [value for value in values["meanDifference"] if value is not None],
            dtype=float,
        )
        standardized = np.asarray(
            [
                value
                for value in values["standardizedMeanDifference"]
                if value is not None
            ],
            dtype=float,
        )
        fp = fingerprint_summary[
            (fingerprint_summary["analysisPhase"] == phase)
            & (fingerprint_summary["candidateId"] == candidate)
            & (fingerprint_summary["labelId"] == meta["labelId"])
        ].iloc[0]
        temporal_key = (
            candidate,
            str(meta["implementationId"]),
            str(meta["metric"]),
            str(meta["transform"]),
        )
        row = {
            "analysisPhase": phase,
            "candidateId": candidate,
            "pipelineId": pid,
            "implementationId": meta["implementationId"],
            "metric": meta["metric"],
            "transform": meta["transform"],
            "labelId": meta["labelId"],
            "labelFamily": meta["labelFamily"],
            "labelEvidenceTier": meta["labelEvidenceTier"],
            "alignment": meta["alignment"],
            "definedTrajectoryCount": int(rhos.size),
            "positiveCorrelationCount": int(np.count_nonzero(rhos > 0)),
            "positiveCorrelationFraction": float(np.mean(rhos > 0))
            if rhos.size
            else None,
            "positiveSignificantCount": int(
                np.count_nonzero((rho_for_p > 0) & (pvalues < 0.05))
            ),
            "positiveSignificantFraction": (
                float(np.mean((rho_for_p > 0) & (pvalues < 0.05)))
                if pvalues.size
                else None
            ),
            "meanCorrelation": float(np.mean(rhos)) if rhos.size else None,
            "medianCorrelation": float(np.median(rhos)) if rhos.size else None,
            "higherDuringReplicationCount": int(np.count_nonzero(differences > 0)),
            "higherDuringReplicationFraction": (
                float(np.mean(differences > 0)) if differences.size else None
            ),
            "medianMeanDifference": (
                float(np.median(differences)) if differences.size else None
            ),
            "medianStandardizedMeanDifference": (
                float(np.median(standardized)) if standardized.size else None
            ),
            "labelProbabilityMean": float(fp["probabilityMean"]),
            "labelPersistenceMean": float(fp["persistenceMean"]),
            "labelConsistencyMean": float(fp["consistencyMean"]),
            "labelTimeToFirstMean": float(fp["timeToFirstMean"]),
            **temporal_summary[temporal_key],
        }
        row.update(resemblance_score(row))
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(details), inference_payloads


def ensemble_ranking(candidate_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pipeline_id_value, group in candidate_results.groupby("pipelineId", sort=True):
        if set(group["candidateId"]) != set(CANDIDATE_IDS):
            continue
        scores = group["directionalResemblanceScore"].to_numpy(float)
        medians = group["medianCorrelation"].fillna(-np.inf).to_numpy(float)
        drifts = group["higherDuringReplicationFraction"].fillna(0).to_numpy(float)
        rows.append(
            {
                "pipelineId": pipeline_id_value,
                "meanCandidateScore": float(np.mean(scores)),
                "minimumCandidateScore": float(np.min(scores)),
                "ensembleDirectionalScore": float(
                    0.6 * np.mean(scores) + 0.4 * np.min(scores)
                ),
                "bothMedianCorrelationsPositive": bool(np.all(medians > 0)),
                "bothHigherDuringReplicationMajority": bool(np.all(drifts > 0.5)),
                "candidateScoreRange": float(np.ptp(scores)),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["ensembleDirectionalScore", "pipelineId"],
        ascending=[False, True],
        ignore_index=True,
    )
    result.insert(0, "developmentRank", np.arange(1, len(result) + 1))
    return result


def _transform_group(transform: str) -> str:
    if "DIFFERENCE" in transform:
        return "DIFFERENCE"
    if transform.startswith("GENERATION_"):
        return "GENERATION"
    if transform.startswith("TRAILING_"):
        return "TRAILING"
    return "LEVEL"


def select_diagnostic_pipelines(
    ranking: pd.DataFrame, registry: pd.DataFrame, count: int = 12
) -> pd.DataFrame:
    merged = ranking.merge(registry, on="pipelineId", how="left", validate="one_to_one")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: pd.Series, reason: str) -> None:
        pid = str(row["pipelineId"])
        if pid in seen or len(selected) >= count:
            return
        selected.append({**row.to_dict(), "selectionReason": reason})
        seen.add(pid)

    baseline = merged[
        (merged["implementationId"] == "IIGR_CORRECTED_SOURCE")
        & (merged["metric"] == "emergence")
        & (merged["transform"] == "LEVEL")
        & (merged["labelId"] == "PF_HISTORICAL_ADJACENT_AVERAGE_H090")
        & (merged["alignment"] == "SAME_STATE")
    ]
    if not baseline.empty:
        add(baseline.iloc[0], "HISTORICAL_REFERENCE_PIPELINE")
    add(merged.iloc[0], "TOP_CONTINUOUS_RESEMBLANCE")
    for metric in METRICS:
        subset = merged[merged["metric"] == metric]
        if not subset.empty:
            add(subset.iloc[0], f"BEST_{metric.upper()}_SCALAR")
    for group_name in ("DIFFERENCE", "GENERATION", "TRAILING", "LEVEL"):
        subset = merged[merged["transform"].map(_transform_group) == group_name]
        if not subset.empty:
            add(subset.iloc[0], f"BEST_{group_name}_TEMPORAL_REPRESENTATION")
    for prefix, reason in (
        ("POSTFISSION", "BEST_POSTFISSION_LABEL_FAMILY"),
        ("MOLECULAR", "BEST_MOLECULAR_LABEL_FAMILY"),
    ):
        subset = merged[merged["labelFamily"].str.startswith(prefix)]
        if not subset.empty:
            add(subset.iloc[0], reason)
    for _, row in merged.iterrows():
        add(row, "RANK_FILL_WITH_DIVERSITY_ALREADY_REPRESENTED")
        if len(selected) == count:
            break
    result = pd.DataFrame(selected)
    result.insert(0, "diagnosticSelectionOrder", np.arange(1, len(result) + 1))
    return result


def _circular_vectors(
    values: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if values.size < 3 or np.unique(values).size < 2 or np.unique(labels).size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    ranked = pd.Series(values).rank(method="average").to_numpy(float)
    ranked = (ranked - ranked.mean()) / ranked.std()
    binary = labels.astype(float)
    binary_standard = (binary - binary.mean()) / binary.std()
    correlations = np.fft.ifft(
        np.conj(np.fft.fft(ranked)) * np.fft.fft(binary_standard)
    ).real / len(ranked)
    count_positive = float(binary.sum())
    count_negative = float(len(binary) - count_positive)
    raw_cross = np.fft.ifft(np.conj(np.fft.fft(values)) * np.fft.fft(binary)).real
    total = float(values.sum())
    differences = raw_cross / count_positive - (total - raw_cross) / count_negative
    return correlations, differences


def diagnostic_inference(
    candidate_results: pd.DataFrame,
    details: pd.DataFrame,
    payloads: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for (pipeline_id_value, candidate), group in details.groupby(
        ["pipelineId", "candidateId"], sort=True
    ):
        rhos = group["rho"].dropna().to_numpy(float)
        differences = group["meanDifference"].dropna().to_numpy(float)
        summary = candidate_results[
            (candidate_results["pipelineId"] == pipeline_id_value)
            & (candidate_results["candidateId"] == candidate)
        ].iloc[0]
        if rhos.size == 0 or differences.size == 0:
            rows.append(
                {
                    "pipelineId": pipeline_id_value,
                    "candidateId": candidate,
                    "status": "INELIGIBLE_UNDEFINED_TRAJECTORY_STATISTICS",
                    "definedCorrelationCount": len(rhos),
                    "medianCorrelation": summary["medianCorrelation"],
                    "correlationBootstrapLower95": None,
                    "correlationBootstrapUpper95": None,
                    "circularShiftPositiveP": None,
                    "definedDriftCount": len(differences),
                    "medianMeanDifference": summary["medianMeanDifference"],
                    "driftBootstrapLower95": None,
                    "driftBootstrapUpper95": None,
                    "blockAwareCircularPositiveP": None,
                    "bootstrapReplicates": 0,
                    "circularShiftReplicates": 0,
                }
            )
            continue
        rng_boot = np.random.default_rng(
            derive_seed("diagnostic", pipeline_id_value, candidate, "bootstrap")
        )
        rng_shift = np.random.default_rng(
            derive_seed("diagnostic", pipeline_id_value, candidate, "circular_shift")
        )
        boot_rho = np.median(
            rhos[
                rng_boot.integers(0, len(rhos), size=(BOOTSTRAP_REPLICATES, len(rhos)))
            ],
            axis=1,
        )
        boot_diff = np.median(
            differences[
                rng_boot.integers(
                    0,
                    len(differences),
                    size=(BOOTSTRAP_REPLICATES, len(differences)),
                )
            ],
            axis=1,
        )
        circular_columns = []
        drift_columns = []
        for matrix_index in sorted(group["matrixIndex"].astype(int)):
            values, labels = payloads[(pipeline_id_value, candidate, matrix_index)]
            correlations, drift_values = _circular_vectors(values, labels)
            if correlations.size > 1:
                offsets = rng_shift.integers(
                    1, len(correlations), size=SHIFT_REPLICATES
                )
                circular_columns.append(correlations[offsets])
                drift_columns.append(drift_values[offsets])
        circular_null = (
            np.median(np.column_stack(circular_columns), axis=1)
            if circular_columns
            else np.asarray([])
        )
        drift_null = (
            np.median(np.column_stack(drift_columns), axis=1)
            if drift_columns
            else np.asarray([])
        )
        observed_rho = float(summary["medianCorrelation"])
        observed_diff = float(summary["medianMeanDifference"])
        rows.append(
            {
                "pipelineId": pipeline_id_value,
                "candidateId": candidate,
                "status": "VALID",
                "definedCorrelationCount": len(rhos),
                "medianCorrelation": observed_rho,
                "correlationBootstrapLower95": float(np.quantile(boot_rho, 0.025)),
                "correlationBootstrapUpper95": float(np.quantile(boot_rho, 0.975)),
                "circularShiftPositiveP": float(
                    (1 + np.count_nonzero(circular_null >= observed_rho))
                    / (1 + len(circular_null))
                ),
                "definedDriftCount": len(differences),
                "medianMeanDifference": observed_diff,
                "driftBootstrapLower95": float(np.quantile(boot_diff, 0.025)),
                "driftBootstrapUpper95": float(np.quantile(boot_diff, 0.975)),
                "blockAwareCircularPositiveP": float(
                    (1 + np.count_nonzero(drift_null >= observed_diff))
                    / (1 + len(drift_null))
                ),
                "bootstrapReplicates": BOOTSTRAP_REPLICATES,
                "circularShiftReplicates": SHIFT_REPLICATES,
            }
        )
    return pd.DataFrame(rows)


def append_screen_ledger(ranking: pd.DataFrame, registry: pd.DataFrame) -> None:
    path = STEP_ROOT / "chronological_search_ledger.csv"
    existing = pd.read_csv(path)
    merged = ranking.merge(registry, on="pipelineId", how="left", validate="one_to_one")
    rows = []
    for row in merged.sort_values("searchSequence").itertuples(index=False):
        rows.append(
            {
                "attemptSequence": 4 + int(row.searchSequence),
                "attemptId": f"S13X-SCREEN-A{int(row.searchSequence):05d}",
                "phase": "DEVELOPMENT_SYSTEMATIC_SCREEN",
                "choiceFamily": "LABEL_METRIC_TRANSFORM_ALIGNMENT",
                "specification": json.dumps(
                    {
                        "implementationId": row.implementationId,
                        "metric": row.metric,
                        "transform": row.transform,
                        "labelId": row.labelId,
                        "alignment": row.alignment,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "evidenceTier": (
                    f"{row.labelEvidenceTier}/{row.metricEvidenceTier}/"
                    f"{row.transformEvidenceTier}"
                ),
                "outcome": (
                    f"rank={int(row.developmentRank)}; ensembleScore={row.ensembleDirectionalScore:.6f}; "
                    f"bothMedianPositive={row.bothMedianCorrelationsPositive}; "
                    f"bothDriftMajority={row.bothHigherDuringReplicationMajority}"
                ),
                "negativeResult": bool(
                    not row.bothMedianCorrelationsPositive
                    or not row.bothHigherDuringReplicationMajority
                ),
                "selectionUse": "Eligible for diversity-controlled diagnostic ranking; no confirmatory meaning.",
            }
        )
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    write_csv(path, combined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    input_validation = validate_frozen_inputs()
    source = pd.read_parquet(SOURCE_VALUES)
    if len(source) != 351_148 or set(source["candidateId"]) != set(CANDIDATE_IDS):
        raise RuntimeError("unexpected frozen S13RRR full-source table")
    manifest, fingerprints, label_seconds = build_label_caches(args.workers)
    replay = validate_label_replay(manifest)
    write_csv(STEP_ROOT / "label_cache_manifest.csv", manifest.drop(columns=[]))
    write_parquet(STEP_ROOT / "label_fingerprints.parquet", fingerprints)
    fingerprint_summary = label_fingerprint_summary(fingerprints)
    write_csv(STEP_ROOT / "label_fingerprint_summary.csv", fingerprint_summary)

    registry = pipeline_registry()
    write_csv(STEP_ROOT / "pipeline_search_registry.csv", registry)
    development, _, _ = evaluate(
        source,
        registry,
        fingerprint_summary,
        phase="DEVELOPMENT",
        include_details=False,
    )
    ranking = ensemble_ranking(development)
    selected = select_diagnostic_pipelines(ranking, registry, count=12)
    write_csv(STEP_ROOT / "development_pipeline_results.csv", development)
    write_csv(STEP_ROOT / "development_ensemble_ranking.csv", ranking)
    write_csv(STEP_ROOT / "diagnostic_candidate_registry.csv", selected)
    append_screen_ledger(ranking, registry)

    selected_ids = set(selected["pipelineId"])
    diagnostic, details, payloads = evaluate(
        source,
        registry,
        fingerprint_summary,
        phase="DIAGNOSTIC",
        pipeline_ids=selected_ids,
        include_details=True,
    )
    diagnostic_rank = ensemble_ranking(diagnostic)
    inference = diagnostic_inference(diagnostic, details, payloads)
    write_csv(STEP_ROOT / "diagnostic_pipeline_results.csv", diagnostic)
    write_parquet(STEP_ROOT / "diagnostic_trajectory_results.parquet", details)
    write_csv(STEP_ROOT / "diagnostic_ensemble_ranking.csv", diagnostic_rank)
    write_csv(STEP_ROOT / "diagnostic_inference.csv", inference)

    top = diagnostic_rank.iloc[0]
    top_candidates = diagnostic[diagnostic["pipelineId"] == top["pipelineId"]]
    result_payload = {
        "schema": "eidosoma.e01.s13x_stage1_result.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "adaptiveOutcomeGuided": True,
        "pipelineSpecificationCount": len(registry),
        "developmentCandidateResultCount": len(development),
        "diagnosticPipelineCount": len(selected),
        "diagnosticCandidateResultCount": len(diagnostic),
        "topDiagnosticPipelineId": top["pipelineId"],
        "topDiagnosticEnsembleScore": float(top["ensembleDirectionalScore"]),
        "topDiagnosticBothMedianPositive": bool(top["bothMedianCorrelationsPositive"]),
        "topDiagnosticBothDriftMajority": bool(
            top["bothHigherDuringReplicationMajority"]
        ),
        "topCandidateResults": top_candidates.to_dict("records"),
        "notConfirmatory": True,
    }
    write_json(STEP_ROOT / "stage1_result.json", result_payload)
    runtime = {
        "schema": "eidosoma.e01.s13x_stage1_runtime.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "workers": args.workers,
        "blasThreadsPerWorker": 1,
        "labelSeconds": label_seconds,
        "totalWallSeconds": time.perf_counter() - start,
        "gpuUsed": False,
        "repositoryHead": git("rev-parse", "HEAD"),
        "repositoryRemoteHead": git("rev-parse", "origin/eidosoma/groups/42"),
    }
    write_json(STEP_ROOT / "stage1_runtime.json", runtime)
    validation = {
        "schema": "eidosoma.e01.s13x_stage1_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "inputValidationPassed": input_validation["passed"],
        "labelTaskCount": len(manifest),
        "labelSpecificationCount": len(label_specs()),
        "labelCacheRows": int(manifest["rowCount"].sum()),
        "labelReplayPassed": replay["passed"],
        "pipelineSpecificationCount": len(registry),
        "developmentExpectedRows": 2 * len(registry),
        "developmentObservedRows": len(development),
        "diagnosticPipelineCount": len(selected),
        "diagnosticExpectedRows": 2 * len(selected),
        "diagnosticObservedRows": len(diagnostic),
        "diagnosticTrajectoryExpectedRows": 2 * len(selected) * len(DIAGNOSTIC_INDICES),
        "diagnosticTrajectoryObservedRows": len(details),
        "inferenceExpectedRows": 2 * len(selected),
        "inferenceObservedRows": len(inference),
        "developmentDiagnosticOverlap": sorted(
            set(DEVELOPMENT_INDICES) & set(DIAGNOSTIC_INDICES)
        ),
        "passed": bool(
            input_validation["passed"]
            and replay["passed"]
            and len(manifest) == 200
            and len(development) == 2 * len(registry)
            and len(diagnostic) == 2 * len(selected)
            and len(details) == 2 * len(selected) * len(DIAGNOSTIC_INDICES)
            and len(inference) == 2 * len(selected)
            and not (set(DEVELOPMENT_INDICES) & set(DIAGNOSTIC_INDICES))
        ),
    }
    write_json(STEP_ROOT / "stage1_validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError("S13X stage-1 validation failed")
    print(json.dumps(result_payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
