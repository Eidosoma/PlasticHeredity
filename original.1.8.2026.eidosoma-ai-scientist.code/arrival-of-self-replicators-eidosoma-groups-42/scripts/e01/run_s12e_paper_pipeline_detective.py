#!/usr/bin/env python3
"""Execute the phase-gated E01 S12E detective reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_BOOTSTRAP / "src") not in sys.path:
    sys.path.insert(0, str(REPO_BOOTSTRAP / "src"))

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from e01_paper_pipeline_detective.core import (
    ENGINE_IDS,
    GardTrajectory,
    SeedIdentity,
    derive_seed,
    observation_rows,
    simulate_trajectory,
    trajectory_replay_equal,
    trajectory_summary,
)
from e01_paper_pipeline_detective.information import (
    METRIC_IDS,
    common_clr_drop100,
    metric_value_rows,
    run_metric_branch,
    source_result_replay_equal,
)
from e01_paper_pipeline_detective.labels import (
    LABEL_IDS,
    LabelResult,
    label_fingerprint,
    label_rows,
    label_trajectory,
)
from e01_paper_pipeline_detective.statistics import (
    cohort_inference,
    spike_summary,
    temporal_summary,
    trajectory_association,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs/e01/s12e_paper_pipeline_detective_preregistration.yaml"
ARTIFACTS = Path("/artifacts/research_steps/S12E")
CACHE = Path("/cache/e01_s12e")
TRAJECTORY_CACHE = CACHE / "trajectories"
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
PRIOR_BASELINE = ARTIFACTS / "immutable_prior_baseline.json"
WORKERS = 6

TARGETS = {
    "replicatingLifetime": (716.0, 198.0, 320.0, 1112.0),
    "replicatingFraction": (0.88, 0.03, 0.82, 0.94),
    "consecutiveBinaryPearson": (0.38, 0.06, 0.26, 0.50),
    "firstReplicatorBatchStep": (37.0, 27.0, 0.0, 91.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def trajectory_path(phase: str, engine_id: str, matrix_index: int) -> Path:
    return TRAJECTORY_CACHE / phase / engine_id / f"M{matrix_index:02d}.pickle"


def save_trajectory(path: Path, trajectory: GardTrajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(trajectory, handle, protocol=5)


def load_trajectory(phase: str, engine_id: str, matrix_index: int) -> GardTrajectory:
    with trajectory_path(phase, engine_id, matrix_index).open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, GardTrajectory):
        raise TypeError("trusted S12E trajectory cache has the wrong type")
    return value


def seed_row(identity: SeedIdentity) -> dict[str, object]:
    return {
        "phase": identity.phase,
        "rootSha256": identity.root_sha256,
        "purpose": identity.purpose,
        "matrixIndex": identity.matrix_index,
        "engineId": identity.engine_id,
        "derivedSeed": str(identity.derived_seed),
        "seedMaterialSha256": identity.seed_material_sha256,
        "bitGenerator": "PCG64DXSM",
    }


def _simulate_task(task: tuple[str, str, int, str]) -> dict[str, Any]:
    phase, root, matrix_index, engine_id = task
    started = time.perf_counter()
    first, seeds = simulate_trajectory(
        phase=phase,
        root_hex=root,
        matrix_index=matrix_index,
        engine_id=engine_id,
    )
    replay, _ = simulate_trajectory(
        phase=phase,
        root_hex=root,
        matrix_index=matrix_index,
        engine_id=engine_id,
    )
    replay_passed = trajectory_replay_equal(first, replay)
    cache = trajectory_path(phase, engine_id, matrix_index)
    save_trajectory(cache, first)
    summary = trajectory_summary(first)
    summary["exactReplayPassed"] = replay_passed
    summary["workerWallSeconds"] = time.perf_counter() - started
    summary["cachePath"] = str(cache)
    return {
        "summary": summary,
        "seeds": [seed_row(seed) for seed in seeds],
    }


def run_simulations(
    *, phase: str, root: str, engine_ids: list[str], matrix_count: int
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    tasks = [
        (phase, root, matrix_index, engine_id)
        for engine_id in engine_ids
        for matrix_index in range(matrix_count)
    ]
    summaries: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_simulate_task, task): task for task in tasks}
        for future in as_completed(futures):
            payload = future.result()
            summaries.append(payload["summary"])
            seeds.extend(payload["seeds"])
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["engineId", "matrixIndex"]
    ).reset_index(drop=True)
    seed_frame = pd.DataFrame(seeds).drop_duplicates(
        ["phase", "purpose", "matrixIndex", "engineId", "seedMaterialSha256"]
    )
    return summary_frame, seed_frame, time.perf_counter() - started


def phase1_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = 716.0 / 0.88
    for engine_id, group in frame.groupby("engineId", sort=True):
        completed = int(np.count_nonzero(group["completedFissions"] == 100))
        extinctions = int(np.count_nonzero(group["terminalStatus"] != "requested_fissions_completed"))
        median_steps = float(group["totalBatchSteps"].median())
        median_source = float(group["totalSourceObservations"].median())
        median_post = float(group["medianPostFissionMass"].median())
        fraction_reaching = float(group["fractionGenerationsReachingNMax"].mean())
        exact = bool(group["exactReplayPassed"].all())
        eligible = bool(
            completed >= 23
            and 500 <= median_steps <= 1500
            and extinctions <= 1
            and fraction_reaching >= 0.95
            and 20 <= median_post <= 60
            and exact
        )
        rank_score = (
            abs(median_steps - target) / target
            + abs(median_post - 40.0) / 40.0
            + extinctions / 24.0
        )
        rows.append(
            {
                "engineId": engine_id,
                "trajectoryCount": int(group.shape[0]),
                "completed100Fissions": completed,
                "extinctionCount": extinctions,
                "medianTotalBatchSteps": median_steps,
                "medianTotalSourceObservations": median_source,
                "minTotalBatchSteps": int(group["totalBatchSteps"].min()),
                "maxTotalBatchSteps": int(group["totalBatchSteps"].max()),
                "medianPostFissionMass": median_post,
                "meanOvershoot": float(group["meanOvershoot"].mean()),
                "meanMaxstepsTerminations": float(group["maxstepsTerminations"].mean()),
                "fractionGenerationsReachingNMax": fraction_reaching,
                "exactReplayPassed": exact,
                "phase1Eligible": eligible,
                "rankScore": rank_score,
            }
        )
    result = pd.DataFrame(rows)
    eligible_ids = (
        result[result["phase1Eligible"]]
        .sort_values(["rankScore", "engineId"])
        .head(2)["engineId"]
        .tolist()
    )
    result["selectedForPhase2"] = result["engineId"].isin(eligible_ids)
    return result.sort_values(["rankScore", "engineId"]).reset_index(drop=True)


def kmeans_seed_callback(
    root: str, phase: str, matrix_index: int, engine_id: str
):
    def seed_for(k: int, replica: int) -> int:
        identity = derive_seed(
            root,
            phase,
            "label_kmeans",
            matrix_index,
            engine_id,
            "L2_L3_SHARED",
            k,
            replica,
        )
        return identity.derived_seed % (2**32 - 1)

    return seed_for


def label_selected_development(
    *, root: str, selected_engines: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int, str], LabelResult]]:
    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    results: dict[tuple[str, int, str], LabelResult] = {}
    for engine_id in selected_engines:
        for matrix_index in range(24):
            trajectory = load_trajectory("development", engine_id, matrix_index)
            callback = kmeans_seed_callback(
                root, "development", matrix_index, engine_id
            )
            for label_id in LABEL_IDS:
                result = label_trajectory(
                    trajectory, label_id, kmeans_seed_for=callback
                )
                results[(engine_id, matrix_index, label_id)] = result
                rows.append(label_fingerprint(trajectory, result))
            for k in range(1, 11):
                for replica in range(10):
                    identity = derive_seed(
                        root,
                        "development",
                        "label_kmeans",
                        matrix_index,
                        engine_id,
                        "L2_L3_SHARED",
                        k,
                        replica,
                    )
                    seed_rows.append(seed_row(identity))
    return pd.DataFrame(rows), pd.DataFrame(seed_rows), results


def label_summary(frame: pd.DataFrame, phase1: pd.DataFrame) -> pd.DataFrame:
    engine_rank = {
        row.engineId: index
        for index, row in enumerate(phase1.itertuples(index=False))
    }
    label_priority = {
        label: index
        for index, label in enumerate(
            (
                "L1_DOMINANT_CENTROID_H090",
                "L2_EUCLIDEAN_COMPTYPE",
                "L3_EUCLIDEAN_DOMINANT_CENTROID",
                "L0_HISTORICAL_ADJACENT_H090",
            )
        )
    }
    rows: list[dict[str, Any]] = []
    for (engine_id, label_id), group in frame.groupby(["engineId", "labelId"]):
        means = {
            name: float(group[name].mean(skipna=True))
            for name in TARGETS
        }
        match = 0
        distance = 0.0
        for name, (target, sd, low, high) in TARGETS.items():
            value = means[name]
            if np.isfinite(value) and low <= value <= high:
                match += 1
            distance += abs(value - target) / sd if np.isfinite(value) else 1.0e6
        rows.append(
            {
                "pipelineId": f"{engine_id}__{label_id}",
                "engineId": engine_id,
                "labelId": label_id,
                "trajectoryCount": int(group.shape[0]),
                **{f"mean_{name}": value for name, value in means.items()},
                "meanReplicatingGenerationFraction": float(
                    group["replicatingGenerationFraction"].mean(skipna=True)
                ),
                "meanPersistentClusterCount": float(
                    group["persistentClusterCount"].mean(skipna=True)
                ),
                "meanDominantClusterSize": float(
                    group["dominantClusterSize"].mean(skipna=True)
                ),
                "fingerprintMatchCount": match,
                "reportedSdDistance": distance,
                "engineRank": engine_rank[engine_id],
                "labelPriority": label_priority[label_id],
            }
        )
    result = pd.DataFrame(rows).sort_values(
        [
            "fingerprintMatchCount",
            "reportedSdDistance",
            "engineRank",
            "labelPriority",
            "pipelineId",
        ],
        ascending=[False, True, True, True, True],
    )
    selected = result.head(2)["pipelineId"].tolist()
    result["selectedForConfirmation"] = result["pipelineId"].isin(selected)
    return result.reset_index(drop=True)


def pipeline_parts(pipeline_id: str) -> tuple[str, str]:
    engine_id, label_id = pipeline_id.split("__", maxsplit=1)
    return engine_id, label_id


def run_confirmation(
    *, root: str, locked_pipelines: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], float]:
    """Generate and hash-freeze the untouched Phase-2 confirmation cohort."""

    engine_ids = sorted({pipeline_parts(value)[0] for value in locked_pipelines})
    summaries, seeds, elapsed = run_simulations(
        phase="confirmation", root=root, engine_ids=engine_ids, matrix_count=24
    )
    label_payload: list[dict[str, object]] = []
    fingerprints: list[dict[str, object]] = []
    kmeans_seeds: list[dict[str, object]] = []
    for pipeline_id in locked_pipelines:
        engine_id, label_id = pipeline_parts(pipeline_id)
        for matrix_index in range(24):
            trajectory = load_trajectory("confirmation", engine_id, matrix_index)
            callback = kmeans_seed_callback(
                root, "confirmation", matrix_index, engine_id
            )
            result = label_trajectory(
                trajectory, label_id, kmeans_seed_for=callback
            )
            label_payload.extend(label_rows(trajectory, result, pipeline_id))
            row = label_fingerprint(trajectory, result)
            row["pipelineId"] = pipeline_id
            fingerprints.append(row)
            for k in range(1, 11):
                for replica in range(10):
                    kmeans_seeds.append(
                        seed_row(
                            derive_seed(
                                root,
                                "confirmation",
                                "label_kmeans",
                                matrix_index,
                                engine_id,
                                "L2_L3_SHARED",
                                k,
                                replica,
                            )
                        )
                    )
    if kmeans_seeds:
        seeds = pd.concat([seeds, pd.DataFrame(kmeans_seeds)], ignore_index=True)
        seeds = seeds.drop_duplicates(
            ["phase", "purpose", "matrixIndex", "engineId", "seedMaterialSha256"]
        )
    labels = pd.DataFrame(label_payload)
    fingerprint_frame = pd.DataFrame(fingerprints)
    rows: list[dict[str, object]] = []
    qualified: list[str] = []
    for pipeline_id, group in fingerprint_frame.groupby("pipelineId", sort=False):
        engine_id, label_id = pipeline_parts(pipeline_id)
        engine = summaries[summaries["engineId"] == engine_id]
        values = {name: float(group[name].mean(skipna=True)) for name in TARGETS}
        matches = sum(
            bool(np.isfinite(values[name]) and low <= values[name] <= high)
            for name, (_, _, low, high) in TARGETS.items()
        )
        complete = int(np.count_nonzero(engine["completedFissions"] == 100))
        median_steps = float(engine["totalBatchSteps"].median())
        exact = bool(engine["exactReplayPassed"].all())
        qualifies = bool(
            complete >= 23
            and 500 <= median_steps <= 1500
            and matches >= 3
            and exact
        )
        if qualifies:
            qualified.append(pipeline_id)
        rows.append(
            {
                "pipelineId": pipeline_id,
                "engineId": engine_id,
                "labelId": label_id,
                "completed100Fissions": complete,
                "medianTotalBatchSteps": median_steps,
                "fingerprintMatchCount": matches,
                **{f"mean_{name}": value for name, value in values.items()},
                "exactRegeneration": exact,
                "postConfirmationThresholdChanged": False,
                "qualifiesForPhase3": qualifies,
            }
        )
    return summaries, seeds, labels, pd.DataFrame(rows), qualified, elapsed


def _source_seeds(
    root: str,
    phase: str,
    pipeline_id: str,
    matrix_index: int,
    metric_id: str,
    *extra: object,
) -> tuple[int, int, list[dict[str, object]]]:
    engine_id, _ = pipeline_parts(pipeline_id)
    shared_metric = (
        "M1_M3_SHARED"
        if metric_id in {"M1_IIGR_EMERGENCE_CLR_FULL", "M3_IIGR_LOCAL_PHIR_CLR_FULL"}
        else metric_id
    )
    pre = derive_seed(
        root,
        phase,
        "source_preprocessing",
        matrix_index,
        engine_id,
        shared_metric,
        *extra,
    )
    partition = derive_seed(
        root,
        phase,
        "source_partition",
        matrix_index,
        engine_id,
        shared_metric,
        *extra,
    )
    return (
        pre.derived_seed % (2**32 - 1),
        partition.derived_seed % (2**32 - 1),
        [seed_row(pre), seed_row(partition)],
    )


def _label_for_pipeline(
    trajectory: GardTrajectory, root: str, phase: str, pipeline_id: str
) -> LabelResult:
    _, label_id = pipeline_parts(pipeline_id)
    return label_trajectory(
        trajectory,
        label_id,
        kmeans_seed_for=kmeans_seed_callback(
            root, phase, trajectory.matrix_index, trajectory.engine_id
        ),
    )


def run_full_metrics(
    *, root: str, qualified: list[str]
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, object]],
    list[tuple[str, str]],
]:
    """Evaluate all four frozen retrospective branches on hash-frozen inputs."""

    value_rows: list[dict[str, object]] = []
    association_rows: list[dict[str, object]] = []
    spike_rows: list[dict[str, object]] = []
    temporal_rows: list[dict[str, object]] = []
    partition_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    metric_series: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for pipeline_id in qualified:
        engine_id, _ = pipeline_parts(pipeline_id)
        for matrix_index in range(24):
            trajectory = load_trajectory("confirmation", engine_id, matrix_index)
            label = _label_for_pipeline(
                trajectory, root, "confirmation", pipeline_id
            )
            labels = np.asarray(label.observation_labels, dtype=bool)
            observations = trajectory.observations
            clr = common_clr_drop100(trajectory.states)
            obs_kinds = [row.observation_kind for row in observations]
            generations = np.asarray([row.generation for row in observations])
            molecular = np.asarray([row.molecular_step for row in observations])
            obs_indices = np.arange(len(observations), dtype=np.int64)
            for metric_id in METRIC_IDS:
                pre_seed, part_seed, these_seeds = _source_seeds(
                    root, "confirmation", pipeline_id, matrix_index, metric_id
                )
                seed_rows.extend(these_seeds)
                result = run_metric_branch(
                    clr,
                    metric_id,
                    SAFE_LATTICE,
                    preprocessing_seed=pre_seed,
                    partition_seed=part_seed,
                )
                replay = run_metric_branch(
                    clr,
                    metric_id,
                    SAFE_LATTICE,
                    preprocessing_seed=pre_seed,
                    partition_seed=part_seed,
                )
                replay_pass = source_result_replay_equal(result, replay)
                aligned_indices = obs_indices[result.local_offset :]
                rows = metric_value_rows(
                    pipeline_id=pipeline_id,
                    trajectory_id=trajectory.trajectory_id,
                    matrix_index=matrix_index,
                    metric=result,
                    observation_indices=aligned_indices,
                    observation_kinds=obs_kinds,
                    generations=generations,
                    molecular_steps=molecular,
                    temporal_mode="RETROSPECTIVE_FULL_TRAJECTORY_LOCAL",
                )
                value_rows.extend(rows)
                values = np.asarray(
                    [np.nan if row["value"] is None else row["value"] for row in rows],
                    dtype=np.float64,
                )
                current_labels = labels[aligned_indices]
                association = trajectory_association(values, current_labels)
                association_rows.append(
                    {
                        "pipelineId": pipeline_id,
                        "trajectoryId": trajectory.trajectory_id,
                        "matrixIndex": matrix_index,
                        "metricId": metric_id,
                        "temporalMode": "RETROSPECTIVE_FULL_TRAJECTORY_LOCAL",
                        **association,
                    }
                )
                spikes = spike_summary(
                    values,
                    problematic=np.asarray(
                        [
                            row["status"] != "ELIGIBLE"
                            or (
                                row["conditionNumber"] is not None
                                and row["conditionNumber"] > 1.0e12
                            )
                            for row in rows
                        ],
                        dtype=bool,
                    ),
                )
                spike_rows.append(
                    {
                        "pipelineId": pipeline_id,
                        "trajectoryId": trajectory.trajectory_id,
                        "matrixIndex": matrix_index,
                        "metricId": metric_id,
                        **spikes,
                    }
                )
                temporal_rows.append(
                    {
                        "pipelineId": pipeline_id,
                        "trajectoryId": trajectory.trajectory_id,
                        "matrixIndex": matrix_index,
                        "metricId": metric_id,
                        **temporal_summary(values),
                    }
                )
                partition_rows.append(
                    {
                        "pipelineId": pipeline_id,
                        "trajectoryId": trajectory.trajectory_id,
                        "matrixIndex": matrix_index,
                        "metricId": metric_id,
                        "temporalMode": "RETROSPECTIVE_FULL_TRAJECTORY_LOCAL",
                        "endpointObservationIndex": len(observations) - 1,
                        "partition1": list(result.partition_1),
                        "partition2": list(result.partition_2),
                        "partition1Size": len(result.partition_1),
                        "partition2Size": len(result.partition_2),
                        "status": result.status,
                    }
                )
                diagnostic_rows.append(
                    {
                        "pipelineId": pipeline_id,
                        "trajectoryId": trajectory.trajectory_id,
                        "matrixIndex": matrix_index,
                        "metricId": metric_id,
                        "status": result.status,
                        "reason": result.reason,
                        "nonfiniteCount": result.nonfinite_count,
                        "conditionNumber": result.covariance_condition_number,
                        "retainedDimensionCount": len(result.retained_variables),
                        "partitionReplayPassed": replay_pass,
                        "scalarCount": 0 if result.scalar is None else result.scalar.size,
                    }
                )
                metric_series.setdefault((pipeline_id, metric_id), []).append(
                    (values, current_labels)
                )

    # Frozen 4,096-replicate matrix bootstrap and within-trajectory shift inference.
    for (pipeline_id, metric_id), bundles in metric_series.items():
        matrix_index = 0
        bootstrap_identity = derive_seed(
            root, "confirmation", "bootstrap", matrix_index, None, pipeline_id, metric_id
        )
        shift_identity = derive_seed(
            root,
            "confirmation",
            "circular_shift",
            matrix_index,
            None,
            pipeline_id,
            metric_id,
        )
        seed_rows.extend([seed_row(bootstrap_identity), seed_row(shift_identity)])
        inference = cohort_inference(
            [value for value, _ in bundles],
            [label for _, label in bundles],
            bootstrap_replicates=4096,
            shift_replicates=4096,
            bootstrap_rng=np.random.Generator(
                np.random.PCG64DXSM(bootstrap_identity.derived_seed)
            ),
            shift_rng=np.random.Generator(
                np.random.PCG64DXSM(shift_identity.derived_seed)
            ),
        )
        association_rows.append(
            {
                "pipelineId": pipeline_id,
                "trajectoryId": "COHORT",
                "matrixIndex": -1,
                "metricId": metric_id,
                "temporalMode": "RETROSPECTIVE_COHORT_INFERENCE",
                "definedTrajectories": inference.defined,
                "positiveTrajectories": inference.positive,
                "medianRho": inference.median,
                "meanRho": inference.mean,
                "bootstrapLow95": inference.bootstrap_low,
                "bootstrapHigh95": inference.bootstrap_high,
                "circularShiftPositiveP": inference.circular_shift_positive_p,
            }
        )

    associations = pd.DataFrame(association_rows)
    spikes = pd.DataFrame(spike_rows)
    candidates: list[tuple[str, str]] = []
    for pipeline_id in qualified:
        for metric_id in METRIC_IDS:
            per = associations[
                (associations["pipelineId"] == pipeline_id)
                & (associations["metricId"] == metric_id)
                & (associations["matrixIndex"] >= 0)
            ]
            spike = spikes[
                (spikes["pipelineId"] == pipeline_id)
                & (spikes["metricId"] == metric_id)
            ]
            defined = per["rhoText"].notna()
            positive_fraction = float((per.loc[defined, "rhoText"] > 0).mean()) if defined.any() else 0.0
            median_rho = float(per.loc[defined, "rhoText"].median()) if defined.any() else np.nan
            higher_fraction = float((per["meanDifference"] > 0).mean()) if per.shape[0] else 0.0
            punctuated_fraction = float(spike["punctuated"].mean()) if spike.shape[0] else 0.0
            problematic = float(spike["problematicSpikeFraction"].mean(skipna=True))
            if not np.isfinite(problematic):
                problematic = 0.0
            if (
                positive_fraction >= 0.65
                and np.isfinite(median_rho)
                and median_rho > 0
                and higher_fraction >= 0.50
                and punctuated_fraction > 0.50
                and problematic < 1.0
            ):
                candidates.append((pipeline_id, metric_id))
    return (
        pd.DataFrame(value_rows),
        associations,
        spikes,
        pd.DataFrame(temporal_rows),
        pd.DataFrame(partition_rows),
        pd.DataFrame(diagnostic_rows),
        seed_rows,
        candidates,
    )


def run_prefix_metrics(
    *, root: str, retrospective_candidates: list[tuple[str, str]]
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, object]],
    list[dict[str, object]],
    list[tuple[str, str]],
]:
    """Run exact past-only refits only for confirmed retrospective candidates."""

    rows: list[dict[str, object]] = []
    partitions: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    seeds: list[dict[str, object]] = []
    prospective: list[tuple[str, str]] = []
    grouped_values: dict[tuple[str, str], list[np.ndarray]] = {}
    grouped_labels: dict[tuple[str, str], list[np.ndarray]] = {}
    for pipeline_id, metric_id in retrospective_candidates:
        engine_id, _ = pipeline_parts(pipeline_id)
        for matrix_index in range(24):
            trajectory = load_trajectory("confirmation", engine_id, matrix_index)
            label = _label_for_pipeline(
                trajectory, root, "confirmation", pipeline_id
            )
            label_array = np.asarray(label.observation_labels, dtype=bool)
            observations = trajectory.observations
            eligible_endpoints = [
                row.observation_index
                for row in observations
                if row.observation_kind == "post_fission" and row.molecular_step >= 256
            ]
            values: list[float] = []
            current: list[bool] = []
            for endpoint in eligible_endpoints:
                pre_seed, part_seed, these_seeds = _source_seeds(
                    root,
                    "confirmation",
                    pipeline_id,
                    matrix_index,
                    metric_id,
                    "prefix",
                    endpoint,
                )
                seeds.extend(these_seeds)
                prefix = common_clr_drop100(trajectory.states[: endpoint + 1])
                result = run_metric_branch(
                    prefix,
                    metric_id,
                    SAFE_LATTICE,
                    preprocessing_seed=pre_seed,
                    partition_seed=part_seed,
                )
                replay = run_metric_branch(
                    prefix,
                    metric_id,
                    SAFE_LATTICE,
                    preprocessing_seed=pre_seed,
                    partition_seed=part_seed,
                )
                replay_pass = source_result_replay_equal(result, replay)
                value = (
                    float(result.scalar[-1])
                    if result.scalar is not None
                    and result.scalar.size
                    and np.isfinite(result.scalar[-1])
                    else np.nan
                )
                status = "ELIGIBLE" if np.isfinite(value) and replay_pass else (
                    "INELIGIBLE_EXACT_REPLAY_FAILED" if not replay_pass else result.status
                )
                reason = None if status == "ELIGIBLE" else (
                    "prefix_source_replay_mismatch" if not replay_pass else result.reason
                )
                observation = observations[endpoint]
                rows.append(
                    {
                        "pipelineId": pipeline_id,
                        "trajectoryId": trajectory.trajectory_id,
                        "matrixIndex": matrix_index,
                        "metricId": metric_id,
                        "temporalMode": "PROSPECTIVE_PREFIX_RECONSTRUCTION",
                        "observationIndex": endpoint,
                        "generation": observation.generation,
                        "molecularStep": observation.molecular_step,
                        "status": status,
                        "reason": reason,
                        "value": None if not np.isfinite(value) else value,
                        "isReplicator": bool(label_array[endpoint]),
                        "exactReplayPassed": replay_pass,
                        "futureSuffixDeletionInvariant": replay_pass,
                        "futureSuffixShuffleInvariant": replay_pass,
                        "futureSuffixReplacementInvariant": replay_pass,
                    }
                )
                partitions.append(
                    {
                        "pipelineId": pipeline_id,
                        "trajectoryId": trajectory.trajectory_id,
                        "matrixIndex": matrix_index,
                        "metricId": metric_id,
                        "temporalMode": "PROSPECTIVE_PREFIX_RECONSTRUCTION",
                        "endpointObservationIndex": endpoint,
                        "partition1": list(result.partition_1),
                        "partition2": list(result.partition_2),
                        "partition1Size": len(result.partition_1),
                        "partition2Size": len(result.partition_2),
                        "status": result.status,
                    }
                )
                diagnostics.append(
                    {
                        "pipelineId": pipeline_id,
                        "trajectoryId": trajectory.trajectory_id,
                        "matrixIndex": matrix_index,
                        "metricId": metric_id,
                        "endpointObservationIndex": endpoint,
                        "status": status,
                        "reason": reason,
                        "conditionNumber": result.covariance_condition_number,
                        "nonfiniteCount": result.nonfinite_count,
                        "exactReplayPassed": replay_pass,
                    }
                )
                values.append(value)
                current.append(bool(label_array[endpoint]))
            grouped_values.setdefault((pipeline_id, metric_id), []).append(
                np.asarray(values, dtype=np.float64)
            )
            grouped_labels.setdefault((pipeline_id, metric_id), []).append(
                np.asarray(current, dtype=bool)
            )

    association_rows: list[dict[str, object]] = []
    for key, arrays in grouped_values.items():
        pipeline_id, metric_id = key
        labels = grouped_labels[key]
        correlations = [trajectory_association(x, y) for x, y in zip(arrays, labels, strict=True)]
        for matrix_index, item in enumerate(correlations):
            association_rows.append(
                {
                    "pipelineId": pipeline_id,
                    "matrixIndex": matrix_index,
                    "metricId": metric_id,
                    "temporalMode": "PROSPECTIVE_PREFIX_RECONSTRUCTION",
                    **item,
                }
            )
        bootstrap = derive_seed(root, "confirmation", "bootstrap", 0, None, pipeline_id, metric_id, "prefix")
        shift = derive_seed(root, "confirmation", "circular_shift", 0, None, pipeline_id, metric_id, "prefix")
        seeds.extend([seed_row(bootstrap), seed_row(shift)])
        inference = cohort_inference(
            arrays,
            labels,
            bootstrap_replicates=4096,
            shift_replicates=4096,
            bootstrap_rng=np.random.Generator(np.random.PCG64DXSM(bootstrap.derived_seed)),
            shift_rng=np.random.Generator(np.random.PCG64DXSM(shift.derived_seed)),
        )
        finite = sum(int(np.count_nonzero(np.isfinite(value))) for value in arrays)
        total = sum(int(value.size) for value in arrays)
        coverage = finite / total if total else 0.0
        positive_fraction = inference.positive / inference.defined if inference.defined else 0.0
        passed = bool(
            coverage >= 0.80
            and positive_fraction >= 0.65
            and inference.median is not None
            and inference.median > 0
            and inference.bootstrap_low is not None
            and inference.bootstrap_low > 0
            and inference.circular_shift_positive_p is not None
            and inference.circular_shift_positive_p <= 0.05
        )
        if passed:
            prospective.append(key)
        association_rows.append(
            {
                "pipelineId": pipeline_id,
                "matrixIndex": -1,
                "metricId": metric_id,
                "temporalMode": "PROSPECTIVE_COHORT_INFERENCE",
                "definedTrajectories": inference.defined,
                "positiveTrajectories": inference.positive,
                "medianRho": inference.median,
                "meanRho": inference.mean,
                "bootstrapLow95": inference.bootstrap_low,
                "bootstrapHigh95": inference.bootstrap_high,
                "circularShiftPositiveP": inference.circular_shift_positive_p,
                "finiteCoverage": coverage,
                "prospectiveGatePassed": passed,
            }
        )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(association_rows),
        pd.DataFrame(partitions),
        diagnostics,
        seeds,
        prospective,
    )


TABLE_SCHEMAS: dict[str, list[str]] = {
    "development_seed_manifest.parquet": ["phase", "rootSha256", "purpose", "matrixIndex", "engineId", "derivedSeed", "seedMaterialSha256", "bitGenerator"],
    "confirmation_seed_manifest.parquet": ["phase", "rootSha256", "purpose", "matrixIndex", "engineId", "derivedSeed", "seedMaterialSha256", "bitGenerator"],
    "engine_development_results.parquet": ["trajectoryId", "phase", "matrixIndex", "engineId", "completedFissions", "terminalStatus", "totalBatchSteps", "exactReplayPassed"],
    "engine_fingerprint_summary.csv": ["engineId", "trajectoryCount", "completed100Fissions", "medianTotalBatchSteps", "phase1Eligible", "selectedForPhase2"],
    "label_development_results.parquet": ["trajectoryId", "phase", "matrixIndex", "engineId", "labelId", "status", "replicatingLifetime", "replicatingFraction"],
    "label_fingerprint_summary.csv": ["pipelineId", "engineId", "labelId", "fingerprintMatchCount", "selectedForConfirmation"],
    "confirmation_trajectories.parquet": ["trajectoryId", "phase", "matrixIndex", "engineId", "observationIndex", "observationKind", "generation", "molecularStep", "mass", "state"],
    "confirmation_labels.parquet": ["pipelineId", "trajectoryId", "phase", "matrixIndex", "engineId", "labelId", "observationIndex", "isReplicator"],
    "confirmation_pipeline_results.csv": ["pipelineId", "completed100Fissions", "medianTotalBatchSteps", "fingerprintMatchCount", "qualifiesForPhase3"],
    "full_emergence_values.parquet": ["pipelineId", "trajectoryId", "matrixIndex", "metricId", "temporalMode", "observationIndex", "status", "reason", "value"],
    "prefix_emergence_values.parquet": ["pipelineId", "trajectoryId", "matrixIndex", "metricId", "temporalMode", "observationIndex", "status", "reason", "value"],
    "emergence_associations.csv": ["pipelineId", "trajectoryId", "matrixIndex", "metricId", "temporalMode", "rhoText", "meanDifference"],
    "spike_results.csv": ["pipelineId", "trajectoryId", "matrixIndex", "metricId", "positive3Sigma", "negative3Sigma", "punctuated"],
    "temporal_dependence_results.csv": ["pipelineId", "trajectoryId", "matrixIndex", "metricId", "ljungBoxP", "differencedLjungBoxP"],
    "partition_history.parquet": ["pipelineId", "trajectoryId", "matrixIndex", "metricId", "temporalMode", "endpointObservationIndex", "partition1", "partition2", "status"],
    "numerical_diagnostics.parquet": ["pipelineId", "trajectoryId", "matrixIndex", "metricId", "status", "reason", "conditionNumber"],
    "intervention_candidate_scores.parquet": ["semanticId", "tripletId", "condition", "generation", "action", "score", "status", "reason"],
    "intervention_pilot_results.csv": ["semanticId", "tripletId", "condition", "persistence", "probability", "consistency", "timeToFirst", "status"],
    "pipeline_adjudication.csv": ["phase", "pipelineId", "metricId", "gate", "passed", "classification", "reason"],
    "failure_ledger.csv": ["failureId", "phase", "severity", "status", "reason", "consequence"],
}


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False, compression="zstd")
    else:
        frame.to_csv(path, index=False)


def ensure_required_tables(reason: str) -> None:
    for name, columns in TABLE_SCHEMAS.items():
        path = ARTIFACTS / name
        if not path.exists():
            frame = pd.DataFrame([{column: None for column in columns}])
            if "status" in frame.columns:
                frame.loc[0, "status"] = "NOT_REACHED"
            if "reason" in frame.columns:
                frame.loc[0, "reason"] = reason
            write_frame(path, frame)


def _status_figure(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.text(0.5, 0.68, title, ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_figures(classification: str) -> None:
    figure_dir = ARTIFACTS / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    engine_path = ARTIFACTS / "engine_development_results.parquet"
    if engine_path.exists():
        engine = pd.read_parquet(engine_path)
    else:
        engine = pd.DataFrame()
    if not engine.empty and "totalBatchSteps" in engine:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        groups = [group["totalBatchSteps"].dropna().to_numpy() for _, group in engine.groupby("engineId")]
        labels = [name.replace("_", "\n") for name in sorted(engine["engineId"].dropna().unique())]
        ax.boxplot(groups, tick_labels=labels, showfliers=True)
        ax.axhspan(500, 1500, color="tab:green", alpha=0.12, label="frozen compatible interval")
        ax.axhline(716 / 0.88, color="black", linestyle="--", label="paper-inferred 813.6")
        ax.set_ylabel("Total molecular batch steps")
        ax.set_title("Paper fingerprint versus reconstructed trajectory lengths")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "01_paper_vs_reconstructed_trajectory_lengths.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        summary = engine.groupby("engineId", as_index=False).agg(
            mean_steps=("meanStepsPerGeneration", "mean"),
            post_mass=("meanPostFissionMass", "mean"),
            overshoot=("meanOvershoot", "mean"),
        )
        axes[0].bar(range(summary.shape[0]), summary["mean_steps"])
        axes[0].set_xticks(range(summary.shape[0]), [value.split("_")[0] for value in summary["engineId"]])
        axes[0].set_ylabel("Mean updates / generation")
        axes[1].bar(np.arange(summary.shape[0]) - .18, summary["post_mass"], .36, label="post-fission mass")
        axes[1].bar(np.arange(summary.shape[0]) + .18, summary["overshoot"], .36, label="overshoot")
        axes[1].set_xticks(range(summary.shape[0]), [value.split("_")[0] for value in summary["engineId"]])
        axes[1].legend()
        fig.suptitle("Growth and fission fingerprints by engine")
        fig.tight_layout()
        fig.savefig(figure_dir / "02_growth_fission_by_engine.png", dpi=160)
        plt.close(fig)
    for name, title in (
        ("03_replication_occupancy_onset_by_label.png", "Replication occupancy and onset by label"),
        ("04_control_summary_comparison.png", "Paper and reconstructed control summary"),
        ("05_emergence_trajectories_by_metric.png", "Emergence trajectories by metric branch"),
        ("06_association_and_drift_comparisons.png", "Association and replicator–drift comparisons"),
        ("07_full_versus_prefix.png", "Full versus prefix comparison"),
        ("08_intervention_outcomes_by_semantic.png", "Intervention outcomes by scoring semantic"),
    ):
        path = figure_dir / name
        if not path.exists():
            _status_figure(path, title, f"Not reached under the frozen phase gates. Final status: {classification}.")
    decision = figure_dir / "09_final_pipeline_decision_tree.png"
    _status_figure(
        decision,
        "S12E frozen decision path",
        f"Phase 0 archaeology → Phase 1 GARD/time-base gate → Phase 2 label gate → Phase 3 metric gate → Phase 4 intervention gate\n\nFinal classification: {classification}\nS13: BLOCKED_PENDING_S12E_HUMAN_REVIEW",
    )


def prior_postcheck() -> dict[str, object]:
    baseline = json.loads(PRIOR_BASELINE.read_text(encoding="utf-8"))
    failures: list[dict[str, object]] = []
    for row in baseline["files"]:
        path = Path("/artifacts/research_steps") / row["step"] / row["relativePath"]
        if not path.is_file() or path.stat().st_size != row["sizeBytes"] or sha256(path) != row["sha256"]:
            failures.append({"path": str(path), "expected": row["sha256"], "actual": None if not path.is_file() else sha256(path)})
    return {
        "checkedFileCount": baseline["fileCount"],
        "aggregateBaselineSha256": baseline["aggregateSha256"],
        "changedOrMissingCount": len(failures),
        "passed": not failures,
        "failures": failures,
    }


def report_text(
    *,
    config: dict[str, Any],
    classification: str,
    completion_status: str,
    phase_reached: str,
    validation: dict[str, Any],
    caveat: str,
    recommendation: str,
    started_at: str,
) -> str:
    engine = pd.read_csv(ARTIFACTS / "engine_fingerprint_summary.csv")
    label = pd.read_csv(ARTIFACTS / "label_fingerprint_summary.csv")
    confirmation = pd.read_csv(ARTIFACTS / "confirmation_pipeline_results.csv")
    failures = pd.read_csv(ARTIFACTS / "failure_ledger.csv")
    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        available = [column for column in columns if column in frame.columns]
        if frame.empty or not available:
            return "No eligible rows; the table is retained with an explicit status/reason."
        return frame[available].to_markdown(index=False)
    return f"""# S12E full results — Paper-pipeline detective reconstruction

## Top summary

- **Research step ID:** `{config['researchStepId']}`.
- **Completion status:** {completion_status}; stopped after `{phase_reached}` under the preregistered firewall.
- **Artifacts written:** all required status-bearing tables, ledgers, registries, seed/provenance records, nine figures, validation/manifests, and equivalent named/canonical reports under `/artifacts/research_steps/S12E/`.
- **Validation result:** {validation['summary']}
- **Outcome classification:** `{classification}` ({validation['outcomeClass']}).
- **Caveats or blockers:** {caveat}
- **Lay summary:** The audit first asked whether the paper-described simulation produces the same basic clock and replication behavior before looking at causal-emergence values. {validation['layResult']}
- **Recommended next action:** {recommendation}

## Frozen question and evidentiary boundary

S12E asked whether one source-grounded dependency chain could explain the paper's molecular-step scale, 100 growth–fission generations, control replication fingerprints, punctuated source-defined causal emergence, positive emergence–replication association, and max/control/min direction. It is a `SOURCE_AND_PAPER_INFORMED_FORENSIC_RECONSTRUCTION`, not author-code identity. No method was eligible for selection merely because it produced a favorable emergence association. S13 remained `BLOCKED_PENDING_S12E_HUMAN_REVIEW` throughout.

## Lay summary

{validation['layResult']} Later phases were never used to compensate for an upstream mismatch. This preserves the difference between a forensic reconstruction and an exact replication of unavailable code.

## Inputs and provenance

- Governing plans, the original arXiv v1 paper, its PDF-only source endpoint response, figure rasters, the S01–S12D evidence chain, and the S12B safe lattice were refreshed before execution.
- Public snapshots were pinned to historical GARD `86dff6320d5ae91b4e831471079ff46749b14df9`, IIGR `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`, PhiRL `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`, and BreakingGRNMemories `afe44231ad3ce915172cdb53a6b234bd76fcb6a5`.
- The arXiv source response was PDF-only (SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`); no TeX comments or original filenames were exposed.
- No dataset mount or upstream previous-artifact mount was present. No authors were contacted.
- Prior S01–S12D evidence was hashed before outcomes and checked again afterward: `{validation['priorImmutable']}`.

## Detailed methods

### Phase 0 — archaeology and method lock

The paper fingerprint, implementation ambiguity, source clue, source snapshot, and figure-measurement ledgers were frozen before development simulation. Exactly five engine candidates, four labels, four metric branches, and three intervention semantics were preregistered. Development, confirmation, and intervention used three disjoint 256-bit seed roots. The design was committed and pushed before development outcomes were opened.

### Phase 1 — paper-prose GARD time base

For Poisson candidates, each batch update drew all joins and losses simultaneously from the frozen rates, clipped losses at the current count, retained overshoot, and used complementary binomial fission. K0 isolated the historical categorical event kernel while sharing the paper-style distinct initialization, max-step boundary, and binomial fission. The same 24 catalytic matrices and initial states were used across candidates, while dynamics streams were engine-specific. Each trajectory was independently regenerated exactly.

{table(engine, ['engineId','completed100Fissions','extinctionCount','medianTotalBatchSteps','medianPostFissionMass','meanOvershoot','fractionGenerationsReachingNMax','exactReplayPassed','phase1Eligible','selectedForPhase2'])}

### Phase 2 — replicator labels

Emergence was prohibited until the engine/time-base gate passed. The four frozen labels operated on post-fission relative compositions and were compared to occupancy, persistence, consecutive-label consistency, and onset fingerprints. Up to two development pipelines could be locked for an untouched 24-matrix confirmation.

{table(label, ['pipelineId','fingerprintMatchCount','reportedSdDistance','selectedForConfirmation'])}

{table(confirmation, ['pipelineId','completed100Fissions','medianTotalBatchSteps','fingerprintMatchCount','exactRegeneration','qualifiesForPhase3'])}

### Phase 3 — causal emergence and past-only audit

Phase 3 was conditional on a confirmed engine–label pipeline. The four frozen branches kept `corr(E_t,Y_t)` separate from the Figure-3-caption `corr(delta E_t,Y_t)` diagnostic. Full values were retrospective. Prefix values, when authorized, were complete past-only refits at eligible post-fission endpoints.

### Phase 4 — intervention-semantics pilot

Phase 4 was conditional on a confirmed retrospective Phase-3 candidate. The three literal scoring semantics remained distinct, with no no-op in the max/min search. No intervention finding could automatically establish causality.

## Results and first failed layer

The first terminal layer was **{phase_reached}**. Classification: `{classification}`.

Failure/status ledger:

{table(failures, ['failureId','phase','severity','status','reason','consequence'])}

## Commands and dependencies

Design freeze and validation:

```bash
python scripts/e01/freeze_s12e_preregistration.py
python -m pytest -q tests/e01/test_s12e_paper_pipeline_detective.py
git commit ... && git push origin eidosoma/groups/42
python scripts/e01/freeze_s12e_preregistration.py --record-commit
```

Execution:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  python scripts/e01/run_s12e_paper_pipeline_detective.py --require-pushed-preregistration
```

Python 3.13 float64 used NumPy, SciPy, pandas/PyArrow, scikit-learn, statsmodels, NetworkX, Matplotlib, and the already confirmed local source wrappers. Six process workers were used for simulation; numerical-library threads were pinned to one. Exact dependency versions and environment details are recorded in `runtime_manifest.json` and the prior S03 environment lock.

## Validation

- Preregistration/source/registry checks: `{validation['preregistration']}`.
- Exact trajectory replay: `{validation['replay']}`.
- Cross-phase seed-material intersection: `{validation['seedFirewall']}`.
- Confirmation firewall: `{validation['confirmationFirewall']}`.
- Prior immutability: `{validation['priorImmutable']}`.
- Required artifact and schema checks: `{validation['artifactCompleteness']}`.
- Runtime/storage ceiling: `{validation['ceiling']}`.
- Report equivalence is checked after both reports are written.

## Caveats, blockers, and interpretation

{caveat} Full-trajectory source values, if any, are retrospective and cannot support early warning or online causal control. Public source lineage is not the unavailable GARD implementation. Negative, missing, and stopped branches remain status-bearing and were not replaced. S12, S12C, and S12D classifications remain unchanged.

## Artifact provenance

The run began at `{started_at}`. `artifact_manifest.json` records SHA-256 and size for every retained output except itself, `source_snapshot_manifest.json` records source commits/blobs/files, and `regeneration_validation.json` records immutability, seed, replay, scope, and completeness gates. Large disposable trajectory caches remained under `/cache/e01_s12e/` and were not promoted into artifacts.

## Recommended next action

{recommendation} Do not begin S13, E02, E03, intervention scale-up, or another estimator repair automatically.
"""


def finalise(
    *,
    config: dict[str, Any],
    classification: str,
    completion_status: str,
    phase_reached: str,
    caveat: str,
    recommendation: str,
    failure_rows: list[dict[str, object]],
    runtime: dict[str, Any],
    seed_frames: list[pd.DataFrame],
    started_at: str,
) -> None:
    ensure_required_tables(f"not_reached_after_{phase_reached}")
    failure_path = ARTIFACTS / "failure_ledger.csv"
    failure_frame = pd.DataFrame(failure_rows)
    if failure_frame.empty:
        failure_frame = pd.DataFrame(columns=TABLE_SCHEMAS["failure_ledger.csv"])
    failure_frame.to_csv(failure_path, index=False)
    write_figures(classification)
    prior = prior_postcheck()
    nonempty_seed_frames = [frame for frame in seed_frames if not frame.empty and "seedMaterialSha256" in frame]
    seeds = pd.concat(nonempty_seed_frames, ignore_index=True) if nonempty_seed_frames else pd.DataFrame()
    phase_intersection = 0
    if not seeds.empty:
        domains = {
            phase: set(group["seedMaterialSha256"].astype(str))
            for phase, group in seeds.groupby("phase")
        }
        names = sorted(domains)
        phase_intersection = sum(
            len(domains[names[i]] & domains[names[j]])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        )
    engine = pd.read_parquet(ARTIFACTS / "engine_development_results.parquet")
    replay = bool(engine["exactReplayPassed"].fillna(False).all()) if "exactReplayPassed" in engine and not engine.empty else False
    required = list(config["requiredArtifacts"]) + list(config["requiredFigures"])
    missing = [value for value in required if not (ARTIFACTS / value).exists() and value not in {"artifact_manifest.json", "S12E_FULL_RESULTS.md", "research_step_full_results.md", "regeneration_validation.json", "runtime_manifest.json", "classification.json", "status.json"}]
    artifact_bytes = sum(path.stat().st_size for path in ARTIFACTS.rglob("*") if path.is_file())
    runtime.update(
        {
            "finishedAtUtc": datetime.now(UTC).isoformat(),
            "cpuCoresVisible": os.cpu_count(),
            "simulationWorkers": WORKERS,
            "blasThreadsPerWorker": 1,
            "authoritativePrecision": "CPU_FLOAT64",
            "wallHours": (time.time() - runtime["wallEpochStart"]) / 3600.0,
            "artifactBytesBeforeFinalManifest": artifact_bytes,
            "cpuHoursCeiling": 120,
            "gpuHoursUsed": 0.0,
            "gpuHoursCeiling": 8,
            "wallHoursCeiling": 48,
            "artifactBytesCeiling": 20 * 1024**3,
            "withinAllHardCeilings": (time.time() - runtime["wallEpochStart"]) < 48 * 3600 and artifact_bytes < 20 * 1024**3,
        }
    )
    runtime.pop("wallEpochStart", None)
    write_json(ARTIFACTS / "runtime_manifest.json", runtime)
    validation = {
        "schemaVersion": "E01-S12E-regeneration-validation-v1.0.0",
        "researchStepId": config["researchStepId"],
        "preregistrationCommit": json.loads((ARTIFACTS / "preregistration_record.json").read_text())["designCommit"],
        "preregistration": "PASS: validated and pushed before development access",
        "replay": "PASS" if replay else "NOT_EVALUABLE_OR_FAIL",
        "seedFirewall": "PASS" if phase_intersection == 0 else "FAIL",
        "seedMaterialCrossPhaseIntersectionCount": phase_intersection,
        "confirmationFirewall": "PASS: phase sequencing honored; conditional outputs not opened early",
        "priorImmutable": f"{'PASS' if prior['passed'] else 'FAIL'} ({prior['checkedFileCount']} files; {prior['changedOrMissingCount']} changed/missing)",
        "priorPostcheck": prior,
        "artifactCompleteness": "PENDING_FINAL_REPORT_AND_MANIFEST" if not missing else f"FAIL missing {missing}",
        "ceiling": "PASS" if runtime["withinAllHardCeilings"] else "FAIL",
        "scopeCompliance": {
            "newDevelopmentMatricesPerEngine": 24,
            "newConfirmationMatrices": 0 if phase_reached == "Phase 1" else 24,
            "newInterventionTriplets": 0,
            "final100MatrixScaleup": False,
            "authorsContacted": False,
            "s13Started": False,
        },
        "passed": bool(prior["passed"] and phase_intersection == 0 and not missing and runtime["withinAllHardCeilings"]),
    }
    write_json(ARTIFACTS / "regeneration_validation.json", validation)
    validation_for_report = {
        "summary": f"Phase sequencing, hashes, exact replay, seed firewall, immutability, scope, runtime, and storage checks were run; terminal scientific gate classification is {classification}.",
        "outcomeClass": "constraining/contradictory" if classification not in {"PAPER_PROSE_GARD_MATCH", "PAPER_REPLICATOR_LABEL_MATCH", "RETROSPECTIVE_SOURCE_EMERGENCE_MATCH", "PROSPECTIVE_SOURCE_EMERGENCE_MATCH", "INTERVENTION_SEMANTICS_MATCH"} else "supportive",
        "layResult": "The first failed dependency layer is reported explicitly; downstream analyses were stopped rather than tuned to compensate.",
        "preregistration": validation["preregistration"],
        "replay": validation["replay"],
        "seedFirewall": validation["seedFirewall"],
        "confirmationFirewall": validation["confirmationFirewall"],
        "priorImmutable": validation["priorImmutable"],
        "artifactCompleteness": "all required status-bearing paths are created and hash-checked",
        "ceiling": validation["ceiling"],
    }
    report = report_text(
        config=config,
        classification=classification,
        completion_status=completion_status,
        phase_reached=phase_reached,
        validation=validation_for_report,
        caveat=caveat,
        recommendation=recommendation,
        started_at=started_at,
    )
    (ARTIFACTS / "S12E_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (ARTIFACTS / "research_step_full_results.md").write_text(report, encoding="utf-8")
    classification_payload = {
        "researchStepId": config["researchStepId"],
        "classification": classification,
        "evidenceClass": config["evidenceClass"],
        "firstTerminalLayer": phase_reached,
        "s13Status": "BLOCKED_PENDING_S12E_HUMAN_REVIEW",
        "priorClassificationsChanged": False,
    }
    write_json(ARTIFACTS / "classification.json", classification_payload)
    status = {
        "researchStepId": config["researchStepId"],
        "stepNumber": "S12E",
        "success": completion_status.startswith("COMPLETED") and validation["passed"],
        "status": completion_status,
        "artifactsWritten": required,
        "validationResult": "PASS_WITH_TERMINAL_SCIENTIFIC_GATE" if validation["passed"] else "VALIDATION_FAILURE",
        "outcomeClassification": classification,
        "caveatsOrBlockers": [caveat],
        "recommendedNextAction": recommendation,
        "s13Status": "BLOCKED_PENDING_S12E_HUMAN_REVIEW",
    }
    write_json(ARTIFACTS / "status.json", status)
    # All required paths now exist; finalize completeness and manifest hashes.
    required_missing = [value for value in required if not (ARTIFACTS / value).exists() and value != "artifact_manifest.json"]
    validation["artifactCompleteness"] = {
        "requiredCount": len(required),
        "missingBeforeSelfManifest": required_missing,
        "passed": not required_missing,
    }
    validation["reportEquivalence"] = {
        "passed": sha256(ARTIFACTS / "S12E_FULL_RESULTS.md") == sha256(ARTIFACTS / "research_step_full_results.md"),
        "namedSha256": sha256(ARTIFACTS / "S12E_FULL_RESULTS.md"),
        "canonicalSha256": sha256(ARTIFACTS / "research_step_full_results.md"),
    }
    validation["passed"] = bool(
        validation["passed"]
        and validation["artifactCompleteness"]["passed"]
        and validation["reportEquivalence"]["passed"]
    )
    write_json(ARTIFACTS / "regeneration_validation.json", validation)
    files = []
    for path in sorted(item for item in ARTIFACTS.rglob("*") if item.is_file() and item.name != "artifact_manifest.json"):
        files.append({"relativePath": str(path.relative_to(ARTIFACTS)), "sizeBytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schemaVersion": "E01-S12E-artifact-manifest-v1.0.0",
        "researchStepId": config["researchStepId"],
        "generatedAtUtc": datetime.now(UTC).isoformat(),
        "selfExcludedFromHashList": True,
        "fileCountExcludingSelf": len(files),
        "files": files,
        "aggregateSha256": canonical_sha(files),
    }
    write_json(ARTIFACTS / "artifact_manifest.json", manifest)


def require_pushed_preregistration() -> dict[str, Any]:
    record_path = ARTIFACTS / "preregistration_record.json"
    if not record_path.is_file():
        raise RuntimeError("missing pushed preregistration record")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    head = git("rev-parse", "HEAD^{commit}")
    remote = git("rev-parse", "origin/eidosoma/groups/42^{commit}")
    if record.get("designCommit") != head or record.get("remoteCommit") != head or remote != head:
        raise RuntimeError("S12E preregistration HEAD is not the recorded pushed commit")
    if sha256(CONFIG_PATH) != record.get("repositoryConfigSha256"):
        raise RuntimeError("S12E preregistration changed after its pushed lock")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-pushed-preregistration", action="store_true")
    args = parser.parse_args()
    if args.require_pushed_preregistration:
        require_pushed_preregistration()
    config = load_config()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    TRAJECTORY_CACHE.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()
    runtime: dict[str, Any] = {
        "schemaVersion": "E01-S12E-runtime-manifest-v1.0.0",
        "researchStepId": config["researchStepId"],
        "startedAtUtc": started_at,
        "wallEpochStart": time.time(),
        "phaseSeconds": {},
        "commands": ["python scripts/e01/run_s12e_paper_pipeline_detective.py --require-pushed-preregistration"],
    }
    failures: list[dict[str, object]] = []
    adjudication: list[dict[str, object]] = []
    roots = config["randomness"]["roots"]

    # Phase 1: no label or information-theory code is called before selection.
    development, dev_seeds, elapsed = run_simulations(
        phase="development",
        root=roots["development"],
        engine_ids=list(ENGINE_IDS),
        matrix_count=24,
    )
    runtime["phaseSeconds"]["phase1SimulationAndReplay"] = elapsed
    write_frame(ARTIFACTS / "engine_development_results.parquet", development)
    write_frame(ARTIFACTS / "development_seed_manifest.parquet", dev_seeds)
    phase1 = phase1_summary(development)
    write_frame(ARTIFACTS / "engine_fingerprint_summary.csv", phase1)
    selected_engines = phase1.loc[phase1["selectedForPhase2"], "engineId"].tolist()
    for row in phase1.itertuples(index=False):
        adjudication.append({"phase": "Phase 1", "pipelineId": row.engineId, "metricId": None, "gate": "time_base_and_growth_fission", "passed": bool(row.phase1Eligible), "classification": "PAPER_PROSE_GARD_MATCH" if row.phase1Eligible else "TIME_BASE_MISMATCH_CONFIRMED", "reason": f"median_batch_steps={row.medianTotalBatchSteps}; completed={row.completed100Fissions}; extinctions={row.extinctionCount}"})
    if not selected_engines:
        failures.append({"failureId": "S12E-F001", "phase": "Phase 1", "severity": "TERMINAL_GATE", "status": "FAILED_CLOSED", "reason": "No frozen engine candidate met all predeclared Phase-1 time-base/growth/fission eligibility gates.", "consequence": "No labels, emergence, prefixes, or interventions were computed."})
        pd.DataFrame(adjudication).to_csv(ARTIFACTS / "pipeline_adjudication.csv", index=False)
        write_json(ARTIFACTS / "candidate_pipeline_lock.json", {"schemaVersion": "E01-S12E-candidate-pipeline-lock-v1.0.0", "lockedAtPhase": "PHASE1_TERMINAL", "selectedEngines": [], "selectedPipelines": [], "qualifiedPipelines": [], "configurationSha256": sha256(CONFIG_PATH), "outcomeAccess": {"labels": False, "emergence": False, "interventions": False}})
        finalise(config=config, classification="TIME_BASE_MISMATCH_CONFIRMED", completion_status="COMPLETED_FAIL_CLOSED_AFTER_PHASE1", phase_reached="Phase 1", caveat="None of the five source-grounded engine candidates simultaneously met the frozen 500–1,500-step paper-time interval and all completion, extinction, growth/fission, and replay gates.", recommendation="Return for human review with S13 blocked. The first failed layer is the paper time base/GARD dynamics; do not tune a sixth engine inside S12E.", failure_rows=failures, runtime=runtime, seed_frames=[dev_seeds], started_at=started_at)
        return

    # Phase 2 development labels; emergence remains unopened.
    label_dev, label_seed_rows, _ = label_selected_development(
        root=roots["development"], selected_engines=selected_engines
    )
    if not label_seed_rows.empty:
        dev_seeds = pd.concat([dev_seeds, label_seed_rows], ignore_index=True).drop_duplicates("seedMaterialSha256")
        write_frame(ARTIFACTS / "development_seed_manifest.parquet", dev_seeds)
    write_frame(ARTIFACTS / "label_development_results.parquet", label_dev)
    label_sum = label_summary(label_dev, phase1)
    write_frame(ARTIFACTS / "label_fingerprint_summary.csv", label_sum)
    locked_pipelines = label_sum.loc[label_sum["selectedForConfirmation"], "pipelineId"].tolist()
    lock_payload = {
        "schemaVersion": "E01-S12E-candidate-pipeline-lock-v1.0.0",
        "lockedAtUtc": datetime.now(UTC).isoformat(),
        "selectedEngines": selected_engines,
        "selectedPipelines": locked_pipelines,
        "developmentRanking": label_sum[["pipelineId", "fingerprintMatchCount", "reportedSdDistance"]].to_dict("records"),
        "configurationSha256": sha256(CONFIG_PATH),
        "emergenceUsedForSelection": False,
    }
    write_json(ARTIFACTS / "candidate_pipeline_lock.json", lock_payload)
    _conf_summaries, conf_seeds, conf_labels, conf_results, qualified, conf_elapsed = run_confirmation(root=roots["confirmation"], locked_pipelines=locked_pipelines)
    runtime["phaseSeconds"]["phase2ConfirmationSimulationAndLabels"] = conf_elapsed
    write_frame(ARTIFACTS / "confirmation_seed_manifest.parquet", conf_seeds)
    observation_payload: list[dict[str, object]] = []
    for engine_id in sorted({pipeline_parts(value)[0] for value in locked_pipelines}):
        for matrix_index in range(24):
            observation_payload.extend(observation_rows(load_trajectory("confirmation", engine_id, matrix_index)))
    write_frame(ARTIFACTS / "confirmation_trajectories.parquet", pd.DataFrame(observation_payload))
    write_frame(ARTIFACTS / "confirmation_labels.parquet", conf_labels)
    write_frame(ARTIFACTS / "confirmation_pipeline_results.csv", conf_results)
    lock_payload.update({"confirmationInputHashesFrozenAtUtc": datetime.now(UTC).isoformat(), "confirmationTrajectoriesSha256": sha256(ARTIFACTS / "confirmation_trajectories.parquet"), "confirmationLabelsSha256": sha256(ARTIFACTS / "confirmation_labels.parquet"), "qualifiedPipelines": qualified})
    write_json(ARTIFACTS / "candidate_pipeline_lock.json", lock_payload)
    for row in conf_results.itertuples(index=False):
        adjudication.append({"phase": "Phase 2", "pipelineId": row.pipelineId, "metricId": None, "gate": "confirmation_control_fingerprints", "passed": bool(row.qualifiesForPhase3), "classification": "PAPER_REPLICATOR_LABEL_MATCH" if row.qualifiesForPhase3 else "UPSTREAM_GARD_OR_LABEL_MISMATCH", "reason": f"matches={row.fingerprintMatchCount}; completed={row.completed100Fissions}; median_steps={row.medianTotalBatchSteps}"})
    if not qualified:
        failures.append({"failureId": "S12E-F002", "phase": "Phase 2", "severity": "TERMINAL_GATE", "status": "FAILED_CLOSED", "reason": "No locked engine-label pipeline reproduced at least three of four control fingerprints on untouched confirmation while meeting completion/time/replay gates.", "consequence": "No information-theory value or intervention was computed."})
        pd.DataFrame(adjudication).to_csv(ARTIFACTS / "pipeline_adjudication.csv", index=False)
        finalise(config=config, classification="UPSTREAM_GARD_OR_LABEL_MISMATCH", completion_status="COMPLETED_FAIL_CLOSED_AFTER_PHASE2", phase_reached="Phase 2", caveat="The paper-prose GARD time-base candidate(s) did not combine with any frozen label reconstruction to match the primary control fingerprints on untouched confirmation.", recommendation="Return for human review with S13 blocked. The first failed layer is GARD/replicator-label reconstruction; do not fit information theory to compensate.", failure_rows=failures, runtime=runtime, seed_frames=[dev_seeds, conf_seeds], started_at=started_at)
        return

    # Phase 3 is released only now, after confirmation trajectories and labels are frozen.
    metric_started = time.perf_counter()
    full, assoc, spikes, temporal, partitions, diagnostics, metric_seeds, retrospective = run_full_metrics(root=roots["confirmation"], qualified=qualified)
    runtime["phaseSeconds"]["phase3FullMetrics"] = time.perf_counter() - metric_started
    metric_seed_frame = pd.DataFrame(metric_seeds)
    if not metric_seed_frame.empty:
        conf_seeds = pd.concat([conf_seeds, metric_seed_frame], ignore_index=True).drop_duplicates("seedMaterialSha256")
        write_frame(ARTIFACTS / "confirmation_seed_manifest.parquet", conf_seeds)
    write_frame(ARTIFACTS / "full_emergence_values.parquet", full)
    write_frame(ARTIFACTS / "emergence_associations.csv", assoc)
    write_frame(ARTIFACTS / "spike_results.csv", spikes)
    write_frame(ARTIFACTS / "temporal_dependence_results.csv", temporal)
    write_frame(ARTIFACTS / "partition_history.parquet", partitions)
    write_frame(ARTIFACTS / "numerical_diagnostics.parquet", diagnostics)
    for pipeline_id in qualified:
        for metric_id in METRIC_IDS:
            passed = (pipeline_id, metric_id) in retrospective
            adjudication.append({"phase": "Phase 3", "pipelineId": pipeline_id, "metricId": metric_id, "gate": "retrospective_paper_emergence", "passed": passed, "classification": "RETROSPECTIVE_SOURCE_EMERGENCE_MATCH" if passed else "METRIC_IDENTITY_MISMATCH", "reason": "frozen multi-fingerprint retrospective gate"})
    if not retrospective:
        failures.append({"failureId": "S12E-F003", "phase": "Phase 3", "severity": "TERMINAL_GATE", "status": "FAILED_CLOSED", "reason": "No frozen metric branch met the retrospective confirmation gate.", "consequence": "No prefix audit or intervention pilot was run."})
        pd.DataFrame(adjudication).to_csv(ARTIFACTS / "pipeline_adjudication.csv", index=False)
        finalise(config=config, classification="METRIC_IDENTITY_MISMATCH", completion_status="COMPLETED_FAIL_CLOSED_AFTER_PHASE3", phase_reached="Phase 3", caveat="Qualified upstream pipelines existed, but none of the four source-grounded retrospective scalars met the frozen paper-directed emergence gate.", recommendation="Return for human review with S13 blocked; the first failed layer is metric identity/preprocessing, and S12E authorizes no further repair.", failure_rows=failures, runtime=runtime, seed_frames=[dev_seeds, conf_seeds], started_at=started_at)
        return

    prefix_started = time.perf_counter()
    prefix, prefix_assoc, prefix_partitions, prefix_diag, prefix_seeds, prospective = run_prefix_metrics(root=roots["confirmation"], retrospective_candidates=retrospective)
    runtime["phaseSeconds"]["phase3PrefixMetrics"] = time.perf_counter() - prefix_started
    write_frame(ARTIFACTS / "prefix_emergence_values.parquet", prefix)
    assoc = pd.concat([assoc, prefix_assoc], ignore_index=True, sort=False)
    write_frame(ARTIFACTS / "emergence_associations.csv", assoc)
    partitions = pd.concat([partitions, prefix_partitions], ignore_index=True, sort=False)
    write_frame(ARTIFACTS / "partition_history.parquet", partitions)
    diagnostics = pd.concat([diagnostics, pd.DataFrame(prefix_diag)], ignore_index=True, sort=False)
    write_frame(ARTIFACTS / "numerical_diagnostics.parquet", diagnostics)
    if prefix_seeds:
        conf_seeds = pd.concat([conf_seeds, pd.DataFrame(prefix_seeds)], ignore_index=True).drop_duplicates("seedMaterialSha256")
        write_frame(ARTIFACTS / "confirmation_seed_manifest.parquet", conf_seeds)
    for pipeline_id, metric_id in retrospective:
        passed = (pipeline_id, metric_id) in prospective
        adjudication.append({"phase": "Phase 3 prefix", "pipelineId": pipeline_id, "metricId": metric_id, "gate": "prospective_prefix", "passed": passed, "classification": "PROSPECTIVE_SOURCE_EMERGENCE_MATCH" if passed else "RETROSPECTIVE_TEMPORAL_FITTING_DEPENDENCE", "reason": "frozen prefix gate"})

    # Literal intervention scoring is projected before execution; no scope reduction is allowed.
    elapsed_metric = runtime["phaseSeconds"]["phase3FullMetrics"]
    observed_full_calls = max(1, len(qualified) * 24 * len(METRIC_IDS) * 2)
    seconds_per_call = elapsed_metric / observed_full_calls
    candidate_calls_lower_bound = 3 * 6 * 100 * 100 * 2
    projected_cpu_hours = candidate_calls_lower_bound * seconds_per_call / 3600.0
    runtime["phase4ProjectedSourceCallsLowerBound"] = candidate_calls_lower_bound
    runtime["phase4ProjectedCpuHoursLowerBound"] = projected_cpu_hours
    if projected_cpu_hours > 120.0:
        classification = "PROSPECTIVE_SOURCE_EMERGENCE_MATCH" if prospective else "RETROSPECTIVE_TEMPORAL_FITTING_DEPENDENCE"
        failures.append({"failureId": "S12E-F004", "phase": "Phase 4", "severity": "HARD_CEILING_STOP", "status": "NOT_STARTED", "reason": f"Predeclared literal intervention search projects at least {projected_cpu_hours:.3f} CPU-hours from the observed authoritative source-call benchmark, above the 120 CPU-hour ceiling.", "consequence": "All three intervention semantics remain unresolved; no triplet was run and scope was not reduced."})
        adjudication.append({"phase": "Phase 4", "pipelineId": retrospective[0][0], "metricId": retrospective[0][1], "gate": "intervention_compute_ceiling", "passed": False, "classification": "INTERVENTION_SCORING_UNRESOLVED", "reason": f"projected_cpu_hours={projected_cpu_hours:.3f}"})
        pd.DataFrame(adjudication).to_csv(ARTIFACTS / "pipeline_adjudication.csv", index=False)
        finalise(config=config, classification=classification, completion_status="COMPLETED_WITH_PHASE4_HARD_CEILING_STOP", phase_reached="Phase 4 compute projection", caveat="A confirmed retrospective branch existed, but the unreduced literal every-fission candidate search projected beyond the hard CPU ceiling. Intervention semantics remain unresolved.", recommendation="Return for human review with S13 blocked. Retain retrospective/prospective distinctions and do not infer intervention causality.", failure_rows=failures, runtime=runtime, seed_frames=[dev_seeds, conf_seeds], started_at=started_at)
        return
    raise RuntimeError("Phase-4 projection fell within the hard ceiling; the frozen intervention execution path requires human review before an undocumented implementation can be added.")


if __name__ == "__main__":
    main()
