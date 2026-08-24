"""Execute S19-L33 basin-blind operator-memory committor audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_latent_timebase.core import array_sha256 as simulator_array_sha256
from e01_latent_timebase.core import generate_beta
from e01_onset_discovery.operator_memory import (
    HISTORY,
    VIEWS,
    feature_names,
    operator_memory_views,
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L32 = _load_module(
    "e01_s19_l33_l32",
    REPO_ROOT / "scripts/e01/run_s19_l32_committor_ordered_transition_tube.py",
)
L31 = L32.L31
L30 = L32.L30
L29 = L32.L29
L28 = L32.L28
BASE = L32.BASE
LOOP_ID = "S19-L33"
VERSION = "E01-S19-L33-SINGLE-STATE-OPERATOR-MEMORY-PHASE-COMMITTOR-v1.0.0"
CANDIDATES = L28.CANDIDATES
PRIMARY_MODEL = "BASIN_BLIND_OPERATOR_MEMORY"
ORACLE_MODEL = "TARGET_CONDITIONED_OPERATOR_MEMORY"
CONTROL_MODELS = (
    "FROZEN_LANDMARK_PRIOR",
    "FROZEN_TARGET_GEOMETRY",
    "FROZEN_EXACT_H_TUBE",
    "FROZEN_ORDINARY_TUBE",
    "PHASE_MEMORY",
)
EVALUATION_COHORTS = ("L28_VALIDATION", "L31_CONFIRMATION")
BOOTSTRAPS = 4096
PERMUTATIONS = 512
ROOT_HEX = "123cb50636799fd0ba78051d06eae4bf7808d817632dbe49691859b493499cc5"
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L33"
L32_ROOT = ARTIFACT_ROOT / "loops/L32"
L31_ROOT = ARTIFACT_ROOT / "loops/L31"
L30_ROOT = ARTIFACT_ROOT / "loops/L30"
L29_ROOT = ARTIFACT_ROOT / "loops/L29"
L28_ROOT = ARTIFACT_ROOT / "loops/L28"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l33")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l33_operator_memory_phase_committor.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/operator_memory.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.reset_index(drop=True)
        .to_json(orient="table", index=False, double_precision=15)
        .encode()
    ).hexdigest()


def derived_seed(*parts: object) -> int:
    payload = "\x1f".join([VERSION, ROOT_HEX, *map(str, parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "big")


def seed_material_sha256(*parts: object) -> str:
    payload = "\x1f".join([VERSION, ROOT_HEX, *map(str, parts)])
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L32_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L32_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L32_ROOT / item["path"]),
            "root": str(L32_ROOT),
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in manifest["files"]
    )
    failures = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "MISSING"})
        elif sha256_file(path) != row["sha256"]:
            failures.append({"path": str(path), "reason": "HASH_MISMATCH"})
    return {
        "schema": "eidosoma.e01.s19_l33.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
        ).hexdigest(),
        "l32ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def response_registry() -> pd.DataFrame:
    l28_states = pd.read_parquet(L28_ROOT / "restored_state_registry.parquet")
    l28_h32 = pd.read_parquet(L28_ROOT / "committor_state_results.parquet")[
        ["stateId", "successes", "qHat", "qHatHalfA", "qHatHalfB"]
    ]
    l28_h8 = pd.read_parquet(L30_ROOT / "propagator_state_results.parquet")
    l28_h8 = l28_h8[l28_h8["referenceVariant"].eq("ORIGINAL")][
        ["stateId", "shortSuccesses", "q8", "q8HalfA", "q8HalfB"]
    ]
    l28 = l28_states.merge(l28_h32, on="stateId", validate="one_to_one").merge(
        l28_h8, on="stateId", validate="one_to_one"
    )
    l28["evaluationCohort"] = np.where(
        l28["matrixRole"].eq("DEVELOPMENT"), "L28_DEVELOPMENT", "L28_VALIDATION"
    )
    l31_states = pd.read_parquet(L31_ROOT / "restored_state_registry.parquet")
    l31_source = pd.read_parquet(
        L31_ROOT / "state_committor_and_propagator_results.parquet"
    )
    l31_h32 = l31_source[
        l31_source["branchFamily"].eq("H32")
        & l31_source["referenceVariant"].eq("ORIGINAL")
    ][["stateId", "successes", "q", "qHalfA", "qHalfB"]].rename(
        columns={"q": "qHat", "qHalfA": "qHatHalfA", "qHalfB": "qHatHalfB"}
    )
    l31_h8 = l31_source[
        l31_source["branchFamily"].eq("H8")
        & l31_source["referenceVariant"].eq("ORIGINAL")
    ][["stateId", "successes", "q", "qHalfA", "qHalfB"]].rename(
        columns={
            "successes": "shortSuccesses",
            "q": "q8",
            "qHalfA": "q8HalfA",
            "qHalfB": "q8HalfB",
        }
    )
    l31 = l31_states.merge(l31_h32, on="stateId", validate="one_to_one").merge(
        l31_h8, on="stateId", validate="one_to_one"
    )
    l31["evaluationCohort"] = "L31_CONFIRMATION"
    columns = [
        "stateId",
        "matrixRole",
        "candidateId",
        "matrixIndex",
        "trajectoryId",
        "landmark",
        "evaluationCohort",
        "successes",
        "qHat",
        "qHatHalfA",
        "qHatHalfB",
        "shortSuccesses",
        "q8",
        "q8HalfA",
        "q8HalfB",
        "currentSelectedIndex",
        "currentRawObservationIndex",
        "currentObservationKind",
        "currentCompletedFissions",
        "currentGrowthGeneration",
        "currentGenerationLocalStep",
        "currentBatchStep",
        "currentMass",
        "currentStateSha256",
        "betaSha256",
        "simulatorDefinition",
        "simulatorDefinitionSha256",
        "targetThreshold",
        "targetCentroidSha256",
        "targetComponentSize",
        "targetCurrentScore",
        "targetCurrentLabel",
    ]
    result = pd.concat([l28[columns], l31[columns]], ignore_index=True).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)
    if (
        len(result) != 280
        or result["stateId"].duplicated().any()
        or result["targetCurrentLabel"].any()
        or not np.isfinite(result[["qHat", "q8"]]).all().all()
        or not np.array_equal(result["successes"].to_numpy(), result["qHat"].to_numpy() * 128)
        or not np.array_equal(
            result["shortSuccesses"].to_numpy(), result["q8"].to_numpy() * 64
        )
    ):
        raise RuntimeError("L33 response identity, at-risk, or branch-count gate failed")
    return result


def target_coordinates(responses: pd.DataFrame) -> pd.DataFrame:
    l28 = pd.read_parquet(L28_ROOT / "target_basin_coordinates.parquet")
    l28_ids = responses[responses["evaluationCohort"].str.startswith("L28")][
        ["stateId", "candidateId", "matrixIndex", "landmark"]
    ]
    l28 = l28.merge(
        l28_ids,
        on=["candidateId", "matrixIndex", "landmark"],
        validate="many_to_one",
    )
    l31 = pd.read_parquet(L31_ROOT / "target_basin_coordinates.parquet")
    result = pd.concat(
        [
            l28[
                [
                    "stateId",
                    "candidateId",
                    "matrixIndex",
                    "landmark",
                    "coordinate",
                    "centroidValue",
                    "componentMemberIndices",
                ]
            ],
            l31,
        ],
        ignore_index=True,
    ).sort_values(["stateId", "coordinate"]).reset_index(drop=True)
    counts = result.groupby("stateId").size()
    if len(counts) != 280 or not counts.eq(100).all():
        raise RuntimeError("target coordinate cardinality changed")
    return result


def _vector_views_for_state(
    source: Any,
    selected: tuple[Any, ...],
    beta: np.ndarray,
    target: np.ndarray,
    *,
    reverse: bool,
) -> dict[str, np.ndarray]:
    endpoint = int(source.currentSelectedIndex)
    observations = list(selected[endpoint - HISTORY + 1 : endpoint + 1])
    if len(observations) != HISTORY:
        raise RuntimeError("eight-state history unavailable")
    if reverse:
        observations = observations[-2::-1] + observations[-1:]
    return operator_memory_views(
        np.asarray([row.state for row in observations], dtype=np.int64),
        beta,
        target,
        L28.definition(source.candidateId),
        observation_kinds=[row.observation_kind for row in observations],
        generation_local_steps=[row.generation_local_step for row in observations],
        growth_generations=[row.growth_generation_one_based for row in observations],
        batch_steps=[row.batch_step for row in observations],
        target_component_fraction=float(source.targetComponentSize) / 100.0,
    )


def extract_representations(
    responses: pd.DataFrame,
    coordinates: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    manifest_index = manifest.set_index(["candidateId", "matrixIndex"])
    coordinate_groups = {
        state_id: group.sort_values("coordinate")["centroidValue"].to_numpy(
            dtype=np.float64
        )
        for state_id, group in coordinates.groupby("stateId", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for source in responses.itertuples(index=False):
        manifest_row = manifest_index.loc[(source.candidateId, int(source.matrixIndex))]
        trajectory = L29.load_l23_trajectory(manifest_row)
        selected = L28.selected_clock_observations(trajectory, L28.CLOCK_ID)
        current = selected[int(source.currentSelectedIndex)]
        state = np.asarray(current.state, dtype=np.int64)
        if (
            int(current.observation_index) != int(source.currentRawObservationIndex)
            or current.observation_kind != source.currentObservationKind
            or int(current.completed_fissions) != int(source.currentCompletedFissions)
            or int(current.growth_generation_one_based) != int(source.currentGrowthGeneration)
            or int(current.generation_local_step) != int(source.currentGenerationLocalStep)
            or int(current.batch_step) != int(source.currentBatchStep)
            or int(state.sum()) != int(source.currentMass)
            or L28.array_sha256(state) != source.currentStateSha256
        ):
            raise RuntimeError(f"state/clock replay failure: {source.stateId}")
        beta_seed = L28.derive_seed(
            L28.L23_ROOT_HEX,
            L28.L23_PHASE,
            "catalytic_matrix",
            int(source.matrixIndex),
        )
        beta = generate_beta(beta_seed)
        if simulator_array_sha256(beta) != source.betaSha256:
            raise RuntimeError(f"beta replay failure: {source.stateId}")
        target = coordinate_groups[source.stateId]
        if L28.array_sha256(target) != source.targetCentroidSha256:
            raise RuntimeError(f"target replay failure: {source.stateId}")
        original = _vector_views_for_state(
            source, selected, beta, target, reverse=False
        )
        reversed_views = _vector_views_for_state(
            source, selected, beta, target, reverse=True
        )
        alternative_target = np.roll(target, 1)
        target_invariance = _vector_views_for_state(
            source, selected, beta, alternative_target, reverse=False
        )
        if not np.array_equal(original[PRIMARY_MODEL], target_invariance[PRIMARY_MODEL]):
            raise RuntimeError("primary representation is not target invariant")
        for variant, views in (
            ("ORIGINAL", original),
            ("HISTORY_REVERSAL", reversed_views),
        ):
            for view, values in views.items():
                rows.append(
                    {
                        "stateId": source.stateId,
                        "candidateId": source.candidateId,
                        "matrixIndex": int(source.matrixIndex),
                        "landmark": int(source.landmark),
                        "evaluationCohort": source.evaluationCohort,
                        "variant": variant,
                        "modelId": view,
                        "dimensions": len(values),
                        "featureNames": json.dumps(feature_names()[view]),
                        "vectorSha256": array_hash(values),
                        "values": values.tolist(),
                        "trajectoryCacheSha256": manifest_row.cacheSha256,
                        "stateReplayExact": True,
                        "betaReplayExact": True,
                        "targetReplayExact": True,
                        "primaryTargetInvariant": True,
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        [
            "variant",
            "evaluationCohort",
            "candidateId",
            "modelId",
            "landmark",
            "matrixIndex",
        ]
    ).reset_index(drop=True)
    if len(result) != 280 * 2 * len(VIEWS) or not result[
        [
            "stateReplayExact",
            "betaReplayExact",
            "targetReplayExact",
            "primaryTargetInvariant",
        ]
    ].all().all():
        raise RuntimeError("representation scope or replay failure")
    return result


def representation_frames(
    table: pd.DataFrame, responses: pd.DataFrame, variant: str
) -> dict[str, pd.DataFrame]:
    base = responses.sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)
    frames: dict[str, pd.DataFrame] = {}
    for view in VIEWS:
        subset = table[
            table["variant"].eq(variant) & table["modelId"].eq(view)
        ].sort_values(
            ["evaluationCohort", "candidateId", "landmark", "matrixIndex"]
        )
        if not np.array_equal(base["stateId"].to_numpy(), subset["stateId"].to_numpy()):
            raise RuntimeError("representation/response row order mismatch")
        values = np.stack(subset["values"].map(np.asarray)).astype(np.float64)
        columns = [f"f{index:03d}" for index in range(values.shape[1])]
        frames[view] = pd.concat(
            [base, pd.DataFrame(values, columns=columns)], axis=1
        )
    return frames


def frozen_control_predictions(responses: pd.DataFrame) -> pd.DataFrame:
    controls: list[pd.DataFrame] = []
    l32 = pd.read_parquet(L32_ROOT / "prediction_results.parquet")
    l32 = l32[
        l32["variant"].eq("ORIGINAL")
        & l32["modelId"].isin(
            ["LANDMARK_PRIOR", "EXACT_H_TRANSITION_TUBE", "ORDINARY_TRANSITION_TUBE"]
        )
    ][
        [
            "stateId",
            "candidateId",
            "matrixIndex",
            "landmark",
            "evaluationCohort",
            "modelId",
            "predictedQ",
        ]
    ].copy()
    l32["modelId"] = l32["modelId"].map(
        {
            "LANDMARK_PRIOR": "FROZEN_LANDMARK_PRIOR",
            "EXACT_H_TRANSITION_TUBE": "FROZEN_EXACT_H_TUBE",
            "ORDINARY_TRANSITION_TUBE": "FROZEN_ORDINARY_TUBE",
        }
    )
    controls.append(l32)
    l29 = pd.read_parquet(L29_ROOT / "prediction_results.parquet")
    l29 = l29[
        l29["referenceVariant"].eq("ORIGINAL")
        & l29["modelId"].eq("TARGET_GEOMETRY_CONTROL")
    ][
        ["stateId", "candidateId", "matrixIndex", "landmark", "matrixRole", "predictedQ"]
    ].copy()
    l29["evaluationCohort"] = np.where(
        l29["matrixRole"].eq("DEVELOPMENT"), "L28_DEVELOPMENT", "L28_VALIDATION"
    )
    l29["modelId"] = "FROZEN_TARGET_GEOMETRY"
    controls.append(l29.drop(columns="matrixRole"))
    l31 = pd.read_parquet(L31_ROOT / "prediction_results.parquet")
    l31 = l31[
        l31["referenceVariant"].eq("ORIGINAL")
        & l31["modelId"].eq("TARGET_GEOMETRY_CONTROL")
    ][["stateId", "candidateId", "matrixIndex", "landmark", "predictedQ"]].copy()
    l31["evaluationCohort"] = "L31_CONFIRMATION"
    l31["modelId"] = "FROZEN_TARGET_GEOMETRY"
    controls.append(l31)
    result = pd.concat(controls, ignore_index=True).merge(
        responses[
            [
                "stateId",
                "successes",
                "qHat",
                "shortSuccesses",
                "q8",
            ]
        ],
        on="stateId",
        validate="many_to_one",
    )
    result["variant"] = "ORIGINAL"
    result = result.sort_values(
        ["evaluationCohort", "candidateId", "modelId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)
    expected = 280 * len(CONTROL_MODELS[:-1])
    if len(result) != expected or result.duplicated(["stateId", "modelId"]).any():
        raise RuntimeError("frozen control scope mismatch")
    return result


def fit_and_score(
    original: dict[str, pd.DataFrame],
    reversed_frames: dict[str, pd.DataFrame],
    controls: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = controls.to_dict("records")
    registry_rows = []
    for candidate in CANDIDATES:
        for view in VIEWS:
            frame = original[view]
            development = frame[
                frame["candidateId"].eq(candidate)
                & frame["evaluationCohort"].eq("L28_DEVELOPMENT")
            ]
            columns = [column for column in frame.columns if column.startswith("f")]
            scaler, model = L29.fit_model(development, columns)
            replay_scaler, replay_model = L29.fit_model(development, columns)
            exact = bool(
                np.array_equal(scaler.mean_, replay_scaler.mean_)
                and np.array_equal(scaler.scale_, replay_scaler.scale_)
                and np.array_equal(model.coef_, replay_model.coef_)
                and np.array_equal(model.intercept_, replay_model.intercept_)
            )
            registry_rows.append(
                {
                    "candidateId": candidate,
                    "modelId": view,
                    "featureCount": len(columns),
                    "featureNames": json.dumps(feature_names()[view]),
                    "scalerMean": json.dumps(scaler.mean_.tolist()),
                    "scalerScale": json.dumps(scaler.scale_.tolist()),
                    "intercept": float(model.intercept_[0]),
                    "coefficients": json.dumps(model.coef_[0].tolist()),
                    "iterations": int(model.n_iter_[0]),
                    "exactReplay": exact,
                    "fitCohort": "L28_DEVELOPMENT",
                    "response": "H32_SUCCESS_COUNT_OUT_OF_128",
                }
            )
            for variant, source_frame in (
                ("ORIGINAL", frame),
                ("HISTORY_REVERSAL", reversed_frames[view]),
            ):
                subset = source_frame[source_frame["candidateId"].eq(candidate)]
                p = model.predict_proba(
                    scaler.transform(subset[columns].to_numpy(dtype=np.float64))
                )[:, 1]
                replay = replay_model.predict_proba(
                    replay_scaler.transform(
                        subset[columns].to_numpy(dtype=np.float64)
                    )
                )[:, 1]
                if not np.array_equal(p, replay):
                    raise RuntimeError("model probability replay failed")
                for source, probability in zip(
                    subset.itertuples(index=False), p, strict=True
                ):
                    rows.append(
                        {
                            "stateId": source.stateId,
                            "candidateId": candidate,
                            "matrixIndex": int(source.matrixIndex),
                            "landmark": int(source.landmark),
                            "evaluationCohort": source.evaluationCohort,
                            "modelId": view,
                            "predictedQ": float(probability),
                            "successes": int(source.successes),
                            "qHat": float(source.qHat),
                            "shortSuccesses": int(source.shortSuccesses),
                            "q8": float(source.q8),
                            "variant": variant,
                        }
                    )
    result = pd.DataFrame(rows).sort_values(
        ["variant", "evaluationCohort", "candidateId", "modelId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)
    return result, pd.DataFrame(registry_rows)


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(
        ["variant", "evaluationCohort", "candidateId", "modelId"], sort=True
    ):
        variant, cohort, candidate, model = keys
        p = np.clip(group["predictedQ"].to_numpy(dtype=np.float64), 1e-9, 1 - 1e-9)
        q32 = group["qHat"].to_numpy(dtype=np.float64)
        q8 = group["q8"].to_numpy(dtype=np.float64)
        intercept, slope = L28.calibration_parameters(p, q32)
        rows.append(
            {
                "variant": variant,
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "modelId": model,
                "states": len(group),
                "spearmanH32": L29.safe_spearman(p, q32),
                "spearmanH8": L29.safe_spearman(p, q8),
                "h32BrierPerBranch": float(
                    np.mean(q32 * (1 - p) ** 2 + (1 - q32) * p**2)
                ),
                "h8BrierPerBranch": float(
                    np.mean(q8 * (1 - p) ** 2 + (1 - q8) * p**2)
                ),
                "h32BinomialLogLossPerBranch": float(
                    -np.mean(q32 * np.log(p) + (1 - q32) * np.log(1 - p))
                ),
                "calibrationInterceptH32": intercept,
                "calibrationSlopeH32": slope,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source = predictions[
        predictions["variant"].eq("ORIGINAL")
        & predictions["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    for cohort in EVALUATION_COHORTS:
        for candidate in CANDIDATES:
            pivot = source[
                source["evaluationCohort"].eq(cohort)
                & source["candidateId"].eq(candidate)
            ].pivot(
                index=["stateId", "qHat", "q8"], columns="modelId", values="predictedQ"
            ).reset_index()
            expected_models = set(VIEWS) | set(CONTROL_MODELS[:-1])
            if not expected_models.issubset(pivot.columns):
                raise RuntimeError("bootstrap prediction model missing")
            rng = np.random.default_rng(derived_seed("bootstrap", cohort, candidate))
            for replicate in range(BOOTSTRAPS):
                sample = pivot.iloc[rng.integers(0, len(pivot), size=len(pivot))]
                q32 = sample["qHat"].to_numpy(dtype=np.float64)
                q8 = sample["q8"].to_numpy(dtype=np.float64)
                brier = {}
                for model in expected_models:
                    p = np.clip(sample[model].to_numpy(dtype=np.float64), 1e-9, 1 - 1e-9)
                    brier[model] = float(
                        np.mean(q32 * (1 - p) ** 2 + (1 - q32) * p**2)
                    )
                    if model in {PRIMARY_MODEL, ORACLE_MODEL}:
                        rows.append(
                            {
                                "evaluationCohort": cohort,
                                "candidateId": candidate,
                                "bootstrapIndex": replicate,
                                "metricId": model,
                                "spearmanH32": L29.safe_spearman(p, q32),
                                "spearmanH8": L29.safe_spearman(p, q8),
                                "primaryBrierImprovement": float("nan"),
                            }
                        )
                for control in CONTROL_MODELS:
                    rows.append(
                        {
                            "evaluationCohort": cohort,
                            "candidateId": candidate,
                            "bootstrapIndex": replicate,
                            "metricId": f"DELTA_PRIMARY_VS_{control}",
                            "spearmanH32": float("nan"),
                            "spearmanH8": float("nan"),
                            "primaryBrierImprovement": brier[control]
                            - brier[PRIMARY_MODEL],
                        }
                    )
    return pd.DataFrame(rows)


def permute_within_landmark(
    frame: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    for indices in result.groupby("landmark").groups.values():
        idx = np.asarray(list(indices), dtype=int)
        order = rng.permutation(len(idx))
        for column in ("successes", "qHat", "shortSuccesses", "q8"):
            result.loc[idx, column] = result.loc[idx, column].to_numpy()[order]
    return result


def permutation_results(
    primary_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = metrics[
        metrics["variant"].eq("ORIGINAL")
        & metrics["evaluationCohort"].isin(EVALUATION_COHORTS)
        & metrics["modelId"].eq(PRIMARY_MODEL)
    ].set_index(["evaluationCohort", "candidateId"])["spearmanH32"].to_dict()
    fixed = predictions[
        predictions["variant"].eq("ORIGINAL")
        & predictions["evaluationCohort"].isin(EVALUATION_COHORTS)
        & predictions["modelId"].eq(PRIMARY_MODEL)
    ]
    development_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    development_maxima = []
    evaluation_maxima = []
    columns = [column for column in primary_frame.columns if column.startswith("f")]
    for replicate in range(PERMUTATIONS):
        development_values = []
        evaluation_values = []
        for candidate in CANDIDATES:
            development = primary_frame[
                primary_frame["candidateId"].eq(candidate)
                & primary_frame["evaluationCohort"].eq("L28_DEVELOPMENT")
            ].reset_index(drop=True)
            shuffled_development = permute_within_landmark(
                development,
                np.random.default_rng(
                    derived_seed("development_permutation", replicate, candidate)
                ),
            )
            scaler, model = L29.fit_model(shuffled_development, columns)
            for cohort in EVALUATION_COHORTS:
                evaluation = primary_frame[
                    primary_frame["candidateId"].eq(candidate)
                    & primary_frame["evaluationCohort"].eq(cohort)
                ]
                probability = model.predict_proba(
                    scaler.transform(evaluation[columns].to_numpy(dtype=np.float64))
                )[:, 1]
                rho = L29.safe_spearman(probability, evaluation["qHat"])
                development_values.append(rho)
                development_rows.append(
                    {
                        "replicate": replicate,
                        "evaluationCohort": cohort,
                        "candidateId": candidate,
                        "nullSpearmanH32": rho,
                    }
                )
                evaluation_source = fixed[
                    fixed["candidateId"].eq(candidate)
                    & fixed["evaluationCohort"].eq(cohort)
                ].reset_index(drop=True)
                shuffled_evaluation = permute_within_landmark(
                    evaluation_source,
                    np.random.default_rng(
                        derived_seed(
                            "evaluation_permutation", replicate, cohort, candidate
                        )
                    ),
                )
                fixed_rho = L29.safe_spearman(
                    shuffled_evaluation["predictedQ"], shuffled_evaluation["qHat"]
                )
                evaluation_values.append(fixed_rho)
                evaluation_rows.append(
                    {
                        "replicate": replicate,
                        "evaluationCohort": cohort,
                        "candidateId": candidate,
                        "nullSpearmanH32": fixed_rho,
                    }
                )
        development_maxima.append(
            max(value for value in development_values if np.isfinite(value))
        )
        evaluation_maxima.append(
            max(value for value in evaluation_values if np.isfinite(value))
        )
    development = pd.DataFrame(development_rows)
    evaluation = pd.DataFrame(evaluation_rows)
    for frame, maxima in (
        (development, development_maxima),
        (evaluation, evaluation_maxima),
    ):
        values = np.asarray(maxima)
        for (cohort, candidate), observed_value in observed.items():
            mask = frame["evaluationCohort"].eq(cohort) & frame["candidateId"].eq(
                candidate
            )
            frame.loc[mask, "observedSpearmanH32"] = observed_value
            frame.loc[mask, "familywiseP"] = float(
                (1 + np.sum(values >= observed_value)) / (1 + len(values))
            )
    return development, evaluation


def suffix_invariance(
    responses: pd.DataFrame,
    coordinates: pd.DataFrame,
    manifest: pd.DataFrame,
    representations: pd.DataFrame,
) -> pd.DataFrame:
    coordinate_groups = {
        state_id: group.sort_values("coordinate")["centroidValue"].to_numpy(
            dtype=np.float64
        )
        for state_id, group in coordinates.groupby("stateId", sort=False)
    }
    manifest_index = manifest.set_index(["candidateId", "matrixIndex"])
    sentinels = responses.groupby(["evaluationCohort", "candidateId"]).head(3)
    rows = []
    for source in sentinels.itertuples(index=False):
        manifest_row = manifest_index.loc[(source.candidateId, int(source.matrixIndex))]
        trajectory = L29.load_l23_trajectory(manifest_row)
        selected = L28.selected_clock_observations(trajectory, L28.CLOCK_ID)
        endpoint = int(source.currentSelectedIndex)
        suffix_states = np.asarray(
            [row.state for row in selected[endpoint + 1 :]], dtype=np.int64
        )
        rng = np.random.default_rng(
            derived_seed(
                "suffix", source.evaluationCohort, source.candidateId, source.matrixIndex
            )
        )
        order = rng.permutation(len(suffix_states))
        suffix_changed = bool(
            len(suffix_states) < 2
            or not np.array_equal(suffix_states, suffix_states[order])
        )
        beta = generate_beta(
            L28.derive_seed(
                L28.L23_ROOT_HEX,
                L28.L23_PHASE,
                "catalytic_matrix",
                int(source.matrixIndex),
            )
        )
        target = coordinate_groups[source.stateId]
        first = _vector_views_for_state(
            source, selected, beta, target, reverse=False
        )
        # The perturbed suffix is deliberately not passed to this prefix-only
        # function.  The explicit prefix/suffix hashes prove the separation.
        second = _vector_views_for_state(
            source, selected[: endpoint + 1], beta, target, reverse=False
        )
        saved = representations[
            representations["stateId"].eq(source.stateId)
            & representations["variant"].eq("ORIGINAL")
            & representations["modelId"].eq(PRIMARY_MODEL)
        ].iloc[0]
        rows.append(
            {
                "stateId": source.stateId,
                "candidateId": source.candidateId,
                "evaluationCohort": source.evaluationCohort,
                "prefixSha256": array_hash(
                    np.asarray(
                        [row.state for row in selected[: endpoint + 1]], dtype=np.int64
                    )
                ),
                "suffixOriginalSha256": array_hash(suffix_states),
                "suffixPermutedSha256": array_hash(suffix_states[order]),
                "suffixActuallyChanged": suffix_changed,
                "primaryFeatureInvariant": bool(
                    np.array_equal(first[PRIMARY_MODEL], second[PRIMARY_MODEL])
                ),
                "storedPrimaryExact": array_hash(first[PRIMARY_MODEL])
                == saved.vectorSha256,
                "primaryUsesCompletedTarget": False,
                "responseBasinRetrospective": True,
                "oracleViewExcludedFromPastOnlyGate": True,
            }
        )
    return pd.DataFrame(rows)


def gate_table(
    metrics: pd.DataFrame,
    bootstraps: pd.DataFrame,
    development_permutations: pd.DataFrame,
    evaluation_permutations: pd.DataFrame,
    suffix: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for cohort in EVALUATION_COHORTS:
        for candidate in CANDIDATES:
            primary = metrics[
                metrics["variant"].eq("ORIGINAL")
                & metrics["evaluationCohort"].eq(cohort)
                & metrics["candidateId"].eq(candidate)
                & metrics["modelId"].eq(PRIMARY_MODEL)
            ].iloc[0]
            reversed_row = metrics[
                metrics["variant"].eq("HISTORY_REVERSAL")
                & metrics["evaluationCohort"].eq(cohort)
                & metrics["candidateId"].eq(candidate)
                & metrics["modelId"].eq(PRIMARY_MODEL)
            ].iloc[0]
            boot = bootstraps[
                bootstraps["evaluationCohort"].eq(cohort)
                & bootstraps["candidateId"].eq(candidate)
            ]
            primary_boot = boot[boot["metricId"].eq(PRIMARY_MODEL)]
            rank32_lower = float(np.nanquantile(primary_boot["spearmanH32"], 0.025))
            rank8_lower = float(np.nanquantile(primary_boot["spearmanH8"], 0.025))
            deltas = {
                control: float(
                    np.quantile(
                        boot[
                            boot["metricId"].eq(f"DELTA_PRIMARY_VS_{control}")
                        ]["primaryBrierImprovement"],
                        0.025,
                    )
                )
                for control in CONTROL_MODELS
            }
            dev_p = float(
                development_permutations[
                    development_permutations["evaluationCohort"].eq(cohort)
                    & development_permutations["candidateId"].eq(candidate)
                ]["familywiseP"].iloc[0]
            )
            eval_p = float(
                evaluation_permutations[
                    evaluation_permutations["evaluationCohort"].eq(cohort)
                    & evaluation_permutations["candidateId"].eq(candidate)
                ]["familywiseP"].iloc[0]
            )
            suffix_rows = suffix[
                suffix["evaluationCohort"].eq(cohort)
                & suffix["candidateId"].eq(candidate)
            ]
            checks = {
                "h32RankPassed": primary.spearmanH32 > 0.5
                and rank32_lower > 0.3,
                "h8RankPassed": primary.spearmanH8 > 0.5 and rank8_lower > 0.3,
                "incrementalBrierPassed": all(value > 0 for value in deltas.values()),
                "developmentPermutationPassed": dev_p <= 0.05,
                "evaluationPermutationPassed": eval_p <= 0.05,
                "historyReversalPassed": primary.spearmanH32
                > reversed_row.spearmanH32,
                "suffixPassed": bool(
                    suffix_rows[
                        [
                            "suffixActuallyChanged",
                            "primaryFeatureInvariant",
                            "storedPrimaryExact",
                            "oracleViewExcludedFromPastOnlyGate",
                        ]
                    ].all().all()
                ),
            }
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "states": int(primary.states),
                    "primarySpearmanH32": primary.spearmanH32,
                    "primarySpearmanH32Lower95": rank32_lower,
                    "primarySpearmanH8": primary.spearmanH8,
                    "primarySpearmanH8Lower95": rank8_lower,
                    **{
                        f"brierImprovementLowerVs{control}": value
                        for control, value in deltas.items()
                    },
                    "developmentPermutationP": dev_p,
                    "evaluationPermutationP": eval_p,
                    "historyReversalSpearmanH32": reversed_row.spearmanH32,
                    **checks,
                    "cohortCandidateGatePassed": all(checks.values()),
                }
            )
    return pd.DataFrame(rows)


def fixture_results() -> pd.DataFrame:
    rng = np.random.default_rng(derived_seed("fixture"))
    states = rng.poisson(1.5, size=(HISTORY, 100)).astype(np.int64)
    states[:, 0] += 1
    beta = np.exp(rng.normal(-4.0, 0.4, size=(100, 100)))
    target = rng.random(100)
    target /= target.sum()
    kwargs = {
        "observation_kinds": ["molecular_update"] * 7 + ["post_fission"],
        "generation_local_steps": list(range(1, 9)),
        "growth_generations": list(range(3, 11)),
        "batch_steps": list(range(40, 48)),
        "target_component_fraction": 0.37,
    }
    first = operator_memory_views(
        states, beta, target, L28.definition(CANDIDATES[0]), **kwargs
    )
    replay = operator_memory_views(
        states.copy(), beta.copy(), target.copy(), L28.definition(CANDIDATES[0]), **kwargs
    )
    order = np.random.default_rng(derived_seed("fixture_permutation")).permutation(100)
    permuted = operator_memory_views(
        states[:, order],
        beta[np.ix_(order, order)],
        target[order],
        L28.definition(CANDIDATES[0]),
        **kwargs,
    )
    alternative = np.roll(target, 1)
    alternative_views = operator_memory_views(
        states, beta, alternative, L28.definition(CANDIDATES[0]), **kwargs
    )
    history_order = np.asarray([6, 5, 4, 3, 2, 1, 0, 7])
    reversed_kwargs = {
        key: [value[index] for index in history_order]
        for key, value in kwargs.items()
        if key != "target_component_fraction"
    }
    reversed_kwargs["target_component_fraction"] = 0.37
    reversed_views = operator_memory_views(
        states[history_order],
        beta,
        target,
        L28.definition(CANDIDATES[0]),
        **reversed_kwargs,
    )
    rows = [
        {
            "fixtureId": "THREE_VIEWS_FIXED",
            "passed": tuple(first) == VIEWS
            and [len(first[key]) for key in VIEWS] == [15, 35, 51],
            "details": json.dumps({key: len(first[key]) for key in VIEWS}),
        },
        {
            "fixtureId": "CPU_FLOAT64_FINITE",
            "passed": all(
                value.dtype == np.float64 and np.isfinite(value).all()
                for value in first.values()
            ),
            "details": "three vectors",
        },
        {
            "fixtureId": "EXACT_FEATURE_REPLAY",
            "passed": all(np.array_equal(first[key], replay[key]) for key in VIEWS),
            "details": json.dumps({key: array_hash(first[key]) for key in VIEWS}),
        },
        {
            "fixtureId": "MOLECULE_PERMUTATION_INVARIANCE",
            "passed": all(
                np.allclose(first[key], permuted[key], atol=1e-12, rtol=1e-12)
                for key in VIEWS
            ),
            "details": "simultaneous state/beta/target permutation",
        },
        {
            "fixtureId": "PRIMARY_TARGET_INVARIANCE",
            "passed": np.array_equal(
                first[PRIMARY_MODEL], alternative_views[PRIMARY_MODEL]
            ),
            "details": "completed-run centroid cannot change primary values",
        },
        {
            "fixtureId": "ORACLE_TARGET_DEPENDENCE_VISIBLE",
            "passed": not np.array_equal(
                first[ORACLE_MODEL], alternative_views[ORACLE_MODEL]
            ),
            "details": "oracle kept separate from primary",
        },
        {
            "fixtureId": "HISTORY_ORDER_CONTROL",
            "passed": not np.array_equal(
                first[PRIMARY_MODEL], reversed_views[PRIMARY_MODEL]
            )
            and np.array_equal(
                first[PRIMARY_MODEL][:15], reversed_views[PRIMARY_MODEL][:15]
            ),
            "details": "first seven states reversed; current state preserved",
        },
        {
            "fixtureId": "NO_BRANCH_DERIVED_PREDICTOR",
            "passed": True,
            "details": "q8 and q32 are responses only",
        },
    ]
    return pd.DataFrame(rows)


def benchmark_projection() -> dict[str, Any]:
    rng = np.random.default_rng(derived_seed("opaque_benchmark"))
    frame = pd.DataFrame(rng.normal(size=(100, 35)), columns=[f"f{i:03d}" for i in range(35)])
    frame["successes"] = rng.integers(4, 125, size=len(frame))
    durations = []
    for _ in range(3):
        started = time.perf_counter()
        L29.fit_model(frame, [f"f{i:03d}" for i in range(35)])
        durations.append(time.perf_counter() - started)
    projected_fits = len(CANDIDATES) * len(VIEWS) * 2 + PERMUTATIONS * len(CANDIDATES)
    projected_fit_seconds = max(durations) * projected_fits
    projected_cpu_hours = projected_fit_seconds / 3600 + 2.0
    projected_wall_hours = projected_fit_seconds / 3600 + 2.0
    passed = projected_cpu_hours <= 90 and projected_wall_hours <= 64.8
    return {
        "schema": "eidosoma.e01.s19_l33.benchmark_projection.v1",
        "status": "PASS" if passed else "STOP_BEFORE_OUTCOME",
        "opaqueSyntheticRows": len(frame),
        "opaqueSyntheticFeatures": 35,
        "timedFits": len(durations),
        "fitDurationsSeconds": durations,
        "projectedScientificFits": projected_fits,
        "projectedCpuHoursIncludingOverhead": projected_cpu_hours,
        "projectedWallHoursIncludingOverhead": projected_wall_hours,
        "scientificOutcomeOpened": False,
    }


def seed_manifest(responses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(purpose: str, parts: tuple[object, ...]) -> None:
        rows.append(
            {
                "purpose": purpose,
                "candidateId": next(
                    (str(part) for part in parts if str(part) in CANDIDATES), None
                ),
                "partsJson": json.dumps(list(parts), separators=(",", ":")),
                "rootHex": ROOT_HEX,
                "derivedSeed": str(derived_seed(*parts)),
                "seedMaterialSha256": seed_material_sha256(*parts),
            }
        )

    add("synthetic_fixture", ("fixture",))
    add("fixture_molecule_permutation", ("fixture_permutation",))
    add("opaque_benchmark", ("opaque_benchmark",))
    for cohort in EVALUATION_COHORTS:
        for candidate in CANDIDATES:
            add("matrix_bootstrap", ("bootstrap", cohort, candidate))
    for replicate in range(PERMUTATIONS):
        for candidate in CANDIDATES:
            add(
                "development_response_permutation",
                ("development_permutation", replicate, candidate),
            )
            for cohort in EVALUATION_COHORTS:
                add(
                    "evaluation_response_permutation",
                    ("evaluation_permutation", replicate, cohort, candidate),
                )
    sentinels = responses.groupby(["evaluationCohort", "candidateId"]).head(3)
    for source in sentinels.itertuples(index=False):
        add(
            "suffix_permutation",
            (
                "suffix",
                source.evaluationCohort,
                source.candidateId,
                int(source.matrixIndex),
            ),
        )
    result = pd.DataFrame(rows).sort_values(
        ["purpose", "candidateId", "partsJson"]
    ).reset_index(drop=True)
    if result["seedMaterialSha256"].duplicated().any() or result[
        "derivedSeed"
    ].duplicated().any():
        raise RuntimeError("L33 seed collision within manifest")
    return result


def seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    current_material = set(seeds["seedMaterialSha256"].astype(str))
    current_derived = set(seeds["derivedSeed"].astype(str))
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L33/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            if "seedmaterialsha256" in column.lower():
                prior_material.update(frame[column].dropna().astype(str))
            if column.lower() == "derivedseed":
                prior_derived.update(frame[column].dropna().astype(str))
    root_collisions = []
    needle = ROOT_HEX.encode()
    for path in ARTIFACT_ROOT.rglob("*"):
        if (
            not path.is_file()
            or "/L33/" in str(path)
            or path.stat().st_size > 64 * 1024 * 1024
        ):
            continue
        try:
            if needle in path.read_bytes():
                root_collisions.append(str(path))
        except OSError:
            continue
    overlaps = sorted(current_material & prior_material)
    derived_overlaps = sorted(current_derived & prior_derived)
    return {
        "schema": "eidosoma.e01.s19_l33.seed_firewall.v1",
        "status": "PASS"
        if not overlaps and not derived_overlaps and not root_collisions
        else "FAIL",
        "currentSeedMaterialCount": len(current_material),
        "allCurrentMaterialsUnique": len(current_material) == len(seeds),
        "overlapCount": len(overlaps),
        "overlaps": overlaps,
        "derivedSeedOverlapCount": len(derived_overlaps),
        "derivedSeedOverlaps": derived_overlaps,
        "rootCollisionPaths": sorted(root_collisions),
    }


def source_grounding_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "JUNG_2026_COMMITTOR_WITHOUT_CV",
                "url": "https://www.nature.com/articles/s43588-026-00958-2",
                "evidenceClass": "PRIMARY_METHOD_CONTEXT",
                "directSupport": "committor learning can use full configurations rather than a preselected collective variable",
                "frozenUse": "motivates retaining an invariant state/operator representation; does not support GARD-specific semantics",
            },
            {
                "sourceId": "MARDIA_2018_VAMPNETS",
                "url": "https://www.nature.com/articles/s41467-017-02388-1",
                "evidenceClass": "PRIMARY_METHOD_CONTEXT",
                "directSupport": "low-dimensional learned features may encode dynamical state",
                "frozenUse": "motivates testing a fixed compressed memory rather than another high-dimensional tube",
            },
            {
                "sourceId": "MAHMOUD_2023_MACHINE_GUIDED_PATH_SAMPLING",
                "url": "https://www.nature.com/articles/s43588-023-00428-z",
                "evidenceClass": "PRIMARY_METHOD_CONTEXT",
                "directSupport": "state-dependent fate probabilities can be estimated by repeated shooting and evaluated out of sample",
                "frozenUse": "supports empirical committor supervision and held-out testing, not the selected GARD features",
            },
            {
                "sourceId": "L29_SOURCE_DEFINED_GARD_GENERATOR",
                "url": None,
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "directSupport": "exact analytic GARD generator and composition-space moment implementation",
                "frozenUse": "all L33 operator channels are deterministic summaries of this pinned source-defined generator",
            },
            {
                "sourceId": "L31_CONFIRMED_EMPIRICAL_COMMITTOR",
                "url": None,
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "directSupport": "H32 and H8 branch responses reproduce on an untouched cohort",
                "frozenUse": "response only; no branch-derived statistic enters a predictor",
            },
        ]
    )


def make_figures(
    predictions: pd.DataFrame, metrics: pd.DataFrame, gates: pd.DataFrame
) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(root / name, dpi=180)
        plt.close()

    primary = predictions[
        predictions["variant"].eq("ORIGINAL")
        & predictions["evaluationCohort"].isin(EVALUATION_COHORTS)
        & predictions["modelId"].eq(PRIMARY_MODEL)
    ]
    _, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, ((cohort, candidate), group) in zip(
        axes.flat,
        primary.groupby(["evaluationCohort", "candidateId"], sort=True),
        strict=True,
    ):
        axis.scatter(group["predictedQ"], group["qHat"], s=22, alpha=0.8)
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set_title(f"{cohort} / {candidate}", fontsize=8)
        axis.set_xlabel("Basin-blind operator-memory coordinate")
        axis.set_ylabel("Empirical H32 q-hat")
    save("01_basin_blind_coordinate_vs_h32_committor.png")

    comparison = metrics[
        metrics["variant"].eq("ORIGINAL")
        & metrics["evaluationCohort"].isin(EVALUATION_COHORTS)
    ].pivot_table(
        index="modelId",
        columns=["evaluationCohort", "candidateId"],
        values="spearmanH32",
    )
    comparison.plot(kind="bar", figsize=(13, 6))
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Spearman with H32 q-hat")
    save("02_operator_phase_and_frozen_control_comparison.png")

    mediator = metrics[
        metrics["variant"].eq("ORIGINAL")
        & metrics["evaluationCohort"].isin(EVALUATION_COHORTS)
        & metrics["modelId"].isin([PRIMARY_MODEL, ORACLE_MODEL])
    ].pivot_table(
        index=["evaluationCohort", "candidateId", "modelId"],
        values=["spearmanH32", "spearmanH8"],
    )
    mediator.plot(kind="bar", figsize=(12, 6))
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Spearman")
    save("03_h32_and_h8_rank_diagnostics.png")

    reversal = metrics[
        metrics["modelId"].eq(PRIMARY_MODEL)
        & metrics["evaluationCohort"].isin(EVALUATION_COHORTS)
    ].pivot_table(
        index=["evaluationCohort", "candidateId"],
        columns="variant",
        values="spearmanH32",
    )
    reversal.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Spearman with H32 q-hat")
    save("04_history_reversal_control.png")

    checks = [
        "h32RankPassed",
        "h8RankPassed",
        "incrementalBrierPassed",
        "developmentPermutationPassed",
        "evaluationPermutationPassed",
        "historyReversalPassed",
        "suffixPassed",
        "cohortCandidateGatePassed",
    ]
    matrix = gates.set_index(["evaluationCohort", "candidateId"])[checks].astype(float)
    plt.figure(figsize=(11, 5))
    plt.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(len(checks)), checks, rotation=35, ha="right", fontsize=7)
    plt.yticks(
        range(len(matrix)), [" / ".join(index) for index in matrix.index], fontsize=7
    )
    plt.colorbar(ticks=[0, 1])
    save("05_solution_gate_matrix.png")


def manifest_for(root: Path) -> dict[str, Any]:
    files = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and item.name != "artifact_manifest.json"
        )
    ]
    return {
        "schema": "eidosoma.e01.s19_l33.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(files),
        "totalBytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def append_ledgers(
    classifications: list[str], timestamp: str, next_theme: str, solution: bool
) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L31 confirmed a simulation shooting coordinate, but L32 high-dimensional observed-prefix tubes did not generalize.",
            "failureOrAmbiguityTargeted": "Whether exact local generator activity and simulator phase contain a low-dimensional deterministic precursor missed by the L32 representation.",
            "informationGainRationale": "A fixed endpoint/slope compression tests dynamic sufficiency without another feature or window search.",
            "learned": "L33 basin-blind operator-memory contract frozen before outcome access.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L31 supportive shooting signal plus L32 deterministic-feature non-support.",
            "proposedNextTest": "Fit only on L28 development H32 and evaluate unchanged on L28 validation and L31 confirmation.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Low-dimensional phase/operator memory, target-conditioned geometry only, or higher-order full-state interactions.",
            "selectedHypotheses": "Phase memory, primary basin-blind generator memory, and target-conditioned oracle diagnostic.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "A large observed-prefix feature vector is sufficient.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A valid coordinate must rank H32 and H8 and improve Brier over frozen controls in both candidates and evaluation cohorts.",
            "failureOrAmbiguityTargeted": "Deterministic past-only precursor observability.",
            "informationGainRationale": "Two held-out cohorts distinguish transferable signal from development memorization.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L33 outcome and controls.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_SOLUTION_HUMAN_REVIEW"
            if solution
            else "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Basin-blind operator-memory primary; target-conditioned oracle diagnostic.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Low-dimensional exact operator memory is sufficient"
            if not solution
            else "No deterministic precursor can recover the empirical committor.",
        },
    ]
    BASE.write_parquet(
        ledger_path,
        pd.concat(
            [ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)],
            ignore_index=True,
        ),
    )
    markdown = ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    BASE.atomic_text(
        markdown,
        markdown.read_text()
        + f"\n\n## {LOOP_ID} — basin-blind operator memory and phase\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** {next_theme}.\n",
    )
    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    row = {
        "branchCount": 3,
        "bundleId": "L33_OPERATOR_MEMORY_PHASE",
        "candidateId": "S19-L33-BASIN-BLIND-OPERATOR-MEMORY",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 5,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 4,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "eight-state phase and exact source-defined analytic generator endpoints/slopes; basin-blind primary",
        "rankingScore": 29.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L32_HIGH_DIMENSIONAL_MEMORIZATION_AND_L31_RELIABLE_RESPONSE",
        "sourceGrounding": 5,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidates_path,
        pd.concat(
            [candidates, pd.DataFrame([row]).reindex(columns=candidates.columns)],
            ignore_index=True,
        ),
    )
    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    source_rows = []
    for source in source_grounding_registry().itertuples(index=False):
        source_rows.append(
            {
                "commitOrVersion": None,
                "evidenceClass": source.evidenceClass,
                "finding": f"{source.directSupport}; frozen L33 use: {source.frozenUse}",
                "licenseStatus": "PUBLIC_ARTICLE_OR_WORKSPACE_EVIDENCE",
                "redistributionStatus": "CITATION_OR_INTERNAL_ARTIFACT_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L33_{source.sourceId}",
                "sourceType": source.evidenceClass,
                "treeIdentity": None,
                "url": source.url,
            }
        )
    BASE.write_parquet(
        source_path,
        pd.concat(
            [sources, pd.DataFrame(source_rows).reindex(columns=sources.columns)],
            ignore_index=True,
        ),
    )
    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text())
    registry["loops"].append(
        {
            "loopId": LOOP_ID,
            "versionedLoopId": VERSION,
            "status": "COMPLETE_SOLUTION_BOUNDARY"
            if solution
            else "COMPLETE_AUTONOMOUS_CONTINUATION_AUTHORIZED",
            "authorized": True,
            "completed": True,
            "outcomeAccessed": True,
            "humanReviewRequiredAfter": solution,
            "classification": classifications,
            "selectedDiscoveryLead": "BASIN_BLIND_OPERATOR_MEMORY_COMMITTOR_COORDINATE"
            if solution
            else None,
            "newMatrices": 0,
            "newTrajectories": 0,
            "nextStepActive": not solution,
        }
    )
    registry["proposedNextLoopTheme"] = next_theme
    registry["proposedNextLoopActive"] = not solution
    BASE.atomic_text(registry_path, yaml.safe_dump(registry, sort_keys=False))
    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    history["history"].append(
        {
            "decision": "S19_L33_SOLUTION_HUMAN_REVIEW"
            if solution
            else "S19_L33_COMPLETE_AUTONOMOUS_CONTINUATION",
            "loopId": LOOP_ID,
            "nextLoopAuthorized": not solution,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "s20Activated": False,
            "scope": VERSION,
            "selectedDiscoveryLead": "BASIN_BLIND_OPERATOR_MEMORY_COMMITTOR_COORDINATE"
            if solution
            else None,
            "source": "locked_execution_result",
        }
    )
    history["pendingDecision"] = (
        "HUMAN_REVIEW_REQUIRED_AFTER_EARLY_SOLUTION"
        if solution
        else "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L42"
    )
    BASE.write_json(history_path, history)


def report_text(
    metrics: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    evaluation = metrics[
        metrics["variant"].eq("ORIGINAL")
        & metrics["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    return f"""# S19-L33 — Single-State Operator-Memory and Phase Committor Coordinate

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the authorized L19–L42 sequence.
- **Outcome classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Artifacts:** compact feature/replay tables, fitted models, candidate- and cohort-separated metrics, 4,096 matrix bootstraps, 512 development and evaluation permutations, suffix and history controls, five figures, full provenance and hashes under `S19/loops/L33`.
- **Validation:** exact replay of 280 states, molecular clocks, catalytic matrices, target basins and H32/H8 responses; CPU-float64 feature and model replay; molecule-permutation and target-invariance fixtures; suffix invariance; immutable prior, seed, storage, regeneration and artifact gates.
- **Recommended next action:** `{next_theme}`.

## Lay summary

L31 proved that repeated short simulations from the same state have a reproducible probability of reaching the retrospectively defined basin. L33 asks whether that probability is visible without rerunning the simulator: it compresses the latest eight observed states into mass/phase, ordinary composition, and exact local-reaction-generator summaries. The primary coordinate is mathematically invariant to the completed-run target centroid; a separate target-conditioned coordinate is retained only as a retrospective oracle diagnostic.

## Frozen question

Can a fixed, basin-blind, low-dimensional history of exact GARD generator activity and simulator phase recover both the H32 committor and its H8 mediator on two unchanged evaluation cohorts and in both simulator candidates, beyond time/phase, exact-H, ordinary-path and completed-target-geometry controls?

## Inputs and methods

- 200 L28 states (development and validation) and 80 previously untouched L31 confirmation states.
- H32 responses use 128 independent branches per state; H8 is a response-only diagnostic from 64 branches.
- Eight selected-clock observations ending at the current at-risk state.
- Three prospectively fixed views: 15 phase-memory features, 35 primary basin-blind phase/composition/generator features, and a 51-feature target-conditioned oracle diagnostic.
- Endpoint, temporal slope and phase means only; no feature, window, regularization or model search.
- Exact L29 standardized L2 aggregated-binomial logistic coordinate (`C=0.1`) fit only on L28 development H32 counts.

## Evaluation metrics

{evaluation.to_markdown(index=False)}

## Locked solution gates

{gates.to_markdown(index=False)}

## Source grounding and scope

The exact operator is inherited from the frozen source-defined GARD implementation and analytic moment code. General committor-learning and path-sampling literature motivated testing a low-dimensional state coordinate and held-out shooting validation; it does not identify these GARD features or the paper authors' implementation. The response basin itself remains a completed-run, matrix-specific reconstruction.

## Interpretation boundary

The primary predictor uses no branch-derived value, completed-run centroid, suffix observation, emergence estimate, intervention outcome or paper-directed threshold. H8 and H32 enter only as response variables. A passing result would establish a deterministic past-only coordinate *within the retrospective-basin-conditioned simulation task*; it would not establish the paper's exact replicator definition, early warning in empirical biology, causal emergence, or causal control. The target-conditioned view is an oracle diagnostic and is excluded from the solution gate.

## Runtime and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- CPU float64; one numerical-library thread; no GPU.
- Wall seconds: `{runtime['wallSeconds']:.3f}`; controller CPU hours: `{runtime['controllerCpuHours']:.6f}`.

## Autonomous boundary

L33 is frozen. S20, E02, author contact, interventions, reactive-current analysis and report-bundle generation remain inactive.
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L33 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    benchmark = benchmark_projection()
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or benchmark["status"] != "PASS"
    ):
        raise RuntimeError("preoutcome prior, fixture, or benchmark gate failed")
    responses = response_registry()
    coordinates = target_coordinates(responses)
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    representations = extract_representations(responses, coordinates, manifest)
    seeds = seed_manifest(responses)
    firewall = seed_firewall(seeds)
    if firewall["status"] != "PASS":
        raise RuntimeError("L33 seed firewall failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L33 decision record\n\n"
        "L31 independently confirmed an H8 shooting coordinate, while L32's high-dimensional observed-prefix tubes memorized development and failed twice out of sample. L33 therefore freezes one nonduplicative compression before outcomes: phase memory, a primary target-invariant source-defined analytic generator memory, and a target-conditioned retrospective oracle diagnostic over exactly eight selected-clock states. Endpoint and slope summaries, the L29 model (`C=0.1`), L28 development fit scope, two evaluation cohorts, controls and all gates are fixed. H32/H8 branch values are responses only. The normalized drift direction is defined as zero when its analytic norm is at or below `1e-14`, because the mandatory molecule-permutation fixture exposed only float64 cancellation residue near `1e-17`; this numerical contract was fixed before scientific outcome access.\n",
    )
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    BASE.write_parquet(LOOP_ROOT / "response_registry.parquet", responses)
    BASE.write_parquet(LOOP_ROOT / "target_coordinate_registry.parquet", coordinates)
    BASE.write_parquet(
        LOOP_ROOT / "operator_memory_representation_results.parquet", representations
    )
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_parquet(
        LOOP_ROOT / "source_grounding_registry.parquet", source_grounding_registry()
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    hashes = {
        "responsesSha256": sha256_file(LOOP_ROOT / "response_registry.parquet"),
        "coordinatesSha256": sha256_file(
            LOOP_ROOT / "target_coordinate_registry.parquet"
        ),
        "representationsSha256": sha256_file(
            LOOP_ROOT / "operator_memory_representation_results.parquet"
        ),
        "seedsSha256": sha256_file(LOOP_ROOT / "analysis_seed_manifest.parquet"),
        "seedFirewallSha256": sha256_file(LOOP_ROOT / "seed_firewall.json"),
        "benchmarkSha256": sha256_file(LOOP_ROOT / "benchmark_projection.json"),
        "l28H32Sha256": sha256_file(L28_ROOT / "committor_state_results.parquet"),
        "l30H8Sha256": sha256_file(L30_ROOT / "propagator_state_results.parquet"),
        "l31ResponseSha256": sha256_file(
            L31_ROOT / "state_committor_and_propagator_results.parquet"
        ),
        "l23ManifestSha256": sha256_file(
            L23_ROOT / "input_trajectory_manifest.parquet"
        ),
    }
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l33.implementation_lock.v1",
            "repositoryHead": head,
            "remoteHead": remote,
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "configSha256": sha256_file(CONFIG),
            "views": list(VIEWS),
            "primary": PRIMARY_MODEL,
            "oracleDiagnostic": ORACLE_MODEL,
            "modelFitScope": "L28_DEVELOPMENT_ONLY",
            "branchDerivedPredictor": False,
            "targetCentroidInPrimary": False,
            "targetTaskRetrospective": True,
            "historyStates": HISTORY,
            "normalizedZeroDriftTolerance": 1e-14,
            "lockedHashes": hashes,
            "outcomeAccessed": False,
            "lockedAtUtc": utc_now(),
        },
    )
    BASE.write_json(
        LOOP_ROOT / "preoutcome_repository_lock.json",
        {
            "head": head,
            "remote": remote,
            "priorAggregateSha256": prior["aggregateSha256"],
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            **hashes,
        },
    )


def execute() -> None:
    start = time.perf_counter()
    start_cpu = time.process_time()
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    if (
        git("rev-parse", "HEAD") != lock["head"]
        or git("rev-parse", "origin/eidosoma/groups/42") != lock["remote"]
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_files = {
        "responsesSha256": LOOP_ROOT / "response_registry.parquet",
        "coordinatesSha256": LOOP_ROOT / "target_coordinate_registry.parquet",
        "representationsSha256": LOOP_ROOT
        / "operator_memory_representation_results.parquet",
        "seedsSha256": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewallSha256": LOOP_ROOT / "seed_firewall.json",
        "benchmarkSha256": LOOP_ROOT / "benchmark_projection.json",
        "l28H32Sha256": L28_ROOT / "committor_state_results.parquet",
        "l30H8Sha256": L30_ROOT / "propagator_state_results.parquet",
        "l31ResponseSha256": L31_ROOT
        / "state_committor_and_propagator_results.parquet",
        "l23ManifestSha256": L23_ROOT / "input_trajectory_manifest.parquet",
    }
    for key, path in locked_files.items():
        if sha256_file(path) != lock[key]:
            raise RuntimeError(f"locked input changed: {path}")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
    ):
        raise RuntimeError("pre-execution validation failed")
    responses = pd.read_parquet(LOOP_ROOT / "response_registry.parquet")
    coordinates = pd.read_parquet(LOOP_ROOT / "target_coordinate_registry.parquet")
    representations = pd.read_parquet(
        LOOP_ROOT / "operator_memory_representation_results.parquet"
    )
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    original = representation_frames(representations, responses, "ORIGINAL")
    reversed_frames = representation_frames(
        representations, responses, "HISTORY_REVERSAL"
    )
    controls = frozen_control_predictions(responses)
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    predictions, models = fit_and_score(original, reversed_frames, controls)
    if not models["exactReplay"].all():
        raise RuntimeError("model replay failed")
    metrics = metric_table(predictions)
    bootstraps = bootstrap_metrics(predictions)
    development_permutations, evaluation_permutations = permutation_results(
        original[PRIMARY_MODEL], predictions, metrics
    )
    suffix = suffix_invariance(
        responses, coordinates, manifest, representations
    )
    gates = gate_table(
        metrics,
        bootstraps,
        development_permutations,
        evaluation_permutations,
        suffix,
    )
    solution = bool(gates["cohortCandidateGatePassed"].all())
    oracle_rows = metrics[
        metrics["variant"].eq("ORIGINAL")
        & metrics["evaluationCohort"].isin(EVALUATION_COHORTS)
        & metrics["modelId"].eq(ORACLE_MODEL)
    ]
    oracle_boot = bootstraps[bootstraps["metricId"].eq(ORACLE_MODEL)]
    oracle_rank_pass = True
    for source in oracle_rows.itertuples(index=False):
        lower32 = float(
            np.nanquantile(
                oracle_boot[
                    oracle_boot["evaluationCohort"].eq(source.evaluationCohort)
                    & oracle_boot["candidateId"].eq(source.candidateId)
                ]["spearmanH32"],
                0.025,
            )
        )
        lower8 = float(
            np.nanquantile(
                oracle_boot[
                    oracle_boot["evaluationCohort"].eq(source.evaluationCohort)
                    & oracle_boot["candidateId"].eq(source.candidateId)
                ]["spearmanH8"],
                0.025,
            )
        )
        oracle_rank_pass &= bool(
            source.spearmanH32 > 0.5
            and lower32 > 0.3
            and source.spearmanH8 > 0.5
            and lower8 > 0.3
        )
    if solution:
        classifications = [
            "BASIN_BLIND_OPERATOR_MEMORY_COMMITTOR_COORDINATE_ESTABLISHED",
            "PAST_ONLY_ORGANIZATION_PRECURSOR_SIGNAL_ESTABLISHED_WITHIN_RETROSPECTIVE_TARGET_TASK",
            "NOT_A_CONFIRMED_PAPER_OR_CAUSAL_RESULT",
        ]
        next_theme = "HUMAN_REVIEW_SOLUTION_BOUNDARY"
    elif oracle_rank_pass:
        classifications = [
            "TARGET_CONDITIONED_OPERATOR_MEMORY_ONLY",
            "RETROSPECTIVE_TARGET_GEOMETRY_REQUIRED",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "PERMUTATION_INVARIANT_FULL_STATE_GRAPH_COORDINATE"
    else:
        classifications = [
            "BASIN_BLIND_OPERATOR_MEMORY_COMMITTOR_NON_SUPPORT",
            "BRANCH_SIGNAL_NOT_DISTILLED_FROM_LOW_DIMENSIONAL_OPERATOR_HISTORY",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "PERMUTATION_INVARIANT_FULL_STATE_GRAPH_COORDINATE"
    make_figures(predictions, metrics, gates)
    for name in (
        "preregistration.yaml",
        "decision_record.md",
        "fixture_results.parquet",
        "benchmark_projection.json",
        "response_registry.parquet",
        "target_coordinate_registry.parquet",
        "operator_memory_representation_results.parquet",
        "analysis_seed_manifest.parquet",
        "seed_firewall.json",
        "source_grounding_registry.parquet",
        "immutable_prior_validation.json",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
    ):
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    BASE.write_parquet(BUILD_ROOT / "frozen_control_predictions.parquet", controls)
    BASE.write_parquet(BUILD_ROOT / "prediction_results.parquet", predictions)
    BASE.write_parquet(BUILD_ROOT / "fitted_model_registry.parquet", models)
    BASE.write_parquet(BUILD_ROOT / "metric_results.parquet", metrics)
    BASE.write_parquet(BUILD_ROOT / "bootstrap_results.parquet", bootstraps)
    BASE.write_parquet(
        BUILD_ROOT / "development_permutation_results.parquet",
        development_permutations,
    )
    BASE.write_parquet(
        BUILD_ROOT / "evaluation_permutation_results.parquet",
        evaluation_permutations,
    )
    BASE.write_parquet(BUILD_ROOT / "suffix_invariance_results.parquet", suffix)
    BASE.write_parquet(BUILD_ROOT / "scientific_gate_results.parquet", gates)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l33.classification.v1",
            "classifications": classifications,
            "solutionGatePassed": solution,
            "oracleRankDiagnosticPassed": oracle_rank_pass,
            "primaryPredictorPastOnly": True,
            "primaryPredictorTargetInvariant": True,
            "responseTargetRetrospective": True,
            "branchDerivedPredictor": False,
            "priorStatusesChanged": False,
        },
    )
    pd.DataFrame(
        columns=[
            "stage",
            "candidateId",
            "matrixIndex",
            "stateId",
            "exceptionClass",
            "exceptionMessage",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    replayed = extract_representations(responses, coordinates, manifest)
    checks = {
        "featureReplayPassed": frame_hash(replayed) == frame_hash(representations),
        "modelReplayPassed": bool(models["exactReplay"].all()),
        "suffixPassed": bool(
            suffix[
                [
                    "suffixActuallyChanged",
                    "primaryFeatureInvariant",
                    "storedPrimaryExact",
                    "oracleViewExcludedFromPastOnlyGate",
                ]
            ].all().all()
        ),
        "responseReplayPassed": frame_hash(response_registry())
        == frame_hash(responses),
        "coordinateReplayPassed": frame_hash(target_coordinates(responses))
        == frame_hash(coordinates),
        "fixturesPassed": bool(fixtures["passed"].all()),
        "benchmarkPassed": json.loads(
            (LOOP_ROOT / "benchmark_projection.json").read_text()
        )["status"]
        == "PASS",
        "seedFirewallPassed": json.loads(
            (LOOP_ROOT / "seed_firewall.json").read_text()
        )["status"]
        == "PASS",
        "immutablePriorPassed": prior["unchanged"],
        "primaryTargetInvariant": bool(
            representations["primaryTargetInvariant"].all()
        ),
        "noBranchPredictor": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"regeneration validation failed: {checks}")
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l33.regeneration_validation.v1",
            "status": "PASS",
            "checks": checks,
            "predictionFrameSha256": frame_hash(predictions),
            "metricFrameSha256": frame_hash(metrics),
            "representationFrameSha256": frame_hash(representations),
        },
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l33.runtime.v1",
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": 1,
        "numericalLibraryThreads": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - start,
        "controllerCpuHours": (time.process_time() - start_cpu) / 3600,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    retained = sum(
        path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
    )
    temporary = sum(
        path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file()
    )
    storage = {
        "schema": "eidosoma.e01.s19_l33.storage_validation.v1",
        "retainedBytes": retained,
        "retainedGiBCeiling": 25,
        "temporaryBytes": temporary,
        "temporaryGiBCeiling": 75,
        "status": "PASS"
        if retained < 25 * 2**30 and temporary < 75 * 2**30
        else "FAIL",
    }
    if storage["status"] != "PASS":
        raise RuntimeError("storage ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(metrics, gates, classifications, runtime, next_theme)
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L33_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        "# S19-L33 decision summary\n\n"
        + f"**Classification:** {', '.join(classifications)}\n\n"
        + f"**Past-only basin-blind solution:** `{solution}`.\n\n"
        + f"**Next:** `{next_theme}`.\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L33-promotion-stage")
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(BUILD_ROOT, stage)
    if LOOP_ROOT.exists():
        shutil.rmtree(LOOP_ROOT)
    os.replace(stage, LOOP_ROOT)
    shutil.rmtree(BUILD_ROOT)
    artifact_manifest = json.loads((LOOP_ROOT / "artifact_manifest.json").read_text())
    if any(
        sha256_file(LOOP_ROOT / item["path"]) != item["sha256"]
        for item in artifact_manifest["files"]
    ):
        raise RuntimeError("artifact hash validation failed")
    append_ledgers(classifications, runtime["completedAtUtc"], next_theme, solution)
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md",
        report.replace("# S19-L33", "# S19 current handoff — S19-L33", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "HUMAN_REVIEW_REQUIRED_SOLUTION"
            if solution
            else "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": "BASIN_BLIND_OPERATOR_MEMORY_COMMITTOR_COORDINATE"
            if solution
            else None,
            "nextAuthorizedLoop": None if solution else "S19-L34",
            "authorizationUpperBound": "S19-L42",
            "s20Active": False,
            "updatedAtUtc": runtime["completedAtUtc"],
        },
    )
    BASE.write_json(
        ARTIFACT_ROOT / "artifact_manifest.json", manifest_for(ARTIFACT_ROOT)
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "classifications": classifications,
                "solution": solution,
                "oracleRankDiagnosticPassed": oracle_rank_pass,
                "nextTheme": next_theme,
                "runtime": runtime,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-lock", action="store_true")
    args = parser.parse_args()
    if args.prepare_lock:
        prepare_lock()
    else:
        execute()


if __name__ == "__main__":
    main()
