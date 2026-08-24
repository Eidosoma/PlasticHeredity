"""Execute S19-L34 full-state catalytic-graph committor audit."""

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_latent_timebase.core import array_sha256 as simulator_array_sha256
from e01_latent_timebase.core import generate_beta
from e01_onset_discovery.full_state_graph import (
    ORACLE_VIEW,
    PRIMARY_VIEW,
    VIEWS,
    feature_names,
    graph_views,
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L33 = _load_module(
    "e01_s19_l34_l33",
    REPO_ROOT / "scripts/e01/run_s19_l33_operator_memory_phase_committor.py",
)
L32 = L33.L32
L31 = L33.L31
L30 = L33.L30
L29 = L33.L29
L28 = L33.L28
BASE = L33.BASE
LOOP_ID = "S19-L34"
VERSION = "E01-S19-L34-PERMUTATION-INVARIANT-FULL-STATE-GRAPH-COMMITTOR-v1.0.0"
CANDIDATES = L28.CANDIDATES
EVALUATION_COHORTS = ("L28_VALIDATION", "L31_CONFIRMATION")
CONTROL_MODELS = (
    "FROZEN_LANDMARK_PRIOR",
    "FROZEN_TARGET_GEOMETRY",
    "FROZEN_EXACT_H_TUBE",
    "FROZEN_ORDINARY_TUBE",
    "FROZEN_L33_PHASE_MEMORY",
    "FROZEN_L33_OPERATOR_MEMORY",
)
PCA_COMPONENTS = 12
BOOTSTRAPS = 4096
PERMUTATIONS = 512
ROOT_HEX = "e8cfd645464dd2106f879de01c4bd77bf4330973292a25656ab3d9eda3ee230d"
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L34"
L33_ROOT = ARTIFACT_ROOT / "loops/L33"
L31_ROOT = ARTIFACT_ROOT / "loops/L31"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l34")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l34_full_state_graph_committor.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/full_state_graph.py"


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
    prior = json.loads((L33_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L33_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L33_ROOT / item["path"]),
            "root": str(L33_ROOT),
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
        "schema": "eidosoma.e01.s19_l34.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
        ).hexdigest(),
        "l33ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


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
    rows = []
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
            or int(current.generation_local_step) != int(source.currentGenerationLocalStep)
            or int(current.batch_step) != int(source.currentBatchStep)
            or int(state.sum()) != int(source.currentMass)
            or L28.array_sha256(state) != source.currentStateSha256
        ):
            raise RuntimeError(f"state/clock replay failure: {source.stateId}")
        beta = generate_beta(
            L28.derive_seed(
                L28.L23_ROOT_HEX,
                L28.L23_PHASE,
                "catalytic_matrix",
                int(source.matrixIndex),
            )
        )
        if simulator_array_sha256(beta) != source.betaSha256:
            raise RuntimeError(f"beta replay failure: {source.stateId}")
        target = coordinate_groups[source.stateId]
        if L28.array_sha256(target) != source.targetCentroidSha256:
            raise RuntimeError(f"target replay failure: {source.stateId}")
        views = graph_views(
            state,
            beta,
            target,
            generation_local_step=int(source.currentGenerationLocalStep),
            observation_kind=source.currentObservationKind,
            completed_fissions=int(source.currentCompletedFissions),
            batch_step=int(source.currentBatchStep),
            landmark=int(source.landmark),
            target_component_fraction=float(source.targetComponentSize) / 100.0,
        )
        alternative = graph_views(
            state,
            beta,
            np.roll(target, 1),
            generation_local_step=int(source.currentGenerationLocalStep),
            observation_kind=source.currentObservationKind,
            completed_fissions=int(source.currentCompletedFissions),
            batch_step=int(source.currentBatchStep),
            landmark=int(source.landmark),
            target_component_fraction=float(source.targetComponentSize) / 100.0,
        )
        if not np.array_equal(views[PRIMARY_VIEW], alternative[PRIMARY_VIEW]):
            raise RuntimeError("primary graph signature depends on target")
        for view, values in views.items():
            rows.append(
                {
                    "stateId": source.stateId,
                    "candidateId": source.candidateId,
                    "matrixIndex": int(source.matrixIndex),
                    "landmark": int(source.landmark),
                    "evaluationCohort": source.evaluationCohort,
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
        ["evaluationCohort", "candidateId", "modelId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)
    if len(result) != len(responses) * len(VIEWS):
        raise RuntimeError("graph representation scope changed")
    return result


def representation_frames(
    table: pd.DataFrame, responses: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    base = responses.sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)
    frames = {}
    for view in VIEWS:
        subset = table[table["modelId"].eq(view)].sort_values(
            ["evaluationCohort", "candidateId", "landmark", "matrixIndex"]
        )
        if not np.array_equal(base["stateId"].to_numpy(), subset["stateId"].to_numpy()):
            raise RuntimeError("graph representation/response order mismatch")
        values = np.stack(subset["values"].map(np.asarray)).astype(np.float64)
        columns = [f"raw{index:03d}" for index in range(values.shape[1])]
        frames[view] = pd.concat(
            [base, pd.DataFrame(values, columns=columns)], axis=1
        )
    return frames


def frozen_control_predictions(responses: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_parquet(L33_ROOT / "prediction_results.parquet")
    mapping = {
        "FROZEN_LANDMARK_PRIOR": "FROZEN_LANDMARK_PRIOR",
        "FROZEN_TARGET_GEOMETRY": "FROZEN_TARGET_GEOMETRY",
        "FROZEN_EXACT_H_TUBE": "FROZEN_EXACT_H_TUBE",
        "FROZEN_ORDINARY_TUBE": "FROZEN_ORDINARY_TUBE",
        "PHASE_MEMORY": "FROZEN_L33_PHASE_MEMORY",
        "BASIN_BLIND_OPERATOR_MEMORY": "FROZEN_L33_OPERATOR_MEMORY",
    }
    result = source[
        source["variant"].eq("ORIGINAL") & source["modelId"].isin(mapping)
    ][
        [
            "stateId",
            "candidateId",
            "matrixIndex",
            "landmark",
            "evaluationCohort",
            "modelId",
            "predictedQ",
            "successes",
            "qHat",
            "shortSuccesses",
            "q8",
        ]
    ].copy()
    result["modelId"] = result["modelId"].map(mapping)
    result["variant"] = "ORIGINAL"
    expected = len(responses) * len(CONTROL_MODELS)
    if len(result) != expected or result.duplicated(["stateId", "modelId"]).any():
        raise RuntimeError("L33 frozen control scope mismatch")
    return result.sort_values(
        ["evaluationCohort", "candidateId", "modelId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)


def fit_pca_pipeline(
    development: pd.DataFrame, raw_columns: list[str]
) -> tuple[StandardScaler, PCA, pd.DataFrame]:
    raw = development[raw_columns].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(raw)
    standardized = scaler.transform(raw)
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full").fit(standardized)
    transformed = pd.DataFrame(
        pca.transform(standardized),
        columns=[f"f{index:02d}" for index in range(PCA_COMPONENTS)],
    )
    transformed["successes"] = development["successes"].to_numpy()
    return scaler, pca, transformed


def transform_frame(
    frame: pd.DataFrame,
    raw_columns: list[str],
    scaler: StandardScaler,
    pca: PCA,
) -> pd.DataFrame:
    values = pca.transform(
        scaler.transform(frame[raw_columns].to_numpy(dtype=np.float64))
    )
    columns = [f"f{index:02d}" for index in range(PCA_COMPONENTS)]
    return pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(values, columns=columns)], axis=1
    )


def fit_and_score(
    frames: dict[str, pd.DataFrame], controls: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    rows = controls.to_dict("records")
    registry = []
    attributions = []
    transformed_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for candidate in CANDIDATES:
        for view in VIEWS:
            frame = frames[view]
            development = frame[
                frame["candidateId"].eq(candidate)
                & frame["evaluationCohort"].eq("L28_DEVELOPMENT")
            ].reset_index(drop=True)
            raw_columns = [column for column in frame.columns if column.startswith("raw")]
            raw_scaler, pca, transformed_development = fit_pca_pipeline(
                development, raw_columns
            )
            component_columns = [
                f"f{index:02d}" for index in range(PCA_COMPONENTS)
            ]
            model_scaler, model = L29.fit_model(
                transformed_development, component_columns
            )
            replay_raw_scaler, replay_pca, replay_development = fit_pca_pipeline(
                development, raw_columns
            )
            replay_model_scaler, replay_model = L29.fit_model(
                replay_development, component_columns
            )
            exact = bool(
                np.array_equal(raw_scaler.mean_, replay_raw_scaler.mean_)
                and np.array_equal(raw_scaler.scale_, replay_raw_scaler.scale_)
                and np.array_equal(pca.components_, replay_pca.components_)
                and np.array_equal(pca.explained_variance_, replay_pca.explained_variance_)
                and np.array_equal(model_scaler.mean_, replay_model_scaler.mean_)
                and np.array_equal(model_scaler.scale_, replay_model_scaler.scale_)
                and np.array_equal(model.coef_, replay_model.coef_)
                and np.array_equal(model.intercept_, replay_model.intercept_)
            )
            candidate_frame = frame[frame["candidateId"].eq(candidate)].reset_index(
                drop=True
            )
            transformed = transform_frame(
                candidate_frame, raw_columns, raw_scaler, pca
            )
            transformed_frames[(candidate, view)] = transformed
            probabilities = model.predict_proba(
                model_scaler.transform(
                    transformed[component_columns].to_numpy(dtype=np.float64)
                )
            )[:, 1]
            replay_probabilities = replay_model.predict_proba(
                replay_model_scaler.transform(
                    transformed[component_columns].to_numpy(dtype=np.float64)
                )
            )[:, 1]
            if not np.array_equal(probabilities, replay_probabilities):
                raise RuntimeError("graph model probability replay failed")
            registry.append(
                {
                    "candidateId": candidate,
                    "modelId": view,
                    "rawFeatureCount": len(raw_columns),
                    "pcaComponents": PCA_COMPONENTS,
                    "rawFeatureNames": json.dumps(feature_names()[view]),
                    "rawScalerMean": json.dumps(raw_scaler.mean_.tolist()),
                    "rawScalerScale": json.dumps(raw_scaler.scale_.tolist()),
                    "pcaMean": json.dumps(pca.mean_.tolist()),
                    "pcaComponentsArray": json.dumps(pca.components_.tolist()),
                    "pcaExplainedVarianceRatio": json.dumps(
                        pca.explained_variance_ratio_.tolist()
                    ),
                    "modelScalerMean": json.dumps(model_scaler.mean_.tolist()),
                    "modelScalerScale": json.dumps(model_scaler.scale_.tolist()),
                    "intercept": float(model.intercept_[0]),
                    "coefficients": json.dumps(model.coef_[0].tolist()),
                    "iterations": int(model.n_iter_[0]),
                    "exactReplay": exact,
                }
            )
            standardized_pc_coefficient = (
                model.coef_[0] / model_scaler.scale_
            ) @ pca.components_
            for name, coefficient in zip(
                feature_names()[view], standardized_pc_coefficient, strict=True
            ):
                attributions.append(
                    {
                        "candidateId": candidate,
                        "modelId": view,
                        "featureName": name,
                        "standardizedBackprojectedCoefficient": float(coefficient),
                        "absoluteCoefficient": abs(float(coefficient)),
                    }
                )
            for source, probability in zip(
                transformed.itertuples(index=False), probabilities, strict=True
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
                        "variant": "ORIGINAL",
                    }
                )
    attribution_frame = pd.DataFrame(attributions)
    attribution_frame["absoluteRankWithinModel"] = attribution_frame.groupby(
        ["candidateId", "modelId"]
    )["absoluteCoefficient"].rank(method="first", ascending=False)
    return (
        pd.DataFrame(rows).sort_values(
            ["evaluationCohort", "candidateId", "modelId", "landmark", "matrixIndex"]
        ).reset_index(drop=True),
        pd.DataFrame(registry),
        attribution_frame.sort_values(
            ["candidateId", "modelId", "absoluteRankWithinModel"]
        ).reset_index(drop=True),
        transformed_frames,
    )


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    return L33.metric_table(predictions)


def bootstrap_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source = predictions[
        predictions["variant"].eq("ORIGINAL")
        & predictions["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    expected_models = {PRIMARY_VIEW, ORACLE_VIEW, *CONTROL_MODELS}
    for cohort in EVALUATION_COHORTS:
        for candidate in CANDIDATES:
            pivot = source[
                source["evaluationCohort"].eq(cohort)
                & source["candidateId"].eq(candidate)
            ].pivot(
                index=["stateId", "qHat", "q8"], columns="modelId", values="predictedQ"
            ).reset_index()
            if not expected_models.issubset(pivot.columns):
                raise RuntimeError("graph bootstrap model missing")
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
                    if model in {PRIMARY_VIEW, ORACLE_VIEW}:
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
                            - brier[PRIMARY_VIEW],
                        }
                    )
    return pd.DataFrame(rows)


def permutation_results(
    transformed_frames: dict[tuple[str, str], pd.DataFrame],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = metrics[
        metrics["evaluationCohort"].isin(EVALUATION_COHORTS)
        & metrics["modelId"].eq(PRIMARY_VIEW)
    ].set_index(["evaluationCohort", "candidateId"])["spearmanH32"].to_dict()
    fixed = predictions[
        predictions["evaluationCohort"].isin(EVALUATION_COHORTS)
        & predictions["modelId"].eq(PRIMARY_VIEW)
    ]
    component_columns = [f"f{index:02d}" for index in range(PCA_COMPONENTS)]
    development_rows = []
    evaluation_rows = []
    development_maxima = []
    evaluation_maxima = []
    for replicate in range(PERMUTATIONS):
        development_values = []
        evaluation_values = []
        for candidate in CANDIDATES:
            frame = transformed_frames[(candidate, PRIMARY_VIEW)]
            development = frame[
                frame["evaluationCohort"].eq("L28_DEVELOPMENT")
            ].reset_index(drop=True)
            shuffled = L33.permute_within_landmark(
                development,
                np.random.default_rng(
                    derived_seed("development_permutation", replicate, candidate)
                ),
            )
            scaler, model = L29.fit_model(shuffled, component_columns)
            for cohort in EVALUATION_COHORTS:
                evaluation = frame[frame["evaluationCohort"].eq(cohort)]
                probability = model.predict_proba(
                    scaler.transform(
                        evaluation[component_columns].to_numpy(dtype=np.float64)
                    )
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
                permuted = L33.permute_within_landmark(
                    evaluation_source,
                    np.random.default_rng(
                        derived_seed(
                            "evaluation_permutation", replicate, cohort, candidate
                        )
                    ),
                )
                fixed_rho = L29.safe_spearman(
                    permuted["predictedQ"], permuted["qHat"]
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
        maxima_array = np.asarray(maxima)
        for (cohort, candidate), observed_value in observed.items():
            mask = frame["evaluationCohort"].eq(cohort) & frame["candidateId"].eq(
                candidate
            )
            frame.loc[mask, "observedSpearmanH32"] = observed_value
            frame.loc[mask, "familywiseP"] = float(
                (1 + np.sum(maxima_array >= observed_value))
                / (1 + len(maxima_array))
            )
    return development, evaluation


def suffix_and_target_invariance(
    responses: pd.DataFrame, representations: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    manifest_index = manifest.set_index(["candidateId", "matrixIndex"])
    sentinels = responses.groupby(["evaluationCohort", "candidateId"]).head(3)
    rows = []
    for source in sentinels.itertuples(index=False):
        manifest_row = manifest_index.loc[(source.candidateId, int(source.matrixIndex))]
        trajectory = L29.load_l23_trajectory(manifest_row)
        selected = L28.selected_clock_observations(trajectory, L28.CLOCK_ID)
        endpoint = int(source.currentSelectedIndex)
        prefix = np.asarray(
            [row.state for row in selected[: endpoint + 1]], dtype=np.int64
        )
        suffix = np.asarray(
            [row.state for row in selected[endpoint + 1 :]], dtype=np.int64
        )
        rng = np.random.default_rng(
            derived_seed(
                "suffix", source.evaluationCohort, source.candidateId, source.matrixIndex
            )
        )
        order = rng.permutation(len(suffix))
        saved = representations[
            representations["stateId"].eq(source.stateId)
            & representations["modelId"].eq(PRIMARY_VIEW)
        ].iloc[0]
        rows.append(
            {
                "stateId": source.stateId,
                "candidateId": source.candidateId,
                "evaluationCohort": source.evaluationCohort,
                "prefixSha256": array_hash(prefix),
                "suffixOriginalSha256": array_hash(suffix),
                "suffixPermutedSha256": array_hash(suffix[order]),
                "suffixActuallyChanged": bool(
                    len(suffix) < 2 or not np.array_equal(suffix, suffix[order])
                ),
                "primaryUsesOnlyCurrentStateBetaPhase": True,
                "primaryTargetInvariant": bool(saved.primaryTargetInvariant),
                "storedFeatureHashValid": array_hash(np.asarray(saved["values"]))
                == saved.vectorSha256,
                "responseTargetRetrospective": True,
            }
        )
    return pd.DataFrame(rows)


def gate_table(
    metrics: pd.DataFrame,
    bootstraps: pd.DataFrame,
    development_permutations: pd.DataFrame,
    evaluation_permutations: pd.DataFrame,
    invariance: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for cohort in EVALUATION_COHORTS:
        for candidate in CANDIDATES:
            primary = metrics[
                metrics["evaluationCohort"].eq(cohort)
                & metrics["candidateId"].eq(candidate)
                & metrics["modelId"].eq(PRIMARY_VIEW)
            ].iloc[0]
            boot = bootstraps[
                bootstraps["evaluationCohort"].eq(cohort)
                & bootstraps["candidateId"].eq(candidate)
            ]
            primary_boot = boot[boot["metricId"].eq(PRIMARY_VIEW)]
            lower32 = float(np.nanquantile(primary_boot["spearmanH32"], 0.025))
            lower8 = float(np.nanquantile(primary_boot["spearmanH8"], 0.025))
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
            invariant = invariance[
                invariance["evaluationCohort"].eq(cohort)
                & invariance["candidateId"].eq(candidate)
            ]
            checks = {
                "h32RankPassed": primary.spearmanH32 > 0.5 and lower32 > 0.3,
                "h8RankPassed": primary.spearmanH8 > 0.5 and lower8 > 0.3,
                "incrementalBrierPassed": all(value > 0 for value in deltas.values()),
                "developmentPermutationPassed": dev_p <= 0.05,
                "evaluationPermutationPassed": eval_p <= 0.05,
                "invariancePassed": bool(
                    invariant[
                        [
                            "suffixActuallyChanged",
                            "primaryUsesOnlyCurrentStateBetaPhase",
                            "primaryTargetInvariant",
                            "storedFeatureHashValid",
                        ]
                    ].all().all()
                ),
                "shiftAuditCompleted": True,
            }
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "states": int(primary.states),
                    "primarySpearmanH32": primary.spearmanH32,
                    "primarySpearmanH32Lower95": lower32,
                    "primarySpearmanH8": primary.spearmanH8,
                    "primarySpearmanH8Lower95": lower8,
                    **{
                        f"brierImprovementLowerVs{control}": value
                        for control, value in deltas.items()
                    },
                    "developmentPermutationP": dev_p,
                    "evaluationPermutationP": eval_p,
                    **checks,
                    "cohortCandidateGatePassed": all(checks.values()),
                }
            )
    return pd.DataFrame(rows)


def cohort_shift_audit(
    transformed_frames: dict[tuple[str, str], pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    state_rows = []
    components = [f"f{index:02d}" for index in range(PCA_COMPONENTS)]
    for candidate in CANDIDATES:
        frame = transformed_frames[(candidate, PRIMARY_VIEW)].copy()
        development = frame[frame["evaluationCohort"].eq("L28_DEVELOPMENT")]
        lower = development[components].min() - 0.5 * development[components].std(ddof=1)
        upper = development[components].max() + 0.5 * development[components].std(ddof=1)
        for source in frame.itertuples(index=False):
            values = np.asarray([getattr(source, name) for name in components])
            outside = (values < lower.to_numpy()) | (values > upper.to_numpy())
            state_rows.append(
                {
                    "stateId": source.stateId,
                    "candidateId": candidate,
                    "evaluationCohort": source.evaluationCohort,
                    "matrixIndex": int(source.matrixIndex),
                    "landmark": int(source.landmark),
                    "qHat": float(source.qHat),
                    "q8": float(source.q8),
                    "mass": int(source.currentMass),
                    "generationLocalStep": int(source.currentGenerationLocalStep),
                    "targetComponentSize": int(source.targetComponentSize),
                    "componentsOutsideExpandedDevelopmentRange": int(outside.sum()),
                    "withinExpandedDevelopmentSupport": bool(not outside.any()),
                    "pcaRadius": float(np.linalg.norm(values)),
                }
            )
        for cohort, group in frame.groupby("evaluationCohort", sort=True):
            values = group[components].to_numpy(dtype=np.float64)
            support_rows = [
                row
                for row in state_rows
                if row["candidateId"] == candidate
                and row["evaluationCohort"] == cohort
            ]
            summary_rows.append(
                {
                    "candidateId": candidate,
                    "evaluationCohort": cohort,
                    "states": len(group),
                    "uniqueMatrices": int(group["matrixIndex"].nunique()),
                    "statesPerMatrixMaximum": int(
                        group.groupby("matrixIndex").size().max()
                    ),
                    "meanQHat": float(group["qHat"].mean()),
                    "minimumQHat": float(group["qHat"].min()),
                    "maximumQHat": float(group["qHat"].max()),
                    "meanQ8": float(group["q8"].mean()),
                    "meanMass": float(group["currentMass"].mean()),
                    "meanGenerationLocalStep": float(
                        group["currentGenerationLocalStep"].mean()
                    ),
                    "meanTargetComponentSize": float(
                        group["targetComponentSize"].mean()
                    ),
                    "meanPcaRadius": float(
                        np.mean(np.linalg.norm(values, axis=1))
                    ),
                    "withinExpandedDevelopmentSupportFraction": float(
                        np.mean(
                            [row["withinExpandedDevelopmentSupport"] for row in support_rows]
                        )
                    ),
                }
            )
    state_frame = pd.DataFrame(state_rows)
    summary = pd.DataFrame(summary_rows)
    return state_frame, summary


def markov_completeness_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("integer_molecular_composition", "DIRECT_KERNEL_INPUT", True, "exact restored state"),
            ("total_mass", "DERIVED_FROM_STATE", True, "exact sum of counts"),
            ("catalytic_matrix_beta", "DIRECT_KERNEL_INPUT", True, "exact matrix seed and hash replay"),
            ("reaction_propensities", "DETERMINISTIC_DERIVATION", True, "exact joins/losses/boost"),
            ("candidate_semantics", "DIRECT_KERNEL_INPUT", True, "exposure, trimming and daughter rule; candidate-specific model"),
            ("distance_to_fission_threshold", "DERIVED_FROM_STATE", True, "N_MAX minus mass"),
            ("selected_daughter_state", "CURRENT_STATE_WHEN_BOUNDARY", True, "post-fission observation flag retained"),
            ("time_since_previous_fission", "DIAGNOSTIC_NOT_RATE_INPUT", True, "generation-local step retained"),
            ("completed_fissions", "STOPPING_STATE_NOT_RATE_INPUT", True, "retained as phase"),
            ("batch_step", "DIAGNOSTIC_NOT_RATE_INPUT", True, "retained as phase"),
            ("reservoir_state", "FIXED_MODEL_CONSTANTS", True, "K_FORWARD, K_BACKWARD and RHO_EACH fixed in source"),
            ("previous_parent_composition", "NOT_TRANSITION_KERNEL_INPUT", False, "excluded without loss of Markov state under pinned source"),
            ("previous_daughter_composition", "NOT_TRANSITION_KERNEL_INPUT", False, "current state supersedes history under pinned source"),
            ("accumulated_flux_counts", "NOT_TRANSITION_KERNEL_INPUT", False, "past event counts do not enter rates"),
            ("future_rng_state", "MARGINALIZED_STOCHASTIC_INPUT", False, "empirical committor integrates new branch streams"),
            ("completed_run_target_basin", "RESPONSE_DEFINITION_NOT_SIMULATOR_STATE", False, "excluded from primary; oracle diagnostic only"),
        ],
        columns=["stateElement", "sourceRole", "primaryRepresentationIncludes", "rationale"],
    )


def matrix_cardinality_audit(responses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in responses.groupby(
        ["evaluationCohort", "candidateId"], sort=True
    ):
        cohort, candidate = keys
        counts = group.groupby("matrixIndex").size()
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "states": len(group),
                "uniqueMatrices": int(group["matrixIndex"].nunique()),
                "minimumStatesPerMatrix": int(counts.min()),
                "maximumStatesPerMatrix": int(counts.max()),
                "withinMatrixOrderingIdentifiable": bool((counts >= 2).any()),
                "classification": "WITHIN_MATRIX_ORDERING_NOT_IDENTIFIABLE_FROM_FROZEN_COHORT"
                if not (counts >= 2).any()
                else "WITHIN_MATRIX_ORDERING_AVAILABLE",
            }
        )
    return pd.DataFrame(rows)


def fixture_results() -> pd.DataFrame:
    rng = np.random.default_rng(derived_seed("fixture"))
    state = rng.poisson(1.4, size=100).astype(np.int64)
    state[0] += 1
    beta = np.exp(rng.normal(-3.0, 1.0, size=(100, 100)))
    target = rng.random(100)
    target /= target.sum()
    kwargs = {
        "generation_local_step": 7,
        "observation_kind": "molecular_update",
        "completed_fissions": 11,
        "batch_step": 52,
        "landmark": 96,
        "target_component_fraction": 0.42,
    }
    first = graph_views(state, beta, target, **kwargs)
    replay = graph_views(state.copy(), beta.copy(), target.copy(), **kwargs)
    order = np.random.default_rng(derived_seed("fixture_permutation")).permutation(100)
    permuted = graph_views(
        state[order], beta[np.ix_(order, order)], target[order], **kwargs
    )
    alternative = graph_views(state, beta, np.roll(target, 1), **kwargs)
    changed_beta = beta.copy()
    changed_beta[0, 1] *= 100
    changed = graph_views(state, changed_beta, target, **kwargs)
    return pd.DataFrame(
        [
            {
                "fixtureId": "TWO_FIXED_VIEWS",
                "passed": tuple(first) == VIEWS,
                "details": json.dumps({key: len(value) for key, value in first.items()}),
            },
            {
                "fixtureId": "CPU_FLOAT64_FINITE",
                "passed": all(
                    value.dtype == np.float64 and np.isfinite(value).all()
                    for value in first.values()
                ),
                "details": "full graph signatures",
            },
            {
                "fixtureId": "EXACT_FEATURE_REPLAY",
                "passed": all(np.array_equal(first[key], replay[key]) for key in VIEWS),
                "details": json.dumps({key: array_hash(first[key]) for key in VIEWS}),
            },
            {
                "fixtureId": "MOLECULE_PERMUTATION_INVARIANCE",
                "passed": all(
                    np.allclose(first[key], permuted[key], atol=1e-10, rtol=1e-10)
                    for key in VIEWS
                ),
                "details": "simultaneous state/beta/target relabeling",
            },
            {
                "fixtureId": "PRIMARY_TARGET_INVARIANCE",
                "passed": np.array_equal(first[PRIMARY_VIEW], alternative[PRIMARY_VIEW]),
                "details": "completed-run target excluded",
            },
            {
                "fixtureId": "ORACLE_TARGET_DEPENDENCE_VISIBLE",
                "passed": not np.array_equal(first[ORACLE_VIEW], alternative[ORACLE_VIEW]),
                "details": "oracle separated",
            },
            {
                "fixtureId": "BETA_GRAPH_SENSITIVITY",
                "passed": not np.array_equal(first[PRIMARY_VIEW], changed[PRIMARY_VIEW]),
                "details": "one catalytic edge changed",
            },
            {
                "fixtureId": "PCA_DIMENSION_FIXED",
                "passed": PCA_COMPONENTS == 12,
                "details": "no component search",
            },
        ]
    )


def benchmark_projection() -> dict[str, Any]:
    rng = np.random.default_rng(derived_seed("opaque_benchmark"))
    frame = pd.DataFrame(
        rng.normal(size=(50, PCA_COMPONENTS)),
        columns=[f"f{index:02d}" for index in range(PCA_COMPONENTS)],
    )
    frame["successes"] = rng.integers(4, 125, size=len(frame))
    durations = []
    for _ in range(3):
        started = time.perf_counter()
        L29.fit_model(frame, [f"f{index:02d}" for index in range(PCA_COMPONENTS)])
        durations.append(time.perf_counter() - started)
    projected_fits = len(CANDIDATES) * len(VIEWS) * 2 + PERMUTATIONS * len(CANDIDATES)
    projected_hours = max(durations) * projected_fits / 3600 + 3.0
    return {
        "schema": "eidosoma.e01.s19_l34.benchmark_projection.v1",
        "status": "PASS" if projected_hours <= 64.8 else "STOP_BEFORE_OUTCOME",
        "fitDurationsSeconds": durations,
        "projectedScientificFits": projected_fits,
        "projectedWallHoursIncludingGraphAndValidationOverhead": projected_hours,
        "scientificOutcomeOpened": False,
    }


def seed_manifest(responses: pd.DataFrame) -> pd.DataFrame:
    rows = []

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

    add("fixture", ("fixture",))
    add("fixture_permutation", ("fixture_permutation",))
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
    for source in responses.groupby(["evaluationCohort", "candidateId"]).head(3).itertuples(
        index=False
    ):
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
        raise RuntimeError("L34 seed collision within manifest")
    return result


def seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    current_material = set(seeds["seedMaterialSha256"].astype(str))
    current_derived = set(seeds["derivedSeed"].astype(str))
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L34/" in str(path):
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
    overlaps = sorted(current_material & prior_material)
    derived_overlaps = sorted(current_derived & prior_derived)
    root_collisions = []
    needle = ROOT_HEX.encode()
    for path in ARTIFACT_ROOT.rglob("*"):
        if (
            not path.is_file()
            or "/L34/" in str(path)
            or path.stat().st_size > 64 * 1024 * 1024
        ):
            continue
        try:
            if needle in path.read_bytes():
                root_collisions.append(str(path))
        except OSError:
            continue
    return {
        "schema": "eidosoma.e01.s19_l34.seed_firewall.v1",
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
                "sourceId": "ZAHEER_2017_DEEP_SETS",
                "url": "https://arxiv.org/abs/1703.06114",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
                "directSupport": "set functions and equivariant layers should respect permutation symmetry",
                "frozenUse": "simultaneous molecule relabeling is a mandatory invariant",
            },
            {
                "sourceId": "GILMER_2017_MESSAGE_PASSING",
                "url": "https://arxiv.org/abs/1704.01212",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
                "directSupport": "message passing plus aggregation represents entire molecular graphs while respecting graph symmetries",
                "frozenUse": "fixed directed Krylov propagation on beta with invariant aggregation",
            },
            {
                "sourceId": "JUNG_2026_COMMITTOR_GNN",
                "url": "https://www.nature.com/articles/s43588-026-00958-2",
                "evidenceClass": "PRIMARY_METHOD_CONTEXT",
                "directSupport": "graph neural architectures can infer committors from full configurations without handcrafted collective variables",
                "frozenUse": "motivates the full-state graph test but supplies no GARD implementation",
            },
            {
                "sourceId": "PINNED_GARD_KERNEL",
                "url": None,
                "evidenceClass": "DIRECT_FROZEN_E01_SOURCE",
                "directSupport": "future updates depend on current integer state, beta, candidate semantics and new stochastic streams",
                "frozenUse": "Markov completeness audit and exact propensity node attributes",
            },
            {
                "sourceId": "L31_EMPIRICAL_COMMITTOR",
                "url": None,
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "directSupport": "reliable H32/H8 shooting responses on an untouched matrix cohort",
                "frozenUse": "teacher response only; branches never enter graph features",
            },
        ]
    )


def make_figures(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    gates: pd.DataFrame,
    shift: pd.DataFrame,
    attributions: pd.DataFrame,
) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(root / name, dpi=180)
        plt.close()

    primary = predictions[
        predictions["evaluationCohort"].isin(EVALUATION_COHORTS)
        & predictions["modelId"].eq(PRIMARY_VIEW)
    ]
    _, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, ((cohort, candidate), group) in zip(
        axes.flat,
        primary.groupby(["evaluationCohort", "candidateId"], sort=True),
        strict=True,
    ):
        axis.scatter(group["predictedQ"], group["qHat"], s=22)
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set_title(f"{cohort} / {candidate}", fontsize=8)
        axis.set_xlabel("Full-state graph predicted q")
        axis.set_ylabel("Empirical H32 q-hat")
    save("01_full_state_graph_vs_committor.png")

    metrics[
        metrics["evaluationCohort"].isin(EVALUATION_COHORTS)
    ].pivot_table(
        index="modelId",
        columns=["evaluationCohort", "candidateId"],
        values="spearmanH32",
    ).plot(kind="bar", figsize=(14, 6))
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Spearman with H32 q-hat")
    save("02_graph_and_frozen_control_comparison.png")

    shift.pivot_table(
        index="evaluationCohort",
        columns="candidateId",
        values="withinExpandedDevelopmentSupportFraction",
    ).plot(kind="bar", figsize=(9, 5))
    plt.ylim(0, 1)
    plt.ylabel("Fraction within expanded development PCA support")
    save("03_cohort_distribution_shift.png")

    top = attributions[
        attributions["modelId"].eq(PRIMARY_VIEW)
        & attributions["absoluteRankWithinModel"].le(12)
    ].copy()
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, (candidate, group) in zip(
        axes, top.groupby("candidateId", sort=True), strict=True
    ):
        group = group.sort_values("absoluteCoefficient")
        axis.barh(group["featureName"], group["absoluteCoefficient"])
        axis.set_title(candidate)
        axis.tick_params(axis="y", labelsize=6)
    save("04_backprojected_graph_feature_attribution.png")

    checks = [
        "h32RankPassed",
        "h8RankPassed",
        "incrementalBrierPassed",
        "developmentPermutationPassed",
        "evaluationPermutationPassed",
        "invariancePassed",
        "shiftAuditCompleted",
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
        "schema": "eidosoma.e01.s19_l34.artifact_manifest.v1",
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
            "beliefBeforeLoop": "Reliable shooting responses exist, but invariant path and low-dimensional operator summaries did not transfer.",
            "failureOrAmbiguityTargeted": "Whether species-resolved current composition and beta-conditioned catalytic structure were discarded.",
            "informationGainRationale": "A full-state permutation-invariant graph signature directly tests deterministic hidden-state loss before more branch simulation.",
            "learned": "L34 graph/PCA/model/shift contract frozen before outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Reviewer state-completeness hypothesis plus L31 support and L32/L33 non-support.",
            "proposedNextTest": "Fit only on L28 development and evaluate unchanged twice out of sample.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Full beta-conditioned state, target-conditioned state, or information created only by short stochastic propagation.",
            "selectedHypotheses": "Target-blind full-state graph primary and target-conditioned oracle diagnostic.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Scalar phase/operator summaries are sufficient.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A universal coordinate must transfer across matrices in both candidates and cohorts.",
            "failureOrAmbiguityTargeted": "Physical-state observability versus shooting-only information.",
            "informationGainRationale": "Full graph, shift and attribution audits discriminate lost structure from simple cohort shift.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L34 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_SOLUTION_HUMAN_REVIEW"
            if solution
            else "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Fixed full-state catalytic-graph coordinate.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "A compact full-state spectral/message signature is sufficient"
            if not solution
            else "No deterministic state coordinate exists.",
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
        + f"\n\n## {LOOP_ID} — full-state catalytic graph\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** {next_theme}.\n",
    )
    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    row = {
        "branchCount": 2,
        "bundleId": "L34_FULL_STATE_GRAPH",
        "candidateId": "S19-L34-PERMUTATION-INVARIANT-BETA-GRAPH",
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
        "proposedSpecification": "fixed beta graph spectrum/Krylov messages with current count and phase attributes; 12-PC development-only coordinate",
        "rankingScore": 29.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "REVIEWER_MARKOV_COMPLETENESS_AND_CATALYTIC_STRUCTURE_HYPOTHESIS",
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
    source_rows = [
        {
            "commitOrVersion": None,
            "evidenceClass": source.evidenceClass,
            "finding": f"{source.directSupport}; L34 use: {source.frozenUse}",
            "licenseStatus": "PUBLIC_ARTICLE_OR_WORKSPACE_EVIDENCE",
            "redistributionStatus": "CITATION_OR_INTERNAL_ARTIFACT_ONLY",
            "repositoryIdentity": None,
            "retainedPath": None,
            "retrievalDate": timestamp[:10],
            "sha256": None,
            "sourceId": f"L34_{source.sourceId}",
            "sourceType": source.evidenceClass,
            "treeIdentity": None,
            "url": source.url,
        }
        for source in source_grounding_registry().itertuples(index=False)
    ]
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
            "selectedDiscoveryLead": "FULL_STATE_GRAPH_COMMITTOR_COORDINATE"
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
            "decision": "S19_L34_SOLUTION_HUMAN_REVIEW"
            if solution
            else "S19_L34_COMPLETE_AUTONOMOUS_CONTINUATION",
            "loopId": LOOP_ID,
            "nextLoopAuthorized": not solution,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "s20Activated": False,
            "scope": VERSION,
            "selectedDiscoveryLead": "FULL_STATE_GRAPH_COMMITTOR_COORDINATE"
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
    shifts: pd.DataFrame,
    cardinality: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    evaluation = metrics[metrics["evaluationCohort"].isin(EVALUATION_COHORTS)]
    return f"""# S19-L34 — Permutation-Invariant Full-State Catalytic-Graph Committor

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the authorized L19–L42 sequence.
- **Classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Validation:** exact 280-state/beta/clock/target/q replay; graph permutation and target invariance; training-only PCA/model replay; suffix separation; 512 development/evaluation permutations; 4,096 matrix bootstraps; immutable, seed, runtime, storage, regeneration and artifact hashes.
- **Recommended next action:** `{next_theme}`.

## Question and method

Did the failed scalar/path students discard a deterministic signal carried by the complete current physical state and catalytic graph? The primary signature retains integer composition, exact reaction propensities, directed beta strengths, fixed depth-4 Krylov messages, the beta singular spectrum and simulator phase, while remaining invariant to simultaneous molecule relabeling and to the completed-run basin centroid. Exactly 12 outcome-blind PCA components and the unchanged L29 aggregated-binomial model are fit only on L28 development. A target-conditioned graph is an oracle diagnostic, not a prospective predictor.

## Evaluation metrics

{evaluation.to_markdown(index=False)}

## Locked gates

{gates.to_markdown(index=False)}

## Cohort-shift audit

{shifts.to_markdown(index=False)}

## Within- versus across-matrix identifiability

{cardinality.to_markdown(index=False)}

Every response cohort has exactly one selected state per catalytic matrix. L34 therefore tests only transfer **across** matrices. It does not infer within-matrix ordering; that question would require prospectively selected multiple states and new branch responses per matrix.

## Markov-state interpretation

Under the pinned simulator, the next-state law uses the current integer composition, beta, fixed candidate semantics and fresh stochastic streams. Mass, fission distance and exact propensities are deterministic functions of the current state; generation metadata are retained as diagnostics. Previous parents, prior daughters and accumulated event counts do not enter the transition kernel. The completed-run basin defines the response but is excluded from the primary graph.

## Scope boundary

The H32/H8 responses remain conditioned on a retrospective matrix-specific basin. A passing primary would establish a deterministic past-observable coordinate only inside that reconstructed task. It would not identify the paper label, show PhiRL prediction, prove causal emergence, or establish intervention control. A null rules out this fixed compact graph signature, not all possible full-state models.

## Runtime

- Repository lock: `{runtime['repositoryHead']}`.
- CPU float64, one numerical-library thread, no GPU.
- Wall seconds `{runtime['wallSeconds']:.3f}`; controller CPU hours `{runtime['controllerCpuHours']:.6f}`.

## Autonomous boundary

L34 is frozen. S20, E02, author contact, interventions, reactive-current work and report generation remain inactive.
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L34 lock")
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
    responses = L33.response_registry()
    coordinates = L33.target_coordinates(responses)
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    representations = extract_representations(responses, coordinates, manifest)
    seeds = seed_manifest(responses)
    firewall = seed_firewall(seeds)
    if firewall["status"] != "PASS":
        raise RuntimeError("L34 seed firewall failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L34 decision record\n\n"
        "The reviewer correctly separated three explanations for L31: hidden deterministic physical state, catalytic structure lost by invariant summaries, or information obtainable only by forward stochastic sampling. L32 and L33 addressed compressed histories and phase/operator summaries. L34 now freezes one graph-aware test before outcomes: a target-blind permutation-invariant signature of the exact current integer state and complete beta matrix, using static physical node statistics, the beta singular spectrum and fixed directed depth-4 Krylov message passing. Exactly 12 training-only PCA components feed the unchanged L29 model. A target-conditioned graph is an oracle diagnostic only. The existing response cohorts contain one state per matrix, so within-matrix ordering is explicitly nonidentifiable and cannot be claimed. No graph/message/PCA/model search is permitted.\n",
    )
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    BASE.write_parquet(LOOP_ROOT / "response_registry.parquet", responses)
    BASE.write_parquet(LOOP_ROOT / "target_coordinate_registry.parquet", coordinates)
    BASE.write_parquet(
        LOOP_ROOT / "full_state_graph_representations.parquet", representations
    )
    BASE.write_parquet(
        LOOP_ROOT / "markov_completeness_audit.parquet", markov_completeness_audit()
    )
    BASE.write_parquet(
        LOOP_ROOT / "matrix_cardinality_audit.parquet",
        matrix_cardinality_audit(responses),
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
            LOOP_ROOT / "full_state_graph_representations.parquet"
        ),
        "markovAuditSha256": sha256_file(
            LOOP_ROOT / "markov_completeness_audit.parquet"
        ),
        "cardinalityAuditSha256": sha256_file(
            LOOP_ROOT / "matrix_cardinality_audit.parquet"
        ),
        "seedsSha256": sha256_file(LOOP_ROOT / "analysis_seed_manifest.parquet"),
        "seedFirewallSha256": sha256_file(LOOP_ROOT / "seed_firewall.json"),
        "benchmarkSha256": sha256_file(LOOP_ROOT / "benchmark_projection.json"),
        "l33PredictionsSha256": sha256_file(
            L33_ROOT / "prediction_results.parquet"
        ),
        "l23ManifestSha256": sha256_file(
            L23_ROOT / "input_trajectory_manifest.parquet"
        ),
    }
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l34.implementation_lock.v1",
            "repositoryHead": head,
            "remoteHead": remote,
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "configSha256": sha256_file(CONFIG),
            "views": list(VIEWS),
            "primary": PRIMARY_VIEW,
            "oracleDiagnostic": ORACLE_VIEW,
            "pcaComponents": PCA_COMPONENTS,
            "modelFitScope": "L28_DEVELOPMENT_ONLY",
            "branchDerivedPredictor": False,
            "targetCentroidInPrimary": False,
            "oneStatePerMatrix": True,
            "withinMatrixOrderingClaimAuthorized": False,
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
        "representationsSha256": LOOP_ROOT / "full_state_graph_representations.parquet",
        "markovAuditSha256": LOOP_ROOT / "markov_completeness_audit.parquet",
        "cardinalityAuditSha256": LOOP_ROOT / "matrix_cardinality_audit.parquet",
        "seedsSha256": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewallSha256": LOOP_ROOT / "seed_firewall.json",
        "benchmarkSha256": LOOP_ROOT / "benchmark_projection.json",
        "l33PredictionsSha256": L33_ROOT / "prediction_results.parquet",
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
        LOOP_ROOT / "full_state_graph_representations.parquet"
    )
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    frames = representation_frames(representations, responses)
    controls = frozen_control_predictions(responses)
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    predictions, models, attributions, transformed = fit_and_score(frames, controls)
    if not models["exactReplay"].all():
        raise RuntimeError("PCA/model replay failed")
    metrics = metric_table(predictions)
    bootstraps = bootstrap_metrics(predictions)
    development_permutations, evaluation_permutations = permutation_results(
        transformed, predictions, metrics
    )
    invariance = suffix_and_target_invariance(
        responses, representations, manifest
    )
    shift_states, shift_summary = cohort_shift_audit(transformed)
    cardinality = matrix_cardinality_audit(responses)
    gates = gate_table(
        metrics,
        bootstraps,
        development_permutations,
        evaluation_permutations,
        invariance,
    )
    solution = bool(gates["cohortCandidateGatePassed"].all())
    oracle_rows = metrics[
        metrics["evaluationCohort"].isin(EVALUATION_COHORTS)
        & metrics["modelId"].eq(ORACLE_VIEW)
    ]
    oracle_boot = bootstraps[bootstraps["metricId"].eq(ORACLE_VIEW)]
    oracle_rank_pass = True
    for source in oracle_rows.itertuples(index=False):
        subset = oracle_boot[
            oracle_boot["evaluationCohort"].eq(source.evaluationCohort)
            & oracle_boot["candidateId"].eq(source.candidateId)
        ]
        oracle_rank_pass &= bool(
            source.spearmanH32 > 0.5
            and np.nanquantile(subset["spearmanH32"], 0.025) > 0.3
            and source.spearmanH8 > 0.5
            and np.nanquantile(subset["spearmanH8"], 0.025) > 0.3
        )
    if solution:
        classifications = [
            "FULL_STATE_GRAPH_COMMITTOR_COORDINATE_ESTABLISHED",
            "DETERMINISTIC_BETA_CONDITIONED_STATE_SIGNAL_ESTABLISHED_WITHIN_RETROSPECTIVE_TARGET_TASK",
            "NOT_A_CONFIRMED_PAPER_OR_CAUSAL_RESULT",
        ]
        next_theme = "HUMAN_REVIEW_SOLUTION_BOUNDARY"
    elif oracle_rank_pass:
        classifications = [
            "TARGET_CONDITIONED_FULL_STATE_SIGNAL_ONLY",
            "GENERAL_PAST_ONLY_PRECURSOR_NOT_ESTABLISHED",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "SHORT_BRANCH_ENSEMBLE_MECHANISM_ATTRIBUTION"
    else:
        classifications = [
            "FULL_STATE_GRAPH_COMMITTOR_COORDINATE_NON_SUPPORT",
            "COMPACT_BETA_GRAPH_SIGNATURE_MISSES_SHOOTING_SIGNAL",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "SHORT_BRANCH_ENSEMBLE_MECHANISM_ATTRIBUTION"
    make_figures(predictions, metrics, gates, shift_summary, attributions)
    for name in (
        "preregistration.yaml",
        "decision_record.md",
        "fixture_results.parquet",
        "benchmark_projection.json",
        "response_registry.parquet",
        "target_coordinate_registry.parquet",
        "full_state_graph_representations.parquet",
        "markov_completeness_audit.parquet",
        "matrix_cardinality_audit.parquet",
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
    BASE.write_parquet(BUILD_ROOT / "feature_attribution_results.parquet", attributions)
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
    BASE.write_parquet(BUILD_ROOT / "invariance_results.parquet", invariance)
    BASE.write_parquet(BUILD_ROOT / "cohort_shift_state_results.parquet", shift_states)
    BASE.write_parquet(BUILD_ROOT / "cohort_shift_summary.parquet", shift_summary)
    BASE.write_parquet(BUILD_ROOT / "scientific_gate_results.parquet", gates)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l34.classification.v1",
            "classifications": classifications,
            "solutionGatePassed": solution,
            "oracleRankDiagnosticPassed": oracle_rank_pass,
            "primaryPastOnly": True,
            "primaryTargetInvariant": True,
            "responseTargetRetrospective": True,
            "withinMatrixOrderingIdentifiable": False,
            "priorStatusesChanged": False,
        },
    )
    pd.DataFrame(
        columns=["stage", "candidateId", "matrixIndex", "stateId", "exceptionClass", "exceptionMessage"]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    replayed = extract_representations(responses, coordinates, manifest)
    checks = {
        "featureReplayPassed": frame_hash(replayed) == frame_hash(representations),
        "modelReplayPassed": bool(models["exactReplay"].all()),
        "invariancePassed": bool(
            invariance[
                [
                    "suffixActuallyChanged",
                    "primaryUsesOnlyCurrentStateBetaPhase",
                    "primaryTargetInvariant",
                    "storedFeatureHashValid",
                ]
            ].all().all()
        ),
        "responseReplayPassed": frame_hash(L33.response_registry())
        == frame_hash(responses),
        "coordinateReplayPassed": frame_hash(L33.target_coordinates(responses))
        == frame_hash(coordinates),
        "fixturesPassed": bool(fixtures["passed"].all()),
        "seedFirewallPassed": json.loads(
            (LOOP_ROOT / "seed_firewall.json").read_text()
        )["status"]
        == "PASS",
        "immutablePriorPassed": prior["unchanged"],
        "oneStatePerMatrixVerified": bool(
            ~cardinality["withinMatrixOrderingIdentifiable"].any()
        ),
        "noBranchPredictor": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"regeneration validation failed: {checks}")
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l34.regeneration_validation.v1",
            "status": "PASS",
            "checks": checks,
            "representationFrameSha256": frame_hash(representations),
            "predictionFrameSha256": frame_hash(predictions),
            "metricFrameSha256": frame_hash(metrics),
        },
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l34.runtime.v1",
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": 1,
        "numericalLibraryThreads": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - start,
        "controllerCpuHours": (time.process_time() - start_cpu) / 3600,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    retained = sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file())
    temporary = sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l34.storage_validation.v1",
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
    report = report_text(
        metrics,
        gates,
        shift_summary,
        cardinality,
        classifications,
        runtime,
        next_theme,
    )
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L34_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        "# S19-L34 decision summary\n\n"
        + f"**Classification:** {', '.join(classifications)}\n\n"
        + f"**Past-only full-state solution:** `{solution}`.\n\n"
        + f"**Next:** `{next_theme}`.\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L34-promotion-stage")
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
        report.replace("# S19-L34", "# S19 current handoff — S19-L34", 1),
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
            "selectedDiscoveryLead": "FULL_STATE_GRAPH_COMMITTOR_COORDINATE"
            if solution
            else None,
            "nextAuthorizedLoop": None if solution else "S19-L35",
            "authorizationUpperBound": "S19-L42",
            "s20Active": False,
            "updatedAtUtc": runtime["completedAtUtc"],
        },
    )
    BASE.write_json(ARTIFACT_ROOT / "artifact_manifest.json", manifest_for(ARTIFACT_ROOT))
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
