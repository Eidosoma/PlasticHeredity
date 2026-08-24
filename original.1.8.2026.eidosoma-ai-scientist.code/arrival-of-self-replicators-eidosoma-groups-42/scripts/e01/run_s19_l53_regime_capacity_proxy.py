#!/usr/bin/env python3
"""Run S19-L53 past-observable regime-capacity proxy audit."""

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
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from e01_onset_discovery.full_state_graph import (
    PRIMARY_VIEW,
    feature_names,
    graph_views,
)
from e01_onset_discovery.heredity_phi_incremental import (
    fit_binomial_ridge,
    predict_probability,
)
from e01_onset_discovery.regime_capacity_proxy import (
    beta_structure_indices,
    binomial_cell_scores,
    center_within_groups,
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L52 = load_module(
    "e01_l53_l52_runner",
    ROOT / "scripts/e01/run_s19_l52_shooting_residual_regime_compression.py",
)
L51 = L52.L51
L50 = L52.L50
BASE = L52.BASE

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L53"
L50_ROOT = ARTIFACT_ROOT / "loops/L50"
L51_ROOT = ARTIFACT_ROOT / "loops/L51"
L52_ROOT = ARTIFACT_ROOT / "loops/L52"
BUILD_ROOT = Path("/cache/e01_s19_l53/build")
CONFIG = ROOT / "configs/e01/s19_l53_past_observable_regime_capacity_proxy.yaml"
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = ROOT / "src/e01_onset_discovery/regime_capacity_proxy.py"

LOOP_ID = "S19-L53"
VERSION = "E01-S19-L53-PAST-OBSERVABLE-REGIME-CAPACITY-PROXY-v1.0.0"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
HORIZONS = (4, 8, 12)
PRIMARY_HORIZON = 12
TARGETS = ("BREAK", "JOINT_BREAK_RUN3", "RUN3_GIVEN_BREAK")
PRIMARY_TARGET = "JOINT_BREAK_RUN3"
DIRECTIONS = (("A_TO_B", "A", "B"), ("B_TO_A", "B", "A"))
MODELS = (
    "TRAINING_PRIOR",
    "DIRECT_HISTORY_PHASE",
    "BETA_STRUCTURE",
    "FULL_STATE_GRAPH_HISTORY",
)
HISTORY_COLUMNS = (
    "normalizedGeneration",
    "currentMass",
    "prefixInheritanceFraction",
    "recentFiveInheritanceFraction",
    "prefixTrailingInheritanceRun",
    "latestParentDaughterH",
    "fissionsSinceLatestBreak",
    "currentInheritanceState",
    "currentRegimeDuration",
)
PCA_COMPONENTS = 12
RIDGE_C = 0.1
BOOTSTRAPS = 4096
PERMUTATIONS = 512
WORKERS = min(8, os.cpu_count() or 1)
SEED_ROOT = bytes.fromhex(
    "6c39c5d60b5c32ed4f35582dbdc849aeeb8b3456f096e4637a8c45dc67b12120"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    return L52.sha256_file(path)


def frame_hash(frame: pd.DataFrame) -> str:
    return L52.frame_hash(frame)


def array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def seed_material(*parts: object) -> bytes:
    canonical = tuple(part.item() if isinstance(part, np.generic) else part for part in parts)
    return hashlib.sha256(
        SEED_ROOT + b"\x00" + json.dumps(canonical, separators=(",", ":")).encode()
    ).digest()


def derived_seed(*parts: object) -> int:
    return int.from_bytes(seed_material(*parts)[:16], "big")


def generator(*parts: object) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(derived_seed(*parts)))


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or len(np.unique(a[mask])) < 2 or len(np.unique(b[mask])) < 2:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(map(float, np.quantile(finite, [0.025, 0.975])))


def validate_immutable_prior() -> dict[str, Any]:
    upstream = L52.validate_immutable_prior()
    manifest = json.loads((L52_ROOT / "artifact_manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        path = L52_ROOT / row["path"]
        actual = sha256_file(path) if path.is_file() else None
        rows.append(actual == row["sha256"])
    unchanged = bool(upstream["unchanged"] and rows and all(rows))
    return {
        "schema": "eidosoma.e01.s19_l53.immutable_prior_validation.v1",
        "unchanged": unchanged,
        "upstreamUnchanged": bool(upstream["unchanged"]),
        "l52FilesChecked": len(rows),
        "l52FilesUnchanged": int(sum(rows)),
        "aggregateSha256": hashlib.sha256(
            json.dumps(
                {
                    "upstream": upstream["aggregateSha256"],
                    "l52": manifest["aggregateSha256"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L50_FISSION_ALIGNED_PROCESS_RISK",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "F12 process risk is reliable but shooting did not robustly beat direct history in both candidates.",
                "frozenUse": "exact states, branch halves, process counts and development/validation roles",
                "url": None,
            },
            {
                "sourceId": "L52_STATE_LOCAL_HAZARD_COMPRESSION",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "Matrix-transfer and state-local branch hazards compress heldout committor probability in both candidates.",
                "frozenUse": "branch-derived comparison ceiling and teacher decomposition",
                "url": None,
            },
            {
                "sourceId": "L34_TARGET_BLIND_FULL_STATE_GRAPH",
                "evidenceClass": "DIRECT_FROZEN_E01_METHOD",
                "finding": "A fixed molecule-permutation-invariant current-state and beta graph representation is already implemented and validated.",
                "frozenUse": "one unchanged exact-state representation without a feature tournament",
                "url": None,
            },
            {
                "sourceId": "REVIEWER_REGIME_CAPACITY_FRAMING",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Discriminate persistent catalytic-matrix capacity, within-lineage state variation and irreducible shooting noise before adding another broad family.",
                "frozenUse": "separate beta-only and full state-beta students and report centered ordering",
                "url": None,
            },
        ]
    )


def model_registry() -> pd.DataFrame:
    descriptions = {
        "TRAINING_PRIOR": "Jeffreys-smoothed aggregate development fitting-half rate",
        "DIRECT_HISTORY_PHASE": "nine frozen online heredity-history, mass and phase coordinates",
        "BETA_STRUCTURE": "twelve development-only PCs of twenty beta-only graph coordinates",
        "FULL_STATE_GRAPH_HISTORY": "twelve development-only PCs of the 195-coordinate target-blind graph plus nine direct coordinates",
    }
    return pd.DataFrame(
        [
            {
                "modelId": model,
                "description": descriptions[model],
                "pastObservable": True,
                "matrixConstant": model == "BETA_STRUCTURE",
                "usesCurrentPhysicalState": model == "FULL_STATE_GRAPH_HISTORY",
                "usesFutureBranchFeature": False,
                "pcaComponents": PCA_COMPONENTS
                if model in {"BETA_STRUCTURE", "FULL_STATE_GRAPH_HISTORY"}
                else 0,
                "ridgeC": RIDGE_C if model != "TRAINING_PRIOR" else None,
                "hyperparameterSearch": False,
            }
            for model in MODELS
        ]
    )


def fixture_results() -> pd.DataFrame:
    names = feature_names()[PRIMARY_VIEW]
    indices = beta_structure_indices(names)
    rng = generator("fixture")
    beta = np.exp(rng.normal(-3, 0.5, size=(100, 100)))
    state_a = rng.poisson(1.4, size=100).astype(np.int64)
    state_b = rng.poisson(1.7, size=100).astype(np.int64)
    state_a[0] += 1
    state_b[1] += 1
    target_a = np.ones(100) / 100
    target_b = rng.random(100)
    target_b /= target_b.sum()
    kwargs = {
        "generation_local_step": 3,
        "observation_kind": "post_fission",
        "completed_fissions": 20,
        "batch_step": 15,
        "landmark": 20,
        "target_component_fraction": 0.0,
    }
    graph_a = graph_views(state_a, beta, target_a, **kwargs)[PRIMARY_VIEW]
    graph_b = graph_views(state_b, beta, target_a, **kwargs)[PRIMARY_VIEW]
    graph_target = graph_views(state_a, beta, target_b, **kwargs)[PRIMARY_VIEW]
    centered = center_within_groups([1.0, 3.0, 2.0, 8.0], ["a", "a", "b", "b"])
    loss, brier = binomial_cell_scores([0.25, 0.75], [1, 3], [4, 4])
    x = rng.normal(size=(50, 12))
    success = rng.binomial(32, 0.2, size=50)
    model_a = fit_binomial_ridge(x, success, np.full(50, 32), seed=7, c=RIDGE_C)
    model_b = fit_binomial_ridge(x, success, np.full(50, 32), seed=7, c=RIDGE_C)
    rows = [
        ("F01_BETA_INDEX_SCOPE", len(indices) == 20),
        ("F02_BETA_STATE_INVARIANCE", np.array_equal(graph_a[list(indices)], graph_b[list(indices)])),
        ("F03_PRIMARY_TARGET_INVARIANCE", np.array_equal(graph_a, graph_target)),
        ("F04_GRAPH_FINITE", bool(np.isfinite(graph_a).all() and len(graph_a) == 195)),
        ("F05_GROUP_CENTERING", bool(np.allclose(centered, [-1, 1, -3, 3]))),
        ("F06_BINOMIAL_SCORE_FINITE", bool(np.isfinite(loss).all() and np.isfinite(brier).all())),
        ("F07_MODEL_EXACT_REPLAY", np.array_equal(predict_probability(model_a, x), predict_probability(model_b, x))),
        ("F08_DIRECTION_SCOPE", DIRECTIONS == (("A_TO_B", "A", "B"), ("B_TO_A", "B", "A"))),
        ("F09_MODEL_SCOPE", len(MODELS) == 4),
        ("F10_SEED_REPLAY", derived_seed("fixture", np.int64(5)) == derived_seed("fixture", 5)),
    ]
    return pd.DataFrame(rows, columns=["fixtureId", "passed"])


def analysis_seed_manifest() -> pd.DataFrame:
    rows = []
    for purpose, repetitions in (
        ("model", 1),
        ("matrix_bootstrap", BOOTSTRAPS),
        ("whole_matrix_q_permutation", PERMUTATIONS),
    ):
        for candidate in CANDIDATES:
            for direction, _, _ in DIRECTIONS:
                material = seed_material(purpose, candidate, direction)
                rows.append(
                    {
                        "purpose": purpose,
                        "candidateId": candidate,
                        "direction": direction,
                        "repetitions": repetitions,
                        "derivedSeed": str(int.from_bytes(material[:16], "big")),
                        "seedMaterialSha256": material.hex(),
                    }
                )
    return pd.DataFrame(rows)


def seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    prior: set[str] = set()
    for root in (L50_ROOT, L51_ROOT, L52_ROOT):
        path = root / "analysis_seed_manifest.parquet"
        if path.is_file():
            prior.update(pd.read_parquet(path)["seedMaterialSha256"].astype(str))
    current = set(seeds["seedMaterialSha256"].astype(str))
    overlap = current & prior
    return {
        "schema": "eidosoma.e01.s19_l53.seed_firewall.v1",
        "status": "PASS" if len(current) == len(seeds) and not overlap else "FAIL",
        "rootHex": SEED_ROOT.hex(),
        "newAnalysisStreams": len(current),
        "overlapCount": len(overlap),
    }


def _graph_feature_worker(payload: dict[str, Any]) -> dict[str, Any]:
    matrix = int(payload["matrixIndex"])
    beta = L50.L28.generate_beta(
        L50.L28.derive_seed(
            L50.L28.L23_ROOT_HEX,
            L50.L28.L23_PHASE,
            "catalytic_matrix",
            matrix,
        )
    )
    if L50.L28.simulator_array_sha256(beta) != payload["betaSha256"]:
        raise RuntimeError("L53 beta identity mismatch")
    state = np.asarray(payload["state"], dtype=np.int64)
    values = graph_views(
        state,
        beta,
        np.ones(100, dtype=np.float64) / 100,
        generation_local_step=int(payload["currentGenerationLocalStep"]),
        observation_kind=str(payload["currentObservationKind"]),
        completed_fissions=int(payload["currentCompletedFissions"]),
        batch_step=int(payload["currentBatchStep"]),
        landmark=int(payload["completedFissionLandmark"]),
        target_component_fraction=0.0,
    )[PRIMARY_VIEW]
    indices = beta_structure_indices(feature_names()[PRIMARY_VIEW])
    beta_values = values[list(indices)]
    return {
        "stateId": payload["stateId"],
        "graphValues": values.tolist(),
        "graphSha256": array_hash(values),
        "betaValues": beta_values.tolist(),
        "betaFeatureSha256": array_hash(beta_values),
    }


def extract_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, expanded = L50.select_matrices()
    payloads, states, observed, validation = L50.build_states(expanded)
    frozen_states = pd.read_parquet(L50_ROOT / "restored_state_registry.parquet")
    frozen_observed = pd.read_parquet(L50_ROOT / "observed_process_outcomes.parquet")
    if frame_hash(states) != frame_hash(frozen_states) or frame_hash(observed) != frame_hash(frozen_observed):
        raise RuntimeError("L53 exact L50 state/outcome replay failure")
    if not validation.drop(columns=["stateId"]).all().all():
        raise RuntimeError("L53 state restoration validation failure")
    prefix = pd.read_parquet(L51_ROOT / "prefix_state_results.parquet")
    state_table = states.merge(
        prefix[
            [
                "stateId",
                "currentInheritanceState",
                "currentRegimeDuration",
                "targetUsesCompletedTestTrajectory",
            ]
        ],
        on="stateId",
        validate="one_to_one",
        suffixes=("", "_l51"),
    )
    if state_table["targetUsesCompletedTestTrajectory_l51"].any():
        raise RuntimeError("L53 prefix features unexpectedly use a completed target")
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        graph_rows = list(executor.map(_graph_feature_worker, payloads))
    graph = pd.DataFrame(graph_rows)
    features = state_table.merge(graph, on="stateId", validate="one_to_one")
    features = features.sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "completedFissionLandmark"]
    ).reset_index(drop=True)
    beta_consistency = []
    for keys, group in features.groupby(["candidateId", "matrixIndex"], sort=True):
        candidate, matrix = keys
        beta_consistency.append(
            {
                "candidateId": candidate,
                "matrixIndex": int(matrix),
                "states": len(group),
                "uniqueBetaFeatureHashes": int(group["betaFeatureSha256"].nunique()),
                "passed": group["betaFeatureSha256"].nunique() == 1,
            }
        )
    beta_validation = pd.DataFrame(beta_consistency)
    if len(features) != 800 or not beta_validation["passed"].all():
        raise RuntimeError("L53 feature scope or beta-only invariance failure")
    replay = frozen_states[["stateId", "currentStateSha256", "betaSha256"]].merge(
        features[["stateId", "currentStateSha256", "betaSha256", "graphSha256"]],
        on="stateId",
        validate="one_to_one",
        suffixes=("Frozen", "Replayed"),
    )
    replay["stateIdentityPassed"] = replay["currentStateSha256Frozen"].eq(
        replay["currentStateSha256Replayed"]
    )
    replay["betaIdentityPassed"] = replay["betaSha256Frozen"].eq(
        replay["betaSha256Replayed"]
    )
    if not replay[["stateIdentityPassed", "betaIdentityPassed"]].all().all():
        raise RuntimeError("L53 state/beta replay mismatch")
    return features, replay, beta_validation


def transformed_features(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], tuple[StandardScaler, PCA]]]:
    rows = []
    registry = []
    fitted: dict[tuple[str, str], tuple[StandardScaler, PCA]] = {}
    direct = features[list(HISTORY_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(direct).all():
        raise RuntimeError("L53 direct features nonfinite")
    for source, values in zip(features.itertuples(index=False), direct, strict=True):
        rows.append(
            {
                "stateId": source.stateId,
                "candidateId": source.candidateId,
                "matrixRole": source.matrixRole,
                "matrixIndex": int(source.matrixIndex),
                "completedFissionLandmark": int(source.completedFissionLandmark),
                "modelId": "DIRECT_HISTORY_PHASE",
                "values": values.tolist(),
                "featureSha256": array_hash(values),
            }
        )
    for candidate in CANDIDATES:
        candidate_frame = features[features["candidateId"].eq(candidate)].reset_index(drop=True)
        development_mask = candidate_frame["matrixRole"].eq("DEVELOPMENT").to_numpy()
        for model in ("BETA_STRUCTURE", "FULL_STATE_GRAPH_HISTORY"):
            raw = np.stack(
                candidate_frame["betaValues" if model == "BETA_STRUCTURE" else "graphValues"].map(
                    lambda value: np.asarray(value, dtype=np.float64)
                )
            )
            scaler = StandardScaler().fit(raw[development_mask])
            standardized = scaler.transform(raw[development_mask])
            pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full").fit(standardized)
            replay_scaler = StandardScaler().fit(raw[development_mask])
            replay_pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full").fit(
                replay_scaler.transform(raw[development_mask])
            )
            exact = bool(
                np.array_equal(scaler.mean_, replay_scaler.mean_)
                and np.array_equal(scaler.scale_, replay_scaler.scale_)
                and np.array_equal(pca.components_, replay_pca.components_)
                and np.array_equal(pca.explained_variance_, replay_pca.explained_variance_)
            )
            if not exact:
                raise RuntimeError("L53 PCA replay failure")
            transformed = pca.transform(scaler.transform(raw))
            if model == "FULL_STATE_GRAPH_HISTORY":
                history = candidate_frame[list(HISTORY_COLUMNS)].to_numpy(dtype=np.float64)
                transformed = np.column_stack((transformed, history))
            fitted[(candidate, model)] = (scaler, pca)
            registry.append(
                {
                    "candidateId": candidate,
                    "modelId": model,
                    "rawFeatureCount": raw.shape[1],
                    "transformedFeatureCount": transformed.shape[1],
                    "pcaComponents": PCA_COMPONENTS,
                    "developmentStates": int(development_mask.sum()),
                    "explainedVarianceFraction": float(pca.explained_variance_ratio_.sum()),
                    "exactReplay": exact,
                    "rawScalerMean": json.dumps(scaler.mean_.tolist()),
                    "rawScalerScale": json.dumps(scaler.scale_.tolist()),
                    "pcaComponentsArray": json.dumps(pca.components_.tolist()),
                }
            )
            for source, values in zip(candidate_frame.itertuples(index=False), transformed, strict=True):
                rows.append(
                    {
                        "stateId": source.stateId,
                        "candidateId": candidate,
                        "matrixRole": source.matrixRole,
                        "matrixIndex": int(source.matrixIndex),
                        "completedFissionLandmark": int(source.completedFissionLandmark),
                        "modelId": model,
                        "values": values.tolist(),
                        "featureSha256": array_hash(values),
                    }
                )
    frame = pd.DataFrame(rows).sort_values(
        ["modelId", "candidateId", "matrixRole", "matrixIndex", "completedFissionLandmark"]
    ).reset_index(drop=True)
    if len(frame) != 800 * 3 or frame.duplicated(["stateId", "modelId"]).any():
        raise RuntimeError("L53 transformed feature scope failure")
    return frame, pd.DataFrame(registry), fitted


def _feature_matrix(
    transformed: pd.DataFrame, state_ids: pd.Series, model: str
) -> np.ndarray:
    indexed = transformed[transformed["modelId"].eq(model)].set_index("stateId")
    values = indexed.loc[state_ids, "values"]
    return np.stack(values.map(lambda value: np.asarray(value, dtype=np.float64)))


def fit_and_predict(
    transformed: pd.DataFrame,
    pca_registry: dict[tuple[str, str], tuple[StandardScaler, PCA]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estimates = pd.read_parquet(L50_ROOT / "state_committor_results.parquet")
    predictions = []
    model_rows = []
    attribution_rows = []
    for candidate in CANDIDATES:
        for direction, fit_half, score_half in DIRECTIONS:
            fit_success = f"successesHalf{fit_half}"
            fit_trials = f"trialsHalf{fit_half}"
            score_success = f"successesHalf{score_half}"
            score_trials = f"trialsHalf{score_half}"
            for horizon in HORIZONS:
                for target in TARGETS:
                    source = estimates[
                        estimates["candidateId"].eq(candidate)
                        & estimates["horizon"].eq(horizon)
                        & estimates["targetType"].eq(target)
                    ].copy()
                    development = source[
                        source["matrixRole"].eq("DEVELOPMENT") & source[fit_trials].gt(0)
                    ].sort_values(["matrixIndex", "completedFissionLandmark"])
                    validation = source[
                        source["matrixRole"].eq("VALIDATION") & source[score_trials].gt(0)
                    ].sort_values(["matrixIndex", "completedFissionLandmark"])
                    if len(development) < 40 or len(validation) < 40:
                        raise RuntimeError("L53 insufficient fit/score response support")
                    prior = (development[fit_success].sum() + 0.5) / (
                        development[fit_trials].sum() + 1.0
                    )
                    for model_id in MODELS:
                        if model_id == "TRAINING_PRIOR":
                            probability = np.full(len(validation), prior, dtype=np.float64)
                            feature_count = 0
                            exact = True
                            coefficient = np.empty(0)
                            intercept = float(np.log(prior / (1 - prior)))
                            iterations = 0
                        else:
                            x_train = _feature_matrix(
                                transformed, development["stateId"], model_id
                            )
                            x_valid = _feature_matrix(
                                transformed, validation["stateId"], model_id
                            )
                            seed = derived_seed(
                                "model", candidate, direction, horizon, target, model_id
                            ) % (2**32 - 1)
                            fitted = fit_binomial_ridge(
                                x_train,
                                development[fit_success].to_numpy(dtype=np.int64),
                                development[fit_trials].to_numpy(dtype=np.int64),
                                seed=seed,
                                c=RIDGE_C,
                            )
                            replay = fit_binomial_ridge(
                                x_train,
                                development[fit_success].to_numpy(dtype=np.int64),
                                development[fit_trials].to_numpy(dtype=np.int64),
                                seed=seed,
                                c=RIDGE_C,
                            )
                            probability = predict_probability(fitted, x_valid)
                            replay_probability = predict_probability(replay, x_valid)
                            exact = bool(np.array_equal(probability, replay_probability))
                            if not exact:
                                raise RuntimeError("L53 model replay failure")
                            feature_count = x_train.shape[1]
                            scaler = fitted.named_steps["scale"]
                            logistic = fitted.named_steps["model"]
                            coefficient = logistic.coef_[0] / scaler.scale_
                            intercept = float(logistic.intercept_[0])
                            iterations = int(logistic.n_iter_[0])
                            if model_id == "DIRECT_HISTORY_PHASE":
                                names = HISTORY_COLUMNS
                                raw_coefficients = coefficient
                            else:
                                _, pca = pca_registry[(candidate, model_id)]
                                pc_coefficient = coefficient[:PCA_COMPONENTS]
                                raw_coefficients = pc_coefficient @ pca.components_
                                raw_names = (
                                    tuple(
                                        feature_names()[PRIMARY_VIEW][index]
                                        for index in beta_structure_indices(
                                            feature_names()[PRIMARY_VIEW]
                                        )
                                    )
                                    if model_id == "BETA_STRUCTURE"
                                    else feature_names()[PRIMARY_VIEW]
                                )
                                names = tuple(raw_names)
                                if model_id == "FULL_STATE_GRAPH_HISTORY":
                                    names = (*names, *HISTORY_COLUMNS)
                                    raw_coefficients = np.concatenate(
                                        (raw_coefficients, coefficient[PCA_COMPONENTS:])
                                    )
                            for name, value in zip(names, raw_coefficients, strict=True):
                                attribution_rows.append(
                                    {
                                        "candidateId": candidate,
                                        "direction": direction,
                                        "horizon": horizon,
                                        "targetType": target,
                                        "modelId": model_id,
                                        "featureName": name,
                                        "standardizedBackprojectedCoefficient": float(value),
                                        "absoluteCoefficient": abs(float(value)),
                                    }
                                )
                        log_loss, brier = binomial_cell_scores(
                            probability,
                            validation[score_success].to_numpy(dtype=np.int64),
                            validation[score_trials].to_numpy(dtype=np.int64),
                        )
                        q = (
                            validation[score_success].to_numpy(dtype=np.float64) + 0.5
                        ) / (validation[score_trials].to_numpy(dtype=np.float64) + 1.0)
                        for source_row, p, q_value, loss, brier_value in zip(
                            validation.itertuples(index=False),
                            probability,
                            q,
                            log_loss,
                            brier,
                            strict=True,
                        ):
                            predictions.append(
                                {
                                    "stateId": source_row.stateId,
                                    "candidateId": candidate,
                                    "matrixRole": "VALIDATION",
                                    "matrixIndex": int(source_row.matrixIndex),
                                    "completedFissionLandmark": int(
                                        source_row.completedFissionLandmark
                                    ),
                                    "direction": direction,
                                    "fitHalf": fit_half,
                                    "scoreHalf": score_half,
                                    "horizon": horizon,
                                    "targetType": target,
                                    "modelId": model_id,
                                    "predictedProbability": float(p),
                                    "successes": int(getattr(source_row, score_success)),
                                    "trials": int(getattr(source_row, score_trials)),
                                    "empiricalQ": float(q_value),
                                    "branchLogLoss": float(loss),
                                    "qBrier": float(brier_value),
                                }
                            )
                        model_rows.append(
                            {
                                "candidateId": candidate,
                                "direction": direction,
                                "horizon": horizon,
                                "targetType": target,
                                "modelId": model_id,
                                "developmentStates": len(development),
                                "validationStates": len(validation),
                                "featureCount": feature_count,
                                "trainingPrior": float(prior),
                                "intercept": intercept,
                                "coefficients": json.dumps(coefficient.tolist()),
                                "iterations": iterations,
                                "ridgeC": RIDGE_C if model_id != "TRAINING_PRIOR" else None,
                                "fitRole": "DEVELOPMENT",
                                "scoreRole": "VALIDATION",
                                "exactReplay": exact,
                            }
                        )
    prediction_frame = pd.DataFrame(predictions).sort_values(
        ["candidateId", "direction", "horizon", "targetType", "modelId", "matrixIndex", "completedFissionLandmark"]
    ).reset_index(drop=True)
    return prediction_frame, pd.DataFrame(model_rows), pd.DataFrame(attribution_rows)


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_rows = []
    summary_rows = []
    keys = ["candidateId", "direction", "horizon", "targetType", "modelId"]
    for key, group in predictions.groupby(keys, sort=True):
        candidate, direction, horizon, target, model = key
        centered_p = center_within_groups(group["predictedProbability"], group["matrixIndex"])
        centered_q = center_within_groups(group["empiricalQ"], group["matrixIndex"])
        by_matrix = group.groupby("matrixIndex", sort=True).agg(
            branchLogLoss=("branchLogLoss", "mean"),
            qBrier=("qBrier", "mean"),
            predictedProbability=("predictedProbability", "mean"),
            empiricalQ=("empiricalQ", "mean"),
            states=("stateId", "size"),
        )
        for matrix, row in by_matrix.iterrows():
            matrix_rows.append(
                {
                    "candidateId": candidate,
                    "direction": direction,
                    "horizon": int(horizon),
                    "targetType": target,
                    "modelId": model,
                    "matrixIndex": int(matrix),
                    **row.to_dict(),
                }
            )
        summary_rows.append(
            {
                "candidateId": candidate,
                "direction": direction,
                "horizon": int(horizon),
                "targetType": target,
                "modelId": model,
                "matrices": int(group["matrixIndex"].nunique()),
                "states": len(group),
                "equalMatrixMeanBranchLogLoss": float(by_matrix["branchLogLoss"].mean()),
                "equalMatrixMeanQBrier": float(by_matrix["qBrier"].mean()),
                "qSpearman": safe_spearman(
                    group["predictedProbability"].to_numpy(), group["empiricalQ"].to_numpy()
                ),
                "centeredQSpearman": safe_spearman(centered_p, centered_q),
                "meanPredictedProbability": float(group["predictedProbability"].mean()),
                "meanEmpiricalQ": float(group["empiricalQ"].mean()),
            }
        )
    return pd.DataFrame(matrix_rows), pd.DataFrame(summary_rows)


def bootstrap_primary(predictions: pd.DataFrame) -> pd.DataFrame:
    primary = predictions[
        predictions["horizon"].eq(PRIMARY_HORIZON)
        & predictions["targetType"].eq(PRIMARY_TARGET)
    ]
    rows = []
    for (candidate, direction), group in primary.groupby(
        ["candidateId", "direction"], sort=True
    ):
        matrices = np.sort(group["matrixIndex"].unique())
        indexed = {
            model: model_group.set_index(["matrixIndex", "completedFissionLandmark"]).sort_index()
            for model, model_group in group.groupby("modelId", sort=True)
        }
        rng = generator("matrix_bootstrap", candidate, direction)
        for replicate in range(BOOTSTRAPS):
            sampled = rng.choice(matrices, size=len(matrices), replace=True)
            record: dict[str, Any] = {
                "candidateId": candidate,
                "direction": direction,
                "replicate": replicate,
            }
            for model in MODELS:
                parts = []
                labels = []
                for draw, matrix in enumerate(sampled):
                    part = indexed[model].loc[[matrix]].reset_index()
                    parts.append(part)
                    labels.extend([draw] * len(part))
                sample = pd.concat(parts, ignore_index=True)
                centered_p = center_within_groups(sample["predictedProbability"], labels)
                centered_q = center_within_groups(sample["empiricalQ"], labels)
                matrix_loss = sample.groupby(labels)["branchLogLoss"].mean()
                matrix_brier = sample.groupby(labels)["qBrier"].mean()
                record[f"logLoss__{model}"] = float(matrix_loss.mean())
                record[f"qBrier__{model}"] = float(matrix_brier.mean())
                record[f"qSpearman__{model}"] = safe_spearman(
                    sample["predictedProbability"].to_numpy(),
                    sample["empiricalQ"].to_numpy(),
                )
                record[f"centeredQSpearman__{model}"] = safe_spearman(
                    centered_p, centered_q
                )
            record["logLossGain__DIRECT_VS_PRIOR"] = (
                record["logLoss__TRAINING_PRIOR"]
                - record["logLoss__DIRECT_HISTORY_PHASE"]
            )
            record["logLossGain__BETA_VS_PRIOR"] = (
                record["logLoss__TRAINING_PRIOR"] - record["logLoss__BETA_STRUCTURE"]
            )
            record["logLossGain__FULL_VS_PRIOR"] = (
                record["logLoss__TRAINING_PRIOR"]
                - record["logLoss__FULL_STATE_GRAPH_HISTORY"]
            )
            record["logLossGain__FULL_VS_DIRECT"] = (
                record["logLoss__DIRECT_HISTORY_PHASE"]
                - record["logLoss__FULL_STATE_GRAPH_HISTORY"]
            )
            rows.append(record)
    return pd.DataFrame(rows)


def bootstrap_summaries(
    metrics: pd.DataFrame, bootstraps: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = metrics[
        metrics["horizon"].eq(PRIMARY_HORIZON)
        & metrics["targetType"].eq(PRIMARY_TARGET)
    ]
    rank_rows = []
    comparison_rows = []
    comparison_map = {
        "DIRECT_VS_PRIOR": ("DIRECT_HISTORY_PHASE", "TRAINING_PRIOR"),
        "BETA_VS_PRIOR": ("BETA_STRUCTURE", "TRAINING_PRIOR"),
        "FULL_VS_PRIOR": ("FULL_STATE_GRAPH_HISTORY", "TRAINING_PRIOR"),
        "FULL_VS_DIRECT": ("FULL_STATE_GRAPH_HISTORY", "DIRECT_HISTORY_PHASE"),
    }
    for (candidate, direction), boot in bootstraps.groupby(
        ["candidateId", "direction"], sort=True
    ):
        for model in MODELS:
            point = primary[
                primary["candidateId"].eq(candidate)
                & primary["direction"].eq(direction)
                & primary["modelId"].eq(model)
            ].iloc[0]
            q_low, q_high = interval(boot[f"qSpearman__{model}"].to_numpy())
            c_low, c_high = interval(
                boot[f"centeredQSpearman__{model}"].to_numpy()
            )
            rank_rows.append(
                {
                    "candidateId": candidate,
                    "direction": direction,
                    "modelId": model,
                    "qSpearman": float(point.qSpearman),
                    "qSpearmanLower95": q_low,
                    "qSpearmanUpper95": q_high,
                    "centeredQSpearman": float(point.centeredQSpearman),
                    "centeredQSpearmanLower95": c_low,
                    "centeredQSpearmanUpper95": c_high,
                }
            )
        for comparison, (model, reference) in comparison_map.items():
            values = boot[f"logLossGain__{comparison}"].to_numpy(dtype=np.float64)
            low, high = interval(values)
            point_model = primary[
                primary["candidateId"].eq(candidate)
                & primary["direction"].eq(direction)
                & primary["modelId"].eq(model)
            ]["equalMatrixMeanBranchLogLoss"].iloc[0]
            point_reference = primary[
                primary["candidateId"].eq(candidate)
                & primary["direction"].eq(direction)
                & primary["modelId"].eq(reference)
            ]["equalMatrixMeanBranchLogLoss"].iloc[0]
            comparison_rows.append(
                {
                    "candidateId": candidate,
                    "direction": direction,
                    "comparisonId": comparison,
                    "modelId": model,
                    "referenceModelId": reference,
                    "logLossImprovement": float(point_reference - point_model),
                    "logLossImprovementLower95": low,
                    "logLossImprovementUpper95": high,
                    "fractionBootstrapPositive": float(np.mean(values > 0)),
                }
            )
    return pd.DataFrame(rank_rows), pd.DataFrame(comparison_rows)


def q_permutations(predictions: pd.DataFrame) -> pd.DataFrame:
    primary = predictions[
        predictions["horizon"].eq(PRIMARY_HORIZON)
        & predictions["targetType"].eq(PRIMARY_TARGET)
        & predictions["modelId"].isin(
            ["DIRECT_HISTORY_PHASE", "BETA_STRUCTURE", "FULL_STATE_GRAPH_HISTORY"]
        )
    ]
    rows = []
    for (candidate, direction, model), group in primary.groupby(
        ["candidateId", "direction", "modelId"], sort=True
    ):
        p = group.pivot(
            index="matrixIndex",
            columns="completedFissionLandmark",
            values="predictedProbability",
        ).sort_index()
        q = group.pivot(
            index="matrixIndex",
            columns="completedFissionLandmark",
            values="empiricalQ",
        ).reindex_like(p)
        matrix_labels = np.repeat(np.arange(len(p)), p.shape[1])
        observed = safe_spearman(p.to_numpy().ravel(), q.to_numpy().ravel())
        observed_centered = safe_spearman(
            center_within_groups(p.to_numpy().ravel(), matrix_labels),
            center_within_groups(q.to_numpy().ravel(), matrix_labels),
        )
        rng = generator("whole_matrix_q_permutation", candidate, direction, model)
        null = np.empty(PERMUTATIONS)
        null_centered = np.empty(PERMUTATIONS)
        for replicate in range(PERMUTATIONS):
            permuted = q.to_numpy()[rng.permutation(len(q))]
            null[replicate] = safe_spearman(p.to_numpy().ravel(), permuted.ravel())
            null_centered[replicate] = safe_spearman(
                center_within_groups(p.to_numpy().ravel(), matrix_labels),
                center_within_groups(permuted.ravel(), matrix_labels),
            )
        rows.append(
            {
                "candidateId": candidate,
                "direction": direction,
                "modelId": model,
                "observedQSpearman": observed,
                "observedCenteredQSpearman": observed_centered,
                "overallUpperTailP": float(
                    (1 + np.sum(null >= observed)) / (PERMUTATIONS + 1)
                ),
                "centeredUpperTailP": float(
                    (1 + np.sum(null_centered >= observed_centered))
                    / (PERMUTATIONS + 1)
                )
                if np.isfinite(observed_centered)
                else np.nan,
                "permutations": PERMUTATIONS,
                "wholeMatrixTrajectoryPermutation": True,
            }
        )
    return pd.DataFrame(rows)


def branch_ceiling_comparators() -> pd.DataFrame:
    source = pd.read_parquet(L52_ROOT / "q_rank_results.parquet")
    return source.assign(
        comparatorClass="BRANCH_DERIVED_CEILING",
        pastObservable=False,
        reusedWithoutChange=True,
    )


def scientific_gates(
    ranks: pd.DataFrame,
    comparisons: pd.DataFrame,
    permutations: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:
    rows = []
    for candidate in CANDIDATES:
        candidate_rank = ranks[ranks["candidateId"].eq(candidate)]
        candidate_comparison = comparisons[comparisons["candidateId"].eq(candidate)]
        candidate_permutation = permutations[permutations["candidateId"].eq(candidate)]

        def comparison_min(
            name: str,
            candidate_frame: pd.DataFrame = candidate_comparison,
        ) -> float:
            return float(
                candidate_frame[
                    candidate_frame["comparisonId"].eq(name)
                ]["logLossImprovementLower95"].min()
            )

        beta_rank = candidate_rank[candidate_rank["modelId"].eq("BETA_STRUCTURE")]
        beta_perm = candidate_permutation[
            candidate_permutation["modelId"].eq("BETA_STRUCTURE")
        ]
        stable = bool(
            comparison_min("BETA_VS_PRIOR") > 0
            and beta_rank["qSpearmanLower95"].min() > 0.3
            and beta_perm["overallUpperTailP"].max() < 0.01
        )
        full_rank = candidate_rank[
            candidate_rank["modelId"].eq("FULL_STATE_GRAPH_HISTORY")
        ]
        full_perm = candidate_permutation[
            candidate_permutation["modelId"].eq("FULL_STATE_GRAPH_HISTORY")
        ]
        state_local = bool(
            comparison_min("FULL_VS_DIRECT") > 0
            and comparison_min("FULL_VS_PRIOR") > 0
            and full_rank["qSpearmanLower95"].min() > 0.3
            and full_rank["centeredQSpearmanLower95"].min() > 0.1
            and full_perm["overallUpperTailP"].max() < 0.01
            and full_perm["centeredUpperTailP"].max() < 0.01
        )
        rows.extend(
            [
                {
                    "gateId": f"STABLE_CAPACITY::{candidate}",
                    "candidateId": candidate,
                    "gateFamily": "STABLE_CATALYTIC_CAPACITY",
                    "minimumProperScoreLower95": comparison_min("BETA_VS_PRIOR"),
                    "minimumQSpearmanLower95": float(beta_rank["qSpearmanLower95"].min()),
                    "minimumCenteredQSpearmanLower95": np.nan,
                    "maximumOverallPermutationP": float(beta_perm["overallUpperTailP"].max()),
                    "maximumCenteredPermutationP": np.nan,
                    "passed": stable,
                },
                {
                    "gateId": f"STATE_LOCAL_PROXY::{candidate}",
                    "candidateId": candidate,
                    "gateFamily": "STATE_LOCAL_PAST_PROXY",
                    "minimumProperScoreLower95": min(
                        comparison_min("FULL_VS_DIRECT"),
                        comparison_min("FULL_VS_PRIOR"),
                    ),
                    "minimumQSpearmanLower95": float(full_rank["qSpearmanLower95"].min()),
                    "minimumCenteredQSpearmanLower95": float(
                        full_rank["centeredQSpearmanLower95"].min()
                    ),
                    "maximumOverallPermutationP": float(full_perm["overallUpperTailP"].max()),
                    "maximumCenteredPermutationP": float(
                        full_perm["centeredUpperTailP"].max()
                    ),
                    "passed": state_local,
                },
            ]
        )
    gates = pd.DataFrame(rows)

    def both(family: str) -> bool:
        selected = gates[gates["gateFamily"].eq(family)]
        return len(selected) == 2 and bool(selected["passed"].all())

    stable = both("STABLE_CATALYTIC_CAPACITY")
    state_local = both("STATE_LOCAL_PAST_PROXY")
    classifications = [
        "PAST_OBSERVABLE_CATALYTIC_CAPACITY_IDENTIFIED"
        if stable
        else "BETA_STRUCTURE_DOES_NOT_EXPLAIN_TRANSFERABLE_REGIME_CAPACITY",
        "PAST_OBSERVABLE_STATE_LOCAL_HAZARD_PROXY_IDENTIFIED"
        if state_local
        else "STATE_LOCAL_SHOOTING_SIGNAL_NOT_DISTILLED_FROM_FROZEN_FULL_STATE",
    ]
    if stable and not state_local:
        classifications.append("REGIME_RISK_DOMINATED_BY_STABLE_MATRIX_CAPACITY")
        next_theme = "L54_UNTOUCHED_MATRIX_CAPACITY_CONFIRMATION"
    elif state_local:
        classifications.append("PAST_OBSERVABLE_PROCESS_RISK_LEAD")
        next_theme = "L54_UNTOUCHED_PAST_OBSERVABLE_PROCESS_RISK_CONFIRMATION"
    else:
        classifications.append("SHOOTING_REMAINS_NECESSARY_FOR_STATE_LOCAL_RISK")
        next_theme = "L54_BRANCH_PATH_ORDER_INFORMATION_AUDIT"
    classifications.append("NOT_PROMOTABLE_AS_CONFIRMED")
    gates = pd.concat(
        [
            gates,
            pd.DataFrame(
                [
                    {
                        "gateId": "COMPLETE_CROSS_CANDIDATE_ADJUDICATION",
                        "candidateId": "BOTH",
                        "gateFamily": "COMPLETE",
                        "minimumProperScoreLower95": np.nan,
                        "minimumQSpearmanLower95": np.nan,
                        "minimumCenteredQSpearmanLower95": np.nan,
                        "maximumOverallPermutationP": np.nan,
                        "maximumCenteredPermutationP": np.nan,
                        "passed": True,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return gates, classifications, next_theme


def compute_tables() -> tuple[dict[str, pd.DataFrame], list[str], str]:
    features, replay, beta_validation = extract_features()
    transformed, pca, fitted_pca = transformed_features(features)
    predictions, models, attributions = fit_and_predict(transformed, fitted_pca)
    matrix_metrics, metrics = metric_tables(predictions)
    bootstraps = bootstrap_primary(predictions)
    ranks, comparisons = bootstrap_summaries(metrics, bootstraps)
    permutations = q_permutations(predictions)
    gates, classifications, next_theme = scientific_gates(
        ranks, comparisons, permutations
    )
    feature_output = features[
        [
            "stateId",
            "matrixRole",
            "candidateId",
            "matrixIndex",
            "completedFissionLandmark",
            *HISTORY_COLUMNS,
            "graphValues",
            "graphSha256",
            "betaValues",
            "betaFeatureSha256",
            "currentStateSha256",
            "betaSha256",
            "targetUsesCompletedTestTrajectory",
        ]
    ].copy()
    tables = {
        "state_feature_results.parquet": feature_output,
        "state_beta_identity_validation.parquet": replay,
        "beta_structure_invariance.parquet": beta_validation,
        "transformed_feature_results.parquet": transformed,
        "pca_registry.parquet": pca,
        "fitted_model_registry.parquet": models,
        "feature_attribution_results.parquet": attributions,
        "prediction_results.parquet": predictions,
        "matrix_metric_results.parquet": matrix_metrics,
        "predictive_metric_results.parquet": metrics,
        "primary_matrix_bootstrap.parquet": bootstraps,
        "q_rank_results.parquet": ranks,
        "model_comparisons.parquet": comparisons,
        "whole_matrix_permutation_results.parquet": permutations,
        "branch_ceiling_comparators.parquet": branch_ceiling_comparators(),
        "scientific_gate_results.parquet": gates,
    }
    return tables, classifications, next_theme


def make_figures(tables: dict[str, pd.DataFrame]) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)
    metrics = tables["predictive_metric_results.parquet"]
    primary = metrics[
        metrics["horizon"].eq(PRIMARY_HORIZON)
        & metrics["targetType"].eq(PRIMARY_TARGET)
    ].copy()
    labels = {
        "TRAINING_PRIOR": "prior",
        "DIRECT_HISTORY_PHASE": "history",
        "BETA_STRUCTURE": "beta",
        "FULL_STATE_GRAPH_HISTORY": "full state",
    }
    primary["label"] = primary["modelId"].map(labels)

    fig, axis = plt.subplots(figsize=(9, 5))
    for offset, (candidate, group) in enumerate(primary.groupby("candidateId", sort=True)):
        summary = group.groupby("label", sort=False)["qSpearman"].mean().reindex(labels.values())
        x = np.arange(len(summary)) + (offset - 0.5) * 0.32
        axis.bar(x, summary, width=0.32, label=f"C{candidate[-2:]}")
    axis.axhline(0, color="black", lw=0.8)
    axis.set_xticks(np.arange(len(labels)), labels.values(), rotation=15)
    axis.set_ylabel("F12 joint-q Spearman")
    axis.set_title("Past-observable students versus the empirical committor")
    axis.legend()
    fig.tight_layout()
    fig.savefig(root / "01_overall_q_ranking.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for offset, (candidate, group) in enumerate(primary.groupby("candidateId", sort=True)):
        summary = (
            group.groupby("label", sort=False)["centeredQSpearman"]
            .mean()
            .reindex(labels.values())
        )
        x = np.arange(len(summary)) + (offset - 0.5) * 0.32
        axis.bar(x, summary, width=0.32, label=f"C{candidate[-2:]}")
    axis.axhline(0, color="black", lw=0.8)
    axis.set_xticks(np.arange(len(labels)), labels.values(), rotation=15)
    axis.set_ylabel("Within-matrix centered q Spearman")
    axis.set_title("Longitudinal ordering after matrix means are removed")
    axis.legend()
    fig.tight_layout()
    fig.savefig(root / "02_within_matrix_q_ranking.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for offset, (candidate, group) in enumerate(primary.groupby("candidateId", sort=True)):
        summary = (
            group.groupby("label", sort=False)["equalMatrixMeanBranchLogLoss"]
            .mean()
            .reindex(labels.values())
        )
        x = np.arange(len(summary)) + (offset - 0.5) * 0.32
        axis.bar(x, summary, width=0.32, label=f"C{candidate[-2:]}")
    axis.set_xticks(np.arange(len(labels)), labels.values(), rotation=15)
    axis.set_ylabel("Heldout-half branch log loss")
    axis.set_title("Proper-score comparison on validation matrices")
    axis.legend()
    fig.tight_layout()
    fig.savefig(root / "03_heldout_proper_scores.png", dpi=160)
    plt.close(fig)

    comparisons = tables["model_comparisons.parquet"]
    fig, axis = plt.subplots(figsize=(9, 5))
    plot_rows = comparisons.assign(
        label=lambda frame: frame["candidateId"].str[-2:]
        + " "
        + frame["direction"]
        + " "
        + frame["comparisonId"]
    )
    x = np.arange(len(plot_rows))
    axis.errorbar(
        x,
        plot_rows["logLossImprovement"],
        yerr=np.vstack(
            (
                plot_rows["logLossImprovement"]
                - plot_rows["logLossImprovementLower95"],
                plot_rows["logLossImprovementUpper95"]
                - plot_rows["logLossImprovement"],
            )
        ),
        fmt="o",
    )
    axis.axhline(0, color="black", lw=0.8)
    axis.set_xticks(x, plot_rows["label"], rotation=75, ha="right", fontsize=7)
    axis.set_ylabel("Branch log-loss improvement")
    axis.set_title("Registered matrix-bootstrap comparisons")
    fig.tight_layout()
    fig.savefig(root / "04_registered_comparisons.png", dpi=160)
    plt.close(fig)

    attributions = tables["feature_attribution_results.parquet"]
    selected = attributions[
        attributions["horizon"].eq(PRIMARY_HORIZON)
        & attributions["targetType"].eq(PRIMARY_TARGET)
        & attributions["modelId"].eq("FULL_STATE_GRAPH_HISTORY")
    ]
    top = (
        selected.groupby("featureName")["absoluteCoefficient"]
        .mean()
        .nlargest(15)
        .sort_values()
    )
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(top.index, top.values)
    axis.set_xlabel("Mean absolute backprojected coefficient")
    axis.set_title("Frozen full-state student attribution")
    fig.tight_layout()
    fig.savefig(root / "05_full_state_feature_attribution.png", dpi=160)
    plt.close(fig)

    gates = tables["scientific_gate_results.parquet"]
    gate_matrix = gates[gates["candidateId"].isin(CANDIDATES)].pivot(
        index="gateFamily", columns="candidateId", values="passed"
    )
    fig, axis = plt.subplots(figsize=(7, 3.5))
    image = axis.imshow(gate_matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="RdYlGn")
    axis.set_xticks(
        range(len(gate_matrix.columns)), [f"C{value[-2:]}" for value in gate_matrix.columns]
    )
    axis.set_yticks(range(len(gate_matrix.index)), gate_matrix.index)
    axis.set_title("Past-observable capacity/proxy gates")
    fig.colorbar(image, ax=axis, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(root / "06_scientific_gate_matrix.png", dpi=160)
    plt.close(fig)


def report_text(
    tables: dict[str, pd.DataFrame],
    classifications: list[str],
    next_theme: str,
    runtime: dict[str, Any],
) -> str:
    metrics = tables["predictive_metric_results.parquet"]
    metrics = metrics[
        metrics["horizon"].eq(PRIMARY_HORIZON)
        & metrics["targetType"].eq(PRIMARY_TARGET)
    ]
    comparisons = tables["model_comparisons.parquet"]
    ranks = tables["q_rank_results.parquet"]
    permutations = tables["whole_matrix_permutation_results.parquet"]
    gates = tables["scientific_gate_results.parquet"]
    attribution = tables["feature_attribution_results.parquet"]
    top = (
        attribution[
            attribution["horizon"].eq(PRIMARY_HORIZON)
            & attribution["targetType"].eq(PRIMARY_TARGET)
            & attribution["modelId"].eq("FULL_STATE_GRAPH_HISTORY")
        ]
        .groupby(["candidateId", "featureName"])["absoluteCoefficient"]
        .mean()
        .groupby(level=0, group_keys=False)
        .nlargest(10)
        .reset_index()
    )
    return f"""# S19-L53 Full Results — Past-Observable Regime-Capacity Proxy

## Top summary

- **Research step:** `{VERSION}`
- **Completion status:** complete; additive adaptive exploratory analysis-only evidence
- **Artifacts written:** exact 800-state/beta replay, fixed direct-history, beta-only and target-blind full-state graph features, development-only PCA/ridge models, A-to-B/B-to-A branch-half scoring, F4/F8/F12 break/resumption/joint results, 4,096 matrix bootstraps, 512 whole-matrix permutations, feature attributions, six figures, report and hash manifests
- **Validation:** PASS — immutable S01–L52 baseline; ten fixtures; exact L50 state/outcome replay; exact beta hashes and matrix-constant beta signatures; target-blind graph invariance; development/validation and branch-half separation; two exact feature/model/table passes; runtime, storage and artifact hashes
- **Outcome classification:** {', '.join(f'`{value}`' for value in classifications)}
- **Lay summary:** L52 showed that a few simulated futures reveal a reliable local switching law. L53 asks whether that law can instead be inferred from what is already visible: recent heredity, the catalytic network itself, or the complete present physical state. Stable matrix capacity and changing within-matrix warning are adjudicated separately.
- **Recommended next action:** `{next_theme}` under the bounded autonomous authorization through L65. No L54 work occurs inside L53; S20, E02, author contact, Phi and interventions remain inactive.

## Frozen design

The strict parent/daughter `H>0.9` process, F4/F8/F12 horizons, break, conditional run-3 resumption and joint event are unchanged. Models fit only development matrices with branch half A or B and score validation matrices with the opposite half. The beta-only signature is constant across the five states of one matrix and can support only a stable-capacity interpretation. The full graph is the frozen L34 target-blind representation; no graph layer, feature subset, PCA dimension or regularization value was searched.

## Primary F12 joint-event metrics

{metrics.to_markdown(index=False, floatfmt='.7f')}

## Registered proper-score comparisons

{comparisons.to_markdown(index=False, floatfmt='.7f')}

## Overall and within-matrix ranks

{ranks.to_markdown(index=False, floatfmt='.7f')}

## Whole-matrix permutation controls

{permutations.to_markdown(index=False, floatfmt='.7f')}

## Scientific gates

{gates.to_markdown(index=False, floatfmt='.7f')}

## Highest-weight full-state coordinates

{top.to_markdown(index=False, floatfmt='.7f')}

## Interpretation boundary

An overall beta-only signal means catalytic matrices differ stably in their propensity to break and re-establish heredity; it is not a trajectory-local rise toward a replicator. A full-state result must additionally preserve within-matrix ordering. This adaptive reused-cohort analysis is not confirmation. A null constrains only the fixed direct-history and L34 graph summaries; it does not make the empirical committor unreal or prove that every possible observable is uninformative. Branch-derived L52 models remain forward-simulation ceilings and never enter a past-observable predictor.

## Runtime and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- Workers: `{runtime['workers']}` with one numerical-library thread; GPU hours: 0.
- Wall time: `{runtime['wallSeconds'] / 60:.3f}` minutes; CPU upper estimate: `{runtime['estimatedCpuHours']:.6f}` hours.
- Frozen matrices/states: `{runtime['frozenMatrices']}` / `{runtime['frozenStates']}`.
- New matrices, trajectories and branch streams: 0, 0 and 0.
- Matrix bootstraps: {BOOTSTRAPS}; whole-matrix permutations: {PERMUTATIONS}; exact analysis passes: 2.

## Limitations

The feature students are adaptive follow-ups to L52 and use the same L50 matrix cohort. There are only 40 development and 40 validation matrices, with five correlated states per matrix. PCA and model fitting are development-only, but the representation family was motivated by earlier E01 failures. Beta-only predictions cannot order states within a matrix. The full-state graph is permutation invariant and compact, so species-specific higher-order structure may remain compressed. The stochastic future remains represented only through the response counts.
"""


def manifest_for(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema": "eidosoma.e01.s19_l53.artifact_manifest.v1",
        "loopId": LOOP_ID,
        "files": rows,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def append_ledgers(classifications: list[str], timestamp: str, next_theme: str) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L52 showed that matrix-transfer and exact-state-local duration hazards compress the empirical committor, but both use simulated branch futures.",
            "failureOrAmbiguityTargeted": "Whether stable beta structure or the complete visible current state carries the state-local branch signal.",
            "informationGainRationale": "A fixed hierarchy separates stable catalytic capacity, direct online regime history and current-state network interaction without another broad feature search.",
            "learned": "L53 feature, role, half, model, target, score and gate contract locked before derived outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L50–L52 results and reviewer regime-capacity framing.",
            "proposedNextTest": "Fit fixed development-only past-observable students and score opposite halves on validation matrices.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "stable beta capacity, current physical-state signal or irreducible stochastic shooting information",
            "selectedHypotheses": "The L52 branch teacher may be distillable from beta or current-state observables.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Another Phi variant or unrestricted representation tournament has high leverage.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A stable-capacity gate requires beta-only transfer; a state-local gate additionally requires within-matrix ordering and proper-score gain beyond direct history.",
            "failureOrAmbiguityTargeted": "Past-observable distillation of the empirical process committor.",
            "informationGainRationale": "Opposite branch halves and held-out matrices separate response noise, matrix transfer and within-lineage ordering.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L53 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Registered past-observable information hierarchy.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Any registered L53 gate that failed.",
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
        + f"\n\n## {LOOP_ID} — past-observable regime-capacity proxy\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** `{next_theme}`.\n",
    )
    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    candidate = {
        "branchCount": 0,
        "bundleId": "L53_PAST_OBSERVABLE_REGIME_CAPACITY_PROXY",
        "candidateId": "S19-L53-PAST-OBSERVABLE-REGIME-CAPACITY-PROXY",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 5,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 1,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 2,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "fixed history, beta-only and target-blind full-state graph students on L50 A/B halves",
        "rankingScore": 27.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": "PAST_OBSERVABLE_PROCESS_RISK_LEAD" in classifications,
        "selectionReason": "L52_BRANCH_TEACHER_DISTILLATION",
        "sourceGrounding": 4,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidate_path,
        pd.concat(
            [candidates, pd.DataFrame([candidate]).reindex(columns=candidates.columns)],
            ignore_index=True,
        ),
    )
    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    additions = []
    for row in source_registry().itertuples(index=False):
        additions.append(
            {
                "commitOrVersion": None,
                "evidenceClass": row.evidenceClass,
                "finding": f"{row.finding}; L53 use: {row.frozenUse}",
                "licenseStatus": "PUBLIC_METADATA_OR_WORKSPACE_EVIDENCE",
                "redistributionStatus": "REFERENCE_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L53_{row.sourceId}",
                "sourceType": row.evidenceClass,
                "treeIdentity": None,
                "url": row.url,
            }
        )
    BASE.write_parquet(
        source_path,
        pd.concat(
            [sources, pd.DataFrame(additions).reindex(columns=sources.columns)],
            ignore_index=True,
        ),
    )


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L53 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L53 local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    seeds = analysis_seed_manifest()
    firewall = seed_firewall(seeds)
    sources = source_registry()
    registry = model_registry()
    required_inputs = {
        "l50Manifest": L50_ROOT / "artifact_manifest.json",
        "l50States": L50_ROOT / "restored_state_registry.parquet",
        "l50Outcomes": L50_ROOT / "observed_process_outcomes.parquet",
        "l50Committors": L50_ROOT / "state_committor_results.parquet",
        "l51Manifest": L51_ROOT / "artifact_manifest.json",
        "l51PrefixStates": L51_ROOT / "prefix_state_results.parquet",
        "l52Manifest": L52_ROOT / "artifact_manifest.json",
        "l52Ranks": L52_ROOT / "q_rank_results.parquet",
        "l52Predictions": L52_ROOT / "event_state_metrics.parquet",
    }
    input_validation = pd.DataFrame(
        [
            {
                "inputId": name,
                "path": str(path),
                "sha256": sha256_file(path),
                "exists": path.is_file(),
            }
            for name, path in required_inputs.items()
        ]
    )
    benchmark = {
        "schema": "eidosoma.e01.s19_l53.benchmark_projection.v1",
        "outcomeBlind": True,
        "basis": "800 frozen-state graph features, 108 fixed models, 4,096 matrix bootstraps and 512 permutations",
        "workers": WORKERS,
        "projectedCpuHoursUpper": 20.0,
        "projectedWallHoursUpper": 12.0,
        "status": "PASS",
    }
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or firewall["status"] != "PASS"
        or not input_validation["exists"].all()
        or len(registry) != len(MODELS)
    ):
        raise RuntimeError("L53 preoutcome validation failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L53 decision record\n\n"
        "The human-authorized autonomous sequence through L65 remains active. "
        "L52 established transferable matrix-level and additional exact-state-local "
        "branch-derived duration hazards. Before any L53 response is opened, this "
        "record freezes one past-observable hierarchy: nine direct regime-history/phase "
        "coordinates, twenty beta-only structural coordinates, and the existing L34 "
        "195-coordinate target-blind full-state graph. Exactly twelve development-only "
        "PCA components and one C=0.1 aggregated-binomial ridge contract are used. "
        "Development matrices fit; validation matrices and the opposite branch half "
        "score. Break, conditional resumption and joint risk remain separate at F4/F8/F12. "
        "No new simulation, target, threshold, feature family, architecture, Phi quantity "
        "or intervention is authorized.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_parquet(LOOP_ROOT / "input_identity_validation.parquet", input_validation)
    BASE.write_parquet(LOOP_ROOT / "source_registry.parquet", sources)
    BASE.write_parquet(LOOP_ROOT / "model_registry.parquet", registry)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    BASE.write_json(
        LOOP_ROOT / "source_snapshot_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l53.source_snapshot_manifest.v1",
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "fullStateGraphSha256": sha256_file(
                ROOT / "src/e01_onset_discovery/full_state_graph.py"
            ),
            "configSha256": sha256_file(CONFIG),
            "sources": sources.to_dict("records"),
        },
    )
    locked_inputs = {
        **required_inputs,
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewall": LOOP_ROOT / "seed_firewall.json",
        "inputValidation": LOOP_ROOT / "input_identity_validation.parquet",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "modelRegistry": LOOP_ROOT / "model_registry.parquet",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
    }
    hashes = {name: sha256_file(path) for name, path in locked_inputs.items()}
    implementation = {
        "schema": "eidosoma.e01.s19_l53.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "configSha256": sha256_file(CONFIG),
        "models": list(MODELS),
        "historyColumns": list(HISTORY_COLUMNS),
        "betaFeatureIndices": list(
            beta_structure_indices(feature_names()[PRIMARY_VIEW])
        ),
        "pcaComponents": PCA_COMPONENTS,
        "ridgeC": RIDGE_C,
        "directions": [value[0] for value in DIRECTIONS],
        "horizons": list(HORIZONS),
        "targets": list(TARGETS),
        "matrixBootstraps": BOOTSTRAPS,
        "wholeMatrixPermutations": PERMUTATIONS,
        "workers": WORKERS,
        "newSimulation": False,
        "lockedInputHashes": hashes,
        "outcomeAccessed": False,
        "lockedAtUtc": utc_now(),
    }
    BASE.write_json(LOOP_ROOT / "implementation_lock.json", implementation)
    BASE.write_json(
        LOOP_ROOT / "preoutcome_repository_lock.json",
        {
            "head": head,
            "remote": remote,
            "priorAggregateSha256": prior["aggregateSha256"],
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "configSha256": sha256_file(CONFIG),
            "lockedInputHashes": hashes,
        },
    )


def execute() -> None:
    started = time.perf_counter()
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    if (
        git("rev-parse", "HEAD") != lock["head"]
        or git("rev-parse", "origin/eidosoma/groups/42") != lock["remote"]
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("L53 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_inputs = {
        "l50Manifest": L50_ROOT / "artifact_manifest.json",
        "l50States": L50_ROOT / "restored_state_registry.parquet",
        "l50Outcomes": L50_ROOT / "observed_process_outcomes.parquet",
        "l50Committors": L50_ROOT / "state_committor_results.parquet",
        "l51Manifest": L51_ROOT / "artifact_manifest.json",
        "l51PrefixStates": L51_ROOT / "prefix_state_results.parquet",
        "l52Manifest": L52_ROOT / "artifact_manifest.json",
        "l52Ranks": L52_ROOT / "q_rank_results.parquet",
        "l52Predictions": L52_ROOT / "event_state_metrics.parquet",
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewall": LOOP_ROOT / "seed_firewall.json",
        "inputValidation": LOOP_ROOT / "input_identity_validation.parquet",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "modelRegistry": LOOP_ROOT / "model_registry.parquet",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
    }
    if any(
        sha256_file(path) != lock["lockedInputHashes"][name]
        for name, path in locked_inputs.items()
    ):
        raise RuntimeError("L53 locked input changed")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
        or sha256_file(CONFIG) != lock["configSha256"]
    ):
        raise RuntimeError("L53 pre-execution validation failed")
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    tables, classifications, next_theme = compute_tables()
    make_figures(tables)
    tables_again, classifications_again, next_theme_again = compute_tables()
    table_exact = {
        name: frame_hash(frame) == frame_hash(tables_again[name])
        for name, frame in tables.items()
    }
    regeneration = {
        "schema": "eidosoma.e01.s19_l53.regeneration_validation.v1",
        "status": "PASS"
        if all(table_exact.values())
        and classifications == classifications_again
        and next_theme == next_theme_again
        else "FAIL",
        "tableExact": table_exact,
        "classificationExact": classifications == classifications_again,
        "nextThemeExact": next_theme == next_theme_again,
        "analysisPasses": 2,
    }
    if regeneration["status"] != "PASS":
        raise RuntimeError("L53 regeneration failure")
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l53.classification.v1",
            "classifications": classifications,
            "nextTheme": next_theme,
            "priorStatusesChanged": False,
            "promotableAsConfirmed": False,
            "newMatrices": 0,
            "newPrimaryTrajectories": 0,
            "newBranchStreams": 0,
        },
    )
    pd.DataFrame(
        columns=["failureId", "stage", "status", "reason", "scientificValuesReleased"]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    pd.DataFrame(
        columns=["amendmentId", "status", "scientificContractChanged", "reason"]
    ).to_csv(BUILD_ROOT / "technical_amendment_ledger.csv", index=False)
    elapsed = time.perf_counter() - started
    runtime = {
        "schema": "eidosoma.e01.s19_l53.runtime.v1",
        "repositoryHead": lock["head"],
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": elapsed,
        "estimatedCpuHours": elapsed * WORKERS / 3600,
        "frozenMatrices": 80,
        "frozenStates": 800,
        "frozenBranchSequences": 51200,
        "newMatrices": 0,
        "newPrimaryTrajectories": 0,
        "newBranchStreams": 0,
        "matrixBootstraps": BOOTSTRAPS,
        "wholeMatrixPermutations": PERMUTATIONS,
        "analysisPasses": 2,
        "completedAtUtc": utc_now(),
    }
    if runtime["estimatedCpuHours"] > 34 or runtime["wallSeconds"] > 20 * 3600:
        raise RuntimeError("L53 runtime ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    BASE.write_json(BUILD_ROOT / "regeneration_validation.json", regeneration)
    retained_bytes = sum(
        path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
    ) + sum(path.stat().st_size for path in LOOP_ROOT.iterdir() if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l53.storage_validation.v1",
        "status": "PASS" if retained_bytes <= 15 * 1024**3 else "FAIL",
        "retainedBytes": retained_bytes,
        "retainedGiBCeiling": 15,
        "temporaryGiBCeiling": 30,
    }
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(tables, classifications, next_theme, runtime)
    if report != report_text(tables, classifications, next_theme, runtime):
        raise RuntimeError("L53 report regeneration failure")
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L53_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L53 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Next:** `{next_theme}`.\n",
    )
    if storage["status"] != "PASS":
        raise RuntimeError("L53 storage ceiling exceeded")
    for path in (BUILD_ROOT / "figures").glob("*.png"):
        if not path.stat().st_size:
            raise RuntimeError(f"empty L53 figure: {path}")
    for path in BUILD_ROOT.iterdir():
        destination = LOOP_ROOT / path.name
        if path.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)
    BASE.write_json(LOOP_ROOT / "artifact_manifest.json", manifest_for(LOOP_ROOT))
    if manifest_for(LOOP_ROOT) != json.loads(
        (LOOP_ROOT / "artifact_manifest.json").read_text()
    ):
        raise RuntimeError("L53 artifact manifest regeneration failure")
    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    root_report = (
        f"# S19 current-step report\n\nLatest completed loop: `{LOOP_ID}`.\n\n"
        f"Classification: {', '.join(classifications)}.\n\n"
        f"Next autonomous theme: `{next_theme}`.\n"
    )
    for name in (
        "S19_CURRENT_STEP_REPORT.md",
        "CURRENT_STEP_HANDOFF.md",
        "S19_CURRENT_HANDOFF.md",
        "research_step_full_results.md",
    ):
        BASE.atomic_text(ARTIFACT_ROOT / name, root_report)
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "programStatus": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "nextAuthorizedLoop": "S19-L54",
            "nextTheme": next_theme,
            "authorizationUpperBound": "S19-L65",
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
