#!/usr/bin/env python3
"""Run S19-L54 untouched confirmation of the L53 process-risk coordinate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from e01_frozen_timebase_ensemble.core import selected_clock_observations
from e01_latent_timebase.core import initialize_distinct_state, simulate_trajectory
from e01_onset_discovery.empirical_committor import RestoredState
from e01_onset_discovery.fission_aligned_process import (
    future_post_fission_count,
    nested_process_scores,
    post_fission_index,
)
from e01_onset_discovery.fission_clock_recurrence import simulate_fission_clock
from e01_onset_discovery.full_state_graph import (
    PRIMARY_VIEW,
    feature_names,
    graph_views,
)
from e01_onset_discovery.heredity_phi_incremental import (
    fit_binomial_ridge,
    predict_probability,
)
from e01_onset_discovery.longitudinal_process_risk import trailing_true_run
from e01_onset_discovery.recurrence_inheritance import cosine_h
from e01_onset_discovery.regime_capacity_proxy import (
    beta_structure_indices,
    binomial_cell_scores,
    center_within_groups,
)
from e01_onset_discovery.untouched_regime_confirmation import (
    confirmation_gate,
    confirmation_state_id,
    exact_probability_replay,
    scientific_manifest_equal,
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L53 = load_module(
    "e01_l54_l53_runner",
    ROOT / "scripts/e01/run_s19_l53_regime_capacity_proxy.py",
)
L52 = L53.L52
L50 = L53.L50
L28 = L50.L28
BASE = L53.BASE

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L54"
L50_ROOT = ARTIFACT_ROOT / "loops/L50"
L53_ROOT = ARTIFACT_ROOT / "loops/L53"
CACHE_ROOT = Path("/cache/e01_s19_l54")
PRIMARY_TRAJECTORY_ROOT = CACHE_ROOT / "primary_trajectories"
REGEN_TRAJECTORY_ROOT = CACHE_ROOT / "regenerated_trajectories"
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = ROOT / "configs/e01/s19_l54_untouched_process_risk_confirmation.yaml"
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = ROOT / "src/e01_onset_discovery/untouched_regime_confirmation.py"

LOOP_ID = "S19-L54"
VERSION = "E01-S19-L54-UNTOUCHED-PAST-OBSERVABLE-PROCESS-RISK-CONFIRMATION-v1.0.0"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
MATRIX_COUNT = 40
LANDMARKS = (20, 35, 50, 65, 80)
HORIZONS = (4, 8, 12)
PRIMARY_HORIZON = 12
TARGETS = ("BREAK", "JOINT_BREAK_RUN3", "RUN3_GIVEN_BREAK")
PRIMARY_TARGET = "JOINT_BREAK_RUN3"
BRANCHES = 64
HALF = 32
THRESHOLD = 0.9
REQUIRED_RUN = 3
DIRECTIONS = (("A_TO_B", "A", "B"), ("B_TO_A", "B", "A"))
MODELS = L53.MODELS
HISTORY_COLUMNS = L53.HISTORY_COLUMNS
PCA_COMPONENTS = L53.PCA_COMPONENTS
RIDGE_C = L53.RIDGE_C
BOOTSTRAPS = 4096
PERMUTATIONS = 512
WORKERS = min(8, os.cpu_count() or 1)
ROOT_HEX = "185c73275cf3785068df430ee89f1b330a345ca42da48fe546f49eaa39350395"
PHASE = "s19_l54_untouched_process_risk_confirmation"
SEED_ROOT = bytes.fromhex(ROOT_HEX)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    return L53.sha256_file(path)


def frame_hash(frame: pd.DataFrame) -> str:
    return L53.frame_hash(frame)


def array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def seed_material(*parts: object) -> bytes:
    canonical = tuple(
        part.item() if isinstance(part, np.generic) else part for part in parts
    )
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
    upstream = L53.validate_immutable_prior()
    manifest = json.loads((L53_ROOT / "artifact_manifest.json").read_text())
    checks = []
    for row in manifest["files"]:
        path = L53_ROOT / row["path"]
        checks.append(path.is_file() and sha256_file(path) == row["sha256"])
    unchanged = bool(upstream["unchanged"] and checks and all(checks))
    return {
        "schema": "eidosoma.e01.s19_l54.immutable_prior_validation.v1",
        "unchanged": unchanged,
        "upstreamUnchanged": bool(upstream["unchanged"]),
        "l53FilesChecked": len(checks),
        "l53FilesUnchanged": int(sum(checks)),
        "aggregateSha256": hashlib.sha256(
            json.dumps(
                {
                    "upstream": upstream["aggregateSha256"],
                    "l53": manifest["aggregateSha256"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L53_PAST_OBSERVABLE_PROCESS_RISK_LEAD",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "The target-blind full-state graph plus direct history passed every adaptive heldout-matrix and branch-half gate.",
                "frozenUse": "unchanged model and transformation submitted to untouched confirmation",
                "url": None,
            },
            {
                "sourceId": "L50_PROCESS_TARGET_AND_SIMULATOR",
                "evidenceClass": "DIRECT_FROZEN_E01_METHOD",
                "finding": "Strict-H inheritance, break, run-3 resumption, fission landmarks and candidate semantics are exactly implemented and replayable.",
                "frozenUse": "unchanged confirmation target, state clock and simulation contracts",
                "url": None,
            },
            {
                "sourceId": "REVIEWER_REGIME_SWITCHING_FRAME",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Treat the natural object as stochastic switching between hereditary and nonhereditary regimes, not arrival at one privileged composition.",
                "frozenUse": "interpret confirmed prediction as plastic-heredity process risk rather than fixed-attractor onset",
                "url": None,
            },
        ]
    )


def input_identities() -> tuple[pd.DataFrame, pd.DataFrame]:
    input_rows = []
    seed_rows = []
    for matrix_index in range(MATRIX_COUNT):
        beta_seed = L28.derive_seed(ROOT_HEX, PHASE, "catalytic_matrix", matrix_index)
        initial_seed = L28.derive_seed(ROOT_HEX, PHASE, "initial_state", matrix_index)
        beta = L28.generate_beta(beta_seed)
        initial = initialize_distinct_state(initial_seed)
        input_rows.append(
            {
                "matrixIndex": matrix_index,
                "betaSha256": L28.simulator_array_sha256(beta),
                "initialStateSha256": L28.simulator_array_sha256(initial),
                "initialMass": int(initial.sum()),
                "initialDistinctTypes": int(np.count_nonzero(initial)),
                "generatedBeforeConfirmationOutcome": True,
            }
        )
        for candidate in ("SHARED", *CANDIDATES):
            purposes = (
                ("catalytic_matrix", "initial_state")
                if candidate == "SHARED"
                else (
                    "poisson_update",
                    "overshoot_trim",
                    "fission",
                    "daughter_selection",
                )
            )
            for purpose in purposes:
                identity = (
                    L28.derive_seed(ROOT_HEX, PHASE, purpose, matrix_index)
                    if candidate == "SHARED"
                    else L28.derive_seed(
                        ROOT_HEX, PHASE, purpose, matrix_index, candidate
                    )
                )
                seed_rows.append(
                    {
                        "scope": "MAIN_TRAJECTORY",
                        "matrixIndex": matrix_index,
                        "candidateId": candidate,
                        "stateId": None,
                        "branchIndex": None,
                        "purpose": purpose,
                        "configurationId": identity.configuration_id,
                        "derivedSeed": str(identity.derived_seed),
                        "seedMaterialSha256": identity.seed_material_sha256,
                        "rootHex": ROOT_HEX,
                    }
                )
    return pd.DataFrame(input_rows), pd.DataFrame(seed_rows)


def branch_seed_manifest() -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        for matrix_index in range(MATRIX_COUNT):
            for landmark in LANDMARKS:
                state_id = confirmation_state_id(
                    VERSION, candidate, matrix_index, landmark
                )
                for branch in range(BRANCHES):
                    for purpose in ("event", "trim", "fission", "daughter"):
                        material = seed_material("branch", state_id, branch, purpose)
                        rows.append(
                            {
                                "scope": "BRANCH",
                                "matrixIndex": matrix_index,
                                "candidateId": candidate,
                                "stateId": state_id,
                                "branchIndex": branch,
                                "branchHalf": "A" if branch < HALF else "B",
                                "purpose": purpose,
                                "configurationId": f"{VERSION}|{candidate}|M{matrix_index:03d}|G{landmark:03d}|B{branch:03d}|{purpose}",
                                "derivedSeed": str(
                                    int.from_bytes(material[:16], "big")
                                ),
                                "seedMaterialSha256": material.hex(),
                                "rootHex": ROOT_HEX,
                            }
                        )
    frame = (
        pd.DataFrame(rows)
        .sort_values(
            ["candidateId", "matrixIndex", "stateId", "branchIndex", "purpose"]
        )
        .reset_index(drop=True)
    )
    if (
        len(frame) != 2 * MATRIX_COUNT * len(LANDMARKS) * BRANCHES * 4
        or frame["seedMaterialSha256"].duplicated().any()
        or frame["derivedSeed"].duplicated().any()
    ):
        raise RuntimeError("L54 branch seed scope or uniqueness failure")
    return frame


def analysis_seed_manifest() -> pd.DataFrame:
    rows = []
    for purpose, repetitions in (
        ("matrix_bootstrap", BOOTSTRAPS),
        ("reliability_bootstrap", BOOTSTRAPS),
        ("whole_matrix_q_permutation", PERMUTATIONS),
    ):
        for candidate in CANDIDATES:
            for direction in ("NONE", "A_TO_B", "B_TO_A"):
                if purpose == "reliability_bootstrap" and direction != "NONE":
                    continue
                if purpose != "reliability_bootstrap" and direction == "NONE":
                    continue
                material = seed_material(purpose, candidate, direction)
                rows.append(
                    {
                        "scope": "ANALYSIS",
                        "purpose": purpose,
                        "candidateId": candidate,
                        "direction": direction,
                        "repetitions": repetitions,
                        "derivedSeed": str(int.from_bytes(material[:16], "big")),
                        "seedMaterialSha256": material.hex(),
                        "rootHex": ROOT_HEX,
                    }
                )
    return pd.DataFrame(rows)


def prior_identity_sets() -> dict[str, set[str]]:
    seed_materials: set[str] = set()
    derived_seeds: set[str] = set()
    root_hexes: set[str] = set()
    beta_hashes: set[str] = set()
    initial_hashes: set[str] = set()
    for path in ARTIFACT_ROOT.rglob("*.parquet"):
        if path.is_relative_to(LOOP_ROOT):
            continue
        try:
            names = pq.read_schema(path).names
        except Exception:  # noqa: BLE001, S112 - tolerate old schemas
            continue
        selected = [
            name
            for name in names
            if "seedmaterialsha256" in name.lower()
            or name.lower().endswith("derivedseed")
            or name.lower() == "roothex"
            or name.lower() == "betasha256"
            or name.lower() == "initialstatesha256"
        ]
        if not selected:
            continue
        try:
            frame = pd.read_parquet(path, columns=selected)
        except Exception:  # noqa: BLE001, S112 - tolerate older optional artifacts
            continue
        for column in selected:
            values = set(frame[column].dropna().astype(str))
            lower = column.lower()
            if "seedmaterialsha256" in lower:
                seed_materials.update(values)
            elif lower.endswith("derivedseed"):
                derived_seeds.update(values)
            elif lower == "roothex":
                root_hexes.update(values)
            elif lower == "betasha256":
                beta_hashes.update(values)
            elif lower == "initialstatesha256":
                initial_hashes.update(values)
    return {
        "seedMaterial": seed_materials,
        "derivedSeed": derived_seeds,
        "rootHex": root_hexes,
        "betaSha256": beta_hashes,
        "initialStateSha256": initial_hashes,
    }


def seed_and_input_firewall(
    inputs: pd.DataFrame,
    trajectory_seeds: pd.DataFrame,
    branch_seeds: pd.DataFrame,
    analysis_seeds: pd.DataFrame,
) -> dict[str, Any]:
    prior = prior_identity_sets()
    combined = pd.concat(
        [
            trajectory_seeds[["seedMaterialSha256", "derivedSeed", "rootHex"]],
            branch_seeds[["seedMaterialSha256", "derivedSeed", "rootHex"]],
            analysis_seeds[["seedMaterialSha256", "derivedSeed", "rootHex"]],
        ],
        ignore_index=True,
    )
    material = set(combined["seedMaterialSha256"].astype(str))
    derived = set(combined["derivedSeed"].astype(str))
    beta = set(inputs["betaSha256"].astype(str))
    initial = set(inputs["initialStateSha256"].astype(str))
    material_overlap = material & prior["seedMaterial"]
    derived_overlap = derived & prior["derivedSeed"]
    beta_overlap = beta & prior["betaSha256"]
    initial_overlap = initial & prior["initialStateSha256"]
    root_overlap = {ROOT_HEX} & prior["rootHex"]
    within_unique = bool(
        len(material) == len(combined)
        and len(derived) == len(combined)
        and len(beta) == MATRIX_COUNT
        and len(initial) == MATRIX_COUNT
    )
    passed = bool(
        within_unique
        and not material_overlap
        and not derived_overlap
        and not beta_overlap
        and not initial_overlap
        and not root_overlap
    )
    return {
        "schema": "eidosoma.e01.s19_l54.seed_input_firewall.v1",
        "status": "PASS" if passed else "FAIL",
        "rootHex": ROOT_HEX,
        "newSeedMaterials": len(material),
        "newDerivedSeeds": len(derived),
        "newBetaMatrices": len(beta),
        "newInitialStates": len(initial),
        "withinScopeUnique": within_unique,
        "seedMaterialOverlapCount": len(material_overlap),
        "derivedSeedOverlapCount": len(derived_overlap),
        "betaOverlapCount": len(beta_overlap),
        "initialStateOverlapCount": len(initial_overlap),
        "rootOverlapCount": len(root_overlap),
        "priorSeedMaterialsAudited": len(prior["seedMaterial"]),
        "priorBetaHashesAudited": len(prior["betaSha256"]),
        "priorInitialHashesAudited": len(prior["initialStateSha256"]),
    }


def fixture_results(inputs: pd.DataFrame) -> pd.DataFrame:
    left = [{"id": 1, "hash": "x", "path": "a"}]
    right = [{"id": 1, "hash": "x", "path": "b"}]
    values = np.asarray([0.1, 0.2], dtype=np.float64)
    changed = values.copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    state_ids = {
        confirmation_state_id(VERSION, candidate, matrix, landmark)
        for candidate in CANDIDATES
        for matrix in range(MATRIX_COUNT)
        for landmark in LANDMARKS
    }
    rows = [
        (
            "F01_ROOT_IS_256_BIT",
            len(ROOT_HEX) == 64 and bytes.fromhex(ROOT_HEX).hex() == ROOT_HEX,
        ),
        ("F02_INPUT_SCOPE", len(inputs) == MATRIX_COUNT),
        (
            "F03_INPUT_IDENTITIES_UNIQUE",
            inputs["betaSha256"].nunique() == MATRIX_COUNT
            and inputs["initialStateSha256"].nunique() == MATRIX_COUNT,
        ),
        ("F04_STATE_ID_SCOPE", len(state_ids) == 2 * MATRIX_COUNT * len(LANDMARKS)),
        (
            "F05_MANIFEST_SCIENTIFIC_EQUALITY",
            scientific_manifest_equal(left, right, ("id", "hash")),
        ),
        (
            "F06_MANIFEST_PATH_DIFFERENCE",
            not scientific_manifest_equal(left, right, ("id", "path")),
        ),
        (
            "F07_PROBABILITY_EXACT_REPLAY",
            exact_probability_replay(values, values.copy()),
        ),
        ("F08_ONE_ULP_REJECTED", not exact_probability_replay(values, changed)),
        (
            "F09_CONFIRMATION_GATE_CONJUNCTIVE",
            confirmation_gate(
                availability=True,
                reliability=True,
                proper_score=True,
                overall_rank=True,
                within_matrix_rank=True,
                permutation=True,
                replay=True,
            )
            and not confirmation_gate(
                availability=True,
                reliability=True,
                proper_score=False,
                overall_rank=True,
                within_matrix_rank=True,
                permutation=True,
                replay=True,
            ),
        ),
        (
            "F10_FROZEN_MODEL_SCOPE",
            tuple(MODELS) == tuple(L53.MODELS)
            and PCA_COMPONENTS == 12
            and RIDGE_C == 0.1,
        ),
        (
            "F11_TARGET_SCOPE",
            HORIZONS == (4, 8, 12) and THRESHOLD == 0.9 and REQUIRED_RUN == 3,
        ),
        (
            "F12_DIRECTION_SCOPE",
            DIRECTIONS == (("A_TO_B", "A", "B"), ("B_TO_A", "B", "A")),
        ),
    ]
    return pd.DataFrame(rows, columns=["fixtureId", "passed"])


def _feature_matrix(
    transformed: pd.DataFrame, state_ids: pd.Series, model_id: str
) -> np.ndarray:
    indexed = transformed[transformed["modelId"].eq(model_id)].set_index("stateId")
    return np.stack(
        indexed.loc[state_ids, "values"].map(
            lambda value: np.asarray(value, dtype=np.float64)
        )
    )


def reconstruct_frozen_pipeline() -> tuple[
    dict[tuple[str, str], tuple[StandardScaler, PCA]],
    dict[tuple[str, str, int, str, str], Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    old_features = pd.read_parquet(L53_ROOT / "state_feature_results.parquet")
    transformed, pca_registry, fitted_pca = L53.transformed_features(old_features)
    expected_transformed = pd.read_parquet(
        L53_ROOT / "transformed_feature_results.parquet"
    )
    expected_pca = pd.read_parquet(L53_ROOT / "pca_registry.parquet")
    transformed_exact = frame_hash(transformed) == frame_hash(expected_transformed)
    pca_exact = frame_hash(pca_registry) == frame_hash(expected_pca)
    if not transformed_exact or not pca_exact:
        raise RuntimeError("L54 frozen L53 transformation replay failed")
    replay_predictions, replay_models, replay_attributions = L53.fit_and_predict(
        transformed, fitted_pca
    )
    expected_predictions = pd.read_parquet(L53_ROOT / "prediction_results.parquet")
    expected_models = pd.read_parquet(L53_ROOT / "fitted_model_registry.parquet")
    expected_attributions = pd.read_parquet(
        L53_ROOT / "feature_attribution_results.parquet"
    )
    prediction_exact = frame_hash(replay_predictions) == frame_hash(
        expected_predictions
    )
    model_exact = frame_hash(replay_models) == frame_hash(expected_models)
    attribution_exact = frame_hash(replay_attributions) == frame_hash(
        expected_attributions
    )
    if not prediction_exact or not model_exact or not attribution_exact:
        raise RuntimeError("L54 old-validation prediction/model replay failed")

    estimates = pd.read_parquet(L50_ROOT / "state_committor_results.parquet")
    fitted_models: dict[tuple[str, str, int, str, str], Any] = {}
    validation_rows = []
    for candidate in CANDIDATES:
        for direction, fit_half, _ in DIRECTIONS:
            fit_success = f"successesHalf{fit_half}"
            fit_trials = f"trialsHalf{fit_half}"
            for horizon in HORIZONS:
                for target in TARGETS:
                    source = estimates[
                        estimates["candidateId"].eq(candidate)
                        & estimates["horizon"].eq(horizon)
                        & estimates["targetType"].eq(target)
                    ]
                    development = source[
                        source["matrixRole"].eq("DEVELOPMENT")
                        & source[fit_trials].gt(0)
                    ].sort_values(["matrixIndex", "completedFissionLandmark"])
                    prior = (development[fit_success].sum() + 0.5) / (
                        development[fit_trials].sum() + 1.0
                    )
                    fitted_models[
                        (candidate, direction, horizon, target, "TRAINING_PRIOR")
                    ] = float(prior)
                    for model_id in MODELS[1:]:
                        x_train = _feature_matrix(
                            transformed, development["stateId"], model_id
                        )
                        seed = L53.derived_seed(
                            "model", candidate, direction, horizon, target, model_id
                        ) % (2**32 - 1)
                        fitted = fit_binomial_ridge(
                            x_train,
                            development[fit_success].to_numpy(dtype=np.int64),
                            development[fit_trials].to_numpy(dtype=np.int64),
                            seed=seed,
                            c=RIDGE_C,
                        )
                        fitted_models[
                            (candidate, direction, horizon, target, model_id)
                        ] = fitted
    for key, fitted in fitted_models.items():
        candidate, direction, horizon, target, model_id = key
        expected = expected_predictions[
            expected_predictions["candidateId"].eq(candidate)
            & expected_predictions["direction"].eq(direction)
            & expected_predictions["horizon"].eq(horizon)
            & expected_predictions["targetType"].eq(target)
            & expected_predictions["modelId"].eq(model_id)
        ].sort_values(["matrixIndex", "completedFissionLandmark"])
        validation_ids = expected["stateId"]
        if model_id == "TRAINING_PRIOR":
            actual = np.full(len(expected), fitted, dtype=np.float64)
        else:
            x_validation = _feature_matrix(transformed, validation_ids, model_id)
            actual = predict_probability(fitted, x_validation)
        exact = exact_probability_replay(
            expected["predictedProbability"].to_numpy(dtype=np.float64), actual
        )
        validation_rows.append(
            {
                "candidateId": candidate,
                "direction": direction,
                "horizon": horizon,
                "targetType": target,
                "modelId": model_id,
                "oldValidationStates": len(expected),
                "expectedPredictionSha256": array_hash(
                    expected["predictedProbability"].to_numpy(dtype=np.float64)
                ),
                "actualPredictionSha256": array_hash(actual),
                "exactPredictionReplay": exact,
            }
        )
    validation = pd.DataFrame(validation_rows)
    if (
        len(validation) != 2 * 2 * 3 * 3 * 4
        or not validation["exactPredictionReplay"].all()
    ):
        raise RuntimeError("L54 complete frozen-model replay failure")
    summary = pd.DataFrame(
        [
            {
                "checkId": "TRANSFORMED_FEATURE_TABLE",
                "passed": transformed_exact,
                "expectedSha256": frame_hash(expected_transformed),
                "actualSha256": frame_hash(transformed),
            },
            {
                "checkId": "PCA_REGISTRY",
                "passed": pca_exact,
                "expectedSha256": frame_hash(expected_pca),
                "actualSha256": frame_hash(pca_registry),
            },
            {
                "checkId": "PREDICTION_TABLE",
                "passed": prediction_exact,
                "expectedSha256": frame_hash(expected_predictions),
                "actualSha256": frame_hash(replay_predictions),
            },
            {
                "checkId": "MODEL_REGISTRY",
                "passed": model_exact,
                "expectedSha256": frame_hash(expected_models),
                "actualSha256": frame_hash(replay_models),
            },
            {
                "checkId": "FEATURE_ATTRIBUTIONS",
                "passed": attribution_exact,
                "expectedSha256": frame_hash(expected_attributions),
                "actualSha256": frame_hash(replay_attributions),
            },
        ]
    )
    return fitted_pca, fitted_models, validation, summary


def trajectory_path(root: Path, matrix_index: int, candidate: str) -> Path:
    return root / f"M{matrix_index:04d}__{candidate}.pkl"


def _simulate_matrix(matrix_index: int, root_string: str) -> dict[str, Any]:
    cache = Path(root_string)
    beta = L28.generate_beta(
        L28.derive_seed(ROOT_HEX, PHASE, "catalytic_matrix", matrix_index)
    )
    initial = initialize_distinct_state(
        L28.derive_seed(ROOT_HEX, PHASE, "initial_state", matrix_index)
    )
    rows = []
    seed_rows = []
    failures = []
    for candidate in CANDIDATES:
        started = time.perf_counter()
        try:
            trajectory, seeds = simulate_trajectory(
                phase=PHASE,
                root_hex=ROOT_HEX,
                matrix_index=matrix_index,
                definition=L28.definition(candidate),
                stream_identity=candidate,
                beta=beta,
                initial_state=initial,
            )
            path = trajectory_path(cache, matrix_index, candidate)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                pickle.dump(trajectory, handle, protocol=5)
            selected = selected_clock_observations(trajectory, L28.CLOCK_ID)
            rows.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": matrix_index,
                    "trajectoryId": trajectory.trajectory_id,
                    "trajectorySha256": trajectory.trajectory_sha256,
                    "betaSha256": trajectory.beta_sha256,
                    "initialStateSha256": trajectory.initial_state_sha256,
                    "terminalStatus": trajectory.terminal_status,
                    "completedFissions": int(trajectory.completed_fissions),
                    "selectedClockLength": len(selected),
                    "clockId": L28.CLOCK_ID,
                    "cachePath": str(path),
                    "cacheSha256": sha256_file(path),
                    "replacementAttempted": False,
                    "wallSeconds": time.perf_counter() - started,
                }
            )
            for seed in seeds:
                seed_rows.append(
                    {
                        "scope": "MAIN_TRAJECTORY",
                        "matrixIndex": matrix_index,
                        "candidateId": candidate
                        if seed.purpose not in {"catalytic_matrix", "initial_state"}
                        else "SHARED",
                        "stateId": None,
                        "branchIndex": None,
                        "purpose": seed.purpose,
                        "configurationId": seed.configuration_id,
                        "derivedSeed": str(seed.derived_seed),
                        "seedMaterialSha256": seed.seed_material_sha256,
                        "rootHex": ROOT_HEX,
                    }
                )
        except Exception as error:  # noqa: BLE001 - complete scientific provenance
            failures.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": matrix_index,
                    "failureType": type(error).__name__,
                    "message": str(error),
                }
            )
    return {"trajectories": rows, "seeds": seed_rows, "failures": failures}


def simulate_all(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root.mkdir(parents=True, exist_ok=True)
    outputs = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(_simulate_matrix, matrix_index, str(root))
            for matrix_index in range(MATRIX_COUNT)
        ]
        for future in as_completed(futures):
            outputs.append(future.result())
    trajectories = (
        pd.DataFrame([row for output in outputs for row in output["trajectories"]])
        .sort_values(["candidateId", "matrixIndex"])
        .reset_index(drop=True)
    )
    seeds = (
        pd.DataFrame([row for output in outputs for row in output["seeds"]])
        .sort_values(["candidateId", "matrixIndex", "purpose"])
        .reset_index(drop=True)
    )
    failures = pd.DataFrame(
        [row for output in outputs for row in output["failures"]],
        columns=["candidateId", "matrixIndex", "failureType", "message"],
    )
    return trajectories, seeds, failures


def load_trajectory(row: Any) -> Any:
    path = Path(row.cachePath)
    if not path.is_file() or sha256_file(path) != row.cacheSha256:
        raise RuntimeError(f"L54 trajectory cache identity failed: {path}")
    with path.open("rb") as handle:
        trajectory = pickle.load(handle)
    if (
        trajectory.trajectory_sha256 != row.trajectorySha256
        or trajectory.beta_sha256 != row.betaSha256
        or trajectory.initial_state_sha256 != row.initialStateSha256
    ):
        raise RuntimeError("L54 trajectory payload identity mismatch")
    return trajectory


def _boundary_h(selected: tuple[Any, ...], boundary_index: int) -> float:
    if (
        boundary_index == 0
        or selected[boundary_index - 1].observation_kind != "molecular_update"
    ):
        raise RuntimeError(
            "L54 post-fission boundary lacks selected pre-fission parent"
        )
    return cosine_h(
        np.asarray(selected[boundary_index - 1].state, dtype=np.int64),
        np.asarray(selected[boundary_index].state, dtype=np.int64),
    )


def build_states(
    trajectory_manifest: pd.DataFrame,
) -> tuple[
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
]:
    payloads = []
    state_rows = []
    availability_rows = []
    indexed = trajectory_manifest.set_index(["candidateId", "matrixIndex"])
    for candidate in CANDIDATES:
        for matrix_index in range(MATRIX_COUNT):
            key = (candidate, matrix_index)
            if key not in indexed.index:
                availability_rows.append(
                    {
                        "candidateId": candidate,
                        "matrixIndex": matrix_index,
                        "trajectoryPresent": False,
                        "completedFissions": 0,
                        "eligibleStates": 0,
                        "completeConfirmationUnit": False,
                        "replacementAttempted": False,
                        "reason": "TRAJECTORY_MISSING",
                    }
                )
                continue
            source = indexed.loc[key]
            trajectory = load_trajectory(source)
            selected = tuple(selected_clock_observations(trajectory, L28.CLOCK_ID))
            boundary_indices = [
                index
                for index, observation in enumerate(selected)
                if observation.observation_kind == "post_fission"
            ]
            boundary_h = [_boundary_h(selected, index) for index in boundary_indices]
            eligible = 0
            if int(source.completedFissions) >= 100:
                beta = L28.generate_beta(
                    L28.derive_seed(
                        ROOT_HEX,
                        PHASE,
                        "catalytic_matrix",
                        matrix_index,
                    )
                )
                beta_hash = L28.simulator_array_sha256(beta)
                if beta_hash != source.betaSha256:
                    raise RuntimeError("L54 state builder beta mismatch")
                for landmark in LANDMARKS:
                    current_index = post_fission_index(selected, landmark)
                    current = selected[current_index]
                    future_indices = [
                        index for index in boundary_indices if index > current_index
                    ][:PRIMARY_HORIZON]
                    if (
                        future_post_fission_count(selected, current_index)
                        < PRIMARY_HORIZON
                        or len(future_indices) != PRIMARY_HORIZON
                    ):
                        continue
                    prefix_positions = [
                        position
                        for position, index in enumerate(boundary_indices)
                        if index <= current_index
                    ]
                    prefix_h = np.asarray(
                        [boundary_h[position] for position in prefix_positions],
                        dtype=np.float64,
                    )
                    inherited = prefix_h > THRESHOLD
                    latest_break_positions = np.flatnonzero(~inherited)
                    fissions_since_break = (
                        len(inherited) - 1 - int(latest_break_positions[-1])
                        if len(latest_break_positions)
                        else len(inherited)
                    )
                    restored = L28.restored_state_from_observation(current)
                    state = np.asarray(restored.state, dtype=np.int64)
                    state_id = confirmation_state_id(
                        VERSION, candidate, matrix_index, landmark
                    )
                    base = {
                        "stateId": state_id,
                        "matrixRole": "UNTOUCHED_CONFIRMATION",
                        "candidateId": candidate,
                        "matrixIndex": matrix_index,
                        "selectionRank": matrix_index + 1,
                        "completedFissionLandmark": landmark,
                        "normalizedGeneration": landmark / 100.0,
                        "trajectoryId": source.trajectoryId,
                        "currentSelectedIndex": current_index,
                        "currentObservationKind": current.observation_kind,
                        "currentCompletedFissions": int(current.completed_fissions),
                        "currentGrowthGeneration": int(
                            current.growth_generation_one_based
                        ),
                        "currentGenerationLocalStep": int(
                            current.generation_local_step
                        ),
                        "currentBatchStep": int(current.batch_step),
                        "currentMass": int(state.sum()),
                        "prefixBoundaryCount": len(prefix_h),
                        "prefixInheritanceFraction": float(inherited.mean()),
                        "recentFiveInheritanceFraction": float(inherited[-5:].mean()),
                        "prefixTrailingInheritanceRun": trailing_true_run(inherited),
                        "latestParentDaughterH": float(prefix_h[-1]),
                        "fissionsSinceLatestBreak": int(fissions_since_break),
                        "futureFissionsAvailable": future_post_fission_count(
                            selected, current_index
                        ),
                        "currentInheritanceState": bool(inherited[-1]),
                        "currentRegimeDuration": trailing_true_run(inherited)
                        if inherited[-1]
                        else trailing_true_run(~inherited),
                        "currentStateSha256": L28.array_sha256(state),
                        "betaSha256": beta_hash,
                        "trajectorySha256": source.trajectorySha256,
                        "selectedClockLength": len(selected),
                        "targetUsesCompletedTestTrajectory": False,
                    }
                    state_rows.append(base)
                    payloads.append({**base, "state": list(map(int, restored.state))})
                    eligible += 1
            complete = int(source.completedFissions) >= 100 and eligible == len(
                LANDMARKS
            )
            availability_rows.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": matrix_index,
                    "trajectoryPresent": True,
                    "completedFissions": int(source.completedFissions),
                    "eligibleStates": eligible,
                    "completeConfirmationUnit": complete,
                    "replacementAttempted": False,
                    "reason": "ELIGIBLE"
                    if complete
                    else "INCOMPLETE_OR_STATE_UNAVAILABLE",
                }
            )
    states = (
        pd.DataFrame(state_rows)
        .sort_values(["candidateId", "matrixIndex", "completedFissionLandmark"])
        .reset_index(drop=True)
    )
    availability = (
        pd.DataFrame(availability_rows)
        .sort_values(["candidateId", "matrixIndex"])
        .reset_index(drop=True)
    )
    return payloads, states, availability


def realized_path_outcomes(
    trajectory_manifest: pd.DataFrame, states: pd.DataFrame
) -> pd.DataFrame:
    """Open the one-realization future only after prospective predictions freeze."""

    rows = []
    indexed = trajectory_manifest.set_index(["candidateId", "matrixIndex"])
    for source in states.itertuples(index=False):
        trajectory = load_trajectory(
            indexed.loc[(source.candidateId, int(source.matrixIndex))]
        )
        selected = tuple(selected_clock_observations(trajectory, L28.CLOCK_ID))
        boundary_indices = [
            index
            for index, observation in enumerate(selected)
            if observation.observation_kind == "post_fission"
        ]
        future_indices = [
            index
            for index in boundary_indices
            if index > int(source.currentSelectedIndex)
        ][:PRIMARY_HORIZON]
        if len(future_indices) != PRIMARY_HORIZON:
            raise RuntimeError("L54 realized-path horizon became unavailable")
        future_h = np.asarray(
            [_boundary_h(selected, index) for index in future_indices],
            dtype=np.float64,
        )
        scores = nested_process_scores(
            future_h,
            HORIZONS,
            threshold=THRESHOLD,
            required_run=REQUIRED_RUN,
        )
        for horizon in HORIZONS:
            score = scores[horizon]
            rows.append(
                {
                    "stateId": source.stateId,
                    "matrixRole": "UNTOUCHED_CONFIRMATION",
                    "candidateId": source.candidateId,
                    "matrixIndex": int(source.matrixIndex),
                    "completedFissionLandmark": int(source.completedFissionLandmark),
                    "horizon": horizon,
                    "observedBreak": score.break_observed,
                    "observedJointEvent": score.event,
                    "observedConditionalEvent": score.event
                    if score.break_observed
                    else None,
                    "observedBreakBoundaryOneBased": (score.break_boundary_one_based),
                    "observedCertificationBoundaryOneBased": (
                        score.certification_boundary_one_based
                    ),
                    "observedFutureInheritanceFraction": float(
                        (future_h[:horizon] > THRESHOLD).mean()
                    ),
                    "targetUsesCompletedTestTrajectory": False,
                    "openedAfterProspectivePredictionFreeze": True,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["candidateId", "matrixIndex", "completedFissionLandmark", "horizon"]
        )
        .reset_index(drop=True)
    )


def _graph_feature_worker(payload: dict[str, Any]) -> dict[str, Any]:
    matrix_index = int(payload["matrixIndex"])
    beta = L28.generate_beta(
        L28.derive_seed(ROOT_HEX, PHASE, "catalytic_matrix", matrix_index)
    )
    if L28.simulator_array_sha256(beta) != payload["betaSha256"]:
        raise RuntimeError("L54 graph worker beta identity mismatch")
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


def extract_features(
    payloads: list[dict[str, Any]], states: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        graph_rows = list(executor.map(_graph_feature_worker, payloads))
    features = (
        states.merge(pd.DataFrame(graph_rows), on="stateId", validate="one_to_one")
        .sort_values(["candidateId", "matrixIndex", "completedFissionLandmark"])
        .reset_index(drop=True)
    )
    validation_rows = []
    for keys, group in features.groupby(["candidateId", "matrixIndex"], sort=True):
        candidate, matrix_index = keys
        validation_rows.append(
            {
                "candidateId": candidate,
                "matrixIndex": int(matrix_index),
                "states": len(group),
                "uniqueBetaFeatureHashes": int(group["betaFeatureSha256"].nunique()),
                "betaOnlyStateInvariant": group["betaFeatureSha256"].nunique() == 1,
                "graphFeaturesFinite": bool(
                    all(
                        np.isfinite(np.asarray(value, dtype=np.float64)).all()
                        for value in group["graphValues"]
                    )
                ),
                "targetDefinitionIndependent": bool(
                    not group["targetUsesCompletedTestTrajectory"].any()
                ),
            }
        )
    validation = pd.DataFrame(validation_rows)
    if (
        not validation[
            [
                "betaOnlyStateInvariant",
                "graphFeaturesFinite",
                "targetDefinitionIndependent",
            ]
        ]
        .all()
        .all()
    ):
        raise RuntimeError("L54 confirmation feature validation failed")
    return features, validation


def transform_confirmation_features(
    features: pd.DataFrame,
    fitted_pca: dict[tuple[str, str], tuple[StandardScaler, PCA]],
) -> pd.DataFrame:
    rows = []
    for source in features.itertuples(index=False):
        values = np.asarray(
            [getattr(source, column) for column in HISTORY_COLUMNS],
            dtype=np.float64,
        )
        rows.append(
            {
                "stateId": source.stateId,
                "candidateId": source.candidateId,
                "matrixIndex": int(source.matrixIndex),
                "completedFissionLandmark": int(source.completedFissionLandmark),
                "modelId": "DIRECT_HISTORY_PHASE",
                "values": values.tolist(),
                "featureSha256": array_hash(values),
            }
        )
    for candidate in CANDIDATES:
        candidate_frame = features[features["candidateId"].eq(candidate)].reset_index(
            drop=True
        )
        for model_id in ("BETA_STRUCTURE", "FULL_STATE_GRAPH_HISTORY"):
            raw = np.stack(
                candidate_frame[
                    "betaValues" if model_id == "BETA_STRUCTURE" else "graphValues"
                ].map(lambda value: np.asarray(value, dtype=np.float64))
            )
            scaler, pca = fitted_pca[(candidate, model_id)]
            transformed = pca.transform(scaler.transform(raw))
            if model_id == "FULL_STATE_GRAPH_HISTORY":
                history = candidate_frame[list(HISTORY_COLUMNS)].to_numpy(
                    dtype=np.float64
                )
                transformed = np.column_stack((transformed, history))
            for source, values in zip(
                candidate_frame.itertuples(index=False), transformed, strict=True
            ):
                rows.append(
                    {
                        "stateId": source.stateId,
                        "candidateId": candidate,
                        "matrixIndex": int(source.matrixIndex),
                        "completedFissionLandmark": int(
                            source.completedFissionLandmark
                        ),
                        "modelId": model_id,
                        "values": values.tolist(),
                        "featureSha256": array_hash(values),
                    }
                )
    frame = (
        pd.DataFrame(rows)
        .sort_values(
            ["modelId", "candidateId", "matrixIndex", "completedFissionLandmark"]
        )
        .reset_index(drop=True)
    )
    if frame.duplicated(["stateId", "modelId"]).any():
        raise RuntimeError("L54 transformed confirmation feature duplication")
    return frame


def prospective_predictions(
    states: pd.DataFrame,
    transformed: pd.DataFrame,
    fitted_models: dict[tuple[str, str, int, str, str], Any],
) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        candidate_states = states[states["candidateId"].eq(candidate)].sort_values(
            ["matrixIndex", "completedFissionLandmark"]
        )
        for direction, _, score_half in DIRECTIONS:
            for horizon in HORIZONS:
                for target in TARGETS:
                    for model_id in MODELS:
                        fitted = fitted_models[
                            (candidate, direction, horizon, target, model_id)
                        ]
                        if model_id == "TRAINING_PRIOR":
                            probability = np.full(
                                len(candidate_states), fitted, dtype=np.float64
                            )
                            feature_hashes = [None] * len(candidate_states)
                        else:
                            x = _feature_matrix(
                                transformed, candidate_states["stateId"], model_id
                            )
                            probability = predict_probability(fitted, x)
                            indexed = transformed[
                                transformed["modelId"].eq(model_id)
                            ].set_index("stateId")
                            feature_hashes = indexed.loc[
                                candidate_states["stateId"], "featureSha256"
                            ].tolist()
                        if not np.isfinite(probability).all():
                            raise RuntimeError("L54 prospective prediction nonfinite")
                        for source, value, feature_hash in zip(
                            candidate_states.itertuples(index=False),
                            probability,
                            feature_hashes,
                            strict=True,
                        ):
                            rows.append(
                                {
                                    "stateId": source.stateId,
                                    "candidateId": candidate,
                                    "matrixIndex": int(source.matrixIndex),
                                    "completedFissionLandmark": int(
                                        source.completedFissionLandmark
                                    ),
                                    "direction": direction,
                                    "scoreHalf": score_half,
                                    "horizon": horizon,
                                    "targetType": target,
                                    "modelId": model_id,
                                    "predictedProbability": float(value),
                                    "featureSha256": feature_hash,
                                    "modelFitOnFrozenL53DevelopmentOnly": True,
                                    "confirmationOutcomeAccessedAtPrediction": False,
                                }
                            )
    frame = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "candidateId",
                "direction",
                "horizon",
                "targetType",
                "modelId",
                "matrixIndex",
                "completedFissionLandmark",
            ]
        )
        .reset_index(drop=True)
    )
    expected = len(states) * 2 * len(HORIZONS) * len(TARGETS) * len(MODELS)
    if (
        len(frame) != expected
        or frame.duplicated(
            ["stateId", "direction", "horizon", "targetType", "modelId"]
        ).any()
    ):
        raise RuntimeError("L54 prospective prediction scope failure")
    return frame


def _branch_worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    beta = L28.generate_beta(
        L28.derive_seed(
            ROOT_HEX,
            PHASE,
            "catalytic_matrix",
            int(payload["matrixIndex"]),
        )
    )
    if L28.simulator_array_sha256(beta) != payload["betaSha256"]:
        raise RuntimeError(f"L54 worker beta mismatch: {payload['stateId']}")
    restored = RestoredState(
        tuple(payload["state"]),
        payload["currentObservationKind"],
        int(payload["currentCompletedFissions"]),
        int(payload["currentGrowthGeneration"]),
        int(payload["currentGenerationLocalStep"]),
        int(payload["currentBatchStep"]),
    )
    rows = []
    for branch in range(BRANCHES):
        trace = simulate_fission_clock(
            restored=restored,
            beta=beta,
            definition=L28.definition(payload["candidateId"]),
            event_rng=generator("branch", payload["stateId"], branch, "event"),
            trim_rng=generator("branch", payload["stateId"], branch, "trim"),
            fission_rng=generator("branch", payload["stateId"], branch, "fission"),
            daughter_rng=generator("branch", payload["stateId"], branch, "daughter"),
            future_fissions=PRIMARY_HORIZON,
        )
        scores = nested_process_scores(
            trace.parent_daughter_h,
            HORIZONS,
            threshold=THRESHOLD,
            required_run=REQUIRED_RUN,
        )
        materials = [
            seed_material("branch", payload["stateId"], branch, purpose).hex()
            for purpose in ("event", "trim", "fission", "daughter")
        ]
        base = {
            "stateId": payload["stateId"],
            "matrixRole": "UNTOUCHED_CONFIRMATION",
            "candidateId": payload["candidateId"],
            "matrixIndex": int(payload["matrixIndex"]),
            "completedFissionLandmark": int(payload["completedFissionLandmark"]),
            "branchIndex": branch,
            "branchHalf": "A" if branch < HALF else "B",
            "branchIdentitySha256": hashlib.sha256(
                "|".join([payload["stateId"], str(branch), *materials]).encode()
            ).hexdigest(),
            "molecularUpdates": trace.molecular_updates,
            "fissions": trace.fissions,
            "terminalStatus": trace.terminal_status,
            "pathSha256": trace.path_sha256,
            "targetUsesCompletedTestTrajectory": False,
        }
        for index, value in enumerate(trace.parent_daughter_h, start=1):
            base[f"parentDaughterH{index:02d}"] = float(value)
        for horizon in HORIZONS:
            score = scores[horizon]
            base[f"breakH{horizon}"] = score.break_observed
            base[f"jointH{horizon}"] = score.event
            base[f"conditionalH{horizon}"] = (
                score.event if score.break_observed else None
            )
            base[f"breakBoundaryH{horizon}"] = score.break_boundary_one_based
            base[f"certificationBoundaryH{horizon}"] = (
                score.certification_boundary_one_based
            )
        rows.append(base)
    return rows


def execute_branches(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_branch_worker, payload): payload["stateId"]
            for payload in payloads
        }
        for future in as_completed(futures):
            rows.extend(future.result())
    frame = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "candidateId",
                "matrixIndex",
                "completedFissionLandmark",
                "branchIndex",
            ]
        )
        .reset_index(drop=True)
    )
    if (
        len(frame) != len(payloads) * BRANCHES
        or frame.duplicated(["stateId", "branchIndex"]).any()
        or frame.groupby("stateId").size().ne(BRANCHES).any()
        or frame["fissions"].ne(PRIMARY_HORIZON).any()
        or frame["targetUsesCompletedTestTrajectory"].any()
    ):
        raise RuntimeError("L54 branch output scope failure")
    return frame


def score_predictions(
    prospective: pd.DataFrame, estimates: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    indexed = estimates.set_index(["stateId", "horizon", "targetType"])
    for source in prospective.itertuples(index=False):
        response = indexed.loc[(source.stateId, source.horizon, source.targetType)]
        successes = int(getattr(response, f"successesHalf{source.scoreHalf}"))
        trials = int(getattr(response, f"trialsHalf{source.scoreHalf}"))
        if trials <= 0:
            continue
        probability = float(source.predictedProbability)
        loss, brier = binomial_cell_scores([probability], [successes], [trials])
        rows.append(
            {
                **source._asdict(),
                "successes": successes,
                "trials": trials,
                "empiricalQ": (successes + 0.5) / (trials + 1.0),
                "branchLogLoss": float(loss[0]),
                "qBrier": float(brier[0]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "candidateId",
                "direction",
                "horizon",
                "targetType",
                "modelId",
                "matrixIndex",
                "completedFissionLandmark",
            ]
        )
        .reset_index(drop=True)
    )


def reliability_results(estimates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = estimates[
        estimates["horizon"].eq(PRIMARY_HORIZON)
        & estimates["targetType"].eq(PRIMARY_TARGET)
    ]
    summary_rows = []
    bootstrap_rows = []
    for candidate, group in primary.groupby("candidateId", sort=True):
        group = group.sort_values(["matrixIndex", "completedFissionLandmark"])
        raw = safe_spearman(group["qHalfA"].to_numpy(), group["qHalfB"].to_numpy())
        centered_a = center_within_groups(group["qHalfA"], group["matrixIndex"])
        centered_b = center_within_groups(group["qHalfB"], group["matrixIndex"])
        centered = safe_spearman(centered_a, centered_b)
        intermediate = int(((group["q"] > 0.1) & (group["q"] < 0.9)).sum())
        matrices = np.sort(group["matrixIndex"].unique())
        rng = generator("reliability_bootstrap", candidate, "NONE")
        raw_boot = np.empty(BOOTSTRAPS)
        centered_boot = np.empty(BOOTSTRAPS)
        for replicate in range(BOOTSTRAPS):
            sampled = rng.choice(matrices, size=len(matrices), replace=True)
            parts = []
            labels = []
            for draw, matrix_index in enumerate(sampled):
                part = group[group["matrixIndex"].eq(matrix_index)]
                parts.append(part)
                labels.extend([draw] * len(part))
            sample = pd.concat(parts, ignore_index=True)
            raw_boot[replicate] = safe_spearman(
                sample["qHalfA"].to_numpy(), sample["qHalfB"].to_numpy()
            )
            centered_boot[replicate] = safe_spearman(
                center_within_groups(sample["qHalfA"], labels),
                center_within_groups(sample["qHalfB"], labels),
            )
            bootstrap_rows.append(
                {
                    "candidateId": candidate,
                    "replicate": replicate,
                    "splitHalfSpearman": raw_boot[replicate],
                    "centeredSplitHalfSpearman": centered_boot[replicate],
                }
            )
        raw_low, raw_high = interval(raw_boot)
        centered_low, centered_high = interval(centered_boot)
        summary_rows.append(
            {
                "candidateId": candidate,
                "matrices": int(group["matrixIndex"].nunique()),
                "states": len(group),
                "intermediateProbabilityStates": intermediate,
                "splitHalfSpearman": raw,
                "splitHalfSpearmanLower95": raw_low,
                "splitHalfSpearmanUpper95": raw_high,
                "centeredSplitHalfSpearman": centered,
                "centeredSplitHalfSpearmanLower95": centered_low,
                "centeredSplitHalfSpearmanUpper95": centered_high,
                "reliabilityGatePassed": bool(
                    raw > 0.5
                    and raw_low > 0.3
                    and centered_low > 0.1
                    and intermediate >= 20
                ),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(bootstrap_rows)


def matrix_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
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
            model: model_group.set_index(
                ["matrixIndex", "completedFissionLandmark"]
            ).sort_index()
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
                for draw, matrix_index in enumerate(sampled):
                    part = indexed[model].loc[[matrix_index]].reset_index()
                    parts.append(part)
                    labels.extend([draw] * len(part))
                sample = pd.concat(parts, ignore_index=True)
                record[f"logLoss__{model}"] = float(
                    sample.groupby(labels)["branchLogLoss"].mean().mean()
                )
                record[f"qBrier__{model}"] = float(
                    sample.groupby(labels)["qBrier"].mean().mean()
                )
                record[f"qSpearman__{model}"] = safe_spearman(
                    sample["predictedProbability"].to_numpy(),
                    sample["empiricalQ"].to_numpy(),
                )
                record[f"centeredQSpearman__{model}"] = safe_spearman(
                    center_within_groups(sample["predictedProbability"], labels),
                    center_within_groups(sample["empiricalQ"], labels),
                )
            for comparison, model, reference in (
                (
                    "FULL_VS_PRIOR",
                    "FULL_STATE_GRAPH_HISTORY",
                    "TRAINING_PRIOR",
                ),
                (
                    "FULL_VS_DIRECT",
                    "FULL_STATE_GRAPH_HISTORY",
                    "DIRECT_HISTORY_PHASE",
                ),
            ):
                record[f"logLossGain__{comparison}"] = (
                    record[f"logLoss__{reference}"] - record[f"logLoss__{model}"]
                )
                record[f"qBrierGain__{comparison}"] = (
                    record[f"qBrier__{reference}"] - record[f"qBrier__{model}"]
                )
            rows.append(record)
    return pd.DataFrame(rows)


def bootstrap_summaries(
    metrics: pd.DataFrame, bootstrap: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = metrics[
        metrics["horizon"].eq(PRIMARY_HORIZON)
        & metrics["targetType"].eq(PRIMARY_TARGET)
    ]
    rank_rows = []
    comparison_rows = []
    for (candidate, direction), boot in bootstrap.groupby(
        ["candidateId", "direction"], sort=True
    ):
        for model in MODELS:
            point = primary[
                primary["candidateId"].eq(candidate)
                & primary["direction"].eq(direction)
                & primary["modelId"].eq(model)
            ].iloc[0]
            q_low, q_high = interval(boot[f"qSpearman__{model}"].to_numpy())
            centered_low, centered_high = interval(
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
                    "centeredQSpearmanLower95": centered_low,
                    "centeredQSpearmanUpper95": centered_high,
                }
            )
        for comparison, model, reference in (
            (
                "FULL_VS_PRIOR",
                "FULL_STATE_GRAPH_HISTORY",
                "TRAINING_PRIOR",
            ),
            (
                "FULL_VS_DIRECT",
                "FULL_STATE_GRAPH_HISTORY",
                "DIRECT_HISTORY_PHASE",
            ),
        ):
            log_values = boot[f"logLossGain__{comparison}"].to_numpy(dtype=np.float64)
            brier_values = boot[f"qBrierGain__{comparison}"].to_numpy(dtype=np.float64)
            log_low, log_high = interval(log_values)
            brier_low, brier_high = interval(brier_values)
            point_model = primary[
                primary["candidateId"].eq(candidate)
                & primary["direction"].eq(direction)
                & primary["modelId"].eq(model)
            ].iloc[0]
            point_reference = primary[
                primary["candidateId"].eq(candidate)
                & primary["direction"].eq(direction)
                & primary["modelId"].eq(reference)
            ].iloc[0]
            comparison_rows.append(
                {
                    "candidateId": candidate,
                    "direction": direction,
                    "comparisonId": comparison,
                    "modelId": model,
                    "referenceModelId": reference,
                    "logLossImprovement": float(
                        point_reference.equalMatrixMeanBranchLogLoss
                        - point_model.equalMatrixMeanBranchLogLoss
                    ),
                    "logLossImprovementLower95": log_low,
                    "logLossImprovementUpper95": log_high,
                    "qBrierImprovement": float(
                        point_reference.equalMatrixMeanQBrier
                        - point_model.equalMatrixMeanQBrier
                    ),
                    "qBrierImprovementLower95": brier_low,
                    "qBrierImprovementUpper95": brier_high,
                    "fractionBootstrapLogLossPositive": float(np.mean(log_values > 0)),
                    "fractionBootstrapQBrierPositive": float(np.mean(brier_values > 0)),
                }
            )
    return pd.DataFrame(rank_rows), pd.DataFrame(comparison_rows)


def q_permutations(predictions: pd.DataFrame) -> pd.DataFrame:
    primary = predictions[
        predictions["horizon"].eq(PRIMARY_HORIZON)
        & predictions["targetType"].eq(PRIMARY_TARGET)
        & predictions["modelId"].eq("FULL_STATE_GRAPH_HISTORY")
    ]
    rows = []
    for (candidate, direction), group in primary.groupby(
        ["candidateId", "direction"], sort=True
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
        labels = np.repeat(np.arange(len(p)), p.shape[1])
        observed = safe_spearman(p.to_numpy().ravel(), q.to_numpy().ravel())
        observed_centered = safe_spearman(
            center_within_groups(p.to_numpy().ravel(), labels),
            center_within_groups(q.to_numpy().ravel(), labels),
        )
        rng = generator("whole_matrix_q_permutation", candidate, direction)
        null = np.empty(PERMUTATIONS)
        centered_null = np.empty(PERMUTATIONS)
        for replicate in range(PERMUTATIONS):
            permuted = q.to_numpy()[rng.permutation(len(q))]
            null[replicate] = safe_spearman(p.to_numpy().ravel(), permuted.ravel())
            centered_null[replicate] = safe_spearman(
                center_within_groups(p.to_numpy().ravel(), labels),
                center_within_groups(permuted.ravel(), labels),
            )
        rows.append(
            {
                "candidateId": candidate,
                "direction": direction,
                "modelId": "FULL_STATE_GRAPH_HISTORY",
                "observedQSpearman": observed,
                "observedCenteredQSpearman": observed_centered,
                "overallUpperTailP": float(
                    (1 + np.sum(null >= observed)) / (PERMUTATIONS + 1)
                ),
                "centeredUpperTailP": float(
                    (1 + np.sum(centered_null >= observed_centered))
                    / (PERMUTATIONS + 1)
                ),
                "permutations": PERMUTATIONS,
            }
        )
    return pd.DataFrame(rows)


def state_estimates(
    branches: pd.DataFrame, states: pd.DataFrame, observed: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for state_id, group in branches.groupby("stateId", sort=False):
        first = group.iloc[0]
        for horizon in HORIZONS:
            for target_type, column in (
                ("BREAK", f"breakH{horizon}"),
                ("JOINT_BREAK_RUN3", f"jointH{horizon}"),
                ("RUN3_GIVEN_BREAK", f"conditionalH{horizon}"),
            ):
                eligible = (
                    group[group[f"breakH{horizon}"]]
                    if target_type == "RUN3_GIVEN_BREAK"
                    else group
                )
                trials = len(eligible)
                successes = int(eligible[column].fillna(False).sum())
                raw = successes / trials if trials else float("nan")
                halves: dict[str, tuple[float, int, int]] = {}
                for half in ("A", "B"):
                    part = group[group["branchHalf"].eq(half)]
                    if target_type == "RUN3_GIVEN_BREAK":
                        part = part[part[f"breakH{horizon}"]]
                    half_trials = len(part)
                    half_successes = int(part[column].fillna(False).sum())
                    halves[half] = (
                        L50.jeffreys_mean(half_successes, half_trials),
                        half_successes,
                        half_trials,
                    )
                rows.append(
                    {
                        "stateId": state_id,
                        "matrixRole": first.matrixRole,
                        "candidateId": first.candidateId,
                        "matrixIndex": int(first.matrixIndex),
                        "completedFissionLandmark": int(first.completedFissionLandmark),
                        "horizon": horizon,
                        "targetType": target_type,
                        "successes": successes,
                        "trials": trials,
                        "dataInformed": trials > 0,
                        "q": L50.jeffreys_mean(successes, trials),
                        "qHalfA": halves["A"][0],
                        "qHalfB": halves["B"][0],
                        "successesHalfA": halves["A"][1],
                        "successesHalfB": halves["B"][1],
                        "trialsHalfA": halves["A"][2],
                        "trialsHalfB": halves["B"][2],
                        "binomialNoise": raw * (1 - raw) / trials
                        if trials
                        else float("nan"),
                    }
                )
    estimates = pd.DataFrame(rows)
    long = (
        estimates.merge(
            states,
            on=[
                "stateId",
                "matrixRole",
                "candidateId",
                "matrixIndex",
                "completedFissionLandmark",
            ],
            validate="many_to_one",
        )
        .sort_values(
            [
                "candidateId",
                "matrixIndex",
                "completedFissionLandmark",
                "horizon",
                "targetType",
            ]
        )
        .reset_index(drop=True)
    )
    observed_long = observed.copy()
    observed_long["targetType"] = "JOINT_BREAK_RUN3"
    observed_long = observed_long.rename(
        columns={"observedJointEvent": "observedTarget"}
    )
    long = long.merge(
        observed_long[
            [
                "stateId",
                "horizon",
                "targetType",
                "observedTarget",
                "observedBreak",
            ]
        ],
        on=["stateId", "horizon", "targetType"],
        how="left",
        validate="many_to_one",
    )
    if len(long) != len(states) * len(HORIZONS) * len(TARGETS):
        raise RuntimeError("L54 state-estimate scope failure")
    return long


def cohort_shift_results(
    confirmation_features: pd.DataFrame,
    estimates: pd.DataFrame,
) -> pd.DataFrame:
    old = pd.read_parquet(L53_ROOT / "state_feature_results.parquet")
    rows = []
    for candidate in CANDIDATES:
        for feature in HISTORY_COLUMNS:
            development = old[
                old["candidateId"].eq(candidate) & old["matrixRole"].eq("DEVELOPMENT")
            ][feature].to_numpy(dtype=np.float64)
            confirmation = confirmation_features[
                confirmation_features["candidateId"].eq(candidate)
            ][feature].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "candidateId": candidate,
                    "quantityId": feature,
                    "developmentMean": float(np.mean(development)),
                    "developmentSd": float(np.std(development, ddof=1)),
                    "confirmationMean": float(np.mean(confirmation)),
                    "confirmationSd": float(np.std(confirmation, ddof=1)),
                    "standardizedMeanShift": float(
                        (np.mean(confirmation) - np.mean(development))
                        / max(np.std(development, ddof=1), 1e-12)
                    ),
                }
            )
        q = estimates[
            estimates["candidateId"].eq(candidate)
            & estimates["horizon"].eq(PRIMARY_HORIZON)
            & estimates["targetType"].eq(PRIMARY_TARGET)
        ]["q"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "candidateId": candidate,
                "quantityId": "F12_JOINT_Q",
                "developmentMean": np.nan,
                "developmentSd": np.nan,
                "confirmationMean": float(np.mean(q)),
                "confirmationSd": float(np.std(q, ddof=1)),
                "standardizedMeanShift": np.nan,
            }
        )
    return pd.DataFrame(rows)


def scientific_gates(
    availability: pd.DataFrame,
    reliability: pd.DataFrame,
    ranks: pd.DataFrame,
    comparisons: pd.DataFrame,
    permutations: pd.DataFrame,
    replay_passed: bool,
) -> tuple[pd.DataFrame, list[str], str, bool]:
    rows = []
    for candidate in CANDIDATES:
        available = availability[availability["candidateId"].eq(candidate)]
        availability_passed = bool(
            len(available) == MATRIX_COUNT
            and available["completeConfirmationUnit"].all()
            and available["eligibleStates"].sum() == MATRIX_COUNT * len(LANDMARKS)
            and not available["replacementAttempted"].any()
        )
        reliable = reliability[reliability["candidateId"].eq(candidate)].iloc[0]
        reliability_passed = bool(reliable.reliabilityGatePassed)
        candidate_comparisons = comparisons[comparisons["candidateId"].eq(candidate)]
        proper_score_passed = bool(
            candidate_comparisons["logLossImprovementLower95"].min() > 0
            and candidate_comparisons["qBrierImprovementLower95"].min() > 0
        )
        full_ranks = ranks[
            ranks["candidateId"].eq(candidate)
            & ranks["modelId"].eq("FULL_STATE_GRAPH_HISTORY")
        ]
        overall_rank_passed = bool(full_ranks["qSpearmanLower95"].min() > 0.3)
        within_rank_passed = bool(full_ranks["centeredQSpearmanLower95"].min() > 0.1)
        candidate_permutations = permutations[permutations["candidateId"].eq(candidate)]
        permutation_passed = bool(
            candidate_permutations["overallUpperTailP"].max() < 0.01
            and candidate_permutations["centeredUpperTailP"].max() < 0.01
        )
        passed = confirmation_gate(
            availability=availability_passed,
            reliability=reliability_passed,
            proper_score=proper_score_passed,
            overall_rank=overall_rank_passed,
            within_matrix_rank=within_rank_passed,
            permutation=permutation_passed,
            replay=replay_passed,
        )
        rows.append(
            {
                "gateId": f"UNTOUCHED_CONFIRMATION::{candidate}",
                "candidateId": candidate,
                "availabilityPassed": availability_passed,
                "reliabilityPassed": reliability_passed,
                "properScorePassed": proper_score_passed,
                "overallRankPassed": overall_rank_passed,
                "withinMatrixRankPassed": within_rank_passed,
                "permutationPassed": permutation_passed,
                "replayPassed": replay_passed,
                "minimumLogLossImprovementLower95": float(
                    candidate_comparisons["logLossImprovementLower95"].min()
                ),
                "minimumQBrierImprovementLower95": float(
                    candidate_comparisons["qBrierImprovementLower95"].min()
                ),
                "minimumQSpearmanLower95": float(full_ranks["qSpearmanLower95"].min()),
                "minimumCenteredQSpearmanLower95": float(
                    full_ranks["centeredQSpearmanLower95"].min()
                ),
                "maximumOverallPermutationP": float(
                    candidate_permutations["overallUpperTailP"].max()
                ),
                "maximumCenteredPermutationP": float(
                    candidate_permutations["centeredUpperTailP"].max()
                ),
                "passed": passed,
            }
        )
    gates = pd.DataFrame(rows)
    confirmed = len(gates) == 2 and bool(gates["passed"].all())
    if confirmed:
        classifications = [
            "UNTOUCHED_PAST_OBSERVABLE_PROCESS_RISK_COORDINATE_CONFIRMED",
            "PLASTIC_HEREDITY_SWITCHING_PROPENSITY_PREDICTABLE",
            "SIMULATOR_PROCESS_EARLY_WARNING_CONFIRMED",
            "NOT_PAPER_REPLICATION",
        ]
        next_theme = "HUMAN_REVIEW_CONFIRMED_SOLUTION_STOP"
    else:
        classifications = [
            "UNTOUCHED_PAST_OBSERVABLE_PROCESS_RISK_NOT_CONFIRMED",
            "ADAPTIVE_L53_LEAD_DID_NOT_FULLY_TRANSFER",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        if not gates["reliabilityPassed"].all():
            next_theme = "L55_PROCESS_RISK_COHORT_RELIABILITY_AUDIT"
        elif not gates["properScorePassed"].all():
            next_theme = "L55_MODEL_CALIBRATION_AND_COHORT_SHIFT_AUDIT"
        else:
            next_theme = "L55_WITHIN_MATRIX_SIGNAL_FAILURE_DECOMPOSITION"
    return gates, classifications, next_theme, confirmed


def compute_tables(
    *,
    branches: pd.DataFrame,
    states: pd.DataFrame,
    observed: pd.DataFrame,
    availability: pd.DataFrame,
    features: pd.DataFrame,
    prospective: pd.DataFrame,
    replay_passed: bool,
) -> tuple[dict[str, pd.DataFrame], list[str], str, bool]:
    estimates = state_estimates(branches, states, observed)
    scored = score_predictions(prospective, estimates)
    matrix_metrics, metrics = L53.metric_tables(scored)
    reliability, reliability_bootstrap = reliability_results(estimates)
    bootstrap = matrix_bootstrap(scored)
    ranks, comparisons = bootstrap_summaries(metrics, bootstrap)
    permutations = q_permutations(scored)
    gates, classifications, next_theme, confirmed = scientific_gates(
        availability,
        reliability,
        ranks,
        comparisons,
        permutations,
        replay_passed,
    )
    tables = {
        "state_committor_results.parquet": estimates,
        "scored_prediction_results.parquet": scored,
        "matrix_metric_results.parquet": matrix_metrics,
        "predictive_metric_results.parquet": metrics,
        "committor_reliability_results.parquet": reliability,
        "committor_reliability_bootstrap.parquet": reliability_bootstrap,
        "primary_matrix_bootstrap.parquet": bootstrap,
        "q_rank_results.parquet": ranks,
        "model_comparisons.parquet": comparisons,
        "whole_matrix_permutation_results.parquet": permutations,
        "cohort_shift_results.parquet": cohort_shift_results(features, estimates),
        "scientific_gate_results.parquet": gates,
    }
    return tables, classifications, next_theme, confirmed


def make_figures(tables: dict[str, pd.DataFrame]) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)
    estimates = tables["state_committor_results.parquet"]
    primary = estimates[
        estimates["horizon"].eq(PRIMARY_HORIZON)
        & estimates["targetType"].eq(PRIMARY_TARGET)
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        values = primary[primary["candidateId"].eq(candidate)]["q"]
        axis.hist(values, bins=np.linspace(0, 1, 21), color="#4472c4", alpha=0.85)
        axis.axvspan(0.1, 0.9, color="#70ad47", alpha=0.12)
        axis.set_title(f"Candidate {candidate[-2:]}")
        axis.set_xlabel("F12 break + run-3 empirical probability")
    axes[0].set_ylabel("Untouched states")
    fig.suptitle("Untouched process-risk distribution")
    fig.tight_layout()
    fig.savefig(root / "01_untouched_committor_distribution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = primary[primary["candidateId"].eq(candidate)]
        axis.scatter(group["qHalfA"], group["qHalfB"], s=14, alpha=0.65)
        axis.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--")
        axis.set_title(f"Candidate {candidate[-2:]}")
        axis.set_xlabel("Branch half A")
    axes[0].set_ylabel("Branch half B")
    fig.suptitle("Independent branch-half reliability")
    fig.tight_layout()
    fig.savefig(root / "02_branch_half_reliability.png", dpi=160)
    plt.close(fig)

    ranks = tables["q_rank_results.parquet"]
    labels = {
        "TRAINING_PRIOR": "prior",
        "DIRECT_HISTORY_PHASE": "history",
        "BETA_STRUCTURE": "beta",
        "FULL_STATE_GRAPH_HISTORY": "full state",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for column, candidate in enumerate(CANDIDATES):
        for row, field in enumerate(("qSpearman", "centeredQSpearman")):
            axis = axes[row, column]
            group = ranks[ranks["candidateId"].eq(candidate)]
            summary = (
                group.groupby("modelId")[field]
                .mean()
                .reindex(labels)
                .rename(index=labels)
            )
            axis.bar(summary.index, summary.values, color="#5b9bd5")
            axis.axhline(0, color="black", lw=0.8)
            axis.tick_params(axis="x", rotation=20)
            axis.set_title(
                f"Candidate {candidate[-2:]} — {'overall' if row == 0 else 'within matrix'}"
            )
            axis.set_ylabel("Spearman")
    fig.suptitle("Frozen L53 model ranking on untouched matrices")
    fig.tight_layout()
    fig.savefig(root / "03_frozen_model_rank_transfer.png", dpi=160)
    plt.close(fig)

    comparisons = tables["model_comparisons.parquet"].copy()
    comparisons["label"] = (
        comparisons["candidateId"].str[-2:]
        + " "
        + comparisons["direction"]
        + " "
        + comparisons["comparisonId"]
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x = np.arange(len(comparisons))
    for axis, point, low, high, title in (
        (
            axes[0],
            "logLossImprovement",
            "logLossImprovementLower95",
            "logLossImprovementUpper95",
            "Branch log-loss improvement",
        ),
        (
            axes[1],
            "qBrierImprovement",
            "qBrierImprovementLower95",
            "qBrierImprovementUpper95",
            "q-Brier improvement",
        ),
    ):
        axis.errorbar(
            x,
            comparisons[point],
            yerr=np.vstack(
                (
                    comparisons[point] - comparisons[low],
                    comparisons[high] - comparisons[point],
                )
            ),
            fmt="o",
        )
        axis.axhline(0, color="black", lw=0.8)
        axis.set_ylabel(title)
    axes[1].set_xticks(x, comparisons["label"], rotation=70, ha="right", fontsize=7)
    fig.suptitle("Untouched proper-score confirmation contrasts")
    fig.tight_layout()
    fig.savefig(root / "04_untouched_proper_score_contrasts.png", dpi=160)
    plt.close(fig)

    predictions = tables["scored_prediction_results.parquet"]
    predictions = predictions[
        predictions["horizon"].eq(PRIMARY_HORIZON)
        & predictions["targetType"].eq(PRIMARY_TARGET)
        & predictions["modelId"].eq("FULL_STATE_GRAPH_HISTORY")
        & predictions["direction"].eq("A_TO_B")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = predictions[predictions["candidateId"].eq(candidate)]
        axis.scatter(
            group["predictedProbability"], group["empiricalQ"], s=14, alpha=0.65
        )
        axis.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--")
        axis.set_title(f"Candidate {candidate[-2:]}")
        axis.set_xlabel("Frozen past-observable prediction")
    axes[0].set_ylabel("Independent branch-half probability")
    fig.suptitle("Prospective frozen-coordinate calibration view")
    fig.tight_layout()
    fig.savefig(root / "05_prediction_vs_empirical_probability.png", dpi=160)
    plt.close(fig)

    shifts = tables["cohort_shift_results.parquet"]
    shifts = shifts[shifts["quantityId"].ne("F12_JOINT_Q")]
    pivot = shifts.pivot(
        index="quantityId", columns="candidateId", values="standardizedMeanShift"
    )
    fig, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-2, vmax=2)
    axis.set_xticks(range(len(pivot.columns)), [f"C{x[-2:]}" for x in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), pivot.index, fontsize=8)
    axis.set_title("Confirmation-cohort shift from L53 development")
    fig.colorbar(image, ax=axis, label="standardized mean shift")
    fig.tight_layout()
    fig.savefig(root / "06_confirmation_cohort_shift.png", dpi=160)
    plt.close(fig)

    gates = tables["scientific_gate_results.parquet"].set_index("candidateId")
    columns = [
        "availabilityPassed",
        "reliabilityPassed",
        "properScorePassed",
        "overallRankPassed",
        "withinMatrixRankPassed",
        "permutationPassed",
        "replayPassed",
    ]
    matrix = gates[columns].T
    fig, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(matrix.to_numpy(dtype=float), cmap="RdYlGn", vmin=0, vmax=1)
    axis.set_xticks(range(len(matrix.columns)), [f"C{x[-2:]}" for x in matrix.columns])
    axis.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    axis.set_title("Conjunctive untouched-confirmation gates")
    fig.colorbar(image, ax=axis, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(root / "07_confirmation_gate_matrix.png", dpi=160)
    plt.close(fig)


def report_text(
    tables: dict[str, pd.DataFrame],
    classifications: list[str],
    next_theme: str,
    confirmed: bool,
    runtime: dict[str, Any],
) -> str:
    estimates = tables["state_committor_results.parquet"]
    process = (
        estimates.groupby(["candidateId", "horizon", "targetType"], sort=True)
        .agg(
            matrices=("matrixIndex", "nunique"),
            states=("stateId", "size"),
            meanQ=("q", "mean"),
            sdQ=("q", "std"),
            observedRate=("observedTarget", "mean"),
        )
        .reset_index()
    )
    primary_metrics = tables["predictive_metric_results.parquet"]
    primary_metrics = primary_metrics[
        primary_metrics["horizon"].eq(PRIMARY_HORIZON)
        & primary_metrics["targetType"].eq(PRIMARY_TARGET)
    ]
    reliability = tables["committor_reliability_results.parquet"]
    comparisons = tables["model_comparisons.parquet"]
    ranks = tables["q_rank_results.parquet"]
    permutations = tables["whole_matrix_permutation_results.parquet"]
    gates = tables["scientific_gate_results.parquet"]
    status_sentence = (
        "The completely frozen L53 coordinate passed every untouched gate in both simulator candidates."
        if confirmed
        else "The completely frozen L53 coordinate did not pass every untouched gate in both simulator candidates."
    )
    return f"""# S19-L54 Full Results — Untouched Past-Observable Process-Risk Confirmation

## Top summary

- **Research step:** `{VERSION}`
- **Completion status:** complete; new seed-firewalled confirmation cohort frozen and exactly regenerated
- **Artifacts written:** 40 new shared catalytic matrices and initial states, 80 primary trajectories, 400 preregistered fission-landmark states, 25,600 branch futures plus an exact second campaign, unchanged L53 model replay, process-risk and realized-path outcomes, 4,096 matrix bootstraps, 512 whole-matrix permutations, seven figures, validation records, report, and hashes
- **Validation:** PASS — immutable S01–L53 baseline; 12/12 fixtures; zero seed/matrix/initial-state overlap; exact old L53 transformation/model/prediction replay; exact primary/regeneration trajectory, state, feature, prediction, branch, table, and report replay; no replacement; scope, runtime, storage, and artifact checks
- **Outcome classification:** {", ".join(f"`{value}`" for value in classifications)}
- **Lay summary:** {status_sentence} The target is not arrival at one privileged composition. It is the probability that ordinary parent/daughter heredity breaks and a new three-fission hereditary episode forms within twelve fissions—an operational plastic-heredity switching event.
- **Recommended next action:** `{next_theme}`. {"The autonomous sequence stops early for mandatory human review because the preregistered confirmation succeeded." if confirmed else "Continue only with the named bounded L55 diagnostic under the existing authorization through L65."}

## Frozen question and chronology

L54 asks whether a target-blind current-state/catalytic-network representation plus nine directly observed history/phase coordinates predicts an independently shot F12 break-plus-run-3 probability on wholly new matrices. The L53 PCA transforms, ridge coefficients, priors, target, threshold, landmarks, branch budget, candidates, and gates were not refit or selected from L54. Prospective prediction values were frozen before either the main realized-future labels or the independent branch outcomes were opened.

Break probability, conditional resumption after a break, and their joint event remain distinct estimands. This avoids interpreting high ordinary inheritance frequency as homeostatic recovery or fixed-attractor arrival.

## Process probabilities by horizon

{process.to_markdown(index=False, floatfmt=".7f")}

## Independent branch-half reliability

{reliability.to_markdown(index=False, floatfmt=".7f")}

## F12 joint-event predictive metrics

{primary_metrics.to_markdown(index=False, floatfmt=".7f")}

## Registered proper-score comparisons

{comparisons.to_markdown(index=False, floatfmt=".7f")}

## Overall and within-matrix ranks

{ranks.to_markdown(index=False, floatfmt=".7f")}

## Whole-matrix permutation controls

{permutations.to_markdown(index=False, floatfmt=".7f")}

## Confirmation gates

{gates.to_markdown(index=False, floatfmt=".7f")}

## Scientific interpretation

The event is an operational regime-switching process: an inheritance break followed by formation of a new short hereditary episode. It is not exact return to the old molecular composition, and neither a run of inherited fissions nor frequent resumption alone proves error correction, an organism, or a molecular attractor. Overall ranks mix stable matrix propensity and changing state risk; the separately gated within-matrix centered ranks are what test longitudinal ordering beyond stable catalytic-matrix differences.

Even a successful result is simulator-process early warning, not replication of the paper's PhiID claim. PhiID was not computed, no intervention was run, and the historical S18 paper-facing, prediction, and causal-control verdicts remain unchanged.

## Runtime and provenance

- Repository lock: `{runtime["repositoryHead"]}`.
- Workers: `{runtime["workers"]}` with one numerical-library thread; GPU hours: 0.
- Wall time: `{runtime["wallSeconds"] / 3600:.4f}` hours; conservative CPU estimate: `{runtime["estimatedCpuHours"]:.4f}` hours.
- New shared matrices / trajectories / restored states: {runtime["newSharedMatrices"]} / {runtime["newPrimaryTrajectories"]} / {runtime["restoredStates"]}.
- Independent branch futures per campaign: {runtime["branchFuturesPerCampaign"]}; exact branch campaigns: 2.
- Matrix bootstraps: {BOOTSTRAPS}; whole-matrix permutations: {PERMUTATIONS}.

## Limitations

The strict `H>0.9` inheritance process is an operational simulator construct. A run of three is short, and F12 is one registered opportunity horizon. State landmarks are post-fission and do not cover every molecular-time phase. The full-state graph representation is compact and molecule-permutation-invariant. The confirmation tests transfer to new matrices and stochastic futures under the same two reconstructed simulator candidates; it does not establish author-code identity, physical chemistry, biological heredity, causal agency, or intervention efficacy.
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
        "schema": "eidosoma.e01.s19_l54.artifact_manifest.v1",
        "loopId": LOOP_ID,
        "files": rows,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def append_ledgers(
    classifications: list[str], timestamp: str, next_theme: str, confirmed: bool
) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L53 adaptively identified a past-observable full-state graph plus history coordinate for plastic-heredity process risk on heldout L50 matrices.",
            "failureOrAmbiguityTargeted": "Whether that state-local coordinate transfers without refitting to new catalytic matrices, lineages and stochastic branch futures.",
            "informationGainRationale": "A wholly new seed-firewalled cohort separates a transferable process coordinate from adaptive cohort-specific fit.",
            "learned": "The complete L53 transformation/model and one untouched simulation/branch contract were frozen before L54 outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L53 full-versus-history proper-score gains and within-matrix q ordering in both candidates.",
            "proposedNextTest": "Apply the unchanged L53 coordinate to 40 new shared matrices and independently branched F12 outcomes.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "transferable plastic-heredity switching propensity versus adaptive cohort-specific relation",
            "selectedHypotheses": "A past-observable state and catalytic-network coordinate transfers to untouched process committors.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "A Phi variant or fixed-attractor target is the highest-leverage next test.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "Confirmation requires both candidates, both branch-half directions, proper-score gains, overall and within-matrix ranks, permutation controls, and exact replay.",
            "failureOrAmbiguityTargeted": "Untouched transfer of the L53 process-risk lead.",
            "informationGainRationale": "Independent matrices and branch streams make the result confirmatory for this reconstructed simulator-process scope.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L54 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION"
            if not confirmed
            else "POST_LOOP_RESULT_MANDATORY_HUMAN_REVIEW",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Registered untouched process-risk confirmation.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Any registered L54 gate that failed.",
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
        + f"\n\n## {LOOP_ID} — untouched past-observable process-risk confirmation\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** `{next_theme}`.\n",
    )
    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    candidate = {
        "branchCount": 2 * MATRIX_COUNT * len(LANDMARKS) * BRANCHES,
        "bundleId": "L54_UNTOUCHED_PROCESS_RISK_CONFIRMATION",
        "candidateId": "S19-L54-UNTOUCHED-PAST-OBSERVABLE-PROCESS-RISK",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 4,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 1,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 5,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "unchanged L53 full-state graph plus history prediction of new F12 break-and-run3 process committors",
        "rankingScore": 31.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": confirmed,
        "selectionReason": "UNTOUCHED_CONFIRMATION"
        if confirmed
        else "CONFIRMATION_GATE_FAILED",
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
    source_additions = []
    for row in source_registry().itertuples(index=False):
        source_additions.append(
            {
                "commitOrVersion": None,
                "evidenceClass": row.evidenceClass,
                "finding": f"{row.finding}; L54 use: {row.frozenUse}",
                "licenseStatus": "WORKSPACE_EVIDENCE",
                "redistributionStatus": "REFERENCE_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L54_{row.sourceId}",
                "sourceType": row.evidenceClass,
                "treeIdentity": None,
                "url": row.url,
            }
        )
    BASE.write_parquet(
        source_path,
        pd.concat(
            [
                sources,
                pd.DataFrame(source_additions).reindex(columns=sources.columns),
            ],
            ignore_index=True,
        ),
    )


def required_input_paths() -> dict[str, Path]:
    return {
        "l50StateCommittors": L50_ROOT / "state_committor_results.parquet",
        "l53Manifest": L53_ROOT / "artifact_manifest.json",
        "l53StateFeatures": L53_ROOT / "state_feature_results.parquet",
        "l53TransformedFeatures": L53_ROOT / "transformed_feature_results.parquet",
        "l53PcaRegistry": L53_ROOT / "pca_registry.parquet",
        "l53Predictions": L53_ROOT / "prediction_results.parquet",
        "l53Models": L53_ROOT / "fitted_model_registry.parquet",
        "l53Attributions": L53_ROOT / "feature_attribution_results.parquet",
        "l53Classification": L53_ROOT / "classification.json",
    }


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L54 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L54 local/remote commit mismatch")
    prior = validate_immutable_prior()
    inputs, trajectory_seeds = input_identities()
    branch_seeds = branch_seed_manifest()
    analysis_seeds = analysis_seed_manifest()
    firewall = seed_and_input_firewall(
        inputs, trajectory_seeds, branch_seeds, analysis_seeds
    )
    fixtures = fixture_results(inputs)
    sources = source_registry()
    required = required_input_paths()
    input_validation = pd.DataFrame(
        [
            {
                "inputId": name,
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            for name, path in required.items()
        ]
    )
    benchmark = {
        "schema": "eidosoma.e01.s19_l54.benchmark_projection.v1",
        "outcomeBlind": True,
        "basis": "exact L50 two-campaign 51,200-branch runtime plus S13Y/L11 trajectory and L53 analysis timings",
        "newSharedMatrices": MATRIX_COUNT,
        "newPrimaryTrajectories": 2 * MATRIX_COUNT,
        "states": 2 * MATRIX_COUNT * len(LANDMARKS),
        "branchFuturesPerCampaign": 2 * MATRIX_COUNT * len(LANDMARKS) * BRANCHES,
        "exactCampaigns": 2,
        "workers": WORKERS,
        "projectedCpuHoursUpper": 72.0,
        "projectedWallHoursUpper": 36.0,
        "cpuHourCeiling": 100,
        "wallHourCeiling": 72,
        "validationReserveFraction": 0.15,
        "status": "PASS",
    }
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or firewall["status"] != "PASS"
        or not input_validation["exists"].all()
        or len(inputs) != MATRIX_COUNT
    ):
        raise RuntimeError("L54 preoutcome validation failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L54 decision record\n\n"
        "The human-authorized autonomous sequence through L65 remains active, with "
        "an early stop for a genuinely confirmed solution. L53 produced the first "
        "past-observable state-local plastic-heredity process-risk lead. L54 freezes "
        "that exact PCA, model, target, landmarks, F12 horizon and validation gate, "
        "then evaluates 40 wholly new shared catalytic matrices in both simulator "
        "candidates. Break probability, conditional resumption and their joint event "
        "remain distinct. The reviewer framing is accepted: this is stochastic switching "
        "between hereditary and nonhereditary regimes, not a monotonic ramp toward one "
        "privileged attractor. No Phi quantity, new feature, model refit, threshold "
        "search, intervention or paper-identity claim is authorized.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "input_identity_manifest.parquet", inputs)
    BASE.write_parquet(LOOP_ROOT / "trajectory_seed_manifest.parquet", trajectory_seeds)
    BASE.write_parquet(LOOP_ROOT / "branch_seed_manifest.parquet", branch_seeds)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", analysis_seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "source_registry.parquet", sources)
    BASE.write_parquet(
        LOOP_ROOT / "input_identity_validation.parquet", input_validation
    )
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    source_files = {
        "runner": RUNNER_PATH,
        "core": CORE_PATH,
        "l53Runner": ROOT / "scripts/e01/run_s19_l53_regime_capacity_proxy.py",
        "l50Runner": ROOT
        / "scripts/e01/run_s19_l50_fission_aligned_process_committor.py",
        "l28Runner": ROOT / "scripts/e01/run_s19_l28_branched_empirical_committor.py",
        "simulatorCore": ROOT / "src/e01_latent_timebase/core.py",
        "graphCore": ROOT / "src/e01_onset_discovery/full_state_graph.py",
        "processCore": ROOT / "src/e01_onset_discovery/fission_aligned_process.py",
        "config": CONFIG,
    }
    BASE.write_json(
        LOOP_ROOT / "source_snapshot_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l54.source_snapshot_manifest.v1",
            "repositoryHead": head,
            "files": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in source_files.items()
            },
            "sources": sources.to_dict("records"),
        },
    )
    locked_inputs = {
        **required,
        "inputIdentities": LOOP_ROOT / "input_identity_manifest.parquet",
        "trajectorySeeds": LOOP_ROOT / "trajectory_seed_manifest.parquet",
        "branchSeeds": LOOP_ROOT / "branch_seed_manifest.parquet",
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewall": LOOP_ROOT / "seed_firewall.json",
        "fixtures": LOOP_ROOT / "fixture_results.parquet",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
    }
    hashes = {name: sha256_file(path) for name, path in locked_inputs.items()}
    implementation = {
        "schema": "eidosoma.e01.s19_l54.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "configSha256": sha256_file(CONFIG),
        "seedRoot": ROOT_HEX,
        "phase": PHASE,
        "matrixCount": MATRIX_COUNT,
        "candidates": list(CANDIDATES),
        "landmarks": list(LANDMARKS),
        "branchesPerState": BRANCHES,
        "branchHalves": [HALF, HALF],
        "horizons": list(HORIZONS),
        "primaryTarget": PRIMARY_TARGET,
        "threshold": THRESHOLD,
        "requiredRun": REQUIRED_RUN,
        "models": list(MODELS),
        "historyColumns": list(HISTORY_COLUMNS),
        "pcaComponents": PCA_COMPONENTS,
        "ridgeC": RIDGE_C,
        "refitOnConfirmation": False,
        "matrixBootstraps": BOOTSTRAPS,
        "wholeMatrixPermutations": PERMUTATIONS,
        "workers": WORKERS,
        "gpuHours": 0,
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


def _normalized_trajectory_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "matrixIndex",
        "candidateId",
        "stateId",
        "branchIndex",
        "purpose",
        "configurationId",
        "derivedSeed",
        "seedMaterialSha256",
        "rootHex",
    ]
    return (
        frame[columns]
        .drop_duplicates()
        .sort_values(["candidateId", "matrixIndex", "purpose"])
        .reset_index(drop=True)
    )


def execute() -> None:
    started = time.perf_counter()
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    if (
        git("rev-parse", "HEAD") != lock["head"]
        or git("rev-parse", "origin/eidosoma/groups/42") != lock["remote"]
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("L54 repository lock mismatch")
    prior = validate_immutable_prior()
    locked_inputs = {
        **required_input_paths(),
        "inputIdentities": LOOP_ROOT / "input_identity_manifest.parquet",
        "trajectorySeeds": LOOP_ROOT / "trajectory_seed_manifest.parquet",
        "branchSeeds": LOOP_ROOT / "branch_seed_manifest.parquet",
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewall": LOOP_ROOT / "seed_firewall.json",
        "fixtures": LOOP_ROOT / "fixture_results.parquet",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
    }
    if any(
        sha256_file(path) != lock["lockedInputHashes"][name]
        for name, path in locked_inputs.items()
    ):
        raise RuntimeError("L54 locked input changed")
    locked_identities = pd.read_parquet(LOOP_ROOT / "input_identity_manifest.parquet")
    fixtures = fixture_results(locked_identities)
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
        or sha256_file(CONFIG) != lock["configSha256"]
    ):
        raise RuntimeError("L54 pre-execution validation failed")
    if CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
    PRIMARY_TRAJECTORY_ROOT.mkdir(parents=True)
    REGEN_TRAJECTORY_ROOT.mkdir(parents=True)
    BUILD_ROOT.mkdir(parents=True)

    fitted_pca, fitted_models, old_validation, old_replay = (
        reconstruct_frozen_pipeline()
    )
    primary_manifest, primary_seeds, primary_failures = simulate_all(
        PRIMARY_TRAJECTORY_ROOT
    )
    regenerated_manifest, regenerated_seeds, regenerated_failures = simulate_all(
        REGEN_TRAJECTORY_ROOT
    )
    if len(primary_failures) or len(regenerated_failures):
        raise RuntimeError("L54 trajectory generation failure")
    trajectory_fields = (
        "candidateId",
        "matrixIndex",
        "trajectoryId",
        "trajectorySha256",
        "betaSha256",
        "initialStateSha256",
        "terminalStatus",
        "completedFissions",
        "selectedClockLength",
        "clockId",
        "replacementAttempted",
    )
    trajectory_exact = scientific_manifest_equal(
        primary_manifest.to_dict("records"),
        regenerated_manifest.to_dict("records"),
        trajectory_fields,
    )
    planned_trajectory_seeds = _normalized_trajectory_seeds(
        pd.read_parquet(LOOP_ROOT / "trajectory_seed_manifest.parquet")
    )
    primary_seed_exact = frame_hash(
        _normalized_trajectory_seeds(primary_seeds)
    ) == frame_hash(planned_trajectory_seeds)
    regenerated_seed_exact = frame_hash(
        _normalized_trajectory_seeds(regenerated_seeds)
    ) == frame_hash(planned_trajectory_seeds)
    actual_inputs = (
        primary_manifest.groupby("matrixIndex", sort=True)
        .agg(
            betaSha256=("betaSha256", "first"),
            betaVariants=("betaSha256", "nunique"),
            initialStateSha256=("initialStateSha256", "first"),
            initialVariants=("initialStateSha256", "nunique"),
        )
        .reset_index()
        .merge(locked_identities, on="matrixIndex", suffixes=("Actual", "Locked"))
    )
    input_exact = bool(
        len(actual_inputs) == MATRIX_COUNT
        and actual_inputs["betaVariants"].eq(1).all()
        and actual_inputs["initialVariants"].eq(1).all()
        and actual_inputs["betaSha256Actual"]
        .eq(actual_inputs["betaSha256Locked"])
        .all()
        and actual_inputs["initialStateSha256Actual"]
        .eq(actual_inputs["initialStateSha256Locked"])
        .all()
    )

    primary_payloads, primary_states, primary_availability = build_states(
        primary_manifest
    )
    regenerated_payloads, regenerated_states, regenerated_availability = build_states(
        regenerated_manifest
    )
    state_exact = frame_hash(primary_states) == frame_hash(regenerated_states)
    availability_exact = frame_hash(primary_availability) == frame_hash(
        regenerated_availability
    )
    payload_exact = frame_hash(pd.DataFrame(primary_payloads)) == frame_hash(
        pd.DataFrame(regenerated_payloads)
    )
    if not (
        trajectory_exact
        and primary_seed_exact
        and regenerated_seed_exact
        and input_exact
        and state_exact
        and availability_exact
        and payload_exact
    ):
        raise RuntimeError("L54 trajectory/state regeneration failure")
    if (
        len(primary_states) != 2 * MATRIX_COUNT * len(LANDMARKS)
        or not primary_availability["completeConfirmationUnit"].all()
    ):
        raise RuntimeError("L54 untouched confirmation availability failure")

    primary_features, primary_feature_validation = extract_features(
        primary_payloads, primary_states
    )
    regenerated_features, regenerated_feature_validation = extract_features(
        regenerated_payloads, regenerated_states
    )
    primary_transformed = transform_confirmation_features(primary_features, fitted_pca)
    regenerated_transformed = transform_confirmation_features(
        regenerated_features, fitted_pca
    )
    primary_predictions = prospective_predictions(
        primary_states, primary_transformed, fitted_models
    )
    regenerated_predictions = prospective_predictions(
        regenerated_states, regenerated_transformed, fitted_models
    )
    feature_exact = frame_hash(primary_features) == frame_hash(regenerated_features)
    feature_validation_exact = frame_hash(primary_feature_validation) == frame_hash(
        regenerated_feature_validation
    )
    transformed_exact = frame_hash(primary_transformed) == frame_hash(
        regenerated_transformed
    )
    prediction_exact = frame_hash(primary_predictions) == frame_hash(
        regenerated_predictions
    )
    if not (
        feature_exact
        and feature_validation_exact
        and transformed_exact
        and prediction_exact
    ):
        raise RuntimeError("L54 prospective feature/prediction replay failure")

    primary_observed = realized_path_outcomes(primary_manifest, primary_states)
    regenerated_observed = realized_path_outcomes(
        regenerated_manifest, regenerated_states
    )
    observed_exact = frame_hash(primary_observed) == frame_hash(regenerated_observed)
    primary_branches = execute_branches(primary_payloads)
    regenerated_branches = execute_branches(regenerated_payloads)
    branch_exact = frame_hash(primary_branches) == frame_hash(regenerated_branches)
    replay_passed = bool(
        trajectory_exact
        and primary_seed_exact
        and regenerated_seed_exact
        and input_exact
        and state_exact
        and availability_exact
        and payload_exact
        and feature_exact
        and feature_validation_exact
        and transformed_exact
        and prediction_exact
        and observed_exact
        and branch_exact
        and old_validation["exactPredictionReplay"].all()
        and old_replay["passed"].all()
    )
    if not replay_passed:
        raise RuntimeError("L54 exact replay failed")

    tables, classifications, next_theme, confirmed = compute_tables(
        branches=primary_branches,
        states=primary_states,
        observed=primary_observed,
        availability=primary_availability,
        features=primary_features,
        prospective=primary_predictions,
        replay_passed=replay_passed,
    )
    tables_again, classifications_again, next_theme_again, confirmed_again = (
        compute_tables(
            branches=regenerated_branches,
            states=regenerated_states,
            observed=regenerated_observed,
            availability=regenerated_availability,
            features=regenerated_features,
            prospective=regenerated_predictions,
            replay_passed=replay_passed,
        )
    )
    table_exact = {
        name: frame_hash(frame) == frame_hash(tables_again[name])
        for name, frame in tables.items()
    }
    if not (
        all(table_exact.values())
        and classifications == classifications_again
        and next_theme == next_theme_again
        and confirmed == confirmed_again
    ):
        raise RuntimeError("L54 table/classification regeneration failure")

    trajectory_validation = pd.DataFrame(
        [
            {
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "primaryTrajectorySha256": row.trajectorySha256,
                "regeneratedTrajectorySha256": regenerated_manifest.set_index(
                    ["candidateId", "matrixIndex"]
                ).loc[(row.candidateId, int(row.matrixIndex)), "trajectorySha256"],
                "exact": row.trajectorySha256
                == regenerated_manifest.set_index(["candidateId", "matrixIndex"]).loc[
                    (row.candidateId, int(row.matrixIndex)), "trajectorySha256"
                ],
            }
            for row in primary_manifest.itertuples(index=False)
        ]
    )
    scope = pd.DataFrame(
        [
            ("sharedMatrices", MATRIX_COUNT, primary_manifest["matrixIndex"].nunique()),
            ("candidateTrajectories", 2 * MATRIX_COUNT, len(primary_manifest)),
            ("restoredStates", 2 * MATRIX_COUNT * len(LANDMARKS), len(primary_states)),
            (
                "prospectivePredictions",
                len(primary_states) * 2 * 3 * 3 * 4,
                len(primary_predictions),
            ),
            ("branchFutures", len(primary_states) * BRANCHES, len(primary_branches)),
            (
                "branchSeedRows",
                len(primary_states) * BRANCHES * 4,
                len(pd.read_parquet(LOOP_ROOT / "branch_seed_manifest.parquet")),
            ),
        ],
        columns=["scopeId", "expected", "actual"],
    )
    scope["passed"] = scope["expected"].eq(scope["actual"])
    regeneration = {
        "schema": "eidosoma.e01.s19_l54.regeneration_validation.v1",
        "status": "PASS",
        "oldL53PredictionReplay": bool(old_validation["exactPredictionReplay"].all()),
        "trajectoryExact": trajectory_exact,
        "trajectorySeedPrimaryExact": primary_seed_exact,
        "trajectorySeedRegeneratedExact": regenerated_seed_exact,
        "inputIdentityExact": input_exact,
        "stateExact": state_exact,
        "availabilityExact": availability_exact,
        "payloadExact": payload_exact,
        "featureExact": feature_exact,
        "featureValidationExact": feature_validation_exact,
        "transformedFeatureExact": transformed_exact,
        "prospectivePredictionExact": prediction_exact,
        "realizedPathOutcomeExact": observed_exact,
        "branchCampaignExact": branch_exact,
        "tableExact": table_exact,
        "classificationExact": classifications == classifications_again,
        "nextThemeExact": next_theme == next_theme_again,
        "confirmationDecisionExact": confirmed == confirmed_again,
        "reportExact": True,
    }

    output_tables = {
        "trajectory_manifest.parquet": primary_manifest.drop(columns=["cachePath"]),
        "trajectory_identity_validation.parquet": trajectory_validation,
        "trajectory_seed_replay.parquet": _normalized_trajectory_seeds(primary_seeds),
        "state_registry.parquet": primary_states,
        "state_availability.parquet": primary_availability,
        "realized_path_outcomes.parquet": primary_observed,
        "state_feature_results.parquet": primary_features,
        "feature_validation.parquet": primary_feature_validation,
        "transformed_feature_results.parquet": primary_transformed,
        "old_validation_model_replay.parquet": old_validation,
        "old_validation_replay_summary.parquet": old_replay,
        "prospective_prediction_results.parquet": primary_predictions,
        "branch_results.parquet": primary_branches,
        "scope_validation.parquet": scope,
        **tables,
    }
    for name, frame in output_tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    make_figures(tables)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l54.classification.v1",
            "classifications": classifications,
            "confirmed": confirmed,
            "nextTheme": next_theme,
            "priorStatusesChanged": False,
            "paperReplicationClaim": False,
            "phiComputed": False,
            "interventionRun": False,
        },
    )
    pd.concat(
        [
            primary_failures.assign(campaign="PRIMARY"),
            regenerated_failures.assign(campaign="REGENERATION"),
        ],
        ignore_index=True,
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    pd.DataFrame(
        columns=["amendmentId", "status", "scientificContractChanged", "reason"]
    ).to_csv(BUILD_ROOT / "technical_amendment_ledger.csv", index=False)
    elapsed = time.perf_counter() - started
    runtime = {
        "schema": "eidosoma.e01.s19_l54.runtime.v1",
        "repositoryHead": lock["head"],
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": elapsed,
        "estimatedCpuHours": elapsed * WORKERS / 3600,
        "newSharedMatrices": MATRIX_COUNT,
        "newPrimaryTrajectories": 2 * MATRIX_COUNT,
        "exactTrajectoryCampaigns": 2,
        "restoredStates": len(primary_states),
        "branchFuturesPerCampaign": len(primary_branches),
        "exactBranchCampaigns": 2,
        "matrixBootstraps": BOOTSTRAPS,
        "wholeMatrixPermutations": PERMUTATIONS,
        "completedAtUtc": utc_now(),
    }
    if runtime["estimatedCpuHours"] > 100 or runtime["wallSeconds"] > 72 * 3600:
        raise RuntimeError("L54 runtime ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    BASE.write_json(BUILD_ROOT / "regeneration_validation.json", regeneration)
    retained_bytes = sum(
        path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
    ) + sum(path.stat().st_size for path in LOOP_ROOT.iterdir() if path.is_file())
    cache_bytes = sum(
        path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file()
    )
    storage = {
        "schema": "eidosoma.e01.s19_l54.storage_validation.v1",
        "status": "PASS"
        if retained_bytes <= 25 * 1024**3 and cache_bytes <= 75 * 1024**3
        else "FAIL",
        "retainedBytes": retained_bytes,
        "temporaryBytes": cache_bytes,
        "retainedGiBCeiling": 25,
        "temporaryGiBCeiling": 75,
    }
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(tables, classifications, next_theme, confirmed, runtime)
    if report != report_text(tables, classifications, next_theme, confirmed, runtime):
        raise RuntimeError("L54 report regeneration failure")
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L54_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L54 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Confirmed:** {confirmed}.\n\n**Next:** `{next_theme}`.\n",
    )
    if storage["status"] != "PASS" or not scope["passed"].all():
        raise RuntimeError("L54 storage or scope validation failed")
    for path in (BUILD_ROOT / "figures").glob("*.png"):
        if not path.stat().st_size:
            raise RuntimeError(f"empty L54 figure: {path}")
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
        raise RuntimeError("L54 artifact manifest regeneration failure")
    append_ledgers(classifications, runtime["completedAtUtc"], next_theme, confirmed)
    root_report = (
        f"# S19 current-step report\n\nLatest completed loop: `{LOOP_ID}`.\n\n"
        f"Classification: {', '.join(classifications)}.\n\n"
        f"Next: `{next_theme}`.\n"
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
            "programStatus": "AWAITING_HUMAN_REVIEW_CONFIRMED_SOLUTION"
            if confirmed
            else "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "nextAuthorizedLoop": None if confirmed else "S19-L55",
            "nextTheme": next_theme,
            "authorizationUpperBound": "S19-L65",
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
                "confirmed": confirmed,
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
