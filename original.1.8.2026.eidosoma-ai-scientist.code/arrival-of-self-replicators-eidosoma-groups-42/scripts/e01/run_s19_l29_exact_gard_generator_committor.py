"""Execute S19-L29 exact-GARD-generator committor-coordinate discovery."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    os.environ.setdefault(variable, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_latent_timebase.core import (
    array_sha256 as simulator_array_sha256,
)
from e01_latent_timebase.core import (
    derive_seed,
    exposure_for_rates,
    generate_beta,
    generator,
    rates,
)
from e01_onset_discovery.generator_coordinate import (
    KERNEL_HALF_SAMPLES,
    KERNEL_SAMPLES,
    TARGET_THRESHOLD,
    analytic_count_moments,
    brownian_hitting_probability,
    composition_linearized_moments,
    cosine_gradient,
    relative_composition,
    sample_complete_kernel,
    summarize_moments,
    truncated_poisson_moments,
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L28 = _load_module(
    "e01_s19_l29_l28",
    REPO_ROOT / "scripts/e01/run_s19_l28_branched_empirical_committor.py",
)
BASE = L28.BASE
LOOP_ID = "S19-L29"
VERSION = "E01-S19-L29-EXACT-GARD-GENERATOR-COMMITTOR-COORDINATE-v1.0.0"
CANDIDATES = L28.CANDIDATES
LANDMARKS = L28.LANDMARKS
BOOTSTRAPS = 4096
PERMUTATIONS = 512
WORKERS = 8
KERNEL_ROOT_HEX = "8db9a1c44a1cc014576f001b75232ed21ed19bf355261d13cb64700917b13aea"
KERNEL_PHASE = "s19_l29_exact_gard_generator_committor"
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L29"
L28_ROOT = ARTIFACT_ROOT / "loops/L28"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l29")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l29_exact_gard_generator_committor.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/generator_coordinate.py"

MODEL_COLUMNS = {
    "TARGET_GEOMETRY_CONTROL": [
        "landmarkScaled",
        "currentMassScaled",
        "generationLocalStepScaled",
        "nextIsFission",
        "currentDiversity",
        "currentEntropy",
        "currentConcentration",
        "targetScore",
        "targetGap",
        "targetComponentFraction",
        "targetEntropy",
        "targetSupportOverlap",
    ],
    "BASIN_BLIND_GENERATOR": [
        "landmarkScaled",
        "currentMassScaled",
        "generationLocalStepScaled",
        "nextIsFission",
        "currentDiversity",
        "currentEntropy",
        "currentConcentration",
        "boostMean",
        "boostStd",
        "boostMax",
        "reactionActivityPerMass",
        "joinActivityPerMass",
        "lossActivityPerMass",
        "analyticCountMuL1PerMass",
        "analyticCountMuNormPerMass",
        "analyticCountDiffusionTracePerMass2",
        "analyticMassDriftPerMass",
        "analyticMassDiffusionPerMass2",
        "analyticMuNorm",
        "analyticDiffusionTrace",
        "analyticDiffusionEigmax",
        "analyticCurrentMuAlignment",
        "kernelMuNorm",
        "kernelDiffusionTrace",
        "kernelDiffusionEigmax",
        "kernelCurrentMuAlignment",
        "kernelEmptyNextFraction",
    ],
}
MODEL_COLUMNS["ANALYTIC_RADIAL_GENERATOR"] = list(
    dict.fromkeys(
        MODEL_COLUMNS["TARGET_GEOMETRY_CONTROL"]
        + MODEL_COLUMNS["BASIN_BLIND_GENERATOR"][:21]
        + [
            "analyticTargetDirectionDrift",
            "analyticTargetDirectionDiffusion",
            "analyticScoreDrift",
            "analyticScoreVariance",
            "analyticScoreSignalNoise",
            "analyticBrownianHit32",
            "targetNetRateProjection",
            "targetJoinShare",
            "targetLossShare",
        ]
    )
)
MODEL_COLUMNS["COMPLETE_KERNEL_RADIAL_GENERATOR"] = list(
    dict.fromkeys(
        MODEL_COLUMNS["ANALYTIC_RADIAL_GENERATOR"]
        + MODEL_COLUMNS["BASIN_BLIND_GENERATOR"]
        + [
            "kernelTargetDirectionDrift",
            "kernelTargetDirectionDiffusion",
            "kernelScoreDrift",
            "kernelScoreVariance",
            "kernelScoreSignalNoise",
            "kernelBrownianHit32",
            "kernelOneStepHitProbability",
        ]
    )
)
PRIMARY_MODEL = "COMPLETE_KERNEL_RADIAL_GENERATOR"
CONTROL_MODELS = (
    "TARGET_GEOMETRY_CONTROL",
    "BASIN_BLIND_GENERATOR",
    "EXACT_H_TRACE_ANALOG",
    "ORDINARY_PATH_ANALOG",
)


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


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.reset_index(drop=True)
        .to_json(orient="table", index=False, double_precision=15)
        .encode()
    ).hexdigest()


def derived_seed(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256(
            "\x1f".join([VERSION, KERNEL_ROOT_HEX, *map(str, parts)]).encode()
        ).digest()[:16],
        "big",
    )


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L28_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L28_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L28_ROOT / item["path"]),
            "root": str(L28_ROOT),
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
        "schema": "eidosoma.e01.s19_l29.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
        ).hexdigest(),
        "l28ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def kernel_seed_identities(
    candidate: str, matrix: int, landmark: int
) -> dict[str, Any]:
    return {
        purpose: derive_seed(
            KERNEL_ROOT_HEX,
            KERNEL_PHASE,
            purpose,
            matrix,
            candidate,
            landmark,
        )
        for purpose in (
            "generator_event",
            "generator_trim",
            "generator_fission",
            "generator_daughter",
        )
    }


def seed_manifest(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in states.itertuples(index=False):
        identities = kernel_seed_identities(
            state.candidateId, int(state.matrixIndex), int(state.landmark)
        )
        row = {
            "stateId": state.stateId,
            "candidateId": state.candidateId,
            "matrixRole": state.matrixRole,
            "matrixIndex": int(state.matrixIndex),
            "landmark": int(state.landmark),
            "rootHex": KERNEL_ROOT_HEX,
        }
        materials = []
        for purpose, identity in identities.items():
            token = purpose.replace("generator_", "")
            row[f"{token}DerivedSeed"] = str(identity.derived_seed)
            row[f"{token}SeedMaterialSha256"] = identity.seed_material_sha256
            materials.append(identity.seed_material_sha256)
        row["kernelIdentitySha256"] = hashlib.sha256(
            "|".join([state.stateId, *materials]).encode()
        ).hexdigest()
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["candidateId", "matrixRole", "landmark", "matrixIndex"])
        .reset_index(drop=True)
    )


def prior_seed_materials() -> set[str]:
    values: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L29/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            if "seedmaterialsha256" in column.lower():
                values.update(str(value) for value in frame[column].dropna())
    return values


def seed_firewall(manifest: pd.DataFrame, prior: dict[str, Any]) -> dict[str, Any]:
    columns = [column for column in manifest.columns if "SeedMaterialSha256" in column]
    current = set()
    for column in columns:
        current.update(manifest[column].astype(str))
    prior_material = prior_seed_materials()
    overlaps = sorted(current & prior_material)
    root_paths = []
    needle = KERNEL_ROOT_HEX.encode()
    for row in prior["files"]:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            continue
        try:
            if needle in path.read_bytes():
                root_paths.append(str(path))
        except OSError:
            continue
    return {
        "schema": "eidosoma.e01.s19_l29.seed_firewall.v1",
        "status": "PASS" if not overlaps and not root_paths else "FAIL",
        "rootHex": KERNEL_ROOT_HEX,
        "stateCount": len(manifest),
        "seedMaterialCount": len(current),
        "seedMaterialUnique": len(current) == len(manifest) * 4,
        "priorSeedMaterialCount": len(prior_material),
        "overlapCount": len(overlaps),
        "overlaps": overlaps,
        "rootCollisionPaths": root_paths,
    }


def fixture_results() -> pd.DataFrame:
    state = np.zeros(100, dtype=np.int64)
    state[:40] = 1
    target = relative_composition(state)
    beta = np.exp(np.full((100, 100), -4.0, dtype=np.float64))
    definition = L28.definition(CANDIDATES[0])
    mean, variance = truncated_poisson_moments(2.3, 4)
    analytic = analytic_count_moments(state, beta, definition, generation_local_step=0)
    mu_x, d_x = composition_linearized_moments(state, analytic)
    identities = kernel_seed_identities(CANDIDATES[0], 999_901, 64)

    def kernel() -> Any:
        return sample_complete_kernel(
            state,
            beta,
            definition,
            target,
            generation_local_step=0,
            event_rng=generator(identities["generator_event"]),
            trim_rng=generator(identities["generator_trim"]),
            fission_rng=generator(identities["generator_fission"]),
            daughter_rng=generator(identities["generator_daughter"]),
            samples=128,
        )

    first = kernel()
    replay = kernel()
    permutation = np.arange(100)[::-1]
    permuted = analytic_count_moments(
        state[permutation],
        beta[np.ix_(permutation, permutation)],
        definition,
        generation_local_step=0,
    )
    rows = [
        {
            "fixtureId": "TRUNCATED_POISSON_FINITE",
            "passed": np.isfinite([mean, variance]).all() and mean > 0 and variance > 0,
            "details": f"{mean},{variance}",
        },
        {
            "fixtureId": "ANALYTIC_MOMENT_FINITE",
            "passed": np.isfinite(analytic.mean_delta).all()
            and np.isfinite(analytic.covariance_delta).all()
            and np.isfinite(mu_x).all()
            and np.isfinite(d_x).all(),
            "details": analytic.semantics,
        },
        {
            "fixtureId": "KERNEL_EXACT_REPLAY",
            "passed": np.array_equal(first.delta_composition, replay.delta_composition)
            and np.array_equal(first.next_scores, replay.next_scores),
            "details": simulator_array_sha256(first.delta_composition),
        },
        {
            "fixtureId": "MOLECULAR_RELABEL_EQUIVARIANCE",
            "passed": np.allclose(analytic.mean_delta[permutation], permuted.mean_delta)
            and np.allclose(
                analytic.covariance_delta[np.ix_(permutation, permutation)],
                permuted.covariance_delta,
            ),
            "details": "simultaneous state/beta permutation",
        },
        {
            "fixtureId": "COSINE_GRADIENT_ZERO_AT_TARGET",
            "passed": np.allclose(cosine_gradient(target, target), 0, atol=1e-12),
            "details": str(np.linalg.norm(cosine_gradient(target, target))),
        },
        {
            "fixtureId": "BROWNIAN_HIT_BOUNDED",
            "passed": 0 <= brownian_hitting_probability(0.1, 0.01, 0.002, 32) <= 1,
            "details": str(brownian_hitting_probability(0.1, 0.01, 0.002, 32)),
        },
        {
            "fixtureId": "MODEL_COLUMNS_UNIQUE",
            "passed": all(
                len(columns) == len(set(columns)) for columns in MODEL_COLUMNS.values()
            ),
            "details": json.dumps(
                {key: len(value) for key, value in MODEL_COLUMNS.items()}
            ),
        },
    ]
    return pd.DataFrame(rows)


def load_l23_trajectory(row: Any) -> Any:
    path = Path(row.cachePath)
    if not path.is_file() or sha256_file(path) != row.cacheSha256:
        raise RuntimeError("L23 cache identity failure")
    with path.open("rb") as handle:
        trajectory = pickle.load(handle)
    if trajectory.trajectory_sha256 != row.trajectorySha256:
        raise RuntimeError("L23 trajectory identity failure")
    return trajectory


def state_payloads(
    states: pd.DataFrame,
    coordinates: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    reference_variant: str,
) -> list[dict[str, Any]]:
    manifest_index = manifest.set_index(["candidateId", "matrixIndex"])
    centroid_map: dict[str, tuple[list[float], int, str]] = {}
    for state in states.itertuples(index=False):
        centroid = (
            coordinates[
                coordinates["candidateId"].eq(state.candidateId)
                & coordinates["matrixIndex"].eq(int(state.matrixIndex))
                & coordinates["landmark"].eq(int(state.landmark))
            ]
            .sort_values("coordinate")["centroidValue"]
            .to_numpy(dtype=np.float64)
        )
        centroid_map[state.stateId] = (
            centroid.tolist(),
            int(state.targetComponentSize),
            state.stateId,
        )
    if reference_variant == "TARGET_REFERENCE_PERMUTATION":
        for _, group in states.groupby(["candidateId", "landmark"], sort=True):
            ordered = group.sort_values("stateId")["stateId"].tolist()
            donors = ordered[1:] + ordered[:1]
            original = {key: centroid_map[key] for key in ordered}
            for receiver, donor in zip(ordered, donors, strict=True):
                values, size, donor_id = original[donor]
                centroid_map[receiver] = (values, size, donor_id)
    payloads = []
    for state in states.itertuples(index=False):
        manifest_row = manifest_index.loc[(state.candidateId, int(state.matrixIndex))]
        trajectory = load_l23_trajectory(manifest_row)
        selected = L28.selected_clock_observations(trajectory, L28.CLOCK_ID)
        observation = selected[int(state.currentSelectedIndex)]
        centroid, component_size, donor_id = centroid_map[state.stateId]
        payloads.append(
            {
                **state._asdict(),
                "state": list(map(int, observation.state)),
                "centroid": centroid,
                "targetComponentSizeUsed": component_size,
                "targetReferenceDonorStateId": donor_id,
                "referenceVariant": reference_variant,
            }
        )
    return payloads


def _entropy(value: np.ndarray) -> float:
    positive = value[value > 0]
    return float(-np.sum(positive * np.log(positive))) if len(positive) else 0.0


def _moment_summary(
    delta: np.ndarray,
    scores: np.ndarray,
    state: np.ndarray,
    target: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    mu = delta.mean(axis=0)
    covariance = np.cov(delta, rowvar=False, ddof=1)
    current_score = float(L28.cosine_to_reference(state[None, :], target)[0])
    score_delta = scores - current_score
    return summarize_moments(
        state,
        target,
        mu,
        covariance,
        score_drift=float(score_delta.mean()),
        score_variance=float(score_delta.var(ddof=1)),
        one_step_hit_probability=float(np.mean(scores >= TARGET_THRESHOLD)),
        prefix=prefix,
    )


def _feature_worker(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = payload["candidateId"]
    matrix_index = int(payload["matrixIndex"])
    landmark = int(payload["landmark"])
    beta_seed = L28.derive_seed(
        L28.L23_ROOT_HEX, L28.L23_PHASE, "catalytic_matrix", matrix_index
    )
    beta = generate_beta(beta_seed)
    if simulator_array_sha256(beta) != payload["betaSha256"]:
        raise RuntimeError("beta replay failed")
    state = np.asarray(payload["state"], dtype=np.int64)
    target = np.asarray(payload["centroid"], dtype=np.float64)
    definition = L28.definition(candidate)
    local_step = (
        0
        if payload["currentObservationKind"]
        in {"initial_selected_state", "post_fission"}
        else int(payload["currentGenerationLocalStep"])
    )
    analytic = analytic_count_moments(
        state, beta, definition, generation_local_step=local_step
    )
    analytic_mu_x, analytic_d_x = composition_linearized_moments(state, analytic)
    identities = kernel_seed_identities(candidate, matrix_index, landmark)
    kernel = sample_complete_kernel(
        state,
        beta,
        definition,
        target,
        generation_local_step=local_step,
        event_rng=generator(identities["generator_event"]),
        trim_rng=generator(identities["generator_trim"]),
        fission_rng=generator(identities["generator_fission"]),
        daughter_rng=generator(identities["generator_daughter"]),
    )
    if len(kernel.delta_composition) != KERNEL_SAMPLES:
        raise RuntimeError("kernel sample cardinality failure")
    analytic_summary = summarize_moments(
        state,
        target,
        analytic_mu_x,
        analytic_d_x,
        prefix="analytic",
    )
    kernel_summary = _moment_summary(
        kernel.delta_composition, kernel.next_scores, state, target, "kernel"
    )
    half_a = _moment_summary(
        kernel.delta_composition[:KERNEL_HALF_SAMPLES],
        kernel.next_scores[:KERNEL_HALF_SAMPLES],
        state,
        target,
        "halfA",
    )
    half_b = _moment_summary(
        kernel.delta_composition[KERNEL_HALF_SAMPLES:],
        kernel.next_scores[KERNEL_HALF_SAMPLES:],
        state,
        target,
        "halfB",
    )
    x = relative_composition(state)
    score = float(L28.cosine_to_reference(state[None, :], target)[0])
    joins, losses = rates(state, beta)
    exposure = exposure_for_rates(definition.exposure, joins, losses)
    boost = 1.0 + (beta @ state.astype(np.float64)) / float(state.sum())
    net_rate = exposure * (joins - losses)
    target_norm = np.linalg.norm(target)
    target_projection = float(np.dot(net_rate, target) / max(target_norm, 1e-18))
    join_total = float(joins.sum())
    loss_total = float(losses.sum())
    mass = float(state.sum())
    row = {
        "stateId": payload["stateId"],
        "candidateId": candidate,
        "matrixRole": payload["matrixRole"],
        "matrixIndex": matrix_index,
        "landmark": landmark,
        "referenceVariant": payload["referenceVariant"],
        "targetReferenceDonorStateId": payload["targetReferenceDonorStateId"],
        "transitionKind": analytic.transition_kind,
        "analyticSemantics": analytic.semantics,
        "landmarkScaled": landmark / 192.0,
        "currentMassScaled": mass / 80.0,
        "generationLocalStepScaled": local_step / 1000.0,
        "nextIsFission": float(analytic.transition_kind == "FISSION"),
        "currentDiversity": float(np.count_nonzero(state) / 100.0),
        "currentEntropy": _entropy(x) / math.log(100.0),
        "currentConcentration": float(np.sum(x * x)),
        "targetScore": score,
        "targetGap": TARGET_THRESHOLD - score,
        "targetComponentFraction": payload["targetComponentSizeUsed"] / 100.0,
        "targetEntropy": _entropy(target) / math.log(100.0),
        "targetSupportOverlap": float(np.mean((state > 0) & (target > 0))),
        "boostMean": float(boost.mean()),
        "boostStd": float(boost.std(ddof=0)),
        "boostMax": float(boost.max()),
        "reactionActivityPerMass": exposure * (join_total + loss_total) / mass,
        "joinActivityPerMass": exposure * join_total / mass,
        "lossActivityPerMass": exposure * loss_total / mass,
        "analyticCountMuL1PerMass": float(np.abs(analytic.mean_delta).sum() / mass),
        "analyticCountMuNormPerMass": float(np.linalg.norm(analytic.mean_delta) / mass),
        "analyticCountDiffusionTracePerMass2": float(
            np.trace(analytic.covariance_delta) / (mass * mass)
        ),
        "analyticMassDriftPerMass": float(analytic.mean_delta.sum() / mass),
        "analyticMassDiffusionPerMass2": float(
            analytic.covariance_delta.sum() / (mass * mass)
        ),
        "targetNetRateProjection": target_projection / mass,
        "targetJoinShare": float(np.dot(joins, target) / max(join_total, 1e-18)),
        "targetLossShare": float(np.dot(losses, target) / max(loss_total, 1e-18)),
        "kernelEmptyNextFraction": float(kernel.empty_next.mean()),
        **analytic_summary,
        **kernel_summary,
    }
    stability = {
        "stateId": payload["stateId"],
        "candidateId": candidate,
        "matrixRole": payload["matrixRole"],
        "matrixIndex": matrix_index,
        "landmark": landmark,
        "referenceVariant": payload["referenceVariant"],
        "kernelSampleCount": KERNEL_SAMPLES,
        "kernelHalfSampleCount": KERNEL_HALF_SAMPLES,
        "deltaCompositionSha256": simulator_array_sha256(kernel.delta_composition),
        "nextScoreSha256": simulator_array_sha256(kernel.next_scores),
        **half_a,
        **half_b,
    }
    return row, stability


def execute_features(
    payloads: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    stability = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_feature_worker, payload): payload["stateId"]
            for payload in payloads
        }
        for future in as_completed(futures):
            row, stable = future.result()
            rows.append(row)
            stability.append(stable)
    order = ["referenceVariant", "candidateId", "matrixRole", "landmark", "matrixIndex"]
    return (
        pd.DataFrame(rows).sort_values(order).reset_index(drop=True),
        pd.DataFrame(stability).sort_values(order).reset_index(drop=True),
    )


def safe_spearman(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else float("nan")


def fit_model(
    development: pd.DataFrame, columns: list[str]
) -> tuple[StandardScaler, LogisticRegression]:
    x = development[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise RuntimeError("nonfinite model feature")
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    expanded = np.vstack([z, z])
    labels = np.concatenate([np.ones(len(z)), np.zeros(len(z))])
    weights = np.concatenate(
        [development["successes"].to_numpy(), 128 - development["successes"].to_numpy()]
    )
    model = LogisticRegression(
        C=0.1,
        solver="lbfgs",
        max_iter=2000,
        fit_intercept=True,
        random_state=0,
    ).fit(expanded, labels, sample_weight=weights)
    return scaler, model


def model_predictions(
    features: pd.DataFrame, q: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = (
        features.copy()
        if {"successes", "qHat"}.issubset(features.columns)
        else features.merge(
            q[["stateId", "successes", "qHat"]],
            on="stateId",
            validate="many_to_one",
        )
    )
    rows = []
    registry = []
    for candidate in CANDIDATES:
        candidate_rows = merged[merged["candidateId"].eq(candidate)]
        development = candidate_rows[candidate_rows["matrixRole"].eq("DEVELOPMENT")]
        prior = float(development["qHat"].mean())
        for source in candidate_rows.itertuples(index=False):
            rows.append(
                {
                    "stateId": source.stateId,
                    "candidateId": candidate,
                    "matrixRole": source.matrixRole,
                    "matrixIndex": source.matrixIndex,
                    "landmark": source.landmark,
                    "modelId": "DEVELOPMENT_PRIOR",
                    "predictedQ": prior,
                    "qHat": source.qHat,
                    "successes": source.successes,
                    "referenceVariant": source.referenceVariant,
                }
            )
        for model_id, columns in MODEL_COLUMNS.items():
            scaler, model = fit_model(development, columns)
            probabilities = model.predict_proba(
                scaler.transform(candidate_rows[columns].to_numpy(dtype=np.float64))
            )[:, 1]
            replay_scaler, replay_model = fit_model(development, columns)
            replay = replay_model.predict_proba(
                replay_scaler.transform(
                    candidate_rows[columns].to_numpy(dtype=np.float64)
                )
            )[:, 1]
            if not np.array_equal(probabilities, replay):
                raise RuntimeError("model exact replay failed")
            registry.append(
                {
                    "candidateId": candidate,
                    "modelId": model_id,
                    "featureCount": len(columns),
                    "featureNames": json.dumps(columns),
                    "scalerMean": json.dumps(scaler.mean_.tolist()),
                    "scalerScale": json.dumps(scaler.scale_.tolist()),
                    "intercept": float(model.intercept_[0]),
                    "coefficients": json.dumps(model.coef_[0].tolist()),
                    "iterations": int(model.n_iter_[0]),
                    "exactReplay": True,
                }
            )
            for source, probability in zip(
                candidate_rows.itertuples(index=False), probabilities, strict=True
            ):
                rows.append(
                    {
                        "stateId": source.stateId,
                        "candidateId": candidate,
                        "matrixRole": source.matrixRole,
                        "matrixIndex": source.matrixIndex,
                        "landmark": source.landmark,
                        "modelId": model_id,
                        "predictedQ": float(probability),
                        "qHat": source.qHat,
                        "successes": source.successes,
                        "referenceVariant": source.referenceVariant,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(registry)


def append_frozen_predictors(predictions: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_parquet(L28_ROOT / "frozen_predictor_scores.parquet").copy()
    q = pd.read_parquet(L28_ROOT / "committor_state_results.parquet")[
        ["candidateId", "matrixIndex", "landmark", "stateId", "qHat", "successes"]
    ]
    frozen = frozen.merge(
        q,
        on=["candidateId", "matrixIndex", "landmark", "qHat", "successes"],
        validate="many_to_one",
    )
    frozen_rows = frozen.rename(
        columns={"predictorId": "modelId", "score": "predictedQ"}
    )
    frozen_rows["matrixRole"] = "VALIDATION"
    frozen_rows["referenceVariant"] = "ORIGINAL"
    columns = predictions.columns
    return (
        pd.concat(
            [predictions, frozen_rows.reindex(columns=columns)], ignore_index=True
        )
        .sort_values(
            [
                "referenceVariant",
                "candidateId",
                "matrixRole",
                "modelId",
                "landmark",
                "matrixIndex",
            ]
        )
        .reset_index(drop=True)
    )


def calibration(scores: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    return L28.calibration_parameters(scores, q)


def metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    validation = predictions[predictions["matrixRole"].eq("VALIDATION")]
    for (variant, candidate, model), group in validation.groupby(
        ["referenceVariant", "candidateId", "modelId"], sort=True
    ):
        q = group["qHat"].to_numpy(dtype=np.float64)
        p = np.clip(group["predictedQ"].to_numpy(dtype=np.float64), 1e-9, 1 - 1e-9)
        brier = float(np.mean(q * (1 - p) ** 2 + (1 - q) * p**2))
        log_loss = float(-np.mean(q * np.log(p) + (1 - q) * np.log(1 - p)))
        intercept, slope = calibration(p, q)
        rows.append(
            {
                "referenceVariant": variant,
                "candidateId": candidate,
                "modelId": model,
                "states": len(group),
                "spearmanQHat": safe_spearman(p, q),
                "brierScorePerBranch": brier,
                "binomialLogLossPerBranch": log_loss,
                "calibrationIntercept": intercept,
                "calibrationSlope": slope,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    validation = predictions[
        predictions["matrixRole"].eq("VALIDATION")
        & predictions["referenceVariant"].eq("ORIGINAL")
    ]
    for candidate in CANDIDATES:
        source = validation[validation["candidateId"].eq(candidate)]
        pivot = source.pivot(
            index=["stateId", "qHat"], columns="modelId", values="predictedQ"
        ).reset_index()
        rng = np.random.default_rng(derived_seed("matrix_bootstrap", candidate))
        for replicate in range(BOOTSTRAPS):
            sample = pivot.iloc[rng.integers(0, len(pivot), size=len(pivot))]
            q = sample["qHat"].to_numpy(dtype=np.float64)
            briers = {}
            for model in [
                column for column in pivot.columns if column not in {"stateId", "qHat"}
            ]:
                p = np.clip(sample[model].to_numpy(dtype=np.float64), 1e-9, 1 - 1e-9)
                brier = float(np.mean(q * (1 - p) ** 2 + (1 - q) * p**2))
                briers[model] = brier
                rows.append(
                    {
                        "candidateId": candidate,
                        "bootstrapIndex": replicate,
                        "modelId": model,
                        "spearmanQHat": safe_spearman(p, q),
                        "brierScorePerBranch": brier,
                        "primaryBrierImprovement": float("nan"),
                    }
                )
            for control in CONTROL_MODELS:
                rows.append(
                    {
                        "candidateId": candidate,
                        "bootstrapIndex": replicate,
                        "modelId": f"DELTA_PRIMARY_VS_{control}",
                        "spearmanQHat": float("nan"),
                        "brierScorePerBranch": float("nan"),
                        "primaryBrierImprovement": briers[control]
                        - briers[PRIMARY_MODEL],
                    }
                )
    return pd.DataFrame(rows)


def development_label_permutations(
    original_features: pd.DataFrame, q: pd.DataFrame, observed_metrics: pd.DataFrame
) -> pd.DataFrame:
    merged = (
        original_features.copy()
        if {"successes", "qHat"}.issubset(original_features.columns)
        else original_features.merge(
            q[["stateId", "successes", "qHat"]],
            on="stateId",
            validate="one_to_one",
        )
    )
    rows = []
    for candidate in CANDIDATES:
        group = merged[merged["candidateId"].eq(candidate)]
        development = group[group["matrixRole"].eq("DEVELOPMENT")].copy()
        validation = group[group["matrixRole"].eq("VALIDATION")]
        observed = float(
            observed_metrics[
                observed_metrics["candidateId"].eq(candidate)
                & observed_metrics["modelId"].eq(PRIMARY_MODEL)
                & observed_metrics["referenceVariant"].eq("ORIGINAL")
            ]["spearmanQHat"].iloc[0]
        )
        rng = np.random.default_rng(
            derived_seed("development_label_permutation", candidate)
        )
        null_values = []
        for replicate in range(PERMUTATIONS):
            permutation = rng.permutation(len(development))
            permuted = development.copy()
            permuted["successes"] = development["successes"].to_numpy()[permutation]
            permuted["qHat"] = development["qHat"].to_numpy()[permutation]
            scaler, model = fit_model(permuted, MODEL_COLUMNS[PRIMARY_MODEL])
            prediction = model.predict_proba(
                scaler.transform(
                    validation[MODEL_COLUMNS[PRIMARY_MODEL]].to_numpy(dtype=np.float64)
                )
            )[:, 1]
            rho = safe_spearman(prediction, validation["qHat"])
            null_values.append(rho)
            rows.append(
                {
                    "candidateId": candidate,
                    "permutationIndex": replicate,
                    "observedSpearman": observed,
                    "nullSpearman": rho,
                    "familywiseP": float("nan"),
                }
            )
        finite = np.asarray([value for value in null_values if np.isfinite(value)])
        p_value = float((1 + np.sum(finite >= observed)) / (1 + len(finite)))
        for row in rows[-PERMUTATIONS:]:
            row["familywiseP"] = p_value
    return pd.DataFrame(rows)


def kernel_stability_summary(stability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [
        ("MuNorm", "halfAMuNorm", "halfBMuNorm"),
        ("DiffusionTrace", "halfADiffusionTrace", "halfBDiffusionTrace"),
        ("ScoreDrift", "halfAScoreDrift", "halfBScoreDrift"),
        ("ScoreVariance", "halfAScoreVariance", "halfBScoreVariance"),
        ("BrownianHit32", "halfABrownianHit32", "halfBBrownianHit32"),
    ]
    for (variant, candidate), group in stability.groupby(
        ["referenceVariant", "candidateId"], sort=True
    ):
        for feature, left, right in pairs:
            rows.append(
                {
                    "referenceVariant": variant,
                    "candidateId": candidate,
                    "feature": feature,
                    "splitHalfSpearman": safe_spearman(group[left], group[right]),
                    "meanAbsoluteDifference": float(
                        np.mean(np.abs(group[left] - group[right]))
                    ),
                }
            )
    return pd.DataFrame(rows)


def gate_table(
    metrics: pd.DataFrame,
    bootstraps: pd.DataFrame,
    permutations: pd.DataFrame,
    replay_passed: bool,
) -> pd.DataFrame:
    rows = []
    original = metrics[metrics["referenceVariant"].eq("ORIGINAL")]
    for candidate in CANDIDATES:
        primary = original[
            original["candidateId"].eq(candidate)
            & original["modelId"].eq(PRIMARY_MODEL)
        ].iloc[0]
        primary_boot = bootstraps[
            bootstraps["candidateId"].eq(candidate)
            & bootstraps["modelId"].eq(PRIMARY_MODEL)
        ]
        finite = primary_boot["spearmanQHat"].dropna().to_numpy(dtype=np.float64)
        rho_lower = float(np.quantile(finite, 0.025)) if len(finite) else float("nan")
        improvement_lowers = {}
        for control in CONTROL_MODELS:
            values = bootstraps[
                bootstraps["candidateId"].eq(candidate)
                & bootstraps["modelId"].eq(f"DELTA_PRIMARY_VS_{control}")
            ]["primaryBrierImprovement"].to_numpy(dtype=np.float64)
            improvement_lowers[control] = float(np.quantile(values, 0.025))
        permutation_p = float(
            permutations[permutations["candidateId"].eq(candidate)]["familywiseP"].iloc[
                0
            ]
        )
        rank_pass = bool(primary.spearmanQHat > 0.5 and rho_lower > 0.3)
        increment_pass = all(value > 0 for value in improvement_lowers.values())
        rows.append(
            {
                "candidateId": candidate,
                "primarySpearman": primary.spearmanQHat,
                "primarySpearmanBootstrapLower95": rho_lower,
                **{
                    f"brierImprovementLowerVs{control}": value
                    for control, value in improvement_lowers.items()
                },
                "developmentPermutationP": permutation_p,
                "rankGatePassed": rank_pass,
                "allIncrementalBrierGatesPassed": increment_pass,
                "developmentPermutationPassed": permutation_p <= 0.05,
                "exactReplayPassed": replay_passed,
                "candidateCoordinateGatePassed": bool(
                    rank_pass
                    and increment_pass
                    and permutation_p <= 0.05
                    and replay_passed
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    stability: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(root / name, dpi=180)
        plt.close()

    original = features[features["referenceVariant"].eq("ORIGINAL")]
    plt.figure(figsize=(8, 5))
    for candidate in CANDIDATES:
        group = original[original["candidateId"].eq(candidate)]
        plt.scatter(
            group["kernelScoreDrift"], group["qHat"], s=18, alpha=0.6, label=candidate
        )
    plt.xlabel("One-step kernel mean ΔH-to-basin")
    plt.ylabel("Empirical H32 q-hat")
    plt.legend(fontsize=7)
    save("01_kernel_drift_vs_committor.png")

    plt.figure(figsize=(8, 5))
    for candidate in CANDIDATES:
        group = original[original["candidateId"].eq(candidate)]
        plt.scatter(
            group["kernelBrownianHit32"],
            group["qHat"],
            s=18,
            alpha=0.6,
            label=candidate,
        )
    plt.xlabel("Frozen local Brownian H32 approximation")
    plt.ylabel("Empirical H32 q-hat")
    plt.legend(fontsize=7)
    save("02_local_hitting_approximation.png")

    validation = predictions[
        predictions["matrixRole"].eq("VALIDATION")
        & predictions["referenceVariant"].eq("ORIGINAL")
        & predictions["modelId"].eq(PRIMARY_MODEL)
    ]
    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = validation[validation["candidateId"].eq(candidate)]
        axis.scatter(group["predictedQ"], group["qHat"], s=24)
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set_title(candidate)
        axis.set_xlabel("Predicted q")
        axis.set_ylabel("128-branch q-hat")
    save("03_heldout_committor_predictions.png")

    pivot = metrics[metrics["referenceVariant"].eq("ORIGINAL")].pivot(
        index="modelId", columns="candidateId", values="spearmanQHat"
    )
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Held-out Spearman with q-hat")
    save("04_model_rank_comparison.png")

    stability[stability["referenceVariant"].eq("ORIGINAL")].pivot(
        index="feature", columns="candidateId", values="splitHalfSpearman"
    ).plot(kind="bar", figsize=(9, 5))
    plt.axhline(0.9, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Kernel-moment split-half Spearman")
    save("05_kernel_moment_reliability.png")

    gate_columns = [
        "rankGatePassed",
        "allIncrementalBrierGatesPassed",
        "developmentPermutationPassed",
        "exactReplayPassed",
        "candidateCoordinateGatePassed",
    ]
    matrix = gates.set_index("candidateId")[gate_columns].astype(float)
    plt.figure(figsize=(8, 3.5))
    plt.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(
        range(len(gate_columns)), gate_columns, rotation=35, ha="right", fontsize=8
    )
    plt.yticks(range(len(matrix)), matrix.index)
    plt.colorbar(ticks=[0, 1])
    save("06_coordinate_gate_matrix.png")


def manifest_for(root: Path) -> dict[str, Any]:
    rows = [
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
        "schema": "eidosoma.e01.s19_l29.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(rows),
        "totalBytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def append_ledgers(classifications: list[str], timestamp: str, next_theme: str) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L28 established reliable committor variation but every frozen representation missed it.",
            "failureOrAmbiguityTargeted": "Whether exact local GARD drift/diffusion encodes the missing state coordinate.",
            "informationGainRationale": "The simulator generator is the nearest mechanistic state description before any transition-current analysis.",
            "learned": "L29 generator/model/gate contract frozen.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "STATE_DEPENDENT_COMMITTOR_ESTABLISHED;EXISTING_REPRESENTATIONS_MISS_STATE_SIGNAL",
            "proposedNextTest": "Execute held-out exact-generator coordinate audit.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Local generator signal, target geometry alone, or nonlocal/multistep state dependence.",
            "selectedHypotheses": "Basin-blind, analytic radial and complete-kernel radial generator coordinates.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "More generic trajectory representations should precede mechanistic generator testing.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A held-out committor coordinate must add calibrated value beyond target geometry and ordinary controls.",
            "failureOrAmbiguityTargeted": "Local generator sufficiency and target-conditioning dependence.",
            "informationGainRationale": "Development/validation separation and controls distinguish generator information from basin geometry.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L29 results.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "One fixed exact-generator coordinate family.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Exact one-step local generator moments are sufficient"
            if "MISS" in ";".join(classifications)
            else "Existing representations span the generator signal.",
        },
    ]
    BASE.write_parquet(
        ledger_path,
        pd.concat(
            [ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)],
            ignore_index=True,
        ),
    )
    markdown_path = ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    BASE.atomic_text(
        markdown_path,
        markdown_path.read_text()
        + f"\n\n## {LOOP_ID} — exact GARD generator committor coordinate\n\n"
        + "- **Before:** L28 established state-dependent q but frozen representations missed it.\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** {next_theme}.\n",
    )
    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    row = {
        "branchCount": 3,
        "bundleId": "L29_EXACT_GARD_GENERATOR",
        "candidateId": "S19-L29-GENERATOR-COMMITTOR",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 1,
        "computeEfficiency": 5,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 3,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "exact analytic and complete-kernel local drift/diffusion coordinates",
        "rankingScore": 26.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L28_ESTABLISHED_COMMITTOR_FROZEN_REPRESENTATIONS_MISS",
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
    sources_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(sources_path)
    source_row = {
        "commitOrVersion": "10.1063/1.481811",
        "evidenceClass": "PRIMARY_METHOD_PAPER",
        "finding": "Chemical reaction propensities define local drift/diffusion; L29 applies exact frozen GARD rates and complete one-step kernel moments.",
        "licenseStatus": "PUBLIC_ARTICLE",
        "redistributionStatus": "CITATION_ONLY",
        "repositoryIdentity": None,
        "retainedPath": None,
        "retrievalDate": timestamp[:10],
        "sha256": None,
        "sourceId": "L29_GILLESPIE_CHEMICAL_LANGEVIN_2000",
        "sourceType": "PRIMARY_METHOD_PAPER",
        "treeIdentity": None,
        "url": "https://doi.org/10.1063/1.481811",
    }
    BASE.write_parquet(
        sources_path,
        pd.concat(
            [sources, pd.DataFrame([source_row]).reindex(columns=sources.columns)],
            ignore_index=True,
        ),
    )
    loop_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(loop_path.read_text())
    registry["loops"].append(
        {
            "loopId": LOOP_ID,
            "versionedLoopId": VERSION,
            "status": "COMPLETE_AUTONOMOUS_CONTINUATION_AUTHORIZED",
            "authorized": True,
            "completed": True,
            "outcomeAccessed": True,
            "humanReviewRequiredAfter": False,
            "classification": classifications,
            "selectedDiscoveryLead": None,
            "newMatrices": 0,
            "newTrajectories": 0,
            "newOneStepKernelSamples": 200 * KERNEL_SAMPLES,
            "nextStepActive": True,
        }
    )
    registry["laterLoopsAuthorized"] = True
    registry["authorizationUpperBound"] = "S19-L42"
    registry["proposedNextLoopTheme"] = next_theme
    registry["proposedNextLoopActive"] = True
    BASE.atomic_text(loop_path, yaml.safe_dump(registry, sort_keys=False))
    review_path = ARTIFACT_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text())
    review["history"].append(
        {
            "decision": "S19_L29_COMPLETE_AUTONOMOUS_CONTINUATION",
            "loopId": LOOP_ID,
            "scope": VERSION,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "selectedDiscoveryLead": None,
            "source": "locked_execution_result",
            "nextLoopAuthorized": True,
            "s20Activated": False,
        }
    )
    review["pendingDecision"] = "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L42"
    BASE.write_json(review_path, review)


def report_text(
    metrics: pd.DataFrame,
    stability: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    return f"""# S19-L29 — Exact GARD Generator Drift/Diffusion Committor Coordinate

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the authorized L19–L42 sequence.
- **Outcome classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Validation:** exact L28 state/beta/target/q identities; source-defined analytic moments; 2048 complete one-step kernel samples per state; independent moment halves; full feature and model replay; development-only fit; 4,096 matrix bootstraps; 512 development-label permutations; reference, seed, immutable-prior, runtime/storage and artifact gates passed.
- **Next bounded theme:** {next_theme}

## Frozen question

Do exact local GARD birth/death/fission drift and diffusion features recover the reliable L28 H32 committor on held-out matrices beyond target geometry and prior exact-H/ordinary representations?

## Method boundary

The analytical branch uses source-defined clipped-Poisson pre-trim growth moments and exact candidate-specific fission moments. The complete-kernel branch estimates the implemented one-selected-clock transition moments with 2,048 independent samples, including overshoot trim and daughter selection. It generates no new H32 future and does not reuse any L28 branch stream. Target-radial features are explicitly conditioned on the completed-run matrix-specific basin; they are retrospective-basin-conditioned and cannot establish online early warning.

## Held-out validation metrics

{metrics.to_markdown(index=False)}

## One-step moment reliability

{stability.to_markdown(index=False)}

## Gate adjudication

{gates.to_markdown(index=False)}

## Interpretation

A local generator coordinate must rank held-out q with Spearman above 0.5 and bootstrap lower bound above 0.3, improve Brier score beyond the development prior, target geometry, basin-blind generator, frozen exact-H trace and ordinary path with lower bounds above zero, and pass the development-label permutation in both candidates. A target-conditioned pass remains retrospective; a basin-blind pass would be the stronger online-state clue. No result is confirmatory without a later untouched seed-firewalled cohort.

## Runtime and provenance

- Repository lock: `{runtime["repositoryHead"]}`.
- CPU float64, `{runtime["workers"]}` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `{runtime["wallSeconds"]:.3f}`; aggregate worker CPU hours: `{runtime["workerCpuHours"]:.6f}`.
- Method grounding: Gillespie, *The Chemical Langevin Equation*, DOI `10.1063/1.481811`; L28 finite-horizon shooting audit.

## Autonomous boundary

L29 is frozen. Transition-tube/reactive-current work remains prohibited unless this loop establishes a held-out committor-predictive coordinate in both candidates. S20, E02, author contact, interventions and report-bundle work remain inactive.
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local and remote heads differ")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    if not prior["unchanged"] or not fixtures["passed"].all():
        raise RuntimeError("prior or fixture gate failed")
    states = pd.read_parquet(L28_ROOT / "restored_state_registry.parquet")
    coordinates = pd.read_parquet(L28_ROOT / "target_basin_coordinates.parquet")
    q = pd.read_parquet(L28_ROOT / "committor_state_results.parquet")
    if (
        len(states) != 200
        or len(coordinates) != 20_000
        or len(q) != 200
        or set(states.stateId) != set(q.stateId)
    ):
        raise RuntimeError("L28 state/q identity failure")
    seeds = seed_manifest(states)
    firewall = seed_firewall(seeds, prior)
    if firewall["status"] != "PASS" or not firewall["seedMaterialUnique"]:
        raise RuntimeError("seed firewall failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L29 decision record\n\nL28 established reliable state-dependent H32 committor variation in both candidates and found that all frozen representations missed it. Under the human's directed pass branch, L29 freezes exact source-defined local birth/death/fission moments, a complete one-selected-clock kernel moment audit, one basin-blind coordinate, one target-geometry control and target-radial analytic/kernel coordinates before feature or model outcomes. The completed-run target basin is outcome-definition conditioning, not a prospective input claim. No H32 branch, target, landmark, threshold or simulator setting changes.\n",
    )
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "kernel_seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.atomic_text(
        LOOP_ROOT / "source_grounding_report.md",
        "# L29 source grounding\n\n- Gillespie (2000), DOI `10.1063/1.481811`: reaction propensities determine local drift and diffusion moments.\n- L28: independent shooting established the empirical H32 state-conditioned target probability before generator-feature construction.\n- The complete-kernel moments include E01's frozen Poisson clipping, overshoot trim, fission and daughter semantics; the analytic growth generator is explicitly pre-trim.\n",
    )
    hashes = {
        "statesSha256": sha256_file(L28_ROOT / "restored_state_registry.parquet"),
        "coordinatesSha256": sha256_file(L28_ROOT / "target_basin_coordinates.parquet"),
        "qSha256": sha256_file(L28_ROOT / "committor_state_results.parquet"),
        "seedManifestSha256": sha256_file(LOOP_ROOT / "kernel_seed_manifest.parquet"),
    }
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l29.implementation_lock.v1",
            "researchStepId": LOOP_ID,
            "versionedId": VERSION,
            "repositoryHead": head,
            "remoteHead": remote,
            "configSha256": sha256_file(CONFIG),
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "kernelSamples": KERNEL_SAMPLES,
            "kernelHalves": [KERNEL_HALF_SAMPLES, KERNEL_HALF_SAMPLES],
            "modelColumns": MODEL_COLUMNS,
            "primaryModel": PRIMARY_MODEL,
            "bootstrapReplicates": BOOTSTRAPS,
            "labelPermutations": PERMUTATIONS,
            "outcomeAccessed": False,
            "lockedHashes": hashes,
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
    BASE.atomic_text(
        LOOP_ROOT / "model_registry.yaml",
        yaml.safe_dump(
            {
                "models": [
                    {"modelId": key, "columns": value, "primary": key == PRIMARY_MODEL}
                    for key, value in MODEL_COLUMNS.items()
                ],
                "prior": "DEVELOPMENT_PRIOR",
                "family": "L2_AGGREGATED_BINOMIAL_LOGISTIC",
                "C": 0.1,
            },
            sort_keys=False,
        ),
    )


def register_technical_amendment() -> None:
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    existing_path = LOOP_ROOT / "technical_amendment_lock.json"
    existing = (
        json.loads(existing_path.read_text()) if existing_path.is_file() else None
    )
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    previous_head = existing["amendedRepositoryHead"] if existing else lock["head"]
    if git("status", "--porcelain=v1") or head != remote or head == previous_head:
        raise RuntimeError("technical amendment requires a new clean pushed commit")
    for key, path in {
        "statesSha256": L28_ROOT / "restored_state_registry.parquet",
        "coordinatesSha256": L28_ROOT / "target_basin_coordinates.parquet",
        "qSha256": L28_ROOT / "committor_state_results.parquet",
        "seedManifestSha256": LOOP_ROOT / "kernel_seed_manifest.parquet",
    }.items():
        if sha256_file(path) != lock[key]:
            raise RuntimeError(f"scientific lock changed during amendment: {path}")
    amendment_number = (
        int(existing["amendmentId"].rsplit("_", 1)[1]) + 1 if existing else 1
    )
    amendment_id = f"TECHNICAL_{amendment_number:03d}"
    amendment_details = {
        1: (
            "FROZEN_COMPARATOR_TABLE_ASSEMBLY",
            "PANDAS_MERGE_CARDINALITY_ASSERTION",
            "ONE_TO_ONE_TO_MANY_TO_ONE_VALIDATION_FOR_FOUR_REGISTERED_PREDICTORS_PER_STATE",
        ),
        2: (
            "KERNEL_RELIABILITY_FIGURE_ASSEMBLY",
            "PANDAS_PIVOT_DUPLICATE_REFERENCE_VARIANTS",
            "FILTER_REGISTERED_RELIABILITY_FIGURE_TO_ORIGINAL_REFERENCE_VARIANT",
        ),
        3: (
            "FINAL_REGENERATION_VALIDATION_ASSEMBLY",
            "IMMUTABLE_PRIOR_LOCAL_NAME_SHADOWING",
            "RENAME_LOCAL_DEVELOPMENT_PRIOR_METRIC_ROW",
        ),
    }
    if amendment_number not in amendment_details:
        raise RuntimeError("unregistered technical amendment number")
    failure_stage, failure_class, repair = amendment_details[amendment_number]
    record = {
        "schema": "eidosoma.e01.s19_l29.technical_amendment_lock.v1",
        "amendmentId": amendment_id,
        "originalRepositoryHead": lock["head"],
        "previousRepositoryHead": previous_head,
        "amendedRepositoryHead": head,
        "amendedRemoteHead": remote,
        "originalRunnerSha256": lock["runnerSha256"],
        "amendedRunnerSha256": sha256_file(RUNNER_PATH),
        "previousAmendmentSha256": sha256_file(existing_path) if existing else None,
        "failureStage": failure_stage,
        "failureClass": failure_class,
        "repair": repair,
        "scientificContractChanged": False,
        "featureChanged": False,
        "modelChanged": False,
        "predictionChanged": False,
        "metricOrGateChanged": False,
        "freshCacheRequired": True,
        "registeredAtUtc": utc_now(),
    }
    BASE.write_json(existing_path, record)
    ledger_path = LOOP_ROOT / "technical_amendment_ledger.csv"
    ledger = pd.read_csv(ledger_path) if ledger_path.is_file() else pd.DataFrame()
    ledger_row = {
        "amendmentId": amendment_id,
        "stage": failure_stage,
        "outcomesOpened": True,
        "failure": failure_class,
        "repair": repair,
        "scientificValuesChanged": False,
        "freshCacheRerun": True,
    }
    pd.concat([ledger, pd.DataFrame([ledger_row])], ignore_index=True).to_csv(
        ledger_path, index=False
    )
    BASE.atomic_text(
        LOOP_ROOT / f"failed_attempt_{amendment_number:03d}.md",
        {
            1: "# L29 failed attempt 001\n\nThe first locked execution computed the registered features and model predictions in memory, then stopped before scientific aggregation when the four frozen L28 predictors were joined to one state-q table under an incorrect one-to-one assertion. The correct registered cardinality is many predictor rows to one state. No scientific method or value was changed; partial caches are invalidated and the amended execution must recompute from fresh caches.\n",
            2: "# L29 failed attempt 002\n\nThe technically amended execution completed every scientific calculation in memory, then stopped during plotting because the reliability table retained both original and target-reference-permutation variants while a figure pivot expected one row per candidate and feature. The registered figure is now restricted to the original-reference diagnostic. No scientific method or value changed; partial caches are invalidated and the analysis must recompute from fresh caches.\n",
            3: "# L29 failed attempt 003\n\nThe second amended execution completed scientific calculations and figures, then stopped while assembling the regeneration record because a local development-prior metric row shadowed the immutable-prior validation object. Renaming that local variable changes no scientific array, model, prediction, metric, gate or classification. Partial caches are invalidated and the analysis must recompute from fresh caches.\n",
        }[amendment_number],
    )


def execute() -> None:
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    amendment_path = LOOP_ROOT / "technical_amendment_lock.json"
    amendment = (
        json.loads(amendment_path.read_text()) if amendment_path.is_file() else None
    )
    expected_head = amendment["amendedRepositoryHead"] if amendment else lock["head"]
    expected_remote = amendment["amendedRemoteHead"] if amendment else lock["remote"]
    if (
        git("rev-parse", "HEAD") != expected_head
        or git("rev-parse", "origin/eidosoma/groups/42") != expected_remote
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    for key, path in {
        "statesSha256": L28_ROOT / "restored_state_registry.parquet",
        "coordinatesSha256": L28_ROOT / "target_basin_coordinates.parquet",
        "qSha256": L28_ROOT / "committor_state_results.parquet",
        "seedManifestSha256": LOOP_ROOT / "kernel_seed_manifest.parquet",
    }.items():
        if sha256_file(path) != lock[key]:
            raise RuntimeError(f"locked input changed: {path}")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
    ):
        raise RuntimeError("pre-execution validation failed")
    states = pd.read_parquet(L28_ROOT / "restored_state_registry.parquet")
    coordinates = pd.read_parquet(L28_ROOT / "target_basin_coordinates.parquet")
    q = pd.read_parquet(L28_ROOT / "committor_state_results.parquet")
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    original_payloads = state_payloads(
        states, coordinates, manifest, reference_variant="ORIGINAL"
    )
    permuted_payloads = state_payloads(
        states, coordinates, manifest, reference_variant="TARGET_REFERENCE_PERMUTATION"
    )
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    feature_start = time.perf_counter()
    original_features, original_stability = execute_features(original_payloads)
    permuted_features, permuted_stability = execute_features(permuted_payloads)
    feature_seconds = time.perf_counter() - feature_start
    original_features = original_features.merge(
        q[["stateId", "qHat", "successes"]], on="stateId", validate="one_to_one"
    )
    permuted_features = permuted_features.merge(
        q[["stateId", "qHat", "successes"]], on="stateId", validate="one_to_one"
    )
    features = (
        pd.concat([original_features, permuted_features], ignore_index=True)
        .sort_values(
            ["referenceVariant", "candidateId", "matrixRole", "landmark", "matrixIndex"]
        )
        .reset_index(drop=True)
    )
    stability_rows = pd.concat(
        [original_stability, permuted_stability], ignore_index=True
    )
    stability_summary = kernel_stability_summary(stability_rows)

    replay_start = time.perf_counter()
    replay_features, replay_stability = execute_features(original_payloads)
    replay_seconds = time.perf_counter() - replay_start
    feature_replay_exact = frame_hash(
        original_features.drop(columns=["qHat", "successes"])
    ) == frame_hash(replay_features)
    stability_replay_exact = frame_hash(original_stability) == frame_hash(
        replay_stability
    )
    if not (feature_replay_exact and stability_replay_exact):
        raise RuntimeError("feature replay failed")

    original_predictions, original_registry = model_predictions(original_features, q)
    permuted_predictions, permuted_registry = model_predictions(permuted_features, q)
    predictions = append_frozen_predictors(
        pd.concat([original_predictions, permuted_predictions], ignore_index=True)
    )
    registry = pd.concat([original_registry, permuted_registry], ignore_index=True)
    metrics = metrics_table(predictions)
    bootstraps = bootstrap_metrics(predictions)
    permutations = development_label_permutations(original_features, q, metrics)
    gates = gate_table(metrics, bootstraps, permutations, True)
    coordinate_established = bool(gates["candidateCoordinateGatePassed"].all())

    # The basin-blind model is adjudicated with the same rank/prior standard,
    # without pretending it supplies a target-conditioned Brier increment.
    basin_passes = []
    for candidate in CANDIDATES:
        metric = metrics[
            metrics["referenceVariant"].eq("ORIGINAL")
            & metrics["candidateId"].eq(candidate)
            & metrics["modelId"].eq("BASIN_BLIND_GENERATOR")
        ].iloc[0]
        boot = bootstraps[
            bootstraps["candidateId"].eq(candidate)
            & bootstraps["modelId"].eq("BASIN_BLIND_GENERATOR")
        ]["spearmanQHat"].dropna()
        prior_metric = metrics[
            metrics["referenceVariant"].eq("ORIGINAL")
            & metrics["candidateId"].eq(candidate)
            & metrics["modelId"].eq("DEVELOPMENT_PRIOR")
        ].iloc[0]
        blind_brier = bootstraps[
            bootstraps["candidateId"].eq(candidate)
            & bootstraps["modelId"].eq("BASIN_BLIND_GENERATOR")
        ]["brierScorePerBranch"].to_numpy()
        prior_brier = bootstraps[
            bootstraps["candidateId"].eq(candidate)
            & bootstraps["modelId"].eq("DEVELOPMENT_PRIOR")
        ]["brierScorePerBranch"].to_numpy()
        basin_passes.append(
            bool(
                metric.spearmanQHat > 0.5
                and np.quantile(boot, 0.025) > 0.3
                and prior_metric.brierScorePerBranch > metric.brierScorePerBranch
                and np.quantile(prior_brier - blind_brier, 0.025) > 0
            )
        )
    basin_blind_established = all(basin_passes)
    geometry_passes = []
    for candidate in CANDIDATES:
        metric = metrics[
            metrics["referenceVariant"].eq("ORIGINAL")
            & metrics["candidateId"].eq(candidate)
            & metrics["modelId"].eq("TARGET_GEOMETRY_CONTROL")
        ].iloc[0]
        boot = bootstraps[
            bootstraps["candidateId"].eq(candidate)
            & bootstraps["modelId"].eq("TARGET_GEOMETRY_CONTROL")
        ]["spearmanQHat"].dropna()
        geometry_passes.append(
            bool(metric.spearmanQHat > 0.5 and np.quantile(boot, 0.025) > 0.3)
        )

    if coordinate_established:
        classifications = ["GENERATOR_COMMITTOR_COORDINATE_ESTABLISHED"]
        if basin_blind_established:
            classifications.append("BASIN_BLIND_GENERATOR_COMMITTOR_SIGNAL")
        else:
            classifications.append("RETROSPECTIVE_BASIN_CONDITIONED_GENERATOR_SIGNAL")
        classifications.append("NOT_PROMOTABLE_AS_CONFIRMED")
        next_theme = "UNTOUCHED_GENERATOR_COORDINATE_CONFIRMATION"
    elif all(geometry_passes):
        classifications = [
            "ORDINARY_TARGET_GEOMETRY_SUFFICIENT",
            "EXACT_GARD_GENERATOR_NOT_INCREMENTAL",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "ONLINE_TARGET_BASIN_IDENTIFIABILITY_OR_CLOSEOUT"
    else:
        classifications = [
            "EXACT_GARD_GENERATOR_FEATURES_MISS_COMMITTOR_SIGNAL",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "MULTISTEP_GENERATOR_PROPAGATOR_OR_MEMORY_STATE_AUDIT"

    make_figures(original_features, predictions, metrics, stability_summary, gates)
    for name in (
        "preregistration.yaml",
        "decision_record.md",
        "fixture_results.parquet",
        "kernel_seed_manifest.parquet",
        "seed_firewall.json",
        "immutable_prior_validation.json",
        "source_grounding_report.md",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
        "model_registry.yaml",
    ):
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    if amendment is not None:
        shutil.copy2(amendment_path, BUILD_ROOT / amendment_path.name)
        shutil.copy2(
            LOOP_ROOT / "technical_amendment_ledger.csv",
            BUILD_ROOT / "technical_amendment_ledger.csv",
        )
        for failed_attempt in sorted(LOOP_ROOT.glob("failed_attempt_*.md")):
            shutil.copy2(failed_attempt, BUILD_ROOT / failed_attempt.name)
    BASE.write_parquet(BUILD_ROOT / "generator_feature_results.parquet", features)
    BASE.write_parquet(BUILD_ROOT / "kernel_moment_stability.parquet", stability_rows)
    BASE.write_parquet(
        BUILD_ROOT / "kernel_stability_summary.parquet", stability_summary
    )
    BASE.write_parquet(BUILD_ROOT / "fitted_model_registry.parquet", registry)
    BASE.write_parquet(BUILD_ROOT / "prediction_results.parquet", predictions)
    BASE.write_parquet(BUILD_ROOT / "metric_results.parquet", metrics)
    BASE.write_parquet(BUILD_ROOT / "bootstrap_results.parquet", bootstraps)
    BASE.write_parquet(
        BUILD_ROOT / "development_label_permutations.parquet", permutations
    )
    BASE.write_parquet(BUILD_ROOT / "scientific_gate_results.parquet", gates)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l29.classification.v1",
            "researchStepId": LOOP_ID,
            "classifications": classifications,
            "coordinateEstablishedBothCandidates": coordinate_established,
            "basinBlindEstablishedBothCandidates": basin_blind_established,
            "retrospectiveBasinConditioned": True,
            "confirmatory": False,
            "priorStatusesChanged": False,
        },
    )
    pd.DataFrame(
        columns=[
            "stage",
            "candidateId",
            "matrixIndex",
            "landmark",
            "exceptionClass",
            "exceptionMessage",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    checks = {
        "featureReplayExact": feature_replay_exact,
        "stabilityReplayExact": stability_replay_exact,
        "fixtureGatePassed": bool(fixtures["passed"].all()),
        "modelReplayExact": bool(registry["exactReplay"].all()),
        "stateCountExact": len(original_features) == 200,
        "permutedReferenceCountExact": len(permuted_features) == 200,
        "seedFirewallPassed": json.loads(
            (LOOP_ROOT / "seed_firewall.json").read_text()
        )["status"]
        == "PASS",
        "immutablePriorPassed": prior["unchanged"],
        "technicalAmendmentScientificContractUnchanged": amendment is None
        or not amendment["scientificContractChanged"],
    }
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l29.regeneration_validation.v1",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "featureFrameSha256": frame_hash(
                original_features.drop(columns=["qHat", "successes"])
            ),
            "replayFrameSha256": frame_hash(replay_features),
        },
    )
    if not all(checks.values()):
        raise RuntimeError("regeneration gate failed")
    runtime = {
        "schema": "eidosoma.e01.s19_l29.runtime.v1",
        "researchStepId": LOOP_ID,
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": WORKERS,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - start_wall,
        "controllerCpuHours": (time.process_time() - start_cpu) / 3600,
        "workerCpuHours": (feature_seconds + replay_seconds) * WORKERS / 3600,
        "kernelSamplesPerState": KERNEL_SAMPLES,
        "stateCount": 200,
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
        "schema": "eidosoma.e01.s19_l29.storage_validation.v1",
        "retainedBytes": retained,
        "retainedGiBCeiling": 25,
        "temporaryBytes": temporary,
        "temporaryGiBCeiling": 75,
    }
    storage["status"] = (
        "PASS" if retained < 25 * 2**30 and temporary < 75 * 2**30 else "FAIL"
    )
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        metrics, stability_summary, gates, classifications, runtime, next_theme
    )
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L29_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L29 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Held-out coordinate established in both candidates:** `{coordinate_established}`.\n\n**Next bounded theme:** `{next_theme}`.\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L29-promotion-stage")
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(BUILD_ROOT, stage)
    if LOOP_ROOT.exists():
        shutil.rmtree(LOOP_ROOT)
    os.replace(stage, LOOP_ROOT)
    shutil.rmtree(BUILD_ROOT)
    manifest_out = json.loads((LOOP_ROOT / "artifact_manifest.json").read_text())
    if any(
        sha256_file(LOOP_ROOT / item["path"]) != item["sha256"]
        for item in manifest_out["files"]
    ):
        raise RuntimeError("artifact hash failure")
    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md",
        report.replace("# S19-L29", "# S19 current handoff — S19-L29", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": None,
            "nextAuthorizedLoop": "S19-L30",
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
                "coordinateEstablished": coordinate_established,
                "basinBlindEstablished": basin_blind_established,
                "nextTheme": next_theme,
                "runtime": runtime,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-lock", action="store_true")
    parser.add_argument("--register-technical-amendment", action="store_true")
    args = parser.parse_args()
    if args.prepare_lock:
        prepare_lock()
    elif args.register_technical_amendment:
        register_technical_amendment()
    else:
        execute()


if __name__ == "__main__":
    main()
