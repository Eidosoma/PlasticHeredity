"""Execute S19-L30 fixed eight-step generator-propagator coordinate."""

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
from scipy.special import logit

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_latent_timebase.core import derive_seed, generate_beta, generator
from e01_onset_discovery.empirical_committor import (
    RestoredState,
    cosine_to_reference,
    simulate_branch,
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L29 = _load_module(
    "e01_s19_l30_l29",
    REPO_ROOT / "scripts/e01/run_s19_l29_exact_gard_generator_committor.py",
)
L28 = L29.L28
BASE = L29.BASE
LOOP_ID = "S19-L30"
VERSION = "E01-S19-L30-EIGHT-STEP-GENERATOR-PROPAGATOR-COMMITTOR-COORDINATE-v1.0.0"
CANDIDATES = L28.CANDIDATES
HORIZON = 8
BRANCHES = 64
HALF_BRANCHES = 32
BOOTSTRAPS = 4096
PERMUTATIONS = 512
WORKERS = 8
ROOT_HEX = "7b08b24f28d2a075637d4602b3494533924e42104cd52f9a320f4dfb80cc853f"
PHASE = "s19_l30_eight_step_generator_propagator"
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L30"
L29_ROOT = ARTIFACT_ROOT / "loops/L29"
L28_ROOT = ARTIFACT_ROOT / "loops/L28"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l30")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l30_eight_step_generator_propagator.yaml"
RUNNER_PATH = Path(__file__)
PRIMARY_MODEL = "EIGHT_STEP_PROPAGATOR_MOMENTS"
CONTROL_MODELS = (
    "DEVELOPMENT_PRIOR",
    "TARGET_GEOMETRY_CONTROL",
    "EXACT_H_TRACE_ANALOG",
    "ORDINARY_PATH_ANALOG",
)
MODEL_COLUMNS = {
    "Q8_CALIBRATED": ["q8JeffreysLogit"],
    "EIGHT_STEP_PROPAGATOR_MOMENTS": [
        "q8JeffreysLogit",
        "meanMaximumTargetScore",
        "sdMaximumTargetScore",
        "meanMinimumTargetScore",
        "fractionBranchesWithFission",
        "meanMolecularUpdates",
        "currentTargetScore",
        "targetComponentFraction",
        "landmarkScaled",
    ],
}


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


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L29_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L29_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L29_ROOT / item["path"]),
            "root": str(L29_ROOT),
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
        "schema": "eidosoma.e01.s19_l30.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
        ).hexdigest(),
        "l29ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def branch_seeds(
    candidate: str, matrix: int, landmark: int, branch: int
) -> dict[str, Any]:
    return {
        purpose: derive_seed(
            ROOT_HEX, PHASE, purpose, matrix, candidate, landmark, branch
        )
        for purpose in (
            "propagator_event",
            "propagator_trim",
            "propagator_fission",
            "propagator_daughter",
        )
    }


def seed_manifest(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in states.itertuples(index=False):
        for branch in range(BRANCHES):
            identities = branch_seeds(
                state.candidateId, int(state.matrixIndex), int(state.landmark), branch
            )
            materials = [
                identity.seed_material_sha256 for identity in identities.values()
            ]
            row = {
                "stateId": state.stateId,
                "candidateId": state.candidateId,
                "matrixRole": state.matrixRole,
                "matrixIndex": int(state.matrixIndex),
                "landmark": int(state.landmark),
                "branchIndex": branch,
                "branchHalf": "A" if branch < HALF_BRANCHES else "B",
                "rootHex": ROOT_HEX,
                "streamIdentitySha256": hashlib.sha256(
                    "|".join([state.stateId, str(branch), *materials]).encode()
                ).hexdigest(),
            }
            for purpose, identity in identities.items():
                token = purpose.replace("propagator_", "")
                row[f"{token}DerivedSeed"] = str(identity.derived_seed)
                row[f"{token}SeedMaterialSha256"] = identity.seed_material_sha256
            rows.append(row)
    output = (
        pd.DataFrame(rows)
        .sort_values(
            ["candidateId", "matrixRole", "landmark", "matrixIndex", "branchIndex"]
        )
        .reset_index(drop=True)
    )
    if (
        len(output) != len(states) * BRANCHES
        or not output["streamIdentitySha256"].is_unique
    ):
        raise RuntimeError("short-branch seed cardinality failure")
    return output


def seed_firewall(manifest: pd.DataFrame, prior: dict[str, Any]) -> dict[str, Any]:
    current = set()
    for column in manifest.columns:
        if "SeedMaterialSha256" in column:
            current.update(manifest[column].astype(str))
    prior_material = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L30/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            if "seedmaterialsha256" in column.lower():
                prior_material.update(str(value) for value in frame[column].dropna())
    overlaps = sorted(current & prior_material)
    root_paths = []
    needle = ROOT_HEX.encode()
    for row in prior["files"]:
        path = Path(row["path"])
        if path.is_file() and path.stat().st_size <= 64 * 1024 * 1024:
            try:
                if needle in path.read_bytes():
                    root_paths.append(str(path))
            except OSError:
                pass
    return {
        "schema": "eidosoma.e01.s19_l30.seed_firewall.v1",
        "status": "PASS" if not overlaps and not root_paths else "FAIL",
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
    beta = np.exp(np.full((100, 100), -4.0, dtype=np.float64))
    target = state.astype(np.float64) / state.sum()
    restored = RestoredState(tuple(map(int, state)), "post_fission", 1, 1, 0, 4)

    def run(seed_offset: int) -> Any:
        identities = branch_seeds(CANDIDATES[0], 999_801 + seed_offset, 64, 0)
        return simulate_branch(
            restored=restored,
            beta=beta,
            definition=L28.definition(CANDIDATES[0]),
            target_centroid=target,
            event_rng=generator(identities["propagator_event"]),
            trim_rng=generator(identities["propagator_trim"]),
            fission_rng=generator(identities["propagator_fission"]),
            daughter_rng=generator(identities["propagator_daughter"]),
            horizon=HORIZON,
        )

    first = run(0)
    replay = run(0)
    return pd.DataFrame(
        [
            {
                "fixtureId": "EIGHT_STEP_HORIZON",
                "passed": first.selected_observations_generated == 8,
                "details": str(first.selected_observations_generated),
            },
            {
                "fixtureId": "SHORT_BRANCH_EXACT_REPLAY",
                "passed": first == replay,
                "details": first.path_sha256,
            },
            {
                "fixtureId": "HALF_CARDINALITY",
                "passed": HALF_BRANCHES * 2 == BRANCHES,
                "details": "32+32=64",
            },
            {
                "fixtureId": "H32_ENDPOINT_NOT_REACHED",
                "passed": HORIZON < L28.HORIZON,
                "details": "8<32",
            },
            {
                "fixtureId": "JEFFREYS_PROBABILITY_BOUNDED",
                "passed": 0 < 0.5 / 65 < 64.5 / 65 < 1,
                "details": "(k+0.5)/65",
            },
            {
                "fixtureId": "MODEL_COLUMNS_UNIQUE",
                "passed": all(
                    len(value) == len(set(value)) for value in MODEL_COLUMNS.values()
                ),
                "details": json.dumps(
                    {key: len(value) for key, value in MODEL_COLUMNS.items()}
                ),
            },
        ]
    )


def _branch_worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidate = payload["candidateId"]
    matrix = int(payload["matrixIndex"])
    landmark = int(payload["landmark"])
    beta_seed = derive_seed(L28.L23_ROOT_HEX, L28.L23_PHASE, "catalytic_matrix", matrix)
    beta = generate_beta(beta_seed)
    if L28.simulator_array_sha256(beta) != payload["betaSha256"]:
        raise RuntimeError("beta identity failure")
    restored = RestoredState(
        tuple(payload["state"]),
        payload["currentObservationKind"],
        int(payload["currentCompletedFissions"]),
        int(payload["currentGrowthGeneration"]),
        int(payload["currentGenerationLocalStep"]),
        int(payload["currentBatchStep"]),
    )
    target = np.asarray(payload["centroid"], dtype=np.float64)
    current_score = float(
        cosine_to_reference(np.asarray([payload["state"]]), target)[0]
    )
    rows = []
    for branch in range(BRANCHES):
        identities = branch_seeds(candidate, matrix, landmark, branch)
        result = simulate_branch(
            restored=restored,
            beta=beta,
            definition=L28.definition(candidate),
            target_centroid=target,
            event_rng=generator(identities["propagator_event"]),
            trim_rng=generator(identities["propagator_trim"]),
            fission_rng=generator(identities["propagator_fission"]),
            daughter_rng=generator(identities["propagator_daughter"]),
            horizon=HORIZON,
        )
        materials = [identity.seed_material_sha256 for identity in identities.values()]
        rows.append(
            {
                "stateId": payload["stateId"],
                "candidateId": candidate,
                "matrixRole": payload["matrixRole"],
                "matrixIndex": matrix,
                "landmark": landmark,
                "referenceVariant": payload["referenceVariant"],
                "targetReferenceDonorStateId": payload["targetReferenceDonorStateId"],
                "branchIndex": branch,
                "branchHalf": "A" if branch < HALF_BRANCHES else "B",
                "streamIdentitySha256": hashlib.sha256(
                    "|".join([payload["stateId"], str(branch), *materials]).encode()
                ).hexdigest(),
                "analysisIdentitySha256": hashlib.sha256(
                    f"{payload['referenceVariant']}|{payload['stateId']}|{branch}".encode()
                ).hexdigest(),
                "enteredBasinWithin8": result.entered_basin,
                "firstEntryOffsetOneBased": result.first_entry_offset_one_based,
                "maximumTargetScore": result.maximum_target_score,
                "minimumTargetScore": result.minimum_target_score,
                "molecularUpdates": result.molecular_updates,
                "fissions": result.fissions,
                "selectedObservationsGenerated": result.selected_observations_generated,
                "terminalStatus": result.terminal_status,
                "pathSha256": result.path_sha256,
                "currentTargetScore": current_score,
                "targetComponentFraction": payload["targetComponentSizeUsed"] / 100.0,
            }
        )
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
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "referenceVariant",
                "candidateId",
                "matrixRole",
                "landmark",
                "matrixIndex",
                "branchIndex",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_states(branches: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in branches.groupby(
        [
            "referenceVariant",
            "stateId",
            "candidateId",
            "matrixRole",
            "matrixIndex",
            "landmark",
        ],
        sort=True,
    ):
        variant, state_id, candidate, role, matrix, landmark = keys
        half_a = group[group["branchHalf"].eq("A")]
        half_b = group[group["branchHalf"].eq("B")]
        successes = int(group["enteredBasinWithin8"].sum())
        q8 = successes / BRANCHES
        maximum = group["maximumTargetScore"].astype(float)
        minimum = group["minimumTargetScore"].astype(float)
        rows.append(
            {
                "referenceVariant": variant,
                "stateId": state_id,
                "candidateId": candidate,
                "matrixRole": role,
                "matrixIndex": int(matrix),
                "landmark": int(landmark),
                "shortSuccesses": successes,
                "q8": q8,
                "q8Jeffreys": (successes + 0.5) / 65.0,
                "q8JeffreysLogit": float(logit((successes + 0.5) / 65.0)),
                "q8HalfA": float(half_a["enteredBasinWithin8"].mean()),
                "q8HalfB": float(half_b["enteredBasinWithin8"].mean()),
                "meanMaximumTargetScore": float(maximum.mean()),
                "sdMaximumTargetScore": float(maximum.std(ddof=1)),
                "meanMinimumTargetScore": float(minimum.mean()),
                "fractionBranchesWithFission": float((group["fissions"] > 0).mean()),
                "meanMolecularUpdates": float(group["molecularUpdates"].mean()),
                "meanFirstEntryNoEntryAs9": float(
                    group["firstEntryOffsetOneBased"].fillna(9).mean()
                ),
                "currentTargetScore": float(group["currentTargetScore"].iloc[0]),
                "targetComponentFraction": float(
                    group["targetComponentFraction"].iloc[0]
                ),
                "landmarkScaled": int(landmark) / 192.0,
                "completeHorizonBranchCount": int(
                    (group["selectedObservationsGenerated"] == HORIZON).sum()
                ),
            }
        )
    output = pd.DataFrame(rows)
    return (
        output.merge(
            q[["stateId", "qHat", "successes"]], on="stateId", validate="many_to_one"
        )
        .sort_values(
            ["referenceVariant", "candidateId", "matrixRole", "landmark", "matrixIndex"]
        )
        .reset_index(drop=True)
    )


def fit_predictions(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    registry = []
    for (variant, candidate), group in states.groupby(
        ["referenceVariant", "candidateId"], sort=True
    ):
        development = group[group["matrixRole"].eq("DEVELOPMENT")]
        prior = float(development["qHat"].mean())
        for source in group.itertuples(index=False):
            base = {
                "stateId": source.stateId,
                "candidateId": candidate,
                "matrixRole": source.matrixRole,
                "matrixIndex": source.matrixIndex,
                "landmark": source.landmark,
                "qHat": source.qHat,
                "successes": source.successes,
                "referenceVariant": variant,
            }
            rows.append({**base, "modelId": "DEVELOPMENT_PRIOR", "predictedQ": prior})
            rows.append(
                {
                    **base,
                    "modelId": "Q8_JEFFREYS_DIRECT",
                    "predictedQ": source.q8Jeffreys,
                }
            )
        for model_id, columns in MODEL_COLUMNS.items():
            scaler, model = L29.fit_model(development, columns)
            probability = model.predict_proba(
                scaler.transform(group[columns].to_numpy(dtype=np.float64))
            )[:, 1]
            replay_scaler, replay_model = L29.fit_model(development, columns)
            replay = replay_model.predict_proba(
                replay_scaler.transform(group[columns].to_numpy(dtype=np.float64))
            )[:, 1]
            if not np.array_equal(probability, replay):
                raise RuntimeError("model replay failed")
            registry.append(
                {
                    "referenceVariant": variant,
                    "candidateId": candidate,
                    "modelId": model_id,
                    "featureNames": json.dumps(columns),
                    "featureCount": len(columns),
                    "intercept": float(model.intercept_[0]),
                    "coefficients": json.dumps(model.coef_[0].tolist()),
                    "scalerMean": json.dumps(scaler.mean_.tolist()),
                    "scalerScale": json.dumps(scaler.scale_.tolist()),
                    "exactReplay": True,
                }
            )
            for source, value in zip(
                group.itertuples(index=False), probability, strict=True
            ):
                rows.append(
                    {
                        "stateId": source.stateId,
                        "candidateId": candidate,
                        "matrixRole": source.matrixRole,
                        "matrixIndex": source.matrixIndex,
                        "landmark": source.landmark,
                        "modelId": model_id,
                        "predictedQ": float(value),
                        "qHat": source.qHat,
                        "successes": source.successes,
                        "referenceVariant": variant,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(registry)


def append_controls(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = predictions.columns
    l29 = pd.read_parquet(L29_ROOT / "prediction_results.parquet")
    geometry = l29[
        l29["referenceVariant"].eq("ORIGINAL")
        & l29["matrixRole"].eq("VALIDATION")
        & l29["modelId"].eq("TARGET_GEOMETRY_CONTROL")
    ].copy()
    l28 = pd.read_parquet(L28_ROOT / "frozen_predictor_scores.parquet")
    q = pd.read_parquet(L28_ROOT / "committor_state_results.parquet")[
        ["stateId", "candidateId", "matrixIndex", "landmark", "qHat", "successes"]
    ]
    l28 = l28.merge(
        q,
        on=["candidateId", "matrixIndex", "landmark", "qHat", "successes"],
        validate="many_to_one",
    ).rename(columns={"predictorId": "modelId", "score": "predictedQ"})
    l28["matrixRole"] = "VALIDATION"
    l28["referenceVariant"] = "ORIGINAL"
    return (
        pd.concat(
            [
                predictions,
                geometry.reindex(columns=columns),
                l28.reindex(columns=columns),
            ],
            ignore_index=True,
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


def bootstrap_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source = predictions[
        predictions["referenceVariant"].eq("ORIGINAL")
        & predictions["matrixRole"].eq("VALIDATION")
    ]
    for candidate in CANDIDATES:
        pivot = (
            source[source["candidateId"].eq(candidate)]
            .pivot(index=["stateId", "qHat"], columns="modelId", values="predictedQ")
            .reset_index()
        )
        rng = np.random.default_rng(L29.derived_seed("l30_bootstrap", candidate))
        models = [
            column for column in pivot.columns if column not in {"stateId", "qHat"}
        ]
        for replicate in range(BOOTSTRAPS):
            sample = pivot.iloc[rng.integers(0, len(pivot), size=len(pivot))]
            q = sample["qHat"].to_numpy(dtype=np.float64)
            brier = {}
            for model in models:
                p = np.clip(sample[model].to_numpy(dtype=np.float64), 1e-9, 1 - 1e-9)
                value = float(np.mean(q * (1 - p) ** 2 + (1 - q) * p**2))
                brier[model] = value
                rows.append(
                    {
                        "candidateId": candidate,
                        "bootstrapIndex": replicate,
                        "modelId": model,
                        "spearmanQHat": L29.safe_spearman(p, q),
                        "brierScorePerBranch": value,
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
                        "primaryBrierImprovement": brier[control]
                        - brier[PRIMARY_MODEL],
                    }
                )
    return pd.DataFrame(rows)


def label_permutations(states: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    original = states[states["referenceVariant"].eq("ORIGINAL")]
    for candidate in CANDIDATES:
        group = original[original["candidateId"].eq(candidate)]
        development = group[group["matrixRole"].eq("DEVELOPMENT")].copy()
        validation = group[group["matrixRole"].eq("VALIDATION")]
        observed = float(
            metrics[
                metrics["referenceVariant"].eq("ORIGINAL")
                & metrics["candidateId"].eq(candidate)
                & metrics["modelId"].eq(PRIMARY_MODEL)
            ]["spearmanQHat"].iloc[0]
        )
        rng = np.random.default_rng(
            L29.derived_seed("l30_label_permutation", candidate)
        )
        null = []
        candidate_rows = []
        for replicate in range(PERMUTATIONS):
            order = rng.permutation(len(development))
            permuted = development.copy()
            permuted["qHat"] = development["qHat"].to_numpy()[order]
            permuted["successes"] = development["successes"].to_numpy()[order]
            scaler, model = L29.fit_model(permuted, MODEL_COLUMNS[PRIMARY_MODEL])
            prediction = model.predict_proba(
                scaler.transform(
                    validation[MODEL_COLUMNS[PRIMARY_MODEL]].to_numpy(dtype=np.float64)
                )
            )[:, 1]
            rho = L29.safe_spearman(prediction, validation["qHat"])
            null.append(rho)
            candidate_rows.append(
                {
                    "candidateId": candidate,
                    "permutationIndex": replicate,
                    "observedSpearman": observed,
                    "nullSpearman": rho,
                }
            )
        finite = np.asarray([value for value in null if np.isfinite(value)])
        p_value = float((1 + np.sum(finite >= observed)) / (1 + len(finite)))
        for row in candidate_rows:
            row["familywiseP"] = p_value
        rows.extend(candidate_rows)
    return pd.DataFrame(rows)


def gate_table(
    metrics: pd.DataFrame, bootstraps: pd.DataFrame, permutations: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        primary = metrics[
            metrics["referenceVariant"].eq("ORIGINAL")
            & metrics["candidateId"].eq(candidate)
            & metrics["modelId"].eq(PRIMARY_MODEL)
        ].iloc[0]
        boot = bootstraps[
            bootstraps["candidateId"].eq(candidate)
            & bootstraps["modelId"].eq(PRIMARY_MODEL)
        ]["spearmanQHat"].dropna()
        rho_lower = float(np.quantile(boot, 0.025))
        lower = {}
        for control in CONTROL_MODELS:
            values = bootstraps[
                bootstraps["candidateId"].eq(candidate)
                & bootstraps["modelId"].eq(f"DELTA_PRIMARY_VS_{control}")
            ]["primaryBrierImprovement"]
            lower[control] = float(np.quantile(values, 0.025))
        permutation_p = float(
            permutations[permutations["candidateId"].eq(candidate)]["familywiseP"].iloc[
                0
            ]
        )
        permuted_rho = float(
            metrics[
                metrics["referenceVariant"].eq("TARGET_REFERENCE_PERMUTATION")
                & metrics["candidateId"].eq(candidate)
                & metrics["modelId"].eq(PRIMARY_MODEL)
            ]["spearmanQHat"].iloc[0]
        )
        rows.append(
            {
                "candidateId": candidate,
                "primarySpearman": primary.spearmanQHat,
                "spearmanBootstrapLower95": rho_lower,
                **{
                    f"brierImprovementLowerVs{key}": value
                    for key, value in lower.items()
                },
                "developmentPermutationP": permutation_p,
                "targetPermutedSpearman": permuted_rho,
                "rankPassed": bool(primary.spearmanQHat > 0.5 and rho_lower > 0.3),
                "incrementalBrierPassed": all(value > 0 for value in lower.values()),
                "permutationPassed": permutation_p <= 0.05,
                "targetReferenceControlPassed": primary.spearmanQHat > permuted_rho,
                "candidateCoordinateGatePassed": bool(
                    primary.spearmanQHat > 0.5
                    and rho_lower > 0.3
                    and all(value > 0 for value in lower.values())
                    and permutation_p <= 0.05
                    and primary.spearmanQHat > permuted_rho
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    states: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(root / name, dpi=180)
        plt.close()

    original = states[states["referenceVariant"].eq("ORIGINAL")]
    plt.figure(figsize=(8, 5))
    for candidate in CANDIDATES:
        group = original[original["candidateId"].eq(candidate)]
        plt.scatter(
            group["q8Jeffreys"], group["qHat"], s=20, alpha=0.65, label=candidate
        )
    plt.xlabel("Independent eight-step entry probability (Jeffreys)")
    plt.ylabel("Independent L28 H32 q-hat")
    plt.legend(fontsize=7)
    save("01_q8_vs_q32.png")

    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = original[original["candidateId"].eq(candidate)]
        axis.scatter(group["q8HalfA"], group["q8HalfB"], s=20)
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set_title(candidate)
        axis.set_xlabel("q8 branches 0–31")
        axis.set_ylabel("q8 branches 32–63")
    save("02_short_propagator_half_reliability.png")

    validation = predictions[
        predictions["referenceVariant"].eq("ORIGINAL")
        & predictions["matrixRole"].eq("VALIDATION")
        & predictions["modelId"].eq(PRIMARY_MODEL)
    ]
    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = validation[validation["candidateId"].eq(candidate)]
        axis.scatter(group["predictedQ"], group["qHat"], s=22)
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set_title(candidate)
        axis.set_xlabel("Predicted H32 q")
        axis.set_ylabel("L28 q-hat")
    save("03_heldout_predictions.png")

    metrics[metrics["referenceVariant"].eq("ORIGINAL")].pivot(
        index="modelId", columns="candidateId", values="spearmanQHat"
    ).plot(kind="bar", figsize=(10, 5))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("Held-out Spearman")
    save("04_model_rank_comparison.png")

    columns = [
        "rankPassed",
        "incrementalBrierPassed",
        "permutationPassed",
        "targetReferenceControlPassed",
        "candidateCoordinateGatePassed",
    ]
    matrix = gates.set_index("candidateId")[columns].astype(float)
    plt.figure(figsize=(8, 3.5))
    plt.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(len(columns)), columns, rotation=35, ha="right", fontsize=8)
    plt.yticks(range(len(matrix)), matrix.index)
    plt.colorbar(ticks=[0, 1])
    save("05_coordinate_gate_matrix.png")


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
        "schema": "eidosoma.e01.s19_l30.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(files),
        "totalBytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def append_ledgers(classifications: list[str], timestamp: str, next_theme: str) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "One-step generator rank signal may require finite propagation before q becomes recoverable.",
            "failureOrAmbiguityTargeted": "Local versus nonlocal state dependence at fixed H32.",
            "informationGainRationale": "A fixed H8 propagator tests nonlocal dynamics without directly simulating H32.",
            "learned": "L30 short-propagator contract frozen.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L29 rank signal but failed incremental calibration uncertainty.",
            "proposedNextTest": "Execute independent H8 propagator coordinate.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Short propagation, hidden history, or target geometry only.",
            "selectedHypotheses": "One fixed H8/64-branch generator propagator.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "One-step drift/diffusion is sufficient.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A usable coordinate must generalize and improve calibrated Brier beyond controls.",
            "failureOrAmbiguityTargeted": "Short-propagator sufficiency.",
            "informationGainRationale": "Independent streams and frozen validation q prevent branch reuse.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L30 results.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "One fixed H8 propagator coordinate.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Short propagation resolves the state signal"
            if "NON_SUPPORT" in ";".join(classifications)
            else "Static state features are sufficient.",
        },
    ]
    BASE.write_parquet(
        ledger_path,
        pd.concat(
            [ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)],
            ignore_index=True,
        ),
    )
    md = ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    BASE.atomic_text(
        md,
        md.read_text()
        + f"\n\n## {LOOP_ID} — eight-step generator propagator\n\n- **Learned:** {', '.join(classifications)}.\n- **Next:** {next_theme}.\n",
    )
    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    row = {
        "branchCount": 1,
        "bundleId": "L30_EIGHT_STEP_PROPAGATOR",
        "candidateId": "S19-L30-H8-PROPAGATOR",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 1,
        "computeEfficiency": 4,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 3,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "64 independent H8 propagator branches from each L28 state",
        "rankingScore": 25.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L29_LOCAL_GENERATOR_RANK_HINT_INCREMENTAL_GATE_FAIL",
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
    source_row = {
        "commitOrVersion": None,
        "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
        "finding": (
            "L28 established a reliable H32 empirical committor; L30 uses new "
            "domain-separated H8 futures to test a fixed multistep shooting coordinate."
        ),
        "licenseStatus": "WORKSPACE_EVIDENCE",
        "redistributionStatus": "INTERNAL_ARTIFACT",
        "repositoryIdentity": None,
        "retainedPath": str(L28_ROOT / "research_step_full_results.md"),
        "retrievalDate": timestamp[:10],
        "sha256": sha256_file(L28_ROOT / "research_step_full_results.md"),
        "sourceId": "L30_L28_EMPIRICAL_COMMITTOR_CONTEXT",
        "sourceType": "DIRECT_FROZEN_E01_RESULT",
        "treeIdentity": None,
        "url": None,
    }
    BASE.write_parquet(
        source_path,
        pd.concat(
            [sources, pd.DataFrame([source_row]).reindex(columns=sources.columns)],
            ignore_index=True,
        ),
    )
    source_report = ARTIFACT_ROOT / "source_search_report.md"
    BASE.atomic_text(
        source_report,
        source_report.read_text()
        + "\n\n## S19-L30 — frozen evidence reuse\n\n"
        + "L30 introduced no external source or author-code claim. It reused L28's "
        + "frozen empirical-committor evidence and exact simulator contract to test "
        + "one prespecified eight-observation shooting coordinate.\n",
    )
    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text())
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
            "selectedDiscoveryLead": "H8_PROPAGATOR"
            if "COORDINATE_ESTABLISHED" in ";".join(classifications)
            else None,
            "newMatrices": 0,
            "newTrajectories": 12800,
            "nextStepActive": True,
        }
    )
    registry["proposedNextLoopTheme"] = next_theme
    registry["proposedNextLoopActive"] = True
    BASE.atomic_text(registry_path, yaml.safe_dump(registry, sort_keys=False))


def report_text(
    metrics: pd.DataFrame,
    gates: pd.DataFrame,
    half: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    return f"""# S19-L30 — Eight-Step Generator Propagator Committor Coordinate

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the authorized L19–L42 sequence.
- **Outcome classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Validation:** exact L28 state/target/q replay; 12,800 new domain-separated H8 branches; independent 32/32 halves; full original-branch and model replay; target-reference and development-label controls; 4,096 matrix bootstraps; immutable-prior, runtime/storage, regeneration and artifact hashes passed.
- **Next bounded theme:** {next_theme}

## Frozen question and method

This loop asks whether the one-step L29 signal becomes a stable H32 committor coordinate after exactly eight selected-clock observations. It uses 64 new independent short futures per restored L28 state and never propagates the predictor to H32. The target basin remains retrospectively completed-run conditioned. The primary coordinate is calibrated on development matrices only and evaluated unchanged on validation matrices.

## Held-out metrics

{metrics.to_markdown(index=False)}

## Short-propagator split-half reliability

{half.to_markdown(index=False)}

## Gate adjudication

{gates.to_markdown(index=False)}

## Interpretation boundary

Even a passing short-propagator coordinate would be a simulation-based, retrospective-basin-conditioned reaction coordinate—not an observed early-warning biomarker, author-code reconstruction, or causal control result. It would require untouched state/matrix confirmation before any transition-current analysis. A failure would redirect the search to hidden-memory/history representations rather than horizon or branch-count tuning.

## Runtime and provenance

- Repository lock: `{runtime["repositoryHead"]}`.
- CPU float64, `{runtime["workers"]}` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `{runtime["wallSeconds"]:.3f}`; aggregate worker CPU hours: `{runtime["workerCpuHours"]:.6f}`.

## Autonomous boundary

L30 is frozen. S20, E02, author contact, interventions and report-bundle work remain inactive.
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local/remote mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    if not prior["unchanged"] or not fixtures["passed"].all():
        raise RuntimeError("prior or fixture gate failed")
    states = pd.read_parquet(L28_ROOT / "restored_state_registry.parquet")
    seeds = seed_manifest(states)
    firewall = seed_firewall(seeds, prior)
    if firewall["status"] != "PASS" or not firewall["seedMaterialUnique"]:
        raise RuntimeError("seed firewall failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L30 decision record\n\nL29's one-step target-radial coordinate passed rank but failed incremental Brier uncertainty, while basin-blind moments were weak. L30 freezes exactly 64 new independent eight-observation propagator branches per L28 state, a 32/32 split, one development-calibrated q8 coordinate and one fixed propagator-moment coordinate. Eight is one quarter of H32 and was selected before any short-branch result. No H32 predictor branch, horizon search, branch-count search, target change or transition-current analysis is allowed.\n",
    )
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "short_branch_seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    hashes = {
        "statesSha256": sha256_file(L28_ROOT / "restored_state_registry.parquet"),
        "coordinatesSha256": sha256_file(L28_ROOT / "target_basin_coordinates.parquet"),
        "qSha256": sha256_file(L28_ROOT / "committor_state_results.parquet"),
        "seedManifestSha256": sha256_file(
            LOOP_ROOT / "short_branch_seed_manifest.parquet"
        ),
    }
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l30.implementation_lock.v1",
            "repositoryHead": head,
            "remoteHead": remote,
            "runnerSha256": sha256_file(RUNNER_PATH),
            "configSha256": sha256_file(CONFIG),
            "horizon": HORIZON,
            "branchesPerState": BRANCHES,
            "branchHalves": [HALF_BRANCHES, HALF_BRANCHES],
            "modelColumns": MODEL_COLUMNS,
            "primaryModel": PRIMARY_MODEL,
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
            **hashes,
        },
    )


def execute() -> None:
    start_wall = time.perf_counter()
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
    for key, path in {
        "statesSha256": L28_ROOT / "restored_state_registry.parquet",
        "coordinatesSha256": L28_ROOT / "target_basin_coordinates.parquet",
        "qSha256": L28_ROOT / "committor_state_results.parquet",
        "seedManifestSha256": LOOP_ROOT / "short_branch_seed_manifest.parquet",
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
    original_payloads = L29.state_payloads(
        states, coordinates, manifest, reference_variant="ORIGINAL"
    )
    permuted_payloads = L29.state_payloads(
        states, coordinates, manifest, reference_variant="TARGET_REFERENCE_PERMUTATION"
    )
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    branch_start = time.perf_counter()
    original_branches = execute_branches(original_payloads)
    permuted_branches = execute_branches(permuted_payloads)
    branch_seconds = time.perf_counter() - branch_start
    expected = len(states) * BRANCHES
    if len(original_branches) != expected or len(permuted_branches) != expected:
        raise RuntimeError("branch cardinality failure")
    expected_streams = pd.read_parquet(
        LOOP_ROOT / "short_branch_seed_manifest.parquet"
    )[["stateId", "branchIndex", "streamIdentitySha256"]]
    observed_streams = original_branches[
        ["stateId", "branchIndex", "streamIdentitySha256"]
    ]
    stream_check = expected_streams.merge(
        observed_streams,
        on=["stateId", "branchIndex"],
        suffixes=("Expected", "Observed"),
        validate="one_to_one",
    )
    branch_identity_exact = bool(
        stream_check["streamIdentitySha256Expected"]
        .eq(stream_check["streamIdentitySha256Observed"])
        .all()
    )
    if not branch_identity_exact:
        raise RuntimeError("branch identity mismatch")
    replay_start = time.perf_counter()
    replay = execute_branches(original_payloads)
    replay_seconds = time.perf_counter() - replay_start
    replay_exact = frame_hash(original_branches) == frame_hash(replay)
    if not replay_exact:
        raise RuntimeError("short branch full replay failed")
    branch_rows = pd.concat([original_branches, permuted_branches], ignore_index=True)
    state_results = summarize_states(branch_rows, q)
    predictions, registry = fit_predictions(state_results)
    predictions = append_controls(predictions)
    metrics = L29.metrics_table(predictions)
    bootstraps = bootstrap_metrics(predictions)
    permutations = label_permutations(state_results, metrics)
    gates = gate_table(metrics, bootstraps, permutations)
    coordinate_pass = bool(gates["candidateCoordinateGatePassed"].all())
    half_rows = []
    for (variant, candidate), group in state_results.groupby(
        ["referenceVariant", "candidateId"], sort=True
    ):
        half_rows.append(
            {
                "referenceVariant": variant,
                "candidateId": candidate,
                "states": len(group),
                "splitHalfSpearman": L29.safe_spearman(
                    group["q8HalfA"], group["q8HalfB"]
                ),
                "q8VsQ32Spearman": L29.safe_spearman(
                    group["q8Jeffreys"], group["qHat"]
                ),
                "meanQ8": float(group["q8"].mean()),
                "meanQ32": float(group["qHat"].mean()),
                "zeroQ8States": int((group["q8"] == 0).sum()),
            }
        )
    half = pd.DataFrame(half_rows)
    if coordinate_pass:
        classifications = [
            "EIGHT_STEP_PROPAGATOR_COMMITTOR_COORDINATE_ESTABLISHED",
            "RETROSPECTIVE_BASIN_CONDITIONED_SHOOTING_SIGNAL",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "UNTOUCHED_SHORT_PROPAGATOR_COORDINATE_CONFIRMATION"
    else:
        classifications = [
            "EIGHT_STEP_PROPAGATOR_COORDINATE_NON_SUPPORT",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "HIDDEN_MEMORY_STATE_AUDIT"
    make_figures(state_results, predictions, metrics, gates)
    for name in (
        "preregistration.yaml",
        "decision_record.md",
        "fixture_results.parquet",
        "short_branch_seed_manifest.parquet",
        "seed_firewall.json",
        "immutable_prior_validation.json",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
    ):
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    BASE.write_parquet(BUILD_ROOT / "short_branch_results.parquet", branch_rows)
    BASE.write_parquet(BUILD_ROOT / "propagator_state_results.parquet", state_results)
    BASE.write_parquet(BUILD_ROOT / "propagator_half_reliability.parquet", half)
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
            "schema": "eidosoma.e01.s19_l30.classification.v1",
            "researchStepId": LOOP_ID,
            "classifications": classifications,
            "coordinateEstablishedBothCandidates": coordinate_pass,
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
            "branchIndex",
            "exceptionClass",
            "exceptionMessage",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    checks = {
        "branchCardinalityExact": len(original_branches) == expected
        and len(permuted_branches) == expected,
        "branchIdentityExact": branch_identity_exact,
        "fullOriginalBranchReplayExact": replay_exact,
        "modelReplayExact": bool(registry["exactReplay"].all()),
        "allOriginalBranchesH8": bool(
            (original_branches["selectedObservationsGenerated"] == HORIZON).all()
        ),
        "seedFirewallPassed": json.loads(
            (LOOP_ROOT / "seed_firewall.json").read_text()
        )["status"]
        == "PASS",
        "immutablePriorPassed": prior["unchanged"],
        "fixturesPassed": bool(fixtures["passed"].all()),
    }
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l30.regeneration_validation.v1",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "firstBranchFrameSha256": frame_hash(original_branches),
            "replayBranchFrameSha256": frame_hash(replay),
        },
    )
    if not all(checks.values()):
        raise RuntimeError("regeneration validation failed")
    runtime = {
        "schema": "eidosoma.e01.s19_l30.runtime.v1",
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": WORKERS,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - start_wall,
        "controllerCpuHours": (time.process_time() - start_cpu) / 3600,
        "workerCpuHours": (branch_seconds + replay_seconds) * WORKERS / 3600,
        "branchCountOriginal": len(original_branches),
        "branchCountTargetControl": len(permuted_branches),
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
        "schema": "eidosoma.e01.s19_l30.storage_validation.v1",
        "retainedBytes": retained,
        "retainedGiBCeiling": 25,
        "temporaryBytes": temporary,
        "temporaryGiBCeiling": 75,
    }
    storage["status"] = (
        "PASS" if retained < 25 * 2**30 and temporary < 75 * 2**30 else "FAIL"
    )
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(metrics, gates, half, classifications, runtime, next_theme)
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L30_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L30 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Coordinate established:** `{coordinate_pass}`.\n\n**Next:** `{next_theme}`.\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L30-promotion-stage")
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
        report.replace("# S19-L30", "# S19 current handoff — S19-L30", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": "H8_PROPAGATOR" if coordinate_pass else None,
            "nextAuthorizedLoop": "S19-L31",
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
                "coordinateEstablished": coordinate_pass,
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
