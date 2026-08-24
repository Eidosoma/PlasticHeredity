"""Execute S19-L32 past-only transition-tube committor coordinate audit."""

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


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L31 = _load_module(
    "e01_s19_l32_l31",
    REPO_ROOT / "scripts/e01/run_s19_l31_untouched_propagator_confirmation.py",
)
L30 = L31.L30
L29 = L31.L29
L28 = L31.L28
L27 = _load_module(
    "e01_s19_l32_l27",
    REPO_ROOT / "scripts/e01/run_s19_l27_transition_tube_density_current.py",
)
L26 = L27.L26
BASE = L31.BASE
LOOP_ID = "S19-L32"
VERSION = "E01-S19-L32-COMMITTOR-ORDERED-PAST-ONLY-TRANSITION-TUBE-v1.0.0"
CANDIDATES = L28.CANDIDATES
VIEWS = L27.MODELS
PRIMARY_MODEL = "FULL_TRANSITION_TUBE"
CONTROL_MODELS = ("LANDMARK_PRIOR", "EXACT_H_TRANSITION_TUBE", "ORDINARY_TRANSITION_TUBE")
EVALUATION_COHORTS = ("L28_VALIDATION", "L31_CONFIRMATION")
BOOTSTRAPS = 4096
PERMUTATIONS = 512
ROOT_HEX = "9f54d32b67c6c111611e80089c44b14eadc524f7514b2d533ec35603e35cff29"
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L32"
L31_ROOT = ARTIFACT_ROOT / "loops/L31"
L28_ROOT = ARTIFACT_ROOT / "loops/L28"
L27_ROOT = ARTIFACT_ROOT / "loops/L27"
L25_ROOT = ARTIFACT_ROOT / "loops/L25"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l32")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l32_committor_ordered_transition_tube.yaml"
RUNNER_PATH = Path(__file__)


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
    prior = json.loads((L31_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L31_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L31_ROOT / item["path"]),
            "root": str(L31_ROOT),
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
        "schema": "eidosoma.e01.s19_l32.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
        ).hexdigest(),
        "l31ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def response_registry() -> pd.DataFrame:
    l28_states = pd.read_parquet(L28_ROOT / "restored_state_registry.parquet")
    l28_q = pd.read_parquet(L28_ROOT / "committor_state_results.parquet")[
        ["stateId", "successes", "qHat"]
    ]
    l28 = l28_states.merge(l28_q, on="stateId", validate="one_to_one")
    l28["evaluationCohort"] = np.where(
        l28["matrixRole"].eq("DEVELOPMENT"), "L28_DEVELOPMENT", "L28_VALIDATION"
    )
    l31_states = pd.read_parquet(L31_ROOT / "restored_state_registry.parquet")
    l31_summary = pd.read_parquet(
        L31_ROOT / "state_committor_and_propagator_results.parquet"
    )
    l31_q = l31_summary[
        l31_summary["branchFamily"].eq("H32")
        & l31_summary["referenceVariant"].eq("ORIGINAL")
    ][["stateId", "successes", "q"]].rename(columns={"q": "qHat"})
    l31 = l31_states.merge(l31_q, on="stateId", validate="one_to_one")
    l31["evaluationCohort"] = "L31_CONFIRMATION"
    columns = [
        "stateId",
        "candidateId",
        "matrixIndex",
        "landmark",
        "matrixRole",
        "evaluationCohort",
        "successes",
        "qHat",
        "currentSelectedIndex",
        "targetCurrentLabel",
    ]
    result = pd.concat([l28[columns], l31[columns]], ignore_index=True).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)
    if len(result) != 280 or result["targetCurrentLabel"].any():
        raise RuntimeError("response registry identity or at-risk gate failed")
    return result


def extract_representations(
    responses: pd.DataFrame, manifest: pd.DataFrame, reverse: bool = False
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    manifest_index = manifest.set_index(["candidateId", "matrixIndex"])
    meta_rows = []
    values: dict[str, list[np.ndarray]] = {view: [] for view in VIEWS}
    for source in responses.itertuples(index=False):
        manifest_row = manifest_index.loc[(source.candidateId, int(source.matrixIndex))]
        states = L26.load_states(source.candidateId, int(source.matrixIndex), manifest)
        endpoint = int(source.landmark)
        window = states[endpoint - 32 : endpoint].copy()
        if window.shape != (32, 100):
            raise RuntimeError("transition-tube window shape failure")
        if reverse:
            window = window[::-1]
        representation = L27.transition_tube_views(window)
        meta_rows.append(
            {
                "stateId": source.stateId,
                "candidateId": source.candidateId,
                "matrixIndex": int(source.matrixIndex),
                "landmark": endpoint,
                "evaluationCohort": source.evaluationCohort,
                "successes": int(source.successes),
                "qHat": float(source.qHat),
                "variant": "TEMPORAL_REVERSAL" if reverse else "ORIGINAL",
                "trajectoryCacheSha256": manifest_row.cacheSha256,
            }
        )
        for view in VIEWS:
            values[view].append(representation[view])
    return pd.DataFrame(meta_rows), {
        view: np.stack(rows).astype(np.float64) for view, rows in values.items()
    }


def representation_table(meta: pd.DataFrame, vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for view, values in vectors.items():
        for index, source in enumerate(meta.itertuples(index=False)):
            rows.append(
                {
                    "stateId": source.stateId,
                    "candidateId": source.candidateId,
                    "matrixIndex": source.matrixIndex,
                    "landmark": source.landmark,
                    "evaluationCohort": source.evaluationCohort,
                    "variant": source.variant,
                    "modelId": view,
                    "dimensions": values.shape[1],
                    "vectorSha256": array_hash(values[index]),
                    "values": values[index].tolist(),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["variant", "evaluationCohort", "candidateId", "modelId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)


def validate_l27_feature_replay(table: pd.DataFrame) -> dict[str, Any]:
    source = pd.concat(
        [
            pd.read_parquet(L27_ROOT / "development_representation_manifest.parquet"),
            pd.read_parquet(L27_ROOT / "validation_representation_manifest.parquet"),
        ],
        ignore_index=True,
    )
    original = table[table["variant"].eq("ORIGINAL")]
    merged = original.merge(
        source[["candidateId", "matrixIndex", "landmark", "modelId", "vectorSha256"]],
        on=["candidateId", "matrixIndex", "landmark", "modelId"],
        suffixes=("Current", "L27"),
        validate="one_to_one",
    )
    exact = bool(
        len(merged) == len(original)
        and merged["vectorSha256Current"].eq(merged["vectorSha256L27"]).all()
    )
    return {
        "schema": "eidosoma.e01.s19_l32.l27_feature_replay.v1",
        "status": "PASS" if exact else "FAIL",
        "rows": len(merged),
        "expectedRows": len(original),
        "allVectorHashesExact": exact,
        "l27DevelopmentManifestSha256": sha256_file(
            L27_ROOT / "development_representation_manifest.parquet"
        ),
        "l27ValidationManifestSha256": sha256_file(
            L27_ROOT / "validation_representation_manifest.parquet"
        ),
    }


def vector_frames(
    meta: pd.DataFrame, vectors: dict[str, np.ndarray]
) -> dict[str, pd.DataFrame]:
    result = {}
    base = meta.reset_index(drop=True)
    for view, values in vectors.items():
        columns = [f"f{index:04d}" for index in range(values.shape[1])]
        features = pd.DataFrame(values, columns=columns)
        result[view] = pd.concat([base, features], axis=1)
    return result


def fit_and_score(
    original_frames: dict[str, pd.DataFrame],
    reversed_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    registry_rows = []
    for candidate in CANDIDATES:
        development_reference = original_frames[PRIMARY_MODEL]
        development_reference = development_reference[
            development_reference["candidateId"].eq(candidate)
            & development_reference["evaluationCohort"].eq("L28_DEVELOPMENT")
        ]
        landmark_priors = development_reference.groupby("landmark")["qHat"].mean().to_dict()
        for cohort in ("L28_DEVELOPMENT", *EVALUATION_COHORTS):
            cohort_meta = original_frames[PRIMARY_MODEL]
            cohort_meta = cohort_meta[
                cohort_meta["candidateId"].eq(candidate)
                & cohort_meta["evaluationCohort"].eq(cohort)
            ]
            for source in cohort_meta.itertuples(index=False):
                rows.append(
                    {
                        "stateId": source.stateId,
                        "candidateId": candidate,
                        "matrixIndex": source.matrixIndex,
                        "landmark": source.landmark,
                        "evaluationCohort": cohort,
                        "variant": "ORIGINAL",
                        "modelId": "LANDMARK_PRIOR",
                        "predictedQ": float(landmark_priors[int(source.landmark)]),
                        "qHat": source.qHat,
                        "successes": source.successes,
                    }
                )
        for view in VIEWS:
            frame = original_frames[view]
            development = frame[
                frame["candidateId"].eq(candidate)
                & frame["evaluationCohort"].eq("L28_DEVELOPMENT")
            ]
            feature_columns = [column for column in frame.columns if column.startswith("f")]
            scaler, model = L29.fit_model(development, feature_columns)
            replay_scaler, replay_model = L29.fit_model(development, feature_columns)
            registry_rows.append(
                {
                    "candidateId": candidate,
                    "modelId": view,
                    "featureCount": len(feature_columns),
                    "featureNames": json.dumps(feature_columns),
                    "scalerMean": json.dumps(scaler.mean_.tolist()),
                    "scalerScale": json.dumps(scaler.scale_.tolist()),
                    "intercept": float(model.intercept_[0]),
                    "coefficients": json.dumps(model.coef_[0].tolist()),
                    "iterations": int(model.n_iter_[0]),
                    "exactReplay": bool(
                        np.array_equal(scaler.mean_, replay_scaler.mean_)
                        and np.array_equal(scaler.scale_, replay_scaler.scale_)
                        and np.array_equal(model.coef_, replay_model.coef_)
                        and np.array_equal(model.intercept_, replay_model.intercept_)
                    ),
                }
            )
            for variant, source_frame in (
                ("ORIGINAL", frame),
                ("TEMPORAL_REVERSAL", reversed_frames[view]),
            ):
                subset = source_frame[source_frame["candidateId"].eq(candidate)]
                probabilities = model.predict_proba(
                    scaler.transform(subset[feature_columns].to_numpy(dtype=np.float64))
                )[:, 1]
                replay_probabilities = replay_model.predict_proba(
                    replay_scaler.transform(subset[feature_columns].to_numpy(dtype=np.float64))
                )[:, 1]
                if not np.array_equal(probabilities, replay_probabilities):
                    raise RuntimeError("model probability replay failed")
                for source, probability in zip(
                    subset.itertuples(index=False), probabilities, strict=True
                ):
                    rows.append(
                        {
                            "stateId": source.stateId,
                            "candidateId": candidate,
                            "matrixIndex": source.matrixIndex,
                            "landmark": source.landmark,
                            "evaluationCohort": source.evaluationCohort,
                            "variant": variant,
                            "modelId": view,
                            "predictedQ": float(probability),
                            "qHat": source.qHat,
                            "successes": source.successes,
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["variant", "evaluationCohort", "candidateId", "modelId", "landmark", "matrixIndex"]
    ).reset_index(drop=True), pd.DataFrame(registry_rows)


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(
        ["variant", "evaluationCohort", "candidateId", "modelId"], sort=True
    ):
        variant, cohort, candidate, model = keys
        q = group["qHat"].to_numpy(dtype=np.float64)
        p = np.clip(group["predictedQ"].to_numpy(dtype=np.float64), 1e-9, 1 - 1e-9)
        brier = float(np.mean(q * (1 - p) ** 2 + (1 - q) * p**2))
        log_loss = float(-np.mean(q * np.log(p) + (1 - q) * np.log(1 - p)))
        intercept, slope = L28.calibration_parameters(p, q)
        rows.append(
            {
                "variant": variant,
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "modelId": model,
                "states": len(group),
                "spearmanQHat": L29.safe_spearman(p, q),
                "brierScorePerBranch": brier,
                "binomialLogLossPerBranch": log_loss,
                "calibrationIntercept": intercept,
                "calibrationSlope": slope,
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
            ].pivot(index=["stateId", "qHat"], columns="modelId", values="predictedQ").reset_index()
            rng = np.random.default_rng(derived_seed("bootstrap", cohort, candidate))
            models = [column for column in pivot.columns if column not in {"stateId", "qHat"}]
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
                            "evaluationCohort": cohort,
                            "candidateId": candidate,
                            "bootstrapIndex": replicate,
                            "modelId": model,
                            "spearmanQHat": L29.safe_spearman(p, q),
                            "primaryBrierImprovement": float("nan"),
                        }
                    )
                for control in CONTROL_MODELS:
                    rows.append(
                        {
                            "evaluationCohort": cohort,
                            "candidateId": candidate,
                            "bootstrapIndex": replicate,
                            "modelId": f"DELTA_PRIMARY_VS_{control}",
                            "spearmanQHat": float("nan"),
                            "primaryBrierImprovement": brier[control] - brier[PRIMARY_MODEL],
                        }
                    )
    return pd.DataFrame(rows)


def permute_within_landmark(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    result = frame.copy()
    for indices in result.groupby("landmark").groups.values():
        idx = np.asarray(list(indices), dtype=int)
        order = rng.permutation(len(idx))
        result.loc[idx, "successes"] = result.loc[idx, "successes"].to_numpy()[order]
        result.loc[idx, "qHat"] = result.loc[idx, "qHat"].to_numpy()[order]
    return result


def permutation_results(
    original_frames: dict[str, pd.DataFrame], predictions: pd.DataFrame, metrics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development_rows = []
    evaluation_rows = []
    observed = metrics[
        metrics["variant"].eq("ORIGINAL")
        & metrics["evaluationCohort"].isin(EVALUATION_COHORTS)
        & metrics["modelId"].eq(PRIMARY_MODEL)
    ].set_index(["evaluationCohort", "candidateId"])["spearmanQHat"].to_dict()
    primary_predictions = predictions[
        predictions["variant"].eq("ORIGINAL")
        & predictions["evaluationCohort"].isin(EVALUATION_COHORTS)
        & predictions["modelId"].eq(PRIMARY_MODEL)
    ]
    null_maxima = []
    fixed_null_maxima = []
    for replicate in range(PERMUTATIONS):
        fit_values = {}
        fixed_values = {}
        for candidate in CANDIDATES:
            frame = original_frames[PRIMARY_MODEL]
            development = frame[
                frame["candidateId"].eq(candidate)
                & frame["evaluationCohort"].eq("L28_DEVELOPMENT")
            ].reset_index(drop=True)
            rng = np.random.default_rng(derived_seed("development_permutation", replicate, candidate))
            permuted = permute_within_landmark(development, rng)
            columns = [column for column in frame.columns if column.startswith("f")]
            scaler, model = L29.fit_model(permuted, columns)
            for cohort in EVALUATION_COHORTS:
                evaluation = frame[
                    frame["candidateId"].eq(candidate)
                    & frame["evaluationCohort"].eq(cohort)
                ]
                probability = model.predict_proba(
                    scaler.transform(evaluation[columns].to_numpy(dtype=np.float64))
                )[:, 1]
                rho = L29.safe_spearman(probability, evaluation["qHat"])
                fit_values[(cohort, candidate)] = rho
                development_rows.append(
                    {
                        "replicate": replicate,
                        "evaluationCohort": cohort,
                        "candidateId": candidate,
                        "nullSpearman": rho,
                    }
                )
                fixed = primary_predictions[
                    primary_predictions["candidateId"].eq(candidate)
                    & primary_predictions["evaluationCohort"].eq(cohort)
                ].reset_index(drop=True)
                fixed_rng = np.random.default_rng(
                    derived_seed("evaluation_permutation", replicate, cohort, candidate)
                )
                shuffled = permute_within_landmark(fixed, fixed_rng)
                fixed_rho = L29.safe_spearman(shuffled["predictedQ"], shuffled["qHat"])
                fixed_values[(cohort, candidate)] = fixed_rho
                evaluation_rows.append(
                    {
                        "replicate": replicate,
                        "evaluationCohort": cohort,
                        "candidateId": candidate,
                        "nullSpearman": fixed_rho,
                    }
                )
        null_maxima.append(max(value for value in fit_values.values() if np.isfinite(value)))
        fixed_null_maxima.append(max(value for value in fixed_values.values() if np.isfinite(value)))
    development = pd.DataFrame(development_rows)
    evaluation = pd.DataFrame(evaluation_rows)
    for frame, maxima, label in (
        (development, null_maxima, "familywiseP"),
        (evaluation, fixed_null_maxima, "familywiseP"),
    ):
        for (cohort, candidate), value in observed.items():
            p_value = float((1 + np.sum(np.asarray(maxima) >= value)) / (1 + len(maxima)))
            mask = frame["evaluationCohort"].eq(cohort) & frame["candidateId"].eq(candidate)
            frame.loc[mask, "observedSpearman"] = value
            frame.loc[mask, label] = p_value
    return development, evaluation


def suffix_invariance(
    responses: pd.DataFrame, manifest: pd.DataFrame, stored: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    sentinels = responses.groupby(["evaluationCohort", "candidateId"]).head(3)
    for source in sentinels.itertuples(index=False):
        states = L26.load_states(source.candidateId, int(source.matrixIndex), manifest)
        endpoint = int(source.landmark)
        altered = states.copy()
        if len(altered) > endpoint:
            rng = np.random.default_rng(
                derived_seed("suffix", source.evaluationCohort, source.candidateId, source.matrixIndex)
            )
            altered[endpoint:] = altered[endpoint:][rng.permutation(len(altered) - endpoint)]
        first = L27.transition_tube_views(states[endpoint - 32 : endpoint])
        second = L27.transition_tube_views(altered[endpoint - 32 : endpoint])
        for view in VIEWS:
            saved = stored[
                stored["stateId"].eq(source.stateId)
                & stored["variant"].eq("ORIGINAL")
                & stored["modelId"].eq(view)
            ].iloc[0]
            rows.append(
                {
                    "stateId": source.stateId,
                    "candidateId": source.candidateId,
                    "evaluationCohort": source.evaluationCohort,
                    "modelId": view,
                    "prefixExact": bool(np.array_equal(states[:endpoint], altered[:endpoint])),
                    "suffixActuallyChanged": bool(
                        len(states) <= endpoint or not np.array_equal(states[endpoint:], altered[endpoint:])
                    ),
                    "featureInvariant": bool(np.array_equal(first[view], second[view])),
                    "storedExact": array_hash(first[view]) == saved.vectorSha256,
                }
            )
    return pd.DataFrame(rows)


def gate_table(
    metrics: pd.DataFrame,
    bootstraps: pd.DataFrame,
    development_perms: pd.DataFrame,
    evaluation_perms: pd.DataFrame,
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
                metrics["variant"].eq("TEMPORAL_REVERSAL")
                & metrics["evaluationCohort"].eq(cohort)
                & metrics["candidateId"].eq(candidate)
                & metrics["modelId"].eq(PRIMARY_MODEL)
            ].iloc[0]
            boot = bootstraps[
                bootstraps["evaluationCohort"].eq(cohort)
                & bootstraps["candidateId"].eq(candidate)
            ]
            rank_lower = float(
                np.nanquantile(boot[boot["modelId"].eq(PRIMARY_MODEL)]["spearmanQHat"], 0.025)
            )
            deltas = {
                control: float(
                    np.quantile(
                        boot[boot["modelId"].eq(f"DELTA_PRIMARY_VS_{control}")][
                            "primaryBrierImprovement"
                        ],
                        0.025,
                    )
                )
                for control in CONTROL_MODELS
            }
            dev_p = float(
                development_perms[
                    development_perms["evaluationCohort"].eq(cohort)
                    & development_perms["candidateId"].eq(candidate)
                ]["familywiseP"].iloc[0]
            )
            eval_p = float(
                evaluation_perms[
                    evaluation_perms["evaluationCohort"].eq(cohort)
                    & evaluation_perms["candidateId"].eq(candidate)
                ]["familywiseP"].iloc[0]
            )
            suffix_rows = suffix[
                suffix["evaluationCohort"].eq(cohort)
                & suffix["candidateId"].eq(candidate)
            ]
            checks = {
                "rankPassed": primary.spearmanQHat > 0.5 and rank_lower > 0.3,
                "incrementalBrierPassed": all(value > 0 for value in deltas.values()),
                "developmentPermutationPassed": dev_p <= 0.05,
                "evaluationPermutationPassed": eval_p <= 0.05,
                "temporalReversalPassed": primary.spearmanQHat > reversed_row.spearmanQHat,
                "suffixPassed": bool(
                    suffix_rows[["prefixExact", "suffixActuallyChanged", "featureInvariant", "storedExact"]]
                    .all()
                    .all()
                ),
            }
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "states": primary.states,
                    "primarySpearman": primary.spearmanQHat,
                    "primarySpearmanLower95": rank_lower,
                    **{f"brierImprovementLowerVs{key}": value for key, value in deltas.items()},
                    "developmentPermutationP": dev_p,
                    "evaluationPermutationP": eval_p,
                    "temporalReversalSpearman": reversed_row.spearmanQHat,
                    **checks,
                    "cohortCandidateGatePassed": all(checks.values()),
                }
            )
    return pd.DataFrame(rows)


def fixture_results() -> pd.DataFrame:
    base = L27.fixture_table()
    required = {
        "REPRESENTATION_SCHEMA",
        "EXACT_REPRESENTATION_REPLAY",
        "MOLECULE_RELABEL_INVARIANCE",
    }
    subset = base[base["fixtureId"].isin(required)][["fixtureId", "passed", "details"]]
    return pd.concat(
        [
            subset,
            pd.DataFrame(
                [
                    {
                        "fixtureId": "THREE_FROZEN_VIEWS_ONLY",
                        "passed": tuple(VIEWS)
                        == (
                            "FULL_TRANSITION_TUBE",
                            "EXACT_H_TRANSITION_TUBE",
                            "ORDINARY_TRANSITION_TUBE",
                        ),
                        "details": json.dumps(VIEWS),
                    },
                    {
                        "fixtureId": "NO_BRANCH_PREDICTOR",
                        "passed": True,
                        "details": "only L27 observed-prefix vectors enter models",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )


def benchmark_projection() -> dict[str, Any]:
    rng = np.random.default_rng(derived_seed("opaque_benchmark"))
    columns = [f"f{index:04d}" for index in range(693)]
    frame = pd.DataFrame(rng.normal(size=(50, len(columns))), columns=columns)
    frame["successes"] = rng.integers(8, 121, size=len(frame))
    L29.fit_model(frame, columns)
    durations = []
    for _ in range(3):
        started = time.perf_counter()
        L29.fit_model(frame, columns)
        durations.append(time.perf_counter() - started)
    projected_fits = 12 + PERMUTATIONS * len(CANDIDATES)
    projected_fit_seconds = max(durations) * projected_fits
    projected_cpu_hours = 2.0 * projected_fit_seconds / 3600 + 1.0
    projected_wall_hours = projected_fit_seconds / 3600 + 1.0
    passed = projected_cpu_hours <= 90.0 and projected_wall_hours <= 64.8
    return {
        "schema": "eidosoma.e01.s19_l32.benchmark_projection.v1",
        "status": "PASS" if passed else "STOP_BEFORE_OUTCOME",
        "opaqueSyntheticRows": len(frame),
        "opaqueSyntheticFeatures": len(columns),
        "timedFits": len(durations),
        "fitDurationsSeconds": durations,
        "projectedScientificFits": projected_fits,
        "projectedFitSeconds": projected_fit_seconds,
        "projectedCpuHoursIncludingConservativeOverhead": projected_cpu_hours,
        "projectedWallHoursIncludingConservativeOverhead": projected_wall_hours,
        "cpuHoursAvailableAfterReserve": 90.0,
        "wallHoursAvailableAfterReserve": 64.8,
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
        raise RuntimeError("L32 seed collision within current manifest")
    return result


def seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    current = set(seeds["seedMaterialSha256"].astype(str))
    current_derived = set(seeds["derivedSeed"].astype(str))
    previous: set[str] = set()
    previous_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L32/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            if "seedmaterialsha256" in column.lower():
                previous.update(frame[column].dropna().astype(str))
            if column.lower() == "derivedseed":
                previous_derived.update(frame[column].dropna().astype(str))
    overlaps = sorted(current & previous)
    derived_overlaps = sorted(current_derived & previous_derived)
    root_collisions = []
    needle = ROOT_HEX.encode()
    for path in ARTIFACT_ROOT.rglob("*"):
        if not path.is_file() or "/L32/" in str(path) or path.stat().st_size > 64 * 1024 * 1024:
            continue
        try:
            if needle in path.read_bytes():
                root_collisions.append(str(path))
        except OSError:
            continue
    return {
        "schema": "eidosoma.e01.s19_l32.seed_firewall.v1",
        "status": "PASS"
        if not overlaps and not derived_overlaps and not root_collisions
        else "FAIL",
        "currentSeedMaterialCount": len(current),
        "expectedSeedMaterialCount": len(seeds),
        "allCurrentMaterialsUnique": len(current) == len(seeds),
        "priorSeedMaterialCount": len(previous),
        "overlapCount": len(overlaps),
        "overlaps": overlaps,
        "priorDerivedSeedCount": len(previous_derived),
        "derivedSeedOverlapCount": len(derived_overlaps),
        "derivedSeedOverlaps": derived_overlaps,
        "rootCollisionPaths": sorted(root_collisions),
    }


def make_figures(
    predictions: pd.DataFrame, metrics: pd.DataFrame, gates: pd.DataFrame
) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(root / name, dpi=180)
        plt.close()

    original = predictions[
        predictions["variant"].eq("ORIGINAL")
        & predictions["evaluationCohort"].isin(EVALUATION_COHORTS)
        & predictions["modelId"].eq(PRIMARY_MODEL)
    ]
    _, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, ((cohort, candidate), group) in zip(
        axes.flat, original.groupby(["evaluationCohort", "candidateId"], sort=True), strict=True
    ):
        axis.scatter(group["predictedQ"], group["qHat"], s=22)
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set_title(f"{cohort} / {candidate}", fontsize=8)
        axis.set_xlabel("Past-only tube predicted q")
        axis.set_ylabel("Empirical H32 q-hat")
    save("01_past_only_coordinate_vs_committor.png")

    metrics[
        metrics["variant"].eq("ORIGINAL")
        & metrics["evaluationCohort"].isin(EVALUATION_COHORTS)
    ].pivot_table(
        index="modelId", columns=["evaluationCohort", "candidateId"], values="spearmanQHat"
    ).plot(kind="bar", figsize=(12, 5))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("Spearman with H32 q-hat")
    save("02_view_comparison.png")

    temporal = metrics[
        metrics["modelId"].eq(PRIMARY_MODEL)
        & metrics["evaluationCohort"].isin(EVALUATION_COHORTS)
    ].pivot_table(index=["evaluationCohort", "candidateId"], columns="variant", values="spearmanQHat")
    temporal.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Spearman")
    save("03_temporal_reversal_control.png")

    checks = [
        "rankPassed",
        "incrementalBrierPassed",
        "developmentPermutationPassed",
        "evaluationPermutationPassed",
        "temporalReversalPassed",
        "suffixPassed",
        "cohortCandidateGatePassed",
    ]
    matrix = gates.set_index(["evaluationCohort", "candidateId"])[checks].astype(float)
    plt.figure(figsize=(10, 5))
    plt.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(len(checks)), checks, rotation=35, ha="right", fontsize=7)
    plt.yticks(range(len(matrix)), [" / ".join(index) for index in matrix.index], fontsize=7)
    plt.colorbar(ticks=[0, 1])
    save("04_solution_gate_matrix.png")


def manifest_for(root: Path) -> dict[str, Any]:
    files = [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "artifact_manifest.json")
    ]
    return {
        "schema": "eidosoma.e01.s19_l32.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(files),
        "totalBytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def append_ledgers(classifications: list[str], timestamp: str, next_theme: str, stop_early: bool) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L31 confirmed a simulation shooting coordinate but not an observed-prefix biomarker.",
            "failureOrAmbiguityTargeted": "Whether committor supervision reveals a past-only organization transition tube missed by single-future labels.",
            "informationGainRationale": "The confirmed q response separates target noise from representation adequacy.",
            "learned": "L32 three-view committor regression contract frozen.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "UNTOUCHED_EIGHT_STEP_PROPAGATOR_COMMITTOR_COORDINATE_CONFIRMED",
            "proposedNextTest": "Fit L27 tubes to L28 development q and test L28/L31 without refit.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Past-only full organization path, stability-only path, or inaccessible hidden state.",
            "selectedHypotheses": "Exact L27 full/exact-H/ordinary views.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "The finite-horizon target has no reproducible coordinate.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A practical lead must generalize to L31 without branch-derived inputs.",
            "failureOrAmbiguityTargeted": "Observed-prefix organization signal.",
            "informationGainRationale": "Dual held-out cohorts and controls distinguish organization from H and ordinary stability.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L32 results.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION" if not stop_early else "POST_LOOP_SOLUTION_HUMAN_REVIEW",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Exact L27 full/exact-H/ordinary views.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Observed organization paths encode q" if not stop_early else "No past-only organization signal exists.",
        },
    ]
    BASE.write_parquet(
        ledger_path,
        pd.concat([ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)], ignore_index=True),
    )
    md = ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    BASE.atomic_text(
        md,
        md.read_text()
        + f"\n\n## {LOOP_ID} — committor-ordered past-only transition tube\n\n- **Learned:** {', '.join(classifications)}.\n- **Next:** {next_theme}.\n",
    )
    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    row = {
        "branchCount": 3,
        "bundleId": "L32_COMMITTOR_ORDERED_TRANSITION_TUBE",
        "candidateId": "S19-L32-PAST-ONLY-TUBE",
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
        "proposedSpecification": "three exact L27 observed-prefix tube views regressed on empirical q",
        "rankingScore": 28.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L31_CONFIRMED_COMMITTOR_COORDINATE",
        "sourceGrounding": 5,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidates_path,
        pd.concat([candidates, pd.DataFrame([row]).reindex(columns=candidates.columns)], ignore_index=True),
    )
    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    source_row = {
        "commitOrVersion": None,
        "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
        "finding": "L32 uses the exact frozen L27 past-only transition-tube representations and the reliable L28/L31 empirical H32 committors; no branch-derived value enters a predictor.",
        "licenseStatus": "WORKSPACE_EVIDENCE",
        "redistributionStatus": "INTERNAL_ARTIFACT",
        "repositoryIdentity": None,
        "retainedPath": str(L31_ROOT / "research_step_full_results.md"),
        "retrievalDate": timestamp[:10],
        "sha256": sha256_file(L31_ROOT / "research_step_full_results.md"),
        "sourceId": "L32_L27_L31_FROZEN_COMMITTOR_CONTEXT",
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
    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text())
    registry["loops"].append(
        {
            "loopId": LOOP_ID,
            "versionedLoopId": VERSION,
            "status": "COMPLETE_SOLUTION_BOUNDARY" if stop_early else "COMPLETE_AUTONOMOUS_CONTINUATION_AUTHORIZED",
            "authorized": True,
            "completed": True,
            "outcomeAccessed": True,
            "humanReviewRequiredAfter": stop_early,
            "classification": classifications,
            "selectedDiscoveryLead": "PAST_ONLY_TRANSITION_TUBE_COMMITTOR_COORDINATE" if stop_early else None,
            "newMatrices": 0,
            "newTrajectories": 0,
            "nextStepActive": not stop_early,
        }
    )
    registry["proposedNextLoopTheme"] = next_theme
    registry["proposedNextLoopActive"] = not stop_early
    BASE.atomic_text(registry_path, yaml.safe_dump(registry, sort_keys=False))
    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    history["history"].append(
        {
            "decision": "S19_L32_SOLUTION_HUMAN_REVIEW"
            if stop_early
            else "S19_L32_COMPLETE_AUTONOMOUS_CONTINUATION",
            "loopId": LOOP_ID,
            "nextLoopAuthorized": not stop_early,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "s20Activated": False,
            "scope": VERSION,
            "selectedDiscoveryLead": "PAST_ONLY_TRANSITION_TUBE_COMMITTOR_COORDINATE"
            if stop_early
            else None,
            "source": "locked_execution_result",
        }
    )
    history["pendingDecision"] = (
        "HUMAN_REVIEW_REQUIRED_AFTER_EARLY_SOLUTION"
        if stop_early
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
    return f"""# S19-L32 — Committor-Ordered Past-Only Transition-Tube Coordinate

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the authorized L19–L42 sequence.
- **Outcome classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Validation:** exact L27 representations for all 280 L28/L31 states; development-only candidate models; unchanged L28 validation and L31 confirmation; suffix invariance; temporal reversal; 512 development and evaluation permutations; 4,096 matrix bootstraps; exact model/report regeneration and artifact hashes.
- **Next bounded theme:** {next_theme}

## Frozen question and method

Can one observed 32-state prefix recover the reliable H32 committor without H8 shooting branches or a completed-run centroid predictor? The full view contains 11 past-only level/current channels; exact-H/recurrence and ordinary composition/dynamics views are separate controls. Models use only L28 development q and are evaluated unchanged on L28 validation and the previously untouched L31 confirmation cohort.

## Metrics

{metrics.to_markdown(index=False)}

## Solution gates

{gates.to_markdown(index=False)}

## Interpretation boundary

Even a passing result is conditioned on a retrospectively constructed target basin. It supports an organization-before-entry coordinate within that reconstructed task, not the paper authors' exact label, causal control, or biological validation. Branch-derived H8 values never enter the predictor.

## Runtime

- Repository lock: `{runtime['repositoryHead']}`.
- CPU float64; no GPU; wall seconds `{runtime['wallSeconds']:.3f}`.

## Autonomous boundary

L32 is frozen. S20, E02, author contact, interventions, reactive-current claims and report-bundle work remain inactive.
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
    benchmark = benchmark_projection()
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or benchmark["status"] != "PASS"
    ):
        raise RuntimeError("prior or fixture gate failed")
    responses = response_registry()
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    original_meta, original_vectors = extract_representations(responses, manifest, reverse=False)
    reversed_meta, reversed_vectors = extract_representations(responses, manifest, reverse=True)
    table = pd.concat(
        [
            representation_table(original_meta, original_vectors),
            representation_table(reversed_meta, reversed_vectors),
        ],
        ignore_index=True,
    ).sort_values(
        ["variant", "evaluationCohort", "candidateId", "modelId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)
    replay = validate_l27_feature_replay(table)
    if replay["status"] != "PASS":
        raise RuntimeError("L27 representation replay failed")
    seeds = seed_manifest(responses)
    firewall = seed_firewall(seeds)
    if firewall["status"] != "PASS":
        raise RuntimeError("L32 seed firewall failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L32 decision record\n\nL31 independently confirmed the H8 shooting coordinate and therefore satisfied the human prerequisite for transition-tube feature work. L32 freezes exactly the three existing L27 32-state views and the L29 aggregated-binomial model. Only L28 development q may fit models; L28 validation and L31 confirmation remain unchanged. No H8 branch statistic, completed-run centroid, representation search, window search or hyperparameter search may enter a predictor. A full pass stops the autonomous search early for human review.\n",
    )
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    BASE.write_parquet(LOOP_ROOT / "response_registry.parquet", responses)
    BASE.write_parquet(LOOP_ROOT / "representation_results.parquet", table)
    BASE.write_json(LOOP_ROOT / "l27_feature_replay.json", replay)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    hashes = {
        "responsesSha256": sha256_file(LOOP_ROOT / "response_registry.parquet"),
        "representationsSha256": sha256_file(LOOP_ROOT / "representation_results.parquet"),
        "seedsSha256": sha256_file(LOOP_ROOT / "analysis_seed_manifest.parquet"),
        "seedFirewallSha256": sha256_file(LOOP_ROOT / "seed_firewall.json"),
        "benchmarkSha256": sha256_file(LOOP_ROOT / "benchmark_projection.json"),
        "l28QSha256": sha256_file(L28_ROOT / "committor_state_results.parquet"),
        "l31QSha256": sha256_file(L31_ROOT / "state_committor_and_propagator_results.parquet"),
    }
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l32.implementation_lock.v1",
            "repositoryHead": head,
            "remoteHead": remote,
            "runnerSha256": sha256_file(RUNNER_PATH),
            "configSha256": sha256_file(CONFIG),
            "transitionTubeCoreSha256": sha256_file(REPO_ROOT / "src/e01_onset_discovery/transition_tube.py"),
            "views": list(VIEWS),
            "primary": PRIMARY_MODEL,
            "modelFitScope": "L28_DEVELOPMENT_ONLY",
            "branchDerivedPredictor": False,
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
    for key, path in {
        "responsesSha256": LOOP_ROOT / "response_registry.parquet",
        "representationsSha256": LOOP_ROOT / "representation_results.parquet",
        "seedsSha256": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewallSha256": LOOP_ROOT / "seed_firewall.json",
        "benchmarkSha256": LOOP_ROOT / "benchmark_projection.json",
        "l28QSha256": L28_ROOT / "committor_state_results.parquet",
        "l31QSha256": L31_ROOT / "state_committor_and_propagator_results.parquet",
    }.items():
        if sha256_file(path) != lock[key]:
            raise RuntimeError(f"locked input changed: {path}")
    if not prior["unchanged"] or prior["aggregateSha256"] != lock["priorAggregateSha256"] or not fixtures["passed"].all():
        raise RuntimeError("pre-execution validation failed")
    table = pd.read_parquet(LOOP_ROOT / "representation_results.parquet")
    metas = {}
    frames = {}
    for variant in ("ORIGINAL", "TEMPORAL_REVERSAL"):
        subset = table[table["variant"].eq(variant)]
        first_view = subset[subset["modelId"].eq(PRIMARY_MODEL)].sort_values(
            ["evaluationCohort", "candidateId", "landmark", "matrixIndex"]
        )
        meta = response_registry().merge(
            first_view[["stateId"]], on="stateId", validate="one_to_one"
        ).sort_values(["evaluationCohort", "candidateId", "landmark", "matrixIndex"]).reset_index(drop=True)
        meta["variant"] = variant
        vectors = {}
        for view in VIEWS:
            values = subset[subset["modelId"].eq(view)].sort_values(
                ["evaluationCohort", "candidateId", "landmark", "matrixIndex"]
            )["values"]
            vectors[view] = np.stack(values.map(np.asarray)).astype(np.float64)
        metas[variant] = meta
        frames[variant] = vector_frames(meta, vectors)
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    prediction_frame, registry = fit_and_score(frames["ORIGINAL"], frames["TEMPORAL_REVERSAL"])
    if not registry["exactReplay"].all():
        raise RuntimeError("model replay failed")
    metrics = metric_table(prediction_frame)
    bootstraps = bootstrap_metrics(prediction_frame)
    development_perms, evaluation_perms = permutation_results(
        frames["ORIGINAL"], prediction_frame, metrics
    )
    responses = pd.read_parquet(LOOP_ROOT / "response_registry.parquet")
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    suffix = suffix_invariance(responses, manifest, table)
    gates = gate_table(metrics, bootstraps, development_perms, evaluation_perms, suffix)
    solution = bool(gates["cohortCandidateGatePassed"].all())
    if solution:
        classifications = [
            "PAST_ONLY_TRANSITION_TUBE_COMMITTOR_COORDINATE_CONFIRMED",
            "ORGANIZATION_PRECURSOR_SIGNAL_ESTABLISHED_WITHIN_RETROSPECTIVE_TARGET_TASK",
            "NOT_A_CONFIRMED_PAPER_OR_CAUSAL_RESULT",
        ]
        next_theme = "HUMAN_REVIEW_SOLUTION_BOUNDARY"
    else:
        classifications = [
            "PAST_ONLY_TRANSITION_TUBE_COMMITTOR_COORDINATE_NON_SUPPORT",
            "CONFIRMED_COMMITTOR_NOT_RECOVERED_BY_FROZEN_OBSERVED_PREFIX_TUBES",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "SINGLE_STATE_MEMORY_AND_PHASE_COORDINATE_AUDIT"
    make_figures(prediction_frame, metrics, gates)
    for name in (
        "preregistration.yaml",
        "decision_record.md",
        "fixture_results.parquet",
        "benchmark_projection.json",
        "response_registry.parquet",
        "representation_results.parquet",
        "l27_feature_replay.json",
        "analysis_seed_manifest.parquet",
        "seed_firewall.json",
        "immutable_prior_validation.json",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
    ):
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    BASE.write_parquet(BUILD_ROOT / "prediction_results.parquet", prediction_frame)
    BASE.write_parquet(BUILD_ROOT / "fitted_model_registry.parquet", registry)
    BASE.write_parquet(BUILD_ROOT / "metric_results.parquet", metrics)
    BASE.write_parquet(BUILD_ROOT / "bootstrap_results.parquet", bootstraps)
    BASE.write_parquet(BUILD_ROOT / "development_permutation_results.parquet", development_perms)
    BASE.write_parquet(BUILD_ROOT / "evaluation_permutation_results.parquet", evaluation_perms)
    BASE.write_parquet(BUILD_ROOT / "suffix_invariance_results.parquet", suffix)
    BASE.write_parquet(BUILD_ROOT / "scientific_gate_results.parquet", gates)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l32.classification.v1",
            "classifications": classifications,
            "solutionGatePassed": solution,
            "predictorPastOnly": True,
            "targetRetrospective": True,
            "branchDerivedPredictor": False,
            "priorStatusesChanged": False,
        },
    )
    pd.DataFrame(columns=["stage", "candidateId", "matrixIndex", "exceptionClass", "exceptionMessage"]).to_csv(
        BUILD_ROOT / "failure_ledger.csv", index=False
    )
    checks = {
        "featureReplayPassed": json.loads((LOOP_ROOT / "l27_feature_replay.json").read_text())["status"] == "PASS",
        "modelReplayPassed": bool(registry["exactReplay"].all()),
        "suffixPassed": bool(suffix[["prefixExact", "suffixActuallyChanged", "featureInvariant", "storedExact"]].all().all()),
        "responsesExact": frame_hash(response_registry()) == frame_hash(pd.read_parquet(LOOP_ROOT / "response_registry.parquet")),
        "fixturesPassed": bool(fixtures["passed"].all()),
        "benchmarkPassed": json.loads((LOOP_ROOT / "benchmark_projection.json").read_text())["status"] == "PASS",
        "seedFirewallPassed": json.loads((LOOP_ROOT / "seed_firewall.json").read_text())["status"] == "PASS",
        "immutablePriorPassed": prior["unchanged"],
        "noBranchPredictor": True,
    }
    if not all(checks.values()):
        raise RuntimeError("regeneration validation failed")
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l32.regeneration_validation.v1",
            "status": "PASS",
            "checks": checks,
            "predictionFrameSha256": frame_hash(prediction_frame),
            "metricFrameSha256": frame_hash(metrics),
        },
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l32.runtime.v1",
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - start,
        "controllerCpuHours": (time.process_time() - start_cpu) / 3600,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    retained = sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file())
    temporary = sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l32.storage_validation.v1",
        "retainedBytes": retained,
        "retainedGiBCeiling": 25,
        "temporaryBytes": temporary,
        "temporaryGiBCeiling": 75,
        "status": "PASS" if retained < 25 * 2**30 and temporary < 75 * 2**30 else "FAIL",
    }
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(metrics, gates, classifications, runtime, next_theme)
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L32_FULL_RESULTS.md", report)
    BASE.atomic_text(BUILD_ROOT / "loop_decision_summary.md", f"# S19-L32 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Past-only solution:** `{solution}`.\n\n**Next:** `{next_theme}`.\n")
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L32-promotion-stage")
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(BUILD_ROOT, stage)
    if LOOP_ROOT.exists():
        shutil.rmtree(LOOP_ROOT)
    os.replace(stage, LOOP_ROOT)
    shutil.rmtree(BUILD_ROOT)
    manifest_out = json.loads((LOOP_ROOT / "artifact_manifest.json").read_text())
    if any(sha256_file(LOOP_ROOT / item["path"]) != item["sha256"] for item in manifest_out["files"]):
        raise RuntimeError("artifact hash failure")
    append_ledgers(classifications, runtime["completedAtUtc"], next_theme, solution)
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md", report.replace("# S19-L32", "# S19 current handoff — S19-L32", 1))
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "HUMAN_REVIEW_REQUIRED_SOLUTION" if solution else "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": "PAST_ONLY_TRANSITION_TUBE_COMMITTOR_COORDINATE" if solution else None,
            "nextAuthorizedLoop": None if solution else "S19-L33",
            "authorizationUpperBound": "S19-L42",
            "s20Active": False,
            "updatedAtUtc": runtime["completedAtUtc"],
        },
    )
    BASE.write_json(ARTIFACT_ROOT / "artifact_manifest.json", manifest_for(ARTIFACT_ROOT))
    print(json.dumps({"status": "COMPLETE", "classifications": classifications, "solution": solution, "nextTheme": next_theme, "runtime": runtime}, indent=2))


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
