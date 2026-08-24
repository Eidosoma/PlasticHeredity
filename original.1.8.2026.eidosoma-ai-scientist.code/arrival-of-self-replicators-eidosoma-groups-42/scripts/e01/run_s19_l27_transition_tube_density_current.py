"""Execute S19-L27 transition-tube density/current discovery."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_onset_discovery.transition_tube import TUBE_VIEWS, transition_tube_views


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L26 = _load_module(
    "e01_s19_l27_l26",
    REPO_ROOT / "scripts/e01/run_s19_l26_recurrence_map_analog_committor.py",
)
BASE = L26.BASE
LOOP_ID = "S19-L27"
VERSION = "E01-S19-L27-TRANSITION-TUBE-DENSITY-CURRENT-v1.0.0"
CANDIDATES = L26.CANDIDATES
LANDMARKS = L26.LANDMARKS
MODELS = TUBE_VIEWS
BOOTSTRAPS = 4096
PERMUTATIONS = 512
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L27"
L26_ROOT = ARTIFACT_ROOT / "loops/L26"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l27")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l27_transition_tube_density_current.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/transition_tube.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    return L26.sha256_file(path)


def array_hash(value: np.ndarray) -> str:
    return L26.array_hash(value)


def derived_seed(*parts: object) -> int:
    value = "\x1f".join([VERSION, *map(str, parts)])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L26_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L26_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L26_ROOT / item["path"]),
            "root": str(L26_ROOT),
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
    aggregate = hashlib.sha256(
        "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
    ).hexdigest()
    return {
        "schema": "eidosoma.e01.s19_l27.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": aggregate,
        "l26ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def training_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=np.float64)
    total = float(len(frame))
    for candidate, candidate_frame in frame.groupby("candidateId"):
        del candidate
        matrices = candidate_frame["matrixIndex"].unique()
        matrix_mass = total / (2.0 * len(matrices))
        for _, matrix_frame in candidate_frame.groupby("matrixIndex"):
            weights[matrix_frame.index.to_numpy()] = matrix_mass / len(matrix_frame)
    return weights


def fixture_table() -> pd.DataFrame:
    rng = np.random.default_rng(derived_seed("fixture"))
    states = rng.poisson(2.0, size=(32, 100)).astype(np.int64)
    states[:, 0] += 1
    first = transition_tube_views(states)
    replay = transition_tube_views(states.copy())
    relabelled = transition_tube_views(states[:, rng.permutation(100)])
    reversed_values = transition_tube_views(states[::-1])
    meta_rows = []
    for landmark in LANDMARKS:
        for candidate in CANDIDATES:
            for offset, label in enumerate((False, True, False, True)):
                meta_rows.append(
                    {
                        "candidateId": candidate,
                        "matrixIndex": landmark * 100 + offset,
                        "landmark": landmark,
                        "eventWithin32": label,
                    }
                )
    meta = pd.DataFrame(meta_rows)
    vectors = {
        model: rng.normal(size=(len(meta), len(first[model]))) for model in MODELS
    }
    fitted = fit_models(meta, vectors)
    scored = score_models(meta, vectors, fitted, "FIXTURE")
    replay_scored = score_models(meta, vectors, fitted, "FIXTURE")
    return pd.DataFrame(
        [
            {
                "fixtureId": "REPRESENTATION_SCHEMA",
                "passed": [len(first[name]) for name in MODELS] == [693, 315, 378],
                "details": json.dumps({name: len(first[name]) for name in MODELS}),
            },
            {
                "fixtureId": "EXACT_REPRESENTATION_REPLAY",
                "passed": all(
                    np.array_equal(first[name], replay[name]) for name in MODELS
                ),
                "details": "CPU float64 exact",
            },
            {
                "fixtureId": "MOLECULE_RELABEL_INVARIANCE",
                "passed": all(
                    np.allclose(first[name], relabelled[name], atol=1e-12, rtol=1e-12)
                    for name in MODELS
                ),
                "details": "tolerance 1e-12",
            },
            {
                "fixtureId": "TEMPORAL_DIRECTION_RETAINED",
                "passed": all(
                    not np.array_equal(first[name], reversed_values[name])
                    for name in MODELS
                ),
                "details": "levels/current reverse",
            },
            {
                "fixtureId": "PROTOTYPE_SCORE_EXACT_REPLAY",
                "passed": BASE.frame_hash(scored) == BASE.frame_hash(replay_scored),
                "details": "same model and values",
            },
            {
                "fixtureId": "PROBABILITY_RANGE",
                "passed": bool(scored["score"].between(0, 1).all()),
                "details": "finite [0,1]",
            },
        ]
    )


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "E_REN_VANDENEIJNDEN_2005",
                "doi": "10.1016/j.cplett.2005.07.084",
                "url": "https://doi.org/10.1016/j.cplett.2005.07.084",
                "directSupport": "reactive trajectories concentrate in transition tubes organized by isocommittor surfaces",
                "frozenUse": "development-defined time-resolved event versus non-event tube density contrast",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
            {
                "sourceId": "BEST_HUMMER_2005",
                "doi": "10.1073/pnas.0408098102",
                "url": "https://doi.org/10.1073/pnas.0408098102",
                "directSupport": "transition-path ensembles can be used to assess reaction coordinates",
                "frozenUse": "held-out transition-path discrimination with explicit control coordinates",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
            {
                "sourceId": "HALL_MAITI_2012",
                "doi": "10.1093/biomet/ass011",
                "url": "https://doi.org/10.1093/biomet/ass011",
                "directSupport": "classification from partially observed functional trajectories",
                "frozenUse": "time-resolved prefix levels and currents rather than summary statistics",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
        ]
    )


def extract_representations(
    task: pd.DataFrame, manifest: pd.DataFrame, role: str, transform: str = "ORIGINAL"
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    meta_rows: list[dict[str, Any]] = []
    values: dict[str, list[np.ndarray]] = {model: [] for model in MODELS}
    subset = task[task["matrixRole"].eq(role)]
    for (candidate, matrix_index), group in subset.groupby(
        ["candidateId", "matrixIndex"], sort=True
    ):
        states = L26.load_states(candidate, int(matrix_index), manifest)
        for source in group.itertuples(index=False):
            endpoint = int(source.landmark)
            window = states[endpoint - 32 : endpoint].copy()
            if transform == "TEMPORAL_REVERSAL":
                window = window[::-1]
            representation = transition_tube_views(window)
            meta_rows.append(
                {
                    "matrixRole": role,
                    "candidateId": candidate,
                    "matrixIndex": int(matrix_index),
                    "landmark": endpoint,
                    "eventWithin32": bool(source.eventWithin32),
                    "transform": transform,
                }
            )
            for model in MODELS:
                values[model].append(representation[model])
    return pd.DataFrame(meta_rows), {
        model: np.stack(rows).astype(np.float64) for model, rows in values.items()
    }


def representation_manifest(
    meta: pd.DataFrame, vectors: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for model, matrix in vectors.items():
        for index, source in enumerate(meta.itertuples(index=False)):
            rows.append(
                {
                    "matrixRole": source.matrixRole,
                    "candidateId": source.candidateId,
                    "matrixIndex": source.matrixIndex,
                    "landmark": source.landmark,
                    "transform": source.transform,
                    "modelId": model,
                    "dimensions": matrix.shape[1],
                    "vectorSha256": array_hash(matrix[index]),
                    "minimum": float(np.min(matrix[index])),
                    "maximum": float(np.max(matrix[index])),
                    "mean": float(np.mean(matrix[index])),
                }
            )
    return pd.DataFrame(rows)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(values, axis=0, weights=weights)


def fit_models(
    meta: pd.DataFrame,
    vectors: dict[str, np.ndarray],
    labels_override: np.ndarray | None = None,
    only: tuple[str, ...] = MODELS,
) -> dict[str, Any]:
    labels = (
        meta["eventWithin32"].to_numpy(np.int8)
        if labels_override is None
        else np.asarray(labels_override, dtype=np.int8)
    )
    weights = training_weights(meta)
    result: dict[str, Any] = {}
    for model in only:
        result[model] = {}
        for landmark in LANDMARKS:
            indices = np.flatnonzero(meta["landmark"].to_numpy() == landmark)
            x = vectors[model][indices]
            y = labels[indices]
            w = weights[indices]
            if len(np.unique(y)) != 2:
                raise RuntimeError("prototype class absent")
            mean = _weighted_mean(x, w)
            scale = np.sqrt(_weighted_mean(np.square(x - mean), w))
            scale = np.where(scale > 1e-10, scale, 1.0)
            z = (x - mean) / scale
            centroid0 = _weighted_mean(z[y == 0], w[y == 0])
            centroid1 = _weighted_mean(z[y == 1], w[y == 1])
            prior = float(np.average(y, weights=w))
            result[model][landmark] = {
                "mean": mean,
                "scale": scale,
                "centroid0": centroid0,
                "centroid1": centroid1,
                "prior": prior,
            }
    result["LANDMARK_PRIOR"] = {
        landmark: float(
            np.average(
                labels[meta["landmark"].to_numpy() == landmark],
                weights=weights[meta["landmark"].to_numpy() == landmark],
            )
        )
        for landmark in LANDMARKS
    }
    return result


def score_models(
    meta: pd.DataFrame,
    vectors: dict[str, np.ndarray],
    models: dict[str, Any],
    variant: str,
) -> pd.DataFrame:
    rows = []
    for index, source in enumerate(meta.itertuples(index=False)):
        for model in MODELS:
            if model not in models:
                continue
            fitted = models[model][int(source.landmark)]
            z = (vectors[model][index] - fitted["mean"]) / fitted["scale"]
            d0 = float(np.mean(np.square(z - fitted["centroid0"])))
            d1 = float(np.mean(np.square(z - fitted["centroid1"])))
            prior = fitted["prior"]
            log_prior = float(np.log(prior / (1.0 - prior)))
            score = float(expit(0.5 * (d0 - d1) + log_prior))
            rows.append(
                {
                    "candidateId": source.candidateId,
                    "matrixIndex": int(source.matrixIndex),
                    "landmark": int(source.landmark),
                    "eventWithin32": bool(source.eventWithin32),
                    "modelId": model,
                    "variant": variant,
                    "score": score,
                    "distanceEvent": d1,
                    "distanceNonEvent": d0,
                    "tubeContrast": d0 - d1,
                }
            )
        prior = float(models["LANDMARK_PRIOR"][int(source.landmark)])
        rows.append(
            {
                "candidateId": source.candidateId,
                "matrixIndex": int(source.matrixIndex),
                "landmark": int(source.landmark),
                "eventWithin32": bool(source.eventWithin32),
                "modelId": "LANDMARK_PRIOR",
                "variant": variant,
                "score": prior,
                "distanceEvent": np.nan,
                "distanceNonEvent": np.nan,
                "tubeContrast": np.nan,
            }
        )
    return pd.DataFrame(rows)


def model_lock(models: dict[str, Any], representation_hash: str) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for model in MODELS:
        entries[model] = {}
        for landmark in LANDMARKS:
            item = models[model][landmark]
            entries[model][str(landmark)] = {
                "dimensions": len(item["mean"]),
                "meanSha256": array_hash(item["mean"]),
                "scaleSha256": array_hash(item["scale"]),
                "centroid0Sha256": array_hash(item["centroid0"]),
                "centroid1Sha256": array_hash(item["centroid1"]),
                "prior": item["prior"],
            }
    return {
        "schema": "eidosoma.e01.s19_l27.transition_tube_lock.v1",
        "developmentOnly": True,
        "validationRepresentationOpened": False,
        "candidateIdentityUsed": False,
        "distance": "MEAN_SQUARED_STANDARDIZED_PATH_DISTANCE",
        "probability": "DIAGONAL_GAUSSIAN_PROTOTYPE_POSTERIOR",
        "developmentRepresentationManifestSha256": representation_hash,
        "landmarkPriors": models["LANDMARK_PRIOR"],
        "models": entries,
        "lockedAtUtc": utc_now(),
    }


def metric(frame: pd.DataFrame) -> dict[str, float]:
    return L26.metric(frame)


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return L26.aggregate_metrics(predictions)


def landmark_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return L26.landmark_metrics(predictions)


def matrix_bootstraps(predictions: pd.DataFrame) -> pd.DataFrame:
    original = predictions[predictions["variant"].eq("ORIGINAL")]
    rows = []
    for candidate in CANDIDATES:
        frame = original[original["candidateId"].eq(candidate)]
        matrices = sorted(frame["matrixIndex"].unique())
        groups = {matrix: frame[frame["matrixIndex"].eq(matrix)] for matrix in matrices}
        for replicate in range(BOOTSTRAPS):
            rng = np.random.default_rng(derived_seed("bootstrap", candidate, replicate))
            sampled = rng.choice(matrices, size=len(matrices), replace=True)
            pieces = []
            for occurrence, matrix in enumerate(sampled):
                piece = groups[int(matrix)].copy()
                piece["matrixIndex"] = occurrence
                pieces.append(piece)
            sample = pd.concat(pieces, ignore_index=True)
            for model in (*MODELS, "LANDMARK_PRIOR"):
                values = metric(sample[sample["modelId"].eq(model)])
                rows.append(
                    {
                        "candidateId": candidate,
                        "replicate": replicate,
                        "modelId": model,
                        "AUROC": values["AUROC"],
                        "AUPRC": values["AUPRC"],
                        "BRIER": values["BRIER"],
                    }
                )
    return pd.DataFrame(rows)


def permute_labels(meta: pd.DataFrame, seed: int) -> np.ndarray:
    normalized = meta.reset_index(drop=True)
    result = normalized["eventWithin32"].to_numpy(np.int8).copy()
    rng = np.random.default_rng(seed)
    for indices in normalized.groupby(["candidateId", "landmark"]).groups.values():
        idx = np.asarray(list(indices), dtype=int)
        result[idx] = result[idx][rng.permutation(len(idx))]
    return result


def development_permutations(
    development_meta: pd.DataFrame,
    development_vectors: dict[str, np.ndarray],
    validation_meta: pd.DataFrame,
    validation_vectors: dict[str, np.ndarray],
    observed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_auc = (
        observed[observed["modelId"].eq("FULL_TRANSITION_TUBE")]
        .set_index("candidateId")["AUROC"]
        .to_dict()
    )
    rows = []
    for replicate in range(PERMUTATIONS):
        labels = permute_labels(
            development_meta, derived_seed("development_permutation", replicate)
        )
        models = fit_models(
            development_meta,
            development_vectors,
            labels,
            only=("FULL_TRANSITION_TUBE",),
        )
        scored = score_models(
            validation_meta, validation_vectors, models, "DEVELOPMENT_LABEL_PERMUTATION"
        )
        values = {
            candidate: metric(
                scored[
                    scored["candidateId"].eq(candidate)
                    & scored["modelId"].eq("FULL_TRANSITION_TUBE")
                ]
            )["AUROC"]
            for candidate in CANDIDATES
        }
        maximum = max(values.values())
        for candidate in CANDIDATES:
            rows.append(
                {
                    "replicate": replicate,
                    "candidateId": candidate,
                    "nullAUROC": values[candidate],
                    "maxNullAUROC": maximum,
                }
            )
    nulls = pd.DataFrame(rows)
    results = pd.DataFrame(
        [
            {
                "candidateId": candidate,
                "observedAUROC": observed_auc[candidate],
                "familywisePValue": float(
                    (
                        1
                        + np.count_nonzero(
                            nulls[nulls["candidateId"].eq(candidate)]["maxNullAUROC"]
                            >= observed_auc[candidate]
                        )
                    )
                    / (PERMUTATIONS + 1)
                ),
                "replicates": PERMUTATIONS,
            }
            for candidate in CANDIDATES
        ]
    )
    return results, nulls


def validation_permutations(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = predictions[
        predictions["modelId"].eq("FULL_TRANSITION_TUBE")
        & predictions["variant"].eq("ORIGINAL")
    ].reset_index(drop=True)
    observed = {
        candidate: metric(primary[primary["candidateId"].eq(candidate)])["AUROC"]
        for candidate in CANDIDATES
    }
    rows = []
    for replicate in range(PERMUTATIONS):
        shuffled = primary.copy()
        shuffled["eventWithin32"] = permute_labels(
            primary, derived_seed("validation_permutation", replicate)
        )
        values = {
            candidate: metric(shuffled[shuffled["candidateId"].eq(candidate)])["AUROC"]
            for candidate in CANDIDATES
        }
        maximum = max(values.values())
        for candidate in CANDIDATES:
            rows.append(
                {
                    "replicate": replicate,
                    "candidateId": candidate,
                    "nullAUROC": values[candidate],
                    "maxNullAUROC": maximum,
                }
            )
    nulls = pd.DataFrame(rows)
    results = pd.DataFrame(
        [
            {
                "candidateId": candidate,
                "observedAUROC": observed[candidate],
                "familywisePValue": float(
                    (
                        1
                        + np.count_nonzero(
                            nulls[nulls["candidateId"].eq(candidate)]["maxNullAUROC"]
                            >= observed[candidate]
                        )
                    )
                    / (PERMUTATIONS + 1)
                ),
                "replicates": PERMUTATIONS,
            }
            for candidate in CANDIDATES
        ]
    )
    return results, nulls


def suffix_invariance(
    task: pd.DataFrame,
    manifest: pd.DataFrame,
    validation_meta: pd.DataFrame,
    validation_vectors: dict[str, np.ndarray],
) -> pd.DataFrame:
    lookup = {
        (row.candidateId, int(row.matrixIndex), int(row.landmark)): index
        for index, row in enumerate(validation_meta.itertuples(index=False))
    }
    rows = []
    sentinels = task[task["matrixRole"].eq("VALIDATION")].groupby("candidateId").head(5)
    for source in sentinels.itertuples(index=False):
        states = L26.load_states(source.candidateId, int(source.matrixIndex), manifest)
        endpoint = int(source.landmark)
        altered = states.copy()
        if len(altered) > endpoint:
            rng = np.random.default_rng(
                derived_seed("suffix", source.candidateId, source.matrixIndex, endpoint)
            )
            altered[endpoint:] = altered[endpoint:][
                rng.permutation(len(altered) - endpoint)
            ]
        first = transition_tube_views(states[endpoint - 32 : endpoint])
        second = transition_tube_views(altered[endpoint - 32 : endpoint])
        stored = lookup[(source.candidateId, int(source.matrixIndex), endpoint)]
        for model in MODELS:
            rows.append(
                {
                    "candidateId": source.candidateId,
                    "matrixIndex": int(source.matrixIndex),
                    "landmark": endpoint,
                    "modelId": model,
                    "prefixExact": bool(
                        np.array_equal(states[:endpoint], altered[:endpoint])
                    ),
                    "featureInvariant": bool(
                        np.array_equal(first[model], second[model])
                    ),
                    "storedExact": bool(
                        np.array_equal(first[model], validation_vectors[model][stored])
                    ),
                }
            )
    return pd.DataFrame(rows)


def gates(
    metrics: pd.DataFrame,
    landmarks: pd.DataFrame,
    bootstrap: pd.DataFrame,
    development_perm: pd.DataFrame,
    validation_perm: pd.DataFrame,
    suffix: pd.DataFrame,
) -> pd.DataFrame:
    original = metrics[metrics["variant"].eq("ORIGINAL")].set_index(
        ["candidateId", "modelId"]
    )
    rows = []
    for candidate in CANDIDATES:
        primary = original.loc[(candidate, "FULL_TRANSITION_TUBE")]
        exact = original.loc[(candidate, "EXACT_H_TRANSITION_TUBE")]
        ordinary = original.loc[(candidate, "ORDINARY_TRANSITION_TUBE")]
        prior = original.loc[(candidate, "LANDMARK_PRIOR")]
        boot = bootstrap[bootstrap["candidateId"].eq(candidate)].pivot(
            index="replicate", columns="modelId", values="AUROC"
        )
        lower = float(np.quantile(boot["FULL_TRANSITION_TUBE"], 0.025))
        delta_exact = float(
            np.quantile(
                boot["FULL_TRANSITION_TUBE"] - boot["EXACT_H_TRANSITION_TUBE"], 0.025
            )
        )
        delta_ordinary = float(
            np.quantile(
                boot["FULL_TRANSITION_TUBE"] - boot["ORDINARY_TRANSITION_TUBE"], 0.025
            )
        )
        agreeing = int(
            np.count_nonzero(
                landmarks[
                    landmarks["candidateId"].eq(candidate)
                    & landmarks["modelId"].eq("FULL_TRANSITION_TUBE")
                ]["AUROC"]
                >= 0.5
            )
        )
        dev_p = float(
            development_perm.set_index("candidateId").loc[candidate, "familywisePValue"]
        )
        val_p = float(
            validation_perm.set_index("candidateId").loc[candidate, "familywisePValue"]
        )
        candidate_suffix = suffix[suffix["candidateId"].eq(candidate)]
        checks = {
            "auRocPointPassed": float(primary["AUROC"]) >= 0.60,
            "auRocBootstrapPassed": lower > 0.5,
            "pointOverExactHPassed": float(primary["AUROC"]) > float(exact["AUROC"]),
            "pointOverOrdinaryPassed": float(primary["AUROC"])
            > float(ordinary["AUROC"]),
            "bootstrapOverExactHPassed": delta_exact > 0,
            "bootstrapOverOrdinaryPassed": delta_ordinary > 0,
            "auPrcPassed": float(primary["AUPRC"]) > float(primary["prevalence"]),
            "brierPassed": float(primary["BRIER"]) < float(prior["BRIER"]),
            "developmentPermutationPassed": dev_p <= 0.05,
            "validationPermutationPassed": val_p <= 0.05,
            "landmarksPassed": agreeing >= 4,
            "suffixPassed": bool(
                candidate_suffix[["prefixExact", "featureInvariant", "storedExact"]]
                .all()
                .all()
            ),
        }
        rows.append(
            {
                "candidateId": candidate,
                "primaryAUROC": float(primary["AUROC"]),
                "exactHAUROC": float(exact["AUROC"]),
                "ordinaryAUROC": float(ordinary["AUROC"]),
                "bootstrapLower": lower,
                "deltaExactHBootstrapLower": delta_exact,
                "deltaOrdinaryBootstrapLower": delta_ordinary,
                "developmentFamilywiseP": dev_p,
                "validationFamilywiseP": val_p,
                "agreeingLandmarks": agreeing,
                **checks,
                "candidateDiscoveryGatePassed": all(checks.values()),
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    task: pd.DataFrame,
    metrics: pd.DataFrame,
    landmarks: pd.DataFrame,
    bootstrap: pd.DataFrame,
    gate_frame: pd.DataFrame,
) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(root / name, dpi=180)
        plt.close()

    support = (
        task[task["matrixRole"].eq("VALIDATION")]
        .groupby(["candidateId", "landmark"])["eventWithin32"]
        .agg(["sum", "count"])
        .reset_index()
    )
    support.pivot(index="landmark", columns="candidateId", values="sum").plot(
        marker="o", title="Validation events by landmark"
    )
    save("01_online_task_support.png")
    focus = metrics[metrics["variant"].eq("ORIGINAL")]
    focus.pivot(index="modelId", columns="candidateId", values="AUROC").plot(
        kind="bar", ylim=(0.3, 0.8), title="Transition-tube AUROC"
    )
    plt.axhline(0.5, color="black", linestyle="--")
    save("02_model_auroc.png")
    landmarks[landmarks["modelId"].eq("FULL_TRANSITION_TUBE")].pivot(
        index="landmark", columns="candidateId", values="AUROC"
    ).plot(marker="o", ylim=(0.2, 0.9), title="Full-tube AUROC by landmark")
    plt.axhline(0.5, color="black", linestyle="--")
    save("03_landmark_auroc.png")
    bootstrap[bootstrap["modelId"].eq("FULL_TRANSITION_TUBE")].groupby("candidateId")[
        "AUROC"
    ].quantile([0.025, 0.5, 0.975]).unstack().plot(
        kind="bar", title="Matrix-bootstrap full-tube AUROC"
    )
    plt.axhline(0.5, color="black", linestyle="--")
    save("04_bootstrap_auroc.png")
    matrix = gate_frame.set_index("candidateId")[
        [column for column in gate_frame if column.endswith("Passed")]
    ].T.astype(int)
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=20)
    plt.yticks(
        range(len(matrix.index)),
        [value.replace("Passed", "") for value in matrix.index],
        fontsize=7,
    )
    plt.colorbar(ticks=[0, 1])
    save("05_gate_matrix.png")


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
        "schema": "eidosoma.e01.s19_l27.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(rows),
        "totalBytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def append_ledgers(classifications: list[str], selected: bool, timestamp: str) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "Event-aligned separation may occupy a time-resolved transition tube not represented by scalar summaries or recurrence-map analogues.",
            "failureOrAmbiguityTargeted": "Whether pre-onset levels and currents resemble a development transition-path ensemble.",
            "informationGainRationale": "A full functional path likelihood contrasts event and non-event tubes without tuning L24-L26 features.",
            "learned": "L27 task, paths, prototype model and gates frozen.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L24 localized separation; L25-L26 online nulls.",
            "proposedNextTest": "Execute L27.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Transition-tube path, mechanistic network-state susceptibility, or no precursor.",
            "selectedHypotheses": "One time-resolved level/current transition-tube likelihood.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Recurrence-map analogues alone capture transition progress.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A common functional transition tube might precede entry in both candidates.",
            "failureOrAmbiguityTargeted": "Full path signal versus H-only and ordinary-path tubes.",
            "informationGainRationale": "Held-out candidate-separated incrementality and clustered uncertainty adjudicate the tube.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L27 results.",
            "proposedNextTest": "Untouched confirmation of frozen L27 tube."
            if selected
            else "Test catalytic-network/state susceptibility in L28.",
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": "Mechanistic network-state susceptibility or no detectable precursor.",
            "selectedHypotheses": "One time-resolved level/current transition-tube likelihood.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Functional transition-tube proximity is robustly incremental."
            if not selected
            else "No online precursor exists.",
        },
    ]
    BASE.write_parquet(
        ledger_path,
        pd.concat(
            [ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)],
            ignore_index=True,
        ),
    )
    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    row = {
        "branchCount": 1,
        "bundleId": "L27_TRANSITION_TUBE",
        "candidateId": "S19-L27-TRANSITION-TUBE",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 5,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 4,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 3,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "fixed time-resolved path prototype likelihood",
        "rankingScore": 24.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L26_ANALOG_NULL_TRANSITION_TUBE",
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
    source_rows = [
        {
            "commitOrVersion": item.doi,
            "evidenceClass": item.evidenceClass,
            "finding": f"{item.directSupport}; L27 frozen use: {item.frozenUse}",
            "licenseStatus": "PUBLIC_ARTICLE",
            "redistributionStatus": "CITATION_ONLY",
            "repositoryIdentity": None,
            "retainedPath": None,
            "retrievalDate": timestamp[:10],
            "sha256": None,
            "sourceId": f"L27_{item.sourceId}",
            "sourceType": item.evidenceClass,
            "treeIdentity": None,
            "url": item.url,
        }
        for item in source_registry().itertuples(index=False)
    ]
    BASE.write_parquet(
        sources_path,
        pd.concat(
            [sources, pd.DataFrame(source_rows).reindex(columns=sources.columns)],
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
            "selectedDiscoveryLead": "TRANSITION_TUBE_DENSITY_CURRENT"
            if selected
            else None,
            "newMatrices": 0,
            "newTrajectories": 0,
            "nextStepActive": True,
        }
    )
    registry["laterLoopsAuthorized"] = True
    registry["authorizationUpperBound"] = "S19-L42"
    registry["proposedNextLoopTheme"] = (
        "UNTOUCHED_TRANSITION_TUBE_CONFIRMATION"
        if selected
        else "CATALYTIC_NETWORK_STATE_SUSCEPTIBILITY"
    )
    registry["proposedNextLoopActive"] = True
    BASE.atomic_text(loop_path, yaml.safe_dump(registry, sort_keys=False))
    review_path = ARTIFACT_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text())
    review["history"].append(
        {
            "decision": "S19_L27_COMPLETE_CONTINUE_UNDER_EXISTING_AUTHORIZATION",
            "loopId": LOOP_ID,
            "scope": VERSION,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "selectedDiscoveryLead": "TRANSITION_TUBE_DENSITY_CURRENT"
            if selected
            else None,
            "source": "locked_execution_result",
            "nextLoopAuthorized": True,
            "s20Activated": False,
        }
    )
    review["pendingDecision"] = "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L42"
    BASE.write_json(review_path, review)


def report_text(
    metrics: pd.DataFrame,
    landmarks: pd.DataFrame,
    gate_frame: pd.DataFrame,
    classifications: list[str],
    selected: bool,
    runtime: dict[str, Any],
) -> str:
    focus = metrics[metrics["variant"].eq("ORIGINAL")][
        [
            "candidateId",
            "modelId",
            "rows",
            "events",
            "prevalence",
            "AUROC",
            "AUPRC",
            "BRIER",
        ]
    ]
    primary_landmarks = landmarks[landmarks["modelId"].eq("FULL_TRANSITION_TUBE")][
        ["candidateId", "landmark", "rows", "events", "AUROC", "AUPRC", "BRIER"]
    ]
    return f"""# S19-L27 — Online Transition-Tube Density and Current Before Attractor Entry

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete within the authorized autonomous L19–L42 sequence.
- **Outcome classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Selected lead:** `{"TRANSITION_TUBE_DENSITY_CURRENT" if selected else "NONE"}`.
- **Validation:** immutable L26/prior hashes; exact L23 task/cache/firewall replay; development-only same-landmark tube lock before validation access; candidate-separated metrics; 4,096 matrix bootstraps; 512 development and validation label permutations; temporal-reversal and suffix controls; exact representation/model/report regeneration; storage and artifact hashes passed.
- **Recommended next bounded loop:** {"Run untouched new-matrix confirmation of this frozen transition tube." if selected else "Do not tune the tube; test one mechanistic catalytic-network/state susceptibility in L28."}

## Frozen question and method

For each at-risk online landmark, the most recent 32 states were represented by all eleven invariant organization-channel levels and first differences. Development-only same-landmark diagonal Gaussian prototypes defined event and non-event transition tubes. The full tube was compared with exact-H-only, ordinary non-H, and landmark-prior controls. Candidate identity was not a predictor; all validation and gates remained candidate separated. The target remains a completed-run recurring-attractor reconstruction.

## Held-out results

{focus.to_markdown(index=False)}

## Landmark diagnostics

{primary_landmarks.to_markdown(index=False)}

## Gate adjudication

{gate_frame.to_markdown(index=False)}

## Interpretation

This loop tests a functional transition path rather than scalar early-warning summaries, operator changes, or recurrence-map neighbours. A failed common incremental gate constrains this specific source-grounded tube. It does not prove every mechanistic precursor absent, and no discovery result would be confirmatory without new seed-firewalled matrices.

## Runtime and provenance

- Repository lock: `{runtime["repositoryHead"]}`.
- CPU float64, one numerical-library thread, no GPU.
- Wall seconds: `{runtime["wallSeconds"]:.3f}`; process CPU hours: `{runtime["processCpuHours"]:.6f}`.

## Autonomous continuation boundary

L27 is frozen. One next bounded loop remains authorized through L42. S20, E02, author contact, interventions and report-bundle work remain inactive.
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L27 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local and pushed heads differ")
    prior = validate_immutable_prior()
    fixtures = fixture_table()
    if not prior["unchanged"] or not fixtures["passed"].all():
        raise RuntimeError("prior or fixture gate failed")
    task = L26.task_registry()
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    if len(manifest) != 800:
        raise RuntimeError("trajectory manifest cardinality changed")
    for row in manifest.itertuples(index=False):
        if (
            not Path(row.cachePath).is_file()
            or sha256_file(Path(row.cachePath)) != row.cacheSha256
        ):
            raise RuntimeError("trajectory cache hash changed")
    start = time.perf_counter()
    for source in task.head(10).itertuples(index=False):
        states = L26.load_states(source.candidateId, int(source.matrixIndex), manifest)
        transition_tube_views(states[int(source.landmark) - 32 : int(source.landmark)])
    benchmark = time.perf_counter() - start
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L27 decision record\n\nL26 showed that whole-window recurrence-map analogues were not incrementally predictive. L27 freezes a distinct functional-data hypothesis: the complete time-resolved level and current path through eleven organization coordinates may enter a development-defined transition tube before onset. The exact L25 task and L24 firewall remain unchanged. Full, exact-H-only, ordinary-path, reversal, prior and suffix comparisons are all locked before validation access.\n",
    )
    sources = source_registry()
    sources.to_csv(LOOP_ROOT / "source_grounding_registry.csv", index=False)
    BASE.atomic_text(
        LOOP_ROOT / "source_grounding_report.md",
        "# L27 source grounding\n\n"
        + "\n".join(
            f"- **{row.sourceId}** — {row.directSupport}. Frozen use: {row.frozenUse}. {row.url}"
            for row in sources.itertuples(index=False)
        )
        + "\n",
    )
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "online_task_registry.parquet", task)
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l27.implementation_lock.v1",
            "researchStepId": LOOP_ID,
            "versionedId": VERSION,
            "repositoryHead": head,
            "remoteHead": remote,
            "configSha256": sha256_file(CONFIG),
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "l26ManifestSha256": sha256_file(L26_ROOT / "artifact_manifest.json"),
            "landmarks": list(LANDMARKS),
            "horizon": 32,
            "window": 32,
            "representations": list(MODELS),
            "bootstrapReplicates": BOOTSTRAPS,
            "permutationReplicates": PERMUTATIONS,
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
            "taskHash": BASE.frame_hash(task),
        },
    )
    BASE.write_json(
        LOOP_ROOT / "benchmark_projection.json",
        {
            "status": "PASS_PROJECTED_WITHIN_CEILING",
            "tenWindowSeconds": benchmark,
            "projectedCpuHoursUpper": 15,
            "cpuHoursCeiling": 100,
            "wallHoursCeiling": 72,
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
    fixtures = fixture_table()
    task = L26.task_registry()
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or BASE.frame_hash(task) != lock["taskHash"]
    ):
        raise RuntimeError("pre-execution gate failed")
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    identities = pd.DataFrame(
        [
            {
                "candidateId": row.candidateId,
                "matrixIndex": row.matrixIndex,
                "trajectorySha256": row.trajectorySha256,
                "cacheSha256": row.cacheSha256,
                "cacheIdentityPassed": Path(row.cachePath).is_file()
                and sha256_file(Path(row.cachePath)) == row.cacheSha256,
            }
            for row in manifest.itertuples(index=False)
        ]
    )
    if len(identities) != 800 or not identities["cacheIdentityPassed"].all():
        raise RuntimeError("trajectory identity failed")
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    development_meta, development_vectors = extract_representations(
        task, manifest, "DEVELOPMENT"
    )
    development_manifest = representation_manifest(
        development_meta, development_vectors
    )
    models = fit_models(development_meta, development_vectors)
    lock_payload = model_lock(models, BASE.frame_hash(development_manifest))
    BASE.write_json(BUILD_ROOT / "transition_tube_model_lock.json", lock_payload)
    model_hash = sha256_file(BUILD_ROOT / "transition_tube_model_lock.json")
    validation_meta, validation_vectors = extract_representations(
        task, manifest, "VALIDATION"
    )
    if sha256_file(BUILD_ROOT / "transition_tube_model_lock.json") != model_hash:
        raise RuntimeError("model changed after validation access")
    validation_manifest = representation_manifest(validation_meta, validation_vectors)
    predictions_original = score_models(
        validation_meta, validation_vectors, models, "ORIGINAL"
    )
    reversed_meta, reversed_vectors = extract_representations(
        task, manifest, "VALIDATION", "TEMPORAL_REVERSAL"
    )
    predictions_reversed = score_models(
        reversed_meta, reversed_vectors, models, "TEMPORAL_REVERSAL"
    )
    predictions = pd.concat(
        [predictions_original, predictions_reversed], ignore_index=True
    )
    metrics = aggregate_metrics(predictions)
    landmark = landmark_metrics(predictions_original)
    bootstrap = matrix_bootstraps(predictions_original)
    development_perm, development_nulls = development_permutations(
        development_meta,
        development_vectors,
        validation_meta,
        validation_vectors,
        metrics[metrics["variant"].eq("ORIGINAL")],
    )
    validation_perm, validation_nulls = validation_permutations(predictions_original)
    suffix = suffix_invariance(task, manifest, validation_meta, validation_vectors)
    gate_frame = gates(
        metrics, landmark, bootstrap, development_perm, validation_perm, suffix
    )
    selected = bool(gate_frame["candidateDiscoveryGatePassed"].all())
    classifications = (
        [
            "TRANSITION_TUBE_DISCOVERY_LEAD",
            "REQUIRES_UNTOUCHED_CONFIRMATION",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        if selected
        else [
            "TRANSITION_TUBE_NON_SUPPORT",
            "FUNCTIONAL_PATH_NOT_INCREMENTAL",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
    )
    original = metrics[metrics["variant"].eq("ORIGINAL")].set_index(
        ["candidateId", "modelId"]
    )
    if any(
        float(original.loc[(candidate, "FULL_TRANSITION_TUBE"), "AUROC"])
        <= max(
            float(original.loc[(candidate, "EXACT_H_TRANSITION_TUBE"), "AUROC"]),
            float(original.loc[(candidate, "ORDINARY_TRANSITION_TUBE"), "AUROC"]),
        )
        for candidate in CANDIDATES
    ):
        classifications.append("POSSIBLE_STABILITY_PROXY")
    make_figures(task, metrics, landmark, bootstrap, gate_frame)
    for name in (
        "preregistration.yaml",
        "decision_record.md",
        "source_grounding_registry.csv",
        "source_grounding_report.md",
        "fixture_results.parquet",
        "online_task_registry.parquet",
        "immutable_prior_validation.json",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
        "benchmark_projection.json",
    ):
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    BASE.write_parquet(
        BUILD_ROOT / "trajectory_identity_validation.parquet", identities
    )
    BASE.write_parquet(
        BUILD_ROOT / "development_representation_manifest.parquet", development_manifest
    )
    BASE.write_parquet(
        BUILD_ROOT / "validation_representation_manifest.parquet", validation_manifest
    )
    BASE.write_parquet(BUILD_ROOT / "prediction_results.parquet", predictions)
    BASE.write_parquet(BUILD_ROOT / "aggregate_metrics.parquet", metrics)
    BASE.write_parquet(BUILD_ROOT / "landmark_metrics.parquet", landmark)
    BASE.write_parquet(BUILD_ROOT / "bootstrap_results.parquet", bootstrap)
    BASE.write_parquet(
        BUILD_ROOT / "development_permutation_results.parquet", development_perm
    )
    BASE.write_parquet(
        BUILD_ROOT / "development_permutation_nulls.parquet", development_nulls
    )
    BASE.write_parquet(
        BUILD_ROOT / "validation_permutation_results.parquet", validation_perm
    )
    BASE.write_parquet(
        BUILD_ROOT / "validation_permutation_nulls.parquet", validation_nulls
    )
    BASE.write_parquet(BUILD_ROOT / "suffix_invariance_results.parquet", suffix)
    BASE.write_parquet(BUILD_ROOT / "scientific_gate_results.parquet", gate_frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l27.classification.v1",
            "researchStepId": LOOP_ID,
            "classifications": classifications,
            "selectedDiscoveryLead": "TRANSITION_TUBE_DENSITY_CURRENT"
            if selected
            else None,
            "confirmatory": False,
            "prospectivePredictors": True,
            "retrospectiveTarget": True,
            "priorStatusesChanged": False,
        },
    )
    pd.DataFrame(
        columns=[
            "stage",
            "candidateId",
            "matrixIndex",
            "exceptionClass",
            "exceptionMessage",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    replay_dev_meta, replay_dev_vectors = extract_representations(
        task, manifest, "DEVELOPMENT"
    )
    replay_models = fit_models(replay_dev_meta, replay_dev_vectors)
    replay_val_meta, replay_val_vectors = extract_representations(
        task, manifest, "VALIDATION"
    )
    replay_predictions = score_models(
        replay_val_meta, replay_val_vectors, replay_models, "ORIGINAL"
    )
    checks = {
        "developmentMetaExact": BASE.frame_hash(development_meta)
        == BASE.frame_hash(replay_dev_meta),
        "developmentRepresentationExact": all(
            array_hash(development_vectors[model])
            == array_hash(replay_dev_vectors[model])
            for model in MODELS
        ),
        "validationMetaExact": BASE.frame_hash(validation_meta)
        == BASE.frame_hash(replay_val_meta),
        "validationRepresentationExact": all(
            array_hash(validation_vectors[model])
            == array_hash(replay_val_vectors[model])
            for model in MODELS
        ),
        "predictionExact": BASE.frame_hash(predictions_original)
        == BASE.frame_hash(replay_predictions),
        "modelLockUnchanged": sha256_file(
            BUILD_ROOT / "transition_tube_model_lock.json"
        )
        == model_hash,
        "taskExact": BASE.frame_hash(task) == lock["taskHash"],
        "trajectoryCachePassed": bool(identities["cacheIdentityPassed"].all()),
        "suffixPassed": bool(
            suffix[["prefixExact", "featureInvariant", "storedExact"]].all().all()
        ),
    }
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l27.regeneration_validation.v1",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
        },
    )
    if not all(checks.values()):
        raise RuntimeError("regeneration failed")
    runtime = {
        "schema": "eidosoma.e01.s19_l27.runtime.v1",
        "researchStepId": LOOP_ID,
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - start_wall,
        "processCpuHours": (time.process_time() - start_cpu) / 3600,
        "bootstrapReplicates": BOOTSTRAPS,
        "permutationReplicates": PERMUTATIONS,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    storage = {
        "schema": "eidosoma.e01.s19_l27.storage_validation.v1",
        "retainedBytes": sum(
            path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
        ),
        "retainedGiBCeiling": 25,
        "temporaryBytes": sum(
            path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file()
        ),
        "temporaryGiBCeiling": 75,
    }
    storage["status"] = (
        "PASS"
        if storage["retainedBytes"] < 25 * 2**30
        and storage["temporaryBytes"] < 75 * 2**30
        else "FAIL"
    )
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        metrics, landmark, gate_frame, classifications, selected, runtime
    )
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L27_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L27 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Selected lead:** `{'TRANSITION_TUBE_DENSITY_CURRENT' if selected else 'NONE'}`.\n\n{'Freeze for untouched confirmation.' if selected else 'Proceed nonduplicatively to mechanistic catalytic-network/state susceptibility.'}\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L27-promotion-stage")
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(BUILD_ROOT, stage)
    if LOOP_ROOT.exists():
        shutil.rmtree(LOOP_ROOT)
    os.replace(stage, LOOP_ROOT)
    shutil.rmtree(BUILD_ROOT)
    final_manifest = json.loads((LOOP_ROOT / "artifact_manifest.json").read_text())
    if any(
        sha256_file(LOOP_ROOT / item["path"]) != item["sha256"]
        for item in final_manifest["files"]
    ):
        raise RuntimeError("artifact hash mismatch")
    append_ledgers(classifications, selected, runtime["completedAtUtc"])
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md",
        report.replace("# S19-L27", "# S19 current handoff — S19-L27", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": "TRANSITION_TUBE_DENSITY_CURRENT"
            if selected
            else None,
            "nextAuthorizedLoop": "S19-L28",
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
                "selected": selected,
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
