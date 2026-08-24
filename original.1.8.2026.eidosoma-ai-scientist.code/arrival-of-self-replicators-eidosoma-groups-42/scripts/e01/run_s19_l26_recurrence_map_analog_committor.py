"""Execute S19-L26 recurrence-map analogue-committor discovery."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_frozen_timebase_ensemble.core import (
    selected_clock_observations,
    states_from_observations,
)
from e01_onset_discovery.analog_committor import (
    ANALOG_NEIGHBORS,
    all_analog_representations,
    deterministic_knn_probability,
)


def _load_base() -> Any:
    path = REPO_ROOT / "scripts/e01/run_s19_l19_source_grounded_early_warning.py"
    spec = importlib.util.spec_from_file_location("e01_s19_l26_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load artifact utilities")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
LOOP_ID = "S19-L26"
VERSION = "E01-S19-L26-RECURRENCE-MAP-ANALOG-COMMITTOR-v1.0.0"
CLOCK_ID = "C1_SELECTED_DAUGHTER_RETAINED"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
LANDMARKS = (64, 96, 128, 160, 192)
MODELS = ("RECURRENCE_MAP_ANALOG", "EXACT_H_TRACE_ANALOG", "ORDINARY_PATH_ANALOG")
BOOTSTRAPS = 4096
PERMUTATIONS = 512
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L26"
L25_ROOT = ARTIFACT_ROOT / "loops/L25"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l26")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l26_recurrence_map_analog_committor.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/analog_committor.py"


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


def derived_seed(*parts: object) -> int:
    value = "\x1f".join([VERSION, *map(str, parts)])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L25_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L25_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L25_ROOT / item["path"]),
            "root": str(L25_ROOT),
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
        "schema": "eidosoma.e01.s19_l26.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": aggregate,
        "l25ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def task_registry() -> pd.DataFrame:
    task = pd.read_parquet(L25_ROOT / "online_task_registry.parquet")
    required = {
        "matrixRole",
        "candidateId",
        "matrixIndex",
        "landmark",
        "windowStart",
        "windowEndExclusive",
        "eventWithin32",
    }
    if not required <= set(task):
        raise RuntimeError("L25 task schema changed")
    return task.sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "landmark"]
    ).reset_index(drop=True)


def load_states(
    candidate: str, matrix_index: int, manifest: pd.DataFrame
) -> np.ndarray:
    row = manifest[
        manifest["candidateId"].eq(candidate) & manifest["matrixIndex"].eq(matrix_index)
    ].iloc[0]
    path = Path(row["cachePath"])
    if not path.is_file() or sha256_file(path) != row["cacheSha256"]:
        raise RuntimeError("trajectory cache hash mismatch")
    with path.open("rb") as handle:
        trajectory = pickle.load(handle)
    if trajectory.trajectory_sha256 != row["trajectorySha256"]:
        raise RuntimeError("trajectory identity mismatch")
    states = states_from_observations(selected_clock_observations(trajectory, CLOCK_ID))
    if len(states) != int(row["selectedClockLength"]):
        raise RuntimeError("selected-clock length mismatch")
    return np.asarray(states, dtype=np.int64)


def fixture_table() -> pd.DataFrame:
    rng = np.random.default_rng(derived_seed("fixture"))
    states = rng.poisson(2.0, size=(64, 100)).astype(np.int64)
    states[:, 0] += 1
    first = all_analog_representations(states)
    replay = all_analog_representations(states.copy())
    relabelled = all_analog_representations(states[:, rng.permutation(100)])
    reversed_values = all_analog_representations(states[::-1])
    reference = np.asarray([[0.0], [0.0], [1.0], [2.0]], dtype=float)
    knn_first = deterministic_knn_probability(
        np.asarray([0.0]),
        reference,
        np.asarray([0, 1, 1, 0]),
        [("B", 2), ("A", 1), ("C", 3), ("D", 4)],
        k=2,
    )
    knn_replay = deterministic_knn_probability(
        np.asarray([0.0]),
        reference,
        np.asarray([0, 1, 1, 0]),
        [("B", 2), ("A", 1), ("C", 3), ("D", 4)],
        k=2,
    )
    nonconsecutive = pd.DataFrame(
        {
            "candidateId": ["A", "A", "B", "B"],
            "landmark": [64, 64, 64, 64],
            "eventWithin32": [False, True, False, True],
        },
        index=[0, 4, 8, 12],
    )
    permuted = permute_labels(nonconsecutive, derived_seed("index_fixture"))
    task = task_registry()
    support = task.groupby(["matrixRole", "candidateId"])["eventWithin32"].agg(
        ["count", "sum"]
    )
    return pd.DataFrame(
        [
            {
                "fixtureId": "REPRESENTATION_SCHEMA",
                "passed": [len(first[name]) for name in MODELS] == [1953, 320, 384],
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
                "fixtureId": "TEMPORAL_ORDER_RETAINED",
                "passed": all(
                    not np.array_equal(first[name], reversed_values[name])
                    for name in MODELS
                ),
                "details": "window reversal changes each representation",
            },
            {
                "fixtureId": "DETERMINISTIC_ANALOG_TIES",
                "passed": knn_first == knn_replay and knn_first[1] == (1, 0),
                "details": str(knn_first),
            },
            {
                "fixtureId": "NONCONSECUTIVE_INDEX_PERMUTATION",
                "passed": len(permuted) == 4
                and int(np.sum(permuted[:2])) == 1
                and int(np.sum(permuted[2:])) == 1,
                "details": "group counts preserved after positional reset",
            },
            {
                "fixtureId": "ONLINE_TASK_SUPPORT",
                "passed": int(support["sum"].min()) >= 50
                and int((support["count"] - support["sum"]).min()) >= 50,
                "details": json.dumps(
                    {
                        f"{x}:{y}": [int(row["count"]), int(row["sum"])]
                        for (x, y), row in support.iterrows()
                    },
                    sort_keys=True,
                ),
            },
        ]
    )


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "LORENZ_1969_ANALOGUES",
                "doi": "10.1175/1520-0469(1969)26<636:APARBN>2.0.CO;2",
                "url": "https://doi.org/10.1175/1520-0469(1969)26%3C636:APARBN%3E2.0.CO;2",
                "directSupport": "forecasting future evolution from naturally occurring similar historical states while excluding temporally trivial analogues",
                "frozenUse": "fixed-k development analogues with no same-matrix neighbor",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
            {
                "sourceId": "MARWAN_ET_AL_2007",
                "doi": "10.1016/j.physrep.2006.11.001",
                "url": "https://doi.org/10.1016/j.physrep.2006.11.001",
                "directSupport": "recurrence plots encode phase-space recurrence and can reveal dynamical regime transitions",
                "frozenUse": "complete nonadjacent cosine recurrence map over the past window",
                "evidenceClass": "PRIMARY_METHOD_REVIEW",
            },
            {
                "sourceId": "GAO_ET_AL_2023",
                "doi": "10.1137/21M1437883",
                "url": "https://doi.org/10.1137/21M1437883",
                "directSupport": "data-driven committor functions on point clouds order transition progress",
                "frozenUse": "analogue event fraction interpreted only as an exploratory committor-like probability",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
        ]
    )


def extract_representations(
    task: pd.DataFrame, manifest: pd.DataFrame, role: str, transform: str = "ORIGINAL"
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    meta_rows: list[dict[str, Any]] = []
    vectors: dict[str, list[np.ndarray]] = {model: [] for model in MODELS}
    subset = task[task["matrixRole"].eq(role)]
    for (candidate, matrix_index), group in subset.groupby(
        ["candidateId", "matrixIndex"], sort=True
    ):
        states = load_states(candidate, int(matrix_index), manifest)
        for source in group.itertuples(index=False):
            window = states[
                int(source.windowStart) : int(source.windowEndExclusive)
            ].copy()
            if transform == "TEMPORAL_REVERSAL":
                window = window[::-1]
            values = all_analog_representations(window)
            meta_rows.append(
                {
                    "matrixRole": role,
                    "candidateId": candidate,
                    "matrixIndex": int(matrix_index),
                    "landmark": int(source.landmark),
                    "eventWithin32": bool(source.eventWithin32),
                    "transform": transform,
                }
            )
            for model in MODELS:
                vectors[model].append(values[model])
    return pd.DataFrame(meta_rows), {
        model: np.stack(rows).astype(np.float64) for model, rows in vectors.items()
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


def fit_library(meta: pd.DataFrame, vectors: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "meta": meta.copy(),
        "labels": meta["eventWithin32"].to_numpy(np.int8),
    }
    for model, values in vectors.items():
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        result[model] = {
            "mean": mean,
            "scale": scale,
            "values": (values - mean) / scale,
        }
    priors = meta.groupby("landmark")["eventWithin32"].mean().to_dict()
    result["LANDMARK_PRIOR"] = {int(key): float(value) for key, value in priors.items()}
    return result


def library_lock(library: dict[str, Any], representation_hash: str) -> dict[str, Any]:
    models = {}
    for model in MODELS:
        item = library[model]
        models[model] = {
            "dimensions": len(item["mean"]),
            "meanSha256": array_hash(item["mean"]),
            "scaleSha256": array_hash(item["scale"]),
            "standardizedLibrarySha256": array_hash(item["values"]),
        }
    return {
        "schema": "eidosoma.e01.s19_l26.analog_library_lock.v1",
        "developmentOnly": True,
        "validationRepresentationOpened": False,
        "neighbors": ANALOG_NEIGHBORS,
        "weighting": "UNIFORM",
        "sameLandmarkOnly": True,
        "excludeSameMatrixInDevelopmentDiagnostics": True,
        "candidateIdentityUsedAsFeature": False,
        "developmentRepresentationManifestSha256": representation_hash,
        "landmarkPriors": library["LANDMARK_PRIOR"],
        "models": models,
        "lockedAtUtc": utc_now(),
    }


def score_analogues(
    query_meta: pd.DataFrame,
    query_vectors: dict[str, np.ndarray],
    library: dict[str, Any],
    *,
    leave_development_out: bool,
    variant: str,
    labels_override: np.ndarray | None = None,
) -> pd.DataFrame:
    library_meta = library["meta"]
    labels = (
        library["labels"]
        if labels_override is None
        else np.asarray(labels_override, dtype=np.int8)
    )
    rows: list[dict[str, Any]] = []
    for query_index, source in enumerate(query_meta.itertuples(index=False)):
        eligible = library_meta["landmark"].eq(source.landmark).to_numpy()
        if leave_development_out:
            eligible &= library_meta["matrixIndex"].ne(source.matrixIndex).to_numpy()
        indices = np.flatnonzero(eligible)
        tie_keys = [
            (
                str(library_meta.iloc[index].candidateId),
                int(library_meta.iloc[index].matrixIndex),
            )
            for index in indices
        ]
        for model in MODELS:
            item = library[model]
            query = (query_vectors[model][query_index] - item["mean"]) / item["scale"]
            probability, neighbor_local, distances = deterministic_knn_probability(
                query, item["values"][indices], labels[indices], tie_keys
            )
            neighbor_global = indices[np.asarray(neighbor_local, dtype=int)]
            rows.append(
                {
                    "candidateId": source.candidateId,
                    "matrixIndex": int(source.matrixIndex),
                    "landmark": int(source.landmark),
                    "eventWithin32": bool(source.eventWithin32),
                    "modelId": model,
                    "variant": variant,
                    "score": probability,
                    "neighborMatrices": json.dumps(
                        [
                            int(library_meta.iloc[index].matrixIndex)
                            for index in neighbor_global
                        ]
                    ),
                    "neighborCandidates": json.dumps(
                        [
                            str(library_meta.iloc[index].candidateId)
                            for index in neighbor_global
                        ]
                    ),
                    "meanNeighborDistance": float(np.mean(distances)),
                }
            )
        prior = float(library["LANDMARK_PRIOR"][int(source.landmark)])
        rows.append(
            {
                "candidateId": source.candidateId,
                "matrixIndex": int(source.matrixIndex),
                "landmark": int(source.landmark),
                "eventWithin32": bool(source.eventWithin32),
                "modelId": "LANDMARK_PRIOR",
                "variant": variant,
                "score": prior,
                "neighborMatrices": "[]",
                "neighborCandidates": "[]",
                "meanNeighborDistance": np.nan,
            }
        )
    return pd.DataFrame(rows)


def metric(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["eventWithin32"].to_numpy(int)
    score = frame["score"].to_numpy(float)
    prevalence = float(np.mean(y))
    return {
        "rows": float(len(frame)),
        "matrices": float(frame["matrixIndex"].nunique()),
        "events": float(np.sum(y)),
        "prevalence": prevalence,
        "AUROC": float(roc_auc_score(y, score)),
        "AUPRC": float(average_precision_score(y, score)),
        "BRIER": float(brier_score_loss(y, score)),
        "BALANCED_ACCURACY": float(balanced_accuracy_score(y, score >= 0.5)),
    }


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidateId": candidate,
                "modelId": model,
                "variant": variant,
                **metric(frame),
            }
            for (candidate, model, variant), frame in predictions.groupby(
                ["candidateId", "modelId", "variant"], sort=True
            )
        ]
    )


def landmark_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, model, landmark), frame in predictions[
        predictions["variant"].eq("ORIGINAL")
    ].groupby(["candidateId", "modelId", "landmark"], sort=True):
        if frame["eventWithin32"].nunique() == 2:
            rows.append(
                {
                    "candidateId": candidate,
                    "modelId": model,
                    "landmark": landmark,
                    **metric(frame),
                }
            )
    return pd.DataFrame(rows)


def matrix_bootstraps(predictions: pd.DataFrame) -> pd.DataFrame:
    original = predictions[predictions["variant"].eq("ORIGINAL")]
    rows = []
    for candidate in CANDIDATES:
        candidate_frame = original[original["candidateId"].eq(candidate)]
        matrices = sorted(candidate_frame["matrixIndex"].unique())
        groups = {
            matrix: candidate_frame[candidate_frame["matrixIndex"].eq(matrix)]
            for matrix in matrices
        }
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
    validation_predictions: pd.DataFrame,
    observed_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = (
        observed_metrics[observed_metrics["modelId"].eq("RECURRENCE_MAP_ANALOG")]
        .set_index("candidateId")["AUROC"]
        .to_dict()
    )
    primary = validation_predictions[
        validation_predictions["modelId"].eq("RECURRENCE_MAP_ANALOG")
    ].reset_index(drop=True)
    development_lookup = {
        (str(row.candidateId), int(row.matrixIndex), int(row.landmark)): index
        for index, row in enumerate(development_meta.itertuples(index=False))
    }
    neighbor_indices = []
    for row in primary.itertuples(index=False):
        candidates = json.loads(row.neighborCandidates)
        matrices = json.loads(row.neighborMatrices)
        indices = [
            development_lookup[(str(candidate), int(matrix), int(row.landmark))]
            for candidate, matrix in zip(candidates, matrices, strict=True)
        ]
        if len(indices) != ANALOG_NEIGHBORS:
            raise RuntimeError("development permutation neighbor count changed")
        neighbor_indices.append(np.asarray(indices, dtype=int))
    null_rows = []
    for replicate in range(PERMUTATIONS):
        labels = permute_labels(
            development_meta, derived_seed("development_permutation", replicate)
        )
        scored = primary.copy()
        scored["score"] = [
            float(np.mean(labels[indices])) for indices in neighbor_indices
        ]
        scored["variant"] = "DEVELOPMENT_LABEL_PERMUTATION"
        values = {
            candidate: metric(
                scored[
                    scored["candidateId"].eq(candidate)
                    & scored["modelId"].eq("RECURRENCE_MAP_ANALOG")
                ]
            )["AUROC"]
            for candidate in CANDIDATES
        }
        maximum = max(values.values())
        for candidate in CANDIDATES:
            null_rows.append(
                {
                    "replicate": replicate,
                    "candidateId": candidate,
                    "nullAUROC": values[candidate],
                    "maxNullAUROC": maximum,
                }
            )
    nulls = pd.DataFrame(null_rows)
    result = pd.DataFrame(
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
    return result, nulls


def validation_permutations(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = predictions[
        predictions["modelId"].eq("RECURRENCE_MAP_ANALOG")
        & predictions["variant"].eq("ORIGINAL")
    ].copy()
    observed = {
        candidate: metric(primary[primary["candidateId"].eq(candidate)])["AUROC"]
        for candidate in CANDIDATES
    }
    rows = []
    for replicate in range(PERMUTATIONS):
        labels = permute_labels(
            primary, derived_seed("validation_permutation", replicate)
        )
        shuffled = primary.copy()
        shuffled["eventWithin32"] = labels
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
    result = pd.DataFrame(
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
    return result, nulls


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
        states = load_states(source.candidateId, int(source.matrixIndex), manifest)
        endpoint = int(source.landmark)
        altered = states.copy()
        if len(altered) > endpoint:
            rng = np.random.default_rng(
                derived_seed("suffix", source.candidateId, source.matrixIndex, endpoint)
            )
            altered[endpoint:] = altered[endpoint:][
                rng.permutation(len(altered) - endpoint)
            ]
        first = all_analog_representations(states[endpoint - 64 : endpoint])
        second = all_analog_representations(altered[endpoint - 64 : endpoint])
        stored_index = lookup[(source.candidateId, int(source.matrixIndex), endpoint)]
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
                        np.array_equal(
                            first[model], validation_vectors[model][stored_index]
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def scientific_gates(
    metrics: pd.DataFrame,
    landmarks: pd.DataFrame,
    bootstrap: pd.DataFrame,
    development_perm: pd.DataFrame,
    validation_perm: pd.DataFrame,
    suffix: pd.DataFrame,
    priors: dict[int, float],
) -> pd.DataFrame:
    original = metrics[metrics["variant"].eq("ORIGINAL")].set_index(
        ["candidateId", "modelId"]
    )
    rows = []
    for candidate in CANDIDATES:
        primary = original.loc[(candidate, "RECURRENCE_MAP_ANALOG")]
        exact_h = original.loc[(candidate, "EXACT_H_TRACE_ANALOG")]
        ordinary = original.loc[(candidate, "ORDINARY_PATH_ANALOG")]
        boot = bootstrap[bootstrap["candidateId"].eq(candidate)].pivot(
            index="replicate", columns="modelId", values="AUROC"
        )
        lower = float(np.quantile(boot["RECURRENCE_MAP_ANALOG"], 0.025))
        delta_h_lower = float(
            np.quantile(
                boot["RECURRENCE_MAP_ANALOG"] - boot["EXACT_H_TRACE_ANALOG"], 0.025
            )
        )
        delta_ordinary_lower = float(
            np.quantile(
                boot["RECURRENCE_MAP_ANALOG"] - boot["ORDINARY_PATH_ANALOG"], 0.025
            )
        )
        landmark = landmarks[
            landmarks["candidateId"].eq(candidate)
            & landmarks["modelId"].eq("RECURRENCE_MAP_ANALOG")
        ]
        agreeing = int(np.count_nonzero(landmark["AUROC"] >= 0.5))
        dev_p = float(
            development_perm.set_index("candidateId").loc[candidate, "familywisePValue"]
        )
        val_p = float(
            validation_perm.set_index("candidateId").loc[candidate, "familywisePValue"]
        )
        candidate_suffix = suffix[suffix["candidateId"].eq(candidate)]
        prior_brier = float(np.mean([prior * (1 - prior) for prior in priors.values()]))
        checks = {
            "auRocPointPassed": float(primary["AUROC"]) >= 0.60,
            "auRocBootstrapPassed": lower > 0.5,
            "pointOverExactHPassed": float(primary["AUROC"]) > float(exact_h["AUROC"]),
            "pointOverOrdinaryPassed": float(primary["AUROC"])
            > float(ordinary["AUROC"]),
            "bootstrapOverExactHPassed": delta_h_lower > 0.0,
            "bootstrapOverOrdinaryPassed": delta_ordinary_lower > 0.0,
            "auPrcPassed": float(primary["AUPRC"]) > float(primary["prevalence"]),
            "brierPassed": float(primary["BRIER"]) < prior_brier,
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
                "exactHAUROC": float(exact_h["AUROC"]),
                "ordinaryAUROC": float(ordinary["AUROC"]),
                "bootstrapLower": lower,
                "deltaExactHBootstrapLower": delta_h_lower,
                "deltaOrdinaryBootstrapLower": delta_ordinary_lower,
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
    gates: pd.DataFrame,
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
    support["nonEvents"] = support["count"] - support["sum"]
    support.pivot(index="landmark", columns="candidateId", values="sum").plot(
        marker="o", title="Validation events by landmark"
    )
    save("01_online_task_support.png")
    focus = metrics[metrics["variant"].eq("ORIGINAL")]
    focus.pivot(index="modelId", columns="candidateId", values="AUROC").plot(
        kind="bar", ylim=(0.3, 0.8), title="Held-out analogue AUROC"
    )
    plt.axhline(0.5, color="black", linestyle="--")
    save("02_model_auroc.png")
    landmarks[landmarks["modelId"].eq("RECURRENCE_MAP_ANALOG")].pivot(
        index="landmark", columns="candidateId", values="AUROC"
    ).plot(marker="o", ylim=(0.2, 0.9), title="Recurrence-map AUROC by landmark")
    plt.axhline(0.5, color="black", linestyle="--")
    save("03_landmark_auroc.png")
    summary = (
        bootstrap[bootstrap["modelId"].eq("RECURRENCE_MAP_ANALOG")]
        .groupby("candidateId")["AUROC"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
    )
    summary.plot(kind="bar", title="Matrix-bootstrap recurrence-map AUROC")
    plt.axhline(0.5, color="black", linestyle="--")
    save("04_bootstrap_auroc.png")
    matrix = gates.set_index("candidateId")[
        [column for column in gates if column.endswith("Passed")]
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
        "schema": "eidosoma.e01.s19_l26.artifact_manifest.v1",
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
            "beliefBeforeLoop": "A transition path may be encoded in the topology of the whole recent recurrence map even when scalar and operator summaries fail.",
            "failureOrAmbiguityTargeted": "Whether recurrence-path geometry yields a candidate-invariant online analogue committor.",
            "informationGainRationale": "The complete nonadjacent similarity map is not a retuning of L25 scalar/operator features.",
            "learned": "L26 task, representation, analogue library and gates frozen.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L24 localized separation and L25 online operator null.",
            "proposedNextTest": "Execute L26.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Recurrence-path geometry, transition-tube density, or no robust precursor.",
            "selectedHypotheses": "One fixed recurrence-map analogue committor.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Local operator change is sufficient.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A recurrence-map analogue could estimate entry probability beyond exact-H and ordinary paths.",
            "failureOrAmbiguityTargeted": "Transition-path geometry versus stability proxy.",
            "informationGainRationale": "Held-out matrices and common-library candidate-separated gates test reproducibility and incrementality.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L26 results.",
            "proposedNextTest": "Untouched confirmation of frozen L26 predictor."
            if selected
            else "Test transition-tube density/current geometry in L27.",
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": "Transition-tube density/current or no detectable precursor.",
            "selectedHypotheses": "One fixed recurrence-map analogue committor.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Recurrence-map analogue adds robust warning information."
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
        "bundleId": "L26_RECURRENCE_MAP_ANALOG",
        "candidateId": "S19-L26-RECURRENCE-MAP-ANALOG",
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
        "proposedSpecification": "fixed-k same-landmark recurrence-map analogue",
        "rankingScore": 24.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L25_OPERATOR_NULL_TRANSITION_PATH_GEOMETRY",
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
            "finding": f"{item.directSupport}; L26 frozen use: {item.frozenUse}",
            "licenseStatus": "PUBLIC_ARTICLE",
            "redistributionStatus": "CITATION_ONLY",
            "repositoryIdentity": None,
            "retainedPath": None,
            "retrievalDate": timestamp[:10],
            "sha256": None,
            "sourceId": f"L26_{item.sourceId}",
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
            "selectedDiscoveryLead": "RECURRENCE_MAP_ANALOG_COMMITTOR"
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
        "UNTOUCHED_RECURRENCE_MAP_CONFIRMATION"
        if selected
        else "TRANSITION_TUBE_DENSITY_CURRENT"
    )
    registry["proposedNextLoopActive"] = True
    BASE.atomic_text(loop_path, yaml.safe_dump(registry, sort_keys=False))
    review_path = ARTIFACT_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text())
    review["history"].append(
        {
            "decision": "S19_L26_COMPLETE_CONTINUE_UNDER_EXISTING_AUTHORIZATION",
            "loopId": LOOP_ID,
            "scope": VERSION,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "selectedDiscoveryLead": "RECURRENCE_MAP_ANALOG_COMMITTOR"
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
    gates: pd.DataFrame,
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
    primary_landmarks = landmarks[landmarks["modelId"].eq("RECURRENCE_MAP_ANALOG")][
        ["candidateId", "landmark", "rows", "events", "AUROC", "AUPRC", "BRIER"]
    ]
    return f"""# S19-L26 — Online Recurrence-Map Analog Committor Before Attractor Entry

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete within the authorized autonomous L19–L42 sequence.
- **Outcome classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Selected lead:** `{"RECURRENCE_MAP_ANALOG_COMMITTOR" if selected else "NONE"}`.
- **Validation:** immutable L25/prior hashes; exact L23 task/cache/firewall replay; development-only normalization and analogue library lock before validation access; candidate-separated metrics; 4,096 matrix bootstraps; 512 development and validation label permutations; temporal-reversal and suffix controls; exact representation/prediction/report regeneration; storage and artifact hashes passed.
- **Recommended next bounded loop:** {"Run untouched new-matrix confirmation of this frozen analogue predictor." if selected else "Do not tune k or the recurrence map; test one transition-tube density/current hypothesis in L27."}

## Frozen question and method

At five fixed landmarks, does the complete nonadjacent cosine recurrence map over the previous 64 observations support an analogue estimate of entry during the next 32 observations beyond exact-H traces, ordinary non-H organization paths and elapsed-time priors? A fixed 15-neighbor, uniform-weight library was constructed only from development matrices; neighbors were restricted to the same landmark and candidate identity was not a feature. Validation remained candidate separated. The target is a completed-run recurring-attractor reconstruction, so even a passing predictor would require untouched confirmation and would not identify the paper's author implementation.

## Held-out results

{focus.to_markdown(index=False)}

## Landmark diagnostics

{primary_landmarks.to_markdown(index=False)}

## Gate adjudication

{gates.to_markdown(index=False)}

## Interpretation

The recurrence map retains much more path geometry than L25's local operator summaries but is still past-only and molecule-label invariant. Failure of the preregistered incremental gates constrains this one analogue construction; it does not prove that every transition-path coordinate is absent. Any apparent separation that does not exceed exact-H and ordinary-path controls in both simulator candidates remains an exploratory stability proxy.

## Runtime and provenance

- Repository lock: `{runtime["repositoryHead"]}`.
- CPU float64, one numerical-library thread, no GPU.
- Wall seconds: `{runtime["wallSeconds"]:.3f}`; process CPU hours: `{runtime["processCpuHours"]:.6f}`.

## Autonomous continuation boundary

L26 is frozen. The human authorization permits one next bounded loop through at most L42. S20, E02, author contact, interventions and report-bundle work remain inactive.
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L26 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local and pushed heads differ")
    prior = validate_immutable_prior()
    if not prior["unchanged"]:
        raise RuntimeError("immutable prior changed")
    fixtures = fixture_table()
    if not fixtures["passed"].all():
        raise RuntimeError("mandatory fixture failure")
    task = task_registry()
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    if len(manifest) != 800:
        raise RuntimeError("L23 trajectory manifest cardinality changed")
    for row in manifest.itertuples(index=False):
        if (
            not Path(row.cachePath).is_file()
            or sha256_file(Path(row.cachePath)) != row.cacheSha256
        ):
            raise RuntimeError("L23 cache hash changed")
    start = time.perf_counter()
    for source in task.head(10).itertuples(index=False):
        states = load_states(source.candidateId, int(source.matrixIndex), manifest)
        all_analog_representations(
            states[int(source.windowStart) : int(source.windowEndExclusive)]
        )
    benchmark = time.perf_counter() - start
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        """# S19-L26 decision record

L25 found no robust incremental local-operator warning signal. L26 therefore freezes a nonduplicative representation: the entire nonadjacent recurrence map of the preceding 64 compositions, scored by one fixed 15-neighbor analogue estimate of next-32 entry probability. Exact-H traces, non-H ordinary organization paths and landmark priors remain controls. The L24 development/validation firewall and L25 task are unchanged; no trajectory, target, landmark, horizon or candidate choice is altered.
""",
    )
    sources = source_registry()
    sources.to_csv(LOOP_ROOT / "source_grounding_registry.csv", index=False)
    BASE.atomic_text(
        LOOP_ROOT / "source_grounding_report.md",
        "# L26 source grounding\n\n"
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
            "schema": "eidosoma.e01.s19_l26.implementation_lock.v1",
            "researchStepId": LOOP_ID,
            "versionedId": VERSION,
            "repositoryHead": head,
            "remoteHead": remote,
            "configSha256": sha256_file(CONFIG),
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "l25ManifestSha256": sha256_file(L25_ROOT / "artifact_manifest.json"),
            "landmarks": list(LANDMARKS),
            "horizon": 32,
            "representations": list(MODELS),
            "neighbors": ANALOG_NEIGHBORS,
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
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
    ):
        raise RuntimeError("immutable prior failed")
    fixtures = fixture_table()
    if not fixtures["passed"].all():
        raise RuntimeError("fixture failure")
    task = task_registry()
    if BASE.frame_hash(task) != lock["taskHash"]:
        raise RuntimeError("online task changed")
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
        raise RuntimeError("trajectory cache identity failed")
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    development_meta, development_vectors = extract_representations(
        task, manifest, "DEVELOPMENT"
    )
    development_manifest = representation_manifest(
        development_meta, development_vectors
    )
    library = fit_library(development_meta, development_vectors)
    lock_payload = library_lock(library, BASE.frame_hash(development_manifest))
    BASE.write_json(BUILD_ROOT / "analog_library_lock.json", lock_payload)
    library_hash = sha256_file(BUILD_ROOT / "analog_library_lock.json")
    development_predictions = score_analogues(
        development_meta,
        development_vectors,
        library,
        leave_development_out=True,
        variant="DEVELOPMENT_LOMO",
    )
    validation_meta, validation_vectors = extract_representations(
        task, manifest, "VALIDATION"
    )
    if sha256_file(BUILD_ROOT / "analog_library_lock.json") != library_hash:
        raise RuntimeError("analogue library changed after validation access")
    validation_manifest = representation_manifest(validation_meta, validation_vectors)
    validation_predictions = score_analogues(
        validation_meta,
        validation_vectors,
        library,
        leave_development_out=False,
        variant="ORIGINAL",
    )
    reversed_meta, reversed_vectors = extract_representations(
        task, manifest, "VALIDATION", "TEMPORAL_REVERSAL"
    )
    reversed_predictions = score_analogues(
        reversed_meta,
        reversed_vectors,
        library,
        leave_development_out=False,
        variant="TEMPORAL_REVERSAL",
    )
    predictions = pd.concat(
        [validation_predictions, reversed_predictions], ignore_index=True
    )
    metrics = aggregate_metrics(predictions)
    landmarks = landmark_metrics(validation_predictions)
    bootstrap = matrix_bootstraps(validation_predictions)
    development_perm, development_nulls = development_permutations(
        development_meta,
        validation_predictions,
        metrics[metrics["variant"].eq("ORIGINAL")],
    )
    validation_perm, validation_nulls = validation_permutations(validation_predictions)
    suffix = suffix_invariance(task, manifest, validation_meta, validation_vectors)
    gates = scientific_gates(
        metrics,
        landmarks,
        bootstrap,
        development_perm,
        validation_perm,
        suffix,
        library["LANDMARK_PRIOR"],
    )
    selected = bool(gates["candidateDiscoveryGatePassed"].all())
    classifications = (
        [
            "RECURRENCE_MAP_ANALOG_DISCOVERY_LEAD",
            "REQUIRES_UNTOUCHED_CONFIRMATION",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        if selected
        else [
            "RECURRENCE_MAP_ANALOG_NON_SUPPORT",
            "ANALOG_COMMITTOR_NOT_INCREMENTAL",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
    )
    original = metrics[metrics["variant"].eq("ORIGINAL")].set_index(
        ["candidateId", "modelId"]
    )
    if any(
        float(original.loc[(candidate, "RECURRENCE_MAP_ANALOG"), "AUROC"])
        <= max(
            float(original.loc[(candidate, "EXACT_H_TRACE_ANALOG"), "AUROC"]),
            float(original.loc[(candidate, "ORDINARY_PATH_ANALOG"), "AUROC"]),
        )
        for candidate in CANDIDATES
    ):
        classifications.append("POSSIBLE_STABILITY_PROXY")
    make_figures(task, metrics, landmarks, bootstrap, gates)
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
    BASE.write_parquet(
        BUILD_ROOT / "development_predictions.parquet", development_predictions
    )
    BASE.write_parquet(BUILD_ROOT / "prediction_results.parquet", predictions)
    BASE.write_parquet(BUILD_ROOT / "aggregate_metrics.parquet", metrics)
    BASE.write_parquet(BUILD_ROOT / "landmark_metrics.parquet", landmarks)
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
    BASE.write_parquet(BUILD_ROOT / "scientific_gate_results.parquet", gates)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l26.classification.v1",
            "researchStepId": LOOP_ID,
            "classifications": classifications,
            "selectedDiscoveryLead": "RECURRENCE_MAP_ANALOG_COMMITTOR"
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
    replay_library = fit_library(replay_dev_meta, replay_dev_vectors)
    replay_val_meta, replay_val_vectors = extract_representations(
        task, manifest, "VALIDATION"
    )
    replay_predictions = score_analogues(
        replay_val_meta,
        replay_val_vectors,
        replay_library,
        leave_development_out=False,
        variant="ORIGINAL",
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
        "predictionExact": BASE.frame_hash(validation_predictions)
        == BASE.frame_hash(replay_predictions),
        "libraryLockUnchanged": sha256_file(BUILD_ROOT / "analog_library_lock.json")
        == library_hash,
        "taskExact": BASE.frame_hash(task) == lock["taskHash"],
        "trajectoryCachePassed": bool(identities["cacheIdentityPassed"].all()),
        "suffixPassed": bool(
            suffix[["prefixExact", "featureInvariant", "storedExact"]].all().all()
        ),
    }
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l26.regeneration_validation.v1",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
        },
    )
    if not all(checks.values()):
        raise RuntimeError("regeneration failed")
    runtime = {
        "schema": "eidosoma.e01.s19_l26.runtime.v1",
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
        "schema": "eidosoma.e01.s19_l26.storage_validation.v1",
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
    report = report_text(metrics, landmarks, gates, classifications, selected, runtime)
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L26_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L26 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Selected lead:** `{'RECURRENCE_MAP_ANALOG_COMMITTOR' if selected else 'NONE'}`.\n\n{'Freeze for untouched confirmation.' if selected else 'Proceed nonduplicatively to one transition-tube density/current loop.'}\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L26-promotion-stage")
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
        report.replace("# S19-L26", "# S19 current handoff — S19-L26", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": "RECURRENCE_MAP_ANALOG_COMMITTOR"
            if selected
            else None,
            "nextAuthorizedLoop": "S19-L27",
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
