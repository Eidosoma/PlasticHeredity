"""Pure utilities for the matched-dimension nuisance-PCA control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


EPSILON = 1e-7
RIDGE_C = 0.1
PCA_COMPONENTS = 12
DERANGEMENT_REPETITIONS = 32
BOOTSTRAP_REPETITIONS = 4096
RANDOMIZATION_REPETITIONS = 4096
MASTER_SEED = "matched-dimension-nuisance-pca-v1-2026-08-19"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derived_seed(*parts: object) -> int:
    label = "|".join((MASTER_SEED, *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def sattolo_indices(size: int, rng: np.random.Generator) -> NDArray[np.int64]:
    """Return a single-cycle permutation with no fixed points."""

    if size < 2:
        raise ValueError("a fixed-point-free permutation needs at least two rows")
    order = np.arange(size, dtype=np.int64)
    for index in range(size - 1, 0, -1):
        other = int(rng.integers(0, index))
        order[index], order[other] = order[other], order[index]
    if np.any(order == np.arange(size)):
        raise AssertionError("Sattolo construction produced a fixed point")
    return order


def grouped_derangement(
    matrix_ids: NDArray[np.integer],
    groups: NDArray[np.integer],
    seed: int,
) -> NDArray[np.int64]:
    """Map every row to a different matrix in the same phase group."""

    matrices = np.asarray(matrix_ids)
    strata = np.asarray(groups)
    if matrices.ndim != 1 or strata.ndim != 1 or matrices.shape != strata.shape:
        raise ValueError("matrix_ids and groups must be equal-length vectors")
    donors = np.empty(matrices.size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for group in np.unique(strata):
        rows = np.flatnonzero(strata == group)
        if np.unique(matrices[rows]).size != rows.size:
            raise ValueError(f"group {group} contains repeated matrix identifiers")
        donors[rows] = rows[sattolo_indices(rows.size, rng)]
    if not np.array_equal(np.sort(donors), np.arange(matrices.size)):
        raise AssertionError("grouped donor mapping is not a global row permutation")
    if np.any(strata[donors] != strata):
        raise AssertionError("derangement crossed a phase group")
    if np.any(matrices[donors] == matrices):
        raise AssertionError("derangement retained a matrix-state pairing")
    return donors


def sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    values = 1.0 / (1.0 + np.exp(-np.clip(logits, -709.0, 709.0)))
    return np.clip(values, EPSILON, 1.0 - EPSILON)


def fit_composite(
    history_design: NDArray[np.float64],
    components: NDArray[np.float64],
    targets: NDArray[np.integer],
    pipeline: str,
) -> tuple[StandardScaler | None, LogisticRegression]:
    """Fit the implementation-specific final ridge stage."""

    design = np.column_stack((components, history_design))
    if pipeline == "codex":
        if targets.ndim != 2 or targets.shape[0] != design.shape[0]:
            raise ValueError("Codex targets must be states by branches")
        scaler = StandardScaler().fit(design)
        fitted_design = np.repeat(scaler.transform(design), targets.shape[1], axis=0)
        fitted_targets = targets.reshape(-1).astype(np.int64)
        classifier = LogisticRegression(
            C=RIDGE_C,
            penalty="l2",
            solver="lbfgs",
            max_iter=2_000,
            random_state=0,
        ).fit(fitted_design, fitted_targets)
        return scaler, classifier
    if pipeline == "fable":
        if targets.ndim != 1 or targets.shape[0] != design.shape[0]:
            raise ValueError("Fable targets must be a state vector")
        classifier = LogisticRegression(
            C=RIDGE_C,
            penalty="l2",
            solver="lbfgs",
            max_iter=5_000,
        ).fit(design, targets.astype(np.int64))
        return None, classifier
    raise ValueError(f"unknown pipeline: {pipeline}")


def predict_composite(
    history_design: NDArray[np.float64],
    components: NDArray[np.float64],
    scaler: StandardScaler | None,
    classifier: LogisticRegression,
) -> NDArray[np.float64]:
    design = np.column_stack((components, history_design))
    transformed = scaler.transform(design) if scaler is not None else design
    return np.clip(classifier.predict_proba(transformed)[:, 1], EPSILON, 1.0 - EPSILON)


def event_from_h(values: Iterable[float], threshold: float = 0.90) -> bool:
    flags = np.asarray(tuple(values), dtype=np.float64) > threshold
    breaks = np.flatnonzero(~flags)
    if breaks.size == 0:
        return False
    run = 0
    for inherited in flags[int(breaks[0]) + 1 :]:
        run = run + 1 if bool(inherited) else 0
        if run >= 3:
            return True
    return False


def state_log_loss(
    probability: NDArray[np.float64], targets: NDArray[np.integer]
) -> NDArray[np.float64]:
    p = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] != p.size:
        raise ValueError("targets must be states by branches")
    return -(y * np.log(p[:, None]) + (1.0 - y) * np.log(1.0 - p[:, None])).mean(axis=1)


def matrix_means(
    values: NDArray[np.float64], matrix_ids: NDArray[np.integer]
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    matrices = np.asarray(matrix_ids)
    return np.asarray([array[matrices == matrix].mean() for matrix in np.unique(matrices)])


def bootstrap_interval(
    values: NDArray[np.float64],
    matrix_ids: NDArray[np.integer],
    seed: int,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> tuple[float, float]:
    clustered = matrix_means(values, matrix_ids)
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, clustered.size, size=(repetitions, clustered.size))
    draws = clustered[selections].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def sign_randomization_p(
    values: NDArray[np.float64],
    matrix_ids: NDArray[np.integer],
    seed: int,
    repetitions: int = RANDOMIZATION_REPETITIONS,
) -> float:
    clustered = matrix_means(values, matrix_ids)
    observed = float(clustered.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(repetitions, clustered.size))
    randomized = (signs * clustered[None, :]).mean(axis=1)
    return float((1 + np.count_nonzero(randomized >= observed)) / (repetitions + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(np.asarray(p_values))
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(p_values[int(index)]))
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()

