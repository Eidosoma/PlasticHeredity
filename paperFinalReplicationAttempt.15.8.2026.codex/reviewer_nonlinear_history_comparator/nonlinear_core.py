"""Pure models and inference for the nonlinear history-only control."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


EPSILON = 1e-7
PCA_COMPONENTS = 12
RIDGE_C = 0.1
CV_FOLDS = 5
SPLINE_QUANTILES = (0.25, 0.50, 0.75)
HISTORY_CLIP = 8.0
TREE_LEAF_GRID = (3, 7, 15)
TREE_ITERATIONS = 100
TREE_LEARNING_RATE = 0.05
TREE_L2 = 1.0
BOOTSTRAP_REPETITIONS = 4096
RANDOMIZATION_REPETITIONS = 4096
MASTER_SEED = "nonlinear-history-comparator-v1-2026-08-19"


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
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "big")


def _as_history(values: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("history must be a two-dimensional rows-by-features array")
    if not np.isfinite(array).all():
        raise ValueError("history contains non-finite values")
    return array


class SplineInteractionTransformer:
    """Compress a fixed nonlinear history library to twelve components.

    The library contains squared and cubic standardized variables, three
    truncated-cubic terms per variable, and every pairwise product. All
    transformations are fitted using development rows only.
    """

    def __init__(self, n_components: int = PCA_COMPONENTS) -> None:
        self.n_components = int(n_components)
        self.history_scaler: StandardScaler | None = None
        self.knots: NDArray[np.float64] | None = None
        self.keep: NDArray[np.bool_] | None = None
        self.library_scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.library_dimension_before_filter: int | None = None

    @staticmethod
    def _library(
        standardized: NDArray[np.float64], knots: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        z = np.clip(standardized, -HISTORY_CLIP, HISTORY_CLIP)
        columns: list[NDArray[np.float64]] = []
        for feature in range(z.shape[1]):
            column = z[:, feature]
            columns.extend((column**2, column**3))
            for knot in knots[feature]:
                columns.append(np.maximum(column - float(knot), 0.0) ** 3)
        for left in range(z.shape[1]):
            for right in range(left + 1, z.shape[1]):
                columns.append(z[:, left] * z[:, right])
        return np.column_stack(columns).astype(np.float64, copy=False)

    def fit(self, history: NDArray[np.floating]) -> "SplineInteractionTransformer":
        x = _as_history(history)
        self.history_scaler = StandardScaler().fit(x)
        z = self.history_scaler.transform(x)
        self.knots = np.quantile(z, SPLINE_QUANTILES, axis=0).T.astype(np.float64)
        library = self._library(z, self.knots)
        self.library_dimension_before_filter = int(library.shape[1])
        self.keep = np.var(library, axis=0) > 1e-12
        if int(self.keep.sum()) < self.n_components:
            raise ValueError("nonlinear history library has fewer usable columns than requested")
        retained = library[:, self.keep]
        self.library_scaler = StandardScaler().fit(retained)
        scaled = self.library_scaler.transform(retained)
        self.pca = PCA(n_components=self.n_components, svd_solver="full").fit(scaled)
        return self

    def transform(self, history: NDArray[np.floating]) -> NDArray[np.float64]:
        if any(
            item is None
            for item in (
                self.history_scaler,
                self.knots,
                self.keep,
                self.library_scaler,
                self.pca,
            )
        ):
            raise RuntimeError("transformer is not fitted")
        x = _as_history(history)
        assert self.history_scaler is not None
        assert self.knots is not None
        assert self.keep is not None
        assert self.library_scaler is not None
        assert self.pca is not None
        library = self._library(self.history_scaler.transform(x), self.knots)
        scaled = self.library_scaler.transform(library[:, self.keep])
        output = self.pca.transform(scaled)
        if output.shape != (x.shape[0], self.n_components):
            raise AssertionError("unexpected nonlinear component shape")
        return np.asarray(output, dtype=np.float64)

    def fit_transform(self, history: NDArray[np.floating]) -> NDArray[np.float64]:
        return self.fit(history).transform(history)

    def audit(self) -> dict[str, Any]:
        if self.pca is None or self.keep is None:
            raise RuntimeError("transformer is not fitted")
        return {
            "library_dimension_before_filter": self.library_dimension_before_filter,
            "library_dimension_after_filter": int(self.keep.sum()),
            "components": self.n_components,
            "explained_variance_ratio_sum": float(self.pca.explained_variance_ratio_.sum()),
        }


@dataclass
class RidgeBundle:
    transformer: SplineInteractionTransformer
    final_scaler: StandardScaler | None
    classifier: LogisticRegression
    pipeline: str


def expand_targets(
    design: NDArray[np.float64], targets: NDArray[np.integer]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    y = np.asarray(targets)
    if y.ndim == 1:
        if y.shape[0] != design.shape[0]:
            raise ValueError("one-dimensional target length mismatch")
        return design, y.astype(np.int64)
    if y.ndim == 2 and y.shape[0] == design.shape[0]:
        return np.repeat(design, y.shape[1], axis=0), y.reshape(-1).astype(np.int64)
    raise ValueError("targets must be rows or rows-by-branches")


def fit_capacity_matched_ridge(
    history: NDArray[np.floating],
    targets: NDArray[np.integer],
    pipeline: str,
) -> RidgeBundle:
    x = _as_history(history)
    transformer = SplineInteractionTransformer()
    components = transformer.fit_transform(x)
    design = np.column_stack((components, x))
    if pipeline == "codex":
        final_scaler: StandardScaler | None = StandardScaler().fit(design)
        fitted = final_scaler.transform(design)
        max_iter = 2_000
    elif pipeline == "fable":
        final_scaler = None
        fitted = design
        max_iter = 5_000
    else:
        raise ValueError(f"unknown pipeline: {pipeline}")
    fitted, y = expand_targets(fitted, targets)
    classifier = LogisticRegression(
        C=RIDGE_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=0,
    ).fit(fitted, y)
    return RidgeBundle(transformer, final_scaler, classifier, pipeline)


def predict_capacity_matched_ridge(
    bundle: RidgeBundle, history: NDArray[np.floating]
) -> NDArray[np.float64]:
    x = _as_history(history)
    components = bundle.transformer.transform(x)
    design = np.column_stack((components, x))
    if bundle.final_scaler is not None:
        design = bundle.final_scaler.transform(design)
    return np.clip(bundle.classifier.predict_proba(design)[:, 1], EPSILON, 1.0 - EPSILON)


def tree_min_samples(targets: NDArray[np.integer]) -> int:
    y = np.asarray(targets)
    branches = int(y.shape[1]) if y.ndim == 2 else 1
    return max(100, 5 * branches)


def fit_boosted_history(
    history: NDArray[np.floating],
    targets: NDArray[np.integer],
    max_leaf_nodes: int,
    seed: int,
) -> HistGradientBoostingClassifier:
    x = _as_history(history)
    expanded_x, y = expand_targets(x, targets)
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=TREE_LEARNING_RATE,
        max_iter=TREE_ITERATIONS,
        max_leaf_nodes=int(max_leaf_nodes),
        min_samples_leaf=tree_min_samples(targets),
        l2_regularization=TREE_L2,
        early_stopping=False,
        random_state=int(seed),
    ).fit(expanded_x, y)
    return classifier


def predict_boosted_history(
    classifier: HistGradientBoostingClassifier,
    history: NDArray[np.floating],
) -> NDArray[np.float64]:
    x = _as_history(history)
    return np.clip(classifier.predict_proba(x)[:, 1], EPSILON, 1.0 - EPSILON)


def state_log_loss(
    probability: NDArray[np.floating], targets: NDArray[np.integer]
) -> NDArray[np.float64]:
    p = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim == 1:
        if y.shape[0] != p.size:
            raise ValueError("target length mismatch")
        return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    if y.ndim == 2 and y.shape[0] == p.size:
        return -(y * np.log(p[:, None]) + (1.0 - y) * np.log(1.0 - p[:, None])).mean(axis=1)
    raise ValueError("targets must be rows or rows-by-branches")


def grouped_development_cv(
    history: NDArray[np.floating],
    targets: NDArray[np.integer],
    matrix_ids: NDArray[np.integer],
    pipeline: str,
    cohort: str,
    candidate: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x = _as_history(history)
    y = np.asarray(targets)
    groups = np.asarray(matrix_ids)
    if groups.shape != (x.shape[0],):
        raise ValueError("matrix group length mismatch")
    if np.unique(groups).size < CV_FOLDS:
        raise ValueError("too few matrices for grouped cross-validation")
    splitter = GroupKFold(n_splits=CV_FOLDS)
    records: list[dict[str, Any]] = []
    spline_losses: list[tuple[int, float]] = []
    tree_losses: dict[int, list[tuple[int, float]]] = {leaf: [] for leaf in TREE_LEAF_GRID}

    for fold, (train, validation) in enumerate(splitter.split(x, groups=groups), start=1):
        ridge = fit_capacity_matched_ridge(x[train], y[train], pipeline)
        ridge_probability = predict_capacity_matched_ridge(ridge, x[validation])
        ridge_loss = float(state_log_loss(ridge_probability, y[validation]).mean())
        spline_losses.append((len(validation), ridge_loss))
        records.append(
            {
                "cohort": cohort,
                "candidate": candidate,
                "family": "spline_interaction_pca12_ridge",
                "tree_leaves": "",
                "fold": fold,
                "validation_matrices": int(np.unique(groups[validation]).size),
                "validation_rows": int(len(validation)),
                "log_loss_nats": ridge_loss,
            }
        )
        for leaves in TREE_LEAF_GRID:
            tree = fit_boosted_history(
                x[train],
                y[train],
                leaves,
                derived_seed(cohort, candidate, "cv", fold, leaves),
            )
            probability = predict_boosted_history(tree, x[validation])
            loss = float(state_log_loss(probability, y[validation]).mean())
            tree_losses[leaves].append((len(validation), loss))
            records.append(
                {
                    "cohort": cohort,
                    "candidate": candidate,
                    "family": "gradient_boosted_history",
                    "tree_leaves": leaves,
                    "fold": fold,
                    "validation_matrices": int(np.unique(groups[validation]).size),
                    "validation_rows": int(len(validation)),
                    "log_loss_nats": loss,
                }
            )

    def weighted(values: Iterable[tuple[int, float]]) -> float:
        pairs = list(values)
        return float(sum(size * loss for size, loss in pairs) / sum(size for size, _ in pairs))

    spline_score = weighted(spline_losses)
    tree_scores = {leaves: weighted(losses) for leaves, losses in tree_losses.items()}
    selected_leaves = min(TREE_LEAF_GRID, key=lambda leaves: (tree_scores[leaves], leaves))
    selected_tree_score = tree_scores[selected_leaves]
    selected_family = (
        "spline_interaction_pca12_ridge"
        if spline_score <= selected_tree_score
        else "gradient_boosted_history"
    )
    summary = {
        "cohort": cohort,
        "candidate": candidate,
        "spline_cv_log_loss": spline_score,
        "selected_tree_leaves": selected_leaves,
        "selected_tree_cv_log_loss": selected_tree_score,
        "tree_cv_log_loss_by_leaves": {str(k): v for k, v in tree_scores.items()},
        "selected_family": selected_family,
        "selected_cv_log_loss": min(spline_score, selected_tree_score),
    }
    return records, summary


def matrix_means(
    values: NDArray[np.floating], matrix_ids: NDArray[np.integer]
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    matrices = np.asarray(matrix_ids)
    return np.asarray([array[matrices == matrix].mean() for matrix in np.unique(matrices)])


def bootstrap_interval(
    values: NDArray[np.floating],
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
    values: NDArray[np.floating],
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

