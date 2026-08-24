"""Pure models and statistics for the sequence-history response.

This module has no knowledge of confirmation artifact locations.  In
particular, fitting consumes only development replay arrays supplied by the
runner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


THRESHOLD = 0.90
HORIZON = 12
RENEWAL_RUN = 3
DURATION_CAP = 5
LAG_GRID = (5, 10, 20, 40, 100)
C_GRID = (0.01, 0.1, 1.0, 10.0)
CV_FOLDS = 5
BOOTSTRAP_REPETITIONS = 4096
RANDOMIZATION_REPETITIONS = 4096
EPSILON = 1e-7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def event_from_flags(flags: Sequence[bool], run_length: int = RENEWAL_RUN) -> bool:
    """Return break-then-renewal under the strict prospective endpoint."""

    values = np.asarray(tuple(flags), dtype=bool)
    breaks = np.flatnonzero(~values)
    if breaks.size == 0:
        return False
    run = 0
    for value in values[int(breaks[0]) + 1 :]:
        run = run + 1 if bool(value) else 0
        if run >= run_length:
            return True
    return False


def terminal_duration(flags: Sequence[bool], cap: int = DURATION_CAP) -> int:
    values = np.asarray(tuple(flags), dtype=bool)
    if values.size == 0:
        raise ValueError("a launch history must contain at least one boundary")
    last = bool(values[-1])
    length = 1
    for value in values[-2::-1]:
        if bool(value) != last:
            break
        length += 1
    return min(length, cap)


@dataclass(frozen=True)
class TransitionModel:
    """Development-fitted H/break/terminal transition probabilities."""

    duration_aware: bool
    probabilities: NDArray[np.float64]
    counts: NDArray[np.int64]
    alpha: float = 1.0

    def __post_init__(self) -> None:
        expected = (2, DURATION_CAP, 3) if self.duration_aware else (2, 3)
        if self.probabilities.shape != expected or self.counts.shape != expected:
            raise ValueError(f"transition arrays must have shape {expected}")
        if not np.allclose(self.probabilities.sum(axis=-1), 1.0):
            raise ValueError("transition probabilities must sum to one")

    def row(self, previous: bool, duration: int) -> NDArray[np.float64]:
        state = int(bool(previous))
        if self.duration_aware:
            return self.probabilities[state, min(max(duration, 1), 5) - 1]
        return self.probabilities[state]

    def to_json(self) -> dict[str, Any]:
        return {
            "duration_aware": self.duration_aware,
            "alpha": self.alpha,
            "outcome_order": ["break", "inherit", "terminal"],
            "probabilities": self.probabilities.tolist(),
            "counts": self.counts.tolist(),
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "TransitionModel":
        return cls(
            duration_aware=bool(value["duration_aware"]),
            alpha=float(value["alpha"]),
            probabilities=np.asarray(value["probabilities"], dtype=np.float64),
            counts=np.asarray(value["counts"], dtype=np.int64),
        )


def fit_transition_model(
    histories: NDArray[np.float64],
    lengths: NDArray[np.int64],
    died: NDArray[np.bool_],
    *,
    duration_aware: bool,
    threshold: float = THRESHOLD,
    alpha: float = 1.0,
) -> TransitionModel:
    """Fit a candidate-specific transition law on natural development paths.

    Outcomes use order break/inherit/terminal.  Complete 100-fission paths are
    right-censored and do not contribute an artificial terminal transition.
    """

    histories = np.asarray(histories, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    died = np.asarray(died, dtype=bool)
    if histories.ndim != 2 or lengths.shape != (histories.shape[0],):
        raise ValueError("invalid trajectory history shapes")
    shape = (2, DURATION_CAP, 3) if duration_aware else (2, 3)
    counts = np.zeros(shape, dtype=np.int64)
    for row, length, terminated in zip(histories, lengths, died, strict=True):
        flags = np.asarray(row[: int(length)] > threshold, dtype=bool)
        if flags.size == 0:
            continue
        run = 1
        for destination_index in range(1, flags.size):
            previous = int(flags[destination_index - 1])
            destination = int(flags[destination_index])
            if duration_aware:
                counts[previous, min(run, DURATION_CAP) - 1, destination] += 1
            else:
                counts[previous, destination] += 1
            if flags[destination_index] == flags[destination_index - 1]:
                run += 1
            else:
                run = 1
        if bool(terminated):
            previous = int(flags[-1])
            if duration_aware:
                counts[previous, min(run, DURATION_CAP) - 1, 2] += 1
            else:
                counts[previous, 2] += 1
    probabilities = (counts + alpha) / (
        counts.sum(axis=-1, keepdims=True) + 3.0 * alpha
    )
    return TransitionModel(duration_aware, probabilities, counts, alpha)


def transition_event_probability(
    model: TransitionModel,
    prefix_h: Sequence[float],
    *,
    threshold: float = THRESHOLD,
    horizon: int = HORIZON,
    run_length: int = RENEWAL_RUN,
) -> float:
    """Integrate a fitted transition law into a launch-time F12 probability."""

    prefix = np.asarray(tuple(prefix_h), dtype=np.float64)
    if prefix.size == 0:
        raise ValueError("launch prefix cannot be empty")
    flags = prefix > threshold
    # state: last symbol, capped duration, future break seen, renewal run
    states: dict[tuple[bool, int, bool, int], float] = {
        (bool(flags[-1]), terminal_duration(flags), False, 0): 1.0
    }
    success = 0.0
    for _ in range(horizon):
        next_states: dict[tuple[bool, int, bool, int], float] = {}
        for (last, duration, break_seen, renewal), mass in states.items():
            p_break, p_inherit, _p_terminal = model.row(last, duration)
            # Terminal probability is absorbing failure and is intentionally
            # omitted from next_states.
            next_duration = min(duration + 1, DURATION_CAP) if not last else 1
            key_break = (False, next_duration, True, 0)
            next_states[key_break] = next_states.get(key_break, 0.0) + mass * p_break

            inherited_duration = min(duration + 1, DURATION_CAP) if last else 1
            inherited_run = renewal + 1 if break_seen else 0
            inherited_mass = mass * p_inherit
            if break_seen and inherited_run >= run_length:
                success += inherited_mass
            else:
                key_inherit = (True, inherited_duration, break_seen, inherited_run)
                next_states[key_inherit] = (
                    next_states.get(key_inherit, 0.0) + inherited_mass
                )
        states = next_states
    return float(np.clip(success, 0.0, 1.0))


def transition_predictions(
    model: TransitionModel,
    histories: NDArray[np.float64],
    lengths: NDArray[np.int64],
) -> NDArray[np.float64]:
    return np.asarray(
        [
            transition_event_probability(model, row[: int(length)])
            for row, length in zip(histories, lengths, strict=True)
        ],
        dtype=np.float64,
    )


def lagged_history_matrix(
    direct: NDArray[np.float64],
    histories: NDArray[np.float64],
    lengths: NDArray[np.int64],
    lag: int,
    *,
    direct_columns: Sequence[int] | None = None,
    threshold: float = THRESHOLD,
) -> NDArray[np.float64]:
    """Append ordered, right-aligned H/flag/mask history to direct features."""

    direct = np.asarray(direct, dtype=np.float64)
    histories = np.asarray(histories, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if direct_columns is not None:
        direct = direct[:, np.asarray(tuple(direct_columns), dtype=np.int64)]
    if lag <= 0 or histories.ndim != 2 or histories.shape[0] != direct.shape[0]:
        raise ValueError("invalid lag-history inputs")
    h_lags = np.zeros((direct.shape[0], lag), dtype=np.float64)
    inherited = np.zeros_like(h_lags)
    observed = np.zeros_like(h_lags)
    for index, (row, length) in enumerate(zip(histories, lengths, strict=True)):
        take = min(int(length), lag)
        if take == 0:
            continue
        values = row[int(length) - take : int(length)]
        h_lags[index, -take:] = values
        inherited[index, -take:] = values > threshold
        observed[index, -take:] = 1.0
    return np.column_stack((direct, h_lags, inherited, observed))


def _binomial_training_rows(
    features: NDArray[np.float64], targets: NDArray[np.int8]
) -> tuple[NDArray[np.float64], NDArray[np.int8], NDArray[np.float64]]:
    """Compress repeated Bernoulli branches into weighted 0/1 rows."""

    targets = np.asarray(targets, dtype=np.int8)
    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.shape[0] != features.shape[0]:
        raise ValueError("target states do not match features")
    positives = targets.sum(axis=1).astype(np.float64)
    negatives = (targets.shape[1] - positives).astype(np.float64)
    rows = np.repeat(features, 2, axis=0)
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), features.shape[0])
    weights = np.column_stack((negatives, positives)).reshape(-1)
    keep = weights > 0
    return rows[keep], labels[keep], weights[keep]


def state_branch_log_loss(
    targets: NDArray[np.int8], predictions: NDArray[np.float64]
) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[:, None]
    p = np.clip(np.asarray(predictions, dtype=np.float64), EPSILON, 1 - EPSILON)
    if p.shape != (targets.shape[0],):
        raise ValueError("one prediction is required per launch state")
    losses = -(targets * np.log(p[:, None]) + (1 - targets) * np.log1p(-p[:, None]))
    return float(losses.mean())


def _fit_logistic(
    train_features: NDArray[np.float64],
    train_targets: NDArray[np.int8],
    c_value: float,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(train_features)
    weighted_x, weighted_y, weights = _binomial_training_rows(
        scaler.transform(train_features), train_targets
    )
    classifier = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="lbfgs",
        max_iter=5_000,
        random_state=0,
    ).fit(weighted_x, weighted_y, sample_weight=weights)
    return scaler, classifier


@dataclass(frozen=True)
class LaggedRidgeModel:
    lag: int
    c_value: float
    direct_columns: tuple[int, ...]
    scaler_mean: NDArray[np.float64]
    scaler_scale: NDArray[np.float64]
    coefficient: NDArray[np.float64]
    intercept: float

    def predict_features(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        z = (features - self.scaler_mean) / self.scaler_scale
        logits = z @ self.coefficient + self.intercept
        return np.clip(1.0 / (1.0 + np.exp(-np.clip(logits, -709, 709))), EPSILON, 1 - EPSILON)

    def predict(
        self,
        direct: NDArray[np.float64],
        histories: NDArray[np.float64],
        lengths: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        return self.predict_features(
            lagged_history_matrix(
                direct,
                histories,
                lengths,
                self.lag,
                direct_columns=self.direct_columns,
            )
        )

    def arrays(self) -> dict[str, NDArray[Any]]:
        return {
            "lag": np.asarray([self.lag], dtype=np.int16),
            "c_value": np.asarray([self.c_value], dtype=np.float64),
            "direct_columns": np.asarray(self.direct_columns, dtype=np.int16),
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "coefficient": self.coefficient,
            "intercept": np.asarray([self.intercept], dtype=np.float64),
        }

    @classmethod
    def from_archive(cls, archive: Any) -> "LaggedRidgeModel":
        return cls(
            lag=int(archive["lag"][0]),
            c_value=float(archive["c_value"][0]),
            direct_columns=tuple(int(v) for v in archive["direct_columns"]),
            scaler_mean=np.asarray(archive["scaler_mean"], dtype=np.float64),
            scaler_scale=np.asarray(archive["scaler_scale"], dtype=np.float64),
            coefficient=np.asarray(archive["coefficient"], dtype=np.float64),
            intercept=float(archive["intercept"][0]),
        )


def fit_lagged_ridge(
    direct: NDArray[np.float64],
    histories: NDArray[np.float64],
    lengths: NDArray[np.int64],
    targets: NDArray[np.int8],
    matrix_ids: NDArray[np.int64],
    *,
    direct_columns: Sequence[int],
    lag_grid: Sequence[int] = LAG_GRID,
    c_grid: Sequence[float] = C_GRID,
    folds: int = CV_FOLDS,
) -> tuple[LaggedRidgeModel, list[dict[str, Any]]]:
    """Development-only matrix-grouped model selection and final refit."""

    direct = np.asarray(direct, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int8)
    matrix_ids = np.asarray(matrix_ids, dtype=np.int64)
    unique_groups = np.unique(matrix_ids)
    if unique_groups.size < folds:
        raise ValueError("fewer development matrices than CV folds")
    splitter = GroupKFold(n_splits=folds)
    audit: list[dict[str, Any]] = []
    feature_cache = {
        int(lag): lagged_history_matrix(
            direct,
            histories,
            lengths,
            int(lag),
            direct_columns=direct_columns,
        )
        for lag in lag_grid
    }
    for lag in sorted(feature_cache):
        features = feature_cache[lag]
        splits = list(splitter.split(features, groups=matrix_ids))
        for c_value in sorted(float(c) for c in c_grid):
            fold_losses: list[float] = []
            for train_index, test_index in splits:
                scaler, classifier = _fit_logistic(
                    features[train_index], targets[train_index], c_value
                )
                probability = classifier.predict_proba(
                    scaler.transform(features[test_index])
                )[:, 1]
                fold_losses.append(state_branch_log_loss(targets[test_index], probability))
            audit.append(
                {
                    "lag": int(lag),
                    "c_value": float(c_value),
                    "fold_losses": fold_losses,
                    "mean_log_loss": float(np.mean(fold_losses)),
                }
            )
    # Exact deterministic rule: loss, then smaller lag, then smaller C.
    selected = min(audit, key=lambda row: (row["mean_log_loss"], row["lag"], row["c_value"]))
    final_features = feature_cache[int(selected["lag"])]
    scaler, classifier = _fit_logistic(
        final_features, targets, float(selected["c_value"])
    )
    model = LaggedRidgeModel(
        lag=int(selected["lag"]),
        c_value=float(selected["c_value"]),
        direct_columns=tuple(int(v) for v in direct_columns),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        coefficient=np.asarray(classifier.coef_[0], dtype=np.float64),
        intercept=float(classifier.intercept_[0]),
    )
    return model, audit


def branch_losses(
    targets: NDArray[np.int8], predictions: NDArray[np.float64]
) -> NDArray[np.float64]:
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    p = np.clip(np.asarray(predictions, dtype=np.float64), EPSILON, 1 - EPSILON)
    return -(y * np.log(p[:, None]) + (1.0 - y) * np.log1p(-p[:, None]))


def brier_score(targets: NDArray[np.int8], predictions: NDArray[np.float64]) -> float:
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    p = np.asarray(predictions, dtype=np.float64)
    return float(np.mean((y - p[:, None]) ** 2))


def centered(values: NDArray[np.float64], groups: NDArray[np.int64]) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    _, inverse = np.unique(groups, return_inverse=True)
    means = np.bincount(inverse, weights=values) / np.bincount(inverse)
    return values - means[inverse]


def rank_metrics(
    predictions: NDArray[np.float64],
    targets: NDArray[np.int8],
    matrix_ids: NDArray[np.int64],
) -> tuple[float | None, float | None]:
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    q = y.mean(axis=1)
    overall = spearmanr(predictions, q).correlation
    within = spearmanr(centered(predictions, matrix_ids), centered(q, matrix_ids)).correlation
    return (
        None if not np.isfinite(overall) else float(overall),
        None if not np.isfinite(within) else float(within),
    )


def paired_matrix_inference(
    targets: NDArray[np.int8],
    baseline: NDArray[np.float64],
    challenger: NDArray[np.float64],
    matrix_ids: NDArray[np.int64],
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = 0,
) -> dict[str, Any]:
    """Matrix-clustered gain inference; positive favors challenger."""

    delta = branch_losses(targets, baseline) - branch_losses(targets, challenger)
    matrices = np.unique(matrix_ids)
    sums = np.asarray([delta[matrix_ids == m].sum() for m in matrices])
    counts = np.asarray([delta[matrix_ids == m].size for m in matrices], dtype=np.int64)
    observed = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, matrices.size, size=(repetitions, matrices.size))
    bootstrap = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(repetitions, matrices.size))
    randomized = (signs * sums).sum(axis=1) / counts.sum()
    p_value = float((1 + np.count_nonzero(randomized >= observed)) / (repetitions + 1))
    return {
        "gain_nats": observed,
        "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "randomization_p_one_sided": p_value,
        "matrices": int(matrices.size),
        "branches": int(counts.sum()),
        "repetitions": int(repetitions),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(tuple(p_values), dtype=np.float64)
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate(
        np.asarray([(values[index] * (values.size - rank)) for rank, index in enumerate(order)])
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return [float(value) for value in adjusted]

