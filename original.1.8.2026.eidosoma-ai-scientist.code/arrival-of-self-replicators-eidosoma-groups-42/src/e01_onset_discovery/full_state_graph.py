"""Permutation-invariant full-state catalytic-graph features for S19-L34."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from e01_latent_timebase.core import N_MAX, rates

PRIMARY_VIEW = "BETA_CONDITIONED_FULL_STATE_GRAPH"
ORACLE_VIEW = "TARGET_CONDITIONED_FULL_STATE_GRAPH"
VIEWS = (PRIMARY_VIEW, ORACLE_VIEW)
QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
KRYLOV_DEPTH = 4


def _entropy(values: NDArray[np.float64]) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[positive > 0]
    if not len(positive):
        return 0.0
    probability = positive / positive.sum()
    return float(-np.sum(probability * np.log(probability)) / math.log(max(2, len(values))))


def _cosine(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        raise ValueError("cosine requires nonzero vectors")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _safe_correlation(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    left_sd = float(np.std(left))
    right_sd = float(np.std(right))
    if left_sd <= 1e-15 or right_sd <= 1e-15:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _quantile_names(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}__q{int(value * 100):03d}" for value in QUANTILES)


NODE_CHANNELS = (
    "composition",
    "log_count",
    "presence",
    "log_boost",
    "join_share",
    "loss_share",
    "log_beta_row_sum",
    "log_beta_column_sum",
)
PAIR_CHANNELS = (
    ("composition", "log_boost"),
    ("composition", "join_share"),
    ("composition", "loss_share"),
    ("composition", "log_beta_row_sum"),
    ("composition", "log_beta_column_sum"),
    ("log_boost", "join_share"),
    ("log_boost", "loss_share"),
    ("log_beta_row_sum", "log_beta_column_sum"),
)
OPERATORS = ("RAW_FROBENIUS", "ROW_STOCHASTIC", "COLUMN_STOCHASTIC")
SEEDS = ("COMPOSITION", "ACTIVE")
MESSAGE_STATS = ("mean", "sd", "maximum", "composition_projection", "entropy")
PHASE_NAMES = (
    "mass_fraction",
    "distance_to_fission_fraction",
    "generation_local_step_fraction",
    "post_fission_indicator",
    "completed_fissions_fraction",
    "batch_step_fraction",
    "landmark_fraction",
)


def feature_names() -> dict[str, tuple[str, ...]]:
    primary: list[str] = []
    for channel in NODE_CHANNELS:
        primary.extend(_quantile_names(channel))
    primary.extend(f"correlation__{left}__{right}" for left, right in PAIR_CHANNELS)
    primary.extend(
        (
            "beta_log_mean",
            "beta_log_sd",
            "beta_log_maximum",
            "beta_log_frobenius_per_edge",
            "beta_raw_frobenius_log",
        )
    )
    primary.extend(_quantile_names("beta_log_edges"))
    primary.extend(f"beta_singular_share_{index:02d}" for index in range(1, 9))
    primary.extend(("beta_singular_entropy", "beta_singular_effective_rank_fraction"))
    for operator in OPERATORS:
        for seed in SEEDS:
            for depth in range(1, KRYLOV_DEPTH + 1):
                primary.extend(
                    f"message__{operator}__{seed}__k{depth}__{stat}"
                    for stat in MESSAGE_STATS
                )
    primary.extend(PHASE_NAMES)
    oracle = list(primary)
    oracle.extend(
        (
            "target_score",
            "target_support_overlap",
            "target_entropy",
            "target_component_fraction",
        )
    )
    oracle.extend(_quantile_names("target_composition"))
    for operator in OPERATORS:
        for depth in range(1, KRYLOV_DEPTH + 1):
            oracle.extend(
                (
                    f"target_cross__{operator}__k{depth}__target_to_current",
                    f"target_cross__{operator}__k{depth}__current_to_target",
                )
            )
    return {PRIMARY_VIEW: tuple(primary), ORACLE_VIEW: tuple(oracle)}


def _message_statistics(
    operator: NDArray[np.float64],
    seed: NDArray[np.float64],
    composition: NDArray[np.float64],
) -> list[float]:
    values = np.asarray(seed, dtype=np.float64)
    rows: list[float] = []
    for _ in range(KRYLOV_DEPTH):
        values = operator @ values
        rows.extend(
            (
                float(np.mean(values)),
                float(np.std(values, ddof=0)),
                float(np.max(values)),
                float(np.dot(composition, values)),
                _entropy(np.abs(values)),
            )
        )
    return rows


def graph_views(
    state: NDArray[np.integer],
    beta: NDArray[np.floating],
    target: NDArray[np.floating],
    *,
    generation_local_step: int,
    observation_kind: str,
    completed_fissions: int,
    batch_step: int,
    landmark: int,
    target_component_fraction: float,
) -> dict[str, NDArray[np.float64]]:
    """Return fixed target-blind and target-conditioned graph signatures."""

    counts = np.asarray(state, dtype=np.int64)
    matrix = np.asarray(beta, dtype=np.float64)
    centroid = np.asarray(target, dtype=np.float64)
    if counts.shape != (100,) or matrix.shape != (100, 100) or centroid.shape != (100,):
        raise ValueError("full-state graph input shape changed")
    if counts.sum() <= 0 or np.any(counts < 0) or np.any(matrix <= 0):
        raise ValueError("invalid GARD state or catalytic matrix")
    if np.any(centroid < 0) or centroid.sum() <= 0:
        raise ValueError("invalid target centroid")
    centroid = centroid / centroid.sum()
    mass = int(counts.sum())
    composition = counts.astype(np.float64) / mass
    active = (counts > 0).astype(np.float64)
    active /= max(1.0, active.sum())
    joins, losses = rates(counts, matrix)
    boost = 1.0 + matrix @ composition
    node_values = {
        "composition": composition,
        "log_count": np.log1p(counts.astype(np.float64)) / math.log1p(N_MAX),
        "presence": (counts > 0).astype(np.float64),
        "log_boost": np.log1p(boost),
        "join_share": joins / joins.sum(),
        "loss_share": losses / max(float(losses.sum()), 1e-300),
        "log_beta_row_sum": np.log1p(matrix.sum(axis=1)),
        "log_beta_column_sum": np.log1p(matrix.sum(axis=0)),
    }
    primary: list[float] = []
    for channel in NODE_CHANNELS:
        primary.extend(np.quantile(node_values[channel], QUANTILES).tolist())
    primary.extend(
        _safe_correlation(node_values[left], node_values[right])
        for left, right in PAIR_CHANNELS
    )
    log_beta = np.log1p(matrix)
    primary.extend(
        (
            float(np.mean(log_beta)),
            float(np.std(log_beta, ddof=0)),
            float(np.max(log_beta)),
            float(np.linalg.norm(log_beta) / log_beta.size),
            float(np.log1p(np.linalg.norm(matrix))),
        )
    )
    primary.extend(np.quantile(log_beta, QUANTILES).tolist())
    singular = np.linalg.svd(log_beta, compute_uv=False)
    singular_share = singular / singular.sum()
    primary.extend(singular_share[:8].tolist())
    singular_entropy = _entropy(singular_share)
    primary.extend((singular_entropy, float(np.exp(singular_entropy * math.log(100)) / 100)))
    frobenius = float(np.linalg.norm(matrix))
    raw_operator = matrix / frobenius
    row_operator = matrix / matrix.sum(axis=1, keepdims=True)
    column_operator = matrix.T / matrix.sum(axis=0, keepdims=True).T
    operators = {
        "RAW_FROBENIUS": raw_operator,
        "ROW_STOCHASTIC": row_operator,
        "COLUMN_STOCHASTIC": column_operator,
    }
    seeds = {"COMPOSITION": composition, "ACTIVE": active}
    for operator in OPERATORS:
        for seed in SEEDS:
            primary.extend(
                _message_statistics(operators[operator], seeds[seed], composition)
            )
    primary.extend(
        (
            mass / N_MAX,
            (N_MAX - mass) / N_MAX,
            int(generation_local_step) / 1000.0,
            float(observation_kind == "post_fission"),
            int(completed_fissions) / 100.0,
            int(batch_step) / 192.0,
            int(landmark) / 192.0,
        )
    )
    oracle = list(primary)
    oracle.extend(
        (
            _cosine(composition, centroid),
            float(np.mean((counts > 0) & (centroid > 0))),
            _entropy(centroid),
            float(target_component_fraction),
        )
    )
    oracle.extend(np.quantile(centroid, QUANTILES).tolist())
    for operator in OPERATORS:
        current_forward = composition.copy()
        target_forward = centroid.copy()
        for _ in range(KRYLOV_DEPTH):
            current_forward = operators[operator] @ current_forward
            target_forward = operators[operator] @ target_forward
            oracle.extend(
                (
                    float(np.dot(centroid, current_forward)),
                    float(np.dot(composition, target_forward)),
                )
            )
    names = feature_names()
    result = {
        PRIMARY_VIEW: np.asarray(primary, dtype=np.float64),
        ORACLE_VIEW: np.asarray(oracle, dtype=np.float64),
    }
    if any(
        value.shape != (len(names[key]),) or not np.isfinite(value).all()
        for key, value in result.items()
    ):
        raise RuntimeError("full-state graph feature schema or finiteness failure")
    return result
