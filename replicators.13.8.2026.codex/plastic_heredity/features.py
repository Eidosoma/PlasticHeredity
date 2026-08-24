from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import GardConfig
from .simulator import FloatMatrix, Snapshot

FloatVector = NDArray[np.float64]

NODE_PROFILE_NAMES = (
    "count_over_nmax",
    "composition_fraction",
    "present",
    "log_in_catalysis",
    "log_out_catalysis",
    "log_join_propensity",
    "log_leave_propensity",
    "row_log_beta_mean",
    "row_log_beta_sd",
    "column_log_beta_mean",
    "column_log_beta_sd",
    "log_row_beta_mean",
    "log_column_beta_mean",
    "log_active_in_catalysis",
    "log_active_out_catalysis",
)

SUMMARY_NAMES = (
    "mean",
    "sd",
    "minimum",
    "q05",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
    "q95",
    "maximum",
    "composition_weighted_mean",
    "active_mean",
)

STATE_GRAPH_FEATURE_NAMES = tuple(
    f"{profile}__{summary}"
    for profile in NODE_PROFILE_NAMES
    for summary in SUMMARY_NAMES
)

HISTORY_FEATURE_NAMES = (
    "normalized_generation",
    "normalized_current_mass",
    "prefix_inheritance_fraction",
    "recent5_inheritance_fraction",
    "normalized_trailing_inheritance_run",
    "latest_parent_daughter_h",
    "normalized_fissions_since_break",
    "current_inheritance_state",
    "normalized_current_regime_duration",
)


def _safe_log1p(values: NDArray) -> FloatVector:
    return np.log1p(np.maximum(np.asarray(values, dtype=np.float64), 0.0))


def _node_profiles(
    composition: NDArray,
    beta: FloatMatrix,
    config: GardConfig,
) -> NDArray[np.float64]:
    x = np.asarray(composition, dtype=np.float64)
    mass = float(x.sum())
    if mass <= 0.0:
        raise ValueError("features are undefined for an empty assembly")
    fraction = x / mass
    present = (x > 0.0).astype(np.float64)
    active_count = max(float(present.sum()), 1.0)
    incoming = beta @ fraction
    outgoing = beta.T @ fraction
    catalytic_boost = 1.0 + incoming
    join = config.k_join * (1.0 / config.n_types) * mass * catalytic_boost
    leave = config.k_leave * x * catalytic_boost
    log_beta = np.log(np.maximum(beta, np.finfo(np.float64).tiny))
    active_fraction = present / active_count

    return np.column_stack(
        (
            x / config.n_max,
            fraction,
            present,
            _safe_log1p(incoming),
            _safe_log1p(outgoing),
            _safe_log1p(join),
            _safe_log1p(leave),
            log_beta.mean(axis=1),
            log_beta.std(axis=1),
            log_beta.mean(axis=0),
            log_beta.std(axis=0),
            _safe_log1p(beta.mean(axis=1)),
            _safe_log1p(beta.mean(axis=0)),
            _safe_log1p(beta @ active_fraction),
            _safe_log1p(beta.T @ active_fraction),
        )
    )


def _summarize_profile(
    values: FloatVector, composition_weights: FloatVector, active: NDArray
) -> FloatVector:
    quantiles = np.quantile(values, (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95))
    active_values = values[np.asarray(active, dtype=bool)]
    active_mean = float(active_values.mean()) if active_values.size else 0.0
    weighted_mean = float(np.dot(values, composition_weights))
    return np.asarray(
        (
            values.mean(),
            values.std(),
            values.min(),
            quantiles[0],
            quantiles[1],
            quantiles[2],
            quantiles[3],
            quantiles[4],
            quantiles[5],
            quantiles[6],
            values.max(),
            weighted_mean,
            active_mean,
        ),
        dtype=np.float64,
    )


def state_graph_features(
    composition: NDArray, beta: FloatMatrix, config: GardConfig
) -> FloatVector:
    """Return 15 node profiles x 13 symmetric summaries = 195 features.

    Every operation is equivariant before aggregation and symmetric afterwards,
    so a simultaneous relabeling of molecule counts and beta rows/columns leaves
    this vector unchanged.
    """

    x = np.asarray(composition, dtype=np.float64)
    fraction = x / x.sum()
    active = x > 0.0
    profiles = _node_profiles(x, beta, config)
    features = np.concatenate(
        [
            _summarize_profile(profiles[:, column], fraction, active)
            for column in range(profiles.shape[1])
        ]
    )
    if features.shape != (195,):
        raise AssertionError(f"expected 195 features, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("state/graph features contain non-finite values")
    return features


def beta_only_features(beta: FloatMatrix, config: GardConfig) -> FloatVector:
    """Apply the same invariant map to a uniform, state-free pseudo-composition."""

    uniform = np.ones(config.n_types, dtype=np.float64)
    return state_graph_features(uniform, beta, config)


def _terminal_run(values: tuple[bool, ...], target: bool | None = None) -> int:
    if not values:
        return 0
    terminal = values[-1] if target is None else target
    count = 0
    for value in reversed(values):
        if value != terminal:
            break
        count += 1
    return count


def history_features(snapshot: Snapshot, config: GardConfig) -> FloatVector:
    inheritance = snapshot.inheritance
    boundary_h = snapshot.boundary_h
    n = len(inheritance)
    prefix_fraction = float(np.mean(inheritance)) if n else 0.0
    recent = inheritance[-5:]
    recent_fraction = float(np.mean(recent)) if recent else 0.0
    trailing = _terminal_run(inheritance, True)
    latest_h = float(boundary_h[-1]) if boundary_h else 0.0
    break_locations = [index for index, value in enumerate(inheritance) if not value]
    since_break = n - 1 - break_locations[-1] if break_locations else n
    current_state = float(inheritance[-1]) if inheritance else 0.0
    regime_duration = _terminal_run(inheritance)
    scale = max(config.generations, 1)
    return np.asarray(
        (
            snapshot.generation / scale,
            float(snapshot.composition.sum()) / config.n_max,
            prefix_fraction,
            recent_fraction,
            trailing / scale,
            latest_h,
            since_break / scale,
            current_state,
            regime_duration / scale,
        ),
        dtype=np.float64,
    )

