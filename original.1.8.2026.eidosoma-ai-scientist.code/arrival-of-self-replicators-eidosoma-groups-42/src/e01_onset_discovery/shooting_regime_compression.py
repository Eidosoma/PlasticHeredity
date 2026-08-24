"""Cross-fitted regime-hazard compression helpers for shooting ensembles."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


def hazard_fit_scope(
    target_state_id: str,
    matrix_state_ids: Sequence[str],
    model_id: str,
) -> tuple[str, ...]:
    """Return the prospectively allowed state identities for one hazard fit."""

    identities = tuple(str(value) for value in matrix_state_ids)
    if len(identities) != len(set(identities)):
        raise ValueError("matrix_state_ids must be unique")
    if target_state_id not in identities:
        raise ValueError("target state is absent from its matrix scope")
    if model_id == "MATRIX_OTHER_LANDMARK_SEMIMARKOV":
        output = tuple(value for value in identities if value != target_state_id)
        if not output:
            raise ValueError("matrix-transfer scope requires another landmark")
        return output
    if model_id == "STATE_LOCAL_SEMIMARKOV":
        return (target_state_id,)
    raise ValueError(f"unsupported model: {model_id}")


def fit_shrunk_duration_table(
    current_states: Iterable[bool],
    durations: Iterable[int],
    next_states: Iterable[bool],
    anchor: NDArray[np.float64],
    *,
    prior_strength: float,
) -> NDArray[np.float64]:
    """Fit a duration table shrunk to a fixed cell-specific anchor."""

    current = np.asarray(tuple(current_states), dtype=np.bool_)
    dwell = np.asarray(tuple(durations), dtype=np.int64)
    following = np.asarray(tuple(next_states), dtype=np.bool_)
    prior = np.asarray(anchor, dtype=np.float64)
    if current.shape != dwell.shape or current.shape != following.shape:
        raise ValueError("transition arrays must align")
    if prior.ndim != 2 or prior.shape[0] != 2:
        raise ValueError("anchor must have two state rows")
    if prior_strength <= 0 or not np.isfinite(prior).all():
        raise ValueError("invalid prior contract")
    if np.any((prior <= 0) | (prior >= 1)):
        raise ValueError("anchor probabilities must lie strictly inside (0,1)")
    if np.any(dwell < 1):
        raise ValueError("durations must be positive")
    maximum_duration = prior.shape[1]
    capped = np.minimum(dwell, maximum_duration)
    output = np.empty_like(prior)
    for state in (0, 1):
        for duration in range(1, maximum_duration + 1):
            mask = (current == bool(state)) & (capped == duration)
            successes = int(following[mask].sum())
            trials = int(mask.sum())
            output[state, duration - 1] = (
                successes + prior_strength * prior[state, duration - 1]
            ) / (trials + prior_strength)
    return output


def transition_scores(
    current_states: Iterable[bool],
    durations: Iterable[int],
    next_states: Iterable[bool],
    probabilities: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Bernoulli log-loss and Brier arrays for a duration table."""

    current = np.asarray(tuple(current_states), dtype=np.int64)
    dwell = np.asarray(tuple(durations), dtype=np.int64)
    following = np.asarray(tuple(next_states), dtype=np.float64)
    table = np.asarray(probabilities, dtype=np.float64)
    if current.shape != dwell.shape or current.shape != following.shape:
        raise ValueError("transition arrays must align")
    if table.ndim != 2 or table.shape[0] != 2:
        raise ValueError("invalid probability table")
    capped = np.minimum(dwell, table.shape[1]) - 1
    if np.any(capped < 0):
        raise ValueError("durations must be positive")
    predicted = np.clip(table[current, capped], 1e-12, 1 - 1e-12)
    log_losses = -(
        following * np.log(predicted) + (1 - following) * np.log1p(-predicted)
    )
    briers = (following - predicted) ** 2
    return log_losses, briers
