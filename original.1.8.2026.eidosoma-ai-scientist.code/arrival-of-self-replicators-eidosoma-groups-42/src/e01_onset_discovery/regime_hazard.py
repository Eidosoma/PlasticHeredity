"""Two-state heredity-regime hazard and finite-horizon process utilities.

The functions in this module operate on binary post-fission inheritance
sequences.  They deliberately keep ordinary inheritance, first-order state
dependence, duration dependence, and matrix-specific transition propensities
as separate model layers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ProcessProbability:
    """Finite-horizon probabilities for break and renewed heredity."""

    break_probability: float
    joint_break_run_probability: float
    run_probability_given_break: float


def trailing_run_length(values: Iterable[bool]) -> int:
    """Return the length of the final constant-state run."""

    sequence = tuple(bool(value) for value in values)
    if not sequence:
        return 0
    final = sequence[-1]
    length = 0
    for value in reversed(sequence):
        if value != final:
            break
        length += 1
    return length


def transition_rows(
    initial_state: bool,
    initial_duration: int,
    future: Iterable[bool],
) -> tuple[tuple[bool, int, bool], ...]:
    """Expand a future sequence into current-state/duration/next-state rows."""

    if initial_duration < 1:
        raise ValueError("initial_duration must be positive")
    current = bool(initial_state)
    duration = int(initial_duration)
    rows: list[tuple[bool, int, bool]] = []
    for next_state in future:
        next_value = bool(next_state)
        rows.append((current, duration, next_value))
        if next_value == current:
            duration += 1
        else:
            current = next_value
            duration = 1
    return tuple(rows)


def smoothed_probability(successes: int, trials: int) -> float:
    """Jeffreys-smoothed Bernoulli mean."""

    if successes < 0 or trials < 0 or successes > trials:
        raise ValueError("invalid Bernoulli counts")
    return float((successes + 0.5) / (trials + 1.0))


def fit_iid(next_states: Iterable[bool]) -> float:
    values = np.asarray(tuple(next_states), dtype=np.bool_)
    return smoothed_probability(int(values.sum()), len(values))


def fit_markov(
    current_states: Iterable[bool], next_states: Iterable[bool]
) -> NDArray[np.float64]:
    current = np.asarray(tuple(current_states), dtype=np.bool_)
    following = np.asarray(tuple(next_states), dtype=np.bool_)
    if current.shape != following.shape:
        raise ValueError("current and next states must align")
    return np.asarray(
        [
            smoothed_probability(int(following[current == state].sum()), int((current == state).sum()))
            for state in (False, True)
        ],
        dtype=np.float64,
    )


def fit_semimarkov(
    current_states: Iterable[bool],
    durations: Iterable[int],
    next_states: Iterable[bool],
    markov_probabilities: NDArray[np.float64],
    *,
    maximum_duration: int = 12,
    prior_strength: float = 1.0,
) -> NDArray[np.float64]:
    """Fit capped duration-specific transition probabilities.

    Each state-duration cell receives a fixed empirical-Bayes prior centered
    on its pooled first-order Markov probability.  Duration ``maximum_duration``
    is the inclusive overflow bin.
    """

    current = np.asarray(tuple(current_states), dtype=np.bool_)
    dwell = np.asarray(tuple(durations), dtype=np.int64)
    following = np.asarray(tuple(next_states), dtype=np.bool_)
    baseline = np.asarray(markov_probabilities, dtype=np.float64)
    if current.shape != dwell.shape or current.shape != following.shape:
        raise ValueError("transition arrays must align")
    if baseline.shape != (2,) or maximum_duration < 1 or prior_strength <= 0:
        raise ValueError("invalid semi-Markov contract")
    if np.any(dwell < 1):
        raise ValueError("durations must be positive")
    capped = np.minimum(dwell, maximum_duration)
    table = np.empty((2, maximum_duration), dtype=np.float64)
    for state in (0, 1):
        for duration in range(1, maximum_duration + 1):
            mask = (current == bool(state)) & (capped == duration)
            successes = int(following[mask].sum())
            trials = int(mask.sum())
            table[state, duration - 1] = (
                successes + prior_strength * baseline[state]
            ) / (trials + prior_strength)
    return table


def posterior_matrix_markov(
    prefix: Iterable[bool],
    pooled_markov: NDArray[np.float64],
    *,
    prior_strength: float = 1.0,
) -> NDArray[np.float64]:
    """Shrink one matrix's prefix transition probabilities to pooled values."""

    values = np.asarray(tuple(prefix), dtype=np.bool_)
    pooled = np.asarray(pooled_markov, dtype=np.float64)
    if pooled.shape != (2,) or prior_strength <= 0:
        raise ValueError("invalid matrix-Markov contract")
    if len(values) < 2:
        return pooled.copy()
    current = values[:-1]
    following = values[1:]
    output = np.empty(2, dtype=np.float64)
    for state in (0, 1):
        mask = current == bool(state)
        successes = int(following[mask].sum())
        trials = int(mask.sum())
        output[state] = (
            successes + prior_strength * pooled[state]
        ) / (trials + prior_strength)
    return output


def transport_duration_effect(
    matrix_markov: NDArray[np.float64],
    pooled_markov: NDArray[np.float64],
    pooled_semimarkov: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply pooled duration odds ratios to matrix-specific base odds."""

    matrix = np.asarray(matrix_markov, dtype=np.float64)
    pooled = np.asarray(pooled_markov, dtype=np.float64)
    semi = np.asarray(pooled_semimarkov, dtype=np.float64)
    if matrix.shape != (2,) or pooled.shape != (2,) or semi.shape[0] != 2:
        raise ValueError("invalid duration-transport shapes")
    epsilon = 1e-12

    def odds(values: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(values, epsilon, 1.0 - epsilon)
        return clipped / (1.0 - clipped)

    transported_odds = odds(matrix)[:, None] * odds(semi) / odds(pooled)[:, None]
    output = transported_odds / (1.0 + transported_odds)
    return np.clip(output, epsilon, 1.0 - epsilon)


def finite_horizon_process_probability(
    probability: Callable[[bool, int], float],
    *,
    initial_state: bool,
    initial_duration: int,
    horizon: int,
    required_run: int = 3,
    maximum_duration: int = 12,
) -> ProcessProbability:
    """Propagate break and break-then-run probabilities exactly.

    A *break* is the first future non-inherited fission.  Certification occurs
    when ``required_run`` consecutive inherited fissions follow a break.  The
    destination is therefore an online process event, not a completed-run
    composition-space basin.
    """

    if initial_duration < 1 or horizon < 1 or required_run < 1:
        raise ValueError("invalid finite-horizon process contract")
    # key: current state, capped dwell, break seen, post-break inherited run,
    # success already reached
    distribution: dict[tuple[bool, int, bool, int, bool], float] = {
        (
            bool(initial_state),
            min(int(initial_duration), maximum_duration),
            False,
            0,
            False,
        ): 1.0
    }
    for _ in range(horizon):
        updated: defaultdict[tuple[bool, int, bool, int, bool], float] = defaultdict(float)
        for (state, dwell, broken, post_run, success), mass in distribution.items():
            p_inherited = float(probability(state, dwell))
            if not np.isfinite(p_inherited) or not 0.0 <= p_inherited <= 1.0:
                raise ValueError("transition probability outside [0,1]")
            for next_state, branch_probability in (
                (False, 1.0 - p_inherited),
                (True, p_inherited),
            ):
                if branch_probability == 0.0:
                    continue
                next_dwell = min(dwell + 1, maximum_duration) if next_state == state else 1
                next_broken = broken or not next_state
                if success:
                    next_run = post_run
                    next_success = True
                elif not next_state:
                    next_run = 0
                    next_success = False
                elif next_broken:
                    next_run = post_run + 1
                    next_success = next_run >= required_run
                else:
                    next_run = 0
                    next_success = False
                updated[
                    (next_state, next_dwell, next_broken, next_run, next_success)
                ] += mass * branch_probability
        distribution = dict(updated)
    break_probability = float(
        sum(mass for (_, _, broken, _, _), mass in distribution.items() if broken)
    )
    joint_probability = float(
        sum(mass for (_, _, _, _, success), mass in distribution.items() if success)
    )
    conditional = (
        joint_probability / break_probability if break_probability > 0.0 else float("nan")
    )
    return ProcessProbability(
        break_probability=break_probability,
        joint_break_run_probability=joint_probability,
        run_probability_given_break=conditional,
    )
