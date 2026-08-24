"""Pure scientific mechanics for the plasticity-persistence campaign.

There is no filesystem access in this module.  Endpoint definitions are built
from the frozen local GARD simulator and the registered strict-eight evaluator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import factorial
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from plastic_heredity.config import GardConfig, SimulationContract
from plastic_heredity.regime_confirmation import evaluate_regime
from plastic_heredity.simulator import (
    FissionRecord,
    SimulationError,
    advance_fission,
    cosine_similarity,
    generate_initial_composition,
)
from reviewer_lineage_identity_response.followup_transplant_arrival_residence.transplant_core import (
    FutureScore,
    cosine_matrix,
    score_future,
)


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
INHERITANCE_THRESHOLD = 0.90
DISTINCTNESS_THRESHOLD = 0.85
LINEAGE_HORIZON = 32
F12_HORIZON = 12
STRICT_WINDOW = 8


@dataclass(frozen=True)
class DetailedOutcome:
    completed: bool
    f12: bool
    break12: bool
    recovery3_given_break: bool
    first_break: int
    first_run3_end: int
    run8: bool
    coherent8: bool
    distinct8: bool
    strict8: bool
    strict8_onset: int
    boundary_h: FloatArray
    last8: IntArray
    b_state: IntArray
    strict_b_state: IntArray

    def scalars(self) -> dict[str, bool | int]:
        result = asdict(self)
        for name in ("boundary_h", "last8", "b_state", "strict_b_state"):
            result.pop(name)
        return result


def _first_true_run(values: NDArray, length: int, start: int = 0) -> int:
    array = np.asarray(values, dtype=bool)
    for index in range(start, array.size - length + 1):
        if bool(array[index : index + length].all()):
            return index
    return -1


def _minimum_pairwise(states: Sequence[NDArray]) -> float:
    if len(states) < 2:
        return np.nan
    return min(
        cosine_similarity(states[left], states[right])
        for left in range(len(states))
        for right in range(left + 1, len(states))
    )


def score_detailed_records(records: Sequence[FissionRecord], n_types: int) -> DetailedOutcome:
    """Score weak F12 and registered strict-eight without pooling them."""

    values = list(records)
    if not values:
        return DetailedOutcome(
            False, False, False, False, -1, -1, False, False, False,
            False, -1, np.empty(0), np.zeros((STRICT_WINDOW, n_types), dtype=np.int64),
            np.zeros(n_types, dtype=np.int64), np.zeros(n_types, dtype=np.int64),
        )
    inherited = np.asarray([record.h > INHERITANCE_THRESHOLD for record in values], dtype=bool)
    first_twelve = inherited[:F12_HORIZON]
    breaks = np.flatnonzero(~first_twelve)
    first_break = int(breaks[0]) if breaks.size else -1
    run3_start = _first_true_run(first_twelve, 3, first_break + 1) if first_break >= 0 else -1
    run3_end = run3_start + 2 if run3_start >= 0 else -1
    break12 = first_break >= 0
    recovery3 = run3_start >= 0

    regime = evaluate_regime(values)
    all_breaks = np.flatnonzero(~inherited)
    first_break_all = int(all_breaks[0]) if all_breaks.size else -1
    anchor = values[first_break_all].parent if first_break_all >= 0 else None
    run8 = False
    coherent8 = False
    distinct8 = False
    if first_break_all >= 0:
        for start in range(first_break_all + 1, len(values) - STRICT_WINDOW + 1):
            if not bool(inherited[start : start + STRICT_WINDOW].all()):
                continue
            run8 = True
            daughters = [record.daughter for record in values[start : start + STRICT_WINDOW]]
            coherent8 |= _minimum_pairwise(daughters) > INHERITANCE_THRESHOLD
            distinct8 |= max(cosine_similarity(anchor, state) for state in daughters) <= DISTINCTNESS_THRESHOLD

    daughters = [record.daughter for record in values]
    last = daughters[-STRICT_WINDOW:]
    if len(last) < STRICT_WINDOW:
        last = [np.zeros(n_types, dtype=np.int64)] * (STRICT_WINDOW - len(last)) + last
    b_state = (
        values[run3_end].daughter.copy()
        if 0 <= run3_end < len(values)
        else np.zeros(n_types, dtype=np.int64)
    )
    strict_b_state = (
        values[int(regime.primary_all8_onset) + STRICT_WINDOW - 1].daughter.copy()
        if regime.primary_all8 and int(regime.primary_all8_onset) + STRICT_WINDOW <= len(values)
        else np.zeros(n_types, dtype=np.int64)
    )
    return DetailedOutcome(
        completed=len(values) == LINEAGE_HORIZON,
        f12=bool(break12 and recovery3),
        break12=break12,
        recovery3_given_break=recovery3,
        first_break=first_break,
        first_run3_end=run3_end,
        run8=run8,
        coherent8=coherent8,
        distinct8=distinct8,
        strict8=bool(regime.primary_all8),
        strict8_onset=int(regime.primary_all8_onset),
        boundary_h=np.asarray([record.h for record in values], dtype=np.float64),
        last8=np.asarray(last, dtype=np.int64),
        b_state=b_state,
        strict_b_state=strict_b_state,
    )


def simulate_detailed_lineage(
    beta: NDArray,
    config: GardConfig,
    contract: SimulationContract,
    *,
    seed: int,
    horizon: int = LINEAGE_HORIZON,
) -> DetailedOutcome:
    rng = np.random.default_rng(seed)
    current = generate_initial_composition(config, rng)
    records: list[FissionRecord] = []
    for _ in range(horizon):
        try:
            record = advance_fission(current, beta, config, contract, rng)
        except SimulationError:
            break
        records.append(record)
        current = record.daughter
    return score_detailed_records(records, config.n_types)


def simulate_future_scores(
    start: NDArray,
    target: NDArray,
    beta: NDArray,
    config: GardConfig,
    contract: SimulationContract,
    *,
    seed: int,
    horizon: int = LINEAGE_HORIZON,
) -> tuple[FutureScore, IntArray, FloatArray, int]:
    rng = np.random.default_rng(seed)
    current = np.asarray(start, dtype=np.int64).copy()
    daughters = np.zeros((horizon, config.n_types), dtype=np.int64)
    boundary_h = np.full(horizon, np.nan, dtype=np.float64)
    observed = 0
    for generation in range(horizon):
        try:
            record = advance_fission(current, beta, config, contract, rng)
        except SimulationError:
            break
        daughters[generation] = record.daughter
        boundary_h[generation] = record.h
        current = record.daughter
        observed += 1
    score = score_future(daughters, boundary_h, target, observed=observed)
    checkpoints = np.zeros((3, config.n_types), dtype=np.int64)
    for checkpoint_index, generation in enumerate((8, 16, 32)):
        if observed >= generation:
            checkpoints[checkpoint_index] = daughters[generation - 1]
    return score, checkpoints, boundary_h, observed


def last8_coherence(states: NDArray) -> float:
    values = np.asarray(states)
    if values.shape[0] != STRICT_WINDOW:
        raise ValueError("last-eight block has wrong length")
    return _minimum_pairwise(list(values))


def stable_components(
    last8_blocks: NDArray,
    *,
    stable_threshold: float = INHERITANCE_THRESHOLD,
    cluster_threshold: float = INHERITANCE_THRESHOLD,
) -> list[list[int]]:
    """Connected components among terminals whose own last eight are coherent."""

    blocks = np.asarray(last8_blocks)
    stable = [index for index, block in enumerate(blocks) if last8_coherence(block) > stable_threshold]
    if not stable:
        return []
    terminals = blocks[stable, -1]
    similarities = cosine_matrix(terminals)
    unseen = set(range(len(stable)))
    components: list[list[int]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(stable[node])
            neighbours = [other for other in sorted(unseen) if similarities[node, other] > cluster_threshold]
            for other in neighbours:
                unseen.remove(other)
                stack.append(other)
        components.append(sorted(component))
    return sorted(components, key=lambda component: (-len(component), component[0]))


def cluster_centroid(blocks: NDArray, members: Sequence[int]) -> FloatArray:
    terminals = np.asarray(blocks)[np.asarray(members, dtype=int), -1].astype(np.float64)
    return terminals.mean(axis=0)


def reproducible_multiform(
    last8_blocks: NDArray,
    *,
    minimum_fraction: float = 0.05,
    separation_threshold: float = DISTINCTNESS_THRESHOLD,
) -> tuple[bool, int, int]:
    blocks = np.asarray(last8_blocks)
    halves = (np.arange(blocks.shape[0]) % 2 == 0, np.arange(blocks.shape[0]) % 2 == 1)
    retained: list[list[FloatArray]] = []
    counts: list[int] = []
    for mask in halves:
        subset = blocks[mask]
        components = stable_components(subset)
        minimum = max(1, int(np.ceil(minimum_fraction * subset.shape[0])))
        centroids = [cluster_centroid(subset, component) for component in components if len(component) >= minimum]
        separated: list[FloatArray] = []
        for centroid in centroids:
            if all(cosine_similarity(centroid, other) <= separation_threshold for other in separated):
                separated.append(centroid)
        retained.append(separated)
        counts.append(len(separated))
    matches = 0
    used: set[int] = set()
    for left in retained[0]:
        candidates = [
            (cosine_similarity(left, right), index)
            for index, right in enumerate(retained[1])
            if index not in used
        ]
        if candidates:
            similarity, index = max(candidates)
            if similarity > INHERITANCE_THRESHOLD:
                matches += 1
                used.add(index)
    return matches >= 2, counts[0], counts[1]


def f12_decomposition(
    break_left: float,
    recovery_left: float,
    break_right: float,
    recovery_right: float,
) -> tuple[float, float, float]:
    """Symmetric exact decomposition of a product contrast."""

    total = break_left * recovery_left - break_right * recovery_right
    break_supply = (break_left - break_right) * (recovery_left + recovery_right) / 2.0
    recovery_propensity = (recovery_left - recovery_right) * (break_left + break_right) / 2.0
    return float(total), float(break_supply), float(recovery_propensity)


def factorial_shapley(values: Mapping[tuple[int, ...], float]) -> FloatArray:
    """Exact Shapley decomposition from all corners of a binary factorial."""

    if not values:
        raise ValueError("factorial values are empty")
    width = len(next(iter(values)))
    expected = {tuple(bits) for bits in np.ndindex(*(2,) * width)}
    if set(values) != expected:
        raise ValueError("factorial values do not contain every binary corner")
    result = np.zeros(width, dtype=np.float64)
    factors = set(range(width))
    denominator = factorial(width)
    for factor in range(width):
        others = sorted(factors - {factor})
        for size in range(width):
            weight = factorial(size) * factorial(width - size - 1) / denominator
            for subset in combinations(others, size):
                low = [0] * width
                for member in subset:
                    low[member] = 1
                high = low.copy()
                high[factor] = 1
                result[factor] += weight * (values[tuple(high)] - values[tuple(low)])
    return result
