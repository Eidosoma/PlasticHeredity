"""Pure scientific primitives for the strict-B transplant follow-up.

This module deliberately performs no filesystem access.  The runner owns
provenance, seed allocation, simulation, checkpointing, and reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.integer]

INHERITANCE_THRESHOLD = 0.90
DEPARTURE_THRESHOLD = 0.85
CAPTURE_WINDOW = 8
HORIZON = 32


@dataclass(frozen=True)
class FutureScore:
    """All frozen readouts for one stochastic future and one target."""

    observed: int
    completed: bool
    arrival_f4: bool
    arrival_f8: bool
    arrival_f16: bool
    first_arrival: int
    capture_f16: bool
    capture_f32: bool
    first_capture: int
    occupancy: float
    maximum_residence: int
    departed: bool
    reentered: bool
    coherent8: bool
    first_break: int

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def cosine(left: NDArray, right: NDArray) -> float:
    left_f = np.asarray(left, dtype=np.float64)
    right_f = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_f) * np.linalg.norm(right_f))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(left_f, right_f) / denominator, 0.0, 1.0))


def cosine_matrix(values: NDArray) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cosine_matrix expects a two-dimensional array")
    norms = np.linalg.norm(matrix, axis=1)
    denominator = np.outer(norms, norms)
    result = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64)
    np.divide(matrix @ matrix.T, denominator, out=result, where=denominator > 0)
    return np.clip(result, 0.0, 1.0)


def bray_curtis_similarity(left: NDArray, right: NDArray) -> float:
    left_f = np.asarray(left, dtype=np.float64)
    right_f = np.asarray(right, dtype=np.float64)
    denominator = float(np.abs(left_f).sum() + np.abs(right_f).sum())
    if denominator == 0.0:
        return 0.0
    return float(np.clip(1.0 - np.abs(left_f - right_f).sum() / denominator, 0.0, 1.0))


def medoid(values: NDArray) -> NDArray[np.uint8]:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("medoid requires a non-empty two-dimensional array")
    similarities = cosine_matrix(matrix)
    return np.asarray(matrix[int(np.argmax(similarities.mean(axis=1)))], dtype=np.uint8).copy()


def permute_state(state: NDArray, permutation: NDArray) -> NDArray:
    values = np.asarray(state)
    order = np.asarray(permutation, dtype=np.int64)
    if values.shape[-1] != order.size:
        raise ValueError("state width and permutation length differ")
    return values[..., order].copy()


def inverse_permute_state(state: NDArray, permutation: NDArray) -> NDArray:
    order = np.asarray(permutation, dtype=np.int64)
    return permute_state(state, np.argsort(order))


def permute_beta(beta: NDArray, permutation: NDArray) -> FloatArray:
    matrix = np.asarray(beta, dtype=np.float64)
    order = np.asarray(permutation, dtype=np.int64)
    if matrix.shape != (order.size, order.size):
        raise ValueError("beta shape and permutation length differ")
    return matrix[np.ix_(order, order)].copy()


def choose_rule_permutation(
    donors: NDArray,
    *,
    seed: int,
    proposals: int = 4_096,
) -> tuple[NDArray[np.int16], float, FloatArray]:
    """Choose the seeded label permutation minimizing mean donor self-H."""

    states = np.asarray(donors)
    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError("donors must be a non-empty two-dimensional array")
    if proposals < 1:
        raise ValueError("proposals must be positive")
    rng = np.random.default_rng(seed)
    best: NDArray[np.int16] | None = None
    best_score = float("inf")
    best_values: FloatArray | None = None
    for _ in range(proposals):
        candidate = np.asarray(rng.permutation(states.shape[1]), dtype=np.int16)
        values = np.asarray(
            [cosine(state, permute_state(state, candidate)) for state in states],
            dtype=np.float64,
        )
        score = float(values.mean())
        if score < best_score:
            best = candidate
            best_score = score
            best_values = values
    assert best is not None and best_values is not None
    return best, best_score, best_values


def _all_pairwise_above(values: NDArray, threshold: float) -> bool:
    matrix = cosine_matrix(values)
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return bool(upper.size == 0 or np.all(upper > threshold))


def first_arrival(
    daughters: NDArray,
    target: NDArray,
    *,
    observed: int,
    threshold: float = INHERITANCE_THRESHOLD,
) -> int:
    for index in range(observed):
        if cosine(daughters[index], target) > threshold:
            return index + 1
    return -1


def first_target_capture(
    daughters: NDArray,
    target: NDArray,
    *,
    observed: int,
    horizon: int,
    threshold: float = INHERITANCE_THRESHOLD,
    window: int = CAPTURE_WINDOW,
) -> int:
    stop = min(observed, horizon)
    for start in range(0, stop - window + 1):
        values = np.asarray(daughters[start : start + window])
        if not all(cosine(value, target) > threshold for value in values):
            continue
        if _all_pairwise_above(values, threshold):
            return start + 1
    return -1


def first_target_capture_metric(
    daughters: NDArray,
    target: NDArray,
    *,
    observed: int,
    horizon: int,
    threshold: float,
    metric: str,
    window: int = CAPTURE_WINDOW,
) -> int:
    if metric == "cosine":
        similarity = cosine
    elif metric == "bray_curtis":
        similarity = bray_curtis_similarity
    else:
        raise ValueError(f"unknown similarity metric: {metric}")
    stop = min(observed, horizon)
    for start in range(0, stop - window + 1):
        block = np.asarray(daughters[start : start + window])
        if not all(similarity(value, target) > threshold for value in block):
            continue
        passed = True
        for left in range(window):
            for right in range(left + 1, window):
                if similarity(block[left], block[right]) <= threshold:
                    passed = False
                    break
            if not passed:
                break
        if passed:
            return start + 1
    return -1


def maximum_target_residence(
    daughters: NDArray,
    target: NDArray,
    *,
    observed: int,
    threshold: float = INHERITANCE_THRESHOLD,
) -> int:
    longest = 0
    current = 0
    for index in range(observed):
        if cosine(daughters[index], target) > threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def any_coherent_window(
    daughters: NDArray,
    *,
    observed: int,
    threshold: float = INHERITANCE_THRESHOLD,
    window: int = CAPTURE_WINDOW,
) -> bool:
    for start in range(0, observed - window + 1):
        if _all_pairwise_above(daughters[start : start + window], threshold):
            return True
    return False


def score_future(
    daughters: NDArray,
    boundary_h: NDArray,
    target: NDArray,
    *,
    observed: int,
) -> FutureScore:
    values = np.asarray(daughters)
    h = np.asarray(boundary_h, dtype=np.float64)
    if values.ndim != 2 or observed < 0 or observed > values.shape[0]:
        raise ValueError("invalid future array or observed count")
    arrival = first_arrival(values, target, observed=observed)
    capture16 = first_target_capture(values, target, observed=observed, horizon=16)
    capture32 = first_target_capture(values, target, observed=observed, horizon=32)
    similarities = np.asarray(
        [cosine(values[index], target) for index in range(observed)], dtype=np.float64
    )
    departures = np.flatnonzero(similarities <= DEPARTURE_THRESHOLD)
    departed = bool(departures.size)
    reentered = False
    if departed:
        departure = int(departures[0])
        for start in range(departure + 1, observed - CAPTURE_WINDOW + 1):
            block = values[start : start + CAPTURE_WINDOW]
            if all(cosine(item, target) > INHERITANCE_THRESHOLD for item in block) and _all_pairwise_above(
                block, INHERITANCE_THRESHOLD
            ):
                reentered = True
                break
    breaks = np.flatnonzero(h[:observed] <= INHERITANCE_THRESHOLD)
    return FutureScore(
        observed=int(observed),
        completed=bool(observed == HORIZON),
        arrival_f4=bool(arrival != -1 and arrival <= 4),
        arrival_f8=bool(arrival != -1 and arrival <= 8),
        arrival_f16=bool(arrival != -1 and arrival <= 16),
        first_arrival=int(arrival),
        capture_f16=bool(capture16 != -1),
        capture_f32=bool(capture32 != -1),
        first_capture=int(capture32),
        occupancy=float(np.mean(similarities > INHERITANCE_THRESHOLD)) if observed else 0.0,
        maximum_residence=maximum_target_residence(values, target, observed=observed),
        departed=departed,
        reentered=reentered,
        coherent8=any_coherent_window(values, observed=observed),
        first_break=int(breaks[0] + 1) if breaks.size else -1,
    )


def first_capture_class(
    daughters: NDArray,
    targets: Sequence[NDArray],
    *,
    observed: int,
) -> int:
    hits = [
        first_target_capture(daughters, target, observed=observed, horizon=HORIZON)
        for target in targets
    ]
    valid = [(hit, index) for index, hit in enumerate(hits) if hit != -1]
    if not valid:
        return -1
    valid.sort()
    if len(valid) > 1 and valid[0][0] == valid[1][0]:
        return -1
    return int(valid[0][1])


def generate_mass_preserving_perturbations(
    form: NDArray,
    other_form: NDArray,
    *,
    seed: int,
    required: int = 8,
    proposals_per_dose: int = 4_096,
    dose_ladder: Sequence[int] = (4, 8, 12, 16, 20, 24),
) -> tuple[int, list[NDArray[np.uint8]], int]:
    """Find the smallest registered substitution dose yielding valid starts."""

    source = np.asarray(form, dtype=np.int64)
    other = np.asarray(other_form, dtype=np.int64)
    if source.shape != other.shape or source.ndim != 1:
        raise ValueError("forms must be equal one-dimensional vectors")
    mass = int(source.sum())
    if mass <= 0:
        raise ValueError("form has zero mass")
    expanded = np.repeat(np.arange(source.size), source)
    total_proposals = 0
    for dose_index, dose in enumerate(dose_ladder):
        if dose > mass:
            continue
        seed_words = [
            int(seed) & 0xFFFFFFFF,
            (int(seed) >> 32) & 0xFFFFFFFF,
            int(dose_index),
            int(dose),
        ]
        rng = np.random.default_rng(np.random.SeedSequence(seed_words))
        accepted: list[NDArray[np.uint8]] = []
        seen: set[bytes] = set()
        for _ in range(proposals_per_dose):
            total_proposals += 1
            candidate = source.copy()
            removed = rng.choice(expanded, size=dose, replace=False)
            np.add.at(candidate, removed, -1)
            added = rng.choice(source.size, size=dose, replace=True)
            np.add.at(candidate, added, 1)
            encoded = candidate.astype(np.uint8).tobytes()
            if encoded in seen:
                continue
            seen.add(encoded)
            own_h = cosine(source, candidate)
            other_h = cosine(other, candidate)
            if 0.85 <= own_h <= 0.95 and other_h <= 0.85:
                accepted.append(candidate.astype(np.uint8))
                if len(accepted) == required:
                    return int(dose), accepted, total_proposals
    return -1, [], total_proposals


def bootstrap_mean_ci(
    values: NDArray,
    *,
    seed: int,
    repetitions: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(repetitions, array.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(array.mean()), float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        return float("nan"), float("nan"), float("nan")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return float(p), float(center - half), float(center + half)


def early_late_break_hazard(first_breaks: Iterable[int], *, horizon: int = HORIZON) -> dict[str, float]:
    values = np.asarray(list(first_breaks), dtype=np.int16)
    hazards: list[float] = []
    for generation in range(1, horizon + 1):
        at_risk = int(np.sum((values == -1) | (values >= generation)))
        events = int(np.sum(values == generation))
        hazards.append(events / at_risk if at_risk else float("nan"))
    early = float(np.nanmean(hazards[:4]))
    late = float(np.nanmean(hazards[8:]))
    ratio = early / late if late > 0 else float("inf")
    return {"early_hazard": early, "late_hazard": late, "early_late_ratio": ratio}
