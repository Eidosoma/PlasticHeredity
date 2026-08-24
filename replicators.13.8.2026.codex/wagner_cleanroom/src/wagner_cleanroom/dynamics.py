from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .rng import generator


POINT = np.int8(0)
CYCLE = np.int8(1)
NONCONVERGENT = np.int8(2)
_POPCOUNT_TABLES: dict[int, np.ndarray] = {
    10: np.fromiter((value.bit_count() for value in range(1 << 10)), dtype=np.uint8),
}


def state_matrix(genes: int = 10) -> np.ndarray:
    states = np.arange(1 << genes, dtype=np.uint16)[:, None]
    bits = ((states >> np.arange(genes, dtype=np.uint16)) & 1).astype(np.int8)
    return bits * 2 - 1


def encode_state(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    powers = (np.uint16(1) << np.arange(values.shape[-1], dtype=np.uint16))
    return ((values > 0).astype(np.uint16) * powers).sum(axis=-1, dtype=np.uint16)


def decode_state(state: int | np.ndarray, genes: int = 10) -> np.ndarray:
    values = np.asarray(state, dtype=np.uint16)[..., None]
    bits = ((values >> np.arange(genes, dtype=np.uint16)) & 1).astype(np.int8)
    return bits * 2 - 1


def hamming(a: int | np.ndarray, b: int | np.ndarray, genes: int = 10) -> np.ndarray:
    xor = np.bitwise_xor(np.asarray(a, dtype=np.uint16), np.asarray(b, dtype=np.uint16))
    if genes not in _POPCOUNT_TABLES:
        _POPCOUNT_TABLES[genes] = np.fromiter(
            (value.bit_count() for value in range(1 << genes)), dtype=np.uint8
        )
    return _POPCOUNT_TABLES[genes][xor]


def sequential_sweep(state: np.ndarray, weights: np.ndarray, order: Iterable[int] | None = None) -> np.ndarray:
    current = np.asarray(state, dtype=np.int8).copy()
    if current.shape != (weights.shape[0],):
        raise ValueError("state and weight dimensions differ")
    if order is None:
        order = range(weights.shape[0])
    for gene in order:
        field = float(np.dot(weights[gene], current))
        if field > 0.0:
            current[gene] = 1
        elif field < 0.0:
            current[gene] = -1
    return current


@dataclass(slots=True)
class Landscape:
    successor: np.ndarray
    adult: np.ndarray
    kind: np.ndarray
    point_index: np.ndarray
    transient: np.ndarray
    cycle_length: np.ndarray
    point_states: np.ndarray
    basin_sizes: np.ndarray


@dataclass(slots=True)
class Rulebook:
    uid: str
    proposal_index: int
    weights: np.ndarray
    landscape: Landscape
    targets: np.ndarray
    target_point_indices: np.ndarray
    midpoints: np.ndarray
    forced_breaks: np.ndarray
    donors: np.ndarray
    nulls: np.ndarray
    shuffles: np.ndarray
    mark_permutation: np.ndarray


def build_landscape(weights: np.ndarray, max_sweeps: int = 100) -> Landscape:
    genes = int(weights.shape[0])
    matrix = state_matrix(genes)
    swept = matrix.copy()
    for gene in range(genes):
        fields = swept @ weights[gene]
        swept[fields > 0.0, gene] = 1
        swept[fields < 0.0, gene] = -1
    successor = encode_state(swept)

    count = 1 << genes
    adult = np.zeros(count, dtype=np.uint16)
    kind = np.full(count, NONCONVERGENT, dtype=np.int8)
    point_state_for_start = np.full(count, -1, dtype=np.int32)
    transient = np.full(count, max_sweeps + 1, dtype=np.uint16)
    cycle_length = np.zeros(count, dtype=np.uint16)

    for start in range(count):
        current = start
        seen: dict[int, int] = {current: 0}
        path = [current]
        resolved = False
        for sweep in range(1, max_sweeps + 1):
            nxt = int(successor[current])
            if nxt in seen:
                cycle_start = seen[nxt]
                cycle = path[cycle_start:]
                adult[start] = np.uint16(nxt)
                transient[start] = np.uint16(cycle_start)
                cycle_length[start] = np.uint16(len(cycle))
                if len(cycle) == 1:
                    kind[start] = POINT
                    point_state_for_start[start] = nxt
                else:
                    kind[start] = CYCLE
                resolved = True
                break
            seen[nxt] = len(path)
            path.append(nxt)
            current = nxt
        if not resolved:
            adult[start] = np.uint16(current)

    point_states = np.array(sorted(set(point_state_for_start[point_state_for_start >= 0].tolist())), dtype=np.uint16)
    index_of = {int(state): i for i, state in enumerate(point_states)}
    point_index = np.full(count, -1, dtype=np.int16)
    for state in range(count):
        point = int(point_state_for_start[state])
        if point >= 0:
            point_index[state] = np.int16(index_of[point])
    basin_sizes = np.array([(point_index == i).sum() for i in range(len(point_states))], dtype=np.uint16)
    return Landscape(successor, adult, kind, point_index, transient, cycle_length, point_states, basin_sizes)


def _select_pair(landscape: Landscape, genes: int, minimum_basin: float, minimum_distance: float) -> tuple[int, int] | None:
    state_count = 1 << genes
    candidates: list[tuple[tuple[int, int, int, int, int], tuple[int, int]]] = []
    for left in range(len(landscape.point_states)):
        if int(landscape.basin_sizes[left]) / state_count < minimum_basin:
            continue
        for right in range(left + 1, len(landscape.point_states)):
            if int(landscape.basin_sizes[right]) / state_count < minimum_basin:
                continue
            a, b = int(landscape.point_states[left]), int(landscape.point_states[right])
            distance = int(hamming(a, b, genes))
            if distance / genes < minimum_distance:
                continue
            small = min(int(landscape.basin_sizes[left]), int(landscape.basin_sizes[right]))
            total = int(landscape.basin_sizes[left]) + int(landscape.basin_sizes[right])
            score = (-small, -total, -distance, min(a, b), max(a, b))
            candidates.append((score, (left, right)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _midpoints(a: int, b: int, permutation: np.ndarray, genes: int) -> np.ndarray:
    a_bits = ((a >> np.arange(genes)) & 1).astype(np.uint8)
    b_bits = ((b >> np.arange(genes)) & 1).astype(np.uint8)
    first, second = a_bits.copy(), a_bits.copy()
    differing = [int(g) for g in permutation if a_bits[g] != b_bits[g]]
    for position, gene in enumerate(differing):
        if position % 2 == 0:
            first[gene], second[gene] = a_bits[gene], b_bits[gene]
        else:
            first[gene], second[gene] = b_bits[gene], a_bits[gene]
    values = np.stack([first, second]).astype(np.int8) * 2 - 1
    return encode_state(values).astype(np.uint16)


def _forced_break(target: int, point_index: int, landscape: Landscape, permutation: np.ndarray) -> int | None:
    for gene in permutation:
        candidate = target ^ (1 << int(gene))
        if int(landscape.point_index[candidate]) != point_index:
            return candidate
    return None


def _matched_controls(
    target: int,
    midpoint: int,
    point_index: int,
    landscape: Landscape,
    permutation_rank: dict[int, int],
    genes: int,
) -> tuple[int, int]:
    wanted_distance = int(hamming(target, midpoint, genes))
    same: list[tuple[tuple[int, int, int], int]] = []
    other: list[tuple[tuple[int, int, int], int]] = []
    for state in range(1 << genes):
        if state in (target, midpoint) or landscape.kind[state] != POINT:
            continue
        key = (
            abs(int(hamming(state, midpoint, genes)) - wanted_distance),
            int(landscape.transient[state]),
            permutation_rank.get(state, state),
        )
        if int(landscape.point_index[state]) == point_index:
            same.append((key, state))
        else:
            other.append((key, state))
    donor = min(same)[1] if same else target
    null = min(other)[1] if other else midpoint
    return donor, null


def sample_rulebook(
    master_label: str,
    proposal_index: int,
    genes: int = 10,
    max_sweeps: int = 100,
    minimum_basin: float = 0.05,
    minimum_distance: float = 0.4,
) -> Rulebook | None:
    rng = generator(master_label, "rulebook", proposal_index)
    weights = rng.standard_normal((genes, genes), dtype=np.float64)
    landscape = build_landscape(weights, max_sweeps=max_sweeps)
    pair = _select_pair(landscape, genes, minimum_basin, minimum_distance)
    if pair is None:
        return None
    permutation = generator(master_label, "rulebook", proposal_index, "permutation").permutation(genes).astype(np.uint8)
    left, right = pair
    targets = np.array([landscape.point_states[left], landscape.point_states[right]], dtype=np.uint16)
    target_indices = np.array([left, right], dtype=np.int16)
    forced = [
        _forced_break(int(targets[h]), int(target_indices[h]), landscape, permutation)
        for h in range(2)
    ]
    if any(value is None for value in forced):
        return None
    midpoints = _midpoints(int(targets[0]), int(targets[1]), permutation, genes)
    state_order = generator(master_label, "rulebook", proposal_index, "control-order").permutation(1 << genes)
    rank = {int(state): i for i, state in enumerate(state_order)}
    donors = np.zeros((2, 2), dtype=np.uint16)
    nulls = np.zeros((2, 2), dtype=np.uint16)
    for midpoint_index in range(2):
        for history in range(2):
            donor, null = _matched_controls(
                int(targets[history]), int(midpoints[midpoint_index]), int(target_indices[history]),
                landscape, rank, genes,
            )
            donors[midpoint_index, history] = donor
            nulls[midpoint_index, history] = null
    shuffles = np.zeros(2, dtype=np.uint16)
    for history in range(2):
        bits = decode_state(int(targets[history]), genes)
        shuffled = int(encode_state(bits[permutation]))
        if shuffled == int(targets[history]):
            shuffled = int(targets[history]) ^ (1 << int(permutation[0])) ^ (1 << int(permutation[1]))
        shuffles[history] = shuffled
    return Rulebook(
        uid=f"wagner-clean-{proposal_index:06d}",
        proposal_index=proposal_index,
        weights=weights,
        landscape=landscape,
        targets=targets,
        target_point_indices=target_indices,
        midpoints=midpoints,
        forced_breaks=np.asarray(forced, dtype=np.uint16),
        donors=donors,
        nulls=nulls,
        shuffles=shuffles,
        mark_permutation=permutation,
    )
