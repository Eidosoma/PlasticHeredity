from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np

from .engine import sequential_sweep_numpy, states_from_int, states_to_int
from .rng import generator, stable_permutation


@dataclass(frozen=True)
class Landscape:
    successor: np.ndarray
    attractor_index: np.ndarray
    attractors: tuple[tuple[int, ...], ...]
    basin_sizes: np.ndarray
    terminal_state: np.ndarray
    terminal_status: np.ndarray
    terminal_steps: np.ndarray


@dataclass(frozen=True)
class Rulebook:
    source_id: int
    proposal_count: int
    weights: np.ndarray
    target_a: np.ndarray
    target_b: np.ndarray
    midpoints: tuple[np.ndarray, np.ndarray]
    forced_a: np.ndarray
    forced_b: np.ndarray
    basin_a: float
    basin_b: float
    proposal_log: tuple[dict[str, Any], ...]
    landscape: Landscape
    adult_states: np.ndarray

    def target(self, history: str) -> np.ndarray:
        return self.target_a if history == "A" else self.target_b

    def opposite(self, history: str) -> np.ndarray:
        return self.target_b if history == "A" else self.target_a

    def forced(self, history: str) -> np.ndarray:
        return self.forced_a if history == "A" else self.forced_b

    def adult_table(self) -> np.ndarray:
        return self.adult_states


def enumerate_landscape(weights: np.ndarray, genes: int, max_sweeps: int = 100) -> Landscape:
    state_count = 1 << genes
    states = states_from_int(np.arange(state_count, dtype=np.uint16), genes)
    successor = states_to_int(sequential_sweep_numpy(weights, states)).astype(np.int32)
    assignment = np.full(state_count, -1, dtype=np.int32)
    attractors: list[tuple[int, ...]] = []
    for start in range(state_count):
        if assignment[start] >= 0:
            continue
        path: list[int] = []
        position: dict[int, int] = {}
        current = start
        while assignment[current] < 0 and current not in position:
            position[current] = len(path)
            path.append(current)
            current = int(successor[current])
        if assignment[current] >= 0:
            attractor_id = int(assignment[current])
        else:
            cycle = path[position[current]:]
            minimum_index = int(np.argmin(cycle))
            canonical = tuple(cycle[minimum_index:] + cycle[:minimum_index])
            attractor_id = len(attractors)
            attractors.append(canonical)
        for node in path:
            assignment[node] = attractor_id
    basins = np.bincount(assignment, minlength=len(attractors)).astype(np.int32)
    terminal = np.zeros(state_count, dtype=np.int32)
    terminal_status = np.zeros(state_count, dtype=np.int8)
    terminal_steps = np.zeros(state_count, dtype=np.int16)
    for start in range(state_count):
        seen = {start}
        current = start
        for step in range(1, max_sweeps + 1):
            previous = current
            current = int(successor[current])
            if current in seen:
                terminal[start] = current
                terminal_status[start] = 1 if current == previous else 2
                terminal_steps[start] = step
                break
            seen.add(current)
        else:
            terminal[start] = current
            terminal_status[start] = 0
            terminal_steps[start] = max_sweeps
    return Landscape(
        successor,
        assignment,
        tuple(attractors),
        basins,
        terminal,
        terminal_status,
        terminal_steps,
    )


def _eligible_pair(landscape: Landscape, genes: int, minimum_basin: float, minimum_distance: float) -> tuple[int, int, int, int] | None:
    fixed = {cycle[0]: index for index, cycle in enumerate(landscape.attractors) if len(cycle) == 1}
    candidates: list[tuple[int, int, int, int, int, int]] = []
    full_mask = (1 << genes) - 1
    minimum_count = int(np.ceil(minimum_basin * (1 << genes)))
    minimum_hamming = int(np.ceil(minimum_distance * genes))
    for left, left_index in fixed.items():
        right = left ^ full_mask
        if left >= right or right not in fixed:
            continue
        right_index = fixed[right]
        distance = int((left ^ right).bit_count())
        left_basin = int(landscape.basin_sizes[left_index])
        right_basin = int(landscape.basin_sizes[right_index])
        if distance < minimum_hamming or min(left_basin, right_basin) < minimum_count:
            continue
        candidates.append((min(left_basin, right_basin), left_basin + right_basin, -left, left, right, left_index))
    if not candidates:
        return None
    _, _, _, left, right, left_index = max(candidates)
    right_index = next(index for index, cycle in enumerate(landscape.attractors) if cycle == (right,))
    return left, right, left_index, right_index


def _midpoints(left: int, right: int, genes: int, rng: np.random.Generator) -> tuple[int, int]:
    differing = [index for index in range(genes) if ((left ^ right) >> index) & 1]
    take_left = set(int(value) for value in rng.choice(differing, size=len(differing) // 2, replace=False))
    midpoint = 0
    for index in range(genes):
        source = left if index in take_left else right
        midpoint |= ((source >> index) & 1) << index
    return midpoint, midpoint ^ ((1 << genes) - 1)


def _forced_break(
    target: int,
    attractor_id: int,
    landscape: Landscape,
    genes: int,
    master_seed: str,
    domain: str,
    source_id: int,
    history: str,
) -> int:
    candidates = np.flatnonzero(landscape.attractor_index != attractor_id)
    distances = np.asarray([int((target ^ int(value)).bit_count()) for value in candidates])
    closest = candidates[distances == np.min(distances)]
    order = stable_permutation(len(closest), master_seed, "forced-break", domain, source_id, history)
    return int(closest[int(order[0])])


def generate_rulebook(source_id: int, protocol: dict[str, Any], domain: str) -> Rulebook:
    engine = protocol["engine"]
    genes = int(engine["genes"])
    master = str(protocol["master_seed"])
    proposal_log: list[dict[str, Any]] = []
    for proposal in range(int(engine["maximum_source_proposals"])):
        rng = generator(master, "source", domain, source_id, proposal)
        # Wagner genotypes are retained as float64.  The exact same bytes define
        # the deterministic landscape and all subsequent developmental cycles.
        weights = rng.normal(0.0, float(engine["weight_sd"]), size=(genes, genes)).astype(np.float64)
        landscape = enumerate_landscape(weights, genes, int(engine["max_sweeps"]))
        pair = _eligible_pair(
            landscape,
            genes,
            float(engine["minimum_basin_fraction"]),
            float(engine["minimum_form_distance"]),
        )
        digest = sha256(weights.tobytes(order="C")).hexdigest()
        log_entry: dict[str, Any] = {
            "proposal": proposal,
            "weight_sha256": digest,
            "successor_sha256": sha256(
                landscape.successor.astype("<i4", copy=False).tobytes(order="C")
            ).hexdigest(),
            "assignment_sha256": sha256(
                landscape.attractor_index.astype("<i4", copy=False).tobytes(order="C")
            ).hexdigest(),
            "attractor_count": len(landscape.attractors),
            "point_attractor_count": sum(len(cycle) == 1 for cycle in landscape.attractors),
            "basin_sizes": landscape.basin_sizes.astype(int).tolist(),
            "accepted": pair is not None,
        }
        proposal_log.append(log_entry)
        if pair is None:
            continue
        left, right, left_index, right_index = pair
        midpoint_rng = generator(master, "midpoints", domain, source_id, proposal)
        midpoint_values = _midpoints(left, right, genes, midpoint_rng)
        forced_a = _forced_break(left, left_index, landscape, genes, master, domain, source_id, "A")
        forced_b = _forced_break(right, right_index, landscape, genes, master, domain, source_id, "B")
        log_entry.update({
            "target_a": left,
            "target_b": right,
            "basin_a": int(landscape.basin_sizes[left_index]),
            "basin_b": int(landscape.basin_sizes[right_index]),
            "attractors": [list(cycle) for cycle in landscape.attractors],
        })
        return Rulebook(
            source_id=source_id,
            proposal_count=proposal + 1,
            weights=weights,
            target_a=states_from_int([left], genes)[0],
            target_b=states_from_int([right], genes)[0],
            midpoints=tuple(states_from_int(list(midpoint_values), genes)),
            forced_a=states_from_int([forced_a], genes)[0],
            forced_b=states_from_int([forced_b], genes)[0],
            basin_a=float(landscape.basin_sizes[left_index] / (1 << genes)),
            basin_b=float(landscape.basin_sizes[right_index] / (1 << genes)),
            proposal_log=tuple(proposal_log),
            landscape=landscape,
            adult_states=states_from_int(landscape.terminal_state, genes),
        )
    raise RuntimeError(f"source {source_id} exceeded proposal guard")


def rulebook_record(rulebook: Rulebook) -> dict[str, Any]:
    return {
        "source_id": rulebook.source_id,
        "proposal_count": rulebook.proposal_count,
        "weights": rulebook.weights.astype(float).tolist(),
        "target_a": rulebook.target_a.astype(int).tolist(),
        "target_b": rulebook.target_b.astype(int).tolist(),
        "midpoints": [value.astype(int).tolist() for value in rulebook.midpoints],
        "forced_a": rulebook.forced_a.astype(int).tolist(),
        "forced_b": rulebook.forced_b.astype(int).tolist(),
        "basin_a": rulebook.basin_a,
        "basin_b": rulebook.basin_b,
        "proposal_log": list(rulebook.proposal_log),
        "landscape": {
            "successor": rulebook.landscape.successor.astype(int).tolist(),
            "attractor_index": rulebook.landscape.attractor_index.astype(int).tolist(),
            "attractors": [list(cycle) for cycle in rulebook.landscape.attractors],
            "basin_sizes": rulebook.landscape.basin_sizes.astype(int).tolist(),
            "terminal_state": rulebook.landscape.terminal_state.astype(int).tolist(),
            "terminal_status": rulebook.landscape.terminal_status.astype(int).tolist(),
            "terminal_steps": rulebook.landscape.terminal_steps.astype(int).tolist(),
        },
    }
