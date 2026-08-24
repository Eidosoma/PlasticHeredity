from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .engine import in_basin, rollout_numpy, sequential_sweep_numpy
from .rng import generator


@dataclass(frozen=True)
class Rulebook:
    source_id: int
    proposal_count: int
    weights: np.ndarray
    target_a: np.ndarray
    target_b: np.ndarray
    neutral: np.ndarray
    basin_a: float
    basin_b: float

    def target(self, history: str) -> np.ndarray:
        return self.target_a if history == "A" else self.target_b

    def opposite(self, history: str) -> np.ndarray:
        return self.target_b if history == "A" else self.target_a


def _balanced_state(rng: np.random.Generator, genes: int) -> np.ndarray:
    state = -np.ones(genes, dtype=np.int8)
    state[rng.choice(genes, size=genes // 2, replace=False)] = 1
    return state


def _neutral_midpoint(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    neutral = a.copy()
    differing = np.flatnonzero(a != b)
    choice = rng.random(differing.size) < 0.5
    neutral[differing] = np.where(choice, a[differing], b[differing])
    shared = np.flatnonzero(a == b)
    if shared.size:
        flip = rng.random(shared.size) < 0.5
        neutral[shared] = np.where(flip, -a[shared], a[shared])
    return neutral.astype(np.int8)


def _point_stable(weights: np.ndarray, target: np.ndarray) -> bool:
    current = target[None, :]
    return bool(np.array_equal(sequential_sweep_numpy(weights, current)[0], target))


def _basin_fraction(weights: np.ndarray, target: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    trials = 64
    initial = np.repeat(target[None, :], trials, axis=0)
    for row in initial:
        flips = rng.choice(target.size, size=2, replace=False)
        row[flips] *= -1
    history = rollout_numpy(
        weights,
        initial,
        np.zeros_like(initial, dtype=np.float64),
        sweeps=24,
        theta=0.0,
        flip_probability=0.0,
        rng=rng,
    )
    return float(np.mean(in_basin(history[-1], target, maximum_hamming=0)))


def generate_rulebook(source_id: int, protocol: dict[str, Any], domain: str) -> Rulebook:
    engine = protocol["engine"]
    genes = int(engine["genes"])
    master = str(protocol["master_seed"])
    maximum = int(engine["maximum_source_proposals"])
    for proposal in range(maximum):
        rng = generator(master, "source", domain, source_id, proposal)
        a = _balanced_state(rng, genes)
        b = _balanced_state(rng, genes)
        if int(np.sum(a != b)) < int(engine["minimum_target_hamming"]):
            continue
        mask = rng.random((genes, genes)) < float(engine["connectivity"])
        random_part = rng.normal(0.0, float(engine["random_weight_sd"]), (genes, genes)) * mask
        memory = float(engine["memory_strength"]) * (
            np.outer(a, a) + np.outer(b, b)
        ) / genes
        weights = random_part + memory
        np.fill_diagonal(weights, 0.0)
        row_norm = np.linalg.norm(weights, axis=1, keepdims=True)
        weights = weights / np.maximum(row_norm, 1e-8)
        if not (_point_stable(weights, a) and _point_stable(weights, b)):
            continue
        basin_a = _basin_fraction(weights, a, proposal * 2 + 1)
        basin_b = _basin_fraction(weights, b, proposal * 2 + 2)
        if min(basin_a, basin_b) < float(engine["minimum_basin_fraction"]):
            continue
        neutral = _neutral_midpoint(a, b, rng)
        return Rulebook(
            source_id=source_id,
            proposal_count=proposal + 1,
            weights=weights.astype(np.float32),
            target_a=a,
            target_b=b,
            neutral=neutral,
            basin_a=basin_a,
            basin_b=basin_b,
        )
    raise RuntimeError(f"source {source_id} exceeded proposal guard")


def rulebook_metadata(rulebook: Rulebook) -> dict[str, Any]:
    return {
        "source_id": rulebook.source_id,
        "proposal_count": rulebook.proposal_count,
        "basin_a": rulebook.basin_a,
        "basin_b": rulebook.basin_b,
        "target_distance": int(np.sum(rulebook.target_a != rulebook.target_b)),
        "weight_l1": float(np.sum(np.abs(rulebook.weights))),
    }

