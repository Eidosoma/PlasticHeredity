"""Frozen finite-horizon branching primitives for S19-L28.

The functions in this module deliberately separate the completed-run target
basin from the forward branch dynamics.  A branch receives only the current
Markov state, its within-generation phase, the catalytic matrix, the frozen
simulator definition, independent RNG streams, and one already frozen target
centroid.  No observed suffix is an input.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from e01_latent_timebase.core import (
    MAX_STEPS,
    N_MAX,
    SimulationDefinition,
    fission,
    poisson_update,
    select_daughter,
)

VERSION = "E01-S19-L28-BRANCHED-EMPIRICAL-COMMITTOR-IDENTIFIABILITY-v1.0.0"
TARGET_THRESHOLD = 0.9
HORIZON = 32
BRANCHES = 128
HALF_BRANCHES = 64


@dataclass(frozen=True, slots=True)
class RestoredState:
    """Every non-random simulator variable needed to resume one trajectory."""

    state: tuple[int, ...]
    observation_kind: str
    completed_fissions: int
    growth_generation_one_based: int
    generation_local_step: int
    batch_step: int


@dataclass(frozen=True, slots=True)
class BranchResult:
    """Compact, exactly replayable outcome of one finite-horizon branch."""

    entered_basin: bool
    first_entry_offset_one_based: int | None
    selected_observations_generated: int
    molecular_updates: int
    fissions: int
    terminal_status: str
    minimum_target_score: float | None
    maximum_target_score: float | None
    final_state_sha256: str
    path_sha256: str


def array_sha256(value: NDArray[Any]) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\x00")
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def relative_compositions(states: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
    counts = np.asarray(states, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != 100 or np.any(counts < 0):
        raise ValueError("states must be nonnegative observations-by-100 counts")
    masses = counts.sum(axis=1)
    if np.any(masses <= 0):
        raise ValueError("empty composition is not target-evaluable")
    return counts / masses[:, None]


def cosine_to_reference(
    states: NDArray[np.integer[Any]], reference: NDArray[np.floating[Any]]
) -> NDArray[np.float64]:
    compositions = relative_compositions(states)
    target = np.asarray(reference, dtype=np.float64)
    if target.shape != (100,) or np.any(target < 0) or not np.isfinite(target).all():
        raise ValueError("reference must be a finite nonnegative 100-vector")
    if not np.isclose(target.sum(), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("reference must be a unit-sum composition")
    denominator = np.linalg.norm(compositions, axis=1) * np.linalg.norm(target)
    if np.any(denominator <= 0):
        raise ValueError("nonpositive cosine denominator")
    return np.clip((compositions @ target) / denominator, -1.0, 1.0)


def dominant_component_centroid(
    post_fission_states: NDArray[np.integer[Any]],
    *,
    threshold: float = TARGET_THRESHOLD,
) -> tuple[NDArray[np.float64], tuple[int, ...]]:
    """Reproduce the frozen L23 dominant threshold-component reference exactly."""

    compositions = relative_compositions(post_fission_states)
    norms = np.linalg.norm(compositions, axis=1)
    normalized = compositions / norms[:, None]
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    adjacency = similarity >= float(threshold)
    remaining = set(range(len(compositions)))
    components: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        found: set[int] = set()
        while stack:
            item = stack.pop()
            if item in found:
                continue
            found.add(item)
            stack.extend(
                int(value)
                for value in np.flatnonzero(adjacency[item])
                if int(value) not in found
            )
        remaining.difference_update(found)
        components.append(tuple(sorted(found)))
    dominant = min(components, key=lambda value: (-len(value), min(value)))
    centroid = compositions[list(dominant)].mean(axis=0)
    centroid /= centroid.sum()
    return np.ascontiguousarray(centroid, dtype=np.float64), dominant


def restored_state_from_observation(observation: Any) -> RestoredState:
    state = np.asarray(observation.state, dtype=np.int64)
    if state.shape != (100,) or np.any(state < 0) or int(state.sum()) <= 0:
        raise ValueError("restored state must be a nonempty nonnegative 100-vector")
    if observation.observation_kind not in {
        "initial_selected_state",
        "molecular_update",
        "post_fission",
    }:
        raise ValueError("unsupported selected-clock observation kind")
    return RestoredState(
        state=tuple(map(int, state)),
        observation_kind=str(observation.observation_kind),
        completed_fissions=int(observation.completed_fissions),
        growth_generation_one_based=int(observation.growth_generation_one_based),
        generation_local_step=int(observation.generation_local_step),
        batch_step=int(observation.batch_step),
    )


def _path_digest_update(
    digest: Any,
    offset: int,
    kind: str,
    generation: int,
    local_step: int,
    state: NDArray[np.int64],
    score: float,
) -> None:
    digest.update(np.asarray([offset, generation, local_step], dtype="<i8").tobytes())
    digest.update(kind.encode("ascii"))
    digest.update(np.asarray(state, dtype="<i8").tobytes())
    digest.update(np.asarray([score], dtype="<f8").tobytes())


def simulate_branch(
    *,
    restored: RestoredState,
    beta: NDArray[np.floating[Any]],
    definition: SimulationDefinition,
    target_centroid: NDArray[np.floating[Any]],
    event_rng: np.random.Generator,
    trim_rng: np.random.Generator,
    fission_rng: np.random.Generator,
    daughter_rng: np.random.Generator,
    horizon: int = HORIZON,
    threshold: float = TARGET_THRESHOLD,
) -> BranchResult:
    """Resume one branch and emit exactly the next selected-clock observations."""

    definition.validate()
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    matrix = np.asarray(beta, dtype=np.float64)
    if matrix.shape != (100, 100) or not np.isfinite(matrix).all():
        raise ValueError("beta must be a finite 100-by-100 matrix")
    target = np.asarray(target_centroid, dtype=np.float64)
    # Validate the target before any random draw.
    cosine_to_reference(np.asarray([restored.state], dtype=np.int64), target)

    state = np.asarray(restored.state, dtype=np.int64).copy()
    completed = int(restored.completed_fissions)
    if restored.observation_kind == "post_fission":
        generation = int(restored.growth_generation_one_based) + 1
        local_step = 0
    elif restored.observation_kind == "initial_selected_state":
        generation = 1
        local_step = 0
    else:
        generation = int(restored.growth_generation_one_based)
        local_step = int(restored.generation_local_step)

    generated = 0
    updates = 0
    fissions_done = 0
    entered = False
    first_entry: int | None = None
    scores: list[float] = []
    digest = hashlib.sha256()
    digest.update(VERSION.encode("ascii"))
    digest.update(np.asarray(restored.state, dtype="<i8").tobytes())
    digest.update(restored.observation_kind.encode("ascii"))
    terminal = "horizon_completed"

    while generated < horizon:
        mass = int(state.sum())
        if mass <= 0:
            terminal = "empty_state_before_horizon"
            break

        must_fission = mass >= N_MAX or local_step >= MAX_STEPS
        if must_fission:
            child_a, child_b = fission(state, fission_rng)
            state, _ = select_daughter(
                child_a, child_b, definition.daughter_rule, daughter_rng
            )
            completed += 1
            fissions_done += 1
            generated += 1
            score = (
                float(cosine_to_reference(state[None, :], target)[0])
                if int(state.sum()) > 0
                else float("nan")
            )
            scores.append(score)
            _path_digest_update(
                digest, generated, "post_fission", generation, local_step, state, score
            )
            if np.isfinite(score) and score >= threshold and not entered:
                entered = True
                first_entry = generated
            if int(state.sum()) <= 0:
                terminal = "selected_daughter_empty"
                break
            generation += 1
            local_step = 0
            continue

        state, _, _, _, _ = poisson_update(
            state, matrix, definition, event_rng, trim_rng
        )
        local_step += 1
        updates += 1
        generated += 1
        score = (
            float(cosine_to_reference(state[None, :], target)[0])
            if int(state.sum()) > 0
            else float("nan")
        )
        scores.append(score)
        _path_digest_update(
            digest, generated, "molecular_update", generation, local_step, state, score
        )
        if np.isfinite(score) and score >= threshold and not entered:
            entered = True
            first_entry = generated
        if int(state.sum()) <= 0:
            terminal = "extinct_during_growth"
            break

    finite_scores = np.asarray([value for value in scores if np.isfinite(value)])
    return BranchResult(
        entered_basin=entered,
        first_entry_offset_one_based=first_entry,
        selected_observations_generated=generated,
        molecular_updates=updates,
        fissions=fissions_done,
        terminal_status=terminal,
        minimum_target_score=float(np.min(finite_scores))
        if len(finite_scores)
        else None,
        maximum_target_score=float(np.max(finite_scores))
        if len(finite_scores)
        else None,
        final_state_sha256=array_sha256(state),
        path_sha256=digest.hexdigest(),
    )


def corrected_between_state_variance(
    q_hat: NDArray[np.floating[Any]], n: int
) -> dict[str, float]:
    """Subtract the unbiased estimated binomial measurement variance."""

    values = np.asarray(q_hat, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or np.any((values < 0) | (values > 1)):
        raise ValueError("q_hat must contain at least two probabilities")
    if n <= 1:
        raise ValueError("branch count must exceed one")
    observed = float(np.var(values, ddof=1))
    noise = float(np.mean(values * (1.0 - values) / float(n - 1)))
    return {
        "observedBetweenStateVariance": observed,
        "estimatedBinomialNoiseVariance": noise,
        "correctedBetweenStateVariance": observed - noise,
    }
