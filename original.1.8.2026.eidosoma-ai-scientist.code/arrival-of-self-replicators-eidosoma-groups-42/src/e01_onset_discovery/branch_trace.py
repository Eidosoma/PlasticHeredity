"""Exact trace-emitting replay of the frozen S19-L30/L31 H8 branches."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from e01_latent_timebase.core import (
    MAX_STEPS,
    N_MAX,
    SimulationDefinition,
    fission,
    poisson_update,
    rates,
    select_daughter,
)
from e01_onset_discovery.empirical_committor import (
    TARGET_THRESHOLD,
    BranchResult,
    RestoredState,
    _path_digest_update,
    array_sha256,
    cosine_to_reference,
)


@dataclass(frozen=True, slots=True)
class TraceObservation:
    offset: int
    observation_kind: str
    generation: int
    generation_local_step: int
    state: tuple[int, ...]
    target_score: float
    ordinary_adjacent_h: float
    entered_at_or_before: bool
    entered_at_offset: bool
    mass: int
    diversity: float
    entropy: float
    concentration: float
    join_share_maximum: float
    join_share_entropy: float
    loss_share_maximum: float
    loss_share_entropy: float
    boost_maximum: float
    boost_sd: float
    nonzero_reaction_types: int
    gross_sampled_events: int
    overshoot: int
    exposure: float


@dataclass(frozen=True, slots=True)
class TraceBranchResult:
    compact: BranchResult
    observations: tuple[TraceObservation, ...]


def _entropy(values: NDArray[np.float64]) -> float:
    positive = values[values > 0]
    if not len(positive):
        return 0.0
    probability = positive / positive.sum()
    return float(-np.sum(probability * np.log(probability)) / math.log(max(2, len(values))))


def _ordinary_h(left: NDArray[np.int64], right: NDArray[np.int64]) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return float("nan")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _trace_observation(
    *,
    offset: int,
    kind: str,
    generation: int,
    local_step: int,
    state: NDArray[np.int64],
    previous: NDArray[np.int64],
    target: NDArray[np.float64],
    beta: NDArray[np.float64],
    entered_before: bool,
    nonzero_reaction_types: int,
    gross_sampled_events: int,
    overshoot: int,
    exposure: float,
    threshold: float,
) -> TraceObservation:
    mass = int(state.sum())
    score = (
        float(cosine_to_reference(state[None, :], target)[0])
        if mass > 0
        else float("nan")
    )
    entered_now = bool(np.isfinite(score) and score >= threshold and not entered_before)
    if mass > 0:
        composition = state.astype(np.float64) / mass
        joins, losses = rates(state, beta)
        join_share = joins / joins.sum()
        loss_share = losses / max(float(losses.sum()), 1e-300)
        boost = 1.0 + beta @ composition
        positive = composition[composition > 0]
        entropy = float(
            -np.sum(positive * np.log(positive)) / math.log(len(composition))
        )
        diversity = float(np.count_nonzero(state) / len(state))
        concentration = float(np.sum(composition * composition))
        join_maximum = float(np.max(join_share))
        join_entropy = _entropy(join_share)
        loss_maximum = float(np.max(loss_share))
        loss_entropy = _entropy(loss_share)
        boost_maximum = float(np.max(boost))
        boost_sd = float(np.std(boost, ddof=0))
    else:
        diversity = entropy = concentration = float("nan")
        join_maximum = join_entropy = loss_maximum = loss_entropy = float("nan")
        boost_maximum = boost_sd = float("nan")
    return TraceObservation(
        offset=offset,
        observation_kind=kind,
        generation=generation,
        generation_local_step=local_step,
        state=tuple(map(int, state)),
        target_score=score,
        ordinary_adjacent_h=_ordinary_h(previous, state),
        entered_at_or_before=entered_before or entered_now,
        entered_at_offset=entered_now,
        mass=mass,
        diversity=diversity,
        entropy=entropy,
        concentration=concentration,
        join_share_maximum=join_maximum,
        join_share_entropy=join_entropy,
        loss_share_maximum=loss_maximum,
        loss_share_entropy=loss_entropy,
        boost_maximum=boost_maximum,
        boost_sd=boost_sd,
        nonzero_reaction_types=nonzero_reaction_types,
        gross_sampled_events=gross_sampled_events,
        overshoot=overshoot,
        exposure=exposure,
    )


def simulate_branch_trace(
    *,
    restored: RestoredState,
    beta: NDArray[np.floating],
    definition: SimulationDefinition,
    target_centroid: NDArray[np.floating],
    event_rng: np.random.Generator,
    trim_rng: np.random.Generator,
    fission_rng: np.random.Generator,
    daughter_rng: np.random.Generator,
    horizon: int = 8,
    threshold: float = TARGET_THRESHOLD,
) -> TraceBranchResult:
    """Replay the exact branch while retaining every generated observation."""

    definition.validate()
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    matrix = np.asarray(beta, dtype=np.float64)
    target = np.asarray(target_centroid, dtype=np.float64)
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
    generated = updates = fissions_done = 0
    entered = False
    first_entry: int | None = None
    scores: list[float] = []
    observations: list[TraceObservation] = []
    digest = hashlib.sha256()
    from e01_onset_discovery.empirical_committor import VERSION

    digest.update(VERSION.encode("ascii"))
    digest.update(np.asarray(restored.state, dtype="<i8").tobytes())
    digest.update(restored.observation_kind.encode("ascii"))
    terminal = "horizon_completed"
    while generated < horizon:
        mass = int(state.sum())
        if mass <= 0:
            terminal = "empty_state_before_horizon"
            break
        previous = state.copy()
        must_fission = mass >= N_MAX or local_step >= MAX_STEPS
        nonzero = gross = overshoot = 0
        exposure = float("nan")
        if must_fission:
            child_a, child_b = fission(state, fission_rng)
            state, _ = select_daughter(
                child_a, child_b, definition.daughter_rule, daughter_rng
            )
            completed += 1
            fissions_done += 1
            generated += 1
            kind = "post_fission"
        else:
            state, nonzero, gross, overshoot, exposure = poisson_update(
                state, matrix, definition, event_rng, trim_rng
            )
            local_step += 1
            updates += 1
            generated += 1
            kind = "molecular_update"
        observation = _trace_observation(
            offset=generated,
            kind=kind,
            generation=generation,
            local_step=local_step,
            state=state,
            previous=previous,
            target=target,
            beta=matrix,
            entered_before=entered,
            nonzero_reaction_types=nonzero,
            gross_sampled_events=gross,
            overshoot=overshoot,
            exposure=exposure,
            threshold=threshold,
        )
        observations.append(observation)
        scores.append(observation.target_score)
        _path_digest_update(
            digest,
            generated,
            kind,
            generation,
            local_step,
            state,
            observation.target_score,
        )
        if observation.entered_at_offset:
            entered = True
            first_entry = generated
        if int(state.sum()) <= 0:
            terminal = (
                "selected_daughter_empty"
                if kind == "post_fission"
                else "extinct_during_growth"
            )
            break
        if kind == "post_fission":
            generation += 1
            local_step = 0
    finite = np.asarray([value for value in scores if np.isfinite(value)])
    compact = BranchResult(
        entered_basin=entered,
        first_entry_offset_one_based=first_entry,
        selected_observations_generated=generated,
        molecular_updates=updates,
        fissions=fissions_done,
        terminal_status=terminal,
        minimum_target_score=float(np.min(finite)) if len(finite) else None,
        maximum_target_score=float(np.max(finite)) if len(finite) else None,
        final_state_sha256=array_sha256(state),
        path_sha256=digest.hexdigest(),
    )
    return TraceBranchResult(compact=compact, observations=tuple(observations))
