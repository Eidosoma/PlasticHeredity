"""Fission-clock branch replay and online repeated-recurrence scoring.

The primary event is deliberately a process rather than a destination.  A
future post-fission state is a certified return only when it is strictly near
an eligible nonadjacent past boundary and the immediately preceding boundary
was outside that same reference neighbourhood.  Consecutive residence near a
reference therefore counts once, not as repeated recurrence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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
from e01_onset_discovery.empirical_committor import RestoredState, array_sha256
from e01_onset_discovery.recurrence_inheritance import cosine_h

VERSION = "E01-S19-L41-FISSION-CLOCK-REPEATED-CROSS-GENERATION-RECURRENCE-v1.0.0"


@dataclass(frozen=True, slots=True)
class FissionClockTrace:
    future_states: tuple[tuple[int, ...], ...]
    future_generations: tuple[int, ...]
    future_offsets_one_based: tuple[int, ...]
    parent_daughter_h: tuple[float, ...]
    selected_observations_generated: int
    molecular_updates: int
    fissions: int
    terminal_status: str
    final_state_sha256: str
    path_sha256: str


@dataclass(frozen=True, slots=True)
class RepeatedRecurrenceResult:
    event: bool
    certification_boundary_one_based: int | None
    certification_generation: int | None
    certification_offset_one_based: int | None
    first_return_boundary_one_based: int | None
    first_return_generation: int | None
    first_return_offset_one_based: int | None
    future_boundary_count: int
    return_boundary_count: int
    qualifying_pair_count: int
    distinct_reference_generation_count: int
    membership_boundary_count: int
    membership_pair_count: int
    membership_only_event: bool
    maximum_return_h: float | None
    maximum_membership_h: float | None
    return_boundary_flags: tuple[bool, ...]
    membership_boundary_flags: tuple[bool, ...]


def _digest_update(
    digest: object,
    offset: int,
    kind: str,
    completed_fissions: int,
    growth_generation: int,
    local_step: int,
    state: NDArray[np.int64],
) -> None:
    digest.update(
        np.asarray(
            [offset, completed_fissions, growth_generation, local_step], dtype="<i8"
        ).tobytes()
    )
    digest.update(kind.encode("ascii"))
    digest.update(np.asarray(state, dtype="<i8").tobytes())


def simulate_fission_clock(
    *,
    restored: RestoredState,
    beta: NDArray[np.floating],
    definition: SimulationDefinition,
    event_rng: np.random.Generator,
    trim_rng: np.random.Generator,
    fission_rng: np.random.Generator,
    daughter_rng: np.random.Generator,
    future_fissions: int,
) -> FissionClockTrace:
    """Resume a state until a fixed number of future fission opportunities."""

    definition.validate()
    if future_fissions <= 0:
        raise ValueError("future_fissions must be positive")
    matrix = np.asarray(beta, dtype=np.float64)
    if matrix.shape != (100, 100) or not np.isfinite(matrix).all():
        raise ValueError("beta must be a finite 100-by-100 matrix")
    state = np.asarray(restored.state, dtype=np.int64).copy()
    if state.shape != (100,) or np.any(state < 0) or int(state.sum()) <= 0:
        raise ValueError("restored state must be a positive-mass 100-vector")

    completed = int(restored.completed_fissions)
    if restored.observation_kind == "post_fission":
        growth_generation = int(restored.growth_generation_one_based) + 1
        local_step = 0
    elif restored.observation_kind == "initial_selected_state":
        growth_generation = 1
        local_step = 0
    elif restored.observation_kind == "molecular_update":
        growth_generation = int(restored.growth_generation_one_based)
        local_step = int(restored.generation_local_step)
    else:
        raise ValueError("unsupported restored observation kind")

    future_states: list[tuple[int, ...]] = []
    future_generations: list[int] = []
    future_offsets: list[int] = []
    inheritance: list[float] = []
    selected = updates = fissions_done = 0
    terminal = "fission_horizon_completed"
    digest = hashlib.sha256()
    digest.update(VERSION.encode("ascii"))
    digest.update(np.asarray(restored.state, dtype="<i8").tobytes())
    digest.update(restored.observation_kind.encode("ascii"))

    while fissions_done < future_fissions:
        mass = int(state.sum())
        if mass <= 0:
            terminal = "empty_state_before_fission_horizon"
            break
        previous = state.copy()
        if mass >= N_MAX or local_step >= MAX_STEPS:
            child_a, child_b = fission(state, fission_rng)
            state, _ = select_daughter(
                child_a, child_b, definition.daughter_rule, daughter_rng
            )
            completed += 1
            fissions_done += 1
            selected += 1
            score = cosine_h(previous, state) if int(state.sum()) > 0 else float("nan")
            future_states.append(tuple(map(int, state)))
            future_generations.append(completed)
            future_offsets.append(selected)
            inheritance.append(float(score))
            _digest_update(
                digest,
                selected,
                "post_fission",
                completed,
                growth_generation,
                local_step,
                state,
            )
            if int(state.sum()) <= 0:
                terminal = "selected_daughter_empty"
                break
            growth_generation += 1
            local_step = 0
            continue

        state, _, _, _, _ = poisson_update(
            state, matrix, definition, event_rng, trim_rng
        )
        local_step += 1
        updates += 1
        selected += 1
        _digest_update(
            digest,
            selected,
            "molecular_update",
            completed,
            growth_generation,
            local_step,
            state,
        )
        if int(state.sum()) <= 0:
            terminal = "extinct_during_growth"
            break

    return FissionClockTrace(
        future_states=tuple(future_states),
        future_generations=tuple(future_generations),
        future_offsets_one_based=tuple(future_offsets),
        parent_daughter_h=tuple(inheritance),
        selected_observations_generated=selected,
        molecular_updates=updates,
        fissions=fissions_done,
        terminal_status=terminal,
        final_state_sha256=array_sha256(state),
        path_sha256=digest.hexdigest(),
    )


def _validate_history(
    states: NDArray[np.number], generations: NDArray[np.integer], *, name: str
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    values = np.asarray(states, dtype=np.int64)
    generation_values = np.asarray(generations, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError(f"{name} states must be two-dimensional")
    if len(values) != len(generation_values):
        raise ValueError(f"{name} states and generations must align")
    if np.any(values < 0) or (len(values) and np.any(values.sum(axis=1) <= 0)):
        raise ValueError(f"{name} states must have positive mass")
    if len(generation_values) and np.any(np.diff(generation_values) <= 0):
        raise ValueError(f"{name} generations must be strictly increasing")
    return values, generation_values


def score_repeated_recurrence(
    *,
    prefix_states: NDArray[np.number],
    prefix_generations: NDArray[np.integer],
    future_states: NDArray[np.number],
    future_generations: NDArray[np.integer],
    future_offsets_one_based: NDArray[np.integer],
    threshold: float = 0.9,
    minimum_generation_gap: int = 2,
    required_return_boundaries: int = 2,
) -> RepeatedRecurrenceResult:
    """Score repeated far-to-near returns using only sequentially available states."""

    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    if minimum_generation_gap < 2:
        raise ValueError("minimum generation gap must preserve an intervening generation")
    if required_return_boundaries < 2:
        raise ValueError("repeated recurrence requires at least two return boundaries")
    prefix, prefix_g = _validate_history(prefix_states, prefix_generations, name="prefix")
    future, future_g = _validate_history(future_states, future_generations, name="future")
    offsets = np.asarray(future_offsets_one_based, dtype=np.int64)
    if prefix.shape[1:] != future.shape[1:]:
        raise ValueError("prefix and future states must have matching features")
    if len(offsets) != len(future):
        raise ValueError("future offsets must align")
    if len(offsets) and (np.any(offsets <= 0) or np.any(np.diff(offsets) <= 0)):
        raise ValueError("future offsets must be positive and strictly increasing")
    if len(prefix_g) and len(future_g) and future_g[0] <= prefix_g[-1]:
        raise ValueError("future generations must follow the prefix")

    history_states = [row.copy() for row in prefix]
    history_generations = [int(value) for value in prefix_g]
    return_flags: list[bool] = []
    membership_flags: list[bool] = []
    qualifying_pairs = membership_pairs = 0
    reference_generations: set[int] = set()
    maximum_return: float | None = None
    maximum_membership: float | None = None
    first_index: int | None = None
    certification_index: int | None = None

    for future_index, (state, generation) in enumerate(
        zip(future, future_g, strict=True)
    ):
        previous = history_states[-1] if history_states else None
        local_return_scores: list[tuple[float, int]] = []
        local_membership_scores: list[tuple[float, int]] = []
        for reference, reference_generation in zip(
            history_states, history_generations, strict=True
        ):
            if int(generation) - reference_generation < minimum_generation_gap:
                continue
            score = cosine_h(state, reference)
            if not np.isfinite(score) or score <= threshold:
                continue
            local_membership_scores.append((score, reference_generation))
            if previous is not None:
                previous_score = cosine_h(previous, reference)
                if np.isfinite(previous_score) and previous_score <= threshold:
                    local_return_scores.append((score, reference_generation))

        membership = bool(local_membership_scores)
        returned = bool(local_return_scores)
        membership_flags.append(membership)
        return_flags.append(returned)
        membership_pairs += len(local_membership_scores)
        qualifying_pairs += len(local_return_scores)
        if local_membership_scores:
            local_max = max(score for score, _ in local_membership_scores)
            maximum_membership = (
                local_max
                if maximum_membership is None
                else max(maximum_membership, local_max)
            )
        if local_return_scores:
            local_max = max(score for score, _ in local_return_scores)
            maximum_return = (
                local_max if maximum_return is None else max(maximum_return, local_max)
            )
            reference_generations.update(
                reference_generation for _, reference_generation in local_return_scores
            )
            if first_index is None:
                first_index = future_index
            if (
                sum(return_flags) >= required_return_boundaries
                and certification_index is None
            ):
                certification_index = future_index

        history_states.append(state.copy())
        history_generations.append(int(generation))

    return RepeatedRecurrenceResult(
        event=certification_index is not None,
        certification_boundary_one_based=(
            certification_index + 1 if certification_index is not None else None
        ),
        certification_generation=(
            int(future_g[certification_index]) if certification_index is not None else None
        ),
        certification_offset_one_based=(
            int(offsets[certification_index]) if certification_index is not None else None
        ),
        first_return_boundary_one_based=(first_index + 1 if first_index is not None else None),
        first_return_generation=(int(future_g[first_index]) if first_index is not None else None),
        first_return_offset_one_based=(int(offsets[first_index]) if first_index is not None else None),
        future_boundary_count=len(future),
        return_boundary_count=int(sum(return_flags)),
        qualifying_pair_count=qualifying_pairs,
        distinct_reference_generation_count=len(reference_generations),
        membership_boundary_count=int(sum(membership_flags)),
        membership_pair_count=membership_pairs,
        membership_only_event=int(sum(membership_flags)) >= required_return_boundaries,
        maximum_return_h=maximum_return,
        maximum_membership_h=maximum_membership,
        return_boundary_flags=tuple(return_flags),
        membership_boundary_flags=tuple(membership_flags),
    )
