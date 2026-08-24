"""Functional-regime summaries on frozen GARD fission-clock branches.

The module adds observability to the already frozen L41 simulator without
changing its transition kernel or random-number calls.  It separates three
objects: molecular composition, matrix-conditioned catalytic function, and
growth/division performance over a fission interval.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations

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
from e01_onset_discovery.empirical_committor import RestoredState, array_sha256
from e01_onset_discovery.recurrence_inheritance import cosine_h

FROZEN_PATH_VERSION = (
    "E01-S19-L41-FISSION-CLOCK-REPEATED-CROSS-GENERATION-RECURRENCE-v1.0.0"
)


@dataclass(frozen=True, slots=True)
class FissionInterval:
    """One growth interval followed by its selected-daughter boundary."""

    boundary_one_based: int
    generation: int
    selected_offset_one_based: int
    start_state: tuple[int, ...]
    pre_fission_state: tuple[int, ...]
    daughter_state: tuple[int, ...]
    parent_daughter_h: float
    molecular_updates: int
    nonzero_reaction_type_count: int
    gross_sampled_event_count: int
    maximum_overshoot: int
    mean_exposure: float
    pre_fission_mass: int
    daughter_mass: int
    complete_growth_interval: bool


@dataclass(frozen=True, slots=True)
class FunctionalFissionTrace:
    intervals: tuple[FissionInterval, ...]
    selected_observations_generated: int
    molecular_updates: int
    fissions: int
    terminal_status: str
    final_state_sha256: str
    path_sha256: str

    @property
    def future_states(self) -> tuple[tuple[int, ...], ...]:
        return tuple(interval.daughter_state for interval in self.intervals)

    @property
    def parent_daughter_h(self) -> tuple[float, ...]:
        return tuple(interval.parent_daughter_h for interval in self.intervals)


@dataclass(frozen=True, slots=True)
class FunctionalProfile:
    catalytic_activation: NDArray[np.float64]
    expected_net_exchange: NDArray[np.float64]


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
            [offset, completed_fissions, growth_generation, local_step],
            dtype="<i8",
        ).tobytes()
    )
    digest.update(kind.encode("ascii"))
    digest.update(np.asarray(state, dtype="<i8").tobytes())


def simulate_functional_fission_clock(
    *,
    restored: RestoredState,
    beta: NDArray[np.floating],
    definition: SimulationDefinition,
    event_rng: np.random.Generator,
    trim_rng: np.random.Generator,
    fission_rng: np.random.Generator,
    daughter_rng: np.random.Generator,
    future_fissions: int,
) -> FunctionalFissionTrace:
    """Replay the exact frozen fission clock and retain interval summaries."""

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
        complete_interval = True
    elif restored.observation_kind == "initial_selected_state":
        growth_generation = 1
        local_step = 0
        complete_interval = True
    elif restored.observation_kind == "molecular_update":
        growth_generation = int(restored.growth_generation_one_based)
        local_step = int(restored.generation_local_step)
        complete_interval = False
    else:
        raise ValueError("unsupported restored observation kind")

    selected = updates = fissions_done = 0
    interval_updates = interval_nonzero = interval_gross = 0
    interval_overshoot = 0
    interval_exposure_sum = 0.0
    interval_start = state.copy()
    records: list[FissionInterval] = []
    terminal = "fission_horizon_completed"
    digest = hashlib.sha256()
    digest.update(FROZEN_PATH_VERSION.encode("ascii"))
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
            daughter, _ = select_daughter(
                child_a, child_b, definition.daughter_rule, daughter_rng
            )
            state = daughter
            completed += 1
            fissions_done += 1
            selected += 1
            score = (
                cosine_h(previous, state)
                if int(state.sum()) > 0
                else float("nan")
            )
            records.append(
                FissionInterval(
                    boundary_one_based=fissions_done,
                    generation=completed,
                    selected_offset_one_based=selected,
                    start_state=tuple(map(int, interval_start)),
                    pre_fission_state=tuple(map(int, previous)),
                    daughter_state=tuple(map(int, state)),
                    parent_daughter_h=float(score),
                    molecular_updates=interval_updates,
                    nonzero_reaction_type_count=interval_nonzero,
                    gross_sampled_event_count=interval_gross,
                    maximum_overshoot=interval_overshoot,
                    mean_exposure=(
                        interval_exposure_sum / interval_updates
                        if interval_updates
                        else float("nan")
                    ),
                    pre_fission_mass=int(previous.sum()),
                    daughter_mass=int(state.sum()),
                    complete_growth_interval=complete_interval,
                )
            )
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
            interval_start = state.copy()
            interval_updates = interval_nonzero = interval_gross = 0
            interval_overshoot = 0
            interval_exposure_sum = 0.0
            complete_interval = True
            continue

        state, nonzero, gross, overshoot, exposure = poisson_update(
            state, matrix, definition, event_rng, trim_rng
        )
        local_step += 1
        updates += 1
        selected += 1
        interval_updates += 1
        interval_nonzero += int(nonzero)
        interval_gross += int(gross)
        interval_overshoot = max(interval_overshoot, int(overshoot))
        interval_exposure_sum += float(exposure)
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

    return FunctionalFissionTrace(
        intervals=tuple(records),
        selected_observations_generated=selected,
        molecular_updates=updates,
        fissions=fissions_done,
        terminal_status=terminal,
        final_state_sha256=array_sha256(state),
        path_sha256=digest.hexdigest(),
    )


def functional_profile(
    state: NDArray[np.integer], beta: NDArray[np.floating]
) -> FunctionalProfile:
    """Return two source-derived, matrix-conditioned functional profiles."""

    counts = np.asarray(state, dtype=np.int64)
    matrix = np.asarray(beta, dtype=np.float64)
    if counts.ndim != 1 or matrix.shape != (len(counts), len(counts)):
        raise ValueError("state and beta dimensions must align")
    mass = int(counts.sum())
    if mass <= 0 or np.any(counts < 0):
        raise ValueError("functional profiles require a positive composition")
    composition = counts.astype(np.float64) / mass
    activation = matrix @ composition
    joins, losses = rates(counts, matrix)
    net = joins - losses
    if not np.isfinite(activation).all() or not np.isfinite(net).all():
        raise ValueError("functional profile became nonfinite")
    return FunctionalProfile(
        catalytic_activation=activation,
        expected_net_exchange=net,
    )


def cosine(left: NDArray[np.floating], right: NDArray[np.floating]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def mean_pairwise_cosine(vectors: NDArray[np.floating]) -> float:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        return float("nan")
    scores = [cosine(values[i], values[j]) for i, j in combinations(range(len(values)), 2)]
    finite = np.asarray(scores, dtype=np.float64)
    return float(np.mean(finite)) if np.isfinite(finite).all() else float("nan")


def growth_signature(interval: FissionInterval) -> NDArray[np.float64]:
    """Frozen growth/division phenotype, prior to development-only scaling."""

    updates = float(interval.molecular_updates)
    return np.asarray(
        [
            np.log1p(updates),
            np.log1p(float(interval.gross_sampled_event_count)),
            float(interval.nonzero_reaction_type_count) / max(updates, 1.0),
            float(interval.daughter_mass) / max(float(interval.pre_fission_mass), 1.0),
        ],
        dtype=np.float64,
    )


def mean_pairwise_distance(
    vectors: NDArray[np.floating], scale: NDArray[np.floating]
) -> float:
    values = np.asarray(vectors, dtype=np.float64)
    denominator = np.asarray(scale, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or denominator.shape != (values.shape[1],):
        return float("nan")
    safe = np.where(denominator > 0, denominator, 1.0)
    distances = [
        float(np.linalg.norm((values[i] - values[j]) / safe) / np.sqrt(values.shape[1]))
        for i, j in combinations(range(len(values)), 2)
    ]
    return float(np.mean(distances))
