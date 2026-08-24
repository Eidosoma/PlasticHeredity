"""Independent stochastic GARD growth-fission simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import GardConfig


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

PHASE_INITIAL = 0
PHASE_EXCHANGE = 1
PHASE_FISSION = 2


@dataclass(frozen=True)
class InterventionDecision:
    """One add/delete intervention made immediately after fission."""

    species: int = -1
    delta: int = 0
    score: float = float("nan")


InterventionPolicy = Callable[
    [IntArray, IntArray, IntArray, int], InterventionDecision
]


@dataclass(frozen=True)
class RunTrace:
    """Complete molecular-step trace for one GARD lineage."""

    counts: IntArray
    generations: IntArray
    phases: IntArray
    joins: IntArray
    leaves: IntArray
    intervention_species: IntArray
    intervention_delta: IntArray
    intervention_score: FloatArray
    beta: FloatArray
    seed: int

    def validate(self, config: Optional[GardConfig] = None) -> None:
        n_steps, n_types = self.counts.shape
        expected_shapes = {
            "generations": self.generations.shape,
            "phases": self.phases.shape,
            "joins": self.joins.shape,
            "leaves": self.leaves.shape,
            "intervention_species": self.intervention_species.shape,
            "intervention_delta": self.intervention_delta.shape,
            "intervention_score": self.intervention_score.shape,
        }
        if expected_shapes["generations"] != (n_steps,):
            raise ValueError("generations shape does not match counts")
        if expected_shapes["phases"] != (n_steps,):
            raise ValueError("phases shape does not match counts")
        if expected_shapes["joins"] != (n_steps, n_types):
            raise ValueError("joins shape does not match counts")
        if expected_shapes["leaves"] != (n_steps, n_types):
            raise ValueError("leaves shape does not match counts")
        for name in ("intervention_species", "intervention_delta", "intervention_score"):
            if expected_shapes[name] != (n_steps,):
                raise ValueError(f"{name} shape does not match counts")
        if self.beta.shape != (n_types, n_types):
            raise ValueError("beta must be square with one row per molecular type")
        if np.any(self.counts < 0) or np.any(self.counts.sum(axis=1) <= 0):
            raise ValueError("trace contains an invalid assembly")
        if config is not None:
            if n_types != config.n_types:
                raise ValueError("trace molecular type count does not match config")
            if np.any(self.counts.sum(axis=1) > config.max_size):
                raise ValueError("trace exceeds configured maximum assembly size")

    @property
    def relative_counts(self) -> FloatArray:
        totals = self.counts.sum(axis=1, keepdims=True)
        return self.counts / totals

    @property
    def net_flux(self) -> IntArray:
        return self.joins - self.leaves

    def generation_end_indices(self) -> IntArray:
        """Last pre-fission exchange state in each generation."""

        indices = []
        for generation in np.unique(self.generations):
            members = np.flatnonzero(
                (self.generations == generation) & (self.phases == PHASE_EXCHANGE)
            )
            if members.size:
                indices.append(int(members[-1]))
        return np.asarray(indices, dtype=np.int64)


def sample_beta(config: GardConfig, rng: np.random.Generator) -> FloatArray:
    """Sample a directed catalytic matrix from the reported lognormal law."""

    return rng.lognormal(
        mean=config.beta_log_mean,
        sigma=config.beta_log_sigma,
        size=(config.n_types, config.n_types),
    )


def initial_assembly(config: GardConfig, rng: np.random.Generator) -> IntArray:
    """Sample the stated distinct initial molecular types without replacement."""

    counts = np.zeros(config.n_types, dtype=np.int64)
    selected = rng.choice(config.n_types, size=config.initial_size, replace=False)
    counts[selected] = 1
    return counts


def gard_propensities(
    counts: IntArray, beta: FloatArray, config: GardConfig
) -> Tuple[FloatArray, FloatArray]:
    """Forward and backward GARD propensities for one assembly state."""

    size = int(counts.sum())
    if size <= 0:
        raise ValueError("assembly must not be empty")
    catalytic_factor = 1.0 + np.einsum(
        "ij,j->i", beta, counts / size, optimize=True
    )
    joins = (
        config.forward_rate
        * config.environment_concentration
        * size
        * catalytic_factor
    )
    leaves = config.backward_rate * counts * catalytic_factor
    return np.maximum(joins, 0.0), np.maximum(leaves, 0.0)


def _cap_events_without_replacement(
    events: IntArray, capacity: int, rng: np.random.Generator
) -> IntArray:
    """Randomly retain at most ``capacity`` sampled events."""

    total = int(events.sum())
    if total <= capacity:
        return events
    if capacity <= 0:
        return np.zeros_like(events)
    retained = np.zeros_like(events)
    remaining_total = total
    remaining_capacity = capacity
    for index in range(events.size - 1):
        available = int(events[index])
        if available and remaining_capacity:
            chosen = int(
                rng.hypergeometric(
                    ngood=available,
                    nbad=remaining_total - available,
                    nsample=remaining_capacity,
                )
            )
            retained[index] = chosen
            remaining_capacity -= chosen
        remaining_total -= available
    retained[-1] = remaining_capacity
    return retained


def poisson_exchange_step(
    counts: IntArray,
    beta: FloatArray,
    config: GardConfig,
    rng: np.random.Generator,
) -> Tuple[IntArray, IntArray, IntArray]:
    """Perform one vector Poisson tau-leap, respecting finite molecule counts."""

    join_rates, leave_rates = gard_propensities(counts, beta, config)
    proposed_leaves = rng.poisson(leave_rates * config.tau).astype(np.int64)
    leaves = np.minimum(proposed_leaves, counts)
    after_leaves = counts - leaves

    proposed_joins = rng.poisson(join_rates * config.tau).astype(np.int64)
    capacity = config.max_size - int(after_leaves.sum())
    joins = _cap_events_without_replacement(proposed_joins, capacity, rng)
    updated = after_leaves + joins

    if updated.sum() == 0:
        rescue = int(np.argmax(join_rates))
        updated[rescue] = 1
        joins[rescue] += 1
    return updated, joins, leaves


def fission(counts: IntArray, rng: np.random.Generator) -> IntArray:
    """Select one daughter using the paper's component-wise Binomial(ni, 0.5)."""

    daughter = rng.binomial(counts, 0.5).astype(np.int64)
    if daughter.sum() == 0:
        available = np.flatnonzero(counts)
        daughter[int(rng.choice(available))] = 1
    return daughter


def _apply_decision(
    daughter: IntArray, decision: InterventionDecision, max_size: int
) -> IntArray:
    result = daughter.copy()
    if decision.delta == 0:
        return result
    if not 0 <= decision.species < result.size:
        raise ValueError("intervention species is outside the molecular repertoire")
    if decision.delta not in (-1, 1):
        raise ValueError("intervention delta must be -1, 0, or 1")
    if decision.delta < 0 and result[decision.species] == 0:
        raise ValueError("cannot delete an absent molecule")
    if decision.delta > 0 and result.sum() >= max_size:
        raise ValueError("cannot add beyond max_size")
    result[decision.species] += decision.delta
    if result.sum() <= 0:
        raise ValueError("intervention cannot empty the assembly")
    return result


def simulate_gard(
    config: GardConfig,
    seed: int,
    *,
    beta: Optional[FloatArray] = None,
    intervention: Optional[InterventionPolicy] = None,
) -> RunTrace:
    """Simulate one single-lineage GARD run for ``config.generations`` cycles."""

    config.validate()
    seed_sequence = np.random.SeedSequence(seed)
    beta_seed, initial_seed, dynamics_seed, fission_seed = seed_sequence.spawn(4)
    beta_rng = np.random.default_rng(beta_seed)
    initial_rng = np.random.default_rng(initial_seed)
    dynamics_rng = np.random.default_rng(dynamics_seed)
    fission_rng = np.random.default_rng(fission_seed)

    if beta is None:
        beta_value = sample_beta(config, beta_rng)
    else:
        beta_value = np.asarray(beta, dtype=float).copy()
        if beta_value.shape != (config.n_types, config.n_types):
            raise ValueError("provided beta has the wrong shape")
        if np.any(beta_value < 0) or not np.all(np.isfinite(beta_value)):
            raise ValueError("provided beta must be finite and non-negative")

    current = initial_assembly(config, initial_rng)
    zero_flux = np.zeros(config.n_types, dtype=np.int64)
    count_rows = [current.copy()]
    generations = [0]
    phases = [PHASE_INITIAL]
    join_rows = [zero_flux.copy()]
    leave_rows = [zero_flux.copy()]
    intervention_species = [-1]
    intervention_delta = [0]
    intervention_score = [float("nan")]

    for generation in range(config.generations):
        for _ in range(config.max_steps_per_generation):
            if current.sum() >= config.max_size:
                break
            updated, joins, leaves = poisson_exchange_step(
                current, beta_value, config, dynamics_rng
            )
            if np.array_equal(updated, current):
                # A zero-event Poisson leap is a valid molecular-time step.
                if not config.record_zero_event_steps:
                    continue
            current = updated
            count_rows.append(current.copy())
            generations.append(generation)
            phases.append(PHASE_EXCHANGE)
            join_rows.append(joins)
            leave_rows.append(leaves)
            intervention_species.append(-1)
            intervention_delta.append(0)
            intervention_score.append(float("nan"))

        parent = current.copy()
        daughter = fission(parent, fission_rng)
        history = np.asarray(count_rows, dtype=np.int64)
        decision = (
            intervention(parent, daughter.copy(), history, generation)
            if intervention is not None
            else InterventionDecision()
        )
        current = _apply_decision(daughter, decision, config.max_size)
        count_rows.append(current.copy())
        generations.append(generation)
        phases.append(PHASE_FISSION)
        join_rows.append(zero_flux.copy())
        leave_rows.append(zero_flux.copy())
        intervention_species.append(decision.species)
        intervention_delta.append(decision.delta)
        intervention_score.append(decision.score)

    trace = RunTrace(
        counts=np.asarray(count_rows, dtype=np.int64),
        generations=np.asarray(generations, dtype=np.int64),
        phases=np.asarray(phases, dtype=np.int64),
        joins=np.asarray(join_rows, dtype=np.int64),
        leaves=np.asarray(leave_rows, dtype=np.int64),
        intervention_species=np.asarray(intervention_species, dtype=np.int64),
        intervention_delta=np.asarray(intervention_delta, dtype=np.int64),
        intervention_score=np.asarray(intervention_score, dtype=float),
        beta=beta_value,
        seed=seed,
    )
    trace.validate(config)
    return trace
