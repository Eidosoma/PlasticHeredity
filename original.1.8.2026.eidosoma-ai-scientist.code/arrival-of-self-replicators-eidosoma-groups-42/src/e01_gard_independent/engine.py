"""Independent, branch-explicit integer-state GARD simulator.

This implementation is derived from the frozen equations and S02/S03 branch
registry.  It uses NumPy categorical, Poisson, binomial, and multivariate-
hypergeometric primitives and does not import the S04 compatibility engine.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import count

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .records import (
    EventLog,
    FissionLog,
    GenerationResult,
    GrowthResult,
    LineageResult,
    PropensityArrays,
)
from .rng import RNGInput, RNGStreams, generator_state_sha256
from .specification import (
    CatalyticMatrixBranch,
    ClockSemantics,
    DaughterSelection,
    FissionSemantics,
    GardSpecification,
    GrowthBoundary,
    InitialStateSemantics,
    LossNonnegativity,
    MaxStepsSemantics,
    PostFissionSemantics,
    UpdateKernel,
    ZeroPropensitySemantics,
)

EVENT_RECORD_SCHEMA = "eidosoma.e01.s05_event_log.v1"
FISSION_RECORD_SCHEMA = "eidosoma.e01.s05_fission_log.v1"


class IndependentGardError(ValueError):
    """Base error for invalid independent-engine operations."""


class ZeroPropensityError(IndependentGardError):
    """A nonempty state has no possible update under the explicit rates."""


class BatchLossError(IndependentGardError):
    """A vector-Poisson loss draw exceeded the available count."""


class EmptyDaughterError(IndependentGardError):
    """The selected-daughter rule rejected an empty daughter."""


class GrowthLimitError(IndependentGardError):
    """The explicit max-step branch raises at its bound."""

    def __init__(self, result: GrowthResult):
        super().__init__(
            f"Growth limit reached under specification {result.specification_id}."
        )
        self.result = result


def integer_state(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    """Validate and copy a nonempty, nonnegative integer count vector."""

    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise IndependentGardError(f"{name} must be a nonempty one-dimensional vector.")
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise IndependentGardError(f"{name} must contain numeric counts.") from exc
    if not np.all(np.isfinite(numeric)):
        raise IndependentGardError(f"{name} must contain finite counts.")
    if np.any(numeric < 0) or not np.all(numeric == np.floor(numeric)):
        raise IndependentGardError(
            f"{name} must contain nonnegative integer molecule counts."
        )
    return numeric.astype(np.int64, copy=True)


def _matrix(values: ArrayLike, *, n_species: int) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (n_species, n_species):
        raise IndependentGardError(f"beta must have shape ({n_species}, {n_species}).")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise IndependentGardError("beta must be finite and nonnegative.")
    return matrix


def generate_catalytic_matrix(
    specification: GardSpecification,
    rng: RNGInput,
) -> NDArray[np.float64]:
    """Draw ``exp(A + sigma*epsilon)`` with a caller-owned modern generator."""

    standard_normals = rng.generator.standard_normal(
        (specification.n_species, specification.n_species)
    )
    with np.errstate(over="raise", invalid="raise"):
        try:
            return np.exp(
                specification.beta_a + specification.beta_sigma * standard_normals
            ).astype(np.float64, copy=False)
        except FloatingPointError as exc:
            raise IndependentGardError(
                "Catalytic matrix generation overflowed."
            ) from exc


def initialize_state(
    specification: GardSpecification,
    rng: RNGInput,
) -> tuple[int, ...]:
    """Construct an explicit paper-like or historical-comparison initial branch."""

    if (
        specification.initial_state_semantics
        is InitialStateSemantics.DISTINCT_TYPES_WITHOUT_REPLACEMENT
    ):
        indices = rng.generator.choice(
            specification.n_species,
            size=specification.n_min,
            replace=False,
        )
        state = np.zeros(specification.n_species, dtype=np.int64)
        state[np.asarray(indices, dtype=np.int64)] = 1
    else:
        probabilities = np.full(
            specification.n_species,
            1.0 / specification.n_species,
            dtype=np.float64,
        )
        state = rng.generator.multinomial(specification.n_min, probabilities).astype(
            np.int64, copy=False
        )
    return tuple(int(value) for value in state)


def _effective_beta(
    beta: NDArray[np.float64], branch: CatalyticMatrixBranch
) -> NDArray[np.float64]:
    if branch is CatalyticMatrixBranch.HISTORICAL_ORIENTATION_WITH_DIAGONAL:
        return beta
    if branch is CatalyticMatrixBranch.TRANSPOSED_WITH_DIAGONAL:
        return beta.T
    effective = beta.copy()
    np.fill_diagonal(effective, 0.0)
    return effective


def calculate_propensities(
    state: ArrayLike,
    *,
    beta: ArrayLike,
    specification: GardSpecification,
) -> PropensityArrays:
    """Calculate explicit join/leave arrays from the frozen GARD equations."""

    counts = integer_state(state, name="state")
    if counts.size != specification.n_species:
        raise IndependentGardError(
            f"state must contain {specification.n_species} species."
        )
    mass = int(counts.sum())
    if mass <= 0:
        raise IndependentGardError("Propensities are undefined for an empty state.")
    raw_beta = _matrix(beta, n_species=specification.n_species)
    effective_beta = _effective_beta(raw_beta, specification.catalytic_matrix_branch)
    counts_float = counts.astype(np.float64)
    boost = 1.0 + (effective_beta @ counts_float) / float(mass)
    reservoir = np.asarray(specification.rho, dtype=np.float64)
    join = specification.k_f * reservoir * float(mass) * boost
    leave = specification.k_b * counts_float * boost
    concatenated = np.concatenate((join, leave)).astype(np.float64, copy=False)
    if not np.all(np.isfinite(concatenated)) or np.any(concatenated < 0):
        raise IndependentGardError("Propensity arrays must be finite and nonnegative.")
    total = float(concatenated.sum())
    probabilities = (
        np.zeros_like(concatenated) if total == 0.0 else concatenated / total
    )
    return PropensityArrays(
        boost=tuple(float(value) for value in boost),
        join=tuple(float(value) for value in join),
        leave=tuple(float(value) for value in leave),
        concatenated=tuple(float(value) for value in concatenated),
        probabilities=tuple(float(value) for value in probabilities),
        total=total,
        equation_branch=specification.propensity_equation_branch.value,
    )


def _categorical_event(
    state: NDArray[np.int64],
    *,
    propensities: PropensityArrays,
    specification: GardSpecification,
    rng_streams: RNGStreams,
    generation_index_one_based: int,
    step_index_one_based: int,
    model_time_before: float | None,
) -> EventLog:
    probabilities = np.asarray(propensities.probabilities, dtype=np.float64)
    event_rng = rng_streams.events
    event_before = generator_state_sha256(event_rng.generator)
    event_index = int(event_rng.generator.choice(probabilities.size, p=probabilities))
    event_after = generator_state_sha256(event_rng.generator)
    n_species = specification.n_species
    post = state.copy()
    if event_index < n_species:
        species = event_index
        post[species] += 1
        event_kind = "join"
        join_counts = np.zeros(n_species, dtype=np.int64)
        join_counts[species] = 1
        loss_counts = np.zeros(n_species, dtype=np.int64)
    else:
        species = event_index - n_species
        post[species] -= 1
        event_kind = "leave"
        join_counts = np.zeros(n_species, dtype=np.int64)
        loss_counts = np.zeros(n_species, dtype=np.int64)
        loss_counts[species] = 1
    if np.any(post < 0):
        raise IndependentGardError("A categorical loss produced a negative count.")

    waiting_id: str | None = None
    waiting_before: str | None = None
    waiting_after: str | None = None
    increment: float | None = None
    time_after: float | None = None
    if specification.clock_semantics is ClockSemantics.GILLESPIE_EXPONENTIAL:
        waiting_rng = rng_streams.waiting_time
        waiting_id = waiting_rng.stream_id
        waiting_before = generator_state_sha256(waiting_rng.generator)
        increment = float(waiting_rng.generator.exponential(1.0 / propensities.total))
        waiting_after = generator_state_sha256(waiting_rng.generator)
        if not np.isfinite(increment) or increment < 0:
            raise IndependentGardError(
                "Gillespie waiting time must be finite and >= 0."
            )
        if model_time_before is None:
            raise IndependentGardError("Gillespie time requires a numeric prior time.")
        time_after = model_time_before + increment

    pre_mass = int(state.sum())
    post_mass = int(post.sum())
    logged_joins = tuple(int(value) for value in join_counts)
    logged_losses = tuple(int(value) for value in loss_counts)
    return EventLog(
        record_schema_version=EVENT_RECORD_SCHEMA,
        specification_id=specification.specification_id,
        generation_index_one_based=generation_index_one_based,
        step_index_one_based=step_index_one_based,
        update_kernel=specification.update_kernel.value,
        propensity_equation_branch=propensities.equation_branch,
        pre_state=tuple(int(value) for value in state),
        post_state=tuple(int(value) for value in post),
        pre_mass=pre_mass,
        post_mass=post_mass,
        mass_delta=post_mass - pre_mass,
        boost=propensities.boost,
        join_propensities=propensities.join,
        leave_propensities=propensities.leave,
        event_probabilities=propensities.probabilities,
        total_propensity=propensities.total,
        selected_event_index_zero_based=event_index,
        selected_species_index_zero_based=species,
        event_kind=event_kind,
        selection_probability=float(probabilities[event_index]),
        attempted_join_counts=logged_joins,
        attempted_loss_counts=logged_losses,
        applied_join_counts=logged_joins,
        applied_loss_counts=logged_losses,
        boundary_action="eventwise_exact",
        clock_semantics=specification.clock_semantics.value,
        time_increment=increment,
        model_time_before=model_time_before,
        model_time_after=time_after,
        event_rng_stream_id=event_rng.stream_id,
        event_rng_state_sha256_before=event_before,
        event_rng_state_sha256_after=event_after,
        waiting_rng_stream_id=waiting_id,
        waiting_rng_state_sha256_before=waiting_before,
        waiting_rng_state_sha256_after=waiting_after,
    )


def _poisson_batch(
    state: NDArray[np.int64],
    *,
    propensities: PropensityArrays,
    specification: GardSpecification,
    rng_streams: RNGStreams,
    generation_index_one_based: int,
    step_index_one_based: int,
    model_time_before: float,
) -> EventLog:
    exposure = specification.poisson_exposure
    if exposure is None:
        raise IndependentGardError("Poisson exposure was not specified.")
    event_rng = rng_streams.events
    event_before = generator_state_sha256(event_rng.generator)
    joins = event_rng.generator.poisson(
        np.asarray(propensities.join, dtype=np.float64) * exposure
    ).astype(np.int64, copy=False)
    attempted_losses = event_rng.generator.poisson(
        np.asarray(propensities.leave, dtype=np.float64) * exposure
    ).astype(np.int64, copy=False)
    event_after = generator_state_sha256(event_rng.generator)

    if specification.loss_nonnegativity is LossNonnegativity.ERROR_ON_BATCH_EXCESS:
        if np.any(attempted_losses > state):
            raise BatchLossError(
                "Vector-Poisson loss exceeded the state under error_on_batch_excess."
            )
        applied_losses = attempted_losses
    else:
        applied_losses = np.minimum(attempted_losses, state)
    candidate = state + joins - applied_losses
    if np.any(candidate < 0):
        raise IndependentGardError("Vector-Poisson update violated nonnegativity.")

    if (
        int(candidate.sum()) > specification.n_max
        and specification.growth_boundary is GrowthBoundary.REJECT_BATCH_OVERSHOOT
    ):
        post = state.copy()
        applied_joins = np.zeros_like(joins)
        logged_losses = np.zeros_like(applied_losses)
        boundary_action = "batch_overshoot_rejected"
    else:
        post = candidate
        applied_joins = joins
        logged_losses = applied_losses
        boundary_action = (
            "batch_overshoot_retained"
            if int(post.sum()) > specification.n_max
            else "batch_within_boundary"
        )

    pre_mass = int(state.sum())
    post_mass = int(post.sum())
    time_after = model_time_before + exposure
    return EventLog(
        record_schema_version=EVENT_RECORD_SCHEMA,
        specification_id=specification.specification_id,
        generation_index_one_based=generation_index_one_based,
        step_index_one_based=step_index_one_based,
        update_kernel=specification.update_kernel.value,
        propensity_equation_branch=propensities.equation_branch,
        pre_state=tuple(int(value) for value in state),
        post_state=tuple(int(value) for value in post),
        pre_mass=pre_mass,
        post_mass=post_mass,
        mass_delta=post_mass - pre_mass,
        boost=propensities.boost,
        join_propensities=propensities.join,
        leave_propensities=propensities.leave,
        event_probabilities=propensities.probabilities,
        total_propensity=propensities.total,
        selected_event_index_zero_based=None,
        selected_species_index_zero_based=None,
        event_kind="vector_poisson_batch",
        selection_probability=None,
        attempted_join_counts=tuple(int(value) for value in joins),
        attempted_loss_counts=tuple(int(value) for value in attempted_losses),
        applied_join_counts=tuple(int(value) for value in applied_joins),
        applied_loss_counts=tuple(int(value) for value in logged_losses),
        boundary_action=boundary_action,
        clock_semantics=specification.clock_semantics.value,
        time_increment=exposure,
        model_time_before=model_time_before,
        model_time_after=time_after,
        event_rng_stream_id=event_rng.stream_id,
        event_rng_state_sha256_before=event_before,
        event_rng_state_sha256_after=event_after,
        waiting_rng_stream_id=None,
        waiting_rng_state_sha256_before=None,
        waiting_rng_state_sha256_after=None,
    )


def sample_update(
    state: ArrayLike,
    *,
    beta: ArrayLike,
    specification: GardSpecification,
    rng_streams: RNGStreams,
    generation_index_one_based: int,
    step_index_one_based: int,
    model_time_before: float | None,
) -> EventLog:
    """Sample one explicit update and return a complete immutable event record."""

    counts = integer_state(state, name="state")
    propensities = calculate_propensities(
        counts, beta=beta, specification=specification
    )
    if propensities.total <= 0:
        raise ZeroPropensityError("Cannot sample from zero total propensity.")
    if specification.update_kernel is UpdateKernel.VECTOR_POISSON_BATCH:
        if model_time_before is None:
            raise IndependentGardError("Poisson batches require a numeric prior time.")
        return _poisson_batch(
            counts,
            propensities=propensities,
            specification=specification,
            rng_streams=rng_streams,
            generation_index_one_based=generation_index_one_based,
            step_index_one_based=step_index_one_based,
            model_time_before=model_time_before,
        )
    return _categorical_event(
        counts,
        propensities=propensities,
        specification=specification,
        rng_streams=rng_streams,
        generation_index_one_based=generation_index_one_based,
        step_index_one_based=step_index_one_based,
        model_time_before=model_time_before,
    )


def _step_numbers(specification: GardSpecification) -> Iterable[int]:
    if (
        specification.max_steps_semantics
        is MaxStepsSemantics.UNBOUNDED_HISTORICAL_COMPARISON
    ):
        return count(1)
    if specification.max_steps is None:
        raise IndependentGardError("Bounded growth is missing max_steps.")
    return range(1, specification.max_steps + 1)


def grow(
    initial_state: ArrayLike,
    *,
    beta: ArrayLike,
    specification: GardSpecification,
    rng_streams: RNGStreams,
    generation_index_one_based: int,
) -> GrowthResult:
    """Grow an integer state under the explicit update and limit branches."""

    state = integer_state(initial_state, name="initial_state")
    if state.size != specification.n_species:
        raise IndependentGardError(
            f"initial_state must contain {specification.n_species} species."
        )
    if int(state.sum()) <= 0:
        raise IndependentGardError("Growth requires a nonempty initial state.")
    initial = tuple(int(value) for value in state)
    events: list[EventLog] = []
    model_time: float | None = (
        None
        if specification.clock_semantics is ClockSemantics.EVENT_INDEX_ONLY
        else 0.0
    )
    terminal_status = "n_max_reached"

    if int(state.sum()) < specification.n_max:
        for step_index in _step_numbers(specification):
            propensities = calculate_propensities(
                state, beta=beta, specification=specification
            )
            if propensities.total == 0.0:
                if (
                    specification.zero_propensity_semantics
                    is ZeroPropensitySemantics.RAISE
                ):
                    raise ZeroPropensityError(
                        "Nonempty state has zero total propensity."
                    )
                terminal_status = "zero_propensity"
                break
            event = sample_update(
                state,
                beta=beta,
                specification=specification,
                rng_streams=rng_streams,
                generation_index_one_based=generation_index_one_based,
                step_index_one_based=step_index,
                model_time_before=model_time,
            )
            events.append(event)
            state = np.asarray(event.post_state, dtype=np.int64)
            model_time = event.model_time_after
            mass = int(state.sum())
            if mass == 0:
                terminal_status = "extinct"
                break
            if mass >= specification.n_max:
                terminal_status = (
                    "n_max_overshot" if mass > specification.n_max else "n_max_reached"
                )
                break
        else:
            terminal_status = "max_steps_reached"
    result = GrowthResult(
        specification_id=specification.specification_id,
        generation_index_one_based=generation_index_one_based,
        initial_state=initial,
        final_state=tuple(int(value) for value in state),
        terminal_status=terminal_status,
        events=tuple(events),
        elapsed_model_time=model_time,
    )
    if (
        terminal_status == "max_steps_reached"
        and specification.max_steps_semantics is MaxStepsSemantics.RAISE
    ):
        raise GrowthLimitError(result)
    return result


def fission(
    parent: ArrayLike,
    *,
    specification: GardSpecification,
    rng_streams: RNGStreams,
    generation_index_one_based: int,
) -> FissionLog:
    """Apply an explicit fixed-size or binomial-complement fission branch."""

    counts = integer_state(parent, name="parent")
    if counts.size != specification.n_species:
        raise IndependentGardError(
            f"parent must contain {specification.n_species} species."
        )
    mass = int(counts.sum())
    if mass <= 0:
        raise IndependentGardError("Fission requires a nonempty parent.")
    fission_rng = rng_streams.fission
    fission_before = generator_state_sha256(fission_rng.generator)
    if (
        specification.fission_semantics
        is FissionSemantics.FIXED_SIZE_WITHOUT_REPLACEMENT_ODD_DISCARD
    ):
        first = np.asarray(
            fission_rng.generator.multivariate_hypergeometric(
                counts, mass // 2, method="marginals"
            ),
            dtype=np.int64,
        )
        remainder = counts - first
        if mass % 2:
            discarded = np.asarray(
                fission_rng.generator.multivariate_hypergeometric(
                    remainder, 1, method="marginals"
                ),
                dtype=np.int64,
            )
        else:
            discarded = np.zeros_like(counts)
        second = remainder - discarded
    else:
        probability = specification.fission_probability
        if probability is None:
            raise IndependentGardError("Binomial fission probability is missing.")
        first = fission_rng.generator.binomial(counts, probability).astype(
            np.int64, copy=False
        )
        second = counts - first
        discarded = np.zeros_like(counts)
    fission_after = generator_state_sha256(fission_rng.generator)
    if not np.array_equal(first + second + discarded, counts):
        raise IndependentGardError("Fission conservation invariant failed.")

    daughter_rng = rng_streams.daughter
    daughter_before = generator_state_sha256(daughter_rng.generator)
    daughter_consumed = False
    if specification.daughter_selection is DaughterSelection.FIRST:
        label = "first"
        selected = first
    elif specification.daughter_selection is DaughterSelection.SECOND:
        label = "second"
        selected = second
    else:
        daughter_consumed = True
        choose_second = bool(daughter_rng.generator.integers(0, 2))
        label = "second" if choose_second else "first"
        selected = second if choose_second else first
    daughter_after = generator_state_sha256(daughter_rng.generator)
    if (
        specification.post_fission_semantics
        is PostFissionSemantics.ERROR_IF_SELECTED_EMPTY
        and int(selected.sum()) == 0
    ):
        raise EmptyDaughterError(
            "Selected daughter is empty under error_if_selected_empty."
        )

    return FissionLog(
        record_schema_version=FISSION_RECORD_SCHEMA,
        specification_id=specification.specification_id,
        generation_index_one_based=generation_index_one_based,
        fission_semantics=specification.fission_semantics.value,
        fission_probability=specification.fission_probability,
        parent=tuple(int(value) for value in counts),
        child_first=tuple(int(value) for value in first),
        child_second=tuple(int(value) for value in second),
        discarded=tuple(int(value) for value in discarded),
        daughter_selection=specification.daughter_selection.value,
        selected_daughter_label=label,
        selected_daughter=tuple(int(value) for value in selected),
        post_fission_semantics=specification.post_fission_semantics.value,
        conservation_holds=True,
        fission_rng_stream_id=fission_rng.stream_id,
        fission_rng_state_sha256_before=fission_before,
        fission_rng_state_sha256_after=fission_after,
        daughter_rng_stream_id=daughter_rng.stream_id,
        daughter_rng_consumed=daughter_consumed,
        daughter_rng_state_sha256_before=daughter_before,
        daughter_rng_state_sha256_after=daughter_after,
    )


def advance_generation(
    state: ArrayLike,
    *,
    beta: ArrayLike,
    specification: GardSpecification,
    rng_streams: RNGStreams,
    generation_index_one_based: int,
) -> GenerationResult:
    """Run one branch-explicit growth/fission transition."""

    growth_result = grow(
        state,
        beta=beta,
        specification=specification,
        rng_streams=rng_streams,
        generation_index_one_based=generation_index_one_based,
    )
    if growth_result.terminal_status in {"extinct", "zero_propensity"}:
        return GenerationResult(
            specification_id=specification.specification_id,
            generation_index_one_based=generation_index_one_based,
            growth=growth_result,
            fission=None,
            next_state=None,
            terminal_status=growth_result.terminal_status,
        )
    if growth_result.terminal_status == "max_steps_reached" and (
        specification.max_steps_semantics is MaxStepsSemantics.STOP_WITHOUT_FISSION
    ):
        return GenerationResult(
            specification_id=specification.specification_id,
            generation_index_one_based=generation_index_one_based,
            growth=growth_result,
            fission=None,
            next_state=None,
            terminal_status="max_steps_stop_without_fission",
        )
    split = fission(
        growth_result.final_state,
        specification=specification,
        rng_streams=rng_streams,
        generation_index_one_based=generation_index_one_based,
    )
    next_state = split.selected_daughter
    terminal_status = (
        "selected_empty_daughter"
        if sum(next_state) == 0
        else "continued_from_selected_daughter"
    )
    return GenerationResult(
        specification_id=specification.specification_id,
        generation_index_one_based=generation_index_one_based,
        growth=growth_result,
        fission=split,
        next_state=next_state,
        terminal_status=terminal_status,
    )


def simulate_lineage(
    initial_state: ArrayLike,
    *,
    beta: ArrayLike,
    specification: GardSpecification,
    rng_streams: RNGStreams,
) -> LineageResult:
    """Simulate the requested number of generations with flattened event logs."""

    current = tuple(
        int(value) for value in integer_state(initial_state, name="initial_state")
    )
    if len(current) != specification.n_species or sum(current) <= 0:
        raise IndependentGardError(
            "initial_state must match n_species and have positive mass."
        )
    initial = current
    generations: list[GenerationResult] = []
    events: list[EventLog] = []
    fissions: list[FissionLog] = []
    terminal_status = "requested_generations_completed"
    completed_fissions = 0
    for generation_index in range(1, specification.n_generations + 1):
        generation = advance_generation(
            current,
            beta=beta,
            specification=specification,
            rng_streams=rng_streams,
            generation_index_one_based=generation_index,
        )
        generations.append(generation)
        events.extend(generation.growth.events)
        if generation.fission is not None:
            fissions.append(generation.fission)
            completed_fissions += 1
        if generation.next_state is None:
            terminal_status = generation.terminal_status
            current = generation.growth.final_state
            break
        if sum(generation.next_state) == 0:
            terminal_status = generation.terminal_status
            current = generation.next_state
            break
        current = generation.next_state
    return LineageResult(
        specification_id=specification.specification_id,
        initial_state=initial,
        generations=tuple(generations),
        events=tuple(events),
        fissions=tuple(fissions),
        final_state=current,
        requested_generations=specification.n_generations,
        completed_fissions=completed_fissions,
        terminal_status=terminal_status,
    )
