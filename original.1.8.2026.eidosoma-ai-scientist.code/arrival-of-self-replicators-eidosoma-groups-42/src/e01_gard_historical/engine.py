"""Historical-reference GARD kinetics, event loop, and fission.

The implementation follows the control flow in the pinned public GARD v10
MATLAB snapshot.  Random draws are explicit inputs because legacy MATLAB RNG
stream identity is unresolved.  NumPy adapters are deliberately named as
non-MATLAB-compatible convenience paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor, inf
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class HistoricalReferenceError(ValueError):
    """Base error for invalid or unsupported historical-reference operations."""


class HistoricalSourceDomainError(HistoricalReferenceError):
    """The historical source is undefined or errors on the requested input."""


class RandomTapeExhausted(HistoricalReferenceError):
    """An explicit deterministic random-draw tape ran out of values."""


class UniformSource(Protocol):
    """Minimal explicit uniform-draw interface used by the historical loop."""

    compatibility_id: str

    def draw(self) -> float:
        """Return the next scalar draw."""


@dataclass
class UniformTape:
    """Deterministic uniform draws for hand fixtures and exact regeneration."""

    draws: tuple[float, ...]
    compatibility_id: str = "EXPLICIT_DRAW_TAPE_NO_RNG_ASSUMPTION"
    _position: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.draws)
        if any(
            not np.isfinite(value) or value < 0.0 or value > 1.0 for value in values
        ):
            raise HistoricalReferenceError(
                "UniformTape draws must be finite and in [0, 1]."
            )
        self.draws = values

    def draw(self) -> float:
        if self._position >= len(self.draws):
            raise RandomTapeExhausted(
                f"Uniform tape exhausted after {self._position} draw(s)."
            )
        value = self.draws[self._position]
        self._position += 1
        return value

    @property
    def consumed(self) -> int:
        return self._position

    @property
    def remaining(self) -> int:
        return len(self.draws) - self._position


@dataclass
class NumpyUniformSource:
    """Explicit NumPy RNG adapter; never an exact legacy-MATLAB RNG claim."""

    generator: np.random.Generator
    compatibility_id: str = "NUMPY_GENERATOR_EXPLICIT_NOT_MATLAB_LEGACY"

    def draw(self) -> float:
        return float(self.generator.random())


@dataclass(frozen=True)
class Propensities:
    """Historical per-event join and leave weights for one assembly state."""

    boost: NDArray[np.float64]
    join: NDArray[np.float64]
    leave: NDArray[np.float64]
    concatenated: NDArray[np.float64]
    total: float


@dataclass(frozen=True)
class EventRecord:
    """One historical weighted join/leave event."""

    event_number: int
    event_index_zero_based: int
    species_index_one_based: int
    historical_signed_species: int
    kind: str
    uniform_draw: float
    pre_state: tuple[int, ...]
    post_state: tuple[int, ...]
    pre_mass: int
    post_mass: int
    mass_delta: int
    boost: tuple[float, ...]
    join_rates: tuple[float, ...]
    leave_rates: tuple[float, ...]
    total_rate: float


@dataclass(frozen=True)
class GrowthResult:
    """Result of the historical single-event loop through one growth phase."""

    initial_state: tuple[int, ...]
    final_state: tuple[int, ...]
    n_max: int
    terminal_status: str
    events: tuple[EventRecord, ...]
    legacy_dt_accumulator: float
    legacy_inverse_rate_sum: float
    rng_compatibility_id: str


@dataclass(frozen=True)
class FissionSelection:
    """One without-replacement molecular selection during historical fission."""

    phase: str
    uniform_draw: float
    species_index_one_based: int


@dataclass(frozen=True)
class FissionResult:
    """Historical fixed-size fission result and deterministic lineage choice."""

    parent: tuple[int, ...]
    child_a: tuple[int, ...]
    child_b: tuple[int, ...]
    discarded: tuple[int, ...]
    followed_daughter: tuple[int, ...]
    daughter_selection_rule: str
    selections: tuple[FissionSelection, ...]
    rng_compatibility_id: str


@dataclass(frozen=True)
class GenerationResult:
    """One historical growth/fission lineage transition."""

    growth: GrowthResult
    fission: FissionResult | None
    next_state: tuple[int, ...] | None
    terminal_status: str


@dataclass(frozen=True)
class LineageResult:
    """Historical single-lineage generation sequence."""

    initial_state: tuple[int, ...]
    generations: tuple[GenerationResult, ...]
    pre_fission_trace: tuple[tuple[int, ...], ...]
    final_state: tuple[int, ...] | None
    requested_generations: int
    completed_fissions: int
    terminal_status: str


def _state_vector(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise HistoricalReferenceError(
            f"{name} must be a nonempty one-dimensional vector."
        )
    numeric = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise HistoricalReferenceError(f"{name} must contain only finite values.")
    if not np.all(numeric == np.floor(numeric)):
        raise HistoricalReferenceError(f"{name} must contain integer molecule counts.")
    if np.any(numeric < 0):
        raise HistoricalReferenceError(
            f"{name} must contain nonnegative molecule counts."
        )
    return numeric.astype(np.int64)


def _float_vector(values: ArrayLike, *, name: str, length: int) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (length,):
        raise HistoricalReferenceError(f"{name} must have shape ({length},).")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0):
        raise HistoricalReferenceError(f"{name} must be finite and nonnegative.")
    return vector


def _beta_matrix(values: ArrayLike, *, n_g: int) -> NDArray[np.float64]:
    beta = np.asarray(values, dtype=np.float64)
    if beta.shape != (n_g, n_g):
        raise HistoricalReferenceError(f"beta must have shape ({n_g}, {n_g}).")
    if not np.all(np.isfinite(beta)) or np.any(beta < 0):
        raise HistoricalReferenceError("beta must be finite and nonnegative.")
    return beta


def catalytic_matrix_from_standard_normals(
    standard_normals: ArrayLike,
    *,
    a: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Apply the pinned historical ``exp(a + sigma*z)`` transformation.

    The caller supplies every standard-normal draw.  Matrix orientation is
    preserved and diagonal values are retained, matching the historical file.
    """

    draws = np.asarray(standard_normals, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[0] != draws.shape[1] or draws.shape[0] == 0:
        raise HistoricalReferenceError(
            "standard_normals must be a nonempty square matrix."
        )
    if not np.all(np.isfinite(draws)):
        raise HistoricalReferenceError("standard_normals must be finite.")
    if not np.isfinite(a) or not np.isfinite(sigma) or sigma < 0:
        raise HistoricalReferenceError(
            "a must be finite and sigma must be finite and >= 0."
        )
    with np.errstate(over="raise", invalid="raise"):
        try:
            beta = np.exp(float(a) + float(sigma) * draws)
        except FloatingPointError as exc:
            raise HistoricalReferenceError(
                "Catalytic-matrix transform overflowed."
            ) from exc
    return np.asarray(beta, dtype=np.float64)


def catalytic_matrix_from_numpy_rng_explicit(
    n_g: int,
    *,
    a: float,
    sigma: float,
    generator: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate a distribution-compatible matrix with an explicit NumPy RNG.

    This convenience path is intentionally not named MATLAB-compatible: NumPy
    stream algorithms and array fill order do not reproduce legacy MATLAB RNG.
    """

    if not isinstance(n_g, int) or isinstance(n_g, bool) or n_g <= 0:
        raise HistoricalReferenceError("n_g must be a positive integer.")
    if not isinstance(generator, np.random.Generator):
        raise HistoricalReferenceError(
            "generator must be an explicit numpy.random.Generator."
        )
    draws = generator.standard_normal((n_g, n_g))
    return catalytic_matrix_from_standard_normals(draws, a=a, sigma=sigma)


def compute_propensities(
    state: ArrayLike,
    *,
    beta: ArrayLike,
    rho: ArrayLike,
    k_f: float,
    k_b: float,
) -> Propensities:
    """Compute historical GARD join/leave event weights.

    ``beta[i, j]`` is the effect of catalyst ``j`` on joining/leaving species
    ``i``.  The same catalytic boost multiplies join and leave weights.
    """

    n = _state_vector(state, name="state")
    n_g = int(n.size)
    mass = int(n.sum())
    if mass <= 0:
        raise HistoricalSourceDomainError(
            "Historical propensities divide by assembly mass."
        )
    matrix = _beta_matrix(beta, n_g=n_g)
    reservoir = _float_vector(rho, name="rho", length=n_g)
    if not np.isfinite(k_f) or k_f < 0 or not np.isfinite(k_b) or k_b < 0:
        raise HistoricalReferenceError("k_f and k_b must be finite and nonnegative.")

    boost = 1.0 + (matrix @ n.astype(np.float64)) / float(mass)
    join = (float(k_f) * reservoir * float(mass)) * boost
    leave = (float(k_b) * n.astype(np.float64)) * boost
    concatenated = np.concatenate((join, leave)).astype(np.float64, copy=False)
    if not np.all(np.isfinite(concatenated)) or np.any(concatenated < 0):
        raise HistoricalReferenceError(
            "Historical propensity calculation was non-finite."
        )
    total = float(concatenated.sum())
    if total <= 0:
        raise HistoricalSourceDomainError(
            "Historical weighted draw is undefined at total rate 0."
        )
    return Propensities(
        boost=np.asarray(boost, dtype=np.float64),
        join=np.asarray(join, dtype=np.float64),
        leave=np.asarray(leave, dtype=np.float64),
        concatenated=concatenated,
        total=total,
    )


def historical_weighted_index(
    weights: ArrayLike, uniform_draw: float
) -> tuple[int, float]:
    """Translate ``tgs_rndpdf`` strict-boundary selection to zero-based Python.

    The source redraws exactly-zero uniforms before this operation.  Callers of
    this scalar helper must therefore provide ``0 < uniform_draw <= 1``.
    """

    vector = np.asarray(weights, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise HistoricalReferenceError(
            "weights must be a nonempty one-dimensional vector."
        )
    if not np.all(np.isfinite(vector)) or np.any(vector < 0):
        raise HistoricalReferenceError("weights must be finite and nonnegative.")
    total = float(vector.sum())
    if total <= 0:
        raise HistoricalSourceDomainError(
            "Historical weighted draw requires positive total weight."
        )
    draw = float(uniform_draw)
    if not np.isfinite(draw) or draw <= 0.0 or draw > 1.0:
        raise HistoricalReferenceError("uniform_draw must satisfy 0 < draw <= 1.")
    target = draw * total
    index = int(np.searchsorted(np.cumsum(vector), target, side="left"))
    if index >= vector.size or vector[index] <= 0:
        raise HistoricalReferenceError(
            "Weighted selection reached an impossible zero-weight event."
        )
    return index, total


def _next_nonzero_uniform(source: UniformSource) -> float:
    draw = source.draw()
    while draw == 0.0:
        draw = source.draw()
    return draw


def _apply_historical_event(
    state: NDArray[np.int64],
    *,
    event_index: int,
    event_number: int,
    draw: float,
    propensities: Propensities,
) -> EventRecord:
    n_g = int(state.size)
    pre = state.copy()
    pre_mass = int(pre.sum())
    if event_index < n_g:
        species = event_index
        state[species] += 1
        kind = "join"
        signed_species = species + 1
    else:
        species = event_index - n_g
        state[species] -= 1
        kind = "leave"
        signed_species = -(species + 1)
        if state[species] < 0:
            raise HistoricalSourceDomainError(
                "Historical source selected an invalid loss event."
            )
    post_mass = int(state.sum())
    return EventRecord(
        event_number=event_number,
        event_index_zero_based=event_index,
        species_index_one_based=species + 1,
        historical_signed_species=signed_species,
        kind=kind,
        uniform_draw=float(draw),
        pre_state=tuple(int(value) for value in pre),
        post_state=tuple(int(value) for value in state),
        pre_mass=pre_mass,
        post_mass=post_mass,
        mass_delta=post_mass - pre_mass,
        boost=tuple(float(value) for value in propensities.boost),
        join_rates=tuple(float(value) for value in propensities.join),
        leave_rates=tuple(float(value) for value in propensities.leave),
        total_rate=propensities.total,
    )


def historical_single_event(
    state: ArrayLike,
    *,
    beta: ArrayLike,
    rho: ArrayLike,
    k_f: float,
    k_b: float,
    uniform_source: UniformSource,
    event_number: int = 1,
) -> EventRecord:
    """Sample and apply exactly one source-compatible join/leave event."""

    mutable = _state_vector(state, name="state").copy()
    if int(mutable.sum()) <= 0:
        raise HistoricalSourceDomainError(
            "Historical events are undefined for an empty state."
        )
    if (
        not isinstance(event_number, int)
        or isinstance(event_number, bool)
        or event_number <= 0
    ):
        raise HistoricalReferenceError("event_number must be a positive integer.")
    props = compute_propensities(mutable, beta=beta, rho=rho, k_f=k_f, k_b=k_b)
    draw = _next_nonzero_uniform(uniform_source)
    event_index, total = historical_weighted_index(props.concatenated, draw)
    if not np.isclose(total, props.total, rtol=0.0, atol=0.0):
        raise HistoricalReferenceError("Internal total-rate mismatch.")
    return _apply_historical_event(
        mutable,
        event_index=event_index,
        event_number=event_number,
        draw=draw,
        propensities=props,
    )


def grow_to_split_size(
    old_state: ArrayLike,
    *,
    beta: ArrayLike,
    rho: ArrayLike,
    k_f: float,
    k_b: float,
    n_max: int,
    uniform_source: UniformSource,
    event_guard: int | None,
) -> GrowthResult:
    """Run the pinned historical one-event loop until split size or extinction.

    ``event_guard`` is a validation guard, not a historical ``max_steps``
    interpretation.  If hit, it raises and produces no terminal historical state.
    Pass ``None`` explicitly to request the unbounded source loop.
    """

    state = _state_vector(old_state, name="old_state").copy()
    initial = tuple(int(value) for value in state)
    if int(state.sum()) <= 0:
        raise HistoricalSourceDomainError(
            "Historical growth is undefined for an empty start."
        )
    if not isinstance(n_max, int) or isinstance(n_max, bool) or n_max <= 0:
        raise HistoricalReferenceError("n_max must be a positive integer.")
    if event_guard is not None and (
        not isinstance(event_guard, int)
        or isinstance(event_guard, bool)
        or event_guard <= 0
    ):
        raise HistoricalReferenceError(
            "event_guard must be None or a positive integer."
        )

    events: list[EventRecord] = []
    legacy_dt = 0.0
    while int(state.sum()) < n_max:
        if event_guard is not None and len(events) >= event_guard:
            raise HistoricalReferenceError(
                "Validation event_guard reached; historical source has no max_steps terminal rule."
            )
        props = compute_propensities(state, beta=beta, rho=rho, k_f=k_f, k_b=k_b)
        draw = _next_nonzero_uniform(uniform_source)
        event_index, total = historical_weighted_index(props.concatenated, draw)
        if not np.isclose(total, props.total, rtol=0.0, atol=0.0):
            raise HistoricalReferenceError("Internal total-rate mismatch.")
        legacy_dt += props.total
        event = _apply_historical_event(
            state,
            event_index=event_index,
            event_number=len(events) + 1,
            draw=draw,
            propensities=props,
        )
        events.append(event)
        if event.post_mass == 0:
            terminal_status = "extinct"
            break
    else:
        terminal_status = "split_size_reached"

    return GrowthResult(
        initial_state=initial,
        final_state=tuple(int(value) for value in state),
        n_max=n_max,
        terminal_status=terminal_status,
        events=tuple(events),
        legacy_dt_accumulator=float(legacy_dt),
        legacy_inverse_rate_sum=inf if legacy_dt == 0.0 else 1.0 / legacy_dt,
        rng_compatibility_id=uniform_source.compatibility_id,
    )


def split_fixed_size_without_replacement(
    parent: ArrayLike,
    *,
    uniform_source: UniformSource,
) -> FissionResult:
    """Reproduce ``tgs_split_v10`` fixed-size sequential sampling.

    Child A receives ``floor(N/2)`` draws without replacement.  Child B is
    the remainder.  For odd ``N``, one further selected molecule is discarded
    so both returned children have equal size.  This is not independent
    per-species ``Binomial(n_i, 0.5)`` sampling.
    """

    original = _state_vector(parent, name="parent")
    total_mass = int(original.sum())
    if total_mass <= 0:
        raise HistoricalSourceDomainError("Historical fission rejects an empty parent.")
    remaining = original.copy()
    child_a = np.zeros_like(remaining)
    discarded = np.zeros_like(remaining)
    selections: list[FissionSelection] = []

    for _ in range(floor(total_mass / 2)):
        draw = _next_nonzero_uniform(uniform_source)
        index, _ = historical_weighted_index(remaining, draw)
        child_a[index] += 1
        remaining[index] -= 1
        selections.append(
            FissionSelection(
                phase="child_a",
                uniform_draw=draw,
                species_index_one_based=index + 1,
            )
        )

    while int(child_a.sum()) < int(remaining.sum()):
        draw = _next_nonzero_uniform(uniform_source)
        index, _ = historical_weighted_index(remaining, draw)
        remaining[index] -= 1
        discarded[index] += 1
        selections.append(
            FissionSelection(
                phase="odd_parent_discard",
                uniform_draw=draw,
                species_index_one_based=index + 1,
            )
        )

    if not np.array_equal(child_a + remaining + discarded, original):
        raise HistoricalReferenceError(
            "Historical fission conservation invariant failed."
        )
    if int(child_a.sum()) != int(remaining.sum()):
        raise HistoricalReferenceError(
            "Historical fission equal-child-size invariant failed."
        )

    child_a_tuple = tuple(int(value) for value in child_a)
    child_b_tuple = tuple(int(value) for value in remaining)
    return FissionResult(
        parent=tuple(int(value) for value in original),
        child_a=child_a_tuple,
        child_b=child_b_tuple,
        discarded=tuple(int(value) for value in discarded),
        followed_daughter=child_a_tuple,
        daughter_selection_rule="FIRST_OUTPUT_CHILD_A_NO_ADDITIONAL_RANDOM_DRAW",
        selections=tuple(selections),
        rng_compatibility_id=uniform_source.compatibility_id,
    )


def advance_one_generation(
    old_state: ArrayLike,
    *,
    beta: ArrayLike,
    rho: ArrayLike,
    k_f: float,
    k_b: float,
    n_max: int,
    uniform_source: UniformSource,
    event_guard: int | None,
) -> GenerationResult:
    """Grow, fission, and continue from historical child A for one generation."""

    growth = grow_to_split_size(
        old_state,
        beta=beta,
        rho=rho,
        k_f=k_f,
        k_b=k_b,
        n_max=n_max,
        uniform_source=uniform_source,
        event_guard=event_guard,
    )
    if growth.terminal_status == "extinct":
        return GenerationResult(
            growth=growth,
            fission=None,
            next_state=None,
            terminal_status="extinct_before_fission",
        )
    fission = split_fixed_size_without_replacement(
        growth.final_state,
        uniform_source=uniform_source,
    )
    return GenerationResult(
        growth=growth,
        fission=fission,
        next_state=fission.followed_daughter,
        terminal_status="continued_from_child_a",
    )


def simulate_lineage(
    initial_state: ArrayLike,
    *,
    beta: ArrayLike,
    rho: ArrayLike,
    k_f: float,
    k_b: float,
    n_max: int,
    n_generations: int,
    uniform_source: UniformSource,
    event_guard_per_generation: int | None,
) -> LineageResult:
    """Run the historical single-assembly lineage for explicit generations."""

    state = tuple(
        int(value) for value in _state_vector(initial_state, name="initial_state")
    )
    if sum(state) <= 0:
        raise HistoricalSourceDomainError(
            "Historical lineage requires a nonempty start."
        )
    if (
        not isinstance(n_generations, int)
        or isinstance(n_generations, bool)
        or n_generations <= 0
    ):
        raise HistoricalReferenceError("n_generations must be a positive integer.")
    generations: list[GenerationResult] = []
    trace: list[tuple[int, ...]] = []
    current: tuple[int, ...] | None = state
    completed_fissions = 0
    terminal_status = "requested_generations_completed"
    for _ in range(n_generations):
        if current is None:
            break
        generation = advance_one_generation(
            current,
            beta=beta,
            rho=rho,
            k_f=k_f,
            k_b=k_b,
            n_max=n_max,
            uniform_source=uniform_source,
            event_guard=event_guard_per_generation,
        )
        generations.append(generation)
        trace.append(generation.growth.final_state)
        current = generation.next_state
        if generation.fission is None:
            terminal_status = "extinct_before_requested_generations"
            break
        completed_fissions += 1
    return LineageResult(
        initial_state=state,
        generations=tuple(generations),
        pre_fission_trace=tuple(trace),
        final_state=current,
        requested_generations=n_generations,
        completed_fissions=completed_fissions,
        terminal_status=terminal_status,
    )


def historical_initial_state_with_replacement(
    *,
    n_g: int,
    n_min: int,
    uniform_source: UniformSource,
) -> tuple[int, ...]:
    """Reproduce the historical initializer's with-replacement type counts.

    This helper reproduces the count-construction operation only.  It does not
    emulate ``tgs_agard_v10``'s hidden global-state ordering or legacy MATLAB
    seed/state APIs.
    """

    if not isinstance(n_g, int) or isinstance(n_g, bool) or n_g <= 0:
        raise HistoricalReferenceError("n_g must be a positive integer.")
    if not isinstance(n_min, int) or isinstance(n_min, bool) or n_min < 0:
        raise HistoricalReferenceError("n_min must be a nonnegative integer.")
    state = np.zeros(n_g, dtype=np.int64)
    for _ in range(n_min):
        draw = uniform_source.draw()
        if draw < 0.0 or draw >= 1.0:
            raise HistoricalReferenceError(
                "Initializer draws must satisfy 0 <= draw < 1."
            )
        state[min(int(draw * n_g), n_g - 1)] += 1
    return tuple(int(value) for value in state)


def historical_n_max(*, n_g: int, split_size: float) -> int:
    """Return the source's ``ceil(NG * splitsize)`` boundary."""

    if not isinstance(n_g, int) or isinstance(n_g, bool) or n_g <= 0:
        raise HistoricalReferenceError("n_g must be a positive integer.")
    if not np.isfinite(split_size) or split_size <= 0:
        raise HistoricalReferenceError("split_size must be finite and positive.")
    return ceil(n_g * float(split_size))
