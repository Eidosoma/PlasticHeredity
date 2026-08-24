"""Deterministic S12F Poisson-exposure simulator and observation clocks.

This module intentionally contains no label, clustering, information-theory,
prediction, or intervention code.  It extends the paper-prose S12E K1--K3
kernel only with the preregistered exposure and overshoot branches and retains
the exact state sequence required by a later, separately authorized step.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

VERSION = "E01-S12F-LATENT-TIMEBASE-INFERENCE-v1.0.0"
N_SPECIES = 100
N_MIN = 40
N_MAX = 80
N_GENERATIONS = 100
MAX_STEPS = 1000
BETA_A = -4.0
BETA_SIGMA = 4.0
K_FORWARD = 1.0e-2
K_BACKWARD = 1.0e-4
RHO_EACH = 1.0 / N_SPECIES

DaughterRule = Literal["RANDOM_NONEMPTY", "FIRST_DAUGHTER", "RANDOM_LITERAL"]
OvershootRule = Literal["RETAIN_OVERSHOOT", "TRIM_NEW_ENTRANTS_TO_NMAX"]
ExposureFamily = Literal["FIXED_COMMON_EXPOSURE", "ADAPTIVE_GROSS_EVENT_EXPOSURE"]
ClockId = Literal[
    "C0_BATCH_UPDATES_ONLY",
    "C1_SELECTED_DAUGHTER_RETAINED",
    "C2_EXPLICIT_PRE_AND_POST_FISSION",
    "C3_NONZERO_REACTION_CHANNEL",
    "C4_GROSS_MOLECULAR_EVENT",
]

CLOCK_IDS: tuple[str, ...] = (
    "C0_BATCH_UPDATES_ONLY",
    "C1_SELECTED_DAUGHTER_RETAINED",
    "C2_EXPLICIT_PRE_AND_POST_FISSION",
    "C3_NONZERO_REACTION_CHANNEL",
    "C4_GROSS_MOLECULAR_EVENT",
)
DAUGHTER_RULES: tuple[str, ...] = (
    "RANDOM_NONEMPTY",
    "FIRST_DAUGHTER",
    "RANDOM_LITERAL",
)
OVERSHOOT_RULES: tuple[str, ...] = (
    "RETAIN_OVERSHOOT",
    "TRIM_NEW_ENTRANTS_TO_NMAX",
)


@dataclass(frozen=True, slots=True)
class ExposureDefinition:
    family: ExposureFamily
    h: float | None = None
    c: float | None = None
    h_max: float | None = None

    def validate(self) -> None:
        if self.family == "FIXED_COMMON_EXPOSURE":
            if self.h is None or not 0.10 <= self.h <= 1.25:
                raise ValueError("fixed exposure h must be in [0.10, 1.25]")
            if self.c is not None or self.h_max is not None:
                raise ValueError("fixed exposure cannot define c or h_max")
        elif self.family == "ADAPTIVE_GROSS_EVENT_EXPOSURE":
            if self.c is None or not 0.5 <= self.c <= 16.0:
                raise ValueError("adaptive c must be in [0.5, 16]")
            if self.h_max is None or not 0.1 <= self.h_max <= 2.0:
                raise ValueError("adaptive h_max must be in [0.1, 2]")
            if self.h is not None:
                raise ValueError("adaptive exposure cannot define fixed h")
        else:  # pragma: no cover - protected by the type contract
            raise ValueError(f"unknown exposure family {self.family!r}")

    @property
    def identity(self) -> str:
        self.validate()
        if self.family == "FIXED_COMMON_EXPOSURE":
            return f"FIXED-h={self.h:.17g}"
        return f"ADAPTIVE-c={self.c:.17g}-hmax={self.h_max:.17g}"


@dataclass(frozen=True, slots=True)
class SimulationDefinition:
    daughter_rule: DaughterRule
    overshoot_rule: OvershootRule
    exposure: ExposureDefinition

    def validate(self) -> None:
        if self.daughter_rule not in DAUGHTER_RULES:
            raise ValueError(f"unknown daughter rule {self.daughter_rule!r}")
        if self.overshoot_rule not in OVERSHOOT_RULES:
            raise ValueError(f"unknown overshoot rule {self.overshoot_rule!r}")
        self.exposure.validate()

    @property
    def identity(self) -> str:
        self.validate()
        return f"{self.daughter_rule}__{self.overshoot_rule}__{self.exposure.identity}"


@dataclass(frozen=True, slots=True)
class SeedIdentity:
    phase: str
    root_sha256: str
    purpose: str
    matrix_index: int
    configuration_id: str | None
    extra: tuple[str, ...]
    derived_seed: int
    seed_material_sha256: str


@dataclass(frozen=True, slots=True)
class StateObservation:
    observation_index: int
    observation_kind: str
    completed_fissions: int
    growth_generation_one_based: int
    batch_step: int
    generation_local_step: int
    state: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    generation_one_based: int
    terminal_status: str
    update_count: int
    nonzero_reaction_type_count: int
    gross_sampled_event_count: int
    pre_fission_mass: int | None
    post_fission_mass: int | None
    child_a_mass: int | None
    child_b_mass: int | None
    selected_daughter: str | None
    overshoot_before_trim: int | None
    trimmed_new_entrants: int
    maximum_exposure: float
    minimum_exposure: float


@dataclass(frozen=True, slots=True)
class TimebaseTrajectory:
    trajectory_id: str
    phase: str
    matrix_index: int
    configuration_id: str
    definition: SimulationDefinition
    beta_sha256: str
    initial_state_sha256: str
    observations: tuple[StateObservation, ...]
    generations: tuple[GenerationSummary, ...]
    completed_fissions: int
    total_batch_updates: int
    total_nonzero_reaction_types: int
    total_gross_sampled_events: int
    terminal_status: str
    extinction_generation: int | None
    trajectory_sha256: str

    @property
    def states(self) -> NDArray[np.int64]:
        return np.asarray([row.state for row in self.observations], dtype=np.int64)


def _seed_payload(
    root_hex: str,
    phase: str,
    purpose: str,
    matrix_index: int,
    configuration_id: str | None,
    extra: tuple[object, ...],
) -> bytes:
    fields = [
        VERSION,
        root_hex,
        phase,
        purpose,
        str(matrix_index),
        "SHARED" if configuration_id is None else configuration_id,
        *map(str, extra),
    ]
    return "\x1f".join(fields).encode("utf-8")


def derive_seed(
    root_hex: str,
    phase: str,
    purpose: str,
    matrix_index: int,
    configuration_id: str | None = None,
    *extra: object,
) -> SeedIdentity:
    if len(root_hex) != 64 or any(character not in "0123456789abcdef" for character in root_hex):
        raise ValueError("S12F roots must be lowercase 256-bit hexadecimal identities")
    payload = _seed_payload(
        root_hex, phase, purpose, matrix_index, configuration_id, extra
    )
    digest = hashlib.sha256(payload).digest()
    return SeedIdentity(
        phase=phase,
        root_sha256=root_hex,
        purpose=purpose,
        matrix_index=matrix_index,
        configuration_id=configuration_id,
        extra=tuple(map(str, extra)),
        derived_seed=int.from_bytes(digest[:16], "big", signed=False),
        seed_material_sha256=hashlib.sha256(payload).hexdigest(),
    )


def generator(identity: SeedIdentity) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(identity.derived_seed))


def array_sha256(array: NDArray[np.generic]) -> str:
    value = np.ascontiguousarray(array)
    payload = (
        str(value.dtype).encode()
        + b"\x00"
        + json.dumps(value.shape, separators=(",", ":")).encode()
        + b"\x00"
        + value.tobytes(order="C")
    )
    return hashlib.sha256(payload).hexdigest()


def generate_beta(identity: SeedIdentity) -> NDArray[np.float64]:
    rng = generator(identity)
    return np.exp(
        BETA_A + BETA_SIGMA * rng.standard_normal((N_SPECIES, N_SPECIES))
    ).astype(np.float64, copy=False)


def initialize_distinct_state(identity: SeedIdentity) -> NDArray[np.int64]:
    rng = generator(identity)
    state = np.zeros(N_SPECIES, dtype=np.int64)
    state[rng.choice(N_SPECIES, size=N_MIN, replace=False)] = 1
    return state


def rates(
    state: NDArray[np.int64], beta: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mass = int(state.sum())
    if mass <= 0:
        raise ValueError("rates are undefined for an empty assembly")
    boost = 1.0 + (beta @ state.astype(np.float64)) / float(mass)
    joins = K_FORWARD * RHO_EACH * float(mass) * boost
    losses = K_BACKWARD * state.astype(np.float64) * boost
    if (
        np.any(joins < 0)
        or np.any(losses < 0)
        or not np.all(np.isfinite(joins))
        or not np.all(np.isfinite(losses))
    ):
        raise ValueError("nonfinite or negative paper-prose rate")
    return joins, losses


def exposure_for_rates(
    definition: ExposureDefinition,
    joins: NDArray[np.float64],
    losses: NDArray[np.float64],
) -> float:
    definition.validate()
    if definition.family == "FIXED_COMMON_EXPOSURE":
        assert definition.h is not None
        return float(definition.h)
    assert definition.c is not None and definition.h_max is not None
    total = float(joins.sum() + losses.sum())
    if total <= 0.0:
        return float(definition.h_max)
    return float(min(definition.h_max, definition.c / total))


def _trim_new_entrants(
    result: NDArray[np.int64],
    join_draws: NDArray[np.int64],
    trim_rng: np.random.Generator,
) -> tuple[NDArray[np.int64], int]:
    excess = max(0, int(result.sum()) - N_MAX)
    if excess == 0:
        return result, 0
    if excess > int(join_draws.sum()):
        raise AssertionError("overshoot exceeded the newly joined population")
    removed = trim_rng.multivariate_hypergeometric(join_draws, excess).astype(
        np.int64, copy=False
    )
    trimmed = result - removed
    if int(trimmed.sum()) != N_MAX or np.any(trimmed < 0):
        raise AssertionError("new-entrant trim violated mass or nonnegativity")
    return trimmed, excess


def poisson_update(
    state: NDArray[np.int64],
    beta: NDArray[np.float64],
    definition: SimulationDefinition,
    event_rng: np.random.Generator,
    trim_rng: np.random.Generator,
) -> tuple[NDArray[np.int64], int, int, int, float]:
    joins, losses = rates(state, beta)
    exposure = exposure_for_rates(definition.exposure, joins, losses)
    join_draws = event_rng.poisson(exposure * joins).astype(np.int64, copy=False)
    attempted_losses = event_rng.poisson(exposure * losses).astype(
        np.int64, copy=False
    )
    applied_losses = np.minimum(attempted_losses, state)
    result = state + join_draws - applied_losses
    overshoot = max(0, int(result.sum()) - N_MAX)
    trimmed = 0
    if definition.overshoot_rule == "TRIM_NEW_ENTRANTS_TO_NMAX":
        result, trimmed = _trim_new_entrants(result, join_draws, trim_rng)
    if np.any(result < 0):
        raise AssertionError("clipped Poisson update became negative")
    nonzero_types = int(np.count_nonzero((join_draws + attempted_losses) > 0))
    gross_events = int(join_draws.sum() + attempted_losses.sum())
    return (
        result.astype(np.int64, copy=False),
        nonzero_types,
        gross_events,
        overshoot if trimmed else max(0, int(result.sum()) - N_MAX),
        exposure,
    )


def fission(
    parent: NDArray[np.int64], rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    child_a = rng.binomial(parent, 0.5).astype(np.int64, copy=False)
    child_b = parent - child_a
    if not np.array_equal(child_a + child_b, parent):
        raise AssertionError("complementary binomial fission lost mass")
    return child_a, child_b


def select_daughter(
    child_a: NDArray[np.int64],
    child_b: NDArray[np.int64],
    rule: DaughterRule,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], str]:
    if rule == "FIRST_DAUGHTER":
        return child_a.copy(), "A"
    if rule == "RANDOM_LITERAL":
        return (child_a.copy(), "A") if int(rng.integers(0, 2)) == 0 else (child_b.copy(), "B")
    candidates: list[tuple[NDArray[np.int64], str]] = []
    if int(child_a.sum()) > 0:
        candidates.append((child_a, "A"))
    if int(child_b.sum()) > 0:
        candidates.append((child_b, "B"))
    if not candidates:
        return child_a.copy(), "NONE"
    selected = candidates[int(rng.integers(0, len(candidates)))]
    return selected[0].copy(), selected[1]


def clock_length(trajectory: TimebaseTrajectory, clock_id: ClockId) -> int:
    if clock_id == "C0_BATCH_UPDATES_ONLY":
        return trajectory.total_batch_updates
    if clock_id == "C1_SELECTED_DAUGHTER_RETAINED":
        return trajectory.total_batch_updates + trajectory.completed_fissions
    if clock_id == "C2_EXPLICIT_PRE_AND_POST_FISSION":
        return trajectory.total_batch_updates + 2 * trajectory.completed_fissions
    if clock_id == "C3_NONZERO_REACTION_CHANNEL":
        return trajectory.total_nonzero_reaction_types
    if clock_id == "C4_GROSS_MOLECULAR_EVENT":
        return trajectory.total_gross_sampled_events
    raise ValueError(f"unknown clock {clock_id!r}")


def _trajectory_digest(
    trajectory_id: str,
    definition: SimulationDefinition,
    observations: list[StateObservation],
    generations: list[GenerationSummary],
    terminal_status: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(VERSION.encode())
    digest.update(trajectory_id.encode())
    digest.update(definition.identity.encode())
    digest.update(terminal_status.encode())
    for row in observations:
        digest.update(
            np.asarray(
                [
                    row.observation_index,
                    row.completed_fissions,
                    row.growth_generation_one_based,
                    row.batch_step,
                    row.generation_local_step,
                ],
                dtype="<i8",
            ).tobytes()
        )
        digest.update(row.observation_kind.encode())
        digest.update(np.asarray(row.state, dtype="<i8").tobytes())
    for row in generations:
        digest.update(
            json.dumps(
                {
                    field: getattr(row, field)
                    for field in row.__dataclass_fields__
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    return digest.hexdigest()


def simulate_trajectory(
    *,
    phase: str,
    root_hex: str,
    matrix_index: int,
    definition: SimulationDefinition,
    stream_identity: str,
    beta: NDArray[np.float64] | None = None,
    initial_state: NDArray[np.int64] | None = None,
) -> tuple[TimebaseTrajectory, tuple[SeedIdentity, ...]]:
    definition.validate()
    beta_seed = derive_seed(root_hex, phase, "catalytic_matrix", matrix_index)
    init_seed = derive_seed(root_hex, phase, "initial_state", matrix_index)
    event_seed = derive_seed(
        root_hex, phase, "poisson_update", matrix_index, stream_identity
    )
    trim_seed = derive_seed(
        root_hex, phase, "overshoot_trim", matrix_index, stream_identity
    )
    fission_seed = derive_seed(root_hex, phase, "fission", matrix_index, stream_identity)
    daughter_seed = derive_seed(
        root_hex, phase, "daughter_selection", matrix_index, stream_identity
    )
    seeds = (beta_seed, init_seed, event_seed, trim_seed, fission_seed, daughter_seed)
    beta_value = generate_beta(beta_seed) if beta is None else np.asarray(beta, dtype=np.float64)
    state = (
        initialize_distinct_state(init_seed)
        if initial_state is None
        else np.asarray(initial_state, dtype=np.int64).copy()
    )
    if beta_value.shape != (N_SPECIES, N_SPECIES):
        raise ValueError("beta must be 100 by 100")
    if state.shape != (N_SPECIES,) or int(state.sum()) != N_MIN:
        raise ValueError("initial state must be a mass-40 100-vector")
    if np.count_nonzero(state) != N_MIN or np.any((state != 0) & (state != 1)):
        raise ValueError("initial state must contain forty distinct singletons")

    event_rng = generator(event_seed)
    trim_rng = generator(trim_seed)
    fission_rng = generator(fission_seed)
    daughter_rng = generator(daughter_seed)
    trajectory_id = f"E01-S12F-{phase.upper()}-{stream_identity}-M{matrix_index:03d}"
    observations = [
        StateObservation(
            0,
            "initial_selected_state",
            0,
            0,
            0,
            0,
            tuple(map(int, state)),
        )
    ]
    generations: list[GenerationSummary] = []
    batch_step = 0
    completed = 0
    total_nonzero = 0
    total_gross = 0
    terminal_status = "requested_fissions_completed"
    extinction_generation: int | None = None

    for generation_one_based in range(1, N_GENERATIONS + 1):
        local_step = 0
        local_nonzero = 0
        local_gross = 0
        exposures: list[float] = []
        trim_count = 0
        largest_pretrim_overshoot = 0
        growth_status = "n_max_reached"
        while int(state.sum()) < N_MAX and local_step < MAX_STEPS:
            if int(state.sum()) == 0:
                growth_status = "extinct_during_growth"
                break
            state, nonzero, gross, pretrim_overshoot, exposure = poisson_update(
                state, beta_value, definition, event_rng, trim_rng
            )
            local_step += 1
            batch_step += 1
            local_nonzero += nonzero
            local_gross += gross
            total_nonzero += nonzero
            total_gross += gross
            exposures.append(exposure)
            largest_pretrim_overshoot = max(largest_pretrim_overshoot, pretrim_overshoot)
            if definition.overshoot_rule == "TRIM_NEW_ENTRANTS_TO_NMAX" and pretrim_overshoot > 0:
                trim_count += pretrim_overshoot
            observations.append(
                StateObservation(
                    len(observations),
                    "molecular_update",
                    completed,
                    generation_one_based,
                    batch_step,
                    local_step,
                    tuple(map(int, state)),
                )
            )
            if int(state.sum()) == 0:
                growth_status = "extinct_during_growth"
                break
            if int(state.sum()) >= N_MAX:
                growth_status = (
                    "n_max_overshot" if int(state.sum()) > N_MAX else "n_max_reached"
                )
                break
        else:
            if int(state.sum()) < N_MAX:
                growth_status = "max_steps_reached"

        min_exposure = float(min(exposures)) if exposures else float("nan")
        max_exposure = float(max(exposures)) if exposures else float("nan")
        if int(state.sum()) == 0:
            generations.append(
                GenerationSummary(
                    generation_one_based,
                    growth_status,
                    local_step,
                    local_nonzero,
                    local_gross,
                    None,
                    None,
                    None,
                    None,
                    None,
                    largest_pretrim_overshoot,
                    trim_count,
                    max_exposure,
                    min_exposure,
                )
            )
            terminal_status = "extinct_during_growth"
            extinction_generation = generation_one_based
            break

        pre_mass = int(state.sum())
        child_a, child_b = fission(state, fission_rng)
        selected, selected_name = select_daughter(
            child_a, child_b, definition.daughter_rule, daughter_rng
        )
        state = selected
        completed += 1
        generations.append(
            GenerationSummary(
                generation_one_based,
                growth_status,
                local_step,
                local_nonzero,
                local_gross,
                pre_mass,
                int(state.sum()),
                int(child_a.sum()),
                int(child_b.sum()),
                selected_name,
                largest_pretrim_overshoot,
                trim_count,
                max_exposure,
                min_exposure,
            )
        )
        observations.append(
            StateObservation(
                len(observations),
                "post_fission",
                completed,
                generation_one_based,
                batch_step,
                local_step,
                tuple(map(int, state)),
            )
        )
        if int(state.sum()) == 0:
            terminal_status = "selected_daughter_empty"
            extinction_generation = generation_one_based
            break

    digest = _trajectory_digest(
        trajectory_id, definition, observations, generations, terminal_status
    )
    return (
        TimebaseTrajectory(
            trajectory_id=trajectory_id,
            phase=phase,
            matrix_index=matrix_index,
            configuration_id=stream_identity,
            definition=definition,
            beta_sha256=array_sha256(beta_value),
            initial_state_sha256=array_sha256(
                np.asarray(observations[0].state, dtype=np.int64)
            ),
            observations=tuple(observations),
            generations=tuple(generations),
            completed_fissions=completed,
            total_batch_updates=batch_step,
            total_nonzero_reaction_types=total_nonzero,
            total_gross_sampled_events=total_gross,
            terminal_status=terminal_status,
            extinction_generation=extinction_generation,
            trajectory_sha256=digest,
        ),
        seeds,
    )


def trajectory_replay_equal(left: TimebaseTrajectory, right: TimebaseTrajectory) -> bool:
    return left == right and left.trajectory_sha256 == right.trajectory_sha256


def trajectory_summary(trajectory: TimebaseTrajectory) -> dict[str, object]:
    completed = [row for row in trajectory.generations if row.pre_fission_mass is not None]
    pre = np.asarray([row.pre_fission_mass for row in completed], dtype=np.float64)
    post = np.asarray([row.post_fission_mass for row in completed], dtype=np.float64)
    overshoot = np.asarray(
        [row.overshoot_before_trim for row in completed], dtype=np.float64
    )
    return {
        "trajectoryId": trajectory.trajectory_id,
        "phase": trajectory.phase,
        "matrixIndex": trajectory.matrix_index,
        "configurationId": trajectory.configuration_id,
        "exposureFamily": trajectory.definition.exposure.family,
        "h": trajectory.definition.exposure.h,
        "c": trajectory.definition.exposure.c,
        "hMax": trajectory.definition.exposure.h_max,
        "daughterRule": trajectory.definition.daughter_rule,
        "overshootRule": trajectory.definition.overshoot_rule,
        "completedFissions": trajectory.completed_fissions,
        "terminalStatus": trajectory.terminal_status,
        "extinctionGeneration": trajectory.extinction_generation,
        "totalBatchUpdates": trajectory.total_batch_updates,
        "clockC0": clock_length(trajectory, "C0_BATCH_UPDATES_ONLY"),
        "clockC1": clock_length(trajectory, "C1_SELECTED_DAUGHTER_RETAINED"),
        "clockC2": clock_length(trajectory, "C2_EXPLICIT_PRE_AND_POST_FISSION"),
        "clockC3": clock_length(trajectory, "C3_NONZERO_REACTION_CHANNEL"),
        "clockC4": clock_length(trajectory, "C4_GROSS_MOLECULAR_EVENT"),
        "medianPreFissionMass": float(np.median(pre)) if pre.size else np.nan,
        "medianPostFissionMass": float(np.median(post)) if post.size else np.nan,
        "q95Overshoot": float(np.quantile(overshoot, 0.95)) if overshoot.size else np.nan,
        "maximumOvershoot": float(np.max(overshoot)) if overshoot.size else np.nan,
        "maxstepsTerminations": int(
            sum(row.terminal_status == "max_steps_reached" for row in completed)
        ),
        "sourceObservationCount": len(trajectory.observations),
        "betaSha256": trajectory.beta_sha256,
        "initialStateSha256": trajectory.initial_state_sha256,
        "trajectorySha256": trajectory.trajectory_sha256,
    }


def observation_rows(trajectory: TimebaseTrajectory) -> list[dict[str, object]]:
    return [
        {
            "trajectoryId": trajectory.trajectory_id,
            "phase": trajectory.phase,
            "matrixIndex": trajectory.matrix_index,
            "candidateId": trajectory.configuration_id,
            "observationIndex": row.observation_index,
            "observationKind": row.observation_kind,
            "completedFissions": row.completed_fissions,
            "growthGenerationOneBased": row.growth_generation_one_based,
            "batchStep": row.batch_step,
            "generationLocalStep": row.generation_local_step,
            "mass": int(sum(row.state)),
            "state": list(row.state),
            "trajectorySha256": trajectory.trajectory_sha256,
        }
        for row in trajectory.observations
    ]
