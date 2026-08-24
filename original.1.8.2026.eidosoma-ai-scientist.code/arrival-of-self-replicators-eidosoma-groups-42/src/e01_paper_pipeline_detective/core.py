"""Phase-1 GARD candidates frozen for E01 S12E.

The candidate family deliberately separates the paper-prose simultaneous
Poisson update from the public-historical categorical update.  Every candidate
shares paper-style distinct-type initialization, the stated 1,000-update
generation ceiling, retained batch overshoot, and complementary binomial
fission.  It is therefore not a claim that K0 is the unmodified historical v10
program; K0 isolates its categorical event kernel inside the S12E comparison.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

VERSION = "E01-S12E-PAPER-PIPELINE-DETECTIVE-RECONSTRUCTION-v1.0.0"
N_SPECIES = 100
N_MIN = 40
N_MAX = 80
N_GENERATIONS = 100
MAX_STEPS = 1000
BETA_A = -4.0
BETA_SIGMA = 4.0
K_FORWARD = 1.0e-2
K_BACKWARD = 1.0e-4

ENGINE_IDS = (
    "K0_HISTORICAL_EVENTWISE",
    "K1_PAPER_POISSON_RANDOM_NONEMPTY",
    "K2_PAPER_POISSON_FIRST_DAUGHTER",
    "K3_PAPER_POISSON_RANDOM_LITERAL",
    "K4_PAPER_POISSON_RHO_ONE",
)

UpdateKernel = Literal["categorical", "poisson_vector"]
DaughterRule = Literal["first_literal", "uniform_nonempty", "uniform_literal"]


@dataclass(frozen=True, slots=True)
class EngineDefinition:
    """Complete S12E engine branch without an inferred model default."""

    engine_id: str
    update_kernel: UpdateKernel
    rho_each: float
    daughter_rule: DaughterRule


ENGINE_DEFINITIONS: dict[str, EngineDefinition] = {
    "K0_HISTORICAL_EVENTWISE": EngineDefinition(
        "K0_HISTORICAL_EVENTWISE", "categorical", 1.0 / N_SPECIES, "first_literal"
    ),
    "K1_PAPER_POISSON_RANDOM_NONEMPTY": EngineDefinition(
        "K1_PAPER_POISSON_RANDOM_NONEMPTY",
        "poisson_vector",
        1.0 / N_SPECIES,
        "uniform_nonempty",
    ),
    "K2_PAPER_POISSON_FIRST_DAUGHTER": EngineDefinition(
        "K2_PAPER_POISSON_FIRST_DAUGHTER",
        "poisson_vector",
        1.0 / N_SPECIES,
        "first_literal",
    ),
    "K3_PAPER_POISSON_RANDOM_LITERAL": EngineDefinition(
        "K3_PAPER_POISSON_RANDOM_LITERAL",
        "poisson_vector",
        1.0 / N_SPECIES,
        "uniform_literal",
    ),
    "K4_PAPER_POISSON_RHO_ONE": EngineDefinition(
        "K4_PAPER_POISSON_RHO_ONE", "poisson_vector", 1.0, "uniform_nonempty"
    ),
}


@dataclass(frozen=True, slots=True)
class SeedIdentity:
    phase: str
    root_sha256: str
    purpose: str
    matrix_index: int
    engine_id: str | None
    derived_seed: int
    seed_material_sha256: str


@dataclass(frozen=True, slots=True)
class Observation:
    observation_index: int
    observation_kind: str
    generation: int
    growth_generation_one_based: int
    molecular_step: int
    generation_local_step: int
    state: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    generation_one_based: int
    growth_terminal_status: str
    update_count: int
    pre_fission_mass: int | None
    child_a_mass: int | None
    child_b_mass: int | None
    selected_daughter: str | None
    selected_mass: int | None
    overshoot: int | None


@dataclass(frozen=True, slots=True)
class GardTrajectory:
    trajectory_id: str
    phase: str
    matrix_index: int
    engine_id: str
    beta_sha256: str
    initial_state_sha256: str
    observations: tuple[Observation, ...]
    generations: tuple[GenerationSummary, ...]
    completed_fissions: int
    total_batch_steps: int
    terminal_status: str
    extinction_generation: int | None
    trajectory_sha256: str

    @property
    def states(self) -> NDArray[np.int64]:
        return np.asarray([row.state for row in self.observations], dtype=np.int64)


def _seed_material(
    root_hex: str,
    phase: str,
    purpose: str,
    matrix_index: int,
    engine_id: str | None,
    *extra: object,
) -> bytes:
    fields = [
        VERSION,
        root_hex,
        phase,
        purpose,
        str(matrix_index),
        "SHARED" if engine_id is None else engine_id,
        *map(str, extra),
    ]
    return "\x1f".join(fields).encode("utf-8")


def derive_seed(
    root_hex: str,
    phase: str,
    purpose: str,
    matrix_index: int,
    engine_id: str | None = None,
    *extra: object,
) -> SeedIdentity:
    """Derive one auditable 128-bit seed for NumPy PCG64DXSM."""

    if len(root_hex) != 64:
        raise ValueError("S12E roots must be 256-bit hexadecimal identities")
    material = _seed_material(
        root_hex, phase, purpose, matrix_index, engine_id, *extra
    )
    digest = hashlib.sha256(material).digest()
    return SeedIdentity(
        phase=phase,
        root_sha256=root_hex,
        purpose=purpose,
        matrix_index=matrix_index,
        engine_id=engine_id,
        derived_seed=int.from_bytes(digest[:16], "big", signed=False),
        seed_material_sha256=hashlib.sha256(material).hexdigest(),
    )


def generator(identity: SeedIdentity) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(identity.derived_seed))


def array_sha256(array: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(array)
    payload = (
        str(contiguous.dtype).encode()
        + b"\x00"
        + json.dumps(contiguous.shape, separators=(",", ":")).encode()
        + b"\x00"
        + contiguous.tobytes(order="C")
    )
    return hashlib.sha256(payload).hexdigest()


def generate_beta(identity: SeedIdentity) -> NDArray[np.float64]:
    """Draw the frozen historical-orientation lognormal catalytic matrix."""

    rng = generator(identity)
    return np.exp(
        BETA_A + BETA_SIGMA * rng.standard_normal((N_SPECIES, N_SPECIES))
    ).astype(np.float64, copy=False)


def initialize_distinct_state(identity: SeedIdentity) -> NDArray[np.int64]:
    """Set exactly forty distinct types to count one."""

    rng = generator(identity)
    indices = rng.choice(N_SPECIES, size=N_MIN, replace=False)
    state = np.zeros(N_SPECIES, dtype=np.int64)
    state[np.asarray(indices, dtype=np.int64)] = 1
    return state


def catalytic_boost(
    state: NDArray[np.int64], beta: NDArray[np.float64]
) -> NDArray[np.float64]:
    mass = int(state.sum())
    if mass <= 0:
        raise ValueError("catalytic boost is undefined for an empty assembly")
    return 1.0 + (beta @ state.astype(np.float64)) / float(mass)


def rates(
    state: NDArray[np.int64], beta: NDArray[np.float64], rho_each: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return boost, join rates, and loss rates from the frozen equations."""

    mass = int(state.sum())
    boost = catalytic_boost(state, beta)
    joins = K_FORWARD * float(rho_each) * float(mass) * boost
    losses = K_BACKWARD * state.astype(np.float64) * boost
    if (
        not np.all(np.isfinite(joins))
        or not np.all(np.isfinite(losses))
        or np.any(joins < 0)
        or np.any(losses < 0)
    ):
        raise ValueError("nonfinite or negative GARD rates")
    return boost, joins, losses


def categorical_update(
    state: NDArray[np.int64],
    beta: NDArray[np.float64],
    rho_each: float,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    _, joins, losses = rates(state, beta, rho_each)
    weights = np.concatenate((joins, losses))
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("categorical update has zero total rate")
    selected = int(rng.choice(weights.size, p=weights / total))
    result = state.copy()
    if selected < N_SPECIES:
        result[selected] += 1
    else:
        result[selected - N_SPECIES] -= 1
    if np.any(result < 0):
        raise AssertionError("zero-count loss had nonzero probability")
    return result


def poisson_vector_update(
    state: NDArray[np.int64],
    beta: NDArray[np.float64],
    rho_each: float,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    _, joins, losses = rates(state, beta, rho_each)
    join_draws = rng.poisson(joins).astype(np.int64, copy=False)
    attempted_losses = rng.poisson(losses).astype(np.int64, copy=False)
    applied_losses = np.minimum(attempted_losses, state)
    result = state + join_draws - applied_losses
    if np.any(result < 0):
        raise AssertionError("clipped Poisson update became negative")
    return result.astype(np.int64, copy=False)


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
    mass_a, mass_b = int(child_a.sum()), int(child_b.sum())
    if rule == "first_literal":
        return child_a.copy(), "A"
    if rule == "uniform_literal":
        if int(rng.integers(0, 2)) == 0:
            return child_a.copy(), "A"
        return child_b.copy(), "B"
    nonempty: list[tuple[NDArray[np.int64], str]] = []
    if mass_a > 0:
        nonempty.append((child_a, "A"))
    if mass_b > 0:
        nonempty.append((child_b, "B"))
    if not nonempty:
        return child_a.copy(), "NONE"
    chosen = nonempty[int(rng.integers(0, len(nonempty)))]
    return chosen[0].copy(), chosen[1]


def _trajectory_digest(
    trajectory_id: str,
    engine_id: str,
    observations: list[Observation],
    generations: list[GenerationSummary],
    terminal_status: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(VERSION.encode())
    digest.update(trajectory_id.encode())
    digest.update(engine_id.encode())
    digest.update(terminal_status.encode())
    for row in observations:
        digest.update(
            np.asarray(
                [
                    row.observation_index,
                    row.generation,
                    row.growth_generation_one_based,
                    row.molecular_step,
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
                    "generation": row.generation_one_based,
                    "status": row.growth_terminal_status,
                    "updates": row.update_count,
                    "pre": row.pre_fission_mass,
                    "a": row.child_a_mass,
                    "b": row.child_b_mass,
                    "selected": row.selected_daughter,
                    "selectedMass": row.selected_mass,
                    "overshoot": row.overshoot,
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
    engine_id: str,
    beta: NDArray[np.float64] | None = None,
    initial_state: NDArray[np.int64] | None = None,
) -> tuple[GardTrajectory, tuple[SeedIdentity, ...]]:
    """Simulate one candidate lineage and retain every required boundary state."""

    if engine_id not in ENGINE_DEFINITIONS:
        raise ValueError(f"unknown S12E engine {engine_id!r}")
    definition = ENGINE_DEFINITIONS[engine_id]
    beta_seed = derive_seed(root_hex, phase, "catalytic_matrix", matrix_index)
    init_seed = derive_seed(root_hex, phase, "initial_state", matrix_index)
    event_seed = derive_seed(root_hex, phase, "event_update", matrix_index, engine_id)
    fission_seed = derive_seed(root_hex, phase, "fission", matrix_index, engine_id)
    daughter_seed = derive_seed(
        root_hex, phase, "daughter_selection", matrix_index, engine_id
    )
    seeds = (beta_seed, init_seed, event_seed, fission_seed, daughter_seed)
    beta_array = generate_beta(beta_seed) if beta is None else np.asarray(beta, dtype=np.float64)
    state = (
        initialize_distinct_state(init_seed)
        if initial_state is None
        else np.asarray(initial_state, dtype=np.int64).copy()
    )
    if beta_array.shape != (N_SPECIES, N_SPECIES):
        raise ValueError("beta must be 100 by 100")
    if state.shape != (N_SPECIES,) or int(state.sum()) != N_MIN:
        raise ValueError("initial state must be a 100-vector with mass 40")
    if np.count_nonzero(state == 1) != N_MIN or np.any((state != 0) & (state != 1)):
        raise ValueError("initial state must contain forty distinct singleton types")

    event_rng = generator(event_seed)
    fission_rng = generator(fission_seed)
    daughter_rng = generator(daughter_seed)
    trajectory_id = f"E01-S12E-{phase.upper()}-{engine_id}-M{matrix_index:02d}"
    observations: list[Observation] = [
        Observation(
            observation_index=0,
            observation_kind="initial_selected_state",
            generation=0,
            growth_generation_one_based=0,
            molecular_step=0,
            generation_local_step=0,
            state=tuple(map(int, state)),
        )
    ]
    generations: list[GenerationSummary] = []
    molecular_step = 0
    completed_fissions = 0
    terminal_status = "requested_fissions_completed"
    extinction_generation: int | None = None

    for generation_one_based in range(1, N_GENERATIONS + 1):
        local_step = 0
        growth_status = "n_max_reached"
        while int(state.sum()) < N_MAX and local_step < MAX_STEPS:
            if int(state.sum()) == 0:
                growth_status = "extinct_during_growth"
                break
            if definition.update_kernel == "categorical":
                state = categorical_update(
                    state, beta_array, definition.rho_each, event_rng
                )
            else:
                state = poisson_vector_update(
                    state, beta_array, definition.rho_each, event_rng
                )
            local_step += 1
            molecular_step += 1
            observations.append(
                Observation(
                    observation_index=len(observations),
                    observation_kind="molecular_update",
                    generation=completed_fissions,
                    growth_generation_one_based=generation_one_based,
                    molecular_step=molecular_step,
                    generation_local_step=local_step,
                    state=tuple(map(int, state)),
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

        if int(state.sum()) == 0:
            generations.append(
                GenerationSummary(
                    generation_one_based,
                    growth_status,
                    local_step,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
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
        completed_fissions += 1
        state = selected
        generations.append(
            GenerationSummary(
                generation_one_based=generation_one_based,
                growth_terminal_status=growth_status,
                update_count=local_step,
                pre_fission_mass=pre_mass,
                child_a_mass=int(child_a.sum()),
                child_b_mass=int(child_b.sum()),
                selected_daughter=selected_name,
                selected_mass=int(state.sum()),
                overshoot=max(0, pre_mass - N_MAX),
            )
        )
        observations.append(
            Observation(
                observation_index=len(observations),
                observation_kind="post_fission",
                generation=completed_fissions,
                growth_generation_one_based=generation_one_based,
                molecular_step=molecular_step,
                generation_local_step=local_step,
                state=tuple(map(int, state)),
            )
        )
        if int(state.sum()) == 0:
            terminal_status = "selected_daughter_empty"
            extinction_generation = generation_one_based
            break

    digest = _trajectory_digest(
        trajectory_id, engine_id, observations, generations, terminal_status
    )
    result = GardTrajectory(
        trajectory_id=trajectory_id,
        phase=phase,
        matrix_index=matrix_index,
        engine_id=engine_id,
        beta_sha256=array_sha256(beta_array),
        initial_state_sha256=array_sha256(
            np.asarray(observations[0].state, dtype=np.int64)
        ),
        observations=tuple(observations),
        generations=tuple(generations),
        completed_fissions=completed_fissions,
        total_batch_steps=molecular_step,
        terminal_status=terminal_status,
        extinction_generation=extinction_generation,
        trajectory_sha256=digest,
    )
    return result, seeds


def trajectory_replay_equal(left: GardTrajectory, right: GardTrajectory) -> bool:
    """Exact replay includes states, boundary metadata, and terminal status."""

    return left == right and left.trajectory_sha256 == right.trajectory_sha256


def trajectory_summary(trajectory: GardTrajectory) -> dict[str, object]:
    generations = trajectory.generations
    completed = [row for row in generations if row.pre_fission_mass is not None]
    steps = np.asarray([row.update_count for row in completed], dtype=np.int64)
    pre = np.asarray([row.pre_fission_mass for row in completed], dtype=np.float64)
    post = np.asarray([row.selected_mass for row in completed], dtype=np.float64)
    child_a = np.asarray([row.child_a_mass for row in completed], dtype=np.float64)
    child_b = np.asarray([row.child_b_mass for row in completed], dtype=np.float64)
    overshoot = np.asarray([row.overshoot for row in completed], dtype=np.float64)
    maxsteps = sum(row.growth_terminal_status == "max_steps_reached" for row in completed)
    reached = sum(
        row.growth_terminal_status in {"n_max_reached", "n_max_overshot"}
        for row in completed
    )
    return {
        "trajectoryId": trajectory.trajectory_id,
        "phase": trajectory.phase,
        "matrixIndex": trajectory.matrix_index,
        "engineId": trajectory.engine_id,
        "completedFissions": trajectory.completed_fissions,
        "terminalStatus": trajectory.terminal_status,
        "extinctionGeneration": trajectory.extinction_generation,
        "totalBatchSteps": trajectory.total_batch_steps,
        "totalSourceObservations": len(trajectory.observations),
        "meanStepsPerGeneration": float(steps.mean()) if steps.size else np.nan,
        "medianStepsPerGeneration": float(np.median(steps)) if steps.size else np.nan,
        "meanPreFissionMass": float(pre.mean()) if pre.size else np.nan,
        "medianPreFissionMass": float(np.median(pre)) if pre.size else np.nan,
        "meanPostFissionMass": float(post.mean()) if post.size else np.nan,
        "medianPostFissionMass": float(np.median(post)) if post.size else np.nan,
        "meanChildAMass": float(child_a.mean()) if child_a.size else np.nan,
        "meanChildBMass": float(child_b.mean()) if child_b.size else np.nan,
        "meanOvershoot": float(overshoot.mean()) if overshoot.size else np.nan,
        "maxOvershoot": float(overshoot.max()) if overshoot.size else np.nan,
        "maxstepsTerminations": int(maxsteps),
        "fractionGenerationsReachingNMax": (
            float(reached / len(completed)) if completed else 0.0
        ),
        "betaSha256": trajectory.beta_sha256,
        "initialStateSha256": trajectory.initial_state_sha256,
        "trajectorySha256": trajectory.trajectory_sha256,
    }


def observation_rows(trajectory: GardTrajectory) -> list[dict[str, object]]:
    return [
        {
            "trajectoryId": trajectory.trajectory_id,
            "phase": trajectory.phase,
            "matrixIndex": trajectory.matrix_index,
            "engineId": trajectory.engine_id,
            "observationIndex": row.observation_index,
            "observationKind": row.observation_kind,
            "generation": row.generation,
            "growthGenerationOneBased": row.growth_generation_one_based,
            "molecularStep": row.molecular_step,
            "generationLocalStep": row.generation_local_step,
            "mass": int(sum(row.state)),
            "state": list(row.state),
        }
        for row in trajectory.observations
    ]
