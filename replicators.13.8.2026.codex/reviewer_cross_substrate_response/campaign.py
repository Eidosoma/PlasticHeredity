"""Simulation orchestration for the cross-substrate CA campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .core import (
    BREAK_HORIZON,
    MAX_FUTURE_HORIZON,
    canonical_similarity,
    derive_seed,
    exact_order_null_probability,
    score_break_renewal,
)
from .models import (
    BoundaryTransition,
    EvoloopParameters,
    EvoloopRule,
    EvoloopWorld,
    ModelProfile,
    ProtocellParameters,
    ProtocellWorld,
    advance_evoloop_boundary,
    advance_protocell_boundary,
    evoloop_initial,
    mechanics_cells,
    protocell_initial,
)


MODEL_NAMES = ("protocell", "evoloop")


@dataclass(frozen=True)
class MechanicsResult:
    model: str
    parameter_key: str
    seed_index: int
    completed_boundaries: int
    ambiguous_boundaries: int
    occupancy_exceeded: bool
    total_updates: int
    passed: bool


@dataclass(frozen=True)
class CalibrationPair:
    model: str
    block_id: int
    parameter_key: str
    boundary_index: int
    parent_size: int
    child_size: int
    actual_similarity: float
    stranger_similarity: float


def parameter_payload(
    value: ProtocellParameters | EvoloopParameters,
) -> dict[str, Any]:
    payload = asdict(value)
    payload["parameter_key"] = value.key
    payload["kind"] = "protocell" if isinstance(value, ProtocellParameters) else "evoloop"
    return payload


def parameter_from_payload(payload: dict[str, Any]) -> ProtocellParameters | EvoloopParameters:
    if payload["kind"] == "protocell":
        return ProtocellParameters(
            p_y=float(payload["p_y"]),
            a_y=float(payload["a_y"]),
            p_x=float(payload["p_x"]),
            a_x=float(payload.get("a_x", 0.01)),
        )
    if payload["kind"] == "evoloop":
        return EvoloopParameters(
            initial_count=int(payload["initial_count"]),
            immigration_per_10000=float(payload["immigration_per_10000"]),
        )
    raise ValueError(f"unknown parameter payload: {payload.get('kind')}")


@lru_cache(maxsize=1)
def evoloop_rule() -> EvoloopRule:
    return EvoloopRule()


def initial_world(
    model: str,
    parameters: ProtocellParameters | EvoloopParameters,
    profile: ModelProfile,
    rng: np.random.Generator,
) -> ProtocellWorld | EvoloopWorld:
    if model == "protocell" and isinstance(parameters, ProtocellParameters):
        return protocell_initial(profile.side)
    if model == "evoloop" and isinstance(parameters, EvoloopParameters):
        return evoloop_initial(profile.side, parameters.initial_count, rng)
    raise TypeError("model and parameters disagree")


def advance_boundary(
    model: str,
    world: ProtocellWorld | EvoloopWorld,
    parameters: ProtocellParameters | EvoloopParameters,
    profile: ModelProfile,
    rng: np.random.Generator,
) -> tuple[ProtocellWorld | EvoloopWorld, BoundaryTransition]:
    if model == "protocell" and isinstance(world, ProtocellWorld) and isinstance(parameters, ProtocellParameters):
        return advance_protocell_boundary(
            world,
            parameters,
            rng,
            cap=profile.boundary_cap,
            persistence=profile.protocell_persistence,
        )
    if model == "evoloop" and isinstance(world, EvoloopWorld) and isinstance(parameters, EvoloopParameters):
        return advance_evoloop_boundary(
            world,
            parameters,
            evoloop_rule(),
            rng,
            cap=profile.boundary_cap,
            persistence=profile.evoloop_persistence,
            arm_window=profile.evoloop_arm_window,
        )
    raise TypeError("model, world, and parameters disagree")


def mechanics_trial(
    model: str,
    parameter_data: dict[str, Any],
    profile: ModelProfile,
    seed_index: int,
) -> MechanicsResult:
    parameters = parameter_from_payload(parameter_data)
    rng = np.random.default_rng(derive_seed("mechanics", profile.name, model, parameters.key, seed_index))
    try:
        world = initial_world(model, parameters, profile, rng)
    except RuntimeError:
        return MechanicsResult(model, parameters.key, seed_index, 0, 0, False, 0, False)
    completed = 0
    ambiguous = 0
    occupancy = False
    updates = 0
    while completed < profile.mechanics_boundaries and updates < profile.mechanics_cap:
        world, transition = advance_boundary(model, world, parameters, profile, rng)
        updates += int(transition.elapsed_updates)
        ambiguous += int(transition.ambiguous)
        occupancy = occupancy or bool(transition.occupancy_exceeded)
        if transition.extinct or transition.ambiguous or transition.occupancy_exceeded:
            break
        completed += 1
    denominator = max(completed + ambiguous, 1)
    passed = bool(
        completed >= profile.mechanics_boundaries
        and not occupancy
        and ambiguous / denominator < 0.05
        and updates <= profile.mechanics_cap
    )
    return MechanicsResult(
        model=model,
        parameter_key=parameters.key,
        seed_index=seed_index,
        completed_boundaries=completed,
        ambiguous_boundaries=ambiguous,
        occupancy_exceeded=occupancy,
        total_updates=updates,
        passed=passed,
    )


def mechanics_jobs(model: str, profile: ModelProfile) -> list[tuple[str, dict[str, Any], ModelProfile, int]]:
    cells = mechanics_cells(model)
    if profile.name == "smoke":
        if model == "protocell":
            cells = [ProtocellParameters.from_pair(1e-2, 1e-4)]
        else:
            cells = [EvoloopParameters(1, 0.0)]
    return [
        (model, parameter_payload(parameters), profile, seed_index)
        for parameters in cells
        for seed_index in range(profile.mechanics_seeds)
    ]


def copy_world(world: ProtocellWorld | EvoloopWorld) -> ProtocellWorld | EvoloopWorld:
    return world.copy()


def world_to_arrays(world: ProtocellWorld | EvoloopWorld) -> dict[str, NDArray[Any]]:
    if isinstance(world, ProtocellWorld):
        return {"kind": np.asarray([0], dtype=np.int8), "grid": world.grid}
    return {
        "kind": np.asarray([1], dtype=np.int8),
        "grid": world.grid,
        "provenance": world.provenance,
        "focal_label": np.asarray([world.focal_label], dtype=np.int32),
        "next_label": np.asarray([world.next_label], dtype=np.int32),
    }


def world_from_arrays(arrays: dict[str, NDArray[Any]]) -> ProtocellWorld | EvoloopWorld:
    kind = int(np.asarray(arrays["kind"]).ravel()[0])
    if kind == 0:
        return ProtocellWorld(np.asarray(arrays["grid"], dtype=np.uint8).copy())
    if kind == 1:
        return EvoloopWorld(
            np.asarray(arrays["grid"], dtype=np.uint8).copy(),
            np.asarray(arrays["provenance"], dtype=np.int32).copy(),
            int(np.asarray(arrays["focal_label"]).ravel()[0]),
            int(np.asarray(arrays["next_label"]).ravel()[0]),
        )
    raise ValueError(f"unknown serialized world kind: {kind}")


def generate_landmarks(
    model: str,
    parameters: ProtocellParameters | EvoloopParameters,
    profile: ModelProfile,
    block_id: int,
    *,
    stage: str,
    attempts: int = 100,
) -> tuple[list[ProtocellWorld | EvoloopWorld], list[dict[str, Any]], int | None]:
    target = set(profile.landmarks)
    for attempt in range(attempts):
        rng = np.random.default_rng(derive_seed(stage, "main", model, block_id, attempt))
        try:
            world = initial_world(model, parameters, profile, rng)
        except RuntimeError:
            continue
        states: list[ProtocellWorld | EvoloopWorld] = []
        history: list[dict[str, Any]] = []
        failed = False
        for boundary in range(1, max(profile.landmarks) + 1):
            world, transition = advance_boundary(model, world, parameters, profile, rng)
            if transition.extinct or transition.ambiguous or transition.occupancy_exceeded:
                failed = True
                break
            similarity = canonical_similarity(transition.parent, transition.child)
            history.append(
                {
                    "boundary": boundary,
                    "similarity": similarity,
                    "parent_size": int(np.count_nonzero(transition.parent)),
                    "child_size": int(np.count_nonzero(transition.child)),
                    "elapsed_updates": transition.elapsed_updates,
                }
            )
            if boundary in target:
                states.append(copy_world(world))
        if not failed and len(states) == len(profile.landmarks):
            return states, history, attempt
    return [], [], None


def _size_ratio(left: int, right: int) -> float:
    return float(right / left) if left > 0 else float("inf")


def assign_matched_strangers(
    observations: list[dict[str, Any]],
    *,
    seed_parts: Sequence[object],
    different_key: str | None = None,
) -> None:
    """Add deterministic matched-stranger similarities in place."""

    if len(observations) < 2:
        for item in observations:
            item["stranger_similarity"] = float("nan")
        return
    order = np.random.default_rng(derive_seed(*seed_parts)).permutation(len(observations))
    positions = np.empty(len(observations), dtype=np.int64)
    positions[order] = np.arange(len(observations))
    for index, item in enumerate(observations):
        selected: int | None = None
        for widen in ((0.8, 1.25), (0.67, 1.5)):
            for offset in range(1, len(observations)):
                candidate_index = int(
                    order[(int(positions[index]) + offset) % len(order)]
                )
                if candidate_index == index:
                    continue
                candidate = observations[candidate_index]
                if different_key is not None and candidate.get(different_key) == item.get(different_key):
                    continue
                # Match the negative-control child to the focal offspring's
                # size.  The similarity being controlled is parent-to-child,
                # but the nuisance characteristic belongs to the two child
                # rasters, not to the focal parent.
                ratio = _size_ratio(item["child_size"], candidate["child_size"])
                if widen[0] <= ratio <= widen[1]:
                    selected = candidate_index
                    break
            if selected is not None:
                break
        if selected is None:
            item["stranger_similarity"] = float("nan")
        else:
            item["stranger_similarity"] = canonical_similarity(
                item["parent_crop"], observations[selected]["child_crop"]
            )


def calibration_block(
    model: str,
    parameter_data: dict[str, Any],
    profile: ModelProfile,
    block_id: int,
) -> list[dict[str, Any]]:
    parameters = parameter_from_payload(parameter_data)
    observations: list[dict[str, Any]] = []
    for attempt in range(100):
        rng = np.random.default_rng(derive_seed("calibration", model, block_id, attempt))
        try:
            world = initial_world(model, parameters, profile, rng)
        except RuntimeError:
            continue
        while len(observations) < profile.calibration_pairs:
            world, transition = advance_boundary(model, world, parameters, profile, rng)
            if transition.extinct or transition.ambiguous or transition.occupancy_exceeded:
                break
            observations.append(
                {
                    "boundary_index": len(observations),
                    "lineage_attempt": attempt,
                    "parent_size": int(np.count_nonzero(transition.parent)),
                    "child_size": int(np.count_nonzero(transition.child)),
                    "parent_crop": transition.parent,
                    "child_crop": transition.child,
                    "actual_similarity": canonical_similarity(transition.parent, transition.child),
                }
            )
        if len(observations) >= profile.calibration_pairs:
            break
    for item in observations:
        item.update(
            {
                "model": model,
                "block_id": block_id,
                "parameter_key": parameters.key,
            }
        )
    return observations


def simulate_state_futures(
    model: str,
    parameters: ProtocellParameters | EvoloopParameters,
    profile: ModelProfile,
    world: ProtocellWorld | EvoloopWorld,
    threshold: float,
    *,
    stage: str,
    block_id: int,
    landmark: int,
    branches: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    future_rows: list[dict[str, Any]] = []
    observations_by_boundary: dict[int, list[dict[str, Any]]] = {}
    per_branch_similarities: list[list[float]] = []
    per_branch_failure: list[str] = []
    for branch in range(branches):
        rng = np.random.default_rng(
            derive_seed(stage, "future", model, block_id, landmark, branch)
        )
        current = copy_world(world)
        similarities: list[float] = []
        failure = ""
        for boundary in range(MAX_FUTURE_HORIZON):
            current, transition = advance_boundary(model, current, parameters, profile, rng)
            if transition.extinct or transition.ambiguous or transition.occupancy_exceeded:
                failure = (
                    "ambiguous" if transition.ambiguous else
                    "occupancy" if transition.occupancy_exceeded else "extinct_or_timeout"
                )
                break
            similarity = canonical_similarity(transition.parent, transition.child)
            similarities.append(similarity)
            observations_by_boundary.setdefault(boundary, []).append(
                {
                    "branch": branch,
                    "boundary": boundary,
                    "parent_crop": transition.parent,
                    "child_crop": transition.child,
                    "parent_size": int(np.count_nonzero(transition.parent)),
                    "child_size": int(np.count_nonzero(transition.child)),
                    "actual_similarity": similarity,
                    "elapsed_updates": transition.elapsed_updates,
                }
            )
        per_branch_similarities.append(similarities)
        per_branch_failure.append(failure)

    boundary_lookup = {
        (int(item["branch"]), int(item["boundary"])): item
        for observations in observations_by_boundary.values()
        for item in observations
    }
    crop_rows: list[dict[str, Any]] = []
    half_width = branches // 2
    for branch, similarities in enumerate(per_branch_similarities):
        future_id = f"{model}:{stage}:{block_id}:{landmark}:{branch}"
        outcome = score_break_renewal(
            similarities,
            threshold,
            complete_horizon=len(similarities) >= BREAK_HORIZON,
        )
        null_probability = exact_order_null_probability(
            outcome.observed_boundaries, outcome.inherited_count
        )
        future_rows.append(
            {
                "model": model,
                "stage": stage,
                "block_id": block_id,
                "parameter_key": parameters.key,
                "landmark": landmark,
                "branch": branch,
                "future_id": future_id,
                "main_complete": 1,
                "half": "A" if branch < half_width else "B",
                "event": int(outcome.event),
                "break_index": outcome.break_index,
                "renewal_start": outcome.renewal_start,
                "observed_boundaries": outcome.observed_boundaries,
                "inherited_count": outcome.inherited_count,
                "complete_horizon": int(outcome.complete_horizon),
                "order_null_probability": null_probability,
                "event_minus_order_null": int(outcome.event) - null_probability,
                "failure": per_branch_failure[branch],
            }
        )
        for boundary, similarity in enumerate(similarities):
            item = boundary_lookup[(branch, boundary)]
            observation_index = len(crop_rows)
            crop_rows.append(
                {
                    "observation_index": observation_index,
                    "landmark": landmark,
                    "boundary": boundary,
                    "branch": branch,
                    "parent_crop": item["parent_crop"],
                    "child_crop": item["child_crop"],
                }
            )
            item_row = {
                "model": model,
                "stage": stage,
                "block_id": block_id,
                "parameter_key": parameters.key,
                "landmark": landmark,
                "branch": branch,
                "half": "A" if branch < half_width else "B",
                "future_id": future_id,
                "boundary": boundary,
                "similarity": similarity,
                "stranger_similarity": float("nan"),
                "inherited": int(similarity > threshold),
                "parent_size": item["parent_size"],
                "child_size": item["child_size"],
                "elapsed_updates": item["elapsed_updates"],
                "observation_index": observation_index,
            }
            # Rasters are kept in a compressed raw-data sidecar, never in the
            # rectangular scientific table.
            item.clear()
            item.update(item_row)

    boundary_rows = [
        item
        for boundary in sorted(observations_by_boundary)
        for item in observations_by_boundary[boundary]
    ]
    return future_rows, boundary_rows, crop_rows


def assigned_parameter(
    viable_payloads: Sequence[dict[str, Any]], block_id: int
) -> ProtocellParameters | EvoloopParameters:
    if not viable_payloads:
        raise ValueError("at least one viable mechanics cell is required")
    ordered = sorted(viable_payloads, key=lambda item: str(item["parameter_key"]))
    return parameter_from_payload(ordered[block_id % len(ordered)])
