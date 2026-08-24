"""Conditionally gated, post-eligibility S12 intervention simulation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from e01_gard_independent import (
    fission,
    generate_catalytic_matrix,
    grow,
    initialize_state,
)
from e01_gard_independent.records import EventLog, FissionLog
from e01_gard_reproducibility import (
    CouplingPolicy,
    SeedBundle,
    SeedRequest,
    StreamPurpose,
    derive_seed_bundle,
    isolated_stream_namespace,
)

from .core import (
    ENGINE_ID,
    GARD_SPECIFICATION_ID,
    MINIMUM_EFFECTIVE_SAMPLES,
    NUMERIC_TOLERANCE,
    PREPROCESSING_IDS,
    ROOT_SEED_HEX,
    PartitionLock,
    action_null_envelope,
    build_baseline_specification,
    find_past_only_partition_lock,
    preprocess_states,
    score_action_candidates,
)

Condition = Literal["max", "control", "min"]


@dataclass(frozen=True, slots=True)
class InterventionTrajectory:
    """One complete condition with status-bearing actions and candidate scores."""

    matrix_index: int
    trajectory_id: str
    condition: Condition
    seed_payload: dict[str, Any]
    beta: NDArray[np.float64]
    initial_state: tuple[int, ...]
    final_state: tuple[int, ...]
    states: NDArray[np.int64]
    observation_kinds: tuple[str, ...]
    generations: NDArray[np.int64]
    molecular_steps: NDArray[np.int64]
    events: tuple[EventLog, ...]
    fissions: tuple[FissionLog, ...]
    action_rows: tuple[dict[str, Any], ...]
    candidate_rows: tuple[dict[str, Any], ...]
    partition_rows: tuple[dict[str, Any], ...]
    partition_locks: dict[str, PartitionLock]
    completed_fissions: int
    trajectory_sha256: str
    runtime_seconds: float


def intervention_seed_bundle(matrix_index: int, condition: Condition) -> SeedBundle:
    """Share six GARD streams with baseline while isolating auxiliary streams."""

    if matrix_index not in range(12):
        raise ValueError("matrix_index must be in 0..11")
    if condition not in ("max", "control", "min"):
        raise ValueError("unknown intervention condition")
    baseline_trajectory = f"E01-S12-B{matrix_index:02d}"
    common_namespace = isolated_stream_namespace(
        experiment_id="E01",
        specification_id=GARD_SPECIFICATION_ID,
        trajectory_id=baseline_trajectory,
        replicate_index=matrix_index,
    )
    trajectory_id = f"E01-S12-I{matrix_index:02d}-{condition}"
    auxiliary_namespace = isolated_stream_namespace(
        experiment_id="E01",
        specification_id=GARD_SPECIFICATION_ID,
        trajectory_id=trajectory_id,
        replicate_index=matrix_index,
    )
    common = {
        StreamPurpose.CATALYTIC_MATRIX,
        StreamPurpose.INITIAL_STATE,
        StreamPurpose.EVENT,
        StreamPurpose.WAITING_TIME,
        StreamPurpose.FISSION,
        StreamPurpose.DAUGHTER_SELECTION,
    }
    namespaces = {
        purpose: common_namespace if purpose in common else auxiliary_namespace
        for purpose in StreamPurpose
    }
    request = SeedRequest(
        experiment_id="E01",
        specification_id=GARD_SPECIFICATION_ID,
        trajectory_id=trajectory_id,
        replicate_index=matrix_index,
        engine_id=ENGINE_ID,
        root_seed_hex=ROOT_SEED_HEX,
        coupling_policy=CouplingPolicy.EXPLICIT_COMMON_RANDOM_NUMBERS,
        coupling_reason=(
            "S12 paired max/control/min share catalytic, initial, event, waiting, "
            "fission, and daughter streams until state divergence."
        ),
        stream_namespaces=namespaces,
    )
    return derive_seed_bundle(request)


def _trajectory_digest(
    beta: NDArray[np.float64],
    states: NDArray[np.int64],
    final_state: tuple[int, ...],
    actions: list[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    digest.update(beta.astype("<f8", copy=False).tobytes(order="C"))
    digest.update(states.astype("<i8", copy=False).tobytes(order="C"))
    digest.update(np.asarray(final_state, dtype="<i8").tobytes())
    compact = [
        {
            "generation": row["generation"],
            "status": row["status"],
            "candidateId": row.get("selectedCandidateId"),
        }
        for row in actions
    ]
    digest.update(json.dumps(compact, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _attempt_current_partition(
    coordinates: NDArray[np.float64],
    *,
    preprocessing_id: str,
    generations: NDArray[np.int64],
    molecular_steps: NDArray[np.int64],
    estimator_rng: np.random.Generator,
) -> PartitionLock:
    kinds = ["molecular_event"] * coordinates.shape[0]
    kinds[-1] = "post_fission"
    return find_past_only_partition_lock(
        coordinates,
        preprocessing_id=preprocessing_id,
        observation_kinds=tuple(kinds),
        generations=generations,
        molecular_steps=molecular_steps,
        estimator_rng=estimator_rng,
    )


def _choose_action(
    score_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    *,
    direction: Literal["max", "min"],
    intervention_rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(score_rows) != len(replay_rows):
        raise ValueError("candidate replay cardinality mismatch")
    replay_errors: list[float] = []
    for left, right in zip(score_rows, replay_rows, strict=True):
        if (
            left["candidateId"] != right["candidateId"]
            or left["preprocessingId"] != right["preprocessingId"]
            or left["status"] != right["status"]
        ):
            return (
                {
                    "status": "INELIGIBLE_CANDIDATE_REPLAY_MISMATCH",
                    "reason": "CANDIDATE_ID_OR_STATUS_REPLAY_MISMATCH",
                },
                [],
            )
        if left["score"] is not None and right["score"] is not None:
            replay_errors.append(abs(float(left["score"]) - float(right["score"])))
    epsilon_replay = max(replay_errors, default=0.0)
    epsilon_numeric = max(NUMERIC_TOLERANCE, epsilon_replay)

    diagnostics: list[dict[str, Any]] = []
    winners: list[str] = []
    candidate_states: dict[str, list[int]] = {}
    for preprocessing_id in PREPROCESSING_IDS:
        subset = [
            row for row in score_rows if row["preprocessingId"] == preprocessing_id
        ]
        if not subset or any(
            row["status"] != "ELIGIBLE_NUMERIC_STRICT_EXPANDING" for row in subset
        ):
            return (
                {
                    "status": "INELIGIBLE_CANDIDATE_STRICT_GATE",
                    "reason": "ONE_OR_MORE_CANDIDATES_FAILED_STRICT_GATE",
                    "epsilonNumeric": epsilon_numeric,
                },
                diagnostics,
            )
        extreme = (
            max(float(row["score"]) for row in subset)
            if direction == "max"
            else min(float(row["score"]) for row in subset)
        )
        tied_extreme = [
            row for row in subset if abs(float(row["score"]) - extreme) <= 1.0e-12
        ]
        if len(tied_extreme) != 1:
            return (
                {
                    "status": "INELIGIBLE_ACTION_NOT_SEPARABLE",
                    "reason": "MULTIPLE_CANDIDATES_WITHIN_NUMERICAL_TIE_TOLERANCE",
                    "epsilonNumeric": epsilon_numeric,
                },
                diagnostics,
            )
        best = tied_extreme[0]
        remaining = [row for row in subset if row is not best]
        runner_up = (
            max(remaining, key=lambda row: float(row["score"]))
            if direction == "max"
            else min(remaining, key=lambda row: float(row["score"]))
        )
        gap = (
            float(best["score"]) - float(runner_up["score"])
            if direction == "max"
            else float(runner_up["score"]) - float(best["score"])
        )
        envelope = action_null_envelope(
            subset,
            direction=direction,
            rng=intervention_rng,
            families=4096,
        )
        threshold = (
            epsilon_numeric + float(envelope["threshold"])
            if envelope["threshold"] is not None
            else None
        )
        separable = (
            envelope["status"] == "ELIGIBLE"
            and threshold is not None
            and gap > threshold
            and gap > 1.0e-12
        )
        diagnostics.append(
            {
                "preprocessingId": preprocessing_id,
                "direction": direction,
                "bestCandidateId": best["candidateId"],
                "runnerUpCandidateId": runner_up["candidateId"],
                "bestScore": best["score"],
                "runnerUpScore": runner_up["score"],
                "deltaAction": gap,
                "epsilonNumeric": epsilon_numeric,
                "nullEnvelope": envelope,
                "requiredSeparation": threshold,
                "separable": separable,
            }
        )
        if not separable:
            return (
                {
                    "status": "INELIGIBLE_ACTION_NOT_SEPARABLE",
                    "reason": "BEST_RUNNER_UP_GAP_DID_NOT_EXCEED_NUMERIC_PLUS_FULLSET_NULL",
                    "epsilonNumeric": epsilon_numeric,
                },
                diagnostics,
            )
        winners.append(best["candidateId"])
        candidate_states[best["candidateId"]] = best["candidateState"]
    if len(set(winners)) != 1:
        return (
            {
                "status": "INELIGIBLE_ACTION_NOT_SEPARABLE",
                "reason": "PREPROCESSING_BRANCH_WINNERS_DISAGREE",
                "epsilonNumeric": epsilon_numeric,
            },
            diagnostics,
        )
    winner = winners[0]
    return (
        {
            "status": (
                "ELIGIBLE_ACTION_NOOP_SELECTED"
                if winner == "noop"
                else "ELIGIBLE_ACTION_APPLIED"
            ),
            "reason": None,
            "selectedCandidateId": winner,
            "selectedState": candidate_states[winner],
            "epsilonNumeric": epsilon_numeric,
        },
        diagnostics,
    )


def simulate_intervention(
    matrix_index: int,
    condition: Condition,
    *,
    first_authorized_generation: int,
) -> InterventionTrajectory:
    """Run one complete condition under the frozen strict action policy."""

    if first_authorized_generation not in range(1, 101):
        raise ValueError("first_authorized_generation must be in 1..100")

    started = time.perf_counter()
    specification = build_baseline_specification()
    bundle = intervention_seed_bundle(matrix_index, condition)
    generators = bundle.fresh_generators()
    streams = bundle.independent_engine_streams(generators)
    beta = generate_catalytic_matrix(specification, streams.catalytic_matrix)
    initial = initialize_state(specification, streams.initialization)
    current = tuple(initial)

    states: list[tuple[int, ...]] = [current]
    kinds = ["initial_selected_state"]
    generations = [0]
    molecular_steps = [0]
    molecular_step = 0
    events: list[EventLog] = []
    fissions: list[FissionLog] = []
    action_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    locks: dict[str, PartitionLock] = {}
    estimator_rng = generators[StreamPurpose.ESTIMATOR]
    intervention_rng = generators[StreamPurpose.INTERVENTION]

    for generation in range(1, specification.n_generations + 1):
        growth_result = grow(
            current,
            beta=beta,
            specification=specification,
            rng_streams=streams,
            generation_index_one_based=generation,
        )
        if growth_result.terminal_status not in {"n_max_reached", "n_max_overshot"}:
            raise RuntimeError(
                f"intervention trajectory stopped during growth: {growth_result.terminal_status}"
            )
        for event in growth_result.events:
            events.append(event)
            molecular_step += 1
            states.append(event.post_state)
            kinds.append("molecular_event")
            generations.append(generation - 1)
            molecular_steps.append(molecular_step)
        split = fission(
            growth_result.final_state,
            specification=specification,
            rng_streams=streams,
            generation_index_one_based=generation,
        )
        fissions.append(split)
        selected = np.asarray(split.selected_daughter, dtype=np.int64)
        states.append(split.selected_daughter)
        kinds.append("post_fission")
        generations.append(generation)
        molecular_steps.append(molecular_step)
        observation_index = len(states) - 1

        action_base = {
            "trajectoryId": f"E01-S12-I{matrix_index:02d}-{condition}",
            "matrixIndex": matrix_index,
            "condition": condition,
            "generation": generation,
            "observationIndex": observation_index,
            "molecularStep": molecular_step,
            "nEff": observation_index,
            "firstAuthorizedGeneration": first_authorized_generation,
            "preActionState": selected.tolist(),
        }
        if condition == "control":
            action_rows.append(
                {
                    **action_base,
                    "status": "CONTROL_NO_INTERVENTION",
                    "reason": None,
                    "selectedCandidateId": "noop",
                    "postActionState": selected.tolist(),
                    "candidateCount": 0,
                    "diagnostics": [],
                }
            )
            current = tuple(int(value) for value in selected)
            continue
        if generation < first_authorized_generation:
            action_rows.append(
                {
                    **action_base,
                    "status": "INELIGIBLE_PRE_COMMON_RISK_ORIGIN",
                    "reason": "INTERVENTION_NOT_AUTHORIZED_BEFORE_FROZEN_COMMON_RISK_ORIGIN",
                    "selectedCandidateId": None,
                    "postActionState": selected.tolist(),
                    "candidateCount": 0,
                    "diagnostics": [],
                }
            )
            current = tuple(int(value) for value in selected)
            continue
        if observation_index < MINIMUM_EFFECTIVE_SAMPLES:
            action_rows.append(
                {
                    **action_base,
                    "status": "INELIGIBLE_PRE_512",
                    "reason": "INSUFFICIENT_EFFECTIVE_SAMPLES",
                    "selectedCandidateId": None,
                    "postActionState": selected.tolist(),
                    "candidateCount": 0,
                    "diagnostics": [],
                }
            )
            current = tuple(int(value) for value in selected)
            continue

        state_matrix = np.asarray(states, dtype=np.int64)
        preprocessing = preprocess_states(state_matrix)
        generation_array = np.asarray(generations, dtype=np.int64)
        molecular_array = np.asarray(molecular_steps, dtype=np.int64)
        for preprocessing_id in PREPROCESSING_IDS:
            if preprocessing_id in locks:
                continue
            attempt = _attempt_current_partition(
                preprocessing.coordinates[preprocessing_id],
                preprocessing_id=preprocessing_id,
                generations=generation_array,
                molecular_steps=molecular_array,
                estimator_rng=estimator_rng,
            )
            partition_rows.extend(
                {
                    **{
                        key: value
                        for key, value in row.items()
                        if key != "candidateState"
                    },
                    "trajectoryId": action_base["trajectoryId"],
                    "matrixIndex": matrix_index,
                    "condition": condition,
                }
                for row in attempt.history
                if row["observationIndex"] == observation_index
            )
            if attempt.status == "ELIGIBLE_LOCKED":
                locks[preprocessing_id] = attempt
        if set(locks) != set(PREPROCESSING_IDS):
            action_rows.append(
                {
                    **action_base,
                    "status": "INELIGIBLE_CANDIDATE_PARTITION_GATE",
                    "reason": "BOTH_PREPROCESSING_PARTITIONS_NOT_LOCKED",
                    "selectedCandidateId": None,
                    "postActionState": selected.tolist(),
                    "candidateCount": 0,
                    "diagnostics": [],
                }
            )
            current = tuple(int(value) for value in selected)
            continue

        scores = score_action_candidates(
            selected,
            preprocessing_coordinates=preprocessing.coordinates,
            locks=locks,
        )
        replay = score_action_candidates(
            selected,
            preprocessing_coordinates=preprocessing.coordinates,
            locks=locks,
        )
        decision, diagnostics = _choose_action(
            scores,
            replay,
            direction=condition,
            intervention_rng=intervention_rng,
        )
        for row in scores:
            candidate_rows.append(
                {
                    **{
                        key: value
                        for key, value in action_base.items()
                        if key != "preActionState"
                    },
                    **row,
                    "replayScore": next(
                        item["score"]
                        for item in replay
                        if item["candidateId"] == row["candidateId"]
                        and item["preprocessingId"] == row["preprocessingId"]
                    ),
                }
            )
        post = np.asarray(decision.get("selectedState", selected), dtype=np.int64)
        action_rows.append(
            {
                **action_base,
                **{
                    key: value
                    for key, value in decision.items()
                    if key != "selectedState"
                },
                "postActionState": post.tolist(),
                "candidateCount": len(scores) // len(PREPROCESSING_IDS),
                "diagnostics": diagnostics,
            }
        )
        current = tuple(int(value) for value in post)

    state_array = np.asarray(states, dtype=np.int64)
    final = tuple(int(value) for value in current)
    digest = _trajectory_digest(beta, state_array, final, action_rows)
    return InterventionTrajectory(
        matrix_index=matrix_index,
        trajectory_id=f"E01-S12-I{matrix_index:02d}-{condition}",
        condition=condition,
        seed_payload=bundle.to_payload(),
        beta=beta,
        initial_state=tuple(initial),
        final_state=final,
        states=state_array,
        observation_kinds=tuple(kinds),
        generations=np.asarray(generations, dtype=np.int64),
        molecular_steps=np.asarray(molecular_steps, dtype=np.int64),
        events=tuple(events),
        fissions=tuple(fissions),
        action_rows=tuple(action_rows),
        candidate_rows=tuple(candidate_rows),
        partition_rows=tuple(partition_rows),
        partition_locks=locks,
        completed_fissions=len(fissions),
        trajectory_sha256=digest,
        runtime_seconds=time.perf_counter() - started,
    )


def intervention_event_rows(
    trajectory: InterventionTrajectory,
) -> list[dict[str, Any]]:
    """Serialize complete engine event/fission records for one condition."""

    rows: list[dict[str, Any]] = []
    for event in trajectory.events:
        rows.append(
            {
                "trajectoryId": trajectory.trajectory_id,
                "matrixIndex": trajectory.matrix_index,
                "condition": trajectory.condition,
                "recordType": "molecular_event",
                "recordPayload": asdict(event),
            }
        )
    for split in trajectory.fissions:
        rows.append(
            {
                "trajectoryId": trajectory.trajectory_id,
                "matrixIndex": trajectory.matrix_index,
                "condition": trajectory.condition,
                "recordType": "fission",
                "recordPayload": asdict(split),
            }
        )
    return rows
