"""Frozen online prefix-refit intervention machinery for E01/S17.

The scorer deliberately delegates every hypothetical action to the already
confirmed PhiRL source pipeline.  It does not reuse the inexpensive completed-
control scorer from S13X and it never reads a future observation.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from e01_frozen_timebase_ensemble.core import frozen_clr
from e01_latent_timebase.core import (
    MAX_STEPS,
    N_GENERATIONS,
    N_MAX,
    ExposureDefinition,
    GenerationSummary,
    SimulationDefinition,
    StateObservation,
    TimebaseTrajectory,
    _trajectory_digest,
    array_sha256,
    derive_seed,
    fission,
    generate_beta,
    generator,
    initialize_distinct_state,
    poisson_update,
    select_daughter,
)
from e01_source_emergence_metric_identity.core import (
    EmergenceAuditResult,
    result_replay_equal,
    run_emergence_pipeline,
)

VERSION = "E01-S17-INTERVENTION-RECONSTRUCTION-v1.0.0"
RESEARCH_STEP_ID = "S17"
PHASE = "s17_intervention_reconstruction"
BENCHMARK_PHASE = "s17_runtime_benchmark_only"
ROOT_HEX = "554b7d8421c0eadfd740cf9c5f13af7937b462898abb72ed5981331744c73f6a"
BENCHMARK_ROOT_HEX = (
    "4bfd5bcbbb3c2be6ff56cc693a630558e995c6a93265bbb29e37103ca27a46a0"
)
TIE_ROOT_HEX = "58e58f81a76ec57e7f70e14b6486d177a1bd3fed2307ee69e2db62911c4855c8"
SOURCE_ROOT_HEX = (
    "06153d01fca8bbbcc7381cfd0950d152c4f6b95efdf34410324686a8e05d7b71"
)
HISTORICAL_REPLAY_ENVELOPE = 8.331113576787175e-13
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
ELIGIBLE_SOURCE_STATUSES = {
    "ELIGIBLE",
    "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES",
}

Condition = Literal["MAX", "CONTROL", "MIN"]

CANDIDATES: dict[str, SimulationDefinition] = {
    "S12F-CANDIDATE-02": SimulationDefinition(
        daughter_rule="FIRST_DAUGHTER",
        overshoot_rule="TRIM_NEW_ENTRANTS_TO_NMAX",
        exposure=ExposureDefinition(
            family="FIXED_COMMON_EXPOSURE", h=0.6031526490073492
        ),
    ),
    "S12F-CANDIDATE-03": SimulationDefinition(
        daughter_rule="RANDOM_NONEMPTY",
        overshoot_rule="TRIM_NEW_ENTRANTS_TO_NMAX",
        exposure=ExposureDefinition(
            family="FIXED_COMMON_EXPOSURE", h=0.5613315384859516
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class ActionSpec:
    action_order: int
    action_id: str
    operation: str
    component_index_zero_based: int


@dataclass(frozen=True, slots=True)
class SimulationOutput:
    trajectory: TimebaseTrajectory
    candidate_rows: tuple[dict[str, Any], ...]
    action_rows: tuple[dict[str, Any], ...]
    boundary_rows: tuple[dict[str, Any], ...]
    source_replay_rows: tuple[dict[str, Any], ...]
    action_schedule: tuple[dict[str, Any], ...]
    runtime: dict[str, float]


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def state_sha256(state: NDArray[np.integer[Any]]) -> str:
    return array_sha256(np.asarray(state, dtype=np.int64))


def states_sha256(states: NDArray[np.integer[Any]]) -> str:
    return array_sha256(np.asarray(states, dtype=np.int64))


def derive_legacy_seed(*identity: object) -> int:
    material = "\x1f".join([VERSION, SOURCE_ROOT_HEX, *map(str, identity)])
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def tie_rank(
    candidate_id: str,
    matrix_index: int,
    condition: str,
    generation: int,
    action_id: str,
) -> str:
    material = "\x1f".join(
        [
            VERSION,
            TIE_ROOT_HEX,
            candidate_id,
            str(matrix_index),
            condition,
            str(generation),
            action_id,
        ]
    )
    return hashlib.sha256(material.encode()).hexdigest()


def stream_seeds(
    *, root_hex: str, phase: str, matrix_index: int
) -> dict[str, Any]:
    """Return streams shared across both candidates and all conditions."""

    return {
        purpose: derive_seed(root_hex, phase, purpose, matrix_index, None)
        for purpose in (
            "catalytic_matrix",
            "initial_state",
            "poisson_update",
            "overshoot_trim",
            "fission",
            "daughter_selection",
        )
    }


def source_seeds(
    candidate_id: str, matrix_index: int, generation: int
) -> tuple[int, int]:
    """Use the same initialization for every action and both extrema."""

    return (
        derive_legacy_seed(
            "APPEND_AND_REFIT_CURRENT_PREFIX",
            candidate_id,
            matrix_index,
            generation,
            "preprocessing",
        ),
        derive_legacy_seed(
            "APPEND_AND_REFIT_CURRENT_PREFIX",
            candidate_id,
            matrix_index,
            generation,
            "fiedler_partition",
        ),
    )


def enumerate_actions(state: NDArray[np.integer[Any]]) -> tuple[ActionSpec, ...]:
    value = np.asarray(state, dtype=np.int64)
    if value.shape != (100,) or np.any(value < 0) or int(value.sum()) <= 0:
        raise ValueError("action state must be a nonempty nonnegative 100-vector")
    rows: list[ActionSpec] = []
    for component in range(100):
        rows.append(ActionSpec(len(rows), f"ADD_{component + 1:03d}", "ADD", component))
    for component in np.flatnonzero(value > 0):
        index = int(component)
        rows.append(
            ActionSpec(len(rows), f"DELETE_{index + 1:03d}", "DELETE", index)
        )
    return tuple(rows)


def apply_action(
    state: NDArray[np.integer[Any]], action: ActionSpec
) -> NDArray[np.int64]:
    output = np.asarray(state, dtype=np.int64).copy()
    if action.operation == "ADD":
        output[action.component_index_zero_based] += 1
    elif action.operation == "DELETE":
        if output[action.component_index_zero_based] <= 0:
            raise ValueError("cannot delete an absent molecular type")
        output[action.component_index_zero_based] -= 1
    else:
        raise ValueError(f"unknown action operation {action.operation!r}")
    if int(output.sum()) <= 0 or np.any(output < 0):
        raise ValueError("action created an invalid state")
    return output


def _finite_endpoint(result: EmergenceAuditResult) -> float | None:
    if result.emergence is None or len(result.emergence) == 0:
        return None
    value = float(result.emergence[-1])
    if result.status not in ELIGIBLE_SOURCE_STATUSES or not np.isfinite(value):
        return None
    return value


def _endpoint(array: NDArray[np.float64] | None) -> float | None:
    if array is None or len(array) == 0:
        return None
    value = float(array[-1])
    return value if np.isfinite(value) else None


def _reduced_condition(result: EmergenceAuditResult) -> float | None:
    if result.partition_average is None or result.partition_average.shape[1] < 2:
        return None
    covariance = np.asarray(np.cov(result.partition_average, ddof=0), dtype=np.float64)
    try:
        value = float(np.linalg.cond(covariance))
    except np.linalg.LinAlgError:
        return None
    return value if np.isfinite(value) else None


def _score_fit(
    *,
    decision_states: NDArray[np.int64],
    candidate_state: NDArray[np.int64],
    preprocessing_seed: int,
    partition_seed: int,
) -> tuple[EmergenceAuditResult, dict[str, Any]]:
    """Fit the literal source pipeline and retain only endpoint diagnostics."""

    fit_states = np.vstack((decision_states, candidate_state))
    clr, masses, closure_errors = frozen_clr(fit_states)
    started = time.perf_counter()
    cpu_started = time.process_time()
    result = run_emergence_pipeline(
        clr,
        "PHIRL_REGULARIZED_SOURCE",
        SAFE_LATTICE,
        preprocessing_seed=preprocessing_seed,
        partition_seed=partition_seed,
    )
    wall = time.perf_counter() - started
    cpu = time.process_time() - cpu_started
    minimum_fiedler = None
    if result.fiedler_vector is not None and len(result.fiedler_vector):
        value = float(np.min(np.abs(result.fiedler_vector)))
        minimum_fiedler = value if np.isfinite(value) else None
    metadata = {
        "sourceStatus": result.status,
        "sourceReason": result.reason,
        "eligible": _finite_endpoint(result) is not None,
        "emergence": _finite_endpoint(result),
        "synergy": _endpoint(result.synergy),
        "downwardCausation": _endpoint(result.downward_causation),
        "localPhiRComparator": _endpoint(result.local_phi_r),
        "retainedVariableCount": len(result.retained_variables),
        "partitionSize1": len(result.partition_1),
        "partitionSize2": len(result.partition_2),
        "minimumAbsoluteFiedlerEntry": minimum_fiedler,
        "reducedCovarianceConditionNumber": _reduced_condition(result),
        "componentIdentityMaxAbsError": result.component_identity_max_abs_error,
        "sourceLocalOffset": result.local_offset,
        "sourceLocalLength": (
            len(result.emergence) if result.emergence is not None else 0
        ),
        "inputObservationCount": len(fit_states),
        "inputTransitionCount": len(fit_states) - 1,
        "inputMassMinimum": float(np.min(masses)),
        "inputMassMaximum": float(np.max(masses)),
        "closureMaxAbsError": float(np.max(closure_errors)),
        "fitStateSha256": states_sha256(fit_states),
        "fitClrSha256": array_sha256(clr),
        "fitWallSeconds": wall,
        "fitCpuSeconds": cpu,
    }
    return result, metadata


def score_action_set(
    *,
    actual_history_states: NDArray[np.int64],
    unedited_daughter: NDArray[np.int64],
    candidate_id: str,
    matrix_index: int,
    condition: Literal["MAX", "MIN"],
    generation: int,
    replay_full_set: bool,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    ActionSpec,
]:
    """Score and select every literal action at one decision."""

    decision_states = np.vstack((actual_history_states, unedited_daughter))
    decision_history_sha = states_sha256(decision_states)
    preprocessing_seed, partition_seed = source_seeds(
        candidate_id, matrix_index, generation
    )
    specs = enumerate_actions(unedited_daughter)
    scored: list[dict[str, Any]] = []
    raw_results: dict[str, tuple[EmergenceAuditResult, NDArray[np.int64]]] = {}
    for spec in specs:
        candidate_state = apply_action(unedited_daughter, spec)
        result, metadata = _score_fit(
            decision_states=decision_states,
            candidate_state=candidate_state,
            preprocessing_seed=preprocessing_seed,
            partition_seed=partition_seed,
        )
        raw_results[spec.action_id] = (result, candidate_state)
        scored.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "condition": condition,
                "generation": generation,
                "actionOrder": spec.action_order,
                "actionId": spec.action_id,
                "operation": spec.operation,
                "componentIndexZeroBased": spec.component_index_zero_based,
                "unitL1Displacement": 1,
                "preActionMass": int(unedited_daughter.sum()),
                "candidateMass": int(candidate_state.sum()),
                "candidateStateSha256": state_sha256(candidate_state),
                "decisionHistorySha256": decision_history_sha,
                "preprocessingSeed": preprocessing_seed,
                "partitionSeed": partition_seed,
                "tieRankSha256": tie_rank(
                    candidate_id,
                    matrix_index,
                    condition,
                    generation,
                    spec.action_id,
                ),
                **metadata,
            }
        )
    eligible = [row for row in scored if row["eligible"]]
    if not eligible:
        raise RuntimeError(
            f"zero eligible actions for {candidate_id} M{matrix_index} "
            f"{condition} generation {generation}"
        )
    if condition == "MAX":
        ordered = sorted(
            eligible, key=lambda row: (-float(row["emergence"]), row["tieRankSha256"])
        )
    else:
        ordered = sorted(
            eligible, key=lambda row: (float(row["emergence"]), row["tieRankSha256"])
        )
    chosen = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else ordered[0]
    selected_score = float(chosen["emergence"])
    exact_ties = [
        row for row in eligible if float(row["emergence"]) == selected_score
    ]
    for row in scored:
        row["selected"] = row["actionId"] == chosen["actionId"]
        row["runnerUp"] = row["actionId"] == runner_up["actionId"]

    no_action_result, no_action = _score_fit(
        decision_states=decision_states,
        candidate_state=np.asarray(unedited_daughter, dtype=np.int64),
        preprocessing_seed=preprocessing_seed,
        partition_seed=partition_seed,
    )
    del no_action_result

    replay_rows: list[dict[str, Any]] = []
    replay_error_values: list[float] = []
    for role, row in (("SELECTED", chosen), ("RUNNER_UP", runner_up)):
        original, candidate_state = raw_results[str(row["actionId"])]
        replay, replay_metadata = _score_fit(
            decision_states=decision_states,
            candidate_state=candidate_state,
            preprocessing_seed=preprocessing_seed,
            partition_seed=partition_seed,
        )
        original_value = _finite_endpoint(original)
        replay_value = _finite_endpoint(replay)
        error = (
            abs(float(original_value) - float(replay_value))
            if original_value is not None and replay_value is not None
            else None
        )
        if error is not None:
            replay_error_values.append(error)
        replay_rows.append(
            {
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "condition": condition,
                "generation": generation,
                "replayScope": role,
                "actionId": row["actionId"],
                "exactResultReplay": result_replay_equal(original, replay),
                "endpointAbsError": error,
                "originalFitStateSha256": row["fitStateSha256"],
                "replayFitStateSha256": replay_metadata["fitStateSha256"],
            }
        )

    full_set_exact: bool | None = None
    if replay_full_set:
        full_set_exact = True
        for row, spec in zip(scored, specs, strict=True):
            original, candidate_state = raw_results[spec.action_id]
            replay, replay_metadata = _score_fit(
                decision_states=decision_states,
                candidate_state=candidate_state,
                preprocessing_seed=preprocessing_seed,
                partition_seed=partition_seed,
            )
            exact = result_replay_equal(original, replay)
            full_set_exact = full_set_exact and exact
            replay_rows.append(
                {
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "condition": condition,
                    "generation": generation,
                    "replayScope": "FULL_SET_SENTINEL",
                    "actionId": spec.action_id,
                    "exactResultReplay": exact,
                    "endpointAbsError": (
                        abs(float(row["emergence"]) - float(_finite_endpoint(replay)))
                        if row["emergence"] is not None
                        and _finite_endpoint(replay) is not None
                        else None
                    ),
                    "originalFitStateSha256": row["fitStateSha256"],
                    "replayFitStateSha256": replay_metadata["fitStateSha256"],
                }
            )

    selected_spec = next(item for item in specs if item.action_id == chosen["actionId"])
    same_operation = [
        row
        for row in eligible
        if row["operation"] == chosen["operation"]
        and row["actionId"] != chosen["actionId"]
    ]
    matched_random = (
        min(same_operation, key=lambda row: row["tieRankSha256"])
        if same_operation
        else None
    )
    replay_max = max(replay_error_values, default=0.0)
    uncertainty_scale = max(replay_max, HISTORICAL_REPLAY_ENVELOPE)
    state_scores = np.asarray([float(row["emergence"]) for row in eligible])
    displacement_scores = np.asarray(
        [float(row["emergence"]) for row in eligible if row["operation"] == chosen["operation"]]
    )
    action_row = {
        "researchStepId": RESEARCH_STEP_ID,
        "candidateId": candidate_id,
        "matrixIndex": matrix_index,
        "condition": condition,
        "generation": generation,
        "status": "INTERVENTION_APPLIED",
        "actionId": chosen["actionId"],
        "operation": chosen["operation"],
        "componentIndexZeroBased": chosen["componentIndexZeroBased"],
        "selectedScore": selected_score,
        "runnerUpActionId": runner_up["actionId"],
        "runnerUpScore": float(runner_up["emergence"]),
        "bestRunnerUpGap": abs(selected_score - float(runner_up["emergence"])),
        "exactTieCount": len(exact_ties),
        "exactTie": len(exact_ties) > 1,
        "enumeratedCandidateCount": len(scored),
        "eligibleCandidateCount": len(eligible),
        "ineligibleCandidateCount": len(scored) - len(eligible),
        "preActionMass": int(unedited_daughter.sum()),
        "postActionMass": int(apply_action(unedited_daughter, selected_spec).sum()),
        "noActionDiagnosticScore": no_action["emergence"],
        "selectedMinusNoAction": (
            selected_score - float(no_action["emergence"])
            if no_action["emergence"] is not None
            else None
        ),
        "stateMatchedScoreMean": float(np.mean(state_scores)),
        "stateMatchedScoreMedian": float(np.median(state_scores)),
        "stateMatchedSelectedPercentile": float(np.mean(state_scores <= selected_score)),
        "displacementMatchedCount": len(displacement_scores),
        "displacementMatchedScoreMean": float(np.mean(displacement_scores)),
        "displacementMatchedScoreMedian": float(np.median(displacement_scores)),
        "matchedRandomActionId": (
            matched_random["actionId"] if matched_random is not None else None
        ),
        "matchedRandomActionScore": (
            float(matched_random["emergence"]) if matched_random is not None else None
        ),
        "selectedMinusMatchedRandomScore": (
            selected_score - float(matched_random["emergence"])
            if matched_random is not None
            else None
        ),
        "selectedReplayMaxAbsError": replay_max,
        "runnerUpUncertaintyScale": uncertainty_scale,
        "gapExceedsNumericalReplayError": abs(
            selected_score - float(runner_up["emergence"])
        )
        > replay_max,
        "gapExceedsFrozenHistoricalReplayEnvelope": abs(
            selected_score - float(runner_up["emergence"])
        )
        > HISTORICAL_REPLAY_ENVELOPE,
        "gapExceedsRunnerUpUncertaintyScale": abs(
            selected_score - float(runner_up["emergence"])
        )
        > uncertainty_scale,
        "fullSetSentinelReplayExecuted": replay_full_set,
        "fullSetSentinelReplayExact": full_set_exact,
        "decisionHistorySha256": decision_history_sha,
        "preprocessingSeed": preprocessing_seed,
        "partitionSeed": partition_seed,
    }
    return scored, action_row, replay_rows, selected_spec


def _cosine(left: NDArray[np.integer[Any]], right: NDArray[np.integer[Any]]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        return math.nan
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def simulate_condition(
    *,
    candidate_id: str,
    matrix_index: int,
    condition: Condition,
    root_hex: str = ROOT_HEX,
    phase: str = PHASE,
    beta: NDArray[np.float64] | None = None,
    initial_state: NDArray[np.int64] | None = None,
    frozen_action_schedule: dict[int, str] | None = None,
    full_set_replay_generations: tuple[int, ...] = (1, 50, 100),
) -> SimulationOutput:
    """Run one condition; fixed schedules are used only for trajectory replay."""

    if candidate_id not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate_id}")
    if condition not in {"MAX", "CONTROL", "MIN"}:
        raise ValueError(f"unknown condition {condition}")
    if condition == "CONTROL" and frozen_action_schedule:
        raise ValueError("control cannot accept an action schedule")
    definition = CANDIDATES[candidate_id]
    seeds = stream_seeds(root_hex=root_hex, phase=phase, matrix_index=matrix_index)
    beta_value = generate_beta(seeds["catalytic_matrix"]) if beta is None else np.asarray(beta, dtype=np.float64)
    state = (
        initialize_distinct_state(seeds["initial_state"])
        if initial_state is None
        else np.asarray(initial_state, dtype=np.int64).copy()
    )
    if beta_value.shape != (100, 100):
        raise ValueError("beta must be 100 by 100")
    if state.shape != (100,) or int(state.sum()) != 40:
        raise ValueError("initial state must be a mass-40 100-vector")
    event_rng = generator(seeds["poisson_update"])
    trim_rng = generator(seeds["overshoot_trim"])
    fission_rng = generator(seeds["fission"])
    daughter_rng = generator(seeds["daughter_selection"])
    initial_state_value = state.copy()
    observations = [
        StateObservation(0, "initial_selected_state", 0, 0, 0, 0, tuple(map(int, state)))
    ]
    generations: list[GenerationSummary] = []
    candidate_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    source_replay_rows: list[dict[str, Any]] = []
    action_schedule: list[dict[str, Any]] = []
    batch_step = 0
    completed = 0
    total_nonzero = 0
    total_gross = 0
    terminal_status = "requested_fissions_completed"
    extinction_generation: int | None = None
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    scoring_wall = 0.0
    scoring_cpu = 0.0

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
            largest_pretrim_overshoot = max(
                largest_pretrim_overshoot, pretrim_overshoot
            )
            if pretrim_overshoot > 0:
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

        minimum_exposure = float(min(exposures)) if exposures else float("nan")
        maximum_exposure = float(max(exposures)) if exposures else float("nan")
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
                    maximum_exposure,
                    minimum_exposure,
                )
            )
            terminal_status = "extinct_during_growth"
            extinction_generation = generation_one_based
            break

        pre_fission_state = state.copy()
        pre_mass = int(state.sum())
        child_a, child_b = fission(state, fission_rng)
        selected, selected_name = select_daughter(
            child_a, child_b, definition.daughter_rule, daughter_rng
        )
        unedited = selected.copy()
        selected_action: ActionSpec | None = None
        if condition == "CONTROL":
            action_rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "condition": condition,
                    "generation": generation_one_based,
                    "status": "NO_INTERVENTION_CONTROL",
                    "actionId": None,
                    "operation": None,
                    "componentIndexZeroBased": None,
                    "preActionMass": int(unedited.sum()),
                    "postActionMass": int(unedited.sum()),
                    "enumeratedCandidateCount": 0,
                    "eligibleCandidateCount": 0,
                }
            )
        elif frozen_action_schedule is not None:
            action_id = frozen_action_schedule[generation_one_based]
            lookup = {item.action_id: item for item in enumerate_actions(unedited)}
            if action_id not in lookup:
                raise RuntimeError(
                    f"frozen replay action {action_id} unavailable at generation "
                    f"{generation_one_based}"
                )
            selected_action = lookup[action_id]
            selected = apply_action(unedited, selected_action)
            action_rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "condition": condition,
                    "generation": generation_one_based,
                    "status": "FROZEN_ACTION_SCHEDULE_REPLAY",
                    "actionId": selected_action.action_id,
                    "operation": selected_action.operation,
                    "componentIndexZeroBased": selected_action.component_index_zero_based,
                    "preActionMass": int(unedited.sum()),
                    "postActionMass": int(selected.sum()),
                    "enumeratedCandidateCount": len(lookup),
                    "eligibleCandidateCount": len(lookup),
                }
            )
        else:
            score_wall_started = time.perf_counter()
            score_cpu_started = time.process_time()
            history_states = np.asarray([item.state for item in observations], dtype=np.int64)
            scores, action_row, replay_rows, selected_action = score_action_set(
                actual_history_states=history_states,
                unedited_daughter=unedited,
                candidate_id=candidate_id,
                matrix_index=matrix_index,
                condition=condition,
                generation=generation_one_based,
                replay_full_set=generation_one_based in full_set_replay_generations,
            )
            scoring_wall += time.perf_counter() - score_wall_started
            scoring_cpu += time.process_time() - score_cpu_started
            candidate_rows.extend(scores)
            action_rows.append(action_row)
            source_replay_rows.extend(replay_rows)
            selected = apply_action(unedited, selected_action)
        if selected_action is not None:
            action_schedule.append(
                {
                    "generation": generation_one_based,
                    "actionId": selected_action.action_id,
                    "operation": selected_action.operation,
                    "componentIndexZeroBased": selected_action.component_index_zero_based,
                }
            )
        boundary_rows.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "condition": condition,
                "generation": generation_one_based,
                "selectedDaughter": selected_name,
                "preFissionStateSha256": state_sha256(pre_fission_state),
                "uneditedDaughterStateSha256": state_sha256(unedited),
                "retainedPostActionStateSha256": state_sha256(selected),
                "preFissionMass": pre_mass,
                "uneditedDaughterMass": int(unedited.sum()),
                "retainedPostActionMass": int(selected.sum()),
                "parentDaughterSimilarityBeforeAction": _cosine(
                    pre_fission_state, unedited
                ),
                "parentDaughterSimilarityAfterAction": _cosine(
                    pre_fission_state, selected
                ),
                "maximumExposure": maximum_exposure,
                "minimumExposure": minimum_exposure,
                "updateCount": local_step,
            }
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
                maximum_exposure,
                minimum_exposure,
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
            terminal_status = "intervention_created_empty_state"
            extinction_generation = generation_one_based
            break

    trajectory_id = (
        f"E01-S17-{phase.upper()}-{candidate_id}-M{matrix_index:03d}-{condition}"
    )
    digest = _trajectory_digest(
        trajectory_id, definition, observations, generations, terminal_status
    )
    trajectory = TimebaseTrajectory(
        trajectory_id=trajectory_id,
        phase=phase,
        matrix_index=matrix_index,
        configuration_id=f"{candidate_id}::{condition}",
        definition=definition,
        beta_sha256=array_sha256(beta_value),
        initial_state_sha256=array_sha256(initial_state_value),
        observations=tuple(observations),
        generations=tuple(generations),
        completed_fissions=completed,
        total_batch_updates=batch_step,
        total_nonzero_reaction_types=total_nonzero,
        total_gross_sampled_events=total_gross,
        terminal_status=terminal_status,
        extinction_generation=extinction_generation,
        trajectory_sha256=digest,
    )
    return SimulationOutput(
        trajectory=trajectory,
        candidate_rows=tuple(candidate_rows),
        action_rows=tuple(action_rows),
        boundary_rows=tuple(boundary_rows),
        source_replay_rows=tuple(source_replay_rows),
        action_schedule=tuple(action_schedule),
        runtime={
            "wallSeconds": time.perf_counter() - wall_started,
            "cpuSeconds": time.process_time() - cpu_started,
            "scoringWallSeconds": scoring_wall,
            "scoringCpuSeconds": scoring_cpu,
        },
    )


def trajectory_replay_equal(left: TimebaseTrajectory, right: TimebaseTrajectory) -> bool:
    return bool(left == right and left.trajectory_sha256 == right.trajectory_sha256)


def action_schedule_map(output: SimulationOutput) -> dict[int, str]:
    return {
        int(row["generation"]): str(row["actionId"])
        for row in output.action_schedule
    }


def exact_trajectory_replay(
    output: SimulationOutput,
    *,
    root_hex: str,
    phase: str,
) -> tuple[SimulationOutput, bool]:
    trajectory = output.trajectory
    candidate_id, condition = trajectory.configuration_id.split("::", maxsplit=1)
    schedule = action_schedule_map(output) if condition != "CONTROL" else None
    replay = simulate_condition(
        candidate_id=candidate_id,
        matrix_index=trajectory.matrix_index,
        condition=condition,  # type: ignore[arg-type]
        root_hex=root_hex,
        phase=phase,
        frozen_action_schedule=schedule,
        full_set_replay_generations=(),
    )
    return replay, trajectory_replay_equal(trajectory, replay.trajectory)


def trajectory_payload_hash(trajectory: TimebaseTrajectory) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "trajectory": asdict(trajectory),
                "trajectorySha256": trajectory.trajectory_sha256,
            }
        ).encode()
    )


def first_state_divergence(
    left: TimebaseTrajectory, right: TimebaseTrajectory
) -> dict[str, Any]:
    common = min(len(left.observations), len(right.observations))
    for index in range(common):
        a, b = left.observations[index], right.observations[index]
        if a != b:
            return {
                "diverged": True,
                "firstDivergenceObservationIndex": index,
                "leftObservationKind": a.observation_kind,
                "rightObservationKind": b.observation_kind,
                "leftGeneration": a.growth_generation_one_based,
                "rightGeneration": b.growth_generation_one_based,
                "leftStateSha256": state_sha256(np.asarray(a.state, dtype=np.int64)),
                "rightStateSha256": state_sha256(np.asarray(b.state, dtype=np.int64)),
                "commonObservationCountBeforeDivergence": index,
            }
    if len(left.observations) != len(right.observations):
        return {
            "diverged": True,
            "firstDivergenceObservationIndex": common,
            "leftObservationKind": None,
            "rightObservationKind": None,
            "leftGeneration": None,
            "rightGeneration": None,
            "leftStateSha256": None,
            "rightStateSha256": None,
            "commonObservationCountBeforeDivergence": common,
        }
    return {
        "diverged": False,
        "firstDivergenceObservationIndex": None,
        "leftObservationKind": None,
        "rightObservationKind": None,
        "leftGeneration": None,
        "rightGeneration": None,
        "leftStateSha256": None,
        "rightStateSha256": None,
        "commonObservationCountBeforeDivergence": common,
    }


def label_h900(trajectory: TimebaseTrajectory) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Exact frozen molecular adjacent-incoming Y=I(H>0.9)."""

    states = np.asarray([row.state for row in trajectory.observations], dtype=np.float64)
    masses = states.sum(axis=1)
    if np.any(masses <= 0):
        raise ValueError("label substrate contains an empty state")
    compositions = states / masses[:, None]
    normalized = compositions / np.linalg.norm(compositions, axis=1)[:, None]
    adjacent = np.sum(normalized[:-1] * normalized[1:], axis=1)
    h = np.concatenate(([adjacent[0]], adjacent))
    return h.astype(np.float64, copy=False), h > 0.9


def _episodes(labels: NDArray[np.bool_]) -> tuple[NDArray[np.int64], int, int]:
    padded = np.concatenate(([False], labels, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    durations = (ends - starts).astype(np.int64)
    return durations, len(starts), len(ends)


def trajectory_outcomes(trajectory: TimebaseTrajectory) -> dict[str, Any]:
    h, labels = label_h900(trajectory)
    indices = np.flatnonzero(labels)
    consistency = None
    if len(labels) >= 3 and np.unique(labels).size == 2:
        value = float(np.corrcoef(labels[:-1].astype(float), labels[1:].astype(float))[0, 1])
        consistency = value if np.isfinite(value) else None
    durations, entries, exits = _episodes(labels)
    first = int(indices[0]) if len(indices) else None
    return {
        "selectedObservationCount": len(labels),
        "persistence": int(labels.sum()),
        "probability": float(labels.mean()),
        "consistency": consistency,
        "timeToFirstReplicator": first,
        "timeToFirstNormalized": (
            float(first / len(labels)) if first is not None and len(labels) else None
        ),
        "longestReplicatingEpisode": (
            int(durations.max()) if len(durations) else 0
        ),
        "meanReplicatingEpisodeDuration": (
            float(durations.mean()) if len(durations) else 0.0
        ),
        "entryCount": int(entries),
        "exitCount": int(exits),
        "hMinimum": float(h.min()),
        "hMedian": float(np.median(h)),
        "hMaximum": float(h.max()),
        "exactLabelIdentityMismatchCount": int(np.count_nonzero(labels != (h > 0.9))),
    }
