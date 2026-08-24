#!/usr/bin/env python3
"""Run the four-triplet adaptive S13X retrospective directional pilot."""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import hashlib
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e01_creative_directional_search.core import (
    RESEARCH_STEP_ID,
    association_summary,
    label_specs,
    label_trajectory,
)
from e01_creative_directional_search.core import (
    derive_seed as derive_analysis_seed,
)
from e01_creative_directional_search.intervention import (
    FrozenPhiRLScorer,
    build_frozen_phirl_scorer,
    source_replay_max_abs,
)
from e01_frozen_timebase_ensemble.core import (
    frozen_clr,
    selected_clock_observations,
    states_from_observations,
)
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
    result_replay_equal,
    run_emergence_pipeline,
)

VERSION = "E01-S13X-CREATIVE-DIRECTIONAL-REPLICATION-SEARCH-v1.0.0"
ROOT_HEX = "402d1d7dcd40d4613499b420424ef437c408aedd39e98fd13c1afd100d25a3f4"
PHASE = "S13X_INTERVENTION_PILOT"
CANDIDATES = {
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
MATRIX_INDICES = (0, 1)
CONDITIONS = ("MAX", "CONTROL", "MIN")
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
STEP_ROOT = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "research_steps/S13X"
CACHE_ROOT = Path("/cache/e01_s13x_v1/intervention_pilot")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(frame.columns).encode())
    digest.update("\x1f".join(map(str, frame.dtypes)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def _seed_set(candidate_id: str, matrix_index: int) -> dict[str, Any]:
    return {
        purpose: derive_seed(
            ROOT_HEX,
            PHASE,
            purpose,
            matrix_index,
            None if purpose in {"catalytic_matrix", "initial_state"} else candidate_id,
        )
        for purpose in (
            "catalytic_matrix",
            "initial_state",
            "poisson_update",
            "overshoot_trim",
            "fission",
            "daughter_selection",
        )
    }


def _action_frame_hash(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return hashlib.sha256(b"EMPTY_ACTION_FRAME").hexdigest()
    frame = pd.DataFrame(rows)
    sort_columns = [
        name
        for name in ("generation", "actionOrder", "actionId")
        if name in frame.columns
    ]
    return frame_hash(frame.sort_values(sort_columns, kind="stable"))


def _replay_equal(left: TimebaseTrajectory, right: TimebaseTrajectory) -> bool:
    return bool(
        left.trajectory_sha256 == right.trajectory_sha256
        and left.observations == right.observations
        and json.dumps([asdict(row) for row in left.generations], sort_keys=True)
        == json.dumps([asdict(row) for row in right.generations], sort_keys=True)
    )


def simulate_condition(
    *,
    candidate_id: str,
    matrix_index: int,
    condition: str,
    definition: SimulationDefinition,
    beta: np.ndarray,
    initial_state: np.ndarray,
    scorer: FrozenPhiRLScorer | None,
) -> tuple[TimebaseTrajectory, list[dict[str, Any]], list[dict[str, Any]]]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    if condition != "CONTROL" and scorer is None:
        raise ValueError("treated condition requires a frozen scorer")
    seeds = _seed_set(candidate_id, matrix_index)
    event_rng = generator(seeds["poisson_update"])
    trim_rng = generator(seeds["overshoot_trim"])
    fission_rng = generator(seeds["fission"])
    daughter_rng = generator(seeds["daughter_selection"])
    state = np.asarray(initial_state, dtype=np.int64).copy()
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
    candidate_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
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
                state, beta, definition, event_rng, trim_rng
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

        pre_mass = int(state.sum())
        child_a, child_b = fission(state, fission_rng)
        selected, selected_name = select_daughter(
            child_a, child_b, definition.daughter_rule, daughter_rng
        )
        before_action = selected.copy()
        if condition == "CONTROL":
            action_rows.append(
                {
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "condition": condition,
                    "generation": generation_one_based,
                    "status": "NO_INTERVENTION_CONTROL",
                    "actionId": None,
                    "operation": None,
                    "componentIndexZeroBased": None,
                    "selectedScore": None,
                    "runnerUpScore": None,
                    "bestRunnerUpGap": None,
                    "candidateCount": 0,
                    "preActionMass": int(before_action.sum()),
                    "postActionMass": int(before_action.sum()),
                }
            )
        else:
            assert scorer is not None
            scores = scorer.score_count_actions(before_action)
            ordered = sorted(
                scores,
                key=(
                    (lambda row: (-float(row["emergence"]), int(row["actionOrder"])))
                    if condition == "MAX"
                    else (
                        lambda row: (float(row["emergence"]), int(row["actionOrder"]))
                    )
                ),
            )
            chosen, runner_up = ordered[:2]
            component = int(chosen["componentIndexZeroBased"])
            if chosen["operation"] == "ADD":
                selected[component] += 1
            else:
                selected[component] -= 1
            for row in scores:
                candidate_rows.append(
                    {
                        "candidateId": candidate_id,
                        "matrixIndex": matrix_index,
                        "condition": condition,
                        "generation": generation_one_based,
                        **row,
                        "selected": row["actionId"] == chosen["actionId"],
                    }
                )
            action_rows.append(
                {
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "condition": condition,
                    "generation": generation_one_based,
                    "status": "INTERVENTION_APPLIED",
                    "actionId": chosen["actionId"],
                    "operation": chosen["operation"],
                    "componentIndexZeroBased": component,
                    "selectedScore": float(chosen["emergence"]),
                    "runnerUpScore": float(runner_up["emergence"]),
                    "bestRunnerUpGap": abs(
                        float(chosen["emergence"]) - float(runner_up["emergence"])
                    ),
                    "candidateCount": len(scores),
                    "preActionMass": int(before_action.sum()),
                    "postActionMass": int(selected.sum()),
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

    trajectory_id = f"E01-S13X-PILOT-{candidate_id}-M{matrix_index:03d}-{condition}"
    digest = _trajectory_digest(
        trajectory_id, definition, observations, generations, terminal_status
    )
    trajectory = TimebaseTrajectory(
        trajectory_id=trajectory_id,
        phase=PHASE,
        matrix_index=matrix_index,
        configuration_id=f"{candidate_id}::{condition}",
        definition=definition,
        beta_sha256=array_sha256(beta),
        initial_state_sha256=array_sha256(initial_state),
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
    return trajectory, candidate_rows, action_rows


def _label_spec(label_id: str) -> Any:
    return next(item for item in label_specs() if item.label_id == label_id)


def _source_summary(
    trajectory: TimebaseTrajectory, candidate_id: str, matrix_index: int, condition: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = selected_clock_observations(trajectory, "C1_SELECTED_DAUGHTER_RETAINED")
    clr, _, _ = frozen_clr(states_from_observations(selected))
    preprocessing_seed = derive_analysis_seed(
        "pilot_source", candidate_id, matrix_index, condition, "preprocessing"
    )
    partition_seed = derive_analysis_seed(
        "pilot_source", candidate_id, matrix_index, condition, "partition"
    )
    result = run_emergence_pipeline(
        clr,
        "PHIRL_REGULARIZED_SOURCE",
        SAFE_LATTICE,
        preprocessing_seed=preprocessing_seed,
        partition_seed=partition_seed,
    )
    replay = run_emergence_pipeline(
        clr,
        "PHIRL_REGULARIZED_SOURCE",
        SAFE_LATTICE,
        preprocessing_seed=preprocessing_seed,
        partition_seed=partition_seed,
    )
    replay_passed = result_replay_equal(result, replay)
    rows = []
    for label_id in (
        "MOL_ADJACENT_INCOMING_H950",
        "MOL_ADJACENT_INCOMING_H970",
        "MOL_ADJACENT_AVERAGE_H970",
    ):
        label_frame, fingerprint = label_trajectory(trajectory, _label_spec(label_id))
        labels = label_frame.sort_values("selectedSequenceIndex")[
            "isReplicator"
        ].to_numpy(dtype=float)
        values = np.asarray(result.emergence, dtype=float)
        aligned_labels = labels[result.local_offset : result.local_offset + len(values)]
        association = association_summary(values, aligned_labels)
        rows.append(
            {
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "condition": condition,
                "trajectoryId": trajectory.trajectory_id,
                "labelId": label_id,
                "completedFissions": trajectory.completed_fissions,
                "terminalStatus": trajectory.terminal_status,
                "persistence": fingerprint["persistence"],
                "probability": fingerprint["probability"],
                "consistency": fingerprint["consistency"],
                "timeToFirst": fingerprint["timeToFirst"],
                "meanEmergence": float(np.mean(values)),
                **association,
                "sourceStatus": result.status,
                "sourceReplayPassed": replay_passed,
            }
        )
    return rows, {
        "status": result.status,
        "sourceReplayPassed": replay_passed,
        "preprocessingSeed": preprocessing_seed,
        "partitionSeed": partition_seed,
    }


def run_unit(candidate_id: str, matrix_index: int) -> dict[str, Any]:
    started = time.perf_counter()
    definition = CANDIDATES[candidate_id]
    seeds = _seed_set(candidate_id, matrix_index)
    beta = generate_beta(seeds["catalytic_matrix"])
    initial_state = initialize_distinct_state(seeds["initial_state"])
    control, control_scores, control_actions = simulate_condition(
        candidate_id=candidate_id,
        matrix_index=matrix_index,
        condition="CONTROL",
        definition=definition,
        beta=beta,
        initial_state=initial_state,
        scorer=None,
    )
    selected = selected_clock_observations(control, "C1_SELECTED_DAUGHTER_RETAINED")
    clr, _, _ = frozen_clr(states_from_observations(selected))
    pre_seed = derive_analysis_seed(
        "pilot_reference", candidate_id, matrix_index, "preprocessing"
    )
    part_seed = derive_analysis_seed(
        "pilot_reference", candidate_id, matrix_index, "partition"
    )
    source = run_emergence_pipeline(
        clr,
        "PHIRL_REGULARIZED_SOURCE",
        SAFE_LATTICE,
        preprocessing_seed=pre_seed,
        partition_seed=part_seed,
    )
    source_replay = run_emergence_pipeline(
        clr,
        "PHIRL_REGULARIZED_SOURCE",
        SAFE_LATTICE,
        preprocessing_seed=pre_seed,
        partition_seed=part_seed,
    )
    scorer = build_frozen_phirl_scorer(clr, source, SAFE_LATTICE)
    scorer_replay_error = source_replay_max_abs(scorer, source)

    trajectories = {"CONTROL": control}
    candidate_rows = list(control_scores)
    action_rows = list(control_actions)
    replay_rows = []
    for condition in ("MAX", "MIN"):
        trajectory, scores, actions = simulate_condition(
            candidate_id=candidate_id,
            matrix_index=matrix_index,
            condition=condition,
            definition=definition,
            beta=beta,
            initial_state=initial_state,
            scorer=scorer,
        )
        replay, replay_scores, replay_actions = simulate_condition(
            candidate_id=candidate_id,
            matrix_index=matrix_index,
            condition=condition,
            definition=definition,
            beta=beta,
            initial_state=initial_state,
            scorer=scorer,
        )
        trajectories[condition] = trajectory
        candidate_rows.extend(scores)
        action_rows.extend(actions)
        replay_rows.append(
            {
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "condition": condition,
                "trajectoryExact": _replay_equal(trajectory, replay),
                "candidateScoresExact": _action_frame_hash(scores)
                == _action_frame_hash(replay_scores),
                "actionsExact": _action_frame_hash(actions)
                == _action_frame_hash(replay_actions),
            }
        )
    control_replay, _, control_replay_actions = simulate_condition(
        candidate_id=candidate_id,
        matrix_index=matrix_index,
        condition="CONTROL",
        definition=definition,
        beta=beta,
        initial_state=initial_state,
        scorer=None,
    )
    replay_rows.append(
        {
            "candidateId": candidate_id,
            "matrixIndex": matrix_index,
            "condition": "CONTROL",
            "trajectoryExact": _replay_equal(control, control_replay),
            "candidateScoresExact": True,
            "actionsExact": _action_frame_hash(control_actions)
            == _action_frame_hash(control_replay_actions),
        }
    )

    source_rows = []
    source_statuses = []
    cache_rows = []
    for condition, trajectory in trajectories.items():
        rows, status = _source_summary(
            trajectory, candidate_id, matrix_index, condition
        )
        source_rows.extend(rows)
        source_statuses.append({"condition": condition, **status})
        cache_path = (
            CACHE_ROOT / candidate_id / f"M{matrix_index:03d}-{condition}.pickle"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(trajectory, handle, protocol=5)
        cache_rows.append(
            {
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "condition": condition,
                "trajectoryId": trajectory.trajectory_id,
                "trajectorySha256": trajectory.trajectory_sha256,
                "cachePath": str(cache_path),
                "cacheSha256": sha256_file(cache_path),
                "completedFissions": trajectory.completed_fissions,
                "terminalStatus": trajectory.terminal_status,
                "totalBatchUpdates": trajectory.total_batch_updates,
                "betaSha256": trajectory.beta_sha256,
                "initialStateSha256": trajectory.initial_state_sha256,
            }
        )
    seed_rows = []
    for condition in CONDITIONS:
        for purpose, identity in seeds.items():
            seed_rows.append(
                {
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "condition": condition,
                    "purpose": purpose,
                    "derivedSeed": str(identity.derived_seed),
                    "seedMaterialSha256": identity.seed_material_sha256,
                    "rootHex": identity.root_sha256,
                    "sharedWithinTriplet": True,
                }
            )
    return {
        "candidateRows": candidate_rows,
        "actionRows": action_rows,
        "sourceRows": source_rows,
        "sourceStatuses": source_statuses,
        "replayRows": replay_rows,
        "cacheRows": cache_rows,
        "seedRows": seed_rows,
        "reference": {
            "candidateId": candidate_id,
            "matrixIndex": matrix_index,
            "sourceStatus": source.status,
            "sourceExactReplayPassed": result_replay_equal(source, source_replay),
            "preprocessingMaxAbsError": scorer.preprocessing_max_abs_error,
            "sourcePointReplayMaxAbsError": scorer_replay_error,
            "retainedVariableCount": len(scorer.retained_variables),
            "partitionSize1": len(scorer.partition_1_local),
            "partitionSize2": len(scorer.partition_2_local),
        },
        "runtimeSeconds": time.perf_counter() - started,
    }


def directional_results(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, matrix, label_id), group in source.groupby(
        ["candidateId", "matrixIndex", "labelId"], sort=True
    ):
        by_condition = group.set_index("condition")
        for metric in ("persistence", "probability", "consistency", "timeToFirst"):
            maximum = by_condition.loc["MAX", metric]
            control = by_condition.loc["CONTROL", metric]
            minimum = by_condition.loc["MIN", metric]
            favorable_order = (
                maximum <= control <= minimum
                if metric == "timeToFirst"
                else maximum >= control >= minimum
            )
            rows.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": matrix,
                    "labelId": label_id,
                    "outcome": metric,
                    "max": maximum,
                    "control": control,
                    "min": minimum,
                    "maxMinusControl": maximum - control,
                    "controlMinusMin": control - minimum,
                    "paperDirectedOrdering": bool(favorable_order),
                }
            )
    return pd.DataFrame(rows)


def append_ledger(results: pd.DataFrame) -> None:
    path = STEP_ROOT / "chronological_search_ledger.csv"
    existing = pd.read_csv(path)
    start = int(existing["attemptSequence"].max()) + 1
    rows = []
    primary = results[
        (results["labelId"] == "MOL_ADJACENT_INCOMING_H970")
        & (results["outcome"].isin(["persistence", "probability"]))
    ]
    for offset, row in enumerate(primary.itertuples(index=False)):
        rows.append(
            {
                "attemptSequence": start + offset,
                "attemptId": f"S13X-PILOT-{row.candidateId}-M{int(row.matrixIndex):03d}-{row.outcome}",
                "phase": "ADAPTIVE_DIRECTIONAL_INTERVENTION_PILOT",
                "choiceFamily": "RETROSPECTIVE_FROZEN_CONTROL_SCORING",
                "specification": json.dumps(
                    {
                        "candidateId": row.candidateId,
                        "matrixIndex": int(row.matrixIndex),
                        "outcome": row.outcome,
                        "labelId": row.labelId,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "evidenceTier": "SPECULATIVE_RETROSPECTIVE_FORENSIC_INTERVENTION",
                "outcome": (
                    f"max={row.max}; control={row.control}; min={row.min}; "
                    f"paperDirectedOrdering={row.paperDirectedOrdering}"
                ),
                "negativeResult": not bool(row.paperDirectedOrdering),
                "selectionUse": "Small adaptive directional pilot only; no causal or author claim.",
            }
        )
    pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).to_csv(
        path, index=False, lineterminator="\n"
    )


def main() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    units = [
        (candidate, matrix) for candidate in CANDIDATES for matrix in MATRIX_INDICES
    ]
    outputs = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_unit, candidate, matrix): (candidate, matrix)
            for candidate, matrix in units
        }
        for future in as_completed(futures):
            candidate, matrix = futures[future]
            outputs.append(future.result())
            print(
                json.dumps(
                    {
                        "stage": "intervention_pilot_unit_complete",
                        "candidateId": candidate,
                        "matrixIndex": matrix,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    candidate_scores = pd.DataFrame(
        [row for output in outputs for row in output["candidateRows"]]
    ).sort_values(
        ["candidateId", "matrixIndex", "condition", "generation", "actionOrder"]
    )
    actions = pd.DataFrame(
        [row for output in outputs for row in output["actionRows"]]
    ).sort_values(["candidateId", "matrixIndex", "condition", "generation"])
    source = pd.DataFrame(
        [row for output in outputs for row in output["sourceRows"]]
    ).sort_values(["candidateId", "matrixIndex", "condition", "labelId"])
    replay = pd.DataFrame(
        [row for output in outputs for row in output["replayRows"]]
    ).sort_values(["candidateId", "matrixIndex", "condition"])
    cache = pd.DataFrame(
        [row for output in outputs for row in output["cacheRows"]]
    ).sort_values(["candidateId", "matrixIndex", "condition"])
    seeds = pd.DataFrame(
        [row for output in outputs for row in output["seedRows"]]
    ).sort_values(["candidateId", "matrixIndex", "condition", "purpose"])
    references = pd.DataFrame([output["reference"] for output in outputs]).sort_values(
        ["candidateId", "matrixIndex"]
    )
    results = directional_results(source)

    candidate_scores.to_parquet(
        STEP_ROOT / "intervention_candidate_scores.parquet",
        index=False,
        compression="zstd",
    )
    actions.to_parquet(
        STEP_ROOT / "intervention_action_log.parquet", index=False, compression="zstd"
    )
    source.to_csv(STEP_ROOT / "intervention_trajectory_summaries.csv", index=False)
    replay.to_csv(STEP_ROOT / "intervention_replay_validation.csv", index=False)
    cache.to_csv(STEP_ROOT / "intervention_trajectory_manifest.csv", index=False)
    seeds.to_csv(STEP_ROOT / "intervention_seed_manifest.csv", index=False)
    references.to_csv(STEP_ROOT / "intervention_reference_validation.csv", index=False)
    results.to_csv(STEP_ROOT / "intervention_directional_results.csv", index=False)
    append_ledger(results)

    primary = results[
        (results["labelId"] == "MOL_ADJACENT_INCOMING_H970")
        & (results["outcome"].isin(["persistence", "probability"]))
    ]
    validation = {
        "schema": "eidosoma.e01.s13x_intervention_pilot_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "evidenceClass": "ADAPTIVE_RETROSPECTIVE_FORENSIC_DIRECTIONAL_PILOT",
        "tripletCount": len(units),
        "trajectoryCount": len(cache),
        "candidateScoreRowCount": len(candidate_scores),
        "actionRowCount": len(actions),
        "allTrajectoriesCompleted100Fissions": bool(
            (cache["completedFissions"] == 100).all()
        ),
        "allTrajectoryAndActionReplayPassed": bool(
            replay[["trajectoryExact", "candidateScoresExact", "actionsExact"]]
            .astype(bool)
            .all()
            .all()
        ),
        "allSourceReplayPassed": bool(source["sourceReplayPassed"].astype(bool).all()),
        "allReferenceSourceReplayPassed": bool(
            references["sourceExactReplayPassed"].astype(bool).all()
        ),
        "maximumFixedScorerSourceReplayError": float(
            references["sourcePointReplayMaxAbsError"].max()
        ),
        "maximumPreprocessingReplayError": float(
            references["preprocessingMaxAbsError"].max()
        ),
        "primaryPaperDirectedOrderingCount": int(
            primary["paperDirectedOrdering"].sum()
        ),
        "primaryDirectionalComparisonCount": len(primary),
        "wallSeconds": time.perf_counter() - started,
        "passed": bool(
            len(cache) == 12
            and (cache["completedFissions"] == 100).all()
            and replay[["trajectoryExact", "candidateScoresExact", "actionsExact"]]
            .astype(bool)
            .all()
            .all()
            and source["sourceReplayPassed"].astype(bool).all()
            and references["sourceExactReplayPassed"].astype(bool).all()
            and references["sourcePointReplayMaxAbsError"].max() <= 1e-10
            and references["preprocessingMaxAbsError"].max() <= 1e-12
        ),
    }
    write_json(STEP_ROOT / "intervention_pilot_validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError("S13X intervention pilot software validation failed")
    print(json.dumps(validation, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
