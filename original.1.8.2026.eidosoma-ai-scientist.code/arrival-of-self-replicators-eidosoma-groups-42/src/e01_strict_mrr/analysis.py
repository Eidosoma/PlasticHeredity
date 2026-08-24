"""Baseline labels, expanding estimates, and source cross-checks for S12."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from e01_gard_reproducibility import (
    CouplingPolicy,
    SeedBundle,
    SeedRequest,
    StreamPurpose,
    derive_seed_bundle,
    isolated_stream_namespace,
)
from e01_information_dynamics.backends import (
    InformationBackendError,
    compare_decompositions,
    run_omegaid,
    run_phyid,
)
from e01_replicator_labels import (
    ClusterConfiguration,
    cluster_labels,
    historical_technique1_labels,
)

from .core import (
    ENGINE_ID,
    NUMERIC_TOLERANCE,
    PREPROCESSING_IDS,
    REDUNDANCY_IDS,
    ROOT_SEED_HEX,
    BaselineTrajectory,
    PartitionLock,
    PreprocessingResult,
    RunningStrictEstimator,
    StrictEstimate,
    expanding_estimates,
    find_past_only_partition_lock,
    mapped_part_series,
    preprocess_states,
)

ANALYSIS_SPECIFICATION_ID = "E01-S12-ANALYSIS-SEEDS-v1.0.0"


@dataclass(frozen=True, slots=True)
class BaselineAnalysis:
    trajectory: BaselineTrajectory
    preprocessing: PreprocessingResult
    locks: dict[str, PartitionLock]
    estimates: dict[str, tuple[StrictEstimate, ...]]
    label_rows: tuple[dict[str, Any], ...]
    historical_labels: dict[int, bool | None]
    online_labels: dict[int, bool | None]
    whole_rows: tuple[dict[str, Any], ...]
    whole_local_rows: tuple[dict[str, Any], ...]
    numerical_rows: tuple[dict[str, Any], ...]
    analysis_seed_payloads: tuple[dict[str, Any], ...]
    runtime_seconds: float


def analysis_seed_bundle(
    *, trajectory_id: str, replicate_index: int, namespace_tag: str
) -> SeedBundle:
    """Give each analysis operation its own complete S06 identity."""

    analysis_trajectory = f"{trajectory_id}-{namespace_tag}"
    namespace = isolated_stream_namespace(
        experiment_id="E01",
        specification_id=ANALYSIS_SPECIFICATION_ID,
        trajectory_id=analysis_trajectory,
        replicate_index=replicate_index,
    )
    return derive_seed_bundle(
        SeedRequest(
            experiment_id="E01",
            specification_id=ANALYSIS_SPECIFICATION_ID,
            trajectory_id=analysis_trajectory,
            replicate_index=replicate_index,
            engine_id=ENGINE_ID,
            root_seed_hex=ROOT_SEED_HEX,
            coupling_policy=CouplingPolicy.TRAJECTORY_ISOLATED,
            coupling_reason=None,
            stream_namespaces={purpose: namespace for purpose in StreamPurpose},
        )
    )


def _labels(
    trajectory: BaselineTrajectory,
) -> tuple[list[dict[str, Any]], dict[int, bool | None], dict[int, bool | None]]:
    post_indices = [
        index
        for index, kind in enumerate(trajectory.observation_kinds)
        if kind == "post_fission"
    ]
    states = trajectory.states[post_indices]
    observation_ids = tuple(
        f"generation-{index + 1:03d}" for index in range(len(post_indices))
    )
    historical = historical_technique1_labels(
        states,
        trajectory_id=trajectory.trajectory_id,
        observation_ids=observation_ids,
        configuration_id="E01-S08-YH-T1-HGT090-v1.0.0",
        threshold=0.9,
        evidence_class="PINNED_PUBLIC_HISTORICAL_SOURCE_BEHAVIOR",
    )
    online = cluster_labels(
        states,
        trajectory_id=trajectory.trajectory_id,
        observation_ids=observation_ids,
        configuration=ClusterConfiguration(
            configuration_id="E01-S08-YC-COS-HGT090-MIN3-ONLINE-v1.0.0",
            family_id="Y_C",
            family_name="cosine_threshold_graph",
            evidence_class="VALIDATION_ONLY_RECONSTRUCTION_NOT_AUTHOR_DEFAULT",
            metric="cosine",
            representation="raw_nonnegative_vectors",
            threshold=0.9,
            comparator="strict_greater_than",
            minimum_cluster_size=3,
            temporal_scope="past_only_online",
            zero_policy="zero_sum_observation_is_explicitly_ineligible",
        ),
    )
    rows: list[dict[str, Any]] = []
    historical_by_generation: dict[int, bool | None] = {}
    online_by_generation: dict[int, bool | None] = {}
    for family, result, output in (
        ("historical", historical, historical_by_generation),
        ("online_cosine", online, online_by_generation),
    ):
        for generation, record in enumerate(result.rows, start=1):
            payload = record.as_dict()
            payload.update(
                {
                    "matrixIndex": trajectory.matrix_index,
                    "generation": generation,
                    "labelBranch": family,
                }
            )
            rows.append(payload)
            output[generation] = record.is_replicator
    return rows, historical_by_generation, online_by_generation


def _whole_partition(
    coordinates: NDArray[np.float64],
    *,
    preprocessing_id: str,
    trajectory: BaselineTrajectory,
    rng: np.random.Generator,
) -> PartitionLock:
    kinds = ["molecular_event"] * len(trajectory.observation_kinds)
    kinds[-1] = "post_fission"
    return find_past_only_partition_lock(
        coordinates,
        preprocessing_id=preprocessing_id,
        observation_kinds=tuple(kinds),
        generations=trajectory.generations,
        molecular_steps=trajectory.molecular_steps,
        estimator_rng=rng,
    )


def _source_values(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    redundancy: str,
) -> tuple[Any, dict[str, Any], NDArray[np.float64]]:
    short = "MMI" if redundancy.endswith("MMI-v1.0.0") else "CCS"
    try:
        result = run_phyid(
            source,
            target,
            tau=1,
            kind="gaussian",
            redundancy=short,
        )
    except (
        InformationBackendError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        return (
            None,
            {
                "status": "INELIGIBLE",
                "reason": f"PINNED_PHYID_SOURCE_FAILURE::{type(error).__name__}::{error}",
            },
            np.asarray([]),
        )
    if result.status != "ELIGIBLE":
        return (
            result,
            {"status": result.status, "reason": result.reason},
            np.asarray([]),
        )
    means = result.means()
    assert means is not None and result.atoms is not None
    assert result.intermediate_mi is not None
    local_atoms = (
        result.atoms["str"]
        + result.atoms["stx"]
        + result.atoms["sty"]
        + result.atoms["sts"]
        - result.atoms["rtr"]
        - result.atoms["rtx"]
        - result.atoms["rty"]
        - result.atoms["rts"]
    )
    local_direct = (
        result.intermediate_mi["I_xytab"]
        - result.intermediate_mi["I_xtab"]
        - result.intermediate_mi["I_ytab"]
    )
    maximum_local_error = float(np.max(np.abs(local_atoms - local_direct)))
    return (
        result,
        {
            "status": "ELIGIBLE",
            "reason": None,
            "means": means,
            "maximumLocalEquationClosureError": maximum_local_error,
        },
        local_direct,
    )


def analyze_baseline(trajectory: BaselineTrajectory) -> BaselineAnalysis:
    """Analyze one complete trajectory under both representations and redundancies."""

    started = time.perf_counter()
    preprocessing = preprocess_states(trajectory.states)
    locks: dict[str, PartitionLock] = {}
    estimates: dict[str, tuple[StrictEstimate, ...]] = {}
    seed_payloads: list[dict[str, Any]] = []
    for branch_index, preprocessing_id in enumerate(PREPROCESSING_IDS):
        bundle = analysis_seed_bundle(
            trajectory_id=trajectory.trajectory_id,
            replicate_index=trajectory.matrix_index,
            namespace_tag=f"partition-{branch_index}",
        )
        seed_payloads.append(bundle.to_payload())
        rng = bundle.fresh_generators()[StreamPurpose.ESTIMATOR]
        lock = find_past_only_partition_lock(
            preprocessing.coordinates[preprocessing_id],
            preprocessing_id=preprocessing_id,
            observation_kinds=trajectory.observation_kinds,
            generations=trajectory.generations,
            molecular_steps=trajectory.molecular_steps,
            estimator_rng=rng,
        )
        locks[preprocessing_id] = lock
        estimates[preprocessing_id] = tuple(
            expanding_estimates(preprocessing.coordinates[preprocessing_id], lock)
        )

    label_rows, historical, online = _labels(trajectory)
    whole_rows: list[dict[str, Any]] = []
    whole_local_rows: list[dict[str, Any]] = []
    numerical_rows: list[dict[str, Any]] = []
    for branch_index, preprocessing_id in enumerate(PREPROCESSING_IDS):
        bundle = analysis_seed_bundle(
            trajectory_id=trajectory.trajectory_id,
            replicate_index=trajectory.matrix_index,
            namespace_tag=f"whole-partition-{branch_index}",
        )
        seed_payloads.append(bundle.to_payload())
        whole_lock = _whole_partition(
            preprocessing.coordinates[preprocessing_id],
            preprocessing_id=preprocessing_id,
            trajectory=trajectory,
            rng=bundle.fresh_generators()[StreamPurpose.ESTIMATOR],
        )
        if whole_lock.part_a is None:
            for redundancy_id in REDUNDANCY_IDS:
                whole_rows.append(
                    {
                        "trajectoryId": trajectory.trajectory_id,
                        "matrixIndex": trajectory.matrix_index,
                        "preprocessingId": preprocessing_id,
                        "redundancyId": redundancy_id,
                        "scopeLabel": "DESCRIPTIVE_NONPROSPECTIVE",
                        "status": "INELIGIBLE",
                        "reason": whole_lock.reason,
                        "value": None,
                        "partitionId": None,
                        "nEff": trajectory.states.shape[0] - 1,
                        "atomMeansJson": None,
                    }
                )
                for transition_index in range(1, trajectory.states.shape[0]):
                    generation = int(trajectory.generations[transition_index])
                    whole_local_rows.append(
                        {
                            "trajectoryId": trajectory.trajectory_id,
                            "matrixIndex": trajectory.matrix_index,
                            "preprocessingId": preprocessing_id,
                            "redundancyId": redundancy_id,
                            "scopeLabel": "DESCRIPTIVE_NONPROSPECTIVE",
                            "transitionTargetObservationIndex": transition_index,
                            "molecularStep": int(
                                trajectory.molecular_steps[transition_index]
                            ),
                            "generation": generation,
                            "observationKind": trajectory.observation_kinds[
                                transition_index
                            ],
                            "status": "INELIGIBLE",
                            "reason": whole_lock.reason,
                            "value": None,
                            "historicalReplicator": historical.get(generation),
                            "onlineReplicator": online.get(generation),
                        }
                    )
            continue
        source, target = mapped_part_series(
            preprocessing.coordinates[preprocessing_id], whole_lock.part_a
        )
        strict_whole = RunningStrictEstimator.from_series(source, target).estimate()
        for redundancy_id in REDUNDANCY_IDS:
            _source_result, source_summary, local = _source_values(
                source, target, redundancy=redundancy_id
            )
            means = source_summary.get("means")
            whole_rows.append(
                {
                    "trajectoryId": trajectory.trajectory_id,
                    "matrixIndex": trajectory.matrix_index,
                    "preprocessingId": preprocessing_id,
                    "redundancyId": redundancy_id,
                    "scopeLabel": "DESCRIPTIVE_NONPROSPECTIVE",
                    "status": (
                        "ELIGIBLE_NUMERIC_STRICT_WHOLE"
                        if strict_whole.status == "ELIGIBLE_NUMERIC_STRICT_EXPANDING"
                        else "INELIGIBLE"
                    ),
                    "reason": strict_whole.reason,
                    "value": strict_whole.value,
                    "sourceAtomStatus": source_summary["status"],
                    "sourceAtomReason": source_summary["reason"],
                    "partitionId": whole_lock.partition_id,
                    "partAJson": json.dumps(list(whole_lock.part_a)),
                    "partBJson": json.dumps(list(whole_lock.part_b or ())),
                    "nEff": trajectory.states.shape[0] - 1,
                    "conditionNumber": strict_whole.condition_number,
                    "numericalRank": strict_whole.numerical_rank,
                    "latticeClosureError": (
                        means["latticeClosureError"] if means is not None else None
                    ),
                    "paperEquationClosureError": (
                        means["paperEquationClosureError"]
                        if means is not None
                        else None
                    ),
                    "maximumLocalEquationClosureError": source_summary.get(
                        "maximumLocalEquationClosureError"
                    ),
                    "atomMeansJson": (
                        json.dumps(means["atomMeans"], sort_keys=True)
                        if means is not None
                        else None
                    ),
                }
            )
            local_values: list[float | None]
            if local.size == trajectory.states.shape[0] - 1:
                local_values = [float(value) for value in local]
            else:
                local_values = [None] * (trajectory.states.shape[0] - 1)
            for transition_index, value in enumerate(local_values, start=1):
                generation = int(trajectory.generations[transition_index])
                whole_local_rows.append(
                    {
                        "trajectoryId": trajectory.trajectory_id,
                        "matrixIndex": trajectory.matrix_index,
                        "preprocessingId": preprocessing_id,
                        "redundancyId": redundancy_id,
                        "scopeLabel": "DESCRIPTIVE_NONPROSPECTIVE",
                        "transitionTargetObservationIndex": transition_index,
                        "molecularStep": int(
                            trajectory.molecular_steps[transition_index]
                        ),
                        "generation": generation,
                        "observationKind": trajectory.observation_kinds[
                            transition_index
                        ],
                        "status": (
                            "ELIGIBLE"
                            if value is not None
                            else source_summary["status"]
                        ),
                        "reason": (
                            None if value is not None else source_summary["reason"]
                        ),
                        "value": value,
                        "historicalReplicator": historical.get(generation),
                        "onlineReplicator": online.get(generation),
                    }
                )

        if trajectory.matrix_index not in (0, 5, 11):
            continue
        prospective_lock = locks[preprocessing_id]
        prospective_estimates = estimates[preprocessing_id]
        if prospective_lock.part_a is None:
            continue
        prospective_source, prospective_target = mapped_part_series(
            preprocessing.coordinates[preprocessing_id], prospective_lock.part_a
        )
        eligible = [
            index
            for index, estimate in enumerate(prospective_estimates)
            if estimate.status == "ELIGIBLE_NUMERIC_STRICT_EXPANDING"
        ]
        if not eligible:
            continue
        checkpoint_indices = sorted(
            {eligible[0], eligible[len(eligible) // 2], eligible[-1]}
        )
        checkpoint_names = {
            eligible[0]: "first_eligible",
            eligible[len(eligible) // 2]: "middle_eligible",
            eligible[-1]: "final_eligible",
        }
        for checkpoint in checkpoint_indices:
            running = prospective_estimates[checkpoint]
            for redundancy_id in REDUNDANCY_IDS:
                short = "MMI" if redundancy_id.endswith("MMI-v1.0.0") else "CCS"
                try:
                    phyid = run_phyid(
                        prospective_source[: checkpoint + 1],
                        prospective_target[: checkpoint + 1],
                        tau=1,
                        kind="gaussian",
                        redundancy=short,
                    )
                    omega_cpu = run_omegaid(
                        prospective_source[: checkpoint + 1],
                        prospective_target[: checkpoint + 1],
                        tau=1,
                        kind="gaussian",
                        redundancy=short,
                        backend_name="numpy",
                    )
                    omega_gpu = run_omegaid(
                        prospective_source[: checkpoint + 1],
                        prospective_target[: checkpoint + 1],
                        tau=1,
                        kind="gaussian",
                        redundancy=short,
                        backend_name="cupy",
                    )
                except (
                    InformationBackendError,
                    FloatingPointError,
                    np.linalg.LinAlgError,
                ) as error:
                    numerical_rows.append(
                        {
                            "trajectoryId": trajectory.trajectory_id,
                            "matrixIndex": trajectory.matrix_index,
                            "preprocessingId": preprocessing_id,
                            "redundancyId": redundancy_id,
                            "checkpoint": checkpoint_names[checkpoint],
                            "observationIndex": checkpoint,
                            "nEff": checkpoint,
                            "runningValue": running.value,
                            "phyidValue": None,
                            "runningVsPhyidAbsoluteError": None,
                            "phyidVsOmegaCpuSuccess": False,
                            "phyidVsOmegaCpuMaximumAbsoluteError": None,
                            "omegaCpuVsGpuSuccess": False,
                            "omegaCpuVsGpuMaximumAbsoluteError": None,
                            "status": "FAIL",
                            "reason": f"SOURCE_CROSSCHECK_FAILURE::{type(error).__name__}::{error}",
                        }
                    )
                    continue
                phyid_means = phyid.means()
                running_error = (
                    abs(
                        float(running.value)
                        - float(phyid_means["paperEquationAggregateDirect"])
                    )
                    if running.value is not None and phyid_means is not None
                    else None
                )
                phyid_cpu = compare_decompositions(
                    phyid,
                    omega_cpu,
                    absolute_tolerance=NUMERIC_TOLERANCE,
                    relative_tolerance=NUMERIC_TOLERANCE,
                )
                cpu_gpu = compare_decompositions(
                    omega_cpu,
                    omega_gpu,
                    absolute_tolerance=NUMERIC_TOLERANCE,
                    relative_tolerance=NUMERIC_TOLERANCE,
                )
                numerical_rows.append(
                    {
                        "trajectoryId": trajectory.trajectory_id,
                        "matrixIndex": trajectory.matrix_index,
                        "preprocessingId": preprocessing_id,
                        "redundancyId": redundancy_id,
                        "checkpoint": checkpoint_names[checkpoint],
                        "observationIndex": checkpoint,
                        "nEff": checkpoint,
                        "runningValue": running.value,
                        "phyidValue": (
                            phyid_means["paperEquationAggregateDirect"]
                            if phyid_means is not None
                            else None
                        ),
                        "runningVsPhyidAbsoluteError": running_error,
                        "phyidVsOmegaCpuSuccess": phyid_cpu.get("success"),
                        "phyidVsOmegaCpuMaximumAbsoluteError": phyid_cpu.get(
                            "maximumAbsoluteError"
                        ),
                        "omegaCpuVsGpuSuccess": cpu_gpu.get("success"),
                        "omegaCpuVsGpuMaximumAbsoluteError": cpu_gpu.get(
                            "maximumAbsoluteError"
                        ),
                        "status": (
                            "PASS"
                            if running_error is not None
                            and running_error <= NUMERIC_TOLERANCE
                            and phyid_cpu.get("success")
                            and cpu_gpu.get("success")
                            else "FAIL"
                        ),
                    }
                )

    return BaselineAnalysis(
        trajectory=trajectory,
        preprocessing=preprocessing,
        locks=locks,
        estimates=estimates,
        label_rows=tuple(label_rows),
        historical_labels=historical,
        online_labels=online,
        whole_rows=tuple(whole_rows),
        whole_local_rows=tuple(whole_local_rows),
        numerical_rows=tuple(numerical_rows),
        analysis_seed_payloads=tuple(seed_payloads),
        runtime_seconds=time.perf_counter() - started,
    )
