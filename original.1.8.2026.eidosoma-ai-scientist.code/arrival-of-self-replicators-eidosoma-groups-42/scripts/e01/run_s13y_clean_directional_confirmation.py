#!/usr/bin/env python3
"""Execute the frozen E01 S13Y clean directional confirmation campaign."""

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
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy
import yaml
from scipy.stats import linregress, spearmanr
from statsmodels.stats.diagnostic import acorr_ljungbox

from e01_clean_directional_confirmation.core import (
    CANDIDATE_IDS,
    EVIDENCE_CLASS,
    FULL_MODE_ID,
    HISTORICAL_LABEL_ID,
    IMPLEMENTATION_ID,
    METRIC_ID,
    PREFIX_MODE_ID,
    PRIMARY_LABEL_ID,
    RESAMPLING_REPLICATES,
    RESEARCH_STEP_ID,
    ROOT_SEED_HEX,
    S13X_PIPELINE_ID,
    SENSITIVITY_LABEL_ID,
    SIMULATION_PHASE,
    VERSION,
    candidate_registry,
    classify,
    derive_seed,
    exact_label_identity,
    fixed_label_spec,
    outcome_class,
    prefix_gate,
    primary_association_gate,
    primary_drift_gate,
    seed_material_sha256,
    summarize_resampled_direction,
)
from e01_creative_directional_search.core import association_summary, label_trajectory
from e01_frozen_timebase_ensemble.core import (
    ELIGIBLE_SOURCE_STATUSES,
    frozen_clr,
    frozen_generation_labels,
    post_fission_endpoint_records,
    selected_clock_observations,
    sha256_array,
    states_from_observations,
)
from e01_latent_timebase.core import (
    ExposureDefinition,
    SimulationDefinition,
    array_sha256,
    generate_beta,
    initialize_distinct_state,
    simulate_trajectory,
    trajectory_summary,
)
from e01_latent_timebase.core import derive_seed as derive_simulation_seed
from e01_pigozzi_source_audit.core import SourceImplementation
from e01_replay_repair.comparator import compare_seed_tuples, compare_trajectories
from e01_source_emergence_metric_identity.core import (
    result_replay_equal,
    run_emergence_pipeline,
)

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S13Y"
FIGURE_ROOT = STEP_ROOT / "figures"
CACHE_ROOT = Path("/cache/e01_s13y_v1")
RAW_ROOT = CACHE_ROOT / "raw_trajectories"
SOURCE_ROOT = CACHE_ROOT / "source_results"
CONFIG_PATH = (
    REPO / "configs/e01/s13y_clean_directional_confirmation_preregistration.yaml"
)
SAFE_LATTICE = ARTIFACTS / "research_steps/S12B/safe_phi_lattice.json"
PRIMARY_SPEC = fixed_label_spec(PRIMARY_LABEL_ID)
SENSITIVITY_SPEC = fixed_label_spec(SENSITIVITY_LABEL_ID)
PHIRL = SourceImplementation.PHIRL


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def recursive_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def frame_hash(frame: pd.DataFrame) -> str:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def candidate_definition(candidate_id: str) -> SimulationDefinition:
    row = {item["candidateId"]: item for item in candidate_registry()}[candidate_id]
    return SimulationDefinition(
        daughter_rule=row["daughterRule"],
        overshoot_rule=row["overshootRule"],
        exposure=ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=float(row["h"])),
    )


def simulation_seed_row(seed: Any, candidate_id: str | None) -> dict[str, Any]:
    shared = seed.purpose in {"catalytic_matrix", "initial_state"}
    identity_candidate = "SHARED" if shared else str(candidate_id)
    return {
        "researchStepId": RESEARCH_STEP_ID,
        "streamDomain": "simulation",
        "streamId": f"S13Y::SIM::{seed.purpose}::M{int(seed.matrix_index):03d}::{identity_candidate}",
        "purpose": seed.purpose,
        "candidateId": None if shared else candidate_id,
        "matrixIndex": int(seed.matrix_index),
        "implementationId": None,
        "temporalModeId": None,
        "endpointGeneration": None,
        "derivedSeed": str(seed.derived_seed),
        "seedMaterialSha256": seed.seed_material_sha256,
        "rootHex": seed.root_sha256,
        "bitGenerator": "PCG64DXSM",
        "sharedAcrossCandidates": shared,
    }


def simulate_shared_matrix(matrix_index: int) -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    beta_seed = derive_simulation_seed(
        ROOT_SEED_HEX, SIMULATION_PHASE, "catalytic_matrix", matrix_index
    )
    init_seed = derive_simulation_seed(
        ROOT_SEED_HEX, SIMULATION_PHASE, "initial_state", matrix_index
    )
    beta = generate_beta(beta_seed)
    initial = initialize_distinct_state(init_seed)
    beta_hash = array_sha256(beta)
    initial_hash = array_sha256(initial)
    trajectories: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    seeds = [simulation_seed_row(beta_seed, None), simulation_seed_row(init_seed, None)]
    for candidate_id in CANDIDATE_IDS:
        kwargs = {
            "phase": SIMULATION_PHASE,
            "root_hex": ROOT_SEED_HEX,
            "matrix_index": matrix_index,
            "definition": candidate_definition(candidate_id),
            "stream_identity": candidate_id,
            "beta": beta,
            "initial_state": initial,
        }
        primary, primary_seeds = simulate_trajectory(**kwargs)
        replay, replay_seeds = simulate_trajectory(**kwargs)
        comparison = compare_trajectories(primary, replay)
        seed_equal, seed_differences = compare_seed_tuples(primary_seeds, replay_seeds)
        passed = bool(
            comparison.repaired_comparator_passed
            and comparison.discrete_divergence_count == 0
            and comparison.finite_numeric_divergence_count == 0
            and comparison.forbidden_nonfinite_difference_count == 0
            and seed_equal
            and not seed_differences
            and primary.trajectory_sha256 == replay.trajectory_sha256
        )
        if not passed:
            raise RuntimeError(
                f"S13Y simulator replay failed {candidate_id}/M{matrix_index:03d}"
            )
        if (
            primary.completed_fissions != 100
            or primary.terminal_status != "requested_fissions_completed"
        ):
            raise RuntimeError(
                f"S13Y incomplete trajectory {candidate_id}/M{matrix_index:03d}: "
                f"{primary.completed_fissions}/{primary.terminal_status}"
            )
        candidate_root = RAW_ROOT / candidate_id
        candidate_root.mkdir(parents=True, exist_ok=True)
        cache_path = candidate_root / f"M{matrix_index:03d}.pickle"
        with cache_path.open("wb") as handle:
            pickle.dump(primary, handle, protocol=5)
        if (
            primary.beta_sha256 != beta_hash
            or primary.initial_state_sha256 != initial_hash
        ):
            raise RuntimeError(
                "shared beta or initial-state identity changed inside simulator"
            )
        trajectories.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "trajectoryId": primary.trajectory_id,
                "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
                "cachePath": str(cache_path),
                "cacheSha256": sha256_file(cache_path),
                "betaSha256": beta_hash,
                "initialStateSha256": initial_hash,
                "trajectorySha256": primary.trajectory_sha256,
                "completedFissions": primary.completed_fissions,
                "terminalStatus": primary.terminal_status,
                "exactReplayPassed": True,
            }
        )
        summary = trajectory_summary(primary)
        summary.update(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "tPhiLockedC1": int(
                    primary.total_batch_updates + primary.completed_fissions
                ),
            }
        )
        summaries.append(summary)
        replay_rows.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "trajectoryId": primary.trajectory_id,
                "oldComparatorPassed": comparison.old_comparator_passed,
                "repairedComparatorPassed": comparison.repaired_comparator_passed,
                "discreteDivergenceCount": comparison.discrete_divergence_count,
                "finiteNumericDivergenceCount": comparison.finite_numeric_divergence_count,
                "permittedPairedNanCount": comparison.permitted_paired_nan_count,
                "forbiddenNonfiniteDifferenceCount": comparison.forbidden_nonfinite_difference_count,
                "seedTupleExact": seed_equal,
                "trajectorySha256Exact": primary.trajectory_sha256
                == replay.trajectory_sha256,
                "passed": passed,
            }
        )
        for seed in primary_seeds:
            if seed.purpose not in {"catalytic_matrix", "initial_state"}:
                seeds.append(simulation_seed_row(seed, candidate_id))
    return {
        "matrixIndex": matrix_index,
        "trajectories": trajectories,
        "replay": replay_rows,
        "summaries": summaries,
        "seeds": seeds,
        "wallSeconds": time.perf_counter() - started_wall,
        "cpuSeconds": time.process_time() - started_cpu,
    }


def execute_simulations(indices: list[int], workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [simulate_shared_matrix(index) for index in indices]
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(simulate_shared_matrix, index): index for index in indices
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"S13Y simulation failed M{index:03d}: {type(exc).__name__}:{exc}"
                ) from exc
            records.append(record)
            print(
                json.dumps(
                    {
                        "stage": "simulation_pair_complete",
                        "matrixIndex": index,
                        "wallSeconds": record["wallSeconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return sorted(records, key=lambda item: item["matrixIndex"])


def source_seed_row(
    candidate_id: str,
    matrix_index: int,
    mode: str,
    endpoint: str | int,
    purpose: str,
) -> tuple[int, dict[str, Any]]:
    seed = derive_seed(
        "source", candidate_id, matrix_index, IMPLEMENTATION_ID, mode, endpoint, purpose
    )
    stream_id = (
        f"S13Y::SOURCE::{candidate_id}::M{matrix_index:03d}::"
        f"{IMPLEMENTATION_ID}::{endpoint}::{purpose}"
    )
    return seed, {
        "researchStepId": RESEARCH_STEP_ID,
        "streamDomain": "source",
        "streamId": stream_id,
        "purpose": purpose,
        "candidateId": candidate_id,
        "matrixIndex": matrix_index,
        "implementationId": IMPLEMENTATION_ID,
        "temporalModeId": mode,
        "endpointGeneration": None if endpoint == "FULL" else int(endpoint),
        "derivedSeed": str(seed),
        "seedMaterialSha256": seed_material_sha256(
            "source",
            candidate_id,
            matrix_index,
            IMPLEMENTATION_ID,
            mode,
            endpoint,
            purpose,
        ),
        "rootHex": ROOT_SEED_HEX,
        "bitGenerator": "MT19937_LEGACY_RANDOMSTATE",
        "sharedAcrossCandidates": False,
    }


def suffix_seed_row(
    candidate_id: str, matrix_index: int, generation: int, purpose: str
) -> tuple[int, dict[str, Any]]:
    seed = derive_seed("suffix", candidate_id, matrix_index, generation, purpose)
    return seed, {
        "researchStepId": RESEARCH_STEP_ID,
        "streamDomain": "suffix_validation",
        "streamId": f"S13Y::SUFFIX::{candidate_id}::M{matrix_index:03d}::G{generation:03d}::{purpose}",
        "purpose": purpose,
        "candidateId": candidate_id,
        "matrixIndex": matrix_index,
        "implementationId": IMPLEMENTATION_ID,
        "temporalModeId": PREFIX_MODE_ID,
        "endpointGeneration": generation,
        "derivedSeed": str(seed),
        "seedMaterialSha256": seed_material_sha256(
            "suffix", candidate_id, matrix_index, generation, purpose
        ),
        "rootHex": ROOT_SEED_HEX,
        "bitGenerator": "MT19937_LEGACY_RANDOMSTATE",
        "sharedAcrossCandidates": False,
    }


def point_values(result: Any, local_index: int) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for field, attribute in (
        ("synergy", "synergy"),
        ("downwardCausation", "downward_causation"),
        ("emergence", "emergence"),
        ("localPhiR", "local_phi_r"),
    ):
        array = getattr(result, attribute)
        if array is None or local_index < 0 or local_index >= len(array):
            output[field] = None
        else:
            value = float(array[local_index])
            output[field] = value if np.isfinite(value) else None
    return output


def point_status(
    result: Any, replay_passed: bool, values: dict[str, Any]
) -> tuple[str, str | None]:
    if not replay_passed:
        return "INELIGIBLE_EXACT_REPLAY_FAILED", "source_pipeline_replay_failed"
    if result.status not in ELIGIBLE_SOURCE_STATUSES:
        return str(result.status), result.reason
    if values["emergence"] is None:
        return "INELIGIBLE_NONFINITE_EMERGENCE", "emergence_nonfinite_or_absent"
    return "ELIGIBLE", None


def source_task_root(candidate_id: str, matrix_index: int) -> Path:
    return SOURCE_ROOT / candidate_id / f"M{matrix_index:03d}"


def process_source_task(task: dict[str, Any]) -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    candidate_id = str(task["candidateId"])
    matrix_index = int(task["matrixIndex"])
    root = source_task_root(candidate_id, matrix_index)
    completion = root / "completion.json"
    if completion.is_file():
        cached = json.loads(completion.read_text(encoding="utf-8"))
        if cached["inputCacheSha256"] == task["cacheSha256"]:
            cached["resumed"] = True
            return cached
        raise RuntimeError(
            f"stale S13Y source cache {candidate_id}/M{matrix_index:03d}"
        )
    if root.exists():
        raise RuntimeError(f"incomplete S13Y source cache exists: {root}")
    root.mkdir(parents=True)
    cache_path = Path(str(task["cachePath"]))
    if sha256_file(cache_path) != task["cacheSha256"]:
        raise RuntimeError("raw trajectory cache hash changed")
    with cache_path.open("rb") as handle:
        trajectory = pickle.load(handle)
    if (
        trajectory.configuration_id != candidate_id
        or int(trajectory.matrix_index) != matrix_index
        or trajectory.trajectory_sha256 != task["trajectorySha256"]
        or trajectory.beta_sha256 != task["betaSha256"]
        or trajectory.initial_state_sha256 != task["initialStateSha256"]
        or int(trajectory.completed_fissions) != 100
    ):
        raise RuntimeError("locked S13Y trajectory identity mismatch")

    selected = selected_clock_observations(trajectory, "C1_SELECTED_DAUGHTER_RETAINED")
    states = states_from_observations(selected)
    clr, masses, closure_errors = frozen_clr(states)
    compositions = states.astype(np.float64) / states.sum(axis=1, keepdims=True)
    changes = np.linalg.norm(np.diff(compositions, axis=0), axis=1)
    changes = np.concatenate(([changes[0]], changes))
    primary_labels, primary_fingerprint = label_trajectory(trajectory, PRIMARY_SPEC)
    sensitivity_labels, sensitivity_fingerprint = label_trajectory(
        trajectory, SENSITIVITY_SPEC
    )
    frozen_rows, historical_map, _ = frozen_generation_labels(trajectory)
    historical_source = [
        row for row in frozen_rows if row["labelId"] == HISTORICAL_LABEL_ID
    ]
    historical_score_map = {
        int(row["generation"]): row["historicalIncomingH"] for row in historical_source
    }
    primary_by_index = primary_labels.set_index("selectedSequenceIndex")
    sensitivity_by_index = sensitivity_labels.set_index("selectedSequenceIndex")
    if not np.array_equal(
        primary_by_index["isReplicator"].astype(bool).to_numpy(),
        primary_by_index["labelScore"].to_numpy(float) > 0.9,
    ):
        raise RuntimeError("S13Y primary molecular label identity failed inside task")
    label_frames: list[pd.DataFrame] = []
    for frame in (primary_labels, sensitivity_labels):
        value = frame.copy()
        value.insert(0, "researchStepId", RESEARCH_STEP_ID)
        value["labelStatus"] = "ELIGIBLE"
        value["ineligibilityReason"] = None
        label_frames.append(value)
    historical_rows = []
    for index, observation in enumerate(selected):
        generation = int(observation.growth_generation_one_based)
        label = historical_map.get(generation)
        historical_rows.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": candidate_id,
                "trajectoryId": trajectory.trajectory_id,
                "matrixIndex": matrix_index,
                "labelId": HISTORICAL_LABEL_ID,
                "labelFamily": "FROZEN_HISTORICAL_POSTFISSION_TECHNIQUE1_PROPAGATED",
                "labelEvidenceTier": "SOURCE_TRACEABLE_HISTORICAL_RECONSTRUCTION",
                "selectedSequenceIndex": index,
                "rawObservationIndex": int(observation.observation_index),
                "generation": generation,
                "observationKind": str(observation.observation_kind),
                "isReplicator": label,
                "labelScore": historical_score_map.get(generation),
                "labelStatus": "ELIGIBLE"
                if label is not None
                else "INELIGIBLE_NO_GENERATION_LABEL",
                "ineligibilityReason": None
                if label is not None
                else "initial_state_or_undefined_historical_label",
            }
        )
    labels = pd.concat(
        [*label_frames, pd.DataFrame(historical_rows)], ignore_index=True
    )
    fingerprints = pd.DataFrame(
        [
            {"researchStepId": RESEARCH_STEP_ID, **primary_fingerprint},
            {"researchStepId": RESEARCH_STEP_ID, **sensitivity_fingerprint},
        ]
    )

    endpoints = post_fission_endpoint_records(
        trajectory, "C1_SELECTED_DAUGHTER_RETAINED", minimum_prior_transitions=0
    )
    eligible = [
        item for item in endpoints if item.prior_locked_clock_transitions >= 256
    ]
    sentinel_generations = (
        {
            eligible[0].generation,
            eligible[len(eligible) // 2].generation,
            eligible[-1].generation,
        }
        if eligible
        else set()
    )
    endpoint_by_generation = {item.generation: item for item in endpoints}
    full_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    partitions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    suffix_rows: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    evaluations = 0

    full_pre, row = source_seed_row(
        candidate_id, matrix_index, FULL_MODE_ID, "FULL", "source_preprocessing"
    )
    seeds.append(row)
    full_part, row = source_seed_row(
        candidate_id, matrix_index, FULL_MODE_ID, "FULL", "source_partition"
    )
    seeds.append(row)
    full = run_emergence_pipeline(
        clr,
        PHIRL,
        SAFE_LATTICE,
        preprocessing_seed=full_pre,
        partition_seed=full_part,
    )
    full_replay = run_emergence_pipeline(
        clr,
        PHIRL,
        SAFE_LATTICE,
        preprocessing_seed=full_pre,
        partition_seed=full_part,
    )
    evaluations += 2
    full_replay_ok = result_replay_equal(full, full_replay)
    partitions.append(
        {
            "researchStepId": RESEARCH_STEP_ID,
            "candidateId": candidate_id,
            "trajectoryId": trajectory.trajectory_id,
            "matrixIndex": matrix_index,
            "implementationId": IMPLEMENTATION_ID,
            "temporalModeId": FULL_MODE_ID,
            "fitKind": "completed_trajectory",
            "endpointGeneration": 100,
            "endpointSelectedSequenceIndex": len(selected) - 1,
            "status": full.status,
            "reason": full.reason,
            "retainedVariablesJson": json.dumps(
                list(full.retained_variables), separators=(",", ":")
            ),
            "partition1Json": json.dumps(list(full.partition_1), separators=(",", ":")),
            "partition2Json": json.dumps(list(full.partition_2), separators=(",", ":")),
            "partitionSize1": len(full.partition_1),
            "partitionSize2": len(full.partition_2),
            "exactReplayPassed": full_replay_ok,
        }
    )
    diagnostics.append(
        {
            "researchStepId": RESEARCH_STEP_ID,
            "candidateId": candidate_id,
            "trajectoryId": trajectory.trajectory_id,
            "matrixIndex": matrix_index,
            "implementationId": IMPLEMENTATION_ID,
            "temporalModeId": FULL_MODE_ID,
            "fitKind": "completed_trajectory",
            "endpointGeneration": 100,
            "status": full.status,
            "reason": full.reason,
            "componentIdentityMaxAbsError": full.component_identity_max_abs_error,
            "retainedVariableCount": len(full.retained_variables),
            "miFinite": bool(
                full.mi_matrix is not None and np.all(np.isfinite(full.mi_matrix))
            ),
            "partitionAverageFinite": bool(
                full.partition_average is not None
                and np.all(np.isfinite(full.partition_average))
            ),
            "emergenceFiniteCount": int(np.sum(np.isfinite(full.emergence)))
            if full.emergence is not None
            else 0,
        }
    )
    for local_index in range(max(0, len(selected) - full.local_offset)):
        selected_index = local_index + full.local_offset
        observation = selected[selected_index]
        values = point_values(full, local_index)
        status, reason = point_status(full, full_replay_ok, values)
        generation = int(observation.growth_generation_one_based)
        full_rows.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": candidate_id,
                "trajectoryId": trajectory.trajectory_id,
                "matrixIndex": matrix_index,
                "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
                "implementationId": IMPLEMENTATION_ID,
                "temporalModeId": FULL_MODE_ID,
                "temporalLabel": "RETROSPECTIVE_FULL_TRAJECTORY_LOCAL",
                "selectedSequenceIndex": selected_index,
                "rawObservationIndex": int(observation.observation_index),
                "observationKind": str(observation.observation_kind),
                "generation": generation,
                "molecularStep": int(observation.batch_step),
                "status": status,
                "reason": reason,
                **values,
                "incomingCosineH": float(
                    primary_by_index.loc[selected_index, "labelScore"]
                ),
                "euclideanL2ClosedCompositionChange": float(changes[selected_index]),
                "molecularH090Label": bool(
                    primary_by_index.loc[selected_index, "isReplicator"]
                ),
                "molecularH970Label": bool(
                    sensitivity_by_index.loc[selected_index, "isReplicator"]
                ),
                "historicalH090Label": historical_map.get(generation),
                "exactReplayPassed": full_replay_ok,
            }
        )

    prefix_replay_all = True
    suffix_all = True
    for endpoint in endpoints:
        generation = endpoint.generation
        current_index = endpoint.selected_sequence_index
        next_endpoint = endpoint_by_generation.get(generation + 1)
        base = {
            "researchStepId": RESEARCH_STEP_ID,
            "candidateId": candidate_id,
            "trajectoryId": trajectory.trajectory_id,
            "matrixIndex": matrix_index,
            "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
            "implementationId": IMPLEMENTATION_ID,
            "temporalModeId": PREFIX_MODE_ID,
            "temporalLabel": "PAST_ONLY_PREFIX_ENDPOINT",
            "generation": generation,
            "endpointSelectedSequenceIndex": current_index,
            "endpointRawObservationIndex": endpoint.raw_observation_index,
            "endpointObservationKind": endpoint.observation_kind,
            "priorLockedClockTransitions": endpoint.prior_locked_clock_transitions,
            "fitObservationCount": current_index + 1,
            "currentIncomingCosineH": float(
                primary_by_index.loc[current_index, "labelScore"]
            ),
            "currentMolecularH090Label": bool(
                primary_by_index.loc[current_index, "isReplicator"]
            ),
            "currentMolecularH970Label": bool(
                sensitivity_by_index.loc[current_index, "isReplicator"]
            ),
            "nextMolecularH090Label": (
                bool(
                    primary_by_index.loc[
                        next_endpoint.selected_sequence_index, "isReplicator"
                    ]
                )
                if next_endpoint is not None
                else None
            ),
            "nextMolecularH970Label": (
                bool(
                    sensitivity_by_index.loc[
                        next_endpoint.selected_sequence_index, "isReplicator"
                    ]
                )
                if next_endpoint is not None
                else None
            ),
        }
        if endpoint.prior_locked_clock_transitions < 256:
            prefix_rows.append(
                {
                    **base,
                    "status": "INELIGIBLE_BEFORE_256_TRANSITIONS",
                    "reason": "fewer_than_256_prior_locked_clock_transitions",
                    "synergy": None,
                    "downwardCausation": None,
                    "emergence": None,
                    "localPhiR": None,
                    "exactReplayPassed": None,
                    "futureSuffixStructuralGatePassed": None,
                    "futureSuffixExecutedSentinelPassed": None,
                }
            )
            continue
        stop = current_index + 1
        prefix = np.ascontiguousarray(clr[:stop])
        prefix_hash = sha256_array(prefix)
        pre_seed, row = source_seed_row(
            candidate_id,
            matrix_index,
            PREFIX_MODE_ID,
            generation,
            "source_preprocessing",
        )
        seeds.append(row)
        part_seed, row = source_seed_row(
            candidate_id, matrix_index, PREFIX_MODE_ID, generation, "source_partition"
        )
        seeds.append(row)
        result = run_emergence_pipeline(
            prefix,
            PHIRL,
            SAFE_LATTICE,
            preprocessing_seed=pre_seed,
            partition_seed=part_seed,
        )
        replay = run_emergence_pipeline(
            prefix,
            PHIRL,
            SAFE_LATTICE,
            preprocessing_seed=pre_seed,
            partition_seed=part_seed,
        )
        evaluations += 2
        replay_ok = result_replay_equal(result, replay)
        prefix_replay_all &= replay_ok
        values = point_values(result, len(prefix) - result.local_offset - 1)
        status, reason = point_status(result, replay_ok, values)
        structural_pass = True
        sentinel_pass: bool | None = None
        for variant_id in (
            "suffix_deletion",
            "suffix_deterministic_shuffle",
            "suffix_domain_separated_replacement",
        ):
            if variant_id == "suffix_deletion":
                variant_prefix = np.ascontiguousarray(clr[:stop])
            else:
                variant_full = clr.copy()
                suffix_length = len(clr) - stop
                purpose = variant_id
                random_seed, seed_row = suffix_seed_row(
                    candidate_id, matrix_index, generation, purpose
                )
                seeds.append(seed_row)
                if suffix_length:
                    rng = np.random.RandomState(random_seed)
                    if variant_id == "suffix_deterministic_shuffle":
                        variant_full[stop:] = variant_full[stop:][
                            rng.permutation(suffix_length)
                        ]
                    else:
                        variant_full[stop:] = rng.normal(size=variant_full[stop:].shape)
                variant_prefix = np.ascontiguousarray(variant_full[:stop])
            variant_hash = sha256_array(variant_prefix)
            structural_exact = bool(
                variant_hash == prefix_hash
                and np.array_equal(variant_prefix, prefix, equal_nan=True)
            )
            structural_pass &= structural_exact
            result_exact: bool | None = None
            sentinel = "non_sentinel"
            if generation in sentinel_generations:
                positions = [
                    eligible[0].generation,
                    eligible[len(eligible) // 2].generation,
                    eligible[-1].generation,
                ]
                names = ["first", "middle", "last"]
                sentinel = "+".join(
                    name
                    for name, position in zip(names, positions, strict=True)
                    if position == generation
                )
                variant_result = run_emergence_pipeline(
                    variant_prefix,
                    PHIRL,
                    SAFE_LATTICE,
                    preprocessing_seed=pre_seed,
                    partition_seed=part_seed,
                )
                evaluations += 1
                result_exact = result_replay_equal(result, variant_result)
                sentinel_pass = bool(
                    (True if sentinel_pass is None else sentinel_pass)
                    and structural_exact
                    and result_exact
                )
            passed = structural_exact and result_exact is not False
            suffix_rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "candidateId": candidate_id,
                    "trajectoryId": trajectory.trajectory_id,
                    "matrixIndex": matrix_index,
                    "implementationId": IMPLEMENTATION_ID,
                    "endpointGeneration": generation,
                    "validationKind": variant_id,
                    "sentinel": sentinel,
                    "prefixSha256": prefix_hash,
                    "variantPrefixSha256": variant_hash,
                    "structuralExact": structural_exact,
                    "resultExact": result_exact,
                    "status": "PASS" if passed else "FAIL",
                    "reason": None if passed else "future_suffix_invariance_failed",
                }
            )
        suffix_all &= structural_pass and sentinel_pass is not False
        if not structural_pass or sentinel_pass is False:
            status = "INELIGIBLE_FUTURE_SUFFIX_INVARIANCE_FAILED"
            reason = "future_suffix_invariance_failed"
            values = {key: None for key in values}
        prefix_rows.append(
            {
                **base,
                "status": status,
                "reason": reason,
                **values,
                "exactReplayPassed": replay_ok,
                "futureSuffixStructuralGatePassed": structural_pass,
                "futureSuffixExecutedSentinelPassed": sentinel_pass,
            }
        )
        partitions.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": candidate_id,
                "trajectoryId": trajectory.trajectory_id,
                "matrixIndex": matrix_index,
                "implementationId": IMPLEMENTATION_ID,
                "temporalModeId": PREFIX_MODE_ID,
                "fitKind": "past_only_prefix_endpoint",
                "endpointGeneration": generation,
                "endpointSelectedSequenceIndex": current_index,
                "status": result.status,
                "reason": result.reason,
                "retainedVariablesJson": json.dumps(
                    list(result.retained_variables), separators=(",", ":")
                ),
                "partition1Json": json.dumps(
                    list(result.partition_1), separators=(",", ":")
                ),
                "partition2Json": json.dumps(
                    list(result.partition_2), separators=(",", ":")
                ),
                "partitionSize1": len(result.partition_1),
                "partitionSize2": len(result.partition_2),
                "exactReplayPassed": replay_ok,
            }
        )
        diagnostics.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": candidate_id,
                "trajectoryId": trajectory.trajectory_id,
                "matrixIndex": matrix_index,
                "implementationId": IMPLEMENTATION_ID,
                "temporalModeId": PREFIX_MODE_ID,
                "fitKind": "past_only_prefix_endpoint",
                "endpointGeneration": generation,
                "status": result.status,
                "reason": result.reason,
                "componentIdentityMaxAbsError": result.component_identity_max_abs_error,
                "retainedVariableCount": len(result.retained_variables),
                "miFinite": bool(
                    result.mi_matrix is not None
                    and np.all(np.isfinite(result.mi_matrix))
                ),
                "partitionAverageFinite": bool(
                    result.partition_average is not None
                    and np.all(np.isfinite(result.partition_average))
                ),
                "emergenceFiniteCount": int(np.sum(np.isfinite(result.emergence)))
                if result.emergence is not None
                else 0,
            }
        )
        if not replay_ok or not structural_pass or sentinel_pass is False:
            failures.append(
                {
                    "failureId": f"S13Y-{candidate_id}-M{matrix_index:03d}-G{generation:03d}",
                    "stage": "source_execution",
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "severity": "FATAL",
                    "status": status,
                    "reason": reason,
                    "gateImpact": "FAIL_CLOSED",
                    "repairAttempted": False,
                }
            )

    preprocessing = pd.DataFrame(
        [
            {
                "researchStepId": RESEARCH_STEP_ID,
                "candidateId": candidate_id,
                "trajectoryId": trajectory.trajectory_id,
                "matrixIndex": matrix_index,
                "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
                "observationCount": len(selected),
                "inputDimension": 100,
                "outputDimension": 99,
                "minimumMass": float(np.min(masses)),
                "maximumMass": float(np.max(masses)),
                "maximumClosureError": float(np.max(closure_errors)),
                "finite": bool(np.all(np.isfinite(clr))),
                "clrSha256": sha256_array(clr),
                "status": "ELIGIBLE",
                "reason": None,
            }
        ]
    )
    outputs = {
        "labels.parquet": labels,
        "fingerprints.parquet": fingerprints,
        "preprocessing.parquet": preprocessing,
        "full.parquet": pd.DataFrame(full_rows),
        "prefix.parquet": pd.DataFrame(prefix_rows),
        "partition.parquet": pd.DataFrame(partitions),
        "diagnostic.parquet": pd.DataFrame(diagnostics),
        "suffix.parquet": pd.DataFrame(suffix_rows),
        "seeds.parquet": pd.DataFrame(seeds),
        "failures.parquet": pd.DataFrame(
            failures,
            columns=[
                "failureId",
                "stage",
                "candidateId",
                "matrixIndex",
                "severity",
                "status",
                "reason",
                "gateImpact",
                "repairAttempted",
            ],
        ),
    }
    for filename, frame in outputs.items():
        write_parquet(root / filename, frame)
    record = {
        "candidateId": candidate_id,
        "matrixIndex": matrix_index,
        "trajectoryId": trajectory.trajectory_id,
        "inputCacheSha256": task["cacheSha256"],
        "resultRoot": str(root),
        "selectedObservationCount": len(selected),
        "eligibleEndpointCount": len(eligible),
        "fullRows": len(full_rows),
        "prefixRows": len(prefix_rows),
        "suffixRows": len(suffix_rows),
        "failureRows": len(failures),
        "evaluationCount": evaluations,
        "fullReplayAllPassed": full_replay_ok,
        "prefixReplayAllPassed": prefix_replay_all,
        "futureSuffixAllPassed": suffix_all,
        "wallSeconds": time.perf_counter() - started_wall,
        "cpuSeconds": time.process_time() - started_cpu,
        "resumed": False,
    }
    write_json(completion, record)
    return record


def execute_source(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if workers == 1:
        return [process_source_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_source_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"S13Y source task failed {task['candidateId']}/M{int(task['matrixIndex']):03d}: "
                    f"{type(exc).__name__}:{exc}"
                ) from exc
            records.append(record)
            print(
                json.dumps(
                    {
                        "stage": "source_task_complete",
                        "candidateId": record["candidateId"],
                        "matrixIndex": record["matrixIndex"],
                        "wallSeconds": record["wallSeconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return sorted(records, key=lambda row: (row["candidateId"], row["matrixIndex"]))


def validate_method_lock() -> dict[str, Any]:
    lock = json.loads((STEP_ROOT / "method_lock.json").read_text(encoding="utf-8"))
    mismatches = []
    for row in lock["files"]:
        path = REPO / row["path"]
        actual = sha256_file(path)
        if actual != row["sha256"]:
            mismatches.append(
                {"path": str(path), "expected": row["sha256"], "actual": actual}
            )
    state = {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "remoteHead": git("rev-parse", "origin/eidosoma/groups/42"),
        "workingTree": git("status", "--short"),
    }
    passed = bool(
        not mismatches
        and state["branch"] == "eidosoma/groups/42"
        and state["head"] == lock["designState"]["head"]
        and state["remoteHead"] == state["head"]
        and state["workingTree"] == ""
    )
    return {
        "schema": "eidosoma.e01.s13y_implementation_lock_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "mismatches": mismatches,
        "designState": state,
        "passed": passed,
    }


def validate_prior() -> dict[str, Any]:
    baseline = json.loads((STEP_ROOT / "immutable_prior_baseline.json").read_text())
    mismatches = []
    for row in baseline["files"]:
        path = Path(row["path"])
        if not path.is_file():
            mismatches.append({"path": str(path), "reason": "missing"})
        else:
            actual = sha256_file(path)
            if actual != row["sha256"] or path.stat().st_size != row["bytes"]:
                mismatches.append(
                    {
                        "path": str(path),
                        "reason": "hash_or_size_changed",
                        "actualSha256": actual,
                    }
                )
    return {
        "schema": "eidosoma.e01.s13y_immutable_prior_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "expectedFileCount": baseline["fileCount"],
        "checkedFileCount": len(baseline["files"]),
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def load_simulation_records() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    manifest_rows, summary_rows, replay_rows, seed_rows = [], [], [], []
    for matrix_index in range(100):
        for candidate_id in CANDIDATE_IDS:
            path = RAW_ROOT / candidate_id / f"M{matrix_index:03d}.pickle"
            if not path.is_file():
                continue
            with path.open("rb") as handle:
                trajectory = pickle.load(handle)
            manifest_rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "trajectoryId": trajectory.trajectory_id,
                    "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
                    "cachePath": str(path),
                    "cacheSha256": sha256_file(path),
                    "betaSha256": trajectory.beta_sha256,
                    "initialStateSha256": trajectory.initial_state_sha256,
                    "trajectorySha256": trajectory.trajectory_sha256,
                    "completedFissions": trajectory.completed_fissions,
                    "terminalStatus": trajectory.terminal_status,
                    "exactReplayPassed": True,
                }
            )
            summary = trajectory_summary(trajectory)
            summary.update(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "tPhiLockedC1": trajectory.total_batch_updates
                    + trajectory.completed_fissions,
                }
            )
            summary_rows.append(summary)
            kwargs = {
                "phase": SIMULATION_PHASE,
                "root_hex": ROOT_SEED_HEX,
                "matrix_index": matrix_index,
                "definition": candidate_definition(candidate_id),
                "stream_identity": candidate_id,
            }
            beta_seed = derive_simulation_seed(
                ROOT_SEED_HEX, SIMULATION_PHASE, "catalytic_matrix", matrix_index
            )
            init_seed = derive_simulation_seed(
                ROOT_SEED_HEX, SIMULATION_PHASE, "initial_state", matrix_index
            )
            beta = generate_beta(beta_seed)
            initial = initialize_distinct_state(init_seed)
            replay, replay_seeds = simulate_trajectory(
                **kwargs, beta=beta, initial_state=initial
            )
            comparison = compare_trajectories(trajectory, replay)
            replay_rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "trajectoryId": trajectory.trajectory_id,
                    "repairedComparatorPassed": comparison.repaired_comparator_passed,
                    "discreteDivergenceCount": comparison.discrete_divergence_count,
                    "finiteNumericDivergenceCount": comparison.finite_numeric_divergence_count,
                    "forbiddenNonfiniteDifferenceCount": comparison.forbidden_nonfinite_difference_count,
                    "trajectorySha256Exact": trajectory.trajectory_sha256
                    == replay.trajectory_sha256,
                    "passed": bool(
                        comparison.repaired_comparator_passed
                        and trajectory.trajectory_sha256 == replay.trajectory_sha256
                    ),
                }
            )
            for seed in replay_seeds:
                if (
                    seed.purpose in {"catalytic_matrix", "initial_state"}
                    and candidate_id != CANDIDATE_IDS[0]
                ):
                    continue
                seed_rows.append(simulation_seed_row(seed, candidate_id))
    return (
        pd.DataFrame(manifest_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(replay_rows),
        pd.DataFrame(seed_rows),
    )


def collate_source(records: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    mapping = {
        "labels.parquet": "label_values.parquet",
        "fingerprints.parquet": "label_fingerprints.parquet",
        "preprocessing.parquet": "preprocessing_diagnostics.parquet",
        "full.parquet": "full_source_values.parquet",
        "prefix.parquet": "prefix_endpoint_values.parquet",
        "partition.parquet": "partition_history.parquet",
        "diagnostic.parquet": "source_diagnostic_outputs.parquet",
        "suffix.parquet": "replay_suffix_validation.parquet",
        "seeds.parquet": "source_seed_manifest.parquet",
    }
    outputs: dict[str, pd.DataFrame] = {}
    for source_name, target_name in mapping.items():
        frames = [
            pd.read_parquet(Path(record["resultRoot"]) / source_name)
            for record in records
        ]
        frame = pd.concat(frames, ignore_index=True)
        sort_keys = [
            key
            for key in (
                "candidateId",
                "matrixIndex",
                "labelId",
                "implementationId",
                "temporalModeId",
                "generation",
                "selectedSequenceIndex",
                "endpointGeneration",
                "validationKind",
                "purpose",
            )
            if key in frame
        ]
        if sort_keys:
            frame.sort_values(sort_keys, kind="stable", inplace=True, ignore_index=True)
        write_parquet(STEP_ROOT / target_name, frame)
        outputs[target_name] = frame
    failure_frames = [
        pd.read_parquet(Path(record["resultRoot"]) / "failures.parquet")
        for record in records
    ]
    failures = pd.concat(failure_frames, ignore_index=True)
    outputs["worker_failures"] = failures
    return outputs


def trajectory_details(
    frame: pd.DataFrame,
    *,
    label_column: str,
) -> tuple[pd.DataFrame, dict[int, tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for (matrix_index, trajectory_id), group in frame.groupby(
        ["matrixIndex", "trajectoryId"], sort=True
    ):
        ordered = group.sort_values("selectedSequenceIndex", kind="stable")
        values = pd.to_numeric(ordered["emergence"], errors="coerce").to_numpy(float)
        labels = pd.to_numeric(ordered[label_column], errors="coerce").to_numpy(float)
        result = association_summary(values, labels)
        rows.append(
            {
                "candidateId": str(ordered["candidateId"].iloc[0]),
                "matrixIndex": int(matrix_index),
                "trajectoryId": str(trajectory_id),
                "labelId": label_column,
                **result,
            }
        )
        mask = np.isfinite(values) & np.isfinite(labels)
        payloads[int(matrix_index)] = (values[mask], labels[mask].astype(bool))
    return pd.DataFrame(rows), payloads


def compute_retrospective(
    full: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    all_details: list[pd.DataFrame] = []
    result_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    payload_store: dict[tuple[str, str], dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    labels = (
        (PRIMARY_LABEL_ID, "molecularH090Label"),
        (SENSITIVITY_LABEL_ID, "molecularH970Label"),
        (HISTORICAL_LABEL_ID, "historicalH090Label"),
    )
    for candidate_id in CANDIDATE_IDS:
        candidate = full[full["candidateId"] == candidate_id].copy()
        coverage = float(
            np.isfinite(pd.to_numeric(candidate["emergence"], errors="coerce")).mean()
        )
        for label_id, column in labels:
            details, payloads = trajectory_details(candidate, label_column=column)
            details["labelId"] = label_id
            details["implementationId"] = IMPLEMENTATION_ID
            details["metric"] = METRIC_ID
            details["temporalMode"] = "RETROSPECTIVE_FULL_TRAJECTORY_LOCAL"
            all_details.append(details)
            payload_store[(candidate_id, label_id)] = payloads
            summary = summarize_resampled_direction(
                details,
                payloads,
                seed_identity=("statistics", candidate_id, f"retrospective_{label_id}"),
                finite_coverage=coverage,
            )
            ordinary = pd.to_numeric(details["ordinaryTwoSidedP"], errors="coerce")
            rho = pd.to_numeric(details["rho"], errors="coerce")
            row = {
                "candidateId": candidate_id,
                "implementationId": IMPLEMENTATION_ID,
                "metric": METRIC_ID,
                "temporalMode": "RETROSPECTIVE_FULL_TRAJECTORY_LOCAL",
                "labelId": label_id,
                "trajectoryCount": len(details),
                "ordinaryPositivePCount": int(
                    np.count_nonzero((rho > 0) & (ordinary < 0.05))
                ),
                **summary,
            }
            association_pass = (
                primary_association_gate(row) if label_id == PRIMARY_LABEL_ID else None
            )
            drift_pass = (
                primary_drift_gate(row) if label_id == PRIMARY_LABEL_ID else None
            )
            row["associationGatePassed"] = association_pass
            row["driftGatePassed"] = drift_pass
            row["candidatePrimaryPassed"] = (
                bool(association_pass and drift_pass)
                if label_id == PRIMARY_LABEL_ID
                else None
            )
            result_rows.append(row)
            inference_rows.append(
                {
                    **row,
                    "bootstrapMethod": "trajectory_median_4096",
                    "nullMethod": "duration_preserving_nonzero_circular_rotation_4096",
                }
            )
    return (
        pd.concat(all_details, ignore_index=True),
        pd.DataFrame(result_rows),
        pd.DataFrame(inference_rows),
        payload_store,
    )


def compute_prefix(
    prefix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    details_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    label_columns = {
        (PRIMARY_LABEL_ID, "CURRENT_ENDPOINT"): "currentMolecularH090Label",
        (PRIMARY_LABEL_ID, "NEXT_ENDPOINT"): "nextMolecularH090Label",
        (SENSITIVITY_LABEL_ID, "CURRENT_ENDPOINT"): "currentMolecularH970Label",
        (SENSITIVITY_LABEL_ID, "NEXT_ENDPOINT"): "nextMolecularH970Label",
    }
    for candidate_id in CANDIDATE_IDS:
        candidate_all = prefix[
            (prefix["candidateId"] == candidate_id)
            & (prefix["priorLockedClockTransitions"] >= 256)
        ].copy()
        candidate = candidate_all[candidate_all["status"] == "ELIGIBLE"].copy()
        coverage = (
            float(len(candidate) / len(candidate_all)) if len(candidate_all) else 0.0
        )
        for (label_id, alignment), column in label_columns.items():
            local_details: list[dict[str, Any]] = []
            payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for (matrix_index, trajectory_id), group in candidate.groupby(
                ["matrixIndex", "trajectoryId"], sort=True
            ):
                ordered = group.sort_values("generation", kind="stable")
                values = pd.to_numeric(ordered["emergence"], errors="coerce").to_numpy(
                    float
                )
                labels = pd.to_numeric(ordered[column], errors="coerce").to_numpy(float)
                result = association_summary(values, labels)
                local_details.append(
                    {
                        "candidateId": candidate_id,
                        "matrixIndex": int(matrix_index),
                        "trajectoryId": str(trajectory_id),
                        "implementationId": IMPLEMENTATION_ID,
                        "metric": METRIC_ID,
                        "temporalMode": "PAST_ONLY_PREFIX_ENDPOINT",
                        "labelId": label_id,
                        "alignment": alignment,
                        **result,
                    }
                )
                mask = np.isfinite(values) & np.isfinite(labels)
                payloads[int(matrix_index)] = (values[mask], labels[mask].astype(bool))
            details = pd.DataFrame(local_details)
            details_rows.extend(local_details)
            summary = summarize_resampled_direction(
                details,
                payloads,
                seed_identity=(
                    "statistics",
                    candidate_id,
                    f"prefix_{label_id}_{alignment}",
                ),
                finite_coverage=coverage,
            )
            row = {
                "candidateId": candidate_id,
                "implementationId": IMPLEMENTATION_ID,
                "metric": METRIC_ID,
                "temporalMode": "PAST_ONLY_PREFIX_ENDPOINT",
                "labelId": label_id,
                "alignment": alignment,
                "trajectoryCount": len(details),
                **summary,
            }
            row["prefixGatePassed"] = (
                prefix_gate(row)
                if label_id == PRIMARY_LABEL_ID and alignment == "CURRENT_ENDPOINT"
                else None
            )
            summary_rows.append(row)
            inference_rows.append(
                {
                    **row,
                    "bootstrapMethod": "trajectory_median_4096",
                    "nullMethod": "duration_preserving_nonzero_circular_rotation_4096",
                }
            )
    return (
        pd.DataFrame(details_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(inference_rows),
    )


def smooth_incremental_diagnostic(
    candidate: pd.DataFrame, candidate_id: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    cross_products: list[tuple[np.ndarray, np.ndarray, str, int]] = []
    rows = []
    for (matrix_index, trajectory_id), group in candidate.groupby(
        ["matrixIndex", "trajectoryId"], sort=True
    ):
        frame = group.copy()
        e = pd.to_numeric(frame["emergence"], errors="coerce").to_numpy(float)
        h = pd.to_numeric(frame["incomingCosineH"], errors="coerce").to_numpy(float)
        change = pd.to_numeric(
            frame["euclideanL2ClosedCompositionChange"], errors="coerce"
        ).to_numpy(float)
        y = frame["molecularH090Label"].astype(float).to_numpy()
        mask = np.isfinite(e) & np.isfinite(h) & np.isfinite(change) & np.isfinite(y)
        e, h, change, y = e[mask], h[mask], change[mask], y[mask]
        design = np.column_stack(
            (
                h,
                h**2,
                h**3,
                np.maximum(h - 0.95, 0.0),
                np.maximum(h - 0.99, 0.0),
                np.log1p(change),
                y,
            )
        )
        design -= design.mean(axis=0, keepdims=True)
        centered = e - e.mean()
        xtx, xty = design.T @ design, design.T @ centered
        coefficient = float(np.linalg.lstsq(design, centered, rcond=None)[0][-1])
        cross_products.append((xtx, xty, str(trajectory_id), int(matrix_index)))
        rows.append(
            {
                "candidateId": candidate_id,
                "matrixIndex": int(matrix_index),
                "trajectoryId": str(trajectory_id),
                "n": len(e),
                "smoothControlLabelCoefficient": coefficient,
            }
        )
    total_xtx = sum(item[0] for item in cross_products)
    total_xty = sum(item[1] for item in cross_products)
    coefficient = float(np.linalg.lstsq(total_xtx, total_xty, rcond=None)[0][-1])
    rng = np.random.default_rng(
        derive_seed(
            "statistics", candidate_id, "circularity_smooth_diagnostic", "bootstrap"
        )
    )
    bootstrap = np.empty(RESAMPLING_REPLICATES, dtype=float)
    n = len(cross_products)
    for index in range(RESAMPLING_REPLICATES):
        selected = rng.integers(0, n, size=n)
        xtx = sum(cross_products[item][0] for item in selected)
        xty = sum(cross_products[item][1] for item in selected)
        bootstrap[index] = np.linalg.lstsq(xtx, xty, rcond=None)[0][-1]
    return {
        "candidateId": candidate_id,
        "controlId": "FIXED_SMOOTH_H_L2_WITHIN_TRAJECTORY_EMERGENCE_REGRESSION",
        "coefficientForMolecularH090Label": coefficient,
        "bootstrapLower95": float(np.quantile(bootstrap, 0.025)),
        "bootstrapUpper95": float(np.quantile(bootstrap, 0.975)),
        "positiveDiagnostic": bool(
            coefficient > 0 and np.quantile(bootstrap, 0.025) > 0
        ),
        "interpretation": "MODEL_DEPENDENT_THRESHOLD_DISCONTINUITY_DIAGNOSTIC_NOT_INCREMENTAL_INFORMATION_BEYOND_EXACT_H",
    }, pd.DataFrame(rows)


def compute_circularity(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    summaries: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    all_identity = True
    for candidate_id in CANDIDATE_IDS:
        candidate = full[full["candidateId"] == candidate_id].copy()
        identity = exact_label_identity(
            candidate["incomingCosineH"].to_numpy(float),
            candidate["molecularH090Label"].astype(bool).to_numpy(),
        )
        all_identity &= identity["identityPassed"]
        h_rhos, change_rhos = [], []
        for (matrix_index, trajectory_id), group in candidate.groupby(
            ["matrixIndex", "trajectoryId"], sort=True
        ):
            finite = group.dropna(
                subset=[
                    "emergence",
                    "incomingCosineH",
                    "euclideanL2ClosedCompositionChange",
                ]
            )
            emergence = finite["emergence"].to_numpy(float)
            h = finite["incomingCosineH"].to_numpy(float)
            change = finite["euclideanL2ClosedCompositionChange"].to_numpy(float)
            h_rho = (
                float(spearmanr(emergence, h).statistic)
                if len(np.unique(h)) > 1
                else np.nan
            )
            change_rho = (
                float(spearmanr(emergence, -change).statistic)
                if len(np.unique(change)) > 1
                else np.nan
            )
            h_rhos.append(h_rho)
            change_rhos.append(change_rho)
        smooth, smooth_details = smooth_incremental_diagnostic(candidate, candidate_id)
        details.append(smooth_details)
        summaries.append(
            {
                "candidateId": candidate_id,
                "controlId": "EXACT_H_LABEL_DETERMINISM_AND_BASELINE_ASSOCIATIONS",
                **identity,
                "baselineHLabelClassificationAccuracy": 1.0
                if identity["identityPassed"]
                else None,
                "medianEmergenceContinuousHSpearman": float(np.nanmedian(h_rhos)),
                "medianEmergenceNegativeL2ChangeSpearman": float(
                    np.nanmedian(change_rhos)
                ),
                "unrestrictedIncrementalInformationBeyondExactH": 0.0
                if identity["identityPassed"]
                else None,
                "survivesExactHCircularityControl": False
                if identity["identityPassed"]
                else None,
                "interpretation": "LABEL_DETERMINISTICALLY_DEFINED_BY_H",
            }
        )
        summaries.append(smooth)
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True), all_identity


def temporal_and_spikes(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    temporal_rows, spike_rows = [], []
    for candidate_id in CANDIDATE_IDS:
        candidate = full[full["candidateId"] == candidate_id]
        aggregate = (
            candidate.groupby("selectedSequenceIndex", as_index=False)["emergence"]
            .median()
            .dropna()
        )
        fit = (
            linregress(aggregate["selectedSequenceIndex"], aggregate["emergence"])
            if len(aggregate) >= 3
            else None
        )
        per_run = []
        for (matrix_index, trajectory_id), group in candidate.groupby(
            ["matrixIndex", "trajectoryId"], sort=True
        ):
            values = (
                pd.to_numeric(
                    group.sort_values("selectedSequenceIndex")["emergence"],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(float)
            )
            mean, std = float(np.mean(values)), float(np.std(values))
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            robust_scale = 1.4826 * mad
            positive = int(np.count_nonzero(values > mean + 3 * std)) if std > 0 else 0
            negative = int(np.count_nonzero(values < mean - 3 * std)) if std > 0 else 0
            robust = (
                int(np.count_nonzero(np.abs(values - median) > 3 * robust_scale))
                if robust_scale > 0
                else 0
            )
            lag = max(1, min(10, len(values) // 5))
            raw_p = float(
                acorr_ljungbox(values, lags=[lag], return_df=True)["lb_pvalue"].iloc[0]
            )
            differenced = np.diff(values)
            diff_lag = max(1, min(10, len(differenced) // 5))
            diff_p = float(
                acorr_ljungbox(differenced, lags=[diff_lag], return_df=True)[
                    "lb_pvalue"
                ].iloc[0]
            )
            row = {
                "candidateId": candidate_id,
                "matrixIndex": int(matrix_index),
                "trajectoryId": str(trajectory_id),
                "positive3SigmaCount": positive,
                "negative3SigmaCount": negative,
                "robustMadExcursionCount": robust,
                "rawLjungBoxP": raw_p,
                "differencedLjungBoxP": diff_p,
            }
            per_run.append(row)
            spike_rows.append(row)
        run_frame = pd.DataFrame(per_run)
        temporal_rows.append(
            {
                "candidateId": candidate_id,
                "trajectoryCount": len(run_frame),
                "aggregateSlope": float(fit.slope) if fit else None,
                "aggregateSlopeP": float(fit.pvalue) if fit else None,
                "positive3SigmaRunFraction": float(
                    np.mean(run_frame["positive3SigmaCount"] > 0)
                ),
                "negative3SigmaRunFraction": float(
                    np.mean(run_frame["negative3SigmaCount"] > 0)
                ),
                "robustMadRunFraction": float(
                    np.mean(run_frame["robustMadExcursionCount"] > 0)
                ),
                "rawLjungBoxSignificantFraction": float(
                    np.mean(run_frame["rawLjungBoxP"] < 0.05)
                ),
                "differencedLjungBoxSignificantFraction": float(
                    np.mean(run_frame["differencedLjungBoxP"] < 0.05)
                ),
                "punctuatedDescriptiveGatePassed": bool(
                    np.mean(run_frame["positive3SigmaCount"] > 0) >= 0.5
                    and np.mean(run_frame["robustMadExcursionCount"] > 0) >= 0.5
                ),
                "weakAggregateTrendDescriptiveGatePassed": bool(
                    fit and fit.pvalue > 0.05
                ),
            }
        )
    return pd.DataFrame(temporal_rows), pd.DataFrame(spike_rows)


def paired_results(details: pd.DataFrame) -> pd.DataFrame:
    primary = details[details["labelId"] == PRIMARY_LABEL_ID]
    left = primary[primary["candidateId"] == CANDIDATE_IDS[0]].set_index("matrixIndex")
    right = primary[primary["candidateId"] == CANDIDATE_IDS[1]].set_index("matrixIndex")
    rows = []
    for matrix_index in sorted(set(left.index) & set(right.index)):
        rows.append(
            {
                "matrixIndex": int(matrix_index),
                "candidate2Rho": left.loc[matrix_index, "rho"],
                "candidate3Rho": right.loc[matrix_index, "rho"],
                "rhoDifferenceCandidate2Minus3": (
                    left.loc[matrix_index, "rho"] - right.loc[matrix_index, "rho"]
                    if pd.notna(left.loc[matrix_index, "rho"])
                    and pd.notna(right.loc[matrix_index, "rho"])
                    else None
                ),
                "candidate2MeanDifference": left.loc[matrix_index, "meanDifference"],
                "candidate3MeanDifference": right.loc[matrix_index, "meanDifference"],
                "meanDifferenceCandidate2Minus3": (
                    left.loc[matrix_index, "meanDifference"]
                    - right.loc[matrix_index, "meanDifference"]
                    if pd.notna(left.loc[matrix_index, "meanDifference"])
                    and pd.notna(right.loc[matrix_index, "meanDifference"])
                    else None
                ),
                "pairingStatus": "PAIRED_SHARED_BETA_AND_INITIAL_STATE",
            }
        )
    return pd.DataFrame(rows)


def compute_statistics(full: pd.DataFrame, prefix: pd.DataFrame) -> dict[str, Any]:
    details, retrospective, inference, _ = compute_retrospective(full)
    prefix_details, prefix_results, prefix_inference = compute_prefix(prefix)
    circularity, circularity_details, exact_identity = compute_circularity(full)
    temporal, spikes = temporal_and_spikes(full)
    paired = paired_results(details)
    primary_rows = retrospective[retrospective["labelId"] == PRIMARY_LABEL_ID].copy()
    candidate_decisions = []
    for row in primary_rows.to_dict("records"):
        candidate_decisions.append(
            {
                "candidateId": row["candidateId"],
                "associationGatePassed": bool(row["associationGatePassed"]),
                "driftGatePassed": bool(row["driftGatePassed"]),
                "candidatePrimaryPassed": bool(row["candidatePrimaryPassed"]),
            }
        )
    classification = classify(
        candidate_decisions,
        exact_h_identity_passed=exact_identity,
        validation_passed=True,
    )
    decision = {
        "schema": "eidosoma.e01.s13y_decision.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "classification": classification,
        "outcomeClass": outcome_class(classification),
        "candidateDecisions": candidate_decisions,
        "crossCandidatePrimaryPassed": all(
            row["candidatePrimaryPassed"] for row in candidate_decisions
        ),
        "exactHLabelIdentityPassed": exact_identity,
        "unrestrictedIncrementalInformationBeyondExactH": 0.0
        if exact_identity
        else None,
        "retrospectiveOnly": True,
        "earlyWarningSupported": False,
        "predictionSupported": False,
        "interventionOrCausalControlSupported": False,
        "authorCodeIdentityClaimed": False,
        "recommendedNextAction": "Mandatory human review; do not begin evidence synthesis, E02, report-bundle generation, or any later step automatically.",
    }
    return {
        "retrospective_trajectory_results.parquet": details,
        "retrospective_results.csv": retrospective,
        "retrospective_inference.csv": inference,
        "historical_label_comparator.csv": retrospective[
            retrospective["labelId"] == HISTORICAL_LABEL_ID
        ].copy(),
        "circularity_control_results.csv": circularity,
        "circularity_trajectory_diagnostics.parquet": circularity_details,
        "prefix_trajectory_results.parquet": prefix_details,
        "prefix_results.csv": prefix_results,
        "prefix_inference.csv": prefix_inference,
        "temporal_results.csv": temporal,
        "spike_results.csv": spikes,
        "paired_candidate_results.csv": paired,
        "decision": decision,
    }


def result_hashes(results: dict[str, Any]) -> dict[str, str]:
    output = {}
    for name, value in results.items():
        if name == "decision":
            output[name] = hashlib.sha256(
                json.dumps(
                    jsonable(value), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        else:
            output[name] = frame_hash(value)
    return output


def make_figures(results: dict[str, Any], full: pd.DataFrame) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    retrospective = results["retrospective_trajectory_results.parquet"]
    primary = retrospective[retrospective["labelId"] == PRIMARY_LABEL_ID]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [
        primary[primary["candidateId"] == candidate]["rho"].dropna()
        for candidate in CANDIDATE_IDS
    ]
    ax.boxplot(data, tick_labels=["Candidate 2", "Candidate 3"], showfliers=True)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Per-trajectory Spearman rho")
    ax.set_title("Clean retrospective molecular H>0.9 association")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "retrospective_associations.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
    for row, candidate in enumerate(CANDIDATE_IDS):
        for column, matrix_index in enumerate((0, 1)):
            frame = full[
                (full["candidateId"] == candidate)
                & (full["matrixIndex"] == matrix_index)
            ].sort_values("selectedSequenceIndex")
            ax = axes[row, column]
            ax.plot(frame["selectedSequenceIndex"], frame["emergence"], linewidth=0.7)
            ax.set_title(f"{candidate[-2:]} / matrix {matrix_index}")
            ax.set_xlabel("C1 molecular-state index")
            ax.set_ylabel("PhiRL emergence")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "representative_trajectories.png", dpi=180)
    plt.close(fig)

    controls = results["circularity_control_results.csv"]
    exact = controls[
        controls["controlId"] == "EXACT_H_LABEL_DETERMINISM_AND_BASELINE_ASSOCIATIONS"
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(exact))
    ax.bar(
        x - 0.18,
        exact["medianEmergenceContinuousHSpearman"],
        width=0.36,
        label="Emergence vs H",
    )
    ax.bar(
        x + 0.18,
        exact["medianEmergenceNegativeL2ChangeSpearman"],
        width=0.36,
        label="Emergence vs -L2 change",
    )
    ax.set_xticks(x, ["Candidate 2", "Candidate 3"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.set_ylabel("Median trajectory Spearman rho")
    ax.set_title("Label-circularity baseline coordinates")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "circularity_controls.png", dpi=180)
    plt.close(fig)

    prefix = results["prefix_inference.csv"]
    prefix = prefix[
        (prefix["labelId"] == PRIMARY_LABEL_ID)
        & (prefix["alignment"] == "CURRENT_ENDPOINT")
    ]
    retro = results["retrospective_inference.csv"]
    retro = retro[retro["labelId"] == PRIMARY_LABEL_ID]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(2)
    ax.scatter(
        positions - 0.12,
        retro.set_index("candidateId").loc[list(CANDIDATE_IDS), "medianCorrelation"],
        label="Retrospective full",
        s=60,
    )
    ax.scatter(
        positions + 0.12,
        prefix.set_index("candidateId").loc[list(CANDIDATE_IDS), "medianCorrelation"],
        label="Past-only prefix",
        s=60,
    )
    ax.set_xticks(positions, ["Candidate 2", "Candidate 3"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.set_ylabel("Median trajectory rho")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "retrospective_vs_prefix.png", dpi=180)
    plt.close(fig)

    decision = results["decision"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axis("off")
    cell = [
        [
            row["candidateId"],
            str(row["associationGatePassed"]),
            str(row["driftGatePassed"]),
            str(row["candidatePrimaryPassed"]),
        ]
        for row in decision["candidateDecisions"]
    ]
    table = ax.table(
        cellText=cell,
        colLabels=["Candidate", "Association", "Drift", "Primary"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title(decision["classification"], wrap=True)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "final_decision_matrix.png", dpi=180)
    plt.close(fig)


def pairing_audit(manifest: pd.DataFrame) -> dict[str, Any]:
    groups = manifest.groupby("matrixIndex")
    shared = 0
    for _, group in groups:
        if (
            len(group) == 2
            and group["betaSha256"].nunique() == 1
            and group["initialStateSha256"].nunique() == 1
        ):
            shared += 1
    prior_hashes = set()
    prior_manifest = ARTIFACTS / "research_steps/S13/trajectory_manifest.parquet"
    if prior_manifest.is_file():
        prior = pd.read_parquet(prior_manifest)
        prior_hashes.update(prior["betaSha256"].astype(str))
        prior_hashes.update(prior["initialStateSha256"].astype(str))
    current_hashes = set(manifest["betaSha256"].astype(str)) | set(
        manifest["initialStateSha256"].astype(str)
    )
    overlap = sorted(prior_hashes & current_hashes)
    return {
        "schema": "eidosoma.e01.s13y_pairing_audit.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "matrixCount": int(manifest["matrixIndex"].nunique()),
        "trajectoryCount": len(manifest),
        "sharedIdentityPairCount": shared,
        "priorS13BetaOrInitialHashOverlapCount": len(overlap),
        "overlap": overlap,
        "passed": bool(shared == 100 and len(manifest) == 200 and not overlap),
    }


def post_seed_firewall(
    simulation_seeds: pd.DataFrame, source_seeds: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    combined = pd.concat([simulation_seeds, source_seeds], ignore_index=True)
    prior_materials: set[str] = set()
    for path in (ARTIFACTS / "research_steps").rglob("*seed*.parquet"):
        if STEP_ROOT in path.parents:
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception:  # noqa: BLE001, S112 - heterogeneous historical manifests.
            continue
        for column in ("seedMaterialSha256", "seed_material_sha256"):
            if column in frame:
                prior_materials.update(frame[column].dropna().astype(str))
    overlap = set(combined["seedMaterialSha256"].dropna().astype(str)) & prior_materials
    passed = bool(
        combined["streamId"].is_unique
        and not overlap
        and (combined["rootHex"] == ROOT_SEED_HEX).all()
    )
    return combined, {
        "schema": "eidosoma.e01.s13y_seed_firewall_post_execution.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "streamCount": len(combined),
        "uniqueStreamCount": int(combined["streamId"].nunique()),
        "materialCount": int(combined["seedMaterialSha256"].nunique()),
        "priorMaterialOverlapCount": len(overlap),
        "rootExact": bool((combined["rootHex"] == ROOT_SEED_HEX).all()),
        "passed": passed,
    }


def schema_validation(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    minimums = {
        "trajectory_manifest.parquet": 200,
        "simulation_summary.parquet": 200,
        "trajectory_replay_validation.parquet": 200,
        "label_values.parquet": 600,
        "label_fingerprints.parquet": 400,
        "preprocessing_diagnostics.parquet": 200,
        "full_source_values.parquet": 200,
        "prefix_endpoint_values.parquet": 20000,
        "partition_history.parquet": 200,
        "source_diagnostic_outputs.parquet": 200,
        "retrospective_trajectory_results.parquet": 480,
        "prefix_trajectory_results.parquet": 600,
    }
    rows = []
    for name, minimum in minimums.items():
        frame = frames[name]
        passed = len(frame) >= minimum and len(frame.columns) > 0
        rows.append(
            {
                "artifact": name,
                "rowCount": len(frame),
                "columnCount": len(frame.columns),
                "minimumRows": minimum,
                "passed": passed,
            }
        )
    expected_label_rows = 3 * int(
        frames["preprocessing_diagnostics.parquet"]["observationCount"].sum()
    )
    relationships = [
        {
            "check": "three_label_rows_per_selected_observation",
            "expected": expected_label_rows,
            "observed": len(frames["label_values.parquet"]),
            "passed": len(frames["label_values.parquet"]) == expected_label_rows,
        },
        {
            "check": "one_hundred_prefix_status_rows_per_trajectory",
            "expected": 20000,
            "observed": len(frames["prefix_endpoint_values.parquet"]),
            "passed": len(frames["prefix_endpoint_values.parquet"]) == 20000,
        },
    ]
    return {
        "schema": "eidosoma.e01.s13y_schema_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "tables": rows,
        "relationships": relationships,
        "passed": all(row["passed"] for row in rows)
        and all(row["passed"] for row in relationships),
    }


def build_report(
    decision: dict[str, Any],
    retrospective: pd.DataFrame,
    prefix: pd.DataFrame,
    circularity: pd.DataFrame,
    temporal: pd.DataFrame,
    validation: dict[str, Any],
    runtime: dict[str, Any],
    artifacts_written: list[str],
) -> str:
    primary = retrospective[retrospective["labelId"] == PRIMARY_LABEL_ID].set_index(
        "candidateId"
    )
    prefix_primary = prefix[
        (prefix["labelId"] == PRIMARY_LABEL_ID)
        & (prefix["alignment"] == "CURRENT_ENDPOINT")
    ].set_index("candidateId")
    lines = [
        "# S13Y Clean Directional Confirmation: Full Results",
        "",
        "## Concise top summary",
        "",
        f"- **Research step ID:** `{VERSION}` (actual step `S13Y`).",
        "- **Completion status:** Complete; stopped at the mandatory S13Y human-review boundary.",
        f"- **Artifacts written:** {len(artifacts_written)} retained paths under `$ARTIFACTS_DIR/research_steps/S13Y/`, including 200 trajectory identities, all status-bearing source/prefix rows, circularity controls, machine-readable inference, figures, manifests, and this report.",
        f"- **Validation result:** {validation['validationResult']}.",
        f"- **Outcome classification:** `{decision['classification']}` ({decision['outcomeClass']}).",
        "- **Caveats or blockers:** S13X selected this branch adaptively; full fits are future-fitted and retrospective; the primary binary label is exactly determined by H; prior strict, prefix, intervention, sensitivity, and held-out non-support remain unchanged; no author-code identity is established.",
        "- **Recommended next action:** Mandatory human review. Do not begin evidence synthesis, E02, report-bundle generation, S14–S18, prediction, or intervention work automatically.",
        "",
        "## Lay summary",
        "",
        "We generated a completely new set of 100 catalytic matrices and ran each through both previously confirmed simulator time-base candidates. We then tested only the one pattern found in S13X: whether the public PhiRL source-emergence number is higher when consecutive molecular compositions are very similar (`H>0.9`). The completed-trajectory calculation is descriptive because its partition and Gaussian model use the finished run. We also repeated the calculation using only past prefixes. Finally, we checked the central circularity directly: the binary target is literally constructed by thresholding H, so exact H predicts the label perfectly and no other quantity can add unrestricted information about that label conditional on exact H.",
        "",
        "## Frozen question and interpretation boundary",
        "",
        "The frozen hypothesis was the exact S13X lead `S13X-P-684e66c4cffe914c`: PhiRL regularized source-defined emergence, level transform, molecular adjacent-incoming `H>0.9`, and same-state alignment. Candidate-specific evidence is primary and both candidates had to pass the same association and replicator-minus-drift gates. `H>0.97` was descriptive only. No search, intervention, prediction, MLP, estimator change, or extra simulator was allowed.",
        "",
        "Every completed-fit value is labeled `RETROSPECTIVE_FULL_TRAJECTORY_LOCAL`. It cannot support early warning, intervention, prediction, or causal control. S13Y is a clean test of a post-selection hypothesis, not a test of the unavailable author implementation.",
        "",
        "## Inputs and provenance",
        "",
        "- Original paper and its reported directional fingerprints were refreshed before design freeze.",
        "- Simulator candidates were unchanged S12FR candidates 2 and 3, with the exact exposures, daughter rules, trimmed-new-entrant semantics, and C1 clock recorded in the preregistration.",
        "- PhiRL was pinned to commit `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`; the safe lattice hash was `74ecca37f04201088d76a9e8ede7efe04bafebecff85a4882a44f03afbd23aa1`.",
        "- The S13Y seed root was new and domain separated. Catalytic matrices and initial states were shared across candidates; dynamics streams were candidate-specific.",
        "",
        "## Detailed methods",
        "",
        "Exactly 100 new beta matrices were sampled as `exp(-4 + 4Z)` and 100 matched mass-40 distinct-singleton initial states were generated. Each pair was simulated under both candidate contracts for 100 fissions. Counts received additive-0.5 closure, full CLR, and original component 100 removal. PhiRL source emergence was synergy plus the two downward-causation atoms. Full fits used the complete trajectory. Prefix fits independently reran the same source pipeline from the start through each post-fission endpoint after 256 C1 transitions and retained only the final local value.",
        "",
        "For every trajectory we calculated Spearman association and the mean emergence difference between label-positive and label-negative molecular states. Candidate inference used 4,096 trajectory bootstraps and 4,096 nonzero circular rotations, which preserve each cyclic binary label sequence and its episode durations. The clean gate required at least 80 defined trajectories, at least 65% positive correlations, positive median rho with a positive bootstrap lower bound and shift p<=0.05; the drift gate required at least 50% positive differences plus the analogous median/bootstrap/null conditions.",
        "",
        "Circularity controls recorded continuous incoming H, ordinary Euclidean L2 composition change, exact `Y=I(H>0.9)` identity, and a fixed smooth-H/L2 within-trajectory emergence regression. The exact deterministic identity is authoritative: `H(Y|H)=0`, hence unrestricted `I(E;Y|H)=0`. The smooth regression is only a model-dependent threshold-discontinuity diagnostic.",
        "",
        "## Commands",
        "",
        "```bash",
        "python scripts/e01/freeze_s13y_preregistration.py",
        "python scripts/e01/run_s13y_clean_directional_confirmation.py --stage benchmark",
        "python scripts/e01/run_s13y_clean_directional_confirmation.py --stage full --workers 6",
        "pytest -q tests/e01/test_s13y_clean_directional_confirmation.py",
        "```",
        "",
        "## Primary retrospective results",
        "",
        "| Candidate | Defined | Positive | Median rho | 95% bootstrap | shift p | Higher during replication | Median difference | Candidate gate |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for candidate in CANDIDATE_IDS:
        row = primary.loc[candidate]
        lines.append(
            f"| {candidate} | {int(row['definedCorrelationCount'])} | {int(row['positiveCorrelationCount'])} ({row['positiveCorrelationFraction']:.3f}) | {row['medianCorrelation']:.6f} | [{row['bootstrapLower95']:.6f}, {row['bootstrapUpper95']:.6f}] | {row['circularShiftPositiveP']:.6f} | {int(row['higherDuringReplicationCount'])} ({row['higherDuringReplicationFraction']:.3f}) | {row['medianMeanDifference']:.6f} | {bool(row['candidatePrimaryPassed'])} |"
        )
    lines.extend(
        [
            "",
            f"The all-candidate classification is `{decision['classification']}`. A favorable candidate could not rescue the other candidate.",
            "",
            "## Label-circularity controls",
            "",
        ]
    )
    exact_rows = circularity[
        circularity["controlId"]
        == "EXACT_H_LABEL_DETERMINISM_AND_BASELINE_ASSOCIATIONS"
    ]
    for row in exact_rows.itertuples(index=False):
        lines.append(
            f"- {row.candidateId}: {int(row.mismatchCount)}/{int(row.rowCount)} label-identity mismatches; baseline H classification accuracy {row.baselineHLabelClassificationAccuracy:.3f}; median emergence–H rho {row.medianEmergenceContinuousHSpearman:.6f}; median emergence–negative-L2-change rho {row.medianEmergenceNegativeL2ChangeSpearman:.6f}."
        )
    lines.extend(
        [
            "",
            "Because the label is exactly a thresholded H value, the binary target has zero conditional entropy given exact H. This is a structural result, not a finite-sample failure: PhiRL cannot add unrestricted information about that same binary target after exact H is known. Any positive primary association is therefore bounded to the preregistered label-coupled retrospective interpretation.",
            "",
            "## Past-only prefix falsification",
            "",
            "| Candidate | Defined | Positive | Median rho | 95% bootstrap | shift p | Prefix gate |",
            "| --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for candidate in CANDIDATE_IDS:
        row = prefix_primary.loc[candidate]
        lines.append(
            f"| {candidate} | {int(row['definedCorrelationCount'])} | {int(row['positiveCorrelationCount'])} ({row['positiveCorrelationFraction']:.3f}) | {row['medianCorrelation']:.6f} | [{row['bootstrapLower95']:.6f}, {row['bootstrapUpper95']:.6f}] | {row['circularShiftPositiveP']:.6f} | {bool(row['prefixGatePassed'])} |"
        )
    lines.extend(
        [
            "",
            "Prefix results are a secondary falsification and do not change the retrospective-only status of full-fit values. No trajectory was pooled to cross the 256-transition boundary.",
            "",
            "## Temporal and spike descriptions",
            "",
            temporal.to_markdown(index=False),
            "",
            "These are descriptive paper-resemblance checks and are not permitted to rescue a failed primary association.",
            "",
            "## Validation",
            "",
            f"- {validation['validationResult']}.",
            f"- 200/200 trajectories completed 100 fissions and replayed exactly; pairing, source replay, prefix replay, suffix invariance, component identity, finite coverage, schema, seed firewall, immutability, statistics replay, runtime, storage, and artifact gates passed: {validation['allValidationGatesPassed']}.",
            f"- Cumulative CPU envelope after S13Y: {runtime['observedCumulativeE01CpuEnvelopeHours']:.3f}/250 hours; GPU envelope: {runtime['cumulativeGpuEnvelopeHours']:.3f}/80 hours.",
            "- CPU float64 was authoritative; six workers and one numerical-library thread per worker were used; the L4 was not used.",
            "",
            "## Caveats, failed assumptions, and limitations",
            "",
            "- The hypothesis was generated adaptively in S13X, so this is clean post-selection confirmation rather than independent discovery.",
            "- Exact H-label determinism is a circularity constraint. A thresholded label cannot demonstrate incremental information beyond its own defining coordinate.",
            "- PhiRL is a later public source implementation, not the unavailable GARD-paper code.",
            "- Full fits estimate partitions and Gaussian distributions from completed trajectories and are future-dependent.",
            "- The fixed-window and early-time claims remain unavailable; prefix values begin only after 256 locked-clock transitions.",
            "- S12 strict estimates, S13RRR held-out results, S13X prefix and intervention results, and all historical failures remain unchanged and must coexist with this result.",
            "",
            "## Artifact and software provenance",
            "",
            f"Design commit: `{git('rev-parse', 'HEAD')}` on `eidosoma/groups/42`. Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}, SciPy {scipy.__version__}. Raw trajectory and resumable source caches remain under `/cache/e01_s13y_v1`; compact evidence is under `$ARTIFACTS_DIR/research_steps/S13Y/`.",
            "",
            "## Recommended next action",
            "",
            "Return for mandatory human review. Do not begin evidence synthesis, E02, report-bundle generation, S14–S18, another scale-up, intervention, prediction, or estimator work automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def benchmark() -> int:
    lock = validate_method_lock()
    prior = validate_prior()
    compute = json.loads((STEP_ROOT / "compute_gate.json").read_text())
    if not lock["passed"] or not prior["passed"] or not compute["passed"]:
        raise RuntimeError("S13Y pre-benchmark validation failed")
    start = time.perf_counter()
    records = execute_simulations([0], workers=1)
    manifest = records[0]["trajectories"]
    source = execute_source(manifest, workers=1)
    source_cpu = sum(float(row["cpuSeconds"]) for row in source)
    simulation_cpu = sum(float(row["cpuSeconds"]) for row in records)
    projected = 1.25 * (source_cpu + simulation_cpu) * 100 / 3600.0 + 2.0
    prior_cpu = float(
        json.loads((STEP_ROOT / "compute_ledger.json").read_text())[
            "priorCpuEnvelopeHours"
        ]
    )
    passed = projected <= 20.0 and prior_cpu + projected <= 250.0
    payload = {
        "schema": "eidosoma.e01.s13y_runtime_benchmark.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "matrixIndex": 0,
        "retainedInFinal200": True,
        "trajectoryCount": 2,
        "sourceTaskCount": 2,
        "simulationCpuSeconds": simulation_cpu,
        "sourceCpuSeconds": source_cpu,
        "observedWallSeconds": time.perf_counter() - start,
        "projectionFormula": "1.25*(matrix0_two_candidate_simulation_plus_PhiRL_source_CPU)*100+2_CPU_hours",
        "projectedS13YCpuHours": projected,
        "projectedCumulativeE01CpuHours": prior_cpu + projected,
        "s13yProspectiveEnvelopeHours": 20.0,
        "passed": passed,
    }
    write_json(STEP_ROOT / "runtime_benchmark.json", payload)
    if not passed:
        raise RuntimeError(
            f"S13Y benchmark projection exceeded frozen ceiling: {payload}"
        )
    print(
        json.dumps(
            {
                "stage": "benchmark_complete",
                "projectedS13YCpuHours": projected,
                "passed": passed,
            },
            sort_keys=True,
        )
    )
    return 0


def full_run(workers: int) -> int:
    started = time.perf_counter()
    lock = validate_method_lock()
    prior_start = validate_prior()
    benchmark_payload = json.loads((STEP_ROOT / "runtime_benchmark.json").read_text())
    if (
        not lock["passed"]
        or not prior_start["passed"]
        or not benchmark_payload["passed"]
    ):
        raise RuntimeError("S13Y full-run preflight failed")
    missing_matrices = [
        index
        for index in range(100)
        if any(
            not (RAW_ROOT / candidate / f"M{index:03d}.pickle").is_file()
            for candidate in CANDIDATE_IDS
        )
    ]
    simulation_records = (
        execute_simulations(missing_matrices, workers=workers)
        if missing_matrices
        else []
    )
    manifest, summaries, replay, simulation_seeds = load_simulation_records()
    if (
        len(manifest) != 200
        or len(replay) != 200
        or not replay["passed"].astype(bool).all()
    ):
        raise RuntimeError("S13Y simulation cardinality or replay gate failed")
    pairing = pairing_audit(manifest)
    if not pairing["passed"]:
        raise RuntimeError(f"S13Y pairing/firewall gate failed: {pairing}")
    write_parquet(STEP_ROOT / "trajectory_manifest.parquet", manifest)
    write_parquet(STEP_ROOT / "simulation_summary.parquet", summaries)
    write_parquet(STEP_ROOT / "trajectory_replay_validation.parquet", replay)
    write_json(STEP_ROOT / "pairing_audit.json", pairing)

    tasks = manifest.to_dict("records")
    missing_tasks = [
        task
        for task in tasks
        if not (
            source_task_root(task["candidateId"], int(task["matrixIndex"]))
            / "completion.json"
        ).is_file()
    ]
    if missing_tasks:
        execute_source(missing_tasks, workers=workers)
    records = [
        json.loads(
            (
                source_task_root(str(task["candidateId"]), int(task["matrixIndex"]))
                / "completion.json"
            ).read_text()
        )
        for task in tasks
    ]
    frames = collate_source(records)
    full = frames["full_source_values.parquet"]
    prefix = frames["prefix_endpoint_values.parquet"]
    diagnostics = frames["source_diagnostic_outputs.parquet"]
    suffix = frames["replay_suffix_validation.parquet"]
    eligible_prefix = prefix[(prefix["priorLockedClockTransitions"] >= 256)]
    full_coverage = (
        full.assign(
            finite=np.isfinite(pd.to_numeric(full["emergence"], errors="coerce"))
        )
        .groupby("candidateId")["finite"]
        .mean()
    )
    prefix_coverage = (
        eligible_prefix.assign(
            finite=np.isfinite(
                pd.to_numeric(eligible_prefix["emergence"], errors="coerce")
            )
        )
        .groupby("candidateId")["finite"]
        .mean()
    )
    executed_suffix = suffix[suffix["sentinel"] != "non_sentinel"]
    source_gate = bool(
        len(frames["worker_failures"]) == 0
        and all(
            row["fullReplayAllPassed"]
            and row["prefixReplayAllPassed"]
            and row["futureSuffixAllPassed"]
            and row["failureRows"] == 0
            for row in records
        )
        and (full_coverage >= 0.95).all()
        and (prefix_coverage >= 0.80).all()
        and suffix["structuralExact"].astype(bool).all()
        and executed_suffix["resultExact"].eq(True).all()
        and diagnostics["componentIdentityMaxAbsError"].fillna(0).max() <= 1e-12
    )
    if not source_gate:
        raise RuntimeError("S13Y source/replay/coverage/suffix/component gate failed")

    source_seeds = frames["source_seed_manifest.parquet"]
    seeds, seed_firewall = post_seed_firewall(simulation_seeds, source_seeds)
    write_parquet(STEP_ROOT / "seed_manifest.parquet", seeds)
    write_json(STEP_ROOT / "seed_firewall.json", seed_firewall)
    if not seed_firewall["passed"]:
        raise RuntimeError("S13Y post-execution seed firewall failed")

    first_results = compute_statistics(full, prefix)
    second_results = compute_statistics(full, prefix)
    first_hashes, second_hashes = (
        result_hashes(first_results),
        result_hashes(second_results),
    )
    statistics_replay = {
        "schema": "eidosoma.e01.s13y_statistics_replay_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "firstHashes": first_hashes,
        "secondHashes": second_hashes,
        "mismatches": {
            key: [first_hashes[key], second_hashes[key]]
            for key in first_hashes
            if first_hashes[key] != second_hashes[key]
        },
        "passed": first_hashes == second_hashes,
    }
    write_json(STEP_ROOT / "statistics_replay_validation.json", statistics_replay)
    if not statistics_replay["passed"]:
        raise RuntimeError("S13Y exact statistics replay failed")
    for name, value in first_results.items():
        if name == "decision":
            write_json(STEP_ROOT / "decision.json", value)
        elif name.endswith(".parquet"):
            write_parquet(STEP_ROOT / name, value)
        else:
            write_csv(STEP_ROOT / name, value)
    make_figures(first_results, full)

    frame_registry = {
        "trajectory_manifest.parquet": manifest,
        "simulation_summary.parquet": summaries,
        "trajectory_replay_validation.parquet": replay,
        **{
            key: value
            for key, value in frames.items()
            if key != "worker_failures" and key != "source_seed_manifest.parquet"
        },
        **{key: value for key, value in first_results.items() if key != "decision"},
    }
    schemas = schema_validation(frame_registry)
    write_json(STEP_ROOT / "schema_validation.json", schemas)
    if not schemas["passed"]:
        raise RuntimeError("S13Y final schema validation failed")

    prior_end = validate_prior()
    write_json(STEP_ROOT / "immutable_prior_validation.json", prior_end)
    if not prior_end["passed"]:
        raise RuntimeError("S13Y prior artifact immutability failed")
    simulation_cpu = float(benchmark_payload["simulationCpuSeconds"]) + sum(
        float(row.get("cpuSeconds", 0)) for row in simulation_records
    )
    source_cpu = sum(float(row["cpuSeconds"]) for row in records)
    statistics_cpu_upper = (time.perf_counter() - started) * 1.0
    observed_step_cpu = (
        simulation_cpu / 3600 + source_cpu / 3600 + statistics_cpu_upper / 3600
    )
    prior_cpu = float(
        json.loads((STEP_ROOT / "compute_ledger.json").read_text())[
            "priorCpuEnvelopeHours"
        ]
    )
    runtime = {
        "schema": "eidosoma.e01.s13y_runtime_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "wallSeconds": time.perf_counter() - started,
        "simulationWorkerCpuSecondsRecordedThisInvocation": simulation_cpu,
        "sourceWorkerCpuSeconds": source_cpu,
        "orchestrationStatisticsCpuEnvelopeSeconds": statistics_cpu_upper,
        "observedS13YCpuEnvelopeHours": observed_step_cpu,
        "observedCumulativeE01CpuEnvelopeHours": prior_cpu + observed_step_cpu,
        "cumulativeCpuCeilingHours": 250.0,
        "cumulativeGpuEnvelopeHours": 2.0,
        "cumulativeGpuCeilingHours": 80.0,
        "gpuUsed": False,
        "cpuFloat64Authoritative": True,
        "workers": workers,
        "threadEnvironment": {
            name: os.environ[name]
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "passed": bool(prior_cpu + observed_step_cpu <= 250.0),
    }
    write_json(STEP_ROOT / "runtime_manifest.json", runtime)
    storage = {
        "schema": "eidosoma.e01.s13y_storage_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "artifactBytesBeforeReportManifest": recursive_bytes(STEP_ROOT),
        "cacheBytes": recursive_bytes(CACHE_ROOT),
        "artifactCeilingBytes": 30 * 1024**3,
        "passed": recursive_bytes(STEP_ROOT) <= 30 * 1024**3,
    }
    write_json(STEP_ROOT / "storage_validation.json", storage)
    failures = frames["worker_failures"]
    if failures.empty:
        failures = pd.DataFrame(
            columns=[
                "failureId",
                "stage",
                "candidateId",
                "matrixIndex",
                "severity",
                "status",
                "reason",
                "gateImpact",
                "repairAttempted",
            ]
        )
    write_csv(STEP_ROOT / "failure_ledger.csv", failures)
    regeneration = {
        "schema": "eidosoma.e01.s13y_regeneration_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "trajectoryReplayCount": len(replay),
        "trajectoryReplayPassCount": int(replay["passed"].sum()),
        "fullSourceReplayTaskCount": sum(
            bool(row["fullReplayAllPassed"]) for row in records
        ),
        "prefixSourceReplayTaskCount": sum(
            bool(row["prefixReplayAllPassed"]) for row in records
        ),
        "suffixTaskPassCount": sum(
            bool(row["futureSuffixAllPassed"]) for row in records
        ),
        "executedSuffixSentinelCount": len(executed_suffix),
        "executedSuffixSentinelPassCount": int(
            executed_suffix["resultExact"].eq(True).sum()
        ),
        "passed": True,
    }
    write_json(STEP_ROOT / "regeneration_validation.json", regeneration)
    provenance = {
        "schema": "eidosoma.e01.s13y_provenance_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "designCommit": lock["designState"]["head"],
        "branch": "eidosoma/groups/42",
        "sourceCommits": {
            "historicalGARD": "86dff6320d5ae91b4e831471079ff46749b14df9",
            "PhiRL": "a6d1d0d18c7551302724b7158c6ccdc4d3a33373",
            "IIGR_contextOnly": "7c1c22fe39f539d4a453135476f1f0dd5a6b45f7",
        },
        "safeLatticeSha256": sha256_file(SAFE_LATTICE),
        "s13xPipelineId": S13X_PIPELINE_ID,
        "rootSeedId": "E01-S13Y-CLEAN-CONFIRMATION-ROOT-v1.0.0",
        "rawCacheRoot": str(RAW_ROOT),
        "sourceCacheRoot": str(SOURCE_ROOT),
        "matrixCount": 100,
        "trajectoryCount": 200,
        "priorArtifactsMutable": False,
        "passed": True,
    }
    write_json(STEP_ROOT / "provenance_manifest.json", provenance)
    validation = {
        "schema": "eidosoma.e01.s13y_execution_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "completeTrajectoryCount": int((manifest["completedFissions"] == 100).sum()),
        "trajectoryReplayPassCount": int(replay["passed"].sum()),
        "sharedPairCount": pairing["sharedIdentityPairCount"],
        "sourceTaskCount": len(records),
        "sourceTaskPassCount": sum(
            bool(
                row["fullReplayAllPassed"]
                and row["prefixReplayAllPassed"]
                and row["futureSuffixAllPassed"]
                and row["failureRows"] == 0
            )
            for row in records
        ),
        "minimumFullCoverage": float(full_coverage.min()),
        "minimumPrefixCoverage": float(prefix_coverage.min()),
        "suffixStructuralCount": len(suffix),
        "suffixStructuralPassCount": int(suffix["structuralExact"].astype(bool).sum()),
        "executedSuffixSentinelCount": len(executed_suffix),
        "executedSuffixSentinelPassCount": int(
            executed_suffix["resultExact"].eq(True).sum()
        ),
        "maximumComponentIdentityError": float(
            diagnostics["componentIdentityMaxAbsError"].fillna(0).max()
        ),
        "primaryLabelIdentityPassed": bool(
            first_results["decision"]["exactHLabelIdentityPassed"]
        ),
        "statisticsReplayPassed": statistics_replay["passed"],
        "seedFirewallPassed": seed_firewall["passed"],
        "priorImmutabilityPassed": prior_end["passed"],
        "schemaPassed": schemas["passed"],
        "runtimePassed": runtime["passed"],
        "storagePassed": storage["passed"],
        "allValidationGatesPassed": bool(
            source_gate
            and statistics_replay["passed"]
            and seed_firewall["passed"]
            and prior_end["passed"]
            and schemas["passed"]
            and runtime["passed"]
            and storage["passed"]
            and pairing["passed"]
        ),
        "validationResult": "PASS_ALL_FROZEN_COMPUTE_SIMULATION_PAIRING_SOURCE_REPLAY_SUFFIX_LABEL_CIRCULARITY_STATISTICS_SCHEMA_PROVENANCE_IMMUTABILITY_RUNTIME_STORAGE_AND_HASH_GATES",
    }
    write_json(STEP_ROOT / "execution_validation.json", validation)
    if not validation["allValidationGatesPassed"]:
        raise RuntimeError("S13Y aggregate validation failed")

    required = yaml.safe_load(CONFIG_PATH.read_text())["artifacts"]["required"]
    report = build_report(
        first_results["decision"],
        first_results["retrospective_inference.csv"],
        first_results["prefix_inference.csv"],
        first_results["circularity_control_results.csv"],
        first_results["temporal_results.csv"],
        validation,
        runtime,
        required,
    )
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    missing_required = [
        path
        for path in required
        if not (STEP_ROOT / path).is_file()
        and path not in {"artifact_manifest.json", "status.json"}
    ]
    if missing_required:
        raise RuntimeError(
            f"S13Y required artifacts missing before manifest: {missing_required}"
        )
    status = {
        "researchStepId": RESEARCH_STEP_ID,
        "stepNumber": "S13Y",
        "success": True,
        "status": "COMPLETED_AT_MANDATORY_HUMAN_REVIEW_BOUNDARY",
        "artifactsWritten": sorted(set(required)),
        "validationResult": validation["validationResult"],
        "outcomeClassification": first_results["decision"]["classification"],
        "outcomeClass": first_results["decision"]["outcomeClass"],
        "caveatsOrBlockers": [
            "The tested branch was selected adaptively in S13X.",
            "The molecular binary label is exactly determined by incoming H.",
            "Completed-fit source values are retrospective and future-fitted.",
            "Prior strict, prospective, intervention, sensitivity, and held-out findings remain unchanged.",
        ],
        "recommendedNextAction": "Mandatory human review; keep evidence synthesis, E02, report-bundle generation, and all later work blocked.",
    }
    write_json(STEP_ROOT / "status.json", status)
    manifest_rows = []
    for path in sorted(STEP_ROOT.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest_rows.append(
                {
                    "path": str(path.relative_to(STEP_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    artifact_manifest = {
        "schema": "eidosoma.e01.s13y_artifact_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "artifactCountExcludingSelf": len(manifest_rows),
        "totalBytesExcludingSelf": sum(row["bytes"] for row in manifest_rows),
        "artifacts": manifest_rows,
        "requiredArtifactCount": len(required),
        "missingRequired": [
            path
            for path in required
            if path != "artifact_manifest.json" and not (STEP_ROOT / path).is_file()
        ],
        "passed": all(
            (STEP_ROOT / path).is_file()
            for path in required
            if path != "artifact_manifest.json"
        ),
    }
    write_json(STEP_ROOT / "artifact_manifest.json", artifact_manifest)
    if not artifact_manifest["passed"]:
        raise RuntimeError("S13Y artifact completeness failed")
    print(
        json.dumps(
            {
                "stage": "s13y_complete",
                "classification": first_results["decision"]["classification"],
                "validationPassed": True,
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("benchmark", "full"), required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 6:
        raise ValueError("S13Y workers must be in 1..6")
    if args.stage == "benchmark":
        return benchmark()
    return full_run(args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
