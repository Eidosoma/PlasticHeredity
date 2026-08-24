"""Parallel pair campaigns and unchanged scientific replay for S12FR."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import pandas as pd

from e01_latent_timebase.core import (
    ExposureDefinition,
    SeedIdentity,
    SimulationDefinition,
    simulate_trajectory,
    trajectory_summary,
)

from .audit import (
    compare_rng_manifests,
    sha256_file,
    simulate_audited,
    write_trace_payload,
)
from .comparator import compare_seed_tuples, compare_trajectories

WORKERS = 6


def definition_from_payload(payload: dict[str, Any]) -> SimulationDefinition:
    return SimulationDefinition(
        daughter_rule=payload["daughterRule"],
        overshoot_rule=payload["overshootRule"],
        exposure=ExposureDefinition(
            family=payload["family"],
            h=payload.get("h"),
            c=payload.get("c"),
            h_max=payload.get("hMax"),
        ),
    )


def seed_rows(
    seeds: tuple[SeedIdentity, ...], campaign: str, pair_id: str, particle_id: str
) -> list[dict[str, Any]]:
    return [
        {
            "campaign": campaign,
            "pairId": pair_id,
            "particleId": particle_id,
            "phase": seed.phase,
            "rootSha256": seed.root_sha256,
            "purpose": seed.purpose,
            "matrixIndex": seed.matrix_index,
            "configurationId": seed.configuration_id,
            "extra": list(seed.extra),
            "derivedSeed": str(seed.derived_seed),
            "seedMaterialSha256": seed.seed_material_sha256,
            "bitGenerator": "PCG64DXSM",
        }
        for seed in seeds
    ]


def _pair_task(payload: dict[str, Any]) -> dict[str, Any]:
    definition = definition_from_payload(payload)
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    common = {
        "phase": payload["phase"],
        "root_hex": payload["root"],
        "matrix_index": int(payload["matrixIndex"]),
        "definition": definition,
        "stream_identity": payload["streamIdentity"],
    }
    left = simulate_audited(**common)
    right = simulate_audited(**common)
    uninstrumented, uninstrumented_seeds = simulate_trajectory(**common)

    trajectory_comparison = compare_trajectories(left.trajectory, right.trajectory)
    parity_comparison = compare_trajectories(left.trajectory, uninstrumented)
    seed_passed, seed_differences = compare_seed_tuples(left.seeds, right.seeds)
    parity_seed_passed, parity_seed_differences = compare_seed_tuples(
        left.seeds, uninstrumented_seeds
    )
    rng_passed, rng_differences = compare_rng_manifests(
        left.rng_manifest, right.rng_manifest
    )

    all_differences = []
    for comparison_scope, rows in (
        ("LEFT_VS_REPLAY", trajectory_comparison.differences),
        ("LEFT_VS_REPLAY_SEEDS", seed_differences),
        ("LEFT_VS_REPLAY_RNG", rng_differences),
        ("INSTRUMENTED_VS_UNINSTRUMENTED", parity_comparison.differences),
        ("INSTRUMENTED_VS_UNINSTRUMENTED_SEEDS", parity_seed_differences),
    ):
        for row in rows:
            values = row.to_row()
            values["comparisonScope"] = comparison_scope
            all_differences.append(values)

    instrumentation_parity = bool(
        parity_comparison.repaired_comparator_passed
        and parity_seed_passed
        and left.trajectory.trajectory_sha256 == uninstrumented.trajectory_sha256
    )
    pair_gate = bool(
        trajectory_comparison.repaired_comparator_passed
        and seed_passed
        and rng_passed
        and instrumentation_parity
        and left.trace_sha256 == right.trace_sha256
    )
    old_failure_explained = bool(
        trajectory_comparison.old_comparator_passed
        or (
            pair_gate
            and trajectory_comparison.permitted_paired_nan_count > 0
            and all(row.permitted for row in trajectory_comparison.differences)
        )
    )

    trace_path = Path(payload["tracePath"])
    trace_record = write_trace_payload(trace_path, left)
    replay_trace_record: dict[str, Any] | None = None
    if not pair_gate:
        replay_path = trace_path.with_name(trace_path.stem + "__REPLAY.npz")
        replay_trace_record = write_trace_payload(replay_path, right)

    rng_call_count = sum(int(row["callCount"]) for row in left.rng_manifest)
    rng_finite_argument_count = sum(
        int(call["finite_float_argument_count"])
        for stream in left.rng_manifest
        for call in stream["calls"]
    )
    rng_nonfinite_argument_count = sum(
        int(call["nonfinite_float_argument_count"])
        for stream in left.rng_manifest
        for call in stream["calls"]
    )
    row = {
        "campaign": payload["campaign"],
        "pairId": payload["pairId"],
        "particleId": payload["particleId"],
        "matrixIndex": int(payload["matrixIndex"]),
        "phase": payload["phase"],
        "family": payload["family"],
        "daughterRule": payload["daughterRule"],
        "overshootRule": payload["overshootRule"],
        "clockId": payload["clockId"],
        "h": payload.get("h"),
        "c": payload.get("c"),
        "hMax": payload.get("hMax"),
        "oldComparatorPassed": trajectory_comparison.old_comparator_passed,
        "repairedComparatorPassed": trajectory_comparison.repaired_comparator_passed,
        "seedIdentityPassed": seed_passed,
        "rngConsumptionPassed": rng_passed,
        "instrumentationParityPassed": instrumentation_parity,
        "traceDigestPassed": left.trace_sha256 == right.trace_sha256,
        "pairGatePassed": pair_gate,
        "oldFailureFullyExplained": old_failure_explained,
        "betaSha256Left": left.trajectory.beta_sha256,
        "betaSha256Right": right.trajectory.beta_sha256,
        "initialStateSha256Left": left.trajectory.initial_state_sha256,
        "initialStateSha256Right": right.trajectory.initial_state_sha256,
        "trajectorySha256Left": left.trajectory.trajectory_sha256,
        "trajectorySha256Right": right.trajectory.trajectory_sha256,
        "trajectorySha256Uninstrumented": uninstrumented.trajectory_sha256,
        "traceSha256Left": left.trace_sha256,
        "traceSha256Right": right.trace_sha256,
        "discreteDivergenceCount": trajectory_comparison.discrete_divergence_count,
        "finiteNumericDivergenceCount": trajectory_comparison.finite_numeric_divergence_count,
        "permittedPairedNanCount": trajectory_comparison.permitted_paired_nan_count,
        "forbiddenNonfiniteDifferenceCount": trajectory_comparison.forbidden_nonfinite_difference_count,
        "rngDivergenceCount": len(rng_differences),
        "instrumentationParityDifferenceCount": len(parity_comparison.differences)
        + len(parity_seed_differences),
        "rngStreamCount": len(left.rng_manifest),
        "rngCallCount": rng_call_count,
        "rngFiniteFloatArgumentCount": rng_finite_argument_count,
        "rngNonfiniteFloatArgumentCount": rng_nonfinite_argument_count,
        "completedFissions": left.trajectory.completed_fissions,
        "totalBatchUpdates": left.trajectory.total_batch_updates,
        "terminalStatus": left.trajectory.terminal_status,
        "tracePayloadPath": trace_record["path"],
        "tracePayloadSha256": trace_record["sha256"],
        "tracePayloadSizeBytes": trace_record["sizeBytes"],
        "replayTracePayloadPath": None if replay_trace_record is None else replay_trace_record["path"],
        "replayTracePayloadSha256": None if replay_trace_record is None else replay_trace_record["sha256"],
        "workerCpuSeconds": time.process_time() - started_cpu,
        "workerWallSeconds": time.perf_counter() - started_wall,
    }
    for difference in all_differences:
        difference.update(
            {
                "campaign": payload["campaign"],
                "pairId": payload["pairId"],
                "particleId": payload["particleId"],
                "matrixIndex": int(payload["matrixIndex"]),
            }
        )
    trace_manifest = {
        "campaign": payload["campaign"],
        "pairId": payload["pairId"],
        "particleId": payload["particleId"],
        "matrixIndex": int(payload["matrixIndex"]),
        **trace_record,
        "representsBothSides": pair_gate,
        "replayPath": None if replay_trace_record is None else replay_trace_record["path"],
        "replaySha256": None if replay_trace_record is None else replay_trace_record["sha256"],
    }
    return {
        "pair": row,
        "differences": all_differences,
        "trace": trace_manifest,
        "seeds": seed_rows(
            left.seeds, payload["campaign"], payload["pairId"], payload["particleId"]
        ),
    }


def run_pair_campaign(
    tasks: list[dict[str, Any]], workers: int = WORKERS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    pair_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_pair_task, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            pair_rows.append(result["pair"])
            difference_rows.extend(result["differences"])
            trace_rows.append(result["trace"])
            seed_records.extend(result["seeds"])
    pair_frame = pd.DataFrame(pair_rows).sort_values(
        ["particleId", "matrixIndex"]
    ).reset_index(drop=True)
    difference_columns = [
        "campaign",
        "pairId",
        "particleId",
        "matrixIndex",
        "comparisonScope",
        "path",
        "category",
        "left",
        "right",
        "permitted",
        "deterministic_cause",
    ]
    difference_frame = pd.DataFrame(difference_rows, columns=difference_columns).sort_values(
        ["particleId", "matrixIndex", "comparisonScope", "path"]
    ).reset_index(drop=True)
    trace_frame = pd.DataFrame(trace_rows).sort_values(
        ["particleId", "matrixIndex"]
    ).reset_index(drop=True)
    seed_frame = (
        pd.DataFrame(seed_records)
        .drop_duplicates("seedMaterialSha256")
        .sort_values(["purpose", "matrixIndex", "particleId"], na_position="first")
        .reset_index(drop=True)
    )
    return pair_frame, difference_frame, trace_frame, seed_frame, time.perf_counter() - started


def _scientific_task(payload: dict[str, Any]) -> dict[str, Any]:
    definition = definition_from_payload(payload)
    common = {
        "phase": payload["phase"],
        "root_hex": payload["root"],
        "matrix_index": int(payload["matrixIndex"]),
        "definition": definition,
        "stream_identity": payload["streamIdentity"],
    }
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    left, left_seeds = simulate_trajectory(**common)
    right, right_seeds = simulate_trajectory(**common)
    comparison = compare_trajectories(left, right)
    seeds_passed, seed_differences = compare_seed_tuples(left_seeds, right_seeds)
    summary = trajectory_summary(left)
    summary.update(
        {
            "particleId": payload.get("particleId"),
            "candidateId": payload.get("candidateId"),
            "clockId": payload.get("clockId"),
            "repairedReplayPassed": comparison.repaired_comparator_passed
            and seeds_passed,
            "oldReplayPassed": comparison.old_comparator_passed,
            "permittedPairedNanCount": comparison.permitted_paired_nan_count,
            "discreteDivergenceCount": comparison.discrete_divergence_count,
            "finiteNumericDivergenceCount": comparison.finite_numeric_divergence_count,
            "forbiddenNonfiniteDifferenceCount": comparison.forbidden_nonfinite_difference_count,
            "seedDifferenceCount": len(seed_differences),
            "workerCpuSeconds": time.process_time() - started_cpu,
            "workerWallSeconds": time.perf_counter() - started_wall,
        }
    )
    cache_path = payload.get("cachePath")
    if cache_path:
        import pickle

        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(left, handle, protocol=5)
        summary["cachePath"] = str(path)
        summary["cacheSha256"] = sha256_file(path)
    return {
        "summary": summary,
        "seeds": seed_rows(
            left_seeds,
            payload.get("campaign", payload["phase"]),
            payload.get("pairId", f"{payload.get('particleId')}-M{payload['matrixIndex']:03d}"),
            payload.get("particleId", payload["streamIdentity"]),
        ),
        "trajectory": left if payload.get("returnTrajectory") else None,
    }


def run_scientific_tasks(
    tasks: list[dict[str, Any]], workers: int = WORKERS
) -> tuple[pd.DataFrame, pd.DataFrame, list[Any], float]:
    summaries: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    trajectories: list[Any] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scientific_task, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            summaries.append(result["summary"])
            seeds.extend(result["seeds"])
            if result["trajectory"] is not None:
                trajectories.append(result["trajectory"])
    frame = pd.DataFrame(summaries).sort_values(
        [column for column in ("particleId", "candidateId", "matrixIndex") if column in pd.DataFrame(summaries).columns],
        na_position="last",
    ).reset_index(drop=True)
    seed_frame = (
        pd.DataFrame(seeds)
        .drop_duplicates("seedMaterialSha256")
        .sort_values(["purpose", "matrixIndex", "particleId"], na_position="first")
        .reset_index(drop=True)
    )
    trajectories.sort(key=lambda value: (value.configuration_id, value.matrix_index))
    return frame, seed_frame, trajectories, time.perf_counter() - started
