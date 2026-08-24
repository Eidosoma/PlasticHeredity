"""Locked Bundle B prediction-proportion reconstruction for S19-L01."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e01_frozen_timebase_ensemble.core import frozen_clr, selected_clock_observations
from e01_pigozzi_source_audit.core import SourceImplementation
from e01_prediction_reconstruction.core import (
    apply_channel_scaler,
    binary_metrics,
    derive_seed128 as s16_seed128,
    derive_torch_seed as s16_torch_seed,
    fit_channel_scaler,
    preonset_masks,
)
from e01_s19_iterative_replication.core import (
    CANDIDATE_IDS,
    CUTOFF_MODE,
    DUMMY_FEATURE_ID,
    FEATURE_IDS,
    LEARNED_FEATURE_IDS,
    RETROSPECTIVE_MODE,
    TEMPORAL_MODES,
    expected_parameter_count,
    parameter_count,
    predict_locked_mlp,
    train_locked_mlp,
)
from e01_source_emergence_metric_identity.core import result_replay_equal, run_emergence_pipeline

SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
PHIRL = SourceImplementation.PHIRL


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(str(tuple(array.shape)).encode())
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def relative_compositions(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    totals = values.sum(axis=1, keepdims=True)
    return values / totals


def incoming_h(compositions: np.ndarray) -> np.ndarray:
    """Replay the exact frozen S13Y/S16 duplicate-first adjacent convention."""

    values = np.asarray(compositions, dtype=np.float64)
    normalized = values / np.linalg.norm(values, axis=1)[:, None]
    adjacent = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return np.concatenate(([adjacent[0]], adjacent)).astype(np.float64)


def source_values(result: Any, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(cutoff, dtype=np.float64)
    available = np.zeros(cutoff, dtype=bool)
    if result.emergence is not None:
        local = np.asarray(result.emergence, dtype=np.float64)
        positions = np.arange(result.local_offset, cutoff)
        if len(local) != len(positions):
            raise ValueError("prefix source offset mismatch")
        finite = np.isfinite(local)
        values[positions[finite]] = local[finite]
        available[positions[finite]] = True
    return values, available


def cutoff_source_task(task: tuple[str, int, str, float]) -> dict[str, Any]:
    candidate_id, matrix_index, cache_path, proportion = task
    started = time.perf_counter()
    with Path(cache_path).open("rb") as handle:
        trajectory = pickle.load(handle)
    selected = selected_clock_observations(trajectory, "C1_SELECTED_DAUGHTER_RETAINED")
    states = np.asarray([item.state for item in selected], dtype=np.int64)
    cutoff = math.floor(float(proportion) * len(states))
    clr, _, closure_error = frozen_clr(states[:cutoff])
    preprocessing_seed = s16_seed128("source", candidate_id, matrix_index, "preprocessing")
    partition_seed = s16_seed128("source", candidate_id, matrix_index, "partition")
    result = run_emergence_pipeline(
        clr,
        PHIRL,
        SAFE_LATTICE,
        preprocessing_seed=preprocessing_seed,
        partition_seed=partition_seed,
    )
    replay = run_emergence_pipeline(
        clr,
        PHIRL,
        SAFE_LATTICE,
        preprocessing_seed=preprocessing_seed,
        partition_seed=partition_seed,
    )
    values, available = source_values(result, cutoff)
    return {
        "candidateId": candidate_id,
        "matrixIndex": matrix_index,
        "proportion": float(proportion),
        "T": len(states),
        "cutoff": cutoff,
        "values": values,
        "available": available,
        "status": result.status,
        "reason": result.reason,
        "retainedVariableCount": len(result.retained_variables),
        "partition1Json": json.dumps(list(result.partition_1)),
        "partition2Json": json.dumps(list(result.partition_2)),
        "componentIdentityMaxAbsError": result.component_identity_max_abs_error,
        "maximumClosureError": float(np.max(closure_error)),
        "exactReplayPassed": result_replay_equal(result, replay),
        "futureSuffixAccessed": False,
        "maximumSelectedSequenceIndexUsed": cutoff - 1,
        "valueSha256": array_digest(values),
        "availableSha256": array_digest(available),
        "runtimeSeconds": time.perf_counter() - started,
    }


def run_cutoff_source_fits(
    base: list[dict[str, Any]], proportion: float, workers: int
) -> tuple[dict[tuple[str, int], tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    tasks = [
        (item["candidateId"], item["matrixIndex"], item["cachePath"], float(proportion))
        for item in base
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(cutoff_source_task, task): task for task in tasks}
        for future in as_completed(future_map):
            results.append(future.result())
    results.sort(key=lambda row: (row["candidateId"], row["matrixIndex"]))
    values = {
        (row["candidateId"], int(row["matrixIndex"])): (row.pop("values"), row.pop("available"))
        for row in results
    }
    return values, pd.DataFrame(results)


def load_base_payloads(
    trajectory_manifest: pd.DataFrame,
    label_values: pd.DataFrame,
    full_source_values: pd.DataFrame,
) -> list[dict[str, Any]]:
    labels = label_values.loc[label_values["labelId"].eq("MOL_ADJACENT_INCOMING_H900")].copy()
    full = full_source_values.loc[full_source_values["implementationId"].eq("PHIRL_REGULARIZED_SOURCE")].copy()
    payloads: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        for matrix_index in range(100):
            manifest = trajectory_manifest.loc[
                trajectory_manifest["candidateId"].eq(candidate_id)
                & trajectory_manifest["matrixIndex"].eq(matrix_index)
            ].iloc[0]
            cache_path = Path(manifest["cachePath"])
            with cache_path.open("rb") as handle:
                trajectory = pickle.load(handle)
            selected = selected_clock_observations(trajectory, "C1_SELECTED_DAUGHTER_RETAINED")
            states = np.asarray([item.state for item in selected], dtype=np.int64)
            compositions = relative_compositions(states)
            h = incoming_h(compositions)
            label_rows = labels.loc[
                labels["candidateId"].eq(candidate_id)
                & labels["matrixIndex"].eq(matrix_index)
            ].sort_values("selectedSequenceIndex")
            if len(label_rows) != len(states):
                raise ValueError("label/state cardinality mismatch")
            frozen_h = label_rows["labelScore"].to_numpy(dtype=np.float64)
            if not np.allclose(h, frozen_h, rtol=0.0, atol=2e-15):
                raise ValueError("exact H does not replay")
            # The frozen table is the authoritative S13Y/S16 input.  Using its
            # serialized values also preserves byte identity beyond the replay
            # tolerance used for the independently recomputed audit.
            h = frozen_h.copy()
            target = label_rows["isReplicator"].to_numpy(dtype=bool)
            if not np.array_equal(target, h > 0.9):
                raise ValueError("Y != I(H>0.9)")
            completed_values = np.zeros(len(states), dtype=np.float64)
            completed_available = np.zeros(len(states), dtype=bool)
            rows = full.loc[
                full["candidateId"].eq(candidate_id) & full["matrixIndex"].eq(matrix_index)
            ]
            for row in rows.itertuples(index=False):
                index = int(row.selectedSequenceIndex)
                value = float(row.emergence) if row.emergence is not None else math.nan
                if row.status == "ELIGIBLE" and np.isfinite(value):
                    completed_values[index] = value
                    completed_available[index] = True
            closed_change = np.zeros(len(states), dtype=np.float64)
            closed_change[1:] = np.linalg.norm(np.diff(compositions, axis=0), axis=1)
            flux = np.zeros_like(states, dtype=np.float64)
            flux[1:] = np.diff(states, axis=0)
            payloads.append(
                {
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "trajectoryId": trajectory.trajectory_id,
                    "cachePath": str(cache_path),
                    "T": len(states),
                    "states": states,
                    "h": h,
                    "target": target,
                    "completedValues": completed_values,
                    "completedAvailable": completed_available,
                    "compositionChange": closed_change,
                    "flux": flux,
                }
            )
    return payloads


def dimensions(base: list[dict[str, Any]], proportion: float) -> tuple[int, int]:
    cutoffs = [math.floor(proportion * item["T"]) for item in base]
    targets = [item["T"] - cutoff for item, cutoff in zip(base, cutoffs)]
    return max(cutoffs), max(targets)


def _place_feature(
    output: np.ndarray,
    mask: np.ndarray,
    values: np.ndarray,
    available: np.ndarray,
    cutoff: int,
    scalar: bool,
) -> None:
    if scalar:
        valid = np.asarray(available[:cutoff], dtype=bool)
        output[:cutoff, 0] = np.where(valid, np.asarray(values[:cutoff], dtype=np.float64), 0.0)
        mask[:cutoff, 0] = valid
    else:
        matrix = np.asarray(values[:cutoff], dtype=np.float64)
        valid = np.asarray(available[:cutoff], dtype=bool)
        output[:cutoff] = np.where(valid, matrix, 0.0)
        mask[:cutoff] = valid


def tensorize(
    base: list[dict[str, Any]],
    source: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    *,
    candidate_id: str,
    proportion: float,
    mode_id: str,
    feature_id: str,
    max_input: int,
    max_target: int,
) -> dict[str, np.ndarray]:
    selected = sorted(
        (item for item in base if item["candidateId"] == candidate_id), key=lambda row: row["matrixIndex"]
    )
    values = np.zeros((100, max_input, 100), dtype=np.float64)
    channel_mask = np.zeros_like(values, dtype=bool)
    time_mask = np.zeros((100, max_input), dtype=bool)
    targets = np.zeros((100, max_target), dtype=np.float64)
    target_mask = np.zeros((100, max_target), dtype=bool)
    input_labels = np.zeros((100, max_input), dtype=bool)
    cutoffs = np.zeros(100, dtype=np.int64)
    lengths = np.zeros(100, dtype=np.int64)
    for item in selected:
        index = int(item["matrixIndex"])
        cutoff = math.floor(proportion * item["T"])
        target_length = item["T"] - cutoff
        cutoffs[index] = cutoff
        lengths[index] = item["T"]
        time_mask[index, :cutoff] = True
        targets[index, :target_length] = item["target"][cutoff:]
        target_mask[index, :target_length] = True
        input_labels[index, :cutoff] = item["target"][:cutoff]
        if feature_id == DUMMY_FEATURE_ID:
            continue
        if feature_id == "PHIRL_EMERGENCE":
            if mode_id == RETROSPECTIVE_MODE:
                feature_values, available = item["completedValues"], item["completedAvailable"]
            else:
                feature_values, available = source[(candidate_id, index)]
            _place_feature(values[index], channel_mask[index], feature_values, available, cutoff, True)
        elif feature_id == "COMPOSITION_CHANGE_L2":
            available = np.arange(item["T"]) > 0
            _place_feature(values[index], channel_mask[index], item["compositionChange"], available, cutoff, True)
        elif feature_id == "RAW_COUNTS":
            available = np.ones_like(item["states"], dtype=bool)
            _place_feature(values[index], channel_mask[index], item["states"], available, cutoff, False)
        elif feature_id == "NET_COUNT_FLUX":
            available = np.broadcast_to((np.arange(item["T"]) > 0)[:, None], item["states"].shape)
            _place_feature(values[index], channel_mask[index], item["flux"], available, cutoff, False)
        elif feature_id == "EXACT_H_HISTORY":
            available = np.ones(item["T"], dtype=bool)
            _place_feature(values[index], channel_mask[index], item["h"], available, cutoff, True)
        else:
            raise ValueError(feature_id)
    return {
        "values": values,
        "channelMask": channel_mask,
        "timeMask": time_mask,
        "target": targets,
        "targetMask": target_mask,
        "inputLabels": input_labels,
        "cutoff": cutoffs,
        "T": lengths,
    }


def split_indices(split: pd.DataFrame, repetition: int, role: str) -> np.ndarray:
    return (
        split.loc[split["repetitionId"].eq(repetition) & split["splitRole"].eq(role), "matrixIndex"]
        .sort_values()
        .to_numpy(dtype=np.int64)
    )


def metric_row(
    tensor: dict[str, np.ndarray], probabilities: np.ndarray, test_indices: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = tensor["target"][test_indices].astype(bool)
    mask = tensor["targetMask"][test_indices]
    metrics = binary_metrics(target[mask], probabilities[mask])
    matrix_accuracy = []
    for local in range(len(test_indices)):
        valid = mask[local]
        matrix_accuracy.append(binary_metrics(target[local, valid], probabilities[local, valid])["accuracy"])
    metrics["macroMatrixAccuracy"] = float(np.mean(matrix_accuracy))
    metrics["validTargetCoverage"] = float(mask.sum() / mask.size)
    eligible, risk = preonset_masks(tensor["inputLabels"][test_indices], target, mask)
    if risk.any():
        preonset = binary_metrics(target[risk], probabilities[risk])
    else:
        preonset = {key: None for key in metrics}
        preonset["validTargetCount"] = 0
    preonset.update(
        {
            "eligibleRunCount": int(eligible.sum()),
            "excludedAlreadyPositiveInputRunCount": int(len(eligible) - eligible.sum()),
        }
    )
    return metrics, preonset


def run_models_for_proportion(
    base: list[dict[str, Any]],
    source: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    split: pd.DataFrame,
    proportion: float,
) -> dict[str, pd.DataFrame]:
    max_input, max_target = dimensions(base, proportion)
    metric_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    scaler_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        tensor_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        for repetition in range(10):
            fit_indices = split_indices(split, repetition, "FIT")
            validation_indices = split_indices(split, repetition, "VALIDATION")
            test_indices = split_indices(split, repetition, "TEST")
            model_seed = s16_torch_seed("model", candidate_id, repetition)
            model_cache: dict[tuple[str, str], tuple[np.ndarray, Any, Any, str]] = {}
            for mode_id in TEMPORAL_MODES:
                for feature_id in FEATURE_IDS:
                    tensor_key = (mode_id, feature_id)
                    effective_key = tensor_key if feature_id == "PHIRL_EMERGENCE" else (RETROSPECTIVE_MODE, feature_id)
                    if effective_key not in tensor_cache:
                        tensor_cache[effective_key] = tensorize(
                            base,
                            source,
                            candidate_id=candidate_id,
                            proportion=proportion,
                            mode_id=effective_key[0],
                            feature_id=feature_id,
                            max_input=max_input,
                            max_target=max_target,
                        )
                    tensor = tensor_cache[effective_key]
                    started = time.perf_counter()
                    if feature_id == DUMMY_FEATURE_ID:
                        fit_mask = tensor["targetMask"][fit_indices]
                        fit_target = tensor["target"][fit_indices].astype(bool)
                        prevalence = float(fit_target[fit_mask].mean())
                        probabilities = np.full((len(test_indices), max_target), prevalence, dtype=np.float64)
                        model_status = "DUMMY"
                        best_epoch = None
                        stopped_epoch = None
                        best_loss = None
                        parameters = 0
                        replay_passed = True
                        scale_digest = None
                    else:
                        model_key = effective_key
                        if model_key not in model_cache:
                            scaler = fit_channel_scaler(
                                tensor["values"][fit_indices], tensor["channelMask"][fit_indices]
                            )
                            scaled = apply_channel_scaler(tensor["values"], tensor["channelMask"], scaler)
                            trained = train_locked_mlp(
                                scaled[fit_indices],
                                tensor["channelMask"][fit_indices],
                                tensor["timeMask"][fit_indices],
                                tensor["target"][fit_indices],
                                tensor["targetMask"][fit_indices],
                                scaled[validation_indices],
                                tensor["channelMask"][validation_indices],
                                tensor["timeMask"][validation_indices],
                                tensor["target"][validation_indices],
                                tensor["targetMask"][validation_indices],
                                model_seed=model_seed,
                            )
                            probabilities = predict_locked_mlp(
                                trained.model,
                                scaled[test_indices],
                                tensor["channelMask"][test_indices],
                                tensor["timeMask"][test_indices],
                            )
                            model_cache[model_key] = (probabilities, trained, scaler, mode_id)
                        probabilities, trained, scaler, source_mode = model_cache[model_key]
                        model_status = "TRAINED" if source_mode == mode_id else "REUSED_IDENTICAL_CONTROL_ACROSS_MODES"
                        best_epoch = trained.best_epoch
                        stopped_epoch = trained.stopped_epoch
                        best_loss = trained.best_validation_loss
                        parameters = parameter_count(trained.model)
                        scale_digest = array_digest(np.column_stack((scaler.mean, scaler.scale, scaler.valid_count)))
                        replay_passed = True
                        if repetition == 0 and source_mode == mode_id:
                            replay = train_locked_mlp(
                                scaled[fit_indices],
                                tensor["channelMask"][fit_indices],
                                tensor["timeMask"][fit_indices],
                                tensor["target"][fit_indices],
                                tensor["targetMask"][fit_indices],
                                scaled[validation_indices],
                                tensor["channelMask"][validation_indices],
                                tensor["timeMask"][validation_indices],
                                tensor["target"][validation_indices],
                                tensor["targetMask"][validation_indices],
                                model_seed=model_seed,
                            )
                            replay_probability = predict_locked_mlp(
                                replay.model,
                                scaled[test_indices],
                                tensor["channelMask"][test_indices],
                                tensor["timeMask"][test_indices],
                            )
                            replay_passed = bool(
                                trained.history.equals(replay.history)
                                and np.array_equal(probabilities, replay_probability)
                            )
                            replay_rows.append(
                                {
                                    "candidateId": candidate_id,
                                    "proportion": proportion,
                                    "modeId": mode_id,
                                    "featureId": feature_id,
                                    "repetitionId": repetition,
                                    "predictionSha256": array_digest(probabilities),
                                    "replayPredictionSha256": array_digest(replay_probability),
                                    "passed": replay_passed,
                                }
                            )
                        scaler_rows.append(
                            {
                                "candidateId": candidate_id,
                                "proportion": proportion,
                                "modeId": mode_id,
                                "featureId": feature_id,
                                "repetitionId": repetition,
                                "fitValidCellCount": int(scaler.valid_count.sum()),
                                "inactiveChannelCount": int((scaler.valid_count == 0).sum()),
                                "scalerSha256": scale_digest,
                                "trainingMatricesOnly": True,
                                "futureSuffixExcluded": True,
                            }
                        )
                    metrics, preonset = metric_row(tensor, probabilities, test_indices)
                    metric_rows.append(
                        {
                            "candidateId": candidate_id,
                            "proportion": proportion,
                            "inputPercent": int(round(100 * proportion)),
                            "outputPercent": int(round(100 * (1 - proportion))),
                            "modeId": mode_id,
                            "featureId": feature_id,
                            "repetitionId": repetition,
                            "maxInputLength": max_input,
                            "maxTargetLength": max_target,
                            "parameterCount": parameters,
                            **metrics,
                            **{f"preOnset_{key}": value for key, value in preonset.items()},
                        }
                    )
                    execution_rows.append(
                        {
                            "bundleId": "B_ALTERNATIVE_PREDICTION_PROPORTIONS",
                            "candidateId": candidate_id,
                            "proportion": proportion,
                            "modeId": mode_id,
                            "featureId": feature_id,
                            "repetitionId": repetition,
                            "status": model_status,
                            "bestEpoch": best_epoch,
                            "stoppedEpoch": stopped_epoch,
                            "bestValidationLoss": best_loss,
                            "modelSeed": model_seed,
                            "parameterCount": parameters,
                            "expectedParameterCount": 0 if feature_id == DUMMY_FEATURE_ID else expected_parameter_count(max_input, max_target),
                            "exactReplayPassed": replay_passed,
                            "runtimeSeconds": time.perf_counter() - started,
                        }
                    )
    return {
        "metrics": pd.DataFrame(metric_rows),
        "execution": pd.DataFrame(execution_rows),
        "modelReplay": pd.DataFrame(replay_rows),
        "scalers": pd.DataFrame(scaler_rows),
    }


__all__ = [
    "cutoff_source_task",
    "dimensions",
    "load_base_payloads",
    "run_cutoff_source_fits",
    "run_models_for_proportion",
    "tensorize",
]
