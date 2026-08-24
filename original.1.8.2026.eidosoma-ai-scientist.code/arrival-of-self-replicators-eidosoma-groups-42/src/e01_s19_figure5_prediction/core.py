"""Outcome-blind primitives for E01/S19-L13.

This module composes, without changing, the frozen S16 tensor/model contract,
the L10 R1 recurring-compotype implementation, and the repaired L11R U2
implementation.  It contains no filesystem I/O and no outcome-driven branch.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from e01_prediction_reconstruction.core import (
    FEATURE_CHANNEL_CAPACITY,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    apply_channel_scaler,
    build_split_manifest,
    expected_calibration_error,
    fit_channel_scaler,
    parameter_count,
    predict_probabilities,
    split_summary,
    train_masked_mlp,
)
from e01_prediction_reconstruction.core import (
    derive_seed128 as s16_seed128,
)
from e01_prediction_reconstruction.core import (
    derive_torch_seed as s16_torch_seed,
)
from e01_s19_all_comptype_union_repair.core import (
    U2_ID,
    direct_union_scores,
    materialize_u2,
)
from e01_s19_matlab_attractor.core import (
    R1_ID,
    fit_r1_matlab_historical,
    label_against_reference,
)

VERSION = "E01-S19-L13-FIGURE5-RECURRING-TARGET-PREDICTION-RECONSTRUCTION-v1.0.0"
LOOP_ID = "S19-L13"
ROOT_SEED_HEX = "28b695a7f614e5d4072cc9733b75c49fb00846325db4399df69678a18a33d309"

CANDIDATE_IDS = ("CANDIDATE_2", "CANDIDATE_3")
S16_CANDIDATE_ALIAS = {
    "CANDIDATE_2": "S12F-CANDIDATE-02",
    "CANDIDATE_3": "S12F-CANDIDATE-03",
}
R1_TARGET_ID = "F5_R1_HISTORICAL_DOMINANT_COMPTYPE_H090"
U2_TARGET_ID = "F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090"
TARGET_IDS = (R1_TARGET_ID, U2_TARGET_ID)

P1_ID = "P1_PHIRL_EMERGENCE_COMPLETED_FIT"
P2_ID = "P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY"
B1_ID = "B1_COMPOSITION_CHANGE"
B2_ID = "B2_RAW_COMPOSITIONS"
B3_ID = "B3_MOLECULAR_FLUXES"
B4_ID = "B4_ADJACENT_H"
B5_ID = "B5_PREFIX_ATTRACTOR_GEOMETRY"
B6_ID = "B6_TIME_ONLY"
B7_ID = "B7_RANDOM_MATCHED_SHAPE"
P2_B4_ID = "P2_PLUS_B4"
P2_B5_ID = "P2_PLUS_B5"
DUMMY_ID = "MAJORITY_DUMMY"
ORACLE_ID = "O1_COMPLETED_TARGET_CENTROID_ORACLE"
NC1_ID = "NC1_WITHIN_PREFIX_TEMPORAL_PERMUTATION"
NC2_ID = "NC2_MATRIX_LABEL_PERMUTATION"

PRIMARY_MODEL_IDS = (
    P1_ID,
    P2_ID,
    B1_ID,
    B2_ID,
    B3_ID,
    B4_ID,
    B5_ID,
    B6_ID,
    B7_ID,
    P2_B4_ID,
    P2_B5_ID,
    DUMMY_ID,
)
DIAGNOSTIC_MODEL_IDS = (ORACLE_ID, NC1_ID, NC2_ID)
ALL_MODEL_IDS = (*PRIMARY_MODEL_IDS, *DIAGNOSTIC_MODEL_IDS)

PAPER_INTERVALS = {
    P1_ID: (0.80, 0.90),
    B1_ID: (0.75, 0.85),
    B2_ID: (0.75, 0.85),
    B3_ID: (0.74, 0.84),
    DUMMY_ID: (0.55, 0.65),
}


def seed128(*parts: object) -> int:
    material = "\x1f".join([VERSION, ROOT_SEED_HEX, *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def array_sha256(values: NDArray[Any]) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def normalized_compositions(states: NDArray[Any]) -> NDArray[np.float64]:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 100:
        raise ValueError("selected states must be [time,100]")
    mass = values.sum(axis=1)
    if np.any(mass <= 0) or np.any(~np.isfinite(values)):
        raise ValueError("selected states must be finite and nonempty")
    return np.ascontiguousarray(values / mass[:, None], dtype=np.float64)


def incoming_h(compositions: NDArray[Any]) -> NDArray[np.float64]:
    values = np.asarray(compositions, dtype=np.float64)
    if len(values) < 2:
        raise ValueError("incoming-H requires at least two states")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise ValueError("incoming-H has zero norm")
    normalized = values / norms[:, None]
    adjacent = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return np.concatenate(([adjacent[0]], adjacent)).astype(np.float64)


def build_feature(
    values: NDArray[Any],
    available: NDArray[Any],
    cutoff: int,
    *,
    scalar: bool,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    if cutoff < 1 or cutoff > MAX_INPUT_LENGTH:
        raise ValueError("cutoff exceeds frozen S16 input capacity")
    output = np.zeros((MAX_INPUT_LENGTH, FEATURE_CHANNEL_CAPACITY), dtype=np.float64)
    channel_mask = np.zeros_like(output, dtype=bool)
    time_mask = np.zeros(MAX_INPUT_LENGTH, dtype=bool)
    time_mask[:cutoff] = True
    if scalar:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        valid = np.asarray(available, dtype=bool).reshape(-1)
        if len(vector) < cutoff or len(valid) < cutoff:
            raise ValueError("scalar feature shorter than cutoff")
        output[:cutoff, 0] = np.where(valid[:cutoff], vector[:cutoff], 0.0)
        channel_mask[:cutoff, 0] = valid[:cutoff]
    else:
        matrix = np.asarray(values, dtype=np.float64)
        valid = np.asarray(available, dtype=bool)
        if matrix.shape != (cutoff, FEATURE_CHANNEL_CAPACITY) or valid.shape != matrix.shape:
            raise ValueError("vector feature does not match frozen S16 shape")
        output[:cutoff] = np.where(valid, matrix, 0.0)
        channel_mask[:cutoff] = valid
    if np.any(~np.isfinite(output)):
        raise ValueError("feature tensor contains nonfinite values")
    return output, channel_mask, time_mask


def combine_scalar_features(
    left: tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]],
    right: tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    if not np.array_equal(left[2], right[2]):
        raise ValueError("combined features have different time masks")
    values = np.zeros_like(left[0])
    mask = np.zeros_like(left[1])
    values[:, 0] = left[0][:, 0]
    values[:, 1] = right[0][:, 0]
    mask[:, 0] = left[1][:, 0]
    mask[:, 1] = right[1][:, 0]
    return values, mask, left[2].copy()


def build_target_tensor(
    labels: NDArray[Any] | None, total: int
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_], int]:
    cutoff = math.floor(0.25 * total)
    target_length = total - cutoff
    if cutoff > MAX_INPUT_LENGTH or target_length > MAX_TARGET_LENGTH:
        raise ValueError("trajectory exceeds frozen S16 tensor capacity")
    target = np.zeros(MAX_TARGET_LENGTH, dtype=np.float64)
    target_mask = np.zeros(MAX_TARGET_LENGTH, dtype=bool)
    input_labels = np.zeros(MAX_INPUT_LENGTH, dtype=bool)
    if labels is not None:
        y = np.asarray(labels, dtype=bool)
        if len(y) != total:
            raise ValueError("label length does not match selected clock")
        input_labels[:cutoff] = y[:cutoff]
        target[:target_length] = y[cutoff:].astype(np.float64)
        target_mask[:target_length] = True
    return target, target_mask, input_labels, cutoff


def source_values(
    result: Any, *, fit_length: int, retained_length: int
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    values = np.zeros(retained_length, dtype=np.float64)
    available = np.zeros(retained_length, dtype=bool)
    if result.emergence is None:
        return values, available
    local = np.asarray(result.emergence, dtype=np.float64)
    expected = fit_length - int(result.local_offset)
    if local.size != expected:
        raise ValueError("PhiRL local trajectory length mismatch")
    indices = np.arange(int(result.local_offset), fit_length)
    keep = indices < retained_length
    local = local[keep]
    indices = indices[keep]
    finite = np.isfinite(local)
    values[indices[finite]] = local[finite]
    available[indices[finite]] = True
    return values, available


def s16_source_seed(candidate_id: str, matrix_index: int, purpose: str) -> int:
    alias = S16_CANDIDATE_ALIAS[candidate_id]
    return int(s16_seed128("cutoff_source", alias, matrix_index, purpose) % (2**32))


def s16_model_seed(candidate_id: str, repetition: int) -> int:
    return int(s16_torch_seed("model", S16_CANDIDATE_ALIAS[candidate_id], repetition))


def split_indices(split: pd.DataFrame, repetition: int, role: str) -> NDArray[np.int64]:
    return (
        split.loc[
            split["repetitionId"].eq(repetition) & split["splitRole"].eq(role),
            "matrixIndex",
        ]
        .sort_values()
        .to_numpy(dtype=np.int64)
    )


def _calibration_coefficients(
    target: NDArray[np.bool_], probability: NDArray[np.float64]
) -> tuple[float | None, float | None]:
    y = np.asarray(target, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    if y.size == 0 or np.unique(y).size != 2 or np.std(p) < 1e-15:
        return None, None
    x = np.column_stack([np.ones(len(p)), np.log(p / (1.0 - p))])
    beta = np.asarray([0.0, 1.0], dtype=np.float64)
    for _ in range(50):
        eta = x @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        weight = np.maximum(mu * (1.0 - mu), 1e-10)
        information = x.T @ (weight[:, None] * x)
        score = x.T @ (y - mu)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            return None, None
        beta += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    if np.any(~np.isfinite(beta)):
        return None, None
    return float(beta[0]), float(beta[1])


def extended_binary_metrics(
    target: NDArray[Any], probability: NDArray[Any]
) -> dict[str, Any]:
    y = np.asarray(target, dtype=bool).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    if y.size == 0 or len(p) != len(y) or np.any(~np.isfinite(p)):
        return {
            "validTargetCount": int(y.size), "positiveCount": int(y.sum()),
            "prevalence": None if not y.size else float(y.mean()), "accuracy": None,
            "balancedAccuracy": None, "auroc": None, "auprc": None, "brier": None,
            "logLoss": None, "sensitivity": None, "specificity": None,
            "positivePredictiveValue": None, "negativePredictiveValue": None,
            "calibrationIntercept": None, "calibrationSlope": None,
            "calibrationError": None, "metricStatus": "INELIGIBLE_EMPTY_OR_NONFINITE",
        }
    p = np.clip(p, 0.0, 1.0)
    predicted = p >= 0.5
    tp = int(np.count_nonzero(predicted & y))
    tn = int(np.count_nonzero(~predicted & ~y))
    fp = int(np.count_nonzero(predicted & ~y))
    fn = int(np.count_nonzero(~predicted & y))
    both = np.unique(y).size == 2
    intercept, slope = _calibration_coefficients(y, p)
    return {
        "validTargetCount": int(y.size),
        "positiveCount": int(y.sum()),
        "prevalence": float(y.mean()),
        "accuracy": float(np.mean(predicted == y)),
        "balancedAccuracy": float(balanced_accuracy_score(y, predicted)) if both else None,
        "auroc": float(roc_auc_score(y, p)) if both else None,
        "auprc": float(average_precision_score(y, p)) if both else None,
        "brier": float(brier_score_loss(y, p)),
        "logLoss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[False, True])),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "positivePredictiveValue": float(tp / (tp + fp)) if tp + fp else None,
        "negativePredictiveValue": float(tn / (tn + fn)) if tn + fn else None,
        "calibrationIntercept": intercept,
        "calibrationSlope": slope,
        "calibrationError": expected_calibration_error(y, p, bins=10),
        "metricStatus": "ELIGIBLE_BOTH_CLASSES" if both else "ELIGIBLE_SINGLE_CLASS_PARTIAL_METRICS",
    }


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    finite = [(index, float(value)) for index, value in enumerate(p_values) if value is not None and np.isfinite(value)]
    result: list[float | None] = [None] * len(p_values)
    ordered = sorted(finite, key=lambda pair: pair[1])
    running = 0.0
    total = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * value)
        running = max(running, adjusted)
        result[index] = running
    return result


def matrix_bootstrap_metric_difference(
    rows: pd.DataFrame,
    *,
    reference: str,
    comparator: str,
    metric: str,
    seed_identity: tuple[object, ...],
    replicates: int = 4096,
) -> dict[str, Any]:
    selected = rows.loc[rows["modelId"].isin([reference, comparator])]
    pivot = selected.pivot_table(
        index=["repetitionId", "matrixIndex"], columns="modelId", values=metric,
        aggfunc="first",
    ).dropna()
    if reference not in pivot or comparator not in pivot or pivot.empty:
        return {"pairedMatrixCount": 0, "observedDifference": None, "lower95": None, "upper95": None, "positiveP": None}
    grouped = (pivot[reference] - pivot[comparator]).groupby(level="matrixIndex").mean()
    differences = grouped.to_numpy(dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64DXSM(seed128(*seed_identity)))
    indices = rng.integers(0, len(differences), size=(replicates, len(differences)))
    distribution = differences[indices].mean(axis=1)
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    return {
        "pairedMatrixCount": len(differences),
        "observedDifference": float(differences.mean()),
        "lower95": float(lower),
        "upper95": float(upper),
        "positiveP": float((1 + np.count_nonzero(distribution <= 0)) / (replicates + 1)),
    }


def paper_interval_overlap(values: NDArray[Any], model_id: str) -> bool:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return False
    lower, upper = PAPER_INTERVALS[model_id]
    return bool(float(x.max()) >= lower and float(x.min()) <= upper)


def geometry_gate(
    defined_count: int,
    dummy_values: NDArray[Any],
    aggregate_target: NDArray[Any],
) -> dict[str, bool]:
    values = np.asarray(dummy_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    target = np.asarray(aggregate_target, dtype=bool)
    interval_overlap = paper_interval_overlap(values, DUMMY_ID)
    median_in_interval = bool(values.size and 0.55 <= float(np.median(values)) <= 0.65)
    return {
        "minimumDefinedMatrices": bool(defined_count >= 80),
        "bothClasses": bool(target.size and np.unique(target).size == 2),
        "dummyRangeOverlap": interval_overlap,
        "dummyMedianInInterval": median_in_interval,
        "passed": bool(defined_count >= 80 and target.size and np.unique(target).size == 2 and interval_overlap and median_in_interval),
    }


@dataclass(frozen=True, slots=True)
class TargetResult:
    target_id: str
    status: str
    labels: NDArray[np.bool_] | None
    scores: NDArray[np.float64] | None
    centroids: NDArray[np.float64] | None
    selected_k: int | None


def r1_target(
    boundary_compositions: NDArray[Any],
    molecular_compositions: NDArray[Any],
    trajectory_identity: str,
) -> TargetResult:
    fit = fit_r1_matlab_historical(boundary_compositions, trajectory_identity)
    if fit.dominant_centroid is None:
        return TargetResult(R1_TARGET_ID, fit.status, None, None, None, fit.selected_k)
    scores, labels = label_against_reference(molecular_compositions, fit.dominant_centroid)
    return TargetResult(
        R1_TARGET_ID,
        "ELIGIBLE_UNIQUE_RECURRING_COMPTYPE",
        np.asarray(labels, dtype=bool),
        np.asarray(scores, dtype=np.float64),
        np.asarray(fit.dominant_centroid, dtype=np.float64)[None, :],
        fit.selected_k,
    )


def u2_target(
    boundary_compositions: NDArray[Any],
    molecular_compositions: NDArray[Any],
    trajectory_identity: str,
) -> TargetResult:
    result = materialize_u2(boundary_compositions, molecular_compositions, trajectory_identity)
    if result.molecular_labels is None or result.molecular_scores is None or result.recurring_centroids is None:
        return TargetResult(U2_TARGET_ID, result.status, None, None, None, result.fit.selected_k)
    return TargetResult(
        U2_TARGET_ID,
        result.status,
        np.asarray(result.molecular_labels, dtype=bool),
        np.asarray(result.molecular_scores, dtype=np.float64),
        np.asarray(result.recurring_centroids, dtype=np.float64),
        result.fit.selected_k,
    )


def target_geometry(labels: NDArray[Any] | None, total: int) -> dict[str, Any]:
    cutoff = math.floor(0.25 * total)
    if labels is None:
        return {
            "defined": False, "wholeOccupancy": None, "prefixOccupancy": None,
            "suffixOccupancy": None, "firstOnset": None, "normalizedFirstOnset": None,
            "noOnsetBeforeCutoff": None, "firstOnsetInSuffix": None,
            "suffixPositiveEpisodes": None, "suffixNegativeEpisodes": None,
            "suffixConstantPositive": None, "suffixConstantNegative": None,
        }
    y = np.asarray(labels, dtype=bool)
    onset = np.flatnonzero(y)
    suffix = y[cutoff:]

    def episodes(values: NDArray[np.bool_], state: bool) -> int:
        if not len(values):
            return 0
        return int(np.count_nonzero((values == state) & np.concatenate(([True], values[1:] != values[:-1]))))

    return {
        "defined": True,
        "wholeOccupancy": float(y.mean()),
        "prefixOccupancy": float(y[:cutoff].mean()),
        "suffixOccupancy": float(suffix.mean()),
        "firstOnset": None if not onset.size else int(onset[0]),
        "normalizedFirstOnset": None if not onset.size else float(onset[0] / max(total - 1, 1)),
        "noOnsetBeforeCutoff": bool(not np.any(y[:cutoff])),
        "firstOnsetInSuffix": bool(onset.size and onset[0] >= cutoff),
        "suffixPositiveEpisodes": episodes(suffix, True),
        "suffixNegativeEpisodes": episodes(suffix, False),
        "suffixConstantPositive": bool(np.all(suffix)),
        "suffixConstantNegative": bool(np.all(~suffix)),
    }


def train_and_predict(
    tensor: dict[str, NDArray[Any]],
    split: pd.DataFrame,
    candidate_id: str,
    repetition: int,
    *,
    model_seed: int,
    target_override: dict[str, NDArray[Any]] | None = None,
) -> tuple[Any, NDArray[np.float64], Any, NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    fit = split_indices(split, repetition, "FIT")
    validation = split_indices(split, repetition, "VALIDATION")
    test = split_indices(split, repetition, "TEST")
    target = tensor["target"] if target_override is None else target_override["target"]
    target_mask = tensor["targetMask"] if target_override is None else target_override["targetMask"]
    scaler = fit_channel_scaler(tensor["values"][fit], tensor["channelMask"][fit])
    scaled = apply_channel_scaler(tensor["values"], tensor["channelMask"], scaler)
    result = train_masked_mlp(
        scaled[fit], tensor["channelMask"][fit], tensor["timeMask"][fit],
        target[fit], target_mask[fit],
        scaled[validation], tensor["channelMask"][validation], tensor["timeMask"][validation],
        target[validation], target_mask[validation], model_seed=model_seed,
    )
    probabilities = predict_probabilities(
        result.model, scaled[test], tensor["channelMask"][test], tensor["timeMask"][test]
    )
    return result, probabilities, scaler, fit, validation, test


__all__ = [name for name in globals() if name.isupper()] + [
    "TargetResult", "array_sha256", "normalized_compositions", "incoming_h",
    "build_feature", "combine_scalar_features", "build_target_tensor", "source_values",
    "s16_source_seed", "s16_model_seed", "split_indices", "extended_binary_metrics",
    "holm_adjust", "matrix_bootstrap_metric_difference", "paper_interval_overlap",
    "geometry_gate", "r1_target", "u2_target", "target_geometry", "train_and_predict",
    "build_split_manifest", "fit_channel_scaler", "apply_channel_scaler", "split_summary",
    "parameter_count", "direct_union_scores", "R1_ID", "U2_ID",
]
