"""Outcome-blind numerical contracts for E01/S19-L14.

This module deliberately contains no filesystem I/O.  It composes the frozen
S16 tensor/model implementation with the four preregistered train/score mask
semantics and the fixed length-only diagnostics used by L14.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from e01_prediction_reconstruction.core import MAX_TARGET_LENGTH
from e01_s19_figure5_prediction.core import extended_binary_metrics

VERSION = "E01-S19-L14-FIGURE5-PADDING-LENGTH-LEAKAGE-RECONSTRUCTION-v1.0.0"
LOOP_ID = "S19-L14"
ROOT_SEED_HEX = "767f77b086911c523815c912522bb727107dfec4d9dde425c7713784d2d3f04f"

CANDIDATE_IDS = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")

S00 = "S00_MASKED_TRAIN_MASKED_SCORE"
S01 = "S01_MASKED_TRAIN_UNMASKED_SCORE"
S10 = "S10_UNMASKED_TRAIN_MASKED_SCORE"
S11 = "S11_UNMASKED_TRAIN_UNMASKED_SCORE"
MASK_CONDITIONS = (S00, S01, S10, S11)
MASK_CONTRACT = {
    S00: (False, False),
    S01: (False, True),
    S10: (True, False),
    S11: (True, True),
}

P1 = "P1_PHIRL_EMERGENCE_COMPLETED_FIT"
P2 = "P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY"
B1 = "B1_COMPOSITION_CHANGE"
B2 = "B2_RAW_COMPOSITIONS"
B3 = "B3_MOLECULAR_FLUXES"
B4 = "B4_ADJACENT_H"
D0 = "D0_MAJORITY_DUMMY"
D1 = "D1_INPUT_LENGTH_ONLY"
D2 = "D2_DETERMINISTIC_PADDING_BOUNDARY"
D3 = "D3_TIME_ONLY"
LEARNED_FEATURES = (P1, P2, B1, B2, B3, B4)
DIAGNOSTICS = (D0, D1, D2, D3)


def seed128(*parts: object) -> int:
    """Return a domain-separated 128-bit seed."""

    material = "\x1f".join([VERSION, ROOT_SEED_HEX, *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def array_sha256(values: NDArray[Any]) -> str:
    """Hash shape, dtype, and canonical contiguous bytes."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def pixel_to_accuracy(row: float, rows: NDArray[Any], values: NDArray[Any]) -> float:
    """Map an image row to accuracy using the frozen linear tick calibration."""

    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("pixel calibration needs matched tick vectors")
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    if float(np.max(np.abs(predicted - y))) > 1e-12:
        raise ValueError("Figure-5 ticks are not linear under the frozen calibration")
    return float(slope * float(row) + intercept)


def paper_interval(
    row: float, uncertainty_pixels: float, rows: NDArray[Any], values: NDArray[Any]
) -> tuple[float, float]:
    """Return the ordered accuracy interval induced by pixel uncertainty."""

    ends = [
        pixel_to_accuracy(row - uncertainty_pixels, rows, values),
        pixel_to_accuracy(row + uncertainty_pixels, rows, values),
    ]
    return float(min(ends)), float(max(ends))


def padded_target(
    valid_labels: list[NDArray[Any]], *, width: int = MAX_TARGET_LENGTH
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Right-pad variable-length binary sequences with literal zero targets."""

    if not valid_labels:
        raise ValueError("at least one target sequence is required")
    if any(np.asarray(row).ndim != 1 for row in valid_labels):
        raise ValueError("target sequences must be one-dimensional")
    lengths = np.asarray([len(row) for row in valid_labels], dtype=np.int64)
    if np.any(lengths <= 0) or int(lengths.max()) > width:
        raise ValueError("target sequence length is outside the frozen capacity")
    target = np.zeros((len(valid_labels), width), dtype=np.float64)
    mask = np.zeros_like(target, dtype=bool)
    for index, row in enumerate(valid_labels):
        y = np.asarray(row, dtype=bool)
        target[index, : len(y)] = y.astype(np.float64)
        mask[index, : len(y)] = True
    return target, mask


def padding_arithmetic(
    target: NDArray[Any], valid_mask: NDArray[Any]
) -> dict[str, float | int]:
    """Compute and verify the L14 prevalence identity over a fixed-width tensor."""

    y = np.asarray(target, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if y.shape != mask.shape or y.ndim != 2 or y.size == 0:
        raise ValueError("target and mask must be matched nonempty matrices")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("target must be binary")
    if np.any(y[~mask] != 0.0):
        raise ValueError("padding must be literal zero")
    valid_count = int(mask.sum())
    all_count = int(mask.size)
    if valid_count == 0:
        raise ValueError("no valid target cells")
    q = valid_count / all_count
    valid_prevalence = float(y[mask].mean())
    padded_prevalence = float(y.mean())
    identity_error = abs(padded_prevalence - valid_prevalence * q)
    if identity_error > 1e-15:
        raise RuntimeError("padded prevalence identity failed")
    return {
        "matrixCount": int(y.shape[0]),
        "tensorWidth": int(y.shape[1]),
        "validCellCount": valid_count,
        "paddingCellCount": all_count - valid_count,
        "allCellCount": all_count,
        "validFraction": float(q),
        "paddingFraction": float(1.0 - q),
        "validPrevalence": valid_prevalence,
        "paddedPrevalence": padded_prevalence,
        "validOnlyDummyAccuracy": float(max(valid_prevalence, 1.0 - valid_prevalence)),
        "paddedDummyAccuracy": float(max(padded_prevalence, 1.0 - padded_prevalence)),
        "identityAbsoluteError": float(identity_error),
    }


def loss_mask(valid_mask: NDArray[Any], condition_id: str) -> NDArray[np.bool_]:
    """Return the preregistered loss mask for one train/score condition."""

    if condition_id not in MASK_CONTRACT:
        raise ValueError(f"unknown mask condition {condition_id}")
    train_padding, _ = MASK_CONTRACT[condition_id]
    mask = np.asarray(valid_mask, dtype=bool)
    return np.ones_like(mask, dtype=bool) if train_padding else mask.copy()


def score_mask(valid_mask: NDArray[Any], condition_id: str) -> NDArray[np.bool_]:
    """Return the preregistered evaluation mask for one condition."""

    if condition_id not in MASK_CONTRACT:
        raise ValueError(f"unknown mask condition {condition_id}")
    _, score_padding = MASK_CONTRACT[condition_id]
    mask = np.asarray(valid_mask, dtype=bool)
    return np.ones_like(mask, dtype=bool) if score_padding else mask.copy()


def included_training_prevalence(
    target: NDArray[Any], valid_mask: NDArray[Any], condition_id: str
) -> float:
    """Calculate the fit-only class prevalence under the named loss mask."""

    y = np.asarray(target, dtype=bool)
    mask = loss_mask(valid_mask, condition_id)
    if not np.any(mask):
        raise ValueError("training prevalence has no included cells")
    return float(y[mask].mean())


def constant_probability(
    prevalence: float, shape: tuple[int, int]
) -> NDArray[np.float64]:
    """Construct the frozen majority-dummy probability tensor."""

    if not 0.0 <= prevalence <= 1.0:
        raise ValueError("prevalence outside [0,1]")
    return np.full(shape, prevalence, dtype=np.float64)


def metric_views(
    target: NDArray[Any], probability: NDArray[Any], valid_mask: NDArray[Any]
) -> dict[str, dict[str, Any]]:
    """Return all-cell, valid-cell, and padding-cell metric views."""

    y = np.asarray(target, dtype=bool)
    p = np.asarray(probability, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if y.shape != p.shape or y.shape != valid.shape:
        raise ValueError("metric view arrays have incompatible shapes")
    return {
        "all": extended_binary_metrics(y.reshape(-1), p.reshape(-1)),
        "valid": extended_binary_metrics(y[valid], p[valid]),
        "padding": extended_binary_metrics(y[~valid], p[~valid]),
    }


def accuracy_decomposition(
    target: NDArray[Any], probability: NDArray[Any], valid_mask: NDArray[Any]
) -> dict[str, float | int]:
    """Verify all-cell accuracy as a mixture of valid and padding accuracy."""

    y = np.asarray(target, dtype=bool)
    predicted = np.asarray(probability, dtype=np.float64) >= 0.5
    valid = np.asarray(valid_mask, dtype=bool)
    if y.shape != predicted.shape or y.shape != valid.shape:
        raise ValueError("accuracy arrays have incompatible shapes")
    if not np.any(valid) or not np.any(~valid):
        raise ValueError("decomposition requires valid and padding cells")
    all_accuracy = float(np.mean(predicted == y))
    valid_accuracy = float(np.mean(predicted[valid] == y[valid]))
    padding_accuracy = float(np.mean(predicted[~valid] == y[~valid]))
    q = float(valid.mean())
    reconstructed = q * valid_accuracy + (1.0 - q) * padding_accuracy
    correct = predicted == y
    correct_total = int(correct.sum())
    correct_padding = int(np.count_nonzero(correct & ~valid))
    error = abs(all_accuracy - reconstructed)
    if error > 1e-12:
        raise RuntimeError("accuracy decomposition failed")
    return {
        "allCellAccuracy": all_accuracy,
        "validCellAccuracy": valid_accuracy,
        "paddingCellAccuracy": padding_accuracy,
        "validFraction": q,
        "reconstructedAllCellAccuracy": float(reconstructed),
        "absoluteError": float(error),
        "allMinusValidAccuracy": float(all_accuracy - valid_accuracy),
        "correctPredictionCount": correct_total,
        "correctPaddingPredictionCount": correct_padding,
        "fractionCorrectFromPadding": float(correct_padding / correct_total)
        if correct_total
        else math.nan,
    }


def infer_output_length(cutoff: NDArray[Any]) -> NDArray[np.int64]:
    """Apply the frozen midpoint total-length tie rule: m_hat = 3c + 2."""

    c = np.asarray(cutoff, dtype=np.int64)
    if c.ndim != 1 or np.any(c < 1):
        raise ValueError("cutoff must be a positive vector")
    result = 3 * c + 2
    return np.minimum(result, MAX_TARGET_LENGTH).astype(np.int64)


def boundary_predictions(
    cutoff: NDArray[Any], width: int, valid_majority_positive: bool
) -> NDArray[np.float64]:
    """Predict the fit-majority valid class until the inferred pad boundary."""

    inferred = infer_output_length(cutoff)
    position = np.arange(width, dtype=np.int64)[None, :]
    inside = position < inferred[:, None]
    return np.where(inside, float(valid_majority_positive), 0.0).astype(np.float64)


def permute_valid_labels_preserving_padding(
    target: NDArray[Any], valid_mask: NDArray[Any], *, seed_identity: tuple[object, ...]
) -> NDArray[np.float64]:
    """Permute valid labels while leaving literal padding zeros fixed."""

    y = np.asarray(target, dtype=np.float64).copy()
    mask = np.asarray(valid_mask, dtype=bool)
    rng = np.random.Generator(np.random.PCG64DXSM(seed128(*seed_identity)))
    y[mask] = y[mask][rng.permutation(int(mask.sum()))]
    y[~mask] = 0.0
    return y


def permute_valid_time(
    values: NDArray[Any],
    channel_mask: NDArray[Any],
    time_mask: NDArray[Any],
    *,
    seed_identity: tuple[object, ...],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Permute valid feature rows within each matrix while retaining length."""

    x = np.asarray(values, dtype=np.float64).copy()
    channels = np.asarray(channel_mask, dtype=bool).copy()
    valid_time = np.asarray(time_mask, dtype=bool)
    rng = np.random.Generator(np.random.PCG64DXSM(seed128(*seed_identity)))
    for index in range(len(x)):
        positions = np.flatnonzero(valid_time[index])
        order = rng.permutation(len(positions))
        original_values = x[index, positions].copy()
        original_channels = channels[index, positions].copy()
        x[index, positions] = original_values[order]
        channels[index, positions] = original_channels[order]
    return x, channels


def obfuscate_padded_input_values(
    scaled_values: NDArray[Any],
    time_mask: NDArray[Any],
    *,
    seed_identity: tuple[object, ...],
) -> NDArray[np.float64]:
    """Replace padded input values with matched N(0,1) noise; masks stay frozen."""

    x = np.asarray(scaled_values, dtype=np.float64).copy()
    valid_time = np.asarray(time_mask, dtype=bool)
    if x.ndim != 3 or valid_time.shape != x.shape[:2]:
        raise ValueError("obfuscation input shapes do not match")
    rng = np.random.Generator(np.random.PCG64DXSM(seed128(*seed_identity)))
    pad = np.broadcast_to(~valid_time[:, :, None], x.shape)
    x[pad] = rng.normal(0.0, 1.0, size=int(pad.sum()))
    return x


def interval_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    """Return whether two closed intervals intersect."""

    return bool(max(left[0], right[0]) <= min(left[1], right[1]))
