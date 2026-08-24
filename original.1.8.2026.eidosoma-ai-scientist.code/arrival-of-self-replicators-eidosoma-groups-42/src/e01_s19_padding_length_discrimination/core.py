"""Outcome-blind helpers for E01/S19-L15.

The scientific question is deliberately narrow: on a new matrix cohort, can
the exact S16 task and model reproduce Figure 5 when target padding is treated
as ordinary class-zero data?  This module contains only deterministic
contracts and array helpers; it performs no filesystem I/O or outcome-guided
selection.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

VERSION = "E01-S19-L15-UNTOUCHED-PADDING-LENGTH-PANEL-DISCRIMINATION-v1.0.0"
RESEARCH_STEP_ID = "S19-L15"
MATRIX_COUNT = 200
REPETITIONS = 10
FIT_COUNT = 128
VALIDATION_COUNT = 32
TEST_COUNT = 40
CANDIDATE_IDS = ("CANDIDATE_2", "CANDIDATE_3")

P1 = "P1_PHIRL_EMERGENCE_COMPLETED_FIT"
P2 = "P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY"
B1 = "B1_COMPOSITION_CHANGE"
B2 = "B2_RAW_COMPOSITIONS"
B3 = "B3_MOLECULAR_FLUXES"
B4 = "B4_ADJACENT_H"
D0 = "D0_MAJORITY_DUMMY"
D1 = "D1_INPUT_LENGTH_ONLY_LOGISTIC"
D2 = "D2_DETERMINISTIC_PADDING_BOUNDARY"
D3 = "D3_TIME_ONLY_LOGISTIC"
LEARNED_FEATURES = (P1, P2, B1, B2, B3, B4)
PAPER_FEATURES = (P1, B1, B2, B3, D0)

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


def seed_bytes(root_hex: str, *parts: object) -> bytes:
    """Return canonical domain-separated seed material."""

    if len(root_hex) != 64 or any(ch not in "0123456789abcdef" for ch in root_hex):
        raise ValueError("root must be lowercase 256-bit hex")
    return "\x1f".join((VERSION, root_hex, *map(str, parts))).encode("utf-8")


def seed128(root_hex: str, *parts: object) -> int:
    """Return deterministic PCG64DXSM-compatible 128-bit seed."""

    return int.from_bytes(
        hashlib.sha256(seed_bytes(root_hex, *parts)).digest()[:16], "big"
    )


def torch_seed(root_hex: str, *parts: object) -> int:
    """Return deterministic nonnegative PyTorch seed."""

    return int.from_bytes(
        hashlib.sha256(seed_bytes(root_hex, *parts)).digest()[:8], "big"
    ) % (2**63 - 1)


def array_sha256(values: NDArray[Any]) -> str:
    """Hash dtype, shape and contiguous bytes."""

    value = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(b"\0")
    digest.update(str(tuple(value.shape)).encode())
    digest.update(b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()


def build_split_manifest(root_hex: str) -> pd.DataFrame:
    """Create ten outcome-blind 128/32/40 matrix-level repetitions."""

    rows: list[dict[str, Any]] = []
    all_indices = np.arange(MATRIX_COUNT, dtype=np.int64)
    for repetition in range(REPETITIONS):
        test_seed = seed128(root_hex, "split", repetition, "test")
        validation_seed = seed128(root_hex, "split", repetition, "validation")
        test_rng = np.random.Generator(np.random.PCG64DXSM(test_seed))
        validation_rng = np.random.Generator(np.random.PCG64DXSM(validation_seed))
        test = np.sort(test_rng.choice(all_indices, TEST_COUNT, replace=False))
        train_validation = np.setdiff1d(all_indices, test, assume_unique=True)
        validation = np.sort(
            validation_rng.choice(train_validation, VALIDATION_COUNT, replace=False)
        )
        fit = np.setdiff1d(train_validation, validation, assume_unique=True)
        roles = {
            **{int(index): "FIT" for index in fit},
            **{int(index): "VALIDATION" for index in validation},
            **{int(index): "TEST" for index in test},
        }
        for matrix_index in all_indices:
            rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "repetitionId": repetition,
                    "matrixIndex": int(matrix_index),
                    "splitRole": roles[int(matrix_index)],
                    "testSeed128": str(test_seed),
                    "validationSeed128": str(validation_seed),
                    "modelSeedCandidate2": torch_seed(
                        root_hex, "model", "CANDIDATE_2", repetition
                    ),
                    "modelSeedCandidate3": torch_seed(
                        root_hex, "model", "CANDIDATE_3", repetition
                    ),
                    "outcomeStratified": False,
                    "candidateFeaturePairing": True,
                }
            )
    frame = pd.DataFrame(rows)
    validate_split_manifest(frame)
    return frame


def validate_split_manifest(frame: pd.DataFrame) -> None:
    """Raise on cardinality, overlap, pairing, or seed defects."""

    if len(frame) != MATRIX_COUNT * REPETITIONS:
        raise ValueError("split manifest cardinality mismatch")
    if frame.duplicated(["repetitionId", "matrixIndex"]).any():
        raise ValueError("duplicate repetition/matrix identity")
    test_sets: set[tuple[int, ...]] = set()
    for repetition, group in frame.groupby("repetitionId", sort=True):
        if int(repetition) not in range(REPETITIONS):
            raise ValueError("unexpected repetition")
        if set(group["matrixIndex"].astype(int)) != set(range(MATRIX_COUNT)):
            raise ValueError("matrix coverage mismatch")
        counts = group["splitRole"].value_counts().to_dict()
        if counts != {
            "FIT": FIT_COUNT,
            "TEST": TEST_COUNT,
            "VALIDATION": VALIDATION_COUNT,
        }:
            raise ValueError(f"role counts changed: {counts}")
        if (
            group["outcomeStratified"].any()
            or not group["candidateFeaturePairing"].all()
        ):
            raise ValueError("split pairing/stratification contract violated")
        test_sets.add(
            tuple(
                sorted(
                    group.loc[group["splitRole"].eq("TEST"), "matrixIndex"].astype(int)
                )
            )
        )
    if len(test_sets) != REPETITIONS:
        raise ValueError("duplicate test sets")


def split_indices(frame: pd.DataFrame, repetition: int, role: str) -> NDArray[np.int64]:
    """Return sorted matrix indices for one split role."""

    return np.sort(
        frame.loc[
            frame["repetitionId"].eq(repetition) & frame["splitRole"].eq(role),
            "matrixIndex",
        ].to_numpy(dtype=np.int64)
    )


def normalized_compositions(states: NDArray[Any]) -> NDArray[np.float64]:
    """Close positive-mass integer states to relative composition."""

    value = np.asarray(states, dtype=np.float64)
    masses = value.sum(axis=1)
    if value.ndim != 2 or value.shape[1] != 100 or np.any(masses <= 0):
        raise ValueError("states must be positive-mass [time,100]")
    return value / masses[:, None]


def incoming_h(compositions: NDArray[Any]) -> NDArray[np.float64]:
    """Replay the frozen S16 adjacent-incoming H history."""

    value = np.asarray(compositions, dtype=np.float64)
    if value.ndim != 2 or len(value) < 2:
        raise ValueError("incoming H requires at least two compositions")
    norms = np.linalg.norm(value, axis=1)
    if np.any(norms <= 0):
        raise ValueError("zero composition norm")
    unit = value / norms[:, None]
    adjacent = np.sum(unit[:-1] * unit[1:], axis=1)
    return np.concatenate(([adjacent[0]], adjacent)).astype(np.float64)


def included_mask(valid_mask: NDArray[Any], include_padding: bool) -> NDArray[np.bool_]:
    """Return valid-only or all-cell mask."""

    mask = np.asarray(valid_mask, dtype=bool)
    return np.ones_like(mask, dtype=bool) if include_padding else mask.copy()


def mask_pair(
    valid_mask: NDArray[Any], condition_id: str
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return loss and score masks for one registered condition."""

    if condition_id not in MASK_CONTRACT:
        raise ValueError(f"unregistered mask condition: {condition_id}")
    train_padding, score_padding = MASK_CONTRACT[condition_id]
    return included_mask(valid_mask, train_padding), included_mask(
        valid_mask, score_padding
    )


def infer_output_length(cutoff: NDArray[Any] | int) -> NDArray[np.int64]:
    """Frozen midpoint solution to T in {4c,...,4c+3}: m=3c+2."""

    value = np.asarray(cutoff, dtype=np.int64)
    if np.any(value < 0):
        raise ValueError("negative cutoff")
    return (3 * value + 2).astype(np.int64)


def padding_identity(valid_prevalence: float, valid_fraction: float) -> float:
    """Return the expected padded prevalence."""

    if not 0.0 <= valid_prevalence <= 1.0 or not 0.0 <= valid_fraction <= 1.0:
        raise ValueError("probabilities outside [0,1]")
    return float(valid_prevalence * valid_fraction)


def accuracy_decomposition(
    target: NDArray[Any], probability: NDArray[Any], valid_mask: NDArray[Any]
) -> dict[str, float]:
    """Decompose all-cell binary accuracy into real and padding cells."""

    y = np.asarray(target, dtype=bool)
    p = np.asarray(probability, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if y.shape != p.shape or y.shape != valid.shape:
        raise ValueError("accuracy decomposition shape mismatch")
    predicted = p >= 0.5
    correct = predicted == y
    q = float(valid.mean())
    valid_accuracy = float(correct[valid].mean()) if np.any(valid) else float("nan")
    padding_accuracy = float(correct[~valid].mean()) if np.any(~valid) else float("nan")
    all_accuracy = float(correct.mean())
    reconstructed = q * valid_accuracy + (1.0 - q) * padding_accuracy
    return {
        "validFraction": q,
        "allCellAccuracy": all_accuracy,
        "validCellAccuracy": valid_accuracy,
        "paddingCellAccuracy": padding_accuracy,
        "reconstructedAccuracy": float(reconstructed),
        "absoluteError": float(abs(all_accuracy - reconstructed)),
        "correctFromPaddingFraction": float(
            correct[~valid].sum() / max(correct.sum(), 1)
        ),
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "accuracy_decomposition",
    "array_sha256",
    "build_split_manifest",
    "included_mask",
    "incoming_h",
    "infer_output_length",
    "mask_pair",
    "normalized_compositions",
    "padding_identity",
    "seed128",
    "seed_bytes",
    "split_indices",
    "torch_seed",
    "validate_split_manifest",
]
