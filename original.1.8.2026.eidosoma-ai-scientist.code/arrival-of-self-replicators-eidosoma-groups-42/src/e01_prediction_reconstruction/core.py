"""Outcome-blind contracts and deterministic helpers for E01/S16.

The functions in this module implement the single preregistered masked tensor
layout, split hierarchy, MLP, training procedure, and metric definitions.  They
contain no filesystem I/O and do not choose a method from observed outcomes.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from torch import nn

VERSION = "E01-S16-FIRST-QUARTER-PREDICTION-RECONSTRUCTION-v1.0.0"
RESEARCH_STEP_ID = "S16"
ROOT_SEED_HEX = "9a8456c3204eea08a83a7a04d64b4097f7d922fe9c21b8deea0839127f66c2b1"
CANDIDATE_IDS = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
RETROSPECTIVE_MODE = "RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE"
CUTOFF_MODE = "CUTOFF_CAUSAL_FIRST_QUARTER_ONLY"
TEMPORAL_MODES = (RETROSPECTIVE_MODE, CUTOFF_MODE)
LEARNED_FEATURE_IDS = (
    "PHIRL_EMERGENCE",
    "COMPOSITION_CHANGE_L2",
    "RAW_COUNTS",
    "NET_COUNT_FLUX",
    "EXACT_H_HISTORY",
)
DUMMY_FEATURE_ID = "MAJORITY_DUMMY"
FEATURE_IDS = (*LEARNED_FEATURE_IDS, DUMMY_FEATURE_ID)
MAX_INPUT_LENGTH = 367
MAX_TARGET_LENGTH = 1101
FEATURE_CHANNEL_CAPACITY = 100
STEP_EMBED_DIM = 8
HIDDEN_DIM = 64
EXPECTED_PARAMETER_COUNT = 288_789


def seed_material(*identity: object) -> bytes:
    """Canonical domain-separated seed material."""

    return "\x1f".join([VERSION, ROOT_SEED_HEX, *map(str, identity)]).encode()


def derive_seed128(*identity: object) -> int:
    """Return a deterministic 128-bit integer for NumPy PCG64DXSM."""

    return int.from_bytes(hashlib.sha256(seed_material(*identity)).digest()[:16], "big")


def derive_torch_seed(*identity: object) -> int:
    """Return a deterministic nonnegative seed accepted by torch.manual_seed."""

    return int.from_bytes(hashlib.sha256(seed_material(*identity)).digest()[:8], "big") % (
        2**63 - 1
    )


def build_split_manifest() -> pd.DataFrame:
    """Build the ten frozen, outcome-blind matrix-level split assignments."""

    rows: list[dict[str, Any]] = []
    for repetition in range(10):
        split_seed = derive_seed128("split", repetition, "test")
        validation_seed = derive_seed128("split", repetition, "validation")
        test_rng = np.random.Generator(np.random.PCG64DXSM(split_seed))
        validation_rng = np.random.Generator(np.random.PCG64DXSM(validation_seed))
        all_indices = np.arange(100, dtype=np.int64)
        test = np.sort(test_rng.choice(all_indices, size=20, replace=False))
        train_validation = np.setdiff1d(all_indices, test, assume_unique=True)
        validation = np.sort(
            validation_rng.choice(train_validation, size=16, replace=False)
        )
        fit = np.setdiff1d(train_validation, validation, assume_unique=True)
        roles = {
            **{int(value): "FIT" for value in fit},
            **{int(value): "VALIDATION" for value in validation},
            **{int(value): "TEST" for value in test},
        }
        model_seed_base = derive_torch_seed("model", repetition)
        model_seed_candidate_02 = derive_torch_seed(
            "model", "S12F-CANDIDATE-02", repetition
        )
        model_seed_candidate_03 = derive_torch_seed(
            "model", "S12F-CANDIDATE-03", repetition
        )
        bootstrap_seed = derive_seed128("matrix_cluster_bootstrap", repetition)
        for matrix_index in range(100):
            rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "repetitionId": repetition,
                    "matrixIndex": matrix_index,
                    "splitRole": roles[matrix_index],
                    "testSeed128": str(split_seed),
                    "validationSeed128": str(validation_seed),
                    "modelSeedBase": model_seed_base,
                    "modelSeedCandidate02": model_seed_candidate_02,
                    "modelSeedCandidate03": model_seed_candidate_03,
                    "bootstrapSeed128": str(bootstrap_seed),
                    "candidateFeatureModePairing": True,
                    "outcomeStratified": False,
                }
            )
    frame = pd.DataFrame(rows)
    validate_split_manifest(frame)
    return frame


def validate_split_manifest(frame: pd.DataFrame) -> None:
    """Raise if any frozen split or pairing invariant is violated."""

    expected_columns = {
        "researchStepId",
        "repetitionId",
        "matrixIndex",
        "splitRole",
        "testSeed128",
        "validationSeed128",
        "modelSeedBase",
        "modelSeedCandidate02",
        "modelSeedCandidate03",
        "bootstrapSeed128",
        "candidateFeatureModePairing",
        "outcomeStratified",
    }
    if set(frame.columns) != expected_columns or len(frame) != 1_000:
        raise ValueError("split manifest schema/cardinality mismatch")
    if frame.duplicated(["repetitionId", "matrixIndex"]).any():
        raise ValueError("duplicate repetition/matrix split identity")
    for repetition, group in frame.groupby("repetitionId", sort=True):
        if repetition not in range(10) or set(group["matrixIndex"]) != set(range(100)):
            raise ValueError("split matrix coverage mismatch")
        counts = group["splitRole"].value_counts().to_dict()
        if counts != {"FIT": 64, "TEST": 20, "VALIDATION": 16}:
            raise ValueError(f"split role cardinality mismatch: {counts}")
        if group["outcomeStratified"].any() or not group[
            "candidateFeatureModePairing"
        ].all():
            raise ValueError("split outcome/pairing contract violated")
    test_sets = {
        tuple(
            sorted(
                frame.loc[
                    frame["repetitionId"].eq(repetition)
                    & frame["splitRole"].eq("TEST"),
                    "matrixIndex",
                ].tolist()
            )
        )
        for repetition in range(10)
    }
    if len(test_sets) != 10:
        raise ValueError("independently seeded repetitions produced duplicate test sets")


@dataclass(frozen=True, slots=True)
class ChannelScaler:
    """Training-fit-only per-channel scaler."""

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    valid_count: NDArray[np.int64]


def fit_channel_scaler(
    values: NDArray[np.float64], channel_mask: NDArray[np.bool_]
) -> ChannelScaler:
    """Fit channel means/scales using only true feature-mask cells."""

    x = np.asarray(values, dtype=np.float64)
    mask = np.asarray(channel_mask, dtype=bool)
    if x.shape != mask.shape or x.ndim != 3 or x.shape[2] != FEATURE_CHANNEL_CAPACITY:
        raise ValueError("scaler expects matched [batch,time,100] arrays")
    count = mask.sum(axis=(0, 1), dtype=np.int64)
    total = np.where(mask, x, 0.0).sum(axis=(0, 1), dtype=np.float64)
    mean = np.divide(total, count, out=np.zeros(100), where=count > 0)
    centered = np.where(mask, x - mean[None, None, :], 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=(0, 1), dtype=np.float64),
        count,
        out=np.zeros(100),
        where=count > 0,
    )
    scale = np.sqrt(variance)
    scale[(count == 0) | (scale < 1e-12)] = 1.0
    return ChannelScaler(mean.astype(np.float64), scale.astype(np.float64), count)


def apply_channel_scaler(
    values: NDArray[np.float64],
    channel_mask: NDArray[np.bool_],
    scaler: ChannelScaler,
) -> NDArray[np.float64]:
    """Apply a scaler and restore unavailable/padded cells to exact zero."""

    x = np.asarray(values, dtype=np.float64)
    mask = np.asarray(channel_mask, dtype=bool)
    scaled = (x - scaler.mean[None, None, :]) / scaler.scale[None, None, :]
    scaled[~mask] = 0.0
    if not np.all(np.isfinite(scaled)):
        raise ValueError("scaled tensor is nonfinite")
    return np.ascontiguousarray(scaled, dtype=np.float64)


class MaskedSequenceMLP(nn.Module):
    """The single frozen same-capacity MLP used by every learned feature."""

    def __init__(self) -> None:
        super().__init__()
        self.step_encoder = nn.Linear(200, STEP_EMBED_DIM, bias=True)
        sequence_width = MAX_INPUT_LENGTH * STEP_EMBED_DIM + MAX_INPUT_LENGTH
        self.hidden_1 = nn.Linear(sequence_width, HIDDEN_DIM, bias=True)
        self.hidden_2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True)
        self.output = nn.Linear(HIDDEN_DIM, MAX_TARGET_LENGTH, bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.10)

    def forward(
        self,
        values: torch.Tensor,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        if values.ndim != 3 or tuple(values.shape[1:]) != (
            MAX_INPUT_LENGTH,
            FEATURE_CHANNEL_CAPACITY,
        ):
            raise ValueError("unexpected frozen input value shape")
        step_input = torch.cat((values, channel_mask), dim=-1)
        encoded = self.activation(self.step_encoder(step_input))
        encoded = encoded * time_mask.unsqueeze(-1)
        sequence = torch.cat((encoded.flatten(start_dim=1), time_mask), dim=1)
        hidden = self.dropout(self.activation(self.hidden_1(sequence)))
        hidden = self.dropout(self.activation(self.hidden_2(hidden)))
        return self.output(hidden)


def parameter_count(model: nn.Module) -> int:
    """Count trainable parameters."""

    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def masked_binary_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    """Micro-average BCE over valid target cells only."""

    losses = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    denominator = target_mask.sum()
    if denominator.item() <= 0:
        raise ValueError("masked BCE has zero valid target cells")
    return (losses * target_mask).sum() / denominator


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Deterministic fitted-model payload without a persisted checkpoint."""

    model: MaskedSequenceMLP
    history: pd.DataFrame
    best_epoch: int
    stopped_epoch: int
    best_validation_loss: float


def _as_tensor(value: NDArray[Any]) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).to(dtype=torch.float64)


def train_masked_mlp(
    fit_values: NDArray[np.float64],
    fit_channel_mask: NDArray[np.bool_],
    fit_time_mask: NDArray[np.bool_],
    fit_targets: NDArray[np.float64],
    fit_target_mask: NDArray[np.bool_],
    validation_values: NDArray[np.float64],
    validation_channel_mask: NDArray[np.bool_],
    validation_time_mask: NDArray[np.bool_],
    validation_targets: NDArray[np.float64],
    validation_target_mask: NDArray[np.bool_],
    *,
    model_seed: int,
    maximum_epochs: int = 120,
    patience: int = 15,
    minimum_improvement: float = 1e-5,
) -> TrainingResult:
    """Fit the frozen MLP with full-batch deterministic AdamW."""

    torch.manual_seed(int(model_seed))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = MaskedSequenceMLP().to(dtype=torch.float64, device="cpu")
    if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("frozen MLP parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.0001
    )
    fit_tensors = tuple(
        _as_tensor(value)
        for value in (
            fit_values,
            fit_channel_mask.astype(np.float64),
            fit_time_mask.astype(np.float64),
            fit_targets,
            fit_target_mask.astype(np.float64),
        )
    )
    validation_tensors = tuple(
        _as_tensor(value)
        for value in (
            validation_values,
            validation_channel_mask.astype(np.float64),
            validation_time_mask.astype(np.float64),
            validation_targets,
            validation_target_mask.astype(np.float64),
        )
    )
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(maximum_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        fit_logits = model(*fit_tensors[:3])
        fit_loss = masked_binary_cross_entropy(
            fit_logits, fit_tensors[3], fit_tensors[4]
        )
        fit_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(*validation_tensors[:3])
            validation_loss = masked_binary_cross_entropy(
                validation_logits, validation_tensors[3], validation_tensors[4]
            )
        fit_value = float(fit_loss.detach().cpu())
        validation_value = float(validation_loss.detach().cpu())
        history.append(
            {
                "epoch": epoch,
                "fitLoss": fit_value,
                "validationLoss": validation_value,
            }
        )
        if validation_value < best_loss - minimum_improvement:
            best_loss = validation_value
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    if best_state is None or best_epoch < 0:
        raise RuntimeError("early stopping did not retain a model")
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(
        model=model,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        stopped_epoch=int(history[-1]["epoch"]),
        best_validation_loss=best_loss,
    )


def predict_probabilities(
    model: MaskedSequenceMLP,
    values: NDArray[np.float64],
    channel_mask: NDArray[np.bool_],
    time_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return deterministic sigmoid probabilities from one fitted model."""

    model.eval()
    with torch.no_grad():
        logits = model(
            _as_tensor(values),
            _as_tensor(channel_mask.astype(np.float64)),
            _as_tensor(time_mask.astype(np.float64)),
        )
        probabilities = torch.sigmoid(logits).cpu().numpy()
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("model produced nonfinite probability")
    return np.ascontiguousarray(probabilities, dtype=np.float64)


def expected_calibration_error(
    target: NDArray[np.bool_], probability: NDArray[np.float64], *, bins: int = 10
) -> float:
    """Equal-width expected calibration error."""

    y = np.asarray(target, dtype=bool)
    p = np.asarray(probability, dtype=np.float64)
    if y.size == 0 or p.size != y.size:
        return math.nan
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(p, edges[1:-1], right=False), bins - 1)
    total = float(y.size)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        if np.any(selected):
            error += float(np.sum(selected)) / total * abs(
                float(np.mean(p[selected])) - float(np.mean(y[selected]))
            )
    return float(error)


def binary_metrics(
    target: NDArray[np.bool_], probability: NDArray[np.float64]
) -> dict[str, Any]:
    """Calculate the frozen metric set on an already masked vector."""

    y = np.asarray(target, dtype=bool)
    p = np.asarray(probability, dtype=np.float64)
    if y.size == 0 or y.size != p.size or not np.all(np.isfinite(p)):
        return {
            "validTargetCount": int(y.size),
            "positiveCount": int(y.sum()),
            "prevalence": float(np.mean(y)) if y.size else None,
            "accuracy": None,
            "auroc": None,
            "auprc": None,
            "brier": None,
            "calibrationError": None,
            "balancedAccuracy": None,
            "sensitivity": None,
            "specificity": None,
            "metricStatus": "INELIGIBLE_EMPTY_OR_NONFINITE",
        }
    predicted = p >= 0.5
    tp = int(np.sum(predicted & y))
    tn = int(np.sum(~predicted & ~y))
    fp = int(np.sum(predicted & ~y))
    fn = int(np.sum(~predicted & y))
    both_classes = np.unique(y).size == 2
    return {
        "validTargetCount": int(y.size),
        "positiveCount": int(y.sum()),
        "prevalence": float(np.mean(y)),
        "accuracy": float(np.mean(predicted == y)),
        "auroc": float(roc_auc_score(y, p)) if both_classes else None,
        "auprc": float(average_precision_score(y, p)) if both_classes else None,
        "brier": float(brier_score_loss(y, p)),
        "calibrationError": expected_calibration_error(y, p, bins=10),
        "balancedAccuracy": float(balanced_accuracy_score(y, predicted))
        if both_classes
        else None,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "metricStatus": "ELIGIBLE_BOTH_CLASSES"
        if both_classes
        else "ELIGIBLE_SINGLE_CLASS_PARTIAL_METRICS",
    }


def preonset_masks(
    input_labels: NDArray[np.bool_],
    target_labels: NDArray[np.bool_],
    target_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return run eligibility and target risk masks for strict pre-onset audit."""

    input_y = np.asarray(input_labels, dtype=bool)
    target_y = np.asarray(target_labels, dtype=bool)
    valid = np.asarray(target_mask, dtype=bool)
    if input_y.ndim != 2 or target_y.shape != valid.shape:
        raise ValueError("pre-onset arrays have incompatible shapes")
    eligible = ~np.any(input_y, axis=1)
    risk = np.zeros_like(valid)
    for index in range(len(eligible)):
        if not eligible[index]:
            continue
        valid_indices = np.flatnonzero(valid[index])
        positives = valid_indices[target_y[index, valid_indices]]
        stop = int(positives[0]) if positives.size else int(valid_indices[-1])
        risk[index, valid_indices[valid_indices <= stop]] = True
    return eligible, risk


def split_summary(values: NDArray[np.float64]) -> dict[str, float | int | None]:
    """Mean/median/SD and two-sided Student-t interval across split values."""

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "definedSplitCount": 0,
            "mean": None,
            "median": None,
            "sampleStd": None,
            "lower95": None,
            "upper95": None,
        }
    mean = float(np.mean(x))
    if x.size == 1:
        lower = upper = mean
        sample_std = None
    else:
        sample_std = float(np.std(x, ddof=1))
        half = float(stats.t.ppf(0.975, x.size - 1) * sample_std / np.sqrt(x.size))
        lower, upper = mean - half, mean + half
    return {
        "definedSplitCount": int(x.size),
        "mean": mean,
        "median": float(np.median(x)),
        "sampleStd": sample_std,
        "lower95": lower,
        "upper95": upper,
    }


def matrix_cluster_bootstrap(
    paired_matrix_rows: pd.DataFrame,
    *,
    replicates: int = 4096,
    seed_identity: tuple[object, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bootstrap paired per-matrix accuracy differences by matrix identity."""

    required = {"matrixIndex", "accuracyReference", "accuracyComparator"}
    if not required.issubset(paired_matrix_rows.columns):
        raise ValueError("matrix bootstrap input schema mismatch")
    grouped = (
        paired_matrix_rows.groupby("matrixIndex", as_index=False)[
            ["accuracyReference", "accuracyComparator"]
        ]
        .mean()
        .sort_values("matrixIndex")
    )
    differences = (
        grouped["accuracyReference"].to_numpy(np.float64)
        - grouped["accuracyComparator"].to_numpy(np.float64)
    )
    if not differences.size or not np.all(np.isfinite(differences)):
        raise ValueError("matrix bootstrap has no finite paired identities")
    rng = np.random.Generator(np.random.PCG64DXSM(derive_seed128(*seed_identity)))
    indices = rng.integers(0, differences.size, size=(replicates, differences.size))
    distribution = differences[indices].mean(axis=1)
    observed = float(np.mean(differences))
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    positive_p = float((1 + np.count_nonzero(distribution <= 0.0)) / (replicates + 1))
    frame = pd.DataFrame(
        {
            "replicate": np.arange(replicates, dtype=np.int64),
            "meanPairedMacroAccuracyDifference": distribution,
        }
    )
    summary = {
        "pairedMatrixCount": int(differences.size),
        "observedMeanPairedMacroAccuracyDifference": observed,
        "bootstrapLower95": float(lower),
        "bootstrapUpper95": float(upper),
        "positiveP": positive_p,
        "positiveMatrixCount": int(np.count_nonzero(differences > 0.0)),
    }
    return frame, summary
