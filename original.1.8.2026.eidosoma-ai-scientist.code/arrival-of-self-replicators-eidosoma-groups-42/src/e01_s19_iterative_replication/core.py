"""Outcome-blind contracts and numerical helpers for E01/S19 Loop 1.

This module contains the frozen ranking equation, correlation inference,
spike-descriptor rules, seed hierarchy, and the S16 architecture formula
instantiated at data-determined padding dimensions.  It performs no filesystem
I/O and contains no outcome-selected branch.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy import stats
from torch import nn

from e01_prediction_reconstruction.core import (
    apply_channel_scaler,
    binary_metrics,
    fit_channel_scaler,
)

VERSION = "E01-S19-L01-UNEVALUATED-CLAIM-RECOVERY-v1.0.0"
RESEARCH_STEP_ID = "S19"
LOOP_ID = "S19-L01"
ROOT_SEED_HEX = "d72e1dfd986c367c8481f0f24c925cfbe67817583c429f77b7f80b95663d1a66"
CANDIDATE_IDS = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
PROPORTIONS = (0.10, 0.20, 0.25, 0.33, 0.50)
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
FEATURE_CHANNEL_CAPACITY = 100
STEP_EMBED_DIM = 8
HIDDEN_DIM = 64
BOOTSTRAP_REPLICATES = 4096
PERMUTATION_REPLICATES = 4096
EQUIVALENCE_MARGIN = 0.20


def seed_material(*identity: object) -> bytes:
    """Canonical domain-separated seed bytes."""

    return "\x1f".join([VERSION, ROOT_SEED_HEX, *map(str, identity)]).encode()


def derive_seed128(*identity: object) -> int:
    """Derive a deterministic PCG64DXSM-compatible 128-bit seed."""

    return int.from_bytes(hashlib.sha256(seed_material(*identity)).digest()[:16], "big")


def derive_seed32(*identity: object) -> int:
    """Derive a deterministic legacy NumPy-compatible 32-bit seed."""

    return int.from_bytes(hashlib.sha256(seed_material(*identity)).digest()[:4], "big")


def rank_candidate(scores: dict[str, float], penalties: dict[str, float]) -> float:
    """Return the prospectively frozen candidate-prioritization score.

    Positive dimensions are 0..5.  Penalties are also 0..5 except branchCount,
    which is the declared number of scientific specifications.
    """

    positive = (
        2.0 * scores["sourceGrounding"]
        + 1.5 * scores["paperFingerprintSpecificity"]
        + 1.5 * scores["explanatoryLeverage"]
        + scores["testability"]
        + scores["crossCandidateDiscriminability"]
        + scores["computeEfficiency"]
        + 1.5 * scores["independenceFromPriorOutcomeSelection"]
    )
    penalty = (
        2.0 * penalties["outcomeGuidedThresholdSelection"]
        + 2.0 * penalties["deterministicHReuse"]
        + 2.0 * penalties["completedFitLeakage"]
        + 1.5 * penalties["candidateSpecificSuccess"]
        + 1.5 * penalties["undefinedAuthorSemantics"]
        + 0.5 * max(0.0, penalties["branchCount"] - 1.0)
    )
    return float(positive - penalty)


def holm_adjust(pvalues: Iterable[float | None]) -> list[float | None]:
    """Holm familywise adjustment while retaining undefined entries."""

    values = list(pvalues)
    valid = [(index, float(value)) for index, value in enumerate(values) if value is not None and np.isfinite(value)]
    if not valid:
        return [None] * len(values)
    ordered = sorted(valid, key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, (m - rank) * value)
        adjusted[index] = min(1.0, running)
    return [adjusted.get(index) for index in range(len(values))]


def detectable_correlation(n: int, alpha: float, power: float = 0.80) -> float | None:
    """Approximate minimum detectable |r| via the Fisher-z normal equation."""

    if n <= 3 or not 0 < alpha < 1:
        return None
    z = (stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power)) / math.sqrt(n - 3)
    return float(np.tanh(z))


def _correlation(x: NDArray[np.float64], y: NDArray[np.float64], method: str) -> tuple[float, float]:
    if method == "spearman":
        result = stats.spearmanr(x, y)
    elif method == "pearson":
        result = stats.pearsonr(x, y)
    else:
        raise ValueError(f"unknown correlation method: {method}")
    return float(result.statistic), float(result.pvalue)


def correlation_inference(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    method: str,
    seed_identity: tuple[object, ...],
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    permutation_replicates: int = 0,
) -> dict[str, Any]:
    """Correlation, uncertainty, equivalence, and optional permutation test."""

    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[keep], yv[keep]
    n = int(len(xv))
    base: dict[str, Any] = {
        "definedCount": n,
        "statistic": None,
        "pValue": None,
        "ci95Low": None,
        "ci95High": None,
        "equivalenceCi90Low": None,
        "equivalenceCi90High": None,
        "equivalenceMargin": EQUIVALENCE_MARGIN,
        "equivalentSmallEffect": False,
        "permutationPValue": None,
        "status": "UNDEFINED",
    }
    if n < 4 or np.ptp(xv) <= 1e-15 or np.ptp(yv) <= 1e-15:
        base["reason"] = "fewer_than_four_or_constant_input"
        return base
    statistic, pvalue = _correlation(xv, yv, method)
    if not np.isfinite(statistic):
        base["reason"] = "nonfinite_correlation"
        return base
    rng = np.random.Generator(np.random.PCG64DXSM(derive_seed128(*seed_identity, method, "bootstrap")))
    boot = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        sample = rng.integers(0, n, size=n)
        if np.ptp(xv[sample]) <= 1e-15 or np.ptp(yv[sample]) <= 1e-15:
            boot[index] = np.nan
        else:
            boot[index] = _correlation(xv[sample], yv[sample], method)[0]
    finite = boot[np.isfinite(boot)]
    if len(finite) >= max(100, bootstrap_replicates // 2):
        ci95 = np.quantile(finite, [0.025, 0.975])
        ci90 = np.quantile(finite, [0.05, 0.95])
    elif method == "pearson":
        transformed = np.arctanh(np.clip(statistic, -0.999999, 0.999999))
        se = 1.0 / math.sqrt(n - 3)
        ci95 = np.tanh(transformed + np.array([-1.0, 1.0]) * stats.norm.ppf(0.975) * se)
        ci90 = np.tanh(transformed + np.array([-1.0, 1.0]) * stats.norm.ppf(0.95) * se)
    else:
        ci95 = np.array([np.nan, np.nan])
        ci90 = np.array([np.nan, np.nan])
    permutation_p = None
    if permutation_replicates:
        perm_rng = np.random.Generator(
            np.random.PCG64DXSM(derive_seed128(*seed_identity, method, "permutation"))
        )
        extreme = 0
        for _ in range(permutation_replicates):
            permuted = perm_rng.permutation(yv)
            value = _correlation(xv, permuted, method)[0]
            extreme += int(abs(value) >= abs(statistic))
        permutation_p = float((extreme + 1) / (permutation_replicates + 1))
    equivalent = bool(
        np.all(np.isfinite(ci90))
        and float(ci90[0]) > -EQUIVALENCE_MARGIN
        and float(ci90[1]) < EQUIVALENCE_MARGIN
    )
    base.update(
        {
            "statistic": statistic,
            "pValue": pvalue,
            "ci95Low": float(ci95[0]) if np.isfinite(ci95[0]) else None,
            "ci95High": float(ci95[1]) if np.isfinite(ci95[1]) else None,
            "equivalenceCi90Low": float(ci90[0]) if np.isfinite(ci90[0]) else None,
            "equivalenceCi90High": float(ci90[1]) if np.isfinite(ci90[1]) else None,
            "equivalentSmallEffect": equivalent,
            "permutationPValue": permutation_p,
            "status": "DEFINED",
            "reason": None,
        }
    )
    return base


def partial_spearman(x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]) -> float | None:
    """Partial Spearman correlation through residualized midranks."""

    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in (x, y, z)]
    keep = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    xv, yv, zv = [stats.rankdata(value[keep]) for value in arrays]
    if len(xv) < 5 or any(np.ptp(value) <= 1e-15 for value in (xv, yv, zv)):
        return None
    design = np.column_stack((np.ones(len(zv)), zv))
    x_resid = xv - design @ np.linalg.lstsq(design, xv, rcond=None)[0]
    y_resid = yv - design @ np.linalg.lstsq(design, yv, rcond=None)[0]
    if np.ptp(x_resid) <= 1e-15 or np.ptp(y_resid) <= 1e-15:
        return None
    return float(stats.pearsonr(x_resid, y_resid).statistic)


@dataclass(frozen=True, slots=True)
class SpikeEpisode:
    start: int
    end: int
    peak_position: int


def excursion_episodes(values: NDArray[np.float64], threshold: float) -> list[SpikeEpisode]:
    """Contiguous strict exceedances with earliest maximum as the peak."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    hits = np.flatnonzero(np.isfinite(array) & (array > threshold))
    if not len(hits):
        return []
    breaks = np.flatnonzero(np.diff(hits) > 1) + 1
    groups = np.split(hits, breaks)
    episodes: list[SpikeEpisode] = []
    for group in groups:
        local = array[group]
        peak = int(group[int(np.flatnonzero(local == np.nanmax(local))[0])])
        episodes.append(SpikeEpisode(int(group[0]), int(group[-1]), peak))
    return episodes


def all_pair_mean_distance(values: NDArray[np.float64]) -> float | None:
    """Mean distance over all unordered pairs, matching the public lineage."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) < 2:
        return None
    distances = np.abs(array[:, None] - array[None, :])
    return float(np.mean(distances[np.triu_indices(len(array), k=1)]))


class VariableLengthMaskedSequenceMLP(nn.Module):
    """The S16 architecture formula at data-determined padding dimensions.

    The step encoder, hidden width, activations, dropout, and output-by-position
    structure are unchanged.  Only the non-tunable input and target dimensions
    implied by a frozen proportion vary.
    """

    def __init__(self, input_length: int, target_length: int) -> None:
        super().__init__()
        if input_length <= 0 or target_length <= 0:
            raise ValueError("sequence lengths must be positive")
        self.input_length = int(input_length)
        self.target_length = int(target_length)
        self.step_encoder = nn.Linear(200, STEP_EMBED_DIM, bias=True)
        sequence_width = self.input_length * STEP_EMBED_DIM + self.input_length
        self.hidden_1 = nn.Linear(sequence_width, HIDDEN_DIM, bias=True)
        self.hidden_2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True)
        self.output = nn.Linear(HIDDEN_DIM, self.target_length, bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.10)

    def forward(
        self,
        values: torch.Tensor,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        expected = (self.input_length, FEATURE_CHANNEL_CAPACITY)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected:
            raise ValueError("unexpected locked input value shape")
        step_input = torch.cat((values, channel_mask), dim=-1)
        encoded = self.activation(self.step_encoder(step_input))
        encoded = encoded * time_mask.unsqueeze(-1)
        sequence = torch.cat((encoded.flatten(start_dim=1), time_mask), dim=1)
        hidden = self.dropout(self.activation(self.hidden_1(sequence)))
        hidden = self.dropout(self.activation(self.hidden_2(hidden)))
        return self.output(hidden)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def expected_parameter_count(input_length: int, target_length: int) -> int:
    """Closed-form trainable count for the locked architecture formula."""

    step = 200 * STEP_EMBED_DIM + STEP_EMBED_DIM
    sequence_width = input_length * STEP_EMBED_DIM + input_length
    hidden_1 = sequence_width * HIDDEN_DIM + HIDDEN_DIM
    hidden_2 = HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM
    output = HIDDEN_DIM * target_length + target_length
    return int(step + hidden_1 + hidden_2 + output)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: VariableLengthMaskedSequenceMLP
    history: pd.DataFrame
    best_epoch: int
    stopped_epoch: int
    best_validation_loss: float


def _tensor(value: NDArray[Any]) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).to(dtype=torch.float64)


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denominator = mask.sum()
    if denominator.item() <= 0:
        raise ValueError("zero valid target cells")
    return (losses * mask).sum() / denominator


def train_locked_mlp(
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
    """Exact S16 optimizer/regularization/early-stopping contract."""

    input_length = int(fit_values.shape[1])
    target_length = int(fit_targets.shape[1])
    torch.manual_seed(int(model_seed))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = VariableLengthMaskedSequenceMLP(input_length, target_length).to(dtype=torch.float64, device="cpu")
    if parameter_count(model) != expected_parameter_count(input_length, target_length):
        raise RuntimeError("locked parameter-count formula failed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    fit_tensors = tuple(
        _tensor(value)
        for value in (
            fit_values,
            fit_channel_mask.astype(np.float64),
            fit_time_mask.astype(np.float64),
            fit_targets,
            fit_target_mask.astype(np.float64),
        )
    )
    validation_tensors = tuple(
        _tensor(value)
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
    without_improvement = 0
    rows: list[dict[str, float | int]] = []
    stopped_epoch = -1
    for epoch in range(maximum_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        fit_logits = model(*fit_tensors[:3])
        fit_loss = _masked_bce(fit_logits, fit_tensors[3], fit_tensors[4])
        fit_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(*validation_tensors[:3])
            validation_loss = _masked_bce(validation_logits, validation_tensors[3], validation_tensors[4])
        fit_value = float(fit_loss.detach().cpu())
        validation_value = float(validation_loss.detach().cpu())
        rows.append({"epoch": epoch, "fitLoss": fit_value, "validationLoss": validation_value})
        stopped_epoch = epoch
        if validation_value < best_loss - minimum_improvement:
            best_loss = validation_value
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("no locked model retained")
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(model, pd.DataFrame(rows), best_epoch, stopped_epoch, best_loss)


def predict_locked_mlp(
    model: VariableLengthMaskedSequenceMLP,
    values: NDArray[np.float64],
    channel_mask: NDArray[np.bool_],
    time_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    model.eval()
    with torch.no_grad():
        logits = model(_tensor(values), _tensor(channel_mask.astype(np.float64)), _tensor(time_mask.astype(np.float64)))
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)


def evaluate_masked(
    targets: NDArray[np.float64], probabilities: NDArray[np.float64], mask: NDArray[np.bool_]
) -> dict[str, float | int | None]:
    """Apply frozen S16 metrics to valid target cells."""

    y = np.asarray(targets)[mask].astype(bool)
    p = np.asarray(probabilities)[mask].astype(np.float64)
    return binary_metrics(y, p)


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "CANDIDATE_IDS",
    "CUTOFF_MODE",
    "DUMMY_FEATURE_ID",
    "EQUIVALENCE_MARGIN",
    "FEATURE_IDS",
    "LEARNED_FEATURE_IDS",
    "LOOP_ID",
    "PERMUTATION_REPLICATES",
    "PROPORTIONS",
    "RETROSPECTIVE_MODE",
    "TEMPORAL_MODES",
    "VERSION",
    "all_pair_mean_distance",
    "apply_channel_scaler",
    "correlation_inference",
    "derive_seed128",
    "derive_seed32",
    "detectable_correlation",
    "evaluate_masked",
    "excursion_episodes",
    "expected_parameter_count",
    "fit_channel_scaler",
    "holm_adjust",
    "parameter_count",
    "partial_spearman",
    "predict_locked_mlp",
    "rank_candidate",
    "train_locked_mlp",
]
