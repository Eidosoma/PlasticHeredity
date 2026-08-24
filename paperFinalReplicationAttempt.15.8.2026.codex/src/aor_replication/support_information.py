"""Label-blind covariance-support instruments for the Phi-family audit.

This module is additive to the completed formulation bridge.  It never imports
or computes replicator labels.  Its primary purpose is to expose ordinary score
levels, sample-covariance ranks, and ridge dependence on explicit fixed pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, Mapping, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .bridge_information import (
    PHIR_ATOMS,
    beta_physical_partition,
    close_all_clr,
    local_phi_id_atoms,
    rank_gaussianize,
)
from .config import CausalConfig
from .information import LocalGaussianModel, fit_causal_trajectory


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SUPPORT_INSTRUMENTS = (
    "typeset_full_wms",
    "macro_wms",
    "public_nine_atom",
    "pca8_full_revised",
)
DIAGNOSTIC_INSTRUMENTS = ("raw100_full_revised",)
ALL_SUPPORT_INSTRUMENTS = SUPPORT_INSTRUMENTS + DIAGNOSTIC_INSTRUMENTS

PCA_COMPONENTS_PER_MODULE = 8
RELATIVE_RIDGE_FACTOR = 1e-6
ABSOLUTE_RIDGE_FLOOR = 1e-8


def _array_digest(*arrays: ArrayLike) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CovarianceReading:
    dimension: int
    samples: int
    rank: int
    ridge: float
    logdet: float

    @property
    def rank_fraction(self) -> float:
        return float(self.rank / self.dimension)


@dataclass(frozen=True)
class MutualInformationReading:
    value: float
    left: CovarianceReading
    right: CovarianceReading
    joint: CovarianceReading


@dataclass(frozen=True)
class InstrumentReading:
    name: str
    score: float
    active_dimensions: int
    state_dimensions: int
    part_a_dimensions: int
    part_b_dimensions: int
    pairs: int
    whole_joint_dimension: int
    whole_joint_rank: int
    whole_joint_ridge: float
    covariance_rule: str
    components: Mapping[str, float]
    ranks: Mapping[str, int]
    ridges: Mapping[str, float]
    redundancy_channel: str = "none"

    @property
    def samples_per_joint_dimension(self) -> float:
        return float(self.pairs / self.whole_joint_dimension)

    @property
    def whole_joint_rank_fraction(self) -> float:
        return float(self.whole_joint_rank / self.whole_joint_dimension)

    def validate(self) -> None:
        if self.name not in ALL_SUPPORT_INSTRUMENTS:
            raise ValueError(f"unknown support instrument: {self.name}")
        if not np.isfinite(self.score):
            raise ValueError(f"{self.name} produced a non-finite score")
        if self.pairs < 2 or self.whole_joint_dimension < 2:
            raise ValueError("instrument support is too small")
        if not 0 <= self.whole_joint_rank <= self.whole_joint_dimension:
            raise ValueError("invalid covariance rank")
        if self.part_a_dimensions < 1 or self.part_b_dimensions < 1:
            raise ValueError("instrument partition must be nonempty")
        if not all(np.isfinite(float(value)) for value in self.components.values()):
            raise ValueError(f"{self.name} produced a non-finite component")


@dataclass(frozen=True)
class PCAProjection:
    mean: FloatArray
    components: FloatArray
    scale: FloatArray
    eigenvalues: FloatArray
    digest: str

    def transform(self, data: ArrayLike) -> FloatArray:
        values = np.asarray(data, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != self.mean.size:
            raise ValueError("PCA input dimensions do not match fitted projection")
        return np.asarray(
            (self.components @ (values - self.mean[:, None])) / self.scale[:, None],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class PreparedSupportWindow:
    observations: int
    original_z: FloatArray
    original_part_a: IntArray
    original_part_b: IntArray
    original_macro: FloatArray
    px_data: FloatArray
    px_active: IntArray
    px_part_a: IntArray
    px_part_b: IntArray
    pca_a: PCAProjection
    pca_b: PCAProjection
    transform_digest: str
    partition_digest: str
    pca_digest: str


def _sample_covariance_reading(data: ArrayLike) -> CovarianceReading:
    values = np.atleast_2d(np.asarray(data, dtype=np.float64))
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("covariance data must be dimensions by at least two samples")
    centered = values - values.mean(axis=1, keepdims=True)
    covariance = np.asarray(centered @ centered.T / values.shape[1], dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    maximum = max(0.0, float(eigenvalues[-1]))
    tolerance = (
        max(values.shape[0], values.shape[1])
        * np.finfo(np.float64).eps
        * maximum
    )
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    ridge = max(
        ABSOLUTE_RIDGE_FLOOR,
        RELATIVE_RIDGE_FACTOR * float(np.trace(covariance)) / covariance.shape[0],
    )
    regularized = eigenvalues + ridge
    if np.any(regularized <= 0.0) or not np.isfinite(regularized).all():
        raise ValueError("regularized covariance is not positive definite")
    return CovarianceReading(
        dimension=int(values.shape[0]),
        samples=int(values.shape[1]),
        rank=rank,
        ridge=float(ridge),
        logdet=float(np.log(regularized).sum()),
    )


def gaussian_mi_reading(
    left: ArrayLike, right: ArrayLike
) -> MutualInformationReading:
    x = np.atleast_2d(np.asarray(left, dtype=np.float64))
    y = np.atleast_2d(np.asarray(right, dtype=np.float64))
    if x.shape[1] != y.shape[1] or x.shape[1] < 2:
        raise ValueError("Gaussian MI inputs require matching sample support")
    left_covariance = _sample_covariance_reading(x)
    right_covariance = _sample_covariance_reading(y)
    joint_covariance = _sample_covariance_reading(np.vstack((x, y)))
    value = 0.5 * (
        left_covariance.logdet
        + right_covariance.logdet
        - joint_covariance.logdet
    )
    return MutualInformationReading(
        value=float(value),
        left=left_covariance,
        right=right_covariance,
        joint=joint_covariance,
    )


def _deterministic_pca(data: ArrayLike, components: int) -> PCAProjection:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("PCA data must be dimensions by samples")
    if not 1 <= components <= min(values.shape[0], values.shape[1] - 1):
        raise ValueError("requested PCA component count is unsupported")
    mean = values.mean(axis=1)
    centered = values - mean[:, None]
    covariance = centered @ centered.T / values.shape[1]
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, kind="mergesort")[::-1][:components]
    selected_values = np.asarray(eigenvalues[order], dtype=np.float64)
    selected_vectors = np.asarray(eigenvectors[:, order].T, dtype=np.float64)
    if np.any(selected_values <= 0.0):
        raise ValueError("PCA retained a non-positive-variance component")
    for row in selected_vectors:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    scores = selected_vectors @ centered
    scale = scores.std(axis=1, ddof=0)
    if np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise ValueError("PCA score standardization failed")
    return PCAProjection(
        mean=np.asarray(mean, dtype=np.float64),
        components=selected_vectors,
        scale=np.asarray(scale, dtype=np.float64),
        eigenvalues=selected_values,
        digest=_array_digest(mean, selected_vectors, scale, selected_values),
    )


def _active_partition(
    active: IntArray, physical_a: IntArray, physical_b: IntArray
) -> Tuple[IntArray, IntArray]:
    lookup = {int(species): index for index, species in enumerate(active)}
    first = np.asarray(
        [lookup[int(species)] for species in physical_a if int(species) in lookup],
        dtype=np.int64,
    )
    second = np.asarray(
        [lookup[int(species)] for species in physical_b if int(species) in lookup],
        dtype=np.int64,
    )
    if not first.size or not second.size or first.size + second.size != active.size:
        raise ValueError("active beta partition is incomplete")
    return first, second


def _zscore_active(data: ArrayLike) -> FloatArray:
    values = np.asarray(data, dtype=np.float64)
    means = values.mean(axis=1, keepdims=True)
    scales = values.std(axis=1, keepdims=True, ddof=0)
    if np.any(scales <= 1e-8):
        raise ValueError("typeset window contains an inactive CLR coordinate")
    return np.asarray((values - means) / scales, dtype=np.float64)


def prepare_support_window(
    counts: ArrayLike,
    beta: ArrayLike,
    causal_config: CausalConfig = CausalConfig(),
) -> PreparedSupportWindow:
    """Fit all label-blind transforms once on a fixed observation pool."""

    raw = np.asarray(counts)
    matrix = np.asarray(beta, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 65 or raw.shape[1] < 16:
        raise ValueError("support windows require >=65 observations and >=16 types")
    if matrix.shape != (raw.shape[1], raw.shape[1]):
        raise ValueError("counts and beta molecular dimensions do not match")
    if causal_config.lag != 1:
        raise ValueError("support audit is frozen at lag one")

    original = fit_causal_trajectory(raw, causal_config)
    original_z = _zscore_active(original.clr.T)
    original_a = np.flatnonzero(~original.partition).astype(np.int64)
    original_b = np.flatnonzero(original.partition).astype(np.int64)
    original_macro = np.asarray(original.grouped.T, dtype=np.float64)

    px_data, active = rank_gaussianize(close_all_clr(raw))
    physical_a, physical_b = beta_physical_partition(matrix)
    px_a, px_b = _active_partition(active, physical_a, physical_b)
    if px_a.size < PCA_COMPONENTS_PER_MODULE or px_b.size < PCA_COMPONENTS_PER_MODULE:
        raise ValueError("beta module is too small for the frozen PCA8 representation")
    pca_a = _deterministic_pca(
        px_data[px_a, :-1], PCA_COMPONENTS_PER_MODULE
    )
    pca_b = _deterministic_pca(
        px_data[px_b, :-1], PCA_COMPONENTS_PER_MODULE
    )
    return PreparedSupportWindow(
        observations=int(raw.shape[0]),
        original_z=original_z,
        original_part_a=original_a,
        original_part_b=original_b,
        original_macro=original_macro,
        px_data=px_data,
        px_active=active,
        px_part_a=px_a,
        px_part_b=px_b,
        pca_a=pca_a,
        pca_b=pca_b,
        transform_digest=_array_digest(original_z, original_macro, px_data, active),
        partition_digest=_array_digest(original_a, original_b, active[px_a], active[px_b]),
        pca_digest=_array_digest(
            np.frombuffer(bytes.fromhex(pca_a.digest), dtype=np.uint8),
            np.frombuffer(bytes.fromhex(pca_b.digest), dtype=np.uint8),
        ),
    )


def _reading_from_channels(
    *,
    name: str,
    score: float,
    active_dimensions: int,
    state_dimensions: int,
    part_a_dimensions: int,
    part_b_dimensions: int,
    pairs: int,
    whole: MutualInformationReading,
    covariance_rule: str,
    channels: Mapping[str, MutualInformationReading],
    components: Mapping[str, float],
    redundancy_channel: str = "none",
) -> InstrumentReading:
    ranks = {f"{key}_joint_rank": value.joint.rank for key, value in channels.items()}
    ridges = {
        f"{key}_left_ridge": value.left.ridge for key, value in channels.items()
    }
    ridges.update(
        {f"{key}_right_ridge": value.right.ridge for key, value in channels.items()}
    )
    ridges.update(
        {f"{key}_joint_ridge": value.joint.ridge for key, value in channels.items()}
    )
    reading = InstrumentReading(
        name=name,
        score=float(score),
        active_dimensions=int(active_dimensions),
        state_dimensions=int(state_dimensions),
        part_a_dimensions=int(part_a_dimensions),
        part_b_dimensions=int(part_b_dimensions),
        pairs=int(pairs),
        whole_joint_dimension=int(whole.joint.dimension),
        whole_joint_rank=int(whole.joint.rank),
        whole_joint_ridge=float(whole.joint.ridge),
        covariance_rule=covariance_rule,
        components=dict(components),
        ranks=ranks,
        ridges=ridges,
        redundancy_channel=redundancy_channel,
    )
    reading.validate()
    return reading


def typeset_full_wms_reading(
    past: ArrayLike,
    future: ArrayLike,
    part_a: ArrayLike,
    part_b: ArrayLike,
) -> InstrumentReading:
    left = np.asarray(past, dtype=np.float64)
    right = np.asarray(future, dtype=np.float64)
    first = np.asarray(part_a, dtype=np.int64)
    second = np.asarray(part_b, dtype=np.int64)
    whole = gaussian_mi_reading(left, right)
    a_to_whole = gaussian_mi_reading(left[first], right)
    b_to_whole = gaussian_mi_reading(left[second], right)
    score = whole.value - a_to_whole.value - b_to_whole.value
    channels = {
        "whole": whole,
        "a_to_whole": a_to_whole,
        "b_to_whole": b_to_whole,
    }
    return _reading_from_channels(
        name="typeset_full_wms",
        score=score,
        active_dimensions=left.shape[0],
        state_dimensions=left.shape[0],
        part_a_dimensions=first.size,
        part_b_dimensions=second.size,
        pairs=left.shape[1],
        whole=whole,
        covariance_rule="ddof0_relative_ridge=max(1e-8,1e-6*trace/d)",
        channels=channels,
        components={
            "whole_mi": whole.value,
            "a_to_whole_future_mi": a_to_whole.value,
            "b_to_whole_future_mi": b_to_whole.value,
        },
    )


def macro_wms_reading(
    past: ArrayLike,
    future: ArrayLike,
    *,
    active_dimensions: int,
    part_a_dimensions: int,
    part_b_dimensions: int,
    ridge: float,
) -> InstrumentReading:
    left = np.asarray(past, dtype=np.float64)
    right = np.asarray(future, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != 2:
        raise ValueError("macro WMS requires two matching macro variables")
    four = np.column_stack((left.T, right.T))
    scale = four.std(axis=0, ddof=1)
    scale = np.where(scale > 0.0, scale, 1.0)
    standardized = four / scale
    mean = standardized.mean(axis=0)
    covariance = np.asarray(np.cov(standardized, rowvar=False, ddof=1), dtype=float)
    covariance += ridge * np.eye(4)
    model = LocalGaussianModel(
        lag=1,
        scale=scale,
        mean=mean,
        covariance=covariance,
        ridge=ridge,
        redundancy_source=0,
    )
    information = model.local_information(standardized, already_scaled=True)
    local = (
        information["whole_to_future"]
        - information["part1_to_future"]
        - information["part2_to_future"]
    )
    unregularized = np.asarray(
        np.cov(standardized, rowvar=False, ddof=1), dtype=np.float64
    )
    rank = int(np.linalg.matrix_rank(unregularized))
    reading = InstrumentReading(
        name="macro_wms",
        score=float(local.mean()),
        active_dimensions=int(active_dimensions),
        state_dimensions=2,
        part_a_dimensions=int(part_a_dimensions),
        part_b_dimensions=int(part_b_dimensions),
        pairs=int(left.shape[1]),
        whole_joint_dimension=4,
        whole_joint_rank=rank,
        whole_joint_ridge=float(ridge),
        covariance_rule="existing_LocalGaussianModel_ddof1_absolute_ridge",
        components={
            "whole_mi": float(information["whole_to_future"].mean()),
            "a_to_whole_future_mi": float(
                information["part1_to_future"].mean()
            ),
            "b_to_whole_future_mi": float(
                information["part2_to_future"].mean()
            ),
        },
        ranks={"whole_joint_rank": rank},
        ridges={"whole_joint_ridge": float(ridge)},
    )
    reading.validate()
    return reading


def public_nine_atom_reading(
    past: ArrayLike,
    future: ArrayLike,
    *,
    active_dimensions: int,
    part_a_dimensions: int,
    part_b_dimensions: int,
) -> InstrumentReading:
    left = np.asarray(past, dtype=np.float64)
    right = np.asarray(future, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != 2:
        raise ValueError("public nine-atom score requires two macro variables")
    atoms = local_phi_id_atoms(left, right)
    score = float(np.sum([atoms[atom] for atom in PHIR_ATOMS], axis=0).mean())
    whole = gaussian_mi_reading(left, right)
    components = {
        "whole_mi": whole.value,
        **{
            f"selected_atom_{index:02d}": float(atoms[atom].mean())
            for index, atom in enumerate(PHIR_ATOMS)
        },
    }
    reading = InstrumentReading(
        name="public_nine_atom",
        score=score,
        active_dimensions=int(active_dimensions),
        state_dimensions=2,
        part_a_dimensions=int(part_a_dimensions),
        part_b_dimensions=int(part_b_dimensions),
        pairs=int(left.shape[1]),
        whole_joint_dimension=4,
        whole_joint_rank=whole.joint.rank,
        whole_joint_ridge=whole.joint.ridge,
        covariance_rule=(
            "public_local_phiid_ddof0; multivariate ridge=1e-6*trace/d; "
            "rank diagnostic uses ddof0 relative-ridge covariance"
        ),
        components=components,
        ranks={"whole_joint_rank": whole.joint.rank},
        ridges={"whole_joint_ridge": whole.joint.ridge},
    )
    reading.validate()
    return reading


def full_revised_reading(
    past: ArrayLike,
    future: ArrayLike,
    part_a: ArrayLike,
    part_b: ArrayLike,
    *,
    name: str,
    active_dimensions: int,
    original_part_a_dimensions: int,
    original_part_b_dimensions: int,
) -> InstrumentReading:
    left = np.asarray(past, dtype=np.float64)
    right = np.asarray(future, dtype=np.float64)
    first = np.asarray(part_a, dtype=np.int64)
    second = np.asarray(part_b, dtype=np.int64)
    if name not in {"pca8_full_revised", "raw100_full_revised"}:
        raise ValueError("invalid full-revised instrument name")
    channels = {
        "whole": gaussian_mi_reading(left, right),
        "aa": gaussian_mi_reading(left[first], right[first]),
        "ab": gaussian_mi_reading(left[first], right[second]),
        "ba": gaussian_mi_reading(left[second], right[first]),
        "bb": gaussian_mi_reading(left[second], right[second]),
    }
    redundancy = min(("aa", "ab", "ba", "bb"), key=lambda key: channels[key].value)
    score = (
        channels["whole"].value
        - channels["aa"].value
        - channels["bb"].value
        + channels[redundancy].value
    )
    components = {
        f"{key}_mi": reading.value for key, reading in channels.items()
    }
    components["double_redundancy"] = channels[redundancy].value
    return _reading_from_channels(
        name=name,
        score=score,
        active_dimensions=active_dimensions,
        state_dimensions=left.shape[0],
        part_a_dimensions=original_part_a_dimensions,
        part_b_dimensions=original_part_b_dimensions,
        pairs=left.shape[1],
        whole=channels["whole"],
        covariance_rule="ddof0_relative_ridge=max(1e-8,1e-6*trace/d)",
        channels=channels,
        components=components,
        redundancy_channel=redundancy,
    )


def score_prepared_pairs(
    prepared: PreparedSupportWindow, pair_indices: ArrayLike
) -> Dict[str, InstrumentReading]:
    """Score explicit pairs using transforms frozen on the prepared pool."""

    indices = np.asarray(pair_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size < 2:
        raise ValueError("pair indices must be a vector with at least two entries")
    if np.any(indices < 0) or np.any(indices >= prepared.observations - 1):
        raise ValueError("pair index is outside the prepared observation pool")
    if np.unique(indices).size != indices.size:
        raise ValueError("primary support pairs must be unique")

    original_past = prepared.original_z[:, :-1][:, indices]
    original_future = prepared.original_z[:, 1:][:, indices]
    typeset = typeset_full_wms_reading(
        original_past,
        original_future,
        prepared.original_part_a,
        prepared.original_part_b,
    )
    macro_past = prepared.original_macro[:, :-1][:, indices]
    macro_future = prepared.original_macro[:, 1:][:, indices]
    macro = macro_wms_reading(
        macro_past,
        macro_future,
        active_dimensions=prepared.original_z.shape[0],
        part_a_dimensions=prepared.original_part_a.size,
        part_b_dimensions=prepared.original_part_b.size,
        ridge=1e-8,
    )

    px_past = prepared.px_data[:, :-1][:, indices]
    px_future = prepared.px_data[:, 1:][:, indices]
    public_past = np.vstack(
        (
            px_past[prepared.px_part_a].mean(axis=0),
            px_past[prepared.px_part_b].mean(axis=0),
        )
    )
    public_future = np.vstack(
        (
            px_future[prepared.px_part_a].mean(axis=0),
            px_future[prepared.px_part_b].mean(axis=0),
        )
    )
    public = public_nine_atom_reading(
        public_past,
        public_future,
        active_dimensions=prepared.px_data.shape[0],
        part_a_dimensions=prepared.px_part_a.size,
        part_b_dimensions=prepared.px_part_b.size,
    )

    pca_past = np.vstack(
        (
            prepared.pca_a.transform(px_past[prepared.px_part_a]),
            prepared.pca_b.transform(px_past[prepared.px_part_b]),
        )
    )
    pca_future = np.vstack(
        (
            prepared.pca_a.transform(px_future[prepared.px_part_a]),
            prepared.pca_b.transform(px_future[prepared.px_part_b]),
        )
    )
    pca_first = np.arange(PCA_COMPONENTS_PER_MODULE, dtype=np.int64)
    pca_second = np.arange(
        PCA_COMPONENTS_PER_MODULE,
        2 * PCA_COMPONENTS_PER_MODULE,
        dtype=np.int64,
    )
    stabilized = full_revised_reading(
        pca_past,
        pca_future,
        pca_first,
        pca_second,
        name="pca8_full_revised",
        active_dimensions=prepared.px_data.shape[0],
        original_part_a_dimensions=prepared.px_part_a.size,
        original_part_b_dimensions=prepared.px_part_b.size,
    )
    raw = full_revised_reading(
        px_past,
        px_future,
        prepared.px_part_a,
        prepared.px_part_b,
        name="raw100_full_revised",
        active_dimensions=prepared.px_data.shape[0],
        original_part_a_dimensions=prepared.px_part_a.size,
        original_part_b_dimensions=prepared.px_part_b.size,
    )
    output = {
        "typeset_full_wms": typeset,
        "macro_wms": macro,
        "public_nine_atom": public,
        "pca8_full_revised": stabilized,
        "raw100_full_revised": raw,
    }
    if tuple(output) != ALL_SUPPORT_INSTRUMENTS:
        raise AssertionError("support instrument order drifted")
    return output


def score_operational_window(
    counts: ArrayLike,
    beta: ArrayLike,
    causal_config: CausalConfig = CausalConfig(),
) -> Tuple[PreparedSupportWindow, Dict[str, InstrumentReading]]:
    """Refit every permitted transform on one end-anchored window."""

    prepared = prepare_support_window(counts, beta, causal_config)
    indices = np.arange(prepared.observations - 1, dtype=np.int64)
    return prepared, score_prepared_pairs(prepared, indices)


def pca_full_revised_from_pairs(
    past: ArrayLike,
    future: ArrayLike,
    part_a: ArrayLike,
    part_b: ArrayLike,
    *,
    components_per_module: int = PCA_COMPONENTS_PER_MODULE,
) -> InstrumentReading:
    """Synthetic-fixture helper fitting PCA on past pairs only."""

    left = np.asarray(past, dtype=np.float64)
    right = np.asarray(future, dtype=np.float64)
    first = np.asarray(part_a, dtype=np.int64)
    second = np.asarray(part_b, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("synthetic PCA pairs must have matching shapes")
    pca_a = _deterministic_pca(left[first], components_per_module)
    pca_b = _deterministic_pca(left[second], components_per_module)
    reduced_past = np.vstack((pca_a.transform(left[first]), pca_b.transform(left[second])))
    reduced_future = np.vstack((pca_a.transform(right[first]), pca_b.transform(right[second])))
    reduced_a = np.arange(components_per_module, dtype=np.int64)
    reduced_b = np.arange(
        components_per_module, 2 * components_per_module, dtype=np.int64
    )
    return full_revised_reading(
        reduced_past,
        reduced_future,
        reduced_a,
        reduced_b,
        name="pca8_full_revised",
        active_dimensions=left.shape[0],
        original_part_a_dimensions=first.size,
        original_part_b_dimensions=second.size,
    )


__all__ = [
    "ALL_SUPPORT_INSTRUMENTS",
    "DIAGNOSTIC_INSTRUMENTS",
    "PCA_COMPONENTS_PER_MODULE",
    "SUPPORT_INSTRUMENTS",
    "CovarianceReading",
    "InstrumentReading",
    "MutualInformationReading",
    "PCAProjection",
    "PreparedSupportWindow",
    "full_revised_reading",
    "gaussian_mi_reading",
    "macro_wms_reading",
    "pca_full_revised_from_pairs",
    "prepare_support_window",
    "public_nine_atom_reading",
    "score_operational_window",
    "score_prepared_pairs",
    "typeset_full_wms_reading",
]
