"""CLR, spectral bipartition, and local Gaussian causal-emergence estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

from .composition import clr_transform
from .config import CausalConfig


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def lagged_gaussian_mi_matrix(data: ArrayLike, lag: int = 1) -> FloatArray:
    """Pairwise Gaussian mutual information from variables at t to t + lag."""

    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("data must have shape (time, variables)")
    if lag < 1 or values.shape[0] <= lag + 1:
        raise ValueError("trajectory is too short for the requested lag")
    past = values[:-lag]
    future = values[lag:]
    past_centered = past - past.mean(axis=0, keepdims=True)
    future_centered = future - future.mean(axis=0, keepdims=True)
    denominator = np.sqrt(
        np.sum(past_centered**2, axis=0)[:, None]
        * np.sum(future_centered**2, axis=0)[None, :]
    )
    cross_product = np.einsum(
        "ti,tj->ij", past_centered, future_centered, optimize=True
    )
    correlation = np.divide(
        cross_product,
        denominator,
        out=np.zeros((values.shape[1], values.shape[1]), dtype=float),
        where=denominator > 0,
    )
    correlation = np.clip(correlation, -1 + 1e-12, 1 - 1e-12)
    return -0.5 * np.log1p(-(correlation**2))


def fiedler_bipartition(
    data: ArrayLike, *, lag: int = 1, cut: str = "zero"
) -> Tuple[BoolArray, FloatArray, FloatArray]:
    """Approximate the minimum-information bipartition with a Fiedler cut."""

    if cut not in {"zero", "median"}:
        raise ValueError("cut must be 'zero' or 'median'")
    directed_mi = lagged_gaussian_mi_matrix(data, lag=lag)
    affinity = 0.5 * (directed_mi + directed_mi.T)
    np.fill_diagonal(affinity, 0.0)
    degree = affinity.sum(axis=1)
    safe_degree = np.where(degree > 0, degree, 1.0)
    inv_sqrt_degree = 1.0 / np.sqrt(safe_degree)
    laplacian = np.eye(affinity.shape[0]) - (
        inv_sqrt_degree[:, None] * affinity * inv_sqrt_degree[None, :]
    )
    eigenvalues, eigenvectors = linalg.eigh(laplacian, check_finite=True)
    order = np.argsort(eigenvalues)
    fiedler = eigenvectors[:, order[1]]
    threshold = 0.0 if cut == "zero" else float(np.median(fiedler))
    partition = fiedler > threshold
    if partition.all() or (~partition).all():
        rank_order = np.argsort(fiedler, kind="mergesort")
        partition = np.zeros(fiedler.size, dtype=bool)
        partition[rank_order[fiedler.size // 2 :]] = True
    return partition, affinity, fiedler


def project_partition(data: ArrayLike, partition: ArrayLike) -> FloatArray:
    """Average variables within the two sides of a bipartition."""

    values = np.asarray(data, dtype=float)
    mask = np.asarray(partition, dtype=bool)
    if values.ndim != 2 or mask.shape != (values.shape[1],):
        raise ValueError("partition must select columns of data")
    if mask.all() or (~mask).all():
        raise ValueError("both partition components must be non-empty")
    return np.column_stack((values[:, ~mask].mean(axis=1), values[:, mask].mean(axis=1)))


@dataclass(frozen=True)
class LocalGaussianModel:
    """A fitted four-variable Gaussian model for two-part lagged dynamics."""

    lag: int
    scale: FloatArray
    mean: FloatArray
    covariance: FloatArray
    ridge: float
    redundancy_source: int

    @classmethod
    def fit(
        cls, grouped: ArrayLike, *, lag: int = 1, ridge: float = 1e-8
    ) -> "LocalGaussianModel":
        values = np.asarray(grouped, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("grouped data must have shape (time, 2)")
        if values.shape[0] <= lag + 4:
            raise ValueError("at least six lagged observations are required")
        four = np.column_stack((values[:-lag], values[lag:]))
        scale = four.std(axis=0, ddof=1)
        scale = np.where(scale > 0, scale, 1.0)
        standardized = four / scale
        mean = standardized.mean(axis=0)
        covariance = np.cov(standardized, rowvar=False, ddof=1)
        covariance = np.asarray(covariance, dtype=float)
        covariance += ridge * np.eye(4)
        provisional = cls(
            lag=lag,
            scale=scale,
            mean=mean,
            covariance=covariance,
            ridge=ridge,
            redundancy_source=0,
        )
        components = provisional.local_information(standardized, already_scaled=True)
        mean_1 = float(np.nanmean(components["part1_to_future"]))
        mean_2 = float(np.nanmean(components["part2_to_future"]))
        source = 0 if mean_1 <= mean_2 else 1
        return cls(
            lag=lag,
            scale=scale,
            mean=mean,
            covariance=covariance,
            ridge=ridge,
            redundancy_source=source,
        )

    def _local_entropy(self, values: FloatArray, indices: Tuple[int, ...]) -> FloatArray:
        subset = values[:, indices]
        mean = self.mean[list(indices)]
        covariance = self.covariance[np.ix_(indices, indices)]
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            covariance = covariance + self.ridge * np.eye(len(indices))
            sign, logdet = np.linalg.slogdet(covariance)
        inverse = np.linalg.pinv(covariance, hermitian=True)
        centered = subset - mean
        mahalanobis = np.einsum("ni,ij,nj->n", centered, inverse, centered)
        dimension = len(indices)
        return 0.5 * (dimension * np.log(2 * np.pi) + logdet + mahalanobis)

    def local_information(
        self, four_values: ArrayLike, *, already_scaled: bool = False
    ) -> Dict[str, FloatArray]:
        """Return local whole and part-to-future mutual information terms."""

        values = np.asarray(four_values, dtype=float)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("four_values must have shape (observations, 4)")
        standardized = values if already_scaled else values / self.scale
        h_past = self._local_entropy(standardized, (0, 1))
        h_future = self._local_entropy(standardized, (2, 3))
        h_all = self._local_entropy(standardized, (0, 1, 2, 3))
        h_p1 = self._local_entropy(standardized, (0,))
        h_p2 = self._local_entropy(standardized, (1,))
        h_p1_future = self._local_entropy(standardized, (0, 2, 3))
        h_p2_future = self._local_entropy(standardized, (1, 2, 3))
        whole = h_past + h_future - h_all
        part1 = h_p1 + h_future - h_p1_future
        part2 = h_p2 + h_future - h_p2_future
        return {
            "whole_to_future": whole,
            "part1_to_future": part1,
            "part2_to_future": part2,
        }

    def score_transitions(
        self, past: ArrayLike, future: ArrayLike, *, measure: str = "wms"
    ) -> FloatArray:
        """Score one or more transitions with the paper's displayed measure."""

        past_values = np.atleast_2d(np.asarray(past, dtype=float))
        future_values = np.atleast_2d(np.asarray(future, dtype=float))
        if past_values.shape[1:] != (2,) or future_values.shape[1:] != (2,):
            raise ValueError("past and future must have two partition components")
        if past_values.shape[0] == 1 and future_values.shape[0] > 1:
            past_values = np.repeat(past_values, future_values.shape[0], axis=0)
        if future_values.shape[0] == 1 and past_values.shape[0] > 1:
            future_values = np.repeat(future_values, past_values.shape[0], axis=0)
        if past_values.shape[0] != future_values.shape[0]:
            raise ValueError("past and future observation counts do not match")
        information = self.local_information(np.column_stack((past_values, future_values)))
        wms = (
            information["whole_to_future"]
            - information["part1_to_future"]
            - information["part2_to_future"]
        )
        if measure == "wms":
            return wms
        if measure == "mmi_synergy":
            redundant = information[
                "part1_to_future" if self.redundancy_source == 0 else "part2_to_future"
            ]
            return wms + redundant
        raise ValueError("measure must be 'wms' or 'mmi_synergy'")


@dataclass(frozen=True)
class CausalTrajectory:
    """Causal-emergence trajectory and all fitted projection state."""

    values: FloatArray
    time_indices: NDArray[np.int64]
    clr: FloatArray
    partition: BoolArray
    affinity: FloatArray
    fiedler: FloatArray
    grouped: FloatArray
    model: LocalGaussianModel
    config: CausalConfig

    def project_counts(self, counts: ArrayLike) -> FloatArray:
        transformed = clr_transform(
            counts,
            pseudocount=self.config.pseudocount,
            drop_last=self.config.drop_last_clr_component,
        )
        if transformed.ndim == 1:
            transformed = transformed[None, :]
        return project_partition(transformed, self.partition)

    def score_count_transitions(
        self, past_counts: ArrayLike, future_counts: ArrayLike
    ) -> FloatArray:
        past_grouped = self.project_counts(past_counts)
        future_grouped = self.project_counts(future_counts)
        return self.model.score_transitions(
            past_grouped, future_grouped, measure=self.config.measure
        )


def fit_causal_trajectory(
    counts: ArrayLike, config: CausalConfig = CausalConfig()
) -> CausalTrajectory:
    """Fit the preprint's CLR -> MIB -> local Gaussian Phi-r pipeline."""

    config.validate()
    transformed = clr_transform(
        counts,
        pseudocount=config.pseudocount,
        drop_last=config.drop_last_clr_component,
    )
    if transformed.ndim != 2:
        raise ValueError("counts must have shape (time, molecular_types)")
    partition, affinity, fiedler = fiedler_bipartition(
        transformed, lag=config.lag, cut=config.partition_cut
    )
    grouped = project_partition(transformed, partition)
    model = LocalGaussianModel.fit(
        grouped, lag=config.lag, ridge=config.covariance_ridge
    )
    past = grouped[: -config.lag]
    future = grouped[config.lag :]
    values = model.score_transitions(past, future, measure=config.measure)
    return CausalTrajectory(
        values=np.asarray(values, dtype=float),
        time_indices=np.arange(config.lag, grouped.shape[0], dtype=np.int64),
        clr=transformed,
        partition=partition,
        affinity=affinity,
        fiedler=fiedler,
        grouped=grouped,
        model=model,
        config=config,
    )
