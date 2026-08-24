"""Clean-room wrappers for the two pinned Pigozzi local-Phi source behaviors.

This module deliberately does not import either public repository and never
unpickles their lattice.  It consumes the JSON conversion produced by the
isolated converter.  The implementation identities are source-informed
reconstructions, not claims about the unavailable GARD author code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.stats import linregress, multivariate_normal, norm, zscore
from scipy.stats import t as student_t


class SourceImplementation(StrEnum):
    IIGR = "IIGR_CORRECTED_SOURCE"
    PHIRL = "PHIRL_REGULARIZED_SOURCE"


@dataclass(frozen=True)
class AuditResult:
    implementation: str
    status: str
    reason: str | None
    retained_variables: tuple[int, ...]
    mi_matrix: np.ndarray | None
    fiedler_vector: np.ndarray | None
    partition_1: tuple[int, ...]
    partition_2: tuple[int, ...]
    partition_average: np.ndarray | None
    local_phi_r: np.ndarray | None
    emergence: np.ndarray | None
    local_offset: int


BOTTOM_ATOM = (((0,), (1,)), ((0,), (1,)))
INITIAL_PHIR_ATOM = (((0,),), ((0, 1),))
SYNERGY_ATOM = (((0, 1),), ((0, 1),))
CAUSATION_ATOMS = (
    (((0, 1),), ((0,),)),
    (((0, 1),), ((1,),)),
)
PHIR_ATOMS = {
    (((0,), (1,)), ((0, 1),)),
    (((1,),), ((0, 1),)),
    (((0, 1),), ((0,),)),
    (((0, 1),), ((0,), (1,))),
    (((0, 1),), ((1,),)),
    (((0, 1),), ((0, 1),)),
    (((0,),), ((1,),)),
    (((1,),), ((0,),)),
}


def derive_seed(root_seed_hex: str, *identity: object) -> int:
    """Derive an order-independent 32-bit legacy-NumPy seed."""

    material = "\x1f".join(["E01-S12B-SEED-v1", root_seed_hex, *map(str, identity)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


@lru_cache(maxsize=4)
def load_safe_lattice(path: str | Path) -> tuple[list[Any], dict[Any, tuple[Any, ...]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "eidosoma.e01.s12b.safe_phi_lattice.v1":
        raise ValueError("unexpected safe lattice schema")
    order = [_nested_tuple(item) for item in payload["order"]]
    descendants = {
        _nested_tuple(item["atom"]): tuple(_nested_tuple(x) for x in item["descendants"])
        for item in payload["nodes"]
    }
    if set(order) != set(descendants) or order[0] != BOTTOM_ATOM:
        raise ValueError("safe lattice order/node mismatch")
    return order, descendants


def _legacy_normal(shape: tuple[int, ...], rng: np.random.RandomState, scale: float) -> np.ndarray:
    return rng.normal(loc=0.0, scale=scale, size=shape)


def _iigr_corrected_zscore(
    data: np.ndarray, rng: np.random.RandomState, *, noise: float = 1e-6
) -> np.ndarray:
    result = zscore(data, axis=1)
    for row_index, row in enumerate(result):
        if np.all(np.isnan(row)):
            result[row_index] = _legacy_normal((row.size,), rng, noise)
    return np.asarray(result, dtype=np.float64)


def _iigr_preprocess(data: np.ndarray, seed: int) -> tuple[np.ndarray, tuple[int, ...]]:
    rng = np.random.RandomState(seed)
    x = _iigr_corrected_zscore(np.asarray(data, dtype=np.float64).copy(), rng)
    global_signal = np.nanmean(x, axis=0)
    gsr = np.zeros_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        regression = linregress(global_signal, x[i])
        prediction = regression.intercept + regression.slope * global_signal
        gsr[i] = np.nansum([x[i], -prediction], axis=0)
    x = _iigr_corrected_zscore(gsr, rng)
    residuals = np.zeros((x.shape[0], x.shape[1] - 1), dtype=np.float64)
    for i in range(x.shape[0]):
        regression = linregress(x[i, :-1], x[i, 1:])
        prediction = regression.intercept + np.nanprod(
            [np.repeat(regression.slope, x.shape[1] - 1), x[i, :-1]], axis=0
        )
        residuals[i] = np.nansum([x[i, 1:], -prediction], axis=0)
    return _iigr_corrected_zscore(residuals, rng), tuple(range(data.shape[0]))


def _phirl_preprocess(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    x = np.asarray(data, dtype=np.float64)
    retained = np.flatnonzero(x.std(axis=1) > 1e-8)
    if retained.size < 2:
        return np.empty((retained.size, x.shape[1]), dtype=np.float64), tuple(map(int, retained))
    return np.asarray(zscore(x[retained], axis=1), dtype=np.float64), tuple(map(int, retained))


def _iigr_mi_matrix(x: np.ndarray) -> np.ndarray:
    """Vectorized alpha=1, no-Bonferroni, bidirectional source calculation."""

    n = x.shape[0]
    cross = np.corrcoef(np.concatenate([x[:, :-1], x[:, 1:]], axis=0))[:n, n:]
    with np.errstate(divide="ignore", invalid="ignore"):
        directional = -0.5 * np.log1p(-(cross * cross))
    # With alpha=1 the source excludes only exactly-zero correlations (p=1),
    # whose Gaussian MI is already zero.  Nonfinite inputs remain nonfinite.
    directional = np.where(cross == 0.0, 0.0, directional)
    mi = directional + directional.T
    np.fill_diagonal(mi, 0.0)
    return np.asarray(mi, dtype=np.float64)


def _phirl_mi_matrix(x: np.ndarray) -> np.ndarray:
    n, t = x.shape
    forward, backward = x[:, :-1], x[:, 1:]
    r1 = np.corrcoef(np.concatenate([forward, backward], axis=0))[:n, n:]
    r2 = np.corrcoef(np.concatenate([backward, forward], axis=0))[:n, n:]
    r = np.clip((r1 + r2) / 2.0, -0.999999, 0.999999)
    degrees_freedom = t - 2  # Deliberately matches the pinned source.
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = r * np.sqrt(degrees_freedom / (1.0 - r * r))
        p_values = 2.0 * (1.0 - student_t.cdf(np.abs(statistic), degrees_freedom))
        mi = np.zeros_like(r)
        mask = p_values < 1.0
        mi[mask] = -0.5 * np.log1p(-(r[mask] * r[mask]))
    np.fill_diagonal(mi, 0.0)
    return np.asarray(mi, dtype=np.float64)


def _source_partition(mi: np.ndarray, seed: int) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    corrected = np.add(mi, 1e-6)
    graph = nx.from_numpy_array(corrected, create_using=nx.Graph())
    # Passing an explicit RandomState preserves the source algorithm while
    # making the otherwise implicit NetworkX initialization replayable.
    fiedler = np.asarray(
        nx.fiedler_vector(
            graph,
            weight="weight",
            normalized=False,
            seed=np.random.RandomState(seed),
        ),
        dtype=np.float64,
    )
    p1 = tuple(map(int, np.flatnonzero(fiedler > 0.0)))
    p2 = tuple(map(int, np.flatnonzero(fiedler < 0.0)))
    return fiedler, p1, p2


def _entropy_1d(x: np.ndarray) -> np.ndarray:
    mean = x.mean()
    std = x.std()
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log(norm.pdf(x, loc=mean, scale=std))


def _entropy_nd(x: np.ndarray, implementation: SourceImplementation) -> np.ndarray:
    if x.ndim == 1:
        return _entropy_1d(x)
    if x.shape[0] == 1:
        return _entropy_1d(x[0])
    covariance = np.cov(x, ddof=0)
    if implementation is SourceImplementation.PHIRL:
        covariance = covariance + np.eye(covariance.shape[0]) * (
            1e-6 * np.trace(covariance) / covariance.shape[0]
        )
    means = x.mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log(multivariate_normal.pdf(x.T, mean=means, cov=covariance))


def _local_phi_min(
    atom: Any, reduced: np.ndarray, implementation: SourceImplementation
) -> np.ndarray:
    edge = reduced[[0, 1], :]
    i_plus = np.repeat(np.inf, edge.shape[1] - 1)
    i_minus = np.repeat(np.inf, edge.shape[1] - 1)
    for source_antichain in atom[0]:
        source = edge[np.asarray(source_antichain, dtype=int), :][:, :-1]
        i_plus = np.minimum(i_plus, _entropy_nd(source, implementation))
        for target_antichain in atom[1]:
            joint = np.squeeze(
                np.vstack(
                    (
                        edge[np.asarray(source_antichain, dtype=int), :][:, :-1],
                        edge[np.asarray(target_antichain, dtype=int), :][:, 1:],
                    )
                )
            )
            target = edge[np.asarray(target_antichain, dtype=int), :][:, 1:]
            conditional = _entropy_nd(joint, implementation) - _entropy_nd(
                target, implementation
            )
            i_minus = np.minimum(i_minus, conditional)
    return i_plus - i_minus


def _local_decomposition(
    reduced: np.ndarray,
    implementation: SourceImplementation,
    order: Iterable[Any],
    descendants: dict[Any, tuple[Any, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    partials: dict[Any, np.ndarray] = {}
    for atom in order:
        redundancy = _local_phi_min(atom, reduced, implementation)
        if atom == BOTTOM_ATOM:
            partials[atom] = redundancy
        else:
            partials[atom] = redundancy - np.vstack(
                [partials[item] for item in descendants[atom]]
            ).sum(axis=0)
    phi_r = partials[INITIAL_PHIR_ATOM].copy()
    for atom in PHIR_ATOMS:
        phi_r += partials[atom]
    emergence = partials[SYNERGY_ATOM] + sum(partials[item] for item in CAUSATION_ATOMS)
    return phi_r, emergence


def run_source_pipeline(
    observations: np.ndarray,
    implementation: SourceImplementation | str,
    safe_lattice_path: str | Path,
    *,
    preprocessing_seed: int,
    partition_seed: int,
) -> AuditResult:
    """Run one complete supplied array through one pinned-source behavior.

    ``observations`` is time by original retained molecular component.  Output
    local index ``k`` maps to original observation ``k + local_offset``.
    """

    branch = SourceImplementation(implementation)
    raw = np.asarray(observations, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 4 or raw.shape[1] < 2:
        return AuditResult(branch.value, "INELIGIBLE_INPUT_SHAPE", "requires_time_by_dimension_with_at_least_4x2", (), None, None, (), (), None, None, None, 2 if branch is SourceImplementation.IIGR else 1)
    if not np.all(np.isfinite(raw)):
        return AuditResult(branch.value, "INELIGIBLE_NONFINITE_INPUT", "input_contains_nonfinite_values", (), None, None, (), (), None, None, None, 2 if branch is SourceImplementation.IIGR else 1)
    try:
        if branch is SourceImplementation.IIGR:
            processed, retained = _iigr_preprocess(raw.T, preprocessing_seed)
            offset = 2
            mi = _iigr_mi_matrix(processed)
        else:
            processed, retained = _phirl_preprocess(raw.T)
            offset = 1
            if len(retained) < 2:
                return AuditResult(branch.value, "INELIGIBLE_TOO_FEW_ACTIVE_DIMENSIONS", "fewer_than_two_dimensions_above_std_1e-8", retained, None, None, (), (), None, None, None, offset)
            mi = _phirl_mi_matrix(processed)
        if not np.all(np.isfinite(processed)) or not np.all(np.isfinite(mi)):
            return AuditResult(branch.value, "INELIGIBLE_NONFINITE_PREPROCESSING_OR_MI", "processed_array_or_mi_nonfinite", retained, mi, None, (), (), None, None, None, offset)
        fiedler, p1_local, p2_local = _source_partition(mi, partition_seed)
        if not p1_local or not p2_local:
            return AuditResult(branch.value, "INELIGIBLE_FIEDLER_PARTITION_EMPTY", "strict_sign_partition_has_empty_side", retained, mi, fiedler, (), (), None, None, None, offset)
        if np.any(fiedler == 0.0):
            return AuditResult(branch.value, "INELIGIBLE_FIEDLER_PARTITION_AMBIGUOUS", "one_or_more_fiedler_entries_equal_zero", retained, mi, fiedler, (), (), None, None, None, offset)
        partition_average = np.vstack(
            (processed[list(p1_local)].mean(axis=0), processed[list(p2_local)].mean(axis=0))
        )
        order, descendants = load_safe_lattice(safe_lattice_path)
        phi_r, emergence = _local_decomposition(
            partition_average, branch, order, descendants
        )
        p1 = tuple(retained[i] for i in p1_local)
        p2 = tuple(retained[i] for i in p2_local)
        if not np.all(np.isfinite(phi_r)) or not np.all(np.isfinite(emergence)):
            return AuditResult(branch.value, "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES", "one_or_more_local_phi_r_or_diagnostic_values_nonfinite", retained, mi, fiedler, p1, p2, partition_average, phi_r, emergence, offset)
        return AuditResult(branch.value, "ELIGIBLE", None, retained, mi, fiedler, p1, p2, partition_average, phi_r, emergence, offset)
    except Exception as exc:  # noqa: BLE001 - status-bearing source failure is required.
        return AuditResult(branch.value, "INELIGIBLE_SOURCE_PIPELINE_EXCEPTION", f"{type(exc).__name__}:{exc}", (), None, None, (), (), None, None, None, 2 if branch is SourceImplementation.IIGR else 1)
