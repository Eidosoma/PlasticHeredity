"""S12C wrapper-only repair for source-equivalence confirmation.

The only scientific-code change from the immutable S12B wrapper is the IIGR
lagged-MI implementation.  S12B used a vectorized correlation block whose
binary64 rounding could perturb a degenerate Fiedler eigenspace.  This module
reproduces the pinned source's nested pairwise ``scipy.stats.pearsonr`` loop,
assignment order, and significance comparison.  Pinned source files are never
modified or imported by scientific execution.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from e01_pigozzi_source_audit.core import (
    SourceImplementation,
    _iigr_preprocess,
    _local_decomposition,
    _phirl_mi_matrix,
    _phirl_preprocess,
    _source_partition,
    load_safe_lattice,
)


@dataclass(frozen=True)
class ConfirmedAuditResult:
    """Status-bearing result with preprocessing retained for equivalence audit."""

    implementation: str
    status: str
    reason: str | None
    retained_available: bool
    retained_variables: tuple[int, ...]
    processed: np.ndarray | None
    mi_matrix: np.ndarray | None
    fiedler_vector: np.ndarray | None
    partition_1: tuple[int, ...]
    partition_2: tuple[int, ...]
    partition_average: np.ndarray | None
    local_phi_r: np.ndarray | None
    emergence: np.ndarray | None
    local_offset: int


def derive_seed(root_seed_hex: str, *identity: object) -> int:
    """Derive a 32-bit legacy-NumPy seed in the S12C fixture domain."""

    material = "\x1f".join(["E01-S12C-SEED-v1", root_seed_hex, *map(str, identity)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def fixture_array(fixture_id: str, phase: str, root_seed: str) -> np.ndarray:
    """Generate one frozen development or untouched-confirmation fixture."""

    seed = derive_seed(root_seed, phase, "fixture", fixture_id)
    rng = np.random.RandomState(seed)
    shape = (384, 10)
    if fixture_id == "COUPLED_GAUSSIAN":
        data = rng.normal(size=shape)
        data[:, 5:] += 0.35 * data[:, :5]
        return data
    if fixture_id == "COUPLED_AUTOREGRESSIVE":
        innovations = rng.normal(size=shape)
        data = np.zeros(shape, dtype=np.float64)
        for index in range(1, shape[0]):
            data[index] = 0.55 * data[index - 1] + innovations[index]
            data[index, 5:] += 0.25 * data[index - 1, :5]
        return data
    if fixture_id == "CONSTANT_INPUT":
        return np.full(shape, 1.0 if phase == "development" else 2.0)
    if fixture_id == "SINGULAR_DUPLICATE_INPUT":
        base = rng.normal(size=(shape[0], 5))
        return np.column_stack((base, base))
    if fixture_id == "NEAR_SINGULAR_DUPLICATE_INPUT":
        base = rng.normal(size=(shape[0], 5))
        return np.column_stack((base, base + rng.normal(scale=1e-10, size=base.shape)))
    if fixture_id == "LOW_RANK_LINEAR_COMBINATION_INPUT":
        latent = rng.normal(size=(shape[0], 3))
        mixing = rng.normal(size=(3, shape[1]))
        return latent @ mixing
    if fixture_id == "REPLAY_PARTIAL_CONSTANT_INPUT":
        base = rng.normal(size=(shape[0], 8))
        base[:, 4:] += 0.35 * base[:, :4]
        return np.column_stack((np.ones(shape[0]), -np.ones(shape[0]), base))
    raise ValueError(f"unregistered S12C fixture: {fixture_id}")


def iigr_pairwise_source_mi(x: np.ndarray, *, alpha: float = 1.0, lag: int = 1) -> np.ndarray:
    """Reproduce the pinned IIGR nested-loop MI operation order exactly."""

    if lag <= 0:
        raise ValueError("S12C freezes the IIGR source audit at lag=1")
    n_variables = x.shape[0]
    mi_matrix = np.zeros((n_variables, n_variables))
    alpha_corrected = 1 * alpha
    for i in range(n_variables):
        for j in range(i):
            r1, p1 = pearsonr(x[i, :-lag], x[j, lag:])
            r2, p2 = pearsonr(x[i, lag:], x[j, :-lag])
            if p1 < alpha_corrected:
                mi1 = -0.5 * np.log(1.0 - (r1**2.0))
            else:
                mi1 = 0
            if p2 < alpha_corrected:
                mi2 = -0.5 * np.log(1.0 - (r2**2.0))
            else:
                mi2 = 0
            mi_matrix[i, j] = mi1 + mi2
            mi_matrix[j, i] = mi1 + mi2
    return np.array(mi_matrix)


def _result(
    branch: SourceImplementation,
    status: str,
    reason: str | None,
    *,
    retained_available: bool = False,
    retained: tuple[int, ...] = (),
    processed: np.ndarray | None = None,
    mi: np.ndarray | None = None,
    fiedler: np.ndarray | None = None,
    p1: tuple[int, ...] = (),
    p2: tuple[int, ...] = (),
    average: np.ndarray | None = None,
    phi_r: np.ndarray | None = None,
    emergence: np.ndarray | None = None,
) -> ConfirmedAuditResult:
    return ConfirmedAuditResult(
        implementation=branch.value,
        status=status,
        reason=reason,
        retained_available=retained_available,
        retained_variables=retained,
        processed=processed,
        mi_matrix=mi,
        fiedler_vector=fiedler,
        partition_1=p1,
        partition_2=p2,
        partition_average=average,
        local_phi_r=phi_r,
        emergence=emergence,
        local_offset=2 if branch is SourceImplementation.IIGR else 1,
    )


def run_source_pipeline(
    observations: np.ndarray,
    implementation: SourceImplementation | str,
    safe_lattice_path: str | Path,
    *,
    preprocessing_seed: int,
    partition_seed: int,
) -> ConfirmedAuditResult:
    """Run the clean S12C source reconstruction with complete failure state."""

    branch = SourceImplementation(implementation)
    raw = np.asarray(observations, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 4 or raw.shape[1] < 2:
        return _result(
            branch,
            "INELIGIBLE_INPUT_SHAPE",
            "requires_time_by_dimension_with_at_least_4x2",
        )
    if not np.all(np.isfinite(raw)):
        return _result(
            branch,
            "INELIGIBLE_NONFINITE_INPUT",
            "input_contains_nonfinite_values",
        )

    retained: tuple[int, ...] = ()
    processed: np.ndarray | None = None
    mi: np.ndarray | None = None
    fiedler: np.ndarray | None = None
    p1: tuple[int, ...] = ()
    p2: tuple[int, ...] = ()
    average: np.ndarray | None = None
    try:
        if branch is SourceImplementation.IIGR:
            processed, retained = _iigr_preprocess(raw.T, preprocessing_seed)
            mi = iigr_pairwise_source_mi(processed, alpha=1, lag=1)
        else:
            processed, retained = _phirl_preprocess(raw.T)
            if len(retained) < 2:
                return _result(
                    branch,
                    "INELIGIBLE_TOO_FEW_ACTIVE_DIMENSIONS",
                    "fewer_than_two_dimensions_above_std_1e-8",
                    retained_available=True,
                    retained=retained,
                )
            mi = _phirl_mi_matrix(processed)
        if not np.all(np.isfinite(processed)) or not np.all(np.isfinite(mi)):
            return _result(
                branch,
                "INELIGIBLE_NONFINITE_PREPROCESSING_OR_MI",
                "processed_array_or_mi_nonfinite",
                retained_available=True,
                retained=retained,
                processed=processed,
                mi=mi,
            )
        fiedler, p1_local, p2_local = _source_partition(mi, partition_seed)
        p1 = tuple(retained[i] for i in p1_local)
        p2 = tuple(retained[i] for i in p2_local)
        if not p1_local or not p2_local:
            return _result(
                branch,
                "INELIGIBLE_FIEDLER_PARTITION_EMPTY",
                "strict_sign_partition_has_empty_side",
                retained_available=True,
                retained=retained,
                processed=processed,
                mi=mi,
                fiedler=fiedler,
                p1=p1,
                p2=p2,
            )
        if np.any(fiedler == 0.0):
            return _result(
                branch,
                "INELIGIBLE_FIEDLER_PARTITION_AMBIGUOUS",
                "one_or_more_fiedler_entries_equal_zero",
                retained_available=True,
                retained=retained,
                processed=processed,
                mi=mi,
                fiedler=fiedler,
                p1=p1,
                p2=p2,
            )
        average = np.vstack(
            (
                processed[list(p1_local)].mean(axis=0),
                processed[list(p2_local)].mean(axis=0),
            )
        )
        order, descendants = load_safe_lattice(safe_lattice_path)
        phi_r, emergence = _local_decomposition(
            average, branch, order, descendants
        )
        if not np.all(np.isfinite(phi_r)) or not np.all(np.isfinite(emergence)):
            return _result(
                branch,
                "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES",
                "one_or_more_local_phi_r_or_diagnostic_values_nonfinite",
                retained_available=True,
                retained=retained,
                processed=processed,
                mi=mi,
                fiedler=fiedler,
                p1=p1,
                p2=p2,
                average=average,
                phi_r=phi_r,
                emergence=emergence,
            )
        return _result(
            branch,
            "ELIGIBLE",
            None,
            retained_available=True,
            retained=retained,
            processed=processed,
            mi=mi,
            fiedler=fiedler,
            p1=p1,
            p2=p2,
            average=average,
            phi_r=phi_r,
            emergence=emergence,
        )
    except Exception as exc:  # noqa: BLE001 - exact source exception is status-bearing.
        return _result(
            branch,
            "INELIGIBLE_SOURCE_PIPELINE_EXCEPTION",
            f"{type(exc).__name__}:{exc}",
            retained_available=processed is not None,
            retained=retained,
            processed=processed,
            mi=mi,
            fiedler=fiedler,
            p1=p1,
            p2=p2,
        )
