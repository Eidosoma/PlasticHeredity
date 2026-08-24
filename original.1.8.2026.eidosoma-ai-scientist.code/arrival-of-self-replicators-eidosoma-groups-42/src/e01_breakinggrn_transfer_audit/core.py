"""Safe clean-room execution of the pinned BreakingGRNMemories Phi lineage.

The numerical core is the already source-equivalence-confirmed corrected IIGR
implementation.  BreakingGRNMemories commit ``afe44231`` copied that core and
added two public entry points: ``information.compute_circuit_info`` (raw
emergence and integrated values) and ``phi.compute_integrated_info``
(nonfinite-to-zero emergence only).  This module preserves those identities
separately while avoiding public-source pickle loading during GARD execution.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from e01_pigozzi_source_audit.core import (
    BOTTOM_ATOM,
    CAUSATION_ATOMS,
    INITIAL_PHIR_ATOM,
    PHIR_ATOMS,
    SYNERGY_ATOM,
    SourceImplementation,
    _iigr_preprocess,
    _local_phi_min,
    _source_partition,
    load_safe_lattice,
)
from e01_pigozzi_source_equivalence_confirmation.core import (
    iigr_pairwise_source_mi,
)


@dataclass(frozen=True)
class BreakingTransferResult:
    status: str
    reason: str | None
    processed: np.ndarray | None
    mi_matrix: np.ndarray | None
    fiedler_vector: np.ndarray | None
    partition_1: tuple[int, ...]
    partition_2: tuple[int, ...]
    reduced: np.ndarray | None
    synergy_raw: np.ndarray | None
    causation_raw: np.ndarray | None
    emergence_raw: np.ndarray | None
    emergence_nan0: np.ndarray | None
    integrated_raw: np.ndarray | None
    local_offset: int = 2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def derive_seed(*identity: object) -> int:
    material = "\x1f".join(
        ["E01-S19-L17-BREAKINGGRN-TRANSFER-SEED-v1", *map(str, identity)]
    )
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def _empty(status: str, reason: str) -> BreakingTransferResult:
    return BreakingTransferResult(
        status=status,
        reason=reason,
        processed=None,
        mi_matrix=None,
        fiedler_vector=None,
        partition_1=(),
        partition_2=(),
        reduced=None,
        synergy_raw=None,
        causation_raw=None,
        emergence_raw=None,
        emergence_nan0=None,
        integrated_raw=None,
    )


def _decompose(
    reduced: np.ndarray, safe_lattice_path: str | Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order, descendants = load_safe_lattice(safe_lattice_path)
    partials: dict[object, np.ndarray] = {}
    for atom in order:
        redundancy = _local_phi_min(atom, reduced, SourceImplementation.IIGR)
        if atom == BOTTOM_ATOM:
            partials[atom] = redundancy
        else:
            partials[atom] = redundancy - np.vstack(
                [partials[item] for item in descendants[atom]]
            ).sum(axis=0)
    integrated = partials[INITIAL_PHIR_ATOM].copy()
    for atom in PHIR_ATOMS:
        integrated += partials[atom]
    synergy = np.asarray(partials[SYNERGY_ATOM], dtype=np.float64)
    causation = np.asarray(
        partials[CAUSATION_ATOMS[0]] + partials[CAUSATION_ATOMS[1]],
        dtype=np.float64,
    )
    return synergy, causation, np.asarray(integrated, dtype=np.float64)


def run_breaking_transfer(
    observations: np.ndarray,
    safe_lattice_path: str | Path,
    *,
    preprocessing_seed: int,
    partition_seed: int,
) -> BreakingTransferResult:
    """Run the complete frozen BGM/IIGR path on time-by-coordinate input."""

    raw = np.asarray(observations, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 4 or raw.shape[1] < 2:
        return _empty("INELIGIBLE_INPUT_SHAPE", "requires_time_by_dimension_4x2")
    if not np.all(np.isfinite(raw)):
        return _empty("INELIGIBLE_NONFINITE_INPUT", "input_contains_nonfinite")
    processed: np.ndarray | None = None
    mi: np.ndarray | None = None
    fiedler: np.ndarray | None = None
    p1: tuple[int, ...] = ()
    p2: tuple[int, ...] = ()
    reduced: np.ndarray | None = None
    try:
        processed, retained = _iigr_preprocess(raw.T, preprocessing_seed)
        # Current phi.py converts only NaNs after preprocessing; infinities are
        # retained.  All registered GARD inputs are finite and nonconstant.
        processed_phi = np.nan_to_num(processed, nan=0.0, copy=True)
        mi = iigr_pairwise_source_mi(processed_phi, alpha=1, lag=1)
        if not np.all(np.isfinite(mi)):
            return BreakingTransferResult(
                "INELIGIBLE_NONFINITE_MI",
                "source_lagged_mi_contains_nonfinite",
                processed,
                mi,
                None,
                (),
                (),
                None,
                None,
                None,
                None,
                None,
                None,
            )
        fiedler, p1_local, p2_local = _source_partition(mi, partition_seed)
        p1 = tuple(retained[i] for i in p1_local)
        p2 = tuple(retained[i] for i in p2_local)
        if not p1 or not p2:
            return BreakingTransferResult(
                "INELIGIBLE_FIEDLER_PARTITION_EMPTY",
                "strict_sign_partition_has_empty_side",
                processed,
                mi,
                fiedler,
                p1,
                p2,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        if np.any(fiedler == 0.0):
            return BreakingTransferResult(
                "INELIGIBLE_FIEDLER_PARTITION_AMBIGUOUS",
                "one_or_more_fiedler_entries_exactly_zero",
                processed,
                mi,
                fiedler,
                p1,
                p2,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        reduced = np.vstack(
            (
                processed_phi[list(p1_local)].mean(axis=0),
                processed_phi[list(p2_local)].mean(axis=0),
            )
        )
        synergy, causation, integrated = _decompose(reduced, safe_lattice_path)
        emergence = synergy + causation
        emergence_nan0 = np.nan_to_num(
            synergy, nan=0.0, posinf=0.0, neginf=0.0
        ) + np.nan_to_num(causation, nan=0.0, posinf=0.0, neginf=0.0)
        finite_raw = bool(
            np.all(np.isfinite(emergence)) and np.all(np.isfinite(integrated))
        )
        return BreakingTransferResult(
            status="ELIGIBLE" if finite_raw else "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES",
            reason=None if finite_raw else "raw_emergence_or_integrated_nonfinite",
            processed=processed,
            mi_matrix=mi,
            fiedler_vector=fiedler,
            partition_1=p1,
            partition_2=p2,
            reduced=reduced,
            synergy_raw=synergy,
            causation_raw=causation,
            emergence_raw=emergence,
            emergence_nan0=emergence_nan0,
            integrated_raw=integrated,
        )
    except Exception as exc:  # noqa: BLE001 - source failure is evidence.
        return BreakingTransferResult(
            status="INELIGIBLE_SOURCE_PIPELINE_EXCEPTION",
            reason=f"{type(exc).__name__}:{exc}",
            processed=processed,
            mi_matrix=mi,
            fiedler_vector=fiedler,
            partition_1=p1,
            partition_2=p2,
            reduced=reduced,
            synergy_raw=None,
            causation_raw=None,
            emergence_raw=None,
            emergence_nan0=None,
            integrated_raw=None,
        )
