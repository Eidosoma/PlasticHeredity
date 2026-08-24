"""Frozen scientific and randomness contracts for E01 S13Y.

S13Y is deliberately narrow: it confirms one branch selected adaptively in S13X
on genuinely new catalytic matrices.  This module contains only prospective
identities, gates, and deterministic statistics helpers; it performs no I/O.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from e01_creative_directional_search.core import LabelSpec, label_specs

VERSION = "E01-S13Y-CLEAN-DIRECTIONAL-CONFIRMATION-v1.0.0"
RESEARCH_STEP_ID = "S13Y"
EVIDENCE_CLASS = "CLEAN_POST_SELECTION_DIRECTIONAL_CONFIRMATION"
ROOT_SEED_HEX = "b55fa3cbd5ff92fe10640bef1fa36e9678e7ec9ec5fa84e884cacc5f1e6d48c1"
SIMULATION_PHASE = "s13y_clean_directional_confirmation"
CANDIDATE_IDS = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
PRIMARY_LABEL_ID = "MOL_ADJACENT_INCOMING_H900"
SENSITIVITY_LABEL_ID = "MOL_ADJACENT_INCOMING_H970"
HISTORICAL_LABEL_ID = "HISTORICAL_H090_REPLICATOR"
IMPLEMENTATION_ID = "PHIRL_REGULARIZED_SOURCE"
METRIC_ID = "emergence"
FULL_MODE_ID = "PHIRL_REGULARIZED_SOURCE_EMERGENCE_FULL"
PREFIX_MODE_ID = "PHIRL_REGULARIZED_SOURCE_EMERGENCE_PREFIX_ENDPOINT"
S13X_PIPELINE_ID = "S13X-P-684e66c4cffe914c"
N_UNITS = 100
RESAMPLING_REPLICATES = 4096


@dataclass(frozen=True, slots=True)
class DirectionalGate:
    """Frozen candidate-level directional decision thresholds."""

    finite_coverage_at_least: float = 0.95
    defined_trajectories_at_least: int = 80
    positive_correlation_fraction_at_least: float = 0.65
    positive_drift_fraction_at_least: float = 0.50
    positive_p_at_most: float = 0.05


PRIMARY_GATE = DirectionalGate()


def _finite_float(value: Any) -> float | None:
    """Return one finite float or None for an unavailable statistic."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def candidate_registry() -> tuple[dict[str, Any], ...]:
    """Return the only two simulator candidates authorized for S13Y."""

    return (
        {
            "candidateId": "S12F-CANDIDATE-02",
            "s13yId": "S13Y-CLEAN-CANDIDATE-02-v1.0.0",
            "h": 0.6031526490073492,
            "exposureFamily": "FIXED_COMMON_EXPOSURE",
            "daughterRule": "FIRST_DAUGHTER",
            "overshootRule": "TRIM_NEW_ENTRANTS_TO_NMAX",
            "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
            "evidenceStatus": "S12FR_UPSTREAM_CONFIRMED",
        },
        {
            "candidateId": "S12F-CANDIDATE-03",
            "s13yId": "S13Y-CLEAN-CANDIDATE-03-v1.0.0",
            "h": 0.5613315384859516,
            "exposureFamily": "FIXED_COMMON_EXPOSURE",
            "daughterRule": "RANDOM_NONEMPTY",
            "overshootRule": "TRIM_NEW_ENTRANTS_TO_NMAX",
            "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
            "evidenceStatus": "S12FR_UPSTREAM_CONFIRMED",
        },
    )


def seed_material(*identity: object) -> bytes:
    """Canonical domain-separated material for source/statistical seeds."""

    return "\x1f".join(
        [VERSION, ROOT_SEED_HEX, "analysis", *map(str, identity)]
    ).encode("utf-8")


def seed_material_sha256(*identity: object) -> str:
    return hashlib.sha256(seed_material(*identity)).hexdigest()


def derive_seed(*identity: object) -> int:
    """Derive a replayable legacy RandomState-compatible 32-bit integer."""

    return int.from_bytes(hashlib.sha256(seed_material(*identity)).digest()[:4], "big")


def fixed_label_spec(label_id: str) -> LabelSpec:
    """Recover one exact S13X label specification without recreating it."""

    found = [item for item in label_specs() if item.label_id == label_id]
    if len(found) != 1:
        raise ValueError(f"S13X label specification is not unique: {label_id}")
    return found[0]


def primary_association_gate(row: Mapping[str, Any]) -> bool:
    """Apply the frozen candidate-specific retrospective association gate."""

    coverage = _finite_float(row.get("finiteCoverage"))
    positive_fraction = _finite_float(row.get("positiveCorrelationFraction"))
    median = _finite_float(row.get("medianCorrelation"))
    lower = _finite_float(row.get("bootstrapLower95"))
    p_value = _finite_float(row.get("circularShiftPositiveP"))
    return bool(
        coverage is not None
        and coverage >= PRIMARY_GATE.finite_coverage_at_least
        and int(row.get("definedCorrelationCount", 0))
        >= PRIMARY_GATE.defined_trajectories_at_least
        and positive_fraction is not None
        and positive_fraction >= PRIMARY_GATE.positive_correlation_fraction_at_least
        and median is not None
        and median > 0.0
        and lower is not None
        and lower > 0.0
        and p_value is not None
        and p_value <= PRIMARY_GATE.positive_p_at_most
    )


def primary_drift_gate(row: Mapping[str, Any]) -> bool:
    """Apply the frozen replicator-minus-drift candidate gate."""

    positive_fraction = _finite_float(row.get("higherDuringReplicationFraction"))
    median = _finite_float(row.get("medianMeanDifference"))
    lower = _finite_float(row.get("driftBootstrapLower95"))
    p_value = _finite_float(row.get("driftCircularShiftPositiveP"))
    return bool(
        int(row.get("definedDriftCount", 0))
        >= PRIMARY_GATE.defined_trajectories_at_least
        and positive_fraction is not None
        and positive_fraction >= PRIMARY_GATE.positive_drift_fraction_at_least
        and median is not None
        and median > 0.0
        and lower is not None
        and lower > 0.0
        and p_value is not None
        and p_value <= PRIMARY_GATE.positive_p_at_most
    )


def prefix_gate(row: Mapping[str, Any]) -> bool:
    """Apply the frozen secondary past-only directional falsification gate."""

    coverage = _finite_float(row.get("finiteCoverage"))
    positive_fraction = _finite_float(row.get("positiveCorrelationFraction"))
    median = _finite_float(row.get("medianCorrelation"))
    lower = _finite_float(row.get("bootstrapLower95"))
    p_value = _finite_float(row.get("circularShiftPositiveP"))
    return bool(
        coverage is not None
        and coverage >= 0.80
        and int(row.get("definedCorrelationCount", 0))
        >= PRIMARY_GATE.defined_trajectories_at_least
        and positive_fraction is not None
        and positive_fraction >= PRIMARY_GATE.positive_correlation_fraction_at_least
        and median is not None
        and median > 0.0
        and lower is not None
        and lower > 0.0
        and p_value is not None
        and p_value <= PRIMARY_GATE.positive_p_at_most
    )


def exact_label_identity(
    h_values: NDArray[np.floating[Any]],
    labels: NDArray[np.bool_],
    *,
    threshold: float = 0.9,
) -> dict[str, Any]:
    """Prove the binary label is the declared deterministic function of H."""

    h = np.asarray(h_values, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    finite = np.isfinite(h)
    expected = h[finite] > threshold
    observed = y[finite]
    mismatches = int(np.count_nonzero(expected != observed))
    return {
        "rowCount": int(h.size),
        "finiteHCount": int(np.count_nonzero(finite)),
        "mismatchCount": mismatches,
        "identityPassed": bool(np.all(finite) and mismatches == 0),
        "conditionalEntropyYGivenExactHBits": 0.0
        if np.all(finite) and mismatches == 0
        else None,
        "conditionalInformationEmergenceYGivenExactHBits": 0.0
        if np.all(finite) and mismatches == 0
        else None,
    }


def circular_vectors(
    values: NDArray[np.floating[Any]], labels: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """All cyclic-shift association/drift values, preserving cyclic durations."""

    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    finite = np.isfinite(x)
    x, y = x[finite], y[finite]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    ranked = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ranked = (ranked - ranked.mean()) / ranked.std()
    binary = y.astype(np.float64)
    binary_standard = (binary - binary.mean()) / binary.std()
    correlations = np.fft.ifft(
        np.conj(np.fft.fft(ranked)) * np.fft.fft(binary_standard)
    ).real / len(ranked)
    positive_count = float(binary.sum())
    negative_count = float(len(binary) - positive_count)
    raw_cross = np.fft.ifft(np.conj(np.fft.fft(x)) * np.fft.fft(binary)).real
    total = float(x.sum())
    differences = raw_cross / positive_count - (total - raw_cross) / negative_count
    return correlations.astype(np.float64), differences.astype(np.float64)


def summarize_resampled_direction(
    trajectory_rows: pd.DataFrame,
    payloads: Mapping[int, tuple[NDArray[np.float64], NDArray[np.bool_]]],
    *,
    seed_identity: tuple[object, ...],
    finite_coverage: float,
    replicates: int = RESAMPLING_REPLICATES,
) -> dict[str, Any]:
    """Aggregate trajectory effects with frozen bootstrap and cyclic-shift nulls."""

    rhos = (
        pd.to_numeric(trajectory_rows["rho"], errors="coerce").dropna().to_numpy(float)
    )
    differences = (
        pd.to_numeric(trajectory_rows["meanDifference"], errors="coerce")
        .dropna()
        .to_numpy(float)
    )
    if rhos.size == 0 or differences.size == 0:
        return {
            "finiteCoverage": finite_coverage,
            "definedCorrelationCount": int(rhos.size),
            "positiveCorrelationCount": int(np.count_nonzero(rhos > 0)),
            "positiveCorrelationFraction": None,
            "medianCorrelation": None,
            "meanCorrelation": None,
            "bootstrapLower95": None,
            "bootstrapUpper95": None,
            "circularShiftPositiveP": None,
            "definedDriftCount": int(differences.size),
            "higherDuringReplicationCount": int(np.count_nonzero(differences > 0)),
            "higherDuringReplicationFraction": None,
            "medianMeanDifference": None,
            "driftBootstrapLower95": None,
            "driftBootstrapUpper95": None,
            "driftCircularShiftPositiveP": None,
            "resamplingReplicates": 0,
        }
    bootstrap_rng = np.random.default_rng(derive_seed(*seed_identity, "bootstrap"))
    shift_rng = np.random.default_rng(derive_seed(*seed_identity, "circular_shift"))
    bootstrap_rhos = np.median(
        rhos[bootstrap_rng.integers(0, len(rhos), size=(replicates, len(rhos)))],
        axis=1,
    )
    bootstrap_differences = np.median(
        differences[
            bootstrap_rng.integers(
                0, len(differences), size=(replicates, len(differences))
            )
        ],
        axis=1,
    )
    correlation_columns: list[NDArray[np.float64]] = []
    difference_columns: list[NDArray[np.float64]] = []
    for matrix_index in sorted(payloads):
        correlations, shifted_differences = circular_vectors(*payloads[matrix_index])
        if len(correlations) <= 1:
            continue
        offsets = shift_rng.integers(1, len(correlations), size=replicates)
        correlation_columns.append(correlations[offsets])
        difference_columns.append(shifted_differences[offsets])
    correlation_null = np.median(np.column_stack(correlation_columns), axis=1)
    difference_null = np.median(np.column_stack(difference_columns), axis=1)
    median_rho = float(np.median(rhos))
    median_difference = float(np.median(differences))
    return {
        "finiteCoverage": float(finite_coverage),
        "definedCorrelationCount": int(rhos.size),
        "positiveCorrelationCount": int(np.count_nonzero(rhos > 0)),
        "positiveCorrelationFraction": float(np.mean(rhos > 0)),
        "medianCorrelation": median_rho,
        "meanCorrelation": float(np.mean(rhos)),
        "bootstrapLower95": float(np.quantile(bootstrap_rhos, 0.025)),
        "bootstrapUpper95": float(np.quantile(bootstrap_rhos, 0.975)),
        "circularShiftPositiveP": float(
            (1 + np.count_nonzero(correlation_null >= median_rho))
            / (1 + len(correlation_null))
        ),
        "definedDriftCount": int(differences.size),
        "higherDuringReplicationCount": int(np.count_nonzero(differences > 0)),
        "higherDuringReplicationFraction": float(np.mean(differences > 0)),
        "medianMeanDifference": median_difference,
        "driftBootstrapLower95": float(np.quantile(bootstrap_differences, 0.025)),
        "driftBootstrapUpper95": float(np.quantile(bootstrap_differences, 0.975)),
        "driftCircularShiftPositiveP": float(
            (1 + np.count_nonzero(difference_null >= median_difference))
            / (1 + len(difference_null))
        ),
        "resamplingReplicates": int(replicates),
    }


def classify(
    candidate_rows: Iterable[Mapping[str, Any]],
    *,
    exact_h_identity_passed: bool,
    validation_passed: bool,
) -> str:
    """Apply the frozen all-candidate and label-circularity adjudication."""

    if not validation_passed:
        return "S13Y_VALIDATION_FAILED_CLOSED"
    rows = list(candidate_rows)
    if len(rows) != 2 or {row.get("candidateId") for row in rows} != set(CANDIDATE_IDS):
        return "S13Y_VALIDATION_FAILED_CLOSED"
    candidate_passes = [bool(row.get("candidatePrimaryPassed")) for row in rows]
    if not all(candidate_passes):
        return "CLEAN_DIRECTIONAL_BRANCH_NOT_CONFIRMED"
    if exact_h_identity_passed:
        return "LABEL_COUPLED_RETROSPECTIVE_RESEMBLANCE"
    return "CONFIRMED_RETROSPECTIVE_DIRECTIONAL_RESEMBLANCE"


def outcome_class(classification: str) -> str:
    if classification == "CONFIRMED_RETROSPECTIVE_DIRECTIONAL_RESEMBLANCE":
        return "supportive"
    if classification == "LABEL_COUPLED_RETROSPECTIVE_RESEMBLANCE":
        return "supportive_with_structural_circularity_constraint"
    if classification == "CLEAN_DIRECTIONAL_BRANCH_NOT_CONFIRMED":
        return "constraining/contradictory"
    return "constraining/contradictory_operational"
