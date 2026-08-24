"""Wishart-corrected local Gaussian PhiID estimator frozen for E01 S11R."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.special import digamma

from e01_information_dynamics.validation import ATOM_IDS, I_KEYS, aggregate_means
from e01_time_localized_phir.estimator import decompose_local_entropies

ESTIMATOR_ID = "E01-S11R-PHIID-GAUSSIAN-WISHART-LOCAL-v1.0.0"
CALIBRATION_ID = "E01-S11R-NULL-CONDITION-MATCHED-COMPLETE-ROW-SHUFFLE-v1.0.0"

_SUBSETS: dict[str, tuple[int, ...]] = {
    "h_p1": (0,),
    "h_p2": (1,),
    "h_t1": (2,),
    "h_t2": (3,),
    "h_p1p2": (0, 1),
    "h_t1t2": (2, 3),
    "h_p1t1": (0, 2),
    "h_p1t2": (0, 3),
    "h_p2t1": (1, 2),
    "h_p2t2": (1, 3),
    "h_p1p2t1": (0, 1, 2),
    "h_p1p2t2": (0, 1, 3),
    "h_p1t1t2": (0, 2, 3),
    "h_p2t1t2": (1, 2, 3),
    "h_p1p2t1t2": (0, 1, 2, 3),
}


class RepairEstimatorError(ValueError):
    """The requested calculation violates the frozen S11R estimator contract."""


@dataclass(frozen=True, slots=True)
class SmallWindowRepairResult:
    """One status-bearing S11R local decomposition."""

    status: str
    reason: str | None
    redundancy: Literal["MMI", "CCS"]
    tau: int
    effective_sample_count: int | None
    atoms: dict[str, np.ndarray] | None
    intermediate_mi: dict[str, np.ndarray] | None
    diagnostics: dict[str, Any]

    def means(self) -> dict[str, Any] | None:
        if self.atoms is None or self.intermediate_mi is None:
            return None
        return aggregate_means(self.atoms, self.intermediate_mi)


def lagged_four_vector(
    source: np.ndarray, target: np.ndarray, tau: int
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Build ``(x_t,y_t,x_t+tau,y_t+tau)`` without deleting rows."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "estimatorId": ESTIMATOR_ID,
        "tau": tau,
        "rawSampleCount": int(source.size) if source.ndim == 1 else None,
        "rowsDeleted": 0,
    }
    if source.ndim != 1 or target.ndim != 1 or source.shape != target.shape:
        return None, {**diagnostics, "reason": "SCALAR_LENGTH_MISMATCH"}
    if (
        not isinstance(tau, int)
        or isinstance(tau, bool)
        or tau < 1
        or tau >= source.size
    ):
        return None, {**diagnostics, "reason": "INVALID_LAG"}
    effective = int(source.size - tau)
    diagnostics["effectiveSampleCount"] = effective
    if effective < 24:
        return None, {**diagnostics, "reason": "EFFECTIVE_SAMPLE_COUNT_BELOW_24"}
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        return None, {**diagnostics, "reason": "NONFINITE_INPUT_NO_ROW_DELETION"}
    return (
        np.column_stack([source[:-tau], target[:-tau], source[tau:], target[tau:]]),
        {**diagnostics, "reason": None},
    )


def wishart_mean_correction(subset_dimension: int, sample_count: int) -> float:
    """Return the preregistered exact Gaussian in-sample entropy correction."""

    p = int(subset_dimension)
    n = int(sample_count)
    if p < 1 or n <= p:
        raise RepairEstimatorError("Wishart correction requires n > p >= 1.")
    nu = n - 1
    expected_logdet_bias = (
        float(sum(digamma((nu + 1 - index) / 2.0) for index in range(1, p + 1)))
        + p * math.log(2.0)
        - p * math.log(float(nu))
    )
    return 0.5 * (p / n - expected_logdet_bias)


def _local_entropy(
    standardized: np.ndarray, covariance: np.ndarray, indices: tuple[int, ...]
) -> np.ndarray:
    block = standardized[:, indices]
    subcovariance = covariance[np.ix_(indices, indices)]
    sign, logdet = np.linalg.slogdet(subcovariance)
    if sign <= 0 or not np.isfinite(logdet):
        raise RepairEstimatorError("SAMPLE_COVARIANCE_NOT_POSITIVE_DEFINITE")
    solved = np.linalg.solve(subcovariance, block.T).T
    mahalanobis = np.sum(block * solved, axis=1)
    local = 0.5 * (len(indices) * math.log(2.0 * math.pi) + logdet + mahalanobis)
    return np.asarray(
        local + wishart_mean_correction(len(indices), standardized.shape[0]),
        dtype=np.float64,
    )


def run_wishart_local_phiid(
    source: np.ndarray,
    target: np.ndarray,
    *,
    tau: int,
    redundancy: Literal["MMI", "CCS"],
) -> SmallWindowRepairResult:
    """Run the frozen S11R estimator in binary64 CPU arithmetic, without fallback."""

    matrix, base = lagged_four_vector(source, target, tau)
    if matrix is None:
        return SmallWindowRepairResult(
            "INELIGIBLE",
            base["reason"],
            redundancy,
            tau,
            base.get("effectiveSampleCount"),
            None,
            None,
            base,
        )
    if redundancy not in ("MMI", "CCS"):
        raise RepairEstimatorError("redundancy must be explicitly MMI or CCS.")
    try:
        mean = np.mean(matrix, axis=0)
        sample_sd = np.std(matrix, axis=0, ddof=1)
        if np.any(~np.isfinite(sample_sd)) or np.any(sample_sd <= 1.0e-12):
            raise RepairEstimatorError("CONSTANT_OR_NONFINITE_SAMPLE_SCALAR")
        standardized = (matrix - mean) / sample_sd
        covariance = np.cov(standardized, rowvar=False, ddof=1)
        if covariance.shape != (4, 4) or not np.all(np.isfinite(covariance)):
            raise RepairEstimatorError("NONFINITE_SAMPLE_COVARIANCE")
        eigenvalues = np.linalg.eigvalsh(covariance)
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
        condition = maximum / minimum if minimum > 0 else math.inf
        if minimum < 1.0e-10:
            raise RepairEstimatorError(
                "SINGULAR_SAMPLE_COVARIANCE_NO_REGULARIZATION_FALLBACK"
            )
        if not np.isfinite(condition) or condition > 1.0e8:
            raise RepairEstimatorError("SAMPLE_COVARIANCE_CONDITION_GATE_FAILED")
        entropies = {
            key: _local_entropy(standardized, covariance, indices)
            for key, indices in _SUBSETS.items()
        }
        atoms, mi, _, _ = decompose_local_entropies(entropies, redundancy=redundancy)
        if not all(
            np.all(np.isfinite(value)) for value in (*atoms.values(), *mi.values())
        ):
            raise RepairEstimatorError("NONFINITE_LOCAL_DECOMPOSITION")
        means = aggregate_means(atoms, mi)
        if abs(means["latticeClosureError"]) > 5.0e-10:
            raise RepairEstimatorError("LATTICE_CLOSURE_GATE_FAILED")
        if abs(means["paperEquationClosureError"]) > 5.0e-10:
            raise RepairEstimatorError("EQUATION_CLOSURE_GATE_FAILED")
    except (RepairEstimatorError, np.linalg.LinAlgError) as error:
        return SmallWindowRepairResult(
            "INELIGIBLE",
            str(error),
            redundancy,
            tau,
            matrix.shape[0],
            None,
            None,
            {
                **base,
                "covarianceDivisor": "n_eff-1",
                "regularization": "none",
                "precision": "IEEE-754 binary64 CPU",
            },
        )
    return SmallWindowRepairResult(
        "ELIGIBLE",
        None,
        redundancy,
        tau,
        matrix.shape[0],
        atoms,
        mi,
        {
            **base,
            "covarianceDivisor": "n_eff-1",
            "regularization": "none",
            "precision": "IEEE-754 binary64 CPU",
            "sampleSd": sample_sd.tolist(),
            "minimumCovarianceEigenvalue": minimum,
            "maximumCovarianceEigenvalue": maximum,
            "covarianceConditionNumber": condition,
            "wishartCorrections": {
                str(p): wishart_mean_correction(p, matrix.shape[0]) for p in range(1, 5)
            },
        },
    )


def calibrate_means(
    raw_means: dict[str, Any], calibration_means: dict[str, Any]
) -> dict[str, Any]:
    """Subtract a condition-matched bank mean while preserving both closures."""

    atoms = {
        atom: float(raw_means["atomMeans"][atom] - calibration_means["atomMeans"][atom])
        for atom in ATOM_IDS
    }
    mi = {
        key: float(raw_means["miMeans"][key] - calibration_means["miMeans"][key])
        for key in I_KEYS
    }
    total = float(sum(atoms.values()))
    redundancy_total = float(sum(atoms[key] for key in ATOM_IDS[:4]))
    synergy_total = float(sum(atoms[key] for key in ATOM_IDS[12:]))
    equation_atoms = synergy_total - redundancy_total
    equation_direct = mi["I_xytab"] - mi["I_xtab"] - mi["I_ytab"]
    return {
        "calibrationId": CALIBRATION_ID,
        "atomMeans": atoms,
        "miMeans": mi,
        "totalAtomSum": total,
        "totalMi": mi["I_xytab"],
        "latticeClosureError": total - mi["I_xytab"],
        "pastRedundancy": redundancy_total,
        "pastSynergy": synergy_total,
        "paperEquationAggregateFromAtoms": equation_atoms,
        "paperEquationAggregateDirect": equation_direct,
        "paperEquationClosureError": equation_atoms - equation_direct,
    }
