"""Regularized small-window Gaussian PhiID for E01 S11.

This module is a separately versioned validation branch.  It intentionally does
not modify the strict S10 source wrappers or claim identity with the unavailable
paper-author implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from e01_information_dynamics.validation import (
    ATOM_IDS,
    I_KEYS,
    KNOWNS_TO_ATOMS,
    aggregate_means,
)

ESTIMATOR_ID = "E01-S11-PHIID-GAUSSIAN-OAS-CROSSFIT-v1.0.0"
COVARIANCE_ID = "E01-S11-COV-OAS-IDENTITY-v1.0.0"
CALIBRATION_ID = "E01-S11-NULL-CENTERING-EXACT-PAIR-v1.0.0"

_SUBSETS = {
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


class SmallWindowError(ValueError):
    """A request violates the frozen S11 small-window contract."""


@dataclass(frozen=True, slots=True)
class SmallWindowResult:
    """One status-bearing local decomposition."""

    status: str
    reason: str | None
    redundancy: Literal["MMI", "CCS"]
    backend: Literal["numpy", "cupy"]
    tau: int
    effective_sample_count: int | None
    atoms: dict[str, np.ndarray] | None
    intermediate_mi: dict[str, np.ndarray] | None
    intermediate_redundancy: dict[str, np.ndarray] | None
    double_redundancy: np.ndarray | None
    diagnostics: dict[str, Any]

    def means(self) -> dict[str, Any] | None:
        if self.atoms is None or self.intermediate_mi is None:
            return None
        return aggregate_means(self.atoms, self.intermediate_mi)


def lagged_four_vector(
    source: np.ndarray, target: np.ndarray, tau: int
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Build ``(x_t,y_t,x_t+tau,y_t+tau)`` without dropping invalid rows."""

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
    if not isinstance(tau, int) or isinstance(tau, bool) or tau < 1 or tau >= source.size:
        return None, {**diagnostics, "reason": "INVALID_LAG"}
    effective = int(source.size - tau)
    diagnostics["effectiveSampleCount"] = effective
    if effective < 24:
        return None, {
            **diagnostics,
            "reason": "EFFECTIVE_SAMPLE_COUNT_BELOW_24",
        }
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        return None, {
            **diagnostics,
            "reason": "NONFINITE_INPUT_NO_ROW_DELETION",
        }
    matrix = np.column_stack(
        [source[:-tau], target[:-tau], source[tau:], target[tau:]]
    )
    return matrix, {**diagnostics, "reason": None}


def contiguous_folds(sample_count: int, fold_count: int = 4) -> tuple[np.ndarray, ...]:
    """Return balanced contiguous evaluation folds covering every row exactly once."""

    if sample_count < 24 or fold_count != 4:
        raise SmallWindowError("The frozen branch requires four folds and n_eff >= 24.")
    folds = tuple(np.asarray(part, dtype=np.int64) for part in np.array_split(np.arange(sample_count), fold_count))
    if min(part.size for part in folds) < 6:
        raise SmallWindowError("Every evaluation fold must contain at least six rows.")
    if not np.array_equal(np.concatenate(folds), np.arange(sample_count)):
        raise SmallWindowError("Contiguous folds do not cover every row exactly once.")
    return folds


def _array_module(backend: Literal["numpy", "cupy"]):
    if backend == "numpy":
        return np
    if backend != "cupy":
        raise SmallWindowError(f"Unknown backend {backend!r}.")
    try:
        import cupy as cp
    except ImportError as error:  # pragma: no cover - exercised only without GPU env
        raise SmallWindowError("CuPy backend requested but CuPy is unavailable.") from error
    return cp


def _to_float(value: Any) -> float:
    if hasattr(value, "get"):
        value = value.get()
    return float(np.asarray(value))


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=np.float64)


def oas_covariance(
    standardized: Any,
    *,
    backend: Literal["numpy", "cupy"] = "numpy",
    shrinkage_multiplier: float = 1.0,
) -> tuple[Any, dict[str, float]]:
    """Return the frozen OAS covariance and auditable diagnostics."""

    xp = _array_module(backend)
    values = xp.asarray(standardized, dtype=xp.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise SmallWindowError("OAS requires a two-dimensional training matrix.")
    if not np.isfinite(shrinkage_multiplier) or shrinkage_multiplier < 0:
        raise SmallWindowError("shrinkage_multiplier must be finite and nonnegative.")
    centered = values - xp.mean(values, axis=0, keepdims=True)
    sample_count, dimension = centered.shape
    empirical = centered.T @ centered / sample_count
    alpha = xp.mean(empirical**2)
    mu = xp.trace(empirical) / dimension
    denominator = (sample_count + 1.0) * (alpha - (mu**2) / dimension)
    if _to_float(denominator) <= 0:
        base_shrinkage = 1.0
    else:
        base_shrinkage = min(_to_float((alpha + mu**2) / denominator), 1.0)
    shrinkage = min(base_shrinkage * float(shrinkage_multiplier), 1.0)
    covariance = (1.0 - shrinkage) * empirical
    diagonal = xp.arange(dimension)
    covariance[diagonal, diagonal] += shrinkage * mu
    eigenvalues = xp.linalg.eigvalsh(covariance)
    minimum = _to_float(eigenvalues[0])
    maximum = _to_float(eigenvalues[-1])
    condition = maximum / minimum if minimum > 0 else math.inf
    return covariance, {
        "baseShrinkage": base_shrinkage,
        "appliedShrinkage": shrinkage,
        "targetScale": _to_float(mu),
        "minimumEigenvalue": minimum,
        "maximumEigenvalue": maximum,
        "conditionNumber": condition,
    }


def _local_entropy(values: Any, covariance: Any, indices: tuple[int, ...], xp: Any) -> Any:
    sub = values[:, list(indices)]
    cov = covariance[xp.ix_(xp.asarray(indices), xp.asarray(indices))]
    sign, logdet = xp.linalg.slogdet(cov)
    if _to_float(sign) <= 0 or not np.isfinite(_to_float(logdet)):
        raise SmallWindowError("REGULARIZED_COVARIANCE_NOT_POSITIVE_DEFINITE")
    solved = xp.linalg.solve(cov, sub.T).T
    mahalanobis = xp.sum(sub * solved, axis=1)
    return 0.5 * (len(indices) * math.log(2.0 * math.pi) + logdet + mahalanobis)


def _mutual_information_arrays(entropies: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    h = entropies
    return {
        "I_xytab": h["h_p1p2"] + h["h_t1t2"] - h["h_p1p2t1t2"],
        "I_xta": h["h_p1"] + h["h_t1"] - h["h_p1t1"],
        "I_xtb": h["h_p1"] + h["h_t2"] - h["h_p1t2"],
        "I_yta": h["h_p2"] + h["h_t1"] - h["h_p2t1"],
        "I_ytb": h["h_p2"] + h["h_t2"] - h["h_p2t2"],
        "I_xyta": h["h_p1p2"] + h["h_t1"] - h["h_p1p2t1"],
        "I_xytb": h["h_p1p2"] + h["h_t2"] - h["h_p1p2t2"],
        "I_xtab": h["h_p1"] + h["h_t1t2"] - h["h_p1t1t2"],
        "I_ytab": h["h_p2"] + h["h_t1t2"] - h["h_p2t1t2"],
    }


def decompose_local_entropies(
    entropies: dict[str, np.ndarray],
    *,
    redundancy: Literal["MMI", "CCS"],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
]:
    """Apply the explicit phyid lattice formulas to supplied local entropies."""

    if set(entropies) != set(_SUBSETS):
        raise SmallWindowError("All and only 15 local entropy fields are required.")
    mi = _mutual_information_arrays(entropies)

    def redundancy_pair(first: np.ndarray, second: np.ndarray, joint: np.ndarray) -> np.ndarray:
        if redundancy == "MMI":
            return first if float(np.mean(first)) < float(np.mean(second)) else second
        if redundancy != "CCS":
            raise SmallWindowError(f"Unknown redundancy {redundancy!r}.")
        coinfo = joint - first - second
        signs = np.stack(
            [np.sign(first), np.sign(second), np.sign(joint), np.sign(-coinfo)],
            axis=1,
        )
        return np.all(signs == signs[:, :1], axis=1) * (-coinfo)

    redundancy_values = {
        "R_xyta": redundancy_pair(mi["I_xta"], mi["I_yta"], mi["I_xyta"]),
        "R_xytb": redundancy_pair(mi["I_xtb"], mi["I_ytb"], mi["I_xytb"]),
        "R_xytab": redundancy_pair(mi["I_xtab"], mi["I_ytab"], mi["I_xytab"]),
        "R_abtx": redundancy_pair(mi["I_xta"], mi["I_xtb"], mi["I_xtab"]),
        "R_abty": redundancy_pair(mi["I_yta"], mi["I_ytb"], mi["I_ytab"]),
        "R_abtxy": redundancy_pair(mi["I_xyta"], mi["I_xytb"], mi["I_xytab"]),
    }
    if redundancy == "MMI":
        candidates = [mi["I_xta"], mi["I_xtb"], mi["I_yta"], mi["I_ytb"]]
        double_redundancy = candidates[
            int(np.argmin([float(np.mean(value)) for value in candidates]))
        ]
    else:
        double_coinfo = (
            -mi["I_xta"]
            - mi["I_xtb"]
            - mi["I_yta"]
            - mi["I_ytb"]
            + mi["I_xtab"]
            + mi["I_ytab"]
            + mi["I_xyta"]
            + mi["I_xytb"]
            - mi["I_xytab"]
            + redundancy_values["R_xyta"]
            + redundancy_values["R_xytb"]
            - redundancy_values["R_xytab"]
            + redundancy_values["R_abtx"]
            + redundancy_values["R_abty"]
            - redundancy_values["R_abtxy"]
        )
        candidates = [
            mi["I_xta"],
            mi["I_xtb"],
            mi["I_yta"],
            mi["I_ytb"],
            double_coinfo,
        ]
        signs = np.stack([np.sign(value) for value in candidates], axis=1)
        double_redundancy = (
            np.all(signs == signs[:, :1], axis=1) * double_coinfo
        )
    knowns = np.column_stack(
        [
            double_redundancy,
            redundancy_values["R_xyta"],
            redundancy_values["R_xytb"],
            redundancy_values["R_xytab"],
            redundancy_values["R_abtx"],
            redundancy_values["R_abty"],
            redundancy_values["R_abtxy"],
            *(mi[key] for key in I_KEYS[1:5]),
            mi["I_xyta"],
            mi["I_xytb"],
            mi["I_xtab"],
            mi["I_ytab"],
            mi["I_xytab"],
        ]
    )
    atoms_matrix = np.linalg.solve(KNOWNS_TO_ATOMS, knowns.T).T
    atoms = {
        atom: np.asarray(atoms_matrix[:, index], dtype=np.float64)
        for index, atom in enumerate(ATOM_IDS)
    }
    return atoms, mi, redundancy_values, np.asarray(double_redundancy, dtype=np.float64)


def run_small_window_phiid(
    source: np.ndarray,
    target: np.ndarray,
    *,
    tau: int,
    redundancy: Literal["MMI", "CCS"],
    backend: Literal["numpy", "cupy"] = "numpy",
    shrinkage_multiplier: float = 1.0,
) -> SmallWindowResult:
    """Run the frozen four-fold OAS cross-fit branch without any fallback."""

    matrix, base = lagged_four_vector(source, target, tau)
    if matrix is None:
        return SmallWindowResult(
            "INELIGIBLE",
            base["reason"],
            redundancy,
            backend,
            tau,
            base.get("effectiveSampleCount"),
            None,
            None,
            None,
            None,
            base,
        )
    if redundancy not in ("MMI", "CCS"):
        raise SmallWindowError("redundancy must be explicitly MMI or CCS.")
    xp = _array_module(backend)
    values = xp.asarray(matrix, dtype=xp.float64)
    folds = contiguous_folds(matrix.shape[0])
    entropies = {key: np.empty(matrix.shape[0], dtype=np.float64) for key in _SUBSETS}
    fold_diagnostics: list[dict[str, Any]] = []
    all_indices = np.arange(matrix.shape[0])
    try:
        for fold_index, evaluation_indices in enumerate(folds):
            training_indices = np.setdiff1d(all_indices, evaluation_indices, assume_unique=True)
            training = values[training_indices]
            evaluation = values[evaluation_indices]
            mean = xp.mean(training, axis=0)
            sd = xp.std(training, axis=0, ddof=1)
            sd_np = _to_numpy(sd)
            if np.any(~np.isfinite(sd_np)) or np.any(sd_np <= 1.0e-12):
                raise SmallWindowError("CONSTANT_OR_NONFINITE_TRAINING_SCALAR")
            training_z = (training - mean) / sd
            evaluation_z = (evaluation - mean) / sd
            covariance, covariance_diagnostics = oas_covariance(
                training_z,
                backend=backend,
                shrinkage_multiplier=shrinkage_multiplier,
            )
            if covariance_diagnostics["minimumEigenvalue"] < 1.0e-10:
                raise SmallWindowError("REGULARIZED_MINIMUM_EIGENVALUE_GATE_FAILED")
            if covariance_diagnostics["conditionNumber"] > 1.0e8:
                raise SmallWindowError("REGULARIZED_CONDITION_NUMBER_GATE_FAILED")
            for key, indices in _SUBSETS.items():
                local = _local_entropy(evaluation_z, covariance, indices, xp)
                entropies[key][evaluation_indices] = _to_numpy(local)
            fold_diagnostics.append(
                {
                    "foldIndex": fold_index,
                    "trainingRows": int(training_indices.size),
                    "evaluationRows": int(evaluation_indices.size),
                    **covariance_diagnostics,
                }
            )
        if not all(np.all(np.isfinite(value)) for value in entropies.values()):
            raise SmallWindowError("NONFINITE_LOCAL_ENTROPY")
        atoms, mi, redundancy_values, double_redundancy = decompose_local_entropies(
            entropies, redundancy=redundancy
        )
        means = aggregate_means(atoms, mi)
        if abs(means["latticeClosureError"]) > 5.0e-10:
            raise SmallWindowError("LATTICE_CLOSURE_GATE_FAILED")
        if abs(means["paperEquationClosureError"]) > 5.0e-10:
            raise SmallWindowError("EQUATION_CLOSURE_GATE_FAILED")
    except (SmallWindowError, np.linalg.LinAlgError) as error:
        return SmallWindowResult(
            "INELIGIBLE",
            str(error),
            redundancy,
            backend,
            tau,
            matrix.shape[0],
            None,
            None,
            None,
            None,
            {
                **base,
                "covarianceId": COVARIANCE_ID,
                "shrinkageMultiplier": shrinkage_multiplier,
                "folds": fold_diagnostics,
            },
        )
    return SmallWindowResult(
        "ELIGIBLE",
        None,
        redundancy,
        backend,
        tau,
        matrix.shape[0],
        atoms,
        mi,
        redundancy_values,
        double_redundancy,
        {
            **base,
            "covarianceId": COVARIANCE_ID,
            "shrinkageMultiplier": shrinkage_multiplier,
            "folds": fold_diagnostics,
            "minimumRegularizedEigenvalue": min(
                item["minimumEigenvalue"] for item in fold_diagnostics
            ),
            "maximumRegularizedConditionNumber": max(
                item["conditionNumber"] for item in fold_diagnostics
            ),
        },
    )


def calibrated_means(
    raw_means: dict[str, Any], calibration_means: dict[str, Any]
) -> dict[str, Any]:
    """Subtract an independent exact-pair null mean while preserving closures."""

    atoms = {
        atom: float(raw_means["atomMeans"][atom] - calibration_means["atomMeans"][atom])
        for atom in ATOM_IDS
    }
    mi = {
        key: float(raw_means["miMeans"][key] - calibration_means["miMeans"][key])
        for key in I_KEYS
    }
    total_atoms = float(sum(atoms.values()))
    past_redundancy = float(sum(atoms[key] for key in ATOM_IDS[:4]))
    past_synergy = float(sum(atoms[key] for key in ATOM_IDS[12:]))
    equation_atoms = past_synergy - past_redundancy
    equation_direct = mi["I_xytab"] - mi["I_xtab"] - mi["I_ytab"]
    return {
        "calibrationId": CALIBRATION_ID,
        "atomMeans": atoms,
        "miMeans": mi,
        "totalAtomSum": total_atoms,
        "totalMi": mi["I_xytab"],
        "latticeClosureError": total_atoms - mi["I_xytab"],
        "pastRedundancy": past_redundancy,
        "pastSynergy": past_synergy,
        "paperEquationAggregateFromAtoms": equation_atoms,
        "paperEquationAggregateDirect": equation_direct,
        "paperEquationClosureError": equation_atoms - equation_direct,
    }


def population_local_entropies(samples: np.ndarray, covariance: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate exact-covariance Gaussian local entropies for a population oracle."""

    samples = np.asarray(samples, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 4 or covariance.shape != (4, 4):
        raise SmallWindowError("Population entropy inputs must have shapes (n,4) and (4,4).")
    result: dict[str, np.ndarray] = {}
    for key, indices in _SUBSETS.items():
        sub = samples[:, indices]
        cov = covariance[np.ix_(indices, indices)]
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise SmallWindowError("Population covariance is not positive definite.")
        solved = np.linalg.solve(cov, sub.T).T
        result[key] = 0.5 * (
            len(indices) * math.log(2.0 * math.pi)
            + logdet
            + np.sum(sub * solved, axis=1)
        )
    return result
