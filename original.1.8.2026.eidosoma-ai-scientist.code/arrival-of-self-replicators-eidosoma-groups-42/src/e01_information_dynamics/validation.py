"""Independent analytic oracles, gates, and partition utilities for E01 S10."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any, Literal

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

ATOM_IDS = (
    "rtr",
    "rtx",
    "rty",
    "rts",
    "xtr",
    "xtx",
    "xty",
    "xts",
    "ytr",
    "ytx",
    "yty",
    "yts",
    "str",
    "stx",
    "sty",
    "sts",
)
I_KEYS = (
    "I_xytab",
    "I_xta",
    "I_xtb",
    "I_yta",
    "I_ytb",
    "I_xyta",
    "I_xytb",
    "I_xtab",
    "I_ytab",
)

KNOWNS_TO_ATOMS = np.asarray(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.float64,
)


class InformationValidationError(ValueError):
    """A validation specification or numerical input is invalid."""


def aggregate_means(
    atoms: dict[str, np.ndarray],
    intermediate_mi: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Return all atom means and preregistered aggregate identities."""

    if set(atoms) != set(ATOM_IDS):
        raise InformationValidationError(
            "All and only 16 catalogued atoms are required."
        )
    if set(intermediate_mi) != set(I_KEYS):
        raise InformationValidationError(
            "All and only nine intermediate MI fields are required."
        )
    atom_means = {key: float(np.mean(atoms[key])) for key in ATOM_IDS}
    mi_means = {key: float(np.mean(intermediate_mi[key])) for key in I_KEYS}
    total_atoms = float(sum(atom_means.values()))
    past_redundancy = float(sum(atom_means[key] for key in ATOM_IDS[:4]))
    past_synergy = float(sum(atom_means[key] for key in ATOM_IDS[12:]))
    equation_atoms = past_synergy - past_redundancy
    equation_direct = mi_means["I_xytab"] - mi_means["I_xtab"] - mi_means["I_ytab"]
    return {
        "atomMeans": atom_means,
        "miMeans": mi_means,
        "totalAtomSum": total_atoms,
        "totalMi": mi_means["I_xytab"],
        "latticeClosureError": total_atoms - mi_means["I_xytab"],
        "pastRedundancy": past_redundancy,
        "pastSynergy": past_synergy,
        "paperEquationAggregateFromAtoms": equation_atoms,
        "paperEquationAggregateDirect": equation_direct,
        "paperEquationClosureError": equation_atoms - equation_direct,
    }


def strict_sample_gate(
    source: np.ndarray,
    target: np.ndarray,
    *,
    tau: int,
    kind: Literal["gaussian", "discrete"],
) -> dict[str, Any]:
    """Apply the preregistered strict scalar 2x2 sample gate without deletion."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    base = {
        "specificationId": "E01-S10-SAMPLE-GATE-STRICT-v1.0.0",
        "kind": kind,
        "tau": tau,
        "rawSampleCount": int(source.size) if source.ndim == 1 else None,
    }
    if source.ndim != 1 or target.ndim != 1 or source.shape != target.shape:
        return {**base, "status": "INELIGIBLE", "reason": "SCALAR_LENGTH_MISMATCH"}
    if (
        not isinstance(tau, int)
        or isinstance(tau, bool)
        or tau < 1
        or tau >= source.size
    ):
        return {**base, "status": "INELIGIBLE", "reason": "INVALID_TAU"}
    effective = int(source.size - tau)
    base["effectiveSampleCount"] = effective
    if effective < max(512, 20 * 4):
        return {
            **base,
            "status": "INELIGIBLE",
            "reason": "INSUFFICIENT_EFFECTIVE_SAMPLES",
        }
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        return {
            **base,
            "status": "INELIGIBLE",
            "reason": "NONFINITE_INPUT_NO_ROW_DELETION",
        }
    if kind == "discrete":
        source_binary = source > np.mean(source)
        target_binary = target > np.mean(target)
        if np.unique(source_binary).size != 2 or np.unique(target_binary).size != 2:
            return {
                **base,
                "status": "INELIGIBLE",
                "reason": "BINARY_MARGINAL_STATE_MISSING",
            }
        return {**base, "status": "ELIGIBLE", "reason": None, "rowsDeleted": 0}
    if kind != "gaussian":
        return {**base, "status": "INELIGIBLE", "reason": "UNKNOWN_ESTIMATOR_KIND"}
    matrix = np.vstack([source[:-tau], target[:-tau], source[tau:], target[tau:]])
    standard_deviations = np.std(matrix, axis=1, ddof=1)
    if np.any(~np.isfinite(standard_deviations)) or np.any(standard_deviations <= 0):
        return {**base, "status": "INELIGIBLE", "reason": "ZERO_OR_NONFINITE_SAMPLE_SD"}
    normalized = matrix / standard_deviations[:, None]
    covariance = np.cov(normalized)
    singular_values = np.linalg.svd(covariance, compute_uv=False)
    largest = float(singular_values[0])
    rank_tolerance = float(max(effective, 4) * np.finfo(np.float64).eps * largest)
    rank = int(np.sum(singular_values > rank_tolerance))
    condition = (
        float(largest / singular_values[-1]) if singular_values[-1] > 0 else math.inf
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    base.update(
        {
            "jointDimension": 4,
            "numericalRank": rank,
            "rankTolerance": rank_tolerance,
            "conditionNumber": condition,
            "minimumEigenvalue": float(eigenvalues[0]),
            "rowsDeleted": 0,
        }
    )
    if rank != 4:
        return {
            **base,
            "status": "INELIGIBLE",
            "reason": "JOINT_COVARIANCE_RANK_DEFICIENT",
        }
    if eigenvalues[0] <= 0:
        return {
            **base,
            "status": "INELIGIBLE",
            "reason": "JOINT_COVARIANCE_NOT_POSITIVE_DEFINITE",
        }
    if not np.isfinite(condition) or condition > 1.0e12:
        return {
            **base,
            "status": "INELIGIBLE",
            "reason": "JOINT_COVARIANCE_ILL_CONDITIONED",
        }
    return {**base, "status": "ELIGIBLE", "reason": None}


def _logdet(matrix: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(np.asarray(matrix, dtype=np.float64))
    if sign <= 0 or not np.isfinite(value):
        raise InformationValidationError(
            "Covariance submatrix is not positive definite."
        )
    return float(value)


def gaussian_mutual_information(
    covariance: np.ndarray,
    source_indices: tuple[int, ...] | list[int],
    target_indices: tuple[int, ...] | list[int],
) -> float:
    """Analytic Gaussian mutual information in nats from a covariance matrix."""

    source_indices = tuple(source_indices)
    target_indices = tuple(target_indices)
    if (
        not source_indices
        or not target_indices
        or set(source_indices) & set(target_indices)
    ):
        raise InformationValidationError("MI subsets must be nonempty and disjoint.")
    covariance = np.asarray(covariance, dtype=np.float64)
    joint = source_indices + target_indices
    return 0.5 * (
        _logdet(covariance[np.ix_(source_indices, source_indices)])
        + _logdet(covariance[np.ix_(target_indices, target_indices)])
        - _logdet(covariance[np.ix_(joint, joint)])
    )


def gaussian_entropy(covariance: np.ndarray, indices: tuple[int, ...]) -> float:
    """Differential entropy of a Gaussian subvector in nats."""

    return 0.5 * (
        len(indices) * math.log(2.0 * math.pi * math.e)
        + _logdet(covariance[np.ix_(indices, indices)])
    )


def _mi_means_from_covariance(covariance: np.ndarray) -> dict[str, float]:
    mi = lambda a, b: gaussian_mutual_information(covariance, a, b)
    return {
        "I_xytab": mi((0, 1), (2, 3)),
        "I_xta": mi((0,), (2,)),
        "I_xtb": mi((0,), (3,)),
        "I_yta": mi((1,), (2,)),
        "I_ytb": mi((1,), (3,)),
        "I_xyta": mi((0, 1), (2,)),
        "I_xytb": mi((0, 1), (3,)),
        "I_xtab": mi((0,), (2, 3)),
        "I_ytab": mi((1,), (2, 3)),
    }


def _atom_means_from_knowns(
    mi: dict[str, float],
    redundancies: dict[str, float],
    double_redundancy: float,
) -> dict[str, float]:
    knowns = np.asarray(
        [
            double_redundancy,
            redundancies["R_xyta"],
            redundancies["R_xytb"],
            redundancies["R_xytab"],
            redundancies["R_abtx"],
            redundancies["R_abty"],
            redundancies["R_abtxy"],
            mi["I_xta"],
            mi["I_xtb"],
            mi["I_yta"],
            mi["I_ytb"],
            mi["I_xyta"],
            mi["I_xytb"],
            mi["I_xtab"],
            mi["I_ytab"],
            mi["I_xytab"],
        ],
        dtype=np.float64,
    )
    values = np.linalg.solve(KNOWNS_TO_ATOMS, knowns)
    return {atom: float(values[index]) for index, atom in enumerate(ATOM_IDS)}


def gaussian_mmi_oracle(covariance: np.ndarray) -> dict[str, Any]:
    """Population MMI PhiID atom means for a known four-variable covariance."""

    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (4, 4):
        raise InformationValidationError(
            "Gaussian MMI oracle requires a 4x4 covariance."
        )
    mi = _mi_means_from_covariance(covariance)
    redundancies = {
        "R_xyta": min(mi["I_xta"], mi["I_yta"]),
        "R_xytb": min(mi["I_xtb"], mi["I_ytb"]),
        "R_xytab": min(mi["I_xtab"], mi["I_ytab"]),
        "R_abtx": min(mi["I_xta"], mi["I_xtb"]),
        "R_abty": min(mi["I_yta"], mi["I_ytb"]),
        "R_abtxy": min(mi["I_xyta"], mi["I_xytb"]),
    }
    double_redundancy = min(mi["I_xta"], mi["I_xtb"], mi["I_yta"], mi["I_ytb"])
    atoms = _atom_means_from_knowns(mi, redundancies, double_redundancy)
    return {
        "atomMeans": atoms,
        "miMeans": mi,
        "redundancyMeans": redundancies,
        "doubleRedundancyMean": double_redundancy,
        "totalMi": mi["I_xytab"],
        "paperEquationAggregate": mi["I_xytab"] - mi["I_xtab"] - mi["I_ytab"],
    }


def noisy_redundant_covariance(
    *, rho: float = 0.9, innovation_sd: float = 1.0, observation_sd: float = 0.35
) -> np.ndarray:
    """Population covariance of two noisy observations of one AR(1) latent."""

    latent_variance = innovation_sd**2 / (1.0 - rho**2)
    same = latent_variance + observation_sd**2
    cross_same = latent_variance
    cross_lag = rho * latent_variance
    return np.asarray(
        [
            [same, cross_same, cross_lag, cross_lag],
            [cross_same, same, cross_lag, cross_lag],
            [cross_lag, cross_lag, same, cross_same],
            [cross_lag, cross_lag, cross_same, same],
        ],
        dtype=np.float64,
    )


def coupled_ar_covariance(
    transition: np.ndarray | None = None,
    innovation_sd: np.ndarray | None = None,
) -> np.ndarray:
    """Population (x_t,y_t,x_t+1,y_t+1) covariance for a stable VAR(1)."""

    transition = (
        np.asarray([[0.0, 0.0], [0.85, 0.25]], dtype=np.float64)
        if transition is None
        else np.asarray(transition, dtype=np.float64)
    )
    innovation_sd = (
        np.asarray([1.0, 0.5], dtype=np.float64)
        if innovation_sd is None
        else np.asarray(innovation_sd, dtype=np.float64)
    )
    if transition.shape != (2, 2) or innovation_sd.shape != (2,):
        raise InformationValidationError("Coupled AR inputs have invalid dimensions.")
    if np.max(np.abs(np.linalg.eigvals(transition))) >= 1:
        raise InformationValidationError("Coupled AR transition is not stable.")
    stationary = solve_discrete_lyapunov(transition, np.diag(innovation_sd**2))
    past_future = stationary @ transition.T
    return np.block([[stationary, past_future], [past_future.T, stationary]])


def exact_redundant_pmf(flip_probability: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Exact stationary four-bit PMF for redundant Markov copies."""

    states: list[list[int]] = []
    probabilities: list[float] = []
    for past in (0, 1):
        for future in (0, 1):
            transition_probability = (
                1.0 - flip_probability if future == past else flip_probability
            )
            states.append([past, past, future, future])
            probabilities.append(0.5 * transition_probability)
    return np.asarray(states, dtype=np.int8), np.asarray(
        probabilities, dtype=np.float64
    )


def exact_xor_pmf() -> tuple[np.ndarray, np.ndarray]:
    """Exact stationary four-bit PMF for x'=x XOR y and random y'."""

    states: list[list[int]] = []
    for x, y, future_y in itertools.product((0, 1), repeat=3):
        states.append([x, y, x ^ y, future_y])
    return np.asarray(states, dtype=np.int8), np.full(8, 1.0 / 8.0)


def discrete_exact_oracle(
    states: np.ndarray,
    probabilities: np.ndarray,
    *,
    redundancy: Literal["MMI", "CCS"],
) -> dict[str, Any]:
    """Exact local-probability PhiID oracle in nats for a four-bit PMF."""

    states = np.asarray(states, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        states.ndim != 2
        or states.shape[1] != 4
        or probabilities.shape != (states.shape[0],)
    ):
        raise InformationValidationError("Discrete oracle inputs have invalid shape.")
    if np.any(probabilities <= 0) or not np.isclose(np.sum(probabilities), 1.0):
        raise InformationValidationError(
            "Discrete oracle probabilities must be positive and sum to one."
        )

    def entropy(indices: tuple[int, ...]) -> np.ndarray:
        marginal: dict[tuple[int, ...], float] = defaultdict(float)
        for row, probability in zip(states, probabilities, strict=True):
            marginal[tuple(int(row[index]) for index in indices)] += float(probability)
        return np.asarray(
            [
                -math.log(marginal[tuple(int(row[index]) for index in indices)])
                for row in states
            ],
            dtype=np.float64,
        )

    h = {
        "h_p1": entropy((0,)),
        "h_p2": entropy((1,)),
        "h_t1": entropy((2,)),
        "h_t2": entropy((3,)),
        "h_p1p2": entropy((0, 1)),
        "h_t1t2": entropy((2, 3)),
        "h_p1t1": entropy((0, 2)),
        "h_p1t2": entropy((0, 3)),
        "h_p2t1": entropy((1, 2)),
        "h_p2t2": entropy((1, 3)),
        "h_p1p2t1": entropy((0, 1, 2)),
        "h_p1p2t2": entropy((0, 1, 3)),
        "h_p1t1t2": entropy((0, 2, 3)),
        "h_p2t1t2": entropy((1, 2, 3)),
        "h_p1p2t1t2": entropy((0, 1, 2, 3)),
    }
    mi = {
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

    def weighted_mean(values: np.ndarray) -> float:
        return float(np.sum(probabilities * values))

    def redundancy_pair(
        first: np.ndarray, second: np.ndarray, joint: np.ndarray
    ) -> np.ndarray:
        if redundancy == "MMI":
            return first if weighted_mean(first) < weighted_mean(second) else second
        coinfo = joint - first - second
        signs = np.stack(
            [np.sign(first), np.sign(second), np.sign(joint), np.sign(-coinfo)], axis=1
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
        choices = [mi["I_xta"], mi["I_xtb"], mi["I_yta"], mi["I_ytb"]]
        double_redundancy = choices[
            int(np.argmin([weighted_mean(value) for value in choices]))
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
        values = [mi["I_xta"], mi["I_xtb"], mi["I_yta"], mi["I_ytb"], double_coinfo]
        signs = np.stack([np.sign(value) for value in values], axis=1)
        double_redundancy = np.all(signs == signs[:, :1], axis=1) * double_coinfo

    knowns = np.column_stack(
        [
            double_redundancy,
            redundancy_values["R_xyta"],
            redundancy_values["R_xytb"],
            redundancy_values["R_xytab"],
            redundancy_values["R_abtx"],
            redundancy_values["R_abty"],
            redundancy_values["R_abtxy"],
            mi["I_xta"],
            mi["I_xtb"],
            mi["I_yta"],
            mi["I_ytb"],
            mi["I_xyta"],
            mi["I_xytb"],
            mi["I_xtab"],
            mi["I_ytab"],
            mi["I_xytab"],
        ]
    )
    atom_local = np.linalg.solve(KNOWNS_TO_ATOMS, knowns.T).T
    atom_means = {
        atom: weighted_mean(atom_local[:, index]) for index, atom in enumerate(ATOM_IDS)
    }
    mi_means = {key: weighted_mean(value) for key, value in mi.items()}
    return {
        "atomMeans": atom_means,
        "miMeans": mi_means,
        "totalMi": mi_means["I_xytab"],
        "paperEquationAggregate": mi_means["I_xytab"]
        - mi_means["I_xtab"]
        - mi_means["I_ytab"],
        "supportSize": int(states.shape[0]),
        "units": "nats",
    }


def all_bipartitions(
    dimension: int, *, balanced_only: bool = False
) -> list[tuple[int, ...]]:
    """Enumerate unordered nontrivial bipartitions with component zero in A."""

    if dimension < 2:
        raise InformationValidationError("At least two components are required.")
    result: list[tuple[int, ...]] = []
    universe = tuple(range(dimension))
    for size in range(1, dimension):
        for candidate in itertools.combinations(universe[1:], size - 1):
            part_a = (0, *candidate)
            if len(part_a) == dimension:
                continue
            if balanced_only and abs(len(part_a) - (dimension - len(part_a))) > 1:
                continue
            result.append(part_a)
    return result


def complement(part_a: tuple[int, ...], dimension: int) -> tuple[int, ...]:
    return tuple(index for index in range(dimension) if index not in set(part_a))


def _sample_covariance(
    blocks: list[np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    arrays: list[np.ndarray] = []
    groups: list[tuple[int, ...]] = []
    offset = 0
    for block in blocks:
        block = np.asarray(block, dtype=np.float64)
        if block.ndim == 1:
            block = block[:, None]
        if block.ndim != 2:
            raise InformationValidationError(
                "Partition blocks must be vectors or matrices."
            )
        arrays.append(block)
        groups.append(tuple(range(offset, offset + block.shape[1])))
        offset += block.shape[1]
    matrix = np.concatenate(arrays, axis=1)
    if not np.all(np.isfinite(matrix)):
        raise InformationValidationError("Partition input contains nonfinite values.")
    covariance = np.cov(matrix, rowvar=False)
    covariance = np.atleast_2d(covariance)
    return covariance, groups


def _pc1_score(block: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    block = np.asarray(block, dtype=np.float64)
    centered = block - np.mean(block, axis=0, keepdims=True)
    if block.shape[1] == 1:
        return centered[:, 0], {
            "leadingEigenvalue": float(np.var(centered[:, 0], ddof=1)),
            "relativeEigengap": math.inf,
        }
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values, kind="stable")[::-1]
    values = values[order]
    vector = vectors[:, order[0]]
    scale = max(abs(float(values[0])), np.finfo(float).tiny)
    relative_gap = float((values[0] - values[1]) / scale)
    if relative_gap <= 1.0e-10:
        raise InformationValidationError("PC1_LEADING_EIGENTIE")
    largest = np.max(np.abs(vector))
    pivot = int(np.flatnonzero(np.abs(vector) == largest)[0])
    if vector[pivot] < 0:
        vector = -vector
    return centered @ vector, {
        "leadingEigenvalue": float(values[0]),
        "relativeEigengap": relative_gap,
    }


def map_partition(
    data: np.ndarray,
    part_a: tuple[int, ...],
    *,
    mapping: Literal["group_mean", "pc1", "omega_equal_width_vector"],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map a named component partition to explicit scalar or vector parts."""

    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise InformationValidationError(
            "Partition data must be observations by components."
        )
    if 0 not in part_a or len(part_a) == 0 or len(part_a) == data.shape[1]:
        raise InformationValidationError("Partition must be canonical and nontrivial.")
    part_b = complement(part_a, data.shape[1])
    block_a = data[:, part_a]
    block_b = data[:, part_b]
    if mapping == "group_mean":
        return np.mean(block_a, axis=1), np.mean(block_b, axis=1), {}
    if mapping == "pc1":
        score_a, diag_a = _pc1_score(block_a)
        score_b, diag_b = _pc1_score(block_b)
        return score_a, score_b, {"partA": diag_a, "partB": diag_b}
    if mapping == "omega_equal_width_vector":
        if block_a.shape[1] != block_b.shape[1]:
            raise InformationValidationError("OMEGA_VECTOR_REQUIRES_EQUAL_WIDTH_PARTS")
        return block_a, block_b, {"featureWidth": int(block_a.shape[1])}
    raise InformationValidationError(f"Unknown partition mapping {mapping}.")


def gaussian_partition_objective(
    data: np.ndarray,
    part_a: tuple[int, ...],
    *,
    mapping: Literal["group_mean", "pc1", "omega_equal_width_vector"],
    objective: Literal[
        "synchronous_mi", "bidirectional_lagged_mi", "abs_paper_equation"
    ],
    normalization: Literal["none", "min_part_entropy", "geometric_part_size"],
) -> dict[str, Any]:
    """Evaluate one fully named Gaussian partition candidate."""

    try:
        part_a_values, part_b_values, diagnostics = map_partition(
            data, part_a, mapping=mapping
        )
        a = np.asarray(part_a_values)
        b = np.asarray(part_b_values)
        if objective == "synchronous_mi":
            covariance, groups = _sample_covariance([a, b])
            raw = gaussian_mutual_information(covariance, groups[0], groups[1])
        elif objective == "bidirectional_lagged_mi":
            covariance, groups = _sample_covariance([a[:-1], b[:-1], a[1:], b[1:]])
            raw = gaussian_mutual_information(
                covariance, groups[0], groups[3]
            ) + gaussian_mutual_information(covariance, groups[1], groups[2])
        elif objective == "abs_paper_equation":
            covariance, groups = _sample_covariance([a[:-1], b[:-1], a[1:], b[1:]])
            full_past = groups[0] + groups[1]
            full_future = groups[2] + groups[3]
            total = gaussian_mutual_information(covariance, full_past, full_future)
            source_a = gaussian_mutual_information(covariance, groups[0], full_future)
            source_b = gaussian_mutual_information(covariance, groups[1], full_future)
            raw = abs(total - source_a - source_b)
        else:
            raise InformationValidationError(f"Unknown objective {objective}.")

        part_b = complement(part_a, np.asarray(data).shape[1])
        if normalization == "none":
            denominator = 1.0
        elif normalization == "geometric_part_size":
            denominator = math.sqrt(len(part_a) * len(part_b))
        elif normalization == "min_part_entropy":
            covariance, groups = _sample_covariance([a, b])
            denominator = min(
                gaussian_entropy(covariance, groups[0]),
                gaussian_entropy(covariance, groups[1]),
            )
            if not np.isfinite(denominator) or denominator <= 0:
                raise InformationValidationError(
                    "NONPOSITIVE_PART_ENTROPY_NORMALIZATION"
                )
        else:
            raise InformationValidationError(f"Unknown normalization {normalization}.")
        value = float(raw / denominator)
        if not np.isfinite(value):
            raise InformationValidationError("NONFINITE_PARTITION_OBJECTIVE")
        return {
            "status": "ELIGIBLE",
            "reason": None,
            "partA": list(part_a),
            "partB": list(part_b),
            "rawObjective": float(raw),
            "normalizationDenominator": float(denominator),
            "normalizedObjective": value,
            "mappingDiagnostics": diagnostics,
        }
    except (InformationValidationError, np.linalg.LinAlgError) as error:
        return {
            "status": "INELIGIBLE",
            "reason": str(error),
            "partA": list(part_a),
            "partB": list(complement(part_a, np.asarray(data).shape[1])),
            "rawObjective": None,
            "normalizationDenominator": None,
            "normalizedObjective": None,
            "mappingDiagnostics": {},
        }


def exhaustive_partition_search(
    data: np.ndarray,
    *,
    mapping: str,
    objective: str,
    normalization: str,
    balanced_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the frozen lexicographic exhaustive minimum and all candidates."""

    candidates = [
        gaussian_partition_objective(
            data,
            part,
            mapping=mapping,  # type: ignore[arg-type]
            objective=objective,  # type: ignore[arg-type]
            normalization=normalization,  # type: ignore[arg-type]
        )
        for part in all_bipartitions(
            np.asarray(data).shape[1], balanced_only=balanced_only
        )
    ]
    eligible = [item for item in candidates if item["status"] == "ELIGIBLE"]
    if not eligible:
        return {"status": "INELIGIBLE", "reason": "NO_ELIGIBLE_PARTITION"}, candidates
    winner = min(
        eligible, key=lambda item: (item["normalizedObjective"], tuple(item["partA"]))
    )
    return winner, candidates


def spectral_partition(data: np.ndarray) -> dict[str, Any]:
    """Return the frozen absolute-correlation normalized-Laplacian candidate."""

    data = np.asarray(data, dtype=np.float64)
    adjacency = np.abs(np.corrcoef(data, rowvar=False))
    np.fill_diagonal(adjacency, 0.0)
    degrees = np.sum(adjacency, axis=1)
    if np.any(degrees <= 0) or not np.all(np.isfinite(degrees)):
        return {"status": "INELIGIBLE", "reason": "SPECTRAL_ZERO_OR_NONFINITE_DEGREE"}
    inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    laplacian = np.eye(data.shape[1]) - inv_sqrt @ adjacency @ inv_sqrt
    values, vectors = np.linalg.eigh(laplacian)
    if len(values) < 3:
        return {"status": "INELIGIBLE", "reason": "SPECTRAL_DIMENSION_TOO_SMALL"}
    scale = max(abs(float(values[2])), 1.0)
    eigengap = float((values[2] - values[1]) / scale)
    if eigengap <= 1.0e-10:
        return {
            "status": "INELIGIBLE",
            "reason": "SPECTRAL_FIEDLER_EIGENTIE",
            "relativeEigengap": eigengap,
        }
    vector = vectors[:, 1]
    largest = np.max(np.abs(vector))
    pivot = int(np.flatnonzero(np.abs(vector) == largest)[0])
    if vector[pivot] < 0:
        vector = -vector
    part = tuple(int(index) for index in np.flatnonzero(vector >= 0))
    if not part or len(part) == data.shape[1]:
        return {
            "status": "INELIGIBLE",
            "reason": "SPECTRAL_EMPTY_PART",
            "relativeEigengap": eigengap,
        }
    if 0 not in part:
        part = complement(part, data.shape[1])
    return {
        "status": "ELIGIBLE",
        "reason": None,
        "partA": list(part),
        "partB": list(complement(part, data.shape[1])),
        "relativeEigengap": eigengap,
    }


def greedy_partition_search(
    data: np.ndarray,
    *,
    mapping: str,
    objective: str,
    normalization: str,
) -> dict[str, Any]:
    """Apply the preregistered deterministic single-flip local search."""

    dimension = np.asarray(data).shape[1]
    current = (0,)
    current_result = gaussian_partition_objective(
        data,
        current,
        mapping=mapping,  # type: ignore[arg-type]
        objective=objective,  # type: ignore[arg-type]
        normalization=normalization,  # type: ignore[arg-type]
    )
    if current_result["status"] != "ELIGIBLE":
        return {"status": "INELIGIBLE", "reason": "GREEDY_START_INELIGIBLE"}
    iterations = 0
    while True:
        alternatives: list[dict[str, Any]] = []
        for component in range(1, dimension):
            candidate_set = set(current)
            if component in candidate_set:
                candidate_set.remove(component)
            else:
                candidate_set.add(component)
            if len(candidate_set) == dimension:
                continue
            candidate = tuple(sorted(candidate_set))
            result = gaussian_partition_objective(
                data,
                candidate,
                mapping=mapping,  # type: ignore[arg-type]
                objective=objective,  # type: ignore[arg-type]
                normalization=normalization,  # type: ignore[arg-type]
            )
            if result["status"] == "ELIGIBLE":
                alternatives.append(result)
        if not alternatives:
            break
        best = min(
            alternatives,
            key=lambda item: (item["normalizedObjective"], tuple(item["partA"])),
        )
        if (
            current_result["normalizedObjective"] - best["normalizedObjective"]
            <= 1.0e-12
        ):
            break
        current = tuple(best["partA"])
        current_result = best
        iterations += 1
        if iterations > dimension * dimension:
            raise InformationValidationError("GREEDY_ITERATION_BOUND_EXCEEDED")
    return {**current_result, "iterations": iterations}
