"""Fail-closed, status-bearing compositional transforms for E01 S09.

The functions here implement named reconstruction/validation branches.  They do
not infer the paper authors' zero policy or preprocessing implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

ZeroMethod = Literal["additive_pseudocount", "multiplicative_replacement", "none"]
CoordinateFamily = Literal[
    "full_clr",
    "dropped_clr",
    "ilr_helmert",
    "raw_proportions",
    "hellinger",
    "principal_log_ratio",
]

_SENTINEL_PREFIXES = ("UNRESOLVED::", "CONFLICT::", "BRANCH_SET::", "UNAVAILABLE::")


class CompositionalContractError(ValueError):
    """An input or configuration violates the frozen S09 contract."""


def _explicit_string(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CompositionalContractError(f"{name} must be an explicit nonempty string.")
    if value.startswith(_SENTINEL_PREFIXES):
        raise CompositionalContractError(
            f"{name} contains a non-executable registry sentinel: {value}."
        )


@dataclass(frozen=True, slots=True)
class ZeroTreatment:
    """A complete zero treatment with no implicit numerical setting."""

    specification_id: str
    method: ZeroMethod
    delta: float | None
    evidence_class: str

    def __post_init__(self) -> None:
        _explicit_string(self.specification_id, name="specification_id")
        _explicit_string(self.evidence_class, name="evidence_class")
        if self.method not in (
            "additive_pseudocount",
            "multiplicative_replacement",
            "none",
        ):
            raise CompositionalContractError(
                f"Unsupported zero method {self.method!r}."
            )
        if self.method == "none":
            if self.delta is not None:
                raise CompositionalContractError(
                    "The no-replacement branch forbids delta."
                )
        elif (
            self.delta is None
            or not np.isfinite(self.delta)
            or isinstance(self.delta, bool)
            or self.delta <= 0
        ):
            raise CompositionalContractError(
                f"{self.method} requires an explicit finite positive delta."
            )


@dataclass(frozen=True, slots=True)
class CoordinateSpecification:
    """A complete coordinate mapping, including dropped component or basis scope."""

    specification_id: str
    family: CoordinateFamily
    dimension: int
    evidence_class: str
    dropped_component_zero_based: int | None
    basis_fit_scope_id: str | None

    def __post_init__(self) -> None:
        _explicit_string(self.specification_id, name="specification_id")
        _explicit_string(self.evidence_class, name="evidence_class")
        if self.family not in (
            "full_clr",
            "dropped_clr",
            "ilr_helmert",
            "raw_proportions",
            "hellinger",
            "principal_log_ratio",
        ):
            raise CompositionalContractError(
                f"Unsupported coordinate family {self.family!r}."
            )
        if (
            not isinstance(self.dimension, int)
            or isinstance(self.dimension, bool)
            or self.dimension < 2
        ):
            raise CompositionalContractError("dimension must be an integer >= 2.")
        if self.family == "dropped_clr":
            if (
                self.dropped_component_zero_based is None
                or not 0 <= self.dropped_component_zero_based < self.dimension
            ):
                raise CompositionalContractError(
                    "dropped_clr requires an explicit component within dimension."
                )
        elif self.dropped_component_zero_based is not None:
            raise CompositionalContractError(
                "Only dropped_clr may name a dropped component."
            )
        if self.family == "principal_log_ratio":
            if self.basis_fit_scope_id is None:
                raise CompositionalContractError(
                    "principal_log_ratio requires an explicit basis_fit_scope_id."
                )
            _explicit_string(self.basis_fit_scope_id, name="basis_fit_scope_id")
        elif self.basis_fit_scope_id is not None:
            raise CompositionalContractError(
                "Only principal_log_ratio may name a basis-fit scope."
            )


@dataclass(frozen=True, slots=True)
class TreatedComposition:
    """Status-bearing zero-treatment result for one retained observation."""

    status: str
    reason: str | None
    composition: NDArray[np.float64] | None
    input_mass: float
    zero_count: int
    replacement_mass_per_zero: float | None


@dataclass(frozen=True, slots=True)
class TransformResult:
    """Status-bearing coordinate result for one retained observation."""

    status: str
    reason: str | None
    coordinates: NDArray[np.float64] | None
    reconstructed_composition: NDArray[np.float64] | None
    maximum_absolute_inverse_error: float | None
    maximum_relative_inverse_error: float | None
    closure_error: float | None


def _state_vector(values: ArrayLike) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 2:
        raise CompositionalContractError(
            "state must be a one-dimensional vector with at least two components."
        )
    if not np.all(np.isfinite(vector)):
        raise CompositionalContractError("state must contain only finite values.")
    if np.any(vector < 0):
        raise CompositionalContractError("state must be nonnegative.")
    return vector.copy()


def _close(values: ArrayLike) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)) or np.any(vector < 0):
        raise CompositionalContractError(
            "closure requires a finite nonnegative one-dimensional vector."
        )
    total = float(np.sum(vector, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise CompositionalContractError(
            "closure requires strictly positive total mass."
        )
    result = vector / total
    if not np.all(np.isfinite(result)):
        raise CompositionalContractError("closure produced a nonfinite value.")
    return result


def apply_zero_treatment(
    state: ArrayLike,
    treatment: ZeroTreatment,
) -> TreatedComposition:
    """Apply one explicit zero policy without deleting the input observation."""

    vector = _state_vector(state)
    dimension = vector.size
    total = float(np.sum(vector, dtype=np.float64))
    zero_count = int(np.count_nonzero(vector == 0.0))

    if treatment.method == "additive_pseudocount":
        assert treatment.delta is not None
        denominator = total + dimension * treatment.delta
        composition = (vector + treatment.delta) / denominator
        replacement = treatment.delta / denominator
    elif treatment.method == "multiplicative_replacement":
        assert treatment.delta is not None
        if total <= 0:
            return TreatedComposition(
                status="INELIGIBLE",
                reason="ZERO_SUM_NO_COMPOSITION_FOR_MULTIPLICATIVE_REPLACEMENT",
                composition=None,
                input_mass=total,
                zero_count=zero_count,
                replacement_mass_per_zero=None,
            )
        closed = vector / total
        replacement = treatment.delta / (total + dimension * treatment.delta)
        positive_scale = 1.0 - zero_count * replacement
        if positive_scale <= 0:
            raise CompositionalContractError(
                "multiplicative replacement exhausted the positive-part mass."
            )
        composition = np.empty_like(closed)
        zero_mask = vector == 0.0
        composition[zero_mask] = replacement
        composition[~zero_mask] = closed[~zero_mask] * positive_scale
    else:
        replacement = None
        if total <= 0:
            return TreatedComposition(
                status="INELIGIBLE",
                reason="ZERO_SUM_COMPOSITION_NO_REPLACEMENT",
                composition=None,
                input_mass=total,
                zero_count=zero_count,
                replacement_mass_per_zero=None,
            )
        composition = vector / total

    if not np.all(np.isfinite(composition)):
        raise CompositionalContractError("zero treatment produced a nonfinite value.")
    if np.any(composition < 0):
        raise CompositionalContractError("zero treatment produced a negative value.")
    closure_error = abs(float(np.sum(composition, dtype=np.float64)) - 1.0)
    if closure_error > 1e-12:
        raise CompositionalContractError(
            f"zero treatment failed closure by {closure_error:.17g}."
        )
    return TreatedComposition(
        status="ELIGIBLE",
        reason=None,
        composition=composition,
        input_mass=total,
        zero_count=zero_count,
        replacement_mass_per_zero=replacement,
    )


def helmert_simplex_basis(dimension: int) -> NDArray[np.float64]:
    """Return the frozen sequential Helmert basis as D by (D-1)."""

    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 2:
        raise CompositionalContractError("dimension must be an integer >= 2.")
    basis = np.zeros((dimension, dimension - 1), dtype=np.float64)
    for column in range(1, dimension):
        scale = np.sqrt(column * (column + 1.0))
        basis[:column, column - 1] = 1.0 / scale
        basis[column, column - 1] = -column / scale
    validate_simplex_basis(basis, dimension=dimension)
    return basis


def validate_simplex_basis(
    basis: ArrayLike,
    *,
    dimension: int,
    tolerance: float = 1e-12,
) -> NDArray[np.float64]:
    """Require an orthonormal basis orthogonal to the all-ones vector."""

    matrix = np.asarray(basis, dtype=np.float64)
    if matrix.shape != (dimension, dimension - 1):
        raise CompositionalContractError(
            f"simplex basis shape must be {(dimension, dimension - 1)}."
        )
    if not np.all(np.isfinite(matrix)):
        raise CompositionalContractError("simplex basis must be finite.")
    gram_error = float(
        np.max(np.abs(matrix.T @ matrix - np.eye(dimension - 1, dtype=np.float64)))
    )
    ones_error = float(np.max(np.abs(matrix.T @ np.ones(dimension))))
    if gram_error > tolerance or ones_error > tolerance:
        raise CompositionalContractError(
            "simplex basis is not orthonormal/simplex-orthogonal: "
            f"gram_error={gram_error:.17g}, ones_error={ones_error:.17g}."
        )
    return matrix.copy()


def principal_logratio_basis(
    compositions: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit the frozen covariance-eigenvector PLR basis within one named scope."""

    matrix = np.asarray(compositions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise CompositionalContractError(
            "principal-log-ratio fitting requires at least two rows and two components."
        )
    if not np.all(np.isfinite(matrix)) or np.any(matrix <= 0):
        raise CompositionalContractError(
            "principal-log-ratio fitting requires finite strictly positive compositions."
        )
    closed = matrix / np.sum(matrix, axis=1, keepdims=True)
    dimension = closed.shape[1]
    helmert = helmert_simplex_basis(dimension)
    ilr = np.log(closed) @ helmert
    centered = ilr - np.mean(ilr, axis=0, keepdims=True)
    covariance = centered.T @ centered / (closed.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    basis = helmert @ eigenvectors
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1.0
            eigenvectors[:, column] *= -1.0
    validate_simplex_basis(basis, dimension=dimension)
    return basis, eigenvalues


def _stable_inverse_logratio(
    log_composition: NDArray[np.float64],
) -> NDArray[np.float64]:
    shifted = log_composition - np.max(log_composition)
    return _close(np.exp(shifted))


def transform_coordinates(
    composition: ArrayLike,
    specification: CoordinateSpecification,
    *,
    simplex_basis: ArrayLike | None,
) -> NDArray[np.float64]:
    """Map one treated composition into one explicit coordinate family."""

    closed = _close(composition)
    if closed.size != specification.dimension:
        raise CompositionalContractError(
            "composition/specification dimension mismatch."
        )
    family = specification.family
    if family in ("full_clr", "dropped_clr", "ilr_helmert", "principal_log_ratio"):
        if np.any(closed <= 0):
            raise CompositionalContractError(
                "ZERO_COMPONENT_LOG_RATIO_WITHOUT_REPLACEMENT"
            )
        logs = np.log(closed)
        if family == "full_clr":
            output = logs - np.mean(logs)
        elif family == "dropped_clr":
            full = logs - np.mean(logs)
            output = np.delete(full, specification.dropped_component_zero_based)
        else:
            if simplex_basis is None:
                raise CompositionalContractError(
                    f"{family} requires an explicit simplex basis."
                )
            basis = validate_simplex_basis(
                simplex_basis, dimension=specification.dimension
            )
            output = basis.T @ logs
    elif family == "raw_proportions":
        if simplex_basis is not None:
            raise CompositionalContractError("raw_proportions forbids a simplex basis.")
        output = closed.copy()
    elif family == "hellinger":
        if simplex_basis is not None:
            raise CompositionalContractError("hellinger forbids a simplex basis.")
        output = np.sqrt(closed)
    else:  # pragma: no cover - dataclass validation makes this unreachable
        raise CompositionalContractError(f"Unsupported family {family!r}.")
    if not np.all(np.isfinite(output)):
        raise CompositionalContractError(
            "coordinate transform produced nonfinite output."
        )
    return np.asarray(output, dtype=np.float64)


def inverse_coordinates(
    coordinates: ArrayLike,
    specification: CoordinateSpecification,
    *,
    simplex_basis: ArrayLike | None,
) -> NDArray[np.float64]:
    """Invert one supported coordinate mapping back to a closed composition."""

    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise CompositionalContractError("coordinates must be a finite vector.")
    family = specification.family
    dimension = specification.dimension
    expected = (
        dimension
        if family in ("full_clr", "raw_proportions", "hellinger")
        else dimension - 1
    )
    if values.size != expected:
        raise CompositionalContractError(
            f"coordinate length must be {expected} for {family}."
        )
    if family == "full_clr":
        return _stable_inverse_logratio(values)
    if family == "dropped_clr":
        assert specification.dropped_component_zero_based is not None
        full = np.insert(
            values,
            specification.dropped_component_zero_based,
            -float(np.sum(values, dtype=np.float64)),
        )
        return _stable_inverse_logratio(full)
    if family in ("ilr_helmert", "principal_log_ratio"):
        if simplex_basis is None:
            raise CompositionalContractError(f"{family} requires an explicit basis.")
        basis = validate_simplex_basis(simplex_basis, dimension=dimension)
        return _stable_inverse_logratio(basis @ values)
    if family == "raw_proportions":
        return _close(values)
    if family == "hellinger":
        if np.any(values < 0):
            raise CompositionalContractError(
                "Hellinger coordinates must be nonnegative."
            )
        return _close(values * values)
    raise CompositionalContractError(f"Unsupported family {family!r}.")


def evaluate_transform(
    composition: ArrayLike,
    specification: CoordinateSpecification,
    *,
    simplex_basis: ArrayLike | None,
) -> TransformResult:
    """Transform and invert one composition with explicit domain status."""

    closed = _close(composition)
    try:
        coordinates = transform_coordinates(
            closed, specification, simplex_basis=simplex_basis
        )
    except CompositionalContractError as exc:
        if str(exc) == "ZERO_COMPONENT_LOG_RATIO_WITHOUT_REPLACEMENT":
            return TransformResult(
                status="INELIGIBLE",
                reason=str(exc),
                coordinates=None,
                reconstructed_composition=None,
                maximum_absolute_inverse_error=None,
                maximum_relative_inverse_error=None,
                closure_error=None,
            )
        raise
    reconstructed = inverse_coordinates(
        coordinates, specification, simplex_basis=simplex_basis
    )
    absolute = np.abs(reconstructed - closed)
    relative = absolute / np.maximum(np.abs(closed), np.finfo(np.float64).tiny)
    return TransformResult(
        status="ELIGIBLE",
        reason=None,
        coordinates=coordinates,
        reconstructed_composition=reconstructed,
        maximum_absolute_inverse_error=float(np.max(absolute)),
        maximum_relative_inverse_error=float(np.max(relative)),
        closure_error=abs(float(np.sum(reconstructed)) - 1.0),
    )


def pairwise_euclidean(values: ArrayLike) -> NDArray[np.float64]:
    """Return a finite symmetric Euclidean distance matrix."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise CompositionalContractError("pairwise distances require a finite matrix.")
    differences = matrix[:, None, :] - matrix[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    return distances


def covariance_diagnostics(
    coordinates: ArrayLike,
    *,
    condition_threshold: float,
) -> dict[str, Any]:
    """Describe covariance rank/conditioning without repairing singularity."""

    matrix = np.asarray(coordinates, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise CompositionalContractError(
            "covariance diagnostics require a finite observations-by-features matrix."
        )
    observations, features = matrix.shape
    if observations < 2:
        return {
            "status": "INSUFFICIENT_ELIGIBLE_ROWS",
            "covariance": None,
            "rank": 0,
            "rankTolerance": None,
            "conditionNumberRaw": None,
            "effectiveConditionNumber": None,
        }
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    covariance = centered.T @ centered / (observations - 1)
    singular_values = np.linalg.svd(covariance, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = max(observations, features) * np.finfo(np.float64).eps * largest
    positive = singular_values[singular_values > tolerance]
    rank = int(positive.size)
    if singular_values.size == 0 or largest == 0.0:
        raw_condition = float("inf")
        effective = float("inf")
    else:
        smallest = float(singular_values[-1])
        raw_condition = float("inf") if smallest <= tolerance else largest / smallest
        effective = (
            float("inf") if positive.size == 0 else largest / float(positive[-1])
        )
    if rank < features:
        status = "SAMPLE_OR_STRUCTURAL_RANK_DEFICIENT"
    elif raw_condition > condition_threshold:
        status = "ILL_CONDITIONED"
    else:
        status = "READY"
    return {
        "status": status,
        "covariance": covariance,
        "rank": rank,
        "rankTolerance": tolerance,
        "conditionNumberRaw": raw_condition,
        "effectiveConditionNumber": effective,
    }
