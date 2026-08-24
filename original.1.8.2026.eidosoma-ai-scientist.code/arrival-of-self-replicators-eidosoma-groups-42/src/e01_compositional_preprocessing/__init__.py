"""Explicit compositional preprocessing contracts for E01 S09."""

from .transforms import (
    CompositionalContractError,
    CoordinateSpecification,
    TransformResult,
    ZeroTreatment,
    apply_zero_treatment,
    covariance_diagnostics,
    evaluate_transform,
    helmert_simplex_basis,
    inverse_coordinates,
    pairwise_euclidean,
    principal_logratio_basis,
    transform_coordinates,
    validate_simplex_basis,
)

__all__ = [
    "CompositionalContractError",
    "CoordinateSpecification",
    "TransformResult",
    "ZeroTreatment",
    "apply_zero_treatment",
    "covariance_diagnostics",
    "evaluate_transform",
    "helmert_simplex_basis",
    "inverse_coordinates",
    "pairwise_euclidean",
    "principal_logratio_basis",
    "transform_coordinates",
    "validate_simplex_basis",
]

__version__ = "1.0.0"
