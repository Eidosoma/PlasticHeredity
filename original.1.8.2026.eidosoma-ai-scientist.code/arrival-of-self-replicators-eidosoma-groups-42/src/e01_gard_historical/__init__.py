"""Source-traceable port of the pinned public historical GARD v10 behavior.

This package is a forensic compatibility layer.  It is not the unavailable
Pigozzi--Levin author implementation and it does not implement the paper's
separate vector-Poisson or a modern Gillespie branch.
"""

from .engine import (
    EventRecord,
    FissionResult,
    GenerationResult,
    GrowthResult,
    HistoricalReferenceError,
    HistoricalSourceDomainError,
    LineageResult,
    NumpyUniformSource,
    Propensities,
    RandomTapeExhausted,
    UniformTape,
    advance_one_generation,
    catalytic_matrix_from_numpy_rng_explicit,
    catalytic_matrix_from_standard_normals,
    compute_propensities,
    grow_to_split_size,
    historical_initial_state_with_replacement,
    historical_single_event,
    historical_weighted_index,
    simulate_lineage,
    split_fixed_size_without_replacement,
)
from .nondrift import (
    NonDriftResult,
    historical_h,
    historical_nondrift_technique1,
    historical_nondrift_technique2,
)

__all__ = [
    "EventRecord",
    "FissionResult",
    "GenerationResult",
    "GrowthResult",
    "HistoricalReferenceError",
    "HistoricalSourceDomainError",
    "LineageResult",
    "NonDriftResult",
    "NumpyUniformSource",
    "Propensities",
    "RandomTapeExhausted",
    "UniformTape",
    "advance_one_generation",
    "catalytic_matrix_from_numpy_rng_explicit",
    "catalytic_matrix_from_standard_normals",
    "compute_propensities",
    "grow_to_split_size",
    "historical_h",
    "historical_initial_state_with_replacement",
    "historical_nondrift_technique1",
    "historical_nondrift_technique2",
    "historical_single_event",
    "historical_weighted_index",
    "simulate_lineage",
    "split_fixed_size_without_replacement",
]

__version__ = "1.0.0"
