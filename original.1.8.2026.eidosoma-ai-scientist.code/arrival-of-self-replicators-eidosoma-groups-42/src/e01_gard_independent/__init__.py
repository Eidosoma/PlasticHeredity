"""Independent, branch-explicit Python GARD engine for E01 S05.

This package is neither the S04 historical compatibility engine nor the
unavailable paper-author implementation.  Every runnable model is supplied as
a complete validation or analysis specification.
"""

from .engine import (
    BatchLossError,
    EmptyDaughterError,
    GrowthLimitError,
    IndependentGardError,
    ZeroPropensityError,
    advance_generation,
    calculate_propensities,
    fission,
    generate_catalytic_matrix,
    grow,
    initialize_state,
    integer_state,
    sample_update,
    simulate_lineage,
)
from .records import (
    EventLog,
    FissionLog,
    GenerationResult,
    GrowthResult,
    LineageResult,
    PropensityArrays,
)
from .rng import RNGInput, RNGStreams, generator_state_sha256
from .specification import (
    CatalyticMatrixBranch,
    ClockSemantics,
    DaughterSelection,
    FissionSemantics,
    GardSpecification,
    GrowthBoundary,
    InitialStateSemantics,
    LossNonnegativity,
    MaxStepsSemantics,
    PostFissionSemantics,
    ProfileRole,
    PropensityEquationBranch,
    ReservoirSemantics,
    SpecificationError,
    UpdateKernel,
    ZeroPropensitySemantics,
    specification_from_mapping,
)

__all__ = [
    "BatchLossError",
    "CatalyticMatrixBranch",
    "ClockSemantics",
    "DaughterSelection",
    "EmptyDaughterError",
    "EventLog",
    "FissionLog",
    "FissionSemantics",
    "GardSpecification",
    "GenerationResult",
    "GrowthBoundary",
    "GrowthLimitError",
    "GrowthResult",
    "IndependentGardError",
    "InitialStateSemantics",
    "LineageResult",
    "LossNonnegativity",
    "MaxStepsSemantics",
    "PostFissionSemantics",
    "ProfileRole",
    "PropensityArrays",
    "PropensityEquationBranch",
    "RNGInput",
    "RNGStreams",
    "ReservoirSemantics",
    "SpecificationError",
    "UpdateKernel",
    "ZeroPropensityError",
    "ZeroPropensitySemantics",
    "advance_generation",
    "calculate_propensities",
    "fission",
    "generate_catalytic_matrix",
    "generator_state_sha256",
    "grow",
    "initialize_state",
    "integer_state",
    "sample_update",
    "simulate_lineage",
    "specification_from_mapping",
]

__version__ = "1.0.0"
