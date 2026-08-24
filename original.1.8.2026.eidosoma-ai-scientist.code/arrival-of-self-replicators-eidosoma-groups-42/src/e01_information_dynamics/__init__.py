"""Explicit S10 information-dynamics validation utilities.

This package validates named reconstruction branches.  It does not select an
author PhiID mapping, redundancy function, estimator, or minimum-information
bipartition rule.
"""

from .backends import (
    DecompositionResult,
    backend_identity,
    compare_decompositions,
    run_omegaid,
    run_phyid,
)
from .validation import (
    ATOM_IDS,
    I_KEYS,
    aggregate_means,
    all_bipartitions,
    coupled_ar_covariance,
    discrete_exact_oracle,
    exact_redundant_pmf,
    exact_xor_pmf,
    gaussian_mmi_oracle,
    gaussian_mutual_information,
    gaussian_partition_objective,
    noisy_redundant_covariance,
    strict_sample_gate,
)

__all__ = [
    "ATOM_IDS",
    "I_KEYS",
    "DecompositionResult",
    "aggregate_means",
    "all_bipartitions",
    "backend_identity",
    "compare_decompositions",
    "coupled_ar_covariance",
    "discrete_exact_oracle",
    "exact_redundant_pmf",
    "exact_xor_pmf",
    "gaussian_mmi_oracle",
    "gaussian_mutual_information",
    "gaussian_partition_objective",
    "noisy_redundant_covariance",
    "run_omegaid",
    "run_phyid",
    "strict_sample_gate",
]

__version__ = "1.0.0"
