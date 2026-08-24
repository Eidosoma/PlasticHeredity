"""Separately versioned S11R fixed-window validation utilities.

S11R is a bounded repair experiment.  It neither changes the immutable S11
implementation nor identifies the unavailable paper-author implementation.
"""

from .estimator import (
    CALIBRATION_ID,
    ESTIMATOR_ID,
    SmallWindowRepairResult,
    calibrate_means,
    run_wishart_local_phiid,
)
from .partition import (
    AFFINITY_ID,
    SEARCH_ID,
    ThresholdPartitionResult,
    partition_ari,
    threshold_component_partition,
)
from .synthetic import (
    Fixture,
    ccs_population_oracle,
    directional_covariance,
    directional_var,
    highdim_independent_null,
    independent_white,
    mmi_truth,
    noisy_redundant_ar,
    planted_two_block_ar,
    repair_rng,
)

__all__ = [
    "AFFINITY_ID",
    "CALIBRATION_ID",
    "ESTIMATOR_ID",
    "SEARCH_ID",
    "Fixture",
    "SmallWindowRepairResult",
    "ThresholdPartitionResult",
    "calibrate_means",
    "ccs_population_oracle",
    "directional_covariance",
    "directional_var",
    "highdim_independent_null",
    "independent_white",
    "mmi_truth",
    "noisy_redundant_ar",
    "partition_ari",
    "planted_two_block_ar",
    "repair_rng",
    "run_wishart_local_phiid",
    "threshold_component_partition",
]

__version__ = "1.0.0"
