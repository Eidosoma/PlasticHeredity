"""Versioned S11 time-localized Phi-r reconstruction utilities.

These are validation branches, not recovered paper-author defaults.
"""

from .estimator import (
    CALIBRATION_ID,
    COVARIANCE_ID,
    ESTIMATOR_ID,
    SmallWindowResult,
    calibrated_means,
    run_small_window_phiid,
)
from .partition import (
    AFFINITY_ID,
    GROUP_MEAN_ID,
    PC1_ID,
    SEARCH_ID,
    StablePartitionResult,
    evaluate_candidate_grid,
    map_partition,
    partition_ari,
    partition_objective,
    select_partition_candidate,
    stable_partition_candidates,
)
from .temporal import (
    TemporalIndexError,
    WindowIndex,
    fixed_window_index,
    sliding_endpoints,
    whole_trajectory_index,
)

__all__ = [
    "AFFINITY_ID",
    "CALIBRATION_ID",
    "COVARIANCE_ID",
    "ESTIMATOR_ID",
    "GROUP_MEAN_ID",
    "PC1_ID",
    "SEARCH_ID",
    "SmallWindowResult",
    "StablePartitionResult",
    "TemporalIndexError",
    "WindowIndex",
    "calibrated_means",
    "evaluate_candidate_grid",
    "fixed_window_index",
    "map_partition",
    "partition_ari",
    "partition_objective",
    "run_small_window_phiid",
    "select_partition_candidate",
    "sliding_endpoints",
    "stable_partition_candidates",
    "whole_trajectory_index",
]

__version__ = "1.0.0"
