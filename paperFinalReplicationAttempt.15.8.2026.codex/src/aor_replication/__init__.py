"""Clean-room replication package for the arrival-of-replicators preprint."""

from .config import (
    CausalConfig,
    ExperimentConfig,
    GardConfig,
    InterventionConfig,
    ReplicatorConfig,
)
from .gard import RunTrace, simulate_gard
from .information import CausalTrajectory, fit_causal_trajectory
from .replicators import ReplicatorResult, detect_replicators

__all__ = [
    "CausalConfig",
    "CausalTrajectory",
    "ExperimentConfig",
    "GardConfig",
    "InterventionConfig",
    "ReplicatorConfig",
    "ReplicatorResult",
    "RunTrace",
    "detect_replicators",
    "fit_causal_trajectory",
    "simulate_gard",
]

__version__ = "0.1.0"
