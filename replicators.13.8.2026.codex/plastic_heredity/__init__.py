"""Independent replication of the plastic-heredity discovery contract."""

from .config import CANDIDATES, ExperimentConfig, GardConfig, SimulationContract
from .processes import JOINT_BREAK_RUN3, evaluate_process
from .simulator import Snapshot, cosine_similarity

__all__ = [
    "CANDIDATES",
    "ExperimentConfig",
    "GardConfig",
    "JOINT_BREAK_RUN3",
    "SimulationContract",
    "Snapshot",
    "cosine_similarity",
    "evaluate_process",
]

