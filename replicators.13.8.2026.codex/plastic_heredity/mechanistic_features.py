"""Disjoint feature blocks for the prospective mechanistic ablation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import ExperimentConfig
from .experiment import StateCase
from .features import (
    HISTORY_FEATURE_NAMES,
    STATE_GRAPH_FEATURE_NAMES,
    beta_only_features,
    history_features,
    state_graph_features,
)

FloatMatrix = NDArray[np.float64]

# The seventh legacy history variable is an exact duplicate of the fifth under
# the registered definition, so H8 retains the first occurrence only.
H8_INDICES = (0, 1, 2, 3, 4, 5, 7, 8)
H8_FEATURE_NAMES = tuple(HISTORY_FEATURE_NAMES[index] for index in H8_INDICES)
H10_FEATURE_NAMES = H8_FEATURE_NAMES + (
    "normalized_previous_growth_steps",
    "normalized_cumulative_growth_steps",
)
DUPLICATE_CONTROL_FEATURE_NAMES = (
    "duplicate_normalized_generation",
    "duplicate_normalized_current_mass",
)

STATE_ONLY_PROFILES = frozenset(("composition_fraction", "present"))
INTERACTION_PROFILES = frozenset(
    (
        "log_in_catalysis",
        "log_out_catalysis",
        "log_active_in_catalysis",
        "log_active_out_catalysis",
    )
)

STATE_ONLY_INDICES = tuple(
    index
    for index, name in enumerate(STATE_GRAPH_FEATURE_NAMES)
    if name.split("__", 1)[0] in STATE_ONLY_PROFILES
)
INTERACTION_INDICES = tuple(
    index
    for index, name in enumerate(STATE_GRAPH_FEATURE_NAMES)
    if name.split("__", 1)[0] in INTERACTION_PROFILES
)
STATE_ONLY_FEATURE_NAMES = tuple(
    STATE_GRAPH_FEATURE_NAMES[index] for index in STATE_ONLY_INDICES
)
INTERACTION_FEATURE_NAMES = tuple(
    STATE_GRAPH_FEATURE_NAMES[index] for index in INTERACTION_INDICES
)
BETA_ONLY_FEATURE_NAMES = tuple(
    f"beta_only__{name}" for name in STATE_GRAPH_FEATURE_NAMES
)


@dataclass(frozen=True)
class MechanisticRawFeatures:
    h8: FloatMatrix
    h10: FloatMatrix
    state: FloatMatrix
    beta: FloatMatrix
    interaction: FloatMatrix
    duplicate: FloatMatrix

    def selected(self, indices: NDArray[np.bool_]) -> "MechanisticRawFeatures":
        return MechanisticRawFeatures(
            h8=self.h8[indices],
            h10=self.h10[indices],
            state=self.state[indices],
            beta=self.beta[indices],
            interaction=self.interaction[indices],
            duplicate=self.duplicate[indices],
        )


FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "h8": H8_FEATURE_NAMES,
    "h10": H10_FEATURE_NAMES,
    "state": STATE_ONLY_FEATURE_NAMES,
    "beta": BETA_ONLY_FEATURE_NAMES,
    "interaction": INTERACTION_FEATURE_NAMES,
    "duplicate": DUPLICATE_CONTROL_FEATURE_NAMES,
}


def extract_mechanistic_features(
    cases: list[StateCase], experiment: ExperimentConfig
) -> MechanisticRawFeatures:
    legacy_history = np.vstack(
        [history_features(case.snapshot, experiment.gard) for case in cases]
    )
    h8 = legacy_history[:, H8_INDICES]
    clocks = np.asarray(
        [
            (
                case.snapshot.previous_growth_steps
                / max(experiment.gard.max_growth_steps, 1),
                case.snapshot.cumulative_growth_steps
                / max(
                    experiment.gard.generations
                    * experiment.gard.max_growth_steps,
                    1,
                ),
            )
            for case in cases
        ],
        dtype=np.float64,
    )
    h10 = np.column_stack((h8, clocks))
    full_state = np.vstack(
        [
            state_graph_features(case.snapshot.composition, case.beta, experiment.gard)
            for case in cases
        ]
    )
    beta = np.vstack(
        [beta_only_features(case.beta, experiment.gard) for case in cases]
    )
    duplicate = h8[:, (0, 1)].copy()
    return MechanisticRawFeatures(
        h8=h8,
        h10=h10,
        state=full_state[:, STATE_ONLY_INDICES],
        beta=beta,
        interaction=full_state[:, INTERACTION_INDICES],
        duplicate=duplicate,
    )
