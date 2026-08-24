from __future__ import annotations

import numpy as np

from e01_attractor_onset_early_warning.core import build_landmark_target
from e01_latent_timebase.core import derive_seed, generate_beta, initialize_distinct_state
from e01_onset_discovery.outcome_blind_representation import (
    extract_outcome_blind_representation,
)

ROOT = "f3b0fd551b8f182388cad84365b62a2f2f51e82aa11a0c6b6e22a088dbe90544"
PHASE = "s19_l23_powered_frozen_prefix_screen"


def test_new_input_identity_replay() -> None:
    beta_seed = derive_seed(ROOT, PHASE, "catalytic_matrix", 17)
    init_seed = derive_seed(ROOT, PHASE, "initial_state", 17)
    assert np.array_equal(generate_beta(beta_seed), generate_beta(beta_seed))
    assert np.array_equal(
        initialize_distinct_state(init_seed), initialize_distinct_state(init_seed)
    )


def test_landmark_task_contract() -> None:
    labels = np.zeros(250, dtype=bool)
    labels[100:] = True
    target = build_landmark_target(labels)
    assert target["atRiskAtLandmark"]
    assert target["eventWithinHorizon"]
    assert target["firstOnsetIndex0"] == 100


def test_fixed_l22_representation_replay() -> None:
    rng = np.random.default_rng(23)
    states = rng.poisson(2.0, size=(64, 100)).astype(np.int64)
    states[:, 0] += 1
    assert extract_outcome_blind_representation(states) == extract_outcome_blind_representation(
        states.copy()
    )
