from dataclasses import replace

import numpy as np

from aor_replication.config import CausalConfig, GardConfig
from aor_replication.gard import simulate_gard
from aor_replication.information import fit_causal_trajectory
from aor_replication.interventions import (
    OnlinePhiDirectedIntervention,
    PhiDirectedIntervention,
)


def test_extremal_interventions_are_feasible_and_ordered() -> None:
    config = GardConfig(
        n_types=10,
        initial_size=5,
        max_size=10,
        generations=12,
        max_steps_per_generation=100,
        beta_log_sigma=1.5,
        tau=2.0,
    )
    control = simulate_gard(config, 22)
    reference = fit_causal_trajectory(
        control.counts,
        replace(CausalConfig(), pseudocount=0.5),
    )
    parent = control.counts[-2]
    daughter = control.counts[-1]
    maximum = PhiDirectedIntervention(reference, "max", config.max_size)(
        parent, daughter, control.counts, 0
    )
    minimum = PhiDirectedIntervention(reference, "min", config.max_size)(
        parent, daughter, control.counts, 0
    )
    assert maximum.delta in {-1, 1}
    assert minimum.delta in {-1, 1}
    assert maximum.score >= minimum.score
    if maximum.delta == -1:
        assert daughter[maximum.species] > 0
    if minimum.delta == -1:
        assert daughter[minimum.species] > 0


def test_online_intervention_uses_available_history() -> None:
    config = GardConfig(
        n_types=10,
        initial_size=5,
        max_size=10,
        generations=4,
        max_steps_per_generation=100,
        beta_log_sigma=1.5,
        tau=2.0,
    )
    control = simulate_gard(config, 31)
    history = control.counts[: max(8, len(control.counts) // 2)]
    parent = history[-1]
    daughter = np.maximum(parent // 2, 0)
    if daughter.sum() == 0:
        daughter[np.argmax(parent)] = 1
    decision = OnlinePhiDirectedIntervention(
        CausalConfig(), "max", config.max_size
    )(parent, daughter, history, 0)
    assert decision.delta in {-1, 1}
    assert np.isfinite(decision.score)
