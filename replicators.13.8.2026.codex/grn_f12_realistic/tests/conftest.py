from __future__ import annotations

import copy

import pytest

from grn_f12_realistic.config import load_protocol, protocol_for_profile


@pytest.fixture
def tiny_protocol():
    protocol = copy.deepcopy(protocol_for_profile(load_protocol(), "smoke"))
    protocol["landmarks"] = [0, 1]
    protocol["history"].update(burnin_generations=1, cue_generations=1, cue_genes=2)
    protocol["endpoint"].update(horizon=4, secondary_horizon=4)
    protocol["tiers"]["continuous"].update(
        genes=6, edge_probability=0.4, substeps=2, dt=0.5,
        futures=4, calibration_futures=4,
    )
    protocol["tiers"]["molecular"].update(
        genes=4, edge_probability=0.6, substeps=2, dt=0.5,
        futures=4, calibration_futures=4,
    )
    protocol["predictor"].update(width=8, message_layers=2, max_epochs=3, patience=2, batch_states=8)
    return protocol

