from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from grn_f12_realistic.continuous import (
    acquire_continuous_history, continuous_drift_jax, continuous_drift_numpy,
    simulate_continuous_futures,
)
from grn_f12_realistic.molecular import acquire_molecular_history, molecular_rates_numpy, simulate_molecular_futures
from grn_f12_realistic.network import sample_network
from grn_f12_realistic.rng import jax_key


def test_continuous_float64_reference_matches_float32_jax(tiny_protocol):
    network = sample_network(tiny_protocol, "continuous", "development", 0)
    x = network.initial_x.astype(np.float64)
    reference = continuous_drift_numpy(x, network.W.astype(np.float64), network.bias, network.cue_a, 2.5)
    accelerated = np.asarray(continuous_drift_jax(
        jnp.asarray(x, dtype=jnp.float32), jnp.asarray(network.W), jnp.asarray(network.bias),
        jnp.asarray(network.cue_a), 2.5,
    ))
    assert np.allclose(reference, accelerated, rtol=2e-6, atol=2e-6)


def test_continuous_history_cue_and_replay_identity(tiny_protocol):
    network = sample_network(tiny_protocol, "continuous", "development", 1)
    baseline, states = acquire_continuous_history(network, tiny_protocol)
    assert states.shape == (2, 2, 6)
    assert not np.array_equal(states[0, 0], states[1, 0])
    key = jax_key(tiny_protocol["master_seed_label"], "test", "continuous")
    scan = simulate_continuous_futures(network, states[0, 0], tiny_protocol, 4, key, horizon=4, executor="scan")
    loop = simulate_continuous_futures(network, states[0, 0], tiny_protocol, 4, key, horizon=4, executor="loop")
    for left, right in zip(scan, loop):
        assert np.allclose(left, right, rtol=1e-6, atol=1e-6)
    assert np.all(scan[1] >= 0.0)
    assert np.all(scan[1] <= 1.5)


def test_molecular_counts_rates_and_replay_identity(tiny_protocol):
    network = sample_network(tiny_protocol, "molecular", "development", 2)
    baseline, mrna_states, protein_states = acquire_molecular_history(network, tiny_protocol)
    rates = molecular_rates_numpy(
        baseline[0], baseline[1], network.W, network.bias, np.zeros(4), 1.0,
        tiny_protocol["tiers"]["molecular"],
    )
    assert np.all(rates[0] > 0) and np.all(rates[1] >= 0)
    key = jax_key(tiny_protocol["master_seed_label"], "test", "molecular")
    scan = simulate_molecular_futures(
        network, mrna_states[0, 0], protein_states[0, 0], tiny_protocol, 4, key,
        horizon=4, executor="scan",
    )
    loop = simulate_molecular_futures(
        network, mrna_states[0, 0], protein_states[0, 0], tiny_protocol, 4, key,
        horizon=4, executor="loop",
    )
    for left, right in zip(scan, loop):
        assert np.array_equal(left, right)
    assert np.issubdtype(scan[1].dtype, np.integer)
    assert np.issubdtype(scan[2].dtype, np.integer)
    assert np.all(scan[1] >= 0) and np.all(scan[2] >= 0)


def test_inheritance_erasure_changes_continuous_path(tiny_protocol):
    network = sample_network(tiny_protocol, "continuous", "development", 4)
    baseline, states = acquire_continuous_history(network, tiny_protocol)
    key = jax_key(tiny_protocol["master_seed_label"], "test", "erase")
    ordinary = simulate_continuous_futures(network, states[0, 0], tiny_protocol, 4, key, horizon=4)
    erased = simulate_continuous_futures(
        network, states[0, 0], tiny_protocol, 4, key, horizon=4, erase_state=baseline
    )
    assert not np.array_equal(ordinary[1], erased[1])
