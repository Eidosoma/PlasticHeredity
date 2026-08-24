from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from grn_f12_realistic.baselines import fit_hurdle_ridge
from grn_f12_realistic.gnn import forward, init_params
from grn_f12_realistic.predictor import balanced_network_folds


def test_hurdle_ridge_probabilities_are_product_of_heads():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 4))
    total = np.full(30, 40)
    breaks = np.clip((20 + 8 * x[:, 0]).astype(int), 1, 39)
    events = np.clip((breaks * (0.4 + 0.2 * (x[:, 1] > 0))).astype(int), 0, breaks)
    model = fit_hurdle_ridge(x, events, breaks, total, 1.0)
    break_probability, recovery_probability, event_probability = model.predict(x)
    assert np.all((break_probability > 0) & (break_probability < 1))
    assert np.allclose(event_probability, break_probability * recovery_probability)


def test_graph_model_is_node_permutation_invariant_but_not_state_shuffle():
    key = jax.random.PRNGKey(8)
    params = init_params(key, node_features=8, history_features=10, width=12, layers=3)
    rng = np.random.default_rng(2)
    nodes = rng.normal(size=(3, 7, 8)).astype(np.float32)
    weights = rng.normal(size=(3, 7, 7)).astype(np.float32)
    history = rng.normal(size=(3, 10)).astype(np.float32)
    permutation = np.array([3, 0, 6, 1, 5, 2, 4])
    original = forward(params, jnp.asarray(nodes), jnp.asarray(weights), jnp.asarray(history))
    permuted = forward(
        params, jnp.asarray(nodes[:, permutation]),
        jnp.asarray(weights[:, permutation][:, :, permutation]), jnp.asarray(history),
    )
    shuffled_state = forward(params, jnp.asarray(nodes[:, permutation]), jnp.asarray(weights), jnp.asarray(history))
    assert np.allclose(original[0], permuted[0], atol=2e-6)
    assert np.allclose(original[1], permuted[1], atol=2e-6)
    assert not np.allclose(original[0], shuffled_state[0])


def test_whole_network_folds_are_balanced_and_deterministic():
    indices = np.arange(23)
    first = balanced_network_folds("master", "continuous", indices, 5)
    second = balanced_network_folds("master", "continuous", indices, 5)
    counts = np.bincount(list(first.values()), minlength=5)
    assert first == second
    assert counts.max() - counts.min() <= 1
    assert set(first) == set(indices)

