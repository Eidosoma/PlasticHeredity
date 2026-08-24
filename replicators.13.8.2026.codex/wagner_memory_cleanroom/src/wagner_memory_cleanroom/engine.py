from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np


def signed_update(field: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Wagner sign update; exact ties retain the previous bit."""
    return np.where(field > 0.0, 1, np.where(field < 0.0, -1, previous)).astype(np.int8)


def sequential_sweep_numpy(
    weights: np.ndarray,
    state: np.ndarray,
    external_field: np.ndarray | None = None,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    result = np.asarray(state, dtype=np.int8).copy()
    field = np.zeros_like(result, dtype=np.float64) if external_field is None else np.asarray(external_field)
    perturbation = np.zeros_like(result, dtype=np.float64) if noise is None else np.asarray(noise)
    for gene in range(result.shape[-1]):
        regulatory = result @ weights[gene]
        value = regulatory + field[..., gene] + perturbation[..., gene]
        result[..., gene] = signed_update(value, result[..., gene])
    return result


def rollout_numpy(
    weights: np.ndarray,
    initial: np.ndarray,
    external_field: np.ndarray,
    *,
    sweeps: int,
    theta: float,
    flip_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    state = np.asarray(initial, dtype=np.int8).copy()
    history = []
    for _ in range(sweeps):
        noise = rng.normal(0.0, np.sqrt(theta), size=state.shape) if theta > 0 else None
        state = sequential_sweep_numpy(weights, state, external_field, noise)
        if flip_probability > 0:
            flips = rng.random(state.shape) < flip_probability
            state = np.where(flips, -state, state).astype(np.int8)
        history.append(state.copy())
    return np.stack(history)


def _jax_modules() -> tuple[Any, Any]:
    import jax
    import jax.numpy as jnp

    return jax, jnp


@partial(__import__("functools").lru_cache(maxsize=None))
def _compiled_rollout(sweeps: int):
    jax, jnp = _jax_modules()

    @jax.jit
    def run(weights, initial, external_field, theta, flip_probability, key_data):
        key = jax.random.wrap_key_data(key_data)
        noise_key, flip_key = jax.random.split(key)
        noise = jax.random.normal(
            noise_key,
            shape=(sweeps, initial.shape[0], initial.shape[1]),
            dtype=jnp.float32,
        ) * jnp.sqrt(theta)
        flips = jax.random.uniform(
            flip_key,
            shape=(sweeps, initial.shape[0], initial.shape[1]),
            dtype=jnp.float32,
        ) < flip_probability

        def one_sweep(state, sweep_index):
            sweep_noise = noise[sweep_index]

            def one_gene(gene, current):
                regulatory = current @ weights[gene]
                value = regulatory + external_field[:, gene] + sweep_noise[:, gene]
                updated = jnp.where(value > 0, 1, jnp.where(value < 0, -1, current[:, gene]))
                return current.at[:, gene].set(updated.astype(jnp.int8))

            state = jax.lax.fori_loop(0, initial.shape[1], one_gene, state)
            state = jnp.where(flips[sweep_index], -state, state).astype(jnp.int8)
            return state, state

        _, history = jax.lax.scan(one_sweep, initial, jnp.arange(sweeps))
        return history

    return run


def rollout_jax(
    weights: np.ndarray,
    initial: np.ndarray,
    external_field: np.ndarray,
    *,
    sweeps: int,
    theta: float,
    flip_probability: float,
    key_data: np.ndarray,
) -> np.ndarray:
    _, jnp = _jax_modules()
    run = _compiled_rollout(int(sweeps))
    value = run(
        jnp.asarray(weights, dtype=jnp.float32),
        jnp.asarray(initial, dtype=jnp.int8),
        jnp.asarray(external_field, dtype=jnp.float32),
        jnp.asarray(theta, dtype=jnp.float32),
        jnp.asarray(flip_probability, dtype=jnp.float32),
        jnp.asarray(key_data, dtype=jnp.uint32),
    )
    return np.asarray(value, dtype=np.int8)


def in_basin(states: np.ndarray, target: np.ndarray, maximum_hamming: int = 2) -> np.ndarray:
    return np.sum(states != np.asarray(target, dtype=np.int8), axis=-1) <= maximum_hamming


def longest_true_run(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    if values.ndim != 2:
        raise ValueError("values must be sweeps by futures")
    current = np.zeros(values.shape[1], dtype=np.int16)
    longest = np.zeros(values.shape[1], dtype=np.int16)
    for row in values:
        current = np.where(row, current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def strict_destination(
    history: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    strict_run: int,
) -> tuple[np.ndarray, np.ndarray]:
    a = longest_true_run(in_basin(history, target_a)) >= strict_run
    b = longest_true_run(in_basin(history, target_b)) >= strict_run
    return a, b


def apply_challenge(
    states: np.ndarray,
    target: np.ndarray,
    challenge: str,
    *,
    neutral_damage_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.asarray(states, dtype=np.int8).copy()
    futures, genes = result.shape
    if challenge == "release":
        return result
    if challenge == "neutral_damage":
        count = max(1, int(round(genes * neutral_damage_fraction)))
        for future in range(futures):
            indices = rng.choice(genes, size=count, replace=False)
            result[future, indices] *= -1
        return result
    if challenge == "forced_break":
        minimum = min(genes, 3)
        for future in range(futures):
            aligned = np.flatnonzero(result[future] == target)
            if aligned.size < minimum:
                aligned = np.arange(genes)
            indices = rng.choice(aligned, size=min(minimum, aligned.size), replace=False)
            result[future, indices] *= -1
        return result
    raise ValueError(f"unknown challenge: {challenge}")

