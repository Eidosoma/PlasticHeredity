from __future__ import annotations

from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .network import Network
from .rng import jax_key


def continuous_drift_numpy(x: np.ndarray, W: np.ndarray, bias: np.ndarray, cue: np.ndarray, gain: float) -> np.ndarray:
    field = gain * ((2.0 * np.asarray(x) - 1.0) @ np.asarray(W).T) + bias + cue
    target = 1.0 / (1.0 + np.exp(-field))
    return target - x


def continuous_drift_jax(x, W, bias, cue, gain: float):
    field = gain * ((2.0 * x - 1.0) @ W.T) + bias + cue
    return jax.nn.sigmoid(field) - x


def _phenotype_similarity(left, right):
    def transform(value):
        clipped = jnp.clip(value, 1e-4, 1.0 - 1e-4)
        transformed = jnp.log(clipped) - jnp.log1p(-clipped)
        return transformed - jnp.mean(transformed, axis=-1, keepdims=True)

    a = transform(left)
    b = transform(right)
    numerator = jnp.sum(a * b, axis=-1)
    norm_a = jnp.sqrt(jnp.sum(a * a, axis=-1))
    norm_b = jnp.sqrt(jnp.sum(b * b, axis=-1))
    regular = (norm_a > 0) & (norm_b > 0)
    correlation = jnp.where(regular, numerator / jnp.maximum(norm_a * norm_b, 1e-30), 0.0)
    identical = jnp.all(left == right, axis=-1)
    return jnp.where(regular, jnp.clip((1.0 + correlation) / 2.0, 0.0, 1.0), identical.astype(jnp.float32))


def _division(state, key, partition_scale: int):
    counts = jnp.rint(2.0 * partition_scale * jnp.clip(state, 0.0, 1.5)).astype(jnp.int32)
    daughter = jax.random.binomial(key, n=counts, p=0.5, shape=counts.shape)
    return daughter.astype(jnp.float32) / float(partition_scale)


def _regulate_scan(state, W, bias, cue, generation_key, substeps: int, dt: float, gain: float, sigma: float, state_max: float):
    indices = jnp.arange(substeps, dtype=jnp.uint32)

    def step(value, index):
        noise_key = jax.random.fold_in(generation_key, index + jnp.uint32(1))
        noise = jax.random.normal(noise_key, value.shape, dtype=value.dtype)
        drift = continuous_drift_jax(value, W, bias, cue, gain)
        updated = jnp.clip(value + dt * drift + sigma * jnp.sqrt(dt) * noise, 0.0, state_max)
        return updated, None

    return jax.lax.scan(step, state, indices)[0]


def _regulate_loop(state, W, bias, cue, generation_key, substeps: int, dt: float, gain: float, sigma: float, state_max: float):
    def body(step_index, value):
        index = step_index.astype(jnp.uint32)
        noise_key = jax.random.fold_in(generation_key, index + jnp.uint32(1))
        noise = jax.random.normal(noise_key, value.shape, dtype=value.dtype)
        drift = continuous_drift_jax(value, W, bias, cue, gain)
        return jnp.clip(value + dt * drift + sigma * jnp.sqrt(dt) * noise, 0.0, state_max)

    return jax.lax.fori_loop(0, substeps, body, state)


@lru_cache(maxsize=None)
def _future_kernel(
    genes: int,
    futures: int,
    horizon: int,
    substeps: int,
    dt: float,
    gain: float,
    sigma: float,
    partition_scale: int,
    state_max: float,
    executor: str,
    erase: bool,
):
    regulate = _regulate_scan if executor == "scan" else _regulate_loop

    def generation(state, W, bias, cue, root_key, generation_index, erase_state):
        generation_key = jax.random.fold_in(root_key, generation_index)
        division_key = jax.random.fold_in(generation_key, jnp.uint32(0))
        daughter = _division(state, division_key, partition_scale)
        daughter = jnp.broadcast_to(erase_state, daughter.shape) if erase else daughter
        return regulate(daughter, W, bias, cue, generation_key, substeps, dt, gain, sigma, state_max)

    if executor == "scan":
        def kernel(W, bias, start, root_key, erase_state):
            state = jnp.broadcast_to(start, (futures, genes))
            zero_cue = jnp.zeros((genes,), dtype=jnp.float32)

            def body(parent, generation_index):
                child = generation(parent, W, bias, zero_cue, root_key, generation_index, erase_state)
                return child, (_phenotype_similarity(parent, child), child)

            endpoint, (similarities, trajectory) = jax.lax.scan(
                body, state, jnp.arange(horizon, dtype=jnp.uint32)
            )
            return jnp.swapaxes(similarities, 0, 1), endpoint, jnp.swapaxes(trajectory, 0, 1)
    else:
        def kernel(W, bias, start, root_key, erase_state):
            parent = jnp.broadcast_to(start, (futures, genes))
            zero_cue = jnp.zeros((genes,), dtype=jnp.float32)
            similarities = jnp.zeros((horizon, futures), dtype=jnp.float32)
            trajectory = jnp.zeros((horizon, futures, genes), dtype=jnp.float32)

            def body(generation_index, carry):
                current, similarity_values, trajectory_values = carry
                child = generation(
                    current, W, bias, zero_cue, root_key,
                    generation_index.astype(jnp.uint32), erase_state,
                )
                similarity_values = similarity_values.at[generation_index].set(
                    _phenotype_similarity(current, child)
                )
                trajectory_values = trajectory_values.at[generation_index].set(child)
                return child, similarity_values, trajectory_values

            endpoint, similarities, trajectory = jax.lax.fori_loop(
                0, horizon, body, (parent, similarities, trajectory)
            )
            return jnp.swapaxes(similarities, 0, 1), endpoint, jnp.swapaxes(trajectory, 0, 1)
    return jax.jit(kernel)


def simulate_continuous_futures(
    network: Network,
    start: np.ndarray,
    protocol: dict[str, Any],
    futures: int,
    key,
    *,
    horizon: int | None = None,
    executor: str = "scan",
    erase_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if executor not in {"scan", "loop"}:
        raise ValueError(executor)
    cfg = protocol["tiers"]["continuous"]
    horizon_value = int(horizon or protocol["endpoint"]["secondary_horizon"])
    erase = erase_state is not None
    kernel = _future_kernel(
        int(cfg["genes"]), int(futures), horizon_value, int(cfg["substeps"]),
        float(cfg["dt"]), float(cfg["gain"]), float(cfg["noise_sigma"]),
        int(cfg["partition_scale"]), float(cfg["state_max"]), executor, erase,
    )
    erased = np.asarray(erase_state if erase_state is not None else start, dtype=np.float32)
    similarities, endpoint, trajectory = kernel(
        jnp.asarray(network.W), jnp.asarray(network.bias), jnp.asarray(start, dtype=jnp.float32), key,
        jnp.asarray(erased),
    )
    return np.asarray(similarities), np.asarray(endpoint), np.asarray(trajectory)


@lru_cache(maxsize=None)
def _history_kernel(
    genes: int,
    burnin: int,
    cue_generations: int,
    release_generations: int,
    substeps: int,
    dt: float,
    gain: float,
    sigma: float,
    partition_scale: int,
    state_max: float,
):
    def one_generation(state, W, bias, cue, root_key, index):
        generation_key = jax.random.fold_in(root_key, index)
        divided = _division(state, jax.random.fold_in(generation_key, jnp.uint32(0)), partition_scale)
        return _regulate_scan(divided, W, bias, cue, generation_key, substeps, dt, gain, sigma, state_max)

    def run_generations(state, W, bias, cue, root_key, count):
        def body(parent, index):
            child = one_generation(parent, W, bias, cue, root_key, index)
            return child, child
        return jax.lax.scan(body, state, jnp.arange(count, dtype=jnp.uint32))

    def kernel(W, bias, initial, cue_a, cue_b, burn_key, cue_a_key, cue_b_key, release_a_key, release_b_key):
        baseline, _ = run_generations(initial, W, bias, jnp.zeros(genes), burn_key, burnin)
        a0, _ = run_generations(baseline, W, bias, cue_a, cue_a_key, cue_generations)
        b0, _ = run_generations(baseline, W, bias, cue_b, cue_b_key, cue_generations)
        _, release_a = run_generations(a0, W, bias, jnp.zeros(genes), release_a_key, release_generations)
        _, release_b = run_generations(b0, W, bias, jnp.zeros(genes), release_b_key, release_generations)
        return baseline, a0, b0, release_a, release_b
    return jax.jit(kernel)


def acquire_continuous_history(network: Network, protocol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cfg = protocol["tiers"]["continuous"]
    history = protocol["history"]
    master = str(protocol["master_seed_label"])
    coordinate = (network.cohort, network.index)
    kernel = _history_kernel(
        int(cfg["genes"]), int(history["burnin_generations"]), int(history["cue_generations"]),
        max(int(value) for value in protocol["landmarks"]), int(cfg["substeps"]), float(cfg["dt"]),
        float(cfg["gain"]), float(cfg["noise_sigma"]), int(cfg["partition_scale"]), float(cfg["state_max"]),
    )
    baseline, a0, b0, release_a, release_b = kernel(
        jnp.asarray(network.W), jnp.asarray(network.bias), jnp.asarray(network.initial_x),
        jnp.asarray(network.cue_a), jnp.asarray(network.cue_b),
        jax_key(master, "history-burnin", "continuous", *coordinate),
        jax_key(master, "history-cue", "continuous", *coordinate, "A"),
        jax_key(master, "history-cue", "continuous", *coordinate, "B"),
        jax_key(master, "history-release", "continuous", *coordinate, "A"),
        jax_key(master, "history-release", "continuous", *coordinate, "B"),
    )
    landmarks = [int(value) for value in protocol["landmarks"]]
    states = []
    for zero_state, released in ((a0, release_a), (b0, release_b)):
        states.append(jnp.stack([zero_state if age == 0 else released[age - 1] for age in landmarks]))
    return np.asarray(baseline), np.asarray(jnp.stack(states))
