from __future__ import annotations

from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .network import Network
from .rng import jax_key


def molecular_rates_numpy(mrna: np.ndarray, protein: np.ndarray, W: np.ndarray, bias: np.ndarray, cue: np.ndarray, volume: float, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    concentration = protein / (float(cfg["protein_scale"]) * volume)
    field = float(cfg["gain"]) * ((concentration - 0.5) @ W.T) + bias + cue
    activation = 1.0 / (1.0 + np.exp(-field))
    transcription = volume * (float(cfg["transcription_basal"]) + float(cfg["transcription_induced"]) * activation)
    translation = float(cfg["translation"]) * mrna
    return transcription, translation


def _phenotype_similarity(left, right):
    def transform(value):
        transformed = jnp.log1p(value.astype(jnp.float32))
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


def _reaction_step(mrna, protein, W, bias, cue, generation_key, step_index, cfg):
    dt = float(cfg["dt"])
    volume = 1.0 + (step_index.astype(jnp.float32) + 1.0) / float(cfg["substeps"])
    concentration = protein.astype(jnp.float32) / (float(cfg["protein_scale"]) * volume)
    field = float(cfg["gain"]) * ((concentration - 0.5) @ W.T) + bias + cue
    activation = jax.nn.sigmoid(field)
    transcription = volume * (
        float(cfg["transcription_basal"]) + float(cfg["transcription_induced"]) * activation
    )
    keys = [jax.random.fold_in(generation_key, step_index + jnp.uint32(offset)) for offset in (11, 12, 13, 14)]
    born_mrna = jax.random.poisson(keys[0], transcription * dt, shape=mrna.shape).astype(jnp.int32)
    mrna_death_p = 1.0 - jnp.exp(-float(cfg["mrna_decay"]) * dt)
    dead_mrna = jax.random.binomial(keys[1], n=mrna, p=mrna_death_p, shape=mrna.shape).astype(jnp.int32)
    born_protein = jax.random.poisson(
        keys[2], float(cfg["translation"]) * mrna.astype(jnp.float32) * dt, shape=protein.shape
    ).astype(jnp.int32)
    protein_death_p = 1.0 - jnp.exp(-float(cfg["protein_decay"]) * dt)
    dead_protein = jax.random.binomial(keys[3], n=protein, p=protein_death_p, shape=protein.shape).astype(jnp.int32)
    return jnp.maximum(mrna + born_mrna - dead_mrna, 0), jnp.maximum(protein + born_protein - dead_protein, 0)


def _generation_scan(state, W, bias, cue, root_key, generation_index, cfg, erase_state=None):
    generation_key = jax.random.fold_in(root_key, generation_index)
    mrna, protein = state

    def step(carry, index):
        return _reaction_step(carry[0], carry[1], W, bias, cue, generation_key, index, cfg), None

    (mrna, protein), _ = jax.lax.scan(
        step, (mrna, protein), jnp.arange(int(cfg["substeps"]), dtype=jnp.uint32)
    )
    mrna = jax.random.binomial(
        jax.random.fold_in(generation_key, jnp.uint32(1)), n=mrna, p=0.5, shape=mrna.shape
    ).astype(jnp.int32)
    protein = jax.random.binomial(
        jax.random.fold_in(generation_key, jnp.uint32(2)), n=protein, p=0.5, shape=protein.shape
    ).astype(jnp.int32)
    if erase_state is not None:
        mrna = jnp.broadcast_to(erase_state[0], mrna.shape)
        protein = jnp.broadcast_to(erase_state[1], protein.shape)
    return mrna, protein


def _generation_loop(state, W, bias, cue, root_key, generation_index, cfg, erase_state=None):
    generation_key = jax.random.fold_in(root_key, generation_index)
    mrna, protein = state

    def body(step_index, carry):
        return _reaction_step(
            carry[0], carry[1], W, bias, cue, generation_key,
            step_index.astype(jnp.uint32), cfg,
        )

    mrna, protein = jax.lax.fori_loop(0, int(cfg["substeps"]), body, (mrna, protein))
    mrna = jax.random.binomial(
        jax.random.fold_in(generation_key, jnp.uint32(1)), n=mrna, p=0.5, shape=mrna.shape
    ).astype(jnp.int32)
    protein = jax.random.binomial(
        jax.random.fold_in(generation_key, jnp.uint32(2)), n=protein, p=0.5, shape=protein.shape
    ).astype(jnp.int32)
    if erase_state is not None:
        mrna = jnp.broadcast_to(erase_state[0], mrna.shape)
        protein = jnp.broadcast_to(erase_state[1], protein.shape)
    return mrna, protein


@lru_cache(maxsize=None)
def _future_kernel(genes: int, futures: int, horizon: int, cfg_items: tuple[tuple[str, float], ...], executor: str, erase: bool):
    cfg = dict(cfg_items)
    generation = _generation_scan if executor == "scan" else _generation_loop

    if executor == "scan":
        def kernel(W, bias, start_mrna, start_protein, root_key, erase_mrna, erase_protein):
            mrna = jnp.broadcast_to(start_mrna, (futures, genes))
            protein = jnp.broadcast_to(start_protein, (futures, genes))
            zero_cue = jnp.zeros(genes, dtype=jnp.float32)
            erased = (erase_mrna, erase_protein) if erase else None

            def body(parent, generation_index):
                child = generation(parent, W, bias, zero_cue, root_key, generation_index, cfg, erased)
                return child, (_phenotype_similarity(parent[1], child[1]), child[1], child[0])

            endpoint, (similarities, protein_trajectory, mrna_trajectory) = jax.lax.scan(
                body, (mrna, protein), jnp.arange(horizon, dtype=jnp.uint32)
            )
            return (
                jnp.swapaxes(similarities, 0, 1), endpoint[0], endpoint[1],
                jnp.swapaxes(mrna_trajectory, 0, 1), jnp.swapaxes(protein_trajectory, 0, 1),
            )
    else:
        def kernel(W, bias, start_mrna, start_protein, root_key, erase_mrna, erase_protein):
            parent = (
                jnp.broadcast_to(start_mrna, (futures, genes)),
                jnp.broadcast_to(start_protein, (futures, genes)),
            )
            zero_cue = jnp.zeros(genes, dtype=jnp.float32)
            erased = (erase_mrna, erase_protein) if erase else None
            similarities = jnp.zeros((horizon, futures), dtype=jnp.float32)
            mrna_trajectory = jnp.zeros((horizon, futures, genes), dtype=jnp.int32)
            protein_trajectory = jnp.zeros((horizon, futures, genes), dtype=jnp.int32)

            def body(generation_index, carry):
                current, similarity_values, mrna_values, protein_values = carry
                child = generation(
                    current, W, bias, zero_cue, root_key,
                    generation_index.astype(jnp.uint32), cfg, erased,
                )
                similarity_values = similarity_values.at[generation_index].set(
                    _phenotype_similarity(current[1], child[1])
                )
                mrna_values = mrna_values.at[generation_index].set(child[0])
                protein_values = protein_values.at[generation_index].set(child[1])
                return child, similarity_values, mrna_values, protein_values

            parent, similarities, mrna_trajectory, protein_trajectory = jax.lax.fori_loop(
                0, horizon, body, (parent, similarities, mrna_trajectory, protein_trajectory)
            )
            return (
                jnp.swapaxes(similarities, 0, 1), parent[0], parent[1],
                jnp.swapaxes(mrna_trajectory, 0, 1), jnp.swapaxes(protein_trajectory, 0, 1),
            )
    return jax.jit(kernel)


def _cfg_items(cfg: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    names = (
        "substeps", "dt", "gain", "transcription_basal", "transcription_induced",
        "mrna_decay", "translation", "protein_decay", "protein_scale",
    )
    return tuple((name, float(cfg[name])) for name in names)


def simulate_molecular_futures(
    network: Network,
    start_mrna: np.ndarray,
    start_protein: np.ndarray,
    protocol: dict[str, Any],
    futures: int,
    key,
    *,
    horizon: int | None = None,
    executor: str = "scan",
    erase_state: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if executor not in {"scan", "loop"}:
        raise ValueError(executor)
    cfg = protocol["tiers"]["molecular"]
    horizon_value = int(horizon or protocol["endpoint"]["secondary_horizon"])
    erase = erase_state is not None
    erased_mrna, erased_protein = erase_state if erase_state is not None else (start_mrna, start_protein)
    kernel = _future_kernel(
        int(cfg["genes"]), int(futures), horizon_value, _cfg_items(cfg), executor, erase
    )
    values = kernel(
        jnp.asarray(network.W), jnp.asarray(network.bias),
        jnp.asarray(start_mrna, dtype=jnp.int32), jnp.asarray(start_protein, dtype=jnp.int32), key,
        jnp.asarray(erased_mrna, dtype=jnp.int32), jnp.asarray(erased_protein, dtype=jnp.int32),
    )
    return tuple(np.asarray(value) for value in values)  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _history_kernel(genes: int, burnin: int, cue_generations: int, release_generations: int, cfg_items: tuple[tuple[str, float], ...]):
    cfg = dict(cfg_items)

    def run(state, W, bias, cue, root_key, count):
        def body(parent, index):
            child = _generation_scan(parent, W, bias, cue, root_key, index, cfg)
            return child, child
        return jax.lax.scan(body, state, jnp.arange(count, dtype=jnp.uint32))

    def kernel(W, bias, initial_mrna, initial_protein, cue_a, cue_b, burn_key, cue_a_key, cue_b_key, release_a_key, release_b_key):
        baseline, _ = run((initial_mrna, initial_protein), W, bias, jnp.zeros(genes), burn_key, burnin)
        a0, _ = run(baseline, W, bias, cue_a, cue_a_key, cue_generations)
        b0, _ = run(baseline, W, bias, cue_b, cue_b_key, cue_generations)
        _, release_a = run(a0, W, bias, jnp.zeros(genes), release_a_key, release_generations)
        _, release_b = run(b0, W, bias, jnp.zeros(genes), release_b_key, release_generations)
        return baseline, a0, b0, release_a, release_b
    return jax.jit(kernel)


def acquire_molecular_history(network: Network, protocol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = protocol["tiers"]["molecular"]
    history = protocol["history"]
    master = str(protocol["master_seed_label"])
    coordinate = (network.cohort, network.index)
    kernel = _history_kernel(
        int(cfg["genes"]), int(history["burnin_generations"]), int(history["cue_generations"]),
        max(int(value) for value in protocol["landmarks"]), _cfg_items(cfg),
    )
    baseline, a0, b0, release_a, release_b = kernel(
        jnp.asarray(network.W), jnp.asarray(network.bias),
        jnp.asarray(network.initial_mrna), jnp.asarray(network.initial_protein),
        jnp.asarray(network.cue_a), jnp.asarray(network.cue_b),
        jax_key(master, "history-burnin", "molecular", *coordinate),
        jax_key(master, "history-cue", "molecular", *coordinate, "A"),
        jax_key(master, "history-cue", "molecular", *coordinate, "B"),
        jax_key(master, "history-release", "molecular", *coordinate, "A"),
        jax_key(master, "history-release", "molecular", *coordinate, "B"),
    )
    landmarks = [int(value) for value in protocol["landmarks"]]
    mrna_states, protein_states = [], []
    for zero_state, released in ((a0, release_a), (b0, release_b)):
        selected_mrna = [zero_state[0] if age == 0 else released[0][age - 1] for age in landmarks]
        selected_protein = [zero_state[1] if age == 0 else released[1][age - 1] for age in landmarks]
        mrna_states.append(jnp.stack(selected_mrna))
        protein_states.append(jnp.stack(selected_protein))
    baseline_array = np.stack([np.asarray(baseline[0]), np.asarray(baseline[1])])
    return baseline_array, np.asarray(jnp.stack(mrna_states)), np.asarray(jnp.stack(protein_states))
