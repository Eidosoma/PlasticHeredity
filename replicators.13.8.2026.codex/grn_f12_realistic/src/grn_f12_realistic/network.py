from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .rng import generator, stable_digest


@dataclass(frozen=True)
class Network:
    tier: str
    cohort: str
    index: int
    uid: str
    W: np.ndarray
    bias: np.ndarray
    cue_a: np.ndarray
    cue_b: np.ndarray
    initial_x: np.ndarray
    initial_mrna: np.ndarray
    initial_protein: np.ndarray


def _balanced_signed_matrix(rng: np.random.Generator, genes: int, probability: float) -> np.ndarray:
    mask = rng.random((genes, genes)) < probability
    np.fill_diagonal(mask, False)
    for target in np.flatnonzero(mask.sum(axis=1) == 0):
        choices = np.delete(np.arange(genes), target)
        mask[target, int(rng.choice(choices))] = True
    coordinates = np.argwhere(mask)
    signs = np.ones(len(coordinates), dtype=np.float64)
    signs[: len(signs) // 2] = -1.0
    rng.shuffle(signs)
    magnitudes = rng.lognormal(mean=-0.1, sigma=0.5, size=len(coordinates))
    weights = np.zeros((genes, genes), dtype=np.float64)
    weights[coordinates[:, 0], coordinates[:, 1]] = signs * magnitudes
    row_norm = np.abs(weights).sum(axis=1, keepdims=True)
    return weights / row_norm


def sample_network(protocol: dict[str, Any], tier: str, cohort: str, index: int) -> Network:
    if tier not in protocol["tiers"]:
        raise ValueError(tier)
    master = str(protocol["master_seed_label"])
    rng = generator(master, "network", tier, cohort, int(index))
    tier_protocol = protocol["tiers"][tier]
    genes = int(tier_protocol["genes"])
    weights = _balanced_signed_matrix(rng, genes, float(tier_protocol["edge_probability"]))
    bias = rng.normal(0.0, 0.35, size=genes)
    count = min(int(protocol["history"]["cue_genes"]), genes)
    cue_indices = rng.choice(genes, size=count, replace=False)
    cue_signs = np.ones(count, dtype=np.float64)
    cue_signs[: count // 2] = -1.0
    rng.shuffle(cue_signs)
    cue_a = np.zeros(genes, dtype=np.float64)
    cue_a[cue_indices] = cue_signs * float(protocol["history"]["cue_strength"])
    cue_b = -cue_a
    initial_x = rng.uniform(0.2, 0.8, size=genes)
    initial_mrna = rng.poisson(2.5, size=genes).astype(np.int32)
    initial_protein = rng.poisson(25.0, size=genes).astype(np.int32)
    uid = stable_digest(master, "network-uid", tier, cohort, int(index))[:24]
    return Network(
        tier=tier,
        cohort=cohort,
        index=int(index),
        uid=uid,
        W=weights.astype(np.float32),
        bias=bias.astype(np.float32),
        cue_a=cue_a.astype(np.float32),
        cue_b=cue_b.astype(np.float32),
        initial_x=initial_x.astype(np.float32),
        initial_mrna=initial_mrna,
        initial_protein=initial_protein,
    )


def network_arrays(network: Network) -> dict[str, np.ndarray]:
    return {
        "W": network.W,
        "bias": network.bias,
        "cue_a": network.cue_a,
        "cue_b": network.cue_b,
        "initial_x": network.initial_x,
        "initial_mrna": network.initial_mrna,
        "initial_protein": network.initial_protein,
        "network_index": np.asarray(network.index, dtype=np.int32),
        "network_uid": np.asarray(network.uid),
    }

