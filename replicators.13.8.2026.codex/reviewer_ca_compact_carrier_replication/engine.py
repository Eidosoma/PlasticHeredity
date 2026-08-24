"""Codec-aware lineage engine with paired ordinary and moderate environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reviewer_ca_lineage_renewal_replication_v2.engine import (
    assignment_summary,
    carrier_statistics,
    decode_state_hex,
    founder_carriers,
    simulate_generation_batch,
)

from .codec import Codec, load_codecs
from .contract import (
    CHECKPOINT_GENERATIONS,
    CONDITIONS,
    CONTRACT,
    ENVIRONMENTS,
    NAMESPACE,
    semantic_seed,
    sha256_bytes,
)


def paired_random_fields(
    pair_id: str,
    generation: int,
    replicates: int,
    *,
    namespace: str = NAMESPACE,
) -> tuple[np.ndarray, np.ndarray]:
    """Common random numbers shared across histories/conditions/codecs/environments."""

    reader = np.empty((replicates, 32, 16, 16), dtype=np.float64)
    process = np.empty((replicates, 64, 16, 16), dtype=np.float64)
    for future in range(replicates):
        reader_rng = np.random.default_rng(
            semantic_seed(namespace, pair_id, future, generation, "reader")
        )
        process_rng = np.random.default_rng(
            semantic_seed(namespace, pair_id, future, generation, "process")
        )
        reader[future] = reader_rng.random((32, 16, 16))
        process[future] = process_rng.random((64, 16, 16))
    return reader, process


def _fixed_permutation(pair_id: str, candidate_id: str, generation: int, kind: str, size: int) -> np.ndarray:
    rng = np.random.default_rng(
        semantic_seed(NAMESPACE, pair_id, candidate_id, generation, kind)
    )
    return rng.permutation(size)


def _bernoulli_latent_mask(
    pair_id: str,
    candidate_id: str,
    generation: int,
    replicates: int,
    rank: int,
    kind: str,
    probability: float,
) -> np.ndarray:
    mask = np.empty((replicates, rank), dtype=bool)
    for future in range(replicates):
        rng = np.random.default_rng(
            semantic_seed(
                NAMESPACE,
                pair_id,
                candidate_id,
                future,
                generation,
                kind,
            )
        )
        mask[future] = rng.random(rank) < probability
    return mask


def moderate_damage_masks(
    pair_id: str,
    candidate_id: str,
    generation: int,
    replicates: int,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Masks are future/candidate keyed and paired across histories/conditions."""

    erase = _bernoulli_latent_mask(
        pair_id,
        candidate_id,
        generation,
        replicates,
        rank,
        "moderate-erasure",
        float(CONTRACT["moderate_payload_erasure"]),
    )
    sign = _bernoulli_latent_mask(
        pair_id,
        candidate_id,
        generation,
        replicates,
        rank,
        "moderate-sign",
        float(CONTRACT["moderate_payload_sign_corruption"]),
    )
    return erase, sign


def apply_boundary_operations(
    payload: np.ndarray,
    *,
    intact_payload: np.ndarray,
    codec: Codec,
    pair_id: str,
    generation: int,
    environment: str,
    condition_index: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the registered pre-reader boundary order and decode the payload."""

    entry = np.asarray(payload).copy()
    if "zero_every_boundary" in condition_index:
        entry[condition_index["zero_every_boundary"]] = 0
    if "founder_write_disabled" in condition_index and generation == 1:
        entry[condition_index["founder_write_disabled"]] = 0
    if "latent_shuffle_every_boundary" in condition_index:
        index = condition_index["latent_shuffle_every_boundary"]
        permutation = _fixed_permutation(
            pair_id, codec.candidate_id, generation, "latent-permutation", codec.rank
        )
        entry[index] = entry[index][..., permutation]
    for name in (
        "ablate_after_g2",
        "rescue_same_enter_g4",
        "rescue_opposite_enter_g4",
    ):
        if name in condition_index and generation == 3:
            entry[condition_index[name]] = 0
    if "rescue_same_enter_g4" in condition_index and generation == 4:
        entry[condition_index["rescue_same_enter_g4"]] = intact_payload
    if "rescue_opposite_enter_g4" in condition_index and generation == 4:
        entry[condition_index["rescue_opposite_enter_g4"]] = intact_payload[::-1]
    if "latent_corruption_1" in condition_index:
        index = condition_index["latent_corruption_1"]
        mask = _bernoulli_latent_mask(
            pair_id,
            codec.candidate_id,
            generation,
            entry.shape[2],
            codec.rank,
            "registered-one-percent-sign",
            float(CONTRACT["registered_latent_corruption"]),
        )
        entry[index] = np.where(mask[None], -entry[index], entry[index])

    if environment == "moderate_joint":
        erase, sign = moderate_damage_masks(
            pair_id,
            codec.candidate_id,
            generation,
            entry.shape[2],
            codec.rank,
        )
        entry = np.where(erase[None, None], 0, entry)
        entry = np.where(sign[None, None], -entry, entry)
    elif environment != "ordinary":
        raise ValueError(f"unknown environment: {environment}")

    decoded = codec.decode(entry)
    if "decoded_shuffle_every_boundary" in condition_index:
        index = condition_index["decoded_shuffle_every_boundary"]
        permutation = _fixed_permutation(
            pair_id, codec.candidate_id, generation, "decoded-permutation", 512
        )
        decoded[index] = decoded[index][..., permutation]
    return entry, decoded


def simulate_pair_cell(
    *,
    pair: Mapping[str, Any],
    donors: Mapping[str, Mapping[str, Any]],
    reset_state_hex: str,
    reference_probability: np.ndarray,
    targets_primary: Mapping[str, np.ndarray],
    targets_terminal: Mapping[str, np.ndarray],
    codec: Codec,
    environment: str,
    replicates: int = 64,
    generations: int = 16,
    conditions: Sequence[str] = CONDITIONS,
    rng_namespace: str = NAMESPACE,
) -> dict[str, Any]:
    condition_names = list(conditions)
    if "intact" not in condition_names:
        raise ValueError("the intact condition is required")
    unknown = set(condition_names) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    if environment not in ENVIRONMENTS:
        raise ValueError(f"unknown environment: {environment}")
    donor_a = donors[str(pair["a_donor_id"])]
    donor_b = donors[str(pair["b_donor_id"])]
    donor_items = (donor_a, donor_b)
    if tuple(donor["prototype_label"] for donor in donor_items) != ("A", "B"):
        raise ValueError("pair order must be A then B")
    if any(donor["initial_state_hex"] != reset_state_hex for donor in donor_items):
        raise ValueError("donors must share the registered launch reset")
    founder_decoded = founder_carriers(
        [str(donor["donor_state_hex"]) for donor in donor_items],
        np.asarray(reference_probability, dtype=np.float64),
    )
    founder_payload = codec.encode(founder_decoded)
    current = np.broadcast_to(
        founder_payload[:, None, :], (2, replicates, codec.rank)
    ).copy()
    current = current[None].repeat(len(condition_names), axis=0)
    condition_index = {name: index for index, name in enumerate(condition_names)}
    if "opposite_founder" in condition_index:
        index = condition_index["opposite_founder"]
        current[index] = current[index, ::-1].copy()
    alive = np.ones((len(condition_names), 2, replicates), dtype=bool)
    reset = decode_state_hex(reset_state_hex)
    reset_hash = sha256_bytes(reset.tobytes(order="C"))
    read_enabled = np.asarray(
        [name != "read_disabled" for name in condition_names], dtype=bool
    )
    condition_results: dict[str, Any] = {
        name: {
            "condition": name,
            "outcomes": {},
            "carrier_history": {},
            "reset_asserted_before_every_generation": True,
            "reset_sha256": reset_hash,
        }
        for name in condition_names
    }
    checkpoints = set(CHECKPOINT_GENERATIONS) & set(range(1, generations + 1))
    process_probability = float(
        CONTRACT[
            "ordinary_process_noise"
            if environment == "ordinary"
            else "moderate_process_noise"
        ]
    )

    for generation in range(1, generations + 1):
        intact_payload = current[condition_index["intact"]].copy()
        entry_payload, decoded = apply_boundary_operations(
            current,
            intact_payload=intact_payload,
            codec=codec,
            pair_id=str(pair["pair_id"]),
            generation=generation,
            environment=environment,
            condition_index=condition_index,
        )
        reader, process_uniform = paired_random_fields(
            str(pair["pair_id"]),
            generation,
            replicates,
            namespace=rng_namespace,
        )
        result = simulate_generation_batch(
            reset,
            decoded,
            alive,
            read_enabled,
            reader,
            process_uniform < process_probability,
            np.asarray(reference_probability, dtype=np.float64),
            targets_primary,
            targets_terminal,
        )
        next_payload = codec.encode(result["repaired_carriers"])
        if "no_rewrite" in condition_index:
            index = condition_index["no_rewrite"]
            # No daughter writer: attenuate the decoded inherited carrier and
            # pass it through the same registered encoder at every boundary.
            next_payload[index] = codec.encode(
                float(CONTRACT["stale_retention"]) * decoded[index]
            )
        next_payload *= result["alive"][..., None]

        if generation in checkpoints:
            for name, index in condition_index.items():
                condition_results[name]["outcomes"][str(generation)] = {
                    "primary": assignment_summary(result["primary_assignments"][index]),
                    "terminal": assignment_summary(result["terminal_assignments"][index]),
                    "survival": float(np.mean(result["alive"][index])),
                }
                condition_results[name]["carrier_history"][str(generation)] = {
                    "entry": carrier_statistics(decoded[index], alive[index]),
                    "exit": carrier_statistics(
                        codec.decode(next_payload[index]), result["alive"][index]
                    ),
                    "surviving_futures": int(np.sum(result["alive"][index])),
                    "latent_nonzero_fraction_entry": float(np.mean(entry_payload[index] != 0)),
                }
        current = next_payload
        alive = result["alive"]

    founder_batch = np.broadcast_to(founder_decoded[:, None, :], (2, replicates, 512))
    return {
        "pair_id": str(pair["pair_id"]),
        "pair": dict(pair),
        "candidate": codec.metadata(),
        "environment": environment,
        "replicates": replicates,
        "generations": generations,
        "reset": {
            "state_hex": reset_state_hex,
            "array_sha256": reset_hash,
            "live_cells": int(reset.sum()),
        },
        "founder_carrier": carrier_statistics(founder_batch),
        "conditions": condition_results,
        "random_pairing": {
            "ca_reader_process_shared_across_histories_conditions_codecs_environments": True,
            "boundary_damage_shared_across_histories_conditions": True,
            "boundary_damage_keyed_by_candidate_and_future": True,
        },
    }


def worker_cell(argument: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = Path(str(argument["artifacts"]))
    codecs = load_codecs(artifacts / "input")
    codec = codecs[str(argument["candidate_id"])]
    return simulate_pair_cell(
        pair=argument["pair"],
        donors=argument["donors"],
        reset_state_hex=str(argument["reset_state_hex"]),
        reference_probability=np.asarray(argument["reference_probability"], dtype=np.float64),
        targets_primary={
            key: np.asarray(value, dtype=np.float64)
            for key, value in argument["targets_primary"].items()
        },
        targets_terminal={
            key: np.asarray(value, dtype=np.float64)
            for key, value in argument["targets_terminal"].items()
        },
        codec=codec,
        environment=str(argument["environment"]),
        replicates=int(argument["replicates"]),
        generations=int(argument["generations"]),
        conditions=argument["conditions"],
    )
