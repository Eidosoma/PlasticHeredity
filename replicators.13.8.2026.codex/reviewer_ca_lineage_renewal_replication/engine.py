"""Vectorized clean-room implementation of the registered lineage lifecycle."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from reviewer_motif_channel_replication.engine import (
    BIRTH,
    MOTIF_OFFSETS,
    MOTIF_WEIGHTS,
    RING_OFFSETS,
    decode_state_hex,
    deterministic_board,
    parent_statistics,
    write_carrier,
)

from .contract import CONDITIONS, CONTRACT, FIXED_CONFIGURATION, NAMESPACE, semantic_seed, sha256_bytes


def step_rule31649_batch(board: np.ndarray) -> np.ndarray:
    value = np.asarray(board, dtype=np.uint8)
    if value.ndim < 2 or not np.all((value == 0) | (value == 1)):
        raise ValueError("board batch must end in two binary spatial axes")
    neighbours = np.zeros_like(value, dtype=np.uint8)
    for dy, dx in RING_OFFSETS:
        neighbours += np.roll(value, shift=(dy, dx), axis=(-2, -1))
    born = (value == 0) & np.isin(neighbours, BIRTH)
    survives = (value == 1) & np.isin(neighbours, np.array([0, 5, 7, 8], np.uint8))
    return (born | survives).astype(np.uint8)


def motif_addresses_batch(board: np.ndarray) -> np.ndarray:
    value = np.asarray(board, dtype=np.uint8)
    addresses = np.zeros_like(value, dtype=np.uint16)
    for dy, dx in MOTIF_OFFSETS:
        addresses = (addresses << 1) | np.roll(
            value, shift=(-dy, -dx), axis=(-2, -1)
        )
    return addresses


def motif_counts_batch(board: np.ndarray) -> np.ndarray:
    addresses = motif_addresses_batch(board)
    leading = addresses.shape[:-2]
    flat = addresses.reshape(-1, addresses.shape[-2] * addresses.shape[-1])
    offsets = np.arange(flat.shape[0], dtype=np.int64)[:, None] * 512
    counts = np.bincount(
        (flat.astype(np.int64) + offsets).ravel(), minlength=flat.shape[0] * 512
    ).reshape(flat.shape[0], 512)
    return counts.reshape(*leading, 512)


def texture2x2_counts_batch(board: np.ndarray) -> np.ndarray:
    value = np.asarray(board, dtype=np.uint8)
    address = (
        (value << 3)
        | (np.roll(value, -1, axis=-1) << 2)
        | (np.roll(value, -1, axis=-2) << 1)
        | np.roll(np.roll(value, -1, axis=-2), -1, axis=-1)
    )
    leading = address.shape[:-2]
    flat = address.reshape(-1, address.shape[-2] * address.shape[-1])
    offsets = np.arange(flat.shape[0], dtype=np.int64)[:, None] * 16
    counts = np.bincount(
        (flat.astype(np.int64) + offsets).ravel(), minlength=flat.shape[0] * 16
    ).reshape(flat.shape[0], 16)
    return counts[:, 1:].reshape(*leading, 15)


def write_carriers_batch(
    counts: np.ndarray,
    reference_probability: np.ndarray,
    *,
    alpha: float = 0.5,
    energy_clip: float = 4.0,
) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    reference = np.asarray(reference_probability, dtype=np.float64)
    if values.shape[-1] != 512 or reference.shape != (512,):
        raise ValueError("motif writer requires 512 bins")
    probability = (values + alpha) / (values.sum(axis=-1, keepdims=True) + alpha * 512)
    carrier = np.log(probability) - np.log(reference)
    return np.clip(carrier, -energy_clip, energy_clip).astype(np.float32)


def read_motif_energy_batch(
    predicted: np.ndarray,
    carriers: np.ndarray,
    strength: float,
    uniform: np.ndarray,
) -> np.ndarray:
    board = np.asarray(predicted, dtype=np.uint8)
    carrier = np.asarray(carriers, dtype=np.float32)
    if carrier.shape != (*board.shape[:-2], 512):
        raise ValueError("carrier leading axes must match the board batch")
    addresses = motif_addresses_batch(board)
    flat_carrier = carrier.reshape(-1, 512)
    delta = np.zeros((flat_carrier.shape[0], board.shape[-2] * board.shape[-1]), np.float32)
    for offset in MOTIF_OFFSETS:
        containing = np.roll(addresses, shift=offset, axis=(-2, -1)).reshape(
            flat_carrier.shape[0], -1
        )
        flipped = containing ^ MOTIF_WEIGHTS[offset]
        delta += np.take_along_axis(flat_carrier, flipped, axis=1)
        delta -= np.take_along_axis(flat_carrier, containing, axis=1)
    delta = delta.reshape(board.shape)
    random_field = np.broadcast_to(np.asarray(uniform), board.shape)
    flip = (delta > 0.0) & (random_field < strength)
    return np.bitwise_xor(board, flip.astype(np.uint8))


def normalized_texture(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    total = values.sum(axis=-1, keepdims=True)
    return np.divide(values, total, out=np.zeros_like(values), where=total > 0)


def assign_forms_batch(
    counts: np.ndarray,
    targets: Mapping[str, np.ndarray],
    *,
    similarity: float = 0.90,
    margin: float = 0.05,
) -> np.ndarray:
    observation = normalized_texture(counts)
    target_a = np.asarray(targets["A"], dtype=np.float64)
    target_b = np.asarray(targets["B"], dtype=np.float64)

    def cosine(target: np.ndarray) -> np.ndarray:
        numerator = np.sum(observation * target, axis=-1)
        denominator = np.linalg.norm(observation, axis=-1) * np.linalg.norm(target)
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )

    score_a = cosine(target_a)
    score_b = cosine(target_b)
    choose_a = score_a >= score_b
    best = np.where(choose_a, score_a, score_b)
    other = np.where(choose_a, score_b, score_a)
    resolved = (best >= similarity) & ((best - other) >= margin)
    return np.where(resolved, np.where(choose_a, 0, 1), -1).astype(np.int8)


def assignment_summary(assignments: np.ndarray) -> dict[str, float]:
    value = np.asarray(assignments)
    if value.ndim != 2 or value.shape[0] != 2:
        raise ValueError("assignments must be history-by-future")
    p_a_given_a = float(np.mean(value[0] == 0))
    p_b_given_a = float(np.mean(value[0] == 1))
    p_a_given_b = float(np.mean(value[1] == 0))
    p_b_given_b = float(np.mean(value[1] == 1))
    direction_a = p_a_given_a - p_a_given_b
    direction_b = p_b_given_b - p_b_given_a
    return {
        "p_a_given_a": p_a_given_a,
        "p_a_given_b": p_a_given_b,
        "p_b_given_a": p_b_given_a,
        "p_b_given_b": p_b_given_b,
        "direction_a": direction_a,
        "direction_b": direction_b,
        "crossover": min(direction_a, direction_b),
        "correct": 0.5 * (p_a_given_a + p_b_given_b),
        "resolved": float(np.mean(value >= 0)),
    }


def carrier_statistics(carriers: np.ndarray, alive: np.ndarray | None = None) -> dict[str, float]:
    value = np.asarray(carriers, dtype=np.float64)
    if value.ndim != 3 or value.shape[0] != 2 or value.shape[-1] != 512:
        raise ValueError("carriers must be history-by-future-by-address")
    mask = np.ones(value.shape[:2], dtype=bool) if alive is None else np.asarray(alive, bool)
    centroids: list[np.ndarray] = []
    variances: list[float] = []
    absolute: list[np.ndarray] = []
    for history in range(2):
        selected = value[history, mask[history]]
        if selected.size:
            centroid = selected.mean(axis=0)
            variance = float(np.mean((selected - centroid) ** 2))
            absolute.append(np.abs(selected).reshape(-1))
        else:
            centroid = np.zeros(512, dtype=np.float64)
            variance = 0.0
        centroids.append(centroid)
        variances.append(variance)
    return {
        "centroid_l2": float(np.linalg.norm(centroids[0] - centroids[1])),
        "mean_abs": float(np.mean(np.concatenate(absolute))) if absolute else 0.0,
        "within_history_variance": float(np.mean(variances)),
    }


def semantic_fields(
    pair_id: str,
    generation: int,
    replicates: int,
    *,
    namespace: str = NAMESPACE,
) -> tuple[np.ndarray, np.ndarray]:
    reader = np.empty((replicates, 32, 16, 16), dtype=np.float64)
    noise = np.empty((replicates, 64, 16, 16), dtype=bool)
    for replicate in range(replicates):
        reader_rng = np.random.default_rng(
            semantic_seed(namespace, pair_id, replicate, generation, "reader")
        )
        noise_rng = np.random.default_rng(
            semantic_seed(namespace, pair_id, replicate, generation, "process")
        )
        reader[replicate] = reader_rng.random((32, 16, 16))
        noise[replicate] = noise_rng.random((64, 16, 16)) < float(
            CONTRACT["process_noise"]
        )
    return reader, noise


def simulate_generation_batch(
    reset: np.ndarray,
    entry_carriers: np.ndarray,
    alive: np.ndarray,
    read_enabled: np.ndarray,
    reader_fields: np.ndarray,
    noise_fields: np.ndarray,
    reference_probability: np.ndarray,
    targets_primary: Mapping[str, np.ndarray],
    targets_terminal: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    carriers = np.asarray(entry_carriers, dtype=np.float32)
    active = np.asarray(alive, dtype=bool)
    if carriers.ndim != 4 or carriers.shape[1] != 2 or carriers.shape[-1] != 512:
        raise ValueError("entry carriers must be condition-by-history-by-future-by-address")
    conditions, _, replicates, _ = carriers.shape
    reset_board = np.asarray(reset, dtype=np.uint8)
    board = np.broadcast_to(
        reset_board, (conditions, 2, replicates, *reset_board.shape)
    ).copy()
    board *= active[..., None, None]
    writer_counts = np.zeros((conditions, 2, replicates, 512), dtype=np.int64)
    primary_counts = np.zeros((conditions, 2, replicates, 15), dtype=np.int64)
    enabled = np.asarray(read_enabled, dtype=bool)[:, None, None, None, None]
    for sweep in range(1, 65):
        predicted = step_rule31649_batch(board)
        predicted ^= noise_fields[:, sweep - 1][None, None].astype(np.uint8)
        predicted *= active[..., None, None]
        if sweep <= 32:
            read_board = read_motif_energy_batch(
                predicted,
                carriers,
                float(FIXED_CONFIGURATION["strength"]),
                reader_fields[:, sweep - 1][None, None],
            )
            board = np.where(enabled, read_board, predicted).astype(np.uint8)
        else:
            board = predicted
        board *= active[..., None, None]
        if 49 <= sweep <= 64:
            writer_counts += motif_counts_batch(board)
        if 57 <= sweep <= 64:
            primary_counts += texture2x2_counts_batch(board)
    terminal_counts = texture2x2_counts_batch(board)
    end_alive = active & np.any(board, axis=(-2, -1))
    raw_carriers = write_carriers_batch(writer_counts, reference_probability)
    raw_carriers *= end_alive[..., None]
    repaired = float(CONTRACT["repair_gain"]) * raw_carriers
    primary_assignments = assign_forms_batch(primary_counts, targets_primary)
    terminal_assignments = assign_forms_batch(terminal_counts, targets_terminal)
    primary_assignments = np.where(end_alive, primary_assignments, -1)
    terminal_assignments = np.where(end_alive, terminal_assignments, -1)
    return {
        "primary_assignments": primary_assignments,
        "terminal_assignments": terminal_assignments,
        "alive": end_alive,
        "raw_carriers": raw_carriers,
        "repaired_carriers": repaired,
    }


def founder_carriers(
    donor_state_hex: Sequence[str], reference: Mapping[str, np.ndarray]
) -> np.ndarray:
    values = []
    for state_hex in donor_state_hex:
        board = decode_state_hex(state_hex)
        statistics = parent_statistics(board, 32)
        values.append(write_carrier(statistics, reference, "motif_energy512"))
    return np.stack(values).astype(np.float32)


def _permutation(pair_id: str, generation: int) -> np.ndarray:
    rng = np.random.default_rng(
        semantic_seed(NAMESPACE, pair_id, generation, "boundary-permutation")
    )
    return rng.permutation(512)


def _corruption_addresses(pair_id: str, generation: int) -> np.ndarray:
    rng = np.random.default_rng(
        semantic_seed(NAMESPACE, pair_id, generation, "boundary-corruption")
    )
    count = int(round(float(CONTRACT["carrier_corruption"]) * 512))
    return rng.permutation(512)[:count]


def simulate_pair_lineages(
    *,
    pair_id: str,
    donor_state_hex: Sequence[str],
    reference: Mapping[str, np.ndarray],
    targets_primary: Mapping[str, np.ndarray],
    targets_terminal: Mapping[str, np.ndarray],
    replicates: int = 64,
    generations: int = 16,
    conditions: Sequence[str] = tuple(CONDITIONS),
) -> dict[str, Any]:
    condition_names = list(conditions)
    if "intact" not in condition_names:
        raise ValueError("the intact sister is required for registered rescue branches")
    unknown = set(condition_names) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown lineage conditions: {sorted(unknown)}")
    reference_probability = np.asarray(reference["motif_probability"], np.float64)
    founders = founder_carriers(donor_state_hex, reference)
    condition_index = {name: index for index, name in enumerate(condition_names)}
    current = np.broadcast_to(
        founders[:, None, :], (2, replicates, 512)
    ).copy()[None].repeat(len(condition_names), axis=0)
    if "opposite_founder" in condition_index:
        current[condition_index["opposite_founder"]] = current[
            condition_index["opposite_founder"], ::-1
        ].copy()
    alive = np.ones((len(condition_names), 2, replicates), dtype=bool)
    reset = deterministic_board(NAMESPACE, pair_id, "native-reset")
    checkpoints = {1, 2, 4, 8, 16} & set(range(1, generations + 1))
    condition_results: dict[str, Any] = {
        name: {
            "condition": name,
            "outcomes": {},
            "carrier_history": {},
            "reset_asserted_before_every_generation": True,
            "reset_sha256": sha256_bytes(reset.tobytes(order="C")),
        }
        for name in condition_names
    }
    read_enabled = np.array(
        [name != "read_disabled" for name in condition_names], dtype=bool
    )
    for generation in range(1, generations + 1):
        entry = current.copy()
        intact_entry = entry[condition_index["intact"]].copy()
        if "zero_every_boundary" in condition_index:
            entry[condition_index["zero_every_boundary"]] = 0.0
        if "founder_write_disabled" in condition_index and generation == 1:
            entry[condition_index["founder_write_disabled"]] = 0.0
        if "shuffle_every_boundary" in condition_index:
            index = condition_index["shuffle_every_boundary"]
            entry[index] = entry[index][..., _permutation(pair_id, generation)]
        for name in (
            "ablate_after_g2",
            "rescue_same_enter_g4",
            "rescue_opposite_enter_g4",
        ):
            if name in condition_index and generation == 3:
                entry[condition_index[name]] = 0.0
        if "rescue_same_enter_g4" in condition_index and generation == 4:
            entry[condition_index["rescue_same_enter_g4"]] = intact_entry
        if "rescue_opposite_enter_g4" in condition_index and generation == 4:
            entry[condition_index["rescue_opposite_enter_g4"]] = intact_entry[::-1]
        if "carrier_corruption_1" in condition_index:
            index = condition_index["carrier_corruption_1"]
            addresses = _corruption_addresses(pair_id, generation)
            entry[index, ..., addresses] *= -1.0

        reader_fields, noise_fields = semantic_fields(pair_id, generation, replicates)
        generation_result = simulate_generation_batch(
            reset,
            entry,
            alive,
            read_enabled,
            reader_fields,
            noise_fields,
            reference_probability,
            targets_primary,
            targets_terminal,
        )
        next_carriers = generation_result["repaired_carriers"].copy()
        if "no_rewrite" in condition_index:
            index = condition_index["no_rewrite"]
            next_carriers[index] = float(CONTRACT["stale_retention"]) * entry[index]
            next_carriers[index] *= generation_result["alive"][index, ..., None]
        if generation in checkpoints:
            for name, index in condition_index.items():
                condition_results[name]["outcomes"][str(generation)] = {
                    "primary": assignment_summary(
                        generation_result["primary_assignments"][index]
                    ),
                    "terminal": assignment_summary(
                        generation_result["terminal_assignments"][index]
                    ),
                    "survival": float(np.mean(generation_result["alive"][index])),
                }
                condition_results[name]["carrier_history"][str(generation)] = {
                    "entry": carrier_statistics(entry[index], alive[index]),
                    "exit": carrier_statistics(
                        next_carriers[index], generation_result["alive"][index]
                    ),
                    "surviving_futures": int(
                        np.sum(generation_result["alive"][index])
                    ),
                }
        current = next_carriers
        alive = generation_result["alive"]

    founder_batch = np.broadcast_to(founders[:, None, :], (2, replicates, 512))
    return {
        "pair_id": pair_id,
        "replicates": replicates,
        "generations": generations,
        "configuration": dict(FIXED_CONFIGURATION),
        "repair": {"kind": "gain-050", "gain": 0.5, "window": [49, 64]},
        "founder_carrier": carrier_statistics(founder_batch),
        "conditions": condition_results,
    }
