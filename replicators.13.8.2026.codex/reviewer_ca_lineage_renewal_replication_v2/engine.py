"""Standalone vectorized implementation of the corrected lineage lifecycle."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contract import (
    CONDITIONS,
    CONTRACT,
    FIXED_CONFIGURATION,
    NAMESPACE,
    PROFILE,
    SECONDARY_CONDITIONS,
    semantic_seed,
    sha256_bytes,
)


BIRTH = np.array([1, 3, 4, 5, 6], dtype=np.uint8)
SURVIVAL = np.array([0, 5, 7, 8], dtype=np.uint8)
MOTIF_OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
RING_OFFSETS = tuple(offset for offset in MOTIF_OFFSETS if offset != (0, 0))
MOTIF_WEIGHTS = {offset: 1 << index for index, offset in enumerate(MOTIF_OFFSETS)}
CORRELATION_SHIFTS = ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2))
POWER_INDICES = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0))


def as_board(board: np.ndarray) -> np.ndarray:
    value = np.asarray(board, dtype=np.uint8)
    if value.ndim != 2 or not np.all((value == 0) | (value == 1)):
        raise ValueError("board must be a two-dimensional binary array")
    return value


def decode_state_hex(value: str, height: int = 16, width: int = 16) -> np.ndarray:
    """Decode row-major positions from least-significant to most-significant bit."""

    if len(value) * 4 != height * width:
        raise ValueError("state hex length does not match board dimensions")
    integer = int(value, 16)
    bits = np.fromiter(
        ((integer >> position) & 1 for position in range(height * width)),
        dtype=np.uint8,
        count=height * width,
    )
    return bits.reshape(height, width)


def encode_state_hex(board: np.ndarray) -> str:
    value = as_board(board).reshape(-1)
    integer = sum(int(bit) << position for position, bit in enumerate(value))
    return f"{integer:0{value.size // 4}x}"


def step_rule31649_batch(board: np.ndarray) -> np.ndarray:
    value = np.asarray(board, dtype=np.uint8)
    if value.ndim < 2 or not np.all((value == 0) | (value == 1)):
        raise ValueError("board batch must end in two binary spatial axes")
    neighbours = np.zeros_like(value, dtype=np.uint8)
    for dy, dx in RING_OFFSETS:
        neighbours += np.roll(value, shift=(dy, dx), axis=(-2, -1))
    born = (value == 0) & np.isin(neighbours, BIRTH)
    survives = (value == 1) & np.isin(neighbours, SURVIVAL)
    return (born | survives).astype(np.uint8)


def step_rule31649(board: np.ndarray) -> np.ndarray:
    return step_rule31649_batch(as_board(board))


def motif_addresses_batch(board: np.ndarray) -> np.ndarray:
    value = np.asarray(board, dtype=np.uint8)
    addresses = np.zeros_like(value, dtype=np.uint16)
    for index, (dy, dx) in enumerate(MOTIF_OFFSETS):
        addresses |= np.roll(value, shift=(-dy, -dx), axis=(-2, -1)).astype(
            np.uint16
        ) << index
    return addresses


def motif_addresses(board: np.ndarray) -> np.ndarray:
    return motif_addresses_batch(as_board(board))


def _batched_bincount(addresses: np.ndarray, bins: int) -> np.ndarray:
    leading = addresses.shape[:-2]
    flat = addresses.reshape(-1, addresses.shape[-2] * addresses.shape[-1])
    offsets = np.arange(flat.shape[0], dtype=np.int64)[:, None] * bins
    counts = np.bincount(
        (flat.astype(np.int64) + offsets).ravel(), minlength=flat.shape[0] * bins
    ).reshape(flat.shape[0], bins)
    return counts.reshape(*leading, bins)


def motif_counts_batch(board: np.ndarray) -> np.ndarray:
    return _batched_bincount(motif_addresses_batch(board), 512)


def motif_counts(board: np.ndarray) -> np.ndarray:
    return motif_counts_batch(as_board(board))


def texture2x2_addresses_batch(board: np.ndarray) -> np.ndarray:
    value = np.asarray(board, dtype=np.uint8)
    return (
        value
        | (np.roll(value, -1, axis=-1) << 1)
        | (np.roll(value, -1, axis=-2) << 2)
        | (np.roll(np.roll(value, -1, axis=-2), -1, axis=-1) << 3)
    ).astype(np.uint8)


def texture2x2_counts_batch(board: np.ndarray) -> np.ndarray:
    return _batched_bincount(texture2x2_addresses_batch(board), 16)[..., 1:]


def texture2x2_counts(board: np.ndarray) -> np.ndarray:
    return texture2x2_counts_batch(as_board(board))


def normalized_counts(counts: np.ndarray) -> np.ndarray:
    value = np.asarray(counts, dtype=np.float64)
    total = value.sum(axis=-1, keepdims=True)
    return np.divide(value, total, out=np.zeros_like(value), where=total > 0)


def parent_statistics(initial: np.ndarray, window: int) -> np.ndarray:
    board = as_board(initial).copy()
    counts = np.zeros(512, dtype=np.int64)
    for _ in range(window):
        board = step_rule31649(board)
        counts += motif_counts(board)
    return counts


def pooled_reference(
    histories: Iterable[np.ndarray], *, alpha: float = 0.5
) -> np.ndarray:
    items = [np.asarray(item, dtype=np.int64) for item in histories]
    if not items or any(item.shape != (512,) for item in items):
        raise ValueError("pooled reference needs at least one 512-bin history")
    counts = np.sum(items, axis=0, dtype=np.int64)
    return (counts + alpha) / (counts.sum() + alpha * 512)


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
    probability = (values + alpha) / (
        values.sum(axis=-1, keepdims=True) + alpha * 512
    )
    carrier = np.log(probability) - np.log(reference)
    return np.clip(carrier, -energy_clip, energy_clip).astype(np.float32)


def write_carrier(counts: np.ndarray, reference_probability: np.ndarray) -> np.ndarray:
    return write_carriers_batch(np.asarray(counts), reference_probability)


def motif_energy_advantage_batch(
    predicted: np.ndarray, carriers: np.ndarray
) -> np.ndarray:
    board = np.asarray(predicted, dtype=np.uint8)
    carrier = np.asarray(carriers, dtype=np.float32)
    if carrier.shape != (*board.shape[:-2], 512):
        raise ValueError("carrier leading axes must match the board batch")
    addresses = motif_addresses_batch(board)
    flat_carrier = carrier.reshape(-1, 512)
    delta = np.zeros(
        (flat_carrier.shape[0], board.shape[-2] * board.shape[-1]),
        dtype=np.float32,
    )
    for offset in MOTIF_OFFSETS:
        containing = np.roll(addresses, shift=offset, axis=(-2, -1)).reshape(
            flat_carrier.shape[0], -1
        )
        flipped = containing ^ MOTIF_WEIGHTS[offset]
        delta += np.take_along_axis(flat_carrier, flipped, axis=1)
        delta -= np.take_along_axis(flat_carrier, containing, axis=1)
    return delta.reshape(board.shape)


def reader_probability(advantage: np.ndarray, strength: float) -> np.ndarray:
    if strength < 0.0:
        raise ValueError("reader strength must be nonnegative")
    value = np.asarray(advantage, dtype=np.float64)
    return strength * np.tanh(
        np.maximum(value, 0.0) / float(CONTRACT["reader_scale"])
    )


def read_motif_energy_batch(
    predicted: np.ndarray,
    carriers: np.ndarray,
    strength: float,
    uniform: np.ndarray,
) -> np.ndarray:
    board = np.asarray(predicted, dtype=np.uint8)
    advantage = motif_energy_advantage_batch(board, carriers)
    probability = reader_probability(advantage, strength)
    random_field = np.broadcast_to(np.asarray(uniform, dtype=np.float64), board.shape)
    flip = random_field < probability
    return np.bitwise_xor(board, flip.astype(np.uint8))


def read_motif_energy(
    predicted: np.ndarray, carrier: np.ndarray, strength: float, uniform: np.ndarray
) -> np.ndarray:
    return read_motif_energy_batch(
        as_board(predicted), np.asarray(carrier, dtype=np.float32), strength, uniform
    )


def assign_forms_batch(
    counts: np.ndarray,
    targets: Mapping[str, np.ndarray],
    *,
    similarity: float = 0.90,
    margin: float = 0.05,
) -> np.ndarray:
    observation = normalized_counts(counts)
    target_a = np.asarray(targets["A"], dtype=np.float64)
    target_b = np.asarray(targets["B"], dtype=np.float64)

    def cosine(target: np.ndarray) -> np.ndarray:
        numerator = np.sum(observation * target, axis=-1)
        denominator = np.linalg.norm(observation, axis=-1) * np.linalg.norm(target)
        return np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
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
    if value.ndim != 2 or value.shape[0] != 2 or value.shape[1] <= 0:
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


def carrier_statistics(
    carriers: np.ndarray, alive: np.ndarray | None = None
) -> dict[str, float]:
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
            variances.append(float(np.mean((selected - centroid) ** 2)))
            absolute.append(np.abs(selected).reshape(-1))
        else:
            centroid = np.zeros(512, dtype=np.float64)
            variances.append(0.0)
        centroids.append(centroid)
    return {
        "centroid_l2": float(np.linalg.norm(centroids[0] - centroids[1])),
        "mean_abs": float(np.mean(np.concatenate(absolute))) if absolute else 0.0,
        "within_history_variance": float(np.mean(variances)),
    }


def texture_descriptor_batch(board: np.ndarray) -> np.ndarray:
    value = np.asarray(board, dtype=np.float64)
    occupancy = value.mean(axis=(-2, -1))[..., None]
    correlations = np.stack(
        [
            np.mean(
                value * np.roll(value, shift=(-dy, -dx), axis=(-2, -1)),
                axis=(-2, -1),
            )
            for dy, dx in CORRELATION_SHIFTS
        ],
        axis=-1,
    )
    power = np.abs(np.fft.fft2(value, axes=(-2, -1))) ** 2
    selected_power = np.stack([power[..., row, column] for row, column in POWER_INDICES], axis=-1)
    return np.concatenate([occupancy, correlations, selected_power], axis=-1)


def phenotype_features_batch(
    accumulated_counts: np.ndarray, terminal_counts: np.ndarray, terminal_board: np.ndarray
) -> np.ndarray:
    features = np.concatenate(
        [
            normalized_counts(accumulated_counts),
            normalized_counts(terminal_counts),
            texture_descriptor_batch(terminal_board),
        ],
        axis=-1,
    )
    if features.shape[-1] != 41:
        raise AssertionError("visible descriptor must contain exactly 41 features")
    return features


def decoder_split_indices(
    pair_id: str, split: int, replicates: int, feature_kind: str
) -> tuple[np.ndarray, np.ndarray]:
    if replicates < 2 or replicates % 2:
        raise ValueError("decoder requires an even number of at least two futures")
    rng = np.random.default_rng(
        semantic_seed(NAMESPACE, pair_id, "decoder", split)
    )
    order = rng.permutation(replicates)
    return order[: replicates // 2], order[replicates // 2 :]


def heldout_balanced_accuracy(
    source: np.ndarray,
    *,
    pair_id: str,
    feature_kind: str,
    splits: int = 4,
) -> float:
    values = np.asarray(source, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError("decoder input must be history-by-future-by-feature")
    scores: list[float] = []
    for split in range(splits):
        train_index, test_index = decoder_split_indices(
            pair_id, split, values.shape[1], feature_kind
        )
        pooled = values[:, train_index].reshape(-1, values.shape[-1])
        mean = pooled.mean(axis=0)
        scale = pooled.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        standardized_train = (values[:, train_index] - mean) / scale
        centroids = standardized_train.mean(axis=1)
        standardized_test = (values[:, test_index] - mean) / scale
        distance_a = np.sum((standardized_test - centroids[0]) ** 2, axis=-1)
        distance_b = np.sum((standardized_test - centroids[1]) ** 2, axis=-1)
        correct_a = distance_a[0] < distance_b[0]
        correct_b = distance_b[1] < distance_a[1]
        scores.append(0.5 * (float(np.mean(correct_a)) + float(np.mean(correct_b))))
    return float(np.mean(scores))


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


def corrected_sweep(
    board: np.ndarray,
    carriers: np.ndarray,
    read_enabled: np.ndarray,
    reader_uniform: np.ndarray,
    noise_mask: np.ndarray,
    *,
    strength: float = 0.25,
) -> np.ndarray:
    """One recovered sweep: CA step, reader, then process noise."""

    predicted = step_rule31649_batch(board)
    read_board = read_motif_energy_batch(predicted, carriers, strength, reader_uniform)
    enabled = np.asarray(read_enabled, dtype=bool)[..., None, None]
    after_reader = np.where(enabled, read_board, predicted).astype(np.uint8)
    return np.bitwise_xor(after_reader, np.asarray(noise_mask, dtype=np.uint8))


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
    reset_board = as_board(reset)
    board = np.broadcast_to(reset_board, (conditions, 2, replicates, *reset_board.shape)).copy()
    board *= active[..., None, None]
    writer_counts = np.zeros((conditions, 2, replicates, 512), dtype=np.int64)
    primary_counts = np.zeros((conditions, 2, replicates, 15), dtype=np.int64)
    enabled = np.broadcast_to(
        np.asarray(read_enabled, dtype=bool)[:, None, None], active.shape
    )
    for sweep in range(1, 65):
        predicted = step_rule31649_batch(board)
        if sweep <= 32:
            read_board = read_motif_energy_batch(
                predicted,
                carriers,
                float(FIXED_CONFIGURATION["strength"]),
                reader_fields[:, sweep - 1][None, None],
            )
            board = np.where(enabled[..., None, None], read_board, predicted).astype(np.uint8)
        else:
            board = predicted
        board ^= noise_fields[:, sweep - 1][None, None].astype(np.uint8)
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
    phenotype = phenotype_features_batch(primary_counts, terminal_counts, board)
    phenotype *= end_alive[..., None]
    return {
        "primary_assignments": primary_assignments,
        "terminal_assignments": terminal_assignments,
        "alive": end_alive,
        "raw_carriers": raw_carriers,
        "repaired_carriers": repaired,
        "phenotype_features": phenotype,
    }


def founder_carriers(
    donor_state_hex: Sequence[str], reference_probability: np.ndarray
) -> np.ndarray:
    if len(donor_state_hex) != 2:
        raise ValueError("exactly two founder histories are required")
    values = []
    for state_hex in donor_state_hex:
        counts = parent_statistics(decode_state_hex(state_hex), 32)
        values.append(write_carrier(counts, reference_probability))
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
    donor_initial_state_hex: Sequence[str],
    reset_state_hex: str,
    reference_probability: np.ndarray,
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
    if len(donor_initial_state_hex) != 2 or any(
        value != reset_state_hex for value in donor_initial_state_hex
    ):
        raise ValueError("both donors must share the explicit launch reset")
    reset = decode_state_hex(reset_state_hex)
    founders = founder_carriers(donor_state_hex, reference_probability)
    condition_index = {name: index for index, name in enumerate(condition_names)}
    current = np.broadcast_to(founders[:, None, :], (2, replicates, 512)).copy()
    current = current[None].repeat(len(condition_names), axis=0)
    if "opposite_founder" in condition_index:
        index = condition_index["opposite_founder"]
        current[index] = current[index, ::-1].copy()
    alive = np.ones((len(condition_names), 2, replicates), dtype=bool)
    checkpoints = set(PROFILE["generation_checkpoints"]) & set(range(1, generations + 1))
    reset_hash = sha256_bytes(reset.tobytes(order="C"))
    condition_results: dict[str, Any] = {
        name: {
            "condition": name,
            "outcomes": {},
            "carrier_history": {},
            "reset_asserted_before_every_generation": True,
            "reset_sha256": reset_hash,
            "reset_state_hex": reset_state_hex,
        }
        for name in condition_names
    }
    read_enabled = np.array(
        [name != "read_disabled" for name in condition_names], dtype=bool
    )
    secondary_carriers: dict[str, np.ndarray] = {}
    secondary_phenotypes: dict[str, np.ndarray] = {}

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
        for name in ("ablate_after_g2", "rescue_same_enter_g4", "rescue_opposite_enter_g4"):
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
            np.asarray(reference_probability, dtype=np.float64),
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
                    "surviving_futures": int(np.sum(generation_result["alive"][index])),
                }
        if generation == 16:
            for name in SECONDARY_CONDITIONS:
                if name in condition_index:
                    index = condition_index[name]
                    secondary_carriers[name] = next_carriers[index].copy()
                    secondary_phenotypes[name] = generation_result[
                        "phenotype_features"
                    ][index].copy()
        current = next_carriers
        alive = generation_result["alive"]

    secondary: dict[str, Any]
    if generation >= 16 and set(SECONDARY_CONDITIONS) <= set(condition_names):
        secondary = {"generation": 16, "carrier": {}, "phenotype": {}}
        for kind, values in (
            ("carrier", secondary_carriers),
            ("phenotype", secondary_phenotypes),
        ):
            for name in SECONDARY_CONDITIONS:
                secondary[kind][name] = heldout_balanced_accuracy(
                    values[name],
                    pair_id=pair_id,
                    feature_kind=f"{kind}:{name}",
                    splits=int(PROFILE["decoder_splits"]),
                )
    else:
        secondary = {"state": "not_collected_before_generation16"}

    founder_batch = np.broadcast_to(founders[:, None, :], (2, replicates, 512))
    return {
        "pair_id": pair_id,
        "replicates": replicates,
        "generations": generations,
        "configuration": dict(FIXED_CONFIGURATION),
        "repair": {"kind": "gain-050", "gain": 0.5, "window": [49, 64]},
        "reset": {
            "state_hex": reset_state_hex,
            "array_sha256": reset_hash,
            "live_cells": int(reset.sum()),
        },
        "founder_carrier": carrier_statistics(founder_batch),
        "conditions": condition_results,
        "secondary_decoder": secondary,
    }
