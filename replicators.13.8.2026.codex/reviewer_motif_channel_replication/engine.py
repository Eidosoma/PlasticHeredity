"""Independent NumPy implementation of Rule 31649 and its local carriers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .contract import ReaderConfiguration, semantic_seed


BIRTH = np.array([1, 3, 4, 5, 6], dtype=np.uint8)
SURVIVAL = np.array([0, 5, 7, 8], dtype=np.uint8)
MOTIF_OFFSETS = tuple(
    (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
)
RING_OFFSETS = tuple(offset for offset in MOTIF_OFFSETS if offset != (0, 0))
MOTIF_WEIGHTS = {
    offset: 1 << (8 - index) for index, offset in enumerate(MOTIF_OFFSETS)
}
RING_WEIGHTS = {
    offset: 1 << (7 - index) for index, offset in enumerate(RING_OFFSETS)
}


@dataclass(frozen=True)
class CarrierPair:
    a: np.ndarray
    b: np.ndarray


def decode_state_hex(value: str, height: int = 16, width: int = 16) -> np.ndarray:
    if len(value) * 4 != height * width:
        raise ValueError("state hex length does not match board dimensions")
    raw = np.frombuffer(bytes.fromhex(value), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="big").reshape(height, width).astype(np.uint8)


def encode_state_hex(board: np.ndarray) -> str:
    board = as_board(board)
    return np.packbits(board.reshape(-1), bitorder="big").tobytes().hex()


def as_board(board: np.ndarray) -> np.ndarray:
    value = np.asarray(board, dtype=np.uint8)
    if value.ndim != 2 or not np.all((value == 0) | (value == 1)):
        raise ValueError("board must be a two-dimensional binary array")
    return value


def neighbour_count(board: np.ndarray) -> np.ndarray:
    board = as_board(board)
    count = np.zeros_like(board, dtype=np.uint8)
    for dy, dx in RING_OFFSETS:
        count += np.roll(board, shift=(dy, dx), axis=(0, 1))
    return count


def step_rule31649(board: np.ndarray) -> np.ndarray:
    board = as_board(board)
    neighbours = neighbour_count(board)
    born = (board == 0) & np.isin(neighbours, BIRTH)
    survives = (board == 1) & np.isin(neighbours, SURVIVAL)
    return (born | survives).astype(np.uint8)


def motif_addresses(board: np.ndarray) -> np.ndarray:
    board = as_board(board)
    addresses = np.zeros_like(board, dtype=np.uint16)
    for offset in MOTIF_OFFSETS:
        addresses = (addresses << 1) | np.roll(
            board, shift=(-offset[0], -offset[1]), axis=(0, 1)
        )
    return addresses


def ring_addresses(board: np.ndarray) -> np.ndarray:
    board = as_board(board)
    addresses = np.zeros_like(board, dtype=np.uint16)
    for offset in RING_OFFSETS:
        addresses = (addresses << 1) | np.roll(
            board, shift=(-offset[0], -offset[1]), axis=(0, 1)
        )
    return addresses


def motif_counts(board: np.ndarray) -> np.ndarray:
    return np.bincount(motif_addresses(board).ravel(), minlength=512).astype(np.int64)


def contextual_counts(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    addresses = ring_addresses(board)
    totals = np.bincount(addresses.ravel(), minlength=256).astype(np.int64)
    live = np.bincount(
        addresses.ravel(), weights=as_board(board).ravel(), minlength=256
    ).astype(np.int64)
    return totals, live


def parent_statistics(initial: np.ndarray, window: int) -> dict[str, np.ndarray]:
    board = as_board(initial).copy()
    motif = np.zeros(512, dtype=np.int64)
    context_total = np.zeros(256, dtype=np.int64)
    context_live = np.zeros(256, dtype=np.int64)
    for _ in range(window):
        board = step_rule31649(board)
        motif += motif_counts(board)
        total, live = contextual_counts(board)
        context_total += total
        context_live += live
    return {
        "motif": motif,
        "context_total": context_total,
        "context_live": context_live,
    }


def pooled_reference(
    statistics: Iterable[Mapping[str, np.ndarray]], alpha: float = 0.5
) -> dict[str, np.ndarray]:
    items = list(statistics)
    if not items:
        raise ValueError("at least one parent history is required")
    motif = sum((np.asarray(item["motif"]) for item in items), np.zeros(512, int))
    total = sum(
        (np.asarray(item["context_total"]) for item in items), np.zeros(256, int)
    )
    live = sum(
        (np.asarray(item["context_live"]) for item in items), np.zeros(256, int)
    )
    return {
        "motif_probability": (motif + alpha) / (motif.sum() + alpha * 512),
        "context_probability": (live + alpha) / (total + 2.0 * alpha),
    }


def write_carrier(
    statistics: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    family: str,
    alpha: float = 0.5,
    energy_clip: float = 4.0,
) -> np.ndarray:
    if family == "motif_energy512":
        counts = np.asarray(statistics["motif"], dtype=np.float64)
        probability = (counts + alpha) / (counts.sum() + alpha * 512)
        result = np.log(probability) - np.log(reference["motif_probability"])
        return np.clip(result, -energy_clip, energy_clip).astype(np.float32)
    if family == "contextual256":
        total = np.asarray(statistics["context_total"], dtype=np.float64)
        live = np.asarray(statistics["context_live"], dtype=np.float64)
        probability = (live + alpha) / (total + 2.0 * alpha)
        return (probability - reference["context_probability"]).astype(np.float32)
    raise ValueError(f"unknown carrier family: {family}")


def read_motif_energy(
    predicted: np.ndarray,
    carrier: np.ndarray,
    strength: float,
    uniform: np.ndarray,
) -> np.ndarray:
    predicted = as_board(predicted)
    carrier = np.asarray(carrier, dtype=np.float32)
    if carrier.shape != (512,):
        raise ValueError("motif carrier must have 512 entries")
    if strength <= 0.0 or not np.any(carrier):
        return predicted.copy()
    addresses = motif_addresses(predicted)
    delta = np.zeros_like(predicted, dtype=np.float32)
    for offset in MOTIF_OFFSETS:
        containing = np.roll(addresses, shift=offset, axis=(0, 1))
        flipped = containing ^ MOTIF_WEIGHTS[offset]
        delta += carrier[flipped] - carrier[containing]
    decisions = (delta > 0.0) & (np.asarray(uniform) < strength)
    return np.bitwise_xor(predicted, decisions.astype(np.uint8))


def read_contextual(
    predicted: np.ndarray,
    carrier: np.ndarray,
    strength: float,
    uniform: np.ndarray,
) -> np.ndarray:
    predicted = as_board(predicted)
    carrier = np.asarray(carrier, dtype=np.float32)
    if carrier.shape != (256,):
        raise ValueError("contextual carrier must have 256 entries")
    if strength <= 0.0 or not np.any(carrier):
        return predicted.copy()
    marks = carrier[ring_addresses(predicted)]
    eligible = ((predicted == 0) & (marks > 0)) | ((predicted == 1) & (marks < 0))
    probability = np.minimum(1.0, strength * np.abs(marks))
    decisions = eligible & (np.asarray(uniform) < probability)
    return np.bitwise_xor(predicted, decisions.astype(np.uint8))


def apply_reader(
    predicted: np.ndarray,
    carrier: np.ndarray,
    configuration: ReaderConfiguration,
    uniform: np.ndarray,
) -> np.ndarray:
    if configuration.family == "motif_energy512":
        return read_motif_energy(predicted, carrier, configuration.strength, uniform)
    if configuration.family == "contextual256":
        return read_contextual(predicted, carrier, configuration.strength, uniform)
    raise ValueError(f"unknown carrier family: {configuration.family}")


def texture2x2_counts(board: np.ndarray) -> np.ndarray:
    board = as_board(board)
    address = (
        (board << 3)
        | (np.roll(board, -1, axis=1) << 2)
        | (np.roll(board, -1, axis=0) << 1)
        | np.roll(np.roll(board, -1, axis=0), -1, axis=1)
    )
    return np.bincount(address.ravel(), minlength=16).astype(np.int64)[1:]


def normalize_live_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    return counts / total if total > 0 else np.zeros_like(counts)


def texture2x2(board: np.ndarray) -> np.ndarray:
    return normalize_live_counts(texture2x2_counts(board))


def accumulated_texture2x2(states: Iterable[np.ndarray]) -> np.ndarray:
    total = np.zeros(15, dtype=np.int64)
    for state in states:
        total += texture2x2_counts(state)
    return normalize_live_counts(total)


def component_geometry(board: np.ndarray) -> np.ndarray:
    """Translation-invariant 8-connected toroidal component diagnostics."""

    board = as_board(board)
    height, width = board.shape
    unseen = {(int(row), int(column)) for row, column in np.argwhere(board)}
    sizes: list[int] = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        size = 0
        while stack:
            row, column = stack.pop()
            size += 1
            for dy, dx in RING_OFFSETS:
                neighbour = ((row + dy) % height, (column + dx) % width)
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        sizes.append(size)
    cells = float(height * width)
    if not sizes:
        return np.zeros(3, dtype=np.float64)
    return np.array(
        [len(sizes) / cells, max(sizes) / cells, np.mean(sizes) / cells],
        dtype=np.float64,
    )


def spatial_autocorrelation(board: np.ndarray) -> np.ndarray:
    board = as_board(board).astype(np.float64)
    centered = board - board.mean()
    variance = float(np.mean(centered * centered))
    if variance == 0:
        return np.zeros(4, dtype=np.float64)
    return np.array(
        [
            np.mean(centered * np.roll(centered, shift, axis=(0, 1))) / variance
            for shift in ((0, 1), (1, 0), (1, 1), (1, -1))
        ],
        dtype=np.float64,
    )


def low_frequency_power(board: np.ndarray) -> np.ndarray:
    board = as_board(board).astype(np.float64)
    centered = board - board.mean()
    power = np.abs(np.fft.fft2(centered)) ** 2
    total = float(power.sum())
    if total == 0:
        return np.zeros(4, dtype=np.float64)
    height, width = board.shape
    indices = ((0, 1), (1, 0), (1, 1), (1, width - 1))
    return np.array([power[index] / total for index in indices], dtype=np.float64)


def board_diagnostics(board: np.ndarray) -> dict[str, Any]:
    board = as_board(board)
    return {
        "occupancy": float(board.mean()),
        "components": component_geometry(board),
        "autocorrelation": spatial_autocorrelation(board),
        "low_frequency_power": low_frequency_power(board),
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def assign_form(
    observation: np.ndarray,
    targets: Mapping[str, np.ndarray],
    similarity: float = 0.90,
    margin: float = 0.05,
) -> str | None:
    scores = {label: cosine_similarity(observation, targets[label]) for label in ("A", "B")}
    best = max(scores, key=lambda label: (scores[label], label == "A"))
    other = "B" if best == "A" else "A"
    if scores[best] >= similarity and scores[best] - scores[other] >= margin:
        return best
    return None


def deterministic_board(
    namespace: str,
    *parts: object,
    density: float = 0.5,
    shape: tuple[int, int] = (16, 16),
) -> np.ndarray:
    size = shape[0] * shape[1]
    live = int(round(density * size))
    rng = np.random.default_rng(semantic_seed(namespace, *parts, "board"))
    order = rng.permutation(size)
    board = np.zeros(size, dtype=np.uint8)
    board[order[:live]] = 1
    return board.reshape(shape)


def transform_board(board: np.ndarray, transform: str) -> np.ndarray:
    board = as_board(board)
    if transform == "identity":
        return board.copy()
    if transform == "rot90":
        return np.rot90(board, 1).copy()
    if transform == "reflect_x":
        return np.fliplr(board).copy()
    if transform.startswith("translate_"):
        _, dy, dx = transform.split("_")
        return np.roll(board, (int(dy), int(dx)), axis=(0, 1))
    raise ValueError(f"unknown transform: {transform}")


def inverse_transform_name(transform: str) -> str:
    if transform == "rot90":
        return "rot270"
    if transform in {"identity", "reflect_x"}:
        return transform
    if transform.startswith("translate_"):
        _, dy, dx = transform.split("_")
        return f"translate_{-int(dy)}_{-int(dx)}"
    if transform == "rot270":
        return "rot90"
    raise ValueError(f"unknown transform: {transform}")


def inverse_transform_board(board: np.ndarray, transform: str) -> np.ndarray:
    if transform == "rot90":
        return np.rot90(as_board(board), -1).copy()
    if transform == "reflect_x":
        return np.fliplr(as_board(board)).copy()
    if transform == "identity":
        return as_board(board).copy()
    if transform.startswith("translate_"):
        _, dy, dx = transform.split("_")
        return np.roll(as_board(board), (-int(dy), -int(dx)), axis=(0, 1))
    raise ValueError(f"unknown transform: {transform}")


def transform_motif_address(address: int, transform: str) -> int:
    bits = np.array(
        [(address >> (8 - index)) & 1 for index in range(9)], dtype=np.uint8
    ).reshape(3, 3)
    transformed = transform_board(bits, transform)
    result = 0
    for bit in transformed.ravel():
        result = (result << 1) | int(bit)
    return result


def transform_carrier(carrier: np.ndarray, transform: str) -> np.ndarray:
    carrier = np.asarray(carrier)
    if carrier.shape != (512,):
        raise ValueError("only 512-entry motif carriers are covariantly transformed")
    result = np.empty_like(carrier)
    for old in range(512):
        result[transform_motif_address(old, transform)] = carrier[old]
    return result


def corrupt_carrier_signs(
    carrier: np.ndarray, fraction: float, namespace: str, *parts: object
) -> np.ndarray:
    result = np.asarray(carrier).copy()
    count = int(round(fraction * result.size))
    if count:
        rng = np.random.default_rng(semantic_seed(namespace, *parts, "sign-corruption"))
        addresses = rng.permutation(result.size)[:count]
        result[addresses] *= -1
    return result


def permute_carrier(
    carrier: np.ndarray, namespace: str, *parts: object
) -> np.ndarray:
    rng = np.random.default_rng(semantic_seed(namespace, *parts, "carrier-permutation"))
    return np.asarray(carrier)[rng.permutation(len(carrier))]


def write_spatial_latch(
    parent: np.ndarray,
    *,
    window: int = 16,
    upper: float = 0.60,
    lower: float = 0.40,
    retention: float = 1.0,
) -> np.ndarray:
    """Write the retained round-4 positional latch benchmark from occupancy."""

    board = as_board(parent).copy()
    occupancy = np.zeros_like(board, dtype=np.float64)
    for _ in range(window):
        board = step_rule31649(board)
        occupancy += board
    occupancy /= window
    return (
        retention
        * np.where(
        occupancy >= upper,
        1.0,
        np.where(occupancy <= lower, -1.0, 0.0),
        )
    ).astype(np.float32)


def simulate_daughter(
    reset: np.ndarray,
    carrier: np.ndarray,
    configuration: ReaderConfiguration,
    namespace: str,
    seed_parts: tuple[object, ...],
    *,
    horizon: int = 64,
    checkpoints: tuple[int, ...] = (8, 16, 32, 64),
    observation_window: int = 8,
    process_noise: float = 0.0,
    read_enabled: bool = True,
    spatial_latch: np.ndarray | None = None,
    spatial_latch_strength: float | None = None,
    observation_transform: str = "identity",
    collect_diagnostics: bool = False,
) -> dict[int, dict[str, Any]]:
    board = as_board(reset).copy()
    history: deque[np.ndarray] = deque(maxlen=observation_window)
    reader_rng = np.random.default_rng(
        semantic_seed(namespace, *seed_parts, "reader-randomness")
    )
    noise_rng = np.random.default_rng(
        semantic_seed(namespace, *seed_parts, "process-randomness")
    )
    results: dict[int, dict[str, Any]] = {}
    for sweep in range(1, horizon + 1):
        predicted = step_rule31649(board)
        if process_noise > 0:
            predicted ^= (noise_rng.random(predicted.shape) < process_noise).astype(np.uint8)
        uniform = reader_rng.random(predicted.shape)
        if read_enabled and sweep <= configuration.read_duration:
            if spatial_latch is None:
                board = apply_reader(predicted, carrier, configuration, uniform)
            else:
                latch_strength = (
                    configuration.strength
                    if spatial_latch_strength is None
                    else spatial_latch_strength
                )
                probability = np.minimum(1.0, latch_strength * np.abs(spatial_latch))
                desired = spatial_latch > 0
                flip = (predicted != desired) & (uniform < probability)
                board = np.bitwise_xor(predicted, flip.astype(np.uint8))
        else:
            board = predicted
        observed = inverse_transform_board(board, observation_transform)
        history.append(observed)
        if sweep in checkpoints:
            results[sweep] = {
                "primary": accumulated_texture2x2(history),
                "terminal": texture2x2(observed),
                "alive": bool(board.any()),
                "occupancy": float(board.mean()),
            }
            if collect_diagnostics and sweep == max(checkpoints):
                results[sweep]["diagnostics"] = board_diagnostics(observed)
    return results


def paired_outcome(
    reset: np.ndarray,
    carriers: CarrierPair,
    configuration: ReaderConfiguration,
    targets_primary: Mapping[str, np.ndarray],
    targets_terminal: Mapping[str, np.ndarray],
    namespace: str,
    seed_parts: tuple[object, ...],
    **simulation_options: Any,
) -> dict[str, dict[int, dict[str, Any]]]:
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for history_label, carrier in (("A", carriers.a), ("B", carriers.b)):
        # The history label is deliberately absent from seed_parts so both arms
        # consume identical semantic random fields.
        trajectory = simulate_daughter(
            reset,
            carrier,
            configuration,
            namespace,
            seed_parts,
            **simulation_options,
        )
        output[history_label] = {}
        for checkpoint, observation in trajectory.items():
            output[history_label][checkpoint] = {
                **observation,
                "primary_assignment": assign_form(observation["primary"], targets_primary),
                "terminal_assignment": assign_form(observation["terminal"], targets_terminal),
            }
    return output
