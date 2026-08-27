"""Dependency-free Life-like bitboards and the named-object experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import platform
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

from .config import ObserverThresholds
from .metrics import Vector, cosine, mass_support, strict_coherent_event
from .rng import bernoulli_mask, derive_seed, fixed_density_bits, stream


@dataclass(frozen=True)
class LifeConfig:
    width: int = 16
    height: int = 16
    activity_budget: int = 48
    min_sweeps: int = 4
    max_sweeps: int = 64
    flip_noise: float = 0.002
    copy_error: float = 0.005
    named_futures: int = 64
    random_seeds: int = 8
    random_futures_per_seed: int = 16
    thresholds: ObserverThresholds = ObserverThresholds()
    form_mass_quantile: float = 0.75
    seed_namespace: str = "plastic-ca-cleanroom-life-v1"

    @property
    def cells(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict:
        return asdict(self)


LIFE_PROFILES = {
    "smoke": {"named_futures": 8, "random_seeds": 2, "random_futures_per_seed": 2},
    "standard": {"named_futures": 64, "random_seeds": 8, "random_futures_per_seed": 16},
    "reference": {"named_futures": 256, "random_seeds": 32, "random_futures_per_seed": 64},
}


def life_config_for_profile(profile: str) -> LifeConfig:
    if profile not in LIFE_PROFILES:
        raise ValueError(f"unknown Life profile {profile!r}")
    return LifeConfig(**LIFE_PROFILES[profile])


def board_from_cells(cells: Iterable[tuple[int, int]], width: int, height: int) -> int:
    board = 0
    for x, y in cells:
        board |= 1 << ((y % height) * width + (x % width))
    return board


def cells_from_board(board: int, width: int, height: int) -> frozenset[tuple[int, int]]:
    return frozenset(
        (position % width, position // width)
        for position in range(width * height)
        if (board >> position) & 1
    )


def shift_x(board: int, amount: int, width: int, height: int) -> int:
    amount %= width
    row_mask = (1 << width) - 1
    result = 0
    for y in range(height):
        row = (board >> (y * width)) & row_mask
        shifted = ((row << amount) & row_mask) | (row >> (width - amount) if amount else 0)
        result |= shifted << (y * width)
    return result


def shift_y(board: int, amount: int, width: int, height: int) -> int:
    amount %= height
    mask = (1 << (width * height)) - 1
    bits = amount * width
    return ((board << bits) & mask) | (board >> (width * height - bits) if bits else 0)


def shift_board(board: int, dx: int, dy: int, width: int, height: int) -> int:
    return shift_y(shift_x(board, dx, width, height), dy, width, height)


def _count_bitplanes(neighbours: Sequence[int], mask: int) -> tuple[int, int, int, int]:
    ones = twos = fours = eights = 0
    for value in neighbours:
        carry1 = ones & value
        ones ^= value
        carry2 = twos & carry1
        twos ^= carry1
        carry3 = fours & carry2
        fours ^= carry2
        eights ^= carry3
    return ones & mask, twos & mask, fours & mask, eights & mask


def neighbour_count_mask(planes: tuple[int, int, int, int], count: int, mask: int) -> int:
    result = mask
    for bit, plane in enumerate(planes):
        result &= plane if (count >> bit) & 1 else mask ^ plane
    return result


def life_like_step(
    board: int,
    width: int,
    height: int,
    *,
    birth: frozenset[int] = frozenset({3}),
    survive: frozenset[int] = frozenset({2, 3}),
) -> int:
    mask = (1 << (width * height)) - 1
    neighbours = [
        shift_board(board, dx, dy, width, height)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx, dy) != (0, 0)
    ]
    planes = _count_bitplanes(neighbours, mask)
    born = 0
    stays = 0
    for count in birth:
        born |= neighbour_count_mask(planes, count, mask)
    for count in survive:
        stays |= neighbour_count_mask(planes, count, mask)
    return ((mask ^ board) & born) | (board & stays)


def live_2x2_spectrum(board: int, width: int, height: int) -> Vector:
    """Count the 15 non-vacuum periodic 2x2 patterns with bitboard masks."""

    mask = (1 << (width * height)) - 1
    # At each output bit, sample NW, NE, SW, SE from the source board.
    a = board
    b = shift_x(board, -1, width, height)
    c = shift_y(board, -1, width, height)
    d = shift_board(board, -1, -1, width, height)
    sources = (a, b, c, d)
    counts: list[float] = []
    for pattern in range(1, 16):
        matches = mask
        for bit, source in enumerate(sources):
            matches &= source if (pattern >> bit) & 1 else mask ^ source
        counts.append(float(matches.bit_count()))
    return tuple(counts)


@dataclass(frozen=True)
class LifeGeneration:
    terminal: int
    composition: Vector
    dead: bool
    sweeps: int


def activity_generation(board: int, config: LifeConfig, rng) -> LifeGeneration:
    activity = 0
    accumulated = [0.0] * 15
    sweeps = 0
    reached_budget = False
    for sweeps in range(1, config.max_sweeps + 1):
        deterministic = life_like_step(board, config.width, config.height)
        terminal = deterministic ^ bernoulli_mask(rng, config.flip_noise, config.cells)
        activity += (terminal ^ board).bit_count()
        board = terminal
        spectrum = live_2x2_spectrum(board, config.width, config.height)
        for index, value in enumerate(spectrum):
            accumulated[index] += value
        if sweeps >= config.min_sweeps and activity >= config.activity_budget:
            reached_budget = True
            break
    # Round-5 freezes a unit-mass spectrum for each completed generation,
    # after summing the live 2x2 census over all of that generation's sweeps.
    total = sum(accumulated)
    composition = tuple(value / total for value in accumulated) if total else (0.0,) * 15
    return LifeGeneration(board, composition, not reached_budget or board == 0, sweeps)


def named_patterns(width: int = 16, height: int = 16) -> dict[str, int]:
    x0 = width // 2 - 2
    y0 = height // 2 - 2
    patterns = {
        "block_descriptive": {(1, 1), (2, 1), (1, 2), (2, 2)},
        "blinker": {(1, 2), (2, 2), (3, 2)},
        "toad": {(2, 1), (3, 1), (4, 1), (1, 2), (2, 2), (3, 2)},
        "glider": {(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)},
        # RLE bo2bo$o4b$o3bo$4o!, one standard LWSS orientation.
        "lwss": {(1, 0), (4, 0), (0, 1), (0, 2), (4, 2), (0, 3), (1, 3), (2, 3), (3, 3)},
    }
    return {
        name: board_from_cells(((x0 + x, y0 + y) for x, y in cells), width, height)
        for name, cells in patterns.items()
    }


@dataclass(frozen=True)
class LifeLineage:
    compositions: tuple[Vector, ...]
    strict: bool
    break_by_8: bool
    survived: int
    survival_8: bool
    first_break: int | None


def simulate_life_lineage(initial: int, future_index: int, config: LifeConfig, namespace: str) -> LifeLineage:
    rng = stream(config.seed_namespace, namespace, future_index)
    board = initial
    compositions: list[Vector] = []
    # Every observed composition uses the same generation-averaged instrument.
    # One extra physical generation supplies composition zero for 32 boundaries.
    physical_survival = 0
    for _ in range(config.thresholds.horizon + 1):
        generation = activity_generation(board, config, rng)
        if generation.dead or not any(generation.composition):
            break
        physical_survival += 1
        compositions.append(generation.composition)
        board = generation.terminal ^ bernoulli_mask(rng, config.copy_error, config.cells)
    strict = strict_coherent_event(compositions, config.thresholds)
    # Retained E19/E21 naming counts generations 0..7, which expose seven
    # generation-to-generation fidelity boundaries.
    broke = any(
        value <= config.thresholds.inherit
        for value in strict.similarities[: config.thresholds.break_horizon - 1]
    )
    survived_boundaries = max(0, len(compositions) - 1)
    return LifeLineage(
        tuple(compositions),
        strict.occurred,
        broke,
        survived_boundaries,
        physical_survival >= 8,
        strict.first_break,
    )


def _summarize_life(initials: Sequence[int], futures_per_initial: int, config: LifeConfig, namespace: str) -> dict[str, object]:
    lineages: list[LifeLineage] = []
    for initial_index, initial in enumerate(initials):
        for future in range(futures_per_initial):
            lineages.append(
                simulate_life_lineage(initial, future, config, f"{namespace}:initial-{initial_index}")
            )
    similarities = [
        cosine(a, b)
        for lineage in lineages
        for a, b in zip(lineage.compositions, lineage.compositions[1:])
    ]
    # Disclosed round-5 pooling is hierarchical: equal normalized generations
    # within each future through the break-causing generation (inclusive),
    # followed by an equal mean over nonempty futures.  A long-lived future
    # therefore cannot outweigh a short-lived one.
    future_spectra: list[Vector] = []
    for lineage in lineages:
        if not lineage.compositions:
            continue
        stop = (
            min(len(lineage.compositions), lineage.first_break + 2)
            if lineage.first_break is not None
            else len(lineage.compositions)
        )
        pooled = lineage.compositions[:stop]
        future_spectra.append(
            tuple(mean(composition[index] for composition in pooled) for index in range(15))
        )
    spectrum = (
        tuple(mean(composition[index] for composition in future_spectra) for index in range(15))
        if future_spectra
        else (0.0,) * 15
    )
    return {
        "strict": mean(lineage.strict for lineage in lineages),
        "break_by_8": mean(lineage.break_by_8 for lineage in lineages),
        "mean_h": mean(similarities) if similarities else 0.0,
        "survival_8": mean(lineage.survival_8 for lineage in lineages),
        "mean_survival": mean(lineage.survived for lineage in lineages),
        "ensemble_spectrum": spectrum,
        "form_support": mass_support(spectrum, config.form_mass_quantile),
        "form_pooling": "equal futures of equal normalized generations through first break inclusive",
        "form_n_futures": len(future_spectra),
        "n_futures": len(lineages),
    }


def run_life(config: LifeConfig, output: Path) -> dict[str, object]:
    started = time.time()
    patterns = named_patterns(config.width, config.height)
    named: dict[str, dict[str, object]] = {}
    for name in ("glider", "blinker", "toad", "block_descriptive"):
        result = _summarize_life([patterns[name]], config.named_futures, config, f"named:{name}")
        result["n_cells"] = patterns[name].bit_count()
        named[name] = result

    random_results: dict[str, dict[str, object]] = {}
    for name in ("glider", "blinker", "toad"):
        density = patterns[name].bit_count() / config.cells
        initials = [
            fixed_density_bits(f"{config.seed_namespace}:random:{name}", index, config.cells, density)
            for index in range(config.random_seeds)
        ]
        random_results[name] = _summarize_life(
            initials,
            config.random_futures_per_seed,
            config,
            f"random:{name}",
        )

    pairs = (("glider", "blinker"), ("glider", "toad"), ("blinker", "toad"))
    between = {
        f"{a}|{b}": cosine(named[a]["ensemble_spectrum"], named[b]["ensemble_spectrum"])
        for a, b in pairs
    }
    supports = [int(named[name]["form_support"]) for name in ("glider", "blinker", "toad")]
    gates = {
        "gate_distinct_forms": len(set(supports)) == 3 and all(value < 0.999 for value in between.values()),
        "gate_fidelity_absolute": all(float(named[name]["mean_h"]) >= 0.9 for name in ("glider", "blinker", "toad")),
        "gate_persistence": all(
            float(named[name]["survival_8"]) - float(random_results[name]["survival_8"]) > 0.3
            for name in ("glider", "blinker", "toad")
        ),
        "between_object_cosines": between,
    }
    summary = {
        "experiment": "life_named_objects",
        "elapsed_seconds": time.time() - started,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "config": config.to_dict(),
        "gates": gates,
        "named": named,
        "density_matched_random": random_results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "observer.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "COMPLETE").write_text("complete\n")
    return summary
