"""Elementary cellular automata and activity-gated lineage dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Callable, Sequence

from .config import ECAConfig
from .metrics import Vector, break_by, cyclic_kmer_spectrum, strict_coherent_event
from .rng import bernoulli_mask, fixed_density_bits, hash_bits, stream


CLASS_1 = frozenset({0, 8, 32, 40, 128, 136, 160, 168})
CLASS_3_CORE = frozenset({18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150})
CLASS_4 = frozenset({41, 54, 106, 110})
DISPUTED = frozenset({26, 41, 54, 73, 106, 154})
RAW_CHAMPIONS = (35, 43, 11, 57, 184)


def reflect_rule(rule: int) -> int:
    reflected = 0
    for neighbourhood in range(8):
        reverse = ((neighbourhood & 1) << 2) | (neighbourhood & 2) | ((neighbourhood & 4) >> 2)
        reflected |= ((rule >> reverse) & 1) << neighbourhood
    return reflected


def conjugate_rule(rule: int) -> int:
    conjugate = 0
    for neighbourhood in range(8):
        output = 1 - ((rule >> (7 - neighbourhood)) & 1)
        conjugate |= output << neighbourhood
    return conjugate


def rule_orbit(rule: int) -> frozenset[int]:
    if not 0 <= rule <= 255:
        raise ValueError("an ECA rule must lie in [0, 255]")
    reflected = reflect_rule(rule)
    return frozenset({rule, reflected, conjugate_rule(rule), conjugate_rule(reflected)})


def canonical_rule(rule: int) -> int:
    return min(rule_orbit(rule))


def canonical_rules() -> tuple[int, ...]:
    return tuple(sorted({canonical_rule(rule) for rule in range(256)}))


def wolfram_class(rule: int) -> int:
    representative = canonical_rule(rule)
    if representative in CLASS_1:
        return 1
    if representative in CLASS_3_CORE:
        return 3
    if representative in CLASS_4:
        return 4
    return 2


def rotate_left(row: int, amount: int, width: int) -> int:
    amount %= width
    mask = (1 << width) - 1
    return ((row << amount) & mask) | (row >> (width - amount) if amount else 0)


def rotate_right(row: int, amount: int, width: int) -> int:
    amount %= width
    mask = (1 << width) - 1
    return (row >> amount) | (((row << (width - amount)) & mask) if amount else 0)


def eca_step(row: int, rule: int, width: int) -> int:
    """Apply one synchronous periodic ECA sweep."""

    mask = (1 << width) - 1
    left = rotate_left(row, 1, width)
    centre = row & mask
    right = rotate_right(row, 1, width)
    not_left = mask ^ left
    not_centre = mask ^ centre
    not_right = mask ^ right
    output = 0
    for neighbourhood in range(8):
        if (rule >> neighbourhood) & 1:
            output |= (
                (left if neighbourhood & 4 else not_left)
                & (centre if neighbourhood & 2 else not_centre)
                & (right if neighbourhood & 1 else not_right)
            )
    return output & mask


def rule_descriptors(rule: int) -> dict[str, float]:
    bits = tuple((rule >> index) & 1 for index in range(8))
    asymmetry = sum(bits[index] != bits[((index & 1) << 2) | (index & 2) | ((index & 4) >> 2)] for index in range(8)) / 8
    changes = 0
    for index in range(8):
        for input_bit in (1, 2, 4):
            changes += bits[index] != bits[index ^ input_bit]
    return {
        "lambda": sum(bits) / 8,
        "asymmetry": asymmetry,
        "quiescent_defect": float(bits[0] != 0 and bits[7] != 1),
        "sensitivity": changes / 24,
    }


@dataclass(frozen=True)
class Generation:
    terminal: int
    dead: bool
    sweeps: int
    history: tuple[int, int, int]


def activity_generation(row: int, rule: int, config: ECAConfig, rng) -> Generation:
    width = config.width
    mask = (1 << width) - 1
    semantics = config.semantics
    activity = 0
    history = [row, row, row]
    sweeps = 0
    for sweeps in range(1, config.max_sweeps + 1):
        previous = row
        if semantics.process_noise == "pre_rule_each_sweep":
            rule_input = previous ^ bernoulli_mask(rng, config.flip_noise, width)
            deterministic = eca_step(rule_input, rule, width)
            terminal = deterministic
        else:
            rule_input = previous
            deterministic = eca_step(rule_input, rule, width)
            if semantics.monochrome_death == "deterministic_immediate" and deterministic in (0, mask):
                history = [history[-2], history[-1], deterministic]
                return Generation(deterministic, True, sweeps, tuple(history))
            if semantics.process_noise == "post_rule_each_sweep":
                terminal = deterministic ^ bernoulli_mask(rng, config.flip_noise, width)
            else:
                terminal = deterministic
        terminal &= mask
        if semantics.monochrome_death == "deterministic_immediate" and deterministic in (0, mask):
            history = [history[-2], history[-1], deterministic]
            return Generation(deterministic, True, sweeps, tuple(history))
        if semantics.activity_count == "realized":
            activity += (terminal ^ previous).bit_count()
        else:
            activity += (deterministic ^ rule_input).bit_count()
        row = terminal
        history = [history[-2], history[-1], row]
        if semantics.monochrome_death == "realized_immediate" and row in (0, mask):
            return Generation(row, True, sweeps, tuple(history))
        if (
            semantics.monochrome_death == "realized_after_minimum"
            and sweeps >= config.min_sweeps
            and row in (0, mask)
        ):
            return Generation(row, True, sweeps, tuple(history))
        if sweeps >= config.min_sweeps and activity >= config.resolved_activity_budget:
            break
    if semantics.process_noise == "terminal_once":
        row ^= bernoulli_mask(rng, config.flip_noise, width)
        row &= mask
        history[-1] = row
    dead = row == 0 or row == mask
    return Generation(row, dead, sweeps, tuple(history))


def prepare_launch_state(rule: int, seed_index: int, config: ECAConfig) -> tuple[int, tuple[int, int, int]]:
    if config.semantics.seed_mode == "expected_half_hash":
        row = hash_bits(config.seed_namespace, seed_index, config.width)
    elif config.semantics.seed_mode == "exact_half":
        row = fixed_density_bits(config.seed_namespace, seed_index, config.width, 0.5)
    else:
        density = (0.2, 0.4, 0.6, 0.8)[seed_index % 4]
        row = fixed_density_bits(config.seed_namespace, seed_index, config.width, density)
    history = [row, row, row]
    if config.semantics.launch_preparation == "noiseless_generation":
        activity = 0
        for sweep in range(1, config.max_sweeps + 1):
            terminal = eca_step(row, rule, config.width)
            activity += (terminal ^ row).bit_count()
            row = terminal
            history = [history[-2], history[-1], row]
            if sweep >= config.min_sweeps and activity >= config.resolved_activity_budget:
                break
    else:
        for _ in range(config.launch_burnin_sweeps):
            row = eca_step(row, rule, config.width)
            history = [history[-2], history[-1], row]
    return row, tuple(history)


def prepare_launch(rule: int, seed_index: int, config: ECAConfig) -> int:
    return prepare_launch_state(rule, seed_index, config)[0]


@dataclass(frozen=True)
class Lineage:
    compositions: tuple[Vector, ...]
    strict: bool
    break_by_8: bool
    first_break: int | None
    survived: int
    sweeps: tuple[int, ...]
    post_break_daughters: tuple[Vector, ...]


Observer = Callable[[int, tuple[int, int, int]], Vector | None]


def raw_observer(width: int) -> Observer:
    def observe(row: int, history: tuple[int, int, int]) -> Vector:
        return cyclic_kmer_spectrum(row, width, 4)

    return observe


def simulate_lineage(
    rule: int,
    seed_index: int,
    future_index: int,
    config: ECAConfig,
    observer: Observer | None = None,
    *,
    experiment: str = "atlas",
    observe_launch: bool | None = None,
) -> Lineage:
    observer = observer or raw_observer(config.width)
    if observe_launch is None:
        observe_launch = config.semantics.launch_anchor == "prepared_seed"
    row, initial_history = prepare_launch_state(rule, seed_index, config)
    mask = (1 << config.width) - 1
    if row == 0 or row == mask:
        return Lineage((), False, False, None, 0, (), ())
    compositions: list[Vector] = []
    if observe_launch:
        initial = observer(row, initial_history)
        if initial is None:
            return Lineage((), False, False, None, 0, (), ())
        compositions.append(initial)
    sweep_counts: list[int] = []
    rng = stream(
        config.seed_namespace,
        experiment,
        rule,
        seed_index,
        future_index,
        config.width,
        config.flip_noise,
        config.copy_error,
    )
    physical_generations = config.thresholds.horizon if observe_launch else config.thresholds.horizon + 1
    live_generations = 0
    for _generation in range(physical_generations):
        result = activity_generation(row, rule, config, rng)
        sweep_counts.append(result.sweeps)
        if result.dead:
            break
        if config.semantics.observed_daughter == "pre_copy_terminal":
            observed_row = result.terminal
            observed_history = result.history
            composition = observer(observed_row, observed_history)
            if composition is None or not any(composition):
                break
            offspring = result.terminal ^ bernoulli_mask(rng, config.copy_error, config.width)
        else:
            offspring = result.terminal ^ bernoulli_mask(rng, config.copy_error, config.width)
            observed_row = offspring
            observed_history = (result.history[-2], result.history[-1], offspring)
            composition = observer(observed_row, observed_history)
        if composition is None or not any(composition):
            break
        compositions.append(composition)
        live_generations += 1
        row = offspring

    strict = strict_coherent_event(compositions, config.thresholds)
    broke = break_by(strict.similarities, config.thresholds.break_horizon, config.thresholds.inherit)
    if strict.first_break is None:
        post_break: tuple[Vector, ...] = ()
    else:
        post_break = tuple(compositions[strict.first_break + 1 :])
    return Lineage(
        tuple(compositions),
        strict.occurred,
        broke,
        strict.first_break,
        min(config.thresholds.horizon, len(compositions) - 1 if observe_launch else live_generations),
        tuple(sweep_counts),
        post_break,
    )
