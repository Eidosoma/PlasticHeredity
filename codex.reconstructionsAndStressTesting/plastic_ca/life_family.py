"""Clean-room Life-like family atlas and clock-scaling experiments.

The retained E24 campaign sampled 1,024 rules from the 17-bit B/S rule
family.  This module implements that experiment independently: no sibling
implementation is imported, rule outcomes are simulated from an explicit
contract, and every rule is checkpointed under the contract digest.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import mean, median
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .config import ObserverThresholds
from .e19 import PINNED_NUMPY, require_pinned_numpy
from .life import named_patterns
from .rng import stream
from .stats import quantile, spearman


RULE_REGISTRY = Path(__file__).with_name("data") / "life_family_rule_ids.json"

NAMED_RULE_NOTATIONS: dict[str, str] = {
    "life": "B3/S23",
    "highlife": "B36/S23",
    "seeds": "B2/S",
    "daynight": "B3678/S34678",
    "diamoeba": "B35678/S5678",
    "morley": "B368/S245",
    "twobytwo": "B36/S125",
    "anneal": "B4678/S35678",
}


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_life_rule(notation: str) -> int:
    """Encode ``B.../S...`` as 8 birth bits above 9 survival bits."""

    try:
        birth_text, survive_text = notation.upper().split("/", 1)
    except ValueError as error:
        raise ValueError(f"invalid Life-like notation {notation!r}") from error
    if not birth_text.startswith("B") or not survive_text.startswith("S"):
        raise ValueError(f"invalid Life-like notation {notation!r}")
    births = [int(value) for value in birth_text[1:]]
    survives = [int(value) for value in survive_text[1:]]
    if any(value < 1 or value > 8 for value in births) or len(set(births)) != len(births):
        raise ValueError("birth counts must be unique digits from 1 through 8; B0 is excluded")
    if any(value < 0 or value > 8 for value in survives) or len(set(survives)) != len(survives):
        raise ValueError("survival counts must be unique digits from 0 through 8")
    rule = sum(1 << value for value in survives)
    rule |= sum(1 << (8 + value) for value in births)
    return rule


def life_rule_notation(rule: int) -> str:
    if not 0 <= rule < (1 << 17):
        raise ValueError("a Life-like rule must be a 17-bit integer")
    births = "".join(str(value) for value in range(1, 9) if rule & (1 << (8 + value)))
    survives = "".join(str(value) for value in range(9) if rule & (1 << value))
    return f"B{births}/S{survives}"


def rule_sets(rule: int) -> tuple[frozenset[int], frozenset[int]]:
    return (
        frozenset(value for value in range(1, 9) if rule & (1 << (8 + value))),
        frozenset(value for value in range(9) if rule & (1 << value)),
    )


@dataclass(frozen=True)
class LifeFamilyContract:
    implementation_version: str = "life-family-cleanroom-v1"
    width: int = 16
    height: int = 16
    activity_budget: int = 48
    min_sweeps: int = 4
    max_sweeps: int = 64
    flip_noise: float = 0.002
    copy_error: float = 0.005
    futures_per_launch: int = 64
    horizon: int = 32
    thresholds: ObserverThresholds = field(default_factory=ObserverThresholds)
    form_mass_quantile: float = 0.75
    launch_densities: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    launch_tag: str = "life-family-launch-v1"
    rng_tag: str = "life-family-trajectory-v1"
    launch_rows_hex: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Life-family dimensions must be positive")
        if self.activity_budget <= 0 or self.min_sweeps <= 0:
            raise ValueError("activity and sweep limits must be positive")
        if self.max_sweeps < self.min_sweeps:
            raise ValueError("maximum sweeps must be at least the minimum")
        if self.futures_per_launch <= 0 or self.horizon <= 0:
            raise ValueError("future and horizon counts must be positive")
        if len(self.launch_densities) != 4:
            raise ValueError("the frozen family observer uses four density launches")
        if any(not 0.0 <= value <= 1.0 for value in self.launch_densities):
            raise ValueError("launch densities must lie in [0, 1]")
        if any(not 0.0 <= value <= 1.0 for value in (self.flip_noise, self.copy_error)):
            raise ValueError("noise probabilities must lie in [0, 1]")
        if self.launch_rows_hex is not None and len(self.launch_rows_hex) != 8:
            raise ValueError("a launch override must contain exactly eight boards")

    @property
    def cells(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "rule_update": "synchronous_toroidal_Moore_8",
                "process_noise_order": "post_rule_each_sweep",
                "activity_count": "realized_post_noise_state_changes",
                "death": "activity_timeout_or_empty_terminal_board",
                "composition": "unit_normalized_sum_of_live_2x2_censuses_across_generation",
                "observed_daughter": "terminal_pre_copy",
                "copy_draw": "unconditional_generation_batch",
                "composition_zero": "first_completed_generation",
                "form_pooling": (
                    "per launch, equal mean of each broken future's last completed composition; "
                    "mass support at quantile 0.75"
                ),
            }
        )
        return value

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LifeFamilyRuleResult:
    rule: int
    notation: str
    strict: float
    break_by_8: float
    median_gen_sweeps: float
    mean_survival: float
    form_supports: tuple[int, ...]
    n_futures: int
    total_sweeps: int
    death_counts: dict[str, int]
    per_launch: tuple[dict[str, Any], ...]

    @property
    def library(self) -> frozenset[int]:
        return frozenset(value for value in self.form_supports if value)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["form_supports"] = list(self.form_supports)
        value["per_launch"] = list(self.per_launch)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LifeFamilyRuleResult":
        return cls(
            rule=int(value["rule"]),
            notation=str(value["notation"]),
            strict=float(value["strict"]),
            break_by_8=float(value["break_by_8"]),
            median_gen_sweeps=float(value["median_gen_sweeps"]),
            mean_survival=float(value["mean_survival"]),
            form_supports=tuple(int(item) for item in value["form_supports"]),
            n_futures=int(value["n_futures"]),
            total_sweeps=int(value["total_sweeps"]),
            death_counts={str(key): int(item) for key, item in value["death_counts"].items()},
            per_launch=tuple(dict(item) for item in value["per_launch"]),
        )


def load_rule_registry(path: Path = RULE_REGISTRY) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rules = tuple(int(rule) for rule in payload["rule_ids"])
    if len(rules) != 1024 or len(set(rules)) != len(rules):
        raise ValueError("the Life-family registry must contain 1,024 unique rules")
    if rules != tuple(sorted(rules)):
        raise ValueError("the Life-family registry must be sorted")
    return rules


def _board_to_array(board: int, width: int, height: int) -> np.ndarray:
    positions = np.arange(width * height, dtype=object)
    return np.asarray([(board >> int(position)) & 1 for position in positions], dtype=np.bool_).reshape(
        height, width
    )


def _array_to_hex(board: np.ndarray) -> str:
    value = 0
    for position, bit in enumerate(board.reshape(-1)):
        value |= int(bit) << position
    digits = (board.size + 3) // 4
    return f"{value:0{digits}x}"


def _hash_density_board(contract: LifeFamilyContract, density: float) -> np.ndarray:
    count = min(contract.cells, max(0, int(contract.cells * density + 0.5)))
    ranked = sorted(
        range(contract.cells),
        key=lambda cell: hashlib.sha256(
            f"{contract.launch_tag}:{contract.width}x{contract.height}:{density:.8f}:{cell}".encode()
        ).digest(),
    )
    board = np.zeros(contract.cells, dtype=np.bool_)
    board[ranked[:count]] = True
    return board.reshape(contract.height, contract.width)


def launch_library(contract: LifeFamilyContract) -> tuple[np.ndarray, ...]:
    if contract.launch_rows_hex is not None:
        return tuple(
            _board_to_array(int(value, 16), contract.width, contract.height)
            for value in contract.launch_rows_hex
        )
    patterns = named_patterns(contract.width, contract.height)
    named = tuple(
        _board_to_array(patterns[name], contract.width, contract.height)
        for name in ("glider", "blinker", "toad", "block_descriptive")
    )
    densities = tuple(_hash_density_board(contract, value) for value in contract.launch_densities)
    return named + densities


def launch_manifest(contract: LifeFamilyContract) -> list[dict[str, Any]]:
    names = ("glider", "blinker", "toad", "block", "density-0", "density-1", "density-2", "density-3")
    rows = launch_library(contract)
    return [
        {
            "index": index,
            "name": names[index],
            "hex": _array_to_hex(row),
            "live_cells": int(np.count_nonzero(row)),
            "density": float(np.count_nonzero(row) / row.size),
        }
        for index, row in enumerate(rows)
    ]


def _rule_lookups(rule: int) -> tuple[np.ndarray, np.ndarray]:
    births, survives = rule_sets(rule)
    birth_lookup = np.zeros(9, dtype=np.bool_)
    survive_lookup = np.zeros(9, dtype=np.bool_)
    if births:
        birth_lookup[list(births)] = True
    if survives:
        survive_lookup[list(survives)] = True
    return birth_lookup, survive_lookup


def _life_like_step_lookup(
    states: np.ndarray,
    birth_lookup: np.ndarray,
    survive_lookup: np.ndarray,
) -> np.ndarray:
    if states.ndim != 3:
        raise ValueError("Life batches must have shape (future, height, width)")
    counts = np.zeros(states.shape, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                counts += np.roll(states, shift=(dy, dx), axis=(1, 2))
    return ((~states) & birth_lookup[counts]) | (states & survive_lookup[counts])


def life_like_step_batch(states: np.ndarray, rule: int) -> np.ndarray:
    """Apply one B/S rule to boards shaped ``(future, height, width)``."""

    return _life_like_step_lookup(states, *_rule_lookups(rule))


def live_2x2_counts_batch(states: np.ndarray) -> np.ndarray:
    if states.ndim != 3:
        raise ValueError("Life batches must have shape (future, height, width)")
    codes = states.astype(np.uint8)
    codes |= np.roll(states, -1, axis=2).astype(np.uint8) << 1
    codes |= np.roll(states, -1, axis=1).astype(np.uint8) << 2
    codes |= np.roll(np.roll(states, -1, axis=1), -1, axis=2).astype(np.uint8) << 3
    counts = np.zeros((states.shape[0], 15), dtype=np.float64)
    for code in range(1, 16):
        counts[:, code - 1] = np.count_nonzero(codes == code, axis=(1, 2))
    return counts


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _mass_support(vector: np.ndarray, target_mass: float) -> int:
    total = float(np.maximum(vector, 0.0).sum())
    if total <= 0.0:
        return 0
    order = sorted(range(len(vector)), key=lambda index: (-float(vector[index]), index))
    cumulative = 0.0
    result = 0
    for index in order:
        if vector[index] <= 0.0:
            continue
        result |= 1 << index
        cumulative += float(vector[index])
        if cumulative >= target_mass * total:
            break
    return result


def _trajectory_seed(rule: int, launch_index: int, contract: LifeFamilyContract) -> int:
    payload = ":".join(
        (
            contract.rng_tag,
            f"{contract.width}x{contract.height}",
            str(contract.activity_budget),
            str(contract.min_sweeps),
            str(contract.max_sweeps),
            f"{contract.flip_noise:.12g}",
            f"{contract.copy_error:.12g}",
            contract.launch_tag,
            str(rule),
            str(launch_index),
        )
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")


def _update_observer(
    compositions: np.ndarray,
    lengths: np.ndarray,
    first_break: np.ndarray,
    broke_by_8: np.ndarray,
    strict_positive: np.ndarray,
    future: int,
    composition: np.ndarray,
    contract: LifeFamilyContract,
) -> None:
    position = int(lengths[future])
    compositions[future, position] = composition
    lengths[future] += 1
    length = position + 1
    if length < 2:
        return
    thresholds = contract.thresholds
    boundary = length - 2
    similarity = _cosine(compositions[future, boundary], compositions[future, boundary + 1])
    if first_break[future] < 0 and similarity <= thresholds.inherit:
        first_break[future] = boundary
        if boundary < thresholds.break_horizon - 1:
            broke_by_8[future] = True
    break_index = int(first_break[future])
    if break_index < 0 or length < thresholds.strict_run + 2:
        return
    start = length - thresholds.strict_run - 1
    if start < break_index + 1:
        return
    if not all(
        _cosine(compositions[future, index], compositions[future, index + 1]) > thresholds.inherit
        for index in range(start, start + thresholds.strict_run)
    ):
        return
    daughters = compositions[future, start + 1 : start + thresholds.strict_run + 1]
    for left in range(thresholds.strict_run):
        for right in range(left):
            if _cosine(daughters[left], daughters[right]) <= thresholds.coherence:
                return
    anchor = compositions[future, break_index + 1]
    if any(_cosine(daughter, anchor) > thresholds.distinct for daughter in daughters):
        return
    strict_positive[future] = True


def _simulate_launch(
    rule: int,
    launch_index: int,
    initial: np.ndarray,
    contract: LifeFamilyContract,
    *,
    capture: bool = False,
) -> dict[str, Any]:
    n = contract.futures_per_launch
    states = np.repeat(initial[None, :, :], n, axis=0)
    alive = np.ones(n, dtype=np.bool_)
    strict_positive = np.zeros(n, dtype=np.bool_)
    broke_by_8 = np.zeros(n, dtype=np.bool_)
    first_break = np.full(n, -1, dtype=np.int16)
    lengths = np.zeros(n, dtype=np.int16)
    compositions = np.zeros((n, contract.horizon, 15), dtype=np.float64)
    first_generation_times = np.zeros(n, dtype=np.int16)
    total_sweeps = 0
    deaths: Counter[str] = Counter()
    captures: list[list[dict[str, Any]]] | None = ([[] for _ in range(n)] if capture else None)
    rng = np.random.default_rng(_trajectory_seed(rule, launch_index, contract))
    birth_lookup, survive_lookup = _rule_lookups(rule)

    for generation in range(contract.horizon):
        batch = np.flatnonzero(alive & ~strict_positive)
        if not len(batch):
            break
        current = states[batch].copy()
        batch_size = len(batch)
        activity = np.zeros(batch_size, dtype=np.int32)
        sweeps = np.zeros(batch_size, dtype=np.int16)
        reached_budget = np.zeros(batch_size, dtype=np.bool_)
        accumulated = np.zeros((batch_size, 15), dtype=np.float64)

        for sweep in range(1, contract.max_sweeps + 1):
            active = np.flatnonzero(~reached_budget)
            if not len(active):
                break
            previous = current[active]
            terminal = _life_like_step_lookup(previous, birth_lookup, survive_lookup)
            if contract.flip_noise > 0.0:
                terminal ^= rng.random(terminal.shape) < contract.flip_noise
            activity[active] += np.count_nonzero(terminal != previous, axis=(1, 2))
            current[active] = terminal
            accumulated[active] += live_2x2_counts_batch(terminal)
            sweeps[active] = sweep
            if sweep >= contract.min_sweeps:
                reached_budget[active[activity[active] >= contract.activity_budget]] = True

        total_sweeps += int(sweeps.sum())
        if generation == 0:
            first_generation_times[:] = sweeps
        empty = ~current.any(axis=(1, 2))
        timed_out = ~reached_budget
        dead = timed_out | empty
        deaths["timeout"] += int(np.count_nonzero(timed_out & ~empty))
        deaths["empty"] += int(np.count_nonzero(empty & ~timed_out))
        deaths["timeout_and_empty"] += int(np.count_nonzero(timed_out & empty))

        # Mandatory for every entrant, including deaths and new strict events.
        copy_masks = rng.random(current.shape) < contract.copy_error
        offspring = current ^ copy_masks
        totals = accumulated.sum(axis=1)
        divided = np.flatnonzero(~dead & (totals > 0.0))
        for local in divided:
            future = int(batch[local])
            composition = accumulated[local] / totals[local]
            _update_observer(
                compositions,
                lengths,
                first_break,
                broke_by_8,
                strict_positive,
                future,
                composition,
                contract,
            )
            states[future] = offspring[local]
            if captures is not None:
                captures[future].append(
                    {
                        "generation": generation,
                        "terminal_board_hex": _array_to_hex(current[local]),
                        "offspring_board_hex": _array_to_hex(offspring[local]),
                        "composition": composition.astype(float).tolist(),
                        "sweeps": int(sweeps[local]),
                        "activity": int(activity[local]),
                        "strict_after_generation": bool(strict_positive[future]),
                        "first_break": (
                            int(first_break[future]) if first_break[future] >= 0 else None
                        ),
                    }
                )
        alive[batch[dead | (totals <= 0.0)]] = False

    qualifying = [
        compositions[future, int(lengths[future]) - 1]
        for future in range(n)
        if first_break[future] >= 0 and lengths[future] >= first_break[future] + 2
    ]
    support = (
        _mass_support(np.mean(qualifying, axis=0), contract.form_mass_quantile)
        if qualifying
        else 0
    )
    result = {
        "launch_index": launch_index,
        "strict_count": int(strict_positive.sum()),
        "break_by_8_count": int(broke_by_8.sum()),
        "survival_sum": int(lengths.sum()),
        "first_generation_times": first_generation_times.astype(int).tolist(),
        "form_support": int(support),
        "form_n_futures": len(qualifying),
        "total_sweeps": total_sweeps,
        "death_counts": dict(sorted(deaths.items())),
    }
    if captures is not None:
        result["captures"] = captures
    return result


def capture_life_family_launch(
    rule: int,
    launch_index: int,
    initial: np.ndarray,
    contract: LifeFamilyContract = LifeFamilyContract(),
) -> dict[str, Any]:
    """Run a frozen Life-family launch and retain generation-level states."""

    return _simulate_launch(rule, launch_index, initial, contract, capture=True)


def evaluate_life_family_rule(
    rule: int,
    contract: LifeFamilyContract = LifeFamilyContract(),
) -> LifeFamilyRuleResult:
    require_pinned_numpy()
    if not 0 <= rule < (1 << 17):
        raise ValueError("a Life-like rule must be a 17-bit integer")
    launches = tuple(
        _simulate_launch(rule, index, board, contract)
        for index, board in enumerate(launch_library(contract))
    )
    n_futures = len(launches) * contract.futures_per_launch
    death_counts: Counter[str] = Counter()
    for launch in launches:
        death_counts.update(launch["death_counts"])
    first_times = [int(value) for launch in launches for value in launch["first_generation_times"]]
    return LifeFamilyRuleResult(
        rule=rule,
        notation=life_rule_notation(rule),
        strict=sum(int(launch["strict_count"]) for launch in launches) / n_futures,
        break_by_8=sum(int(launch["break_by_8_count"]) for launch in launches) / n_futures,
        median_gen_sweeps=float(median(first_times)),
        mean_survival=sum(int(launch["survival_sum"]) for launch in launches) / n_futures,
        form_supports=tuple(int(launch["form_support"]) for launch in launches),
        n_futures=n_futures,
        total_sweeps=sum(int(launch["total_sweeps"]) for launch in launches),
        death_counts=dict(sorted(death_counts.items())),
        per_launch=launches,
    )


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def family_edges(rules: Sequence[int]) -> tuple[tuple[int, int], ...]:
    available = set(rules)
    edges: list[tuple[int, int]] = []
    for rule in sorted(available):
        for bit in range(17):
            neighbor = rule ^ (1 << bit)
            if neighbor in available and rule < neighbor:
                edges.append((rule, neighbor))
    return tuple(edges)


def adjudicate_life_family(
    results: Sequence[LifeFamilyRuleResult],
    *,
    null_tag: str,
) -> dict[str, Any]:
    by_rule = {result.rule: result for result in results}
    rules = sorted(by_rule)
    edges = family_edges(rules)
    edge_mean = mean(_jaccard(by_rule[a].library, by_rule[b].library) for a, b in edges) if edges else 0.0
    null_rng = stream(null_tag, "life-family-smoothness-null")
    random_pairs: list[tuple[int, int]] = []
    while len(random_pairs) < len(edges):
        left, right = null_rng.sample(rules, 2)
        random_pairs.append((left, right))
    random_mean = mean(_jaccard(by_rule[a].library, by_rule[b].library) for a, b in random_pairs) if random_pairs else 0.0
    smoothness = edge_mean / random_mean if random_mean else None

    census = Counter(form for result in results for form in result.library)
    n_top = max(1, math.ceil(len(census) * 0.10)) if census else 0
    total_mass = sum(census.values())
    heavy_tail = sum(sorted(census.values(), reverse=True)[:n_top]) / total_mass if total_mass else 0.0

    life_rule = parse_life_rule(NAMED_RULE_NOTATIONS["life"])
    life = by_rule.get(life_rule)
    life_rank = 1 + sum(len(result.library) > len(life.library) for result in results) if life else None
    life_top_decile = life_rank is not None and life_rank <= math.ceil(len(results) * 0.10)
    life_in_band = life is not None and 0.005 <= life.strict <= 0.5
    capable = [result for result in results if 0.005 <= result.strict <= 0.5]
    family_clocks = [result.median_gen_sweeps for result in results]
    q1 = quantile(family_clocks, 0.25)
    q3 = quantile(family_clocks, 0.75)
    capable_clock = median(result.median_gen_sweeps for result in capable) if capable else None
    clock_gate = capable_clock is not None and q1 < capable_clock < q3

    named: list[dict[str, Any]] = []
    for name, notation in NAMED_RULE_NOTATIONS.items():
        result = by_rule.get(parse_life_rule(notation))
        if result is None:
            continue
        rank = 1 + sum(len(item.library) > len(result.library) for item in results)
        named.append(
            {
                "name": name,
                "rule": result.rule,
                "notation": notation,
                "strict": result.strict,
                "break_by_8": result.break_by_8,
                "library_size": len(result.library),
                "library_rank": rank,
            }
        )
    return {
        "gate_smoothness": (smoothness >= 2.0) if smoothness is not None else edge_mean > 0.0,
        "smoothness_ratio": smoothness,
        "edge_mean_jaccard": edge_mean,
        "random_mean_jaccard": random_mean,
        "n_edges_in_sample": len(edges),
        "gate_heavy_tail": heavy_tail >= 0.35,
        "heavy_tail_share": heavy_tail,
        "n_forms": len(census),
        "gate_life_rare_book": bool(life_top_decile and life_in_band),
        "life_top_decile": life_top_decile,
        "life_in_band": life_in_band,
        "life_library_rank": life_rank,
        "life_strict": life.strict if life else None,
        "gate_boundary_of_order": clock_gate,
        "family_clock_iqr": [q1, q3],
        "family_clock_iqr_width": q3 - q1,
        "capable_median_clock": capable_clock,
        "n_capable": len(capable),
        "named_rules": named,
    }


def _checkpoint_task(arguments: tuple[int, LifeFamilyContract]) -> dict[str, Any]:
    rule, contract = arguments
    return evaluate_life_family_rule(rule, contract).to_dict()


def run_life_family_rule_set(
    contract: LifeFamilyContract,
    checkpoint_root: Path,
    rules: Sequence[int],
    *,
    workers: int,
    resume: bool,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> list[LifeFamilyRuleResult]:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results: dict[int, LifeFamilyRuleResult] = {}
    missing: list[int] = []
    for rule in rules:
        path = checkpoint_root / f"rule-{rule:06d}.json"
        if resume and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("contract_digest") == contract.digest:
                results[rule] = LifeFamilyRuleResult.from_dict(payload["result"])
                continue
        missing.append(rule)
    completed = len(results)
    started = time.time()

    def save(result: LifeFamilyRuleResult) -> None:
        nonlocal completed
        _atomic_json(
            checkpoint_root / f"rule-{result.rule:06d}.json",
            {"contract_digest": contract.digest, "result": result.to_dict()},
        )
        results[result.rule] = result
        completed += 1
        if progress:
            progress(completed, len(rules), result.rule, int(time.time() - started))

    if workers <= 1:
        for rule in missing:
            save(evaluate_life_family_rule(rule, contract))
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_checkpoint_task, (rule, contract)): rule for rule in missing}
            for future in as_completed(futures):
                save(LifeFamilyRuleResult.from_dict(future.result()))
    return [results[rule] for rule in sorted(rules)]


def _write_family_csv(path: Path, results: Sequence[LifeFamilyRuleResult]) -> None:
    fields = (
        "rule",
        "notation",
        "strict",
        "break_by_8",
        "median_gen_sweeps",
        "mean_survival",
        "library_size",
        "in_band",
        "support_masks",
        "n_futures",
        "total_sweeps",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "rule": result.rule,
                    "notation": result.notation,
                    "strict": result.strict,
                    "break_by_8": result.break_by_8,
                    "median_gen_sweeps": result.median_gen_sweeps,
                    "mean_survival": result.mean_survival,
                    "library_size": len(result.library),
                    "in_band": 0.005 <= result.strict <= 0.5,
                    "support_masks": "|".join(str(value) for value in sorted(result.library)),
                    "n_futures": result.n_futures,
                    "total_sweeps": result.total_sweeps,
                }
            )
    os.replace(temporary, path)


def run_life_family_condition(
    contract: LifeFamilyContract,
    output: Path,
    *,
    rules: Sequence[int] | None = None,
    workers: int = 1,
    resume: bool = False,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> dict[str, Any]:
    started = time.time()
    selected = tuple(load_rule_registry() if rules is None else rules)
    results = run_life_family_rule_set(
        contract,
        output / "checkpoints",
        selected,
        workers=workers,
        resume=resume,
        progress=progress,
    )
    _write_family_csv(output / "family.csv", results)
    gates = adjudicate_life_family(results, null_tag=contract.rng_tag)
    summary = {
        "experiment": "life_family_cleanroom",
        "elapsed_seconds": time.time() - started,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation": "vectorized shrinking Life batches",
        },
        "contract": contract.to_dict(),
        "contract_digest": contract.digest,
        "launches": launch_manifest(contract),
        "n_rules": len(results),
        "n_lineages": sum(result.n_futures for result in results),
        "death_counts": {
            key: sum(result.death_counts.get(key, 0) for result in results)
            for key in sorted({key for result in results for key in result.death_counts})
        },
        "gates": gates,
    }
    _atomic_json(output / "family_summary.json", summary)
    _atomic_text(output / "COMPLETE", "complete\n")
    return summary


def compare_life_family(ours_path: Path, reference_path: Path) -> dict[str, Any]:
    def read(path: Path) -> dict[int, dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return {int(row["rule"]): row for row in csv.DictReader(handle)}

    ours = read(ours_path)
    reference = read(reference_path)
    common = sorted(set(ours) & set(reference))
    fields = ("strict", "break_by_8", "median_gen_sweeps", "library_size")
    correlations: dict[str, float] = {}
    errors: dict[str, float] = {}
    exact: dict[str, int] = {}
    for field_name in fields:
        left = [float(ours[rule][field_name]) for rule in common]
        right = [float(reference[rule][field_name]) for rule in common]
        correlations[field_name] = spearman(left, right)
        errors[field_name] = mean(abs(a - b) for a, b in zip(left, right)) if left else float("nan")
        exact[field_name] = sum(a == b for a, b in zip(left, right))
    exact_libraries = sum(
        ours[rule].get("support_masks", "") == reference[rule].get("support_masks", "")
        for rule in common
    ) if common and "support_masks" in next(iter(reference.values())) else 0
    thresholds = {"strict": 0.70, "break_by_8": 0.90, "median_gen_sweeps": 0.90, "library_size": 0.70}
    return {
        "n_common": len(common),
        "spearman": correlations,
        "mean_absolute_error": errors,
        "field_exact_matches": exact,
        "library_exact_matches": exact_libraries,
        "strong_statistical_thresholds": thresholds,
        "strong_statistical": len(common) == 1024 and all(
            correlations[field_name] >= threshold for field_name, threshold in thresholds.items()
        ),
    }


def fixed_scale_subset(rules: Sequence[int], count: int = 128) -> tuple[int, ...]:
    available = set(rules)
    selected: list[int] = []
    selected_set: set[int] = set()

    def add(rule: int) -> None:
        if rule in available and rule not in selected_set and len(selected) < count:
            selected.append(rule)
            selected_set.add(rule)

    for notation in NAMED_RULE_NOTATIONS.values():
        add(parse_life_rule(notation))
    anchors = sorted(
        available - selected_set,
        key=lambda rule: hashlib.sha256(f"life-family-scale-subset-v1:{rule}".encode()).digest(),
    )
    for anchor in anchors:
        if len(selected) >= count:
            break
        add(anchor)
        for bit in range(17):
            add(anchor ^ (1 << bit))
    if len(selected) != count:
        raise RuntimeError(f"could construct only {len(selected)} of {count} subset rules")
    return tuple(sorted(selected))


def contract_for_condition(
    name: str,
    *,
    futures: int = 64,
) -> LifeFamilyContract:
    conditions: dict[str, dict[str, Any]] = {
        "frozen-b48": {"activity_budget": 48, "max_sweeps": 64},
        "budget-b256": {"activity_budget": 256, "max_sweeps": 64},
        "area-b1024": {"activity_budget": 1024, "max_sweeps": 64},
        "horizon-b1024-t128": {"activity_budget": 1024, "max_sweeps": 128},
        "horizon-b1024-t256": {"activity_budget": 1024, "max_sweeps": 256},
        "scale-16": {"activity_budget": 1024, "max_sweeps": 256},
        "scale-32": {"width": 32, "height": 32, "activity_budget": 4096, "max_sweeps": 256},
        "launch-v2-b48": {"activity_budget": 48, "max_sweeps": 64, "launch_tag": "life-family-launch-v2"},
        "launch-v2-b1024": {"activity_budget": 1024, "max_sweeps": 64, "launch_tag": "life-family-launch-v2"},
        "launch-broad-b48": {
            "activity_budget": 48,
            "max_sweeps": 64,
            "launch_tag": "life-family-launch-broad-v1",
            "launch_densities": (0.20, 0.40, 0.60, 0.80),
        },
        "launch-broad-b1024": {
            "activity_budget": 1024,
            "max_sweeps": 64,
            "launch_tag": "life-family-launch-broad-v1",
            "launch_densities": (0.20, 0.40, 0.60, 0.80),
        },
    }
    if name not in conditions:
        raise ValueError(f"unknown Life-family condition {name!r}")
    values = dict(conditions[name])
    values["futures_per_launch"] = futures
    values["rng_tag"] = f"life-family-trajectory-v1:{name}"
    return LifeFamilyContract(**values)
