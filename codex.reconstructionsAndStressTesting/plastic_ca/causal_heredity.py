"""Causal, common-garden tests of Plastic Heredity in cellular automata.

This module is a clean-room follow-up to the frozen E19/E24 engines.  It uses
their disclosed execution conventions but a new RNG namespace, and treats
the transmitted lattice state as an experimental variable rather than merely
scoring correlations between lineage compositions.
"""

from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .config import ObserverThresholds
from .e19 import (
    E19Contract,
    LAUNCH_HEX,
    _hex_to_row,
    _row_to_hex,
    e19_step,
    figure_states,
    final4_counts,
    require_pinned_numpy,
)
from .life_family import (
    LifeFamilyContract,
    NAMED_RULE_NOTATIONS,
    _array_to_hex,
    _board_to_array,
    _rule_lookups,
    _life_like_step_lookup,
    launch_library,
    life_rule_notation,
    live_2x2_counts_batch,
    parse_life_rule,
)
from .particle_e19 import build_e19_domain_dictionary
from .stats import quantile


ECA_PANEL = (
    0,
    8,
    11,
    18,
    22,
    30,
    35,
    41,
    43,
    45,
    54,
    57,
    90,
    106,
    110,
    122,
    126,
    146,
    150,
    184,
)
ECA_RAW_RULES = frozenset({0, 8, 11, 35, 43, 57, 184})
ECA_MAINTAINER_CONTROLS = frozenset({0, 8, 90, 150})
PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "CAUSAL_HEREDITY_PROTOCOL.md"


@dataclass(frozen=True)
class CausalContract:
    """Complete frozen semantics for causal interventions."""

    implementation_version: str = "causal-heredity-cleanroom-v1"
    namespace: str = "plastic-ca-causal-heredity-v1"
    recovery_horizon: int = 16
    memory_precondition_horizon: int = 4
    memory_garden_horizon: int = 8
    eca_width: int = 64
    eca_activity_budget: int = 256
    eca_min_sweeps: int = 4
    eca_max_sweeps: int = 128
    eca_process_noise: float = 0.01
    eca_copy_error: float = 0.015
    life_width: int = 16
    life_height: int = 16
    life_activity_budget: int = 48
    life_min_sweeps: int = 4
    life_max_sweeps: int = 64
    life_process_noise: float = 0.002
    life_copy_error: float = 0.005
    thresholds: ObserverThresholds = ObserverThresholds()
    transmission_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    noise_multipliers: tuple[int, ...] = (0, 1, 2)
    pedigree_depth: int = 5

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "eca_semantics": E19Contract().to_dict(),
                "life_semantics": LifeFamilyContract().to_dict(),
                "intervention_copy_timing": "initial daughter and after every completed generation",
                "common_garden_rng": "independent condition stream",
                "success": (
                    "last eight valid compositions each target cosine >0.90, pairwise >0.90; "
                    "switchers also <=0.85 to historical break-causing-daughter anchor"
                ),
            }
        )
        return value

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CausalProfile:
    panel_limit: int | None
    donor_target: int
    discovery_cap: int
    discovery_batch: int
    recovery_replicates: int
    pedigree_donors: int
    pedigree_depth: int
    transplant_donors: int
    transplant_replicates: int
    bootstrap_resamples: int


CAUSAL_PROFILES: dict[str, CausalProfile] = {
    "smoke": CausalProfile(3, 1, 32, 16, 1, 1, 2, 1, 1, 100),
    "pilot": CausalProfile(8, 4, 1024, 128, 4, 2, 3, 2, 4, 1_000),
    "reference": CausalProfile(None, 16, 16_384, 256, 16, 8, 5, 4, 8, 10_000),
}


@dataclass
class BatchTrace:
    compositions: np.ndarray
    valid: np.ndarray
    terminals: np.ndarray
    offspring: np.ndarray
    sweeps: np.ndarray
    activity: np.ndarray
    death: list[str | None]


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _hash_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=-1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values, dtype=np.float64), where=totals > 0)


def _cosine(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _state_to_hex(substrate: str, state: np.ndarray) -> str:
    return _row_to_hex(state) if substrate == "eca" else _array_to_hex(state)


def _state_from_hex(substrate: str, value: str) -> np.ndarray:
    return _hex_to_row(value) if substrate == "eca" else _board_to_array(int(value, 16), 16, 16)


def _cyclic_k_counts(states: np.ndarray, k: int) -> np.ndarray:
    if states.ndim != 2:
        raise ValueError("ECA states must have shape (future, width)")
    codes = np.zeros(states.shape, dtype=np.uint8)
    for bit in range(k):
        codes |= np.roll(states, -bit, axis=1).astype(np.uint8) << (k - bit - 1)
    counts = np.zeros((states.shape[0], 1 << k), dtype=np.float64)
    for code in range(1 << k):
        counts[:, code] = np.count_nonzero(codes == code, axis=1)
    return counts


def _eca_multiscale(states: np.ndarray) -> np.ndarray:
    chunks = [_normalize_rows(_cyclic_k_counts(states, k)) for k in range(2, 7)]
    return np.concatenate(chunks, axis=1) / math.sqrt(len(chunks))


def _component_spectrum(board: np.ndarray) -> np.ndarray:
    """Eight-bin Moore-connected component census on a torus."""

    height, width = board.shape
    unseen = set(map(tuple, np.argwhere(board)))
    bins = np.zeros(8, dtype=np.float64)
    while unseen:
        root = unseen.pop()
        queue: deque[tuple[int, int]] = deque((root,))
        size = 0
        while queue:
            y, x = queue.popleft()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    candidate = ((y + dy) % height, (x + dx) % width)
                    if candidate in unseen:
                        unseen.remove(candidate)
                        queue.append(candidate)
        index = 0 if size == 1 else 1 if size == 2 else 2 if size <= 4 else 3 if size <= 8 else 4 if size <= 16 else 5 if size <= 32 else 6 if size <= 64 else 7
        bins[index] += 1.0
    total = bins.sum()
    return bins / total if total else bins


def _observer_vectors(substrate: str, states: np.ndarray) -> dict[str, np.ndarray]:
    if substrate == "eca":
        return {
            "raw4": _normalize_rows(final4_counts(states)),
            "multiscale": _eca_multiscale(states),
        }
    return {
        "terminal2x2": _normalize_rows(live_2x2_counts_batch(states)),
        "components": np.stack([_component_spectrum(board) for board in states]),
    }


def _simulate_batch(
    substrate: str,
    rule: int,
    initial_states: np.ndarray,
    contract: CausalContract,
    *,
    horizon: int,
    rng_seed: int,
    observer: str,
    process_noise: float | None = None,
    copy_error: float | None = None,
    domain_codes: frozenset[int] | None = None,
) -> BatchTrace:
    """Run a fresh batch with the frozen shrinking-batch lifecycle."""

    states = np.asarray(initial_states, dtype=np.bool_).copy()
    n = len(states)
    if substrate == "eca":
        width = contract.eca_width
        budget = contract.eca_activity_budget
        minimum = contract.eca_min_sweeps
        maximum = contract.eca_max_sweeps
        eta = contract.eca_process_noise if process_noise is None else process_noise
        epsilon = contract.eca_copy_error if copy_error is None else copy_error
        state_shape = (width,)
        dimension = 16
    elif substrate == "life":
        budget = contract.life_activity_budget
        minimum = contract.life_min_sweeps
        maximum = contract.life_max_sweeps
        eta = contract.life_process_noise if process_noise is None else process_noise
        epsilon = contract.life_copy_error if copy_error is None else copy_error
        state_shape = (contract.life_height, contract.life_width)
        dimension = 15
    else:
        raise ValueError(f"unknown substrate {substrate!r}")

    if tuple(states.shape[1:]) != state_shape:
        raise ValueError(f"wrong {substrate} state shape {states.shape}")
    if observer == "particle" and (substrate != "eca" or domain_codes is None):
        raise ValueError("the particle observer requires ECA domain codes")

    rng = np.random.default_rng(rng_seed)
    alive = np.ones(n, dtype=np.bool_)
    valid = np.zeros((n, horizon), dtype=np.bool_)
    compositions = np.zeros((n, horizon, dimension), dtype=np.float64)
    terminals = np.zeros((n, horizon) + state_shape, dtype=np.bool_)
    offspring_history = np.zeros_like(terminals)
    sweeps_history = np.zeros((n, horizon), dtype=np.int16)
    activity_history = np.zeros((n, horizon), dtype=np.int32)
    deaths: list[str | None] = [None] * n
    birth_lookup = survive_lookup = None
    if substrate == "life":
        birth_lookup, survive_lookup = _rule_lookups(rule)

    for generation in range(horizon):
        batch = np.flatnonzero(alive)
        if not len(batch):
            break
        current = states[batch].copy()
        batch_size = len(batch)
        activity = np.zeros(batch_size, dtype=np.int32)
        sweep_counts = np.zeros(batch_size, dtype=np.int16)
        reached = np.zeros(batch_size, dtype=np.bool_)
        if substrate == "eca":
            history = [current.copy(), current.copy(), current.copy()]
        else:
            accumulated = np.zeros((batch_size, 15), dtype=np.float64)

        for sweep in range(1, maximum + 1):
            active = np.flatnonzero(~reached)
            if not len(active):
                break
            previous = current[active]
            if substrate == "eca":
                terminal = e19_step(previous, rule)
            else:
                assert birth_lookup is not None and survive_lookup is not None
                terminal = _life_like_step_lookup(previous, birth_lookup, survive_lookup)
            if eta > 0.0:
                terminal ^= rng.random(terminal.shape) < eta
            axes = tuple(range(1, terminal.ndim))
            activity[active] += np.count_nonzero(terminal != previous, axis=axes)
            current[active] = terminal
            if substrate == "eca":
                history[0][active] = history[1][active]
                history[1][active] = history[2][active]
                history[2][active] = terminal
            else:
                accumulated[active] += live_2x2_counts_batch(terminal)
            sweep_counts[active] = sweep
            if sweep >= minimum:
                reached[active[activity[active] >= budget]] = True

        if substrate == "eca":
            terminal_dead = (~current.any(axis=1)) | current.all(axis=1)
        else:
            terminal_dead = ~current.any(axis=(1, 2))
        timed_out = ~reached
        dead = timed_out | terminal_dead
        copy_masks = rng.random(current.shape) < epsilon
        offspring = current ^ copy_masks

        if substrate == "eca":
            if observer == "particle":
                observed = figure_states((history[0], history[1], history[2]), domain_codes or frozenset())
                observer_empty = ~observed.any(axis=1)
            else:
                observed = current
                observer_empty = np.zeros(batch_size, dtype=np.bool_)
            counts = final4_counts(observed)
            totals = counts.sum(axis=1)
            stopped = dead | observer_empty | (totals <= 0)
            normalized = _normalize_rows(counts)
        else:
            totals = accumulated.sum(axis=1)
            stopped = dead | (totals <= 0)
            normalized = _normalize_rows(accumulated)

        for local, future_value in enumerate(batch):
            future = int(future_value)
            terminals[(future, generation)] = current[local]
            offspring_history[(future, generation)] = offspring[local]
            sweeps_history[future, generation] = sweep_counts[local]
            activity_history[future, generation] = activity[local]
            if stopped[local]:
                if timed_out[local] and terminal_dead[local]:
                    deaths[future] = "timeout_and_terminal"
                elif timed_out[local]:
                    deaths[future] = "timeout"
                elif terminal_dead[local]:
                    deaths[future] = "terminal"
                else:
                    deaths[future] = "observer_empty"
                alive[future] = False
                continue
            valid[future, generation] = True
            compositions[future, generation] = normalized[local]
            states[future] = offspring[local]

    return BatchTrace(compositions, valid, terminals, offspring_history, sweeps_history, activity_history, deaths)


def _strict_event(compositions: np.ndarray, thresholds: ObserverThresholds) -> tuple[int, int] | None:
    """Return (first break, coherent run start) using E19's historical anchor."""

    if len(compositions) < thresholds.strict_run + 2:
        return None
    similarities = [_cosine(left, right) for left, right in zip(compositions, compositions[1:])]
    first_break = next((i for i, value in enumerate(similarities) if value <= thresholds.inherit), None)
    if first_break is None:
        return None
    for start in range(first_break + 1, len(similarities) - thresholds.strict_run + 1):
        if not all(value > thresholds.inherit for value in similarities[start : start + thresholds.strict_run]):
            continue
        daughters = compositions[start + 1 : start + thresholds.strict_run + 1]
        if any(
            _cosine(daughters[left], daughters[right]) <= thresholds.coherence
            for left in range(len(daughters))
            for right in range(left)
        ):
            continue
        anchor = compositions[first_break + 1]
        if any(_cosine(daughter, anchor) > thresholds.distinct for daughter in daughters):
            continue
        return first_break, start
    return None


def _is_maintainer(compositions: np.ndarray, thresholds: ObserverThresholds) -> bool:
    if len(compositions) < 17:
        return False
    return all(
        _cosine(compositions[index], compositions[index + 1]) > thresholds.inherit
        for index in range(16)
    )


def _centroid(vectors: Sequence[Sequence[float]] | np.ndarray) -> list[float]:
    array = np.asarray(vectors, dtype=np.float64)
    return np.mean(array, axis=0).astype(float).tolist()


def _donor_record(
    substrate: str,
    rule: int,
    observer: str,
    launch_index: int,
    trial_index: int,
    initial: np.ndarray,
    trace: BatchTrace,
    local: int,
    kind: str,
    event: tuple[int, int] | None,
) -> dict[str, Any]:
    length = int(np.count_nonzero(trace.valid[local]))
    compositions = trace.compositions[local, :length]
    if kind == "switcher":
        assert event is not None
        first_break, run_start = event
        established_indices = list(range(run_start + 1, run_start + 9))
        anchor_index = first_break + 1
        ancestor_index = first_break
    else:
        first_break = run_start = None
        established_indices = list(range(length - 8, length))
        anchor_index = 0
        ancestor_index = 0
    donor_index = established_indices[-1]
    established_states = trace.terminals[local, established_indices]
    observer_vectors = _observer_vectors(substrate, established_states)
    anchor_vectors = _observer_vectors(substrate, trace.terminals[local, anchor_index][None, ...])
    targets = {name: _centroid(values) for name, values in observer_vectors.items()}
    anchors = {name: values[0].astype(float).tolist() for name, values in anchor_vectors.items()}
    targets["primary"] = _centroid(compositions[established_indices])
    anchors["primary"] = compositions[anchor_index].astype(float).tolist()
    return {
        "donor_id": f"{substrate}-{rule}-{launch_index}-{trial_index}",
        "substrate": substrate,
        "rule": rule,
        "notation": life_rule_notation(rule) if substrate == "life" else None,
        "observer": observer,
        "kind": kind,
        "launch_index": launch_index,
        "trial_index": trial_index,
        "first_break": first_break,
        "run_start": run_start,
        "valid_generations": length,
        "initial_state_hex": _state_to_hex(substrate, initial),
        "ancestor_state_hex": _state_to_hex(substrate, trace.terminals[local, ancestor_index]),
        "anchor_state_hex": _state_to_hex(substrate, trace.terminals[local, anchor_index]),
        "donor_state_hex": _state_to_hex(substrate, trace.terminals[local, donor_index]),
        "offspring_state_hex": _state_to_hex(substrate, trace.offspring[local, donor_index]),
        "target_generation_indices": established_indices,
        "target_compositions": targets,
        "anchor_compositions": anchors,
        "generation_sweeps": trace.sweeps[local, :length].astype(int).tolist(),
        "generation_activity": trace.activity[local, :length].astype(int).tolist(),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_rule_panel(life_atlas: Path, profile: CausalProfile) -> dict[str, Any]:
    rows = _read_csv(life_atlas)
    by_rule = {int(row["rule"]): row for row in rows}
    selected: list[int] = []

    def add(rule: int) -> None:
        if rule in by_rule and rule not in selected:
            selected.append(rule)

    for notation in NAMED_RULE_NOTATIONS.values():
        add(parse_life_rule(notation))
    in_band = sorted(
        (row for row in rows if 0.005 <= float(row["strict"]) <= 0.5),
        key=lambda row: (-float(row["strict"]), int(row["rule"])),
    )
    strict_selections: list[dict[str, str]] = []
    for row in in_band:
        before = len(selected)
        add(int(row["rule"]))
        if len(selected) > before:
            strict_selections.append(row)
        if len(strict_selections) == 8:
            break
    libraries = sorted(
        rows,
        key=lambda row: (-int(row["library_size"]), -float(row["strict"]), int(row["rule"])),
    )
    added_libraries = 0
    for row in libraries:
        before = len(selected)
        add(int(row["rule"]))
        if len(selected) > before:
            added_libraries += 1
        if added_libraries == 4:
            break

    controls = [row for row in rows if float(row["strict"]) == 0.0 and int(row["rule"]) not in selected]
    fields = ("break_by_8", "median_gen_sweeps", "mean_survival")
    scales = {
        field: max(float(np.std([float(row[field]) for row in rows])), 1e-12)
        for field in fields
    }
    for target in strict_selections[:4]:
        ranked = sorted(
            controls,
            key=lambda row: (
                sum(abs(float(row[field]) - float(target[field])) / scales[field] for field in fields),
                int(row["rule"]),
            ),
        )
        if ranked:
            chosen = ranked[0]
            add(int(chosen["rule"]))
            controls.remove(chosen)

    life_entries = [
        {
            "substrate": "life",
            "rule": rule,
            "notation": by_rule[rule]["notation"],
            "observer": "primary",
            "development_strict": float(by_rule[rule]["strict"]),
            "desired_kind": "switcher" if float(by_rule[rule]["strict"]) >= 0.005 else "maintainer",
        }
        for rule in selected
    ]
    eca_entries = [
        {
            "substrate": "eca",
            "rule": rule,
            "notation": None,
            "observer": "raw" if rule in ECA_RAW_RULES else "particle",
            "development_strict": 0.0 if rule in ECA_MAINTAINER_CONTROLS else None,
            "desired_kind": "maintainer" if rule in ECA_MAINTAINER_CONTROLS else "switcher",
        }
        for rule in ECA_PANEL
    ]
    if profile.panel_limit is not None:
        # Keep a positive, observer-sensitive, and negative case in smoke;
        # pilot selection remains deterministic and outcome blind.
        smoke_eca = [next(row for row in eca_entries if row["rule"] == rule) for rule in (35, 110, 90)]
        named_life = [parse_life_rule("B3/S23"), parse_life_rule("B36/S23")]
        first_switcher = next(row["rule"] for row in life_entries if row["desired_kind"] == "switcher")
        smoke_life_ids = list(dict.fromkeys(named_life + [first_switcher]))
        if profile.panel_limit <= 3:
            eca_entries = smoke_eca
            life_entries = [next(row for row in life_entries if row["rule"] == rule) for rule in smoke_life_ids]
        else:
            eca_entries = eca_entries[: profile.panel_limit]
            life_entries = life_entries[: profile.panel_limit]
    return {
        "eca": eca_entries,
        "life": life_entries,
        "selection": {
            "life_named": 8,
            "life_top_in_band": 8,
            "life_top_remaining_libraries": 4,
            "life_matched_strict_zero": 4,
            "tie_break": "ascending rule id",
        },
    }


def _launches(substrate: str, contract: CausalContract) -> tuple[np.ndarray, ...]:
    if substrate == "eca":
        return tuple(_hex_to_row(value) for value in LAUNCH_HEX)
    return launch_library(
        LifeFamilyContract(
            width=contract.life_width,
            height=contract.life_height,
            activity_budget=contract.life_activity_budget,
            min_sweeps=contract.life_min_sweeps,
            max_sweeps=contract.life_max_sweeps,
            flip_noise=contract.life_process_noise,
            copy_error=contract.life_copy_error,
            futures_per_launch=1,
        )
    )


def _domain_for_entry(entry: dict[str, Any]) -> frozenset[int] | None:
    if entry["substrate"] == "eca" and entry["observer"] == "particle":
        return build_e19_domain_dictionary(int(entry["rule"])).codes
    return None


def _discover_donors(arguments: tuple[dict[str, Any], CausalContract, CausalProfile]) -> dict[str, Any]:
    entry, contract, profile = arguments
    substrate = str(entry["substrate"])
    rule = int(entry["rule"])
    observer = str(entry["observer"])
    desired = str(entry["desired_kind"])
    launches = _launches(substrate, contract)
    domain_codes = _domain_for_entry(entry)
    donors: list[dict[str, Any]] = []
    examined = 0
    strict_seen = 0
    maintainers_seen = 0
    batch_index = 0
    death_counts: dict[str, int] = defaultdict(int)

    while examined < profile.discovery_cap and len(donors) < profile.donor_target:
        size = min(profile.discovery_batch, profile.discovery_cap - examined)
        launch_indices = [(examined + local) % len(launches) for local in range(size)]
        initial = np.stack([launches[index] for index in launch_indices])
        trace = _simulate_batch(
            substrate,
            rule,
            initial,
            contract,
            horizon=contract.thresholds.horizon,
            rng_seed=_hash_seed(contract.namespace, "donor", substrate, rule, batch_index),
            observer=observer,
            domain_codes=domain_codes,
        )
        for reason in trace.death:
            if reason is not None:
                death_counts[reason] += 1
        for local in range(size):
            length = int(np.count_nonzero(trace.valid[local]))
            if not length:
                continue
            compositions = trace.compositions[local, :length]
            event = _strict_event(compositions, contract.thresholds)
            maintainer = _is_maintainer(compositions, contract.thresholds)
            strict_seen += int(event is not None)
            maintainers_seen += int(maintainer)
            qualifies = event is not None if desired == "switcher" else maintainer
            if not qualifies or len(donors) >= profile.donor_target:
                continue
            donors.append(
                _donor_record(
                    substrate,
                    rule,
                    observer,
                    launch_indices[local],
                    examined + local,
                    initial[local],
                    trace,
                    local,
                    desired,
                    event,
                )
            )
        examined += size
        batch_index += 1

    return {
        "entry": entry,
        "examined": examined,
        "target": profile.donor_target,
        "acquired": len(donors),
        "desired_kind": desired,
        "strict_seen": strict_seen,
        "maintainers_seen": maintainers_seen,
        "death_counts": dict(sorted(death_counts.items())),
        "domain_codes": sorted(domain_codes) if domain_codes is not None else None,
        "donors": donors,
    }


def _ranked_positions(count: int, key: str) -> list[int]:
    return sorted(
        range(count),
        key=lambda position: hashlib.sha256(f"{key}:{position}".encode()).digest(),
    )


def _wrapped_distance(value: int, origin: int, extent: int) -> int:
    delta = abs(value - origin)
    return min(delta, extent - delta)


def _site_mask(shape: tuple[int, ...], fraction: float, geometry: str, key: str) -> np.ndarray:
    cells = math.prod(shape)
    target = min(cells, max(0, int(round(cells * fraction))))
    mask = np.zeros(cells, dtype=np.bool_)
    if target == 0:
        return mask.reshape(shape)
    origin_seed = _hash_seed("mask-origin", key)
    if len(shape) == 1:
        width = shape[0]
        origin = origin_seed % width
        if geometry == "one_interval":
            ordered = [(origin + offset) % width for offset in range(width)]
        elif geometry == "two_interval":
            origins = (origin, (origin + width // 2) % width)
            ordered = sorted(
                range(width),
                key=lambda x: (min(_wrapped_distance(x, value, width) for value in origins), x),
            )
        elif geometry == "dispersed":
            ordered = _ranked_positions(width, key)
        else:
            raise ValueError(f"unknown ECA geometry {geometry!r}")
    else:
        height, width = shape
        oy = origin_seed % height
        ox = (origin_seed // height) % width
        if geometry == "square":
            ordered = sorted(
                range(cells),
                key=lambda index: (
                    max(
                        _wrapped_distance(index // width, oy, height),
                        _wrapped_distance(index % width, ox, width),
                    ),
                    _wrapped_distance(index // width, oy, height)
                    + _wrapped_distance(index % width, ox, width),
                    index,
                ),
            )
        elif geometry == "strip":
            vertical = bool(origin_seed & 1)
            ordered = sorted(
                range(cells),
                key=lambda index: (
                    _wrapped_distance(index % width, ox, width)
                    if vertical
                    else _wrapped_distance(index // width, oy, height),
                    index,
                ),
            )
        elif geometry == "two_lobe":
            origins = ((oy, ox), ((oy + height // 2) % height, (ox + width // 2) % width))
            ordered = sorted(
                range(cells),
                key=lambda index: (
                    min(
                        _wrapped_distance(index // width, y, height) ** 2
                        + _wrapped_distance(index % width, x, width) ** 2
                        for y, x in origins
                    ),
                    index,
                ),
            )
        elif geometry == "dispersed":
            ordered = _ranked_positions(cells, key)
        else:
            raise ValueError(f"unknown Life geometry {geometry!r}")
    mask[np.asarray(ordered[:target], dtype=int)] = True
    return mask.reshape(shape)


def _exact_random_state(shape: tuple[int, ...], live: int, key: str) -> np.ndarray:
    cells = math.prod(shape)
    result = np.zeros(cells, dtype=np.bool_)
    result[np.asarray(_ranked_positions(cells, key)[: max(0, min(cells, live))], dtype=int)] = True
    return result.reshape(shape)


def _intervention_state(
    source: np.ndarray,
    fraction: float,
    intervention: str,
    key: str,
) -> np.ndarray:
    shape = tuple(source.shape)
    if intervention == "intact":
        return source.copy()
    if intervention in {"one_interval", "two_interval", "square", "strip", "two_lobe", "dispersed"}:
        return source & _site_mask(shape, fraction, intervention, key)
    if intervention == "shuffled":
        primary_geometry = "one_interval" if len(shape) == 1 else "square"
        reference = source & _site_mask(shape, fraction, primary_geometry, key + ":reference")
        return _exact_random_state(shape, int(reference.sum()), key + ":shuffle")
    if intervention == "density_random":
        live = int(round(float(source.mean()) * math.prod(shape) * fraction))
        return _exact_random_state(shape, live, key + ":density")
    raise ValueError(f"unknown intervention {intervention!r}")


def _copy_batch(base: np.ndarray, count: int, error: float, seed: int) -> np.ndarray:
    batch = np.repeat(base[None, ...], count, axis=0)
    if error > 0.0:
        rng = np.random.default_rng(seed)
        batch ^= rng.random(batch.shape) < error
    return batch


def _observer_name_for_simulation(donor: dict[str, Any], override: str | None = None) -> str:
    if override is not None:
        return override
    return str(donor["observer"])


def _target_name(donor: dict[str, Any], simulation_observer: str) -> str:
    if donor["substrate"] == "eca" and simulation_observer == "raw":
        return "raw4"
    return "primary"


def _sequence_success(
    vectors: np.ndarray,
    target: Sequence[float],
    anchor: Sequence[float] | None,
    *,
    require_eight: bool = True,
) -> bool:
    if require_eight and len(vectors) < 8:
        return False
    selected = vectors[-8:] if require_eight else vectors[-1:]
    if any(_cosine(vector, target) <= 0.90 for vector in selected):
        return False
    if len(selected) > 1 and any(
        _cosine(selected[left], selected[right]) <= 0.90
        for left in range(len(selected))
        for right in range(left)
    ):
        return False
    if anchor is not None and any(_cosine(vector, anchor) > 0.85 for vector in selected):
        return False
    return True


def _recovery_outcomes(
    trace: BatchTrace,
    donor: dict[str, Any],
    *,
    simulation_observer: str,
) -> dict[str, list[bool]]:
    substrate = str(donor["substrate"])
    kind = str(donor["kind"])
    target_key = _target_name(donor, simulation_observer)
    primary: list[bool] = []
    auxiliary: dict[str, list[bool]] = defaultdict(list)
    for future in range(len(trace.valid)):
        indices = np.flatnonzero(trace.valid[future])
        if len(indices) < 8:
            primary.append(False)
            for name in ("raw4", "multiscale") if substrate == "eca" else ("terminal2x2", "components"):
                auxiliary[name].append(False)
            continue
        final_indices = indices[-8:]
        primary_vectors = trace.compositions[future, final_indices]
        anchor = donor["anchor_compositions"][target_key] if kind == "switcher" else None
        primary.append(
            _sequence_success(
                primary_vectors,
                donor["target_compositions"][target_key],
                anchor,
            )
        )
        state_vectors = _observer_vectors(substrate, trace.terminals[future, final_indices])
        for name, vectors in state_vectors.items():
            aux_anchor = donor["anchor_compositions"].get(name) if kind == "switcher" else None
            auxiliary[name].append(
                _sequence_success(vectors, donor["target_compositions"][name], aux_anchor)
            )
    result = {"primary": primary}
    result.update(auxiliary)
    return result


def _run_recovery(
    donor: dict[str, Any],
    base_state: np.ndarray,
    contract: CausalContract,
    *,
    replicates: int,
    condition_key: str,
    process_noise: float | None = None,
    copy_error: float | None = None,
    horizon: int | None = None,
    observer_override: str | None = None,
    host_rule: int | None = None,
) -> tuple[BatchTrace, dict[str, list[bool]]]:
    substrate = str(donor["substrate"])
    rule = int(donor["rule"] if host_rule is None else host_rule)
    epsilon = (
        contract.eca_copy_error if substrate == "eca" else contract.life_copy_error
    ) if copy_error is None else copy_error
    initial_seed = _hash_seed(contract.namespace, "initial-copy", condition_key)
    initial = _copy_batch(base_state, replicates, epsilon, initial_seed)
    observer = _observer_name_for_simulation(donor, observer_override)
    domain = None
    if substrate == "eca" and observer == "particle":
        domain = build_e19_domain_dictionary(rule).codes
    trace = _simulate_batch(
        substrate,
        rule,
        initial,
        contract,
        horizon=horizon or contract.recovery_horizon,
        rng_seed=_hash_seed(contract.namespace, "garden", condition_key),
        observer=observer,
        process_noise=process_noise,
        copy_error=epsilon,
        domain_codes=domain,
    )
    return trace, _recovery_outcomes(trace, donor, simulation_observer=observer)


def _summarize_outcomes(outcomes: dict[str, list[bool]]) -> dict[str, Any]:
    return {
        "success_count": int(sum(outcomes["primary"])),
        "success_rate": float(np.mean(outcomes["primary"])) if outcomes["primary"] else 0.0,
        "observer_success_counts": {name: int(sum(values)) for name, values in outcomes.items()},
        "observer_success_rates": {
            name: float(np.mean(values)) if values else 0.0 for name, values in outcomes.items()
        },
        "n": len(outcomes["primary"]),
    }


def _structured_geometries(substrate: str) -> tuple[str, ...]:
    return ("one_interval", "two_interval") if substrate == "eca" else ("square", "strip", "two_lobe")


def _common_garden_task(
    arguments: tuple[dict[str, Any], CausalContract, CausalProfile]
) -> dict[str, Any]:
    donor_result, contract, profile = arguments
    entry = donor_result["entry"]
    substrate = str(entry["substrate"])
    rule = int(entry["rule"])
    rows: list[dict[str, Any]] = []
    for donor in donor_result["donors"]:
        donor_id = str(donor["donor_id"])
        source = _state_from_hex(substrate, donor["donor_state_hex"])
        ancestor = _state_from_hex(substrate, donor["ancestor_state_hex"])
        conditions: list[tuple[str, float, np.ndarray]] = [
            ("intact", 1.0, source),
            ("ancestor", 1.0, ancestor),
        ]
        for fraction in contract.transmission_fractions:
            key = f"{donor_id}:{fraction:.2f}"
            primary_geometry = _structured_geometries(substrate)[0]
            primary_fragment: np.ndarray | None = None
            for geometry in _structured_geometries(substrate):
                fragment = _intervention_state(source, fraction, geometry, key + ":" + geometry)
                if geometry == primary_geometry:
                    primary_fragment = fragment
                conditions.append(
                    (geometry, fraction, fragment)
                )
            conditions.append(
                (
                    "dispersed",
                    fraction,
                    _intervention_state(source, fraction, "dispersed", key + ":dispersed"),
                )
            )
            assert primary_fragment is not None
            conditions.append(
                (
                    "shuffled",
                    fraction,
                    _exact_random_state(
                        tuple(source.shape), int(primary_fragment.sum()), key + ":shuffled"
                    ),
                )
            )
            for intervention in ("density_random",):
                conditions.append(
                    (
                        intervention,
                        fraction,
                        _intervention_state(source, fraction, intervention, key + ":" + intervention),
                    )
                )
        for intervention, fraction, base in conditions:
            condition_key = f"common:{substrate}:{rule}:{donor_id}:{intervention}:{fraction:.2f}"
            trace, outcomes = _run_recovery(
                donor,
                base,
                contract,
                replicates=profile.recovery_replicates,
                condition_key=condition_key,
            )
            summary = _summarize_outcomes(outcomes)
            rows.append(
                {
                    "substrate": substrate,
                    "rule": rule,
                    "donor_id": donor_id,
                    "donor_kind": donor["kind"],
                    "intervention": intervention,
                    "fraction": fraction,
                    "transmitted_sites": int(np.count_nonzero(base | ~base)) if intervention == "intact" else int(round(base.size * fraction)),
                    "transmitted_live": int(np.count_nonzero(base)),
                    "death_count": int(sum(reason is not None for reason in trace.death)),
                    **summary,
                }
            )
    return {"entry": entry, "n_donors": len(donor_result["donors"]), "rows": rows}


def _branch_states(parent: np.ndarray, arm: str, substrate: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    if arm == "intact":
        return parent.copy(), parent.copy()
    geometry = "one_interval" if substrate == "eca" else "square"
    mask = _site_mask(tuple(parent.shape), 0.5, geometry, key)
    left = parent & mask
    right = parent & ~mask
    if arm == "complementary_half":
        return left, right
    if arm == "shuffled_half":
        return (
            _exact_random_state(tuple(parent.shape), int(left.sum()), key + ":left"),
            _exact_random_state(tuple(parent.shape), int(right.sum()), key + ":right"),
        )
    raise ValueError(f"unknown pedigree arm {arm!r}")


def _pedigree_task(arguments: tuple[dict[str, Any], CausalContract, CausalProfile]) -> dict[str, Any]:
    donor_result, contract, profile = arguments
    entry = donor_result["entry"]
    substrate = str(entry["substrate"])
    rule = int(entry["rule"])
    observer = str(entry["observer"])
    domain = _domain_for_entry(entry)
    rows: list[dict[str, Any]] = []
    for donor in donor_result["donors"][: profile.pedigree_donors]:
        donor_id = str(donor["donor_id"])
        root = _state_from_hex(substrate, donor["donor_state_hex"])
        for arm in ("intact", "complementary_half", "shuffled_half"):
            parents = [root]
            depth_rows: list[dict[str, Any]] = []
            for depth in range(1, profile.pedigree_depth + 1):
                children: list[np.ndarray] = []
                for parent_index, parent in enumerate(parents):
                    children.extend(
                        _branch_states(
                            parent,
                            arm,
                            substrate,
                            f"{donor_id}:{arm}:{depth}:{parent_index}",
                        )
                    )
                epsilon = contract.eca_copy_error if substrate == "eca" else contract.life_copy_error
                child_batch = np.stack(children)
                if epsilon > 0:
                    copy_rng = np.random.default_rng(
                        _hash_seed(contract.namespace, "pedigree-copy", donor_id, arm, depth)
                    )
                    child_batch ^= copy_rng.random(child_batch.shape) < epsilon
                trace = _simulate_batch(
                    substrate,
                    rule,
                    child_batch,
                    contract,
                    horizon=1,
                    rng_seed=_hash_seed(contract.namespace, "pedigree-generation", donor_id, arm, depth),
                    observer=observer,
                    copy_error=0.0,
                    domain_codes=domain,
                )
                target = donor["target_compositions"]["primary"]
                anchor = donor["anchor_compositions"]["primary"] if donor["kind"] == "switcher" else None
                identity = []
                surviving: list[np.ndarray] = []
                for child in range(len(child_batch)):
                    valid = bool(trace.valid[child, 0])
                    matches = valid and _sequence_success(
                        trace.compositions[child, :1], target, anchor, require_eight=False
                    )
                    identity.append(bool(matches))
                    if valid:
                        surviving.append(trace.terminals[child, 0].copy())
                sibling_pairs = len(identity) // 2
                sibling_concordance = (
                    sum(identity[2 * i] == identity[2 * i + 1] for i in range(sibling_pairs)) / sibling_pairs
                    if sibling_pairs
                    else 0.0
                )
                depth_rows.append(
                    {
                        "depth": depth,
                        "nodes": len(children),
                        "survivors": len(surviving),
                        "identity_rate": float(np.mean(identity)) if identity else 0.0,
                        "sibling_concordance": sibling_concordance,
                    }
                )
                parents = surviving
                if not parents:
                    for missing_depth in range(depth + 1, profile.pedigree_depth + 1):
                        depth_rows.append(
                            {
                                "depth": missing_depth,
                                "nodes": 0,
                                "survivors": 0,
                                "identity_rate": 0.0,
                                "sibling_concordance": 0.0,
                            }
                        )
                    break
            final_successes: list[bool] = []
            if parents:
                certification_base = np.stack(parents)
                epsilon = contract.eca_copy_error if substrate == "eca" else contract.life_copy_error
                if epsilon > 0:
                    copy_rng = np.random.default_rng(
                        _hash_seed(contract.namespace, "pedigree-cert-copy", donor_id, arm)
                    )
                    certification_base ^= copy_rng.random(certification_base.shape) < epsilon
                certification = _simulate_batch(
                    substrate,
                    rule,
                    certification_base,
                    contract,
                    horizon=8,
                    rng_seed=_hash_seed(contract.namespace, "pedigree-cert", donor_id, arm),
                    observer=observer,
                    domain_codes=domain,
                )
                final_successes = _recovery_outcomes(
                    certification, donor, simulation_observer=observer
                )["primary"]
            rows.append(
                {
                    "substrate": substrate,
                    "rule": rule,
                    "donor_id": donor_id,
                    "arm": arm,
                    "donor_kind": donor["kind"],
                    "depths": depth_rows,
                    "final_leaves": len(parents),
                    "final_success_count": int(sum(final_successes)),
                    "final_success_rate": float(np.mean(final_successes)) if final_successes else 0.0,
                }
            )
    return {"entry": entry, "n_donors": min(len(donor_result["donors"]), profile.pedigree_donors), "rows": rows}


def _noise_task(arguments: tuple[dict[str, Any], CausalContract, CausalProfile]) -> dict[str, Any]:
    donor_result, contract, profile = arguments
    entry = donor_result["entry"]
    substrate = str(entry["substrate"])
    rule = int(entry["rule"])
    baseline_eta = contract.eca_process_noise if substrate == "eca" else contract.life_process_noise
    baseline_epsilon = contract.eca_copy_error if substrate == "eca" else contract.life_copy_error
    geometry = "one_interval" if substrate == "eca" else "square"
    rows: list[dict[str, Any]] = []
    for donor in donor_result["donors"][: profile.pedigree_donors]:
        donor_id = str(donor["donor_id"])
        source = _state_from_hex(substrate, donor["donor_state_hex"])
        bases = {
            "intact": source,
            "structured_half": _intervention_state(source, 0.5, geometry, donor_id + ":noise-half"),
        }
        for intervention, base in bases.items():
            for eta_multiplier in contract.noise_multipliers:
                for epsilon_multiplier in contract.noise_multipliers:
                    eta = baseline_eta * eta_multiplier
                    epsilon = baseline_epsilon * epsilon_multiplier
                    key = (
                        f"noise:{substrate}:{rule}:{donor_id}:{intervention}:"
                        f"{eta_multiplier}:{epsilon_multiplier}"
                    )
                    trace, outcomes = _run_recovery(
                        donor,
                        base,
                        contract,
                        replicates=profile.recovery_replicates,
                        condition_key=key,
                        process_noise=eta,
                        copy_error=epsilon,
                    )
                    rows.append(
                        {
                            "substrate": substrate,
                            "rule": rule,
                            "donor_id": donor_id,
                            "donor_kind": donor["kind"],
                            "intervention": intervention,
                            "eta_multiplier": eta_multiplier,
                            "epsilon_multiplier": epsilon_multiplier,
                            "death_count": int(sum(reason is not None for reason in trace.death)),
                            **_summarize_outcomes(outcomes),
                        }
                    )
    return {"entry": entry, "n_donors": min(len(donor_result["donors"]), profile.pedigree_donors), "rows": rows}


def _memory_task(arguments: tuple[dict[str, Any], CausalContract, CausalProfile]) -> dict[str, Any]:
    donor_result, contract, profile = arguments
    entry = donor_result["entry"]
    substrate = str(entry["substrate"])
    rule = int(entry["rule"])
    observer = str(entry["observer"])
    domain = _domain_for_entry(entry)
    baseline_eta = contract.eca_process_noise if substrate == "eca" else contract.life_process_noise
    epsilon = contract.eca_copy_error if substrate == "eca" else contract.life_copy_error
    rows: list[dict[str, Any]] = []
    replicate_count = max(2, profile.recovery_replicates)
    for donor in donor_result["donors"][: profile.pedigree_donors]:
        donor_id = str(donor["donor_id"])
        source = _state_from_hex(substrate, donor["donor_state_hex"])
        samples: list[dict[str, Any]] = []
        for label, multiplier in ((0, 0), (1, 2)):
            initial = _copy_batch(
                source,
                replicate_count,
                epsilon,
                _hash_seed(contract.namespace, "memory-initial", donor_id, label),
            )
            precondition = _simulate_batch(
                substrate,
                rule,
                initial,
                contract,
                horizon=contract.memory_precondition_horizon,
                rng_seed=_hash_seed(contract.namespace, "memory-cue", donor_id, label),
                observer=observer,
                process_noise=baseline_eta * multiplier,
                domain_codes=domain,
            )
            survivors = [
                future
                for future in range(replicate_count)
                if bool(precondition.valid[future, contract.memory_precondition_horizon - 1])
            ]
            if not survivors:
                continue
            garden_initial = precondition.offspring[
                survivors, contract.memory_precondition_horizon - 1
            ].copy()
            garden = _simulate_batch(
                substrate,
                rule,
                garden_initial,
                contract,
                horizon=contract.memory_garden_horizon,
                rng_seed=_hash_seed(contract.namespace, "memory-garden", donor_id, label),
                observer=observer,
                process_noise=baseline_eta,
                domain_codes=domain,
            )
            for local, original_future in enumerate(survivors):
                length = int(np.count_nonzero(garden.valid[local]))
                if not length:
                    continue
                samples.append(
                    {
                        "label": label,
                        "replicate": int(original_future),
                        "train": bool(original_future % 2 == 0),
                        "after_cue": precondition.compositions[
                            original_future, contract.memory_precondition_horizon - 1
                        ].astype(float).tolist(),
                        "after_one": garden.compositions[local, 0].astype(float).tolist(),
                        "after_eight": garden.compositions[local, length - 1].astype(float).tolist(),
                        "garden_generations": length,
                    }
                )
        rows.append(
            {
                "substrate": substrate,
                "rule": rule,
                "donor_id": donor_id,
                "donor_kind": donor["kind"],
                "samples": samples,
                "n_samples": len(samples),
            }
        )
    return {"entry": entry, "n_donors": min(len(donor_result["donors"]), profile.pedigree_donors), "rows": rows}


def _transplant_task(arguments: tuple[dict[str, Any], CausalContract, CausalProfile]) -> dict[str, Any]:
    donor_result, contract, profile = arguments
    entry = donor_result["entry"]
    substrate = str(entry["substrate"])
    native_rule = int(entry["rule"])
    neighbors = [native_rule ^ (1 << bit) for bit in range(8 if substrate == "eca" else 17)]
    rows: list[dict[str, Any]] = []
    switchers = [donor for donor in donor_result["donors"] if donor["kind"] == "switcher"]
    for donor in switchers[: profile.transplant_donors]:
        donor_id = str(donor["donor_id"])
        source = _state_from_hex(substrate, donor["donor_state_hex"])
        for host_rule in [native_rule] + neighbors:
            key = f"transplant:{substrate}:{native_rule}:{donor_id}:{host_rule}"
            trace, outcomes = _run_recovery(
                donor,
                source,
                contract,
                replicates=profile.transplant_replicates,
                condition_key=key,
                observer_override="raw" if substrate == "eca" else None,
                host_rule=host_rule,
            )
            rows.append(
                {
                    "substrate": substrate,
                    "donor_rule": native_rule,
                    "host_rule": host_rule,
                    "native": host_rule == native_rule,
                    "donor_id": donor_id,
                    "donor_kind": donor["kind"],
                    "death_count": int(sum(reason is not None for reason in trace.death)),
                    **_summarize_outcomes(outcomes),
                }
            )
    return {"entry": entry, "n_donors": min(len(switchers), profile.transplant_donors), "rows": rows}


def _flatten_rows(stage_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for result in stage_results for row in result.get("rows", [])]


def _bootstrap_summary(values: Sequence[float], resamples: int, seed: int) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return {"n_donors": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    if len(data) == 1:
        samples = np.repeat(data[0], resamples)
    else:
        indices = rng.integers(0, len(data), size=(resamples, len(data)))
        samples = data[indices].mean(axis=1)
    return {
        "n_donors": len(data),
        "mean": float(data.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def _paired_condition_differences(
    rows: Sequence[dict[str, Any]],
    substrate: str,
    left: tuple[str, float],
    right: tuple[str, float],
    *,
    observer: str = "primary",
    donor_kind: str | None = "switcher",
) -> list[tuple[str, int, float]]:
    field = "success_rate" if observer == "primary" else "observer_success_rates"
    grouped: dict[str, dict[tuple[str, float], float]] = defaultdict(dict)
    rule_by_donor: dict[str, int] = {}
    for row in rows:
        if row["substrate"] != substrate or (
            donor_kind is not None and row.get("donor_kind") != donor_kind
        ):
            continue
        key = (str(row["intervention"]), float(row["fraction"]))
        donor_id = str(row["donor_id"])
        if observer == "primary":
            value = float(row[field])
        else:
            rates = row[field]
            if observer not in rates:
                continue
            value = float(rates[observer])
        grouped[donor_id][key] = value
        rule_by_donor[donor_id] = int(row["rule"])
    return [
        (donor_id, rule_by_donor[donor_id], values[left] - values[right])
        for donor_id, values in grouped.items()
        if left in values and right in values
    ]


def _dose_slopes(
    rows: Sequence[dict[str, Any]], substrate: str, *, donor_kind: str | None = "switcher"
) -> tuple[list[float], list[float]]:
    geometry = "one_interval" if substrate == "eca" else "square"
    grouped: dict[str, dict[float, float]] = defaultdict(dict)
    for row in rows:
        if row["substrate"] != substrate or (
            donor_kind is not None and row.get("donor_kind") != donor_kind
        ):
            continue
        intervention = str(row["intervention"])
        fraction = float(row["fraction"])
        if intervention == geometry or intervention == "intact":
            grouped[str(row["donor_id"])][fraction] = float(row["success_rate"])
    slopes: list[float] = []
    pooled = {fraction: [] for fraction in (0.25, 0.5, 0.75, 1.0)}
    x = np.asarray((0.25, 0.5, 0.75, 1.0), dtype=np.float64)
    for values in grouped.values():
        if all(fraction in values for fraction in x):
            y = np.asarray([values[float(fraction)] for fraction in x])
            slopes.append(float(np.polyfit(x, y, 1)[0]))
            for fraction, value in zip(x, y, strict=True):
                pooled[float(fraction)].append(float(value))
    means = [float(np.mean(pooled[fraction])) if pooled[fraction] else 0.0 for fraction in x]
    return slopes, means


def _paired_pvalue(values: Sequence[float], resamples: int, seed: int) -> float:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return 1.0
    observed = float(data.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(resamples):
        signs = rng.choice(np.asarray((-1.0, 1.0)), size=len(data))
        exceed += int(float(np.mean(data * signs)) >= observed)
    return (exceed + 1) / (resamples + 1)


def _bh_adjust(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    order = sorted(range(len(rows)), key=lambda index: (float(rows[index]["p_value"]), index))
    adjusted = [1.0] * len(rows)
    running = 1.0
    count = len(rows)
    for reverse_rank in range(count - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, float(rows[index]["p_value"]) * count / rank)
        adjusted[index] = min(1.0, running)
    for index, value in enumerate(adjusted):
        rows[index]["q_value"] = value
        rows[index]["significant_q05"] = bool(value <= 0.05)
    return rows


def _memory_accuracy(
    rows: Sequence[dict[str, Any]],
    substrate: str,
    field: str,
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if row["substrate"] == substrate and row.get("donor_kind") == "switcher"
    ]

    def score(label_maps: dict[str, list[int]] | None = None) -> tuple[int, int]:
        correct = total = 0
        for row in usable:
            samples = row["samples"]
            labels = (
                label_maps[str(row["donor_id"])]
                if label_maps is not None and str(row["donor_id"]) in label_maps
                else [int(sample["label"]) for sample in samples]
            )
            centroids: dict[int, np.ndarray] = {}
            for label in (0, 1):
                vectors = [
                    sample[field]
                    for sample, assigned in zip(samples, labels, strict=True)
                    if sample["train"] and assigned == label
                ]
                if vectors:
                    centroids[label] = np.mean(np.asarray(vectors, dtype=np.float64), axis=0)
            if len(centroids) != 2:
                continue
            for sample, assigned in zip(samples, labels, strict=True):
                if sample["train"]:
                    continue
                vector = sample[field]
                prediction = max((0, 1), key=lambda label: (_cosine(vector, centroids[label]), -label))
                correct += int(prediction == assigned)
                total += 1
        return correct, total

    observed_correct, observed_total = score()
    accuracy = observed_correct / observed_total if observed_total else 0.0
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        mappings: dict[str, list[int]] = {}
        for row in usable:
            labels = np.asarray([int(sample["label"]) for sample in row["samples"]], dtype=int)
            permuted = labels.copy()
            train = np.asarray([bool(sample["train"]) for sample in row["samples"]])
            for split in (False, True):
                indices = np.flatnonzero(train == split)
                if len(indices):
                    permuted[indices] = rng.permutation(labels[indices])
            mappings[str(row["donor_id"])] = permuted.astype(int).tolist()
        correct, total = score(mappings)
        permuted = correct / total if total else 0.0
        exceed += int(permuted >= accuracy)
    return {
        "correct": observed_correct,
        "n_test": observed_total,
        "accuracy": accuracy,
        "permutation_p": (exceed + 1) / (permutations + 1),
        "permutations": permutations,
    }


def adjudicate_causal_campaign(
    stages: dict[str, list[dict[str, Any]]],
    contract: CausalContract,
    profile: CausalProfile,
) -> dict[str, Any]:
    common = _flatten_rows(stages.get("common_garden", []))
    pedigrees = _flatten_rows(stages.get("pedigrees", []))
    noise = _flatten_rows(stages.get("noise", []))
    memory = _flatten_rows(stages.get("memory", []))
    transplants = _flatten_rows(stages.get("transplant", []))
    bootstrap_n = profile.bootstrap_resamples
    contrasts: dict[str, Any] = {}
    causal_passes: list[bool] = []
    structure_passes: list[bool] = []
    dose_passes: list[bool] = []
    pedigree_passes: list[bool] = []
    observer_passes: list[bool] = []
    transplant_passes: list[bool] = []
    rule_tests: list[dict[str, Any]] = []

    for substrate in ("eca", "life"):
        geometry = "one_interval" if substrate == "eca" else "square"
        causal = _paired_condition_differences(
            common, substrate, (geometry, 0.5), ("density_random", 0.5)
        )
        structure = _paired_condition_differences(
            common, substrate, (geometry, 0.5), ("shuffled", 0.5)
        )
        maintainer_causal = _paired_condition_differences(
            common,
            substrate,
            (geometry, 0.5),
            ("density_random", 0.5),
            donor_kind="maintainer",
        )
        causal_summary = _bootstrap_summary(
            [value for _, _, value in causal], bootstrap_n, _hash_seed(contract.namespace, "boot", substrate, "causal")
        )
        structure_summary = _bootstrap_summary(
            [value for _, _, value in structure], bootstrap_n, _hash_seed(contract.namespace, "boot", substrate, "structure")
        )
        slopes, dose_means = _dose_slopes(common, substrate)
        slope_summary = _bootstrap_summary(
            slopes, bootstrap_n, _hash_seed(contract.namespace, "boot", substrate, "dose")
        )
        monotonic = all(left <= right for left, right in zip(dose_means, dose_means[1:]))

        pedigree_grouped: dict[str, dict[str, float]] = defaultdict(dict)
        for row in pedigrees:
            if row["substrate"] == substrate and row.get("donor_kind") == "switcher":
                pedigree_grouped[str(row["donor_id"])][str(row["arm"])] = float(row["final_success_rate"])
        pedigree_differences = [
            values["complementary_half"] - values["shuffled_half"]
            for values in pedigree_grouped.values()
            if "complementary_half" in values and "shuffled_half" in values
        ]
        pedigree_summary = _bootstrap_summary(
            pedigree_differences,
            bootstrap_n,
            _hash_seed(contract.namespace, "boot", substrate, "pedigree"),
        )

        auxiliary_names = ("raw4", "multiscale") if substrate == "eca" else ("terminal2x2", "components")
        observer_directions: dict[str, float | None] = {}
        for observer_name in auxiliary_names:
            differences = _paired_condition_differences(
                common,
                substrate,
                (geometry, 0.5),
                ("density_random", 0.5),
                observer=observer_name,
            )
            observer_directions[observer_name] = (
                float(np.mean([value for _, _, value in differences])) if differences else None
            )

        transplant_grouped: dict[str, dict[str, list[float] | float]] = defaultdict(
            lambda: {"neighbors": []}
        )
        for row in transplants:
            if row["substrate"] != substrate:
                continue
            group = transplant_grouped[str(row["donor_id"])]
            if row["native"]:
                group["native"] = float(row["success_rate"])
            else:
                assert isinstance(group["neighbors"], list)
                group["neighbors"].append(float(row["success_rate"]))
        transplant_differences = [
            float(values["native"]) - float(np.mean(values["neighbors"]))
            for values in transplant_grouped.values()
            if "native" in values and values["neighbors"]
        ]
        transplant_summary = _bootstrap_summary(
            transplant_differences,
            bootstrap_n,
            _hash_seed(contract.namespace, "boot", substrate, "transplant"),
        )

        contrasts[substrate] = {
            "causal_50_structured_minus_density_random": causal_summary,
            "structure_50_structured_minus_shuffled": structure_summary,
            "maintainer_50_structured_minus_density_random": _bootstrap_summary(
                [value for _, _, value in maintainer_causal],
                bootstrap_n,
                _hash_seed(contract.namespace, "boot", substrate, "maintainer-causal"),
            ),
            "dose_response": {
                "fractions": [0.25, 0.5, 0.75, 1.0],
                "mean_success": dose_means,
                "monotonic": monotonic,
                "slope": slope_summary,
            },
            "pedigree_half_minus_shuffled_depth_final": pedigree_summary,
            "observer_directions": observer_directions,
            "native_minus_neighbor_transplant": transplant_summary,
        }
        causal_passes.append(
            causal_summary["mean"] is not None
            and causal_summary["mean"] >= 0.15
            and causal_summary["ci95"][0] > 0.0
        )
        structure_passes.append(
            structure_summary["mean"] is not None and structure_summary["ci95"][0] > 0.0
        )
        dose_passes.append(
            monotonic and slope_summary["mean"] is not None and slope_summary["ci95"][0] > 0.0
        )
        pedigree_passes.append(
            pedigree_summary["mean"] is not None and pedigree_summary["ci95"][0] > 0.0
        )
        observer_passes.append(
            causal_summary["mean"] is not None
            and causal_summary["mean"] > 0.0
            and any(value is not None and value > 0.0 for value in observer_directions.values())
        )
        transplant_passes.append(
            transplant_summary["mean"] is not None and transplant_summary["ci95"][0] > 0.0
        )

        by_rule: dict[int, list[float]] = defaultdict(list)
        for _, rule, value in causal:
            by_rule[rule].append(value)
        for rule, values in by_rule.items():
            rule_tests.append(
                {
                    "substrate": substrate,
                    "rule": rule,
                    "n_donors": len(values),
                    "effect": float(np.mean(values)),
                    "p_value": _paired_pvalue(
                        values,
                        bootstrap_n,
                        _hash_seed(contract.namespace, "rule-permutation", substrate, rule),
                    ),
                }
            )

    memory_results: dict[str, Any] = {}
    memory_passes: list[bool] = []
    for substrate in ("eca", "life"):
        after_one = _memory_accuracy(
            memory,
            substrate,
            "after_one",
            permutations=bootstrap_n,
            seed=_hash_seed(contract.namespace, "memory-permutation", substrate, "one"),
        )
        after_eight = _memory_accuracy(
            memory,
            substrate,
            "after_eight",
            permutations=bootstrap_n,
            seed=_hash_seed(contract.namespace, "memory-permutation", substrate, "eight"),
        )
        memory_results[substrate] = {"after_one": after_one, "after_eight": after_eight}
        memory_passes.append(
            after_one["accuracy"] > 0.65
            and after_one["permutation_p"] < 0.05
            and after_eight["accuracy"] > 0.65
            and after_eight["permutation_p"] < 0.05
        )

    noise_summary: dict[str, Any] = {}
    for substrate in ("eca", "life"):
        cells: dict[str, list[float]] = defaultdict(list)
        for row in noise:
            if row["substrate"] == substrate:
                key = f"{row['intervention']}:eta{row['eta_multiplier']}:epsilon{row['epsilon_multiplier']}"
                cells[key].append(float(row["success_rate"]))
        noise_summary[substrate] = {
            key: {"n_donors": len(values), "mean_success": float(np.mean(values))}
            for key, values in sorted(cells.items())
        }

    donor_acquisition = {
        substrate: {
            "rules": len([row for row in stages.get("donors", []) if row["entry"]["substrate"] == substrate]),
            "target_donors": sum(
                int(row["target"]) for row in stages.get("donors", []) if row["entry"]["substrate"] == substrate
            ),
            "acquired_donors": sum(
                int(row["acquired"]) for row in stages.get("donors", []) if row["entry"]["substrate"] == substrate
            ),
            "rules_below_target": [
                int(row["entry"]["rule"])
                for row in stages.get("donors", [])
                if row["entry"]["substrate"] == substrate and int(row["acquired"]) < int(row["target"])
            ],
        }
        for substrate in ("eca", "life")
    }
    gates = {
        "causal_transmission": len(causal_passes) == 2 and all(causal_passes),
        "structure_matters": len(structure_passes) == 2 and all(structure_passes),
        "dose_response": len(dose_passes) == 2 and all(dose_passes),
        "pedigree_persistence": len(pedigree_passes) == 2 and all(pedigree_passes),
        "observer_robustness": len(observer_passes) == 2 and all(observer_passes),
        "environmental_memory": len(memory_passes) == 2 and all(memory_passes),
        "rule_specificity": len(transplant_passes) == 2 and all(transplant_passes),
    }
    return {
        "gates": gates,
        "contrasts": contrasts,
        "environmental_memory": memory_results,
        "noise_decomposition": noise_summary,
        "donor_acquisition": donor_acquisition,
        "rule_level_tests": _bh_adjust(rule_tests),
        "row_counts": {
            "common_garden": len(common),
            "pedigrees": len(pedigrees),
            "noise": len(noise),
            "memory_donors": len(memory),
            "transplant": len(transplants),
        },
    }


def _checkpoint_name(item: dict[str, Any]) -> str:
    entry = item.get("entry", item)
    return f"{entry['substrate']}-rule-{int(entry['rule']):06d}.json"


def _run_parallel_stage(
    output: Path,
    stage: str,
    items: Sequence[dict[str, Any]],
    task: Callable[[tuple[dict[str, Any], CausalContract, CausalProfile]], dict[str, Any]],
    contract: CausalContract,
    profile: CausalProfile,
    *,
    design_digest: str,
    workers: int,
    resume: bool,
    deadline: float | None,
    status: Callable[..., None],
) -> tuple[list[dict[str, Any]], bool]:
    root = output / stage
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[str, int], dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for item in items:
        entry = item.get("entry", item)
        key = (str(entry["substrate"]), int(entry["rule"]))
        path = checkpoints / _checkpoint_name(item)
        if resume and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("design_digest") == design_digest:
                results[key] = payload["result"]
                continue
        missing.append(item)

    total = len(items)
    started = time.time()

    def save(result: dict[str, Any]) -> None:
        entry = result["entry"]
        key = (str(entry["substrate"]), int(entry["rule"]))
        _atomic_json(
            checkpoints / _checkpoint_name(result),
            {"design_digest": design_digest, "stage": stage, "result": result},
        )
        results[key] = result
        completed = len(results)
        elapsed = max(time.time() - started, 1e-9)
        eta = elapsed / max(1, completed) * max(0, total - completed)
        status("running", stage, completed=completed, total=total, eta_seconds=eta)

    truncated = False
    if workers <= 1:
        for item in missing:
            if deadline is not None and time.time() >= deadline:
                truncated = True
                break
            save(task((item, contract, profile)))
    elif missing:
        pool = ProcessPoolExecutor(max_workers=min(workers, len(missing)))
        futures = {
            pool.submit(task, (item, contract, profile)): item
            for item in missing
        }
        processed: set[Any] = set()
        try:
            for future in as_completed(futures):
                save(future.result())
                processed.add(future)
                if deadline is not None and time.time() >= deadline:
                    truncated = True
                    for pending in futures:
                        if pending not in processed:
                            pending.cancel()
                    break
        finally:
            pool.shutdown(wait=True, cancel_futures=truncated)
        if truncated:
            # Running rule checkpoints are allowed to finish at the deadline;
            # retain their completed results rather than throwing work away.
            for future in futures:
                if future in processed or future.cancelled() or not future.done():
                    continue
                save(future.result())

    complete = len(results) == total
    ordered = [
        results[key]
        for key in sorted(results, key=lambda value: (value[0], value[1]))
    ]
    _atomic_json(
        root / "stage_summary.json",
        {
            "stage": stage,
            "design_digest": design_digest,
            "complete": complete,
            "completed": len(results),
            "total": total,
            "budget_truncated": truncated or not complete,
        },
    )
    if complete:
        _atomic_text(root / "COMPLETE", "complete\n")
    return ordered, complete


def _write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        _atomic_text(path, "")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )
    os.replace(temporary, path)


def _render_report(results: dict[str, Any]) -> str:
    adjudication = results["adjudication"]
    lines = [
        "# Causal Plastic Heredity in Cellular Automata",
        "",
        f"Profile: `{results['profile']}`. State: **{results['state']}**.",
        f"Design digest: `{results['design_digest']}`.",
        "",
        "## Donor acquisition",
        "",
    ]
    for substrate, values in adjudication["donor_acquisition"].items():
        lines.append(
            f"- {substrate.upper()}: {values['acquired_donors']}/{values['target_donors']} donors; "
            f"below-target rules `{values['rules_below_target']}`."
        )
    lines.extend(("", "## Registered gates", ""))
    for gate, value in adjudication["gates"].items():
        lines.append(f"- `{gate}`: **{value}**")
    lines.extend(("", "## Primary causal contrasts", ""))
    for substrate, values in adjudication["contrasts"].items():
        causal = values["causal_50_structured_minus_density_random"]
        structure = values["structure_50_structured_minus_shuffled"]
        maintainer = values["maintainer_50_structured_minus_density_random"]
        pedigree = values["pedigree_half_minus_shuffled_depth_final"]
        transplant = values["native_minus_neighbor_transplant"]
        lines.extend(
            (
                f"### {substrate.upper()}",
                "",
                f"- Structured-half minus density-random: `{causal['mean']}`; 95% CI `{causal['ci95']}`.",
                f"- Structured-half minus shuffled: `{structure['mean']}`; 95% CI `{structure['ci95']}`.",
                f"- Stable-maintainer structured-half minus density-random (descriptive control): "
                f"`{maintainer['mean']}`; 95% CI `{maintainer['ci95']}`.",
                f"- Dose means: `{values['dose_response']['mean_success']}`; slope CI "
                f"`{values['dose_response']['slope']['ci95']}`.",
                f"- Pedigree half minus shuffled: `{pedigree['mean']}`; 95% CI `{pedigree['ci95']}`.",
                f"- Native minus one-bit-neighbor transplant: `{transplant['mean']}`; "
                f"95% CI `{transplant['ci95']}`.",
                f"- Independent-observer directions: `{values['observer_directions']}`.",
                "",
            )
        )
    lines.extend(("## Environmental memory", ""))
    for substrate, values in adjudication["environmental_memory"].items():
        lines.append(
            f"- {substrate.upper()}: after one `{values['after_one']['accuracy']:.3f}` "
            f"(p `{values['after_one']['permutation_p']:.4g}`); after eight "
            f"`{values['after_eight']['accuracy']:.3f}` "
            f"(p `{values['after_eight']['permutation_p']:.4g}`)."
        )
    lines.extend(
        (
            "",
            "## Interpretation boundary",
            "",
            "A passed causal gate means inherited structured lattice information regenerated the acquired "
            "observer-level form more reliably than registered mass- and density-matched controls. It does "
            "not establish biochemical genetics, agency, or observer-independent organismhood. All nulls, "
            "reversals, missing donors, and truncated conditions remain in the machine-readable artifacts.",
            "",
        )
    )
    return "\n".join(lines)


def _render_lay_summary(results: dict[str, Any]) -> str:
    gates = results["adjudication"]["gates"]
    passed = [name.replace("_", " ") for name, value in gates.items() if value]
    failed = [name.replace("_", " ") for name, value in gates.items() if not value]
    state_note = (
        "The planned run finished."
        if results["state"] == "complete"
        else "The time budget ended before every planned cell finished, so these are explicitly partial results."
    )
    return "\n\n".join(
        (
            "# Lay summary",
            state_note,
            (
                "We treated the cellular pattern like a parent and physically removed or rearranged parts of "
                "what it would pass to a daughter. The important comparison is whether an ordered piece of the "
                "parent rebuilds the parent's newly acquired pattern better than the same amount of randomly "
                "placed material. Simply staying alive does not count."
            ),
            (
                "We also followed branching family trees, changed developmental and copying noise separately, "
                "removed an environmental cue to look for memory, moved patterns into nearby rules, and checked "
                "whether more than one way of describing the pattern told the same story."
            ),
            f"Passed headline tests: {', '.join(passed) if passed else 'none'}.",
            f"Did not pass headline tests: {', '.join(failed) if failed else 'none'}.",
            (
                "A passed test is evidence that spatial information in the inherited CA state matters causally. "
                "It is not a claim that the automaton has DNA, intentions, or life in the ordinary biological sense."
            ),
        )
    ) + "\n"


def _update_discovery_log(results: dict[str, Any], path: Path) -> None:
    marker_start = "<!-- causal-heredity-round-1:start -->"
    marker_end = "<!-- causal-heredity-round-1:end -->"
    gates = results["adjudication"]["gates"]
    section = "\n".join(
        (
            marker_start,
            "## Causal heredity round 1",
            "",
            f"Completed `{results['completed_unix']}` under design `{results['design_digest']}`.",
            "",
            *[f"- `{name}`: `{value}`" for name, value in gates.items()],
            "",
            "See `results/causal-heredity-round-1/REPORT.md` and `LAY_SUMMARY.md` for the full evidence boundary.",
            marker_end,
        )
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Discovery log\n"
    if marker_start in existing and marker_end in existing:
        prefix = existing.split(marker_start, 1)[0].rstrip()
        suffix = existing.split(marker_end, 1)[1].lstrip()
        updated = prefix + "\n\n" + section + ("\n\n" + suffix if suffix else "\n")
    else:
        updated = existing.rstrip() + "\n\n" + section + "\n"
    _atomic_text(path, updated)


def run_causal_campaign(
    output: Path,
    *,
    life_atlas: Path,
    profile_name: str = "reference",
    workers: int = 20,
    max_hours: float = 24.0,
    resume: bool = False,
    selected_stages: Sequence[str] | None = None,
) -> dict[str, Any]:
    require_pinned_numpy()
    if profile_name not in CAUSAL_PROFILES:
        raise ValueError(f"unknown causal profile {profile_name!r}")
    if not life_atlas.exists():
        raise FileNotFoundError(f"frozen Life-family atlas not found: {life_atlas}")
    profile = CAUSAL_PROFILES[profile_name]
    contract = CausalContract(pedigree_depth=profile.pedigree_depth)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + max_hours * 3600 if max_hours > 0 else None

    def status(state: str, stage: str, **extra: Any) -> None:
        payload = {
            "state": state,
            "stage": stage,
            "profile": profile_name,
            "pid": os.getpid(),
            "started_unix": started,
            "updated_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "deadline_unix": deadline,
            **extra,
        }
        _atomic_json(output / "STATUS.json", payload)
        progress = f" {extra['completed']}/{extra['total']}" if "completed" in extra else ""
        print(f"[{state}] {stage}{progress}", flush=True)

    panel = build_rule_panel(life_atlas, profile)
    entries = list(panel["eca"]) + list(panel["life"])
    implementation_files = (
        Path(__file__),
        Path(__file__).with_name("e19.py"),
        Path(__file__).with_name("life_family.py"),
        Path(__file__).with_name("particle_e19.py"),
    )
    design_payload = {
        "contract": contract.to_dict(),
        "profile": asdict(profile),
        "panel": panel,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "life_atlas_sha256": _sha256(life_atlas),
        "implementation_sha256": {
            path.name: _sha256(path) for path in implementation_files
        },
    }
    design_digest = hashlib.sha256(
        json.dumps(design_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(output / "DESIGN.json", {**design_payload, "design_digest": design_digest})
    _atomic_json(
        output / "MANIFEST.json",
        {
            "experiment": "causal_plastic_heredity",
            "profile": profile_name,
            "design_digest": design_digest,
            "started_unix": started,
            "workers": workers,
            "max_hours": max_hours,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
        },
    )
    stages_to_run = set(selected_stages or ("donors", "common_garden", "pedigrees", "noise", "memory", "transplant"))
    stage_data: dict[str, list[dict[str, Any]]] = {}
    completeness: dict[str, bool] = {}

    try:
        status("running", "donors")
        donors, complete = _run_parallel_stage(
            output,
            "donors",
            entries,
            _discover_donors,
            contract,
            profile,
            design_digest=design_digest,
            workers=workers,
            resume=resume or "donors" not in stages_to_run,
            deadline=deadline if "donors" in stages_to_run else started,
            status=status,
        )
        stage_data["donors"] = donors
        completeness["donors"] = complete
        downstream: tuple[
            tuple[str, Callable[[tuple[dict[str, Any], CausalContract, CausalProfile]], dict[str, Any]]],
            ...,
        ] = (
            ("common_garden", _common_garden_task),
            ("pedigrees", _pedigree_task),
            ("noise", _noise_task),
            ("memory", _memory_task),
            ("transplant", _transplant_task),
        )
        for stage, task in downstream:
            if deadline is not None and time.time() >= deadline:
                completeness[stage] = False
                stage_data[stage] = []
                status("budget_truncated", stage)
                continue
            status("running", stage)
            rows, stage_complete = _run_parallel_stage(
                output,
                stage,
                donors,
                task,
                contract,
                profile,
                design_digest=design_digest,
                workers=workers,
                resume=resume or stage not in stages_to_run,
                deadline=deadline if stage in stages_to_run else started,
                status=status,
            )
            stage_data[stage] = rows
            completeness[stage] = stage_complete

        status("running", "adjudication")
        adjudication = adjudicate_causal_campaign(stage_data, contract, profile)
        for stage in ("common_garden", "pedigrees", "noise", "memory", "transplant"):
            _write_rows_csv(output / stage / f"{stage}.csv", _flatten_rows(stage_data.get(stage, [])))
        _write_rows_csv(output / "rule_level_tests.csv", adjudication["rule_level_tests"])
        all_complete = all(completeness.get(stage, False) for stage in ("donors", "common_garden", "pedigrees", "noise", "memory", "transplant"))
        results = {
            "experiment": "causal_plastic_heredity",
            "profile": profile_name,
            "state": "complete" if all_complete else "partial_budget_exhausted",
            "design_digest": design_digest,
            "contract_digest": contract.digest,
            "started_unix": started,
            "completed_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "stage_completeness": completeness,
            "adjudication": adjudication,
        }
        _atomic_json(output / "RESULTS.json", results)
        _atomic_text(output / "REPORT.md", _render_report(results))
        _atomic_text(output / "LAY_SUMMARY.md", _render_lay_summary(results))
        if all_complete:
            _atomic_text(output / "COMPLETE", "complete\n")
            partial = output / "PARTIAL"
            if partial.exists():
                partial.unlink()
            if profile_name == "reference":
                _update_discovery_log(results, Path("DISCOVERY_LOG_EIDOSOMA_SCIENTIST.md"))
            status("complete", "campaign")
        else:
            _atomic_text(output / "PARTIAL", "budget exhausted; resume supported\n")
            status("partial_budget_exhausted", "campaign")
        return results
    except BaseException as error:
        status("failed", "campaign", error=repr(error))
        raise


def launch_detached(command: Sequence[str], output: Path) -> int:
    """Start a campaign in a new session and leave pollable local artifacts."""

    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    _atomic_text(output / "RUN.pid", f"{process.pid}\n")
    _atomic_json(
        output / "STATUS.json",
        {
            "state": "detached",
            "stage": "starting",
            "pid": process.pid,
            "updated_unix": time.time(),
            "log": str(log_path),
        },
    )
    return process.pid
