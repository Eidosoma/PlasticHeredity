"""Stage-1 clean-room motif-carrier upper bound for Life-like CA.

The experiment is deliberately narrower than a plastic-heredity claim.  It
asks whether a translation-equivariant summary of a mature parent's local
spacetime motifs can control a visibly reset daughter.  Multigenerational
renewal and the full causal ladder are reserved for later, gated stages.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import numpy as np

from .causal_heredity import (
    _atomic_json,
    _atomic_text,
    _component_spectrum,
    _hash_seed,
    _sha256,
    _state_from_hex,
    launch_detached,
)
from .e19 import require_pinned_numpy
from .life_family import _life_like_step_lookup, _rule_lookups, live_2x2_counts_batch
from .lineage_field import apply_field_reader, load_round3_pairs


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "CA_MOTIF_LINEAGE_STAGE1_PROTOCOL.md"
ROUND3_ROOT = ROOT / "results/ca-carrier-round-3"
ROUND4_CALIBRATION = ROOT / "results/ca-lineage-field-round-4/CALIBRATION.json"
RULE = 31649
FAMILIES = ("contextual256", "motif_energy512")
VALIDATION_CONDITIONS = (
    "intact",
    "zero",
    "read_disabled",
    "shuffle",
    "opposite_history",
    "unrelated_pair",
    "process_noise",
    "carrier_corruption_1",
    "spatial_latch",
    "visible64",
)


@dataclass(frozen=True)
class MotifContract:
    implementation_version: str = "ca-motif-lineage-cleanroom-v1"
    namespace: str = "plastic-ca-motif-lineage-stage1-v1"
    rule: int = RULE
    width: int = 16
    height: int = 16
    horizon: int = 64
    checkpoints: tuple[int, ...] = (8, 16, 32, 64)
    observation_window: int = 8
    jeffreys_alpha: float = 0.5
    energy_clip: float = 4.0
    process_noise: float = 0.002
    carrier_corruption: float = 0.01
    assignment_similarity: float = 0.90
    assignment_margin: float = 0.05
    crossover_gate: float = 0.15
    robust_crossover_gate: float = 0.10
    control_advantage_gate: float = 0.10
    survival_gate: float = 0.90
    science_reserve_seconds: float = 1800.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "visible_reset": "bitwise-identical neutral board across histories",
                "writer_access": "local visible spacetime motifs only; no labels or prototypes",
                "reader_access": "current local motif address and inherited carrier only",
                "primary_observer": "trailing-eight-sweep accumulated live 2x2 texture",
                "missing_policy": "dead and unresolved futures remain in denominators",
                "claim_boundary": "controllability upper bound; not renewed plastic heredity",
            }
        )
        return payload

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MotifProfile:
    calibration_pairs: int
    discovery_pairs: int
    validation_pairs: int
    screen_replicates: int
    validation_replicates: int
    bootstrap_resamples: int
    write_windows: tuple[int, ...]
    strengths: tuple[float, ...]
    read_durations: tuple[int, ...]
    nominees_per_family: int = 2


MOTIF_PROFILES: dict[str, MotifProfile] = {
    "smoke": MotifProfile(2, 2, 2, 2, 2, 100, (16,), (0.5, 1.0), (8, 16)),
    "pilot": MotifProfile(16, 12, 16, 4, 8, 1_000, (16, 32), (0.25, 0.5, 1.0), (8, 16, 32)),
    "reference": MotifProfile(64, 48, 64, 16, 64, 10_000, (16, 32), (0.25, 0.5, 0.75, 1.0), (8, 16, 32, 64)),
}
PUBLIC_PROFILES = tuple(MOTIF_PROFILES)


@dataclass(frozen=True)
class ReaderConfiguration:
    family: str
    write_window: int
    strength: float
    read_duration: int

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown motif family {self.family!r}")
        if self.write_window not in (16, 32):
            raise ValueError("write window must be 16 or 32")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("read strength must be in [0, 1]")
        if self.read_duration not in (8, 16, 32, 64):
            raise ValueError("read duration must be 8, 16, 32, or 64")

    @property
    def id(self) -> str:
        strength = f"{int(round(self.strength * 100)):03d}"
        return f"{self.family}-w{self.write_window:02d}-s{strength}-d{self.read_duration:02d}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "configuration_id": self.id}


def _configurations(profile: MotifProfile) -> list[ReaderConfiguration]:
    return [
        ReaderConfiguration(family, window, strength, duration)
        for family in FAMILIES
        for window in profile.write_windows
        for strength in profile.strengths
        for duration in profile.read_durations
    ]


def _step(states: np.ndarray, rule: int = RULE) -> np.ndarray:
    return _life_like_step_lookup(states, *_rule_lookups(rule))


_OFFSETS_3X3 = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
_RING_OFFSETS = tuple(offset for offset in _OFFSETS_3X3 if offset != (0, 0))


def motif3_codes(states: np.ndarray) -> np.ndarray:
    """Return one nine-bit 3x3 code at every toroidal lattice site."""

    states = np.asarray(states, dtype=np.bool_)
    if states.ndim != 3:
        raise ValueError("states must have shape (sample, height, width)")
    codes = np.zeros(states.shape, dtype=np.uint16)
    for bit, (dy, dx) in enumerate(_OFFSETS_3X3):
        neighbour = np.roll(states, shift=(-dy, -dx), axis=(1, 2))
        codes |= neighbour.astype(np.uint16) << bit
    return codes


def context8_codes(states: np.ndarray) -> np.ndarray:
    """Return the eight-neighbour address, excluding the centre bit."""

    states = np.asarray(states, dtype=np.bool_)
    if states.ndim != 3:
        raise ValueError("states must have shape (sample, height, width)")
    codes = np.zeros(states.shape, dtype=np.uint16)
    for bit, (dy, dx) in enumerate(_RING_OFFSETS):
        neighbour = np.roll(states, shift=(-dy, -dx), axis=(1, 2))
        codes |= neighbour.astype(np.uint16) << bit
    return codes


def _bincount_rows(codes: np.ndarray, size: int) -> np.ndarray:
    return np.stack(
        [np.bincount(row.ravel(), minlength=size) for row in codes], axis=0
    ).astype(np.float64)


def collect_trajectory_counts(
    founders: np.ndarray,
    windows: Sequence[int],
    *,
    rule: int = RULE,
) -> dict[int, dict[str, np.ndarray]]:
    """Accumulate context and motif counts over post-update founder sweeps."""

    state = np.asarray(founders, dtype=np.bool_).copy()
    if state.ndim != 3:
        raise ValueError("founders must have shape (sample, height, width)")
    requested = tuple(sorted(set(int(value) for value in windows)))
    if not requested or requested[0] <= 0:
        raise ValueError("at least one positive write window is required")
    context_alive = np.zeros((len(state), 256), dtype=np.float64)
    context_total = np.zeros((len(state), 256), dtype=np.float64)
    motif = np.zeros((len(state), 512), dtype=np.float64)
    snapshots: dict[int, dict[str, np.ndarray]] = {}
    for sweep in range(1, requested[-1] + 1):
        state = _step(state, rule)
        contexts = context8_codes(state)
        motifs = motif3_codes(state)
        context_total += _bincount_rows(contexts, 256)
        for index in range(len(state)):
            alive_codes = contexts[index][state[index]]
            context_alive[index] += np.bincount(alive_codes, minlength=256)
        motif += _bincount_rows(motifs, 512)
        if sweep in requested:
            snapshots[sweep] = {
                "context_alive": context_alive.copy(),
                "context_total": context_total.copy(),
                "motif": motif.copy(),
                "terminal": state.copy(),
            }
    return snapshots


def build_reference(
    calibration_pairs: Sequence[dict[str, Any]],
    windows: Sequence[int],
    contract: MotifContract,
) -> dict[int, dict[str, np.ndarray]]:
    """Build label-blind reference tables by pooling both histories."""

    founders = []
    for pair in calibration_pairs:
        founders.extend(
            (
                _state_from_hex("life", pair["donor_a"]["donor_state_hex"]),
                _state_from_hex("life", pair["donor_b"]["donor_state_hex"]),
            )
        )
    counts = collect_trajectory_counts(np.stack(founders), windows, rule=contract.rule)
    result: dict[int, dict[str, np.ndarray]] = {}
    alpha = contract.jeffreys_alpha
    for window, values in counts.items():
        alive = values["context_alive"].sum(axis=0)
        total = values["context_total"].sum(axis=0)
        context_probability = (alive + alpha) / (total + 2.0 * alpha)
        motif = values["motif"].sum(axis=0)
        motif_probability = (motif + alpha) / (motif.sum() + 512.0 * alpha)
        result[window] = {
            "context_probability": context_probability,
            "motif_probability": motif_probability,
        }
    return result


def write_parent_carriers(
    founders: np.ndarray,
    windows: Sequence[int],
    reference: dict[int, dict[str, np.ndarray]],
    contract: MotifContract,
) -> dict[int, dict[str, np.ndarray]]:
    """Write both registered carriers without form labels or target access."""

    counts = collect_trajectory_counts(founders, windows, rule=contract.rule)
    alpha = contract.jeffreys_alpha
    result: dict[int, dict[str, np.ndarray]] = {}
    for window, values in counts.items():
        alive = values["context_alive"]
        total = values["context_total"]
        conditional = (alive + alpha) / (total + 2.0 * alpha)
        contextual_marks = conditional - reference[window]["context_probability"][None, :]
        motif = values["motif"]
        motif_probability = (motif + alpha) / (
            motif.sum(axis=1, keepdims=True) + 512.0 * alpha
        )
        energy_marks = np.log(motif_probability) - np.log(
            reference[window]["motif_probability"][None, :]
        )
        result[window] = {
            "contextual256": contextual_marks.astype(np.float32),
            "motif_energy512": np.clip(
                energy_marks, -contract.energy_clip, contract.energy_clip
            ).astype(np.float32),
            "terminal": values["terminal"],
        }
    return result


def _lookup(carrier: np.ndarray, codes: np.ndarray) -> np.ndarray:
    flat = np.take_along_axis(carrier, codes.reshape(len(codes), -1), axis=1)
    return flat.reshape(codes.shape)


def apply_contextual_reader(
    predicted: np.ndarray,
    contexts: np.ndarray,
    carrier: np.ndarray,
    uniforms: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Apply signed context marks; an all-zero carrier is exactly inert."""

    result = np.asarray(predicted, dtype=np.bool_).copy()
    marks = _lookup(carrier, contexts)
    probability = np.clip(np.abs(marks) * strength, 0.0, 1.0)
    result[(marks > 0.0) & (~result) & (uniforms < probability)] = True
    result[(marks < 0.0) & result & (uniforms < probability)] = False
    return result


def motif_energy_advantage(predicted: np.ndarray, carrier: np.ndarray) -> np.ndarray:
    """Energy gain from flipping each cell, evaluated over its nine local motifs."""

    codes = motif3_codes(predicted)
    advantage = np.zeros(predicted.shape, dtype=np.float32)
    for bit, (dy, dx) in enumerate(_OFFSETS_3X3):
        affected = np.roll(codes, shift=(dy, dx), axis=(1, 2))
        current = _lookup(carrier, affected)
        flipped = _lookup(carrier, affected ^ np.uint16(1 << bit))
        advantage += flipped - current
    return advantage


def apply_energy_reader(
    predicted: np.ndarray,
    carrier: np.ndarray,
    uniforms: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Synchronously flip cells only when the inherited motif energy favours it."""

    result = np.asarray(predicted, dtype=np.bool_).copy()
    advantage = motif_energy_advantage(result, carrier)
    probability = strength * np.tanh(np.maximum(advantage, 0.0) / 9.0)
    result[uniforms < probability] ^= True
    return result


def _paired_uniforms(pair_id: str, purpose: str, sweep: int, replicates: int) -> np.ndarray:
    rng = np.random.default_rng(_hash_seed("motif-lineage", pair_id, purpose, sweep))
    half = rng.random((replicates, 16, 16))
    return np.concatenate((half, half), axis=0)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values, dtype=np.float64), where=totals > 0)


def _cosine_labels(
    vectors: np.ndarray,
    target_a: Sequence[float],
    target_b: Sequence[float],
    contract: MotifContract,
) -> np.ndarray:
    a = np.asarray(target_a, dtype=np.float64)
    b = np.asarray(target_b, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    sim_a = np.divide(vectors @ a, norms * np.linalg.norm(a), out=np.zeros(len(vectors)), where=norms > 0)
    sim_b = np.divide(vectors @ b, norms * np.linalg.norm(b), out=np.zeros(len(vectors)), where=norms > 0)
    labels = np.zeros(len(vectors), dtype=np.int8)
    labels[(sim_a >= contract.assignment_similarity) & (sim_a - sim_b >= contract.assignment_margin)] = 1
    labels[(sim_b >= contract.assignment_similarity) & (sim_b - sim_a >= contract.assignment_margin)] = -1
    return labels


def _outcome(labels: np.ndarray, alive: np.ndarray, replicates: int) -> dict[str, float]:
    first = slice(0, replicates)
    second = slice(replicates, 2 * replicates)
    p_a_a = float(np.count_nonzero((labels[first] == 1) & alive[first]) / replicates)
    p_b_a = float(np.count_nonzero((labels[first] == -1) & alive[first]) / replicates)
    p_a_b = float(np.count_nonzero((labels[second] == 1) & alive[second]) / replicates)
    p_b_b = float(np.count_nonzero((labels[second] == -1) & alive[second]) / replicates)
    direction_a = p_a_a - p_a_b
    direction_b = p_b_b - p_b_a
    return {
        "p_a_given_a": p_a_a,
        "p_b_given_a": p_b_a,
        "p_a_given_b": p_a_b,
        "p_b_given_b": p_b_b,
        "direction_a": direction_a,
        "direction_b": direction_b,
        "crossover": min(direction_a, direction_b),
        "correct": 0.5 * (p_a_a + p_b_b),
        "resolved": 0.5 * (p_a_a + p_b_a + p_a_b + p_b_b),
    }


def _texture_descriptor(states: np.ndarray) -> np.ndarray:
    values: list[list[float]] = []
    for board in states.astype(np.float64):
        occupancy = float(board.mean())
        correlations = [
            float(np.mean(board * np.roll(board, shift, axis=(0, 1))))
            for shift in ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2))
        ]
        spectrum = np.abs(np.fft.fft2(board)) ** 2 / float(board.size**2)
        low_frequency = [
            float(spectrum[index]) for index in ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0))
        ]
        values.append([occupancy, *correlations, *low_frequency])
    return np.asarray(values, dtype=np.float64)


def _nearest_labels(vectors: np.ndarray, target_a: np.ndarray, target_b: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(target_a) + np.abs(target_b), 1e-6)
    distance_a = np.linalg.norm((vectors - target_a) / scale, axis=1)
    distance_b = np.linalg.norm((vectors - target_b) / scale, axis=1)
    return np.where(distance_a < distance_b, 1, np.where(distance_b < distance_a, -1, 0)).astype(np.int8)


def _score_checkpoint(
    state: np.ndarray,
    accumulated: np.ndarray,
    pair: dict[str, Any],
    founder_terminal: np.ndarray,
    replicates: int,
    contract: MotifContract,
    *,
    diagnostics: bool,
) -> dict[str, Any]:
    alive = state.any(axis=(1, 2))
    targets = pair["targets"]
    primary_vectors = _normalize_rows(accumulated)
    terminal_vectors = _normalize_rows(live_2x2_counts_batch(state))
    result: dict[str, Any] = {
        "survival": float(np.count_nonzero(alive) / len(alive)),
        "primary": _outcome(
            _cosine_labels(primary_vectors, targets["primary"]["A"], targets["primary"]["B"], contract),
            alive,
            replicates,
        ),
        "terminal": _outcome(
            _cosine_labels(
                terminal_vectors,
                targets.get("primary_terminal", targets["primary"])["A"],
                targets.get("primary_terminal", targets["primary"])["B"],
                contract,
            ),
            alive,
            replicates,
        ),
    }
    if diagnostics:
        take = min(replicates, 8)
        indices = np.r_[0:take, replicates : replicates + take]
        component_vectors = np.stack([_component_spectrum(board) for board in state[indices]])
        if "components" in targets:
            component_labels = _cosine_labels(
                    component_vectors,
                    targets["components"]["A"],
                    targets["components"]["B"],
                    contract,
            )
        else:
            founder_components = np.stack(
                [_component_spectrum(board) for board in founder_terminal]
            )
            component_labels = _nearest_labels(
                component_vectors, founder_components[0], founder_components[1]
            )
        result["components"] = _outcome(
            component_labels,
            alive[indices],
            take,
        )
        descriptor = _texture_descriptor(state[indices])
        founder_descriptor = _texture_descriptor(founder_terminal)
        result["texture_diagnostic"] = _outcome(
            _nearest_labels(descriptor, founder_descriptor[0], founder_descriptor[1]),
            alive[indices],
            take,
        )
        result["diagnostic_replicates_per_history"] = take
    return result


def _repeat_histories(values: np.ndarray, replicates: int) -> np.ndarray:
    return np.concatenate(
        (np.repeat(values[0:1], replicates, axis=0), np.repeat(values[1:2], replicates, axis=0)),
        axis=0,
    )


def _spatial_latch_carrier(
    founders: np.ndarray, write_window: int, parameters: dict[str, Any], contract: MotifContract
) -> np.ndarray:
    state = founders.copy()
    occupancy = np.zeros(state.shape, dtype=np.float32)
    for _ in range(write_window):
        state = _step(state, contract.rule)
        occupancy += state
    carrier = np.zeros(state.shape, dtype=np.float32)
    carrier[occupancy / write_window >= float(parameters["upper"])] = 1.0
    carrier[occupancy / write_window <= float(parameters["lower"])] = -1.0
    return carrier * float(parameters["decay"])


def simulate_pair_condition(
    pair: dict[str, Any],
    configuration: ReaderConfiguration,
    written: dict[str, np.ndarray],
    condition: str,
    replicates: int,
    contract: MotifContract,
    *,
    unrelated_carrier: np.ndarray | None = None,
    spatial_parameters: dict[str, Any] | None = None,
    diagnostics: bool = False,
) -> dict[str, Any]:
    """Run one paired, bitwise-reset daughter counterfactual."""

    pair_id = str(pair["pair_id"])
    reset_a = _state_from_hex("life", pair["donor_a"]["initial_state_hex"])
    reset_b = _state_from_hex("life", pair["donor_b"]["initial_state_hex"])
    if not np.array_equal(reset_a, reset_b):
        raise AssertionError(f"visible reset mismatch in pair {pair_id}")
    state = np.repeat(reset_a[None, ...], 2 * replicates, axis=0)
    base_carrier = written[configuration.family]
    carrier = _repeat_histories(base_carrier, replicates)
    if condition in ("zero", "read_disabled"):
        carrier.fill(0.0)
    elif condition == "shuffle":
        permutation = np.random.default_rng(
            _hash_seed(contract.namespace, pair_id, configuration.family, "shuffle")
        ).permutation(carrier.shape[1])
        carrier = carrier[:, permutation]
    elif condition == "opposite_history":
        carrier = np.concatenate((carrier[replicates:], carrier[:replicates]), axis=0)
    elif condition == "unrelated_pair":
        if unrelated_carrier is None:
            raise ValueError("unrelated_pair requires a carrier")
        carrier = _repeat_histories(unrelated_carrier, replicates)
    elif condition == "carrier_corruption_1":
        mask = np.random.default_rng(
            _hash_seed(contract.namespace, pair_id, configuration.family, "corruption")
        ).random((replicates, carrier.shape[1])) < contract.carrier_corruption
        mask = np.concatenate((mask, mask), axis=0)
        carrier[mask] *= -1.0

    founder_terminal = written["terminal"]
    spatial = None
    if condition == "spatial_latch":
        if spatial_parameters is None:
            raise ValueError("spatial_latch requires frozen Round-4 parameters")
        founders = np.stack(
            (
                _state_from_hex("life", pair["donor_a"]["donor_state_hex"]),
                _state_from_hex("life", pair["donor_b"]["donor_state_hex"]),
            )
        )
        spatial = _repeat_histories(
            _spatial_latch_carrier(founders, configuration.write_window, spatial_parameters, contract),
            replicates,
        )
    if condition == "visible64":
        indices = np.random.default_rng(
            _hash_seed(contract.namespace, pair_id, "visible64")
        ).choice(256, size=64, replace=False)
        terminals = _repeat_histories(founder_terminal, replicates).reshape(2 * replicates, 256)
        flat = state.reshape(2 * replicates, 256)
        flat[:, indices] = terminals[:, indices]

    read_enabled = condition not in ("zero", "read_disabled", "visible64")
    process_noise = contract.process_noise if condition == "process_noise" else 0.0
    recent: deque[np.ndarray] = deque(maxlen=contract.observation_window)
    outcomes: dict[str, Any] = {}
    for sweep in range(1, contract.horizon + 1):
        contexts = context8_codes(state) if configuration.family == "contextual256" else None
        predicted = _step(state, contract.rule)
        if condition == "spatial_latch" and sweep <= 8:
            uniforms = _paired_uniforms(pair_id, "read", sweep, replicates)
            predicted = apply_field_reader(
                predicted, spatial, uniforms, float(spatial_parameters["kappa"])
            )
        elif read_enabled and sweep <= configuration.read_duration:
            uniforms = _paired_uniforms(pair_id, "read", sweep, replicates)
            if configuration.family == "contextual256":
                predicted = apply_contextual_reader(
                    predicted, contexts, carrier, uniforms, configuration.strength
                )
            else:
                predicted = apply_energy_reader(
                    predicted, carrier, uniforms, configuration.strength
                )
        if process_noise:
            predicted ^= _paired_uniforms(pair_id, "process", sweep, replicates) < process_noise
        state = predicted
        recent.append(live_2x2_counts_batch(state))
        if sweep in contract.checkpoints:
            outcomes[str(sweep)] = _score_checkpoint(
                state,
                np.sum(np.stack(tuple(recent)), axis=0),
                pair,
                founder_terminal,
                replicates,
                contract,
                diagnostics=diagnostics,
            )
    return {
        "reset_sha256": hashlib.sha256(reset_a.tobytes()).hexdigest(),
        "reset_asserted_identical": True,
        "condition": condition,
        "carrier_mean_abs": float(np.mean(np.abs(carrier))) if carrier.size else 0.0,
        "outcomes": outcomes,
    }


def _founders(pair: dict[str, Any]) -> np.ndarray:
    return np.stack(
        (
            _state_from_hex("life", pair["donor_a"]["donor_state_hex"]),
            _state_from_hex("life", pair["donor_b"]["donor_state_hex"]),
        )
    )


def _screen_task(payload: tuple[dict[str, Any], MotifContract, dict[int, dict[str, np.ndarray]]]) -> dict[str, Any]:
    item, contract, reference = payload
    pair = item["pair"]
    configurations = [ReaderConfiguration(**row) for row in item["configurations"]]
    windows = sorted({configuration.write_window for configuration in configurations})
    written_by_window = write_parent_carriers(_founders(pair), windows, reference, contract)
    results: dict[str, Any] = {}
    for configuration in configurations:
        outcomes = simulate_pair_condition(
            pair,
            configuration,
            written_by_window[configuration.write_window],
            "intact",
            int(item["replicates"]),
            contract,
        )
        results[configuration.id] = {
            "configuration": configuration.to_dict(),
            **outcomes,
        }
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": pair["pair_id"],
        "phase": "screen",
        "replicates": int(item["replicates"]),
        "results": results,
    }


def _validation_task(payload: tuple[dict[str, Any], MotifContract, dict[int, dict[str, np.ndarray]]]) -> dict[str, Any]:
    item, contract, reference = payload
    pair = item["pair"]
    unrelated = item["unrelated_pair"]
    nominees = [ReaderConfiguration(**row) for row in item["nominees"]]
    windows = sorted({configuration.write_window for configuration in nominees})
    written_by_window = write_parent_carriers(_founders(pair), windows, reference, contract)
    unrelated_by_window = write_parent_carriers(_founders(unrelated), windows, reference, contract)
    results: dict[str, Any] = {}
    for configuration in nominees:
        written = written_by_window[configuration.write_window]
        unrelated_written = unrelated_by_window[configuration.write_window]
        conditions: dict[str, Any] = {}
        for condition in VALIDATION_CONDITIONS:
            conditions[condition] = simulate_pair_condition(
                pair,
                configuration,
                written,
                condition,
                int(item["replicates"]),
                contract,
                unrelated_carrier=unrelated_written[configuration.family],
                spatial_parameters=item["spatial_parameters"],
                diagnostics=True,
            )
        results[configuration.id] = {
            "configuration": configuration.to_dict(),
            "conditions": conditions,
        }
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": pair["pair_id"],
        "unrelated_pair_id": unrelated["pair_id"],
        "phase": "validation",
        "replicates": int(item["replicates"]),
        "results": results,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def select_cohorts(profile: MotifProfile, contract: MotifContract) -> dict[str, list[dict[str, Any]]]:
    rows = load_round3_pairs()[contract.rule]
    ordered = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{contract.namespace}:cohort:{row['pair_id']}".encode()).hexdigest(),
            row["pair_id"],
        ),
    )
    counts = (profile.calibration_pairs, profile.discovery_pairs, profile.validation_pairs)
    if sum(counts) > len(ordered):
        raise ValueError("not enough frozen Rule-31649 pairs for disjoint cohorts")
    first = counts[0]
    second = first + counts[1]
    third = second + counts[2]
    cohorts = {
        "calibration": ordered[:first],
        "discovery": ordered[first:second],
        "validation": ordered[second:third],
    }
    ids = [row["pair_id"] for values in cohorts.values() for row in values]
    if len(ids) != len(set(ids)):
        raise AssertionError("motif-lineage cohorts overlap")
    return cohorts


def _reference_to_json(reference: dict[int, dict[str, np.ndarray]]) -> dict[str, Any]:
    return {
        str(window): {name: values.tolist() for name, values in tables.items()}
        for window, tables in reference.items()
    }


def _reference_from_json(payload: dict[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    return {
        int(window): {name: np.asarray(values, dtype=np.float64) for name, values in tables.items()}
        for window, tables in payload.items()
    }


def _run_checkpoints(
    output: Path,
    phase: str,
    items: Sequence[dict[str, Any]],
    contract: MotifContract,
    reference: dict[int, dict[str, np.ndarray]],
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status: Any,
) -> tuple[list[dict[str, Any]], bool]:
    root = output / phase / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for item in items:
        path = root / f"{item['checkpoint']}.json"
        if resume and path.exists():
            payload = _load_json(path)
            if payload.get("design_digest") != design_digest:
                raise ValueError(f"checkpoint design mismatch: {path}")
            results[item["checkpoint"]] = payload["result"]
        else:
            missing.append(item)
    initial = len(results)
    phase_started = time.monotonic()
    truncated = False
    task = _screen_task if phase == "screen" else _validation_task

    def save(item: dict[str, Any], result: dict[str, Any]) -> None:
        key = item["checkpoint"]
        _atomic_json(
            root / f"{key}.json",
            {"design_digest": design_digest, "phase": phase, "checkpoint": key, "result": result},
        )
        results[key] = result
        completed_new = max(1, len(results) - initial)
        elapsed = max(time.monotonic() - phase_started, 1e-6)
        eta = elapsed / completed_new * max(0, len(items) - len(results))
        status(
            "running",
            phase,
            completed=len(results),
            total=len(items),
            eta_seconds=eta,
            latest_checkpoint=key,
        )

    if missing and time.time() < deadline:
        pool = ProcessPoolExecutor(max_workers=max(1, min(workers, len(missing))))
        iterator = iter(missing)
        pending: dict[Any, dict[str, Any]] = {}

        def submit_one() -> bool:
            if time.time() >= deadline:
                return False
            try:
                item = next(iterator)
            except StopIteration:
                return False
            pending[pool.submit(task, (item, contract, reference))] = item
            return True

        for _ in range(min(len(missing), max(1, workers * 2))):
            submit_one()
        try:
            while pending:
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    truncated = True
                    break
                done, _ = wait(tuple(pending), timeout=min(10.0, remaining), return_when=FIRST_COMPLETED)
                for future in done:
                    item = pending.pop(future)
                    save(item, future.result())
                    submit_one()
            if truncated:
                for future in pending:
                    future.cancel()
        finally:
            pool.shutdown(wait=True, cancel_futures=truncated)
    elif missing:
        truncated = True

    complete = len(results) == len(items)
    _atomic_json(
        output / phase / "stage_summary.json",
        {
            "phase": phase,
            "design_digest": design_digest,
            "complete": complete,
            "completed": len(results),
            "total": len(items),
            "budget_truncated": truncated or not complete,
        },
    )
    if complete:
        _atomic_text(output / phase / "COMPLETE", "complete\n")
    return [results[key] for key in sorted(results)], complete


def _metric(
    rows: Sequence[dict[str, Any]], configuration_id: str, condition: str, checkpoint: int,
    observer: str, field: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        candidate = row["results"].get(configuration_id)
        if not candidate:
            continue
        payload = candidate if row["phase"] == "screen" else candidate["conditions"].get(condition)
        if not payload:
            continue
        outcome = payload["outcomes"].get(str(checkpoint))
        if not outcome:
            continue
        if field == "survival":
            values.append(float(outcome["survival"]))
        elif observer in outcome and field in outcome[observer]:
            values.append(float(outcome[observer][field]))
    return values


def _paired_difference(
    rows: Sequence[dict[str, Any]], configuration_id: str, left: str, right: str,
    checkpoint: int, observer: str = "primary",
) -> list[float]:
    result: list[float] = []
    for row in rows:
        candidate = row["results"].get(configuration_id)
        if not candidate:
            continue
        try:
            a = candidate["conditions"][left]["outcomes"][str(checkpoint)][observer]["crossover"]
            b = candidate["conditions"][right]["outcomes"][str(checkpoint)][observer]["crossover"]
        except KeyError:
            continue
        result.append(float(a) - float(b))
    return result


def _bootstrap(values: Sequence[float], resamples: int, seed: int, alpha: float = 0.05) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return {"n_pairs": 0, "mean": None, "ci": [None, None], "alpha": alpha}
    rng = np.random.default_rng(seed)
    if len(data) == 1:
        samples = np.repeat(data[0], resamples)
    else:
        indices = rng.integers(0, len(data), size=(resamples, len(data)))
        samples = data[indices].mean(axis=1)
    return {
        "n_pairs": len(data),
        "mean": float(data.mean()),
        "ci": [float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))],
        "alpha": alpha,
    }


def select_nominees(
    rows: Sequence[dict[str, Any]], configurations: Sequence[ReaderConfiguration], profile: MotifProfile
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    nominees: list[dict[str, Any]] = []
    for configuration in configurations:
        checkpoints: dict[str, Any] = {}
        ranked: list[tuple[tuple[float, ...], int]] = []
        for checkpoint in (8, 16, 32, 64):
            crossover = _metric(rows, configuration.id, "intact", checkpoint, "primary", "crossover")
            survival = _metric(rows, configuration.id, "intact", checkpoint, "primary", "survival")
            terminal = _metric(rows, configuration.id, "intact", checkpoint, "terminal", "crossover")
            direction_a = _metric(rows, configuration.id, "intact", checkpoint, "primary", "direction_a")
            direction_b = _metric(rows, configuration.id, "intact", checkpoint, "primary", "direction_b")
            summary = {
                "crossover_mean": float(np.mean(crossover)) if crossover else None,
                "survival_mean": float(np.mean(survival)) if survival else None,
                "terminal_crossover_mean": float(np.mean(terminal)) if terminal else None,
                "direction_a_mean": float(np.mean(direction_a)) if direction_a else None,
                "direction_b_mean": float(np.mean(direction_b)) if direction_b else None,
                "fraction_pairs_positive": float(np.mean(np.asarray(crossover) > 0.0)) if crossover else None,
                "n_pairs": len(crossover),
            }
            checkpoints[str(checkpoint)] = summary
            if crossover:
                stability = min(
                    float(np.mean(_metric(rows, configuration.id, "intact", later, "primary", "crossover")))
                    for later in (8, 16, 32, 64)
                    if _metric(rows, configuration.id, "intact", later, "primary", "crossover")
                )
                rank = (
                    summary["crossover_mean"] or -2.0,
                    summary["survival_mean"] or 0.0,
                    summary["terminal_crossover_mean"] or -2.0,
                    summary["fraction_pairs_positive"] or 0.0,
                    stability,
                    -float(checkpoint),
                )
                ranked.append((rank, checkpoint))
        best_rank, selected_checkpoint = max(ranked, default=((-2.0,), 8))
        summaries[configuration.id] = {
            "configuration": configuration.to_dict(),
            "checkpoints": checkpoints,
            "selected_checkpoint": selected_checkpoint,
            "rank": list(best_rank),
        }
    for family in FAMILIES:
        eligible = [value for value in summaries.values() if value["configuration"]["family"] == family]
        ordered = sorted(
            eligible,
            key=lambda value: (
                tuple(value["rank"]),
                -int(value["configuration"]["read_duration"]),
                -float(value["configuration"]["strength"]),
                -int(value["configuration"]["write_window"]),
                value["configuration"]["configuration_id"],
            ),
            reverse=True,
        )
        nominees.extend(ordered[: profile.nominees_per_family])
    return {
        "selection_data": "discovery cohort only",
        "nominees_per_family": profile.nominees_per_family,
        "nominees": nominees,
        "all_configurations": summaries,
    }


def _positive_bound(summary: dict[str, Any]) -> bool:
    return summary["ci"][0] is not None and float(summary["ci"][0]) > 0.0


def adjudicate_validation(
    rows: Sequence[dict[str, Any]], selection: dict[str, Any], profile: MotifProfile, complete: bool
) -> dict[str, Any]:
    if not complete:
        return {"verdict": "INCOMPLETE", "nominees": {}, "passing_configurations": []}
    nominees: dict[str, Any] = {}
    passing: list[str] = []
    robust_passing: list[str] = []
    alpha = 0.05 / max(1, len(selection["nominees"]))
    for nominee in selection["nominees"]:
        configuration = nominee["configuration"]
        identifier = configuration["configuration_id"]
        checkpoint = int(nominee["selected_checkpoint"])

        def boot(values: Sequence[float], name: str) -> dict[str, Any]:
            return _bootstrap(
                values,
                profile.bootstrap_resamples,
                _hash_seed("motif-stage1-gate", identifier, checkpoint, name),
                alpha,
            )

        intact = boot(_metric(rows, identifier, "intact", checkpoint, "primary", "crossover"), "intact")
        survival = boot(_metric(rows, identifier, "intact", checkpoint, "primary", "survival"), "survival")
        directions_a = _metric(rows, identifier, "intact", checkpoint, "primary", "direction_a")
        directions_b = _metric(rows, identifier, "intact", checkpoint, "primary", "direction_b")
        pair_values = _metric(rows, identifier, "intact", checkpoint, "primary", "crossover")
        controls = {
            control: boot(_paired_difference(rows, identifier, "intact", control, checkpoint), f"adv-{control}")
            for control in ("zero", "shuffle", "read_disabled")
        }
        opposite = boot(
            _metric(rows, identifier, "opposite_history", checkpoint, "primary", "crossover"),
            "opposite",
        )
        independent = {
            observer: boot(_metric(rows, identifier, "intact", checkpoint, observer, "crossover"), observer)
            for observer in ("terminal", "components", "texture_diagnostic")
        }
        process = boot(
            _metric(rows, identifier, "process_noise", checkpoint, "primary", "crossover"),
            "process-noise",
        )
        corruption = boot(
            _metric(rows, identifier, "carrier_corruption_1", checkpoint, "primary", "crossover"),
            "carrier-corruption",
        )
        direction_a = float(np.mean(directions_a)) if directions_a else 0.0
        direction_b = float(np.mean(directions_b)) if directions_b else 0.0
        fraction_positive = float(np.mean(np.asarray(pair_values) > 0.0)) if pair_values else 0.0
        independent_pass = any(
            float(value["mean"] or 0.0) > 0.0 and _positive_bound(value)
            for value in independent.values()
        )
        opposite_pass = bool(
            float(opposite["mean"] or 0.0) <= -0.10
            and opposite["ci"][1] is not None
            and float(opposite["ci"][1]) < 0.0
        )
        gate = bool(
            float(intact["mean"] or 0.0) >= 0.15
            and _positive_bound(intact)
            and direction_a > 0.0
            and direction_b > 0.0
            and fraction_positive >= 0.50
            and float(survival["mean"] or 0.0) >= 0.90
            and all(
                float(value["mean"] or 0.0) >= 0.10 and _positive_bound(value)
                for value in controls.values()
            )
            and opposite_pass
            and independent_pass
        )
        robust_gate = bool(
            gate
            and float(process["mean"] or 0.0) >= 0.10
            and _positive_bound(process)
            and float(corruption["mean"] or 0.0) >= 0.10
            and _positive_bound(corruption)
        )
        if gate:
            passing.append(identifier)
        if robust_gate:
            robust_passing.append(identifier)
        nominees[identifier] = {
            "configuration": configuration,
            "checkpoint": checkpoint,
            "controllability_gate": gate,
            "robust_gate": robust_gate,
            "intact": intact,
            "survival": survival,
            "direction_a_mean": direction_a,
            "direction_b_mean": direction_b,
            "fraction_pairs_positive": fraction_positive,
            "control_advantages": controls,
            "opposite_history": opposite,
            "opposite_reversal_gate": opposite_pass,
            "independent_observers": independent,
            "independent_observer_gate": independent_pass,
            "process_noise": process,
            "carrier_corruption_1": corruption,
            "diagnostics": {
                condition: boot(
                    _metric(rows, identifier, condition, checkpoint, "primary", "crossover"),
                    f"diagnostic-{condition}",
                )
                for condition in ("unrelated_pair", "spatial_latch", "visible64")
            },
        }
    if robust_passing:
        verdict = "ROBUST_LOCAL_MOTIF_CONTROLLABILITY"
    elif passing:
        verdict = "LOCAL_MOTIF_CONTROLLABILITY"
    else:
        verdict = "NO_MOTIF_CONTROLLABILITY_UNDER_TESTED_READERS"
    return {
        "verdict": verdict,
        "claim_boundary": "one-generation controllability only; not plastic heredity",
        "familywise_interval_alpha": alpha,
        "passing_configurations": passing,
        "robust_passing_configurations": robust_passing,
        "nominees": nominees,
    }


def _choose_stage2_input(adjudication: dict[str, Any]) -> dict[str, Any] | None:
    passing = adjudication.get("passing_configurations", [])
    if not passing:
        return None
    robust = set(adjudication.get("robust_passing_configurations", []))
    candidates = [adjudication["nominees"][identifier] for identifier in passing]
    return max(
        candidates,
        key=lambda value: (
            value["configuration"]["family"] == "contextual256",
            value["configuration"]["configuration_id"] in robust,
            float(value["intact"]["mean"] or -2.0),
            -int(value["configuration"]["read_duration"]),
            -float(value["configuration"]["strength"]),
        ),
    )["configuration"]


def _queue_payload(design_digest: str, state: str = "blocked") -> dict[str, Any]:
    return {
        "programme": "ca_motif_lineage_five_stage",
        "stage1_design_digest": design_digest,
        "automatic_chaining": False,
        "per_stage_max_hours": 8.0,
        "stages": [
            {"stage": 1, "name": "motif_carrier_upper_bound", "state": state},
            {"stage": 2, "name": "freeze_and_generalize_reader", "state": "blocked_pending_stage1_review"},
            {"stage": 3, "name": "renewed_heredity_causal_ladder", "state": "blocked_pending_stage2_review"},
            {"stage": 4, "name": "compression_and_robustness", "state": "blocked_pending_stage3_review"},
            {"stage": 5, "name": "localize_inheritance", "state": "blocked_pending_stage4_review"},
        ],
    }


def _render_report(results: dict[str, Any]) -> str:
    adjudication = results["adjudication"]
    lines = [
        "# CA motif-lineage Stage 1",
        "",
        f"State: **{results['state']}**. Profile: `{results['profile']}`.",
        f"Verdict: **{adjudication['verdict']}**.",
        f"Elapsed: `{results['elapsed_seconds'] / 3600.0:.3f}` wall hours.",
        "",
        "## Validation nominees",
        "",
    ]
    for identifier, value in adjudication.get("nominees", {}).items():
        lines.append(
            f"- `{identifier}` at sweep {value['checkpoint']}: intact crossover "
            f"`{value['intact']['mean']}`, CI `{value['intact']['ci']}`; "
            f"controllability `{value['controllability_gate']}`; robust `{value['robust_gate']}`."
        )
    lines.extend(
        (
            "",
            "## Interpretation boundary",
            "",
            "This stage tests whether a motif-indexed hidden channel can steer one visibly reset daughter. "
            "It does not test daughter rewriting or persistence across generations and therefore cannot, by itself, establish plastic heredity.",
            "",
            "Stages 2--5 remain blocked until this result and its frozen decision artifact are reviewed.",
        )
    )
    return "\n".join(lines) + "\n"


def _render_lay_summary(results: dict[str, Any]) -> str:
    verdict = results["adjudication"]["verdict"]
    if verdict == "ROBUST_LOCAL_MOTIF_CONTROLLABILITY":
        finding = "At least one local motif memory reliably steered a freshly reset daughter, including under the registered noise tests."
    elif verdict == "LOCAL_MOTIF_CONTROLLABILITY":
        finding = "At least one local motif memory steered a freshly reset daughter, but it did not clear every noise test."
    elif verdict == "INCOMPLETE":
        finding = "The wall-time budget ended before the upper-bound test was complete; its checkpoints can be resumed."
    else:
        finding = "Neither tested motif memory could reliably steer a freshly reset daughter under the registered controls."
    return (
        "# Lay summary\n\n"
        f"{finding}\n\n"
        "The parent was summarized as a table of recurring tiny neighbourhood patterns. The ordinary daughter grid was then "
        "reset to exactly the same bits in the A-parent and B-parent comparisons. Only the hidden pattern table differed. "
        "Controls erased it, scrambled its addresses, turned off reading, or supplied the opposite history.\n\n"
        "This first stage asks only whether such a channel can control form. It is not yet heredity: later stages must show "
        "that daughters rewrite the channel, retain it for many generations, lose the effect after ablation, and recover it after rescue.\n"
    )


def _update_discovery_log(results: dict[str, Any]) -> None:
    path = ROOT / "DISCOVERY_LOG_EIDOSOMA_SCIENTIST.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Discovery log\n"
    start = "<!-- ca-motif-lineage-stage-1:start -->"
    end = "<!-- ca-motif-lineage-stage-1:end -->"
    section = "\n".join(
        (
            start,
            "## CA motif-lineage Stage 1",
            "",
            f"Upper-bound verdict: `{results['adjudication']['verdict']}`.",
            f"Profile: `{results['profile']}`; elapsed `{results['elapsed_seconds'] / 3600.0:.3f}` wall hours.",
            "See `results/ca-motif-lineage-stage-1/REPORT.md` and `LAY_SUMMARY.md`.",
            end,
        )
    )
    if start in existing and end in existing:
        prefix, remainder = existing.split(start, 1)
        _, suffix = remainder.split(end, 1)
        updated = prefix.rstrip() + "\n\n" + section + suffix
    else:
        updated = existing.rstrip() + "\n\n" + section + "\n"
    _atomic_text(path, updated)


def run_motif_lineage_stage1(
    output: Path,
    *,
    profile_name: str = "reference",
    workers: int = 20,
    max_hours: float = 8.0,
    resume: bool = False,
) -> dict[str, Any]:
    require_pinned_numpy()
    if profile_name not in PUBLIC_PROFILES:
        raise ValueError(f"unknown motif-lineage profile {profile_name!r}")
    if max_hours <= 0.0 or max_hours > 8.0:
        raise ValueError("motif-lineage max-hours must be in (0, 8]")
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)
    if not ROUND4_CALIBRATION.exists():
        raise FileNotFoundError(ROUND4_CALIBRATION)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hard_deadline = started + max_hours * 3600.0
    contract = MotifContract()
    profile = MOTIF_PROFILES[profile_name]
    reserve = min(contract.science_reserve_seconds, max(60.0, max_hours * 3600.0 * 0.10))
    science_deadline = max(started, hard_deadline - reserve)

    def status(state: str, phase: str, **extra: Any) -> None:
        now = time.time()
        payload = {
            "state": state,
            "stage": "1-upper-bound",
            "phase": phase,
            "profile": profile_name,
            "pid": os.getpid(),
            "started_unix": started,
            "updated_unix": now,
            "elapsed_seconds": now - started,
            "hard_deadline_unix": hard_deadline,
            "science_deadline_unix": science_deadline,
            "deadline_remaining_seconds": max(0.0, hard_deadline - now),
            **extra,
        }
        _atomic_json(output / "STATUS.json", payload)
        progress = f" {extra['completed']}/{extra['total']}" if "completed" in extra else ""
        print(f"[{state}] {phase}{progress}", flush=True)

    try:
        status("running", "design")
        cohorts = select_cohorts(profile, contract)
        configurations = _configurations(profile)
        spatial_parameters = _load_json(ROUND4_CALIBRATION)["mechanisms"]["latch"]["selected"]
        input_paths = [
            PROTOCOL_PATH,
            ROUND3_ROOT / "CALIBRATION.json",
            ROUND3_ROOT / "NARROW_COHORTS.json",
            *(ROUND3_ROOT / f"narrow_acquire/checkpoints/launch-{index}.json" for index in range(4)),
            ROUND4_CALIBRATION,
        ]
        design_payload = {
            "experiment": "ca_motif_lineage_stage_1",
            "contract": contract.to_dict(),
            "profile_name": profile_name,
            "profile": asdict(profile),
            "configurations": [configuration.to_dict() for configuration in configurations],
            "cohorts": {
                name: [pair["pair_id"] for pair in rows] for name, rows in cohorts.items()
            },
            "spatial_latch_benchmark": spatial_parameters,
            "input_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in input_paths},
            "implementation_sha256": {
                "motif_lineage.py": _sha256(Path(__file__)),
                "life_family.py": _sha256(Path(__file__).with_name("life_family.py")),
                "lineage_field.py": _sha256(Path(__file__).with_name("lineage_field.py")),
            },
            "cleanroom_exclusion": "no Wagner or Fable implementation source is read, imported, hashed, or executed",
        }
        design_digest = hashlib.sha256(
            json.dumps(design_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        design = {**design_payload, "design_digest": design_digest}
        design_path = output / "DESIGN.json"
        if resume and design_path.exists():
            previous = _load_json(design_path)
            if previous.get("design_digest") != design_digest:
                raise ValueError("resume design digest mismatch")
        _atomic_json(design_path, design)
        _atomic_json(
            output / "MANIFEST.json",
            {
                "experiment": "ca_motif_lineage_stage_1",
                "stage": 1,
                "profile": profile_name,
                "design_digest": design_digest,
                "contract_digest": contract.digest,
                "workers": workers,
                "max_hours": max_hours,
                "started_unix": started,
                "environment": {
                    "python": sys.version,
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                    "cpu_count": os.cpu_count(),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                },
            },
        )
        _atomic_json(output / "COHORTS.json", {"design_digest": design_digest, "cohorts": design_payload["cohorts"]})
        _atomic_json(output / "QUEUE.json", _queue_payload(design_digest, "running"))

        reference_path = output / "CALIBRATION.json"
        if resume and reference_path.exists():
            calibration_payload = _load_json(reference_path)
            if calibration_payload.get("design_digest") != design_digest:
                raise ValueError("calibration design mismatch")
            reference = _reference_from_json(calibration_payload["reference"])
        else:
            status("running", "label_blind_calibration")
            reference = build_reference(cohorts["calibration"], profile.write_windows, contract)
            _atomic_json(
                reference_path,
                {
                    "design_digest": design_digest,
                    "label_access": False,
                    "pairs": [pair["pair_id"] for pair in cohorts["calibration"]],
                    "reference": _reference_to_json(reference),
                },
            )

        screen_items = [
            {
                "checkpoint": f"discovery-{index:04d}",
                "pair": pair,
                "replicates": profile.screen_replicates,
                "configurations": [
                    {key: value for key, value in configuration.to_dict().items() if key != "configuration_id"}
                    for configuration in configurations
                ],
            }
            for index, pair in enumerate(cohorts["discovery"])
        ]
        status("running", "screen", completed=0, total=len(screen_items))
        screen_rows, screen_complete = _run_checkpoints(
            output,
            "screen",
            screen_items,
            contract,
            reference,
            design_digest,
            workers=workers,
            resume=resume,
            deadline=science_deadline,
            status=status,
        )
        if screen_complete:
            status("running", "selection")
            selection = select_nominees(screen_rows, configurations, profile)
            _atomic_json(output / "SELECTION.json", {"design_digest": design_digest, **selection})
        else:
            selection = {"selection_data": "incomplete", "nominees": [], "all_configurations": {}}
            _atomic_json(output / "SELECTION.json", {"design_digest": design_digest, **selection})

        validation_rows: list[dict[str, Any]] = []
        validation_complete = False
        if screen_complete:
            nominee_configs = [
                {
                    key: value
                    for key, value in nominee["configuration"].items()
                    if key != "configuration_id"
                }
                for nominee in selection["nominees"]
            ]
            validation = cohorts["validation"]
            validation_items = [
                {
                    "checkpoint": f"validation-{index:04d}",
                    "pair": pair,
                    "unrelated_pair": validation[(index + 1) % len(validation)],
                    "replicates": profile.validation_replicates,
                    "nominees": nominee_configs,
                    "spatial_parameters": spatial_parameters,
                }
                for index, pair in enumerate(validation)
            ]
            status("running", "validation", completed=0, total=len(validation_items))
            validation_rows, validation_complete = _run_checkpoints(
                output,
                "validation",
                validation_items,
                contract,
                reference,
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
            )

        status("running", "adjudication")
        adjudication = adjudicate_validation(validation_rows, selection, profile, validation_complete)
        complete = screen_complete and validation_complete
        state = "complete" if complete else "partial_budget_exhausted"
        results = {
            "experiment": "ca_motif_lineage_stage_1",
            "state": state,
            "profile": profile_name,
            "design_digest": design_digest,
            "started_unix": started,
            "completed_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "stage_completeness": {"screen": screen_complete, "validation": validation_complete},
            "adjudication": adjudication,
        }
        _atomic_json(output / "RESULTS.json", results)
        _atomic_text(output / "REPORT.md", _render_report(results))
        _atomic_text(output / "LAY_SUMMARY.md", _render_lay_summary(results))
        selected_for_stage2 = _choose_stage2_input(adjudication)
        decision = {
            "stage": 1,
            "design_digest": design_digest,
            "verdict": adjudication["verdict"],
            "review_required": True,
            "automatic_launch": False,
            "decision": "advance_to_stage_2_after_review" if selected_for_stage2 else "halt_and_replan_dual_attractor_search",
            "selected_stage2_input": selected_for_stage2,
            "claim_boundary": "Stage 1 cannot establish plastic heredity",
        }
        _atomic_json(output / "STAGE_DECISION.json", decision)
        queue = _queue_payload(design_digest, "complete" if complete else "partial_resumable")
        queue["stages"][0]["verdict"] = adjudication["verdict"]
        queue["stages"][1]["state"] = (
            "blocked_pending_human_review" if selected_for_stage2 else "blocked_stage1_gate_failed"
        )
        _atomic_json(output / "QUEUE.json", queue)
        if complete:
            _atomic_text(output / "COMPLETE", "complete\n")
            partial = output / "PARTIAL"
            if partial.exists():
                partial.unlink()
            if profile_name == "reference":
                _update_discovery_log(results)
            status("complete", "campaign", verdict=adjudication["verdict"])
        else:
            _atomic_text(output / "PARTIAL", "wall budget exhausted; resume is supported\n")
            status("partial_budget_exhausted", "campaign", verdict=adjudication["verdict"])
        return results
    except BaseException as error:
        status("failed", "campaign", error=repr(error))
        raise


__all__ = [
    "FAMILIES",
    "MOTIF_PROFILES",
    "MotifContract",
    "PUBLIC_PROFILES",
    "ReaderConfiguration",
    "apply_contextual_reader",
    "apply_energy_reader",
    "build_reference",
    "collect_trajectory_counts",
    "context8_codes",
    "launch_detached",
    "motif3_codes",
    "motif_energy_advantage",
    "run_motif_lineage_stage1",
    "select_cohorts",
    "simulate_pair_condition",
    "write_parent_carriers",
]
