"""Clean-room two-timescale lineage carrier experiment for Life-like CA.

The visible board is reset at every reproduction boundary.  Only a separately
represented, locally written field can transmit history.  This module depends
on the frozen CA round-3 artifacts, never on another substrate's implementation.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np

from .ca_carrier_v3 import V3Contract, pair_prototype_donors
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


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "CA_LINEAGE_FIELD_PROTOCOL.md"
ROUND3_ROOT = ROOT / "results/ca-carrier-round-3"
NARROW_CALIBRATION = ROUND3_ROOT / "CALIBRATION.json"
NARROW_COHORTS = ROUND3_ROOT / "NARROW_COHORTS.json"
WIDE_31648 = ROUND3_ROOT / "wide_holdout_acquire/checkpoints/candidate-001-life-031648.json"
WIDE_70366 = ROUND3_ROOT / "wide_holdout_acquire/checkpoints/candidate-000-life-070366.json"
RULES = (31649, 31648, 70366)
MECHANISMS = ("latch", "diffuse")
CHECKPOINT_GENERATIONS = (1, 2, 4, 8, 16)

CORE_CONDITIONS = (
    "intact",
    "zero",
    "shuffle",
    "read_disabled",
    "founder_write_disabled",
    "no_rewrite",
    "ablate_g2",
    "rescue_same_g3",
    "rescue_opposite_g3",
    "opposite_founder",
)

DIAGNOSTIC_CONDITIONS = (
    "intact",
    "compress_block2",
    "compress_block4",
    "compress_block8",
    "compress_global",
    "random64",
    "random16",
    "random4",
    "carrier_noise_1",
    "carrier_noise_5",
    "visible64",
    "visible16",
)


@dataclass(frozen=True)
class LineageFieldContract:
    implementation_version: str = "ca-lineage-field-cleanroom-v1"
    namespace: str = "plastic-ca-lineage-field-v1"
    width: int = 16
    height: int = 16
    founder_sweeps: int = 16
    generation_sweeps: int = 32
    read_sweeps: int = 8
    write_start: int = 17
    observe_start: int = 25
    process_noise: float = 0.002
    assignment_similarity: float = 0.90
    assignment_margin: float = 0.05
    primary_crossover: float = 0.15
    durable_crossover: float = 0.10
    control_advantage: float = 0.10
    survival_gate: float = 0.90
    science_reserve_seconds: float = 1800.0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "visible_reset": "bitwise matched launch before every descendant generation",
                "writer_inputs": "local visible spacetime state only; no form labels or prototypes",
                "reader_inputs": "local carrier and semantic RNG only; no form labels or prototypes",
                "independent_unit": "matched founder pair",
                "missing_policy": "dead and unresolved futures remain in denominators",
            }
        )
        return value

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LineageFieldProfile:
    primary_pairs: int
    primary_replicates: int
    holdout_pairs: int
    holdout_replicates: int
    diagnostic_pairs: int
    diagnostic_replicates: int
    generations: int
    bootstrap_resamples: int


FIELD_PROFILES: dict[str, LineageFieldProfile] = {
    "smoke": LineageFieldProfile(1, 2, 1, 2, 1, 2, 4, 100),
    "pilot": LineageFieldProfile(4, 8, 3, 4, 4, 4, 8, 1_000),
    "floor": LineageFieldProfile(16, 32, 12, 16, 16, 16, 16, 10_000),
    "reduced": LineageFieldProfile(24, 48, 16, 24, 24, 24, 16, 10_000),
    "reference": LineageFieldProfile(32, 64, 24, 32, 32, 32, 16, 10_000),
}

PUBLIC_PROFILES = ("smoke", "pilot", "reference")


@dataclass(frozen=True)
class MechanismParameters:
    mechanism: str
    kappa: float
    decay: float
    upper: float = 0.60
    lower: float = 0.40
    diffusion: float = 0.04
    write_gain: float = 0.08
    eligible: bool = True
    calibration_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values, dtype=np.float64), where=totals > 0)


def apply_field_reader(
    predicted: np.ndarray,
    carrier: np.ndarray,
    uniforms: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Generic label-blind signed reader used by both mechanisms."""

    result = np.asarray(predicted, dtype=np.bool_).copy()
    magnitude = np.clip(np.abs(carrier) * kappa, 0.0, 1.0)
    excite = (carrier > 0.0) & (~result) & (uniforms < magnitude)
    inhibit = (carrier < 0.0) & result & (uniforms < magnitude)
    result[excite] = True
    result[inhibit] = False
    return result


def latch_write(carrier: np.ndarray, occupancy: np.ndarray, upper: float, lower: float) -> np.ndarray:
    result = np.asarray(carrier, dtype=np.float32).copy()
    result[occupancy >= upper] = 1.0
    result[occupancy <= lower] = -1.0
    return result


def diffuse_write_step(
    carrier: np.ndarray,
    visible: np.ndarray,
    diffusion: float,
    write_gain: float,
) -> np.ndarray:
    neighbours = np.zeros_like(carrier, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                neighbours += np.roll(carrier, shift=(dy, dx), axis=(1, 2))
    neighbours /= 8.0
    result = (
        (1.0 - diffusion - write_gain) * carrier
        + diffusion * neighbours
        + write_gain * (2.0 * visible.astype(np.float32) - 1.0)
    )
    return np.clip(result, -1.0, 1.0).astype(np.float32, copy=False)


def block_compress(carrier: np.ndarray, block: int) -> np.ndarray:
    if carrier.ndim != 3 or carrier.shape[1:] != (16, 16):
        raise ValueError("carrier must have shape (sample, 16, 16)")
    if block not in (1, 2, 4, 8, 16):
        raise ValueError("block must divide the 16-cell extent")
    sample = carrier.shape[0]
    coarse = carrier.reshape(sample, 16 // block, block, 16 // block, block).mean(axis=(2, 4))
    return np.repeat(np.repeat(coarse, block, axis=1), block, axis=2).astype(np.float32)


def random_retain(carrier: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count < 0 or count > 256:
        raise ValueError("retained coordinate count must be between 0 and 256")
    rng = np.random.default_rng(seed)
    indices = rng.choice(256, size=count, replace=False)
    result = np.zeros_like(carrier)
    result.reshape(len(carrier), 256)[:, indices] = carrier.reshape(len(carrier), 256)[:, indices]
    return result


def _semantic_rng(pair_id: str, mechanism: str, purpose: str, *parts: object) -> np.random.Generator:
    return np.random.default_rng(_hash_seed("lineage-field", pair_id, mechanism, purpose, *parts))


def _step_visible(states: np.ndarray, rule: int) -> np.ndarray:
    return _life_like_step_lookup(states, *_rule_lookups(rule))


def _write_founder(
    founder: np.ndarray,
    rule: int,
    pair_id: str,
    parameters: MechanismParameters,
    contract: LineageFieldContract,
) -> tuple[np.ndarray, np.ndarray]:
    state = founder.copy()
    carrier = np.zeros(state.shape, dtype=np.float32)
    occupancy = np.zeros(state.shape, dtype=np.float32)
    for sweep in range(1, contract.founder_sweeps + 1):
        state = _step_visible(state, rule)
        noise = _semantic_rng(pair_id, parameters.mechanism, "founder-noise", sweep).random(state.shape)
        state ^= noise < contract.process_noise
        if parameters.mechanism == "latch":
            occupancy += state
        else:
            carrier = diffuse_write_step(
                carrier, state, parameters.diffusion, parameters.write_gain
            )
    if parameters.mechanism == "latch":
        carrier = latch_write(
            carrier,
            occupancy / float(contract.founder_sweeps),
            parameters.upper,
            parameters.lower,
        )
    return carrier * parameters.decay, state


def _swap_histories(values: np.ndarray, replicates: int) -> np.ndarray:
    shaped = values.reshape(2, replicates, *values.shape[1:])
    return shaped[::-1].reshape(values.shape).copy()


def _apply_boundary_condition(
    carrier: np.ndarray,
    condition: str,
    generation: int,
    pair_id: str,
    mechanism: str,
    replicates: int,
    source_entries: Sequence[np.ndarray] | None,
) -> np.ndarray:
    result = carrier.copy()
    if condition == "zero":
        result.fill(0.0)
    elif condition == "shuffle":
        permutation = _semantic_rng(pair_id, mechanism, "shuffle", generation).permutation(256)
        result = result.reshape(len(result), 256)[:, permutation].reshape(result.shape)
    elif condition == "opposite_founder" and generation == 1:
        result = _swap_histories(result, replicates)
    elif condition in ("ablate_g2", "rescue_same_g3", "rescue_opposite_g3") and generation == 3:
        result.fill(0.0)
    elif condition in ("rescue_same_g3", "rescue_opposite_g3") and generation == 4:
        if source_entries is None or len(source_entries) < 4:
            raise ValueError("rescue requires an intact contemporaneous sister carrier")
        result = source_entries[3].copy()
        if condition == "rescue_opposite_g3":
            result = _swap_histories(result, replicates)
    elif condition.startswith("compress_block"):
        result = block_compress(result, int(condition.removeprefix("compress_block")))
    elif condition == "compress_global":
        result = block_compress(result, 16)
    elif condition.startswith("random"):
        count = int(condition.removeprefix("random"))
        result = random_retain(
            result, count, _hash_seed(pair_id, mechanism, condition, generation)
        )
    elif condition.startswith("carrier_noise_"):
        percentage = int(condition.removeprefix("carrier_noise_"))
        mask = _semantic_rng(pair_id, mechanism, "carrier-corruption", percentage, generation).random(result.shape)
        result[mask < percentage / 100.0] *= -1.0
    return result


def _cosine_assign(
    vectors: np.ndarray,
    target_a: Sequence[float],
    target_b: Sequence[float],
    contract: LineageFieldContract,
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


def _observer_outcome(labels: np.ndarray, alive: np.ndarray, replicates: int) -> dict[str, float]:
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


def _component_vectors(states: np.ndarray) -> np.ndarray:
    return np.stack([_component_spectrum(board) for board in states])


def _score_generation(
    state: np.ndarray,
    accumulated: np.ndarray,
    alive: np.ndarray,
    pair: dict[str, Any],
    replicates: int,
    contract: LineageFieldContract,
    *,
    components: bool,
) -> dict[str, Any]:
    targets = pair["targets"]
    primary = _normalize_rows(accumulated)
    terminal = _normalize_rows(live_2x2_counts_batch(state))
    result: dict[str, Any] = {
        "survival": float(np.count_nonzero(alive) / len(alive)),
        "primary": _observer_outcome(
            _cosine_assign(primary, targets["primary"]["A"], targets["primary"]["B"], contract),
            alive,
            replicates,
        ),
        "terminal": _observer_outcome(
            _cosine_assign(
                terminal,
                targets.get("primary_terminal", targets["primary"])["A"],
                targets.get("primary_terminal", targets["primary"])["B"],
                contract,
            ),
            alive,
            replicates,
        ),
    }
    if components and "components" in targets:
        take = min(replicates, 8)
        indices = np.r_[0:take, replicates : replicates + take]
        component_alive = alive[indices]
        vectors = _component_vectors(state[indices])
        result["components"] = _observer_outcome(
            _cosine_assign(vectors, targets["components"]["A"], targets["components"]["B"], contract),
            component_alive,
            take,
        )
        result["components"]["diagnostic_replicates_per_history"] = take
    return result


def _simulate_condition(
    pair: dict[str, Any],
    parameters: MechanismParameters,
    condition: str,
    replicates: int,
    generations: int,
    contract: LineageFieldContract,
    *,
    source_entries: Sequence[np.ndarray] | None = None,
    retain_entries: bool = False,
    component_diagnostics: bool = False,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    rule = int(pair["rule"])
    pair_id = str(pair["pair_id"])
    reset_a = _state_from_hex("life", pair["donor_a"]["initial_state_hex"])
    reset_b = _state_from_hex("life", pair["donor_b"]["initial_state_hex"])
    if not np.array_equal(reset_a, reset_b):
        raise AssertionError(f"visible reset mismatch in pair {pair_id}")
    reset = np.repeat(reset_a[None, ...], 2 * replicates, axis=0)
    reset_digest = hashlib.sha256(reset_a.tobytes()).hexdigest()
    founder = np.concatenate(
        (
            np.repeat(_state_from_hex("life", pair["donor_a"]["donor_state_hex"])[None, ...], replicates, axis=0),
            np.repeat(_state_from_hex("life", pair["donor_b"]["donor_state_hex"])[None, ...], replicates, axis=0),
        )
    )
    if condition == "founder_write_disabled" or condition.startswith("visible"):
        carrier = np.zeros(founder.shape, dtype=np.float32)
        founder_terminal = founder.copy()
    else:
        carrier, founder_terminal = _write_founder(founder, rule, pair_id, parameters, contract)
    founder_accuracy = float(np.mean((carrier > 0.0) == founder))
    founder_difference = float(
        np.mean(np.abs(carrier[:replicates].mean(axis=0) - carrier[replicates:].mean(axis=0)))
    )
    alive = np.ones(2 * replicates, dtype=np.bool_)
    previous_terminal = founder_terminal.copy()
    checkpoints = set(g for g in CHECKPOINT_GENERATIONS if g <= generations)
    outcomes: dict[str, Any] = {}
    entries: list[np.ndarray] = []

    for generation in range(1, generations + 1):
        carrier = _apply_boundary_condition(
            carrier,
            condition,
            generation,
            pair_id,
            parameters.mechanism,
            replicates,
            source_entries,
        )
        if retain_entries:
            entries.append(carrier.copy())
        state = reset.copy()
        if not all(np.array_equal(board, reset_a) for board in state):
            raise AssertionError("visible reset was not bitwise identical")
        if condition.startswith("visible"):
            count = int(condition.removeprefix("visible"))
            indices = np.random.default_rng(
                _hash_seed(pair_id, condition, generation)
            ).choice(256, size=count, replace=False)
            flat = state.reshape(len(state), 256)
            previous = previous_terminal.reshape(len(state), 256)
            flat[:, indices] = previous[:, indices]
        state[~alive] = False
        accumulated = np.zeros((len(state), 15), dtype=np.float64)
        occupancy = np.zeros(state.shape, dtype=np.float32)

        for sweep in range(1, contract.generation_sweeps + 1):
            predicted = _step_visible(state, rule)
            read_enabled = condition != "read_disabled" and not condition.startswith("visible")
            if read_enabled and sweep <= contract.read_sweeps:
                uniforms = _semantic_rng(pair_id, parameters.mechanism, "read", generation, sweep).random(state.shape)
                predicted = apply_field_reader(predicted, carrier, uniforms, parameters.kappa)
            noise = _semantic_rng(pair_id, parameters.mechanism, "process", generation, sweep).random(state.shape)
            predicted ^= noise < contract.process_noise
            predicted[~alive] = False
            state = predicted
            if sweep >= contract.observe_start:
                accumulated += live_2x2_counts_batch(state)
            write_enabled = condition != "no_rewrite" and not condition.startswith("visible")
            if write_enabled and sweep >= contract.write_start:
                if parameters.mechanism == "latch":
                    occupancy += state
                else:
                    carrier = diffuse_write_step(
                        carrier, state, parameters.diffusion, parameters.write_gain
                    )

        alive &= state.any(axis=(1, 2))
        if generation in checkpoints:
            outcomes[str(generation)] = _score_generation(
                state,
                accumulated,
                alive,
                pair,
                replicates,
                contract,
                components=component_diagnostics and generation in (8, 16),
            )
        write_enabled = condition != "no_rewrite" and not condition.startswith("visible")
        if write_enabled and parameters.mechanism == "latch":
            window = contract.generation_sweeps - contract.write_start + 1
            carrier = latch_write(
                carrier, occupancy / float(window), parameters.upper, parameters.lower
            )
        if condition.startswith("visible"):
            carrier.fill(0.0)
        else:
            carrier *= parameters.decay
        carrier[~alive] = 0.0
        previous_terminal = state.copy()

    return (
        {
            "reset_sha256": reset_digest,
            "founder_carrier_mean_abs": float(np.mean(np.abs(entries[0] if entries else carrier))),
            "founder_direct_reconstruction_accuracy": founder_accuracy,
            "founder_history_field_difference": founder_difference,
            "outcomes": outcomes,
        },
        entries,
    )


def _pair_task(payload: tuple[dict[str, Any], LineageFieldContract]) -> dict[str, Any]:
    item, contract = payload
    pair = item["pair"]
    parameters = MechanismParameters(**item["parameters"])
    conditions = tuple(item["conditions"])
    intact, entries = _simulate_condition(
        pair,
        parameters,
        "intact",
        int(item["replicates"]),
        int(item["generations"]),
        contract,
        retain_entries=True,
        component_diagnostics=bool(item.get("component_diagnostics", False)),
    )
    rows: dict[str, Any] = {"intact": intact}
    for condition in conditions:
        if condition == "intact":
            continue
        result, _ = _simulate_condition(
            pair,
            parameters,
            condition,
            int(item["replicates"]),
            int(item["generations"]),
            contract,
            source_entries=entries,
            component_diagnostics=bool(item.get("component_diagnostics", False)),
        )
        rows[condition] = result
    return {
        "checkpoint": item["checkpoint"],
        "stage": item["stage"],
        "mechanism": parameters.mechanism,
        "rule": int(pair["rule"]),
        "pair_id": pair["pair_id"],
        "replicates": int(item["replicates"]),
        "conditions": rows,
    }


def _synthetic_calibration_states() -> np.ndarray:
    rng = np.random.default_rng(_hash_seed("lineage-field-calibration-boards"))
    densities = np.linspace(0.20, 0.80, 16)
    return np.stack([rng.random((16, 16)) < density for density in densities])


def calibrate_mechanism(mechanism: str, contract: LineageFieldContract) -> dict[str, Any]:
    boards = _synthetic_calibration_states()
    candidates: list[dict[str, Any]] = []
    kappas = (0.025, 0.05, 0.10)
    decays = (0.40, 0.55, 0.70)
    if mechanism == "latch":
        grids: Iterable[tuple[float, float, float, float]] = tuple(
            (upper, 1.0 - upper, 0.04, 0.08) for upper in (0.60, 0.70)
        )
    else:
        grids = tuple(
            (0.60, 0.40, diffusion, gain)
            for diffusion in (0.04, 0.08)
            for gain in (0.08, 0.12)
        )
    for kappa in kappas:
        for decay in decays:
            for upper, lower, diffusion, gain in grids:
                parameters = MechanismParameters(
                    mechanism, kappa, decay, upper, lower, diffusion, gain
                )
                carrier, _ = _write_founder(
                    boards, 31649, "calibration", parameters, contract
                )
                baseline = boards.copy()
                read = boards.copy()
                for sweep in range(1, contract.read_sweeps + 1):
                    baseline = _step_visible(baseline, 31649)
                    predicted = _step_visible(read, 31649)
                    uniforms = np.random.default_rng(
                        _hash_seed("lineage-field-calibration-reader", sweep)
                    ).random(boards.shape)
                    read = apply_field_reader(predicted, carrier, uniforms, kappa)
                perturbation = float(np.mean(read != baseline))
                magnitude = float(np.mean(np.abs(carrier)))
                saturation = float(np.mean(np.abs(carrier) >= 0.99))
                residual4 = float(np.mean(np.abs(carrier * decay**4)))
                viability = float(np.mean(read.any(axis=(1, 2))))
                eligible = bool(
                    viability >= 0.90
                    and 0.02 <= perturbation <= 0.20
                    and 0.02 <= magnitude <= 0.95
                    and saturation <= 0.95
                    and residual4 <= 0.15
                )
                score = (
                    abs(perturbation - 0.05)
                    + 0.25 * abs(magnitude - 0.50)
                    + 0.25 * residual4
                    + 0.25 * saturation
                )
                candidates.append(
                    {
                        **parameters.to_dict(),
                        "eligible": eligible,
                        "calibration_score": score,
                        "metrics": {
                            "viability": viability,
                            "perturbation": perturbation,
                            "mean_abs": magnitude,
                            "saturation": saturation,
                            "no_write_residual_g4": residual4,
                        },
                    }
                )
    eligible_rows = [row for row in candidates if row["eligible"]]
    pool = eligible_rows or sorted(candidates, key=lambda row: (row["metrics"]["perturbation"], row["calibration_score"]))[:1]
    selected = min(
        pool,
        key=lambda row: (
            float(row["calibration_score"]),
            hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
        ),
    )
    parameter_keys = set(inspect.signature(MechanismParameters).parameters)
    parameters = {key: value for key, value in selected.items() if key in parameter_keys}
    return {
        "mechanism": mechanism,
        "eligible_count": len(eligible_rows),
        "candidate_count": len(candidates),
        "selected": parameters,
        "selected_metrics": selected["metrics"],
        "calibration_failure": not bool(eligible_rows),
        "candidates": candidates,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_round3_pairs() -> dict[int, list[dict[str, Any]]]:
    missing = [path for path in (NARROW_CALIBRATION, NARROW_COHORTS, WIDE_31648, WIDE_70366) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen round-3 inputs: {missing}")
    calibration = _load_json(NARROW_CALIBRATION)["calibration"]["prototype"]
    acquisitions = [
        _load_json(ROUND3_ROOT / f"narrow_acquire/checkpoints/launch-{launch}.json")["result"]
        for launch in range(4)
    ]
    narrow = pair_prototype_donors(acquisitions, calibration, V3Contract())
    broad_31648 = _load_json(WIDE_31648)["result"]["pairs"]
    broad_70366 = _load_json(WIDE_70366)["result"]["pairs"]

    def ordered(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(f"lineage-field-cohort:{row['pair_id']}".encode()).hexdigest(),
                row["pair_id"],
            ),
        )

    return {31649: ordered(narrow), 31648: ordered(broad_31648), 70366: ordered(broad_70366)}


def _profile_units(profile: LineageFieldProfile) -> int:
    core = 2 * len(MECHANISMS) * profile.generations * (
        profile.primary_pairs * profile.primary_replicates * len(CORE_CONDITIONS)
        + 2 * profile.holdout_pairs * profile.holdout_replicates * len(CORE_CONDITIONS)
    )
    diagnostic = (
        2
        * len(MECHANISMS)
        * profile.generations
        * profile.diagnostic_pairs
        * profile.diagnostic_replicates
        * len(DIAGNOSTIC_CONDITIONS)
    )
    return core + diagnostic


def choose_timing_profile(seconds_per_unit: float, max_hours: float, workers: int) -> tuple[str, dict[str, float]]:
    usable = max(60.0, max_hours * 3600.0 - 1800.0)
    projections: dict[str, float] = {}
    for name in ("reference", "reduced", "floor"):
        projections[name] = seconds_per_unit * _profile_units(FIELD_PROFILES[name]) / max(1, workers) * 1.5 + 900.0
        if projections[name] <= usable:
            return name, projections
    return "floor", projections


def _timing_benchmark(
    pair: dict[str, Any],
    calibrations: dict[str, dict[str, Any]],
    contract: LineageFieldContract,
) -> float:
    started = time.monotonic()
    for mechanism in MECHANISMS:
        item = {
            "checkpoint": f"benchmark-{mechanism}",
            "stage": "benchmark",
            "pair": pair,
            "parameters": calibrations[mechanism]["selected"],
            "conditions": CORE_CONDITIONS,
            "replicates": 2,
            "generations": 2,
        }
        _pair_task((item, contract))
    elapsed = max(time.monotonic() - started, 1e-6)
    units = len(MECHANISMS) * 2 * 2 * 2 * len(CORE_CONDITIONS)
    return elapsed / units


def _make_items(
    stage: str,
    cohorts: dict[int, list[dict[str, Any]]],
    profile: LineageFieldProfile,
    calibrations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    conditions = CORE_CONDITIONS if stage == "core" else DIAGNOSTIC_CONDITIONS
    rules = RULES if stage == "core" else (31649,)
    items: list[dict[str, Any]] = []
    for mechanism in MECHANISMS:
        for rule in rules:
            if stage == "diagnostics":
                count = profile.diagnostic_pairs
                replicates = profile.diagnostic_replicates
            elif rule == 31649:
                count = profile.primary_pairs
                replicates = profile.primary_replicates
            else:
                count = profile.holdout_pairs
                replicates = profile.holdout_replicates
            for index, pair in enumerate(cohorts[rule][:count]):
                items.append(
                    {
                        "checkpoint": f"{stage}-{mechanism}-{rule:06d}-{index:04d}",
                        "stage": stage,
                        "pair": pair,
                        "parameters": calibrations[mechanism]["selected"],
                        "conditions": conditions,
                        "replicates": replicates,
                        "generations": profile.generations,
                        "component_diagnostics": stage == "diagnostics",
                    }
                )
    return items


def _run_stage(
    output: Path,
    stage: str,
    items: Sequence[dict[str, Any]],
    contract: LineageFieldContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status: Any,
) -> tuple[list[dict[str, Any]], bool]:
    checkpoint_root = output / stage / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for item in items:
        key = item["checkpoint"]
        path = checkpoint_root / f"{key}.json"
        if resume and path.exists():
            payload = _load_json(path)
            if payload.get("design_digest") != design_digest:
                raise ValueError(f"checkpoint design mismatch: {path}")
            results[key] = payload["result"]
        else:
            missing.append(item)
    started = time.monotonic()
    truncated = False

    def save(item: dict[str, Any], result: dict[str, Any]) -> None:
        key = item["checkpoint"]
        _atomic_json(
            checkpoint_root / f"{key}.json",
            {"design_digest": design_digest, "stage": stage, "checkpoint": key, "result": result},
        )
        results[key] = result
        elapsed = max(time.monotonic() - started, 1e-6)
        done_new = max(1, len(results) - (len(items) - len(missing)))
        eta = elapsed / done_new * max(0, len(items) - len(results))
        status("running", stage, completed=len(results), total=len(items), eta_seconds=eta)

    if missing and time.time() < deadline:
        pool = ProcessPoolExecutor(max_workers=min(workers, len(missing)))
        iterator = iter(missing)
        pending: dict[Any, dict[str, Any]] = {}

        def submit_one() -> bool:
            if time.time() >= deadline:
                return False
            try:
                item = next(iterator)
            except StopIteration:
                return False
            pending[pool.submit(_pair_task, (item, contract))] = item
            return True

        for _ in range(min(len(missing), workers * 2)):
            submit_one()
        try:
            while pending:
                remaining = deadline - time.time()
                if remaining <= 0:
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
        output / stage / "stage_summary.json",
        {
            "stage": stage,
            "design_digest": design_digest,
            "complete": complete,
            "completed": len(results),
            "total": len(items),
            "budget_truncated": truncated or not complete,
        },
    )
    if complete:
        _atomic_text(output / stage / "COMPLETE", "complete\n")
    return [results[key] for key in sorted(results)], complete


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


def _rows_for(rows: Sequence[dict[str, Any]], mechanism: str, rule: int) -> list[dict[str, Any]]:
    return [row for row in rows if row["mechanism"] == mechanism and int(row["rule"]) == rule]


def _metric_values(
    rows: Sequence[dict[str, Any]], condition: str, generation: int, observer: str, metric: str
) -> list[float]:
    values: list[float] = []
    for row in rows:
        outcome = row["conditions"].get(condition, {}).get("outcomes", {}).get(str(generation))
        if outcome is None:
            continue
        if metric == "survival":
            values.append(float(outcome["survival"]))
        elif observer in outcome and metric in outcome[observer]:
            values.append(float(outcome[observer][metric]))
    return values


def _paired_difference(
    rows: Sequence[dict[str, Any]], left: str, right: str, generation: int, observer: str = "primary"
) -> list[float]:
    result: list[float] = []
    for row in rows:
        conditions = row["conditions"]
        try:
            a = conditions[left]["outcomes"][str(generation)][observer]["crossover"]
            b = conditions[right]["outcomes"][str(generation)][observer]["crossover"]
        except KeyError:
            continue
        result.append(float(a) - float(b))
    return result


def _summarize_group(
    rows: Sequence[dict[str, Any]], profile: LineageFieldProfile, seed_parts: Sequence[object], alpha: float
) -> dict[str, Any]:
    conditions = sorted({condition for row in rows for condition in row["conditions"]})
    result: dict[str, Any] = {}
    for condition in conditions:
        generations = sorted(
            {
                int(generation)
                for row in rows
                for generation in row["conditions"].get(condition, {}).get("outcomes", {})
            }
        )
        result[condition] = {}
        for generation in generations:
            summary: dict[str, Any] = {
                "survival": _bootstrap(
                    _metric_values(rows, condition, generation, "primary", "survival"),
                    profile.bootstrap_resamples,
                    _hash_seed(*seed_parts, condition, generation, "survival"),
                    alpha,
                )
            }
            for observer in ("primary", "terminal", "components"):
                values = _metric_values(rows, condition, generation, observer, "crossover")
                if values:
                    summary[observer] = {
                        "crossover": _bootstrap(
                            values,
                            profile.bootstrap_resamples,
                            _hash_seed(*seed_parts, condition, generation, observer),
                            alpha,
                        ),
                        "direction_a_mean": float(np.mean(_metric_values(rows, condition, generation, observer, "direction_a"))),
                        "direction_b_mean": float(np.mean(_metric_values(rows, condition, generation, observer, "direction_b"))),
                        "fraction_pairs_positive": float(np.mean(np.asarray(values) > 0.0)),
                    }
            result[condition][str(generation)] = summary
    return result


def _gate_summary(
    rows: Sequence[dict[str, Any]], profile: LineageFieldProfile, mechanism: str
) -> dict[str, Any]:
    if profile.generations < 16:
        return {"verdict": "NOT_ADJUDICATED_PROFILE", "renewed_gate": False}
    alpha = 0.025

    def boot(values: Sequence[float], name: str) -> dict[str, Any]:
        return _bootstrap(values, profile.bootstrap_resamples, _hash_seed("gate", mechanism, name), alpha)

    intact8 = boot(_metric_values(rows, "intact", 8, "primary", "crossover"), "intact8")
    intact16 = boot(_metric_values(rows, "intact", 16, "primary", "crossover"), "intact16")
    terminal8 = boot(_metric_values(rows, "intact", 8, "terminal", "crossover"), "terminal8")
    survival8 = boot(_metric_values(rows, "intact", 8, "primary", "survival"), "survival8")
    controls = {
        name: boot(_paired_difference(rows, "intact", name, 8), f"adv-{name}")
        for name in ("zero", "shuffle", "read_disabled", "founder_write_disabled")
    }
    no_rewrite8 = boot(_metric_values(rows, "no_rewrite", 8, "primary", "crossover"), "no-rewrite8")
    intact4 = boot(_metric_values(rows, "intact", 4, "primary", "crossover"), "intact4")
    ablate4 = boot(_metric_values(rows, "ablate_g2", 4, "primary", "crossover"), "ablate4")
    rescue4 = boot(_metric_values(rows, "rescue_same_g3", 4, "primary", "crossover"), "rescue4")
    rescue_adv = boot(_paired_difference(rows, "rescue_same_g3", "ablate_g2", 4), "rescue-adv")
    opposite4 = boot(_metric_values(rows, "opposite_founder", 4, "primary", "crossover"), "opposite4")
    intact_mean8 = float(intact8["mean"] or 0.0)
    intact_mean4 = float(intact4["mean"] or 0.0)
    direction_a = float(np.mean(_metric_values(rows, "intact", 8, "primary", "direction_a")))
    direction_b = float(np.mean(_metric_values(rows, "intact", 8, "primary", "direction_b")))
    pair_values = _metric_values(rows, "intact", 8, "primary", "crossover")
    renewed = bool(
        intact_mean8 >= 0.15
        and intact8["ci"][0] is not None and intact8["ci"][0] > 0.0
        and float(intact16["mean"] or 0.0) >= 0.10 and intact16["ci"][0] is not None and intact16["ci"][0] > 0.0
        and direction_a > 0.0 and direction_b > 0.0
        and pair_values and float(np.mean(np.asarray(pair_values) > 0.0)) >= 0.50
        and float(survival8["mean"] or 0.0) >= 0.90
        and all(float(value["mean"] or 0.0) >= 0.10 and value["ci"][0] is not None and value["ci"][0] > 0.0 for value in controls.values())
        and float(no_rewrite8["mean"] or 0.0) <= 0.30 * intact_mean8
        and float(ablate4["mean"] or 0.0) <= 0.30 * intact_mean4
        and float(rescue4["mean"] or 0.0) >= 0.70 * intact_mean4
        and float(rescue_adv["mean"] or 0.0) >= 0.10 and rescue_adv["ci"][0] is not None and rescue_adv["ci"][0] > 0.0
        and float(opposite4["mean"] or 0.0) <= -0.10 and opposite4["ci"][1] is not None and opposite4["ci"][1] < 0.0
        and float(terminal8["mean"] or 0.0) >= 0.10 and terminal8["ci"][0] is not None and terminal8["ci"][0] > 0.0
    )
    static = bool(
        not renewed
        and intact_mean8 >= 0.15
        and intact8["ci"][0] is not None and intact8["ci"][0] > 0.0
        and float(no_rewrite8["mean"] or 0.0) > 0.30 * intact_mean8
    )
    return {
        "verdict": "RENEWED_LINEAGE_CARRIER" if renewed else "STATIC_HIDDEN_TEMPLATE" if static else "NO_CAUSAL_FIELD_HEREDITY",
        "renewed_gate": renewed,
        "static_template_gate": static,
        "intact_generation8": intact8,
        "intact_generation16": intact16,
        "terminal_generation8": terminal8,
        "survival_generation8": survival8,
        "control_advantages_generation8": controls,
        "no_rewrite_generation8": no_rewrite8,
        "ablation_generation4": ablate4,
        "rescue_generation4": rescue4,
        "rescue_advantage_generation4": rescue_adv,
        "opposite_generation4": opposite4,
        "direction_a_mean": direction_a,
        "direction_b_mean": direction_b,
        "fraction_pairs_positive": float(np.mean(np.asarray(pair_values) > 0.0)) if pair_values else 0.0,
    }


def adjudicate(
    core_rows: Sequence[dict[str, Any]],
    diagnostic_rows: Sequence[dict[str, Any]],
    profile: LineageFieldProfile,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"verdict": "INCOMPLETE", "mechanisms": {}}
    mechanisms: dict[str, Any] = {}
    any_cross_rule = False
    for mechanism in MECHANISMS:
        by_rule: dict[str, Any] = {}
        for rule in RULES:
            rows = _rows_for(core_rows, mechanism, rule)
            by_rule[str(rule)] = {
                "summary": _summarize_group(rows, profile, (mechanism, rule, "core"), 0.025 if rule == 31649 else 0.0125),
                "primary_gate": _gate_summary(rows, profile, mechanism) if rule == 31649 else None,
            }
        primary_gate = by_rule["31649"]["primary_gate"] or {"renewed_gate": False, "verdict": "INCOMPLETE"}
        holdout_passes: list[int] = []
        for rule in (31648, 70366):
            rows = _rows_for(core_rows, mechanism, rule)
            intact = _bootstrap(
                _metric_values(rows, "intact", 8, "primary", "crossover"),
                profile.bootstrap_resamples,
                _hash_seed("holdout", mechanism, rule),
                0.0125,
            )
            zero_adv = _bootstrap(
                _paired_difference(rows, "intact", "zero", 8),
                profile.bootstrap_resamples,
                _hash_seed("holdout-zero", mechanism, rule),
                0.0125,
            )
            passed = bool(
                profile.generations >= 8
                and float(intact["mean"] or 0.0) >= 0.15
                and intact["ci"][0] is not None and intact["ci"][0] > 0.0
                and float(zero_adv["mean"] or 0.0) >= 0.10
                and zero_adv["ci"][0] is not None and zero_adv["ci"][0] > 0.0
            )
            by_rule[str(rule)]["holdout_gate"] = {"passed": passed, "intact_generation8": intact, "zero_advantage": zero_adv}
            if passed:
                holdout_passes.append(rule)
        diagnostics = _rows_for(diagnostic_rows, mechanism, 31649)
        diagnostic_summary = _summarize_group(diagnostics, profile, (mechanism, 31649, "diagnostics"), 0.025)
        compressed = False
        if profile.generations >= 8 and diagnostics:
            intact_values = _metric_values(diagnostics, "intact", 8, "primary", "crossover")
            compressed_values = _metric_values(diagnostics, "compress_block4", 8, "primary", "crossover")
            compressed = bool(
                intact_values
                and compressed_values
                and float(np.mean(compressed_values)) >= 0.70 * float(np.mean(intact_values))
                and float(np.mean(compressed_values)) >= 0.10
            )
        cross_rule = bool(primary_gate.get("renewed_gate") and holdout_passes)
        any_cross_rule |= cross_rule
        verdict = str(primary_gate.get("verdict"))
        if primary_gate.get("renewed_gate") and compressed:
            verdict = "COMPRESSED_RENEWED_LINEAGE_CARRIER"
        if cross_rule:
            verdict = "CROSS_RULE_" + verdict
        mechanisms[mechanism] = {
            "verdict": verdict,
            "primary_gate": primary_gate,
            "holdout_passes": holdout_passes,
            "compressed_16dof_gate": compressed,
            "rules": by_rule,
            "diagnostics": diagnostic_summary,
        }
    overall = (
        "BOTH_MECHANISMS_CROSS_RULE_RENEWED"
        if all(mechanisms[m]["verdict"].startswith("CROSS_RULE") for m in MECHANISMS)
        else "AT_LEAST_ONE_CROSS_RULE_RENEWED"
        if any_cross_rule
        else "MECHANISM_SPECIFIC_RENEWED"
        if any(mechanisms[m]["primary_gate"].get("renewed_gate") for m in MECHANISMS)
        else "NO_RENEWED_LINEAGE_CARRIER"
    )
    return {"verdict": overall, "mechanisms": mechanisms}


def _render_report(results: dict[str, Any]) -> str:
    adjudication = results["adjudication"]
    lines = [
        "# CA lineage-field round 4",
        "",
        f"State: **{results['state']}**. Profile: `{results['selected_profile']}`.",
        f"Overall verdict: **{adjudication['verdict']}**.",
        f"Elapsed: {results['elapsed_seconds'] / 3600.0:.2f} wall hours.",
        "",
        "## Mechanisms",
        "",
    ]
    for mechanism in MECHANISMS:
        value = adjudication.get("mechanisms", {}).get(mechanism)
        if not value:
            lines.append(f"- `{mechanism}`: not adjudicated.")
            continue
        gate = value["primary_gate"]
        g8 = gate.get("intact_generation8", {})
        lines.append(
            f"- `{mechanism}`: **{value['verdict']}**; Rule-31649 generation-8 "
            f"crossover `{g8.get('mean')}`, CI `{g8.get('ci')}`; holdouts `{value['holdout_passes']}`; "
            f"16-degree compression `{value['compressed_16dof_gate']}`."
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "A positive result is a synthetic, two-timescale local carrier mechanism after a verified visible-state reset. "
            "The carrier is still part of the total CA state. No verdict implies metabolism, agency, biological life, or a global organism.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_lay_summary(results: dict[str, Any]) -> str:
    verdict = results["adjudication"]["verdict"]
    return (
        "# Lay summary\n\n"
        f"The run's overall verdict is **{verdict}**. We gave each cellular automaton a second, slow field, "
        "then erased the ordinary visible board completely before every daughter developed. The decisive comparison "
        "asks whether the daughter remembers which form its parent had only through that slow field, whether erasing "
        "the field removes the memory, whether putting it back rescues the memory, and whether the daughter must renew it.\n\n"
        "The latch and diffusing-field mechanisms received equal resources. Detailed intervention results and confidence "
        "intervals are in `REPORT.md` and `RESULTS.json`. Even a positive result describes an engineered CA memory channel, "
        "not biological life or memory outside the automaton's total physical state.\n"
    )


def _update_discovery_log(results: dict[str, Any]) -> None:
    path = ROOT / "DISCOVERY_LOG_EIDOSOMA_SCIENTIST.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Discovery log\n"
    start = "<!-- ca-lineage-field-round-4:start -->"
    end = "<!-- ca-lineage-field-round-4:end -->"
    section = "\n".join(
        (
            start,
            "## CA lineage-field round 4",
            "",
            f"Overall verdict: `{results['adjudication']['verdict']}`.",
            f"Selected timing profile: `{results['selected_profile']}`; elapsed `{results['elapsed_seconds'] / 3600.0:.2f}` wall hours.",
            "See `results/ca-lineage-field-round-4/REPORT.md` and `LAY_SUMMARY.md`.",
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


def run_lineage_field_campaign(
    output: Path,
    *,
    profile_name: str = "reference",
    workers: int = 20,
    max_hours: float = 8.0,
    resume: bool = False,
    selected_stages: Sequence[str] | None = None,
) -> dict[str, Any]:
    require_pinned_numpy()
    if profile_name not in PUBLIC_PROFILES:
        raise ValueError(f"unknown lineage-field profile {profile_name!r}")
    if max_hours <= 0.0 or max_hours > 8.0:
        raise ValueError("lineage-field max-hours must be in (0, 8]")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hard_deadline = started + max_hours * 3600.0
    contract = LineageFieldContract()
    reserve_seconds = min(contract.science_reserve_seconds, max(60.0, max_hours * 3600.0 * 0.10))
    science_deadline = max(started, hard_deadline - reserve_seconds)
    stages = set(selected_stages or ("calibrate", "seal", "core", "diagnostics", "holdouts", "adjudication"))

    def status(state: str, stage: str, **extra: Any) -> None:
        now = time.time()
        payload = {
            "state": state,
            "stage": stage,
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
        print(f"[{state}] {stage}{progress}", flush=True)

    try:
        cohorts = load_round3_pairs()
        design_path = output / "DESIGN.json"
        calibration_path = output / "CALIBRATION.json"
        if resume and design_path.exists() and calibration_path.exists():
            prior_design = _load_json(design_path)
            calibrations = _load_json(calibration_path)["mechanisms"]
            selected_profile_name = str(prior_design["selected_profile"])
            timing = prior_design["timing"]
        else:
            status("running", "calibrate")
            calibrations = {
                mechanism: calibrate_mechanism(mechanism, contract) for mechanism in MECHANISMS
            }
            _atomic_json(calibration_path, {"mechanisms": calibrations, "label_access": False})
            if profile_name == "reference":
                seconds_per_unit = _timing_benchmark(cohorts[31649][0], calibrations, contract)
                selected_profile_name, projections = choose_timing_profile(seconds_per_unit, max_hours, workers)
                timing = {"seconds_per_unit": seconds_per_unit, "projected_seconds": projections}
            else:
                selected_profile_name = profile_name
                timing = {"seconds_per_unit": None, "projected_seconds": {}}
        selected_profile = FIELD_PROFILES[selected_profile_name]
        input_paths = [PROTOCOL_PATH, NARROW_CALIBRATION, NARROW_COHORTS, WIDE_31648, WIDE_70366]
        design_payload = {
            "contract": contract.to_dict(),
            "requested_profile": profile_name,
            "selected_profile": selected_profile_name,
            "profile": asdict(selected_profile),
            "timing": timing,
            "calibrations": {mechanism: value["selected"] for mechanism, value in calibrations.items()},
            "cohorts": {
                str(rule): [pair["pair_id"] for pair in cohorts[rule]][: max(selected_profile.primary_pairs, selected_profile.holdout_pairs)]
                for rule in RULES
            },
            "input_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in input_paths},
            "implementation_sha256": {
                "lineage_field.py": _sha256(Path(__file__)),
                "life_family.py": _sha256(Path(__file__).with_name("life_family.py")),
            },
            "cleanroom_exclusion": "no Wagner implementation artifact is read, imported, hashed, or executed",
        }
        design_digest = hashlib.sha256(
            json.dumps(design_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        design = {**design_payload, "design_digest": design_digest}
        if resume and design_path.exists():
            previous = _load_json(design_path)
            if previous.get("design_digest") != design_digest:
                raise ValueError("resume design digest mismatch")
        _atomic_json(design_path, design)
        _atomic_json(
            output / "MANIFEST.json",
            {
                "experiment": "ca_lineage_field_round_4",
                "requested_profile": profile_name,
                "selected_profile": selected_profile_name,
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
        _atomic_json(
            output / "COHORTS.json",
            {
                "design_digest": design_digest,
                "pairs": {
                    str(rule): [pair["pair_id"] for pair in cohorts[rule]][: selected_profile.primary_pairs if rule == 31649 else selected_profile.holdout_pairs]
                    for rule in RULES
                },
            },
        )

        core_items = _make_items("core", cohorts, selected_profile, calibrations)
        diagnostic_items = _make_items("diagnostics", cohorts, selected_profile, calibrations)
        if "core" in stages or "holdouts" in stages:
            status("running", "core", completed=0, total=len(core_items))
            core_rows, core_complete = _run_stage(
                output,
                "core",
                core_items,
                contract,
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
            )
        else:
            core_rows, core_complete = _run_stage(
                output, "core", core_items, contract, design_digest, workers=workers, resume=True, deadline=started, status=status
            )
        if "diagnostics" in stages:
            status("running", "diagnostics", completed=0, total=len(diagnostic_items))
            diagnostic_rows, diagnostics_complete = _run_stage(
                output,
                "diagnostics",
                diagnostic_items,
                contract,
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
            )
        else:
            diagnostic_rows, diagnostics_complete = _run_stage(
                output, "diagnostics", diagnostic_items, contract, design_digest, workers=workers, resume=True, deadline=started, status=status
            )
        all_complete = core_complete and diagnostics_complete
        status("running", "adjudication")
        adjudication = adjudicate(core_rows, diagnostic_rows, selected_profile, all_complete)
        state = "complete" if all_complete else "partial_budget_exhausted"
        results = {
            "experiment": "ca_lineage_field_round_4",
            "state": state,
            "requested_profile": profile_name,
            "selected_profile": selected_profile_name,
            "design_digest": design_digest,
            "started_unix": started,
            "completed_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "stage_completeness": {"core": core_complete, "diagnostics": diagnostics_complete},
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
    "CORE_CONDITIONS",
    "DIAGNOSTIC_CONDITIONS",
    "FIELD_PROFILES",
    "LineageFieldContract",
    "MechanismParameters",
    "apply_field_reader",
    "block_compress",
    "calibrate_mechanism",
    "choose_timing_profile",
    "diffuse_write_step",
    "latch_write",
    "launch_detached",
    "load_round3_pairs",
    "random_retain",
    "run_lineage_field_campaign",
]
