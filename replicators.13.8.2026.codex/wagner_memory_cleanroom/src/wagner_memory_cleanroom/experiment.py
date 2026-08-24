from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable

import numpy as np

from .config import Registration, scaled_futures, stage_source_count
from .engine import apply_challenge, in_basin, rollout_jax, strict_destination
from .rng import generator, jax_key_data, stable_permutation
from .source import Rulebook, generate_rulebook, rulebook_metadata


HISTORIES = ("A", "B")
HALVES = (0, 1)


@dataclass
class StageResult:
    records: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    elapsed_seconds: float
    simulated_futures: int


def _key(registration: Registration, stage: str, *coordinates: Any) -> np.ndarray:
    base = stage.removesuffix("_audit")
    return jax_key_data(str(registration.protocol["master_seed"]), base, *coordinates)


def _rng(registration: Registration, stage: str, *coordinates: Any) -> np.random.Generator:
    base = stage.removesuffix("_audit")
    return generator(str(registration.protocol["master_seed"]), base, *coordinates)


def _destination_record(
    *,
    stage: str,
    source_id: int,
    history_name: str,
    arm: str,
    challenge: str,
    half: int,
    correct: np.ndarray,
    wrong: np.ndarray,
    acquired: float,
    writer: str = "",
    age: int | None = None,
    checkpoint: int | None = None,
    theta: float | None = None,
    half_life: int | None = None,
    coupling: float | None = None,
) -> dict[str, Any]:
    both = correct & wrong
    correct_only = correct & ~wrong
    wrong_only = wrong & ~correct
    return {
        "stage": stage.removesuffix("_audit"),
        "source_id": int(source_id),
        "history": history_name,
        "writer": writer,
        "arm": arm,
        "challenge": challenge,
        "age": age,
        "checkpoint": checkpoint,
        "theta": theta,
        "half_life": half_life,
        "coupling": coupling,
        "half": int(half),
        "n": int(correct.size),
        "correct": int(np.sum(correct_only)),
        "wrong": int(np.sum(wrong_only)),
        "both": int(np.sum(both)),
        "unresolved": int(np.sum(~correct & ~wrong)),
        "acquired": float(acquired),
    }


def _assay(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    initial: np.ndarray,
    field: np.ndarray,
    *,
    history_name: str,
    challenge: str,
    half: int,
    coordinate: tuple[Any, ...],
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    protocol = registration.protocol
    engine = protocol["engine"]
    target = rulebook.target(history_name)
    opposite = rulebook.opposite(history_name)
    challenge_rng = _rng(registration, stage, "challenge", rulebook.source_id, history_name, challenge, half, *coordinate)
    challenged = apply_challenge(
        initial,
        target,
        challenge,
        neutral_damage_fraction=float(protocol["state"]["neutral_damage_fraction"]),
        rng=challenge_rng,
    )
    history = rollout_jax(
        rulebook.weights,
        challenged,
        field,
        sweeps=int(engine["horizon_sweeps"]),
        theta=theta,
        flip_probability=float(engine["expression_flip_probability"]),
        key_data=_key(registration, stage, "assay", rulebook.source_id, history_name, challenge, half, *coordinate),
    )
    return strict_destination(history, target, opposite, int(engine["strict_run"]))


def _write_state(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    history_name: str,
    writer: dict[str, Any],
    futures: int,
    half: int,
    theta_override: float | None = None,
) -> tuple[np.ndarray, float]:
    target = rulebook.target(history_name)
    if writer["mode"] == "hard":
        state = np.repeat(target[None, :], futures, axis=0)
        return state.astype(np.int8), 1.0
    initial = np.repeat(rulebook.neutral[None, :], futures, axis=0)
    field = np.repeat((float(writer["strength"]) * target)[None, :], futures, axis=0)
    theta = float(writer["theta"] if theta_override is None else theta_override)
    written_history = rollout_jax(
        rulebook.weights,
        initial,
        field,
        sweeps=int(registration.engine["generation_sweeps"]) * int(writer["dwell"]),
        theta=theta,
        flip_probability=0.0,
        key_data=_key(registration, stage, "writer", rulebook.source_id, history_name, writer["name"], half, theta),
    )
    state = written_history[-1]
    acquisition = float(np.mean(in_basin(state, target)))
    return state, acquisition


def _age_state(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    state: np.ndarray,
    history_name: str,
    writer_name: str,
    age: int,
    half: int,
    theta: float,
) -> np.ndarray:
    if age == 0:
        return state
    field = np.zeros_like(state, dtype=np.float32)
    aged = rollout_jax(
        rulebook.weights,
        state,
        field,
        sweeps=age * int(registration.engine["generation_sweeps"]),
        theta=theta,
        flip_probability=float(registration.engine["expression_flip_probability"]),
        key_data=_key(registration, stage, "age", rulebook.source_id, history_name, writer_name, age, half),
    )
    return aged[-1]


def _state_arm(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    written: np.ndarray,
    history_name: str,
    arm: str,
) -> np.ndarray:
    futures, genes = written.shape
    target = rulebook.target(history_name)
    if arm in {"self", "state_transplant"}:
        return written.copy()
    if arm == "reset":
        return np.repeat(rulebook.neutral[None, :], futures, axis=0)
    if arm == "destination_matched":
        return np.repeat(target[None, :], futures, axis=0)
    if arm == "descriptor_matched":
        result = written.copy()
        for future in range(futures):
            permutation = stable_permutation(genes, str(registration.protocol["master_seed"]), stage.removesuffix("_audit"), "descriptor", rulebook.source_id, history_name, future)
            result[future] = result[future, permutation]
        return result
    if arm == "pattern_shuffle":
        permutation = stable_permutation(genes, str(registration.protocol["master_seed"]), stage.removesuffix("_audit"), "shuffle", rulebook.source_id, history_name)
        return written[:, permutation]
    raise ValueError(arm)


def _state_schedule(registration: Registration) -> list[tuple[list[str], str, list[int], int]]:
    state = registration.protocol["state"]
    primary = scaled_futures(int(state["primary_futures"]), registration)
    persistence = scaled_futures(int(state["persistence_futures"]), registration)
    return [
        (list(state["arms"]), "neutral_damage", [0], primary),
        (["state_transplant", "reset"], "forced_break", [0], primary),
        (["state_transplant", "reset", "pattern_shuffle"], "all", [1, 2, 4, 8], persistence),
    ]


def run_state_source(registration: Registration, stage: str, rulebook: Rulebook) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for writer in registration.protocol["state"]["writers"]:
        for history_name in HISTORIES:
            for half in HALVES:
                cache: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
                for arms, challenge_selector, ages, futures in _state_schedule(registration):
                    cache_key = (futures, half)
                    if cache_key not in cache:
                        cache[cache_key] = _write_state(registration, stage, rulebook, history_name, writer, futures, half)
                    base_written, acquisition = cache[cache_key]
                    challenges = registration.protocol["state"]["challenges"] if challenge_selector == "all" else [challenge_selector]
                    for age in ages:
                        aged = _age_state(registration, stage, rulebook, base_written, history_name, writer["name"], age, half, float(writer["theta"]))
                        for challenge in challenges:
                            for arm in arms:
                                initial = _state_arm(registration, stage, rulebook, aged, history_name, arm)
                                field = np.zeros_like(initial, dtype=np.float32)
                                correct, wrong = _assay(
                                    registration,
                                    stage,
                                    rulebook,
                                    initial,
                                    field,
                                    history_name=history_name,
                                    challenge=challenge,
                                    half=half,
                                    coordinate=(writer["name"], age),
                                    theta=float(writer["theta"]),
                                )
                                records.append(_destination_record(
                                    stage=stage,
                                    source_id=rulebook.source_id,
                                    history_name=history_name,
                                    writer=writer["name"],
                                    arm=arm,
                                    challenge=challenge,
                                    age=age,
                                    checkpoint=None,
                                    half=half,
                                    correct=correct,
                                    wrong=wrong,
                                    acquired=acquisition,
                                    theta=float(writer["theta"]),
                                ))
    return records


def run_boundary_source(registration: Registration, stage: str, rulebook: Rulebook) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    futures = scaled_futures(int(registration.protocol["boundary"]["futures"]), registration)
    writer_lookup = {row["mode"]: row for row in registration.protocol["state"]["writers"]}
    for mode in registration.protocol["boundary"]["writers"]:
        writer = dict(writer_lookup[mode])
        for theta in registration.protocol["boundary"]["thetas"]:
            writer["name"] = f"{mode}-theta-{theta:g}"
            for history_name in HISTORIES:
                for half in HALVES:
                    written, acquisition = _write_state(registration, stage, rulebook, history_name, writer, futures, half, float(theta))
                    for arm in ("state_transplant", "reset"):
                        initial = _state_arm(registration, stage, rulebook, written, history_name, arm)
                        correct, wrong = _assay(
                            registration,
                            stage,
                            rulebook,
                            initial,
                            np.zeros_like(initial, dtype=np.float32),
                            history_name=history_name,
                            challenge="neutral_damage",
                            half=half,
                            coordinate=(mode, float(theta)),
                            theta=float(theta),
                        )
                        records.append(_destination_record(
                            stage=stage,
                            source_id=rulebook.source_id,
                            history_name=history_name,
                            writer=writer["name"],
                            arm=arm,
                            challenge="neutral_damage",
                            age=0,
                            checkpoint=None,
                            half=half,
                            correct=correct,
                            wrong=wrong,
                            acquired=acquisition,
                            theta=float(theta),
                        ))
    return records


def _slow_mark_field(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    history_name: str,
    arm: str,
    half_life: int,
    coupling: float,
    futures: int,
) -> np.ndarray:
    target = rulebook.target(history_name).astype(np.float32)
    mark = np.repeat(target[None, :], futures, axis=0)
    genes = target.size
    if arm in {"reset", "write_disabled"}:
        mark[:] = 0
    elif arm == "shuffle":
        permutation = stable_permutation(genes, str(registration.protocol["master_seed"]), stage.removesuffix("_audit"), "mark-shuffle", rulebook.source_id, history_name)
        mark = mark[:, permutation]
    elif arm == "targeted_ablation":
        ranking = np.argsort(-np.sum(np.abs(rulebook.weights), axis=0))
        mark[:, ranking[: genes // 2]] = 0
    elif arm == "random_ablation":
        selection = stable_permutation(genes, str(registration.protocol["master_seed"]), stage.removesuffix("_audit"), "mark-ablation", rulebook.source_id, history_name)
        mark[:, selection[: genes // 2]] = 0
    elif arm == "rescue":
        pass
    elif arm not in {"mark", "inert"}:
        raise ValueError(arm)
    effective = 0.0 if arm == "inert" else coupling
    decay = 0.5 ** (8.0 / float(half_life))
    return (effective * decay * mark).astype(np.float32)


def run_slow_mark_source(registration: Registration, stage: str, rulebook: Rulebook) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    protocol = registration.protocol["slow_mark"]
    futures = scaled_futures(int(protocol["futures"]), registration)
    for half_life in protocol["half_lives"]:
        for coupling in protocol["couplings"]:
            for history_name in HISTORIES:
                for half in HALVES:
                    for arm in protocol["arms"]:
                        initial = np.repeat(rulebook.neutral[None, :], futures, axis=0)
                        field = _slow_mark_field(registration, stage, rulebook, history_name, arm, int(half_life), float(coupling), futures)
                        correct, wrong = _assay(
                            registration,
                            stage,
                            rulebook,
                            initial,
                            field,
                            history_name=history_name,
                            challenge="forced_break",
                            half=half,
                            coordinate=(int(half_life), float(coupling)),
                            theta=0.0,
                        )
                        records.append(_destination_record(
                            stage=stage,
                            source_id=rulebook.source_id,
                            history_name=history_name,
                            writer="slow_mark",
                            arm=arm,
                            challenge="forced_break",
                            age=8,
                            checkpoint=None,
                            half=half,
                            correct=correct,
                            wrong=wrong,
                            acquired=1.0,
                            half_life=int(half_life),
                            coupling=float(coupling),
                        ))
    return records


def _carrier_mask(rulebook: Rulebook, k: int, targeted: bool, registration: Registration, stage: str, history_name: str) -> np.ndarray:
    genes = rulebook.target_a.size
    mask = np.zeros(genes, dtype=bool)
    if targeted:
        ranking = np.argsort(-np.sum(np.abs(rulebook.weights), axis=0))
    else:
        ranking = stable_permutation(genes, str(registration.protocol["master_seed"]), stage.removesuffix("_audit"), "carrier-random", rulebook.source_id, history_name, k)
    mask[np.asarray(ranking[:k], dtype=int)] = True
    return mask


def _founder_carrier(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    history_name: str,
    arm: str,
    futures: int,
    half: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    carrier_protocol = registration.protocol["carrier"]
    target = rulebook.target(history_name)
    opposite = rulebook.opposite(history_name)
    if arm == "exact_full":
        carrier = np.repeat(target[None, :], futures, axis=0)
    elif arm == "opposite_history":
        carrier = np.repeat(opposite[None, :], futures, axis=0)
    else:
        carrier = np.repeat(target[None, :], futures, axis=0)
        founder_rng = _rng(registration, stage, "founder-write", rulebook.source_id, history_name, half)
        flips = founder_rng.random(carrier.shape) < float(registration.engine["expression_flip_probability"])
        carrier = np.where(flips, -carrier, carrier).astype(np.int8)
    acquired = float(np.mean(in_basin(carrier, target)))
    ttl = np.full(carrier.shape, int(carrier_protocol["retention_cycles"]), dtype=np.int16)
    if arm in {"zero", "write_disabled"}:
        carrier[:] = 0
        ttl[:] = 0
    if arm == "pattern_shuffle":
        permutation = stable_permutation(carrier.shape[1], str(registration.protocol["master_seed"]), stage.removesuffix("_audit"), "carrier-shuffle", rulebook.source_id, history_name)
        carrier = carrier[:, permutation]
        ttl = ttl[:, permutation]
    if arm.startswith("targeted_k"):
        k = int(arm.rsplit("k", 1)[1])
        mask = _carrier_mask(rulebook, k, True, registration, stage, history_name)
        carrier[:, ~mask] = 0
        ttl[:, ~mask] = 0
    if arm.startswith("random_k"):
        k = int(arm.rsplit("k", 1)[1])
        mask = _carrier_mask(rulebook, k, False, registration, stage, history_name)
        carrier[:, ~mask] = 0
        ttl[:, ~mask] = 0
    return carrier.astype(np.int8), ttl, acquired


def _apply_carrier_bottleneck(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    history_name: str,
    arm: str,
    carrier: np.ndarray,
    ttl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if arm.startswith("targeted_k"):
        k = int(arm.rsplit("k", 1)[1])
        mask = _carrier_mask(rulebook, k, True, registration, stage, history_name)
    elif arm.startswith("random_k"):
        k = int(arm.rsplit("k", 1)[1])
        mask = _carrier_mask(rulebook, k, False, registration, stage, history_name)
    else:
        return carrier, ttl
    carrier[:, ~mask] = 0
    ttl[:, ~mask] = 0
    return carrier, ttl


def _advance_carrier(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    history_name: str,
    arm: str,
    carrier: np.ndarray,
    ttl: np.ndarray,
    checkpoint: int,
    half: int,
) -> tuple[np.ndarray, np.ndarray]:
    protocol = registration.protocol
    carrier_protocol = protocol["carrier"]
    retention = int(carrier_protocol["retention_cycles"])
    sweeps = int(protocol["engine"]["generation_sweeps"])
    target = rulebook.target(history_name)
    rewrite = arm != "no_rewrite"
    read_enabled = arm != "read_disabled"
    write_enabled = arm != "write_disabled"
    for generation_index in range(1, checkpoint + 1):
        if arm == "ablate_2_rescue_3" and generation_index == 3:
            carrier[:] = target
            ttl[:] = retention
        initial = np.repeat(rulebook.neutral[None, :], carrier.shape[0], axis=0)
        field = float(carrier_protocol["coupling"]) * carrier if read_enabled else np.zeros_like(carrier, dtype=np.float32)
        trajectory = rollout_jax(
            rulebook.weights,
            initial,
            field.astype(np.float32),
            sweeps=sweeps,
            theta=0.0,
            flip_probability=float(protocol["engine"]["expression_flip_probability"]),
            key_data=_key(registration, stage, "lineage", rulebook.source_id, history_name, generation_index, half),
        )
        candidate = trajectory[-1]
        ttl = ttl - sweeps
        if rewrite and write_enabled:
            same = candidate == carrier
            empty = carrier == 0
            expired = ttl <= 0
            replace = empty | expired
            carrier = np.where(replace, candidate, carrier).astype(np.int8)
            ttl = np.where(same | replace, retention, ttl).astype(np.int16)
        carrier = np.where(ttl > 0, carrier, 0).astype(np.int8)
        ttl = np.maximum(ttl, 0).astype(np.int16)
        carrier, ttl = _apply_carrier_bottleneck(registration, stage, rulebook, history_name, arm, carrier, ttl)
        if arm == "pattern_shuffle":
            permutation = stable_permutation(carrier.shape[1], str(protocol["master_seed"]), stage.removesuffix("_audit"), "carrier-shuffle", rulebook.source_id, history_name)
            carrier = carrier[:, permutation]
            ttl = ttl[:, permutation]
        if arm in {"ablate_generation_2", "ablate_2_rescue_3"} and generation_index == 2:
            carrier[:] = 0
            ttl[:] = 0
    return carrier, ttl


def run_carrier_source(registration: Registration, stage: str, rulebook: Rulebook) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    protocol = registration.protocol
    carrier_protocol = protocol["carrier"]
    futures = scaled_futures(int(carrier_protocol["futures_per_half"]), registration)
    for history_name in HISTORIES:
        for half in HALVES:
            for arm in carrier_protocol["arms"]:
                founder, founder_ttl, acquisition = _founder_carrier(registration, stage, rulebook, history_name, arm, futures, half)
                for checkpoint in carrier_protocol["checkpoints"]:
                    carrier, _ = _advance_carrier(
                        registration,
                        stage,
                        rulebook,
                        history_name,
                        arm,
                        founder.copy(),
                        founder_ttl.copy(),
                        int(checkpoint),
                        half,
                    )
                    for challenge in carrier_protocol["challenges"]:
                        initial = np.repeat(rulebook.neutral[None, :], futures, axis=0)
                        field = float(carrier_protocol["coupling"]) * carrier
                        if arm == "read_disabled":
                            field[:] = 0
                        correct, wrong = _assay(
                            registration,
                            stage,
                            rulebook,
                            initial,
                            field.astype(np.float32),
                            history_name=history_name,
                            challenge=challenge,
                            half=half,
                            coordinate=(arm, int(checkpoint)),
                            theta=0.0,
                        )
                        records.append(_destination_record(
                            stage=stage,
                            source_id=rulebook.source_id,
                            history_name=history_name,
                            writer="natural_latch" if arm != "exact_full" else "exact_latch",
                            arm=arm,
                            challenge=challenge,
                            age=None,
                            checkpoint=int(checkpoint),
                            half=half,
                            correct=correct,
                            wrong=wrong,
                            acquired=acquisition,
                        ))
    return records


def run_stage_shard(registration: Registration, stage: str, worker_index: int, worker_count: int) -> StageResult:
    started = time.monotonic()
    source_count = stage_source_count(stage, registration)
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    runner = {
        "state": run_state_source,
        "boundary": run_boundary_source,
        "slow_mark": run_slow_mark_source,
        "carrier": run_carrier_source,
    }[stage.removesuffix("_audit")]
    for source_id in range(worker_index, source_count, worker_count):
        rulebook = generate_rulebook(source_id, registration.protocol, stage.removesuffix("_audit"))
        sources.append(rulebook_metadata(rulebook))
        records.extend(runner(registration, stage, rulebook))
    simulated = sum(int(row["n"]) for row in records)
    return StageResult(records, sources, time.monotonic() - started, simulated)


def iter_replay_subset(records: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: (
        row["stage"], row["source_id"], row["history"], row["arm"], row["challenge"],
        -1 if row["age"] is None else row["age"],
        -1 if row["checkpoint"] is None else row["checkpoint"], row["half"],
    ))
    return ordered[:limit]

