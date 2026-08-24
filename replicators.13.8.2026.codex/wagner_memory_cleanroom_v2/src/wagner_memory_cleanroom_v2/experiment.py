from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import time
from typing import Any, Callable

import numpy as np

from .config import Registration, half_futures, stage_source_count
from .engine import (
    STATUS_POINT,
    apply_challenge,
    develop_one_cycle_jax,
    exact_match,
    primary_destinations,
    rollout_adult_cycles_jax,
    rollout_latch_cycles_jax,
    rollout_noisy_adult_cycles_jax,
    sequential_sweep_numpy,
    states_from_int,
    states_to_int,
    strict_destinations,
)
from .rng import generator, jax_key_data, stable_derangement, stable_permutation
from .source import Rulebook, generate_rulebook, rulebook_record


HISTORIES = ("A", "B")
HALVES = (0, 1)
OUTCOME_A = 1
OUTCOME_B = 2
OUTCOME_OTHER = 3


@dataclass(frozen=True)
class AssayOutcome:
    destinations: np.ndarray
    hold_a: np.ndarray
    hold_b: np.ndarray
    trajectory_digest: str


@dataclass
class StageResult:
    records: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    replay: list[dict[str, Any]]
    elapsed_seconds: float
    simulated_futures: int


def _base_stage(stage: str) -> str:
    return stage.removesuffix("_audit")


def _key(registration: Registration, stage: str, *coordinates: Any) -> np.ndarray:
    return jax_key_data(str(registration.protocol["master_seed"]), _base_stage(stage), *coordinates)


def _rng(registration: Registration, stage: str, *coordinates: Any) -> np.random.Generator:
    return generator(str(registration.protocol["master_seed"]), _base_stage(stage), *coordinates)


def _cell_id(fields: dict[str, Any]) -> str:
    ordered = (
        fields["stage"], fields["source_id"], fields["midpoint"], fields["history"],
        fields["condition"], fields.get("schedule"), fields["arm"], fields["challenge"],
        fields.get("age"), fields.get("checkpoint"), fields.get("theta"),
        fields.get("half_life"), fields.get("coupling"), fields["half"],
    )
    return "|".join("-" if value is None else str(value) for value in ordered)


def _future_id(cell_id: str, future_index: int) -> str:
    return f"{cell_id}|{future_index}"


def _append_record(
    records: list[dict[str, Any]],
    replay: list[dict[str, Any]],
    replay_limit: int,
    *,
    stage: str,
    source_id: int,
    midpoint: int,
    history: str,
    condition: str,
    arm: str,
    challenge: str,
    half: int,
    outcomes: AssayOutcome,
    acquired_exact: float,
    age: int | None = None,
    checkpoint: int | None = None,
    theta: float | None = None,
    half_life: int | None = None,
    coupling: float | None = None,
    schedule: str | None = None,
) -> None:
    values = np.asarray(outcomes.destinations, dtype=np.uint8)
    hold_a = np.asarray(outcomes.hold_a, dtype=bool)
    hold_b = np.asarray(outcomes.hold_b, dtype=bool)
    if values.shape != hold_a.shape or values.shape != hold_b.shape:
        raise ValueError("assay endpoint arrays are not aligned")
    packed = np.column_stack((values, hold_a.astype(np.uint8), hold_b.astype(np.uint8)))
    fields: dict[str, Any] = {
        "stage": _base_stage(stage),
        "source_id": int(source_id),
        "midpoint": int(midpoint),
        "history": history,
        "condition": condition,
        "arm": arm,
        "challenge": challenge,
        "age": age,
        "checkpoint": checkpoint,
        "theta": theta,
        "half_life": half_life,
        "coupling": coupling,
        "schedule": schedule,
        "half": int(half),
        "n": int(values.size),
        "dest_a": int(np.sum(values == OUTCOME_A)),
        "dest_b": int(np.sum(values == OUTCOME_B)),
        "dest_other": int(np.sum(values == OUTCOME_OTHER)),
        "hold_a": int(np.sum(hold_a)),
        "hold_b": int(np.sum(hold_b)),
        "hold_both": int(np.sum(hold_a & hold_b)),
        "acquired_exact": float(acquired_exact),
        "future_digest": sha256(packed.tobytes(order="C")).hexdigest(),
        "trajectory_digest": outcomes.trajectory_digest,
    }
    fields["cell_id"] = _cell_id(fields)
    records.append(fields)
    remaining = max(0, replay_limit - len(replay))
    for index in range(min(remaining, values.size)):
        replay.append({
            "future_id": _future_id(fields["cell_id"], index),
            "destination": int(values[index]),
            "hold_a": bool(hold_a[index]),
            "hold_b": bool(hold_b[index]),
        })


def _adult_cycles(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    initial: np.ndarray,
    *,
    cycles: int,
    coordinate: tuple[Any, ...],
    gamma_variance: float = 0.0,
    read_mode: str = "none",
    mark: np.ndarray | None = None,
    coupling: float = 0.0,
    half_life: int = 0,
    read_enabled: bool = True,
    write_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    flip_probability = float(registration.engine["expression_flip_probability"])
    key = _key(registration, stage, "adult-cycles", rulebook.source_id, *coordinate)
    if gamma_variance > 0.0:
        if read_mode != "none" or mark is not None:
            raise ValueError("registered noisy boundary does not combine with an auxiliary mark")
        adults, adult, statuses = rollout_noisy_adult_cycles_jax(
            rulebook.weights,
            initial,
            cycles=cycles,
            max_sweeps=int(registration.engine["max_sweeps"]),
            gamma_variance=gamma_variance,
            expression_flip_probability=flip_probability,
            key_data=key,
        )
        return adults, adult, np.zeros_like(initial, dtype=np.float64), statuses
    adults, adult, final_mark = rollout_adult_cycles_jax(
        rulebook.adult_table(),
        initial,
        cycles=cycles,
        expression_flip_probability=flip_probability,
        key_data=key,
        read_mode=read_mode,
        mark=mark,
        coupling=coupling,
        half_life=half_life,
        read_enabled=read_enabled,
        write_enabled=write_enabled,
    )
    return adults, adult, final_mark, None


def _assay(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    initial: np.ndarray,
    *,
    history: str,
    midpoint: int,
    challenge: str,
    half: int,
    coordinate: tuple[Any, ...],
    gamma_variance: float = 0.0,
    read_mode: str = "none",
    mark: np.ndarray | None = None,
    coupling: float = 0.0,
    half_life: int = 0,
    read_enabled: bool = True,
    write_enabled: bool = True,
    latch_carrier: np.ndarray | None = None,
    latch_ttl: np.ndarray | None = None,
) -> AssayOutcome:
    # Arm is absent from all stochastic coordinates. Interventions therefore
    # share challenge, expression-flip, mark-read, and regulatory-noise draws.
    challenge_rng = _rng(
        registration, stage, "challenge", rulebook.source_id, midpoint, history,
        challenge, half, *coordinate,
    )
    challenged = apply_challenge(
        initial,
        challenge,
        forced_state=rulebook.forced(history),
        neutral_damage_fraction=float(registration.protocol["state"]["neutral_damage_fraction"]),
        rng=challenge_rng,
    )
    assay_coordinate = ("assay", midpoint, history, challenge, half, *coordinate)
    if latch_carrier is not None or latch_ttl is not None:
        if latch_carrier is None or latch_ttl is None:
            raise ValueError("latch carrier and TTL must be supplied together")
        if read_mode != "none" or mark is not None or gamma_variance > 0.0:
            raise ValueError("latch assay cannot be combined with another carrier or regulatory noise")
        zeros_carrier = np.zeros_like(latch_carrier, dtype=np.int8)
        zeros_ttl = np.zeros_like(latch_ttl, dtype=np.int16)
        adults, _, _, _, _, _ = rollout_latch_cycles_jax(
            rulebook.adult_table(),
            challenged,
            np.asarray(latch_carrier, dtype=np.int8),
            np.asarray(latch_ttl, dtype=np.int16),
            zeros_carrier,
            zeros_ttl,
            cycles=int(registration.engine["horizon_cycles"]),
            expression_flip_probability=float(registration.engine["expression_flip_probability"]),
            coupling=coupling,
            retention=int(registration.protocol["carrier"]["retention_cycles"]),
            threshold=int(registration.protocol["carrier"]["write_threshold"]),
            read_enabled=read_enabled,
            rewrite=False,
            key_data=_key(registration, stage, "adult-cycles", rulebook.source_id, *assay_coordinate),
        )
    else:
        adults, _, _, statuses = _adult_cycles(
            registration,
            stage,
            rulebook,
            challenged,
            cycles=int(registration.engine["horizon_cycles"]),
            coordinate=assay_coordinate,
            gamma_variance=gamma_variance,
            read_mode=read_mode,
            mark=mark,
            coupling=coupling,
            half_life=half_life,
            read_enabled=read_enabled,
            write_enabled=write_enabled,
        )
    if latch_carrier is not None:
        statuses = None
    valid_points = None if statuses is None else statuses == STATUS_POINT
    hold_a, hold_b = strict_destinations(
        adults,
        rulebook.target_a,
        rulebook.target_b,
        int(registration.engine["strict_run"]),
        valid_points,
    )
    point_states = states_from_int(
        [cycle[0] for cycle in rulebook.landscape.attractors if len(cycle) == 1],
        rulebook.target_a.size,
    )
    destinations = primary_destinations(
        adults,
        rulebook.target_a,
        rulebook.target_b,
        point_states,
        int(registration.engine["stable_run"]),
        valid_points,
    )
    return AssayOutcome(
        destinations,
        hold_a,
        hold_b,
        sha256(np.asarray(adults, dtype=np.int8).tobytes(order="C")).hexdigest(),
    )


def _write_state(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    history: str,
    midpoint: int,
    writer: dict[str, Any],
    futures: int,
    half: int,
    theta_override: float | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    target = rulebook.target(history)
    state = np.repeat(rulebook.midpoints[midpoint][None, :], futures, axis=0)
    trajectory: list[np.ndarray] = []
    gamma_variance = float(writer["theta"] if theta_override is None else theta_override)
    for dwell_index in range(int(writer["dwell"])):
        field = np.zeros_like(state, dtype=np.float64)
        hard_mask = np.zeros(target.size, dtype=bool)
        hard_values = state
        if writer["mode"] == "persistent_hard":
            hard_mask[:] = True
            hard_values = np.repeat(target[None, :], futures, axis=0)
        elif writer["mode"] == "target_field":
            row_norms = np.linalg.norm(rulebook.weights, axis=1)
            vector = float(writer["strength"]) * row_norms * target
            field = np.repeat(vector[None, :], futures, axis=0)
        else:
            raise ValueError(writer["mode"])
        state, status, _ = develop_one_cycle_jax(
            rulebook.weights,
            state,
            external_field=field,
            gamma_variance=gamma_variance,
            expression_flip_probability=float(registration.engine["expression_flip_probability"]),
            key_data=_key(
                registration, stage, "writer", rulebook.source_id, midpoint, history,
                writer["mode"], half, gamma_variance, dwell_index, futures,
            ),
            max_sweeps=int(registration.engine["max_sweeps"]),
            hard_mask=hard_mask,
            hard_values=hard_values,
        )
        trajectory.append(state.copy())
    acquired = float(np.mean(exact_match(state, target) & (status == STATUS_POINT)))
    return state, acquired, np.stack(trajectory)


def _age_state(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    state: np.ndarray,
    history: str,
    midpoint: int,
    condition: str,
    age: int,
    half: int,
    gamma_variance: float,
) -> np.ndarray:
    if age == 0:
        return state.copy()
    _, adult, _, _ = _adult_cycles(
        registration,
        stage,
        rulebook,
        state,
        cycles=age,
        coordinate=("age", midpoint, history, condition, age, half, state.shape[0]),
        gamma_variance=gamma_variance,
    )
    return adult


def _fixed_derangement(registration: Registration, stage: str, rulebook: Rulebook, label: str) -> np.ndarray:
    return stable_derangement(
        rulebook.target_a.size,
        str(registration.protocol["master_seed"]),
        _base_stage(stage), label, rulebook.source_id,
    )


def _state_arm(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    written: np.ndarray,
    midpoint: int,
    history: str,
    arm: str,
) -> np.ndarray:
    futures = written.shape[0]
    if arm in {"self_continuation", "state_transplant"}:
        return written.copy()
    if arm == "reset_both":
        return np.repeat(rulebook.midpoints[midpoint][None, :], futures, axis=0)
    if arm == "destination_matched_donor":
        return np.repeat(rulebook.target(history)[None, :], futures, axis=0)
    if arm == "descriptor_matched_null":
        permutation = _fixed_derangement(registration, stage, rulebook, f"descriptor-{history}-{midpoint}")
        return written[:, permutation]
    if arm == "state_pattern_shuffle":
        permutation = _fixed_derangement(registration, stage, rulebook, f"state-shuffle-{history}")
        return written[:, permutation]
    raise ValueError(arm)


def _state_schedule(registration: Registration) -> list[tuple[list[str], list[str], list[int], int]]:
    state = registration.protocol["state"]
    primary = half_futures(int(state["primary_futures_per_cell"]), registration)
    persistence = half_futures(int(state["persistence_futures_per_cell"]), registration)
    return [
        (list(state["arms"]), ["neutral_damage"], [0], primary),
        (["state_transplant", "reset_both"], ["forced_break"], [0], primary),
        (["state_transplant", "reset_both", "state_pattern_shuffle"], list(state["challenges"]), [1, 2, 4, 8], persistence),
    ]


def run_state_source(registration: Registration, stage: str, rulebook: Rulebook) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    limit = int(registration.operations["replay_futures_per_stage"])
    for writer in registration.protocol["state"]["writers"]:
        condition = str(writer["name"])
        for history in HISTORIES:
            for midpoint in range(int(registration.engine["midpoint_count"])):
                for half in HALVES:
                    cache: dict[int, tuple[np.ndarray, float]] = {}
                    for arms, challenges, ages, futures in _state_schedule(registration):
                        if futures not in cache:
                            state, acquired, _ = _write_state(
                                registration, stage, rulebook, history, midpoint, writer, futures, half
                            )
                            cache[futures] = (state, acquired)
                        base, acquired = cache[futures]
                        for age in ages:
                            aged = _age_state(
                                registration, stage, rulebook, base, history, midpoint,
                                condition, age, half, float(writer["theta"]),
                            )
                            for challenge in challenges:
                                for arm in arms:
                                    initial = _state_arm(
                                        registration, stage, rulebook, aged, midpoint, history, arm
                                    )
                                    outcomes = _assay(
                                        registration, stage, rulebook, initial,
                                        history=history, midpoint=midpoint, challenge=challenge,
                                        half=half, coordinate=(condition, age, futures),
                                        gamma_variance=float(writer["theta"]),
                                    )
                                    _append_record(
                                        records, replay, limit, stage=stage,
                                        source_id=rulebook.source_id, midpoint=midpoint,
                                        history=history, condition=condition, arm=arm,
                                        challenge=challenge, age=age, half=half,
                                        outcomes=outcomes, acquired_exact=acquired,
                                        theta=float(writer["theta"]),
                                    )
    return records, replay


def run_boundary_source(registration: Registration, stage: str, rulebook: Rulebook) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    limit = int(registration.operations["replay_futures_per_stage"])
    futures = half_futures(int(registration.protocol["boundary"]["futures_per_cell"]), registration)
    lookup = {row["mode"]: row for row in registration.protocol["state"]["writers"]}
    for mode in registration.protocol["boundary"]["writers"]:
        for theta_value in registration.protocol["boundary"]["thetas"]:
            theta = float(theta_value)
            writer = dict(lookup[mode])
            condition = f"{mode}-theta-{theta:g}"
            for history in HISTORIES:
                for midpoint in range(int(registration.engine["midpoint_count"])):
                    for half in HALVES:
                        written, acquired, _ = _write_state(
                            registration, stage, rulebook, history, midpoint, writer, futures, half, theta
                        )
                        for arm in ("state_transplant", "reset_both"):
                            initial = _state_arm(registration, stage, rulebook, written, midpoint, history, arm)
                            outcomes = _assay(
                                registration, stage, rulebook, initial,
                                history=history, midpoint=midpoint, challenge="neutral_damage",
                                half=half, coordinate=(mode, theta, futures), gamma_variance=theta,
                            )
                            _append_record(
                                records, replay, limit, stage=stage,
                                source_id=rulebook.source_id, midpoint=midpoint,
                                history=history, condition=condition, arm=arm,
                                challenge="neutral_damage", age=0, half=half,
                                outcomes=outcomes, acquired_exact=acquired, theta=theta,
                            )
    return records, replay


def _one_step_sensitivity(rulebook: Rulebook, history: str) -> np.ndarray:
    genes = rulebook.target_a.size
    states = states_from_int(np.arange(1 << genes, dtype=np.uint16), genes)
    # Exact Boolean influence of each matching carrier coordinate on one full
    # sequential recipient sweep, averaged over the entire expression state
    # space.  The history parameter is intentionally outcome-blind; it remains
    # in the signature because carrier magnitude can be history-specific in
    # more general arms, while the selected full latch has unit magnitude.
    _ = history
    baseline = sequential_sweep_numpy(rulebook.weights, states)
    scores = np.zeros(genes, dtype=float)
    for gene in range(genes):
        perturbed = states.copy()
        perturbed[:, gene] *= -1
        changed = sequential_sweep_numpy(rulebook.weights, perturbed) != baseline
        scores[gene] = float(np.mean(changed))
    return scores


def _targeted_mask(rulebook: Rulebook, history: str, k: int) -> np.ndarray:
    influence = _one_step_sensitivity(rulebook, history)
    ranking = np.lexsort((np.arange(influence.size), -influence))
    mask = np.zeros(influence.size, dtype=bool)
    mask[ranking[:k]] = True
    return mask


def _random_mask(registration: Registration, stage: str, rulebook: Rulebook, history: str, midpoint: int, k: int) -> np.ndarray:
    ranking = stable_permutation(
        rulebook.target_a.size,
        str(registration.protocol["master_seed"]),
        _base_stage(stage), "random-bottleneck", rulebook.source_id, history, midpoint, k,
    )
    mask = np.zeros(rulebook.target_a.size, dtype=bool)
    mask[np.asarray(ranking[:k], dtype=int)] = True
    return mask


def _trained_mark(trajectory: np.ndarray, half_life: int) -> np.ndarray:
    rho = 2.0 ** (-1.0 / float(half_life))
    mark = np.zeros_like(trajectory[0], dtype=np.float64)
    for adult in trajectory:
        mark = rho * mark + (1.0 - rho) * adult.astype(np.float64)
    return mark


def _aged_mark_donor(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    history: str,
    midpoint: int,
    age: int,
    half_life: int,
    coupling: float,
    futures: int,
    half: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = rulebook.target(history)
    aged_state = np.repeat(target[None, :], futures, axis=0)
    mark = _trained_mark(aged_state[None, :, :], half_life)
    if age > 0:
        # Washout age advances the complete donor system.  The inherited mark
        # is read and rewritten after every realized adult, exactly as in the
        # registered cycle order; it is not held frozen while expression alone
        # ages.  Arm is deliberately absent from the stochastic coordinates.
        _, aged_state, mark, _ = _adult_cycles(
            registration,
            stage,
            rulebook,
            aged_state,
            cycles=age,
            coordinate=(
                "mark-donor-age", midpoint, history, half_life, coupling,
                age, half, futures,
            ),
            read_mode="recurrent",
            mark=mark,
            coupling=coupling,
            half_life=half_life,
            read_enabled=True,
            write_enabled=True,
        )
    return aged_state, mark


def _mark_arm(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    history: str,
    midpoint: int,
    arm: str,
    age: int,
    half_life: int,
    coupling: float,
    futures: int,
    half: int,
    donor: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, bool, bool]:
    if donor is None:
        aged_state, mark = _aged_mark_donor(
            registration, stage, rulebook, history, midpoint, age,
            half_life, coupling, futures, half,
        )
    else:
        aged_state, mark = (np.asarray(value).copy() for value in donor)
    midpoint_state = np.repeat(rulebook.midpoints[midpoint][None, :], futures, axis=0)
    initial = midpoint_state
    read_enabled = True
    write_enabled = True
    if arm == "state_and_mark_transplant":
        initial = aged_state
    elif arm == "reset_both":
        mark[:] = 0.0
    elif arm == "write_disabled":
        mark[:] = 0.0
        write_enabled = False
    elif arm == "mark_pattern_shuffle":
        permutation = _fixed_derangement(registration, stage, rulebook, f"mark-shuffle-{history}")
        mark = mark[:, permutation]
    elif arm == "mark_inert":
        read_enabled = False
    elif arm == "mark_ablation":
        # Ablate the two most influential coordinates.  The carrier
        # bottleneck arms below *retain* their ranked mask, whereas this arm
        # removes its ranked mask; keeping those two meanings explicit avoids
        # silently turning the targeted ablation into a weakest-two ablation.
        ablated = _targeted_mask(rulebook, history, 2)
        mark[:, ablated] = 0.0
    elif arm == "mark_random_ablation":
        ablated = _random_mask(
            registration, stage, rulebook, history, midpoint, 2
        )
        mark[:, ablated] = 0.0
    elif arm in {"mark_transplant", "mark_rescue"}:
        pass
    else:
        raise ValueError(arm)
    return initial, mark, 1.0, read_enabled, write_enabled


def run_slow_mark_source(registration: Registration, stage: str, rulebook: Rulebook) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    limit = int(registration.operations["replay_futures_per_stage"])
    protocol = registration.protocol["slow_mark"]
    schedules = (
        ("screen", list(protocol["screen_arms"]), list(protocol["screen_challenges"]), half_futures(int(protocol["screen_futures_per_cell"]), registration)),
        ("mechanism", list(protocol["mechanism_arms"]), list(protocol["mechanism_challenges"]), half_futures(int(protocol["mechanism_futures_per_cell"]), registration)),
    )
    for half_life_value in protocol["half_lives"]:
        half_life = int(half_life_value)
        for coupling_value in protocol["couplings"]:
            coupling = float(coupling_value)
            condition = f"half-{half_life}.mu-{coupling:g}"
            for history in HISTORIES:
                for midpoint in range(int(registration.engine["midpoint_count"])):
                    for half in HALVES:
                        for schedule_name, arms, challenges, futures in schedules:
                            for age_value in protocol["ages"]:
                                age = int(age_value)
                                donor = _aged_mark_donor(
                                    registration, stage, rulebook, history, midpoint,
                                    age, half_life, coupling, futures, half,
                                )
                                for challenge in challenges:
                                    for arm in arms:
                                        initial, mark, acquired, read_enabled, write_enabled = _mark_arm(
                                            registration, stage, rulebook, history, midpoint, arm,
                                            age, half_life, coupling, futures, half, donor,
                                        )
                                        outcomes = _assay(
                                            registration, stage, rulebook, initial,
                                            history=history, midpoint=midpoint, challenge=challenge,
                                            half=half, coordinate=(condition, schedule_name, age, futures),
                                            read_mode="recurrent", mark=mark, coupling=coupling,
                                            half_life=half_life, read_enabled=read_enabled,
                                            write_enabled=write_enabled,
                                        )
                                        _append_record(
                                            records, replay, limit, stage=stage,
                                            source_id=rulebook.source_id, midpoint=midpoint,
                                            history=history, condition=condition, arm=arm,
                                            challenge=challenge, age=age, half=half,
                                            outcomes=outcomes, acquired_exact=acquired,
                                            half_life=half_life, coupling=coupling,
                                            schedule=schedule_name,
                                        )
    return records, replay


def _update_latch(
    carrier: np.ndarray,
    ttl: np.ndarray,
    pending: np.ndarray,
    streak: np.ndarray,
    adult_trajectory: np.ndarray,
    *,
    retention: int,
    threshold: int,
    rewrite: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    carrier = carrier.copy()
    ttl = ttl.copy()
    pending = pending.copy()
    streak = streak.copy()
    for expression in np.asarray(adult_trajectory, dtype=np.int8):
        ttl = np.maximum(ttl - 1, 0).astype(np.int16)
        if not rewrite:
            carrier = np.where(ttl > 0, carrier, 0).astype(np.int8)
            continue
        same_pending = pending == expression
        pending = expression.copy()
        streak = np.where(same_pending, streak + 1, 1).astype(np.int16)
        matching = (carrier != 0) & (expression == carrier)
        ttl = np.where(matching, retention, ttl).astype(np.int16)
        writable = ((carrier == 0) | (ttl <= 0)) & (streak >= threshold)
        carrier = np.where(writable, expression, carrier).astype(np.int8)
        ttl = np.where(writable, retention, ttl).astype(np.int16)
        carrier = np.where(ttl > 0, carrier, 0).astype(np.int8)
    return carrier, ttl, pending, streak


def _apply_carrier_intervention(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    history: str,
    midpoint: int,
    arm: str,
    carrier: np.ndarray,
    ttl: np.ndarray,
    pending: np.ndarray,
    streak: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    carrier, ttl, pending, streak = (value.copy() for value in (carrier, ttl, pending, streak))
    mask: np.ndarray | None = None
    if arm == "zero":
        mask = np.zeros(rulebook.target_a.size, dtype=bool)
    elif arm.startswith("targeted_k"):
        mask = _targeted_mask(rulebook, history, int(arm.rsplit("k", 1)[1]))
    elif arm.startswith("random_k"):
        mask = _random_mask(
            registration, stage, rulebook, history, midpoint, int(arm.rsplit("k", 1)[1])
        )
    if mask is not None:
        carrier[:, ~mask] = 0
        ttl[:, ~mask] = 0
        pending[:, ~mask] = 0
        streak[:, ~mask] = 0
    if arm == "pattern_shuffle":
        permutation = _fixed_derangement(registration, stage, rulebook, f"carrier-shuffle-{history}")
        carrier = carrier[:, permutation]
        ttl = ttl[:, permutation]
        pending = pending[:, permutation]
        streak = streak[:, permutation]
    return carrier, ttl, pending, streak


def _founder_latch(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    history: str,
    midpoint: int,
    arm: str,
    futures: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    protocol = registration.protocol["carrier"]
    genes = rulebook.target_a.size
    retention = int(protocol["retention_cycles"])
    threshold = int(protocol["write_threshold"])
    carrier = np.zeros((futures, genes), dtype=np.int8)
    ttl = np.zeros((futures, genes), dtype=np.int16)
    pending = np.zeros((futures, genes), dtype=np.int8)
    streak = np.zeros((futures, genes), dtype=np.int16)
    founder_history = "B" if arm == "opposite_history" and history == "A" else "A" if arm == "opposite_history" else history
    if arm == "exact_full":
        carrier[:] = rulebook.target(history)
        ttl[:] = retention
    elif arm != "write_disabled":
        # The selected natural writer is a full dwell-one hard clamp. Its one
        # observed adult, rather than the requested label, writes the latch.
        adult_trajectory = np.repeat(
            rulebook.target(founder_history)[None, None, :], futures, axis=1
        )
        carrier, ttl, pending, streak = _update_latch(
            carrier, ttl, pending, streak, adult_trajectory,
            retention=retention, threshold=threshold, rewrite=True,
        )
    carrier, ttl, pending, streak = _apply_carrier_intervention(
        registration, stage, rulebook, history, midpoint, arm,
        carrier, ttl, pending, streak,
    )
    acquired = float(np.mean(np.all(carrier == rulebook.target(history), axis=1)))
    return carrier, ttl, pending, streak, acquired


def _lineage_snapshots(
    registration: Registration,
    stage: str,
    rulebook: Rulebook,
    history: str,
    midpoint: int,
    arm: str,
    half: int,
    futures: int,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], float]:
    protocol = registration.protocol["carrier"]
    retention = int(protocol["retention_cycles"])
    threshold = int(protocol["write_threshold"])
    carrier, ttl, pending, streak, acquired = _founder_latch(
        registration, stage, rulebook, history, midpoint, arm, futures
    )
    checkpoints = set(int(value) for value in protocol["checkpoints"])
    snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = {
        0: (carrier.copy(), ttl.copy())
    }
    for generation in range(1, max(checkpoints) + 1):
        if arm == "ablate_2_rescue_3" and generation == 3:
            carrier, ttl, pending, streak, _ = _founder_latch(
                registration, stage, rulebook, history, midpoint, "natural_full", futures
            )
        initial = np.repeat(rulebook.midpoints[midpoint][None, :], futures, axis=0)
        adults, _, carrier, ttl, pending, streak = rollout_latch_cycles_jax(
            rulebook.adult_table(),
            initial,
            carrier,
            ttl,
            pending,
            streak,
            cycles=int(registration.engine["generation_cycles"]),
            expression_flip_probability=float(registration.engine["expression_flip_probability"]),
            coupling=float(protocol["coupling"]),
            retention=retention,
            threshold=threshold,
            read_enabled=arm != "read_disabled",
            rewrite=arm not in {"no_rewrite", "write_disabled"},
            key_data=_key(
                registration, stage, "latch-lineage", rulebook.source_id,
                midpoint, history, generation, half, futures,
            ),
        )
        carrier, ttl, pending, streak = _apply_carrier_intervention(
            registration, stage, rulebook, history, midpoint, arm,
            carrier, ttl, pending, streak,
        )
        if arm in {"ablate_generation_2", "ablate_2_rescue_3"} and generation == 2:
            carrier[:] = 0
            ttl[:] = 0
            pending[:] = 0
            streak[:] = 0
        if generation in checkpoints:
            snapshots[generation] = (carrier.copy(), ttl.copy())
    return snapshots, acquired


def run_carrier_source(registration: Registration, stage: str, rulebook: Rulebook) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    limit = int(registration.operations["replay_futures_per_stage"])
    protocol = registration.protocol["carrier"]
    futures = half_futures(int(protocol["futures_per_cell"]), registration)
    condition = f"latch.threshold-{int(protocol['write_threshold'])}.retention-{int(protocol['retention_cycles'])}.mu-{float(protocol['coupling']):g}"
    for history in HISTORIES:
        for midpoint in range(int(registration.engine["midpoint_count"])):
            for half in HALVES:
                for arm in protocol["arms"]:
                    snapshots, acquired = _lineage_snapshots(
                        registration, stage, rulebook, history, midpoint, arm, half, futures
                    )
                    for checkpoint_value in protocol["checkpoints"]:
                        checkpoint = int(checkpoint_value)
                        carrier, ttl = snapshots[checkpoint]
                        initial = np.repeat(rulebook.midpoints[midpoint][None, :], futures, axis=0)
                        for challenge in protocol["challenges"]:
                            outcomes = _assay(
                                registration, stage, rulebook, initial,
                                history=history, midpoint=midpoint, challenge=challenge,
                                half=half, coordinate=("latch", checkpoint, futures),
                                coupling=float(protocol["coupling"]),
                                read_enabled=arm != "read_disabled", write_enabled=False,
                                latch_carrier=carrier, latch_ttl=ttl,
                            )
                            _append_record(
                                records, replay, limit, stage=stage,
                                source_id=rulebook.source_id, midpoint=midpoint,
                                history=history, condition=condition, arm=arm,
                                challenge=challenge, checkpoint=checkpoint, half=half,
                                outcomes=outcomes, acquired_exact=acquired,
                            )
    return records, replay


def run_stage_shard(
    registration: Registration,
    stage: str,
    worker_index: int,
    worker_count: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
    source_domain: str | None = None,
) -> StageResult:
    started = time.monotonic()
    source_count = stage_source_count(stage, registration)
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    simulated = 0
    runner = {
        "state": run_state_source,
        "boundary": run_boundary_source,
        "slow_mark": run_slow_mark_source,
        "carrier": run_carrier_source,
    }[_base_stage(stage)]
    replay_limit = int(registration.operations["replay_futures_per_stage"])
    assigned = list(range(worker_index, source_count, worker_count))
    for position, source_id in enumerate(assigned, start=1):
        domain = source_domain or f"{registration.profile_name}:{_base_stage(stage)}"
        rulebook = generate_rulebook(source_id, registration.protocol, domain)
        sources.append(rulebook_record(rulebook))
        source_records, source_replay = runner(registration, stage, rulebook)
        records.extend(source_records)
        simulated += sum(int(row["n"]) for row in source_records)
        if len(replay) < replay_limit:
            replay.extend(source_replay[: replay_limit - len(replay)])
        if progress is not None:
            elapsed = time.monotonic() - started
            progress({
                "stage": stage,
                "worker": worker_index,
                "sources_complete": position,
                "sources_total": len(assigned),
                "last_source_id": source_id,
                "records": len(records),
                "simulated_futures": simulated,
                "elapsed_seconds": elapsed,
                "eta_seconds": (elapsed / position) * (len(assigned) - position),
            })
    return StageResult(records, sources, replay, time.monotonic() - started, simulated)
