from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

from .dynamics import CYCLE, NONCONVERGENT, POINT, Rulebook, hamming, sample_rulebook
from .protocol import digest, ensure_registration, write_json_atomic
from .rng import generator
from .storage import load_rulebook, save_npz_atomic, save_rulebook, update_status


ARM_NAMES = (
    "self_continuation", "state_transplant", "reset",
    "destination_donor", "descriptor_null", "state_shuffle",
)
ARM_CODE = {name: index for index, name in enumerate(ARM_NAMES)}
CONDITION_CODE = {"primary": 0, "recurrence": 1, "persistence": 2}
CHALLENGE_CODE = {"release": 0, "neutral_damage": 1, "forced_break": 2}


PRIMARY_DTYPE = np.dtype([
    ("condition", "u1"), ("arm", "u1"), ("history", "u1"),
    ("midpoint", "u1"), ("challenge", "u1"), ("age", "u1"),
    ("future", "u2"), ("half", "u1"), ("destination", "i1"),
    ("match", "u1"), ("gen1_destination", "i1"), ("gen1_match", "u1"), ("hold_pre", "u1"),
    ("f12_event", "u1"), ("strict_event", "u1"), ("trajectory_digest", "u8"),
])


def expected_rows(protocol: dict[str, Any]) -> int:
    per_source = 0
    for condition in protocol["conditions"]:
        per_source += (
            2 * 2 * len(condition["arms"]) * len(condition["challenges"])
            * len(condition["ages"]) * int(condition["futures"])
        )
    return per_source * int(protocol["source_count"])


def source_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "sources").glob("source_*.npz"))


def sample_sources(run_dir: Path, protocol: dict[str, Any], deadline: float | None = None) -> list[Path]:
    ensure_registration(run_dir, protocol)
    source_dir = run_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    wanted = int(protocol["source_count"])
    existing = source_paths(run_dir)
    if len(existing) >= wanted:
        return existing[:wanted]
    next_proposal = 0
    if existing:
        next_proposal = max(load_rulebook(path).proposal_index for path in existing) + 1
    accepted = len(existing)
    label = str(protocol["master_seed_label"])
    maximum = int(protocol["maximum_source_proposals"])
    while accepted < wanted and next_proposal < maximum:
        if deadline is not None and time.monotonic() >= deadline:
            break
        rulebook = sample_rulebook(
            label, next_proposal, genes=int(protocol["genes"]),
            max_sweeps=int(protocol["max_sweeps"]),
            minimum_basin=float(protocol["minimum_basin_fraction"]),
            minimum_distance=float(protocol["minimum_form_distance"]),
        )
        if rulebook is not None:
            path = source_dir / f"source_{accepted:04d}.npz"
            save_rulebook(path, rulebook)
            accepted += 1
            update_status(run_dir, phase="sampling", accepted_sources=accepted, last_proposal=next_proposal)
        next_proposal += 1
    if accepted < wanted:
        raise RuntimeError(f"source sampling incomplete: {accepted}/{wanted}")
    index = [
        {"source_index": i, "file": path.name, "proposal_index": load_rulebook(path).proposal_index}
        for i, path in enumerate(source_paths(run_dir)[:wanted])
    ]
    write_json_atomic(run_dir / "source_index.json", {"sources": index})
    return source_paths(run_dir)[:wanted]


def _flip_masks(rng: np.random.Generator, rows: int, genes: int, probability: float) -> np.ndarray:
    bits = rng.random((rows, genes)) < probability
    powers = np.uint16(1) << np.arange(genes, dtype=np.uint16)
    return (bits.astype(np.uint16) * powers).sum(axis=1, dtype=np.uint16)


def _damage_masks(rng: np.random.Generator, rows: int, genes: int, bit_count: int) -> np.ndarray:
    priorities = rng.random((rows, genes))
    chosen = np.argpartition(priorities, bit_count - 1, axis=1)[:, :bit_count]
    masks = np.zeros(rows, dtype=np.uint16)
    for column in range(bit_count):
        masks |= np.uint16(1) << chosen[:, column].astype(np.uint16)
    return masks


def _forced_states(states: np.ndarray, rulebook: Rulebook) -> np.ndarray:
    result = states.copy()
    for row, state_value in enumerate(states):
        state = int(state_value)
        current_point = int(rulebook.landscape.point_index[state])
        chosen = state ^ (1 << int(rulebook.mark_permutation[0]))
        for gene in rulebook.mark_permutation:
            candidate = state ^ (1 << int(gene))
            if int(rulebook.landscape.point_index[candidate]) != current_point:
                chosen = candidate
                break
        result[row] = np.uint16(chosen)
    return result


def _develop(states: np.ndarray, rulebook: Rulebook) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    adult = rulebook.landscape.adult[states]
    kind = rulebook.landscape.kind[states]
    point_index = rulebook.landscape.point_index[states]
    return adult, kind, point_index


def _destination_class(point_index: np.ndarray, kind: np.ndarray, rulebook: Rulebook) -> np.ndarray:
    result = np.full(len(point_index), 2, dtype=np.int8)
    result[kind == CYCLE] = 3
    result[kind == NONCONVERGENT] = 4
    for history in range(2):
        result[(kind == POINT) & (point_index == rulebook.target_point_indices[history])] = history
    return result


def _events(trajectory: np.ndarray, old_state: np.ndarray, genes: int) -> tuple[np.ndarray, np.ndarray]:
    rows, horizon = trajectory.shape
    f12 = np.zeros(rows, dtype=np.uint8)
    strict = np.zeros(rows, dtype=np.uint8)
    for row in range(rows):
        previous = int(old_state[row])
        broken = False
        inherited_run = 0
        stable_run = 0
        stable_state = -1
        for cycle in range(horizon):
            current = int(trajectory[row, cycle])
            similarity = 1.0 - int(hamming(previous, current, genes)) / genes
            if not broken and similarity <= 0.9:
                broken = True
                stable_state = current
                stable_run = 1
            elif broken:
                if similarity > 0.9:
                    inherited_run += 1
                else:
                    inherited_run = 0
                if current == stable_state:
                    stable_run += 1
                else:
                    stable_state = current
                    stable_run = 1
                if cycle < 12 and inherited_run >= 3:
                    f12[row] = 1
                if stable_run >= 8 and int(hamming(int(old_state[row]), current, genes)) / genes >= 0.2:
                    strict[row] = 1
            previous = current
    return f12, strict


def _simulate_cell(
    rulebook: Rulebook,
    protocol: dict[str, Any],
    condition: str,
    arm: str,
    history: int,
    midpoint: int,
    challenge: str,
    age: int,
    futures: int,
) -> np.ndarray:
    genes = int(protocol["genes"])
    flip_probability = float(protocol["expression_flip_probability"])
    target = int(rulebook.targets[history])
    if arm in ("self_continuation", "state_transplant"):
        initial_value = target
        stream_arm = "state_pair"
    elif arm == "reset":
        initial_value = int(rulebook.midpoints[midpoint])
        stream_arm = arm
    elif arm == "destination_donor":
        initial_value = int(rulebook.donors[midpoint, history])
        stream_arm = arm
    elif arm == "descriptor_null":
        initial_value = int(rulebook.nulls[midpoint, history])
        stream_arm = arm
    elif arm == "state_shuffle":
        initial_value = int(rulebook.shuffles[history])
        stream_arm = arm
    else:
        raise ValueError(arm)

    coordinate = (rulebook.uid, condition, stream_arm, history, midpoint, challenge, age)
    cell_rng = generator(str(protocol["master_seed_label"]), "future-cell", *coordinate)
    states = np.full(futures, initial_value, dtype=np.uint16)
    for _ in range(age):
        states ^= _flip_masks(cell_rng, futures, genes, flip_probability)
        states, _, _ = _develop(states, rulebook)
    hold_pre = (states == target).astype(np.uint8)
    old_state = states.copy()
    if challenge == "neutral_damage":
        states ^= _damage_masks(cell_rng, futures, genes, int(protocol["neutral_damage_bits"]))
    elif challenge == "forced_break":
        states = _forced_states(states, rulebook)

    horizon = int(protocol["horizon"])
    trajectory = np.zeros((futures, horizon), dtype=np.uint16)
    trajectory_classes = np.zeros((futures, horizon), dtype=np.int8)
    trajectory_kinds = np.zeros((futures, horizon), dtype=np.int8)
    for cycle in range(horizon):
        states ^= _flip_masks(cell_rng, futures, genes, flip_probability)
        states, kinds, points = _develop(states, rulebook)
        trajectory[:, cycle] = states
        trajectory_classes[:, cycle] = _destination_class(points, kinds, rulebook)
        trajectory_kinds[:, cycle] = kinds

    stable_run = int(protocol["stable_run"])
    destination = np.full(futures, -1, dtype=np.int8)
    current_class = np.full(futures, -1, dtype=np.int8)
    run_length = np.zeros(futures, dtype=np.uint8)
    for cycle in range(horizon):
        cls = trajectory_classes[:, cycle]
        is_point = trajectory_kinds[:, cycle] == POINT
        same = (cls == current_class) & is_point
        run_length = np.where(same, run_length + 1, np.where(is_point, 1, 0)).astype(np.uint8)
        current_class = np.where(is_point, cls, -1).astype(np.int8)
        newly = (destination < 0) & (run_length >= stable_run)
        destination[newly] = cls[newly]
    unresolved = destination < 0
    destination[unresolved] = trajectory_classes[unresolved, -1]
    f12, strict = _events(trajectory, old_state, genes)
    gen1_match = (trajectory_classes[:, 0] == history).astype(np.uint8)
    match = (destination == history).astype(np.uint8)
    rolling = np.full(futures, np.uint64(1469598103934665603), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for cycle in range(horizon):
            rolling = (rolling ^ trajectory[:, cycle].astype(np.uint64)) * np.uint64(1099511628211)

    rows = np.zeros(futures, dtype=PRIMARY_DTYPE)
    rows["condition"] = CONDITION_CODE[condition]
    rows["arm"] = ARM_CODE[arm]
    rows["history"] = history
    rows["midpoint"] = midpoint
    rows["challenge"] = CHALLENGE_CODE[challenge]
    rows["age"] = age
    rows["future"] = np.arange(futures, dtype=np.uint16)
    rows["half"] = (np.arange(futures) >= futures // 2).astype(np.uint8)
    rows["destination"] = destination
    rows["match"] = match
    rows["gen1_destination"] = trajectory_classes[:, 0]
    rows["gen1_match"] = gen1_match
    rows["hold_pre"] = hold_pre
    rows["f12_event"] = f12
    rows["strict_event"] = strict
    rows["trajectory_digest"] = rolling
    return rows


def simulate_primary_rulebook(rulebook: Rulebook, protocol: dict[str, Any]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for cell in protocol["conditions"]:
        for arm in cell["arms"]:
            for history in range(2):
                for midpoint in range(2):
                    for challenge in cell["challenges"]:
                        for age in cell["ages"]:
                            chunks.append(_simulate_cell(
                                rulebook, protocol, str(cell["name"]), str(arm), history,
                                midpoint, str(challenge), int(age), int(cell["futures"]),
                            ))
    return np.concatenate(chunks)


def _primary_worker(source_path: str, shard_path: str, protocol: dict[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    source = load_rulebook(Path(source_path))
    started = time.monotonic()
    rows = simulate_primary_rulebook(source, protocol)
    save_npz_atomic(Path(shard_path), rows=rows, proposal_index=np.asarray(source.proposal_index))
    return {"uid": source.uid, "rows": len(rows), "seconds": time.monotonic() - started}


def run_primary(
    run_dir: Path,
    protocol: dict[str, Any],
    workers: int = 12,
    soft_deadline: float | None = None,
) -> dict[str, Any]:
    ensure_registration(run_dir, protocol)
    sources = sample_sources(run_dir, protocol, deadline=soft_deadline)
    shard_dir = run_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (index, source, shard_dir / f"source_{index:04d}.npz")
        for index, source in enumerate(sources)
        if not (shard_dir / f"source_{index:04d}.npz").exists()
    ]
    completed = len(sources) - len(jobs)
    total_rows = sum(len(np.load(path, allow_pickle=False)["rows"]) for path in shard_dir.glob("source_*.npz"))
    timings: list[float] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending: dict[Any, tuple[int, Path]] = {}
        iterator = iter(jobs)
        while len(pending) < workers:
            try:
                index, source, shard = next(iterator)
            except StopIteration:
                break
            pending[executor.submit(_primary_worker, str(source), str(shard), protocol)] = (index, shard)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                result = future.result()
                completed += 1
                total_rows += int(result["rows"])
                timings.append(float(result["seconds"]))
                update_status(
                    run_dir, phase="simulation", completed_sources=completed,
                    total_sources=len(sources), rows=total_rows,
                    median_source_seconds=float(np.median(timings)) if timings else None,
                )
                if soft_deadline is None or time.monotonic() < soft_deadline:
                    try:
                        index, source, shard = next(iterator)
                    except StopIteration:
                        continue
                    pending[executor.submit(_primary_worker, str(source), str(shard), protocol)] = (index, shard)
    complete = completed == len(sources)
    summary = {
        "complete": complete,
        "sources": completed,
        "expected_sources": len(sources),
        "rows": total_rows,
        "expected_rows": expected_rows(protocol),
        "median_source_seconds": float(np.median(timings)) if timings else None,
        "registration_digest": digest(protocol),
    }
    write_json_atomic(run_dir / "simulation_summary.json", summary)
    update_status(run_dir, phase="simulation-complete" if complete else "checkpointed", **summary)
    return summary
