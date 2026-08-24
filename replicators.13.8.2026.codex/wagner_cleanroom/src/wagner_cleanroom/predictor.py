from __future__ import annotations

import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

from .dynamics import POINT, Rulebook, decode_state, hamming, sample_rulebook
from .experiment import _develop, _flip_masks
from .protocol import digest, ensure_registration, write_json_atomic
from .rng import generator
from .storage import load_rulebook, save_npz_atomic, save_rulebook, update_status


def _predictor_source_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "sources").glob("source_*.npz"))


def sample_predictor_sources(
    run_dir: Path,
    protocol: dict[str, Any],
    cohort: str,
    deadline: float | None = None,
) -> list[Path]:
    if cohort not in ("development", "evaluation"):
        raise ValueError(cohort)
    wanted = int(protocol[f"{cohort}_sources"])
    label = str(protocol[f"{cohort}_seed_label"])
    run_dir.mkdir(parents=True, exist_ok=True)
    source_dir = run_dir / "sources"
    source_dir.mkdir(exist_ok=True)
    existing = _predictor_source_paths(run_dir)
    next_proposal = 0 if not existing else max(load_rulebook(p).proposal_index for p in existing) + 1
    accepted = len(existing)
    while accepted < wanted and next_proposal < int(protocol["maximum_source_proposals"]):
        if deadline is not None and time.monotonic() >= deadline:
            break
        source = sample_rulebook(
            label, next_proposal, genes=int(protocol["genes"]), max_sweeps=100,
            minimum_basin=float(protocol["minimum_basin_fraction"]),
            minimum_distance=float(protocol["minimum_form_distance"]),
        )
        if source is not None:
            save_rulebook(source_dir / f"source_{accepted:04d}.npz", source)
            accepted += 1
            update_status(run_dir, phase="sampling", cohort=cohort, accepted_sources=accepted, last_proposal=next_proposal)
        next_proposal += 1
    if accepted < wanted:
        raise RuntimeError(f"{cohort} source sampling incomplete: {accepted}/{wanted}")
    write_json_atomic(run_dir / "source_index.json", {
        "cohort": cohort,
        "sources": [
            {"source_index": i, "file": p.name, "proposal_index": load_rulebook(p).proposal_index}
            for i, p in enumerate(_predictor_source_paths(run_dir)[:wanted])
        ],
    })
    return _predictor_source_paths(run_dir)[:wanted]


def _start_states(rulebook: Rulebook, count: int, genes: int) -> np.ndarray:
    all_states = np.arange(1 << genes, dtype=np.uint16)
    da = hamming(all_states, int(rulebook.targets[0]), genes).astype(np.int16)
    db = hamming(all_states, int(rulebook.targets[1]), genes).astype(np.int16)
    margin = np.abs(da - db)
    ordering = np.lexsort((all_states, rulebook.landscape.transient, margin))
    positions = np.linspace(0, len(ordering) - 1, count).round().astype(int)
    return all_states[ordering[positions]]


def _history_features(history: np.ndarray, genes: int) -> np.ndarray:
    similarities = 1.0 - hamming(history[:-1], history[1:], genes).astype(float) / genes
    lagged = similarities[-5:]
    if len(lagged) < 5:
        lagged = np.pad(lagged, (5 - len(lagged), 0), constant_values=1.0)
    identical = similarities > 0.9
    run = 0
    for value in identical[::-1]:
        if not value:
            break
        run += 1
    since_break = len(similarities)
    for index, value in enumerate(identical[::-1]):
        if not value:
            since_break = index
            break
    return np.concatenate([
        lagged,
        np.asarray([run, since_break, int((~identical).sum()), float(similarities.mean())]),
    ]).astype(np.float64)


def _structural_features(rulebook: Rulebook, state: int, genes: int) -> np.ndarray:
    landscape = rulebook.landscape
    point = int(landscape.point_index[state])
    basin_fraction = float(landscape.basin_sizes[point]) / (1 << genes) if point >= 0 else 0.0
    probabilities = landscape.basin_sizes.astype(float) / max(1, int(landscape.basin_sizes.sum()))
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log(positive)).sum() / math.log(max(2, len(probabilities))))
    nearest = min(int(hamming(state, int(target), genes)) for target in rulebook.targets) / genes
    base_destination = int(landscape.point_index[state])
    changed = 0
    for gene in range(genes):
        if int(landscape.point_index[state ^ (1 << gene)]) != base_destination:
            changed += 1
    expression = decode_state(state, genes).astype(float)
    fields = rulebook.weights @ expression
    margin = float(np.min(np.abs(fields)))
    return np.asarray([
        basin_fraction, entropy, float(landscape.transient[state]) / 100.0,
        nearest, changed / genes, margin,
    ], dtype=np.float64)


def _full_features(rulebook: Rulebook, state: int, genes: int) -> np.ndarray:
    expression = decode_state(state, genes).astype(np.float64)
    weights = rulebook.weights
    fields = weights @ expression
    interactions = weights * expression[None, :]
    singular = np.linalg.svd(weights, compute_uv=False)
    symmetric = 0.5 * (weights + weights.T)
    asymmetric = 0.5 * (weights - weights.T)
    return np.concatenate([
        expression, weights.ravel(), fields, interactions.ravel(), singular,
        np.asarray([np.linalg.norm(symmetric), np.linalg.norm(asymmetric)]),
    ]).astype(np.float64)


def _future_events(
    rulebook: Rulebook,
    current: int,
    futures: int,
    horizon: int,
    flip_probability: float,
    rng: np.random.Generator,
    genes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.full(futures, current, dtype=np.uint16)
    trajectory = np.zeros((futures, horizon), dtype=np.uint16)
    point = np.zeros((futures, horizon), dtype=bool)
    for cycle in range(horizon):
        states ^= _flip_masks(rng, futures, genes, flip_probability)
        states, kinds, _ = _develop(states, rulebook)
        trajectory[:, cycle] = states
        point[:, cycle] = kinds == POINT
    return _score_future_trajectories(trajectory, point, current, genes)


def _score_future_trajectories(
    trajectory: np.ndarray,
    point: np.ndarray,
    current: int,
    genes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    futures, horizon = trajectory.shape
    f12 = np.zeros(futures, dtype=np.uint8)
    strict = np.zeros(futures, dtype=np.uint8)
    sensitivity = np.zeros((futures, 4), dtype=np.uint8)
    for row in range(futures):
        old = current
        previous = old
        broken = False
        inherited = 0
        sensitivity_broken = [False, False]
        sensitivity_inherited = [0, 0]
        stable_state = -1
        stable_run = 0
        for cycle in range(horizon):
            adult = int(trajectory[row, cycle])
            similarity = 1.0 - int(hamming(previous, adult, genes)) / genes
            for threshold_index, threshold in enumerate((0.8, 0.9)):
                if not sensitivity_broken[threshold_index] and similarity <= threshold:
                    sensitivity_broken[threshold_index] = True
                elif sensitivity_broken[threshold_index]:
                    sensitivity_inherited[threshold_index] = (
                        sensitivity_inherited[threshold_index] + 1
                        if similarity > threshold else 0
                    )
                    if cycle < 12 and sensitivity_inherited[threshold_index] >= 3:
                        sensitivity[row, threshold_index] = 1
            if not broken and similarity <= 0.9:
                broken = True
                stable_state = adult
                stable_run = 1
            elif broken:
                inherited = inherited + 1 if similarity > 0.9 else 0
                if adult == stable_state and point[row, cycle]:
                    stable_run += 1
                else:
                    stable_state = adult
                    stable_run = 1 if point[row, cycle] else 0
                if cycle < 12 and inherited >= 3:
                    f12[row] = 1
                distance = int(hamming(old, adult, genes)) / genes
                if stable_run >= 8 and distance >= 0.2:
                    strict[row] = 1
                if stable_run >= 8 and distance >= 0.2:
                    sensitivity[row, 2] = 1
                if stable_run >= 8 and distance >= 0.3:
                    sensitivity[row, 3] = 1
            previous = adult
    return f12, strict, sensitivity


def simulate_predictor_rulebook(
    rulebook: Rulebook,
    protocol: dict[str, Any],
    cohort: str,
    source_index: int,
) -> dict[str, np.ndarray]:
    genes = int(protocol["genes"])
    history_count = int(protocol["histories_per_source"])
    starts = _start_states(rulebook, history_count, genes)
    futures = int(protocol["futures_per_state"])
    horizon = int(protocol["horizon"])
    flip_probability = float(protocol["expression_flip_probability"])
    x_history: list[np.ndarray] = []
    x_structural: list[np.ndarray] = []
    x_full: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    current_states: list[int] = []
    f12_counts: list[int] = []
    strict_counts: list[int] = []
    half_counts: list[np.ndarray] = []
    sensitivity_counts: list[np.ndarray] = []
    label = str(protocol[f"{cohort}_seed_label"])
    prelaunch = int(protocol["prelaunch_boundaries"])
    for history_index, start in enumerate(starts):
        history_rng = generator(label, rulebook.uid, "history", history_index)
        states = np.asarray([start], dtype=np.uint16)
        adult_history = [int(start)]
        for _ in range(prelaunch):
            states ^= _flip_masks(history_rng, 1, genes, flip_probability)
            states, _, _ = _develop(states, rulebook)
            adult_history.append(int(states[0]))
        history_array = np.asarray(adult_history, dtype=np.uint16)
        current = int(history_array[-1])
        future_rng = generator(label, rulebook.uid, "future-cell", history_index)
        f12, strict, sensitivity = _future_events(
            rulebook, current, futures, horizon, flip_probability, future_rng, genes,
        )
        x_history.append(_history_features(history_array, genes))
        x_structural.append(_structural_features(rulebook, current, genes))
        x_full.append(_full_features(rulebook, current, genes))
        histories.append(history_array)
        current_states.append(current)
        f12_counts.append(int(f12.sum()))
        strict_counts.append(int(strict.sum()))
        half_counts.append(np.asarray([f12[: futures // 2].sum(), f12[futures // 2 :].sum()], dtype=np.uint16))
        sensitivity_counts.append(sensitivity.sum(axis=0).astype(np.uint16))
    return {
        "source_index": np.full(history_count, source_index, dtype=np.uint16),
        "history_index": np.arange(history_count, dtype=np.uint8),
        "starts": starts,
        "current_states": np.asarray(current_states, dtype=np.uint16),
        "histories": np.stack(histories),
        "x_history": np.stack(x_history),
        "x_structural": np.stack(x_structural),
        "x_full": np.stack(x_full),
        "f12_counts": np.asarray(f12_counts, dtype=np.uint16),
        "strict_counts": np.asarray(strict_counts, dtype=np.uint16),
        "half_counts": np.stack(half_counts),
        "sensitivity_counts": np.stack(sensitivity_counts),
        "futures": np.full(history_count, futures, dtype=np.uint16),
    }


def _predictor_worker(
    source_path: str,
    shard_path: str,
    protocol: dict[str, Any],
    cohort: str,
    source_index: int,
) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    source = load_rulebook(Path(source_path))
    started = time.monotonic()
    arrays = simulate_predictor_rulebook(source, protocol, cohort, source_index)
    save_npz_atomic(Path(shard_path), **arrays)
    return {"source_index": source_index, "seconds": time.monotonic() - started}


def _merge_predictor_shards(run_dir: Path) -> Path:
    shards = sorted((run_dir / "shards").glob("source_*.npz"))
    keys = (
        "source_index", "history_index", "starts", "current_states", "histories",
        "x_history", "x_structural", "x_full", "f12_counts", "strict_counts",
        "half_counts", "sensitivity_counts", "futures",
    )
    merged: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for path in shards:
        with np.load(path, allow_pickle=False) as data:
            for key in keys:
                merged[key].append(data[key].copy())
    output = run_dir / "predictor_data.npz"
    save_npz_atomic(output, **{key: np.concatenate(values, axis=0) for key, values in merged.items()})
    return output


def run_predictor_cohort(
    run_dir: Path,
    protocol: dict[str, Any],
    cohort: str,
    workers: int = 12,
    soft_deadline: float | None = None,
) -> dict[str, Any]:
    cohort_protocol = dict(protocol)
    cohort_protocol["active_cohort"] = cohort
    ensure_registration(run_dir, cohort_protocol)
    sources = sample_predictor_sources(run_dir, protocol, cohort, deadline=soft_deadline)
    shard_dir = run_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    jobs = [
        (index, source, shard_dir / f"source_{index:04d}.npz")
        for index, source in enumerate(sources)
        if not (shard_dir / f"source_{index:04d}.npz").exists()
    ]
    timings: list[float] = []
    already_complete = len(sources) - len(jobs)
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending: dict[Any, int] = {}
        iterator = iter(jobs)
        while len(pending) < workers:
            try:
                index, source, shard = next(iterator)
            except StopIteration:
                break
            if soft_deadline is not None and time.monotonic() >= soft_deadline:
                break
            pending[executor.submit(_predictor_worker, str(source), str(shard), protocol, cohort, index)] = index
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                result = future.result()
                completed += 1
                timings.append(float(result["seconds"]))
                update_status(
                    run_dir, phase="simulation", cohort=cohort,
                    completed_sources=already_complete + completed,
                    total_sources=len(sources),
                )
                if soft_deadline is None or time.monotonic() < soft_deadline:
                    try:
                        index, source, shard = next(iterator)
                    except StopIteration:
                        continue
                    pending[executor.submit(_predictor_worker, str(source), str(shard), protocol, cohort, index)] = index
    shard_count = len(list(shard_dir.glob("source_*.npz")))
    complete = shard_count == len(sources)
    if complete:
        _merge_predictor_shards(run_dir)
    summary = {
        "complete": complete,
        "cohort": cohort,
        "sources": shard_count,
        "expected_sources": len(sources),
        "states": shard_count * int(protocol["histories_per_source"]),
        "futures": shard_count * int(protocol["histories_per_source"]) * int(protocol["futures_per_state"]),
        "median_source_seconds": float(np.median(timings)) if timings else None,
        "registration_digest": digest(cohort_protocol),
    }
    write_json_atomic(run_dir / "simulation_summary.json", summary)
    update_status(run_dir, phase="simulation-complete" if complete else "checkpointed", **summary)
    return summary
