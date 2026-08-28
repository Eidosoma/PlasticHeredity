"""Stage-3 renewed-lineage test for the clean-room CA motif carrier."""

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

from .causal_heredity import _atomic_json, _atomic_text, _hash_seed, _sha256, _state_from_hex
from .e19 import require_pinned_numpy
from .life_family import live_2x2_counts_batch
from .lineage_field import load_round3_pairs
from .motif_generalization import DEVELOPMENT_PAIR_IDS, load_frozen_stage1
from .motif_lineage import (
    MotifContract,
    ReaderConfiguration,
    _bootstrap,
    _founders,
    _paired_uniforms,
    _score_checkpoint,
    _step,
    apply_energy_reader,
    motif3_codes,
    write_parent_carriers,
)


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "CA_MOTIF_LINEAGE_STAGE3_PROTOCOL.md"
DEFAULT_STAGE1_ROOT = ROOT / "results/ca-motif-lineage-stage-1"
DEFAULT_STAGE2_ROOT = ROOT / "results/ca-motif-lineage-stage-2"
RULE = 31649
CHECKPOINT_GENERATIONS = (1, 2, 4, 8, 16)
CONDITIONS = (
    "intact",
    "zero_every_boundary",
    "shuffle_every_boundary",
    "read_disabled",
    "founder_write_disabled",
    "no_rewrite",
    "ablate_after_g2",
    "rescue_same_enter_g4",
    "rescue_opposite_enter_g4",
    "opposite_founder",
    "carrier_corruption_1",
)


@dataclass(frozen=True)
class Stage3Contract:
    implementation_version: str = "ca-motif-lineage-stage3-cleanroom-v1"
    namespace: str = "plastic-ca-motif-lineage-stage3-v1"
    rule: int = RULE
    generation_sweeps: int = 64
    read_sweeps: int = 32
    write_start: int = 33
    observe_start: int = 57
    stale_retention: float = 0.50
    process_noise: float = 0.002
    carrier_corruption: float = 0.01
    primary_crossover: float = 0.15
    durable_crossover: float = 0.10
    control_advantage: float = 0.10
    survival_gate: float = 0.90
    loss_fraction: float = 0.70
    rescue_fraction: float = 0.70
    science_reserve_seconds: float = 1800.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "visible_reset": "bitwise-identical native board before every generation",
                "renewal_separation": "read sweeps 1-32; write from visible sweeps 33-64 only",
                "independent_unit": "new matched founder pair",
                "missing_policy": "dead and unresolved futures remain in denominators",
                "claim_boundary": "synthetic CA Plastic Heredity; not biological life or agency",
            }
        )
        return payload

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Stage3Profile:
    cohort_role: str
    pairs: int
    replicates: int
    generations: int
    bootstrap_resamples: int


STAGE3_PROFILES: dict[str, Stage3Profile] = {
    "smoke": Stage3Profile("development", 2, 2, 4, 100),
    "pilot": Stage3Profile("pilot", 16, 8, 8, 1_000),
    "reference": Stage3Profile("reference", 64, 64, 16, 10_000),
}
PUBLIC_PROFILES = tuple(STAGE3_PROFILES)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_stage2(
    stage2_root: Path = DEFAULT_STAGE2_ROOT,
    stage1_root: Path = DEFAULT_STAGE1_ROOT,
) -> dict[str, Any]:
    stage2_root = stage2_root.resolve()
    stage1_root = stage1_root.resolve()
    paths = {
        name: stage2_root / filename
        for name, filename in (
            ("decision", "STAGE_DECISION.json"),
            ("results", "RESULTS.json"),
            ("design", "DESIGN.json"),
            ("cohorts", "COHORTS.json"),
            ("manifest", "MANIFEST.json"),
            ("audit", "WRITER_AUDIT.json"),
        )
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen Stage-2 artifacts: {missing}")
    payload = {name: _load_json(path) for name, path in paths.items()}
    digest = str(payload["decision"]["design_digest"])
    for name in ("results", "design", "cohorts", "manifest", "audit"):
        if str(payload[name].get("design_digest")) != digest:
            raise ValueError(f"Stage-2 {name} design digest does not match the decision")
    if payload["results"].get("state") != "complete":
        raise ValueError("Stage 2 is not complete")
    if payload["decision"].get("decision") != "advance_to_stage_3_after_review":
        raise ValueError("Stage-2 decision does not permit Stage 3")
    selected = payload["decision"].get("selected_stage3_input")
    if not selected:
        raise ValueError("Stage 2 did not freeze a Stage-3 input")
    configuration = ReaderConfiguration(
        family=str(selected["family"]),
        write_window=int(selected["write_window"]),
        strength=float(selected["strength"]),
        read_duration=int(selected["read_duration"]),
    )
    if configuration.id != selected.get("configuration_id"):
        raise ValueError("Stage-2 configuration ID does not match its parameters")
    if configuration.family != "motif_energy512":
        raise ValueError("Stage-3 v1 requires the frozen motif-energy carrier")
    stage1 = load_frozen_stage1(stage1_root)
    if stage1["design_digest"] != payload["decision"].get("stage1_design_digest"):
        raise ValueError("Stage-1 ancestry does not match Stage 2")
    used_ids = set(stage1["used_pair_ids"])
    used_ids.update(payload["cohorts"]["stage2_pair_ids"])
    used_ids.update(DEVELOPMENT_PAIR_IDS)
    return {
        **payload,
        "root": stage2_root,
        "paths": paths,
        "design_digest": digest,
        "configuration": configuration,
        "stage1": stage1,
        "reference": stage1["reference"],
        "used_pair_ids": used_ids,
    }


def select_stage3_pairs(
    profile: Stage3Profile,
    frozen: dict[str, Any],
    contract: Stage3Contract,
) -> list[dict[str, Any]]:
    unused = [
        pair for pair in load_round3_pairs()[contract.rule]
        if pair["pair_id"] not in frozen["used_pair_ids"]
    ]
    ordered = sorted(
        unused,
        key=lambda pair: (
            hashlib.sha256(f"{contract.namespace}:cohort:{pair['pair_id']}".encode()).hexdigest(),
            pair["pair_id"],
        ),
    )
    if profile.cohort_role == "development":
        selected = ordered[: profile.pairs]
    elif profile.cohort_role == "reference":
        selected = ordered[2 : 2 + profile.pairs]
    else:
        selected = ordered[66 : 66 + profile.pairs]
    if len(selected) != profile.pairs:
        raise ValueError("not enough untouched pairs for Stage 3")
    if any(pair["pair_id"] in frozen["used_pair_ids"] for pair in selected):
        raise AssertionError("Stage-3 cohort overlaps an earlier stage")
    return selected


def motif_counts_batch(codes: np.ndarray) -> np.ndarray:
    if codes.ndim != 3:
        raise ValueError("motif codes must have shape (sample, height, width)")
    sample = len(codes)
    offsets = np.arange(sample, dtype=np.int64)[:, None] * 512
    encoded = codes.reshape(sample, -1).astype(np.int64) + offsets
    return np.bincount(encoded.ravel(), minlength=sample * 512).reshape(sample, 512).astype(np.float64)


def write_energy_from_counts(
    counts: np.ndarray,
    reference_probability: np.ndarray,
    writer_contract: MotifContract,
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    reference_probability = np.asarray(reference_probability, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != 512:
        raise ValueError("motif counts must have shape (sample, 512)")
    if reference_probability.shape != (512,):
        raise ValueError("reference motif probability must have length 512")
    alpha = writer_contract.jeffreys_alpha
    probability = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + 512.0 * alpha)
    marks = np.log(probability) - np.log(reference_probability[None, :])
    return np.clip(marks, -writer_contract.energy_clip, writer_contract.energy_clip).astype(np.float32)


def _repeat_histories(values: np.ndarray, replicates: int) -> np.ndarray:
    return np.concatenate(
        (np.repeat(values[0:1], replicates, axis=0), np.repeat(values[1:2], replicates, axis=0)),
        axis=0,
    )


def _swap_histories(values: np.ndarray, replicates: int) -> np.ndarray:
    return np.concatenate((values[replicates:], values[:replicates]), axis=0)


def _lineage_uniforms(
    pair_id: str, purpose: str, generation: int, sweep: int, replicates: int
) -> np.ndarray:
    return _paired_uniforms(
        pair_id,
        f"stage3-{purpose}-generation-{generation}",
        sweep,
        replicates,
    )


def carrier_diagnostic(carrier: np.ndarray, replicates: int) -> dict[str, float]:
    shaped = carrier.reshape(2, replicates, 512)
    mean_a = shaped[0].mean(axis=0)
    mean_b = shaped[1].mean(axis=0)
    return {
        "mean_abs": float(np.mean(np.abs(carrier))),
        "centroid_l2": float(np.linalg.norm(mean_a - mean_b)),
        "paired_l2_mean": float(np.mean(np.linalg.norm(shaped[0] - shaped[1], axis=1))),
    }


def _apply_entry_intervention(
    carrier: np.ndarray,
    condition: str,
    generation: int,
    pair_id: str,
    replicates: int,
    contract: Stage3Contract,
    source_exits: Sequence[np.ndarray] | None,
) -> np.ndarray:
    result = carrier.copy()
    if condition == "zero_every_boundary":
        result.fill(0.0)
    elif condition == "shuffle_every_boundary":
        permutation = np.random.default_rng(
            _hash_seed(contract.namespace, pair_id, "shuffle", generation)
        ).permutation(512)
        result = result[:, permutation]
    elif condition == "opposite_founder" and generation == 1:
        result = _swap_histories(result, replicates)
    elif condition in ("ablate_after_g2", "rescue_same_enter_g4", "rescue_opposite_enter_g4") and generation == 3:
        result.fill(0.0)
    elif condition in ("rescue_same_enter_g4", "rescue_opposite_enter_g4") and generation == 4:
        if source_exits is None or len(source_exits) < 3:
            raise ValueError("rescue requires a contemporaneous intact sister carrier")
        result = source_exits[2].copy()
        if condition == "rescue_opposite_enter_g4":
            result = _swap_histories(result, replicates)
    elif condition == "carrier_corruption_1":
        mask = np.random.default_rng(
            _hash_seed(contract.namespace, pair_id, "carrier-corruption", generation)
        ).random((replicates, 512)) < contract.carrier_corruption
        result[np.concatenate((mask, mask), axis=0)] *= -1.0
    return result


def simulate_lineage(
    pair: dict[str, Any],
    configuration: ReaderConfiguration,
    condition: str,
    replicates: int,
    generations: int,
    reference: dict[int, dict[str, np.ndarray]],
    writer_contract: MotifContract,
    contract: Stage3Contract,
    *,
    source_exits: Sequence[np.ndarray] | None = None,
    retain_exits: bool = False,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    pair_id = str(pair["pair_id"])
    reset_a = _state_from_hex("life", pair["donor_a"]["initial_state_hex"])
    reset_b = _state_from_hex("life", pair["donor_b"]["initial_state_hex"])
    if not np.array_equal(reset_a, reset_b):
        raise AssertionError(f"visible reset mismatch in pair {pair_id}")
    reset = np.repeat(reset_a[None, ...], 2 * replicates, axis=0)
    founder_written = write_parent_carriers(
        _founders(pair), (configuration.write_window,), reference, writer_contract
    )[configuration.write_window]
    carrier = _repeat_histories(founder_written[configuration.family], replicates)
    if condition == "founder_write_disabled":
        carrier.fill(0.0)
    founder_terminal = founder_written["terminal"]
    alive = np.ones(2 * replicates, dtype=np.bool_)
    checkpoints = set(value for value in CHECKPOINT_GENERATIONS if value <= generations)
    outcomes: dict[str, Any] = {}
    carrier_history: dict[str, Any] = {}
    exits: list[np.ndarray] = []
    reference_probability = reference[configuration.write_window]["motif_probability"]

    for generation in range(1, generations + 1):
        carrier = _apply_entry_intervention(
            carrier,
            condition,
            generation,
            pair_id,
            replicates,
            contract,
            source_exits,
        )
        entry_diagnostic = carrier_diagnostic(carrier, replicates)
        state = reset.copy()
        state[~alive] = False
        if not np.array_equal(state[alive], reset[alive]):
            raise AssertionError("visible reset was not bitwise identical for living branches")
        recent: deque[np.ndarray] = deque(maxlen=writer_contract.observation_window)
        write_counts = np.zeros((2 * replicates, 512), dtype=np.float64)
        for sweep in range(1, contract.generation_sweeps + 1):
            predicted = _step(state, contract.rule)
            if condition != "read_disabled" and sweep <= configuration.read_duration:
                predicted = apply_energy_reader(
                    predicted,
                    carrier,
                    _lineage_uniforms(pair_id, "read", generation, sweep, replicates),
                    configuration.strength,
                )
            predicted ^= (
                _lineage_uniforms(pair_id, "process", generation, sweep, replicates)
                < contract.process_noise
            )
            predicted[~alive] = False
            state = predicted
            if sweep >= contract.write_start:
                write_counts += motif_counts_batch(motif3_codes(state))
            if sweep >= contract.observe_start:
                recent.append(live_2x2_counts_batch(state))
        alive &= state.any(axis=(1, 2))
        if generation in checkpoints:
            outcomes[str(generation)] = _score_checkpoint(
                state,
                np.sum(np.stack(tuple(recent)), axis=0),
                pair,
                founder_terminal,
                replicates,
                writer_contract,
                diagnostics=True,
            )
        if condition == "no_rewrite":
            carrier = carrier * contract.stale_retention
        else:
            carrier = write_energy_from_counts(
                write_counts, reference_probability, writer_contract
            )
        carrier[~alive] = 0.0
        exit_diagnostic = carrier_diagnostic(carrier, replicates)
        if generation in checkpoints:
            carrier_history[str(generation)] = {
                "entry": entry_diagnostic,
                "exit": exit_diagnostic,
                "surviving_futures": int(np.count_nonzero(alive)),
            }
        if retain_exits:
            exits.append(carrier.copy())
    return (
        {
            "condition": condition,
            "reset_sha256": hashlib.sha256(reset_a.tobytes()).hexdigest(),
            "reset_asserted_before_every_generation": True,
            "founder_carrier": carrier_diagnostic(
                _repeat_histories(founder_written[configuration.family], replicates), replicates
            ),
            "outcomes": outcomes,
            "carrier_history": carrier_history,
        },
        exits,
    )


def _pair_task(
    payload: tuple[
        dict[str, Any],
        MotifContract,
        Stage3Contract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, writer_contract, contract, reference = payload
    configuration = ReaderConfiguration(**item["configuration"])
    pair = item["pair"]
    intact, exits = simulate_lineage(
        pair,
        configuration,
        "intact",
        int(item["replicates"]),
        int(item["generations"]),
        reference,
        writer_contract,
        contract,
        retain_exits=True,
    )
    conditions: dict[str, Any] = {"intact": intact}
    for condition in item["conditions"]:
        if condition == "intact":
            continue
        result, _ = simulate_lineage(
            pair,
            configuration,
            condition,
            int(item["replicates"]),
            int(item["generations"]),
            reference,
            writer_contract,
            contract,
            source_exits=exits,
        )
        conditions[condition] = result
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": pair["pair_id"],
        "replicates": int(item["replicates"]),
        "generations": int(item["generations"]),
        "configuration": configuration.to_dict(),
        "conditions": conditions,
    }


def _run_checkpoints(
    output: Path,
    items: Sequence[dict[str, Any]],
    writer_contract: MotifContract,
    contract: Stage3Contract,
    reference: dict[int, dict[str, np.ndarray]],
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status: Any,
) -> tuple[list[dict[str, Any]], bool]:
    root = output / "lineages/checkpoints"
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

    def save(item: dict[str, Any], result: dict[str, Any]) -> None:
        key = item["checkpoint"]
        _atomic_json(
            root / f"{key}.json",
            {"design_digest": design_digest, "stage": 3, "checkpoint": key, "result": result},
        )
        results[key] = result
        elapsed = max(time.monotonic() - phase_started, 1e-6)
        completed_new = max(1, len(results) - initial)
        eta = elapsed / completed_new * max(0, len(items) - len(results))
        status(
            "running",
            "lineages",
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
            pending[pool.submit(_pair_task, (item, writer_contract, contract, reference))] = item
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
        output / "lineages/stage_summary.json",
        {
            "stage": 3,
            "design_digest": design_digest,
            "complete": complete,
            "completed": len(results),
            "total": len(items),
            "budget_truncated": truncated or not complete,
        },
    )
    if complete:
        _atomic_text(output / "lineages/COMPLETE", "complete\n")
    return [results[key] for key in sorted(results)], complete


def _metric(
    rows: Sequence[dict[str, Any]], condition: str, generation: int, observer: str, metric: str
) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            outcome = row["conditions"][condition]["outcomes"][str(generation)]
        except KeyError:
            continue
        if metric == "survival":
            values.append(float(outcome["survival"]))
        elif observer in outcome and metric in outcome[observer]:
            values.append(float(outcome[observer][metric]))
    return values


def _difference(
    rows: Sequence[dict[str, Any]], left: str, right: str, generation: int,
    observer: str = "primary",
) -> list[float]:
    result: list[float] = []
    for row in rows:
        try:
            a = row["conditions"][left]["outcomes"][str(generation)][observer]["crossover"]
            b = row["conditions"][right]["outcomes"][str(generation)][observer]["crossover"]
        except KeyError:
            continue
        result.append(float(a) - float(b))
    return result


def _carrier_values(
    rows: Sequence[dict[str, Any]], condition: str, generation: int, boundary: str, metric: str
) -> list[float]:
    result: list[float] = []
    for row in rows:
        try:
            value = row["conditions"][condition]["carrier_history"][str(generation)][boundary][metric]
        except KeyError:
            continue
        result.append(float(value))
    return result


def adjudicate(
    rows: Sequence[dict[str, Any]],
    profile: Stage3Profile,
    contract: Stage3Contract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"verdict": "INCOMPLETE", "renewed_gate": False}
    if profile.generations < 16:
        return {"verdict": "NOT_ADJUDICATED_PROFILE", "renewed_gate": False}
    alpha = 0.025

    def boot(values: Sequence[float], name: str) -> dict[str, Any]:
        return _bootstrap(
            values,
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "gate", name),
            alpha,
        )

    def positive(summary: dict[str, Any]) -> bool:
        return summary["ci"][0] is not None and float(summary["ci"][0]) > 0.0

    intact4 = boot(_metric(rows, "intact", 4, "primary", "crossover"), "intact4")
    intact8 = boot(_metric(rows, "intact", 8, "primary", "crossover"), "intact8")
    intact16 = boot(_metric(rows, "intact", 16, "primary", "crossover"), "intact16")
    terminal8 = boot(_metric(rows, "intact", 8, "terminal", "crossover"), "terminal8")
    survival8 = boot(_metric(rows, "intact", 8, "primary", "survival"), "survival8")
    survival16 = boot(_metric(rows, "intact", 16, "primary", "survival"), "survival16")
    controls = {
        condition: boot(_difference(rows, "intact", condition, 8), f"control-{condition}")
        for condition in (
            "zero_every_boundary",
            "shuffle_every_boundary",
            "read_disabled",
            "founder_write_disabled",
        )
    }
    no_rewrite8 = boot(_metric(rows, "no_rewrite", 8, "primary", "crossover"), "no-rewrite8")
    ablation4 = boot(_metric(rows, "ablate_after_g2", 4, "primary", "crossover"), "ablation4")
    rescue4 = boot(_metric(rows, "rescue_same_enter_g4", 4, "primary", "crossover"), "rescue4")
    rescue_advantage = boot(
        _difference(rows, "rescue_same_enter_g4", "ablate_after_g2", 4), "rescue-advantage"
    )
    opposite_rescue4 = boot(
        _metric(rows, "rescue_opposite_enter_g4", 4, "primary", "crossover"),
        "opposite-rescue4",
    )
    opposite_founder8 = boot(
        _metric(rows, "opposite_founder", 8, "primary", "crossover"), "opposite-founder8"
    )
    corruption8 = boot(
        _metric(rows, "carrier_corruption_1", 8, "primary", "crossover"), "corruption8"
    )
    direction_a = float(np.mean(_metric(rows, "intact", 8, "primary", "direction_a")))
    direction_b = float(np.mean(_metric(rows, "intact", 8, "primary", "direction_b")))
    pair_values = _metric(rows, "intact", 8, "primary", "crossover")
    fraction_positive = float(np.mean(np.asarray(pair_values) > 0.0)) if pair_values else 0.0
    intact4_mean = float(intact4["mean"] or 0.0)
    intact8_mean = float(intact8["mean"] or 0.0)
    no_rewrite_loss = 1.0 - float(no_rewrite8["mean"] or 0.0) / intact8_mean if intact8_mean > 0 else None
    ablation_loss = 1.0 - float(ablation4["mean"] or 0.0) / intact4_mean if intact4_mean > 0 else None
    rescue_fraction = float(rescue4["mean"] or 0.0) / intact4_mean if intact4_mean > 0 else None
    renewed = bool(
        intact8_mean >= contract.primary_crossover
        and positive(intact8)
        and float(intact16["mean"] or 0.0) >= contract.durable_crossover
        and positive(intact16)
        and direction_a > 0.0
        and direction_b > 0.0
        and fraction_positive >= 0.50
        and float(survival8["mean"] or 0.0) >= contract.survival_gate
        and float(survival16["mean"] or 0.0) >= contract.survival_gate
        and float(terminal8["mean"] or 0.0) >= contract.durable_crossover
        and positive(terminal8)
        and all(
            float(value["mean"] or 0.0) >= contract.control_advantage and positive(value)
            for value in controls.values()
        )
        and no_rewrite_loss is not None
        and no_rewrite_loss >= contract.loss_fraction
        and ablation_loss is not None
        and ablation_loss >= contract.loss_fraction
        and rescue_fraction is not None
        and rescue_fraction >= contract.rescue_fraction
        and float(rescue_advantage["mean"] or 0.0) >= contract.control_advantage
        and positive(rescue_advantage)
        and float(opposite_rescue4["mean"] or 0.0) <= -contract.durable_crossover
        and opposite_rescue4["ci"][1] is not None
        and float(opposite_rescue4["ci"][1]) < 0.0
        and float(opposite_founder8["mean"] or 0.0) <= -contract.durable_crossover
        and opposite_founder8["ci"][1] is not None
        and float(opposite_founder8["ci"][1]) < 0.0
        and float(corruption8["mean"] or 0.0) >= contract.durable_crossover
        and positive(corruption8)
    )
    static = bool(
        not renewed
        and intact8_mean >= contract.primary_crossover
        and positive(intact8)
        and no_rewrite_loss is not None
        and no_rewrite_loss < contract.loss_fraction
    )
    transient = bool(
        not renewed
        and not static
        and intact8_mean >= contract.primary_crossover
        and positive(intact8)
        and (float(intact16["mean"] or 0.0) < contract.durable_crossover or not positive(intact16))
    )
    verdict = (
        "RENEWED_CA_PLASTIC_HEREDITY"
        if renewed
        else "STATIC_HIDDEN_TEMPLATE"
        if static
        else "TRANSIENT_LINEAGE_MEMORY"
        if transient
        else "NO_RENEWED_CA_PLASTIC_HEREDITY"
    )
    carrier_renewal = {
        str(generation): {
            "intact_exit_centroid_l2": boot(
                _carrier_values(rows, "intact", generation, "exit", "centroid_l2"),
                f"carrier-intact-{generation}",
            ),
            "no_rewrite_entry_mean_abs": boot(
                _carrier_values(rows, "no_rewrite", generation, "entry", "mean_abs"),
                f"carrier-no-rewrite-{generation}",
            ),
        }
        for generation in CHECKPOINT_GENERATIONS
    }
    return {
        "verdict": verdict,
        "renewed_gate": renewed,
        "static_template_gate": static,
        "transient_memory_gate": transient,
        "claim_boundary": "synthetic CA Plastic Heredity; no metabolism, agency, or biological-life claim",
        "interval_alpha": alpha,
        "intact_generation4": intact4,
        "intact_generation8": intact8,
        "intact_generation16": intact16,
        "terminal_generation8": terminal8,
        "survival_generation8": survival8,
        "survival_generation16": survival16,
        "direction_a_mean": direction_a,
        "direction_b_mean": direction_b,
        "fraction_pairs_positive": fraction_positive,
        "control_advantages_generation8": controls,
        "no_rewrite_generation8": no_rewrite8,
        "no_rewrite_loss_fraction": no_rewrite_loss,
        "ablation_generation4": ablation4,
        "ablation_loss_fraction": ablation_loss,
        "rescue_generation4": rescue4,
        "rescue_restoration_fraction": rescue_fraction,
        "rescue_advantage_generation4": rescue_advantage,
        "opposite_rescue_generation4": opposite_rescue4,
        "opposite_founder_generation8": opposite_founder8,
        "carrier_corruption_generation8": corruption8,
        "carrier_renewal": carrier_renewal,
    }


def _queue(design_digest: str, state: str, verdict: str | None = None) -> dict[str, Any]:
    stages = [
        {"stage": 1, "name": "motif_carrier_upper_bound", "state": "complete"},
        {"stage": 2, "name": "freeze_and_generalize_reader", "state": "complete"},
        {"stage": 3, "name": "renewed_heredity_causal_ladder", "state": state},
        {"stage": 4, "name": "compression_and_robustness", "state": "blocked_pending_stage3_review"},
        {"stage": 5, "name": "localize_inheritance", "state": "blocked_pending_stage4_review"},
    ]
    if verdict:
        stages[2]["verdict"] = verdict
    return {
        "programme": "ca_motif_lineage_five_stage",
        "stage3_design_digest": design_digest,
        "automatic_chaining": False,
        "per_stage_max_hours": 8.0,
        "stages": stages,
    }


def _render_report(results: dict[str, Any]) -> str:
    value = results["adjudication"]
    lines = [
        "# CA motif-lineage Stage 3",
        "",
        f"State: **{results['state']}**. Profile: `{results['profile']}`.",
        f"Verdict: **{value['verdict']}**.",
        f"Elapsed: `{results['elapsed_seconds'] / 3600.0:.3f}` wall hours.",
        "",
        "## Decisive lineage evidence",
        "",
    ]
    if "intact_generation8" in value:
        lines.extend(
            (
                f"- Intact generation 8: `{value['intact_generation8']['mean']}`, CI `{value['intact_generation8']['ci']}`.",
                f"- Intact generation 16: `{value['intact_generation16']['mean']}`, CI `{value['intact_generation16']['ci']}`.",
                f"- No-rewrite loss: `{value['no_rewrite_loss_fraction']}`.",
                f"- Ablation loss: `{value['ablation_loss_fraction']}`.",
                f"- Same-history rescue restoration: `{value['rescue_restoration_fraction']}`.",
                f"- Opposite rescue generation 4: `{value['opposite_rescue_generation4']['mean']}`.",
                f"- Opposite founder generation 8: `{value['opposite_founder_generation8']['mean']}`.",
            )
        )
    lines.extend(
        (
            "",
            "## Evidence boundary",
            "",
            "The experiment tests a synthetic, explicitly represented CA carrier across complete visible resets. "
            "Even a positive verdict is not evidence for metabolism, agency, biological life, or memory outside the automaton's physical state.",
        )
    )
    return "\n".join(lines) + "\n"


def _render_lay(results: dict[str, Any]) -> str:
    verdict = results["adjudication"]["verdict"]
    if verdict == "RENEWED_CA_PLASTIC_HEREDITY":
        finding = "The texture memory renewed itself across 16 generations and passed the complete ablation-and-rescue test."
    elif verdict == "STATIC_HIDDEN_TEMPLATE":
        finding = "The original texture memory persisted, but daughters did not need to renew it; this is a hidden template rather than Plastic Heredity."
    elif verdict == "TRANSIENT_LINEAGE_MEMORY":
        finding = "The texture memory crossed several generations but faded before generation 16."
    elif verdict in ("INCOMPLETE", "NOT_ADJUDICATED_PROFILE"):
        finding = "This profile does not issue the full 16-generation scientific verdict."
    else:
        finding = "The motif carrier did not pass the registered self-renewing heredity test."
    return (
        "# Lay summary\n\n"
        f"{finding}\n\n"
        "Every daughter began from the same erased visible board. It used the inherited texture recipe for the first half of its life, "
        "then the recipe was turned off. During the second half, the daughter had to write a new recipe from its own recovered body and "
        "pass that to the next generation.\n\n"
        "The controls ask whether this is truly a renewed lineage memory: stop rewriting, erase it, restore the correct contemporary recipe, "
        "or restore the opposite recipe. Only the complete causal pattern can receive a Plastic Heredity verdict.\n"
    )


def _update_discovery_log(results: dict[str, Any]) -> None:
    path = ROOT / "DISCOVERY_LOG_EIDOSOMA_SCIENTIST.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Discovery log\n"
    start = "<!-- ca-motif-lineage-stage-3:start -->"
    end = "<!-- ca-motif-lineage-stage-3:end -->"
    section = "\n".join(
        (
            start,
            "## CA motif-lineage Stage 3",
            "",
            f"Renewed-lineage verdict: `{results['adjudication']['verdict']}`.",
            f"Profile: `{results['profile']}`; elapsed `{results['elapsed_seconds'] / 3600.0:.3f}` wall hours.",
            "See `results/ca-motif-lineage-stage-3/REPORT.md` and `LAY_SUMMARY.md`.",
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


def run_motif_lineage_stage3(
    output: Path,
    *,
    stage2_root: Path = DEFAULT_STAGE2_ROOT,
    stage1_root: Path = DEFAULT_STAGE1_ROOT,
    profile_name: str = "reference",
    workers: int = 20,
    max_hours: float = 8.0,
    resume: bool = False,
) -> dict[str, Any]:
    require_pinned_numpy()
    if profile_name not in PUBLIC_PROFILES:
        raise ValueError(f"unknown Stage-3 profile {profile_name!r}")
    if max_hours <= 0.0 or max_hours > 8.0:
        raise ValueError("Stage-3 max-hours must be in (0, 8]")
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hard_deadline = started + max_hours * 3600.0
    contract = Stage3Contract()
    writer_contract = MotifContract()
    profile = STAGE3_PROFILES[profile_name]
    reserve = min(contract.science_reserve_seconds, max(60.0, max_hours * 3600.0 * 0.10))
    science_deadline = max(started, hard_deadline - reserve)

    def status(state: str, phase: str, **extra: Any) -> None:
        now = time.time()
        payload = {
            "state": state,
            "stage": "3-renewed-lineage",
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
        status("running", "freeze_and_cohort")
        frozen = load_frozen_stage2(stage2_root, stage1_root)
        configuration = frozen["configuration"]
        pairs = select_stage3_pairs(profile, frozen, contract)
        all_unused = select_stage3_pairs(STAGE3_PROFILES["smoke"], frozen, contract)
        development_ids = [pair["pair_id"] for pair in all_unused]
        input_paths = [PROTOCOL_PATH, *frozen["paths"].values(), frozen["stage1"]["paths"]["calibration"]]
        design_payload = {
            "experiment": "ca_motif_lineage_stage_3",
            "contract": contract.to_dict(),
            "writer_contract_digest": writer_contract.digest,
            "profile_name": profile_name,
            "profile": asdict(profile),
            "configuration": configuration.to_dict(),
            "stage2_design_digest": frozen["design_digest"],
            "stage1_design_digest": frozen["stage1"]["design_digest"],
            "stage2_review_authorization": "user explicitly requested Stage 3 after reviewing Stage 2",
            "prior_pair_ids_excluded": len(frozen["used_pair_ids"]),
            "development_pair_ids": development_ids,
            "pair_ids": [pair["pair_id"] for pair in pairs],
            "conditions": CONDITIONS,
            "input_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in input_paths},
            "implementation_sha256": {
                "motif_lineage_stage3.py": _sha256(Path(__file__)),
                "motif_lineage.py": _sha256(Path(__file__).with_name("motif_lineage.py")),
            },
            "cleanroom_exclusion": "no Wagner or Fable implementation source is read, imported, hashed, or executed",
            "retuning": False,
        }
        design_digest = hashlib.sha256(
            json.dumps(design_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        design = {**design_payload, "design_digest": design_digest}
        design_path = output / "DESIGN.json"
        if resume and design_path.exists() and _load_json(design_path).get("design_digest") != design_digest:
            raise ValueError("resume design digest mismatch")
        _atomic_json(design_path, design)
        _atomic_json(
            output / "MANIFEST.json",
            {
                "experiment": "ca_motif_lineage_stage_3",
                "stage": 3,
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
        _atomic_json(
            output / "COHORTS.json",
            {
                "design_digest": design_digest,
                "prior_pair_ids_excluded": sorted(frozen["used_pair_ids"]),
                "development_pair_ids": development_ids,
                "stage3_pair_ids": [pair["pair_id"] for pair in pairs],
            },
        )
        _atomic_json(output / "QUEUE.json", _queue(design_digest, "running"))
        configuration_payload = {
            key: value for key, value in configuration.to_dict().items()
            if key != "configuration_id"
        }
        items = [
            {
                "checkpoint": f"lineage-{index:04d}",
                "pair": pair,
                "replicates": profile.replicates,
                "generations": profile.generations,
                "configuration": configuration_payload,
                "conditions": CONDITIONS,
            }
            for index, pair in enumerate(pairs)
        ]
        status("running", "lineages", completed=0, total=len(items))
        rows, complete = _run_checkpoints(
            output,
            items,
            writer_contract,
            contract,
            frozen["reference"],
            design_digest,
            workers=workers,
            resume=resume,
            deadline=science_deadline,
            status=status,
        )
        status("running", "adjudication")
        adjudication = adjudicate(rows, profile, contract, complete)
        state = "complete" if complete else "partial_budget_exhausted"
        results = {
            "experiment": "ca_motif_lineage_stage_3",
            "state": state,
            "profile": profile_name,
            "design_digest": design_digest,
            "stage2_design_digest": frozen["design_digest"],
            "configuration": configuration.to_dict(),
            "started_unix": started,
            "completed_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "adjudication": adjudication,
        }
        _atomic_json(output / "RESULTS.json", results)
        _atomic_text(output / "REPORT.md", _render_report(results))
        _atomic_text(output / "LAY_SUMMARY.md", _render_lay(results))
        passed = bool(adjudication.get("renewed_gate"))
        decision = {
            "stage": 3,
            "design_digest": design_digest,
            "stage2_design_digest": frozen["design_digest"],
            "verdict": adjudication["verdict"],
            "review_required": True,
            "automatic_launch": False,
            "decision": "advance_to_stage_4_after_review" if passed else "halt_and_replan_renewal_mechanism",
            "selected_stage4_input": configuration.to_dict() if passed else None,
            "claim_boundary": "synthetic CA Plastic Heredity only; no biological-life or agency claim",
        }
        _atomic_json(output / "STAGE_DECISION.json", decision)
        queue = _queue(
            design_digest,
            "complete" if complete else "partial_resumable",
            adjudication["verdict"],
        )
        queue["stages"][3]["state"] = (
            "blocked_pending_human_review" if passed else "blocked_stage3_gate_failed"
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
    "CHECKPOINT_GENERATIONS",
    "CONDITIONS",
    "PUBLIC_PROFILES",
    "STAGE3_PROFILES",
    "Stage3Contract",
    "adjudicate",
    "carrier_diagnostic",
    "load_frozen_stage2",
    "motif_counts_batch",
    "run_motif_lineage_stage3",
    "select_stage3_pairs",
    "simulate_lineage",
    "write_energy_from_counts",
]
