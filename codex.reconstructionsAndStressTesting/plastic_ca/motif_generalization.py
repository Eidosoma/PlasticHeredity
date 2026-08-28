"""Stage-2 frozen-reader generalization for the CA motif carrier programme."""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from functools import lru_cache
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
from .motif_lineage import (
    MotifContract,
    ReaderConfiguration,
    _bootstrap,
    _founders,
    _paired_uniforms,
    _reference_from_json,
    _score_checkpoint,
    _step,
    apply_energy_reader,
    collect_trajectory_counts,
    motif3_codes,
    write_parent_carriers,
)


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "CA_MOTIF_LINEAGE_STAGE2_PROTOCOL.md"
DEFAULT_STAGE1_ROOT = ROOT / "results/ca-motif-lineage-stage-1"
RULE = 31649
PRIMARY_ENVIRONMENTS = (
    "native",
    "launch0",
    "launch1",
    "launch2",
    "launch3",
    "native_translate_3_5",
    "native_rot90",
    "native_reflect_x",
)
STRESS_ENVIRONMENTS = ("random_density_10", "random_density_30", "random_density_50")
CORE_CONDITIONS = (
    "intact",
    "zero",
    "read_disabled",
    "shuffle",
    "matched_random",
    "opposite_history",
    "unrelated_pair",
    "midpoint",
)
STRESS_CONDITIONS = ("intact", "zero", "opposite_history", "unrelated_pair")
ROBUSTNESS_CONDITIONS = ("process_noise", "carrier_corruption_1")
DOSE_CONTRASTS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEVELOPMENT_PAIR_IDS = (
    "narrow-0468-life-31649-2-1381-life-31649-2-1497",
    "narrow-0759-life-31649-3-528-life-31649-3-91",
)


@dataclass(frozen=True)
class GeneralizationContract:
    implementation_version: str = "ca-motif-generalization-cleanroom-v1"
    namespace: str = "plastic-ca-motif-lineage-stage2-v1"
    rule: int = RULE
    horizon: int = 64
    checkpoints: tuple[int, ...] = (8, 16, 32, 64)
    gate_checkpoint: int = 64
    primary_crossover: float = 0.15
    stress_crossover: float = 0.10
    control_advantage: float = 0.10
    survival_gate: float = 0.90
    unrelated_retention: float = 0.70
    midpoint_tolerance: float = 0.02
    monotonic_tolerance: float = 0.03
    writer_accuracy_gate: float = 0.80
    symmetry_tolerance: float = 1e-6
    science_reserve_seconds: float = 1800.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "reader_policy": "exact Stage-1 winner; no Stage-2 tuning",
                "pair_policy": "all Stage-1 pair IDs excluded",
                "independent_unit": "new matched founder pair",
                "claim_boundary": "general reusable form channel; not multigenerational heredity",
            }
        )
        return payload

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GeneralizationProfile:
    cohort_role: str
    pairs: int
    replicates: int
    bootstrap_resamples: int
    audit_pairs: int
    primary_environments: tuple[str, ...]
    stress_environments: tuple[str, ...]
    core_conditions: tuple[str, ...]
    stress_conditions: tuple[str, ...]
    dose_contrasts: tuple[float, ...]


GENERALIZATION_PROFILES: dict[str, GeneralizationProfile] = {
    "smoke": GeneralizationProfile(
        "development",
        2,
        2,
        100,
        2,
        ("native", "native_translate_3_5", "native_rot90"),
        ("random_density_30",),
        CORE_CONDITIONS,
        STRESS_CONDITIONS,
        DOSE_CONTRASTS,
    ),
    "pilot": GeneralizationProfile(
        "pilot",
        16,
        8,
        1_000,
        16,
        ("native", "launch0", "launch1", "launch2", "launch3", "native_rot90"),
        ("random_density_30",),
        CORE_CONDITIONS,
        STRESS_CONDITIONS,
        DOSE_CONTRASTS,
    ),
    "reference": GeneralizationProfile(
        "reference",
        96,
        64,
        10_000,
        32,
        PRIMARY_ENVIRONMENTS,
        STRESS_ENVIRONMENTS,
        CORE_CONDITIONS,
        STRESS_CONDITIONS,
        DOSE_CONTRASTS,
    ),
}
PUBLIC_PROFILES = tuple(GENERALIZATION_PROFILES)


def load_frozen_stage1(stage1_root: Path = DEFAULT_STAGE1_ROOT) -> dict[str, Any]:
    paths = {
        name: stage1_root / filename
        for name, filename in (
            ("decision", "STAGE_DECISION.json"),
            ("results", "RESULTS.json"),
            ("design", "DESIGN.json"),
            ("calibration", "CALIBRATION.json"),
            ("cohorts", "COHORTS.json"),
            ("manifest", "MANIFEST.json"),
        )
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen Stage-1 artifacts: {missing}")
    payload = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    digest = str(payload["decision"]["design_digest"])
    for name in ("results", "design", "calibration", "cohorts", "manifest"):
        if str(payload[name].get("design_digest")) != digest:
            raise ValueError(f"Stage-1 {name} design digest does not match the decision")
    if payload["results"].get("state") != "complete":
        raise ValueError("Stage 1 is not complete")
    if payload["decision"].get("decision") != "advance_to_stage_2_after_review":
        raise ValueError("Stage-1 decision does not permit Stage-2 planning")
    selected = payload["decision"].get("selected_stage2_input")
    if not selected:
        raise ValueError("Stage 1 did not freeze a Stage-2 reader")
    configuration = ReaderConfiguration(
        family=str(selected["family"]),
        write_window=int(selected["write_window"]),
        strength=float(selected["strength"]),
        read_duration=int(selected["read_duration"]),
    )
    if configuration.family != "motif_energy512":
        raise ValueError("Stage-2 v1 requires the retained motif-energy winner")
    if configuration.id != selected.get("configuration_id"):
        raise ValueError("Stage-1 configuration ID does not match its parameters")
    used_ids = {
        pair_id
        for cohort in payload["cohorts"]["cohorts"].values()
        for pair_id in cohort
    }
    return {
        **payload,
        "root": stage1_root,
        "design_digest": digest,
        "configuration": configuration,
        "reference": _reference_from_json(payload["calibration"]["reference"]),
        "used_pair_ids": used_ids,
        "paths": paths,
    }


def select_stage2_pairs(
    profile: GeneralizationProfile,
    frozen: dict[str, Any],
    contract: GeneralizationContract,
) -> list[dict[str, Any]]:
    unused = [
        pair for pair in load_round3_pairs()[contract.rule]
        if pair["pair_id"] not in frozen["used_pair_ids"]
    ]
    ordered_all = sorted(
        unused,
        key=lambda pair: (
            hashlib.sha256(f"{contract.namespace}:cohort:{pair['pair_id']}".encode()).hexdigest(),
            pair["pair_id"],
        ),
    )
    by_id = {pair["pair_id"]: pair for pair in ordered_all}
    if profile.cohort_role == "development":
        missing = [pair_id for pair_id in DEVELOPMENT_PAIR_IDS if pair_id not in by_id]
        if missing:
            raise ValueError(f"missing registered development-only pairs: {missing}")
        selected = [by_id[pair_id] for pair_id in DEVELOPMENT_PAIR_IDS]
    else:
        ordered = [pair for pair in ordered_all if pair["pair_id"] not in DEVELOPMENT_PAIR_IDS]
        offset = 96 if profile.cohort_role == "pilot" else 0
        selected = ordered[offset : offset + profile.pairs]
    if len(selected) < profile.pairs:
        raise ValueError("not enough unused Rule-31649 pairs for Stage 2")
    if any(pair["pair_id"] in frozen["used_pair_ids"] for pair in selected):
        raise AssertionError("Stage-2 cohort overlaps Stage 1")
    return selected


def launch_reset_bank() -> dict[int, np.ndarray]:
    bank: dict[int, np.ndarray] = {}
    hexes: dict[int, str] = {}
    for pair in load_round3_pairs()[RULE]:
        launch = int(pair["launch_index"])
        for donor in (pair["donor_a"], pair["donor_b"]):
            value = str(donor["initial_state_hex"])
            if launch in hexes and hexes[launch] != value:
                raise ValueError(f"launch {launch} has multiple reset rows")
            hexes[launch] = value
    if set(hexes) != {0, 1, 2, 3}:
        raise ValueError("the four registered launch resets are unavailable")
    for launch, value in hexes.items():
        bank[launch] = _state_from_hex("life", value)
    return bank


def _forward_board(values: np.ndarray, environment: str) -> np.ndarray:
    if environment == "native_translate_3_5":
        return np.roll(values, shift=(3, 5), axis=(-2, -1))
    if environment == "native_rot90":
        return np.rot90(values, k=1, axes=(-2, -1))
    if environment == "native_reflect_x":
        return np.flip(values, axis=-1)
    return values.copy()


def _inverse_board(values: np.ndarray, environment: str) -> np.ndarray:
    if environment == "native_translate_3_5":
        return np.roll(values, shift=(-3, -5), axis=(-2, -1))
    if environment == "native_rot90":
        return np.rot90(values, k=-1, axes=(-2, -1))
    if environment == "native_reflect_x":
        return np.flip(values, axis=-1)
    return values.copy()


def _decode_motif(code: int) -> np.ndarray:
    board = np.zeros((3, 3), dtype=np.bool_)
    for bit in range(9):
        board.flat[bit] = bool(code & (1 << bit))
    return board


def _encode_motif(board: np.ndarray) -> int:
    return sum(int(bool(board.flat[bit])) << bit for bit in range(9))


@lru_cache(maxsize=None)
def motif_code_permutation(environment: str) -> np.ndarray:
    if environment not in ("native_rot90", "native_reflect_x"):
        return np.arange(512, dtype=np.int64)
    permutation = np.zeros(512, dtype=np.int64)
    for code in range(512):
        board = _decode_motif(code)
        transformed = (
            np.rot90(board, k=1) if environment == "native_rot90" else np.fliplr(board)
        )
        permutation[code] = _encode_motif(transformed)
    if len(np.unique(permutation)) != 512:
        raise AssertionError("motif symmetry mapping is not a permutation")
    return permutation


def transform_energy_carrier(carrier: np.ndarray, environment: str) -> np.ndarray:
    result = np.asarray(carrier).copy()
    permutation = motif_code_permutation(environment)
    result[..., permutation] = carrier
    return result


def environment_reset(
    pair: dict[str, Any], environment: str, bank: dict[int, np.ndarray], contract: GeneralizationContract
) -> np.ndarray:
    native_a = _state_from_hex("life", pair["donor_a"]["initial_state_hex"])
    native_b = _state_from_hex("life", pair["donor_b"]["initial_state_hex"])
    if not np.array_equal(native_a, native_b):
        raise AssertionError(f"visible reset mismatch in pair {pair['pair_id']}")
    if environment == "native" or environment.startswith("native_"):
        return _forward_board(native_a, environment)
    if environment.startswith("launch"):
        return bank[int(environment.removeprefix("launch"))].copy()
    if environment.startswith("random_density_"):
        percentage = int(environment.removeprefix("random_density_"))
        count = int(round(256 * percentage / 100.0))
        indices = np.random.default_rng(
            _hash_seed(contract.namespace, pair["pair_id"], environment, "reset")
        ).choice(256, size=count, replace=False)
        board = np.zeros(256, dtype=np.bool_)
        board[indices] = True
        return board.reshape(16, 16)
    raise ValueError(f"unknown Stage-2 environment {environment!r}")


def mix_history_carriers(carrier: np.ndarray, contrast: float) -> np.ndarray:
    if carrier.shape != (2, 512):
        raise ValueError("energy carrier must have shape (2, 512)")
    if not 0.0 <= contrast <= 1.0:
        raise ValueError("carrier contrast must be in [0, 1]")
    if contrast == 1.0:
        return np.asarray(carrier, dtype=np.float32).copy()
    midpoint = 0.5 * (carrier[0] + carrier[1])
    if contrast == 0.0:
        return np.repeat(midpoint[None, :], 2, axis=0).astype(np.float32)
    half_difference = 0.5 * (carrier[0] - carrier[1])
    return np.stack(
        (midpoint + contrast * half_difference, midpoint - contrast * half_difference)
    ).astype(np.float32)


def _repeat_histories(carrier: np.ndarray, replicates: int) -> np.ndarray:
    return np.concatenate(
        (np.repeat(carrier[0:1], replicates, axis=0), np.repeat(carrier[1:2], replicates, axis=0)),
        axis=0,
    )


def _environment_uniforms(
    pair_id: str, purpose: str, sweep: int, replicates: int, environment: str
) -> np.ndarray:
    base = _paired_uniforms(pair_id, f"stage2-{purpose}", sweep, replicates)
    return _forward_board(base, environment)


def simulate_generalization_condition(
    pair: dict[str, Any],
    configuration: ReaderConfiguration,
    parent_carrier: np.ndarray,
    founder_terminal: np.ndarray,
    source_carrier: np.ndarray,
    reset_bank: dict[int, np.ndarray],
    environment: str,
    condition: str,
    replicates: int,
    writer_contract: MotifContract,
    contract: GeneralizationContract,
    *,
    carrier_override: np.ndarray | None = None,
) -> dict[str, Any]:
    reset = environment_reset(pair, environment, reset_bank, contract)
    state = np.repeat(reset[None, ...], 2 * replicates, axis=0)
    carrier_pair = np.asarray(carrier_override if carrier_override is not None else parent_carrier).copy()
    if condition in ("zero", "read_disabled"):
        carrier_pair.fill(0.0)
    elif condition == "shuffle":
        permutation = np.random.default_rng(
            _hash_seed(contract.namespace, pair["pair_id"], environment, "shuffle")
        ).permutation(512)
        carrier_pair = carrier_pair[:, permutation]
    elif condition == "matched_random":
        rng = np.random.default_rng(
            _hash_seed(contract.namespace, pair["pair_id"], environment, "matched-random")
        )
        carrier_pair = np.stack(
            (carrier_pair[0, rng.permutation(512)], carrier_pair[1, rng.permutation(512)])
        )
    elif condition == "opposite_history":
        carrier_pair = carrier_pair[::-1].copy()
    elif condition == "unrelated_pair":
        carrier_pair = source_carrier.copy()
    elif condition == "midpoint":
        midpoint = carrier_pair.mean(axis=0, keepdims=True)
        carrier_pair = np.repeat(midpoint, 2, axis=0)

    carrier_pair = transform_energy_carrier(carrier_pair, environment)
    carrier = _repeat_histories(carrier_pair, replicates)
    if condition == "carrier_corruption_1":
        mask = np.random.default_rng(
            _hash_seed(contract.namespace, pair["pair_id"], environment, "corruption")
        ).random((replicates, 512)) < writer_contract.carrier_corruption
        carrier[np.concatenate((mask, mask), axis=0)] *= -1.0
    read_enabled = condition not in ("zero", "read_disabled")
    process_noise = writer_contract.process_noise if condition == "process_noise" else 0.0
    recent: deque[np.ndarray] = deque(maxlen=writer_contract.observation_window)
    outcomes: dict[str, Any] = {}
    for sweep in range(1, contract.horizon + 1):
        predicted = _step(state, contract.rule)
        if read_enabled and sweep <= configuration.read_duration:
            predicted = apply_energy_reader(
                predicted,
                carrier,
                _environment_uniforms(str(pair["pair_id"]), "read", sweep, replicates, environment),
                configuration.strength,
            )
        if process_noise:
            predicted ^= (
                _environment_uniforms(str(pair["pair_id"]), "process", sweep, replicates, environment)
                < process_noise
            )
        state = predicted
        observed = _inverse_board(state, environment)
        recent.append(live_2x2_counts_batch(observed))
        if sweep in contract.checkpoints:
            outcomes[str(sweep)] = _score_checkpoint(
                observed,
                np.sum(np.stack(tuple(recent)), axis=0),
                pair,
                founder_terminal,
                replicates,
                writer_contract,
                diagnostics=True,
            )
    return {
        "environment": environment,
        "condition": condition,
        "reset_sha256": hashlib.sha256(reset.tobytes()).hexdigest(),
        "reset_identical_between_histories": True,
        "outcomes": outcomes,
    }


def _pair_task(
    payload: tuple[
        dict[str, Any],
        MotifContract,
        GeneralizationContract,
        dict[int, dict[str, np.ndarray]],
        dict[int, np.ndarray],
    ]
) -> dict[str, Any]:
    item, writer_contract, contract, reference, reset_bank = payload
    pair = item["pair"]
    source = item["source_pair"]
    configuration = ReaderConfiguration(**item["configuration"])
    written = write_parent_carriers(
        _founders(pair), (configuration.write_window,), reference, writer_contract
    )[configuration.write_window]
    source_written = write_parent_carriers(
        _founders(source), (configuration.write_window,), reference, writer_contract
    )[configuration.write_window]
    carrier = written[configuration.family]
    source_carrier = source_written[configuration.family]
    environments: dict[str, Any] = {}
    for environment in item["primary_environments"]:
        conditions = {
            condition: simulate_generalization_condition(
                pair,
                configuration,
                carrier,
                written["terminal"],
                source_carrier,
                reset_bank,
                environment,
                condition,
                int(item["replicates"]),
                writer_contract,
                contract,
            )
            for condition in item["core_conditions"]
        }
        if environment == "native":
            for condition in ROBUSTNESS_CONDITIONS:
                conditions[condition] = simulate_generalization_condition(
                    pair,
                    configuration,
                    carrier,
                    written["terminal"],
                    source_carrier,
                    reset_bank,
                    environment,
                    condition,
                    int(item["replicates"]),
                    writer_contract,
                    contract,
                )
        environments[environment] = conditions
    for environment in item["stress_environments"]:
        environments[environment] = {
            condition: simulate_generalization_condition(
                pair,
                configuration,
                carrier,
                written["terminal"],
                source_carrier,
                reset_bank,
                environment,
                condition,
                int(item["replicates"]),
                writer_contract,
                contract,
            )
            for condition in item["stress_conditions"]
        }
    dose = {
        f"{contrast:.2f}": simulate_generalization_condition(
            pair,
            configuration,
            carrier,
            written["terminal"],
            source_carrier,
            reset_bank,
            "native",
            f"dose_{contrast:.2f}",
            int(item["replicates"]),
            writer_contract,
            contract,
            carrier_override=mix_history_carriers(carrier, float(contrast)),
        )
        for contrast in item["dose_contrasts"]
    }
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": pair["pair_id"],
        "source_pair_id": source["pair_id"],
        "replicates": int(item["replicates"]),
        "configuration": configuration.to_dict(),
        "carrier_l2": float(np.linalg.norm(carrier[0] - carrier[1])),
        "environments": environments,
        "dose": dose,
    }


def writer_audit(
    pairs: Sequence[dict[str, Any]],
    configuration: ReaderConfiguration,
    reference: dict[int, dict[str, np.ndarray]],
    writer_contract: MotifContract,
    contract: GeneralizationContract,
) -> dict[str, Any]:
    carriers: list[np.ndarray] = []
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    reflection_errors: list[float] = []
    for pair in pairs:
        founders = _founders(pair)
        raw_original = collect_trajectory_counts(
            founders, (configuration.write_window,), rule=writer_contract.rule
        )[configuration.write_window]["motif"]
        raw_original /= raw_original.sum(axis=1, keepdims=True)
        original = write_parent_carriers(
            founders, (configuration.write_window,), reference, writer_contract
        )[configuration.write_window][configuration.family]
        carriers.append(original)
        raw_translated = collect_trajectory_counts(
            np.roll(founders, shift=(3, 5), axis=(1, 2)),
            (configuration.write_window,),
            rule=writer_contract.rule,
        )[configuration.write_window]["motif"]
        raw_rotated = collect_trajectory_counts(
            np.rot90(founders, k=1, axes=(1, 2)),
            (configuration.write_window,),
            rule=writer_contract.rule,
        )[configuration.write_window]["motif"]
        raw_reflected = collect_trajectory_counts(
            np.flip(founders, axis=2),
            (configuration.write_window,),
            rule=writer_contract.rule,
        )[configuration.write_window]["motif"]
        raw_translated /= raw_translated.sum(axis=1, keepdims=True)
        raw_rotated /= raw_rotated.sum(axis=1, keepdims=True)
        raw_reflected /= raw_reflected.sum(axis=1, keepdims=True)
        translation_errors.append(float(np.max(np.abs(raw_original - raw_translated))))
        rotation_errors.append(float(np.max(np.abs(
            transform_energy_carrier(raw_original, "native_rot90") - raw_rotated
        ))))
        reflection_errors.append(float(np.max(np.abs(
            transform_energy_carrier(raw_original, "native_reflect_x") - raw_reflected
        ))))
    stack = np.stack(carriers)
    correct = 0
    total = 0
    for held_out in range(len(stack)):
        training = np.delete(stack, held_out, axis=0)
        centroid_a = training[:, 0].mean(axis=0)
        centroid_b = training[:, 1].mean(axis=0)
        for label, vector in enumerate(stack[held_out]):
            distance_a = float(np.linalg.norm(vector - centroid_a))
            distance_b = float(np.linalg.norm(vector - centroid_b))
            prediction = 0 if distance_a < distance_b else 1
            correct += int(prediction == label)
            total += 1
    accuracy = float(correct / total) if total else 0.0
    max_translation = max(translation_errors, default=float("inf"))
    max_rotation = max(rotation_errors, default=float("inf"))
    max_reflection = max(reflection_errors, default=float("inf"))
    return {
        "pair_count": len(pairs),
        "leave_one_pair_out_accuracy": accuracy,
        "mean_a_b_l2": float(np.mean([np.linalg.norm(value[0] - value[1]) for value in stack])),
        "raw_motif_symmetry_max_abs_error": {
            "translation": max_translation,
            "rotation90": max_rotation,
            "reflection_x": max_reflection,
        },
        "writer_gate": bool(
            accuracy >= contract.writer_accuracy_gate
            and max_translation <= contract.symmetry_tolerance
            and max_rotation <= contract.symmetry_tolerance
            and max_reflection <= contract.symmetry_tolerance
        ),
        "label_use": "adjudication-only LOPO diagnostic; never parameter selection",
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_checkpoints(
    output: Path,
    items: Sequence[dict[str, Any]],
    writer_contract: MotifContract,
    contract: GeneralizationContract,
    reference: dict[int, dict[str, np.ndarray]],
    reset_bank: dict[int, np.ndarray],
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status: Any,
) -> tuple[list[dict[str, Any]], bool]:
    root = output / "generalization/checkpoints"
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
            {"design_digest": design_digest, "stage": 2, "checkpoint": key, "result": result},
        )
        results[key] = result
        elapsed = max(time.monotonic() - phase_started, 1e-6)
        completed_new = max(1, len(results) - initial)
        eta = elapsed / completed_new * max(0, len(items) - len(results))
        status(
            "running",
            "generalization",
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
            pending[
                pool.submit(
                    _pair_task, (item, writer_contract, contract, reference, reset_bank)
                )
            ] = item
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
        output / "generalization/stage_summary.json",
        {
            "stage": 2,
            "design_digest": design_digest,
            "complete": complete,
            "completed": len(results),
            "total": len(items),
            "budget_truncated": truncated or not complete,
        },
    )
    if complete:
        _atomic_text(output / "generalization/COMPLETE", "complete\n")
    return [results[key] for key in sorted(results)], complete


def _values(
    rows: Sequence[dict[str, Any]], environment: str, condition: str, observer: str, metric: str,
    checkpoint: int = 64,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            outcome = row["environments"][environment][condition]["outcomes"][str(checkpoint)]
        except KeyError:
            continue
        if metric == "survival":
            values.append(float(outcome["survival"]))
        elif observer in outcome and metric in outcome[observer]:
            values.append(float(outcome[observer][metric]))
    return values


def _difference(
    rows: Sequence[dict[str, Any]], environment: str, left: str, right: str,
    observer: str = "primary", checkpoint: int = 64,
) -> list[float]:
    result: list[float] = []
    for row in rows:
        try:
            conditions = row["environments"][environment]
            a = conditions[left]["outcomes"][str(checkpoint)][observer]["crossover"]
            b = conditions[right]["outcomes"][str(checkpoint)][observer]["crossover"]
        except KeyError:
            continue
        result.append(float(a) - float(b))
    return result


def _dose_values(rows: Sequence[dict[str, Any]], contrast: float, checkpoint: int = 64) -> list[float]:
    key = f"{contrast:.2f}"
    result: list[float] = []
    for row in rows:
        try:
            value = row["dose"][key]["outcomes"][str(checkpoint)]["primary"]["crossover"]
        except KeyError:
            continue
        result.append(float(value))
    return result


def adjudicate(
    rows: Sequence[dict[str, Any]],
    audit: dict[str, Any],
    profile: GeneralizationProfile,
    contract: GeneralizationContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"verdict": "INCOMPLETE", "generalization_gate": False}
    alpha = 0.05 / max(1, len(profile.primary_environments))

    def boot(values: Sequence[float], name: str) -> dict[str, Any]:
        return _bootstrap(
            values,
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "gate", name),
            alpha,
        )

    def positive(summary: dict[str, Any]) -> bool:
        return summary["ci"][0] is not None and float(summary["ci"][0]) > 0.0

    environment_results: dict[str, Any] = {}
    primary_passes: list[str] = []
    for environment in profile.primary_environments:
        intact = boot(_values(rows, environment, "intact", "primary", "crossover"), f"{environment}-intact")
        survival = boot(_values(rows, environment, "intact", "primary", "survival"), f"{environment}-survival")
        terminal = boot(_values(rows, environment, "intact", "terminal", "crossover"), f"{environment}-terminal")
        direction_a = float(np.mean(_values(rows, environment, "intact", "primary", "direction_a")))
        direction_b = float(np.mean(_values(rows, environment, "intact", "primary", "direction_b")))
        controls = {
            name: boot(_difference(rows, environment, "intact", name), f"{environment}-adv-{name}")
            for name in ("zero", "read_disabled", "shuffle", "matched_random")
        }
        opposite = boot(
            _values(rows, environment, "opposite_history", "primary", "crossover"),
            f"{environment}-opposite",
        )
        unrelated = boot(
            _values(rows, environment, "unrelated_pair", "primary", "crossover"),
            f"{environment}-unrelated",
        )
        midpoint = boot(
            _values(rows, environment, "midpoint", "primary", "crossover"),
            f"{environment}-midpoint",
        )
        passed = bool(
            float(intact["mean"] or 0.0) >= contract.primary_crossover
            and positive(intact)
            and direction_a > 0.0
            and direction_b > 0.0
            and float(survival["mean"] or 0.0) >= contract.survival_gate
            and float(terminal["mean"] or 0.0) >= contract.stress_crossover
            and positive(terminal)
            and all(
                float(value["mean"] or 0.0) >= contract.control_advantage and positive(value)
                for value in controls.values()
            )
            and float(opposite["mean"] or 0.0) <= -contract.stress_crossover
            and opposite["ci"][1] is not None
            and float(opposite["ci"][1]) < 0.0
            and float(unrelated["mean"] or 0.0) >= contract.stress_crossover
            and positive(unrelated)
            and float(unrelated["mean"] or 0.0)
            >= contract.unrelated_retention * float(intact["mean"] or 0.0)
            and abs(float(midpoint["mean"] or 0.0)) <= contract.midpoint_tolerance
        )
        if passed:
            primary_passes.append(environment)
        environment_results[environment] = {
            "passed": passed,
            "intact": intact,
            "survival": survival,
            "terminal": terminal,
            "direction_a_mean": direction_a,
            "direction_b_mean": direction_b,
            "control_advantages": controls,
            "opposite_history": opposite,
            "unrelated_pair": unrelated,
            "unrelated_retention_fraction": (
                float(unrelated["mean"] or 0.0) / float(intact["mean"])
                if intact["mean"] not in (None, 0.0) else None
            ),
            "midpoint": midpoint,
        }

    stress_results: dict[str, Any] = {}
    stress_passes: list[str] = []
    for environment in profile.stress_environments:
        intact = boot(_values(rows, environment, "intact", "primary", "crossover"), f"{environment}-intact")
        zero_advantage = boot(_difference(rows, environment, "intact", "zero"), f"{environment}-zero")
        opposite = boot(_values(rows, environment, "opposite_history", "primary", "crossover"), f"{environment}-opposite")
        unrelated = boot(_values(rows, environment, "unrelated_pair", "primary", "crossover"), f"{environment}-unrelated")
        passed = bool(
            float(intact["mean"] or 0.0) >= contract.stress_crossover
            and positive(intact)
            and float(zero_advantage["mean"] or 0.0) >= contract.control_advantage
            and positive(zero_advantage)
            and float(opposite["mean"] or 0.0) <= -contract.stress_crossover
            and opposite["ci"][1] is not None
            and float(opposite["ci"][1]) < 0.0
            and float(unrelated["mean"] or 0.0) >= contract.stress_crossover
            and positive(unrelated)
        )
        if passed:
            stress_passes.append(environment)
        stress_results[environment] = {
            "passed": passed,
            "intact": intact,
            "zero_advantage": zero_advantage,
            "opposite_history": opposite,
            "unrelated_pair": unrelated,
        }

    dose_means = {
        f"{contrast:.2f}": boot(_dose_values(rows, contrast), f"dose-{contrast:.2f}")
        for contrast in profile.dose_contrasts
    }
    slopes: list[float] = []
    x = np.asarray(profile.dose_contrasts, dtype=np.float64)
    for index in range(len(rows)):
        y = []
        valid = True
        for contrast in profile.dose_contrasts:
            values = _dose_values((rows[index],), contrast)
            if not values:
                valid = False
                break
            y.append(values[0])
        if valid:
            slopes.append(float(np.polyfit(x, np.asarray(y), 1)[0]))
    slope = boot(slopes, "dose-slope")
    means = [float(dose_means[f"{contrast:.2f}"]["mean"] or 0.0) for contrast in profile.dose_contrasts]
    monotone = all(right + contract.monotonic_tolerance >= left for left, right in zip(means, means[1:]))
    rank_correlation = float(np.corrcoef(np.arange(len(means)), np.argsort(np.argsort(means)))[0, 1])
    dose_gate = bool(
        abs(means[0]) <= contract.midpoint_tolerance
        and means[-1] >= contract.primary_crossover
        and positive(dose_means[f"{profile.dose_contrasts[-1]:.2f}"])
        and monotone
        and rank_correlation >= 0.90
        and float(slope["mean"] or 0.0) >= contract.stress_crossover
        and positive(slope)
    )
    native_robustness = {}
    robustness_gate = True
    if "native" in profile.primary_environments:
        for condition in ROBUSTNESS_CONDITIONS:
            summary = boot(_values(rows, "native", condition, "primary", "crossover"), f"native-{condition}")
            native_robustness[condition] = summary
            robustness_gate &= bool(
                float(summary["mean"] or 0.0) >= contract.stress_crossover and positive(summary)
            )
    generalization_gate = bool(
        audit.get("writer_gate")
        and len(primary_passes) == len(profile.primary_environments)
        and dose_gate
        and robustness_gate
    )
    density_gate = bool(
        generalization_gate
        and len(stress_passes) == len(profile.stress_environments)
        and bool(profile.stress_environments)
    )
    verdict = (
        "DENSITY_ROBUST_GENERAL_MOTIF_CHANNEL"
        if density_gate
        else "GENERAL_REUSABLE_MOTIF_CHANNEL"
        if generalization_gate
        else "NO_GENERAL_REUSABLE_MOTIF_CHANNEL"
    )
    return {
        "verdict": verdict,
        "claim_boundary": "frozen-reader generalization only; not renewed Plastic Heredity",
        "interval_alpha": alpha,
        "writer_audit": audit,
        "generalization_gate": generalization_gate,
        "density_robust_gate": density_gate,
        "primary_passes": primary_passes,
        "primary_environments": environment_results,
        "stress_passes": stress_passes,
        "stress_environments": stress_results,
        "dose_response": {
            "passed": dose_gate,
            "contrasts": dose_means,
            "monotone_with_tolerance": monotone,
            "rank_correlation": rank_correlation,
            "slope": slope,
        },
        "native_robustness": native_robustness,
        "native_robustness_gate": robustness_gate,
    }


def _queue(design_digest: str, state: str, verdict: str | None = None) -> dict[str, Any]:
    stages = [
        {"stage": 1, "name": "motif_carrier_upper_bound", "state": "complete"},
        {"stage": 2, "name": "freeze_and_generalize_reader", "state": state},
        {"stage": 3, "name": "renewed_heredity_causal_ladder", "state": "blocked_pending_stage2_review"},
        {"stage": 4, "name": "compression_and_robustness", "state": "blocked_pending_stage3_review"},
        {"stage": 5, "name": "localize_inheritance", "state": "blocked_pending_stage4_review"},
    ]
    if verdict:
        stages[1]["verdict"] = verdict
    return {
        "programme": "ca_motif_lineage_five_stage",
        "stage2_design_digest": design_digest,
        "automatic_chaining": False,
        "per_stage_max_hours": 8.0,
        "stages": stages,
    }


def _render_report(results: dict[str, Any]) -> str:
    adjudication = results["adjudication"]
    lines = [
        "# CA motif-lineage Stage 2",
        "",
        f"State: **{results['state']}**. Profile: `{results['profile']}`.",
        f"Verdict: **{adjudication['verdict']}**.",
        f"Elapsed: `{results['elapsed_seconds'] / 3600.0:.3f}` wall hours.",
        "",
        "## Frozen reader",
        "",
        f"`{results['configuration']['configuration_id']}` was imported from the frozen Stage-1 decision without retuning.",
        "",
        "## Generalization",
        "",
    ]
    for environment, value in adjudication.get("primary_environments", {}).items():
        lines.append(
            f"- `{environment}`: pass `{value['passed']}`; intact `{value['intact']['mean']}` "
            f"CI `{value['intact']['ci']}`; unrelated retention `{value['unrelated_retention_fraction']}`."
        )
    dose = adjudication.get("dose_response", {})
    lines.extend(
        (
            "",
            f"Dose-response gate: `{dose.get('passed')}`; rank correlation `{dose.get('rank_correlation')}`.",
            "",
            "## Evidence boundary",
            "",
            "A positive result establishes a reusable, frozen, motif-based form-control channel across new parents and resets. "
            "Only Stage 3 can test active rewriting and multigenerational Plastic Heredity.",
        )
    )
    return "\n".join(lines) + "\n"


def _render_lay(results: dict[str, Any]) -> str:
    verdict = results["adjudication"]["verdict"]
    if verdict == "DENSITY_ROBUST_GENERAL_MOTIF_CHANNEL":
        finding = "The frozen texture-memory reader worked across every registered reset, symmetry, transfer, and density challenge."
    elif verdict == "GENERAL_REUSABLE_MOTIF_CHANNEL":
        finding = "The frozen texture-memory reader generalized across the core reset tests, though not every density stress passed."
    elif verdict == "INCOMPLETE":
        finding = "The run stopped before all frozen-reader tests finished and can be resumed."
    else:
        finding = "The Stage-1 reader did not generalize across the complete registered challenge panel."
    return (
        "# Lay summary\n\n"
        f"{finding}\n\n"
        "We locked the successful settings before this run and applied them to entirely new parent pairs. Daughters began from "
        "different standard launch boards, shifted and rotated boards, reflected boards, and random boards of several densities. "
        "Controls erased, scrambled, averaged, randomized, or reversed the inherited pattern memory.\n\n"
        "This stage asks whether the memory is a reusable channel rather than a trick tied to one starting board. It still does not "
        "show heredity across generations; that requires daughters to rewrite and pass the memory onward in Stage 3.\n"
    )


def _update_discovery_log(results: dict[str, Any]) -> None:
    path = ROOT / "DISCOVERY_LOG_EIDOSOMA_SCIENTIST.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Discovery log\n"
    start = "<!-- ca-motif-lineage-stage-2:start -->"
    end = "<!-- ca-motif-lineage-stage-2:end -->"
    section = "\n".join(
        (
            start,
            "## CA motif-lineage Stage 2",
            "",
            f"Frozen-reader verdict: `{results['adjudication']['verdict']}`.",
            f"Profile: `{results['profile']}`; elapsed `{results['elapsed_seconds'] / 3600.0:.3f}` wall hours.",
            "See `results/ca-motif-lineage-stage-2/REPORT.md` and `LAY_SUMMARY.md`.",
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


def run_motif_generalization(
    output: Path,
    *,
    stage1_root: Path = DEFAULT_STAGE1_ROOT,
    profile_name: str = "reference",
    workers: int = 20,
    max_hours: float = 8.0,
    resume: bool = False,
) -> dict[str, Any]:
    require_pinned_numpy()
    if profile_name not in PUBLIC_PROFILES:
        raise ValueError(f"unknown motif-generalization profile {profile_name!r}")
    if max_hours <= 0.0 or max_hours > 8.0:
        raise ValueError("motif-generalization max-hours must be in (0, 8]")
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hard_deadline = started + max_hours * 3600.0
    contract = GeneralizationContract()
    writer_contract = MotifContract()
    profile = GENERALIZATION_PROFILES[profile_name]
    reserve = min(contract.science_reserve_seconds, max(60.0, max_hours * 3600.0 * 0.10))
    science_deadline = max(started, hard_deadline - reserve)

    def status(state: str, phase: str, **extra: Any) -> None:
        now = time.time()
        payload = {
            "state": state,
            "stage": "2-generalization",
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
        status("running", "freeze_audit")
        frozen = load_frozen_stage1(stage1_root)
        configuration = frozen["configuration"]
        pairs = select_stage2_pairs(profile, frozen, contract)
        reset_bank = launch_reset_bank()
        audit = writer_audit(
            pairs[: profile.audit_pairs],
            configuration,
            frozen["reference"],
            writer_contract,
            contract,
        )
        stage1_paths = list(frozen["paths"].values())
        input_paths = [PROTOCOL_PATH, *stage1_paths]
        design_payload = {
            "experiment": "ca_motif_lineage_stage_2",
            "contract": contract.to_dict(),
            "writer_contract_digest": writer_contract.digest,
            "profile_name": profile_name,
            "profile": asdict(profile),
            "configuration": configuration.to_dict(),
            "stage1_design_digest": frozen["design_digest"],
            "stage1_review_authorization": "user explicitly requested the next stage after reviewing the Stage-1 result",
            "pair_ids": [pair["pair_id"] for pair in pairs],
            "source_pair_ids": [pairs[(index + 1) % len(pairs)]["pair_id"] for index in range(len(pairs))],
            "stage1_pair_ids_excluded": len(frozen["used_pair_ids"]),
            "development_pair_ids_excluded_from_reference": list(DEVELOPMENT_PAIR_IDS),
            "launch_resets": {
                str(index): hashlib.sha256(board.tobytes()).hexdigest()
                for index, board in reset_bank.items()
            },
            "input_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in input_paths},
            "implementation_sha256": {
                "motif_generalization.py": _sha256(Path(__file__)),
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
        _atomic_json(output / "WRITER_AUDIT.json", {"design_digest": design_digest, **audit})
        _atomic_json(
            output / "MANIFEST.json",
            {
                "experiment": "ca_motif_lineage_stage_2",
                "stage": 2,
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
                "stage1_pair_ids_excluded": sorted(frozen["used_pair_ids"]),
                "stage2_pair_ids": [pair["pair_id"] for pair in pairs],
            },
        )
        _atomic_json(output / "QUEUE.json", _queue(design_digest, "running"))
        configuration_payload = {
            key: value for key, value in configuration.to_dict().items()
            if key != "configuration_id"
        }
        items = [
            {
                "checkpoint": f"generalization-{index:04d}",
                "pair": pair,
                "source_pair": pairs[(index + 1) % len(pairs)],
                "replicates": profile.replicates,
                "configuration": configuration_payload,
                "primary_environments": profile.primary_environments,
                "stress_environments": profile.stress_environments,
                "core_conditions": profile.core_conditions,
                "stress_conditions": profile.stress_conditions,
                "dose_contrasts": profile.dose_contrasts,
            }
            for index, pair in enumerate(pairs)
        ]
        status("running", "generalization", completed=0, total=len(items))
        rows, complete = _run_checkpoints(
            output,
            items,
            writer_contract,
            contract,
            frozen["reference"],
            reset_bank,
            design_digest,
            workers=workers,
            resume=resume,
            deadline=science_deadline,
            status=status,
        )
        status("running", "adjudication")
        adjudication = adjudicate(rows, audit, profile, contract, complete)
        state = "complete" if complete else "partial_budget_exhausted"
        results = {
            "experiment": "ca_motif_lineage_stage_2",
            "state": state,
            "profile": profile_name,
            "design_digest": design_digest,
            "stage1_design_digest": frozen["design_digest"],
            "configuration": configuration.to_dict(),
            "started_unix": started,
            "completed_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "adjudication": adjudication,
        }
        _atomic_json(output / "RESULTS.json", results)
        _atomic_text(output / "REPORT.md", _render_report(results))
        _atomic_text(output / "LAY_SUMMARY.md", _render_lay(results))
        passed = bool(adjudication.get("generalization_gate"))
        decision = {
            "stage": 2,
            "design_digest": design_digest,
            "stage1_design_digest": frozen["design_digest"],
            "verdict": adjudication["verdict"],
            "review_required": True,
            "automatic_launch": False,
            "decision": "advance_to_stage_3_after_review" if passed else "halt_and_replan_writer_reader_channel",
            "selected_stage3_input": configuration.to_dict() if passed else None,
            "claim_boundary": "Stage 2 cannot establish multigenerational Plastic Heredity",
        }
        _atomic_json(output / "STAGE_DECISION.json", decision)
        queue_state = "complete" if complete else "partial_resumable"
        queue = _queue(design_digest, queue_state, adjudication["verdict"])
        queue["stages"][2]["state"] = (
            "blocked_pending_human_review" if passed else "blocked_stage2_gate_failed"
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
    "CORE_CONDITIONS",
    "DOSE_CONTRASTS",
    "GENERALIZATION_PROFILES",
    "GeneralizationContract",
    "PRIMARY_ENVIRONMENTS",
    "PUBLIC_PROFILES",
    "STRESS_ENVIRONMENTS",
    "adjudicate",
    "environment_reset",
    "launch_reset_bank",
    "load_frozen_stage1",
    "mix_history_carriers",
    "motif_code_permutation",
    "run_motif_generalization",
    "select_stage2_pairs",
    "simulate_generalization_condition",
    "transform_energy_carrier",
    "writer_audit",
]
