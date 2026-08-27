"""Stage-3R semantic-closure diagnosis and repair for the CA motif carrier."""

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
from typing import Any, Iterable, Sequence

import numpy as np

from .causal_heredity import (
    _atomic_json,
    _atomic_text,
    _hash_seed,
    _sha256,
    _state_from_hex,
)
from .e19 import require_pinned_numpy
from .life_family import live_2x2_counts_batch
from .lineage_field import load_round3_pairs
from .motif_lineage import (
    MotifContract,
    ReaderConfiguration,
    _bootstrap,
    _founders,
    _paired_uniforms,
    _score_checkpoint,
    _step,
    _texture_descriptor,
    apply_energy_reader,
    motif3_codes,
    write_parent_carriers,
)
from .motif_lineage_stage3 import (
    CHECKPOINT_GENERATIONS,
    CONDITIONS,
    DEFAULT_STAGE1_ROOT,
    DEFAULT_STAGE2_ROOT,
    Stage3Contract,
    Stage3Profile,
    adjudicate as adjudicate_stage3,
    load_frozen_stage2,
    motif_counts_batch,
    write_energy_from_counts,
)


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "CA_MOTIF_LINEAGE_STAGE3R_PROTOCOL.md"
DEFAULT_STAGE3_ROOT = ROOT / "results/ca-motif-lineage-stage-3"
RULE = 31649
PHASES = ("diagnose", "fit", "screen", "qualify", "adjudicate", "confirm")
DEFAULT_PRECONFIRMATION_PHASES = PHASES[:-1]

WINDOWS: dict[str, tuple[int, int, str]] = {
    "strict-33-48": (33, 48, "strict"),
    "strict-33-64": (33, 64, "strict"),
    "strict-41-56": (41, 56, "strict"),
    "strict-49-64": (49, 64, "strict"),
    "overlap-17-32": (17, 32, "overlap"),
    "overlap-25-40": (25, 40, "overlap"),
    "overlap-17-48": (17, 48, "overlap"),
}
CURRENT_WINDOW_ID = "strict-33-64"
SIMPLE_KINDS = (
    "identity",
    "gain-050",
    "gain-100",
    "gain-200",
    "gain-400",
    "gauge-norm",
)
LEARNED_KINDS = ("scalar-affine", "diagonal-ridge", "reduced-rank-ridge")
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
RANKS = (8, 16, 32, 64)
CHECKPOINT_INDEX = {generation: index for index, generation in enumerate(CHECKPOINT_GENERATIONS)}


@dataclass(frozen=True)
class RepairContract:
    implementation_version: str = "ca-motif-lineage-stage3r-cleanroom-v1"
    namespace: str = "plastic-ca-motif-lineage-stage3r-v1"
    rule: int = RULE
    generation_sweeps: int = 64
    read_sweeps: int = 32
    observe_start: int = 57
    stale_retention: float = 0.50
    process_noise: float = 0.002
    carrier_corruption: float = 0.01
    screen_generation4: float = 0.20
    screen_generation8: float = 0.15
    screen_generation16: float = 0.10
    control_advantage: float = 0.10
    survival_gate: float = 0.90
    loss_fraction: float = 0.70
    rescue_fraction: float = 0.70
    strict_alpha: float = 0.025
    confirmation_alpha_per_class: float = 0.0125
    decoder_mean_gate: float = 0.65
    decoder_lower_gate: float = 0.55
    decoder_null_ceiling: float = 0.55
    decoder_advantage: float = 0.10
    decoder_splits: int = 4
    science_reserve_seconds: float = 1800.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "visible_reset": "bitwise-identical native board before every generation",
                "runtime_repair_access": "raw daughter carrier and frozen universal parameters only",
                "independent_unit": "matched founder pair",
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
class RepairProfile:
    diagnostic_role: str
    diagnostic_pairs: int
    diagnostic_replicates: int
    diagnostic_generations: int
    selection_role: str
    selection_pairs: int
    selection_replicates: int
    selection_generations: int
    confirmation_pairs: int
    confirmation_replicates: int
    confirmation_generations: int
    bootstrap_resamples: int


REPAIR_PROFILES: dict[str, RepairProfile] = {
    "smoke": RepairProfile(
        "smoke", 2, 2, 4, "smoke", 2, 2, 4, 2, 2, 4, 100
    ),
    "pilot": RepairProfile(
        "reference_subset", 16, 8, 8, "pilot", 16, 8, 8, 16, 8, 8, 1_000
    ),
    "reference": RepairProfile(
        "reference", 64, 64, 16, "selection", 64, 32, 16, 96, 64, 16, 10_000
    ),
}
PUBLIC_PROFILES = tuple(REPAIR_PROFILES)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)


def _repeat_histories(values: np.ndarray, replicates: int) -> np.ndarray:
    return np.concatenate(
        (
            np.repeat(values[0:1], replicates, axis=0),
            np.repeat(values[1:2], replicates, axis=0),
        ),
        axis=0,
    )


def _swap_histories(values: np.ndarray, replicates: int) -> np.ndarray:
    return np.concatenate((values[replicates:], values[:replicates]), axis=0)


def load_frozen_stage3(
    stage3_root: Path = DEFAULT_STAGE3_ROOT,
    stage2_root: Path = DEFAULT_STAGE2_ROOT,
    stage1_root: Path = DEFAULT_STAGE1_ROOT,
) -> dict[str, Any]:
    stage3_root = stage3_root.resolve()
    paths = {
        name: stage3_root / filename
        for name, filename in (
            ("decision", "STAGE_DECISION.json"),
            ("results", "RESULTS.json"),
            ("design", "DESIGN.json"),
            ("cohorts", "COHORTS.json"),
            ("manifest", "MANIFEST.json"),
        )
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen Stage-3 artifacts: {missing}")
    payload = {name: _load_json(path) for name, path in paths.items()}
    digest = str(payload["decision"]["design_digest"])
    for name in ("results", "design", "cohorts", "manifest"):
        if str(payload[name].get("design_digest")) != digest:
            raise ValueError(f"Stage-3 {name} design digest does not match the decision")
    if payload["results"].get("state") != "complete":
        raise ValueError("Stage 3 is not complete")
    if payload["decision"].get("decision") != "halt_and_replan_renewal_mechanism":
        raise ValueError("Stage-3 decision is not the registered repair branch")
    frozen2 = load_frozen_stage2(stage2_root, stage1_root)
    if payload["results"].get("stage2_design_digest") != frozen2["design_digest"]:
        raise ValueError("Stage-2 ancestry does not match Stage 3")
    configuration = frozen2["configuration"]
    if configuration.id != payload["results"]["configuration"]["configuration_id"]:
        raise ValueError("Stage-3 reader does not match the frozen Stage-2 winner")

    all_pairs = load_round3_pairs()[RULE]
    by_id = {pair["pair_id"]: pair for pair in all_pairs}
    development_ids = tuple(payload["cohorts"]["development_pair_ids"])
    reference_ids = tuple(payload["cohorts"]["stage3_pair_ids"])
    used_ids = set(payload["cohorts"]["prior_pair_ids_excluded"])
    used_ids.update(development_ids)
    used_ids.update(reference_ids)
    missing_ids = [pair_id for pair_id in (*development_ids, *reference_ids) if pair_id not in by_id]
    if missing_ids:
        raise ValueError(f"Stage-3 pairs missing from the frozen pair bank: {missing_ids}")
    return {
        **payload,
        "root": stage3_root,
        "paths": paths,
        "design_digest": digest,
        "stage2": frozen2,
        "configuration": configuration,
        "reference": frozen2["reference"],
        "used_pair_ids": used_ids,
        "development_pairs": [by_id[pair_id] for pair_id in development_ids],
        "reference_pairs": [by_id[pair_id] for pair_id in reference_ids],
        "all_pairs": all_pairs,
        "by_id": by_id,
    }


def select_repair_cohorts(
    profile: RepairProfile,
    frozen: dict[str, Any],
    contract: RepairContract,
) -> dict[str, list[dict[str, Any]]]:
    unused = [
        pair for pair in frozen["all_pairs"]
        if pair["pair_id"] not in frozen["used_pair_ids"]
    ]
    ordered = sorted(
        unused,
        key=lambda pair: (
            hashlib.sha256(
                f"{contract.namespace}:cohort:{pair['pair_id']}".encode()
            ).hexdigest(),
            pair["pair_id"],
        ),
    )
    if profile.diagnostic_role == "smoke":
        diagnostic = frozen["development_pairs"][: profile.diagnostic_pairs]
    else:
        diagnostic = frozen["reference_pairs"][: profile.diagnostic_pairs]
    if profile.selection_role == "smoke":
        selection = frozen["development_pairs"][: profile.selection_pairs]
        confirmation = frozen["development_pairs"][: profile.confirmation_pairs]
    elif profile.selection_role == "selection":
        selection = ordered[: profile.selection_pairs]
        confirmation = ordered[
            profile.selection_pairs : profile.selection_pairs + profile.confirmation_pairs
        ]
    else:
        start = 160
        selection = ordered[start : start + profile.selection_pairs]
        confirmation = ordered[
            start + profile.selection_pairs :
            start + profile.selection_pairs + profile.confirmation_pairs
        ]
    expected = (
        profile.diagnostic_pairs,
        profile.selection_pairs,
        profile.confirmation_pairs,
    )
    observed = (len(diagnostic), len(selection), len(confirmation))
    if observed != expected:
        raise ValueError(f"insufficient Stage-3R pairs: expected {expected}, observed {observed}")
    if profile.selection_role != "smoke":
        selection_ids = {pair["pair_id"] for pair in selection}
        confirmation_ids = {pair["pair_id"] for pair in confirmation}
        if selection_ids & confirmation_ids:
            raise AssertionError("selection and confirmation cohorts overlap")
        if (selection_ids | confirmation_ids) & frozen["used_pair_ids"]:
            raise AssertionError("Stage-3R scientific cohort contains an exposed pair")
    return {
        "diagnostic": diagnostic,
        "selection": selection,
        "confirmation": confirmation,
    }


def _repair_uniforms(
    pair_id: str,
    purpose: str,
    generation: int,
    sweep: int,
    replicates: int,
) -> np.ndarray:
    return _paired_uniforms(
        pair_id,
        f"stage3r-{purpose}-generation-{generation}",
        sweep,
        replicates,
    )


def _stage3_uniforms(
    pair_id: str,
    purpose: str,
    generation: int,
    sweep: int,
    replicates: int,
) -> np.ndarray:
    return _paired_uniforms(
        pair_id,
        f"stage3-{purpose}-generation-{generation}",
        sweep,
        replicates,
    )


def phenotype_vectors(state: np.ndarray, accumulated: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            _normalize_rows(accumulated),
            _normalize_rows(live_2x2_counts_batch(state)),
            _texture_descriptor(state),
        ),
        axis=1,
    )


def heldout_lineage_accuracy(
    vectors: np.ndarray,
    replicates: int,
    seed: int,
    splits: int = 4,
) -> float:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or len(values) != 2 * replicates:
        raise ValueError("decoder vectors must have two equal history blocks")
    if replicates < 2:
        return 0.5
    accuracies: list[float] = []
    for split in range(splits):
        permutation = np.random.default_rng(_hash_seed(seed, split)).permutation(replicates)
        take = max(1, replicates // 2)
        train = permutation[:take]
        test = permutation[take:]
        if not len(test):
            test = train
        train_indices = np.concatenate((train, replicates + train))
        pooled = values[train_indices]
        centre = pooled.mean(axis=0)
        scale = pooled.std(axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (values - centre) / scale
        centroid_a = standardized[train].mean(axis=0)
        centroid_b = standardized[replicates + train].mean(axis=0)
        test_a = standardized[test]
        test_b = standardized[replicates + test]
        correct_a = np.mean(
            np.linalg.norm(test_a - centroid_a, axis=1)
            < np.linalg.norm(test_a - centroid_b, axis=1)
        )
        correct_b = np.mean(
            np.linalg.norm(test_b - centroid_b, axis=1)
            < np.linalg.norm(test_b - centroid_a, axis=1)
        )
        accuracies.append(0.5 * float(correct_a + correct_b))
    return float(np.mean(accuracies))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def carrier_transition_metrics(
    entry: np.ndarray,
    written: np.ndarray,
    founder_delta: np.ndarray,
    replicates: int,
) -> dict[str, float]:
    entry_shaped = entry.reshape(2, replicates, 512)
    written_shaped = written.reshape(2, replicates, 512)
    entry_centroids = entry_shaped.mean(axis=1)
    written_centroids = written_shaped.mean(axis=1)
    entry_delta = entry_centroids[0] - entry_centroids[1]
    written_delta = written_centroids[0] - written_centroids[1]
    denominator = max(float(np.sqrt(np.mean(entry * entry))), 1e-12)
    bias = np.mean(
        np.linalg.norm(written_centroids - entry_centroids, axis=1)
    )
    within = np.mean(
        np.var(written_shaped, axis=1)
    )
    return {
        "normalized_rmse": float(np.sqrt(np.mean((written - entry) ** 2)) / denominator),
        "centroid_bias_l2": float(bias),
        "within_history_variance": float(within),
        "entry_delta_l2": float(np.linalg.norm(entry_delta)),
        "written_delta_l2": float(np.linalg.norm(written_delta)),
        "parent_child_delta_cosine": _cosine(entry_delta, written_delta),
        "founder_delta_cosine": _cosine(founder_delta, written_delta),
    }


def _core_outcomes(outcomes: dict[str, Any]) -> dict[str, Any]:
    return {
        generation: {
            "survival": value["survival"],
            "primary": value["primary"],
            "terminal": value["terminal"],
        }
        for generation, value in outcomes.items()
    }


def _score_state(
    state: np.ndarray,
    recent: deque[np.ndarray],
    pair: dict[str, Any],
    founder_terminal: np.ndarray,
    replicates: int,
    writer_contract: MotifContract,
) -> tuple[dict[str, Any], np.ndarray]:
    accumulated = np.sum(np.stack(tuple(recent)), axis=0)
    outcome = _score_checkpoint(
        state,
        accumulated,
        pair,
        founder_terminal,
        replicates,
        writer_contract,
        diagnostics=False,
    )
    return outcome, phenotype_vectors(state, accumulated)


def _diagnostic_mode(
    pair: dict[str, Any],
    founder_carrier: np.ndarray,
    founder_terminal: np.ndarray,
    mode: str,
    replicates: int,
    generations: int,
    reference_probability: np.ndarray,
    writer_contract: MotifContract,
    contract: RepairContract,
    *,
    retain_trace: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    pair_id = str(pair["pair_id"])
    reset_state = _state_from_hex("life", pair["donor_a"]["initial_state_hex"])
    other_reset = _state_from_hex("life", pair["donor_b"]["initial_state_hex"])
    if not np.array_equal(reset_state, other_reset):
        raise AssertionError(f"visible reset mismatch in pair {pair_id}")
    reset = np.repeat(reset_state[None, ...], 2 * replicates, axis=0)
    founder = _repeat_histories(founder_carrier, replicates)
    carrier = founder.copy()
    alive = np.ones(2 * replicates, dtype=np.bool_)
    checkpoints = set(value for value in CHECKPOINT_GENERATIONS if value <= generations)
    outcomes: dict[str, Any] = {}
    decoders: dict[str, Any] = {}
    transitions: dict[str, Any] = {}
    entries: list[np.ndarray] = []
    exits: list[np.ndarray] = []
    checkpoint_carriers: list[np.ndarray] = []
    checkpoint_phenotypes: list[np.ndarray] = []
    window_centroids: list[np.ndarray] = []
    window_ids = tuple(WINDOWS)

    for generation in range(1, generations + 1):
        if mode == "founder_clamped":
            carrier = founder.copy()
        entry = carrier.copy()
        state = reset.copy()
        state[~alive] = False
        recent: deque[np.ndarray] = deque(maxlen=writer_contract.observation_window)
        window_counts = {
            window_id: np.zeros((2 * replicates, 512), dtype=np.float64)
            for window_id in window_ids
        }
        for sweep in range(1, contract.generation_sweeps + 1):
            predicted = _step(state, contract.rule)
            if sweep <= contract.read_sweeps:
                predicted = apply_energy_reader(
                    predicted,
                    entry,
                    _stage3_uniforms(pair_id, "read", generation, sweep, replicates),
                    0.25,
                )
            predicted ^= (
                _stage3_uniforms(pair_id, "process", generation, sweep, replicates)
                < contract.process_noise
            )
            predicted[~alive] = False
            state = predicted
            codes: np.ndarray | None = None
            for window_id, (start, end, _) in WINDOWS.items():
                if start <= sweep <= end:
                    if codes is None:
                        codes = motif3_codes(state)
                    window_counts[window_id] += motif_counts_batch(codes)
            if sweep >= contract.observe_start:
                recent.append(live_2x2_counts_batch(state))
        alive &= state.any(axis=(1, 2))
        raw_by_window = {
            window_id: write_energy_from_counts(
                window_counts[window_id], reference_probability, writer_contract
            )
            for window_id in window_ids
        }
        raw_current = raw_by_window[CURRENT_WINDOW_ID]
        if mode == "intact":
            carrier = raw_current
        elif mode == "no_rewrite":
            carrier = entry * contract.stale_retention
        elif mode in ("exact_parent", "founder_clamped"):
            carrier = entry
        else:
            raise ValueError(f"unknown diagnostic mode {mode!r}")
        carrier[~alive] = 0.0
        if generation in checkpoints:
            outcome, vectors = _score_state(
                state, recent, pair, founder_terminal, replicates, writer_contract
            )
            outcomes[str(generation)] = outcome
            decoders[str(generation)] = {
                "carrier_balanced_accuracy": heldout_lineage_accuracy(
                    carrier,
                    replicates,
                    _hash_seed(contract.namespace, pair_id, mode, generation, "carrier"),
                    contract.decoder_splits,
                ),
                "phenotype_balanced_accuracy": heldout_lineage_accuracy(
                    vectors,
                    replicates,
                    _hash_seed(contract.namespace, pair_id, mode, generation, "phenotype"),
                    contract.decoder_splits,
                ),
            }
            if retain_trace:
                checkpoint_carriers.append(carrier.copy())
                checkpoint_phenotypes.append(vectors.copy())
        if retain_trace:
            entries.append(entry)
            exits.append(carrier.copy())
            founder_delta = founder[:replicates].mean(axis=0) - founder[replicates:].mean(axis=0)
            transitions[str(generation)] = carrier_transition_metrics(
                entry, raw_current, founder_delta, replicates
            )
            if generation <= 4:
                window_centroids.append(
                    np.stack(
                        [
                            raw_by_window[window_id].reshape(2, replicates, 512).mean(axis=1)
                            for window_id in window_ids
                        ]
                    )
                )

    trace: dict[str, np.ndarray] = {}
    if retain_trace:
        trace = {
            "entries": np.stack(entries).astype(np.float32),
            "exits": np.stack(exits).astype(np.float32),
            "checkpoint_carriers": np.stack(checkpoint_carriers).astype(np.float32),
            "checkpoint_phenotypes": np.stack(checkpoint_phenotypes).astype(np.float32),
            "window_centroids": np.stack(window_centroids, axis=1).astype(np.float32),
        }
    return {
        "mode": mode,
        "outcomes": outcomes,
        "decoders": decoders,
        "transitions": transitions,
    }, trace


def _assay_carrier(
    pair: dict[str, Any],
    carrier: np.ndarray,
    founder_terminal: np.ndarray,
    replicates: int,
    writer_contract: MotifContract,
    contract: RepairContract,
    purpose: str,
) -> tuple[dict[str, Any], np.ndarray]:
    pair_id = str(pair["pair_id"])
    reset_state = _state_from_hex("life", pair["donor_a"]["initial_state_hex"])
    state = np.repeat(reset_state[None, ...], 2 * replicates, axis=0)
    recent: deque[np.ndarray] = deque(maxlen=writer_contract.observation_window)
    for sweep in range(1, contract.generation_sweeps + 1):
        predicted = _step(state, contract.rule)
        if sweep <= contract.read_sweeps:
            predicted = apply_energy_reader(
                predicted,
                carrier,
                _repair_uniforms(pair_id, purpose, 1, sweep, replicates),
                0.25,
            )
        predicted ^= (
            _repair_uniforms(pair_id, f"{purpose}-process", 1, sweep, replicates)
            < contract.process_noise
        )
        state = predicted
        if sweep >= contract.observe_start:
            recent.append(live_2x2_counts_batch(state))
    return _score_state(
        state, recent, pair, founder_terminal, replicates, writer_contract
    )


def _ensemble_carrier(carrier: np.ndarray, replicates: int, group: int) -> np.ndarray:
    if group <= 1:
        return carrier.copy()
    if replicates % group:
        raise ValueError("ensemble group must divide the replicate count")
    shaped = carrier.reshape(2, replicates, 512)
    grouped = shaped.reshape(2, replicates // group, group, 512).mean(axis=2)
    return np.repeat(grouped, group, axis=1).reshape(2 * replicates, 512).astype(np.float32)


def _diagnostic_pair_task(payload: tuple[dict[str, Any], MotifContract, RepairContract, dict[int, dict[str, np.ndarray]]]) -> dict[str, Any]:
    item, writer_contract, contract, reference = payload
    pair = item["pair"]
    replicates = int(item["replicates"])
    generations = int(item["generations"])
    configuration = ReaderConfiguration(**item["configuration"])
    written = write_parent_carriers(
        _founders(pair), (configuration.write_window,), reference, writer_contract
    )[configuration.write_window]
    founder_carrier = written[configuration.family]
    founder_terminal = written["terminal"]
    reference_probability = reference[configuration.write_window]["motif_probability"]
    modes: dict[str, Any] = {}
    intact, trace = _diagnostic_mode(
        pair,
        founder_carrier,
        founder_terminal,
        "intact",
        replicates,
        generations,
        reference_probability,
        writer_contract,
        contract,
        retain_trace=True,
    )
    modes["intact"] = intact
    expected = item.get("expected_outcomes")
    if expected is not None and _core_outcomes(intact["outcomes"]) != _core_outcomes(expected):
        raise AssertionError(f"Stage-3 baseline replay mismatch for {pair['pair_id']}")
    for mode in ("no_rewrite", "exact_parent", "founder_clamped"):
        result, _ = _diagnostic_mode(
            pair,
            founder_carrier,
            founder_terminal,
            mode,
            replicates,
            generations,
            reference_probability,
            writer_contract,
            contract,
            retain_trace=False,
        )
        modes[mode] = result

    window_assays: dict[str, Any] = {}
    first_generation_centroids = trace["window_centroids"][:, 0]
    founder = _repeat_histories(founder_carrier, replicates)
    founder_delta = founder[:replicates].mean(axis=0) - founder[replicates:].mean(axis=0)
    for index, window_id in enumerate(WINDOWS):
        carrier = np.concatenate(
            (
                np.repeat(first_generation_centroids[index, 0:1], replicates, axis=0),
                np.repeat(first_generation_centroids[index, 1:2], replicates, axis=0),
            ),
            axis=0,
        ).astype(np.float32)
        outcome, vectors = _assay_carrier(
            pair,
            carrier,
            founder_terminal,
            replicates,
            writer_contract,
            contract,
            "window-assay",
        )
        delta = first_generation_centroids[index, 0] - first_generation_centroids[index, 1]
        window_assays[window_id] = {
            "outcome": outcome,
            "founder_delta_cosine": _cosine(founder_delta, delta),
            "carrier_balanced_accuracy": heldout_lineage_accuracy(
                carrier,
                replicates,
                _hash_seed(contract.namespace, pair["pair_id"], window_id, "carrier"),
                contract.decoder_splits,
            ),
            "phenotype_balanced_accuracy": heldout_lineage_accuracy(
                vectors,
                replicates,
                _hash_seed(contract.namespace, pair["pair_id"], window_id, "phenotype"),
                contract.decoder_splits,
            ),
        }

    ensemble_assays: dict[str, Any] = {}
    raw_current = trace["exits"][0]
    for group in (1, 4, 16):
        if group > replicates or replicates % group:
            continue
        carrier = _ensemble_carrier(raw_current, replicates, group)
        outcome, _ = _assay_carrier(
            pair,
            carrier,
            founder_terminal,
            replicates,
            writer_contract,
            contract,
            "ensemble-assay",
        )
        ensemble_assays[str(group)] = outcome
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": pair["pair_id"],
        "summary": {
            "pair_id": pair["pair_id"],
            "replicates": replicates,
            "generations": generations,
            "baseline_reproduced": expected is None or True,
            "modes": modes,
            "window_assays": window_assays,
            "ensemble_assays": ensemble_assays,
        },
        "trace": trace,
    }


def _model_complexity(kind: str, rank: int | None = None) -> int:
    if kind == "identity":
        return 0
    if kind.startswith("gain-"):
        return 1
    if kind in ("gauge-norm", "scalar-affine"):
        return 2
    if kind == "diagonal-ridge":
        return 1024
    if kind == "reduced-rank-ridge":
        return int(1024 * int(rank or 0) + 512)
    raise ValueError(f"unknown repair kind {kind!r}")


def _candidate_id(window_id: str, mechanism_class: str, kind: str, suffix: str = "") -> str:
    value = f"{mechanism_class}--{window_id}--{kind}"
    return f"{value}--{suffix}" if suffix else value


def apply_repair(raw_carrier: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Apply a frozen repair without access to parent state, labels, or targets."""

    raw = np.asarray(raw_carrier, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 512:
        raise ValueError("raw carrier must have shape (sample, 512)")
    kind = str(model["kind"])
    if kind == "identity":
        repaired = raw.copy()
    elif kind.startswith("gain-"):
        repaired = raw * float(model["gain"])
    elif kind == "gauge-norm":
        centred = raw - raw.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centred, axis=1, keepdims=True)
        repaired = np.divide(
            centred * float(model["norm_target"]),
            norms,
            out=np.zeros_like(centred),
            where=norms > 0.0,
        )
    elif kind == "scalar-affine":
        repaired = raw * float(model["slope"]) + float(model["intercept"])
    elif kind == "diagonal-ridge":
        repaired = raw * np.asarray(model["slope"], dtype=np.float32)[None, :] + np.asarray(
            model["intercept"], dtype=np.float32
        )[None, :]
    elif kind == "reduced-rank-ridge":
        repaired = raw @ np.asarray(model["coefficient"], dtype=np.float32) + np.asarray(
            model["intercept"], dtype=np.float32
        )[None, :]
    else:
        raise ValueError(f"unknown repair kind {kind!r}")
    return np.clip(repaired, -4.0, 4.0).astype(np.float32, copy=False)


def build_simple_models(
    window_ids: Sequence[str], norm_target: float
) -> list[dict[str, Any]]:
    gains = {
        "gain-050": 0.5,
        "gain-100": 1.0,
        "gain-200": 2.0,
        "gain-400": 4.0,
    }
    result: list[dict[str, Any]] = []
    for window_id in window_ids:
        tier = WINDOWS[window_id][2]
        for kind in SIMPLE_KINDS:
            model: dict[str, Any] = {
                "candidate_id": _candidate_id(window_id, "simple", kind),
                "mechanism_class": "simple",
                "window_id": window_id,
                "tier": tier,
                "kind": kind,
                "complexity": _model_complexity(kind),
                "runtime_parent_access": False,
                "runtime_label_access": False,
                "runtime_target_access": False,
            }
            if kind.startswith("gain-"):
                model["gain"] = gains[kind]
            elif kind == "gauge-norm":
                model["norm_target"] = float(norm_target)
            result.append(model)
    return result


def _fit_scalar(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    x_flat = np.asarray(x, dtype=np.float64).ravel()
    y_flat = np.asarray(y, dtype=np.float64).ravel()
    x_mean = float(x_flat.mean())
    y_mean = float(y_flat.mean())
    centred_x = x_flat - x_mean
    slope = float(np.dot(centred_x, y_flat - y_mean) / (np.dot(centred_x, centred_x) + alpha))
    return {"slope": slope, "intercept": y_mean - slope * x_mean}


def _fit_diagonal(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    x_mean = x_values.mean(axis=0)
    y_mean = y_values.mean(axis=0)
    centred_x = x_values - x_mean
    centred_y = y_values - y_mean
    slope = np.sum(centred_x * centred_y, axis=0) / (
        np.sum(centred_x * centred_x, axis=0) + alpha
    )
    intercept = y_mean - slope * x_mean
    return {"slope": slope.astype(np.float32), "intercept": intercept.astype(np.float32)}


def _fit_reduced_rank(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    rank: int,
) -> dict[str, Any]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    x_mean = x_values.mean(axis=0)
    y_mean = y_values.mean(axis=0)
    centred_x = x_values - x_mean
    centred_y = y_values - y_mean
    u, singular, vt = np.linalg.svd(centred_x, full_matrices=False)
    _, _, y_vt = np.linalg.svd(centred_y, full_matrices=False)
    actual_rank = max(1, min(int(rank), len(y_vt), len(singular)))
    components = y_vt[:actual_rank]
    target_scores = centred_y @ components.T
    weights = singular / (singular * singular + alpha)
    score_coefficient = vt.T @ (weights[:, None] * (u.T @ target_scores))
    coefficient = score_coefficient @ components
    intercept = y_mean - x_mean @ coefficient
    return {
        "coefficient": coefficient.astype(np.float32),
        "intercept": intercept.astype(np.float32),
        "actual_rank": actual_rank,
    }


def _predict_fit(x: np.ndarray, kind: str, fitted: dict[str, Any]) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if kind == "scalar-affine":
        return values * float(fitted["slope"]) + float(fitted["intercept"])
    if kind == "diagonal-ridge":
        return values * np.asarray(fitted["slope"])[None, :] + np.asarray(
            fitted["intercept"]
        )[None, :]
    if kind == "reduced-rank-ridge":
        return values @ np.asarray(fitted["coefficient"]) + np.asarray(
            fitted["intercept"]
        )[None, :]
    raise ValueError(kind)


def _fit_kind(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
    alpha: float,
    rank: int | None,
) -> dict[str, Any]:
    if kind == "scalar-affine":
        return _fit_scalar(x, y, alpha)
    if kind == "diagonal-ridge":
        return _fit_diagonal(x, y, alpha)
    if kind == "reduced-rank-ridge":
        if rank is None:
            raise ValueError("reduced-rank repair requires a rank")
        return _fit_reduced_rank(x, y, alpha, rank)
    raise ValueError(kind)


def _normalized_error(predicted: np.ndarray, target: np.ndarray) -> float:
    denominator = max(float(np.sqrt(np.mean(np.asarray(target) ** 2))), 1e-12)
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(target)) ** 2)) / denominator)


def _direction_cosine(predicted: np.ndarray, target: np.ndarray) -> float:
    if len(predicted) % 2:
        raise ValueError("repair samples must be ordered in A/B pairs")
    shaped_predicted = np.asarray(predicted).reshape(-1, 2, 512)
    shaped_target = np.asarray(target).reshape(-1, 2, 512)
    values = [
        _cosine(row_predicted[0] - row_predicted[1], row_target[0] - row_target[1])
        for row_predicted, row_target in zip(shaped_predicted, shaped_target)
    ]
    return float(np.mean(values)) if values else 0.0


def _fold_assignments(groups: Sequence[str], folds: int) -> np.ndarray:
    unique = sorted(set(groups))
    count = max(2, min(int(folds), len(unique)))
    mapping = {
        group: int(_hash_seed("stage3r-repair-fold", group) % count)
        for group in unique
    }
    assignments = np.asarray([mapping[group] for group in groups], dtype=np.int16)
    # A hash can leave a fold empty for a small cohort. Remap groups cyclically if needed.
    if len(set(assignments.tolist())) < count:
        mapping = {group: index % count for index, group in enumerate(unique)}
        assignments = np.asarray([mapping[group] for group in groups], dtype=np.int16)
    return assignments


def cross_validate_repair(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    kind: str,
    alpha: float,
    rank: int | None = None,
    folds: int = 8,
) -> dict[str, float]:
    if len(x) != len(y) or len(x) != len(groups):
        raise ValueError("repair training arrays and groups must align")
    assignments = _fold_assignments(groups, folds)
    errors: list[float] = []
    cosines: list[float] = []
    for fold in sorted(set(assignments.tolist())):
        train = assignments != fold
        validation = assignments == fold
        fitted = _fit_kind(x[train], y[train], kind, alpha, rank)
        prediction = _predict_fit(x[validation], kind, fitted)
        errors.append(_normalized_error(prediction, y[validation]))
        cosines.append(_direction_cosine(prediction, y[validation]))
    return {
        "normalized_error": float(np.mean(errors)),
        "direction_cosine": float(np.mean(cosines)),
    }


def _training_data(
    diagnostic_root: Path,
    window_id: str,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    if window_id not in WINDOWS:
        raise ValueError(window_id)
    window_index = tuple(WINDOWS).index(window_id)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    groups: list[str] = []
    founder_norms: list[float] = []
    manifests = sorted((diagnostic_root / "checkpoints").glob("*.json"))
    if not manifests:
        raise FileNotFoundError("no diagnostic checkpoints available for repair fitting")
    for manifest_path in manifests:
        manifest = _load_json(manifest_path)
        pair_id = str(manifest["pair_id"])
        trace_path = diagnostic_root / "traces" / f"{manifest_path.stem}.npz"
        if _sha256(trace_path) != manifest["trace_sha256"]:
            raise ValueError(f"diagnostic trace hash mismatch: {trace_path}")
        with np.load(trace_path, allow_pickle=False) as trace:
            entries = np.asarray(trace["entries"], dtype=np.float64)
            raw_centroids = np.asarray(trace["window_centroids"], dtype=np.float64)[window_index]
        generations = min(4, len(raw_centroids), len(entries))
        for generation in range(generations):
            replicates = entries.shape[1] // 2
            target = entries[generation].reshape(2, replicates, 512).mean(axis=1)
            raw = raw_centroids[generation]
            xs.extend((raw[0], raw[1]))
            ys.extend((target[0], target[1]))
            groups.extend((pair_id, pair_id))
            if generation == 0:
                founder_norms.extend((float(np.linalg.norm(target[0])), float(np.linalg.norm(target[1]))))
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        groups,
        float(np.median(founder_norms)),
    )


def fit_learned_models(
    diagnostic_root: Path,
    window_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    models: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"windows": {}}
    norm_targets: list[float] = []
    for window_id in window_ids:
        x, y, groups, norm_target = _training_data(diagnostic_root, window_id)
        norm_targets.append(norm_target)
        window_audit: dict[str, Any] = {}
        for kind in LEARNED_KINDS:
            candidates: list[dict[str, Any]] = []
            ranks: Iterable[int | None] = RANKS if kind == "reduced-rank-ridge" else (None,)
            for rank in ranks:
                for alpha in RIDGES:
                    score = cross_validate_repair(
                        x, y, groups, kind, alpha, rank=rank, folds=8
                    )
                    candidates.append(
                        {
                            "alpha": alpha,
                            "rank": rank,
                            **score,
                        }
                    )
            selected = min(
                candidates,
                key=lambda row: (
                    row["normalized_error"],
                    -row["direction_cosine"],
                    float(row["alpha"]),
                    int(row["rank"] or 0),
                ),
            )
            fitted = _fit_kind(
                x,
                y,
                kind,
                float(selected["alpha"]),
                int(selected["rank"]) if selected["rank"] is not None else None,
            )
            model: dict[str, Any] = {
                "candidate_id": _candidate_id(window_id, "learned", kind),
                "mechanism_class": "learned",
                "window_id": window_id,
                "tier": WINDOWS[window_id][2],
                "kind": kind,
                "alpha": float(selected["alpha"]),
                "rank": int(selected["rank"]) if selected["rank"] is not None else None,
                "complexity": _model_complexity(kind, selected["rank"]),
                "runtime_parent_access": False,
                "runtime_label_access": False,
                "runtime_target_access": False,
                **fitted,
            }
            models.append(model)
            window_audit[kind] = {
                "selected": selected,
                "grid": candidates,
                "training_samples": len(x),
                "pair_groups": len(set(groups)),
            }
        audit["windows"][window_id] = window_audit
    return models, audit, float(np.median(norm_targets))


def save_repair_models(
    output: Path,
    models: Sequence[dict[str, Any]],
    *,
    design_digest: str,
    diagnostic_digest: str,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    metadata: list[dict[str, Any]] = []
    for index, model in enumerate(models):
        row: dict[str, Any] = {}
        array_keys: dict[str, str] = {}
        for key, value in model.items():
            if isinstance(value, np.ndarray):
                array_key = f"model_{index:03d}__{key}"
                arrays[array_key] = np.asarray(value)
                array_keys[key] = array_key
            else:
                row[key] = value
        row["array_keys"] = array_keys
        metadata.append(row)
    model_path = output / "REPAIR_MODELS.npz"
    _atomic_npz(model_path, **arrays)
    manifest = {
        "design_digest": design_digest,
        "diagnostic_digest": diagnostic_digest,
        "allow_pickle": False,
        "model_sha256": _sha256(model_path),
        "models": metadata,
    }
    _atomic_json(output / "REPAIR_MODELS.json", manifest)
    return manifest


def load_repair_models(output: Path, design_digest: str) -> list[dict[str, Any]]:
    manifest = _load_json(output / "REPAIR_MODELS.json")
    if manifest.get("design_digest") != design_digest:
        raise ValueError("repair-model design digest mismatch")
    model_path = output / "REPAIR_MODELS.npz"
    if _sha256(model_path) != manifest.get("model_sha256"):
        raise ValueError("repair-model hash mismatch")
    models: list[dict[str, Any]] = []
    with np.load(model_path, allow_pickle=False) as arrays:
        for metadata in manifest["models"]:
            model = {key: value for key, value in metadata.items() if key != "array_keys"}
            for key, array_key in metadata["array_keys"].items():
                model[key] = np.asarray(arrays[array_key])
            models.append(model)
    return models


def _apply_boundary_intervention(
    carrier: np.ndarray,
    condition: str,
    generation: int,
    pair_id: str,
    replicates: int,
    contract: RepairContract,
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
    elif condition in (
        "ablate_after_g2",
        "rescue_same_enter_g4",
        "rescue_opposite_enter_g4",
    ) and generation == 3:
        result.fill(0.0)
    elif condition in (
        "rescue_same_enter_g4",
        "rescue_opposite_enter_g4",
    ) and generation == 4:
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


def _carrier_summary(carrier: np.ndarray, replicates: int) -> dict[str, float]:
    shaped = carrier.reshape(2, replicates, 512)
    delta = shaped[0].mean(axis=0) - shaped[1].mean(axis=0)
    return {
        "mean_abs": float(np.mean(np.abs(carrier))),
        "centroid_l2": float(np.linalg.norm(delta)),
        "within_history_variance": float(np.mean(np.var(shaped, axis=1))),
    }


def simulate_repair_lineage(
    pair: dict[str, Any],
    configuration: ReaderConfiguration,
    model: dict[str, Any],
    condition: str,
    replicates: int,
    generations: int,
    reference: dict[int, dict[str, np.ndarray]],
    writer_contract: MotifContract,
    contract: RepairContract,
    *,
    source_exits: Sequence[np.ndarray] | None = None,
    retain_exits: bool = False,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    pair_id = str(pair["pair_id"])
    reset_state = _state_from_hex("life", pair["donor_a"]["initial_state_hex"])
    other_reset = _state_from_hex("life", pair["donor_b"]["initial_state_hex"])
    if not np.array_equal(reset_state, other_reset):
        raise AssertionError(f"visible reset mismatch in pair {pair_id}")
    reset = np.repeat(reset_state[None, ...], 2 * replicates, axis=0)
    written = write_parent_carriers(
        _founders(pair), (configuration.write_window,), reference, writer_contract
    )[configuration.write_window]
    founder_carrier = _repeat_histories(written[configuration.family], replicates)
    carrier = founder_carrier.copy()
    if condition == "founder_write_disabled":
        carrier.fill(0.0)
    founder_terminal = written["terminal"]
    alive = np.ones(2 * replicates, dtype=np.bool_)
    checkpoints = set(value for value in CHECKPOINT_GENERATIONS if value <= generations)
    outcomes: dict[str, Any] = {}
    decoders: dict[str, Any] = {}
    carrier_history: dict[str, Any] = {}
    exits: list[np.ndarray] = []
    reference_probability = reference[configuration.write_window]["motif_probability"]
    window_start, window_end, _ = WINDOWS[str(model["window_id"])]

    for generation in range(1, generations + 1):
        carrier = _apply_boundary_intervention(
            carrier,
            condition,
            generation,
            pair_id,
            replicates,
            contract,
            source_exits,
        )
        entry = carrier.copy()
        entry_summary = _carrier_summary(entry, replicates)
        state = reset.copy()
        state[~alive] = False
        if not np.array_equal(state[alive], reset[alive]):
            raise AssertionError("visible reset was not bitwise identical")
        recent: deque[np.ndarray] = deque(maxlen=writer_contract.observation_window)
        counts = np.zeros((2 * replicates, 512), dtype=np.float64)
        for sweep in range(1, contract.generation_sweeps + 1):
            predicted = _step(state, contract.rule)
            if condition != "read_disabled" and sweep <= contract.read_sweeps:
                predicted = apply_energy_reader(
                    predicted,
                    entry,
                    _repair_uniforms(pair_id, "read", generation, sweep, replicates),
                    configuration.strength,
                )
            predicted ^= (
                _repair_uniforms(pair_id, "process", generation, sweep, replicates)
                < contract.process_noise
            )
            predicted[~alive] = False
            state = predicted
            if window_start <= sweep <= window_end:
                counts += motif_counts_batch(motif3_codes(state))
            if sweep >= contract.observe_start:
                recent.append(live_2x2_counts_batch(state))
        alive &= state.any(axis=(1, 2))
        raw = write_energy_from_counts(counts, reference_probability, writer_contract)
        if condition == "no_rewrite":
            carrier = entry * contract.stale_retention
        else:
            carrier = apply_repair(raw, model)
        carrier[~alive] = 0.0
        if generation in checkpoints:
            outcome, vectors = _score_state(
                state, recent, pair, founder_terminal, replicates, writer_contract
            )
            outcomes[str(generation)] = outcome
            decoders[str(generation)] = {
                "carrier_balanced_accuracy": heldout_lineage_accuracy(
                    carrier,
                    replicates,
                    _hash_seed(
                        contract.namespace,
                        pair_id,
                        model["candidate_id"],
                        condition,
                        generation,
                        "carrier",
                    ),
                    contract.decoder_splits,
                ),
                "phenotype_balanced_accuracy": heldout_lineage_accuracy(
                    vectors,
                    replicates,
                    _hash_seed(
                        contract.namespace,
                        pair_id,
                        model["candidate_id"],
                        condition,
                        generation,
                        "phenotype",
                    ),
                    contract.decoder_splits,
                ),
            }
            carrier_history[str(generation)] = {
                "entry": entry_summary,
                "exit": _carrier_summary(carrier, replicates),
                "surviving_futures": int(np.count_nonzero(alive)),
            }
        if retain_exits:
            exits.append(carrier.copy())
    return (
        {
            "candidate_id": model["candidate_id"],
            "condition": condition,
            "reset_sha256": hashlib.sha256(reset_state.tobytes()).hexdigest(),
            "reset_asserted_before_every_generation": True,
            "founder_carrier": _carrier_summary(founder_carrier, replicates),
            "outcomes": outcomes,
            "decoders": decoders,
            "carrier_history": carrier_history,
        },
        exits,
    )


def _screen_pair_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        RepairContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, models, writer_contract, contract, reference = payload
    pair = item["pair"]
    configuration = ReaderConfiguration(**item["configuration"])
    replicates = int(item["replicates"])
    generations = int(item["generations"])
    candidates: dict[str, Any] = {}
    for model in models:
        result, _ = simulate_repair_lineage(
            pair,
            configuration,
            model,
            "intact",
            replicates,
            generations,
            reference,
            writer_contract,
            contract,
        )
        candidates[str(model["candidate_id"])] = result
    baseline_model = next(
        (model for model in models if model["kind"] == "identity"), models[0]
    )
    no_rewrite, _ = simulate_repair_lineage(
        pair,
        configuration,
        baseline_model,
        "no_rewrite",
        replicates,
        generations,
        reference,
        writer_contract,
        contract,
    )
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": pair["pair_id"],
        "replicates": replicates,
        "generations": generations,
        "candidates": candidates,
        "no_rewrite": no_rewrite,
    }


def _qualification_pair_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        RepairContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, models, writer_contract, contract, reference = payload
    pair = item["pair"]
    configuration = ReaderConfiguration(**item["configuration"])
    replicates = int(item["replicates"])
    generations = int(item["generations"])
    candidates: dict[str, Any] = {}
    for model in models:
        intact, exits = simulate_repair_lineage(
            pair,
            configuration,
            model,
            "intact",
            replicates,
            generations,
            reference,
            writer_contract,
            contract,
            retain_exits=True,
        )
        conditions: dict[str, Any] = {"intact": intact}
        for condition in CONDITIONS:
            if condition == "intact":
                continue
            result, _ = simulate_repair_lineage(
                pair,
                configuration,
                model,
                condition,
                replicates,
                generations,
                reference,
                writer_contract,
                contract,
                source_exits=exits,
            )
            conditions[condition] = result
        candidates[str(model["candidate_id"])] = {
            "candidate_id": model["candidate_id"],
            "conditions": conditions,
        }
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": pair["pair_id"],
        "replicates": replicates,
        "generations": generations,
        "candidates": candidates,
    }


def _screen_metric(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    generation: int,
    observer: str,
    metric: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            outcome = row["candidates"][candidate_id]["outcomes"][str(generation)]
        except KeyError:
            continue
        if metric == "survival":
            values.append(float(outcome["survival"]))
        else:
            values.append(float(outcome[observer][metric]))
    return values


def _screen_advantage(
    rows: Sequence[dict[str, Any]], candidate_id: str, generation: int
) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            intact = row["candidates"][candidate_id]["outcomes"][str(generation)][
                "primary"
            ]["crossover"]
            stale = row["no_rewrite"]["outcomes"][str(generation)]["primary"][
                "crossover"
            ]
        except KeyError:
            continue
        values.append(float(intact) - float(stale))
    return values


def adjudicate_screen(
    rows: Sequence[dict[str, Any]],
    models: Sequence[dict[str, Any]],
    profile: RepairProfile,
    contract: RepairContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"state": "incomplete", "selected_candidate_ids": []}

    def boot(values: Sequence[float], name: str) -> dict[str, Any]:
        return _bootstrap(
            values,
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "screen", name),
            contract.strict_alpha,
        )

    summaries: dict[str, Any] = {}
    for model in models:
        candidate_id = str(model["candidate_id"])
        checkpoints: dict[str, Any] = {}
        for generation in CHECKPOINT_GENERATIONS:
            if generation > profile.selection_generations:
                continue
            crossover = _screen_metric(
                rows, candidate_id, generation, "primary", "crossover"
            )
            checkpoints[str(generation)] = {
                "crossover": boot(crossover, f"{candidate_id}-g{generation}"),
                "survival_mean": float(
                    np.mean(
                        _screen_metric(
                            rows, candidate_id, generation, "primary", "survival"
                        )
                    )
                ),
                "direction_a_mean": float(
                    np.mean(
                        _screen_metric(
                            rows, candidate_id, generation, "primary", "direction_a"
                        )
                    )
                ),
                "direction_b_mean": float(
                    np.mean(
                        _screen_metric(
                            rows, candidate_id, generation, "primary", "direction_b"
                        )
                    )
                ),
                "fraction_pairs_positive": float(
                    np.mean(np.asarray(crossover) > 0.0)
                ),
            }
        if profile.selection_generations >= 8:
            advantage = boot(
                _screen_advantage(rows, candidate_id, 8),
                f"{candidate_id}-stale-advantage",
            )
        else:
            advantage = {
                "n_pairs": 0,
                "mean": None,
                "ci": [None, None],
                "alpha": contract.strict_alpha,
            }
        eligible = False
        if profile.selection_generations >= 16:
            g4 = checkpoints["4"]
            g8 = checkpoints["8"]
            g16 = checkpoints["16"]
            eligible = bool(
                model["tier"] == "strict"
                and float(g4["crossover"]["mean"] or 0.0) >= contract.screen_generation4
                and float(g8["crossover"]["mean"] or 0.0) >= contract.screen_generation8
                and float(g16["crossover"]["mean"] or 0.0) >= contract.screen_generation16
                and g8["crossover"]["ci"][0] is not None
                and float(g8["crossover"]["ci"][0]) > 0.0
                and g16["crossover"]["ci"][0] is not None
                and float(g16["crossover"]["ci"][0]) > 0.0
                and g8["survival_mean"] >= contract.survival_gate
                and g16["survival_mean"] >= contract.survival_gate
                and g8["direction_a_mean"] > 0.0
                and g8["direction_b_mean"] > 0.0
                and g8["fraction_pairs_positive"] >= 0.50
                and float(advantage["mean"] or 0.0) >= contract.control_advantage
                and advantage["ci"][0] is not None
                and float(advantage["ci"][0]) > 0.0
            )
        minimum = min(
            float(checkpoints[str(g)]["crossover"]["mean"] or -1.0)
            for g in (4, 8, 16)
            if str(g) in checkpoints
        )
        summaries[candidate_id] = {
            "candidate": {
                key: value for key, value in model.items() if not isinstance(value, np.ndarray)
            },
            "checkpoints": checkpoints,
            "generation8_no_rewrite_advantage": advantage,
            "screen_eligible": eligible,
            "minimum_checkpoint_crossover": minimum,
        }

    selected: list[str] = []
    if profile.selection_generations < 16:
        for mechanism_class in ("simple", "learned"):
            candidates = [
                model
                for model in models
                if model["mechanism_class"] == mechanism_class and model["tier"] == "strict"
            ]
            if candidates:
                selected.append(
                    min(
                        candidates,
                        key=lambda model: (
                            -summaries[model["candidate_id"]]["minimum_checkpoint_crossover"],
                            int(model["complexity"]),
                            str(model["candidate_id"]),
                        ),
                    )["candidate_id"]
                )
    else:
        for mechanism_class in ("simple", "learned"):
            candidates = [
                model
                for model in models
                if model["mechanism_class"] == mechanism_class
                and summaries[model["candidate_id"]]["screen_eligible"]
            ]
            if candidates:
                selected.append(
                    min(
                        candidates,
                        key=lambda model: (
                            -summaries[model["candidate_id"]]["minimum_checkpoint_crossover"],
                            -float(
                                summaries[model["candidate_id"]]["checkpoints"]["16"][
                                    "crossover"
                                ]["ci"][0]
                            ),
                            int(model["complexity"]),
                            str(model["candidate_id"]),
                        ),
                    )["candidate_id"]
                )
    overlap = [model for model in models if model["tier"] == "overlap"]
    selected_overlap = (
        min(
            overlap,
            key=lambda model: (
                -summaries[model["candidate_id"]]["minimum_checkpoint_crossover"],
                int(model["complexity"]),
                str(model["candidate_id"]),
            ),
        )["candidate_id"]
        if overlap
        else None
    )
    return {
        "state": "complete",
        "candidate_summaries": summaries,
        "selected_candidate_ids": selected,
        "selected_overlap_candidate_id": selected_overlap,
        "scientific_gate_applied": profile.selection_generations >= 16,
    }


def _run_diagnostic_checkpoints(
    output: Path,
    items: Sequence[dict[str, Any]],
    writer_contract: MotifContract,
    contract: RepairContract,
    reference: dict[int, dict[str, np.ndarray]],
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status: Any,
) -> tuple[list[dict[str, Any]], bool]:
    root = output / "diagnostics"
    checkpoint_root = root / "checkpoints"
    trace_root = root / "traces"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for item in items:
        key = item["checkpoint"]
        checkpoint_path = checkpoint_root / f"{key}.json"
        trace_path = trace_root / f"{key}.npz"
        if resume and checkpoint_path.exists() and trace_path.exists():
            payload = _load_json(checkpoint_path)
            if payload.get("design_digest") != design_digest:
                raise ValueError(f"diagnostic checkpoint design mismatch: {checkpoint_path}")
            if _sha256(trace_path) != payload.get("trace_sha256"):
                raise ValueError(f"diagnostic trace hash mismatch: {trace_path}")
            results[key] = payload["summary"]
        else:
            missing.append(item)
    initial = len(results)
    started = time.monotonic()
    truncated = False

    def save(item: dict[str, Any], result: dict[str, Any]) -> None:
        key = item["checkpoint"]
        trace_path = trace_root / f"{key}.npz"
        _atomic_npz(trace_path, **result["trace"])
        trace_shapes = {
            name: list(values.shape) for name, values in result["trace"].items()
        }
        payload = {
            "design_digest": design_digest,
            "phase": "diagnose",
            "checkpoint": key,
            "pair_id": result["pair_id"],
            "trace_sha256": _sha256(trace_path),
            "trace_shapes": trace_shapes,
            "summary": result["summary"],
        }
        _atomic_json(checkpoint_root / f"{key}.json", payload)
        results[key] = result["summary"]
        elapsed = max(time.monotonic() - started, 1e-6)
        completed_new = max(1, len(results) - initial)
        eta = elapsed / completed_new * max(0, len(items) - len(results))
        status(
            "running",
            "diagnose",
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
            future = pool.submit(
                _diagnostic_pair_task,
                (item, writer_contract, contract, reference),
            )
            pending[future] = item
            return True

        for _ in range(min(len(missing), max(1, workers * 2))):
            submit_one()
        try:
            while pending:
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    truncated = True
                    break
                done, _ = wait(
                    tuple(pending),
                    timeout=min(10.0, remaining),
                    return_when=FIRST_COMPLETED,
                )
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
        root / "stage_summary.json",
        {
            "design_digest": design_digest,
            "phase": "diagnose",
            "complete": complete,
            "completed": len(results),
            "total": len(items),
            "budget_truncated": truncated or not complete,
        },
    )
    if complete:
        _atomic_text(root / "COMPLETE", "complete\n")
    return [results[key] for key in sorted(results)], complete


def _run_json_checkpoints(
    output: Path,
    phase: str,
    items: Sequence[dict[str, Any]],
    models: list[dict[str, Any]],
    task: Any,
    writer_contract: MotifContract,
    contract: RepairContract,
    reference: dict[int, dict[str, np.ndarray]],
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status: Any,
) -> tuple[list[dict[str, Any]], bool]:
    root = output / phase
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for item in items:
        key = item["checkpoint"]
        path = checkpoint_root / f"{key}.json"
        if resume and path.exists():
            payload = _load_json(path)
            if payload.get("design_digest") != design_digest:
                raise ValueError(f"{phase} checkpoint design mismatch: {path}")
            results[key] = payload["result"]
        else:
            missing.append(item)
    initial = len(results)
    started = time.monotonic()
    truncated = False

    def save(item: dict[str, Any], result: dict[str, Any]) -> None:
        key = item["checkpoint"]
        _atomic_json(
            checkpoint_root / f"{key}.json",
            {
                "design_digest": design_digest,
                "phase": phase,
                "checkpoint": key,
                "result": result,
            },
        )
        results[key] = result
        elapsed = max(time.monotonic() - started, 1e-6)
        completed_new = max(1, len(results) - initial)
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
            future = pool.submit(
                task,
                (item, models, writer_contract, contract, reference),
            )
            pending[future] = item
            return True

        for _ in range(min(len(missing), max(1, workers * 2))):
            submit_one()
        try:
            while pending:
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    truncated = True
                    break
                done, _ = wait(
                    tuple(pending),
                    timeout=min(10.0, remaining),
                    return_when=FIRST_COMPLETED,
                )
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
        root / "stage_summary.json",
        {
            "design_digest": design_digest,
            "phase": phase,
            "complete": complete,
            "completed": len(results),
            "total": len(items),
            "budget_truncated": truncated or not complete,
        },
    )
    if complete:
        _atomic_text(root / "COMPLETE", "complete\n")
    return [results[key] for key in sorted(results)], complete


def summarize_diagnostics(
    rows: Sequence[dict[str, Any]],
    profile: RepairProfile,
    contract: RepairContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"state": "incomplete", "retained_window_ids": []}

    def boot(values: Sequence[float], name: str) -> dict[str, Any]:
        return _bootstrap(
            values,
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "diagnostic", name),
            contract.strict_alpha,
        )

    modes: dict[str, Any] = {}
    for mode in ("intact", "no_rewrite", "exact_parent", "founder_clamped"):
        generations: dict[str, Any] = {}
        for generation in CHECKPOINT_GENERATIONS:
            if generation > profile.diagnostic_generations:
                continue
            crossover = [
                float(row["modes"][mode]["outcomes"][str(generation)]["primary"]["crossover"])
                for row in rows
            ]
            carrier_decode = [
                float(row["modes"][mode]["decoders"][str(generation)]["carrier_balanced_accuracy"])
                for row in rows
            ]
            phenotype_decode = [
                float(row["modes"][mode]["decoders"][str(generation)]["phenotype_balanced_accuracy"])
                for row in rows
            ]
            generations[str(generation)] = {
                "fixed_form_crossover": boot(crossover, f"{mode}-g{generation}-form"),
                "carrier_decodability": boot(carrier_decode, f"{mode}-g{generation}-carrier"),
                "phenotype_decodability": boot(
                    phenotype_decode, f"{mode}-g{generation}-phenotype"
                ),
            }
        modes[mode] = generations

    transitions: dict[str, Any] = {}
    for generation in range(1, profile.diagnostic_generations + 1):
        values = [row["modes"]["intact"]["transitions"][str(generation)] for row in rows]
        transitions[str(generation)] = {
            metric: boot([float(value[metric]) for value in values], f"transition-g{generation}-{metric}")
            for metric in (
                "normalized_rmse",
                "centroid_bias_l2",
                "within_history_variance",
                "entry_delta_l2",
                "written_delta_l2",
                "parent_child_delta_cosine",
                "founder_delta_cosine",
            )
        }

    window_summaries: dict[str, Any] = {}
    for window_id in WINDOWS:
        crossover = [
            float(row["window_assays"][window_id]["outcome"]["primary"]["crossover"])
            for row in rows
        ]
        cosine = [
            float(row["window_assays"][window_id]["founder_delta_cosine"])
            for row in rows
        ]
        window_summaries[window_id] = {
            "tier": WINDOWS[window_id][2],
            "fixed_form_crossover": boot(crossover, f"window-{window_id}-form"),
            "founder_delta_cosine_mean": float(np.mean(cosine)),
            "carrier_decodability_mean": float(
                np.mean(
                    [row["window_assays"][window_id]["carrier_balanced_accuracy"] for row in rows]
                )
            ),
            "phenotype_decodability_mean": float(
                np.mean(
                    [row["window_assays"][window_id]["phenotype_balanced_accuracy"] for row in rows]
                )
            ),
        }

    def rank_window(window_id: str) -> tuple[float, float, str]:
        value = window_summaries[window_id]
        return (
            -float(value["fixed_form_crossover"]["mean"] or -1.0),
            -float(value["founder_delta_cosine_mean"]),
            window_id,
        )

    strict = sorted(
        (window_id for window_id in WINDOWS if WINDOWS[window_id][2] == "strict"),
        key=rank_window,
    )
    overlap = sorted(
        (window_id for window_id in WINDOWS if WINDOWS[window_id][2] == "overlap"),
        key=rank_window,
    )
    retained = [*strict[:2], *overlap[:1]]
    ensembles: dict[str, Any] = {}
    for group in ("1", "4", "16"):
        values = [
            float(row["ensemble_assays"][group]["primary"]["crossover"])
            for row in rows
            if group in row["ensemble_assays"]
        ]
        if values:
            ensembles[group] = boot(values, f"ensemble-{group}")
    payload = {
        "state": "complete",
        "baseline_reproduced_all_pairs": all(row["baseline_reproduced"] for row in rows),
        "modes": modes,
        "transitions": transitions,
        "window_summaries": window_summaries,
        "retained_strict_window_ids": strict[:2],
        "retained_overlap_window_id": overlap[0] if overlap else None,
        "retained_window_ids": retained,
        "ensemble_assays": ensembles,
    }
    payload["diagnostic_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def adjudicate_qualification(
    rows: Sequence[dict[str, Any]],
    selected_candidate_ids: Sequence[str],
    profile: RepairProfile,
    contract: RepairContract,
    complete: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    qualified: list[str] = []
    for candidate_id in selected_candidate_ids:
        candidate_rows = [
            {"conditions": row["candidates"][candidate_id]["conditions"]}
            for row in rows
            if candidate_id in row["candidates"]
        ]
        value = adjudicate_stage3(
            candidate_rows,
            Stage3Profile(
                "repair-selection",
                profile.selection_pairs,
                profile.selection_replicates,
                profile.selection_generations,
                profile.bootstrap_resamples,
            ),
            Stage3Contract(),
            complete and len(candidate_rows) == profile.selection_pairs,
        )
        results[candidate_id] = value
        if profile.selection_generations < 16 or value.get("renewed_gate"):
            qualified.append(candidate_id)
    return {
        "state": "complete" if complete else "incomplete",
        "candidates": results,
        "qualified_candidate_ids": qualified,
        "scientific_gate_applied": profile.selection_generations >= 16,
    }


def _qualification_metric(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    condition: str,
    generation: int,
    observer: str,
    metric: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            result = row["candidates"][candidate_id]["conditions"][condition]
            if metric == "survival":
                value = result["outcomes"][str(generation)]["survival"]
            elif observer in ("carrier_decoder", "phenotype_decoder"):
                key = (
                    "carrier_balanced_accuracy"
                    if observer == "carrier_decoder"
                    else "phenotype_balanced_accuracy"
                )
                value = result["decoders"][str(generation)][key]
            else:
                value = result["outcomes"][str(generation)][observer][metric]
        except KeyError:
            continue
        values.append(float(value))
    return values


def _qualification_difference(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    left: str,
    right: str,
    generation: int,
    *,
    observer: str = "primary",
    metric: str = "crossover",
) -> list[float]:
    left_values = _qualification_metric(
        rows, candidate_id, left, generation, observer, metric
    )
    right_values = _qualification_metric(
        rows, candidate_id, right, generation, observer, metric
    )
    if len(left_values) != len(right_values):
        raise ValueError("paired qualification metrics do not align")
    return [left - right for left, right in zip(left_values, right_values)]


def _strict_confirmation_gate(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    profile: RepairProfile,
    contract: RepairContract,
    alpha: float,
) -> dict[str, Any]:
    def boot(values: Sequence[float], name: str) -> dict[str, Any]:
        return _bootstrap(
            values,
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "confirmation", candidate_id, name),
            alpha,
        )

    def positive(summary: dict[str, Any]) -> bool:
        return summary["ci"][0] is not None and float(summary["ci"][0]) > 0.0

    intact4 = boot(
        _qualification_metric(rows, candidate_id, "intact", 4, "primary", "crossover"),
        "intact4",
    )
    intact8 = boot(
        _qualification_metric(rows, candidate_id, "intact", 8, "primary", "crossover"),
        "intact8",
    )
    intact16 = boot(
        _qualification_metric(rows, candidate_id, "intact", 16, "primary", "crossover"),
        "intact16",
    )
    terminal8 = boot(
        _qualification_metric(rows, candidate_id, "intact", 8, "terminal", "crossover"),
        "terminal8",
    )
    survival8 = boot(
        _qualification_metric(rows, candidate_id, "intact", 8, "primary", "survival"),
        "survival8",
    )
    survival16 = boot(
        _qualification_metric(rows, candidate_id, "intact", 16, "primary", "survival"),
        "survival16",
    )
    controls = {
        condition: boot(
            _qualification_difference(rows, candidate_id, "intact", condition, 8),
            f"control-{condition}",
        )
        for condition in (
            "zero_every_boundary",
            "shuffle_every_boundary",
            "read_disabled",
            "founder_write_disabled",
        )
    }
    no_rewrite8 = boot(
        _qualification_metric(
            rows, candidate_id, "no_rewrite", 8, "primary", "crossover"
        ),
        "no-rewrite8",
    )
    active_advantage = boot(
        _qualification_difference(rows, candidate_id, "intact", "no_rewrite", 8),
        "active-advantage",
    )
    ablation4 = boot(
        _qualification_metric(
            rows, candidate_id, "ablate_after_g2", 4, "primary", "crossover"
        ),
        "ablation4",
    )
    rescue4 = boot(
        _qualification_metric(
            rows, candidate_id, "rescue_same_enter_g4", 4, "primary", "crossover"
        ),
        "rescue4",
    )
    rescue_advantage = boot(
        _qualification_difference(
            rows,
            candidate_id,
            "rescue_same_enter_g4",
            "ablate_after_g2",
            4,
        ),
        "rescue-advantage",
    )
    opposite_rescue4 = boot(
        _qualification_metric(
            rows,
            candidate_id,
            "rescue_opposite_enter_g4",
            4,
            "primary",
            "crossover",
        ),
        "opposite-rescue4",
    )
    opposite_founder8 = boot(
        _qualification_metric(
            rows, candidate_id, "opposite_founder", 8, "primary", "crossover"
        ),
        "opposite-founder8",
    )
    corruption8 = boot(
        _qualification_metric(
            rows, candidate_id, "carrier_corruption_1", 8, "primary", "crossover"
        ),
        "corruption8",
    )
    directions_a = _qualification_metric(
        rows, candidate_id, "intact", 8, "primary", "direction_a"
    )
    directions_b = _qualification_metric(
        rows, candidate_id, "intact", 8, "primary", "direction_b"
    )
    pair_values = _qualification_metric(
        rows, candidate_id, "intact", 8, "primary", "crossover"
    )
    direction_a = float(np.mean(directions_a))
    direction_b = float(np.mean(directions_b))
    fraction_positive = float(np.mean(np.asarray(pair_values) > 0.0))
    intact4_mean = float(intact4["mean"] or 0.0)
    intact8_mean = float(intact8["mean"] or 0.0)
    no_rewrite_loss = (
        1.0 - float(no_rewrite8["mean"] or 0.0) / intact8_mean
        if intact8_mean > 0.0
        else None
    )
    ablation_loss = (
        1.0 - float(ablation4["mean"] or 0.0) / intact4_mean
        if intact4_mean > 0.0
        else None
    )
    rescue_fraction = (
        float(rescue4["mean"] or 0.0) / intact4_mean
        if intact4_mean > 0.0
        else None
    )
    renewed = bool(
        intact4_mean >= contract.screen_generation4
        and intact8_mean >= contract.screen_generation8
        and positive(intact8)
        and float(intact16["mean"] or 0.0) >= contract.screen_generation16
        and positive(intact16)
        and direction_a > 0.0
        and direction_b > 0.0
        and fraction_positive >= 0.50
        and float(survival8["mean"] or 0.0) >= contract.survival_gate
        and float(survival16["mean"] or 0.0) >= contract.survival_gate
        and float(terminal8["mean"] or 0.0) >= contract.screen_generation16
        and positive(terminal8)
        and all(
            float(value["mean"] or 0.0) >= contract.control_advantage
            and positive(value)
            for value in controls.values()
        )
        and float(active_advantage["mean"] or 0.0) >= contract.control_advantage
        and positive(active_advantage)
        and no_rewrite_loss is not None
        and no_rewrite_loss >= contract.loss_fraction
        and ablation_loss is not None
        and ablation_loss >= contract.loss_fraction
        and rescue_fraction is not None
        and rescue_fraction >= contract.rescue_fraction
        and float(rescue_advantage["mean"] or 0.0) >= contract.control_advantage
        and positive(rescue_advantage)
        and float(opposite_rescue4["mean"] or 0.0) <= -contract.screen_generation16
        and opposite_rescue4["ci"][1] is not None
        and float(opposite_rescue4["ci"][1]) < 0.0
        and float(opposite_founder8["mean"] or 0.0) <= -contract.screen_generation16
        and opposite_founder8["ci"][1] is not None
        and float(opposite_founder8["ci"][1]) < 0.0
        and float(corruption8["mean"] or 0.0) >= contract.screen_generation16
        and positive(corruption8)
    )
    return {
        "verdict": (
            "STRICT_RENEWED_CA_PLASTIC_HEREDITY"
            if renewed
            else "NO_STRICT_RENEWED_CA_PLASTIC_HEREDITY"
        ),
        "renewed_gate": renewed,
        "alpha": alpha,
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
        "active_rewrite_advantage_generation8": active_advantage,
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
    }


def _decoder_gate(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    observer: str,
    profile: RepairProfile,
    contract: RepairContract,
    alpha: float,
) -> dict[str, Any]:
    metric = "balanced_accuracy"

    def boot(values: Sequence[float], name: str) -> dict[str, Any]:
        return _bootstrap(
            values,
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "decoder", candidate_id, observer, name),
            alpha,
        )

    intact = boot(
        _qualification_metric(rows, candidate_id, "intact", 16, observer, metric),
        "intact",
    )
    no_rewrite = boot(
        _qualification_metric(rows, candidate_id, "no_rewrite", 16, observer, metric),
        "no-rewrite",
    )
    read_disabled = boot(
        _qualification_metric(rows, candidate_id, "read_disabled", 16, observer, metric),
        "read-disabled",
    )
    ablated = boot(
        _qualification_metric(rows, candidate_id, "ablate_after_g2", 16, observer, metric),
        "ablated",
    )
    rescued = boot(
        _qualification_metric(
            rows, candidate_id, "rescue_same_enter_g4", 16, observer, metric
        ),
        "rescued",
    )
    opposite = boot(
        _qualification_metric(
            rows, candidate_id, "rescue_opposite_enter_g4", 16, observer, metric
        ),
        "opposite",
    )
    stale_advantage = boot(
        _qualification_difference(
            rows,
            candidate_id,
            "intact",
            "no_rewrite",
            16,
            observer=observer,
            metric=metric,
        ),
        "stale-advantage",
    )
    read_advantage = boot(
        _qualification_difference(
            rows,
            candidate_id,
            "intact",
            "read_disabled",
            16,
            observer=observer,
            metric=metric,
        ),
        "read-advantage",
    )
    intact_excess = float(intact["mean"] or 0.5) - 0.5
    restoration = (
        (float(rescued["mean"] or 0.5) - 0.5) / intact_excess
        if intact_excess > 0.0
        else None
    )
    passed = bool(
        float(intact["mean"] or 0.0) >= contract.decoder_mean_gate
        and intact["ci"][0] is not None
        and float(intact["ci"][0]) > contract.decoder_lower_gate
        and float(stale_advantage["mean"] or 0.0) >= contract.decoder_advantage
        and stale_advantage["ci"][0] is not None
        and float(stale_advantage["ci"][0]) > 0.0
        and float(read_advantage["mean"] or 0.0) >= contract.decoder_advantage
        and read_advantage["ci"][0] is not None
        and float(read_advantage["ci"][0]) > 0.0
        and float(ablated["mean"] or 1.0) <= contract.decoder_null_ceiling
        and restoration is not None
        and restoration >= contract.rescue_fraction
        and float(opposite["mean"] or 1.0) <= 1.0 - contract.decoder_lower_gate
    )
    return {
        "passed": passed,
        "intact_generation16": intact,
        "no_rewrite_generation16": no_rewrite,
        "read_disabled_generation16": read_disabled,
        "ablation_generation16": ablated,
        "same_rescue_generation16": rescued,
        "opposite_rescue_generation16": opposite,
        "active_advantage_over_no_rewrite": stale_advantage,
        "read_advantage": read_advantage,
        "rescue_restoration_fraction": restoration,
    }


def adjudicate_confirmation(
    rows: Sequence[dict[str, Any]],
    candidate_models: Sequence[dict[str, Any]],
    profile: RepairProfile,
    contract: RepairContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"state": "incomplete", "verdict": "INCOMPLETE", "candidates": {}}
    candidates: dict[str, Any] = {}
    strict_any = False
    expressed_any = False
    cryptic_any = False
    for model in candidate_models:
        candidate_id = str(model["candidate_id"])
        strict = _strict_confirmation_gate(
            rows,
            candidate_id,
            profile,
            contract,
            contract.confirmation_alpha_per_class,
        )
        carrier = _decoder_gate(
            rows,
            candidate_id,
            "carrier_decoder",
            profile,
            contract,
            contract.confirmation_alpha_per_class,
        )
        phenotype = _decoder_gate(
            rows,
            candidate_id,
            "phenotype_decoder",
            profile,
            contract,
            contract.confirmation_alpha_per_class,
        )
        if strict["renewed_gate"]:
            secondary = "STRICT_FORM_PRIMARY"
        elif carrier["passed"] and phenotype["passed"]:
            secondary = "EXPRESSED_DRIFTED_LINEAGE_HEREDITY"
        elif carrier["passed"]:
            secondary = "CRYPTIC_RENEWED_CARRIER_MEMORY"
        else:
            secondary = "NO_DURABLE_RENEWAL"
        strict_any |= bool(strict["renewed_gate"])
        expressed_any |= secondary == "EXPRESSED_DRIFTED_LINEAGE_HEREDITY"
        cryptic_any |= secondary == "CRYPTIC_RENEWED_CARRIER_MEMORY"
        candidates[candidate_id] = {
            "model": {
                key: value for key, value in model.items() if not isinstance(value, np.ndarray)
            },
            "strict": strict,
            "carrier_decoder": carrier,
            "phenotype_decoder": phenotype,
            "secondary_verdict": secondary,
        }
    verdict = (
        "STRICT_RENEWED_CA_PLASTIC_HEREDITY"
        if strict_any
        else "EXPRESSED_DRIFTED_LINEAGE_HEREDITY"
        if expressed_any
        else "CRYPTIC_RENEWED_CARRIER_MEMORY"
        if cryptic_any
        else "NO_DURABLE_RENEWAL"
    )
    return {
        "state": "complete",
        "verdict": verdict,
        "strict_primary_passed": strict_any,
        "candidates": candidates,
        "claim_boundary": "synthetic CA lineage memory only; no metabolism, agency, or biological-life claim",
    }


def _stage3_expected_outcomes(frozen: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    checkpoint_root = frozen["root"] / "lineages/checkpoints"
    for path in sorted(checkpoint_root.glob("*.json")):
        payload = _load_json(path)["result"]
        result[str(payload["pair_id"])] = payload["conditions"]["intact"]["outcomes"]
    return result


def _configuration_payload(configuration: ReaderConfiguration) -> dict[str, Any]:
    return {
        key: value
        for key, value in configuration.to_dict().items()
        if key != "configuration_id"
    }


def _json_model(model: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in model.items() if not isinstance(value, np.ndarray)}


def _queue(
    design_digest: str,
    state: str,
    *,
    eligible_candidate_ids: Sequence[str] = (),
    final_verdict: str | None = None,
) -> dict[str, Any]:
    stages: list[dict[str, Any]] = [
        {"stage": 1, "name": "motif_carrier_upper_bound", "state": "complete"},
        {"stage": 2, "name": "freeze_and_generalize_reader", "state": "complete"},
        {
            "stage": 3,
            "name": "renewed_heredity_causal_ladder",
            "state": "complete_negative",
        },
        {
            "stage": "3R",
            "name": "semantic_closure_and_repair",
            "state": state,
            "eligible_candidate_ids": list(eligible_candidate_ids),
        },
        {
            "stage": 4,
            "name": "compression_and_robustness",
            "state": "blocked_pending_strict_stage3r_confirmation",
        },
        {
            "stage": 5,
            "name": "localize_inheritance",
            "state": "blocked_pending_stage4_review",
        },
    ]
    if final_verdict:
        stages[3]["verdict"] = final_verdict
        if final_verdict == "STRICT_RENEWED_CA_PLASTIC_HEREDITY":
            stages[4]["state"] = "blocked_pending_human_review"
        else:
            stages[4]["state"] = "blocked_stage3r_strict_gate_failed"
    return {
        "programme": "ca_motif_lineage_five_stage_with_stage3r",
        "stage3r_design_digest": design_digest,
        "automatic_chaining": False,
        "confirmation_requires_explicit_authorization": True,
        "per_invocation_max_hours": 8.0,
        "stages": stages,
    }


def _render_preconfirmation_report(results: dict[str, Any]) -> str:
    diagnostic = results["diagnostic"]
    decision = results["selection_decision"]
    lines = [
        "# CA motif-lineage Stage 3R preconfirmation",
        "",
        f"State: **{results['state']}**. Profile: `{results['profile']}`.",
        f"Elapsed: `{results['elapsed_seconds'] / 3600.0:.3f}` wall hours.",
        "",
        "## Semantic-drift diagnosis",
        "",
        f"- Exact Stage-3 baseline replay: `{diagnostic.get('baseline_reproduced_all_pairs')}`.",
        f"- Retained strict windows: `{diagnostic.get('retained_strict_window_ids')}`.",
        f"- Retained overlap window: `{diagnostic.get('retained_overlap_window_id')}`.",
        "",
        "## Selection",
        "",
        f"- Screen winners: `{decision.get('screen_selected_candidate_ids', [])}`.",
        f"- Causally qualified candidates: `{decision.get('eligible_confirmation_candidate_ids', [])}`.",
        f"- Confirmation state: `{decision.get('confirmation_state')}`.",
        "",
        "The 96-pair confirmation cohort has not been simulated. Confirmation requires a separate explicit command after review.",
    ]
    return "\n".join(lines) + "\n"


def _render_preconfirmation_lay(results: dict[str, Any]) -> str:
    eligible = results["selection_decision"].get(
        "eligible_confirmation_candidate_ids", []
    )
    if eligible:
        finding = (
            "One or more repairs survived both the fresh selection screen and the full causal checks. "
            "They are candidates, not confirmed results."
        )
    else:
        finding = (
            "None of the tested repairs was strong enough to justify opening the final confirmation cohort."
        )
    return (
        "# Lay summary\n\n"
        f"{finding}\n\n"
        "This round first measured how the daughter changes the inherited texture recipe, then tested simple rescaling and a universal error-corrector. "
        "The error-corrector never sees which ancestral form is desired; it only learns a common copying grammar from already exposed lineages.\n\n"
        "The final 96 pairs remain untouched until a person reviews these selection results and explicitly authorizes confirmation.\n"
    )


def _render_final_report(results: dict[str, Any]) -> str:
    adjudication = results["adjudication"]
    lines = [
        "# CA motif-lineage Stage 3R confirmation",
        "",
        f"State: **{results['state']}**. Verdict: **{adjudication['verdict']}**.",
        f"Elapsed: `{results['elapsed_seconds'] / 3600.0:.3f}` wall hours.",
        "",
        "## Candidate mechanisms",
        "",
    ]
    for candidate_id, value in adjudication["candidates"].items():
        lines.append(
            f"- `{candidate_id}`: strict `{value['strict']['verdict']}`; secondary `{value['secondary_verdict']}`; "
            f"generation-16 fixed form `{value['strict']['intact_generation16']['mean']}`."
        )
    lines.extend(
        (
            "",
            "## Evidence boundary",
            "",
            "This is a synthetic CA carrier test. It makes no claim of metabolism, agency, biological life, or memory outside the automaton's total state.",
        )
    )
    return "\n".join(lines) + "\n"


def _render_final_lay(results: dict[str, Any]) -> str:
    verdict = results["adjudication"]["verdict"]
    explanations = {
        "STRICT_RENEWED_CA_PLASTIC_HEREDITY": "A universal daughter-side repair closed the copying loop and passed the complete 16-generation causal test.",
        "EXPRESSED_DRIFTED_LINEAGE_HEREDITY": "The original form drifted, but a causally renewed and visibly expressed lineage identity remained.",
        "CRYPTIC_RENEWED_CARRIER_MEMORY": "The carrier retained a renewed lineage identity, but the visible cellular form no longer expressed it reliably.",
        "NO_DURABLE_RENEWAL": "Neither ancestral form nor a wider causal lineage identity survived the confirmation test.",
    }
    return (
        "# Lay summary\n\n"
        f"{explanations.get(verdict, verdict)}\n\n"
        "Every confirmation daughter began from the same erased visible board. Correct rescue, opposite rescue, stopped rewriting, and carrier ablation determine whether any surviving pattern is genuinely carried and renewed.\n"
    )


def _update_discovery_log(
    state: str,
    verdict: str,
    elapsed_seconds: float,
) -> None:
    path = ROOT / "DISCOVERY_LOG_EIDOSOMA_SCIENTIST.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Discovery log\n"
    start = "<!-- ca-motif-lineage-stage-3r:start -->"
    end = "<!-- ca-motif-lineage-stage-3r:end -->"
    section = "\n".join(
        (
            start,
            "## CA motif-lineage Stage 3R",
            "",
            f"State: `{state}`; verdict: `{verdict}`.",
            f"Elapsed `{elapsed_seconds / 3600.0:.3f}` wall hours.",
            "See `results/ca-motif-lineage-stage-3r/REPORT.md` and `LAY_SUMMARY.md`.",
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


def _selected_models(
    models: Sequence[dict[str, Any]], candidate_ids: Sequence[str]
) -> list[dict[str, Any]]:
    by_id = {str(model["candidate_id"]): model for model in models}
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in by_id]
    if missing:
        raise ValueError(f"selected repair models are missing: {missing}")
    return [by_id[candidate_id] for candidate_id in candidate_ids]


def run_motif_repair(
    output: Path,
    *,
    stage3_root: Path = DEFAULT_STAGE3_ROOT,
    stage2_root: Path = DEFAULT_STAGE2_ROOT,
    stage1_root: Path = DEFAULT_STAGE1_ROOT,
    profile_name: str = "reference",
    workers: int = 20,
    max_hours: float = 8.0,
    resume: bool = False,
    phases: Sequence[str] | None = None,
    authorize_confirmation: bool = False,
) -> dict[str, Any]:
    require_pinned_numpy()
    if profile_name not in PUBLIC_PROFILES:
        raise ValueError(f"unknown Stage-3R profile {profile_name!r}")
    if max_hours <= 0.0 or max_hours > 8.0:
        raise ValueError("Stage-3R max-hours must be in (0, 8]")
    selected_phases = tuple(phases or DEFAULT_PRECONFIRMATION_PHASES)
    unknown = [phase for phase in selected_phases if phase not in PHASES]
    if unknown:
        raise ValueError(f"unknown Stage-3R phases: {unknown}")
    if "confirm" in selected_phases:
        if selected_phases != ("confirm",):
            raise ValueError("confirmation must be a separate invocation")
        if not authorize_confirmation:
            raise ValueError("confirmation requires explicit authorization")
        if not resume:
            raise ValueError("confirmation requires --resume against reviewed artifacts")
    elif authorize_confirmation:
        raise ValueError("confirmation authorization is valid only for the confirm phase")
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hard_deadline = started + max_hours * 3600.0
    contract = RepairContract()
    writer_contract = MotifContract()
    profile = REPAIR_PROFILES[profile_name]
    reserve = min(
        contract.science_reserve_seconds,
        max(60.0, max_hours * 3600.0 * 0.10),
    )
    science_deadline = max(started, hard_deadline - reserve)

    def status(state: str, phase: str, **extra: Any) -> None:
        now = time.time()
        payload = {
            "state": state,
            "stage": "3R-semantic-closure-repair",
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
        frozen = load_frozen_stage3(stage3_root, stage2_root, stage1_root)
        cohorts = select_repair_cohorts(profile, frozen, contract)
        configuration = frozen["configuration"]
        input_paths = [
            PROTOCOL_PATH,
            *frozen["paths"].values(),
            frozen["stage2"]["stage1"]["paths"]["calibration"],
        ]
        design_payload = {
            "experiment": "ca_motif_lineage_stage_3r",
            "contract": contract.to_dict(),
            "writer_contract_digest": writer_contract.digest,
            "profile_name": profile_name,
            "profile": asdict(profile),
            "configuration": configuration.to_dict(),
            "stage3_design_digest": frozen["design_digest"],
            "stage2_design_digest": frozen["stage2"]["design_digest"],
            "phases_contract": PHASES,
            "confirmation_separate_invocation": True,
            "windows": WINDOWS,
            "simple_kinds": SIMPLE_KINDS,
            "learned_kinds": LEARNED_KINDS,
            "ridge_grid": RIDGES,
            "rank_grid": RANKS,
            "diagnostic_pair_ids": [pair["pair_id"] for pair in cohorts["diagnostic"]],
            "selection_pair_ids": [pair["pair_id"] for pair in cohorts["selection"]],
            "confirmation_pair_ids": [pair["pair_id"] for pair in cohorts["confirmation"]],
            "prior_pair_ids_excluded": len(frozen["used_pair_ids"]),
            "input_sha256": {
                str(path.relative_to(ROOT)): _sha256(path) for path in input_paths
            },
            "implementation_sha256": {
                "motif_repair.py": _sha256(Path(__file__)),
                "motif_lineage_stage3.py": _sha256(
                    Path(__file__).with_name("motif_lineage_stage3.py")
                ),
                "motif_lineage.py": _sha256(Path(__file__).with_name("motif_lineage.py")),
            },
            "cleanroom_exclusion": "no Wagner or Fable implementation source is read, imported, hashed, or executed",
            "retuning_on_confirmation": False,
        }
        design_digest = hashlib.sha256(
            json.dumps(design_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        design = {**design_payload, "design_digest": design_digest}
        design_path = output / "DESIGN.json"
        if resume and design_path.exists():
            if _load_json(design_path).get("design_digest") != design_digest:
                raise ValueError("Stage-3R resume design digest mismatch")
        elif "confirm" in selected_phases:
            raise FileNotFoundError("confirmation requires a reviewed Stage-3R design")
        _atomic_json(design_path, design)
        _atomic_json(
            output / "MANIFEST.json",
            {
                "experiment": "ca_motif_lineage_stage_3r",
                "profile": profile_name,
                "design_digest": design_digest,
                "contract_digest": contract.digest,
                "workers": workers,
                "max_hours": max_hours,
                "invocation_phases": selected_phases,
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
                "diagnostic_pair_ids": [pair["pair_id"] for pair in cohorts["diagnostic"]],
                "selection_pair_ids": [pair["pair_id"] for pair in cohorts["selection"]],
                "confirmation_pair_ids": [pair["pair_id"] for pair in cohorts["confirmation"]],
                "confirmation_trajectory_state": "untouched",
            },
        )
        if not (output / "QUEUE.json").exists() or "confirm" not in selected_phases:
            _atomic_json(output / "QUEUE.json", _queue(design_digest, "running"))

        configuration_payload = _configuration_payload(configuration)
        expected = _stage3_expected_outcomes(frozen)
        if "diagnose" in selected_phases:
            items = [
                {
                    "checkpoint": f"diagnostic-{index:04d}",
                    "pair": pair,
                    "replicates": profile.diagnostic_replicates,
                    "generations": profile.diagnostic_generations,
                    "configuration": configuration_payload,
                    "expected_outcomes": expected.get(pair["pair_id"]),
                }
                for index, pair in enumerate(cohorts["diagnostic"])
            ]
            status("running", "diagnose", completed=0, total=len(items))
            diagnostic_rows, diagnostic_complete = _run_diagnostic_checkpoints(
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
            diagnostic = summarize_diagnostics(
                diagnostic_rows, profile, contract, diagnostic_complete
            )
            diagnostic["design_digest"] = design_digest
            _atomic_json(output / "DIAGNOSTIC.json", diagnostic)
            if not diagnostic_complete:
                status("partial_budget_exhausted", "diagnose")
                return {"state": "partial_budget_exhausted", "phase": "diagnose"}
        else:
            diagnostic = _load_json(output / "DIAGNOSTIC.json")
            if diagnostic.get("design_digest") != design_digest:
                raise ValueError("diagnostic design digest mismatch")

        if "fit" in selected_phases:
            status("running", "fit")
            learned, fit_audit, norm_target = fit_learned_models(
                output / "diagnostics", diagnostic["retained_window_ids"]
            )
            simple = build_simple_models(
                diagnostic["retained_window_ids"], norm_target
            )
            models = [*simple, *learned]
            fit_audit.update(
                {
                    "design_digest": design_digest,
                    "diagnostic_digest": diagnostic["diagnostic_digest"],
                    "norm_target": norm_target,
                    "model_count": len(models),
                }
            )
            _atomic_json(output / "FIT_AUDIT.json", fit_audit)
            save_repair_models(
                output,
                models,
                design_digest=design_digest,
                diagnostic_digest=diagnostic["diagnostic_digest"],
            )
        else:
            models = load_repair_models(output, design_digest)

        if "screen" in selected_phases:
            items = [
                {
                    "checkpoint": f"screen-{index:04d}",
                    "pair": pair,
                    "replicates": profile.selection_replicates,
                    "generations": profile.selection_generations,
                    "configuration": configuration_payload,
                }
                for index, pair in enumerate(cohorts["selection"])
            ]
            status("running", "screen", completed=0, total=len(items))
            screen_rows, screen_complete = _run_json_checkpoints(
                output,
                "screen",
                items,
                models,
                _screen_pair_task,
                writer_contract,
                contract,
                frozen["reference"],
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
            )
            screen = adjudicate_screen(
                screen_rows, models, profile, contract, screen_complete
            )
            screen["design_digest"] = design_digest
            _atomic_json(output / "SCREEN.json", screen)
            if not screen_complete:
                status("partial_budget_exhausted", "screen")
                return {"state": "partial_budget_exhausted", "phase": "screen"}
        else:
            screen = _load_json(output / "SCREEN.json")
            if screen.get("design_digest") != design_digest:
                raise ValueError("screen design digest mismatch")

        selected_ids = list(screen["selected_candidate_ids"])
        qualification_models = _selected_models(models, selected_ids)
        if "qualify" in selected_phases:
            if qualification_models:
                items = [
                    {
                        "checkpoint": f"qualify-{index:04d}",
                        "pair": pair,
                        "replicates": profile.selection_replicates,
                        "generations": profile.selection_generations,
                        "configuration": configuration_payload,
                    }
                    for index, pair in enumerate(cohorts["selection"])
                ]
                status("running", "qualify", completed=0, total=len(items))
                qualification_rows, qualification_complete = _run_json_checkpoints(
                    output,
                    "qualification",
                    items,
                    qualification_models,
                    _qualification_pair_task,
                    writer_contract,
                    contract,
                    frozen["reference"],
                    design_digest,
                    workers=workers,
                    resume=resume,
                    deadline=science_deadline,
                    status=status,
                )
            else:
                qualification_rows = []
                qualification_complete = True
                (output / "qualification").mkdir(parents=True, exist_ok=True)
                _atomic_text(output / "qualification/COMPLETE", "no eligible screen candidates\n")
            qualification = adjudicate_qualification(
                qualification_rows,
                selected_ids,
                profile,
                contract,
                qualification_complete,
            )
            qualification["design_digest"] = design_digest
            _atomic_json(output / "QUALIFICATION.json", qualification)
            if not qualification_complete:
                status("partial_budget_exhausted", "qualify")
                return {"state": "partial_budget_exhausted", "phase": "qualify"}
        else:
            qualification = _load_json(output / "QUALIFICATION.json")
            if qualification.get("design_digest") != design_digest:
                raise ValueError("qualification design digest mismatch")

        eligible_ids = list(qualification["qualified_candidate_ids"])
        confirmation_models = _selected_models(models, eligible_ids)

        if "confirm" in selected_phases:
            selection_decision = _load_json(output / "SELECTION_DECISION.json")
            if selection_decision.get("design_digest") != design_digest:
                raise ValueError("selection-decision design mismatch")
            if selection_decision.get("confirmation_state") != "awaiting_human_review":
                raise ValueError("confirmation is not authorized by the frozen selection decision")
            frozen_ids = selection_decision["eligible_confirmation_candidate_ids"]
            if eligible_ids != frozen_ids:
                raise ValueError("confirmation candidates changed after review")
            if not confirmation_models:
                raise ValueError("there are no eligible confirmation candidates")
            confirmation_design = _load_json(output / "CONFIRMATION_DESIGN.json")
            if confirmation_design.get("model_sha256") != _sha256(
                output / "REPAIR_MODELS.npz"
            ):
                raise ValueError("confirmation repair-model hash mismatch")
            items = [
                {
                    "checkpoint": f"confirm-{index:04d}",
                    "pair": pair,
                    "replicates": profile.confirmation_replicates,
                    "generations": profile.confirmation_generations,
                    "configuration": configuration_payload,
                }
                for index, pair in enumerate(cohorts["confirmation"])
            ]
            status("running", "confirm", completed=0, total=len(items))
            confirmation_rows, confirmation_complete = _run_json_checkpoints(
                output,
                "confirmation",
                items,
                confirmation_models,
                _qualification_pair_task,
                writer_contract,
                contract,
                frozen["reference"],
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
            )
            adjudication = adjudicate_confirmation(
                confirmation_rows,
                confirmation_models,
                profile,
                contract,
                confirmation_complete,
            )
            state = "complete" if confirmation_complete else "partial_budget_exhausted"
            results = {
                "experiment": "ca_motif_lineage_stage_3r",
                "state": state,
                "profile": profile_name,
                "design_digest": design_digest,
                "stage3_design_digest": frozen["design_digest"],
                "elapsed_seconds": time.time() - started,
                "adjudication": adjudication,
            }
            _atomic_json(output / "RESULTS.json", results)
            _atomic_text(output / "REPORT.md", _render_final_report(results))
            _atomic_text(output / "LAY_SUMMARY.md", _render_final_lay(results))
            _atomic_json(
                output / "STAGE_DECISION.json",
                {
                    "design_digest": design_digest,
                    "verdict": adjudication["verdict"],
                    "decision": (
                        "advance_to_stage_4_after_review"
                        if adjudication.get("strict_primary_passed")
                        else "halt_and_replan_carrier_alphabet"
                    ),
                    "automatic_launch": False,
                    "review_required": True,
                },
            )
            _atomic_json(
                output / "QUEUE.json",
                _queue(
                    design_digest,
                    state,
                    eligible_candidate_ids=eligible_ids,
                    final_verdict=adjudication["verdict"],
                ),
            )
            if confirmation_complete:
                _atomic_text(output / "COMPLETE", "complete\n")
                _update_discovery_log(
                    state,
                    adjudication["verdict"],
                    results["elapsed_seconds"],
                )
                status("complete", "campaign", verdict=adjudication["verdict"])
            else:
                status("partial_budget_exhausted", "confirm")
            return results

        if "adjudicate" not in selected_phases:
            state = "phases_complete"
            status(state, "campaign")
            return {"state": state, "completed_phases": selected_phases}

        confirmation_state = (
            "awaiting_human_review" if eligible_ids else "not_opened_no_qualified_candidate"
        )
        selection_decision = {
            "design_digest": design_digest,
            "stage3_design_digest": frozen["design_digest"],
            "screen_selected_candidate_ids": selected_ids,
            "eligible_confirmation_candidate_ids": eligible_ids,
            "selected_overlap_candidate_id": screen.get(
                "selected_overlap_candidate_id"
            ),
            "confirmation_state": confirmation_state,
            "confirmation_requires_separate_invocation": True,
            "automatic_launch": False,
            "review_required": bool(eligible_ids),
        }
        _atomic_json(output / "SELECTION_DECISION.json", selection_decision)
        confirmation_design = {
            "design_digest": design_digest,
            "protocol_sha256": _sha256(PROTOCOL_PATH),
            "model_sha256": _sha256(output / "REPAIR_MODELS.npz"),
            "eligible_candidate_ids": eligible_ids,
            "eligible_models": [_json_model(model) for model in confirmation_models],
            "confirmation_pair_ids": [
                pair["pair_id"] for pair in cohorts["confirmation"]
            ],
            "replicates": profile.confirmation_replicates,
            "generations": profile.confirmation_generations,
            "alpha_per_class": contract.confirmation_alpha_per_class,
            "trajectory_state": "untouched",
            "authorization_required": True,
        }
        confirmation_design["confirmation_design_digest"] = hashlib.sha256(
            json.dumps(
                confirmation_design, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        _atomic_json(output / "CONFIRMATION_DESIGN.json", confirmation_design)
        state = (
            "awaiting_human_review"
            if eligible_ids
            else "no_candidate_for_confirmation"
        )
        results = {
            "experiment": "ca_motif_lineage_stage_3r_preconfirmation",
            "state": state,
            "profile": profile_name,
            "design_digest": design_digest,
            "stage3_design_digest": frozen["design_digest"],
            "elapsed_seconds": time.time() - started,
            "diagnostic": diagnostic,
            "screen": screen,
            "qualification": qualification,
            "selection_decision": selection_decision,
        }
        _atomic_json(output / "PRECONFIRMATION_RESULTS.json", results)
        _atomic_text(
            output / "PRECONFIRMATION_REPORT.md",
            _render_preconfirmation_report(results),
        )
        _atomic_text(output / "REPORT.md", _render_preconfirmation_report(results))
        _atomic_text(
            output / "PRECONFIRMATION_LAY_SUMMARY.md",
            _render_preconfirmation_lay(results),
        )
        _atomic_text(output / "LAY_SUMMARY.md", _render_preconfirmation_lay(results))
        _atomic_json(
            output / "QUEUE.json",
            _queue(
                design_digest,
                state,
                eligible_candidate_ids=eligible_ids,
            ),
        )
        _atomic_text(output / "PRECONFIRMATION_COMPLETE", "complete\n")
        _update_discovery_log(state, state.upper(), results["elapsed_seconds"])
        status(state, "campaign", eligible_candidates=len(eligible_ids))
        return results
    except BaseException as error:
        status("failed", "campaign", error=repr(error))
        raise


__all__ = [
    "DEFAULT_PRECONFIRMATION_PHASES",
    "PHASES",
    "PUBLIC_PROFILES",
    "REPAIR_PROFILES",
    "RepairContract",
    "RepairProfile",
    "WINDOWS",
    "adjudicate_confirmation",
    "adjudicate_screen",
    "apply_repair",
    "build_simple_models",
    "carrier_transition_metrics",
    "cross_validate_repair",
    "fit_learned_models",
    "heldout_lineage_accuracy",
    "load_frozen_stage3",
    "load_repair_models",
    "run_motif_repair",
    "select_repair_cohorts",
    "simulate_repair_lineage",
]
