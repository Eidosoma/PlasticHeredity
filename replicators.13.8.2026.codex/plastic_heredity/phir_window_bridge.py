"""Fresh 24-matrix bridge between pooled and rolling Phi-r estimators.

This module is intentionally separate from :mod:`plastic_heredity.phir_ch5`.
It cannot authorize or launch the already sealed 48-matrix Chapter 5
confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from .config import CANDIDATES, GardConfig
from .intervention_core import FrozenFullPredictor, _records_digest, edited_snapshot
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .mechanistic_metrics import holm_adjust
from .phir_ch5 import (
    AUTHORIZATION as CH5_CONFIRMATION_AUTHORIZATION,
    DEFAULT_CONFIRMATION as CH5_CONFIRMATION,
    DEFAULT_CONFIRMATION_WORK as CH5_CONFIRMATION_WORK,
    DEFAULT_PILOT as CH5_PILOT,
    DEFAULT_REGISTRATION as CH5_REGISTRATION,
    EXPECTED_MODEL_SHA256,
    BufferState,
    _append_ledger,
    _append_observation,
    _buffer_state,
    _json_ready,
    _restore_buffer,
    _select_controller_edit,
    _snapshot_after_record,
    verify_registration as verify_ch5_registration,
    verify_result as verify_ch5_result,
)
from .phir_instruments import (
    ATOM_NAMES,
    _active_zscore,
    _canonical_array_digest,
    close_clr_drop_last,
    fiedler_bipartition,
    lagged_gaussian_mi_graph,
    revised_phi_from_partition,
    score_phi_window,
    typeset_whole_minus_parts,
    advance_fission_traced,
    records_equal,
    rng_states_equal,
)
from .seeds import derive_seed
from .simulator import (
    FissionRecord,
    SimulationError,
    Snapshot,
    advance_fission,
    generate_beta,
    generate_initial_composition,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCUMENT = "CODEX_CH5_PHIR_WINDOW_BRIDGE_PREREGISTRATION.md"
LEDGER = "PHIR_RESULTS_LEDGER.md"

DEFAULT_VALIDATION = RESULTS / "phir_window_bridge_validation"
DEFAULT_REGISTRATION = RESULTS / "phir_window_bridge_registration"
DEFAULT_SMOKE = RESULTS / "phir_window_bridge_smoke"
DEFAULT_OUTPUT = RESULTS / "phir_window_bridge24"
DEFAULT_WORK = RESULTS / ".phir_window_bridge24_work"
DEFAULT_LOG = RESULTS / "phir_window_bridge24.log"

LABEL = "CODEX_CH5_PHIR_WINDOW_BRIDGE_V1"
PROGRAM_FORMAT = "codex-ch5-phir-window-bridge-program-v1"
REGISTRATION_FORMAT = "codex-ch5-phir-window-bridge-registration-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-window-bridge-checkpoint-v1"
RESULT_FORMAT = "codex-ch5-phir-window-bridge-result-v1"
STATUS_FORMAT = "codex-ch5-phir-window-bridge-status-v1"

MATRICES = 24
REPLICATES = 2
NATURAL_GENERATIONS = 60
BRIDGE_HORIZON = 60
POOLED20_START = 41
POOLED30_START = 31
ROLLING_WINDOW = 512
FULL_TYPESET_ROLLING_STEPS = (40, 60)
BOOTSTRAP_REPETITIONS = 4096
RANDOMIZATION_REPETITIONS = 4096
MINIMUM_FREE_DISK_BYTES = 1_000_000_000
ARMS = ("MODEL_STABILIZE", "MODEL_DESTABILIZE")
PREPROCESSINGS = ("clr", "raw_count")

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/phir_window_bridge.py",
    "tests/test_phir_window_bridge.py",
    "plastic_heredity/phir_instruments.py",
    "plastic_heredity/phir_ch5.py",
    "plastic_heredity/config.py",
    "plastic_heredity/features.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/mechanistic.py",
    "plastic_heredity/mechanistic_metrics.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/simulator.py",
)


def _seed_value(name: str) -> str:
    return hashlib.sha256(f"{LABEL}::{name}".encode("utf-8")).hexdigest()


SEED_DOMAINS = {
    name: _seed_value(name)
    for name in (
        "matrix",
        "initial",
        "main_path",
        "future",
        "controller_action",
        "bootstrap",
        "randomization",
        "validation",
        "smoke",
    )
}


@dataclass(frozen=True)
class RunSpec:
    label: str
    matrices: int
    replicates: int
    natural_generations: int
    bridge_horizon: int
    pooled20_start: int
    pooled30_start: int
    rolling_window: int
    bootstrap_repetitions: int
    randomization_repetitions: int


@dataclass(frozen=True)
class InstrumentScore:
    revised: float
    full_typeset: float
    macro_typeset: float
    normalized_full: float
    causation: float
    emergence: float
    synergy: float
    atoms: NDArray[np.float64]
    active_coordinates: tuple[int, ...]
    partition_a: tuple[int, ...]
    partition_b: tuple[int, ...]
    observations: int
    transitions: int
    digest: str


@dataclass(frozen=True)
class NaturalLaunch:
    candidate: str
    replicate: int
    snapshot: Snapshot
    buffer: BufferState
    record_digest: str
    path_attempt: int


@dataclass(frozen=True)
class WindowBridgeBatch:
    matrix_id: int
    beta: NDArray[np.float64]
    initial_composition: NDArray[np.int16]
    lineage_rows: tuple[dict[str, Any], ...]
    window_rows: tuple[dict[str, Any], ...]
    selected_edit_rows: tuple[dict[str, Any], ...]
    scientific_digest: str


def scientific_spec() -> RunSpec:
    return RunSpec(
        label="window_bridge24",
        matrices=MATRICES,
        replicates=REPLICATES,
        natural_generations=NATURAL_GENERATIONS,
        bridge_horizon=BRIDGE_HORIZON,
        pooled20_start=POOLED20_START,
        pooled30_start=POOLED30_START,
        rolling_window=ROLLING_WINDOW,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        randomization_repetitions=RANDOMIZATION_REPETITIONS,
    )


def smoke_spec() -> RunSpec:
    return RunSpec(
        label="smoke",
        matrices=1,
        replicates=1,
        natural_generations=8,
        bridge_horizon=6,
        pooled20_start=5,
        pooled30_start=4,
        rolling_window=64,
        bootstrap_repetitions=32,
        randomization_repetitions=32,
    )


def _runtime_versions() -> dict[str, str]:
    return {
        "python": ".".join(str(value) for value in os.sys.version_info[:3]),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "threadpoolctl": importlib.metadata.version("threadpoolctl"),
    }


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def protocol() -> dict[str, Any]:
    ch5 = verify_ch5_result(CH5_PILOT)
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "program": "fresh 24-matrix pooled-versus-rolling Phi-r bridge",
        "completed_ch5_pilot_registration_id": ch5["registration_id"],
        "completed_ch5_pilot_manifest_sha256": sha256_file(CH5_PILOT / "manifest.json"),
        "original_confirmation_locked": bool(
            not CH5_CONFIRMATION.exists()
            and not CH5_CONFIRMATION_WORK.exists()
            and not CH5_CONFIRMATION_AUTHORIZATION.exists()
        ),
        "spec": asdict(scientific_spec()),
        "candidates": list(CANDIDATES),
        "arms": list(ARMS),
        "preprocessings": list(PREPROCESSINGS),
        "window_estimators": {
            "pooled20": "one score on controlled fissions 41-60",
            "rolling20": "mean rolling-512 score after fissions 41-60",
            "pooled30": "one score on controlled fissions 31-60",
            "rolling30": "mean rolling-512 score after fissions 31-60",
        },
        "typeset_taxonomy": {
            "full_typeset": "full coordinate blocks against whole future",
            "macro_typeset": "two averaged Fiedler halves",
            "full_typeset_rolling_steps": list(FULL_TYPESET_ROLLING_STEPS),
        },
        "frozen_model_sha256": EXPECTED_MODEL_SHA256,
        "seed_domains": SEED_DOMAINS,
        "inference": {
            "unit": "whole catalytic matrix",
            "candidate_pooling": False,
            "replicate_pooling": False,
            "bootstrap": BOOTSTRAP_REPETITIONS,
            "sign_randomization": RANDOMIZATION_REPETITIONS,
            "holm_within_four_cell_family": True,
        },
        "replay": "complete deterministic regeneration of all 24 matrices",
        "raw_molecular_trajectories_persisted": False,
        "cannot_authorize_or_launch_original_confirmation": True,
        "claim_boundary": [
            "temporal-estimator moderation is not a uniquely correct Phi-r definition",
            "gauge response is not hereditary control",
            "no consciousness, life, agency, or biological-memory claim",
            "no universal origin-of-life mechanism or Platonic-space portal",
        ],
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_ready(value), sort_keys=True, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=5)
    temporary.replace(path)


def _batch_digest(batch: WindowBridgeBatch) -> str:
    value = WindowBridgeBatch(
        matrix_id=batch.matrix_id,
        beta=batch.beta,
        initial_composition=batch.initial_composition,
        lineage_rows=batch.lineage_rows,
        window_rows=batch.window_rows,
        selected_edit_rows=batch.selected_edit_rows,
        scientific_digest="",
    )
    return _canonical_digest(_json_ready(asdict(value)))


def _nan_score(observations: int = 0) -> InstrumentScore:
    return InstrumentScore(
        revised=float("nan"),
        full_typeset=float("nan"),
        macro_typeset=float("nan"),
        normalized_full=float("nan"),
        causation=float("nan"),
        emergence=float("nan"),
        synergy=float("nan"),
        atoms=np.full(len(ATOM_NAMES), np.nan, dtype=np.float64),
        active_coordinates=(),
        partition_a=(),
        partition_b=(),
        observations=int(observations),
        transitions=max(0, int(observations) - 1),
        digest="",
    )


def score_counts(
    counts: NDArray,
    preprocessing: str,
    *,
    include_full_typeset: bool,
) -> InstrumentScore:
    raw = np.asarray(counts, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 3 or raw.shape[1] != GardConfig().n_types:
        raise ValueError("window counts must be observations by molecule types")
    if preprocessing == "clr":
        transformed = close_clr_drop_last(raw)
    elif preprocessing == "raw_count":
        transformed = np.asarray(raw.T, dtype=np.float64)
    else:
        raise ValueError(f"unknown preprocessing {preprocessing}")
    data, active = _active_zscore(transformed)
    # A coordinate can vary over the complete window yet be constant in only
    # the lagged past or future slice.  The sealed graph routine deliberately
    # converts the resulting undefined correlation to zero; suppress only the
    # corresponding NumPy warning, without changing that numeric path.
    with np.errstate(divide="ignore", invalid="ignore"):
        graph = lagged_gaussian_mi_graph(data)
    partition_a, partition_b = fiedler_bipartition(graph)
    revised, causation, emergence, synergy, atoms = revised_phi_from_partition(
        data, partition_a, partition_b
    )
    macro = np.vstack(
        (data[partition_a].mean(axis=0), data[partition_b].mean(axis=0))
    )
    macro_typeset, _ = typeset_whole_minus_parts(
        macro,
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
    )
    if include_full_typeset:
        full_typeset, whole = typeset_whole_minus_parts(
            data, partition_a, partition_b
        )
        normalized = full_typeset / whole if abs(whole) > 1e-12 else float("nan")
    else:
        full_typeset = normalized = float("nan")
    original_a = tuple(int(active[index]) for index in partition_a)
    original_b = tuple(int(active[index]) for index in partition_b)
    digest = _canonical_array_digest(
        raw, data, graph, partition_a, partition_b, atoms
    )
    return InstrumentScore(
        revised=float(revised),
        full_typeset=float(full_typeset),
        macro_typeset=float(macro_typeset),
        normalized_full=float(normalized),
        causation=float(causation),
        emergence=float(emergence),
        synergy=float(synergy),
        atoms=np.asarray(atoms, dtype=np.float64),
        active_coordinates=tuple(int(value) for value in active),
        partition_a=original_a,
        partition_b=original_b,
        observations=int(raw.shape[0]),
        transitions=int(raw.shape[0] - 1),
        digest=digest,
    )


def _safe_score(
    counts: NDArray,
    preprocessing: str,
    *,
    include_full_typeset: bool,
) -> InstrumentScore:
    try:
        return score_counts(
            counts, preprocessing, include_full_typeset=include_full_typeset
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return _nan_score(len(counts))


def _score_fields(prefix: str, score: InstrumentScore) -> dict[str, Any]:
    output: dict[str, Any] = {
        f"{prefix}_revised": score.revised,
        f"{prefix}_full_typeset": score.full_typeset,
        f"{prefix}_macro_typeset": score.macro_typeset,
        f"{prefix}_normalized_full": score.normalized_full,
        f"{prefix}_causation": score.causation,
        f"{prefix}_emergence": score.emergence,
        f"{prefix}_synergy": score.synergy,
        f"{prefix}_active_coordinates": list(score.active_coordinates),
        f"{prefix}_partition_a": list(score.partition_a),
        f"{prefix}_partition_b": list(score.partition_b),
        f"{prefix}_observations": score.observations,
        f"{prefix}_transitions": score.transitions,
        f"{prefix}_digest": score.digest,
    }
    for name, value in zip(ATOM_NAMES, score.atoms, strict=True):
        output[f"{prefix}_atom_{name}"] = float(value)
    return output


def partition_disagreement(
    first_a: Sequence[int],
    first_b: Sequence[int],
    second_a: Sequence[int],
    second_b: Sequence[int],
) -> float:
    first_a_set, first_b_set = set(first_a), set(first_b)
    second_a_set, second_b_set = set(second_a), set(second_b)
    common = (first_a_set | first_b_set) & (second_a_set | second_b_set)
    if len(common) < 2:
        return float("nan")
    direct = sum((value in first_a_set) != (value in second_a_set) for value in common)
    flipped = sum((value in first_a_set) != (value in second_b_set) for value in common)
    return float(min(direct, flipped) / len(common))


def _matrix_seed(spec: RunSpec, matrix_id: int, purpose: str) -> int:
    domain = SEED_DOMAINS["smoke"] if spec.label == "smoke" else SEED_DOMAINS[purpose]
    return derive_seed(domain, LABEL, spec.label, matrix_id)


def _future_seed(spec: RunSpec, candidate: str, matrix_id: int, replicate: int) -> int:
    domain = SEED_DOMAINS["smoke"] if spec.label == "smoke" else SEED_DOMAINS["future"]
    return derive_seed(domain, LABEL, spec.label, candidate, matrix_id, replicate)


def _natural_launch(
    matrix_id: int,
    beta: NDArray,
    initial: NDArray,
    candidate: str,
    replicate: int,
    spec: RunSpec,
) -> NaturalLaunch:
    config = GardConfig()
    domain = SEED_DOMAINS["smoke"] if spec.label == "smoke" else SEED_DOMAINS["main_path"]
    for attempt in range(100):
        rng = np.random.default_rng(
            derive_seed(
                domain,
                LABEL,
                spec.label,
                "main_path",
                candidate,
                matrix_id,
                replicate,
                attempt,
            )
        )
        snapshot = Snapshot(np.asarray(initial, dtype=np.int64).copy(), 0, (), ())
        observations: list[NDArray[np.int64]] = [snapshot.composition.copy()]
        kinds: list[int] = []
        daughters: list[NDArray[np.int64]] = []
        records: list[FissionRecord] = []
        try:
            for _ in range(spec.natural_generations):
                traced = advance_fission_traced(
                    snapshot.composition,
                    beta,
                    config,
                    CANDIDATES[candidate],
                    rng,
                )
                for composition in traced.growth_observations:
                    _append_observation(observations, kinds, composition, 0)
                _append_observation(observations, kinds, traced.record.daughter, 1)
                records.append(traced.record)
                snapshot = _snapshot_after_record(snapshot, traced.record)
                daughters.append(snapshot.composition.copy())
            return NaturalLaunch(
                candidate=candidate,
                replicate=replicate,
                snapshot=snapshot,
                buffer=_buffer_state(observations, kinds, daughters),
                record_digest=_records_digest(records),
                path_attempt=attempt,
            )
        except SimulationError:
            continue
    raise SimulationError(
        f"no complete natural launch c{candidate} m{matrix_id} r{replicate}"
    )


def _append_bridge_observation(
    rolling_observations: list[NDArray[np.int64]],
    rolling_kinds: list[int],
    bridge_observations: list[NDArray[np.int64]],
    bridge_kinds: list[int],
    composition: NDArray,
    kind: int,
) -> None:
    _append_observation(rolling_observations, rolling_kinds, composition, kind)
    _append_observation(bridge_observations, bridge_kinds, composition, kind)


def _run_arm(
    matrix_id: int,
    launch: NaturalLaunch,
    beta: NDArray,
    predictor: FrozenFullPredictor,
    spec: RunSpec,
    arm: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = GardConfig()
    rolling_observations, rolling_kinds, _ = _restore_buffer(launch.buffer)
    snapshot = launch.snapshot
    bridge_observations: list[NDArray[np.int64]] = [snapshot.composition.copy()]
    bridge_kinds: list[int] = []
    records: list[FissionRecord] = []
    inherited: list[int] = []
    window_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    boundary30: int | None = None
    boundary20: int | None = None
    rng = np.random.default_rng(
        _future_seed(spec, launch.candidate, matrix_id, launch.replicate)
    )
    action_domain = (
        SEED_DOMAINS["smoke"]
        if spec.label == "smoke"
        else SEED_DOMAINS["controller_action"]
    )
    action_rng = np.random.default_rng(
        derive_seed(
            action_domain,
            LABEL,
            spec.label,
            launch.candidate,
            matrix_id,
            launch.replicate,
        )
    )
    for step in range(1, spec.bridge_horizon + 1):
        try:
            traced = advance_fission_traced(
                snapshot.composition,
                beta,
                config,
                CANDIDATES[launch.candidate],
                rng,
            )
        except SimulationError:
            break
        for composition in traced.growth_observations:
            _append_bridge_observation(
                rolling_observations,
                rolling_kinds,
                bridge_observations,
                bridge_kinds,
                composition,
                0,
            )
        _append_bridge_observation(
            rolling_observations,
            rolling_kinds,
            bridge_observations,
            bridge_kinds,
            traced.record.daughter,
            1,
        )
        records.append(traced.record)
        inherited.append(int(traced.record.h > config.inheritance_threshold))
        snapshot = _snapshot_after_record(snapshot, traced.record)
        edit, risk_before, risk_after = _select_controller_edit(
            arm,
            predictor,
            launch.candidate,
            snapshot,
            beta,
            action_rng,
        )
        if edit is None:
            raise AssertionError("model controller unexpectedly selected no edit")
        snapshot = edited_snapshot(snapshot, edit)
        _append_bridge_observation(
            rolling_observations,
            rolling_kinds,
            bridge_observations,
            bridge_kinds,
            snapshot.composition,
            2,
        )
        edit_rows.append(
            {
                "matrix_id": matrix_id,
                "candidate": launch.candidate,
                "replicate": launch.replicate,
                "arm": arm,
                "step": step,
                "remove_type": edit.remove_type,
                "add_type": edit.add_type,
                "risk_before": risk_before,
                "risk_after": risk_after,
            }
        )
        if step == spec.pooled30_start - 1:
            boundary30 = len(bridge_observations) - 1
        if step == spec.pooled20_start - 1:
            boundary20 = len(bridge_observations) - 1
        if step >= spec.pooled30_start:
            counts = np.asarray(
                rolling_observations[-spec.rolling_window :], dtype=np.int64
            )
            row: dict[str, Any] = {
                "matrix_id": matrix_id,
                "candidate": launch.candidate,
                "replicate": launch.replicate,
                "arm": arm,
                "step": step,
            }
            include_full = step in FULL_TYPESET_ROLLING_STEPS
            for preprocessing in PREPROCESSINGS:
                score = _safe_score(
                    counts,
                    preprocessing,
                    include_full_typeset=include_full,
                )
                row.update(_score_fields(preprocessing, score))
            window_rows.append(row)

    complete = len(records) == spec.bridge_horizon
    if complete and boundary20 is not None and boundary30 is not None:
        pooled20_counts = np.asarray(bridge_observations[boundary20:], dtype=np.int64)
        pooled30_counts = np.asarray(bridge_observations[boundary30:], dtype=np.int64)
    else:
        pooled20_counts = pooled30_counts = np.empty(
            (0, config.n_types), dtype=np.int64
        )
    lineage: dict[str, Any] = {
        "matrix_id": matrix_id,
        "candidate": launch.candidate,
        "replicate": launch.replicate,
        "arm": arm,
        "completed_horizon": int(complete),
        "information_eligible": int(complete),
        "completed_fissions": len(records),
        "extinct": int(not complete),
        "inherited_31_60": float(
            sum(inherited[spec.pooled30_start - 1 :])
            / (spec.bridge_horizon - spec.pooled30_start + 1)
        ),
        "natural_record_digest": launch.record_digest,
        "controlled_record_digest": _records_digest(records),
        "controlled_observation_digest": (
            _canonical_array_digest(
                np.asarray(bridge_observations, dtype=np.int64),
                np.asarray(bridge_kinds, dtype=np.int8),
            )
            if bridge_observations
            else ""
        ),
        "final_rng_state_digest": _canonical_digest(
            _json_ready(rng.bit_generator.state)
        ),
        "final_composition": snapshot.composition.astype(int).tolist(),
        "path_attempt": launch.path_attempt,
    }
    pooled_scores: dict[tuple[str, str], InstrumentScore] = {}
    for window_name, counts in (
        ("pooled20", pooled20_counts),
        ("pooled30", pooled30_counts),
    ):
        for preprocessing in PREPROCESSINGS:
            score = (
                _safe_score(counts, preprocessing, include_full_typeset=True)
                if len(counts) >= 3
                else _nan_score(len(counts))
            )
            pooled_scores[(window_name, preprocessing)] = score
            lineage.update(
                _score_fields(f"{window_name}_{preprocessing}", score)
            )
    for row in window_rows:
        for preprocessing in PREPROCESSINGS:
            current_a = row[f"{preprocessing}_partition_a"]
            current_b = row[f"{preprocessing}_partition_b"]
            for window_name in ("pooled20", "pooled30"):
                pooled = pooled_scores[(window_name, preprocessing)]
                row[f"{preprocessing}_disagreement_{window_name}"] = (
                    partition_disagreement(
                        current_a,
                        current_b,
                        pooled.partition_a,
                        pooled.partition_b,
                    )
                )
        if not complete:
            for key in tuple(row):
                if key.startswith(("clr_", "raw_count_")) and not key.endswith(
                    ("_observations", "_transitions", "_digest")
                ):
                    row[key] = float("nan")
    return lineage, window_rows, edit_rows


def _run_matrix(args: tuple[int, RunSpec, str]) -> WindowBridgeBatch:
    matrix_id, spec, model_path = args
    with threadpool_limits(limits=1):
        config = GardConfig()
        beta = generate_beta(
            config,
            np.random.default_rng(_matrix_seed(spec, matrix_id, "matrix")),
        )
        initial = generate_initial_composition(
            config,
            np.random.default_rng(_matrix_seed(spec, matrix_id, "initial")),
        )
        predictor = FrozenFullPredictor.load(model_path)
        lineage_rows: list[dict[str, Any]] = []
        window_rows: list[dict[str, Any]] = []
        edit_rows: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            for replicate in range(spec.replicates):
                launch = _natural_launch(
                    matrix_id,
                    beta,
                    initial,
                    candidate,
                    replicate,
                    spec,
                )
                for arm in ARMS:
                    lineage, windows, edits = _run_arm(
                        matrix_id,
                        launch,
                        beta,
                        predictor,
                        spec,
                        arm,
                    )
                    lineage_rows.append(lineage)
                    window_rows.extend(windows)
                    edit_rows.extend(edits)
        provisional = WindowBridgeBatch(
            matrix_id=matrix_id,
            beta=np.asarray(beta, dtype=np.float64),
            initial_composition=np.asarray(initial, dtype=np.int16),
            lineage_rows=tuple(lineage_rows),
            window_rows=tuple(window_rows),
            selected_edit_rows=tuple(edit_rows),
            scientific_digest="",
        )
        return WindowBridgeBatch(
            **{**asdict(provisional), "scientific_digest": _batch_digest(provisional)}
        )


def _checkpoint_contract(
    spec: RunSpec, registration_id: str, stage: str
) -> dict[str, Any]:
    value = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "protocol_id": protocol()["protocol_id"],
        "stage": stage,
        "spec": asdict(spec),
        "source_hashes": _source_hashes(),
    }
    value["contract_id"] = _canonical_digest(_json_ready(value))
    return value


def _write_status(
    work: Path, stage: str, completed: int, total: int, **extra: Any
) -> None:
    safe = stage.replace("/", "_")
    start_path = work / f"started_at_{safe}.txt"
    if not start_path.exists():
        start_path.parent.mkdir(parents=True, exist_ok=True)
        start_path.write_text(str(time.time()), encoding="ascii")
    elapsed = max(0.0, time.time() - float(start_path.read_text(encoding="ascii")))
    rate = completed / elapsed if completed and elapsed else 0.0
    payload = {
        "format": STATUS_FORMAT,
        "stage": stage,
        "completed": completed,
        "total": total,
        "fraction": completed / total if total else 1.0,
        "elapsed_seconds": elapsed,
        "eta_seconds": (total - completed) / rate if rate else None,
        "pid": os.getpid(),
        **extra,
    }
    _atomic_json(work / "campaign_status.json", payload)


def _run_checkpointed(
    spec: RunSpec,
    registration_id: str,
    directory: Path,
    work: Path,
    stage: str,
    workers: int,
) -> list[WindowBridgeBatch]:
    directory.mkdir(parents=True, exist_ok=True)
    contract = _checkpoint_contract(spec, registration_id, stage)
    contract_path = directory / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != _json_ready(contract):
            raise ValueError("window-bridge checkpoint contract changed")
    else:
        _atomic_json(contract_path, contract)
    batches: list[WindowBridgeBatch | None] = [None] * spec.matrices
    missing: list[int] = []
    for matrix_id in range(spec.matrices):
        path = directory / f"matrix_{matrix_id:04d}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                batch = pickle.load(handle)
            if not isinstance(batch, WindowBridgeBatch) or batch.matrix_id != matrix_id:
                raise ValueError(f"invalid checkpoint {path}")
            if batch.scientific_digest != _batch_digest(batch):
                raise ValueError(f"checkpoint digest mismatch {path}")
            batches[matrix_id] = batch
        else:
            missing.append(matrix_id)
    completed = spec.matrices - len(missing)
    _write_status(work, stage, completed, spec.matrices, reused=completed)
    arguments = [
        (matrix_id, spec, str(DEFAULT_REGISTRATION / "frozen_full_predictor.npz"))
        for matrix_id in missing
    ]
    executor: ProcessPoolExecutor | None = None
    generated: Iterable[WindowBridgeBatch]
    if workers <= 1:
        generated = map(_run_matrix, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        generated = executor.map(_run_matrix, arguments, chunksize=1)
    try:
        for matrix_id, batch in zip(missing, generated, strict=True):
            if batch.matrix_id != matrix_id:
                raise AssertionError(
                    f"worker returned matrix {batch.matrix_id}, expected {matrix_id}"
                )
            observed = _batch_digest(batch)
            if batch.scientific_digest != observed:
                raise AssertionError(
                    f"worker digest mismatch stored={batch.scientific_digest} observed={observed}"
                )
            batches[matrix_id] = batch
            _atomic_pickle(directory / f"matrix_{matrix_id:04d}.pkl", batch)
            completed += 1
            _write_status(work, stage, completed, spec.matrices, reused=spec.matrices - len(missing))
            print(f"[{stage}] {completed}/{spec.matrices} matrices", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    if any(batch is None for batch in batches):
        raise AssertionError("window-bridge checkpoint stage incomplete")
    return [batch for batch in batches if batch is not None]


def _seeded_rng(domain: str, *keys: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(SEED_DOMAINS[domain], LABEL, *keys))


def _summary(
    values: NDArray,
    repetitions: int,
    key: str,
    arrays: dict[str, NDArray],
    *,
    direction: str,
) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if vector.size:
        bootstrap_rng = _seeded_rng("bootstrap", key)
        indices = bootstrap_rng.integers(
            0, vector.size, size=(repetitions, vector.size)
        )
        bootstrap = vector[indices].mean(axis=1)
        random_rng = _seeded_rng("randomization", key)
        signs = random_rng.choice((-1.0, 1.0), size=(repetitions, vector.size))
        randomized = (signs * vector).mean(axis=1)
        observed = float(vector.mean())
        if direction == "positive":
            p_value = (1 + np.count_nonzero(randomized >= observed)) / (repetitions + 1)
        elif direction == "negative":
            p_value = (1 + np.count_nonzero(randomized <= observed)) / (repetitions + 1)
        else:
            raise ValueError("direction must be positive or negative")
        ci95 = np.quantile(bootstrap, (0.025, 0.975))
    else:
        bootstrap = randomized = np.full(repetitions, np.nan)
        p_value = float("nan")
        ci95 = np.asarray((np.nan, np.nan))
    safe = key.replace("/", "__")
    arrays[f"{safe}__matrix_values"] = vector
    arrays[f"{safe}__bootstrap"] = np.asarray(bootstrap, dtype=np.float64)
    arrays[f"{safe}__sign_randomization"] = np.asarray(randomized, dtype=np.float64)
    return {
        "effect": float(vector.mean()) if vector.size else float("nan"),
        "ci95": [float(ci95[0]), float(ci95[1])],
        "one_sided_sign_randomization_p": float(p_value),
        "direction": direction,
        "matrices": int(vector.size),
        "matrices_expected_sign": int(
            np.count_nonzero(vector > 0 if direction == "positive" else vector < 0)
        ),
        "maximum_absolute_matrix_effect": (
            float(np.max(np.abs(vector))) if vector.size else float("nan")
        ),
    }


def _apply_holm(items: Sequence[dict[str, Any]]) -> None:
    locations = [
        index
        for index, item in enumerate(items)
        if np.isfinite(item.get("one_sided_sign_randomization_p", np.nan))
    ]
    if not locations:
        return
    adjusted = holm_adjust(
        [float(items[index]["one_sided_sign_randomization_p"]) for index in locations]
    )
    for index, value in zip(locations, adjusted, strict=True):
        items[index]["holm_adjusted_p"] = float(value)


def _score_suffixes() -> tuple[str, ...]:
    return (
        "revised",
        "full_typeset",
        "macro_typeset",
        "normalized_full",
        "causation",
        "emergence",
        "synergy",
        *(f"atom_{name}" for name in ATOM_NAMES),
    )


def _assemble_lineage_metrics(
    lineage: pd.DataFrame, windows: pd.DataFrame, spec: RunSpec
) -> pd.DataFrame:
    keys = ["matrix_id", "candidate", "replicate", "arm"]
    output = lineage.copy()
    for range_name, start in (
        ("rolling20", spec.pooled20_start),
        ("rolling30", spec.pooled30_start),
    ):
        selected = windows[windows["step"] >= start]
        value_columns = [
            f"{preprocessing}_{suffix}"
            for preprocessing in PREPROCESSINGS
            for suffix in _score_suffixes()
            if f"{preprocessing}_{suffix}" in selected.columns
        ]
        means = selected.groupby(keys, sort=True)[value_columns].mean().reset_index()
        means = means.rename(
            columns={name: f"{range_name}_{name}" for name in value_columns}
        )
        output = output.merge(means, on=keys, how="left", validate="one_to_one")
    return output


def _arm_effect_series(frame: pd.DataFrame, metric: str, candidate: str, replicate: int) -> pd.Series:
    selected = frame[
        (frame["candidate"].astype(str).str.zfill(2) == candidate)
        & (frame["replicate"] == replicate)
    ]
    table = selected.groupby(["matrix_id", "arm"], sort=True)[metric].mean().unstack("arm")
    if not set(ARMS).issubset(table.columns):
        return pd.Series(dtype=float)
    return (table["MODEL_STABILIZE"] - table["MODEL_DESTABILIZE"]).dropna()


def analyze_batches(
    batches: Sequence[WindowBridgeBatch], spec: RunSpec
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    lineage = pd.DataFrame([row for batch in batches for row in batch.lineage_rows])
    windows = pd.DataFrame([row for batch in batches for row in batch.window_rows])
    edits = pd.DataFrame([row for batch in batches for row in batch.selected_edit_rows])
    metrics_frame = _assemble_lineage_metrics(lineage, windows, spec)
    arrays: dict[str, NDArray] = {}
    matrix_rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    metric_names = ["inherited_31_60"]
    for window_name in ("pooled20", "pooled30", "rolling20", "rolling30"):
        for preprocessing in PREPROCESSINGS:
            for suffix in _score_suffixes():
                name = f"{window_name}_{preprocessing}_{suffix}"
                if name in metrics_frame:
                    metric_names.append(name)
    for metric in metric_names:
        local_cells: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            for replicate in range(spec.replicates):
                series = _arm_effect_series(
                    metrics_frame, metric, candidate, replicate
                )
                for matrix_id, value in series.items():
                    matrix_rows.append(
                        {
                            "family": "arm_effect",
                            "metric": metric,
                            "candidate": candidate,
                            "replicate": replicate,
                            "matrix_id": int(matrix_id),
                            "value": float(value),
                        }
                    )
                summary = _summary(
                    series.to_numpy(float),
                    spec.bootstrap_repetitions,
                    f"{spec.label}/arm_effect/{metric}/c{candidate}/r{replicate}",
                    arrays,
                    direction="positive",
                )
                summary.update(
                    {
                        "family": "arm_effect",
                        "metric": metric,
                        "candidate": candidate,
                        "replicate": replicate,
                    }
                )
                local_cells.append(summary)
                cells.append(summary)
        _apply_holm(local_cells)
    moderation_cells: list[dict[str, Any]] = []
    for range_name in ("20", "30"):
        local_cells = []
        pooled_metric = f"pooled{range_name}_clr_revised"
        rolling_metric = f"rolling{range_name}_clr_revised"
        for candidate in CANDIDATES:
            for replicate in range(spec.replicates):
                pooled = _arm_effect_series(
                    metrics_frame, pooled_metric, candidate, replicate
                )
                rolling = _arm_effect_series(
                    metrics_frame, rolling_metric, candidate, replicate
                )
                common = pooled.index.intersection(rolling.index)
                moderation = rolling.loc[common] - pooled.loc[common]
                for matrix_id, value in moderation.items():
                    matrix_rows.append(
                        {
                            "family": f"moderation{range_name}",
                            "metric": "rolling_minus_pooled_clr_revised",
                            "candidate": candidate,
                            "replicate": replicate,
                            "matrix_id": int(matrix_id),
                            "value": float(value),
                        }
                    )
                summary = _summary(
                    moderation.to_numpy(float),
                    spec.bootstrap_repetitions,
                    f"{spec.label}/moderation{range_name}/c{candidate}/r{replicate}",
                    arrays,
                    direction="negative",
                )
                summary.update(
                    {
                        "family": f"moderation{range_name}",
                        "metric": "rolling_minus_pooled_clr_revised",
                        "candidate": candidate,
                        "replicate": replicate,
                    }
                )
                local_cells.append(summary)
                moderation_cells.append(summary)
        _apply_holm(local_cells)
    partition_rows: list[dict[str, Any]] = []
    for preprocessing in PREPROCESSINGS:
        for pooled_name in ("pooled20", "pooled30"):
            column = f"{preprocessing}_disagreement_{pooled_name}"
            if column not in windows:
                continue
            for (candidate, replicate, arm), group in windows.groupby(
                ["candidate", "replicate", "arm"], sort=True
            ):
                values = np.asarray(group[column], dtype=float)
                values = values[np.isfinite(values)]
                partition_rows.append(
                    {
                        "preprocessing": preprocessing,
                        "pooled_reference": pooled_name,
                        "candidate": str(candidate).zfill(2),
                        "replicate": int(replicate),
                        "arm": arm,
                        "mean_label_invariant_disagreement": (
                            float(values.mean()) if values.size else float("nan")
                        ),
                        "windows": int(values.size),
                    }
                )
    completion_rows = [
        {
            "candidate": str(candidate).zfill(2),
            "replicate": int(replicate),
            "arm": arm,
            "lineages": int(len(group)),
            "completed_horizon": int(group["completed_horizon"].sum()),
            "information_eligible": int(group["information_eligible"].sum()),
        }
        for (candidate, replicate, arm), group in lineage.groupby(
            ["candidate", "replicate", "arm"], sort=True
        )
    ]

    def positive_gate(metric: str) -> bool:
        selected = [
            item
            for item in cells
            if item["metric"] == metric and item["family"] == "arm_effect"
        ]
        return bool(
            len(selected) == 4
            and all(
                item["effect"] > 0
                and item["ci95"][0] > 0
                and item.get("holm_adjusted_p", 1.0) < 0.05
                for item in selected
            )
        )

    def negative_gate(family: str) -> bool:
        selected = [item for item in moderation_cells if item["family"] == family]
        return bool(
            len(selected) == 4
            and all(
                item["effect"] < 0
                and item["ci95"][1] < 0
                and item.get("holm_adjusted_p", 1.0) < 0.05
                for item in selected
            )
        )

    rolling30 = [
        item
        for item in cells
        if item["family"] == "arm_effect"
        and item["metric"] == "rolling30_clr_revised"
    ]
    gates = {
        "heredity_validity": positive_gate("inherited_31_60"),
        "pooled20_response": positive_gate("pooled20_clr_revised"),
        "moderation20": negative_gate("moderation20"),
        "moderation30": negative_gate("moderation30"),
        "full_sign_reversal": bool(
            positive_gate("pooled20_clr_revised")
            and len(rolling30) == 4
            and all(item["effect"] < 0 and item["ci95"][1] < 0 for item in rolling30)
        ),
    }
    metrics = {
        "format": "codex-ch5-phir-window-bridge-metrics-v1",
        "phase": spec.label,
        "matrices": spec.matrices,
        "cells": cells,
        "moderation_cells": moderation_cells,
        "partition_diagnostics": partition_rows,
        "completion": completion_rows,
        "gates": gates,
        "decision_status": "fresh_24_window_bridge_complete_awaiting_user_review",
    }
    frames = {
        "lineages": lineage,
        "rolling_windows": windows,
        "lineage_metrics": metrics_frame,
        "selected_edits": edits,
        "matrix_effects": pd.DataFrame(matrix_rows),
        "partition_diagnostics": pd.DataFrame(partition_rows),
    }
    return metrics, frames, arrays


def _replay_audit(
    generated: Sequence[WindowBridgeBatch], replayed: Sequence[WindowBridgeBatch]
) -> dict[str, Any]:
    rows = [
        {
            "matrix_id": left.matrix_id,
            "generated_digest": left.scientific_digest,
            "replay_digest": right.scientific_digest,
            "exact": left.scientific_digest == right.scientific_digest,
        }
        for left, right in zip(generated, replayed, strict=True)
    ]
    return {
        "format": "codex-ch5-phir-window-bridge-replay-v1",
        "matrices": rows,
        "complete_exact_replay": bool(len(rows) == MATRICES and all(row["exact"] for row in rows)),
    }


def _format_effect(item: dict[str, Any]) -> str:
    return f"{item['effect']:+.4f} [{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}]"


def _reports(metrics: dict[str, Any], registration_id: str) -> tuple[str, str]:
    lines: list[str] = []
    for metric in (
        "inherited_31_60",
        "pooled20_clr_revised",
        "rolling20_clr_revised",
        "pooled30_clr_revised",
        "rolling30_clr_revised",
    ):
        for item in metrics["cells"]:
            if item["metric"] == metric:
                lines.append(
                    f"| {metric} | {item['candidate']} | {item['replicate']} | "
                    f"{_format_effect(item)} | {item.get('holm_adjusted_p', float('nan')):.4g} |"
                )
    moderation_lines = [
        f"| {item['family']} | {item['candidate']} | {item['replicate']} | "
        f"{_format_effect(item)} | {item.get('holm_adjusted_p', float('nan')):.4g} |"
        for item in metrics["moderation_cells"]
    ]
    secondary_names = (
        "pooled20_clr_causation",
        "pooled20_clr_emergence",
        "pooled20_clr_synergy",
        "rolling30_clr_causation",
        "rolling30_clr_emergence",
        "rolling30_clr_synergy",
        "pooled20_raw_count_revised",
        "rolling30_raw_count_revised",
        "pooled20_clr_full_typeset",
        "pooled20_clr_macro_typeset",
        "rolling30_clr_full_typeset",
        "rolling30_clr_macro_typeset",
    )
    secondary_lines = [
        f"| {item['metric']} | {item['candidate']} | {item['replicate']} | "
        f"{_format_effect(item)} |"
        for item in metrics["cells"]
        if item["metric"] in secondary_names
    ]
    partition_lines = [
        f"| {item['preprocessing']} | {item['pooled_reference']} | "
        f"{item['candidate']} | {item['replicate']} | {item['arm']} | "
        f"{item['mean_label_invariant_disagreement']:.3f} | {item['windows']} |"
        for item in metrics["partition_diagnostics"]
    ]
    completion_lines = [
        f"| {item['candidate']} | {item['replicate']} | {item['arm']} | "
        f"{item['completed_horizon']}/{item['lineages']} | "
        f"{item['information_eligible']}/{item['lineages']} |"
        for item in metrics["completion"]
    ]
    technical = "\n".join(
        [
            "# Fresh 24-matrix Phi-r window bridge",
            "",
            f"Registration: `{registration_id}`. This result is separate from the completed Chapter 5 pilot and its locked confirmation.",
            "",
            "## Arm effects: stabilization minus destabilization",
            "",
            "| Metric | Candidate | Replicate | Effect [95% matrix CI] | Holm p |",
            "| --- | --- | ---: | ---: | ---: |",
            *lines,
            "",
            "## Paired estimator moderation: rolling minus pooled arm effect",
            "",
            "| Range | Candidate | Replicate | Effect [95% matrix CI] | Holm p |",
            "| --- | --- | ---: | ---: | ---: |",
            *moderation_lines,
            "",
            "## Registered gates",
            "",
            *(f"- {name}: **{value}**" for name, value in metrics["gates"].items()),
            "",
            "## Registered atom, preprocessing, and typeset sensitivities",
            "",
            "The full-dimensional rolling typeset value uses only boundaries 40 and 60, matching the completed Codex pilot; macro-typeset and revised readings use every registered rolling boundary.",
            "",
            "| Metric | Candidate | Replicate | Effect [95% matrix CI] |",
            "| --- | --- | ---: | ---: |",
            *secondary_lines,
            "",
            "All 16 individual atom effects are retained in `primary_metrics.json` and `matrix_effects.csv.gz`.",
            "",
            "## Partition reconfiguration",
            "",
            "Zero is an identical bipartition up to swapping its labels; 0.5 is maximal disagreement on common active coordinates.",
            "",
            "| Preprocessing | Pooled reference | Candidate | Replicate | Arm | Mean disagreement | Windows |",
            "| --- | --- | --- | ---: | --- | ---: | ---: |",
            *partition_lines,
            "",
            "## Completion and eligibility",
            "",
            "| Candidate | Replicate | Arm | Completed | Information eligible |",
            "| --- | ---: | --- | ---: | ---: |",
            *completion_lines,
            "",
            "## Boundaries",
            "",
            "The result tests temporal estimator dependence inside the two Codex GARD contracts. It does not select a uniquely correct Phi-r, make Phi-r a controller, or support consciousness, life, agency, biological memory, a universal origin-of-life mechanism, or a Platonic-space portal.",
            "",
        ]
    )
    gates = metrics["gates"]
    lay = "\n".join(
        [
            "# Lay summary — fresh Phi-r window bridge",
            "",
            "We used the same simulated trajectories as input to two ways of reading the Phi-r gauge. One method takes a single long measurement across many fissions. The other repeatedly takes overlapping recent measurements and averages them. This is like comparing one long-exposure photograph with an average of many moving snapshots.",
            "",
            f"The heredity reality check passed: **{gates['heredity_validity']}**. The long pooled reading rose with stabilization in every registered cell: **{gates['pooled20_response']}**. The rolling reading shifted downward relative to the pooled reading over the matched 20-fission range: **{gates['moderation20']}**, and over 30 fissions: **{gates['moderation30']}**. A complete positive-to-negative sign reversal occurred in every cell: **{gates['full_sign_reversal']}**.",
            "",
            "A window effect means the information gauge is sensitive to timescale and aggregation. It does not mean either reading is universally correct, nor does it turn the gauge into the cause of heredity.",
            "",
        ]
    )
    return technical, lay


def _jsonify_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if output[column].map(lambda value: isinstance(value, (list, tuple, dict))).any():
            output[column] = output[column].map(
                lambda value: json.dumps(_json_ready(value), separators=(",", ":"))
                if isinstance(value, (list, tuple, dict))
                else value
            )
    return output


def _write_result(
    output: Path,
    registration: dict[str, Any],
    spec: RunSpec,
    batches: Sequence[WindowBridgeBatch],
    replay: dict[str, Any],
    metrics: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    arrays: dict[str, NDArray],
) -> None:
    technical, lay = _reports(metrics, registration["registration_id"])
    with _atomic_destination(output) as destination:
        _atomic_json(destination / "primary_metrics.json", metrics)
        (destination / "SCIENTIFIC_REPORT.md").write_text(technical, encoding="utf-8")
        (destination / "LAY_SUMMARY.md").write_text(lay, encoding="utf-8")
        _atomic_json(destination / "replay_audit.json", replay)
        _atomic_json(
            destination / "claim_boundaries.json",
            {
                "supported_claims": [],
                "pilot_is_not_confirmation": True,
                "original_confirmation_remains_locked": True,
                "prohibited_interpretations": protocol()["claim_boundary"],
            },
        )
        row_counts: dict[str, int] = {}
        for name, frame in frames.items():
            table = _jsonify_table(frame)
            table.to_csv(destination / f"{name}.csv.gz", index=False, compression="gzip")
            row_counts[name] = int(len(table))
        np.savez_compressed(destination / "inference_arrays.npz", **arrays)
        np.savez_compressed(
            destination / "matrix_inputs.npz",
            matrix_id=np.asarray([batch.matrix_id for batch in batches], dtype=np.int16),
            beta=np.stack([batch.beta for batch in batches]),
            initial_composition=np.stack([batch.initial_composition for batch in batches]),
            scientific_digest=np.asarray([batch.scientific_digest for batch in batches]),
        )
        readback = {
            "table_row_counts": {
                name: int(len(pd.read_csv(destination / f"{name}.csv.gz")))
                for name in frames
            },
            "expected_table_row_counts": row_counts,
            "replay_exact": replay["complete_exact_replay"],
        }
        readback["complete_readback_exact"] = bool(
            readback["table_row_counts"] == row_counts and readback["replay_exact"]
        )
        if not readback["complete_readback_exact"]:
            raise AssertionError(f"window-bridge readback failed: {readback}")
        _atomic_json(destination / "readback_audit.json", readback)
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "matrices": spec.matrices,
            "candidates": list(CANDIDATES),
            "replicates": spec.replicates,
            "complete_exact_replay": replay["complete_exact_replay"],
            "complete_readback_exact": True,
            "original_ch5_pilot_modified": False,
            "original_confirmation_authorized": False,
            "original_confirmation_launched": False,
            "raw_molecular_trajectories_persisted": False,
            "row_counts": row_counts,
            "runtime": _runtime_versions(),
        }
        _atomic_json(destination / "manifest.json", manifest)
        write_checksums(destination)
    verify_checksums(output)


def validation_checks() -> dict[str, bool]:
    ch5_registration = verify_ch5_registration()
    ch5_pilot = verify_ch5_result(CH5_PILOT)
    fixture = np.asarray(
        [[2 + ((time + molecule) % 4) for molecule in range(100)] for time in range(40)],
        dtype=np.int64,
    )
    ours = score_counts(fixture, "clr", include_full_typeset=True)
    sealed = score_phi_window(fixture, include_typeset=True)
    batch = WindowBridgeBatch(
        matrix_id=0,
        beta=np.eye(2),
        initial_composition=np.asarray([1, 0], dtype=np.int16),
        lineage_rows=({"x": float("nan")},),
        window_rows=({"x": float("nan")},),
        selected_edit_rows=(),
        scientific_digest="",
    )
    transported = pickle.loads(pickle.dumps(batch, protocol=5))
    config = GardConfig()
    beta = generate_beta(config, np.random.default_rng(9001))
    initial = generate_initial_composition(config, np.random.default_rng(9002))
    trace_rng = np.random.default_rng(9003)
    plain_rng = np.random.default_rng(9003)
    traced = advance_fission_traced(initial, beta, config, CANDIDATES["02"], trace_rng)
    plain = advance_fission(initial, beta, config, CANDIDATES["02"], plain_rng)
    checks = {
        "01_completed_ch5_pilot_verified": ch5_pilot["complete_exact_replay"],
        "02_original_confirmation_absent": not CH5_CONFIRMATION.exists(),
        "03_original_confirmation_unauthorized": not CH5_CONFIRMATION_AUTHORIZATION.exists(),
        "03b_original_confirmation_work_absent": not CH5_CONFIRMATION_WORK.exists(),
        "04_original_registration_preserved": ch5_registration["registration_id"] == ch5_pilot["registration_id"],
        "05_fresh_24_matrices": scientific_spec().matrices == 24,
        "06_two_replicates": scientific_spec().replicates == 2,
        "07_fixed_horizon_60": scientific_spec().bridge_horizon == 60,
        "08_fixed_pooled20_boundary": scientific_spec().pooled20_start == 41,
        "09_fixed_pooled30_boundary": scientific_spec().pooled30_start == 31,
        "10_fixed_rolling_window": scientific_spec().rolling_window == 512,
        "11_only_two_model_arms": ARMS == ("MODEL_STABILIZE", "MODEL_DESTABILIZE"),
        "12_new_matrix_seed": SEED_DOMAINS["matrix"] not in ch5_registration["seed_registry"].values(),
        "13_new_future_seed": SEED_DOMAINS["future"] not in ch5_registration["seed_registry"].values(),
        "14_clr_revised_matches_sealed_instrument": abs(ours.revised - sealed.revised_phi_r) < 1e-12,
        "15_clr_atoms_match_sealed_instrument": np.allclose(ours.atoms, sealed.atoms, atol=1e-12, rtol=0),
        "16_full_typeset_matches_sealed_instrument": abs(ours.full_typeset - sealed.typeset_phi_r) < 1e-12,
        "17_macro_and_full_typeset_distinct": not np.isclose(ours.macro_typeset, ours.full_typeset),
        "18_partition_label_invariant": partition_disagreement((0, 1), (2, 3), (2, 3), (0, 1)) == 0.0,
        "19_partition_change_detected": partition_disagreement((0, 1), (2, 3), (0, 2), (1, 3)) == 0.5,
        "20_batch_digest_pickle_stable": _batch_digest(batch) == _batch_digest(transported),
        "21_trace_matches_plain_record": records_equal(traced.record, plain),
        "22_trace_matches_plain_rng": rng_states_equal(trace_rng.bit_generator.state, plain_rng.bit_generator.state),
        "23_frozen_model_hash": sha256_file(CH5_REGISTRATION / "frozen_full_predictor.npz") == EXPECTED_MODEL_SHA256,
        "24_matrix_inference_unit": protocol()["inference"]["unit"] == "whole catalytic matrix",
        "25_complete_replay_registered": protocol()["replay"].startswith("complete deterministic"),
        "26_no_raw_trace_persistence": not protocol()["raw_molecular_trajectories_persisted"],
        "27_cannot_launch_confirmation": protocol()["cannot_authorize_or_launch_original_confirmation"],
        "28_all_source_files_exist": all((ROOT / name).is_file() for name in SOURCE_FILES),
    }
    if len(checks) != 29:
        raise AssertionError("window-bridge validation check count changed")
    return checks


def run_validation(output: Path = DEFAULT_VALIDATION) -> dict[str, Any]:
    checks = validation_checks()
    if not all(checks.values()):
        raise AssertionError([name for name, passed in checks.items() if not passed])
    payload = {
        "format": "codex-ch5-phir-window-bridge-validation-v1",
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": True,
        "source_hashes": _source_hashes(),
        "scientific_matrices_generated": 0,
    }
    with _atomic_destination(output) as destination:
        _atomic_json(destination / "validation.json", payload)
        write_checksums(destination)
    verify_checksums(output)
    print("Window bridge validation passed: 29/29", flush=True)
    return payload


def register_program(
    validation_directory: Path = DEFAULT_VALIDATION,
    output: Path = DEFAULT_REGISTRATION,
) -> dict[str, Any]:
    verify_checksums(validation_directory)
    validation = json.loads((validation_directory / "validation.json").read_text(encoding="utf-8"))
    if validation["source_hashes"] != _source_hashes():
        raise ValueError("source changed after window-bridge validation")
    for forbidden in (DEFAULT_REGISTRATION, DEFAULT_SMOKE, DEFAULT_OUTPUT, DEFAULT_WORK):
        if forbidden.exists():
            raise FileExistsError(f"pre-scientific artifact already exists: {forbidden}")
    ch5 = verify_ch5_result(CH5_PILOT)
    body: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol": protocol(),
        "protocol_id": protocol()["protocol_id"],
        "source_hashes": _source_hashes(),
        "source_tree_sha256": _canonical_digest(_source_hashes()),
        "seed_registry": SEED_DOMAINS,
        "frozen_model_sha256": EXPECTED_MODEL_SHA256,
        "completed_ch5_pilot_registration_id": ch5["registration_id"],
        "completed_ch5_pilot_manifest_sha256": sha256_file(CH5_PILOT / "manifest.json"),
        "completed_ch5_pilot_checksums_sha256": sha256_file(CH5_PILOT / "SHA256SUMS"),
        "scientific_matrices_at_registration": 0,
        "external_code_data_seeds_models_imported": False,
        "numeric_environment": _runtime_versions(),
    }
    body["registration_id"] = _canonical_digest(_json_ready(body))
    with _atomic_destination(output) as destination:
        shutil.copy2(ROOT / DOCUMENT, destination / "preregistration.md")
        shutil.copy2(validation_directory / "validation.json", destination / "validation.json")
        shutil.copy2(CH5_REGISTRATION / "frozen_full_predictor.npz", destination / "frozen_full_predictor.npz")
        _atomic_json(destination / "protocol.json", protocol())
        _atomic_json(destination / "seed_registry.json", SEED_DOMAINS)
        _atomic_json(destination / "registration.json", body)
        write_checksums(destination)
    verify_checksums(output)
    _append_ledger(
        f"<!-- phir-window-bridge-registration-{body['registration_id']} -->",
        (
            "## Fresh 24-matrix Phi-r window bridge registered",
            "",
            f"- Registration: `{body['registration_id']}`.",
            "- The completed Chapter 5 pilot is unchanged and the original confirmation remains locked.",
            "- Pooled and rolling readings will be computed on identical fresh trajectories.",
            "- No scientific bridge matrix existed at registration.",
        ),
    )
    print(f"Window bridge registered: {body['registration_id']}", flush=True)
    return body


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    verify_checksums(directory)
    registration = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    body = dict(registration)
    observed = body.pop("registration_id")
    if registration["format"] != REGISTRATION_FORMAT or _canonical_digest(_json_ready(body)) != observed:
        raise ValueError("window-bridge registration identity failed")
    if registration["source_hashes"] != _source_hashes():
        raise ValueError("window-bridge source tree changed")
    if registration["protocol"] != _json_ready(protocol()):
        raise ValueError("window-bridge protocol changed")
    if sha256_file(directory / "frozen_full_predictor.npz") != EXPECTED_MODEL_SHA256:
        raise ValueError("window-bridge frozen model changed")
    return registration


def run_smoke(output: Path = DEFAULT_SMOKE) -> dict[str, Any]:
    registration = verify_registration()
    spec = smoke_spec()
    model = str(DEFAULT_REGISTRATION / "frozen_full_predictor.npz")
    first = [_run_matrix((0, spec, model))]
    second = [_run_matrix((0, spec, model))]
    metrics, frames, arrays = analyze_batches(first, spec)
    payload = {
        "format": "codex-ch5-phir-window-bridge-smoke-v1",
        "registration_id": registration["registration_id"],
        "artificial_non_scientific_fixture": True,
        "exact_replay": first[0].scientific_digest == second[0].scientific_digest,
        "all_paths_exercised": bool(metrics and frames and arrays),
        "effect_sizes_arm_order_rates_and_candidate_differences_disclosed": False,
        "scientific_matrices_generated": 0,
    }
    if not payload["exact_replay"] or not payload["all_paths_exercised"]:
        raise AssertionError(f"window-bridge smoke failed: {payload}")
    with _atomic_destination(output) as destination:
        _atomic_json(destination / "smoke.json", payload)
        write_checksums(destination)
    verify_checksums(output)
    print("Window bridge non-scientific smoke passed", flush=True)
    return payload


def _prepare_work(
    work: Path, output: Path, registration_id: str, spec: RunSpec
) -> None:
    if output.exists():
        raise FileExistsError(f"completed output exists: {output}")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("window bridge requires at least 1 GB free")
    work.mkdir(parents=True, exist_ok=True)
    expected = {
        "format": "codex-ch5-phir-window-bridge-work-v1",
        "registration_id": registration_id,
        "protocol_id": protocol()["protocol_id"],
        "spec": asdict(spec),
    }
    path = work / "campaign_contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != _json_ready(expected):
            raise ValueError("window-bridge work contract changed")
    else:
        _atomic_json(path, expected)


def run_scientific(workers: int = min(os.cpu_count() or 1, 12)) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if (
        CH5_CONFIRMATION.exists()
        or CH5_CONFIRMATION_WORK.exists()
        or CH5_CONFIRMATION_AUTHORIZATION.exists()
    ):
        raise RuntimeError("original confirmation state changed after bridge registration")
    spec = scientific_spec()
    _prepare_work(DEFAULT_WORK, DEFAULT_OUTPUT, registration["registration_id"], spec)
    try:
        generated = _run_checkpointed(
            spec,
            registration["registration_id"],
            DEFAULT_WORK / "generated",
            DEFAULT_WORK,
            "generated",
            workers,
        )
        replayed = _run_checkpointed(
            spec,
            registration["registration_id"],
            DEFAULT_WORK / "replay",
            DEFAULT_WORK,
            "replay",
            workers,
        )
        replay = _replay_audit(generated, replayed)
        if not replay["complete_exact_replay"]:
            raise AssertionError("window-bridge complete replay failed")
        _write_status(DEFAULT_WORK, "analysis", 0, 1)
        metrics, frames, arrays = analyze_batches(generated, spec)
        _write_result(
            DEFAULT_OUTPUT,
            registration,
            spec,
            generated,
            replay,
            metrics,
            frames,
            arrays,
        )
        _write_status(
            DEFAULT_WORK,
            "awaiting_user_review",
            1,
            1,
            output=str(DEFAULT_OUTPUT),
        )
    except BaseException as error:
        _write_status(
            DEFAULT_WORK,
            "failed",
            0,
            1,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    _append_ledger(
        f"<!-- phir-window-bridge-result-{sha256_file(DEFAULT_OUTPUT / 'manifest.json')} -->",
        (
            "## Fresh 24-matrix Phi-r window bridge completed",
            "",
            f"- Result: `{DEFAULT_OUTPUT.relative_to(ROOT)}`.",
            "- Complete exact replay and readback passed.",
            f"- Registered gates: `{json.dumps(metrics['gates'], sort_keys=True)}`.",
            "- The original Chapter 5 confirmation remains locked and unlaunched.",
        ),
    )
    return metrics


def verify_result(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    verify_checksums(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    registration = verify_registration()
    if manifest["format"] != RESULT_FORMAT:
        raise ValueError("unsupported window-bridge result")
    if manifest["registration_id"] != registration["registration_id"]:
        raise ValueError("window-bridge result registration mismatch")
    if not manifest["complete_exact_replay"] or not manifest["complete_readback_exact"]:
        raise ValueError("window-bridge result integrity failed")
    return manifest


def status_payload() -> dict[str, Any]:
    output: dict[str, Any] = {
        "registration": DEFAULT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "complete": DEFAULT_OUTPUT.exists(),
        "original_confirmation_authorized": CH5_CONFIRMATION_AUTHORIZATION.exists(),
        "original_confirmation_complete": CH5_CONFIRMATION.exists(),
    }
    status = DEFAULT_WORK / "campaign_status.json"
    if status.exists():
        output["campaign"] = json.loads(status.read_text(encoding="utf-8"))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("register")
    subparsers.add_parser("smoke")
    run = subparsers.add_parser("run")
    run.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    subparsers.add_parser("status")
    subparsers.add_parser("verify")
    arguments = parser.parse_args(argv)
    if arguments.command == "validate":
        run_validation()
    elif arguments.command == "register":
        register_program()
    elif arguments.command == "smoke":
        run_smoke()
    elif arguments.command == "run":
        run_scientific(arguments.workers)
    elif arguments.command == "status":
        print(json.dumps(status_payload(), sort_keys=True, indent=2))
    elif arguments.command == "verify":
        print(json.dumps(verify_result(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
