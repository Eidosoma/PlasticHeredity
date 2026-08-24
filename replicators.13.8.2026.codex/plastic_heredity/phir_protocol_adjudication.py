"""Chapter 5 post-clean-room protocol adjudication bridge.

The module first replays and remeasures selected sealed D24 lineages (PAB-R),
then runs a prospectively sealed 24-matrix launch-by-selector factorial (PAB24).
It never imports or executes code or data from the independent Fable tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
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
from .features import history_features
from .intervention_core import (
    FrozenFullPredictor,
    MolecularEdit,
    ScoredEdit,
    _records_digest,
    apply_molecular_edit,
    edited_snapshot,
    enumerate_legal_edits,
    score_legal_edits,
    state_graph_features_many,
)
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .mechanistic_metrics import holm_adjust
from .phir_ch5 import (
    _append_ledger,
    _json_ready,
    _runtime_versions,
    _snapshot_after_record,
)
from .phir_feedback_dose import (
    DEFAULT_OUTPUT as D24_OUTPUT,
    DEFAULT_REGISTRATION as D24_REGISTRATION,
    EXPECTED_MODEL_SHA256,
    _action_seed as d24_action_seed,
    _future_seed as d24_future_seed,
    _matrix_seed as d24_matrix_seed,
    _natural_launch as d24_natural_launch,
    _select_arm_choice as d24_select_arm_choice,
    scientific_spec as d24_scientific_spec,
    verify_result as verify_d24_result,
)
from .phir_instruments import (
    ATOM_NAMES,
    _canonical_array_digest,
    advance_fission_traced,
    records_equal,
    rng_states_equal,
)
from .phir_window_bridge import (
    _jsonify_table,
    _nan_score,
    _safe_score,
    _score_fields,
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
DOCUMENT = "CODEX_CH5_PHIR_PROTOCOL_ADJUDICATION_PREREGISTRATION.md"

DEFAULT_VALIDATION = RESULTS / "phir_protocol_adjudication_validation"
DEFAULT_REGISTRATION = RESULTS / "phir_protocol_adjudication_registration"
DEFAULT_SMOKE = RESULTS / "phir_protocol_adjudication_smoke"
DEFAULT_OUTPUT = RESULTS / "phir_protocol_adjudication24"
DEFAULT_WORK = RESULTS / ".phir_protocol_adjudication_work"
DEFAULT_LOG = RESULTS / "phir_protocol_adjudication24.log"

LABEL = "CODEX_CH5_PHIR_PROTOCOL_ADJUDICATION_V1"
PROGRAM_FORMAT = "codex-ch5-phir-protocol-adjudication-program-v1"
REGISTRATION_FORMAT = "codex-ch5-phir-protocol-adjudication-registration-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-protocol-adjudication-checkpoint-v1"
RESULT_FORMAT = "codex-ch5-phir-protocol-adjudication-result-v1"
STATUS_FORMAT = "codex-ch5-phir-protocol-adjudication-status-v1"
SERVICE_NAME = "codex-phir-protocol-adjudication24-20260818"

MATRICES = 24
REPLICATES = 2
NATURAL_GENERATIONS = 60
CONTROL_HORIZON = 60
FINAL_START = 31
PHASE_POINTS = 16
PANEL_SIZE = 12
BOOTSTRAP_REPETITIONS = 4096
RANDOMIZATION_REPETITIONS = 4096
MINIMUM_FREE_DISK_BYTES = 1_500_000_000

LAUNCHES = ("FRESH", "MATURE")
SELECTORS = ("PANEL12", "EXHAUSTIVE")
DIRECTIONS = ("STABILIZE", "DESTABILIZE")
REPRESENTATIONS = (
    "endpoint_explicit",
    "fable_style",
    "phase_normalized",
    "generational",
)
PABR_ARMS = (
    "NOOP",
    "STABILIZE_50",
    "DESTABILIZE_50",
    "STABILIZE_100",
    "DESTABILIZE_100",
)

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/phir_protocol_adjudication.py",
    "tests/test_phir_protocol_adjudication.py",
    "plastic_heredity/phir_feedback_dose.py",
    "plastic_heredity/phir_window_bridge.py",
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

# Recorded only as provenance for the completed read-only audit.  The runner
# neither resolves these paths nor imports those files.
EXTERNAL_AUDIT_HASHES = {
    "run_phir_confirm.py": "15baaa5ed9460419f2884c60ee34b85cad4ca60228647028912d4412b60fd50f",
    "run_phir_bridge.py": "78ca182e4cc9e0b9ffcfb5b2ecf41a864a45f55e323b013b8d6afef275308cb8",
    "phir_code.py": "76bfe85a34a5d6a59be1aa64db7920c78fa4ba5e6b3c027427f8c738ab8dfef1",
    "sim.py": "289b01ea6b5c2bf62ecb3a1151505c8d92a7cf8dc9afc9ef00e64e6239a822d4",
}


def _seed_value(name: str) -> str:
    return hashlib.sha256(f"{LABEL}::{name}".encode("utf-8")).hexdigest()


SEED_DOMAINS = {
    name: _seed_value(name)
    for name in (
        "matrix",
        "initial",
        "main_path",
        "future",
        "panel_action",
        "bootstrap",
        "randomization",
        "validation",
        "smoke",
        "replay",
    )
}


@dataclass(frozen=True)
class RunSpec:
    label: str
    matrices: int
    replicates: int
    natural_generations: int
    control_horizon: int
    final_start: int
    phase_points: int
    panel_size: int
    bootstrap_repetitions: int
    randomization_repetitions: int


@dataclass(frozen=True)
class Launch:
    name: str
    candidate: str
    replicate: int
    snapshot: Snapshot
    record_digest: str
    path_attempt: int


@dataclass(frozen=True)
class Segment:
    step: int
    pre_growth: NDArray[np.int64]
    growth_observations: tuple[NDArray[np.int64], ...]
    record: FissionRecord
    post_control: NDArray[np.int64]
    edit: MolecularEdit | None


@dataclass(frozen=True)
class ProtocolBatch:
    phase: str
    matrix_id: int
    beta: NDArray[np.float64]
    initial_composition: NDArray[np.int16]
    lineage_rows: tuple[dict[str, Any], ...]
    selected_edit_rows: tuple[dict[str, Any], ...]
    scientific_digest: str


def scientific_spec() -> RunSpec:
    return RunSpec(
        label="pab24",
        matrices=MATRICES,
        replicates=REPLICATES,
        natural_generations=NATURAL_GENERATIONS,
        control_horizon=CONTROL_HORIZON,
        final_start=FINAL_START,
        phase_points=PHASE_POINTS,
        panel_size=PANEL_SIZE,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        randomization_repetitions=RANDOMIZATION_REPETITIONS,
    )


def smoke_spec() -> RunSpec:
    return RunSpec(
        label="smoke",
        matrices=1,
        replicates=1,
        natural_generations=5,
        control_horizon=6,
        final_start=4,
        phase_points=4,
        panel_size=4,
        bootstrap_repetitions=32,
        randomization_repetitions=32,
    )


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def protocol() -> dict[str, Any]:
    d24 = verify_d24_result()
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "program": "post-clean-room Chapter 5 protocol-adjudication bridge",
        "completed_d24_registration_id": d24["registration_id"],
        "completed_d24_manifest_sha256": sha256_file(D24_OUTPUT / "manifest.json"),
        "spec": asdict(scientific_spec()),
        "pabr_arms": list(PABR_ARMS),
        "launches": list(LAUNCHES),
        "selectors": list(SELECTORS),
        "directions": list(DIRECTIONS),
        "representations": list(REPRESENTATIONS),
        "endpoint": "final 30 controlled fissions, edits after fissions 1-59 only",
        "panel12": {
            "size": PANEL_SIZE,
            "sampling": "with replacement; remove uniformly over present types; add uniformly over all other types",
            "ties": "predicted probability then remove type then add type",
        },
        "common_random_streams": "future key excludes launch, selector, direction, and arm",
        "inference": {
            "unit": "whole catalytic matrix",
            "candidate_pooling": False,
            "replicate_pooling": False,
            "bootstrap_draws": BOOTSTRAP_REPETITIONS,
            "sign_randomizations": RANDOMIZATION_REPETITIONS,
            "holm_across_four_cells": True,
        },
        "replay": "complete deterministic replay of PAB-R and PAB24",
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "raw_molecular_trajectories_persisted": False,
        "no_48_matrix_continuation": True,
        "external_code_data_models_or_seeds_imported": False,
        "external_audit_hashes_provenance_only": EXTERNAL_AUDIT_HASHES,
        "claim_boundary": [
            "protocol moderation does not select a universal Phi-r",
            "information response is not the cause of hereditary control",
            "no consciousness, agency, life, or metaphysical claim",
            "previous sealed results remain unchanged",
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


def _batch_digest(batch: ProtocolBatch) -> str:
    blank = ProtocolBatch(
        phase=batch.phase,
        matrix_id=batch.matrix_id,
        beta=batch.beta,
        initial_composition=batch.initial_composition,
        lineage_rows=batch.lineage_rows,
        selected_edit_rows=batch.selected_edit_rows,
        scientific_digest="",
    )
    return _canonical_digest(_json_ready(asdict(blank)))


def _matrix_seed(spec: RunSpec, matrix_id: int, purpose: str) -> int:
    domain = SEED_DOMAINS["smoke"] if spec.label == "smoke" else SEED_DOMAINS[purpose]
    return derive_seed(domain, LABEL, spec.label, matrix_id)


def _future_seed(spec: RunSpec, candidate: str, matrix_id: int, replicate: int) -> int:
    domain = SEED_DOMAINS["smoke"] if spec.label == "smoke" else SEED_DOMAINS["future"]
    return derive_seed(domain, LABEL, spec.label, candidate, matrix_id, replicate)


def _panel_seed(
    spec: RunSpec,
    candidate: str,
    matrix_id: int,
    replicate: int,
    launch: str,
    step: int,
) -> int:
    domain = SEED_DOMAINS["smoke"] if spec.label == "smoke" else SEED_DOMAINS["panel_action"]
    return derive_seed(domain, LABEL, spec.label, candidate, matrix_id, replicate, launch, step)


def _fresh_launch(initial: NDArray, candidate: str, replicate: int) -> Launch:
    return Launch(
        name="FRESH",
        candidate=candidate,
        replicate=replicate,
        snapshot=Snapshot(np.asarray(initial, dtype=np.int64).copy(), 0, (), ()),
        record_digest=_records_digest(()),
        path_attempt=-1,
    )


def _mature_launch(
    matrix_id: int,
    beta: NDArray,
    initial: NDArray,
    candidate: str,
    replicate: int,
    spec: RunSpec,
) -> Launch:
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
                records.append(traced.record)
                snapshot = _snapshot_after_record(snapshot, traced.record)
            return Launch(
                name="MATURE",
                candidate=candidate,
                replicate=replicate,
                snapshot=snapshot,
                record_digest=_records_digest(records),
                path_attempt=attempt,
            )
        except SimulationError:
            continue
    raise SimulationError(f"no complete mature launch c{candidate} m{matrix_id} r{replicate}")


def sample_panel_edits(
    composition: NDArray,
    rng: np.random.Generator,
    size: int = PANEL_SIZE,
) -> tuple[MolecularEdit, ...]:
    values = np.asarray(composition, dtype=np.int64)
    present = np.flatnonzero(values > 0)
    if not present.size or values.size < 2 or size < 1:
        raise ValueError("panel sampling requires a nonempty multitype composition")
    output: list[MolecularEdit] = []
    for _ in range(size):
        remove = int(present[int(rng.integers(0, present.size))])
        raw_add = int(rng.integers(0, values.size - 1))
        add = raw_add if raw_add < remove else raw_add + 1
        output.append(MolecularEdit(remove, add))
    return tuple(output)


def _score_panel(
    predictor: FrozenFullPredictor,
    candidate: str,
    snapshot: Snapshot,
    beta: NDArray,
    edits: Sequence[MolecularEdit],
) -> tuple[float, tuple[ScoredEdit, ...]]:
    config = GardConfig()
    noop = predictor.predict_snapshot(candidate, snapshot, beta, config)
    compositions = np.vstack(
        [apply_molecular_edit(snapshot.composition, edit) for edit in edits]
    )
    state = state_graph_features_many(compositions, beta, config)
    direct = history_features(snapshot, config)
    history = np.broadcast_to(direct, (len(edits), direct.size))
    probabilities = predictor.predict_features(candidate, state, history)
    return float(noop), tuple(
        ScoredEdit(edit, float(value), float(value - noop))
        for edit, value in zip(edits, probabilities, strict=True)
    )


def _extreme(scores: Sequence[ScoredEdit], direction: str) -> ScoredEdit:
    if direction not in DIRECTIONS or not scores:
        raise ValueError("invalid extreme selection")
    sign = 1.0 if direction == "STABILIZE" else -1.0
    return min(
        scores,
        key=lambda item: (
            sign * item.predicted_probability,
            item.edit.remove_type,
            item.edit.add_type,
        ),
    )


def _select_pab_edit(
    predictor: FrozenFullPredictor,
    candidate: str,
    snapshot: Snapshot,
    beta: NDArray,
    selector: str,
    direction: str,
    panel_rng: np.random.Generator,
    panel_size: int,
) -> tuple[float, ScoredEdit, int, int]:
    config = GardConfig()
    if selector == "PANEL12":
        sampled = sample_panel_edits(snapshot.composition, panel_rng, panel_size)
        noop, scores = _score_panel(predictor, candidate, snapshot, beta, sampled)
        choice = _extreme(scores, direction)
        return noop, choice, len(scores), len(set(sampled))
    if selector == "EXHAUSTIVE":
        noop, scores = score_legal_edits(predictor, candidate, snapshot, beta, config)
        choice = _extreme(scores, direction)
        return noop, choice, len(scores), len(scores)
    raise ValueError(f"unknown selector {selector}")


def resample_phase(path: NDArray, points: int = PHASE_POINTS) -> NDArray[np.float64]:
    values = np.asarray(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or points < 2:
        raise ValueError("phase resampling requires observations by types and >=2 points")
    if values.shape[0] == 1:
        return np.repeat(values, points, axis=0)
    positions = np.linspace(0.0, values.shape[0] - 1.0, points)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    weight = (positions - lower)[:, None]
    output = (1.0 - weight) * values[lower] + weight * values[upper]
    output[0] = values[0]
    output[-1] = values[-1]
    return np.asarray(output, dtype=np.float64)


def trace_representations(
    segments: Sequence[Segment],
    final_start: int,
    phase_points: int,
    include_registered: bool,
) -> dict[str, NDArray]:
    if len(segments) < final_start or not 1 <= final_start <= len(segments):
        return {}
    start_index = final_start - 1
    final_segments = tuple(segments[start_index:])
    boundary = np.asarray(segments[start_index - 1].post_control, dtype=np.int64)
    endpoint: list[NDArray] = [boundary]
    registered: list[NDArray] = [boundary]
    fable: list[NDArray] = []
    phase: list[NDArray] = []
    for index, segment in enumerate(final_segments):
        endpoint.extend(segment.growth_observations)
        endpoint.append(segment.record.daughter)
        registered.extend(segment.growth_observations)
        registered.append(segment.record.daughter)
        if segment.edit is not None and index < len(final_segments) - 1:
            endpoint.append(segment.post_control)
        if segment.edit is not None:
            registered.append(segment.post_control)
        fable.extend(segment.growth_observations)
        local_path = np.vstack((segment.pre_growth, *segment.growth_observations))
        phase.extend(resample_phase(local_path, phase_points))
    generational = [boundary]
    generational.extend(segment.post_control for segment in final_segments[:-1])
    generational.append(final_segments[-1].record.daughter)
    output = {
        "endpoint_explicit": np.asarray(endpoint),
        "fable_style": np.asarray(fable),
        "phase_normalized": np.asarray(phase),
        "generational": np.asarray(generational),
    }
    if include_registered:
        output["registered_explicit"] = np.asarray(registered)
    return output


def _fraction(composition: NDArray) -> NDArray[np.float64]:
    values = np.asarray(composition, dtype=np.float64)
    mass = float(values.sum())
    return values / mass if mass > 0.0 else np.zeros_like(values)


def _l1_fraction_jump(left: NDArray, right: NDArray) -> float:
    return float(np.abs(_fraction(left) - _fraction(right)).sum())


def _composition_metrics(composition: NDArray, beta: NDArray) -> dict[str, float]:
    fraction = _fraction(composition)
    positive = fraction[fraction > 0.0]
    return {
        "entropy": float(-np.sum(positive * np.log(positive))),
        "occupied_types": float(np.count_nonzero(fraction)),
        "top1_share": float(fraction.max(initial=0.0)),
        "throughput": float(fraction @ np.asarray(beta, dtype=np.float64) @ fraction),
    }


def _physical_fields(
    segments: Sequence[Segment], beta: NDArray, final_start: int
) -> dict[str, Any]:
    inherited = np.asarray(
        [segment.record.h > GardConfig().inheritance_threshold for segment in segments],
        dtype=np.float64,
    )
    growth = np.asarray([segment.record.growth_steps for segment in segments], dtype=float)
    fission_l1 = np.asarray(
        [
            _l1_fraction_jump(segment.record.parent, segment.record.daughter)
            for segment in segments
        ],
        dtype=float,
    )
    edit_l1 = np.asarray(
        [
            _l1_fraction_jump(segment.record.daughter, segment.post_control)
            for segment in segments
        ],
        dtype=float,
    )
    end_metrics = [
        _composition_metrics(
            segment.record.daughter if index == len(segments) - 1 else segment.post_control,
            beta,
        )
        for index, segment in enumerate(segments)
    ]
    start = final_start - 1

    def mean(values: NDArray, begin: int = 0) -> float:
        selected = np.asarray(values[begin:], dtype=float)
        return float(selected.mean()) if selected.size else float("nan")

    final_state = end_metrics[-1] if end_metrics else {
        "entropy": float("nan"),
        "occupied_types": float("nan"),
        "top1_share": float("nan"),
        "throughput": float("nan"),
    }
    output: dict[str, Any] = {
        "inherited_1_60": mean(inherited),
        "inherited_31_60": mean(inherited, start),
        "breaks_1_60": int(np.count_nonzero(inherited == 0.0)),
        "breaks_31_60": int(np.count_nonzero(inherited[start:] == 0.0)),
        "growth_updates_mean_1_60": mean(growth),
        "growth_updates_mean_31_60": mean(growth, start),
        "fission_l1_mean_31_60": mean(fission_l1, start),
        "fission_cosine_distance_mean_31_60": mean(1.0 - np.asarray([s.record.h for s in segments]), start),
        "edit_l1_mean_31_60": mean(edit_l1, start),
        "edits_applied": int(sum(segment.edit is not None for segment in segments)),
        "final_entropy": final_state["entropy"],
        "final_occupied_types": final_state["occupied_types"],
        "final_top1_share": final_state["top1_share"],
        "final_throughput": final_state["throughput"],
    }
    for name in ("entropy", "occupied_types", "top1_share", "throughput"):
        output[f"mean_{name}_31_60"] = float(
            np.mean([item[name] for item in end_metrics[start:]])
        ) if len(end_metrics) > start else float("nan")
    return output


def _score_trace_fields(
    segments: Sequence[Segment],
    spec: RunSpec,
    include_registered: bool,
) -> dict[str, Any]:
    representations = trace_representations(
        segments,
        spec.final_start,
        spec.phase_points,
        include_registered,
    )
    output: dict[str, Any] = {}
    expected = list(REPRESENTATIONS)
    if include_registered:
        expected.append("registered_explicit")
    for name in expected:
        counts = representations.get(name)
        score = (
            _safe_score(counts, "clr", include_full_typeset=True)
            if counts is not None and counts.shape[0] >= 3
            else _nan_score(0 if counts is None else int(counts.shape[0]))
        )
        output.update(_score_fields(name, score))
    return output


def _run_d24_arm(
    matrix_id: int,
    launch: Any,
    beta: NDArray,
    predictor: FrozenFullPredictor,
    arm: str,
    spec: RunSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Regenerate one sealed D24 lineage while retaining only compact summaries."""

    config = GardConfig()
    d24_spec = d24_scientific_spec()
    rng = np.random.default_rng(
        d24_future_seed(d24_spec, launch.candidate, matrix_id, launch.replicate)
    )
    action_rng = np.random.default_rng(
        d24_action_seed(d24_spec, launch.candidate, matrix_id, launch.replicate)
    )
    snapshot = launch.snapshot
    records: list[FissionRecord] = []
    segments: list[Segment] = []
    observations: list[NDArray[np.int64]] = [snapshot.composition.copy()]
    kinds: list[int] = []
    edit_rows: list[dict[str, Any]] = []
    for step in range(1, d24_spec.control_horizon + 1):
        pre_growth = snapshot.composition.copy()
        try:
            traced = advance_fission_traced(
                pre_growth,
                beta,
                config,
                CANDIDATES[launch.candidate],
                rng,
            )
        except SimulationError:
            break
        for composition in traced.growth_observations:
            observations.append(np.asarray(composition, dtype=np.int64).copy())
            kinds.append(0)
        observations.append(traced.record.daughter.copy())
        kinds.append(1)
        records.append(traced.record)
        snapshot = _snapshot_after_record(snapshot, traced.record)
        choice = d24_select_arm_choice(
            arm,
            predictor,
            launch.candidate,
            snapshot,
            beta,
            action_rng,
        )
        if choice.edit is not None:
            snapshot = edited_snapshot(snapshot, choice.edit)
            observations.append(snapshot.composition.copy())
            kinds.append(2)
        segments.append(
            Segment(
                step=step,
                pre_growth=pre_growth,
                growth_observations=tuple(
                    np.asarray(value, dtype=np.int64).copy()
                    for value in traced.growth_observations
                ),
                record=traced.record,
                post_control=snapshot.composition.copy(),
                edit=choice.edit,
            )
        )
        archived_action = {
            "matrix_id": matrix_id,
            "candidate": launch.candidate,
            "replicate": launch.replicate,
            "arm": arm,
            "step": step,
            "edit_applied": int(choice.edit is not None),
            "remove_type": choice.edit.remove_type if choice.edit is not None else -1,
            "add_type": choice.edit.add_type if choice.edit is not None else -1,
            "requested_alpha": choice.requested_alpha,
            "achieved_alpha": choice.achieved_alpha,
            "noop_probability": choice.noop_probability,
            "neutral_probability": choice.neutral_probability,
            "minimum_probability": choice.minimum_probability,
            "maximum_probability": choice.maximum_probability,
            "target_probability": choice.target_probability,
            "selected_probability": choice.selected_probability,
        }
        archived_action["action_digest"] = _canonical_digest(_json_ready(archived_action))
        edit_rows.append({"phase": "PAB-R", **archived_action})
    complete = len(segments) == d24_spec.control_horizon
    row: dict[str, Any] = {
        "phase": "PAB-R",
        "matrix_id": matrix_id,
        "candidate": launch.candidate,
        "replicate": launch.replicate,
        "launch": "MATURE_D24",
        "selector": "D24_EXHAUSTIVE_DOSE",
        "direction": "NOOP" if arm == "NOOP" else arm.split("_")[0],
        "dose": 0.0 if arm == "NOOP" else float(arm.rsplit("_", 1)[1]) / 100.0,
        "arm": arm,
        "completed_horizon": int(complete),
        "information_eligible": int(complete),
        "completed_fissions": len(segments),
        "extinct": int(not complete),
        "natural_record_digest": launch.record_digest,
        "controlled_record_digest": _records_digest(records),
        "final_rng_state_digest": _canonical_digest(_json_ready(rng.bit_generator.state)),
        "controlled_observation_digest": _canonical_array_digest(
            np.asarray(observations, dtype=np.int64), np.asarray(kinds, dtype=np.int8)
        ),
        "path_attempt": launch.path_attempt,
        "final_composition": snapshot.composition.astype(int).tolist(),
    }
    row.update(_physical_fields(segments, beta, spec.final_start))
    row.update(_score_trace_fields(segments, spec, include_registered=True))
    return row, edit_rows


def _run_pab_arm(
    matrix_id: int,
    launch: Launch,
    beta: NDArray,
    predictor: FrozenFullPredictor,
    selector: str,
    direction: str,
    spec: RunSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = GardConfig()
    rng = np.random.default_rng(
        _future_seed(spec, launch.candidate, matrix_id, launch.replicate)
    )
    snapshot = launch.snapshot
    records: list[FissionRecord] = []
    segments: list[Segment] = []
    edit_rows: list[dict[str, Any]] = []
    is_noop = selector == "NOOP"
    arm = f"{launch.name}__NOOP" if is_noop else f"{launch.name}__{selector}__{direction}"
    for step in range(1, spec.control_horizon + 1):
        pre_growth = snapshot.composition.copy()
        try:
            traced = advance_fission_traced(
                pre_growth,
                beta,
                config,
                CANDIDATES[launch.candidate],
                rng,
            )
        except SimulationError:
            break
        records.append(traced.record)
        snapshot = _snapshot_after_record(snapshot, traced.record)
        edit: MolecularEdit | None = None
        noop_probability = predictor.predict_snapshot(
            launch.candidate, snapshot, beta, config
        )
        selected_probability = noop_probability
        scored = unique_scored = 0
        if not is_noop and step < spec.control_horizon:
            panel_rng = np.random.default_rng(
                _panel_seed(
                    spec,
                    launch.candidate,
                    matrix_id,
                    launch.replicate,
                    launch.name,
                    step,
                )
            )
            noop_probability, choice, scored, unique_scored = _select_pab_edit(
                predictor,
                launch.candidate,
                snapshot,
                beta,
                selector,
                direction,
                panel_rng,
                spec.panel_size,
            )
            edit = choice.edit
            selected_probability = choice.predicted_probability
            snapshot = edited_snapshot(snapshot, edit)
        segments.append(
            Segment(
                step=step,
                pre_growth=pre_growth,
                growth_observations=tuple(
                    np.asarray(value, dtype=np.int64).copy()
                    for value in traced.growth_observations
                ),
                record=traced.record,
                post_control=snapshot.composition.copy(),
                edit=edit,
            )
        )
        action = {
            "phase": "PAB24",
            "matrix_id": matrix_id,
            "candidate": launch.candidate,
            "replicate": launch.replicate,
            "launch": launch.name,
            "selector": selector,
            "direction": direction,
            "arm": arm,
            "step": step,
            "edit_applied": int(edit is not None),
            "remove_type": edit.remove_type if edit is not None else -1,
            "add_type": edit.add_type if edit is not None else -1,
            "noop_probability": float(noop_probability),
            "selected_probability": float(selected_probability),
            "predicted_shift": float(selected_probability - noop_probability),
            "scored_edits": scored,
            "unique_scored_edits": unique_scored,
        }
        action["action_digest"] = _canonical_digest(_json_ready(action))
        edit_rows.append(action)
    complete = len(segments) == spec.control_horizon
    row: dict[str, Any] = {
        "phase": "PAB24",
        "matrix_id": matrix_id,
        "candidate": launch.candidate,
        "replicate": launch.replicate,
        "launch": launch.name,
        "selector": selector,
        "direction": direction,
        "dose": 1.0 if not is_noop else 0.0,
        "arm": arm,
        "completed_horizon": int(complete),
        "information_eligible": int(complete),
        "completed_fissions": len(segments),
        "extinct": int(not complete),
        "natural_record_digest": launch.record_digest,
        "controlled_record_digest": _records_digest(records),
        "final_rng_state_digest": _canonical_digest(_json_ready(rng.bit_generator.state)),
        "path_attempt": launch.path_attempt,
        "final_composition": snapshot.composition.astype(int).tolist(),
    }
    row.update(_physical_fields(segments, beta, spec.final_start))
    row.update(_score_trace_fields(segments, spec, include_registered=False))
    return row, edit_rows


def _run_pabr_matrix(matrix_id: int, spec: RunSpec, model_path: str) -> ProtocolBatch:
    d24_spec = d24_scientific_spec()
    config = GardConfig()
    beta = generate_beta(
        config,
        np.random.default_rng(d24_matrix_seed(d24_spec, matrix_id, "matrix")),
    )
    initial = generate_initial_composition(
        config,
        np.random.default_rng(d24_matrix_seed(d24_spec, matrix_id, "initial")),
    )
    predictor = FrozenFullPredictor.load(model_path)
    rows: list[dict[str, Any]] = []
    edits: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for replicate in range(spec.replicates):
            launch = d24_natural_launch(
                matrix_id, beta, initial, candidate, replicate, d24_spec
            )
            for arm in PABR_ARMS:
                row, local_edits = _run_d24_arm(
                    matrix_id, launch, beta, predictor, arm, spec
                )
                rows.append(row)
                edits.extend(local_edits)
    provisional = ProtocolBatch(
        phase="PAB-R",
        matrix_id=matrix_id,
        beta=np.asarray(beta, dtype=np.float64),
        initial_composition=np.asarray(initial, dtype=np.int16),
        lineage_rows=tuple(rows),
        selected_edit_rows=tuple(edits),
        scientific_digest="",
    )
    return ProtocolBatch(
        **{**asdict(provisional), "scientific_digest": _batch_digest(provisional)}
    )


def _run_pab24_matrix(matrix_id: int, spec: RunSpec, model_path: str) -> ProtocolBatch:
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
    rows: list[dict[str, Any]] = []
    edits: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for replicate in range(spec.replicates):
            launches = (
                _fresh_launch(initial, candidate, replicate),
                _mature_launch(matrix_id, beta, initial, candidate, replicate, spec),
            )
            for launch in launches:
                row, local = _run_pab_arm(
                    matrix_id, launch, beta, predictor, "NOOP", "NOOP", spec
                )
                rows.append(row)
                edits.extend(local)
                for selector in SELECTORS:
                    for direction in DIRECTIONS:
                        row, local = _run_pab_arm(
                            matrix_id,
                            launch,
                            beta,
                            predictor,
                            selector,
                            direction,
                            spec,
                        )
                        rows.append(row)
                        edits.extend(local)
    provisional = ProtocolBatch(
        phase="PAB24",
        matrix_id=matrix_id,
        beta=np.asarray(beta, dtype=np.float64),
        initial_composition=np.asarray(initial, dtype=np.int16),
        lineage_rows=tuple(rows),
        selected_edit_rows=tuple(edits),
        scientific_digest="",
    )
    return ProtocolBatch(
        **{**asdict(provisional), "scientific_digest": _batch_digest(provisional)}
    )


def _run_matrix(args: tuple[str, int, RunSpec, str]) -> ProtocolBatch:
    phase, matrix_id, spec, model_path = args
    with threadpool_limits(limits=1):
        if phase == "PAB-R":
            return _run_pabr_matrix(matrix_id, spec, model_path)
        if phase == "PAB24":
            return _run_pab24_matrix(matrix_id, spec, model_path)
    raise ValueError(f"unknown protocol-adjudication phase {phase}")


def _checkpoint_contract(
    spec: RunSpec, registration_id: str, phase: str, stage: str
) -> dict[str, Any]:
    value = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "protocol_id": protocol()["protocol_id"],
        "phase": phase,
        "stage": stage,
        "spec": asdict(spec),
        "source_hashes": _source_hashes(),
    }
    value["contract_id"] = _canonical_digest(_json_ready(value))
    return value


def _write_status(stage: str, completed: int, total: int, **extra: Any) -> None:
    safe = stage.replace("/", "_")
    started = DEFAULT_WORK / f"started_at_{safe}.txt"
    if not started.exists():
        started.parent.mkdir(parents=True, exist_ok=True)
        started.write_text(str(time.time()), encoding="ascii")
    elapsed = max(0.0, time.time() - float(started.read_text(encoding="ascii")))
    rate = completed / elapsed if completed and elapsed else 0.0
    _atomic_json(
        DEFAULT_WORK / "campaign_status.json",
        {
            "format": STATUS_FORMAT,
            "stage": stage,
            "completed": completed,
            "total": total,
            "fraction": completed / total if total else 1.0,
            "elapsed_seconds": elapsed,
            "eta_seconds": (total - completed) / rate if rate else None,
            "pid": os.getpid(),
            "free_disk_bytes": shutil.disk_usage(ROOT).free,
            **extra,
        },
    )


def _run_checkpointed(
    phase: str,
    spec: RunSpec,
    registration_id: str,
    directory: Path,
    stage: str,
    workers: int,
) -> list[ProtocolBatch]:
    directory.mkdir(parents=True, exist_ok=True)
    contract = _checkpoint_contract(spec, registration_id, phase, stage)
    contract_path = directory / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != _json_ready(contract):
            raise ValueError(f"checkpoint contract changed: {directory}")
    else:
        _atomic_json(contract_path, contract)
    batches: list[ProtocolBatch | None] = [None] * spec.matrices
    missing: list[int] = []
    for matrix_id in range(spec.matrices):
        path = directory / f"matrix_{matrix_id:04d}.pkl"
        if not path.exists():
            missing.append(matrix_id)
            continue
        with path.open("rb") as handle:
            batch = pickle.load(handle)
        if (
            not isinstance(batch, ProtocolBatch)
            or batch.phase != phase
            or batch.matrix_id != matrix_id
            or batch.scientific_digest != _batch_digest(batch)
        ):
            raise ValueError(f"invalid checkpoint {path}")
        batches[matrix_id] = batch
    completed = spec.matrices - len(missing)
    _write_status(stage, completed, spec.matrices, reused=completed)
    model_path = str(DEFAULT_REGISTRATION / "frozen_full_predictor.npz")
    arguments = [(phase, matrix_id, spec, model_path) for matrix_id in missing]
    executor: ProcessPoolExecutor | None = None
    generated: Iterable[ProtocolBatch]
    if workers <= 1:
        generated = map(_run_matrix, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        generated = executor.map(_run_matrix, arguments, chunksize=1)
    try:
        for matrix_id, batch in zip(missing, generated, strict=True):
            if (
                batch.phase != phase
                or batch.matrix_id != matrix_id
                or batch.scientific_digest != _batch_digest(batch)
            ):
                raise AssertionError("worker returned an invalid adjudication batch")
            batches[matrix_id] = batch
            _atomic_pickle(directory / f"matrix_{matrix_id:04d}.pkl", batch)
            completed += 1
            _write_status(
                stage,
                completed,
                spec.matrices,
                reused=spec.matrices - len(missing),
            )
            print(f"[{stage}] {completed}/{spec.matrices} matrices", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    if any(batch is None for batch in batches):
        raise AssertionError(f"checkpoint stage incomplete: {stage}")
    return [batch for batch in batches if batch is not None]


def _replay_audit(
    phase: str,
    generated: Sequence[ProtocolBatch],
    replayed: Sequence[ProtocolBatch],
    expected_matrices: int,
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
        "format": "codex-ch5-phir-protocol-adjudication-replay-v1",
        "phase": phase,
        "matrices": rows,
        "complete_exact_replay": bool(
            len(rows) == expected_matrices and all(row["exact"] for row in rows)
        ),
    }


def _normalize_candidate(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["candidate"] = output["candidate"].astype(str).str.zfill(2)
    return output


def _archive_audit(batches: Sequence[ProtocolBatch]) -> dict[str, Any]:
    generated = _normalize_candidate(
        pd.DataFrame([row for batch in batches for row in batch.lineage_rows])
    )
    generated_actions = _normalize_candidate(
        pd.DataFrame([row for batch in batches for row in batch.selected_edit_rows])
    )
    archived = _normalize_candidate(pd.read_csv(D24_OUTPUT / "lineages.csv.gz"))
    archived = archived[archived["arm"].isin(PABR_ARMS)].copy()
    archived_actions = _normalize_candidate(
        pd.read_csv(D24_OUTPUT / "selected_edits.csv.gz")
    )
    archived_actions = archived_actions[archived_actions["arm"].isin(PABR_ARMS)].copy()
    keys = ["matrix_id", "candidate", "replicate", "arm"]
    action_keys = [*keys, "step"]
    generated = generated.sort_values(keys).reset_index(drop=True)
    archived = archived.sort_values(keys).reset_index(drop=True)
    generated_actions = generated_actions.sort_values(action_keys).reset_index(drop=True)
    archived_actions = archived_actions.sort_values(action_keys).reset_index(drop=True)
    key_exact = bool(
        len(generated) == len(archived)
        and generated[keys].equals(archived[keys])
        and len(generated_actions) == len(archived_actions)
        and generated_actions[action_keys].equals(archived_actions[action_keys])
    )
    exact_columns = (
        "completed_horizon",
        "information_eligible",
        "completed_fissions",
        "extinct",
        "natural_record_digest",
        "controlled_record_digest",
        "final_rng_state_digest",
        "controlled_observation_digest",
        "path_attempt",
    )
    exact_mismatches: dict[str, int] = {}
    if key_exact:
        for column in exact_columns:
            exact_mismatches[column] = int(
                np.count_nonzero(
                    generated[column].astype(str).to_numpy()
                    != archived[column].astype(str).to_numpy()
                )
            )
    score_columns = (
        "revised",
        "full_typeset",
        "macro_typeset",
        "normalized_full",
        "causation",
        "emergence",
        "synergy",
        *(f"atom_{name}" for name in ATOM_NAMES),
    )
    maximum_score_error = 0.0
    score_mismatches = 0
    if key_exact:
        for suffix in score_columns:
            left = generated[f"registered_explicit_{suffix}"].to_numpy(float)
            right = archived[f"pooled30_clr_{suffix}"].to_numpy(float)
            finite = np.isfinite(left) & np.isfinite(right)
            errors = np.abs(left[finite] - right[finite])
            if errors.size:
                maximum_score_error = max(maximum_score_error, float(errors.max()))
            score_mismatches += int(np.count_nonzero(~(finite | (np.isnan(left) & np.isnan(right)))))
            score_mismatches += int(np.count_nonzero(errors > 1e-12))
    action_digest_mismatches = (
        int(
            np.count_nonzero(
                generated_actions["action_digest"].astype(str).to_numpy()
                != archived_actions["action_digest"].astype(str).to_numpy()
            )
        )
        if key_exact
        else max(len(generated_actions), len(archived_actions))
    )
    inherited_error = (
        float(
            np.max(
                np.abs(
                    generated["inherited_31_60"].to_numpy(float)
                    - archived["inherited_31_60"].to_numpy(float)
                )
            )
        )
        if key_exact and len(generated)
        else float("inf")
    )
    passed = bool(
        key_exact
        and not any(exact_mismatches.values())
        and action_digest_mismatches == 0
        and score_mismatches == 0
        and maximum_score_error <= 1e-12
        and inherited_error <= 1e-15
    )
    return {
        "format": "codex-ch5-phir-protocol-adjudication-archive-audit-v1",
        "archived_lineages": int(len(archived)),
        "regenerated_lineages": int(len(generated)),
        "archived_actions": int(len(archived_actions)),
        "regenerated_actions": int(len(generated_actions)),
        "key_exact": key_exact,
        "exact_mismatches": exact_mismatches,
        "action_digest_mismatches": action_digest_mismatches,
        "score_mismatches": score_mismatches,
        "maximum_score_absolute_error": maximum_score_error,
        "maximum_inherited_fraction_error": inherited_error,
        "passed": passed,
    }


def _seeded_rng(domain: str, *keys: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(SEED_DOMAINS[domain], LABEL, *keys))


def _summary(
    values: NDArray,
    repetitions: int,
    key: str,
    arrays: dict[str, NDArray],
) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if vector.size:
        bootstrap_rng = _seeded_rng("bootstrap", key)
        indices = bootstrap_rng.integers(0, vector.size, size=(repetitions, vector.size))
        bootstrap = vector[indices].mean(axis=1)
        random_rng = _seeded_rng("randomization", key)
        signs = random_rng.choice((-1.0, 1.0), size=(repetitions, vector.size))
        randomized = (signs * vector).mean(axis=1)
        observed = float(vector.mean())
        positive_p = (1 + np.count_nonzero(randomized >= observed)) / (repetitions + 1)
        negative_p = (1 + np.count_nonzero(randomized <= observed)) / (repetitions + 1)
        two_sided_p = min(1.0, 2.0 * min(positive_p, negative_p))
        ci95 = np.quantile(bootstrap, (0.025, 0.975))
    else:
        bootstrap = randomized = np.full(repetitions, np.nan)
        positive_p = negative_p = two_sided_p = float("nan")
        ci95 = np.asarray((np.nan, np.nan))
    safe = key.replace("/", "__")
    arrays[f"{safe}__matrix_values"] = vector
    arrays[f"{safe}__bootstrap"] = np.asarray(bootstrap, dtype=np.float64)
    arrays[f"{safe}__sign_randomization"] = np.asarray(randomized, dtype=np.float64)
    return {
        "effect": float(vector.mean()) if vector.size else float("nan"),
        "ci95": [float(ci95[0]), float(ci95[1])],
        "positive_sign_randomization_p": float(positive_p),
        "negative_sign_randomization_p": float(negative_p),
        "two_sided_sign_randomization_p": float(two_sided_p),
        "matrices": int(vector.size),
        "matrices_positive": int(np.count_nonzero(vector > 0)),
        "matrices_negative": int(np.count_nonzero(vector < 0)),
        "maximum_absolute_matrix_effect": (
            float(np.max(np.abs(vector))) if vector.size else float("nan")
        ),
    }


def _holm(items: Sequence[dict[str, Any]], source: str, destination: str) -> None:
    finite = [item for item in items if np.isfinite(item.get(source, np.nan))]
    if not finite:
        return
    adjusted = holm_adjust([float(item[source]) for item in finite])
    for item, value in zip(finite, adjusted, strict=True):
        item[destination] = float(value)


def _effect_series(
    frame: pd.DataFrame,
    metric: str,
    candidate: str,
    replicate: int,
    launch: str,
    selector: str,
) -> pd.Series:
    selected = frame[
        (frame["candidate"].astype(str).str.zfill(2) == candidate)
        & (frame["replicate"] == replicate)
        & (frame["launch"] == launch)
        & (frame["selector"] == selector)
        & (frame["direction"].isin(DIRECTIONS))
        & (frame["information_eligible"] == 1)
    ]
    pivot = selected.pivot(index="matrix_id", columns="direction", values=metric)
    if not set(DIRECTIONS).issubset(pivot.columns):
        return pd.Series(dtype=float)
    return pivot["STABILIZE"] - pivot["DESTABILIZE"]


def _pabr_effect_series(
    frame: pd.DataFrame,
    metric: str,
    candidate: str,
    replicate: int,
    dose: float,
) -> pd.Series:
    tag = int(round(100 * dose))
    selected = frame[
        (frame["candidate"].astype(str).str.zfill(2) == candidate)
        & (frame["replicate"] == replicate)
        & (frame["arm"].isin((f"STABILIZE_{tag:02d}", f"DESTABILIZE_{tag:02d}")))
        & (frame["information_eligible"] == 1)
    ].copy()
    selected["direction_local"] = selected["arm"].str.split("_").str[0]
    pivot = selected.pivot(index="matrix_id", columns="direction_local", values=metric)
    if not set(DIRECTIONS).issubset(pivot.columns):
        return pd.Series(dtype=float)
    return pivot["STABILIZE"] - pivot["DESTABILIZE"]


def _record_series(
    rows: list[dict[str, Any]],
    family: str,
    metric: str,
    candidate: str,
    replicate: int,
    series: pd.Series,
    summary: dict[str, Any],
    **labels: Any,
) -> dict[str, Any]:
    for matrix_id, value in series.items():
        rows.append(
            {
                "family": family,
                "metric": metric,
                "candidate": candidate,
                "replicate": replicate,
                "matrix_id": int(matrix_id),
                "value": float(value),
                **labels,
            }
        )
    summary.update(
        {
            "family": family,
            "metric": metric,
            "candidate": candidate,
            "replicate": replicate,
            **labels,
        }
    )
    return summary


def analyze_batches(
    pabr_batches: Sequence[ProtocolBatch],
    pab24_batches: Sequence[ProtocolBatch],
    spec: RunSpec,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    pabr = _normalize_candidate(
        pd.DataFrame([row for batch in pabr_batches for row in batch.lineage_rows])
    )
    pab24 = _normalize_candidate(
        pd.DataFrame([row for batch in pab24_batches for row in batch.lineage_rows])
    )
    edits = _normalize_candidate(
        pd.DataFrame(
            [
                row
                for batch in (*pabr_batches, *pab24_batches)
                for row in batch.selected_edit_rows
            ]
        )
    )
    arrays: dict[str, NDArray] = {}
    matrix_rows: list[dict[str, Any]] = []
    pabr_cells: list[dict[str, Any]] = []
    pab24_cells: list[dict[str, Any]] = []
    moderation_cells: list[dict[str, Any]] = []

    information_suffixes = (
        "revised",
        "full_typeset",
        "macro_typeset",
        "normalized_full",
        "causation",
        "emergence",
        "synergy",
        *(f"atom_{name}" for name in ATOM_NAMES),
    )
    for dose in (0.5, 1.0):
        for representation in (*REPRESENTATIONS, "registered_explicit"):
            for suffix in information_suffixes:
                metric = f"{representation}_{suffix}"
                local: list[dict[str, Any]] = []
                for candidate in CANDIDATES:
                    for replicate in range(spec.replicates):
                        series = _pabr_effect_series(
                            pabr, metric, candidate, replicate, dose
                        )
                        summary = _summary(
                            series.to_numpy(float),
                            spec.bootstrap_repetitions,
                            f"pabr/dose{dose}/{metric}/c{candidate}/r{replicate}",
                            arrays,
                        )
                        local.append(
                            _record_series(
                                matrix_rows,
                                "pabr_effect",
                                metric,
                                candidate,
                                replicate,
                                series,
                                summary,
                                dose=dose,
                                representation=representation,
                            )
                        )
                _holm(local, "positive_sign_randomization_p", "holm_positive_p")
                _holm(local, "negative_sign_randomization_p", "holm_negative_p")
                pabr_cells.extend(local)
        local = []
        for candidate in CANDIDATES:
            for replicate in range(spec.replicates):
                series = _pabr_effect_series(
                    pabr, "inherited_31_60", candidate, replicate, dose
                )
                summary = _summary(
                    series.to_numpy(float),
                    spec.bootstrap_repetitions,
                    f"pabr/dose{dose}/inherited/c{candidate}/r{replicate}",
                    arrays,
                )
                local.append(
                    _record_series(
                        matrix_rows,
                        "pabr_effect",
                        "inherited_31_60",
                        candidate,
                        replicate,
                        series,
                        summary,
                        dose=dose,
                        representation="physical",
                    )
                )
        _holm(local, "positive_sign_randomization_p", "holm_positive_p")
        _holm(local, "negative_sign_randomization_p", "holm_negative_p")
        pabr_cells.extend(local)

    primary_representations = ("endpoint_explicit", "fable_style")
    clock_representations = ("phase_normalized", "generational")
    physical_metrics = (
        "inherited_31_60",
        "inherited_1_60",
        "breaks_31_60",
        "growth_updates_mean_31_60",
        "fission_l1_mean_31_60",
        "fission_cosine_distance_mean_31_60",
        "edit_l1_mean_31_60",
        "mean_entropy_31_60",
        "mean_occupied_types_31_60",
        "mean_top1_share_31_60",
        "mean_throughput_31_60",
    )
    metric_groups: list[tuple[str, ...]] = []
    for representation in primary_representations:
        metric_groups.append(
            tuple(f"{representation}_{suffix}" for suffix in information_suffixes)
        )
    for representation in clock_representations:
        metric_groups.append(
            tuple(
                f"{representation}_{suffix}"
                for suffix in (
                    "revised",
                    "full_typeset",
                    "causation",
                    "emergence",
                    "synergy",
                )
            )
        )
    metric_groups.append(physical_metrics)
    for launch in LAUNCHES:
        for selector in SELECTORS:
            for metrics_group in metric_groups:
                for metric in metrics_group:
                    local = []
                    for candidate in CANDIDATES:
                        for replicate in range(spec.replicates):
                            series = _effect_series(
                                pab24,
                                metric,
                                candidate,
                                replicate,
                                launch,
                                selector,
                            )
                            summary = _summary(
                                series.to_numpy(float),
                                spec.bootstrap_repetitions,
                                f"pab24/{launch}/{selector}/{metric}/c{candidate}/r{replicate}",
                                arrays,
                            )
                            local.append(
                                _record_series(
                                    matrix_rows,
                                    "pab24_effect",
                                    metric,
                                    candidate,
                                    replicate,
                                    series,
                                    summary,
                                    launch=launch,
                                    selector=selector,
                                )
                            )
                    _holm(local, "positive_sign_randomization_p", "holm_positive_p")
                    _holm(local, "negative_sign_randomization_p", "holm_negative_p")
                    pab24_cells.extend(local)

    def interaction(
        family: str,
        metric: str,
        left: tuple[str, str],
        right: tuple[str, str],
    ) -> None:
        local: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            for replicate in range(spec.replicates):
                left_values = _effect_series(
                    pab24, metric, candidate, replicate, left[0], left[1]
                )
                right_values = _effect_series(
                    pab24, metric, candidate, replicate, right[0], right[1]
                )
                common = left_values.index.intersection(right_values.index)
                series = left_values.loc[common] - right_values.loc[common]
                summary = _summary(
                    series.to_numpy(float),
                    spec.bootstrap_repetitions,
                    f"{family}/{metric}/c{candidate}/r{replicate}",
                    arrays,
                )
                local.append(
                    _record_series(
                        matrix_rows,
                        family,
                        metric,
                        candidate,
                        replicate,
                        series,
                        summary,
                        left=f"{left[0]}__{left[1]}",
                        right=f"{right[0]}__{right[1]}",
                    )
                )
        _holm(local, "positive_sign_randomization_p", "holm_positive_p")
        _holm(local, "negative_sign_randomization_p", "holm_negative_p")
        _holm(local, "two_sided_sign_randomization_p", "holm_two_sided_p")
        moderation_cells.extend(local)

    interaction(
        "launch_moderation",
        "fable_style_revised",
        ("FRESH", "PANEL12"),
        ("MATURE", "PANEL12"),
    )
    interaction(
        "selector_moderation",
        "fable_style_revised",
        ("FRESH", "PANEL12"),
        ("FRESH", "EXHAUSTIVE"),
    )
    # Encoding is a within-cell metric contrast rather than a launch/selector
    # contrast, so construct it explicitly.
    encoding_local: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for replicate in range(spec.replicates):
            fable = _effect_series(
                pab24,
                "fable_style_revised",
                candidate,
                replicate,
                "FRESH",
                "PANEL12",
            )
            explicit = _effect_series(
                pab24,
                "endpoint_explicit_revised",
                candidate,
                replicate,
                "FRESH",
                "PANEL12",
            )
            common = fable.index.intersection(explicit.index)
            series = fable.loc[common] - explicit.loc[common]
            summary = _summary(
                series.to_numpy(float),
                spec.bootstrap_repetitions,
                f"encoding_moderation/c{candidate}/r{replicate}",
                arrays,
            )
            encoding_local.append(
                _record_series(
                    matrix_rows,
                    "encoding_moderation",
                    "fable_minus_endpoint_revised",
                    candidate,
                    replicate,
                    series,
                    summary,
                    left="fable_style",
                    right="endpoint_explicit",
                )
            )
    _holm(encoding_local, "positive_sign_randomization_p", "holm_positive_p")
    _holm(encoding_local, "negative_sign_randomization_p", "holm_negative_p")
    _holm(encoding_local, "two_sided_sign_randomization_p", "holm_two_sided_p")
    moderation_cells.extend(encoding_local)

    def positive(items: Sequence[dict[str, Any]]) -> bool:
        return bool(
            len(items) == 4
            and all(
                item["effect"] > 0.0
                and item["ci95"][0] > 0.0
                and item.get("holm_positive_p", 1.0) < 0.05
                for item in items
            )
        )

    direct = [
        item
        for item in pab24_cells
        if item["metric"] == "fable_style_revised"
        and item["launch"] == "FRESH"
        and item["selector"] == "PANEL12"
    ]
    launch_cells = [
        item for item in moderation_cells if item["family"] == "launch_moderation"
    ]
    encoding_cells = [
        item for item in moderation_cells if item["family"] == "encoding_moderation"
    ]
    selector_cells = [
        item for item in moderation_cells if item["family"] == "selector_moderation"
    ]
    heredity_families: dict[str, bool] = {}
    for launch in LAUNCHES:
        for selector in SELECTORS:
            selected = [
                item
                for item in pab24_cells
                if item["metric"] == "inherited_31_60"
                and item["launch"] == launch
                and item["selector"] == selector
            ]
            heredity_families[f"{launch}_{selector}"] = positive(selected)
    phase_cells = [
        item
        for item in pab24_cells
        if item["metric"] == "phase_normalized_revised"
        and item["launch"] == "FRESH"
        and item["selector"] == "PANEL12"
    ]
    generational_cells = [
        item
        for item in pab24_cells
        if item["metric"] == "generational_revised"
        and item["launch"] == "FRESH"
        and item["selector"] == "PANEL12"
    ]
    gates = {
        "direct_protocol_bridge": positive(direct),
        "launch_moderation": positive(launch_cells),
        "encoding_moderation": positive(encoding_cells),
        "selector_moderation_all_four_two_sided": bool(
            len(selector_cells) == 4
            and all(item.get("holm_two_sided_p", 1.0) < 0.05 for item in selector_cells)
        ),
        "phase_normalized_positive_sign_all_cells": bool(
            len(phase_cells) == 4 and all(item["effect"] > 0 for item in phase_cells)
        ),
        "generational_positive_sign_all_cells": bool(
            len(generational_cells) == 4
            and all(item["effect"] > 0 for item in generational_cells)
        ),
        "heredity_manipulation_validity": bool(all(heredity_families.values())),
        "heredity_families": heredity_families,
    }
    if gates["direct_protocol_bridge"] and gates["launch_moderation"]:
        classification = "launch_maturity_supported_as_major_moderator"
    elif gates["direct_protocol_bridge"] and gates["encoding_moderation"]:
        classification = "observation_encoding_supported_as_major_moderator"
    elif gates["selector_moderation_all_four_two_sided"]:
        classification = "selection_search_supported_as_moderator"
    elif (
        len(direct) == 4
        and all(item["effect"] < 0 for item in direct)
        and not gates["launch_moderation"]
        and not gates["encoding_moderation"]
    ):
        classification = "residual_simulator_or_trajectory_level_disagreement"
    else:
        classification = "bounded_mixed_or_unresolved_disagreement"

    completion = [
        {
            "phase": phase,
            "candidate": candidate,
            "replicate": int(replicate),
            "launch": launch,
            "selector": selector,
            "direction": direction,
            "lineages": int(len(group)),
            "complete": int(group["completed_horizon"].sum()),
        }
        for phase, frame in (("PAB-R", pabr), ("PAB24", pab24))
        for (candidate, replicate, launch, selector, direction), group in frame.groupby(
            ["candidate", "replicate", "launch", "selector", "direction"], sort=True
        )
    ]
    metrics = {
        "format": "codex-ch5-phir-protocol-adjudication-metrics-v1",
        "pabr_cells": pabr_cells,
        "pab24_cells": pab24_cells,
        "moderation_cells": moderation_cells,
        "completion": completion,
        "gates": gates,
        "classification": classification,
        "decision_status": "pabr_and_pab24_complete_awaiting_human_review",
    }
    frames = {
        "pabr_lineages": pabr,
        "pab24_lineages": pab24,
        "selected_edits": edits,
        "matrix_effects": pd.DataFrame(matrix_rows),
        "completion": pd.DataFrame(completion),
    }
    return metrics, frames, arrays


def _effect_text(item: dict[str, Any]) -> str:
    return f"{item['effect']:+.4f} [{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}]"


def _reports(metrics: dict[str, Any], registration_id: str) -> tuple[str, str]:
    selected_cells = [
        item
        for item in metrics["pab24_cells"]
        if (
            item["metric"]
            in {
                "inherited_31_60",
                "fable_style_revised",
                "endpoint_explicit_revised",
                "phase_normalized_revised",
                "generational_revised",
                "fable_style_full_typeset",
            }
            and item["launch"] == "FRESH"
            and item["selector"] == "PANEL12"
        )
    ]
    primary_rows = [
        "| "
        + " | ".join(
            (
                item["metric"],
                str(item["candidate"]).zfill(2),
                str(item["replicate"]),
                _effect_text(item),
                f"{item.get('holm_positive_p', float('nan')):.4g}",
            )
        )
        + " |"
        for item in selected_cells
    ]
    moderation_rows = [
        f"| {item['family']} | {str(item['candidate']).zfill(2)} | {item['replicate']} | {_effect_text(item)} | {item.get('holm_positive_p', float('nan')):.4g} | {item.get('holm_two_sided_p', float('nan')):.4g} |"
        for item in metrics["moderation_cells"]
    ]
    pabr_rows = [
        f"| {item['dose']:.1f} | {item['representation']} | {str(item['candidate']).zfill(2)} | {item['replicate']} | {_effect_text(item)} |"
        for item in metrics["pabr_cells"]
        if item["metric"].endswith("_revised")
    ]
    gate_lines = [
        f"- {name}: **{value}**"
        for name, value in metrics["gates"].items()
        if name != "heredity_families"
    ]
    gate_lines.extend(
        f"  - heredity {name}: **{value}**"
        for name, value in metrics["gates"]["heredity_families"].items()
    )
    technical = "\n".join(
        (
            "# Chapter 5 PAB24 protocol-adjudication report",
            "",
            f"Registration: `{registration_id}`.",
            "",
            "Both the archived remeasurement and the fresh factorial received a complete exact replay. PAB-R first reproduced the selected sealed D24 lineages before any fresh PAB24 matrix was admitted.",
            "",
            "## Fresh-launch PANEL12 primary readings",
            "",
            "All effects are stabilization minus destabilization. The confidence interval and randomization unit is the catalytic matrix.",
            "",
            "| Metric | Candidate | Replicate | Effect [95% matrix CI] | Holm p(+) |",
            "| --- | --- | ---: | ---: | ---: |",
            *primary_rows,
            "",
            "## Registered moderation tests",
            "",
            "| Family | Candidate | Replicate | Effect [95% matrix CI] | Holm p(+) | Holm p(two-sided) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            *moderation_rows,
            "",
            "## Archived D24 remeasurement",
            "",
            "| Dose | Encoding | Candidate | Replicate | Revised Phi-r effect [95% matrix CI] |",
            "| ---: | --- | --- | ---: | ---: |",
            *pabr_rows,
            "",
            "## Frozen gates and classification",
            "",
            *gate_lines,
            "",
            f"Classification: **{metrics['classification']}**.",
            "",
            "## Interpretation boundary",
            "",
            "This post-clean-room experiment can identify a protocol moderator of the cross-implementation Phi-r disagreement. It does not select a universal information measure, alter the prior causal-heredity results, or support claims about consciousness, agency, life, or metaphysical organization.",
            "",
        )
    )
    classification_text = {
        "launch_maturity_supported_as_major_moderator": "The main difference was where control began. On young, diffuse assemblies the revised information gauge moved in the Fable direction much more strongly than on already evolved assemblies.",
        "observation_encoding_supported_as_major_moderator": "The main difference was how the same molecular movie was sampled. Hiding fission/edit boundaries made the revised information gauge move in the Fable direction.",
        "selection_search_supported_as_moderator": "The size of the controller's search mattered: the 12-option controller and exhaustive controller produced materially different information responses.",
        "residual_simulator_or_trajectory_level_disagreement": "Matching launch age, controller style, and observation clocks did not reverse the Codex sign. The remaining difference therefore lies in the trajectories generated by the two simulator reconstructions, not in the shared Phi-r arithmetic.",
        "bounded_mixed_or_unresolved_disagreement": "No single planned protocol factor explained all four cells. The disagreement is now narrower, but remains mixed or unresolved.",
    }[metrics["classification"]]
    lay = "\n".join(
        (
            "# Lay summary — protocol-adjudication bridge",
            "",
            "We asked whether the two clean rooms disagreed because they watched different kinds of assemblies, used controllers with different search power, or cut the molecular movie into frames differently. We first replayed the old Codex trajectories exactly, then ran all of those choices side by side on 24 new catalytic worlds.",
            "",
            classification_text,
            "",
            "The heredity result and the information-gauge result remain separate. Feedback can genuinely stabilize or destabilize compositional inheritance even if no single Phi-r formula responds universally across simulator contracts.",
            "",
        )
    )
    return technical, lay


def _write_result(
    registration: dict[str, Any],
    spec: RunSpec,
    pabr_batches: Sequence[ProtocolBatch],
    pab24_batches: Sequence[ProtocolBatch],
    pabr_replay: dict[str, Any],
    pab24_replay: dict[str, Any],
    archive_audit: dict[str, Any],
    metrics: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    arrays: dict[str, NDArray],
) -> None:
    technical, lay = _reports(metrics, registration["registration_id"])
    with _atomic_destination(DEFAULT_OUTPUT) as destination:
        _atomic_json(destination / "primary_metrics.json", metrics)
        _atomic_json(destination / "pabr_archive_audit.json", archive_audit)
        _atomic_json(
            destination / "replay_audit.json",
            {"PAB-R": pabr_replay, "PAB24": pab24_replay},
        )
        (destination / "SCIENTIFIC_REPORT.md").write_text(technical, encoding="utf-8")
        (destination / "LAY_SUMMARY.md").write_text(lay, encoding="utf-8")
        _atomic_json(
            destination / "claim_boundaries.json",
            {
                "supported_claims": [metrics["classification"]],
                "previous_results_modified": False,
                "post_clean_room_adjudication": True,
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
            pabr_matrix_id=np.asarray([b.matrix_id for b in pabr_batches], dtype=np.int16),
            pabr_beta=np.stack([b.beta for b in pabr_batches]),
            pabr_initial=np.stack([b.initial_composition for b in pabr_batches]),
            pabr_digest=np.asarray([b.scientific_digest for b in pabr_batches]),
            pab24_matrix_id=np.asarray([b.matrix_id for b in pab24_batches], dtype=np.int16),
            pab24_beta=np.stack([b.beta for b in pab24_batches]),
            pab24_initial=np.stack([b.initial_composition for b in pab24_batches]),
            pab24_digest=np.asarray([b.scientific_digest for b in pab24_batches]),
        )
        readback_counts = {
            name: int(len(pd.read_csv(destination / f"{name}.csv.gz")))
            for name in frames
        }
        readback = {
            "expected_table_row_counts": row_counts,
            "table_row_counts": readback_counts,
            "pabr_replay_exact": pabr_replay["complete_exact_replay"],
            "pab24_replay_exact": pab24_replay["complete_exact_replay"],
            "archive_reproduction_exact": archive_audit["passed"],
        }
        readback["complete_readback_exact"] = bool(
            readback_counts == row_counts
            and pabr_replay["complete_exact_replay"]
            and pab24_replay["complete_exact_replay"]
            and archive_audit["passed"]
        )
        if not readback["complete_readback_exact"]:
            raise AssertionError(f"protocol-adjudication readback failed: {readback}")
        _atomic_json(destination / "readback_audit.json", readback)
        _atomic_json(
            destination / "manifest.json",
            {
                "format": RESULT_FORMAT,
                "registration_id": registration["registration_id"],
                "matrices": spec.matrices,
                "candidates": list(CANDIDATES),
                "replicates": spec.replicates,
                "pabr_complete_exact_replay": True,
                "pab24_complete_exact_replay": True,
                "archive_reproduction_exact": True,
                "complete_readback_exact": True,
                "raw_molecular_trajectories_persisted": False,
                "no_48_matrix_continuation": True,
                "row_counts": row_counts,
                "runtime": _runtime_versions(),
            },
        )
        write_checksums(destination)
    verify_checksums(DEFAULT_OUTPUT)


def validation_checks() -> dict[str, bool]:
    d24 = verify_d24_result()
    config = GardConfig()
    model_path = D24_REGISTRATION / "frozen_full_predictor.npz"
    predictor = FrozenFullPredictor.load(model_path)
    composition = np.asarray([2, 1, 0, 0], dtype=np.int64)
    panel_rng = np.random.default_rng(923001)
    panel = sample_panel_edits(composition, panel_rng, 60_000)
    counts = {edit: panel.count(edit) for edit in enumerate_legal_edits(composition)}
    expected = len(panel) / len(counts)
    uniform_error = max(abs(value - expected) / expected for value in counts.values())
    tie_scores = (
        ScoredEdit(MolecularEdit(1, 3), 0.2, 0.0),
        ScoredEdit(MolecularEdit(0, 2), 0.2, 0.0),
        ScoredEdit(MolecularEdit(0, 1), 0.8, 0.0),
        ScoredEdit(MolecularEdit(1, 0), 0.8, 0.0),
    )
    edited = edited_snapshot(
        Snapshot(composition.copy(), 7, (True, False), (0.91, 0.7), 11, 50),
        MolecularEdit(0, 3),
    )
    beta = generate_beta(config, np.random.default_rng(923002))
    initial = generate_initial_composition(config, np.random.default_rng(923003))
    traced_rng = np.random.default_rng(923004)
    plain_rng = np.random.default_rng(923004)
    traced = advance_fission_traced(
        initial, beta, config, CANDIDATES["02"], traced_rng
    )
    plain = advance_fission(initial, beta, config, CANDIDATES["02"], plain_rng)
    path = np.asarray([[1.0, 3.0], [2.0, 5.0], [5.0, 9.0]])
    resampled = resample_phase(path, 7)
    fixture_segments: list[Segment] = []
    current = np.zeros(config.n_types, dtype=np.int64)
    current[:2] = (20, 20)
    for step in range(1, 4):
        parent = current.copy()
        parent[0] += 1
        daughter = current.copy()
        post = daughter.copy()
        post[0] -= 1
        post[2] += 1
        fixture_segments.append(
            Segment(
                step,
                current.copy(),
                (parent.copy(),),
                FissionRecord(parent, daughter, 0.95, 1),
                post,
                MolecularEdit(0, 2),
            )
        )
        current = post
    representations = trace_representations(fixture_segments, 2, 4, True)
    fixture_expected = (
        0.01967850472393684,
        0.01051244553924286,
        0.015065247445371925,
        0.019098973618591485,
        0.009858618341213151,
    )
    fixture_observed: list[float] = []
    atom_identity = True
    selected_atom_names = {
        "r_to_s",
        "u0_to_s",
        "u1_to_s",
        "s_to_r",
        "s_to_u0",
        "s_to_u1",
        "s_to_s",
        "u0_to_u1",
        "u1_to_u0",
    }
    for seed in range(5):
        fixture_rng = np.random.default_rng(seed)
        base = fixture_rng.lognormal(1.0, 1.0, size=config.n_types)
        values = fixture_rng.poisson(base, size=(700, config.n_types)).astype(float)
        values[:, 0] += np.arange(700) % 7
        values[:, 1] += (np.arange(700) // 5) % 9
        score = _safe_score(values, "clr", include_full_typeset=False)
        fixture_observed.append(score.revised)
        selected_sum = sum(
            value
            for name, value in zip(ATOM_NAMES, score.atoms, strict=True)
            if name in selected_atom_names
        )
        atom_identity = atom_identity and abs(selected_sum - score.revised) < 1e-12
    serial_predictor = pickle.loads(pickle.dumps(predictor, protocol=5))
    snapshot = Snapshot(initial.copy(), 0, (), ())
    prediction_before = predictor.predict_snapshot("02", snapshot, beta, config)
    prediction_after = serial_predictor.predict_snapshot("02", snapshot, beta, config)
    matrix_fixture = pd.DataFrame(
        {
            "matrix_id": np.repeat(np.arange(4), 2),
            "replicate": np.tile((0, 1), 4),
            "value": np.arange(8, dtype=float),
        }
    )
    grouped_fixture = matrix_fixture.groupby("matrix_id")["value"].mean()
    checks = {
        "01_completed_d24_verified": bool(d24["complete_exact_replay"]),
        "02_frozen_model_hash": sha256_file(model_path) == EXPECTED_MODEL_SHA256,
        "03_twenty_four_fresh_matrices": scientific_spec().matrices == 24,
        "04_two_replicates": scientific_spec().replicates == 2,
        "05_edits_stop_at_59": scientific_spec().control_horizon == 60,
        "06_panel_size_12": scientific_spec().panel_size == 12,
        "07_panel_every_edit_legal": all(edit in enumerate_legal_edits(composition) for edit in panel),
        "08_panel_with_replacement": len(set(panel)) < len(panel),
        "09_panel_uniform": uniform_error < 0.03,
        "10_stabilize_tie_deterministic": _extreme(tie_scores, "STABILIZE").edit == MolecularEdit(0, 2),
        "11_destabilize_tie_deterministic": _extreme(tie_scores, "DESTABILIZE").edit == MolecularEdit(0, 1),
        "12_future_seed_arm_free_by_signature": _future_seed(scientific_spec(), "02", 4, 1) == _future_seed(scientific_spec(), "02", 4, 1),
        "13_panel_stream_separate": _panel_seed(scientific_spec(), "02", 4, 1, "FRESH", 7) != _future_seed(scientific_spec(), "02", 4, 1),
        "14_phase_shape": resampled.shape == (7, 2),
        "15_phase_endpoints_exact": np.array_equal(resampled[[0, -1]], path[[0, -1]]),
        "16_endpoint_omits_final_edit": not np.array_equal(representations["endpoint_explicit"][-1], fixture_segments[-1].post_control),
        "17_registered_includes_final_edit": np.array_equal(representations["registered_explicit"][-1], fixture_segments[-1].post_control),
        "18_phase_trace_length": representations["phase_normalized"].shape[0] == 8,
        "19_generational_trace_length": representations["generational"].shape[0] == 3,
        "20_fable_growth_only_length": representations["fable_style"].shape[0] == 2,
        "21_trace_matches_plain_record": records_equal(traced.record, plain),
        "22_trace_matches_plain_rng": rng_states_equal(traced_rng.bit_generator.state, plain_rng.bit_generator.state),
        "23_history_unchanged_by_edit": (
            edited.generation == 7
            and edited.inheritance == (True, False)
            and edited.boundary_h == (0.91, 0.7)
            and edited.previous_growth_steps == 11
            and edited.cumulative_growth_steps == 50
        ),
        "24_edit_mass_preserved": int(edited.composition.sum()) == int(composition.sum()),
        "25_predictor_serialization_exact": np.asarray(prediction_before, dtype=np.float64).tobytes() == np.asarray(prediction_after, dtype=np.float64).tobytes(),
        "26_cross_source_revised_fixture": np.allclose(fixture_observed, fixture_expected, atol=1e-12, rtol=0.0),
        "27_nine_atom_identity": atom_identity,
        "28_matrix_block_fixture": len(grouped_fixture) == 4 and grouped_fixture.index.is_unique,
        "29_source_files_exist": all((ROOT / name).is_file() for name in SOURCE_FILES),
        "30_no_raw_trace_persistence": not protocol()["raw_molecular_trajectories_persisted"],
        "31_no_48_continuation": protocol()["no_48_matrix_continuation"],
        "32_external_assets_not_imported": not protocol()["external_code_data_models_or_seeds_imported"],
    }
    return checks


def run_validation(output: Path = DEFAULT_VALIDATION) -> dict[str, Any]:
    checks = validation_checks()
    if not all(checks.values()):
        raise AssertionError([name for name, passed in checks.items() if not passed])
    payload = {
        "format": "codex-ch5-phir-protocol-adjudication-validation-v1",
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
    print(f"Protocol-adjudication validation passed: {len(checks)}/{len(checks)}", flush=True)
    return payload


def register_program() -> dict[str, Any]:
    verify_checksums(DEFAULT_VALIDATION)
    validation = json.loads(
        (DEFAULT_VALIDATION / "validation.json").read_text(encoding="utf-8")
    )
    if validation["source_hashes"] != _source_hashes():
        raise ValueError("source changed after protocol-adjudication validation")
    for forbidden in (DEFAULT_REGISTRATION, DEFAULT_SMOKE, DEFAULT_OUTPUT, DEFAULT_WORK):
        if forbidden.exists():
            raise FileExistsError(f"pre-scientific artifact already exists: {forbidden}")
    d24 = verify_d24_result()
    body: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol": protocol(),
        "protocol_id": protocol()["protocol_id"],
        "source_hashes": _source_hashes(),
        "source_tree_sha256": _canonical_digest(_source_hashes()),
        "seed_registry": SEED_DOMAINS,
        "frozen_model_sha256": EXPECTED_MODEL_SHA256,
        "completed_d24_manifest_sha256": sha256_file(D24_OUTPUT / "manifest.json"),
        "completed_d24_registration_id": d24["registration_id"],
        "scientific_matrices_at_registration": 0,
        "post_clean_room_adjudication": True,
        "external_code_data_models_or_seeds_imported": False,
        "numeric_environment": _runtime_versions(),
    }
    body["registration_id"] = _canonical_digest(_json_ready(body))
    with _atomic_destination(DEFAULT_REGISTRATION) as destination:
        shutil.copy2(ROOT / DOCUMENT, destination / "preregistration.md")
        shutil.copy2(DEFAULT_VALIDATION / "validation.json", destination / "validation.json")
        shutil.copy2(
            D24_REGISTRATION / "frozen_full_predictor.npz",
            destination / "frozen_full_predictor.npz",
        )
        _atomic_json(destination / "protocol.json", protocol())
        _atomic_json(destination / "seed_registry.json", SEED_DOMAINS)
        _atomic_json(destination / "registration.json", body)
        write_checksums(destination)
    verify_checksums(DEFAULT_REGISTRATION)
    _append_ledger(
        f"<!-- phir-protocol-adjudication-registration-{body['registration_id']} -->",
        (
            "## Chapter 5 protocol-adjudication bridge registered",
            "",
            f"- Registration: `{body['registration_id']}`.",
            "- PAB-R exact D24 replay and a fresh 24-matrix launch-by-selector factorial were prospectively sealed.",
            "- This is explicitly post-clean-room adjudication; previous Chapter 5 results remain unchanged.",
            "- No 48-matrix continuation or simulator-contract port is authorized.",
        ),
    )
    print(f"Protocol-adjudication registered: {body['registration_id']}", flush=True)
    return body


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    verify_checksums(directory)
    registration = json.loads(
        (directory / "registration.json").read_text(encoding="utf-8")
    )
    body = dict(registration)
    observed = body.pop("registration_id")
    if registration["format"] != REGISTRATION_FORMAT:
        raise ValueError("unsupported protocol-adjudication registration")
    if _canonical_digest(_json_ready(body)) != observed:
        raise ValueError("protocol-adjudication registration identity failed")
    if registration["source_hashes"] != _source_hashes():
        raise ValueError("protocol-adjudication source tree changed")
    if registration["protocol"] != _json_ready(protocol()):
        raise ValueError("protocol-adjudication protocol changed")
    if sha256_file(directory / "frozen_full_predictor.npz") != EXPECTED_MODEL_SHA256:
        raise ValueError("protocol-adjudication frozen predictor changed")
    return registration


def run_smoke(output: Path = DEFAULT_SMOKE) -> dict[str, Any]:
    registration = verify_registration()
    spec = smoke_spec()
    model_path = str(DEFAULT_REGISTRATION / "frozen_full_predictor.npz")
    first = _run_pab24_matrix(0, spec, model_path)
    second = _run_pab24_matrix(0, spec, model_path)
    rows = pd.DataFrame(first.lineage_rows)
    actions = pd.DataFrame(first.selected_edit_rows)
    d24_spec = d24_scientific_spec()
    config = GardConfig()
    d24_beta = generate_beta(
        config,
        np.random.default_rng(d24_matrix_seed(d24_spec, 0, "matrix")),
    )
    d24_initial = generate_initial_composition(
        config,
        np.random.default_rng(d24_matrix_seed(d24_spec, 0, "initial")),
    )
    d24_launch = d24_natural_launch(0, d24_beta, d24_initial, "02", 0, d24_spec)
    predictor = FrozenFullPredictor.load(model_path)
    pabr_row, pabr_actions = _run_d24_arm(
        0, d24_launch, d24_beta, predictor, "NOOP", scientific_spec()
    )
    archive = _normalize_candidate(pd.read_csv(D24_OUTPUT / "lineages.csv.gz"))
    reference = archive[
        (archive["matrix_id"] == 0)
        & (archive["candidate"] == "02")
        & (archive["replicate"] == 0)
        & (archive["arm"] == "NOOP")
    ].iloc[0]
    pabr_fixture_exact = bool(
        pabr_row["controlled_record_digest"] == reference["controlled_record_digest"]
        and pabr_row["controlled_observation_digest"] == reference["controlled_observation_digest"]
        and abs(pabr_row["registered_explicit_revised"] - float(reference["pooled30_clr_revised"])) < 1e-12
        and len(pabr_actions) == 60
    )
    payload = {
        "format": "codex-ch5-phir-protocol-adjudication-smoke-v1",
        "registration_id": registration["registration_id"],
        "artificial_non_scientific_fixture": True,
        "exact_pab24_replay": first.scientific_digest == second.scientific_digest,
        "pabr_archived_noop_fixture_exact": pabr_fixture_exact,
        "all_ten_arm_paths_exercised": len(rows) == len(CANDIDATES) * len(LAUNCHES) * 5,
        "no_final_interventions": bool(
            actions.loc[actions["step"] == spec.control_horizon, "edit_applied"].sum() == 0
        ),
        "directed_edit_count_exact": bool(
            actions["edit_applied"].sum()
            == len(CANDIDATES)
            * len(LAUNCHES)
            * len(SELECTORS)
            * len(DIRECTIONS)
            * (spec.control_horizon - 1)
        ),
        "scientific_effect_sizes_or_arm_order_disclosed": False,
        "scientific_matrices_generated": 0,
    }
    if not all(
        payload[name]
        for name in (
            "exact_pab24_replay",
            "pabr_archived_noop_fixture_exact",
            "all_ten_arm_paths_exercised",
            "no_final_interventions",
            "directed_edit_count_exact",
        )
    ):
        raise AssertionError(f"protocol-adjudication smoke failed: {payload}")
    with _atomic_destination(output) as destination:
        _atomic_json(destination / "smoke.json", payload)
        write_checksums(destination)
    verify_checksums(output)
    print("Protocol-adjudication non-scientific smoke passed", flush=True)
    return payload


def _prepare_work(registration_id: str, spec: RunSpec) -> None:
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"completed output exists: {DEFAULT_OUTPUT}")
    free = shutil.disk_usage(ROOT).free
    if free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError(
            f"protocol-adjudication campaign requires 1.5 GB free; observed {free} bytes"
        )
    DEFAULT_WORK.mkdir(parents=True, exist_ok=True)
    expected = {
        "format": "codex-ch5-phir-protocol-adjudication-work-v1",
        "registration_id": registration_id,
        "protocol_id": protocol()["protocol_id"],
        "spec": asdict(spec),
    }
    path = DEFAULT_WORK / "campaign_contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != _json_ready(expected):
            raise ValueError("protocol-adjudication work contract changed")
    else:
        _atomic_json(path, expected)


def _load_stage(phase: str, directory: Path, spec: RunSpec) -> list[ProtocolBatch]:
    output: list[ProtocolBatch] = []
    for matrix_id in range(spec.matrices):
        path = directory / f"matrix_{matrix_id:04d}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint {path}")
        with path.open("rb") as handle:
            batch = pickle.load(handle)
        if (
            not isinstance(batch, ProtocolBatch)
            or batch.phase != phase
            or batch.matrix_id != matrix_id
            or batch.scientific_digest != _batch_digest(batch)
        ):
            raise ValueError(f"invalid checkpoint {path}")
        output.append(batch)
    return output


def run_replay_only(workers: int = min(os.cpu_count() or 1, 12)) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    spec = scientific_spec()
    _prepare_work(registration["registration_id"], spec)
    pabr_generated = _load_stage("PAB-R", DEFAULT_WORK / "pabr_generated", spec)
    pabr_replayed = _run_checkpointed(
        "PAB-R",
        spec,
        registration["registration_id"],
        DEFAULT_WORK / "pabr_replay",
        "PAB-R replay",
        workers,
    )
    pabr_audit = _replay_audit("PAB-R", pabr_generated, pabr_replayed, spec.matrices)
    archive = _archive_audit(pabr_generated)
    if not pabr_audit["complete_exact_replay"] or not archive["passed"]:
        raise AssertionError("PAB-R replay or archive audit failed")
    pab24_generated = _load_stage("PAB24", DEFAULT_WORK / "pab24_generated", spec)
    pab24_replayed = _run_checkpointed(
        "PAB24",
        spec,
        registration["registration_id"],
        DEFAULT_WORK / "pab24_replay",
        "PAB24 replay",
        workers,
    )
    pab24_audit = _replay_audit(
        "PAB24", pab24_generated, pab24_replayed, spec.matrices
    )
    if not pab24_audit["complete_exact_replay"]:
        raise AssertionError("PAB24 replay failed")
    payload = {
        "PAB-R": pabr_audit,
        "PAB24": pab24_audit,
        "archive": archive,
    }
    _atomic_json(DEFAULT_WORK / "replay_summary.json", payload)
    return payload


def run_analysis() -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    spec = scientific_spec()
    _prepare_work(registration["registration_id"], spec)
    pabr_generated = _load_stage("PAB-R", DEFAULT_WORK / "pabr_generated", spec)
    pabr_replayed = _load_stage("PAB-R", DEFAULT_WORK / "pabr_replay", spec)
    pab24_generated = _load_stage("PAB24", DEFAULT_WORK / "pab24_generated", spec)
    pab24_replayed = _load_stage("PAB24", DEFAULT_WORK / "pab24_replay", spec)
    pabr_replay = _replay_audit(
        "PAB-R", pabr_generated, pabr_replayed, spec.matrices
    )
    pab24_replay = _replay_audit(
        "PAB24", pab24_generated, pab24_replayed, spec.matrices
    )
    archive = _archive_audit(pabr_generated)
    if not (
        pabr_replay["complete_exact_replay"]
        and pab24_replay["complete_exact_replay"]
        and archive["passed"]
    ):
        raise AssertionError("analysis integrity gate failed")
    _write_status("analysis", 0, 1)
    metrics, frames, arrays = analyze_batches(pabr_generated, pab24_generated, spec)
    _write_result(
        registration,
        spec,
        pabr_generated,
        pab24_generated,
        pabr_replay,
        pab24_replay,
        archive,
        metrics,
        frames,
        arrays,
    )
    _write_status("awaiting_user_review", 1, 1, output=str(DEFAULT_OUTPUT))
    _append_ledger(
        f"<!-- phir-protocol-adjudication-result-{sha256_file(DEFAULT_OUTPUT / 'manifest.json')} -->",
        (
            "## Chapter 5 protocol-adjudication bridge completed",
            "",
            f"- Result: `{DEFAULT_OUTPUT.relative_to(ROOT)}`.",
            "- Selected archived D24 lineages were reproduced before PAB24; both phases passed complete exact replay.",
            f"- Registered classification: `{metrics['classification']}`.",
            f"- Frozen gates: `{json.dumps(metrics['gates'], sort_keys=True)}`.",
            "- The experiment stopped at 24 matrices for human review; no 48-matrix continuation ran.",
        ),
    )
    return metrics


def run_scientific(workers: int = min(os.cpu_count() or 1, 12)) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    spec = scientific_spec()
    _prepare_work(registration["registration_id"], spec)
    try:
        pabr_generated = _run_checkpointed(
            "PAB-R",
            spec,
            registration["registration_id"],
            DEFAULT_WORK / "pabr_generated",
            "PAB-R generation",
            workers,
        )
        pabr_replayed = _run_checkpointed(
            "PAB-R",
            spec,
            registration["registration_id"],
            DEFAULT_WORK / "pabr_replay",
            "PAB-R replay",
            workers,
        )
        pabr_replay = _replay_audit(
            "PAB-R", pabr_generated, pabr_replayed, spec.matrices
        )
        archive = _archive_audit(pabr_generated)
        _atomic_json(DEFAULT_WORK / "pabr_archive_audit.json", archive)
        if not pabr_replay["complete_exact_replay"]:
            raise AssertionError("PAB-R complete exact replay failed")
        if not archive["passed"]:
            raise AssertionError(f"PAB-R archived D24 reproduction failed: {archive}")
        _write_status(
            "PAB-R gate passed",
            1,
            1,
            archive_reproduction_exact=True,
            next_stage="PAB24 generation",
        )
        pab24_generated = _run_checkpointed(
            "PAB24",
            spec,
            registration["registration_id"],
            DEFAULT_WORK / "pab24_generated",
            "PAB24 generation",
            workers,
        )
        pab24_replayed = _run_checkpointed(
            "PAB24",
            spec,
            registration["registration_id"],
            DEFAULT_WORK / "pab24_replay",
            "PAB24 replay",
            workers,
        )
        pab24_replay = _replay_audit(
            "PAB24", pab24_generated, pab24_replayed, spec.matrices
        )
        if not pab24_replay["complete_exact_replay"]:
            raise AssertionError("PAB24 complete exact replay failed")
        return run_analysis()
    except BaseException as error:
        _write_status(
            "failed",
            0,
            1,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def verify_result(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    verify_checksums(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    registration = verify_registration()
    if manifest["format"] != RESULT_FORMAT:
        raise ValueError("unsupported protocol-adjudication result")
    if manifest["registration_id"] != registration["registration_id"]:
        raise ValueError("protocol-adjudication result registration mismatch")
    required = (
        "pabr_complete_exact_replay",
        "pab24_complete_exact_replay",
        "archive_reproduction_exact",
        "complete_readback_exact",
    )
    if not all(manifest[name] for name in required):
        raise ValueError("protocol-adjudication result integrity failed")
    return manifest


def status_payload() -> dict[str, Any]:
    output: dict[str, Any] = {
        "validation": DEFAULT_VALIDATION.exists(),
        "registration": DEFAULT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "complete": DEFAULT_OUTPUT.exists(),
        "service": SERVICE_NAME,
        "free_disk_bytes": shutil.disk_usage(ROOT).free,
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "no_48_continuation": True,
    }
    status = DEFAULT_WORK / "campaign_status.json"
    if status.exists():
        output["campaign"] = json.loads(status.read_text(encoding="utf-8"))
    launch = DEFAULT_WORK / "detached_launch.json"
    if launch.exists():
        output["detached_launch"] = json.loads(launch.read_text(encoding="utf-8"))
    return output


def launch_detached(workers: int = min(os.cpu_count() or 1, 12)) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"completed output exists: {DEFAULT_OUTPUT}")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("detached launch refused below the sealed 1.5 GB disk floor")
    DEFAULT_WORK.mkdir(parents=True, exist_ok=True)
    command = [
        "systemd-run",
        "--user",
        f"--unit={SERVICE_NAME}",
        "--collect",
        "--property",
        f"WorkingDirectory={ROOT}",
        "--property",
        f"StandardOutput=append:{DEFAULT_LOG}",
        "--property",
        f"StandardError=append:{DEFAULT_LOG}",
        sys.executable,
        "-m",
        "plastic_heredity.phir_protocol_adjudication",
        "run",
        "--workers",
        str(workers),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = {
        "format": "codex-ch5-phir-protocol-adjudication-detached-launch-v1",
        "registration_id": registration["registration_id"],
        "service": SERVICE_NAME,
        "workers": workers,
        "launched_at_unix": time.time(),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    _atomic_json(DEFAULT_WORK / "detached_launch.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("register")
    subparsers.add_parser("smoke")
    run = subparsers.add_parser("run")
    run.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    launch = subparsers.add_parser("launch")
    launch.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    replay = subparsers.add_parser("replay")
    replay.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    subparsers.add_parser("analyze")
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
    elif arguments.command == "launch":
        print(json.dumps(launch_detached(arguments.workers), sort_keys=True, indent=2))
    elif arguments.command == "replay":
        print(json.dumps(run_replay_only(arguments.workers), sort_keys=True, indent=2))
    elif arguments.command == "analyze":
        print(json.dumps(run_analysis()["gates"], sort_keys=True, indent=2))
    elif arguments.command == "status":
        print(json.dumps(status_payload(), sort_keys=True, indent=2))
    elif arguments.command == "verify":
        print(json.dumps(verify_result(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
