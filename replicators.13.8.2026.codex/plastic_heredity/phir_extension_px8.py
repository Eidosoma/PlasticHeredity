"""PX8 high-support confirmation of one PX7-developed resilience gauge.

The only eligible information reading is the generational, beta-partitioned,
full-dimensional unnormalized whole-minus-parts statistic.  The public revised
Phi-r is retained only as a negative control.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from . import intervention_cr5 as cr5
from . import phir_extension_px7 as px7
from .config import CANDIDATES, GardConfig
from .experiment import StateCase
from .intervention_core import MolecularEdit, apply_molecular_edit
from .mechanistic import sha256_file, verify_checksums, write_checksums
from .mechanistic_metrics import holm_adjust
from .phir_ch5 import _append_ledger, _snapshot_after_record
from .phir_instruments import ANTICHAINS, ATOM_NAMES, PHIR_ATOMS, advance_fission_traced
from .seeds import derive_seed
from .simulator import (
    FissionRecord,
    SimulationError,
    Snapshot,
    generate_beta,
    generate_initial_composition,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "phir_extension"
DOCUMENT = "CODEX_CH5_PHIR_PX8_PREREGISTRATION.md"
LEDGER = ROOT / "PHIR_RESULTS_LEDGER.md"

DEFAULT_VALIDATION = RESULT_ROOT / "px8_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "px8_registration"
DEFAULT_SMOKE = RESULT_ROOT / "px8_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "px8_high_support_resilience"
DEFAULT_WORK = RESULT_ROOT / ".px8_high_support_work"
DEFAULT_LOG = RESULT_ROOT / "px8_high_support_resilience.log"

MODEL_SOURCE = (
    ROOT
    / "results_intervention_replication"
    / "cr5_confirmation_registration"
    / "frozen_cr5_students.npz"
)
MODEL_CONTRACT_SOURCE = (
    ROOT
    / "results_intervention_replication"
    / "cr5_confirmation_registration"
    / "model_contract.json"
)

PROGRAM_FORMAT = "codex-ch5-phir-px8-high-support-v1"
REGISTRATION_FORMAT = "codex-ch5-phir-px8-registration-v1"
RESULT_FORMAT = "codex-ch5-phir-px8-result-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-px8-checkpoint-v1"
STATUS_FORMAT = "codex-ch5-phir-px8-status-v1"
LABEL = "CODEX_CH5_PHIR_PX8_HIGH_SUPPORT_V1"

MATRICES = 48
LANDMARKS = (20, 35, 50, 65, 80)
BRANCHES = 256
HALVES = {"A": tuple(range(0, 128)), "B": tuple(range(128, 256))}
SUPPORT_LEVELS = (16, 32, 64, 128)
PRIMARY_SUPPORT = 128
HORIZON = 8
ACQUISITION_LIMIT = 60
PAST_WINDOW = 512
MINIMUM_ELIGIBLE_MATRICES = 40
BOOTSTRAP_DRAWS = 4096
RANDOMIZATION_DRAWS = 4096
OUTCOME_EQUIVALENCE_MARGIN = 0.025
MINIMUM_FINITE_FRACTION = 0.95
MAX_WORKERS = 8
MAX_CPU_HOURS = 30.0
MINIMUM_FREE_DISK_BYTES = 1_500_000_000
SMOKE_MATRIX_ID = 1

ARMS = ("RENEWAL_UP", "RENEWAL_DOWN", "RANDOM", "NOOP")
STABILIZING = "RENEWAL_UP"
DESTABILIZING = "RENEWAL_DOWN"
TARGET_FORMULATION = "generational__beta__typeset"
NEGATIVE_CONTROL = "molecular__self__revised"

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/phir_extension_px8.py",
    "tests/test_phir_extension_px8.py",
    "plastic_heredity/phir_extension_px7.py",
    "plastic_heredity/phir_instruments.py",
    "plastic_heredity/phir_rescue_instruments.py",
    "plastic_heredity/intervention_cr5.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/simulator.py",
    "plastic_heredity/config.py",
    "plastic_heredity/seeds.py",
    "pyproject.toml",
    "requirements-lock.txt",
)


def _seed_domain(name: str) -> str:
    return hashlib.sha256(f"{LABEL}::{name}".encode("utf-8")).hexdigest()


SEED_DOMAINS = {
    name: _seed_domain(name)
    for name in (
        "matrix",
        "initial",
        "main_path",
        "acquisition",
        "selection",
        "random_action",
        "future",
        "bootstrap",
        "randomization",
        "replay",
        "validation",
        "smoke",
    )
}


def _json_ready(value: Any) -> Any:
    return px7._json_ready(value)


def _digest(value: Any) -> str:
    return px7._digest(value)


def _array_digest(*arrays: NDArray) -> str:
    return px7._array_digest(*arrays)


def _atomic_json(path: Path, value: Any) -> None:
    px7._atomic_json(path, value)


def _atomic_pickle(path: Path, value: Any) -> None:
    px7._atomic_pickle(path, value)


@dataclass(frozen=True)
class PX8Spec:
    label: str
    matrices: int
    landmarks: tuple[int, ...]
    branches: int
    horizon: int
    acquisition_limit: int
    bootstrap_draws: int
    randomization_draws: int


def scientific_spec() -> PX8Spec:
    return PX8Spec(
        "scientific",
        MATRICES,
        LANDMARKS,
        BRANCHES,
        HORIZON,
        ACQUISITION_LIMIT,
        BOOTSTRAP_DRAWS,
        RANDOMIZATION_DRAWS,
    )


def smoke_spec() -> PX8Spec:
    # All five landmarks make the I/O smoke robust to a no-break outcome at any
    # single non-scientific fixture state.
    return PX8Spec("smoke", 1, LANDMARKS, 16, 3, 60, 32, 32)


def _supports(spec: PX8Spec) -> tuple[int, ...]:
    if spec.branches == BRANCHES:
        return SUPPORT_LEVELS
    half = spec.branches // 2
    return tuple(sorted({max(1, half // 2), half}))


def _halves(spec: PX8Spec) -> dict[str, tuple[int, ...]]:
    midpoint = spec.branches // 2
    return {"A": tuple(range(midpoint)), "B": tuple(range(midpoint, spec.branches))}


def _seed(spec: PX8Spec, domain: str, *keys: object) -> int:
    selected = "smoke" if spec.label == "smoke" else domain
    return derive_seed(SEED_DOMAINS[selected], LABEL, spec.label, domain, *keys)


@dataclass(frozen=True)
class ResilienceCase:
    state_id: str
    candidate: str
    matrix_id: int
    landmark: int
    beta: NDArray[np.float64]
    snapshot: Snapshot
    history_counts: NDArray[np.int16]

    def as_state_case(self) -> StateCase:
        return StateCase(
            state_id=self.state_id,
            cohort="PX8_RESILIENCE",
            candidate=self.candidate,
            matrix_id=self.matrix_id,
            landmark=self.landmark,
            beta=self.beta,
            snapshot=self.snapshot,
        )


@dataclass(frozen=True)
class PairBlock:
    molecular_past: NDArray[np.int16]
    molecular_future: NDArray[np.int16]
    generational_past: NDArray[np.int16]
    generational_future: NDArray[np.int16]


@dataclass(frozen=True)
class PX8Batch:
    matrix_id: int
    beta: NDArray[np.float64]
    initial: NDArray[np.int16]
    acquisition_rows: tuple[dict[str, Any], ...]
    edit_rows: tuple[dict[str, Any], ...]
    branch_rows: tuple[dict[str, Any], ...]
    score_rows: tuple[dict[str, Any], ...]
    cpu_seconds: float
    scientific_digest: str


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def protocol(spec: PX8Spec | None = None) -> dict[str, Any]:
    spec = scientific_spec() if spec is None else spec
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "question": "high-support generational beta-typeset gauge of post-break resilience",
        "strict_eight_excluded": True,
        "predecessor_px7_immutable": True,
        "cohort": {
            "matrices": spec.matrices,
            "candidates": list(CANDIDATES),
            "landmarks": list(spec.landmarks),
            "branches_per_arm": spec.branches,
            "halves": {key: list(value) for key, value in _halves(spec).items()},
            "horizon": spec.horizon,
            "acquisition_limit": spec.acquisition_limit,
            "minimum_eligible_matrices": MINIMUM_ELIGIBLE_MATRICES,
        },
        "arms": list(ARMS),
        "target": {
            "formulation": TARGET_FORMULATION,
            "clock": "generational",
            "partition": "fixed beta Fiedler",
            "functional": "unnormalized full-dimensional whole-minus-parts",
            "support_branches": list(_supports(spec)),
            "primary_support": max(_supports(spec)),
        },
        "negative_control": {
            "formulation": NEGATIVE_CONTROL,
            "pass_path": False,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_draws": spec.bootstrap_draws,
            "randomization_draws": spec.randomization_draws,
            "outcome_equivalence_margin": OUTCOME_EQUIVALENCE_MARGIN,
            "minimum_finite_fraction": MINIMUM_FINITE_FRACTION,
            "primary_families": ["reliability", "forecast", "response"],
            "family_adjustment": "Holm",
        },
        "frozen_students": {
            "archive_sha256": sha256_file(MODEL_SOURCE),
            "contract_sha256": sha256_file(MODEL_CONTRACT_SOURCE),
            "refit_or_recalibration": False,
        },
        "randomness": {
            "seed_domains": SEED_DOMAINS,
            "arm_in_future_seed": False,
            "common_random_streams": True,
            "random_action_separate": True,
            "replacement": False,
        },
        "operational": {
            "workers_max": MAX_WORKERS,
            "cpu_hours_max": MAX_CPU_HOURS,
            "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
            "detached_science": True,
            "matrix_checkpointing": True,
            "complete_replay": True,
        },
    }
    value["protocol_id"] = _digest(value)
    return value


def _run_natural_candidate(
    matrix_id: int,
    beta: NDArray[np.float64],
    initial: NDArray[np.int16],
    candidate: str,
    spec: PX8Spec,
) -> list[ResilienceCase]:
    config = GardConfig()
    maximum = max(spec.landmarks)
    for attempt in range(100):
        rng = np.random.default_rng(
            _seed(spec, "main_path", candidate, matrix_id, attempt)
        )
        snapshot = Snapshot(initial.copy(), 0, (), ())
        observations: list[NDArray[np.int64]] = [initial.astype(np.int64).copy()]
        output: list[ResilienceCase] = []
        try:
            for generation in range(1, maximum + 1):
                traced = advance_fission_traced(
                    snapshot.composition,
                    beta,
                    config,
                    CANDIDATES[candidate],
                    rng,
                )
                observations.extend(
                    np.asarray(item, dtype=np.int64).copy()
                    for item in traced.growth_observations
                )
                observations.append(
                    np.asarray(traced.record.daughter, dtype=np.int64).copy()
                )
                snapshot = _snapshot_after_record(snapshot, traced.record)
                if generation in spec.landmarks:
                    output.append(
                        ResilienceCase(
                            state_id=(
                                f"PX8-c{candidate}-m{matrix_id:03d}-g{generation:03d}"
                            ),
                            candidate=candidate,
                            matrix_id=matrix_id,
                            landmark=generation,
                            beta=beta,
                            snapshot=snapshot,
                            history_counts=np.asarray(
                                observations[-PAST_WINDOW:], dtype=np.int16
                            ),
                        )
                    )
            if len(output) != len(spec.landmarks):
                raise AssertionError("PX8 natural path omitted a landmark")
            return output
        except SimulationError:
            continue
    raise SimulationError(
        f"PX8 failed bounded natural retry for c{candidate} m{matrix_id}"
    )


def _acquire_break(
    source: ResilienceCase, spec: PX8Spec
) -> tuple[ResilienceCase | None, dict[str, Any]]:
    config = GardConfig()
    rng = np.random.default_rng(
        _seed(spec, "acquisition", source.candidate, source.matrix_id, source.landmark)
    )
    snapshot = source.snapshot
    observations = [
        np.asarray(row, dtype=np.int64).copy() for row in source.history_counts
    ]
    for offset in range(1, spec.acquisition_limit + 1):
        try:
            traced = advance_fission_traced(
                snapshot.composition,
                source.beta,
                config,
                CANDIDATES[source.candidate],
                rng,
            )
        except SimulationError:
            return None, {
                "source_state_id": source.state_id,
                "candidate": source.candidate,
                "matrix_id": source.matrix_id,
                "landmark": source.landmark,
                "eligible": 0,
                "reason": "extinction_before_break",
                "observed_fissions": offset - 1,
            }
        observations.extend(
            np.asarray(item, dtype=np.int64).copy()
            for item in traced.growth_observations
        )
        observations.append(np.asarray(traced.record.daughter, dtype=np.int64).copy())
        snapshot = _snapshot_after_record(snapshot, traced.record)
        if traced.record.h <= config.inheritance_threshold:
            case = ResilienceCase(
                state_id=f"{source.state_id}-break-f{offset:02d}",
                candidate=source.candidate,
                matrix_id=source.matrix_id,
                landmark=source.landmark,
                beta=source.beta,
                snapshot=snapshot,
                history_counts=np.asarray(observations[-PAST_WINDOW:], dtype=np.int16),
            )
            return case, {
                "source_state_id": source.state_id,
                "broken_state_id": case.state_id,
                "candidate": source.candidate,
                "matrix_id": source.matrix_id,
                "landmark": source.landmark,
                "eligible": 1,
                "reason": "first_natural_break",
                "observed_fissions": offset,
                "break_h": float(traced.record.h),
                "broken_state_digest": _array_digest(case.snapshot.composition),
            }
    return None, {
        "source_state_id": source.state_id,
        "candidate": source.candidate,
        "matrix_id": source.matrix_id,
        "landmark": source.landmark,
        "eligible": 0,
        "reason": "no_break_within_limit",
        "observed_fissions": spec.acquisition_limit,
    }


def _selection_seed(spec: PX8Spec, case: ResilienceCase) -> int:
    return _seed(
        spec,
        "random_action",
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def _future_seed(spec: PX8Spec, case: ResilienceCase, branch: int) -> int:
    return _seed(
        spec,
        "future",
        case.candidate,
        case.matrix_id,
        case.landmark,
        branch,
    )


def _select_edits(
    case: ResilienceCase,
    students: Mapping[tuple[str, str], cr5.FrozenCR5Student],
    spec: PX8Spec,
) -> tuple[NDArray[np.float64], tuple[MolecularEdit | None, ...]]:
    config = GardConfig()
    student = students[("renewal", case.candidate)]
    noop, scores = cr5.score_student_edits(student, case.as_state_case(), config)
    return cr5.select_student_edits(
        noop,
        scores,
        np.random.default_rng(_selection_seed(spec, case)),
    )


def _records_digest(records: Iterable[FissionRecord]) -> str:
    return px7._records_digest(records)


def _first_run(values: Sequence[bool], length: int) -> int:
    return px7._first_run(values, length)


def _simulate_branch(
    case: ResilienceCase,
    edit: MolecularEdit | None,
    branch: int,
    spec: PX8Spec,
) -> tuple[dict[str, Any], PairBlock]:
    config = GardConfig()
    composition = (
        case.snapshot.composition
        if edit is None
        else apply_molecular_edit(case.snapshot.composition, edit)
    )
    snapshot = Snapshot(
        composition=np.asarray(composition, dtype=np.int64).copy(),
        generation=case.snapshot.generation,
        inheritance=case.snapshot.inheritance,
        boundary_h=case.snapshot.boundary_h,
        previous_growth_steps=case.snapshot.previous_growth_steps,
        cumulative_growth_steps=case.snapshot.cumulative_growth_steps,
    )
    rng = np.random.default_rng(_future_seed(spec, case, branch))
    molecular: list[NDArray[np.int64]] = [snapshot.composition.copy()]
    generational_past: list[NDArray[np.int64]] = []
    generational_future: list[NDArray[np.int64]] = []
    records: list[FissionRecord] = []
    for _step in range(spec.horizon):
        try:
            traced = advance_fission_traced(
                snapshot.composition,
                case.beta,
                config,
                CANDIDATES[case.candidate],
                rng,
            )
        except SimulationError:
            break
        molecular.extend(
            np.asarray(item, dtype=np.int64).copy()
            for item in traced.growth_observations
        )
        molecular.append(np.asarray(traced.record.daughter, dtype=np.int64).copy())
        generational_past.append(np.asarray(traced.record.parent, dtype=np.int64))
        generational_future.append(np.asarray(traced.record.daughter, dtype=np.int64))
        records.append(traced.record)
        snapshot = _snapshot_after_record(snapshot, traced.record)
    completed = len(records) == spec.horizon
    inherited = [record.h > config.inheritance_threshold for record in records]
    first_event = _first_run(inherited, 3)
    primary = int(first_event >= 0)
    molecular_array = np.asarray(molecular, dtype=np.int16)
    molecular_past = molecular_array[:-1]
    molecular_future = molecular_array[1:]
    if generational_past:
        generation_past = np.asarray(generational_past, dtype=np.int16)
        generation_future = np.asarray(generational_future, dtype=np.int16)
    else:
        generation_past = np.empty((0, config.n_types), dtype=np.int16)
        generation_future = np.empty((0, config.n_types), dtype=np.int16)
    row = {
        "state_id": case.state_id,
        "axis": "resilience",
        "candidate": case.candidate,
        "matrix_id": case.matrix_id,
        "landmark": case.landmark,
        "branch": branch,
        "half": "A" if branch < spec.branches // 2 else "B",
        "primary": primary,
        "completed": int(completed),
        "survived": int(completed),
        "inherited_fraction": float(sum(inherited) / spec.horizon),
        "first_event_time": int(first_event),
        "run5": int(_first_run(inherited, 5) >= 0),
        "record_digest": _records_digest(records),
        "rng_state_digest": _digest(rng.bit_generator.state),
        "molecular_pairs": int(molecular_past.shape[0]),
        "generational_pairs": int(generation_past.shape[0]),
    }
    return row, PairBlock(
        molecular_past,
        molecular_future,
        generation_past,
        generation_future,
    )


def _concatenate_pairs(
    blocks: Sequence[PairBlock], clock: str
) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
    if clock == "molecular":
        left = [block.molecular_past for block in blocks if len(block.molecular_past)]
        right = [
            block.molecular_future for block in blocks if len(block.molecular_future)
        ]
    elif clock == "generational":
        left = [
            block.generational_past for block in blocks if len(block.generational_past)
        ]
        right = [
            block.generational_future
            for block in blocks
            if len(block.generational_future)
        ]
    else:
        raise ValueError(clock)
    n_types = GardConfig().n_types
    return (
        np.concatenate(left, axis=0)
        if left
        else np.empty((0, n_types), dtype=np.int16),
        np.concatenate(right, axis=0)
        if right
        else np.empty((0, n_types), dtype=np.int16),
    )


def _score_row(
    case: ResilienceCase,
    arm: str,
    half: str,
    support: int,
    formulation: str,
    score: Mapping[str, Any],
) -> dict[str, Any]:
    if formulation == TARGET_FORMULATION:
        value = float(score["typeset"])
        reading = "eligible_target"
    elif formulation == NEGATIVE_CONTROL:
        value = float(score["revised"])
        reading = "negative_control"
    else:  # pragma: no cover
        raise ValueError(formulation)
    row: dict[str, Any] = {
        "state_id": case.state_id,
        "axis": "resilience",
        "candidate": case.candidate,
        "matrix_id": case.matrix_id,
        "landmark": case.landmark,
        "arm": arm,
        "source_half": half,
        "support_branches": support,
        "formulation": formulation,
        "reading_role": reading,
        "value": value,
        "revised": float(score["revised"]),
        "typeset": float(score["typeset"]),
        "ratio": float(score["ratio"]),
        "whole_mi": float(score["whole_mi"]),
        "causation": float(score["causation"]),
        "emergence": float(score["emergence"]),
        "synergy_persistence": float(score["synergy_persistence"]),
        "active_dimensions": int(score["active_dimensions"]),
        "part_a_dimensions": int(score["part_a_dimensions"]),
        "part_b_dimensions": int(score["part_b_dimensions"]),
        "transitions": int(score["transitions"]),
        "partition_digest": str(score["partition_digest"]),
    }
    row.update(
        {
            f"atom_{name}": float(value)
            for name, value in zip(ATOM_NAMES, score["atoms"], strict=True)
        }
    )
    return row


def _score_arm_halves(
    arm: str,
    case: ResilienceCase,
    blocks: Sequence[PairBlock],
    spec: PX8Spec,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    supports = _supports(spec)
    for half, indices in _halves(spec).items():
        half_blocks = [blocks[index] for index in indices]
        for support in supports:
            selected = half_blocks[:support]
            past, future = _concatenate_pairs(selected, "generational")
            target = px7._safe_score_pairs(past, future, "beta", case)
            output.append(
                _score_row(
                    case,
                    arm,
                    half,
                    support,
                    TARGET_FORMULATION,
                    target,
                )
            )
            if support == max(supports):
                past, future = _concatenate_pairs(selected, "molecular")
                control = px7._safe_score_pairs(past, future, "self", case)
                output.append(
                    _score_row(
                        case,
                        arm,
                        half,
                        support,
                        NEGATIVE_CONTROL,
                        control,
                    )
                )
    return output


def _run_case(
    case: ResilienceCase,
    students: Mapping[tuple[str, str], cr5.FrozenCR5Student],
    spec: PX8Spec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, edits = _select_edits(case, students, spec)
    branch_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    for arm, prediction, edit in zip(ARMS, predictions, edits, strict=True):
        edit_rows.append(
            {
                "state_id": case.state_id,
                "axis": "resilience",
                "candidate": case.candidate,
                "matrix_id": case.matrix_id,
                "landmark": case.landmark,
                "arm": arm,
                "prediction": float(prediction),
                "remove_type": -1 if edit is None else edit.remove_type,
                "add_type": -1 if edit is None else edit.add_type,
                "history_digest": _array_digest(case.history_counts),
            }
        )
        blocks: list[PairBlock] = []
        for branch in range(spec.branches):
            row, block = _simulate_branch(case, edit, branch, spec)
            row.update({"arm": arm, "prediction": float(prediction)})
            branch_rows.append(row)
            blocks.append(block)
        score_rows.extend(_score_arm_halves(arm, case, blocks, spec))
    return branch_rows, score_rows, edit_rows


def _matrix_digest(
    matrix_id: int,
    beta: NDArray,
    initial: NDArray,
    acquisition_rows: Sequence[dict[str, Any]],
    edit_rows: Sequence[dict[str, Any]],
    branch_rows: Sequence[dict[str, Any]],
    score_rows: Sequence[dict[str, Any]],
) -> str:
    return _digest(
        {
            "matrix_id": matrix_id,
            "beta": _array_digest(beta),
            "initial": _array_digest(initial),
            "acquisition": acquisition_rows,
            "edits": edit_rows,
            "branches": branch_rows,
            "scores": score_rows,
        }
    )


def _run_matrix(arguments: tuple[int, PX8Spec, str, str]) -> PX8Batch:
    matrix_id, spec, model_path, contract_path = arguments
    started = time.process_time()
    with threadpool_limits(limits=1):
        config = GardConfig()
        beta = generate_beta(
            config, np.random.default_rng(_seed(spec, "matrix", matrix_id))
        )
        initial = generate_initial_composition(
            config, np.random.default_rng(_seed(spec, "initial", matrix_id))
        ).astype(np.int16)
        students = cr5.load_students(Path(model_path), Path(contract_path))
        natural: list[ResilienceCase] = []
        for candidate in CANDIDATES:
            natural.extend(
                _run_natural_candidate(matrix_id, beta, initial, candidate, spec)
            )
        cases: list[ResilienceCase] = []
        acquisition_rows: list[dict[str, Any]] = []
        for source in natural:
            broken, acquisition = _acquire_break(source, spec)
            acquisition_rows.append(acquisition)
            if broken is not None:
                cases.append(broken)
        branch_rows: list[dict[str, Any]] = []
        score_rows: list[dict[str, Any]] = []
        edit_rows: list[dict[str, Any]] = []
        for case in cases:
            branches, scores, edits = _run_case(case, students, spec)
            branch_rows.extend(branches)
            score_rows.extend(scores)
            edit_rows.extend(edits)
        scientific_digest = _matrix_digest(
            matrix_id,
            beta,
            initial,
            acquisition_rows,
            edit_rows,
            branch_rows,
            score_rows,
        )
        return PX8Batch(
            matrix_id=matrix_id,
            beta=beta,
            initial=initial,
            acquisition_rows=tuple(acquisition_rows),
            edit_rows=tuple(edit_rows),
            branch_rows=tuple(branch_rows),
            score_rows=tuple(score_rows),
            cpu_seconds=float(time.process_time() - started),
            scientific_digest=scientific_digest,
        )


def _bootstrap_summary(
    series: pd.Series,
    repetitions: int,
    key: str,
    arrays: dict[str, NDArray],
) -> dict[str, Any]:
    local = series.dropna().sort_index()
    values = np.asarray(local, dtype=np.float64)
    matrix_ids = np.asarray(local.index, dtype=np.int64)
    safe = key.replace("/", "__")
    if not values.size:
        return {
            "effect": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "ci90": [float("nan"), float("nan")],
            "one_sided_p": float("nan"),
            "matrices": 0,
            "matrices_positive": 0,
        }
    rng = np.random.default_rng(_seed(scientific_spec(), "bootstrap", key))
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    bootstrap = values[indices].mean(axis=1)
    sign_rng = np.random.default_rng(_seed(scientific_spec(), "randomization", key))
    signs = sign_rng.choice((-1.0, 1.0), size=(repetitions, values.size))
    randomized = (signs * values).mean(axis=1)
    observed = float(values.mean())
    result = {
        "effect": observed,
        "ci95": [float(item) for item in np.quantile(bootstrap, (0.025, 0.975))],
        "ci90": [float(item) for item in np.quantile(bootstrap, (0.05, 0.95))],
        "one_sided_p": float(
            (1 + np.count_nonzero(randomized >= observed)) / (repetitions + 1)
        ),
        "matrices": int(values.size),
        "matrices_positive": int(np.count_nonzero(values > 0)),
        "maximum_absolute_matrix_effect": float(np.max(np.abs(values))),
    }
    arrays[f"{safe}__matrix_ids"] = matrix_ids
    arrays[f"{safe}__matrix_values"] = values
    arrays[f"{safe}__bootstrap"] = bootstrap
    return result


def _matrix_arm_effect(
    frame: pd.DataFrame,
    value: str,
    high: str,
    low: str,
    filters: Mapping[str, Any],
) -> pd.Series:
    return px7._matrix_arm_effect(frame, value, high, low, filters)


def _matrix_centered_spearman(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    return px7._matrix_centered_spearman(frame, left, right)


def _adjust_family(items: list[dict[str, Any]]) -> None:
    adjusted = holm_adjust([float(item["one_sided_p"]) for item in items])
    for item, value in zip(items, adjusted, strict=True):
        item["holm_adjusted_p"] = float(value)
        item["pass"] = bool(
            item["effect"] > 0
            and item["ci95"][0] > 0
            and item["holm_adjusted_p"] < 0.05
        )


def _outcome_frame(branches: pd.DataFrame) -> pd.DataFrame:
    return px7._outcome_frame(branches)


def _outcome_validity(
    outcomes: pd.DataFrame,
    spec: PX8Spec,
    arrays: dict[str, NDArray],
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for half in ("A", "B"):
            filters = {
                "axis": "resilience",
                "candidate": candidate,
                "half": half,
            }
            target = _matrix_arm_effect(
                outcomes,
                "q",
                STABILIZING,
                DESTABILIZING,
                filters,
            )
            item = _bootstrap_summary(
                target,
                spec.bootstrap_draws,
                f"outcome/c{candidate}/h{half}/targeted",
                arrays,
            )
            random = _matrix_arm_effect(
                outcomes,
                "q",
                "RANDOM",
                "NOOP",
                filters,
            )
            random_summary = _bootstrap_summary(
                random,
                spec.bootstrap_draws,
                f"outcome/c{candidate}/h{half}/random",
                arrays,
            )
            item.update(
                {
                    "candidate": candidate,
                    "target_half": half,
                    "random_minus_noop": random_summary,
                    "random_tost": bool(
                        random_summary["ci90"][0] > -OUTCOME_EQUIVALENCE_MARGIN
                        and random_summary["ci90"][1] < OUTCOME_EQUIVALENCE_MARGIN
                    ),
                }
            )
            rows.append(item)
    _adjust_family(rows)
    for item in rows:
        item["pass"] = bool(item["pass"] and item["random_tost"])
    return rows, bool(len(rows) == 4 and all(item["pass"] for item in rows))


def _merge_forecast(
    scores: pd.DataFrame,
    outcomes: pd.DataFrame,
    candidate: str,
    formulation: str,
    support: int,
    source_half: str,
) -> pd.DataFrame:
    target_half = "B" if source_half == "A" else "A"
    phi = scores[
        (scores["candidate"] == candidate)
        & (scores["formulation"] == formulation)
        & (scores["support_branches"] == support)
        & (scores["source_half"] == source_half)
    ][["matrix_id", "state_id", "landmark", "arm", "value"]]
    target = outcomes[
        (outcomes["candidate"] == candidate) & (outcomes["half"] == target_half)
    ][
        [
            "matrix_id",
            "state_id",
            "landmark",
            "arm",
            "q",
            "prediction",
            "successes",
            "trials",
        ]
    ]
    return phi.merge(
        target,
        on=["matrix_id", "state_id", "landmark", "arm"],
        how="inner",
        validate="one_to_one",
    )


def _arm_centered_spearman(frame: pd.DataFrame) -> pd.Series:
    selected = frame.copy()
    selected["value_centered"] = selected["value"] - selected.groupby(
        ["matrix_id", "arm"], sort=False
    )["value"].transform("mean")
    selected["q_centered"] = selected["q"] - selected.groupby(
        ["matrix_id", "arm"], sort=False
    )["q"].transform("mean")
    return _matrix_centered_spearman(selected, "value_centered", "q_centered")


def _effect_forecast(frame: pd.DataFrame) -> pd.Series:
    phi = frame.pivot_table(
        index=["matrix_id", "state_id", "landmark"],
        columns="arm",
        values="value",
        aggfunc="first",
    )
    outcome = frame.pivot_table(
        index=["matrix_id", "state_id", "landmark"],
        columns="arm",
        values="q",
        aggfunc="first",
    )
    required = {STABILIZING, DESTABILIZING}
    if not required.issubset(phi.columns) or not required.issubset(outcome.columns):
        return pd.Series(dtype=float)
    joined = pd.DataFrame(
        {
            "phi_delta": phi[STABILIZING] - phi[DESTABILIZING],
            "q_delta": outcome[STABILIZING] - outcome[DESTABILIZING],
        }
    ).reset_index()
    return _matrix_centered_spearman(joined, "phi_delta", "q_delta")


def _crossfit_incremental_gain(frame: pd.DataFrame) -> pd.Series:
    return px7._crossfit_incremental_gain(frame, "resilience")


def _primary_information_tests(
    scores: pd.DataFrame,
    outcomes: pd.DataFrame,
    spec: PX8Spec,
    arrays: dict[str, NDArray],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    reliability_items: list[dict[str, Any]] = []
    forecast_items: list[dict[str, Any]] = []
    response_items: list[dict[str, Any]] = []
    arm_centered_items: list[dict[str, Any]] = []
    effect_forecast_items: list[dict[str, Any]] = []
    support = max(_supports(spec))
    for candidate in CANDIDATES:
        local = scores[
            (scores["candidate"] == candidate)
            & (scores["formulation"] == TARGET_FORMULATION)
            & (scores["support_branches"] == support)
        ]
        paired = local.pivot_table(
            index=["matrix_id", "state_id", "landmark", "arm"],
            columns="source_half",
            values="value",
            aggfunc="first",
        ).reset_index()
        reliability = (
            _matrix_centered_spearman(paired, "A", "B")
            if {"A", "B"}.issubset(paired.columns)
            else pd.Series(dtype=float)
        )
        item = _bootstrap_summary(
            reliability,
            spec.bootstrap_draws,
            f"reliability/c{candidate}",
            arrays,
        )
        item.update({"family": "reliability", "candidate": candidate})
        reliability_items.append(item)
        for source_half in ("A", "B"):
            direction = f"{source_half}_to_{'B' if source_half == 'A' else 'A'}"
            merged = _merge_forecast(
                scores,
                outcomes,
                candidate,
                TARGET_FORMULATION,
                support,
                source_half,
            )
            forecast = _matrix_centered_spearman(merged, "value", "q")
            forecast_item = _bootstrap_summary(
                forecast,
                spec.bootstrap_draws,
                f"forecast/c{candidate}/{direction}",
                arrays,
            )
            forecast_item.update(
                {
                    "family": "forecast",
                    "candidate": candidate,
                    "direction": direction,
                }
            )
            forecast_items.append(forecast_item)
            centered = _arm_centered_spearman(merged)
            centered_item = _bootstrap_summary(
                centered,
                spec.bootstrap_draws,
                f"arm_centered/c{candidate}/{direction}",
                arrays,
            )
            centered_item.update({"candidate": candidate, "direction": direction})
            arm_centered_items.append(centered_item)
            effect = _effect_forecast(merged)
            effect_item = _bootstrap_summary(
                effect,
                spec.bootstrap_draws,
                f"effect_forecast/c{candidate}/{direction}",
                arrays,
            )
            effect_item.update({"candidate": candidate, "direction": direction})
            effect_forecast_items.append(effect_item)
            response = _matrix_arm_effect(
                scores,
                "value",
                STABILIZING,
                DESTABILIZING,
                {
                    "candidate": candidate,
                    "formulation": TARGET_FORMULATION,
                    "support_branches": support,
                    "source_half": source_half,
                },
            )
            response_item = _bootstrap_summary(
                response,
                spec.bootstrap_draws,
                f"response/c{candidate}/h{source_half}",
                arrays,
            )
            response_item.update(
                {
                    "family": "response",
                    "candidate": candidate,
                    "source_half": source_half,
                }
            )
            response_items.append(response_item)
    _adjust_family(reliability_items)
    _adjust_family(forecast_items)
    _adjust_family(response_items)
    return (
        reliability_items,
        forecast_items,
        response_items,
        arm_centered_items,
        effect_forecast_items,
    )


def _support_analysis(
    scores: pd.DataFrame,
    spec: PX8Spec,
    arrays: dict[str, NDArray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    response_rows: list[dict[str, Any]] = []
    finite_rows: list[dict[str, Any]] = []
    for support in _supports(spec):
        for candidate in CANDIDATES:
            for half in ("A", "B"):
                filters = {
                    "candidate": candidate,
                    "formulation": TARGET_FORMULATION,
                    "support_branches": support,
                    "source_half": half,
                }
                response = _matrix_arm_effect(
                    scores,
                    "value",
                    STABILIZING,
                    DESTABILIZING,
                    filters,
                )
                item = _bootstrap_summary(
                    response,
                    spec.bootstrap_draws,
                    f"support/s{support}/c{candidate}/h{half}",
                    arrays,
                )
                item.update(
                    {
                        "support_branches": support,
                        "candidate": candidate,
                        "source_half": half,
                    }
                )
                response_rows.append(item)
                local = scores.copy()
                for column, expected in filters.items():
                    local = local[local[column] == expected]
                finite_rows.append(
                    {
                        "support_branches": support,
                        "candidate": candidate,
                        "source_half": half,
                        "finite_fraction": float(
                            np.isfinite(local["value"].to_numpy(float)).mean()
                        )
                        if len(local)
                        else 0.0,
                        "median_transitions": float(local["transitions"].median())
                        if len(local)
                        else float("nan"),
                    }
                )
    lookup = {
        (item["support_branches"], item["candidate"], item["source_half"]): item
        for item in response_rows
    }
    cells = [(candidate, half) for candidate in CANDIDATES for half in ("A", "B")]
    stable_sign = bool(
        all(
            lookup[(64, candidate, half)]["effect"] > 0
            and lookup[(128, candidate, half)]["effect"] > 0
            for candidate, half in cells
        )
    )
    first_steps = np.asarray(
        [
            abs(
                lookup[(64, candidate, half)]["effect"]
                - lookup[(32, candidate, half)]["effect"]
            )
            for candidate, half in cells
        ],
        dtype=np.float64,
    )
    final_steps = np.asarray(
        [
            abs(
                lookup[(128, candidate, half)]["effect"]
                - lookup[(64, candidate, half)]["effect"]
            )
            for candidate, half in cells
        ],
        dtype=np.float64,
    )
    finite_lookup = {
        (item["support_branches"], item["candidate"], item["source_half"]): item
        for item in finite_rows
    }
    finite_pass = bool(
        all(
            finite_lookup[(support, candidate, half)]["finite_fraction"]
            >= MINIMUM_FINITE_FRACTION
            for support in (64, 128)
            for candidate, half in cells
        )
    )
    convergence = {
        "finite_pass": finite_pass,
        "positive_at_64_and_128_all_cells": stable_sign,
        "median_abs_change_32_to_64": float(np.median(first_steps)),
        "median_abs_change_64_to_128": float(np.median(final_steps)),
        "contracting_final_step": bool(np.median(final_steps) < np.median(first_steps)),
    }
    convergence["pass"] = bool(
        convergence["finite_pass"]
        and convergence["positive_at_64_and_128_all_cells"]
        and convergence["contracting_final_step"]
    )
    return response_rows, finite_rows, convergence


def _secondary_reading_analysis(
    scores: pd.DataFrame,
    outcomes: pd.DataFrame,
    spec: PX8Spec,
    arrays: dict[str, NDArray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    support = max(_supports(spec))
    controls: list[dict[str, Any]] = []
    atoms: list[dict[str, Any]] = []
    incremental: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for half in ("A", "B"):
            response = _matrix_arm_effect(
                scores,
                "value",
                STABILIZING,
                DESTABILIZING,
                {
                    "candidate": candidate,
                    "formulation": NEGATIVE_CONTROL,
                    "support_branches": support,
                    "source_half": half,
                },
            )
            item = _bootstrap_summary(
                response,
                spec.bootstrap_draws,
                f"negative_control/response/c{candidate}/h{half}",
                arrays,
            )
            item.update(
                {
                    "family": "response",
                    "candidate": candidate,
                    "source_half": half,
                }
            )
            controls.append(item)
            for metric in ("causation", "emergence", "synergy_persistence", "revised"):
                vector = _matrix_arm_effect(
                    scores,
                    metric,
                    STABILIZING,
                    DESTABILIZING,
                    {
                        "candidate": candidate,
                        "formulation": TARGET_FORMULATION,
                        "support_branches": support,
                        "source_half": half,
                    },
                )
                atom_item = _bootstrap_summary(
                    vector,
                    spec.bootstrap_draws,
                    f"mechanism/{metric}/c{candidate}/h{half}",
                    arrays,
                )
                atom_item.update(
                    {
                        "metric": metric,
                        "candidate": candidate,
                        "source_half": half,
                    }
                )
                atoms.append(atom_item)
            direction = f"{half}_to_{'B' if half == 'A' else 'A'}"
            merged = _merge_forecast(
                scores,
                outcomes,
                candidate,
                TARGET_FORMULATION,
                support,
                half,
            )
            gain = _crossfit_incremental_gain(merged)
            gain_item = _bootstrap_summary(
                gain,
                spec.bootstrap_draws,
                f"incremental/c{candidate}/{direction}",
                arrays,
            )
            gain_item.update({"candidate": candidate, "direction": direction})
            incremental.append(gain_item)
    _adjust_family(controls)
    _adjust_family(atoms)
    return controls, atoms, incremental


def analyze_batches(
    batches: Sequence[PX8Batch], spec: PX8Spec
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    acquisitions = pd.DataFrame(
        [row for batch in batches for row in batch.acquisition_rows]
    )
    edits = pd.DataFrame([row for batch in batches for row in batch.edit_rows])
    branches = pd.DataFrame([row for batch in batches for row in batch.branch_rows])
    scores = pd.DataFrame([row for batch in batches for row in batch.score_rows])
    outcomes = _outcome_frame(branches)
    arrays: dict[str, NDArray] = {}
    outcome_rows, manipulation_valid = _outcome_validity(outcomes, spec, arrays)

    eligibility: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        local = acquisitions[
            (acquisitions["candidate"] == candidate) & (acquisitions["eligible"] == 1)
        ]
        matrices = int(local["matrix_id"].nunique())
        eligibility[candidate] = {
            "eligible_states": int(len(local)),
            "eligible_matrices": matrices,
            "minimum": MINIMUM_ELIGIBLE_MATRICES,
            "pass": matrices >= MINIMUM_ELIGIBLE_MATRICES,
        }
    eligibility_pass = bool(all(item["pass"] for item in eligibility.values()))

    (
        reliability,
        forecast,
        response,
        arm_centered,
        effect_forecast,
    ) = _primary_information_tests(scores, outcomes, spec, arrays)
    support_response, support_finite, convergence = _support_analysis(
        scores, spec, arrays
    )
    negative_control, mechanisms, incremental = _secondary_reading_analysis(
        scores, outcomes, spec, arrays
    )

    primary_pass = bool(
        len(reliability) == 2
        and len(forecast) == 4
        and len(response) == 4
        and all(item["pass"] for item in reliability)
        and all(item["pass"] for item in forecast)
        and all(item["pass"] for item in response)
    )
    gates = {
        "resilience_eligibility": eligibility_pass,
        "resilience_manipulation_valid": manipulation_valid,
        "estimator_support_converged": bool(convergence["pass"]),
        "primary_information_tests_pass": primary_pass,
        "specialized_resilience_gauge_confirmed": bool(
            eligibility_pass
            and manipulation_valid
            and convergence["pass"]
            and primary_pass
        ),
        "public_revised_negative_control_positive_response_all_cells": bool(
            len(negative_control) == 4
            and all(item["pass"] for item in negative_control)
        ),
    }
    matrix_rows: list[dict[str, Any]] = []
    for key, array in arrays.items():
        if key.endswith("__matrix_values"):
            ids_key = key.replace("__matrix_values", "__matrix_ids")
            if ids_key in arrays:
                family = key.removesuffix("__matrix_values")
                matrix_rows.extend(
                    {
                        "family": family,
                        "matrix_id": int(matrix_id),
                        "value": float(value),
                    }
                    for matrix_id, value in zip(arrays[ids_key], array, strict=True)
                )
    metrics = {
        "format": "codex-ch5-phir-px8-primary-metrics-v1",
        "eligibility": eligibility,
        "outcome_validity": outcome_rows,
        "reliability": reliability,
        "forecast": forecast,
        "response": response,
        "support_response": support_response,
        "support_finite": support_finite,
        "support_convergence": convergence,
        "arm_centered_forecast": arm_centered,
        "causal_effect_forecast": effect_forecast,
        "negative_control": negative_control,
        "mechanism_readings": mechanisms,
        "incremental_log_loss": incremental,
        "gates": gates,
    }
    tables = {
        "acquisition": acquisitions,
        "selected_edits": edits,
        "branches": branches,
        "state_outcomes": outcomes,
        "state_scores": scores,
        "matrix_effects": pd.DataFrame(matrix_rows),
    }
    return metrics, tables, arrays


def _validation_fixture() -> dict[str, Any]:
    spec = smoke_spec()
    rng = np.random.default_rng(_seed(spec, "validation", "fixture"))
    counts = rng.poisson(2.0, size=(1025, GardConfig().n_types)).astype(np.int16)
    counts[counts.sum(axis=1) == 0, 0] = 1
    beta = np.exp(rng.normal(-4.0, 1.0, size=(100, 100)))
    snapshot = Snapshot(counts[-1].astype(np.int64), 20, (True,) * 20, (0.95,) * 20)
    case = ResilienceCase(
        "PX8-fixture",
        "02",
        0,
        20,
        beta,
        snapshot,
        counts[-PAST_WINDOW:],
    )
    score = px7._score_pairs(counts[:-1], counts[1:], "beta", case)
    atom_lookup = {
        atom: float(value)
        for atom, value in zip(
            ((source, target) for source in ANTICHAINS for target in ANTICHAINS),
            score["atoms"],
            strict=True,
        )
    }
    block = PairBlock(
        counts[:8],
        counts[1:9],
        counts[:8],
        counts[1:9],
    )
    concatenated = _concatenate_pairs([block, block], "generational")
    checks = {
        "one_eligible_formulation": TARGET_FORMULATION == "generational__beta__typeset",
        "negative_control_cannot_win": NEGATIVE_CONTROL == "molecular__self__revised",
        "scientific_support_ladder_exact": _supports(scientific_spec())
        == SUPPORT_LEVELS,
        "halves_disjoint_complete": set(HALVES["A"]).isdisjoint(HALVES["B"])
        and set(HALVES["A"]) | set(HALVES["B"]) == set(range(BRANCHES)),
        "support_nested": all(
            set(HALVES["A"][:left]).issubset(HALVES["A"][:right])
            for left, right in zip(SUPPORT_LEVELS[:-1], SUPPORT_LEVELS[1:], strict=True)
        ),
        "target_score_finite": np.isfinite(score["typeset"]),
        "target_partition_nonempty": score["part_a_dimensions"] > 0
        and score["part_b_dimensions"] > 0,
        "target_transition_count": score["transitions"] == 1024,
        "nine_atom_identity": abs(
            score["revised"] - sum(atom_lookup[atom] for atom in PHIR_ATOMS)
        )
        < 1e-12,
        "explicit_blocks_not_joined": concatenated[0].shape[0] == 16
        and concatenated[1].shape[0] == 16,
        "future_seed_arm_free": "arm" not in inspect.signature(_future_seed).parameters,
        "future_and_action_streams_distinct": _future_seed(spec, case, 0)
        != _selection_seed(spec, case),
        "px8_seed_distinct_from_px7": _future_seed(spec, case, 0)
        != px7._seed(px7.smoke_spec(), "future", "resilience", "02", 0, 20, 0),
        "strict_threshold": not bool(0.9 > GardConfig().inheritance_threshold)
        and bool(np.nextafter(0.9, 1.0) > GardConfig().inheritance_threshold),
        "model_sources_exist": MODEL_SOURCE.is_file()
        and MODEL_CONTRACT_SOURCE.is_file(),
        "disk_available": shutil.disk_usage(ROOT).free >= MINIMUM_FREE_DISK_BYTES,
    }
    return {"checks": checks, "score": _json_ready(score)}


def validate(output: Path = DEFAULT_VALIDATION) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    fixture = _validation_fixture()
    first = cr5.load_students(MODEL_SOURCE, MODEL_CONTRACT_SOURCE)
    second = cr5.load_students(MODEL_SOURCE, MODEL_CONTRACT_SOURCE)
    fixture["checks"]["all_four_students_present"] = set(first) == {
        ("break", "02"),
        ("break", "03"),
        ("renewal", "02"),
        ("renewal", "03"),
    }
    fixture["checks"]["student_serialization_exact"] = bool(
        set(first) == set(second)
        and all(
            np.array_equal(first[key].coefficient, second[key].coefficient)
            and first[key].intercept == second[key].intercept
            for key in first
        )
    )
    payload = {
        "format": "codex-ch5-phir-px8-validation-v1",
        "checks": fixture["checks"],
        "all_checks_passed": bool(all(fixture["checks"].values())),
        "source_hashes": _source_hashes(),
        "model_sha256": sha256_file(MODEL_SOURCE),
        "model_contract_sha256": sha256_file(MODEL_CONTRACT_SOURCE),
        "fixture_score_digest": _digest(fixture["score"]),
    }
    if not payload["all_checks_passed"]:
        raise AssertionError(
            [key for key, value in payload["checks"].items() if not value]
        )
    output.mkdir(parents=True)
    _atomic_json(output / "validation.json", payload)
    write_checksums(output)
    verify_checksums(output)
    return payload


def register(
    validation_directory: Path = DEFAULT_VALIDATION,
    output: Path = DEFAULT_REGISTRATION,
) -> dict[str, Any]:
    verify_checksums(validation_directory)
    validation = json.loads(
        (validation_directory / "validation.json").read_text(encoding="utf-8")
    )
    if not validation["all_checks_passed"]:
        raise ValueError("PX8 validation did not pass")
    if validation["source_hashes"] != _source_hashes():
        raise ValueError("PX8 source changed after validation")
    if output.exists() or DEFAULT_OUTPUT.exists() or DEFAULT_WORK.exists():
        raise FileExistsError("PX8 registration, work, or output already exists")
    payload: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol": protocol(),
        "source_hashes": _source_hashes(),
        "validation_sha256": sha256_file(validation_directory / "validation.json"),
        "model_sha256": sha256_file(MODEL_SOURCE),
        "model_contract_sha256": sha256_file(MODEL_CONTRACT_SOURCE),
        "scientific_matrices_generated_at_registration": 0,
        "scientific_outcomes_generated_at_registration": 0,
        "created_at_unix": time.time(),
    }
    payload["registration_id"] = _digest(payload)
    output.mkdir(parents=True)
    shutil.copy2(ROOT / DOCUMENT, output / "preregistration.md")
    shutil.copy2(MODEL_SOURCE, output / "frozen_cr5_students.npz")
    shutil.copy2(MODEL_CONTRACT_SOURCE, output / "model_contract.json")
    _atomic_json(output / "protocol.json", payload["protocol"])
    _atomic_json(output / "seed_registry.json", SEED_DOMAINS)
    _atomic_json(output / "registration.json", payload)
    write_checksums(output)
    verify_registration(output)
    _append_ledger(
        f"<!-- phir-extension-px8-registration-{payload['registration_id']} -->",
        [
            "## Phi-r extension PX8 registered",
            "",
            f"- Registration: `{payload['registration_id']}`.",
            "- Forty-eight fresh matrices, shared-break resilience, one eligible generational beta-typeset reading, and 16/32/64/128-source support were sealed.",
            "- The public revised score is a non-winning negative control; no PX8 scientific outcome existed at registration.",
        ],
    )
    return payload


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    if payload.get("format") != REGISTRATION_FORMAT:
        raise ValueError("unsupported PX8 registration")
    if payload["source_hashes"] != _source_hashes():
        raise ValueError("PX8 source changed after registration")
    if payload["protocol"] != protocol():
        raise ValueError("PX8 protocol changed after registration")
    expected = _digest(
        {key: value for key, value in payload.items() if key != "registration_id"}
    )
    if payload["registration_id"] != expected:
        raise ValueError("PX8 registration ID mismatch")
    if sha256_file(directory / "frozen_cr5_students.npz") != payload["model_sha256"]:
        raise ValueError("PX8 frozen model archive changed")
    if (
        sha256_file(directory / "model_contract.json")
        != payload["model_contract_sha256"]
    ):
        raise ValueError("PX8 frozen model contract changed")
    return payload


def smoke(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_SMOKE,
) -> dict[str, Any]:
    registration = verify_registration(registration_directory)
    if output.exists():
        raise FileExistsError(output)
    spec = smoke_spec()
    arguments = (
        SMOKE_MATRIX_ID,
        spec,
        str(registration_directory / "frozen_cr5_students.npz"),
        str(registration_directory / "model_contract.json"),
    )
    first = _run_matrix(arguments)
    second = _run_matrix(arguments)
    smoke_fissions = max(1, len(first.branch_rows) * spec.horizon)
    projected_fissions_per_matrix = 2 * len(LANDMARKS) * len(ARMS) * BRANCHES * HORIZON
    projected_cpu_hours = float(
        first.cpu_seconds
        * projected_fissions_per_matrix
        / smoke_fissions
        * MATRICES
        * 2
        / 3600.0
    )
    score_formulations = {row["formulation"] for row in first.score_rows}
    payload = {
        "format": "codex-ch5-phir-px8-smoke-v1",
        "registration_id": registration["registration_id"],
        "exact_replay": first.scientific_digest == second.scientific_digest,
        "branches_created": len(first.branch_rows),
        "score_rows_created": len(first.score_rows),
        "both_readings_exercised": score_formulations
        == {TARGET_FORMULATION, NEGATIVE_CONTROL},
        "support_levels_exercised": sorted(
            {int(row["support_branches"]) for row in first.score_rows}
        ),
        "cpu_seconds_per_smoke_matrix": first.cpu_seconds,
        "projected_complete_cpu_hours_upper_bound": projected_cpu_hours,
        "projected_within_cpu_ceiling": projected_cpu_hours <= MAX_CPU_HOURS,
        "scientific_effects_disclosed": False,
    }
    if not (
        payload["exact_replay"]
        and payload["branches_created"] > 0
        and payload["score_rows_created"] > 0
        and payload["both_readings_exercised"]
        and payload["projected_within_cpu_ceiling"]
    ):
        raise AssertionError("PX8 smoke gate failed")
    output.mkdir(parents=True)
    _atomic_json(output / "smoke.json", payload)
    write_checksums(output)
    verify_checksums(output)
    return payload


def _checkpoint_contract(
    registration_id: str, spec: PX8Spec, stage: str
) -> dict[str, Any]:
    value = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "stage": stage,
        "spec": _json_ready(spec.__dict__),
        "source_hashes": _source_hashes(),
        "model_sha256": sha256_file(DEFAULT_REGISTRATION / "frozen_cr5_students.npz"),
    }
    value["contract_id"] = _digest(value)
    return value


def _prepare_work(
    work: Path,
    registration_id: str,
    spec: PX8Spec,
    cpu_budget_hours: float,
) -> None:
    work.mkdir(parents=True, exist_ok=True)
    contract_path = work / "checkpoint_contract.json"
    expected = _checkpoint_contract(registration_id, spec, "generation")
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != expected:
            raise ValueError("PX8 checkpoint contract mismatch")
    else:
        _atomic_json(contract_path, expected)
    budget_path = work / "cpu_budget.json"
    budget = {
        "maximum_cpu_hours": MAX_CPU_HOURS,
        "declared_cpu_hours": cpu_budget_hours,
    }
    if cpu_budget_hours <= 0 or cpu_budget_hours > MAX_CPU_HOURS:
        raise ValueError("PX8 CPU budget must be in (0, 30]")
    if budget_path.exists():
        if json.loads(budget_path.read_text(encoding="utf-8")) != budget:
            raise ValueError("PX8 CPU budget changed after launch")
    else:
        _atomic_json(budget_path, budget)


def _status_write(work: Path, payload: dict[str, Any]) -> None:
    _atomic_json(
        work / "status.json",
        {"format": STATUS_FORMAT, "updated_at_unix": time.time(), **payload},
    )


def _load_checkpoint(path: Path) -> PX8Batch:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, PX8Batch):
        raise TypeError(f"unexpected PX8 checkpoint type: {path}")
    return value


def _run_checkpoint_stage(
    spec: PX8Spec,
    directory: Path,
    workers: int,
    model_path: Path,
    contract_path: Path,
    work: Path,
    stage: str,
    cpu_budget_seconds: float,
) -> list[PX8Batch]:
    directory.mkdir(parents=True, exist_ok=True)
    batches: list[PX8Batch | None] = [None] * spec.matrices
    for matrix_id in range(spec.matrices):
        path = directory / f"matrix_{matrix_id:03d}.pkl"
        if path.exists():
            batch = _load_checkpoint(path)
            if batch.matrix_id != matrix_id:
                raise ValueError("PX8 checkpoint matrix ID mismatch")
            batches[matrix_id] = batch
    missing = [index for index, batch in enumerate(batches) if batch is None]
    consumed = float(sum(batch.cpu_seconds for batch in batches if batch is not None))
    _status_write(
        work,
        {
            "state": "running",
            "stage": stage,
            "completed_matrices": spec.matrices - len(missing),
            "total_matrices": spec.matrices,
            "cpu_seconds": consumed,
        },
    )
    arguments = [
        (index, spec, str(model_path), str(contract_path)) for index in missing
    ]
    if missing:
        if workers == 1:
            generated: Iterable[PX8Batch] = map(_run_matrix, arguments)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            generated = executor.map(_run_matrix, arguments, chunksize=1)
        try:
            for index, batch in zip(missing, generated, strict=True):
                batches[index] = batch
                _atomic_pickle(directory / f"matrix_{index:03d}.pkl", batch)
                consumed += batch.cpu_seconds
                _status_write(
                    work,
                    {
                        "state": "running",
                        "stage": stage,
                        "completed_matrices": sum(item is not None for item in batches),
                        "total_matrices": spec.matrices,
                        "cpu_seconds": consumed,
                    },
                )
                if consumed > cpu_budget_seconds:
                    raise RuntimeError("PX8 declared CPU budget exhausted")
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
    if any(batch is None for batch in batches):
        raise AssertionError("PX8 checkpoint stage incomplete")
    return [batch for batch in batches if batch is not None]


def _effect_text(item: Mapping[str, Any]) -> str:
    return (
        f"{float(item['effect']):+.4f} "
        f"[{float(item['ci95'][0]):+.4f}, {float(item['ci95'][1]):+.4f}]"
    )


def _reports(metrics: dict[str, Any], registration_id: str) -> tuple[str, str]:
    outcome_lines = [
        f"| {item['candidate']} | {item['target_half']} | {_effect_text(item)} | "
        f"{item['holm_adjusted_p']:.4g} | {item['random_tost']} | {item['pass']} |"
        for item in metrics["outcome_validity"]
    ]
    reliability_lines = [
        f"| {item['candidate']} | {_effect_text(item)} | "
        f"{item['holm_adjusted_p']:.4g} | {item['pass']} |"
        for item in metrics["reliability"]
    ]
    forecast_lines = [
        f"| {item['candidate']} | {item['direction']} | {_effect_text(item)} | "
        f"{item['holm_adjusted_p']:.4g} | {item['pass']} |"
        for item in metrics["forecast"]
    ]
    response_lines = [
        f"| {item['candidate']} | {item['source_half']} | {_effect_text(item)} | "
        f"{item['holm_adjusted_p']:.4g} | {item['pass']} |"
        for item in metrics["response"]
    ]
    support_lines = [
        f"| {item['support_branches']} | {item['candidate']} | "
        f"{item['source_half']} | {_effect_text(item)} |"
        for item in metrics["support_response"]
    ]
    control_lines = [
        f"| {item['candidate']} | {item['source_half']} | {_effect_text(item)} |"
        for item in metrics["negative_control"]
    ]
    gates = metrics["gates"]
    technical = "\n".join(
        (
            "# PX8 high-support resilience gauge confirmation",
            "",
            f"Registration: `{registration_id}`.",
            "",
            "## Renewal manipulation validity",
            "",
            "| Candidate | Target half | UP−DOWN renewal [95% CI] | Holm p | Random equivalent | Pass |",
            "| --- | --- | ---: | ---: | --- | --- |",
            *outcome_lines,
            "",
            "## Primary 128-branch reliability",
            "",
            "| Candidate | A-vs-B association [95% CI] | Holm p | Pass |",
            "| --- | ---: | ---: | --- |",
            *reliability_lines,
            "",
            "## Primary opposite-half forecast",
            "",
            "| Candidate | Direction | Association [95% CI] | Holm p | Pass |",
            "| --- | --- | ---: | ---: | --- |",
            *forecast_lines,
            "",
            "## Primary causal information response",
            "",
            "| Candidate | Source half | UP−DOWN information [95% CI] | Holm p | Pass |",
            "| --- | --- | ---: | ---: | --- |",
            *response_lines,
            "",
            "## Support ladder",
            "",
            "| Source branches | Candidate | Half | UP−DOWN information [95% CI] |",
            "| ---: | --- | --- | ---: |",
            *support_lines,
            "",
            f"- Finite-support gate: **{metrics['support_convergence']['finite_pass']}**",
            f"- Positive at 64 and 128 in all cells: **{metrics['support_convergence']['positive_at_64_and_128_all_cells']}**",
            f"- Median |32→64 change|: **{metrics['support_convergence']['median_abs_change_32_to_64']:.6g}**",
            f"- Median |64→128 change|: **{metrics['support_convergence']['median_abs_change_64_to_128']:.6g}**",
            f"- Contracting final step: **{metrics['support_convergence']['contracting_final_step']}**",
            "",
            "## Public revised negative control",
            "",
            "| Candidate | Half | UP−DOWN revised Phi-r [95% CI] |",
            "| --- | --- | ---: |",
            *control_lines,
            "",
            "## Registered classification",
            "",
            *(f"- {key}: **{value}**" for key, value in gates.items()),
            "",
            "## Boundary",
            "",
            "PX8 can confirm only the named high-support generational beta-typeset resilience gauge. It cannot rescue the exact public nine-atom Phi-r, make Phi-r causal, or support consciousness, life, agency, real chemistry, or metaphysical claims.",
            "",
        )
    )
    if gates["specialized_resilience_gauge_confirmed"]:
        result = (
            "The one information thermometer selected from PX7 survived a fresh, "
            "higher-support confirmation for post-break recovery."
        )
    elif gates["estimator_support_converged"]:
        result = (
            "The information reading settled as more futures were supplied, but it "
            "still failed at least one reliability, forecasting, or causal-response requirement."
        )
    else:
        result = (
            "The apparent PX7 information signal did not stabilize and pass the full "
            "fresh-cohort test when substantially more futures were supplied."
        )
    lay = "\n".join(
        (
            "# Lay summary — PX8 high-support resilience gauge",
            "",
            "PX8 gave the most promising remaining information score much more data. Every test began after a natural heredity break, and every intervention arm started from exactly the same broken assembly. One group of futures supplied the information reading while the other supplied the recovery outcome.",
            "",
            result,
            "",
            "This does not change whether molecular edits can control recovery; it asks only whether this particular information number is a dependable thermometer for that control.",
            "",
        )
    )
    return technical, lay


def run(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_OUTPUT,
    work: Path = DEFAULT_WORK,
    workers: int = MAX_WORKERS,
    cpu_budget_hours: float = MAX_CPU_HOURS,
) -> dict[str, Any]:
    registration = verify_registration(registration_directory)
    verify_checksums(DEFAULT_SMOKE)
    if output.exists():
        raise FileExistsError(output)
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"PX8 workers must be in [1,{MAX_WORKERS}]")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise OSError("PX8 free-disk gate failed")
    spec = scientific_spec()
    _prepare_work(work, registration["registration_id"], spec, cpu_budget_hours)
    model_path = registration_directory / "frozen_cr5_students.npz"
    contract_path = registration_directory / "model_contract.json"
    budget_seconds = cpu_budget_hours * 3600.0
    try:
        generated = _run_checkpoint_stage(
            spec,
            work / "generation",
            workers,
            model_path,
            contract_path,
            work,
            "generation",
            budget_seconds,
        )
        generation_cpu = float(sum(batch.cpu_seconds for batch in generated))
        replayed = _run_checkpoint_stage(
            spec,
            work / "replay",
            workers,
            model_path,
            contract_path,
            work,
            "replay",
            max(1.0, budget_seconds - generation_cpu),
        )
        replay_rows = [
            {
                "matrix_id": left.matrix_id,
                "generated_digest": left.scientific_digest,
                "replay_digest": right.scientific_digest,
                "exact": left.scientific_digest == right.scientific_digest,
            }
            for left, right in zip(generated, replayed, strict=True)
        ]
        replay_audit = {
            "format": "codex-ch5-phir-px8-replay-v1",
            "matrices": replay_rows,
            "complete_exact_replay": bool(
                len(replay_rows) == spec.matrices
                and all(item["exact"] for item in replay_rows)
            ),
        }
        if not replay_audit["complete_exact_replay"]:
            raise AssertionError("PX8 complete replay failed")
        replay_cpu = float(sum(batch.cpu_seconds for batch in replayed))
        _status_write(
            work,
            {
                "state": "analyzing",
                "stage": "analysis",
                "completed_matrices": spec.matrices,
                "total_matrices": spec.matrices,
                "cpu_seconds": generation_cpu + replay_cpu,
            },
        )
        metrics, tables, arrays = analyze_batches(generated, spec)
        staging = work / "final_staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        for name, frame in tables.items():
            frame.to_csv(staging / f"{name}.csv.gz", index=False, compression="gzip")
        np.savez_compressed(staging / "inference_arrays.npz", **arrays)
        np.savez_compressed(
            staging / "matrix_inputs.npz",
            matrix_ids=np.asarray(
                [batch.matrix_id for batch in generated], dtype=np.int16
            ),
            betas=np.asarray([batch.beta for batch in generated], dtype=np.float64),
            initials=np.asarray([batch.initial for batch in generated], dtype=np.int16),
        )
        _atomic_json(staging / "primary_metrics.json", metrics)
        _atomic_json(staging / "replay_audit.json", replay_audit)
        scientific_digest = _digest(
            {
                "registration_id": registration["registration_id"],
                "matrix_digests": [batch.scientific_digest for batch in generated],
                "metrics": metrics,
            }
        )
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "matrices": spec.matrices,
            "candidates": list(CANDIDATES),
            "branches_per_arm": spec.branches,
            "generation_cpu_seconds": generation_cpu,
            "replay_cpu_seconds": replay_cpu,
            "workers": workers,
            "declared_cpu_budget_hours": cpu_budget_hours,
            "scientific_digest": scientific_digest,
            "gates": metrics["gates"],
        }
        _atomic_json(staging / "manifest.json", manifest)
        technical, lay = _reports(metrics, registration["registration_id"])
        (staging / "SCIENTIFIC_REPORT.md").write_text(technical, encoding="utf-8")
        (staging / "LAY_SUMMARY.md").write_text(lay, encoding="utf-8")
        readback = {
            "format": "codex-ch5-phir-px8-readback-v1",
            "table_rows": {
                name: int(len(pd.read_csv(staging / f"{name}.csv.gz")))
                for name in tables
            },
            "inference_array_keys": sorted(
                np.load(staging / "inference_arrays.npz").files
            ),
            "all_tables_nonempty": bool(
                all(len(frame) > 0 for frame in tables.values())
            ),
            "manifest_scientific_digest": scientific_digest,
        }
        if not readback["all_tables_nonempty"]:
            raise AssertionError("PX8 artifact readback found an empty table")
        _atomic_json(staging / "readback_audit.json", readback)
        write_checksums(staging)
        if output.exists():
            raise FileExistsError(output)
        staging.replace(output)
        verify_result(output, registration_directory)
        _append_ledger(
            f"<!-- phir-extension-px8-result-{registration['registration_id']} -->",
            [
                "## Phi-r extension PX8 completed",
                "",
                f"- Result: `{output.relative_to(ROOT)}`.",
                f"- Gates: `{json.dumps(metrics['gates'], sort_keys=True)}`.",
                "- The single-formulation high-support confirmation received complete exact replay and artifact readback.",
            ],
        )
        _status_write(
            work,
            {
                "state": "complete",
                "stage": "sealed",
                "completed_matrices": spec.matrices,
                "total_matrices": spec.matrices,
                "output": str(output),
                "gates": metrics["gates"],
            },
        )
        return manifest
    except BaseException as error:
        _status_write(
            work,
            {
                "state": "failed",
                "stage": "exception",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def verify_result(
    directory: Path = DEFAULT_OUTPUT,
    registration_directory: Path = DEFAULT_REGISTRATION,
) -> dict[str, Any]:
    verify_checksums(directory)
    registration = verify_registration(registration_directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    replay = json.loads((directory / "replay_audit.json").read_text(encoding="utf-8"))
    readback = json.loads(
        (directory / "readback_audit.json").read_text(encoding="utf-8")
    )
    if manifest.get("format") != RESULT_FORMAT:
        raise ValueError("unsupported PX8 result")
    if manifest["registration_id"] != registration["registration_id"]:
        raise ValueError("PX8 result registration mismatch")
    if not replay["complete_exact_replay"]:
        raise ValueError("PX8 result lacks complete replay")
    if not readback["all_tables_nonempty"]:
        raise ValueError("PX8 result readback failed")
    return manifest


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def launch_detached(
    workers: int = MAX_WORKERS,
    cpu_budget_hours: float = MAX_CPU_HOURS,
) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(DEFAULT_OUTPUT)
    launch_path = DEFAULT_WORK / "detached_launch.json"
    if launch_path.exists():
        existing = json.loads(launch_path.read_text(encoding="utf-8"))
        if _pid_alive(int(existing.get("pid", -1))):
            raise RuntimeError(f"PX8 already runs as PID {existing['pid']}")
    DEFAULT_WORK.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "plastic_heredity.phir_extension_px8",
        "run",
        "--workers",
        str(workers),
        "--cpu-budget-hours",
        str(cpu_budget_hours),
    ]
    with DEFAULT_LOG.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    payload = {
        "format": "codex-ch5-phir-px8-detached-launch-v1",
        "registration_id": registration["registration_id"],
        "pid": process.pid,
        "workers": workers,
        "cpu_budget_hours": cpu_budget_hours,
        "command": command,
        "log": str(DEFAULT_LOG),
        "launched_at_unix": time.time(),
    }
    _atomic_json(launch_path, payload)
    return payload


def status() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": "codex-ch5-phir-px8-status-report-v1",
        "validation": DEFAULT_VALIDATION.exists(),
        "registration": DEFAULT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "complete": DEFAULT_OUTPUT.exists(),
        "log": str(DEFAULT_LOG),
    }
    launch_path = DEFAULT_WORK / "detached_launch.json"
    if launch_path.exists():
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        payload["launch"] = launch
        payload["pid_alive"] = _pid_alive(int(launch.get("pid", -1)))
    status_path = DEFAULT_WORK / "status.json"
    if status_path.exists():
        payload["work"] = json.loads(status_path.read_text(encoding="utf-8"))
    if DEFAULT_LOG.exists():
        payload["log_tail"] = DEFAULT_LOG.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-20:]
    if DEFAULT_OUTPUT.exists():
        payload["manifest"] = verify_result(DEFAULT_OUTPUT)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("register")
    sub.add_parser("smoke")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    run_parser.add_argument("--cpu-budget-hours", type=float, default=MAX_CPU_HOURS)
    launch = sub.add_parser("launch")
    launch.add_argument("--workers", type=int, default=MAX_WORKERS)
    launch.add_argument("--cpu-budget-hours", type=float, default=MAX_CPU_HOURS)
    sub.add_parser("status")
    sub.add_parser("verify-registration")
    sub.add_parser("verify-result")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        value = validate()
    elif args.command == "register":
        value = register()
    elif args.command == "smoke":
        value = smoke()
    elif args.command == "run":
        value = run(workers=args.workers, cpu_budget_hours=args.cpu_budget_hours)
    elif args.command == "launch":
        value = launch_detached(args.workers, args.cpu_budget_hours)
    elif args.command == "status":
        value = status()
    elif args.command == "verify-registration":
        value = verify_registration()
    elif args.command == "verify-result":
        value = verify_result()
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(_json_ready(value), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
