"""PX9 prospective gauge-identity pilot for post-break plastic heredity.

PX9 freezes the PX8 beta-partitioned whole-minus-within reading and asks
whether its response is temporal, beta-topology specific, graded, and
nonredundant with ordinary process summaries.  It never searches over Phi-r
definitions and the public revised score is a non-winning negative control.
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
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from . import intervention_cr5 as cr5
from . import intervention_cr2_dose_response as cr2
from . import phir_extension_px7 as px7
from .config import CANDIDATES, GardConfig
from .experiment import StateCase
from .intervention_core import MolecularEdit, ScoredEdit, apply_molecular_edit
from .mechanistic import sha256_file, verify_checksums, write_checksums
from .mechanistic_metrics import holm_adjust
from .phir_ch5 import _append_ledger, _snapshot_after_record
from .phir_instruments import advance_fission_traced, gaussian_mutual_information
from .phir_rescue_instruments import beta_physical_partition
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
DOCUMENT = "CODEX_CH5_PHIR_PX9_PREREGISTRATION.md"
LEDGER = ROOT / "PHIR_RESULTS_LEDGER.md"

DEFAULT_VALIDATION = RESULT_ROOT / "px9_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "px9_registration"
DEFAULT_SMOKE = RESULT_ROOT / "px9_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "px9_gauge_identity24"
DEFAULT_WORK = RESULT_ROOT / ".px9_gauge_identity_work"
DEFAULT_LOG = RESULT_ROOT / "px9_gauge_identity24.log"

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

PROGRAM_FORMAT = "codex-ch5-phir-px9-gauge-identity-v1"
REGISTRATION_FORMAT = "codex-ch5-phir-px9-registration-v1"
RESULT_FORMAT = "codex-ch5-phir-px9-result-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-px9-checkpoint-v1"
STATUS_FORMAT = "codex-ch5-phir-px9-status-v1"
LABEL = "CODEX_CH5_PHIR_PX9_GAUGE_IDENTITY_V1"

MATRICES = 24
LANDMARKS = (20, 35, 50, 65, 80)
BRANCHES = 256
HALVES = {"A": tuple(range(0, 128)), "B": tuple(range(128, 256))}
SUPPORT_LEVELS = (64, 128)
PRIMARY_SUPPORT = 128
HORIZON = 8
ACQUISITION_LIMIT = 60
PAST_WINDOW = 512
MINIMUM_ELIGIBLE_MATRICES = 20
BOOTSTRAP_DRAWS = 4096
RANDOMIZATION_DRAWS = 4096
OUTCOME_EQUIVALENCE_MARGIN = 0.025
MAX_WORKERS = 8
MAX_CPU_HOURS = 30.0
MINIMUM_FREE_DISK_BYTES = 350_000_000
SMOKE_MATRIX_ID = 2

QUANTILES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
QUANTILE_ARMS = ("Q00", "Q20", "Q40", "Q60", "Q80", "Q100")
ARMS = (*QUANTILE_ARMS, "RANDOM", "NOOP")
STABILIZING = "Q100"
DESTABILIZING = "Q00"
TEMPORAL_SHIFTS = (1, 3, 5, 7, 11, 13, 17, 19)
RANDOM_PARTITIONS = 8
CROSS_BETA_OFFSETS = (1, 5, 11)
PRIMARY_FORMULATION = "generational__beta__typeset"
PUBLIC_NEGATIVE_CONTROL = "molecular__self__revised"

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/phir_extension_px9.py",
    "tests/test_phir_extension_px9.py",
    "plastic_heredity/phir_extension_px8.py",
    "plastic_heredity/phir_extension_px7.py",
    "plastic_heredity/phir_instruments.py",
    "plastic_heredity/phir_rescue_instruments.py",
    "plastic_heredity/intervention_cr5.py",
    "plastic_heredity/intervention_cr2_dose_response.py",
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
        "random_action",
        "future",
        "partition",
        "temporal_shuffle",
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


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


@dataclass(frozen=True)
class PX9Spec:
    label: str
    matrices: int
    landmarks: tuple[int, ...]
    branches: int
    horizon: int
    acquisition_limit: int
    bootstrap_draws: int
    randomization_draws: int


def scientific_spec() -> PX9Spec:
    return PX9Spec(
        "scientific",
        MATRICES,
        LANDMARKS,
        BRANCHES,
        HORIZON,
        ACQUISITION_LIMIT,
        BOOTSTRAP_DRAWS,
        RANDOMIZATION_DRAWS,
    )


def smoke_spec() -> PX9Spec:
    return PX9Spec("smoke", 1, LANDMARKS, 16, 3, 60, 32, 32)


def _supports(spec: PX9Spec) -> tuple[int, ...]:
    if spec.branches == BRANCHES:
        return SUPPORT_LEVELS
    return (spec.branches // 2,)


def _halves(spec: PX9Spec) -> dict[str, tuple[int, ...]]:
    midpoint = spec.branches // 2
    return {"A": tuple(range(midpoint)), "B": tuple(range(midpoint, spec.branches))}


def _seed(spec: PX9Spec, domain: str, *keys: object) -> int:
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
            cohort="PX9_RESILIENCE",
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
class PX9Batch:
    matrix_id: int
    beta: NDArray[np.float64]
    initial: NDArray[np.int16]
    acquisition_rows: tuple[dict[str, Any], ...]
    edit_rows: tuple[dict[str, Any], ...]
    branch_rows: tuple[dict[str, Any], ...]
    score_rows: tuple[dict[str, Any], ...]
    cpu_seconds: float
    scientific_digest: str


def protocol(spec: PX9Spec | None = None) -> dict[str, Any]:
    spec = scientific_spec() if spec is None else spec
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "question": "identity of the high-support PX8 resilience gauge",
        "predecessors_immutable": True,
        "public_revised_can_win": False,
        "cohort": {
            "matrices": spec.matrices,
            "candidates": list(CANDIDATES),
            "landmarks": list(spec.landmarks),
            "branches_per_arm": spec.branches,
            "halves": {key: list(item) for key, item in _halves(spec).items()},
            "horizon": spec.horizon,
            "acquisition_limit": spec.acquisition_limit,
            "minimum_eligible_matrices": MINIMUM_ELIGIBLE_MATRICES,
            "replacement": False,
            "pilot_only": True,
            "automatic_48_matrix_continuation": False,
        },
        "arms": {
            "ordered": list(ARMS),
            "quantile_arms": list(QUANTILE_ARMS),
            "quantiles": list(QUANTILES),
            "exhaustive_scoring": True,
        },
        "target": {
            "formulation": PRIMARY_FORMULATION,
            "clock": "explicit generational parent-to-selected-daughter pairs",
            "partition": "fixed unedited-beta Fiedler",
            "functional": "I(X;X') - I(A;A') - I(B;B')",
            "support_branches": list(_supports(spec)),
            "primary_support": max(_supports(spec)),
        },
        "temporal_null": {
            "shifts": list(TEMPORAL_SHIFTS),
            "within_fission_depth": True,
            "preserves_marginals": True,
            "corrected": "paired minus mean deranged",
        },
        "partition_null": {
            "size_matched_random_partitions": RANDOM_PARTITIONS,
            "cross_beta_offsets": list(CROSS_BETA_OFFSETS),
            "topology_excess": "beta minus median size-matched random",
        },
        "ordinary_process_baseline": {
            "matrix_folds": 5,
            "ridge_C": 1.0,
            "training_only_standardization": True,
            "target": "opposite-half branch-level renewal log loss",
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_draws": spec.bootstrap_draws,
            "randomization_draws": spec.randomization_draws,
            "holm_within_named_families": True,
            "outcome_equivalence_margin": OUTCOME_EQUIVALENCE_MARGIN,
        },
        "frozen_students": {
            "archive_sha256": sha256_file(MODEL_SOURCE),
            "contract_sha256": sha256_file(MODEL_CONTRACT_SOURCE),
            "refit_or_recalibration": False,
        },
        "randomness": {
            "domains": SEED_DOMAINS,
            "arm_in_future_seed": False,
            "common_random_streams": True,
            "random_action_separate": True,
        },
        "operational": {
            "workers_max": MAX_WORKERS,
            "cpu_hours_max": MAX_CPU_HOURS,
            "detached_science": True,
            "matrix_checkpointing": True,
            "complete_replay": True,
            "work_path_is_operational": True,
        },
        "classifications": [
            "beta_topology_specific_nonredundant_gauge",
            "generic_transition_information_gauge",
            "reliable_behavioral_echo",
            "finite_sample_or_marginal_explanation",
            "plastic_heredity_manipulation_invalid",
        ],
    }
    value["protocol_id"] = _digest(value)
    return value


def _run_natural_candidate(
    matrix_id: int,
    beta: NDArray[np.float64],
    initial: NDArray[np.int16],
    candidate: str,
    spec: PX9Spec,
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
                            state_id=f"PX9-c{candidate}-m{matrix_id:03d}-g{generation:03d}",
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
                raise AssertionError("PX9 natural path omitted a landmark")
            return output
        except SimulationError:
            continue
    raise SimulationError(
        f"PX9 failed bounded natural retry for c{candidate} m{matrix_id}"
    )


def _acquire_break(
    source: ResilienceCase, spec: PX9Spec
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


def _selection_seed(spec: PX9Spec, case: ResilienceCase) -> int:
    return _seed(
        spec,
        "random_action",
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def _future_seed(spec: PX9Spec, case: ResilienceCase, branch: int) -> int:
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
    spec: PX9Spec,
) -> tuple[tuple[dict[str, Any], ...], tuple[MolecularEdit | None, ...]]:
    student = students[("renewal", case.candidate)]
    noop, scores = cr5.score_student_edits(
        student, case.as_state_case(), GardConfig()
    )
    selected, ranks = cr2.select_quantile_edits(scores)
    rng = np.random.default_rng(_selection_seed(spec, case))
    random_index = int(rng.integers(0, len(scores)))
    records: list[dict[str, Any]] = []
    edits: list[MolecularEdit | None] = []
    for arm, quantile, scored, rank in zip(
        QUANTILE_ARMS, QUANTILES, selected, ranks, strict=True
    ):
        records.append(
            {
                "arm": arm,
                "empirical_quantile": float(quantile),
                "selected_rank": int(rank),
                "prediction": float(scored.predicted_probability),
                "predicted_shift": float(scored.predicted_shift),
                "legal_edits": len(scores),
            }
        )
        edits.append(scored.edit)
    random_score = scores[random_index]
    records.append(
        {
            "arm": "RANDOM",
            "empirical_quantile": float("nan"),
            "selected_rank": random_index,
            "prediction": float(random_score.predicted_probability),
            "predicted_shift": float(random_score.predicted_shift),
            "legal_edits": len(scores),
        }
    )
    edits.append(random_score.edit)
    records.append(
        {
            "arm": "NOOP",
            "empirical_quantile": float("nan"),
            "selected_rank": -1,
            "prediction": float(noop),
            "predicted_shift": 0.0,
            "legal_edits": len(scores),
        }
    )
    edits.append(None)
    probabilities = [item["prediction"] for item in records[: len(QUANTILE_ARMS)]]
    if np.any(np.diff(np.asarray(probabilities, dtype=np.float64)) < 0):
        raise AssertionError("PX9 selected quantiles are not monotone")
    return tuple(records), tuple(edits)


def _first_run(values: Sequence[bool], length: int) -> int:
    return px7._first_run(values, length)


def _records_digest(records: Iterable[FissionRecord]) -> str:
    return px7._records_digest(records)


def _composition_summary(
    composition: NDArray, beta: NDArray
) -> dict[str, float | int]:
    counts = np.asarray(composition, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return {
            "entropy": 0.0,
            "occupied_types": 0,
            "top1_share": 0.0,
            "throughput": 0.0,
        }
    x = counts / total
    positive = x[x > 0]
    return {
        "entropy": float(-np.sum(positive * np.log(positive))),
        "occupied_types": int(np.count_nonzero(counts)),
        "top1_share": float(x.max()),
        "throughput": float(x @ np.asarray(beta, dtype=np.float64) @ x),
    }


def _simulate_branch(
    case: ResilienceCase,
    edit: MolecularEdit | None,
    branch: int,
    spec: PX9Spec,
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
    generation_past: list[NDArray[np.int64]] = []
    generation_future: list[NDArray[np.int64]] = []
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
        generation_past.append(np.asarray(traced.record.parent, dtype=np.int64))
        generation_future.append(np.asarray(traced.record.daughter, dtype=np.int64))
        records.append(traced.record)
        snapshot = _snapshot_after_record(snapshot, traced.record)
    inherited = [record.h > config.inheritance_threshold for record in records]
    first_event = _first_run(inherited, 3)
    h_values = np.asarray([record.h for record in records], dtype=np.float64)
    molecular_array = np.asarray(molecular, dtype=np.int16)
    if generation_past:
        generational_left = np.asarray(generation_past, dtype=np.int16)
        generational_right = np.asarray(generation_future, dtype=np.int16)
    else:
        generational_left = np.empty((0, config.n_types), dtype=np.int16)
        generational_right = np.empty((0, config.n_types), dtype=np.int16)
    row = {
        "state_id": case.state_id,
        "axis": "resilience",
        "candidate": case.candidate,
        "matrix_id": case.matrix_id,
        "landmark": case.landmark,
        "branch": branch,
        "half": "A" if branch < spec.branches // 2 else "B",
        "primary": int(first_event >= 0),
        "completed": int(len(records) == spec.horizon),
        "survived": int(len(records) == spec.horizon),
        "inherited_fraction": float(sum(inherited) / spec.horizon),
        "mean_h": float(h_values.mean()) if h_values.size else 0.0,
        "minimum_h": float(h_values.min()) if h_values.size else 0.0,
        "first_event_time": int(first_event),
        "run5": int(_first_run(inherited, 5) >= 0),
        "record_digest": _records_digest(records),
        "rng_state_digest": _digest(rng.bit_generator.state),
        "molecular_pairs": int(max(0, molecular_array.shape[0] - 1)),
        "generational_pairs": int(generational_left.shape[0]),
    }
    return row, PairBlock(
        molecular_array[:-1],
        molecular_array[1:],
        generational_left,
        generational_right,
    )


def _concatenate_pairs(
    blocks: Sequence[PairBlock], clock: str
) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
    return px7._concatenate_pairs(blocks, clock)


def _temporal_derangement(
    blocks: Sequence[PairBlock], shift: int
) -> tuple[NDArray[np.int16], NDArray[np.int16], int]:
    """Derange daughters across branches within each fission depth."""

    left_rows: list[NDArray[np.int16]] = []
    right_rows: list[NDArray[np.int16]] = []
    self_pairs = 0
    maximum = max((len(block.generational_past) for block in blocks), default=0)
    for depth in range(maximum):
        eligible = [
            index
            for index, block in enumerate(blocks)
            if depth < len(block.generational_past)
        ]
        if not eligible:
            continue
        if len(eligible) == 1:
            index = eligible[0]
            left_rows.append(blocks[index].generational_past[depth])
            right_rows.append(blocks[index].generational_future[depth])
            continue
        offset = int(shift % len(eligible))
        if offset == 0:
            offset = 1
        for position, source_index in enumerate(eligible):
            destination_index = eligible[(position + offset) % len(eligible)]
            if destination_index == source_index:
                self_pairs += 1
            left_rows.append(blocks[source_index].generational_past[depth])
            right_rows.append(blocks[destination_index].generational_future[depth])
    n_types = GardConfig().n_types
    return (
        np.asarray(left_rows, dtype=np.int16)
        if left_rows
        else np.empty((0, n_types), dtype=np.int16),
        np.asarray(right_rows, dtype=np.int16)
        if right_rows
        else np.empty((0, n_types), dtype=np.int16),
        self_pairs,
    )


def _random_partitions(
    case: ResilienceCase,
    spec: PX9Spec,
) -> tuple[tuple[NDArray[np.int64], NDArray[np.int64]], ...]:
    first, _second = beta_physical_partition(case.beta)
    species = np.arange(GardConfig().n_types, dtype=np.int64)
    output: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    for control in range(RANDOM_PARTITIONS):
        rng = np.random.default_rng(
            _seed(spec, "partition", case.matrix_id, "random", control)
        )
        order = rng.permutation(species)
        split = len(first)
        output.append(
            (
                np.sort(order[:split]).astype(np.int64),
                np.sort(order[split:]).astype(np.int64),
            )
        )
    return tuple(output)


def _cross_beta_partitions(
    case: ResilienceCase,
    spec: PX9Spec,
) -> tuple[tuple[NDArray[np.int64], NDArray[np.int64]], ...]:
    config = GardConfig()
    modulus = max(MATRICES, spec.matrices)
    output: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    for offset in CROSS_BETA_OFFSETS:
        other = (case.matrix_id + offset) % modulus
        if other == case.matrix_id:
            other = case.matrix_id + offset
        beta = generate_beta(
            config, np.random.default_rng(_seed(spec, "matrix", other))
        )
        output.append(beta_physical_partition(beta))
    return tuple(output)


def _fixed_partition_score(
    past: NDArray,
    future: NDArray,
    first_species: Sequence[int],
    second_species: Sequence[int],
) -> dict[str, Any]:
    try:
        left, right, active = px7._explicit_transform(past, future)
        first, second = px7._map_partition(active, first_species, second_species)
        whole = gaussian_mutual_information(left, right)
        aa = gaussian_mutual_information(left[first], right[first])
        bb = gaussian_mutual_information(left[second], right[second])
        return {
            "value": float(whole - aa - bb),
            "whole_mi": float(whole),
            "aa_mi": float(aa),
            "bb_mi": float(bb),
            "active_dimensions": int(active.size),
            "part_a_dimensions": int(first.size),
            "part_b_dimensions": int(second.size),
            "transitions": int(left.shape[1]),
            "partition_digest": _array_digest(active, first, second),
        }
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return {
            "value": float("nan"),
            "whole_mi": float("nan"),
            "aa_mi": float("nan"),
            "bb_mi": float("nan"),
            "active_dimensions": 0,
            "part_a_dimensions": 0,
            "part_b_dimensions": 0,
            "transitions": int(len(past)),
            "partition_digest": "invalid",
        }


def _score_row(
    case: ResilienceCase,
    arm: str,
    half: str,
    support: int,
    score_kind: str,
    control_id: int,
    score: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "state_id": case.state_id,
        "axis": "resilience",
        "candidate": case.candidate,
        "matrix_id": case.matrix_id,
        "landmark": case.landmark,
        "arm": arm,
        "source_half": half,
        "support_branches": support,
        "score_kind": score_kind,
        "control_id": control_id,
        "value": float(score["value"]),
        "whole_mi": float(score.get("whole_mi", float("nan"))),
        "aa_mi": float(score.get("aa_mi", float("nan"))),
        "bb_mi": float(score.get("bb_mi", float("nan"))),
        "active_dimensions": int(score.get("active_dimensions", 0)),
        "part_a_dimensions": int(score.get("part_a_dimensions", 0)),
        "part_b_dimensions": int(score.get("part_b_dimensions", 0)),
        "transitions": int(score.get("transitions", 0)),
        "branches_used": int(score.get("branches_used", support)),
        "partition_digest": str(score.get("partition_digest", "invalid")),
    }


def _score_arm_halves(
    arm: str,
    case: ResilienceCase,
    blocks: Sequence[PairBlock],
    spec: PX9Spec,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    beta_first, beta_second = beta_physical_partition(case.beta)
    random_partitions = _random_partitions(case, spec)
    cross_partitions = _cross_beta_partitions(case, spec)
    primary_support = max(_supports(spec))
    for half, indices in _halves(spec).items():
        half_blocks = [blocks[index] for index in indices]
        for support in _supports(spec):
            selected = half_blocks[:support]
            past, future = _concatenate_pairs(selected, "generational")
            paired = _fixed_partition_score(
                past, future, beta_first, beta_second
            )
            output.append(
                _score_row(case, arm, half, support, "paired_beta", -1, paired)
            )
            if support != primary_support:
                continue
            for control, shift in enumerate(TEMPORAL_SHIFTS):
                shuffled_past, shuffled_future, self_pairs = _temporal_derangement(
                    selected, shift
                )
                if self_pairs:
                    raise AssertionError("PX9 temporal null retained a self-pair")
                shuffled = _fixed_partition_score(
                    shuffled_past,
                    shuffled_future,
                    beta_first,
                    beta_second,
                )
                output.append(
                    _score_row(
                        case,
                        arm,
                        half,
                        support,
                        "shuffled_beta",
                        control,
                        shuffled,
                    )
                )
            for control, (first, second) in enumerate(random_partitions):
                score = _fixed_partition_score(past, future, first, second)
                output.append(
                    _score_row(
                        case,
                        arm,
                        half,
                        support,
                        "random_partition",
                        control,
                        score,
                    )
                )
            for control, (first, second) in enumerate(cross_partitions):
                score = _fixed_partition_score(past, future, first, second)
                output.append(
                    _score_row(
                        case,
                        arm,
                        half,
                        support,
                        "cross_beta_partition",
                        control,
                        score,
                    )
                )
            molecular_past, molecular_future = _concatenate_pairs(
                selected, "molecular"
            )
            public = px7._safe_score_pairs(
                molecular_past, molecular_future, "self", case
            )
            output.append(
                _score_row(
                    case,
                    arm,
                    half,
                    support,
                    "public_revised",
                    -1,
                    {
                        "value": public["revised"],
                        "whole_mi": public["whole_mi"],
                        "active_dimensions": public["active_dimensions"],
                        "part_a_dimensions": public["part_a_dimensions"],
                        "part_b_dimensions": public["part_b_dimensions"],
                        "transitions": public["transitions"],
                        "partition_digest": public["partition_digest"],
                    },
                )
            )
    return output


def _score_concordant_extremes(
    case: ResilienceCase,
    blocks_by_arm: Mapping[str, Sequence[PairBlock]],
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    spec: PX9Spec,
) -> list[dict[str, Any]]:
    """Descriptive score after conditioning on concordant extreme-arm outcomes."""

    if not {STABILIZING, DESTABILIZING}.issubset(blocks_by_arm):
        return []
    first, second = beta_physical_partition(case.beta)
    output: list[dict[str, Any]] = []
    for half, indices in _halves(spec).items():
        concordant = [
            index
            for index in indices
            if rows_by_arm[STABILIZING][index]["primary"]
            == rows_by_arm[DESTABILIZING][index]["primary"]
            and rows_by_arm[STABILIZING][index]["survived"]
            == rows_by_arm[DESTABILIZING][index]["survived"]
        ]
        for arm in (STABILIZING, DESTABILIZING):
            selected = [blocks_by_arm[arm][index] for index in concordant]
            past, future = _concatenate_pairs(selected, "generational")
            score = _fixed_partition_score(past, future, first, second)
            score["branches_used"] = len(concordant)
            output.append(
                _score_row(
                    case,
                    arm,
                    half,
                    max(_supports(spec)),
                    "concordant_outcome_beta",
                    -1,
                    score,
                )
            )
    return output


def _run_case(
    case: ResilienceCase,
    students: Mapping[tuple[str, str], cr5.FrozenCR5Student],
    spec: PX9Spec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selection_rows, edits = _select_edits(case, students, spec)
    branch_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    blocks_by_arm: dict[str, list[PairBlock]] = {}
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for selected, edit in zip(selection_rows, edits, strict=True):
        arm = str(selected["arm"])
        composition = (
            case.snapshot.composition
            if edit is None
            else apply_molecular_edit(case.snapshot.composition, edit)
        )
        summary = _composition_summary(composition, case.beta)
        edit_rows.append(
            {
                "state_id": case.state_id,
                "axis": "resilience",
                "candidate": case.candidate,
                "matrix_id": case.matrix_id,
                "landmark": case.landmark,
                **selected,
                "remove_type": -1 if edit is None else edit.remove_type,
                "add_type": -1 if edit is None else edit.add_type,
                "history_digest": _array_digest(case.history_counts),
                **{f"state_{key}": value for key, value in summary.items()},
            }
        )
        blocks: list[PairBlock] = []
        arm_rows: list[dict[str, Any]] = []
        for branch in range(spec.branches):
            row, block = _simulate_branch(case, edit, branch, spec)
            row.update(
                {
                    "arm": arm,
                    "prediction": float(selected["prediction"]),
                    "empirical_quantile": float(selected["empirical_quantile"]),
                }
            )
            branch_rows.append(row)
            arm_rows.append(row)
            blocks.append(block)
        blocks_by_arm[arm] = blocks
        rows_by_arm[arm] = arm_rows
    for arm in ARMS:
        score_rows.extend(_score_arm_halves(arm, case, blocks_by_arm[arm], spec))
    score_rows.extend(
        _score_concordant_extremes(case, blocks_by_arm, rows_by_arm, spec)
    )
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


def _run_matrix(arguments: tuple[int, PX9Spec, str, str]) -> PX9Batch:
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
        return PX9Batch(
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
    local = series.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
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
    sign_rng = np.random.default_rng(
        _seed(scientific_spec(), "randomization", key)
    )
    signs = sign_rng.choice((-1.0, 1.0), size=(repetitions, values.size))
    randomized = (signs * values).mean(axis=1)
    observed = float(values.mean())
    item = {
        "effect": observed,
        "ci95": [float(value) for value in np.quantile(bootstrap, (0.025, 0.975))],
        "ci90": [float(value) for value in np.quantile(bootstrap, (0.05, 0.95))],
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
    return item


def _matrix_arm_effect(
    frame: pd.DataFrame,
    value: str,
    high: str,
    low: str,
    filters: Mapping[str, Any],
) -> pd.Series:
    return px7._matrix_arm_effect(frame, value, high, low, filters)


def _matrix_centered_spearman(
    frame: pd.DataFrame, left: str, right: str
) -> pd.Series:
    return px7._matrix_centered_spearman(frame, left, right)


def _statewise_spearman(
    frame: pd.DataFrame,
    left: str,
    right: str,
    require_quantiles: bool = False,
) -> pd.Series:
    values: list[dict[str, Any]] = []
    for (matrix_id, state_id), local in frame.groupby(
        ["matrix_id", "state_id"], sort=True
    ):
        if require_quantiles:
            local = local[local["arm"].isin(QUANTILE_ARMS)]
        x = pd.to_numeric(local[left], errors="coerce").to_numpy(float)
        y = pd.to_numeric(local[right], errors="coerce").to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
            continue
        statistic = float(spearmanr(x, y).statistic)
        if np.isfinite(statistic):
            values.append(
                {
                    "matrix_id": int(matrix_id),
                    "state_id": str(state_id),
                    "value": statistic,
                }
            )
    if not values:
        return pd.Series(dtype=float)
    return (
        pd.DataFrame(values)
        .groupby("matrix_id", sort=True)["value"]
        .mean()
        .sort_index()
    )


def _adjust_family(items: list[dict[str, Any]]) -> None:
    if not items:
        return
    adjusted = holm_adjust([float(item["one_sided_p"]) for item in items])
    for item, value in zip(items, adjusted, strict=True):
        item["holm_adjusted_p"] = float(value)
        item["pass"] = bool(
            item["effect"] > 0
            and item["ci95"][0] > 0
            and item["holm_adjusted_p"] < 0.05
        )


def _outcome_frame(branches: pd.DataFrame, edits: pd.DataFrame) -> pd.DataFrame:
    keys = ["matrix_id", "candidate", "state_id", "landmark", "arm", "half"]
    output = (
        branches.groupby(keys, sort=True)
        .agg(
            q=("primary", "mean"),
            successes=("primary", "sum"),
            trials=("primary", "size"),
            inherited_fraction=("inherited_fraction", "mean"),
            run5=("run5", "mean"),
            survived=("survived", "mean"),
            mean_h=("mean_h", "mean"),
            minimum_h=("minimum_h", "mean"),
            generational_pairs=("generational_pairs", "sum"),
        )
        .reset_index()
    )
    edit_columns = [
        "matrix_id",
        "candidate",
        "state_id",
        "landmark",
        "arm",
        "prediction",
        "predicted_shift",
        "empirical_quantile",
        "state_entropy",
        "state_occupied_types",
        "state_top1_share",
        "state_throughput",
    ]
    return output.merge(
        edits[edit_columns],
        on=["matrix_id", "candidate", "state_id", "landmark", "arm"],
        how="left",
        validate="many_to_one",
    )


def _derived_score_frame(scores: pd.DataFrame, support: int) -> pd.DataFrame:
    keys = [
        "matrix_id",
        "candidate",
        "state_id",
        "landmark",
        "arm",
        "source_half",
    ]
    local = scores[scores["support_branches"] == support]
    paired_columns = keys + [
        "value",
        "whole_mi",
        "aa_mi",
        "bb_mi",
        "active_dimensions",
        "part_a_dimensions",
        "part_b_dimensions",
        "transitions",
    ]
    paired = local[local["score_kind"] == "paired_beta"][paired_columns].rename(
        columns={"value": "paired_value"}
    )
    shuffled = (
        local[local["score_kind"] == "shuffled_beta"]
        .groupby(keys, sort=True)["value"]
        .agg(shuffled_value="mean", shuffled_controls="count")
        .reset_index()
    )
    random = (
        local[local["score_kind"] == "random_partition"]
        .groupby(keys, sort=True)["value"]
        .agg(random_partition_value="median", random_partition_controls="count")
        .reset_index()
    )
    cross = (
        local[local["score_kind"] == "cross_beta_partition"]
        .groupby(keys, sort=True)["value"]
        .agg(cross_beta_value="median", cross_beta_controls="count")
        .reset_index()
    )
    output = paired.merge(shuffled, on=keys, how="left", validate="one_to_one")
    output = output.merge(random, on=keys, how="left", validate="one_to_one")
    output = output.merge(cross, on=keys, how="left", validate="one_to_one")
    output["temporal_value"] = output["paired_value"] - output["shuffled_value"]
    output["topology_value"] = (
        output["paired_value"] - output["random_partition_value"]
    )
    output["cross_beta_excess"] = (
        output["paired_value"] - output["cross_beta_value"]
    )
    output["within_sum_mi"] = output["aa_mi"] + output["bb_mi"]
    output.loc[
        output["shuffled_controls"] != len(TEMPORAL_SHIFTS), "temporal_value"
    ] = np.nan
    output.loc[
        output["random_partition_controls"] != RANDOM_PARTITIONS,
        "topology_value",
    ] = np.nan
    output.loc[
        output["cross_beta_controls"] != len(CROSS_BETA_OFFSETS),
        "cross_beta_excess",
    ] = np.nan
    return output


def _outcome_validity(
    outcomes: pd.DataFrame,
    spec: PX9Spec,
    arrays: dict[str, NDArray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    extreme: list[dict[str, Any]] = []
    dose: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for half in ("A", "B"):
            filters = {"candidate": candidate, "half": half}
            effect = _matrix_arm_effect(
                outcomes, "q", STABILIZING, DESTABILIZING, filters
            )
            item = _bootstrap_summary(
                effect,
                spec.bootstrap_draws,
                f"outcome/extreme/c{candidate}/h{half}",
                arrays,
            )
            random = _matrix_arm_effect(
                outcomes, "q", "RANDOM", "NOOP", filters
            )
            random_item = _bootstrap_summary(
                random,
                spec.bootstrap_draws,
                f"outcome/random/c{candidate}/h{half}",
                arrays,
            )
            item.update(
                {
                    "candidate": candidate,
                    "target_half": half,
                    "random_minus_noop": random_item,
                    "random_tost": bool(
                        random_item["ci90"][0] > -OUTCOME_EQUIVALENCE_MARGIN
                        and random_item["ci90"][1] < OUTCOME_EQUIVALENCE_MARGIN
                    ),
                }
            )
            extreme.append(item)
            local = outcomes[
                (outcomes["candidate"] == candidate)
                & (outcomes["half"] == half)
                & outcomes["arm"].isin(QUANTILE_ARMS)
            ]
            series = _statewise_spearman(
                local, "empirical_quantile", "q", require_quantiles=True
            )
            dose_item = _bootstrap_summary(
                series,
                spec.bootstrap_draws,
                f"outcome/dose/c{candidate}/h{half}",
                arrays,
            )
            dose_item.update({"candidate": candidate, "target_half": half})
            dose.append(dose_item)
    _adjust_family(extreme)
    _adjust_family(dose)
    for item in extreme:
        item["pass"] = bool(item["pass"] and item["random_tost"])
    valid = bool(
        len(extreme) == 4
        and len(dose) == 4
        and all(item["pass"] for item in extreme)
        and all(item["pass"] for item in dose)
    )
    return extreme, dose, valid


def _forecast_frame(
    derived: pd.DataFrame,
    outcomes: pd.DataFrame,
    candidate: str,
    source_half: str,
) -> pd.DataFrame:
    target_half = "B" if source_half == "A" else "A"
    keys = ["matrix_id", "candidate", "state_id", "landmark", "arm"]
    source_scores = derived[
        (derived["candidate"] == candidate)
        & (derived["source_half"] == source_half)
    ].copy()
    source_process = outcomes[
        (outcomes["candidate"] == candidate) & (outcomes["half"] == source_half)
    ][
        keys
        + [
            "q",
            "inherited_fraction",
            "run5",
            "survived",
            "mean_h",
            "prediction",
            "predicted_shift",
            "empirical_quantile",
            "state_entropy",
            "state_occupied_types",
            "state_top1_share",
            "state_throughput",
        ]
    ].rename(
        columns={
            "q": "source_q",
            "inherited_fraction": "source_inherited_fraction",
            "run5": "source_run5",
            "survived": "source_survived",
            "mean_h": "source_mean_h",
        }
    )
    target = outcomes[
        (outcomes["candidate"] == candidate) & (outcomes["half"] == target_half)
    ][keys + ["q", "successes", "trials"]].rename(
        columns={
            "q": "target_q",
            "successes": "target_successes",
            "trials": "target_trials",
        }
    )
    merged = source_scores.merge(
        source_process, on=keys, how="inner", validate="one_to_one"
    )
    return merged.merge(target, on=keys, how="inner", validate="one_to_one")


def _reliability_family(
    derived: pd.DataFrame,
    value: str,
    label: str,
    spec: PX9Spec,
    arrays: dict[str, NDArray],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        local = derived[derived["candidate"] == candidate]
        paired = local.pivot_table(
            index=["matrix_id", "state_id", "landmark", "arm"],
            columns="source_half",
            values=value,
            aggfunc="first",
        ).reset_index()
        series = (
            _matrix_centered_spearman(paired, "A", "B")
            if {"A", "B"}.issubset(paired.columns)
            else pd.Series(dtype=float)
        )
        item = _bootstrap_summary(
            series,
            spec.bootstrap_draws,
            f"{label}/reliability/c{candidate}",
            arrays,
        )
        item.update({"candidate": candidate})
        items.append(item)
    _adjust_family(items)
    return items


def _response_family(
    derived: pd.DataFrame,
    value: str,
    label: str,
    spec: PX9Spec,
    arrays: dict[str, NDArray],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for half in ("A", "B"):
            series = _matrix_arm_effect(
                derived,
                value,
                STABILIZING,
                DESTABILIZING,
                {"candidate": candidate, "source_half": half},
            )
            item = _bootstrap_summary(
                series,
                spec.bootstrap_draws,
                f"{label}/response/c{candidate}/h{half}",
                arrays,
            )
            item.update({"candidate": candidate, "source_half": half})
            items.append(item)
    _adjust_family(items)
    return items


def _forecast_family(
    derived: pd.DataFrame,
    outcomes: pd.DataFrame,
    value: str,
    label: str,
    spec: PX9Spec,
    arrays: dict[str, NDArray],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for half in ("A", "B"):
            direction = f"{half}_to_{'B' if half == 'A' else 'A'}"
            merged = _forecast_frame(derived, outcomes, candidate, half)
            series = _matrix_centered_spearman(merged, value, "target_q")
            item = _bootstrap_summary(
                series,
                spec.bootstrap_draws,
                f"{label}/forecast/c{candidate}/{direction}",
                arrays,
            )
            item.update({"candidate": candidate, "direction": direction})
            items.append(item)
    _adjust_family(items)
    return items


def _dose_concordance_family(
    derived: pd.DataFrame,
    outcomes: pd.DataFrame,
    value: str,
    label: str,
    spec: PX9Spec,
    arrays: dict[str, NDArray],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for half in ("A", "B"):
            direction = f"{half}_to_{'B' if half == 'A' else 'A'}"
            merged = _forecast_frame(derived, outcomes, candidate, half)
            merged = merged[merged["arm"].isin(QUANTILE_ARMS)]
            series = _statewise_spearman(
                merged, value, "target_q", require_quantiles=True
            )
            item = _bootstrap_summary(
                series,
                spec.bootstrap_draws,
                f"{label}/dose/c{candidate}/{direction}",
                arrays,
            )
            item.update({"candidate": candidate, "direction": direction})
            items.append(item)
    _adjust_family(items)
    return items


PROCESS_BASELINE_FEATURES = (
    "prediction_logit",
    "source_q",
    "source_inherited_fraction",
    "source_run5",
    "source_survived",
    "source_mean_h",
    "whole_mi",
    "active_dimensions",
    "transitions",
)


def _prepare_fold_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    left = train.loc[:, list(columns)].to_numpy(float)
    right = test.loc[:, list(columns)].to_numpy(float)
    medians = np.nanmedian(left, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    left = np.where(np.isfinite(left), left, medians)
    right = np.where(np.isfinite(right), right, medians)
    means = left.mean(axis=0)
    scales = left.std(axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
    return (left - means) / scales, (right - means) / scales


def _crossfit_process_gain(frame: pd.DataFrame) -> pd.Series:
    selected = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["prediction", "target_successes", "target_trials", "temporal_value"]
    ).copy()
    if selected.empty:
        return pd.Series(dtype=float)
    selected["prediction_logit"] = logit(
        np.clip(selected["prediction"].to_numpy(float), 1e-6, 1 - 1e-6)
    )
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        train = selected[selected["matrix_id"] % 5 != fold]
        test = selected[selected["matrix_id"] % 5 == fold]
        if train.empty or test.empty:
            continue
        base_train, base_test = _prepare_fold_features(
            train, test, PROCESS_BASELINE_FEATURES
        )
        full_train, full_test = _prepare_fold_features(
            train, test, (*PROCESS_BASELINE_FEATURES, "temporal_value")
        )
        successes = train["target_successes"].to_numpy(float)
        trials = train["target_trials"].to_numpy(float)
        labels = np.tile(np.asarray([1, 0], dtype=int), len(train))
        weights = np.column_stack((successes, trials - successes)).reshape(-1)
        if weights[labels == 1].sum() <= 0 or weights[labels == 0].sum() <= 0:
            continue
        base_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        full_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        try:
            base_model.fit(
                np.repeat(base_train, 2, axis=0), labels, sample_weight=weights
            )
            full_model.fit(
                np.repeat(full_train, 2, axis=0), labels, sample_weight=weights
            )
        except (ValueError, FloatingPointError):
            continue
        p_base = np.clip(
            base_model.predict_proba(base_test)[:, 1], 1e-9, 1 - 1e-9
        )
        p_full = np.clip(
            full_model.predict_proba(full_test)[:, 1], 1e-9, 1 - 1e-9
        )
        q = (
            test["target_successes"].to_numpy(float)
            / test["target_trials"].to_numpy(float)
        )
        base_loss = -(q * np.log(p_base) + (1 - q) * np.log(1 - p_base))
        full_loss = -(q * np.log(p_full) + (1 - q) * np.log(1 - p_full))
        for matrix_id, gain in zip(
            test["matrix_id"].to_numpy(int), base_loss - full_loss, strict=True
        ):
            rows.append({"matrix_id": int(matrix_id), "gain": float(gain)})
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).groupby("matrix_id")["gain"].mean().sort_index()


def _nonredundancy_family(
    derived: pd.DataFrame,
    outcomes: pd.DataFrame,
    spec: PX9Spec,
    arrays: dict[str, NDArray],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for half in ("A", "B"):
            direction = f"{half}_to_{'B' if half == 'A' else 'A'}"
            merged = _forecast_frame(derived, outcomes, candidate, half)
            series = _crossfit_process_gain(merged)
            item = _bootstrap_summary(
                series,
                spec.bootstrap_draws,
                f"nonredundancy/c{candidate}/{direction}",
                arrays,
            )
            item.update({"candidate": candidate, "direction": direction})
            items.append(item)
    _adjust_family(items)
    return items


def _raw_support_analysis(
    scores: pd.DataFrame,
    spec: PX9Spec,
    arrays: dict[str, NDArray],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for support in _supports(spec):
        for candidate in CANDIDATES:
            for half in ("A", "B"):
                series = _matrix_arm_effect(
                    scores,
                    "value",
                    STABILIZING,
                    DESTABILIZING,
                    {
                        "support_branches": support,
                        "candidate": candidate,
                        "source_half": half,
                        "score_kind": "paired_beta",
                    },
                )
                item = _bootstrap_summary(
                    series,
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
                items.append(item)
    return items


def _negative_control_analysis(
    scores: pd.DataFrame,
    spec: PX9Spec,
    arrays: dict[str, NDArray],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    support = max(_supports(spec))
    for candidate in CANDIDATES:
        for half in ("A", "B"):
            series = _matrix_arm_effect(
                scores,
                "value",
                STABILIZING,
                DESTABILIZING,
                {
                    "support_branches": support,
                    "candidate": candidate,
                    "source_half": half,
                    "score_kind": "public_revised",
                },
            )
            item = _bootstrap_summary(
                series,
                spec.bootstrap_draws,
                f"public_revised/c{candidate}/h{half}",
                arrays,
            )
            item.update({"candidate": candidate, "source_half": half})
            items.append(item)
    _adjust_family(items)
    return items


def _concordant_outcome_analysis(
    scores: pd.DataFrame,
    spec: PX9Spec,
    arrays: dict[str, NDArray],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    support = max(_supports(spec))
    for candidate in CANDIDATES:
        for half in ("A", "B"):
            series = _matrix_arm_effect(
                scores,
                "value",
                STABILIZING,
                DESTABILIZING,
                {
                    "support_branches": support,
                    "candidate": candidate,
                    "source_half": half,
                    "score_kind": "concordant_outcome_beta",
                },
            )
            item = _bootstrap_summary(
                series,
                spec.bootstrap_draws,
                f"concordant/c{candidate}/h{half}",
                arrays,
            )
            local = scores[
                (scores["support_branches"] == support)
                & (scores["candidate"] == candidate)
                & (scores["source_half"] == half)
                & (scores["score_kind"] == "concordant_outcome_beta")
            ]
            item.update(
                {
                    "candidate": candidate,
                    "source_half": half,
                    "median_concordant_branches": float(
                        local["branches_used"].median()
                    )
                    if len(local)
                    else float("nan"),
                    "descriptive_post_treatment_only": True,
                }
            )
            items.append(item)
    return items


def _secondary_contrasts(
    derived: pd.DataFrame,
    spec: PX9Spec,
    arrays: dict[str, NDArray],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for half in ("A", "B"):
            filters = {"candidate": candidate, "source_half": half}
            for value in (
                "paired_value",
                "temporal_value",
                "topology_value",
                "cross_beta_excess",
                "whole_mi",
                "within_sum_mi",
            ):
                for high, low, name in (
                    ("Q100", "NOOP", "q100_minus_noop"),
                    ("NOOP", "Q00", "noop_minus_q00"),
                    ("RANDOM", "NOOP", "random_minus_noop"),
                ):
                    series = _matrix_arm_effect(
                        derived, value, high, low, filters
                    )
                    item = _bootstrap_summary(
                        series,
                        spec.bootstrap_draws,
                        f"secondary/{value}/{name}/c{candidate}/h{half}",
                        arrays,
                    )
                    item.update(
                        {
                            "candidate": candidate,
                            "source_half": half,
                            "metric": value,
                            "contrast": name,
                        }
                    )
                    items.append(item)
    return items


def _score_completeness(derived: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for half in ("A", "B"):
            local = derived[
                (derived["candidate"] == candidate)
                & (derived["source_half"] == half)
            ]
            rows.append(
                {
                    "candidate": candidate,
                    "source_half": half,
                    "rows": int(len(local)),
                    "temporal_finite_fraction": float(
                        np.isfinite(local["temporal_value"].to_numpy(float)).mean()
                    )
                    if len(local)
                    else 0.0,
                    "topology_finite_fraction": float(
                        np.isfinite(local["topology_value"].to_numpy(float)).mean()
                    )
                    if len(local)
                    else 0.0,
                    "all_temporal_controls_present": bool(
                        len(local)
                        and (local["shuffled_controls"] == len(TEMPORAL_SHIFTS)).all()
                    ),
                    "all_random_partitions_present": bool(
                        len(local)
                        and (
                            local["random_partition_controls"] == RANDOM_PARTITIONS
                        ).all()
                    ),
                }
            )
    return rows


def analyze_batches(
    batches: Sequence[PX9Batch], spec: PX9Spec
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    acquisitions = pd.DataFrame(
        [row for batch in batches for row in batch.acquisition_rows]
    )
    edits = pd.DataFrame([row for batch in batches for row in batch.edit_rows])
    branches = pd.DataFrame([row for batch in batches for row in batch.branch_rows])
    scores = pd.DataFrame([row for batch in batches for row in batch.score_rows])
    outcomes = _outcome_frame(branches, edits)
    derived = _derived_score_frame(scores, max(_supports(spec)))
    arrays: dict[str, NDArray] = {}

    eligibility: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        local = acquisitions[
            (acquisitions["candidate"] == candidate)
            & (acquisitions["eligible"] == 1)
        ]
        matrices = int(local["matrix_id"].nunique())
        eligibility[candidate] = {
            "eligible_states": int(len(local)),
            "eligible_matrices": matrices,
            "minimum": MINIMUM_ELIGIBLE_MATRICES,
            "pass": matrices >= MINIMUM_ELIGIBLE_MATRICES,
        }
    eligibility_pass = bool(all(item["pass"] for item in eligibility.values()))

    outcome_extreme, outcome_dose, manipulation_valid = _outcome_validity(
        outcomes, spec, arrays
    )
    temporal_response = _response_family(
        derived, "temporal_value", "temporal", spec, arrays
    )
    temporal_reliability = _reliability_family(
        derived, "temporal_value", "temporal", spec, arrays
    )
    temporal_forecast = _forecast_family(
        derived, outcomes, "temporal_value", "temporal", spec, arrays
    )
    temporal_dose = _dose_concordance_family(
        derived, outcomes, "temporal_value", "temporal", spec, arrays
    )
    topology_response = _response_family(
        derived, "topology_value", "topology", spec, arrays
    )
    topology_forecast = _forecast_family(
        derived, outcomes, "topology_value", "topology", spec, arrays
    )
    nonredundancy = _nonredundancy_family(derived, outcomes, spec, arrays)
    support = _raw_support_analysis(scores, spec, arrays)
    public_control = _negative_control_analysis(scores, spec, arrays)
    concordant = _concordant_outcome_analysis(scores, spec, arrays)
    secondary = _secondary_contrasts(derived, spec, arrays)
    completeness = _score_completeness(derived)

    temporal_complete = bool(
        len(completeness) == 4
        and all(
            item["temporal_finite_fraction"] >= 0.95
            and item["all_temporal_controls_present"]
            for item in completeness
        )
    )
    topology_complete = bool(
        len(completeness) == 4
        and all(
            item["topology_finite_fraction"] >= 0.95
            and item["all_random_partitions_present"]
            for item in completeness
        )
    )

    temporal_authenticity = bool(
        temporal_complete
        and all(
            len(items) == expected and all(item["pass"] for item in items)
            for items, expected in (
                (temporal_response, 4),
                (temporal_reliability, 2),
                (temporal_forecast, 4),
                (temporal_dose, 4),
            )
        )
    )
    topology_specificity = bool(
        topology_complete
        and len(topology_response) == 4
        and len(topology_forecast) == 4
        and all(item["pass"] for item in topology_response)
        and all(item["pass"] for item in topology_forecast)
    )
    behaviorally_nonredundant = bool(
        len(nonredundancy) == 4 and all(item["pass"] for item in nonredundancy)
    )
    eligible_manipulation = bool(eligibility_pass and manipulation_valid)
    if not eligible_manipulation:
        classification = "plastic_heredity_manipulation_invalid"
    elif not temporal_authenticity:
        classification = "finite_sample_or_marginal_explanation"
    elif behaviorally_nonredundant and topology_specificity:
        classification = "beta_topology_specific_nonredundant_gauge"
    elif behaviorally_nonredundant:
        classification = "generic_transition_information_gauge"
    else:
        classification = "reliable_behavioral_echo"
    gates = {
        "eligibility": eligibility_pass,
        "plastic_heredity_manipulation_valid": eligible_manipulation,
        "temporal_score_complete": temporal_complete,
        "topology_score_complete": topology_complete,
        "temporal_authenticity": temporal_authenticity,
        "beta_topology_specificity": topology_specificity,
        "behaviorally_nonredundant": behaviorally_nonredundant,
        "public_revised_positive_all_cells": bool(
            len(public_control) == 4 and all(item["pass"] for item in public_control)
        ),
        "pilot_classification": classification,
        "automatic_48_matrix_continuation_authorized": False,
    }
    matrix_rows: list[dict[str, Any]] = []
    for key, values in arrays.items():
        if not key.endswith("__matrix_values"):
            continue
        ids_key = key.removesuffix("__matrix_values") + "__matrix_ids"
        if ids_key not in arrays:
            continue
        family = key.removesuffix("__matrix_values")
        matrix_rows.extend(
            {
                "family": family,
                "matrix_id": int(matrix_id),
                "value": float(value),
            }
            for matrix_id, value in zip(arrays[ids_key], values, strict=True)
        )
    metrics = {
        "format": "codex-ch5-phir-px9-primary-metrics-v1",
        "eligibility": eligibility,
        "outcome_extreme": outcome_extreme,
        "outcome_dose": outcome_dose,
        "temporal_response": temporal_response,
        "temporal_reliability": temporal_reliability,
        "temporal_forecast": temporal_forecast,
        "temporal_dose_concordance": temporal_dose,
        "topology_response": topology_response,
        "topology_forecast": topology_forecast,
        "nonredundancy_log_loss": nonredundancy,
        "raw_support_response": support,
        "public_revised_negative_control": public_control,
        "concordant_outcome_diagnostic": concordant,
        "secondary_contrasts": secondary,
        "score_completeness": completeness,
        "gates": gates,
    }
    tables = {
        "acquisition": acquisitions,
        "selected_edits": edits,
        "branches": branches,
        "state_outcomes": outcomes,
        "state_scores": scores,
        "derived_scores": derived,
        "matrix_effects": pd.DataFrame(matrix_rows),
    }
    return metrics, tables, arrays


def _validation_fixture() -> dict[str, Any]:
    spec = smoke_spec()
    rng = np.random.default_rng(_seed(spec, "validation", "fixture"))
    counts = rng.poisson(2.0, size=(1025, GardConfig().n_types)).astype(np.int16)
    counts[counts.sum(axis=1) == 0, 0] = 1
    beta = np.exp(rng.normal(-4.0, 1.0, size=(100, 100)))
    composition = np.zeros(100, dtype=np.int64)
    composition[:40] = 1
    case = ResilienceCase(
        "PX9-fixture",
        "02",
        0,
        20,
        beta,
        Snapshot(composition, 20, (True,) * 20, (0.95,) * 20),
        counts[-PAST_WINDOW:],
    )
    first, second = beta_physical_partition(beta)
    frozen = px7._score_pairs(counts[:-1], counts[1:], "beta", case)
    direct = _fixed_partition_score(counts[:-1], counts[1:], first, second)
    blocks = [
        PairBlock(
            counts[index * 4 : index * 4 + 3],
            counts[index * 4 + 1 : index * 4 + 4],
            counts[index * 4 : index * 4 + 3],
            counts[index * 4 + 1 : index * 4 + 4],
        )
        for index in range(8)
    ]
    actual_past, actual_future = _concatenate_pairs(blocks, "generational")
    shuffled_past, shuffled_future, self_pairs = _temporal_derangement(blocks, 1)
    actual_future_rows = sorted(row.tobytes() for row in actual_future)
    shuffled_future_rows = sorted(row.tobytes() for row in shuffled_future)
    partitions = _random_partitions(case, spec)
    synthetic = tuple(
        ScoredEdit(MolecularEdit(index, index + 1), index / 9.0, index / 9.0 - 0.5)
        for index in range(10)
    )
    selected_left, ranks_left = cr2.select_quantile_edits(synthetic)
    selected_right, ranks_right = cr2.select_quantile_edits(synthetic)
    permutation_rng = np.random.default_rng(901)
    permutation = np.concatenate(
        (permutation_rng.permutation(99), np.asarray([99], dtype=np.int64))
    )
    inverse = np.argsort(permutation)
    permuted = _fixed_partition_score(
        counts[:-1, permutation],
        counts[1:, permutation],
        inverse[first],
        inverse[second],
    )
    noop_spec = PX9Spec("smoke", 1, (20,), 2, 1, 1, 8, 8)
    noop_row, noop_pairs = _simulate_branch(case, None, 0, noop_spec)
    plain = advance_fission_traced(
        case.snapshot.composition,
        case.beta,
        GardConfig(),
        CANDIDATES[case.candidate],
        np.random.default_rng(_future_seed(noop_spec, case, 0)),
    )
    partition_ok = all(
        len(left) == len(first)
        and len(right) == len(second)
        and not set(left).intersection(right)
        and set(left).union(right) == set(range(GardConfig().n_types))
        for left, right in partitions
    )
    partition_digests = {
        _array_digest(left, right) for left, right in partitions
    }
    checks = {
        "pilot_matrix_count_exact": MATRICES == 24,
        "manual_48_barrier": protocol()["cohort"]["automatic_48_matrix_continuation"]
        is False,
        "eight_arms_exact": ARMS
        == ("Q00", "Q20", "Q40", "Q60", "Q80", "Q100", "RANDOM", "NOOP"),
        "quantiles_exact": QUANTILES == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "quantile_selection_deterministic": ranks_left.tolist()
        == ranks_right.tolist()
        and selected_left == selected_right,
        "quantile_selection_monotone": all(
            left.predicted_probability <= right.predicted_probability
            for left, right in zip(selected_left[:-1], selected_left[1:], strict=True)
        ),
        "halves_disjoint_complete": set(HALVES["A"]).isdisjoint(HALVES["B"])
        and set(HALVES["A"]).union(HALVES["B"]) == set(range(BRANCHES)),
        "support_exact": _supports(scientific_spec()) == SUPPORT_LEVELS,
        "fixed_scorer_matches_px8": abs(direct["value"] - frozen["typeset"])
        < 1e-10,
        "fixed_scorer_whole_matches": abs(direct["whole_mi"] - frozen["whole_mi"])
        < 1e-10,
        "fixed_dropped_coordinate_label_invariance": abs(
            direct["value"] - permuted["value"]
        )
        < 1e-8,
        "temporal_shuffle_preserves_past_count": len(shuffled_past)
        == len(actual_past),
        "temporal_shuffle_preserves_future_marginal": shuffled_future_rows
        == actual_future_rows,
        "temporal_shuffle_has_no_self_pairs": self_pairs == 0,
        "random_partitions_count": len(partitions) == RANDOM_PARTITIONS,
        "random_partitions_unique": len(partition_digests) == RANDOM_PARTITIONS,
        "random_partitions_size_matched_complete": partition_ok,
        "random_partitions_deterministic": all(
            np.array_equal(left, again_left)
            and np.array_equal(right, again_right)
            for (left, right), (again_left, again_right) in zip(
                partitions, _random_partitions(case, spec), strict=True
            )
        ),
        "future_seed_arm_free": "arm" not in inspect.signature(_future_seed).parameters,
        "future_and_action_streams_distinct": _future_seed(spec, case, 0)
        != _selection_seed(spec, case),
        "noop_matches_plain_simulator": np.array_equal(
            noop_pairs.generational_past[0], plain.record.parent
        )
        and np.array_equal(noop_pairs.generational_future[0], plain.record.daughter),
        "run3_cannot_certify_in_one_fission": noop_row["primary"] == 0,
        "strict_threshold": not bool(0.9 > GardConfig().inheritance_threshold)
        and bool(np.nextafter(0.9, 1.0) > GardConfig().inheritance_threshold),
        "model_sources_exist": MODEL_SOURCE.is_file()
        and MODEL_CONTRACT_SOURCE.is_file(),
        "disk_available_for_compact_artifacts": shutil.disk_usage(ROOT).free
        >= MINIMUM_FREE_DISK_BYTES,
    }
    return {
        "checks": checks,
        "fixture_digest": _digest(
            {
                "direct": direct,
                "frozen": frozen,
                "partitions": partitions,
                "shuffled": _array_digest(shuffled_past, shuffled_future),
            }
        ),
    }


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
        "format": "codex-ch5-phir-px9-validation-v1",
        "checks": fixture["checks"],
        "all_checks_passed": bool(all(fixture["checks"].values())),
        "source_hashes": _source_hashes(),
        "model_sha256": sha256_file(MODEL_SOURCE),
        "model_contract_sha256": sha256_file(MODEL_CONTRACT_SOURCE),
        "fixture_digest": fixture["fixture_digest"],
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
        raise ValueError("PX9 validation did not pass")
    if validation["source_hashes"] != _source_hashes():
        raise ValueError("PX9 source changed after validation")
    if output.exists() or DEFAULT_OUTPUT.exists():
        raise FileExistsError("PX9 registration or output already exists")
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
        f"<!-- phir-extension-px9-registration-{payload['registration_id']} -->",
        [
            "## Phi-r extension PX9 registered",
            "",
            f"- Registration: `{payload['registration_id']}`.",
            "- Twenty-four fresh matrices, eight frozen edit-dose arms, temporal derangements, size-matched partition controls, and an ordinary-process nonredundancy test were sealed.",
            "- This is a pilot with a manual barrier; no 48-matrix continuation is authorized.",
            "- No PX9 scientific matrix or outcome existed at registration.",
        ],
    )
    return payload


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    if payload.get("format") != REGISTRATION_FORMAT:
        raise ValueError("unsupported PX9 registration")
    if payload["source_hashes"] != _source_hashes():
        raise ValueError("PX9 source changed after registration")
    if payload["protocol"] != protocol():
        raise ValueError("PX9 protocol changed after registration")
    expected = _digest(
        {key: value for key, value in payload.items() if key != "registration_id"}
    )
    if payload["registration_id"] != expected:
        raise ValueError("PX9 registration ID mismatch")
    if sha256_file(directory / "frozen_cr5_students.npz") != payload["model_sha256"]:
        raise ValueError("PX9 frozen model archive changed")
    if (
        sha256_file(directory / "model_contract.json")
        != payload["model_contract_sha256"]
    ):
        raise ValueError("PX9 frozen model contract changed")
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
    projected_fissions = (
        2 * MATRICES * len(CANDIDATES) * len(LANDMARKS) * len(ARMS) * BRANCHES * HORIZON
    )
    projected_cpu = float(
        first.cpu_seconds * projected_fissions / smoke_fissions / 3600.0
    )
    score_kinds = {row["score_kind"] for row in first.score_rows}
    required = {
        "paired_beta",
        "shuffled_beta",
        "random_partition",
        "cross_beta_partition",
        "public_revised",
        "concordant_outcome_beta",
    }
    payload = {
        "format": "codex-ch5-phir-px9-smoke-v1",
        "registration_id": registration["registration_id"],
        "exact_replay": first.scientific_digest == second.scientific_digest,
        "branches_created": len(first.branch_rows),
        "score_rows_created": len(first.score_rows),
        "all_score_kinds_exercised": required.issubset(score_kinds),
        "all_arms_exercised": {row["arm"] for row in first.edit_rows} == set(ARMS),
        "cpu_seconds_per_smoke_matrix": first.cpu_seconds,
        "projected_complete_cpu_hours": projected_cpu,
        "projected_within_cpu_ceiling": projected_cpu <= MAX_CPU_HOURS,
        "scientific_effects_disclosed": False,
    }
    if not (
        payload["exact_replay"]
        and payload["branches_created"] > 0
        and payload["score_rows_created"] > 0
        and payload["all_score_kinds_exercised"]
        and payload["all_arms_exercised"]
        and payload["projected_within_cpu_ceiling"]
    ):
        raise AssertionError("PX9 smoke gate failed")
    output.mkdir(parents=True)
    _atomic_json(output / "smoke.json", payload)
    write_checksums(output)
    verify_checksums(output)
    return payload


def _checkpoint_contract(
    registration_id: str, spec: PX9Spec, stage: str
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
    spec: PX9Spec,
    cpu_budget_hours: float,
) -> None:
    work.mkdir(parents=True, exist_ok=True)
    contract_path = work / "checkpoint_contract.json"
    expected = _checkpoint_contract(registration_id, spec, "generation")
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != expected:
            raise ValueError("PX9 checkpoint contract mismatch")
    else:
        _atomic_json(contract_path, expected)
    budget_path = work / "cpu_budget.json"
    budget = {
        "maximum_cpu_hours": MAX_CPU_HOURS,
        "declared_cpu_hours": cpu_budget_hours,
    }
    if cpu_budget_hours <= 0 or cpu_budget_hours > MAX_CPU_HOURS:
        raise ValueError("PX9 CPU budget must be in (0, 30]")
    if budget_path.exists():
        if json.loads(budget_path.read_text(encoding="utf-8")) != budget:
            raise ValueError("PX9 CPU budget changed after launch")
    else:
        _atomic_json(budget_path, budget)


def _status_write(work: Path, payload: dict[str, Any]) -> None:
    _atomic_json(
        work / "status.json",
        {"format": STATUS_FORMAT, "updated_at_unix": time.time(), **payload},
    )


class _PX9Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "PX9Batch":
            return PX9Batch
        return super().find_class(module, name)


def _load_checkpoint(path: Path) -> PX9Batch:
    with path.open("rb") as handle:
        value = _PX9Unpickler(handle).load()
    if not isinstance(value, PX9Batch):
        raise TypeError(f"unexpected PX9 checkpoint type: {path}")
    return value


def _run_checkpoint_stage(
    spec: PX9Spec,
    directory: Path,
    workers: int,
    model_path: Path,
    contract_path: Path,
    work: Path,
    stage: str,
    cpu_budget_seconds: float,
) -> list[PX9Batch]:
    directory.mkdir(parents=True, exist_ok=True)
    batches: list[PX9Batch | None] = [None] * spec.matrices
    for matrix_id in range(spec.matrices):
        path = directory / f"matrix_{matrix_id:03d}.pkl"
        if path.exists():
            batch = _load_checkpoint(path)
            if batch.matrix_id != matrix_id:
                raise ValueError("PX9 checkpoint matrix ID mismatch")
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
            generated: Iterable[PX9Batch] = map(_run_matrix, arguments)
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
                    raise RuntimeError("PX9 declared CPU budget exhausted")
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
    if any(batch is None for batch in batches):
        raise AssertionError("PX9 checkpoint stage incomplete")
    return [batch for batch in batches if batch is not None]


def _effect_text(item: Mapping[str, Any]) -> str:
    return (
        f"{float(item['effect']):+.4f} "
        f"[{float(item['ci95'][0]):+.4f}, {float(item['ci95'][1]):+.4f}]"
    )


def _family_table(
    items: Sequence[Mapping[str, Any]],
    identity: str,
) -> list[str]:
    lines: list[str] = []
    for item in items:
        cell = str(item.get("candidate", ""))
        detail = str(
            item.get(
                "source_half",
                item.get("target_half", item.get("direction", "")),
            )
        )
        lines.append(
            f"| {identity} | {cell} | {detail} | {_effect_text(item)} | "
            f"{float(item.get('holm_adjusted_p', float('nan'))):.4g} | "
            f"{item.get('pass', False)} |"
        )
    return lines


def _reports(metrics: dict[str, Any], registration_id: str) -> tuple[str, str]:
    gates = metrics["gates"]
    outcome_lines = _family_table(metrics["outcome_extreme"], "renewal Q100-Q00")
    dose_lines = _family_table(metrics["outcome_dose"], "renewal dose Spearman")
    temporal_lines = [
        *_family_table(metrics["temporal_response"], "temporal response"),
        *_family_table(metrics["temporal_reliability"], "temporal reliability"),
        *_family_table(metrics["temporal_forecast"], "temporal forecast"),
        *_family_table(metrics["temporal_dose_concordance"], "temporal dose"),
    ]
    topology_lines = [
        *_family_table(metrics["topology_response"], "topology response"),
        *_family_table(metrics["topology_forecast"], "topology forecast"),
    ]
    nonredundancy_lines = _family_table(
        metrics["nonredundancy_log_loss"], "incremental log loss"
    )
    support_lines = [
        f"| {item['support_branches']} | {item['candidate']} | "
        f"{item['source_half']} | {_effect_text(item)} |"
        for item in metrics["raw_support_response"]
    ]
    control_lines = [
        f"| {item['candidate']} | {item['source_half']} | {_effect_text(item)} |"
        for item in metrics["public_revised_negative_control"]
    ]
    concordant_lines = [
        f"| {item['candidate']} | {item['source_half']} | {_effect_text(item)} | "
        f"{item['median_concordant_branches']:.1f} |"
        for item in metrics["concordant_outcome_diagnostic"]
    ]
    technical = "\n".join(
        (
            "# PX9 high-support gauge-identity pilot",
            "",
            f"Registration: `{registration_id}`.",
            "",
            "## Plastic-heredity dose validity",
            "",
            "| Test | Candidate | Cell | Effect [95% CI] | Holm p | Pass |",
            "| --- | --- | --- | ---: | ---: | --- |",
            *outcome_lines,
            *dose_lines,
            "",
            "## Temporal authenticity",
            "",
            "| Test | Candidate | Cell | Effect [95% CI] | Holm p | Pass |",
            "| --- | --- | --- | ---: | ---: | --- |",
            *temporal_lines,
            "",
            "## Beta-topology specificity",
            "",
            "| Test | Candidate | Cell | Effect [95% CI] | Holm p | Pass |",
            "| --- | --- | --- | ---: | ---: | --- |",
            *topology_lines,
            "",
            "## Behavioral nonredundancy",
            "",
            "| Test | Candidate | Direction | Log-loss gain [95% CI] | Holm p | Pass |",
            "| --- | --- | --- | ---: | ---: | --- |",
            *nonredundancy_lines,
            "",
            "## Raw extension support check",
            "",
            "| Source branches | Candidate | Half | Q100-Q00 [95% CI] |",
            "| ---: | --- | --- | ---: |",
            *support_lines,
            "",
            "## Public revised negative control",
            "",
            "| Candidate | Half | Q100-Q00 [95% CI] |",
            "| --- | --- | ---: |",
            *control_lines,
            "",
            "## Concordant-outcome descriptive diagnostic",
            "",
            "| Candidate | Half | Q100-Q00 [95% CI] | Median matched branches |",
            "| --- | --- | ---: | ---: |",
            *concordant_lines,
            "",
            "This diagnostic conditions on post-treatment renewal and survival agreement and is not a causal mediation analysis.",
            "",
            "## Registered gates and classification",
            "",
            *(f"- {key}: **{value}**" for key, value in gates.items()),
            "",
            "## Claim boundary",
            "",
            "PX9 is a prospective 24-matrix mechanistic pilot. It identifies what the PX8 extension behaves like; it cannot rescue the public nine-atom Phi-r, make an information statistic causal, or automatically authorize a 48-matrix continuation.",
            "",
        )
    )
    classification = gates["pilot_classification"]
    explanations = {
        "beta_topology_specific_nonredundant_gauge": (
            "The PX8 reading survived temporal shuffling controls, specifically depended on the catalytic-network partition, and added information beyond ordinary recovery summaries."
        ),
        "generic_transition_information_gauge": (
            "The reading carried nonredundant temporal information, but arbitrary molecule partitions worked about as well as the beta partition."
        ),
        "reliable_behavioral_echo": (
            "The reading reliably followed real parent-to-daughter recovery dynamics, but did not add information beyond simpler observations of how often heredity recovered."
        ),
        "finite_sample_or_marginal_explanation": (
            "The molecular dose changed recovery, but the apparent information response did not survive the controls that break real parent-to-daughter pairing."
        ),
        "plastic_heredity_manipulation_invalid": (
            "The registered molecular dose did not cleanly manipulate recovery in every required cell, so the information comparison cannot adjudicate the gauge."
        ),
    }
    lay = "\n".join(
        (
            "# Lay summary — PX9 gauge identity pilot",
            "",
            "PX9 asked whether the promising PX8 information number is a genuine extra thermometer for hereditary recovery, or simply a complicated restatement of behavior we can already count directly.",
            "",
            explanations[classification],
            "",
            "This 24-matrix result is a pilot. It does not launch a larger confirmation automatically, and it does not turn Phi-r into the cause of plastic heredity.",
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
        raise ValueError(f"PX9 workers must be in [1,{MAX_WORKERS}]")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise OSError("PX9 compact-artifact free-disk gate failed")
    if shutil.disk_usage(work.parent if work.parent.exists() else ROOT).free < 600_000_000:
        raise OSError("PX9 work-volume free-disk gate failed")
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
            "format": "codex-ch5-phir-px9-replay-v1",
            "matrices": replay_rows,
            "complete_exact_replay": bool(
                len(replay_rows) == spec.matrices
                and all(item["exact"] for item in replay_rows)
            ),
        }
        if not replay_audit["complete_exact_replay"]:
            raise AssertionError("PX9 complete replay failed")
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
            "arms": list(ARMS),
            "branches_per_arm": spec.branches,
            "generation_cpu_seconds": generation_cpu,
            "replay_cpu_seconds": replay_cpu,
            "workers": workers,
            "declared_cpu_budget_hours": cpu_budget_hours,
            "work_directory": str(work),
            "scientific_digest": scientific_digest,
            "gates": metrics["gates"],
        }
        _atomic_json(staging / "manifest.json", manifest)
        technical, lay = _reports(metrics, registration["registration_id"])
        (staging / "SCIENTIFIC_REPORT.md").write_text(technical, encoding="utf-8")
        (staging / "LAY_SUMMARY.md").write_text(lay, encoding="utf-8")
        _atomic_json(
            staging / "claim_boundaries.json",
            {
                "supported": [metrics["gates"]["pilot_classification"]],
                "not_supported": [
                    "public nine-atom Phi-r rescue",
                    "Phi-r causation",
                    "consciousness, agency, or life",
                    "automatic 48-matrix continuation",
                ],
            },
        )
        readback = {
            "format": "codex-ch5-phir-px9-readback-v1",
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
            raise AssertionError("PX9 artifact readback found an empty table")
        _atomic_json(staging / "readback_audit.json", readback)
        write_checksums(staging)
        if output.exists():
            raise FileExistsError(output)
        local_staging = output.with_name(f".{output.name}.staging")
        if local_staging.exists():
            shutil.rmtree(local_staging)
        shutil.copytree(staging, local_staging)
        local_staging.replace(output)
        verify_result(output, registration_directory)
        shutil.rmtree(staging)
        _append_ledger(
            f"<!-- phir-extension-px9-result-{registration['registration_id']} -->",
            [
                "## Phi-r extension PX9 completed",
                "",
                f"- Result: `{output.relative_to(ROOT)}`.",
                f"- Gates: `{json.dumps(metrics['gates'], sort_keys=True)}`.",
                "- The 24-matrix gauge-identity pilot received complete exact replay and artifact readback.",
                "- No 48-matrix continuation was launched or authorized.",
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
        raise ValueError("unsupported PX9 result")
    if manifest["registration_id"] != registration["registration_id"]:
        raise ValueError("PX9 result registration mismatch")
    if not replay["complete_exact_replay"]:
        raise ValueError("PX9 result lacks complete replay")
    if not readback["all_tables_nonempty"]:
        raise ValueError("PX9 result readback failed")
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
    work: Path = DEFAULT_WORK,
) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(DEFAULT_OUTPUT)
    launch_path = work / "detached_launch.json"
    if launch_path.exists():
        existing = json.loads(launch_path.read_text(encoding="utf-8"))
        if _pid_alive(int(existing.get("pid", -1))):
            raise RuntimeError(f"PX9 already runs as PID {existing['pid']}")
    work.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "plastic_heredity.phir_extension_px9",
        "run",
        "--workers",
        str(workers),
        "--cpu-budget-hours",
        str(cpu_budget_hours),
        "--work-dir",
        str(work),
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
        "format": "codex-ch5-phir-px9-detached-launch-v1",
        "registration_id": registration["registration_id"],
        "pid": process.pid,
        "workers": workers,
        "cpu_budget_hours": cpu_budget_hours,
        "work": str(work),
        "command": command,
        "log": str(DEFAULT_LOG),
        "launched_at_unix": time.time(),
    }
    _atomic_json(launch_path, payload)
    return payload


def status(work: Path = DEFAULT_WORK) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": "codex-ch5-phir-px9-status-report-v1",
        "validation": DEFAULT_VALIDATION.exists(),
        "registration": DEFAULT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "complete": DEFAULT_OUTPUT.exists(),
        "work": str(work),
        "log": str(DEFAULT_LOG),
    }
    launch_path = work / "detached_launch.json"
    if launch_path.exists():
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        payload["launch"] = launch
        payload["pid_alive"] = _pid_alive(int(launch.get("pid", -1)))
    status_path = work / "status.json"
    if status_path.exists():
        payload["work_status"] = json.loads(status_path.read_text(encoding="utf-8"))
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
    run_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    launch_parser.add_argument("--cpu-budget-hours", type=float, default=MAX_CPU_HOURS)
    launch_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
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
        value = run(
            workers=args.workers,
            cpu_budget_hours=args.cpu_budget_hours,
            work=args.work_dir,
        )
    elif args.command == "launch":
        value = launch_detached(
            args.workers, args.cpu_budget_hours, args.work_dir
        )
    elif args.command == "status":
        value = status(args.work_dir)
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
