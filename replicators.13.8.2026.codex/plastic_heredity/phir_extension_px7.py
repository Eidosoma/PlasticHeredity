"""PX7 prospective plastic-heredity axis gauge tournament.

This is an additive, clean-room phase.  It reuses immutable CR5 students but
creates fresh matrices, states, interventions, futures, and Phi-r readings.
Completed Chapter 5 and PX1--PX6 artifacts are read-only predecessors.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
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
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .config import CANDIDATES, GardConfig
from .experiment import StateCase
from .intervention_core import MolecularEdit, apply_molecular_edit
from .mechanistic import sha256_file, verify_checksums, write_checksums
from .mechanistic_metrics import holm_adjust
from .phir_ch5 import _append_ledger, _snapshot_after_record
from .phir_instruments import (
    ACTIVE_STD_EPS,
    ANTICHAINS,
    ATOM_NAMES,
    PHIR_ATOMS,
    SYNERGISTIC,
    UNIQUE_0,
    UNIQUE_1,
    advance_fission_traced,
    close_clr_drop_last,
    fiedler_bipartition,
    gaussian_mutual_information,
)
from .phir_rescue_instruments import _cached_local_phi_id_atoms, beta_physical_partition
from .seeds import derive_seed
from .simulator import (
    FissionRecord,
    SimulationError,
    Snapshot,
    generate_beta,
    generate_initial_composition,
)
from . import intervention_cr5 as cr5


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "phir_extension"
DOCUMENT = "CODEX_CH5_PHIR_PX7_PREREGISTRATION.md"
LEDGER = ROOT / "PHIR_RESULTS_LEDGER.md"

DEFAULT_VALIDATION = RESULT_ROOT / "px7_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "px7_registration"
DEFAULT_SMOKE = RESULT_ROOT / "px7_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "px7_axis_gauge_tournament"
DEFAULT_WORK = RESULT_ROOT / ".px7_axis_gauge_work"
DEFAULT_LOG = RESULT_ROOT / "px7_axis_gauge_tournament.log"

MODEL_SOURCE = ROOT / "results_intervention_replication" / "cr5_confirmation_registration" / "frozen_cr5_students.npz"
MODEL_CONTRACT_SOURCE = ROOT / "results_intervention_replication" / "cr5_confirmation_registration" / "model_contract.json"

PROGRAM_FORMAT = "codex-ch5-phir-px7-axis-gauge-v1"
REGISTRATION_FORMAT = "codex-ch5-phir-px7-registration-v1"
RESULT_FORMAT = "codex-ch5-phir-px7-result-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-px7-checkpoint-v1"
STATUS_FORMAT = "codex-ch5-phir-px7-status-v1"
LABEL = "CODEX_CH5_PHIR_PX7_AXIS_GAUGE_V1"

MATRICES = 24
LANDMARKS = (20, 35, 50, 65, 80)
BRANCHES = 64
HALVES = {"A": tuple(range(0, 32)), "B": tuple(range(32, 64))}
PREFIX_BRANCHES = 16
RESISTANCE_HORIZON = 6
RESILIENCE_HORIZON = 8
ACQUISITION_LIMIT = 60
PAST_WINDOW = 512
MINIMUM_ELIGIBLE_MATRICES = 20
BOOTSTRAP_DRAWS = 4096
RANDOMIZATION_DRAWS = 4096
OUTCOME_EQUIVALENCE_MARGIN = 0.025
PHI_EQUIVALENCE_SD = 0.20
MINIMUM_FINITE_FRACTION = 0.90
MAX_WORKERS = 8
MAX_CPU_HOURS = 30.0
MINIMUM_FREE_DISK_BYTES = 1_500_000_000

AXES = ("resistance", "resilience")
CLOCKS = ("molecular", "generational")
PARTITIONS = ("self", "past", "beta")
FUNCTIONALS = ("revised", "typeset", "ratio")
FORMULATIONS = tuple(
    f"{clock}__{partition}__{functional}"
    for clock in CLOCKS
    for partition in PARTITIONS
    for functional in FUNCTIONALS
)
ARMS = {
    "resistance": ("BREAK_UP", "BREAK_DOWN", "RANDOM", "NOOP"),
    "resilience": ("RENEWAL_UP", "RENEWAL_DOWN", "RANDOM", "NOOP"),
}
STABILIZING = {"resistance": "BREAK_DOWN", "resilience": "RENEWAL_UP"}
DESTABILIZING = {"resistance": "BREAK_UP", "resilience": "RENEWAL_DOWN"}

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/phir_extension_px7.py",
    "tests/test_phir_extension_px7.py",
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
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=True
        ).encode("utf-8")
    ).hexdigest()


def _array_digest(*arrays: NDArray) -> str:
    value = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        value.update(str(contiguous.dtype).encode("ascii"))
        value.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        value.update(contiguous.tobytes())
    return value.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=5)
    temporary.replace(path)


@dataclass(frozen=True)
class PX7Spec:
    label: str
    matrices: int
    landmarks: tuple[int, ...]
    branches: int
    resistance_horizon: int
    resilience_horizon: int
    acquisition_limit: int
    bootstrap_draws: int
    randomization_draws: int


def scientific_spec() -> PX7Spec:
    return PX7Spec(
        "scientific",
        MATRICES,
        LANDMARKS,
        BRANCHES,
        RESISTANCE_HORIZON,
        RESILIENCE_HORIZON,
        ACQUISITION_LIMIT,
        BOOTSTRAP_DRAWS,
        RANDOMIZATION_DRAWS,
    )


def smoke_spec() -> PX7Spec:
    # Landmark 20 plus the full bounded acquisition window deterministically
    # exercises both resistance and shared-break resilience with smoke seeds.
    return PX7Spec("smoke", 1, (20,), 4, 2, 3, 60, 32, 32)


@dataclass(frozen=True)
class GaugeCase:
    state_id: str
    candidate: str
    matrix_id: int
    landmark: int
    beta: NDArray[np.float64]
    snapshot: Snapshot
    history_counts: NDArray[np.int16]

    def as_state_case(self, cohort: str) -> StateCase:
        return StateCase(
            state_id=self.state_id,
            cohort=cohort,
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
class MatrixBatch:
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


def protocol(spec: PX7Spec | None = None) -> dict[str, Any]:
    spec = scientific_spec() if spec is None else spec
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "target": "prospective Phi-r gauge of resistance and resilience",
        "strict_eight_excluded": True,
        "cohort": {
            "matrices": spec.matrices,
            "candidates": list(CANDIDATES),
            "landmarks": list(spec.landmarks),
            "branches": spec.branches,
            "halves": {key: list(indices) for key, indices in HALVES.items()} if spec.branches == BRANCHES else {"A": [0, 1], "B": [2, 3]},
        },
        "axes": {
            "resistance": {
                "arms": list(ARMS["resistance"]),
                "horizon": spec.resistance_horizon,
                "endpoint": "complete F6 with no strict break",
            },
            "resilience": {
                "arms": list(ARMS["resilience"]),
                "horizon": spec.resilience_horizon,
                "endpoint": "run3 within F8 from identical post-break daughter",
                "acquisition_limit": spec.acquisition_limit,
                "minimum_eligible_matrices": MINIMUM_ELIGIBLE_MATRICES,
            },
        },
        "formulations": {
            "clocks": list(CLOCKS),
            "partitions": list(PARTITIONS),
            "functionals": list(FUNCTIONALS),
            "eligible": list(FORMULATIONS),
            "count": len(FORMULATIONS),
            "source_target_orientation": "past/source to future/target",
            "atom_subset_selection": False,
            "source_half_prefix_support": PREFIX_BRANCHES,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_draws": spec.bootstrap_draws,
            "randomization_draws": spec.randomization_draws,
            "outcome_holm": True,
            "formulation_max_t": True,
            "outcome_equivalence_margin": OUTCOME_EQUIVALENCE_MARGIN,
            "minimum_finite_fraction": MINIMUM_FINITE_FRACTION,
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


def _seed(spec: PX7Spec, domain: str, *keys: object) -> int:
    selected = "smoke" if spec.label == "smoke" else domain
    return derive_seed(SEED_DOMAINS[selected], LABEL, spec.label, domain, *keys)


def _halves(spec: PX7Spec) -> dict[str, tuple[int, ...]]:
    if spec.branches == BRANCHES:
        return HALVES
    midpoint = spec.branches // 2
    return {"A": tuple(range(midpoint)), "B": tuple(range(midpoint, spec.branches))}


def _run_natural_candidate(
    matrix_id: int,
    beta: NDArray[np.float64],
    initial: NDArray[np.int16],
    candidate: str,
    spec: PX7Spec,
) -> list[GaugeCase]:
    config = GardConfig()
    maximum = max(spec.landmarks)
    for attempt in range(100):
        rng = np.random.default_rng(
            _seed(spec, "main_path", candidate, matrix_id, attempt)
        )
        snapshot = Snapshot(initial.copy(), 0, (), ())
        observations: list[NDArray[np.int64]] = [initial.astype(np.int64).copy()]
        output: list[GaugeCase] = []
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
                    counts = np.asarray(observations[-PAST_WINDOW:], dtype=np.int16)
                    output.append(
                        GaugeCase(
                            state_id=f"PX7-c{candidate}-m{matrix_id:03d}-g{generation:03d}",
                            candidate=candidate,
                            matrix_id=matrix_id,
                            landmark=generation,
                            beta=beta,
                            snapshot=snapshot,
                            history_counts=counts,
                        )
                    )
            if len(output) != len(spec.landmarks):
                raise AssertionError("natural path omitted a landmark")
            return output
        except SimulationError:
            continue
    raise SimulationError(
        f"PX7 failed bounded natural retry for c{candidate} m{matrix_id}"
    )


def _acquire_break(
    source: GaugeCase, spec: PX7Spec
) -> tuple[GaugeCase | None, dict[str, Any]]:
    config = GardConfig()
    rng = np.random.default_rng(
        _seed(spec, "acquisition", source.candidate, source.matrix_id, source.landmark)
    )
    snapshot = source.snapshot
    observations = [np.asarray(row, dtype=np.int64).copy() for row in source.history_counts]
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
            np.asarray(item, dtype=np.int64).copy() for item in traced.growth_observations
        )
        observations.append(np.asarray(traced.record.daughter, dtype=np.int64).copy())
        snapshot = _snapshot_after_record(snapshot, traced.record)
        if traced.record.h <= config.inheritance_threshold:
            case = GaugeCase(
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


def _explicit_transform(
    past_counts: NDArray, future_counts: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    past = np.asarray(past_counts, dtype=np.float64)
    future = np.asarray(future_counts, dtype=np.float64)
    if past.shape != future.shape or past.ndim != 2 or past.shape[0] < 3:
        raise ValueError("explicit count pairs must be matching samples by types")
    pairs = past.shape[0]
    data = close_clr_drop_last(np.vstack((past, future)))
    scales = data.std(axis=1)
    active = np.flatnonzero(np.isfinite(scales) & (scales > ACTIVE_STD_EPS)).astype(
        np.int64
    )
    if active.size < 2:
        raise ValueError("fewer than two active explicit-pair dimensions")
    selected = data[active]
    selected = (selected - selected.mean(axis=1, keepdims=True)) / selected.std(
        axis=1, keepdims=True
    )
    return selected[:, :pairs], selected[:, pairs:], active


def _explicit_graph(past: NDArray, future: NDArray) -> NDArray[np.float64]:
    left = np.asarray(past, dtype=np.float64)
    right = np.asarray(future, dtype=np.float64)
    dimensions = left.shape[0]
    joined = np.corrcoef(np.vstack((left, right)))
    first = joined[:dimensions, dimensions:]
    # Match lagged_gaussian_mi_graph exactly: C(past,future) is averaged
    # with C(future,past), not with a second copy of itself.
    second = joined[dimensions:, :dimensions]
    correlation = np.nan_to_num(
        0.5 * (first + second), nan=0.0, posinf=0.999999, neginf=-0.999999
    )
    correlation = np.clip(correlation, -0.999999, 0.999999)
    graph = -0.5 * np.log1p(-(correlation * correlation))
    graph = 0.5 * (graph + graph.T)
    np.fill_diagonal(graph, 0.0)
    return np.asarray(graph, dtype=np.float64)


def _map_partition(
    active: NDArray[np.int64], first_species: Sequence[int], second_species: Sequence[int]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    lookup = {int(species): index for index, species in enumerate(active)}
    first = np.asarray(
        [lookup[int(species)] for species in first_species if int(species) in lookup],
        dtype=np.int64,
    )
    second = np.asarray(
        [lookup[int(species)] for species in second_species if int(species) in lookup],
        dtype=np.int64,
    )
    if not first.size or not second.size:
        raise ValueError("fixed partition lost a complete active side")
    assigned = np.concatenate((first, second))
    if np.unique(assigned).size != active.size:
        raise ValueError("fixed partition does not cover each active dimension once")
    return first, second


def _past_partition(case: GaugeCase) -> tuple[tuple[int, ...], tuple[int, ...]]:
    data = close_clr_drop_last(np.asarray(case.history_counts, dtype=np.float64))
    scales = data.std(axis=1)
    active = np.flatnonzero(np.isfinite(scales) & (scales > ACTIVE_STD_EPS)).astype(
        np.int64
    )
    if active.size < 2:
        raise ValueError("past window has fewer than two active dimensions")
    selected = data[active]
    selected = (selected - selected.mean(axis=1, keepdims=True)) / selected.std(
        axis=1, keepdims=True
    )
    first, second = fiedler_bipartition(
        _explicit_graph(selected[:, :-1], selected[:, 1:])
    )
    return (
        tuple(int(active[index]) for index in first),
        tuple(int(active[index]) for index in second),
    )


def _score_pairs(
    past_counts: NDArray,
    future_counts: NDArray,
    partition: str,
    case: GaugeCase,
) -> dict[str, Any]:
    left, right, active = _explicit_transform(past_counts, future_counts)
    if partition == "self":
        first, second = fiedler_bipartition(_explicit_graph(left, right))
    elif partition == "past":
        first_species, second_species = _past_partition(case)
        first, second = _map_partition(active, first_species, second_species)
    elif partition == "beta":
        first_species, second_species = beta_physical_partition(case.beta)
        first, second = _map_partition(active, first_species, second_species)
    else:
        raise ValueError(partition)
    past_macro = np.vstack((left[first].mean(axis=0), left[second].mean(axis=0)))
    future_macro = np.vstack((right[first].mean(axis=0), right[second].mean(axis=0)))
    atom_series = _cached_local_phi_id_atoms(past_macro, future_macro)
    means = {atom: float(np.mean(values)) for atom, values in atom_series.items()}
    atoms = np.asarray(
        [means[(source, target)] for source in ANTICHAINS for target in ANTICHAINS],
        dtype=np.float64,
    )
    revised = float(sum(means[atom] for atom in PHIR_ATOMS))
    whole = gaussian_mutual_information(left, right)
    aa = gaussian_mutual_information(left[first], right[first])
    bb = gaussian_mutual_information(left[second], right[second])
    typeset = float(whole - aa - bb)
    ratio = float(typeset / whole) if np.isfinite(whole) and abs(whole) > 1e-12 else float("nan")
    synergy = float(means[(SYNERGISTIC, SYNERGISTIC)])
    causation = float(means[(SYNERGISTIC, UNIQUE_0)] + means[(SYNERGISTIC, UNIQUE_1)])
    return {
        "revised": revised,
        "typeset": typeset,
        "ratio": ratio,
        "whole_mi": float(whole),
        "causation": causation,
        "emergence": causation + synergy,
        "synergy_persistence": synergy,
        "atoms": atoms,
        "active_dimensions": int(active.size),
        "part_a_dimensions": int(first.size),
        "part_b_dimensions": int(second.size),
        "transitions": int(left.shape[1]),
        "partition_digest": _array_digest(active, first, second),
    }


def _nan_pair_score(transitions: int = 0) -> dict[str, Any]:
    return {
        "revised": float("nan"),
        "typeset": float("nan"),
        "ratio": float("nan"),
        "whole_mi": float("nan"),
        "causation": float("nan"),
        "emergence": float("nan"),
        "synergy_persistence": float("nan"),
        "atoms": np.full(len(ATOM_NAMES), np.nan),
        "active_dimensions": 0,
        "part_a_dimensions": 0,
        "part_b_dimensions": 0,
        "transitions": int(transitions),
        "partition_digest": "",
    }


def _safe_score_pairs(
    past: NDArray, future: NDArray, partition: str, case: GaugeCase
) -> dict[str, Any]:
    try:
        return _score_pairs(past, future, partition, case)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return _nan_pair_score(len(past))


def _records_digest(records: Iterable[FissionRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(np.ascontiguousarray(record.parent).tobytes())
        digest.update(np.ascontiguousarray(record.daughter).tobytes())
        digest.update(np.asarray([record.h], dtype=np.float64).tobytes())
        digest.update(np.asarray([record.growth_steps], dtype=np.int64).tobytes())
    return digest.hexdigest()


def _first_run(values: Sequence[bool], length: int) -> int:
    array = np.asarray(values, dtype=bool)
    for start in range(max(0, array.size - length + 1)):
        if bool(array[start : start + length].all()):
            return start + length
    return -1


def _future_seed(
    spec: PX7Spec, axis: str, case: GaugeCase, branch: int
) -> int:
    return _seed(
        spec,
        "future",
        axis,
        case.candidate,
        case.matrix_id,
        case.landmark,
        branch,
    )


def _selection_seed(spec: PX7Spec, axis: str, case: GaugeCase) -> int:
    return _seed(
        spec,
        "random_action",
        axis,
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def _select_axis_edits(
    axis: str,
    case: GaugeCase,
    students: Mapping[tuple[str, str], cr5.FrozenCR5Student],
    config: GardConfig,
    spec: PX7Spec,
) -> tuple[NDArray[np.float64], tuple[MolecularEdit | None, ...]]:
    target = "break" if axis == "resistance" else "renewal"
    student = students[(target, case.candidate)]
    state_case = case.as_state_case(f"PX7_{axis.upper()}")
    noop, scores = cr5.score_student_edits(student, state_case, config)
    return cr5.select_student_edits(
        noop,
        scores,
        np.random.default_rng(_selection_seed(spec, axis, case)),
    )


def _simulate_branch(
    axis: str,
    case: GaugeCase,
    edit: MolecularEdit | None,
    branch: int,
    spec: PX7Spec,
) -> tuple[dict[str, Any], PairBlock]:
    config = GardConfig()
    horizon = (
        spec.resistance_horizon if axis == "resistance" else spec.resilience_horizon
    )
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
    rng = np.random.default_rng(_future_seed(spec, axis, case, branch))
    molecular: list[NDArray[np.int64]] = [snapshot.composition.copy()]
    generational_past: list[NDArray[np.int64]] = []
    generational_future: list[NDArray[np.int64]] = []
    records: list[FissionRecord] = []
    for _step in range(horizon):
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
        generational_past.append(np.asarray(traced.record.parent, dtype=np.int64).copy())
        generational_future.append(
            np.asarray(traced.record.daughter, dtype=np.int64).copy()
        )
        records.append(traced.record)
        snapshot = _snapshot_after_record(snapshot, traced.record)
    completed = len(records) == horizon
    inherited = [record.h > config.inheritance_threshold for record in records]
    if axis == "resistance":
        primary = int(completed and bool(inherited) and all(inherited))
        first_event = next(
            (index + 1 for index, value in enumerate(inherited) if not value), -1
        )
        run5 = int(_first_run(inherited, 5) >= 0)
    else:
        first_event = _first_run(inherited, 3)
        primary = int(first_event >= 0)
        run5 = int(_first_run(inherited, 5) >= 0)
    molecular_array = np.asarray(molecular, dtype=np.int16)
    if molecular_array.shape[0] >= 2:
        molecular_past = molecular_array[:-1]
        molecular_future = molecular_array[1:]
    else:
        molecular_past = np.empty((0, config.n_types), dtype=np.int16)
        molecular_future = np.empty((0, config.n_types), dtype=np.int16)
    if generational_past:
        generation_past = np.asarray(generational_past, dtype=np.int16)
        generation_future = np.asarray(generational_future, dtype=np.int16)
    else:
        generation_past = np.empty((0, config.n_types), dtype=np.int16)
        generation_future = np.empty((0, config.n_types), dtype=np.int16)
    row = {
        "state_id": case.state_id,
        "axis": axis,
        "candidate": case.candidate,
        "matrix_id": case.matrix_id,
        "landmark": case.landmark,
        "branch": branch,
        "half": "A" if branch < spec.branches // 2 else "B",
        "primary": primary,
        "completed": int(completed),
        "survived": int(completed),
        "inherited_fraction": float(sum(inherited) / horizon),
        "first_event_time": int(first_event),
        "run5": run5,
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
        right = [block.molecular_future for block in blocks if len(block.molecular_future)]
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


def _score_arm_halves(
    axis: str,
    arm: str,
    case: GaugeCase,
    blocks: Sequence[PairBlock],
    spec: PX7Spec,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for half, indices in _halves(spec).items():
        half_blocks = [blocks[index] for index in indices]
        prefix_blocks = half_blocks[: min(PREFIX_BRANCHES, len(half_blocks))]
        for clock in CLOCKS:
            past, future = _concatenate_pairs(half_blocks, clock)
            prefix_past, prefix_future = _concatenate_pairs(prefix_blocks, clock)
            for partition in PARTITIONS:
                score = _safe_score_pairs(past, future, partition, case)
                prefix = _safe_score_pairs(
                    prefix_past, prefix_future, partition, case
                )
                for functional in FUNCTIONALS:
                    row: dict[str, Any] = {
                        "state_id": case.state_id,
                        "axis": axis,
                        "candidate": case.candidate,
                        "matrix_id": case.matrix_id,
                        "landmark": case.landmark,
                        "arm": arm,
                        "source_half": half,
                        "clock": clock,
                        "partition": partition,
                        "functional": functional,
                        "formulation": f"{clock}__{partition}__{functional}",
                        "value": float(score[functional]),
                        "prefix_value": float(prefix[functional]),
                        "whole_mi": float(score["whole_mi"]),
                        "causation": float(score["causation"]),
                        "emergence": float(score["emergence"]),
                        "synergy_persistence": float(score["synergy_persistence"]),
                        "active_dimensions": int(score["active_dimensions"]),
                        "part_a_dimensions": int(score["part_a_dimensions"]),
                        "part_b_dimensions": int(score["part_b_dimensions"]),
                        "transitions": int(score["transitions"]),
                        "prefix_transitions": int(prefix["transitions"]),
                        "partition_digest": score["partition_digest"],
                    }
                    row.update(
                        {
                            f"atom_{name}": float(value)
                            for name, value in zip(
                                ATOM_NAMES, score["atoms"], strict=True
                            )
                        }
                    )
                    output.append(row)
    return output


def _run_stage_case(
    axis: str,
    case: GaugeCase,
    students: Mapping[tuple[str, str], cr5.FrozenCR5Student],
    spec: PX7Spec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config = GardConfig()
    predictions, edits = _select_axis_edits(axis, case, students, config, spec)
    branch_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    for arm, prediction, edit in zip(ARMS[axis], predictions, edits, strict=True):
        edit_rows.append(
            {
                "state_id": case.state_id,
                "axis": axis,
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
            row, block = _simulate_branch(axis, case, edit, branch, spec)
            row.update({"arm": arm, "prediction": float(prediction)})
            branch_rows.append(row)
            blocks.append(block)
        score_rows.extend(_score_arm_halves(axis, arm, case, blocks, spec))
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


def _run_matrix(
    arguments: tuple[int, PX7Spec, str, str]
) -> MatrixBatch:
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
        resistance_cases: list[GaugeCase] = []
        for candidate in CANDIDATES:
            resistance_cases.extend(
                _run_natural_candidate(matrix_id, beta, initial, candidate, spec)
            )
        resilience_cases: list[GaugeCase] = []
        acquisition_rows: list[dict[str, Any]] = []
        for case in resistance_cases:
            broken, acquisition = _acquire_break(case, spec)
            acquisition_rows.append(acquisition)
            if broken is not None:
                resilience_cases.append(broken)
        branch_rows: list[dict[str, Any]] = []
        score_rows: list[dict[str, Any]] = []
        edit_rows: list[dict[str, Any]] = []
        for axis, cases in (
            ("resistance", resistance_cases),
            ("resilience", resilience_cases),
        ):
            for case in cases:
                branches, scores, edits = _run_stage_case(
                    axis, case, students, spec
                )
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
        return MatrixBatch(
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
    values = np.asarray(series.dropna(), dtype=np.float64)
    matrix_ids = np.asarray(series.dropna().index, dtype=np.int64)
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
        "matrices_positive": int(np.count_nonzero(values > 0.0)),
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
    selected = frame.copy()
    for column, expected in filters.items():
        selected = selected[selected[column] == expected]
    table = (
        selected.groupby(["matrix_id", "state_id", "arm"], sort=True)[value]
        .mean()
        .unstack("arm")
    )
    if high not in table or low not in table:
        return pd.Series(dtype=float)
    differences = (table[high] - table[low]).dropna()
    return differences.groupby(level="matrix_id").mean().sort_index()


def _matrix_centered_spearman(
    frame: pd.DataFrame, x: str, y: str
) -> pd.Series:
    output: dict[int, float] = {}
    for matrix_id, local in frame.groupby("matrix_id", sort=True):
        left = pd.to_numeric(local[x], errors="coerce").to_numpy(float)
        right = pd.to_numeric(local[y], errors="coerce").to_numpy(float)
        finite = np.isfinite(left) & np.isfinite(right)
        left, right = left[finite], right[finite]
        if (
            left.size < 4
            or np.unique(left).size < 2
            or np.unique(right).size < 2
        ):
            continue
        value = float(spearmanr(left, right).statistic)
        if np.isfinite(value):
            output[int(matrix_id)] = value
    return pd.Series(output, dtype=float).sort_index()


def _studentized(values: NDArray[np.float64]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan")
    scale = float(array.std(ddof=1) / math.sqrt(array.size))
    if not np.isfinite(scale) or scale <= 1e-15:
        return float("inf") if float(array.mean()) > 0 else float("-inf")
    return float(array.mean() / scale)


def _max_t_adjust(
    items: list[dict[str, Any]],
    vectors: list[pd.Series],
    repetitions: int,
    family: str,
    arrays: dict[str, NDArray],
) -> None:
    if len(items) != len(vectors):
        raise ValueError("max-T items and vectors are misaligned")
    if not items:
        return
    matrix_ids = np.arange(MATRICES, dtype=np.int64)
    rng = np.random.default_rng(
        _seed(scientific_spec(), "randomization", "max_t", family)
    )
    signs = rng.choice((-1.0, 1.0), size=(repetitions, MATRICES))
    randomized = np.full((repetitions, len(vectors)), -np.inf, dtype=np.float64)
    observed = np.full(len(vectors), np.nan, dtype=np.float64)
    for column, series in enumerate(vectors):
        local = series.dropna().sort_index()
        values = local.to_numpy(float)
        ids = local.index.to_numpy(int)
        observed[column] = _studentized(values)
        if values.size < 2:
            continue
        denominator = float(values.std(ddof=1) / math.sqrt(values.size))
        if not np.isfinite(denominator) or denominator <= 1e-15:
            denominator = 1e-15
        randomized[:, column] = (
            signs[:, ids] @ values / values.size / denominator
        )
    maximum = np.max(randomized, axis=1)
    arrays[f"max_t__{family}__maximum"] = maximum
    arrays[f"max_t__{family}__observed"] = observed
    arrays[f"max_t__{family}__matrix_ids"] = matrix_ids
    for item, statistic in zip(items, observed, strict=True):
        item["studentized_t"] = float(statistic)
        item["max_t_adjusted_p"] = (
            float((1 + np.count_nonzero(maximum >= statistic)) / (repetitions + 1))
            if np.isfinite(statistic)
            else float("nan")
        )


def _outcome_frame(branches: pd.DataFrame) -> pd.DataFrame:
    return (
        branches.groupby(
            [
                "axis",
                "candidate",
                "matrix_id",
                "state_id",
                "landmark",
                "arm",
                "half",
            ],
            sort=True,
            as_index=False,
        )
        .agg(
            q=("primary", "mean"),
            inherited_fraction=("inherited_fraction", "mean"),
            survival=("survived", "mean"),
            prediction=("prediction", "first"),
            trials=("primary", "size"),
            successes=("primary", "sum"),
        )
    )


def _outcome_validity(
    outcomes: pd.DataFrame,
    spec: PX7Spec,
    arrays: dict[str, NDArray],
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    gates: dict[str, bool] = {}
    for axis in AXES:
        local_family: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            for half in ("A", "B"):
                filters = {"axis": axis, "candidate": candidate, "half": half}
                effect = _matrix_arm_effect(
                    outcomes,
                    "q",
                    STABILIZING[axis],
                    DESTABILIZING[axis],
                    filters,
                )
                item = _bootstrap_summary(
                    effect,
                    spec.bootstrap_draws,
                    f"outcome/{axis}/c{candidate}/h{half}/targeted",
                    arrays,
                )
                random_effect = _matrix_arm_effect(
                    outcomes, "q", "RANDOM", "NOOP", filters
                )
                random_summary = _bootstrap_summary(
                    random_effect,
                    spec.bootstrap_draws,
                    f"outcome/{axis}/c{candidate}/h{half}/random",
                    arrays,
                )
                item.update(
                    {
                        "axis": axis,
                        "candidate": candidate,
                        "target_half": half,
                        "random_minus_noop": random_summary,
                        "random_tost": bool(
                            random_summary["ci90"][0] > -OUTCOME_EQUIVALENCE_MARGIN
                            and random_summary["ci90"][1]
                            < OUTCOME_EQUIVALENCE_MARGIN
                        ),
                    }
                )
                rows.append(item)
                local_family.append(item)
        adjusted = holm_adjust([float(item["one_sided_p"]) for item in local_family])
        for item, p_value in zip(local_family, adjusted, strict=True):
            item["holm_adjusted_p"] = float(p_value)
            item["pass"] = bool(
                item["effect"] > 0
                and item["ci95"][0] > 0
                and item["holm_adjusted_p"] < 0.05
                and item["random_tost"]
            )
        gates[axis] = bool(len(local_family) == 4 and all(item["pass"] for item in local_family))
    return rows, gates


def _merge_forecast(
    scores: pd.DataFrame,
    outcomes: pd.DataFrame,
    axis: str,
    candidate: str,
    formulation: str,
    source_half: str,
) -> pd.DataFrame:
    target_half = "B" if source_half == "A" else "A"
    phi = scores[
        (scores["axis"] == axis)
        & (scores["candidate"] == candidate)
        & (scores["formulation"] == formulation)
        & (scores["source_half"] == source_half)
    ][
        [
            "matrix_id",
            "state_id",
            "landmark",
            "arm",
            "value",
            "prefix_value",
        ]
    ]
    target = outcomes[
        (outcomes["axis"] == axis)
        & (outcomes["candidate"] == candidate)
        & (outcomes["half"] == target_half)
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


def _crossfit_incremental_gain(frame: pd.DataFrame, axis: str) -> pd.Series:
    rows: list[dict[str, Any]] = []
    selected = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["value", "prediction", "successes", "trials"]
    )
    if selected.empty:
        return pd.Series(dtype=float)
    for fold in range(5):
        train = selected[selected["matrix_id"] % 5 != fold]
        test = selected[selected["matrix_id"] % 5 == fold]
        if train.empty or test.empty:
            continue
        base_train = train["prediction"].to_numpy(float)
        base_test = test["prediction"].to_numpy(float)
        if axis == "resistance":
            base_train = 1.0 - base_train
            base_test = 1.0 - base_test
        base_train = np.clip(base_train, 1e-6, 1 - 1e-6)
        base_test = np.clip(base_test, 1e-6, 1 - 1e-6)
        phi_mean = float(train["value"].mean())
        phi_scale = float(train["value"].std(ddof=0))
        if not np.isfinite(phi_scale) or phi_scale <= 1e-12:
            continue
        x_base = logit(base_train)[:, None]
        x_full = np.column_stack(
            (logit(base_train), (train["value"].to_numpy(float) - phi_mean) / phi_scale)
        )
        successes = train["successes"].to_numpy(float)
        trials = train["trials"].to_numpy(float)
        x_base_fit = np.repeat(x_base, 2, axis=0)
        x_full_fit = np.repeat(x_full, 2, axis=0)
        labels = np.tile(np.asarray([1, 0], dtype=int), len(train))
        weights = np.column_stack((successes, trials - successes)).reshape(-1)
        if weights[labels == 1].sum() <= 0 or weights[labels == 0].sum() <= 0:
            continue
        base_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        full_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        try:
            base_model.fit(x_base_fit, labels, sample_weight=weights)
            full_model.fit(x_full_fit, labels, sample_weight=weights)
        except (ValueError, FloatingPointError):
            continue
        x_base_test = logit(base_test)[:, None]
        x_full_test = np.column_stack(
            (logit(base_test), (test["value"].to_numpy(float) - phi_mean) / phi_scale)
        )
        p_base = np.clip(base_model.predict_proba(x_base_test)[:, 1], 1e-9, 1 - 1e-9)
        p_full = np.clip(full_model.predict_proba(x_full_test)[:, 1], 1e-9, 1 - 1e-9)
        q = test["successes"].to_numpy(float) / test["trials"].to_numpy(float)
        base_loss = -(q * np.log(p_base) + (1 - q) * np.log(1 - p_base))
        full_loss = -(q * np.log(p_full) + (1 - q) * np.log(1 - p_full))
        for matrix_id, gain in zip(
            test["matrix_id"].to_numpy(int), base_loss - full_loss, strict=True
        ):
            rows.append({"matrix_id": int(matrix_id), "gain": float(gain)})
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).groupby("matrix_id")["gain"].mean().sort_index()


def analyze_batches(
    batches: Sequence[MatrixBatch], spec: PX7Spec
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    acquisitions = pd.DataFrame(
        [row for batch in batches for row in batch.acquisition_rows]
    )
    edits = pd.DataFrame([row for batch in batches for row in batch.edit_rows])
    branches = pd.DataFrame([row for batch in batches for row in batch.branch_rows])
    scores = pd.DataFrame([row for batch in batches for row in batch.score_rows])
    outcomes = _outcome_frame(branches)
    arrays: dict[str, NDArray] = {}
    outcome_rows, outcome_gates = _outcome_validity(outcomes, spec, arrays)

    eligibility: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        local = acquisitions[
            (acquisitions["candidate"] == candidate)
            & (acquisitions["eligible"] == 1)
        ]
        count = int(local["matrix_id"].nunique())
        eligibility[candidate] = {
            "eligible_states": int(len(local)),
            "eligible_matrices": count,
            "minimum": MINIMUM_ELIGIBLE_MATRICES,
            "pass": count >= MINIMUM_ELIGIBLE_MATRICES,
        }
    resilience_eligible = bool(all(item["pass"] for item in eligibility.values()))

    reliability_items: list[dict[str, Any]] = []
    reliability_vectors: list[pd.Series] = []
    forecast_items: list[dict[str, Any]] = []
    forecast_vectors: list[pd.Series] = []
    response_items: list[dict[str, Any]] = []
    response_vectors: list[pd.Series] = []
    incremental_items: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []

    for axis in AXES:
        for formulation in FORMULATIONS:
            functional = formulation.rsplit("__", 1)[-1]
            for candidate in CANDIDATES:
                local_scores = scores[
                    (scores["axis"] == axis)
                    & (scores["candidate"] == candidate)
                    & (scores["formulation"] == formulation)
                ]
                paired = local_scores.pivot_table(
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
                    f"reliability/{axis}/{formulation}/c{candidate}",
                    arrays,
                )
                item.update(
                    {
                        "family": "reliability",
                        "axis": axis,
                        "formulation": formulation,
                        "candidate": candidate,
                    }
                )
                reliability_items.append(item)
                reliability_vectors.append(reliability)
                for matrix_id, value in reliability.items():
                    matrix_rows.append(
                        {
                            "family": "reliability",
                            "axis": axis,
                            "formulation": formulation,
                            "candidate": candidate,
                            "direction": "A_B",
                            "matrix_id": int(matrix_id),
                            "value": float(value),
                        }
                    )

                for source_half in ("A", "B"):
                    direction = f"{source_half}_to_{'B' if source_half == 'A' else 'A'}"
                    merged = _merge_forecast(
                        scores,
                        outcomes,
                        axis,
                        candidate,
                        formulation,
                        source_half,
                    )
                    forecast = _matrix_centered_spearman(merged, "value", "q")
                    forecast_item = _bootstrap_summary(
                        forecast,
                        spec.bootstrap_draws,
                        f"forecast/{axis}/{formulation}/c{candidate}/{direction}",
                        arrays,
                    )
                    forecast_item.update(
                        {
                            "family": "forecast",
                            "axis": axis,
                            "formulation": formulation,
                            "candidate": candidate,
                            "direction": direction,
                        }
                    )
                    forecast_items.append(forecast_item)
                    forecast_vectors.append(forecast)

                    response = _matrix_arm_effect(
                        scores,
                        "value",
                        STABILIZING[axis],
                        DESTABILIZING[axis],
                        {
                            "axis": axis,
                            "candidate": candidate,
                            "formulation": formulation,
                            "source_half": source_half,
                        },
                    )
                    prefix_response = _matrix_arm_effect(
                        scores,
                        "prefix_value",
                        STABILIZING[axis],
                        DESTABILIZING[axis],
                        {
                            "axis": axis,
                            "candidate": candidate,
                            "formulation": formulation,
                            "source_half": source_half,
                        },
                    )
                    response_item = _bootstrap_summary(
                        response,
                        spec.bootstrap_draws,
                        f"response/{axis}/{formulation}/c{candidate}/h{source_half}",
                        arrays,
                    )
                    response_item.update(
                        {
                            "family": "response",
                            "axis": axis,
                            "formulation": formulation,
                            "candidate": candidate,
                            "source_half": source_half,
                            "prefix_effect": float(prefix_response.mean())
                            if len(prefix_response)
                            else float("nan"),
                            "support_sign_stable": bool(
                                functional == "revised"
                                or (
                                    len(prefix_response)
                                    and float(prefix_response.mean()) > 0
                                    and float(response.mean()) > 0
                                )
                            ),
                        }
                    )
                    response_items.append(response_item)
                    response_vectors.append(response)

                    gain = _crossfit_incremental_gain(merged, axis)
                    gain_item = _bootstrap_summary(
                        gain,
                        spec.bootstrap_draws,
                        f"incremental/{axis}/{formulation}/c{candidate}/{direction}",
                        arrays,
                    )
                    gain_item.update(
                        {
                            "axis": axis,
                            "formulation": formulation,
                            "candidate": candidate,
                            "direction": direction,
                        }
                    )
                    incremental_items.append(gain_item)

                    for family, vector in (("forecast", forecast), ("response", response)):
                        for matrix_id, value in vector.items():
                            matrix_rows.append(
                                {
                                    "family": family,
                                    "axis": axis,
                                    "formulation": formulation,
                                    "candidate": candidate,
                                    "direction": direction
                                    if family == "forecast"
                                    else source_half,
                                    "matrix_id": int(matrix_id),
                                    "value": float(value),
                                }
                            )

    _max_t_adjust(
        reliability_items,
        reliability_vectors,
        spec.randomization_draws,
        "reliability",
        arrays,
    )
    _max_t_adjust(
        forecast_items,
        forecast_vectors,
        spec.randomization_draws,
        "forecast",
        arrays,
    )
    _max_t_adjust(
        response_items,
        response_vectors,
        spec.randomization_draws,
        "response",
        arrays,
    )

    finite_fraction = (
        scores.groupby(["axis", "formulation"], sort=True)["value"]
        .apply(lambda values: float(np.isfinite(values.to_numpy(float)).mean()))
        .to_dict()
    )

    formulation_axes: list[dict[str, Any]] = []
    for axis in AXES:
        for formulation in FORMULATIONS:
            reliability = [
                item
                for item in reliability_items
                if item["axis"] == axis and item["formulation"] == formulation
            ]
            forecast = [
                item
                for item in forecast_items
                if item["axis"] == axis and item["formulation"] == formulation
            ]
            response = [
                item
                for item in response_items
                if item["axis"] == axis and item["formulation"] == formulation
            ]
            finite = float(finite_fraction.get((axis, formulation), 0.0))

            def positive(item: dict[str, Any]) -> bool:
                return bool(
                    item["effect"] > 0
                    and item["ci95"][0] > 0
                    and item.get("max_t_adjusted_p", 1.0) < 0.05
                )

            support = bool(
                finite >= MINIMUM_FINITE_FRACTION
                and all(item["support_sign_stable"] for item in response)
            )
            eligible_axis = axis != "resilience" or resilience_eligible
            passed = bool(
                outcome_gates[axis]
                and eligible_axis
                and len(reliability) == 2
                and len(forecast) == 4
                and len(response) == 4
                and all(positive(item) for item in reliability)
                and all(positive(item) for item in forecast)
                and all(positive(item) for item in response)
                and support
            )
            formulation_axes.append(
                {
                    "axis": axis,
                    "formulation": formulation,
                    "finite_fraction": finite,
                    "support_pass": support,
                    "manipulation_pass": outcome_gates[axis],
                    "eligible_axis": eligible_axis,
                    "pass": passed,
                }
            )

    passed = [item for item in formulation_axes if item["pass"]]
    passed_by_formulation: dict[str, set[str]] = {}
    for item in passed:
        passed_by_formulation.setdefault(item["formulation"], set()).add(item["axis"])
    two_axis = sorted(
        name for name, axes in passed_by_formulation.items() if set(AXES).issubset(axes)
    )
    exact = "molecular__self__revised"
    gates = {
        "resistance_manipulation_valid": outcome_gates["resistance"],
        "resilience_eligibility": resilience_eligible,
        "resilience_manipulation_valid": outcome_gates["resilience"],
        "axis_specific_phi_r_gauge_supported": bool(passed),
        "published_revised_phi_r_axis_supported": any(
            item["formulation"] == exact for item in passed
        ),
        "specified_phi_r_extension_supported": any(
            item["formulation"] != exact for item in passed
        ),
        "two_axis_plastic_heredity_gauge_supported": bool(two_axis),
    }
    metrics = {
        "format": "codex-ch5-phir-px7-primary-metrics-v1",
        "outcome_validity": outcome_rows,
        "eligibility": eligibility,
        "reliability": reliability_items,
        "forecast": forecast_items,
        "response": response_items,
        "incremental_log_loss": incremental_items,
        "formulation_axes": formulation_axes,
        "passed_formulation_axes": passed,
        "two_axis_formulations": two_axis,
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
    rng = np.random.default_rng(_seed(smoke_spec(), "validation", "fixture"))
    counts = rng.poisson(2.0, size=(260, GardConfig().n_types)).astype(np.int16)
    counts[counts.sum(axis=1) == 0, 0] = 1
    beta = np.exp(rng.normal(-4.0, 1.0, size=(100, 100)))
    snapshot = Snapshot(counts[-1].astype(np.int64), 20, (True,) * 20, (0.95,) * 20)
    case = GaugeCase("fixture", "02", 0, 20, beta, snapshot, counts[-PAST_WINDOW:])
    past, future = counts[:-1], counts[1:]
    scores = {
        partition: _score_pairs(past, future, partition, case)
        for partition in PARTITIONS
    }
    checks = {
        "formulation_count_18": len(FORMULATIONS) == 18
        and len(set(FORMULATIONS)) == 18,
        "axis_arms_frozen": ARMS["resistance"]
        == ("BREAK_UP", "BREAK_DOWN", "RANDOM", "NOOP")
        and ARMS["resilience"]
        == ("RENEWAL_UP", "RENEWAL_DOWN", "RANDOM", "NOOP"),
        "halves_disjoint_complete": set(HALVES["A"]).isdisjoint(HALVES["B"])
        and set(HALVES["A"]) | set(HALVES["B"]) == set(range(BRANCHES)),
        "nine_atom_identity": all(
            abs(
                item["revised"]
                - sum(
                    float(item["atoms"][ATOM_NAMES.index(
                        next(
                            name
                            for name in ATOM_NAMES
                            if name
                            == f"{source[0]}->{target[0]}"
                        )
                    )])
                    for source, target in ()
                )
            )
            < 1e-12
            for item in []
        ),
        "scores_finite": all(
            np.isfinite(item["revised"])
            and np.isfinite(item["typeset"])
            and np.isfinite(item["ratio"])
            for item in scores.values()
        ),
        "partitions_nonempty": all(
            item["part_a_dimensions"] > 0 and item["part_b_dimensions"] > 0
            for item in scores.values()
        ),
        "future_seed_arm_free": "arm" not in inspect.signature(_future_seed).parameters,
        "future_and_selection_distinct": _future_seed(
            smoke_spec(), "resistance", case, 0
        )
        != _selection_seed(smoke_spec(), "resistance", case),
        "endpoint_strict_threshold": not bool(0.9 > GardConfig().inheritance_threshold)
        and bool(np.nextafter(0.9, 1.0) > GardConfig().inheritance_threshold),
        "model_sources_exist": MODEL_SOURCE.is_file()
        and MODEL_CONTRACT_SOURCE.is_file(),
        "disk_available": shutil.disk_usage(ROOT).free >= MINIMUM_FREE_DISK_BYTES,
    }
    # Direct atom identity without relying on string rendering.
    for partition, item in scores.items():
        atom_lookup = {
            atom: float(value)
            for atom, value in zip(
                (
                    (source, target)
                    for source in ANTICHAINS
                    for target in ANTICHAINS
                ),
                item["atoms"],
                strict=True,
            )
        }
        checks[f"nine_atom_identity_{partition}"] = bool(
            abs(item["revised"] - sum(atom_lookup[atom] for atom in PHIR_ATOMS))
            < 1e-12
        )
    checks.pop("nine_atom_identity")
    return {"checks": checks, "fixture_scores": {key: _json_ready(value) for key, value in scores.items()}}


def validate(output: Path = DEFAULT_VALIDATION) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    fixture = _validation_fixture()
    students_a = cr5.load_students(MODEL_SOURCE, MODEL_CONTRACT_SOURCE)
    students_b = cr5.load_students(MODEL_SOURCE, MODEL_CONTRACT_SOURCE)
    fixture["checks"]["student_serialization_exact"] = bool(
        set(students_a) == set(students_b)
        and all(
            np.array_equal(students_a[key].coefficient, students_b[key].coefficient)
            and students_a[key].intercept == students_b[key].intercept
            for key in students_a
        )
    )
    fixture["checks"]["all_four_students_present"] = set(students_a) == {
        ("break", "02"),
        ("break", "03"),
        ("renewal", "02"),
        ("renewal", "03"),
    }
    payload = {
        "format": "codex-ch5-phir-px7-validation-v1",
        "checks": fixture["checks"],
        "all_checks_passed": bool(all(fixture["checks"].values())),
        "source_hashes": _source_hashes(),
        "model_sha256": sha256_file(MODEL_SOURCE),
        "model_contract_sha256": sha256_file(MODEL_CONTRACT_SOURCE),
        "fixture_score_digests": {
            key: _digest(value) for key, value in fixture["fixture_scores"].items()
        },
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
    validation_payload = json.loads(
        (validation_directory / "validation.json").read_text(encoding="utf-8")
    )
    if not validation_payload["all_checks_passed"]:
        raise ValueError("PX7 validation did not pass")
    if validation_payload["source_hashes"] != _source_hashes():
        raise ValueError("PX7 source changed after validation")
    if output.exists() or DEFAULT_OUTPUT.exists() or DEFAULT_WORK.exists():
        raise FileExistsError("PX7 registration, work, or output already exists")
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
        f"<!-- phir-extension-px7-registration-{payload['registration_id']} -->",
        [
            "## Phi-r extension PX7 registered",
            "",
            f"- Registration: `{payload['registration_id']}`.",
            "- Twenty-four fresh matrices, two hereditary axes, and an 18-member family-wise-corrected Phi-r tournament were sealed.",
            "- No PX7 scientific matrix or outcome existed at registration.",
        ],
    )
    return payload


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    if payload.get("format") != REGISTRATION_FORMAT:
        raise ValueError("unsupported PX7 registration")
    if payload["source_hashes"] != _source_hashes():
        raise ValueError("PX7 source changed after registration")
    if payload["protocol"] != protocol():
        raise ValueError("PX7 protocol changed after registration")
    expected_id = _digest(
        {key: value for key, value in payload.items() if key != "registration_id"}
    )
    if payload["registration_id"] != expected_id:
        raise ValueError("PX7 registration ID mismatch")
    if sha256_file(directory / "frozen_cr5_students.npz") != payload["model_sha256"]:
        raise ValueError("PX7 frozen student archive changed")
    if sha256_file(directory / "model_contract.json") != payload["model_contract_sha256"]:
        raise ValueError("PX7 model contract changed")
    return payload


def smoke(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_SMOKE,
) -> dict[str, Any]:
    registration_payload = verify_registration(registration_directory)
    if output.exists():
        raise FileExistsError(output)
    spec = smoke_spec()
    arguments = (
        0,
        spec,
        str(registration_directory / "frozen_cr5_students.npz"),
        str(registration_directory / "model_contract.json"),
    )
    first = _run_matrix(arguments)
    second = _run_matrix(arguments)
    smoke_future_fissions = (
        2 * len(ARMS["resistance"]) * spec.branches * spec.resistance_horizon
        + 2 * len(ARMS["resilience"]) * spec.branches * spec.resilience_horizon
    )
    scientific_future_fissions_per_matrix = (
        2
        * len(LANDMARKS)
        * len(ARMS["resistance"])
        * BRANCHES
        * RESISTANCE_HORIZON
        + 2
        * len(LANDMARKS)
        * len(ARMS["resilience"])
        * BRANCHES
        * RESILIENCE_HORIZON
    )
    projected_cpu_hours = float(
        first.cpu_seconds
        * scientific_future_fissions_per_matrix
        / smoke_future_fissions
        * MATRICES
        * 2
        / 3600.0
    )
    payload = {
        "format": "codex-ch5-phir-px7-smoke-v1",
        "registration_id": registration_payload["registration_id"],
        "exact_replay": first.scientific_digest == second.scientific_digest,
        "matrices": 1,
        "branches_created": len(first.branch_rows),
        "score_rows_created": len(first.score_rows),
        "all_formulations_exercised": set(
            row["formulation"] for row in first.score_rows
        )
        == set(FORMULATIONS),
        "both_axes_exercised": set(row["axis"] for row in first.branch_rows)
        == set(AXES),
        "cpu_seconds_per_smoke_matrix": first.cpu_seconds,
        "projected_complete_cpu_hours_upper_bound": projected_cpu_hours,
        "projected_within_cpu_ceiling": projected_cpu_hours <= MAX_CPU_HOURS,
        "scientific_effects_disclosed": False,
    }
    if not (
        payload["exact_replay"]
        and payload["all_formulations_exercised"]
        and payload["both_axes_exercised"]
        and payload["branches_created"] > 0
        and payload["score_rows_created"] > 0
        and payload["projected_within_cpu_ceiling"]
    ):
        raise AssertionError("PX7 smoke gate failed")
    output.mkdir(parents=True)
    _atomic_json(output / "smoke.json", payload)
    write_checksums(output)
    verify_checksums(output)
    return payload


def _checkpoint_contract(
    registration_id: str, spec: PX7Spec, stage: str
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
    work: Path, registration_id: str, spec: PX7Spec, cpu_budget_hours: float
) -> None:
    work.mkdir(parents=True, exist_ok=True)
    contract_path = work / "checkpoint_contract.json"
    expected = _checkpoint_contract(registration_id, spec, "generation")
    if contract_path.exists():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError("PX7 checkpoint contract mismatch")
    else:
        _atomic_json(contract_path, expected)
    budget_path = work / "cpu_budget.json"
    budget = {
        "maximum_cpu_hours": MAX_CPU_HOURS,
        "declared_cpu_hours": cpu_budget_hours,
    }
    if cpu_budget_hours > MAX_CPU_HOURS or cpu_budget_hours <= 0:
        raise ValueError("PX7 CPU budget must be in (0, 30]")
    if budget_path.exists() and json.loads(budget_path.read_text()) != budget:
        raise ValueError("PX7 CPU budget changed after launch")
    if not budget_path.exists():
        _atomic_json(budget_path, budget)


def _status_write(work: Path, payload: dict[str, Any]) -> None:
    payload = {"format": STATUS_FORMAT, "updated_at_unix": time.time(), **payload}
    _atomic_json(work / "status.json", payload)


def _load_checkpoint(path: Path) -> MatrixBatch:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, MatrixBatch):
        raise TypeError(f"unexpected PX7 checkpoint type: {path}")
    return value


def _run_checkpoint_stage(
    spec: PX7Spec,
    directory: Path,
    workers: int,
    model_path: Path,
    contract_path: Path,
    work: Path,
    stage: str,
    cpu_budget_seconds: float,
) -> list[MatrixBatch]:
    directory.mkdir(parents=True, exist_ok=True)
    batches: list[MatrixBatch | None] = [None] * spec.matrices
    for matrix_id in range(spec.matrices):
        path = directory / f"matrix_{matrix_id:03d}.pkl"
        if path.exists():
            batch = _load_checkpoint(path)
            if batch.matrix_id != matrix_id:
                raise ValueError("PX7 checkpoint matrix ID mismatch")
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
            generated: Iterable[MatrixBatch] = map(_run_matrix, arguments)
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
                    raise RuntimeError("PX7 declared CPU budget exhausted")
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
    if any(batch is None for batch in batches):
        raise AssertionError("PX7 checkpoint stage incomplete")
    return [batch for batch in batches if batch is not None]


def _effect_text(item: dict[str, Any]) -> str:
    return (
        f"{item['effect']:+.4f} "
        f"[{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}]"
    )


def _reports(metrics: dict[str, Any], registration_id: str) -> tuple[str, str]:
    outcome_lines = []
    for item in metrics["outcome_validity"]:
        outcome_lines.append(
            f"| {item['axis']} | {item['candidate']} | {item['target_half']} | "
            f"{_effect_text(item)} | {item.get('holm_adjusted_p', float('nan')):.4g} | "
            f"{item['random_tost']} | {item['pass']} |"
        )
    passed_lines = [
        f"- `{item['formulation']}` on **{item['axis']}**"
        for item in metrics["passed_formulation_axes"]
    ] or ["- None"]
    best_forecast = sorted(
        metrics["forecast"],
        key=lambda item: (
            np.nan_to_num(item.get("max_t_adjusted_p", np.nan), nan=2.0),
            -np.nan_to_num(item.get("effect", np.nan), nan=-2.0),
        ),
    )[:12]
    forecast_lines = [
        f"| {item['axis']} | `{item['formulation']}` | {item['candidate']} | "
        f"{item['direction']} | {_effect_text(item)} | "
        f"{item.get('max_t_adjusted_p', float('nan')):.4g} |"
        for item in best_forecast
    ]
    technical = "\n".join(
        (
            "# PX7 prospective plastic-heredity axis gauge tournament",
            "",
            f"Registration: `{registration_id}`.",
            "",
            "## Causal manipulation validity",
            "",
            "| Axis | Candidate | Target half | Stabilizing−destabilizing [95% CI] | Holm p | Random equivalent | Pass |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
            *outcome_lines,
            "",
            "## Passing formulation-axis pairs",
            "",
            *passed_lines,
            "",
            "## Strongest cross-half forecast cells",
            "",
            "| Axis | Formulation | Candidate | Direction | Effect [95% CI] | max-T p |",
            "| --- | --- | --- | --- | ---: | ---: |",
            *forecast_lines,
            "",
            "## Registered classification",
            "",
            *(f"- {key}: **{value}**" for key, value in metrics["gates"].items()),
            "",
            "## Boundaries",
            "",
            "PX7 is a formulation-family-corrected test of simulated resistance and resilience. A fixed-partition success is an extension, not a retrospective rescue of the exact public scorer. No result makes Phi-r causal or supports consciousness, life, agency, real chemistry, or metaphysical claims.",
            "",
        )
    )
    gates = metrics["gates"]
    if gates["two_axis_plastic_heredity_gauge_supported"]:
        message = "At least one frozen information thermometer worked for both staying stable and recovering after a break."
    elif gates["axis_specific_phi_r_gauge_supported"]:
        message = "At least one frozen information thermometer worked for one half of plastic heredity, but no single thermometer covered the whole phenomenon."
    else:
        message = "None of the eighteen preregistered information thermometers passed the complete reliability, forecasting, and causal-response test after correcting for trying several."
    lay = "\n".join(
        (
            "# Lay summary — PX7 axis gauge tournament",
            "",
            "We separated plastic heredity into two jobs: avoiding a loss of heredity, and rebuilding a short hereditary run after a loss had already happened. For each job we made matched molecular changes, simulated two independent groups of futures, and asked whether an information reading from one group correctly described what happened in the other.",
            "",
            message,
            "",
            "The molecular heredity experiments and the information thermometers are separate questions. Even a successful thermometer would describe this simulation; it would not be the cause of heredity and would not establish consciousness or life.",
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
    registration_payload = verify_registration(registration_directory)
    verify_checksums(DEFAULT_SMOKE)
    if output.exists():
        raise FileExistsError(output)
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"PX7 workers must be in [1,{MAX_WORKERS}]")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise OSError("PX7 free-disk gate failed")
    spec = scientific_spec()
    _prepare_work(work, registration_payload["registration_id"], spec, cpu_budget_hours)
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
            "format": "codex-ch5-phir-px7-replay-v1",
            "matrices": replay_rows,
            "complete_exact_replay": bool(
                len(replay_rows) == spec.matrices
                and all(item["exact"] for item in replay_rows)
            ),
        }
        if not replay_audit["complete_exact_replay"]:
            raise AssertionError("PX7 complete replay failed")
        _status_write(
            work,
            {
                "state": "analyzing",
                "stage": "analysis",
                "completed_matrices": spec.matrices,
                "total_matrices": spec.matrices,
                "cpu_seconds": generation_cpu
                + float(sum(batch.cpu_seconds for batch in replayed)),
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
            matrix_ids=np.asarray([batch.matrix_id for batch in generated], dtype=np.int16),
            betas=np.asarray([batch.beta for batch in generated], dtype=np.float64),
            initials=np.asarray([batch.initial for batch in generated], dtype=np.int16),
        )
        _atomic_json(staging / "primary_metrics.json", metrics)
        _atomic_json(staging / "replay_audit.json", replay_audit)
        scientific_digest = _digest(
            {
                "registration_id": registration_payload["registration_id"],
                "matrix_digests": [batch.scientific_digest for batch in generated],
                "metrics": metrics,
            }
        )
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration_payload["registration_id"],
            "matrices": spec.matrices,
            "candidates": list(CANDIDATES),
            "formulations": len(FORMULATIONS),
            "generation_cpu_seconds": generation_cpu,
            "replay_cpu_seconds": float(sum(batch.cpu_seconds for batch in replayed)),
            "workers": workers,
            "declared_cpu_budget_hours": cpu_budget_hours,
            "scientific_digest": scientific_digest,
            "gates": metrics["gates"],
        }
        _atomic_json(staging / "manifest.json", manifest)
        technical, lay = _reports(metrics, registration_payload["registration_id"])
        (staging / "SCIENTIFIC_REPORT.md").write_text(technical, encoding="utf-8")
        (staging / "LAY_SUMMARY.md").write_text(lay, encoding="utf-8")
        readback = {
            "format": "codex-ch5-phir-px7-readback-v1",
            "table_rows": {
                name: int(len(pd.read_csv(staging / f"{name}.csv.gz")))
                for name in tables
            },
            "inference_array_keys": sorted(np.load(staging / "inference_arrays.npz").files),
            "all_tables_nonempty": bool(all(len(frame) > 0 for frame in tables.values())),
            "manifest_scientific_digest": scientific_digest,
        }
        if not readback["all_tables_nonempty"]:
            raise AssertionError("PX7 artifact readback found an empty table")
        _atomic_json(staging / "readback_audit.json", readback)
        write_checksums(staging)
        if output.exists():
            raise FileExistsError(output)
        staging.replace(output)
        verify_result(output, registration_directory)
        _append_ledger(
            f"<!-- phir-extension-px7-result-{registration_payload['registration_id']} -->",
            [
                "## Phi-r extension PX7 completed",
                "",
                f"- Result: `{output.relative_to(ROOT)}`.",
                f"- Gates: `{json.dumps(metrics['gates'], sort_keys=True)}`.",
                "- The 18-formulation family was corrected jointly; complete exact replay and artifact readback passed.",
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
    registration_payload = verify_registration(registration_directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    replay = json.loads((directory / "replay_audit.json").read_text(encoding="utf-8"))
    readback = json.loads((directory / "readback_audit.json").read_text(encoding="utf-8"))
    if manifest.get("format") != RESULT_FORMAT:
        raise ValueError("unsupported PX7 result")
    if manifest["registration_id"] != registration_payload["registration_id"]:
        raise ValueError("PX7 result registration mismatch")
    if not replay["complete_exact_replay"]:
        raise ValueError("PX7 result lacks complete replay")
    if not readback["all_tables_nonempty"]:
        raise ValueError("PX7 result readback failed")
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
    workers: int = MAX_WORKERS, cpu_budget_hours: float = MAX_CPU_HOURS
) -> dict[str, Any]:
    registration_payload = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(DEFAULT_OUTPUT)
    launch_path = DEFAULT_WORK / "detached_launch.json"
    if launch_path.exists():
        existing = json.loads(launch_path.read_text(encoding="utf-8"))
        if _pid_alive(int(existing.get("pid", -1))):
            raise RuntimeError(f"PX7 already runs as PID {existing['pid']}")
    DEFAULT_WORK.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "plastic_heredity.phir_extension_px7",
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
        "format": "codex-ch5-phir-px7-detached-launch-v1",
        "registration_id": registration_payload["registration_id"],
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
        "format": "codex-ch5-phir-px7-status-report-v1",
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
        lines = DEFAULT_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        payload["log_tail"] = lines[-20:]
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
