"""Prospective CR9M launch-state moderation adjudication."""

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from .config import CANDIDATES, CohortConfig, ExperimentConfig, GardConfig
from .experiment import _json_ready, _runtime_manifest
from .intervention_core import (
    FrozenFullPredictor,
    MolecularEdit,
    _records_digest,
    score_legal_edits,
    simulate_controlled,
)
from .intervention_cr9_feedback import (
    _entropy,
    _holm_adjust,
    _interval,
    _maximum_leave_one_out_influence,
    _post_fission_snapshots,
    _snapshot_equal,
    _throughput,
    _top1,
    spearman_constant_zero,
)
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .seeds import derive_seed
from .simulator import (
    SimulationError,
    Snapshot,
    cosine_similarity,
    generate_beta,
    generate_initial_composition,
    simulate_lineage,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_intervention_replication"
CR9_REGISTRATION = RESULT_ROOT / "cr9_feedback_registration"
CR9_RESULT = RESULT_ROOT / "cr9_control_half_life"

DEFAULT_VALIDATION = RESULT_ROOT / "cr9m_launch_moderation_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "cr9m_launch_moderation_registration"
DEFAULT_SMOKE = RESULT_ROOT / "cr9m_launch_moderation_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "cr9m_launch_moderation"
DEFAULT_WORK = RESULT_ROOT / ".cr9m_launch_moderation_work"

DOCUMENT = "CODEX_INTERVENTION_CR9M_PREREGISTRATION.md"
PROGRAM_FORMAT = "codex-intervention-cr9m-launch-v1"
VALIDATION_FORMAT = "codex-intervention-cr9m-validation-v1"
REGISTRATION_FORMAT = "codex-intervention-cr9m-registration-v1"
RESULT_FORMAT = "codex-intervention-cr9m-result-v1"
CHECKPOINT_FORMAT = "codex-intervention-cr9m-checkpoint-v1"
STATUS_FORMAT = "codex-intervention-cr9m-status-v1"
LABEL = "INTCR9M_LAUNCH_V1"

EXPECTED_CR9_REGISTRATION_ID = (
    "a2f8340b632e9a75725c8bc42ec3069d59839be2119e6122268440a866d7fe00"
)
EXPECTED_MODEL_SHA256 = (
    "9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af"
)

MATRICES = 48
REPLICATES = 3
MATURE_LANDMARK = 60
RELEASE_HORIZON = 60
PULSE_LENGTHS = (1, 2, 4, 8, 16, 32, 60)
LAUNCHES = ("NASCENT", "MATURE")
CONVENTIONS = ("RELAXED", "POST_EDIT")
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
DEPARTURE_THRESHOLD = 0.7
INHERITANCE_THRESHOLD = 0.9
MINIMUM_FREE_DISK_BYTES = 2_500_000_000

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_cr9m_launch.py",
    "tests/test_intervention_cr9m_launch.py",
    "plastic_heredity/intervention_cr9_feedback.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/config.py",
    "plastic_heredity/experiment.py",
    "plastic_heredity/features.py",
    "plastic_heredity/simulator.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/mechanistic.py",
    "pyproject.toml",
    "requirements-lock.txt",
)


def _seed(name: str) -> str:
    return hashlib.sha256(
        f"codex-clean-room-cr9m-launch-v1::{name}".encode("utf-8")
    ).hexdigest()


SEEDS = {
    name: _seed(name)
    for name in (
        "validation",
        "smoke",
        "matrix_generation",
        "initial_composition",
        "mature_main_trajectory",
        "factorial_future",
        "bootstrap",
        "randomization",
        "replay",
    )
}


@dataclass(frozen=True)
class LaunchCase:
    state_id: str
    candidate: str
    matrix_id: int
    launch: str
    beta: NDArray[np.float64]
    snapshot: Snapshot


@dataclass(frozen=True)
class PulseLineage:
    launch: str
    convention: str
    pulse_length: int
    replicate: int
    pulse_completed: bool
    pulse_observed_fissions: int
    release_completed: bool
    release_observed_fissions: int
    edits_applied: int
    action_steps: tuple[int, ...]
    actions: tuple[MolecularEdit, ...]
    pulse_record_digest: str
    release_record_digest: str
    anchor_composition: NDArray[np.int64]
    final_snapshot: Snapshot
    persistence: int
    first_departure_time: int
    similarity_to_anchor: NDArray[np.float64]
    risk: NDArray[np.float64]
    boundary_h: NDArray[np.float64]
    growth_updates: NDArray[np.int32]
    entropy: NDArray[np.float64]
    top1_share: NDArray[np.float64]
    occupied_types: NDArray[np.int16]
    throughput: NDArray[np.float64]
    action_risk_before: NDArray[np.float64]
    action_risk_after: NDArray[np.float64]
    launch_risk: float
    launch_entropy: float
    launch_top1_share: float
    launch_occupied_types: int
    launch_throughput: float
    anchor_risk: float
    anchor_entropy: float
    anchor_top1_share: float
    anchor_occupied_types: int
    anchor_throughput: float
    release_interventions_applied: int
    simulation_rng_state: dict[str, Any]


@dataclass(frozen=True)
class PhaseBatch:
    format: str
    registration_id: str
    state_id: str
    candidate: str
    matrix_id: int
    launch: str
    case_digest: str
    lineages: tuple[PulseLineage, ...]


class _Trace:
    def __init__(self) -> None:
        self.risk_before: list[float] = []
        self.risk_after: list[float] = []
        self.actions: list[MolecularEdit] = []
        self.action_steps: list[int] = []


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_cr9m_scientific_matrix",
        "boundary": {
            "sealed_cr9_unchanged": True,
            "post_result_prospective_moderation_followup": True,
            "cannot_rescue_or_replace_cr9": True,
            "strict_eight_excluded": True,
        },
        "upstream": {
            "cr9_registration_id": EXPECTED_CR9_REGISTRATION_ID,
            "cr9_failed_hysteresis_gate_retained": True,
            "frozen_model_sha256": EXPECTED_MODEL_SHA256,
        },
        "cohort": {
            "fresh_matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "replicates": REPLICATES,
            "paired_launches": list(LAUNCHES),
            "mature_landmark": MATURE_LANDMARK,
            "same_beta_and_initial_composition_within_matrix": True,
            "lineage_retry_or_replacement": False,
        },
        "factorial": {
            "launches": list(LAUNCHES),
            "conventions": list(CONVENTIONS),
            "pulse_lengths": list(PULSE_LENGTHS),
            "relaxed": "P fissions; edits after 1..P-1; unedited final daughter anchor",
            "post_edit": "P fissions; edits after 1..P; post-edit anchor",
            "selector": "frozen exhaustive legal mass-preserving MODEL_DOWN in all cells",
            "release_fissions": RELEASE_HORIZON,
            "release_callback": None,
        },
        "endpoint": {
            "departure": "unrounded float64 cosine < 0.7",
            "right_censor_cap": RELEASE_HORIZON + 1,
            "incomplete_pulse_persistence": 1,
            "incomplete_release": "first unobserved registered release boundary",
            "rows_dropped": 0,
            "constant_persistence_spearman": 0.0,
        },
        "primary": {
            "matrix_estimand": "mean-over-replicates seven-point Spearman within each factorial cell",
            "contrast": "0.5*((NASCENT_RELAXED-MATURE_RELAXED)+(NASCENT_POST_EDIT-MATURE_POST_EDIT))",
            "gate": "positive mean, CI95 lower > 0, Holm sign p < 0.05 in both candidates",
            "holm_family": "two candidate launch-moderation cells",
        },
        "supporting": {
            "protocol_robust_nascent_hysteresis": "all four candidate-by-convention nascent cells pass positive CI and Holm sign test",
            "packing_diagnostics_cannot_rescue_primary": True,
            "survivor_only_is_nonprimary": True,
        },
        "randomness": {
            "seed_domains": SEEDS,
            "future_seed_excludes": ["launch", "convention", "pulse_length"],
            "common_random_streams_not_identical_futures": True,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "candidates_never_pooled": True,
        },
        "integrity": {
            "complete_exact_replay": True,
            "release_zero_interventions": True,
            "artifact_readback": True,
        },
        "stop_rule": "seal CR9M and stop without launching CR10 or another moderator search",
        "prohibited_claims": [
            "autonomous restoring basin or installed compotype",
            "biological memory, agency, life, or error correction",
            "real prebiotic chemistry or universal origin-of-life mechanism",
            "strict-eight or Phi/PhiID control",
        ],
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


def experiment() -> ExperimentConfig:
    cohort = CohortConfig(MATRICES, REPLICATES, (0, MATURE_LANDMARK))
    return ExperimentConfig(
        gard=GardConfig(),
        development=cohort,
        confirmation=cohort,
        horizon=RELEASE_HORIZON,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
        master_seed=SEEDS["matrix_generation"],
    )


def build_cr9m_cohort(current_experiment: ExperimentConfig) -> list[LaunchCase]:
    cases: list[LaunchCase] = []
    for matrix_id in range(MATRICES):
        beta = generate_beta(
            current_experiment.gard,
            np.random.default_rng(
                derive_seed(SEEDS["matrix_generation"], f"{LABEL}.beta", matrix_id)
            ),
        )
        initial = generate_initial_composition(
            current_experiment.gard,
            np.random.default_rng(
                derive_seed(SEEDS["initial_composition"], f"{LABEL}.initial", matrix_id)
            ),
        )
        nascent = Snapshot(
            composition=initial.copy(),
            generation=0,
            inheritance=(),
            boundary_h=(),
            previous_growth_steps=0,
            cumulative_growth_steps=0,
        )
        for candidate, contract in CANDIDATES.items():
            lineage: list[Snapshot] | None = None
            for attempt in range(100):
                rng = np.random.default_rng(
                    derive_seed(
                        SEEDS["mature_main_trajectory"],
                        f"{LABEL}.natural_main_path",
                        candidate,
                        matrix_id,
                        attempt,
                    )
                )
                try:
                    lineage = simulate_lineage(
                        initial, beta, current_experiment.gard, contract, rng
                    )
                    break
                except SimulationError:
                    continue
            if lineage is None:
                raise SimulationError(
                    f"failed CR9M mature trajectory for candidate {candidate}, matrix {matrix_id}"
                )
            mature = {item.generation: item for item in lineage}[MATURE_LANDMARK]
            for launch, snapshot in (("NASCENT", nascent), ("MATURE", mature)):
                cases.append(
                    LaunchCase(
                        state_id=(
                            f"{LABEL}-c{candidate}-m{matrix_id:03d}-{launch.lower()}"
                        ),
                        candidate=candidate,
                        matrix_id=matrix_id,
                        launch=launch,
                        beta=beta,
                        snapshot=snapshot,
                    )
                )
    return cases


def _case_digest(case: LaunchCase) -> str:
    digest = hashlib.sha256()
    for value in (case.state_id, case.candidate, str(case.matrix_id), case.launch):
        digest.update(value.encode())
    digest.update(np.ascontiguousarray(case.beta, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(case.snapshot.composition, dtype=np.int64).tobytes())
    digest.update(np.asarray(case.snapshot.boundary_h, dtype=np.float64).tobytes())
    digest.update(np.asarray(case.snapshot.inheritance, dtype=np.int8).tobytes())
    digest.update(
        np.asarray(
            (
                case.snapshot.generation,
                case.snapshot.previous_growth_steps,
                case.snapshot.cumulative_growth_steps,
            ),
            dtype=np.int64,
        ).tobytes()
    )
    return digest.hexdigest()


def _model_down(
    predictor: FrozenFullPredictor,
    candidate: str,
    snapshot: Snapshot,
    beta: NDArray,
    config: GardConfig,
) -> tuple[float, MolecularEdit, float]:
    noop, scores = score_legal_edits(predictor, candidate, snapshot, beta, config)
    probabilities = np.asarray(
        [item.predicted_probability for item in scores], dtype=np.float64
    )
    minimum = probabilities.min()
    index = int(np.flatnonzero(probabilities == minimum)[0])
    selected = scores[index]
    return float(noop), selected.edit, float(selected.predicted_probability)


def _future_seed(case: LaunchCase, replicate: int) -> int:
    return derive_seed(
        SEEDS["factorial_future"],
        f"{LABEL}.factorial.future",
        case.candidate,
        case.matrix_id,
        replicate,
    )


def _pulse_controller(
    convention: str,
    pulse_length: int,
    predictor: FrozenFullPredictor,
    config: GardConfig,
) -> tuple[Callable[[Snapshot, NDArray, str, int], MolecularEdit | None], _Trace]:
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown CR9M convention: {convention}")
    if pulse_length < 1:
        raise ValueError("pulse length must be positive")
    trace = _Trace()

    def callback(
        snapshot: Snapshot, beta: NDArray, candidate: str, step: int
    ) -> MolecularEdit | None:
        boundary = step + 1
        if convention == "RELAXED" and boundary == pulse_length:
            return None
        before, edit, after = _model_down(
            predictor, candidate, snapshot, beta, config
        )
        trace.risk_before.append(before)
        trace.risk_after.append(after)
        trace.actions.append(edit)
        trace.action_steps.append(boundary)
        return edit

    return callback, trace


def _empty_float(length: int) -> NDArray[np.float64]:
    return np.full(length, np.nan, dtype=np.float64)


def _state_metrics(
    snapshot: Snapshot,
    beta: NDArray,
    predictor: FrozenFullPredictor,
    candidate: str,
    config: GardConfig,
) -> tuple[float, float, float, int, float]:
    return (
        predictor.predict_snapshot(candidate, snapshot, beta, config),
        _entropy(snapshot.composition),
        _top1(snapshot.composition),
        int(np.count_nonzero(snapshot.composition)),
        _throughput(snapshot.composition, beta),
    )


def _pulse_lineage(
    case: LaunchCase,
    current_experiment: ExperimentConfig,
    predictor: FrozenFullPredictor,
    convention: str,
    pulse_length: int,
    replicate: int,
    release_horizon: int,
) -> PulseLineage:
    rng = np.random.default_rng(_future_seed(case, replicate))
    callback, trace = _pulse_controller(
        convention, pulse_length, predictor, current_experiment.gard
    )
    pulse = simulate_controlled(
        case.snapshot,
        case.beta,
        case.candidate,
        current_experiment,
        pulse_length,
        rng,
        callback,
    )
    expected_edits = pulse_length - 1 if convention == "RELAXED" else pulse_length
    if pulse.completed_horizon and pulse.interventions_applied != expected_edits:
        raise AssertionError("CR9M completed pulse has the wrong edit count")
    if tuple(trace.actions) != tuple(pulse.selected_edits):
        raise AssertionError("CR9M controller trace differs from applied actions")
    if tuple(trace.action_steps) != tuple(
        range(1, pulse.interventions_applied + 1)
    ):
        raise AssertionError("CR9M pulse action steps are not the registered prefix")
    anchor = pulse.final_snapshot
    if pulse.completed_horizon:
        release = simulate_controlled(
            anchor,
            case.beta,
            case.candidate,
            current_experiment,
            release_horizon,
            rng,
            None,
        )
    else:
        from .intervention_core import ControlledResult

        release = ControlledResult((), False, anchor, 0, ())

    snapshots = _post_fission_snapshots(anchor, release.records)
    if snapshots and not _snapshot_equal(snapshots[-1], release.final_snapshot):
        raise AssertionError("CR9M release snapshot reconstruction differs")
    similarity = _empty_float(release_horizon)
    risk = _empty_float(release_horizon)
    boundary_h = _empty_float(release_horizon)
    growth = np.full(release_horizon, -1, dtype=np.int32)
    entropy = _empty_float(release_horizon)
    top1 = _empty_float(release_horizon)
    occupied = np.full(release_horizon, -1, dtype=np.int16)
    throughput = _empty_float(release_horizon)
    for index, (snapshot, record) in enumerate(zip(snapshots, release.records, strict=True)):
        similarity[index] = cosine_similarity(anchor.composition, snapshot.composition)
        risk[index] = predictor.predict_snapshot(
            case.candidate, snapshot, case.beta, current_experiment.gard
        )
        boundary_h[index] = float(record.h)
        growth[index] = int(record.growth_steps)
        entropy[index] = _entropy(snapshot.composition)
        top1[index] = _top1(snapshot.composition)
        occupied[index] = int(np.count_nonzero(snapshot.composition))
        throughput[index] = _throughput(snapshot.composition, case.beta)
    crossing = np.flatnonzero(similarity < DEPARTURE_THRESHOLD)
    if not pulse.completed_horizon:
        persistence = 1
    elif crossing.size:
        persistence = int(crossing[0]) + 1
    elif release.completed_horizon:
        persistence = release_horizon + 1
    else:
        persistence = min(len(release.records) + 1, release_horizon)
    if release.interventions_applied != 0 or release.selected_edits:
        raise AssertionError("CR9M release applied an intervention")
    launch_metrics = _state_metrics(
        case.snapshot,
        case.beta,
        predictor,
        case.candidate,
        current_experiment.gard,
    )
    anchor_metrics = _state_metrics(
        anchor,
        case.beta,
        predictor,
        case.candidate,
        current_experiment.gard,
    )
    return PulseLineage(
        launch=case.launch,
        convention=convention,
        pulse_length=pulse_length,
        replicate=replicate,
        pulse_completed=bool(pulse.completed_horizon),
        pulse_observed_fissions=len(pulse.records),
        release_completed=bool(release.completed_horizon),
        release_observed_fissions=len(release.records),
        edits_applied=pulse.interventions_applied,
        action_steps=tuple(trace.action_steps),
        actions=tuple(pulse.selected_edits),
        pulse_record_digest=_records_digest(pulse.records),
        release_record_digest=_records_digest(release.records),
        anchor_composition=anchor.composition.copy(),
        final_snapshot=release.final_snapshot,
        persistence=persistence,
        first_departure_time=(int(crossing[0]) + 1 if crossing.size else -1),
        similarity_to_anchor=similarity,
        risk=risk,
        boundary_h=boundary_h,
        growth_updates=growth,
        entropy=entropy,
        top1_share=top1,
        occupied_types=occupied,
        throughput=throughput,
        action_risk_before=np.asarray(trace.risk_before, dtype=np.float64),
        action_risk_after=np.asarray(trace.risk_after, dtype=np.float64),
        launch_risk=launch_metrics[0],
        launch_entropy=launch_metrics[1],
        launch_top1_share=launch_metrics[2],
        launch_occupied_types=launch_metrics[3],
        launch_throughput=launch_metrics[4],
        anchor_risk=anchor_metrics[0],
        anchor_entropy=anchor_metrics[1],
        anchor_top1_share=anchor_metrics[2],
        anchor_occupied_types=anchor_metrics[3],
        anchor_throughput=anchor_metrics[4],
        release_interventions_applied=release.interventions_applied,
        simulation_rng_state=_json_ready(rng.bit_generator.state),
    )


def _run_case(
    case: LaunchCase,
    current_experiment: ExperimentConfig,
    model_path: str | Path,
    registration_id: str,
    *,
    conventions: tuple[str, ...] = CONVENTIONS,
    pulse_lengths: tuple[int, ...] = PULSE_LENGTHS,
    replicates: int = REPLICATES,
    release_horizon: int = RELEASE_HORIZON,
) -> PhaseBatch:
    predictor = FrozenFullPredictor.load(model_path)
    lineages = tuple(
        _pulse_lineage(
            case,
            current_experiment,
            predictor,
            convention,
            pulse_length,
            replicate,
            release_horizon,
        )
        for replicate in range(replicates)
        for convention in conventions
        for pulse_length in pulse_lengths
    )
    return PhaseBatch(
        format=CHECKPOINT_FORMAT,
        registration_id=registration_id,
        state_id=case.state_id,
        candidate=case.candidate,
        matrix_id=case.matrix_id,
        launch=case.launch,
        case_digest=_case_digest(case),
        lineages=lineages,
    )


def batch_digest(batch: PhaseBatch) -> str:
    return hashlib.sha256(pickle.dumps(batch, protocol=5)).hexdigest()


def replay_audit(
    generated: list[PhaseBatch], replayed: list[PhaseBatch]
) -> dict[str, Any]:
    if len(generated) != len(replayed):
        raise ValueError("CR9M replay batch count differs")
    rows: list[dict[str, Any]] = []
    for left, right in zip(generated, replayed, strict=True):
        left_digest = batch_digest(left)
        right_digest = batch_digest(right)
        rows.append(
            {
                "state_id": left.state_id,
                "candidate": left.candidate,
                "matrix_id": left.matrix_id,
                "launch": left.launch,
                "generated_digest": left_digest,
                "replay_digest": right_digest,
                "exact": left_digest == right_digest,
            }
        )
    return {
        "format": "codex-intervention-cr9m-replay-audit-v1",
        "state_batches": len(rows),
        "exact_state_action_endpoint_process_and_rng": bool(
            all(row["exact"] for row in rows)
        ),
        "rows": rows,
    }


def _checkpoint_path(directory: Path, case: LaunchCase) -> Path:
    return directory / (
        f"c{case.candidate}_m{case.matrix_id:03d}_{case.launch.lower()}.pkl"
    )


def _write_checkpoint(path: Path, batch: PhaseBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(batch, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_checkpoint(
    path: Path, case: LaunchCase, registration_id: str
) -> PhaseBatch | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(value, PhaseBatch):
        return None
    expected = (
        CHECKPOINT_FORMAT,
        registration_id,
        case.state_id,
        case.candidate,
        case.matrix_id,
        case.launch,
        _case_digest(case),
        REPLICATES * len(CONVENTIONS) * len(PULSE_LENGTHS),
    )
    observed = (
        value.format,
        value.registration_id,
        value.state_id,
        value.candidate,
        value.matrix_id,
        value.launch,
        value.case_digest,
        len(value.lineages),
    )
    return value if observed == expected else None


def _write_status(
    work: Path, stage: str, completed: int, total: int, **extra: Any
) -> None:
    work.mkdir(parents=True, exist_ok=True)
    now = time.time()
    path = work / "campaign_status.json"
    prior: dict[str, Any] = {}
    if path.is_file():
        try:
            prior = json.loads(path.read_text())
        except Exception:
            prior = {}
    if prior.get("stage") == stage:
        started = float(prior.get("stage_started_unix", now))
        started_count = int(prior.get("stage_started_completed_state_batches", completed))
    else:
        started = now
        started_count = completed
    elapsed = max(0.0, now - started)
    newly_completed = max(0, completed - started_count)
    rate = newly_completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 else None
    payload = {
        "format": STATUS_FORMAT,
        "stage": stage,
        "completed_state_batches": completed,
        "total_state_batches": total,
        "updated_at_unix": now,
        "stage_started_unix": started,
        "stage_started_completed_state_batches": started_count,
        "stage_elapsed_seconds": elapsed,
        "state_batches_per_second": rate,
        "estimated_stage_seconds_remaining": eta,
        "free_disk_bytes": shutil.disk_usage(ROOT).free,
        **extra,
    }
    temporary = work / f".status-{os.getpid()}.tmp"
    temporary.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _phase_worker(arguments: tuple[Any, ...]) -> PhaseBatch:
    limiter = threadpool_limits(limits=1)
    try:
        return _run_case(*arguments)
    finally:
        limiter.restore_original_limits()


def run_phase_batches(
    cases: list[LaunchCase],
    current_experiment: ExperimentConfig,
    model_path: Path,
    registration_id: str,
    directory: Path,
    workers: int,
    work: Path,
    stage: str,
) -> list[PhaseBatch]:
    directory.mkdir(parents=True, exist_ok=True)
    batches: dict[str, PhaseBatch] = {}
    missing: list[LaunchCase] = []
    for case in cases:
        checkpoint = _read_checkpoint(
            _checkpoint_path(directory, case), case, registration_id
        )
        if checkpoint is None:
            missing.append(case)
        else:
            batches[case.state_id] = checkpoint
    reused = len(batches)
    _write_status(work, stage, reused, len(cases), reused=reused)
    arguments = [
        (case, current_experiment, model_path, registration_id) for case in missing
    ]
    if workers == 1:
        for case, argument in zip(missing, arguments, strict=True):
            batch = _phase_worker(argument)
            _write_checkpoint(_checkpoint_path(directory, case), batch)
            batches[case.state_id] = batch
            _write_status(work, stage, len(batches), len(cases), reused=reused)
            print(f"[{stage}] {len(batches)}/{len(cases)} state batches", flush=True)
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_phase_worker, argument): case
                for argument, case in zip(arguments, missing, strict=True)
            }
            for future in as_completed(futures):
                case = futures[future]
                batch = future.result()
                _write_checkpoint(_checkpoint_path(directory, case), batch)
                batches[case.state_id] = batch
                _write_status(work, stage, len(batches), len(cases), reused=reused)
                print(f"[{stage}] {len(batches)}/{len(cases)} state batches", flush=True)
    ordered = [batches[case.state_id] for case in cases]
    if len(ordered) != len(cases):
        raise AssertionError("CR9M checkpoint cohort is incomplete")
    return ordered


def inference_draws() -> dict[str, NDArray]:
    bootstrap_rng = np.random.default_rng(
        derive_seed(SEEDS["bootstrap"], f"{LABEL}.whole_matrix_bootstrap")
    )
    randomization_rng = np.random.default_rng(
        derive_seed(SEEDS["randomization"], f"{LABEL}.whole_matrix_signs")
    )
    bootstrap = bootstrap_rng.integers(
        0, MATRICES, size=(BOOTSTRAP_REPETITIONS, MATRICES), dtype=np.int16
    )
    signs = randomization_rng.integers(
        0, 2, size=(RANDOMIZATION_REPETITIONS, MATRICES), dtype=np.int8
    ).astype(np.float64)
    return {
        "bootstrap_indices": bootstrap,
        "randomization_signs": 2.0 * signs - 1.0,
    }


def _contrast_summary(
    values: NDArray, bootstrap: NDArray, signs: NDArray
) -> tuple[dict[str, Any], NDArray, NDArray]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (MATRICES,) or np.any(~np.isfinite(array)):
        raise ValueError("CR9M contrast must contain one finite value per matrix")
    boot = array[np.asarray(bootstrap, dtype=np.int64)].mean(axis=1)
    observed = float(array.mean())
    null = np.asarray(signs @ array / len(array), dtype=np.float64)
    raw_p = float((np.count_nonzero(null >= observed) + 1) / (len(null) + 1))
    summary = {
        "estimate": observed,
        "bootstrap_ci95": _interval(boot),
        "bootstrap_ci90": _interval(boot, 0.10),
        "raw_one_sided_randomization_p": raw_p,
        "positive_matrices": int(np.count_nonzero(array > 0.0)),
        "negative_matrices": int(np.count_nonzero(array < 0.0)),
        "zero_matrices": int(np.count_nonzero(array == 0.0)),
        "maximum_leave_one_matrix_out_influence": _maximum_leave_one_out_influence(
            array
        ),
    }
    return summary, boot, null


def _finite_last(values: NDArray) -> float:
    array = np.asarray(values)
    finite = array[np.isfinite(array)]
    return float(finite[-1]) if finite.size else float("nan")


def pulse_tables(
    cases: list[LaunchCase], batches: list[PhaseBatch]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, NDArray]]:
    case_by_id = {case.state_id: case for case in cases}
    lineage_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    arrays: dict[str, list[NDArray]] = {
        "similarity_to_anchor": [],
        "risk": [],
        "boundary_h": [],
        "growth_updates": [],
        "entropy": [],
        "top1_share": [],
        "occupied_types": [],
        "throughput": [],
        "anchor_composition": [],
        "final_composition": [],
    }
    row_index = 0
    for batch in batches:
        case = case_by_id[batch.state_id]
        for lineage in batch.lineages:
            finite_h = lineage.boundary_h[np.isfinite(lineage.boundary_h)]
            inherited = int(np.count_nonzero(finite_h > INHERITANCE_THRESHOLD))
            throughput_ratio = (
                lineage.anchor_throughput / lineage.launch_throughput
                if lineage.launch_throughput > 0.0
                else float("nan")
            )
            lineage_rows.append(
                {
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "state_id": case.state_id,
                    "launch": lineage.launch,
                    "convention": lineage.convention,
                    "pulse_length": lineage.pulse_length,
                    "replicate": lineage.replicate,
                    "row_index": row_index,
                    "pulse_completed": lineage.pulse_completed,
                    "pulse_observed_fissions": lineage.pulse_observed_fissions,
                    "release_completed": lineage.release_completed,
                    "release_observed_fissions": lineage.release_observed_fissions,
                    "edits_applied": lineage.edits_applied,
                    "release_interventions_applied": lineage.release_interventions_applied,
                    "persistence": lineage.persistence,
                    "first_departure_time": lineage.first_departure_time,
                    "release_inherited_fixed_fraction": inherited / RELEASE_HORIZON,
                    "final_similarity": _finite_last(lineage.similarity_to_anchor),
                    "minimum_similarity": float(
                        np.nanmin(lineage.similarity_to_anchor)
                        if np.any(np.isfinite(lineage.similarity_to_anchor))
                        else np.nan
                    ),
                    "launch_risk": lineage.launch_risk,
                    "launch_entropy": lineage.launch_entropy,
                    "launch_top1_share": lineage.launch_top1_share,
                    "launch_occupied_types": lineage.launch_occupied_types,
                    "launch_throughput": lineage.launch_throughput,
                    "anchor_risk": lineage.anchor_risk,
                    "anchor_entropy": lineage.anchor_entropy,
                    "anchor_top1_share": lineage.anchor_top1_share,
                    "anchor_occupied_types": lineage.anchor_occupied_types,
                    "anchor_throughput": lineage.anchor_throughput,
                    "risk_reduction": lineage.launch_risk - lineage.anchor_risk,
                    "entropy_reduction": lineage.launch_entropy - lineage.anchor_entropy,
                    "top1_increase": lineage.anchor_top1_share - lineage.launch_top1_share,
                    "occupied_reduction": lineage.launch_occupied_types
                    - lineage.anchor_occupied_types,
                    "log_throughput_ratio": float(np.log(throughput_ratio))
                    if throughput_ratio > 0.0
                    else float("nan"),
                    "pulse_record_digest": lineage.pulse_record_digest,
                    "release_record_digest": lineage.release_record_digest,
                }
            )
            for action_index, (step, edit) in enumerate(
                zip(lineage.action_steps, lineage.actions, strict=True)
            ):
                edit_rows.append(
                    {
                        "candidate": case.candidate,
                        "matrix_id": case.matrix_id,
                        "state_id": case.state_id,
                        "launch": lineage.launch,
                        "convention": lineage.convention,
                        "pulse_length": lineage.pulse_length,
                        "replicate": lineage.replicate,
                        "lineage_row_index": row_index,
                        "action_index": action_index,
                        "boundary": step,
                        "remove_type": edit.remove_type,
                        "add_type": edit.add_type,
                        "risk_before": float(lineage.action_risk_before[action_index]),
                        "risk_after": float(lineage.action_risk_after[action_index]),
                    }
                )
            arrays["similarity_to_anchor"].append(lineage.similarity_to_anchor)
            arrays["risk"].append(lineage.risk)
            arrays["boundary_h"].append(lineage.boundary_h)
            arrays["growth_updates"].append(lineage.growth_updates)
            arrays["entropy"].append(lineage.entropy)
            arrays["top1_share"].append(lineage.top1_share)
            arrays["occupied_types"].append(lineage.occupied_types)
            arrays["throughput"].append(lineage.throughput)
            arrays["anchor_composition"].append(lineage.anchor_composition)
            arrays["final_composition"].append(lineage.final_snapshot.composition)
            row_index += 1
    lineage_frame = pd.DataFrame(lineage_rows)
    group = ["candidate", "matrix_id", "launch", "convention", "pulse_length"]
    numeric = [
        "persistence",
        "pulse_completed",
        "release_completed",
        "edits_applied",
        "release_interventions_applied",
        "release_inherited_fixed_fraction",
        "final_similarity",
        "minimum_similarity",
        "launch_risk",
        "launch_entropy",
        "launch_top1_share",
        "launch_occupied_types",
        "launch_throughput",
        "anchor_risk",
        "anchor_entropy",
        "anchor_top1_share",
        "anchor_occupied_types",
        "anchor_throughput",
        "risk_reduction",
        "entropy_reduction",
        "top1_increase",
        "occupied_reduction",
        "log_throughput_ratio",
    ]
    matrix_frame = (
        lineage_frame.groupby(group, sort=True, as_index=False)[numeric].mean()
    )
    counts = lineage_frame.groupby(group, sort=True).size()
    if not bool((counts == REPLICATES).all()):
        raise AssertionError("CR9M matrix table lost replicate rows")
    stacked = {name: np.stack(values) for name, values in arrays.items()}
    return lineage_frame, matrix_frame, pd.DataFrame(edit_rows), stacked


def _matrix_spearman(
    selected: pd.DataFrame, value: str
) -> NDArray[np.float64]:
    pivot = selected.pivot(
        index="matrix_id", columns="pulse_length", values=value
    ).reindex(index=np.arange(MATRICES), columns=PULSE_LENGTHS)
    if pivot.isna().any().any():
        raise ValueError(f"CR9M lacks a complete matrix block for {value}")
    x = np.asarray(PULSE_LENGTHS, dtype=np.float64)
    return np.asarray(
        [
            spearman_constant_zero(x, row.to_numpy(dtype=np.float64))
            for _, row in pivot.iterrows()
        ],
        dtype=np.float64,
    )


def _matrix_association(
    selected: pd.DataFrame, left: str, right: str
) -> NDArray[np.float64]:
    a = selected.pivot(
        index="matrix_id", columns="pulse_length", values=left
    ).reindex(index=np.arange(MATRICES), columns=PULSE_LENGTHS)
    b = selected.pivot(
        index="matrix_id", columns="pulse_length", values=right
    ).reindex(index=np.arange(MATRICES), columns=PULSE_LENGTHS)
    if a.isna().any().any() or b.isna().any().any():
        raise ValueError("CR9M mechanism association lacks a complete matrix block")
    return np.asarray(
        [
            spearman_constant_zero(
                a.iloc[index].to_numpy(dtype=np.float64),
                b.iloc[index].to_numpy(dtype=np.float64),
            )
            for index in range(MATRICES)
        ],
        dtype=np.float64,
    )


def compute_inference(
    matrix_frame: pd.DataFrame,
    draws: dict[str, NDArray],
    *,
    replay_exact: bool,
    release_zero_interventions: bool,
    readback_exact: bool = True,
) -> tuple[dict[str, Any], dict[str, NDArray], pd.DataFrame]:
    bootstrap = np.asarray(draws["bootstrap_indices"], dtype=np.int64)
    signs = np.asarray(draws["randomization_signs"], dtype=np.float64)
    if bootstrap.shape != (BOOTSTRAP_REPETITIONS, MATRICES):
        raise ValueError("CR9M bootstrap lost whole-matrix blocks")
    if signs.shape != (RANDOMIZATION_REPETITIONS, MATRICES):
        raise ValueError("CR9M randomization lost whole-matrix blocks")
    stored: dict[str, NDArray] = {
        "bootstrap_indices": bootstrap,
        "randomization_signs": signs,
    }
    cell_rows: list[dict[str, Any]] = []
    cell_arrays: dict[tuple[str, str, str], NDArray] = {}
    cell_summaries: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    mechanism: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        for launch in LAUNCHES:
            for convention in CONVENTIONS:
                selected = matrix_frame[
                    (matrix_frame["candidate"].astype(str).str.zfill(2) == candidate)
                    & (matrix_frame["launch"] == launch)
                    & (matrix_frame["convention"] == convention)
                ]
                rho = _matrix_spearman(selected, "persistence")
                cell_arrays[(candidate, launch, convention)] = rho
                summary, boot, null = _contrast_summary(rho, bootstrap, signs)
                summary.update(
                    {
                        "candidate": candidate,
                        "launch": launch,
                        "convention": convention,
                        "positive_ci95": bool(summary["bootstrap_ci95"][0] > 0.0),
                    }
                )
                p1 = selected[selected["pulse_length"] == PULSE_LENGTHS[0]][
                    "persistence"
                ].to_numpy(dtype=np.float64)
                p60 = selected[selected["pulse_length"] == PULSE_LENGTHS[-1]][
                    "persistence"
                ].to_numpy(dtype=np.float64)
                endpoint, endpoint_boot, endpoint_null = _contrast_summary(
                    p60 - p1, bootstrap, signs
                )
                summary["p60_minus_p1"] = endpoint
                stored[f"c{candidate}_{launch}_{convention}_rho"] = rho
                stored[f"c{candidate}_{launch}_{convention}_rho_bootstrap"] = boot
                stored[f"c{candidate}_{launch}_{convention}_rho_randomization"] = null
                stored[f"c{candidate}_{launch}_{convention}_endpoint_bootstrap"] = endpoint_boot
                stored[f"c{candidate}_{launch}_{convention}_endpoint_randomization"] = endpoint_null
                cell_summaries.append(summary)
                for matrix_id, value in enumerate(rho):
                    cell_rows.append(
                        {
                            "candidate": candidate,
                            "matrix_id": matrix_id,
                            "launch": launch,
                            "convention": convention,
                            "persistence_spearman": value,
                        }
                    )
                mechanism_item: dict[str, Any] = {
                    "candidate": candidate,
                    "launch": launch,
                    "convention": convention,
                    "pulse_to_anchor_consolidation": {},
                    "anchor_consolidation_to_persistence": {},
                }
                for metric in (
                    "top1_increase",
                    "entropy_reduction",
                    "occupied_reduction",
                    "log_throughput_ratio",
                    "risk_reduction",
                ):
                    pulse_rho = _matrix_spearman(selected, metric)
                    association = _matrix_association(selected, metric, "persistence")
                    pulse_summary, _, _ = _contrast_summary(pulse_rho, bootstrap, signs)
                    association_summary, _, _ = _contrast_summary(
                        association, bootstrap, signs
                    )
                    mechanism_item["pulse_to_anchor_consolidation"][metric] = pulse_summary
                    mechanism_item["anchor_consolidation_to_persistence"][metric] = (
                        association_summary
                    )
                mechanism.append(mechanism_item)

        nr = cell_arrays[(candidate, "NASCENT", "RELAXED")]
        np_edit = cell_arrays[(candidate, "NASCENT", "POST_EDIT")]
        mr = cell_arrays[(candidate, "MATURE", "RELAXED")]
        mp = cell_arrays[(candidate, "MATURE", "POST_EDIT")]
        launch_contrast = 0.5 * ((nr - mr) + (np_edit - mp))
        convention_contrast = 0.5 * ((nr - np_edit) + (mr - mp))
        interaction = (nr - np_edit) - (mr - mp)
        launch_summary, launch_boot, launch_null = _contrast_summary(
            launch_contrast, bootstrap, signs
        )
        convention_summary, convention_boot, convention_null = _contrast_summary(
            convention_contrast, bootstrap, signs
        )
        interaction_summary, interaction_boot, interaction_null = _contrast_summary(
            interaction, bootstrap, signs
        )
        candidate_summaries.append(
            {
                "candidate": candidate,
                "launch_moderation": launch_summary,
                "relaxed_minus_post_edit": convention_summary,
                "launch_by_convention_interaction": interaction_summary,
            }
        )
        stored[f"c{candidate}_launch_moderation_matrix"] = launch_contrast
        stored[f"c{candidate}_launch_moderation_bootstrap"] = launch_boot
        stored[f"c{candidate}_launch_moderation_randomization"] = launch_null
        stored[f"c{candidate}_convention_matrix"] = convention_contrast
        stored[f"c{candidate}_convention_bootstrap"] = convention_boot
        stored[f"c{candidate}_convention_randomization"] = convention_null
        stored[f"c{candidate}_interaction_matrix"] = interaction
        stored[f"c{candidate}_interaction_bootstrap"] = interaction_boot
        stored[f"c{candidate}_interaction_randomization"] = interaction_null

    primary_adjusted = _holm_adjust(
        [item["launch_moderation"]["raw_one_sided_randomization_p"] for item in candidate_summaries]
    )
    for item, adjusted in zip(candidate_summaries, primary_adjusted, strict=True):
        launch = item["launch_moderation"]
        launch["holm_adjusted_p"] = adjusted
        launch["candidate_primary_gate"] = bool(
            launch["estimate"] > 0.0
            and launch["bootstrap_ci95"][0] > 0.0
            and adjusted < 0.05
        )

    nascent = [item for item in cell_summaries if item["launch"] == "NASCENT"]
    nascent_adjusted = _holm_adjust(
        [item["raw_one_sided_randomization_p"] for item in nascent]
    )
    for item, adjusted in zip(nascent, nascent_adjusted, strict=True):
        item["holm_adjusted_nascent_family_p"] = adjusted
        item["nascent_cell_gate"] = bool(
            item["estimate"] > 0.0
            and item["bootstrap_ci95"][0] > 0.0
            and adjusted < 0.05
        )
    for item in cell_summaries:
        if item["launch"] != "NASCENT":
            item["holm_adjusted_nascent_family_p"] = None
            item["nascent_cell_gate"] = None

    efficacy = bool(
        all(
            item["launch_moderation"]["candidate_primary_gate"]
            for item in candidate_summaries
        )
    )
    robust_nascent = bool(all(item["nascent_cell_gate"] for item in nascent))
    integrity = bool(replay_exact and release_zero_interventions and readback_exact)
    metrics = {
        "format": "codex-intervention-cr9m-primary-metrics-v1",
        "primary_launch_moderation_gate": efficacy,
        "protocol_robust_nascent_hysteresis": robust_nascent,
        "complete_registered_gate_with_integrity": bool(efficacy and integrity),
        "candidate_contrasts": candidate_summaries,
        "factorial_cells": cell_summaries,
        "packing_diagnostics": mechanism,
        "integrity": {
            "exact_replay": bool(replay_exact),
            "release_interventions_exactly_zero": bool(release_zero_interventions),
            "artifact_readback_exact": bool(readback_exact),
        },
        "sealed_cr9_result_unchanged": True,
        "packing_diagnostics_cannot_rescue_primary": True,
    }
    return metrics, stored, pd.DataFrame(cell_rows)


def _verify_upstream() -> dict[str, Any]:
    verify_checksums(CR9_REGISTRATION)
    verify_checksums(CR9_RESULT)
    registration = json.loads((CR9_REGISTRATION / "registration.json").read_text())
    manifest = json.loads((CR9_RESULT / "manifest.json").read_text())
    metrics = json.loads((CR9_RESULT / "primary_metrics.json").read_text())
    if registration["registration_id"] != EXPECTED_CR9_REGISTRATION_ID:
        raise ValueError("sealed CR9 registration ID changed")
    if sha256_file(CR9_REGISTRATION / "frozen_full_predictor.npz") != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen JOINT_BREAK_RUN3 predictor changed")
    if not (
        manifest["complete_exact_replay"]
        and manifest["complete_readback_exact"]
        and manifest["release_interventions_exactly_zero"]
    ):
        raise ValueError("sealed CR9 integrity is no longer passing")
    if manifest["complete_two_candidate_hysteresis_gate"]:
        raise ValueError("sealed CR9 failed gate was unexpectedly changed")
    if metrics["pulse"]["complete_two_candidate_hysteresis_gate"]:
        raise ValueError("sealed CR9 primary metrics were unexpectedly changed")
    return {
        "cr9_registration_id": registration["registration_id"],
        "cr9_registration_checksum_manifest_sha256": sha256_file(
            CR9_REGISTRATION / "SHA256SUMS"
        ),
        "cr9_result_checksum_manifest_sha256": sha256_file(
            CR9_RESULT / "SHA256SUMS"
        ),
        "cr9_failed_hysteresis_gate_retained": True,
        "cr9_exact_replay": manifest["complete_exact_replay"],
        "cr9_complete_readback": manifest["complete_readback_exact"],
    }


def _artificial_case() -> tuple[LaunchCase, ExperimentConfig]:
    config = GardConfig(
        n_types=100,
        n_min=4,
        n_max=8,
        beta_log_mean=-4.0,
        beta_log_sd=4.0,
        max_growth_steps=1_000,
        generations=60,
    )
    current_experiment = ExperimentConfig(
        gard=config,
        development=CohortConfig(1, 1, (0, 60)),
        confirmation=CohortConfig(1, 1, (0, 60)),
        horizon=3,
        bootstrap_repetitions=16,
        permutation_repetitions=16,
        master_seed=SEEDS["smoke"],
    )
    beta = np.full((100, 100), 1_000.0, dtype=np.float64)
    np.fill_diagonal(beta, 1_100.0)
    composition = np.zeros(100, dtype=np.int64)
    composition[:4] = 1
    snapshot = Snapshot(
        composition=composition,
        generation=0,
        inheritance=(),
        boundary_h=(),
        previous_growth_steps=0,
        cumulative_growth_steps=0,
    )
    return (
        LaunchCase("CR9M-ARTIFICIAL", "02", 0, "NASCENT", beta, snapshot),
        current_experiment,
    )


def _artificial_execution(model_path: Path, registration_id: str) -> PhaseBatch:
    case, current_experiment = _artificial_case()
    return _run_case(
        case,
        current_experiment,
        model_path,
        registration_id,
        conventions=CONVENTIONS,
        pulse_lengths=(1, 2),
        replicates=1,
        release_horizon=3,
    )


def _history_clock_diagnostic(model_path: Path) -> dict[str, Any]:
    case, current_experiment = _artificial_case()
    predictor = FrozenFullPredictor.load(model_path)
    young = Snapshot(
        composition=case.snapshot.composition.copy(),
        generation=1,
        inheritance=(True,),
        boundary_h=(0.95,),
        previous_growth_steps=10,
        cumulative_growth_steps=10,
    )
    old = Snapshot(
        composition=case.snapshot.composition.copy(),
        generation=60,
        inheritance=tuple([True, False, True] * 20),
        boundary_h=tuple([0.95, 0.80, 0.93] * 20),
        previous_growth_steps=75,
        cumulative_growth_steps=3_500,
    )
    young_noop, young_scores = score_legal_edits(
        predictor, "02", young, case.beta, current_experiment.gard
    )
    old_noop, old_scores = score_legal_edits(
        predictor, "02", old, case.beta, current_experiment.gard
    )
    young_prob = np.asarray(
        [item.predicted_probability for item in young_scores], dtype=np.float64
    )
    old_prob = np.asarray(
        [item.predicted_probability for item in old_scores], dtype=np.float64
    )
    young_index = int(np.flatnonzero(young_prob == young_prob.min())[0])
    old_index = int(np.flatnonzero(old_prob == old_prob.min())[0])
    return {
        "completed": True,
        "same_composition_beta_candidate": True,
        "history_and_clocks_only_changed": True,
        "legal_edit_order_identical": [item.edit for item in young_scores]
        == [item.edit for item in old_scores],
        "selected_edit_identical": young_scores[young_index].edit
        == old_scores[old_index].edit,
        "young_selected_edit": {
            "remove_type": young_scores[young_index].edit.remove_type,
            "add_type": young_scores[young_index].edit.add_type,
        },
        "old_selected_edit": {
            "remove_type": old_scores[old_index].edit.remove_type,
            "add_type": old_scores[old_index].edit.add_type,
        },
        "noop_predictions_differ": bool(young_noop != old_noop),
        "young_noop_probability": float(young_noop),
        "old_noop_probability": float(old_noop),
    }


def validation_checks() -> dict[str, Any]:
    upstream = _verify_upstream()
    model_path = CR9_REGISTRATION / "frozen_full_predictor.npz"
    first = _artificial_execution(model_path, "pre-registration-artificial")
    second = _artificial_execution(model_path, "pre-registration-artificial")
    by_cell = {
        (item.convention, item.pulse_length): item for item in first.lineages
    }
    relaxed_one = by_cell[("RELAXED", 1)]
    relaxed_two = by_cell[("RELAXED", 2)]
    post_one = by_cell[("POST_EDIT", 1)]
    post_two = by_cell[("POST_EDIT", 2)]
    case, _ = _artificial_case()
    paired = LaunchCase(
        "CR9M-ARTIFICIAL-MATURE",
        case.candidate,
        case.matrix_id,
        "MATURE",
        case.beta,
        Snapshot(
            case.snapshot.composition.copy(),
            60,
            (True,),
            (0.95,),
            10,
            600,
        ),
    )
    draws = inference_draws()
    history_diagnostic = _history_clock_diagnostic(model_path)
    predictor_one = FrozenFullPredictor.load(model_path)
    predictor_two = FrozenFullPredictor.load(model_path)
    serialization_prediction_one = predictor_one.predict_snapshot(
        case.candidate, case.snapshot, case.beta, _artificial_case()[1].gard
    )
    serialization_prediction_two = predictor_two.predict_snapshot(
        case.candidate, case.snapshot, case.beta, _artificial_case()[1].gard
    )
    checks = {
        "upstream_sealed_cr9_integrity_pass": True,
        "design_exact": MATRICES == 48
        and REPLICATES == 3
        and MATURE_LANDMARK == 60
        and RELEASE_HORIZON == 60
        and PULSE_LENGTHS == (1, 2, 4, 8, 16, 32, 60)
        and LAUNCHES == ("NASCENT", "MATURE")
        and CONVENTIONS == ("RELAXED", "POST_EDIT"),
        "seed_domains_unique": len(SEEDS) == len(set(SEEDS.values())),
        "future_seed_excludes_launch_convention_and_pulse": _future_seed(case, 0)
        == _future_seed(paired, 0),
        "future_replicates_distinct": _future_seed(case, 0) != _future_seed(case, 1),
        "nascent_fixture_generation_history_and_clocks_exact": case.snapshot.generation
        == 0
        and case.snapshot.inheritance == ()
        and case.snapshot.boundary_h == ()
        and case.snapshot.previous_growth_steps == 0
        and case.snapshot.cumulative_growth_steps == 0,
        "nascent_fixture_mass_and_distinct_types_exact": int(
            case.snapshot.composition.sum()
        )
        == 4
        and int(np.count_nonzero(case.snapshot.composition)) == 4,
        "relaxed_p1_zero_edit": relaxed_one.edits_applied == 0
        and relaxed_one.action_steps == (),
        "relaxed_p2_prefix_exact": relaxed_two.edits_applied == 1
        and relaxed_two.action_steps == (1,),
        "post_edit_p1_exact": post_one.edits_applied == 1
        and post_one.action_steps == (1,),
        "post_edit_p2_prefix_exact": post_two.edits_applied == 2
        and post_two.action_steps == (1, 2),
        "artificial_release_zero_interventions": all(
            item.release_interventions_applied == 0 for item in first.lineages
        ),
        "artificial_complete_deterministic_replay": batch_digest(first)
        == batch_digest(second),
        "all_artificial_actions_legal_type_substitutions": all(
            edit.remove_type != edit.add_type
            and 0 <= edit.remove_type < 100
            and 0 <= edit.add_type < 100
            for item in first.lineages
            for edit in item.actions
        ),
        "history_clock_diagnostic_completed": history_diagnostic["completed"],
        "whole_matrix_draw_shapes_exact": draws["bootstrap_indices"].shape
        == (BOOTSTRAP_REPETITIONS, MATRICES)
        and draws["randomization_signs"].shape
        == (RANDOMIZATION_REPETITIONS, MATRICES),
        "constant_spearman_rule_exact": spearman_constant_zero(
            np.arange(7), np.ones(7)
        )
        == 0.0,
        "increasing_spearman_fixture_exact": spearman_constant_zero(
            np.arange(7), np.arange(7)
        )
        == 1.0,
        "frozen_model_hash_exact": sha256_file(model_path) == EXPECTED_MODEL_SHA256,
        "frozen_model_serialization_predictions_exact": serialization_prediction_one
        == serialization_prediction_two,
        "claim_boundary_exact": protocol()["boundary"]["cannot_rescue_or_replace_cr9"]
        and any("biological memory" in item for item in protocol()["prohibited_claims"]),
    }
    return {
        "format": VALIDATION_FORMAT,
        "checks": checks,
        "check_count": len(checks),
        "all_checks_passed": bool(all(checks.values())),
        "history_clock_diagnostic": history_diagnostic,
        "upstream": upstream,
        "artificial_non_scientific_fixture_only": True,
        "scientific_cr9m_matrices_generated": 0,
        "scientific_cr9m_lineages_generated": 0,
    }


def validate(output: Path = DEFAULT_VALIDATION) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    value = validation_checks()
    if not value["all_checks_passed"]:
        raise AssertionError(
            {key: result for key, result in value["checks"].items() if not result}
        )
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CR9M full repository validation failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    with _atomic_destination(output) as destination:
        payload = dict(value)
        payload["source_hashes"] = source_hashes()
        payload["source_tree_sha256"] = _canonical_digest(payload["source_hashes"])
        payload["pytest_returncode"] = completed.returncode
        payload["pytest_summary"] = completed.stdout.strip().splitlines()[-1]
        (destination / "validation.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n"
        )
        (destination / "pytest_output.txt").write_text(
            "$ " + " ".join(command) + "\n" + completed.stdout + completed.stderr
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR9M validation sealed: {output}", flush=True)


def _append_ledger(marker: str, lines: list[str]) -> None:
    path = ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    current = path.read_text(encoding="utf-8").rstrip() + "\n"
    if marker in current:
        return
    path.write_text(current + "\n" + marker + "\n" + "\n".join(lines))


def register(
    validation_directory: Path = DEFAULT_VALIDATION,
    output: Path = DEFAULT_REGISTRATION,
) -> None:
    validation_directory = validation_directory.resolve()
    output = output.resolve()
    verify_checksums(validation_directory)
    validation = json.loads((validation_directory / "validation.json").read_text())
    if not validation["all_checks_passed"]:
        raise ValueError("CR9M validation did not pass")
    if validation["source_hashes"] != source_hashes():
        raise ValueError("CR9M source changed after validation")
    upstream = _verify_upstream()
    for forbidden in (DEFAULT_REGISTRATION, DEFAULT_SMOKE, DEFAULT_OUTPUT, DEFAULT_WORK):
        if forbidden.exists():
            raise FileExistsError(
                f"CR9M preregistration artifact already exists: {forbidden}"
            )
    body: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol": protocol(),
        "protocol_id": protocol()["protocol_id"],
        "source_hashes": source_hashes(),
        "source_tree_sha256": _canonical_digest(source_hashes()),
        "seed_registry": SEEDS,
        "frozen_model_sha256": EXPECTED_MODEL_SHA256,
        "validation_checksum_manifest_sha256": sha256_file(
            validation_directory / "SHA256SUMS"
        ),
        "upstream": upstream,
        "scientific_matrices_at_registration": 0,
        "scientific_lineages_at_registration": 0,
    }
    registration_id = _canonical_digest(_json_ready(body))
    body["registration_id"] = registration_id
    with _atomic_destination(output) as destination:
        shutil.copy2(ROOT / DOCUMENT, destination / "preregistration.md")
        shutil.copy2(
            validation_directory / "validation.json", destination / "validation.json"
        )
        shutil.copy2(
            CR9_REGISTRATION / "frozen_full_predictor.npz",
            destination / "frozen_full_predictor.npz",
        )
        (destination / "intervention_protocol.json").write_text(
            json.dumps(_json_ready(protocol()), indent=2, sort_keys=True) + "\n"
        )
        (destination / "intervention_seed_registry.json").write_text(
            json.dumps(SEEDS, indent=2, sort_keys=True) + "\n"
        )
        (destination / "registration.json").write_text(
            json.dumps(_json_ready(body), indent=2, sort_keys=True) + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)
    _append_ledger(
        f"<!-- registered-cr9m-{registration_id} -->",
        [
            "## CR9M launch-state moderation registered",
            "",
            f"- Registration: `{registration_id}`.",
            "- CR9 remains sealed as a failed generation-60 accumulating-hysteresis gate; CR9M cannot rescue or replace it.",
            "- Forty-eight new matrices, two candidates, paired nascent and natural generation-60 launches, two frozen anchor conventions, three replicates, and seven pulse lengths were sealed before scientific generation.",
            "- The primary estimand is the paired fresh-minus-mature matrix-Spearman contrast averaged over conventions.",
            "- No CR9M scientific matrix or lineage existed at registration.",
            "",
        ],
    )
    print(f"CR9M registered: {registration_id}", flush=True)


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    registration = json.loads((directory / "registration.json").read_text())
    if registration["format"] != REGISTRATION_FORMAT:
        raise ValueError("unsupported CR9M registration format")
    if registration["source_hashes"] != source_hashes():
        raise ValueError("CR9M registered source tree changed")
    body = dict(registration)
    observed = body.pop("registration_id")
    if _canonical_digest(_json_ready(body)) != observed:
        raise ValueError("CR9M registration ID changed")
    if registration["protocol"] != protocol() or registration["seed_registry"] != SEEDS:
        raise ValueError("CR9M registered protocol or seed registry changed")
    if sha256_file(directory / "frozen_full_predictor.npz") != EXPECTED_MODEL_SHA256:
        raise ValueError("CR9M frozen predictor copy changed")
    _verify_upstream()
    return registration


def smoke(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_SMOKE,
) -> None:
    registration = verify_registration(registration_directory)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    model_path = registration_directory / "frozen_full_predictor.npz"
    first = _artificial_execution(model_path, registration["registration_id"])
    second = _artificial_execution(model_path, registration["registration_id"])
    payload = {
        "format": "codex-intervention-cr9m-smoke-v1",
        "registration_id": registration["registration_id"],
        "artificial_non_scientific_fixture": True,
        "both_conventions_and_p1_zero_edit_exercised": True,
        "exact_replay": batch_digest(first) == batch_digest(second),
        "release_applied_zero_interventions": all(
            item.release_interventions_applied == 0 for item in first.lineages
        ),
        "effect_sizes_cell_order_event_rates_and_candidate_differences_disclosed": False,
        "scientific_cr9m_matrices_generated": 0,
        "scientific_cr9m_lineages_generated": 0,
    }
    if not all(
        payload[key]
        for key in (
            "artificial_non_scientific_fixture",
            "both_conventions_and_p1_zero_edit_exercised",
            "exact_replay",
            "release_applied_zero_interventions",
        )
    ):
        raise AssertionError("CR9M artificial smoke failed")
    with _atomic_destination(output) as destination:
        (destination / "smoke.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR9M non-scientific smoke passed: {output}", flush=True)


def _reports(metrics: dict[str, Any]) -> tuple[str, str]:
    technical = [
        "# CR9M launch-state moderation",
        "",
        "CR9 remains sealed and unchanged. CR9M primary launch-moderation gate: "
        f"**{metrics['primary_launch_moderation_gate']}**.",
        "Protocol-robust nascent-hysteresis classification: "
        f"**{metrics['protocol_robust_nascent_hysteresis']}**.",
        "Complete registered gate including integrity: "
        f"**{metrics['complete_registered_gate_with_integrity']}**.",
        "",
        "## Primary launch-moderation contrasts",
        "",
    ]
    for item in metrics["candidate_contrasts"]:
        value = item["launch_moderation"]
        technical.extend(
            [
                f"### Candidate {item['candidate']}",
                "",
                f"- Fresh-minus-mature mean matrix Spearman: {value['estimate']:+.6f}.",
                f"- 95% whole-matrix bootstrap CI: [{value['bootstrap_ci95'][0]:+.6f}, {value['bootstrap_ci95'][1]:+.6f}].",
                f"- 90% whole-matrix bootstrap CI: [{value['bootstrap_ci90'][0]:+.6f}, {value['bootstrap_ci90'][1]:+.6f}].",
                f"- Holm-adjusted one-sided matrix-randomization p: {value['holm_adjusted_p']:.6g}.",
                f"- Candidate primary gate: **{value['candidate_primary_gate']}**.",
                "",
            ]
        )
    technical.extend(["## Factorial cell correlations", ""])
    for item in metrics["factorial_cells"]:
        technical.append(
            f"- Candidate {item['candidate']} / {item['launch']} / {item['convention']}: "
            f"rho {item['estimate']:+.6f}, CI95 "
            f"[{item['bootstrap_ci95'][0]:+.6f}, {item['bootstrap_ci95'][1]:+.6f}], "
            f"P60-P1 {item['p60_minus_p1']['estimate']:+.3f}."
        )
    technical.extend(
        [
            "",
            "## Convention and interaction diagnostics",
            "",
        ]
    )
    for item in metrics["candidate_contrasts"]:
        convention = item["relaxed_minus_post_edit"]
        interaction = item["launch_by_convention_interaction"]
        technical.extend(
            [
                f"- Candidate {item['candidate']} relaxed-minus-post-edit: "
                f"{convention['estimate']:+.6f} "
                f"[{convention['bootstrap_ci95'][0]:+.6f}, {convention['bootstrap_ci95'][1]:+.6f}].",
                f"- Candidate {item['candidate']} launch×convention interaction: "
                f"{interaction['estimate']:+.6f} "
                f"[{interaction['bootstrap_ci95'][0]:+.6f}, {interaction['bootstrap_ci95'][1]:+.6f}].",
            ]
        )
    technical.extend(
        [
            "",
            "## Integrity and claim boundary",
            "",
            f"- Exact replay: **{metrics['integrity']['exact_replay']}**.",
            f"- Release interventions exactly zero: **{metrics['integrity']['release_interventions_exactly_zero']}**.",
            f"- Artifact readback exact: **{metrics['integrity']['artifact_readback_exact']}**.",
            "- Registered packing diagnostics are included in `primary_metrics.json`; they cannot rescue the primary gate.",
            "- CR9M tests transient consolidation, not an autonomous restoring basin or installed biological memory.",
            "",
        ]
    )
    if metrics["primary_launch_moderation_gate"]:
        main_sentence = (
            "The paired test found that longer steering leaves a more strongly duration-dependent temporary trace when it starts from a young assembly than when it starts from an already evolved assembly."
        )
    else:
        main_sentence = (
            "The paired test did not establish in both simulator candidates that young launch states have a stronger steering-duration effect than evolved launch states."
        )
    if metrics["protocol_robust_nascent_hysteresis"]:
        support_sentence = (
            "The young-assembly effect also survived both ways of defining the last pulse boundary and anchor."
        )
    else:
        support_sentence = (
            "The stricter requirement that the young-assembly effect survive both anchor conventions in both candidates did not pass."
        )
    lay = [
        "# CR9M in plain language",
        "",
        main_sentence,
        "",
        support_sentence,
        "",
        "This follow-up does not change the earlier CR9 result. It asks why Fable's young, initially diffuse assemblies and Codex's already evolved assemblies behaved differently.",
        "",
        "Even a positive result would mean only that outside editing can temporarily consolidate a young chemical assembly. It would not show that the assembly remembers, repairs, or restores itself after the controller is removed.",
        "",
    ]
    return "\n".join(technical), "\n".join(lay)


def _write_result(
    output: Path,
    registration: dict[str, Any],
    cases: list[LaunchCase],
    metrics: dict[str, Any],
    stored_inference: dict[str, NDArray],
    cell_effects: pd.DataFrame,
    replay: dict[str, Any],
    data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, NDArray]],
) -> None:
    lineage, matrix, edits, arrays = data
    technical, lay = _reports(metrics)
    supported: list[str] = []
    if metrics["primary_launch_moderation_gate"]:
        supported.append(
            "launch maturity moderates the steering-duration versus transient-persistence relationship in both Codex candidates"
        )
    if metrics["protocol_robust_nascent_hysteresis"]:
        supported.append(
            "nascent accumulating transient hysteresis is positive under both registered anchor conventions in both candidates"
        )
    failed: list[str] = []
    if not metrics["primary_launch_moderation_gate"]:
        failed.append("two-candidate launch-moderation gate")
    if not metrics["protocol_robust_nascent_hysteresis"]:
        failed.append("protocol-robust nascent-hysteresis classification")
    claims = {
        "supported": supported,
        "failed_predictions": failed,
        "sealed_cr9_failure_retained": True,
        "packing_diagnostics_cannot_rescue_primary": True,
        "unresolved": [
            "whether any transient trace is a self-restoring basin",
            "whether the moderation generalizes beyond the two registered simulator candidates",
            "whether composition packing formally mediates the duration effect",
        ],
        "prohibited": protocol()["prohibited_claims"],
    }
    beta_by_matrix = np.stack(
        [
            next(case.beta for case in cases if case.matrix_id == matrix_id)
            for matrix_id in range(MATRICES)
        ]
    )
    launch_compositions = np.stack([case.snapshot.composition for case in cases])
    with _atomic_destination(output) as destination:
        metrics_text = json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n"
        (destination / "primary_metrics.json").write_text(metrics_text)
        (destination / "SCIENTIFIC_REPORT.md").write_text(technical)
        (destination / "LAY_SUMMARY.md").write_text(lay)
        (destination / "claim_boundaries.json").write_text(
            json.dumps(claims, indent=2, sort_keys=True) + "\n"
        )
        replay_directory = destination / "replay_audits"
        replay_directory.mkdir()
        (replay_directory / "factorial_replay_audit.json").write_text(
            json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n"
        )
        lineage.to_csv(
            destination / "factorial_lineages.csv.gz", index=False, compression="gzip"
        )
        matrix.to_csv(destination / "factorial_matrix_effects.csv", index=False)
        edits.to_csv(
            destination / "selected_edits.csv.gz", index=False, compression="gzip"
        )
        cell_effects.to_csv(destination / "cell_matrix_spearman.csv", index=False)
        np.savez_compressed(destination / "trajectory_arrays.npz", **arrays)
        np.savez_compressed(destination / "inference_arrays.npz", **stored_inference)
        np.savez_compressed(
            destination / "state_and_matrix_arrays.npz",
            beta=beta_by_matrix,
            launch_compositions=launch_compositions,
            candidate=np.asarray([case.candidate for case in cases]),
            matrix_id=np.asarray([case.matrix_id for case in cases], dtype=np.int16),
            launch=np.asarray([case.launch for case in cases]),
            generation=np.asarray(
                [case.snapshot.generation for case in cases], dtype=np.int16
            ),
        )
        expected_lineages = (
            MATRICES
            * len(CANDIDATES)
            * len(LAUNCHES)
            * len(CONVENTIONS)
            * len(PULSE_LENGTHS)
            * REPLICATES
        )
        expected_matrix_rows = (
            MATRICES
            * len(CANDIDATES)
            * len(LAUNCHES)
            * len(CONVENTIONS)
            * len(PULSE_LENGTHS)
        )
        expected_cell_rows = MATRICES * len(CANDIDATES) * len(LAUNCHES) * len(CONVENTIONS)
        readback = {
            "primary_metrics_exact": (destination / "primary_metrics.json").read_text()
            == metrics_text,
            "lineage_rows_exact": len(pd.read_csv(destination / "factorial_lineages.csv.gz"))
            == expected_lineages,
            "matrix_rows_exact": len(pd.read_csv(destination / "factorial_matrix_effects.csv"))
            == expected_matrix_rows,
            "cell_rows_exact": len(pd.read_csv(destination / "cell_matrix_spearman.csv"))
            == expected_cell_rows,
        }
        with np.load(destination / "trajectory_arrays.npz", allow_pickle=False) as archive:
            readback["trajectory_shape_exact"] = archive["boundary_h"].shape == (
                expected_lineages,
                RELEASE_HORIZON,
            )
        readback["complete_readback_exact"] = bool(all(readback.values()))
        if not readback["complete_readback_exact"]:
            raise AssertionError(f"CR9M written-artifact readback failed: {readback}")
        (destination / "readback_audit.json").write_text(
            json.dumps(readback, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "launches": list(LAUNCHES),
            "conventions": list(CONVENTIONS),
            "replicates": REPLICATES,
            "pulse_lengths": list(PULSE_LENGTHS),
            "release_horizon": RELEASE_HORIZON,
            "generated_lineages": expected_lineages,
            "replayed_lineages": expected_lineages,
            "primary_launch_moderation_gate": metrics[
                "primary_launch_moderation_gate"
            ],
            "protocol_robust_nascent_hysteresis": metrics[
                "protocol_robust_nascent_hysteresis"
            ],
            "complete_registered_gate_with_integrity": metrics[
                "complete_registered_gate_with_integrity"
            ],
            "exact_replay": replay[
                "exact_state_action_endpoint_process_and_rng"
            ],
            "release_interventions_exactly_zero": metrics["integrity"][
                "release_interventions_exactly_zero"
            ],
            "complete_readback_exact": True,
            "sealed_cr9_unchanged": True,
            "cr10_launched": False,
            "mandatory_stop_after_this_stage": True,
            "runtime": _runtime_manifest(),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)


def _prepare_work(work: Path, output: Path, registration_id: str) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed CR9M result: {output}")
    free = shutil.disk_usage(ROOT).free
    if free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError(
            f"CR9M requires at least {MINIMUM_FREE_DISK_BYTES:,} free bytes; found {free:,}"
        )
    work.mkdir(parents=True, exist_ok=True)
    expected = {
        "format": "codex-intervention-cr9m-work-contract-v1",
        "registration_id": registration_id,
        "protocol_id": protocol()["protocol_id"],
    }
    path = work / "campaign_contract.json"
    if path.is_file():
        if json.loads(path.read_text()) != expected:
            raise ValueError("CR9M work directory belongs to another campaign")
    else:
        path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")


def run(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_OUTPUT,
    work: Path = DEFAULT_WORK,
    workers: int = min(os.cpu_count() or 1, 12),
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration_directory = registration_directory.resolve()
    output = output.resolve()
    work = work.resolve()
    registration = verify_registration(registration_directory)
    verify_checksums(DEFAULT_SMOKE)
    _prepare_work(work, output, registration["registration_id"])
    current_experiment = experiment()
    model_path = registration_directory / "frozen_full_predictor.npz"

    expected_cases = MATRICES * len(CANDIDATES) * len(LAUNCHES)
    _write_status(work, "building_paired_launch_states", 0, expected_cases)
    print(
        f"[cr9m 1/6] Building {MATRICES} matrices and {expected_cases} paired launch states",
        flush=True,
    )
    with threadpool_limits(limits=1):
        cases = build_cr9m_cohort(current_experiment)
    if len(cases) != expected_cases:
        raise AssertionError("CR9M fresh paired cohort is incomplete")

    lineages = expected_cases * REPLICATES * len(CONVENTIONS) * len(PULSE_LENGTHS)
    print(f"[cr9m 2/6] Running {lineages:,} factorial pulse/release lineages", flush=True)
    generated = run_phase_batches(
        cases,
        current_experiment,
        model_path,
        registration["registration_id"],
        work / "factorial" / "generate",
        workers,
        work,
        "factorial_generate",
    )
    print(f"[cr9m 3/6] Replaying all {lineages:,} lineages exactly", flush=True)
    replayed = run_phase_batches(
        cases,
        current_experiment,
        model_path,
        registration["registration_id"],
        work / "factorial" / "replay",
        workers,
        work,
        "factorial_replay",
    )
    replay = replay_audit(generated, replayed)
    if not replay["exact_state_action_endpoint_process_and_rng"]:
        raise AssertionError("CR9M exact replay failed")
    del replayed

    _write_status(work, "whole_matrix_inference", len(cases), len(cases))
    print("[cr9m 4/6] Building tables and whole-matrix factorial inference", flush=True)
    data = pulse_tables(cases, generated)
    release_zero = bool(
        all(
            item.release_interventions_applied == 0
            for batch in generated
            for item in batch.lineages
        )
    )
    metrics, stored, cell_effects = compute_inference(
        data[1],
        inference_draws(),
        replay_exact=True,
        release_zero_interventions=release_zero,
        readback_exact=True,
    )
    _write_status(work, "writing_and_reading_back_artifacts", len(cases), len(cases))
    print("[cr9m 5/6] Writing reports and exact readback audit", flush=True)
    _write_result(
        output,
        registration,
        cases,
        metrics,
        stored,
        cell_effects,
        replay,
        data,
    )
    _append_ledger(
        f"<!-- sealed-cr9m-{registration['registration_id']} -->",
        [
            "## CR9M launch-state moderation sealed",
            "",
            f"- Registration: `{registration['registration_id']}`.",
            f"- Result: `{output.relative_to(ROOT)}`.",
            f"- Two-candidate launch-moderation gate: **{metrics['primary_launch_moderation_gate']}**.",
            f"- Protocol-robust nascent-hysteresis classification: **{metrics['protocol_robust_nascent_hysteresis']}**.",
            f"- Complete gate with replay/release/readback integrity: **{metrics['complete_registered_gate_with_integrity']}**.",
            "- Sealed CR9 remains unchanged; packing diagnostics did not rescue or alter the primary gate.",
            "- CR10 was not launched automatically; mandatory review stop observed.",
            "",
        ],
    )
    _write_status(
        work,
        "sealed_complete_mandatory_review_stop",
        len(cases),
        len(cases),
        output=str(output),
        primary_launch_moderation_gate=metrics["primary_launch_moderation_gate"],
        protocol_robust_nascent_hysteresis=metrics[
            "protocol_robust_nascent_hysteresis"
        ],
    )
    print("[cr9m 6/6] Result sealed; STOPPED without launching CR10", flush=True)


def read_status(work: Path = DEFAULT_WORK) -> dict[str, Any]:
    work = work.resolve()
    path = work / "campaign_status.json"
    if not path.is_file():
        raise FileNotFoundError(f"CR9M status does not exist: {path}")
    value = json.loads(path.read_text())
    value["checkpoint_counts"] = {
        relative: len(list((work / relative).glob("*.pkl")))
        if (work / relative).is_dir()
        else 0
        for relative in ("factorial/generate", "factorial/replay")
    }
    value["free_disk_bytes"] = shutil.disk_usage(ROOT).free
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").add_argument(
        "--output", type=Path, default=DEFAULT_VALIDATION
    )
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    register_parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    commands.add_parser("verify").add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    smoke_parser.add_argument("--output", type=Path, default=DEFAULT_SMOKE)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    run_parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    commands.add_parser("status").add_argument(
        "--work-dir", type=Path, default=DEFAULT_WORK
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        validate(arguments.output)
    elif arguments.command == "register":
        register(arguments.validation, arguments.output)
    elif arguments.command == "verify":
        print(
            json.dumps(
                verify_registration(arguments.registration), indent=2, sort_keys=True
            )
        )
    elif arguments.command == "smoke":
        smoke(arguments.registration, arguments.output)
    elif arguments.command == "run":
        run(arguments.registration, arguments.output, arguments.work_dir, arguments.workers)
    elif arguments.command == "status":
        print(json.dumps(read_status(arguments.work_dir), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
