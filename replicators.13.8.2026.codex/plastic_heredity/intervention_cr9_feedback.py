"""Prospectively frozen CR9 control-half-life and sparse-feedback campaign."""

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
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from .config import CANDIDATES, CohortConfig, ExperimentConfig, GardConfig
from .experiment import StateCase, _json_ready, _runtime_manifest
from .intervention_core import (
    ControlledResult,
    FrozenFullPredictor,
    MolecularEdit,
    _records_digest,
    edited_snapshot,
    enumerate_legal_edits,
    score_legal_edits,
    simulate_controlled,
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
CR0_VALIDATION = RESULT_ROOT / "cr0_validation"
CR1_REGISTRATION = RESULT_ROOT / "cr1_confirmation_registration"
CR7_REGISTRATION = RESULT_ROOT / "cr7_steering_registration"
CR7_RESULT = RESULT_ROOT / "cr7_closed_loop_steering"
CR8_RESULT = RESULT_ROOT / "cr8_steer_release_challenge"

DEFAULT_VALIDATION = RESULT_ROOT / "cr9_feedback_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "cr9_feedback_registration"
DEFAULT_SMOKE = RESULT_ROOT / "cr9_feedback_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "cr9_control_half_life"
DEFAULT_WORK = RESULT_ROOT / ".cr9_control_half_life_work"

DOCUMENT = "CODEX_INTERVENTION_CR9_PREREGISTRATION.md"
PROGRAM_FORMAT = "codex-intervention-cr9-feedback-v1"
VALIDATION_FORMAT = "codex-intervention-cr9-validation-v1"
REGISTRATION_FORMAT = "codex-intervention-cr9-registration-v1"
RESULT_FORMAT = "codex-intervention-cr9-result-v1"
CHECKPOINT_FORMAT = "codex-intervention-cr9-checkpoint-v1"
STATUS_FORMAT = "codex-intervention-cr9-status-v1"
LABEL = "INTCR9_FEEDBACK_V1"

EXPECTED_CR7_REGISTRATION_ID = (
    "41cf815a63129f40c04c7fb260f0f90c713adb9743eaae8479a5f6046e826e70"
)
EXPECTED_MODEL_SHA256 = (
    "9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af"
)

MATRICES = 48
LANDMARK = 60
REPLICATES = 6
HORIZON = 60
PULSE_LENGTHS = (1, 2, 4, 8, 16, 32, 60)
PERIODS = (1, 2, 4, 8, 16)
PERIODIC_POLICIES = tuple(
    [f"MODEL_EVERY_{period}" for period in PERIODS]
    + [f"RANDOM_EVERY_{period}" for period in PERIODS]
    + ["NOOP"]
)
THRESHOLDS = (0.15, 0.25, 0.35)
EVENT_POLICIES = ("THRESHOLD_015", "THRESHOLD_025", "THRESHOLD_035", "CONTINUOUS", "NOOP")
THRESHOLD_BY_POLICY = {
    "THRESHOLD_015": 0.15,
    "THRESHOLD_025": 0.25,
    "THRESHOLD_035": 0.35,
}
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
INHERITANCE_THRESHOLD = 0.9
DEPARTURE_THRESHOLD = 0.7
MINIMUM_FREE_DISK_BYTES = 2_500_000_000

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_cr9_feedback.py",
    "tests/test_intervention_cr9_feedback.py",
    "plastic_heredity/intervention_cr7_steering.py",
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
        f"codex-clean-room-cr9-feedback-v1::{name}".encode("utf-8")
    ).hexdigest()


SEEDS = {
    name: _seed(name)
    for name in (
        "validation",
        "smoke",
        "matrix_generation",
        "initial_composition",
        "main_trajectory",
        "landmark_restoration",
        "pulse_future",
        "periodic_future",
        "periodic_random_action",
        "event_future",
        "bootstrap",
        "randomization",
        "replay",
    )
}


@dataclass(frozen=True)
class PulseLineage:
    pulse_length: int
    replicate: int
    pulse_completed: bool
    pulse_observed_fissions: int
    release_completed: bool
    release_observed_fissions: int
    edits_applied: int
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
    actions: tuple[MolecularEdit, ...]
    action_steps: tuple[int, ...]
    action_risk_before: NDArray[np.float64]
    action_risk_after: NDArray[np.float64]
    simulation_rng_state: dict[str, Any]
    release_interventions_applied: int


@dataclass(frozen=True)
class PolicyLineage:
    phase: str
    policy: str
    replicate: int
    completed_horizon: bool
    observed_fissions: int
    inherited_observed_fraction: float
    inherited_fixed_horizon_fraction: float
    inherited_boundary_count: int
    total_breaks: int
    episode_count: int
    longest_inherited_run: int
    edits_applied: int
    edits_per_inherited_boundary: float
    threshold_excursions: int
    residual_excursions: int
    mean_risk_before: float
    mean_risk_after: float
    record_digest: str
    boundary_h: NDArray[np.float64]
    risk_before: NDArray[np.float64]
    risk_after: NDArray[np.float64]
    final_snapshot: Snapshot
    final_entropy: float
    final_top1_share: float
    final_occupied_types: int
    final_throughput: float
    mean_growth_updates: float
    actions: tuple[MolecularEdit, ...]
    action_steps: tuple[int, ...]
    simulation_rng_state: dict[str, Any]
    noop_plain_bitwise_exact: bool


@dataclass(frozen=True)
class PhaseBatch:
    format: str
    registration_id: str
    mode: str
    state_id: str
    candidate: str
    matrix_id: int
    case_digest: str
    lineages: tuple[PulseLineage | PolicyLineage, ...]


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_cr9_scientific_matrix",
        "upstream": {
            "cr7_registration_id": EXPECTED_CR7_REGISTRATION_ID,
            "cr7_complete_gate_required": True,
            "cr8_context_only_not_authorizing_or_tuning": True,
        },
        "cohort": {
            "fresh_matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "natural_launch_landmark": LANDMARK,
            "replicates_per_policy": REPLICATES,
            "main_path_only_uses_frozen_retry_contract": True,
            "policy_lineage_retry_or_replacement": False,
        },
        "pulse_ladder": {
            "pulse_lengths": list(PULSE_LENGTHS),
            "active_action": "frozen exhaustive MODEL_DOWN after every successful pulse fission",
            "written_anchor": "post-edit state at final pulse boundary",
            "untreated_release_fissions": HORIZON,
            "departure": "unrounded cosine < 0.7",
            "right_censor_cap": HORIZON + 1,
            "incomplete_before_crossing": "first unobserved boundary",
            "primary": "mean whole-matrix Spearman(pulse length, persistence)",
            "gate": "95% whole-matrix bootstrap lower bound > 0 in each candidate",
            "constant_persistence_vector_spearman": 0.0,
        },
        "periodic": {
            "periods": list(PERIODS),
            "policies": list(PERIODIC_POLICIES),
            "horizon": HORIZON,
            "action_boundary": "successful fissions K,2K,3K,...",
            "random_uniform_over_current_legal_swaps": True,
            "descriptive_minimum_interval": "largest K with CI95 lower > 0 versus matched random and NOOP",
        },
        "event_triggered": {
            "thresholds": list(THRESHOLDS),
            "policies": list(EVENT_POLICIES),
            "horizon": HORIZON,
            "trigger": "unedited frozen risk strictly greater than threshold",
            "action": "frozen exhaustive MODEL_DOWN",
            "no_threshold_selection_after_outcomes": True,
        },
        "outcomes": {
            "strict_inheritance": "unrounded H > 0.9",
            "fixed_horizon_adverse_fraction": "unobserved registered boundaries count as non-inherited",
            "state_metrics": ["risk", "entropy", "top1_share", "occupied_types", "xT_beta_x"],
            "process_metrics": ["breaks", "episodes", "longest_run", "growth_updates", "survival"],
        },
        "randomness": {
            "seed_domains": SEEDS,
            "future_seeds_exclude_policy": True,
            "random_action_stream_separate": True,
            "common_random_streams_not_identical_realized_futures": True,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "holm_within_periodic_and_event_families": True,
            "candidates_never_pooled": True,
        },
        "integrity": {
            "complete_replay_all_three_stages": True,
            "noop_callback_plain_bitwise_identity": True,
            "release_zero_interventions": True,
            "artifact_readback": True,
        },
        "stop_rule": "seal CR9 and stop before CR10",
        "claim_boundary": {
            "transient_hysteresis_not_restoring_basin": True,
            "sparse_feedback_not_passive_memory": True,
            "prohibited": [
                "installed compotype or autonomous attractor",
                "biological memory, agency, life, or error correction",
                "real prebiotic chemistry or universal origin-of-life mechanism",
                "strict-eight or Phi/PhiID control",
            ],
        },
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


def experiment() -> ExperimentConfig:
    cohort = CohortConfig(MATRICES, REPLICATES, (LANDMARK,))
    return ExperimentConfig(
        gard=GardConfig(),
        development=cohort,
        confirmation=cohort,
        horizon=HORIZON,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
        master_seed=SEEDS["matrix_generation"],
    )


def build_cr9_cohort(current_experiment: ExperimentConfig) -> list[StateCase]:
    cases: list[StateCase] = []
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
        for candidate, contract in CANDIDATES.items():
            lineage = None
            for attempt in range(100):
                rng = np.random.default_rng(
                    derive_seed(
                        SEEDS["main_trajectory"],
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
                    f"failed CR9 natural trajectory for candidate {candidate}, matrix {matrix_id}"
                )
            snapshot = {item.generation: item for item in lineage}[LANDMARK]
            cases.append(
                StateCase(
                    state_id=f"{LABEL}-c{candidate}-m{matrix_id:03d}-g{LANDMARK:03d}",
                    cohort=LABEL,
                    candidate=candidate,
                    matrix_id=matrix_id,
                    landmark=LANDMARK,
                    beta=beta,
                    snapshot=snapshot,
                )
            )
    return cases


def _case_digest(case: StateCase) -> str:
    digest = hashlib.sha256()
    digest.update(case.state_id.encode())
    digest.update(case.candidate.encode())
    digest.update(np.asarray((case.matrix_id, case.landmark), dtype=np.int64).tobytes())
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


def _entropy(composition: NDArray) -> float:
    values = np.asarray(composition, dtype=np.float64)
    mass = float(values.sum())
    if mass <= 0.0:
        return 0.0
    positive = values[values > 0.0] / mass
    return float(-np.dot(positive, np.log(positive)))


def _top1(composition: NDArray) -> float:
    values = np.asarray(composition, dtype=np.float64)
    mass = float(values.sum())
    return float(values.max() / mass) if mass > 0 else 0.0


def _throughput(composition: NDArray, beta: NDArray) -> float:
    values = np.asarray(composition, dtype=np.float64)
    mass = float(values.sum())
    if mass <= 0.0:
        return 0.0
    x = values / mass
    return float(x @ np.asarray(beta, dtype=np.float64) @ x)


def _snapshot_equal(left: Snapshot, right: Snapshot) -> bool:
    return bool(
        np.array_equal(left.composition, right.composition)
        and left.generation == right.generation
        and left.inheritance == right.inheritance
        and left.boundary_h == right.boundary_h
        and left.previous_growth_steps == right.previous_growth_steps
        and left.cumulative_growth_steps == right.cumulative_growth_steps
    )


def _rng_state_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(_json_ready(left), sort_keys=True) == json.dumps(
        _json_ready(right), sort_keys=True
    )


def _controlled_equal(
    left: ControlledResult,
    right: ControlledResult,
    left_rng: np.random.Generator,
    right_rng: np.random.Generator,
) -> bool:
    return bool(
        left.completed_horizon == right.completed_horizon
        and left.interventions_applied == right.interventions_applied == 0
        and left.selected_edits == right.selected_edits == ()
        and _records_digest(left.records) == _records_digest(right.records)
        and _snapshot_equal(left.final_snapshot, right.final_snapshot)
        and _rng_state_equal(left_rng.bit_generator.state, right_rng.bit_generator.state)
    )


def _post_fission_snapshots(
    launch: Snapshot, records: Iterable[Any]
) -> tuple[Snapshot, ...]:
    current = launch
    cumulative = launch.cumulative_growth_steps
    output: list[Snapshot] = []
    for record in records:
        cumulative += int(record.growth_steps)
        current = Snapshot(
            composition=np.asarray(record.daughter, dtype=np.int64).copy(),
            generation=current.generation + 1,
            inheritance=current.inheritance + (bool(record.h > INHERITANCE_THRESHOLD),),
            boundary_h=current.boundary_h + (float(record.h),),
            previous_growth_steps=int(record.growth_steps),
            cumulative_growth_steps=cumulative,
        )
        output.append(current)
    return tuple(output)


def count_nonoverlapping_episodes(boundary_h: NDArray) -> int:
    seeking_break = True
    trailing = 0
    episodes = 0
    for value in np.asarray(boundary_h, dtype=np.float64):
        inherited = bool(value > INHERITANCE_THRESHOLD)
        if seeking_break:
            if not inherited:
                seeking_break = False
                trailing = 0
            continue
        if inherited:
            trailing += 1
            if trailing == 3:
                episodes += 1
                seeking_break = True
                trailing = 0
        else:
            trailing = 0
    return episodes


def longest_inherited_run(boundary_h: NDArray) -> int:
    best = 0
    current = 0
    for value in np.asarray(boundary_h, dtype=np.float64):
        if value > INHERITANCE_THRESHOLD:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


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


class _Trace:
    def __init__(self) -> None:
        self.risk_before: list[float] = []
        self.risk_after: list[float] = []
        self.actions: list[MolecularEdit] = []
        self.action_steps: list[int] = []
        self.threshold_excursions = 0
        self.residual_excursions = 0


def _periodic_controller(
    policy: str,
    predictor: FrozenFullPredictor,
    config: GardConfig,
    action_rng: np.random.Generator,
) -> tuple[Callable[[Snapshot, NDArray, str, int], MolecularEdit | None], _Trace]:
    if policy not in PERIODIC_POLICIES:
        raise ValueError(f"unknown periodic policy: {policy}")
    trace = _Trace()

    def callback(
        snapshot: Snapshot, beta: NDArray, candidate: str, step: int
    ) -> MolecularEdit | None:
        before = predictor.predict_snapshot(candidate, snapshot, beta, config)
        after = before
        edit: MolecularEdit | None = None
        if policy != "NOOP":
            period = int(policy.rsplit("_", 1)[1])
            due = (step + 1) % period == 0
            if due and policy.startswith("MODEL_"):
                before, edit, after = _model_down(
                    predictor, candidate, snapshot, beta, config
                )
            elif due and policy.startswith("RANDOM_"):
                legal = enumerate_legal_edits(snapshot.composition)
                edit = legal[int(action_rng.integers(0, len(legal)))]
                after = predictor.predict_snapshot(
                    candidate, edited_snapshot(snapshot, edit), beta, config
                )
        trace.risk_before.append(float(before))
        trace.risk_after.append(float(after))
        if edit is not None:
            trace.actions.append(edit)
            trace.action_steps.append(step + 1)
        return edit

    return callback, trace


def _event_controller(
    policy: str,
    predictor: FrozenFullPredictor,
    config: GardConfig,
) -> tuple[Callable[[Snapshot, NDArray, str, int], MolecularEdit | None], _Trace]:
    if policy not in EVENT_POLICIES:
        raise ValueError(f"unknown event policy: {policy}")
    trace = _Trace()

    def callback(
        snapshot: Snapshot, beta: NDArray, candidate: str, step: int
    ) -> MolecularEdit | None:
        before = predictor.predict_snapshot(candidate, snapshot, beta, config)
        after = before
        edit: MolecularEdit | None = None
        if policy == "CONTINUOUS":
            before, edit, after = _model_down(
                predictor, candidate, snapshot, beta, config
            )
        elif policy in THRESHOLD_BY_POLICY:
            threshold = THRESHOLD_BY_POLICY[policy]
            if before > threshold:
                trace.threshold_excursions += 1
                before, edit, after = _model_down(
                    predictor, candidate, snapshot, beta, config
                )
                if after > threshold:
                    trace.residual_excursions += 1
        trace.risk_before.append(float(before))
        trace.risk_after.append(float(after))
        if edit is not None:
            trace.actions.append(edit)
            trace.action_steps.append(step + 1)
        return edit

    return callback, trace


def _pulse_future_seed(case: StateCase, replicate: int) -> int:
    return derive_seed(
        SEEDS["pulse_future"],
        f"{LABEL}.pulse.future",
        case.candidate,
        case.matrix_id,
        replicate,
    )


def _periodic_future_seed(case: StateCase, replicate: int) -> int:
    return derive_seed(
        SEEDS["periodic_future"],
        f"{LABEL}.periodic.future",
        case.candidate,
        case.matrix_id,
        replicate,
    )


def _periodic_action_seed(case: StateCase, replicate: int, policy: str) -> int:
    return derive_seed(
        SEEDS["periodic_random_action"],
        f"{LABEL}.periodic.action",
        case.candidate,
        case.matrix_id,
        replicate,
        policy,
    )


def _event_future_seed(case: StateCase, replicate: int) -> int:
    return derive_seed(
        SEEDS["event_future"],
        f"{LABEL}.event.future",
        case.candidate,
        case.matrix_id,
        replicate,
    )


def _empty_float(length: int) -> NDArray[np.float64]:
    return np.full(length, np.nan, dtype=np.float64)


def _pulse_lineage(
    case: StateCase,
    current_experiment: ExperimentConfig,
    predictor: FrozenFullPredictor,
    pulse_length: int,
    replicate: int,
    release_horizon: int,
) -> PulseLineage:
    rng = np.random.default_rng(_pulse_future_seed(case, replicate))
    trace = _Trace()

    def active_controller(
        snapshot: Snapshot, beta: NDArray, candidate: str, _step: int
    ) -> MolecularEdit:
        before, edit, after = _model_down(
            predictor, candidate, snapshot, beta, current_experiment.gard
        )
        trace.risk_before.append(before)
        trace.risk_after.append(after)
        trace.actions.append(edit)
        return edit

    pulse = simulate_controlled(
        case.snapshot,
        case.beta,
        case.candidate,
        current_experiment,
        pulse_length,
        rng,
        active_controller,
    )
    if tuple(trace.actions) != tuple(pulse.selected_edits):
        raise AssertionError("CR9 pulse controller trace differs from applied actions")
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
        release = ControlledResult((), False, anchor, 0, ())
    snapshots = _post_fission_snapshots(anchor, release.records)
    if snapshots and not _snapshot_equal(snapshots[-1], release.final_snapshot):
        raise AssertionError("CR9 pulse release snapshot reconstruction differs")
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
    if crossing.size:
        persistence = int(crossing[0]) + 1
    elif release.completed_horizon:
        persistence = release_horizon + 1
    else:
        persistence = min(len(release.records) + 1, release_horizon)
    if release.interventions_applied != 0 or release.selected_edits:
        raise AssertionError("CR9 pulse release applied an intervention")
    return PulseLineage(
        pulse_length=pulse_length,
        replicate=replicate,
        pulse_completed=bool(pulse.completed_horizon),
        pulse_observed_fissions=len(pulse.records),
        release_completed=bool(release.completed_horizon),
        release_observed_fissions=len(release.records),
        edits_applied=pulse.interventions_applied,
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
        actions=tuple(pulse.selected_edits),
        action_steps=tuple(range(1, pulse.interventions_applied + 1)),
        action_risk_before=np.asarray(trace.risk_before, dtype=np.float64),
        action_risk_after=np.asarray(trace.risk_after, dtype=np.float64),
        simulation_rng_state=_json_ready(rng.bit_generator.state),
        release_interventions_applied=release.interventions_applied,
    )


def _policy_summary(
    phase: str,
    policy: str,
    replicate: int,
    result: ControlledResult,
    trace: _Trace,
    beta: NDArray,
    horizon: int,
    rng_state: dict[str, Any],
    noop_exact: bool,
) -> PolicyLineage:
    observed = len(result.records)
    boundary_h = _empty_float(horizon)
    risk_before = _empty_float(horizon)
    risk_after = _empty_float(horizon)
    growth = np.full(horizon, -1, dtype=np.int32)
    for index, record in enumerate(result.records):
        boundary_h[index] = float(record.h)
        growth[index] = int(record.growth_steps)
    risk_before[: len(trace.risk_before)] = trace.risk_before
    risk_after[: len(trace.risk_after)] = trace.risk_after
    if len(trace.risk_before) != observed or len(trace.risk_after) != observed:
        raise AssertionError("CR9 controller trace length differs from observed fissions")
    if tuple(trace.actions) != tuple(result.selected_edits):
        raise AssertionError("CR9 controller trace actions differ from applied actions")
    finite_h = boundary_h[np.isfinite(boundary_h)]
    inherited_count = int(np.count_nonzero(finite_h > INHERITANCE_THRESHOLD))
    observed_fraction = float(inherited_count / observed) if observed else 0.0
    fixed_fraction = float(inherited_count / horizon)
    actions = tuple(result.selected_edits)
    composition = result.final_snapshot.composition
    return PolicyLineage(
        phase=phase,
        policy=policy,
        replicate=replicate,
        completed_horizon=bool(result.completed_horizon),
        observed_fissions=observed,
        inherited_observed_fraction=observed_fraction,
        inherited_fixed_horizon_fraction=fixed_fraction,
        inherited_boundary_count=inherited_count,
        total_breaks=int(observed - inherited_count),
        episode_count=count_nonoverlapping_episodes(finite_h),
        longest_inherited_run=longest_inherited_run(finite_h),
        edits_applied=len(actions),
        edits_per_inherited_boundary=(
            float(len(actions) / inherited_count) if inherited_count else float("nan")
        ),
        threshold_excursions=trace.threshold_excursions,
        residual_excursions=trace.residual_excursions,
        mean_risk_before=(float(np.mean(trace.risk_before)) if observed else float("nan")),
        mean_risk_after=(float(np.mean(trace.risk_after)) if observed else float("nan")),
        record_digest=_records_digest(result.records),
        boundary_h=boundary_h,
        risk_before=risk_before,
        risk_after=risk_after,
        final_snapshot=result.final_snapshot,
        final_entropy=_entropy(composition),
        final_top1_share=_top1(composition),
        final_occupied_types=int(np.count_nonzero(composition)),
        final_throughput=_throughput(composition, beta),
        mean_growth_updates=(float(growth[:observed].mean()) if observed else float("nan")),
        actions=actions,
        action_steps=tuple(trace.action_steps),
        simulation_rng_state=_json_ready(rng_state),
        noop_plain_bitwise_exact=bool(noop_exact),
    )


def _run_pulse_case(
    case: StateCase,
    current_experiment: ExperimentConfig,
    model_path: str | Path,
    registration_id: str,
    *,
    pulse_lengths: tuple[int, ...] = PULSE_LENGTHS,
    replicates: int = REPLICATES,
    release_horizon: int = HORIZON,
) -> PhaseBatch:
    predictor = FrozenFullPredictor.load(model_path)
    lineages = tuple(
        _pulse_lineage(
            case,
            current_experiment,
            predictor,
            pulse_length,
            replicate,
            release_horizon,
        )
        for replicate in range(replicates)
        for pulse_length in pulse_lengths
    )
    return PhaseBatch(
        format=CHECKPOINT_FORMAT,
        registration_id=registration_id,
        mode="pulse",
        state_id=case.state_id,
        candidate=case.candidate,
        matrix_id=case.matrix_id,
        case_digest=_case_digest(case),
        lineages=lineages,
    )


def _run_periodic_case(
    case: StateCase,
    current_experiment: ExperimentConfig,
    model_path: str | Path,
    registration_id: str,
    *,
    policies: tuple[str, ...] = PERIODIC_POLICIES,
    replicates: int = REPLICATES,
    horizon: int = HORIZON,
) -> PhaseBatch:
    predictor = FrozenFullPredictor.load(model_path)
    lineages: list[PolicyLineage] = []
    for replicate in range(replicates):
        simulation_seed = _periodic_future_seed(case, replicate)
        for policy in policies:
            rng = np.random.default_rng(simulation_seed)
            action_rng = np.random.default_rng(
                _periodic_action_seed(case, replicate, policy)
            )
            callback, trace = _periodic_controller(
                policy, predictor, current_experiment.gard, action_rng
            )
            result = simulate_controlled(
                case.snapshot,
                case.beta,
                case.candidate,
                current_experiment,
                horizon,
                rng,
                callback,
            )
            noop_exact = True
            if policy == "NOOP":
                plain_rng = np.random.default_rng(simulation_seed)
                plain = simulate_controlled(
                    case.snapshot,
                    case.beta,
                    case.candidate,
                    current_experiment,
                    horizon,
                    plain_rng,
                    None,
                )
                noop_exact = _controlled_equal(result, plain, rng, plain_rng)
            lineages.append(
                _policy_summary(
                    "periodic",
                    policy,
                    replicate,
                    result,
                    trace,
                    case.beta,
                    horizon,
                    rng.bit_generator.state,
                    noop_exact,
                )
            )
    return PhaseBatch(
        format=CHECKPOINT_FORMAT,
        registration_id=registration_id,
        mode="periodic",
        state_id=case.state_id,
        candidate=case.candidate,
        matrix_id=case.matrix_id,
        case_digest=_case_digest(case),
        lineages=tuple(lineages),
    )


def _run_event_case(
    case: StateCase,
    current_experiment: ExperimentConfig,
    model_path: str | Path,
    registration_id: str,
    *,
    policies: tuple[str, ...] = EVENT_POLICIES,
    replicates: int = REPLICATES,
    horizon: int = HORIZON,
) -> PhaseBatch:
    predictor = FrozenFullPredictor.load(model_path)
    lineages: list[PolicyLineage] = []
    for replicate in range(replicates):
        simulation_seed = _event_future_seed(case, replicate)
        for policy in policies:
            rng = np.random.default_rng(simulation_seed)
            callback, trace = _event_controller(
                policy, predictor, current_experiment.gard
            )
            result = simulate_controlled(
                case.snapshot,
                case.beta,
                case.candidate,
                current_experiment,
                horizon,
                rng,
                callback,
            )
            noop_exact = True
            if policy == "NOOP":
                plain_rng = np.random.default_rng(simulation_seed)
                plain = simulate_controlled(
                    case.snapshot,
                    case.beta,
                    case.candidate,
                    current_experiment,
                    horizon,
                    plain_rng,
                    None,
                )
                noop_exact = _controlled_equal(result, plain, rng, plain_rng)
            lineages.append(
                _policy_summary(
                    "event",
                    policy,
                    replicate,
                    result,
                    trace,
                    case.beta,
                    horizon,
                    rng.bit_generator.state,
                    noop_exact,
                )
            )
    return PhaseBatch(
        format=CHECKPOINT_FORMAT,
        registration_id=registration_id,
        mode="event",
        state_id=case.state_id,
        candidate=case.candidate,
        matrix_id=case.matrix_id,
        case_digest=_case_digest(case),
        lineages=tuple(lineages),
    )


def _snapshot_digest(snapshot: Snapshot) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(snapshot.composition, dtype=np.int64).tobytes())
    digest.update(
        np.asarray(
            (
                snapshot.generation,
                snapshot.previous_growth_steps,
                snapshot.cumulative_growth_steps,
            ),
            dtype=np.int64,
        ).tobytes()
    )
    digest.update(np.asarray(snapshot.boundary_h, dtype=np.float64).tobytes())
    digest.update(np.asarray(snapshot.inheritance, dtype=np.int8).tobytes())
    return digest.hexdigest()


def _edit_array(actions: tuple[MolecularEdit, ...]) -> NDArray[np.int16]:
    return np.asarray(
        [(item.remove_type, item.add_type) for item in actions], dtype=np.int16
    ).reshape(-1, 2)


def _lineage_digest(lineage: PulseLineage | PolicyLineage) -> str:
    digest = hashlib.sha256()
    digest.update(type(lineage).__name__.encode())
    if isinstance(lineage, PulseLineage):
        digest.update(
            np.asarray(
                (
                    lineage.pulse_length,
                    lineage.replicate,
                    lineage.pulse_completed,
                    lineage.pulse_observed_fissions,
                    lineage.release_completed,
                    lineage.release_observed_fissions,
                    lineage.edits_applied,
                    lineage.persistence,
                    lineage.first_departure_time,
                    lineage.release_interventions_applied,
                ),
                dtype=np.int64,
            ).tobytes()
        )
        digest.update(lineage.pulse_record_digest.encode())
        digest.update(lineage.release_record_digest.encode())
        arrays = (
            lineage.anchor_composition,
            lineage.similarity_to_anchor,
            lineage.risk,
            lineage.boundary_h,
            lineage.growth_updates,
            lineage.entropy,
            lineage.top1_share,
            lineage.occupied_types,
            lineage.throughput,
            lineage.action_risk_before,
            lineage.action_risk_after,
        )
    else:
        digest.update(lineage.phase.encode())
        digest.update(lineage.policy.encode())
        digest.update(
            np.asarray(
                (
                    lineage.replicate,
                    lineage.completed_horizon,
                    lineage.observed_fissions,
                    lineage.inherited_observed_fraction,
                    lineage.inherited_fixed_horizon_fraction,
                    lineage.inherited_boundary_count,
                    lineage.total_breaks,
                    lineage.episode_count,
                    lineage.longest_inherited_run,
                    lineage.edits_applied,
                    lineage.edits_per_inherited_boundary,
                    lineage.threshold_excursions,
                    lineage.residual_excursions,
                    lineage.mean_risk_before,
                    lineage.mean_risk_after,
                    lineage.final_entropy,
                    lineage.final_top1_share,
                    lineage.final_occupied_types,
                    lineage.final_throughput,
                    lineage.mean_growth_updates,
                    lineage.noop_plain_bitwise_exact,
                ),
                dtype=np.float64,
            ).tobytes()
        )
        digest.update(lineage.record_digest.encode())
        arrays = (lineage.boundary_h, lineage.risk_before, lineage.risk_after)
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(_edit_array(lineage.actions).tobytes())
    digest.update(np.asarray(lineage.action_steps, dtype=np.int16).tobytes())
    digest.update(_snapshot_digest(lineage.final_snapshot).encode())
    digest.update(
        json.dumps(_json_ready(lineage.simulation_rng_state), sort_keys=True).encode()
    )
    return digest.hexdigest()


def batch_digest(batch: PhaseBatch) -> str:
    digest = hashlib.sha256()
    for value in (
        batch.format,
        batch.registration_id,
        batch.mode,
        batch.state_id,
        batch.candidate,
        str(batch.matrix_id),
        batch.case_digest,
    ):
        digest.update(value.encode())
    for lineage in batch.lineages:
        digest.update(_lineage_digest(lineage).encode())
    return digest.hexdigest()


def replay_audit(
    generated: list[PhaseBatch], replayed: list[PhaseBatch], mode: str
) -> dict[str, Any]:
    if len(generated) != len(replayed):
        raise ValueError(f"CR9 {mode} replay batch count differs")
    rows: list[dict[str, Any]] = []
    for left, right in zip(generated, replayed, strict=True):
        left_digest = batch_digest(left)
        right_digest = batch_digest(right)
        rows.append(
            {
                "state_id": left.state_id,
                "candidate": left.candidate,
                "matrix_id": left.matrix_id,
                "generated_digest": left_digest,
                "replay_digest": right_digest,
                "exact": left_digest == right_digest,
            }
        )
    return {
        "format": f"codex-intervention-cr9-{mode}-replay-audit-v1",
        "state_batches": len(rows),
        "exact_state_action_endpoint_process_and_rng": bool(
            all(row["exact"] for row in rows)
        ),
        "rows": rows,
    }


def _checkpoint_path(directory: Path, case: StateCase) -> Path:
    return directory / f"c{case.candidate}_m{case.matrix_id:03d}.pkl"


def _write_checkpoint(path: Path, batch: PhaseBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(batch, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _expected_lineage_count(mode: str) -> int:
    if mode == "pulse":
        return REPLICATES * len(PULSE_LENGTHS)
    if mode == "periodic":
        return REPLICATES * len(PERIODIC_POLICIES)
    if mode == "event":
        return REPLICATES * len(EVENT_POLICIES)
    raise ValueError(f"unknown CR9 mode: {mode}")


def _read_checkpoint(
    path: Path, case: StateCase, registration_id: str, mode: str
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
        mode,
        case.state_id,
        case.candidate,
        case.matrix_id,
        _case_digest(case),
        _expected_lineage_count(mode),
    )
    observed = (
        value.format,
        value.registration_id,
        value.mode,
        value.state_id,
        value.candidate,
        value.matrix_id,
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
    mode, rest = arguments[0], arguments[1:]
    limiter = threadpool_limits(limits=1)
    try:
        if mode == "pulse":
            return _run_pulse_case(*rest)
        if mode == "periodic":
            return _run_periodic_case(*rest)
        if mode == "event":
            return _run_event_case(*rest)
        raise ValueError(f"unknown CR9 worker mode: {mode}")
    finally:
        limiter.restore_original_limits()


def run_phase_batches(
    mode: str,
    cases: list[StateCase],
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
    missing: list[StateCase] = []
    for case in cases:
        checkpoint = _read_checkpoint(
            _checkpoint_path(directory, case), case, registration_id, mode
        )
        if checkpoint is None:
            missing.append(case)
        else:
            batches[case.state_id] = checkpoint
    reused = len(batches)
    _write_status(work, stage, reused, len(cases), reused=reused)
    arguments = [
        (mode, case, current_experiment, model_path, registration_id)
        for case in missing
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
        raise AssertionError(f"CR9 {mode} checkpoint cohort is incomplete")
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
    return {"bootstrap_indices": bootstrap, "randomization_signs": 2.0 * signs - 1.0}


def _interval(values: NDArray, alpha: float = 0.05) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan"), float("nan")
    lower, upper = np.quantile(finite, (alpha / 2.0, 1.0 - alpha / 2.0))
    return float(lower), float(upper)


def _maximum_leave_one_out_influence(values: NDArray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return float("nan")
    estimate = float(array.mean())
    leave_one = (array.sum() - array) / (array.size - 1)
    return float(np.max(np.abs(leave_one - estimate)))


def _positive_sign_p(values: NDArray, signs: NDArray) -> tuple[float, NDArray]:
    array = np.asarray(values, dtype=np.float64)
    observed = float(array.mean())
    null = np.asarray(signs @ array / array.size, dtype=np.float64)
    p_value = float((np.count_nonzero(null >= observed) + 1) / (len(null) + 1))
    return p_value, null


def _holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return [float(item) for item in adjusted]


def spearman_constant_zero(x: NDArray, y: NDArray) -> float:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.size != right.size or left.size < 2:
        raise ValueError("Spearman inputs require equal lengths of at least two")
    if np.all(left == left[0]) or np.all(right == right[0]):
        return 0.0
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def pulse_tables(
    cases: list[StateCase], batches: list[PhaseBatch]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, NDArray]]:
    rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    packed: dict[str, list[NDArray]] = {
        name: []
        for name in (
            "similarity_to_anchor",
            "risk",
            "boundary_h",
            "growth_updates",
            "entropy",
            "top1_share",
            "occupied_types",
            "throughput",
            "anchor_composition",
            "final_composition",
        )
    }
    for case, batch in zip(cases, batches, strict=True):
        if batch.mode != "pulse" or batch.case_digest != _case_digest(case):
            raise ValueError("CR9 pulse batch no longer matches its launch state")
        for lineage in batch.lineages:
            if not isinstance(lineage, PulseLineage):
                raise TypeError("CR9 pulse batch contains another lineage type")
            finite_h = lineage.boundary_h[np.isfinite(lineage.boundary_h)]
            finite_similarity = lineage.similarity_to_anchor[
                np.isfinite(lineage.similarity_to_anchor)
            ]
            finite_risk = lineage.risk[np.isfinite(lineage.risk)]
            row_index = len(rows)
            rows.append(
                {
                    "row_index": row_index,
                    "state_id": case.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "landmark": case.landmark,
                    "pulse_length": lineage.pulse_length,
                    "replicate": lineage.replicate,
                    "pulse_completed": int(lineage.pulse_completed),
                    "pulse_observed_fissions": lineage.pulse_observed_fissions,
                    "release_completed": int(lineage.release_completed),
                    "release_observed_fissions": lineage.release_observed_fissions,
                    "edits_applied": lineage.edits_applied,
                    "release_interventions_applied": lineage.release_interventions_applied,
                    "persistence": lineage.persistence,
                    "first_departure_time": lineage.first_departure_time,
                    "release_inherited_fixed_fraction": float(
                        np.count_nonzero(finite_h > INHERITANCE_THRESHOLD) / HORIZON
                    ),
                    "release_break_count": int(
                        len(finite_h)
                        - np.count_nonzero(finite_h > INHERITANCE_THRESHOLD)
                    ),
                    "final_similarity": (
                        float(finite_similarity[-1]) if finite_similarity.size else float("nan")
                    ),
                    "minimum_similarity": (
                        float(finite_similarity.min()) if finite_similarity.size else float("nan")
                    ),
                    "final_risk": float(finite_risk[-1]) if finite_risk.size else float("nan"),
                    "final_entropy": _entropy(lineage.final_snapshot.composition),
                    "final_top1_share": _top1(lineage.final_snapshot.composition),
                    "final_occupied_types": int(
                        np.count_nonzero(lineage.final_snapshot.composition)
                    ),
                    "final_throughput": _throughput(
                        lineage.final_snapshot.composition, case.beta
                    ),
                    "pulse_record_digest": lineage.pulse_record_digest,
                    "release_record_digest": lineage.release_record_digest,
                    "anchor_composition_digest": hashlib.sha256(
                        np.ascontiguousarray(lineage.anchor_composition).tobytes()
                    ).hexdigest(),
                    "final_composition_digest": hashlib.sha256(
                        np.ascontiguousarray(lineage.final_snapshot.composition).tobytes()
                    ).hexdigest(),
                }
            )
            for action_number, (step, edit, before, after) in enumerate(
                zip(
                    lineage.action_steps,
                    lineage.actions,
                    lineage.action_risk_before,
                    lineage.action_risk_after,
                    strict=True,
                ),
                start=1,
            ):
                edit_rows.append(
                    {
                        "phase": "pulse",
                        "state_id": case.state_id,
                        "candidate": case.candidate,
                        "matrix_id": case.matrix_id,
                        "pulse_length": lineage.pulse_length,
                        "replicate": lineage.replicate,
                        "action_number": action_number,
                        "fission_step": step,
                        "remove_type": edit.remove_type,
                        "add_type": edit.add_type,
                        "risk_before": float(before),
                        "risk_after": float(after),
                    }
                )
            for name in packed:
                if name == "final_composition":
                    value = lineage.final_snapshot.composition
                else:
                    value = getattr(lineage, name)
                packed[name].append(np.asarray(value))
    lineage_table = pd.DataFrame(rows)
    matrix_table = (
        lineage_table.groupby(
            ["candidate", "matrix_id", "pulse_length"], as_index=False
        ).mean(numeric_only=True)
    )
    counts = lineage_table.groupby(["candidate", "matrix_id", "pulse_length"]).size()
    if not bool((counts == REPLICATES).all()):
        raise ValueError("CR9 pulse whole-matrix block lost a replicate")
    arrays = {name: np.stack(values) for name, values in packed.items()}
    arrays.update(
        {
            "candidate": lineage_table["candidate"].to_numpy(dtype="<U2"),
            "matrix_id": lineage_table["matrix_id"].to_numpy(dtype=np.int16),
            "pulse_length": lineage_table["pulse_length"].to_numpy(dtype=np.int8),
            "replicate": lineage_table["replicate"].to_numpy(dtype=np.int8),
        }
    )
    return lineage_table, matrix_table, pd.DataFrame(edit_rows), arrays


def policy_tables(
    phase: str, cases: list[StateCase], batches: list[PhaseBatch]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, NDArray]]:
    if phase not in ("periodic", "event"):
        raise ValueError("CR9 policy table phase must be periodic or event")
    rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    boundary_arrays: list[NDArray] = []
    before_arrays: list[NDArray] = []
    after_arrays: list[NDArray] = []
    final_arrays: list[NDArray] = []
    for case, batch in zip(cases, batches, strict=True):
        if batch.mode != phase or batch.case_digest != _case_digest(case):
            raise ValueError(f"CR9 {phase} batch no longer matches its launch state")
        for lineage in batch.lineages:
            if not isinstance(lineage, PolicyLineage) or lineage.phase != phase:
                raise TypeError(f"CR9 {phase} batch contains another lineage type")
            row_index = len(rows)
            rows.append(
                {
                    "row_index": row_index,
                    "state_id": case.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "landmark": case.landmark,
                    "policy": lineage.policy,
                    "replicate": lineage.replicate,
                    "completed_horizon": int(lineage.completed_horizon),
                    "observed_fissions": lineage.observed_fissions,
                    "inherited_observed_fraction": lineage.inherited_observed_fraction,
                    "inherited_fixed_horizon_fraction": lineage.inherited_fixed_horizon_fraction,
                    "inherited_boundary_count": lineage.inherited_boundary_count,
                    "total_breaks": lineage.total_breaks,
                    "episode_count": lineage.episode_count,
                    "longest_inherited_run": lineage.longest_inherited_run,
                    "edits_applied": lineage.edits_applied,
                    "edits_per_inherited_boundary": lineage.edits_per_inherited_boundary,
                    "threshold_excursions": lineage.threshold_excursions,
                    "residual_excursions": lineage.residual_excursions,
                    "mean_risk_before": lineage.mean_risk_before,
                    "mean_risk_after": lineage.mean_risk_after,
                    "final_entropy": lineage.final_entropy,
                    "final_top1_share": lineage.final_top1_share,
                    "final_occupied_types": lineage.final_occupied_types,
                    "final_throughput": lineage.final_throughput,
                    "mean_growth_updates": lineage.mean_growth_updates,
                    "record_digest": lineage.record_digest,
                    "noop_plain_bitwise_exact": int(lineage.noop_plain_bitwise_exact),
                    "final_composition_digest": hashlib.sha256(
                        np.ascontiguousarray(lineage.final_snapshot.composition).tobytes()
                    ).hexdigest(),
                }
            )
            for action_number, (step, edit) in enumerate(
                zip(lineage.action_steps, lineage.actions, strict=True), start=1
            ):
                edit_rows.append(
                    {
                        "phase": phase,
                        "state_id": case.state_id,
                        "candidate": case.candidate,
                        "matrix_id": case.matrix_id,
                        "policy": lineage.policy,
                        "replicate": lineage.replicate,
                        "action_number": action_number,
                        "fission_step": step,
                        "remove_type": edit.remove_type,
                        "add_type": edit.add_type,
                        "risk_before": float(lineage.risk_before[step - 1]),
                        "risk_after": float(lineage.risk_after[step - 1]),
                    }
                )
            boundary_arrays.append(lineage.boundary_h)
            before_arrays.append(lineage.risk_before)
            after_arrays.append(lineage.risk_after)
            final_arrays.append(lineage.final_snapshot.composition)
    lineage_table = pd.DataFrame(rows)
    counts = lineage_table.groupby(["candidate", "matrix_id", "policy"]).size()
    if not bool((counts == REPLICATES).all()):
        raise ValueError(f"CR9 {phase} whole-matrix block lost a replicate")
    matrix_table = (
        lineage_table.groupby(["candidate", "matrix_id", "policy"], as_index=False)
        .mean(numeric_only=True)
    )
    arrays = {
        "boundary_h": np.stack(boundary_arrays),
        "risk_before": np.stack(before_arrays),
        "risk_after": np.stack(after_arrays),
        "final_composition": np.stack(final_arrays),
        "candidate": lineage_table["candidate"].to_numpy(dtype="<U2"),
        "matrix_id": lineage_table["matrix_id"].to_numpy(dtype=np.int16),
        "policy": lineage_table["policy"].to_numpy(dtype="<U20"),
        "replicate": lineage_table["replicate"].to_numpy(dtype=np.int8),
    }
    return lineage_table, matrix_table, pd.DataFrame(edit_rows), arrays


def _contrast_summary(
    values: NDArray,
    bootstrap: NDArray,
    signs: NDArray,
) -> tuple[dict[str, Any], NDArray, NDArray]:
    array = np.asarray(values, dtype=np.float64)
    boot = array[bootstrap].mean(axis=1)
    raw_p, null = _positive_sign_p(array, signs)
    return (
        {
            "estimate": float(array.mean()),
            "bootstrap_ci95": _interval(boot),
            "bootstrap_ci90": _interval(boot, alpha=0.10),
            "positive_one_sided_randomization_p_raw": raw_p,
            "matrices_positive": int(np.count_nonzero(array > 0)),
            "matrices_negative": int(np.count_nonzero(array < 0)),
            "matrices_zero": int(np.count_nonzero(array == 0)),
            "maximum_leave_one_matrix_out_influence": _maximum_leave_one_out_influence(
                array
            ),
        },
        boot,
        null,
    )


def _policy_pivot(table: pd.DataFrame, candidate: str, outcome: str) -> pd.DataFrame:
    selected = table[table["candidate"].astype(str).str.zfill(2) == candidate]
    pivot = selected.pivot(index="matrix_id", columns="policy", values=outcome)
    return pivot.reindex(index=np.arange(MATRICES))


def compute_inference(
    pulse_matrix: pd.DataFrame,
    periodic_matrix: pd.DataFrame,
    event_matrix: pd.DataFrame,
    draws: dict[str, NDArray],
    *,
    pulse_replay_exact: bool,
    periodic_replay_exact: bool,
    event_replay_exact: bool,
    noop_plain_exact: bool,
    release_zero_interventions: bool,
    readback_exact: bool = True,
) -> tuple[dict[str, Any], dict[str, NDArray]]:
    bootstrap = np.asarray(draws["bootstrap_indices"], dtype=np.int64)
    signs = np.asarray(draws["randomization_signs"], dtype=np.float64)
    if bootstrap.shape != (BOOTSTRAP_REPETITIONS, MATRICES):
        raise ValueError("CR9 bootstrap lost whole-matrix blocks")
    if signs.shape != (RANDOMIZATION_REPETITIONS, MATRICES):
        raise ValueError("CR9 randomization lost whole-matrix blocks")
    stored: dict[str, NDArray] = {
        "bootstrap_indices": bootstrap,
        "randomization_signs": signs,
    }
    pulse_candidates: list[dict[str, Any]] = []
    periodic_candidates: list[dict[str, Any]] = []
    event_candidates: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        selected_pulse = pulse_matrix[
            pulse_matrix["candidate"].astype(str).str.zfill(2) == candidate
        ]
        persistence = selected_pulse.pivot(
            index="matrix_id", columns="pulse_length", values="persistence"
        ).reindex(index=np.arange(MATRICES), columns=PULSE_LENGTHS)
        if persistence.isna().any().any():
            raise ValueError(f"candidate {candidate} lacks a complete pulse block")
        matrix_rho = np.asarray(
            [
                spearman_constant_zero(
                    np.asarray(PULSE_LENGTHS, dtype=np.float64),
                    row.to_numpy(dtype=np.float64),
                )
                for _, row in persistence.iterrows()
            ],
            dtype=np.float64,
        )
        rho_summary, rho_boot, rho_null = _contrast_summary(
            matrix_rho, bootstrap, signs
        )
        rho_summary["primary_positive_lower_bound"] = bool(
            rho_summary["bootstrap_ci95"][0] > 0.0
        )
        stored[f"pulse_c{candidate}_matrix_spearman"] = matrix_rho
        stored[f"pulse_c{candidate}_bootstrap_spearman"] = rho_boot
        stored[f"pulse_c{candidate}_randomization_spearman"] = rho_null
        pulse_lengths: list[dict[str, Any]] = []
        for length in PULSE_LENGTHS:
            values = persistence[length].to_numpy(dtype=np.float64)
            boot = values[bootstrap].mean(axis=1)
            pulse_lengths.append(
                {
                    "pulse_length": length,
                    "mean_persistence": float(values.mean()),
                    "bootstrap_ci95": _interval(boot),
                    "completed_release_fraction": float(
                        selected_pulse[selected_pulse["pulse_length"] == length][
                            "release_completed"
                        ].mean()
                    ),
                    "mean_final_similarity": float(
                        selected_pulse[selected_pulse["pulse_length"] == length][
                            "final_similarity"
                        ].mean()
                    ),
                }
            )
            stored[f"pulse_c{candidate}_L{length}_matrix_persistence"] = values
            stored[f"pulse_c{candidate}_L{length}_bootstrap_persistence"] = boot
        pulse_candidates.append(
            {
                "candidate": candidate,
                "matrix_spearman": rho_summary,
                "pulse_lengths": pulse_lengths,
                "candidate_primary_gate": rho_summary["primary_positive_lower_bound"],
            }
        )

        periodic_inheritance = _policy_pivot(
            periodic_matrix, candidate, "inherited_fixed_horizon_fraction"
        )
        periodic_edits = _policy_pivot(periodic_matrix, candidate, "edits_applied")
        if periodic_inheritance.reindex(columns=PERIODIC_POLICIES).isna().any().any():
            raise ValueError(f"candidate {candidate} lacks a complete periodic block")
        periodic_results: list[dict[str, Any]] = []
        p_locations: list[dict[str, Any]] = []
        p_values: list[float] = []
        for period in PERIODS:
            model = f"MODEL_EVERY_{period}"
            random = f"RANDOM_EVERY_{period}"
            item: dict[str, Any] = {
                "period": period,
                "model_mean_inheritance": float(periodic_inheritance[model].mean()),
                "random_mean_inheritance": float(periodic_inheritance[random].mean()),
                "noop_mean_inheritance": float(periodic_inheritance["NOOP"].mean()),
                "model_mean_edits": float(periodic_edits[model].mean()),
                "random_mean_edits": float(periodic_edits[random].mean()),
            }
            for label, control in (("minus_random", random), ("minus_noop", "NOOP")):
                values = periodic_inheritance[model].to_numpy() - periodic_inheritance[
                    control
                ].to_numpy()
                summary, boot, null = _contrast_summary(values, bootstrap, signs)
                item[label] = summary
                p_locations.append(summary)
                p_values.append(summary["positive_one_sided_randomization_p_raw"])
                stored[f"periodic_c{candidate}_K{period}_{label}_matrix"] = values
                stored[f"periodic_c{candidate}_K{period}_{label}_bootstrap"] = boot
                stored[f"periodic_c{candidate}_K{period}_{label}_randomization"] = null
            periodic_results.append(item)
        for location, adjusted in zip(p_locations, _holm_adjust(p_values), strict=True):
            location["holm_adjusted_p"] = adjusted
        qualifying = [
            item["period"]
            for item in periodic_results
            if item["minus_random"]["bootstrap_ci95"][0] > 0.0
            and item["minus_noop"]["bootstrap_ci95"][0] > 0.0
        ]
        periodic_candidates.append(
            {
                "candidate": candidate,
                "periods": periodic_results,
                "descriptive_minimum_feedback_interval": max(qualifying, default=0),
            }
        )

        event_inheritance = _policy_pivot(
            event_matrix, candidate, "inherited_fixed_horizon_fraction"
        )
        event_edits = _policy_pivot(event_matrix, candidate, "edits_applied")
        if event_inheritance.reindex(columns=EVENT_POLICIES).isna().any().any():
            raise ValueError(f"candidate {candidate} lacks a complete event block")
        continuous_effect = (
            event_inheritance["CONTINUOUS"].to_numpy()
            - event_inheritance["NOOP"].to_numpy()
        )
        continuous_boot = continuous_effect[bootstrap].mean(axis=1)
        event_results: list[dict[str, Any]] = []
        event_p_locations: list[dict[str, Any]] = []
        event_p_values: list[float] = []
        for policy in EVENT_POLICIES[:-1]:
            values = event_inheritance[policy].to_numpy() - event_inheritance[
                "NOOP"
            ].to_numpy()
            summary, boot, null = _contrast_summary(values, bootstrap, signs)
            event_p_locations.append(summary)
            event_p_values.append(summary["positive_one_sided_randomization_p_raw"])
            ratio_boot = np.divide(
                boot,
                continuous_boot,
                out=np.full_like(boot, np.nan),
                where=np.abs(continuous_boot) > 1e-12,
            )
            ratio_estimate = (
                float(values.mean() / continuous_effect.mean())
                if abs(float(continuous_effect.mean())) > 1e-12
                else float("nan")
            )
            item = {
                "policy": policy,
                "inheritance_mean": float(event_inheritance[policy].mean()),
                "mean_edits": float(event_edits[policy].mean()),
                "inheritance_minus_noop": summary,
                "edit_savings_vs_continuous": float(
                    event_edits["CONTINUOUS"].mean() - event_edits[policy].mean()
                ),
                "fraction_continuous_gain_recovered": {
                    "estimate": ratio_estimate,
                    "bootstrap_ci95": _interval(ratio_boot),
                },
            }
            event_results.append(item)
            stored[f"event_c{candidate}_{policy}_matrix_effect"] = values
            stored[f"event_c{candidate}_{policy}_bootstrap_effect"] = boot
            stored[f"event_c{candidate}_{policy}_randomization"] = null
            stored[f"event_c{candidate}_{policy}_bootstrap_recovery_fraction"] = ratio_boot
        for location, adjusted in zip(
            event_p_locations, _holm_adjust(event_p_values), strict=True
        ):
            location["holm_adjusted_p"] = adjusted
        event_candidates.append(
            {
                "candidate": candidate,
                "noop_mean_inheritance": float(event_inheritance["NOOP"].mean()),
                "continuous_mean_inheritance": float(
                    event_inheritance["CONTINUOUS"].mean()
                ),
                "continuous_mean_edits": float(event_edits["CONTINUOUS"].mean()),
                "policies": event_results,
            }
        )

    integrity = {
        "pulse_exact_replay": bool(pulse_replay_exact),
        "periodic_exact_replay": bool(periodic_replay_exact),
        "event_exact_replay": bool(event_replay_exact),
        "noop_callback_plain_bitwise_exact": bool(noop_plain_exact),
        "release_interventions_exactly_zero": bool(release_zero_interventions),
        "artifact_readback_exact": bool(readback_exact),
    }
    integrity["complete_integrity"] = bool(all(integrity.values()))
    efficacy = bool(all(item["candidate_primary_gate"] for item in pulse_candidates))
    return (
        {
            "format": "codex-intervention-cr9-primary-metrics-v1",
            "pulse": {
                "candidates": pulse_candidates,
                "complete_two_candidate_hysteresis_gate": efficacy,
            },
            "periodic": {"candidates": periodic_candidates},
            "event_triggered": {"candidates": event_candidates},
            "integrity": integrity,
            "complete_cr9_registered_gate": bool(efficacy and integrity["complete_integrity"]),
            "periodic_and_event_results_do_not_rescue_pulse_gate": True,
        },
        stored,
    )


def _prior_seed_values() -> set[str]:
    values: set[str] = set()
    for path in RESULT_ROOT.glob("*registration*/registration.json"):
        if path.parent == DEFAULT_REGISTRATION:
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        for key in ("seed_registry", "seeds"):
            registry = payload.get(key, {})
            if isinstance(registry, dict):
                values.update(str(item) for item in registry.values())
    return values


def _verify_upstream() -> dict[str, Any]:
    for directory in (CR0_VALIDATION, CR7_REGISTRATION, CR7_RESULT, CR8_RESULT):
        verify_checksums(directory)
    cr0 = json.loads((CR0_VALIDATION / "validation.json").read_text())
    cr7_registration = json.loads((CR7_REGISTRATION / "registration.json").read_text())
    cr7_result = json.loads((CR7_RESULT / "manifest.json").read_text())
    cr8_result = json.loads((CR8_RESULT / "manifest.json").read_text())
    if not cr0["all_checks_passed"]:
        raise ValueError("CR0 validation is no longer passing")
    if cr7_registration["registration_id"] != EXPECTED_CR7_REGISTRATION_ID:
        raise ValueError("CR7 registration ID changed")
    if not (
        cr7_result["complete_cr7_60_fission_gate"]
        and cr7_result["exact_replay"]
        and cr7_result["complete_readback_exact"]
        and cr7_result["noop_callback_plain_bitwise_exact"]
    ):
        raise ValueError("sealed CR7 result does not authorize CR9")
    if not (
        cr8_result["release_exact_replay"]
        and cr8_result["challenge_exact_replay"]
        and cr8_result["complete_readback_exact"]
        and cr8_result["release_interventions_exactly_zero"]
    ):
        raise ValueError("sealed CR8 context failed an integrity requirement")
    predictor_path = CR7_REGISTRATION / "frozen_full_predictor.npz"
    if sha256_file(predictor_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen JOINT_BREAK_RUN3 predictor changed")
    return {
        "cr0_checksum_manifest_sha256": sha256_file(CR0_VALIDATION / "SHA256SUMS"),
        "cr7_registration_checksum_manifest_sha256": sha256_file(
            CR7_REGISTRATION / "SHA256SUMS"
        ),
        "cr7_result_checksum_manifest_sha256": sha256_file(CR7_RESULT / "SHA256SUMS"),
        "cr8_result_checksum_manifest_sha256": sha256_file(CR8_RESULT / "SHA256SUMS"),
        "cr7_registration_id": cr7_registration["registration_id"],
        "cr7_primary_gate": cr7_result["complete_cr7_60_fission_gate"],
        "cr8_registered_shared_basin_radius": cr8_result[
            "registered_shared_basin_radius"
        ],
        "cr8_context_did_not_tune_cr9": True,
    }


def _artificial_case() -> tuple[StateCase, ExperimentConfig]:
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
        development=CohortConfig(1, 1, (60,)),
        confirmation=CohortConfig(1, 1, (60,)),
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
        generation=60,
        inheritance=(True, False, True),
        boundary_h=(0.95, 0.8, 0.93),
        previous_growth_steps=20,
        cumulative_growth_steps=2_400,
    )
    return (
        StateCase("CR9-ARTIFICIAL", LABEL, "02", 0, LANDMARK, beta, snapshot),
        current_experiment,
    )


def _artificial_execution(model_path: Path, registration_id: str) -> tuple[PhaseBatch, ...]:
    case, current_experiment = _artificial_case()
    return (
        _run_pulse_case(
            case,
            current_experiment,
            model_path,
            registration_id,
            pulse_lengths=(1, 2),
            replicates=1,
            release_horizon=3,
        ),
        _run_periodic_case(
            case,
            current_experiment,
            model_path,
            registration_id,
            policies=("MODEL_EVERY_1", "RANDOM_EVERY_1", "NOOP"),
            replicates=1,
            horizon=3,
        ),
        _run_event_case(
            case,
            current_experiment,
            model_path,
            registration_id,
            policies=("THRESHOLD_015", "CONTINUOUS", "NOOP"),
            replicates=1,
            horizon=3,
        ),
    )


def validation_checks() -> dict[str, Any]:
    upstream = _verify_upstream()
    model_path = CR7_REGISTRATION / "frozen_full_predictor.npz"
    first = _artificial_execution(model_path, "pre-registration-artificial")
    second = _artificial_execution(model_path, "pre-registration-artificial")
    pulse, periodic, event = first
    pulse_lineages = [item for item in pulse.lineages if isinstance(item, PulseLineage)]
    policy_lineages = [
        item
        for batch in (periodic, event)
        for item in batch.lineages
        if isinstance(item, PolicyLineage)
    ]
    noops = [item for item in policy_lineages if item.policy == "NOOP"]
    periodic_model = next(
        item for item in periodic.lineages if isinstance(item, PolicyLineage) and item.policy == "MODEL_EVERY_1"
    )
    periodic_random = next(
        item for item in periodic.lineages if isinstance(item, PolicyLineage) and item.policy == "RANDOM_EVERY_1"
    )
    draw = inference_draws()
    checks = {
        "upstream_cr0_cr7_cr8_integrity_pass": True,
        "design_exact": MATRICES == 48
        and LANDMARK == 60
        and REPLICATES == 6
        and HORIZON == 60
        and PULSE_LENGTHS == (1, 2, 4, 8, 16, 32, 60)
        and PERIODS == (1, 2, 4, 8, 16)
        and THRESHOLDS == (0.15, 0.25, 0.35),
        "seed_domains_unique": len(SEEDS) == len(set(SEEDS.values())),
        "seed_domains_disjoint_from_prior_registrations": set(SEEDS.values()).isdisjoint(
            _prior_seed_values()
        ),
        "pulse_future_seed_excludes_pulse_length": len(
            {_pulse_future_seed(_artificial_case()[0], 0) for _length in PULSE_LENGTHS}
        )
        == 1,
        "periodic_future_seed_excludes_policy": len(
            {_periodic_future_seed(_artificial_case()[0], 0) for _policy in PERIODIC_POLICIES}
        )
        == 1,
        "event_future_seed_excludes_policy": len(
            {_event_future_seed(_artificial_case()[0], 0) for _policy in EVENT_POLICIES}
        )
        == 1,
        "random_action_stream_distinct_from_future": _periodic_action_seed(
            _artificial_case()[0], 0, "RANDOM_EVERY_1"
        )
        != _periodic_future_seed(_artificial_case()[0], 0),
        "artificial_complete_deterministic_replay": [batch_digest(item) for item in first]
        == [batch_digest(item) for item in second],
        "artificial_all_three_modes_exercised": tuple(item.mode for item in first)
        == ("pulse", "periodic", "event"),
        "artificial_release_zero_interventions": all(
            item.release_interventions_applied == 0 for item in pulse_lineages
        ),
        "artificial_noop_plain_bitwise_exact": bool(noops)
        and all(item.noop_plain_bitwise_exact for item in noops),
        "artificial_periodic_action_steps_exact": periodic_model.action_steps
        == tuple(range(1, periodic_model.observed_fissions + 1)),
        "artificial_random_action_count_matches_schedule": len(
            periodic_random.action_steps
        )
        == periodic_random.observed_fissions,
        "strict_trigger_operator_frozen": bool(
            not (0.15 > 0.15) and (np.nextafter(0.15, 1.0) > 0.15)
        ),
        "spearman_increasing_fixture_exact": spearman_constant_zero(
            np.arange(7), np.arange(7)
        )
        == 1.0,
        "spearman_constant_fixture_zero": spearman_constant_zero(
            np.arange(7), np.ones(7)
        )
        == 0.0,
        "whole_matrix_draw_shapes_exact": draw["bootstrap_indices"].shape
        == (BOOTSTRAP_REPETITIONS, MATRICES)
        and draw["randomization_signs"].shape
        == (RANDOMIZATION_REPETITIONS, MATRICES),
        "frozen_model_hash_exact": sha256_file(model_path) == EXPECTED_MODEL_SHA256,
        "strict_eight_excluded": "strict-eight" in " ".join(
            protocol()["claim_boundary"]["prohibited"]
        ),
    }
    return {
        "format": VALIDATION_FORMAT,
        "checks": checks,
        "check_count": len(checks),
        "all_checks_passed": bool(all(checks.values())),
        "upstream": upstream,
        "artificial_non_scientific_fixture_only": True,
        "scientific_cr9_matrices_generated": 0,
        "scientific_cr9_policy_lineages_generated": 0,
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
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "CR9 full repository validation failed\n"
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
    print(f"CR9 validation sealed: {output}", flush=True)


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
        raise ValueError("CR9 validation did not pass")
    if validation["source_hashes"] != source_hashes():
        raise ValueError("CR9 source changed after validation")
    upstream = _verify_upstream()
    for forbidden in (DEFAULT_REGISTRATION, DEFAULT_SMOKE, DEFAULT_OUTPUT, DEFAULT_WORK):
        if forbidden.exists():
            raise FileExistsError(f"CR9 preregistration artifact already exists: {forbidden}")
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
        "scientific_policy_lineages_at_registration": 0,
    }
    registration_id = _canonical_digest(_json_ready(body))
    body["registration_id"] = registration_id
    with _atomic_destination(output) as destination:
        shutil.copy2(ROOT / DOCUMENT, destination / "preregistration.md")
        shutil.copy2(validation_directory / "validation.json", destination / "validation.json")
        shutil.copy2(
            CR7_REGISTRATION / "frozen_full_predictor.npz",
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
        f"<!-- registered-cr9-{registration_id} -->",
        [
            "## CR9 control half-life and minimum feedback registered",
            "",
            f"- Registration: `{registration_id}`.",
            "- Forty-eight fresh matrices, both candidates, six replicates, the seven-pulse ladder, five periodic schedules with budget-matched random controls, and three fixed risk thresholds were frozen before scientific generation.",
            "- The only confirmatory efficacy gate is positive whole-matrix pulse-length/persistence Spearman with a positive 95% lower bound in both candidates.",
            "- Periodic and event-triggered results map active-feedback economy and cannot rescue the pulse gate.",
            "- No CR9 scientific matrix or policy lineage existed at this seal.",
            "",
        ],
    )
    print(f"CR9 registered: {registration_id}", flush=True)


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    registration = json.loads((directory / "registration.json").read_text())
    if registration["format"] != REGISTRATION_FORMAT:
        raise ValueError("unsupported CR9 registration format")
    if registration["source_hashes"] != source_hashes():
        raise ValueError("CR9 registered source tree changed")
    body = dict(registration)
    observed = body.pop("registration_id")
    if _canonical_digest(_json_ready(body)) != observed:
        raise ValueError("CR9 registration ID changed")
    if registration["protocol"] != protocol() or registration["seed_registry"] != SEEDS:
        raise ValueError("CR9 registered protocol or seed registry changed")
    if sha256_file(directory / "frozen_full_predictor.npz") != EXPECTED_MODEL_SHA256:
        raise ValueError("CR9 frozen predictor copy changed")
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
        "format": "codex-intervention-cr9-smoke-v1",
        "registration_id": registration["registration_id"],
        "artificial_non_scientific_fixture": True,
        "all_three_io_paths_exercised": True,
        "exact_replay": [batch_digest(item) for item in first]
        == [batch_digest(item) for item in second],
        "release_applied_zero_interventions": all(
            isinstance(item, PulseLineage) and item.release_interventions_applied == 0
            for item in first[0].lineages
        ),
        "effect_sizes_arm_order_event_rates_and_candidate_differences_disclosed": False,
        "scientific_cr9_matrices_generated": 0,
        "scientific_cr9_policy_lineages_generated": 0,
    }
    if not all(
        payload[key]
        for key in (
            "artificial_non_scientific_fixture",
            "all_three_io_paths_exercised",
            "exact_replay",
            "release_applied_zero_interventions",
        )
    ):
        raise AssertionError("CR9 artificial smoke failed")
    with _atomic_destination(output) as destination:
        (destination / "smoke.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR9 non-scientific smoke passed: {output}", flush=True)


def _reports(metrics: dict[str, Any]) -> tuple[str, str]:
    technical = [
        "# CR9 control half-life and minimum feedback rate",
        "",
        "Registered two-candidate accumulating-hysteresis gate: "
        f"**{metrics['pulse']['complete_two_candidate_hysteresis_gate']}**.",
        f"Complete gate including replay/no-op/readback integrity: **{metrics['complete_cr9_registered_gate']}**.",
        "",
        "## Steering-pulse ladder",
        "",
        "| Candidate | Mean matrix Spearman | 95% matrix-bootstrap CI | One-sided randomization p | Primary gate |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate in metrics["pulse"]["candidates"]:
        rho = candidate["matrix_spearman"]
        technical.append(
            f"| {candidate['candidate']} | {rho['estimate']:+.6f} | "
            f"{rho['bootstrap_ci95']} | "
            f"{rho['positive_one_sided_randomization_p_raw']:.6g} | "
            f"{candidate['candidate_primary_gate']} |"
        )
    technical.extend(
        [
            "",
            "Mean persistence (fissions before anchor similarity first falls below 0.7; cap 61):",
            "",
            "| Candidate | Pulse 1 | Pulse 2 | Pulse 4 | Pulse 8 | Pulse 16 | Pulse 32 | Pulse 60 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in metrics["pulse"]["candidates"]:
        values = " | ".join(
            f"{item['mean_persistence']:.3f}" for item in candidate["pulse_lengths"]
        )
        technical.append(f"| {candidate['candidate']} | {values} |")
    technical.extend(
        [
            "",
            "## Periodic active feedback",
            "",
            "The descriptive minimum-feedback interval is the largest registered K whose MODEL_EVERY_K 95% lower bound is positive against both budget-matched random editing and NOOP. It is not a confirmatory rescue gate.",
            "",
            "| Candidate | Descriptive largest supported interval |",
            "|---|---:|",
        ]
    )
    for candidate in metrics["periodic"]["candidates"]:
        technical.append(
            f"| {candidate['candidate']} | {candidate['descriptive_minimum_feedback_interval']} |"
        )
    technical.extend(
        [
            "",
            "## Event-triggered active feedback",
            "",
            "| Candidate | Policy | Inheritance | Mean edits/60 | Gain vs NOOP | 95% CI | Fraction continuous gain |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in metrics["event_triggered"]["candidates"]:
        for item in candidate["policies"]:
            effect = item["inheritance_minus_noop"]
            recovery = item["fraction_continuous_gain_recovered"]
            technical.append(
                f"| {candidate['candidate']} | {item['policy']} | "
                f"{item['inheritance_mean']:.6f} | {item['mean_edits']:.3f} | "
                f"{effect['estimate']:+.6f} | {effect['bootstrap_ci95']} | "
                f"{recovery['estimate']:.3f} |"
            )
    technical.extend(
        [
            "",
            "All inference treats the catalytic matrix as the unit and keeps candidates separate. Missing registered boundaries after extinction count adversely in the fixed-horizon inheritance outcome. Full schedules, contrasts, action records, state trajectories, bootstrap draws, sign randomizations, and replay audits are machine-readable alongside this report.",
            "",
            "A longer-lived post-control trace would be transient hysteresis, not an autonomous restoring basin. Periodic and triggered policies remain active external feedback, even when they use few edits.",
            "",
        ]
    )
    pulse_pass = metrics["pulse"]["complete_two_candidate_hysteresis_gate"]
    lay = [
        "# CR9 in plain language",
        "",
        "CR9 asks whether longer training by the outside controller leaves a longer-lasting trace after the controller is switched off, and how often the controller really needs to intervene while it remains active.",
        "",
        (
            "The registered test found that longer steering reliably produced longer post-release persistence in both independent simulator candidates."
            if pulse_pass
            else "The registered test did not show reliable longer post-release persistence in both independent simulator candidates."
        ),
        "The periodic and risk-triggered results then show how much inheritance can be maintained with fewer molecular edits; they are reported as an economy map and cannot change the pulse result.",
        "",
        "Even a strong sparse-feedback result means an external sensor and controller can maintain organization efficiently. It does not mean that the assembly has installed its own memory or learned to repair itself without help.",
        "",
    ]
    return "\n".join(technical), "\n".join(lay)


def _write_result(
    output: Path,
    registration: dict[str, Any],
    cases: list[StateCase],
    metrics: dict[str, Any],
    stored_inference: dict[str, NDArray],
    replay: dict[str, dict[str, Any]],
    pulse_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, NDArray]],
    periodic_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, NDArray]],
    event_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, NDArray]],
) -> None:
    pulse_lineage, pulse_matrix, pulse_edits, pulse_arrays = pulse_data
    periodic_lineage, periodic_matrix, periodic_edits, periodic_arrays = periodic_data
    event_lineage, event_matrix, event_edits, event_arrays = event_data
    technical, lay = _reports(metrics)
    supported: list[str] = []
    if metrics["pulse"]["complete_two_candidate_hysteresis_gate"]:
        supported.append(
            "longer MODEL_DOWN steering pulses cause longer transient post-release compositional persistence in both Codex candidates"
        )
    for item in metrics["periodic"]["candidates"]:
        if item["descriptive_minimum_feedback_interval"] > 0:
            supported.append(
                f"candidate {item['candidate']} supports active MODEL_DOWN maintenance at a descriptive interval of {item['descriptive_minimum_feedback_interval']} fissions"
            )
    failed = [
        f"candidate {item['candidate']} accumulating-hysteresis gate"
        for item in metrics["pulse"]["candidates"]
        if not item["candidate_primary_gate"]
    ]
    claims = {
        "supported": supported,
        "failed_predictions": failed,
        "unresolved": [
            "whether another physical embodiment can internalize the corrective rule",
            "whether transient persistence generalizes beyond registered candidates and parameters",
            "whether sparse control transfers across parameter regimes",
        ],
        "prohibited": protocol()["claim_boundary"]["prohibited"],
        "transient_not_restoring_basin": True,
        "sparse_feedback_still_external_active_control": True,
        "periodic_and_event_results_do_not_rescue_pulse_gate": True,
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
        for mode, audit in replay.items():
            (replay_directory / f"{mode}_replay_audit.json").write_text(
                json.dumps(_json_ready(audit), indent=2, sort_keys=True) + "\n"
            )
        for name, data in (
            ("pulse", pulse_data),
            ("periodic", periodic_data),
            ("event", event_data),
        ):
            lineage, matrix, edits, arrays = data
            lineage.to_csv(destination / f"{name}_lineages.csv.gz", index=False, compression="gzip")
            matrix.to_csv(destination / f"{name}_matrix_effects.csv", index=False)
            edits.to_csv(destination / f"{name}_selected_edits.csv.gz", index=False, compression="gzip")
            np.savez_compressed(destination / f"{name}_trajectory_arrays.npz", **arrays)
        np.savez_compressed(destination / "inference_arrays.npz", **stored_inference)
        np.savez_compressed(
            destination / "state_and_matrix_arrays.npz",
            beta=beta_by_matrix,
            launch_compositions=launch_compositions,
            candidate=np.asarray([case.candidate for case in cases]),
            matrix_id=np.asarray([case.matrix_id for case in cases], dtype=np.int16),
            landmark=np.asarray([case.landmark for case in cases], dtype=np.int16),
        )
        readback_tables = {
            "pulse_lineage_rows": len(pd.read_csv(destination / "pulse_lineages.csv.gz")),
            "pulse_matrix_rows": len(pd.read_csv(destination / "pulse_matrix_effects.csv")),
            "periodic_lineage_rows": len(pd.read_csv(destination / "periodic_lineages.csv.gz")),
            "periodic_matrix_rows": len(pd.read_csv(destination / "periodic_matrix_effects.csv")),
            "event_lineage_rows": len(pd.read_csv(destination / "event_lineages.csv.gz")),
            "event_matrix_rows": len(pd.read_csv(destination / "event_matrix_effects.csv")),
        }
        with np.load(destination / "pulse_trajectory_arrays.npz", allow_pickle=False) as archive:
            pulse_shape_exact = archive["boundary_h"].shape == (
                len(cases) * REPLICATES * len(PULSE_LENGTHS),
                HORIZON,
            )
        with np.load(destination / "periodic_trajectory_arrays.npz", allow_pickle=False) as archive:
            periodic_shape_exact = archive["boundary_h"].shape == (
                len(cases) * REPLICATES * len(PERIODIC_POLICIES),
                HORIZON,
            )
        with np.load(destination / "event_trajectory_arrays.npz", allow_pickle=False) as archive:
            event_shape_exact = archive["boundary_h"].shape == (
                len(cases) * REPLICATES * len(EVENT_POLICIES),
                HORIZON,
            )
        readback = {
            "primary_metrics_exact": (destination / "primary_metrics.json").read_text()
            == metrics_text,
            "lineage_and_matrix_row_counts_exact": readback_tables
            == {
                "pulse_lineage_rows": len(cases) * REPLICATES * len(PULSE_LENGTHS),
                "pulse_matrix_rows": len(cases) * len(PULSE_LENGTHS),
                "periodic_lineage_rows": len(cases) * REPLICATES * len(PERIODIC_POLICIES),
                "periodic_matrix_rows": len(cases) * len(PERIODIC_POLICIES),
                "event_lineage_rows": len(cases) * REPLICATES * len(EVENT_POLICIES),
                "event_matrix_rows": len(cases) * len(EVENT_POLICIES),
            },
            "array_shapes_exact": bool(
                pulse_shape_exact and periodic_shape_exact and event_shape_exact
            ),
        }
        readback["complete_readback_exact"] = bool(all(readback.values()))
        if not readback["complete_readback_exact"]:
            raise AssertionError(f"CR9 written-artifact readback failed: {readback}")
        (destination / "readback_audit.json").write_text(
            json.dumps(readback, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "natural_launch_landmark": LANDMARK,
            "replicates_per_policy": REPLICATES,
            "pulse_lengths": list(PULSE_LENGTHS),
            "periodic_periods": list(PERIODS),
            "event_thresholds": list(THRESHOLDS),
            "horizon": HORIZON,
            "generated_policy_lineages": len(pulse_lineage)
            + len(periodic_lineage)
            + len(event_lineage),
            "replayed_policy_lineages": len(pulse_lineage)
            + len(periodic_lineage)
            + len(event_lineage),
            "complete_two_candidate_hysteresis_gate": metrics["pulse"][
                "complete_two_candidate_hysteresis_gate"
            ],
            "complete_cr9_registered_gate": metrics["complete_cr9_registered_gate"],
            "complete_exact_replay": all(
                audit["exact_state_action_endpoint_process_and_rng"]
                for audit in replay.values()
            ),
            "noop_callback_plain_bitwise_exact": metrics["integrity"][
                "noop_callback_plain_bitwise_exact"
            ],
            "release_interventions_exactly_zero": metrics["integrity"][
                "release_interventions_exactly_zero"
            ],
            "complete_readback_exact": True,
            "no_policy_lineage_retry_or_matrix_replacement": True,
            "no_refit_recalibration_or_threshold_selection": True,
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
        raise FileExistsError(f"refusing to overwrite completed CR9 result: {output}")
    free = shutil.disk_usage(ROOT).free
    if free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError(
            f"CR9 requires at least {MINIMUM_FREE_DISK_BYTES:,} free bytes; found {free:,}"
        )
    work.mkdir(parents=True, exist_ok=True)
    expected = {
        "format": "codex-intervention-cr9-work-contract-v1",
        "registration_id": registration_id,
        "protocol_id": protocol()["protocol_id"],
    }
    path = work / "campaign_contract.json"
    if path.is_file():
        if json.loads(path.read_text()) != expected:
            raise ValueError("CR9 work directory belongs to another campaign")
    else:
        path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")


def run(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_OUTPUT,
    work: Path = DEFAULT_WORK,
    workers: int = min(os.cpu_count() or 1, 14),
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

    _write_status(work, "building_fresh_natural_launch_states", 0, 2 * MATRICES)
    print(
        f"[cr9 1/10] Building {MATRICES} fresh matrices and {2 * MATRICES} natural generation-{LANDMARK} states",
        flush=True,
    )
    with threadpool_limits(limits=1):
        cases = build_cr9_cohort(current_experiment)
    if len(cases) != 2 * MATRICES:
        raise AssertionError("CR9 fresh cohort is incomplete")

    print(
        f"[cr9 2/10] Running {len(cases) * REPLICATES * len(PULSE_LENGTHS):,} pulse-and-release lineages",
        flush=True,
    )
    pulse = run_phase_batches(
        "pulse", cases, current_experiment, model_path, registration["registration_id"],
        work / "pulse" / "generate", workers, work, "pulse_generate"
    )
    print("[cr9 3/10] Replaying every pulse and untreated release", flush=True)
    pulse_replayed = run_phase_batches(
        "pulse", cases, current_experiment, model_path, registration["registration_id"],
        work / "pulse" / "replay", workers, work, "pulse_replay"
    )
    pulse_audit = replay_audit(pulse, pulse_replayed, "pulse")
    if not pulse_audit["exact_state_action_endpoint_process_and_rng"]:
        raise AssertionError("CR9 pulse exact replay failed")
    del pulse_replayed

    print(
        f"[cr9 4/10] Running {len(cases) * REPLICATES * len(PERIODIC_POLICIES):,} periodic-feedback lineages",
        flush=True,
    )
    periodic = run_phase_batches(
        "periodic", cases, current_experiment, model_path, registration["registration_id"],
        work / "periodic" / "generate", workers, work, "periodic_generate"
    )
    print("[cr9 5/10] Replaying every periodic-feedback lineage", flush=True)
    periodic_replayed = run_phase_batches(
        "periodic", cases, current_experiment, model_path, registration["registration_id"],
        work / "periodic" / "replay", workers, work, "periodic_replay"
    )
    periodic_audit = replay_audit(periodic, periodic_replayed, "periodic")
    if not periodic_audit["exact_state_action_endpoint_process_and_rng"]:
        raise AssertionError("CR9 periodic exact replay failed")
    del periodic_replayed

    print(
        f"[cr9 6/10] Running {len(cases) * REPLICATES * len(EVENT_POLICIES):,} event-triggered lineages",
        flush=True,
    )
    event = run_phase_batches(
        "event", cases, current_experiment, model_path, registration["registration_id"],
        work / "event" / "generate", workers, work, "event_generate"
    )
    print("[cr9 7/10] Replaying every event-triggered lineage", flush=True)
    event_replayed = run_phase_batches(
        "event", cases, current_experiment, model_path, registration["registration_id"],
        work / "event" / "replay", workers, work, "event_replay"
    )
    event_audit = replay_audit(event, event_replayed, "event")
    if not event_audit["exact_state_action_endpoint_process_and_rng"]:
        raise AssertionError("CR9 event exact replay failed")
    del event_replayed

    _write_status(work, "whole_matrix_inference", len(cases), len(cases))
    print("[cr9 8/10] Building lineage/action tables and whole-matrix inference", flush=True)
    pulse_data = pulse_tables(cases, pulse)
    periodic_data = policy_tables("periodic", cases, periodic)
    event_data = policy_tables("event", cases, event)
    noop_exact = bool(
        all(
            lineage.noop_plain_bitwise_exact
            for batch in periodic + event
            for lineage in batch.lineages
            if isinstance(lineage, PolicyLineage) and lineage.policy == "NOOP"
        )
    )
    release_zero = bool(
        all(
            lineage.release_interventions_applied == 0
            for batch in pulse
            for lineage in batch.lineages
            if isinstance(lineage, PulseLineage)
        )
    )
    metrics, stored = compute_inference(
        pulse_data[1],
        periodic_data[1],
        event_data[1],
        inference_draws(),
        pulse_replay_exact=True,
        periodic_replay_exact=True,
        event_replay_exact=True,
        noop_plain_exact=noop_exact,
        release_zero_interventions=release_zero,
        readback_exact=True,
    )
    _write_status(work, "writing_and_reading_back_artifacts", len(cases), len(cases))
    print("[cr9 9/10] Writing reports, machine-readable results, and readback audit", flush=True)
    replay = {"pulse": pulse_audit, "periodic": periodic_audit, "event": event_audit}
    _write_result(
        output,
        registration,
        cases,
        metrics,
        stored,
        replay,
        pulse_data,
        periodic_data,
        event_data,
    )
    _append_ledger(
        f"<!-- sealed-cr9-{registration['registration_id']} -->",
        [
            "## CR9 control half-life and minimum feedback sealed",
            "",
            f"- Registration: `{registration['registration_id']}`.",
            f"- Result: `{output.relative_to(ROOT)}`.",
            f"- Two-candidate accumulating-hysteresis gate: **{metrics['pulse']['complete_two_candidate_hysteresis_gate']}**.",
            f"- Complete registered gate with integrity: **{metrics['complete_cr9_registered_gate']}**.",
            f"- Pulse, periodic, and event replay exact: **{metrics['integrity']['pulse_exact_replay'] and metrics['integrity']['periodic_exact_replay'] and metrics['integrity']['event_exact_replay']}**; no-op/plain identity: **{noop_exact}**; zero release interventions: **{release_zero}**.",
            "- Periodic and threshold results are active-feedback economy maps and did not rescue or alter the pulse gate.",
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
        complete_hysteresis_gate=metrics["pulse"]["complete_two_candidate_hysteresis_gate"],
        complete_registered_gate=metrics["complete_cr9_registered_gate"],
    )
    print("[cr9 10/10] Result sealed; STOPPED before CR10", flush=True)


def read_status(work: Path = DEFAULT_WORK) -> dict[str, Any]:
    work = work.resolve()
    path = work / "campaign_status.json"
    if not path.is_file():
        raise FileNotFoundError(f"CR9 status does not exist: {path}")
    value = json.loads(path.read_text())
    value["checkpoint_counts"] = {
        relative: len(list((work / relative).glob("*.pkl")))
        if (work / relative).is_dir()
        else 0
        for relative in (
            "pulse/generate",
            "pulse/replay",
            "periodic/generate",
            "periodic/replay",
            "event/generate",
            "event/replay",
        )
    }
    value["free_disk_bytes"] = shutil.disk_usage(ROOT).free
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").add_argument("--output", type=Path, default=DEFAULT_VALIDATION)
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    register_parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    commands.add_parser("verify").add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    smoke_parser.add_argument("--output", type=Path, default=DEFAULT_SMOKE)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    run_parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    commands.add_parser("status").add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        validate(arguments.output)
    elif arguments.command == "register":
        register(arguments.validation, arguments.output)
    elif arguments.command == "verify":
        print(json.dumps(verify_registration(arguments.registration), indent=2, sort_keys=True))
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
