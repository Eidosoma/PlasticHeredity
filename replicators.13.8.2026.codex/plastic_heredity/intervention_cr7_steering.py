"""Prospectively frozen CR7 closed-loop hereditary steering campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from . import intervention_replication as base
from .config import CANDIDATES, CohortConfig, ExperimentConfig, GardConfig
from .experiment import StateCase, _json_ready, _runtime_manifest, build_cohort
from .features import history_features, state_graph_features
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
from .intervention_outgoing_rule import select_outgoing_rule_edits
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
CR1_RESULT = RESULT_ROOT / "cr1_model_guided_confirmation"
CR3_REGISTRATION = RESULT_ROOT / "cr3_confirmation_registration"
CR3_RESULT = RESULT_ROOT / "cr3_physical_rule_confirmation"
FROZEN_ARRAY_SOURCE = ROOT / "results/scaled5/analysis_arrays.npz"

DEFAULT_VALIDATION = RESULT_ROOT / "cr7_steering_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "cr7_steering_registration"
DEFAULT_SMOKE = RESULT_ROOT / "cr7_steering_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "cr7_closed_loop_steering"
DEFAULT_WORK = RESULT_ROOT / ".cr7_closed_loop_steering_work"

DOCUMENT = "CODEX_INTERVENTION_CR7_PREREGISTRATION.md"
PROGRAM_FORMAT = "codex-intervention-cr7-steering-v1"
VALIDATION_FORMAT = "codex-intervention-cr7-validation-v1"
REGISTRATION_FORMAT = "codex-intervention-cr7-registration-v1"
RESULT_FORMAT = "codex-intervention-cr7-result-v1"
CHECKPOINT_FORMAT = "codex-intervention-cr7-checkpoint-v1"
STATUS_FORMAT = "codex-intervention-cr7-status-v1"
LABEL = "INTCR7_CLOSED_LOOP_V1"

CR1_REGISTRATION_ID = "a8743234235e82133d2938c15ead062c7c85004c5f640d7359e5b075cb31368e"
CR3_REGISTRATION_ID = "64e871db56b3958a14bdad47b404f6c9f1ad09d0bda1e996e24498598523d189"
EXPECTED_MODEL_SHA256 = "9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af"

MATRICES = 48
LANDMARK = 60
REPLICATES = 6
HORIZON = 60
EXTENSION_HORIZON = 60
ARMS = ("MODEL_UP", "MODEL_DOWN", "RULE_UP", "RULE_DOWN", "RANDOM", "NOOP")
EXTENSION_ARMS = ("MODEL_DOWN", "RULE_DOWN", "NOOP")
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
RANDOM_EQUIVALENCE_MARGIN = 0.025
INHERITANCE_THRESHOLD = 0.9
MINIMUM_FREE_DISK_BYTES = 2_500_000_000

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_cr7_steering.py",
    "tests/test_intervention_cr7_steering.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/intervention_outgoing_rule.py",
    "plastic_heredity/intervention_replication.py",
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
        f"codex-clean-room-cr7-closed-loop-v1::{name}".encode("utf-8")
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
        "future_simulation",
        "controller_action",
        "bootstrap",
        "randomization",
        "replay",
        "conditional_extension",
    )
}


@dataclass(frozen=True)
class LineageSummary:
    controller: str
    replicate: int
    completed_horizon: bool
    observed_fissions: int
    inherited_boundary_count: int
    inherited_fraction: float
    total_breaks: int
    episode_count: int
    longest_inherited_run: int
    mean_growth_updates: float
    final_entropy: float
    final_occupied_types: int
    final_top1_share: float
    final_throughput: float
    final_risk: float
    mean_predicted_action_shift: float
    out_of_development_envelope_fraction: float
    distinct_swaps: int
    repeated_swaps: int
    immediately_reversing_swaps: int
    record_digest: str
    boundary_h: NDArray[np.float64]
    growth_updates: NDArray[np.int32]
    final_snapshot: Snapshot
    actions: tuple[MolecularEdit, ...]
    risk_before: NDArray[np.float64]
    risk_after: NDArray[np.float64]
    out_of_envelope: NDArray[np.int8]
    simulation_rng_state: dict[str, Any]
    noop_plain_bitwise_exact: bool


@dataclass(frozen=True)
class SteeringBatch:
    format: str
    registration_id: str
    mode: str
    state_id: str
    candidate: str
    matrix_id: int
    landmark: int
    case_digest: str
    lineages: tuple[LineageSummary, ...]


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_cr7_scientific_matrix",
        "upstream": {
            "cr1_registration_id": CR1_REGISTRATION_ID,
            "cr1_model_guided_gate_passed": True,
            "cr3_registration_id": CR3_REGISTRATION_ID,
            "cr3_outgoing_rule_gate_passed": True,
            "cr6_complete_gate": False,
            "cr6_not_used_to_tune_or_authorize_cr7": True,
        },
        "target": {
            "name": "JOINT_BREAK_RUN3",
            "inheritance": "strict unrounded float64 H > 0.9",
            "episode_counter": "non-overlapping; new break required after certification",
            "strict_eight_excluded": True,
        },
        "cohort": {
            "fresh_matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "untreated_launch_landmark": LANDMARK,
            "replicates_per_controller": REPLICATES,
            "controllers": list(ARMS),
            "fissions": HORIZON,
            "primary_lineages": MATRICES * len(CANDIDATES) * REPLICATES * len(ARMS),
            "maximum_primary_boundaries": MATRICES
            * len(CANDIDATES)
            * REPLICATES
            * len(ARMS)
            * HORIZON,
            "complete_replay": True,
            "controlled_lineage_retry_or_replacement": False,
        },
        "controllers": {
            "MODEL_UP": "exhaustive frozen-predictor maximum; first lexicographic tie",
            "MODEL_DOWN": "exhaustive frozen-predictor minimum; first lexicographic tie",
            "RULE_UP": "frozen outgoing x@beta destabilizing rule",
            "RULE_DOWN": "frozen outgoing x@beta stabilizing rule",
            "RANDOM": "uniform legal swap from separate action stream",
            "NOOP": "callback returning no edit",
            "callback_after_every_successful_fission_including_last": True,
        },
        "randomness": {
            "seed_domains": SEEDS,
            "future_seed_excludes_controller": True,
            "common_random_streams_not_identical_realized_futures": True,
            "action_stream_separate_from_simulation": True,
        },
        "outcomes": {
            "primary": [
                "inherited_boundary_fraction",
                "total_breaks",
                "nonoverlapping_joint_break_run3_episode_count",
                "longest_inherited_run",
            ],
            "secondary": [
                "final_entropy",
                "final_occupied_types",
                "final_top1_share",
                "final_xT_beta_x_throughput",
                "mean_growth_updates",
                "survival",
                "cross_lineage_final_cosine",
                "distinct_repeated_and_immediately_reversing_swaps",
                "out_of_development_envelope_fraction",
            ],
            "failed_lineages_retained": True,
            "fractions_use_observed_boundaries": True,
        },
        "development_envelope": {
            "source": "original 5x development matrices only",
            "coordinates": 21,
            "candidate_separated_coordinatewise_minimum_and_maximum": True,
            "post_action_any_coordinate_outside_is_ood": True,
            "descriptive_only": True,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions_descriptive": RANDOMIZATION_REPETITIONS,
            "candidates_never_pooled": True,
            "random_noop_tost_margin": RANDOM_EQUIVALENCE_MARGIN,
        },
        "primary_gates_per_candidate": [
            "MODEL_DOWN-NOOP inheritance CI95 lower > 0",
            "RULE_DOWN-NOOP inheritance CI95 lower > 0",
            "MODEL_UP-NOOP inheritance CI95 upper < 0",
            "MODEL_UP-MODEL_DOWN episode-count CI95 lower > 0",
            "RANDOM-NOOP inheritance CI90 inside +/-0.025",
        ],
        "integrity_gates": ["all NOOP callbacks bitwise plain", "complete exact replay"],
        "rule_recovery": {
            "formula": "(RULE_DOWN-NOOP)/(MODEL_DOWN-NOOP)",
            "external_point_hypothesis_at_least": 0.80,
            "strong_classification_bootstrap_lower_above": 0.70,
            "nonpositive_denominator_draws_invalid": True,
            "strong_classification_requires_at_least_95_percent_valid_draws": True,
        },
        "conditional_extension": {
            "launch_only_if_all_primary_and_integrity_gates_pass": True,
            "arms": list(EXTENSION_ARMS),
            "additional_fissions": EXTENSION_HORIZON,
            "continues_exact_state_and_simulation_rng": True,
            "active_feedback_not_passive_persistence": True,
            "complete_replay": True,
        },
        "operational": {
            "checkpoint_resumable": True,
            "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
            "mandatory_stop_after_seal": True,
            "cr8_and_cr9_not_launched_automatically": True,
        },
        "claim_boundary": {
            "prohibited": [
                "strict-eight control",
                "autonomous persistence or installed attractor",
                "biological memory, agency, life, or error correction",
                "real prebiotic chemistry or universal origin-of-life mechanism",
                "Phi or PhiID intervention",
            ]
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
        regenerate_confirmation=True,
        master_seed=SEEDS["matrix_generation"],
    )


def build_cr7_cohort(current_experiment: ExperimentConfig) -> list[StateCase]:
    """Build the frozen fresh cohort with explicitly separated seed domains."""

    cases: list[StateCase] = []
    for matrix_id in range(MATRICES):
        beta = generate_beta(
            current_experiment.gard,
            np.random.default_rng(
                derive_seed(
                    SEEDS["matrix_generation"], f"{LABEL}.beta", matrix_id
                )
            ),
        )
        initial = generate_initial_composition(
            current_experiment.gard,
            np.random.default_rng(
                derive_seed(
                    SEEDS["initial_composition"], f"{LABEL}.initial", matrix_id
                )
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
                    f"failed to obtain CR7 natural trajectory for candidate "
                    f"{candidate}, matrix {matrix_id} in 100 attempts"
                )
            by_generation = {snapshot.generation: snapshot for snapshot in lineage}
            snapshot = by_generation[LANDMARK]
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


def model_coordinates(
    predictor: FrozenFullPredictor,
    candidate: str,
    snapshot: Snapshot,
    beta: NDArray,
    config: GardConfig,
) -> NDArray[np.float64]:
    state = state_graph_features(snapshot.composition, beta, config)
    history = history_features(snapshot, config)
    base_name = f"c{candidate}"
    state_scaled = (
        state - predictor.arrays[f"{base_name}__full_state_scaler_mean"]
    ) / predictor.arrays[f"{base_name}__full_state_scaler_scale"]
    components = (
        state_scaled - predictor.arrays[f"{base_name}__full_state_pca_mean"]
    ) @ predictor.arrays[f"{base_name}__full_state_pca_components"].T
    unscaled = np.concatenate((components, history))
    return np.asarray(
        (unscaled - predictor.arrays[f"{base_name}__full__scaler_mean"])
        / predictor.arrays[f"{base_name}__full__scaler_scale"],
        dtype=np.float64,
    )


def development_envelope(
    predictor: FrozenFullPredictor,
) -> dict[str, NDArray[np.float64]]:
    with np.load(FROZEN_ARRAY_SOURCE, allow_pickle=False) as arrays:
        state = np.asarray(arrays["development_state_graph"], dtype=np.float64)
        history = np.asarray(arrays["development_history"], dtype=np.float64)
    row = np.arange(state.shape[0]) % (2 * len(base.LANDMARKS))
    output: dict[str, NDArray[np.float64]] = {}
    for candidate, mask in (
        ("02", row < len(base.LANDMARKS)),
        ("03", row >= len(base.LANDMARKS)),
    ):
        base_name = f"c{candidate}"
        state_scaled = (
            state[mask] - predictor.arrays[f"{base_name}__full_state_scaler_mean"]
        ) / predictor.arrays[f"{base_name}__full_state_scaler_scale"]
        components = (
            state_scaled - predictor.arrays[f"{base_name}__full_state_pca_mean"]
        ) @ predictor.arrays[f"{base_name}__full_state_pca_components"].T
        unscaled = np.column_stack((components, history[mask]))
        transformed = (
            unscaled - predictor.arrays[f"{base_name}__full__scaler_mean"]
        ) / predictor.arrays[f"{base_name}__full__scaler_scale"]
        output[f"c{candidate}__minimum"] = transformed.min(axis=0)
        output[f"c{candidate}__maximum"] = transformed.max(axis=0)
    return output


def is_out_of_envelope(
    coordinates: NDArray,
    candidate: str,
    envelope: dict[str, NDArray],
) -> bool:
    values = np.asarray(coordinates, dtype=np.float64)
    lower = np.asarray(envelope[f"c{candidate}__minimum"], dtype=np.float64)
    upper = np.asarray(envelope[f"c{candidate}__maximum"], dtype=np.float64)
    return bool(np.any(values < lower) or np.any(values > upper))


def count_nonoverlapping_episodes(
    boundary_h: NDArray | list[float] | tuple[float, ...],
    threshold: float = INHERITANCE_THRESHOLD,
) -> int:
    seeking_break = True
    trailing = 0
    episodes = 0
    for value in np.asarray(boundary_h, dtype=np.float64):
        inherited = bool(value > threshold)
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


def longest_inherited_run(
    boundary_h: NDArray | list[float] | tuple[float, ...],
    threshold: float = INHERITANCE_THRESHOLD,
) -> int:
    best = 0
    current = 0
    for value in np.asarray(boundary_h, dtype=np.float64):
        if value > threshold:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _entropy(composition: NDArray) -> float:
    values = np.asarray(composition, dtype=np.float64)
    mass = float(values.sum())
    if mass <= 0.0:
        return 0.0
    positive = values[values > 0.0] / mass
    return float(-np.dot(positive, np.log(positive)))


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
        and np.array_equal(
            np.asarray(left.boundary_h, dtype=np.float64),
            np.asarray(right.boundary_h, dtype=np.float64),
        )
        and left.previous_growth_steps == right.previous_growth_steps
        and left.cumulative_growth_steps == right.cumulative_growth_steps
    )


def _rng_state_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(_json_ready(left), sort_keys=True) == json.dumps(
        _json_ready(right), sort_keys=True
    )


class _Trace:
    def __init__(self) -> None:
        self.risk_before: list[float] = []
        self.risk_after: list[float] = []
        self.out_of_envelope: list[int] = []
        self.actions: list[MolecularEdit] = []


def _controller(
    name: str,
    predictor: FrozenFullPredictor,
    config: GardConfig,
    envelope: dict[str, NDArray],
    action_rng: np.random.Generator,
) -> tuple[Callable[[Snapshot, NDArray, str, int], MolecularEdit | None], _Trace]:
    if name not in ARMS:
        raise ValueError(f"unknown CR7 controller: {name}")
    trace = _Trace()

    def callback(
        snapshot: Snapshot, beta: NDArray, candidate: str, _step: int
    ) -> MolecularEdit | None:
        before = predictor.predict_snapshot(candidate, snapshot, beta, config)
        edit: MolecularEdit | None
        after = before
        if name in ("MODEL_UP", "MODEL_DOWN"):
            noop, scores = score_legal_edits(
                predictor, candidate, snapshot, beta, config
            )
            probabilities = np.asarray(
                [item.predicted_probability for item in scores], dtype=np.float64
            )
            extreme = probabilities.max() if name == "MODEL_UP" else probabilities.min()
            index = int(np.flatnonzero(probabilities == extreme)[0])
            edit = scores[index].edit
            before = float(noop)
            after = float(scores[index].predicted_probability)
        elif name in ("RULE_UP", "RULE_DOWN"):
            edit = select_outgoing_rule_edits(snapshot.composition, beta)[name]
            after = predictor.predict_snapshot(
                candidate, edited_snapshot(snapshot, edit), beta, config
            )
        elif name == "RANDOM":
            legal = enumerate_legal_edits(snapshot.composition)
            edit = legal[int(action_rng.integers(0, len(legal)))]
            after = predictor.predict_snapshot(
                candidate, edited_snapshot(snapshot, edit), beta, config
            )
        else:
            edit = None
        post = edited_snapshot(snapshot, edit) if edit is not None else snapshot
        coordinates = model_coordinates(predictor, candidate, post, beta, config)
        trace.risk_before.append(float(before))
        trace.risk_after.append(float(after))
        trace.out_of_envelope.append(int(is_out_of_envelope(coordinates, candidate, envelope)))
        if edit is not None:
            trace.actions.append(edit)
        return edit

    return callback, trace


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


def _lineage_summary(
    name: str,
    replicate: int,
    result: ControlledResult,
    trace: _Trace,
    beta: NDArray,
    predictor: FrozenFullPredictor,
    candidate: str,
    config: GardConfig,
    horizon: int,
    rng_state: dict[str, Any],
    noop_plain_exact: bool,
) -> LineageSummary:
    observed = len(result.records)
    h = np.full(horizon, np.nan, dtype=np.float64)
    growth = np.full(horizon, -1, dtype=np.int32)
    for index, record in enumerate(result.records):
        h[index] = float(record.h)
        growth[index] = int(record.growth_steps)
    seen_h = h[:observed]
    inherited = seen_h > INHERITANCE_THRESHOLD
    composition = np.asarray(result.final_snapshot.composition, dtype=np.int64)
    mass = int(composition.sum())
    actions = tuple(result.selected_edits)
    if tuple(trace.actions) != actions:
        raise AssertionError("controller trace and applied edit sequence differ")
    distinct = len(set(actions))
    reversing = sum(
        current.remove_type == previous.add_type
        and current.add_type == previous.remove_type
        for previous, current in zip(actions, actions[1:])
    )
    risk_before = np.asarray(trace.risk_before, dtype=np.float64)
    risk_after = np.asarray(trace.risk_after, dtype=np.float64)
    ood = np.asarray(trace.out_of_envelope, dtype=np.int8)
    if risk_before.size != observed or risk_after.size != observed or ood.size != observed:
        raise AssertionError("controller callback count differs from observed fissions")
    return LineageSummary(
        controller=name,
        replicate=replicate,
        completed_horizon=bool(result.completed_horizon),
        observed_fissions=observed,
        inherited_boundary_count=int(inherited.sum()),
        inherited_fraction=float(inherited.mean()) if observed else 0.0,
        total_breaks=int((~inherited).sum()) if observed else 0,
        episode_count=count_nonoverlapping_episodes(seen_h),
        longest_inherited_run=longest_inherited_run(seen_h),
        mean_growth_updates=float(growth[:observed].mean()) if observed else float("nan"),
        final_entropy=_entropy(composition),
        final_occupied_types=int(np.count_nonzero(composition)),
        final_top1_share=(float(composition.max() / mass) if mass > 0 else 0.0),
        final_throughput=_throughput(composition, beta),
        final_risk=predictor.predict_snapshot(
            candidate, result.final_snapshot, beta, config
        ),
        mean_predicted_action_shift=(
            float(np.mean(risk_after - risk_before)) if observed else 0.0
        ),
        out_of_development_envelope_fraction=(float(ood.mean()) if observed else 0.0),
        distinct_swaps=distinct,
        repeated_swaps=len(actions) - distinct,
        immediately_reversing_swaps=int(reversing),
        record_digest=_records_digest(result.records),
        boundary_h=h,
        growth_updates=growth,
        final_snapshot=result.final_snapshot,
        actions=actions,
        risk_before=risk_before,
        risk_after=risk_after,
        out_of_envelope=ood,
        simulation_rng_state=_json_ready(rng_state),
        noop_plain_bitwise_exact=bool(noop_plain_exact),
    )


def _future_seed(case: StateCase, replicate: int) -> int:
    return derive_seed(
        SEEDS["future_simulation"],
        f"{LABEL}.future",
        case.candidate,
        case.matrix_id,
        replicate,
    )


def _action_seed(case: StateCase, replicate: int) -> int:
    return derive_seed(
        SEEDS["controller_action"],
        f"{LABEL}.random_controller",
        case.candidate,
        case.matrix_id,
        replicate,
    )


def _run_case(
    case: StateCase,
    current_experiment: ExperimentConfig,
    model_path: str | Path,
    envelope_path: str | Path,
    registration_id: str,
    *,
    horizon: int = HORIZON,
    replicates: int = REPLICATES,
    arms: tuple[str, ...] = ARMS,
) -> SteeringBatch:
    predictor = FrozenFullPredictor.load(model_path)
    with np.load(envelope_path, allow_pickle=False) as archive:
        envelope = {name: archive[name] for name in archive.files}
    lineages: list[LineageSummary] = []
    for replicate in range(replicates):
        simulation_seed = _future_seed(case, replicate)
        for name in arms:
            rng = np.random.default_rng(simulation_seed)
            action_rng = np.random.default_rng(_action_seed(case, replicate))
            callback, trace = _controller(
                name, predictor, current_experiment.gard, envelope, action_rng
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
            if name == "NOOP":
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
                _lineage_summary(
                    name,
                    replicate,
                    result,
                    trace,
                    case.beta,
                    predictor,
                    case.candidate,
                    current_experiment.gard,
                    horizon,
                    rng.bit_generator.state,
                    noop_exact,
                )
            )
    return SteeringBatch(
        format=CHECKPOINT_FORMAT,
        registration_id=registration_id,
        mode="primary",
        state_id=case.state_id,
        candidate=case.candidate,
        matrix_id=case.matrix_id,
        landmark=case.landmark,
        case_digest=_case_digest(case),
        lineages=tuple(lineages),
    )


def _worker(arguments: tuple[Any, ...]) -> SteeringBatch:
    limiter = threadpool_limits(limits=1)
    try:
        return _run_case(*arguments)
    finally:
        limiter.restore_original_limits()


def _lineage_digest(lineage: LineageSummary) -> str:
    digest = hashlib.sha256()
    for value in (
        lineage.controller,
        str(lineage.replicate),
        str(int(lineage.completed_horizon)),
        str(lineage.observed_fissions),
        lineage.record_digest,
    ):
        digest.update(value.encode())
    scalar_values = (
        lineage.inherited_boundary_count,
        lineage.inherited_fraction,
        lineage.total_breaks,
        lineage.episode_count,
        lineage.longest_inherited_run,
        lineage.mean_growth_updates,
        lineage.final_entropy,
        lineage.final_occupied_types,
        lineage.final_top1_share,
        lineage.final_throughput,
        lineage.final_risk,
        lineage.mean_predicted_action_shift,
        lineage.out_of_development_envelope_fraction,
        lineage.distinct_swaps,
        lineage.repeated_swaps,
        lineage.immediately_reversing_swaps,
        int(lineage.noop_plain_bitwise_exact),
    )
    digest.update(np.asarray(scalar_values, dtype=np.float64).tobytes())
    for array in (
        lineage.boundary_h,
        lineage.growth_updates,
        lineage.final_snapshot.composition,
        lineage.risk_before,
        lineage.risk_after,
        lineage.out_of_envelope,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    action_array = np.asarray(
        [(item.remove_type, item.add_type) for item in lineage.actions],
        dtype=np.int16,
    ).reshape(-1, 2)
    digest.update(action_array.tobytes())
    digest.update(
        json.dumps(_json_ready(lineage.simulation_rng_state), sort_keys=True).encode()
    )
    digest.update(
        np.asarray(
            (
                lineage.final_snapshot.generation,
                lineage.final_snapshot.previous_growth_steps,
                lineage.final_snapshot.cumulative_growth_steps,
            ),
            dtype=np.int64,
        ).tobytes()
    )
    digest.update(np.asarray(lineage.final_snapshot.boundary_h, dtype=np.float64).tobytes())
    digest.update(np.asarray(lineage.final_snapshot.inheritance, dtype=np.int8).tobytes())
    return digest.hexdigest()


def batch_digest(batch: SteeringBatch) -> str:
    digest = hashlib.sha256()
    for value in (
        batch.format,
        batch.registration_id,
        batch.mode,
        batch.state_id,
        batch.candidate,
        str(batch.matrix_id),
        str(batch.landmark),
        batch.case_digest,
    ):
        digest.update(value.encode())
    for lineage in batch.lineages:
        digest.update(_lineage_digest(lineage).encode())
    return digest.hexdigest()


def _checkpoint_path(directory: Path, case: StateCase) -> Path:
    return directory / f"c{case.candidate}_m{case.matrix_id:03d}.pkl"


def _write_checkpoint(path: Path, batch: SteeringBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(batch, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_checkpoint(
    path: Path, case: StateCase, registration_id: str
) -> SteeringBatch | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            batch = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(batch, SteeringBatch):
        return None
    expected = (CHECKPOINT_FORMAT, registration_id, case.state_id, _case_digest(case))
    observed = (batch.format, batch.registration_id, batch.state_id, batch.case_digest)
    if observed != expected:
        return None
    if len(batch.lineages) != REPLICATES * len(ARMS):
        return None
    return batch


def _write_status(
    work: Path,
    stage: str,
    completed: int,
    total: int,
    **extra: Any,
) -> None:
    work.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": STATUS_FORMAT,
        "stage": stage,
        "completed_state_batches": completed,
        "total_state_batches": total,
        **extra,
    }
    temporary = work / f".status-{os.getpid()}.tmp"
    temporary.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(work / "campaign_status.json")


def run_batches(
    cases: list[StateCase],
    current_experiment: ExperimentConfig,
    model_path: Path,
    envelope_path: Path,
    registration_id: str,
    directory: Path,
    workers: int,
    work: Path,
    stage: str,
) -> list[SteeringBatch]:
    directory.mkdir(parents=True, exist_ok=True)
    batches: dict[str, SteeringBatch] = {}
    missing: list[StateCase] = []
    for case in cases:
        checkpoint = _read_checkpoint(
            _checkpoint_path(directory, case), case, registration_id
        )
        if checkpoint is None:
            missing.append(case)
        else:
            batches[case.state_id] = checkpoint
    _write_status(work, stage, len(batches), len(cases), reused=len(batches))
    arguments = [
        (case, current_experiment, model_path, envelope_path, registration_id)
        for case in missing
    ]
    if workers == 1:
        iterator = ((_worker(argument), case) for argument, case in zip(arguments, missing))
        for batch, case in iterator:
            _write_checkpoint(_checkpoint_path(directory, case), batch)
            batches[case.state_id] = batch
            _write_status(work, stage, len(batches), len(cases), reused=len(cases) - len(missing))
            print(f"[{stage}] {len(batches)}/{len(cases)} state batches", flush=True)
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_case = {
                executor.submit(_worker, argument): case
                for argument, case in zip(arguments, missing)
            }
            for future in as_completed(future_case):
                case = future_case[future]
                batch = future.result()
                _write_checkpoint(_checkpoint_path(directory, case), batch)
                batches[case.state_id] = batch
                _write_status(
                    work,
                    stage,
                    len(batches),
                    len(cases),
                    reused=len(cases) - len(missing),
                )
                print(f"[{stage}] {len(batches)}/{len(cases)} state batches", flush=True)
    ordered = [batches[case.state_id] for case in cases]
    if len(ordered) != len(cases):
        raise AssertionError("CR7 checkpoint cohort is incomplete")
    return ordered


def inference_draws() -> dict[str, NDArray]:
    bootstrap_rng = np.random.default_rng(
        derive_seed(SEEDS["bootstrap"], f"{LABEL}.whole_matrix_bootstrap")
    )
    randomization_rng = np.random.default_rng(
        derive_seed(SEEDS["randomization"], f"{LABEL}.whole_matrix_signs")
    )
    indices = bootstrap_rng.integers(
        0,
        MATRICES,
        size=(BOOTSTRAP_REPETITIONS, MATRICES),
        dtype=np.int64,
    )
    signs = randomization_rng.integers(
        0,
        2,
        size=(RANDOMIZATION_REPETITIONS, MATRICES),
        dtype=np.int8,
    ).astype(np.float64)
    return {"bootstrap_indices": indices, "randomization_signs": 2.0 * signs - 1.0}


def _interval(values: NDArray, alpha: float = 0.05) -> tuple[float, float]:
    lower, upper = np.quantile(np.asarray(values, dtype=np.float64), (alpha / 2, 1 - alpha / 2))
    return float(lower), float(upper)


def _maximum_leave_one_out_influence(values: NDArray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return float("nan")
    estimate = float(array.mean())
    leave_one = (array.sum() - array) / (array.size - 1)
    return float(np.max(np.abs(leave_one - estimate)))


def _sign_p(values: NDArray, signs: NDArray, direction: str) -> tuple[float, NDArray]:
    array = np.asarray(values, dtype=np.float64)
    oriented = array if direction == "positive" else -array
    observed = float(oriented.mean())
    null = np.asarray(signs @ oriented / oriented.size, dtype=np.float64)
    p_value = float((np.count_nonzero(null >= observed) + 1) / (null.size + 1))
    return p_value, null


def _pairwise_cosine(compositions: list[NDArray]) -> float:
    values = [np.asarray(item, dtype=np.float64) for item in compositions]
    if len(values) < 2:
        return float("nan")
    similarities = [
        cosine_similarity(values[left], values[right])
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    return float(np.mean(similarities))


def _lineage_and_matrix_tables(
    cases: list[StateCase], batches: list[SteeringBatch]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lineage_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    final_by_group: dict[tuple[str, int, str], list[NDArray]] = {}
    for case, batch in zip(cases, batches, strict=True):
        if batch.case_digest != _case_digest(case):
            raise ValueError("CR7 batch no longer matches its launch state")
        for lineage in batch.lineages:
            key = (case.candidate, case.matrix_id, lineage.controller)
            final_by_group.setdefault(key, []).append(lineage.final_snapshot.composition)
            lineage_rows.append(
                {
                    "state_id": case.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "landmark": case.landmark,
                    "controller": lineage.controller,
                    "replicate": lineage.replicate,
                    "completed_horizon": int(lineage.completed_horizon),
                    "observed_fissions": lineage.observed_fissions,
                    "inherited_boundary_count": lineage.inherited_boundary_count,
                    "inherited_fraction": lineage.inherited_fraction,
                    "total_breaks": lineage.total_breaks,
                    "episode_count": lineage.episode_count,
                    "longest_inherited_run": lineage.longest_inherited_run,
                    "mean_growth_updates": lineage.mean_growth_updates,
                    "final_entropy": lineage.final_entropy,
                    "final_occupied_types": lineage.final_occupied_types,
                    "final_top1_share": lineage.final_top1_share,
                    "final_throughput": lineage.final_throughput,
                    "final_risk": lineage.final_risk,
                    "mean_predicted_action_shift": lineage.mean_predicted_action_shift,
                    "out_of_development_envelope_fraction": lineage.out_of_development_envelope_fraction,
                    "distinct_swaps": lineage.distinct_swaps,
                    "repeated_swaps": lineage.repeated_swaps,
                    "immediately_reversing_swaps": lineage.immediately_reversing_swaps,
                    "record_digest": lineage.record_digest,
                    "noop_plain_bitwise_exact": int(lineage.noop_plain_bitwise_exact),
                    "final_composition_digest": hashlib.sha256(
                        np.ascontiguousarray(lineage.final_snapshot.composition).tobytes()
                    ).hexdigest(),
                }
            )
            for step, edit in enumerate(lineage.actions, start=1):
                edit_rows.append(
                    {
                        "state_id": case.state_id,
                        "candidate": case.candidate,
                        "matrix_id": case.matrix_id,
                        "controller": lineage.controller,
                        "replicate": lineage.replicate,
                        "action_number": step,
                        "remove_type": edit.remove_type,
                        "add_type": edit.add_type,
                        "predicted_risk_before": float(lineage.risk_before[step - 1]),
                        "predicted_risk_after": float(lineage.risk_after[step - 1]),
                        "out_of_development_envelope": int(lineage.out_of_envelope[step - 1]),
                    }
                )
    lineage = pd.DataFrame(lineage_rows)
    edits = pd.DataFrame(edit_rows)
    matrix_rows: list[dict[str, Any]] = []
    numeric = (
        "completed_horizon",
        "observed_fissions",
        "inherited_boundary_count",
        "inherited_fraction",
        "total_breaks",
        "episode_count",
        "longest_inherited_run",
        "mean_growth_updates",
        "final_entropy",
        "final_occupied_types",
        "final_top1_share",
        "final_throughput",
        "final_risk",
        "mean_predicted_action_shift",
        "out_of_development_envelope_fraction",
        "distinct_swaps",
        "repeated_swaps",
        "immediately_reversing_swaps",
    )
    for (candidate, matrix_id, controller), group in lineage.groupby(
        ["candidate", "matrix_id", "controller"], sort=True
    ):
        if len(group) != REPLICATES:
            raise ValueError("whole-matrix controller block lost a replicate")
        row: dict[str, Any] = {
            "candidate": str(candidate).zfill(2),
            "matrix_id": int(matrix_id),
            "controller": controller,
            "cross_lineage_final_cosine": _pairwise_cosine(
                final_by_group[(str(candidate).zfill(2), int(matrix_id), controller)]
            ),
        }
        row.update({name: float(group[name].mean()) for name in numeric})
        matrix_rows.append(row)
    matrix = pd.DataFrame(matrix_rows)
    return lineage, matrix, edits


def compute_inference(
    matrix_table: pd.DataFrame,
    draws: dict[str, NDArray],
    *,
    replay_exact: bool,
    noop_plain_exact: bool,
) -> tuple[dict[str, Any], dict[str, NDArray]]:
    bootstrap = np.asarray(draws["bootstrap_indices"], dtype=np.int64)
    signs = np.asarray(draws["randomization_signs"], dtype=np.float64)
    if bootstrap.shape != (BOOTSTRAP_REPETITIONS, MATRICES):
        raise ValueError("CR7 bootstrap draws lost whole-matrix blocks")
    if signs.shape != (RANDOMIZATION_REPETITIONS, MATRICES):
        raise ValueError("CR7 randomization draws lost whole-matrix blocks")
    candidates: list[dict[str, Any]] = []
    stored: dict[str, NDArray] = {
        "bootstrap_indices": bootstrap,
        "randomization_signs": signs,
    }
    for candidate in CANDIDATES:
        selected = matrix_table[
            matrix_table["candidate"].astype(str).str.zfill(2) == candidate
        ]
        pivots: dict[str, pd.DataFrame] = {}
        for outcome in (
            "inherited_fraction",
            "total_breaks",
            "episode_count",
            "longest_inherited_run",
        ):
            pivot = selected.pivot(index="matrix_id", columns="controller", values=outcome)
            pivot = pivot.reindex(index=np.arange(MATRICES), columns=ARMS)
            if pivot.isna().any().any():
                raise ValueError(f"candidate {candidate} lacks a complete matrix block")
            pivots[outcome] = pivot
        inherited = pivots["inherited_fraction"]
        episodes = pivots["episode_count"]
        contrasts = {
            "model_down_minus_noop_inheritance": inherited["MODEL_DOWN"].to_numpy()
            - inherited["NOOP"].to_numpy(),
            "rule_down_minus_noop_inheritance": inherited["RULE_DOWN"].to_numpy()
            - inherited["NOOP"].to_numpy(),
            "model_up_minus_noop_inheritance": inherited["MODEL_UP"].to_numpy()
            - inherited["NOOP"].to_numpy(),
            "model_up_minus_model_down_episodes": episodes["MODEL_UP"].to_numpy()
            - episodes["MODEL_DOWN"].to_numpy(),
            "random_minus_noop_inheritance": inherited["RANDOM"].to_numpy()
            - inherited["NOOP"].to_numpy(),
        }
        contrast_summaries: dict[str, Any] = {}
        for name, values in contrasts.items():
            boot = values[bootstrap].mean(axis=1)
            direction = "negative" if name == "model_up_minus_noop_inheritance" else "positive"
            raw_p, null = _sign_p(values, signs, direction)
            stored[f"c{candidate}__matrix__{name}"] = values
            stored[f"c{candidate}__bootstrap__{name}"] = boot
            stored[f"c{candidate}__randomization__{name}"] = null
            contrast_summaries[name] = {
                "estimate": float(values.mean()),
                "bootstrap_ci95": _interval(boot),
                "bootstrap_ci90": _interval(boot, alpha=0.10),
                "descriptive_one_sided_randomization_p": raw_p,
                "matrices_positive": int(np.count_nonzero(values > 0)),
                "matrices_negative": int(np.count_nonzero(values < 0)),
                "matrices_zero": int(np.count_nonzero(values == 0)),
                "maximum_leave_one_matrix_out_influence": _maximum_leave_one_out_influence(values),
            }
        arm_means: dict[str, Any] = {}
        for arm in ARMS:
            arm_means[arm] = {}
            arm_rows = selected[selected["controller"] == arm].sort_values("matrix_id")
            for outcome in (
                "inherited_fraction",
                "total_breaks",
                "episode_count",
                "longest_inherited_run",
                "completed_horizon",
                "final_entropy",
                "final_occupied_types",
                "final_top1_share",
                "final_throughput",
                "mean_growth_updates",
                "cross_lineage_final_cosine",
                "distinct_swaps",
                "repeated_swaps",
                "immediately_reversing_swaps",
                "out_of_development_envelope_fraction",
                "final_risk",
                "mean_predicted_action_shift",
            ):
                values = arm_rows[outcome].to_numpy(dtype=np.float64)
                boot = values[bootstrap].mean(axis=1)
                arm_means[arm][outcome] = {
                    "mean": float(values.mean()),
                    "bootstrap_ci95": _interval(boot),
                }
        random_ci90 = contrast_summaries["random_minus_noop_inheritance"]["bootstrap_ci90"]
        gates = {
            "model_down_above_noop": contrast_summaries[
                "model_down_minus_noop_inheritance"
            ]["bootstrap_ci95"][0]
            > 0.0,
            "rule_down_above_noop": contrast_summaries[
                "rule_down_minus_noop_inheritance"
            ]["bootstrap_ci95"][0]
            > 0.0,
            "model_up_below_noop": contrast_summaries[
                "model_up_minus_noop_inheritance"
            ]["bootstrap_ci95"][1]
            < 0.0,
            "model_up_more_episodes_than_model_down": contrast_summaries[
                "model_up_minus_model_down_episodes"
            ]["bootstrap_ci95"][0]
            > 0.0,
            "random_equivalent_to_noop": random_ci90[0]
            > -RANDOM_EQUIVALENCE_MARGIN
            and random_ci90[1] < RANDOM_EQUIVALENCE_MARGIN,
        }
        numerator = contrasts["rule_down_minus_noop_inheritance"]
        denominator = contrasts["model_down_minus_noop_inheritance"]
        numerator_boot = numerator[bootstrap].mean(axis=1)
        denominator_boot = denominator[bootstrap].mean(axis=1)
        valid = denominator_boot > 0.0
        ratio_boot = numerator_boot[valid] / denominator_boot[valid]
        estimate = float(numerator.mean() / denominator.mean()) if denominator.mean() > 0 else float("nan")
        ratio_ci = _interval(ratio_boot) if ratio_boot.size else (float("nan"), float("nan"))
        valid_fraction = float(valid.mean())
        recovery = {
            "estimate": estimate,
            "bootstrap_ci95": ratio_ci,
            "valid_bootstrap_fraction": valid_fraction,
            "point_at_least_0_80": bool(np.isfinite(estimate) and estimate >= 0.80),
            "lower_bound_above_0_70": bool(np.isfinite(ratio_ci[0]) and ratio_ci[0] > 0.70),
        }
        recovery["strong_external_replication"] = bool(
            recovery["point_at_least_0_80"]
            and recovery["lower_bound_above_0_70"]
            and valid_fraction >= 0.95
        )
        stored[f"c{candidate}__rule_recovery_bootstrap_valid"] = ratio_boot
        candidate_pass = bool(all(gates.values()))
        candidates.append(
            {
                "candidate": candidate,
                "matrices": MATRICES,
                "replicates_per_controller": REPLICATES,
                "arm_means": arm_means,
                "contrasts": contrast_summaries,
                "gates": gates,
                "candidate_primary_gate": candidate_pass,
                "rule_recovery_fraction": recovery,
            }
        )
    all_candidate_gates = bool(all(item["candidate_primary_gate"] for item in candidates))
    complete = bool(all_candidate_gates and replay_exact and noop_plain_exact)
    return (
        {
            "format": "codex-intervention-cr7-primary-metrics-v1",
            "candidates": candidates,
            "all_candidate_primary_gates": all_candidate_gates,
            "noop_callback_plain_bitwise_exact": bool(noop_plain_exact),
            "complete_exact_replay": bool(replay_exact),
            "complete_cr7_60_fission_gate": complete,
            "conditional_extension_authorized": complete,
            "candidates_never_pooled": True,
            "whole_matrix_inference": True,
        },
        stored,
    )


def replay_audit(
    generated: list[SteeringBatch], replayed: list[SteeringBatch]
) -> dict[str, Any]:
    if len(generated) != len(replayed):
        raise ValueError("CR7 replay batch count differs")
    rows = []
    for left, right in zip(generated, replayed, strict=True):
        exact = batch_digest(left) == batch_digest(right)
        rows.append(
            {
                "state_id": left.state_id,
                "candidate": left.candidate,
                "matrix_id": left.matrix_id,
                "exact": exact,
                "generated_digest": batch_digest(left),
                "replay_digest": batch_digest(right),
            }
        )
    return {
        "format": "codex-intervention-cr7-replay-audit-v1",
        "state_batches": len(rows),
        "exact_state_edit_endpoint_process_and_rng": bool(all(row["exact"] for row in rows)),
        "rows": rows,
    }


def _pack_arrays(
    cases: list[StateCase], batches: list[SteeringBatch]
) -> dict[str, NDArray]:
    shape = (len(cases), REPLICATES, len(ARMS), HORIZON)
    h = np.full(shape, np.nan, dtype=np.float64)
    growth = np.full(shape, -1, dtype=np.int32)
    risk_before = np.full(shape, np.nan, dtype=np.float64)
    risk_after = np.full(shape, np.nan, dtype=np.float64)
    ood = np.full(shape, -1, dtype=np.int8)
    actions = np.full(shape + (2,), -1, dtype=np.int16)
    final = np.zeros((len(cases), REPLICATES, len(ARMS), cases[0].beta.shape[0]), dtype=np.int64)
    completed = np.zeros((len(cases), REPLICATES, len(ARMS)), dtype=np.int8)
    observed = np.zeros_like(completed, dtype=np.int16)
    arm_index = {name: index for index, name in enumerate(ARMS)}
    for case_index, batch in enumerate(batches):
        for lineage in batch.lineages:
            index = (case_index, lineage.replicate, arm_index[lineage.controller])
            h[index] = lineage.boundary_h
            growth[index] = lineage.growth_updates
            count = lineage.observed_fissions
            risk_before[index][:count] = lineage.risk_before
            risk_after[index][:count] = lineage.risk_after
            ood[index][:count] = lineage.out_of_envelope
            for step, edit in enumerate(lineage.actions):
                actions[index + (step,)] = (edit.remove_type, edit.add_type)
            final[index] = lineage.final_snapshot.composition
            completed[index] = int(lineage.completed_horizon)
            observed[index] = count
    return {
        "boundary_h": h,
        "growth_updates": growth,
        "risk_before": risk_before,
        "risk_after": risk_after,
        "out_of_development_envelope": ood,
        "selected_edits": actions,
        "final_compositions": final,
        "completed_horizon": completed,
        "observed_fissions": observed,
        "candidate": np.asarray([case.candidate for case in cases]),
        "matrix_id": np.asarray([case.matrix_id for case in cases], dtype=np.int16),
        "arm_names": np.asarray(ARMS),
    }


def _run_extension_case(
    case: StateCase,
    primary: SteeringBatch,
    current_experiment: ExperimentConfig,
    model_path: str | Path,
    envelope_path: str | Path,
    registration_id: str,
) -> SteeringBatch:
    predictor = FrozenFullPredictor.load(model_path)
    with np.load(envelope_path, allow_pickle=False) as archive:
        envelope = {name: archive[name] for name in archive.files}
    primary_lookup = {
        (lineage.controller, lineage.replicate): lineage
        for lineage in primary.lineages
    }
    lineages: list[LineageSummary] = []
    for replicate in range(REPLICATES):
        for name in EXTENSION_ARMS:
            previous = primary_lookup[(name, replicate)]
            rng = np.random.default_rng()
            rng.bit_generator.state = previous.simulation_rng_state
            action_rng = np.random.default_rng(
                derive_seed(
                    SEEDS["conditional_extension"],
                    f"{LABEL}.extension.action",
                    case.candidate,
                    case.matrix_id,
                    replicate,
                    name,
                )
            )
            callback, trace = _controller(
                name, predictor, current_experiment.gard, envelope, action_rng
            )
            result = simulate_controlled(
                previous.final_snapshot,
                case.beta,
                case.candidate,
                current_experiment,
                EXTENSION_HORIZON,
                rng,
                callback,
            )
            noop_exact = True
            if name == "NOOP":
                plain_rng = np.random.default_rng()
                plain_rng.bit_generator.state = previous.simulation_rng_state
                plain = simulate_controlled(
                    previous.final_snapshot,
                    case.beta,
                    case.candidate,
                    current_experiment,
                    EXTENSION_HORIZON,
                    plain_rng,
                    None,
                )
                noop_exact = _controlled_equal(result, plain, rng, plain_rng)
            lineages.append(
                _lineage_summary(
                    name,
                    replicate,
                    result,
                    trace,
                    case.beta,
                    predictor,
                    case.candidate,
                    current_experiment.gard,
                    EXTENSION_HORIZON,
                    rng.bit_generator.state,
                    noop_exact,
                )
            )
    return SteeringBatch(
        format=CHECKPOINT_FORMAT,
        registration_id=registration_id,
        mode="conditional_active_extension",
        state_id=case.state_id,
        candidate=case.candidate,
        matrix_id=case.matrix_id,
        landmark=case.landmark,
        case_digest=hashlib.sha256(
            (_case_digest(case) + batch_digest(primary)).encode()
        ).hexdigest(),
        lineages=tuple(lineages),
    )


def _extension_worker(arguments: tuple[Any, ...]) -> SteeringBatch:
    limiter = threadpool_limits(limits=1)
    try:
        return _run_extension_case(*arguments)
    finally:
        limiter.restore_original_limits()


def _read_extension_checkpoint(
    path: Path,
    case: StateCase,
    primary: SteeringBatch,
    registration_id: str,
) -> SteeringBatch | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            batch = pickle.load(handle)
    except Exception:
        return None
    expected_digest = hashlib.sha256(
        (_case_digest(case) + batch_digest(primary)).encode()
    ).hexdigest()
    if not isinstance(batch, SteeringBatch):
        return None
    if (
        batch.format != CHECKPOINT_FORMAT
        or batch.registration_id != registration_id
        or batch.mode != "conditional_active_extension"
        or batch.state_id != case.state_id
        or batch.case_digest != expected_digest
        or len(batch.lineages) != REPLICATES * len(EXTENSION_ARMS)
    ):
        return None
    return batch


def run_extension_batches(
    cases: list[StateCase],
    primary: list[SteeringBatch],
    current_experiment: ExperimentConfig,
    model_path: Path,
    envelope_path: Path,
    registration_id: str,
    directory: Path,
    workers: int,
    work: Path,
    stage: str,
) -> list[SteeringBatch]:
    directory.mkdir(parents=True, exist_ok=True)
    batches: dict[str, SteeringBatch] = {}
    missing: list[tuple[StateCase, SteeringBatch]] = []
    for case, parent in zip(cases, primary, strict=True):
        checkpoint = _read_extension_checkpoint(
            _checkpoint_path(directory, case), case, parent, registration_id
        )
        if checkpoint is None:
            missing.append((case, parent))
        else:
            batches[case.state_id] = checkpoint
    _write_status(work, stage, len(batches), len(cases), reused=len(batches))
    arguments = [
        (case, parent, current_experiment, model_path, envelope_path, registration_id)
        for case, parent in missing
    ]
    if workers == 1:
        for argument, (case, _parent) in zip(arguments, missing):
            batch = _extension_worker(argument)
            _write_checkpoint(_checkpoint_path(directory, case), batch)
            batches[case.state_id] = batch
            _write_status(work, stage, len(batches), len(cases))
            print(f"[{stage}] {len(batches)}/{len(cases)} state batches", flush=True)
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_case = {
                executor.submit(_extension_worker, argument): pair[0]
                for argument, pair in zip(arguments, missing)
            }
            for future in as_completed(future_case):
                case = future_case[future]
                batch = future.result()
                _write_checkpoint(_checkpoint_path(directory, case), batch)
                batches[case.state_id] = batch
                _write_status(work, stage, len(batches), len(cases))
                print(f"[{stage}] {len(batches)}/{len(cases)} state batches", flush=True)
    return [batches[case.state_id] for case in cases]


def extension_summary(
    cases: list[StateCase],
    primary: list[SteeringBatch],
    extension: list[SteeringBatch],
    draws: dict[str, NDArray],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    extension_lineage, extension_matrix, extension_edits = _lineage_and_matrix_tables(
        cases, extension
    )
    bootstrap = np.asarray(draws["bootstrap_indices"], dtype=np.int64)
    primary_lookup = {
        (batch.candidate, batch.matrix_id, item.controller, item.replicate): item
        for batch in primary
        for item in batch.lineages
    }
    extension_lookup = {
        (batch.candidate, batch.matrix_id, item.controller, item.replicate): item
        for batch in extension
        for item in batch.lineages
    }
    combined_rows: list[dict[str, Any]] = []
    for key, latter in extension_lookup.items():
        former = primary_lookup[key]
        observed = former.observed_fissions + latter.observed_fissions
        inherited = former.inherited_boundary_count + latter.inherited_boundary_count
        combined_rows.append(
            {
                "candidate": key[0],
                "matrix_id": key[1],
                "controller": key[2],
                "replicate": key[3],
                "observed_fissions": observed,
                "inherited_fraction": float(inherited / observed) if observed else 0.0,
                "completed_120": int(
                    former.completed_horizon and latter.completed_horizon
                ),
            }
        )
    combined_lineage = pd.DataFrame(combined_rows)
    combined_matrix = (
        combined_lineage.groupby(["candidate", "matrix_id", "controller"], as_index=False)
        .mean(numeric_only=True)
    )
    candidate_summaries: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        segment = extension_matrix[
            extension_matrix["candidate"].astype(str).str.zfill(2) == candidate
        ]
        combined = combined_matrix[
            combined_matrix["candidate"].astype(str).str.zfill(2) == candidate
        ]
        segment_pivot = segment.pivot(
            index="matrix_id", columns="controller", values="inherited_fraction"
        ).reindex(index=np.arange(MATRICES), columns=EXTENSION_ARMS)
        combined_pivot = combined.pivot(
            index="matrix_id", columns="controller", values="inherited_fraction"
        ).reindex(index=np.arange(MATRICES), columns=EXTENSION_ARMS)
        if segment_pivot.isna().any().any() or combined_pivot.isna().any().any():
            raise ValueError("CR7 extension lost a whole-matrix controller block")
        item: dict[str, Any] = {"candidate": candidate, "active_control": True}
        for label, pivot in (("fissions_61_120", segment_pivot), ("fissions_1_120", combined_pivot)):
            values: dict[str, Any] = {"arm_means": {}, "contrasts": {}}
            for arm in EXTENSION_ARMS:
                matrix_values = pivot[arm].to_numpy(dtype=np.float64)
                boot = matrix_values[bootstrap].mean(axis=1)
                values["arm_means"][arm] = {
                    "mean_inherited_fraction": float(matrix_values.mean()),
                    "bootstrap_ci95": _interval(boot),
                }
            for contrast, up, down in (
                ("model_down_minus_noop", "MODEL_DOWN", "NOOP"),
                ("rule_down_minus_noop", "RULE_DOWN", "NOOP"),
            ):
                difference = pivot[up].to_numpy(dtype=np.float64) - pivot[down].to_numpy(dtype=np.float64)
                boot = difference[bootstrap].mean(axis=1)
                values["contrasts"][contrast] = {
                    "estimate": float(difference.mean()),
                    "bootstrap_ci95": _interval(boot),
                }
            item[label] = values
        candidate_summaries.append(item)
    return (
        {
            "format": "codex-intervention-cr7-active-extension-v1",
            "launched_because_primary_gate_passed": True,
            "additional_fissions": EXTENSION_HORIZON,
            "active_feedback_not_passive_persistence": True,
            "candidates": candidate_summaries,
        },
        extension_lineage,
        extension_edits,
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
    for directory in (CR0_VALIDATION, CR1_REGISTRATION, CR1_RESULT, CR3_REGISTRATION, CR3_RESULT):
        verify_checksums(directory)
    cr0 = json.loads((CR0_VALIDATION / "validation.json").read_text())
    cr1_registration = json.loads((CR1_REGISTRATION / "registration.json").read_text())
    cr1_result = json.loads((CR1_RESULT / "manifest.json").read_text())
    cr3_registration = json.loads((CR3_REGISTRATION / "registration.json").read_text())
    cr3_result = json.loads((CR3_RESULT / "manifest.json").read_text())
    if not cr0["all_checks_passed"]:
        raise ValueError("inherited CR0 validation is not passing")
    if cr1_registration["registration_id"] != CR1_REGISTRATION_ID:
        raise ValueError("CR1 registration changed")
    if not (
        cr1_result["full_four_cell_gate"]
        and cr1_result["exact_replay"]
        and cr1_result["complete_readback_exact"]
    ):
        raise ValueError("CR1 no longer authorizes CR7")
    if cr3_registration["registration_id"] != CR3_REGISTRATION_ID:
        raise ValueError("CR3 registration changed")
    if not (
        cr3_result["full_four_cell_cr3_gate"]
        and cr3_result["exact_replay"]
        and cr3_result["complete_readback_exact"]
        and cr3_result["rule_expression"] == "x @ beta == beta.T @ x"
    ):
        raise ValueError("CR3 outgoing rule no longer authorizes CR7")
    model_path = CR1_REGISTRATION / "frozen_full_predictor.npz"
    if sha256_file(model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen JOINT_BREAK_RUN3 predictor changed")
    return {
        "cr0_checksum_manifest_sha256": sha256_file(CR0_VALIDATION / "SHA256SUMS"),
        "cr1_registration_checksum_manifest_sha256": sha256_file(CR1_REGISTRATION / "SHA256SUMS"),
        "cr1_result_checksum_manifest_sha256": sha256_file(CR1_RESULT / "SHA256SUMS"),
        "cr3_registration_checksum_manifest_sha256": sha256_file(CR3_REGISTRATION / "SHA256SUMS"),
        "cr3_result_checksum_manifest_sha256": sha256_file(CR3_RESULT / "SHA256SUMS"),
    }


def _artificial_case() -> tuple[StateCase, ExperimentConfig]:
    config = GardConfig(
        n_types=4,
        n_min=4,
        n_max=8,
        beta_log_mean=-4.0,
        beta_log_sd=1.0,
        max_growth_steps=10_000,
        generations=60,
    )
    current_experiment = ExperimentConfig(
        gard=config,
        development=CohortConfig(1, 1, (60,)),
        confirmation=CohortConfig(1, 1, (60,)),
        horizon=2,
        master_seed=SEEDS["smoke"],
    )
    beta = np.asarray(
        [
            [10.0, 4.0, 2.0, 1.0],
            [3.0, 11.0, 1.5, 2.0],
            [2.5, 1.0, 12.0, 3.0],
            [1.0, 2.0, 4.0, 9.0],
        ],
        dtype=np.float64,
    )
    snapshot = Snapshot(
        composition=np.asarray([1, 1, 1, 1], dtype=np.int64),
        generation=60,
        inheritance=(True, False, True),
        boundary_h=(0.95, 0.8, 0.93),
        previous_growth_steps=17,
        cumulative_growth_steps=900,
    )
    return (
        StateCase("CR7-ARTIFICIAL", "CR7-ARTIFICIAL", "02", 0, 60, beta, snapshot),
        current_experiment,
    )


def validation_checks() -> dict[str, Any]:
    upstream = _verify_upstream()
    predictor = FrozenFullPredictor.load(CR1_REGISTRATION / "frozen_full_predictor.npz")
    envelope = development_envelope(predictor)
    model_audit, _ = base._model_prediction_audit()
    fixture_h = np.asarray([0.8, 0.91, 0.92, 0.93, 0.7, 0.95, 0.96, 0.97])
    case, artificial_experiment = _artificial_case()
    artificial_envelope = {
        "c02__minimum": np.full(21, -np.inf),
        "c02__maximum": np.full(21, np.inf),
    }
    callback, _trace = _controller(
        "NOOP",
        predictor,
        artificial_experiment.gard,
        artificial_envelope,
        np.random.default_rng(1),
    )
    left_rng = np.random.default_rng(derive_seed(SEEDS["validation"], "noop"))
    right_rng = np.random.default_rng(derive_seed(SEEDS["validation"], "noop"))
    left = simulate_controlled(
        case.snapshot,
        case.beta,
        case.candidate,
        artificial_experiment,
        2,
        left_rng,
        callback,
    )
    right = simulate_controlled(
        case.snapshot,
        case.beta,
        case.candidate,
        artificial_experiment,
        2,
        right_rng,
        None,
    )
    rule = select_outgoing_rule_edits(case.snapshot.composition, case.beta)
    influence = (case.snapshot.composition / case.snapshot.composition.sum()) @ case.beta
    down_difference = influence[rule["RULE_DOWN"].add_type] - influence[
        rule["RULE_DOWN"].remove_type
    ]
    up_difference = influence[rule["RULE_UP"].add_type] - influence[
        rule["RULE_UP"].remove_type
    ]
    checks = {
        "inherited_cr0_all_checks_pass": True,
        "cr1_model_guided_gate_replay_and_readback_pass": True,
        "cr3_outgoing_rule_gate_replay_and_readback_pass": True,
        "frozen_model_hash_exact": sha256_file(CR1_REGISTRATION / "frozen_full_predictor.npz")
        == EXPECTED_MODEL_SHA256,
        "frozen_model_reproduces_archive": bool(model_audit["all_within_tolerance"]),
        "design_exact": MATRICES == 48
        and LANDMARK == 60
        and REPLICATES == 6
        and HORIZON == 60
        and ARMS
        == ("MODEL_UP", "MODEL_DOWN", "RULE_UP", "RULE_DOWN", "RANDOM", "NOOP"),
        "seed_domains_unique": len(SEEDS) == len(set(SEEDS.values())),
        "seed_domains_disjoint_from_prior_registrations": set(SEEDS.values()).isdisjoint(
            _prior_seed_values()
        ),
        "future_seed_controller_free": len({_future_seed(case, 0) for _arm in ARMS}) == 1,
        "random_action_seed_separate_from_future": _action_seed(case, 0)
        != _future_seed(case, 0),
        "noop_callback_plain_bitwise_exact": _controlled_equal(
            left, right, left_rng, right_rng
        ),
        "outgoing_rule_orientation_and_direction_exact": down_difference
        == max(
            influence[edit.add_type] - influence[edit.remove_type]
            for edit in enumerate_legal_edits(case.snapshot.composition)
        )
        and up_difference
        == min(
            influence[edit.add_type] - influence[edit.remove_type]
            for edit in enumerate_legal_edits(case.snapshot.composition)
        ),
        "episode_counter_nonoverlap_exact": count_nonoverlapping_episodes(fixture_h)
        == 2,
        "threshold_is_strict": count_nonoverlapping_episodes([0.9, 0.91, 0.92, 0.93])
        == 1,
        "longest_run_exact": longest_inherited_run(fixture_h) == 3,
        "development_envelope_dimensions_exact": all(
            envelope[f"c{candidate}__minimum"].shape == (21,)
            and envelope[f"c{candidate}__maximum"].shape == (21,)
            and np.all(
                envelope[f"c{candidate}__minimum"]
                <= envelope[f"c{candidate}__maximum"]
            )
            for candidate in CANDIDATES
        ),
        "whole_matrix_draw_counts_exact": inference_draws()["bootstrap_indices"].shape
        == (4096, 48)
        and inference_draws()["randomization_signs"].shape == (4096, 48),
        "conditional_extension_frozen_and_active": EXTENSION_ARMS
        == ("MODEL_DOWN", "RULE_DOWN", "NOOP")
        and EXTENSION_HORIZON == 60,
        "strict_eight_excluded": protocol()["target"]["strict_eight_excluded"],
    }
    return {
        "format": VALIDATION_FORMAT,
        "checks": checks,
        "check_count": len(checks),
        "all_checks_passed": bool(all(checks.values())),
        "upstream": upstream,
        "scientific_matrices_generated": 0,
        "scientific_controlled_lineages_generated": 0,
    }


def validate(output: Path = DEFAULT_VALIDATION) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    checks = validation_checks()
    if not checks["all_checks_passed"]:
        raise AssertionError(
            {key: value for key, value in checks["checks"].items() if not value}
        )
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "CR7 full repository validation failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    with _atomic_destination(output) as destination:
        payload = dict(checks)
        payload["source_hashes"] = source_hashes()
        payload["source_tree_sha256"] = _canonical_digest(payload["source_hashes"])
        (destination / "validation.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n"
        )
        (destination / "pytest_output.txt").write_text(
            "$ " + " ".join(command) + "\n" + completed.stdout + completed.stderr
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR7 validation sealed: {output}", flush=True)


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
        raise ValueError("CR7 validation did not pass")
    if validation["source_hashes"] != source_hashes():
        raise ValueError("CR7 source changed after validation")
    upstream = _verify_upstream()
    for forbidden in (DEFAULT_REGISTRATION, DEFAULT_SMOKE, DEFAULT_OUTPUT, DEFAULT_WORK):
        if forbidden.exists():
            raise FileExistsError(f"CR7 pre-registration artifact already exists: {forbidden}")
    predictor = FrozenFullPredictor.load(CR1_REGISTRATION / "frozen_full_predictor.npz")
    envelope = development_envelope(predictor)
    body: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol": protocol(),
        "protocol_id": protocol()["protocol_id"],
        "source_hashes": source_hashes(),
        "source_tree_sha256": _canonical_digest(source_hashes()),
        "seed_registry": SEEDS,
        "frozen_model_sha256": EXPECTED_MODEL_SHA256,
        "frozen_array_source_sha256": sha256_file(FROZEN_ARRAY_SOURCE),
        "validation_checksum_manifest_sha256": sha256_file(
            validation_directory / "SHA256SUMS"
        ),
        "upstream": upstream,
        "scientific_matrices_at_registration": 0,
        "scientific_controlled_lineages_at_registration": 0,
    }
    registration_id = _canonical_digest(_json_ready(body))
    body["registration_id"] = registration_id
    with _atomic_destination(output) as destination:
        shutil.copy2(ROOT / DOCUMENT, destination / "preregistration.md")
        shutil.copy2(validation_directory / "validation.json", destination / "validation.json")
        shutil.copy2(
            CR1_REGISTRATION / "frozen_full_predictor.npz",
            destination / "frozen_full_predictor.npz",
        )
        np.savez_compressed(destination / "development_envelope.npz", **envelope)
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
        f"<!-- registered-cr7-{registration_id} -->",
        [
            "## CR7 closed-loop hereditary steering registered",
            "",
            f"- Registration: `{registration_id}`.",
            "- Forty-eight fresh matrices, both candidates, six replicate lineages per controller, and 60 fissions were frozen before scientific generation.",
            "- Controllers, outcomes, whole-matrix inference, random equivalence margin, active-extension rule, seed domains, and claim boundaries are sealed.",
            "- CR6 remains a failed complete gate and did not tune or authorize CR7; CR1 and CR3 independently authorize this phase.",
            "- No CR7 scientific matrix or controlled lineage existed at this seal.",
            "",
        ],
    )
    print(f"CR7 registered: {registration_id}", flush=True)


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    registration = json.loads((directory / "registration.json").read_text())
    if registration["format"] != REGISTRATION_FORMAT:
        raise ValueError("unsupported CR7 registration format")
    if registration["source_hashes"] != source_hashes():
        raise ValueError("CR7 registered source tree changed")
    body = dict(registration)
    observed = body.pop("registration_id")
    if _canonical_digest(_json_ready(body)) != observed:
        raise ValueError("CR7 registration ID changed")
    if registration["protocol"] != protocol():
        raise ValueError("CR7 frozen protocol changed")
    if registration["seed_registry"] != SEEDS:
        raise ValueError("CR7 seed registry changed")
    if sha256_file(directory / "frozen_full_predictor.npz") != EXPECTED_MODEL_SHA256:
        raise ValueError("CR7 frozen predictor copy changed")
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
    case, current_experiment = _artificial_case()
    predictor = FrozenFullPredictor.load(registration_directory / "frozen_full_predictor.npz")
    envelope = {
        "c02__minimum": np.full(21, -np.inf),
        "c02__maximum": np.full(21, np.inf),
    }

    def execute() -> list[str]:
        digests: list[str] = []
        for arm in ARMS:
            simulation_seed = derive_seed(SEEDS["smoke"], "artificial.future")
            action_seed = derive_seed(SEEDS["smoke"], "artificial.action")
            rng = np.random.default_rng(simulation_seed)
            callback, trace = _controller(
                arm,
                predictor,
                current_experiment.gard,
                envelope,
                np.random.default_rng(action_seed),
            )
            result = simulate_controlled(
                case.snapshot,
                case.beta,
                case.candidate,
                current_experiment,
                2,
                rng,
                callback,
            )
            summary = _lineage_summary(
                arm,
                0,
                result,
                trace,
                case.beta,
                predictor,
                case.candidate,
                current_experiment.gard,
                2,
                rng.bit_generator.state,
                True,
            )
            digests.append(_lineage_digest(summary))
        return digests

    first = execute()
    second = execute()
    payload = {
        "format": "codex-intervention-cr7-smoke-v1",
        "registration_id": registration["registration_id"],
        "artificial_non_scientific_fixture": True,
        "controllers_exercised": len(first) == len(ARMS),
        "io_serialization_exercised": True,
        "exact_replay": first == second,
        "effect_sizes_event_rates_arm_order_and_candidate_differences_disclosed": False,
        "scientific_matrices_generated": 0,
    }
    if not all(
        payload[key]
        for key in ("artificial_non_scientific_fixture", "controllers_exercised", "exact_replay")
    ):
        raise AssertionError("CR7 artificial smoke failed")
    with _atomic_destination(output) as destination:
        (destination / "smoke.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR7 non-scientific smoke passed: {output}", flush=True)


def _reports(metrics: dict[str, Any], extension: dict[str, Any] | None) -> tuple[str, str]:
    technical = [
        "# CR7 closed-loop hereditary steering",
        "",
        f"Complete registered 60-fission CR7 gate: **{metrics['complete_cr7_60_fission_gate']}**.",
        f"Exact replay: **{metrics['complete_exact_replay']}**.",
        f"No-op callback/plain identity: **{metrics['noop_callback_plain_bitwise_exact']}**.",
        "",
        "| Candidate | Contrast | Estimate | 95% CI | Gate |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {
        "model_down_minus_noop_inheritance": "MODEL_DOWN − NOOP inheritance",
        "rule_down_minus_noop_inheritance": "RULE_DOWN − NOOP inheritance",
        "model_up_minus_noop_inheritance": "MODEL_UP − NOOP inheritance",
        "model_up_minus_model_down_episodes": "MODEL_UP − MODEL_DOWN episodes",
        "random_minus_noop_inheritance": "RANDOM − NOOP inheritance",
    }
    gate_names = {
        "model_down_minus_noop_inheritance": "model_down_above_noop",
        "rule_down_minus_noop_inheritance": "rule_down_above_noop",
        "model_up_minus_noop_inheritance": "model_up_below_noop",
        "model_up_minus_model_down_episodes": "model_up_more_episodes_than_model_down",
        "random_minus_noop_inheritance": "random_equivalent_to_noop",
    }
    for candidate in metrics["candidates"]:
        for key, label in labels.items():
            contrast = candidate["contrasts"][key]
            technical.append(
                f"| {candidate['candidate']} | {label} | {contrast['estimate']:+.6f} | "
                f"{contrast['bootstrap_ci95']} | {candidate['gates'][gate_names[key]]} |"
            )
        recovery = candidate["rule_recovery_fraction"]
        technical.append(
            f"| {candidate['candidate']} | RULE_DOWN recovery fraction | "
            f"{recovery['estimate']:+.6f} | {recovery['bootstrap_ci95']} | "
            f"strong={recovery['strong_external_replication']} |"
        )
    technical.extend(
        [
            "",
            "All confidence intervals use whole catalytic matrices as blocks; candidates were not pooled. Random equivalence uses the complete 90% interval inside +/-0.025, not merely an interval crossing zero.",
            "",
            (
                "The conditional second 60-fission active-control extension ran and is reported separately."
                if extension is not None
                else "The conditional active-control extension did not run because the complete 60-fission stabilization gate did not pass."
            ),
            "",
            "This phase tests externally maintained control while interventions continue. It does not test autonomous persistence after release.",
            "",
        ]
    )
    lay = [
        "# CR7 in plain language",
        "",
        "CR7 repeatedly watches each simulated assembly after every division and makes a one-molecule change. One controller tries to make future break-and-renewal more likely, one tries to make it less likely, a simple chemistry-based rule acts in both directions, and random and no-change controls show what untargeted editing does.",
        "",
        (
            "Every predeclared steering test passed in both simulator candidates."
            if metrics["complete_cr7_60_fission_gate"]
            else "The campaign did not pass every predeclared steering test in both simulator candidates. The successful and failed tests remain separate in the technical report."
        ),
        "The exact replay and no-op checks determine whether this conclusion is technically trustworthy; their results are shown above and in the audit files.",
        "",
        "Even a positive result would mean that an outside controller can maintain a pattern while it keeps acting. It would not yet mean the assembly remembers or repairs itself after the controller is removed.",
        "",
    ]
    return "\n".join(technical), "\n".join(lay)


def _write_result(
    output: Path,
    registration: dict[str, Any],
    cases: list[StateCase],
    generated: list[SteeringBatch],
    replay: dict[str, Any],
    metrics: dict[str, Any],
    stored_inference: dict[str, NDArray],
    extension_metrics: dict[str, Any] | None,
    extension_lineage: pd.DataFrame | None,
    extension_edits: pd.DataFrame | None,
    extension_replay: dict[str, Any] | None,
) -> None:
    lineage, matrix, edits = _lineage_and_matrix_tables(cases, generated)
    arrays = _pack_arrays(cases, generated)
    technical, lay = _reports(metrics, extension_metrics)
    supported = []
    if metrics["complete_cr7_60_fission_gate"]:
        supported.append(
            "repeated state-dependent molecular intervention externally steers hereditary stability for 60 fissions in both Codex candidates"
        )
    if all(
        item["rule_recovery_fraction"]["strong_external_replication"]
        for item in metrics["candidates"]
    ):
        supported.append(
            "the outgoing catalytic-support rule strongly recovers the model-guided maintenance gain under the registered criterion"
        )
    failed = [
        f"candidate {item['candidate']} complete steering gate"
        for item in metrics["candidates"]
        if not item["candidate_primary_gate"]
    ]
    claims = {
        "supported": supported,
        "failed_predictions": failed,
        "unresolved": [
            "passive persistence after controller release",
            "return after challenge and autonomous basin radius",
            "control half-life and minimum feedback frequency",
            "transfer of long-horizon feedback beyond the home regime",
        ],
        "prohibited": protocol()["claim_boundary"]["prohibited"],
        "maintained_not_installed_boundary": True,
    }
    beta_by_matrix = np.stack(
        [
            next(case.beta for case in cases if case.matrix_id == matrix_id)
            for matrix_id in range(MATRICES)
        ]
    )
    launch_compositions = np.stack([case.snapshot.composition for case in cases])
    with _atomic_destination(output) as destination:
        (destination / "primary_metrics.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n"
        )
        (destination / "secondary_outcomes.json").write_text(
            json.dumps(
                {
                    "candidate_arm_summaries": [
                        {
                            "candidate": item["candidate"],
                            "arm_means": item["arm_means"],
                        }
                        for item in metrics["candidates"]
                    ]
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (destination / "replay_audit.json").write_text(
            json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n"
        )
        (destination / "SCIENTIFIC_REPORT.md").write_text(technical)
        (destination / "LAY_SUMMARY.md").write_text(lay)
        (destination / "claim_boundaries.json").write_text(
            json.dumps(claims, indent=2, sort_keys=True) + "\n"
        )
        lineage.to_csv(destination / "lineages.csv.gz", index=False, compression="gzip")
        matrix.to_csv(destination / "matrix_effects.csv", index=False)
        edits.to_csv(destination / "selected_edits.csv.gz", index=False, compression="gzip")
        np.savez_compressed(destination / "lineage_arrays.npz", **arrays)
        np.savez_compressed(destination / "inference_arrays.npz", **stored_inference)
        np.savez_compressed(
            destination / "state_and_matrix_arrays.npz",
            beta=beta_by_matrix,
            launch_compositions=launch_compositions,
            candidate=np.asarray([case.candidate for case in cases]),
            matrix_id=np.asarray([case.matrix_id for case in cases], dtype=np.int16),
            landmark=np.asarray([case.landmark for case in cases], dtype=np.int16),
        )
        if extension_metrics is not None:
            extension_directory = destination / "conditional_active_extension"
            extension_directory.mkdir()
            (extension_directory / "metrics.json").write_text(
                json.dumps(_json_ready(extension_metrics), indent=2, sort_keys=True) + "\n"
            )
            (extension_directory / "replay_audit.json").write_text(
                json.dumps(_json_ready(extension_replay), indent=2, sort_keys=True) + "\n"
            )
            assert extension_lineage is not None and extension_edits is not None
            extension_lineage.to_csv(
                extension_directory / "lineages.csv.gz", index=False, compression="gzip"
            )
            extension_edits.to_csv(
                extension_directory / "selected_edits.csv.gz", index=False, compression="gzip"
            )
        expected_metrics_text = (
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n"
        )
        readback_metrics_text = (destination / "primary_metrics.json").read_text()
        readback_lineage = pd.read_csv(destination / "lineages.csv.gz")
        readback_matrix = pd.read_csv(destination / "matrix_effects.csv")
        with np.load(destination / "lineage_arrays.npz", allow_pickle=False) as archive:
            array_shapes_exact = archive["boundary_h"].shape == (
                len(cases),
                REPLICATES,
                len(ARMS),
                HORIZON,
            ) and archive["final_compositions"].shape[-1] == cases[0].beta.shape[0]
        readback = {
            "primary_metrics_exact": readback_metrics_text == expected_metrics_text,
            "lineage_row_count_exact": len(readback_lineage)
            == len(cases) * REPLICATES * len(ARMS),
            "matrix_row_count_exact": len(readback_matrix)
            == MATRICES * len(CANDIDATES) * len(ARMS),
            "array_shapes_exact": bool(array_shapes_exact),
        }
        readback["complete_readback_exact"] = bool(all(readback.values()))
        if not readback["complete_readback_exact"]:
            raise AssertionError(f"CR7 written-artifact readback failed: {readback}")
        (destination / "readback_audit.json").write_text(
            json.dumps(readback, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "controllers": list(ARMS),
            "replicates_per_controller": REPLICATES,
            "fissions_per_primary_lineage": HORIZON,
            "primary_controlled_lineages": len(cases) * REPLICATES * len(ARMS),
            "maximum_primary_boundaries": len(cases) * REPLICATES * len(ARMS) * HORIZON,
            "replay_controlled_lineages": len(cases) * REPLICATES * len(ARMS),
            "complete_cr7_60_fission_gate": metrics["complete_cr7_60_fission_gate"],
            "exact_replay": metrics["complete_exact_replay"],
            "noop_callback_plain_bitwise_exact": metrics[
                "noop_callback_plain_bitwise_exact"
            ],
            "complete_readback_exact": True,
            "conditional_active_extension_launched": extension_metrics is not None,
            "conditional_extension_exact_replay": (
                extension_replay["exact_state_edit_endpoint_process_and_rng"]
                if extension_replay is not None
                else None
            ),
            "no_controlled_lineage_retry_or_matrix_replacement": True,
            "no_refit_recalibration_or_threshold_change": True,
            "cr8_and_cr9_launched": False,
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
        raise FileExistsError(f"refusing to overwrite completed CR7 result: {output}")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError(
            f"CR7 requires at least {MINIMUM_FREE_DISK_BYTES:,} free bytes"
        )
    work.mkdir(parents=True, exist_ok=True)
    contract_path = work / "campaign_contract.json"
    expected = {
        "format": "codex-intervention-cr7-work-contract-v1",
        "registration_id": registration_id,
        "protocol_id": protocol()["protocol_id"],
    }
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != expected:
            raise ValueError("CR7 work directory belongs to another campaign")
    else:
        contract_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")


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
    envelope_path = registration_directory / "development_envelope.npz"
    _write_status(work, "building_fresh_natural_launch_states", 0, 2 * MATRICES)
    print(f"[cr7 1/7] Building {MATRICES} fresh matrices and {2 * MATRICES} natural generation-60 states", flush=True)
    with threadpool_limits(limits=1):
        cases = build_cr7_cohort(current_experiment)
    if len(cases) != 2 * MATRICES:
        raise AssertionError("CR7 fresh cohort is incomplete")
    _write_status(work, "primary_controlled_lineages", 0, len(cases))
    print(
        f"[cr7 2/7] Running {len(cases) * REPLICATES * len(ARMS):,} primary controlled lineages",
        flush=True,
    )
    generated = run_batches(
        cases,
        current_experiment,
        model_path,
        envelope_path,
        registration["registration_id"],
        work / "primary" / "generate",
        workers,
        work,
        "primary_controlled_lineages",
    )
    _write_status(work, "complete_exact_replay", 0, len(cases))
    print("[cr7 3/7] Replaying every primary state, action, boundary, and RNG outcome", flush=True)
    replayed = run_batches(
        cases,
        current_experiment,
        model_path,
        envelope_path,
        registration["registration_id"],
        work / "primary" / "replay",
        workers,
        work,
        "complete_exact_replay",
    )
    replay = replay_audit(generated, replayed)
    if not replay["exact_state_edit_endpoint_process_and_rng"]:
        raise AssertionError("CR7 primary exact replay failed")
    noop_exact = bool(
        all(
            lineage.noop_plain_bitwise_exact
            for batch in generated
            for lineage in batch.lineages
            if lineage.controller == "NOOP"
        )
    )
    _write_status(work, "whole_matrix_inference", len(cases), len(cases))
    print("[cr7 4/7] Computing candidate-separated whole-matrix inference", flush=True)
    _lineage, matrix, _edits = _lineage_and_matrix_tables(cases, generated)
    draws = inference_draws()
    metrics, stored = compute_inference(
        matrix,
        draws,
        replay_exact=True,
        noop_plain_exact=noop_exact,
    )
    extension_metrics = None
    extension_lineage = None
    extension_edits = None
    extension_replay = None
    if metrics["conditional_extension_authorized"]:
        _write_status(work, "conditional_active_extension", 0, len(cases))
        print("[cr7 5/7] Primary gate passed; extending MODEL_DOWN, RULE_DOWN, and NOOP under active control", flush=True)
        extended = run_extension_batches(
            cases,
            generated,
            current_experiment,
            model_path,
            envelope_path,
            registration["registration_id"],
            work / "extension" / "generate",
            workers,
            work,
            "conditional_active_extension",
        )
        extended_replay = run_extension_batches(
            cases,
            generated,
            current_experiment,
            model_path,
            envelope_path,
            registration["registration_id"],
            work / "extension" / "replay",
            workers,
            work,
            "conditional_active_extension_replay",
        )
        extension_replay = replay_audit(extended, extended_replay)
        if not extension_replay["exact_state_edit_endpoint_process_and_rng"]:
            raise AssertionError("CR7 active-extension exact replay failed")
        extension_metrics, extension_lineage, extension_edits = extension_summary(
            cases, generated, extended, draws
        )
    else:
        print("[cr7 5/7] Primary gate did not pass; conditional active extension not launched", flush=True)
    _write_status(work, "writing_and_reading_back_artifacts", len(cases), len(cases))
    print("[cr7 6/7] Writing machine-readable artifacts and exact readback audit", flush=True)
    _write_result(
        output,
        registration,
        cases,
        generated,
        replay,
        metrics,
        stored,
        extension_metrics,
        extension_lineage,
        extension_edits,
        extension_replay,
    )
    _append_ledger(
        f"<!-- sealed-cr7-{registration['registration_id']} -->",
        [
            "## CR7 closed-loop hereditary steering sealed",
            "",
            f"- Registration: `{registration['registration_id']}`.",
            f"- Result: `{output.relative_to(ROOT)}`.",
            f"- Complete registered 60-fission gate: **{metrics['complete_cr7_60_fission_gate']}**.",
            f"- Exact replay: **{metrics['complete_exact_replay']}**; no-op callback/plain identity: **{metrics['noop_callback_plain_bitwise_exact']}**.",
            f"- Conditional continued-active-control extension launched: **{extension_metrics is not None}**.",
            "- CR8 and CR9 were not launched automatically; mandatory review stop observed.",
            "",
        ],
    )
    _write_status(
        work,
        "sealed_complete_mandatory_review_stop",
        len(cases),
        len(cases),
        output=str(output),
        complete_cr7_gate=metrics["complete_cr7_60_fission_gate"],
        extension_launched=extension_metrics is not None,
    )
    print("[cr7 7/7] Result sealed; STOPPED before CR8/CR9", flush=True)


def read_status(work: Path = DEFAULT_WORK) -> dict[str, Any]:
    work = work.resolve()
    status_path = work / "campaign_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"CR7 status does not exist: {status_path}")
    value = json.loads(status_path.read_text())
    value["checkpoint_counts"] = {
        relative: len(list((work / relative).glob("*.pkl")))
        if (work / relative).is_dir()
        else 0
        for relative in (
            "primary/generate",
            "primary/replay",
            "extension/generate",
            "extension/replay",
        )
    }
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
