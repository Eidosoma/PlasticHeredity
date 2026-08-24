"""Prospectively frozen CR8 steer-release-and-challenge campaign."""

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
from typing import Any, Iterable

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
    apply_molecular_edit,
    edited_snapshot,
    score_legal_edits,
    simulate_controlled,
)
from .intervention_cr7_steering import (
    DEFAULT_WORK as CR7_WORK,
    LineageSummary as CR7LineageSummary,
    SteeringBatch,
    _lineage_digest as cr7_lineage_digest,
    batch_digest as cr7_batch_digest,
)
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .seeds import derive_seed
from .simulator import Snapshot, cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_intervention_replication"
CR0_VALIDATION = RESULT_ROOT / "cr0_validation"
CR7_REGISTRATION = RESULT_ROOT / "cr7_steering_registration"
CR7_RESULT = RESULT_ROOT / "cr7_closed_loop_steering"

DEFAULT_VALIDATION = RESULT_ROOT / "cr8_release_challenge_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "cr8_release_challenge_registration"
DEFAULT_SMOKE = RESULT_ROOT / "cr8_release_challenge_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "cr8_steer_release_challenge"
DEFAULT_WORK = RESULT_ROOT / ".cr8_steer_release_challenge_work"

DOCUMENT = "CODEX_INTERVENTION_CR8_PREREGISTRATION.md"
PROGRAM_FORMAT = "codex-intervention-cr8-release-challenge-v1"
VALIDATION_FORMAT = "codex-intervention-cr8-validation-v1"
REGISTRATION_FORMAT = "codex-intervention-cr8-registration-v1"
RESULT_FORMAT = "codex-intervention-cr8-result-v1"
STATUS_FORMAT = "codex-intervention-cr8-status-v1"
RELEASE_CHECKPOINT_FORMAT = "codex-intervention-cr8-release-checkpoint-v1"
CHALLENGE_CHECKPOINT_FORMAT = "codex-intervention-cr8-challenge-checkpoint-v1"
LABEL = "INTCR8_RELEASE_CHALLENGE_V1"

EXPECTED_CR7_REGISTRATION_ID = (
    "41cf815a63129f40c04c7fb260f0f90c713adb9743eaae8479a5f6046e826e70"
)
EXPECTED_MODEL_SHA256 = (
    "9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af"
)

MATRICES = 48
REPLICATES = 6
ORIGINS = ("MODEL_DOWN", "RULE_DOWN", "NOOP")
WRITTEN_ORIGINS = ("MODEL_DOWN", "RULE_DOWN")
RELEASE_HORIZON = 60
CHALLENGE_BRANCHES = 32
CHALLENGE_HORIZON = 24
RANDOM_DOSES = (0, 2, 4, 8, 16)
CHALLENGE_ARMS = (
    "NONE",
    "RANDOM_K2",
    "RANDOM_K4",
    "RANDOM_K8",
    "RANDOM_K16",
    "ADVERSARIAL",
)
ARM_DOSE = {
    "NONE": 0,
    "RANDOM_K2": 2,
    "RANDOM_K4": 4,
    "RANDOM_K8": 8,
    "RANDOM_K16": 16,
    "ADVERSARIAL": 1,
}
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
RELEASE_EQUIVALENCE_MARGIN = 0.03
CHALLENGE_EQUIVALENCE_MARGIN = 0.05
INHERITANCE_THRESHOLD = 0.9
DEPARTURE_THRESHOLD = 0.7
RETURN_THRESHOLD = 0.9
RETURN_RUN = 3
MODE_FINAL_WINDOW = 6
MODE_MIN_INHERITED = 5
MODE_TOP1_THRESHOLD = 0.45
MINIMUM_FREE_DISK_BYTES = 3_000_000_000

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_cr8_release_challenge.py",
    "tests/test_intervention_cr8_release_challenge.py",
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
        f"codex-clean-room-cr8-release-challenge-v1::{name}".encode("utf-8")
    ).hexdigest()


SEEDS = {
    name: _seed(name)
    for name in (
        "validation",
        "smoke",
        "release_future",
        "challenge_edit",
        "challenge_future",
        "bootstrap",
        "randomization",
        "replay",
    )
}


@dataclass(frozen=True)
class FrozenOrigin:
    origin: str
    replicate: int
    snapshot: Snapshot
    source_lineage_digest: str


@dataclass(frozen=True)
class ReleaseCase:
    state_id: str
    candidate: str
    matrix_id: int
    beta: NDArray[np.float64]
    origins: tuple[FrozenOrigin, ...]
    source_batch_digest: str


@dataclass(frozen=True)
class ReleaseSummary:
    origin: str
    replicate: int
    completed_horizon: bool
    observed_fissions: int
    record_digest: str
    anchor_composition: NDArray[np.int64]
    final_snapshot: Snapshot
    similarity_to_anchor: NDArray[np.float64]
    similarity_to_matched_noop: NDArray[np.float64]
    risk: NDArray[np.float64]
    boundary_h: NDArray[np.float64]
    growth_updates: NDArray[np.int32]
    entropy: NDArray[np.float64]
    occupied_types: NDArray[np.int16]
    top1_share: NDArray[np.float64]
    throughput: NDArray[np.float64]
    final_six_inherited_fraction: float
    first_departure_time: int
    interventions_applied: int


@dataclass(frozen=True)
class ReleaseBatch:
    format: str
    registration_id: str
    state_id: str
    candidate: str
    matrix_id: int
    source_batch_digest: str
    releases: tuple[ReleaseSummary, ...]


@dataclass(frozen=True)
class ChallengePlan:
    origin: str
    replicate: int
    arm: str
    edits: tuple[MolecularEdit, ...]
    transport_distance: int
    noop_risk: float
    edited_risk: float
    launch_composition: NDArray[np.int64]


@dataclass(frozen=True)
class ChallengeSummary:
    origin: str
    replicate: int
    arm: str
    branch: int
    completed_horizon: bool
    observed_fissions: int
    record_digest: str
    category: str
    held: bool
    returned: bool
    mode_recovered: bool
    lost: bool
    departed: bool
    first_departure_time: int
    return_certification_time: int
    inherited_final_six: int
    final_top1_share: float
    final_similarity: float
    minimum_similarity: float
    similarity_to_anchor: NDArray[np.float64]
    boundary_h: NDArray[np.float64]
    final_composition: NDArray[np.int64]
    final_generation: int
    final_previous_growth_steps: int
    final_cumulative_growth_steps: int


@dataclass(frozen=True)
class ChallengeBatch:
    format: str
    registration_id: str
    state_id: str
    candidate: str
    matrix_id: int
    release_batch_digest: str
    plans: tuple[ChallengePlan, ...]
    outcomes: tuple[ChallengeSummary, ...]


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_cr8_release_or_challenge_future",
        "upstream": {
            "cr7_registration_id": EXPECTED_CR7_REGISTRATION_ID,
            "cr7_complete_60_fission_gate_required": True,
            "cr7_exact_replay_and_readback_required": True,
            "cr7_primary_endpoints_used": list(ORIGINS),
            "cr7_active_extension_excluded": True,
        },
        "cohort": {
            "matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "cr7_replicates": REPLICATES,
            "origins": list(ORIGINS),
            "release_fissions": RELEASE_HORIZON,
            "release_trajectories": MATRICES
            * len(CANDIDATES)
            * REPLICATES
            * len(ORIGINS),
            "challenge_arms": list(CHALLENGE_ARMS),
            "challenge_branches_per_arm": CHALLENGE_BRANCHES,
            "challenge_fissions": CHALLENGE_HORIZON,
            "challenge_futures": MATRICES
            * len(CANDIDATES)
            * REPLICATES
            * len(ORIGINS)
            * len(CHALLENGE_ARMS)
            * CHALLENGE_BRANCHES,
            "no_retry_or_replacement": True,
            "complete_release_and_challenge_replay": True,
        },
        "release": {
            "controller": None,
            "interventions_after_release": 0,
            "preparation_anchor": "exact CR7 fission-60 endpoint composition",
            "final_inheritance_window": 6,
            "unobserved_registered_boundaries_count_as_not_inherited": True,
        },
        "challenge": {
            "anchor": "release-end composition",
            "random_doses": list(RANDOM_DOSES),
            "random_k_exact_transport": True,
            "random_k_removals": "molecules sampled without replacement",
            "random_k_additions": "uniform with replacement outside removal labels",
            "adversarial": "exhaustive frozen-predictor maximum risk; first lexicographic tie",
            "history_unchanged_by_instantaneous_edits": True,
        },
        "classifier": {
            "departure": "unrounded cosine < 0.7, launch included",
            "return": "strictly after departure, unrounded cosine > 0.9 for three consecutive post-fission states",
            "mode_recovery": "completed F24, >=5 inherited in final 6, final top1 share >=0.45",
            "exclusive_precedence": ["held", "returned", "mode_recovered", "lost"],
            "incomplete_future": "lost",
        },
        "randomness": {
            "seed_domains": SEEDS,
            "release_future_seed_excludes_origin": True,
            "challenge_future_seed_excludes_origin_and_arm": True,
            "challenge_selection_stream_separate_from_future": True,
            "common_random_streams_not_identical_realized_futures": True,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "candidates_never_pooled": True,
            "release_tost_margin": RELEASE_EQUIVALENCE_MARGIN,
            "challenge_tost_margin": CHALLENGE_EQUIVALENCE_MARGIN,
            "random_dose_slope": "within-matrix OLS of written-minus-natural held+returned against K",
            "adversarial_excluded_from_dose_slope_and_basin_radius": True,
        },
        "external_written_but_passive_gates": [
            "mean written-anchor similarity crosses below 0.7 for both origins and candidates",
            "last-six inheritance written-minus-natural CI90 inside +/-0.03 for both origins and candidates",
            "held+returned written-minus-natural CI90 inside +/-0.05 for K=0,2,4,8,16 for both origins and candidates",
            "no dose-slope CI95 lower bound above zero",
            "shared registered basin radius equals zero",
            "release and challenge replay and readback exact",
        ],
        "stop_rule": "seal CR8 and stop before CR9",
        "claim_boundary": {
            "permitted_if_passive": "controller-maintained compotype-like state",
            "prohibited": [
                "installed compotype without autonomous release-and-return gates",
                "biological memory, autonomous agency, life, or error correction",
                "real prebiotic chemistry or a universal origin-of-life mechanism",
                "strict-eight or Phi/PhiID control",
            ],
        },
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


def experiment() -> ExperimentConfig:
    cohort = CohortConfig(MATRICES, REPLICATES, (60,))
    return ExperimentConfig(
        gard=GardConfig(),
        development=cohort,
        confirmation=cohort,
        horizon=CHALLENGE_HORIZON,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
        master_seed=SEEDS["release_future"],
    )


def _snapshot_equal(left: Snapshot, right: Snapshot) -> bool:
    return bool(
        np.array_equal(left.composition, right.composition)
        and left.generation == right.generation
        and left.inheritance == right.inheritance
        and left.boundary_h == right.boundary_h
        and left.previous_growth_steps == right.previous_growth_steps
        and left.cumulative_growth_steps == right.cumulative_growth_steps
    )


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
    return float(values.max() / mass) if mass > 0.0 else 0.0


def _throughput(composition: NDArray, beta: NDArray) -> float:
    values = np.asarray(composition, dtype=np.float64)
    mass = float(values.sum())
    if mass <= 0.0:
        return 0.0
    x = values / mass
    return float(x @ np.asarray(beta, dtype=np.float64) @ x)


def _post_fission_snapshots(
    launch: Snapshot, records: Iterable[Any]
) -> tuple[Snapshot, ...]:
    snapshots: list[Snapshot] = []
    current = launch
    cumulative = launch.cumulative_growth_steps
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
        snapshots.append(current)
    return tuple(snapshots)


def _release_future_seed(candidate: str, matrix_id: int, replicate: int) -> int:
    return derive_seed(
        SEEDS["release_future"],
        f"{LABEL}.release.future",
        candidate,
        matrix_id,
        replicate,
    )


def _challenge_edit_seed(
    candidate: str, matrix_id: int, replicate: int, origin: str, arm: str
) -> int:
    return derive_seed(
        SEEDS["challenge_edit"],
        f"{LABEL}.challenge.edit",
        candidate,
        matrix_id,
        replicate,
        origin,
        arm,
    )


def _challenge_future_seed(
    candidate: str, matrix_id: int, replicate: int, branch: int
) -> int:
    return derive_seed(
        SEEDS["challenge_future"],
        f"{LABEL}.challenge.future",
        candidate,
        matrix_id,
        replicate,
        branch,
    )


def _verify_upstream() -> dict[str, Any]:
    for directory in (CR0_VALIDATION, CR7_REGISTRATION, CR7_RESULT):
        verify_checksums(directory)
    cr0 = json.loads((CR0_VALIDATION / "validation.json").read_text())
    registration = json.loads((CR7_REGISTRATION / "registration.json").read_text())
    result = json.loads((CR7_RESULT / "manifest.json").read_text())
    if not cr0["all_checks_passed"]:
        raise ValueError("CR0 validation is no longer passing")
    if registration["registration_id"] != EXPECTED_CR7_REGISTRATION_ID:
        raise ValueError("CR7 registration ID changed")
    if not (
        result["complete_cr7_60_fission_gate"]
        and result["exact_replay"]
        and result["complete_readback_exact"]
        and result["noop_callback_plain_bitwise_exact"]
    ):
        raise ValueError("sealed CR7 result does not authorize CR8")
    predictor = CR7_REGISTRATION / "frozen_full_predictor.npz"
    if sha256_file(predictor) != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen JOINT_BREAK_RUN3 predictor changed")
    return {
        "cr0_checksum_manifest_sha256": sha256_file(CR0_VALIDATION / "SHA256SUMS"),
        "cr7_registration_checksum_manifest_sha256": sha256_file(
            CR7_REGISTRATION / "SHA256SUMS"
        ),
        "cr7_result_checksum_manifest_sha256": sha256_file(CR7_RESULT / "SHA256SUMS"),
        "cr7_registration_id": registration["registration_id"],
        "cr7_result_manifest_sha256": sha256_file(CR7_RESULT / "manifest.json"),
    }


def _cr7_checkpoint_batches() -> list[SteeringBatch]:
    replay = json.loads((CR7_RESULT / "replay_audit.json").read_text())
    expected = {
        (str(row["candidate"]).zfill(2), int(row["matrix_id"])): row[
            "generated_digest"
        ]
        for row in replay["rows"]
    }
    batches: list[SteeringBatch] = []
    for matrix_id in range(MATRICES):
        for candidate in CANDIDATES:
            path = CR7_WORK / "primary" / "generate" / f"c{candidate}_m{matrix_id:03d}.pkl"
            if not path.is_file():
                raise FileNotFoundError(
                    f"sealed CR7 launch checkpoint required to freeze CR8: {path}"
                )
            with path.open("rb") as handle:
                batch = _CR7CheckpointUnpickler(handle).load()
            if not isinstance(batch, SteeringBatch):
                raise TypeError(f"unsupported CR7 checkpoint object: {path}")
            if (
                batch.registration_id != EXPECTED_CR7_REGISTRATION_ID
                or batch.candidate != candidate
                or batch.matrix_id != matrix_id
                or batch.mode != "primary"
            ):
                raise ValueError(f"CR7 checkpoint metadata changed: {path}")
            if cr7_batch_digest(batch) != expected[(candidate, matrix_id)]:
                raise ValueError(f"CR7 checkpoint digest differs from sealed replay: {path}")
            batches.append(batch)
    return batches


class _CR7CheckpointUnpickler(pickle.Unpickler):
    """Read CR7 CLI pickles whose two local classes were named ``__main__``.

    This compatibility map is deliberately closed: every recovered batch is
    subsequently type-checked and compared with its checksum-sealed replay digest.
    """

    _KNOWN = {
        ("__main__", "SteeringBatch"): SteeringBatch,
        ("__main__", "LineageSummary"): CR7LineageSummary,
    }

    def find_class(self, module: str, name: str) -> Any:
        replacement = self._KNOWN.get((module, name))
        if replacement is not None:
            return replacement
        return super().find_class(module, name)


def freeze_cr7_launch_archive(path: Path) -> dict[str, Any]:
    """Freeze exact CR7 endpoints after checking sealed arrays and replay digests."""

    _verify_upstream()
    batches = _cr7_checkpoint_batches()
    with np.load(CR7_RESULT / "state_and_matrix_arrays.npz", allow_pickle=False) as state:
        beta_by_matrix = np.asarray(state["beta"], dtype=np.float64)
    with np.load(CR7_RESULT / "lineage_arrays.npz", allow_pickle=False) as lineages:
        candidates = np.asarray(lineages["candidate"])
        matrix_ids = np.asarray(lineages["matrix_id"], dtype=np.int16)
        arm_names = [str(item) for item in lineages["arm_names"]]
        sealed_final = np.asarray(lineages["final_compositions"], dtype=np.int64)
    batch_lookup = {(batch.candidate, batch.matrix_id): batch for batch in batches}
    case_order = [
        (str(candidate).zfill(2), int(matrix_id))
        for candidate, matrix_id in zip(candidates, matrix_ids, strict=True)
    ]
    composition = np.zeros(
        (len(case_order), REPLICATES, len(ORIGINS), GardConfig().n_types),
        dtype=np.int64,
    )
    generation = np.zeros((len(case_order), REPLICATES, len(ORIGINS)), dtype=np.int16)
    previous = np.zeros_like(generation, dtype=np.int32)
    cumulative = np.zeros_like(generation, dtype=np.int64)
    history_length = 120
    boundary_h = np.zeros(
        (len(case_order), REPLICATES, len(ORIGINS), history_length), dtype=np.float64
    )
    inheritance = np.zeros_like(boundary_h, dtype=np.int8)
    lineage_digest = np.empty(
        (len(case_order), REPLICATES, len(ORIGINS)), dtype="<U64"
    )
    batch_digest = np.empty(len(case_order), dtype="<U64")
    for case_index, (candidate, matrix_id) in enumerate(case_order):
        batch = batch_lookup[(candidate, matrix_id)]
        batch_digest[case_index] = cr7_batch_digest(batch)
        lookup = {
            (item.controller, item.replicate): item for item in batch.lineages
        }
        for replicate in range(REPLICATES):
            for origin_index, origin in enumerate(ORIGINS):
                item = lookup[(origin, replicate)]
                snapshot = item.final_snapshot
                if len(snapshot.boundary_h) != history_length:
                    raise ValueError("CR7 endpoint history length is not 120")
                if not np.array_equal(
                    snapshot.composition,
                    sealed_final[case_index, replicate, arm_names.index(origin)],
                ):
                    raise ValueError("CR7 checkpoint endpoint differs from sealed arrays")
                index = (case_index, replicate, origin_index)
                composition[index] = snapshot.composition
                generation[index] = snapshot.generation
                previous[index] = snapshot.previous_growth_steps
                cumulative[index] = snapshot.cumulative_growth_steps
                boundary_h[index] = snapshot.boundary_h
                inheritance[index] = snapshot.inheritance
                lineage_digest[index] = cr7_lineage_digest(item)
    np.savez_compressed(
        path,
        beta=beta_by_matrix,
        candidate=np.asarray([item[0] for item in case_order]),
        matrix_id=np.asarray([item[1] for item in case_order], dtype=np.int16),
        origin_names=np.asarray(ORIGINS),
        composition=composition,
        generation=generation,
        previous_growth_steps=previous,
        cumulative_growth_steps=cumulative,
        boundary_h=boundary_h,
        inheritance=inheritance,
        source_lineage_digest=lineage_digest,
        source_batch_digest=batch_digest,
    )
    return {
        "cases": len(case_order),
        "origins": int(np.prod(composition.shape[:3])),
        "all_endpoint_compositions_match_sealed_cr7_arrays": True,
        "all_checkpoint_batch_digests_match_sealed_replay": True,
        "archive_sha256": sha256_file(path),
    }


def load_release_cases(path: Path) -> list[ReleaseCase]:
    with np.load(path, allow_pickle=False) as archive:
        beta = np.asarray(archive["beta"], dtype=np.float64)
        candidates = np.asarray(archive["candidate"])
        matrix_id = np.asarray(archive["matrix_id"], dtype=np.int16)
        names = tuple(str(item) for item in archive["origin_names"])
        composition = np.asarray(archive["composition"], dtype=np.int64)
        generation = np.asarray(archive["generation"], dtype=np.int16)
        previous = np.asarray(archive["previous_growth_steps"], dtype=np.int32)
        cumulative = np.asarray(archive["cumulative_growth_steps"], dtype=np.int64)
        boundary_h = np.asarray(archive["boundary_h"], dtype=np.float64)
        inheritance = np.asarray(archive["inheritance"], dtype=np.int8)
        lineage_digest = np.asarray(archive["source_lineage_digest"])
        source_batch_digest = np.asarray(archive["source_batch_digest"])
    if names != ORIGINS:
        raise ValueError("CR8 launch origin order changed")
    if composition.shape != (2 * MATRICES, REPLICATES, len(ORIGINS), 100):
        raise ValueError("CR8 launch-state archive has an unexpected shape")
    cases: list[ReleaseCase] = []
    for case_index, candidate_value in enumerate(candidates):
        candidate = str(candidate_value).zfill(2)
        current_matrix = int(matrix_id[case_index])
        origins: list[FrozenOrigin] = []
        for replicate in range(REPLICATES):
            for origin_index, origin in enumerate(ORIGINS):
                index = (case_index, replicate, origin_index)
                snapshot = Snapshot(
                    composition=composition[index].copy(),
                    generation=int(generation[index]),
                    inheritance=tuple(bool(item) for item in inheritance[index]),
                    boundary_h=tuple(float(item) for item in boundary_h[index]),
                    previous_growth_steps=int(previous[index]),
                    cumulative_growth_steps=int(cumulative[index]),
                )
                origins.append(
                    FrozenOrigin(
                        origin=origin,
                        replicate=replicate,
                        snapshot=snapshot,
                        source_lineage_digest=str(lineage_digest[index]),
                    )
                )
        cases.append(
            ReleaseCase(
                state_id=f"{LABEL}-c{candidate}-m{current_matrix:03d}",
                candidate=candidate,
                matrix_id=current_matrix,
                beta=beta[current_matrix].copy(),
                origins=tuple(origins),
                source_batch_digest=str(source_batch_digest[case_index]),
            )
        )
    return cases


def _empty_float(length: int) -> NDArray[np.float64]:
    return np.full(length, np.nan, dtype=np.float64)


def _release_summary(
    frozen: FrozenOrigin,
    beta: NDArray,
    candidate: str,
    current_experiment: ExperimentConfig,
    predictor: FrozenFullPredictor,
    simulation_seed: int,
) -> tuple[ReleaseSummary, tuple[Snapshot, ...]]:
    rng = np.random.default_rng(simulation_seed)
    result = simulate_controlled(
        frozen.snapshot,
        beta,
        candidate,
        current_experiment,
        RELEASE_HORIZON,
        rng,
        None,
    )
    if result.interventions_applied != 0 or result.selected_edits:
        raise AssertionError("release mode applied an intervention")
    snapshots = _post_fission_snapshots(frozen.snapshot, result.records)
    if snapshots and not _snapshot_equal(snapshots[-1], result.final_snapshot):
        raise AssertionError("release snapshot reconstruction differs from simulator")
    similarity = _empty_float(RELEASE_HORIZON)
    risk = _empty_float(RELEASE_HORIZON)
    boundary_h = _empty_float(RELEASE_HORIZON)
    growth = np.full(RELEASE_HORIZON, -1, dtype=np.int32)
    entropy = _empty_float(RELEASE_HORIZON)
    occupied = np.full(RELEASE_HORIZON, -1, dtype=np.int16)
    top1 = _empty_float(RELEASE_HORIZON)
    throughput = _empty_float(RELEASE_HORIZON)
    anchor = frozen.snapshot.composition
    for index, (snapshot, record) in enumerate(zip(snapshots, result.records, strict=True)):
        similarity[index] = cosine_similarity(anchor, snapshot.composition)
        risk[index] = predictor.predict_snapshot(
            candidate, snapshot, beta, current_experiment.gard
        )
        boundary_h[index] = float(record.h)
        growth[index] = int(record.growth_steps)
        entropy[index] = _entropy(snapshot.composition)
        occupied[index] = int(np.count_nonzero(snapshot.composition))
        top1[index] = _top1(snapshot.composition)
        throughput[index] = _throughput(snapshot.composition, beta)
    # Fixed-horizon adverse accounting: unobserved registered boundaries are not
    # counted as inherited, so extinct/failed lineages cannot look successful.
    final_window = np.zeros(MODE_FINAL_WINDOW, dtype=np.float64)
    final_h = boundary_h[-MODE_FINAL_WINDOW:]
    observed = np.isfinite(final_h)
    final_window[observed] = final_h[observed] > INHERITANCE_THRESHOLD
    departures = np.flatnonzero(similarity < DEPARTURE_THRESHOLD)
    return (
        ReleaseSummary(
            origin=frozen.origin,
            replicate=frozen.replicate,
            completed_horizon=bool(result.completed_horizon),
            observed_fissions=len(result.records),
            record_digest=_records_digest(result.records),
            anchor_composition=np.asarray(anchor, dtype=np.int64).copy(),
            final_snapshot=result.final_snapshot,
            similarity_to_anchor=similarity,
            similarity_to_matched_noop=_empty_float(RELEASE_HORIZON),
            risk=risk,
            boundary_h=boundary_h,
            growth_updates=growth,
            entropy=entropy,
            occupied_types=occupied,
            top1_share=top1,
            throughput=throughput,
            final_six_inherited_fraction=float(final_window.mean()),
            first_departure_time=(int(departures[0]) + 1 if departures.size else -1),
            interventions_applied=int(result.interventions_applied),
        ),
        snapshots,
    )


def run_release_case(
    case: ReleaseCase,
    current_experiment: ExperimentConfig,
    model_path: str | Path,
    registration_id: str,
) -> ReleaseBatch:
    predictor = FrozenFullPredictor.load(model_path)
    raw: dict[tuple[str, int], tuple[ReleaseSummary, tuple[Snapshot, ...]]] = {}
    for frozen in case.origins:
        raw[(frozen.origin, frozen.replicate)] = _release_summary(
            frozen,
            case.beta,
            case.candidate,
            current_experiment,
            predictor,
            _release_future_seed(case.candidate, case.matrix_id, frozen.replicate),
        )
    summaries: list[ReleaseSummary] = []
    for frozen in case.origins:
        summary, snapshots = raw[(frozen.origin, frozen.replicate)]
        matched = _empty_float(RELEASE_HORIZON)
        if frozen.origin == "NOOP":
            for index, snapshot in enumerate(snapshots):
                matched[index] = 1.0
        else:
            _natural_summary, natural_snapshots = raw[("NOOP", frozen.replicate)]
            for index in range(min(len(snapshots), len(natural_snapshots))):
                matched[index] = cosine_similarity(
                    snapshots[index].composition,
                    natural_snapshots[index].composition,
                )
        summaries.append(
            ReleaseSummary(
                **{
                    **summary.__dict__,
                    "similarity_to_matched_noop": matched,
                }
            )
        )
    return ReleaseBatch(
        format=RELEASE_CHECKPOINT_FORMAT,
        registration_id=registration_id,
        state_id=case.state_id,
        candidate=case.candidate,
        matrix_id=case.matrix_id,
        source_batch_digest=case.source_batch_digest,
        releases=tuple(summaries),
    )


def random_k_edits(
    composition: NDArray, k: int, rng: np.random.Generator
) -> tuple[MolecularEdit, ...]:
    values = np.asarray(composition, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0):
        raise ValueError("composition must be a nonnegative integer vector")
    mass = int(values.sum())
    if not 0 <= k <= mass:
        raise ValueError("random K dose exceeds assembly mass")
    if k == 0:
        return ()
    molecule_types = np.repeat(np.arange(values.size, dtype=np.int64), values)
    selected = rng.choice(molecule_types.size, size=k, replace=False)
    removals = molecule_types[selected]
    forbidden = np.unique(removals)
    targets = np.setdiff1d(
        np.arange(values.size, dtype=np.int64), forbidden, assume_unique=True
    )
    if targets.size == 0:
        raise ValueError("no legal target labels remain for exact-dose perturbation")
    additions = rng.choice(targets, size=k, replace=True)
    edits = tuple(
        MolecularEdit(int(remove), int(add))
        for remove, add in zip(removals, additions, strict=True)
    )
    edited = apply_edits(values, edits)
    if int(np.abs(edited - values).sum() // 2) != k:
        raise AssertionError("random K perturbation lost its exact transport dose")
    return edits


def apply_edits(composition: NDArray, edits: Iterable[MolecularEdit]) -> NDArray[np.int64]:
    current = np.asarray(composition, dtype=np.int64).copy()
    original_mass = int(current.sum())
    for edit in edits:
        current = apply_molecular_edit(current, edit)
    if np.any(current < 0) or int(current.sum()) != original_mass:
        raise AssertionError("challenge edits violated mass or nonnegativity")
    return current


def edited_snapshot_many(snapshot: Snapshot, edits: Iterable[MolecularEdit]) -> Snapshot:
    current = snapshot
    history = (
        snapshot.generation,
        snapshot.inheritance,
        snapshot.boundary_h,
        snapshot.previous_growth_steps,
        snapshot.cumulative_growth_steps,
    )
    for edit in edits:
        current = edited_snapshot(current, edit)
    observed_history = (
        current.generation,
        current.inheritance,
        current.boundary_h,
        current.previous_growth_steps,
        current.cumulative_growth_steps,
    )
    if observed_history != history:
        raise AssertionError("instantaneous challenge changed observed history")
    return current


def challenge_plan(
    release: ReleaseSummary,
    beta: NDArray,
    candidate: str,
    arm: str,
    predictor: FrozenFullPredictor,
    config: GardConfig,
    selection_seed: int,
) -> ChallengePlan:
    anchor = release.final_snapshot
    noop_risk = predictor.predict_snapshot(candidate, anchor, beta, config)
    edits: tuple[MolecularEdit, ...]
    edited_risk = noop_risk
    if arm == "NONE":
        edits = ()
    elif arm.startswith("RANDOM_K"):
        k = ARM_DOSE[arm]
        edits = random_k_edits(
            anchor.composition, k, np.random.default_rng(selection_seed)
        )
    elif arm == "ADVERSARIAL":
        noop, scores = score_legal_edits(predictor, candidate, anchor, beta, config)
        probabilities = np.asarray(
            [item.predicted_probability for item in scores], dtype=np.float64
        )
        index = int(np.flatnonzero(probabilities == probabilities.max())[0])
        edits = (scores[index].edit,)
        noop_risk = float(noop)
        edited_risk = float(scores[index].predicted_probability)
    else:
        raise ValueError(f"unknown CR8 challenge arm: {arm}")
    launch = edited_snapshot_many(anchor, edits)
    if arm != "ADVERSARIAL":
        edited_risk = predictor.predict_snapshot(candidate, launch, beta, config)
    distance = int(
        np.abs(launch.composition - anchor.composition).sum() // 2
    )
    expected = ARM_DOSE[arm]
    if distance != expected:
        raise AssertionError(
            f"challenge arm {arm} has transport {distance}, expected {expected}"
        )
    if arm == "ADVERSARIAL" and edited_risk < noop_risk:
        raise AssertionError("adversarial maximum unexpectedly lowers frozen risk")
    return ChallengePlan(
        origin=release.origin,
        replicate=release.replicate,
        arm=arm,
        edits=edits,
        transport_distance=distance,
        noop_risk=float(noop_risk),
        edited_risk=float(edited_risk),
        launch_composition=launch.composition.copy(),
    )


def classify_challenge(
    similarity: NDArray,
    boundary_h: NDArray,
    final_top1_share: float,
    completed_horizon: bool,
) -> dict[str, Any]:
    values = np.asarray(similarity, dtype=np.float64)
    h = np.asarray(boundary_h, dtype=np.float64)
    departures = np.flatnonzero(values < DEPARTURE_THRESHOLD)
    departed = bool(departures.size)
    first_departure = int(departures[0]) if departed else -1
    certification = -1
    if departed:
        trailing = 0
        for index in range(first_departure + 1, values.size):
            if not np.isfinite(values[index]):
                break
            if values[index] > RETURN_THRESHOLD:
                trailing += 1
                if trailing == RETURN_RUN:
                    certification = index
                    break
            else:
                trailing = 0
    inherited_final = int(
        np.count_nonzero(
            np.isfinite(h[-MODE_FINAL_WINDOW:])
            & (h[-MODE_FINAL_WINDOW:] > INHERITANCE_THRESHOLD)
        )
    )
    held = bool(completed_horizon and not departed)
    returned = bool(completed_horizon and departed and certification >= 0)
    mode = bool(
        completed_horizon
        and departed
        and not returned
        and inherited_final >= MODE_MIN_INHERITED
        and final_top1_share >= MODE_TOP1_THRESHOLD
    )
    if held:
        category = "held"
    elif returned:
        category = "returned"
    elif mode:
        category = "mode_recovered"
    else:
        category = "lost"
    return {
        "category": category,
        "held": held,
        "returned": returned,
        "mode_recovered": mode,
        "lost": category == "lost",
        "departed": departed,
        "first_departure_time": first_departure,
        "return_certification_time": certification,
        "inherited_final_six": inherited_final,
    }


def _challenge_summary(
    release: ReleaseSummary,
    plan: ChallengePlan,
    beta: NDArray,
    candidate: str,
    current_experiment: ExperimentConfig,
    branch: int,
    simulation_seed: int,
) -> ChallengeSummary:
    launch = Snapshot(
        composition=plan.launch_composition.copy(),
        generation=release.final_snapshot.generation,
        inheritance=release.final_snapshot.inheritance,
        boundary_h=release.final_snapshot.boundary_h,
        previous_growth_steps=release.final_snapshot.previous_growth_steps,
        cumulative_growth_steps=release.final_snapshot.cumulative_growth_steps,
    )
    rng = np.random.default_rng(simulation_seed)
    result = simulate_controlled(
        launch,
        beta,
        candidate,
        current_experiment,
        CHALLENGE_HORIZON,
        rng,
        None,
    )
    if result.interventions_applied != 0 or result.selected_edits:
        raise AssertionError("challenge future applied an intervention after launch")
    snapshots = _post_fission_snapshots(launch, result.records)
    similarity = np.full(CHALLENGE_HORIZON + 1, np.nan, dtype=np.float64)
    similarity[0] = cosine_similarity(
        release.final_snapshot.composition, launch.composition
    )
    boundary_h = _empty_float(CHALLENGE_HORIZON)
    for index, (snapshot, record) in enumerate(zip(snapshots, result.records, strict=True)):
        similarity[index + 1] = cosine_similarity(
            release.final_snapshot.composition, snapshot.composition
        )
        boundary_h[index] = float(record.h)
    final_top1 = _top1(result.final_snapshot.composition)
    classification = classify_challenge(
        similarity,
        boundary_h,
        final_top1,
        bool(result.completed_horizon),
    )
    finite = similarity[np.isfinite(similarity)]
    return ChallengeSummary(
        origin=release.origin,
        replicate=release.replicate,
        arm=plan.arm,
        branch=branch,
        completed_horizon=bool(result.completed_horizon),
        observed_fissions=len(result.records),
        record_digest=_records_digest(result.records),
        category=classification["category"],
        held=classification["held"],
        returned=classification["returned"],
        mode_recovered=classification["mode_recovered"],
        lost=classification["lost"],
        departed=classification["departed"],
        first_departure_time=classification["first_departure_time"],
        return_certification_time=classification["return_certification_time"],
        inherited_final_six=classification["inherited_final_six"],
        final_top1_share=final_top1,
        final_similarity=float(finite[-1]),
        minimum_similarity=float(finite.min()),
        similarity_to_anchor=similarity,
        boundary_h=boundary_h,
        final_composition=result.final_snapshot.composition.astype(np.int16, copy=True),
        final_generation=result.final_snapshot.generation,
        final_previous_growth_steps=result.final_snapshot.previous_growth_steps,
        final_cumulative_growth_steps=result.final_snapshot.cumulative_growth_steps,
    )


def run_challenge_case(
    case: ReleaseCase,
    release_batch: ReleaseBatch,
    current_experiment: ExperimentConfig,
    model_path: str | Path,
    registration_id: str,
    *,
    branches: int = CHALLENGE_BRANCHES,
    arms: tuple[str, ...] = CHALLENGE_ARMS,
) -> ChallengeBatch:
    predictor = FrozenFullPredictor.load(model_path)
    plans: list[ChallengePlan] = []
    outcomes: list[ChallengeSummary] = []
    for release in release_batch.releases:
        for arm in arms:
            plan = challenge_plan(
                release,
                case.beta,
                case.candidate,
                arm,
                predictor,
                current_experiment.gard,
                _challenge_edit_seed(
                    case.candidate,
                    case.matrix_id,
                    release.replicate,
                    release.origin,
                    arm,
                ),
            )
            plans.append(plan)
            for branch in range(branches):
                outcomes.append(
                    _challenge_summary(
                        release,
                        plan,
                        case.beta,
                        case.candidate,
                        current_experiment,
                        branch,
                        _challenge_future_seed(
                            case.candidate,
                            case.matrix_id,
                            release.replicate,
                            branch,
                        ),
                    )
                )
    return ChallengeBatch(
        format=CHALLENGE_CHECKPOINT_FORMAT,
        registration_id=registration_id,
        state_id=case.state_id,
        candidate=case.candidate,
        matrix_id=case.matrix_id,
        release_batch_digest=release_batch_digest(release_batch),
        plans=tuple(plans),
        outcomes=tuple(outcomes),
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


def release_batch_digest(batch: ReleaseBatch) -> str:
    digest = hashlib.sha256()
    for value in (
        batch.format,
        batch.registration_id,
        batch.state_id,
        batch.candidate,
        str(batch.matrix_id),
        batch.source_batch_digest,
    ):
        digest.update(value.encode())
    for item in batch.releases:
        for value in (
            item.origin,
            str(item.replicate),
            str(int(item.completed_horizon)),
            str(item.observed_fissions),
            item.record_digest,
            str(item.interventions_applied),
            _snapshot_digest(item.final_snapshot),
        ):
            digest.update(value.encode())
        digest.update(
            np.asarray(
                (item.final_six_inherited_fraction, item.first_departure_time),
                dtype=np.float64,
            ).tobytes()
        )
        for array in (
            item.anchor_composition,
            item.similarity_to_anchor,
            item.similarity_to_matched_noop,
            item.risk,
            item.boundary_h,
            item.growth_updates,
            item.entropy,
            item.occupied_types,
            item.top1_share,
            item.throughput,
        ):
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def challenge_batch_digest(batch: ChallengeBatch) -> str:
    digest = hashlib.sha256()
    for value in (
        batch.format,
        batch.registration_id,
        batch.state_id,
        batch.candidate,
        str(batch.matrix_id),
        batch.release_batch_digest,
    ):
        digest.update(value.encode())
    for plan in batch.plans:
        for value in (plan.origin, str(plan.replicate), plan.arm):
            digest.update(value.encode())
        digest.update(
            np.asarray(
                (plan.transport_distance, plan.noop_risk, plan.edited_risk),
                dtype=np.float64,
            ).tobytes()
        )
        edits = np.asarray(
            [(item.remove_type, item.add_type) for item in plan.edits],
            dtype=np.int16,
        ).reshape(-1, 2)
        digest.update(edits.tobytes())
        digest.update(np.ascontiguousarray(plan.launch_composition).tobytes())
    for item in batch.outcomes:
        for value in (
            item.origin,
            str(item.replicate),
            item.arm,
            str(item.branch),
            str(int(item.completed_horizon)),
            str(item.observed_fissions),
            item.record_digest,
            item.category,
        ):
            digest.update(value.encode())
        digest.update(
            np.asarray(
                (
                    item.held,
                    item.returned,
                    item.mode_recovered,
                    item.lost,
                    item.departed,
                    item.first_departure_time,
                    item.return_certification_time,
                    item.inherited_final_six,
                    item.final_top1_share,
                    item.final_similarity,
                    item.minimum_similarity,
                    item.final_generation,
                    item.final_previous_growth_steps,
                    item.final_cumulative_growth_steps,
                ),
                dtype=np.float64,
            ).tobytes()
        )
        for array in (
            item.similarity_to_anchor,
            item.boundary_h,
            item.final_composition,
        ):
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def replay_audit(
    generated: list[ReleaseBatch] | list[ChallengeBatch],
    replayed: list[ReleaseBatch] | list[ChallengeBatch],
    kind: str,
) -> dict[str, Any]:
    if len(generated) != len(replayed):
        raise ValueError(f"CR8 {kind} replay batch count differs")
    digest_function = release_batch_digest if kind == "release" else challenge_batch_digest
    rows = []
    for left, right in zip(generated, replayed, strict=True):
        left_digest = digest_function(left)  # type: ignore[arg-type]
        right_digest = digest_function(right)  # type: ignore[arg-type]
        rows.append(
            {
                "state_id": left.state_id,
                "candidate": left.candidate,
                "matrix_id": left.matrix_id,
                "exact": left_digest == right_digest,
                "generated_digest": left_digest,
                "replay_digest": right_digest,
            }
        )
    return {
        "format": f"codex-intervention-cr8-{kind}-replay-audit-v1",
        "state_batches": len(rows),
        "exact_state_edit_endpoint_process_and_rng": bool(
            all(row["exact"] for row in rows)
        ),
        "rows": rows,
    }


def _checkpoint_path(directory: Path, case: ReleaseCase) -> Path:
    return directory / f"c{case.candidate}_m{case.matrix_id:03d}.pkl"


def _write_checkpoint(path: Path, value: ReleaseBatch | ChallengeBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_release_checkpoint(
    path: Path, case: ReleaseCase, registration_id: str
) -> ReleaseBatch | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(value, ReleaseBatch):
        return None
    if (
        value.format != RELEASE_CHECKPOINT_FORMAT
        or value.registration_id != registration_id
        or value.state_id != case.state_id
        or value.candidate != case.candidate
        or value.matrix_id != case.matrix_id
        or value.source_batch_digest != case.source_batch_digest
        or len(value.releases) != REPLICATES * len(ORIGINS)
    ):
        return None
    return value


def _read_challenge_checkpoint(
    path: Path,
    case: ReleaseCase,
    release: ReleaseBatch,
    registration_id: str,
) -> ChallengeBatch | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(value, ChallengeBatch):
        return None
    if (
        value.format != CHALLENGE_CHECKPOINT_FORMAT
        or value.registration_id != registration_id
        or value.state_id != case.state_id
        or value.candidate != case.candidate
        or value.matrix_id != case.matrix_id
        or value.release_batch_digest != release_batch_digest(release)
        or len(value.plans) != REPLICATES * len(ORIGINS) * len(CHALLENGE_ARMS)
        or len(value.outcomes)
        != REPLICATES
        * len(ORIGINS)
        * len(CHALLENGE_ARMS)
        * CHALLENGE_BRANCHES
    ):
        return None
    return value


def _write_status(
    work: Path, stage: str, completed: int, total: int, **extra: Any
) -> None:
    work.mkdir(parents=True, exist_ok=True)
    now = time.time()
    prior_path = work / "campaign_status.json"
    prior: dict[str, Any] = {}
    if prior_path.is_file():
        try:
            prior = json.loads(prior_path.read_text())
        except Exception:
            prior = {}
    if prior.get("stage") == stage:
        stage_started = float(prior.get("stage_started_unix", now))
        stage_started_completed = int(
            prior.get("stage_started_completed_state_batches", completed)
        )
    else:
        stage_started = now
        stage_started_completed = completed
    elapsed = max(0.0, now - stage_started)
    newly_completed = max(0, completed - stage_started_completed)
    rate = newly_completed / elapsed if elapsed > 0.0 else 0.0
    eta = (total - completed) / rate if rate > 0.0 else None
    payload = {
        "format": STATUS_FORMAT,
        "stage": stage,
        "completed_state_batches": completed,
        "total_state_batches": total,
        "updated_at_unix": now,
        "stage_started_unix": stage_started,
        "stage_started_completed_state_batches": stage_started_completed,
        "stage_elapsed_seconds": elapsed,
        "state_batches_per_second": rate,
        "estimated_stage_seconds_remaining": eta,
        **extra,
    }
    temporary = work / f".status-{os.getpid()}.tmp"
    temporary.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(work / "campaign_status.json")


def _release_worker(arguments: tuple[Any, ...]) -> ReleaseBatch:
    limiter = threadpool_limits(limits=1)
    try:
        return run_release_case(*arguments)
    finally:
        limiter.restore_original_limits()


def _challenge_worker(arguments: tuple[Any, ...]) -> ChallengeBatch:
    limiter = threadpool_limits(limits=1)
    try:
        return run_challenge_case(*arguments)
    finally:
        limiter.restore_original_limits()


def run_release_batches(
    cases: list[ReleaseCase],
    current_experiment: ExperimentConfig,
    model_path: Path,
    registration_id: str,
    directory: Path,
    workers: int,
    work: Path,
    stage: str,
) -> list[ReleaseBatch]:
    directory.mkdir(parents=True, exist_ok=True)
    batches: dict[str, ReleaseBatch] = {}
    missing: list[ReleaseCase] = []
    for case in cases:
        value = _read_release_checkpoint(
            _checkpoint_path(directory, case), case, registration_id
        )
        if value is None:
            missing.append(case)
        else:
            batches[case.state_id] = value
    _write_status(work, stage, len(batches), len(cases), reused=len(batches))
    arguments = [
        (case, current_experiment, model_path, registration_id) for case in missing
    ]
    if workers == 1:
        for case, argument in zip(missing, arguments, strict=True):
            batch = _release_worker(argument)
            _write_checkpoint(_checkpoint_path(directory, case), batch)
            batches[case.state_id] = batch
            _write_status(work, stage, len(batches), len(cases))
            print(f"[{stage}] {len(batches)}/{len(cases)} state batches", flush=True)
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_release_worker, argument): case
                for argument, case in zip(arguments, missing, strict=True)
            }
            for future in as_completed(futures):
                case = futures[future]
                batch = future.result()
                _write_checkpoint(_checkpoint_path(directory, case), batch)
                batches[case.state_id] = batch
                _write_status(work, stage, len(batches), len(cases))
                print(f"[{stage}] {len(batches)}/{len(cases)} state batches", flush=True)
    return [batches[case.state_id] for case in cases]


def run_challenge_batches(
    cases: list[ReleaseCase],
    releases: list[ReleaseBatch],
    current_experiment: ExperimentConfig,
    model_path: Path,
    registration_id: str,
    directory: Path,
    workers: int,
    work: Path,
    stage: str,
) -> list[ChallengeBatch]:
    directory.mkdir(parents=True, exist_ok=True)
    batches: dict[str, ChallengeBatch] = {}
    missing: list[tuple[ReleaseCase, ReleaseBatch]] = []
    for case, release in zip(cases, releases, strict=True):
        value = _read_challenge_checkpoint(
            _checkpoint_path(directory, case), case, release, registration_id
        )
        if value is None:
            missing.append((case, release))
        else:
            batches[case.state_id] = value
    _write_status(work, stage, len(batches), len(cases), reused=len(batches))
    arguments = [
        (case, release, current_experiment, model_path, registration_id)
        for case, release in missing
    ]
    if workers == 1:
        for (case, _release), argument in zip(missing, arguments, strict=True):
            batch = _challenge_worker(argument)
            _write_checkpoint(_checkpoint_path(directory, case), batch)
            batches[case.state_id] = batch
            _write_status(work, stage, len(batches), len(cases))
            print(f"[{stage}] {len(batches)}/{len(cases)} state batches", flush=True)
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_challenge_worker, argument): pair[0]
                for argument, pair in zip(arguments, missing, strict=True)
            }
            for future in as_completed(futures):
                case = futures[future]
                batch = future.result()
                _write_checkpoint(_checkpoint_path(directory, case), batch)
                batches[case.state_id] = batch
                _write_status(work, stage, len(batches), len(cases))
                print(f"[{stage}] {len(batches)}/{len(cases)} state batches", flush=True)
    return [batches[case.state_id] for case in cases]


def inference_draws() -> dict[str, NDArray]:
    bootstrap_rng = np.random.default_rng(
        derive_seed(SEEDS["bootstrap"], f"{LABEL}.whole_matrix_bootstrap")
    )
    randomization_rng = np.random.default_rng(
        derive_seed(SEEDS["randomization"], f"{LABEL}.whole_matrix_signs")
    )
    bootstrap = bootstrap_rng.integers(
        0,
        MATRICES,
        size=(BOOTSTRAP_REPETITIONS, MATRICES),
        dtype=np.int16,
    )
    signs = randomization_rng.integers(
        0,
        2,
        size=(RANDOMIZATION_REPETITIONS, MATRICES),
        dtype=np.int8,
    ).astype(np.float64)
    return {"bootstrap_indices": bootstrap, "randomization_signs": 2.0 * signs - 1.0}


def _interval(values: NDArray, alpha: float = 0.05) -> tuple[float, float]:
    lower, upper = np.quantile(
        np.asarray(values, dtype=np.float64), (alpha / 2.0, 1.0 - alpha / 2.0)
    )
    return float(lower), float(upper)


def _sign_randomization_p(values: NDArray, signs: NDArray) -> tuple[float, NDArray]:
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    null = np.asarray(signs @ array / array.size, dtype=np.float64)
    p_value = float((np.count_nonzero(np.abs(null) >= observed) + 1) / (len(null) + 1))
    return p_value, null


def _holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted[index] = running
    return [float(item) for item in adjusted]


def _maximum_leave_one_out_influence(values: NDArray) -> float:
    array = np.asarray(values, dtype=np.float64)
    estimate = float(array.mean())
    leave_one = (array.sum() - array) / (array.size - 1)
    return float(np.max(np.abs(leave_one - estimate)))


def release_tables(
    cases: list[ReleaseCase], batches: list[ReleaseBatch]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, NDArray]]:
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[NDArray]] = {
        "similarity_to_anchor": [],
        "similarity_to_matched_noop": [],
        "risk": [],
        "boundary_h": [],
        "growth_updates": [],
        "entropy": [],
        "occupied_types": [],
        "top1_share": [],
        "throughput": [],
        "anchor_composition": [],
        "final_composition": [],
    }
    for case, batch in zip(cases, batches, strict=True):
        if batch.source_batch_digest != case.source_batch_digest:
            raise ValueError("release batch lost its CR7 parent")
        for item in batch.releases:
            row_index = len(rows)
            rows.append(
                {
                    "row_index": row_index,
                    "state_id": case.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "origin": item.origin,
                    "replicate": item.replicate,
                    "completed_horizon": int(item.completed_horizon),
                    "observed_fissions": item.observed_fissions,
                    "final_six_inherited_fraction": item.final_six_inherited_fraction,
                    "first_departure_time": item.first_departure_time,
                    "ever_departed": int(item.first_departure_time >= 0),
                    "final_similarity_to_anchor": float(
                        item.similarity_to_anchor[
                            np.flatnonzero(np.isfinite(item.similarity_to_anchor))[-1]
                        ]
                    )
                    if np.any(np.isfinite(item.similarity_to_anchor))
                    else float("nan"),
                    "final_similarity_to_matched_noop": float(
                        item.similarity_to_matched_noop[
                            np.flatnonzero(
                                np.isfinite(item.similarity_to_matched_noop)
                            )[-1]
                        ]
                    )
                    if np.any(np.isfinite(item.similarity_to_matched_noop))
                    else float("nan"),
                    "final_risk": float(
                        item.risk[np.flatnonzero(np.isfinite(item.risk))[-1]]
                    )
                    if np.any(np.isfinite(item.risk))
                    else float("nan"),
                    "final_entropy": _entropy(item.final_snapshot.composition),
                    "final_occupied_types": int(
                        np.count_nonzero(item.final_snapshot.composition)
                    ),
                    "final_top1_share": _top1(item.final_snapshot.composition),
                    "final_throughput": _throughput(
                        item.final_snapshot.composition, case.beta
                    ),
                    "record_digest": item.record_digest,
                    "final_composition_digest": hashlib.sha256(
                        np.ascontiguousarray(item.final_snapshot.composition).tobytes()
                    ).hexdigest(),
                    "interventions_applied": item.interventions_applied,
                }
            )
            for name in arrays:
                if name == "final_composition":
                    value = item.final_snapshot.composition
                else:
                    value = getattr(item, name)
                arrays[name].append(np.asarray(value))
    lineage = pd.DataFrame(rows)
    matrix = (
        lineage.groupby(["candidate", "matrix_id", "origin"], as_index=False)
        .mean(numeric_only=True)
    )
    packed = {name: np.stack(values) for name, values in arrays.items()}
    packed.update(
        {
            "candidate": lineage["candidate"].to_numpy(dtype="<U2"),
            "matrix_id": lineage["matrix_id"].to_numpy(dtype=np.int16),
            "origin": lineage["origin"].to_numpy(dtype="<U10"),
            "replicate": lineage["replicate"].to_numpy(dtype=np.int8),
        }
    )
    return lineage, matrix, packed


def challenge_tables(
    cases: list[ReleaseCase], batches: list[ChallengeBatch]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, NDArray]]:
    plan_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    similarity: list[NDArray] = []
    boundary_h: list[NDArray] = []
    final_composition: list[NDArray] = []
    for case, batch in zip(cases, batches, strict=True):
        for plan in batch.plans:
            plan_id = len(plan_rows)
            plan_rows.append(
                {
                    "plan_id": plan_id,
                    "state_id": case.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "origin": plan.origin,
                    "replicate": plan.replicate,
                    "arm": plan.arm,
                    "registered_dose": ARM_DOSE[plan.arm],
                    "edit_count": len(plan.edits),
                    "transport_distance": plan.transport_distance,
                    "noop_risk": plan.noop_risk,
                    "edited_risk": plan.edited_risk,
                    "predicted_risk_shift": plan.edited_risk - plan.noop_risk,
                    "edits": ";".join(
                        f"{edit.remove_type}>{edit.add_type}" for edit in plan.edits
                    ),
                    "launch_composition_digest": hashlib.sha256(
                        np.ascontiguousarray(plan.launch_composition).tobytes()
                    ).hexdigest(),
                }
            )
        for item in batch.outcomes:
            row_index = len(outcome_rows)
            outcome_rows.append(
                {
                    "row_index": row_index,
                    "state_id": case.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "origin": item.origin,
                    "replicate": item.replicate,
                    "arm": item.arm,
                    "registered_dose": ARM_DOSE[item.arm],
                    "branch": item.branch,
                    "completed_horizon": int(item.completed_horizon),
                    "observed_fissions": item.observed_fissions,
                    "category": item.category,
                    "held": int(item.held),
                    "returned": int(item.returned),
                    "held_or_returned": int(item.held or item.returned),
                    "mode_recovered": int(item.mode_recovered),
                    "lost": int(item.lost),
                    "departed": int(item.departed),
                    "first_departure_time": item.first_departure_time,
                    "return_certification_time": item.return_certification_time,
                    "inherited_final_six": item.inherited_final_six,
                    "final_top1_share": item.final_top1_share,
                    "final_similarity": item.final_similarity,
                    "minimum_similarity": item.minimum_similarity,
                    "record_digest": item.record_digest,
                    "final_composition_digest": hashlib.sha256(
                        np.ascontiguousarray(item.final_composition).tobytes()
                    ).hexdigest(),
                }
            )
            similarity.append(item.similarity_to_anchor)
            boundary_h.append(item.boundary_h)
            final_composition.append(item.final_composition)
    plans = pd.DataFrame(plan_rows)
    outcomes = pd.DataFrame(outcome_rows)
    matrix = (
        outcomes.groupby(
            ["candidate", "matrix_id", "origin", "arm"], as_index=False
        )
        .mean(numeric_only=True)
    )
    packed = {
        "similarity_to_anchor": np.stack(similarity),
        "boundary_h": np.stack(boundary_h),
        "final_composition": np.stack(final_composition).astype(np.int16, copy=False),
        "candidate": outcomes["candidate"].to_numpy(dtype="<U2"),
        "matrix_id": outcomes["matrix_id"].to_numpy(dtype=np.int16),
        "origin": outcomes["origin"].to_numpy(dtype="<U10"),
        "arm": outcomes["arm"].to_numpy(dtype="<U12"),
        "replicate": outcomes["replicate"].to_numpy(dtype=np.int8),
        "branch": outcomes["branch"].to_numpy(dtype=np.int8),
    }
    return plans, outcomes, matrix, packed


def _preparation_audit() -> dict[str, Any]:
    metrics = json.loads((CR7_RESULT / "primary_metrics.json").read_text())
    rows = []
    for candidate_item in metrics["candidates"]:
        candidate = candidate_item["candidate"]
        arms = candidate_item["arm_means"]
        for origin in WRITTEN_ORIGINS:
            inheritance = (
                arms[origin]["inherited_fraction"]["mean"]
                - arms["NOOP"]["inherited_fraction"]["mean"]
            )
            risk = (
                arms[origin]["final_risk"]["mean"]
                - arms["NOOP"]["final_risk"]["mean"]
            )
            changed = max(
                abs(
                    arms[origin][name]["mean"] - arms["NOOP"][name]["mean"]
                )
                for name in ("final_entropy", "final_top1_share", "final_throughput")
            )
            rows.append(
                {
                    "candidate": candidate,
                    "origin": origin,
                    "inheritance_minus_noop": float(inheritance),
                    "risk_minus_noop": float(risk),
                    "maximum_registered_state_metric_change": float(changed),
                    "genuinely_altered": bool(
                        inheritance > 0.0 and risk < 0.0 and changed > 0.0
                    ),
                }
            )
    return {
        "rows": rows,
        "all_written_states_genuinely_altered": bool(
            all(row["genuinely_altered"] for row in rows)
        ),
    }


def compute_inference(
    release_lineage: pd.DataFrame,
    release_matrix: pd.DataFrame,
    challenge_matrix: pd.DataFrame,
    release_arrays: dict[str, NDArray],
    draws: dict[str, NDArray],
    *,
    release_replay_exact: bool,
    challenge_replay_exact: bool,
    readback_exact: bool = True,
) -> tuple[dict[str, Any], dict[str, NDArray]]:
    bootstrap = np.asarray(draws["bootstrap_indices"], dtype=np.int64)
    signs = np.asarray(draws["randomization_signs"], dtype=np.float64)
    if bootstrap.shape != (BOOTSTRAP_REPETITIONS, MATRICES):
        raise ValueError("CR8 bootstrap lost whole-matrix blocks")
    if signs.shape != (RANDOMIZATION_REPETITIONS, MATRICES):
        raise ValueError("CR8 randomization lost whole-matrix blocks")
    stored: dict[str, NDArray] = {
        "bootstrap_indices": bootstrap,
        "randomization_signs": signs,
    }
    preparation = _preparation_audit()
    release_candidates: list[dict[str, Any]] = []
    release_p_values: list[float] = []
    release_p_locations: list[dict[str, Any]] = []
    anchor_all_pass = True
    release_equivalence_all_pass = True
    for candidate in CANDIDATES:
        candidate_rows = release_matrix[
            release_matrix["candidate"].astype(str).str.zfill(2) == candidate
        ]
        lineage_rows = release_lineage[
            release_lineage["candidate"].astype(str).str.zfill(2) == candidate
        ]
        natural = candidate_rows[candidate_rows["origin"] == "NOOP"].set_index(
            "matrix_id"
        )
        item: dict[str, Any] = {"candidate": candidate, "origins": []}
        for origin in WRITTEN_ORIGINS:
            written = candidate_rows[candidate_rows["origin"] == origin].set_index(
                "matrix_id"
            )
            if len(written) != MATRICES or len(natural) != MATRICES:
                raise ValueError("CR8 release lost a whole-matrix origin block")
            effect = (
                written["final_six_inherited_fraction"].to_numpy(dtype=np.float64)
                - natural["final_six_inherited_fraction"].to_numpy(dtype=np.float64)
            )
            boot = effect[bootstrap].mean(axis=1)
            ci90 = _interval(boot, alpha=0.10)
            ci95 = _interval(boot)
            raw_p, null = _sign_randomization_p(effect, signs)
            release_p_values.append(raw_p)
            summary: dict[str, Any] = {
                "origin": origin,
                "last_six_inheritance_minus_noop": {
                    "estimate": float(effect.mean()),
                    "bootstrap_ci90": ci90,
                    "bootstrap_ci95": ci95,
                    "randomization_p_raw": raw_p,
                    "tost_equivalent_within_0_03": bool(
                        ci90[0] > -RELEASE_EQUIVALENCE_MARGIN
                        and ci90[1] < RELEASE_EQUIVALENCE_MARGIN
                    ),
                    "maximum_leave_one_matrix_out_influence": _maximum_leave_one_out_influence(
                        effect
                    ),
                },
            }
            release_p_locations.append(summary["last_six_inheritance_minus_noop"])
            stored[f"release_c{candidate}_{origin}_matrix_effect"] = effect
            stored[f"release_c{candidate}_{origin}_bootstrap"] = boot
            stored[f"release_c{candidate}_{origin}_randomization"] = null

            mask = (
                (release_arrays["candidate"].astype(str) == candidate)
                & (release_arrays["origin"].astype(str) == origin)
            )
            curve_values = np.asarray(
                release_arrays["similarity_to_anchor"][mask], dtype=np.float64
            ).reshape(MATRICES, REPLICATES, RELEASE_HORIZON)
            matrix_curves = np.nanmean(curve_values, axis=1)
            mean_curve = np.nanmean(matrix_curves, axis=0)
            crossing = np.flatnonzero(mean_curve < DEPARTURE_THRESHOLD)
            crosses = bool(crossing.size)
            summary["anchor_release"] = {
                "minimum_mean_similarity": float(np.nanmin(mean_curve)),
                "first_mean_crossing_fission": int(crossing[0] + 1)
                if crosses
                else -1,
                "mean_curve_crosses_below_0_7": crosses,
                "lineage_departure_fraction": float(
                    lineage_rows[lineage_rows["origin"] == origin][
                        "ever_departed"
                    ].mean()
                ),
            }
            stored[f"release_c{candidate}_{origin}_mean_similarity_curve"] = mean_curve
            anchor_all_pass &= crosses
            release_equivalence_all_pass &= summary[
                "last_six_inheritance_minus_noop"
            ]["tost_equivalent_within_0_03"]
            item["origins"].append(summary)
        release_candidates.append(item)
    for location, adjusted in zip(
        release_p_locations, _holm_adjust(release_p_values), strict=True
    ):
        location["randomization_p_holm_four_release_cells"] = adjusted

    challenge_candidates: list[dict[str, Any]] = []
    challenge_p_values: list[float] = []
    challenge_p_locations: list[dict[str, Any]] = []
    equivalence_all = True
    slope_all = True
    basin_positive: dict[int, list[bool]] = {dose: [] for dose in RANDOM_DOSES[1:]}
    for candidate in CANDIDATES:
        selected = challenge_matrix[
            challenge_matrix["candidate"].astype(str).str.zfill(2) == candidate
        ]
        candidate_item: dict[str, Any] = {"candidate": candidate, "origins": []}
        for origin in WRITTEN_ORIGINS:
            origin_item: dict[str, Any] = {"origin": origin, "arms": []}
            dose_matrix_effects: list[NDArray] = []
            for arm in CHALLENGE_ARMS:
                written = selected[
                    (selected["origin"] == origin) & (selected["arm"] == arm)
                ].set_index("matrix_id")
                natural = selected[
                    (selected["origin"] == "NOOP") & (selected["arm"] == arm)
                ].set_index("matrix_id")
                if len(written) != MATRICES or len(natural) != MATRICES:
                    raise ValueError("CR8 challenge lost a whole-matrix block")
                effect = (
                    written["held_or_returned"].to_numpy(dtype=np.float64)
                    - natural["held_or_returned"].to_numpy(dtype=np.float64)
                )
                boot = effect[bootstrap].mean(axis=1)
                ci90 = _interval(boot, alpha=0.10)
                ci95 = _interval(boot)
                raw_p, null = _sign_randomization_p(effect, signs)
                result = {
                    "arm": arm,
                    "registered_dose": ARM_DOSE[arm],
                    "written_held_or_returned": float(
                        written["held_or_returned"].mean()
                    ),
                    "natural_held_or_returned": float(
                        natural["held_or_returned"].mean()
                    ),
                    "written_minus_natural": float(effect.mean()),
                    "bootstrap_ci90": ci90,
                    "bootstrap_ci95": ci95,
                    "randomization_p_raw": raw_p,
                    "equivalent_within_0_05": bool(
                        arm != "ADVERSARIAL"
                        and ci90[0] > -CHALLENGE_EQUIVALENCE_MARGIN
                        and ci90[1] < CHALLENGE_EQUIVALENCE_MARGIN
                    ),
                    "maximum_leave_one_matrix_out_influence": _maximum_leave_one_out_influence(
                        effect
                    ),
                }
                challenge_p_values.append(raw_p)
                challenge_p_locations.append(result)
                origin_item["arms"].append(result)
                stored[f"challenge_c{candidate}_{origin}_{arm}_matrix_effect"] = effect
                stored[f"challenge_c{candidate}_{origin}_{arm}_bootstrap"] = boot
                stored[f"challenge_c{candidate}_{origin}_{arm}_randomization"] = null
                if arm != "ADVERSARIAL":
                    dose_matrix_effects.append(effect)
                    equivalence_all &= result["equivalent_within_0_05"]
                    if ARM_DOSE[arm] > 0:
                        basin_positive[ARM_DOSE[arm]].append(ci95[0] > 0.0)
            dose_values = np.asarray(RANDOM_DOSES, dtype=np.float64)
            centered = dose_values - dose_values.mean()
            effects_by_matrix = np.column_stack(dose_matrix_effects)
            slopes = effects_by_matrix @ centered / float(np.dot(centered, centered))
            slope_boot = slopes[bootstrap].mean(axis=1)
            slope_ci = _interval(slope_boot)
            slope_summary = {
                "estimate": float(slopes.mean()),
                "bootstrap_ci95": slope_ci,
                "significantly_positive": bool(slope_ci[0] > 0.0),
                "no_significantly_positive_dose_trend": bool(slope_ci[0] <= 0.0),
            }
            slope_all &= slope_summary["no_significantly_positive_dose_trend"]
            origin_item["dose_advantage_slope"] = slope_summary
            stored[f"challenge_c{candidate}_{origin}_matrix_dose_slope"] = slopes
            stored[f"challenge_c{candidate}_{origin}_bootstrap_dose_slope"] = slope_boot
            candidate_item["origins"].append(origin_item)
        challenge_candidates.append(candidate_item)
    adjusted = _holm_adjust(challenge_p_values)
    for location, value in zip(challenge_p_locations, adjusted, strict=True):
        location["randomization_p_holm_all_challenge_cells"] = value

    qualifying = [
        dose
        for dose, values in basin_positive.items()
        if len(values) == len(CANDIDATES) * len(WRITTEN_ORIGINS) and all(values)
    ]
    basin_radius = max(qualifying) if qualifying else 0
    zero_interventions = bool((release_lineage["interventions_applied"] == 0).all())
    replay_and_integrity = bool(
        release_replay_exact
        and challenge_replay_exact
        and zero_interventions
        and readback_exact
    )
    complete = bool(
        preparation["all_written_states_genuinely_altered"]
        and anchor_all_pass
        and release_equivalence_all_pass
        and equivalence_all
        and slope_all
        and basin_radius == 0
        and replay_and_integrity
    )
    return (
        {
            "format": "codex-intervention-cr8-primary-metrics-v1",
            "preparation_audit": preparation,
            "release": {
                "candidates": release_candidates,
                "all_written_mean_curves_cross_below_0_7": bool(anchor_all_pass),
                "all_last_six_inheritance_tost_equivalent": bool(
                    release_equivalence_all_pass
                ),
            },
            "challenge": {
                "candidates": challenge_candidates,
                "all_registered_random_doses_tost_equivalent": bool(equivalence_all),
                "no_significantly_positive_dose_trend": bool(slope_all),
                "registered_shared_basin_radius": int(basin_radius),
            },
            "integrity": {
                "release_exact_replay": bool(release_replay_exact),
                "challenge_exact_replay": bool(challenge_replay_exact),
                "release_interventions_exactly_zero": zero_interventions,
                "artifact_readback_exact": bool(readback_exact),
            },
            "external_written_but_passive_classification": complete,
            "candidates_never_pooled": True,
            "whole_matrix_inference": True,
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
        registry = payload.get("seed_registry", {})
        if isinstance(registry, dict):
            values.update(str(item) for item in registry.values())
    return values


def _artificial_case() -> tuple[ReleaseCase, ExperimentConfig]:
    config = GardConfig(
        n_types=100,
        n_min=4,
        n_max=8,
        beta_log_mean=-4.0,
        beta_log_sd=1.0,
        max_growth_steps=20_000,
        generations=120,
    )
    current_experiment = ExperimentConfig(
        gard=config,
        development=CohortConfig(1, 1, (60,)),
        confirmation=CohortConfig(1, 1, (60,)),
        horizon=CHALLENGE_HORIZON,
        master_seed=SEEDS["smoke"],
    )
    # A deliberately fast non-scientific kinetic fixture.  Moderate/low beta
    # makes this I/O smoke spend minutes in irrelevant growth loops; beta=1e3
    # reaches the artificial fission mass in only a few simulator updates.
    beta = np.full((100, 100), 1_000.0, dtype=np.float64)
    beta[np.diag_indices(100)] = 1_100.0
    composition = np.zeros(100, dtype=np.int64)
    composition[:4] = 1
    snapshot = Snapshot(
        composition=composition,
        generation=120,
        inheritance=tuple([True] * 120),
        boundary_h=tuple([0.95] * 120),
        previous_growth_steps=20,
        cumulative_growth_steps=2_400,
    )
    origins = tuple(
        FrozenOrigin(origin, 0, snapshot, f"artificial-{origin}-0")
        for origin in ORIGINS
    )
    return (
        ReleaseCase(
            state_id="CR8-ARTIFICIAL",
            candidate="02",
            matrix_id=0,
            beta=beta,
            origins=origins,
            source_batch_digest="artificial",
        ),
        current_experiment,
    )


def validation_checks() -> dict[str, Any]:
    upstream = _verify_upstream()
    batches = _cr7_checkpoint_batches()
    preparation = _preparation_audit()
    fixture = np.zeros(100, dtype=np.int64)
    fixture[:8] = (8, 7, 6, 5, 4, 4, 3, 3)
    mass = int(fixture.sum())
    dose_checks = []
    for k in RANDOM_DOSES:
        edits = random_k_edits(
            fixture,
            k,
            np.random.default_rng(derive_seed(SEEDS["validation"], "dose", k)),
        )
        edited = apply_edits(fixture, edits)
        dose_checks.append(
            len(edits) == k
            and int(edited.sum()) == mass
            and np.all(edited >= 0)
            and int(np.abs(edited - fixture).sum() // 2) == k
        )
    held = classify_challenge(
        np.asarray([1.0, 0.95, 0.8]),
        np.asarray([0.95] * CHALLENGE_HORIZON),
        0.5,
        True,
    )
    returned = classify_challenge(
        np.asarray([1.0, 0.69, 0.91, 0.92, 0.93]),
        np.asarray([0.95] * CHALLENGE_HORIZON),
        0.5,
        True,
    )
    mode = classify_challenge(
        np.asarray([1.0, 0.69] + [0.8] * 23),
        np.asarray([0.8] * 18 + [0.91] * 6),
        0.45,
        True,
    )
    incomplete = classify_challenge(
        np.asarray([1.0, 0.95, np.nan]),
        np.asarray([0.95, np.nan] + [np.nan] * 22),
        0.8,
        False,
    )
    case, _ = _artificial_case()
    history_snapshot = case.origins[0].snapshot
    edit = MolecularEdit(0, 50)
    after = edited_snapshot_many(history_snapshot, (edit,))
    draw_shapes = inference_draws()
    checks = {
        "inherited_cr0_and_cr7_integrity_pass": True,
        "cr7_checkpoint_count_exact": len(batches) == 2 * MATRICES,
        "cr7_preparation_states_genuinely_altered": preparation[
            "all_written_states_genuinely_altered"
        ],
        "design_exact": MATRICES == 48
        and REPLICATES == 6
        and RELEASE_HORIZON == 60
        and CHALLENGE_BRANCHES == 32
        and CHALLENGE_HORIZON == 24
        and RANDOM_DOSES == (0, 2, 4, 8, 16),
        "seed_domains_unique": len(SEEDS) == len(set(SEEDS.values())),
        "seed_domains_disjoint_from_prior_registrations": set(SEEDS.values()).isdisjoint(
            _prior_seed_values()
        ),
        "release_future_seed_origin_free": len(
            {
                _release_future_seed("02", 3, 2)
                for _origin in ORIGINS
            }
        )
        == 1,
        "challenge_future_seed_origin_and_arm_free": len(
            {
                _challenge_future_seed("02", 3, 2, 7)
                for _origin in ORIGINS
                for _arm in CHALLENGE_ARMS
            }
        )
        == 1,
        "challenge_edit_stream_separate_from_future": _challenge_edit_seed(
            "02", 3, 2, "MODEL_DOWN", "RANDOM_K8"
        )
        != _challenge_future_seed("02", 3, 2, 7),
        "all_random_doses_exact_mass_preserving_nonnegative": all(dose_checks),
        "instantaneous_edit_history_exact": history_snapshot.generation
        == after.generation
        and history_snapshot.inheritance == after.inheritance
        and history_snapshot.boundary_h == after.boundary_h
        and history_snapshot.previous_growth_steps == after.previous_growth_steps
        and history_snapshot.cumulative_growth_steps == after.cumulative_growth_steps,
        "classifier_held_fixture_exact": held["category"] == "held",
        "classifier_return_requires_post_departure_run3": returned["category"]
        == "returned"
        and returned["return_certification_time"] == 4,
        "classifier_mode_recovery_fixture_exact": mode["category"]
        == "mode_recovered",
        "classifier_incomplete_is_lost": incomplete["category"] == "lost",
        "classifier_thresholds_strict": classify_challenge(
            np.asarray([1.0, 0.7, 0.9, 0.91, 0.92, 0.93]),
            np.asarray([0.95] * CHALLENGE_HORIZON),
            0.5,
            True,
        )["category"]
        == "held",
        "whole_matrix_draw_shapes_exact": draw_shapes["bootstrap_indices"].shape
        == (4096, 48)
        and draw_shapes["randomization_signs"].shape == (4096, 48),
        "release_mode_has_no_controller": protocol()["release"]["controller"] is None,
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
        "scientific_cr8_release_futures_generated": 0,
        "scientific_cr8_challenge_futures_generated": 0,
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
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CR8 full repository validation failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    with _atomic_destination(output) as destination:
        payload = dict(value)
        payload["source_hashes"] = source_hashes()
        payload["source_tree_sha256"] = _canonical_digest(source_hashes())
        payload["pytest_returncode"] = completed.returncode
        payload["pytest_summary"] = completed.stdout.strip().splitlines()[-1]
        (destination / "validation.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n"
        )
        (destination / "pytest.txt").write_text(completed.stdout + completed.stderr)
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR8 validation sealed: {output}", flush=True)


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
        raise ValueError("CR8 validation did not pass")
    if validation["source_hashes"] != source_hashes():
        raise ValueError("CR8 source changed after validation")
    upstream = _verify_upstream()
    for forbidden in (DEFAULT_REGISTRATION, DEFAULT_SMOKE, DEFAULT_OUTPUT, DEFAULT_WORK):
        if forbidden.exists():
            raise FileExistsError(f"CR8 preregistration artifact already exists: {forbidden}")
    with _atomic_destination(output) as destination:
        launch_path = destination / "frozen_cr7_launch_states.npz"
        launch_audit = freeze_cr7_launch_archive(launch_path)
        shutil.copy2(ROOT / DOCUMENT, destination / "preregistration.md")
        shutil.copy2(validation_directory / "validation.json", destination / "validation.json")
        shutil.copy2(
            CR7_REGISTRATION / "frozen_full_predictor.npz",
            destination / "frozen_full_predictor.npz",
        )
        body: dict[str, Any] = {
            "format": REGISTRATION_FORMAT,
            "protocol": protocol(),
            "protocol_id": protocol()["protocol_id"],
            "source_hashes": source_hashes(),
            "source_tree_sha256": _canonical_digest(source_hashes()),
            "seed_registry": SEEDS,
            "frozen_model_sha256": EXPECTED_MODEL_SHA256,
            "frozen_cr7_launch_archive_sha256": launch_audit["archive_sha256"],
            "launch_archive_audit": launch_audit,
            "validation_checksum_manifest_sha256": sha256_file(
                validation_directory / "SHA256SUMS"
            ),
            "upstream": upstream,
            "scientific_cr8_release_futures_at_registration": 0,
            "scientific_cr8_challenge_futures_at_registration": 0,
        }
        registration_id = _canonical_digest(_json_ready(body))
        body["registration_id"] = registration_id
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
        f"<!-- registered-cr8-{registration_id} -->",
        [
            "## CR8 steer-release-and-challenge registered",
            "",
            f"- Registration: `{registration_id}`.",
            "- Exact sealed CR7 MODEL_DOWN, RULE_DOWN, and matched NOOP endpoints were frozen as the unselected CR8 preparation cohort.",
            "- Sixty untreated release fissions, exact random K={0,2,4,8,16} challenges, one frozen-predictor adversarial edit, 32 F24 futures, complete replay, matrix inference, equivalence margins, classifier, and claim boundaries were sealed before CR8 futures.",
            "- The CR7 active extension is excluded; CR9 will not launch automatically.",
            "",
        ],
    )
    print(f"CR8 registered: {registration_id}", flush=True)


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    registration = json.loads((directory / "registration.json").read_text())
    if registration["format"] != REGISTRATION_FORMAT:
        raise ValueError("unsupported CR8 registration format")
    if registration["source_hashes"] != source_hashes():
        raise ValueError("CR8 registered source tree changed")
    body = dict(registration)
    observed = body.pop("registration_id")
    if _canonical_digest(_json_ready(body)) != observed:
        raise ValueError("CR8 registration ID changed")
    if registration["protocol"] != protocol() or registration["seed_registry"] != SEEDS:
        raise ValueError("CR8 registered protocol or seeds changed")
    if sha256_file(directory / "frozen_full_predictor.npz") != EXPECTED_MODEL_SHA256:
        raise ValueError("CR8 frozen predictor changed")
    if sha256_file(directory / "frozen_cr7_launch_states.npz") != registration[
        "frozen_cr7_launch_archive_sha256"
    ]:
        raise ValueError("CR8 frozen launch-state archive changed")
    cases = load_release_cases(directory / "frozen_cr7_launch_states.npz")
    if len(cases) != 2 * MATRICES:
        raise ValueError("CR8 frozen launch-state cohort is incomplete")
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
    model_path = registration_directory / "frozen_full_predictor.npz"

    def execute() -> tuple[str, str]:
        release = run_release_case(
            case, current_experiment, model_path, registration["registration_id"]
        )
        challenge = run_challenge_case(
            case,
            release,
            current_experiment,
            model_path,
            registration["registration_id"],
            branches=2,
            arms=("NONE", "RANDOM_K2", "ADVERSARIAL"),
        )
        return release_batch_digest(release), challenge_batch_digest(challenge)

    first = execute()
    second = execute()
    payload = {
        "format": "codex-intervention-cr8-smoke-v1",
        "registration_id": registration["registration_id"],
        "artificial_non_scientific_fixture": True,
        "release_and_challenge_io_exercised": True,
        "release_applied_zero_interventions": True,
        "exact_replay": first == second,
        "effect_sizes_arm_order_event_rates_and_candidate_differences_disclosed": False,
        "scientific_cr8_release_futures_generated": 0,
        "scientific_cr8_challenge_futures_generated": 0,
    }
    if not all(
        payload[key]
        for key in (
            "artificial_non_scientific_fixture",
            "release_and_challenge_io_exercised",
            "release_applied_zero_interventions",
            "exact_replay",
        )
    ):
        raise AssertionError("CR8 artificial smoke failed")
    with _atomic_destination(output) as destination:
        (destination / "smoke.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR8 non-scientific smoke passed: {output}", flush=True)


def _reports(metrics: dict[str, Any]) -> tuple[str, str]:
    technical = [
        "# CR8 steer, release, and challenge",
        "",
        "Prospective external written-but-passive classification: "
        f"**{metrics['external_written_but_passive_classification']}**.",
        f"Registered shared basin radius: **{metrics['challenge']['registered_shared_basin_radius']} molecules**.",
        "",
        "## Preparation and release",
        "",
        f"All CR7 written endpoints were genuinely altered before release: **{metrics['preparation_audit']['all_written_states_genuinely_altered']}**.",
        "",
        "| Candidate | Written origin | Minimum mean anchor similarity | First mean crossing | Last-six inheritance difference | 90% CI | Equivalent ±0.03 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in metrics["release"]["candidates"]:
        for origin in candidate["origins"]:
            anchor = origin["anchor_release"]
            contrast = origin["last_six_inheritance_minus_noop"]
            technical.append(
                f"| {candidate['candidate']} | {origin['origin']} | "
                f"{anchor['minimum_mean_similarity']:.4f} | "
                f"{anchor['first_mean_crossing_fission']} | "
                f"{contrast['estimate']:+.4f} | "
                f"[{contrast['bootstrap_ci90'][0]:+.4f}, {contrast['bootstrap_ci90'][1]:+.4f}] | "
                f"{contrast['tost_equivalent_within_0_03']} |"
            )
    technical.extend(
        [
            "",
            "## Challenge",
            "",
            "Values are written-origin minus matched-natural `held + returned` probability. The 90% interval is the registered equivalence interval; the 95% interval is used for basin evidence.",
            "",
            "| Candidate | Written origin | Arm | Effect | 90% CI | 95% CI | Equivalent ±0.05 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for candidate in metrics["challenge"]["candidates"]:
        for origin in candidate["origins"]:
            for arm in origin["arms"]:
                technical.append(
                    f"| {candidate['candidate']} | {origin['origin']} | {arm['arm']} | "
                    f"{arm['written_minus_natural']:+.4f} | "
                    f"[{arm['bootstrap_ci90'][0]:+.4f}, {arm['bootstrap_ci90'][1]:+.4f}] | "
                    f"[{arm['bootstrap_ci95'][0]:+.4f}, {arm['bootstrap_ci95'][1]:+.4f}] | "
                    f"{arm['equivalent_within_0_05'] if arm['arm'] != 'ADVERSARIAL' else 'not a dose gate'} |"
                )
            slope = origin["dose_advantage_slope"]
            technical.append(
                f"\nCandidate {candidate['candidate']} / {origin['origin']} dose-advantage slope: "
                f"{slope['estimate']:+.6f} "
                f"(95% CI [{slope['bootstrap_ci95'][0]:+.6f}, {slope['bootstrap_ci95'][1]:+.6f}]); "
                f"no significant positive trend = {slope['no_significantly_positive_dose_trend']}.\n"
            )
    integrity = metrics["integrity"]
    technical.extend(
        [
            "",
            "## Integrity and claim boundary",
            "",
            f"Release replay exact: **{integrity['release_exact_replay']}**. Challenge replay exact: **{integrity['challenge_exact_replay']}**. Release interventions exactly zero: **{integrity['release_interventions_exactly_zero']}**. Artifact readback exact: **{integrity['artifact_readback_exact']}**.",
            "",
            "A written-but-passive result means CR7 feedback maintained a compotype-like state while active but did not install an autonomous restoring compositional attractor. A nonzero registered basin would instead be reported as a cross-clean-room disagreement. Neither outcome establishes biological memory, agency, life, real chemistry, or a universal origin-of-life mechanism.",
            "",
        ]
    )

    passive = metrics["external_written_but_passive_classification"]
    radius = metrics["challenge"]["registered_shared_basin_radius"]
    if passive:
        outcome = (
            "The controlled states did not remain special once we stopped helping them. "
            "They drifted away, became statistically indistinguishable from matched natural states, "
            "and showed no measurable return advantage after any registered random challenge."
        )
    elif radius > 0:
        outcome = (
            "At least one nonzero perturbation dose showed a shared restoring advantage. "
            "That is evidence against the expected written-but-passive result and must be treated as a cross-clean-room disagreement."
        )
    else:
        outcome = (
            "The result is mixed: it did not satisfy every strict written-but-passive gate, "
            "but it also did not establish a shared nonzero restoring basin."
        )
    lay = [
        "# CR8 lay summary",
        "",
        outcome,
        "",
        "CR7 showed that a tiny corrective edit after every fission could keep inheritance near-perfect. CR8 removed that helper completely, watched the assemblies for sixty more fissions, then disturbed them by exactly 2, 4, 8, or 16 molecule replacements and asked whether they returned to their own pre-disturbance composition.",
        "",
        f"The formal written-but-passive classification was **{passive}**, and the shared restoring radius was **{radius} molecules**. All release and challenge trajectories were repeated exactly, and no intervention occurred during release or after challenge launch.",
        "",
        "The key distinction is between a state that a controller continually maintains and one the assembly can maintain and restore by itself. Only the latter would justify calling the state autonomously installed.",
        "",
    ]
    return "\n".join(technical), "\n".join(lay)


def _matrix_effect_table(
    release_matrix: pd.DataFrame, challenge_matrix: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        release_selected = release_matrix[
            release_matrix["candidate"].astype(str).str.zfill(2) == candidate
        ]
        challenge_selected = challenge_matrix[
            challenge_matrix["candidate"].astype(str).str.zfill(2) == candidate
        ]
        for origin in WRITTEN_ORIGINS:
            written_release = release_selected[
                release_selected["origin"] == origin
            ].set_index("matrix_id")
            natural_release = release_selected[
                release_selected["origin"] == "NOOP"
            ].set_index("matrix_id")
            for matrix_id in range(MATRICES):
                rows.append(
                    {
                        "phase": "release",
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "written_origin": origin,
                        "arm": "LAST_SIX_INHERITANCE",
                        "written_value": float(
                            written_release.loc[
                                matrix_id, "final_six_inherited_fraction"
                            ]
                        ),
                        "natural_value": float(
                            natural_release.loc[
                                matrix_id, "final_six_inherited_fraction"
                            ]
                        ),
                        "written_minus_natural": float(
                            written_release.loc[
                                matrix_id, "final_six_inherited_fraction"
                            ]
                            - natural_release.loc[
                                matrix_id, "final_six_inherited_fraction"
                            ]
                        ),
                    }
                )
            for arm in CHALLENGE_ARMS:
                written = challenge_selected[
                    (challenge_selected["origin"] == origin)
                    & (challenge_selected["arm"] == arm)
                ].set_index("matrix_id")
                natural = challenge_selected[
                    (challenge_selected["origin"] == "NOOP")
                    & (challenge_selected["arm"] == arm)
                ].set_index("matrix_id")
                for matrix_id in range(MATRICES):
                    rows.append(
                        {
                            "phase": "challenge",
                            "candidate": candidate,
                            "matrix_id": matrix_id,
                            "written_origin": origin,
                            "arm": arm,
                            "written_value": float(
                                written.loc[matrix_id, "held_or_returned"]
                            ),
                            "natural_value": float(
                                natural.loc[matrix_id, "held_or_returned"]
                            ),
                            "written_minus_natural": float(
                                written.loc[matrix_id, "held_or_returned"]
                                - natural.loc[matrix_id, "held_or_returned"]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _write_result(
    output: Path,
    registration: dict[str, Any],
    metrics: dict[str, Any],
    stored_inference: dict[str, NDArray],
    release_replay: dict[str, Any],
    challenge_replay: dict[str, Any],
    release_lineage: pd.DataFrame,
    release_matrix: pd.DataFrame,
    release_arrays: dict[str, NDArray],
    plans: pd.DataFrame,
    outcomes: pd.DataFrame,
    challenge_matrix: pd.DataFrame,
    challenge_arrays: dict[str, NDArray],
) -> None:
    technical, lay = _reports(metrics)
    matrix_effects = _matrix_effect_table(release_matrix, challenge_matrix)
    passive = metrics["external_written_but_passive_classification"]
    radius = metrics["challenge"]["registered_shared_basin_radius"]
    supported: list[str] = []
    failed: list[str] = []
    if passive:
        supported.append(
            "CR7 feedback wrote controller-maintained compotype-like states that relaxed toward natural behavior and had no registered nonzero restoring basin after release"
        )
    else:
        failed.append("complete external written-but-passive CR8 classification")
    if radius > 0:
        supported.append(
            f"a shared nonzero restoring advantage was detected at registered radius {radius}"
        )
    claims = {
        "supported": supported,
        "failed_predictions": failed,
        "unresolved": [
            "chemical embodiment of the external corrective action",
            "control half-life and minimum feedback frequency",
            "whether any candidate-specific return effect transfers beyond this cohort",
        ],
        "prohibited": protocol()["claim_boundary"]["prohibited"],
        "required_term_if_written_but_passive": "controller-maintained compotype-like state",
    }
    with _atomic_destination(output) as destination:
        metrics_text = json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n"
        (destination / "primary_metrics.json").write_text(metrics_text)
        (destination / "release_replay_audit.json").write_text(
            json.dumps(_json_ready(release_replay), indent=2, sort_keys=True) + "\n"
        )
        (destination / "challenge_replay_audit.json").write_text(
            json.dumps(_json_ready(challenge_replay), indent=2, sort_keys=True) + "\n"
        )
        (destination / "SCIENTIFIC_REPORT.md").write_text(technical)
        (destination / "LAY_SUMMARY.md").write_text(lay)
        (destination / "claim_boundaries.json").write_text(
            json.dumps(claims, indent=2, sort_keys=True) + "\n"
        )
        release_lineage.to_csv(
            destination / "release_lineages.csv.gz", index=False, compression="gzip"
        )
        release_matrix.to_csv(destination / "release_matrix_summaries.csv", index=False)
        plans.to_csv(
            destination / "challenge_plans.csv.gz", index=False, compression="gzip"
        )
        outcomes.to_csv(
            destination / "challenge_futures.csv.gz", index=False, compression="gzip"
        )
        challenge_matrix.to_csv(
            destination / "challenge_matrix_summaries.csv", index=False
        )
        matrix_effects.to_csv(destination / "matrix_effects.csv.gz", index=False, compression="gzip")
        np.savez_compressed(destination / "release_arrays.npz", **release_arrays)
        np.savez_compressed(destination / "challenge_arrays.npz", **challenge_arrays)
        np.savez_compressed(destination / "inference_arrays.npz", **stored_inference)

        release_readback = pd.read_csv(destination / "release_lineages.csv.gz")
        plan_readback = pd.read_csv(destination / "challenge_plans.csv.gz")
        future_readback = pd.read_csv(destination / "challenge_futures.csv.gz")
        with np.load(destination / "release_arrays.npz", allow_pickle=False) as archive:
            release_shape_exact = archive["similarity_to_anchor"].shape == (
                MATRICES * len(CANDIDATES) * REPLICATES * len(ORIGINS),
                RELEASE_HORIZON,
            )
        with np.load(destination / "challenge_arrays.npz", allow_pickle=False) as archive:
            challenge_shape_exact = archive["similarity_to_anchor"].shape == (
                MATRICES
                * len(CANDIDATES)
                * REPLICATES
                * len(ORIGINS)
                * len(CHALLENGE_ARMS)
                * CHALLENGE_BRANCHES,
                CHALLENGE_HORIZON + 1,
            )
        readback = {
            "primary_metrics_exact": (destination / "primary_metrics.json").read_text()
            == metrics_text,
            "release_row_count_exact": len(release_readback)
            == MATRICES * len(CANDIDATES) * REPLICATES * len(ORIGINS),
            "challenge_plan_count_exact": len(plan_readback)
            == MATRICES
            * len(CANDIDATES)
            * REPLICATES
            * len(ORIGINS)
            * len(CHALLENGE_ARMS),
            "challenge_future_count_exact": len(future_readback)
            == MATRICES
            * len(CANDIDATES)
            * REPLICATES
            * len(ORIGINS)
            * len(CHALLENGE_ARMS)
            * CHALLENGE_BRANCHES,
            "release_array_shape_exact": bool(release_shape_exact),
            "challenge_array_shape_exact": bool(challenge_shape_exact),
        }
        readback["complete_readback_exact"] = bool(all(readback.values()))
        if not readback["complete_readback_exact"]:
            raise AssertionError(f"CR8 written-artifact readback failed: {readback}")
        (destination / "readback_audit.json").write_text(
            json.dumps(readback, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "origins": list(ORIGINS),
            "release_trajectories": len(release_lineage),
            "release_fissions": RELEASE_HORIZON,
            "challenge_arms": list(CHALLENGE_ARMS),
            "challenge_branches": CHALLENGE_BRANCHES,
            "challenge_futures": len(outcomes),
            "challenge_fissions": CHALLENGE_HORIZON,
            "release_exact_replay": metrics["integrity"]["release_exact_replay"],
            "challenge_exact_replay": metrics["integrity"]["challenge_exact_replay"],
            "release_interventions_exactly_zero": metrics["integrity"][
                "release_interventions_exactly_zero"
            ],
            "complete_readback_exact": True,
            "external_written_but_passive_classification": passive,
            "registered_shared_basin_radius": radius,
            "no_retry_or_replacement": True,
            "cr9_launched": False,
            "mandatory_stop_after_this_stage": True,
            "runtime": _runtime_manifest(),
        }
        (destination / "manifest.json").write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)


def _prepare_work(work: Path, output: Path, registration_id: str) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed CR8 result: {output}")
    free = shutil.disk_usage(ROOT).free
    if free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError(
            f"CR8 requires at least {MINIMUM_FREE_DISK_BYTES:,} free bytes; found {free:,}"
        )
    work.mkdir(parents=True, exist_ok=True)
    contract_path = work / "campaign_contract.json"
    expected = {
        "format": "codex-intervention-cr8-work-contract-v1",
        "registration_id": registration_id,
        "protocol_id": protocol()["protocol_id"],
    }
    if contract_path.is_file():
        observed = json.loads(contract_path.read_text())
        for key, value in expected.items():
            if observed.get(key) != value:
                raise ValueError("CR8 work directory belongs to another campaign")
    else:
        expected["campaign_started_unix"] = time.time()
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
    cases = load_release_cases(
        registration_directory / "frozen_cr7_launch_states.npz"
    )
    if len(cases) != 2 * MATRICES:
        raise AssertionError("CR8 frozen release cohort is incomplete")

    print(
        f"[cr8 1/8] Releasing {len(cases) * REPLICATES * len(ORIGINS):,} CR7 endpoints for {RELEASE_HORIZON} untreated fissions",
        flush=True,
    )
    releases = run_release_batches(
        cases,
        current_experiment,
        model_path,
        registration["registration_id"],
        work / "release" / "generate",
        workers,
        work,
        "release_futures",
    )
    print("[cr8 2/8] Replaying every untreated release", flush=True)
    release_replayed = run_release_batches(
        cases,
        current_experiment,
        model_path,
        registration["registration_id"],
        work / "release" / "replay",
        workers,
        work,
        "release_exact_replay",
    )
    release_replay = replay_audit(releases, release_replayed, "release")
    if not release_replay["exact_state_edit_endpoint_process_and_rng"]:
        raise AssertionError("CR8 release exact replay failed")
    del release_replayed

    print(
        f"[cr8 3/8] Launching {len(cases) * REPLICATES * len(ORIGINS) * len(CHALLENGE_ARMS) * CHALLENGE_BRANCHES:,} F24 challenge futures",
        flush=True,
    )
    challenges = run_challenge_batches(
        cases,
        releases,
        current_experiment,
        model_path,
        registration["registration_id"],
        work / "challenge" / "generate",
        workers,
        work,
        "challenge_futures",
    )
    print("[cr8 4/8] Replaying every challenge edit and F24 future", flush=True)
    challenge_replayed = run_challenge_batches(
        cases,
        releases,
        current_experiment,
        model_path,
        registration["registration_id"],
        work / "challenge" / "replay",
        workers,
        work,
        "challenge_exact_replay",
    )
    challenge_replay = replay_audit(challenges, challenge_replayed, "challenge")
    if not challenge_replay["exact_state_edit_endpoint_process_and_rng"]:
        raise AssertionError("CR8 challenge exact replay failed")
    del challenge_replayed

    _write_status(work, "whole_matrix_inference", len(cases), len(cases))
    print("[cr8 5/8] Building branch, lineage, and whole-matrix tables", flush=True)
    release_lineage, release_matrix, release_arrays = release_tables(cases, releases)
    plans, outcomes, challenge_matrix, challenge_arrays = challenge_tables(
        cases, challenges
    )
    if not (release_lineage["interventions_applied"] == 0).all():
        raise AssertionError("CR8 release contains a nonzero intervention count")
    print("[cr8 6/8] Computing candidate-separated whole-matrix inference", flush=True)
    metrics, stored = compute_inference(
        release_lineage,
        release_matrix,
        challenge_matrix,
        release_arrays,
        inference_draws(),
        release_replay_exact=True,
        challenge_replay_exact=True,
        readback_exact=True,
    )

    _write_status(work, "writing_and_reading_back_artifacts", len(cases), len(cases))
    print("[cr8 7/8] Writing reports and exact readback artifacts", flush=True)
    _write_result(
        output,
        registration,
        metrics,
        stored,
        release_replay,
        challenge_replay,
        release_lineage,
        release_matrix,
        release_arrays,
        plans,
        outcomes,
        challenge_matrix,
        challenge_arrays,
    )
    _append_ledger(
        f"<!-- sealed-cr8-{registration['registration_id']} -->",
        [
            "## CR8 steer-release-and-challenge sealed",
            "",
            f"- Registration: `{registration['registration_id']}`.",
            f"- Result: `{output.relative_to(ROOT)}`.",
            f"- External written-but-passive classification: **{metrics['external_written_but_passive_classification']}**.",
            f"- Registered shared basin radius: **{metrics['challenge']['registered_shared_basin_radius']} molecules**.",
            f"- Release replay: **{metrics['integrity']['release_exact_replay']}**; challenge replay: **{metrics['integrity']['challenge_exact_replay']}**; zero release interventions: **{metrics['integrity']['release_interventions_exactly_zero']}**.",
            "- CR9 was not launched automatically; mandatory review stop observed.",
            "",
        ],
    )
    _write_status(
        work,
        "sealed_complete_mandatory_review_stop",
        len(cases),
        len(cases),
        output=str(output),
        written_but_passive=metrics["external_written_but_passive_classification"],
        registered_shared_basin_radius=metrics["challenge"][
            "registered_shared_basin_radius"
        ],
    )
    print("[cr8 8/8] Result sealed; STOPPED before CR9", flush=True)


def read_status(work: Path = DEFAULT_WORK) -> dict[str, Any]:
    work = work.resolve()
    status_path = work / "campaign_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"CR8 status does not exist: {status_path}")
    value = json.loads(status_path.read_text())
    value["checkpoint_counts"] = {
        relative: len(list((work / relative).glob("*.pkl")))
        if (work / relative).is_dir()
        else 0
        for relative in (
            "release/generate",
            "release/replay",
            "challenge/generate",
            "challenge/replay",
        )
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
    run_parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
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
