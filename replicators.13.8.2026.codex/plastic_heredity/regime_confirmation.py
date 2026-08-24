"""Prospective confirmation of coherent, distinct, persistent hereditary regimes.

This workflow is deliberately separate from every earlier registered campaign.
It has three irreversible stages:

``register-design``
    Seal endpoint, cohort, feature, seed, inference, and power contracts before
    generating development matrices.
``develop``
    Verify the design, generate the development cohort, select penalties using
    development matrices only, and seal portable models.
``confirm``
    Verify both seals, generate a disjoint untouched cohort, score it once, and
    evaluate the prespecified occurrence and prediction gates.

The primary endpoint requires eight consecutive strict-H>0.90 inheritances,
mutual strict-H>0.90 similarity among all eight daughters, and inclusive
H<=0.85 separation of every daughter from the pre-break parent.  Two frozen
secondary definitions are reported but cannot rescue a failed primary test.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from .config import CANDIDATES, CohortConfig, ExperimentConfig
from .experiment import StateCase, _json_ready, _runtime_manifest, build_cohort
from .mechanistic import (
    MECHCONF_MASTER_SEED,
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .mechanistic_metrics import (
    _paired_gain,
    _rank_metrics,
    _reliability_bootstrap,
    _state_brier,
    _state_log_loss,
    holm_adjust,
    paired_matrix_randomization_p,
)
from .mechanistic_v2 import MECHCONF2_MASTER_SEED
from .mechanistic_v2_features import (
    MechanisticV2RawFeatures,
    extract_mechanistic_v2_features,
    provenance_contract,
)
from .mechanistic_v2_models import (
    CV_FOLDS,
    RIDGE_LAMBDAS,
    CandidateRegistryV2,
    fit_candidate_registry_v2,
    load_registries_v2,
    predict_candidate_registry_v2,
    save_registries_v2,
)
from .memory import MEMORY_CONFIRM_MASTER_SEED
from .seeds import derive_seed
from .simulator import FissionRecord, cosine_similarity, simulate_future_absorbing

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DESIGN_FORMAT = "plastic-heredity-regime-design-v1"
DEVELOPMENT_FORMAT = "plastic-heredity-regime-development-v1"
CONFIRMATION_FORMAT = "plastic-heredity-regime-confirmation-v1"

PRIMARY_ENDPOINT = "primary_all8"
SECONDARY_FIRST5 = "secondary_first5"
SECONDARY_CENTROID = "secondary_centroid"
ENDPOINTS = (PRIMARY_ENDPOINT, SECONDARY_FIRST5, SECONDARY_CENTROID)

INHERITANCE_THRESHOLD = 0.90
COHERENCE_THRESHOLD = 0.90
DISTINCTNESS_THRESHOLD = 0.85
RUN_LENGTH = 8
FIRST5_LENGTH = 5
HORIZON = 32
MATRICES = 200
BRANCHES = 128
LANDMARKS = (20, 35, 50, 65, 80)
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
MINIMUM_EVENTS = 100
MINIMUM_EVENT_MATRICES = 20
CONTINUOUS_TOLERANCE = 1e-14
PORTABLE_PREDICTION_TOLERANCE = 1e-12

DEVELOPMENT_MASTER_SEED = (
    "4ab47ce6f50cc194f28418be8cf8048d9e47972eb2864492f6dc3449869fa931"
)
CONFIRMATION_MASTER_SEED = (
    "c498f2e934bfd17e0fb175dee2332eedec9e4c5914ab58fa621aaf7c5d99ac9a"
)
BOOTSTRAP_MASTER_SEED = (
    "78fc1508f7e42bdcbbbee9576bff758cf187b9754831ae8f14973063bd33846e"
)
RANDOMIZATION_MASTER_SEED = (
    "c22ae1e6071d072fa707ca31c77b682a339768761710decb5d2cd2559a76c8c2"
)

SOURCE_FILES = (
    "plastic_heredity/config.py",
    "plastic_heredity/experiment.py",
    "plastic_heredity/features.py",
    "plastic_heredity/mechanistic.py",
    "plastic_heredity/mechanistic_metrics.py",
    "plastic_heredity/mechanistic_v2.py",
    "plastic_heredity/mechanistic_v2_features.py",
    "plastic_heredity/mechanistic_v2_models.py",
    "plastic_heredity/memory.py",
    "plastic_heredity/metrics.py",
    "plastic_heredity/regime_confirmation.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/simulator.py",
    "requirements-lock.txt",
)


@dataclass(frozen=True)
class WindowGeometry:
    minimum_pairwise_all8: float
    minimum_pairwise_first5: float
    maximum_anchor_all8: float
    maximum_anchor_first5: float
    minimum_centroid_all8: float


@dataclass(frozen=True)
class RegimeOutcome:
    primary_all8: bool
    secondary_first5: bool
    secondary_centroid: bool
    primary_all8_onset: int
    secondary_first5_onset: int
    secondary_centroid_onset: int
    first_break_index: int
    first_run8_start: int
    first_run8_minimum_pairwise_all8: float
    first_run8_minimum_pairwise_first5: float
    first_run8_maximum_anchor_all8: float
    first_run8_maximum_anchor_first5: float
    first_run8_minimum_centroid_all8: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegimeBranchBatch:
    targets: NDArray[np.int8]
    onsets: NDArray[np.int16]
    completed_horizon: NDArray[np.int8]
    observed_fissions: NDArray[np.int16]
    first_break_index: NDArray[np.int16]
    first_run8_start: NDArray[np.int16]
    geometry: NDArray[np.float64]


GEOMETRY_COLUMNS = (
    "first_run8_minimum_pairwise_all8",
    "first_run8_minimum_pairwise_first5",
    "first_run8_maximum_anchor_all8",
    "first_run8_maximum_anchor_first5",
    "first_run8_minimum_centroid_all8",
)


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def _experiment(master_seed: str) -> ExperimentConfig:
    cohort = CohortConfig(
        matrices=MATRICES,
        branches_per_state=BRANCHES,
        landmarks=LANDMARKS,
    )
    return ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=HORIZON,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
        regenerate_confirmation=True,
        master_seed=master_seed,
    )


def _pairwise_minimum(vectors: Sequence[NDArray]) -> float:
    if len(vectors) < 2:
        raise ValueError("pairwise coherence requires at least two vectors")
    return min(
        cosine_similarity(vectors[left], vectors[right])
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    )


def _window_geometry(
    records: list[FissionRecord], start: int, anchor: NDArray
) -> WindowGeometry:
    window = records[start : start + RUN_LENGTH]
    if len(window) != RUN_LENGTH:
        raise ValueError("regime geometry requires a complete eight-fission window")
    daughters = [record.daughter for record in window]
    centroid = np.mean(np.vstack(daughters).astype(np.float64), axis=0)
    anchor_similarities = [cosine_similarity(anchor, item) for item in daughters]
    centroid_similarities = [cosine_similarity(centroid, item) for item in daughters]
    return WindowGeometry(
        minimum_pairwise_all8=_pairwise_minimum(daughters),
        minimum_pairwise_first5=_pairwise_minimum(daughters[:FIRST5_LENGTH]),
        maximum_anchor_all8=max(anchor_similarities),
        maximum_anchor_first5=max(anchor_similarities[:FIRST5_LENGTH]),
        minimum_centroid_all8=min(centroid_similarities),
    )


def evaluate_regime(
    records: list[FissionRecord],
    inheritance_threshold: float = INHERITANCE_THRESHOLD,
    coherence_threshold: float = COHERENCE_THRESHOLD,
    distinctness_threshold: float = DISTINCTNESS_THRESHOLD,
) -> RegimeOutcome:
    """Evaluate the three prospectively frozen regime endpoints."""

    inherited = np.asarray(
        [record.h > inheritance_threshold for record in records], dtype=bool
    )
    breaks = np.flatnonzero(~inherited)
    if breaks.size == 0:
        return RegimeOutcome(
            False,
            False,
            False,
            -1,
            -1,
            -1,
            -1,
            -1,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    first_break = int(breaks[0])
    anchor = records[first_break].parent
    onsets = {name: -1 for name in ENDPOINTS}
    first_run_start = -1
    first_geometry: WindowGeometry | None = None
    last_start = len(records) - RUN_LENGTH
    for start in range(first_break + 1, last_start + 1):
        if not bool(inherited[start : start + RUN_LENGTH].all()):
            continue
        geometry = _window_geometry(records, start, anchor)
        if first_run_start < 0:
            first_run_start = start
            first_geometry = geometry
        if (
            onsets[PRIMARY_ENDPOINT] < 0
            and geometry.minimum_pairwise_all8 > coherence_threshold
            and geometry.maximum_anchor_all8 <= distinctness_threshold
        ):
            onsets[PRIMARY_ENDPOINT] = start
        if (
            onsets[SECONDARY_FIRST5] < 0
            and geometry.minimum_pairwise_first5 > coherence_threshold
            and geometry.maximum_anchor_first5 <= distinctness_threshold
        ):
            onsets[SECONDARY_FIRST5] = start
        if (
            onsets[SECONDARY_CENTROID] < 0
            and geometry.minimum_centroid_all8 > coherence_threshold
            and geometry.maximum_anchor_all8 <= distinctness_threshold
        ):
            onsets[SECONDARY_CENTROID] = start
        if all(onsets[name] >= 0 for name in ENDPOINTS):
            break

    if first_geometry is None:
        geometry_values = (np.nan,) * len(GEOMETRY_COLUMNS)
    else:
        geometry_values = (
            first_geometry.minimum_pairwise_all8,
            first_geometry.minimum_pairwise_first5,
            first_geometry.maximum_anchor_all8,
            first_geometry.maximum_anchor_first5,
            first_geometry.minimum_centroid_all8,
        )
    return RegimeOutcome(
        primary_all8=onsets[PRIMARY_ENDPOINT] >= 0,
        secondary_first5=onsets[SECONDARY_FIRST5] >= 0,
        secondary_centroid=onsets[SECONDARY_CENTROID] >= 0,
        primary_all8_onset=onsets[PRIMARY_ENDPOINT],
        secondary_first5_onset=onsets[SECONDARY_FIRST5],
        secondary_centroid_onset=onsets[SECONDARY_CENTROID],
        first_break_index=first_break,
        first_run8_start=first_run_start,
        first_run8_minimum_pairwise_all8=float(geometry_values[0]),
        first_run8_minimum_pairwise_first5=float(geometry_values[1]),
        first_run8_maximum_anchor_all8=float(geometry_values[2]),
        first_run8_maximum_anchor_first5=float(geometry_values[3]),
        first_run8_minimum_centroid_all8=float(geometry_values[4]),
    )


def _branch_worker(args: tuple[StateCase, ExperimentConfig, int]) -> RegimeBranchBatch:
    case, experiment, branches = args
    limiter = threadpool_limits(limits=1)
    try:
        targets = np.empty((branches, len(ENDPOINTS)), dtype=np.int8)
        onsets = np.empty((branches, len(ENDPOINTS)), dtype=np.int16)
        completed = np.empty(branches, dtype=np.int8)
        observed = np.empty(branches, dtype=np.int16)
        first_break = np.empty(branches, dtype=np.int16)
        first_run = np.empty(branches, dtype=np.int16)
        geometry = np.empty((branches, len(GEOMETRY_COLUMNS)), dtype=np.float64)
        contract = CANDIDATES[case.candidate]
        for branch in range(branches):
            rng = np.random.default_rng(
                derive_seed(
                    experiment.master_seed,
                    f"{case.cohort}.future",
                    case.candidate,
                    case.matrix_id,
                    case.landmark,
                    branch,
                )
            )
            records, completed_horizon = simulate_future_absorbing(
                case.snapshot,
                case.beta,
                experiment.gard,
                contract,
                experiment.horizon,
                rng,
            )
            outcome = evaluate_regime(records)
            values = outcome.to_dict()
            targets[branch] = [int(values[name]) for name in ENDPOINTS]
            onsets[branch] = [int(values[f"{name}_onset"]) for name in ENDPOINTS]
            completed[branch] = int(completed_horizon)
            observed[branch] = len(records)
            first_break[branch] = outcome.first_break_index
            first_run[branch] = outcome.first_run8_start
            geometry[branch] = [float(values[name]) for name in GEOMETRY_COLUMNS]
        return RegimeBranchBatch(
            targets=targets,
            onsets=onsets,
            completed_horizon=completed,
            observed_fissions=observed,
            first_break_index=first_break,
            first_run8_start=first_run,
            geometry=geometry,
        )
    finally:
        limiter.restore_original_limits()


def run_regime_branches(
    cases: list[StateCase],
    experiment: ExperimentConfig,
    branches: int,
    workers: int,
    label: str,
) -> list[RegimeBranchBatch]:
    arguments = [(case, experiment, branches) for case in cases]
    total = len(arguments)
    progress_every = max(1, min(25, total // 20))
    batches: list[RegimeBranchBatch] = []
    if workers <= 1:
        iterator: Iterable[RegimeBranchBatch] = map(_branch_worker, arguments)
        for index, batch in enumerate(iterator, start=1):
            batches.append(batch)
            if index % progress_every == 0 or index == total:
                print(f"[{label}] states {index}/{total}", flush=True)
        return batches
    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(_branch_worker, arguments, chunksize=1)
        for index, batch in enumerate(iterator, start=1):
            batches.append(batch)
            if index % progress_every == 0 or index == total:
                print(f"[{label}] states {index}/{total}", flush=True)
    return batches


def _batch_digest(batches: list[RegimeBranchBatch]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        for values in (
            batch.targets,
            batch.onsets,
            batch.completed_horizon,
            batch.observed_fissions,
            batch.first_break_index,
            batch.first_run8_start,
        ):
            digest.update(np.ascontiguousarray(values).tobytes())
        canonical = np.nan_to_num(
            batch.geometry, nan=-999.0, posinf=999.0, neginf=-999.0
        )
        digest.update(np.ascontiguousarray(canonical).tobytes())
    return digest.hexdigest()


def _replay_audit(
    original: list[RegimeBranchBatch], replay: list[RegimeBranchBatch]
) -> dict[str, Any]:
    if len(original) != len(replay):
        raise ValueError("replay batch count differs")
    discrete_exact = True
    maximum_error = 0.0
    continuous_values = 0
    for left, right in zip(original, replay):
        for name in (
            "targets",
            "onsets",
            "completed_horizon",
            "observed_fissions",
            "first_break_index",
            "first_run8_start",
        ):
            discrete_exact &= bool(
                np.array_equal(getattr(left, name), getattr(right, name))
            )
        finite = np.isfinite(left.geometry) | np.isfinite(right.geometry)
        if not np.array_equal(np.isfinite(left.geometry), np.isfinite(right.geometry)):
            maximum_error = float("inf")
        elif finite.any():
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(left.geometry[finite] - right.geometry[finite]))),
            )
            continuous_values += int(finite.sum())
    if not discrete_exact:
        raise ValueError("discrete endpoint replay mismatch")
    if maximum_error > CONTINUOUS_TOLERANCE:
        raise ValueError(f"continuous replay mismatch: {maximum_error}")
    return {
        "discrete_exact": discrete_exact,
        "continuous_values_compared": continuous_values,
        "maximum_continuous_absolute_error": maximum_error,
        "continuous_within_1e-14": maximum_error <= CONTINUOUS_TOLERANCE,
        "first_digest": _batch_digest(original),
        "second_digest": _batch_digest(replay),
        "digests_exact": _batch_digest(original) == _batch_digest(replay),
    }


def _stack_endpoint_labels(
    batches: list[RegimeBranchBatch], endpoint: str
) -> NDArray[np.int8]:
    endpoint_index = ENDPOINTS.index(endpoint)
    return np.stack([batch.targets[:, endpoint_index] for batch in batches])


def _all_labels(
    batches: list[RegimeBranchBatch],
) -> dict[str, NDArray[np.int8]]:
    return {name: _stack_endpoint_labels(batches, name) for name in ENDPOINTS}


def _metadata_arrays(
    cases: list[StateCase],
) -> tuple[NDArray[np.str_], NDArray[np.int64]]:
    candidates = np.asarray([case.candidate for case in cases])
    matrix_ids = np.asarray([case.matrix_id for case in cases], dtype=np.int64)
    return candidates, matrix_ids


def _candidate_mask(candidates: NDArray[np.str_], candidate: str) -> NDArray[np.bool_]:
    return np.asarray(candidates == candidate, dtype=bool)


def _power_counts(
    labels: NDArray[np.int8],
    candidates: NDArray[np.str_],
    matrix_ids: NDArray[np.int64],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        selected = _candidate_mask(candidates, candidate)
        candidate_labels = labels[selected]
        candidate_matrices = matrix_ids[selected]
        event_matrices = {
            int(matrix_id)
            for matrix_id in np.unique(candidate_matrices)
            if bool(candidate_labels[candidate_matrices == matrix_id].any())
        }
        events = int(candidate_labels.sum())
        output[candidate] = {
            "events": events,
            "event_matrices": len(event_matrices),
            "minimum_events": MINIMUM_EVENTS,
            "minimum_event_matrices": MINIMUM_EVENT_MATRICES,
            "adequate": events >= MINIMUM_EVENTS
            and len(event_matrices) >= MINIMUM_EVENT_MATRICES,
        }
    return output


def _matrix_rate_interval(
    labels: NDArray[np.int8],
    matrix_ids: NDArray[np.int64],
    seed_parts: tuple[Any, ...],
) -> tuple[float, tuple[float, float], int, int]:
    unique = np.unique(matrix_ids)
    matrix_rates = np.asarray(
        [labels[matrix_ids == matrix_id].mean() for matrix_id in unique],
        dtype=np.float64,
    )
    rng = np.random.default_rng(
        derive_seed(BOOTSTRAP_MASTER_SEED, "regime.occurrence", *seed_parts)
    )
    indices = rng.integers(0, unique.size, size=(BOOTSTRAP_REPETITIONS, unique.size))
    samples = matrix_rates[indices].mean(axis=1)
    interval = np.quantile(samples, (0.025, 0.975))
    event_matrices = sum(
        bool(labels[matrix_ids == matrix_id].any()) for matrix_id in unique
    )
    return (
        float(matrix_rates.mean()),
        (float(interval[0]), float(interval[1])),
        int(labels.sum()),
        int(event_matrices),
    )


def compute_occurrence_metrics(
    labels: dict[str, NDArray[np.int8]],
    candidates: NDArray[np.str_],
    matrix_ids: NDArray[np.int64],
    cohort: str,
) -> dict[str, Any]:
    endpoint_results: dict[str, Any] = {}
    primary_cells: list[bool] = []
    for endpoint in ENDPOINTS:
        candidate_results: dict[str, Any] = {}
        for candidate in CANDIDATES:
            selected = _candidate_mask(candidates, candidate)
            candidate_labels = labels[endpoint][selected]
            candidate_matrix_ids = matrix_ids[selected]
            split = candidate_labels.shape[1] // 2
            halves: dict[str, Any] = {}
            for half, values in (
                ("A", candidate_labels[:, :split]),
                ("B", candidate_labels[:, split:]),
            ):
                rate, interval, events, event_matrices = _matrix_rate_interval(
                    values,
                    candidate_matrix_ids,
                    (cohort, endpoint, candidate, half),
                )
                passes = interval[0] > 0.0
                halves[half] = {
                    "rate": rate,
                    "ci95": interval,
                    "events": events,
                    "event_matrices": event_matrices,
                    "passes_positive_lower_bound": passes,
                }
                if endpoint == PRIMARY_ENDPOINT:
                    primary_cells.append(passes)
            candidate_results[candidate] = {
                "all_branches_rate": float(candidate_labels.mean()),
                "all_branches_events": int(candidate_labels.sum()),
                "halves": halves,
            }
        endpoint_results[endpoint] = candidate_results
    return {
        "cohort": cohort,
        "bootstrap_unit": "catalytic_matrix",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "endpoints": endpoint_results,
        "primary_occurrence_supported": bool(primary_cells) and all(primary_cells),
        "primary_decision_rule": (
            "whole-matrix bootstrap 95% lower event-rate bound above zero in "
            "both branch halves and both simulator candidates"
        ),
    }


def _fit_endpoint_registries(
    cases: list[StateCase],
    raw: MechanisticV2RawFeatures,
    labels: dict[str, NDArray[np.int8]],
) -> dict[str, dict[str, CandidateRegistryV2]]:
    candidates, matrix_ids = _metadata_arrays(cases)
    output: dict[str, dict[str, CandidateRegistryV2]] = {}
    for endpoint in ENDPOINTS:
        print(f"[models] fitting {endpoint}", flush=True)
        output[endpoint] = {}
        for candidate in CANDIDATES:
            selected = _candidate_mask(candidates, candidate)
            output[endpoint][candidate] = fit_candidate_registry_v2(
                candidate,
                raw.selected(selected),
                labels[endpoint][selected],
                matrix_ids[selected],
            )
    return output


def _predict_endpoint_registries(
    registries: dict[str, dict[str, CandidateRegistryV2]],
    cases: list[StateCase],
    raw: MechanisticV2RawFeatures,
) -> dict[str, dict[str, dict[str, NDArray[np.float64]]]]:
    candidates, _ = _metadata_arrays(cases)
    output: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {}
    for endpoint in ENDPOINTS:
        output[endpoint] = {}
        for candidate in CANDIDATES:
            selected = _candidate_mask(candidates, candidate)
            values = predict_candidate_registry_v2(
                registries[endpoint][candidate], raw.selected(selected)
            )
            expected = int(selected.sum())
            for model, prediction in values.items():
                if (
                    prediction.shape != (expected,)
                    or not np.isfinite(prediction).all()
                    or np.any((prediction < 0.0) | (prediction > 1.0))
                ):
                    raise ValueError(
                        f"invalid prediction {endpoint}/{candidate}/{model}"
                    )
            output[endpoint][candidate] = values
    return output


def _portable_prediction_audit(
    left: dict[str, dict[str, dict[str, NDArray[np.float64]]]],
    right: dict[str, dict[str, dict[str, NDArray[np.float64]]]],
) -> dict[str, Any]:
    errors: dict[str, Any] = {}
    maximum = 0.0
    for endpoint in ENDPOINTS:
        errors[endpoint] = {}
        for candidate in CANDIDATES:
            model_errors = {
                model: float(
                    np.max(
                        np.abs(
                            left[endpoint][candidate][model]
                            - right[endpoint][candidate][model]
                        )
                    )
                )
                for model in left[endpoint][candidate]
            }
            maximum = max(maximum, *model_errors.values())
            errors[endpoint][candidate] = model_errors
    if maximum > PORTABLE_PREDICTION_TOLERANCE:
        raise ValueError(f"portable model prediction mismatch: {maximum}")
    return {
        "maximum_absolute_error": maximum,
        "all_within_1e-12": maximum <= PORTABLE_PREDICTION_TOLERANCE,
        "errors": errors,
    }


def _save_registries(
    output: Path, registries: dict[str, dict[str, CandidateRegistryV2]]
) -> None:
    for endpoint in ENDPOINTS:
        save_registries_v2(
            output / f"models_{endpoint}.npz",
            output / f"model_contract_{endpoint}.json",
            registries[endpoint],
        )


def _load_registries(
    directory: Path,
) -> dict[str, dict[str, CandidateRegistryV2]]:
    return {
        endpoint: load_registries_v2(
            directory / f"models_{endpoint}.npz",
            directory / f"model_contract_{endpoint}.json",
        )
        for endpoint in ENDPOINTS
    }


def _model_summary(
    registries: dict[str, dict[str, CandidateRegistryV2]],
) -> dict[str, Any]:
    return {
        endpoint: {
            candidate: {
                "selected_lambdas": registry.selected_lambdas,
                "cv_scores": registry.cv_scores,
                "retained_features": {
                    block: transform.output_features
                    for block, transform in registry.transforms.items()
                },
                "uses_pca": False,
            }
            for candidate, registry in candidate_registries.items()
        }
        for endpoint, candidate_registries in registries.items()
    }


def _safe_reliability_bootstrap(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    matrix_ids: NDArray[np.int64],
    centered: bool,
    rng: np.random.Generator,
) -> tuple[float, tuple[float, float]]:
    """Return undefined reliability cleanly for constant rare-event arrays."""

    try:
        return _reliability_bootstrap(
            left,
            right,
            matrix_ids,
            centered,
            BOOTSTRAP_REPETITIONS,
            rng,
        )
    except (IndexError, ValueError):
        return float("nan"), (float("nan"), float("nan"))


def _safe_rank_metrics(
    prediction: NDArray[np.float64],
    q_a: NDArray[np.float64],
    q_b: NDArray[np.float64],
    matrix_ids: NDArray[np.int64],
) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return _rank_metrics(prediction, q_a, q_b, matrix_ids)


def compute_prediction_metrics(
    labels: dict[str, NDArray[np.int8]],
    predictions: dict[str, dict[str, dict[str, NDArray[np.float64]]]],
    candidates: NDArray[np.str_],
    matrix_ids: NDArray[np.int64],
    development_power: dict[str, dict[str, Any]],
    confirmation_power: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {"endpoints": {}, "primary_tests": []}
    primary_rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        endpoint_result: dict[str, Any] = {"candidates": {}, "state_gain_tests": []}
        for candidate in CANDIDATES:
            selected = _candidate_mask(candidates, candidate)
            candidate_labels = labels[endpoint][selected]
            selected_matrix_ids = matrix_ids[selected]
            split = candidate_labels.shape[1] // 2
            q_a = candidate_labels[:, :split].mean(axis=1)
            q_b = candidate_labels[:, split:].mean(axis=1)
            model_predictions = predictions[endpoint][candidate]
            reliability_rng = np.random.default_rng(
                derive_seed(
                    BOOTSTRAP_MASTER_SEED,
                    "regime.prediction.reliability",
                    endpoint,
                    candidate,
                )
            )
            reliability, reliability_ci = _safe_reliability_bootstrap(
                q_a,
                q_b,
                selected_matrix_ids,
                False,
                reliability_rng,
            )
            centered_reliability, centered_reliability_ci = _safe_reliability_bootstrap(
                q_a,
                q_b,
                selected_matrix_ids,
                True,
                reliability_rng,
            )
            endpoint_result["candidates"][candidate] = {
                "branch_half_reliability": reliability,
                "branch_half_reliability_ci95": reliability_ci,
                "centered_branch_half_reliability": centered_reliability,
                "centered_branch_half_reliability_ci95": centered_reliability_ci,
                "models": {
                    model: _safe_rank_metrics(prediction, q_a, q_b, selected_matrix_ids)
                    for model, prediction in model_predictions.items()
                },
            }
            for half, q in (("A", q_a), ("B", q_b)):
                seed_parts = (endpoint, candidate, half)
                gain, interval = _paired_gain(
                    q,
                    model_predictions["h10"],
                    model_predictions["h10_state"],
                    selected_matrix_ids,
                    _state_log_loss,
                    BOOTSTRAP_REPETITIONS,
                    np.random.default_rng(
                        derive_seed(
                            BOOTSTRAP_MASTER_SEED,
                            "regime.prediction.state_gain",
                            *seed_parts,
                        )
                    ),
                )
                brier_gain, brier_interval = _paired_gain(
                    q,
                    model_predictions["h10"],
                    model_predictions["h10_state"],
                    selected_matrix_ids,
                    _state_brier,
                    BOOTSTRAP_REPETITIONS,
                    np.random.default_rng(
                        derive_seed(
                            BOOTSTRAP_MASTER_SEED,
                            "regime.prediction.state_brier",
                            *seed_parts,
                        )
                    ),
                )
                row: dict[str, Any] = {
                    "endpoint": endpoint,
                    "candidate": candidate,
                    "half": half,
                    "baseline": "h10",
                    "enhanced": "h10_state",
                    "log_loss_gain": gain,
                    "log_loss_gain_ci95": interval,
                    "q_brier_gain": brier_gain,
                    "q_brier_gain_ci95": brier_interval,
                    "confirmatory": endpoint == PRIMARY_ENDPOINT,
                }
                if endpoint == PRIMARY_ENDPOINT:
                    row["randomization_p_raw"] = paired_matrix_randomization_p(
                        q,
                        model_predictions["h10"],
                        model_predictions["h10_state"],
                        selected_matrix_ids,
                        RANDOMIZATION_REPETITIONS,
                        np.random.default_rng(
                            derive_seed(
                                RANDOMIZATION_MASTER_SEED,
                                "regime.prediction.state_randomization",
                                *seed_parts,
                            )
                        ),
                    )
                    primary_rows.append(row)
                endpoint_result["state_gain_tests"].append(row)
        output["endpoints"][endpoint] = endpoint_result

    adjusted = holm_adjust([row["randomization_p_raw"] for row in primary_rows])
    for row, adjusted_p in zip(primary_rows, adjusted):
        row["randomization_p_holm"] = adjusted_p
        row["passes_statistical_gate"] = bool(
            row["log_loss_gain"] > 0.0
            and row["log_loss_gain_ci95"][0] > 0.0
            and adjusted_p < 0.05
        )
    power_adequate = all(
        development_power[candidate]["adequate"]
        and confirmation_power[candidate]["adequate"]
        for candidate in CANDIDATES
    )
    output["primary_tests"] = primary_rows
    output["family_size"] = len(primary_rows)
    output["development_power"] = development_power
    output["confirmation_power"] = confirmation_power
    output["prediction_power_adequate"] = power_adequate
    output["primary_prediction_supported"] = bool(
        power_adequate
        and primary_rows
        and all(row["passes_statistical_gate"] for row in primary_rows)
    )
    output["decision_rule"] = (
        "at least 100 primary events across at least 20 matrices per candidate "
        "in development and confirmation, plus positive h10_state-over-h10 "
        "log-loss gain, matrix-bootstrap lower 95% bound above zero, and "
        "Holm-adjusted paired whole-matrix randomization p<0.05 in both "
        "candidates and both confirmation branch halves"
    )
    return output


def _write_branch_table(
    path: Path, cases: list[StateCase], batches: list[RegimeBranchBatch]
) -> None:
    columns = (
        "state_id",
        "cohort",
        "candidate",
        "matrix_id",
        "landmark",
        "branch",
        "half",
        *ENDPOINTS,
        *(f"{name}_onset" for name in ENDPOINTS),
        "completed_horizon",
        "observed_fissions",
        "first_break_index",
        "first_run8_start",
        *GEOMETRY_COLUMNS,
    )
    with path.open("wb") as raw_file:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_file, mtime=0
        ) as compressed:
            with io.TextIOWrapper(
                compressed, encoding="utf-8", newline=""
            ) as text_file:
                writer = csv.writer(text_file, lineterminator="\n")
                writer.writerow(columns)
                for case, batch in zip(cases, batches):
                    split = batch.targets.shape[0] // 2
                    for branch in range(batch.targets.shape[0]):
                        geometry = [
                            "" if np.isnan(value) else f"{value:.17g}"
                            for value in batch.geometry[branch]
                        ]
                        writer.writerow(
                            (
                                case.state_id,
                                case.cohort,
                                case.candidate,
                                case.matrix_id,
                                case.landmark,
                                branch,
                                "A" if branch < split else "B",
                                *[int(value) for value in batch.targets[branch]],
                                *[int(value) for value in batch.onsets[branch]],
                                int(batch.completed_horizon[branch]),
                                int(batch.observed_fissions[branch]),
                                int(batch.first_break_index[branch]),
                                int(batch.first_run8_start[branch]),
                                *geometry,
                            )
                        )


def _state_table(
    cases: list[StateCase],
    labels: dict[str, NDArray[np.int8]],
    predictions: dict[str, dict[str, dict[str, NDArray[np.float64]]]] | None,
) -> pd.DataFrame:
    candidates, _ = _metadata_arrays(cases)
    offsets = {candidate: 0 for candidate in CANDIDATES}
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        local = offsets[case.candidate]
        offsets[case.candidate] += 1
        row: dict[str, Any] = {
            "state_id": case.state_id,
            "cohort": case.cohort,
            "candidate": case.candidate,
            "matrix_id": case.matrix_id,
            "landmark": case.landmark,
            "mass": int(case.snapshot.composition.sum()),
            "previous_growth_steps": case.snapshot.previous_growth_steps,
            "cumulative_growth_steps": case.snapshot.cumulative_growth_steps,
        }
        for endpoint in ENDPOINTS:
            values = labels[endpoint][index]
            split = values.size // 2
            row[f"q_{endpoint}_all"] = float(values.mean())
            row[f"q_{endpoint}_A"] = float(values[:split].mean())
            row[f"q_{endpoint}_B"] = float(values[split:].mean())
            if predictions is not None:
                for model, model_values in predictions[endpoint][
                    candidates[index]
                ].items():
                    row[f"prediction_{endpoint}_{model}"] = float(model_values[local])
        rows.append(row)
    return pd.DataFrame(rows)


def _save_analysis_arrays(
    path: Path,
    cases: list[StateCase],
    raw: MechanisticV2RawFeatures,
    batches: list[RegimeBranchBatch],
) -> None:
    labels = _all_labels(batches)
    np.savez_compressed(
        path,
        state_ids=np.asarray([case.state_id for case in cases]),
        candidates=np.asarray([case.candidate for case in cases]),
        matrix_ids=np.asarray([case.matrix_id for case in cases], dtype=np.int64),
        landmarks=np.asarray([case.landmark for case in cases], dtype=np.int64),
        compositions=np.vstack([case.snapshot.composition for case in cases]),
        h10=raw.h10,
        state_block=raw.state,
        beta_block=raw.beta,
        interaction_block=raw.interaction,
        **{f"labels_{name}": values for name, values in labels.items()},
        onsets=np.stack([batch.onsets for batch in batches]),
        completed_horizon=np.stack([batch.completed_horizon for batch in batches]),
        observed_fissions=np.stack([batch.observed_fissions for batch in batches]),
        first_break_index=np.stack([batch.first_break_index for batch in batches]),
        first_run8_start=np.stack([batch.first_run8_start for batch in batches]),
        first_run8_geometry=np.stack([batch.geometry for batch in batches]),
    )


def _labels_from_branch_table(
    path: Path, state_ids: Sequence[str]
) -> dict[str, NDArray[np.int8]]:
    table = pd.read_csv(path, dtype={"candidate": str})
    expected_rows = len(state_ids) * BRANCHES
    if len(table) != expected_rows:
        raise ValueError(
            f"branch readback has {len(table)} rows, expected {expected_rows}"
        )
    state_lookup = {state_id: index for index, state_id in enumerate(state_ids)}
    state_index = table["state_id"].map(state_lookup)
    if state_index.isna().any():
        raise ValueError("branch readback contains an unknown state")
    indices = state_index.to_numpy(dtype=np.int64)
    branches = table["branch"].to_numpy(dtype=np.int64)
    if np.any((branches < 0) | (branches >= BRANCHES)):
        raise ValueError("branch readback contains an invalid branch index")
    marker = np.zeros((len(state_ids), BRANCHES), dtype=np.int8)
    np.add.at(marker, (indices, branches), 1)
    if not np.all(marker == 1):
        raise ValueError("branch readback contains duplicate or missing rows")
    output: dict[str, NDArray[np.int8]] = {}
    for endpoint in ENDPOINTS:
        values = np.full((len(state_ids), BRANCHES), -1, dtype=np.int8)
        values[indices, branches] = table[endpoint].to_numpy(dtype=np.int8)
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f"invalid readback labels for {endpoint}")
        output[endpoint] = values
    return output


def _predictions_from_state_table(
    table: pd.DataFrame,
    model_names: dict[str, tuple[str, ...]],
) -> dict[str, dict[str, dict[str, NDArray[np.float64]]]]:
    output: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {}
    for endpoint in ENDPOINTS:
        output[endpoint] = {}
        for candidate in CANDIDATES:
            selected = table["candidate"] == candidate
            output[endpoint][candidate] = {
                model: table.loc[selected, f"prediction_{endpoint}_{model}"].to_numpy(
                    dtype=np.float64
                )
                for model in model_names[endpoint]
            }
    return output


def _compare_structures(left: Any, right: Any) -> tuple[int, float]:
    checked = 0
    maximum = 0.0

    def compare(a: Any, b: Any, location: str) -> None:
        nonlocal checked, maximum
        if isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                raise ValueError(f"metric structure differs at {location}")
            for key in sorted(a):
                compare(a[key], b[key], f"{location}.{key}")
            return
        if isinstance(a, (list, tuple)):
            if not isinstance(b, (list, tuple)) or len(a) != len(b):
                raise ValueError(f"metric sequence differs at {location}")
            for index, (left_item, right_item) in enumerate(zip(a, b)):
                compare(left_item, right_item, f"{location}[{index}]")
            return
        if isinstance(a, (bool, str)) or a is None:
            if a != b:
                raise ValueError(f"metric value differs at {location}: {a} != {b}")
            return
        if isinstance(a, (int, float, np.integer, np.floating)):
            left_value = float(a)
            right_value = float(b)
            if np.isnan(left_value) and np.isnan(right_value):
                return
            difference = abs(left_value - right_value)
            checked += 1
            maximum = max(maximum, difference)
            if difference > CONTINUOUS_TOLERANCE:
                raise ValueError(f"metric value differs at {location}: {difference}")
            return
        if a != b:
            raise ValueError(f"metric value differs at {location}")

    compare(left, right, "metrics")
    return checked, maximum


def _protocol() -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "format": DESIGN_FORMAT,
        "status": "sealed_before_development_matrix_generation",
        "scope": "prospective coherent-distinct-persistent regime occurrence and prediction",
        "endpoints": {
            PRIMARY_ENDPOINT: {
                "role": "primary_confirmatory",
                "selection": "earliest qualifying eight-fission window after the first inheritance break",
                "inheritance": "eight consecutive fissions with strict H>0.90",
                "coherence": "all 28 daughter pairs have strict H>0.90",
                "distinctness": "all eight daughters have inclusive H<=0.85 to the pre-break parent",
            },
            SECONDARY_FIRST5: {
                "role": "prespecified_secondary_no_rescue",
                "selection": "earliest qualifying eight-fission window after the first inheritance break",
                "inheritance": "eight consecutive fissions with strict H>0.90",
                "coherence": "all 10 pairs among the first five daughters have strict H>0.90",
                "distinctness": "the first five daughters have inclusive H<=0.85 to the pre-break parent",
            },
            SECONDARY_CENTROID: {
                "role": "prespecified_secondary_no_rescue",
                "selection": "earliest qualifying eight-fission window after the first inheritance break",
                "inheritance": "eight consecutive fissions with strict H>0.90",
                "coherence": "all eight daughters have strict H>0.90 to their arithmetic-mean composition",
                "distinctness": "all eight daughters have inclusive H<=0.85 to the pre-break parent",
            },
        },
        "boundary_rules": {
            "first_break": "first fission with H<=0.90",
            "search_begins": "fission immediately after the first break",
            "extinction_no_certification": "negative",
            "no_break": "negative",
            "horizon_without_certification": "negative",
            "coherence_equality_0.90": "fails",
            "distinctness_equality_0.85": "passes",
        },
        "cohorts": {
            "development": _experiment(DEVELOPMENT_MASTER_SEED).to_dict(),
            "confirmation": _experiment(CONFIRMATION_MASTER_SEED).to_dict(),
            "matrices_are_disjoint_by_master_seed": True,
        },
        "feature_model": {
            "baseline": "unpenalized unique h10 history/mass/phase/all-clock block",
            "primary_added_block": "provenance-selected state-only composition block",
            "additional_descriptive_blocks": [
                "complete beta-only",
                "beta-conditioned state interaction",
            ],
            "uses_pca": False,
            "ridge_lambda_grid": RIDGE_LAMBDAS,
            "cv_folds": CV_FOLDS,
            "cv_split": "development matrix_id modulo 5",
            "cv_tie_break": "largest lambda within 1e-12 of minimum loss",
            "model_sequence": ["h10", "state", "beta", "interaction"],
        },
        "inference": {
            "occurrence_primary_gate": (
                "whole-matrix bootstrap 95% lower event-rate bound above zero "
                "in both candidates and both branch halves"
            ),
            "prediction_primary_contrast": ["h10", "h10_state"],
            "prediction_gate": (
                "positive gain, whole-matrix bootstrap 95% lower bound above "
                "zero, and Holm-adjusted paired matrix randomization p<0.05 "
                "in both candidates and both halves"
            ),
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "prediction_family_size": 4,
            "power_gate": {
                "minimum_events_per_candidate_per_cohort": MINIMUM_EVENTS,
                "minimum_event_matrices_per_candidate_per_cohort": MINIMUM_EVENT_MATRICES,
                "failure_language": "prediction underpowered; occurrence unaffected; no adaptive sampling",
            },
            "secondary_endpoints_may_rescue_primary": False,
        },
        "claim_boundary": {
            "occurrence_pass": "distinct coherent persistent new hereditary regime",
            "prediction_pass": "its probability is compositionally predictable beyond h10",
            "not_tested": [
                "compositional recurrence",
                "attractor switching",
                "perturbation recovery",
                "causality",
                "biological memory",
                "prebiotic realism",
            ],
        },
    }
    protocol["protocol_id"] = _canonical_digest(protocol)
    return protocol


def register_design(output_directory: Path) -> None:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    seeds = {
        "scaled5": ExperimentConfig.scaled5().master_seed,
        "MECHCONF": MECHCONF_MASTER_SEED,
        "MEMCONF": MEMORY_CONFIRM_MASTER_SEED,
        "MECHCONF2": MECHCONF2_MASTER_SEED,
        "REGDEV": DEVELOPMENT_MASTER_SEED,
        "REGCONF": CONFIRMATION_MASTER_SEED,
        "bootstrap": BOOTSTRAP_MASTER_SEED,
        "randomization": RANDOMIZATION_MASTER_SEED,
    }
    if len(set(seeds.values())) != len(seeds):
        raise ValueError(
            "prospective regime seed domain collides with another campaign"
        )
    with _atomic_destination(output_directory) as output:
        protocol = _protocol()
        (output / "protocol.json").write_text(
            json.dumps(_json_ready(protocol), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        provenance = provenance_contract()
        (output / "feature_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload: dict[str, Any] = {
            "format": DESIGN_FORMAT,
            "status": "sealed_before_development_matrix_generation",
            "protocol_id": protocol["protocol_id"],
            "protocol_digest": sha256_file(output / "protocol.json"),
            "feature_provenance_digest": sha256_file(
                output / "feature_provenance.json"
            ),
            "source_hashes": _source_hashes(),
            "seed_domains": seeds,
            "all_seed_domains_unique": True,
            "development_output_must_not_exist": True,
            "confirmation_requires_sealed_development_models": True,
        }
        payload["registration_id"] = _canonical_digest(payload)
        (output / "registration.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(output)
    print(f"Prospective regime design sealed at {output_directory}", flush=True)


def verify_design(registration: Path) -> dict[str, Any]:
    registration = registration.resolve()
    verify_checksums(registration)
    payload = json.loads(
        (registration / "registration.json").read_text(encoding="utf-8")
    )
    if payload.get("format") != DESIGN_FORMAT:
        raise ValueError("unsupported regime design registration")
    registration_id = payload.pop("registration_id")
    if _canonical_digest(payload) != registration_id:
        raise ValueError("regime design registration identifier mismatch")
    payload["registration_id"] = registration_id
    current_sources = _source_hashes()
    if payload["source_hashes"] != current_sources:
        changed = [
            name
            for name, digest in payload["source_hashes"].items()
            if current_sources.get(name) != digest
        ]
        raise ValueError(f"registered regime source changed: {changed}")
    if payload["protocol_digest"] != sha256_file(registration / "protocol.json"):
        raise ValueError("regime protocol digest mismatch")
    protocol = json.loads((registration / "protocol.json").read_text())
    if protocol != json.loads(json.dumps(_protocol())):
        raise ValueError("regime protocol implementation diverged from registration")
    if payload["feature_provenance_digest"] != sha256_file(
        registration / "feature_provenance.json"
    ):
        raise ValueError("regime feature provenance digest mismatch")
    if not payload["all_seed_domains_unique"]:
        raise ValueError("regime seed domains are not unique")
    return payload


def _design_reference(registration: Path, payload: dict[str, Any]) -> dict[str, str]:
    return {
        "path": str(registration.resolve()),
        "registration_id": payload["registration_id"],
        "sha256sums_digest": sha256_file(registration.resolve() / "SHA256SUMS"),
    }


def _verify_design_reference(reference: dict[str, str]) -> dict[str, Any]:
    registration = Path(reference["path"])
    payload = verify_design(registration)
    if payload["registration_id"] != reference["registration_id"]:
        raise ValueError("design registration ID changed")
    if sha256_file(registration / "SHA256SUMS") != reference["sha256sums_digest"]:
        raise ValueError("design checksum seal changed")
    return payload


def _development_report(
    output: Path,
    power: dict[str, Any],
    model_summary: dict[str, Any],
    replay: dict[str, Any],
) -> None:
    lines = [
        "# Prospective regime development",
        "",
        "The endpoint design was sealed before these matrices were generated. This cohort selected model penalties and cannot confirm either scientific claim.",
        "",
        "## Primary endpoint development prevalence",
        "",
        "| Candidate | Events | Event matrices | Prediction count gate |",
        "|---|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        item = power[candidate]
        lines.append(
            f"| {candidate} | {item['events']} | {item['event_matrices']} | {item['adequate']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen models",
            "",
            "All three endpoints use the provenance-complete no-PCA model suite. The primary comparison is the unpenalized h10 baseline versus h10 plus penalized state-only composition.",
            "",
        ]
    )
    for endpoint in ENDPOINTS:
        lines.extend(
            [
                f"### {endpoint}",
                "",
                "| Candidate | State lambda | Beta lambda | Interaction lambda |",
                "|---|---:|---:|---:|",
            ]
        )
        for candidate in CANDIDATES:
            selected = model_summary[endpoint][candidate]["selected_lambdas"]
            lines.append(
                f"| {candidate} | {selected['state']:g} | {selected['beta']:g} | {selected['interaction']:g} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Replay",
            "",
            f"All discrete values exact: **{replay['discrete_exact']}**. Maximum continuous error: `{replay['maximum_continuous_absolute_error']:.3g}`.",
            "",
            "Confirmation matrices were not generated during this stage.",
            "",
        ]
    )
    (output / "DEVELOPMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_development(registration: Path, output_directory: Path, workers: int) -> None:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    registration = registration.resolve()
    design = verify_design(registration)
    experiment = _experiment(DEVELOPMENT_MASTER_SEED)
    print("[develop 1/8] Generating prospective development trajectories", flush=True)
    with threadpool_limits(limits=1):
        cases = build_cohort(experiment, "REGDEV", experiment.development)
        raw = extract_mechanistic_v2_features(cases, experiment)
    print("[develop 2/8] Shooting F32 development futures", flush=True)
    batches = run_regime_branches(
        cases, experiment, BRANCHES, workers, "develop-generate"
    )
    print("[develop 3/8] Replaying every development future", flush=True)
    replay_batches = run_regime_branches(
        cases, experiment, BRANCHES, workers, "develop-replay"
    )
    replay = _replay_audit(batches, replay_batches)
    labels = _all_labels(batches)
    candidates, matrix_ids = _metadata_arrays(cases)
    power = _power_counts(labels[PRIMARY_ENDPOINT], candidates, matrix_ids)
    occurrence = compute_occurrence_metrics(labels, candidates, matrix_ids, "REGDEV")
    print("[develop 4/8] Fitting development-only model suite", flush=True)
    registries = _fit_endpoint_registries(cases, raw, labels)
    predictions = _predict_endpoint_registries(registries, cases, raw)
    model_summary = _model_summary(registries)

    with _atomic_destination(output_directory) as output:
        print("[develop 5/8] Writing development artifacts", flush=True)
        _save_registries(output, registries)
        reloaded = _load_registries(output)
        reloaded_predictions = _predict_endpoint_registries(reloaded, cases, raw)
        portable_audit = _portable_prediction_audit(predictions, reloaded_predictions)
        _write_branch_table(output / "development_branches.csv.gz", cases, batches)
        _state_table(cases, labels, predictions).to_csv(
            output / "development_states.csv", index=False
        )
        _save_analysis_arrays(output / "development_arrays.npz", cases, raw, batches)
        (output / "occurrence_metrics.json").write_text(
            json.dumps(_json_ready(occurrence), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "replay_audit.json").write_text(
            json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "portable_prediction_audit.json").write_text(
            json.dumps(_json_ready(portable_audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _development_report(output, power, model_summary, replay)
        print("[develop 6/8] Sealing portable model contract", flush=True)
        seal: dict[str, Any] = {
            "format": DEVELOPMENT_FORMAT,
            "status": "models_sealed_before_confirmation_matrix_generation",
            "design": _design_reference(registration, design),
            "source_hashes": _source_hashes(),
            "experiment": experiment.to_dict(),
            "states": len(cases),
            "futures": len(cases) * BRANCHES,
            "endpoint_order": ENDPOINTS,
            "development_power": power,
            "occurrence_metrics_digest": sha256_file(
                output / "occurrence_metrics.json"
            ),
            "replay_audit": replay,
            "portable_prediction_audit": portable_audit,
            "model_summary": model_summary,
            "feature_provenance_digest": design["feature_provenance_digest"],
            "runtime": _runtime_manifest(),
        }
        seal["model_seal_id"] = _canonical_digest(seal)
        (output / "model_seal.json").write_text(
            json.dumps(_json_ready(seal), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format": DEVELOPMENT_FORMAT,
            "model_seal_id": seal["model_seal_id"],
            "design_registration_id": design["registration_id"],
            "development_replay_exact": replay["discrete_exact"]
            and replay["continuous_within_1e-14"],
            "confirmation_generated": False,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("[develop 7/8] Writing immutable checksums", flush=True)
        write_checksums(output)
    print("[develop 8/8] Development and models sealed", flush=True)
    print(f"Development bundle written to {output_directory}", flush=True)


def verify_development(development: Path) -> dict[str, Any]:
    development = development.resolve()
    verify_checksums(development)
    seal = json.loads((development / "model_seal.json").read_text())
    if seal.get("format") != DEVELOPMENT_FORMAT:
        raise ValueError("unsupported regime development bundle")
    model_seal_id = seal.pop("model_seal_id")
    if _canonical_digest(seal) != model_seal_id:
        raise ValueError("regime model seal identifier mismatch")
    seal["model_seal_id"] = model_seal_id
    _verify_design_reference(seal["design"])
    if seal["source_hashes"] != _source_hashes():
        changed = [
            name
            for name, digest in seal["source_hashes"].items()
            if _source_hashes().get(name) != digest
        ]
        raise ValueError(f"sealed regime model source changed: {changed}")
    expected = json.loads(json.dumps(_experiment(DEVELOPMENT_MASTER_SEED).to_dict()))
    if seal["experiment"] != expected:
        raise ValueError("development experiment diverged from model seal")
    if seal["occurrence_metrics_digest"] != sha256_file(
        development / "occurrence_metrics.json"
    ):
        raise ValueError("development occurrence metrics changed")
    _load_registries(development)
    return seal


def _readback_metrics(
    branch_path: Path,
    state_path: Path,
    model_names: dict[str, tuple[str, ...]],
    development_power: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    state_table = pd.read_csv(state_path, dtype={"candidate": str})
    state_table["candidate"] = state_table["candidate"].str.zfill(2)
    state_ids = state_table["state_id"].tolist()
    labels = _labels_from_branch_table(branch_path, state_ids)
    candidates = state_table["candidate"].to_numpy(dtype=str)
    matrix_ids = state_table["matrix_id"].to_numpy(dtype=np.int64)
    predictions = _predictions_from_state_table(state_table, model_names)
    occurrence = compute_occurrence_metrics(labels, candidates, matrix_ids, "REGCONF")
    confirmation_power = _power_counts(labels[PRIMARY_ENDPOINT], candidates, matrix_ids)
    prediction = compute_prediction_metrics(
        labels,
        predictions,
        candidates,
        matrix_ids,
        development_power,
        confirmation_power,
    )
    return occurrence, prediction, confirmation_power


def _confirmation_report(
    output: Path,
    occurrence: dict[str, Any],
    prediction: dict[str, Any],
    replay: dict[str, Any],
    design_id: str,
    model_id: str,
) -> None:
    occurrence_pass = occurrence["primary_occurrence_supported"]
    prediction_pass = prediction["primary_prediction_supported"]
    power_pass = prediction["prediction_power_adequate"]
    lines = [
        "# Prospective coherent-regime confirmation",
        "",
        "## Outcome",
        "",
        (
            "The primary occurrence gate passed: the simulator prospectively produced distinct, mutually coherent, persistent eight-fission hereditary regimes in both candidates and both branch halves."
            if occurrence_pass
            else "The primary occurrence gate did not pass. The secondary definitions cannot rescue the claim of a distinct, mutually coherent, persistent eight-fission regime."
        ),
        (
            "Current composition added confirmed predictive information beyond the unique all-clock h10 baseline."
            if prediction_pass
            else (
                "The predictive test was underpowered under the frozen minimum-count rule; this does not alter the occurrence result."
                if not power_pass
                else "The frozen state-added predictor did not pass all four prospective gates."
            )
        ),
        "",
        "## Endpoint occurrence",
        "",
        "| Endpoint | Candidate | Half | Rate | 95% matrix-bootstrap CI | Events | Event matrices |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for endpoint in ENDPOINTS:
        for candidate in CANDIDATES:
            for half in ("A", "B"):
                item = occurrence["endpoints"][endpoint][candidate]["halves"][half]
                interval = item["ci95"]
                lines.append(
                    f"| {endpoint} | {candidate} | {half} | {item['rate']:.6f} | "
                    f"[{interval[0]:.6f}, {interval[1]:.6f}] | {item['events']} | "
                    f"{item['event_matrices']} |"
                )
    lines.extend(
        [
            "",
            "## Primary state-added prediction tests",
            "",
            "| Candidate | Half | Log-loss gain | 95% CI | Holm p | Statistical pass |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in prediction["primary_tests"]:
        interval = row["log_loss_gain_ci95"]
        lines.append(
            f"| {row['candidate']} | {row['half']} | {row['log_loss_gain']:.6f} | "
            f"[{interval[0]:.6f}, {interval[1]:.6f}] | "
            f"{row['randomization_p_holm']:.6f} | {row['passes_statistical_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Prespecified secondary state-added contrasts",
            "",
            "| Endpoint | Candidate | Half | Log-loss gain | 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for endpoint in (SECONDARY_FIRST5, SECONDARY_CENTROID):
        for row in prediction["endpoints"][endpoint]["state_gain_tests"]:
            interval = row["log_loss_gain_ci95"]
            lines.append(
                f"| {endpoint} | {row['candidate']} | {row['half']} | "
                f"{row['log_loss_gain']:.6f} | "
                f"[{interval[0]:.6f}, {interval[1]:.6f}] |"
            )
    lines.extend(
        [
            "",
            "## Descriptive frozen-model ranks",
            "",
            "| Endpoint | Candidate | Model | Overall Spearman mean | Within-matrix Spearman mean |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for endpoint in ENDPOINTS:
        for candidate in CANDIDATES:
            models = prediction["endpoints"][endpoint]["candidates"][candidate][
                "models"
            ]
            for model, values in models.items():
                lines.append(
                    f"| {endpoint} | {candidate} | {model} | "
                    f"{values['overall_spearman_mean']:.6f} | "
                    f"{values['centered_spearman_mean']:.6f} |"
                )
    lines.extend(
        [
            "",
            "## Audit and boundary",
            "",
            f"Design registration: `{design_id}`. Model seal: `{model_id}`.",
            f"All discrete futures replayed exactly: **{replay['discrete_exact']}**. Maximum continuous replay error: `{replay['maximum_continuous_absolute_error']:.3g}`.",
            "",
            "The first-five and centroid endpoints were prespecified secondary analyses and cannot replace the all-eight pairwise primary endpoint. This campaign does not test recurrence, attractor switching, perturbation recovery, causality, biological memory, or prebiotic realism.",
            "",
        ]
    )
    (output / "REGIME_CONFIRMATION_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    lay_lines = [
        "# Lay summary: prospective hereditary-regime test",
        "",
        (
            "The strict test succeeded: new molecular compositions formed groups that stayed mutually similar for eight divisions while remaining clearly different from the old composition."
            if occurrence_pass
            else "The strict test did not establish a genuinely new hereditary regime. Some looser patterns may occur, but they cannot replace the failed strict definition."
        ),
        (
            "The starting composition also helped predict which states would enter such a regime, beyond their mass, phase, clocks, and inheritance history."
            if prediction_pass
            else (
                "There were too few strict events for the predeclared prediction test to be considered adequately powered."
                if not power_pass
                else "The current composition did not add consistently confirmed predictive information beyond mass, phase, clocks, and inheritance history."
            )
        ),
        "",
        "This is evidence only about the reconstructed simulator. It is not evidence that real prebiotic chemistry behaves this way.",
        "",
    ]
    (output / "LAY_SUMMARY.md").write_text("\n".join(lay_lines), encoding="utf-8")


def run_confirmation(
    registration: Path,
    development: Path,
    output_directory: Path,
    workers: int,
) -> None:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    registration = registration.resolve()
    development = development.resolve()
    design = verify_design(registration)
    model_seal = verify_development(development)
    if model_seal["design"]["registration_id"] != design["registration_id"]:
        raise ValueError("development models reference a different design")
    registries = _load_registries(development)
    experiment = _experiment(CONFIRMATION_MASTER_SEED)
    print("[confirm 1/9] Generating untouched confirmation trajectories", flush=True)
    with threadpool_limits(limits=1):
        cases = build_cohort(experiment, "REGCONF", experiment.confirmation)
        raw = extract_mechanistic_v2_features(cases, experiment)
    print("[confirm 2/9] Scoring frozen models before future shooting", flush=True)
    predictions = _predict_endpoint_registries(registries, cases, raw)
    print("[confirm 3/9] Shooting untouched F32 confirmation futures", flush=True)
    batches = run_regime_branches(
        cases, experiment, BRANCHES, workers, "confirm-generate"
    )
    print("[confirm 4/9] Replaying every confirmation future", flush=True)
    replay_batches = run_regime_branches(
        cases, experiment, BRANCHES, workers, "confirm-replay"
    )
    replay = _replay_audit(batches, replay_batches)
    labels = _all_labels(batches)
    candidates, matrix_ids = _metadata_arrays(cases)
    development_power = model_seal["development_power"]
    confirmation_power = _power_counts(labels[PRIMARY_ENDPOINT], candidates, matrix_ids)
    print("[confirm 5/9] Computing frozen occurrence and prediction tests", flush=True)
    occurrence = compute_occurrence_metrics(labels, candidates, matrix_ids, "REGCONF")
    prediction = compute_prediction_metrics(
        labels,
        predictions,
        candidates,
        matrix_ids,
        development_power,
        confirmation_power,
    )

    with _atomic_destination(output_directory) as output:
        print("[confirm 6/9] Writing complete confirmation artifacts", flush=True)
        branch_path = output / "confirmation_branches.csv.gz"
        state_path = output / "confirmation_states.csv"
        _write_branch_table(branch_path, cases, batches)
        _state_table(cases, labels, predictions).to_csv(state_path, index=False)
        _save_analysis_arrays(output / "confirmation_arrays.npz", cases, raw, batches)
        model_names = {
            endpoint: tuple(predictions[endpoint]["02"].keys())
            for endpoint in ENDPOINTS
        }
        read_occurrence, read_prediction, read_power = _readback_metrics(
            branch_path, state_path, model_names, development_power
        )
        occurrence_checked, occurrence_error = _compare_structures(
            occurrence, read_occurrence
        )
        prediction_checked, prediction_error = _compare_structures(
            prediction, read_prediction
        )
        if read_power != confirmation_power:
            raise ValueError("confirmation power readback mismatch")
        recomputation = {
            "point_estimates_checked": occurrence_checked + prediction_checked,
            "maximum_absolute_error": max(occurrence_error, prediction_error),
            "all_within_1e-14": max(occurrence_error, prediction_error)
            <= CONTINUOUS_TOLERANCE,
            "branch_artifact": branch_path.name,
            "state_artifact": state_path.name,
        }
        (output / "occurrence_metrics.json").write_text(
            json.dumps(_json_ready(occurrence), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "prediction_metrics.json").write_text(
            json.dumps(_json_ready(prediction), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "replay_audit.json").write_text(
            json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "metric_recomputation_audit.json").write_text(
            json.dumps(recomputation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _confirmation_report(
            output,
            occurrence,
            prediction,
            replay,
            design["registration_id"],
            model_seal["model_seal_id"],
        )
        manifest = {
            "format": CONFIRMATION_FORMAT,
            "status": "prospective_confirmation_complete",
            "design_registration_id": design["registration_id"],
            "model_seal_id": model_seal["model_seal_id"],
            "design_checksum_digest": sha256_file(registration / "SHA256SUMS"),
            "development_checksum_digest": sha256_file(development / "SHA256SUMS"),
            "source_hashes": _source_hashes(),
            "experiment": experiment.to_dict(),
            "states": len(cases),
            "futures": len(cases) * BRANCHES,
            "primary_occurrence_supported": occurrence["primary_occurrence_supported"],
            "prediction_power_adequate": prediction["prediction_power_adequate"],
            "primary_prediction_supported": prediction["primary_prediction_supported"],
            "confirmation_power": confirmation_power,
            "confirmation_replay_exact": replay["discrete_exact"]
            and replay["continuous_within_1e-14"],
            "metric_recomputation_exact": recomputation["all_within_1e-14"],
            "runtime": _runtime_manifest(),
            "claim_boundary": (
                "secondary endpoints cannot rescue the primary; recurrence, "
                "attractor switching, perturbation recovery, causality, "
                "biological memory, and chemistry are not tested"
            ),
        }
        (output / "manifest.json").write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("[confirm 7/9] Writing immutable checksums", flush=True)
        write_checksums(output)
    print("[confirm 8/9] Confirmation sealed", flush=True)
    print(f"[confirm 9/9] Results written to {output_directory}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prospective coherent-regime registration and confirmation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser(
        "register-design", help="seal the protocol before development generation"
    )
    register.add_argument(
        "--output", type=Path, default=Path("results/regime_design_registration")
    )
    develop = commands.add_parser(
        "develop", help="generate development data and seal portable models"
    )
    develop.add_argument(
        "--registration",
        type=Path,
        default=Path("results/regime_design_registration"),
    )
    develop.add_argument(
        "--output", type=Path, default=Path("results/regime_development")
    )
    develop.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16))
    confirm = commands.add_parser(
        "confirm", help="run the untouched prospective confirmation"
    )
    confirm.add_argument(
        "--registration",
        type=Path,
        default=Path("results/regime_design_registration"),
    )
    confirm.add_argument(
        "--development",
        type=Path,
        default=Path("results/regime_development"),
    )
    confirm.add_argument(
        "--output", type=Path, default=Path("results/regime_confirmation")
    )
    confirm.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16))
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "register-design":
        register_design(arguments.output)
    elif arguments.command == "develop":
        if arguments.workers < 1:
            raise ValueError("workers must be positive")
        run_development(arguments.registration, arguments.output, arguments.workers)
    elif arguments.command == "confirm":
        if arguments.workers < 1:
            raise ValueError("workers must be positive")
        run_confirmation(
            arguments.registration,
            arguments.development,
            arguments.output,
            arguments.workers,
        )
    else:  # pragma: no cover
        raise AssertionError(f"unknown command {arguments.command}")


if __name__ == "__main__":
    main()
