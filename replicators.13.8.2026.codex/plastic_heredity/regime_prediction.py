"""Prospective strict-regime prediction program.

The workflow is isolated from ``regime_confirmation`` and its sealed result.
It supports a post-hoc diagnostic, immutable registration, an 80-matrix pilot
with model selection, and a single 200-matrix untouched confirmation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import logit
from threadpoolctl import threadpool_limits

from .config import CANDIDATES, CohortConfig, ExperimentConfig
from .experiment import _json_ready, _runtime_manifest
from .episode_coherence import BOOTSTRAP_MASTER_SEED as EPISODE_BOOTSTRAP_SEED
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
    _state_brier,
    _state_log_loss,
    holm_adjust,
    paired_matrix_randomization_p,
)
from .mechanistic_v2_models import fit_linear
from .metrics import centered_spearman, spearman
from .mechanistic_v2 import MECHCONF2_MASTER_SEED
from .memory import MEMORY_CONFIRM_MASTER_SEED
from .regime_confirmation import (
    BOOTSTRAP_MASTER_SEED as OLD_REGIME_BOOTSTRAP_SEED,
    BRANCHES as SEALED_BRANCHES,
    COHERENCE_THRESHOLD,
    CONFIRMATION_MASTER_SEED as OLD_REGIME_CONFIRMATION_SEED,
    DEVELOPMENT_MASTER_SEED as OLD_REGIME_DEVELOPMENT_SEED,
    DISTINCTNESS_THRESHOLD,
    ENDPOINTS,
    INHERITANCE_THRESHOLD,
    PRIMARY_ENDPOINT,
    RANDOMIZATION_MASTER_SEED as OLD_REGIME_RANDOMIZATION_SEED,
    RUN_LENGTH,
)
from .regime_prediction_endpoints import WINDOW_METRIC_NAMES, evaluate_rich_regime
from .regime_prediction_features import (
    POST_BREAK_FEATURE_NAMES,
    PredictionRawFeatures,
    compact_post_break_features,
    extract_prediction_features,
    prediction_provenance_contract,
)
from .regime_prediction_models import (
    BOOTSTRAP_SELECTION_FRACTION,
    BOOTSTRAP_SELECTION_REPETITIONS,
    MODEL_FAMILIES,
    MODEL_SIMPLICITY,
    NONLINEAR_GRID,
    RIDGE_LAMBDAS,
    PredictionFamilyModel,
    SequentialRidgeModel,
    crossfit_prediction_family,
    fit_sequential_ridge,
    load_prediction_models,
    model_summary,
    save_prediction_models,
)
from .seeds import derive_seed
from .simulator import (
    FloatMatrix,
    SimulationError,
    Snapshot,
    generate_beta,
    generate_initial_composition,
    simulate_future_absorbing,
    simulate_lineage,
)

FloatArray = NDArray[np.float64]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DESIGN_FORMAT = "plastic-heredity-regime-prediction-design-v1"
PILOT_FORMAT = "plastic-heredity-regime-prediction-pilot-v1"
CONFIRMATION_FORMAT = "plastic-heredity-regime-prediction-confirmation-v1"
DIAGNOSTIC_FORMAT = "plastic-heredity-regime-prediction-diagnostic-v1"
CHECKPOINT_FORMAT = "plastic-heredity-regime-prediction-checkpoint-v1"

PILOT_MATRICES = 80
CONFIRMATION_MATRICES = 200
BRANCHES = 128
HORIZON = 32
LANDMARKS = (20, 35, 50, 65, 80)
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
MINIMUM_EVENTS = 100
MINIMUM_EVENT_MATRICES = 20
CONTINUOUS_TOLERANCE = 1e-14

PILOT_MASTER_SEED = "031aac456d56d90737a0b44497bc431bac4856812c40a2318a6de328d27a08dc"
CONFIRMATION_MASTER_SEED = (
    "669690801dc9f385e8f45290626b11373ab856c45eeb4d987d717444a107fc12"
)
BOOTSTRAP_MASTER_SEED = (
    "199c003c86f05195767e90a98781acb869d8cc3317f160b72b6a2af36d568d4f"
)
RANDOMIZATION_MASTER_SEED = (
    "dc4e3d2e66984d997cf76431273c6be5b2da6c297befd2584cdc2dbe5cf1240c"
)
DIAGNOSTIC_MASTER_SEED = (
    "fcf6a91470b4464d2a903c38abfa0a2787a7d113be53dcac921b47d9cc56e7a3"
)

SOURCE_FILES = (
    "plastic_heredity/config.py",
    "plastic_heredity/episode_coherence.py",
    "plastic_heredity/experiment.py",
    "plastic_heredity/features.py",
    "plastic_heredity/mechanistic.py",
    "plastic_heredity/mechanistic_features.py",
    "plastic_heredity/mechanistic_metrics.py",
    "plastic_heredity/mechanistic_v2_features.py",
    "plastic_heredity/mechanistic_v2_models.py",
    "plastic_heredity/mechanistic_v2.py",
    "plastic_heredity/memory.py",
    "plastic_heredity/metrics.py",
    "plastic_heredity/regime_confirmation.py",
    "plastic_heredity/regime_prediction.py",
    "plastic_heredity/regime_prediction_endpoints.py",
    "plastic_heredity/regime_prediction_features.py",
    "plastic_heredity/regime_prediction_models.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/simulator.py",
    "requirements-lock.txt",
)


@dataclass(frozen=True)
class PredictionCase:
    state_id: str
    cohort: str
    candidate: str
    matrix_id: int
    landmark: int
    beta: FloatMatrix
    snapshot: Snapshot
    previous_composition: NDArray[np.int64] | None


@dataclass
class PredictionBranchBatch:
    targets: NDArray[np.int8]
    stages: NDArray[np.int8]
    onsets: NDArray[np.int16]
    completed_horizon: NDArray[np.int8]
    observed_fissions: NDArray[np.int16]
    first_break_index: NDArray[np.int16]
    first_run8_start: NDArray[np.int16]
    longest_run: NDArray[np.int16]
    window_count: NDArray[np.int16]
    best_margins: NDArray[np.float64]
    post_break_features: NDArray[np.float64]
    windows: tuple[NDArray[np.float64], ...]


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def _seed_domains() -> dict[str, str]:
    return {
        "original_replication": ExperimentConfig().master_seed,
        "MECHCONF": MECHCONF_MASTER_SEED,
        "MECHCONF2": MECHCONF2_MASTER_SEED,
        "MEMCONF": MEMORY_CONFIRM_MASTER_SEED,
        "episode_bootstrap": EPISODE_BOOTSTRAP_SEED,
        "old_regime_development": OLD_REGIME_DEVELOPMENT_SEED,
        "old_regime_confirmation": OLD_REGIME_CONFIRMATION_SEED,
        "old_regime_bootstrap": OLD_REGIME_BOOTSTRAP_SEED,
        "old_regime_randomization": OLD_REGIME_RANDOMIZATION_SEED,
        "prediction_diagnostic": DIAGNOSTIC_MASTER_SEED,
        "prediction_pilot": PILOT_MASTER_SEED,
        "prediction_confirmation": CONFIRMATION_MASTER_SEED,
        "prediction_bootstrap": BOOTSTRAP_MASTER_SEED,
        "prediction_randomization": RANDOMIZATION_MASTER_SEED,
    }


def _experiment(master_seed: str, matrices: int) -> ExperimentConfig:
    cohort = CohortConfig(
        matrices=matrices, branches_per_state=BRANCHES, landmarks=LANDMARKS
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


def build_prediction_cohort(
    experiment: ExperimentConfig, cohort_name: str, cohort: CohortConfig
) -> list[PredictionCase]:
    """Build landmark states while retaining the preceding post-fission state."""

    cases: list[PredictionCase] = []
    for matrix_id in range(cohort.matrices):
        beta = generate_beta(
            experiment.gard,
            np.random.default_rng(
                derive_seed(experiment.master_seed, f"{cohort_name}.beta", matrix_id)
            ),
        )
        initial = generate_initial_composition(
            experiment.gard,
            np.random.default_rng(
                derive_seed(experiment.master_seed, f"{cohort_name}.initial", matrix_id)
            ),
        )
        for candidate, contract in CANDIDATES.items():
            lineage: list[Snapshot] | None = None
            for attempt in range(100):
                rng = np.random.default_rng(
                    derive_seed(
                        experiment.master_seed,
                        f"{cohort_name}.main_path",
                        candidate,
                        matrix_id,
                        attempt,
                    )
                )
                try:
                    lineage = simulate_lineage(
                        initial, beta, experiment.gard, contract, rng
                    )
                    break
                except SimulationError:
                    continue
            if lineage is None:
                raise SimulationError(
                    f"failed complete {cohort_name} trajectory for candidate "
                    f"{candidate}, matrix {matrix_id}"
                )
            by_generation = {item.generation: item for item in lineage}
            for landmark in cohort.landmarks:
                snapshot = by_generation[landmark]
                previous = by_generation.get(landmark - 1)
                cases.append(
                    PredictionCase(
                        state_id=(
                            f"{cohort_name}-c{candidate}-m{matrix_id:03d}-g{landmark:03d}"
                        ),
                        cohort=cohort_name,
                        candidate=candidate,
                        matrix_id=matrix_id,
                        landmark=landmark,
                        beta=beta,
                        snapshot=snapshot,
                        previous_composition=(
                            previous.composition.copy()
                            if previous is not None
                            else None
                        ),
                    )
                )
    return cases


def _branch_worker(
    args: tuple[PredictionCase, ExperimentConfig, int],
) -> PredictionBranchBatch:
    case, experiment, branches = args
    limiter = threadpool_limits(limits=1)
    try:
        targets = np.empty((branches, len(ENDPOINTS)), dtype=np.int8)
        stages = np.empty((branches, 2), dtype=np.int8)
        onsets = np.empty((branches, len(ENDPOINTS)), dtype=np.int16)
        completed = np.empty(branches, dtype=np.int8)
        observed = np.empty(branches, dtype=np.int16)
        first_break = np.empty(branches, dtype=np.int16)
        first_run8 = np.empty(branches, dtype=np.int16)
        longest = np.empty(branches, dtype=np.int16)
        window_count = np.empty(branches, dtype=np.int16)
        margins = np.empty((branches, len(ENDPOINTS)), dtype=np.float64)
        post_break = np.full(
            (branches, len(POST_BREAK_FEATURE_NAMES)), np.nan, dtype=np.float64
        )
        window_rows: list[NDArray[np.float64]] = []
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
            records, complete = simulate_future_absorbing(
                case.snapshot,
                case.beta,
                experiment.gard,
                contract,
                experiment.horizon,
                rng,
            )
            outcome = evaluate_rich_regime(records)
            targets[branch] = [int(value) for value in outcome.targets]
            stages[branch] = (
                int(outcome.break_event),
                int(outcome.any_run8_after_break),
            )
            onsets[branch] = outcome.onsets
            completed[branch] = int(complete)
            observed[branch] = len(records)
            first_break[branch] = outcome.first_break_index
            first_run8[branch] = outcome.first_run8_start
            longest[branch] = outcome.longest_post_break_inheritance_run
            window_count[branch] = outcome.run8_window_count
            margins[branch] = outcome.best_margins
            window_rows.append(
                np.vstack([item.to_row() for item in outcome.windows])
                if outcome.windows
                else np.empty((0, len(WINDOW_METRIC_NAMES)), dtype=np.float64)
            )
            if outcome.first_break_index >= 0:
                stop = outcome.first_break_index + 1
                record = records[outcome.first_break_index]
                inheritance = case.snapshot.inheritance + tuple(
                    item.h > experiment.gard.inheritance_threshold
                    for item in records[:stop]
                )
                boundary_h = case.snapshot.boundary_h + tuple(
                    float(item.h) for item in records[:stop]
                )
                break_snapshot = Snapshot(
                    composition=record.daughter.copy(),
                    generation=case.snapshot.generation + stop,
                    inheritance=inheritance,
                    boundary_h=boundary_h,
                    previous_growth_steps=record.growth_steps,
                    cumulative_growth_steps=(
                        case.snapshot.cumulative_growth_steps
                        + sum(item.growth_steps for item in records[:stop])
                    ),
                )
                post_break[branch] = compact_post_break_features(
                    break_snapshot,
                    case.beta,
                    experiment,
                    case.candidate,
                    record.parent,
                )
        return PredictionBranchBatch(
            targets=targets,
            stages=stages,
            onsets=onsets,
            completed_horizon=completed,
            observed_fissions=observed,
            first_break_index=first_break,
            first_run8_start=first_run8,
            longest_run=longest,
            window_count=window_count,
            best_margins=margins,
            post_break_features=post_break,
            windows=tuple(window_rows),
        )
    finally:
        limiter.restore_original_limits()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _checkpoint_contract(
    cases: list[PredictionCase],
    experiment: ExperimentConfig,
    branches: int,
    label: str,
) -> dict[str, Any]:
    state_digest = hashlib.sha256(
        "\n".join(case.state_id for case in cases).encode("utf-8")
    ).hexdigest()
    return {
        "format": CHECKPOINT_FORMAT,
        "label": label,
        "source_hashes": _source_hashes(),
        "experiment": experiment.to_dict(),
        "branches": branches,
        "states": len(cases),
        "state_id_digest": state_digest,
    }


def _prepare_checkpoint(
    directory: Path,
    cases: list[PredictionCase],
    experiment: ExperimentConfig,
    branches: int,
    label: str,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "checkpoint_contract.json"
    expected = json.loads(
        json.dumps(
            _json_ready(_checkpoint_contract(cases, experiment, branches, label))
        )
    )
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError(f"checkpoint contract changed: {directory}")
    else:
        unexpected = [item for item in directory.iterdir() if item.name != path.name]
        if unexpected:
            raise ValueError(f"unregistered files in new checkpoint: {directory}")
        _atomic_json(path, expected)
    return expected


def _validate_checkpoint_batch(batch: Any, branches: int) -> PredictionBranchBatch:
    if not isinstance(batch, PredictionBranchBatch):
        raise ValueError("checkpoint does not contain a prediction branch batch")
    expected = {
        "targets": (branches, len(ENDPOINTS)),
        "stages": (branches, 2),
        "onsets": (branches, len(ENDPOINTS)),
        "completed_horizon": (branches,),
        "observed_fissions": (branches,),
        "first_break_index": (branches,),
        "first_run8_start": (branches,),
        "longest_run": (branches,),
        "window_count": (branches,),
        "best_margins": (branches, len(ENDPOINTS)),
        "post_break_features": (branches, len(POST_BREAK_FEATURE_NAMES)),
    }
    for name, shape in expected.items():
        if getattr(batch, name).shape != shape:
            raise ValueError(f"invalid checkpoint batch shape: {name}")
    if len(batch.windows) != branches:
        raise ValueError("invalid checkpoint window count")
    return batch


def _save_checkpoint_batch(
    directory: Path, index: int, batch: PredictionBranchBatch
) -> None:
    destination = directory / f"batch_{index:05d}.pkl"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(batch, handle, protocol=5)
    os.replace(temporary, destination)


def _checkpoint_status(directory: Path, label: str, complete: int, total: int) -> None:
    _atomic_json(
        directory / "status.json",
        {
            "format": CHECKPOINT_FORMAT,
            "label": label,
            "completed_states": complete,
            "total_states": total,
            "fraction_complete": complete / max(total, 1),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _campaign_status(directory: Path, campaign: str, phase: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        directory / "campaign_status.json",
        {
            "format": CHECKPOINT_FORMAT,
            "campaign": campaign,
            "phase": phase,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def run_prediction_branches(
    cases: list[PredictionCase],
    experiment: ExperimentConfig,
    branches: int,
    workers: int,
    label: str,
    checkpoint_directory: Path | None = None,
) -> list[PredictionBranchBatch]:
    arguments = [(case, experiment, branches) for case in cases]
    total = len(arguments)
    progress_every = max(1, min(25, total // 20))
    completed: dict[int, PredictionBranchBatch] = {}
    if checkpoint_directory is not None:
        checkpoint_directory = checkpoint_directory.resolve()
        _prepare_checkpoint(checkpoint_directory, cases, experiment, branches, label)
        for index in range(total):
            path = checkpoint_directory / f"batch_{index:05d}.pkl"
            if path.exists():
                with path.open("rb") as handle:
                    completed[index] = _validate_checkpoint_batch(
                        pickle.load(handle), branches
                    )
        if completed:
            print(
                f"[{label}] resumed {len(completed)}/{total} checkpointed states",
                flush=True,
            )
        _checkpoint_status(checkpoint_directory, label, len(completed), total)
    remaining = [index for index in range(total) if index not in completed]

    def retain(index: int, batch: PredictionBranchBatch) -> None:
        completed[index] = batch
        if checkpoint_directory is not None:
            _save_checkpoint_batch(checkpoint_directory, index, batch)
            _checkpoint_status(checkpoint_directory, label, len(completed), total)
        count = len(completed)
        if count % progress_every == 0 or count == total:
            print(f"[{label}] states {count}/{total}", flush=True)

    if workers <= 1:
        iterator: Iterable[PredictionBranchBatch] = map(
            _branch_worker, (arguments[index] for index in remaining)
        )
        for index, batch in zip(remaining, iterator):
            retain(index, batch)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(
                _branch_worker,
                (arguments[index] for index in remaining),
                chunksize=1,
            )
            for index, batch in zip(remaining, iterator):
                retain(index, batch)
    if len(completed) != total:
        raise RuntimeError(f"{label} checkpoint ended before every state completed")
    return [completed[index] for index in range(total)]


def _batch_digest(batches: list[PredictionBranchBatch]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        for value in (
            batch.targets,
            batch.stages,
            batch.onsets,
            batch.completed_horizon,
            batch.observed_fissions,
            batch.first_break_index,
            batch.first_run8_start,
            batch.longest_run,
            batch.window_count,
        ):
            digest.update(np.ascontiguousarray(value).tobytes())
        for value in (batch.best_margins, batch.post_break_features, *batch.windows):
            canonical = np.nan_to_num(value, nan=-999.0, posinf=999.0, neginf=-999.0)
            digest.update(np.ascontiguousarray(canonical).tobytes())
    return digest.hexdigest()


def replay_audit(
    original: list[PredictionBranchBatch], replay: list[PredictionBranchBatch]
) -> dict[str, Any]:
    if len(original) != len(replay):
        raise ValueError("replay batch count differs")
    maximum = 0.0
    continuous = 0
    for left, right in zip(original, replay):
        for name in (
            "targets",
            "stages",
            "onsets",
            "completed_horizon",
            "observed_fissions",
            "first_break_index",
            "first_run8_start",
            "longest_run",
            "window_count",
        ):
            if not np.array_equal(getattr(left, name), getattr(right, name)):
                raise ValueError(f"discrete replay mismatch: {name}")
        if len(left.windows) != len(right.windows):
            raise ValueError("window replay count mismatch")
        for lvalue, rvalue in (
            (left.best_margins, right.best_margins),
            (left.post_break_features, right.post_break_features),
            *zip(left.windows, right.windows),
        ):
            if lvalue.shape != rvalue.shape or not np.array_equal(
                np.isfinite(lvalue), np.isfinite(rvalue)
            ):
                raise ValueError("continuous replay structure mismatch")
            finite = np.isfinite(lvalue)
            if finite.any():
                maximum = max(
                    maximum, float(np.max(np.abs(lvalue[finite] - rvalue[finite])))
                )
                continuous += int(finite.sum())
    if maximum > CONTINUOUS_TOLERANCE:
        raise ValueError(f"continuous replay mismatch: {maximum}")
    first = _batch_digest(original)
    second = _batch_digest(replay)
    return {
        "discrete_exact": True,
        "continuous_values_compared": continuous,
        "maximum_continuous_absolute_error": maximum,
        "continuous_within_1e-14": maximum <= CONTINUOUS_TOLERANCE,
        "first_digest": first,
        "second_digest": second,
        "digests_exact": first == second,
    }


def _labels(batches: list[PredictionBranchBatch]) -> dict[str, NDArray[np.int8]]:
    targets = np.stack([batch.targets for batch in batches])
    stages = np.stack([batch.stages for batch in batches])
    return {
        "strict": targets[:, :, 0],
        "first5": targets[:, :, 1],
        "centroid": targets[:, :, 2],
        "break": stages[:, :, 0],
        "run8": stages[:, :, 1],
    }


def _candidate_mask(cases: list[PredictionCase], candidate: str) -> NDArray[np.bool_]:
    return np.asarray([case.candidate == candidate for case in cases], dtype=bool)


def _matrix_ids(cases: list[PredictionCase]) -> NDArray[np.int64]:
    return np.asarray([case.matrix_id for case in cases], dtype=np.int64)


def _power(
    labels: NDArray[np.int8], cases: list[PredictionCase]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    matrix_ids = _matrix_ids(cases)
    for candidate in CANDIDATES:
        selected = _candidate_mask(cases, candidate)
        values = labels[selected]
        ids = matrix_ids[selected]
        event_matrices = sum(
            bool(values[ids == matrix_id].any()) for matrix_id in np.unique(ids)
        )
        events = int(values.sum())
        output[candidate] = {
            "events": events,
            "event_matrices": int(event_matrices),
            "minimum_events": MINIMUM_EVENTS,
            "minimum_event_matrices": MINIMUM_EVENT_MATRICES,
            "adequate": bool(
                events >= MINIMUM_EVENTS and event_matrices >= MINIMUM_EVENT_MATRICES
            ),
        }
    return output


def _matrix_values(values: FloatArray, matrix_ids: NDArray[np.int64]) -> FloatArray:
    return np.asarray(
        [values[matrix_ids == item].mean() for item in np.unique(matrix_ids)],
        dtype=np.float64,
    )


def select_model_family(
    cases: list[PredictionCase],
    labels: dict[str, NDArray[np.int8]],
    oof: dict[str, dict[str, dict[str, FloatArray]]],
) -> dict[str, Any]:
    """Apply consistency, one-SE, and bootstrap-selection gates."""

    ids = _matrix_ids(cases)
    rows: list[dict[str, Any]] = []
    per_matrix_loss: dict[str, dict[str, FloatArray]] = {
        family: {} for family in MODEL_FAMILIES
    }
    common_matrix_order: NDArray[np.int64] | None = None
    for family in MODEL_FAMILIES:
        gains: dict[str, dict[str, float]] = {}
        candidate_losses: list[float] = []
        matrix_losses: list[FloatArray] = []
        all_positive = True
        for candidate in CANDIDATES:
            selected = _candidate_mask(cases, candidate)
            candidate_labels = labels["strict"][selected]
            candidate_ids = ids[selected]
            predictions = oof[family][candidate]
            gains[candidate] = {}
            for half, values in (
                ("A", candidate_labels[:, : candidate_labels.shape[1] // 2]),
                ("B", candidate_labels[:, candidate_labels.shape[1] // 2 :]),
            ):
                q = values.mean(axis=1)
                gain = float(
                    np.mean(
                        _state_log_loss(q, predictions["h10"])
                        - _state_log_loss(q, predictions["enhanced"])
                    )
                )
                gains[candidate][half] = gain
                all_positive &= gain > 0.0
            q_all = candidate_labels.mean(axis=1)
            loss = _state_log_loss(q_all, predictions["enhanced"])
            matrix_order = np.unique(candidate_ids)
            if common_matrix_order is None:
                common_matrix_order = matrix_order
            elif not np.array_equal(matrix_order, common_matrix_order):
                raise ValueError("candidate matrix supports differ during selection")
            grouped = _matrix_values(loss, candidate_ids)
            per_matrix_loss[family][candidate] = grouped
            matrix_losses.append(grouped)
            candidate_losses.append(float(grouped.mean()))
        # Both simulator candidates use the same catalytic matrix.  Average
        # their losses within matrix before estimating the one-SE threshold;
        # treating the two candidate rows as independent would be
        # pseudoreplication and can underestimate uncertainty.
        combined = np.mean(np.vstack(matrix_losses), axis=0)
        rows.append(
            {
                "family": family,
                "candidate_equal_log_loss": float(np.mean(candidate_losses)),
                "matrix_loss_standard_error": float(
                    combined.std(ddof=1) / np.sqrt(combined.size)
                ),
                "half_gains": gains,
                "positive_all_candidate_halves": bool(all_positive),
            }
        )

    eligible = [row for row in rows if row["positive_all_candidate_halves"]]
    if not eligible:
        return {
            "passed": False,
            "reason": "no family improved both candidates and all branch halves",
            "selected_family": None,
            "families": rows,
        }
    best = min(eligible, key=lambda row: row["candidate_equal_log_loss"])
    threshold = best["candidate_equal_log_loss"] + best["matrix_loss_standard_error"]
    within_one_se = {
        row["family"]
        for row in eligible
        if row["candidate_equal_log_loss"] <= threshold
    }
    selected_family = next(name for name in MODEL_SIMPLICITY if name in within_one_se)

    rng = np.random.default_rng(
        derive_seed(BOOTSTRAP_MASTER_SEED, "pilot.model_selection")
    )
    wins = {family: 0 for family in MODEL_FAMILIES}
    if common_matrix_order is None:  # pragma: no cover - candidates are fixed
        raise ValueError("model selection has no catalytic matrices")
    for _ in range(BOOTSTRAP_SELECTION_REPETITIONS):
        # One paired matrix draw is shared by both candidates and every family
        # so model comparisons preserve all dependencies in the design.
        sample = rng.integers(
            0, common_matrix_order.size, size=common_matrix_order.size
        )
        bootstrap_losses: dict[str, float] = {}
        for family in within_one_se:
            candidate_values = [
                float(per_matrix_loss[family][candidate][sample].mean())
                for candidate in CANDIDATES
            ]
            bootstrap_losses[family] = float(np.mean(candidate_values))
        winner = min(
            within_one_se,
            key=lambda name: (bootstrap_losses[name], MODEL_SIMPLICITY.index(name)),
        )
        wins[winner] += 1
    frequencies = {
        family: wins[family] / BOOTSTRAP_SELECTION_REPETITIONS
        for family in MODEL_FAMILIES
    }
    stable = frequencies[selected_family] >= BOOTSTRAP_SELECTION_FRACTION
    return {
        "passed": bool(stable),
        "reason": (
            "all pilot selection gates passed"
            if stable
            else "one-SE selection frequency below 75%"
        ),
        "selected_family": selected_family if stable else None,
        "provisional_family": selected_family,
        "one_se_threshold": threshold,
        "within_one_se": sorted(within_one_se, key=MODEL_SIMPLICITY.index),
        "bootstrap_selection_frequencies": frequencies,
        "bootstrap_repetitions": BOOTSTRAP_SELECTION_REPETITIONS,
        "bootstrap_unit": "paired catalytic matrix across candidates and families",
        "required_selection_frequency": BOOTSTRAP_SELECTION_FRACTION,
        "families": rows,
    }


def _fit_post_break_models(
    cases: list[PredictionCase],
    batches: list[PredictionBranchBatch],
) -> tuple[dict[str, SequentialRidgeModel], dict[str, Any]]:
    models: dict[str, SequentialRidgeModel] = {}
    audit: dict[str, Any] = {}
    for candidate in CANDIDATES:
        features: list[FloatArray] = []
        outcomes: list[NDArray[np.int8]] = []
        ids: list[NDArray[np.int64]] = []
        for case, batch in zip(cases, batches):
            if case.candidate != candidate:
                continue
            selected = batch.stages[:, 0].astype(bool)
            if selected.any():
                features.append(batch.post_break_features[selected])
                outcomes.append(batch.targets[selected, 0])
                ids.append(np.full(int(selected.sum()), case.matrix_id, dtype=np.int64))
        if not features:
            audit[candidate] = {
                "break_futures": 0,
                "strict_events_after_break": 0,
                "matrices": 0,
                "model_fitted": False,
                "reason": "no observed breaks",
            }
            continue
        x = np.vstack(features)
        y = np.concatenate(outcomes)
        matrix_ids = np.concatenate(ids)
        if np.unique(matrix_ids).size < 2:
            audit[candidate] = {
                "break_futures": int(y.size),
                "strict_events_after_break": int(y.sum()),
                "matrices": int(np.unique(matrix_ids).size),
                "model_fitted": False,
                "reason": "fewer than two break-positive matrices",
            }
            continue
        if not np.isfinite(x).all():
            raise ValueError("post-break features contain non-finite values")
        model = fit_sequential_ridge(
            {"postbreak": x},
            y.astype(np.float64),
            np.ones(y.size, dtype=np.float64),
            matrix_ids,
            ("postbreak",),
        )
        models[candidate] = model
        audit[candidate] = {
            "break_futures": int(y.size),
            "strict_events_after_break": int(y.sum()),
            "matrices": int(np.unique(matrix_ids).size),
            "model_fitted": True,
        }
    return models, audit


def _write_branch_tables(
    directory: Path,
    stem: str,
    cases: list[PredictionCase],
    batches: list[PredictionBranchBatch],
) -> None:
    branch_path = directory / f"{stem}_branches.csv.gz"
    window_path = directory / f"{stem}_windows.csv.gz"
    branch_columns = (
        "state_id",
        "candidate",
        "matrix_id",
        "landmark",
        "branch",
        "half",
        "break_event",
        "any_run8_after_break",
        *ENDPOINTS,
        *(f"{name}_onset" for name in ENDPOINTS),
        "completed_horizon",
        "observed_fissions",
        "first_break_index",
        "first_run8_start",
        "longest_post_break_inheritance_run",
        "run8_window_count",
        "best_strict_margin",
        "best_first5_margin",
        "best_centroid_margin",
    )
    with gzip.open(branch_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(branch_columns)
        for case, batch in zip(cases, batches):
            for branch in range(batch.targets.shape[0]):
                writer.writerow(
                    (
                        case.state_id,
                        case.candidate,
                        case.matrix_id,
                        case.landmark,
                        branch,
                        "A" if branch < batch.targets.shape[0] // 2 else "B",
                        *batch.stages[branch].tolist(),
                        *batch.targets[branch].tolist(),
                        *batch.onsets[branch].tolist(),
                        int(batch.completed_horizon[branch]),
                        int(batch.observed_fissions[branch]),
                        int(batch.first_break_index[branch]),
                        int(batch.first_run8_start[branch]),
                        int(batch.longest_run[branch]),
                        int(batch.window_count[branch]),
                        *batch.best_margins[branch].tolist(),
                    )
                )
    with gzip.open(window_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("state_id", "branch", *WINDOW_METRIC_NAMES))
        for case, batch in zip(cases, batches):
            for branch, rows in enumerate(batch.windows):
                for row in rows:
                    writer.writerow((case.state_id, branch, *row.tolist()))


def _save_arrays(
    path: Path,
    cases: list[PredictionCase],
    raw: PredictionRawFeatures,
    batches: list[PredictionBranchBatch],
) -> None:
    np.savez_compressed(
        path,
        state_ids=np.asarray([case.state_id for case in cases]),
        candidates=np.asarray([case.candidate for case in cases]),
        matrix_ids=_matrix_ids(cases),
        landmarks=np.asarray([case.landmark for case in cases], dtype=np.int64),
        compositions=np.vstack([case.snapshot.composition for case in cases]),
        previous_compositions=np.vstack([case.previous_composition for case in cases]),
        h10=raw.h10,
        state=raw.state,
        beta=raw.beta,
        interaction=raw.interaction,
        dynamics=raw.dynamics,
        targets=np.stack([batch.targets for batch in batches]),
        stages=np.stack([batch.stages for batch in batches]),
        onsets=np.stack([batch.onsets for batch in batches]),
        completed_horizon=np.stack([batch.completed_horizon for batch in batches]),
        observed_fissions=np.stack([batch.observed_fissions for batch in batches]),
        first_break_index=np.stack([batch.first_break_index for batch in batches]),
        first_run8_start=np.stack([batch.first_run8_start for batch in batches]),
        longest_run=np.stack([batch.longest_run for batch in batches]),
        window_count=np.stack([batch.window_count for batch in batches]),
        best_margins=np.stack([batch.best_margins for batch in batches]),
        post_break_features=np.stack([batch.post_break_features for batch in batches]),
    )


def _strict_labels_from_branch_table(
    path: Path, cases: list[PredictionCase], branches: int = BRANCHES
) -> NDArray[np.int8]:
    table = pd.read_csv(
        path,
        usecols=["state_id", "branch", PRIMARY_ENDPOINT],
        dtype={"state_id": str, PRIMARY_ENDPOINT: np.int8},
    )
    expected_rows = len(cases) * branches
    if len(table) != expected_rows:
        raise ValueError("confirmation branch table row count mismatch")
    expected_states = np.repeat(np.asarray([case.state_id for case in cases]), branches)
    expected_branches = np.tile(np.arange(branches), len(cases))
    if not np.array_equal(table["state_id"].to_numpy(), expected_states):
        raise ValueError("confirmation branch table state order mismatch")
    if not np.array_equal(table["branch"].to_numpy(), expected_branches):
        raise ValueError("confirmation branch table branch order mismatch")
    values = (
        table[PRIMARY_ENDPOINT].to_numpy(dtype=np.int8).reshape(len(cases), branches)
    )
    if np.any((values != 0) & (values != 1)):
        raise ValueError("confirmation branch table contains a nonbinary endpoint")
    return values


def _protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": DESIGN_FORMAT,
        "status": "sealed_before_pilot_matrix_generation",
        "primary_endpoint": (
            "unconditional break followed by a distinct coherent inherited "
            "eight-fission episode; exact legacy thresholds and F32 horizon"
        ),
        "endpoint_contract": {
            "horizon_fissions": HORIZON,
            "inheritance": {
                "rule": "parent-daughter cosine similarity > threshold",
                "threshold": INHERITANCE_THRESHOLD,
            },
            "first_break": {
                "rule": "first future fission not satisfying inheritance",
                "inclusive_upper_bound": INHERITANCE_THRESHOLD,
            },
            "required_consecutive_inherited_fissions": RUN_LENGTH,
            "post_break_windows_only": True,
            "search_every_eligible_window": True,
            "coherence": {
                "rule": "every daughter pair in the window > threshold",
                "threshold": COHERENCE_THRESHOLD,
            },
            "old_anchor_distinctness": {
                "anchor": "parent composition at the first break",
                "rule": "every window daughter <= threshold from anchor",
                "threshold": DISTINCTNESS_THRESHOLD,
            },
            "primary_is_unconditional": True,
            "continuous_margin": (
                "max over eligible windows of min(minimum_pairwise - "
                f"{COHERENCE_THRESHOLD:g}, "
                f"nextafter({DISTINCTNESS_THRESHOLD:g},+infinity) - "
                "maximum_old_anchor_similarity); primary event iff best margin > 0"
            ),
        },
        "secondary_no_rescue": [
            "first-five coherence",
            "centroid coherence",
            "continuous best margins",
            "hurdle stages",
            "post-break prediction",
        ],
        "cohorts": {
            "pilot": _experiment(PILOT_MASTER_SEED, PILOT_MATRICES).to_dict(),
            "confirmation": _experiment(
                CONFIRMATION_MASTER_SEED, CONFIRMATION_MATRICES
            ).to_dict(),
            "pilot_futures": PILOT_MATRICES * 2 * len(LANDMARKS) * BRANCHES,
            "confirmation_futures": (
                CONFIRMATION_MATRICES * 2 * len(LANDMARKS) * BRANCHES
            ),
            "full_exact_replay": True,
        },
        "features": {
            "baseline": "unpenalized provenance-complete unique h10",
            "added": [
                "state-only",
                "complete beta-only without PCA",
                "state-beta interaction",
                "analytic local dynamics and stability",
            ],
            "future_information_in_primary_predictor": False,
            "molecule_label_permutation_invariant": True,
        },
        "model_families": list(MODEL_FAMILIES),
        "family_contracts": {
            "direct_ridge": "h10 -> state -> complete beta -> interaction offsets",
            "hurdle": (
                "product of P(break), P(later run8|break), and "
                "P(strict geometry|later run8), each using all registered blocks"
            ),
            "hierarchical_offset": (
                "complete beta propensity -> h10 -> state -> local dynamics"
            ),
            "local_dynamics": "h10 -> complete beta -> local dynamics offsets",
            "auxiliary_stack": (
                "h10+dynamics first5 and centroid predictions generated out of fold, "
                "then h10 -> dynamics -> two auxiliary-logit offsets"
            ),
            "guarded_nonlinear": (
                "bounded histogram-gradient model on h10, dynamics, and an "
                "out-of-fold complete-beta propensity logit; out-of-fold Platt calibration"
            ),
        },
        "model_contract": {
            "ridge_lambdas": list(RIDGE_LAMBDAS),
            "common_h10_baseline_unpenalized": True,
            "registered_simplicity_order": list(MODEL_SIMPLICITY),
            "nonlinear_grid": list(NONLINEAR_GRID),
            "nonlinear_calibration": "training-only out-of-fold Platt calibration",
            "auxiliary_stack": "first5 and centroid logits generated out of fold",
            "hyperparameters_selected_once_on_pilot_folds": True,
        },
        "selection": {
            "split": "five whole-catalytic-matrix folds",
            "required_positive_cells": "both candidates x both fixed branch halves",
            "rule": "one standard error, then registered simplicity order",
            "bootstrap_unit": (
                "paired catalytic matrix, shared across candidates and model families"
            ),
            "bootstrap_selection_repetitions": BOOTSTRAP_SELECTION_REPETITIONS,
            "minimum_selection_frequency": BOOTSTRAP_SELECTION_FRACTION,
            "stop_if_no_model_passes": True,
        },
        "power": {
            "minimum_events_per_candidate": MINIMUM_EVENTS,
            "minimum_event_matrices_per_candidate": MINIMUM_EVENT_MATRICES,
        },
        "confirmation": {
            "primary_contrast": "frozen enhanced predictor versus h10",
            "cells": "two candidates x two fixed branch halves",
            "gate": (
                "positive log-loss gain, whole-matrix bootstrap 95% lower bound "
                "above zero, and Holm-adjusted matrix randomization p<0.05 in all cells"
            ),
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "secondary_can_rescue": False,
        },
        "operations": {
            "per_state_checkpoints": True,
            "checkpoint_contract_includes_source_experiment_and_state_digest": True,
            "generation_and_replay_checkpoints_separate": True,
            "status_command_read_only": True,
            "checkpoint_cleanup_automatic": False,
        },
        "claim_boundary": {
            "tested": "predictability of strict break-and-distinct-renewal probability",
            "not_tested": [
                "causal control",
                "molecular intervention",
                "recurrence",
                "attractor switching",
                "origin-of-life realism",
            ],
        },
    }
    value["protocol_id"] = _canonical_digest(value)
    return value


def register_design(output_directory: Path) -> None:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    seed_domains = _seed_domains()
    if len(set(seed_domains.values())) != len(seed_domains):
        raise ValueError("prediction seed domain collision")
    with _atomic_destination(output_directory) as output:
        protocol = _protocol()
        (output / "protocol.json").write_text(
            json.dumps(_json_ready(protocol), indent=2, sort_keys=True) + "\n"
        )
        provenance = prediction_provenance_contract()
        (output / "feature_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        payload: dict[str, Any] = {
            "format": DESIGN_FORMAT,
            "status": "sealed_before_pilot_matrix_generation",
            "protocol_id": protocol["protocol_id"],
            "protocol_digest": sha256_file(output / "protocol.json"),
            "feature_provenance_digest": sha256_file(
                output / "feature_provenance.json"
            ),
            "source_hashes": _source_hashes(),
            "seed_domains": seed_domains,
            "all_seed_domains_unique": True,
        }
        payload["registration_id"] = _canonical_digest(payload)
        (output / "registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        write_checksums(output)
    print(f"Prediction design sealed at {output_directory}", flush=True)


def verify_design(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text())
    identifier = payload.pop("registration_id")
    if (
        payload.get("format") != DESIGN_FORMAT
        or _canonical_digest(payload) != identifier
    ):
        raise ValueError("invalid prediction registration")
    payload["registration_id"] = identifier
    if payload["source_hashes"] != _source_hashes():
        changed = [
            name
            for name, digest in payload["source_hashes"].items()
            if _source_hashes().get(name) != digest
        ]
        raise ValueError(f"registered prediction source changed: {changed}")
    if payload["protocol_digest"] != sha256_file(directory / "protocol.json"):
        raise ValueError("prediction protocol digest mismatch")
    if (
        payload["seed_domains"] != _seed_domains()
        or not payload["all_seed_domains_unique"]
    ):
        raise ValueError("prediction seed-domain contract changed")
    if json.loads((directory / "protocol.json").read_text()) != json.loads(
        json.dumps(_protocol())
    ):
        raise ValueError("prediction protocol implementation diverged")
    return payload


def run_diagnostic(
    source: Path, output_directory: Path, reconstruct: bool = True
) -> None:
    source = source.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    verify_checksums(source)
    with np.load(source / "confirmation_arrays.npz") as arrays:
        candidates = arrays["candidates"].astype(str)
        matrix_ids = arrays["matrix_ids"].astype(np.int64)
        strict = arrays["labels_primary_all8"].astype(np.int8)
        breaks = arrays["first_break_index"] >= 0
        run8 = arrays["first_run8_start"] >= 0
        archived = {
            name: arrays[name].copy()
            for name in ("h10", "state_block", "beta_block", "interaction_block")
        }
    states = pd.read_csv(source / "confirmation_states.csv", dtype={"candidate": str})
    states["candidate"] = states["candidate"].str.zfill(2)
    stage: dict[str, Any] = {}
    stability: dict[str, Any] = {}
    rng = np.random.default_rng(
        derive_seed(DIAGNOSTIC_MASTER_SEED, "sealed_subsamples")
    )
    for candidate in CANDIDATES:
        selected = candidates == candidate
        stage[candidate] = {
            "break_rate": float(breaks[selected].mean()),
            "any_run8_rate": float(run8[selected].mean()),
            "strict_rate": float(strict[selected].mean()),
            "strict_given_run8": float(
                strict[selected].sum() / max(int(run8[selected].sum()), 1)
            ),
        }
        ids = matrix_ids[selected]
        values = strict[selected]
        state_rows = states[states["candidate"] == candidate]
        baseline = state_rows["prediction_primary_all8_h10"].to_numpy()
        enhanced = state_rows["prediction_primary_all8_h10_state"].to_numpy()
        unique = np.unique(ids)
        samples: dict[str, Any] = {}
        for half, q in (
            ("A", values[:, : SEALED_BRANCHES // 2].mean(axis=1)),
            ("B", values[:, SEALED_BRANCHES // 2 :].mean(axis=1)),
        ):
            per_matrix = _matrix_values(
                _state_log_loss(q, baseline) - _state_log_loss(q, enhanced), ids
            )
            draws = np.empty(BOOTSTRAP_SELECTION_REPETITIONS)
            for index in range(draws.size):
                chosen = rng.choice(unique.size, size=PILOT_MATRICES, replace=False)
                draws[index] = per_matrix[chosen].mean()
            samples[half] = {
                "mean_gain": float(draws.mean()),
                "positive_fraction": float(np.mean(draws > 0.0)),
                "quantiles_025_50_975": np.quantile(
                    draws, (0.025, 0.5, 0.975)
                ).tolist(),
            }
        stability[candidate] = samples

    reconstruction: dict[str, Any] = {"performed": False}
    if reconstruct:
        experiment = _experiment(OLD_REGIME_CONFIRMATION_SEED, CONFIRMATION_MATRICES)
        cases = build_prediction_cohort(experiment, "REGCONF", experiment.confirmation)
        raw = extract_prediction_features(cases, experiment)
        comparisons = {
            "h10": raw.h10,
            "state_block": raw.state,
            "beta_block": raw.beta,
            "interaction_block": raw.interaction,
        }
        errors = {
            name: float(np.max(np.abs(values - archived[name])))
            for name, values in comparisons.items()
        }
        if any(value > CONTINUOUS_TOLERANCE for value in errors.values()):
            raise ValueError(f"sealed feature reconstruction mismatch: {errors}")
        reconstruction = {
            "performed": True,
            "states": len(cases),
            "maximum_absolute_errors": errors,
            "within_1e-14": True,
            "dynamic_coordinates_per_state": int(raw.dynamics.shape[1]),
            "dynamic_features_all_finite": bool(np.isfinite(raw.dynamics).all()),
            "dynamic_feature_digest": hashlib.sha256(
                np.ascontiguousarray(raw.dynamics).tobytes()
            ).hexdigest(),
        }
    metrics = {
        "format": DIAGNOSTIC_FORMAT,
        "status": "post_hoc_nonconfirmatory",
        "source": str(source),
        "source_checksum_digest": sha256_file(source / "SHA256SUMS"),
        "implementation_source_hashes": _source_hashes(),
        "diagnostic_seed": DIAGNOSTIC_MASTER_SEED,
        "stage_rates": stage,
        "80_matrix_subsample_stability": stability,
        "feature_reconstruction": reconstruction,
        "cannot_select_or_validate_final_model": True,
    }
    with _atomic_destination(output_directory) as output:
        (output / "metrics.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n"
        )
        lines = [
            "# Strict-regime prediction diagnostic",
            "",
            "This is a post-hoc analysis of the already-seen sealed confirmation cohort. It cannot select or validate the next predictor.",
            "",
            "| Candidate | Break | Any later run8 | Strict | Strict given run8 |",
            "|---|---:|---:|---:|---:|",
        ]
        for candidate, item in stage.items():
            lines.append(
                f"| {candidate} | {item['break_rate']:.4f} | {item['any_run8_rate']:.4f} | {item['strict_rate']:.4f} | {item['strict_given_run8']:.4f} |"
            )
        lines.extend(
            [
                "",
                "The subsampling analysis estimates how variable an 80-matrix pilot may be; it is not a prospective success test.",
                "",
            ]
        )
        (output / "DIAGNOSTIC_REPORT.md").write_text("\n".join(lines))
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "format": DIAGNOSTIC_FORMAT,
                    "claim_status": "post_hoc_only",
                    "source_unchanged": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        write_checksums(output)


def _save_pilot_state_table(
    path: Path,
    cases: list[PredictionCase],
    labels: dict[str, NDArray[np.int8]],
    oof: dict[str, dict[str, dict[str, FloatArray]]],
) -> None:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        row: dict[str, Any] = {
            "state_id": case.state_id,
            "candidate": case.candidate,
            "matrix_id": case.matrix_id,
            "landmark": case.landmark,
            "q_strict_all": float(labels["strict"][index].mean()),
            "q_strict_A": float(labels["strict"][index, : BRANCHES // 2].mean()),
            "q_strict_B": float(labels["strict"][index, BRANCHES // 2 :].mean()),
            "q_break": float(labels["break"][index].mean()),
            "q_run8": float(labels["run8"][index].mean()),
            "q_first5": float(labels["first5"][index].mean()),
            "q_centroid": float(labels["centroid"][index].mean()),
        }
        candidate_indices = np.flatnonzero(_candidate_mask(cases, case.candidate))
        local = int(np.flatnonzero(candidate_indices == index)[0])
        for family in MODEL_FAMILIES:
            row[f"prediction_{family}_h10"] = float(
                oof[family][case.candidate]["h10"][local]
            )
            row[f"prediction_{family}_enhanced"] = float(
                oof[family][case.candidate]["enhanced"][local]
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.17g")


def run_pilot(
    registration: Path,
    output_directory: Path,
    workers: int,
    work_directory: Path | None = None,
) -> None:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    design = verify_design(registration)
    experiment = _experiment(PILOT_MASTER_SEED, PILOT_MATRICES)
    work = (
        work_directory.resolve()
        if work_directory is not None
        else output_directory.with_name(f".{output_directory.name}.work")
    )
    _campaign_status(work, "pilot", "building_trajectories_and_features")
    print("[pilot 1/9] Generating fresh pilot trajectories and features", flush=True)
    with threadpool_limits(limits=1):
        cases = build_prediction_cohort(experiment, "REGPILOT", experiment.development)
        raw = extract_prediction_features(cases, experiment)
    print("[pilot 2/9] Shooting 102,400 F32 futures", flush=True)
    _campaign_status(work, "pilot", "shooting_futures")
    batches = run_prediction_branches(
        cases,
        experiment,
        BRANCHES,
        workers,
        "pilot",
        checkpoint_directory=work / "generate",
    )
    print("[pilot 3/9] Replaying every pilot future", flush=True)
    _campaign_status(work, "pilot", "replaying_futures")
    replay_batches = run_prediction_branches(
        cases,
        experiment,
        BRANCHES,
        workers,
        "pilot-replay",
        checkpoint_directory=work / "replay",
    )
    replay = replay_audit(batches, replay_batches)
    labels = _labels(batches)
    power = _power(labels["strict"], cases)
    if not all(item["adequate"] for item in power.values()):
        selection: dict[str, Any] = {
            "passed": False,
            "reason": "pilot strict-event power gate failed",
            "selected_family": None,
            "families": [],
        }
        oof: dict[str, Any] = {}
        final_models: dict[str, Any] = {}
    else:
        print("[pilot 4/9] Cross-fitting the fixed model menu", flush=True)
        _campaign_status(work, "pilot", "crossfitting_models")
        oof = {family: {} for family in MODEL_FAMILIES}
        final_models = {family: {} for family in MODEL_FAMILIES}
        matrix_ids = _matrix_ids(cases)
        for family in MODEL_FAMILIES:
            for candidate in CANDIDATES:
                print(f"[models] {family}, candidate {candidate}", flush=True)
                selected = _candidate_mask(cases, candidate)
                predictions, model = crossfit_prediction_family(
                    family,
                    candidate,
                    raw.selected(selected),
                    {name: values[selected] for name, values in labels.items()},
                    matrix_ids[selected],
                )
                oof[family][candidate] = predictions
                final_models[family][candidate] = model
        selection = select_model_family(cases, labels, oof)
    print("[pilot 5/9] Fitting secondary post-break models", flush=True)
    _campaign_status(work, "pilot", "fitting_secondary_models")
    post_break_models, post_break_audit = _fit_post_break_models(cases, batches)

    with _atomic_destination(output_directory) as output:
        print("[pilot 6/9] Writing complete pilot artifacts", flush=True)
        _write_branch_tables(output, "pilot", cases, batches)
        _save_arrays(output / "pilot_arrays.npz", cases, raw, batches)
        if oof:
            _save_pilot_state_table(output / "pilot_states.csv", cases, labels, oof)
        (output / "selection.json").write_text(
            json.dumps(_json_ready(selection), indent=2, sort_keys=True) + "\n"
        )
        (output / "replay_audit.json").write_text(
            json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n"
        )
        if final_models:
            # Every fitted family is retained for audit, but only the selected
            # family is legal for confirmation scoring.
            with (output / "all_pilot_models.pkl").open("wb") as handle:
                pickle.dump(final_models, handle, protocol=5)
        if selection.get("passed"):
            selected_family = str(selection["selected_family"])
            selected_models = final_models[selected_family]
            save_prediction_models(output / "selected_models.pkl", selected_models)
            reloaded = load_prediction_models(output / "selected_models.pkl")
            maximum = 0.0
            for candidate in CANDIDATES:
                selected = _candidate_mask(cases, candidate)
                before = selected_models[candidate].predict(raw.selected(selected))
                after = reloaded[candidate].predict(raw.selected(selected))
                maximum = max(
                    maximum,
                    *(
                        float(np.max(np.abs(before[name] - after[name])))
                        for name in before
                    ),
                )
            portable = {"maximum_absolute_error": maximum, "exact": maximum == 0.0}
            summaries = {
                candidate: model_summary(selected_models[candidate])
                for candidate in CANDIDATES
            }
        else:
            portable = {"not_run": True, "reason": selection["reason"]}
            summaries = {}
        with (output / "post_break_models.pkl").open("wb") as handle:
            pickle.dump(post_break_models, handle, protocol=5)
        (output / "model_summary.json").write_text(
            json.dumps(_json_ready(summaries), indent=2, sort_keys=True) + "\n"
        )
        (output / "post_break_audit.json").write_text(
            json.dumps(_json_ready(post_break_audit), indent=2, sort_keys=True) + "\n"
        )
        seal: dict[str, Any] = {
            "format": PILOT_FORMAT,
            "status": (
                "model_frozen_for_confirmation"
                if selection.get("passed")
                else "stopped_before_confirmation"
            ),
            "design_registration_id": design["registration_id"],
            "design_path": str(registration.resolve()),
            "design_checksum_digest": sha256_file(
                registration.resolve() / "SHA256SUMS"
            ),
            "source_hashes": _source_hashes(),
            "experiment": experiment.to_dict(),
            "states": len(cases),
            "futures": len(cases) * BRANCHES,
            "power": power,
            "selection": selection,
            "replay_audit": replay,
            "portable_model_audit": portable,
            "checkpoint_audit": {
                "work_directory": str(work),
                "generation_contract_digest": sha256_file(
                    work / "generate" / "checkpoint_contract.json"
                ),
                "replay_contract_digest": sha256_file(
                    work / "replay" / "checkpoint_contract.json"
                ),
                "generation_complete": len(batches) == len(cases),
                "replay_complete": len(replay_batches) == len(cases),
                "resumable_per_state": True,
            },
            "runtime": _runtime_manifest(),
        }
        seal["pilot_seal_id"] = _canonical_digest(seal)
        (output / "pilot_seal.json").write_text(
            json.dumps(_json_ready(seal), indent=2, sort_keys=True) + "\n"
        )
        report = [
            "# Strict-regime prediction pilot",
            "",
            "This fresh cohort selected a model recipe; it cannot confirm the scientific prediction claim.",
            "",
            f"Selection passed: **{selection.get('passed', False)}**.",
            f"Decision: {selection.get('reason')}.",
            "",
            "A failed selection stops the program before the expensive confirmation by design.",
            "",
        ]
        (output / "PILOT_REPORT.md").write_text("\n".join(report))
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "format": PILOT_FORMAT,
                    "pilot_seal_id": seal["pilot_seal_id"],
                    "confirmation_authorized": bool(selection.get("passed")),
                    "selected_family": selection.get("selected_family"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print("[pilot 7/9] Sealing artifacts", flush=True)
        write_checksums(output)
    _campaign_status(work, "pilot", "sealed_complete")
    print("[pilot 8/9] Pilot complete", flush=True)
    print(f"[pilot 9/9] Results: {output_directory}", flush=True)


def verify_pilot(registration: Path, pilot: Path) -> dict[str, Any]:
    design = verify_design(registration)
    pilot = pilot.resolve()
    verify_checksums(pilot)
    seal = json.loads((pilot / "pilot_seal.json").read_text())
    identifier = seal.pop("pilot_seal_id")
    if seal.get("format") != PILOT_FORMAT or _canonical_digest(seal) != identifier:
        raise ValueError("invalid pilot seal")
    seal["pilot_seal_id"] = identifier
    if seal["design_registration_id"] != design["registration_id"]:
        raise ValueError("pilot references a different design")
    if seal["source_hashes"] != _source_hashes():
        raise ValueError("scientific source changed after pilot freeze")
    if seal["status"] != "model_frozen_for_confirmation":
        raise ValueError("pilot stopped; confirmation is not authorized")
    if not seal["selection"]["passed"] or not seal["selection"]["selected_family"]:
        raise ValueError("pilot has no eligible frozen predictor")
    load_prediction_models(pilot / "selected_models.pkl")
    return seal


def confirmation_metrics(
    cases: list[PredictionCase],
    labels: NDArray[np.int8],
    predictions: dict[str, dict[str, FloatArray]],
    development_power: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ids = _matrix_ids(cases)
    confirmation_power = _power(labels, cases)
    rows: list[dict[str, Any]] = []
    candidate_summaries: dict[str, Any] = {}
    for candidate in CANDIDATES:
        selected = _candidate_mask(cases, candidate)
        values = labels[selected]
        candidate_ids = ids[selected]
        q_all = values.mean(axis=1)
        q_a = values[:, : values.shape[1] // 2].mean(axis=1)
        q_b = values[:, values.shape[1] // 2 :].mean(axis=1)
        model_summaries: dict[str, Any] = {}
        for model_name, prediction in predictions[candidate].items():
            calibration = fit_linear(
                f"confirmation_calibration_{candidate}_{model_name}",
                "prediction_logit",
                logit(np.clip(prediction, 1e-8, 1.0 - 1e-8))[:, None],
                values.sum(axis=1).astype(np.float64),
                np.full(values.shape[0], values.shape[1], dtype=np.float64),
                ridge_lambda=0.0,
            )
            model_summaries[model_name] = {
                "pooled_log_loss": float(_state_log_loss(q_all, prediction).mean()),
                "pooled_q_brier": float(_state_brier(q_all, prediction).mean()),
                "calibration_intercept": calibration.intercept,
                "calibration_slope": float(calibration.coefficient[0]),
                "spearman_by_half": [
                    spearman(prediction, q_a),
                    spearman(prediction, q_b),
                ],
                "matrix_centered_spearman_by_half": [
                    centered_spearman(prediction, q_a, candidate_ids),
                    centered_spearman(prediction, q_b, candidate_ids),
                ],
            }
        candidate_summaries[candidate] = {"models": model_summaries}
        for half, q in (
            ("A", values[:, : values.shape[1] // 2].mean(axis=1)),
            ("B", values[:, values.shape[1] // 2 :].mean(axis=1)),
        ):
            baseline = predictions[candidate]["h10"]
            enhanced = predictions[candidate]["enhanced"]
            seed_parts = (candidate, half)
            gain, interval = _paired_gain(
                q,
                baseline,
                enhanced,
                candidate_ids,
                _state_log_loss,
                BOOTSTRAP_REPETITIONS,
                np.random.default_rng(
                    derive_seed(
                        BOOTSTRAP_MASTER_SEED, "confirmation.log_loss", *seed_parts
                    )
                ),
            )
            brier, brier_interval = _paired_gain(
                q,
                baseline,
                enhanced,
                candidate_ids,
                _state_brier,
                BOOTSTRAP_REPETITIONS,
                np.random.default_rng(
                    derive_seed(
                        BOOTSTRAP_MASTER_SEED, "confirmation.brier", *seed_parts
                    )
                ),
            )
            p_value = paired_matrix_randomization_p(
                q,
                baseline,
                enhanced,
                candidate_ids,
                RANDOMIZATION_REPETITIONS,
                np.random.default_rng(
                    derive_seed(
                        RANDOMIZATION_MASTER_SEED,
                        "confirmation.randomization",
                        *seed_parts,
                    )
                ),
            )
            rows.append(
                {
                    "candidate": candidate,
                    "half": half,
                    "log_loss_gain": gain,
                    "log_loss_gain_ci95": interval,
                    "q_brier_gain": brier,
                    "q_brier_gain_ci95": brier_interval,
                    "randomization_p_raw": p_value,
                }
            )
    adjusted = holm_adjust([row["randomization_p_raw"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["randomization_p_holm"] = value
        row["passes_statistical_gate"] = bool(
            row["log_loss_gain"] > 0.0
            and row["log_loss_gain_ci95"][0] > 0.0
            and value < 0.05
        )
    adequate = all(
        development_power[candidate]["adequate"]
        and confirmation_power[candidate]["adequate"]
        for candidate in CANDIDATES
    )
    return {
        "primary_tests": rows,
        "candidates": candidate_summaries,
        "development_power": development_power,
        "confirmation_power": confirmation_power,
        "power_adequate": adequate,
        "primary_prediction_supported": bool(
            adequate and rows and all(row["passes_statistical_gate"] for row in rows)
        ),
        "family_size": 4,
        "decision_rule": (
            "positive enhanced-over-h10 log-loss gain, matrix-bootstrap lower "
            "95% bound >0, and Holm-adjusted whole-matrix randomization p<0.05 "
            "in both candidates and both fixed branch halves"
        ),
    }


def _secondary_metrics(
    cases: list[PredictionCase],
    batches: list[PredictionBranchBatch],
    predictions: dict[str, dict[str, FloatArray]],
    post_break_models: dict[str, SequentialRidgeModel],
    post_break_audit: dict[str, Any],
    raw: PredictionRawFeatures,
    all_pilot_models: dict[str, dict[str, PredictionFamilyModel]],
) -> dict[str, Any]:
    output: dict[str, Any] = {"candidates": {}}
    labels = _labels(batches)
    margins = np.stack([batch.best_margins for batch in batches])
    for candidate in CANDIDATES:
        selected = _candidate_mask(cases, candidate)
        enhanced = predictions[candidate]["enhanced"]
        candidate_raw = raw.selected(selected)
        item: dict[str, Any] = {
            "rates": {
                name: float(labels[name][selected].mean())
                for name in ("break", "run8", "strict", "first5", "centroid")
            },
            "continuous_margin_correlation": {},
        }
        for index, name in enumerate(("strict", "first5", "centroid")):
            selected_margins = margins[selected, :, index]
            finite_counts = np.isfinite(selected_margins).sum(axis=1)
            mean_margin = np.divide(
                np.nansum(selected_margins, axis=1),
                finite_counts,
                out=np.full(selected_margins.shape[0], np.nan),
                where=finite_counts > 0,
            )
            finite = np.isfinite(mean_margin)
            item["continuous_margin_correlation"][name] = (
                float(np.corrcoef(enhanced[finite], mean_margin[finite])[0, 1])
                if finite.sum() >= 3
                else float("nan")
            )
        hurdle = all_pilot_models["hurdle"][candidate].enhanced
        break_probability = hurdle.break_model.predict(candidate_raw)
        run_probability = hurdle.run8_model.predict(candidate_raw)
        conditional_probability = hurdle.strict_model.predict(candidate_raw)
        stage_scores: dict[str, Any] = {}
        for name, numerator, denominator, probability in (
            (
                "break",
                labels["break"][selected].sum(axis=1),
                np.full(int(selected.sum()), BRANCHES),
                break_probability,
            ),
            (
                "run8_given_break",
                labels["run8"][selected].sum(axis=1),
                labels["break"][selected].sum(axis=1),
                run_probability,
            ),
            (
                "strict_given_run8",
                labels["strict"][selected].sum(axis=1),
                labels["run8"][selected].sum(axis=1),
                conditional_probability,
            ),
        ):
            eligible = denominator > 0
            q = numerator[eligible] / denominator[eligible]
            losses = _state_log_loss(q, probability[eligible])
            stage_scores[name] = {
                "eligible_states": int(eligible.sum()),
                "weighted_log_loss": float(
                    np.average(losses, weights=denominator[eligible])
                ),
                "mean_prediction": float(probability[eligible].mean()),
            }
        item["hurdle_prediction"] = {
            "components": stage_scores,
            "mean_product_probability": float(
                np.mean(break_probability * run_probability * conditional_probability)
            ),
            "secondary_no_rescue": True,
        }
        auxiliary = all_pilot_models["auxiliary_stack"][candidate].enhanced
        first5_prediction = auxiliary.first5_model.predict(candidate_raw)
        centroid_prediction = auxiliary.centroid_model.predict(candidate_raw)
        item["auxiliary_prediction"] = {
            "first5_log_loss": float(
                _state_log_loss(
                    labels["first5"][selected].mean(axis=1), first5_prediction
                ).mean()
            ),
            "centroid_log_loss": float(
                _state_log_loss(
                    labels["centroid"][selected].mean(axis=1), centroid_prediction
                ).mean()
            ),
            "secondary_no_rescue": True,
        }
        x_parts: list[FloatArray] = []
        y_parts: list[NDArray[np.int8]] = []
        id_parts: list[NDArray[np.int64]] = []
        for case, batch in zip(cases, batches):
            if case.candidate != candidate:
                continue
            broken = batch.stages[:, 0].astype(bool)
            if broken.any():
                x_parts.append(batch.post_break_features[broken])
                y_parts.append(batch.targets[broken, 0])
                id_parts.append(
                    np.full(int(broken.sum()), case.matrix_id, dtype=np.int64)
                )
        x = np.vstack(x_parts)
        y = np.concatenate(y_parts).astype(np.float64)
        break_ids = np.concatenate(id_parts)
        predicted = post_break_models[candidate].predict({"postbreak": x})
        pilot_item = post_break_audit[candidate]
        pilot_probability = pilot_item["strict_events_after_break"] / max(
            pilot_item["break_futures"], 1
        )
        base = np.full(y.size, np.clip(pilot_probability, 1e-8, 1.0 - 1e-8))
        item["post_break"] = {
            "futures": int(y.size),
            "events": int(y.sum()),
            "descriptive_log_loss_gain_over_frozen_pilot_constant": float(
                _matrix_values(
                    _state_log_loss(y, base) - _state_log_loss(y, predicted), break_ids
                ).mean()
            ),
            "not_confirmatory": True,
        }
        output["candidates"][candidate] = item
    return output


def run_confirmation(
    registration: Path,
    pilot: Path,
    output_directory: Path,
    workers: int,
    work_directory: Path | None = None,
) -> None:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    seal = verify_pilot(registration, pilot)
    models = load_prediction_models(pilot.resolve() / "selected_models.pkl")
    with (pilot.resolve() / "post_break_models.pkl").open("rb") as handle:
        post_break_models = pickle.load(handle)
    with (pilot.resolve() / "all_pilot_models.pkl").open("rb") as handle:
        all_pilot_models = pickle.load(handle)
    post_break_audit = json.loads(
        (pilot.resolve() / "post_break_audit.json").read_text()
    )
    experiment = _experiment(CONFIRMATION_MASTER_SEED, CONFIRMATION_MATRICES)
    work = (
        work_directory.resolve()
        if work_directory is not None
        else output_directory.with_name(f".{output_directory.name}.work")
    )
    _campaign_status(work, "confirmation", "building_trajectories_and_features")
    print("[confirm 1/9] Generating untouched trajectories and features", flush=True)
    with threadpool_limits(limits=1):
        cases = build_prediction_cohort(
            experiment, "REGPREDCONF", experiment.confirmation
        )
        raw = extract_prediction_features(cases, experiment)
    print("[confirm 2/9] Shooting 256,000 F32 futures", flush=True)
    _campaign_status(work, "confirmation", "shooting_futures")
    batches = run_prediction_branches(
        cases,
        experiment,
        BRANCHES,
        workers,
        "confirm",
        checkpoint_directory=work / "generate",
    )
    print("[confirm 3/9] Replaying every confirmation future", flush=True)
    _campaign_status(work, "confirmation", "replaying_futures")
    replay_batches = run_prediction_branches(
        cases,
        experiment,
        BRANCHES,
        workers,
        "confirm-replay",
        checkpoint_directory=work / "replay",
    )
    replay = replay_audit(batches, replay_batches)
    labels = _labels(batches)
    predictions: dict[str, dict[str, FloatArray]] = {}
    for candidate in CANDIDATES:
        selected = _candidate_mask(cases, candidate)
        predictions[candidate] = models[candidate].predict(raw.selected(selected))
    primary = confirmation_metrics(cases, labels["strict"], predictions, seal["power"])
    _campaign_status(work, "confirmation", "computing_inference")
    secondary = _secondary_metrics(
        cases,
        batches,
        predictions,
        post_break_models,
        post_break_audit,
        raw,
        all_pilot_models,
    )
    with _atomic_destination(output_directory) as output:
        print("[confirm 4/9] Writing readback-complete artifacts", flush=True)
        _write_branch_tables(output, "confirmation", cases, batches)
        _save_arrays(output / "confirmation_arrays.npz", cases, raw, batches)
        rows: list[dict[str, Any]] = []
        for index, case in enumerate(cases):
            selected_indices = np.flatnonzero(_candidate_mask(cases, case.candidate))
            local = int(np.flatnonzero(selected_indices == index)[0])
            rows.append(
                {
                    "state_id": case.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "landmark": case.landmark,
                    "q_strict_all": float(labels["strict"][index].mean()),
                    "q_strict_A": float(
                        labels["strict"][index, : BRANCHES // 2].mean()
                    ),
                    "q_strict_B": float(
                        labels["strict"][index, BRANCHES // 2 :].mean()
                    ),
                    "prediction_h10": float(predictions[case.candidate]["h10"][local]),
                    "prediction_enhanced": float(
                        predictions[case.candidate]["enhanced"][local]
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(
            output / "confirmation_states.csv", index=False, float_format="%.17g"
        )
        for name, value in (
            ("primary_metrics.json", primary),
            ("secondary_metrics.json", secondary),
            ("replay_audit.json", replay),
        ):
            (output / name).write_text(
                json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n"
            )
        # Recompute the primary decision from round-trip CSV values.
        restored = pd.read_csv(
            output / "confirmation_states.csv",
            dtype={"candidate": str},
            float_precision="round_trip",
        )
        restored["candidate"] = restored["candidate"].str.zfill(2)
        restored_predictions = {
            candidate: {
                "h10": restored.loc[
                    restored["candidate"] == candidate, "prediction_h10"
                ].to_numpy(),
                "enhanced": restored.loc[
                    restored["candidate"] == candidate, "prediction_enhanced"
                ].to_numpy(),
            }
            for candidate in CANDIDATES
        }
        restored_labels = _strict_labels_from_branch_table(
            output / "confirmation_branches.csv.gz", cases
        )
        if not np.array_equal(restored_labels, labels["strict"]):
            raise ValueError("round-trip confirmation label readback mismatch")
        readback = confirmation_metrics(
            cases, restored_labels, restored_predictions, seal["power"]
        )
        if _json_ready(readback) != _json_ready(primary):
            raise ValueError("round-trip confirmation metric readback mismatch")
        (output / "readback_audit.json").write_text(
            json.dumps(
                {
                    "float_precision": "round_trip",
                    "branch_labels_exact": True,
                    "primary_metrics_exact": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print("[confirm 5/9] Writing scientific report", flush=True)
        lines = [
            "# Strict-regime prediction confirmation",
            "",
            f"Frozen family: **{seal['selection']['selected_family']}**.",
            f"Primary prediction supported: **{primary['primary_prediction_supported']}**.",
            "",
            "The primary claim requires all four candidate-by-half tests. First-five, centroid, continuous-margin, hurdle, and post-break analyses cannot rescue it.",
            "",
            "This experiment tests prediction, not causal control, attractor switching, biological memory, or prebiotic realism.",
            "",
        ]
        (output / "CONFIRMATION_REPORT.md").write_text("\n".join(lines))
        manifest = {
            "format": CONFIRMATION_FORMAT,
            "pilot_seal_id": seal["pilot_seal_id"],
            "selected_family": seal["selection"]["selected_family"],
            "primary_prediction_supported": primary["primary_prediction_supported"],
            "replay_exact": replay["digests_exact"],
            "claim_boundary": "prediction, not control or regime switching",
            "checkpoint_audit": {
                "work_directory": str(work),
                "generation_contract_digest": sha256_file(
                    work / "generate" / "checkpoint_contract.json"
                ),
                "replay_contract_digest": sha256_file(
                    work / "replay" / "checkpoint_contract.json"
                ),
                "resumable_per_state": True,
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print("[confirm 6/9] Sealing confirmation", flush=True)
        write_checksums(output)
    _campaign_status(work, "confirmation", "sealed_complete")
    print("[confirm 7/9] Confirmation complete", flush=True)
    print("[confirm 8/9] Existing sealed cohorts unchanged", flush=True)
    print(f"[confirm 9/9] Results: {output_directory}", flush=True)


def run_smoke(output_directory: Path, workers: int) -> None:
    """Exercise new cohort, features, endpoint archive, replay, and I/O cheaply."""

    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    cohort = CohortConfig(matrices=2, branches_per_state=2, landmarks=(20,))
    experiment = ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=12,
        bootstrap_repetitions=8,
        permutation_repetitions=8,
        master_seed=derive_seed(PILOT_MASTER_SEED, "smoke").to_bytes(32, "big").hex(),
    )
    cases = build_prediction_cohort(experiment, "REGPREDSMOKE", cohort)
    raw = extract_prediction_features(cases, experiment)
    batches = run_prediction_branches(cases, experiment, 2, workers, "smoke")
    replay = run_prediction_branches(cases, experiment, 2, workers, "smoke-replay")
    audit = replay_audit(batches, replay)
    with _atomic_destination(output_directory) as output:
        _write_branch_tables(output, "smoke", cases, batches)
        _save_arrays(output / "smoke_arrays.npz", cases, raw, batches)
        (output / "replay_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "plastic-heredity-regime-prediction-smoke-v1",
                    "states": len(cases),
                    "futures": len(cases) * 2,
                    "scientific_result": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        write_checksums(output)


def read_checkpoint_status(work_directory: Path) -> dict[str, Any]:
    """Return checkpoint progress without changing the running campaign."""

    work = work_directory.resolve()
    if not work.is_dir():
        raise FileNotFoundError(f"checkpoint work directory not found: {work}")
    output: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "work_directory": str(work),
        "campaign": None,
        "stages": {},
    }
    campaign_path = work / "campaign_status.json"
    if campaign_path.is_file():
        output["campaign"] = json.loads(campaign_path.read_text(encoding="utf-8"))
    for stage in ("generate", "replay"):
        status_path = work / stage / "status.json"
        if status_path.is_file():
            output["stages"][stage] = json.loads(
                status_path.read_text(encoding="utf-8")
            )
    if output["campaign"] is None and not output["stages"]:
        raise ValueError(f"no checkpoint status found in {work}")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict-regime prediction program")
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostic = commands.add_parser("diagnose")
    diagnostic.add_argument(
        "--source", type=Path, default=Path("results/regime_confirmation")
    )
    diagnostic.add_argument(
        "--output", type=Path, default=Path("results/regime_prediction_diagnostic")
    )
    diagnostic.add_argument("--skip-reconstruction", action="store_true")
    register = commands.add_parser("register-design")
    register.add_argument(
        "--output", type=Path, default=Path("results/regime_prediction_registration")
    )
    pilot = commands.add_parser("pilot")
    pilot.add_argument(
        "--registration",
        type=Path,
        default=Path("results/regime_prediction_registration"),
    )
    pilot.add_argument(
        "--output", type=Path, default=Path("results/regime_prediction_pilot")
    )
    pilot.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    pilot.add_argument("--work-dir", type=Path, default=None)
    confirm = commands.add_parser("confirm")
    confirm.add_argument(
        "--registration",
        type=Path,
        default=Path("results/regime_prediction_registration"),
    )
    confirm.add_argument(
        "--pilot", type=Path, default=Path("results/regime_prediction_pilot")
    )
    confirm.add_argument(
        "--output", type=Path, default=Path("results/regime_prediction_confirmation")
    )
    confirm.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    confirm.add_argument("--work-dir", type=Path, default=None)
    smoke = commands.add_parser("smoke")
    smoke.add_argument(
        "--output", type=Path, default=Path("results/regime_prediction_smoke")
    )
    smoke.add_argument("--workers", type=int, default=1)
    status = commands.add_parser("status")
    status.add_argument("--work-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "diagnose":
        run_diagnostic(
            arguments.source,
            arguments.output,
            reconstruct=not arguments.skip_reconstruction,
        )
    elif arguments.command == "register-design":
        register_design(arguments.output)
    elif arguments.command == "pilot":
        if arguments.workers < 1:
            raise ValueError("workers must be positive")
        run_pilot(
            arguments.registration,
            arguments.output,
            arguments.workers,
            arguments.work_dir,
        )
    elif arguments.command == "confirm":
        if arguments.workers < 1:
            raise ValueError("workers must be positive")
        run_confirmation(
            arguments.registration,
            arguments.pilot,
            arguments.output,
            arguments.workers,
            arguments.work_dir,
        )
    elif arguments.command == "smoke":
        if arguments.workers < 1:
            raise ValueError("workers must be positive")
        run_smoke(arguments.output, arguments.workers)
    elif arguments.command == "status":
        print(
            json.dumps(
                read_checkpoint_status(arguments.work_dir), indent=2, sort_keys=True
            )
        )
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
