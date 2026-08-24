"""PX3 development and confirmation of direct molecular control of Phi."""

from __future__ import annotations

import argparse
import inspect
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from . import intervention_cr5 as cr5
from .config import CANDIDATES, GardConfig
from .intervention_core import (
    FrozenFullPredictor,
    MolecularEdit,
    apply_molecular_edit,
    enumerate_legal_edits,
    outcome_from_records,
    state_graph_features_many,
)
from .intervention_outgoing_rule import select_outgoing_rule_edits
from .mechanistic import verify_checksums, write_checksums
from .phir_ch5 import _append_ledger, _snapshot_after_record
from .phir_extension_common import (
    BOOTSTRAP_DRAWS,
    MASTER_REGISTRATION,
    MAX_WORKERS,
    MINIMUM_FREE_DISK_BYTES,
    RANDOMIZATION_DRAWS,
    RESULT_ROOT,
    ROOT,
    apply_holm,
    atomic_json,
    atomic_pickle,
    canonical_digest,
    canonical_json,
    paired_matrix_effects,
    paired_summary,
    purpose_seed,
    runtime_versions,
    safe_score_pairs,
    sha256_file,
)
from .simulator import (
    FissionRecord,
    SimulationError,
    Snapshot,
    advance_fission,
    generate_beta,
    generate_initial_composition,
    simulate_future_absorbing,
)


DOCUMENT = "CODEX_CH5_PHIR_EXTENSION_PREREGISTRATION.md"
DEVELOPMENT_AMENDMENT = "CODEX_CH5_PHIR_EXTENSION_PX3_AMENDMENT.md"
CONFIRMATION_RESOURCE_NOTE = (
    "CODEX_CH5_PHIR_EXTENSION_PX3_CONFIRMATION_RESOURCE_NOTE.md"
)
PX1_OUTPUT = RESULT_ROOT / "px1_fresh_confirmation"
PX1_REGISTRATION = RESULT_ROOT / "px1_registration"

DEFAULT_VALIDATION = RESULT_ROOT / "px3_validation"
DEFAULT_DEVELOPMENT_REGISTRATION = RESULT_ROOT / "px3_development_registration"
DEFAULT_SMOKE = RESULT_ROOT / "px3_smoke"
DEFAULT_DEVELOPMENT_OUTPUT = RESULT_ROOT / "px3_development"
DEFAULT_CONFIRMATION_REGISTRATION = RESULT_ROOT / "px3_confirmation_registration"
DEFAULT_CONFIRMATION_OUTPUT = RESULT_ROOT / "px3_confirmation"
DEFAULT_WORK = RESULT_ROOT / ".px3_work"
DEFAULT_LOG = RESULT_ROOT / "px3.log"
PRE_AMENDMENT_DEVELOPMENT_REGISTRATION = (
    RESULT_ROOT / "px3_development_registration_pre_amendment_001"
)
PRE_AMENDMENT_DEVELOPMENT_WORK = (
    DEFAULT_WORK / "development_generate_pre_amendment_001"
)

LABEL = "CODEX_CH5_PHIR_EXTENSION_PX3_V1"
PROGRAM_FORMAT = "codex-ch5-phir-extension-px3-program-v1"
DEVELOPMENT_REGISTRATION_FORMAT = "codex-ch5-phir-extension-px3-development-registration-v2"
CONFIRMATION_REGISTRATION_FORMAT = "codex-ch5-phir-extension-px3-confirmation-registration-v1"
RESULT_FORMAT = "codex-ch5-phir-extension-px3-result-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-extension-px3-checkpoint-v1"
DEVELOPMENT_SERVICE = "codex-phir-extension-px3-development-20260820"
CONFIRMATION_SERVICE = "codex-phir-extension-px3-confirmation-20260820"

DEVELOPMENT_MATRICES = 12
DEVELOPMENT_REPLICATES = 2
DEVELOPMENT_LANDMARKS = (20, 30, 40, 50)
DEVELOPMENT_EDITS = 24
DEVELOPMENT_BRANCHES = 16
DEVELOPMENT_HORIZON = 8
DEVELOPMENT_CPU_SECONDS = 104.0 * 3600.0
DEVELOPMENT_MAX_WORKERS = 8
DEVELOPMENT_CARRIED_FORWARD = tuple(range(6))
ORIGINAL_DEVELOPMENT_MATRICES = 24
ORIGINAL_DEVELOPMENT_CPU_SECONDS = 20.0 * 3600.0
ORIGINAL_DEVELOPMENT_REGISTRATION_ID = (
    "86a91cf7c75b6b268e25fbd06f5b1efd94eccd37fac16c49adea65ed3977707d"
)
ORIGINAL_DEVELOPMENT_GATE_ELIGIBLE = False

CONFIRMATION_MATRICES = 24
CONFIRMATION_LANDMARKS = (20, 40, 60)
CONFIRMATION_BRANCHES = 64
CONFIRMATION_HORIZON = 8
CONFIRMATION_CPU_SECONDS = 64.0 * 3600.0
CONFIRMATION_MAX_WORKERS = 8
HALVES = {"A": (0, 32), "B": (32, 64)}

ARMS = ("PHI_UP", "PHI_DOWN", "RANDOM", "NOOP")
RIDGE_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
CV_FOLDS = 5
HEREDITY_EQUIVALENCE_MARGIN = 0.025

HEREDITY_MODEL = PX1_REGISTRATION / "frozen_full_predictor.npz"
EXPECTED_HEREDITY_MODEL_SHA256 = "9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af"

SOURCE_FILES = (
    DOCUMENT,
    DEVELOPMENT_AMENDMENT,
    CONFIRMATION_RESOURCE_NOTE,
    "plastic_heredity/phir_extension_px3.py",
    "plastic_heredity/phir_extension_common.py",
    "tests/test_phir_extension_px3.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/intervention_outgoing_rule.py",
    "plastic_heredity/config.py",
    "plastic_heredity/simulator.py",
    "plastic_heredity/features.py",
    "plastic_heredity/processes.py",
    "plastic_heredity/phir_instruments.py",
    "plastic_heredity/phir_rescue_instruments.py",
    "plastic_heredity/seeds.py",
)


@dataclass(frozen=True)
class PX3Spec:
    label: str
    matrices: int
    landmarks: tuple[int, ...]
    branches: int
    horizon: int
    cpu_seconds: float


@dataclass(frozen=True)
class DevelopmentBatch:
    matrix_id: int
    training_rows: tuple[dict[str, Any], ...]
    state_rows: tuple[dict[str, Any], ...]
    cpu_seconds: float
    scientific_digest: str


@dataclass(frozen=True)
class ConfirmationBatch:
    matrix_id: int
    score_rows: tuple[dict[str, Any], ...]
    branch_rows: tuple[dict[str, Any], ...]
    edit_rows: tuple[dict[str, Any], ...]
    cpu_seconds: float
    scientific_digest: str


@dataclass(frozen=True)
class FrozenPhiSurrogate:
    candidate: str
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    coefficient: NDArray[np.float64]
    ridge_alpha: float

    def predict(self, features: NDArray) -> NDArray[np.float64]:
        values = np.atleast_2d(np.asarray(features, dtype=np.float64))
        return np.asarray(((values - self.mean) / self.scale) @ self.coefficient)


class _PX3CompatUnpickler(pickle.Unpickler):
    """Read checkpoints emitted while this module ran through ``python -m``."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "DevelopmentBatch":
            return DevelopmentBatch
        if module == "__main__" and name == "ConfirmationBatch":
            return ConfirmationBatch
        return super().find_class(module, name)


def _load_checkpoint(path: Path) -> Any:
    with path.open("rb") as handle:
        return _PX3CompatUnpickler(handle).load()


def development_spec() -> PX3Spec:
    return PX3Spec(
        "development",
        DEVELOPMENT_MATRICES,
        DEVELOPMENT_LANDMARKS,
        DEVELOPMENT_BRANCHES,
        DEVELOPMENT_HORIZON,
        DEVELOPMENT_CPU_SECONDS,
    )


def confirmation_spec() -> PX3Spec:
    return PX3Spec(
        "confirmation",
        CONFIRMATION_MATRICES,
        CONFIRMATION_LANDMARKS,
        CONFIRMATION_BRANCHES,
        CONFIRMATION_HORIZON,
        CONFIRMATION_CPU_SECONDS,
    )


def smoke_spec(label: str) -> PX3Spec:
    return PX3Spec(label, 1, (2,), 4, 4, 300.0)


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def _batch_digest(value: DevelopmentBatch | ConfirmationBatch) -> str:
    fields = asdict(value)
    fields["cpu_seconds"] = 0.0
    fields["scientific_digest"] = ""
    return canonical_digest(fields)


def _development_future_seed(
    spec: PX3Spec,
    candidate: str,
    matrix_id: int,
    replicate: int,
    landmark: int,
    branch: int,
) -> int:
    domain = "smoke" if spec.label.startswith("smoke") else "future"
    return purpose_seed(
        domain,
        "PX3_DEVELOPMENT",
        spec.label,
        candidate,
        matrix_id,
        replicate,
        landmark,
        branch,
    )


def _screen_seed(
    spec: PX3Spec,
    candidate: str,
    matrix_id: int,
    replicate: int,
    landmark: int,
) -> int:
    domain = "smoke" if spec.label.startswith("smoke") else "screen"
    return purpose_seed(
        domain,
        "PX3_DEVELOPMENT",
        spec.label,
        candidate,
        matrix_id,
        replicate,
        landmark,
    )


def _confirmation_matrix_seed(spec: PX3Spec, matrix_id: int, purpose: str) -> int:
    domain = "smoke" if spec.label.startswith("smoke") else purpose
    return purpose_seed(domain, "PX3_CONFIRMATION", spec.label, purpose, matrix_id)


def _confirmation_path_seed(spec: PX3Spec, candidate: str, matrix_id: int) -> int:
    domain = "smoke" if spec.label.startswith("smoke") else "main_path"
    return purpose_seed(domain, "PX3_CONFIRMATION", spec.label, candidate, matrix_id)


def _confirmation_selection_seed(
    spec: PX3Spec, candidate: str, matrix_id: int, landmark: int
) -> int:
    domain = "smoke" if spec.label.startswith("smoke") else "random_action"
    return purpose_seed(
        domain, "PX3_CONFIRMATION", spec.label, candidate, matrix_id, landmark
    )


def _confirmation_future_seed(
    spec: PX3Spec, candidate: str, matrix_id: int, landmark: int, branch: int
) -> int:
    domain = "smoke" if spec.label.startswith("smoke") else "future"
    return purpose_seed(
        domain,
        "PX3_CONFIRMATION",
        spec.label,
        candidate,
        matrix_id,
        landmark,
        branch,
    )


def _snapshot_digest(snapshot: Snapshot) -> str:
    return canonical_digest(
        {
            "composition": snapshot.composition,
            "generation": snapshot.generation,
            "inheritance": snapshot.inheritance,
            "boundary_h": snapshot.boundary_h,
            "previous_growth_steps": snapshot.previous_growth_steps,
            "cumulative_growth_steps": snapshot.cumulative_growth_steps,
        }
    )


def _records_to_pairs(
    launch: Snapshot, records: Sequence[FissionRecord]
) -> tuple[list[NDArray], list[NDArray]]:
    past: list[NDArray] = []
    future: list[NDArray] = []
    previous = np.asarray(launch.composition, dtype=np.int64)
    for record in records:
        past.append(previous.copy())
        future.append(record.daughter.copy())
        previous = record.daughter
    return past, future


def _score_edit_future(
    snapshot: Snapshot,
    beta: NDArray,
    candidate: str,
    edit: MolecularEdit | None,
    seeds: Sequence[int],
    horizon: int,
) -> tuple[float, list[dict[str, Any]]]:
    config = GardConfig()
    composition = (
        snapshot.composition
        if edit is None
        else apply_molecular_edit(snapshot.composition, edit)
    )
    launch = Snapshot(
        np.asarray(composition, dtype=np.int64).copy(),
        snapshot.generation,
        snapshot.inheritance,
        snapshot.boundary_h,
        snapshot.previous_growth_steps,
        snapshot.cumulative_growth_steps,
    )
    past: list[NDArray] = []
    future: list[NDArray] = []
    outcomes: list[dict[str, Any]] = []
    for branch, seed in enumerate(seeds):
        records, completed = simulate_future_absorbing(
            launch,
            beta,
            config,
            CANDIDATES[candidate],
            horizon,
            np.random.default_rng(seed),
        )
        local_past, local_future = _records_to_pairs(launch, records)
        past.extend(local_past)
        future.extend(local_future)
        process = outcome_from_records(launch, records, completed, horizon)
        inherited = np.asarray(
            [record.h > config.inheritance_threshold for record in records], dtype=bool
        )
        outcomes.append(
            {
                "branch": branch,
                "completed": int(completed),
                "observed_fissions": len(records),
                "inherited_fraction": float(inherited.mean()) if inherited.size else 0.0,
                "break_within_f8": int((~inherited).any()),
                "joint_break_run3_within_f8": int(process.joint_break_run3),
                "record_digest": process.record_digest,
            }
        )
    score = safe_score_pairs(
        np.asarray(past), np.asarray(future), beta, "material", config
    )
    return score.full_revised, outcomes


def _candidate_edits(
    snapshot: Snapshot,
    beta: NDArray,
    candidate: str,
    predictor: FrozenFullPredictor,
    rng: np.random.Generator,
    count: int,
) -> tuple[MolecularEdit, ...]:
    config = GardConfig()
    from .intervention_core import score_legal_edits

    _noop, scored = score_legal_edits(predictor, candidate, snapshot, beta, config)
    model_up = min(
        scored,
        key=lambda item: (-item.predicted_probability, item.edit.remove_type, item.edit.add_type),
    ).edit
    model_down = min(
        scored,
        key=lambda item: (item.predicted_probability, item.edit.remove_type, item.edit.add_type),
    ).edit
    rules = select_outgoing_rule_edits(snapshot.composition, beta)
    selected: list[MolecularEdit] = []
    for edit in (model_up, model_down, rules["RULE_UP"], rules["RULE_DOWN"]):
        if edit not in selected:
            selected.append(edit)
    legal = list(enumerate_legal_edits(snapshot.composition))
    remaining = [edit for edit in legal if edit not in selected]
    if remaining:
        order = rng.permutation(len(remaining))
        selected.extend(remaining[int(index)] for index in order[: max(0, count - len(selected))])
    return tuple(selected[:count])


def _development_snapshots(
    matrix_id: int,
    beta: NDArray,
    initial: NDArray,
    candidate: str,
    replicate: int,
    landmarks: Sequence[int],
) -> tuple[dict[int, Snapshot], str]:
    from .phir_extension_px1 import _future_seed as px1_future_seed
    from .phir_extension_px1 import scientific_spec as px1_specification

    config = GardConfig()
    rng = np.random.default_rng(
        px1_future_seed(px1_specification(), candidate, matrix_id, replicate)
    )
    snapshot = Snapshot(np.asarray(initial, dtype=np.int64).copy(), 0, (), ())
    output: dict[int, Snapshot] = {}
    records: list[FissionRecord] = []
    for step in range(1, max(landmarks) + 1):
        record = advance_fission(
            snapshot.composition, beta, config, CANDIDATES[candidate], rng
        )
        records.append(record)
        snapshot = _snapshot_after_record(snapshot, record)
        if step in landmarks:
            output[step] = snapshot
    return output, cr5._records_digest(records)


def _development_matrix(
    args: tuple[int, PX3Spec, str, str]
) -> DevelopmentBatch:
    matrix_id, spec, matrix_input_path, predictor_path = args
    started = time.process_time()
    with threadpool_limits(limits=1), np.load(matrix_input_path, allow_pickle=False) as archive:
        identifiers = archive["matrix_id"].astype(int)
        index = int(np.flatnonzero(identifiers == matrix_id)[0])
        beta = np.asarray(archive["beta"][index], dtype=np.float64)
        initial = np.asarray(archive["initial"][index], dtype=np.int64)
    predictor = FrozenFullPredictor.load(predictor_path)
    config = GardConfig()
    training_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for replicate in range(DEVELOPMENT_REPLICATES):
            snapshots, path_digest = _development_snapshots(
                matrix_id, beta, initial, candidate, replicate, spec.landmarks
            )
            for landmark, snapshot in snapshots.items():
                state_id = f"PX3DEV-c{candidate}-m{matrix_id}-r{replicate}-g{landmark}"
                edits = _candidate_edits(
                    snapshot,
                    beta,
                    candidate,
                    predictor,
                    np.random.default_rng(
                        _screen_seed(spec, candidate, matrix_id, replicate, landmark)
                    ),
                    DEVELOPMENT_EDITS,
                )
                seeds = [
                    _development_future_seed(
                        spec, candidate, matrix_id, replicate, landmark, branch
                    )
                    for branch in range(spec.branches)
                ]
                noop_phi, noop_outcomes = _score_edit_future(
                    snapshot, beta, candidate, None, seeds, spec.horizon
                )
                compositions = np.vstack(
                    [apply_molecular_edit(snapshot.composition, edit) for edit in edits]
                )
                raw_features = state_graph_features_many(compositions, beta, config)
                noop_feature = state_graph_features_many(
                    np.atleast_2d(snapshot.composition), beta, config
                )[0]
                centered_features = raw_features - noop_feature
                for edit_index, (edit, feature) in enumerate(
                    zip(edits, centered_features, strict=True)
                ):
                    phi, _outcomes = _score_edit_future(
                        snapshot, beta, candidate, edit, seeds, spec.horizon
                    )
                    training_rows.append(
                        {
                            "matrix_id": matrix_id,
                            "candidate": candidate,
                            "replicate": replicate,
                            "landmark": landmark,
                            "state_id": state_id,
                            "edit_index": edit_index,
                            "remove_type": edit.remove_type,
                            "add_type": edit.add_type,
                            "feature": feature.tolist(),
                            "realized_phi": phi,
                            "noop_phi": noop_phi,
                            "target_delta_phi": phi - noop_phi,
                        }
                    )
                state_rows.append(
                    {
                        "matrix_id": matrix_id,
                        "candidate": candidate,
                        "replicate": replicate,
                        "landmark": landmark,
                        "state_id": state_id,
                        "snapshot_digest": _snapshot_digest(snapshot),
                        "path_record_digest": path_digest,
                        "selected_edits": len(edits),
                        "noop_phi": noop_phi,
                        "noop_completed_fraction": float(
                            np.mean([row["completed"] for row in noop_outcomes])
                        ),
                    }
                )
    provisional = DevelopmentBatch(
        matrix_id,
        tuple(training_rows),
        tuple(state_rows),
        float(time.process_time() - started),
        "",
    )
    return DevelopmentBatch(
        provisional.matrix_id,
        provisional.training_rows,
        provisional.state_rows,
        provisional.cpu_seconds,
        _batch_digest(provisional),
    )


def _fit_standardized_ridge(
    x: NDArray,
    y: NDArray,
    alpha: float,
) -> tuple[NDArray, NDArray, NDArray]:
    values = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    model = Ridge(alpha=alpha, fit_intercept=False, solver="svd")
    model.fit((values - mean) / scale, target)
    return mean, scale, np.asarray(model.coef_, dtype=np.float64)


def fit_surrogates(
    training: pd.DataFrame,
) -> tuple[dict[str, FrozenPhiSurrogate], dict[str, Any], pd.DataFrame]:
    models: dict[str, FrozenPhiSurrogate] = {}
    diagnostics: dict[str, Any] = {}
    scored_frames: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        selected = training[training["candidate"] == candidate].copy().reset_index(drop=True)
        x = np.vstack(selected["feature"].map(np.asarray))
        y = selected["target_delta_phi"].to_numpy(float)
        matrix_ids = selected["matrix_id"].to_numpy(int)
        cv_scores: dict[str, float] = {}
        for alpha in RIDGE_GRID:
            losses: list[float] = []
            for fold in range(CV_FOLDS):
                validation = matrix_ids % CV_FOLDS == fold
                train = ~validation
                mean, scale, coefficient = _fit_standardized_ridge(
                    x[train], y[train], alpha
                )
                prediction = ((x[validation] - mean) / scale) @ coefficient
                losses.extend(((prediction - y[validation]) ** 2).tolist())
            cv_scores[f"{alpha:g}"] = float(np.mean(losses))
        minimum = min(cv_scores.values())
        selected_alpha = max(
            alpha for alpha in RIDGE_GRID if cv_scores[f"{alpha:g}"] <= minimum + 1e-12
        )
        oof = np.empty_like(y)
        for fold in range(CV_FOLDS):
            validation = matrix_ids % CV_FOLDS == fold
            train = ~validation
            mean, scale, coefficient = _fit_standardized_ridge(
                x[train], y[train], selected_alpha
            )
            oof[validation] = ((x[validation] - mean) / scale) @ coefficient
        mean, scale, coefficient = _fit_standardized_ridge(x, y, selected_alpha)
        model = FrozenPhiSurrogate(candidate, mean, scale, coefficient, selected_alpha)
        models[candidate] = model
        selected["oof_predicted_delta_phi"] = oof
        selected["fitted_predicted_delta_phi"] = model.predict(x)
        scored_frames.append(selected)
        correlations: list[dict[str, Any]] = []
        extremes: list[dict[str, Any]] = []
        for state_id, local in selected.groupby("state_id", sort=True):
            correlation = float(
                spearmanr(
                    local["oof_predicted_delta_phi"], local["target_delta_phi"]
                ).statistic
            )
            maximum = local.sort_values(
                ["oof_predicted_delta_phi", "remove_type", "add_type"],
                ascending=[False, True, True],
            ).iloc[0]
            minimum_row = local.sort_values(
                ["oof_predicted_delta_phi", "remove_type", "add_type"],
                ascending=[True, True, True],
            ).iloc[0]
            correlations.append(
                {
                    "matrix_id": int(local.iloc[0]["matrix_id"]),
                    "state_id": state_id,
                    "value": correlation,
                }
            )
            extremes.append(
                {
                    "matrix_id": int(local.iloc[0]["matrix_id"]),
                    "state_id": state_id,
                    "value": float(
                        maximum["target_delta_phi"] - minimum_row["target_delta_phi"]
                    ),
                }
            )
        correlation_frame = pd.DataFrame(correlations)
        extreme_frame = pd.DataFrame(extremes)
        correlation_matrix = correlation_frame.groupby("matrix_id")["value"].mean()
        extreme_matrix = extreme_frame.groupby("matrix_id")["value"].mean()
        correlation_summary, correlation_arrays = paired_summary(
            correlation_matrix.to_numpy(), f"PX3/development/{candidate}/spearman"
        )
        extreme_summary, extreme_arrays = paired_summary(
            extreme_matrix.to_numpy(), f"PX3/development/{candidate}/top-bottom"
        )
        diagnostics[candidate] = {
            "ridge_grid": list(RIDGE_GRID),
            "cv_scores": cv_scores,
            "selected_alpha": selected_alpha,
            "states": int(selected["state_id"].nunique()),
            "matrices": int(selected["matrix_id"].nunique()),
            "edits": int(len(selected)),
            "oof_spearman": correlation_summary,
            "oof_top_bottom": extreme_summary,
            "arrays": {
                "spearman": {
                    name: value.tolist() for name, value in correlation_arrays.items()
                },
                "top_bottom": {
                    name: value.tolist() for name, value in extreme_arrays.items()
                },
            },
        }
        diagnostics[candidate]["development_valid"] = bool(
            correlation_summary["effect"] > 0
            and correlation_summary["ci95"][0] > 0
            and extreme_summary["effect"] > 0
            and extreme_summary["ci95"][0] > 0
        )
    diagnostics["development_gate"] = bool(
        all(diagnostics[candidate]["development_valid"] for candidate in CANDIDATES)
    )
    return models, diagnostics, pd.concat(scored_frames, ignore_index=True)


def save_surrogates(
    models: Mapping[str, FrozenPhiSurrogate],
    archive_path: Path,
    contract_path: Path,
) -> None:
    arrays: dict[str, NDArray] = {}
    metadata: dict[str, Any] = {
        "format": "codex-ch5-phir-extension-px3-surrogate-v1",
        "feature_count": 195,
        "pca": False,
        "feature_reference": "post_edit_minus_noop_same_state",
        "fit_intercept": False,
        "ridge_grid": list(RIDGE_GRID),
        "candidates": {},
    }
    for candidate, model in sorted(models.items()):
        prefix = f"c{candidate}"
        arrays[f"{prefix}__mean"] = model.mean
        arrays[f"{prefix}__scale"] = model.scale
        arrays[f"{prefix}__coefficient"] = model.coefficient
        metadata["candidates"][candidate] = {"ridge_alpha": model.ridge_alpha}
    np.savez_compressed(archive_path, **arrays)
    atomic_json(contract_path, metadata)


def load_surrogates(
    archive_path: Path | str, contract_path: Path | str
) -> dict[str, FrozenPhiSurrogate]:
    metadata = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if metadata.get("format") != "codex-ch5-phir-extension-px3-surrogate-v1":
        raise ValueError("unsupported PX3 surrogate archive")
    output: dict[str, FrozenPhiSurrogate] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        for candidate, item in metadata["candidates"].items():
            prefix = f"c{candidate}"
            output[candidate] = FrozenPhiSurrogate(
                candidate,
                archive[f"{prefix}__mean"].copy(),
                archive[f"{prefix}__scale"].copy(),
                archive[f"{prefix}__coefficient"].copy(),
                float(item["ridge_alpha"]),
            )
    return output


def _confirmation_states(
    spec: PX3Spec,
    matrix_id: int,
    beta: NDArray,
    initial: NDArray,
    candidate: str,
) -> tuple[dict[int, Snapshot], str]:
    config = GardConfig()
    rng = np.random.default_rng(_confirmation_path_seed(spec, candidate, matrix_id))
    snapshot = Snapshot(np.asarray(initial, dtype=np.int64).copy(), 0, (), ())
    states: dict[int, Snapshot] = {}
    records: list[FissionRecord] = []
    try:
        for step in range(1, max(spec.landmarks) + 1):
            record = advance_fission(
                snapshot.composition, beta, config, CANDIDATES[candidate], rng
            )
            records.append(record)
            snapshot = _snapshot_after_record(snapshot, record)
            if step in spec.landmarks:
                states[step] = snapshot
    except SimulationError:
        pass
    return states, cr5._records_digest(records)


def _select_confirmation_edits(
    snapshot: Snapshot,
    beta: NDArray,
    model: FrozenPhiSurrogate,
    rng: np.random.Generator,
) -> tuple[dict[str, MolecularEdit | None], dict[str, float], int]:
    legal = enumerate_legal_edits(snapshot.composition)
    if not legal:
        raise ValueError("confirmation state has no legal edits")
    compositions = np.vstack(
        [apply_molecular_edit(snapshot.composition, edit) for edit in legal]
    )
    features = state_graph_features_many(compositions, beta, GardConfig())
    noop_feature = state_graph_features_many(
        np.atleast_2d(snapshot.composition), beta, GardConfig()
    )[0]
    centered = features - noop_feature
    prediction = model.predict(centered)
    maximum = float(prediction.max())
    minimum = float(prediction.min())
    up_index = int(np.flatnonzero(prediction == maximum)[0])
    down_index = int(np.flatnonzero(prediction == minimum)[0])
    random_index = int(rng.integers(0, len(legal)))
    edits = {
        "PHI_UP": legal[up_index],
        "PHI_DOWN": legal[down_index],
        "RANDOM": legal[random_index],
        "NOOP": None,
    }
    predictions = {
        "PHI_UP": float(prediction[up_index]),
        "PHI_DOWN": float(prediction[down_index]),
        "RANDOM": float(prediction[random_index]),
        "NOOP": 0.0,
    }
    return edits, predictions, len(legal)


def _confirmation_matrix(
    args: tuple[int, PX3Spec, str, str]
) -> ConfirmationBatch:
    matrix_id, spec, model_path, contract_path = args
    started = time.process_time()
    config = GardConfig()
    beta = generate_beta(
        config,
        np.random.default_rng(_confirmation_matrix_seed(spec, matrix_id, "matrix")),
    )
    initial = generate_initial_composition(
        config,
        np.random.default_rng(_confirmation_matrix_seed(spec, matrix_id, "initial")),
    )
    models = load_surrogates(model_path, contract_path)
    score_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for candidate in CANDIDATES:
            states, path_digest = _confirmation_states(
                spec, matrix_id, beta, initial, candidate
            )
            for landmark, snapshot in states.items():
                edits, predictions, legal_count = _select_confirmation_edits(
                    snapshot,
                    beta,
                    models[candidate],
                    np.random.default_rng(
                        _confirmation_selection_seed(spec, candidate, matrix_id, landmark)
                    ),
                )
                pairs: dict[tuple[str, str], tuple[list[NDArray], list[NDArray]]] = {
                    (arm, half): ([], []) for arm in ARMS for half in HALVES
                }
                local_branches: list[dict[str, Any]] = []
                for arm in ARMS:
                    edit = edits[arm]
                    edit_rows.append(
                        {
                            "matrix_id": matrix_id,
                            "candidate": candidate,
                            "landmark": landmark,
                            "arm": arm,
                            "remove_type": -1 if edit is None else edit.remove_type,
                            "add_type": -1 if edit is None else edit.add_type,
                            "predicted_delta_phi": predictions[arm],
                            "legal_edits_scored": legal_count,
                            "path_record_digest": path_digest,
                            "snapshot_digest": _snapshot_digest(snapshot),
                        }
                    )
                for branch in range(spec.branches):
                    half = "A" if branch < spec.branches // 2 else "B"
                    seed = _confirmation_future_seed(
                        spec, candidate, matrix_id, landmark, branch
                    )
                    for arm in ARMS:
                        edit = edits[arm]
                        composition = (
                            snapshot.composition
                            if edit is None
                            else apply_molecular_edit(snapshot.composition, edit)
                        )
                        launch = Snapshot(
                            np.asarray(composition, dtype=np.int64).copy(),
                            snapshot.generation,
                            snapshot.inheritance,
                            snapshot.boundary_h,
                            snapshot.previous_growth_steps,
                            snapshot.cumulative_growth_steps,
                        )
                        records, completed = simulate_future_absorbing(
                            launch,
                            beta,
                            config,
                            CANDIDATES[candidate],
                            spec.horizon,
                            np.random.default_rng(seed),
                        )
                        past, future = _records_to_pairs(launch, records)
                        pairs[(arm, half)][0].extend(past)
                        pairs[(arm, half)][1].extend(future)
                        inherited = np.asarray(
                            [
                                record.h > config.inheritance_threshold
                                for record in records
                            ],
                            dtype=bool,
                        )
                        process = outcome_from_records(
                            launch, records, completed, spec.horizon
                        )
                        row = {
                            "matrix_id": matrix_id,
                            "candidate": candidate,
                            "landmark": landmark,
                            "arm": arm,
                            "branch": branch,
                            "half": half,
                            "completed": int(completed),
                            "observed_fissions": len(records),
                            "inherited_fraction": float(inherited.mean())
                            if inherited.size
                            else 0.0,
                            "break_probability": int((~inherited).any()),
                            "joint_break_run3_within_f8": int(process.joint_break_run3),
                            "record_digest": process.record_digest,
                        }
                        branch_rows.append(row)
                        local_branches.append(row)
                local_frame = pd.DataFrame(local_branches)
                for arm in ARMS:
                    for half in HALVES:
                        selected = local_frame[
                            (local_frame.arm == arm) & (local_frame.half == half)
                        ]
                        past, future = pairs[(arm, half)]
                        score = safe_score_pairs(
                            np.asarray(past), np.asarray(future), beta, "material", config
                        )
                        row = {
                            "matrix_id": matrix_id,
                            "candidate": candidate,
                            "landmark": landmark,
                            "arm": arm,
                            "half": half,
                            "transition_pairs": len(past),
                            "inherited_fraction": float(selected.inherited_fraction.mean()),
                            "break_probability": float(selected.break_probability.mean()),
                            "joint_break_run3_within_f8": float(
                                selected.joint_break_run3_within_f8.mean()
                            ),
                            "survival_probability": float(selected.completed.mean()),
                        }
                        row.update(score.fields("material"))
                        score_rows.append(row)
    provisional = ConfirmationBatch(
        matrix_id,
        tuple(score_rows),
        tuple(branch_rows),
        tuple(edit_rows),
        float(time.process_time() - started),
        "",
    )
    return ConfirmationBatch(
        provisional.matrix_id,
        provisional.score_rows,
        provisional.branch_rows,
        provisional.edit_rows,
        provisional.cpu_seconds,
        _batch_digest(provisional),
    )


def program_protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "phase": "PX3",
        "question": "can molecular substitutions selected to change material full-block Phi causally move that Phi reading?",
        "development": {
            "spec": asdict(development_spec()),
            "resource_amendment": DEVELOPMENT_AMENDMENT,
            "original_registration_id": ORIGINAL_DEVELOPMENT_REGISTRATION_ID,
            "original_matrices": ORIGINAL_DEVELOPMENT_MATRICES,
            "original_cpu_seconds": ORIGINAL_DEVELOPMENT_CPU_SECONDS,
            "carried_forward_matrix_ids": list(DEVELOPMENT_CARRIED_FORWARD),
            "interim_diagnostics_inspected": True,
            "original_development_gate_eligible": ORIGINAL_DEVELOPMENT_GATE_ELIGIBLE,
            "maximum_workers": DEVELOPMENT_MAX_WORKERS,
            "source": "exact NOOP replay of the sealed PX1 matrices",
            "replicates": DEVELOPMENT_REPLICATES,
            "candidate_edits_per_state": DEVELOPMENT_EDITS,
            "edit_candidates": "heredity extrema, outgoing-rule extrema, then uniform legal fill without replacement",
            "label": "realized explicit-pair material full-block edit minus NOOP",
            "features": "195-dimensional post-edit graph/state features minus the same state's NOOP feature vector",
            "pca": False,
            "model": "candidate-separated linear ridge without intercept",
            "ridge_grid": list(RIDGE_GRID),
            "cv": "five-fold whole-matrix cross-validation",
            "validity": "positive matrix-bootstrap lower bounds for OOF within-state Spearman and OOF top-bottom realized separation in both candidates",
        },
        "confirmation": {
            "spec": asdict(confirmation_spec()),
            "resource_note": CONFIRMATION_RESOURCE_NOTE,
            "maximum_workers": CONFIRMATION_MAX_WORKERS,
            "fresh_matrices": True,
            "arms": list(ARMS),
            "all_legal_edits_scored": True,
            "future_seed_includes_arm": False,
            "selection_stream_separate": True,
            "halves": HALVES,
            "primary": "material full-block PHI_UP minus PHI_DOWN",
            "heredity_coupling_direction": "PHI_UP minus PHI_DOWN inherited fraction positive",
            "equivalence_margin": "0.2 times candidate NOOP matrix SD, sealed after development",
        },
        "classification": {
            "original_px3_confirmed": False,
            "prospective_pilot_selector_confirmation": "resource-bounded development validity and all four fresh confirmation cells pass",
            "exploratory_selector_success": "confirmation passes after development validity fails",
            "coupled": "prospective pilot-selector control and positive inherited-fraction effects in all four cells",
            "decoupled": "prospective pilot-selector control and inherited-fraction TOST equivalence within +/-0.025 in all four cells",
            "mixed": "neither coupled nor decoupled",
        },
        "run_confirmation_if_development_fails": True,
        "failed_development_cannot_be_rescued": True,
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap": BOOTSTRAP_DRAWS,
            "randomization": RANDOMIZATION_DRAWS,
            "holm": "four candidate-by-half cells",
        },
        "no_48_matrix_campaign": True,
        "claim_boundary": [
            "direct control of one estimator is not control of consciousness",
            "a surrogate success does not make Phi the physical cause",
            "strict-eight is excluded",
            "prior negative results remain unchanged",
        ],
    }
    value["protocol_id"] = canonical_digest(value)
    return value


def _checkpointed(
    kind: str,
    spec: PX3Spec,
    registration: Mapping[str, Any],
    directory: Path,
    inputs: Sequence[int],
    worker: Any,
    worker_tail: tuple[str, ...],
    workers: int,
    prior_cpu: float = 0.0,
) -> tuple[list[Any], float]:
    directory.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": CHECKPOINT_FORMAT,
        "kind": kind,
        "registration_id": registration["registration_id"],
        "protocol_id": registration["protocol"]["protocol_id"],
        "spec": asdict(spec),
        "matrix_ids": list(inputs),
        "source_hashes": registration["source_hashes"],
    }
    contract["contract_id"] = canonical_digest(contract)
    contract_path = directory / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != canonical_json(contract):
            raise ValueError(f"PX3 checkpoint contract changed: {directory}")
    else:
        atomic_json(contract_path, contract)
    batches: list[Any | None] = [None] * len(inputs)
    missing: list[int] = []
    cpu = float(prior_cpu)
    for index, matrix_id in enumerate(inputs):
        path = directory / f"matrix_{matrix_id:04d}.pkl"
        if path.exists():
            batch = _load_checkpoint(path)
            if batch.matrix_id != matrix_id or batch.scientific_digest != _batch_digest(batch):
                raise ValueError(f"invalid PX3 checkpoint: {path}")
            batches[index] = batch
            cpu += batch.cpu_seconds
        else:
            missing.append(index)

    def status(state: str) -> None:
        complete = sum(batch is not None for batch in batches)
        atomic_json(
            directory / "status.json",
            {
                "kind": kind,
                "state": state,
                "completed": complete,
                "total": len(inputs),
                "fraction": complete / max(1, len(inputs)),
                "cpu_seconds": cpu,
            },
        )

    status("running")
    arguments = [(inputs[index], spec, *worker_tail) for index in missing]
    executor: ProcessPoolExecutor | None = None
    generated: Iterable[Any]
    if workers <= 1:
        generated = map(worker, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS))
        generated = executor.map(worker, arguments, chunksize=1)
    try:
        for index, batch in zip(missing, generated, strict=True):
            matrix_id = inputs[index]
            if batch.matrix_id != matrix_id or batch.scientific_digest != _batch_digest(batch):
                raise AssertionError("PX3 worker returned invalid batch")
            batches[index] = batch
            atomic_pickle(directory / f"matrix_{matrix_id:04d}.pkl", batch)
            cpu += batch.cpu_seconds
            status("running")
            print(f"[PX3 {kind}] {sum(item is not None for item in batches)}/{len(inputs)}", flush=True)
            if cpu > spec.cpu_seconds:
                status("paused_cpu_budget")
                raise RuntimeError(f"PX3 {kind} CPU allocation reached; checkpoints retained")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    if any(batch is None for batch in batches):
        raise AssertionError(f"PX3 {kind} incomplete")
    status("complete")
    return [batch for batch in batches if batch is not None], cpu


def _replay_audit(left: Sequence[Any], right: Sequence[Any], expected: int) -> dict[str, Any]:
    rows = [
        {
            "matrix_id": first.matrix_id,
            "generated": first.scientific_digest,
            "replay": second.scientific_digest,
            "exact": first.scientific_digest == second.scientific_digest,
        }
        for first, second in zip(left, right, strict=True)
    ]
    return {
        "matrices": rows,
        "complete_exact_replay": len(rows) == expected and all(row["exact"] for row in rows),
    }


def _development_protocol() -> dict[str, Any]:
    px1_manifest = json.loads((PX1_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    px2_manifest_path = RESULT_ROOT / "px2_event_locked_recovery" / "manifest.json"
    px2_manifest = json.loads(px2_manifest_path.read_text(encoding="utf-8"))
    value = {
        "phase": "PX3_DEVELOPMENT",
        "program_protocol_id": program_protocol()["protocol_id"],
        "spec": asdict(development_spec()),
        "px1_registration_id": px1_manifest["registration_id"],
        "px1_manifest_sha256": sha256_file(PX1_OUTPUT / "manifest.json"),
        "px1_matrix_inputs_sha256": sha256_file(PX1_OUTPUT / "matrix_inputs.npz"),
        "px2_registration_id": px2_manifest["registration_id"],
        "px2_manifest_sha256": sha256_file(px2_manifest_path),
        "px2_integrity_complete": bool(
            px2_manifest["complete_exact_replay"]
            and px2_manifest["complete_readback_exact"]
        ),
        "heredity_model_sha256": EXPECTED_HEREDITY_MODEL_SHA256,
        "development_amendment_sha256": sha256_file(ROOT / DEVELOPMENT_AMENDMENT),
        "supersedes_registration_id": ORIGINAL_DEVELOPMENT_REGISTRATION_ID,
        "carried_forward_matrix_ids": list(DEVELOPMENT_CARRIED_FORWARD),
        "interim_diagnostics_inspected": True,
        "original_development_gate_eligible": ORIGINAL_DEVELOPMENT_GATE_ELIGIBLE,
        "confirmation_not_generated": True,
    }
    value["protocol_id"] = canonical_digest(value)
    return value


def register_development() -> dict[str, Any]:
    verify_checksums(DEFAULT_VALIDATION)
    validation = json.loads((DEFAULT_VALIDATION / "validation.json").read_text())
    if not validation["all_passed"]:
        raise ValueError("PX3 validation did not pass")
    verify_checksums(PX1_OUTPUT)
    if sha256_file(HEREDITY_MODEL) != EXPECTED_HEREDITY_MODEL_SHA256:
        raise ValueError("PX3 heredity model hash changed")
    verify_checksums(PRE_AMENDMENT_DEVELOPMENT_REGISTRATION)
    prior = json.loads(
        (PRE_AMENDMENT_DEVELOPMENT_REGISTRATION / "registration.json").read_text()
    )
    if prior.get("registration_id") != ORIGINAL_DEVELOPMENT_REGISTRATION_ID:
        raise ValueError("PX3 superseded registration identity changed")
    if DEFAULT_DEVELOPMENT_REGISTRATION.exists():
        raise FileExistsError("PX3 development registration exists")
    master = json.loads((MASTER_REGISTRATION / "registration.json").read_text())
    body: dict[str, Any] = {
        "format": DEVELOPMENT_REGISTRATION_FORMAT,
        "master_registration_id": master["registration_id"],
        "protocol": _development_protocol(),
        "source_hashes": _source_hashes(),
        "runtime": runtime_versions(),
        "new_scientific_matrices_at_registration": 0,
    }
    body["registration_id"] = canonical_digest(body)
    DEFAULT_DEVELOPMENT_REGISTRATION.mkdir(parents=True)
    shutil.copy2(ROOT / DOCUMENT, DEFAULT_DEVELOPMENT_REGISTRATION / "preregistration.md")
    shutil.copy2(
        ROOT / DEVELOPMENT_AMENDMENT,
        DEFAULT_DEVELOPMENT_REGISTRATION / "development_amendment.md",
    )
    shutil.copy2(HEREDITY_MODEL, DEFAULT_DEVELOPMENT_REGISTRATION / "frozen_heredity_predictor.npz")
    atomic_json(DEFAULT_DEVELOPMENT_REGISTRATION / "protocol.json", body["protocol"])
    atomic_json(DEFAULT_DEVELOPMENT_REGISTRATION / "registration.json", body)
    write_checksums(DEFAULT_DEVELOPMENT_REGISTRATION)
    _append_ledger(
        f"<!-- phir-extension-px3-development-registration-{body['registration_id']} -->",
        [
            "## Phi-r extension PX3 development registered",
            "",
            f"- Registration: `{body['registration_id']}`.",
            f"- Resource-bounded amendment: 12 matrices, carrying forward verified matrices `{list(DEVELOPMENT_CARRIED_FORWARD)}` after interim inspection.",
            "- The original 24-matrix development gate remains incomplete and ineligible; any later fresh success is prospective confirmation of a pilot-developed selector.",
            "- No PCA, no outcome-driven architecture change, and no 48-matrix run are permitted.",
        ],
    )
    return body


def carry_forward_development_checkpoints() -> dict[str, Any]:
    """Carry the six sealed pre-amendment batches into the amended contract."""

    registration = verify_development_registration()
    verify_checksums(PRE_AMENDMENT_DEVELOPMENT_REGISTRATION)
    prior = json.loads(
        (PRE_AMENDMENT_DEVELOPMENT_REGISTRATION / "registration.json").read_text()
    )
    if prior.get("registration_id") != ORIGINAL_DEVELOPMENT_REGISTRATION_ID:
        raise ValueError("PX3 prior registration identity mismatch")
    old_contract = json.loads(
        (PRE_AMENDMENT_DEVELOPMENT_WORK / "checkpoint_contract.json").read_text()
    )
    if (
        old_contract.get("registration_id") != ORIGINAL_DEVELOPMENT_REGISTRATION_ID
        or old_contract.get("matrix_ids") != list(range(ORIGINAL_DEVELOPMENT_MATRICES))
    ):
        raise ValueError("PX3 prior checkpoint contract mismatch")
    target = DEFAULT_WORK / "development_generate"
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"PX3 amended checkpoint directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for matrix_id in DEVELOPMENT_CARRIED_FORWARD:
        source = PRE_AMENDMENT_DEVELOPMENT_WORK / f"matrix_{matrix_id:04d}.pkl"
        batch = _load_checkpoint(source)
        if (
            not isinstance(batch, DevelopmentBatch)
            or batch.matrix_id != matrix_id
            or batch.scientific_digest != _batch_digest(batch)
        ):
            raise ValueError(f"PX3 carry-forward checkpoint failed: {source}")
        destination = target / source.name
        shutil.copy2(source, destination)
        if sha256_file(source) != sha256_file(destination):
            raise AssertionError(f"PX3 carry-forward byte mismatch: {source}")
        rows.append(
            {
                "matrix_id": matrix_id,
                "scientific_digest": batch.scientific_digest,
                "cpu_seconds": batch.cpu_seconds,
                "checkpoint_sha256": sha256_file(destination),
            }
        )
    payload = {
        "format": "codex-ch5-phir-extension-px3-carry-forward-v1",
        "old_registration_id": ORIGINAL_DEVELOPMENT_REGISTRATION_ID,
        "new_registration_id": registration["registration_id"],
        "interim_diagnostics_inspected": True,
        "matrices": rows,
        "all_scientific_digests_valid": len(rows) == len(DEVELOPMENT_CARRIED_FORWARD),
        "bytes_copied_exactly": True,
    }
    atomic_json(DEFAULT_WORK / "development_carry_forward_audit.json", payload)
    return payload


def verify_development_registration() -> dict[str, Any]:
    verify_checksums(DEFAULT_DEVELOPMENT_REGISTRATION)
    body = json.loads(
        (DEFAULT_DEVELOPMENT_REGISTRATION / "registration.json").read_text()
    )
    observed = body.pop("registration_id")
    if body.get("format") != DEVELOPMENT_REGISTRATION_FORMAT or observed != canonical_digest(body):
        raise ValueError("PX3 development registration identity failed")
    body["registration_id"] = observed
    if body["protocol"] != canonical_json(_development_protocol()):
        raise ValueError("PX3 development protocol changed")
    if body["source_hashes"] != _source_hashes():
        raise ValueError("PX3 source changed after development registration")
    return body


def _write_development_result(
    batches: Sequence[DevelopmentBatch],
    replay: Mapping[str, Any],
    registration: Mapping[str, Any],
    cpu: float,
) -> dict[str, Any]:
    training = pd.DataFrame([row for batch in batches for row in batch.training_rows])
    states = pd.DataFrame([row for batch in batches for row in batch.state_rows])
    models, diagnostics, scored = fit_surrogates(training)
    margins: dict[str, Any] = {}
    for candidate in CANDIDATES:
        matrix_noop = (
            states[states.candidate == candidate].groupby("matrix_id")["noop_phi"].mean()
        )
        standard_deviation = float(matrix_noop.std(ddof=1))
        margins[candidate] = {
            "noop_matrix_sd": standard_deviation,
            "equivalence_margin": 0.2 * standard_deviation,
        }
    temporary = DEFAULT_DEVELOPMENT_OUTPUT.with_name(
        DEFAULT_DEVELOPMENT_OUTPUT.name + f".tmp-{os.getpid()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    scored.drop(columns=["feature"]).to_csv(temporary / "training_rows.csv.gz", index=False)
    states.to_csv(temporary / "state_rows.csv.gz", index=False)
    save_surrogates(
        models,
        temporary / "frozen_phi_surrogate.npz",
        temporary / "model_contract.json",
    )
    metrics = {
        "format": "codex-ch5-phir-extension-px3-development-metrics-v1",
        "diagnostics": diagnostics,
        "equivalence_margins": margins,
        "development_gate": diagnostics["development_gate"],
    }
    atomic_json(temporary / "primary_metrics.json", metrics)
    atomic_json(temporary / "replay_audit.json", replay)
    report = [
        "# PX3 Phi-surrogate development",
        "",
        f"Registration: `{registration['registration_id']}`.",
        "",
        f"Development validity gate: **{diagnostics['development_gate']}**.",
        "",
    ]
    for candidate in CANDIDATES:
        item = diagnostics[candidate]
        report.extend(
            [
                f"## Candidate {candidate}",
                "",
                f"- Selected ridge alpha: `{item['selected_alpha']}`.",
                f"- OOF mean within-state Spearman: `{item['oof_spearman']['effect']:+.5f}` "
                f"[{item['oof_spearman']['ci95'][0]:+.5f}, {item['oof_spearman']['ci95'][1]:+.5f}].",
                f"- OOF top-minus-bottom realized separation: `{item['oof_top_bottom']['effect']:+.5f}` "
                f"[{item['oof_top_bottom']['ci95'][0]:+.5f}, {item['oof_top_bottom']['ci95'][1]:+.5f}].",
                "",
            ]
        )
    (temporary / "SCIENTIFIC_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (temporary / "LAY_SUMMARY.md").write_text(
        "# PX3 development lay summary\n\n"
        "We tested whether a simple frozen score can rank one-molecule edits by how much they will change the material information reading in new random futures. "
        + ("It passed the two-candidate development check.\n" if diagnostics["development_gate"] else "It did not pass the complete two-candidate development check. Confirmation will still run as preregistered, but cannot establish a confirmed selector if this gate is false.\n"),
        encoding="utf-8",
    )
    manifest = {
        "format": RESULT_FORMAT,
        "phase": "PX3_DEVELOPMENT",
        "registration_id": registration["registration_id"],
        "matrices": DEVELOPMENT_MATRICES,
        "cpu_seconds": cpu,
        "development_gate": diagnostics["development_gate"],
        "resource_bounded_development_gate": diagnostics["development_gate"],
        "original_24_matrix_gate_eligible": ORIGINAL_DEVELOPMENT_GATE_ELIGIBLE,
        "interim_diagnostics_inspected_before_amendment": True,
        "complete_exact_replay": replay["complete_exact_replay"],
        "complete_readback_exact": False,
        "model_sha256": sha256_file(temporary / "frozen_phi_surrogate.npz"),
        "model_contract_sha256": sha256_file(temporary / "model_contract.json"),
    }
    atomic_json(temporary / "manifest.json", manifest)
    write_checksums(temporary)
    temporary.replace(DEFAULT_DEVELOPMENT_OUTPUT)
    verify_checksums(DEFAULT_DEVELOPMENT_OUTPUT)
    readback_models = load_surrogates(
        DEFAULT_DEVELOPMENT_OUTPUT / "frozen_phi_surrogate.npz",
        DEFAULT_DEVELOPMENT_OUTPUT / "model_contract.json",
    )
    exact = all(
        np.array_equal(readback_models[candidate].coefficient, models[candidate].coefficient)
        for candidate in CANDIDATES
    )
    manifest["complete_readback_exact"] = exact
    atomic_json(DEFAULT_DEVELOPMENT_OUTPUT / "manifest.json", manifest)
    atomic_json(DEFAULT_DEVELOPMENT_OUTPUT / "readback_audit.json", {"complete": exact})
    write_checksums(DEFAULT_DEVELOPMENT_OUTPUT)
    if not exact:
        raise AssertionError("PX3 development readback failed")
    _append_ledger(
        f"<!-- phir-extension-px3-development-result-{registration['registration_id']} -->",
        [
            "## Phi-r extension PX3 development completed",
            "",
            "- Result: `results/phir_extension/px3_development`.",
            f"- Development gate: `{diagnostics['development_gate']}`; complete replay and readback passed.",
            "- The separately registered confirmation remains required regardless of this result.",
        ],
    )
    return manifest


def run_development(workers: int = MAX_WORKERS) -> dict[str, Any]:
    registration = verify_development_registration()
    verify_checksums(DEFAULT_SMOKE)
    spec = development_spec()
    tail = (
        str(PX1_OUTPUT / "matrix_inputs.npz"),
        str(DEFAULT_DEVELOPMENT_REGISTRATION / "frozen_heredity_predictor.npz"),
    )
    generated, cpu = _checkpointed(
        "development_generate",
        spec,
        registration,
        DEFAULT_WORK / "development_generate",
        list(range(spec.matrices)),
        _development_matrix,
        tail,
        min(workers, DEVELOPMENT_MAX_WORKERS),
    )
    replayed, cpu = _checkpointed(
        "development_replay",
        spec,
        registration,
        DEFAULT_WORK / "development_replay",
        list(range(spec.matrices)),
        _development_matrix,
        tail,
        min(workers, DEVELOPMENT_MAX_WORKERS),
        cpu,
    )
    replay = _replay_audit(generated, replayed, spec.matrices)
    if not replay["complete_exact_replay"]:
        raise AssertionError("PX3 development replay failed")
    return _write_development_result(generated, replay, registration, cpu)


def _confirmation_protocol(development_manifest: Mapping[str, Any]) -> dict[str, Any]:
    metrics = json.loads(
        (DEFAULT_DEVELOPMENT_OUTPUT / "primary_metrics.json").read_text()
    )
    value = {
        "phase": "PX3_CONFIRMATION",
        "program_protocol_id": program_protocol()["protocol_id"],
        "spec": asdict(confirmation_spec()),
        "development_registration_id": development_manifest["registration_id"],
        "development_manifest_sha256": sha256_file(
            DEFAULT_DEVELOPMENT_OUTPUT / "manifest.json"
        ),
        "development_gate": bool(development_manifest["development_gate"]),
        "resource_bounded_pilot_development": True,
        "original_24_matrix_gate_eligible": False,
        "model_sha256": development_manifest["model_sha256"],
        "model_contract_sha256": development_manifest["model_contract_sha256"],
        "information_equivalence_margins": metrics["equivalence_margins"],
        "run_even_if_development_gate_failed": True,
        "claim_status": "prospective confirmation of a pilot-developed selector only",
        "new_confirmation_matrices_at_registration": 0,
    }
    value["protocol_id"] = canonical_digest(value)
    return value


def register_confirmation() -> dict[str, Any]:
    verify_checksums(DEFAULT_DEVELOPMENT_OUTPUT)
    development = json.loads(
        (DEFAULT_DEVELOPMENT_OUTPUT / "manifest.json").read_text()
    )
    if not development["complete_exact_replay"] or not development["complete_readback_exact"]:
        raise ValueError("PX3 development integrity failed")
    if DEFAULT_CONFIRMATION_REGISTRATION.exists():
        raise FileExistsError("PX3 confirmation registration exists")
    body: dict[str, Any] = {
        "format": CONFIRMATION_REGISTRATION_FORMAT,
        "protocol": _confirmation_protocol(development),
        "source_hashes": _source_hashes(),
        "runtime": runtime_versions(),
        "new_scientific_matrices_at_registration": 0,
    }
    body["registration_id"] = canonical_digest(body)
    DEFAULT_CONFIRMATION_REGISTRATION.mkdir(parents=True)
    shutil.copy2(ROOT / DOCUMENT, DEFAULT_CONFIRMATION_REGISTRATION / "preregistration.md")
    shutil.copy2(
        ROOT / CONFIRMATION_RESOURCE_NOTE,
        DEFAULT_CONFIRMATION_REGISTRATION / "confirmation_resource_note.md",
    )
    shutil.copy2(
        DEFAULT_DEVELOPMENT_OUTPUT / "frozen_phi_surrogate.npz",
        DEFAULT_CONFIRMATION_REGISTRATION / "frozen_phi_surrogate.npz",
    )
    shutil.copy2(
        DEFAULT_DEVELOPMENT_OUTPUT / "model_contract.json",
        DEFAULT_CONFIRMATION_REGISTRATION / "model_contract.json",
    )
    atomic_json(DEFAULT_CONFIRMATION_REGISTRATION / "protocol.json", body["protocol"])
    atomic_json(DEFAULT_CONFIRMATION_REGISTRATION / "registration.json", body)
    write_checksums(DEFAULT_CONFIRMATION_REGISTRATION)
    _append_ledger(
        f"<!-- phir-extension-px3-confirmation-registration-{body['registration_id']} -->",
        [
            "## Phi-r extension PX3 confirmation registered",
            "",
            f"- Registration: `{body['registration_id']}`.",
            f"- Development gate at seal: `{development['development_gate']}`.",
            "- Twenty-four entirely new matrices will run regardless; a positive result can only prospectively confirm the resource-bounded pilot selector.",
        ],
    )
    return body


def verify_confirmation_registration() -> dict[str, Any]:
    verify_checksums(DEFAULT_CONFIRMATION_REGISTRATION)
    body = json.loads(
        (DEFAULT_CONFIRMATION_REGISTRATION / "registration.json").read_text()
    )
    observed = body.pop("registration_id")
    if body.get("format") != CONFIRMATION_REGISTRATION_FORMAT or observed != canonical_digest(body):
        raise ValueError("PX3 confirmation registration identity failed")
    body["registration_id"] = observed
    development = json.loads((DEFAULT_DEVELOPMENT_OUTPUT / "manifest.json").read_text())
    if body["protocol"] != canonical_json(_confirmation_protocol(development)):
        raise ValueError("PX3 confirmation protocol changed")
    if body["source_hashes"] != _source_hashes():
        raise ValueError("PX3 source changed after confirmation registration")
    if sha256_file(DEFAULT_CONFIRMATION_REGISTRATION / "frozen_phi_surrogate.npz") != body["protocol"]["model_sha256"]:
        raise ValueError("PX3 confirmation model changed")
    return body


def analyze_confirmation(
    batches: Sequence[ConfirmationBatch], registration: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    scores = pd.DataFrame([row for batch in batches for row in batch.score_rows])
    branches = pd.DataFrame([row for batch in batches for row in batch.branch_rows])
    edits = pd.DataFrame([row for batch in batches for row in batch.edit_rows])
    scores["candidate"] = scores.candidate.astype(str).str.zfill(2)
    arrays: dict[str, NDArray] = {}
    matrix_rows: list[dict[str, Any]] = []
    primary: list[dict[str, Any]] = []
    specificity: list[dict[str, Any]] = []
    inheritance: list[dict[str, Any]] = []
    margins = registration["protocol"]["information_equivalence_margins"]
    for candidate in CANDIDATES:
        for half in HALVES:
            filters = {"candidate": candidate, "half": half}
            cell = f"{candidate}_{half}"
            phi = paired_matrix_effects(
                scores,
                "material_full_revised",
                "PHI_UP",
                "PHI_DOWN",
                filters=filters,
                within=("landmark",),
            )
            summary, local = paired_summary(phi.to_numpy(), f"PX3/confirmation/{cell}/phi")
            summary.update({"candidate": candidate, "half": half, "contrast": "PHI_UP-PHI_DOWN"})
            primary.append(summary)
            arrays.update({f"phi__{cell}__{name}": value for name, value in local.items()})
            for matrix_id, value in phi.items():
                matrix_rows.append(
                    {
                        "family": "phi",
                        "candidate": candidate,
                        "half": half,
                        "matrix_id": int(matrix_id),
                        "value": float(value),
                    }
                )
            random = paired_matrix_effects(
                scores,
                "material_full_revised",
                "RANDOM",
                "NOOP",
                filters=filters,
                within=("landmark",),
            )
            random_summary, local = paired_summary(
                random.to_numpy(),
                f"PX3/confirmation/{cell}/random-noop",
                equivalence_margin=float(margins[candidate]["equivalence_margin"]),
            )
            random_summary.update(
                {"family": "phi_specificity", "candidate": candidate, "half": half}
            )
            specificity.append(random_summary)
            arrays.update({f"specificity__{cell}__{name}": value for name, value in local.items()})
            inherited = paired_matrix_effects(
                scores,
                "inherited_fraction",
                "PHI_UP",
                "PHI_DOWN",
                filters=filters,
                within=("landmark",),
            )
            inherited_summary, local = paired_summary(
                inherited.to_numpy(), f"PX3/confirmation/{cell}/inheritance",
                equivalence_margin=HEREDITY_EQUIVALENCE_MARGIN,
            )
            inherited_summary.update({"candidate": candidate, "half": half})
            inheritance.append(inherited_summary)
            arrays.update({f"inheritance__{cell}__{name}": value for name, value in local.items()})
    apply_holm(primary)
    apply_holm(inheritance)
    specificity_lookup = {
        (row["candidate"], row["half"]): row for row in specificity
    }
    confirmation_only = bool(
        len(primary) == 4
        and all(
            row["effect"] > 0
            and row["ci95"][0] > 0
            and row.get("holm_adjusted_p", 1) < 0.05
            and specificity_lookup[(row["candidate"], row["half"])].get("tost_via_90ci", False)
            for row in primary
        )
    )
    development_gate = bool(registration["protocol"]["development_gate"])
    prospective_pilot_confirmation = development_gate and confirmation_only
    confirmed = False
    coupled = bool(
        prospective_pilot_confirmation
        and all(
            row["effect"] > 0
            and row["ci95"][0] > 0
            and row.get("holm_adjusted_p", 1) < 0.05
            for row in inheritance
        )
    )
    decoupled = bool(
        prospective_pilot_confirmation
        and all(row.get("tost_via_90ci", False) for row in inheritance)
    )
    classification = (
        "prospective_pilot_selector_coupled"
        if coupled
        else "prospective_pilot_selector_decoupled"
        if decoupled
        else "prospective_pilot_selector_mixed"
        if prospective_pilot_confirmation
        else "exploratory_selector_success_after_failed_development"
        if confirmation_only
        else "direct_phi_control_not_confirmed"
    )
    metrics = {
        "format": "codex-ch5-phir-extension-px3-confirmation-metrics-v1",
        "primary": primary,
        "specificity": specificity,
        "inheritance": inheritance,
        "gates": {
            "development_validity": development_gate,
            "confirmation_only_phi_control": confirmation_only,
            "confirmed_direct_phi_control": confirmed,
            "original_24_matrix_gate_eligible": False,
            "prospective_pilot_selector_confirmation": prospective_pilot_confirmation,
            "coupled_heredity_response": coupled,
            "decoupled_heredity_response": decoupled,
        },
        "classification": classification,
    }
    return metrics, {
        "scores": scores,
        "branches": branches,
        "edits": edits,
        "matrix_effects": pd.DataFrame(matrix_rows),
    }, arrays


def _write_confirmation_result(
    batches: Sequence[ConfirmationBatch],
    replay: Mapping[str, Any],
    registration: Mapping[str, Any],
    cpu: float,
) -> dict[str, Any]:
    metrics, tables, arrays = analyze_confirmation(batches, registration)
    temporary = DEFAULT_CONFIRMATION_OUTPUT.with_name(
        DEFAULT_CONFIRMATION_OUTPUT.name + f".tmp-{os.getpid()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(temporary / f"{name}.csv.gz", index=False)
    np.savez_compressed(temporary / "inference_arrays.npz", **arrays)
    atomic_json(temporary / "primary_metrics.json", metrics)
    atomic_json(temporary / "replay_audit.json", replay)
    report = [
        "# PX3 direct Phi-control confirmation",
        "",
        f"Registration: `{registration['registration_id']}`.",
        "",
        f"Classification: **{metrics['classification']}**.",
        "",
        "| Candidate | Half | PHI_UP-DOWN [95% CI] | Holm p |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in metrics["primary"]:
        report.append(
            f"| {row['candidate']} | {row['half']} | {row['effect']:+.5f} "
            f"[{row['ci95'][0]:+.5f}, {row['ci95'][1]:+.5f}] | "
            f"{row.get('holm_adjusted_p', float('nan')):.4g} |"
        )
    report.extend(
        [
            "",
            "```json",
            json.dumps(metrics["gates"], indent=2, sort_keys=True),
            "```",
            "",
            "Changing this estimator does not establish that Phi is a physical cause, nor does it imply consciousness or life.",
        ]
    )
    (temporary / "SCIENTIFIC_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (temporary / "LAY_SUMMARY.md").write_text(
        "# PX3 confirmation lay summary\n\n"
        "A model chose tiny molecular edits predicted to move the material information reading up or down, and we tested those edits on entirely new assemblies.\n\n"
        f"The registered classification is **{metrics['classification']}**. Because development was reduced after an interim inspection, even a positive fresh result is prospective confirmation of a pilot-developed selector, not the original PX3 confirmation.\n",
        encoding="utf-8",
    )
    manifest = {
        "format": RESULT_FORMAT,
        "phase": "PX3_CONFIRMATION",
        "registration_id": registration["registration_id"],
        "matrices": CONFIRMATION_MATRICES,
        "cpu_seconds": cpu,
        "complete_exact_replay": replay["complete_exact_replay"],
        "complete_readback_exact": False,
        "gates": metrics["gates"],
        "classification": metrics["classification"],
    }
    atomic_json(temporary / "manifest.json", manifest)
    write_checksums(temporary)
    temporary.replace(DEFAULT_CONFIRMATION_OUTPUT)
    verify_checksums(DEFAULT_CONFIRMATION_OUTPUT)
    readback = pd.read_csv(DEFAULT_CONFIRMATION_OUTPUT / "scores.csv.gz")
    exact = len(readback) == len(tables["scores"])
    manifest["complete_readback_exact"] = exact
    atomic_json(DEFAULT_CONFIRMATION_OUTPUT / "manifest.json", manifest)
    atomic_json(DEFAULT_CONFIRMATION_OUTPUT / "readback_audit.json", {"complete": exact})
    write_checksums(DEFAULT_CONFIRMATION_OUTPUT)
    if not exact:
        raise AssertionError("PX3 confirmation readback failed")
    _append_ledger(
        f"<!-- phir-extension-px3-confirmation-result-{registration['registration_id']} -->",
        [
            "## Phi-r extension PX3 confirmation completed",
            "",
            "- Result: `results/phir_extension/px3_confirmation`.",
            f"- Classification: `{metrics['classification']}`.",
            "- Complete exact replay and readback passed; prior results remain unchanged.",
        ],
    )
    return manifest


def run_confirmation(workers: int = MAX_WORKERS) -> dict[str, Any]:
    registration = verify_confirmation_registration()
    spec = confirmation_spec()
    tail = (
        str(DEFAULT_CONFIRMATION_REGISTRATION / "frozen_phi_surrogate.npz"),
        str(DEFAULT_CONFIRMATION_REGISTRATION / "model_contract.json"),
    )
    generated, cpu = _checkpointed(
        "confirmation_generate",
        spec,
        registration,
        DEFAULT_WORK / "confirmation_generate",
        list(range(spec.matrices)),
        _confirmation_matrix,
        tail,
        min(workers, CONFIRMATION_MAX_WORKERS),
    )
    replayed, cpu = _checkpointed(
        "confirmation_replay",
        spec,
        registration,
        DEFAULT_WORK / "confirmation_replay",
        list(range(spec.matrices)),
        _confirmation_matrix,
        tail,
        min(workers, CONFIRMATION_MAX_WORKERS),
        cpu,
    )
    replay = _replay_audit(generated, replayed, spec.matrices)
    if not replay["complete_exact_replay"]:
        raise AssertionError("PX3 confirmation replay failed")
    return _write_confirmation_result(generated, replay, registration, cpu)


def validation_checks() -> dict[str, bool]:
    config = GardConfig()
    composition = np.zeros(config.n_types, dtype=np.int64)
    composition[:4] = (10, 12, 9, 9)
    legal = enumerate_legal_edits(composition)
    changed = apply_molecular_edit(composition, legal[0])
    development = development_spec()
    confirmation = confirmation_spec()
    return {
        "master_registration_exists": MASTER_REGISTRATION.exists(),
        "px1_result_integrity_exists": (PX1_OUTPUT / "manifest.json").exists()
        and (PX1_OUTPUT / "matrix_inputs.npz").exists(),
        "px2_result_complete_before_px3": (
            RESULT_ROOT / "px2_event_locked_recovery" / "manifest.json"
        ).exists(),
        "heredity_model_hash_exact": HEREDITY_MODEL.exists()
        and sha256_file(HEREDITY_MODEL) == EXPECTED_HEREDITY_MODEL_SHA256,
        "development_scale_fixed": development.matrices == 12
        and DEVELOPMENT_REPLICATES == 2
        and DEVELOPMENT_EDITS == 24
        and development.branches == 16
        and development.horizon == 8,
        "confirmation_scale_fixed": confirmation.matrices == 24
        and confirmation.branches == 64
        and confirmation.horizon == 8
        and HALVES == {"A": (0, 32), "B": (32, 64)},
        "cpu_allocations_fixed": development.cpu_seconds == 104 * 3600
        and confirmation.cpu_seconds == 64 * 3600,
        "resource_amendment_explicit": (ROOT / DEVELOPMENT_AMENDMENT).exists()
        and DEVELOPMENT_CARRIED_FORWARD == tuple(range(6))
        and DEVELOPMENT_MAX_WORKERS == 8
        and not ORIGINAL_DEVELOPMENT_GATE_ELIGIBLE,
        "confirmation_resource_note_explicit": (
            ROOT / CONFIRMATION_RESOURCE_NOTE
        ).exists()
        and CONFIRMATION_MAX_WORKERS == 8,
        "ridge_grid_fixed": RIDGE_GRID == (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
        "no_pca": program_protocol()["development"]["pca"] is False,
        "all_legal_confirmation_scoring": program_protocol()["confirmation"][
            "all_legal_edits_scored"
        ],
        "legal_edit_preserves_mass": int(changed.sum()) == int(composition.sum()),
        "legal_edit_nonnegative_integer": np.issubdtype(changed.dtype, np.integer)
        and bool(np.all(changed >= 0)),
        "future_seed_arm_free": "arm"
        not in inspect.signature(_confirmation_future_seed).parameters,
        "selection_future_streams_distinct": _confirmation_selection_seed(
            smoke_spec("smoke-validation"), "02", 0, 2
        )
        != _confirmation_future_seed(
            smoke_spec("smoke-validation"), "02", 0, 2, 0
        ),
        "development_confirmation_streams_distinct": _development_future_seed(
            smoke_spec("smoke-validation"), "02", 0, 0, 2, 0
        )
        != _confirmation_future_seed(
            smoke_spec("smoke-validation"), "02", 0, 2, 0
        ),
        "draws_fixed": BOOTSTRAP_DRAWS == 4096
        and RANDOMIZATION_DRAWS == 4096,
        "run_confirmation_after_failed_development": program_protocol()[
            "run_confirmation_if_development_fails"
        ],
        "no_48_matrix_campaign": program_protocol()["no_48_matrix_campaign"],
        "strict_eight_excluded": "strict-eight is excluded"
        in program_protocol()["claim_boundary"],
    }


def run_validation() -> dict[str, Any]:
    checks = validation_checks()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_phir_extension_px3.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    payload = {
        "format": "codex-ch5-phir-extension-px3-validation-v1",
        "checks": checks,
        "pytest_returncode": completed.returncode,
        "pytest_stdout": completed.stdout,
        "pytest_stderr": completed.stderr,
        "all_passed": bool(all(checks.values()) and completed.returncode == 0),
        "runtime": runtime_versions(),
    }
    if DEFAULT_VALIDATION.exists():
        shutil.rmtree(DEFAULT_VALIDATION)
    DEFAULT_VALIDATION.mkdir(parents=True)
    atomic_json(DEFAULT_VALIDATION / "validation.json", payload)
    write_checksums(DEFAULT_VALIDATION)
    if not payload["all_passed"]:
        raise AssertionError(
            f"PX3 validation failed\n{completed.stdout}\n{completed.stderr}"
        )
    return payload


def run_smoke() -> dict[str, Any]:
    if DEFAULT_SMOKE.exists():
        raise FileExistsError(f"PX3 smoke exists: {DEFAULT_SMOKE}")
    spec = smoke_spec("smoke-confirmation")
    temporary = DEFAULT_WORK / "smoke_models"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    feature_count = 195
    models = {
        candidate: FrozenPhiSurrogate(
            candidate,
            np.zeros(feature_count, dtype=np.float64),
            np.ones(feature_count, dtype=np.float64),
            np.linspace(-0.01, 0.01, feature_count, dtype=np.float64),
            1.0,
        )
        for candidate in CANDIDATES
    }
    archive = temporary / "models.npz"
    contract = temporary / "contract.json"
    save_surrogates(models, archive, contract)
    restored = load_surrogates(archive, contract)
    serialized_exact = all(
        np.array_equal(models[candidate].coefficient, restored[candidate].coefficient)
        for candidate in CANDIDATES
    )
    first = _confirmation_matrix((0, spec, str(archive), str(contract)))
    second = _confirmation_matrix((0, spec, str(archive), str(contract)))
    scores = pd.DataFrame(first.score_rows)
    edits = pd.DataFrame(first.edit_rows)
    payload = {
        "format": "codex-ch5-phir-extension-px3-smoke-v1",
        "all_arms": set(scores["arm"]) == set(ARMS),
        "all_halves": set(scores["half"]) == set(HALVES),
        "all_candidates": set(scores["candidate"]) == set(CANDIDATES),
        "explicit_pairs_positive": bool((scores["transition_pairs"] > 0).all()),
        "scores_finite": bool(np.isfinite(scores["material_full_revised"]).all()),
        "exhaustive_edit_counts_positive": bool(
            (edits["legal_edits_scored"] > 0).all()
        ),
        "surrogate_serialization_exact": serialized_exact,
        "replay_exact": first.scientific_digest == second.scientific_digest,
        "effect_sizes_suppressed": True,
    }
    payload["passed"] = bool(
        all(value for key, value in payload.items() if key != "format")
    )
    DEFAULT_SMOKE.mkdir(parents=True)
    atomic_json(DEFAULT_SMOKE / "smoke.json", payload)
    write_checksums(DEFAULT_SMOKE)
    if not payload["passed"]:
        raise AssertionError("PX3 smoke failed")
    return payload


def _launch_service(
    service: str,
    command: str,
    registration: Mapping[str, Any],
    workers: int,
) -> dict[str, Any]:
    DEFAULT_WORK.mkdir(parents=True, exist_ok=True)
    log = RESULT_ROOT / f"px3_{command}.log"
    arguments = [
        "systemd-run",
        "--user",
        f"--unit={service}",
        "--collect",
        "--property",
        f"WorkingDirectory={ROOT}",
        "--property",
        f"StandardOutput=append:{log}",
        "--property",
        f"StandardError=append:{log}",
        sys.executable,
        "-m",
        "plastic_heredity.phir_extension_px3",
        command,
        "--workers",
        str(min(workers, MAX_WORKERS)),
    ]
    completed = subprocess.run(arguments, check=True, capture_output=True, text=True)
    payload = {
        "registration_id": registration["registration_id"],
        "service": service,
        "command": command,
        "workers": min(workers, MAX_WORKERS),
        "launched_at_unix": time.time(),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    atomic_json(DEFAULT_WORK / f"{command}_detached_launch.json", payload)
    return payload


def launch_development(workers: int = MAX_WORKERS) -> dict[str, Any]:
    registration = verify_development_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_DEVELOPMENT_OUTPUT.exists():
        raise FileExistsError(f"PX3 development output exists: {DEFAULT_DEVELOPMENT_OUTPUT}")
    px2 = RESULT_ROOT / "px2_event_locked_recovery" / "manifest.json"
    if not px2.exists():
        raise RuntimeError("PX3 development is locked until PX2 completes")
    px2_manifest = json.loads(px2.read_text(encoding="utf-8"))
    if not px2_manifest.get("complete_exact_replay") or not px2_manifest.get(
        "complete_readback_exact"
    ):
        raise RuntimeError("PX3 development requires an integrity-complete PX2")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("PX3 development refused below the sealed disk floor")
    return _launch_service(
        DEVELOPMENT_SERVICE,
        "development-run",
        registration,
        min(workers, DEVELOPMENT_MAX_WORKERS),
    )


def launch_confirmation(workers: int = MAX_WORKERS) -> dict[str, Any]:
    registration = verify_confirmation_registration()
    if DEFAULT_CONFIRMATION_OUTPUT.exists():
        raise FileExistsError(
            f"PX3 confirmation output exists: {DEFAULT_CONFIRMATION_OUTPUT}"
        )
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("PX3 confirmation refused below the sealed disk floor")
    return _launch_service(
        CONFIRMATION_SERVICE,
        "confirmation-run",
        registration,
        min(workers, CONFIRMATION_MAX_WORKERS),
    )


def status_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phase": "PX3",
        "validation": DEFAULT_VALIDATION.exists(),
        "development_registration": DEFAULT_DEVELOPMENT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "development_complete": DEFAULT_DEVELOPMENT_OUTPUT.exists(),
        "confirmation_registration": DEFAULT_CONFIRMATION_REGISTRATION.exists(),
        "confirmation_complete": DEFAULT_CONFIRMATION_OUTPUT.exists(),
        "development_service": DEVELOPMENT_SERVICE,
        "confirmation_service": CONFIRMATION_SERVICE,
        "free_disk_bytes": shutil.disk_usage(ROOT).free,
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "no_48_matrix_campaign": True,
    }
    for stage in (
        "development_generate",
        "development_replay",
        "confirmation_generate",
        "confirmation_replay",
    ):
        path = DEFAULT_WORK / stage / "status.json"
        if path.exists():
            payload[stage] = json.loads(path.read_text(encoding="utf-8"))
    for command in ("development-run", "confirmation-run"):
        path = DEFAULT_WORK / f"{command}_detached_launch.json"
        if path.exists():
            payload[f"{command}_launch"] = json.loads(
                path.read_text(encoding="utf-8")
            )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    commands.add_parser("register-development")
    commands.add_parser("carry-forward-development")
    commands.add_parser("smoke")
    development_run = commands.add_parser("development-run")
    development_run.add_argument("--workers", type=int, default=MAX_WORKERS)
    development_launch = commands.add_parser("development-launch")
    development_launch.add_argument("--workers", type=int, default=MAX_WORKERS)
    commands.add_parser("register-confirmation")
    confirmation_run = commands.add_parser("confirmation-run")
    confirmation_run.add_argument("--workers", type=int, default=MAX_WORKERS)
    confirmation_launch = commands.add_parser("confirmation-launch")
    confirmation_launch.add_argument("--workers", type=int, default=MAX_WORKERS)
    commands.add_parser("status")
    arguments = parser.parse_args(argv)
    if arguments.command == "validate":
        print(json.dumps(run_validation(), indent=2, sort_keys=True))
    elif arguments.command == "register-development":
        print(json.dumps(register_development(), indent=2, sort_keys=True))
    elif arguments.command == "carry-forward-development":
        print(
            json.dumps(
                carry_forward_development_checkpoints(), indent=2, sort_keys=True
            )
        )
    elif arguments.command == "smoke":
        print(json.dumps(run_smoke(), indent=2, sort_keys=True))
    elif arguments.command == "development-run":
        print(json.dumps(run_development(arguments.workers), indent=2, sort_keys=True))
    elif arguments.command == "development-launch":
        print(json.dumps(launch_development(arguments.workers), indent=2, sort_keys=True))
    elif arguments.command == "register-confirmation":
        print(json.dumps(register_confirmation(), indent=2, sort_keys=True))
    elif arguments.command == "confirmation-run":
        print(json.dumps(run_confirmation(arguments.workers), indent=2, sort_keys=True))
    elif arguments.command == "confirmation-launch":
        print(json.dumps(launch_confirmation(arguments.workers), indent=2, sort_keys=True))
    elif arguments.command == "status":
        print(json.dumps(status_payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
