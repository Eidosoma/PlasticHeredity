"""Read-only adapters for the two retained clean-room implementations."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nuisance_core import EPSILON, event_from_h, sha256_file, sigmoid


TASK_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_ROOT.parent.parent
CODEX_ROOT = REPOSITORY_ROOT / "replicators.13.8.2026.codex"
FABLE_ROOT = REPOSITORY_ROOT / "replicators.13.8.2026.fable" / "replication"
ARTIFACTS = TASK_ROOT / "artifacts"
REPLAY_DIR = ARTIFACTS / "replays"


@dataclass(frozen=True)
class CohortSpec:
    key: str
    implementation: str
    role: str
    pipeline: str
    history_dimension: int
    state_dimension: int
    development_matrices: int
    confirmation_matrices: int
    source: str
    development_tag: str | None = None
    confirmation_tag: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


COHORTS: dict[str, CohortSpec] = {
    "codex_headline": CohortSpec(
        "codex_headline", "independent_test_1_codex", "matched_40_matrix_headline",
        "codex", 9, 195, 40, 40, "full"
    ),
    "fable_headline": CohortSpec(
        "fable_headline", "independent_test_2_fable", "matched_40_matrix_headline",
        "fable", 9, 195, 40, 40, "results", "2026-08-13", "2026-08-13"
    ),
    "codex_primary": CohortSpec(
        "codex_primary", "independent_test_1_codex", "scaled_robustness",
        "codex", 9, 195, 200, 200, "scaled5"
    ),
    "fable_primary": CohortSpec(
        "fable_primary", "independent_test_2_fable", "scaled_revised_robustness",
        "fable", 8, 142, 1000, 200, "results_v2",
        "25x-2026-08-13", "v2-conf-2026-08-13"
    ),
}


def source_paths() -> dict[str, Path]:
    paths = {
        "analysis_runner": TASK_ROOT / "run_analysis.py",
        "analysis_adapters": TASK_ROOT / "adapters.py",
        "analysis_core": TASK_ROOT / "nuisance_core.py",
        "analysis_reporting": TASK_ROOT / "reporting.py",
        "analysis_tests": TASK_ROOT / "test_nuisance_control.py",
        "codex_models_source": CODEX_ROOT / "plastic_heredity" / "models.py",
        "codex_full_arrays": CODEX_ROOT / "results/full/analysis_arrays.npz",
        "codex_full_models": CODEX_ROOT / "results/full/frozen_models.npz",
        "codex_full_states": CODEX_ROOT / "results/full/confirmation_states.csv",
        "codex_full_development": CODEX_ROOT / "results/full/development_branches.csv.gz",
        "codex_scaled_arrays": CODEX_ROOT / "results/scaled5/analysis_arrays.npz",
        "codex_scaled_models": CODEX_ROOT / "results/scaled5/frozen_models.npz",
        "codex_scaled_states": CODEX_ROOT / "results/scaled5/confirmation_states.csv",
        "codex_scaled_development": CODEX_ROOT / "results/scaled5/development_branches.csv.gz",
        "fable_cohort_source": FABLE_ROOT / "cohort.py",
        "fable_feature_source": FABLE_ROOT / "features.py",
        "fable_model_source": FABLE_ROOT / "models.py",
        "fable_v2_source": FABLE_ROOT / "registry_v2.py",
        "fable_headline_models": FABLE_ROOT / "results/frozen_models.pkl",
        "fable_headline_outcomes": FABLE_ROOT / "results/conf_data.pkl",
        "fable_headline_summary": FABLE_ROOT / "results/dev_summary.json",
        "fable_primary_models": FABLE_ROOT / "results_v2/frozen_models_v2.pkl",
        "fable_primary_outcomes": FABLE_ROOT / "results_sensitivity/v2_cohort.pkl",
        "fable_primary_summary": FABLE_ROOT / "results_25x/dev_summary.json",
        "fable_primary_results": FABLE_ROOT / "results_v2/v2_results.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source artifacts: {missing}")
    return paths


def source_contract() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in source_paths().items()
    }


def atomic_npz(path: Path, **arrays: NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _candidate_label(value: Any) -> str:
    return f"{int(value):02d}"


def _codex_transform(
    archive: Any, candidate: str, state: NDArray[np.float64]
) -> NDArray[np.float64]:
    prefix = f"c{candidate}__full_state"
    scaled = (state - archive[f"{prefix}_scaler_mean"]) / archive[f"{prefix}_scaler_scale"]
    return (scaled - archive[f"{prefix}_pca_mean"]) @ archive[
        f"{prefix}_pca_components"
    ].T


def _codex_predictions(
    archive: Any,
    candidate: str,
    state: NDArray[np.float64],
    history: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    base = f"c{candidate}"
    scaled_history = (history - archive[f"{base}__history__scaler_mean"]) / archive[
        f"{base}__history__scaler_scale"
    ]
    direct = sigmoid(
        (scaled_history @ archive[f"{base}__history__classifier_coef"].T).reshape(-1)
        + float(archive[f"{base}__history__classifier_intercept"][0])
    )
    full = np.column_stack((_codex_transform(archive, candidate, state), history))
    full = (full - archive[f"{base}__full__scaler_mean"]) / archive[
        f"{base}__full__scaler_scale"
    ]
    aligned = sigmoid(
        (full @ archive[f"{base}__full__classifier_coef"].T).reshape(-1)
        + float(archive[f"{base}__full__classifier_intercept"][0])
    )
    return direct, aligned


def prepare_codex_replay(spec: CohortSpec) -> dict[str, Any]:
    source = CODEX_ROOT / "results" / spec.source
    branches = pd.read_csv(
        source / "development_branches.csv.gz",
        usecols=["state_id", "candidate", "matrix_id", "landmark", "joint_break_run3"],
    )
    development_meta = branches.drop_duplicates("state_id", keep="first").reset_index(drop=True)
    confirmation_meta = pd.read_csv(source / "confirmation_states.csv")
    audit: dict[str, Any] = {"cohort": spec.key, "candidates": {}}
    with np.load(source / "analysis_arrays.npz", allow_pickle=False) as arrays, np.load(
        source / "frozen_models.npz", allow_pickle=False
    ) as models:
        if len(development_meta) != arrays["development_state_graph"].shape[0]:
            raise AssertionError(f"{spec.key}: development metadata length mismatch")
        if len(confirmation_meta) != arrays["confirmation_state_graph"].shape[0]:
            raise AssertionError(f"{spec.key}: confirmation metadata length mismatch")
        grouped_q = branches.groupby("state_id", sort=False)["joint_break_run3"].mean().to_numpy()
        if not np.array_equal(grouped_q, arrays["development_targets"].mean(axis=1)):
            raise AssertionError(f"{spec.key}: development target ordering mismatch")
        if not np.allclose(
            confirmation_meta["q_all"].to_numpy(),
            arrays["confirmation_targets"].mean(axis=1),
            atol=1e-15,
            rtol=0.0,
        ):
            raise AssertionError(f"{spec.key}: confirmation target ordering mismatch")

        for candidate in ("02", "03"):
            dev_select = development_meta["candidate"].map(_candidate_label).to_numpy() == candidate
            conf_select = confirmation_meta["candidate"].map(_candidate_label).to_numpy() == candidate
            dev_state = np.asarray(arrays["development_state_graph"][dev_select], dtype=np.float64)
            dev_history = np.asarray(arrays["development_history"][dev_select], dtype=np.float64)
            dev_targets = np.asarray(arrays["development_targets"][dev_select], dtype=np.int8)
            conf_state = np.asarray(arrays["confirmation_state_graph"][conf_select], dtype=np.float64)
            conf_history = np.asarray(arrays["confirmation_history"][conf_select], dtype=np.float64)
            dev_components = _codex_transform(models, candidate, dev_state)
            conf_components = _codex_transform(models, candidate, conf_state)
            direct, aligned = _codex_predictions(models, candidate, conf_state, conf_history)
            stored = confirmation_meta.loc[conf_select]
            direct_error = float(np.max(np.abs(direct - stored["prediction_history"].to_numpy())))
            aligned_error = float(np.max(np.abs(aligned - stored["prediction_full"].to_numpy())))
            if direct_error > 2e-12 or aligned_error > 2e-12:
                raise AssertionError(f"{spec.key} c{candidate}: frozen prediction replay failed")
            dev_rows, conf_rows = development_meta.loc[dev_select], confirmation_meta.loc[conf_select]
            atomic_npz(
                REPLAY_DIR / f"{spec.key}_c{candidate}.npz",
                dev_history=dev_history,
                dev_components=dev_components,
                dev_targets=dev_targets,
                dev_matrix=dev_rows["matrix_id"].to_numpy(dtype=np.int32),
                dev_group=dev_rows["landmark"].to_numpy(dtype=np.int16),
                conf_history=conf_history,
                conf_components=conf_components,
                conf_matrix=conf_rows["matrix_id"].to_numpy(dtype=np.int32),
                conf_group=conf_rows["landmark"].to_numpy(dtype=np.int16),
                conf_direct=direct,
                conf_aligned=aligned,
            )
            audit["candidates"][candidate] = {
                "development_rows": int(dev_state.shape[0]),
                "confirmation_rows": int(conf_state.shape[0]),
                "development_state_sha256": hashlib.sha256(
                    np.ascontiguousarray(dev_state).tobytes()
                ).hexdigest(),
                "direct_prediction_max_abs_error": direct_error,
                "aligned_prediction_max_abs_error": aligned_error,
                "history_dimension": int(dev_history.shape[1]),
                "component_dimension": int(dev_components.shape[1]),
            }
    return audit


def _import_fable() -> tuple[Any, Any]:
    root = str(FABLE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import cohort  # type: ignore
    import features  # type: ignore

    return cohort, features


def _fable_dev_worker(arguments: tuple[str, int, str]) -> dict[str, Any]:
    tag, matrix_id, candidate = arguments
    cohort, _ = _import_fable()
    cohort.DEV_ENTROPY = cohort.domain_entropy("dev", tag)
    return cohort.dev_unit((matrix_id, candidate))


def _fable_conf_worker(arguments: tuple[str, int, str]) -> dict[str, Any]:
    tag, matrix_id, candidate = arguments
    cohort, _ = _import_fable()
    cohort.CONF_ENTROPY = cohort.domain_entropy("confirmation", tag)
    return cohort.conf_features_unit((matrix_id, candidate))


def _map_jobs(
    function: Callable[[Any], Any], jobs: list[Any], workers: int
) -> list[Any]:
    if workers <= 1:
        return [function(job) for job in jobs]
    with get_context("fork").Pool(processes=workers) as pool:
        chunk = max(1, len(jobs) // (workers * 8))
        return list(pool.imap(function, jobs, chunksize=chunk))


def _load_fable_bundles(spec: CohortSpec) -> dict[str, Any]:
    path = (
        FABLE_ROOT / "results/frozen_models.pkl"
        if spec.key == "fable_headline"
        else FABLE_ROOT / "results_v2/frozen_models_v2.pkl"
    )
    with path.open("rb") as handle:
        return pickle.load(handle)


def _fable_design(
    spec: CohortSpec,
    bundle: dict[str, Any],
    x9: NDArray[np.float64],
    x195: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if spec.key == "fable_headline":
        return (
            bundle["sc9"].transform(x9),
            bundle["pca"].transform(bundle["sc195"].transform(x195)),
        )
    _, features = _import_fable()
    direct_columns = np.asarray((0, 1, 2, 3, 4, 5, 7, 8), dtype=np.int64)
    beta_indices = np.asarray(features.beta_conditioned_indices(), dtype=np.int64)
    return (
        bundle["sc8"].transform(x9[:, direct_columns]),
        bundle["pca"].transform(bundle["scb"].transform(x195[:, beta_indices])),
    )


def _fable_predictions(
    spec: CohortSpec,
    bundle: dict[str, Any],
    history: NDArray[np.float64],
    components: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    direct_model = bundle["direct"] if spec.key == "fable_headline" else bundle["direct8"]
    full_model = bundle["full"] if spec.key == "fable_headline" else bundle["v2"]
    direct = np.clip(direct_model.predict_proba(history)[:, 1], EPSILON, 1.0 - EPSILON)
    aligned = np.clip(
        full_model.predict_proba(np.column_stack((components, history)))[:, 1],
        EPSILON,
        1.0 - EPSILON,
    )
    return direct, aligned


def _fable_confirmation_features(
    spec: CohortSpec, candidate: str, workers: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32], NDArray[np.int16], list[dict[str, Any]]]:
    if spec.key == "fable_headline":
        jobs = [
            (str(spec.confirmation_tag), matrix_id, candidate)
            for matrix_id in range(spec.confirmation_matrices)
        ]
        units = _map_jobs(_fable_conf_worker, jobs, workers)
        feature_lookup = {
            (int(state["matrix"]), int(state["landmark"])): state
            for unit in units
            for state in unit["states"]
        }
        with (FABLE_ROOT / "results/conf_data.pkl").open("rb") as handle:
            rows = [row for row in pickle.load(handle)["table"] if row["candidate"] == candidate]
        rows.sort(key=lambda row: (int(row["matrix"]), int(row["landmark"])))
        x9 = np.stack(
            [feature_lookup[(int(row["matrix"]), int(row["landmark"]))]["X9"] for row in rows]
        ).astype(np.float64)
        x195 = np.stack(
            [feature_lookup[(int(row["matrix"]), int(row["landmark"]))]["X195"] for row in rows]
        ).astype(np.float64)
    else:
        with (FABLE_ROOT / "results_sensitivity/v2_cohort.pkl").open("rb") as handle:
            rows = [row for row in pickle.load(handle)["table"] if row["candidate"] == candidate]
        rows.sort(key=lambda row: (int(row["matrix"]), int(row["landmark"])))
        x9 = np.stack([row["X9"] for row in rows]).astype(np.float64)
        x195 = np.stack([row["X195"] for row in rows]).astype(np.float64)
    matrices = np.asarray([row["matrix"] for row in rows], dtype=np.int32)
    groups = np.asarray([row["landmark"] for row in rows], dtype=np.int16)
    return x9, x195, matrices, groups, rows


def prepare_fable_replay(spec: CohortSpec, workers: int) -> dict[str, Any]:
    bundles = _load_fable_bundles(spec)
    summary_path = (
        FABLE_ROOT / "results/dev_summary.json"
        if spec.key == "fable_headline"
        else FABLE_ROOT / "results_25x/dev_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit: dict[str, Any] = {"cohort": spec.key, "candidates": {}}
    for candidate in ("02", "03"):
        started = time.time()
        jobs = [
            (str(spec.development_tag), matrix_id, candidate)
            for matrix_id in range(spec.development_matrices)
        ]
        units = _map_jobs(_fable_dev_worker, jobs, workers)
        units.sort(key=lambda unit: int(unit["matrix"]))
        nonempty = [unit for unit in units if len(unit["y"]) > 0]
        x9 = np.vstack([unit["X9"] for unit in nonempty]).astype(np.float64)
        x195 = np.vstack([unit["X195"] for unit in nonempty]).astype(np.float64)
        targets = np.concatenate([unit["y"] for unit in nonempty]).astype(np.int8)
        matrices = np.concatenate(
            [np.full(len(unit["y"]), int(unit["matrix"]), dtype=np.int32) for unit in nonempty]
        )
        groups = np.concatenate([unit["g"].astype(np.int16) for unit in nonempty])
        if targets.size != int(summary[candidate]["n_examples"]):
            raise AssertionError(f"{spec.key} c{candidate}: development row count mismatch")
        if abs(float(targets.mean()) - float(summary[candidate]["prevalence"])) > 1e-15:
            raise AssertionError(f"{spec.key} c{candidate}: development target mismatch")
        history, components = _fable_design(spec, bundles[candidate], x9, x195)
        conf_x9, conf_x195, conf_matrices, conf_groups, retained_rows = (
            _fable_confirmation_features(spec, candidate, workers)
        )
        conf_history, conf_components = _fable_design(
            spec, bundles[candidate], conf_x9, conf_x195
        )
        direct, aligned = _fable_predictions(
            spec, bundles[candidate], conf_history, conf_components
        )
        direct_error = aligned_error = 0.0
        if spec.key == "fable_headline":
            direct_error = float(
                np.max(np.abs(direct - np.asarray([row["p_direct"] for row in retained_rows])))
            )
            aligned_error = float(
                np.max(np.abs(aligned - np.asarray([row["p_full"] for row in retained_rows])))
            )
            if direct_error > 2e-12 or aligned_error > 2e-12:
                raise AssertionError(f"{spec.key} c{candidate}: frozen prediction replay failed")
        atomic_npz(
            REPLAY_DIR / f"{spec.key}_c{candidate}.npz",
            dev_history=history,
            dev_components=components,
            dev_targets=targets,
            dev_matrix=matrices,
            dev_group=groups,
            conf_history=conf_history,
            conf_components=conf_components,
            conf_matrix=conf_matrices,
            conf_group=conf_groups,
            conf_direct=direct,
            conf_aligned=aligned,
        )
        audit["candidates"][candidate] = {
            "development_rows": int(targets.size),
            "confirmation_rows": int(conf_history.shape[0]),
            "development_state_sha256": hashlib.sha256(
                np.ascontiguousarray(x195).tobytes()
            ).hexdigest(),
            "direct_prediction_max_abs_error": direct_error,
            "aligned_prediction_max_abs_error": aligned_error,
            "history_dimension": int(history.shape[1]),
            "component_dimension": int(components.shape[1]),
            "development_replay_seconds": time.time() - started,
        }
        print(
            f"  {spec.key} c{candidate}: {targets.size} development rows, "
            f"{conf_history.shape[0]} confirmation rows",
            flush=True,
        )
        del units, nonempty, x9, x195
    return audit


def confirmation_targets(spec: CohortSpec, candidate: str) -> NDArray[np.int8]:
    if spec.pipeline == "codex":
        source = CODEX_ROOT / "results" / spec.source
        metadata = pd.read_csv(source / "confirmation_states.csv")
        selected = metadata["candidate"].map(_candidate_label).to_numpy() == candidate
        with np.load(source / "analysis_arrays.npz", allow_pickle=False) as archive:
            return np.asarray(archive["confirmation_targets"][selected], dtype=np.int8)
    path = (
        FABLE_ROOT / "results/conf_data.pkl"
        if spec.key == "fable_headline"
        else FABLE_ROOT / "results_sensitivity/v2_cohort.pkl"
    )
    with path.open("rb") as handle:
        rows = [row for row in pickle.load(handle)["table"] if row["candidate"] == candidate]
    rows.sort(key=lambda row: (int(row["matrix"]), int(row["landmark"])))
    if spec.key == "fable_headline":
        return np.stack([row["y64"] for row in rows]).astype(np.int8)
    output = np.zeros((len(rows), 64), dtype=np.int8)
    for row_index, row in enumerate(rows):
        for branch in range(64):
            length = int(row["lens"][branch])
            output[row_index, branch] = int(event_from_h(row["H64"][branch, :length]))
    return output
