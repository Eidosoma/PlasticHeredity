"""Read-only clean-room adapters and deterministic main-path replay.

The adapter deliberately keeps confirmation outcomes out of replay files.
`load_confirmation_outcomes` is called only by the analyze stage.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from reviewer_sequence_history_response.sequence_core import (
    EPSILON,
    HORIZON,
    event_from_flags,
    sha256_file,
)


TASK_ROOT = Path(__file__).resolve().parent
CODEX_ROOT = TASK_ROOT.parent
REPOSITORY_ROOT = CODEX_ROOT.parent
FABLE_ROOT = REPOSITORY_ROOT / "replicators.13.8.2026.fable" / "replication"
WORK_ROOT = TASK_ROOT / "artifacts" / "work"
REPLAY_ROOT = TASK_ROOT / "artifacts" / "replays"


@dataclass(frozen=True)
class CohortSpec:
    key: str
    implementation: str
    role: str
    development_matrices: int
    confirmation_matrices: int
    development_tag: str
    confirmation_tag: str
    direct_columns: tuple[int, ...]
    source_directory: str
    development_source_directory: str
    composite_label: str
    direct_label: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


COHORTS: dict[str, CohortSpec] = {
    "codex_primary": CohortSpec(
        key="codex_primary",
        implementation="independent_test_1_codex",
        role="primary",
        development_matrices=200,
        confirmation_matrices=200,
        development_tag="VALI",
        confirmation_tag="CONF",
        direct_columns=tuple(range(9)),
        source_directory=str(CODEX_ROOT / "results" / "scaled5"),
        development_source_directory=str(CODEX_ROOT / "results" / "scaled5"),
        composite_label="frozen_composite",
        direct_label="registered_history9",
    ),
    "codex_headline": CohortSpec(
        key="codex_headline",
        implementation="independent_test_1_codex",
        role="secondary_headline",
        development_matrices=40,
        confirmation_matrices=40,
        development_tag="VALI",
        confirmation_tag="CONF",
        direct_columns=tuple(range(9)),
        source_directory=str(CODEX_ROOT / "results" / "full"),
        development_source_directory=str(CODEX_ROOT / "results" / "full"),
        composite_label="frozen_composite",
        direct_label="registered_history9",
    ),
    "fable_primary": CohortSpec(
        key="fable_primary",
        implementation="independent_test_2_fable",
        role="primary",
        development_matrices=1000,
        confirmation_matrices=200,
        development_tag="25x-2026-08-13",
        confirmation_tag="v2-conf-2026-08-13",
        direct_columns=(0, 1, 2, 3, 4, 5, 7, 8),
        source_directory=str(FABLE_ROOT / "results_v2"),
        development_source_directory=str(FABLE_ROOT / "results_25x"),
        composite_label="frozen_v2",
        direct_label="registered_direct8",
    ),
    "fable_headline": CohortSpec(
        key="fable_headline",
        implementation="independent_test_2_fable",
        role="secondary_headline",
        development_matrices=40,
        confirmation_matrices=40,
        development_tag="2026-08-13",
        confirmation_tag="2026-08-13",
        direct_columns=tuple(range(9)),
        source_directory=str(FABLE_ROOT / "results"),
        development_source_directory=str(FABLE_ROOT / "results"),
        composite_label="frozen_full",
        direct_label="registered_direct9",
    ),
}


def source_files() -> dict[str, Path]:
    """Files whose identity defines the read-only source contract."""

    paths: dict[str, Path] = {
        "codex_experiment": CODEX_ROOT / "plastic_heredity" / "experiment.py",
        "codex_simulator": CODEX_ROOT / "plastic_heredity" / "simulator.py",
        "codex_features": CODEX_ROOT / "plastic_heredity" / "features.py",
        "fable_cohort": FABLE_ROOT / "cohort.py",
        "fable_simulator": FABLE_ROOT / "sim.py",
        "fable_features": FABLE_ROOT / "features.py",
        "fable_registry_v2": FABLE_ROOT / "registry_v2.py",
        "codex_full_arrays": CODEX_ROOT / "results" / "full" / "analysis_arrays.npz",
        "codex_full_models": CODEX_ROOT / "results" / "full" / "frozen_models.npz",
        "codex_full_states": CODEX_ROOT / "results" / "full" / "confirmation_states.csv",
        "codex_full_metrics": CODEX_ROOT / "results" / "full" / "metrics.json",
        "codex_scaled_arrays": CODEX_ROOT / "results" / "scaled5" / "analysis_arrays.npz",
        "codex_scaled_models": CODEX_ROOT / "results" / "scaled5" / "frozen_models.npz",
        "codex_scaled_states": CODEX_ROOT / "results" / "scaled5" / "confirmation_states.csv",
        "codex_scaled_metrics": CODEX_ROOT / "results" / "scaled5" / "metrics.json",
        "fable_headline_conf": FABLE_ROOT / "results" / "conf_data.pkl",
        "fable_headline_models": FABLE_ROOT / "results" / "frozen_models.pkl",
        "fable_headline_metrics": FABLE_ROOT / "results" / "confirmation_metrics.json",
        "fable_headline_dev_summary": FABLE_ROOT / "results" / "dev_summary.json",
        "fable_v2_cohort": FABLE_ROOT / "results_sensitivity" / "v2_cohort.pkl",
        "fable_v2_models": FABLE_ROOT / "results_v2" / "frozen_models_v2.pkl",
        "fable_v2_results": FABLE_ROOT / "results_v2" / "v2_results.json",
        "fable_25x_models": FABLE_ROOT / "results_25x" / "frozen_models.pkl",
        "fable_25x_dev_summary": FABLE_ROOT / "results_25x" / "dev_summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required source artifacts: {missing}")
    return paths


def source_contract() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in source_files().items()
    }


def atomic_npz(path: Path, **arrays: NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _codex_replay_unit(arguments: tuple[str, str, int]) -> dict[str, Any]:
    profile, candidate, matrix_id = arguments
    from plastic_heredity.config import CANDIDATES, ExperimentConfig
    from plastic_heredity.features import history_features
    from plastic_heredity.seeds import derive_seed
    from plastic_heredity.simulator import (
        SimulationError,
        generate_beta,
        generate_initial_composition,
        simulate_lineage,
    )

    experiment = ExperimentConfig.scaled5() if profile == "scaled5" else ExperimentConfig()
    def run_for(cohort_name: str) -> tuple[NDArray[np.float64], list[dict[str, Any]]]:
        brng = np.random.default_rng(
            derive_seed(experiment.master_seed, f"{cohort_name}.beta", matrix_id)
        )
        irng = np.random.default_rng(
            derive_seed(experiment.master_seed, f"{cohort_name}.initial", matrix_id)
        )
        beta = generate_beta(experiment.gard, brng)
        initial = generate_initial_composition(experiment.gard, irng)
        lineage = None
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
                    initial, beta, experiment.gard, CANDIDATES[candidate], rng
                )
                break
            except SimulationError:
                continue
        if lineage is None:
            raise SimulationError(
                f"failed exact {cohort_name} replay for c{candidate} m{matrix_id}"
            )
        by_generation = {snapshot.generation: snapshot for snapshot in lineage}
        config = experiment.development if cohort_name == "VALI" else experiment.confirmation
        states: list[dict[str, Any]] = []
        for landmark in config.landmarks:
            snapshot = by_generation[landmark]
            states.append(
                {
                    "key": f"{cohort_name}-c{candidate}-m{matrix_id:03d}-g{landmark:03d}",
                    "landmark": landmark,
                    "direct": history_features(snapshot, experiment.gard),
                    "history": np.asarray(snapshot.boundary_h, dtype=np.float64),
                }
            )
        return np.asarray(lineage[-1].boundary_h, dtype=np.float64), states

    development_h, development_states = run_for("VALI")
    confirmation_h, confirmation_states = run_for("CONF")
    return {
        "candidate": candidate,
        "matrix_id": matrix_id,
        "development_h": development_h,
        "development_died": False,
        "development_states": development_states,
        "confirmation_h": confirmation_h,
        "confirmation_died": False,
        "confirmation_states": confirmation_states,
    }


def _import_fable() -> tuple[Any, Any, Any]:
    path = str(FABLE_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    import cohort  # type: ignore
    import features  # type: ignore
    import sim  # type: ignore

    return cohort, features, sim


def _fable_path(
    matrix_id: int, candidate: str, entropy: int
) -> tuple[dict[str, Any], Any, Any, Any]:
    cohort, features, sim = _import_fable()
    candidate_index = cohort.CANDIDATES.index(candidate)
    beta, initial = cohort.matrix_and_init(entropy, matrix_id)
    rng = cohort._rng(entropy, 2, candidate_index, matrix_id)
    trajectory = sim.run_fissions(initial, beta, candidate, cohort.N_FISSIONS, rng)
    return trajectory, beta, features, cohort


def _fable_replay_unit(
    arguments: tuple[str, str, str, int, bool, bool]
) -> dict[str, Any]:
    development_tag, confirmation_tag, candidate, matrix_id, do_development, do_confirmation = arguments
    cohort, _, _ = _import_fable()
    dev_entropy = cohort.domain_entropy("dev", development_tag)
    conf_entropy = cohort.domain_entropy("confirmation", confirmation_tag)

    dev_states: list[dict[str, Any]] = []
    hs = np.empty(0, dtype=np.float64)
    dev_died = False
    if do_development:
        dev, _dev_beta, features, cohort = _fable_path(matrix_id, candidate, dev_entropy)
        hs = np.asarray(dev["H"], dtype=np.float64)
        inherited = np.asarray(dev["inherited"], dtype=bool)
        daughters = np.asarray(dev["daughters"])
        dev_died = bool(dev["died"])
        for generation in range(1, int(dev["n_done"]) - HORIZON + 1):
            target = int(features.joint_break_run3(inherited[generation : generation + HORIZON]))
            dev_states.append(
                {
                    "key": f"DEV-c{candidate}-m{matrix_id:04d}-g{generation:03d}",
                    "landmark": generation,
                    "direct": features.direct9(
                        generation,
                        cohort.N_FISSIONS,
                        hs[:generation],
                        int(daughters[generation - 1].sum()),
                    ),
                    "history": hs[:generation].copy(),
                    "target": target,
                }
            )

    conf_h = np.empty(0, dtype=np.float64)
    conf_states: list[dict[str, Any]] = []
    conf_died = False
    if do_confirmation:
        conf, _conf_beta, features, cohort = _fable_path(matrix_id, candidate, conf_entropy)
        conf_h = np.asarray(conf["H"], dtype=np.float64)
        conf_daughters = np.asarray(conf["daughters"])
        conf_died = bool(conf["died"])
        for landmark in cohort.LANDMARKS:
            if landmark > int(conf["n_done"]):
                continue
            conf_states.append(
                {
                    "key": f"CONF-c{candidate}-m{matrix_id:04d}-g{landmark:03d}",
                    "landmark": landmark,
                    "direct": features.direct9(
                        landmark,
                        cohort.N_FISSIONS,
                        conf_h[:landmark],
                        int(conf_daughters[landmark - 1].sum()),
                    ),
                    "history": conf_h[:landmark].copy(),
                }
            )
    return {
        "candidate": candidate,
        "matrix_id": matrix_id,
        "development_h": hs,
        "development_died": dev_died,
        "development_states": dev_states,
        "confirmation_h": conf_h,
        "confirmation_died": conf_died,
        "confirmation_states": conf_states,
    }


def _checkpoint_path(spec: CohortSpec, candidate: str, matrix_id: int) -> Path:
    return WORK_ROOT / spec.key / f"c{candidate}-m{matrix_id:04d}.npz"


def _pack_checkpoint(
    path: Path, result: dict[str, Any], protocol_id: str
) -> None:
    arrays: dict[str, NDArray[Any]] = {
        "protocol_id": np.asarray([protocol_id]),
        "candidate": np.asarray([result["candidate"]]),
        "matrix_id": np.asarray([result["matrix_id"]], dtype=np.int32),
    }
    for split in ("development", "confirmation"):
        states = result[f"{split}_states"]
        maximum = max([len(state["history"]) for state in states] + [1])
        history = np.zeros((len(states), maximum), dtype=np.float64)
        lengths = np.zeros(len(states), dtype=np.int16)
        for index, state in enumerate(states):
            length = len(state["history"])
            history[index, :length] = state["history"]
            lengths[index] = length
        trajectory = np.asarray(result[f"{split}_h"], dtype=np.float64)
        arrays.update(
            {
                f"{split}_trajectory_h": trajectory,
                f"{split}_trajectory_length": np.asarray([len(trajectory)], dtype=np.int16),
                f"{split}_trajectory_died": np.asarray(
                    [result[f"{split}_died"]], dtype=np.int8
                ),
                f"{split}_keys": np.asarray([state["key"] for state in states]),
                f"{split}_landmarks": np.asarray(
                    [state["landmark"] for state in states], dtype=np.int16
                ),
                f"{split}_direct": np.asarray(
                    [state["direct"] for state in states], dtype=np.float64
                ).reshape(len(states), 9),
                f"{split}_history_h": history,
                f"{split}_history_length": lengths,
            }
        )
        if split == "development" and states and "target" in states[0]:
            arrays["development_replayed_target"] = np.asarray(
                [state["target"] for state in states], dtype=np.int8
            )[:, None]
    atomic_npz(path, **arrays)


def _valid_checkpoint(path: Path, protocol_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return str(archive["protocol_id"][0]) == protocol_id
    except Exception:
        return False


def replay_cohort(
    spec: CohortSpec,
    *,
    protocol_id: str,
    workers: int,
    progress: callable | None = None,
) -> dict[str, Any]:
    """Run or resume per-matrix natural-path replay, then consolidate."""

    profile = "scaled5" if spec.key == "codex_primary" else "full"
    maximum_matrices = max(spec.development_matrices, spec.confirmation_matrices)
    jobs = [
        (candidate, matrix_id)
        for candidate in ("02", "03")
        for matrix_id in range(maximum_matrices)
    ]
    pending = [
        item
        for item in jobs
        if not _valid_checkpoint(_checkpoint_path(spec, *item), protocol_id)
    ]
    worker = _codex_replay_unit if spec.key.startswith("codex") else _fable_replay_unit
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {}
            for candidate, matrix_id in pending:
                if spec.key.startswith("codex"):
                    arguments: Any = (profile, candidate, matrix_id)
                else:
                    arguments = (
                        spec.development_tag,
                        spec.confirmation_tag,
                        candidate,
                        matrix_id,
                        matrix_id < spec.development_matrices,
                        matrix_id < spec.confirmation_matrices,
                    )
                futures[executor.submit(worker, arguments)] = (candidate, matrix_id)
            completed = 0
            for future in as_completed(futures):
                candidate, matrix_id = futures[future]
                result = future.result()
                _pack_checkpoint(
                    _checkpoint_path(spec, candidate, matrix_id), result, protocol_id
                )
                completed += 1
                if progress is not None and (completed % 20 == 0 or completed == len(pending)):
                    progress(spec.key, completed, len(pending))
    development = _consolidate(spec, "development", protocol_id)
    confirmation = _consolidate(spec, "confirmation", protocol_id)
    return {
        "cohort": spec.key,
        "pending_completed": len(pending),
        "checkpoints": len(jobs),
        "development_states": development["states"],
        "confirmation_states": confirmation["states"],
        "development_trajectories": development["trajectories"],
    }


def _codex_development_target_map(spec: CohortSpec) -> dict[str, NDArray[np.int8]]:
    source = Path(spec.development_source_directory)
    with np.load(source / "analysis_arrays.npz", allow_pickle=False) as archive:
        targets = np.asarray(archive["development_targets"], dtype=np.int8)
    output: dict[str, NDArray[np.int8]] = {}
    index = 0
    for matrix_id in range(spec.development_matrices):
        for candidate in ("02", "03"):
            for landmark in (20, 35, 50, 65, 80):
                key = f"VALI-c{candidate}-m{matrix_id:03d}-g{landmark:03d}"
                output[key] = targets[index]
                index += 1
    if index != targets.shape[0]:
        raise AssertionError("Codex development source ordering mismatch")
    return output


def _consolidate(
    spec: CohortSpec, split: str, protocol_id: str
) -> dict[str, int]:
    matrix_limit = (
        spec.development_matrices if split == "development" else spec.confirmation_matrices
    )
    checkpoints = [
        _checkpoint_path(spec, candidate, matrix_id)
        for candidate in ("02", "03")
        for matrix_id in range(matrix_limit)
    ]
    if not all(_valid_checkpoint(path, protocol_id) for path in checkpoints):
        raise RuntimeError(f"incomplete {spec.key} {split} replay")
    state_rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for path in checkpoints:
        with np.load(path, allow_pickle=False) as archive:
            candidate = str(archive["candidate"][0])
            matrix_id = int(archive["matrix_id"][0])
            t_length = int(archive[f"{split}_trajectory_length"][0])
            trajectories.append(
                {
                    "candidate": candidate,
                    "matrix_id": matrix_id,
                    "h": np.asarray(
                        archive[f"{split}_trajectory_h"][:t_length], dtype=np.float64
                    ),
                    "died": bool(archive[f"{split}_trajectory_died"][0]),
                }
            )
            keys = archive[f"{split}_keys"]
            for state_index, key in enumerate(keys):
                length = int(archive[f"{split}_history_length"][state_index])
                row = {
                    "key": str(key),
                    "candidate": candidate,
                    "matrix_id": matrix_id,
                    "landmark": int(archive[f"{split}_landmarks"][state_index]),
                    "direct": np.asarray(
                        archive[f"{split}_direct"][state_index], dtype=np.float64
                    ),
                    "history": np.asarray(
                        archive[f"{split}_history_h"][state_index, :length],
                        dtype=np.float64,
                    ),
                }
                if split == "development" and "development_replayed_target" in archive:
                    row["target"] = np.asarray(
                        archive["development_replayed_target"][state_index], dtype=np.int8
                    )
                state_rows.append(row)

    # Source archive order is the canonical state order for Codex.  Fable uses
    # candidate/matrix/generation order, which is explicit and deterministic.
    if spec.key.startswith("codex"):
        target_map = _codex_development_target_map(spec) if split == "development" else None
        cohort_label = "VALI" if split == "development" else "CONF"
        ordered_keys = [
            f"{cohort_label}-c{candidate}-m{matrix_id:03d}-g{landmark:03d}"
            for matrix_id in range(matrix_limit)
            for candidate in ("02", "03")
            for landmark in (20, 35, 50, 65, 80)
        ]
        lookup = {row["key"]: row for row in state_rows}
        state_rows = [lookup[key] for key in ordered_keys]
        if target_map is not None:
            for row in state_rows:
                row["target"] = target_map[row["key"]]
    else:
        state_rows.sort(key=lambda row: (row["candidate"], row["matrix_id"], row["landmark"]))
    trajectories.sort(key=lambda row: (row["candidate"], row["matrix_id"]))

    max_state_history = max(len(row["history"]) for row in state_rows)
    state_h = np.zeros((len(state_rows), max_state_history), dtype=np.float64)
    state_lengths = np.zeros(len(state_rows), dtype=np.int16)
    for index, row in enumerate(state_rows):
        state_lengths[index] = len(row["history"])
        state_h[index, : len(row["history"])] = row["history"]
    max_trajectory = max(len(row["h"]) for row in trajectories)
    trajectory_h = np.zeros((len(trajectories), max_trajectory), dtype=np.float64)
    trajectory_lengths = np.zeros(len(trajectories), dtype=np.int16)
    for index, row in enumerate(trajectories):
        trajectory_lengths[index] = len(row["h"])
        trajectory_h[index, : len(row["h"])] = row["h"]

    arrays: dict[str, NDArray[Any]] = {
        "protocol_id": np.asarray([protocol_id]),
        "state_keys": np.asarray([row["key"] for row in state_rows]),
        "candidate": np.asarray([row["candidate"] for row in state_rows]),
        "matrix_id": np.asarray([row["matrix_id"] for row in state_rows], dtype=np.int32),
        "landmark": np.asarray([row["landmark"] for row in state_rows], dtype=np.int16),
        "direct": np.asarray([row["direct"] for row in state_rows], dtype=np.float64),
        "history_h": state_h,
        "history_length": state_lengths,
        "trajectory_candidate": np.asarray([row["candidate"] for row in trajectories]),
        "trajectory_matrix_id": np.asarray(
            [row["matrix_id"] for row in trajectories], dtype=np.int32
        ),
        "trajectory_h": trajectory_h,
        "trajectory_length": trajectory_lengths,
        "trajectory_died": np.asarray([row["died"] for row in trajectories], dtype=np.int8),
    }
    if split == "development":
        arrays["targets"] = np.asarray([row["target"] for row in state_rows], dtype=np.int8)
        if arrays["targets"].ndim == 1:
            arrays["targets"] = arrays["targets"][:, None]
    target = REPLAY_ROOT / f"{spec.key}_{split}.npz"
    atomic_npz(target, **arrays)
    return {"states": len(state_rows), "trajectories": len(trajectories)}


def load_development_replay(spec: CohortSpec) -> dict[str, NDArray[Any]]:
    path = REPLAY_ROOT / f"{spec.key}_development.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_confirmation_replay(spec: CohortSpec) -> dict[str, NDArray[Any]]:
    path = REPLAY_ROOT / f"{spec.key}_confirmation.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _codex_confirmation_outcomes(
    spec: CohortSpec, replay: dict[str, NDArray[Any]]
) -> dict[str, NDArray[Any]]:
    source = Path(spec.source_directory)
    with np.load(source / "analysis_arrays.npz", allow_pickle=False) as archive:
        targets = np.asarray(archive["confirmation_targets"], dtype=np.int8)
        source_direct = np.asarray(archive["confirmation_history"], dtype=np.float64)
    if not np.allclose(replay["direct"], source_direct, atol=1e-14, rtol=0.0):
        raise AssertionError(f"{spec.key}: replayed direct features do not match archive")
    table = pd.read_csv(source / "confirmation_states.csv")
    lookup = table.set_index("state_id")
    keys = [str(key) for key in replay["state_keys"]]
    predictions_direct = lookup.loc[keys, "prediction_history"].to_numpy(dtype=np.float64)
    predictions_composite = lookup.loc[keys, "prediction_full"].to_numpy(dtype=np.float64)
    if not np.allclose(targets.mean(axis=1), lookup.loc[keys, "q_all"], atol=1e-15):
        raise AssertionError(f"{spec.key}: retained target table mismatch")
    return {
        "targets": targets,
        "prediction_direct": predictions_direct,
        "prediction_composite": predictions_composite,
        "source_direct": source_direct,
    }


def _fable_v1_predictions(bundle: dict[str, Any], x9: NDArray[np.float64]) -> NDArray[np.float64]:
    probability = bundle["direct"].predict_proba(bundle["sc9"].transform(x9))[:, 1]
    return np.clip(probability, EPSILON, 1 - EPSILON)


def _fable_v2_predictions(
    bundle: dict[str, Any], x9: NDArray[np.float64], x195: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    direct_columns = np.asarray((0, 1, 2, 3, 4, 5, 7, 8), dtype=np.int64)
    z8 = bundle["sc8"].transform(x9[:, direct_columns])
    z = bundle["pca"].transform(bundle["scb"].transform(x195[:, _fable_beta_indices()]))
    direct = bundle["direct8"].predict_proba(z8)[:, 1]
    composite = bundle["v2"].predict_proba(np.column_stack((z, z8)))[:, 1]
    return (
        np.clip(direct, EPSILON, 1 - EPSILON),
        np.clip(composite, EPSILON, 1 - EPSILON),
    )


def _fable_beta_indices() -> NDArray[np.int64]:
    _cohort, features, _sim = _import_fable()
    return np.asarray(features.beta_conditioned_indices(), dtype=np.int64)


def _fable_confirmation_outcomes(
    spec: CohortSpec, replay: dict[str, NDArray[Any]]
) -> dict[str, NDArray[Any]]:
    keys = [
        (str(candidate), int(matrix_id), int(landmark))
        for candidate, matrix_id, landmark in zip(
            replay["candidate"], replay["matrix_id"], replay["landmark"], strict=True
        )
    ]
    if spec.key == "fable_headline":
        with (FABLE_ROOT / "results" / "conf_data.pkl").open("rb") as handle:
            table = pickle.load(handle)["table"]
        lookup = {
            (row["candidate"], int(row["matrix"]), int(row["landmark"])): row
            for row in table
        }
        rows = [lookup[key] for key in keys]
        targets = np.stack([row["y64"] for row in rows]).astype(np.int8)
        direct = np.asarray([row["p_direct"] for row in rows], dtype=np.float64)
        composite = np.asarray([row["p_full"] for row in rows], dtype=np.float64)
        # Recompute direct predictions as a replay audit.
        with (FABLE_ROOT / "results" / "frozen_models.pkl").open("rb") as handle:
            bundles = pickle.load(handle)
        for candidate in ("02", "03"):
            selected = replay["candidate"] == candidate
            reproduced = _fable_v1_predictions(bundles[candidate], replay["direct"][selected])
            if not np.allclose(reproduced, direct[selected], atol=2e-12, rtol=0.0):
                raise AssertionError(f"{spec.key} c{candidate}: direct prediction replay failed")
        return {
            "targets": targets,
            "prediction_direct": direct,
            "prediction_composite": composite,
            "source_direct": replay["direct"].copy(),
        }

    with (FABLE_ROOT / "results_sensitivity" / "v2_cohort.pkl").open("rb") as handle:
        table = pickle.load(handle)["table"]
    lookup = {
        (row["candidate"], int(row["matrix"]), int(row["landmark"])): row
        for row in table
    }
    rows = [lookup[key] for key in keys]
    source_direct = np.stack([row["X9"] for row in rows]).astype(np.float64)
    if not np.allclose(source_direct, replay["direct"], atol=1e-12, rtol=0.0):
        raise AssertionError("fable_primary: replayed direct features mismatch v2 cohort")
    targets = np.zeros((len(rows), 64), dtype=np.int8)
    x195 = np.stack([row["X195"] for row in rows]).astype(np.float64)
    for state_index, row in enumerate(rows):
        for branch in range(64):
            length = int(row["lens"][branch])
            targets[state_index, branch] = int(
                event_from_flags(np.asarray(row["H64"][branch, :length]) > 0.90)
            )
    with (FABLE_ROOT / "results_v2" / "frozen_models_v2.pkl").open("rb") as handle:
        bundles = pickle.load(handle)
    direct = np.empty(len(rows), dtype=np.float64)
    composite = np.empty(len(rows), dtype=np.float64)
    for candidate in ("02", "03"):
        selected = replay["candidate"] == candidate
        direct[selected], composite[selected] = _fable_v2_predictions(
            bundles[candidate], source_direct[selected], x195[selected]
        )
    return {
        "targets": targets,
        "prediction_direct": direct,
        "prediction_composite": composite,
        "source_direct": source_direct,
    }


def load_confirmation_outcomes(spec: CohortSpec) -> dict[str, NDArray[Any]]:
    """Load retained futures and frozen predictions; analyze-stage use only."""

    replay = load_confirmation_replay(spec)
    if spec.key.startswith("codex"):
        return _codex_confirmation_outcomes(spec, replay)
    return _fable_confirmation_outcomes(spec, replay)


def development_audit(spec: CohortSpec, replay: dict[str, NDArray[Any]]) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for candidate in ("02", "03"):
        selected = replay["candidate"] == candidate
        y = replay["targets"][selected]
        candidates[candidate] = {
            "states": int(selected.sum()),
            "branches": int(y.size),
            "prevalence": float(y.mean()),
        }
    if spec.key.startswith("fable"):
        summary_path = Path(spec.development_source_directory) / "dev_summary.json"
        if summary_path.is_file():
            stored = json.loads(summary_path.read_text(encoding="utf-8"))
            for candidate in ("02", "03"):
                observed = candidates[candidate]
                if observed["states"] != int(stored[candidate]["n_examples"]):
                    raise AssertionError(f"{spec.key} c{candidate}: development row mismatch")
                if abs(observed["prevalence"] - float(stored[candidate]["prevalence"])) > 1e-15:
                    raise AssertionError(f"{spec.key} c{candidate}: development target mismatch")
    return candidates
