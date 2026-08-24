from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import CANDIDATES, CohortConfig, ExperimentConfig
from .features import (
    HISTORY_FEATURE_NAMES,
    STATE_GRAPH_FEATURE_NAMES,
    beta_only_features,
    history_features,
    state_graph_features,
)
from .metrics import (
    bootstrap_by_matrix,
    centered_spearman,
    confidence_interval,
    log_loss_from_q,
    permute_matrix_blocks,
    q_brier,
    spearman,
)
from .models import CandidateStudents, fit_students, predict_students
from .processes import ProcessOutcome, evaluate_process
from .reference_targets import REPORTED_TARGETS, in_range
from .seeds import derive_seed
from .simulator import (
    FloatMatrix,
    Snapshot,
    generate_beta,
    generate_initial_composition,
    simulate_future_absorbing,
    simulate_lineage,
    SimulationError,
)

PROCESS_COLUMNS = (
    "break_event",
    "resume_2",
    "episode_3",
    "persist_5",
    "old_return",
    "positive_gain",
    "repeat_return",
    "old_anchor_gain",
)


def _nanmean(values: list[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


@dataclass(frozen=True)
class StateCase:
    state_id: str
    cohort: str
    candidate: str
    matrix_id: int
    landmark: int
    beta: FloatMatrix
    snapshot: Snapshot


@dataclass
class CohortFeatures:
    state_graph: NDArray[np.float64]
    history: NDArray[np.float64]
    beta: NDArray[np.float64]


@dataclass
class BranchBatch:
    target: NDArray[np.int8]
    process: NDArray[np.float64]
    completed_horizon: NDArray[np.int8]


@dataclass
class ReplicationArtifacts:
    metrics: dict[str, Any]
    process_summary: list[dict[str, Any]]
    state_table: pd.DataFrame
    comparison_table: pd.DataFrame
    replay_exact: bool | None


def build_cohort(
    experiment: ExperimentConfig, cohort_name: str, cohort: CohortConfig
) -> list[StateCase]:
    cases: list[StateCase] = []
    for matrix_id in range(cohort.matrices):
        beta_rng = np.random.default_rng(
            derive_seed(experiment.master_seed, f"{cohort_name}.beta", matrix_id)
        )
        initial_rng = np.random.default_rng(
            derive_seed(experiment.master_seed, f"{cohort_name}.initial", matrix_id)
        )
        beta = generate_beta(experiment.gard, beta_rng)
        initial = generate_initial_composition(experiment.gard, initial_rng)
        for candidate, contract in CANDIDATES.items():
            lineage = None
            for attempt in range(100):
                path_rng = np.random.default_rng(
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
                        initial, beta, experiment.gard, contract, path_rng
                    )
                    break
                except SimulationError:
                    continue
            if lineage is None:
                raise SimulationError(
                    f"failed to obtain a complete {cohort_name} trajectory for "
                    f"candidate {candidate}, matrix {matrix_id} in 100 attempts"
                )
            by_generation = {snapshot.generation: snapshot for snapshot in lineage}
            for landmark in cohort.landmarks:
                snapshot = by_generation[landmark]
                cases.append(
                    StateCase(
                        state_id=f"{cohort_name}-c{candidate}-m{matrix_id:03d}-g{landmark:03d}",
                        cohort=cohort_name,
                        candidate=candidate,
                        matrix_id=matrix_id,
                        landmark=landmark,
                        beta=beta,
                        snapshot=snapshot,
                    )
                )
    return cases


def extract_features(cases: list[StateCase], experiment: ExperimentConfig) -> CohortFeatures:
    return CohortFeatures(
        state_graph=np.vstack(
            [
                state_graph_features(case.snapshot.composition, case.beta, experiment.gard)
                for case in cases
            ]
        ),
        history=np.vstack(
            [history_features(case.snapshot, experiment.gard) for case in cases]
        ),
        beta=np.vstack(
            [beta_only_features(case.beta, experiment.gard) for case in cases]
        ),
    )


def _branch_worker(
    args: tuple[StateCase, ExperimentConfig, int]
) -> BranchBatch:
    case, experiment, branches = args
    # Prevent nested BLAS pools from multiplying ProcessPool workers.
    try:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=1)
    except Exception:  # pragma: no cover - optional runtime guard
        limiter = None
    try:
        target = np.empty(branches, dtype=np.int8)
        process = np.empty((branches, len(PROCESS_COLUMNS)), dtype=np.float64)
        completed = np.empty(branches, dtype=np.int8)
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
            outcome: ProcessOutcome = evaluate_process(
                records, experiment.gard.inheritance_threshold
            )
            target[branch] = int(outcome.joint_break_run3)
            completed[branch] = int(completed_horizon)
            values = outcome.to_dict()
            process[branch] = [float(values[column]) for column in PROCESS_COLUMNS]
        return BranchBatch(
            target=target, process=process, completed_horizon=completed
        )
    finally:
        if limiter is not None:
            limiter.restore_original_limits()


def run_branches(
    cases: list[StateCase],
    experiment: ExperimentConfig,
    branches: int,
    workers: int,
) -> list[BranchBatch]:
    arguments = [(case, experiment, branches) for case in cases]
    if workers <= 1:
        return [_branch_worker(argument) for argument in arguments]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_branch_worker, arguments, chunksize=1))


def _stack_targets(batches: list[BranchBatch]) -> NDArray[np.int8]:
    return np.vstack([batch.target for batch in batches])


def _digest_batches(batches: list[BranchBatch]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(np.ascontiguousarray(batch.target).tobytes())
        digest.update(np.ascontiguousarray(batch.completed_horizon).tobytes())
        canonical = np.nan_to_num(batch.process, nan=-999.0)
        digest.update(np.ascontiguousarray(canonical).tobytes())
    return digest.hexdigest()


def _candidate_indices(cases: list[StateCase], candidate: str) -> NDArray[np.int64]:
    return np.asarray(
        [index for index, case in enumerate(cases) if case.candidate == candidate],
        dtype=np.int64,
    )


def _fit_all_students(
    cases: list[StateCase],
    features: CohortFeatures,
    labels: NDArray[np.int8],
    experiment: ExperimentConfig,
) -> dict[str, CandidateStudents]:
    students: dict[str, CandidateStudents] = {}
    for candidate in CANDIDATES:
        selected = _candidate_indices(cases, candidate)
        students[candidate] = fit_students(
            state_graph=features.state_graph[selected],
            history=features.history[selected],
            beta=features.beta[selected],
            branch_labels=labels[selected],
            pca_components=experiment.pca_components,
            c=experiment.logistic_c,
        )
    return students


def _predict_all_students(
    students: dict[str, CandidateStudents],
    cases: list[StateCase],
    features: CohortFeatures,
) -> dict[str, dict[str, NDArray[np.float64]]]:
    predictions: dict[str, dict[str, NDArray[np.float64]]] = {}
    for candidate in CANDIDATES:
        selected = _candidate_indices(cases, candidate)
        predictions[candidate] = predict_students(
            students[candidate],
            state_graph=features.state_graph[selected],
            history=features.history[selected],
            beta=features.beta[selected],
        )
    return predictions


def _bootstrap_rank_lower(
    left: NDArray,
    right: NDArray,
    matrix_ids: NDArray,
    repetitions: int,
    rng: np.random.Generator,
    centered: bool,
) -> float:
    statistic = (
        (lambda values, groups: centered_spearman(values["left"], values["right"], groups))
        if centered
        else (lambda values, groups: spearman(values["left"], values["right"]))
    )
    samples = bootstrap_by_matrix(
        {"left": left, "right": right}, matrix_ids, statistic, repetitions, rng
    )
    return confidence_interval(samples)[0]


def _confirmation_metrics(
    cases: list[StateCase],
    batches: list[BranchBatch],
    predictions: dict[str, dict[str, NDArray[np.float64]]],
    experiment: ExperimentConfig,
) -> dict[str, Any]:
    labels = _stack_targets(batches)
    metrics: dict[str, Any] = {}
    for candidate in CANDIDATES:
        selected = _candidate_indices(cases, candidate)
        candidate_labels = labels[selected]
        split = candidate_labels.shape[1] // 2
        q_a = candidate_labels[:, :split].mean(axis=1)
        q_b = candidate_labels[:, split:].mean(axis=1)
        q_all = candidate_labels.mean(axis=1)
        matrix_ids = np.asarray([cases[index].matrix_id for index in selected])
        rng = np.random.default_rng(
            derive_seed(experiment.master_seed, "metrics.bootstrap", candidate)
        )
        reliability = spearman(q_a, q_b)
        centered_reliability = centered_spearman(q_a, q_b, matrix_ids)
        candidate_metrics: dict[str, Any] = {
            "states": int(selected.size),
            "transition_region_states": int(((q_all > 0.1) & (q_all < 0.9)).sum()),
            "branch_half_reliability": reliability,
            "branch_half_reliability_lower_95": _bootstrap_rank_lower(
                q_a,
                q_b,
                matrix_ids,
                experiment.bootstrap_repetitions,
                rng,
                centered=False,
            ),
            "centered_branch_half_reliability": centered_reliability,
            "centered_branch_half_reliability_lower_95": _bootstrap_rank_lower(
                q_a,
                q_b,
                matrix_ids,
                experiment.bootstrap_repetitions,
                rng,
                centered=True,
            ),
            "models": {},
            "directions": {},
        }
        for model, prediction in predictions[candidate].items():
            overall = [spearman(prediction, q_a), spearman(prediction, q_b)]
            centered = [
                centered_spearman(prediction, q_a, matrix_ids),
                centered_spearman(prediction, q_b, matrix_ids),
            ]
            candidate_metrics["models"][model] = {
                "overall_spearman": overall,
                "overall_spearman_mean": _nanmean(overall),
                "centered_spearman": centered,
                "centered_spearman_mean": _nanmean(centered),
            }

        for direction, q in (("A", q_a), ("B", q_b)):
            history_prediction = predictions[candidate]["history"]
            full_prediction = predictions[candidate]["full"]
            log_loss_history = log_loss_from_q(q, history_prediction)
            log_loss_full = log_loss_from_q(q, full_prediction)
            brier_history = q_brier(q, history_prediction)
            brier_full = q_brier(q, full_prediction)
            observed_gain = log_loss_history - log_loss_full

            gain_samples = bootstrap_by_matrix(
                {"q": q, "history": history_prediction, "full": full_prediction},
                matrix_ids,
                lambda values, groups: log_loss_from_q(values["q"], values["history"])
                - log_loss_from_q(values["q"], values["full"]),
                experiment.bootstrap_repetitions,
                rng,
            )
            brier_samples = bootstrap_by_matrix(
                {"q": q, "history": history_prediction, "full": full_prediction},
                matrix_ids,
                lambda values, groups: q_brier(values["q"], values["history"])
                - q_brier(values["q"], values["full"]),
                experiment.bootstrap_repetitions,
                rng,
            )

            exceedances = 0
            permutation_rng = np.random.default_rng(
                derive_seed(
                    experiment.master_seed,
                    "metrics.permutation",
                    candidate,
                    direction,
                )
            )
            for _ in range(experiment.permutation_repetitions):
                permuted = permute_matrix_blocks(
                    full_prediction, matrix_ids, permutation_rng
                )
                null_gain = log_loss_history - log_loss_from_q(q, permuted)
                exceedances += int(null_gain >= observed_gain)

            candidate_metrics["directions"][direction] = {
                "log_loss_history": log_loss_history,
                "log_loss_full": log_loss_full,
                "log_loss_gain": observed_gain,
                "log_loss_gain_ci95": confidence_interval(gain_samples),
                "q_brier_history": brier_history,
                "q_brier_full": brier_full,
                "q_brier_gain": brier_history - brier_full,
                "q_brier_gain_ci95": confidence_interval(brier_samples),
                "matrix_permutation_p": (exceedances + 1)
                / (experiment.permutation_repetitions + 1),
            }
        metrics[candidate] = candidate_metrics
    return metrics


def _process_prevalence(
    cases: list[StateCase],
    batches: list[BranchBatch],
    experiment: ExperimentConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    repetitions = min(experiment.bootstrap_repetitions, 512)
    for candidate in CANDIDATES:
        selected = _candidate_indices(cases, candidate)
        process = np.vstack([batches[index].process for index in selected])
        matrix_ids = np.repeat(
            [cases[index].matrix_id for index in selected], batches[0].process.shape[0]
        )
        unique = np.unique(matrix_ids)
        locations = {key: np.flatnonzero(matrix_ids == key) for key in unique}
        rng = np.random.default_rng(
            derive_seed(
                experiment.master_seed,
                f"{cases[0].cohort}.process_bootstrap",
                candidate,
            )
        )
        for column, name in enumerate(PROCESS_COLUMNS):
            values = process[:, column]
            estimate = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            samples = np.empty(repetitions, dtype=np.float64)
            for repetition in range(repetitions):
                sampled = rng.choice(unique, size=unique.size, replace=True)
                indices = np.concatenate([locations[key] for key in sampled])
                sampled_values = values[indices]
                samples[repetition] = (
                    float(np.nanmean(sampled_values))
                    if np.isfinite(sampled_values).any()
                    else np.nan
                )
            lower, upper = confidence_interval(samples)
            output.append(
                {
                    "cohort": cases[0].cohort,
                    "candidate": candidate,
                    "metric": name,
                    "estimate": estimate,
                    "lower_95": lower,
                    "upper_95": upper,
                    "defined_n": int(np.isfinite(values).sum()),
                }
            )
    return output


def _state_table(
    cases: list[StateCase],
    batches: list[BranchBatch],
    predictions: dict[str, dict[str, NDArray[np.float64]]],
) -> pd.DataFrame:
    labels = _stack_targets(batches)
    rows: list[dict[str, Any]] = []
    candidate_offsets = {candidate: 0 for candidate in CANDIDATES}
    for index, case in enumerate(cases):
        local_index = candidate_offsets[case.candidate]
        candidate_offsets[case.candidate] += 1
        split = labels.shape[1] // 2
        row: dict[str, Any] = {
            "state_id": case.state_id,
            "cohort": case.cohort,
            "candidate": case.candidate,
            "matrix_id": case.matrix_id,
            "landmark": case.landmark,
            "mass": int(case.snapshot.composition.sum()),
            "q_all": float(labels[index].mean()),
            "q_half_A": float(labels[index, :split].mean()),
            "q_half_B": float(labels[index, split:].mean()),
        }
        for model, values in predictions[case.candidate].items():
            row[f"prediction_{model}"] = float(values[local_index])
        rows.append(row)
    return pd.DataFrame(rows)


def _comparison_table(
    metrics: dict[str, Any], process_summary: list[dict[str, Any]], replay_exact: bool | None
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    process_lookup = {
        (row["cohort"], row["candidate"], row["metric"]): row["estimate"]
        for row in process_summary
    }
    for candidate in CANDIDATES:
        for metric, bounds in REPORTED_TARGETS["process"].items():
            source_metric = "old_anchor_gain" if metric == "old_anchor_gain_mean" else metric
            value = process_lookup.get(("CONF", candidate, source_metric), float("nan"))
            rows.append(
                {
                    "candidate": candidate,
                    "section": "process",
                    "metric": metric,
                    "replicated_value": value,
                    "reported_low": bounds[0],
                    "reported_high": bounds[1],
                    "within_reported_range": bool(np.isfinite(value) and in_range(value, bounds)),
                }
            )

        candidate_metrics = metrics[candidate]
        model_metrics = candidate_metrics["models"]
        directional = candidate_metrics["directions"]
        values = {
            "branch_half_spearman": candidate_metrics["branch_half_reliability"],
            "centered_branch_half_spearman": candidate_metrics[
                "centered_branch_half_reliability"
            ],
            "full_overall_spearman": model_metrics["full"]["overall_spearman_mean"],
            "full_centered_spearman": model_metrics["full"]["centered_spearman_mean"],
            "history_overall_spearman": model_metrics["history"]["overall_spearman_mean"],
            "history_centered_spearman": model_metrics["history"]["centered_spearman_mean"],
            "log_loss_gain": float(
                np.mean([directional[key]["log_loss_gain"] for key in ("A", "B")])
            ),
            "q_brier_gain": float(
                np.mean([directional[key]["q_brier_gain"] for key in ("A", "B")])
            ),
        }
        for metric, bounds in REPORTED_TARGETS["confirmation"].items():
            if metric in ("beta_overall_abs_max", "permutation_p_max"):
                if metric == "beta_overall_abs_max":
                    value = abs(model_metrics["beta"]["overall_spearman_mean"])
                else:
                    value = max(
                        directional[key]["matrix_permutation_p"] for key in ("A", "B")
                    )
                rows.append(
                    {
                        "candidate": candidate,
                        "section": "confirmation",
                        "metric": metric,
                        "replicated_value": value,
                        "reported_low": 0.0,
                        "reported_high": bounds,
                        "within_reported_range": bool(value <= bounds + 1e-12),
                    }
                )
            else:
                value = values[metric]
                rows.append(
                    {
                        "candidate": candidate,
                        "section": "confirmation",
                        "metric": metric,
                        "replicated_value": value,
                        "reported_low": bounds[0],
                        "reported_high": bounds[1],
                        "within_reported_range": bool(in_range(value, bounds)),
                    }
                )
    rows.append(
        {
            "candidate": "both",
            "section": "replay",
            "metric": "exact_regeneration",
            "replicated_value": replay_exact,
            "reported_low": True,
            "reported_high": True,
            "within_reported_range": replay_exact is True,
        }
    )
    return pd.DataFrame(rows)


def _save_branch_table(
    path: Path, cases: list[StateCase], batches: list[BranchBatch]
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        columns = [
            "state_id",
            "cohort",
            "candidate",
            "matrix_id",
            "landmark",
            "branch",
            "half",
            "joint_break_run3",
            "completed_horizon",
            *PROCESS_COLUMNS,
        ]
        handle.write(",".join(columns) + "\n")
        for case, batch in zip(cases, batches):
            split = batch.target.size // 2
            for branch in range(batch.target.size):
                values = [
                    case.state_id,
                    case.cohort,
                    case.candidate,
                    str(case.matrix_id),
                    str(case.landmark),
                    str(branch),
                    "A" if branch < split else "B",
                    str(int(batch.target[branch])),
                    str(int(batch.completed_horizon[branch])),
                ]
                for value in batch.process[branch]:
                    values.append("" if np.isnan(value) else f"{value:.17g}")
                handle.write(",".join(values) + "\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _add_student_arrays(
    arrays: dict[str, NDArray], prefix: str, student: Any
) -> None:
    arrays[f"{prefix}__scaler_mean"] = student.scaler.mean_
    arrays[f"{prefix}__scaler_scale"] = student.scaler.scale_
    arrays[f"{prefix}__classifier_coef"] = student.classifier.coef_
    arrays[f"{prefix}__classifier_intercept"] = student.classifier.intercept_
    arrays[f"{prefix}__classifier_classes"] = student.classifier.classes_
    if student.pca is not None:
        arrays[f"{prefix}__pca_mean"] = student.pca.mean_
        arrays[f"{prefix}__pca_components"] = student.pca.components_
        arrays[f"{prefix}__pca_explained_variance"] = student.pca.explained_variance_


def save_frozen_students(
    path: Path, students: dict[str, CandidateStudents]
) -> None:
    arrays: dict[str, NDArray] = {}
    for candidate, group in students.items():
        arrays[f"c{candidate}__prior"] = np.asarray([group.prior], dtype=np.float64)
        _add_student_arrays(arrays, f"c{candidate}__history", group.history)
        _add_student_arrays(arrays, f"c{candidate}__beta", group.beta)
        _add_student_arrays(arrays, f"c{candidate}__full", group.full)
        arrays[f"c{candidate}__full_state_scaler_mean"] = group.full.state_scaler.mean_  # type: ignore[attr-defined]
        arrays[f"c{candidate}__full_state_scaler_scale"] = group.full.state_scaler.scale_  # type: ignore[attr-defined]
        arrays[f"c{candidate}__full_state_pca_mean"] = group.full.state_pca.mean_  # type: ignore[attr-defined]
        arrays[f"c{candidate}__full_state_pca_components"] = group.full.state_pca.components_  # type: ignore[attr-defined]
        arrays[f"c{candidate}__full_state_pca_explained_variance"] = group.full.state_pca.explained_variance_  # type: ignore[attr-defined]
    np.savez_compressed(path, **arrays)


def save_model_contract(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "state_graph_feature_names": STATE_GRAPH_FEATURE_NAMES,
                "history_feature_names": HISTORY_FEATURE_NAMES,
                "model_order": ["prior", "history", "beta", "full"],
                "archive_format": (
                    "candidate/model/transform parameter arrays; keys use double-underscore separators"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_manifest() -> dict[str, str]:
    packages = ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "threadpoolctl")
    return {
        "python": platform.python_version(),
        **{package: importlib.metadata.version(package) for package in packages},
    }


def run_replication(
    experiment: ExperimentConfig,
    output_directory: Path,
    workers: int | None = None,
) -> ReplicationArtifacts:
    workers = workers or max(1, min(os.cpu_count() or 1, 12))
    output_directory.mkdir(parents=True, exist_ok=True)

    print("[1/8] Generating independent development trajectories", flush=True)
    development_cases = build_cohort(experiment, "VALI", experiment.development)
    development_features = extract_features(development_cases, experiment)

    print("[2/8] Shooting development futures", flush=True)
    development_batches = run_branches(
        development_cases,
        experiment,
        experiment.development.branches_per_state,
        workers,
    )
    development_labels = _stack_targets(development_batches)

    print("[3/8] Freezing candidate-separated PCA/ridge students", flush=True)
    students = _fit_all_students(
        development_cases, development_features, development_labels, experiment
    )

    print("[4/8] Generating untouched confirmation trajectories", flush=True)
    confirmation_cases = build_cohort(experiment, "CONF", experiment.confirmation)
    confirmation_features = extract_features(confirmation_cases, experiment)
    predictions = _predict_all_students(students, confirmation_cases, confirmation_features)

    print("[5/8] Shooting untouched confirmation futures", flush=True)
    confirmation_batches = run_branches(
        confirmation_cases,
        experiment,
        experiment.confirmation.branches_per_state,
        workers,
    )
    first_digest = _digest_batches(confirmation_batches)

    replay_exact: bool | None = None
    second_digest: str | None = None
    if experiment.regenerate_confirmation:
        print("[6/8] Exactly regenerating confirmation futures", flush=True)
        regenerated = run_branches(
            confirmation_cases,
            experiment,
            experiment.confirmation.branches_per_state,
            workers,
        )
        second_digest = _digest_batches(regenerated)
        replay_exact = first_digest == second_digest
        if not replay_exact:
            raise AssertionError("confirmation regeneration was not byte-exact")
    else:
        print("[6/8] Exact regeneration disabled by profile", flush=True)

    print("[7/8] Computing matrix-aware metrics and reference comparison", flush=True)
    metrics = _confirmation_metrics(
        confirmation_cases, confirmation_batches, predictions, experiment
    )
    process_summary = _process_prevalence(
        development_cases, development_batches, experiment
    ) + _process_prevalence(confirmation_cases, confirmation_batches, experiment)
    state_table = _state_table(confirmation_cases, confirmation_batches, predictions)
    comparison_table = _comparison_table(metrics, process_summary, replay_exact)

    print("[8/8] Writing auditable artifacts", flush=True)
    manifest = {
        "clean_room": True,
        "scope": "plastic-heredity discovery only",
        "specified_contract": {
            "target": "within 12 fissions, a break followed by three inherited fissions",
            "inheritance": "strict parent-selected-daughter cosine H > 0.9",
            "state_graph_features": 195,
            "pca_components": experiment.pca_components,
            "logistic_c": experiment.logistic_c,
        },
        "inferred_contract": {
            "candidate_02_and_03_details": (
                "explicit source-constrained alternatives; the paper does not publish their executable definitions"
            ),
            "node_feature_basis": "15 equivariant profiles x 13 symmetric summaries",
            "conditional_process_definitions": (
                "inferred from prose and embedded plot labels; see REPLICATION.md"
            ),
        },
        "experiment": experiment.to_dict(),
        "confirmation_digest_first": first_digest,
        "confirmation_digest_second": second_digest,
        "confirmation_replay_exact": replay_exact,
        "runtime": _runtime_manifest(),
        "inputs": {
            "PRE_PRINT_PAPER_DRAFT.md": _sha256_file(Path("PRE_PRINT_PAPER_DRAFT.md")),
            "PRE_PRINT_DISTILL.PUB.html": _sha256_file(Path("PRE_PRINT_DISTILL.PUB.html")),
            "paper_pdf": _sha256_file(
                Path(
                    "Causal Architecture Dynamics Prior to Arrival of Self-replicators "
                    "in a Model of Catalytic Networks Relevant to Origin-of-Life.pdf"
                )
            ),
            "historical_gard_specification_commit": (
                "86dff6320d5ae91b4e831471079ff46749b14df9"
            ),
        },
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / "metrics.json").write_text(
        json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(process_summary).to_csv(output_directory / "process_summary.csv", index=False)
    state_table.to_csv(output_directory / "confirmation_states.csv", index=False)
    comparison_table.to_csv(output_directory / "reported_comparison.csv", index=False)
    _save_branch_table(
        output_directory / "development_branches.csv.gz",
        development_cases,
        development_batches,
    )
    _save_branch_table(
        output_directory / "confirmation_branches.csv.gz",
        confirmation_cases,
        confirmation_batches,
    )
    np.savez_compressed(
        output_directory / "analysis_arrays.npz",
        development_state_graph=development_features.state_graph,
        development_history=development_features.history,
        development_beta=development_features.beta,
        development_targets=development_labels,
        confirmation_state_graph=confirmation_features.state_graph,
        confirmation_history=confirmation_features.history,
        confirmation_beta=confirmation_features.beta,
        confirmation_targets=_stack_targets(confirmation_batches),
    )
    save_frozen_students(output_directory / "frozen_models.npz", students)
    save_model_contract(output_directory / "model_contract.json")

    return ReplicationArtifacts(
        metrics=metrics,
        process_summary=process_summary,
        state_table=state_table,
        comparison_table=comparison_table,
        replay_exact=replay_exact,
    )
