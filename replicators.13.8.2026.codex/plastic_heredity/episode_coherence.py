"""Post-hoc geometry audit for registered break-and-renewal episodes.

This module does not redefine ``JOINT_BREAK_RUN3`` and does not create a new
confirmation cohort.  It deterministically regenerates positive branches from
three archived 200-matrix confirmations, verifies their registered outcomes,
and describes whether the first qualifying three-fission episode is coherent,
distinct from the pre-break anchor, persistent, or followed by a second renewal
within F12.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import CANDIDATES, CohortConfig, ExperimentConfig, GardConfig
from .experiment import (
    PROCESS_COLUMNS,
    StateCase,
    _json_ready,
    _runtime_manifest,
    build_cohort,
    extract_features,
)
from .mechanistic import (
    _atomic_destination,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .processes import evaluate_process
from .seeds import derive_seed
from .simulator import FissionRecord, cosine_similarity, simulate_future_absorbing

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_FORMAT = "plastic-heredity-episode-coherence-audit-v1"
BOOTSTRAP_REPETITIONS = 4_096
BOOTSTRAP_MASTER_SEED = (
    "f293c044078250501195bfa9d7fcb49e3622b887bb4b4d098ee3dbc2fbc63fcf"
)
COHERENCE_THRESHOLDS = (0.90, 0.95, 0.975)
DISTINCTNESS_THRESHOLDS = (0.90, 0.85, 0.80)
SOURCE_FILES = (
    "plastic_heredity/config.py",
    "plastic_heredity/episode_coherence.py",
    "plastic_heredity/experiment.py",
    "plastic_heredity/features.py",
    "plastic_heredity/mechanistic.py",
    "plastic_heredity/processes.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/simulator.py",
    "requirements-lock.txt",
)


@dataclass(frozen=True)
class CohortSource:
    label: str
    cohort_name: str
    directory: Path


@dataclass(frozen=True)
class EpisodeGeometry:
    first_break_index: int
    episode_start_index: int
    episode_end_index: int
    daughter_similarity_01: float
    daughter_similarity_02: float
    daughter_similarity_12: float
    first_last_daughter_similarity: float
    minimum_pairwise_daughter_similarity: float
    mean_pairwise_daughter_similarity: float
    maximum_pairwise_daughter_similarity: float
    anchor_similarity_0: float
    anchor_similarity_1: float
    anchor_similarity_2: float
    minimum_anchor_similarity: float
    mean_anchor_similarity: float
    maximum_anchor_similarity: float
    onset_parent_to_final_daughter_similarity: float
    observed_inherited_run_length: int
    persistence_5_status: str
    later_break_observed: bool
    second_renewal_after_later_break_observed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_run(values: np.ndarray, length: int, start: int = 0) -> int | None:
    for index in range(start, values.size - length + 1):
        if bool(values[index : index + length].all()):
            return index
    return None


def episode_geometry(
    records: list[FissionRecord], inheritance_threshold: float = 0.9
) -> EpisodeGeometry:
    """Describe the first run that certifies an existing positive endpoint."""

    inherited = np.asarray(
        [record.h > inheritance_threshold for record in records], dtype=bool
    )
    breaks = np.flatnonzero(~inherited)
    if breaks.size == 0:
        raise ValueError("episode geometry requires an observed inheritance break")
    first_break = int(breaks[0])
    episode_start = _first_run(inherited, 3, first_break + 1)
    if episode_start is None:
        raise ValueError("episode geometry requires a post-break run of three")
    episode_end = episode_start + 2

    episode_records = records[episode_start : episode_start + 3]
    daughters = [record.daughter for record in episode_records]
    pairwise = np.asarray(
        [
            cosine_similarity(daughters[0], daughters[1]),
            cosine_similarity(daughters[0], daughters[2]),
            cosine_similarity(daughters[1], daughters[2]),
        ],
        dtype=np.float64,
    )
    anchor = records[first_break].parent
    anchor_similarity = np.asarray(
        [cosine_similarity(anchor, daughter) for daughter in daughters],
        dtype=np.float64,
    )

    run_length = 0
    for value in inherited[episode_start:]:
        if not value:
            break
        run_length += 1
    if run_length >= 5:
        persistence_status = "observed"
    elif episode_start + run_length < len(records):
        persistence_status = "failed"
    else:
        persistence_status = "right_censored"

    later_breaks = np.flatnonzero(~inherited[episode_end + 1 :])
    later_break = episode_end + 1 + int(later_breaks[0]) if later_breaks.size else None
    second_renewal = False
    if later_break is not None:
        second_renewal = _first_run(inherited, 3, later_break + 1) is not None

    return EpisodeGeometry(
        first_break_index=first_break,
        episode_start_index=episode_start,
        episode_end_index=episode_end,
        daughter_similarity_01=float(pairwise[0]),
        daughter_similarity_02=float(pairwise[1]),
        daughter_similarity_12=float(pairwise[2]),
        first_last_daughter_similarity=float(pairwise[1]),
        minimum_pairwise_daughter_similarity=float(pairwise.min()),
        mean_pairwise_daughter_similarity=float(pairwise.mean()),
        maximum_pairwise_daughter_similarity=float(pairwise.max()),
        anchor_similarity_0=float(anchor_similarity[0]),
        anchor_similarity_1=float(anchor_similarity[1]),
        anchor_similarity_2=float(anchor_similarity[2]),
        minimum_anchor_similarity=float(anchor_similarity.min()),
        mean_anchor_similarity=float(anchor_similarity.mean()),
        maximum_anchor_similarity=float(anchor_similarity.max()),
        onset_parent_to_final_daughter_similarity=cosine_similarity(
            episode_records[0].parent, daughters[-1]
        ),
        observed_inherited_run_length=run_length,
        persistence_5_status=persistence_status,
        later_break_observed=later_break is not None,
        second_renewal_after_later_break_observed=second_renewal,
    )


def _experiment_from_manifest(manifest: dict[str, Any]) -> ExperimentConfig:
    raw = manifest["experiment"]
    expected_candidates = {name: asdict(value) for name, value in CANDIDATES.items()}
    if raw.get("candidates") != expected_candidates:
        raise ValueError("archived candidate contracts differ from the implementation")

    def cohort(value: dict[str, Any]) -> CohortConfig:
        return CohortConfig(
            matrices=int(value["matrices"]),
            branches_per_state=int(value["branches_per_state"]),
            landmarks=tuple(int(item) for item in value["landmarks"]),
        )

    return ExperimentConfig(
        gard=GardConfig(**raw["gard"]),
        development=cohort(raw["development"]),
        confirmation=cohort(raw["confirmation"]),
        horizon=int(raw["horizon"]),
        pca_components=int(raw["pca_components"]),
        logistic_c=float(raw["logistic_c"]),
        bootstrap_repetitions=int(raw["bootstrap_repetitions"]),
        permutation_repetitions=int(raw["permutation_repetitions"]),
        regenerate_confirmation=bool(raw["regenerate_confirmation"]),
        master_seed=str(raw["master_seed"]),
    )


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def _canonical_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_branch_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"candidate": str})
    table["candidate"] = table["candidate"].str.zfill(2)
    return table.reset_index(drop=True)


def _validate_case_order(
    cases: list[StateCase], state_table: pd.DataFrame, branches: pd.DataFrame
) -> None:
    expected_states = pd.DataFrame(
        {
            "state_id": [case.state_id for case in cases],
            "cohort": [case.cohort for case in cases],
            "candidate": [case.candidate for case in cases],
            "matrix_id": [case.matrix_id for case in cases],
            "landmark": [case.landmark for case in cases],
            "mass": [int(case.snapshot.composition.sum()) for case in cases],
        }
    )
    observed = state_table.loc[:, expected_states.columns].copy()
    observed["candidate"] = observed["candidate"].astype(str).str.zfill(2)
    if not expected_states.equals(observed.reset_index(drop=True)):
        raise ValueError("regenerated state identifiers or masses differ from archive")

    branches_per_state = int(branches.groupby("state_id").size().iloc[0])
    expected_keys = pd.DataFrame(
        {
            "state_id": np.repeat(
                expected_states["state_id"].to_numpy(), branches_per_state
            ),
            "candidate": np.repeat(
                expected_states["candidate"].to_numpy(), branches_per_state
            ),
            "matrix_id": np.repeat(
                expected_states["matrix_id"].to_numpy(), branches_per_state
            ),
            "landmark": np.repeat(
                expected_states["landmark"].to_numpy(), branches_per_state
            ),
            "branch": np.tile(np.arange(branches_per_state), len(cases)),
        }
    )
    observed_keys = branches.loc[:, expected_keys.columns].reset_index(drop=True)
    if not expected_keys.equals(observed_keys):
        raise ValueError("archived branch keys are incomplete or out of contract")


def _validate_reconstruction_arrays(
    source: CohortSource,
    cases: list[StateCase],
    experiment: ExperimentConfig,
    branches: pd.DataFrame,
) -> dict[str, bool]:
    audit: dict[str, bool] = {}
    with np.load(source.directory / "analysis_arrays.npz") as archive:
        targets = (
            branches["joint_break_run3"]
            .to_numpy(dtype=np.int8)
            .reshape(len(cases), experiment.confirmation.branches_per_state)
        )
        audit["targets_exact"] = bool(
            np.array_equal(targets, archive["confirmation_targets"])
        )
        if "confirmation_compositions" in archive.files:
            audit["compositions_exact"] = bool(
                np.array_equal(
                    np.vstack([case.snapshot.composition for case in cases]),
                    archive["confirmation_compositions"],
                )
            )
        else:
            with threadpool_limits(limits=1):
                features = extract_features(cases, experiment)
            audit.update(
                {
                    "state_graph_exact": bool(
                        np.array_equal(
                            features.state_graph, archive["confirmation_state_graph"]
                        )
                    ),
                    "history_exact": bool(
                        np.array_equal(
                            features.history, archive["confirmation_history"]
                        )
                    ),
                    "beta_exact": bool(
                        np.array_equal(features.beta, archive["confirmation_beta"])
                    ),
                }
            )
    if not all(audit.values()):
        raise ValueError(f"{source.label} state reconstruction diverged: {audit}")
    return audit


def _audit_case_worker(
    args: tuple[str, StateCase, ExperimentConfig, tuple[int, ...]],
) -> list[dict[str, Any]]:
    cohort_label, case, experiment, branch_indices = args
    rows: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for branch in branch_indices:
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
            records, completed = simulate_future_absorbing(
                case.snapshot,
                case.beta,
                experiment.gard,
                CANDIDATES[case.candidate],
                experiment.horizon,
                rng,
            )
            outcome = evaluate_process(records, experiment.gard.inheritance_threshold)
            if not outcome.joint_break_run3:
                raise AssertionError(
                    f"archived positive branch regenerated negative: {case.state_id}/{branch}"
                )
            geometry = episode_geometry(records, experiment.gard.inheritance_threshold)
            row: dict[str, Any] = {
                "source_cohort": cohort_label,
                "state_id": case.state_id,
                "cohort": case.cohort,
                "candidate": case.candidate,
                "matrix_id": case.matrix_id,
                "landmark": case.landmark,
                "branch": branch,
                "completed_horizon_regenerated": int(completed),
                "joint_break_run3_regenerated": int(outcome.joint_break_run3),
            }
            row.update(
                {
                    f"process_{name}_regenerated": value
                    for name, value in outcome.to_dict().items()
                    if name != "joint_break_run3"
                }
            )
            row.update(geometry.to_dict())
            rows.append(row)
    return rows


def _validate_replayed_row(expected: pd.Series, observed: dict[str, Any]) -> float:
    if int(expected["joint_break_run3"]) != observed["joint_break_run3_regenerated"]:
        raise ValueError("regenerated target differs from archived branch")
    if int(expected["completed_horizon"]) != observed["completed_horizon_regenerated"]:
        raise ValueError("regenerated completion status differs from archived branch")
    maximum_error = 0.0
    for name in PROCESS_COLUMNS:
        left = float(expected[name])
        right = float(observed[f"process_{name}_regenerated"])
        if np.isnan(left) and np.isnan(right):
            continue
        error = abs(left - right)
        maximum_error = max(maximum_error, error)
        tolerance = 1e-14 if name == "old_anchor_gain" else 0.0
        if error > tolerance:
            raise ValueError(
                f"regenerated process value differs for {name}: {left} != {right}"
            )
    return maximum_error


def _regenerate_positive_events(
    source: CohortSource, workers: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    verify_checksums(source.directory)
    manifest = json.loads(
        (source.directory / "manifest.json").read_text(encoding="utf-8")
    )
    experiment = _experiment_from_manifest(manifest)
    if experiment.confirmation.matrices != 200:
        raise ValueError(f"{source.label} is not a 200-matrix confirmation")

    with threadpool_limits(limits=1):
        cases = build_cohort(experiment, source.cohort_name, experiment.confirmation)
    states = pd.read_csv(
        source.directory / "confirmation_states.csv", dtype={"candidate": str}
    )
    branches = _load_branch_table(source.directory / "confirmation_branches.csv.gz")
    _validate_case_order(cases, states, branches)
    reconstruction = _validate_reconstruction_arrays(
        source, cases, experiment, branches
    )

    positives = branches.loc[branches["joint_break_run3"] == 1].copy()
    branches_by_state = {
        state_id: tuple(group["branch"].astype(int).tolist())
        for state_id, group in positives.groupby("state_id", sort=False)
    }
    arguments = [
        (source.label, case, experiment, branches_by_state.get(case.state_id, ()))
        for case in cases
        if case.state_id in branches_by_state
    ]
    event_rows: list[dict[str, Any]] = []
    if workers <= 1:
        iterator: Iterable[list[dict[str, Any]]] = map(_audit_case_worker, arguments)
        for index, rows in enumerate(iterator, start=1):
            event_rows.extend(rows)
            if index % 200 == 0:
                print(
                    f"  {source.label}: regenerated {index}/{len(arguments)} positive states",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(_audit_case_worker, arguments, chunksize=1)
            for index, rows in enumerate(iterator, start=1):
                event_rows.extend(rows)
                if index % 200 == 0:
                    print(
                        f"  {source.label}: regenerated {index}/{len(arguments)} positive states",
                        flush=True,
                    )

    events = (
        pd.DataFrame(event_rows)
        .sort_values(["matrix_id", "candidate", "landmark", "branch"], kind="stable")
        .reset_index(drop=True)
    )
    expected = positives.set_index(["state_id", "branch"], drop=False)
    maximum_process_error = 0.0
    for row in events.to_dict("records"):
        archived = expected.loc[(row["state_id"], row["branch"])]
        maximum_process_error = max(
            maximum_process_error, _validate_replayed_row(archived, row)
        )
    if len(events) != len(positives):
        raise ValueError("not every archived positive branch was regenerated")

    half_split = experiment.confirmation.branches_per_state // 2
    events["half"] = np.where(events["branch"] < half_split, "A", "B")
    events["cohort_matrix_count"] = experiment.confirmation.matrices
    audit = {
        "source_cohort": source.label,
        "cohort_name": source.cohort_name,
        "input_sha256sums_digest": sha256_file(source.directory / "SHA256SUMS"),
        "states": len(cases),
        "all_archived_branches": len(branches),
        "positive_branches": len(positives),
        "positive_branches_regenerated": len(events),
        "all_positive_targets_and_discrete_process_values_exact": True,
        "maximum_continuous_process_absolute_error": maximum_process_error,
        "continuous_process_values_within_1e-14": maximum_process_error <= 1e-14,
        "state_reconstruction": reconstruction,
        "master_seed": experiment.master_seed,
        "horizon": experiment.horizon,
    }
    return events, audit


def _cluster_mean_interval(
    values: np.ndarray,
    matrix_ids: np.ndarray,
    key: tuple[object, ...],
    repetitions: int = BOOTSTRAP_REPETITIONS,
    matrix_universe: np.ndarray | None = None,
) -> tuple[float, float, float, int, int]:
    values = np.asarray(values, dtype=np.float64)
    matrix_ids = np.asarray(matrix_ids, dtype=np.int64)
    finite = np.isfinite(values)
    values = values[finite]
    matrix_ids = matrix_ids[finite]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0
    observed_matrices = np.unique(matrix_ids)
    matrices = (
        observed_matrices
        if matrix_universe is None
        else np.asarray(matrix_universe, dtype=np.int64)
    )
    if not np.isin(observed_matrices, matrices).all():
        raise ValueError("observed matrix ID is absent from bootstrap universe")
    sums = np.asarray([values[matrix_ids == item].sum() for item in matrices])
    counts = np.asarray([np.count_nonzero(matrix_ids == item) for item in matrices])
    rng = np.random.default_rng(
        derive_seed(BOOTSTRAP_MASTER_SEED, "episode_coherence.bootstrap", *key)
    )
    draws = rng.integers(0, len(matrices), size=(repetitions, len(matrices)))
    sampled_counts = counts[draws].sum(axis=1)
    valid = sampled_counts > 0
    if not valid.any():
        raise ValueError("matrix bootstrap produced no eligible events")
    sampled = sums[draws][valid].sum(axis=1) / sampled_counts[valid]
    lower, upper = np.quantile(sampled, (0.025, 0.975))
    return (
        float(values.mean()),
        float(lower),
        float(upper),
        int(values.size),
        int(matrices.size),
    )


def _groups(events: pd.DataFrame) -> Iterable[tuple[str, str, str, pd.DataFrame]]:
    for cohort in events["source_cohort"].drop_duplicates():
        cohort_rows = events.loc[events["source_cohort"] == cohort]
        for candidate in CANDIDATES:
            candidate_rows = cohort_rows.loc[cohort_rows["candidate"] == candidate]
            yield cohort, candidate, "combined", candidate_rows
            for half in ("A", "B"):
                yield (
                    cohort,
                    candidate,
                    half,
                    candidate_rows.loc[candidate_rows["half"] == half],
                )


def _matrix_universe(selected: pd.DataFrame) -> np.ndarray:
    counts = selected["cohort_matrix_count"].drop_duplicates().to_numpy(dtype=np.int64)
    if counts.size != 1 or counts[0] <= 0:
        raise ValueError("each summary group must declare one positive matrix count")
    return np.arange(int(counts[0]), dtype=np.int64)


def summarize_geometry(events: pd.DataFrame) -> pd.DataFrame:
    numeric_metrics = (
        "first_last_daughter_similarity",
        "minimum_pairwise_daughter_similarity",
        "mean_pairwise_daughter_similarity",
        "maximum_anchor_similarity",
        "mean_anchor_similarity",
        "onset_parent_to_final_daughter_similarity",
        "observed_inherited_run_length",
    )
    rows: list[dict[str, Any]] = []
    for cohort, candidate, half, selected in _groups(events):
        derived = {
            "persistence_5_resolved": np.where(
                selected["persistence_5_status"] == "right_censored",
                np.nan,
                (selected["persistence_5_status"] == "observed").astype(float),
            ),
            "persistence_5_right_censored": (
                selected["persistence_5_status"] == "right_censored"
            ).astype(float),
            "later_break_observed": selected["later_break_observed"].astype(float),
            "second_renewal_after_later_break_observed": selected[
                "second_renewal_after_later_break_observed"
            ].astype(float),
        }
        for metric in (*numeric_metrics, *derived):
            values = (
                selected[metric].to_numpy(dtype=np.float64)
                if metric in selected
                else np.asarray(derived[metric], dtype=np.float64)
            )
            estimate, lower, upper, count, matrices = _cluster_mean_interval(
                values,
                selected["matrix_id"].to_numpy(),
                (cohort, candidate, half, metric),
                matrix_universe=_matrix_universe(selected),
            )
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "source_cohort": cohort,
                    "candidate": candidate,
                    "half": half,
                    "metric": metric,
                    "estimate": estimate,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "median": float(np.median(finite)) if finite.size else np.nan,
                    "q05": float(np.quantile(finite, 0.05)) if finite.size else np.nan,
                    "q95": float(np.quantile(finite, 0.95)) if finite.size else np.nan,
                    "events": count,
                    "matrices": matrices,
                }
            )
    return pd.DataFrame(rows)


def summarize_sensitivity(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, candidate, half, selected in _groups(events):
        minimum_pairwise = selected["minimum_pairwise_daughter_similarity"].to_numpy(
            dtype=np.float64
        )
        maximum_anchor = selected["maximum_anchor_similarity"].to_numpy(
            dtype=np.float64
        )
        resolved = selected["persistence_5_status"] != "right_censored"
        persisted = selected["persistence_5_status"] == "observed"
        for coherence_threshold in COHERENCE_THRESHOLDS:
            coherent = minimum_pairwise > coherence_threshold
            for distinctness_threshold in DISTINCTNESS_THRESHOLDS:
                distinct = maximum_anchor <= distinctness_threshold
                metrics = {
                    "coherent": coherent.astype(float),
                    "distinct": distinct.astype(float),
                    "coherent_and_distinct": (coherent & distinct).astype(float),
                    "coherent_distinct_and_persistent_5": np.where(
                        resolved,
                        (coherent & distinct & persisted).astype(float),
                        np.nan,
                    ),
                }
                row: dict[str, Any] = {
                    "source_cohort": cohort,
                    "candidate": candidate,
                    "half": half,
                    "coherence_threshold_strict": coherence_threshold,
                    "distinctness_max_anchor_threshold_inclusive": distinctness_threshold,
                    "all_events": len(selected),
                    "persistence_resolved_events": int(resolved.sum()),
                    "persistence_censored_events": int((~resolved).sum()),
                }
                for name, values in metrics.items():
                    estimate, lower, upper, count, matrices = _cluster_mean_interval(
                        values,
                        selected["matrix_id"].to_numpy(),
                        (
                            cohort,
                            candidate,
                            half,
                            coherence_threshold,
                            distinctness_threshold,
                            name,
                        ),
                        matrix_universe=_matrix_universe(selected),
                    )
                    row[f"{name}_rate"] = estimate
                    row[f"{name}_ci95_lower"] = lower
                    row[f"{name}_ci95_upper"] = upper
                    row[f"{name}_events"] = count
                    row[f"{name}_matrices"] = matrices
                rows.append(row)
    return pd.DataFrame(rows)


def _protocol() -> dict[str, Any]:
    return {
        "format": AUDIT_FORMAT,
        "status": "post_hoc_exploratory_not_prospective_confirmation",
        "registered_endpoint_unchanged": "JOINT_BREAK_RUN3",
        "episode_selection": "first post-break run certifying three strict H>0.9 fissions",
        "geometry": {
            "coherence_primary_description": "continuous pairwise cosine similarities among the three episode daughters",
            "distinctness_primary_description": "continuous cosine similarities from every episode daughter to the pre-break parent",
            "persistence": "same first qualifying uninterrupted inherited run reaches five; boundary-limited runs are right-censored",
            "second_renewal_within_f12": "later break followed by another observed run of three within the same F12 future; this is not compositional recurrence",
        },
        "sensitivity_thresholds": {
            "coherence_strict_minimum_pairwise": COHERENCE_THRESHOLDS,
            "distinctness_inclusive_maximum_anchor": DISTINCTNESS_THRESHOLDS,
        },
        "inference": {
            "bootstrap_unit": "catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_master_seed": BOOTSTRAP_MASTER_SEED,
            "candidates_reported_separately": True,
            "cohorts_reported_separately": True,
            "branch_halves_reported_separately": True,
        },
        "claim_boundary": "favourable geometry motivates but cannot replace a new prospective regime test; compositional recurrence is not tested",
    }


def _lookup(
    table: pd.DataFrame,
    cohort: str,
    candidate: str,
    metric: str,
    half: str = "combined",
) -> pd.Series:
    return table.loc[
        (table["source_cohort"] == cohort)
        & (table["candidate"] == candidate)
        & (table["half"] == half)
        & (table["metric"] == metric)
    ].iloc[0]


def _estimate_with_interval(row: pd.Series) -> str:
    return f"{row['estimate']:.4f} [{row['ci95_lower']:.4f}, {row['ci95_upper']:.4f}]"


def _rate_with_interval(row: pd.Series, prefix: str) -> str:
    return (
        f"{row[f'{prefix}_rate']:.4f} "
        f"[{row[f'{prefix}_ci95_lower']:.4f}, "
        f"{row[f'{prefix}_ci95_upper']:.4f}]"
    )


def _write_report(
    output: Path,
    geometry: pd.DataFrame,
    sensitivity: pd.DataFrame,
    replay_audits: list[dict[str, Any]],
) -> None:
    lines = [
        "# Exploratory episode-coherence audit",
        "",
        "## Status and outcome",
        "",
        "This is a post-hoc descriptive audit of existing positive `JOINT_BREAK_RUN3` futures. It does not redefine or prospectively confirm the endpoint. Every reported episode is still, first and foremost, a break followed by three inherited fissions.",
        "",
        "The table below describes the first qualifying episode. Similarities are cosine values; larger pairwise values mean tighter episode geometry, while smaller old-anchor values mean greater separation from the pre-break composition.",
        "",
        "| Cohort | Candidate | Episodes | Mean minimum pairwise H [95% CI] | Mean first-to-last H [95% CI] | Mean maximum old-anchor H [95% CI] | Same-run persist-5, resolved [95% CI] | Second renewal after later break in F12 [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for audit in replay_audits:
        cohort = audit["source_cohort"]
        for candidate in CANDIDATES:
            minimum = _lookup(
                geometry, cohort, candidate, "minimum_pairwise_daughter_similarity"
            )
            first_last = _lookup(
                geometry, cohort, candidate, "first_last_daughter_similarity"
            )
            anchor = _lookup(geometry, cohort, candidate, "maximum_anchor_similarity")
            persistence = _lookup(geometry, cohort, candidate, "persistence_5_resolved")
            second_renewal = _lookup(
                geometry,
                cohort,
                candidate,
                "second_renewal_after_later_break_observed",
            )
            lines.append(
                f"| {cohort} | {candidate} | {int(minimum['events'])} | "
                f"{_estimate_with_interval(minimum)} | "
                f"{_estimate_with_interval(first_last)} | "
                f"{_estimate_with_interval(anchor)} | "
                f"{_estimate_with_interval(persistence)} | "
                f"{_estimate_with_interval(second_renewal)} |"
            )
    lines.extend(
        [
            "",
            "## Threshold sensitivity",
            "",
            "These cutoffs were chosen after the original result and are sensitivity views, not discovery gates. The least restrictive view requires minimum pairwise daughter similarity `>0.90` and maximum similarity to the old anchor `<=0.90`.",
            "",
            "| Cohort | Candidate | Coherent [95% CI] | Distinct [95% CI] | Both [95% CI] | Both + persist-5 among resolved [95% CI] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    selected = sensitivity.loc[
        (sensitivity["half"] == "combined")
        & (sensitivity["coherence_threshold_strict"] == 0.90)
        & (sensitivity["distinctness_max_anchor_threshold_inclusive"] == 0.90)
    ]
    for audit in replay_audits:
        cohort = audit["source_cohort"]
        for candidate in CANDIDATES:
            row = selected.loc[
                (selected["source_cohort"] == cohort)
                & (selected["candidate"] == candidate)
            ].iloc[0]
            lines.append(
                f"| {cohort} | {candidate} | "
                f"{_rate_with_interval(row, 'coherent')} | "
                f"{_rate_with_interval(row, 'distinct')} | "
                f"{_rate_with_interval(row, 'coherent_and_distinct')} | "
                f"{_rate_with_interval(row, 'coherent_distinct_and_persistent_5')} |"
            )
    lines.extend(
        [
            "",
            "## Branch-half consistency",
            "",
            "The two preassigned branch halves are descriptive consistency checks. No cohort, candidate, or half is pooled to rescue disagreement.",
            "",
            "| Cohort | Candidate | Half | Coherent >0.90 | Distinct <=0.90 | Same-run persist-5, resolved | Second renewal after later break |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    half_sensitivity = sensitivity.loc[
        sensitivity["half"].isin(("A", "B"))
        & (sensitivity["coherence_threshold_strict"] == 0.90)
        & (sensitivity["distinctness_max_anchor_threshold_inclusive"] == 0.90)
    ]
    for audit in replay_audits:
        cohort = audit["source_cohort"]
        for candidate in CANDIDATES:
            for half in ("A", "B"):
                row = half_sensitivity.loc[
                    (half_sensitivity["source_cohort"] == cohort)
                    & (half_sensitivity["candidate"] == candidate)
                    & (half_sensitivity["half"] == half)
                ].iloc[0]
                persistence = _lookup(
                    geometry, cohort, candidate, "persistence_5_resolved", half
                )
                second_renewal = _lookup(
                    geometry,
                    cohort,
                    candidate,
                    "second_renewal_after_later_break_observed",
                    half,
                )
                lines.append(
                    f"| {cohort} | {candidate} | {half} | "
                    f"{row['coherent_rate']:.4f} | {row['distinct_rate']:.4f} | "
                    f"{persistence['estimate']:.4f} | {second_renewal['estimate']:.4f} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Even a high descriptive coherence rate would not make the original target a registered regime test: the endpoint never required coherence, distinctness, persistence, or recurrence. The F12 second-renewal summary is not a test of return to the same composition. Conversely, weak geometry directly argues against regime language.",
            "",
            "The defensible claim remains that a break followed by renewed short-run inheritance has a reproducible, predictable probability. A distinct new hereditary regime requires a frozen coherence/distinctness/persistence endpoint on another untouched cohort.",
            "",
            "## Replay audit",
            "",
            "| Cohort | Archived futures | Positive episodes replayed | State reconstruction | Target/process replay |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for audit in replay_audits:
        lines.append(
            f"| {audit['source_cohort']} | {audit['all_archived_branches']} | "
            f"{audit['positive_branches_regenerated']} | "
            f"{all(audit['state_reconstruction'].values())} | "
            f"{audit['all_positive_targets_and_discrete_process_values_exact']} / "
            f"continuous within 1e-14: {audit['continuous_process_values_within_1e-14']} |"
        )
    lines.append("")
    (output / "EPISODE_COHERENCE_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_lay_summary(
    output: Path, geometry: pd.DataFrame, sensitivity: pd.DataFrame
) -> None:
    combined_geometry = geometry.loc[geometry["half"] == "combined"]
    least_restrictive = sensitivity.loc[
        (sensitivity["half"] == "combined")
        & (sensitivity["coherence_threshold_strict"] == 0.90)
        & (sensitivity["distinctness_max_anchor_threshold_inclusive"] == 0.90)
    ]

    def metric_range(metric: str) -> tuple[float, float]:
        values = combined_geometry.loc[
            combined_geometry["metric"] == metric, "estimate"
        ]
        return float(values.min()), float(values.max())

    minimum_pairwise = metric_range("minimum_pairwise_daughter_similarity")
    persistence = metric_range("persistence_5_resolved")
    second_renewal = metric_range("second_renewal_after_later_break_observed")
    coherent = (
        float(least_restrictive["coherent_rate"].min()),
        float(least_restrictive["coherent_rate"].max()),
    )
    distinct = (
        float(least_restrictive["distinct_rate"].min()),
        float(least_restrictive["distinct_rate"].max()),
    )
    lines = [
        "# Lay summary of the episode-coherence audit",
        "",
        "The original target asks whether heredity breaks and is then followed by three successful parent-to-daughter inheritance steps. It does not ask whether the three resulting daughters all resemble one another.",
        "",
        "We replayed all 145,516 qualifying episodes from three existing 200-matrix confirmation campaigns and compared the episode compositions.",
        "",
        "What we found:",
        "",
        f"- Only {100 * coherent[0]:.1f}–{100 * coherent[1]:.1f}% of episodes placed every daughter pair above the original 0.9 similarity standard.",
        f"- Mean weakest pairwise similarity was {minimum_pairwise[0]:.3f}–{minimum_pairwise[1]:.3f}, showing that episode-wide compositional coherence is uncommon.",
        f"- {100 * distinct[0]:.1f}–{100 * distinct[1]:.1f}% kept all three daughters outside the old composition's 0.9 neighbourhood.",
        f"- Among episodes where five-step persistence could be decided within the horizon, {100 * persistence[0]:.1f}–{100 * persistence[1]:.1f}% continued in the same inherited run to five.",
        f"- A later break followed by another run of three was observed in {100 * second_renewal[0]:.1f}–{100 * second_renewal[1]:.1f}% of qualifying F12 futures; this does not mean return to the same composition.",
        "",
        "In plain terms: the system often loses heredity and then regains short-run parent-to-daughter similarity somewhere away from the old composition, but the daughters usually drift too much to call the episode one coherent new compositional regime.",
        "",
        "Therefore the supported discovery is **predictable plastic-heredity break-and-renewal**. A distinct new hereditary regime remains a hypothesis for a new prospective test.",
        "",
        "This audit is post-hoc: it narrows interpretation but cannot create a new confirmation claim.",
        "",
    ]
    (output / "LAY_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def independently_recompute_summaries(
    events: pd.DataFrame, geometry: pd.DataFrame, sensitivity: pd.DataFrame
) -> dict[str, Any]:
    """Recompute saved point estimates without calling either summary function."""

    errors: list[float] = []
    checks = 0
    for cohort in events["source_cohort"].drop_duplicates():
        for candidate in CANDIDATES:
            base = events.loc[
                (events["source_cohort"] == cohort) & (events["candidate"] == candidate)
            ]
            for half in ("combined", "A", "B"):
                selected = (
                    base if half == "combined" else base.loc[base["half"] == half]
                )
                resolved = selected["persistence_5_status"] != "right_censored"
                geometry_values = {
                    "first_last_daughter_similarity": selected[
                        "first_last_daughter_similarity"
                    ].mean(),
                    "minimum_pairwise_daughter_similarity": selected[
                        "minimum_pairwise_daughter_similarity"
                    ].mean(),
                    "mean_pairwise_daughter_similarity": selected[
                        "mean_pairwise_daughter_similarity"
                    ].mean(),
                    "maximum_anchor_similarity": selected[
                        "maximum_anchor_similarity"
                    ].mean(),
                    "mean_anchor_similarity": selected["mean_anchor_similarity"].mean(),
                    "onset_parent_to_final_daughter_similarity": selected[
                        "onset_parent_to_final_daughter_similarity"
                    ].mean(),
                    "observed_inherited_run_length": selected[
                        "observed_inherited_run_length"
                    ].mean(),
                    "persistence_5_resolved": (
                        selected.loc[resolved, "persistence_5_status"] == "observed"
                    ).mean(),
                    "persistence_5_right_censored": (~resolved).mean(),
                    "later_break_observed": selected["later_break_observed"].mean(),
                    "second_renewal_after_later_break_observed": selected[
                        "second_renewal_after_later_break_observed"
                    ].mean(),
                }
                saved_geometry = geometry.loc[
                    (geometry["source_cohort"] == cohort)
                    & (geometry["candidate"] == candidate)
                    & (geometry["half"] == half)
                ].set_index("metric")
                for metric, value in geometry_values.items():
                    errors.append(
                        abs(float(saved_geometry.loc[metric, "estimate"]) - value)
                    )
                    checks += 1

                minimum_pairwise = selected[
                    "minimum_pairwise_daughter_similarity"
                ].to_numpy()
                maximum_anchor = selected["maximum_anchor_similarity"].to_numpy()
                resolved_values = resolved.to_numpy()
                persisted = (
                    selected.loc[resolved, "persistence_5_status"].to_numpy()
                    == "observed"
                )
                for coherence_threshold in COHERENCE_THRESHOLDS:
                    coherent = minimum_pairwise > coherence_threshold
                    for distinctness_threshold in DISTINCTNESS_THRESHOLDS:
                        distinct = maximum_anchor <= distinctness_threshold
                        sensitivity_values = {
                            "coherent": coherent.mean(),
                            "distinct": distinct.mean(),
                            "coherent_and_distinct": (coherent & distinct).mean(),
                            "coherent_distinct_and_persistent_5": (
                                coherent[resolved_values]
                                & distinct[resolved_values]
                                & persisted
                            ).mean(),
                        }
                        saved = sensitivity.loc[
                            (sensitivity["source_cohort"] == cohort)
                            & (sensitivity["candidate"] == candidate)
                            & (sensitivity["half"] == half)
                            & (
                                sensitivity["coherence_threshold_strict"]
                                == coherence_threshold
                            )
                            & (
                                sensitivity[
                                    "distinctness_max_anchor_threshold_inclusive"
                                ]
                                == distinctness_threshold
                            )
                        ].iloc[0]
                        for metric, value in sensitivity_values.items():
                            errors.append(abs(float(saved[f"{metric}_rate"]) - value))
                            checks += 1
    maximum_error = max(errors, default=float("inf"))
    return {
        "event_rows": len(events),
        "point_estimates_checked": checks,
        "maximum_absolute_error": maximum_error,
        "all_within_1e-14": maximum_error <= 1e-14,
    }


def run_audit(
    sources: tuple[CohortSource, ...],
    output_directory: Path,
    workers: int | None = None,
) -> None:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    workers = workers or max(1, min(os.cpu_count() or 1, 12))
    protocol = _protocol()
    print(
        "[coherence 1/5] Verifying source bundles and reconstructing cohorts",
        flush=True,
    )
    all_events: list[pd.DataFrame] = []
    replay_audits: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        print(
            f"[coherence 2/5] Replaying positive episodes {index}/{len(sources)}: {source.label}",
            flush=True,
        )
        events, audit = _regenerate_positive_events(source, workers)
        all_events.append(events)
        replay_audits.append(audit)
    events = pd.concat(all_events, ignore_index=True)

    print("[coherence 3/5] Computing matrix-bootstrap descriptions", flush=True)
    geometry = summarize_geometry(events)
    sensitivity = summarize_sensitivity(events)

    print(
        "[coherence 4/5] Writing an explicitly exploratory immutable bundle", flush=True
    )
    with _atomic_destination(output_directory) as output:
        protocol["protocol_id"] = _canonical_digest(protocol)
        (output / "protocol.json").write_text(
            json.dumps(_json_ready(protocol), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        events.to_csv(
            output / "episode_measurements.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        geometry.to_csv(output / "geometry_summary.csv", index=False)
        sensitivity.to_csv(output / "threshold_sensitivity.csv", index=False)
        (output / "metrics.json").write_text(
            json.dumps(
                _json_ready(
                    {
                        "geometry_summary": geometry.to_dict("records"),
                        "threshold_sensitivity": sensitivity.to_dict("records"),
                    }
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (output / "replay_audit.json").write_text(
            json.dumps(_json_ready(replay_audits), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        saved_events = pd.read_csv(
            output / "episode_measurements.csv.gz", dtype={"candidate": str}
        )
        saved_events["candidate"] = saved_events["candidate"].str.zfill(2)
        saved_geometry = pd.read_csv(
            output / "geometry_summary.csv", dtype={"candidate": str}
        )
        saved_geometry["candidate"] = saved_geometry["candidate"].str.zfill(2)
        saved_sensitivity = pd.read_csv(
            output / "threshold_sensitivity.csv", dtype={"candidate": str}
        )
        saved_sensitivity["candidate"] = saved_sensitivity["candidate"].str.zfill(2)
        metric_audit = independently_recompute_summaries(
            saved_events, saved_geometry, saved_sensitivity
        )
        if not metric_audit["all_within_1e-14"]:
            raise AssertionError("saved episode-coherence summaries failed readback")
        (output / "metric_recomputation_audit.json").write_text(
            json.dumps(_json_ready(metric_audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format": AUDIT_FORMAT,
            "status": "post_hoc_exploratory_not_prospective_confirmation",
            "scope": "episode coherence, distinctness, persistence, and second-renewal-within-F12 description",
            "protocol_id": protocol["protocol_id"],
            "protocol_digest": sha256_file(output / "protocol.json"),
            "source_hashes": _source_hashes(),
            "input_bundles": replay_audits,
            "cohorts_pooled_for_claims": False,
            "positive_episodes": len(events),
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "runtime": _runtime_manifest(),
            "claim_boundary": "cannot establish a distinct regime without a new prospective endpoint and cohort",
        }
        (output / "manifest.json").write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_report(output, geometry, sensitivity, replay_audits)
        _write_lay_summary(output, geometry, sensitivity)
        print("[coherence 5/5] Sealing checksums", flush=True)
        write_checksums(output)
    print(
        f"Episode-coherence audit written to {output_directory.resolve()}", flush=True
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-hoc geometry audit of archived JOINT_BREAK_RUN3 episodes"
    )
    parser.add_argument("--scaled5", type=Path, default=Path("results/scaled5"))
    parser.add_argument(
        "--mechconf", type=Path, default=Path("results/mechanistic_confirmation")
    )
    parser.add_argument(
        "--mechconf2", type=Path, default=Path("results/beta_complete_confirmation")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/episode_coherence_audit")
    )
    parser.add_argument("--workers", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sources = (
        CohortSource("scaled5", "CONF", args.scaled5.resolve()),
        CohortSource("MECHCONF", "MECHCONF", args.mechconf.resolve()),
        CohortSource("MECHCONF2", "MECHCONF2", args.mechconf2.resolve()),
    )
    run_audit(sources, args.output, args.workers)


if __name__ == "__main__":
    main()
