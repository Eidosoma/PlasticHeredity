"""Matrix-cross-fitted scoring and inference for inheritance memory models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from .memory_models import (
    binary_log_loss_bits,
    fit_memory_models,
    fitted_model_rows,
    model_probabilities,
    transition_arrays,
)
from .seeds import derive_seed

PRIMARY_CONTRASTS = {
    "markov_vs_iid": ("iid", "markov"),
    "semimarkov_vs_markov": ("markov", "semimarkov"),
}


@dataclass(frozen=True)
class SequenceRecord:
    state_index: int
    state_id: str
    candidate: str
    matrix_id: int
    landmark: int
    branch: int
    symbols: tuple[bool, ...]
    completed_horizon: bool
    observed_fissions: int
    first_break_index: int

    @property
    def fold(self) -> int:
        """Preassigned catalytic-matrix fold: even IDs 0, odd IDs 1."""

        return self.matrix_id % 2


@dataclass(frozen=True)
class CandidateCrossfitScores:
    candidate: str
    matrix_id: NDArray[np.int64]
    state_index: NDArray[np.int64]
    branch: NDArray[np.int64]
    test_fold: NDArray[np.int8]
    previous: NDArray[np.bool_]
    duration_bin: NDArray[np.int8]
    destination: NDArray[np.bool_]
    losses: dict[str, NDArray[np.float64]]

    def __post_init__(self) -> None:
        sizes = {
            self.matrix_id.size,
            self.state_index.size,
            self.branch.size,
            self.test_fold.size,
            self.previous.size,
            self.duration_bin.size,
            self.destination.size,
            *(values.size for values in self.losses.values()),
        }
        if len(sizes) != 1:
            raise ValueError(f"cross-fitted score arrays differ in size: {sizes}")
        required = {"iid", "markov", "semimarkov"}
        if not required.issubset(self.losses):
            raise ValueError(f"missing model losses: {required - set(self.losses)}")
        if not all(np.isfinite(values).all() for values in self.losses.values()):
            raise ValueError("cross-fitted losses must be finite")


def _direction(train_fold: int, test_fold: int) -> str:
    names = {0: "even", 1: "odd"}
    return f"{names[train_fold]}_to_{names[test_fold]}"


def _score_test_records(
    candidate: str,
    records: list[SequenceRecord],
    fitted: Any,
    test_fold: int,
) -> CandidateCrossfitScores:
    sequences = [record.symbols for record in records]
    previous, destination, durations = transition_arrays(sequences)
    count = destination.size
    matrix_id = np.empty(count, dtype=np.int64)
    state_index = np.empty(count, dtype=np.int64)
    branch = np.empty(count, dtype=np.int64)
    folds = np.full(count, test_fold, dtype=np.int8)
    offset = 0
    for record in records:
        transitions = max(len(record.symbols) - 1, 0)
        if transitions == 0:
            continue
        stop = offset + transitions
        matrix_id[offset:stop] = record.matrix_id
        state_index[offset:stop] = record.state_index
        branch[offset:stop] = record.branch
        offset = stop
    if offset != count:
        raise AssertionError("transition metadata did not cover scored destinations")
    probabilities = model_probabilities(fitted, previous, durations)
    losses = {
        name: binary_log_loss_bits(destination, probability)
        for name, probability in probabilities.items()
    }
    return CandidateCrossfitScores(
        candidate=candidate,
        matrix_id=matrix_id,
        state_index=state_index,
        branch=branch,
        test_fold=folds,
        previous=previous,
        duration_bin=durations,
        destination=destination,
        losses=losses,
    )


def _concatenate_scores(
    candidate: str, parts: list[CandidateCrossfitScores]
) -> CandidateCrossfitScores:
    loss_names = set(parts[0].losses)
    if any(set(part.losses) != loss_names for part in parts):
        raise ValueError("cross-fit directions produced different model sets")
    return CandidateCrossfitScores(
        candidate=candidate,
        matrix_id=np.concatenate([part.matrix_id for part in parts]),
        state_index=np.concatenate([part.state_index for part in parts]),
        branch=np.concatenate([part.branch for part in parts]),
        test_fold=np.concatenate([part.test_fold for part in parts]),
        previous=np.concatenate([part.previous for part in parts]),
        duration_bin=np.concatenate([part.duration_bin for part in parts]),
        destination=np.concatenate([part.destination for part in parts]),
        losses={
            name: np.concatenate([part.losses[name] for part in parts])
            for name in sorted(loss_names)
        },
    )


def crossfit_memory_models(
    records: Iterable[SequenceRecord], *, include_legacy_iid: bool = False
) -> tuple[dict[str, CandidateCrossfitScores], list[dict[str, object]]]:
    """Two-way cross-fit by whole catalytic matrix, separately by candidate."""

    all_records = list(records)
    candidates = sorted({record.candidate for record in all_records})
    scores: dict[str, CandidateCrossfitScores] = {}
    fit_rows: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_records = [
            record for record in all_records if record.candidate == candidate
        ]
        observed_folds = {record.fold for record in candidate_records}
        if observed_folds != {0, 1}:
            raise ValueError(
                f"candidate {candidate} must contain both matrix folds; got {observed_folds}"
            )
        parts: list[CandidateCrossfitScores] = []
        for test_fold in (0, 1):
            train_fold = 1 - test_fold
            training = [
                record for record in candidate_records if record.fold == train_fold
            ]
            testing = [record for record in candidate_records if record.fold == test_fold]
            fitted = fit_memory_models(
                [record.symbols for record in training],
                include_legacy_iid=include_legacy_iid,
            )
            direction = _direction(train_fold, test_fold)
            fit_rows.extend(
                fitted_model_rows(
                    fitted,
                    candidate=candidate,
                    direction=direction,
                    train_fold=train_fold,
                    test_fold=test_fold,
                )
            )
            parts.append(_score_test_records(candidate, testing, fitted, test_fold))
        scores[candidate] = _concatenate_scores(candidate, parts)
    return scores, fit_rows


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=np.float64)
    values = np.where(np.isfinite(values), values, 1.0)
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty_like(values)
    running = 0.0
    total = values.size
    for rank, index in enumerate(order):
        running = max(running, float(values[index]) * (total - rank))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _matrix_sums(
    difference: NDArray[np.float64], matrix_ids: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    unique, inverse = np.unique(matrix_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=difference, minlength=unique.size).astype(
        np.float64
    )
    counts = np.bincount(inverse, minlength=unique.size).astype(np.int64)
    return unique, sums, counts


def _weighted_gain_interval(
    difference: NDArray[np.float64],
    matrix_ids: NDArray[np.int64],
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, tuple[float, float]]:
    if difference.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    _, sums, counts = _matrix_sums(difference, matrix_ids)
    observed = float(sums.sum() / counts.sum())
    sampled = rng.integers(0, sums.size, size=(repetitions, sums.size))
    numerators = sums[sampled].sum(axis=1)
    denominators = counts[sampled].sum(axis=1)
    bootstrap = numerators / denominators
    interval = np.quantile(bootstrap, (0.025, 0.975))
    return observed, (float(interval[0]), float(interval[1]))


def _paired_matrix_randomization_p(
    difference: NDArray[np.float64],
    matrix_ids: NDArray[np.int64],
    repetitions: int,
    rng: np.random.Generator,
) -> float:
    if difference.size == 0:
        return 1.0
    _, sums, counts = _matrix_sums(difference, matrix_ids)
    observed = float(sums.sum() / counts.sum())
    signs = rng.integers(0, 2, size=(repetitions, sums.size), dtype=np.int8)
    signs = signs.astype(np.float64) * 2.0 - 1.0
    null = (signs @ sums) / counts.sum()
    return float((np.count_nonzero(null >= observed) + 1) / (repetitions + 1))


def _state_macro_gain(
    difference: NDArray[np.float64], state_indices: NDArray[np.int64]
) -> float:
    if difference.size == 0:
        return float("nan")
    unique, inverse = np.unique(state_indices, return_inverse=True)
    sums = np.bincount(inverse, weights=difference, minlength=unique.size)
    counts = np.bincount(inverse, minlength=unique.size)
    return float(np.mean(sums / counts))


def _contrast_summary(
    score: CandidateCrossfitScores,
    baseline: str,
    enhanced: str,
    repetitions: int,
    master_seed: str,
    seed_domain: str,
) -> dict[str, Any]:
    difference = score.losses[baseline] - score.losses[enhanced]
    observed, interval = _weighted_gain_interval(
        difference,
        score.matrix_id,
        repetitions,
        np.random.default_rng(
            derive_seed(master_seed, f"{seed_domain}.bootstrap", score.candidate)
        ),
    )
    p_value = _paired_matrix_randomization_p(
        difference,
        score.matrix_id,
        repetitions,
        np.random.default_rng(
            derive_seed(master_seed, f"{seed_domain}.randomization", score.candidate)
        ),
    )
    direction_results: dict[str, Any] = {}
    for test_fold in (0, 1):
        selected = score.test_fold == test_fold
        train_fold = 1 - test_fold
        name = _direction(train_fold, test_fold)
        gain, fold_interval = _weighted_gain_interval(
            difference[selected],
            score.matrix_id[selected],
            repetitions,
            np.random.default_rng(
                derive_seed(
                    master_seed,
                    f"{seed_domain}.direction_bootstrap",
                    score.candidate,
                    name,
                )
            ),
        )
        direction_results[name] = {
            "test_fold": test_fold,
            "gain_bits_per_transition": gain,
            "gain_ci95": fold_interval,
            "transitions": int(selected.sum()),
            "matrices": int(np.unique(score.matrix_id[selected]).size),
        }
    return {
        "gain_bits_per_transition": observed,
        "gain_ci95": interval,
        "equal_state_macro_gain_bits": _state_macro_gain(
            difference, score.state_index
        ),
        "randomization_p_raw": p_value,
        "directions": direction_results,
        "transitions": int(difference.size),
        "states_with_transitions": int(np.unique(score.state_index).size),
        "matrices": int(np.unique(score.matrix_id).size),
    }


def compute_memory_metrics(
    scores: dict[str, CandidateCrossfitScores],
    *,
    repetitions: int,
    master_seed: str,
    confirmatory: bool,
) -> dict[str, Any]:
    """Compute registered transition-weighted tests and diagnostic contrasts."""

    primary_rows: list[dict[str, Any]] = []
    candidates: dict[str, Any] = {}
    for candidate, score in sorted(scores.items()):
        model_scores = {
            name: {
                "pooled_bits_per_transition": float(loss.mean()),
                "even_test_bits_per_transition": float(
                    loss[score.test_fold == 0].mean()
                ),
                "odd_test_bits_per_transition": float(
                    loss[score.test_fold == 1].mean()
                ),
            }
            for name, loss in score.losses.items()
        }
        candidates[candidate] = {
            "transitions": int(score.destination.size),
            "positive_destination_fraction": float(score.destination.mean()),
            "model_scores": model_scores,
        }
        for contrast, (baseline, enhanced) in PRIMARY_CONTRASTS.items():
            summary = _contrast_summary(
                score,
                baseline,
                enhanced,
                repetitions,
                master_seed,
                f"memory.{contrast}",
            )
            primary_rows.append(
                {
                    "contrast": contrast,
                    "baseline": baseline,
                    "enhanced": enhanced,
                    "candidate": candidate,
                    **summary,
                }
            )

        if "legacy_iid" in score.losses:
            candidates[candidate]["support_mismatch_diagnostic"] = {
                "legacy_markov_gain": _contrast_summary(
                    score,
                    "legacy_iid",
                    "markov",
                    repetitions,
                    master_seed,
                    "memory.legacy_markov",
                ),
                "corrected_markov_gain": _contrast_summary(
                    score,
                    "iid",
                    "markov",
                    repetitions,
                    master_seed,
                    "memory.corrected_markov",
                ),
                "legacy_minus_corrected_iid_loss": _contrast_summary(
                    score,
                    "legacy_iid",
                    "iid",
                    repetitions,
                    master_seed,
                    "memory.support_mismatch",
                ),
            }

    adjusted = holm_adjust(row["randomization_p_raw"] for row in primary_rows)
    for row, adjusted_value in zip(primary_rows, adjusted):
        row["randomization_p_holm"] = adjusted_value
        direction_positive = all(
            values["gain_bits_per_transition"] > 0.0
            for values in row["directions"].values()
        )
        row["both_directions_positive"] = direction_positive
        row["passes_gate"] = bool(
            confirmatory
            and direction_positive
            and row["gain_bits_per_transition"] > 0.0
            and row["gain_ci95"][0] > 0.0
            and adjusted_value < 0.05
        )
    support = {
        contrast: bool(
            confirmatory
            and all(
                row["passes_gate"]
                for row in primary_rows
                if row["contrast"] == contrast
            )
        )
        for contrast in PRIMARY_CONTRASTS
    }
    return {
        "confirmatory": confirmatory,
        "candidates": candidates,
        "primary_tests": primary_rows,
        "primary_contrasts": PRIMARY_CONTRASTS,
        "family_size": len(primary_rows),
        "support": support,
        "decision_rule": (
            "transition-weighted out-of-matrix gain > 0, matrix-bootstrap lower "
            "95% bound > 0, Holm-adjusted whole-matrix sign-randomization p < "
            "0.05, and positive point gain in both cross-fit directions, for "
            "both simulator candidates"
        ),
    }


def sequence_count_rows(records: Iterable[SequenceRecord]) -> list[dict[str, Any]]:
    all_records = list(records)
    rows: list[dict[str, Any]] = []
    candidates = sorted({record.candidate for record in all_records})
    for candidate in candidates:
        candidate_records = [
            record for record in all_records if record.candidate == candidate
        ]
        for fold_label, fold in (("all", None), ("even", 0), ("odd", 1)):
            selected = (
                candidate_records
                if fold is None
                else [record for record in candidate_records if record.fold == fold]
            )
            lengths = np.asarray([len(record.symbols) for record in selected])
            has_break = np.asarray(
                [record.first_break_index >= 0 for record in selected], dtype=bool
            )
            rows.append(
                {
                    "candidate": candidate,
                    "matrix_fold": fold_label,
                    "futures": len(selected),
                    "completed_horizon": sum(
                        int(record.completed_horizon) for record in selected
                    ),
                    "extinctions": sum(
                        int(not record.completed_horizon) for record in selected
                    ),
                    "no_break": sum(
                        int(record.first_break_index < 0) for record in selected
                    ),
                    "empty_post_break_suffix": int(
                        (has_break & (lengths == 0)).sum()
                    ),
                    "singleton_post_break_suffix": int((lengths == 1).sum()),
                    "usable_suffixes": int((lengths >= 2).sum()),
                    "scored_transitions": int(np.maximum(lengths - 1, 0).sum()),
                }
            )
    return rows


def score_archive_arrays(
    scores: dict[str, CandidateCrossfitScores]
) -> dict[str, NDArray[Any]]:
    arrays: dict[str, NDArray[Any]] = {}
    for candidate, score in sorted(scores.items()):
        prefix = f"c{candidate}"
        for name in (
            "matrix_id",
            "state_index",
            "branch",
            "test_fold",
            "previous",
            "duration_bin",
            "destination",
        ):
            arrays[f"{prefix}__{name}"] = np.asarray(getattr(score, name))
        for name, values in sorted(score.losses.items()):
            arrays[f"{prefix}__loss_{name}"] = np.asarray(values)
    return arrays


def calibration_rows(
    scores: dict[str, CandidateCrossfitScores]
) -> list[dict[str, Any]]:
    """Held-out observed versus fitted probabilities for every model cell."""

    rows: list[dict[str, Any]] = []
    for candidate, score in sorted(scores.items()):
        target = score.destination.astype(np.float64)
        for test_fold in (0, 1):
            fold_selected = score.test_fold == test_fold
            direction = _direction(1 - test_fold, test_fold)
            for model, losses in sorted(score.losses.items()):
                prediction = np.where(
                    score.destination,
                    np.exp2(-losses),
                    1.0 - np.exp2(-losses),
                )
                if model in {"iid", "legacy_iid"}:
                    cells = [(None, None, fold_selected)]
                elif model == "markov":
                    cells = [
                        (previous, None, fold_selected & (score.previous == previous))
                        for previous in (0, 1)
                    ]
                else:
                    cells = [
                        (
                            previous,
                            "5+" if duration == 5 else str(duration),
                            fold_selected
                            & (score.previous == previous)
                            & (score.duration_bin == duration),
                        )
                        for previous in (0, 1)
                        for duration in (1, 2, 3, 4, 5)
                    ]
                for previous, duration, selected in cells:
                    if not selected.any():
                        continue
                    fitted_probability = float(prediction[selected].mean())
                    observed_probability = float(target[selected].mean())
                    rows.append(
                        {
                            "candidate": candidate,
                            "direction": direction,
                            "test_fold": test_fold,
                            "model": model,
                            "previous": previous,
                            "duration_bin": duration,
                            "transitions": int(selected.sum()),
                            "fitted_probability": fitted_probability,
                            "observed_probability": observed_probability,
                            "calibration_error": (
                                fitted_probability - observed_probability
                            ),
                        }
                    )
    return rows
