"""Fission-clock inheritance renewal and process-family summaries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from math import log

import numpy as np
from numpy.typing import NDArray

from e01_onset_discovery.sustained_inheritance import exact_order_null_probability


@dataclass(frozen=True, slots=True)
class BinaryEpisodeSummary:
    opportunities: int
    positives: int
    positive_fraction: float
    maximum_positive_run: int
    maximum_negative_run: int
    positive_run_lengths: tuple[int, ...]
    negative_run_lengths: tuple[int, ...]
    transition_00: int
    transition_01: int
    transition_10: int
    transition_11: int
    run2_event: bool
    run3_event: bool
    run5_event: bool
    run2_certification_one_based: int | None
    run3_certification_one_based: int | None
    run5_certification_one_based: int | None
    run2_order_null_probability: float
    run3_order_null_probability: float
    run5_order_null_probability: float


def _runs(values: NDArray[np.bool_], state: bool) -> tuple[int, ...]:
    output: list[int] = []
    current = 0
    for value in values:
        if bool(value) is state:
            current += 1
        elif current:
            output.append(current)
            current = 0
    if current:
        output.append(current)
    return tuple(output)


def _first_certification(values: NDArray[np.bool_], required: int) -> int | None:
    current = 0
    for index, value in enumerate(values):
        current = current + 1 if bool(value) else 0
        if current >= required:
            return index + 1
    return None


def summarize_binary_episode(values: Iterable[bool]) -> BinaryEpisodeSummary:
    array = np.asarray(tuple(values), dtype=np.bool_)
    positive_runs = _runs(array, True)
    negative_runs = _runs(array, False)
    if len(array) > 1:
        previous = array[:-1]
        current = array[1:]
        n00 = int((~previous & ~current).sum())
        n01 = int((~previous & current).sum())
        n10 = int((previous & ~current).sum())
        n11 = int((previous & current).sum())
    else:
        n00 = n01 = n10 = n11 = 0
    maximum_positive = max(positive_runs, default=0)
    maximum_negative = max(negative_runs, default=0)
    positives = int(array.sum())
    opportunities = len(array)
    return BinaryEpisodeSummary(
        opportunities=opportunities,
        positives=positives,
        positive_fraction=float(array.mean()) if opportunities else float("nan"),
        maximum_positive_run=maximum_positive,
        maximum_negative_run=maximum_negative,
        positive_run_lengths=positive_runs,
        negative_run_lengths=negative_runs,
        transition_00=n00,
        transition_01=n01,
        transition_10=n10,
        transition_11=n11,
        run2_event=maximum_positive >= 2,
        run3_event=maximum_positive >= 3,
        run5_event=maximum_positive >= 5,
        run2_certification_one_based=_first_certification(array, 2),
        run3_certification_one_based=_first_certification(array, 3),
        run5_certification_one_based=_first_certification(array, 5),
        run2_order_null_probability=exact_order_null_probability(
            opportunities, positives, 2
        ),
        run3_order_null_probability=exact_order_null_probability(
            opportunities, positives, 3
        ),
        run5_order_null_probability=exact_order_null_probability(
            opportunities, positives, 5
        ),
    )


def _smoothed_probability(successes: int, trials: int) -> float:
    return (successes + 0.5) / (trials + 1.0)


def _log_probability(value: bool, probability: float) -> float:
    probability = float(np.clip(probability, 1e-12, 1 - 1e-12))
    return log(probability if value else 1.0 - probability)


def fit_iid_probability(sequences: Iterable[Iterable[bool]]) -> float:
    arrays = [np.asarray(tuple(sequence), dtype=np.bool_) for sequence in sequences]
    positives = sum(int(array.sum()) for array in arrays)
    trials = sum(len(array) for array in arrays)
    return _smoothed_probability(positives, trials)


def fit_markov_probabilities(
    sequences: Iterable[Iterable[bool]],
) -> tuple[float, float]:
    n00 = n01 = n10 = n11 = 0
    for sequence in sequences:
        array = np.asarray(tuple(sequence), dtype=np.bool_)
        if len(array) < 2:
            continue
        previous = array[:-1]
        current = array[1:]
        n00 += int((~previous & ~current).sum())
        n01 += int((~previous & current).sum())
        n10 += int((previous & ~current).sum())
        n11 += int((previous & current).sum())
    return _smoothed_probability(n01, n00 + n01), _smoothed_probability(n11, n10 + n11)


def score_iid_log_loss(
    sequences: Iterable[Iterable[bool]], probability: float
) -> tuple[float, int]:
    loss = 0.0
    count = 0
    for sequence in sequences:
        array = np.asarray(tuple(sequence), dtype=np.bool_)
        for value in array[1:]:
            loss -= _log_probability(bool(value), probability)
            count += 1
    return (loss / count if count else float("nan")), count


def score_markov_log_loss(
    sequences: Iterable[Iterable[bool]],
    probability_after_zero: float,
    probability_after_one: float,
) -> tuple[float, int]:
    loss = 0.0
    count = 0
    for sequence in sequences:
        array = np.asarray(tuple(sequence), dtype=np.bool_)
        for previous, value in pairwise(array):
            probability = (
                probability_after_one if bool(previous) else probability_after_zero
            )
            loss -= _log_probability(bool(value), probability)
            count += 1
    return (loss / count if count else float("nan")), count


def crossfit_markov_gain_bits(
    half_a: Iterable[Iterable[bool]], half_b: Iterable[Iterable[bool]]
) -> dict[str, float]:
    a = [tuple(sequence) for sequence in half_a]
    b = [tuple(sequence) for sequence in half_b]
    iid_a = fit_iid_probability(a)
    iid_b = fit_iid_probability(b)
    markov_a = fit_markov_probabilities(a)
    markov_b = fit_markov_probabilities(b)
    iid_loss_b, count_b = score_iid_log_loss(b, iid_a)
    markov_loss_b, _ = score_markov_log_loss(b, *markov_a)
    iid_loss_a, count_a = score_iid_log_loss(a, iid_b)
    markov_loss_a, _ = score_markov_log_loss(a, *markov_b)
    total = count_a + count_b
    if not total:
        return {
            "iidLogLoss": float("nan"),
            "markovLogLoss": float("nan"),
            "markovGainBitsPerTransition": float("nan"),
            "transitions": 0,
        }
    iid_loss = (iid_loss_a * count_a + iid_loss_b * count_b) / total
    markov_loss = (markov_loss_a * count_a + markov_loss_b * count_b) / total
    return {
        "iidLogLoss": iid_loss,
        "markovLogLoss": markov_loss,
        "markovGainBitsPerTransition": (iid_loss - markov_loss) / log(2),
        "transitions": total,
    }
