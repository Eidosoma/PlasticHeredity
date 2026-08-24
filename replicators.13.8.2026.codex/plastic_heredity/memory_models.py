"""Support-matched IID, Markov, and duration-aware inheritance models.

Every registered model is fitted and scored on the same transition
destinations.  The deliberately mismatched legacy IID estimator is retained
only for the retrospective diagnostic of the reviewer-identified bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

DURATION_BINS = (1, 2, 3, 4, 5)
SMOOTHING_ALPHA = 1.0
SMOOTHING_BETA = 1.0


def _as_bool_array(sequence: Sequence[bool]) -> NDArray[np.bool_]:
    return np.asarray(tuple(sequence), dtype=bool)


def _smoothed_probability(positives: int, trials: int) -> float:
    if positives < 0 or trials < 0 or positives > trials:
        raise ValueError("invalid Bernoulli counts")
    return float(
        (positives + SMOOTHING_ALPHA)
        / (trials + SMOOTHING_ALPHA + SMOOTHING_BETA)
    )


def duration_bin(sequence: Sequence[bool], destination_index: int) -> int:
    """Return the 1,2,3,4,5+ run length before one destination.

    ``destination_index`` addresses the symbol being predicted.  The run ends
    at the immediately preceding symbol and is computed using past symbols
    only.  The return value is in ``DURATION_BINS`` with 5 representing 5+.
    """

    values = _as_bool_array(sequence)
    if not 1 <= destination_index < values.size:
        raise ValueError("destination index must identify a scored transition")
    previous = bool(values[destination_index - 1])
    length = 1
    cursor = destination_index - 2
    while cursor >= 0 and bool(values[cursor]) == previous:
        length += 1
        cursor -= 1
    return min(length, 5)


def transition_arrays(
    sequences: Iterable[Sequence[bool]],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.int8]]:
    """Flatten common transition support into previous/destination/duration."""

    arrays = [_as_bool_array(sequence) for sequence in sequences]
    count = sum(max(array.size - 1, 0) for array in arrays)
    previous = np.empty(count, dtype=bool)
    destination = np.empty(count, dtype=bool)
    durations = np.empty(count, dtype=np.int8)
    offset = 0
    for array in arrays:
        transitions = max(array.size - 1, 0)
        if transitions == 0:
            continue
        stop = offset + transitions
        previous[offset:stop] = array[:-1]
        destination[offset:stop] = array[1:]
        run_length = 1
        for local_index in range(1, array.size):
            durations[offset + local_index - 1] = min(run_length, 5)
            if bool(array[local_index]) == bool(array[local_index - 1]):
                run_length += 1
            else:
                run_length = 1
        offset = stop
    return previous, destination, durations


def fit_iid_probability(sequences: Iterable[Sequence[bool]]) -> float:
    """Fit IID on precisely the destinations that will be scored."""

    _, destinations, _ = transition_arrays(sequences)
    return _smoothed_probability(int(destinations.sum()), int(destinations.size))


def fit_legacy_mismatched_iid_probability(
    sequences: Iterable[Sequence[bool]],
) -> float:
    """Reviewer-identified old behavior, available only for diagnosis."""

    arrays = [_as_bool_array(sequence) for sequence in sequences]
    positives = sum(int(array.sum()) for array in arrays)
    trials = sum(int(array.size) for array in arrays)
    return _smoothed_probability(positives, trials)


@dataclass(frozen=True)
class FittedMemoryModels:
    iid_probability: float
    markov_probabilities: NDArray[np.float64]
    semimarkov_probabilities: NDArray[np.float64]
    iid_positives: int
    iid_trials: int
    markov_positives: NDArray[np.int64]
    markov_trials: NDArray[np.int64]
    semimarkov_positives: NDArray[np.int64]
    semimarkov_trials: NDArray[np.int64]
    legacy_iid_probability: float | None = None
    legacy_iid_positives: int | None = None
    legacy_iid_trials: int | None = None

    def __post_init__(self) -> None:
        if self.markov_probabilities.shape != (2,):
            raise ValueError("Markov probabilities must have shape (2,)")
        if self.semimarkov_probabilities.shape != (2, 5):
            raise ValueError("semi-Markov probabilities must have shape (2, 5)")
        common = (
            self.iid_trials,
            int(self.markov_trials.sum()),
            int(self.semimarkov_trials.sum()),
        )
        if len(set(common)) != 1:
            raise AssertionError(f"model fitting supports differ: {common}")


def fit_memory_models(
    sequences: Iterable[Sequence[bool]], *, include_legacy_iid: bool = False
) -> FittedMemoryModels:
    arrays = [_as_bool_array(sequence) for sequence in sequences]
    previous, destination, durations = transition_arrays(arrays)
    iid_positives = int(destination.sum())
    iid_trials = int(destination.size)

    markov_positives = np.zeros(2, dtype=np.int64)
    markov_trials = np.zeros(2, dtype=np.int64)
    semimarkov_positives = np.zeros((2, 5), dtype=np.int64)
    semimarkov_trials = np.zeros((2, 5), dtype=np.int64)
    for previous_value in (0, 1):
        selected_previous = previous == bool(previous_value)
        markov_trials[previous_value] = int(selected_previous.sum())
        markov_positives[previous_value] = int(destination[selected_previous].sum())
        for bin_value in DURATION_BINS:
            selected = selected_previous & (durations == bin_value)
            semimarkov_trials[previous_value, bin_value - 1] = int(selected.sum())
            semimarkov_positives[previous_value, bin_value - 1] = int(
                destination[selected].sum()
            )

    markov_probability = np.asarray(
        [
            _smoothed_probability(
                int(markov_positives[value]), int(markov_trials[value])
            )
            for value in (0, 1)
        ],
        dtype=np.float64,
    )
    semimarkov_probability = np.empty((2, 5), dtype=np.float64)
    for previous_value in (0, 1):
        for bin_value in DURATION_BINS:
            semimarkov_probability[previous_value, bin_value - 1] = (
                _smoothed_probability(
                    int(semimarkov_positives[previous_value, bin_value - 1]),
                    int(semimarkov_trials[previous_value, bin_value - 1]),
                )
            )

    legacy_probability: float | None = None
    legacy_positives: int | None = None
    legacy_trials: int | None = None
    if include_legacy_iid:
        legacy_positives = sum(int(array.sum()) for array in arrays)
        legacy_trials = sum(int(array.size) for array in arrays)
        legacy_probability = _smoothed_probability(legacy_positives, legacy_trials)

    return FittedMemoryModels(
        iid_probability=_smoothed_probability(iid_positives, iid_trials),
        markov_probabilities=markov_probability,
        semimarkov_probabilities=semimarkov_probability,
        iid_positives=iid_positives,
        iid_trials=iid_trials,
        markov_positives=markov_positives,
        markov_trials=markov_trials,
        semimarkov_positives=semimarkov_positives,
        semimarkov_trials=semimarkov_trials,
        legacy_iid_probability=legacy_probability,
        legacy_iid_positives=legacy_positives,
        legacy_iid_trials=legacy_trials,
    )


def model_probabilities(
    fitted: FittedMemoryModels,
    previous: NDArray[np.bool_],
    durations: NDArray[np.int8],
) -> dict[str, NDArray[np.float64]]:
    previous_index = np.asarray(previous, dtype=np.int8)
    duration_index = np.asarray(durations, dtype=np.int8) - 1
    if np.any((duration_index < 0) | (duration_index >= 5)):
        raise ValueError("duration bins must be in 1..5")
    output = {
        "iid": np.full(previous_index.size, fitted.iid_probability, dtype=np.float64),
        "markov": fitted.markov_probabilities[previous_index],
        "semimarkov": fitted.semimarkov_probabilities[
            previous_index, duration_index
        ],
    }
    if fitted.legacy_iid_probability is not None:
        output["legacy_iid"] = np.full(
            previous_index.size,
            fitted.legacy_iid_probability,
            dtype=np.float64,
        )
    return output


def binary_log_loss_bits(
    destination: NDArray[np.bool_], probability: NDArray[np.float64]
) -> NDArray[np.float64]:
    target = np.asarray(destination, dtype=np.float64)
    prediction = np.asarray(probability, dtype=np.float64)
    if target.shape != prediction.shape:
        raise ValueError("targets and probabilities must have identical shapes")
    if np.any((prediction <= 0.0) | (prediction >= 1.0)):
        raise ValueError("smoothed probabilities must lie strictly inside (0,1)")
    return -(target * np.log2(prediction) + (1.0 - target) * np.log2(1.0 - prediction))


def fitted_model_rows(
    fitted: FittedMemoryModels,
    *,
    candidate: str,
    direction: str,
    train_fold: int,
    test_fold: int,
) -> list[dict[str, object]]:
    common = {
        "candidate": candidate,
        "direction": direction,
        "train_fold": train_fold,
        "test_fold": test_fold,
    }
    rows: list[dict[str, object]] = [
        {
            **common,
            "model": "iid",
            "previous": None,
            "duration_bin": None,
            "positives": fitted.iid_positives,
            "trials": fitted.iid_trials,
            "probability": fitted.iid_probability,
        }
    ]
    for previous in (0, 1):
        rows.append(
            {
                **common,
                "model": "markov",
                "previous": previous,
                "duration_bin": None,
                "positives": int(fitted.markov_positives[previous]),
                "trials": int(fitted.markov_trials[previous]),
                "probability": float(fitted.markov_probabilities[previous]),
            }
        )
        for bin_value in DURATION_BINS:
            rows.append(
                {
                    **common,
                    "model": "semimarkov",
                    "previous": previous,
                    "duration_bin": "5+" if bin_value == 5 else str(bin_value),
                    "positives": int(
                        fitted.semimarkov_positives[previous, bin_value - 1]
                    ),
                    "trials": int(
                        fitted.semimarkov_trials[previous, bin_value - 1]
                    ),
                    "probability": float(
                        fitted.semimarkov_probabilities[previous, bin_value - 1]
                    ),
                }
            )
    if fitted.legacy_iid_probability is not None:
        rows.append(
            {
                **common,
                "model": "legacy_mismatched_iid",
                "previous": None,
                "duration_bin": None,
                "positives": fitted.legacy_iid_positives,
                "trials": fitted.legacy_iid_trials,
                "probability": fitted.legacy_iid_probability,
            }
        )
    return rows
