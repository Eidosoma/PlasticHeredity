from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def average_ranks(values: NDArray) -> NDArray[np.float64]:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def spearman(left: NDArray, right: NDArray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    left_rank = average_ranks(left[valid])
    right_rank = average_ranks(right[valid])
    if left_rank.std() == 0.0 or right_rank.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def matrix_center(values: NDArray, matrix_ids: NDArray) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=np.float64)
    matrix_ids = np.asarray(matrix_ids)
    centered = np.empty_like(values)
    for matrix_id in np.unique(matrix_ids):
        selected = matrix_ids == matrix_id
        centered[selected] = values[selected] - values[selected].mean()
    return centered


def centered_spearman(left: NDArray, right: NDArray, matrix_ids: NDArray) -> float:
    return spearman(matrix_center(left, matrix_ids), matrix_center(right, matrix_ids))


def log_loss_from_q(q: NDArray, prediction: NDArray) -> float:
    q = np.asarray(q, dtype=np.float64)
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(q * np.log(prediction) + (1.0 - q) * np.log(1.0 - prediction)))


def q_brier(q: NDArray, prediction: NDArray) -> float:
    return float(np.mean((np.asarray(q) - np.asarray(prediction)) ** 2))


def bootstrap_by_matrix(
    values: dict[str, NDArray],
    matrix_ids: NDArray,
    statistic: Callable[[dict[str, NDArray], NDArray], float],
    repetitions: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    matrix_ids = np.asarray(matrix_ids)
    unique = np.unique(matrix_ids)
    output = np.empty(repetitions, dtype=np.float64)
    locations = {key: np.flatnonzero(matrix_ids == key) for key in unique}
    for repetition in range(repetitions):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([locations[key] for key in sampled])
        sampled_groups = np.repeat(np.arange(unique.size), [locations[key].size for key in sampled])
        sampled_values = {name: np.asarray(value)[indices] for name, value in values.items()}
        output[repetition] = statistic(sampled_values, sampled_groups)
    return output


def confidence_interval(samples: NDArray, level: float = 0.95) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    finite = np.asarray(samples, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(finite, (alpha, 1.0 - alpha)))


def permute_matrix_blocks(
    values: NDArray,
    matrix_ids: NDArray,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=np.float64)
    matrix_ids = np.asarray(matrix_ids)
    unique = np.unique(matrix_ids)
    blocks = [np.flatnonzero(matrix_ids == key) for key in unique]
    sizes = {block.size for block in blocks}
    if len(sizes) != 1:
        raise ValueError("matrix-block permutation requires equal states per matrix")
    permutation = rng.permutation(len(blocks))
    output = np.empty_like(values)
    for destination, source in enumerate(permutation):
        output[blocks[destination]] = values[blocks[source]]
    return output

