"""Shared substrate-neutral scoring and integrity helpers.

This module intentionally has no dependency on :mod:`plastic_heredity`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.stats import rankdata


BREAK_HORIZON = 12
MAX_FUTURE_HORIZON = 16
RENEWAL_RUN = 3
MASTER_SEED = "20260820-cross-substrate-ca-v1"


@dataclass(frozen=True)
class FutureOutcome:
    event: bool
    break_index: int | None
    renewal_start: int | None
    observed_boundaries: int
    inherited_count: int
    complete_horizon: bool


@dataclass(frozen=True)
class BoundaryRecord:
    index: int
    similarity: float
    inherited: bool
    parent_size: int
    child_size: int
    elapsed_updates: int
    ambiguous: bool = False


def json_ready(value: Any) -> Any:
    """Convert NumPy and dataclass values to strict JSON values."""

    if hasattr(value, "__dataclass_fields__"):
        return json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        json_ready(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(*parts: object, master: str = MASTER_SEED) -> int:
    material = "\x1f".join([master, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def score_break_renewal(
    similarities: Sequence[float],
    threshold: float,
    *,
    horizon: int = BREAK_HORIZON,
    run_length: int = RENEWAL_RUN,
    complete_horizon: bool | None = None,
) -> FutureOutcome:
    """Score a strict break followed later by an uninterrupted inherited run."""

    values = np.asarray(similarities[:horizon], dtype=np.float64)
    observed = int(values.size)
    if complete_horizon is None:
        complete_horizon = observed >= horizon
    inherited = values > float(threshold)
    breaks = np.flatnonzero(~inherited)
    break_index: int | None = int(breaks[0]) if breaks.size else None
    renewal_start: int | None = None
    if break_index is not None:
        for start in range(break_index + 1, observed - run_length + 1):
            if bool(np.all(inherited[start : start + run_length])):
                renewal_start = start
                break
    return FutureOutcome(
        event=renewal_start is not None,
        break_index=break_index,
        renewal_start=renewal_start,
        observed_boundaries=observed,
        inherited_count=int(inherited.sum()),
        complete_horizon=bool(complete_horizon),
    )


def score_binary_break_renewal(
    inherited: Sequence[bool], run_length: int = RENEWAL_RUN
) -> bool:
    values = tuple(bool(value) for value in inherited)
    try:
        first_break = values.index(False)
    except ValueError:
        return False
    return any(
        all(values[start : start + run_length])
        for start in range(first_break + 1, len(values) - run_length + 1)
    )


@lru_cache(maxsize=None)
def exact_order_null_probability(
    length: int, inherited_count: int, run_length: int = RENEWAL_RUN
) -> float:
    """Exact event probability over unique flag orderings with fixed counts."""

    if length < 0 or inherited_count < 0 or inherited_count > length:
        raise ValueError("invalid binary-sequence counts")
    total = 0
    positive = 0
    for positions in combinations(range(length), inherited_count):
        values = [False] * length
        for position in positions:
            values[position] = True
        total += 1
        positive += int(score_binary_break_renewal(values, run_length))
    return float(positive / total) if total else 0.0


def connected_components_torus(
    occupied: NDArray[np.bool_], *, min_size: int = 1
) -> list[NDArray[np.int32]]:
    """Four-neighbour components on a periodic rectangular lattice."""

    mask = np.asarray(occupied, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("occupied mask must be two dimensional")
    structure = np.asarray(
        ((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8
    )
    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return []

    parents = np.arange(count + 1, dtype=np.int32)

    def find(label: int) -> int:
        while int(parents[label]) != label:
            parents[label] = parents[int(parents[label])]
            label = int(parents[label])
        return label

    def union(left: int, right: int) -> None:
        if left == 0 or right == 0:
            return
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parents[max(root_left, root_right)] = min(root_left, root_right)

    for row in np.flatnonzero(mask[:, 0] & mask[:, -1]):
        union(int(labels[row, 0]), int(labels[row, -1]))
    for col in np.flatnonzero(mask[0, :] & mask[-1, :]):
        union(int(labels[0, col]), int(labels[-1, col]))
    roots = np.asarray([find(label) for label in range(count + 1)], dtype=np.int32)
    points = np.argwhere(labels > 0).astype(np.int32, copy=False)
    point_roots = roots[labels[points[:, 0], points[:, 1]]]
    output: list[NDArray[np.int32]] = []
    for root in np.unique(point_roots):
        component = points[point_roots == root]
        if component.shape[0] >= min_size:
            output.append(component)
    output.sort(key=lambda item: (-item.shape[0], int(item[:, 0].min()), int(item[:, 1].min())))
    return output


def _periodic_axis_coordinates(values: NDArray[np.int32], size: int) -> NDArray[np.int32]:
    unique = np.unique(values)
    if unique.size <= 1:
        start = int(unique[0]) if unique.size else 0
        return (values - start) % size
    gaps = (np.roll(unique, -1) - unique) % size
    cut = int(np.argmax(gaps))
    start = int(unique[(cut + 1) % unique.size])
    return (values - start) % size


def crop_component(
    grid: NDArray[np.integer], component: NDArray[np.int32]
) -> NDArray[np.uint8]:
    """Extract the minimal periodic crop containing one component."""

    array = np.asarray(grid)
    if component.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    rr = _periodic_axis_coordinates(component[:, 0], array.shape[0])
    cc = _periodic_axis_coordinates(component[:, 1], array.shape[1])
    crop = np.zeros((int(rr.max()) + 1, int(cc.max()) + 1), dtype=np.uint8)
    crop[rr, cc] = array[component[:, 0], component[:, 1]].astype(np.uint8)
    return crop


def _centroid_canvas(array: NDArray[np.uint8], side: int) -> NDArray[np.uint8]:
    """Place foreground so its integer-rounded centroid is at canvas centre."""

    canvas = np.zeros((side, side), dtype=np.uint8)
    points = np.argwhere(array != 0)
    if not points.size:
        return canvas
    target = np.asarray((side // 2, side // 2), dtype=np.int64)
    offset = target - np.rint(points.mean(axis=0)).astype(np.int64)
    shifted = points + offset
    if np.any(shifted < 0) or np.any(shifted >= side):
        raise ValueError("centroid-aligned individual does not fit on canvas")
    canvas[shifted[:, 0], shifted[:, 1]] = array[points[:, 0], points[:, 1]]
    return canvas


def _shift_without_wrap(
    array: NDArray[np.uint8], row_shift: int, col_shift: int
) -> NDArray[np.uint8]:
    output = np.zeros_like(array)
    src_r0 = max(0, -row_shift)
    src_r1 = min(array.shape[0], array.shape[0] - row_shift)
    src_c0 = max(0, -col_shift)
    src_c1 = min(array.shape[1], array.shape[1] - col_shift)
    if src_r1 <= src_r0 or src_c1 <= src_c0:
        return output
    output[
        src_r0 + row_shift : src_r1 + row_shift,
        src_c0 + col_shift : src_c1 + col_shift,
    ] = array[src_r0:src_r1, src_c0:src_c1]
    return output


def canonical_similarity(
    left: NDArray[np.integer], right: NDArray[np.integer]
) -> float:
    """C4/translation-invariant cosine of non-background one-hot rasters."""

    a = np.asarray(left, dtype=np.uint8)
    b = np.asarray(right, dtype=np.uint8)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("individual rasters must be two dimensional")
    if not np.any(a) or not np.any(b):
        return 0.0
    # Twice the largest extent leaves room for asymmetric individuals whose
    # foreground centroid lies close to one crop edge, plus the registered
    # one-cell residual translation search.
    side = 2 * max(a.shape + b.shape) + 6
    aa = _centroid_canvas(a, side)
    left_norm = float(np.sqrt(np.count_nonzero(aa)))
    best = 0.0
    for turns in range(4):
        rotated = np.rot90(b, turns)
        base = _centroid_canvas(rotated, side)
        for row_shift, col_shift in product((-1, 0, 1), repeat=2):
            shifted = _shift_without_wrap(base, row_shift, col_shift)
            denominator = left_norm * float(np.sqrt(np.count_nonzero(shifted)))
            if denominator:
                overlap = (aa != 0) & (aa == shifted)
                score = float(np.count_nonzero(overlap) / denominator)
                best = max(best, score)
    return min(1.0, best)


def calibrated_threshold(stranger_similarities: Sequence[float]) -> float:
    values = np.asarray(stranger_similarities, dtype=np.float64)
    if values.size < 1 or not np.isfinite(values).all():
        raise ValueError("finite stranger similarities are required")
    return float(np.quantile(values, 0.95, method="higher"))


def block_bootstrap_interval(
    values: Sequence[float],
    blocks: Sequence[object],
    *,
    repetitions: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    labels = np.asarray(blocks)
    if data.shape[0] != labels.shape[0] or data.size == 0:
        raise ValueError("values and blocks must be nonempty and aligned")
    unique = np.unique(labels)
    # The preregistered sampling unit is the independently seeded world
    # block, so every block contributes one mean regardless of how many
    # completed branches or boundaries it contains.
    block_means = np.asarray([data[labels == label].mean() for label in unique])
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        picks = rng.integers(0, block_means.size, size=block_means.size)
        draws[index] = float(block_means[picks].mean())
    tail = (1.0 - confidence) / 2.0
    return tuple(float(value) for value in np.quantile(draws, [tail, 1.0 - tail]))


def sign_randomization_p(
    block_effects: Sequence[float], *, repetitions: int, seed: int
) -> float:
    values = np.asarray(block_effects, dtype=np.float64)
    if values.size == 0:
        raise ValueError("block effects are required")
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(repetitions):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=values.size)
        extreme += int(float(np.mean(values * signs)) >= observed)
    return float((extreme + 1) / (repetitions + 1))


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.size != b.size or a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    ra = rankdata(a)
    rb = rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def block_center(values: Sequence[float], blocks: Sequence[object]) -> NDArray[np.float64]:
    data = np.asarray(values, dtype=np.float64)
    labels = np.asarray(blocks)
    output = data.copy()
    for label in np.unique(labels):
        mask = labels == label
        output[mask] -= output[mask].mean()
    return output


def checksum_lines(directory: Path) -> list[str]:
    lines: list[str] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS":
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(directory)}")
    return lines
