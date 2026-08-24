"""Frozen endpoint and analysis primitives for lineage-identity tests 2--4.

The functions in this module are deliberately free of filesystem access.  The
runner owns provenance, simulation, checkpointing, and reporting; this module
owns the scientific definitions and is exercised by synthetic unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.integer]

INHERITANCE_THRESHOLD = 0.90
DISTINCTNESS_THRESHOLD = 0.85
STRICT_RUN = 8
F12_RUN = 3
WINDOW = 32
BURN_IN = 32
PRIMARY_LINEAGES = 128
MAX_LINEAGES = 256
BANK_SIZE = 20
PRIMARY_RESIDENCE = 8
PRIMARY_START_SUPPORT = 8
PRIMARY_DURABLE_SUPPORT = 4
PRIMARY_SEPARATION = 0.85


@dataclass(frozen=True)
class Episode:
    """One earliest endpoint episode recovered from a lineage."""

    kind: str
    window_index: int
    break_index: int
    run_start: int
    daughters: NDArray[np.uint8]
    anchor: NDArray[np.uint8]

    @property
    def final(self) -> NDArray[np.uint8]:
        return self.daughters[-1]


@dataclass(frozen=True)
class ResidenceEpisode:
    """A maximal run of overlapping coherent residence windows."""

    lineage: int
    start: int
    end: int
    representative: NDArray[np.uint8]

    @property
    def duration(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class StableForm:
    """One internally coherent cross-lineage residence cluster."""

    cluster_id: int
    medoid: NDArray[np.uint8]
    starts: tuple[int, ...]
    episodes: tuple[int, ...]
    durable_starts: tuple[int, ...]


@dataclass(frozen=True)
class CensusResult:
    residence_length: int
    start_support: int
    durable_support: int
    separation: float
    residence_episodes: int
    coherent_clusters: int
    stable_forms: tuple[StableForm, ...]
    distinct_forms: tuple[StableForm, ...]


def cosine(left: NDArray, right: NDArray) -> float:
    left_f = np.asarray(left, dtype=np.float64)
    right_f = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_f) * np.linalg.norm(right_f))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(left_f, right_f) / denominator, 0.0, 1.0))


def cosine_matrix(values: NDArray) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cosine_matrix expects a two-dimensional array")
    norms = np.linalg.norm(matrix, axis=1)
    denominator = np.outer(norms, norms)
    out = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64)
    np.divide(matrix @ matrix.T, denominator, out=out, where=denominator > 0)
    return np.clip(out, 0.0, 1.0)


def medoid(values: NDArray) -> NDArray[np.uint8]:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("medoid requires at least one row")
    similarities = cosine_matrix(matrix)
    index = int(np.argmax(similarities.mean(axis=1)))
    return np.asarray(matrix[index], dtype=np.uint8).copy()


def _all_pairwise_above(values: NDArray, threshold: float) -> bool:
    matrix = cosine_matrix(values)
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return bool(upper.size == 0 or np.all(upper > threshold))


def _first_break(boundary_h: FloatArray, start: int, stop: int) -> int | None:
    locations = np.flatnonzero(boundary_h[start:stop] <= INHERITANCE_THRESHOLD)
    return None if locations.size == 0 else start + int(locations[0])


def find_earliest_episode(
    parents: NDArray,
    daughters: NDArray,
    boundary_h: NDArray,
    *,
    kind: str,
    burn_in: int = BURN_IN,
    window: int = WINDOW,
) -> Episode | None:
    """Return the earliest strict-B or F12-control episode in fixed windows.

    A strict episode uses the registered coherent-eight geometry.  An F12
    control uses only the first post-break run of three adjacent inherited
    boundaries.  Equality semantics match the paper: H=0.90 is a break and
    fails inheritance/coherence; H=0.85 passes old-anchor distinctness.
    """

    parents_i = np.asarray(parents)
    daughters_i = np.asarray(daughters)
    h = np.asarray(boundary_h, dtype=np.float64)
    if parents_i.shape != daughters_i.shape or parents_i.ndim != 2:
        raise ValueError("parent and daughter trajectories must have equal 2-D shape")
    if h.shape != (parents_i.shape[0],):
        raise ValueError("boundary-H length differs from trajectory")
    if kind not in {"strict", "f12"}:
        raise ValueError("kind must be 'strict' or 'f12'")
    run_length = STRICT_RUN if kind == "strict" else F12_RUN
    n_windows = max(0, (len(h) - burn_in) // window)
    for window_index in range(n_windows):
        start = burn_in + window_index * window
        stop = start + window
        first_break = _first_break(h, start, stop)
        if first_break is None:
            continue
        anchor = parents_i[first_break]
        for run_start in range(first_break + 1, stop - run_length + 1):
            run_stop = run_start + run_length
            if not bool(np.all(h[run_start:run_stop] > INHERITANCE_THRESHOLD)):
                continue
            episode_daughters = daughters_i[run_start:run_stop]
            if kind == "strict":
                if not _all_pairwise_above(
                    episode_daughters, INHERITANCE_THRESHOLD
                ):
                    continue
                if not all(
                    cosine(anchor, daughter) <= DISTINCTNESS_THRESHOLD
                    for daughter in episode_daughters
                ):
                    continue
            return Episode(
                kind=kind,
                window_index=window_index,
                break_index=first_break,
                run_start=run_start,
                daughters=np.asarray(episode_daughters, dtype=np.uint8).copy(),
                anchor=np.asarray(anchor, dtype=np.uint8).copy(),
            )
    return None


def split_centroid_similarity(episode: Episode) -> float:
    if episode.daughters.shape[0] < 2:
        raise ValueError("episode must have at least two daughters")
    split = episode.daughters.shape[0] // 2
    left = episode.daughters[:split].astype(np.float64).mean(axis=0)
    right = episode.daughters[split:].astype(np.float64).mean(axis=0)
    return cosine(left, right)


def split_centroids(episode: Episode) -> tuple[FloatArray, FloatArray]:
    split = episode.daughters.shape[0] // 2
    if split == 0:
        raise ValueError("episode must have at least two daughters")
    return (
        episode.daughters[:split].astype(np.float64).mean(axis=0),
        episode.daughters[split:].astype(np.float64).mean(axis=0),
    )


def sibling_stranger_values(
    episodes: Sequence[Episode],
) -> tuple[FloatArray, FloatArray]:
    """Return same-episode and all different-episode split-centroid H."""

    if len(episodes) < 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    halves = [split_centroids(item) for item in episodes]
    within = np.asarray([cosine(left, right) for left, right in halves])
    cross = np.asarray(
        [
            cosine(halves[i][0], halves[j][1])
            for i in range(len(halves))
            for j in range(len(halves))
            if i != j
        ],
        dtype=np.float64,
    )
    return within, cross


def probability_superiority(left: NDArray, right: NDArray) -> float:
    """P(left > right) with half credit for exact ties."""

    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    # Search-sorted computation avoids a potentially very large pair matrix.
    ordered = np.sort(b)
    below = np.searchsorted(ordered, a, side="left")
    through = np.searchsorted(ordered, a, side="right")
    ties = through - below
    return float(np.mean((below + 0.5 * ties) / ordered.size))


def empirical_range_overlap(left: NDArray, right: NDArray) -> tuple[bool, float]:
    """Literal finite-sample overlap and fraction of right inside left range."""

    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return False, float("nan")
    low, high = float(a.min()), float(a.max())
    inside = (b >= low) & (b <= high)
    return bool(np.any(inside)), float(inside.mean())


def select_capable_rules(
    event_counts: dict[str, dict[int, int]],
    *,
    count: int,
    selection_seed: str,
) -> list[int]:
    """Select shared event-capable rules by outcome-blind hash ordering."""

    if set(event_counts) != {"02", "03"}:
        raise ValueError("event counts must contain candidates 02 and 03")
    capable = sorted(
        matrix_id
        for matrix_id in set(event_counts["02"]) & set(event_counts["03"])
        if event_counts["02"][matrix_id] > 0
        and event_counts["03"][matrix_id] > 0
    )
    if len(capable) < count:
        raise ValueError(f"only {len(capable)} shared capable rules; need {count}")

    def key(matrix_id: int) -> tuple[str, int]:
        material = f"{selection_seed}|{matrix_id}".encode("utf-8")
        return hashlib.sha256(material).hexdigest(), matrix_id

    return sorted(capable, key=key)[:count]


def coherent_residences(
    daughters: NDArray,
    *,
    lineage: int,
    burn_in: int = BURN_IN,
    residence_length: int = PRIMARY_RESIDENCE,
    threshold: float = INHERITANCE_THRESHOLD,
) -> list[ResidenceEpisode]:
    """Find maximal runs of overlapping all-pair coherent windows."""

    values = np.asarray(daughters)
    if values.ndim != 2:
        raise ValueError("daughter trajectory must be two-dimensional")
    if residence_length < 2:
        raise ValueError("residence length must be at least two")
    starts: list[int] = []
    for start in range(burn_in, values.shape[0] - residence_length + 1):
        if _all_pairwise_above(values[start : start + residence_length], threshold):
            starts.append(start)
    if not starts:
        return []
    groups: list[list[int]] = [[starts[0]]]
    for start in starts[1:]:
        if start == groups[-1][-1] + 1:
            groups[-1].append(start)
        else:
            groups.append([start])
    output: list[ResidenceEpisode] = []
    for group in groups:
        first = group[0]
        last_state = group[-1] + residence_length - 1
        representative = medoid(values[first : first + residence_length])
        output.append(
            ResidenceEpisode(
                lineage=lineage,
                start=first,
                end=last_state,
                representative=representative,
            )
        )
    return output


def complete_link_clusters(values: NDArray, *, threshold: float) -> list[list[int]]:
    """Deterministic agglomeration where every merged pair exceeds threshold."""

    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError("cluster values must be two-dimensional")
    if matrix.shape[0] == 0:
        return []
    if matrix.shape[0] == 1:
        return [[0]]
    similarities = cosine_matrix(matrix)
    distances = np.clip(1.0 - similarities, 0.0, 1.0)
    np.fill_diagonal(distances, 0.0)
    tree = linkage(squareform(distances, checks=False), method="complete")
    # fcluster includes equality; move one floating-point step inward to retain
    # the registered strict H>threshold cluster boundary.
    cutoff = np.nextafter(1.0 - threshold, -np.inf)
    labels = fcluster(tree, t=cutoff, criterion="distance")
    clusters = [np.flatnonzero(labels == label).astype(int).tolist() for label in np.unique(labels)]
    return sorted(clusters, key=lambda item: item[0])


def _durable_lineages(
    cluster_indices: Sequence[int],
    episodes: Sequence[ResidenceEpisode],
    *,
    persistent_length: int = 16,
    departure_length: int = 8,
) -> tuple[int, ...]:
    by_lineage: dict[int, list[ResidenceEpisode]] = {}
    for index in cluster_indices:
        episode = episodes[index]
        by_lineage.setdefault(episode.lineage, []).append(episode)
    durable: list[int] = []
    for lineage, items in by_lineage.items():
        ordered = sorted(items, key=lambda item: item.start)
        persistent = any(item.duration >= persistent_length for item in ordered)
        reentry = any(
            right.start - left.end - 1 >= departure_length
            for left, right in zip(ordered, ordered[1:])
        )
        if persistent or reentry:
            durable.append(lineage)
    return tuple(sorted(durable))


def _distinct_greedy(forms: Sequence[StableForm], separation: float) -> tuple[StableForm, ...]:
    ordered = sorted(
        forms,
        key=lambda item: (
            -len(item.starts),
            -len(item.durable_starts),
            -len(item.episodes),
            item.cluster_id,
        ),
    )
    selected: list[StableForm] = []
    for form in ordered:
        if all(cosine(form.medoid, other.medoid) <= separation for other in selected):
            selected.append(form)
    return tuple(selected)


def attractor_census(
    trajectories: NDArray,
    *,
    burn_in: int = BURN_IN,
    residence_length: int = PRIMARY_RESIDENCE,
    start_support: int = PRIMARY_START_SUPPORT,
    durable_support: int = PRIMARY_DURABLE_SUPPORT,
    separation: float = PRIMARY_SEPARATION,
    coherence_threshold: float = INHERITANCE_THRESHOLD,
) -> CensusResult:
    """Count stable, mutually distinct forms across random-start lineages."""

    values = np.asarray(trajectories)
    if values.ndim != 3:
        raise ValueError("trajectories must have lineage, time, and type axes")
    residences, clusters = residence_clusters(
        values,
        burn_in=burn_in,
        residence_length=residence_length,
        coherence_threshold=coherence_threshold,
    )
    return census_from_clusters(
        residences,
        clusters,
        residence_length=residence_length,
        start_support=start_support,
        durable_support=durable_support,
        separation=separation,
    )


def residence_clusters(
    trajectories: NDArray,
    *,
    burn_in: int = BURN_IN,
    residence_length: int = PRIMARY_RESIDENCE,
    coherence_threshold: float = INHERITANCE_THRESHOLD,
) -> tuple[list[ResidenceEpisode], list[list[int]]]:
    """Build residence episodes and their complete-link clusters once."""

    values = np.asarray(trajectories)
    if values.ndim != 3:
        raise ValueError("trajectories must have lineage, time, and type axes")
    residences: list[ResidenceEpisode] = []
    for lineage in range(values.shape[0]):
        residences.extend(
            coherent_residences(
                values[lineage],
                lineage=lineage,
                burn_in=burn_in,
                residence_length=residence_length,
                threshold=coherence_threshold,
            )
        )
    if not residences:
        return [], []
    representatives = np.vstack([item.representative for item in residences])
    clusters = complete_link_clusters(representatives, threshold=coherence_threshold)
    return residences, clusters


def census_from_clusters(
    residences: Sequence[ResidenceEpisode],
    clusters: Sequence[Sequence[int]],
    *,
    residence_length: int,
    start_support: int,
    durable_support: int,
    separation: float,
) -> CensusResult:
    """Apply support and separation rules to a frozen residence clustering."""

    if not residences:
        return CensusResult(
            residence_length,
            start_support,
            durable_support,
            separation,
            0,
            0,
            (),
            (),
        )
    representatives = np.vstack([item.representative for item in residences])
    stable: list[StableForm] = []
    for cluster_id, indices in enumerate(clusters):
        starts = tuple(sorted({residences[index].lineage for index in indices}))
        durable = _durable_lineages(indices, residences)
        if len(starts) < start_support or len(durable) < durable_support:
            continue
        stable.append(
            StableForm(
                cluster_id=cluster_id,
                medoid=medoid(representatives[indices]),
                starts=starts,
                episodes=tuple(indices),
                durable_starts=durable,
            )
        )
    distinct = _distinct_greedy(stable, separation)
    return CensusResult(
        residence_length=residence_length,
        start_support=start_support,
        durable_support=durable_support,
        separation=separation,
        residence_episodes=len(residences),
        coherent_clusters=len(clusters),
        stable_forms=tuple(stable),
        distinct_forms=distinct,
    )


def fork_scores(
    starts: NDArray,
    fork_a: NDArray,
    fork_b: NDArray,
    *,
    distinctness: float = DISTINCTNESS_THRESHOLD,
) -> tuple[FloatArray, FloatArray, list[tuple[int, int]]]:
    """Return sibling minima and eligible stranger maxima.

    `fork_a` and `fork_b` have shape `(B, 8, types)`.  Stranger pairs are all
    ordered different-lineage pairs whose initial B states are at most 0.85.
    """

    initial = np.asarray(starts)
    a = np.asarray(fork_a)
    b = np.asarray(fork_b)
    if a.shape != b.shape or a.ndim != 3 or initial.shape != (a.shape[0], a.shape[2]):
        raise ValueError("fork arrays have incompatible shapes")
    sibling = np.asarray(
        [min(cosine(a[i, time], b[i, time]) for time in range(a.shape[1])) for i in range(a.shape[0])],
        dtype=np.float64,
    )
    pairs = [
        (i, j)
        for i in range(a.shape[0])
        for j in range(a.shape[0])
        if i != j and cosine(initial[i], initial[j]) <= distinctness
    ]
    stranger = np.asarray(
        [max(cosine(a[i, time], b[j, time]) for time in range(a.shape[1])) for i, j in pairs],
        dtype=np.float64,
    )
    return sibling, stranger, pairs


def bootstrap_mean_ci(
    values: NDArray,
    *,
    repetitions: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Whole-rule percentile interval for the mean of rule-level statistics."""

    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan"), float("nan")
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.integers(0, data.size, size=data.size)
        estimates[index] = float(data[sample].mean())
    alpha = (1.0 - confidence) / 2.0
    return (
        float(data.mean()),
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def nearest_identity_accuracy(episodes: Sequence[Episode]) -> float:
    """Classify each late-half centroid from all early-half centroids."""

    if len(episodes) < 2:
        return float("nan")
    halves = [split_centroids(item) for item in episodes]
    correct = 0
    for target, (_, late) in enumerate(halves):
        scores = [cosine(early, late) for early, _ in halves]
        predicted = int(np.argmax(np.asarray(scores)))
        correct += int(predicted == target)
    return correct / len(halves)


def strict_literal_fork_rate(sibling_scores: NDArray) -> float:
    values = np.asarray(sibling_scores, dtype=np.float64)
    return float(np.mean(values > INHERITANCE_THRESHOLD)) if values.size else float("nan")


def stranger_literal_distinct_rate(stranger_scores: NDArray) -> float:
    values = np.asarray(stranger_scores, dtype=np.float64)
    return float(np.mean(values <= INHERITANCE_THRESHOLD)) if values.size else 0.0


def sensitivity_grid() -> Iterable[tuple[int, int, int, float]]:
    for residence_length in (4, 8, 16):
        for start_support in (4, 8, 16):
            for separation in (0.80, 0.85, 0.90):
                yield residence_length, start_support, min(4, start_support), separation
