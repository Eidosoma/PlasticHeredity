"""Core contracts for the S13X adaptive directional replication search.

This module deliberately separates software-correct deterministic operations from
scientific evidential status. S13X is adaptive and outcome-guided: its results are
exploratory even when these functions replay exactly.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import rankdata
from scipy.stats import t as student_t
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from e01_frozen_timebase_ensemble.core import selected_clock_observations

VERSION = "E01-S13X-CREATIVE-DIRECTIONAL-REPLICATION-SEARCH-v1.0.0"
RESEARCH_STEP_ID = "S13X"
EVIDENCE_CLASS = "ADAPTIVE_OUTCOME_GUIDED_EXPLORATORY_RECONSTRUCTION"
ROOT_SEED_HEX = "ed70c5404d73015a6742dc4a37ca15f4388dc571fce71f9129088ae23bc27c23"

CANDIDATE_IDS = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
DEVELOPMENT_INDICES = tuple(range(60))
DIAGNOSTIC_INDICES = tuple(range(60, 100))

MetricName = Literal["synergy", "downwardCausation", "emergence", "localPhiR"]


@dataclass(frozen=True, slots=True)
class LabelSpec:
    """One explicit adaptive label definition."""

    label_id: str
    family: str
    threshold: float | None
    evidence_tier: str
    rationale: str


def derive_seed(*identity: object) -> int:
    """Derive a deterministic 32-bit seed in the isolated S13X domain."""

    material = "\x1f".join([VERSION, ROOT_SEED_HEX, *map(str, identity)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def label_specs() -> tuple[LabelSpec, ...]:
    """Return the complete first-stage label search registry.

    The threshold grid is intentionally explicit because S13X is adaptive. Values
    above 0.9 are Table-1-directed sensitivities, not recovered author settings.
    """

    output = [
        LabelSpec(
            "PF_HISTORICAL_ADJACENT_AVERAGE_H090",
            "POSTFISSION_HISTORICAL_ADJACENT_AVERAGE",
            0.9,
            "SOURCE_GROUNDED",
            "Historical tgs_nondrift technique 1 applied to post-fission states.",
        ),
        LabelSpec(
            "PF_EUCLIDEAN_KMEANS_DOMINANT",
            "POSTFISSION_EUCLIDEAN_KMEANS_DOMINANT",
            None,
            "PAPER_INFERRED",
            "Paper says Euclidean recurring composition; deterministic silhouette k=2..10.",
        ),
    ]
    for threshold in (0.80, 0.85, 0.90, 0.95):
        token = f"H{round(threshold * 1000):03d}"
        output.extend(
            [
                LabelSpec(
                    f"PF_DOMINANT_COMPONENT_CENTROID_{token}",
                    "POSTFISSION_DOMINANT_COMPONENT_CENTROID",
                    threshold,
                    "PAPER_INFERRED" if threshold == 0.90 else "SPECULATIVE_SENSITIVITY",
                    "Largest cosine-threshold connected component, then its centroid.",
                ),
                LabelSpec(
                    f"PF_MAX_NEIGHBOR_MEDOID_{token}",
                    "POSTFISSION_MAX_NEIGHBOR_MEDOID",
                    threshold,
                    "PAPER_INFERRED" if threshold == 0.90 else "SPECULATIVE_SENSITIVITY",
                    "Post-fission state with the most threshold neighbors as recurring medoid.",
                ),
            ]
        )
    for threshold in (0.90, 0.95, 0.965, 0.97, 0.975, 0.98):
        token = f"H{round(threshold * 1000):03d}"
        tier = "SOURCE_TRANSPLANT" if threshold == 0.90 else "TABLE1_DIRECTED_SPECULATIVE"
        output.extend(
            [
                LabelSpec(
                    f"MOL_ADJACENT_INCOMING_{token}",
                    "MOLECULAR_ADJACENT_INCOMING",
                    threshold,
                    tier,
                    "Incoming consecutive-state cosine on the paper's molecular clock.",
                ),
                LabelSpec(
                    f"MOL_ADJACENT_AVERAGE_{token}",
                    "MOLECULAR_ADJACENT_AVERAGE",
                    threshold,
                    tier,
                    "Historical incoming/outgoing average transplanted to molecular states.",
                ),
            ]
        )
    return tuple(output)


def _compositions(states: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 100 or np.any(values < 0):
        raise ValueError("states must be nonnegative observations-by-100 counts")
    masses = values.sum(axis=1)
    if np.any(masses <= 0):
        raise ValueError("label substrate contains an empty state")
    return values / masses[:, None]


def _cosine_matrix(values: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise ValueError("nonpositive or nonfinite composition norm")
    normalized = values / norms[:, None]
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def _cosine_to_reference(
    values: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    denominator = np.linalg.norm(values, axis=1) * np.linalg.norm(reference)
    if np.any(denominator <= 0):
        raise ValueError("nonpositive cosine denominator")
    return np.clip((values @ reference) / denominator, -1.0, 1.0)


def _connected_components(adjacency: NDArray[np.bool_]) -> list[tuple[int, ...]]:
    remaining = set(range(adjacency.shape[0]))
    components: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        found: set[int] = set()
        while stack:
            item = stack.pop()
            if item in found:
                continue
            found.add(item)
            stack.extend(
                int(value)
                for value in np.flatnonzero(adjacency[item])
                if int(value) not in found
            )
        remaining.difference_update(found)
        components.append(tuple(sorted(found)))
    return components


def _episode_count(labels: NDArray[np.bool_]) -> int:
    if labels.size == 0:
        return 0
    return int(np.count_nonzero(labels & ~np.concatenate(([False], labels[:-1]))))


def _fingerprint(labels: NDArray[np.bool_]) -> dict[str, float | int | None]:
    if labels.size == 0:
        return {
            "persistence": 0,
            "probability": None,
            "consistency": None,
            "timeToFirst": None,
            "episodeCount": 0,
        }
    indices = np.flatnonzero(labels)
    consistency: float | None = None
    if labels.size >= 3 and np.unique(labels).size == 2:
        value = float(
            np.corrcoef(labels[:-1].astype(float), labels[1:].astype(float))[0, 1]
        )
        consistency = value if np.isfinite(value) else None
    return {
        "persistence": int(indices.size),
        "probability": float(np.mean(labels)),
        "consistency": consistency,
        "timeToFirst": int(indices[0]) if indices.size else None,
        "episodeCount": _episode_count(labels),
    }


def label_trajectory(
    trajectory: Any,
    spec: LabelSpec,
    *,
    clock_id: str = "C1_SELECTED_DAUGHTER_RETAINED",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize one label on every selected-clock observation."""

    selected = selected_clock_observations(trajectory, clock_id)
    selected_states = np.asarray([item.state for item in selected], dtype=np.int64)
    selected_compositions = _compositions(selected_states)
    post = tuple(
        item for item in trajectory.observations if item.observation_kind == "post_fission"
    )
    if len(post) != int(trajectory.completed_fissions):
        raise ValueError("post-fission state cardinality mismatch")
    post_compositions = _compositions(np.asarray([item.state for item in post], dtype=np.int64))

    labels = np.zeros(len(selected), dtype=bool)
    scores = np.full(len(selected), np.nan, dtype=np.float64)
    reference_size: int | None = None
    selected_k: int | None = None
    silhouette: float | None = None

    if spec.family == "POSTFISSION_HISTORICAL_ADJACENT_AVERAGE":
        normalized = post_compositions / np.linalg.norm(post_compositions, axis=1)[:, None]
        adjacent = np.sum(normalized[:-1] * normalized[1:], axis=1)
        generation_scores = (
            np.concatenate(([adjacent[0]], adjacent))
            + np.concatenate((adjacent, [adjacent[-1]]))
        ) / 2.0
        generation_labels = generation_scores > float(spec.threshold)
        for index, item in enumerate(selected):
            generation = int(item.growth_generation_one_based)
            if 1 <= generation <= len(generation_labels):
                labels[index] = bool(generation_labels[generation - 1])
                scores[index] = float(generation_scores[generation - 1])
    elif spec.family in {
        "POSTFISSION_DOMINANT_COMPONENT_CENTROID",
        "POSTFISSION_MAX_NEIGHBOR_MEDOID",
    }:
        threshold = float(spec.threshold)
        similarity = _cosine_matrix(post_compositions)
        if spec.family == "POSTFISSION_DOMINANT_COMPONENT_CENTROID":
            adjacency = similarity >= threshold
            components = _connected_components(adjacency)
            dominant = min(components, key=lambda value: (-len(value), min(value)))
            reference = post_compositions[list(dominant)].mean(axis=0)
            reference /= reference.sum()
            reference_size = len(dominant)
        else:
            neighbor_counts = np.sum(similarity >= threshold, axis=1)
            reference_index = int(np.flatnonzero(neighbor_counts == neighbor_counts.max())[0])
            reference = post_compositions[reference_index]
            reference_size = int(neighbor_counts[reference_index])
        scores = _cosine_to_reference(selected_compositions, reference)
        labels = scores >= threshold
    elif spec.family in {
        "MOLECULAR_ADJACENT_INCOMING",
        "MOLECULAR_ADJACENT_AVERAGE",
    }:
        normalized = selected_compositions / np.linalg.norm(
            selected_compositions, axis=1
        )[:, None]
        adjacent = np.sum(normalized[:-1] * normalized[1:], axis=1)
        if spec.family == "MOLECULAR_ADJACENT_INCOMING":
            scores = np.concatenate(([adjacent[0]], adjacent))
        else:
            scores = (
                np.concatenate(([adjacent[0]], adjacent))
                + np.concatenate((adjacent, [adjacent[-1]]))
            ) / 2.0
        labels = scores > float(spec.threshold)
    elif spec.family == "POSTFISSION_EUCLIDEAN_KMEANS_DOMINANT":
        candidates: list[tuple[float, int, KMeans, NDArray[np.int64]]] = []
        for k in range(2, min(10, len(post_compositions) - 1) + 1):
            model = KMeans(
                n_clusters=k,
                n_init=10,
                algorithm="lloyd",
                random_state=derive_seed(
                    "label", trajectory.configuration_id, trajectory.matrix_index, spec.label_id, k
                ),
            )
            assigned = model.fit_predict(post_compositions).astype(np.int64, copy=False)
            if np.unique(assigned).size != k:
                continue
            value = float(silhouette_score(post_compositions, assigned))
            if np.isfinite(value):
                candidates.append((value, k, model, assigned))
        if not candidates:
            raise ValueError("no eligible Euclidean k-means solution")
        silhouette, selected_k, model, assigned = min(
            candidates, key=lambda value: (-value[0], value[1])
        )
        counts = np.bincount(assigned, minlength=selected_k)
        dominant_id = int(np.flatnonzero(counts == counts.max())[0])
        all_assigned = model.predict(selected_compositions)
        labels = all_assigned == dominant_id
        distances = np.linalg.norm(
            selected_compositions - model.cluster_centers_[dominant_id][None, :], axis=1
        )
        scores = -distances
        reference_size = int(counts[dominant_id])
    else:
        raise ValueError(f"unsupported label family: {spec.family}")

    rows = []
    for index, (item, label, score) in enumerate(
        zip(selected, labels, scores, strict=True)
    ):
        rows.append(
            {
                "candidateId": str(trajectory.configuration_id),
                "trajectoryId": str(trajectory.trajectory_id),
                "matrixIndex": int(trajectory.matrix_index),
                "labelId": spec.label_id,
                "labelFamily": spec.family,
                "labelEvidenceTier": spec.evidence_tier,
                "selectedSequenceIndex": index,
                "rawObservationIndex": int(item.observation_index),
                "generation": int(item.growth_generation_one_based),
                "observationKind": str(item.observation_kind),
                "isReplicator": bool(label),
                "labelScore": float(score) if np.isfinite(score) else None,
            }
        )
    fingerprint = {
        "candidateId": str(trajectory.configuration_id),
        "trajectoryId": str(trajectory.trajectory_id),
        "matrixIndex": int(trajectory.matrix_index),
        "labelId": spec.label_id,
        "labelFamily": spec.family,
        "labelEvidenceTier": spec.evidence_tier,
        "threshold": spec.threshold,
        "selectedObservationCount": len(selected),
        "referenceSize": reference_size,
        "selectedK": selected_k,
        "silhouette": silhouette,
        **_fingerprint(labels),
    }
    return pd.DataFrame(rows), fingerprint


TRANSFORMS = (
    "LEVEL",
    "BACKWARD_DIFFERENCE",
    "FORWARD_DIFFERENCE",
    "POSITIVE_BACKWARD_DIFFERENCE",
    "ABS_BACKWARD_DIFFERENCE",
    "TRAILING_MEAN_5",
    "TRAILING_MEAN_15",
    "GENERATION_MEAN",
    "GENERATION_MAX",
    "GENERATION_ENDPOINT",
)

ALIGNMENTS = (
    "SAME_STATE",
    "NEXT_STATE",
    "PREVIOUS_STATE",
    "NEXT_GENERATION",
    "PREVIOUS_GENERATION",
)


def transform_values(frame: pd.DataFrame, metric: MetricName, transform: str) -> pd.DataFrame:
    """Return one deterministic trajectory-level metric view."""

    if transform not in TRANSFORMS:
        raise ValueError(f"unknown transform {transform}")
    required = {"selectedSequenceIndex", "generation", metric}
    if not required.issubset(frame.columns):
        raise ValueError(f"metric frame lacks {sorted(required - set(frame.columns))}")
    ordered = frame.sort_values("selectedSequenceIndex", kind="stable").copy()
    values = pd.to_numeric(ordered[metric], errors="coerce").to_numpy(dtype=np.float64)
    if transform == "LEVEL":
        transformed = values
    elif transform == "BACKWARD_DIFFERENCE":
        transformed = np.concatenate(([np.nan], np.diff(values)))
    elif transform == "FORWARD_DIFFERENCE":
        transformed = np.concatenate((np.diff(values), [np.nan]))
    elif transform == "POSITIVE_BACKWARD_DIFFERENCE":
        transformed = np.concatenate(([np.nan], np.maximum(np.diff(values), 0.0)))
    elif transform == "ABS_BACKWARD_DIFFERENCE":
        transformed = np.concatenate(([np.nan], np.abs(np.diff(values))))
    elif transform in {"TRAILING_MEAN_5", "TRAILING_MEAN_15"}:
        window = 5 if transform.endswith("_5") else 15
        transformed = (
            pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy()
        )
    else:
        ordered["value"] = values
        grouped = ordered[ordered["generation"] > 0].groupby("generation", sort=True)
        rows: list[dict[str, Any]] = []
        for generation, subset in grouped:
            finite = subset[np.isfinite(subset["value"])]
            value = np.nan
            if not finite.empty:
                if transform == "GENERATION_MEAN":
                    value = float(finite["value"].mean())
                elif transform == "GENERATION_MAX":
                    value = float(finite["value"].max())
                else:
                    value = float(finite.iloc[-1]["value"])
            endpoint = subset.iloc[-1]
            rows.append(
                {
                    "selectedSequenceIndex": int(endpoint["selectedSequenceIndex"]),
                    "rawObservationIndex": int(endpoint["rawObservationIndex"]),
                    "generation": int(generation),
                    "value": value,
                    "timeIndex": int(generation),
                    "granularity": "GENERATION",
                }
            )
        return pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "selectedSequenceIndex": ordered["selectedSequenceIndex"].astype(int),
            "rawObservationIndex": ordered["rawObservationIndex"].astype(int),
            "generation": ordered["generation"].astype(int),
            "value": transformed,
            "timeIndex": ordered["selectedSequenceIndex"].astype(int),
            "granularity": "MOLECULAR",
        }
    )


def align_labels(
    transformed: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    alignment: str,
    transform: str,
) -> NDArray[np.float64]:
    """Align a label to one transformed value array with explicit boundary NaNs."""

    if alignment not in ALIGNMENTS:
        raise ValueError(f"unknown alignment {alignment}")
    label_by_index = dict(
        zip(
            labels["selectedSequenceIndex"].astype(int),
            labels["isReplicator"].astype(bool),
            strict=True,
        )
    )
    generation_groups = labels[labels["generation"] > 0].groupby("generation", sort=True)
    generation_majority = {
        int(generation): bool(group["isReplicator"].astype(bool).mean() >= 0.5)
        for generation, group in generation_groups
    }
    generation_endpoint = {
        int(generation): bool(
            group.sort_values("selectedSequenceIndex", kind="stable").iloc[-1][
                "isReplicator"
            ]
        )
        for generation, group in generation_groups
    }
    output = np.full(len(transformed), np.nan, dtype=np.float64)
    generation_level = bool((transformed["granularity"] == "GENERATION").all())
    for position, row in enumerate(transformed.itertuples(index=False)):
        generation = int(row.generation)
        index = int(row.selectedSequenceIndex)
        if alignment == "SAME_STATE":
            if generation_level:
                mapping = (
                    generation_endpoint
                    if transform == "GENERATION_ENDPOINT"
                    else generation_majority
                )
                value = mapping.get(generation)
            else:
                value = label_by_index.get(index)
        elif alignment == "NEXT_STATE":
            if generation_level:
                continue
            value = label_by_index.get(index + 1)
        elif alignment == "PREVIOUS_STATE":
            if generation_level:
                continue
            value = label_by_index.get(index - 1)
        elif alignment == "NEXT_GENERATION":
            value = generation_majority.get(generation + 1)
        else:
            value = generation_majority.get(generation - 1)
        if value is not None:
            output[position] = float(value)
    return output


def binary_spearman(
    values: NDArray[np.float64], labels: NDArray[np.float64]
) -> tuple[float | None, float | None]:
    """Compute Spearman correlation against a binary label without warnings."""

    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None, None
    ranked = rankdata(x, method="average")
    rho = float(np.corrcoef(ranked, y)[0, 1])
    if not np.isfinite(rho):
        return None, None
    denominator = max(1.0 - rho * rho, np.finfo(np.float64).tiny)
    statistic = rho * math.sqrt((x.size - 2) / denominator)
    p = float(2.0 * student_t.sf(abs(statistic), df=x.size - 2))
    return rho, p


def association_summary(
    values: NDArray[np.float64], labels: NDArray[np.float64]
) -> dict[str, Any]:
    """Return one trajectory's directional association and drift contrast."""

    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask].astype(bool)
    rho, p = binary_spearman(x, y.astype(float))
    positive, negative = x[y], x[~y]
    mean_difference = (
        float(np.mean(positive) - np.mean(negative))
        if positive.size and negative.size
        else None
    )
    median_difference = (
        float(np.median(positive) - np.median(negative))
        if positive.size and negative.size
        else None
    )
    scale = float(np.std(x)) if x.size else np.nan
    standardized = (
        float(mean_difference / scale)
        if mean_difference is not None and np.isfinite(scale) and scale > 0
        else None
    )
    return {
        "n": int(x.size),
        "labelProbability": float(np.mean(y)) if y.size else None,
        "rho": rho,
        "ordinaryTwoSidedP": p,
        "meanDifference": mean_difference,
        "medianDifference": median_difference,
        "standardizedMeanDifference": standardized,
    }


def _similarity(value: float | None, target: float, width: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(math.exp(-abs(float(value) - target) / width))


def _direction(value: float | None, scale: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(0.5 + 0.5 * math.tanh(float(value) / scale))


def resemblance_score(row: dict[str, Any]) -> dict[str, float]:
    """Score continuous paper-direction resemblance; this is not a gate."""

    association = (
        0.45 * _similarity(row.get("positiveCorrelationFraction"), 0.73, 0.25)
        + 0.25 * _similarity(row.get("positiveSignificantFraction"), 0.54, 0.25)
        + 0.30 * _direction(row.get("medianCorrelation"), 0.05)
    )
    drift = (
        0.65 * _similarity(row.get("higherDuringReplicationFraction"), 0.57, 0.25)
        + 0.35 * _direction(row.get("medianStandardizedMeanDifference"), 0.10)
    )
    spikes = (
        0.60 * _similarity(row.get("positiveThreeSigmaRunFraction"), 0.80, 0.25)
        + 0.40 * _similarity(row.get("robustSpikeRunFraction"), 0.80, 0.25)
    )
    trend_p = row.get("aggregateTrendP")
    weak_trend = 0.0
    if trend_p is not None and np.isfinite(trend_p):
        weak_trend = float(min(1.0, max(0.0, float(trend_p) / 0.1995)))
    temporal = (
        0.5 * _similarity(row.get("rawLjungBoxFraction"), 0.86, 0.25)
        + 0.5 * _similarity(row.get("differencedLjungBoxFraction"), 1.0, 0.20)
    )
    label = (
        0.35 * _similarity(row.get("labelProbabilityMean"), 0.88, 0.20)
        + 0.30 * _similarity(row.get("labelPersistenceMean"), 716.0, 300.0)
        + 0.20 * _similarity(row.get("labelConsistencyMean"), 0.38, 0.20)
        + 0.15 * _similarity(row.get("labelTimeToFirstMean"), 37.0, 50.0)
    )
    total = (
        0.30 * association
        + 0.20 * drift
        + 0.20 * spikes
        + 0.10 * weak_trend
        + 0.10 * temporal
        + 0.10 * label
    )
    return {
        "associationResemblance": association,
        "driftResemblance": drift,
        "spikeResemblance": spikes,
        "weakTrendResemblance": weak_trend,
        "temporalResemblance": temporal,
        "labelFingerprintResemblance": label,
        "directionalResemblanceScore": total,
    }
