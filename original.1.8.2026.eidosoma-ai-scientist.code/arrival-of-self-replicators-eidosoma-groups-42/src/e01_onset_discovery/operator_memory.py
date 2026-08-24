"""Fixed low-dimensional exact-generator memory features for S19-L33."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from e01_latent_timebase.core import (
    N_MAX,
    SimulationDefinition,
    exposure_for_rates,
    rates,
)
from e01_onset_discovery.generator_coordinate import (
    analytic_count_moments,
    brownian_hitting_probability,
    composition_linearized_moments,
    cosine_gradient,
    relative_composition,
)

HISTORY = 8
TARGET_THRESHOLD = 0.9

PHASE_CHANNELS = (
    "mass_fraction",
    "generation_local_step_fraction",
    "post_fission_indicator",
    "growth_generation_fraction",
    "batch_step_fraction",
)
ORDINARY_CHANNELS = (
    "composition_diversity",
    "composition_entropy",
    "composition_concentration",
)
BASIN_BLIND_OPERATOR_CHANNELS = (
    "reaction_activity_per_mass",
    "boost_mean",
    "boost_sd",
    "boost_maximum",
    "composition_drift_norm",
    "composition_diffusion_trace",
    "current_drift_alignment",
)
TARGET_CHANNELS = (
    "target_score",
    "target_support_overlap",
    "target_score_drift",
    "target_score_variance",
    "target_direction_drift",
    "target_direction_diffusion",
    "brownian_hit_h32",
)
ALL_CHANNELS = (
    PHASE_CHANNELS
    + ORDINARY_CHANNELS
    + BASIN_BLIND_OPERATOR_CHANNELS
    + TARGET_CHANNELS
)
VIEWS = (
    "PHASE_MEMORY",
    "BASIN_BLIND_OPERATOR_MEMORY",
    "TARGET_CONDITIONED_OPERATOR_MEMORY",
)


def _entropy(composition: NDArray[np.float64]) -> float:
    positive = composition[composition > 0]
    return float(-np.sum(positive * np.log(positive)) / math.log(len(composition)))


def _cosine(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        raise ValueError("cosine requires nonzero vectors")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def channel_sequence(
    states: NDArray[np.integer],
    beta: NDArray[np.floating],
    target: NDArray[np.floating],
    definition: SimulationDefinition,
    *,
    observation_kinds: Sequence[str],
    generation_local_steps: Sequence[int],
    growth_generations: Sequence[int],
    batch_steps: Sequence[int],
) -> NDArray[np.float64]:
    """Return the frozen eight-by-22 past-only state/operator channel array."""

    counts = np.asarray(states, dtype=np.int64)
    matrix = np.asarray(beta, dtype=np.float64)
    centroid = np.asarray(target, dtype=np.float64)
    if counts.shape != (HISTORY, 100) or matrix.shape != (100, 100):
        raise ValueError("operator-memory input shape changed")
    if centroid.shape != (100,) or np.any(centroid < 0) or centroid.sum() <= 0:
        raise ValueError("invalid target centroid")
    centroid = centroid / centroid.sum()
    metadata = (
        observation_kinds,
        generation_local_steps,
        growth_generations,
        batch_steps,
    )
    if any(len(values) != HISTORY for values in metadata):
        raise ValueError("operator-memory metadata length changed")

    rows = []
    for index, state in enumerate(counts):
        mass = int(state.sum())
        if mass <= 0:
            raise ValueError("empty state in operator memory")
        kind = str(observation_kinds[index])
        local_step = (
            0
            if kind in {"initial_selected_state", "post_fission"}
            else int(generation_local_steps[index])
        )
        composition = relative_composition(state)
        score = _cosine(composition, centroid)
        moments = analytic_count_moments(
            state,
            matrix,
            definition,
            generation_local_step=local_step,
        )
        mean_delta, covariance_delta = composition_linearized_moments(
            state, moments
        )
        gradient = cosine_gradient(composition, centroid)
        score_drift = float(np.dot(gradient, mean_delta))
        score_variance = max(
            0.0, float(gradient @ covariance_delta @ gradient)
        )
        direction = centroid - composition
        direction_norm = float(np.linalg.norm(direction))
        unit_direction = (
            direction / direction_norm if direction_norm > 0 else direction
        )
        direction_drift = float(np.dot(unit_direction, mean_delta))
        direction_diffusion = max(
            0.0, float(unit_direction @ covariance_delta @ unit_direction)
        )
        drift_norm = float(np.linalg.norm(mean_delta))
        diffusion_trace = float(np.trace(covariance_delta))
        composition_norm = float(np.linalg.norm(composition))
        # The first-order compositional drift is analytically zero in some
        # symmetric states.  Float64 cancellation leaves permutation-order
        # residues around 1e-17, for which a normalized direction is
        # undefined rather than scientifically meaningful.
        current_alignment = (
            float(np.dot(composition, mean_delta) / (composition_norm * drift_norm))
            if drift_norm > 1e-14
            else 0.0
        )
        joins, losses = rates(state, matrix)
        exposure = exposure_for_rates(definition.exposure, joins, losses)
        boost = 1.0 + (matrix @ state.astype(np.float64)) / mass
        rows.append(
            [
                mass / N_MAX,
                local_step / 1000.0,
                float(kind == "post_fission"),
                int(growth_generations[index]) / 100.0,
                int(batch_steps[index]) / 192.0,
                float(np.count_nonzero(state) / 100.0),
                _entropy(composition),
                float(np.sum(composition * composition)),
                float(exposure * (joins.sum() + losses.sum()) / mass),
                float(np.mean(boost)),
                float(np.std(boost, ddof=0)),
                float(np.max(boost)),
                drift_norm,
                diffusion_trace,
                current_alignment,
                score,
                float(np.mean((state > 0) & (centroid > 0))),
                score_drift,
                score_variance,
                direction_drift,
                direction_diffusion,
                brownian_hitting_probability(
                    max(0.0, TARGET_THRESHOLD - score),
                    score_drift,
                    score_variance,
                    32,
                ),
            ]
        )
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (HISTORY, len(ALL_CHANNELS)) or not np.isfinite(result).all():
        raise RuntimeError("operator-memory channel schema or finiteness failure")
    return result


def _summary(
    channels: NDArray[np.float64], indices: Sequence[int]
) -> NDArray[np.float64]:
    selected = channels[:, list(indices)]
    time = np.linspace(-1.0, 0.0, HISTORY, dtype=np.float64)
    centered = time - np.mean(time)
    slopes = centered @ selected / float(centered @ centered)
    endpoint = selected[-1]
    return np.concatenate([endpoint, slopes]).astype(np.float64, copy=False)


def feature_names() -> dict[str, tuple[str, ...]]:
    phase = tuple(
        [f"endpoint__{name}" for name in PHASE_CHANNELS]
        + [f"slope__{name}" for name in PHASE_CHANNELS]
        + [f"mean__{name}" for name in PHASE_CHANNELS]
    )
    basin_blind_channels = (
        PHASE_CHANNELS + ORDINARY_CHANNELS + BASIN_BLIND_OPERATOR_CHANNELS
    )
    basin_blind = tuple(
        [f"endpoint__{name}" for name in basin_blind_channels]
        + [f"slope__{name}" for name in basin_blind_channels]
        + [f"mean__{name}" for name in PHASE_CHANNELS]
    )
    target_conditioned = tuple(
        [f"endpoint__{name}" for name in ALL_CHANNELS]
        + [f"slope__{name}" for name in ALL_CHANNELS]
        + [f"mean__{name}" for name in PHASE_CHANNELS]
        + ["target_component_fraction", "target_entropy"]
    )
    return {
        "PHASE_MEMORY": phase,
        "BASIN_BLIND_OPERATOR_MEMORY": basin_blind,
        "TARGET_CONDITIONED_OPERATOR_MEMORY": target_conditioned,
    }


def operator_memory_views(
    states: NDArray[np.integer],
    beta: NDArray[np.floating],
    target: NDArray[np.floating],
    definition: SimulationDefinition,
    *,
    observation_kinds: Sequence[str],
    generation_local_steps: Sequence[int],
    growth_generations: Sequence[int],
    batch_steps: Sequence[int],
    target_component_fraction: float,
) -> dict[str, NDArray[np.float64]]:
    """Calculate exactly three nested endpoint/slope operator-memory views."""

    centroid = np.asarray(target, dtype=np.float64)
    centroid = centroid / centroid.sum()
    channels = channel_sequence(
        states,
        beta,
        centroid,
        definition,
        observation_kinds=observation_kinds,
        generation_local_steps=generation_local_steps,
        growth_generations=growth_generations,
        batch_steps=batch_steps,
    )
    phase_indices = range(len(PHASE_CHANNELS))
    basin_blind_indices = range(
        len(PHASE_CHANNELS)
        + len(ORDINARY_CHANNELS)
        + len(BASIN_BLIND_OPERATOR_CHANNELS)
    )
    all_indices = range(len(ALL_CHANNELS))
    phase = np.concatenate(
        [
            _summary(channels, phase_indices),
            np.mean(channels[:, : len(PHASE_CHANNELS)], axis=0),
        ]
    )
    constants = np.asarray(
        [float(target_component_fraction), _entropy(centroid)], dtype=np.float64
    )
    phase_means = np.mean(channels[:, : len(PHASE_CHANNELS)], axis=0)
    result = {
        "PHASE_MEMORY": phase,
        "BASIN_BLIND_OPERATOR_MEMORY": np.concatenate(
            [_summary(channels, basin_blind_indices), phase_means]
        ),
        "TARGET_CONDITIONED_OPERATOR_MEMORY": np.concatenate(
            [_summary(channels, all_indices), phase_means, constants]
        ),
    }
    names = feature_names()
    if tuple(result) != VIEWS or any(
        values.shape != (len(names[key]),) or not np.isfinite(values).all()
        for key, values in result.items()
    ):
        raise RuntimeError("operator-memory view schema changed")
    return result
