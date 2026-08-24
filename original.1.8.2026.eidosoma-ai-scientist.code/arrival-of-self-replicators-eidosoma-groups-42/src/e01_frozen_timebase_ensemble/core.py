"""Frozen input, clock, label, preprocessing, and seed contracts for S12G.

This module never simulates GARD.  It only exposes deterministic views of the
96 immutable S12FR confirmation trajectories and delegates labels/source
metrics to already validated E01 implementations.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from e01_replicator_labels import (
    ClusterConfiguration,
    cluster_labels,
    historical_technique1_labels,
)

VERSION = "E01-S12G-FROZEN-TIMEBASE-ENSEMBLE-v1.0.0"
RESEARCH_STEP_ID = "S12G"
EVIDENCE_CLASS = "SOURCE_INFORMED_FROZEN_TIMEBASE_ENSEMBLE_RECONSTRUCTION"
ANALYSIS_ROOT_SEED_HEX = (
    "9aa1b76069d0d264ed738ba50a5c04eccbba5ff0547ac93ceb29624caf11dc53"
)
CANDIDATE_IDS = (
    "S12F-CANDIDATE-01",
    "S12F-CANDIDATE-02",
    "S12F-CANDIDATE-03",
)
HISTORICAL_LABEL_ID = "HISTORICAL_H090_REPLICATOR"
ONLINE_LABEL_ID = "PAST_ONLY_COSINE_REPLICATOR"
HISTORICAL_CONFIGURATION_ID = "E01-S08-YH-T1-HGT090-v1.0.0"
ONLINE_CONFIGURATION_ID = "E01-S08-YC-COS-HGT090-MIN3-ONLINE-v1.0.0"
PREPROCESSING_ID = "E01-S12G-PREPROC-ADD0p5-FULLCLR-DROP100-v1.0.0"
ELIGIBLE_SOURCE_STATUSES = {"ELIGIBLE", "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES"}


@dataclass(frozen=True, slots=True)
class EndpointRecord:
    """Frozen mapping from one fission decision to a locked-clock endpoint."""

    generation: int
    selected_sequence_index: int
    raw_observation_index: int
    observation_kind: str
    prior_locked_clock_transitions: int


def derive_seed(*identity: object) -> int:
    """Return a deterministic 32-bit legacy NumPy seed in the S12G domain."""

    material = "\x1f".join(
        ["E01-S12G-DOMAIN-SEPARATION-v1", ANALYSIS_ROOT_SEED_HEX, *map(str, identity)]
    )
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def sha256_array(array: NDArray[Any]) -> str:
    """Hash shape, dtype, and canonical C-order bytes."""

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\x00")
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(b"\x00")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def selected_clock_observations(
    trajectory: Any, clock_id: str
) -> tuple[Any, ...]:
    """Materialize exactly the C0 or C1 state sequence frozen by S12FR."""

    observations = tuple(trajectory.observations)
    if not observations or observations[0].observation_kind != "initial_selected_state":
        raise ValueError("trajectory must begin with initial_selected_state")
    if clock_id == "C0_BATCH_UPDATES_ONLY":
        selected = tuple(
            item
            for item in observations
            if item.observation_kind in {"initial_selected_state", "molecular_update"}
        )
    elif clock_id == "C1_SELECTED_DAUGHTER_RETAINED":
        selected = tuple(
            item
            for item in observations
            if item.observation_kind
            in {"initial_selected_state", "molecular_update", "post_fission"}
        )
    else:
        raise ValueError(f"unsupported locked clock: {clock_id}")
    raw_indices = [int(item.observation_index) for item in selected]
    if raw_indices != sorted(raw_indices) or len(raw_indices) != len(set(raw_indices)):
        raise ValueError("selected observations are not a strict raw-index subsequence")
    expected = 1 + int(trajectory.total_batch_updates)
    if clock_id == "C1_SELECTED_DAUGHTER_RETAINED":
        expected += int(trajectory.completed_fissions)
    if len(selected) != expected:
        raise ValueError(f"clock cardinality mismatch: {len(selected)} != {expected}")
    return selected


def post_fission_endpoint_records(
    trajectory: Any, clock_id: str, *, minimum_prior_transitions: int = 256
) -> tuple[EndpointRecord, ...]:
    """Map every generation boundary into the selected clock without invention.

    C0 intentionally maps to the final molecular update before fission because
    its locked observation clock excludes selected-daughter states.  C1 maps to
    the selected post-fission daughter state.
    """

    selected = selected_clock_observations(trajectory, clock_id)
    sequence_index = {
        int(item.observation_index): index for index, item in enumerate(selected)
    }
    records: list[EndpointRecord] = []
    for generation in range(1, int(trajectory.completed_fissions) + 1):
        if clock_id == "C1_SELECTED_DAUGHTER_RETAINED":
            matches = [
                item
                for item in trajectory.observations
                if item.observation_kind == "post_fission"
                and int(item.growth_generation_one_based) == generation
            ]
        else:
            matches = [
                item
                for item in trajectory.observations
                if item.observation_kind == "molecular_update"
                and int(item.growth_generation_one_based) == generation
            ]
            if matches:
                matches = [max(matches, key=lambda item: int(item.observation_index))]
        if len(matches) != 1:
            raise ValueError(
                f"generation {generation} has {len(matches)} endpoint observations"
            )
        item = matches[0]
        index = sequence_index[int(item.observation_index)]
        if index >= minimum_prior_transitions:
            records.append(
                EndpointRecord(
                    generation=generation,
                    selected_sequence_index=index,
                    raw_observation_index=int(item.observation_index),
                    observation_kind=str(item.observation_kind),
                    prior_locked_clock_transitions=index,
                )
            )
    return tuple(records)


def states_from_observations(observations: Sequence[Any]) -> NDArray[np.int64]:
    """Return a validated observations-by-100 integer count matrix."""

    states = np.asarray([item.state for item in observations], dtype=np.int64)
    if states.ndim != 2 or states.shape[1] != 100 or states.shape[0] == 0:
        raise ValueError("S12G requires a nonempty observations-by-100 state matrix")
    if np.any(states < 0):
        raise ValueError("GARD count states must be nonnegative")
    return states


def frozen_clr(
    states: NDArray[np.integer[Any]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Apply additive-0.5 closure, full CLR, then remove original component 100."""

    counts = np.asarray(states, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != 100 or np.any(counts < 0):
        raise ValueError("states must be nonnegative observations-by-100 counts")
    masses = np.sum(counts, axis=1)
    closed = (counts + 0.5) / (masses[:, None] + 50.0)
    closure_errors = np.abs(np.sum(closed, axis=1) - 1.0)
    logs = np.log(closed)
    full_clr = logs - np.mean(logs, axis=1, keepdims=True)
    dropped = np.ascontiguousarray(full_clr[:, :99], dtype=np.float64)
    if not np.all(np.isfinite(dropped)):
        raise ValueError("frozen CLR produced a nonfinite coordinate")
    return dropped, masses, closure_errors


def _online_configuration() -> ClusterConfiguration:
    return ClusterConfiguration(
        configuration_id=ONLINE_CONFIGURATION_ID,
        family_id="Y_C",
        family_name="past_only_cosine_threshold_graph",
        evidence_class="VALIDATION_ONLY_PAST_ONLY_CAUSAL_COMPANION",
        metric="cosine",
        representation="raw_nonnegative_count_composition",
        threshold=0.9,
        comparator="strict_greater_than",
        minimum_cluster_size=3,
        temporal_scope="past_only_online",
        zero_policy="zero_sum_ineligible_no_deletion",
    )


def frozen_generation_labels(
    trajectory: Any,
) -> tuple[list[dict[str, Any]], dict[int, bool | None], dict[int, bool | None]]:
    """Evaluate exactly the two frozen post-fission label families."""

    post = tuple(
        item for item in trajectory.observations if item.observation_kind == "post_fission"
    )
    if len(post) != int(trajectory.completed_fissions):
        raise ValueError("post-fission label substrate cardinality mismatch")
    states = states_from_observations(post)
    ids = tuple(
        f"{trajectory.trajectory_id}::generation-{int(item.growth_generation_one_based):03d}"
        for item in post
    )
    historical = historical_technique1_labels(
        states,
        trajectory_id=str(trajectory.trajectory_id),
        observation_ids=ids,
        configuration_id=HISTORICAL_CONFIGURATION_ID,
        threshold=0.9,
        evidence_class="SOURCE_TRACEABLE_HISTORICAL_RECONSTRUCTION",
    )
    online = cluster_labels(
        states,
        trajectory_id=str(trajectory.trajectory_id),
        observation_ids=ids,
        configuration=_online_configuration(),
    )
    rows: list[dict[str, Any]] = []
    historical_map: dict[int, bool | None] = {}
    online_map: dict[int, bool | None] = {}
    for label_id, result, destination in (
        (HISTORICAL_LABEL_ID, historical, historical_map),
        (ONLINE_LABEL_ID, online, online_map),
    ):
        if len(result.rows) != len(post):
            raise ValueError(f"{label_id} row count mismatch")
        for item, label in zip(post, result.rows, strict=True):
            generation = int(item.growth_generation_one_based)
            destination[generation] = label.is_replicator
            rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "candidateId": str(trajectory.configuration_id),
                    "trajectoryId": str(trajectory.trajectory_id),
                    "matrixIndex": int(trajectory.matrix_index),
                    "generation": generation,
                    "postFissionRawObservationIndex": int(item.observation_index),
                    "labelId": label_id,
                    "labelStatus": label.label_status,
                    "isReplicator": label.is_replicator,
                    "historicalIncomingH": label.historical_incoming_h,
                    "historicalLocalScore": label.historical_local_score,
                    "clusterId": label.cluster_id,
                    "referenceObservationId": label.reference_observation_id,
                    "metricToReference": label.metric_to_reference,
                    "ineligibilityReason": label.ineligibility_reason,
                }
            )
    return rows, historical_map, online_map


def output_value_at_sequence_index(
    values: NDArray[np.float64] | None,
    *,
    local_offset: int,
    selected_sequence_index: int,
) -> float | None:
    """Map a source-local array to its raw selected-sequence endpoint."""

    if values is None:
        return None
    local_index = int(selected_sequence_index) - int(local_offset)
    if local_index < 0 or local_index >= len(values):
        return None
    value = float(values[local_index])
    return value if np.isfinite(value) else None


def endpoint_value(values: NDArray[np.float64] | None) -> float | None:
    """Return the final finite local value from a prefix, else explicit null."""

    if values is None or len(values) == 0:
        return None
    value = float(values[-1])
    return value if np.isfinite(value) else None
