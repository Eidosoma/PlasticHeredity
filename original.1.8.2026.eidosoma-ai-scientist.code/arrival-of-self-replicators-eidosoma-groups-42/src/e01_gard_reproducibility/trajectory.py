"""Lossless trajectory capture and exact same-engine regeneration for E01 S06."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, fields
from enum import StrEnum
from math import isclose
from typing import Any

import numpy as np

from e01_gard_independent import (
    EventLog,
    FissionLog,
    GardSpecification,
    GenerationResult,
    LineageResult,
    generate_catalytic_matrix,
    generator_state_sha256,
    initialize_state,
    simulate_lineage,
    specification_from_mapping,
)

from .seed import (
    CANONICAL_STREAM_PURPOSES,
    SeedBundle,
    StreamPurpose,
    canonical_json_bytes,
    derive_seed_bundle,
    seed_request_from_payload,
    sha256_hex,
)
from .serialization import SerializationError, float_from_hex, float_to_hex

TRAJECTORY_SCHEMA_VERSION = "E01-trajectory-schema-v1.0.0"
SPECIFICATION_PAYLOAD_VERSION = "E01-gard-specification-payload-v1.0.0"
PRECISION_CONTRACT_VERSION = "E01-trajectory-precision-v1.0.0"
MAX_SIGNED_INT64 = (1 << 63) - 1


class TrajectoryContractError(ValueError):
    """A captured trajectory violates the frozen S06 schema or invariants."""


@dataclass(frozen=True, slots=True)
class CaptureIdentity:
    """Explicit engine, source, runtime, schema, and precision identity."""

    engine_id: str
    engine_package: str
    engine_version: str
    repository_commit: str
    engine_source_sha256: str
    adapter_source_sha256: str
    python_version: str
    numpy_version: str
    platform: str
    byte_order: str
    runtime_fingerprint: str
    seed_schema_sha256: str
    trajectory_schema_sha256: str
    precision_contract_sha256: str
    numeric_thread_environment: tuple[tuple[str, str], ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "engineId": self.engine_id,
            "enginePackage": self.engine_package,
            "engineVersion": self.engine_version,
            "repositoryCommit": self.repository_commit,
            "engineSourceSha256": self.engine_source_sha256,
            "adapterSourceSha256": self.adapter_source_sha256,
            "pythonVersion": self.python_version,
            "numpyVersion": self.numpy_version,
            "platform": self.platform,
            "byteOrder": self.byte_order,
            "runtimeFingerprint": self.runtime_fingerprint,
            "seedSchemaSha256": self.seed_schema_sha256,
            "trajectorySchemaSha256": self.trajectory_schema_sha256,
            "precisionContractSha256": self.precision_contract_sha256,
            "numericThreadEnvironment": dict(self.numeric_thread_environment),
            "sameEngineRegenerationScope": (
                "Exact only for this engine source, adapter source, NumPy version, "
                "bit generator, platform/runtime identity, and precision/thread contract."
            ),
        }


@dataclass(frozen=True, slots=True)
class RegistryBoundary:
    """Immutable ambiguity-registry boundary copied into every trajectory."""

    registry_version: str
    registry_sha256: str
    registry_executable: bool
    no_silent_defaults: bool
    parameter_count: int
    unresolved_parameter_count: int
    unexpanded_branch_set_count: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "registryVersion": self.registry_version,
            "registrySha256": self.registry_sha256,
            "registryExecutable": self.registry_executable,
            "noSilentDefaults": self.no_silent_defaults,
            "parameterCount": self.parameter_count,
            "unresolvedParameterCount": self.unresolved_parameter_count,
            "unexpandedBranchSetCount": self.unexpanded_branch_set_count,
            "authorInitialStateRngStream": "UNRESOLVED::E01-A020",
            "analysisStateSamplingInstant": "UNRESOLVED::E01-A025",
            "authorCodeIdentity": "UNAVAILABLE::NO_AUTHOR_CODE_RELEASE_FOUND",
            "legacyMatlabRngIdentity": (
                "UNRESOLVED::LEGACY_MATLAB_RNG_ALGORITHM_AND_GLOBAL_STATE_ORDER"
            ),
            "rule": (
                "This fixture instantiates explicit reconstruction branches only; it "
                "does not update, execute, or resolve the closed author registry."
            ),
        }


def _hex_sequence(values: tuple[float, ...] | list[float] | np.ndarray) -> list[str]:
    return [float_to_hex(float(value)) for value in values]


def _specification_fields(specification: GardSpecification) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields(specification):
        value = getattr(specification, field.name)
        if isinstance(value, StrEnum):
            payload[field.name] = value.value
        elif field.name == "rho":
            payload[field.name] = _hex_sequence(value)
        elif isinstance(value, float):
            payload[field.name] = float_to_hex(value)
        elif value is None or isinstance(value, (str, int)):
            payload[field.name] = value
        else:
            raise TrajectoryContractError(
                f"Unsupported specification value for {field.name}: {type(value).__name__}."
            )
    return payload


def specification_to_payload(specification: GardSpecification) -> dict[str, Any]:
    """Encode a complete S05 specification without decimal-float ambiguity."""

    values = _specification_fields(specification)
    return {
        "specificationPayloadVersion": SPECIFICATION_PAYLOAD_VERSION,
        "specificationId": specification.specification_id,
        "fieldsSha256": sha256_hex(canonical_json_bytes(values)),
        "fields": values,
    }


def specification_from_payload(payload: dict[str, Any]) -> GardSpecification:
    """Decode and checksum-verify a complete S05 specification."""

    required = {
        "specificationPayloadVersion",
        "specificationId",
        "fieldsSha256",
        "fields",
    }
    if set(payload) != required:
        raise TrajectoryContractError("Specification payload fields are incomplete.")
    if payload["specificationPayloadVersion"] != SPECIFICATION_PAYLOAD_VERSION:
        raise TrajectoryContractError("Unsupported specification payload version.")
    values = payload["fields"]
    if not isinstance(values, dict):
        raise TrajectoryContractError("Specification fields must be an object.")
    if sha256_hex(canonical_json_bytes(values)) != payload["fieldsSha256"]:
        raise TrajectoryContractError("Specification field checksum mismatch.")
    if values.get("specification_id") != payload["specificationId"]:
        raise TrajectoryContractError("Specification identity mismatch.")
    decoded = dict(values)
    decoded["rho"] = [float_from_hex(value) for value in decoded["rho"]]
    for field_name in ("beta_a", "beta_sigma", "k_f", "k_b"):
        decoded[field_name] = float_from_hex(decoded[field_name])
    if decoded["poisson_exposure"] is not None:
        decoded["poisson_exposure"] = float_from_hex(decoded["poisson_exposure"])
    if decoded["fission_probability"] is not None:
        decoded["fission_probability"] = float_from_hex(decoded["fission_probability"])
    return specification_from_mapping(decoded)


def _rng_use(
    *,
    purpose: StreamPurpose,
    stream_id: str,
    before: str,
    after: str,
    consumed: bool,
) -> dict[str, Any]:
    return {
        "purpose": purpose.value,
        "streamId": stream_id,
        "stateSha256Before": before,
        "stateSha256After": after,
        "consumed": consumed,
    }


def _event_payload(event: EventLog, *, trajectory_id: str) -> dict[str, Any]:
    generation = event.generation_index_one_based
    step = event.step_index_one_based
    waiting: dict[str, Any] | None = None
    if event.waiting_rng_stream_id is not None:
        waiting = _rng_use(
            purpose=StreamPurpose.WAITING_TIME,
            stream_id=event.waiting_rng_stream_id,
            before=event.waiting_rng_state_sha256_before or "",
            after=event.waiting_rng_state_sha256_after or "",
            consumed=True,
        )
    return {
        "eventId": f"{trajectory_id}/generation-{generation}/event-{step}",
        "specificationId": event.specification_id,
        "generationIndexOneBased": generation,
        "stepIndexOneBased": step,
        "preEventState": list(event.pre_state),
        "postEventState": list(event.post_state),
        "preEventMass": event.pre_mass,
        "postEventMass": event.post_mass,
        "massDelta": event.mass_delta,
        "eventIdentity": {
            "updateKernel": event.update_kernel,
            "eventKind": event.event_kind,
            "selectedEventIndexZeroBased": event.selected_event_index_zero_based,
            "selectedSpeciesIndexZeroBased": event.selected_species_index_zero_based,
            "selectionProbabilityHex": None
            if event.selection_probability is None
            else float_to_hex(event.selection_probability),
        },
        "propensities": {
            "equationBranch": event.propensity_equation_branch,
            "boostHex": _hex_sequence(event.boost),
            "joinHex": _hex_sequence(event.join_propensities),
            "leaveHex": _hex_sequence(event.leave_propensities),
            "concatenatedHex": _hex_sequence(
                event.join_propensities + event.leave_propensities
            ),
            "normalizedHex": _hex_sequence(event.event_probabilities),
            "totalHex": float_to_hex(event.total_propensity),
        },
        "updateCounts": {
            "attemptedJoin": list(event.attempted_join_counts),
            "attemptedLoss": list(event.attempted_loss_counts),
            "appliedJoin": list(event.applied_join_counts),
            "appliedLoss": list(event.applied_loss_counts),
        },
        "boundaryAction": event.boundary_action,
        "clock": {
            "semantics": event.clock_semantics,
            "timeIncrementHex": None
            if event.time_increment is None
            else float_to_hex(event.time_increment),
            "modelTimeBeforeHex": None
            if event.model_time_before is None
            else float_to_hex(event.model_time_before),
            "modelTimeAfterHex": None
            if event.model_time_after is None
            else float_to_hex(event.model_time_after),
        },
        "rngUses": {
            "event": _rng_use(
                purpose=StreamPurpose.EVENT,
                stream_id=event.event_rng_stream_id,
                before=event.event_rng_state_sha256_before,
                after=event.event_rng_state_sha256_after,
                consumed=True,
            ),
            "waitingTime": waiting,
        },
    }


def _fission_payload(fission: FissionLog, *, trajectory_id: str) -> dict[str, Any]:
    generation = fission.generation_index_one_based
    return {
        "fissionId": f"{trajectory_id}/generation-{generation}/fission",
        "specificationId": fission.specification_id,
        "generationIndexOneBased": generation,
        "preFissionState": list(fission.parent),
        "fissionSemantics": fission.fission_semantics,
        "fissionProbabilityHex": None
        if fission.fission_probability is None
        else float_to_hex(fission.fission_probability),
        "result": {
            "childFirst": list(fission.child_first),
            "childSecond": list(fission.child_second),
            "discarded": list(fission.discarded),
            "conservationHolds": fission.conservation_holds,
        },
        "daughterChoice": {
            "selectionSemantics": fission.daughter_selection,
            "selectedLabel": fission.selected_daughter_label,
            "selectedState": list(fission.selected_daughter),
            "postFissionSemantics": fission.post_fission_semantics,
        },
        "rngUses": {
            "fission": _rng_use(
                purpose=StreamPurpose.FISSION,
                stream_id=fission.fission_rng_stream_id,
                before=fission.fission_rng_state_sha256_before,
                after=fission.fission_rng_state_sha256_after,
                consumed=(
                    fission.fission_rng_state_sha256_before
                    != fission.fission_rng_state_sha256_after
                ),
            ),
            "daughterSelection": _rng_use(
                purpose=StreamPurpose.DAUGHTER_SELECTION,
                stream_id=fission.daughter_rng_stream_id,
                before=fission.daughter_rng_state_sha256_before,
                after=fission.daughter_rng_state_sha256_after,
                consumed=fission.daughter_rng_consumed,
            ),
        },
    }


def _generation_payload(
    generation: GenerationResult,
    *,
    trajectory_id: str,
) -> dict[str, Any]:
    growth = generation.growth
    return {
        "generationId": (
            f"{trajectory_id}/generation-{generation.generation_index_one_based}"
        ),
        "specificationId": generation.specification_id,
        "generationIndexOneBased": generation.generation_index_one_based,
        "growth": {
            "initialState": list(growth.initial_state),
            "finalState": list(growth.final_state),
            "stoppingReason": growth.terminal_status,
            "elapsedModelTimeHex": None
            if growth.elapsed_model_time is None
            else float_to_hex(growth.elapsed_model_time),
            "events": [
                _event_payload(event, trajectory_id=trajectory_id)
                for event in growth.events
            ],
        },
        "fission": None
        if generation.fission is None
        else _fission_payload(generation.fission, trajectory_id=trajectory_id),
        "postGenerationState": None
        if generation.next_state is None
        else list(generation.next_state),
        "stoppingReason": generation.terminal_status,
    }


def _matrix_payload(beta: np.ndarray) -> dict[str, Any]:
    values = [_hex_sequence(row) for row in np.asarray(beta, dtype=np.float64)]
    return {
        "dtype": "float64",
        "shape": [int(beta.shape[0]), int(beta.shape[1])],
        "valuesHex": values,
        "canonicalValuesSha256": sha256_hex(canonical_json_bytes(values)),
    }


def _lineage_payload(
    lineage: LineageResult,
    *,
    trajectory_id: str,
) -> list[dict[str, Any]]:
    return [
        _generation_payload(generation, trajectory_id=trajectory_id)
        for generation in lineage.generations
    ]


def capture_independent_trajectory(
    *,
    specification: GardSpecification,
    seed_bundle: SeedBundle,
    capture_identity: CaptureIdentity,
    registry_boundary: RegistryBoundary,
) -> dict[str, Any]:
    """Generate and losslessly capture one complete independent-engine lineage."""

    request = seed_bundle.request
    if request.experiment_id != "E01":
        raise TrajectoryContractError("S06 trajectory capture requires experiment E01.")
    if request.specification_id != specification.specification_id:
        raise TrajectoryContractError("Seed/specification identity mismatch.")
    if request.engine_id != capture_identity.engine_id:
        raise TrajectoryContractError("Seed/engine identity mismatch.")
    if registry_boundary.registry_executable:
        raise TrajectoryContractError(
            "The closed author registry cannot be relabeled executable by an S06 fixture."
        )
    if not registry_boundary.no_silent_defaults:
        raise TrajectoryContractError(
            "Registry no-silent-defaults guard must remain true."
        )

    generators = seed_bundle.fresh_generators()
    initial_digests = {
        purpose: generator_state_sha256(generators[purpose])
        for purpose in CANONICAL_STREAM_PURPOSES
    }
    engine_streams = seed_bundle.independent_engine_streams(generators)

    matrix_before = generator_state_sha256(generators[StreamPurpose.CATALYTIC_MATRIX])
    beta = generate_catalytic_matrix(specification, engine_streams.catalytic_matrix)
    matrix_after = generator_state_sha256(generators[StreamPurpose.CATALYTIC_MATRIX])

    initialization_before = generator_state_sha256(
        generators[StreamPurpose.INITIAL_STATE]
    )
    initial_state = initialize_state(specification, engine_streams.initialization)
    initialization_after = generator_state_sha256(
        generators[StreamPurpose.INITIAL_STATE]
    )

    lineage = simulate_lineage(
        initial_state,
        beta=beta,
        specification=specification,
        rng_streams=engine_streams,
    )
    terminal_digests = {
        purpose: generator_state_sha256(generators[purpose])
        for purpose in CANONICAL_STREAM_PURPOSES
    }
    seed_payload = seed_bundle.to_payload()
    seed_checksum = sha256_hex(canonical_json_bytes(seed_payload))
    trajectory_id = request.trajectory_id
    payload: dict[str, Any] = {
        "trajectorySchemaVersion": TRAJECTORY_SCHEMA_VERSION,
        "precisionContractVersion": PRECISION_CONTRACT_VERSION,
        "experimentId": request.experiment_id,
        "specificationId": specification.specification_id,
        "trajectoryId": trajectory_id,
        "replicateIndex": request.replicate_index,
        "engineIdentity": capture_identity.to_payload(),
        "specification": specification_to_payload(specification),
        "seedIdentity": {
            "seedManifestPayloadSha256": seed_checksum,
            "seedManifest": seed_payload,
        },
        "registryBoundary": registry_boundary.to_payload(),
        "samplingBoundary": {
            "recordedStateBoundaries": [
                "initial_state",
                "pre_event",
                "post_event",
                "pre_fission",
                "child_first",
                "child_second",
                "discarded",
                "selected_daughter",
                "terminal_state",
            ],
            "analysisSamplingInstant": "UNRESOLVED::E01-A025",
            "rule": (
                "All event and fission boundaries are stored losslessly; no downstream "
                "analysis sampling instant is selected by this schema."
            ),
        },
        "setup": {
            "catalyticMatrix": _matrix_payload(beta),
            "catalyticMatrixRngUse": _rng_use(
                purpose=StreamPurpose.CATALYTIC_MATRIX,
                stream_id=seed_bundle.streams[StreamPurpose.CATALYTIC_MATRIX].stream_id,
                before=matrix_before,
                after=matrix_after,
                consumed=matrix_before != matrix_after,
            ),
            "initialState": list(initial_state),
            "initialStateRngUse": _rng_use(
                purpose=StreamPurpose.INITIAL_STATE,
                stream_id=seed_bundle.streams[StreamPurpose.INITIAL_STATE].stream_id,
                before=initialization_before,
                after=initialization_after,
                consumed=initialization_before != initialization_after,
            ),
        },
        "generations": _lineage_payload(lineage, trajectory_id=trajectory_id),
        "terminal": {
            "finalState": list(lineage.final_state),
            "requestedGenerations": lineage.requested_generations,
            "completedFissions": lineage.completed_fissions,
            "stoppingReason": lineage.terminal_status,
        },
        "rngTerminalStates": {
            purpose.value: {
                "streamId": seed_bundle.streams[purpose].stream_id,
                "initialStateSha256": initial_digests[purpose],
                "terminalStateSha256": terminal_digests[purpose],
                "consumed": initial_digests[purpose] != terminal_digests[purpose],
            }
            for purpose in CANONICAL_STREAM_PURPOSES
        },
    }
    validate_trajectory_invariants(payload)
    return payload


def _decode_vector(values: Any, *, length: int, name: str) -> tuple[int, ...]:
    if not isinstance(values, list) or len(values) != length:
        raise TrajectoryContractError(f"{name} must contain {length} integer counts.")
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > MAX_SIGNED_INT64
        for value in values
    ):
        raise TrajectoryContractError(
            f"{name} contains molecule counts outside nonnegative signed int64."
        )
    return tuple(values)


def _validate_rng_use(
    record: dict[str, Any],
    *,
    expected_purpose: StreamPurpose,
    expected_stream_id: str,
    expected_state_before: str,
    require_consumed: bool | None = None,
) -> str:
    if record["purpose"] != expected_purpose.value:
        raise TrajectoryContractError("RNG purpose mismatch.")
    if record["streamId"] != expected_stream_id:
        raise TrajectoryContractError("RNG stream identity mismatch.")
    if record["stateSha256Before"] != expected_state_before:
        raise TrajectoryContractError("RNG state chain is discontinuous.")
    changed = record["stateSha256Before"] != record["stateSha256After"]
    if record["consumed"] != changed:
        raise TrajectoryContractError("RNG consumption flag contradicts state digests.")
    if require_consumed is not None and record["consumed"] is not require_consumed:
        raise TrajectoryContractError(
            "RNG consumption does not match branch semantics."
        )
    return record["stateSha256After"]


def _validate_event(
    event: dict[str, Any],
    *,
    specification: GardSpecification,
    trajectory_id: str,
    specification_id: str,
    generation_index: int,
    expected_step: int,
    stream_ids: dict[StreamPurpose, str],
    rng_cursors: dict[StreamPurpose, str],
) -> tuple[int, ...]:
    n_species = specification.n_species
    if event["eventId"] != (
        f"{trajectory_id}/generation-{generation_index}/event-{expected_step}"
    ):
        raise TrajectoryContractError("Event identity is not canonical.")
    if event["specificationId"] != specification_id:
        raise TrajectoryContractError("Event specification identity mismatch.")
    if event["generationIndexOneBased"] != generation_index:
        raise TrajectoryContractError("Event generation index mismatch.")
    if event["stepIndexOneBased"] != expected_step:
        raise TrajectoryContractError("Event step index mismatch.")
    pre = _decode_vector(event["preEventState"], length=n_species, name="preEventState")
    post = _decode_vector(
        event["postEventState"], length=n_species, name="postEventState"
    )
    if sum(pre) != event["preEventMass"] or sum(post) != event["postEventMass"]:
        raise TrajectoryContractError("Event mass fields do not equal state sums.")
    if sum(post) - sum(pre) != event["massDelta"]:
        raise TrajectoryContractError("Event mass delta is inconsistent.")

    identity = event["eventIdentity"]
    if identity["updateKernel"] != specification.update_kernel.value:
        raise TrajectoryContractError("Event update-kernel identity mismatch.")

    propensities = event["propensities"]
    if propensities["equationBranch"] != specification.propensity_equation_branch.value:
        raise TrajectoryContractError("Event propensity-branch identity mismatch.")
    boost = [float_from_hex(value) for value in propensities["boostHex"]]
    join = [float_from_hex(value) for value in propensities["joinHex"]]
    leave = [float_from_hex(value) for value in propensities["leaveHex"]]
    concatenated = [float_from_hex(value) for value in propensities["concatenatedHex"]]
    normalized = [float_from_hex(value) for value in propensities["normalizedHex"]]
    if not all(
        len(values) == expected
        for values, expected in (
            (boost, n_species),
            (join, n_species),
            (leave, n_species),
            (concatenated, 2 * n_species),
            (normalized, 2 * n_species),
        )
    ):
        raise TrajectoryContractError("Propensity vector dimension mismatch.")
    if concatenated != join + leave:
        raise TrajectoryContractError("Concatenated propensity vector is inconsistent.")
    if any(value < 0.0 for value in boost + concatenated + normalized):
        raise TrajectoryContractError("Propensity records contain negative values.")
    total = float_from_hex(propensities["totalHex"])
    if float(np.asarray(concatenated, dtype=np.float64).sum()) != total:
        raise TrajectoryContractError("Total propensity is not the stored vector sum.")
    if total <= 0:
        raise TrajectoryContractError("A recorded event must have positive propensity.")
    expected_normalized = list(
        np.asarray(concatenated, dtype=np.float64) / np.float64(total)
    )
    if normalized != expected_normalized:
        raise TrajectoryContractError(
            "Normalized propensities are not lossless ratios."
        )
    if not isclose(sum(normalized), 1.0, rel_tol=1e-15, abs_tol=1e-15):
        raise TrajectoryContractError("Normalized propensities do not sum to one.")

    counts = event["updateCounts"]
    attempted_join = _decode_vector(
        counts["attemptedJoin"], length=n_species, name="attemptedJoin"
    )
    attempted_loss = _decode_vector(
        counts["attemptedLoss"], length=n_species, name="attemptedLoss"
    )
    applied_join = _decode_vector(
        counts["appliedJoin"], length=n_species, name="appliedJoin"
    )
    applied_loss = _decode_vector(
        counts["appliedLoss"], length=n_species, name="appliedLoss"
    )
    if any(
        applied_join[index] > attempted_join[index]
        or applied_loss[index] > attempted_loss[index]
        for index in range(n_species)
    ):
        raise TrajectoryContractError("Applied update counts exceed attempted counts.")
    reconstructed_post = tuple(
        pre[index] + applied_join[index] - applied_loss[index]
        for index in range(n_species)
    )
    if reconstructed_post != post:
        raise TrajectoryContractError(
            "Event update counts do not reconstruct post-state."
        )

    if identity["eventKind"] == "vector_poisson_batch":
        if (
            identity["selectedEventIndexZeroBased"] is not None
            or identity["selectedSpeciesIndexZeroBased"] is not None
            or identity["selectionProbabilityHex"] is not None
        ):
            raise TrajectoryContractError(
                "Vector-Poisson event has categorical identity."
            )
    else:
        selected_event = identity["selectedEventIndexZeroBased"]
        selected_species = identity["selectedSpeciesIndexZeroBased"]
        if (
            not isinstance(selected_event, int)
            or isinstance(selected_event, bool)
            or selected_event < 0
            or selected_event >= 2 * n_species
        ):
            raise TrajectoryContractError("Categorical event index is invalid.")
        expected_species = selected_event % n_species
        expected_kind = "join" if selected_event < n_species else "leave"
        if (
            selected_species != expected_species
            or identity["eventKind"] != expected_kind
        ):
            raise TrajectoryContractError("Categorical event identity is inconsistent.")
        selection_probability = float_from_hex(identity["selectionProbabilityHex"])
        if selection_probability != normalized[selected_event]:
            raise TrajectoryContractError("Selected-event probability is inconsistent.")
        expected_join = tuple(
            1 if expected_kind == "join" and index == expected_species else 0
            for index in range(n_species)
        )
        expected_loss = tuple(
            1 if expected_kind == "leave" and index == expected_species else 0
            for index in range(n_species)
        )
        if not (
            attempted_join == applied_join == expected_join
            and attempted_loss == applied_loss == expected_loss
        ):
            raise TrajectoryContractError("Categorical update counts are inconsistent.")

    event_rng = event["rngUses"]["event"]
    rng_cursors[StreamPurpose.EVENT] = _validate_rng_use(
        event_rng,
        expected_purpose=StreamPurpose.EVENT,
        expected_stream_id=stream_ids[StreamPurpose.EVENT],
        expected_state_before=rng_cursors[StreamPurpose.EVENT],
        require_consumed=True,
    )
    waiting = event["rngUses"]["waitingTime"]
    clock = event["clock"]
    if clock["semantics"] != specification.clock_semantics.value:
        raise TrajectoryContractError("Event clock-semantics identity mismatch.")
    clock_fields = (
        clock["timeIncrementHex"],
        clock["modelTimeBeforeHex"],
        clock["modelTimeAfterHex"],
    )
    if clock["semantics"] == "gillespie_exponential":
        if waiting is None or any(value is None for value in clock_fields):
            raise TrajectoryContractError("Gillespie event omitted waiting-time data.")
        rng_cursors[StreamPurpose.WAITING_TIME] = _validate_rng_use(
            waiting,
            expected_purpose=StreamPurpose.WAITING_TIME,
            expected_stream_id=stream_ids[StreamPurpose.WAITING_TIME],
            expected_state_before=rng_cursors[StreamPurpose.WAITING_TIME],
            require_consumed=True,
        )
        before = float_from_hex(clock["modelTimeBeforeHex"])
        increment = float_from_hex(clock["timeIncrementHex"])
        after = float_from_hex(clock["modelTimeAfterHex"])
        if increment < 0 or before + increment != after:
            raise TrajectoryContractError("Event clock addition is inconsistent.")
    elif clock["semantics"] == "event_index_only":
        if waiting is not None or any(value is not None for value in clock_fields):
            raise TrajectoryContractError("Event-index clock contains time/RNG data.")
    else:
        if waiting is not None or any(value is None for value in clock_fields):
            raise TrajectoryContractError(
                "Fixed-exposure clock fields are inconsistent."
            )
        before = float_from_hex(clock["modelTimeBeforeHex"])
        increment = float_from_hex(clock["timeIncrementHex"])
        after = float_from_hex(clock["modelTimeAfterHex"])
        if increment != specification.poisson_exposure or before + increment != after:
            raise TrajectoryContractError(
                "Fixed-exposure clock addition is inconsistent."
            )
    return post


def _validate_fission(
    fission: dict[str, Any],
    *,
    specification: GardSpecification,
    trajectory_id: str,
    specification_id: str,
    generation_index: int,
    stream_ids: dict[StreamPurpose, str],
    rng_cursors: dict[StreamPurpose, str],
) -> tuple[int, ...]:
    n_species = specification.n_species
    if fission["fissionId"] != f"{trajectory_id}/generation-{generation_index}/fission":
        raise TrajectoryContractError("Fission identity is not canonical.")
    if (
        fission["specificationId"] != specification_id
        or fission["generationIndexOneBased"] != generation_index
    ):
        raise TrajectoryContractError("Fission identity fields mismatch.")
    parent = _decode_vector(
        fission["preFissionState"], length=n_species, name="preFissionState"
    )
    result = fission["result"]
    first = _decode_vector(result["childFirst"], length=n_species, name="childFirst")
    second = _decode_vector(result["childSecond"], length=n_species, name="childSecond")
    discarded = _decode_vector(result["discarded"], length=n_species, name="discarded")
    conservation = all(
        first[index] + second[index] + discarded[index] == parent[index]
        for index in range(n_species)
    )
    if result["conservationHolds"] is not True or not conservation:
        raise TrajectoryContractError("Fission conservation is false.")
    if fission["fissionSemantics"] != specification.fission_semantics.value:
        raise TrajectoryContractError("Fission semantics mismatch specification.")
    probability = fission["fissionProbabilityHex"]
    decoded_probability = None if probability is None else float_from_hex(probability)
    if decoded_probability != specification.fission_probability:
        raise TrajectoryContractError("Fission probability mismatch specification.")
    if fission["fissionSemantics"] == "fixed_size_without_replacement_odd_discard":
        target = sum(parent) // 2
        if (
            sum(first) != target
            or sum(second) != target
            or sum(discarded) != sum(parent) % 2
        ):
            raise TrajectoryContractError("Fixed-size fission masses are inconsistent.")
    elif any(discarded):
        raise TrajectoryContractError(
            "Binomial-complement fission discarded molecules."
        )
    choice = fission["daughterChoice"]
    selected = _decode_vector(
        choice["selectedState"], length=n_species, name="selectedState"
    )
    expected = first if choice["selectedLabel"] == "first" else second
    if selected != expected:
        raise TrajectoryContractError("Selected daughter does not match its label.")
    if (
        choice["selectionSemantics"] != specification.daughter_selection.value
        or choice["postFissionSemantics"] != specification.post_fission_semantics.value
    ):
        raise TrajectoryContractError(
            "Daughter-choice semantics mismatch specification."
        )
    if (
        choice["selectionSemantics"] in {"first", "second"}
        and choice["selectedLabel"] != choice["selectionSemantics"]
    ):
        raise TrajectoryContractError("Deterministic daughter label is inconsistent.")
    rng_cursors[StreamPurpose.FISSION] = _validate_rng_use(
        fission["rngUses"]["fission"],
        expected_purpose=StreamPurpose.FISSION,
        expected_stream_id=stream_ids[StreamPurpose.FISSION],
        expected_state_before=rng_cursors[StreamPurpose.FISSION],
    )
    rng_cursors[StreamPurpose.DAUGHTER_SELECTION] = _validate_rng_use(
        fission["rngUses"]["daughterSelection"],
        expected_purpose=StreamPurpose.DAUGHTER_SELECTION,
        expected_stream_id=stream_ids[StreamPurpose.DAUGHTER_SELECTION],
        expected_state_before=rng_cursors[StreamPurpose.DAUGHTER_SELECTION],
        require_consumed=choice["selectionSemantics"] == "uniform_random",
    )
    return selected


def validate_trajectory_invariants(payload: dict[str, Any]) -> None:
    """Validate identities, checksums, integer-state chains, and RNG bindings."""

    required = {
        "trajectorySchemaVersion",
        "precisionContractVersion",
        "experimentId",
        "specificationId",
        "trajectoryId",
        "replicateIndex",
        "engineIdentity",
        "specification",
        "seedIdentity",
        "registryBoundary",
        "samplingBoundary",
        "setup",
        "generations",
        "terminal",
        "rngTerminalStates",
    }
    if set(payload) != required:
        raise TrajectoryContractError(
            f"Trajectory fields mismatch; missing={sorted(required - set(payload))}, "
            f"extra={sorted(set(payload) - required)}."
        )
    if payload["trajectorySchemaVersion"] != TRAJECTORY_SCHEMA_VERSION:
        raise TrajectoryContractError("Unsupported trajectory schema version.")
    if payload["precisionContractVersion"] != PRECISION_CONTRACT_VERSION:
        raise TrajectoryContractError("Unsupported precision contract version.")
    specification = specification_from_payload(payload["specification"])
    if specification.specification_id != payload["specificationId"]:
        raise TrajectoryContractError("Top-level specification ID mismatch.")
    seed_payload = payload["seedIdentity"]["seedManifest"]
    if (
        sha256_hex(canonical_json_bytes(seed_payload))
        != payload["seedIdentity"]["seedManifestPayloadSha256"]
    ):
        raise TrajectoryContractError("Embedded seed manifest checksum mismatch.")
    request = seed_request_from_payload(seed_payload)
    for top, expected in (
        ("experimentId", request.experiment_id),
        ("specificationId", request.specification_id),
        ("trajectoryId", request.trajectory_id),
        ("replicateIndex", request.replicate_index),
    ):
        if payload[top] != expected:
            raise TrajectoryContractError(f"Top-level {top} mismatches seed identity.")
    if payload["engineIdentity"]["engineId"] != request.engine_id:
        raise TrajectoryContractError("Engine identity mismatches seed identity.")
    boundary = payload["registryBoundary"]
    if (
        boundary["registryExecutable"] is not False
        or boundary["noSilentDefaults"] is not True
    ):
        raise TrajectoryContractError("Closed registry guard was altered.")
    if boundary["authorInitialStateRngStream"] != "UNRESOLVED::E01-A020":
        raise TrajectoryContractError("Author RNG sentinel was altered.")
    if boundary["analysisStateSamplingInstant"] != "UNRESOLVED::E01-A025":
        raise TrajectoryContractError("Sampling-instant sentinel was altered.")
    if payload["samplingBoundary"]["analysisSamplingInstant"] != "UNRESOLVED::E01-A025":
        raise TrajectoryContractError("Sampling boundary silently selected an instant.")

    n_species = specification.n_species
    matrix = payload["setup"]["catalyticMatrix"]
    if (
        matrix["shape"] != [n_species, n_species]
        or len(matrix["valuesHex"]) != n_species
    ):
        raise TrajectoryContractError("Catalytic matrix dimensions mismatch.")
    for row in matrix["valuesHex"]:
        if len(row) != n_species:
            raise TrajectoryContractError("Catalytic matrix row dimension mismatch.")
        for value in row:
            float_from_hex(value)
    if (
        sha256_hex(canonical_json_bytes(matrix["valuesHex"]))
        != matrix["canonicalValuesSha256"]
    ):
        raise TrajectoryContractError("Catalytic matrix value checksum mismatch.")
    initial = _decode_vector(
        payload["setup"]["initialState"], length=n_species, name="initialState"
    )
    stream_ids = {
        purpose: seed_payload["streams"][purpose.value]["streamId"]
        for purpose in CANONICAL_STREAM_PURPOSES
    }
    rng_cursors = {
        purpose: seed_payload["streams"][purpose.value]["initialStateSha256"]
        for purpose in CANONICAL_STREAM_PURPOSES
    }
    rng_cursors[StreamPurpose.CATALYTIC_MATRIX] = _validate_rng_use(
        payload["setup"]["catalyticMatrixRngUse"],
        expected_purpose=StreamPurpose.CATALYTIC_MATRIX,
        expected_stream_id=stream_ids[StreamPurpose.CATALYTIC_MATRIX],
        expected_state_before=rng_cursors[StreamPurpose.CATALYTIC_MATRIX],
    )
    rng_cursors[StreamPurpose.INITIAL_STATE] = _validate_rng_use(
        payload["setup"]["initialStateRngUse"],
        expected_purpose=StreamPurpose.INITIAL_STATE,
        expected_stream_id=stream_ids[StreamPurpose.INITIAL_STATE],
        expected_state_before=rng_cursors[StreamPurpose.INITIAL_STATE],
    )

    current = initial
    completed_fissions = 0
    generations = payload["generations"]
    if not isinstance(generations, list) or not generations:
        raise TrajectoryContractError(
            "Trajectory must contain at least one generation."
        )
    if len(generations) > specification.n_generations:
        raise TrajectoryContractError("Trajectory exceeds requested generations.")
    for expected_generation, generation in enumerate(generations, start=1):
        if generation["generationId"] != (
            f"{payload['trajectoryId']}/generation-{expected_generation}"
        ):
            raise TrajectoryContractError("Generation identity is not canonical.")
        if (
            generation["specificationId"] != specification.specification_id
            or generation["generationIndexOneBased"] != expected_generation
        ):
            raise TrajectoryContractError("Generation identity fields mismatch.")
        growth = generation["growth"]
        growth_initial = _decode_vector(
            growth["initialState"], length=n_species, name="growth.initialState"
        )
        if growth_initial != current:
            raise TrajectoryContractError("Generation chain is discontinuous.")
        event_current = growth_initial
        for expected_step, event in enumerate(growth["events"], start=1):
            if tuple(event["preEventState"]) != event_current:
                raise TrajectoryContractError("Event state chain is discontinuous.")
            event_current = _validate_event(
                event,
                specification=specification,
                trajectory_id=payload["trajectoryId"],
                specification_id=specification.specification_id,
                generation_index=expected_generation,
                expected_step=expected_step,
                stream_ids=stream_ids,
                rng_cursors=rng_cursors,
            )
        final_state = _decode_vector(
            growth["finalState"], length=n_species, name="growth.finalState"
        )
        if final_state != event_current:
            raise TrajectoryContractError("Growth final state mismatches event chain.")
        fission = generation["fission"]
        if fission is None:
            if generation["postGenerationState"] is not None:
                raise TrajectoryContractError(
                    "Non-fission generation has a next state."
                )
            current = final_state
        else:
            if tuple(fission["preFissionState"]) != final_state:
                raise TrajectoryContractError(
                    "Fission parent mismatches growth final state."
                )
            selected = _validate_fission(
                fission,
                specification=specification,
                trajectory_id=payload["trajectoryId"],
                specification_id=specification.specification_id,
                generation_index=expected_generation,
                stream_ids=stream_ids,
                rng_cursors=rng_cursors,
            )
            next_state = _decode_vector(
                generation["postGenerationState"],
                length=n_species,
                name="postGenerationState",
            )
            if selected != next_state:
                raise TrajectoryContractError(
                    "Post-generation state is not selected daughter."
                )
            current = next_state
            completed_fissions += 1

        elapsed = growth["elapsedModelTimeHex"]
        if specification.clock_semantics.value == "event_index_only":
            if elapsed is not None:
                raise TrajectoryContractError("Event-index growth has elapsed time.")
        else:
            decoded_elapsed = float_from_hex(elapsed)
            expected_elapsed = (
                0.0
                if not growth["events"]
                else float_from_hex(growth["events"][-1]["clock"]["modelTimeAfterHex"])
            )
            if decoded_elapsed != expected_elapsed:
                raise TrajectoryContractError("Growth elapsed time is inconsistent.")

        if fission is None:
            valid_nonfission_statuses = {
                growth["stoppingReason"],
                "max_steps_stop_without_fission",
            }
            if generation["stoppingReason"] not in valid_nonfission_statuses:
                raise TrajectoryContractError(
                    "Non-fission stopping reason is inconsistent."
                )
        else:
            expected_status = (
                "selected_empty_daughter"
                if sum(current) == 0
                else "continued_from_selected_daughter"
            )
            if generation["stoppingReason"] != expected_status:
                raise TrajectoryContractError(
                    "Post-fission stopping reason is inconsistent."
                )
        if expected_generation < len(generations) and (
            fission is None or sum(current) == 0
        ):
            raise TrajectoryContractError(
                "Trajectory continued after a terminal generation."
            )

    terminal = payload["terminal"]
    final = _decode_vector(
        terminal["finalState"], length=n_species, name="terminal.finalState"
    )
    if final != current:
        raise TrajectoryContractError(
            "Terminal final state mismatches generation chain."
        )
    if terminal["requestedGenerations"] != specification.n_generations:
        raise TrajectoryContractError("Requested-generation identity mismatch.")
    if terminal["completedFissions"] != completed_fissions:
        raise TrajectoryContractError("Completed-fission count mismatch.")
    last_generation = generations[-1]
    expected_terminal_reason = (
        "requested_generations_completed"
        if len(generations) == specification.n_generations
        and last_generation["postGenerationState"] is not None
        and sum(last_generation["postGenerationState"]) > 0
        else last_generation["stoppingReason"]
    )
    if terminal["stoppingReason"] != expected_terminal_reason:
        raise TrajectoryContractError("Terminal stopping reason is inconsistent.")

    terminal_states = payload["rngTerminalStates"]
    if set(terminal_states) != {purpose.value for purpose in CANONICAL_STREAM_PURPOSES}:
        raise TrajectoryContractError("Terminal RNG inventory is incomplete.")
    for purpose in CANONICAL_STREAM_PURPOSES:
        record = terminal_states[purpose.value]
        if record["streamId"] != stream_ids[purpose]:
            raise TrajectoryContractError("Terminal RNG stream ID mismatch.")
        if (
            record["initialStateSha256"]
            != seed_payload["streams"][purpose.value]["initialStateSha256"]
        ):
            raise TrajectoryContractError("Terminal RNG initial digest mismatch.")
        if record["terminalStateSha256"] != rng_cursors[purpose]:
            raise TrajectoryContractError(
                "Terminal RNG digest mismatches logged draw chain."
            )
        changed = record["initialStateSha256"] != record["terminalStateSha256"]
        if changed != record["consumed"]:
            raise TrajectoryContractError("Terminal RNG consumption flag mismatch.")
    for auxiliary in (
        StreamPurpose.INTERVENTION,
        StreamPurpose.ESTIMATOR,
        StreamPurpose.MACHINE_LEARNING,
    ):
        if terminal_states[auxiliary.value]["consumed"]:
            raise TrajectoryContractError(
                f"Auxiliary stream {auxiliary.value} was coupled into simulation."
            )


def regenerate_independent_trajectory(
    *,
    reference_payload: dict[str, Any],
    capture_identity: CaptureIdentity,
    registry_boundary: RegistryBoundary,
) -> dict[str, Any]:
    """Regenerate a trajectory from its embedded specification and seed manifest."""

    validate_trajectory_invariants(reference_payload)
    if reference_payload["engineIdentity"] != capture_identity.to_payload():
        raise TrajectoryContractError(
            "Current capture identity differs from the reference same-engine identity."
        )
    if reference_payload["registryBoundary"] != registry_boundary.to_payload():
        raise TrajectoryContractError(
            "Current registry boundary differs from reference."
        )
    specification = specification_from_payload(reference_payload["specification"])
    request = seed_request_from_payload(
        reference_payload["seedIdentity"]["seedManifest"]
    )
    regenerated = capture_independent_trajectory(
        specification=specification,
        seed_bundle=derive_seed_bundle(request),
        capture_identity=capture_identity,
        registry_boundary=registry_boundary,
    )
    return regenerated


def binary64_ulp_distance(left: float, right: float) -> int:
    """Return the unsigned ULP distance between two finite binary64 values."""

    left_value = float(left)
    right_value = float(right)
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        raise SerializationError("ULP distance requires finite values.")
    left_bits = int.from_bytes(struct.pack(">d", left_value), "big")
    right_bits = int.from_bytes(struct.pack(">d", right_value), "big")

    def ordered(bits: int) -> int:
        sign = 1 << 63
        return (~bits & ((1 << 64) - 1)) if bits & sign else bits | sign

    return abs(ordered(left_bits) - ordered(right_bits))


def payload_sha256(payload: dict[str, Any]) -> str:
    """Hash one canonical trajectory payload."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
