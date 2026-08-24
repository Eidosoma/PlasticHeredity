"""Versioned, domain-separated random-stream identities for E01.

The seed hierarchy in this module is a reconstruction engineering contract.  It
does not identify the unavailable paper-author RNG policy and it does not claim
legacy MATLAB stream compatibility.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np

from e01_gard_independent import RNGInput, RNGStreams

SEED_SCHEMA_VERSION = "E01-seed-schema-v1.0.0"
SEED_DERIVATION_ALGORITHM = "E01-SHA256-DOMAIN-SEPARATION-PCG64DXSM-v1"
CANONICAL_JSON_VERSION = "E01-canonical-json-v1.0.0"
BIT_GENERATOR_NAME = "PCG64DXSM"
DERIVATION_DOMAIN = b"EIDOSOMA-E01-SEED-DERIVATION-v1\x00"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$")


class SeedContractError(ValueError):
    """A seed manifest violates the frozen S06 contract."""


class StreamPurpose(StrEnum):
    """Canonical domain labels; values are part of the seed preimage."""

    CATALYTIC_MATRIX = "catalytic_matrix"
    INITIAL_STATE = "initial_state"
    EVENT = "event"
    WAITING_TIME = "waiting_time"
    FISSION = "fission"
    DAUGHTER_SELECTION = "daughter_selection"
    INTERVENTION = "intervention"
    ESTIMATOR = "estimator"
    MACHINE_LEARNING = "machine_learning"


class CouplingPolicy(StrEnum):
    """Whether namespaces isolate one trajectory or explicitly couple runs."""

    TRAJECTORY_ISOLATED = "trajectory_isolated"
    EXPLICIT_COMMON_RANDOM_NUMBERS = "explicit_common_random_numbers"


CANONICAL_STREAM_PURPOSES = tuple(StreamPurpose)


def _assert_json_domain(value: Any, *, path: str = "$.") -> None:
    """Reject values outside the canonical JSON subset used for checksums."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise SeedContractError(
            f"{path} contains a JSON float; encode binary64 values as hex strings."
        )
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_domain(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SeedContractError(f"{path} contains a non-string object key.")
            _assert_json_domain(item, path=f"{path}{key}.")
        return
    raise SeedContractError(
        f"{path} contains unsupported canonical JSON type {type(value).__name__}."
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize the frozen no-float JSON subset deterministically as UTF-8."""

    _assert_json_domain(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return a lowercase SHA-256 hexadecimal digest."""

    return hashlib.sha256(data).hexdigest()


def _validate_identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise SeedContractError(
            f"{name} must be a nonempty ASCII identifier matching "
            "[A-Za-z0-9][A-Za-z0-9._:/@+-]*."
        )
    return value


def isolated_stream_namespace(
    *,
    experiment_id: str,
    specification_id: str,
    trajectory_id: str,
    replicate_index: int,
) -> str:
    """Return the only namespace accepted by trajectory-isolated manifests."""

    for name, value in (
        ("experiment_id", experiment_id),
        ("specification_id", specification_id),
        ("trajectory_id", trajectory_id),
    ):
        _validate_identifier(value, name=name)
    if (
        not isinstance(replicate_index, int)
        or isinstance(replicate_index, bool)
        or replicate_index < 0
    ):
        raise SeedContractError("replicate_index must be a nonnegative integer.")
    return (
        "urn:eidosoma:seed-namespace:"
        f"{experiment_id}:{specification_id}:{trajectory_id}:r{replicate_index}"
    )


@dataclass(frozen=True, slots=True)
class SeedRequest:
    """Complete caller-supplied identity used to derive all stream records."""

    experiment_id: str
    specification_id: str
    trajectory_id: str
    replicate_index: int
    engine_id: str
    root_seed_hex: str
    coupling_policy: CouplingPolicy
    coupling_reason: str | None
    stream_namespaces: Mapping[StreamPurpose, str]

    def __post_init__(self) -> None:
        for name in (
            "experiment_id",
            "specification_id",
            "trajectory_id",
            "engine_id",
        ):
            _validate_identifier(getattr(self, name), name=name)
        if (
            not isinstance(self.replicate_index, int)
            or isinstance(self.replicate_index, bool)
            or self.replicate_index < 0
        ):
            raise SeedContractError("replicate_index must be a nonnegative integer.")
        if not isinstance(self.root_seed_hex, str) or not _SHA256_PATTERN.fullmatch(
            self.root_seed_hex
        ):
            raise SeedContractError(
                "root_seed_hex must be exactly 32 bytes encoded as 64 lowercase hex characters."
            )
        if not isinstance(self.coupling_policy, CouplingPolicy):
            raise SeedContractError(
                "coupling_policy must be an explicit CouplingPolicy."
            )
        if not isinstance(self.stream_namespaces, Mapping):
            raise SeedContractError("stream_namespaces must be an explicit mapping.")
        namespaces = dict(self.stream_namespaces)
        if set(namespaces) != set(CANONICAL_STREAM_PURPOSES):
            missing = sorted(
                purpose.value
                for purpose in set(CANONICAL_STREAM_PURPOSES) - set(namespaces)
            )
            extra = sorted(
                str(purpose)
                for purpose in set(namespaces) - set(CANONICAL_STREAM_PURPOSES)
            )
            raise SeedContractError(
                f"stream_namespaces must name all canonical purposes; missing={missing}, extra={extra}."
            )
        for purpose, namespace in namespaces.items():
            if not isinstance(purpose, StreamPurpose):
                raise SeedContractError(
                    "stream_namespaces keys must be StreamPurpose values."
                )
            _validate_identifier(namespace, name=f"namespace[{purpose.value}]")

        isolated = isolated_stream_namespace(
            experiment_id=self.experiment_id,
            specification_id=self.specification_id,
            trajectory_id=self.trajectory_id,
            replicate_index=self.replicate_index,
        )
        if self.coupling_policy is CouplingPolicy.TRAJECTORY_ISOLATED:
            if self.coupling_reason is not None:
                raise SeedContractError(
                    "trajectory_isolated requires coupling_reason=null."
                )
            if set(namespaces.values()) != {isolated}:
                raise SeedContractError(
                    "trajectory_isolated requires every explicit stream namespace to equal "
                    "the canonical trajectory namespace."
                )
        else:
            if (
                not isinstance(self.coupling_reason, str)
                or not self.coupling_reason.strip()
            ):
                raise SeedContractError(
                    "explicit_common_random_numbers requires a nonempty coupling_reason."
                )
        object.__setattr__(
            self,
            "stream_namespaces",
            MappingProxyType(namespaces),
        )


@dataclass(frozen=True, slots=True)
class DerivedStream:
    """One immutable domain-separated generator identity."""

    purpose: StreamPurpose
    namespace: str
    stream_id: str
    derivation_context_sha256: str
    seed_material_hex: str
    seed_integer_hex: str
    bit_generator: str
    numpy_version: str
    initial_state_sha256: str

    def generator(self) -> np.random.Generator:
        """Create a fresh generator at the canonical initial state."""

        seed_integer = int(self.seed_integer_hex, 16)
        generator = np.random.Generator(np.random.PCG64DXSM(seed_integer))
        if _generator_state_sha256(generator) != self.initial_state_sha256:
            raise SeedContractError(
                f"Generator state mismatch for canonical stream {self.purpose.value}."
            )
        return generator

    def to_payload(self) -> dict[str, str]:
        return {
            "purpose": self.purpose.value,
            "namespace": self.namespace,
            "streamId": self.stream_id,
            "derivationContextSha256": self.derivation_context_sha256,
            "seedMaterialHex": self.seed_material_hex,
            "seedIntegerHex": self.seed_integer_hex,
            "bitGenerator": self.bit_generator,
            "numpyVersion": self.numpy_version,
            "initialStateSha256": self.initial_state_sha256,
        }


def _jsonable_generator_state(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _jsonable_generator_state(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_generator_state(item) for item in value]
    return value


def _generator_state_sha256(generator: np.random.Generator) -> str:
    return sha256_hex(
        canonical_json_bytes(_jsonable_generator_state(generator.bit_generator.state))
    )


def _derive_stream(request: SeedRequest, purpose: StreamPurpose) -> DerivedStream:
    namespace = request.stream_namespaces[purpose]
    context = {
        "seedSchemaVersion": SEED_SCHEMA_VERSION,
        "derivationAlgorithm": SEED_DERIVATION_ALGORITHM,
        "experimentId": request.experiment_id,
        "replicateIndex": request.replicate_index,
        "streamPurpose": purpose.value,
        "streamNamespace": namespace,
    }
    context_bytes = canonical_json_bytes(context)
    root_seed = bytes.fromhex(request.root_seed_hex)
    preimage = (
        DERIVATION_DOMAIN
        + len(root_seed).to_bytes(4, "big")
        + root_seed
        + len(context_bytes).to_bytes(8, "big")
        + context_bytes
    )
    seed_material = hashlib.sha256(preimage).digest()
    seed_hex = seed_material.hex()
    seed_integer_hex = "0x" + seed_hex
    stream_id = f"urn:eidosoma:rng:{SEED_SCHEMA_VERSION}:{purpose.value}:{seed_hex}"
    generator = np.random.Generator(
        np.random.PCG64DXSM(int.from_bytes(seed_material, "big", signed=False))
    )
    return DerivedStream(
        purpose=purpose,
        namespace=namespace,
        stream_id=stream_id,
        derivation_context_sha256=sha256_hex(context_bytes),
        seed_material_hex=seed_hex,
        seed_integer_hex=seed_integer_hex,
        bit_generator=BIT_GENERATOR_NAME,
        numpy_version=np.__version__,
        initial_state_sha256=_generator_state_sha256(generator),
    )


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """The complete nine-stream seed payload plus generator constructors."""

    request: SeedRequest
    streams: Mapping[StreamPurpose, DerivedStream]

    def __post_init__(self) -> None:
        if not isinstance(self.streams, Mapping):
            raise SeedContractError("SeedBundle streams must be an explicit mapping.")
        streams = dict(self.streams)
        if set(streams) != set(CANONICAL_STREAM_PURPOSES):
            raise SeedContractError(
                "SeedBundle does not contain every canonical stream."
            )
        ids = [stream.stream_id for stream in streams.values()]
        material = [stream.seed_material_hex for stream in streams.values()]
        if len(set(ids)) != len(ids) or len(set(material)) != len(material):
            raise SeedContractError(
                "Domain-separated streams must have unique identities."
            )
        object.__setattr__(self, "streams", MappingProxyType(streams))

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seedSchemaVersion": SEED_SCHEMA_VERSION,
            "derivationAlgorithm": SEED_DERIVATION_ALGORITHM,
            "canonicalJsonVersion": CANONICAL_JSON_VERSION,
            "rootSeedHex": self.request.root_seed_hex,
            "rootSeedSha256": sha256_hex(bytes.fromhex(self.request.root_seed_hex)),
            "experimentId": self.request.experiment_id,
            "specificationId": self.request.specification_id,
            "trajectoryId": self.request.trajectory_id,
            "replicateIndex": self.request.replicate_index,
            "engineId": self.request.engine_id,
            "couplingPolicy": self.request.coupling_policy.value,
            "couplingReason": self.request.coupling_reason,
            "streams": {
                purpose.value: self.streams[purpose].to_payload()
                for purpose in CANONICAL_STREAM_PURPOSES
            },
            "uncertaintyBoundary": {
                "authorInitialStateRngStream": "UNRESOLVED::E01-A020",
                "legacyMatlabRngIdentity": "UNRESOLVED::LEGACY_MATLAB_RNG_ALGORITHM_AND_GLOBAL_STATE_ORDER",
                "authorCodeIdentity": "UNAVAILABLE::NO_AUTHOR_CODE_RELEASE_FOUND",
                "interpretation": (
                    "Canonical reconstruction engineering only; not evidence of author or "
                    "legacy MATLAB random-number semantics."
                ),
            },
        }
        return payload

    def fresh_generators(self) -> dict[StreamPurpose, np.random.Generator]:
        """Create pairwise-distinct fresh generators for every purpose."""

        generators = {
            purpose: self.streams[purpose].generator()
            for purpose in CANONICAL_STREAM_PURPOSES
        }
        if len({id(generator) for generator in generators.values()}) != len(generators):
            raise SeedContractError("Fresh stream generators are not distinct objects.")
        return generators

    def independent_engine_streams(
        self,
        generators: dict[StreamPurpose, np.random.Generator],
    ) -> RNGStreams:
        """Bind the six S05 engine inputs without folding auxiliary streams into them."""

        if set(generators) != set(CANONICAL_STREAM_PURPOSES):
            raise SeedContractError(
                "Generator mapping must contain all canonical streams."
            )

        def item(purpose: StreamPurpose) -> RNGInput:
            return RNGInput(
                self.streams[purpose].stream_id,
                generators[purpose],
            )

        return RNGStreams(
            catalytic_matrix=item(StreamPurpose.CATALYTIC_MATRIX),
            initialization=item(StreamPurpose.INITIAL_STATE),
            events=item(StreamPurpose.EVENT),
            waiting_time=item(StreamPurpose.WAITING_TIME),
            fission=item(StreamPurpose.FISSION),
            daughter=item(StreamPurpose.DAUGHTER_SELECTION),
        )


def derive_seed_bundle(request: SeedRequest) -> SeedBundle:
    """Derive every required stream under the frozen domain-separation algorithm."""

    streams = {
        purpose: _derive_stream(request, purpose)
        for purpose in CANONICAL_STREAM_PURPOSES
    }
    return SeedBundle(request=request, streams=streams)


def seed_request_from_payload(payload: dict[str, Any]) -> SeedRequest:
    """Reconstruct and verify a seed request from a serialized seed payload."""

    required = {
        "seedSchemaVersion",
        "derivationAlgorithm",
        "canonicalJsonVersion",
        "rootSeedHex",
        "rootSeedSha256",
        "experimentId",
        "specificationId",
        "trajectoryId",
        "replicateIndex",
        "engineId",
        "couplingPolicy",
        "couplingReason",
        "streams",
        "uncertaintyBoundary",
    }
    if set(payload) != required:
        raise SeedContractError(
            f"Seed payload fields mismatch; missing={sorted(required - set(payload))}, "
            f"extra={sorted(set(payload) - required)}."
        )
    if payload["seedSchemaVersion"] != SEED_SCHEMA_VERSION:
        raise SeedContractError("Unsupported seed schema version.")
    if payload["derivationAlgorithm"] != SEED_DERIVATION_ALGORITHM:
        raise SeedContractError("Unsupported seed derivation algorithm.")
    if payload["canonicalJsonVersion"] != CANONICAL_JSON_VERSION:
        raise SeedContractError("Unsupported canonical JSON version.")
    stream_payloads = payload["streams"]
    if not isinstance(stream_payloads, dict):
        raise SeedContractError("streams must be an object.")
    namespaces: dict[StreamPurpose, str] = {}
    for purpose in CANONICAL_STREAM_PURPOSES:
        record = stream_payloads.get(purpose.value)
        if not isinstance(record, dict):
            raise SeedContractError(f"Missing stream record {purpose.value}.")
        namespaces[purpose] = record.get("namespace")
    request = SeedRequest(
        experiment_id=payload["experimentId"],
        specification_id=payload["specificationId"],
        trajectory_id=payload["trajectoryId"],
        replicate_index=payload["replicateIndex"],
        engine_id=payload["engineId"],
        root_seed_hex=payload["rootSeedHex"],
        coupling_policy=CouplingPolicy(payload["couplingPolicy"]),
        coupling_reason=payload["couplingReason"],
        stream_namespaces=namespaces,
    )
    regenerated = derive_seed_bundle(request)
    if regenerated.to_payload() != payload:
        raise SeedContractError("Serialized stream identities do not match derivation.")
    return request
