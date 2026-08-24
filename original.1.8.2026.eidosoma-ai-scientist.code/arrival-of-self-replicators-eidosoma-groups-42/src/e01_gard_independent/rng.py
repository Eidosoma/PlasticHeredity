"""Separated, caller-owned NumPy random generators for the S05 engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .specification import SpecificationError


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def generator_state_sha256(generator: np.random.Generator) -> str:
    """Return a stable digest of the current bit-generator state."""

    encoded = json.dumps(
        _jsonable(generator.bit_generator.state),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class RNGInput:
    """One explicitly identified modern NumPy generator."""

    stream_id: str
    generator: np.random.Generator

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, str) or not self.stream_id:
            raise SpecificationError("Every RNG input needs a nonempty stream_id.")
        if not isinstance(self.generator, np.random.Generator):
            raise SpecificationError(
                "Every RNG input requires an explicit numpy.random.Generator."
            )


@dataclass(frozen=True, slots=True)
class RNGStreams:
    """Six distinct generator inputs; no seed-derivation policy is implied."""

    catalytic_matrix: RNGInput
    initialization: RNGInput
    events: RNGInput
    waiting_time: RNGInput
    fission: RNGInput
    daughter: RNGInput

    def __post_init__(self) -> None:
        streams = (
            self.catalytic_matrix,
            self.initialization,
            self.events,
            self.waiting_time,
            self.fission,
            self.daughter,
        )
        if not all(isinstance(stream, RNGInput) for stream in streams):
            raise SpecificationError("All RNGStreams fields must be RNGInput objects.")
        ids = [stream.stream_id for stream in streams]
        if len(set(ids)) != len(ids):
            raise SpecificationError("RNG stream IDs must be distinct.")
        generators = [id(stream.generator) for stream in streams]
        if len(set(generators)) != len(generators):
            raise SpecificationError(
                "RNG generators must be distinct objects; shared streams are prohibited."
            )

    def descriptions(self) -> tuple[dict[str, str], ...]:
        """Describe algorithms and IDs without defining the later S06 seed schema."""

        named = (
            ("catalytic_matrix", self.catalytic_matrix),
            ("initialization", self.initialization),
            ("events", self.events),
            ("waiting_time", self.waiting_time),
            ("fission", self.fission),
            ("daughter", self.daughter),
        )
        return tuple(
            {
                "purpose": purpose,
                "streamId": stream.stream_id,
                "bitGenerator": type(stream.generator.bit_generator).__name__,
            }
            for purpose, stream in named
        )
