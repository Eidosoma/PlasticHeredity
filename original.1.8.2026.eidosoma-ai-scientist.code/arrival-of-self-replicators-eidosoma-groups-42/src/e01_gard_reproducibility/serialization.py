"""Canonical, checksum-protected JSON serialization for E01 S06 artifacts."""

from __future__ import annotations

import json
import math
import re
import struct
from collections.abc import Callable
from typing import Any

from .seed import (
    CANONICAL_JSON_VERSION,
    SeedContractError,
    canonical_json_bytes,
    sha256_hex,
)

_FLOAT_HEX_PATTERN = re.compile(r"^-?0x[0-9a-f]+(?:\.[0-9a-f]+)?p[+-][0-9]+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SerializationError(ValueError):
    """Canonical JSON, checksum, or lossless-float validation failed."""


def float_to_hex(value: float) -> str:
    """Encode a finite IEEE-754 binary64 value without information loss."""

    converted = float(value)
    if not math.isfinite(converted):
        raise SerializationError("Only finite binary64 values can be serialized.")
    return converted.hex()


def float_from_hex(value: str) -> float:
    """Decode and canonicalize one lossless binary64 hexadecimal string."""

    if not isinstance(value, str) or not _FLOAT_HEX_PATTERN.fullmatch(value):
        raise SerializationError(f"Invalid binary64 hexadecimal value: {value!r}.")
    converted = float.fromhex(value)
    if not math.isfinite(converted) or converted.hex() != value:
        raise SerializationError(f"Noncanonical binary64 hexadecimal value: {value!r}.")
    return converted


def float_bits_hex(value: float) -> str:
    """Return the exact big-endian IEEE-754 bit pattern for audit output."""

    return struct.pack(">d", float(value)).hex()


def make_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a canonical payload with an unambiguous SHA-256 checksum."""

    if not isinstance(payload, dict):
        raise SerializationError("Envelope payload must be an object.")
    try:
        encoded = canonical_json_bytes(payload)
    except SeedContractError as exc:
        raise SerializationError(str(exc)) from exc
    return {
        "serializationVersion": CANONICAL_JSON_VERSION,
        "payloadSha256": sha256_hex(encoded),
        "payload": payload,
    }


def serialize_envelope(envelope: dict[str, Any]) -> bytes:
    """Return byte-stable canonical JSON with no trailing newline."""

    verify_envelope(envelope)
    try:
        return canonical_json_bytes(envelope)
    except SeedContractError as exc:
        raise SerializationError(str(exc)) from exc


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SerializationError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def _reject_json_float(_: str) -> float:
    raise SerializationError(
        "JSON floating-point literals are forbidden; use canonical binary64 hex strings."
    )


def _reject_json_constant(value: str) -> None:
    raise SerializationError(
        f"Nonstandard JSON numeric constant is forbidden: {value}."
    )


def deserialize_envelope(
    data: bytes,
    *,
    require_canonical: bool,
) -> dict[str, Any]:
    """Decode JSON without duplicate keys or numeric floats and verify its checksum."""

    if not isinstance(data, bytes):
        raise SerializationError("Serialized input must be bytes.")
    try:
        decoded = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SerializationError("Input is not valid canonical UTF-8 JSON.") from exc
    if not isinstance(decoded, dict):
        raise SerializationError("Top-level serialized value must be an object.")
    verify_envelope(decoded)
    if require_canonical and serialize_envelope(decoded) != data:
        raise SerializationError("Serialized bytes are valid but not canonical.")
    return decoded


def verify_envelope(envelope: dict[str, Any]) -> None:
    """Check exact envelope fields, version, and payload digest."""

    required = {"serializationVersion", "payloadSha256", "payload"}
    if set(envelope) != required:
        raise SerializationError(
            f"Envelope fields mismatch; missing={sorted(required - set(envelope))}, "
            f"extra={sorted(set(envelope) - required)}."
        )
    if envelope["serializationVersion"] != CANONICAL_JSON_VERSION:
        raise SerializationError("Unsupported canonical serialization version.")
    checksum = envelope["payloadSha256"]
    if not isinstance(checksum, str) or not _SHA256_PATTERN.fullmatch(checksum):
        raise SerializationError("payloadSha256 must be lowercase SHA-256 hex.")
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise SerializationError("Envelope payload must be an object.")
    try:
        actual = sha256_hex(canonical_json_bytes(payload))
    except SeedContractError as exc:
        raise SerializationError(str(exc)) from exc
    if actual != checksum:
        raise SerializationError(
            f"Payload checksum mismatch: expected {checksum}, calculated {actual}."
        )


def validate_json_schema(
    instance: dict[str, Any],
    schema: dict[str, Any],
    *,
    validator_factory: Callable[[dict[str, Any]], Any],
) -> None:
    """Validate with an explicitly supplied Draft 2020-12 validator factory."""

    validator = validator_factory(schema)
    errors = sorted(
        validator.iter_errors(instance),
        key=lambda item: tuple(str(part) for part in item.path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '$'}: {error.message}"
            for error in errors[:10]
        )
        raise SerializationError(f"JSON Schema conformance failed: {details}")
