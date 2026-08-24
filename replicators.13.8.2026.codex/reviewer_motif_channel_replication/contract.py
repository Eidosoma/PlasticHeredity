"""Frozen contracts, hashing, seeds, and atomic artifact helpers.

This module has no knowledge of the historical source tree.  Scientific
commands operate only on the immutable local snapshot rooted at ``artifacts``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = PACKAGE_ROOT / "artifacts"

SCHEMA_VERSION = "reviewer-motif-channel-replication-v1"
STAGE1_NAMESPACE = "reviewer-motif-channel-stage1-fresh-v1"
STAGE2_NAMESPACE = "reviewer-motif-channel-stage2-fresh-v1"
PAIRING_NAMESPACE = "reviewer-motif-channel-fresh-donor-pairing-v1"

PAIR_RE = re.compile(
    r"narrow-[0-9]{4}-(life-31649-[0-3]-[0-9]+)-"
    r"(life-31649-[0-3]-[0-9]+)"
)


@dataclass(frozen=True)
class ReaderConfiguration:
    family: str
    write_window: int
    strength: float
    read_duration: int

    @property
    def configuration_id(self) -> str:
        strength = int(round(self.strength * 100))
        return (
            f"{self.family}-w{self.write_window:02d}-"
            f"s{strength:03d}-d{self.read_duration:02d}"
        )


FIXED_PRIMARY = ReaderConfiguration(
    family="motif_energy512",
    write_window=32,
    strength=0.25,
    read_duration=32,
)

STAGE1_PROFILE: dict[str, Any] = {
    "calibration_pairs": 64,
    "discovery_pairs": 48,
    "validation_pairs": 64,
    "screen_replicates": 16,
    "validation_replicates": 64,
    "write_windows": [16, 32],
    "strengths": [0.25, 0.50, 0.75, 1.00],
    "read_durations": [8, 16, 32, 64],
    "nominees_per_family": 2,
    "bootstrap_resamples": 10_000,
}

STAGE1_CONTRACT: dict[str, Any] = {
    "rule": 31649,
    "notation": "B13456/S0578",
    "height": 16,
    "width": 16,
    "horizon": 64,
    "checkpoints": [8, 16, 32, 64],
    "observation_window": 8,
    "jeffreys_alpha": 0.5,
    "energy_clip": 4.0,
    "assignment_similarity": 0.90,
    "assignment_margin": 0.05,
    "process_noise": 0.002,
    "carrier_corruption": 0.01,
    "crossover_gate": 0.15,
    "robust_crossover_gate": 0.10,
    "control_advantage_gate": 0.10,
    "survival_gate": 0.90,
    "familywise_alpha": 0.05 / 4.0,
    "primary_observer": "trailing-eight-sweep accumulated live 2x2 texture",
    "non_gating_diagnostics": [
        "terminal live 2x2 texture",
        "occupancy",
        "8-connected toroidal component geometry",
        "nearest-lag autocorrelation",
        "low-frequency spatial power",
    ],
    "missing_policy": "dead and unresolved futures remain in denominators",
    "claim_boundary": "one-generation controllability; not plastic heredity",
}

STAGE1_CONDITIONS = [
    "intact",
    "zero",
    "read_disabled",
    "shuffle",
    "opposite_history",
    "unrelated_same_form",
    "process_noise",
    "carrier_sign_corruption",
    "spatial_latch_benchmark",
    "incomplete_visible64_reset",
]

STAGE2_PROFILE: dict[str, Any] = {
    "audit_pairs": 32,
    "pairs": 96,
    "replicates": 64,
    "bootstrap_resamples": 10_000,
    "primary_environments": [
        "native",
        "launch0",
        "launch1",
        "launch2",
        "launch3",
        "native_translate_3_5",
        "native_rot90",
        "native_reflect_x",
    ],
    "core_conditions": [
        "intact",
        "zero",
        "read_disabled",
        "shuffle",
        "matched_random",
        "opposite_history",
        "unrelated_pair",
        "midpoint",
    ],
    "stress_environments": [
        "random_density_10",
        "random_density_30",
        "random_density_50",
    ],
    "stress_conditions": [
        "intact",
        "zero",
        "opposite_history",
        "unrelated_pair",
    ],
    "dose_contrasts": [0.0, 0.25, 0.50, 0.75, 1.0],
}

STAGE2_CONTRACT: dict[str, Any] = {
    **STAGE1_CONTRACT,
    "gate_checkpoint": 64,
    "primary_crossover": 0.15,
    "stress_crossover": 0.10,
    "control_advantage": 0.10,
    "midpoint_tolerance": 0.02,
    "monotonic_tolerance": 0.03,
    "dose_rank_gate": 0.90,
    "dose_slope_gate": 0.10,
    "unrelated_retention": 0.70,
    "writer_accuracy_gate": 0.80,
    "symmetry_tolerance": 1e-6,
    "familywise_alpha": 0.05 / 8.0,
    "claim_boundary": "one-generation reusable form channel; not plastic heredity",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_seed(*parts: object) -> int:
    """Return a worker/order-independent 128-bit seed."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def hash_order(items: Iterable[str], namespace: str) -> list[str]:
    return sorted(
        items,
        key=lambda item: (
            hashlib.sha256(f"{namespace}\x1f{item}".encode()).digest(),
            item,
        ),
    )


def parse_historical_pair_id(pair_id: str) -> tuple[str, str]:
    match = PAIR_RE.fullmatch(pair_id)
    if match is None:
        raise ValueError(f"invalid historical pair id: {pair_id}")
    return match.group(1), match.group(2)


def reader_configurations() -> list[ReaderConfiguration]:
    return [
        ReaderConfiguration(family, window, strength, duration)
        for family in ("contextual256", "motif_energy512")
        for window in STAGE1_PROFILE["write_windows"]
        for strength in STAGE1_PROFILE["strengths"]
        for duration in STAGE1_PROFILE["read_durations"]
    ]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _atomic_write(path, encoded.encode("utf-8"))


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, value.encode("utf-8"))


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def checkpoint_envelope(binding: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "binding_sha256": binding,
        "payload_sha256": sha256_json(body),
        "payload": body,
    }


def write_checkpoint(path: Path, binding: str, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, checkpoint_envelope(binding, payload))


def read_checkpoint(path: Path, binding: str) -> dict[str, Any]:
    envelope = load_json(path)
    if envelope.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema mismatch in {path}")
    if envelope.get("binding_sha256") != binding:
        raise ValueError(f"design binding mismatch in {path}")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint payload in {path}")
    if envelope.get("payload_sha256") != sha256_json(payload):
        raise ValueError(f"payload checksum mismatch in {path}")
    return payload


def implementation_manifest() -> dict[str, str]:
    paths = sorted(PACKAGE_ROOT.glob("*.py"))
    return {path.name: sha256_file(path) for path in paths}


def registration_digest(registration: Mapping[str, Any]) -> str:
    unsealed = {key: value for key, value in registration.items() if key != "design_digest"}
    return sha256_json(unsealed)


def seal_registration(registration: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(registration)
    sealed["design_digest"] = registration_digest(sealed)
    return sealed


def verify_registration(registration: Mapping[str, Any]) -> None:
    expected = registration_digest(registration)
    if registration.get("design_digest") != expected:
        raise ValueError("registration digest mismatch")


def as_jsonable_configuration(config: ReaderConfiguration) -> dict[str, Any]:
    return {**asdict(config), "configuration_id": config.configuration_id}
