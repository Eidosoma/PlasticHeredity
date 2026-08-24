"""Frozen v2 contract, hashing, and atomic artifact helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_ARTIFACTS = PACKAGE_ROOT / "artifacts"
DEFAULT_UPSTREAM = WORKSPACE_ROOT / "reviewer_ca_lineage_renewal_replication" / "artifacts"

SCHEMA_VERSION = "reviewer-ca-lineage-renewal-replication-v2"
NAMESPACE = "reviewer-ca-lineage-renewal-stage3r-corrected-v2"
PAIRING_NAMESPACE = "reviewer-ca-lineage-renewal-corrected-pairing-v2"

FIXED_CONFIGURATION: dict[str, Any] = {
    "configuration_id": "motif_energy512-w32-s025-d32-tanh9",
    "family": "motif_energy512",
    "write_window": 32,
    "strength": 0.25,
    "read_duration": 32,
    "transfer_function": "strength*tanh(max(energy_advantage,0)/9)",
}

CONDITIONS = [
    "intact",
    "zero_every_boundary",
    "shuffle_every_boundary",
    "read_disabled",
    "founder_write_disabled",
    "no_rewrite",
    "ablate_after_g2",
    "rescue_same_enter_g4",
    "rescue_opposite_enter_g4",
    "opposite_founder",
    "carrier_corruption_1",
]

SECONDARY_CONDITIONS = (
    "intact",
    "no_rewrite",
    "read_disabled",
    "ablate_after_g2",
    "rescue_same_enter_g4",
    "rescue_opposite_enter_g4",
)

PROFILE: dict[str, Any] = {
    "quarantine_pairs": 2,
    "confirmation_pairs": 92,
    "replicates": 64,
    "generations": 16,
    "generation_checkpoints": [1, 2, 4, 8, 16],
    "bootstrap_resamples": 10_000,
    "decoder_splits": 4,
    "default_workers": 8,
}

CONTRACT: dict[str, Any] = {
    "rule": 31649,
    "notation": "B13456/S0578",
    "height": 16,
    "width": 16,
    "generation_sweeps": 64,
    "read_sweeps": 32,
    "write_window": [49, 64],
    "observation_window": [57, 64],
    "state_bit_order": "row-major least-significant-bit first",
    "motif_bit_order": "row-major least-significant-bit first",
    "texture2x2_bit_order": "TL bit0, TR bit1, BL bit2, BR bit3",
    "visible_reset": "launch-specific donor initial_state_hex before every generation",
    "sweep_order": ["CA step", "reader", "process noise", "write/observe"],
    "process_noise": 0.002,
    "reader_scale": 9.0,
    "repair_kind": "universal_scalar_gain",
    "repair_gain": 0.5,
    "stale_retention": 0.5,
    "carrier_corruption": 0.01,
    "jeffreys_alpha": 0.5,
    "energy_clip": 4.0,
    "assignment_similarity": 0.90,
    "assignment_margin": 0.05,
    "primary_crossover_generation4": 0.20,
    "primary_crossover_generation8": 0.15,
    "durable_crossover_generation16": 0.10,
    "control_advantage": 0.10,
    "loss_fraction": 0.70,
    "rescue_fraction": 0.70,
    "survival_gate": 0.90,
    "corruption_crossover": 0.10,
    "confirmation_alpha": 0.0125,
    "decoder_mean_gate": 0.65,
    "decoder_lower_gate": 0.55,
    "decoder_null_ceiling": 0.55,
    "decoder_advantage": 0.10,
    "decoder_splits": 4,
    "density_caliper": 0.02,
    "independent_unit": "matched founder pair",
    "missing_policy": "dead and unresolved futures remain in denominators",
    "runtime_repair_access": "raw daughter carrier and fixed gain only",
    "runtime_label_access": False,
    "runtime_parent_access": False,
    "runtime_target_access": False,
    "claim_boundary": (
        "synthetic CA lineage memory only; no metabolism, agency, "
        "biological-life, or extra-automaton memory claim"
    ),
}


def canonical_bytes(value: Any) -> bytes:
    assert_finite_json(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def assert_finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, (float,)):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite number at {path}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_json(child, f"{path}[{index}]")


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
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def hash_order(items: Iterable[str], namespace: str) -> list[str]:
    return sorted(
        items,
        key=lambda item: (
            hashlib.sha256(f"{namespace}\x1f{item}".encode()).digest(), item
        ),
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def atomic_write_bytes(path: Path, value: bytes) -> None:
    _atomic_write(path, value)


def atomic_write_json(path: Path, value: Any) -> None:
    assert_finite_json(value)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _atomic_write(path, encoded.encode("utf-8"))


def atomic_write_text(path: Path, value: str) -> None:
    _atomic_write(path, value.encode("utf-8"))


def implementation_manifest() -> dict[str, str]:
    paths = [*PACKAGE_ROOT.glob("*.py"), *PACKAGE_ROOT.glob("*.md")]
    paths += list((PACKAGE_ROOT / "tests").glob("*.py"))
    return {
        str(path.relative_to(PACKAGE_ROOT)): sha256_file(path)
        for path in sorted(paths)
    }


def registration_digest(registration: Mapping[str, Any]) -> str:
    unsealed = {key: value for key, value in registration.items() if key != "design_digest"}
    return sha256_json(unsealed)


def seal_registration(registration: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(registration)
    sealed["design_digest"] = registration_digest(sealed)
    return sealed


def verify_registration(registration: Mapping[str, Any]) -> None:
    if registration.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("registration schema mismatch")
    if registration.get("design_digest") != registration_digest(registration):
        raise ValueError("registration digest mismatch")


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
        raise ValueError(f"checkpoint schema mismatch in {path}")
    if envelope.get("binding_sha256") != binding:
        raise ValueError(f"checkpoint design mismatch in {path}")
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != sha256_json(payload):
        raise ValueError(f"checkpoint checksum mismatch in {path}")
    return payload
