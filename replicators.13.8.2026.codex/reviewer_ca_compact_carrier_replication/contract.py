"""Frozen contract and tamper-evident artifact helpers.

The source Stage-4 campaign is used only to specify a hypothesis.  This
package never imports or executes code from that tree.
"""

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
DEFAULT_LOCAL_INPUT = WORKSPACE_ROOT / "reviewer_ca_lineage_renewal_replication_v2" / "artifacts"
DEFAULT_SOURCE_ROOT = (
    WORKSPACE_ROOT.parent
    / "NewIdeas"
    / "preprints"
    / "ingressing-minds-v-ruliad-paper-ideas"
    / "codex.reconstructionsAndStressTesting"
)

SCHEMA_VERSION = "reviewer-ca-compact-carrier-replication-v1"
NAMESPACE = "reviewer-ca-compact-carrier-cleanroom-v1"
ACQUISITION_NAMESPACE = "reviewer-ca-fresh-donor-bank-2026-08-23-v1"
PAIRING_NAMESPACE = "reviewer-ca-fresh-pairing-2026-08-23-v1"

CANDIDATE_IDS = (
    "identity-r512-f32",
    "pca-r008-q04",
    "walsh-r016-q04",
)
ENVIRONMENTS = ("ordinary", "moderate_joint")
CONDITIONS = (
    "intact",
    "zero_every_boundary",
    "decoded_shuffle_every_boundary",
    "latent_shuffle_every_boundary",
    "read_disabled",
    "founder_write_disabled",
    "no_rewrite",
    "ablate_after_g2",
    "rescue_same_enter_g4",
    "rescue_opposite_enter_g4",
    "opposite_founder",
    "latent_corruption_1",
)
CHECKPOINT_GENERATIONS = (1, 2, 4, 8, 16)

PROFILE: dict[str, Any] = {
    "acquisition_candidates_per_launch": 512,
    "acquisition_launches": 4,
    "engineering_pairs": 4,
    "confirmation_pairs": 128,
    "audit_reserve_pairs": 32,
    "minimum_fresh_pairs": 164,
    "confirmation_replicates": 64,
    "confirmation_generations": 16,
    "bootstrap_resamples": 10_000,
    "default_workers": 20,
    "wall_budget_hours": 8.0,
    "reserve_minutes": 30.0,
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
    "visible_reset": "launch-specific bitwise-identical board before every generation",
    "sweep_order": ["CA step", "reader", "process noise", "write/observe"],
    "boundary_order": [
        "lineage intervention",
        "environment damage",
        "decode",
        "decoded-address intervention",
        "reader",
        "daughter write",
        "gain 0.5",
        "encode",
    ],
    "ordinary_process_noise": 0.002,
    "moderate_process_noise": 0.004,
    "moderate_payload_erasure": 0.10,
    "moderate_payload_sign_corruption": 0.05,
    "reader_strength": 0.25,
    "reader_scale": 9.0,
    "repair_gain": 0.5,
    "stale_retention": 0.5,
    "registered_latent_corruption": 0.01,
    "jeffreys_alpha": 0.5,
    "energy_clip": 4.0,
    "acquisition_similarity": 0.95,
    "acquisition_margin": 0.05,
    "density_caliper": 0.02,
    "primary_crossover_generation4": 0.20,
    "primary_crossover_generation8": 0.15,
    "durable_crossover_generation16": 0.10,
    "control_advantage": 0.10,
    "loss_fraction": 0.70,
    "rescue_fraction": 0.70,
    "survival_gate": 0.90,
    "corruption_crossover": 0.10,
    "confirmation_alpha_per_codec": 0.005,
    "independent_unit": "fresh matched founder pair",
    "missing_policy": "dead and unresolved futures remain in denominators",
    "runtime_label_access": False,
    "runtime_parent_access": False,
    "runtime_target_access": False,
    "source_outcomes_evidential_role": "none; hypothesis specification only",
    "claim_boundary": (
        "synthetic cellular-automaton lineage memory only; no cross-substrate, "
        "metabolism, agency, biological-life, or full-Ruliad claim"
    ),
}


def assert_finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite number at {path}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_json(child, f"{path}[{index}]")


def canonical_bytes(value: Any) -> bytes:
    assert_finite_json(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
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
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _atomic_write(path, payload.encode("utf-8"))


def atomic_write_text(path: Path, value: str) -> None:
    _atomic_write(path, value.encode("utf-8"))


def implementation_manifest() -> dict[str, str]:
    files = [*PACKAGE_ROOT.glob("*.py"), *PACKAGE_ROOT.glob("*.md")]
    files += list((PACKAGE_ROOT / "tests").glob("*.py"))
    return {
        str(path.relative_to(PACKAGE_ROOT)): sha256_file(path)
        for path in sorted(files)
    }


def registration_digest(registration: Mapping[str, Any]) -> str:
    body = {key: value for key, value in registration.items() if key != "design_digest"}
    return sha256_json(body)


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
