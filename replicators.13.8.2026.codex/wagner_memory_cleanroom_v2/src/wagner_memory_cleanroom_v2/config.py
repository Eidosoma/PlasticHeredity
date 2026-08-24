from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT_DIR / "protocols" / "wagner-memory-v2.json"


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Registration:
    protocol: dict[str, Any]
    protocol_path: Path
    protocol_digest: str
    profile_name: str
    profile: dict[str, Any]

    @property
    def scientific(self) -> bool:
        return bool(self.profile["scientific"])

    @property
    def engine(self) -> dict[str, Any]:
        return self.protocol["engine"]

    @property
    def operations(self) -> dict[str, Any]:
        return self.protocol["operations"]


def load_registration(profile: str, protocol_path: Path | None = None) -> Registration:
    path = (protocol_path or DEFAULT_PROTOCOL).resolve()
    protocol = json.loads(path.read_text())
    if protocol.get("format") != "wagner-memory-cleanroom-protocol-v2":
        raise ValueError(f"unsupported protocol format in {path}")
    try:
        selected = dict(protocol["profiles"][profile])
    except KeyError as exc:
        raise ValueError(f"unknown profile: {profile}") from exc
    return Registration(protocol, path, file_digest(path), profile, selected)


def load_run_registration(run_dir: Path) -> Registration:
    run = run_dir.resolve()
    registration_path = run / "registration.json"
    payload = json.loads(registration_path.read_text())
    if payload.get("format") != "wagner-memory-registration-v2":
        raise ValueError(f"unsupported run registration in {registration_path}")
    frozen_relative = Path(str(payload["frozen_protocol_path"]))
    frozen_path = (run / frozen_relative).resolve()
    if not frozen_path.is_relative_to(run):
        raise ValueError("frozen protocol escapes the run directory")
    if file_digest(frozen_path) != str(payload["protocol_digest"]):
        raise ValueError("frozen run protocol digest mismatch")
    protocol = json.loads(frozen_path.read_text())
    if canonical_json(protocol) != canonical_json(payload["protocol"]):
        raise ValueError("embedded and frozen run protocols differ")
    profile_name = str(payload["profile"])
    profile = dict(protocol["profiles"][profile_name])
    if canonical_json(profile) != canonical_json(payload["profile_values"]):
        raise ValueError("registered profile values differ from the frozen protocol")
    scientific = bool(payload["scientific"])
    if scientific != bool(profile["scientific"]):
        raise ValueError("registered scientific flag differs from profile")
    return Registration(
        protocol=protocol,
        protocol_path=frozen_path,
        protocol_digest=str(payload["protocol_digest"]),
        profile_name=profile_name,
        profile=profile,
    )


def scaled_cell_futures(value: int, registration: Registration) -> int:
    scaled = max(2, int(round(value * float(registration.profile["futures_scale"]))))
    return scaled if scaled % 2 == 0 else scaled + 1


def half_futures(value: int, registration: Registration) -> int:
    return scaled_cell_futures(value, registration) // 2


def stage_source_count(stage: str, registration: Registration) -> int:
    base = stage.removesuffix("_audit")
    field = {
        "state": "state_sources",
        "boundary": "boundary_sources",
        "slow_mark": "mark_sources",
        "carrier": "carrier_sources",
    }[base]
    return int(registration.profile[field])
