from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT_DIR / "protocols" / "wagner-memory-v1.json"


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
    if protocol.get("format") != "wagner-memory-cleanroom-protocol-v1":
        raise ValueError(f"unsupported protocol format in {path}")
    try:
        selected = dict(protocol["profiles"][profile])
    except KeyError as exc:
        raise ValueError(f"unknown profile: {profile}") from exc
    return Registration(
        protocol=protocol,
        protocol_path=path,
        protocol_digest=file_digest(path),
        profile_name=profile,
        profile=selected,
    )


def scaled_futures(value: int, registration: Registration) -> int:
    scale = float(registration.profile["futures_scale"])
    return max(2, int(round(value * scale / 2.0)) * 2)


def stage_source_count(stage: str, registration: Registration) -> int:
    base = stage.removesuffix("_audit")
    field = {
        "state": "state_sources",
        "boundary": "boundary_sources",
        "slow_mark": "mark_sources",
        "carrier": "carrier_sources",
    }[base]
    return int(registration.profile[field])

