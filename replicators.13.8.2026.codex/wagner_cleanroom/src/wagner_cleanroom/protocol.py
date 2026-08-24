from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = PROJECT_ROOT / "protocols"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def label_seed(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def load_protocol(name: str, profile: str = "full") -> dict[str, Any]:
    path = PROTOCOL_ROOT / f"{name}-v1.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if profile == "full":
        return protocol
    if profile != "smoke":
        raise ValueError(f"unknown profile: {profile}")
    protocol = deepcopy(protocol)
    protocol["scientific"] = False
    protocol["profile"] = "smoke"
    protocol["bootstrap_repetitions"] = 256
    protocol["maximum_source_proposals"] = 10000
    if name == "primary":
        protocol["source_count"] = 6
        for cell in protocol["conditions"]:
            cell["futures"] = min(16, int(cell["futures"]))
    elif name == "predictor":
        protocol["development_sources"] = 6
        protocol["evaluation_sources"] = 8
        protocol["futures_per_state"] = 16
        protocol["histories_per_source"] = 3
    return protocol


def registration(protocol: dict[str, Any]) -> dict[str, Any]:
    body = deepcopy(protocol)
    labels = [v for k, v in body.items() if k.endswith("seed_label")]
    return {
        "format": "wagner-cleanroom-registration-v1",
        "protocol": body,
        "protocol_digest": digest(body),
        "derived_master_seeds": {label: label_seed(label) for label in labels},
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def ensure_registration(run_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    wanted = registration(protocol)
    path = run_dir / "registration.json"
    if path.exists():
        found = json.loads(path.read_text(encoding="utf-8"))
        if found.get("protocol_digest") != wanted["protocol_digest"]:
            raise RuntimeError(f"registration mismatch in {run_dir}")
        return found
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, wanted)
    return wanted

