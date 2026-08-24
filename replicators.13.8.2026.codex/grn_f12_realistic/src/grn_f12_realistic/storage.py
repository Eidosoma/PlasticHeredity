from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_npz_atomic(path: str | Path, **arrays: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    included: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(
            part in {".venv", ".pytest_cache", "__pycache__", "runs"} or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        included[str(relative)] = sha256_file(path)
    return included


def ensure_registration(run_dir: str | Path, registration: dict[str, Any]) -> None:
    target = Path(run_dir) / "registration.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != registration:
            raise RuntimeError("run registration differs from the frozen contract")
    else:
        write_json_atomic(target, registration)


def update_status(run_dir: str | Path, **changes: Any) -> dict[str, Any]:
    path = Path(run_dir) / "STATUS.json"
    status: dict[str, Any] = {}
    if path.exists():
        status = json.loads(path.read_text(encoding="utf-8"))
    status.update(changes)
    write_json_atomic(path, status)
    return status


def free_gib(path: str | Path) -> float:
    return shutil.disk_usage(Path(path)).free / (1024**3)


def seal_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    entries: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"SHA256SUMS", "verification.json", "STATUS.json"}:
            continue
        entries[str(path.relative_to(root))] = sha256_file(path)
    lines = [f"{digest}  {name}" for name, digest in entries.items()]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "format": "grn-f12-seal-v1", "files": len(entries),
        "verified": False, "status": "sealed_pending_readback",
    }
    write_json_atomic(root / "verification.json", result)
    return result


def verify_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    checksum_file = root / "SHA256SUMS"
    if not checksum_file.exists():
        return {"verified": False, "error": "SHA256SUMS is missing"}
    mismatches: list[str] = []
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split("  ", 1)
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(name)
        checked += 1
    return {"verified": not mismatches, "checked": checked, "mismatches": mismatches}
