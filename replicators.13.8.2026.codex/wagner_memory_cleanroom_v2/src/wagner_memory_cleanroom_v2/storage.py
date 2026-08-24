from __future__ import annotations

import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .config import canonical_json


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_bytes(canonical_json(value))
    os.replace(temporary, path)


def write_records(path: Path, records: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    digest = sha256()
    count = 0
    with gzip.GzipFile(filename=str(temporary), mode="wb", compresslevel=6, mtime=0) as handle:
        for record in records:
            line = canonical_json(record)
            digest.update(line)
            handle.write(line)
            count += 1
    os.replace(temporary, path)
    return count, digest.hexdigest()


def read_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def records_digest(paths: Iterable[Path]) -> tuple[int, str]:
    rows = read_records(paths)
    rows.sort(key=lambda row: str(row["cell_id"]))
    digest = sha256()
    for row in rows:
        digest.update(canonical_json(row))
    return len(rows), digest.hexdigest()


def objects_digest(paths: Iterable[Path], key: str) -> tuple[int, str]:
    rows = read_records(paths)
    rows.sort(key=lambda row: (row[key], canonical_json(row)))
    digest = sha256()
    for row in rows:
        digest.update(canonical_json(row))
    return len(rows), digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_manifest(root: Path, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    omit = excluded or {"SHA256SUMS", "manifest.json"}
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in omit or ".launcher" in relative:
            continue
        entries.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return entries
