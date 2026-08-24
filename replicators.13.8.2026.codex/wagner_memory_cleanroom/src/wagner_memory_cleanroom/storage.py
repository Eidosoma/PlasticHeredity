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
    digest = sha256()
    count = 0
    records = read_records(paths)
    records.sort(key=lambda row: tuple(str(row.get(key, "")) for key in sorted(row)))
    for record in records:
        digest.update(canonical_json(record))
        count += 1
    return count, digest.hexdigest()


def sha256_manifest(root: Path, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    omit = excluded or {"SHA256SUMS"}
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in omit or ".launcher" in relative:
            continue
        digest = sha256(path.read_bytes()).hexdigest()
        entries.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
    return entries
