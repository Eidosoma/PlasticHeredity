#!/usr/bin/env python3
"""Write deterministic SHA-256 checksums for a result directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    destination = arguments.directory / "SHA256SUMS"
    paths = sorted(
        path
        for path in arguments.directory.rglob("*")
        if path.is_file() and path != destination
    )
    lines = [
        f"{digest(path)}  {path.relative_to(arguments.directory)}" for path in paths
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(destination)


if __name__ == "__main__":
    main()
