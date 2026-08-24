from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .dynamics import Landscape, Rulebook
from .protocol import write_json_atomic


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def save_rulebook(path: Path, rulebook: Rulebook) -> None:
    save_npz_atomic(
        path,
        uid=np.asarray(rulebook.uid),
        proposal_index=np.asarray(rulebook.proposal_index, dtype=np.int64),
        weights=rulebook.weights,
        successor=rulebook.landscape.successor,
        adult=rulebook.landscape.adult,
        kind=rulebook.landscape.kind,
        point_index=rulebook.landscape.point_index,
        transient=rulebook.landscape.transient,
        cycle_length=rulebook.landscape.cycle_length,
        point_states=rulebook.landscape.point_states,
        basin_sizes=rulebook.landscape.basin_sizes,
        targets=rulebook.targets,
        target_point_indices=rulebook.target_point_indices,
        midpoints=rulebook.midpoints,
        forced_breaks=rulebook.forced_breaks,
        donors=rulebook.donors,
        nulls=rulebook.nulls,
        shuffles=rulebook.shuffles,
        mark_permutation=rulebook.mark_permutation,
    )


def load_rulebook(path: Path) -> Rulebook:
    with np.load(path, allow_pickle=False) as data:
        landscape = Landscape(
            data["successor"].copy(), data["adult"].copy(), data["kind"].copy(),
            data["point_index"].copy(), data["transient"].copy(),
            data["cycle_length"].copy(), data["point_states"].copy(),
            data["basin_sizes"].copy(),
        )
        return Rulebook(
            uid=str(data["uid"].item()),
            proposal_index=int(data["proposal_index"].item()),
            weights=data["weights"].copy(), landscape=landscape,
            targets=data["targets"].copy(),
            target_point_indices=data["target_point_indices"].copy(),
            midpoints=data["midpoints"].copy(),
            forced_breaks=data["forced_breaks"].copy(),
            donors=data["donors"].copy(), nulls=data["nulls"].copy(),
            shuffles=data["shuffles"].copy(),
            mark_permutation=data["mark_permutation"].copy(),
        )


def sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def seal_directory(root: Path, excluded: set[str] | None = None) -> dict[str, str]:
    excluded = set() if excluded is None else set(excluded)
    excluded.add("SHA256SUMS")
    files = [path for path in root.rglob("*") if path.is_file()]
    manifest = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(files)
        if str(path.relative_to(root)) not in excluded
        and ".tmp." not in path.name
    }
    lines = [f"{checksum}  {name}" for name, checksum in manifest.items()]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def verify_checksums(root: Path) -> list[str]:
    path = root / "SHA256SUMS"
    if not path.exists():
        return ["missing SHA256SUMS"]
    failures: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        checksum, name = line.split("  ", 1)
        target = root / name
        if not target.exists():
            failures.append(f"missing {name}")
        elif sha256_file(target) != checksum:
            failures.append(f"checksum mismatch {name}")
    return failures


def update_status(run_dir: Path, **fields: Any) -> None:
    path = run_dir / "STATUS.json"
    status: dict[str, Any] = {}
    if path.exists():
        status = json.loads(path.read_text(encoding="utf-8"))
    status.update(fields)
    write_json_atomic(path, status)

