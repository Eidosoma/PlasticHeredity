#!/usr/bin/env python3
"""Exact-environment worker for frozen L45 PhiID state features."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_breakinggrn_transfer_audit.core import run_breaking_transfer
from e01_frozen_timebase_ensemble.core import frozen_clr


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--safe-lattice", required=True, type=Path)
    parser.add_argument("--preprocessing-seed", required=True, type=int)
    parser.add_argument("--partition-seed", required=True, type=int)
    args = parser.parse_args()
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    states = np.load(args.input, allow_pickle=False)["states"].astype(
        np.int64, copy=False
    )
    metadata: dict[str, object] = {
        "status": "WORKER_STARTED",
        "reason": None,
        "inputObservations": len(states),
    }
    arrays: dict[str, np.ndarray] = {}
    try:
        clr, _, closure = frozen_clr(states)
        result = run_breaking_transfer(
            clr,
            args.safe_lattice,
            preprocessing_seed=args.preprocessing_seed,
            partition_seed=args.partition_seed,
        )
        metadata.update(
            status=result.status,
            reason=result.reason,
            partition1Size=len(result.partition_1),
            partition2Size=len(result.partition_2),
            closureErrorMaximum=float(np.max(closure)),
            localOffset=int(result.local_offset),
        )
        for name, value in {
            "emergence_nan0": result.emergence_nan0,
            "integrated_raw": result.integrated_raw,
            "synergy_raw": result.synergy_raw,
            "downward_causation_raw": result.causation_raw,
        }.items():
            if value is not None:
                arrays[name] = np.asarray(value, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - complete provenance is mandatory.
        metadata.update(
            status="UNREGISTERED_WORKER_EXCEPTION",
            reason=f"{type(exc).__name__}:{exc}",
        )
    metadata["wallSeconds"] = time.perf_counter() - started_wall
    metadata["cpuSeconds"] = time.process_time() - started_cpu
    metadata["python"] = sys.version
    metadata["numpyVersion"] = np.__version__
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    atomic_npz(args.output, arrays)


if __name__ == "__main__":
    main()
