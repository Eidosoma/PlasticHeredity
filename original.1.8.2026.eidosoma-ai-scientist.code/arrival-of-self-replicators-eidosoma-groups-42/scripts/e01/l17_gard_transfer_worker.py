#!/usr/bin/env python3
"""One exact-environment worker for the frozen L17 GARD transfer cohort."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_breakinggrn_transfer_audit.core import (  # noqa: E402
    array_sha256,
    derive_seed,
    run_breaking_transfer,
    sha256_file,
)
from e01_frozen_timebase_ensemble.core import (  # noqa: E402
    frozen_clr,
    selected_clock_observations,
    states_from_observations,
)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-json", required=True, type=Path)
    parser.add_argument("--safe-lattice", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    task = json.loads(args.task_json.read_text(encoding="utf-8"))
    candidate = str(task["candidateId"])
    matrix_index = int(task["matrixIndex"])
    trajectory_path = Path(task["cachePath"])
    metadata: dict[str, object] = {
        "candidateId": candidate,
        "matrixIndex": matrix_index,
        "trajectoryId": task["trajectoryId"],
        "status": "WORKER_STARTED",
        "reason": None,
        "workerPython": sys.version,
        "numpyVersion": np.__version__,
    }
    arrays: dict[str, np.ndarray] = {}
    try:
        if sha256_file(trajectory_path) != task["cacheSha256"]:
            raise RuntimeError("trajectory cache SHA-256 mismatch")
        with trajectory_path.open("rb") as handle:
            trajectory = pickle.load(handle)
        if (
            trajectory.trajectory_id != task["trajectoryId"]
            or trajectory.trajectory_sha256 != task["trajectorySha256"]
            or trajectory.beta_sha256 != task["betaSha256"]
            or trajectory.initial_state_sha256 != task["initialStateSha256"]
            or trajectory.configuration_id != candidate
            or int(trajectory.matrix_index) != matrix_index
            or int(trajectory.completed_fissions) != 100
        ):
            raise RuntimeError("frozen trajectory identity mismatch")
        selected = selected_clock_observations(
            trajectory, "C1_SELECTED_DAUGHTER_RETAINED"
        )
        states = states_from_observations(selected)
        clr, masses, closure_errors = frozen_clr(states)
        pre_seed = derive_seed("preprocess", candidate, matrix_index)
        partition_seed = derive_seed("partition", candidate, matrix_index)
        result = run_breaking_transfer(
            clr,
            args.safe_lattice,
            preprocessing_seed=pre_seed,
            partition_seed=partition_seed,
        )
        metadata.update(
            status=result.status,
            reason=result.reason,
            selectedObservationCount=int(len(selected)),
            outputLocalOffset=int(result.local_offset),
            preprocessingSeed=int(pre_seed),
            partitionSeed=int(partition_seed),
            partition1=list(result.partition_1),
            partition2=list(result.partition_2),
            closureErrorMaximum=float(np.max(closure_errors)),
            massMinimum=float(np.min(masses)),
            massMaximum=float(np.max(masses)),
        )
        arrays.update(
            selected_sequence_index=np.arange(len(selected), dtype=np.int64),
            raw_observation_index=np.asarray(
                [int(item.observation_index) for item in selected], dtype=np.int64
            ),
            generation=np.asarray(
                [int(item.growth_generation_one_based) for item in selected],
                dtype=np.int64,
            ),
            molecular_step=np.asarray(
                [int(item.batch_step) for item in selected],
                dtype=np.int64,
            ),
            clr_hash=np.array(array_sha256(clr)),
        )
        optional = {
            "processed": result.processed,
            "mi": result.mi_matrix,
            "fiedler": result.fiedler_vector,
            "reduced": result.reduced,
            "synergy_raw": result.synergy_raw,
            "causation_raw": result.causation_raw,
            "emergence_raw": result.emergence_raw,
            "emergence_nan0": result.emergence_nan0,
            "integrated_raw": result.integrated_raw,
        }
        for name, value in optional.items():
            if value is not None:
                arrays[name] = np.asarray(value)
                metadata[f"{name}Sha256"] = array_sha256(np.asarray(value))
    except Exception as exc:  # noqa: BLE001 - complete failure provenance required.
        metadata.update(
            status="UNREGISTERED_WORKER_EXCEPTION",
            reason=f"{type(exc).__name__}:{exc}",
        )
    metadata["wallSeconds"] = time.perf_counter() - started_wall
    metadata["cpuSeconds"] = time.process_time() - started_cpu
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    atomic_npz(args.output, arrays)


if __name__ == "__main__":
    main()
