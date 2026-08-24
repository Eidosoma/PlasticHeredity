#!/usr/bin/env python3
"""Isolated adapter for the pinned BreakingGRNMemories Phi functions.

The adapter is used only for deterministic synthetic source-equivalence
fixtures.  It imports the audited public ``information.py`` (and therefore its
pickle) inside a disposable isolated process.  Frozen GARD trajectories are
never passed to this adapter; scientific transfer uses the previously
validated safe-JSON clean-room implementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


SYNERGY_ATOM = (((0, 1),), ((0, 1),))
DOWNWARD_ATOMS = (
    (((0, 1),), ((0,),)),
    (((0, 1),), ((1,),)),
)


def load_information(source_dir: Path):
    os.chdir(source_dir)
    spec = importlib.util.spec_from_file_location(
        "breakinggrn_information_l17", source_dir / "information.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct BreakingGRNMemories information module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not getattr(sys.flags, "isolated", 0):
        raise RuntimeError("L17 source adapter must run with python -I")
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preprocessing-seed", required=True, type=int)
    parser.add_argument("--partition-seed", required=True, type=int)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    raw = np.load(args.input, allow_pickle=False)["observations"].astype(
        np.float64, copy=False
    )
    info = load_information(source_dir)
    metadata: dict[str, object] = {
        "status": "ELIGIBLE",
        "reason": None,
        "localOffset": 2,
        "sourceEntrypoint": "information.py functions equivalent to phi.py::compute_integrated_info",
    }
    arrays: dict[str, np.ndarray] = {}
    try:
        np.random.seed(args.preprocessing_seed)
        processed = info.preprocess_data(raw.T.copy())
        processed_phi = np.nan_to_num(processed, nan=0.0, copy=True)
        np.random.seed(args.partition_seed)
        mi = info.mutual_information_matrix(
            processed_phi, alpha=1, bonferonni=False, lag=1
        )
        partition = info.minimum_information_bipartition(mi, noise=True)
        p1 = np.asarray(partition[0], dtype=np.int64)
        p2 = np.asarray(partition[1], dtype=np.int64)
        if p1.size == 0 or p2.size == 0:
            raise RuntimeError("empty strict-sign Fiedler partition")
        reduced = np.vstack(
            (processed_phi[p1].mean(axis=0), processed_phi[p2].mean(axis=0))
        )
        lattice = info.local_phi_id(0, 1, reduced)
        synergy_raw = np.asarray(lattice.nodes[SYNERGY_ATOM]["pi"], dtype=np.float64)
        causation_raw = np.asarray(
            lattice.nodes[DOWNWARD_ATOMS[0]]["pi"]
            + lattice.nodes[DOWNWARD_ATOMS[1]]["pi"],
            dtype=np.float64,
        )
        integrated_raw = np.asarray(info.local_phi_r(lattice), dtype=np.float64)
        synergy_nan0 = np.nan_to_num(
            synergy_raw, nan=0.0, posinf=0.0, neginf=0.0
        )
        causation_nan0 = np.nan_to_num(
            causation_raw, nan=0.0, posinf=0.0, neginf=0.0
        )
        arrays.update(
            processed=processed,
            processed_phi=processed_phi,
            mi=mi,
            partition_1=p1,
            partition_2=p2,
            reduced=reduced,
            synergy_raw=synergy_raw,
            causation_raw=causation_raw,
            emergence_raw=synergy_raw + causation_raw,
            integrated_raw=integrated_raw,
            emergence_nan0=synergy_nan0 + causation_nan0,
        )
    except Exception as exc:  # noqa: BLE001 - exact source failure is evidence.
        metadata.update(
            status="INELIGIBLE_SOURCE_PIPELINE_EXCEPTION",
            reason=f"{type(exc).__name__}:{exc}",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )


if __name__ == "__main__":
    main()
