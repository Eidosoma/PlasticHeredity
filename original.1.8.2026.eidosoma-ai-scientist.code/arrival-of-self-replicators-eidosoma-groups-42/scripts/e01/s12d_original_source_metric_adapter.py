#!/usr/bin/env python3
"""Isolated adapter for S12D source-metric identity checks only.

This process imports the unmodified pinned public ``information.py`` and its
already audited pickle.  It receives synthetic fixtures only.  No GARD input
is ever passed to it, and S12D scientific execution uses the safe JSON wrapper.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import zscore

SYNERGY_ATOM = (((0, 1),), ((0, 1),))
DOWNWARD_ATOMS = (
    (((0, 1),), ((0,),)),
    (((0, 1),), ((1,),)),
)


def canonical_atom_key(atom: Any) -> str:
    return json.dumps(atom, separators=(",", ":"))


def extract_function(path: Path, name: str, namespace: dict[str, Any]):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(functions) != 1:
        raise RuntimeError(f"expected exactly one {name} in {path}")
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(  # noqa: S102 - executes only one audited pinned FunctionDef.
        compile(module, str(path), "exec"), namespace
    )
    return namespace[name]


def load_information(source_dir: Path):
    os.chdir(source_dir)
    spec = importlib.util.spec_from_file_location(
        "pinned_information_s12d", source_dir / "information.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct pinned information module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_result(
    path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )


def main() -> None:
    if not getattr(sys.flags, "isolated", 0):
        raise RuntimeError("source adapter must run with python -I")
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["IIGR", "PHIRL"], required=True)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preprocessing-seed", required=True, type=int)
    parser.add_argument("--partition-seed", required=True, type=int)
    args = parser.parse_args()

    raw = np.load(args.input, allow_pickle=False)["observations"].astype(
        np.float64, copy=False
    )
    source_dir = args.source_dir.resolve()
    info = load_information(source_dir)
    namespace: dict[str, Any] = {
        "np": np,
        "zscore": zscore,
        "corrected_zscore": info.corrected_zscore,
        "global_signal_regression": info.global_signal_regression,
        "remove_autocorrelation": info.remove_autocorrelation,
    }
    preprocess = extract_function(source_dir / "main.py", "preprocess_data", namespace)
    offset = 2 if args.implementation == "IIGR" else 1
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "implementation": args.implementation,
        "status": "ELIGIBLE",
        "reason": None,
        "localOffset": offset,
        "atomKeySerialization": [
            canonical_atom_key(SYNERGY_ATOM),
            canonical_atom_key(DOWNWARD_ATOMS[0]),
            canonical_atom_key(DOWNWARD_ATOMS[1]),
        ],
    }
    try:
        data = raw.T.copy()
        np.random.seed(args.preprocessing_seed)
        if args.implementation == "IIGR":
            retained = np.arange(data.shape[0], dtype=np.int64)
            processed = preprocess(data)
            mi = info.mutual_information_matrix(
                processed, alpha=1, bonferonni=False, lag=1
            )
        else:
            retained = np.flatnonzero(data.std(axis=1) > 1e-8).astype(np.int64)
            if retained.size < 2:
                metadata.update(
                    status="INELIGIBLE_TOO_FEW_ACTIVE_DIMENSIONS",
                    reason="fewer_than_two_dimensions_above_std_1e-8",
                )
                arrays["retained"] = retained
                write_result(args.output, metadata, arrays)
                return
            processed = preprocess(data)
            mi = info.mutual_information_matrix_fast(
                processed, alpha=1, bonferonni=False, lag=1
            )
        arrays.update(retained=retained, processed=processed, mi=mi)
        if not np.all(np.isfinite(processed)) or not np.all(np.isfinite(mi)):
            metadata.update(
                status="INELIGIBLE_NONFINITE_PREPROCESSING_OR_MI",
                reason="processed_array_or_mi_nonfinite",
            )
            write_result(args.output, metadata, arrays)
            return
        np.random.seed(args.partition_seed)
        partition = info.minimum_information_bipartition(mi, noise=True)
        p1_local = np.asarray(partition[0], dtype=np.int64)
        p2_local = np.asarray(partition[1], dtype=np.int64)
        arrays["partition_1_local"] = p1_local
        arrays["partition_2_local"] = p2_local
        if p1_local.size == 0 or p2_local.size == 0:
            metadata.update(
                status="INELIGIBLE_FIEDLER_PARTITION_EMPTY",
                reason="strict_sign_partition_has_empty_side",
            )
            write_result(args.output, metadata, arrays)
            return
        reduced = np.vstack(
            (processed[p1_local].mean(axis=0), processed[p2_local].mean(axis=0))
        )
        lattice = info.local_phi_id(0, 1, reduced)
        synergy = np.asarray(lattice.nodes[SYNERGY_ATOM]["pi"], dtype=np.float64)
        downward = np.asarray(
            lattice.nodes[DOWNWARD_ATOMS[0]]["pi"]
            + lattice.nodes[DOWNWARD_ATOMS[1]]["pi"],
            dtype=np.float64,
        )
        emergence = synergy + downward
        local_phi_r = np.asarray(info.local_phi_r(lattice), dtype=np.float64)
        arrays.update(
            partition_1=retained[p1_local],
            partition_2=retained[p2_local],
            partition_average=reduced,
            synergy=synergy,
            downward_causation=downward,
            emergence=emergence,
            local_phi_r=local_phi_r,
        )
        # Preserve the exact S12C/source-adapter availability policy.  Separate
        # atom masks are compared independently by S12D, but do not redefine
        # the already confirmed source status.
        if not np.all(np.isfinite(emergence)) or not np.all(np.isfinite(local_phi_r)):
            metadata.update(
                status="ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES",
                reason="one_or_more_local_metric_component_values_nonfinite",
            )
    except Exception as exc:  # noqa: BLE001 - exact source status is evidence.
        metadata.update(
            status="INELIGIBLE_SOURCE_PIPELINE_EXCEPTION",
            reason=f"{type(exc).__name__}:{exc}",
        )
    write_result(args.output, metadata, arrays)


if __name__ == "__main__":
    main()
