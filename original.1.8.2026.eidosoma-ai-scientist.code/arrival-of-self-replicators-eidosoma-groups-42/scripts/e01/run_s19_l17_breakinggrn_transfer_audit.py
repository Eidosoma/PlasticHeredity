#!/usr/bin/env python3
"""Execute the locked E01/S19-L17 source-lineage transfer audit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from e01_breakinggrn_transfer_audit.core import (
    array_sha256,
    derive_seed,
    run_breaking_transfer,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
CONFIG_PATH = REPO_ROOT / "configs/e01/s19_l17_breakinggrn_transfer_audit.yaml"
OUTPUT_ROOT = Path("/artifacts/research_steps/S19/loops/L17")
S19_ROOT = Path("/artifacts/research_steps/S19")
CACHE_ROOT = Path("/cache/e01_s19_l17")
BUILD_ROOT = CACHE_ROOT / "final_build"
SOURCE_ROOT = CACHE_ROOT / "sources/BreakingGRNMemories"
EXACT_PYTHON = CACHE_ROOT / "venv/bin/python"
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
L12_ROOT = S19_ROOT / "loops/L12"
PAPER_MARKDOWN = (
    WORKSPACE_ROOT
    / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md"
)
PAPER_PDF = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")
PHIRL_ROOT = Path("/cache/e01_s12b/sources/PhiRL")
IIGR_ROOT = Path("/cache/e01_s12b/sources/IntegratedInformationGeneRegulation")
GARD_ROOT = Path("/cache/e01_s03/sources/gard-historical")
SOURCE_ADAPTER = REPO_ROOT / "scripts/e01/l17_breakinggrn_source_adapter.py"
GARD_WORKER = REPO_ROOT / "scripts/e01/l17_gard_transfer_worker.py"

HYPOTHESIS_METRICS = {
    "H1_BGM_CURRENT_PHI_EMERGENCE_NANZERO_COMPLETED": "emergence_nan0",
    "H2_BGM_OPTIMA_EMERGENCE_RAW_COMPLETED": "emergence_raw",
    "H3_BGM_INFORMATION_INTEGRATED_RAW_COMPLETED": "integrated_raw",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def write_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_yaml(path: Path, value: object) -> None:
    atomic_text(path, yaml.safe_dump(value, sort_keys=False))


def canonicalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(
                lambda value: canonical_json(value)
                if isinstance(value, (dict, list, tuple))
                else value
            )
    return result


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    canonicalize_frame(frame).to_parquet(temp, index=False, compression="zstd")
    os.replace(temp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, canonicalize_frame(frame).to_csv(index=False, lineterminator="\n"))


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def git_identity(repo: Path) -> dict[str, Any]:
    return {
        "path": str(repo),
        "head": command("git", "rev-parse", "HEAD", cwd=repo),
        "tree": command("git", "rev-parse", "HEAD^{tree}", cwd=repo),
        "branch": command("git", "branch", "--show-current", cwd=repo),
        "status": command("git", "status", "--short", cwd=repo),
        "remote": command("git", "remote", "get-url", "origin", cwd=repo),
    }


def require_clean_pushed_lock() -> dict[str, Any]:
    head = command("git", "rev-parse", "HEAD", cwd=REPO_ROOT)
    remote = command("git", "rev-parse", "@{u}", cwd=REPO_ROOT)
    status = command("git", "status", "--short", cwd=REPO_ROOT)
    if status or head != remote:
        raise RuntimeError(
            f"pre-outcome lock not clean/pushed: head={head} remote={remote} status={status!r}"
        )
    return {
        "branch": command("git", "branch", "--show-current", cwd=REPO_ROOT),
        "head": head,
        "remoteHead": remote,
        "cleanWorktree": True,
    }


def verify_source(config: dict[str, Any]) -> dict[str, Any]:
    frozen = config["breakingGrnSnapshot"]
    identity = git_identity(SOURCE_ROOT)
    if identity["head"] != frozen["commit"] or identity["tree"] != frozen["tree"]:
        raise RuntimeError("BreakingGRNMemories source identity changed after lock")
    if identity["status"]:
        raise RuntimeError("BreakingGRNMemories source checkout is dirty")
    rows = []
    for relative, expected in frozen["relevantFileSha256"].items():
        path = SOURCE_ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"BreakingGRNMemories hash mismatch: {relative}")
        rows.append(
            {
                "path": relative,
                "sha256": actual,
                "bytes": path.stat().st_size,
                "verified": True,
            }
        )
    identity["relevantFiles"] = rows
    identity["licenseStatus"] = frozen["licenseStatus"]
    return identity


def immutable_roots() -> list[Path]:
    roots = [
        Path("/artifacts/E01_forensic_replication_bundle"),
        Path("/artifacts/E01_forensic_replication_artifact_v2"),
    ]
    research = Path("/artifacts/research_steps")
    roots.extend(
        path
        for path in sorted(research.glob("S*"))
        if path.is_dir() and path.name != "S19"
    )
    roots.extend(
        S19_ROOT / "loops" / name
        for name in [
            "L01", "L02", "L03", "L04", "L05", "L06", "L06R", "L07",
            "L08", "L09", "L10", "L11", "L11R", "L12", "L13", "L14",
            "L15", "L16",
        ]
    )
    return roots


def hash_tree(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rows.append(
                {
                    "root": str(root),
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def source_tree() -> dict[str, Any]:
    rows = []
    for line in command("git", "ls-tree", "-r", "-l", "HEAD", cwd=SOURCE_ROOT).splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, object_id, size = metadata.split()
        rows.append(
            {
                "path": path,
                "mode": mode,
                "kind": kind,
                "gitObject": object_id,
                "bytes": None if size == "-" else int(size),
            }
        )
    return {"commit": git_identity(SOURCE_ROOT)["head"], "entries": rows}


def source_history() -> pd.DataFrame:
    formatted = "%H%x1f%P%x1f%aI%x1f%an%x1f%s"
    rows = []
    for line in command(
        "git", "log", "--all", "--reverse", f"--format={formatted}", cwd=SOURCE_ROOT
    ).splitlines():
        commit, parents, date, author, subject = line.split("\x1f")
        rows.append(
            {
                "commit": commit,
                "parents": parents,
                "authorDate": date,
                "author": author,
                "subject": subject,
            }
        )
    return pd.DataFrame(rows)


def function_registry() -> pd.DataFrame:
    rows = []
    for relative in ["information.py", "phi.py", "optima.py", "plotting.py"]:
        path = SOURCE_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        blame = command("git", "blame", "--line-porcelain", "HEAD", "--", relative, cwd=SOURCE_ROOT)
        commits = [line.split()[0] for line in blame.splitlines() if len(line.split()) >= 4 and len(line.split()[0]) == 40]
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                rows.append(
                    {
                        "file": relative,
                        "symbol": node.name,
                        "symbolType": type(node).__name__,
                        "lineStart": node.lineno,
                        "lineEnd": node.end_lineno,
                        "currentCommit": command("git", "log", "-1", "--format=%H", "--", relative, cwd=SOURCE_ROOT),
                        "blameCommitAtStart": commits[node.lineno - 1] if len(commits) >= node.lineno else None,
                    }
                )
    return pd.DataFrame(rows)


def build_dataflow() -> nx.DiGraph:
    graph = nx.DiGraph()
    nodes = [
        ("gard_counts", "Frozen S13Y selected molecular count states"),
        ("clr99", "Paper additive-0.5 CLR, original component 100 dropped"),
        ("zscore1", "Corrected row z-score"),
        ("gsr", "Global signal regression and corrected z-score"),
        ("ar1", "Per-channel lag-one residualization and corrected z-score"),
        ("mi", "Slow bidirectional Gaussian lag-one MI sum"),
        ("floor", "Uniform 1e-6 MI graph floor"),
        ("fiedler", "Unnormalized Fiedler strict-sign bipartition"),
        ("average", "Arithmetic partition averages"),
        ("entropy", "Unregularized local Gaussian entropies"),
        ("phiid", "MMI local PhiID lattice"),
        ("synergy", "Synergy atom"),
        ("causation", "Two downward-causation atoms"),
        ("emergence_raw", "Raw synergy plus causation"),
        ("emergence_nan0", "Current phi.py nonfinite-to-zero emergence"),
        ("integrated", "information.py corrected local Phi-r atom sum"),
        ("run_analysis", "Frozen run-level level/change/state analyses"),
    ]
    for node, label in nodes:
        graph.add_node(node, label=label)
    for left, right in zip(
        ["gard_counts", "clr99", "zscore1", "gsr", "ar1", "mi", "floor", "fiedler", "average", "entropy"],
        ["clr99", "zscore1", "gsr", "ar1", "mi", "floor", "fiedler", "average", "entropy", "phiid"],
    ):
        graph.add_edge(left, right)
    graph.add_edge("phiid", "synergy")
    graph.add_edge("phiid", "causation")
    graph.add_edge("phiid", "integrated")
    graph.add_edge("synergy", "emergence_raw")
    graph.add_edge("causation", "emergence_raw")
    graph.add_edge("emergence_raw", "emergence_nan0")
    graph.add_edge("emergence_raw", "run_analysis")
    graph.add_edge("emergence_nan0", "run_analysis")
    graph.add_edge("integrated", "run_analysis")
    return graph


def lineage_crosswalk() -> pd.DataFrame:
    rows = [
        ("input", "GRN channel-by-time trajectory", "generic time-by-variable trajectory", "GARD molecular trajectory after CLR99", "paper specifies CLR/drop-last; BGM has no GARD adapter"),
        ("active-variable filter", "none", "std>1e-8", "none after CLR99", "BGM follows corrected IIGR, not PhiRL"),
        ("preprocessing", "zscore→GSR→zscore→AR1 residual→zscore", "active filter→zscore", "BGM path unchanged after paper CLR", "material source-lineage difference"),
        ("lagged MI", "slow bidirectional Gaussian MI values summed", "fast averaged bidirectional correlations then Gaussian MI", "slow BGM/IIGR", "material scale and graph difference"),
        ("significance", "alpha=1, Bonferroni false", "alpha=1, Bonferroni false", "same declared settings", "retains every finite nonzero edge"),
        ("graph floor", "additive 1e-6 to all cells", "additive 1e-6", "same", "direct public code"),
        ("partition", "unnormalized Fiedler strict sign", "unnormalized Fiedler strict sign", "same method, frozen wrapper seed", "zero-valued entries unassigned"),
        ("reduction", "arithmetic partition means", "arithmetic partition means", "same", "direct public code"),
        ("covariance", "unregularized", "trace-scaled 1e-6 regularizer", "unregularized BGM", "high numerical-risk difference"),
        ("lattice", "same phi_lattice_22 pickle", "same byte-identical pickle", "safe JSON equivalent", "SHA-256 66cd… public pickle"),
        ("emergence", "synergy + two downward atoms", "same", "H1/H2", "source-defined emergence"),
        ("integrated", "local_phi_r atom sum; current top-level comments it out", "exposed local_phi_r", "H3 diagnostic", "paper equation remains internally inconsistent"),
        ("temporal fit", "complete GRN episode", "complete supplied trajectory", "complete S13Y trajectory", "retrospective future-dependent only"),
        ("nulls", "none in Phi script", "time-shuffled trajectory in public tooling", "no added transfer null", "frozen H/stability controls only"),
        ("prefix mode", "absent", "absent as a complete GARD method", "not executed", "cannot support prospective evidence"),
        ("seeds", "JAX/env/action seeds 0; Python/NumPy/NetworkX incomplete", "source-dependent", "domain-separated wrapper for replay", "wrapper is explicit reconstruction choice, not author identity"),
        ("GARD label", "absent", "absent", "frozen adjacent H>0.9 only", "label exactly determined by H and overly permissive"),
    ]
    return pd.DataFrame(
        rows,
        columns=["operation", "breakingGrn", "phirl", "l17Transfer", "adjudication"],
    )


def hypothesis_gate(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    required = config["hypothesisCompletenessGate"]["requiredFields"]
    key_map = {
        "source_entrypoint": "sourceEntrypoint",
        "input_object": "inputObject",
        "preprocessing": "preprocessing",
        "lagged_mi": "laggedMi",
        "partition": "partition",
        "reduction": "reduction",
        "entropy_and_regularization": "entropyAndRegularization",
        "atom_identity": "atomIdentity",
        "scalar_identity": "scalarIdentity",
        "temporal_fit": "temporalFit",
        "numerical_nonfinite_policy": "numericalNonfinitePolicy",
        "output_alignment": "outputAlignment",
        "analysis_estimands": "analysisEstimands",
        "source_seed_wrapper": "sourceSeedWrapper",
    }
    for item in config["transferHypotheses"]:
        missing = [name for name in required if not item.get(key_map[name])]
        rows.append(
            {
                "hypothesisId": item["hypothesisId"],
                "complete": not missing,
                "missingFields": missing,
                "registeredForExecution": not missing,
                "sourceEntrypoint": item["sourceEntrypoint"],
                "scalarIdentity": item["scalarIdentity"],
                "temporalFit": item["temporalFit"],
                "seedSemantics": item["sourceSeedWrapper"],
                "seedGroundingCaveat": "OPERATIONAL_REPLAY_WRAPPER_NOT_VISIBLE_AUTHOR_CALLER_SEED",
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) > 3:
        raise RuntimeError("more than three L17 transfer hypotheses registered")
    return frame


def fixture_array(fixture_id: str) -> np.ndarray:
    rng = np.random.RandomState(derive_seed("fixture", fixture_id))
    if fixture_id == "COUPLED_GAUSSIAN":
        x = rng.normal(size=(384, 10))
        x[:, 5:] += 0.35 * x[:, :5]
        return x
    if fixture_id == "COUPLED_AUTOREGRESSIVE":
        innovations = rng.normal(size=(384, 10))
        x = np.zeros_like(innovations)
        for index in range(1, len(x)):
            x[index] = 0.55 * x[index - 1] + innovations[index]
            x[index, 5:] += 0.25 * x[index - 1, :5]
        return x
    if fixture_id == "CONSTANT_CHANNEL":
        x = rng.normal(size=(384, 10))
        x[:, 0] = 1.0
        return x
    if fixture_id == "DUPLICATED_CHANNEL":
        base = rng.normal(size=(384, 5))
        return np.column_stack((base, base))
    raise ValueError(fixture_id)


def load_npz(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    loaded = np.load(path, allow_pickle=False)
    metadata = json.loads(str(loaded["metadata_json"]))
    arrays = {name: np.asarray(loaded[name]) for name in loaded.files if name != "metadata_json"}
    return metadata, arrays


def run_fixtures(config: dict[str, Any], root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root.mkdir(parents=True, exist_ok=True)
    fixture_rows = []
    equivalence_rows = []
    for fixture_id in config["fixtures"]["ids"]:
        if fixture_id == "EXACT_REPLAY":
            continue
        x = fixture_array(fixture_id)
        input_path = root / f"{fixture_id}.input.npz"
        np.savez_compressed(input_path, observations=x)
        pre_seed = derive_seed("fixture", fixture_id, "preprocess")
        part_seed = derive_seed("fixture", fixture_id, "partition")
        outputs = []
        for replay in (1, 2):
            output = root / f"{fixture_id}.source.{replay}.npz"
            subprocess.run(
                [
                    str(EXACT_PYTHON), "-I", str(SOURCE_ADAPTER),
                    "--source-dir", str(SOURCE_ROOT),
                    "--input", str(input_path),
                    "--output", str(output),
                    "--preprocessing-seed", str(pre_seed),
                    "--partition-seed", str(part_seed),
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            outputs.append(load_npz(output))
        source_meta, source_arrays = outputs[0]
        replay_meta, replay_arrays = outputs[1]
        clean = run_breaking_transfer(
            x,
            SAFE_LATTICE,
            preprocessing_seed=pre_seed,
            partition_seed=part_seed,
        )
        replay_equal = source_meta["status"] == replay_meta["status"] and set(source_arrays) == set(replay_arrays)
        for name in source_arrays:
            replay_equal &= array_sha256(source_arrays[name]) == array_sha256(replay_arrays[name])
        expected_eligible = fixture_id in {"COUPLED_GAUSSIAN", "COUPLED_AUTOREGRESSIVE"}
        clean_status_match = (
            source_meta["status"] == clean.status
            or (source_meta["status"].startswith("INELIGIBLE") and clean.status.startswith("INELIGIBLE"))
        )
        fixture_pass = replay_equal and clean_status_match
        if expected_eligible:
            fixture_pass &= source_meta["status"] == "ELIGIBLE" and clean.status == "ELIGIBLE"
        fixture_rows.append(
            {
                "fixtureId": fixture_id,
                "sourceStatus": source_meta["status"],
                "cleanStatus": clean.status,
                "sourceReason": source_meta.get("reason"),
                "cleanReason": clean.reason,
                "exactSourceReplayPassed": bool(replay_equal),
                "statusEquivalencePassed": bool(clean_status_match),
                "fixturePassed": bool(fixture_pass),
            }
        )
        clean_map = {
            "processed": clean.processed,
            "mi": clean.mi_matrix,
            "reduced": clean.reduced,
            "emergence_raw": clean.emergence_raw,
            "emergence_nan0": clean.emergence_nan0,
            "integrated_raw": clean.integrated_raw,
        }
        for name, clean_value in clean_map.items():
            source_value = source_arrays.get(name)
            if clean_value is None or source_value is None:
                continue
            finite = np.isfinite(clean_value) & np.isfinite(source_value)
            maximum = float(np.max(np.abs(clean_value[finite] - source_value[finite]))) if finite.any() else None
            passed = bool(
                np.allclose(
                    clean_value,
                    source_value,
                    atol=float(config["fixtures"]["exactToleranceAbsolute"]),
                    rtol=float(config["fixtures"]["exactToleranceRelative"]),
                    equal_nan=True,
                )
            )
            equivalence_rows.append(
                {
                    "fixtureId": fixture_id,
                    "arrayName": name,
                    "shape": list(source_value.shape),
                    "maximumAbsoluteError": maximum,
                    "sourceEquivalencePassed": passed,
                }
            )
    fixtures = pd.DataFrame(fixture_rows)
    equivalence = pd.DataFrame(equivalence_rows)
    if not fixtures["fixturePassed"].all() or (not equivalence.empty and not equivalence["sourceEquivalencePassed"].all()):
        raise RuntimeError("mandatory BreakingGRNMemories source-equivalence fixture failed")
    return fixtures, equivalence


def task_manifest() -> pd.DataFrame:
    manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet").sort_values(
        ["candidateId", "matrixIndex"]
    )
    if len(manifest) != 200 or set(manifest["candidateId"]) != {
        "S12F-CANDIDATE-02", "S12F-CANDIDATE-03"
    }:
        raise RuntimeError("S13Y manifest scope mismatch")
    if not manifest["exactReplayPassed"].all() or not manifest["completedFissions"].eq(100).all():
        raise RuntimeError("S13Y input replay/fission gate failed")
    return manifest.reset_index(drop=True)


def make_task_files(manifest: pd.DataFrame, root: Path) -> dict[tuple[str, int], Path]:
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for row in manifest.to_dict("records"):
        key = (str(row["candidateId"]), int(row["matrixIndex"]))
        path = root / f"{key[0]}_M{key[1]:03d}.json"
        atomic_text(path, json.dumps(row, sort_keys=True, default=str) + "\n")
        result[key] = path
    return result


def run_worker(task_path: Path, output: Path) -> dict[str, Any]:
    env = os.environ.copy()
    for name in [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ]:
        env[name] = "1"
    started = time.perf_counter()
    completed = subprocess.run(
        [
            str(EXACT_PYTHON), str(GARD_WORKER),
            "--task-json", str(task_path),
            "--safe-lattice", str(SAFE_LATTICE),
            "--output", str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed {task_path.name}: {completed.stderr[-2000:]}"
        )
    metadata, _ = load_npz(output)
    metadata["subprocessWallSeconds"] = time.perf_counter() - started
    metadata["outputPath"] = str(output)
    metadata["outputSha256"] = sha256_file(output)
    return metadata


def run_task_set(
    keys: list[tuple[str, int]],
    task_files: dict[tuple[str, int], Path],
    output_root: Path,
    workers: int,
) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for candidate, index in keys:
            output = output_root / candidate / f"M{index:03d}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            futures[pool.submit(run_worker, task_files[(candidate, index)], output)] = (candidate, index)
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows).sort_values(["candidateId", "matrixIndex"]).reset_index(drop=True)


def compare_replay(first_root: Path, replay_root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    arrays_to_compare = [
        "processed", "mi", "fiedler", "reduced", "synergy_raw", "causation_raw",
        "emergence_raw", "emergence_nan0", "integrated_raw",
    ]
    for row in manifest.itertuples(index=False):
        first = first_root / row.candidateId / f"M{int(row.matrixIndex):03d}.npz"
        replay = replay_root / row.candidateId / f"M{int(row.matrixIndex):03d}.npz"
        meta_a, array_a = load_npz(first)
        meta_b, array_b = load_npz(replay)
        array_results = {}
        for name in arrays_to_compare:
            present = name in array_a or name in array_b
            array_results[name] = (
                not present
                or (
                    name in array_a
                    and name in array_b
                    and array_sha256(array_a[name]) == array_sha256(array_b[name])
                )
            )
        passed = (
            meta_a["status"] == meta_b["status"]
            and meta_a.get("reason") == meta_b.get("reason")
            and meta_a.get("partition1") == meta_b.get("partition1")
            and meta_a.get("partition2") == meta_b.get("partition2")
            and all(array_results.values())
        )
        rows.append(
            {
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "firstStatus": meta_a["status"],
                "replayStatus": meta_b["status"],
                "arrayChecks": array_results,
                "exactReplayPassed": passed,
            }
        )
    return pd.DataFrame(rows)


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float | None, float | None, int]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None, None, len(x)
    result = spearmanr(x, y) if method == "spearman" else pearsonr(x, y)
    return float(result.statistic), float(result.pvalue), len(x)


def aggregate_scientific(
    manifest: pd.DataFrame, execution_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frozen = pd.read_parquet(S13Y_ROOT / "full_source_values.parquet")
    frozen = frozen[
        frozen["implementationId"].eq("PHIRL_REGULARIZED_SOURCE")
    ].set_index(["candidateId", "matrixIndex", "selectedSequenceIndex"])
    value_rows = []
    trajectory_rows = []
    failure_rows = []
    for row in manifest.itertuples(index=False):
        path = execution_root / row.candidateId / f"M{int(row.matrixIndex):03d}.npz"
        meta, arrays = load_npz(path)
        trajectory_rows.append(
            {
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "trajectoryId": row.trajectoryId,
                "status": meta["status"],
                "reason": meta.get("reason"),
                "selectedObservationCount": meta.get("selectedObservationCount"),
                "partition1Size": len(meta.get("partition1", [])),
                "partition2Size": len(meta.get("partition2", [])),
                "cpuSeconds": meta.get("cpuSeconds"),
                "wallSeconds": meta.get("wallSeconds"),
                "closureErrorMaximum": meta.get("closureErrorMaximum"),
                "sourceOutputSha256": sha256_file(path),
            }
        )
        if not meta["status"].startswith("ELIGIBLE"):
            failure_rows.append(
                {
                    "candidateId": row.candidateId,
                    "matrixIndex": int(row.matrixIndex),
                    "stage": "BGM_TRANSFER_PIPELINE",
                    "status": meta["status"],
                    "reason": meta.get("reason"),
                }
            )
            continue
        indices = np.arange(2, int(meta["selectedObservationCount"]), dtype=np.int64)
        for hypothesis, array_name in HYPOTHESIS_METRICS.items():
            values = arrays.get(array_name)
            if values is None or len(values) != len(indices):
                failure_rows.append(
                    {
                        "candidateId": row.candidateId,
                        "matrixIndex": int(row.matrixIndex),
                        "stage": hypothesis,
                        "status": "MISSING_OR_MISALIGNED_LOCAL_VALUES",
                        "reason": f"{array_name}:{None if values is None else len(values)} expected {len(indices)}",
                    }
                )
                continue
            for selected_index, value in zip(indices, values, strict=True):
                frozen_row = frozen.loc[(row.candidateId, int(row.matrixIndex), int(selected_index))]
                value_rows.append(
                    {
                        "candidateId": row.candidateId,
                        "matrixIndex": int(row.matrixIndex),
                        "trajectoryId": row.trajectoryId,
                        "hypothesisId": hypothesis,
                        "metricIdentity": array_name,
                        "temporalMode": "RETROSPECTIVE_COMPLETED_TRAJECTORY_FUTURE_DEPENDENT",
                        "selectedSequenceIndex": int(selected_index),
                        "metricValue": float(value),
                        "incomingCosineH": float(frozen_row["incomingCosineH"]),
                        "compositionChange": float(frozen_row["euclideanL2ClosedCompositionChange"]),
                        "frozenMolecularH090Label": bool(frozen_row["molecularH090Label"]),
                    }
                )
    values = pd.DataFrame(value_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    failures = pd.DataFrame(
        failure_rows,
        columns=["candidateId", "matrixIndex", "stage", "status", "reason"],
    )
    analysis_rows = []
    if not values.empty:
        for (candidate, matrix_index, hypothesis), frame in values.groupby(
            ["candidateId", "matrixIndex", "hypothesisId"], sort=True
        ):
            metric = frame["metricValue"].to_numpy(float)
            label = frame["frozenMolecularH090Label"].to_numpy(float)
            exact_h = frame["incomingCosineH"].to_numpy(float)
            change = frame["compositionChange"].to_numpy(float)
            for outcome, y in [
                ("FROZEN_H090_LABEL", label),
                ("EXACT_INCOMING_H", exact_h),
                ("COMPOSITION_CHANGE", change),
            ]:
                for method in ["spearman", "pearson"]:
                    estimate, pvalue, n = safe_corr(metric, y, method)
                    analysis_rows.append(
                        {
                            "candidateId": candidate,
                            "matrixIndex": int(matrix_index),
                            "hypothesisId": hypothesis,
                            "analysis": "LEVEL",
                            "controlOrOutcome": outcome,
                            "method": method.upper(),
                            "estimate": estimate,
                            "pValue": pvalue,
                            "n": n,
                        }
                    )
            if len(metric) > 1:
                delta = np.diff(metric)
                for outcome, y in [
                    ("FROZEN_H090_LABEL", label[1:]),
                    ("EXACT_INCOMING_H", exact_h[1:]),
                    ("COMPOSITION_CHANGE", change[1:]),
                ]:
                    for method in ["spearman", "pearson"]:
                        estimate, pvalue, n = safe_corr(delta, y, method)
                        analysis_rows.append(
                            {
                                "candidateId": candidate,
                                "matrixIndex": int(matrix_index),
                                "hypothesisId": hypothesis,
                                "analysis": "CHANGE",
                                "controlOrOutcome": outcome,
                                "method": method.upper(),
                                "estimate": estimate,
                                "pValue": pvalue,
                                "n": n,
                            }
                        )
            positive = metric[label == 1]
            negative = metric[label == 0]
            effect = float(np.mean(positive) - np.mean(negative)) if len(positive) and len(negative) else None
            mw_p = None
            if len(positive) and len(negative):
                mw_p = float(mannwhitneyu(positive, negative, alternative="two-sided").pvalue)
            analysis_rows.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": int(matrix_index),
                    "hypothesisId": hypothesis,
                    "analysis": "STATE_MEAN_DIFFERENCE",
                    "controlOrOutcome": "REPLICATOR_MINUS_DRIFT",
                    "method": "MEAN_AND_MANN_WHITNEY",
                    "estimate": effect,
                    "pValue": mw_p,
                    "n": int(len(metric)),
                }
            )
    analyses = pd.DataFrame(analysis_rows)
    return values, trajectories, analyses, failures


def bootstrap_and_summary(analyses: pd.DataFrame, replicates: int = 4096) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    boot_rows = []
    summary_columns = [
        "candidateId", "hypothesisId", "analysis", "controlOrOutcome", "method",
        "definedMatrices", "medianEstimate", "bootstrapLower95", "bootstrapUpper95",
        "positiveCount", "negativeCount", "significantCountNominal",
    ]
    bootstrap_columns = [
        "candidateId", "hypothesisId", "analysis", "controlOrOutcome", "method",
        "bootstrapReplicate", "medianEstimate",
    ]
    if analyses.empty:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=bootstrap_columns)
    groups = ["candidateId", "hypothesisId", "analysis", "controlOrOutcome", "method"]
    for key, frame in analyses.groupby(groups, sort=True):
        finite = frame.dropna(subset=["estimate"])
        estimates = finite["estimate"].to_numpy(float)
        estimate = float(np.median(estimates)) if len(estimates) else None
        rng = np.random.RandomState(derive_seed("bootstrap", *key))
        boot = np.full(replicates, np.nan)
        if len(estimates):
            for index in range(replicates):
                boot[index] = np.median(rng.choice(estimates, size=len(estimates), replace=True))
        lower = float(np.nanquantile(boot, 0.025)) if len(estimates) else None
        upper = float(np.nanquantile(boot, 0.975)) if len(estimates) else None
        summaries.append(
            {
                **dict(zip(groups, key, strict=True)),
                "definedMatrices": int(len(estimates)),
                "medianEstimate": estimate,
                "bootstrapLower95": lower,
                "bootstrapUpper95": upper,
                "positiveCount": int(np.sum(estimates > 0)),
                "negativeCount": int(np.sum(estimates < 0)),
                "significantCountNominal": int(
                    np.sum(finite["pValue"].notna() & (finite["pValue"] < 0.05))
                ),
            }
        )
        for index, value in enumerate(boot):
            boot_rows.append(
                {
                    **dict(zip(groups, key, strict=True)),
                    "bootstrapReplicate": index,
                    "medianEstimate": float(value) if np.isfinite(value) else None,
                }
            )
    return (
        pd.DataFrame(summaries, columns=summary_columns),
        pd.DataFrame(boot_rows, columns=bootstrap_columns),
    )


def adjudicate(
    execution: pd.DataFrame, summary: pd.DataFrame, replay: pd.DataFrame
) -> dict[str, Any]:
    eligible = execution["status"].eq("ELIGIBLE").sum()
    partial = execution["status"].eq("ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES").sum()
    failed = len(execution) - eligible - partial
    replay_pass = bool(replay["exactReplayPassed"].all())
    classes = ["AUTHOR_AMBIGUITY_UNRESOLVED", "NOT_PROMOTABLE"]
    narrative = "The transfer produced no eligible scientific values."
    if eligible + partial == 0:
        classes += ["EXPLORATORY_NON_SUPPORT", "POSSIBLE_PIPELINE_ARTIFACT"]
    else:
        label_rows = summary[
            summary["analysis"].eq("LEVEL")
            & summary["controlOrOutcome"].eq("FROZEN_H090_LABEL")
            & summary["method"].eq("SPEARMAN")
        ]
        control_rows = summary[
            summary["analysis"].eq("LEVEL")
            & summary["controlOrOutcome"].eq("EXACT_INCOMING_H")
            & summary["method"].eq("SPEARMAN")
        ]
        positive_both = False
        for hypothesis in HYPOTHESIS_METRICS:
            hrows = label_rows[label_rows["hypothesisId"].eq(hypothesis)]
            if len(hrows) == 2 and (hrows["medianEstimate"] > 0).all():
                positive_both = True
        if positive_both:
            classes += ["RETROSPECTIVE_ONLY_LEAD", "POSSIBLE_STABILITY_PROXY"]
            narrative = "At least one completed-fit source transfer had positive label association in both candidates; exact-H/stability controls remain co-primary and the result is retrospective only."
        else:
            classes += ["EXPLORATORY_NON_SUPPORT"]
            narrative = "No transferred scalar produced a directionally positive label association in both candidates."
        if not control_rows.empty and control_rows["medianEstimate"].abs().max() >= label_rows["medianEstimate"].abs().max():
            if "POSSIBLE_STABILITY_PROXY" not in classes:
                classes.append("POSSIBLE_STABILITY_PROXY")
    if not replay_pass:
        classes = ["LOOP_FAILED_CLOSED", "NOT_PROMOTABLE"]
        narrative = "Exact replay failed; L17 is failed closed."
    return {
        "researchStepId": "S19-L17",
        "versionedStepId": "E01-S19-L17-BREAKINGGRNMEMORIES-PHI-LINEAGE-TRANSFER-AUDIT-v1.0.0",
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW" if replay_pass else "LOOP_FAILED_CLOSED",
        "classifications": list(dict.fromkeys(classes)),
        "eligibleTrajectories": int(eligible),
        "partialNonfiniteTrajectories": int(partial),
        "failedTrajectories": int(failed),
        "exactReplayPassed": replay_pass,
        "narrative": narrative,
        "prospectiveEvidence": False,
        "causalEvidence": False,
        "authorCodeIdentity": False,
        "promotable": False,
    }


def write_source_audit(build: Path, source_manifest: dict[str, Any], config: dict[str, Any]) -> None:
    write_json(build / "source_snapshot_manifest.json", source_manifest)
    write_json(build / "breakinggrn_repository_tree.json", source_tree())
    write_csv(build / "breakinggrn_commit_history.csv", source_history())
    write_parquet(build / "breakinggrn_function_registry.parquet", function_registry())
    nx.write_graphml(build_dataflow(), build / "breakinggrn_phi_executable_dataflow.graphml")
    write_csv(build / "source_lineage_crosswalk.csv", lineage_crosswalk())
    write_json(
        build / "license_audit.json",
        {
            "repository": config["breakingGrnSnapshot"]["repository"],
            "commit": config["breakingGrnSnapshot"]["commit"],
            "licenseStatus": "NO_LICENSE_FILE_DETECTED",
            "licenseFilesFound": [],
            "redistributionPolicy": "NO_SOURCE_REDISTRIBUTION; RECORD HASHES, IDENTITIES, AND FINDINGS ONLY",
        },
    )
    write_json(
        build / "native_example_reproducibility.json",
        {
            "repositoryTestsPresent": False,
            "repositoryPhiFixturesPresent": False,
            "rawPhiInputTrajectoriesBundled": False,
            "currentPhiScriptOutputColumns": 14,
            "trackedInfoColumns": 18,
            "originalPhiScriptOutputColumnsBeforeExit": 20,
            "trackedOutputRegenerableFromVisibleExactRevision": False,
            "reason": "tracked info has five phases and two measures; original script declared six phases/two measures and exited before computation, while current script declares six phases/emergence only",
            "disposition": "NO_NATIVE_REGENERABLE_PHI_EXAMPLE; USE SYNTHETIC SOURCE_EQUIVALENCE_FIXTURES",
        },
    )
    write_csv(
        build / "metric_paper_crosswalk.csv",
        pd.DataFrame(
            [
                {
                    "identity": "BGM_CURRENT_EMERGENCE",
                    "formula": "synergy + two downward-causation atoms",
                    "publicSource": "phi.py current top-level output",
                    "paperRelationship": "prose/source naming compatible; displayed equation not uniquely compatible",
                    "l12Adjudication": "PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT",
                },
                {
                    "identity": "BGM_INFORMATION_INTEGRATED",
                    "formula": "corrected local_phi_r atom sum",
                    "publicSource": "information.compute_circuit_info; removed from current phi.py output",
                    "paperRelationship": "Phi-r naming compatible; displayed equation still differs on L12 fixture",
                    "l12Adjudication": "PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT",
                },
                {
                    "identity": "PAPER_DIRECT_WHOLE_MINUS_PARTS",
                    "formula": "displayed paper equation",
                    "publicSource": "not equal to either public scalar on L12 fixture",
                    "paperRelationship": "direct equation",
                    "l12Adjudication": "PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT",
                },
            ]
        ),
    )
    audit = """# BreakingGRNMemories source audit\n\nBreakingGRNMemories was pinned at commit `afe44231ad3ce915172cdb53a6b234bd76fcb6a5` (tree `56f66ab8b57a2c60e830370842926708eee0767d`). No license file, test suite, Phi fixture, or raw Phi input bundle was present. Source is not redistributed.\n\nThe numerical information core is the corrected IIGR lineage: corrected z-scoring, global-signal regression, lag-one residualization, slow bidirectional Gaussian MI summed across directions, an additive `1e-6` graph floor, unnormalized Fiedler strict-sign partition, arithmetic partition means, unregularized Gaussian entropy, the shared PhiID lattice, and both `emergence` and corrected `local_phi_r` identities. Current `phi.py` retains only nonfinite-to-zero `emergence`.\n\nThe tracked `info.txt` is not regenerable from any visible exact script state: the original script declared six phases and two measures but stopped before its loop; the current script declares six phases and one measure; the tracked file contains five phases and two measures. The repository does not specify a GARD adapter or a past-only/prefix refit. L17 therefore treats it as related-team lineage inspiration, never as proof of the unavailable author implementation.\n"""
    atomic_text(build / "source_audit.md", audit)


def update_root_ledgers(timestamp: str, classification: dict[str, Any], source_manifest: dict[str, Any]) -> None:
    candidate_path = S19_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    if candidates["candidateId"].astype(str).str.startswith("S19-L17-").any():
        raise RuntimeError("L17 candidate ledger rows already exist")
    additions = []
    start_order = int(candidates["registryOrder"].max())
    for offset, hypothesis in enumerate(HYPOTHESIS_METRICS, start=1):
        row = {column: None for column in candidates.columns}
        row.update(
            {
                "candidateId": f"S19-L17-{hypothesis}",
                "bundleId": "L17_BREAKINGGRN_PHI_LINEAGE_TRANSFER",
                "proposedSpecification": hypothesis,
                "registryOrder": start_order + offset,
                "selected": True,
                "selectionReason": "COMPLETE_DIRECT_SOURCE_LINEAGE_TRANSFER_REGISTERED_BEFORE_GARD_OUTCOMES",
                "sourceGrounding": 5,
                "paperFingerprintSpecificity": 3,
                "explanatoryLeverage": 4,
                "testability": 4,
                "computeEfficiency": 4,
                "crossCandidateDiscriminability": 4,
                "independenceFromPriorOutcomeSelection": 3,
                "branchCount": 3,
                "completedFitLeakage": 1,
                "deterministicHReuse": 0,
                "candidateSpecificSuccess": 0,
                "outcomeGuidedThresholdSelection": 0,
                "undefinedAuthorSemantics": 2,
                "frozenRank": offset,
                "rankingScore": float(20 - offset),
            }
        )
        additions.append(row)
    write_parquet(candidate_path, pd.concat([candidates, pd.DataFrame(additions)], ignore_index=True)[candidates.columns])

    source_path = S19_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    if sources["sourceId"].astype(str).str.startswith("L17_").any():
        raise RuntimeError("L17 source ledger rows already exist")
    source_rows = []
    entries = [
        (
            "L17_BREAKINGGRNMEMORIES_CURRENT_DEFAULT",
            "PUBLIC_GIT_REPOSITORY",
            "DIRECT_PUBLIC_CODE",
            "Pinned corrected-IIGR Phi lineage; current top-level exports emergence only; tracked Phi output is not regenerable from visible script states.",
            config_url := "https://github.com/pigozzif/BreakingGRNMemories",
            source_manifest["breakingGrnMemories"]["head"],
            source_manifest["breakingGrnMemories"]["tree"],
        ),
        (
            "L17_L12_METRIC_CROSSWALK",
            "FROZEN_INTERNAL_SOURCE_CROSSWALK",
            "DIRECT_FROZEN_E01_RESULT",
            "L12 proves public emergence, integrated and paper direct-WMS identities differ and author code remains required.",
            None,
            sha256_file(L12_ROOT / "metric_identity_adjudication.json"),
            None,
        ),
    ]
    for source_id, source_type, evidence, finding, url, commit, tree in entries:
        row = {column: None for column in sources.columns}
        row.update(
            {
                "sourceId": source_id,
                "sourceType": source_type,
                "evidenceClass": evidence,
                "finding": finding,
                "url": url,
                "repositoryIdentity": url or "Eidosoma frozen L12 artifact",
                "commitOrVersion": commit,
                "treeIdentity": tree,
                "retrievalDate": "2026-08-10",
                "licenseStatus": "NO_LICENSE_FILE_DETECTED" if url else "WORKSPACE_ARTIFACT",
                "redistributionStatus": "IDENTITY_AND_FINDING_ONLY" if url else "REFERENCE_ONLY",
                "retainedPath": str(SOURCE_ROOT if url else L12_ROOT / "metric_identity_adjudication.json"),
            }
        )
        source_rows.append(row)
    write_parquet(source_path, pd.concat([sources, pd.DataFrame(source_rows)], ignore_index=True)[sources.columns])

    ledger_path = S19_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    if ledger["loopId"].eq("S19-L17").any():
        raise RuntimeError("L17 self-improvement rows already exist")
    start = int(ledger["ledgerSequence"].max()) + 1
    ledger_rows = []
    for sequence, phase, belief, learned in [
        (
            start,
            "PRE_LOOP_SOURCE_LINEAGE_LOCK",
            "A latest related-team repository might expose a materially different Phi pipeline that explains the E01 mismatch.",
            "The exact repository, history, license state and three complete transfer identities were frozen before GARD outcomes.",
        ),
        (
            start + 1,
            "POST_LOOP_RESULT_AND_HUMAN_REVIEW_HANDOFF",
            "A source-equivalent transfer might yield independent Phi evidence on the clean S13Y cohort.",
            classification["narrative"],
        ),
    ]:
        row = {column: None for column in ledger.columns}
        row.update(
            {
                "appendOnly": True,
                "beliefBeforeLoop": belief,
                "failureOrAmbiguityTargeted": "Phi pipeline identity and transferability from a related-team public source lineage.",
                "informationGainRationale": "Pin and audit the complete source before a bounded unchanged transfer; retain exact H/stability controls and completed-fit caveat.",
                "learned": learned,
                "ledgerSequence": sequence,
                "loopId": "S19-L17",
                "motivatingEvidence": "BreakingGRNMemories current default branch, paper, PhiRL/IIGR, L12, and frozen S13Y evidence.",
                "proposedNextTest": "Mandatory human review; no later loop, S20 or E02 is active.",
                "recordPhase": phase,
                "remainingPlausibleHypotheses": "Unavailable author GARD implementation may differ in preprocessing, scalar, temporal fit, label or caller seed semantics.",
                "selectedHypotheses": ";".join(HYPOTHESIS_METRICS),
                "timestampUtc": timestamp,
                "weakenedHypotheses": "BreakingGRNMemories can be treated as proof of the unavailable GARD author pipeline.",
            }
        )
        ledger_rows.append(row)
    write_parquet(ledger_path, pd.concat([ledger, pd.DataFrame(ledger_rows)], ignore_index=True)[ledger.columns])

    markdown_path = S19_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    existing = markdown_path.read_text(encoding="utf-8")
    marker = "## S19-L17 — BreakingGRNMemories Phi-lineage transfer audit"
    if marker in existing:
        raise RuntimeError("L17 markdown ledger entry already exists")
    addition = f"""

{marker}

- **Belief before:** The latest related-team repository might expose a materially different Phi path capable of resolving E01's Phi mismatch.
- **Source evidence:** Current default branch pinned at `afe44231…`; corrected-IIGR preprocessing/MI/unregularized entropy; current top-level nonfinite-to-zero emergence; no GARD adapter, prefix mode, tests, raw Phi fixtures, or license file.
- **Execution:** Three complete scalar/entry-point transfers were locked before outcomes and applied unchanged to the frozen S13Y cohort after synthetic source-equivalence. Candidate 2 and candidate 3 remained separate; exact H, composition change, and the frozen H>0.9 label remained co-primary controls.
- **What was learned:** {classification['narrative']}
- **Boundary:** All results are exploratory and completed-fit retrospective. No author-code identity, prospective prediction, intervention, causal-control, S20, E02, L18, or report generation follows automatically.
- **Next action:** mandatory human review.
"""
    atomic_text(markdown_path, existing.rstrip() + addition)

    loop_path = S19_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(loop_path.read_text(encoding="utf-8"))
    if any(row.get("loopId") == "S19-L17" for row in registry["loops"]):
        raise RuntimeError("L17 loop registry row already exists")
    registry["loops"].append(
        {
            "loopId": "S19-L17",
            "versionedLoopId": classification["versionedStepId"],
            "status": classification["status"],
            "authorized": True,
            "completed": True,
            "humanReviewRequiredAfter": True,
            "classification": classification["classifications"],
            "newMatrices": 0,
            "newTrajectories": 0,
            "nextStepActive": False,
        }
    )
    registry["laterLoopsAuthorized"] = False
    registry["s20Status"] = "DEFINED_INACTIVE"
    registry["proposedNextLoopActive"] = False
    registry["proposedNextLoopTheme"] = "MANDATORY_HUMAN_REVIEW"
    write_yaml(loop_path, registry)

    review_path = S19_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["history"].append(
        {
            "decision": "AUTHORIZE_EXACTLY_ONE_L17_BREAKINGGRNMEMORIES_PHI_LINEAGE_TRANSFER_AUDIT",
            "loopId": "S19-L17",
            "recordedAtUtc": timestamp,
            "source": "explicit_human_direction",
            "scope": classification["versionedStepId"],
            "result": classification["classifications"],
            "status": "CONSUMED_AND_RETURNED_FOR_MANDATORY_REVIEW",
            "nextLoopAuthorized": False,
            "s20Activated": False,
        }
    )
    review["pendingDecision"] = "POST_S19_L17_MANDATORY_HUMAN_REVIEW_REQUIRED"
    write_json(review_path, review)

    status_path = S19_ROOT / "s19_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "currentLoop": "S19-L17",
            "status": "AWAITING_MANDATORY_HUMAN_REVIEW",
            "lastCompletedLoop": "S19-L17",
            "nextLoopAuthorized": False,
            "s20Status": "DEFINED_INACTIVE",
            "updatedAtUtc": timestamp,
        }
    )
    write_json(status_path, status)


def artifact_manifest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "artifact_manifest.json"):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": "eidosoma.e01.s19.l17.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(files),
        "totalBytes": sum(row["bytes"] for row in files),
        "files": files,
    }


def report_text(
    classification: dict[str, Any],
    execution: pd.DataFrame,
    summary: pd.DataFrame,
    fixture_results: pd.DataFrame,
    runtime: dict[str, Any],
) -> str:
    classes = ", ".join(f"`{item}`" for item in classification["classifications"])
    status_counts = execution["status"].value_counts().to_dict()
    if summary.empty:
        label = pd.DataFrame()
        hcontrol = pd.DataFrame()
    else:
        label = summary[
            summary["analysis"].eq("LEVEL")
            & summary["controlOrOutcome"].eq("FROZEN_H090_LABEL")
            & summary["method"].eq("SPEARMAN")
        ].copy()
        hcontrol = summary[
            summary["analysis"].eq("LEVEL")
            & summary["controlOrOutcome"].eq("EXACT_INCOMING_H")
            & summary["method"].eq("SPEARMAN")
        ].copy()
    lines = [
        "# S19-L17 — BreakingGRNMemories Phi Lineage Transfer Audit",
        "",
        "## Chief/human handoff",
        "",
        f"- **Step:** `E01-S19-L17-BREAKINGGRNMEMORIES-PHI-LINEAGE-TRANSFER-AUDIT-v1.0.0`",
        f"- **Status:** `{classification['status']}`",
        f"- **Outcome classification:** {classes}",
        f"- **Validation:** source-equivalence fixtures `{int(fixture_results['fixturePassed'].sum())}/{len(fixture_results)}` passed; scientific replay `{int(classification['exactReplayPassed'])}`; immutable prior validation passed.",
        "- **Artifacts:** source snapshot/history/license audit, executable dataflow, lineage/function/metric crosswalks, frozen hypothesis registry, fixture evidence, 200-trajectory conditional transfer evidence, controls, bootstrap summaries, replay/runtime/storage/hash manifests, and this report.",
        f"- **Lay summary:** {classification['narrative']}",
        "- **Caveat:** BreakingGRNMemories is related-team source-lineage inspiration, not the unavailable GARD author implementation. Every scientific value is completed-fit and retrospective; the frozen label remains exactly determined by adjacent H.",
        "- **Recommended next action:** mandatory human review. No L18, S20, E02, author contact, prediction, intervention, confirmation, or report generation is active.",
        "",
        "## Authoritative human direction preserved verbatim",
        "",
        "> We have plenty of time and we are in exploratory mode. Check this repo: https://github.com/pigozzif/BreakingGRNMemories",
        ">",
        "> This is the latest code we have from the team that did the paper we're trying to replicate and it has phi work as well which has been what we are not able to replicate - we can replicate the replicators, even too well - let this be an inspiration for L17",
        "",
        "## Frozen question",
        "",
        "Does the latest public BreakingGRNMemories Phi lineage specify a complete transferable pipeline, and—after source equivalence—does an unchanged transfer to the frozen S13Y cohort recover independent Phi evidence that was absent from prior E01 branches?",
        "",
        "## Source audit anchor results",
        "",
        "- Default branch `master` was frozen at commit `afe44231ad3ce915172cdb53a6b234bd76fcb6a5`, tree `56f66ab8b57a2c60e830370842926708eee0767d`.",
        "- No license file, unit tests, Phi fixtures, or raw Phi input trajectories are present; public source is not redistributed.",
        "- `information.py` is the corrected-IIGR lineage plus preprocessing/circuit wrappers. It differs materially from PhiRL through GSR and AR residualization, slow summed bidirectional MI, no active-variable filter, and unregularized covariance.",
        "- Current `phi.py` exports only nonfinite-to-zero `emergence`; `information.compute_circuit_info` also exposes raw `emergence` and corrected `integrated/local_phi_r`.",
        "- The tracked `info.txt` cannot be regenerated from any visible exact script state because its phase/measure schema conflicts with both the original and current scripts.",
        "- No GARD adapter, self-replicator label, prefix refit, prediction model, or intervention scorer exists in this repository.",
        "",
        "## Registered transfer hypotheses",
        "",
        "Exactly three were frozen before GARD outcomes: current `phi.py` nonfinite-to-zero emergence, `optima.py`/information raw emergence, and information integrated/local-Phi-r. They share one source computation but retain distinct scalar and numerical identities. No prefix mode was run because the source does not specify one.",
        "",
        "## Source equivalence",
        "",
        fixture_results.to_markdown(index=False),
        "",
        "Synthetic fixtures were evaluated against the unmodified pinned public functions in an isolated process using the repository's exact NumPy/SciPy/NetworkX versions. The GARD transfer used the safe-JSON clean-room implementation already validated against the corrected IIGR lineage.",
        "",
        "## Frozen cohort and execution",
        "",
        "The exact S13Y 100-shared-matrix/200-trajectory candidate-2/candidate-3 cohort was used. No matrix, trajectory, label, threshold, exposure, feature tensor, model, or intervention was generated. Paper-frozen additive-0.5 closure, CLR99 and selected molecular clock were retained. Six workers used one numerical-library thread each under CPU float64.",
        "",
        f"Trajectory status counts: `{canonical_json(status_counts)}`.",
        "",
        "## Primary run-level results",
        "",
    ]
    if label.empty:
        lines.extend(["No eligible run-level label correlations were produced.", ""])
    else:
        table = label[
            ["candidateId", "hypothesisId", "definedMatrices", "medianEstimate", "bootstrapLower95", "bootstrapUpper95", "positiveCount", "negativeCount"]
        ].merge(
            hcontrol[
                ["candidateId", "hypothesisId", "medianEstimate"]
            ].rename(columns={"medianEstimate": "medianExactHControlRho"}),
            on=["candidateId", "hypothesisId"],
            how="left",
        )
        lines.extend([table.to_markdown(index=False), ""])
    lines.extend(
        [
            "Exact incoming H, continuous composition change, and the frozen adjacent-H label were retained side by side. A label association is not independent replication evidence merely because it resembles the paper: `Y=I(H>0.9)` exactly, and completed-fit BGM values use the full future suffix.",
            "",
            "## Metric and temporal interpretation",
            "",
            "L12's metric adjudication remains unchanged: the paper equation, public `emergence`, and public `integrated/local_phi_r` are not algebraically interchangeable. BreakingGRNMemories strengthens the public lineage for both names but does not resolve which GARD scalar the authors used. Its public pipeline is global/completed-fit; L17 therefore supplies no future-suffix-independent or prospective evidence.",
            "",
            "## Numerical and provenance findings",
            "",
            "- CPU float64 was authoritative; GPU time was zero.",
            "- The source's unregularized Gaussian entropy was retained. Source exceptions and partial nonfinite arrays were status-bearing and never repaired.",
            "- The public caller does not fully seed Python, NumPy, or the NetworkX Fiedler initialization. L17's domain-separated NumPy seed is an explicit exact-replay wrapper, not evidence of author seed identity.",
            "- Every unit was recomputed into a separate replay cache and compared at array-hash, partition, status, and reason level.",
            f"- Runtime: `{runtime['wallSeconds']:.3f}` wall seconds, `{runtime['scientificCpuHours']:.6f}` reported worker CPU-hours; retained artifacts and temporary cache remained below their hard ceilings.",
            "",
            "## Limitations",
            "",
            "This is a post hoc exploratory transfer from a related paper and team lineage. The repository is newer than the target preprint, does not contain the GARD analysis, lacks a license file and reproducible Phi fixture, and has a tracked-output provenance inconsistency. The paper-to-BGM CLR adapter and replay seed wrapper are explicit reconstruction choices. Neither a favorable nor unfavorable transfer can identify the unavailable author code.",
            "",
            "## Mandatory human-review boundary",
            "",
            "Stop here. All L17 outputs are frozen. No L18, S20, E02, author contact, new simulation, prediction, intervention, confirmation, or report-bundle generation begins automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    started_wall = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["researchStepId"] != "S19-L17":
        raise RuntimeError("wrong L17 configuration")
    lock = require_clean_pushed_lock()
    source_identity = verify_source(config)
    if BUILD_ROOT.exists():
        raise RuntimeError(f"fresh L17 build required; path exists: {BUILD_ROOT}")
    BUILD_ROOT.mkdir(parents=True)

    prior_before = hash_tree(immutable_roots())
    prior_digest = hashlib.sha256(canonical_json(prior_before).encode()).hexdigest()
    write_json(
        BUILD_ROOT / "immutable_prior_validation.json",
        {
            "status": "BASELINE_CAPTURED_PRE_OUTCOME",
            "rootCount": len(immutable_roots()),
            "fileCount": len(prior_before),
            "aggregateSha256": prior_digest,
            "files": prior_before,
        },
    )
    source_manifest = {
        "capturedAtUtc": utc_now(),
        "breakingGrnMemories": source_identity,
        "PhiRL": git_identity(PHIRL_ROOT),
        "IIGR": git_identity(IIGR_ROOT),
        "historicalGARD": git_identity(GARD_ROOT),
        "paper": {"path": str(PAPER_PDF), "sha256": sha256_file(PAPER_PDF)},
        "paperMarkdown": {"path": str(PAPER_MARKDOWN), "sha256": sha256_file(PAPER_MARKDOWN)},
        "safeLattice": {"path": str(SAFE_LATTICE), "sha256": sha256_file(SAFE_LATTICE)},
        "l12MetricAdjudication": {
            "path": str(L12_ROOT / "metric_identity_adjudication.json"),
            "sha256": sha256_file(L12_ROOT / "metric_identity_adjudication.json"),
        },
        "implementationLock": lock,
    }
    write_source_audit(BUILD_ROOT, source_manifest, config)
    shutil.copy2(args.config, BUILD_ROOT / "preregistration.yaml")
    amendment_path = REPO_ROOT / "configs/e01/s19_l17_technical_amendment_001.json"
    if amendment_path.exists():
        shutil.copy2(amendment_path, BUILD_ROOT / amendment_path.name)
    decision = """# L17 decision record\n\n## Authoritative human direction (verbatim)\n\n“We have plenty of time and we are in exploratory mode. Check this repo: https://github.com/pigozzif/BreakingGRNMemories\n\nThis is the latest code we have from the team that did the paper we're trying to replicate and it has phi work as well which has been what we are not able to replicate - we can replicate the replicators, even too well - let this be an inspiration for L17”.\n\nOnly `E01-S19-L17-BREAKINGGRNMEMORIES-PHI-LINEAGE-TRANSFER-AUDIT-v1.0.0` is authorized. BreakingGRNMemories is source-lineage inspiration, not proof of the unavailable GARD author implementation. All prior artifacts and classifications remain immutable.\n"""
    atomic_text(BUILD_ROOT / "decision_record.md", decision)
    gate = hypothesis_gate(config)
    write_csv(BUILD_ROOT / "hypothesis_completeness_gate.csv", gate)
    write_yaml(
        BUILD_ROOT / "transfer_hypothesis_registry.yaml",
        {"hypotheses": config["transferHypotheses"], "frozenBeforeGardOutcomes": True},
    )
    write_json(
        BUILD_ROOT / "implementation_lock.json",
        {
            "lockCommit": lock["head"],
            "pushed": True,
            "clean": True,
            "configSha256": sha256_file(args.config),
            "registeredHypotheses": gate.to_dict("records"),
            "gardOutcomeAccessedAtLock": False,
        },
    )

    fixture_results, source_equivalence = run_fixtures(config, CACHE_ROOT / "fixtures")
    write_parquet(BUILD_ROOT / "fixture_results.parquet", fixture_results)
    write_parquet(BUILD_ROOT / "source_equivalence_results.parquet", source_equivalence)
    write_json(
        BUILD_ROOT / "fixture_manifest.json",
        {
            "fixtures": config["fixtures"]["ids"],
            "allPassed": bool(fixture_results["fixturePassed"].all()),
            "isolatedPublicSourceAdapter": str(SOURCE_ADAPTER),
            "publicSourceLoadedOnlyInIsolatedProcess": True,
        },
    )
    if not gate["registeredForExecution"].any():
        raise RuntimeError("no complete L17 transfer hypothesis; audit-only path not implemented")

    manifest = task_manifest()
    write_parquet(BUILD_ROOT / "input_trajectory_manifest.parquet", manifest)
    task_files = make_task_files(manifest, CACHE_ROOT / "tasks")
    benchmark_keys = [(str(c), int(i)) for c, i in config["execution"]["benchmarkUnits"]]
    benchmark = run_task_set(
        benchmark_keys,
        task_files,
        CACHE_ROOT / "benchmark",
        int(config["execution"]["workers"]),
    )
    write_parquet(BUILD_ROOT / "benchmark_results.parquet", benchmark)
    mean_cpu = float(benchmark["cpuSeconds"].mean())
    mean_wall = float(benchmark["subprocessWallSeconds"].mean())
    projected_cpu_hours = mean_cpu * len(manifest) * 2 / 3600.0
    projected_wall_hours = mean_wall * len(manifest) * 2 / int(config["execution"]["workers"]) / 3600.0
    write_json(
        BUILD_ROOT / "benchmark_projection.json",
        {
            "benchmarkUnits": len(benchmark),
            "meanWorkerCpuSeconds": mean_cpu,
            "meanWorkerWallSeconds": mean_wall,
            "projectedTwoPassCpuHours": projected_cpu_hours,
            "projectedTwoPassWallHours": projected_wall_hours,
            "cpuCeilingHours": config["resources"]["cpuHoursMaximum"],
            "wallCeilingHours": config["resources"]["wallHoursMaximum"],
            "reserveFraction": config["resources"]["reserveFraction"],
            "passed": projected_cpu_hours <= 90 and projected_wall_hours <= 64.8,
        },
    )
    if projected_cpu_hours > 90 or projected_wall_hours > 64.8:
        raise RuntimeError("L17 benchmark exceeds ceiling after validation reserve")

    keys = [(str(row.candidateId), int(row.matrixIndex)) for row in manifest.itertuples(index=False)]
    first_execution = run_task_set(
        keys,
        task_files,
        CACHE_ROOT / "execution/first",
        int(config["execution"]["workers"]),
    )
    replay_execution = run_task_set(
        keys,
        task_files,
        CACHE_ROOT / "execution/replay",
        int(config["execution"]["workers"]),
    )
    replay = compare_replay(
        CACHE_ROOT / "execution/first", CACHE_ROOT / "execution/replay", manifest
    )
    write_parquet(BUILD_ROOT / "gard_execution_status.parquet", first_execution)
    write_parquet(BUILD_ROOT / "gard_replay_execution_status.parquet", replay_execution)
    write_parquet(BUILD_ROOT / "replay_validation.parquet", replay)
    if not replay["exactReplayPassed"].all():
        raise RuntimeError("L17 exact full-cohort replay failed")

    values, trajectories, analyses, failures = aggregate_scientific(
        manifest, CACHE_ROOT / "execution/first"
    )
    summary, bootstrap = bootstrap_and_summary(
        analyses, int(config["execution"]["bootstrapReplicates"])
    )
    write_parquet(BUILD_ROOT / "gard_phi_values.parquet", values)
    write_parquet(BUILD_ROOT / "trajectory_results.parquet", trajectories)
    write_parquet(BUILD_ROOT / "runwise_control_results.parquet", analyses)
    write_parquet(BUILD_ROOT / "candidate_results.parquet", summary)
    write_parquet(BUILD_ROOT / "bootstrap_results.parquet", bootstrap)
    write_csv(BUILD_ROOT / "failure_ledger.csv", failures)
    if summary.empty:
        negative = summary.copy()
    else:
        negative = summary[
            summary["medianEstimate"].isna()
            | ((summary["bootstrapLower95"] <= 0) & (summary["bootstrapUpper95"] >= 0))
        ].copy()
    write_csv(BUILD_ROOT / "negative_result_ledger.csv", negative)
    classification = adjudicate(trajectories, summary, replay)
    write_json(BUILD_ROOT / "classification.json", classification)

    prior_after = hash_tree(immutable_roots())
    after_digest = hashlib.sha256(canonical_json(prior_after).encode()).hexdigest()
    if prior_digest != after_digest or prior_before != prior_after:
        raise RuntimeError("immutable prior changed during L17")
    write_json(
        BUILD_ROOT / "immutable_prior_validation.json",
        {
            "status": "PASS",
            "rootCount": len(immutable_roots()),
            "fileCount": len(prior_before),
            "preOutcomeAggregateSha256": prior_digest,
            "postOutcomeAggregateSha256": after_digest,
            "unchanged": True,
            "files": prior_after,
        },
    )

    worker_cpu_seconds = float(first_execution["cpuSeconds"].sum() + replay_execution["cpuSeconds"].sum())
    runtime = {
        "startedAtUtc": utc_now(),
        "wallSeconds": time.perf_counter() - started_wall,
        "scientificWorkerCpuSeconds": worker_cpu_seconds,
        "scientificCpuHours": worker_cpu_seconds / 3600.0,
        "workers": int(config["execution"]["workers"]),
        "threadsPerWorker": 1,
        "gpuHours": 0,
        "python": platform.python_version(),
        "exactWorkerPython": benchmark.iloc[0]["workerPython"],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "exactEnvironmentRequirements": (SOURCE_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines(),
        "benchmarkProjectedCpuHours": projected_cpu_hours,
        "benchmarkProjectedWallHours": projected_wall_hours,
    }
    write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    cache_bytes = sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
    retained_bytes = sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file())
    storage = {
        "status": "PASS",
        "retainedBytesBeforeManifest": retained_bytes,
        "retainedGiB": retained_bytes / (1024**3),
        "retainedGiBCeiling": 25,
        "temporaryBytes": cache_bytes,
        "temporaryGiB": cache_bytes / (1024**3),
        "temporaryGiBCeiling": 75,
    }
    if storage["retainedGiB"] > 25 or storage["temporaryGiB"] > 75:
        raise RuntimeError("L17 storage ceiling exceeded")
    write_json(BUILD_ROOT / "storage_validation.json", storage)
    write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "status": "PASS",
            "trajectoryUnits": 200,
            "exactReplayUnits": int(replay["exactReplayPassed"].sum()),
            "sourceFixtureReplayPassed": bool(fixture_results["exactSourceReplayPassed"].all()),
            "scientificArraysExact": True,
            "reportRegeneratedFromMachineReadableTables": True,
        },
    )
    atomic_text(
        BUILD_ROOT / "research_step_full_results.md",
        report_text(classification, trajectories, summary, fixture_results, runtime),
    )
    atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        "\n".join(
            [
                "# S19-L17 one-page decision summary",
                "",
                f"**Status:** `{classification['status']}`",
                "",
                f"**Classifications:** {', '.join(classification['classifications'])}",
                "",
                classification["narrative"],
                "",
                "BreakingGRNMemories is a corrected-IIGR related-team lineage, not GARD author code. Its tracked Phi table is not reproducible from visible exact script states, its current scalar is emergence, and it supplies no GARD adapter or prefix mode. Every L17 GARD value is completed-fit and retrospective; exact H/stability controls remain co-primary.",
                "",
                "**Decision boundary:** stop for mandatory human review. No L18, S20, E02, author contact, prediction, intervention, confirmation, or report generation is active.",
                "",
            ]
        ),
    )
    write_json(BUILD_ROOT / "artifact_manifest.json", artifact_manifest(BUILD_ROOT))

    if OUTPUT_ROOT.exists():
        raise RuntimeError("L17 final output already exists")
    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BUILD_ROOT, OUTPUT_ROOT)
    final_manifest = artifact_manifest(OUTPUT_ROOT)
    write_json(OUTPUT_ROOT / "artifact_manifest.json", final_manifest)

    timestamp = utc_now()
    update_root_ledgers(timestamp, classification, source_manifest)
    atomic_text(S19_ROOT / "research_step_full_results.md", (OUTPUT_ROOT / "research_step_full_results.md").read_text(encoding="utf-8"))
    write_json(S19_ROOT / "artifact_manifest.json", artifact_manifest(S19_ROOT))
    print(json.dumps(classification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
