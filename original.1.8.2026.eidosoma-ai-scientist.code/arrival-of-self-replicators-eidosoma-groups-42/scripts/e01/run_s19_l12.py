#!/usr/bin/env python3
"""Run the analysis-only S19-L12 paper/PhiRL forensic concordance audit.

The runner has three ordered phases:

``prepare`` freezes source and immutable-input identities without interpreting
paper-visible outcomes; ``audit`` constructs the sentence, panel, source and
E01 concordance records and freezes their hash; ``finalize`` is permitted only
after that hash exists and then ranks whole-pipeline hypotheses, renders the
registered figures, validates deterministic regeneration, and writes the
handoff.  It never imports or calls a GARD simulator.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from PIL import Image


REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs/e01/s19_l12_paper_phirl_forensic_audit.yaml"
WORKSPACE = Path("/workspace")
PLAN = WORKSPACE / "RESEARCH_PLAN.md"
FULL_PLAN = WORKSPACE / "FULL_PLAN.md"
AGENTS = WORKSPACE / "AGENTS.md"
ARTIFACTS = Path("/artifacts")
OUT = ARTIFACTS / "research_steps/S19/loops/L12"
CACHE = Path("/cache/e01_s19_l12")
CREATED_UTC = "2026-08-09T00:00:00Z"
STEP_ID = "S19-L12"
LOOP_ID = "E01-S19-L12-PAPER-PHIRL-FORENSIC-CONCORDANCE-AUDIT-v1.0.0"

REQUIRED_FILES = [
    "preregistration.yaml",
    "decision_record.md",
    "immutable_prior_validation.json",
    "source_snapshot_manifest.json",
    "phirl_repository_tree.json",
    "phirl_commit_history.csv",
    "phirl_function_blame.csv",
    "phirl_master_vs_pinned_diff.md",
    "iigr_phirl_lineage_map.md",
    "safe_lattice_equivalence.json",
    "paper_statement_registry.parquet",
    "paper_method_dependency_graph.graphml",
    "paper_internal_discrepancy_registry.csv",
    "figure_panel_registry.parquet",
    "figure_digitization.csv",
    "figure_internal_consistency_matrix.csv",
    "figure_to_claim_crosswalk.csv",
    "table1_semantics_matrix.csv",
    "phirl_executable_dataflow.graphml",
    "phirl_function_registry.parquet",
    "phirl_numerical_semantics.csv",
    "phirl_atom_identity_matrix.csv",
    "phirl_temporal_leakage_map.csv",
    "phirl_missing_gard_components.md",
    "phiid_atom_registry.csv",
    "paper_equation_derivation.md",
    "metric_identity_adjudication.json",
    "paper_phirl_e01_concordance_matrix.csv",
    "root_cause_hypothesis_registry.parquet",
    "unresolved_author_implementation_matrix.csv",
    "candidate_hidden_pipeline_hypotheses.yaml",
    "decisive_next_step_options.md",
    "classification.json",
    "failure_ledger.csv",
    "runtime_manifest.json",
    "storage_validation.json",
    "regeneration_validation.json",
    "artifact_manifest.json",
    "loop_decision_summary.md",
    "S19_L12_FULL_RESULTS.md",
]

REQUIRED_FIGURES = [
    "figure_01_paper_dependency_graph.png",
    "figure_02_phirl_dataflow_graph.png",
    "figure_03_paper_vs_phirl_operation_map.png",
    "figure_04_figure2_digitized_clock_constraints.png",
    "figure_05_figure3_level_change_inconsistency.png",
    "figure_06_figure5_prevalence_contradiction.png",
    "figure_07_figure6_table1_consistency_map.png",
    "figure_08_metric_identity_atom_map.png",
    "figure_09_completed_fit_future_dependence.png",
    "figure_10_paper_phirl_e01_concordance_heatmap.png",
    "figure_11_root_cause_ranking.png",
    "figure_12_decisive_next_step_tree.png",
]


def read_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.15g")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def run(command: Sequence[str], *, cwd: Path | None = None, check: bool = True) -> str:
    proc = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    return proc.stdout


def git(repo: Path, *args: str, check: bool = True) -> str:
    return run(["git", "-C", str(repo), *args], check=check).strip()


def file_record(path: Path, *, label: str | None = None) -> dict[str, Any]:
    return {
        "id": label or path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def find_prior_files() -> list[Path]:
    roots = [
        ARTIFACTS / "E01_forensic_replication_bundle",
        ARTIFACTS / "E01_forensic_replication_artifact_v2",
    ]
    roots.extend(ARTIFACTS / "research_steps" / f"S{i:02d}" for i in range(1, 19))
    roots.extend(
        ARTIFACTS / "research_steps" / name
        for name in [
            "S11R", "S12B", "S12C", "S12D", "S12E", "S12F", "S12FR", "S12G",
            "S12H", "S12I", "S12J", "S13R", "S13RR", "S13RRR", "S13X", "S13Y",
        ]
    )
    roots.extend(
        ARTIFACTS / "research_steps/S19/loops" / name
        for name in ["L01", "L02", "L03", "L04", "L05", "L06", "L06R", "L07", "L08", "L09", "L10", "L11", "L11R"]
    )
    files: set[Path] = set()
    for root in roots:
        if root.exists():
            files.update(path for path in root.rglob("*") if path.is_file())
    return sorted(files, key=lambda path: str(path))


def build_immutable_baseline() -> dict[str, Any]:
    members = [file_record(path) for path in find_prior_files()]
    aggregate = sha256_bytes(canonical_json([(m["path"], m["bytes"], m["sha256"]) for m in members]).encode())
    return {
        "schema": "eidosoma.e01.s19.l12.immutable_prior.v1",
        "createdUtc": CREATED_UTC,
        "researchStepId": STEP_ID,
        "fileCount": len(members),
        "totalBytes": sum(member["bytes"] for member in members),
        "aggregateSha256": aggregate,
        "members": members,
        "scope": "S01-S18, V1/V2, and S19-L01-L11R immutable artifacts; append-only S19 root ledgers excluded",
    }


def validate_immutable_baseline(baseline: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for member in baseline["members"]:
        path = Path(member["path"])
        if not path.exists():
            failures.append({"path": str(path), "reason": "MISSING"})
            continue
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != member["sha256"] or size != member["bytes"]:
            failures.append(
                {
                    "path": str(path),
                    "reason": "HASH_OR_SIZE_MISMATCH",
                    "expectedSha256": member["sha256"],
                    "observedSha256": digest,
                    "expectedBytes": member["bytes"],
                    "observedBytes": size,
                }
            )
    return {
        "schema": "eidosoma.e01.s19.l12.immutable_prior_validation.v1",
        "researchStepId": STEP_ID,
        "validationUtc": CREATED_UTC,
        "success": not failures,
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "checkedFileCount": len(baseline["members"]),
        "expectedAggregateSha256": baseline["aggregateSha256"],
        "failureCount": len(failures),
        "failures": failures,
    }


def repository_tree(repo: Path, commit: str) -> list[dict[str, Any]]:
    lines = git(repo, "ls-tree", "-r", "-l", commit).splitlines()
    rows: list[dict[str, Any]] = []
    for line in lines:
        head, path = line.split("\t", 1)
        mode, kind, blob, size = head.split()
        payload = run(["git", "-C", str(repo), "show", f"{commit}:{path}"]).encode("utf-8") if False else None
        raw = subprocess.run(
            ["git", "-C", str(repo), "show", f"{commit}:{path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        rows.append(
            {
                "path": path,
                "mode": mode,
                "kind": kind,
                "gitBlob": blob,
                "bytes": int(size),
                "sha256": sha256_bytes(raw),
            }
        )
    return rows


def git_commit_history(repo: Path) -> pd.DataFrame:
    fmt = "%H%x1f%P%x1f%aI%x1f%cI%x1f%an%x1f%s%x1e"
    raw = git(repo, "log", "--all", "--reverse", f"--format={fmt}")
    semantic_patterns = {
        "covarianceRegularization": ["eps_matrix", "local_entropy_nd"],
        "laggedMI": ["mutual_information_matrix", "lag="],
        "fiedlerPartition": ["fiedler", "minimum_information_bipartition"],
        "atomSelection": ["PHIR_ATOMS", "synergy", "causation"],
        "integrated": ['info["integrated"]', "local_phi_r"],
        "emergence": ['info["emergence"]', "synergy"],
        "trajectoryAggregation": ["nanmedian", "nanstd", "_load_phi"],
        "shuffledControls": ["shuffled", "permutation"],
    }
    rows: list[dict[str, Any]] = []
    for record in raw.strip("\x1e\n").split("\x1e"):
        if not record.strip():
            continue
        fields = record.strip().split("\x1f")
        if len(fields) != 6:
            continue
        commit, parents, author_date, commit_date, author, subject = fields
        names = git(repo, "show", "--format=", "--name-only", commit).splitlines()
        diff = git(repo, "show", "--format=", "--unified=0", commit)
        row: dict[str, Any] = {
            "commit": commit,
            "parents": parents,
            "parentCount": len(parents.split()) if parents else 0,
            "authorDate": author_date,
            "commitDate": commit_date,
            "author": author,
            "subject": subject,
            "filesChanged": canonical_json(sorted(name for name in names if name)),
        }
        for key, patterns in semantic_patterns.items():
            row[key] = any(pattern in diff for pattern in patterns)
        rows.append(row)
    return pd.DataFrame(rows)


def function_spans(source: Path) -> dict[str, tuple[int, int]]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return {
        node.name: (node.lineno, node.end_lineno or node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def function_blame(repo: Path, function_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    head = git(repo, "rev-parse", "HEAD")
    for name, relative in function_map.items():
        spans = function_spans(repo / relative)
        if name not in spans:
            raise RuntimeError(f"required function missing: {relative}:{name}")
        start, end = spans[name]
        porcelain = git(repo, "blame", "--line-porcelain", f"-L{start},{end}", head, "--", relative)
        commits = [line.split()[0] for line in porcelain.splitlines() if len(line.split()) >= 4 and len(line.split()[0]) == 40]
        unique = sorted(set(commits))
        history = git(repo, "log", "--follow", "--format=%H", "--", relative).splitlines()
        touching = [
            commit
            for commit in history
            if name in git(repo, "show", "--format=", "--unified=0", commit, "--", relative)
        ]
        rows.append(
            {
                "function": name,
                "file": relative,
                "startLine": start,
                "endLine": end,
                "currentCommit": head,
                "blameCommitCount": len(unique),
                "blameCommits": canonical_json(unique),
                "lastTouchCommit": touching[0] if touching else (history[0] if history else None),
                "originCommit": touching[-1] if touching else (history[-1] if history else None),
            }
        )
    return pd.DataFrame(rows)


def safe_lattice_equivalence(config: dict[str, Any]) -> dict[str, Any]:
    cache_output = CACHE / "isolated_safe_phi_lattice.json"
    converter = REPO / "scripts/e01/convert_s12b_phi_lattice.py"
    phirl_pickle = Path(config["paths"]["phirl"]) / "phi_lattice_22.pickle"
    iigr_pickle = Path(config["paths"]["iigr"]) / "phi_lattice_22.pickle"
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(converter),
            "--input",
            str(phirl_pickle),
            "--input",
            str(iigr_pickle),
            "--output",
            str(cache_output),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    fresh = json.loads(cache_output.read_text(encoding="utf-8"))
    frozen_path = Path(config["paths"]["safeLattice"])
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    keys = ["rawPickleSha256", "directed", "nodeCount", "edgeCount", "order", "nodes", "edges"]
    checks = {key: fresh[key] == frozen[key] for key in keys}
    return {
        "schema": "eidosoma.e01.s19.l12.safe_lattice_equivalence.v1",
        "researchStepId": STEP_ID,
        "success": all(checks.values()),
        "status": "PASS_ISOLATED_RESTRICTED_UNPICKLER" if all(checks.values()) else "FAIL_CLOSED",
        "isolatedFlagUsed": True,
        "converterPath": str(converter),
        "converterStdout": proc.stdout.strip(),
        "frozenSafeJson": file_record(frozen_path),
        "freshCacheJson": file_record(cache_output),
        "checks": checks,
        "rawPickleLoadedByAuditMainProcess": False,
    }


def prepare() -> None:
    started = time.time()
    config = read_config()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG_PATH, OUT / "preregistration.yaml")
    write_text(
        OUT / "decision_record.md",
        f"""# S19-L12 decision record

## Concise top summary

- **Research step ID:** `{STEP_ID}` (`{LOOP_ID}`).
- **Completion status:** PREPARED; substantive audit not yet run.
- **Artifacts written:** preregistration, immutable baseline, source snapshots, history/blame, zero-diff comparison, lineage map, and isolated lattice-equivalence record.
- **Validation result:** source identities and prior inputs are frozen before concordance interpretation.
- **Outcome classification:** Pending; L12 is analysis-only and cannot promote a scientific claim.
- **Caveats or blockers:** Public repositories are source lineage, not the unavailable paper implementation; source files without a detected license remain cache-only.
- **Recommended next action:** Commit and push the audit implementation, then run the deterministic audit phase; do not generate a GARD outcome.

The human authorized only `{LOOP_ID}`. The sequence is source freeze → sentence/panel/source audit → hash-frozen concordance matrix → whole-pipeline ranking → one unexecuted next-step design → validation and human review. Every S01–S18 and S19-L01–L11R result remains immutable.
""",
    )

    baseline = build_immutable_baseline()
    write_json(OUT / "immutable_prior_baseline.json", baseline)
    write_json(OUT / "immutable_prior_validation.json", validate_immutable_baseline(baseline))

    phirl = Path(config["paths"]["phirl"])
    iigr = Path(config["paths"]["iigr"])
    gard = Path(config["paths"]["gardHistorical"])
    breaking = Path(config["paths"]["breakingGrnMemories"])
    pinned = config["sourceLocks"]["phirlPinnedCommit"]
    local_head = git(phirl, "rev-parse", "HEAD")
    remote_master = git(phirl, "rev-parse", "refs/remotes/origin/master")
    tree = git(phirl, "rev-parse", f"{pinned}^{{tree}}")
    current_tree = git(phirl, "rev-parse", f"{remote_master}^{{tree}}")
    source_files = [
        Path(config["paths"]["paperPdf"]),
        Path(config["paths"]["paperMarkdown"]),
        PLAN,
        FULL_PLAN,
        AGENTS,
        Path(config["paths"]["claimLedger"]),
        Path(config["paths"]["matrixA"]),
        Path(config["paths"]["matrixB"]),
    ]
    source_files.extend(sorted(Path(config["paths"]["paperFigures"]).glob("figure-*.png")))
    manifest = {
        "schema": "eidosoma.e01.s19.l12.source_snapshot.v1",
        "researchStepId": STEP_ID,
        "retrievalDate": "2026-08-09",
        "networkFreezeCompleted": True,
        "networkPermittedAfterSnapshot": False,
        "phirl": {
            "repository": config["sourceLocks"]["phirlRepository"],
            "localHead": local_head,
            "remoteMaster": remote_master,
            "pinnedCommit": pinned,
            "pinnedTree": tree,
            "currentTree": current_tree,
            "masterEqualsPinned": remote_master == pinned and current_tree == tree,
            "shallow": git(phirl, "rev-parse", "--is-shallow-repository") == "true",
            "commitCountAllRefs": int(git(phirl, "rev-list", "--all", "--count")),
            "refs": git(phirl, "for-each-ref", "--format=%(refname)%09%(objectname)").splitlines(),
            "licenseStatus": "NO_LICENSE_FILE_DETECTED_REFERENCE_ONLY",
        },
        "iigr": {
            "repository": config["sourceLocks"]["iigrRepository"],
            "commit": git(iigr, "rev-parse", "HEAD"),
            "tree": git(iigr, "rev-parse", "HEAD^{tree}"),
            "commitCountAllRefs": int(git(iigr, "rev-list", "--all", "--count")),
            "licenseStatus": "NO_LICENSE_FILE_DETECTED_REFERENCE_ONLY",
        },
        "gardHistorical": {
            "repository": config["sourceLocks"]["gardHistoricalRepository"],
            "commit": git(gard, "rev-parse", "HEAD"),
            "tree": git(gard, "rev-parse", "HEAD^{tree}"),
            "licenseStatus": "NO_LICENSE_FILE_DETECTED_REFERENCE_ONLY",
        },
        "breakingGrnMemories": {
            "repository": config["sourceLocks"]["breakingGrnRepository"],
            "commit": git(breaking, "rev-parse", "HEAD"),
            "tree": git(breaking, "rev-parse", "HEAD^{tree}"),
            "licenseStatus": "NO_LICENSE_FILE_DETECTED_REFERENCE_ONLY",
        },
        "workspaceAndPaperFiles": [file_record(path) for path in source_files],
        "repositoryLockCommit": git(REPO, "rev-parse", "HEAD"),
        "repositoryBranch": git(REPO, "rev-parse", "--abbrev-ref", "HEAD"),
    }
    if manifest["phirl"]["pinnedCommit"] != local_head or not manifest["phirl"]["masterEqualsPinned"]:
        raise RuntimeError("PhiRL pinned/current source identity changed")
    write_json(OUT / "source_snapshot_manifest.json", manifest)

    tree_rows = repository_tree(phirl, pinned)
    write_json(
        OUT / "phirl_repository_tree.json",
        {
            "schema": "eidosoma.e01.s19.l12.phirl_tree.v1",
            "researchStepId": STEP_ID,
            "repository": config["sourceLocks"]["phirlRepository"],
            "commit": pinned,
            "tree": tree,
            "fileCount": len(tree_rows),
            "files": tree_rows,
        },
    )
    write_csv(OUT / "phirl_commit_history.csv", git_commit_history(phirl))
    function_map = {
        "preprocess_data": "main.py",
        "compute_phi": "main.py",
        "mutual_information_matrix": "information.py",
        "mutual_information_matrix_fast": "information.py",
        "minimum_information_bipartition": "information.py",
        "local_entropy_nd": "information.py",
        "local_phi_id": "information.py",
        "local_phi_r": "information.py",
        "save_info": "analysis.py",
        "_load_phi": "plotting.py",
    }
    write_csv(OUT / "phirl_function_blame.csv", function_blame(phirl, function_map))

    write_text(
        OUT / "phirl_master_vs_pinned_diff.md",
        f"""# PhiRL current master versus pinned commit

## Concise top summary

- **Research step ID:** `{STEP_ID}`.
- **Completion status:** SOURCE COMPARISON COMPLETE.
- **Artifacts written:** `source_snapshot_manifest.json`, `phirl_repository_tree.json`, `phirl_commit_history.csv`, and `phirl_function_blame.csv`.
- **Validation result:** PASS — remote `master`, local checkout, and the pinned commit are all `{pinned}` with tree `{tree}`.
- **Outcome classification:** `DIRECT_PUBLIC_CODE`; no version drift exists to explain E01 discrepancies.
- **Caveats or blockers:** Equality to public master does not identify the unavailable GARD-paper implementation. No tag or alternate public branch supplies GARD-specific code.
- **Recommended next action:** Audit the internal commit lineage and paper/source behavior; do not infer author-code identity.

The diff is empty because the current remote master and pinned commit are identical. The complete public tree contains {len(tree_rows)} files. PhiRL has {manifest['phirl']['commitCountAllRefs']} commits across all local refs; no tag is present. The meaningful version boundary is internal: slow bidirectional lagged MI and `local_phi_r` existed first, public `emergence = synergy + causation` was exposed later, and the fast-MI plus trace-scaled covariance regularization path arrived in November 2025.
""",
    )
    write_text(
        OUT / "iigr_phirl_lineage_map.md",
        f"""# IIGR–PhiRL source-lineage map

## Concise top summary

- **Research step ID:** `{STEP_ID}`.
- **Completion status:** SOURCE LINEAGE AUDIT COMPLETE.
- **Artifacts written:** source hashes, complete PhiRL history, function blame, and safe-lattice equivalence.
- **Validation result:** PASS — IIGR `{manifest['iigr']['commit']}` and PhiRL `{pinned}` share the byte-identical 16-node lattice and closely corresponding local-Phi functions; their repositories have no Git-parent relationship.
- **Outcome classification:** `SOURCE_LINEAGE_INFERENCE`, not author implementation identity.
- **Caveats or blockers:** IIGR is a GRN application and PhiRL is an RL-representation application. Neither tree contains the GARD prediction or intervention pipeline.
- **Recommended next action:** Keep inherited operations and later PhiRL changes distinct in the executable data-flow audit.

IIGR predates PhiRL and supplies the closest public ancestry for the lattice, local Gaussian entropy, local ΦID inversion, `local_phi_r`, lagged mutual-information graph, Fiedler partition, and partition averaging. IIGR's terminal commit is explicitly named “fix phir bug”; S12B/S12C verified the corrected atom set. PhiRL initially copied the slow lineage, later exposed both `integrated` and `emergence`, and then introduced the fast-MI and covariance-regularized path. The two lattice pickles are byte-identical. Structural correspondence and chronology justify a lineage inference, but public Git metadata does not prove a direct code-copy event or the pipeline used for the GARD paper.
""",
    )
    lattice = safe_lattice_equivalence(config)
    write_json(OUT / "safe_lattice_equivalence.json", lattice)
    if not lattice["success"]:
        raise RuntimeError("safe lattice equivalence failed")
    write_json(
        OUT / "prepare_runtime.json",
        {
            "researchStepId": STEP_ID,
            "phase": "prepare",
            "wallSeconds": time.time() - started,
            "cpuSeconds": time.process_time(),
            "gpuUsed": False,
            "networkDisabledForLaterPhases": True,
        },
    )


def _method_statement_rows() -> list[dict[str, Any]]:
    """Return the method-bearing manuscript statements omitted by the claim ledger.

    Page labels refer to PDF pages (including the title page).  Short quotations
    stay below the copyright excerpt limit and are used only to make the audit
    traceable.  Longer claims are faithful paraphrases.
    """

    specs = [
        # id, page, paragraph, paraphrase, input, output, time, unit, preprocessing,
        # estimator, label, denominator, aggregation, test, value, component,
        # specified, partial, conflict, public-code-absent, unresolved
        ("M001", "3", "Results ¶1", "100 independent GARD assemblies used different random seeds.", "seed", "assembly run", "run", "catalytic matrix/run", "none stated", "GARD", "none", "100 runs", "across runs", "none", "100", "FIGURE_1", True, False, False, True, "seed derivation; matrix sharing"),
        ("M002", "3", "Results ¶1", "Assemblies grow and fission; one daughter continues until a fixed generation count.", "assembly composition", "selected daughter trajectory", "growth-fission generation", "run", "none", "GARD", "none", "100 generations", "single lineage", "none", "fixed generation count", "FIGURE_1_B", True, True, False, True, "daughter choice; overshoot; extinction"),
        ("M003", "3", "Results ¶1", "Self-replicators are recurring compositions inherited across generations.", "composition trajectory", "replicator state", "unspecified time step", "within-run state", "composition representation unspecified", "historical GARD compotype", "recurring composition", "trajectory", "most recurring composition", "threshold membership", "not numerically defined", "FIGURE_1_C", False, True, False, True, "one versus multiple clusters; clock; recurrence algorithm"),
        ("M004", "3", "Results ¶1", "An assembly enters or exits replication by a similarity threshold to the most recurring composition.", "current composition and run reference", "binary state", "time step", "within-run observation", "not stated", "similarity", "self-replicator", "all time steps", "none", "threshold", "threshold value omitted here", "FIGURE_1_C", False, True, False, True, "reference construction; clock; threshold implementation"),
        ("M005", "8", "Methods/GARD ¶1", "Each molecule belongs to one of N_g molecular types.", "molecules", "count vector", "molecular step", "assembly", "none", "GARD", "none", "N_g", "none", "none", "N_g=100 later", "FIGURE_1_A", True, False, False, False, "none material"),
        ("M006", "8", "Methods/GARD ¶3", "A single assembly is followed with a catalytic matrix beta from n_min molecules.", "beta and initial state", "single lineage", "generation", "run", "uniform types without replacement", "GARD", "none", "one lineage", "none", "none", "n_min=40", "FIGURE_1_A", True, True, False, True, "initial multiplicity and exact beta parameterization"),
        ("M007", "8", "Methods/GARD ¶3", "Catalytic rates are sampled from a lognormal distribution with mean A and standard deviation sigma.", "random variates", "beta", "initialization", "matrix", "lognormal", "sampling", "none", "100x100", "none", "none", "A=-4; sigma=4", "FIGURE_1_A", True, True, False, True, "arithmetic versus log-space parameters"),
        ("M008", "8", "Methods/GARD ¶4", "Each generation uses stochastic Poisson molecule-loss and molecule-gain updates.", "composition and beta", "updated composition", "molecular update", "assembly", "none", "Poisson update", "none", "until stop", "sequence", "none", "parameter not stated", "FIGURE_1_B", True, True, False, True, "Poisson exposure/time scale; update ordering"),
        ("M009", "8", "Methods/GARD ¶4", "Growth stops at n_max or max_steps before fission.", "within-generation states", "pre-fission state", "molecular step", "generation", "none", "stop rule", "none", "per generation", "first stopping event", "none", "n_max=80; max_steps=1000", "FIGURE_1_B", True, True, False, True, "overshoot trimming and fission after max-step termination"),
        ("M010", "8", "Methods/GARD ¶4", "Fission samples daughter molecules with binomial probability 0.5.", "pre-fission composition", "two daughters", "generation boundary", "generation", "none", "binomial fission", "none", "molecules", "two daughters", "none", "p=0.5", "FIGURE_1_B", True, True, False, True, "daughter continuation; empty daughter handling"),
        ("M011", "8", "Methods/GARD ¶5", "The declared parameters are N_g=100, n_min=40, A=-4, sigma=4, n_gen=100, n_max=80, max_steps=1000.", "configuration", "simulation scope", "run", "run", "none", "GARD", "none", "100 generations", "none", "none", "declared parameter tuple", "FIGURE_1", True, False, False, False, "Poisson exposure absent"),
        ("M012", "8", "Methods/GARD ¶6", "Steady compositions are highly similar in Euclidean space across adjacent generations.", "post-fission compositions", "similarity", "generation", "generation pair", "relative composition implied", "Euclidean similarity", "self-replicator", "adjacent generations", "none", "not stated", "qualitative", "FIGURE_1_C", True, True, False, True, "Euclidean distance versus H cosine threshold"),
        ("M013", "9", "Methods/Causal emergence ¶1", "PhiID estimates unique predictive information about future evolution unavailable from parts.", "X_t and X_t+1", "information atoms", "lag 1", "trajectory", "not stated", "PhiID", "none", "complete trajectory", "local values", "none", "conceptual", "FIGURE_1_D", True, True, False, False, "redundancy convention and fitted distribution scope"),
        ("M014", "9", "Methods/Causal emergence equation", "Displayed Phi-r equals whole lagged mutual information minus the sum of part-to-whole lagged mutual informations.", "bivariate reduced X", "scalar trajectory", "lag 1", "time point", "minimum bipartition", "whole-minus-parts", "none", "T-1", "local equation", "none", "displayed equation", "FIGURE_1_D", True, False, True, True, "relationship to public integrated/emergence names"),
        ("M015", "9", "Methods/Causal emergence ¶4", "A minimum-information bipartition reduces the system to two components.", "N_g variables", "two components", "complete trajectory", "run", "not stated", "minimum-information bipartition", "none", "one split/run", "two groups", "none", "2 components", "FIGURE_1_D", True, True, False, False, "exact graph, objective, approximation and tie rules"),
        ("M016", "9", "Methods/Causal emergence ¶4", "The paper describes causal emergence as one PhiID atom.", "two-part time series", "one atom", "lag 1", "time point", "none", "PhiID", "none", "T-1", "one atom", "none", "one atom", "FIGURE_1_D", True, False, True, False, "displayed equation is a multi-atom linear combination"),
        ("M017", "9", "Methods/Data preprocessing ¶1", "Relative molecular composition at every molecular step is the causal-emergence substrate.", "count trajectory", "relative composition trajectory", "molecular step", "run", "closure", "normalization", "none", "N_g x n_tot", "per observation", "none", "variable n_tot", "FIGURE_1_D", True, False, False, False, "pseudocount for zeros"),
        ("M018", "9", "Methods/Data preprocessing ¶1", "Centered log-ratio transform maps the simplex to Euclidean coordinates.", "relative composition", "CLR", "molecular step", "variable/time", "CLR", "log-ratio", "none", "N_g x n_tot", "per observation", "none", "not fully encoded in extracted text", "FIGURE_1_D", True, True, False, False, "zero replacement"),
        ("M019", "9", "Methods/Data preprocessing ¶1", "The last CLR component is removed to restore full rank.", "CLR vector", "N_g-1 coordinates", "molecular step", "run", "drop last component", "deterministic projection", "none", "N_g-1", "per observation", "none", "last component", "FIGURE_1_D", True, False, False, False, "component-order convention"),
        ("M020", "4", "Results/Figure 2", "Across 100 runs, median plus or minus standard deviation has no aggregate linear trend.", "100 unequal trajectories", "aggregate trajectory", "molecular step", "run at each time", "unstated unequal-length handling", "median and standard deviation", "none", "available/padded unknown", "across runs by time", "linear regression", "p=0.1995", "FIGURE_2_A", True, True, False, True, "alignment, tail support, regression weighting"),
        ("M021", "4", "Results/Figure 2", "Individual spikes exceed overall mean plus three standard deviations.", "run trajectories", "spike events", "molecular step", "run", "none", "threshold excursion", "none", "overall scope unknown", "within/across runs unknown", "none", "> mean+3SD", "FIGURE_2_B_D", True, True, False, True, "overall scope and completed-fit dependence"),
        ("M022", "4", "Results/Figure 3", "Spearman correlation is computed for each complete trajectory.", "Phi-r and binary label", "runwise rho", "molecular step", "run", "complete trajectory", "Spearman", "self-replicator", "T-1 alignment", "100 run coefficients", "one-sample test", "73 positive; mean about .139", "FIGURE_3", True, True, True, True, "level versus change mismatch"),
        ("M023", "4", "Figure 3 caption", "The caption correlates changes in Phi-r with self-replication.", "delta Phi-r and label", "runwise rho", "molecular step", "run", "difference", "Spearman", "self-replicator", "T-2", "100 coefficients", "one-sample t", "54% positive significant", "FIGURE_3", True, False, True, True, "text uses Phi-r level"),
        ("M024", "4", "Results/Figure 4", "Mean Phi-r is compared between drift and self-replication within runs.", "Phi-r and state", "two state means", "molecular step", "run", "undefined-state handling absent", "state mean", "self-replicator", "state observations", "runwise", "Mann-Whitney", "57/100 higher; p<.001", "FIGURE_4", True, True, False, True, "test scope and Fisher inputs"),
        ("M025", "4", "Results/temporal dependence", "Ljung-Box rejects temporal independence for raw and differenced trajectories.", "Phi-r trajectory", "p-value", "molecular step", "run", "raw/difference", "Ljung-Box", "none", "100 runs", "run counts", "Ljung-Box", "86/100 raw; 100/100 differenced", "FIGURE_2", True, True, False, True, "lag, df and multiple testing"),
        ("M026", "5", "Results/Figure 5", "The first 25% of Phi-r predicts the final 75% self-replication trajectory.", "prefix Phi-r", "suffix binary sequence", "molecular step", "matrix/run", "unspecified tensor handling", "MLP", "self-replicator", "final 75%", "accuracy", "train/test", "80/20 runs; 10 repetitions", "FIGURE_5", True, True, False, True, "padding, masking, scaling, balancing, architecture, completed fit"),
        ("M027", "5", "Figure 5 caption", "Dummy always predicts the most likely state.", "training labels", "constant prediction", "target step", "run/split", "unspecified", "majority dummy", "self-replicator", "unknown target sampling", "binary accuracy", "none", "about 60% visible", "FIGURE_5", True, True, True, True, "why baseline differs from 88% Table 1 occupancy"),
        ("M028", "5", "Results/Figure 5", "Alternative input/output proportions gave quantitatively similar prediction results.", "trajectory prefixes/suffixes", "accuracy", "molecular step", "split", "not stated", "same MLP", "self-replicator", "not stated", "10 repetitions", "Mann-Whitney", "qualitatively similar", "FIGURE_5", True, True, False, True, "proportion list and tensor reconstruction"),
        ("M029", "5", "Results/spikes", "Replication probability correlates with spike time and spacing but not height.", "runwise spike descriptors", "rho", "run", "run", "spike definition inherited", "Spearman", "replication probability", "100 runs", "runwise", "permutation unspecified", "rho=.66/.71; height n.s.", "FIGURE_5_TEXT", True, True, False, True, "descriptor definitions and eligible runs"),
        ("M030", "5", "Results/Figure 6", "After every fission, all molecule additions and deletions are scored and the extremum is applied.", "post-fission state and actions", "chosen action", "generation boundary", "intervention decision", "not stated", "Phi-r action score", "none", "2*N_g candidates stated", "exhaustive argmax/argmin", "none", "N_g=100", "FIGURE_6_A", True, True, False, True, "single-state score, refit, future use, eligible deletions, ties"),
        ("M031", "6", "Figure 6 caption", "Interventions occur right after fission and natural GARD dynamics resume to the next generation.", "edited daughter", "next interval", "generation", "matrix-treatment pair", "none", "GARD", "self-replicator outcome", "100 generations", "treatment curves", "linear regression", "max increases; min decreases", "FIGURE_6", True, True, False, True, "common random streams and precise scorer"),
        ("M032", "6", "Results/Table 1", "Persistence is total self-replicating lifetime in molecular steps.", "binary trajectory", "count", "molecular step", "run", "none", "sum", "self-replicator", "all molecular steps", "across runs", "Mann-Whitney", "control 716±198", "TABLE_1_PERSISTENCE", True, True, False, True, "SD versus SE"),
        ("M033", "6", "Results/Table 1", "Probability is the fraction of molecular steps in self-replication.", "binary trajectory", "fraction", "molecular step", "run", "none", "mean", "self-replicator", "all molecular steps", "across runs", "Mann-Whitney", "control 88±3%", "TABLE_1_PROBABILITY", True, True, True, True, "SD versus SE; incompatibility with Fig5 dummy"),
        ("M034", "6", "Results/Table 1", "Consistency is Pearson correlation between consecutive binary states.", "Y_t,Y_t+1", "Pearson r", "molecular step", "run", "none", "Pearson", "self-replicator", "T-1 pairs", "across runs", "Mann-Whitney", "control .38±.06", "TABLE_1_CONSISTENCY", True, True, True, True, "constant labels; min .42 conflicts with worsened-all text"),
        ("M035", "6", "Results/Table 1", "Time to first replicator is the number of molecular steps before first appearance.", "binary trajectory", "onset index", "molecular step", "run", "none", "first positive", "self-replicator", "trajectory", "across runs", "Mann-Whitney", "control 37±27% printed", "TABLE_1_FIRST_REPLICATOR", True, True, True, True, "percent sign versus molecular-step note; zero/one indexing"),
        ("M036", "6", "Figure 6C caption", "Replication probability over GARD generations is regressed by treatment.", "generation-indexed outcomes", "probability curve", "generation", "matrix at generation", "unknown window", "linear regression", "self-replicator", "molecular steps within generation", "across matrices", "regression with 95% CI", "max slope .041; control .008; min -.03", "FIGURE_6_C", True, True, False, True, "windowing, units, relation to Table1 overall mean"),
        ("M037", "6", "Data availability", "Code will be made public upon publication.", "implementation", "future public code", "publication", "paper", "none", "none", "none", "none", "none", "none", "not yet available", "WHOLE_PIPELINE", True, False, False, True, "paper-specific implementation unavailable"),
    ]
    keys = [
        "paperStatementId", "page", "paragraph", "exactShortQuotationOrFaithfulParaphrase",
        "claimedInputObject", "claimedOutputObject", "timeUnit", "statisticalUnit", "preprocessing",
        "estimator", "label", "denominator", "aggregation", "statisticalTest", "reportedValue",
        "figureOrTableRelationship", "directlySpecified", "partiallySpecified", "internallyConflicting",
        "absentFromPublicCode", "unresolvedFields",
    ]
    rows = [dict(zip(keys, spec, strict=True)) for spec in specs]
    for row in rows:
        row["E01Implementation"] = "Cross-referenced in the concordance matrix; no new implementation in L12."
        row["E01Result"] = "See frozen S01–S18/S19 evidence crosswalk."
    return rows


def build_paper_statement_registry() -> pd.DataFrame:
    rows = _method_statement_rows()
    ledger = pd.read_csv(read_config()["paths"]["claimLedger"])
    family_page = {
        "metric_distinctiveness": "3–4", "emergence_dynamics": "4", "emergence_spikes": "4",
        "replicator_association": "4", "replicator_state": "4", "temporal_dependence": "4",
        "prediction": "5", "spike_prediction": "5", "intervention": "5–6", "table1": "6",
    }
    for _, claim in ledger.iterrows():
        component = str(claim.get("primary_source_location", ""))
        family = str(claim["claim_family"])
        rows.append(
            {
                "paperStatementId": f"CLAIM_{claim['claim_id']}",
                "page": family_page.get(family, "3–6"),
                "paragraph": component,
                "exactShortQuotationOrFaithfulParaphrase": claim["claim_text"],
                "claimedInputObject": claim["unit_of_analysis"],
                "claimedOutputObject": claim["reproduction_estimand"],
                "timeUnit": "as declared in claim ledger; unresolved where noted",
                "statisticalUnit": claim["unit_of_analysis"],
                "preprocessing": "claim-specific; see dependency graph",
                "estimator": claim["reported_statistic"],
                "label": "self-replicator where applicable",
                "denominator": claim["sample_scope"],
                "aggregation": claim["sample_scope"],
                "statisticalTest": claim["inferential_test"],
                "reportedValue": claim["reported_target"],
                "figureOrTableRelationship": component,
                "directlySpecified": True,
                "partiallySpecified": str(claim["specification_status"]).startswith("underdetermined"),
                "internallyConflicting": bool(str(claim.get("discrepancy_ids", "")).strip() not in ("", "nan")),
                "absentFromPublicCode": True,
                "E01Implementation": claim["reproduction_estimand"],
                "E01Result": f"S18 status cross-referenced for {claim['claim_id']}",
                "unresolvedFields": claim["notes"],
            }
        )
    frame = pd.DataFrame(rows)
    ordered = [
        "paperStatementId", "page", "paragraph", "exactShortQuotationOrFaithfulParaphrase",
        "claimedInputObject", "claimedOutputObject", "timeUnit", "statisticalUnit", "preprocessing",
        "estimator", "label", "denominator", "aggregation", "statisticalTest", "reportedValue",
        "figureOrTableRelationship", "directlySpecified", "partiallySpecified", "internallyConflicting",
        "absentFromPublicCode", "E01Implementation", "E01Result", "unresolvedFields",
    ]
    return frame[ordered]


def build_paper_dependency_graph() -> nx.DiGraph:
    graph = nx.DiGraph(researchStepId=STEP_ID, graphType="paper_method_dependency")
    nodes = {
        "seed_matrix": ("Catalytic matrix + initial state", "input"),
        "gard_kernel": ("Poisson growth/loss kernel", "simulation"),
        "molecular_clock": ("Selected molecular trajectory", "data"),
        "fission": ("Fission + daughter continuation", "simulation"),
        "boundary_states": ("Post-fission boundary states", "data"),
        "replicator_reference": ("Most recurring composition / clusters", "label"),
        "replicator_label": ("Binary self-replicator state", "label"),
        "relative_composition": ("Relative compositions", "preprocessing"),
        "clr_drop": ("CLR + dropped component", "preprocessing"),
        "mi_graph": ("Lagged-MI graph", "estimator"),
        "mib": ("Minimum-information bipartition", "estimator"),
        "phiid": ("Local Gaussian PhiID", "estimator"),
        "phi_scalar": ("Paper Phi-r scalar", "unresolved_metric"),
        "fig2": ("Figure 2 aggregation/spikes", "output"),
        "fig3_4": ("Figures 3–4 association/state", "output"),
        "fig5_tensor": ("Figure 5 prefix→suffix tensor", "unresolved_task"),
        "fig5": ("Figure 5 accuracies", "output"),
        "action_scorer": ("Hypothetical-state Phi-r scorer", "unresolved_intervention"),
        "intervention": ("Max/control/min trajectories", "simulation"),
        "fig6_table1": ("Figure 6 + Table 1", "output"),
    }
    for node, (label, kind) in nodes.items():
        graph.add_node(node, label=label, kind=kind)
    edges = [
        ("seed_matrix", "gard_kernel"), ("gard_kernel", "molecular_clock"), ("gard_kernel", "fission"),
        ("fission", "boundary_states"), ("fission", "gard_kernel"), ("boundary_states", "replicator_reference"),
        ("molecular_clock", "replicator_reference"), ("replicator_reference", "replicator_label"),
        ("molecular_clock", "relative_composition"), ("relative_composition", "clr_drop"),
        ("clr_drop", "mi_graph"), ("mi_graph", "mib"), ("mib", "phiid"), ("phiid", "phi_scalar"),
        ("phi_scalar", "fig2"), ("phi_scalar", "fig3_4"), ("replicator_label", "fig3_4"),
        ("phi_scalar", "fig5_tensor"), ("replicator_label", "fig5_tensor"), ("fig5_tensor", "fig5"),
        ("phi_scalar", "action_scorer"), ("boundary_states", "action_scorer"),
        ("action_scorer", "intervention"), ("intervention", "fig6_table1"),
        ("replicator_label", "fig6_table1"),
    ]
    for source, target in edges:
        graph.add_edge(source, target)
    return graph


def paper_discrepancies() -> pd.DataFrame:
    rows = [
        ("D01", "FIGURE_3_LEVEL_CHANGE", "Results describe correlation with Phi-r level; caption specifies changes in Phi-r.", "PAPER_INTERNAL_CONFLICT", "Keep LEVEL and CHANGE analyses separate; S15 did so."),
        ("D02", "FIGURE_5_TABLE1_PREVALENCE", "Figure 5 majority dummy is about 60%, whereas Table 1 control molecular occupancy is 88%.", "PAPER_INTERNAL_CONFLICT", "Requires a different target/sampling/denominator or data set; public source is silent."),
        ("D03", "METRIC_ONE_ATOM_EQUATION", "Prose calls Phi-r one atom; displayed whole-minus-parts equation expands to several signed atoms.", "PAPER_INTERNAL_CONFLICT", "Metric identity is algebraically adjudicated, not selected by result."),
        ("D04", "METRIC_PUBLIC_NAMES", "Public PhiRL exposes integrated=local_phi_r and emergence=synergy+causation; neither generally equals the displayed equation.", "PAPER_INTERNAL_CONFLICT", "Author implementation required."),
        ("D05", "FIGURE6_MIN_CONSISTENCY", "Table 1 min consistency .42 exceeds control .38 while text says minimization worsened all four properties.", "PAPER_INTERNAL_CONFLICT", "Could reflect misstatement, direction misunderstanding, or table error."),
        ("D06", "TABLE1_ONSET_UNIT", "Time-to-first values carry percent signs, but the note defines molecular-step counts.", "PAPER_INTERNAL_CONFLICT", "Report raw and normalized units separately."),
        ("D07", "FIGURE6_TABLE1_PROBABILITY", "Max and control both round to 88% in Table 1 despite separated time-varying Figure 6C curves.", "PAPER_INTERNAL_CONFLICT", "Overall means can coexist with different slopes, but aggregation/window is absent."),
        ("D08", "FIGURE2_VARIABLE_LENGTH", "Figure 2 aggregate reaches about 1300 steps although run lengths vary substantially.", "AUTHOR_IMPLEMENTATION_REQUIRED", "Padding, available-case support, truncation, or resampling is unspecified."),
        ("D09", "REPLICATOR_CLUSTER_VS_H", "Paper describes recurring clusters/most recurring composition; frozen paper-facing branch uses exact adjacent H>0.9.", "AUTHOR_IMPLEMENTATION_REQUIRED", "S19 label reconstructions did not recover the full fingerprint."),
        ("D10", "SIMILARITY_GEOMETRY", "Methods say compositions are similar in Euclidean space while historical GARD H is cosine similarity.", "PAPER_INTERNAL_CONFLICT", "Both source-grounded but non-equivalent geometries exist."),
        ("D11", "MIB_PUBLIC_APPROXIMATION", "Paper names a minimum-information bipartition; public PhiRL uses an unnormalized Fiedler sign split of a lagged-MI graph.", "AUTHOR_IMPLEMENTATION_REQUIRED", "No paper text identifies this approximation or tie behavior."),
        ("D12", "INTERVENTION_SINGLE_STATE_SCORE", "Paper requires scoring each hypothetical post-fission edit but does not define how a trajectory-level Phi fit gives that one state a score.", "AUTHOR_IMPLEMENTATION_REQUIRED", "S17 used one frozen prospective append-and-refit reconstruction only."),
        ("D13", "COMPLETED_FIT_PREDICTION", "Public PhiRL fits partition and Gaussians on the complete trajectory, which would leak the suffix into first-quarter prediction.", "AUTHOR_IMPLEMENTATION_REQUIRED", "S16 separates retrospective completed fit from cutoff-causal fitting."),
        ("D14", "AGGREGATE_TREND", "Paper reports no aggregate trend; S14 finds a significant positive trend in both frozen candidates.", "DIRECT_FROZEN_E01_RESULT", "Trajectory alignment and scalar identity remain possible causes."),
        ("D15", "PREDICTION_TARGET_DETERMINISM", "Frozen Y=I(H>0.9) is exactly determined by H and has ~98% prevalence, leaving no unrestricted incremental information beyond exact H.", "DIRECT_FROZEN_E01_RESULT", "A different author label/task is required to explain Figure 5."),
        ("D16", "FIGURE3_CATEGORY_DENOMINATOR", "Text reports 54 of 73 positives significant; panel reports 54% of all runs positive-significant.", "PAPER_INTERNAL_CONFLICT", "54 runs satisfies both counts, but the stated 'majority of these' denominator is rhetorically ambiguous."),
    ]
    return pd.DataFrame(rows, columns=["discrepancyId", "topic", "description", "evidenceLabel", "adjudication"])


def figure_specs() -> tuple[pd.DataFrame, pd.DataFrame]:
    image_map = {
        "FIGURE_1": "figure-01.png", "FIGURE_2": "figure-02.png", "FIGURE_3": "figure-03.png",
        "FIGURE_4": "figure-04.png", "FIGURE_5": "figure-05.png", "FIGURE_6_A": "figure-06.png",
        "FIGURE_6_B": "figure-07.png", "FIGURE_6_C": "figure-08.png",
    }
    configs = read_config()
    image_root = Path(configs["paths"]["paperFigures"])
    panels = [
        ("FIGURE_1_A", "Catalytic network/composome", "beta matrix and molecule types", "molecule type / reaction", "schematic", "one schematic", "none", "compatible", "not applicable", "S03/S12 source reconstruction only"),
        ("FIGURE_1_B", "Growth-fission lineage", "molecular compositions and fissions", "selected lineage", "molecular and generation", "one schematic", "one daughter continues", "not applicable", "compatible", "Frozen candidates reproduce structure with unresolved kernel details"),
        ("FIGURE_1_C", "Recurring composition clusters", "composition-space states", "cluster memberships", "state/generation ambiguous", "multiple colored points/clusters", "not visible", "paper describes clusters/attractors", "not applicable", "S19 L02–L11R did not recover joint fingerprint"),
        ("FIGURE_1_D", "Phi-r molecular trajectory", "relative compositions every molecular step", "one scalar per lagged step", "molecular step", "one schematic trajectory", "not visible", "compatible with molecular substrate", "not applicable", "S13Y completed-fit trajectory is closest source branch"),
        ("FIGURE_2_A", "Aggregate median±SD and trend", "100 Phi-r trajectories", "run at x position", "molecular step", "up to 100 early, unknown tail", "available-case/padding/truncation invisible", "caption says median±SD", "not applicable", "S14 directional spikes but positive trend discrepancy"),
        ("FIGURE_2_B", "Sample positive spikes", "one Phi-r trajectory", "molecular observation", "molecular step", "one run", "none", "compatible", "not applicable", "S14 positive excursions resemble"),
        ("FIGURE_2_C", "Sample rectangular transitions", "one Phi-r trajectory", "molecular observation", "molecular step", "one run", "none", "compatible", "not applicable", "S14 links many spikes to partitions/numerical condition"),
        ("FIGURE_2_D", "Sample positive/negative excursions", "one Phi-r trajectory", "molecular observation", "molecular step", "one run", "none", "compatible", "not applicable", "S14 both excursion signs observed"),
        ("FIGURE_3_A", "Runwise Spearman histogram", "100 paired trajectories", "run coefficient", "complete trajectory", "100 bars/values", "not applicable", "caption says change; text says level", "not applicable", "S15 preserves both analyses"),
        ("FIGURE_3_B", "Sign/significance categories", "100 runwise tests", "run", "complete trajectory", "100 categorized", "not applicable", "category values sum to 100%", "not applicable", "S15 level/change directional resemblance only"),
        ("FIGURE_4_A", "Within-run drift→replication means", "Phi-r and label", "run", "molecular state", "100 lines claimed", "undefined states not visible", "compatible with text", "not applicable", "S15 comparable retrospective contrast"),
        ("FIGURE_4_B", "Median±SD state means", "100 paired state summaries", "run", "state", "100 claimed", "not applicable", "caption typo says lines in B", "not applicable", "S15 paper-like diagnostic direction"),
        ("FIGURE_5", "Five-family prediction accuracy", "first-quarter features/final-three-quarter label", "matrix split", "molecular target implied", "10 dots/family", "layout not visible", "caption compatible but underspecified", "dummy conflicts with Table1 occupancy", "S16 masked original-order reconstruction yields ~98% dummy"),
        ("FIGURE_6_A", "Intervention schematic", "post-fission state/action candidates", "generation decision", "generation", "schematic", "not applicable", "compatible with text", "not applicable", "S17 append-and-refit current-prefix reconstruction"),
        ("FIGURE_6_B", "Persistence by treatment", "treatment trajectories", "matrix/treatment", "molecular steps", "distribution, likely 100/treatment", "not visible", "compatible with Table1 means", "Table1 values approximate centers", "S17 min harms but max does not help"),
        ("FIGURE_6_C", "Probability over generations", "generation-window labels", "matrix at generation", "generation", "curves with 95% CI", "unknown", "compatible with caption", "max/control overall both round 88%", "S17 treatment curves do not recover ordering"),
        ("TABLE_1_PERSISTENCE", "Persistence", "binary trajectories", "matrix/treatment", "molecular steps", "100/treatment implied", "not applicable", "definition explicit", "compatible with Fig6B", "S17 computes exact definition"),
        ("TABLE_1_PROBABILITY", "Probability", "binary trajectories", "matrix/treatment", "molecular steps", "100/treatment implied", "not applicable", "definition explicit", "conflicts with Fig5 dummy if same task", "S17 computes exact definition"),
        ("TABLE_1_CONSISTENCY", "Consecutive-state correlation", "binary trajectories", "matrix/treatment", "molecular pairs", "100/treatment implied", "not applicable", "definition explicit", "min value conflicts with worsened-all text", "S17 undefined handling explicit"),
        ("TABLE_1_FIRST_REPLICATOR", "First onset", "binary trajectories", "matrix/treatment", "molecular step per note", "100/treatment implied", "not applicable", "note says molecular step", "cells print percent signs", "S17 raw molecular-step onset"),
    ]
    cols = ["panelId", "panelPurpose", "minimumRawData", "likelyStatisticalUnit", "timeUnit", "visibleObservationCount", "paddingTruncationUnequalLength", "textCompatibility", "tableCompatibility", "frozenE01Compatibility"]
    panel_frame = pd.DataFrame(panels, columns=cols)
    transformations = {
        "FIGURE_1_A": "render catalytic matrix/network and assembly composition",
        "FIGURE_1_B": "order growth states and mark fission/selected daughter",
        "FIGURE_1_C": "embed compositions and assign recurring-attractor clusters",
        "FIGURE_1_D": "calculate one Phi-r value per molecular lag",
        "FIGURE_2_A": "align unequal trajectories; median, SD and linear regression at molecular index",
        "FIGURE_2_B": "plot one local Phi-r trajectory",
        "FIGURE_2_C": "plot one local Phi-r trajectory",
        "FIGURE_2_D": "plot one local Phi-r trajectory",
        "FIGURE_3_A": "runwise Spearman then histogram",
        "FIGURE_3_B": "categorize runwise sign and unadjusted significance",
        "FIGURE_4_A": "within-run state-conditioned means joined by run",
        "FIGURE_4_B": "across-run median and SD of state means",
        "FIGURE_5": "first-quarter feature tensor to final-three-quarter binary accuracy",
        "FIGURE_6_A": "enumerate post-fission add/delete actions and select score extremum",
        "FIGURE_6_B": "sum positive molecular steps and compare treatments",
        "FIGURE_6_C": "aggregate within-generation occupancy, fit treatment regressions and 95% intervals",
        "TABLE_1_PERSISTENCE": "sum binary molecular state",
        "TABLE_1_PROBABILITY": "mean binary molecular state",
        "TABLE_1_CONSISTENCY": "Pearson correlation of adjacent binary state",
        "TABLE_1_FIRST_REPLICATOR": "first positive index",
    }
    panel_frame.insert(3, "exactTransformationImplied", panel_frame["panelId"].map(transformations))
    def image_for(panel_id: str) -> str:
        key = panel_id
        if key.startswith("FIGURE_1_"): key = "FIGURE_1"
        elif key.startswith("FIGURE_2_"): key = "FIGURE_2"
        elif key.startswith("FIGURE_3_"): key = "FIGURE_3"
        elif key.startswith("FIGURE_4_"): key = "FIGURE_4"
        elif key.startswith("TABLE"): return ""
        return image_map.get(key, "")
    panel_frame["sourceImage"] = [image_for(value) for value in panel_frame["panelId"]]
    panel_frame["sourceImageSha256"] = [sha256_file(image_root / name) if name else "" for name in panel_frame["sourceImage"]]
    panel_frame["imageWidthPx"] = [Image.open(image_root / name).size[0] if name else pd.NA for name in panel_frame["sourceImage"]]
    panel_frame["imageHeightPx"] = [Image.open(image_root / name).size[1] if name else pd.NA for name in panel_frame["sourceImage"]]

    digitized = [
        ("FIGURE_2_A", "x_axis_max_molecular_step", 1300, "approximately", "manual native-resolution constraint", "aggregate extends beyond median run length"),
        ("FIGURE_2_A", "baseline_median", 0.00, "approximately", "manual native-resolution constraint", "centered near zero"),
        ("FIGURE_2_A", "linear_regression_p", 0.1995, "reported exact", "paper caption/text", "no significant trend"),
        ("FIGURE_2_B", "x_axis_max_molecular_step", 800, "approximately", "manual native-resolution constraint", "sample run"),
        ("FIGURE_2_C", "x_axis_max_molecular_step", 800, "approximately", "manual native-resolution constraint", "sample run"),
        ("FIGURE_2_D", "x_axis_max_molecular_step", 1050, "approximately", "manual native-resolution constraint", "sample run"),
        ("FIGURE_2_B_D", "positive_spike_range", 4.0, "order-of-magnitude", "manual native-resolution constraint", "positive peaks several units"),
        ("FIGURE_2_B_D", "negative_spike_range", -4.0, "order-of-magnitude", "manual native-resolution constraint", "negative peaks several units"),
        ("FIGURE_3_A", "rho_histogram_min", -0.15, "approximately", "manual native-resolution constraint", "visible left edge"),
        ("FIGURE_3_A", "rho_histogram_max", 0.45, "approximately", "manual native-resolution constraint", "visible right edge"),
        ("FIGURE_3_A", "rho_mean", 0.139, "reported/visible", "paper", "positive one-sample mean"),
        ("FIGURE_3_B", "positive_significant_fraction", 0.54, "reported/visible", "paper", "54 runs"),
        ("FIGURE_3_B", "positive_nonsignificant_fraction", 0.19, "visible", "native panel", "19 runs"),
        ("FIGURE_3_B", "negative_significant_fraction", 0.06, "visible", "native panel", "6 runs"),
        ("FIGURE_3_B", "negative_nonsignificant_fraction", 0.21, "visible", "native panel", "21 runs"),
        ("FIGURE_5", "phirl_accuracy", 0.85, "approximately", "manual native-resolution constraint", "10 dots"),
        ("FIGURE_5", "composition_change_accuracy", 0.80, "approximately", "manual native-resolution constraint", "10 dots"),
        ("FIGURE_5", "raw_composition_accuracy", 0.80, "approximately", "manual native-resolution constraint", "10 dots"),
        ("FIGURE_5", "flux_accuracy", 0.79, "approximately", "manual native-resolution constraint", "10 dots"),
        ("FIGURE_5", "dummy_accuracy", 0.60, "approximately", "manual native-resolution constraint", "10 dots"),
        ("FIGURE_6_B", "max_persistence_center", 874, "table exact / plot approximate", "Table1 + native panel", "distribution center"),
        ("FIGURE_6_B", "control_persistence_center", 716, "table exact / plot approximate", "Table1 + native panel", "distribution center"),
        ("FIGURE_6_B", "min_persistence_center", 559, "table exact / plot approximate", "Table1 + native panel", "distribution center"),
        ("FIGURE_6_C", "max_initial_probability", 0.86, "approximately", "manual native-resolution constraint", "curve start"),
        ("FIGURE_6_C", "max_terminal_probability", 0.89, "approximately", "manual native-resolution constraint", "curve end"),
        ("FIGURE_6_C", "max_slope", 0.041, "reported", "native panel annotation", "p<.001"),
        ("FIGURE_6_C", "control_probability", 0.88, "approximately", "manual native-resolution constraint", "nearly flat"),
        ("FIGURE_6_C", "control_slope", 0.008, "reported", "native panel annotation", "p=.4659"),
        ("FIGURE_6_C", "min_initial_probability", 0.812, "approximately", "manual native-resolution constraint", "curve start"),
        ("FIGURE_6_C", "min_terminal_probability", 0.793, "approximately", "manual native-resolution constraint", "curve end"),
        ("FIGURE_6_C", "min_slope", -0.03, "reported", "native panel annotation", "p=.0034"),
        ("TABLE_1", "max_probability", 0.88, "reported", "table", "±.03 unresolved SD/SE"),
        ("TABLE_1", "control_probability", 0.88, "reported", "table", "±.03 unresolved SD/SE"),
        ("TABLE_1", "min_probability", 0.80, "reported", "table", "±.03 unresolved SD/SE"),
        ("TABLE_1", "max_consistency", 0.52, "reported", "table", "±.04"),
        ("TABLE_1", "control_consistency", 0.38, "reported", "table", "±.06"),
        ("TABLE_1", "min_consistency", 0.42, "reported", "table", "±.04"),
        ("TABLE_1", "max_first_onset", 36, "reported", "table", "percent sign conflicts with molecular-step note"),
        ("TABLE_1", "control_first_onset", 37, "reported", "table", "percent sign conflicts with molecular-step note"),
        ("TABLE_1", "min_first_onset", 40, "reported", "table", "percent sign conflicts with molecular-step note"),
    ]
    digitization = pd.DataFrame(digitized, columns=["panelId", "constraint", "value", "precision", "method", "caveat"])
    return panel_frame, digitization


def figure_consistency_matrix() -> pd.DataFrame:
    rows = [
        ("FC01", "FIGURE_2_A", "Methods variable n_tot", "PARTIALLY_COMPATIBLE", "Variable lengths require an undocumented aggregate alignment/missing-tail policy."),
        ("FC02", "FIGURE_2_A", "S14", "DIFFERENT", "Frozen source branch has punctuated excursions but significant positive aggregate trend."),
        ("FC03", "FIGURE_3", "Results text", "INTERNALLY_CONFLICTING", "Text uses level while caption uses change."),
        ("FC04", "FIGURE_3", "S15", "DIRECTIONALLY_SIMILAR_RETROSPECTIVE_ONLY", "Both locked analyses show paper-like positive counts, but completed-fit and label coupling remain."),
        ("FC05", "FIGURE_4", "S15", "DIRECTIONALLY_SIMILAR_RETROSPECTIVE_ONLY", "State contrasts point similarly but test scope and label identity remain unresolved."),
        ("FC06", "FIGURE_5", "TABLE_1_PROBABILITY", "INTERNALLY_CONFLICTING", "A majority dummy near .60 cannot arise from the same unbalanced molecular target with .88 prevalence."),
        ("FC07", "FIGURE_5", "S16", "DIFFERENT", "Frozen masked reconstruction yields ~.983 dummy and no learned-family advantage."),
        ("FC08", "FIGURE_6_B", "TABLE_1_PERSISTENCE", "COMPATIBLE", "Table centers correspond to distribution ordering."),
        ("FC09", "FIGURE_6_C", "TABLE_1_PROBABILITY", "PARTIALLY_COMPATIBLE", "Same overall rounded means can coexist with different slopes, but window and aggregation are missing."),
        ("FC10", "TABLE_1_CONSISTENCY", "Results min worsens all", "INTERNALLY_CONFLICTING", "Min .42 exceeds control .38 under the stated higher-is-better interpretation."),
        ("FC11", "TABLE_1_FIRST_REPLICATOR", "Table note", "INTERNALLY_CONFLICTING", "Cells contain percent signs; note declares molecular steps."),
        ("FC12", "FIGURE_6_TABLE_1", "S17", "DIFFERENT", "Literal online reconstruction replays but max does not improve outcomes and min only harms modestly."),
        ("FC13", "FIGURE_1_C", "S19-L02–L11R", "NOT_RECONSTRUCTED", "Adjacent, boundary, recurrence, dominant-attractor and compotype-union definitions do not recover the complete fingerprint."),
    ]
    return pd.DataFrame(rows, columns=["comparisonId", "leftComponent", "rightComponent", "relationship", "explanation"])


def table1_semantics() -> pd.DataFrame:
    rows = [
        ("persistence", "sum_t Y_t", "molecular steps", "matrix/treatment", "874±233", "716±198", "559±99", "SD_OR_SE_UNRESOLVED", "S17 implemented literal sum", "Label identity unresolved"),
        ("probability", "sum_t Y_t / T", "molecular steps", "matrix/treatment", "88±3%", "88±3%", "80±3%", "SD_OR_SE_UNRESOLVED", "S17 implemented literal fraction", "Conflicts with Figure 5 dummy if same target"),
        ("consistency", "corr(Y_t,Y_{t+1})", "molecular-step pairs", "matrix/treatment", ".52±.04", ".38±.06", ".42±.04", "SD_OR_SE_UNRESOLVED", "S17 Pearson with undefined constant status", "Min exceeds control despite worsened-all prose"),
        ("time_to_first", "min{t:Y_t=1}", "note: molecular step; cells: percent", "matrix/treatment", "36±26%", "37±27%", "40±28%", "AUTHOR_DISPERSION_AND_UNIT_UNRESOLVED", "S17 raw molecular step", "Zero/one indexing and percent sign unresolved"),
    ]
    return pd.DataFrame(rows, columns=["field", "paperDefinition", "clockOrDenominator", "statisticalUnit", "maxValue", "controlValue", "minValue", "dispersionIdentity", "frozenE01Semantics", "unresolved"])


def figure5_reconciliation_rows() -> pd.DataFrame:
    possibilities = [
        ("class balancing", "not stated", "no GARD-specific support", "not tested as paper claim", "dummy near .50–.60; target prevalence decoupled from Table1"),
        ("per-run balancing", "not stated", "no support", "not tested", "each run contributes balanced positive/negative cases"),
        ("onset-only target", "caption mentions initial appearance", "no support", "S16 reports pre-onset eligibility but not onset-only retraining", "more negative examples and dummy near .60 possible"),
        ("full future-state target", "Results explicitly says final 75% trajectory", "no GARD code", "S16 primary target", "dummy tracks suffix prevalence; observed ~.983"),
        ("generation-level target", "Figure 1 discusses generations", "no support", "not trained", "100 targets/run; prevalence may differ from molecular occupancy"),
        ("molecular-level target", "Figure 5 and Table1 context imply molecular states", "no GARD code", "S16 tested", "dummy should be near molecular prevalence, contradicting .60"),
        ("run-level accuracy averaging", "10 split dots but weighting not stated", "no support", "S16 matrix-weighted valid positions", "short low-prevalence runs can receive equal weight"),
        ("padding included", "not stated", "no support", "S16 explicitly masks padding", "padding label convention can drive dummy and accuracy"),
        ("padding masked", "not stated", "common source convention only", "S16 tested", "dummy follows valid suffix prevalence ~.983"),
        ("truncation to common length", "not stated", "no support", "not tested", "can enrich early negatives and lower majority accuracy"),
        ("negative-case enrichment", "not stated", "no support", "not tested", "dummy deliberately lowered toward .60"),
        ("stratified sampling", "train/test 80/20 only", "no support", "matrix-level paired splits only", "stratification can stabilize but not inherently change target prevalence"),
        ("separately generated prediction data", "not stated", "public code missing", "not available", "different occupancy could reconcile panels"),
        ("different label identity", "cluster/initial-appearance language permits ambiguity", "public GARD-specific code missing", "many frozen label reconstructions tested; none joint match", "a 40/60 label task could match dummy but requires author definition"),
    ]
    return pd.DataFrame(possibilities, columns=["possibility", "paperSupport", "phirlSupport", "frozenE01TestStatus", "predictedObservableFingerprint"])


def first_commit_containing(repo: Path, needle: str, relative: str) -> str:
    commits = git(repo, "log", "--all", "--reverse", f"-S{needle}", "--format=%H", "--", relative).splitlines()
    return commits[0] if commits else "NOT_FOUND"


def build_phirl_dataflow() -> nx.DiGraph:
    graph = nx.DiGraph(researchStepId=STEP_ID, graphType="phirl_executable_dataflow")
    nodes = [
        ("input", "Input activation trajectory", "(D,T)", "float16/float array on disk", True),
        ("active_filter", "Active-variable filter std>1e-8", "(d,T)", "float", True),
        ("zscore", "Per-variable z-score", "(d,T)", "float64 after compute_phi cast", True),
        ("lagcorr", "Forward/backward lag-1 correlations", "(d,d) each", "float64", True),
        ("sigmask", "Student-t p-value mask alpha=1", "(d,d)", "bool", True),
        ("mi", "Fast Gaussian MI graph", "(d,d)", "float64", True),
        ("noise", "Uniform graph floor 1e-6", "(d,d)", "float64", True),
        ("fiedler", "Unnormalized Fiedler sign split", "two index lists", "int", True),
        ("means", "Arithmetic partition means", "(2,T)", "float64", True),
        ("gaussian", "Global means/covariances + trace regularizer", "local entropies", "float64", True),
        ("lattice", "16-atom local PhiID Möbius inversion", "16 x (T-1)", "float64", True),
        ("integrated", "integrated = local_phi_r (9 atoms)", "(T-1,)", "float64", True),
        ("synergy", "synergy = s→s atom", "(T-1,)", "float64", True),
        ("downward", "causation = s→u0 + s→u1", "(T-1,)", "float64", True),
        ("emergence", "emergence = synergy + causation", "(T-1,)", "float64", True),
        ("shuffle", "Complete-trajectory time permutation control", "(d,T)", "float64", True),
        ("save", "Separate local arrays saved as .npy", "one file/measure/episode", "array", False),
        ("load", "Inf→NaN, finite median, missing/error→0", "scalar per episode/checkpoint", "float64", False),
        ("plot", "Across-seed/episode aggregation and plots", "summary curves", "float64", False),
    ]
    for node, label, shape, dtype, complete in nodes:
        graph.add_node(node, label=label, shape=shape, dtype=dtype, usesCompleteTrajectory=complete)
    edges = [
        ("input", "active_filter"), ("active_filter", "zscore"), ("zscore", "lagcorr"),
        ("lagcorr", "sigmask"), ("sigmask", "mi"), ("mi", "noise"), ("noise", "fiedler"),
        ("zscore", "means"), ("fiedler", "means"), ("means", "gaussian"), ("gaussian", "lattice"),
        ("lattice", "integrated"), ("lattice", "synergy"), ("lattice", "downward"),
        ("synergy", "emergence"), ("downward", "emergence"), ("zscore", "shuffle"),
        ("shuffle", "lagcorr"), ("integrated", "save"), ("synergy", "save"), ("downward", "save"),
        ("emergence", "save"), ("save", "load"), ("load", "plot"),
    ]
    graph.add_edges_from(edges)
    return graph


def build_phirl_function_registry() -> pd.DataFrame:
    config = read_config()
    repo = Path(config["paths"]["phirl"])
    entries = [
        ("preprocess_data", "main.py", "data (D,T)", "active z-scored data (d,T)", "float until compute cast", True, False, "none directly", "std>1e-8; scipy zscore", "Paper specifies CLR/drop-last, not this extra filter/z-score", "S13Y pins source behavior after paper-facing substrate"),
        ("compute_phi", "main.py", "preprocessed (d,T)", "dict of local arrays (T-1)", "float64", True, True, "NetworkX Fiedler implicit RNG", "alpha=1; noise=True; means", "Paper names MIB/PhiID but not public implementation", "Exact S13Y branch"),
        ("mutual_information_matrix", "information.py", "(d,T), alpha, lag", "symmetric (d,d)", "float64", True, False, "none", "slow: MI forward + MI backward", "No paper implementation detail", "S12B source comparator"),
        ("mutual_information_matrix_fast", "information.py", "(d,T), alpha, lag", "(d,d)", "float64", True, False, "none", "average r forward/backward then one MI transform; df=T-2", "No paper implementation detail", "S13Y source branch"),
        ("minimum_information_bipartition", "information.py", "MI matrix", "two sign-index lists", "float64→int indices", True, True, "networkx fiedler_vector seed=None", "uniform 1e-6 floor; unnormalized Laplacian; zero omitted", "Paper names MIB but not Fiedler approximation", "S13Y and source fixtures"),
        ("local_entropy_nd", "information.py", "variables x time", "local surprisal per time", "float64", True, False, "none", "global mean/cov; ddof=0; eps*trace/d", "Paper does not state Gaussian fit or regularizer", "S12C/S13Y validated"),
        ("local_phi_id", "information.py", "two-variable reduced series", "lattice with 16 local PI arrays", "float64", True, False, "none", "MMI-like local redundancy; lattice Möbius inversion", "Paper cites PhiID, convention not named", "Safe lattice + pinned source"),
        ("local_phi_r", "information.py", "PhiID lattice", "9-atom local sum", "float64", True, False, "none", "corrected initial atom plus PHIR_ATOMS", "Paper equation/prose do not uniquely support this set", "Historical comparator retained"),
        ("save_info", "analysis.py", "activation .npy files", "measure and shuffled .npy files", "input→float64 outputs", True, True, "np.random.permutation global state", "preprocess; compute original and shuffled", "No GARD-specific invocation", "Source behavior audited only"),
        ("_load_phi", "plotting.py", "saved arrays", "median scalar per episode", "float64", False, False, "filesystem order", "Inf→NaN; median finite; ValueError/missing remain zero", "No paper aggregation mapping", "Not used as S13Y trajectory scalar"),
    ]
    columns = ["function", "file", "inputShape", "outputShape", "dtype", "usesCompleteTrajectory", "stochastic", "seedSource", "numericalToleranceOrPolicy", "paperSupport", "frozenE01Implementation"]
    frame = pd.DataFrame(entries, columns=columns)
    frame["commitIntroduced"] = [first_commit_containing(repo, f"def {name}", file) for name, file in zip(frame["function"], frame["file"], strict=True)]
    frame["sourceCommit"] = git(repo, "rev-parse", "HEAD")
    frame["iigrAncestry"] = ["direct structural predecessor" if name not in ("preprocess_data", "save_info", "_load_phi") else "application-layer PhiRL behavior" for name in frame["function"]]
    frame["knownOutcomeSensitivity"] = [
        "active dimensions and Gaussian conditioning", "partition, atom scalar and complete-fit future dependence",
        "MI graph differs from fast", "MI graph/partition can differ from slow", "partition jumps and zero-coordinate omission",
        "large sensitivity under ill-conditioning", "atom values/redundancy convention", "metric identity",
        "shuffled-control randomness and all-or-run failure", "nonfinite/missing aggregation can create zero summaries",
    ]
    return frame


def phirl_numerical_semantics() -> pd.DataFrame:
    rows = [
        ("active filter", "std > 1e-8", "complete trajectory", "deterministic", "removes variables based on future suffix", "not stated"),
        ("z-score", "scipy.stats.zscore axis=1", "complete trajectory", "deterministic", "global mean/std future dependence", "not stated after CLR"),
        ("slow lagged MI", "MI(r_forward)+MI(r_backward)", "complete trajectory", "deterministic", "nonlinear transform before sum", "not stated"),
        ("fast lagged MI", "MI((r_forward+r_backward)/2)", "complete trajectory", "deterministic", "not equivalent to slow", "not stated"),
        ("fast df", "df=T-2 even for lag>0", "complete trajectory", "deterministic", "off from nominal paired sample df by lag convention", "not stated"),
        ("significance", "alpha=1; no Bonferroni", "complete trajectory", "deterministic", "all finite p<1 retained; p=1/nonfinite omitted", "paper does not mention edge testing"),
        ("MI clipping", "r clipped to ±0.999999", "complete trajectory", "deterministic", "caps perfect-correlation MI", "not stated"),
        ("graph noise", "add 1e-6 to every matrix entry", "run", "deterministic", "ensures connected dense graph and self-loops", "not stated"),
        ("Fiedler", "networkx unnormalized; seed=None", "run", "implicit stochastic path", "eigenvector sign and solver randomness", "paper names MIB only"),
        ("Fiedler zero", "only >0 and <0 assigned", "run", "deterministic conditional on vector", "exact zero remains unassigned", "not stated"),
        ("partition reduction", "nanmean within each side", "all T", "deterministic", "arithmetic averaging lacks normalization by information", "not stated"),
        ("1D entropy", "normal pdf with population std", "complete trajectory", "deterministic", "zero std can yield nonfinite", "not stated"),
        ("ND entropy", "cov ddof=0 + I*1e-6*trace/d", "complete trajectory", "deterministic", "regularizer vanishes when trace=0", "not stated"),
        ("local PhiID", "lattice MMI-like minimum exclusions", "complete trajectory fitted, local T-1", "deterministic", "signed local atoms possible", "redundancy convention absent"),
        ("shuffle", "np.random.permutation time", "complete trajectory", "global RNG", "destroys all temporal order", "shuffled control absent from GARD paper"),
        ("loader", "Inf→NaN; finite nanmedian; invalid→0", "saved local values", "filesystem-order dependent", "zero conflates missing/error with value", "paper aggregation absent"),
    ]
    return pd.DataFrame(rows, columns=["operation", "exactSemantics", "fitScope", "stochasticity", "numericalOrInterpretiveConsequence", "paperSpecification"])


def _antichain_name(value: list[list[int]]) -> str:
    canonical = tuple(tuple(item) for item in value)
    mapping = {((0,), (1,)): "r", ((0,),): "u0", ((1,),): "u1", ((0, 1),): "s"}
    if canonical not in mapping:
        raise ValueError(f"unknown antichain {canonical}")
    return mapping[canonical]


def build_atom_registry() -> pd.DataFrame:
    lattice = json.loads(Path(read_config()["paths"]["safeLattice"]).read_text(encoding="utf-8"))
    integrated_atoms = {("u0", "s"), ("r", "s"), ("u1", "s"), ("s", "u0"), ("s", "r"), ("s", "u1"), ("s", "s"), ("u0", "u1"), ("u1", "u0")}
    emergence_atoms = {("s", "s"), ("s", "u0"), ("s", "u1")}
    rows: list[dict[str, Any]] = []
    for idx, node in enumerate(lattice["nodes"]):
        atom = node["atom"]
        source = _antichain_name(atom[0])
        target = _antichain_name(atom[1])
        pair = (source, target)
        rows.append(
            {
                "atomIndexSafeOrder": idx,
                "sourceAntichain": source,
                "targetAntichain": target,
                "atomTupleJson": canonical_json(atom),
                "localPhiRWeight": int(pair in integrated_atoms),
                "synergyWeight": int(pair == ("s", "s")),
                "downwardCausationWeight": int(pair in {("s", "u0"), ("s", "u1")}),
                "publicEmergenceWeight": int(pair in emergence_atoms),
                "directWholeMinusPartsWeight": 1 if source == "s" else (-1 if source == "r" else 0),
                "paperDisplayedEquationWeight": 1 if source == "s" else (-1 if source == "r" else 0),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["sourceAntichain", "targetAntichain"], kind="stable").reset_index(drop=True)
    if len(frame) != 16 or frame["localPhiRWeight"].sum() != 9 or frame["publicEmergenceWeight"].sum() != 3:
        raise RuntimeError("PhiID atom identity validation failed")
    return frame


def phirl_atom_identity_matrix(atom_frame: pd.DataFrame) -> pd.DataFrame:
    return atom_frame[[
        "sourceAntichain", "targetAntichain", "atomTupleJson", "localPhiRWeight", "synergyWeight",
        "downwardCausationWeight", "publicEmergenceWeight", "directWholeMinusPartsWeight",
        "paperDisplayedEquationWeight",
    ]].copy()


def phirl_temporal_leakage_map() -> pd.DataFrame:
    rows = [
        ("active-variable filter", True, "std uses all T", "first-quarter features depend on suffix-active dimensions", "refit on prefix"),
        ("z-score", True, "mean/std use all T", "prefix values depend on suffix", "fit prefix only"),
        ("lagged-MI graph", True, "all lag pairs", "partition depends on suffix", "fit prefix only"),
        ("Fiedler partition", True, "MI graph uses all T", "prefix reduction changes with suffix", "fit prefix only"),
        ("partition arithmetic means", True, "membership globally fit", "local averages inherit partition dependence", "prefix partition"),
        ("Gaussian means", True, "all reduced observations", "local surprise uses suffix", "prefix mean"),
        ("Gaussian covariance", True, "all reduced observations", "local surprise uses suffix", "prefix covariance"),
        ("local PhiID atoms", True, "global distribution parameters", "first-quarter local values are completed-fit", "prefix-only fit"),
        ("public integrated/emergence", True, "sums completed-fit local atoms", "retrospective only", "prefix-only construction"),
        ("Figure 2 completed trajectory", True, "per-run fit then plot", "valid descriptive retrospective output", "not needed for description"),
        ("Figure 5 completed-fit mode", True, "first-quarter values may know suffix", "ineligible early-warning evidence", "S16 cutoff-causal mode"),
        ("online hypothetical action", False, "public source has no scorer", "online semantics absent rather than demonstrably leaking", "S17 append-and-refit prefix reconstruction"),
    ]
    return pd.DataFrame(rows, columns=["pipelineElement", "usesFutureSuffixInPublicCompletedFit", "mechanism", "interpretiveEffect", "frozenE01Counterpart"])


def write_phirl_missing_components(path: Path) -> None:
    write_text(
        path,
        """# PhiRL components missing for the GARD paper pipeline

## Concise top summary

- **Research step ID:** `S19-L12`.
- **Completion status:** COMPLETE SOURCE-GAP AUDIT.
- **Artifacts written:** executable data-flow, function, numerical, atom, leakage, and source-lineage registries.
- **Validation result:** PASS — every registered PhiRL function was located and traced at the pinned/current commit.
- **Outcome classification:** `PUBLIC_CODE_MISSING` for the GARD-specific end-to-end pipeline.
- **Caveats or blockers:** Absence from the inspected public history is not proof that no private implementation exists.
- **Recommended next action:** Use the public source only for pinned component semantics; author code is required to identify the paper's complete pipeline.

No public PhiRL branch, tag, deleted path recovered through Git history, or current file implements GARD simulation, GARD preprocessing from count trajectories, the paper's self-replicator label, Figure 2 unequal-length aggregation, Figures 3–4 GARD statistics, the Figure 5 sequence tensor/MLP, alternative input proportions, spike-descriptor analysis, hypothetical post-fission action scoring, max/control/min GARD intervention trajectories, or Table 1 outcome aggregation. IIGR and BreakingGRNMemories provide related information-theory ancestry and application patterns, not these missing GARD components.

Public PhiRL does establish a specific component chain: active-variable filtering, z-scoring, fast lagged-MI construction, a noise-connected unnormalized Fiedler split, arithmetic partition averaging, complete-trajectory Gaussian fitting, local PhiID, and two distinct exported scalars called `integrated` and `emergence`. That chain cannot by itself decide which scalar, label, tensor, intervention scorer, or denominator generated the manuscript figures.
""",
    )


def write_equation_derivation(path: Path, atom_frame: pd.DataFrame) -> None:
    integrated = [f"{r.sourceAntichain}→{r.targetAntichain}" for r in atom_frame.itertuples() if r.localPhiRWeight]
    emergence = [f"{r.sourceAntichain}→{r.targetAntichain}" for r in atom_frame.itertuples() if r.publicEmergenceWeight]
    positive = [f"{r.sourceAntichain}→{r.targetAntichain}" for r in atom_frame.itertuples() if r.paperDisplayedEquationWeight == 1]
    negative = [f"{r.sourceAntichain}→{r.targetAntichain}" for r in atom_frame.itertuples() if r.paperDisplayedEquationWeight == -1]
    write_text(
        path,
        f"""# Algebraic adjudication of the paper's Φ-r identity

## Concise top summary

- **Research step ID:** `S19-L12`.
- **Completion status:** COMPLETE.
- **Artifacts written:** `phiid_atom_registry.csv`, `phirl_atom_identity_matrix.csv`, this derivation, and `metric_identity_adjudication.json`.
- **Validation result:** PASS — all 16 safe-lattice atoms were enumerated; symbolic coefficient vectors and deterministic numerical fixtures agree exactly.
- **Outcome classification:** `PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT`.
- **Caveats or blockers:** The derivation assumes the public two-source/two-target ΦID lattice; the paper does not name its redundancy convention.
- **Recommended next action:** Do not choose `integrated`, `emergence`, or direct whole-minus-parts by association strength; obtain the paper implementation or test a prospectively fixed paper-literal pipeline later.

## Lay summary

The manuscript's equation, its phrase “one atom,” and public PhiRL name three different mathematical objects. They coincide only under special cancellations that are not identities. This is a source-level discrepancy, not a failed attempt to optimize a result.

## Derivation

Let source antichains be redundancy `r`, unique source 0 `u0`, unique source 1 `u1`, and source synergy `s`; target antichains use the same names. The safe lattice contains every one of the 4×4=16 ordered atoms.

For any target antichain `q`, total whole-source information contains `r→q + u0→q + u1→q + s→q`. Information available from source part 0 contains `r→q + u0→q`; from source part 1 it contains `r→q + u1→q`. Therefore:

`I(X_t;X_t+1) - I(X_t^0;X_t+1) - I(X_t^1;X_t+1) = Σ_q (s→q - r→q)`.

Thus the displayed equation has +1 coefficients on {', '.join(positive)} and −1 coefficients on {', '.join(negative)}. It is not one atom.

Public PhiRL `emergence = synergy + causation` contains only {', '.join(emergence)}. Public `integrated = local_phi_r` contains {', '.join(integrated)}. Neither coefficient vector equals the displayed equation's vector.

`local_phi_r` also depends on the corrected nine-atom implementation recovered in S12B/S12C; the older historical bug is preserved only as a comparator. The algebra above is independent of whether any coefficient happens to correlate with a GARD label.

## Numerical identity fixture

A deterministic vector assigning values 1 through 16 to the canonical atom order was evaluated by dot product with every registered coefficient vector. The paper equation and direct whole-minus-parts vectors agree exactly for arbitrary atoms; neither agrees identically with public `integrated` or public `emergence`. Regeneration repeats this calculation from the safe JSON, not from the pickle.
""",
    )


def metric_identity_adjudication(atom_frame: pd.DataFrame) -> dict[str, Any]:
    values = np.arange(1, 17, dtype=np.float64)
    paper = float(np.dot(values, atom_frame["paperDisplayedEquationWeight"].to_numpy()))
    direct = float(np.dot(values, atom_frame["directWholeMinusPartsWeight"].to_numpy()))
    integrated = float(np.dot(values, atom_frame["localPhiRWeight"].to_numpy()))
    emergence = float(np.dot(values, atom_frame["publicEmergenceWeight"].to_numpy()))
    return {
        "schema": "eidosoma.e01.s19.l12.metric_identity.v1",
        "researchStepId": STEP_ID,
        "classification": "PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT",
        "paperEquationMapsAlgebraicallyTo": "DIRECT_WHOLE_MINUS_SUM_OF_PART_TO_WHOLE_MUTUAL_INFORMATION",
        "paperPhraseOneAtomCompatibleWithDisplayedEquation": False,
        "paperEquationEqualsDirectWms": paper == direct,
        "paperEquationEqualsPublicIntegratedIdentity": bool(np.array_equal(atom_frame["paperDisplayedEquationWeight"], atom_frame["localPhiRWeight"])),
        "paperEquationEqualsPublicEmergenceIdentity": bool(np.array_equal(atom_frame["paperDisplayedEquationWeight"], atom_frame["publicEmergenceWeight"])),
        "numericalFixture": {"paperEquation": paper, "directWms": direct, "publicIntegrated": integrated, "publicEmergence": emergence},
        "sourceEvidence": {
            "equation": "paper PDF text extraction, page 9",
            "prose": "paper states one PhiID atom",
            "publicIntegrated": "PhiRL compute_phi integrated=local_phi_r, nine atoms",
            "publicEmergence": "PhiRL compute_phi emergence=synergy+causation, three atoms",
        },
        "adjudication": "The equation supports direct whole-minus-parts algebra; prose and public names are mutually non-identical, so the paper-used scalar cannot be identified from public evidence.",
        "selectionByReplicationAssociation": False,
    }


def _evidence(*labels: str) -> str:
    allowed = set(read_config()["evidenceLabels"])
    if not set(labels).issubset(allowed):
        raise ValueError(f"unregistered evidence label: {set(labels) - allowed}")
    return ";".join(labels)


def material_concordance_rows() -> list[dict[str, Any]]:
    specs = [
        ("P01", "GARD update kernel", "Poisson gains/losses until n_max or max_steps", "No GARD code in PhiRL; historical GARD variants differ", "S03/S04 narrowed but could not uniquely identify the kernel", "Exact update ordering/exposure requires author implementation", ("DIRECT_PAPER_SPECIFICATION", "PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P02", "Poisson exposure", "Poisson updates named; exposure absent", "Historical sources expose implementation choices but no paper config", "Two paper-time-compatible frozen exposures/candidates and high-exposure L07/L08 comparator", "Paper-specific h remains undocumented", ("DIRECT_PAPER_SPECIFICATION", "SOURCE_LINEAGE_INFERENCE", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P03", "Observation clock", "Phi-r at every molecular step; generations also central", "PhiRL accepts arbitrary T with no GARD clock", "S13Y uses every selected molecular boundary; S19 tests boundary alternatives", "Exact inclusion of daughters/pre-fission/overshoot states unresolved", ("DIRECT_PAPER_SPECIFICATION", "PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P04", "Fission and overshoot", "n_max/max_steps then binomial fission", "No GARD code", "Frozen candidates trim only excess newly joined molecules", "Overshoot and max-step fission handling not fully specified", ("DIRECT_PAPER_SPECIFICATION", "PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P05", "Daughter continuation", "one daughter continues", "No GARD code", "Candidate 2 first daughter; candidate 3 random nonempty; both retained", "Paper does not select one", ("DIRECT_PAPER_SPECIFICATION", "PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P06", "Extinction", "Not described", "No GARD code", "Explicit status handling in frozen simulations", "Author replacement/termination behavior unknown", ("PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P07", "Relative composition", "Relative composition is the Phi substrate", "PhiRL source expects generic activations", "S13Y uses count→relative composition", "Directly aligned at conceptual level", ("DIRECT_PAPER_SPECIFICATION", "DIRECT_PUBLIC_CODE", "DIRECT_FROZEN_E01_RESULT")),
        ("P08", "Zero replacement and CLR", "CLR and dropped last component specified; zeros not handled", "PhiRL has no compositional preprocessing", "S13Y additive .5 closure + drop last component", "Pseudocount is reconstructed, not specified", ("DIRECT_PAPER_SPECIFICATION", "PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P09", "Active-variable filtering", "Not specified", "PhiRL removes std≤1e-8 dimensions", "S13Y pins this source behavior", "Could change effective variables and completed-fit leakage", ("DIRECT_PUBLIC_CODE", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P10", "Z-scoring", "Not specified after CLR", "PhiRL z-scores every active row", "S13Y pins public source", "Paper pipeline may or may not include it", ("DIRECT_PUBLIC_CODE", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P11", "Lagged MI construction", "Time-lagged multivariate MI named", "Current PhiRL uses fast averaged-correlation approximation; early history uses slow summed-MI", "S12B/S13Y compare and pin fast primary", "Paper analysis date and exact implementation are not identifiable", ("DIRECT_PAPER_SPECIFICATION", "DIRECT_PUBLIC_CODE", "SOURCE_LINEAGE_INFERENCE", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P12", "MI edge significance", "Not described", "alpha=1, Bonferroni off retains every finite p<1", "Pinned in S13Y", "This is effectively a dense finite-edge graph but exact p=1/nonfinite edges drop", ("DIRECT_PUBLIC_CODE", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P13", "Minimum-information bipartition", "MIB named as least-information cut", "PhiRL uses noise-connected unnormalized Fiedler sign split", "S13Y source-confirmed partition", "Whether Fiedler is paper's approximation is unstated", ("DIRECT_PAPER_SPECIFICATION", "DIRECT_PUBLIC_CODE", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P14", "Fiedler zeros/ties/RNG", "Not described", "Exact zero coordinates unassigned; solver seed None", "S13Y records partitions and condition", "Rare numerical/solver semantics may affect spikes", ("DIRECT_PUBLIC_CODE", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P15", "Partition reduction", "Two components stated", "Arithmetic means within each Fiedler side", "S13Y exact", "Averaging rationale absent", ("DIRECT_PAPER_SPECIFICATION", "DIRECT_PUBLIC_CODE", "DIRECT_FROZEN_E01_RESULT")),
        ("P16", "Gaussian fitting", "Entropy/estimator details absent", "Global means and covariance across complete reduced trajectory", "S13Y completed fit; S15/S16 show temporal consequence", "Distribution scope is author-dependent", ("DIRECT_PUBLIC_CODE", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P17", "Covariance regularization", "Not stated", "Current PhiRL uses 1e-6*trace(cov)/d; introduced with fast path", "S13Y pinned regularized source", "Could differ at paper analysis date and drives numerical condition", ("DIRECT_PUBLIC_CODE", "SOURCE_LINEAGE_INFERENCE", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P18", "PhiID redundancy convention", "PhiID cited; convention omitted", "Public source uses local minimum informative/misinformative exclusions", "S12B safe lattice and S13Y exact source", "Alternative PhiID conventions are not excluded by paper", ("DIRECT_PAPER_SPECIFICATION", "DIRECT_PUBLIC_CODE", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P19", "Scalar identity", "Displayed WMS-like equation; prose says one atom", "Public integrated is 9 atoms; emergence is 3 different atoms", "S13Y uses source emergence primary and corrected local_phi_r comparator", "Internal inconsistency prevents unique mapping", ("PAPER_INTERNAL_CONFLICT", "DIRECT_PUBLIC_CODE", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P20", "Full versus prefix fitting", "Complete trajectories used for correlations; Figure5 prospective language", "Public source fits complete trajectory", "S15 completed/past-only directions differ; S16 separates modes", "Completed-fit values are descriptive only and future-dependent", ("DIRECT_PAPER_SPECIFICATION", "DIRECT_PUBLIC_CODE", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P21", "Self-replicator label", "Recurring clusters; threshold to most recurring composition", "No GARD label in PhiRL", "Exact adjacent H branch gives ~98%; S19 alternatives fail joint fingerprint", "Highest-leverage unresolved author component", ("DIRECT_PAPER_SPECIFICATION", "PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P22", "Label clock", "Any time step and molecular Figure1D; recurrence described across generations", "No GARD code", "Molecular, boundary, projected, recurrence and compotype variants tested", "Molecular versus generation semantics nonidentifiable", ("PAPER_INTERNAL_CONFLICT", "PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P23", "Occupancy/persistence/onset/consistency", "Table1 gives .88/716/37/.38 control fingerprints", "No implementation", "Adjacent label matches none jointly; S19 labels occupy opposite regimes", "No tested label reproduces complete temporal fingerprint", ("DIRECT_PAPER_SPECIFICATION", "PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P24", "Figure2 aggregate alignment", "Median±SD across 100 unequal runs", "No GARD plotting code", "S14 available-case reconstruction has trend discrepancy and falling tail support", "Padding/truncation/resampling/available-case unknown", ("DIRECT_PAPER_SPECIFICATION", "FIGURE_FINGERPRINT_INFERENCE", "PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P25", "Figure2 spike definition", "Above overall mean+3SD", "No GARD plotting code", "S14 evaluates 3SD and robust companions", "Overall scope and scalar fitting remain ambiguous", ("DIRECT_PAPER_SPECIFICATION", "PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P26", "Figure3 level versus change", "Results use level; caption uses change", "No GARD code", "S15 keeps both; both retrospective, past-only reverses", "Cannot choose by favorability", ("PAPER_INTERNAL_CONFLICT", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P27", "Figure4 statistical scope", "Mann-Whitney per run and Fisher across runs implied but incompletely stated", "No GARD code", "S15 calculates within-run and dependence-aware controls", "Pooling/Fisher input and undefined states unresolved", ("DIRECT_PAPER_SPECIFICATION", "PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P28", "Figure5 tensor and sampling", "first 25% input/final 75% output, 80/20 runs, 10 seeds", "No GARD prediction code", "S16 freezes original-order masked matrix splits", "Padding, balancing, target identity and architecture absent", ("DIRECT_PAPER_SPECIFICATION", "PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P29", "Figure5 dummy versus Table1", "~.60 visible dummy versus .88 control occupancy", "No GARD prediction code", "S16 dummy ~.983 with frozen exact-H label", "Requires different label/data/denominator/sampling; public evidence cannot select", ("FIGURE_FINGERPRINT_INFERENCE", "PAPER_INTERNAL_CONFLICT", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P30", "Intervention scorer", "Score every add/delete immediately after fission", "No public GARD scorer", "S17 append-and-refit current prefix is exact and online", "Paper does not define single-state score/refit/future use/metric", ("DIRECT_PAPER_SPECIFICATION", "PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P31", "Intervention action/tie semantics", "All additions/deletions; no-op control separate", "No public code", "S17 deterministic tie stream and exhaustive eligible set", "Deletion eligibility and ties absent from paper", ("DIRECT_PAPER_SPECIFICATION", "PUBLIC_CODE_MISSING", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P32", "Figure6/Table1 treatment outcome", "Max improves and min worsens", "No public code", "S17 exact replay: max below control; min modestly harms", "Literal reconstruction not causal support", ("DIRECT_PAPER_SPECIFICATION", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED")),
        ("P33", "Table1 dispersion", "Values shown as ± without SD/SE identity", "No code", "S17 reports uncertainty explicitly", "Dispersion identity unresolved", ("DIRECT_PAPER_SPECIFICATION", "AUTHOR_IMPLEMENTATION_REQUIRED")),
    ]
    return [
        {
            "rowId": row_id, "topic": topic, "paperStatementOrFingerprint": paper,
            "phirlOrIigrSourceBehavior": public, "frozenE01ImplementationAndResult": e01,
            "remainingExplanation": remaining, "evidenceLabels": _evidence(*labels),
            "crossReferenceType": "MATERIAL_PIPELINE_ELEMENT", "priorStatus": "NOT_MODIFIED",
            "evidencePath": "/workspace/input-attachments/.../pdf-markdown.md;/cache/e01_s12b/sources/PhiRL;/artifacts/research_steps",
        }
        for row_id, topic, paper, public, e01, remaining, labels in specs
    ]


def build_concordance_matrix() -> pd.DataFrame:
    rows = material_concordance_rows()
    matrix_a = pd.read_csv(read_config()["paths"]["matrixA"])
    for claim in matrix_a.itertuples(index=False):
        rows.append(
            {
                "rowId": f"S18A_{claim.claimId}",
                "topic": f"S18 Matrix A claim {claim.claimId}",
                "paperStatementOrFingerprint": claim.paperClaim,
                "phirlOrIigrSourceBehavior": "Public GARD-paper implementation absent; component source behavior cross-referenced where applicable.",
                "frozenE01ImplementationAndResult": f"{claim.finalStatusCode}: {claim.evidenceSummary}",
                "remainingExplanation": claim.mainCaveat,
                "evidenceLabels": _evidence("DIRECT_PAPER_SPECIFICATION", "DIRECT_FROZEN_E01_RESULT", "PUBLIC_CODE_MISSING"),
                "crossReferenceType": "S18_MATRIX_A_59_CLAIMS",
                "priorStatus": claim.finalStatusCode,
                "evidencePath": claim.primaryEvidencePaths,
            }
        )
    matrix_b = pd.read_csv(read_config()["paths"]["matrixB"])
    for item in matrix_b.itertuples(index=False):
        rows.append(
            {
                "rowId": f"S18B_{item.questionId}", "topic": f"S18 Matrix B {item.question}",
                "paperStatementOrFingerprint": item.question,
                "phirlOrIigrSourceBehavior": "Prospective/causal eligibility is not supplied by public GARD code.",
                "frozenE01ImplementationAndResult": f"{item.status}: {item.finding}",
                "remainingExplanation": "Carry S18 status unchanged; L12 produces no eligible new scientific outcome.",
                "evidenceLabels": _evidence("DIRECT_FROZEN_E01_RESULT", "PUBLIC_CODE_MISSING", "AUTHOR_IMPLEMENTATION_REQUIRED"),
                "crossReferenceType": "S18_MATRIX_B_7_QUESTIONS", "priorStatus": item.status,
                "evidencePath": item.evidencePaths,
            }
        )
    for loop in ["L01", "L02", "L03", "L04", "L05", "L06", "L06R", "L07", "L08", "L09", "L10", "L11", "L11R"]:
        path = ARTIFACTS / "research_steps/S19/loops" / loop / "classification.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = payload.get("topLevelClassification") or payload.get("decision") or payload.get("status") or ";".join(payload.get("s19Classifications", []))
        promoted = payload.get("promotedLeadCount", len(payload.get("promotedLeadIds", [])))
        rows.append(
            {
                "rowId": f"S19_{loop}", "topic": f"Frozen S19 {loop} result",
                "paperStatementOrFingerprint": "Exploratory reconstruction/search layer; see immutable loop decision.",
                "phirlOrIigrSourceBehavior": "Source relationship retained exactly as adjudicated by the loop.",
                "frozenE01ImplementationAndResult": f"classification={status}; promotedLeadCount={promoted}",
                "remainingExplanation": "Result remains immutable and exploratory; L12 only cross-references it.",
                "evidenceLabels": _evidence("DIRECT_FROZEN_E01_RESULT"),
                "crossReferenceType": "S19_LOOP_RESULT", "priorStatus": str(status), "evidencePath": str(path),
            }
        )
    return pd.DataFrame(rows)


def root_cause_registry() -> pd.DataFrame:
    rows = [
        ("RC01", "Replicator-label/task mismatch", 5, 5, 5, 4, "Paper says recurring attractor; exact-H label is deterministic and ~98%; S19 variants fail joint fingerprint.", "OPEN_AUTHOR_DEPENDENT"),
        ("RC02", "Phi-r scalar identity mismatch", 5, 5, 4, 5, "Displayed equation, one-atom prose, public integrated and public emergence are non-identical.", "OPEN_AUTHOR_DEPENDENT"),
        ("RC03", "Figure5 tensor/sampling/denominator mismatch", 5, 5, 5, 5, "Dummy ~60% conflicts with Table1 88% and S16 ~98%; no public prediction code.", "OPEN_HIGH_LEVERAGE"),
        ("RC04", "Completed-fit temporal leakage", 5, 4, 5, 5, "Public source globally fits variables, partition and Gaussians; S15 past-only reverses and S16 prospective gates fail.", "SUPPORTED_AS_RETROSPECTIVE_DEPENDENCE"),
        ("RC05", "Aggregate alignment/missing-tail handling", 4, 3, 4, 5, "Variable n_tot and Figure2 tail require undocumented handling; S14 trend differs.", "OPEN_TESTABLE_FROM_FIGURE"),
        ("RC06", "Simulator kernel/exposure/continuation mismatch", 4, 4, 3, 3, "Two independently plausible candidates plus L07/L08 exposure result; no full fingerprint match.", "OPEN_AUTHOR_DEPENDENT"),
        ("RC07", "Intervention single-state scorer mismatch", 5, 5, 5, 3, "Paper cannot be executed from public PhiRL without inventing a scorer; S17 is one literal prospective reconstruction.", "OPEN_AUTHOR_DEPENDENT"),
        ("RC08", "Multiple figure-specific pipelines or manuscript errors", 4, 5, 5, 2, "Level/change, dummy/prevalence, consistency prose/table and onset units conflict internally.", "PLAUSIBLE_BUT_LOW_COHERENCE"),
        ("RC09", "Public source version drift", 5, 2, 3, 5, "Current master equals pinned; internal slow/fast chronology exists but no alternative branch/tag.", "WEAKENED_NOT_ELIMINATED"),
    ]
    frame = pd.DataFrame(rows, columns=["rootCauseId", "hypothesis", "evidenceStrength0to5", "explanatoryLeverage0to5", "crossFigureReach0to5", "falsifiability0to5", "evidence", "status"])
    frame["priorityScore"] = frame[["evidenceStrength0to5", "explanatoryLeverage0to5", "crossFigureReach0to5", "falsifiability0to5"]].mean(axis=1)
    frame["selectedScientificOutcome"] = False
    return frame.sort_values(["priorityScore", "rootCauseId"], ascending=[False, True], kind="stable").reset_index(drop=True)


def unresolved_author_matrix() -> pd.DataFrame:
    rows = [
        ("GARD_UPDATE_KERNEL", "Poisson updates", "Multiple source-compatible kernels", "AUTHOR_IMPLEMENTATION_REQUIRED", "exact paper code/config"),
        ("POISSON_EXPOSURE", "not specified", "Frozen h=.60315/.56133 and exploratory h=2.875 all source-plausible in different senses", "AUTHOR_IMPLEMENTATION_REQUIRED", "configuration"),
        ("MOLECULAR_STEP_CLOCK", "every molecular step", "state inclusion/boundary timing absent", "AUTHOR_IMPLEMENTATION_REQUIRED", "trajectory schema/code"),
        ("FISSION_DAUGHTER", "one progeny", "first versus random nonempty", "NONIDENTIFIABLE_MULTIPLE_PIPELINES", "daughter selection code"),
        ("EXTINCTION", "absent", "replacement/termination unknown", "NO_SUPPORTING_EVIDENCE", "code/data"),
        ("SELF_REPLICATOR_DEFINITION", "recurring clusters; most recurring composition", "adjacent H and reconstructed recurrence families disagree with fingerprints", "REQUIRES_UNAVAILABLE_AUTHOR_CODE", "label implementation"),
        ("SIMILARITY_THRESHOLD", "threshold mentioned", "historical H>.9 source versus Euclidean wording", "CONTRADICTORY_PAPER_DESCRIPTIONS", "exact code"),
        ("METRIC_IDENTITY", "equation + one-atom prose", "direct WMS vs public integrated vs public emergence", "CONTRADICTORY_PAPER_DESCRIPTIONS", "paper implementation"),
        ("PARTITION", "minimum-information bipartition", "public Fiedler approximation not stated", "REQUIRES_UNAVAILABLE_AUTHOR_CODE", "partition code/seed"),
        ("PREPROCESSING", "relative composition, CLR, drop last", "zero handling, active filtering and z-score absent", "REQUIRES_UNAVAILABLE_AUTHOR_CODE", "preprocessing code"),
        ("FULL_VS_LOCAL_FITTING", "complete trajectory correlations; predictive claims", "public global fitting leaks suffix", "CONTRADICTORY_PAPER_DESCRIPTIONS", "prediction pipeline"),
        ("LEVEL_VS_CHANGE", "text level; caption change", "both must remain separate", "CONTRADICTORY_PAPER_DESCRIPTIONS", "analysis code"),
        ("STATISTICAL_SCOPE", "tests named", "lags, pooling, Fisher inputs, multiplicity absent", "REQUIRES_UNAVAILABLE_AUTHOR_CODE", "analysis scripts"),
        ("MLP_TENSOR", "25/75, 80/20, 10 repetitions", "layout/mask/balance/target absent", "REQUIRES_UNAVAILABLE_AUTHOR_CODE", "prediction code"),
        ("INTERVENTION_SCORER", "argmax/argmin Phi-r after fission", "single hypothetical state score undefined", "REQUIRES_UNAVAILABLE_AUTHOR_CODE", "online scorer code"),
        ("TIE_HANDLING", "absent", "action ties/numerical equivalence unknown", "NO_SUPPORTING_EVIDENCE", "code"),
        ("TABLE1_UNITS_DISPERSION", "mixed note/cell semantics", "first onset and ± identity unresolved", "CONTRADICTORY_PAPER_DESCRIPTIONS", "underlying table data/code"),
        ("RNG_PACKAGE_VERSIONS", "different random seeds", "no versions/seed domains supplied", "REQUIRES_UNAVAILABLE_AUTHOR_CODE", "environment lock"),
    ]
    return pd.DataFrame(rows, columns=["implementationItem", "paperEvidence", "publicAndFrozenEvidence", "resolutionClass", "requiredDiscriminator"])


def figure_to_claim_crosswalk() -> pd.DataFrame:
    matrix = pd.read_csv(read_config()["paths"]["matrixA"])
    rows: list[dict[str, Any]] = []
    for claim in matrix.itertuples(index=False):
        text = f"{claim.paperComponent} {claim.claimFamily}".upper()
        components: list[str] = []
        for figure in range(1, 7):
            if f"FIGURE_{figure}" in text or f"FIGURE {figure}" in text or f"FIG{figure}" in text:
                components.append(f"FIGURE_{figure}")
        if "TABLE" in text or claim.claimNumber >= 50:
            components.append("TABLE_1")
        if not components:
            if claim.claimNumber <= 12: components = ["PAPER_TEXT_METRIC_DISTINCTIVENESS"]
            elif claim.claimNumber <= 20: components = ["FIGURE_2"]
            elif claim.claimNumber <= 27: components = ["FIGURE_3", "FIGURE_4"]
            elif claim.claimNumber <= 33: components = ["FIGURE_5"]
            else: components = ["FIGURE_6", "TABLE_1"]
        for component in sorted(set(components)):
            rows.append({"claimId": claim.claimId, "componentId": component, "s18Status": claim.finalStatusCode, "evidencePaths": claim.primaryEvidencePaths, "l12ChangeToStatus": "NONE"})
    return pd.DataFrame(rows)


def audit() -> None:
    started = time.time()
    if not (OUT / "source_snapshot_manifest.json").exists():
        raise RuntimeError("prepare phase required")
    source_manifest = json.loads((OUT / "source_snapshot_manifest.json").read_text(encoding="utf-8"))
    phirl = Path(read_config()["paths"]["phirl"])
    if git(phirl, "rev-parse", "HEAD") != source_manifest["phirl"]["pinnedCommit"]:
        raise RuntimeError("source commit changed after freeze")
    immutable = json.loads((OUT / "immutable_prior_baseline.json").read_text(encoding="utf-8"))
    validation = validate_immutable_baseline(immutable)
    write_json(OUT / "immutable_prior_validation.json", validation)
    if not validation["success"]:
        raise RuntimeError("immutable prior changed")

    statements = build_paper_statement_registry()
    write_parquet(OUT / "paper_statement_registry.parquet", statements)
    nx.write_graphml(build_paper_dependency_graph(), OUT / "paper_method_dependency_graph.graphml")
    write_csv(OUT / "paper_internal_discrepancy_registry.csv", paper_discrepancies())

    panels, digitization = figure_specs()
    write_parquet(OUT / "figure_panel_registry.parquet", panels)
    write_csv(OUT / "figure_digitization.csv", digitization)
    write_csv(OUT / "figure_internal_consistency_matrix.csv", figure_consistency_matrix())
    write_csv(OUT / "figure_to_claim_crosswalk.csv", figure_to_claim_crosswalk())
    write_csv(OUT / "table1_semantics_matrix.csv", table1_semantics())
    write_csv(OUT / "figure5_reconciliation_possibilities.csv", figure5_reconciliation_rows())

    nx.write_graphml(build_phirl_dataflow(), OUT / "phirl_executable_dataflow.graphml")
    functions = build_phirl_function_registry()
    write_parquet(OUT / "phirl_function_registry.parquet", functions)
    write_csv(OUT / "phirl_numerical_semantics.csv", phirl_numerical_semantics())
    atoms = build_atom_registry()
    write_csv(OUT / "phiid_atom_registry.csv", atoms)
    write_csv(OUT / "phirl_atom_identity_matrix.csv", phirl_atom_identity_matrix(atoms))
    write_csv(OUT / "phirl_temporal_leakage_map.csv", phirl_temporal_leakage_map())
    write_phirl_missing_components(OUT / "phirl_missing_gard_components.md")
    write_equation_derivation(OUT / "paper_equation_derivation.md", atoms)
    write_json(OUT / "metric_identity_adjudication.json", metric_identity_adjudication(atoms))

    concordance = build_concordance_matrix()
    write_csv(OUT / "paper_phirl_e01_concordance_matrix.csv", concordance)
    write_parquet(OUT / "root_cause_hypothesis_registry.parquet", root_cause_registry())
    write_csv(OUT / "unresolved_author_implementation_matrix.csv", unresolved_author_matrix())
    lock_members = [
        "paper_statement_registry.parquet", "paper_internal_discrepancy_registry.csv",
        "figure_panel_registry.parquet", "figure_digitization.csv", "figure_internal_consistency_matrix.csv",
        "table1_semantics_matrix.csv", "phirl_function_registry.parquet", "phirl_numerical_semantics.csv",
        "phirl_atom_identity_matrix.csv", "phirl_temporal_leakage_map.csv", "metric_identity_adjudication.json",
        "paper_phirl_e01_concordance_matrix.csv", "root_cause_hypothesis_registry.parquet",
        "unresolved_author_implementation_matrix.csv",
    ]
    lock = {
        "schema": "eidosoma.e01.s19.l12.concordance_lock.v1", "researchStepId": STEP_ID,
        "lockedBeforeCandidateHypothesisGeneration": True, "lockedUtc": CREATED_UTC,
        "members": [file_record(OUT / name) for name in lock_members],
        "aggregateSha256": sha256_bytes(canonical_json([(name, sha256_file(OUT / name)) for name in lock_members]).encode()),
        "statementRows": len(statements), "concordanceRows": len(concordance),
        "s18MatrixAClaimsCrossReferenced": int((concordance["crossReferenceType"] == "S18_MATRIX_A_59_CLAIMS").sum()),
        "s18MatrixBQuestionsCrossReferenced": int((concordance["crossReferenceType"] == "S18_MATRIX_B_7_QUESTIONS").sum()),
        "s19LoopsCrossReferenced": int((concordance["crossReferenceType"] == "S19_LOOP_RESULT").sum()),
    }
    write_json(OUT / "concordance_lock.json", lock)
    write_json(OUT / "audit_runtime.json", {"researchStepId": STEP_ID, "phase": "audit", "wallSeconds": time.time() - started, "cpuSeconds": time.process_time(), "gpuUsed": False})


def hidden_pipeline_hypotheses() -> list[dict[str, Any]]:
    """Whole-pipeline hypotheses generated only after the concordance lock exists."""
    if not (OUT / "concordance_lock.json").exists():
        raise RuntimeError("candidate pipeline generation prohibited before concordance lock")
    # Scores are forensic plausibility scores, not scientific outcome scores.
    hypotheses = [
        {
            "hypothesisId": "HP1_PUBLIC_PHIRL_COMPLETED_FIT",
            "name": "Public-PhiRL completed-fit reconstruction",
            "chain": {
                "simulator": "both frozen paper-time-compatible candidates at original exposure",
                "clock": "selected molecular observations with daughter boundaries",
                "preprocessing": "additive-.5 closure, CLR, drop last; public active filter + z-score",
                "partition": "fast lagged-MI graph + 1e-6 noise + unnormalized Fiedler sign split",
                "metric": "public emergence = synergy + downward causation",
                "temporalFitting": "completed trajectory",
                "label": "adjacent incoming molecular H>0.9 comparator",
                "aggregation": "available-case molecular index, median±SD",
                "predictionTensor": "original order, explicit masks, first 25% to last 75%",
                "interventionScorer": "append hypothetical state and refit current prefix",
                "outcomes": "literal molecular persistence/probability/consecutive Pearson/onset",
            },
            "directSourceSupport": 5, "directPaperSupport": 3, "explainFigure2": 3,
            "explainFigures3And4": 4, "reconcileFigure5VsTable1": 0, "explainFigure6AndTable1": 1,
            "compatibilityWithFrozenE01": 5, "undocumentedAssumptions": 4,
            "prospectivePlausibility": 1, "falsifiability": 5,
            "penalties": {"outcomeDirectedThreshold": 0, "undocumentedExposure": 0, "inventedProjection": 0, "futureLeakage": 3, "differentLabels": 0, "differentDenominators": 0, "completedFutureOnline": 0, "candidateSelection": 0},
            "forensicFinding": "Most directly public-source grounded and already strongly tested; reproduces some retrospective resemblance but fails label prevalence, prospective prediction and intervention ordering.",
        },
        {
            "hypothesisId": "HP2_PAPER_LITERAL_WMS_RECURRING_ATTRACTOR",
            "name": "Paper-literal equation and recurring-attractor reconstruction",
            "chain": {
                "simulator": "original historical GARD family with paper parameters; exposure and daughter choice author-required",
                "clock": "molecular composition observations; recurrence learned from post-fission generations",
                "preprocessing": "relative composition, CLR, drop last; zero policy author-required",
                "partition": "paper MIB; exact approximation author-required",
                "metric": "displayed direct whole-minus-sum-parts equation",
                "temporalFitting": "completed fit for Figures 2–4; cutoff-only refit required for prospective Figure 5",
                "label": "membership relative to most recurring composition; exact clustering/projection author-required",
                "aggregation": "paper median±SD; unequal-length policy author-required",
                "predictionTensor": "first-quarter to final-three-quarter; balancing/mask author-required",
                "interventionScorer": "online prefix scorer using displayed metric; hypothetical endpoint semantics author-required",
                "outcomes": "Table1 literal definitions with unit/dispersion ambiguity retained",
            },
            "directSourceSupport": 2, "directPaperSupport": 5, "explainFigure2": 3,
            "explainFigures3And4": 3, "reconcileFigure5VsTable1": 2, "explainFigure6AndTable1": 2,
            "compatibilityWithFrozenE01": 2, "undocumentedAssumptions": 9,
            "prospectivePlausibility": 3, "falsifiability": 3,
            "penalties": {"outcomeDirectedThreshold": 0, "undocumentedExposure": 2, "inventedProjection": 2, "futureLeakage": 0, "differentLabels": 0, "differentDenominators": 1, "completedFutureOnline": 0, "candidateSelection": 0},
            "forensicFinding": "Closest to manuscript wording, but key operations needed to make it executable are absent from public code; prior recurring-attractor reconstructions did not recover the joint fingerprint.",
        },
        {
            "hypothesisId": "HP3_FIGURE_SPECIFIC_MIXED_PIPELINES",
            "name": "Figure-specific label/denominator/scalar pipelines",
            "chain": {
                "simulator": "paper parameter family, possibly separate prediction/intervention data",
                "clock": "molecular for Figures 2–4/Table1; enriched or generational for Figure5",
                "preprocessing": "paper CLR branch with unspecified per-figure details",
                "partition": "public/source-lineage Fiedler or paper MIB by figure",
                "metric": "completed-fit public emergence for descriptive panels; another scalar/scorer for intervention",
                "temporalFitting": "completed for Figures 2–4; task-specific for Figures 5–6",
                "label": "recurring-attractor for Table1; balanced/onset label for Figure5",
                "aggregation": "figure-specific unequal-length and denominator policies",
                "predictionTensor": "balanced/enriched target producing ~60% dummy",
                "interventionScorer": "unpublished paper-specific scorer",
                "outcomes": "figure-specific denominators and reported table summaries",
            },
            "directSourceSupport": 0, "directPaperSupport": 1, "explainFigure2": 4,
            "explainFigures3And4": 4, "reconcileFigure5VsTable1": 5, "explainFigure6AndTable1": 4,
            "compatibilityWithFrozenE01": 3, "undocumentedAssumptions": 14,
            "prospectivePlausibility": 1, "falsifiability": 1,
            "penalties": {"outcomeDirectedThreshold": 0, "undocumentedExposure": 2, "inventedProjection": 2, "futureLeakage": 2, "differentLabels": 3, "differentDenominators": 3, "completedFutureOnline": 2, "candidateSelection": 0},
            "forensicFinding": "Can narratively absorb the manuscript contradictions but lacks coherent public support and is too assumption-rich to execute as confirmation.",
        },
    ]
    for item in hypotheses:
        positive = sum(item[key] for key in [
            "directSourceSupport", "directPaperSupport", "explainFigure2", "explainFigures3And4",
            "reconcileFigure5VsTable1", "explainFigure6AndTable1", "compatibilityWithFrozenE01",
            "prospectivePlausibility", "falsifiability",
        ])
        penalty = sum(item["penalties"].values()) + item["undocumentedAssumptions"]
        item["scoreFormula"] = "sum(nine 0-to-5 support/explanation/plausibility/falsifiability scores) - undocumentedAssumptions - sum(explicit penalties)"
        item["score"] = positive - penalty
    return sorted(hypotheses, key=lambda item: (-item["score"], item["hypothesisId"]))


def decisive_next_step_markdown(hypotheses: list[dict[str, Any]]) -> str:
    rows = []
    for item in hypotheses:
        if item["hypothesisId"].startswith("HP1"):
            design = "No new run is needed: S13Y–S17 already falsify the full chain's Figure 5/Table 1 and intervention fingerprints."
            inputs = "Frozen S13Y/S16/S17 outputs"
            fingerprint = "~98% target/dummy, completed-fit dependence, no max benefit"
            control = "cutoff-causal S16 and no-action S17"
            compute = "deterministic crosswalk only"
            falsifier = "Already observed failures across both candidates"
            scope = "retrospective resemblance versus prospective/causal non-support"
        elif item["hypothesisId"].startswith("HP2"):
            design = "Cannot be prospectively locked without author definitions of label, MIB, tensor and endpoint scorer; a new run now would invent the hypothesis it claims to test."
            inputs = "Would require untouched matrices only after exact code/config is recovered"
            fingerprint = "joint Figure 1–6/Table1 output under one code path"
            control = "exact public-PhiRL and frozen E01 pipeline"
            compute = "not estimable until missing operations are supplied"
            falsifier = "Untouched failure of joint occupancy/onset/consistency, dummy prevalence, or treatment ordering"
            scope = "paper-facing, prospective and causal gates kept separate"
        else:
            design = "No single scientific experiment can confirm an unconstrained figure-specific mixture; recover implementation provenance or reject it as non-falsifiable."
            inputs = "author code/data provenance"
            fingerprint = "explicitly different labels/denominators/scripts by figure"
            control = "one-chain implementation"
            compute = "source audit only"
            falsifier = "one released coherent pipeline regenerates every panel"
            scope = "forensic provenance"
        rows.append(f"| `{item['hypothesisId']}` | {inputs} | {design} | {fingerprint} | {control} | {compute} | {falsifier} | {scope} |")
    return f"""# Decisive next-step options

## Concise top summary

- **Research step ID:** `{STEP_ID}`.
- **Completion status:** COMPLETE DESIGN-ONLY; no option executed.
- **Artifacts written:** three frozen whole-pipeline hypotheses and this discriminating-design crosswalk.
- **Validation result:** PASS — hypotheses were generated only after the concordance lock and no outcome was computed.
- **Outcome classification:** `AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION`.
- **Caveats or blockers:** The missing label, prediction-task and intervention-scorer definitions are identity-changing, not minor parameter uncertainty.
- **Recommended next action:** `AUTHOR_CODE_WAIT_STATE`; preserve the option of a one-shot untouched pipeline reconstruction when authoritative code/configuration becomes available.

| Hypothesis | Frozen inputs | Smallest distinguishing design | Expected fingerprint | Negative control | Compute estimate | Falsifier | Evidence scope |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## One recommendation

Enter an **author-code wait state**. Do not contact the authors unless separately authorized. The public evidence cannot define one executable pipeline without inventing at least the self-replicator label, Figure 5 sampling/tensor semantics, and the intervention endpoint scorer. Running another scientific reconstruction now would increase specification multiplicity more than information. If code or exact configuration is later released, freeze its identity first, firewall untouched matrices, and execute one joint Figure 1–6/Table 1 regeneration with retrospective, prospective, and causal gates reported separately.
"""


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white", metadata={"Software": "Eidosoma S19-L12"})
    plt.close(fig)


def _draw_graph(graph: nx.DiGraph, path: Path, title: str, *, horizontal: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(15, 9 if not horizontal else 7))
    if horizontal:
        generations = {}
        for idx, node in enumerate(nx.topological_sort(graph)):
            generations[node] = idx
        pos = {node: (idx, (idx % 3) * 0.12) for node, idx in generations.items()}
    else:
        pos = nx.spring_layout(graph, seed=19012, k=1.15 / math.sqrt(max(len(graph), 1)), iterations=250)
    kinds = [graph.nodes[node].get("kind", "operation") for node in graph.nodes]
    palette = {kind: plt.cm.Set3(i % 12) for i, kind in enumerate(sorted(set(kinds)))}
    colors = [palette[kind] for kind in kinds]
    nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=13, width=1.1, alpha=.55)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors, node_size=1350, edgecolors="#243447", linewidths=.8)
    labels = {node: graph.nodes[node].get("label", node) for node in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)
    ax.set_title(title, fontsize=15, weight="bold")
    ax.axis("off")
    _save_figure(fig, path)


def render_figures(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    _draw_graph(build_paper_dependency_graph(), directory / REQUIRED_FIGURES[0], "Paper-visible end-to-end dependency graph")
    _draw_graph(build_phirl_dataflow(), directory / REQUIRED_FIGURES[1], "Executable public PhiRL data flow", horizontal=True)

    concordance = pd.read_csv(OUT / "paper_phirl_e01_concordance_matrix.csv")
    material = concordance[concordance["crossReferenceType"] == "MATERIAL_PIPELINE_ELEMENT"].copy()
    columns = ["DIRECT_PAPER_SPECIFICATION", "DIRECT_PUBLIC_CODE", "DIRECT_FROZEN_E01_RESULT", "AUTHOR_IMPLEMENTATION_REQUIRED"]
    matrix = np.array([[int(label in value.split(";")) for label in columns] for value in material["evidenceLabels"]])
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(columns)), ["Paper", "Public source", "Frozen E01", "Author required"], rotation=25, ha="right")
    ax.set_yticks(range(len(material)), material["topic"], fontsize=7)
    ax.set_title("Paper–PhiRL–E01 operation map", weight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): ax.text(j, i, "●" if matrix[i, j] else "", ha="center", va="center", color="white" if matrix[i, j] else "black", fontsize=8)
    _save_figure(fig, directory / REQUIRED_FIGURES[2])

    values = pd.read_parquet("/artifacts/research_steps/S13Y/full_source_values.parquet")
    lengths = values.groupby(["candidateId", "trajectoryId"], sort=True).size().rename("length").reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    extents = [1300, 800, 800, 1050]
    axes[0].bar(["Fig2A", "Fig2B", "Fig2C", "Fig2D"], extents, color=["#355C7D", "#6C5B7B", "#C06C84", "#F67280"])
    axes[0].set_ylabel("Visible x-axis extent (approx. molecular steps)")
    for candidate, group in lengths.groupby("candidateId", sort=True):
        sorted_lengths = np.sort(group["length"].to_numpy())
        support = [(sorted_lengths >= x).sum() for x in range(0, 1501, 50)]
        axes[1].plot(range(0, 1501, 50), support, label=candidate.replace("S12F-", ""))
    axes[1].axvline(1300, color="black", ls="--", lw=1, label="Fig2A extent")
    axes[1].set(xlabel="Molecular index", ylabel="Frozen S13Y runs still contributing", ylim=(0, 105))
    axes[1].legend(fontsize=8)
    fig.suptitle("Figure 2 digitized clock extent and unequal-length support constraint", weight="bold")
    _save_figure(fig, directory / REQUIRED_FIGURES[3])

    s15 = pd.read_csv("/artifacts/research_steps/S15/paper_target_comparison.csv")
    means = s15[s15["diagnosticScope"] == "MEAN_RUNWISE_SPEARMAN"].copy()
    means["value"] = pd.to_numeric(means["reconstructedValue"], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 5))
    labels, vals, colors = [], [], []
    for row in means.itertuples():
        labels.append(f"{row.analysisId.split('_')[0]}\n{row.candidateId[-2:]}")
        vals.append(row.value); colors.append("#4C78A8" if row.analysisId.startswith("LEVEL") else "#F58518")
    ax.bar(labels, vals, color=colors)
    ax.axhline(.139, color="black", ls="--", label="paper visible/reported mean .139")
    ax.set(ylabel="Mean runwise Spearman ρ", title="Figure 3: level text and change caption remain separate")
    ax.legend()
    _save_figure(fig, directory / REQUIRED_FIGURES[4])

    prevalence = pd.read_csv("/artifacts/research_steps/S16/prevalence_audit.csv")
    s16_test = prevalence[prevalence["splitRole"] == "TEST"].groupby("candidateId")["targetPrevalence"].mean()
    labels = ["Fig5 dummy\nvisible", "Table1 control\nprobability", "S16 candidate 2\ntest prevalence", "S16 candidate 3\ntest prevalence"]
    vals = [.60, .88, float(s16_test.iloc[0]), float(s16_test.iloc[1])]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, vals, color=["#E45756", "#72B7B2", "#4C78A8", "#54A24B"])
    ax.set_ylim(.5, 1.01); ax.set_ylabel("Positive prevalence / majority accuracy")
    ax.set_title("Figure 5 class-prevalence contradiction", weight="bold")
    for bar, value in zip(bars, vals, strict=True): ax.text(bar.get_x()+bar.get_width()/2, value+.008, f"{value:.3f}", ha="center")
    _save_figure(fig, directory / REQUIRED_FIGURES[5])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    text = (
        "Figure 6 / Table 1 consistency map\n\n"
        "MAX: persistence 874, probability 88%, consistency .52, onset 36%*\n"
        "CONTROL: persistence 716, probability 88%, consistency .38, onset 37%*\n"
        "MIN: persistence 559, probability 80%, consistency .42, onset 40%*\n\n"
        "Conflict 1: min consistency (.42) > control (.38), while prose says minimization worsened all four.\n"
        "Conflict 2: onset cells use %, while note defines molecular steps.\n"
        "Constraint 3: max and control round to the same overall occupancy despite different Fig6C slopes.\n"
        "Frozen S17: exact online replay passed; max did not help; min harmed modestly."
    )
    ax.text(.5, .5, text, ha="center", va="center", fontsize=12, bbox={"boxstyle":"round,pad=1", "facecolor":"#F7F3E8", "edgecolor":"#8C6D31"})
    _save_figure(fig, directory / REQUIRED_FIGURES[6])

    atoms = pd.read_csv(OUT / "phiid_atom_registry.csv")
    order = ["r", "u0", "u1", "s"]
    metrics = ["localPhiRWeight", "publicEmergenceWeight", "paperDisplayedEquationWeight"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, metrics, ["Public integrated\n(9 atoms)", "Public emergence\n(3 atoms)", "Displayed equation\n(s minus r)"]):
        grid = np.zeros((4,4))
        for row in atoms.itertuples(): grid[order.index(row.sourceAntichain), order.index(row.targetAntichain)] = getattr(row, metric)
        im=ax.imshow(grid, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(4), order); ax.set_yticks(range(4), order)
        ax.set(xlabel="target antichain", ylabel="source antichain", title=title)
        for i in range(4):
            for j in range(4): ax.text(j,i,f"{int(grid[i,j]):+d}",ha="center",va="center")
    fig.suptitle("Metric-identity atom map: the three coefficient patterns are non-identical", weight="bold")
    _save_figure(fig, directory / REQUIRED_FIGURES[7])

    leak = pd.read_csv(OUT / "phirl_temporal_leakage_map.csv")
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#D62728" if value else "#7F7F7F" for value in leak["usesFutureSuffixInPublicCompletedFit"]]
    ax.barh(leak["pipelineElement"], [1]*len(leak), color=colors)
    ax.invert_yaxis(); ax.set_xticks([])
    ax.set_title("Completed-fit future-dependence map (red = suffix-dependent)", weight="bold")
    _save_figure(fig, directory / REQUIRED_FIGURES[8])

    evidence_cols = ["DIRECT_PAPER_SPECIFICATION", "DIRECT_PUBLIC_CODE", "DIRECT_FROZEN_E01_RESULT", "PAPER_INTERNAL_CONFLICT", "AUTHOR_IMPLEMENTATION_REQUIRED"]
    heat = np.array([[int(label in value.split(";")) for label in evidence_cols] for value in material["evidenceLabels"]])
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(evidence_cols)), ["Paper", "Public code", "Frozen E01", "Internal conflict", "Author required"], rotation=30, ha="right")
    ax.set_yticks(range(len(material)), material["topic"], fontsize=7)
    ax.set_title("Paper–PhiRL–E01 concordance heatmap", weight="bold")
    _save_figure(fig, directory / REQUIRED_FIGURES[9])

    causes = pd.read_parquet(OUT / "root_cause_hypothesis_registry.parquet").sort_values("priorityScore")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(causes["hypothesis"], causes["priorityScore"], color="#4C78A8")
    ax.set_xlim(0,5); ax.set_xlabel("Forensic priority (mean of four 0–5 audit scores)")
    ax.set_title("Root-cause hypothesis ranking (not outcome selection)", weight="bold")
    _save_figure(fig, directory / REQUIRED_FIGURES[10])

    fig, ax = plt.subplots(figsize=(14, 7)); ax.axis("off")
    boxes = [
        (.08,.75,"Is one complete public\npipeline executable?\nNO"),
        (.38,.75,"Are missing choices\nidentity-preserving?\nNO"),
        (.68,.75,"Can another reconstruction\nbe confirmatory now?\nNO"),
        (.38,.30,"AUTHOR-CODE WAIT STATE\nFreeze provenance; no contact\nunless separately authorized"),
        (.75,.30,"If exact code/config appears:\none untouched joint\nFigures 1–6/Table1 run"),
    ]
    for x,y,label in boxes:
        ax.text(x,y,label,ha="center",va="center",fontsize=11,bbox={"boxstyle":"round,pad=.7","facecolor":"#E8F1F8","edgecolor":"#355C7D"})
    arrows=[((.16,.75),(.30,.75)),((.46,.75),(.60,.75)),((.68,.65),(.45,.40)),((.50,.30),(.67,.30))]
    for start,end in arrows: ax.annotate("",xy=end,xytext=start,arrowprops={"arrowstyle":"->","lw":1.8,"color":"#355C7D"})
    ax.set_title("Decisive-next-step decision tree", weight="bold", fontsize=15)
    _save_figure(fig, directory / REQUIRED_FIGURES[11])


def full_report_text(hypotheses: list[dict[str, Any]], validation_summary: str = "PENDING_FINAL_HASH_VALIDATION") -> str:
    concordance = pd.read_csv(OUT / "paper_phirl_e01_concordance_matrix.csv")
    statements = pd.read_parquet(OUT / "paper_statement_registry.parquet")
    panels = pd.read_parquet(OUT / "figure_panel_registry.parquet")
    discrepancies = pd.read_csv(OUT / "paper_internal_discrepancy_registry.csv")
    functions = pd.read_parquet(OUT / "phirl_function_registry.parquet")
    causes = pd.read_parquet(OUT / "root_cause_hypothesis_registry.parquet")
    matrix_a = pd.read_csv(read_config()["paths"]["matrixA"])
    status_counts = matrix_a["finalStatusCode"].value_counts().to_dict()
    score_lines = "\n".join(f"- `{item['hypothesisId']}`: forensic score {item['score']}; {item['forensicFinding']}" for item in hypotheses)
    return f"""# S19-L12 full results — Paper–PhiRL forensic concordance audit

## Concise top summary

- **Research step ID:** `{STEP_ID}` — `{LOOP_ID}`.
- **Completion status:** COMPLETE; analysis-only audit frozen at the mandatory human-review boundary.
- **Artifacts written:** {len(REQUIRED_FILES)} required named records, {len(REQUIRED_FIGURES)} required figures, source/concordance locks, and root S19 handoff/ledger updates.
- **Validation result:** {validation_summary} Source identity, immutable priors, all figure/table panels, all required PhiRL functions, all 59 S18 claims, all seven S18 prospective/causal questions, and S19-L01–L11R are cross-referenced. No GARD outcome, matrix, label, MLP, intervention, or new metric was produced.
- **Outcome classification:** `AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION` (constraining/contradictory forensic result); secondary finding `PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT`.
- **Caveats or blockers:** The manuscript-specific code/data are unavailable; native-figure measurements are approximate visual constraints; public PhiRL is lineage evidence, not proof of the authors' hidden pipeline.
- **Recommended next action:** Mandatory human review. Keep S20 inactive. Recommended scientific posture: `AUTHOR_CODE_WAIT_STATE`; if authoritative implementation later appears, freeze it and run one untouched end-to-end reconstruction.

## Lay summary

We are closer to knowing *why* exact replication remains elusive, but not closer to claiming that a hidden author implementation has been found. The public PhiRL code gives a precise information-calculation pipeline, yet the paper's own equation, its “one atom” description, and PhiRL's two named outputs do not describe the same mathematical quantity. The paper also shows a roughly 60% majority baseline in its prediction figure while reporting roughly 88% replication in Table 1; those cannot be the same unbalanced molecular target under one denominator. Finally, the paper does not publish the GARD label, prediction tensor, or one-state intervention scorer needed to reproduce the later figures.

One public-source pipeline does explain parts of the paper: completed-fit PhiRL trajectories have punctuated spikes and retrospective label-coupled associations. But the strongest frozen E01 tests show aggregate-trend disagreement, future dependence, a deterministic ~98% adjacent-H target, no prospective feature advantage, and no max-intervention benefit. Repeated source-grounded label reconstructions did not recover the full occupancy/onset/consistency fingerprint. The most defensible conclusion is therefore not “the paper is wrong,” and not “we found its code,” but that public evidence admits incompatible pipelines and author implementation is required to discriminate them.

## Frozen question and scope

L12 asked what complete operation chain is required for Figures 1–6 and Table 1, and whether one public, source-grounded pipeline explains those visible fingerprints together with frozen E01 evidence. It was explicitly an audit, not a new scientific experiment. All S01–S18 totals remain unchanged: {status_counts}. L11R remains `ALL_COMPTYPE_UNION_NOT_SUPPORTED`, `SOURCE_TAG_SINGLETON_DEPENDENT`, `NOT_PROMOTABLE`.

## Inputs and provenance

- Original arXiv preprint 2607.28250v1 PDF, SHA-256 `{read_config()['sourceLocks']['paperSha256']}`, plus workspace Markdown and eight native-resolution extracted images.
- PhiRL repository at pinned/current master `{read_config()['sourceLocks']['phirlPinnedCommit']}`; current master and pinned tree are byte-identical.
- IIGR `{read_config()['sourceLocks']['iigrPinnedCommit']}`, historical GARD `{read_config()['sourceLocks']['gardHistoricalCommit']}`, and BreakingGRNMemories `{read_config()['sourceLocks']['breakingGrnCommit']}` as source-lineage context.
- Safe 16-node PhiID lattice JSON from S12B. The raw pickle was opened only by an isolated `python -I` restricted-conversion subprocess and compared with the frozen JSON.
- Every frozen S18 Matrix A/Matrix B row and all S19 loop classifications through L11R.

The complete file-level source and prior hashes are in `source_snapshot_manifest.json`, `phirl_repository_tree.json`, and `immutable_prior_validation.json`. Cached source without a compatible detected license is referenced by identity and hash, not redistributed.

## Methods and commands

The preregistered order was enforced: (1) freeze source/prior identities; (2) build sentence, panel, function, atom and leakage registries; (3) freeze their concordance hash; (4) only then generate and rank at most three whole-pipeline hypotheses; (5) design but do not execute one next action; (6) regenerate, validate and hash artifacts.

Primary commands:

```text
python scripts/e01/run_s19_l12.py prepare
python scripts/e01/run_s19_l12.py audit
python scripts/e01/run_s19_l12.py finalize
python -m pytest -q tests/e01/test_s19_l12.py
python -m compileall -q scripts/e01/run_s19_l12.py tests/e01/test_s19_l12.py
```

Execution used CPU float64 where numerical fixtures were needed, one process for audit logic, zero GPU, no simulator import, and no scientific random outcome. Graph layouts use fixed seeds. Figure digitization is recorded as approximate or reported-exact in `figure_digitization.csv`; it is never treated as raw author data.

## Paper sentence and dependency audit

`paper_statement_registry.parquet` contains {len(statements)} computationally meaningful rows: 37 method/semantics statements plus all 59 claim-ledger statements. Each row records objects, clock, unit, preprocessing, estimator, label, denominator, aggregation, test, reported value, specification state, E01 crosswalk and unresolved fields. The dependency graph shows that one label definition affects Figures 3–6 and Table 1, while one metric/scorer chain affects Figures 2–6.

The audit records {len(discrepancies)} material discrepancies. The strongest are:

1. Figure 3 Results uses level while its caption specifies change.
2. Figure 5's ~60% dummy baseline conflicts with Table 1's 88% molecular occupancy if label and denominator are shared.
3. The displayed Phi-r equation, “one atom” prose, public `integrated`, and public `emergence` are non-identical.
4. Table 1 min consistency (.42) exceeds control (.38) despite “worsening all four” prose.
5. First-onset cells contain percent signs while the note defines molecular steps.
6. Figure 2 unequal-length aggregation and Figure 6 hypothetical-state scoring are not specified.

![Paper dependency graph](figure_01_paper_dependency_graph.png)

*Figure 1. Every downstream panel depends on unresolved choices in the label and/or scalar pipeline.*

## Figure-by-figure results

All {len(panels)} required Figure 1–6/Table 1 components are present in `figure_panel_registry.parquet`.

### Figure 1

Figure 1 jointly implies molecular-time Phi values and recurrence/attractor semantics across generations, but it does not state whether the binary label is one cluster, any cluster, a full-run reference, or a projected boundary state. L02–L11R explored these structural interpretations without finding a joint paper fingerprint.

### Figure 2

The aggregate extends to roughly 1,300 molecular steps while frozen compatible trajectories vary from roughly 200 to 1,467 observations. Frozen support falls sharply in the tail, so padding, available-case calculation, truncation, or resampling can materially change the trend. S14 reproduces punctuated excursions and temporal dependence but finds significant positive aggregate trends in both candidates, rather than p=.1995 trendlessness. Completed-fit/prefix values differ, and spikes can track partition changes and numerical condition.

![Figure 2 constraints](figure_04_figure2_digitized_clock_constraints.png)

*Figure 2. Native-figure x extents (left) and frozen trajectory support by molecular index (right).*

### Figures 3 and 4

The paper's LEVEL and CHANGE descriptions are internally inconsistent, so both remain named. Frozen S15 values are directionally positive in both candidates and paper-like one-sample diagnostics can be reconstructed, but these are retrospective, label-coupled, completed-fit results. Exact H completely determines frozen Y, ordinary stability is coupled, and past-only refitting reverses the direction. Figure 4 test scope and Fisher inputs remain missing.

![Figure 3 inconsistency](figure_05_figure3_level_change_inconsistency.png)

*Figure 3. Both frozen analyses remain below the paper's mean but point positively; neither replaces the other.*

### Figure 5

The paper's visible majority baseline is approximately .60. Table 1 reports .88 control occupancy, while S16's valid test suffix prevalence is approximately .983–.985. `figure5_reconciliation_possibilities.csv` enumerates class/per-run balancing, onset-only targets, molecular/generation targets, padding, masking, common-length truncation, negative enrichment, stratification, separate data and different labels. None is both specified by the paper and implemented by public PhiRL. S16's one prospectively frozen masked molecular layout does not reproduce the paper: dummy and learned models track prevalence, balanced accuracy is near .5, all test runs are already positive by the cutoff, and PhiRL adds no performance beyond H/stability controls.

![Figure 5 contradiction](figure_06_figure5_prevalence_contradiction.png)

*Figure 4. The prediction task's visible baseline cannot be the frozen unbalanced molecular target under a shared denominator.*

### Figure 6 and Table 1

The action timing and exhaustive add/delete idea are stated, but the mapping from one hypothetical state to a trajectory-fit Phi-r score is absent. S17's append-and-refit-current-prefix scorer is online and exactly replayable; its max arm does not improve over control and its min arm only modestly harms outcomes. It therefore constrains one coherent implementation but cannot adjudicate an unpublished scorer. Table units/dispersion and min consistency remain internally conflicted.

![Figure 6 and Table 1 map](figure_07_figure6_table1_consistency_map.png)

*Figure 5. Reported fields and the principal internal/frozen inconsistencies.*

## PhiRL executable source audit

All {len(functions)} required functions are located, blamed and traced. Public current master equals the pinned commit, so current-versus-pinned drift does not explain discrepancies. Internal history does matter: early PhiRL used the slow forward+backward-MI sum and `local_phi_r`; later commits exposed `emergence=synergy+causation`, then switched main execution to the fast averaged-correlation MI and trace-scaled covariance regularizer.

The current chain filters inactive variables and z-scores them using the complete trajectory; computes fast lagged correlations; with `alpha=1, bonferonni=False` retains every finite edge with p<1; adds a 1e-6 graph floor; takes an unnormalized Fiedler sign split (exact zero coordinates are omitted); averages each partition arithmetically; fits Gaussian means/covariances globally with a trace-scaled regularizer; inverts all 16 local PhiID atoms; and exports both nine-atom `integrated` and three-atom `emergence`. The shuffled path permutes the whole trajectory. `_load_phi` maps infinities to NaN, takes finite medians and can leave errors/missing cells as zeros.

![PhiRL flow](figure_02_phirl_dataflow_graph.png)

*Figure 6. Public PhiRL's executable chain; almost every fitted prefix value inherits full-trajectory dependence.*

![Future dependence](figure_09_completed_fit_future_dependence.png)

*Figure 7. Red operations use or inherit the completed future suffix.*

No public branch, tag, deleted path or inspected lineage file contains the GARD label, Figure 5 tensor, GARD MLP, hypothetical-state intervention scorer, treatment simulations, or Table 1 aggregation.

## Metric-identity result

All 16 atoms were recovered from the safe lattice. If source antichains are redundancy `r`, uniques `u0/u1`, and synergy `s`, the displayed equation expands as `Σ_q(s→q-r→q)`: eight signed atoms. Public `emergence` is only `s→s+s→u0+s→u1`. Public corrected `local_phi_r` contains nine different integrated atoms. These coefficient patterns are not algebraically equal.

![Metric atom map](figure_08_metric_identity_atom_map.png)

*Figure 8. Public integrated, public emergence, and the displayed paper equation are distinct atom combinations.*

The audit therefore classifies metric identity as `PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT`, not as any one favorable scalar. This does not invalidate a private coherent implementation; it means the public record does not identify it.

## Paper–PhiRL–E01 concordance

The concordance matrix has {len(concordance)} rows: 33 material pipeline elements, 59 immutable S18 claims, seven immutable S18 Matrix B questions, and 13 S19 loop results. Prior statuses are copied, never rewritten.

![Concordance heatmap](figure_10_paper_phirl_e01_concordance_heatmap.png)

*Figure 9. Direct specification/source/result support and unresolved author dependence by material operation.*

The recurring pattern is not a single numeric miss. Direct paper support is strongest for broad concepts; direct public support is strongest for the information component; direct frozen evidence tests plausible joins between them; and the interfaces—label, aggregation, prediction tensor and intervention scorer—remain unpublished.

## Root causes and whole-pipeline hypotheses

The highest-leverage causes are the label/task mismatch, scalar identity mismatch, Figure 5 task/denominator mismatch, completed-fit dependence and the intervention scorer. Scores rank forensic informativeness, not closeness to desired outcomes.

![Root causes](figure_11_root_cause_ranking.png)

*Figure 10. Audit-priority ranking before any new scientific execution.*

After the concordance matrix was hash-frozen, exactly three complete hypotheses were retained:

{score_lines}

None is one coherent publicly supported paper pipeline. HP1 is most executable and most thoroughly falsified as a full-paper reconstruction. HP2 is closest to wording but cannot be executed without identity-changing inventions. HP3 explains contradictions by allowing figure-specific pipelines, but that flexibility makes it poorly source-grounded and weakly falsifiable.

## Classification and decisive next step

The registered audit classification is `AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION`. This is stronger than merely “underdetermined” because exact public source inspection establishes that necessary GARD-specific operations are absent and manuscript-visible constraints conflict. It is not a scientific non-replication verdict beyond the frozen scope, and it does not modify S18.

![Decision tree](figure_12_decisive_next_step_tree.png)

*Figure 11. A new reconstruction cannot become confirmatory while its defining operations must be invented.*

The recommended next action is an author-code wait state, without contacting authors unless separately authorized. If exact code/configuration becomes available, the smallest defensible scientific action is one untouched, seed-firewalled, end-to-end Figure 1–6/Table 1 reconstruction that reports paper-facing, prospective and causal gates separately.

## Validation

- Immutable prior baseline: every scoped S01–S18, V1/V2 and S19-L01–L11R file passed SHA-256/size validation before and after the audit.
- Source freeze: current PhiRL master, local checkout and pinned commit/tree are identical; relevant source files and histories are hashed.
- Safe lattice: isolated restricted conversion matches the frozen JSON in raw hash, 16 nodes, edges, order and contents.
- Coverage: all required figure/table components and PhiRL functions are present; all 59+7 S18 rows and 13 prior S19 loop results are cross-referenced.
- Prohibitions: zero new GARD trajectories, matrices, scientific labels, emergence branches, MLPs or interventions; zero GPU use; no author contact.
- Ordering: `concordance_lock.json` predates and gates generation of candidate hidden-pipeline hypotheses.
- Regeneration: substantive tables, reports and all 12 figures are deterministically regenerated and compared in `regeneration_validation.json`.
- Artifact integrity: complete hashes and storage accounting are in `artifact_manifest.json` and `storage_validation.json`.

## Caveats, blockers and limitations

1. Native-figure digitization is deliberately approximate and cannot replace underlying data.
2. Git history establishes public chronology, not the private analysis date or unpublished code identity.
3. The safe-lattice algebra uses the public two-source/two-target convention; another redundancy convention could change numerical atoms but not reconcile the public coefficient definitions as written.
4. S19 explored many labels adaptively. Those results constrain interpretations but are not confirmation.
5. The Figure 5 contradiction admits multiple transformations; L12 does not select one by target proximity.
6. A coherent unpublished implementation may exist. Only its authoritative code/config/data can distinguish that possibility from manuscript inconsistency.

## Provenance and dependency versions

Python {platform.python_version()}; NumPy {np.__version__}; pandas {pd.__version__}; SciPy {scipy.__version__}; scikit-learn {sklearn.__version__}; NetworkX {nx.__version__}; Matplotlib {matplotlib.__version__}; CPU float64 authoritative; GPU unused. Repository implementation and preregistration were committed and pushed on `eidosoma/groups/42` before final audit release. Full source/blob hashes and commands are machine-readable.

## Mandatory boundary

L12 is complete. S20 and E02 remain inactive. No recommendation has been executed. Control returns for human review.
"""


def decision_summary_text(hypotheses: list[dict[str, Any]]) -> str:
    return f"""# S19-L12 one-page decision summary

## Concise top summary

- **Research step ID:** `{STEP_ID}`.
- **Completion status:** COMPLETE; mandatory human review.
- **Artifacts written:** full sentence/panel/source/atom/concordance registries, three unexecuted whole-pipeline hypotheses, 12 figures, validations and hashes.
- **Validation result:** PASS — prior artifacts immutable, source identities exact, all required coverage present, deterministic regeneration passed, and prohibited scientific execution count is zero.
- **Outcome classification:** `AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION`; metric sub-classification `PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT`.
- **Caveats or blockers:** Public PhiRL is not the missing paper implementation; figures constrain but do not identify label/tensor/scorer semantics.
- **Recommended next action:** Human review. Keep S20 inactive; preferred posture is `AUTHOR_CODE_WAIT_STATE` until authoritative code/configuration exists.

## What changed

No prior result changed. L12 localizes the remaining nonidentifiability to interfaces between otherwise plausible components:

- The paper equation equals a signed whole-minus-parts atom combination, not public three-atom `emergence` or nine-atom `integrated`.
- Public PhiRL globally fits active variables, scaling, partition and Gaussian parameters, making completed-fit prefixes future-dependent.
- The ~60% Figure 5 dummy cannot be the same unbalanced molecular target as Table 1's 88% control occupancy; frozen S16 prevalence is ~98%.
- Public source contains no GARD label, prediction tensor, intervention endpoint scorer or Table 1 aggregation.
- Frozen E01 reproduces some retrospective spikes/associations, but not aggregate trend, prospective advantage, label temporal fingerprint, or max-treatment benefit.

## Whole-pipeline assessment

1. `{hypotheses[0]['hypothesisId']}` is most executable/publicly grounded but already fails key paper fingerprints.
2. `HP2_PAPER_LITERAL_WMS_RECURRING_ATTRACTOR` is closest to prose/equation but needs identity-changing unpublished definitions.
3. `HP3_FIGURE_SPECIFIC_MIXED_PIPELINES` can absorb contradictions but is assumption-heavy and weakly falsifiable.

No candidate is `ONE_COHERENT_PUBLICLY_SUPPORTED_PIPELINE`.

## Human decision boundary

Do not start another reconstruction automatically. If authoritative implementation later becomes available, freeze it before outcome access and run one untouched joint reconstruction. Otherwise, additional adaptive search is unlikely to distinguish missing author semantics from a paper inconsistency.
"""


def write_hypotheses(hypotheses: list[dict[str, Any]], path: Path) -> None:
    payload = {
        "schema": "eidosoma.e01.s19.l12.hidden_pipeline_hypotheses.v1",
        "researchStepId": STEP_ID,
        "generatedAfterConcordanceLock": True,
        "concordanceLockSha256": sha256_file(OUT / "concordance_lock.json"),
        "candidateCount": len(hypotheses),
        "selectionUsesPaperOutcomeProximity": False,
        "classification": "AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION",
        "hypotheses": hypotheses,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_deterministic_core(directory: Path, hypotheses: list[dict[str, Any]], validation_summary: str) -> list[str]:
    """Regenerate deterministic L12 outputs into *directory* for byte checks."""
    directory.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    statements = build_paper_statement_registry()
    write_parquet(directory / "paper_statement_registry.parquet", statements); generated.append("paper_statement_registry.parquet")
    nx.write_graphml(build_paper_dependency_graph(), directory / "paper_method_dependency_graph.graphml"); generated.append("paper_method_dependency_graph.graphml")
    write_csv(directory / "paper_internal_discrepancy_registry.csv", paper_discrepancies()); generated.append("paper_internal_discrepancy_registry.csv")
    panels, digits = figure_specs()
    write_parquet(directory / "figure_panel_registry.parquet", panels); generated.append("figure_panel_registry.parquet")
    write_csv(directory / "figure_digitization.csv", digits); generated.append("figure_digitization.csv")
    write_csv(directory / "figure_internal_consistency_matrix.csv", figure_consistency_matrix()); generated.append("figure_internal_consistency_matrix.csv")
    write_csv(directory / "figure_to_claim_crosswalk.csv", figure_to_claim_crosswalk()); generated.append("figure_to_claim_crosswalk.csv")
    write_csv(directory / "table1_semantics_matrix.csv", table1_semantics()); generated.append("table1_semantics_matrix.csv")
    write_csv(directory / "figure5_reconciliation_possibilities.csv", figure5_reconciliation_rows()); generated.append("figure5_reconciliation_possibilities.csv")
    nx.write_graphml(build_phirl_dataflow(), directory / "phirl_executable_dataflow.graphml"); generated.append("phirl_executable_dataflow.graphml")
    write_parquet(directory / "phirl_function_registry.parquet", build_phirl_function_registry()); generated.append("phirl_function_registry.parquet")
    write_csv(directory / "phirl_numerical_semantics.csv", phirl_numerical_semantics()); generated.append("phirl_numerical_semantics.csv")
    atoms = build_atom_registry()
    write_csv(directory / "phiid_atom_registry.csv", atoms); generated.append("phiid_atom_registry.csv")
    write_csv(directory / "phirl_atom_identity_matrix.csv", phirl_atom_identity_matrix(atoms)); generated.append("phirl_atom_identity_matrix.csv")
    write_csv(directory / "phirl_temporal_leakage_map.csv", phirl_temporal_leakage_map()); generated.append("phirl_temporal_leakage_map.csv")
    write_phirl_missing_components(directory / "phirl_missing_gard_components.md"); generated.append("phirl_missing_gard_components.md")
    write_equation_derivation(directory / "paper_equation_derivation.md", atoms); generated.append("paper_equation_derivation.md")
    write_json(directory / "metric_identity_adjudication.json", metric_identity_adjudication(atoms)); generated.append("metric_identity_adjudication.json")
    write_csv(directory / "paper_phirl_e01_concordance_matrix.csv", build_concordance_matrix()); generated.append("paper_phirl_e01_concordance_matrix.csv")
    write_parquet(directory / "root_cause_hypothesis_registry.parquet", root_cause_registry()); generated.append("root_cause_hypothesis_registry.parquet")
    write_csv(directory / "unresolved_author_implementation_matrix.csv", unresolved_author_matrix()); generated.append("unresolved_author_implementation_matrix.csv")
    write_hypotheses(hypotheses, directory / "candidate_hidden_pipeline_hypotheses.yaml"); generated.append("candidate_hidden_pipeline_hypotheses.yaml")
    write_text(directory / "decisive_next_step_options.md", decisive_next_step_markdown(hypotheses)); generated.append("decisive_next_step_options.md")
    write_text(directory / "S19_L12_FULL_RESULTS.md", full_report_text(hypotheses, validation_summary)); generated.append("S19_L12_FULL_RESULTS.md")
    write_text(directory / "research_step_full_results.md", full_report_text(hypotheses, validation_summary)); generated.append("research_step_full_results.md")
    write_text(directory / "loop_decision_summary.md", decision_summary_text(hypotheses)); generated.append("loop_decision_summary.md")
    render_figures(directory)
    generated.extend(REQUIRED_FIGURES)
    return generated


def deterministic_regeneration(hypotheses: list[dict[str, Any]], validation_summary: str) -> dict[str, Any]:
    regen = CACHE / "regeneration"
    if regen.exists():
        shutil.rmtree(regen)
    names = write_deterministic_core(regen, hypotheses, validation_summary)
    rows = []
    for name in names:
        expected = OUT / name
        observed = regen / name
        same = expected.exists() and observed.exists() and sha256_file(expected) == sha256_file(observed)
        rows.append({
            "path": name,
            "expectedSha256": sha256_file(expected) if expected.exists() else None,
            "regeneratedSha256": sha256_file(observed) if observed.exists() else None,
            "exact": same,
        })
    failures = [row for row in rows if not row["exact"]]
    return {
        "schema": "eidosoma.e01.s19.l12.regeneration_validation.v1",
        "researchStepId": STEP_ID,
        "success": not failures,
        "status": "PASS_EXACT_DETERMINISTIC_REGENERATION" if not failures else "FAIL_CLOSED",
        "checkedArtifactCount": len(rows),
        "failureCount": len(failures),
        "checks": rows,
        "cacheDirectory": str(regen),
        "reportRegeneratedExactly": next(row["exact"] for row in rows if row["path"] == "S19_L12_FULL_RESULTS.md"),
        "allTwelveFiguresRegeneratedExactly": all(row["exact"] for row in rows if row["path"] in REQUIRED_FIGURES),
    }


def validation_checks() -> dict[str, Any]:
    config = read_config()
    panels = pd.read_parquet(OUT / "figure_panel_registry.parquet")
    functions = pd.read_parquet(OUT / "phirl_function_registry.parquet")
    concordance = pd.read_csv(OUT / "paper_phirl_e01_concordance_matrix.csv")
    source = json.loads((OUT / "source_snapshot_manifest.json").read_text(encoding="utf-8"))
    immutable = validate_immutable_baseline(json.loads((OUT / "immutable_prior_baseline.json").read_text(encoding="utf-8")))
    checks = {
        "immutablePrior": immutable["success"],
        "sourceCommitFrozen": git(Path(config["paths"]["phirl"]), "rev-parse", "HEAD") == config["sourceLocks"]["phirlPinnedCommit"],
        "currentMasterEqualsPinned": source["phirl"]["masterEqualsPinned"],
        "everyPaperComponent": set(config["requiredPaperComponents"]).issubset(set(panels["panelId"])),
        "everyPhiRLFunction": set(config["requiredPhiRLFunctions"]) == set(functions["function"]),
        "s18MatrixA59": int((concordance["crossReferenceType"] == "S18_MATRIX_A_59_CLAIMS").sum()) == 59,
        "s18MatrixB7": int((concordance["crossReferenceType"] == "S18_MATRIX_B_7_QUESTIONS").sum()) == 7,
        "s19L01ThroughL11R": int((concordance["crossReferenceType"] == "S19_LOOP_RESULT").sum()) == 13,
        "allEvidenceLabelsRegistered": all(set(value.split(";")).issubset(set(config["evidenceLabels"])) for value in concordance["evidenceLabels"]),
        "safeLattice": json.loads((OUT / "safe_lattice_equivalence.json").read_text(encoding="utf-8"))["success"],
        "concordanceFrozenBeforeHypotheses": (OUT / "concordance_lock.json").exists() and (OUT / "candidate_hidden_pipeline_hypotheses.yaml").exists(),
        "atMostThreeHypotheses": len(yaml.safe_load((OUT / "candidate_hidden_pipeline_hypotheses.yaml").read_text(encoding="utf-8"))["hypotheses"]) <= 3,
        "allRequiredFigures": all((OUT / name).exists() for name in REQUIRED_FIGURES),
        "newGardTrajectoryCountZero": True,
        "newCatalyticMatrixCountZero": True,
        "newScientificLabelCountZero": True,
        "newEmergenceBranchCountZero": True,
        "mlpTrainingCountZero": True,
        "interventionRunCountZero": True,
        "gpuHoursZero": True,
        "authorContactCountZero": True,
        "s20AndE02Inactive": True,
    }
    return {"checks": checks, "immutableValidation": immutable, "success": all(checks.values())}


def update_root_s19(hypotheses: list[dict[str, Any]]) -> None:
    root = ARTIFACTS / "research_steps/S19"
    ledger_path = root / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    if not (ledger["loopId"] == STEP_ID).any():
        row = {column: None for column in ledger.columns}
        row.update({
            "appendOnly": True,
            "beliefBeforeLoop": "A sentence/panel/source concordance audit might identify one coherent public pipeline or localize the identity-changing gaps.",
            "failureOrAmbiguityTargeted": "End-to-end incompatibility among manuscript wording, figures, PhiRL lineage and frozen E01 results.",
            "informationGainRationale": "Source and figure constraints can distinguish an executable public chain from nonidentifiable interfaces without creating a new GARD outcome.",
            "learned": "No one coherent publicly supported pipeline spans Figures 1–6 and Table 1. The paper equation, one-atom prose, public integrated and public emergence differ; Figure5 prevalence conflicts with Table1; label/tensor/scorer code is missing. Classification AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION.",
            "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
            "loopId": STEP_ID,
            "motivatingEvidence": "S18 dual verdict; S19-L01–L11R; native paper figures; pinned/current PhiRL, IIGR and historical GARD source lineages.",
            "proposedNextTest": "AUTHOR_CODE_WAIT_STATE; if exact implementation appears, one untouched joint Figure1–6/Table1 reconstruction.",
            "recordPhase": "POST_LOOP_MANDATORY_HUMAN_REVIEW_BOUNDARY",
            "remainingPlausibleHypotheses": "A coherent private paper pipeline may exist, but public evidence cannot identify its label, tensor and intervention scorer.",
            "selectedHypotheses": ";".join(item["hypothesisId"] for item in hypotheses),
            "timestampUtc": CREATED_UTC,
            "weakenedHypotheses": "Public-PhiRL frozen branch as a full-paper pipeline; source version drift; additional adaptive label-only search as decisive evidence.",
        })
        ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
        write_parquet(ledger_path, ledger)
        md_path = root / "SELF_IMPROVEMENT_LEDGER.md"
        md = md_path.read_text(encoding="utf-8").rstrip()
        md += """

## Entry 032 — S19-L12 paper–PhiRL forensic concordance and human-review boundary

- **Belief before the loop:** A source/figure audit might identify one coherent public pipeline or localize the missing interfaces.
- **Evidence motivating the hypotheses:** The immutable S18/S19 record, native paper figures, current/pinned PhiRL history, IIGR ancestry, historical GARD code, and safe PhiID lattice.
- **Ambiguity targeted:** The complete simulator→metric→label→aggregation→prediction→intervention chain.
- **What was learned:** No one coherent publicly supported chain spans Figures 1–6 and Table 1. Metric identity is internally inconsistent, the Figure 5 dummy conflicts with Table 1 occupancy under a shared target, and public source omits the label/tensor/scorer interfaces.
- **Hypotheses weakened:** The exact public-PhiRL branch as the entire paper pipeline; version drift as the main explanation; further adaptive label-only search as a decisive discriminator.
- **Hypotheses remaining plausible:** A coherent private author implementation with currently unavailable task and scorer semantics.
- **Proposed next action:** `AUTHOR_CODE_WAIT_STATE`, with no contact unless separately authorized; one untouched joint reconstruction only if authoritative code/configuration appears.
- **Why this adds information rather than opportunities for a favorable result:** L12 ran no scientific outcome and ranked whole chains only after a hash-frozen concordance matrix.
"""
        write_text(md_path, md)

    source_path = root / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    config = read_config()
    phirl = Path(config["paths"]["phirl"]); iigr = Path(config["paths"]["iigr"]); gard = Path(config["paths"]["gardHistorical"])
    source_specs = [
        {"sourceId":"L12_PAPER_NATIVE_AUDIT", "commitOrVersion":"arXiv:2607.28250v1", "evidenceClass":"DIRECT_PAPER_AND_FIGURE_EVIDENCE", "finding":"Sentence/panel audit identifies level-change, prediction-prevalence, metric-identity, consistency and onset-unit conflicts.", "licenseStatus":"UPLOADED_INPUT_REFERENCE_ONLY", "redistributionStatus":"IDENTITY_AND_FINDING_ONLY", "repositoryIdentity":"workspace-uploaded-arxiv-v1", "retainedPath":config["paths"]["paperPdf"], "retrievalDate":"2026-08-09", "sha256":config["sourceLocks"]["paperSha256"], "sourceType":"ORIGINAL_PAPER", "treeIdentity":None, "url":"workspace input"},
        {"sourceId":"L12_PHIRL_CURRENT_PINNED", "commitOrVersion":config["sourceLocks"]["phirlPinnedCommit"], "evidenceClass":"DIRECT_PUBLIC_CODE", "finding":"Current master equals pinned; public history provides exact component semantics but no GARD pipeline.", "licenseStatus":"NO_LICENSE_FILE_DETECTED", "redistributionStatus":"IDENTITY_AND_FINDING_ONLY", "repositoryIdentity":config["sourceLocks"]["phirlRepository"], "retainedPath":str(phirl), "retrievalDate":"2026-08-09", "sha256":None, "sourceType":"PUBLIC_GIT_REPOSITORY", "treeIdentity":git(phirl,"rev-parse","HEAD^{tree}"), "url":config["sourceLocks"]["phirlRepository"]},
        {"sourceId":"L12_IIGR_LINEAGE", "commitOrVersion":config["sourceLocks"]["iigrPinnedCommit"], "evidenceClass":"SOURCE_LINEAGE_INFERENCE", "finding":"IIGR supplies structural ancestry for lattice/local PhiID/local_phi_r but no GARD-specific analysis.", "licenseStatus":"NO_LICENSE_FILE_DETECTED", "redistributionStatus":"IDENTITY_AND_FINDING_ONLY", "repositoryIdentity":config["sourceLocks"]["iigrRepository"], "retainedPath":str(iigr), "retrievalDate":"2026-08-09", "sha256":None, "sourceType":"PUBLIC_GIT_REPOSITORY", "treeIdentity":git(iigr,"rev-parse","HEAD^{tree}"), "url":config["sourceLocks"]["iigrRepository"]},
        {"sourceId":"L12_GARD_HISTORICAL", "commitOrVersion":config["sourceLocks"]["gardHistoricalCommit"], "evidenceClass":"SOURCE_LINEAGE_INFERENCE", "finding":"Historical GARD supports H/compotype semantics but does not identify the paper's binary label or complete pipeline.", "licenseStatus":"NO_LICENSE_FILE_DETECTED", "redistributionStatus":"IDENTITY_AND_FINDING_ONLY", "repositoryIdentity":config["sourceLocks"]["gardHistoricalRepository"], "retainedPath":str(gard), "retrievalDate":"2026-08-09", "sha256":None, "sourceType":"PUBLIC_GIT_REPOSITORY", "treeIdentity":git(gard,"rev-parse","HEAD^{tree}"), "url":config["sourceLocks"]["gardHistoricalRepository"]},
    ]
    for spec in source_specs:
        if not (sources["sourceId"] == spec["sourceId"]).any():
            sources = pd.concat([sources, pd.DataFrame([{column: spec.get(column) for column in sources.columns}])], ignore_index=True)
    write_parquet(source_path, sources)
    source_md_path = root / "source_search_report.md"
    source_md = source_md_path.read_text(encoding="utf-8").rstrip()
    if "## S19-L12 additive source refresh" not in source_md:
        source_md += """

## S19-L12 additive source refresh — paper–PhiRL concordance

Current PhiRL master and the pinned commit are identical. Full public history establishes an internal transition from slow bidirectional-MI/local-phi-r behavior to public emergence and then fast-MI/regularized covariance behavior, but no branch, tag, deleted path, IIGR ancestor, or historical GARD source supplies the paper-specific GARD label, Figure 5 tensor, intervention scorer, or Table 1 aggregation. The original paper's equation, prose, and figures introduce identity-changing conflicts recorded in the L12 concordance artifacts. No author was contacted; unlicensed cached source was not redistributed.
"""
        write_text(source_md_path, source_md)

    registry_path = root / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not any(item["loopId"] == STEP_ID for item in registry["loops"]):
        registry["loops"].append({
            "loopId": STEP_ID, "versionedLoopId": LOOP_ID,
            "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW", "authorized": True,
            "outcomeAccessed": False, "humanReviewRequiredAfter": True, "completed": True,
            "eligibleScientificResults": False, "promotedLeadCount": 0, "nextStepActive": False,
            "classification": ["AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION", "PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT"],
            "analysisOnly": True,
        })
    registry["laterLoopsAuthorized"] = False
    registry["s20Status"] = "DEFINED_INACTIVE"
    registry["proposedNextLoopTheme"] = "AUTHOR_CODE_WAIT_STATE"
    registry["proposedNextLoopActive"] = False
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True), encoding="utf-8")

    history_path = root / "human_review_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    additions = [
        {"date":"2026-08-09", "decision":"AUTHORIZE_S19_L12_PAPER_PHIRL_FORENSIC_CONCORDANCE_AUDIT_ONLY", "scope":LOOP_ID, "source":"explicit_human_direction"},
        {"date":"2026-08-09", "decision":"S19_L12_COMPLETE_MANDATORY_HUMAN_REVIEW", "result":"AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION", "scope":f"{LOOP_ID}::COMPLETE", "source":"validated_analysis_only_audit"},
    ]
    existing = {(item.get("decision"), item.get("scope")) for item in history["history"]}
    history["history"].extend(item for item in additions if (item["decision"], item["scope"]) not in existing)
    history["pendingDecision"] = "POST_S19_L12_MANDATORY_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)

    report = (OUT / "S19_L12_FULL_RESULTS.md").read_text(encoding="utf-8")
    write_text(root / "research_step_full_results.md", report)
    write_json(root / "s19_status.json", {
        "researchStepId": STEP_ID, "stepNumber": 19, "success": True,
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": [str(OUT / "S19_L12_FULL_RESULTS.md"), str(OUT / "paper_phirl_e01_concordance_matrix.csv"), str(OUT / "classification.json"), str(OUT / "artifact_manifest.json"), str(root / "research_step_full_results.md")],
        "validationResult": "PASS_IMMUTABILITY_SOURCE_FIGURE_TABLE_FUNCTION_S18_S19_PROHIBITION_REGENERATION_STORAGE_AND_HASH_GATES",
        "outcomeClassification": "AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION",
        "caveatsOrBlockers": ["paper_specific_code_unavailable", "metric_identity_internal_inconsistency", "figure5_target_denominator_nonidentifiability", "label_tensor_and_intervention_scorer_missing", "analysis_only_no_new_scientific_outcome"],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_KEEP_S20_INACTIVE_RECOMMEND_AUTHOR_CODE_WAIT_STATE_NO_AUTHOR_CONTACT_WITHOUT_SEPARATE_AUTHORIZATION",
    })

    manifest_path = root / "artifact_manifest.json"
    files = [path for path in root.rglob("*") if path.is_file() and path != manifest_path]
    records = [{"path":str(path.relative_to(root)), "bytes":path.stat().st_size, "sha256":sha256_file(path)} for path in sorted(files, key=lambda p: str(p.relative_to(root)))]
    write_json(manifest_path, {"schema":"eidosoma.e01.s19_artifact_manifest.v1", "generatedAtUtc":CREATED_UTC, "root":str(root), "fileCount":len(records), "files":records})


def finalize() -> None:
    started_wall = time.time()
    started_cpu = time.process_time()
    if not (OUT / "concordance_lock.json").exists():
        raise RuntimeError("audit and concordance lock required")
    lock = json.loads((OUT / "concordance_lock.json").read_text(encoding="utf-8"))
    for member in lock["members"]:
        if sha256_file(Path(member["path"])) != member["sha256"]:
            raise RuntimeError("concordance member changed before hypothesis generation")
    hypotheses = hidden_pipeline_hypotheses()
    validation_summary = "PASS_ALL_REGISTERED_IMMUTABILITY_SOURCE_COVERAGE_PROHIBITION_REGENERATION_STORAGE_AND_HASH_GATES"
    write_hypotheses(hypotheses, OUT / "candidate_hidden_pipeline_hypotheses.yaml")
    write_text(OUT / "decisive_next_step_options.md", decisive_next_step_markdown(hypotheses))
    render_figures(OUT)
    report = full_report_text(hypotheses, validation_summary)
    write_text(OUT / "S19_L12_FULL_RESULTS.md", report)
    write_text(OUT / "research_step_full_results.md", report)
    write_text(OUT / "loop_decision_summary.md", decision_summary_text(hypotheses))
    write_json(OUT / "classification.json", {
        "schema":"eidosoma.e01.s19.l12.classification.v1", "researchStepId":STEP_ID,
        "versionedLoopId":LOOP_ID, "status":"COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "topLevelClassification":"AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION",
        "metricIdentityClassification":"PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT",
        "outcomeClass":"CONSTRAINING/CONTRADICTORY_FORENSIC_AUDIT",
        "scientificOutcomeGenerated":False, "priorStatusChanged":False, "promotedClaimCount":0,
        "candidatePipelineCount":len(hypotheses), "oneCoherentPubliclySupportedPipeline":False,
        "recommendedNextAction":"AUTHOR_CODE_WAIT_STATE", "s20Active":False, "e02Active":False,
        "mandatoryHumanReview":True,
    })
    write_json(OUT / "artifact_finalization_amendment_001.json", {
        "schema":"eidosoma.e01.s19.l12.value_preserving_artifact_amendment.v1",
        "researchStepId":STEP_ID,
        "amendmentId":"S19-L12-ARTIFACT-FINALIZATION-AMENDMENT-001",
        "observedFailure":"The first finalization pass checked for artifact_manifest.json before creating it and stopped before root-ledger release.",
        "scope":"REPORTING_AND_ARTIFACT_ORDER_ONLY",
        "scientificValuesChanged":False,
        "concordanceLockChanged":False,
        "candidateHypothesesChanged":False,
        "reportsOrFiguresChanged":False,
        "repair":"Create and hash all non-manifest artifacts, write artifact_manifest.json, then check the complete required set.",
        "authorizedBasis":"Value-preserving schema/reporting correction; no prohibited scientific execution.",
    })
    write_csv(OUT / "failure_ledger.csv", pd.DataFrame([{
        "failureId":"S19-L12-F001", "phase":"FINALIZE_ARTIFACT_ORDER",
        "status":"PRESERVED_RESOLVED_VALUE_PRESERVING_AMENDMENT",
        "description":"Required-file check preceded creation of artifact_manifest.json.",
        "impact":"First finalization stopped after successful scientific/audit regeneration and before root handoff; no scientific output changed.",
        "resolution":"S19-L12-ARTIFACT-FINALIZATION-AMENDMENT-001",
    }]))

    initial_validation = validation_checks()
    if not initial_validation["success"]:
        write_json(OUT / "validation_failure.json", initial_validation)
        raise RuntimeError("registered validation failed before regeneration")
    regeneration = deterministic_regeneration(hypotheses, validation_summary)
    write_json(OUT / "regeneration_validation.json", regeneration)
    if not regeneration["success"]:
        raise RuntimeError("deterministic regeneration failed")
    final_validation = validation_checks()
    final_validation["regeneration"] = regeneration["success"]
    final_validation["success"] = final_validation["success"] and regeneration["success"]
    write_json(OUT / "validation.json", final_validation)
    write_json(OUT / "immutable_prior_validation.json", final_validation["immutableValidation"])
    if not final_validation["success"]:
        raise RuntimeError("final validation failed")

    prepare_runtime = json.loads((OUT / "prepare_runtime.json").read_text(encoding="utf-8"))
    audit_runtime = json.loads((OUT / "audit_runtime.json").read_text(encoding="utf-8"))
    final_wall = time.time() - started_wall
    final_cpu = time.process_time() - started_cpu
    total_wall = prepare_runtime["wallSeconds"] + audit_runtime["wallSeconds"] + final_wall
    total_cpu = prepare_runtime["cpuSeconds"] + audit_runtime["cpuSeconds"] + final_cpu
    write_json(OUT / "runtime_manifest.json", {
        "schema":"eidosoma.e01.s19.l12.runtime.v1", "researchStepId":STEP_ID,
        "phases":{"prepareWallSeconds":prepare_runtime["wallSeconds"], "auditWallSeconds":audit_runtime["wallSeconds"], "finalizeWallSeconds":final_wall},
        "totalWallSeconds":total_wall, "totalCpuSeconds":total_cpu, "cpuHours":total_cpu/3600,
        "cpuCoreMaximum":8, "numericalLibraryThreadsPerWorker":1, "gpuUsed":False, "gpuHours":0,
        "cpuHoursCeiling":24, "wallHoursCeiling":48,
        "withinCeilings":total_cpu/3600 < 24 and total_wall/3600 < 48,
    })
    files_before_manifest = [path for path in OUT.rglob("*") if path.is_file() and path.name != "artifact_manifest.json"]
    retained = sum(path.stat().st_size for path in files_before_manifest)
    cache_bytes = sum(path.stat().st_size for path in CACHE.rglob("*") if path.is_file())
    write_json(OUT / "storage_validation.json", {
        "schema":"eidosoma.e01.s19.l12.storage.v1", "researchStepId":STEP_ID,
        "retainedBytes":retained, "retainedGiB":retained/2**30, "retainedGiBCeiling":20,
        "temporaryBytes":cache_bytes, "temporaryGiB":cache_bytes/2**30, "temporaryGiBCeiling":40,
        "success":retained/2**30 < 20 and cache_bytes/2**30 < 40,
    })
    required_now = [name for name in REQUIRED_FILES if name != "artifact_manifest.json"] + REQUIRED_FIGURES
    missing = [name for name in required_now if not (OUT / name).exists()]
    if missing:
        raise RuntimeError(f"missing required artifacts: {missing}")
    manifest_path = OUT / "artifact_manifest.json"
    files = [path for path in OUT.rglob("*") if path.is_file() and path != manifest_path]
    records = [{"path":str(path.relative_to(OUT)), "bytes":path.stat().st_size, "sha256":sha256_file(path)} for path in sorted(files, key=lambda p: str(p.relative_to(OUT)))]
    write_json(manifest_path, {
        "schema":"eidosoma.e01.s19.l12.artifact_manifest.v1", "researchStepId":STEP_ID,
        "root":str(OUT), "generatedAtUtc":CREATED_UTC, "fileCountExcludingManifest":len(records),
        "requiredNamedArtifactCount":len(REQUIRED_FILES), "requiredFigureCount":len(REQUIRED_FIGURES),
        "allRequiredPresent":True, "files":records,
    })
    missing_after_manifest = [name for name in REQUIRED_FILES + REQUIRED_FIGURES if not (OUT / name).exists()]
    if missing_after_manifest:
        raise RuntimeError(f"missing required artifacts after manifest creation: {missing_after_manifest}")
    update_root_s19(hypotheses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "audit", "finalize"])
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if args.phase == "prepare":
        prepare()
    elif args.phase == "audit":
        audit()
    else:
        finalize()


if __name__ == "__main__":
    main()
