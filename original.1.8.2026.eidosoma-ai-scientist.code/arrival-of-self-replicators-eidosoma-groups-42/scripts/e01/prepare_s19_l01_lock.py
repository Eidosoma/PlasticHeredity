#!/usr/bin/env python3
"""Prepare the outcome-blind E01/S19-L01 evidence refresh and method lock.

This script reads only prior frozen evidence, source snapshots, schemas, and
prospective contracts.  It deliberately does not calculate a new scientific
outcome.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import torch
import yaml

from e01_prediction_reconstruction.core import build_split_manifest, derive_seed128 as s16_seed128
from e01_s19_iterative_replication.core import (
    CANDIDATE_IDS,
    FEATURE_IDS,
    PROPORTIONS,
    ROOT_SEED_HEX,
    TEMPORAL_MODES,
    VERSION,
    derive_seed128,
    rank_candidate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L01"
CACHE_ROOT = Path("/cache/e01_s19_l01")
SOURCE_ROOT = CACHE_ROOT / "source_search"
PREREG_PATH = REPO_ROOT / "configs/e01/s19_l01_preregistration.yaml"
METHOD_LOCK_PATH = REPO_ROOT / "configs/e01/s19_l01_method_lock.json"
RANKING_PATH = REPO_ROOT / "configs/e01/s19_l01_candidate_ranking.csv"
SPLIT_PATH = REPO_ROOT / "configs/e01/s16_split_manifest.csv"
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
S18_MATRIX = Path("/artifacts/research_steps/S18/matrix_a_59_claims.csv")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_roots() -> dict[str, Any]:
    roots: list[tuple[Path, str]] = []
    research = Path("/artifacts/research_steps")
    for step in sorted(item for item in research.iterdir() if item.is_dir()):
        name = step.name
        digits = ""
        for character in name[1:]:
            if character.isdigit():
                digits += character
            else:
                break
        if name.startswith("S") and digits and int(digits) <= 18:
            roots.append((step, f"IMMUTABLE_{name}"))
    roots.extend(
        [
            (Path("/artifacts/E01_forensic_replication_bundle"), "IMMUTABLE_V1_BUNDLE"),
            (Path("/artifacts/E01_forensic_replication_artifact_v2"), "IMMUTABLE_V2_BUNDLE"),
            (Path("/cache/e01_s13y_v1/raw_trajectories"), "FROZEN_S13Y_TRAJECTORIES"),
        ]
    )
    file_rows: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root, role in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        root_count = 0
        root_bytes = 0
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            identity = str(path.resolve())
            if identity in seen:
                continue
            seen.add(identity)
            size = path.stat().st_size
            file_rows.append(
                {
                    "path": identity,
                    "role": role,
                    "bytes": size,
                    "sha256": sha256_file(path),
                }
            )
            root_count += 1
            root_bytes += size
        root_rows.append(
            {"path": str(root), "role": role, "fileCount": root_count, "totalBytes": root_bytes}
        )
    for path, role in [
        (WORKSPACE_ROOT / "FULL_PLAN.md", "GOVERNING_FULL_PLAN"),
        (WORKSPACE_ROOT / "AGENTS.md", "GOVERNING_POLICY"),
        (WORKSPACE_ROOT / "input-attachments/MANIFEST.json", "UPLOADED_INPUT_MANIFEST"),
        (
            WORKSPACE_ROOT
            / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/_metadata/ATTACHMENT.md",
            "UPLOADED_INPUT_SIDECAR",
        ),
        (Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf"), "ORIGINAL_PAPER_V1"),
    ]:
        size = path.stat().st_size
        file_rows.append({"path": str(path), "role": role, "bytes": size, "sha256": sha256_file(path)})
    aggregate = hashlib.sha256()
    for row in sorted(file_rows, key=lambda item: item["path"]):
        aggregate.update(canonical_json(row).encode())
        aggregate.update(b"\n")
    return {
        "schema": "eidosoma.e01.s19_s18_immutable_baseline.v1",
        "researchStepId": "S19",
        "loopId": "S19-L01",
        "historicalBoundary": "S01-S18_PLUS_V1_V2_AND_FROZEN_S13Y_TRAJECTORIES",
        "fileCount": len(file_rows),
        "totalBytes": int(sum(row["bytes"] for row in file_rows)),
        "aggregateSha256": aggregate.hexdigest(),
        "roots": root_rows,
        "files": sorted(file_rows, key=lambda item: item["path"]),
    }


def git_value(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def source_rows(retrieved: str) -> pd.DataFrame:
    rows = [
        {
            "sourceId": "PAPER_ARXIV_2607_28250V1",
            "sourceType": "PRIMARY_PAPER",
            "url": "https://arxiv.org/abs/2607.28250v1",
            "repositoryIdentity": None,
            "commitOrVersion": "arXiv:2607.28250v1",
            "treeIdentity": None,
            "retrievalDate": retrieved,
            "retainedPath": "/cache/e01_s03/downloads/paper-2607.28250v1.pdf",
            "sha256": "77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4",
            "licenseStatus": "CC-BY-4.0",
            "evidenceClass": "DIRECT_PAPER_EVIDENCE",
            "finding": "Defines named claims and primary 3-SD spike rule but omits metric objects, alternate proportions, and spike descriptor reductions.",
            "redistributionStatus": "CITABLE_INPUT",
        },
        {
            "sourceId": "BREAKING_GRN_MEMORIES",
            "sourceType": "SAME_AUTHOR_PUBLIC_CODE_LINEAGE",
            "url": "https://github.com/pigozzif/BreakingGRNMemories",
            "repositoryIdentity": "pigozzif/BreakingGRNMemories",
            "commitOrVersion": "afe44231ad3ce915172cdb53a6b234bd76fcb6a5",
            "treeIdentity": "56f66ab8b57a2c60e830370842926708eee0767d",
            "retrievalDate": retrieved,
            "retainedPath": str(SOURCE_ROOT / "BreakingGRNMemories"),
            "sha256": None,
            "licenseStatus": "NO_LICENSE_FILE_FOUND",
            "evidenceClass": "DIRECT_PUBLIC_CODE_LINEAGE_NOT_AUTHOR_CODE_FOR_TARGET_PAPER",
            "finding": "Defines the exact seven graph and five dynamical metric families; nolds is pinned to 0.6.1.",
            "redistributionStatus": "CACHE_ONLY_NO_SOURCE_REDISTRIBUTION",
        },
        {
            "sourceId": "IIGR",
            "sourceType": "SAME_AUTHOR_PUBLIC_CODE_LINEAGE",
            "url": "https://github.com/pigozzif/IntegratedInformationGeneRegulation",
            "repositoryIdentity": "pigozzif/IntegratedInformationGeneRegulation",
            "commitOrVersion": "7c1c22fe39f539d4a453135476f1f0dd5a6b45f7",
            "treeIdentity": "b0baf451876f4c8760f25096b7d426add68f6881",
            "retrievalDate": retrieved,
            "retainedPath": str(SOURCE_ROOT / "IntegratedInformationGeneRegulation"),
            "sha256": None,
            "licenseStatus": "NO_LICENSE_FILE_FOUND",
            "evidenceClass": "DIRECT_PUBLIC_CODE_LINEAGE_NOT_AUTHOR_CODE_FOR_TARGET_PAPER",
            "finding": "Independently contains matching graph/dynamical functions and all-pairs peak-distance plus mean peak-height feature reductions.",
            "redistributionStatus": "CACHE_ONLY_NO_SOURCE_REDISTRIBUTION",
        },
        {
            "sourceId": "PHIRL",
            "sourceType": "PINNED_INFORMATION_SOURCE_LINEAGE",
            "url": "https://github.com/pigozzif/PhiRL",
            "repositoryIdentity": "pigozzif/PhiRL",
            "commitOrVersion": "a6d1d0d18c7551302724b7158c6ccdc4d3a33373",
            "treeIdentity": "e59fa8e311c2f727724acf3c1f1885dc8d840ee5",
            "retrievalDate": retrieved,
            "retainedPath": str(SOURCE_ROOT / "PhiRL"),
            "sha256": None,
            "licenseStatus": "NO_LICENSE_FILE_FOUND",
            "evidenceClass": "DIRECT_PUBLIC_CODE_LINEAGE_NOT_AUTHOR_CODE_FOR_TARGET_PAPER",
            "finding": "Confirms feature-name vocabulary and the already-frozen PhiRL information lineage; supplies no GARD prediction script.",
            "redistributionStatus": "CACHE_ONLY_NO_SOURCE_REDISTRIBUTION",
        },
        {
            "sourceId": "NOLDS_0_6_1",
            "sourceType": "ARCHIVED_PACKAGE_VERSION",
            "url": "https://pypi.org/project/nolds/0.6.1/",
            "repositoryIdentity": "PyPI:nolds",
            "commitOrVersion": "0.6.1",
            "treeIdentity": None,
            "retrievalDate": retrieved,
            "retainedPath": "/cache/e01_s19_l01/packages/nolds-0.6.1-py2.py3-none-any.whl",
            "sha256": "208714600333f03e428c968a0cea0e8029d75ac30b454e28e968630b14973829",
            "licenseStatus": "MIT",
            "evidenceClass": "DIRECT_DEPENDENCY_EVIDENCE",
            "finding": "Pinned implementation for sample entropy, correlation dimension, Lyapunov exponent, DFA, and generalized Hurst exponent.",
            "redistributionStatus": "CACHE_ONLY_DEPENDENCY_IDENTITY_RECORDED",
        },
        {
            "sourceId": "PIGOZZIF_PUBLIC_REPO_LIST_20260808",
            "sourceType": "PUBLIC_METADATA_SEARCH",
            "url": "https://api.github.com/users/pigozzif/repos?per_page=100",
            "repositoryIdentity": "GitHub user pigozzif public repository listing",
            "commitOrVersion": "retrieved_2026-08-08",
            "treeIdentity": None,
            "retrievalDate": retrieved,
            "retainedPath": str(SOURCE_ROOT / "api/pigozzif_repos.json"),
            "sha256": sha256_file(SOURCE_ROOT / "api/pigozzif_repos.json"),
            "licenseStatus": "METADATA_ONLY",
            "evidenceClass": "DIRECT_SEARCH_RECORD_WITH_LIMITED_ABSENCE_INFERENCE",
            "finding": "Visible list contains PhiRL, IIGR, and BreakingGRNMemories but no repository named for this GARD paper; absence is not proof that code does not exist elsewhere.",
            "redistributionStatus": "HASH_AND_URL_ONLY",
        },
    ]
    file_specs = [
        ("BREAKING_NETWORK", "BreakingGRNMemories/network.py", "b3c80e89c5bf48250794f527ecffa78eff9d3b54c528d916c528a6a42c1ceab2", "Exact graph feature list and NetworkX calls."),
        ("BREAKING_DYNAMICAL", "BreakingGRNMemories/dynamical.py", "d4b54674bfc4ee584a52db4b75f66c94b57481b20d6acc8d71efab83fe7193e3", "Exact dynamical feature list, preprocessing, and nolds calls."),
        ("IIGR_NETWORK", "IntegratedInformationGeneRegulation/network.py", "19a5796dd606eb6aa8199cdcdd69e1191285ae970a8dc231b5ba4f3c84b795f7", "Independent same-lineage graph implementation."),
        ("IIGR_DYNAMICAL", "IntegratedInformationGeneRegulation/dynamical.py", "b1476ae612b520b8ee845d67c7d99ca8921b17bf6c4142cfd32e146b814378d1", "Independent same-lineage dynamical implementation."),
        ("IIGR_FEATURES", "IntegratedInformationGeneRegulation/features.py", "46a3a52354582ee1a47bedd1078903c376131e41c6e8ee28a29b65c24475ca86", "All-unordered-pairs peak spacing and standardized peak-height reductions."),
    ]
    for source_id, relative, digest, finding in file_specs:
        repo = relative.split("/", 1)[0]
        rows.append(
            {
                "sourceId": source_id,
                "sourceType": "PINNED_SOURCE_FILE",
                "url": f"https://github.com/pigozzif/{repo}/blob/master/{relative.split('/',1)[1]}",
                "repositoryIdentity": f"pigozzif/{repo}",
                "commitOrVersion": "afe44231ad3ce915172cdb53a6b234bd76fcb6a5" if repo == "BreakingGRNMemories" else "7c1c22fe39f539d4a453135476f1f0dd5a6b45f7",
                "treeIdentity": None,
                "retrievalDate": retrieved,
                "retainedPath": str(SOURCE_ROOT / relative),
                "sha256": digest,
                "licenseStatus": "NO_LICENSE_FILE_FOUND",
                "evidenceClass": "DIRECT_PUBLIC_CODE_LINEAGE_NOT_TARGET_AUTHOR_CODE",
                "finding": finding,
                "redistributionStatus": "CACHE_ONLY_NO_SOURCE_REDISTRIBUTION",
            }
        )
    return pd.DataFrame(rows)


def build_candidate_registry() -> pd.DataFrame:
    frame = pd.read_csv(RANKING_PATH)
    score_columns = [
        "sourceGrounding",
        "paperFingerprintSpecificity",
        "explanatoryLeverage",
        "testability",
        "crossCandidateDiscriminability",
        "computeEfficiency",
        "independenceFromPriorOutcomeSelection",
    ]
    penalty_columns = [
        "outcomeGuidedThresholdSelection",
        "deterministicHReuse",
        "completedFitLeakage",
        "candidateSpecificSuccess",
        "undefinedAuthorSemantics",
        "branchCount",
    ]
    frame["rankingScore"] = [
        rank_candidate(
            {name: float(row[name]) for name in score_columns},
            {name: float(row[name]) for name in penalty_columns},
        )
        for _, row in frame.iterrows()
    ]
    frame["frozenRank"] = frame["rankingScore"].rank(method="first", ascending=False).astype(int)
    frame["registryOrder"] = np.arange(1, len(frame) + 1)
    return frame


def seed_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split = build_split_manifest()
    for row in split.itertuples(index=False):
        rows.append(
            {
                "researchStepId": "S19",
                "loopId": "S19-L01",
                "bundleId": "B_ALTERNATIVE_PREDICTION_PROPORTIONS",
                "streamDomain": "S16_FROZEN_SPLIT",
                "streamId": f"S16::split::R{row.repetitionId:02d}::M{row.matrixIndex:03d}",
                "candidateId": None,
                "matrixIndex": int(row.matrixIndex),
                "repetitionId": int(row.repetitionId),
                "proportion": None,
                "purpose": row.splitRole,
                "derivedSeed": row.testSeed128,
                "rootHex": "9a8456c3204eea08a83a7a04d64b4097f7d922fe9c21b8deea0839127f66c2b1",
                "generator": "PCG64DXSM",
                "reusedFrozenS16": True,
            }
        )
    for candidate in CANDIDATE_IDS:
        for proportion in PROPORTIONS:
            for repetition in range(10):
                rows.append(
                    {
                        "researchStepId": "S19",
                        "loopId": "S19-L01",
                        "bundleId": "B_ALTERNATIVE_PREDICTION_PROPORTIONS",
                        "streamDomain": "S16_FROZEN_MODEL",
                        "streamId": f"S16::model::{candidate}::R{repetition:02d}",
                        "candidateId": candidate,
                        "matrixIndex": None,
                        "repetitionId": repetition,
                        "proportion": proportion,
                        "purpose": "model_initialization_identical_across_proportions_features_modes",
                        "derivedSeed": None,
                        "rootHex": "9a8456c3204eea08a83a7a04d64b4097f7d922fe9c21b8deea0839127f66c2b1",
                        "generator": "torch_manual_seed",
                        "reusedFrozenS16": True,
                    }
                )
    for bundle in ("A_METRIC_DISTINCTIVENESS", "C_SPIKE_TIMING_SPACING_HEIGHT"):
        for candidate in CANDIDATE_IDS:
            for purpose in ("bootstrap", "permutation", "replay"):
                rows.append(
                    {
                        "researchStepId": "S19",
                        "loopId": "S19-L01",
                        "bundleId": bundle,
                        "streamDomain": "S19_ANALYSIS",
                        "streamId": f"S19-L01::{bundle}::{candidate}::{purpose}",
                        "candidateId": candidate,
                        "matrixIndex": None,
                        "repetitionId": None,
                        "proportion": None,
                        "purpose": purpose,
                        "derivedSeed": str(derive_seed128(bundle, candidate, purpose)),
                        "rootHex": ROOT_SEED_HEX,
                        "generator": "PCG64DXSM",
                        "reusedFrozenS16": False,
                    }
                )
    return pd.DataFrame(rows)


def markdown_top(title: str, artifacts: str, next_action: str) -> str:
    return f"""# {title}

## Concise top summary

- **Research step ID:** S19-L01
- **Completion status:** PRE-OUTCOME METHOD LOCK PREPARED; scientific execution not yet accessed
- **Artifacts written:** {artifacts}
- **Validation result:** Immutable baseline, source identities, candidate ordering, seeds, and preregistration serialized; clean pushed-repository gate remains required before outcomes
- **Outcome classification:** NOT YET CLASSIFIED
- **Caveats or blockers:** No target-paper code or exact alternate proportion list was recovered; public lineages are source evidence, not author-code identity
- **Recommended next action:** {next_action}
"""


def main() -> None:
    started = datetime.now(timezone.utc)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    prereg = yaml.safe_load(PREREG_PATH.read_text(encoding="utf-8"))
    method_lock = json.loads(METHOD_LOCK_PATH.read_text(encoding="utf-8"))
    if prereg["outcomeAccessedAtLock"] is not False or method_lock["outcomeAccessedAtLock"] is not False:
        raise RuntimeError("pre-outcome lock is not outcome blind")
    baseline = hash_roots()
    write_json(ARTIFACT_ROOT / "s18_immutable_baseline.json", baseline)
    retrieved = started.date().isoformat()
    sources = source_rows(retrieved)
    sources.to_parquet(ARTIFACT_ROOT / "source_search_ledger.parquet", index=False)
    candidates = build_candidate_registry()
    candidates.to_parquet(ARTIFACT_ROOT / "candidate_registry.parquet", index=False)
    candidates.to_csv(LOOP_ROOT / "candidate_ranking.csv", index=False)
    seeds = seed_manifest()
    seeds.to_parquet(LOOP_ROOT / "seed_manifest.parquet", index=False)
    shutil.copy2(PREREG_PATH, LOOP_ROOT / "preregistration.yaml")
    shutil.copy2(METHOD_LOCK_PATH, LOOP_ROOT / "method_lock.json")
    bundle_registry = {
        "schema": "eidosoma.e01.s19_l01_candidate_bundle_registry.v1",
        "loopId": "S19-L01",
        "outcomeAccessedAtLock": False,
        "bundleCount": 3,
        "bundles": prereg["bundles"],
        "selectedCandidateIds": candidates.loc[candidates["selected"], "candidateId"].tolist(),
        "maximumSpecificationsPerBundle": 8,
    }
    (LOOP_ROOT / "candidate_bundle_registry.yaml").write_text(
        yaml.safe_dump(bundle_registry, sort_keys=False), encoding="utf-8"
    )
    source_snapshot = {
        "schema": "eidosoma.e01.s19_l01_source_snapshot_manifest.v1",
        "loopId": "S19-L01",
        "retrievalDate": retrieved,
        "sourceCount": len(sources),
        "sources": sources.to_dict(orient="records"),
        "rawUnlicensedSourceCollectedAsArtifact": False,
        "noldsWheelSha256": "208714600333f03e428c968a0cea0e8029d75ac30b454e28e968630b14973829",
    }
    write_json(LOOP_ROOT / "source_snapshot_manifest.json", source_snapshot)
    input_paths = [
        S13Y_ROOT / "full_source_values.parquet",
        S13Y_ROOT / "prefix_endpoint_values.parquet",
        S13Y_ROOT / "label_values.parquet",
        S13Y_ROOT / "trajectory_manifest.parquet",
        S13Y_ROOT / "simulation_summary.parquet",
        Path("/artifacts/research_steps/S16/split_metrics.csv"),
        SPLIT_PATH,
        S18_MATRIX,
        Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf"),
    ]
    input_manifest = {
        "schema": "eidosoma.e01.s19_l01_input_manifest.v1",
        "loopId": "S19-L01",
        "newGardTrajectories": 0,
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in input_paths
        ],
        "rawTrajectoryManifestPath": str(S13Y_ROOT / "trajectory_manifest.parquet"),
        "rawTrajectoryHashesCheckedDuringExecution": True,
    }
    write_json(LOOP_ROOT / "input_manifest.json", input_manifest)
    decision = markdown_top(
        "E01 V3 continuation decision",
        "continuation decision, S18 immutable baseline, source/candidate/loop registries, and L01 pre-outcome contracts",
        "verify tests, benchmark, commit and push the complete lock, then execute only S19-L01",
    ) + """

## Authorization

The explicit post-S18 human override reopens E01 only as the additive continuation `E01-FORENSIC-REPLICATION-CONTINUATION-v3.0.0`. It authorizes `E01-S19-L01-UNEVALUATED-CLAIM-RECOVERY-v1.0.0` and no later loop. S18 remains an immutable historical snapshot; its 3/17/21/2/16 claim totals and Matrix A/Matrix B are not rewritten.

## Operational meaning of a loop

A loop is one bounded research decision cycle, not an instruction to spend the maximum budget. Loop 1 targets a few wall-clock hours by reusing frozen data, using eight bounded workers where safe, and avoiding trajectory generation. The 100-CPU-hour/72-wall-hour values remain hard ceilings, not runtime targets. Every loop ends at human review.

## Stop boundary

After the L01 reports and hashes are written, permitted human decisions are `CONTINUE_S19`, `ACTIVATE_S20_CONFIRMATION`, `ACTIVATE_S20_CLOSEOUT_ONLY`, or `PAUSE_PROGRAM`. No option is selected automatically.
"""
    (ARTIFACT_ROOT / "continuation_decision.md").write_text(decision, encoding="utf-8")
    source_report = markdown_top(
        "S19-L01 source search report",
        "source_search_ledger.parquet and source_snapshot_manifest.json",
        "retain the source-grounded graph/dynamics/spacing definitions in the frozen lock and treat unresolved GARD graph/proportion semantics explicitly",
    ) + f"""

## Search scope and method

The refresh covered the original v1 paper, its named public information lineages, public repository metadata, branches/tags/forks, commit histories, same-author metric code, historical E01 source records, and the pinned `nolds==0.6.1` dependency. URLs, repository/tree identities, retrieval date `{retrieved}`, retained-cache hashes, and licensing status are in the machine ledger. No author was contacted.

## Direct findings

1. `BreakingGRNMemories` and `IntegratedInformationGeneRegulation` independently implement the paper's exact seven network families: node and edge counts, in/out degree, betweenness, PageRank, and HITS. Both summarize each vector-valued metric by its mean and standard deviation.
2. The same lineages implement all five dynamical families with `nolds`: multivariate sample entropy, mean/std per-variable correlation dimension, mean/std (and in the newer lineage maximum) per-variable largest Lyapunov exponent, multivariate DFA, and the first generalized-Hurst result.
3. The IIGR feature lineage defines inter-peak spacing as the mean of **all unordered pairwise distances**, not merely adjacent distances, and uses mean standardized peak heights. Its local-window peak detector conflicts with the paper's declared global three-standard-deviation rule, so that detector is not substituted for the directed primary rule.
4. The source graph builder constructs an unweighted reaction topology. Applying that literally to a positive lognormal GARD catalytic matrix creates a complete directed graph. Therefore several mean graph metrics may be constant; undefined correlations are a falsification result and cannot count as evidence of distinctive PhiRL information.
5. Neither the paper, visible target-author public repositories, nor the inspected histories supply this paper's GARD implementation, a prediction tensor, or the exact alternative input/output proportions. The directed minimal 10/90, 20/80, 25/75, 33/67, 50/50 family is therefore retained without claiming author identity.

## Inference boundary

The same-author lineages are strong source grounding for metric formulas and descriptive fingerprints. They are not the unavailable target-paper code. The absence of a named target repository in the visible public list is limited negative search evidence, not proof that no code exists.

## License handling

No LICENSE/COPYING file was present at the pinned PhiRL, IIGR, or BreakingGRNMemories commits. Their raw source remains cache-only and is not redistributed in artifacts. The `nolds` wheel reports MIT licensing; only its identity and hash are retained here.
"""
    (ARTIFACT_ROOT / "source_search_report.md").write_text(source_report, encoding="utf-8")
    ledger_rows = pd.DataFrame(
        [
            {
                "ledgerSequence": 1,
                "timestampUtc": started.isoformat(),
                "loopId": "S19-L01",
                "recordPhase": "PRE_LOOP_BELIEF_AND_SELECTION",
                "beliefBeforeLoop": "S18 left C001-C012, C029, and C031-C033 unevaluated; metric objects, alternate proportions, and spike reductions were underdocumented.",
                "motivatingEvidence": "Same-author public lineages expose exact named metric functions and all-pairs spike spacing; frozen S16 and S14 supply bounded baselines.",
                "failureOrAmbiguityTargeted": "Recover the 16 NOT_EVALUATED claims without altering S18 or using outcome-guided method search.",
                "selectedHypotheses": "A: direct source-lineage metrics; B: five locked proportions; C: global 3SD/all-pairs descriptors plus one within-run normalized companion.",
                "learned": None,
                "weakenedHypotheses": None,
                "remainingPlausibleHypotheses": None,
                "proposedNextTest": "Pending bounded execution and falsification.",
                "informationGainRationale": "The loop resolves previously untested claim families with specific public fingerprints and exact frozen-data controls rather than adding arbitrary branches.",
                "appendOnly": True,
            }
        ]
    )
    ledger_rows.to_parquet(ARTIFACT_ROOT / "self_improvement_ledger.parquet", index=False)
    ledger_md = markdown_top(
        "S19 append-only self-improvement ledger",
        "SELF_IMPROVEMENT_LEDGER.md and self_improvement_ledger.parquet",
        "append the post-execution learning record once, freeze hashes, and stop for human review",
    ) + """

## Entry 001 — S19-L01 pre-loop belief and selection

- **Belief before the loop:** S18 left C001–C012, C029, and C031–C033 unevaluated because the paper omits metric objects, alternate proportions, and spike reductions.
- **Evidence motivating the hypotheses:** Same-author public lineages define all named metric functions and an all-pairs spike-spacing reduction; S14 and S16 provide frozen completed/prefix and prediction baselines.
- **Previous failure or ambiguity addressed:** Complete the sixteen unevaluated claims without rewriting S18 or selecting a method from outcomes.
- **Selected hypotheses:** direct source-lineage graph/dynamical metrics; five prospectively locked prediction proportions; paper-global three-SD spikes with public all-pairs spacing and one inherited within-run normalized companion.
- **Expected information gain:** each bundle has a falsifiable fingerprint, preserves both simulator candidates, and contains no broad grid.
- **What was learned / weakened / remains plausible:** pending execution.
- **Next test:** pending execution; no later loop is authorized.
"""
    (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").write_text(ledger_md, encoding="utf-8")
    loop_registry = {
        "schema": "eidosoma.e01.s19_loop_registry.v1",
        "continuationId": "E01-FORENSIC-REPLICATION-CONTINUATION-v3.0.0",
        "loops": [
            {
                "loopId": "S19-L01",
                "versionedLoopId": VERSION,
                "status": "PRE_OUTCOME_LOCK_PREPARED",
                "authorized": True,
                "outcomeAccessed": False,
                "humanReviewRequiredAfter": True,
            }
        ],
        "laterLoopsAuthorized": False,
        "s20Status": "DEFINED_INACTIVE",
    }
    (ARTIFACT_ROOT / "loop_registry.yaml").write_text(yaml.safe_dump(loop_registry, sort_keys=False), encoding="utf-8")
    write_json(
        ARTIFACT_ROOT / "human_review_history.json",
        {
            "schema": "eidosoma.e01.s19_human_review_history.v1",
            "history": [
                {
                    "date": retrieved,
                    "decision": "REOPEN_E01_AND_AUTHORIZE_S19_L01_ONLY",
                    "scope": VERSION,
                    "source": "explicit_human_direction",
                },
                {
                    "date": retrieved,
                    "decision": "LOOP_RUNTIME_INTERPRETATION",
                    "scope": "few_hours_preferred_longer_only_if_needed_within_hard_ceiling",
                    "source": "explicit_human_clarification",
                },
            ],
            "pendingDecision": "POST_S19_L01_HUMAN_REVIEW",
        },
    )
    write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "researchStepId": "S19-L01",
            "stepNumber": 19,
            "success": False,
            "status": "PRE_OUTCOME_METHOD_LOCK_PREPARED",
            "artifactsWritten": [
                str(ARTIFACT_ROOT / "s18_immutable_baseline.json"),
                str(ARTIFACT_ROOT / "source_search_ledger.parquet"),
                str(ARTIFACT_ROOT / "candidate_registry.parquet"),
                str(LOOP_ROOT / "preregistration.yaml"),
                str(LOOP_ROOT / "method_lock.json"),
            ],
            "validationResult": "PENDING_CLEAN_PUSHED_LOCK_AND_EXECUTION",
            "caveatsOrBlockers": ["target_author_code_not_recovered", "alternate_proportion_list_not_recovered"],
            "recommendedNextAction": "test_benchmark_commit_push_then_execute_only_S19_L01",
        },
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l01_preparation_runtime.v1",
        "startedUtc": started.isoformat(),
        "completedUtc": datetime.now(timezone.utc).isoformat(),
        "stage": "PREPARE_NO_OUTCOME_ACCESS",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "threadEnvironment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "repositoryHeadDuringPreparation": git_value("rev-parse", "HEAD"),
        "outcomeAccessed": False,
    }
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime)
    print(canonical_json({"success": True, "baselineFiles": baseline["fileCount"], "sources": len(sources), "candidates": len(candidates), "seedRows": len(seeds)}))


if __name__ == "__main__":
    main()
